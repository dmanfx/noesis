"""Runtime-neutral loading, evaluation, and floorplan composition for scene priors."""

from __future__ import annotations

import base64
import io
import json
import math
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pydantic import ValidationError

from noesis_core.coordinate_frames import (
    BACKEND_WORLD_FRAME_ID,
    CAMERA_LOCAL_RASTER_ORIENTATION,
    CameraGroundFrame,
    CoordinateFrameError,
    RevisionedFrameTransform,
    camera_ground_frame_from_extrinsics_col_major,
    revisioned_frame_sha256,
    transform_positions,
)
from noesis_core.contracts.scene_prior import (
    MAX_SCENE_PRIOR_GRID_CELLS,
    MAX_SCENE_PRIOR_GRID_DIMENSION,
    ScenePriorArtifact,
    ScenePriorCameraBinding,
    ScenePriorCatalog,
    ScenePriorFrameBinding,
    ScenePriorRevision,
)
from noesis_core.scene_files import (
    SceneFileError,
    absolute_path_without_resolving,
    load_strict_json,
    read_scene_file,
    read_scene_root_file,
)


MAX_SCENE_PRIOR_CATALOG_BYTES = 4 * 1024 * 1024
MAX_SCENE_PRIOR_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_SCENE_PRIOR_GRID_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_SCENE_PRIOR_GRID_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_SCENE_PRIOR_POINTS_GLB_BYTES = 128 * 1024 * 1024
_GLB_JSON_CHUNK = 0x4E4F534A
_GLB_BINARY_CHUNK = 0x004E4942
_DIAGNOSTIC_SURFACE_HEIGHT_MAX_M = 1.80
_DIAGNOSTIC_FURNITURE_MIN_M = 0.12
_DIAGNOSTIC_FURNITURE_MAX_M = 1.65
_DIAGNOSTIC_HEIGHT_BIN_M = 0.05
_FULL_EVIDENCE_DERIVATION = "noesis_scene_prior_2_5d_full_evidence_v2"
_GRID_ARRAYS = (
    "authored_walkable",
    "observed",
    "evidence_confidence",
    "floor_supported",
    "obstacle_mask",
    "walkable_candidate",
    "floor_height_m",
    "height_agl_p95_m",
    "boundary_signed_distance_m",
    "obstacle_signed_clearance_m",
    "point_count",
    "floor_support_count",
    "obstacle_support_count",
)


class ScenePriorError(RuntimeError):
    """Raised when configured scene-prior evidence is invalid or inconsistent."""


def _load_model(data: bytes, model: type[Any], *, label: str) -> Any:
    try:
        payload = load_strict_json(data, label=label)
        if not isinstance(payload, Mapping):
            raise ScenePriorError(f"{label} must contain a JSON object")
        return model.model_validate(payload)
    except (SceneFileError, ValidationError, TypeError, ValueError) as exc:
        if isinstance(exc, ScenePriorError):
            raise
        raise ScenePriorError(f"{label} is invalid: {exc}") from exc


def _read_grid_npz(data: bytes, *, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(data), mode="r") as archive:
            infos = archive.infolist()
            if not infos or len(infos) > 64:
                raise ScenePriorError("scene-prior grid archive inventory is invalid")
            total = sum(int(info.file_size) for info in infos)
            if total > MAX_SCENE_PRIOR_GRID_UNCOMPRESSED_BYTES:
                raise ScenePriorError("scene-prior grid archive expands beyond its bound")
            if any(
                info.is_dir()
                or not info.filename.endswith(".npy")
                or "/" in info.filename
                or "\\" in info.filename
                for info in infos
            ):
                raise ScenePriorError("scene-prior grid archive contains an invalid entry")
        arrays: dict[str, np.ndarray] = {}
        with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
            if set(loaded.files) != set(_GRID_ARRAYS):
                raise ScenePriorError("scene-prior grid archive has an unexpected array inventory")
            for name in _GRID_ARRAYS:
                value = np.asarray(loaded[name])
                if value.shape != shape:
                    raise ScenePriorError(
                        f"scene-prior grid array {name} has shape {value.shape}; expected {shape}"
                    )
                if value.dtype.kind not in "buif":
                    raise ScenePriorError(f"scene-prior grid array {name} has an invalid dtype")
                arrays[name] = np.array(value, copy=True)
    except ScenePriorError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ScenePriorError("scene-prior grid archive cannot be decoded") from exc

    finite_required = (
        "evidence_confidence",
        "floor_height_m",
        "boundary_signed_distance_m",
        "obstacle_signed_clearance_m",
    )
    for name in finite_required:
        if not np.all(np.isfinite(arrays[name])):
            raise ScenePriorError(f"scene-prior grid array {name} must be finite")
    height = arrays["height_agl_p95_m"]
    if np.any(np.isinf(height)):
        raise ScenePriorError("scene-prior height grid contains infinity")
    confidence = arrays["evidence_confidence"].astype(np.float64, copy=False)
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ScenePriorError("scene-prior evidence confidence is outside [0, 1]")
    for name in ("point_count", "floor_support_count", "obstacle_support_count"):
        if np.any(arrays[name] < 0):
            raise ScenePriorError(f"scene-prior count grid {name} contains negative values")
    return arrays


def _read_points_glb(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode the immutable point-only GLB emitted by the scene-prior builder."""

    if len(data) < 20 or len(data) > MAX_SCENE_PRIOR_POINTS_GLB_BYTES:
        raise ScenePriorError("scene-prior points GLB size is outside its bound")
    try:
        magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    except struct.error as exc:
        raise ScenePriorError("scene-prior points GLB header is invalid") from exc
    if magic != b"glTF" or version != 2 or declared_length != len(data):
        raise ScenePriorError("scene-prior points GLB header is unsupported")

    chunks: dict[int, bytes] = {}
    offset = 12
    while offset < len(data):
        if offset + 8 > len(data):
            raise ScenePriorError("scene-prior points GLB chunk header is truncated")
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_end = offset + int(chunk_length)
        if chunk_length <= 0 or chunk_end > len(data) or chunk_type in chunks:
            raise ScenePriorError("scene-prior points GLB chunk inventory is invalid")
        chunks[chunk_type] = data[offset:chunk_end]
        offset = chunk_end
    if offset != len(data) or set(chunks) != {_GLB_JSON_CHUNK, _GLB_BINARY_CHUNK}:
        raise ScenePriorError("scene-prior points GLB must contain one JSON and one binary chunk")

    try:
        document = json.loads(chunks[_GLB_JSON_CHUNK].decode("utf-8").rstrip(" \t\r\n\x00"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScenePriorError("scene-prior points GLB JSON is invalid") from exc
    asset = document.get("asset") if isinstance(document, Mapping) else None
    if not isinstance(asset, Mapping) or asset.get("version") != "2.0":
        raise ScenePriorError("scene-prior points GLB document is unsupported")
    buffers = document.get("buffers")
    buffer_views = document.get("bufferViews")
    accessors = document.get("accessors")
    meshes = document.get("meshes")
    nodes = document.get("nodes")
    if not all(isinstance(value, list) for value in (buffers, buffer_views, accessors, meshes, nodes)):
        raise ScenePriorError("scene-prior points GLB document inventory is invalid")
    if len(buffers) != 1 or not isinstance(buffers[0], Mapping) or buffers[0].get("uri") is not None:
        raise ScenePriorError("scene-prior points GLB must use one embedded buffer")
    binary = chunks[_GLB_BINARY_CHUNK]
    declared_binary_length = int(buffers[0].get("byteLength") or 0)
    if declared_binary_length <= 0 or declared_binary_length > len(binary):
        raise ScenePriorError("scene-prior points GLB binary length is invalid")

    primitive: Mapping[str, Any] | None = None
    for node in nodes:
        if not isinstance(node, Mapping) or "mesh" not in node:
            continue
        if any(key in node for key in ("matrix", "translation", "rotation", "scale")):
            raise ScenePriorError("scene-prior points GLB node transforms are unsupported")
        mesh_index = int(node["mesh"])
        if mesh_index < 0 or mesh_index >= len(meshes) or not isinstance(meshes[mesh_index], Mapping):
            raise ScenePriorError("scene-prior points GLB mesh reference is invalid")
        primitives = meshes[mesh_index].get("primitives")
        if not isinstance(primitives, list) or len(primitives) != 1 or not isinstance(primitives[0], Mapping):
            raise ScenePriorError("scene-prior points GLB must contain one point primitive")
        primitive = primitives[0]
        break
    if primitive is None or int(primitive.get("mode", 4)) != 0:
        raise ScenePriorError("scene-prior points GLB has no point primitive")
    attributes = primitive.get("attributes")
    if not isinstance(attributes, Mapping) or set(attributes) < {"POSITION", "COLOR_0"}:
        raise ScenePriorError("scene-prior points GLB lacks position or color attributes")

    component_dtypes = {
        5121: np.dtype(np.uint8),
        5123: np.dtype("<u2"),
        5126: np.dtype("<f4"),
    }

    def _accessor(index_value: Any, *, label: str) -> tuple[np.ndarray, Mapping[str, Any]]:
        index = int(index_value)
        if index < 0 or index >= len(accessors) or not isinstance(accessors[index], Mapping):
            raise ScenePriorError(f"scene-prior points GLB {label} accessor is invalid")
        accessor = accessors[index]
        if accessor.get("sparse") is not None or accessor.get("type") != "VEC3":
            raise ScenePriorError(f"scene-prior points GLB {label} accessor is unsupported")
        view_index = int(accessor.get("bufferView", -1))
        if view_index < 0 or view_index >= len(buffer_views) or not isinstance(buffer_views[view_index], Mapping):
            raise ScenePriorError(f"scene-prior points GLB {label} buffer view is invalid")
        view = buffer_views[view_index]
        if int(view.get("buffer", -1)) != 0:
            raise ScenePriorError(f"scene-prior points GLB {label} uses another buffer")
        component_type = int(accessor.get("componentType", 0))
        dtype = component_dtypes.get(component_type)
        count = int(accessor.get("count") or 0)
        if dtype is None or count <= 0 or count > 10_000_000:
            raise ScenePriorError(f"scene-prior points GLB {label} component layout is invalid")
        packed_stride = dtype.itemsize * 3
        stride = int(view.get("byteStride", packed_stride))
        if stride < packed_stride or stride % dtype.itemsize:
            raise ScenePriorError(f"scene-prior points GLB {label} stride is invalid")
        start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        end = start + ((count - 1) * stride) + packed_stride
        view_end = int(view.get("byteOffset", 0)) + int(view.get("byteLength", 0))
        if start < 0 or end > declared_binary_length or end > view_end:
            raise ScenePriorError(f"scene-prior points GLB {label} data is out of bounds")
        try:
            values = np.ndarray(
                shape=(count, 3),
                dtype=dtype,
                buffer=binary,
                offset=start,
                strides=(stride, dtype.itemsize),
            ).copy()
        except (TypeError, ValueError) as exc:
            raise ScenePriorError(f"scene-prior points GLB {label} data cannot be decoded") from exc
        return values, accessor

    positions, position_accessor = _accessor(attributes["POSITION"], label="position")
    colors, color_accessor = _accessor(attributes["COLOR_0"], label="color")
    if int(position_accessor.get("componentType", 0)) != 5126:
        raise ScenePriorError("scene-prior points GLB positions must be float32")
    if colors.shape[0] != positions.shape[0]:
        raise ScenePriorError("scene-prior points GLB position and color counts differ")
    color_component = int(color_accessor.get("componentType", 0))
    if color_component == 5126:
        colors = np.clip(np.rint(colors.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    elif color_component == 5123 and bool(color_accessor.get("normalized")):
        colors = np.clip(np.rint(colors.astype(np.float32) / 257.0), 0, 255).astype(np.uint8)
    elif color_component == 5121 and bool(color_accessor.get("normalized")):
        colors = colors.astype(np.uint8, copy=False)
    else:
        raise ScenePriorError("scene-prior points GLB colors must be normalized RGB")
    positions = positions.astype(np.float32, copy=False)
    if not np.all(np.isfinite(positions)):
        raise ScenePriorError("scene-prior points GLB positions must be finite")
    return positions, np.ascontiguousarray(colors, dtype=np.uint8)


@dataclass(frozen=True)
class LoadedScenePrior:
    manifest: ScenePriorRevision
    root: Path
    arrays: Mapping[str, np.ndarray]
    points_world_m: np.ndarray
    colors_rgb_u8: np.ndarray

    def artifact(self, role: str) -> tuple[ScenePriorArtifact, bytes]:
        requested = str(role).strip()
        artifact = next(
            (item for item in self.manifest.artifacts if item.role == requested),
            None,
        )
        if artifact is None:
            raise ScenePriorError(
                f"scene-prior {self.manifest.prior_id} has no {requested!r} artifact"
            )
        try:
            verified = read_scene_root_file(
                self.root,
                artifact.relative_path,
                label=(
                    f"scene-prior {self.manifest.prior_id} artifact "
                    f"{artifact.role}"
                ),
                max_bytes=artifact.size_bytes,
                expected_sha256=artifact.sha256,
                expected_size=artifact.size_bytes,
            )
        except SceneFileError as exc:
            raise ScenePriorError(str(exc)) from exc
        return artifact, verified.data

    def _indices(self, world_x: np.ndarray, world_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grid = self.manifest.grid
        columns = np.floor(
            (np.asarray(world_x, dtype=np.float64) - float(grid.bounds.min_x))
            / float(grid.resolution_m)
        ).astype(np.int64)
        rows = np.floor(
            (np.asarray(world_z, dtype=np.float64) - float(grid.bounds.min_z))
            / float(grid.resolution_m)
        ).astype(np.int64)
        inside = (
            (rows >= 0)
            & (rows < int(grid.rows))
            & (columns >= 0)
            & (columns < int(grid.columns))
        )
        return rows, columns, inside

    def sample(self, world_x: np.ndarray, world_z: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(world_x, dtype=np.float64)
        z = np.asarray(world_z, dtype=np.float64)
        if x.shape != z.shape:
            raise ScenePriorError("scene-prior sample X/Z shapes must match")
        rows, columns, inside = self._indices(x, z)
        safe_rows = np.clip(rows, 0, int(self.manifest.grid.rows) - 1)
        safe_columns = np.clip(columns, 0, int(self.manifest.grid.columns) - 1)
        result: dict[str, np.ndarray] = {"inside_extent": inside}
        for name in _GRID_ARRAYS:
            source = np.asarray(self.arrays[name])
            sampled = source[safe_rows, safe_columns]
            if name == "height_agl_p95_m":
                sampled = np.where(inside, sampled, np.nan)
            elif source.dtype.kind in "bu":
                sampled = np.where(inside, sampled, 0)
            else:
                sampled = np.where(inside, sampled, 0.0)
            result[name] = sampled
        return result

    def evaluate(self, world_position_m: Sequence[float]) -> dict[str, Any]:
        if len(world_position_m) < 3:
            raise ScenePriorError("scene-prior world position must contain X/Y/Z")
        try:
            world = tuple(float(world_position_m[index]) for index in range(3))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ScenePriorError("scene-prior world position is invalid") from exc
        if not all(math.isfinite(value) for value in world):
            raise ScenePriorError("scene-prior world position must be finite")

        sampled = self.sample(np.asarray(world[0]), np.asarray(world[2]))
        inside_extent = bool(np.asarray(sampled["inside_extent"]).item())
        if not inside_extent:
            bounds = self.manifest.grid.bounds
            outside_x_m = max(
                float(bounds.min_x) - world[0],
                0.0,
                world[0] - float(bounds.max_x),
            )
            outside_z_m = max(
                float(bounds.min_z) - world[2],
                0.0,
                world[2] - float(bounds.max_z),
            )
            return {
                "contract": "noesis.scene_prior.track_diagnostic",
                "contract_version": 1,
                "prior_id": self.manifest.prior_id,
                "space_id": self.manifest.space_id,
                "mode": "shadow",
                "coordinate_frame": "backend_world_m",
                "status": "unknown",
                "inside_extent": False,
                "inside_authored_space": False,
                "extent_outside_distance_m": round(
                    math.hypot(outside_x_m, outside_z_m), 6
                ),
                "evidence_observed": False,
                "evidence_confidence": 0.0,
                "reasons": ["outside_prior_extent"],
            }

        def scalar(name: str) -> Any:
            return np.asarray(sampled[name]).item()

        inside_authored = bool(scalar("authored_walkable"))
        observed = bool(scalar("observed"))
        obstacle = bool(scalar("obstacle_mask"))
        confidence = float(scalar("evidence_confidence"))
        boundary = float(scalar("boundary_signed_distance_m"))
        obstacle_clearance = float(scalar("obstacle_signed_clearance_m"))
        static_height_raw = float(scalar("height_agl_p95_m"))
        static_height = static_height_raw if math.isfinite(static_height_raw) else None
        floor_height = float(scalar("floor_height_m"))
        reasons: list[str] = []
        status = "pass"
        if not inside_authored:
            status = "warning" if boundary >= -0.15 else "fail"
            reasons.append("outside_authored_space")
        elif not observed:
            status = "unknown"
            reasons.append("scene_prior_unobserved")
        else:
            if obstacle:
                status = "warning"
                reasons.append("inside_static_obstacle_candidate")
            if boundary < 0.15:
                status = "warning"
                reasons.append("near_authored_boundary")
        if not reasons:
            reasons.append("scene_consistent")
        return {
            "contract": "noesis.scene_prior.track_diagnostic",
            "contract_version": 1,
            "prior_id": self.manifest.prior_id,
            "space_id": self.manifest.space_id,
            "mode": "shadow",
            "coordinate_frame": "backend_world_m",
            "status": status,
            "inside_extent": True,
            "inside_authored_space": inside_authored,
            "extent_outside_distance_m": 0.0,
            "evidence_observed": observed,
            "evidence_confidence": round(confidence, 6),
            "boundary_signed_distance_m": round(boundary, 6),
            "obstacle_signed_clearance_m": round(obstacle_clearance, 6),
            "static_height_agl_m": (
                round(static_height, 6) if static_height is not None else None
            ),
            "floor_height_m": round(floor_height, 6),
            "reasons": reasons,
        }


@dataclass(frozen=True)
class _DiagnosticRasterGeometry:
    rows: int
    columns: int
    resolution_m: float
    min_x: float
    max_x: float
    min_z: float
    max_z: float
    camera_world_m: np.ndarray
    camera_right_world_xz: np.ndarray
    camera_forward_world_xz: np.ndarray

    def bounds_payload(self) -> dict[str, float]:
        return {
            "min_x": float(self.min_x),
            "max_x": float(self.max_x),
            "min_z": float(self.min_z),
            "max_z": float(self.max_z),
        }


def _decode_float32_layer(
    payload: Mapping[str, Any],
    name: str,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    layer = payload.get(name)
    if not isinstance(layer, Mapping):
        raise ScenePriorError(f"floorplan layer {name} is unavailable")
    shape_raw = layer.get("grid_shape", layer.get("shape"))
    if not isinstance(shape_raw, (list, tuple)) or len(shape_raw) != 2:
        raise ScenePriorError(f"floorplan layer {name} has no valid grid shape")
    shape = (int(shape_raw[0]), int(shape_raw[1]))
    if shape[0] <= 0 or shape[1] <= 0 or shape[0] * shape[1] > 16 * 1024 * 1024:
        raise ScenePriorError(f"floorplan layer {name} exceeds the grid bound")
    if expected_shape is not None and shape != expected_shape:
        raise ScenePriorError(f"floorplan layer {name} shape does not match height_agl")
    encoded = layer.get("grid_b64")
    if not isinstance(encoded, str) or not encoded:
        raise ScenePriorError(f"floorplan layer {name} has no encoded grid")
    cell_count = shape[0] * shape[1]
    encoded_payload = encoded
    dtype = np.dtype(np.float32)
    packed_bits = False
    if encoded.startswith("bit:"):
        try:
            _, encoded_count, encoded_payload = encoded.split(":", 2)
            packed_count = int(encoded_count)
        except (TypeError, ValueError) as exc:
            raise ScenePriorError(
                f"floorplan layer {name} has an invalid packed-mask header"
            ) from exc
        if packed_count != cell_count:
            raise ScenePriorError(
                f"floorplan layer {name} packed-mask count does not match its shape"
            )
        packed_bits = True
        expected_bytes = (cell_count + 7) // 8
    else:
        if encoded.startswith("f16:"):
            encoded_payload = encoded.removeprefix("f16:")
            dtype = np.dtype(np.float16)
        elif encoded.startswith("u8:"):
            encoded_payload = encoded.removeprefix("u8:")
            dtype = np.dtype(np.uint8)
        expected_bytes = cell_count * dtype.itemsize
    maximum_encoded_bytes = 4 * ((expected_bytes + 2) // 3)
    if len(encoded_payload) > maximum_encoded_bytes:
        raise ScenePriorError(f"floorplan layer {name} encoded grid exceeds its shape bound")
    try:
        raw = base64.b64decode(encoded_payload, validate=True)
    except Exception as exc:
        raise ScenePriorError(f"floorplan layer {name} is not valid base64") from exc
    if len(raw) != expected_bytes:
        raise ScenePriorError(f"floorplan layer {name} byte length does not match its shape")
    if packed_bits:
        return np.unpackbits(
            np.frombuffer(raw, dtype=np.uint8),
            count=cell_count,
            bitorder="big",
        ).astype(np.float32).reshape(shape)
    return np.frombuffer(raw, dtype=dtype).astype(np.float32).reshape(shape)


def _encoded_layer(values: np.ndarray, *, value_min: float, value_max: float) -> dict[str, Any]:
    # Scene-prior rasters are a visualization contract. Explicitly marked FP16
    # preserves binary masks exactly and retains sub-cell precision for metric
    # layers while keeping the complete response within the bounded transport.
    grid = np.asarray(values, dtype=np.float16)
    return {
        "grid_b64": "f16:" + base64.b64encode(grid.tobytes(order="C")).decode("ascii"),
        "grid_shape": [int(grid.shape[0]), int(grid.shape[1])],
        "value_min": float(value_min),
        "value_max": float(value_max),
    }


def _encoded_mask_layer(values: np.ndarray) -> dict[str, Any]:
    mask = np.ascontiguousarray(np.asarray(values) > 0, dtype=np.uint8)
    packed = np.packbits(mask.reshape(-1), bitorder="big")
    return {
        "grid_b64": (
            f"bit:{mask.size}:"
            + base64.b64encode(packed.tobytes(order="C")).decode("ascii")
        ),
        "grid_shape": [int(mask.shape[0]), int(mask.shape[1])],
        "value_min": 0.0,
        "value_max": 1.0,
    }


def _encoded_uint8_layer(
    values: np.ndarray,
    *,
    value_min: float,
    value_max: float,
) -> dict[str, Any]:
    grid = np.ascontiguousarray(values, dtype=np.uint8)
    return {
        "grid_b64": "u8:" + base64.b64encode(grid.tobytes(order="C")).decode("ascii"),
        "grid_shape": [int(grid.shape[0]), int(grid.shape[1])],
        "value_min": float(value_min),
        "value_max": float(value_max),
    }


def _encoded_rgb_layer(rgb: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    colors = np.ascontiguousarray(rgb, dtype=np.uint8)
    mask = np.ascontiguousarray(np.asarray(observed) > 0, dtype=np.uint8)
    packed_mask = np.packbits(mask.reshape(-1), bitorder="big")
    return {
        "rgb_b64": base64.b64encode(colors.tobytes()).decode("ascii"),
        "rgb_shape": [int(colors.shape[0]), int(colors.shape[1]), 3],
        "observed_b64": (
            f"bit:{mask.size}:"
            + base64.b64encode(packed_mask.tobytes()).decode("ascii")
        ),
    }


def _finite_percentile(values: np.ndarray, percentile: float, default: float = 0.0) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else float(default)


def _binary_perimeter(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    padded = np.pad(values, 1, mode="constant", constant_values=False)
    eroded = np.ones_like(values, dtype=bool)
    for row_offset in range(3):
        for column_offset in range(3):
            eroded &= padded[
                row_offset : row_offset + values.shape[0],
                column_offset : column_offset + values.shape[1],
            ]
    return values & ~eroded


def _sobel_gradient(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    source = np.where(
        np.asarray(observed, dtype=bool) & np.isfinite(values),
        np.asarray(values, dtype=np.float32),
        0.0,
    )
    padded = np.pad(source, 1, mode="edge")
    gx = (
        padded[:-2, 2:]
        + (2.0 * padded[1:-1, 2:])
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - (2.0 * padded[1:-1, :-2])
        - padded[2:, :-2]
    )
    gz = (
        padded[2:, :-2]
        + (2.0 * padded[2:, 1:-1])
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - (2.0 * padded[:-2, 1:-1])
        - padded[:-2, 2:]
    )
    magnitude = np.sqrt((gx * gx) + (gz * gz)).astype(np.float32)
    scale = _finite_percentile(magnitude, 95.0)
    if scale <= 1e-6:
        return np.zeros_like(source, dtype=np.float32)
    return np.clip(magnitude / scale, 0.0, 1.0).astype(np.float32)


def _camera_ground_basis(
    extrinsics_col_major: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        frame = camera_ground_frame_from_extrinsics_col_major(
            extrinsics_col_major
        )
    except CoordinateFrameError as exc:
        raise ScenePriorError(str(exc)) from exc
    return (
        frame.camera_world_m,
        frame.camera_right_world,
        frame.camera_forward_world,
    )


def _preview_raster_geometry(revision: LoadedScenePrior) -> _DiagnosticRasterGeometry:
    preview = revision.manifest.preview
    if preview is None:
        raise ScenePriorError(
            "scene-prior diagnostic floorplan requires camera preview geometry"
        )
    return _DiagnosticRasterGeometry(
        rows=int(preview.rows),
        columns=int(preview.columns),
        resolution_m=float(preview.resolution_m),
        min_x=float(preview.bounds.min_x),
        max_x=float(preview.bounds.max_x),
        min_z=float(preview.bounds.min_z),
        max_z=float(preview.bounds.max_z),
        camera_world_m=np.asarray(preview.camera_position_world_m, dtype=np.float64),
        camera_right_world_xz=np.asarray(
            preview.camera_right_world_xz, dtype=np.float64
        ),
        camera_forward_world_xz=np.asarray(
            preview.camera_forward_world_xz, dtype=np.float64
        ),
    )


def _calibrated_raster_geometry(
    revision: LoadedScenePrior,
    extrinsics_col_major: Sequence[float],
) -> _DiagnosticRasterGeometry:
    """Bound the canonical world prior in the current static-camera frame."""

    preview = revision.manifest.preview
    if preview is None:
        raise ScenePriorError(
            "scene-prior diagnostic floorplan requires camera preview resolution"
        )
    camera_world, right_world, forward_world = _camera_ground_basis(
        extrinsics_col_major
    )
    source_grid = revision.manifest.grid
    extent = np.asarray(revision.arrays["authored_walkable"]) > 0
    if revision.manifest.derivation.algorithm == _FULL_EVIDENCE_DERIVATION:
        extent |= np.asarray(revision.arrays["observed"]) > 0
    extent_rows, extent_columns = np.nonzero(extent)
    if extent_rows.size == 0:
        raise ScenePriorError("scene-prior authored room has no cells")
    source_resolution = float(source_grid.resolution_m)
    world_x = float(source_grid.bounds.min_x) + (
        extent_columns.astype(np.float64) + 0.5
    ) * source_resolution
    world_z = float(source_grid.bounds.min_z) + (
        extent_rows.astype(np.float64) + 0.5
    ) * source_resolution
    local_x = (
        (world_x - float(camera_world[0])) * float(right_world[0])
        + (world_z - float(camera_world[2])) * float(right_world[2])
    )
    local_z = (
        (world_x - float(camera_world[0])) * float(forward_world[0])
        + (world_z - float(camera_world[2])) * float(forward_world[2])
    )
    resolution = float(preview.resolution_m)
    epsilon = resolution * 1e-6
    half_extent_x = 0.5 * source_resolution * (
        abs(float(right_world[0])) + abs(float(right_world[2]))
    )
    half_extent_z = 0.5 * source_resolution * (
        abs(float(forward_world[0])) + abs(float(forward_world[2]))
    )
    min_column = math.floor(
        (float(np.min(local_x)) - half_extent_x + epsilon) / resolution
    )
    max_column = math.ceil(
        (float(np.max(local_x)) + half_extent_x - epsilon) / resolution
    )
    min_row = math.floor(
        (float(np.min(local_z)) - half_extent_z + epsilon) / resolution
    )
    max_row = math.ceil(
        (float(np.max(local_z)) + half_extent_z - epsilon) / resolution
    )
    columns = int(max_column - min_column)
    rows = int(max_row - min_row)
    if (
        rows <= 0
        or columns <= 0
        or rows > MAX_SCENE_PRIOR_GRID_DIMENSION
        or columns > MAX_SCENE_PRIOR_GRID_DIMENSION
        or rows * columns > MAX_SCENE_PRIOR_GRID_CELLS
    ):
        raise ScenePriorError("calibrated scene-prior diagnostic grid is outside bounds")
    return _DiagnosticRasterGeometry(
        rows=rows,
        columns=columns,
        resolution_m=resolution,
        min_x=float(min_column) * resolution,
        max_x=float(max_column) * resolution,
        min_z=float(min_row) * resolution,
        max_z=float(max_row) * resolution,
        camera_world_m=np.asarray(camera_world, dtype=np.float64),
        camera_right_world_xz=np.asarray(
            [right_world[0], right_world[2]], dtype=np.float64
        ),
        camera_forward_world_xz=np.asarray(
            [forward_world[0], forward_world[2]], dtype=np.float64
        ),
    )


def _derive_preview_diagnostics(
    revision: LoadedScenePrior,
    geometry: _DiagnosticRasterGeometry,
) -> dict[str, np.ndarray]:
    """Derive every dashboard raster from the admitted PCF revision only."""

    rows = int(geometry.rows)
    columns = int(geometry.columns)
    resolution_x = float(geometry.resolution_m)
    resolution_z = float(geometry.resolution_m)
    local_x = float(geometry.min_x) + (
        np.arange(columns, dtype=np.float64) + 0.5
    ) * resolution_x
    local_z = float(geometry.max_z) - (
        np.arange(rows, dtype=np.float64) + 0.5
    ) * resolution_z
    x_grid, z_grid = np.meshgrid(local_x, local_z)
    camera_world = np.asarray(geometry.camera_world_m, dtype=np.float64)
    right_world = np.asarray(geometry.camera_right_world_xz, dtype=np.float64)
    forward_world = np.asarray(geometry.camera_forward_world_xz, dtype=np.float64)
    world_x = (
        float(camera_world[0])
        + x_grid * float(right_world[0])
        + z_grid * float(forward_world[0])
    )
    world_z = (
        float(camera_world[2])
        + x_grid * float(right_world[1])
        + z_grid * float(forward_world[1])
    )
    sampled = revision.sample(world_x, world_z)
    inside = np.asarray(sampled["inside_extent"], dtype=bool)
    authored = inside & (np.asarray(sampled["authored_walkable"]) > 0)
    full_evidence = revision.manifest.derivation.algorithm == _FULL_EVIDENCE_DERIVATION
    evidence_extent = inside if full_evidence else authored
    observed = evidence_extent & (np.asarray(sampled["observed"]) > 0)
    floor_supported = evidence_extent & (
        np.asarray(sampled["floor_supported"]) > 0
    )
    obstacle_mask = evidence_extent & (np.asarray(sampled["obstacle_mask"]) > 0)
    height_agl = np.asarray(sampled["height_agl_p95_m"], dtype=np.float32)
    height_agl = np.where(
        observed & np.isfinite(height_agl),
        np.maximum(height_agl, 0.0),
        np.nan,
    ).astype(np.float32)
    evidence_confidence = np.where(
        observed,
        np.asarray(sampled["evidence_confidence"], dtype=np.float32),
        0.0,
    ).astype(np.float32)

    points = np.asarray(revision.points_world_m, dtype=np.float64)
    colors = np.asarray(revision.colors_rgb_u8, dtype=np.uint8)
    point_local = transform_positions(
        points,
        CameraGroundFrame(
            camera_world_m=camera_world,
            camera_right_world=np.asarray(
                [right_world[0], 0.0, right_world[1]],
                dtype=np.float64,
            ),
            camera_forward_world=np.asarray(
                [forward_world[0], 0.0, forward_world[1]],
                dtype=np.float64,
            ),
        ).world_to_camera_local_display_matrix(
            float(revision.manifest.derivation.floor_y_m)
        ),
    )
    point_columns = np.floor(
        (point_local[:, 0] - float(geometry.min_x)) / resolution_x
    ).astype(np.int64)
    point_rows = np.floor(
        (float(geometry.max_z) - point_local[:, 2]) / resolution_z
    ).astype(np.int64)
    source_grid = revision.manifest.grid
    source_columns = np.floor(
        (points[:, 0] - float(source_grid.bounds.min_x))
        / float(source_grid.resolution_m)
    ).astype(np.int64)
    source_rows = np.floor(
        (points[:, 2] - float(source_grid.bounds.min_z))
        / float(source_grid.resolution_m)
    ).astype(np.int64)
    valid = (
        np.isfinite(point_local).all(axis=1)
        & (point_columns >= 0)
        & (point_columns < columns)
        & (point_rows >= 0)
        & (point_rows < rows)
        & (source_columns >= 0)
        & (source_columns < int(source_grid.columns))
        & (source_rows >= 0)
        & (source_rows < int(source_grid.rows))
    )
    point_columns = point_columns[valid]
    point_rows = point_rows[valid]
    point_local = point_local[valid]
    colors = colors[valid]
    source_columns = source_columns[valid]
    source_rows = source_rows[valid]
    confidence = np.clip(
        np.asarray(revision.arrays["evidence_confidence"], dtype=np.float32)[
            source_rows, source_columns
        ],
        0.0,
        1.0,
    )
    weights = np.maximum(confidence, 1e-3)
    linear = (point_rows * columns) + point_columns
    cell_count = rows * columns

    density_count = np.bincount(linear, minlength=cell_count).reshape((rows, columns))
    density_scale = _finite_percentile(density_count, 99.0)
    density = (
        np.clip(density_count.astype(np.float32) / density_scale, 0.0, 1.0)
        if density_scale > 0.0
        else np.zeros((rows, columns), dtype=np.float32)
    ).astype(np.float32)
    distance = np.where(observed, np.hypot(x_grid, z_grid), np.nan).astype(np.float32)
    walkable_extent = evidence_extent if full_evidence else authored
    walkable = (
        walkable_extent
        & (np.asarray(sampled["walkable_candidate"]) > 0)
        & ~obstacle_mask
    ).astype(np.float32)
    obstacle_height = np.where(obstacle_mask, height_agl, np.nan).astype(np.float32)

    surface_bin_count = int(
        math.ceil(_DIAGNOSTIC_SURFACE_HEIGHT_MAX_M / _DIAGNOSTIC_HEIGHT_BIN_M)
    )
    surface_histogram = np.zeros(
        (cell_count, surface_bin_count), dtype=np.float32
    )
    heights = point_local[:, 1].astype(np.float32)
    surface_candidates = (
        (heights >= 0.0) & (heights <= _DIAGNOSTIC_SURFACE_HEIGHT_MAX_M)
    )
    if np.any(surface_candidates):
        height_bins = np.clip(
            np.floor(
                heights[surface_candidates] / _DIAGNOSTIC_HEIGHT_BIN_M
            ).astype(np.int64),
            0,
            surface_bin_count - 1,
        )
        np.add.at(
            surface_histogram,
            (linear[surface_candidates], height_bins),
            weights[surface_candidates],
        )
    smoothed_histogram = surface_histogram.copy()
    smoothed_histogram[:, 1:] += surface_histogram[:, :-1] * 0.25
    smoothed_histogram[:, :-1] += surface_histogram[:, 1:] * 0.25
    modal_bin = np.argmax(smoothed_histogram, axis=1)
    modal_support = smoothed_histogram[np.arange(cell_count), modal_bin]
    surface_observed = modal_support > 0.0
    modal_height = (
        (modal_bin.astype(np.float32) + 0.5) * _DIAGNOSTIC_HEIGHT_BIN_M
    )
    structural_height = np.full(cell_count, np.nan, dtype=np.float32)
    structural_height[observed.reshape(-1)] = 0.0
    furniture = (
        surface_observed
        & (modal_height >= _DIAGNOSTIC_FURNITURE_MIN_M)
        & (modal_height <= _DIAGNOSTIC_FURNITURE_MAX_M)
    )
    structural_height[furniture] = modal_height[furniture]

    surface_rgb_sum = np.zeros((cell_count, 3), dtype=np.float64)
    surface_rgb_weight = np.zeros(cell_count, dtype=np.float64)
    point_modal_height = modal_height[np.clip(linear, 0, cell_count - 1)]
    rgb_valid = (
        surface_candidates
        & surface_observed[np.clip(linear, 0, cell_count - 1)]
        & (
            np.abs(heights - point_modal_height)
            <= (_DIAGNOSTIC_HEIGHT_BIN_M * 1.5)
        )
    )
    if np.any(rgb_valid):
        selected_cells = linear[rgb_valid]
        selected_weights = weights[rgb_valid].astype(np.float64)
        np.add.at(surface_rgb_weight, selected_cells, selected_weights)
        for channel in range(3):
            np.add.at(
                surface_rgb_sum[:, channel],
                selected_cells,
                colors[rgb_valid, channel].astype(np.float64) * selected_weights,
            )
    np.divide(
        surface_rgb_sum,
        surface_rgb_weight[:, None],
        out=surface_rgb_sum,
        where=surface_rgb_weight[:, None] > 1e-9,
    )
    surface_rgb = np.clip(np.rint(surface_rgb_sum), 0, 255).astype(np.uint8)
    surface_rgb_observed = surface_rgb_weight > 1e-9

    vertical_bin_m = 0.10
    vertical_bin_count = int(math.ceil(2.40 / vertical_bin_m))
    vertical_histogram = np.zeros(
        (cell_count, vertical_bin_count), dtype=np.uint16
    )
    vertical_candidates = (heights >= 0.0) & (heights <= 2.40)
    if np.any(vertical_candidates):
        vertical_bins = np.clip(
            np.floor(heights[vertical_candidates] / vertical_bin_m).astype(np.int64),
            0,
            vertical_bin_count - 1,
        )
        np.add.at(
            vertical_histogram,
            (linear[vertical_candidates], vertical_bins),
            1,
        )
    occupied_bins = vertical_histogram > 0
    occupied_count = np.count_nonzero(occupied_bins, axis=1)
    first_bin = np.argmax(occupied_bins, axis=1)
    last_bin = vertical_bin_count - 1 - np.argmax(occupied_bins[:, ::-1], axis=1)
    vertical_span = np.where(
        occupied_count > 0,
        (last_bin - first_bin).astype(np.float32) * vertical_bin_m,
        0.0,
    )
    wall_support = (
        np.clip((occupied_count.astype(np.float32) - 2.0) / 8.0, 0.0, 1.0)
        * np.clip((vertical_span - 0.30) / 1.20, 0.0, 1.0)
    )
    wall_support[occupied_count < 4] = 0.0
    wall_support = wall_support.reshape((rows, columns)).astype(np.float32)
    measured_perimeter = _binary_perimeter(observed).astype(np.float32)
    room_footprint = authored.astype(np.float32)
    reconstruction_extent = (authored | observed).astype(np.float32)
    room_boundary = np.maximum(
        wall_support,
        _binary_perimeter(authored).astype(np.float32) * 0.35,
    ).astype(np.float32)

    diagnostics: dict[str, np.ndarray] = {
        "density": density,
        "height": height_agl.copy(),
        "height_agl": height_agl,
        "distance": distance,
        "gradient": _sobel_gradient(height_agl, observed),
        "obstacle_height": obstacle_height,
        "obstacle_mask": obstacle_mask.astype(np.float32),
        "walkable": walkable,
        "observed": observed.astype(np.float32),
        "unknown": (
            (reconstruction_extent.astype(bool) & ~observed)
            if full_evidence
            else ~observed
        ).astype(np.float32),
        "inferred_walkable": (authored & ~observed & ~obstacle_mask).astype(np.float32),
        "structural_height": structural_height.reshape((rows, columns)),
        "surface_observed": surface_observed.reshape((rows, columns)).astype(np.float32),
        "room_footprint": room_footprint,
        "reconstruction_extent": reconstruction_extent,
        "wall_support": wall_support,
        "room_boundary": room_boundary,
        "measured_perimeter": measured_perimeter,
        "surface_rgb": surface_rgb.reshape((rows, columns, 3)),
        "surface_rgb_observed": surface_rgb_observed.reshape((rows, columns)).astype(np.float32),
        "confidence": evidence_confidence,
        "floor_supported": floor_supported.astype(np.float32),
        "floor_height": np.where(floor_supported, 0.0, np.nan).astype(np.float32),
    }
    return diagnostics


def _scene_prior_diagnostic_payload(
    revision: LoadedScenePrior,
    diagnostics: Mapping[str, np.ndarray],
    geometry: _DiagnosticRasterGeometry,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "scene_prior_diagnostic_density": _encoded_layer(
            diagnostics["density"], value_min=0.0, value_max=1.0
        ),
        "scene_prior_diagnostic_height": _encoded_layer(
            diagnostics["height"],
            value_min=0.0,
            value_max=_finite_percentile(diagnostics["height"], 99.0),
        ),
        "scene_prior_diagnostic_height_agl": _encoded_layer(
            diagnostics["height_agl"],
            value_min=0.0,
            value_max=_finite_percentile(diagnostics["height_agl"], 99.0),
        ),
        "scene_prior_diagnostic_distance": _encoded_layer(
            diagnostics["distance"],
            value_min=_finite_percentile(diagnostics["distance"], 1.0),
            value_max=_finite_percentile(diagnostics["distance"], 99.0),
        ),
        "scene_prior_diagnostic_gradient": _encoded_layer(
            diagnostics["gradient"], value_min=0.0, value_max=1.0
        ),
        "scene_prior_diagnostic_obstacle_height": _encoded_layer(
            diagnostics["obstacle_height"],
            value_min=0.0,
            value_max=_finite_percentile(diagnostics["obstacle_height"], 99.0),
        ),
        "scene_prior_diagnostic_obstacle_mask": _encoded_mask_layer(
            diagnostics["obstacle_mask"]
        ),
        "scene_prior_diagnostic_walkable": _encoded_mask_layer(diagnostics["walkable"]),
        "scene_prior_diagnostic_observed": _encoded_mask_layer(diagnostics["observed"]),
        "scene_prior_diagnostic_unknown": _encoded_mask_layer(diagnostics["unknown"]),
        "scene_prior_diagnostic_inferred_walkable": _encoded_mask_layer(
            diagnostics["inferred_walkable"]
        ),
        "scene_prior_diagnostic_structural_height": _encoded_layer(
            diagnostics["structural_height"],
            value_min=0.0,
            value_max=_DIAGNOSTIC_FURNITURE_MAX_M,
        ),
        "scene_prior_diagnostic_surface_observed": _encoded_mask_layer(
            diagnostics["surface_observed"]
        ),
        "scene_prior_diagnostic_room_footprint": _encoded_mask_layer(
            diagnostics["room_footprint"]
        ),
        "scene_prior_diagnostic_reconstruction_extent": _encoded_mask_layer(
            diagnostics["reconstruction_extent"]
        ),
        "scene_prior_diagnostic_wall_support": _encoded_layer(
            diagnostics["wall_support"], value_min=0.0, value_max=1.0
        ),
        "scene_prior_diagnostic_room_boundary": _encoded_layer(
            diagnostics["room_boundary"], value_min=0.0, value_max=1.0
        ),
        "scene_prior_diagnostic_measured_perimeter": _encoded_mask_layer(
            diagnostics["measured_perimeter"]
        ),
        "scene_prior_diagnostic_surface_rgb": _encoded_rgb_layer(
            diagnostics["surface_rgb"], diagnostics["surface_rgb_observed"]
        ),
        "scene_prior_diagnostic_confidence": _encoded_layer(
            diagnostics["confidence"], value_min=0.0, value_max=1.0
        ),
        "scene_prior_floor_supported": _encoded_mask_layer(
            diagnostics["floor_supported"]
        ),
        "scene_prior_floor_height": _encoded_layer(
            diagnostics["floor_height"], value_min=0.0, value_max=0.0
        ),
        "scene_prior_diagnostic_meta": {
            "contract": "noesis.scene_prior.diagnostic_layers",
            "contract_version": 1,
            "source": revision.manifest.source.model,
            "derivation": "prior_conditioned_fusion_points_and_grid",
            "derived_at_catalog_load": False,
            "mapanything_inference_triggered": False,
            "registration_triggered": False,
            "bounds": geometry.bounds_payload(),
            "grid_shape": [int(geometry.rows), int(geometry.columns)],
            "resolution_m": float(geometry.resolution_m),
            "camera_position_world_m": [
                float(value) for value in geometry.camera_world_m.tolist()
            ],
            "camera_right_world_xz": [
                float(value) for value in geometry.camera_right_world_xz.tolist()
            ],
            "camera_forward_world_xz": [
                float(value) for value in geometry.camera_forward_world_xz.tolist()
            ],
            "raster_orientation": CAMERA_LOCAL_RASTER_ORIENTATION,
            "point_count": int(revision.points_world_m.shape[0]),
        },
    }
    return payload


class ScenePriorSet:
    """Exact configured scene priors and camera bindings for one site."""

    def __init__(
        self,
        catalog: ScenePriorCatalog,
        revisions: Mapping[str, LoadedScenePrior],
    ) -> None:
        self.catalog = catalog
        self._revisions = dict(revisions)
        self._bindings = {binding.camera_id: binding for binding in catalog.camera_bindings}

    @classmethod
    def load(cls, catalog_path: str | Path) -> "ScenePriorSet":
        path = absolute_path_without_resolving(catalog_path)
        try:
            catalog_file = read_scene_file(
                path,
                label="scene-prior catalog",
                max_bytes=MAX_SCENE_PRIOR_CATALOG_BYTES,
            )
            catalog = _load_model(
                catalog_file.data,
                ScenePriorCatalog,
                label="scene-prior catalog",
            )
            entry_by_id = {entry.prior_id: entry for entry in catalog.revisions}
            required_ids = {binding.prior_id for binding in catalog.camera_bindings}
            revisions: dict[str, LoadedScenePrior] = {}
            for prior_id in sorted(required_ids):
                entry = entry_by_id[prior_id]
                manifest_file = read_scene_root_file(
                    path.parent,
                    entry.manifest_path,
                    label=f"scene-prior manifest {prior_id}",
                    max_bytes=MAX_SCENE_PRIOR_MANIFEST_BYTES,
                    expected_sha256=entry.manifest_sha256,
                    expected_size=entry.manifest_size_bytes,
                )
                manifest = _load_model(
                    manifest_file.data,
                    ScenePriorRevision,
                    label=f"scene-prior manifest {prior_id}",
                )
                if manifest.prior_id != entry.prior_id or manifest.space_id != entry.space_id:
                    raise ScenePriorError(
                        f"scene-prior catalog identity does not match manifest {prior_id}"
                    )
                if manifest.site_id != catalog.site_id:
                    raise ScenePriorError(
                        f"scene-prior manifest {prior_id} belongs to another site"
                    )
                manifest_root = path.parent / Path(entry.manifest_path).parent
                artifacts: dict[str, bytes] = {}
                for artifact in manifest.artifacts:
                    verified = read_scene_root_file(
                        manifest_root,
                        artifact.relative_path,
                        label=f"scene-prior {prior_id} artifact {artifact.role}",
                        max_bytes=artifact.size_bytes,
                        expected_sha256=artifact.sha256,
                        expected_size=artifact.size_bytes,
                    )
                    artifacts[artifact.role] = verified.data
                grid_bytes = artifacts.get("grid_npz")
                if grid_bytes is None or len(grid_bytes) > MAX_SCENE_PRIOR_GRID_ARCHIVE_BYTES:
                    raise ScenePriorError(f"scene-prior {prior_id} grid artifact is invalid")
                arrays = _read_grid_npz(
                    grid_bytes,
                    shape=(int(manifest.grid.rows), int(manifest.grid.columns)),
                )
                points_bytes = artifacts.get("points_glb")
                if points_bytes is None:
                    raise ScenePriorError(
                        f"scene-prior {prior_id} points artifact is unavailable"
                    )
                points_world_m, colors_rgb_u8 = _read_points_glb(points_bytes)
                revisions[prior_id] = LoadedScenePrior(
                    manifest=manifest,
                    root=manifest_root,
                    arrays=arrays,
                    points_world_m=points_world_m,
                    colors_rgb_u8=colors_rgb_u8,
                )
            for binding in catalog.camera_bindings:
                frame_binding = binding.frame_binding
                if frame_binding is None:
                    continue
                revision = revisions[binding.prior_id]
                preview = revision.manifest.preview
                if preview is None:
                    raise ScenePriorError(
                        f"scene-prior camera {binding.camera_id} frame binding requires preview provenance"
                    )
                if (
                    preview.camera_calibration.sha256
                    != frame_binding.source_camera_calibration_sha256
                ):
                    raise ScenePriorError(
                        f"scene-prior camera {binding.camera_id} calibration revision mismatch"
                    )
                if (
                    revision.manifest.world_to_scene.sha256
                    != frame_binding.source_world_alignment_sha256
                ):
                    raise ScenePriorError(
                        f"scene-prior camera {binding.camera_id} world-alignment revision mismatch"
                    )
                if (
                    preview.target_revision_metadata.sha256
                    != frame_binding.target_revision_metadata_sha256
                ):
                    raise ScenePriorError(
                        f"scene-prior camera {binding.camera_id} target revision mismatch"
                    )
        except ScenePriorError:
            raise
        except SceneFileError as exc:
            raise ScenePriorError(str(exc)) from exc
        return cls(catalog, revisions)

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._bindings))

    def binding(self, camera_id: str) -> ScenePriorCameraBinding | None:
        return self._bindings.get(str(camera_id))

    def frame_binding(self, camera_id: str) -> ScenePriorFrameBinding | None:
        binding = self.binding(camera_id)
        return binding.frame_binding if binding is not None else None

    def frame_bindings(self) -> dict[str, ScenePriorFrameBinding]:
        return {
            camera_id: binding.frame_binding
            for camera_id, binding in self._bindings.items()
            if binding.frame_binding is not None
        }

    def world_frame_transform(
        self,
        camera_id: str,
    ) -> RevisionedFrameTransform | None:
        binding = self.frame_binding(camera_id)
        return binding.frame_transform() if binding is not None else None

    def _scene_world_extrinsics(
        self,
        camera_id: str,
        calibration_extrinsics_col_major: Sequence[float],
    ) -> Sequence[float]:
        transform = self.world_frame_transform(camera_id)
        if transform is None:
            return calibration_extrinsics_col_major
        try:
            return transform.camera_from_target_col_major(
                calibration_extrinsics_col_major
            )
        except CoordinateFrameError as exc:
            raise ScenePriorError(str(exc)) from exc

    def _world_frame_metadata(self, camera_id: str) -> tuple[str, str]:
        binding = self.frame_binding(camera_id)
        if binding is None:
            revision = self.revision_for_camera(camera_id)
            if revision is None:
                raise ScenePriorError(
                    f"scene-prior camera {camera_id!r} has no loaded revision"
                )
            preview = revision.manifest.preview
            if preview is None:
                raise ScenePriorError(
                    f"scene-prior camera {camera_id!r} has no raw frame provenance"
                )
            return (
                BACKEND_WORLD_FRAME_ID,
                revisioned_frame_sha256(
                    BACKEND_WORLD_FRAME_ID,
                    artifact_sha256s=(
                        preview.camera_calibration.sha256,
                        revision.manifest.world_to_scene.sha256,
                    ),
                ),
            )
        return (
            binding.target_frame.frame_id,
            binding.target_frame.revision,
        )

    def revision_for_camera(self, camera_id: str) -> LoadedScenePrior | None:
        binding = self.binding(camera_id)
        if binding is None:
            return None
        return self._revisions[binding.prior_id]

    def camera_view_metadata(
        self,
        camera_id: str,
        *,
        extrinsics_col_major: Sequence[float],
    ) -> dict[str, Any]:
        binding = self.binding(camera_id)
        revision = self.revision_for_camera(camera_id)
        if binding is None or revision is None:
            raise ScenePriorError(
                f"scene-prior camera {camera_id!r} has no loaded revision"
            )
        world_extrinsics = self._scene_world_extrinsics(
            camera_id,
            extrinsics_col_major,
        )
        camera_world, right_world, forward_world = _camera_ground_basis(
            world_extrinsics,
        )
        world_frame, world_frame_revision = self._world_frame_metadata(camera_id)
        frame_binding = self.frame_binding(camera_id)
        floor_y_m = float(revision.manifest.derivation.floor_y_m)
        world_to_camera_local = CameraGroundFrame(
            camera_world_m=camera_world,
            camera_right_world=right_world,
            camera_forward_world=forward_world,
        ).world_to_camera_local_display_matrix(floor_y_m)
        linear_determinant = float(
            np.linalg.det(world_to_camera_local[:3, :3])
        )
        points_artifact = next(
            (
                artifact
                for artifact in revision.manifest.artifacts
                if artifact.role == "points_glb"
            ),
            None,
        )
        if points_artifact is None:
            raise ScenePriorError(
                f"scene-prior {revision.manifest.prior_id} has no points_glb artifact"
            )
        return {
            "contract": "noesis.scene_prior.camera_view",
            "contract_version": 1,
            "camera_id": str(camera_id),
            "site_id": revision.manifest.site_id,
            "space_id": revision.manifest.space_id,
            "prior_id": revision.manifest.prior_id,
            "mode": binding.mode,
            "source_type": revision.manifest.source.source_type,
            "source_model": revision.manifest.source.model,
            "source_coordinate_frame": world_frame,
            "source_coordinate_frame_revision": world_frame_revision,
            "target_coordinate_frame": "camera_local_ground_m",
            "transform_semantics": "presentation_only_coordinate_frame_conversion",
            "backend_geometry_mutated": False,
            "linear_determinant": linear_determinant,
            "orientation": {
                "screen_right": "camera_right_positive_x",
                "screen_up": "camera_forward_positive_z",
                "vertical": "height_above_floor_positive_y",
            },
            "camera_position_world_m": [
                float(value) for value in camera_world.tolist()
            ],
            "camera_right_world": [float(value) for value in right_world.tolist()],
            "camera_forward_world": [
                float(value) for value in forward_world.tolist()
            ],
            "floor_y_m": floor_y_m,
            "frame_transform_sha256": (
                frame_binding.target_from_source_sha256
                if frame_binding is not None
                else None
            ),
            "target_revision_id": (
                frame_binding.target_revision_id
                if frame_binding is not None
                else None
            ),
            "world_to_camera_local_row_major": world_to_camera_local.tolist(),
            "quality": revision.manifest.quality.model_dump(mode="json"),
            "points": {
                "role": points_artifact.role,
                "sha256": points_artifact.sha256,
                "size_bytes": points_artifact.size_bytes,
            },
        }

    def artifact_for_camera(
        self,
        camera_id: str,
        role: str,
    ) -> tuple[ScenePriorArtifact, bytes]:
        revision = self.revision_for_camera(camera_id)
        if revision is None:
            raise ScenePriorError(
                f"scene-prior camera {camera_id!r} has no loaded revision"
            )
        return revision.artifact(role)

    def evaluate(self, camera_id: str, world_position_m: Sequence[float]) -> dict[str, Any] | None:
        revision = self.revision_for_camera(camera_id)
        if revision is None:
            return None
        return revision.evaluate(world_position_m)

    def compose_static_floorplan(
        self,
        camera_id: str,
        payload: Mapping[str, Any],
        *,
        extrinsics_col_major: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Compose either the explicit PCF presentation or cache-miss prior."""

        binding = self.binding(camera_id)
        revision = self.revision_for_camera(camera_id)
        if binding is None or revision is None or not binding.include_floorplan_layers:
            return dict(payload)
        preview = revision.manifest.preview
        if preview is None:
            raise ScenePriorError(
                "scene-prior static-only floorplan requires camera preview geometry"
            )

        result = dict(payload)
        explicit_scene_prior_only = result.get("scene_prior_only") is True
        if explicit_scene_prior_only:
            if extrinsics_col_major is None:
                raise ScenePriorError(
                    "canonical PCF presentation requires calibrated camera extrinsics"
                )
            geometry = _calibrated_raster_geometry(
                revision,
                self._scene_world_extrinsics(camera_id, extrinsics_col_major),
            )
        else:
            geometry = _preview_raster_geometry(revision)

        rows = int(geometry.rows)
        columns = int(geometry.columns)
        local_x = float(geometry.min_x) + (
            np.arange(columns, dtype=np.float64) + 0.5
        ) * float(geometry.resolution_m)
        local_z = float(geometry.max_z) - (
            np.arange(rows, dtype=np.float64) + 0.5
        ) * float(geometry.resolution_m)
        x_grid, z_grid = np.meshgrid(local_x, local_z)
        camera_world = np.asarray(geometry.camera_world_m, dtype=np.float64)
        right_world = np.asarray(geometry.camera_right_world_xz, dtype=np.float64)
        forward_world = np.asarray(geometry.camera_forward_world_xz, dtype=np.float64)
        world_x = (
            float(camera_world[0])
            + x_grid * float(right_world[0])
            + z_grid * float(forward_world[0])
        )
        world_z = (
            float(camera_world[2])
            + x_grid * float(right_world[1])
            + z_grid * float(forward_world[1])
        )
        sampled = revision.sample(world_x, world_z)
        static_height = np.asarray(sampled["height_agl_p95_m"], dtype=np.float32)
        static_valid = (
            np.asarray(sampled["inside_extent"], dtype=bool)
            & (np.asarray(sampled["authored_walkable"]) > 0)
            & (np.asarray(sampled["observed"]) > 0)
            & np.isfinite(static_height)
        )
        static_height = np.where(
            static_valid & np.isfinite(static_height),
            np.maximum(static_height, 0.0),
            np.nan,
        ).astype(np.float32)
        static_observed = static_valid.astype(np.float32)
        static_confidence = np.where(
            static_valid,
            np.asarray(sampled["evidence_confidence"], dtype=np.float32),
            0.0,
        ).astype(np.float32)
        finite_static = static_height[np.isfinite(static_height)]
        static_max = (
            float(np.percentile(finite_static, 99)) if finite_static.size else 0.0
        )
        static_source = static_valid.astype(np.float32)

        live_error = str(result.pop("error", "") or "").strip()
        catalog_entry = next(
            (
                entry
                for entry in self.catalog.revisions
                if entry.prior_id == revision.manifest.prior_id
            ),
            None,
        )
        if catalog_entry is None:
            raise ScenePriorError(
                f"scene-prior {revision.manifest.prior_id} is absent from its catalog"
            )
        world_frame, world_frame_revision = self._world_frame_metadata(camera_id)
        frame_binding = self.frame_binding(camera_id)
        presentation_identity: dict[str, Any] = {
            "camera_id": camera_id,
            "scene_prior_only": explicit_scene_prior_only,
            "point_count": int(revision.points_world_m.shape[0]),
        }
        if explicit_scene_prior_only:
            presentation_identity.update(
                {
                    "display_source": "pcf",
                    "served_from_cache": True,
                    "ts": int(revision.manifest.created_at_us),
                }
            )
        result.update(
            {
                **presentation_identity,
                "snapshot_id": revision.manifest.prior_id,
                "snapshot_content_sha256": catalog_entry.manifest_sha256,
                "world_frame": world_frame,
                "world_frame_revision": world_frame_revision,
                "frame": preview.coordinate_frame,
                "units": preview.units,
                "bounds": geometry.bounds_payload(),
                "scale_m_per_px": float(geometry.resolution_m),
                "scene_static_height_agl": _encoded_layer(
                    static_height,
                    value_min=0.0,
                    value_max=max(0.0, static_max),
                ),
                "scene_static_observed": _encoded_mask_layer(static_observed),
                "scene_static_confidence": _encoded_layer(
                    static_confidence,
                    value_min=0.0,
                    value_max=1.0,
                ),
                "scene_composite_height_agl": _encoded_layer(
                    static_height,
                    value_min=0.0,
                    value_max=max(0.0, static_max),
                ),
                "scene_composite_observed": _encoded_mask_layer(static_observed),
                "scene_composite_source": _encoded_uint8_layer(
                    static_source,
                    value_min=0.0,
                    value_max=2.0,
                ),
                "scene_prior_meta": {
                    "contract": "noesis.scene_prior.floorplan_composite",
                    "contract_version": 1,
                    "status": "pcf" if explicit_scene_prior_only else "static_only",
                    "prior_id": revision.manifest.prior_id,
                    "revision_manifest_path": catalog_entry.manifest_path,
                    "revision_manifest_sha256": catalog_entry.manifest_sha256,
                    "space_id": revision.manifest.space_id,
                    "mode": binding.mode,
                    "source_type": revision.manifest.source.source_type,
                    "source_model": revision.manifest.source.model,
                    "source_frame": world_frame,
                    "source_frame_revision": world_frame_revision,
                    "frame_transform_sha256": (
                        frame_binding.target_from_source_sha256
                        if frame_binding is not None
                        else None
                    ),
                    "target_revision_id": (
                        frame_binding.target_revision_id
                        if frame_binding is not None
                        else None
                    ),
                    "target_frame": preview.coordinate_frame,
                    "camera_geometry_source": (
                        "current_calibrated_extrinsics"
                        if explicit_scene_prior_only
                        else "manifest_preview"
                    ),
                    "display_source": (
                        "pcf" if explicit_scene_prior_only else "static_fallback"
                    ),
                    "raster_orientation": CAMERA_LOCAL_RASTER_ORIENTATION,
                    "floor_y_m": float(revision.manifest.derivation.floor_y_m),
                    "composition_policy": (
                        "pcf_only"
                        if explicit_scene_prior_only
                        else "static_only_live_unavailable"
                    ),
                    "live_observed_cells": 0,
                    "static_available_cells": int(np.count_nonzero(static_valid)),
                    "static_fill_cells": int(np.count_nonzero(static_valid)),
                    "composite_observed_cells": int(np.count_nonzero(static_valid)),
                    "quality": revision.manifest.quality.model_dump(mode="json"),
                    "reason": (
                        "canonical_pcf_presentation_source"
                        if explicit_scene_prior_only
                        else "live_floorplan_unavailable"
                    ),
                },
            }
        )
        if explicit_scene_prior_only:
            diagnostics = _derive_preview_diagnostics(revision, geometry)
            result.update(
                _scene_prior_diagnostic_payload(revision, diagnostics, geometry)
            )
            for redundant_layer in (
                "scene_static_height_agl",
                "scene_static_observed",
                "scene_static_confidence",
                "scene_composite_height_agl",
                "scene_composite_observed",
                "scene_composite_source",
            ):
                result.pop(redundant_layer, None)
        if live_error:
            result["live_floorplan_error"] = live_error
        return result

    def compose_floorplan(
        self,
        camera_id: str,
        payload: Mapping[str, Any],
        *,
        extrinsics_col_major: Sequence[float],
        floor_y_m: float,
    ) -> dict[str, Any]:
        binding = self.binding(camera_id)
        revision = self.revision_for_camera(camera_id)
        if binding is None or revision is None or not binding.include_floorplan_layers:
            return dict(payload)
        if (
            not math.isfinite(float(floor_y_m))
            or abs(float(floor_y_m) - float(revision.manifest.derivation.floor_y_m)) > 1e-6
        ):
            raise ScenePriorError("scene-prior floor reference does not match its revision")
        if payload.get("frame") != "camera_local_ground_m" or payload.get("units") != "meters":
            raise ScenePriorError("scene-prior composition requires camera_local_ground_m meters")
        bounds = payload.get("bounds")
        if not isinstance(bounds, Mapping):
            raise ScenePriorError("scene-prior composition requires floorplan bounds")
        world_extrinsics = self._scene_world_extrinsics(
            camera_id,
            extrinsics_col_major,
        )
        world_frame, world_frame_revision = self._world_frame_metadata(camera_id)
        frame_binding = self.frame_binding(camera_id)
        live_height = _decode_float32_layer(payload, "height_agl")
        shape = live_height.shape
        try:
            live_observed = _decode_float32_layer(payload, "observed", expected_shape=shape)
        except ScenePriorError:
            live_observed = _decode_float32_layer(payload, "density", expected_shape=shape)
        min_x = float(bounds["min_x"])
        max_x = float(bounds["max_x"])
        min_z = float(bounds["min_z"])
        max_z = float(bounds["max_z"])
        if not all(math.isfinite(value) for value in (min_x, max_x, min_z, max_z)):
            raise ScenePriorError("scene-prior composition floorplan bounds must be finite")
        if max_x <= min_x or max_z <= min_z:
            raise ScenePriorError("scene-prior composition floorplan bounds are invalid")

        rows, columns = shape
        local_x = min_x + (np.arange(columns, dtype=np.float64) + 0.5) * (
            (max_x - min_x) / float(columns)
        )
        # Floorplan rasters are display-oriented: row zero is the far/forward
        # edge and increasing rows move back toward the camera.  Preserve that
        # contract while sampling the backend-world prior; treating row zero as
        # min_z vertically mirrors only the static/composite layers.
        local_z = max_z - (np.arange(rows, dtype=np.float64) + 0.5) * (
            (max_z - min_z) / float(rows)
        )
        x_grid, z_grid = np.meshgrid(local_x, local_z)
        camera_world, right_world, forward_world = _camera_ground_basis(
            world_extrinsics,
        )
        world_x = (
            float(camera_world[0])
            + x_grid * float(right_world[0])
            + z_grid * float(forward_world[0])
        )
        world_z = (
            float(camera_world[2])
            + x_grid * float(right_world[2])
            + z_grid * float(forward_world[2])
        )
        sampled = revision.sample(world_x, world_z)
        static_height = np.asarray(sampled["height_agl_p95_m"], dtype=np.float32)
        static_valid = (
            np.asarray(sampled["inside_extent"], dtype=bool)
            & (np.asarray(sampled["authored_walkable"]) > 0)
            & (np.asarray(sampled["observed"]) > 0)
            & np.isfinite(static_height)
        )
        static_height = np.where(static_valid & np.isfinite(static_height), np.maximum(static_height, 0.0), np.nan)
        static_observed = static_valid.astype(np.float32)
        static_confidence = np.where(
            static_valid,
            np.asarray(sampled["evidence_confidence"], dtype=np.float32),
            0.0,
        )
        live_valid = (live_observed > 0.0) & np.isfinite(live_height)
        composite_valid = live_valid | static_valid
        composite_height = np.where(live_valid, live_height, static_height).astype(np.float32)
        composite_height[~composite_valid] = np.nan
        composite_source = np.zeros(shape, dtype=np.float32)
        composite_source[static_valid] = 1.0
        composite_source[live_valid] = 2.0
        finite_static = static_height[np.isfinite(static_height)]
        finite_composite = composite_height[np.isfinite(composite_height)]
        static_max = float(np.percentile(finite_static, 99)) if finite_static.size else 0.0
        composite_max = float(np.percentile(finite_composite, 99)) if finite_composite.size else 0.0

        result = dict(payload)
        result["scene_static_height_agl"] = _encoded_layer(
            static_height,
            value_min=0.0,
            value_max=max(0.0, static_max),
        )
        result["scene_static_observed"] = _encoded_mask_layer(static_observed)
        result["scene_static_confidence"] = _encoded_layer(
            static_confidence,
            value_min=0.0,
            value_max=1.0,
        )
        result["scene_composite_height_agl"] = _encoded_layer(
            composite_height,
            value_min=0.0,
            value_max=max(0.0, composite_max),
        )
        result["scene_composite_observed"] = _encoded_mask_layer(composite_valid)
        result["scene_composite_source"] = _encoded_uint8_layer(
            composite_source,
            value_min=0.0,
            value_max=2.0,
        )
        result["scene_prior_meta"] = {
            "contract": "noesis.scene_prior.floorplan_composite",
            "contract_version": 1,
            "prior_id": revision.manifest.prior_id,
            "space_id": revision.manifest.space_id,
            "mode": binding.mode,
            "source_type": revision.manifest.source.source_type,
            "source_model": revision.manifest.source.model,
            "source_frame": world_frame,
            "source_frame_revision": world_frame_revision,
            "frame_transform_sha256": (
                frame_binding.target_from_source_sha256
                if frame_binding is not None
                else None
            ),
            "target_revision_id": (
                frame_binding.target_revision_id
                if frame_binding is not None
                else None
            ),
            "target_frame": "camera_local_ground_m",
            "raster_orientation": CAMERA_LOCAL_RASTER_ORIENTATION,
            "floor_y_m": float(floor_y_m),
            "composition_policy": "live_observed_wins_static_fills_live_unknown",
            "live_observed_cells": int(np.count_nonzero(live_valid)),
            "static_available_cells": int(np.count_nonzero(static_valid)),
            "static_fill_cells": int(np.count_nonzero(static_valid & ~live_valid)),
            "composite_observed_cells": int(np.count_nonzero(composite_valid)),
        }
        result["world_frame"] = world_frame
        result["world_frame_revision"] = world_frame_revision
        return result

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "contract": "noesis.scene_prior.health",
            "contract_version": 1,
            "site_id": self.catalog.site_id,
            "mode": "shadow",
            "camera_count": len(self._bindings),
            "revision_count": len(self.catalog.revisions),
            "loaded_revision_count": len(self._revisions),
            "cameras": {
                camera_id: {
                    "space_id": binding.space_id,
                    "prior_id": binding.prior_id,
                    "include_floorplan_layers": binding.include_floorplan_layers,
                    "world_frame": (
                        self._world_frame_metadata(camera_id)[0]
                    ),
                    "world_frame_revision": (
                        self._world_frame_metadata(camera_id)[1]
                    ),
                }
                for camera_id, binding in sorted(self._bindings.items())
            },
        }


__all__ = [
    "LoadedScenePrior",
    "ScenePriorError",
    "ScenePriorSet",
]
