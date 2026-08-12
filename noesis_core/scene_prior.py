"""Runtime-neutral loading, evaluation, and floorplan composition for scene priors."""

from __future__ import annotations

import base64
import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pydantic import ValidationError

from noesis_core.contracts.scene_prior import (
    ScenePriorArtifact,
    ScenePriorCameraBinding,
    ScenePriorCatalog,
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


@dataclass(frozen=True)
class LoadedScenePrior:
    manifest: ScenePriorRevision
    root: Path
    arrays: Mapping[str, np.ndarray]

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
    expected_bytes = shape[0] * shape[1] * np.dtype(np.float32).itemsize
    maximum_encoded_bytes = 4 * ((expected_bytes + 2) // 3)
    if len(encoded) > maximum_encoded_bytes:
        raise ScenePriorError(f"floorplan layer {name} encoded grid exceeds its shape bound")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ScenePriorError(f"floorplan layer {name} is not valid base64") from exc
    if len(raw) != expected_bytes:
        raise ScenePriorError(f"floorplan layer {name} byte length does not match its shape")
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()


def _encoded_layer(values: np.ndarray, *, value_min: float, value_max: float) -> dict[str, Any]:
    grid = np.asarray(values, dtype=np.float32)
    return {
        "grid_b64": base64.b64encode(grid.tobytes(order="C")).decode("ascii"),
        "grid_shape": [int(grid.shape[0]), int(grid.shape[1])],
        "value_min": float(value_min),
        "value_max": float(value_max),
    }


def _camera_ground_basis(
    extrinsics_col_major: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(extrinsics_col_major, dtype=np.float64)
    if values.size != 16 or not np.all(np.isfinite(values)):
        raise ScenePriorError("camera extrinsics must contain 16 finite values")
    world_to_camera = values.reshape((4, 4), order="F")
    if not np.allclose(world_to_camera[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ScenePriorError("camera extrinsics are not affine")
    rotation_wc = world_to_camera[:3, :3].T
    camera_world = -rotation_wc @ world_to_camera[:3, 3]
    forward = np.asarray([rotation_wc[0, 2], 0.0, rotation_wc[2, 2]], dtype=np.float64)
    forward_norm = float(np.linalg.norm(forward))
    if not math.isfinite(forward_norm) or forward_norm <= 1e-6:
        raise ScenePriorError("camera forward axis has no stable ground projection")
    forward /= forward_norm
    right = np.cross(np.asarray([0.0, 1.0, 0.0]), forward)
    right_norm = float(np.linalg.norm(right))
    if not math.isfinite(right_norm) or right_norm <= 1e-6:
        raise ScenePriorError("camera right axis has no stable ground projection")
    right /= right_norm
    if float(np.dot(right, rotation_wc[:, 0])) < 0.0:
        right *= -1.0
    return camera_world, right, forward


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
                revisions[prior_id] = LoadedScenePrior(
                    manifest=manifest,
                    root=manifest_root,
                    arrays=arrays,
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
        camera_world, right_world, forward_world = _camera_ground_basis(
            extrinsics_col_major,
        )
        floor_y_m = float(revision.manifest.derivation.floor_y_m)
        world_to_camera_local = np.asarray(
            [
                [
                    float(right_world[0]),
                    float(right_world[1]),
                    float(right_world[2]),
                    -float(np.dot(right_world, camera_world)),
                ],
                [0.0, 1.0, 0.0, -floor_y_m],
                [
                    float(forward_world[0]),
                    float(forward_world[1]),
                    float(forward_world[2]),
                    -float(np.dot(forward_world, camera_world)),
                ],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
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
            "source_coordinate_frame": "backend_world_m",
            "target_coordinate_frame": "camera_local_ground_m",
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
    ) -> dict[str, Any]:
        """Serve the verified static prior when the live floorplan cache misses."""

        binding = self.binding(camera_id)
        revision = self.revision_for_camera(camera_id)
        if binding is None or revision is None or not binding.include_floorplan_layers:
            return dict(payload)
        preview = revision.manifest.preview
        if preview is None:
            raise ScenePriorError(
                "scene-prior static-only floorplan requires camera preview geometry"
            )

        rows = int(preview.rows)
        columns = int(preview.columns)
        bounds = preview.bounds
        local_x = float(bounds.min_x) + (
            np.arange(columns, dtype=np.float64) + 0.5
        ) * ((float(bounds.max_x) - float(bounds.min_x)) / float(columns))
        local_z = float(bounds.max_z) - (
            np.arange(rows, dtype=np.float64) + 0.5
        ) * ((float(bounds.max_z) - float(bounds.min_z)) / float(rows))
        x_grid, z_grid = np.meshgrid(local_x, local_z)
        camera_world = np.asarray(preview.camera_position_world_m, dtype=np.float64)
        right_world = np.asarray(preview.camera_right_world_xz, dtype=np.float64)
        forward_world = np.asarray(preview.camera_forward_world_xz, dtype=np.float64)
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

        result = dict(payload)
        live_error = str(result.pop("error", "") or "").strip()
        result.update(
            {
                "camera_id": camera_id,
                "frame": preview.coordinate_frame,
                "units": preview.units,
                "bounds": bounds.model_dump(mode="json"),
                "scale_m_per_px": float(preview.resolution_m),
                "scene_static_height_agl": _encoded_layer(
                    static_height,
                    value_min=0.0,
                    value_max=max(0.0, static_max),
                ),
                "scene_static_observed": _encoded_layer(
                    static_observed,
                    value_min=0.0,
                    value_max=1.0,
                ),
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
                "scene_composite_observed": _encoded_layer(
                    static_observed,
                    value_min=0.0,
                    value_max=1.0,
                ),
                "scene_composite_source": _encoded_layer(
                    static_source,
                    value_min=0.0,
                    value_max=2.0,
                ),
                "scene_prior_meta": {
                    "contract": "noesis.scene_prior.floorplan_composite",
                    "contract_version": 1,
                    "status": "static_only",
                    "prior_id": revision.manifest.prior_id,
                    "space_id": revision.manifest.space_id,
                    "mode": binding.mode,
                    "source_type": revision.manifest.source.source_type,
                    "source_model": revision.manifest.source.model,
                    "source_frame": "backend_world_m",
                    "target_frame": preview.coordinate_frame,
                    "raster_orientation": (
                        "row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x"
                    ),
                    "floor_y_m": float(revision.manifest.derivation.floor_y_m),
                    "composition_policy": "static_only_live_unavailable",
                    "live_observed_cells": 0,
                    "static_available_cells": int(np.count_nonzero(static_valid)),
                    "static_fill_cells": int(np.count_nonzero(static_valid)),
                    "composite_observed_cells": int(np.count_nonzero(static_valid)),
                    "reason": "live_floorplan_unavailable",
                },
            }
        )
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
            extrinsics_col_major,
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
        result["scene_static_observed"] = _encoded_layer(
            static_observed,
            value_min=0.0,
            value_max=1.0,
        )
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
        result["scene_composite_observed"] = _encoded_layer(
            composite_valid.astype(np.float32),
            value_min=0.0,
            value_max=1.0,
        )
        result["scene_composite_source"] = _encoded_layer(
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
            "source_frame": "backend_world_m",
            "target_frame": "camera_local_ground_m",
            "raster_orientation": (
                "row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x"
            ),
            "floor_y_m": float(floor_y_m),
            "composition_policy": "live_observed_wins_static_fills_live_unknown",
            "live_observed_cells": int(np.count_nonzero(live_valid)),
            "static_available_cells": int(np.count_nonzero(static_valid)),
            "static_fill_cells": int(np.count_nonzero(static_valid & ~live_valid)),
            "composite_observed_cells": int(np.count_nonzero(composite_valid)),
        }
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
                }
                for camera_id, binding in sorted(self._bindings.items())
            },
        }


__all__ = [
    "LoadedScenePrior",
    "ScenePriorError",
    "ScenePriorSet",
]
