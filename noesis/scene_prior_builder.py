"""Build immutable, room-agnostic scene-prior revisions from aligned room scans."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import secrets
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

import numpy as np

from noesis.validation.authored_scene import AuthoredSceneGeometry, load_similarity
from noesis.virtual_twin.artifacts import write_points_glb
from noesis_core.coordinate_frames import (
    BACKEND_WORLD_FRAME_ID,
    CAMERA_LOCAL_RASTER_ORIENTATION,
    CoordinateFrameError,
    RevisionedFrame,
    camera_ground_frame_from_camera_to_world,
    revisioned_frame_sha256,
    revisioned_transform_sha256,
)
from noesis_core.contracts.base import ArtifactFingerprint
from noesis_core.contracts.scene_prior import (
    ScenePriorArtifact,
    ScenePriorBounds,
    ScenePriorCameraMapLock,
    ScenePriorCameraBinding,
    ScenePriorCatalog,
    ScenePriorCatalogEntry,
    ScenePriorDerivation,
    ScenePriorFloorPlane,
    ScenePriorFrameBinding,
    ScenePriorFrameRef,
    ScenePriorGrid,
    ScenePriorPreview,
    ScenePriorQuality,
    ScenePriorRevision,
    ScenePriorSemanticBinding,
    ScenePriorSource,
    ScenePriorMetricFrame,
    ScenePriorWorldToScenePresentation,
    metric_frame_revision_sha256,
    scene_revision_sha256,
    world_to_scene_presentation_sha256,
)
from noesis_core.scene_files import (
    SceneFileError,
    ensure_scene_directory,
    load_strict_json,
    materialize_scene_tree,
    read_scene_file,
)


MAX_BUNDLE_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_REFERENCE_BYTES = 512 * 1024
MAX_ALIGNMENT_BYTES = 4 * 1024 * 1024
MAX_CAMERA_CALIBRATION_BYTES = 4 * 1024 * 1024
MAX_ALIGNED_POINTS_BYTES = 64 * 1024 * 1024
MAX_ROOM_GROUP_MAP_BYTES = 512 * 1024
MAX_WORLD_TO_SCENE_BYTES = 4 * 1024 * 1024
MAX_SHA256SUMS_BYTES = 512 * 1024
MAX_TARGET_REVISION_METADATA_BYTES = 4 * 1024 * 1024
MAX_CAMERA_MAP_LOCK_BYTES = 512 * 1024
MAX_PREVIEW_DIMENSION = 2_048
MAX_REVIEW_POINTS = 250_000
_ACCEPTED_REGISTRATION_DISPOSITIONS = frozenset(
    {"validated_cross_session_registration"}
)
_ACCEPTED_REGISTRATION_SCHEMAS = frozenset(
    {"noesis.pcf.connector_multianchor_pose_graph.v1"}
)


class ScenePriorBuildError(RuntimeError):
    """Raised when source evidence cannot produce a valid immutable revision."""


@dataclass(frozen=True)
class ScenePriorBuildConfig:
    source_bundle: Path
    site_id: str
    space_id: str
    semantic_rooms: tuple[str, ...]
    authored_scene: Path
    room_group_map: Path
    world_to_scene: Path
    output_root: Path
    camera_map_lock: Path | None = None
    camera_pose_anchor: Path | None = None
    camera_ids: tuple[str, ...] = ()
    grid_resolution_m: float = 0.05
    floor_support_band_m: float = 0.12
    obstacle_min_height_m: float = 0.18
    obstacle_max_height_m: float = 2.20
    obstacle_min_support: int = 3
    max_source_height_m: float = 3.20
    include_floorplan_layers: bool = True
    # Optional v2 publication metadata.  Omitting these fields preserves the
    # exact v1 catalog contract for existing shadow-only room priors.
    metric_frame: ScenePriorMetricFrame | Mapping[str, Any] | None = None
    world_presentation: ScenePriorWorldToScenePresentation | Mapping[str, Any] | None = None
    # v2 is admitted only from one exact binding and the producer's passed
    # registration report.  A binding is deliberately separate from the
    # artifact revision created by this builder.
    accepted_frame_binding: ScenePriorFrameBinding | Mapping[str, Any] | Path | None = None
    registration_acceptance_report: Mapping[str, Any] | Path | None = None

    def validate(self) -> "ScenePriorBuildConfig":
        identifiers = {
            "site_id": self.site_id,
            "space_id": self.space_id,
            **{
                f"camera_id[{index}]": value
                for index, value in enumerate(self.camera_ids)
            },
        }
        for label, value in identifiers.items():
            text = str(value).strip()
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", text) is None:
                raise ScenePriorBuildError(
                    f"{label} is not a valid portable identifier"
                )
        if not self.semantic_rooms or any(
            not str(value).strip() for value in self.semantic_rooms
        ):
            raise ScenePriorBuildError(
                "at least one non-empty semantic room label is required"
            )
        if len(set(self.semantic_rooms)) != len(self.semantic_rooms):
            raise ScenePriorBuildError("semantic room labels must be unique")
        if len(set(self.camera_ids)) != len(self.camera_ids):
            raise ScenePriorBuildError("camera bindings must be unique")
        numeric = (
            self.grid_resolution_m,
            self.floor_support_band_m,
            self.obstacle_min_height_m,
            self.obstacle_max_height_m,
            self.max_source_height_m,
        )
        if not all(
            math.isfinite(float(value)) and float(value) > 0.0 for value in numeric
        ):
            raise ScenePriorBuildError(
                "scene-prior thresholds must be finite and positive"
            )
        if not 0.01 <= float(self.grid_resolution_m) <= 0.50:
            raise ScenePriorBuildError("grid_resolution_m must be in [0.01, 0.50]")
        if self.obstacle_max_height_m <= self.obstacle_min_height_m:
            raise ScenePriorBuildError("obstacle maximum must exceed obstacle minimum")
        if self.max_source_height_m < self.obstacle_max_height_m:
            raise ScenePriorBuildError(
                "maximum source height must cover the obstacle range"
            )
        if int(self.obstacle_min_support) < 1:
            raise ScenePriorBuildError("obstacle_min_support must be positive")
        if self.world_presentation is not None and self.metric_frame is None:
            raise ScenePriorBuildError(
                "world_presentation requires an accepted metric_frame"
            )
        if self.registration_acceptance_report is not None and self.accepted_frame_binding is None:
            raise ScenePriorBuildError(
                "registration_acceptance_report requires accepted_frame_binding"
            )
        if self.accepted_frame_binding is not None and self.metric_frame is None:
            # The metric frame is carried by the imported v2 binding.  The
            # legacy optional knobs cannot be used to relabel that target.
            if self.world_presentation is not None:
                raise ScenePriorBuildError(
                    "world_presentation requires accepted_frame_binding and metric_frame"
                )
        if (self.metric_frame is not None or self.world_presentation is not None) and self.accepted_frame_binding is None:
            raise ScenePriorBuildError(
                "v2 metric/presentation metadata requires an accepted_frame_binding and registration report"
            )
        return self


@dataclass(frozen=True)
class ScenePriorBuildResult:
    prior_id: str
    revision_dir: Path
    manifest_path: Path
    catalog_path: Path
    manifest_sha256: str
    quality: ScenePriorQuality

    def to_dict(self) -> dict[str, Any]:
        return {
            "prior_id": self.prior_id,
            "revision_dir": str(self.revision_dir),
            "manifest_path": str(self.manifest_path),
            "catalog_path": str(self.catalog_path),
            "manifest_sha256": self.manifest_sha256,
            "quality": self.quality.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class _CameraPreviewFrame:
    camera_id: str
    camera_position_world_m: tuple[float, float, float]
    camera_right_world_xz: tuple[float, float]
    camera_forward_world_xz: tuple[float, float]
    camera_calibration: ArtifactFingerprint
    target_revision_metadata: ArtifactFingerprint
    target_revision_id: str
    camera_map_lock: ScenePriorCameraMapLock | None
    camera_pose_anchor: ArtifactFingerprint | None
    target_from_source_col_major: tuple[float, ...]
    source_floor_normal: tuple[float, float, float]
    source_floor_offset_m: float
    target_floor_y_m: float


def _canonical_json(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _compact_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _physical_camera_calibration_sha256(extrinsics: Mapping[str, Any]) -> str:
    """Hash only calibrated E matrices, matching CalibrationManager v2."""

    cameras = extrinsics.get("cameras") if isinstance(extrinsics, Mapping) else {}
    physical: dict[str, dict[str, list[float]]] = {}
    if isinstance(cameras, Mapping):
        for camera_id, entry in sorted(cameras.items(), key=lambda item: str(item[0])):
            if not isinstance(entry, Mapping):
                continue
            values = entry.get("E")
            if not isinstance(values, list) or len(values) != 16:
                continue
            try:
                physical[str(camera_id)] = {"E": [float(value) for value in values]}
            except (TypeError, ValueError):
                continue
    return _sha256(_compact_json({"cameras": physical}))


def _alignment_matrix_values(transform: Mapping[str, Any]) -> list[float]:
    values = transform.get("matrix")
    if isinstance(values, list) and len(values) == 16:
        return [float(value) for value in values]
    values = transform.get("world_from_source_row_major")
    if isinstance(values, list) and len(values) == 4 and all(
        isinstance(row, list) and len(row) == 4 for row in values
    ):
        return [float(value) for row in values for value in row]
    raise ScenePriorBuildError(
        "alignment transform must declare a 16-value physical matrix"
    )


def _physical_world_alignment_sha256(
    transform: Mapping[str, Any],
    target_metadata: Mapping[str, Any],
) -> str:
    """Hash source alignment physics, excluding authored presentation fields."""

    try:
        floor_y = float(
            target_metadata.get(
                "calibrated_floor_y",
                target_metadata["floor_alignment"]["target_floor_y"],
            )
        )
        scale = float(transform.get("scale", 1.0))
        matrix = _alignment_matrix_values(transform)
    except (KeyError, TypeError, ValueError) as exc:
        raise ScenePriorBuildError("alignment transform physical identity is malformed") from exc
    if not all(math.isfinite(value) for value in (*matrix, floor_y, scale)):
        raise ScenePriorBuildError("alignment transform physical identity is non-finite")
    return _sha256(
        _compact_json(
            {
                "matrix": matrix,
                "floor_y": floor_y,
                "units": {"s_obj_to_m": scale},
            }
        )
    )


def _source_physical_frame(inputs: Mapping[str, Any]) -> tuple[RevisionedFrame, str, str]:
    camera_sha256 = _physical_camera_calibration_sha256(inputs["camera_calibration"])
    alignment_sha256 = _physical_world_alignment_sha256(
        inputs["transform"], inputs["target_revision_metadata"]
    )
    revision = revisioned_frame_sha256(
        BACKEND_WORLD_FRAME_ID,
        artifact_sha256s=(camera_sha256, alignment_sha256),
    )
    return (
        RevisionedFrame(frame_id=BACKEND_WORLD_FRAME_ID, revision=revision),
        camera_sha256,
        alignment_sha256,
    )


def _col_major_matrix(values: Sequence[float]) -> np.ndarray:
    try:
        matrix = np.asarray(tuple(float(value) for value in values), dtype=np.float64).reshape(
            (4, 4), order="F"
        )
    except (TypeError, ValueError) as exc:
        raise ScenePriorBuildError("frame transform matrix must contain 16 values") from exc
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8
    ):
        raise ScenePriorBuildError("frame transform matrix must be finite affine")
    try:
        determinant = float(np.linalg.det(matrix[:3, :3]))
        np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ScenePriorBuildError("frame transform matrix must be invertible") from exc
    if not math.isfinite(determinant) or determinant <= 1e-12:
        raise ScenePriorBuildError("frame transform matrix must preserve orientation")
    return matrix


def _apply_col_major_transform(points: np.ndarray, values: Sequence[float]) -> np.ndarray:
    matrix = _col_major_matrix(values)
    source = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate(
        (source, np.ones((source.shape[0], 1), dtype=np.float64)), axis=1
    )
    transformed = (matrix @ homogeneous.T).T
    if not np.allclose(transformed[:, 3], 1.0, atol=1e-8):
        raise ScenePriorBuildError("frame transform produced non-unit homogeneous coordinates")
    return transformed[:, :3]


def _reframe_camera_preview(
    preview: _CameraPreviewFrame,
    target_from_source_col_major: Sequence[float],
    *,
    source_floor_normal: Sequence[float],
    source_floor_offset_m: float,
    target_floor_y_m: float,
) -> _CameraPreviewFrame:
    """Move the camera pose from the metadata target into the accepted target."""

    old = _col_major_matrix(preview.target_from_source_col_major)
    new = _col_major_matrix(target_from_source_col_major)
    try:
        delta = new @ np.linalg.inv(old)
    except np.linalg.LinAlgError as exc:
        raise ScenePriorBuildError("camera preview frame transform is singular") from exc
    position = delta[:3, :3] @ np.asarray(preview.camera_position_world_m) + delta[:3, 3]
    right = delta[:3, :3] @ np.asarray((preview.camera_right_world_xz[0], 0.0, preview.camera_right_world_xz[1]))
    forward = delta[:3, :3] @ np.asarray((preview.camera_forward_world_xz[0], 0.0, preview.camera_forward_world_xz[1]))
    right_xz = right[[0, 2]]
    forward_xz = forward[[0, 2]]
    right_norm = float(np.linalg.norm(right_xz))
    forward_norm = float(np.linalg.norm(forward_xz))
    if right_norm <= 1e-8 or forward_norm <= 1e-8:
        raise ScenePriorBuildError("accepted frame transform collapses camera ground axes")
    right_xz /= right_norm
    forward_xz /= forward_norm
    if abs(float(np.dot(right_xz, forward_xz))) > 1e-4:
        raise ScenePriorBuildError("accepted frame transform distorts camera ground axes")
    return _CameraPreviewFrame(
        camera_id=preview.camera_id,
        camera_position_world_m=tuple(float(value) for value in position),
        camera_right_world_xz=tuple(float(value) for value in right_xz),
        camera_forward_world_xz=tuple(float(value) for value in forward_xz),
        camera_calibration=preview.camera_calibration,
        target_revision_metadata=preview.target_revision_metadata,
        target_revision_id=preview.target_revision_id,
        camera_map_lock=preview.camera_map_lock,
        camera_pose_anchor=preview.camera_pose_anchor,
        target_from_source_col_major=tuple(float(value) for value in target_from_source_col_major),
        source_floor_normal=tuple(float(value) for value in source_floor_normal),
        source_floor_offset_m=float(source_floor_offset_m),
        target_floor_y_m=float(target_floor_y_m),
    )


def _timestamp_us(value: Any, *, label: str) -> int:
    text = str(value or "").strip()
    if not text:
        raise ScenePriorBuildError(f"{label} is required")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ScenePriorBuildError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ScenePriorBuildError(f"{label} must include a timezone")
    result = int(parsed.timestamp() * 1_000_000)
    if result <= 0:
        raise ScenePriorBuildError(f"{label} must be after the Unix epoch")
    return result


def _timestamp_slug(timestamp_us: int) -> str:
    value = datetime.fromtimestamp(timestamp_us / 1_000_000, tz=timezone.utc)
    return value.strftime("%Y%m%dT%H%M%SZ")


def _read_json_file(
    path: Path, *, label: str, max_bytes: int
) -> tuple[bytes, Mapping[str, Any]]:
    try:
        verified = read_scene_file(path, label=label, max_bytes=max_bytes)
        payload = load_strict_json(verified.data, label=label)
    except SceneFileError as exc:
        raise ScenePriorBuildError(str(exc)) from exc
    if not isinstance(payload, Mapping):
        raise ScenePriorBuildError(f"{label} must contain a JSON object")
    return verified.data, payload


def _parse_sha256sums(payload: bytes) -> dict[str, str]:
    try:
        lines = payload.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise ScenePriorBuildError("bundle SHA256SUMS is not valid UTF-8") from exc
    result: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        fields = line.split("  ", 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ScenePriorBuildError(
                f"bundle SHA256SUMS line {line_number} is invalid"
            )
        digest, relative = fields
        if any(char not in "0123456789abcdef" for char in digest):
            raise ScenePriorBuildError(
                f"bundle SHA256SUMS line {line_number} has an invalid hash"
            )
        if relative in result:
            raise ScenePriorBuildError(f"bundle SHA256SUMS repeats {relative}")
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
            raise ScenePriorBuildError(
                f"bundle SHA256SUMS has unsafe path {relative!r}"
            )
        result[relative] = digest
    return result


def _bundle_relative_path(value: Any, *, label: str) -> str:
    text = str(value or "")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ScenePriorBuildError(f"{label} path is invalid")
    return text


def _verified_bundle_file(
    bundle_root: Path,
    relative_path: str,
    *,
    label: str,
    max_bytes: int,
    checksums: Mapping[str, str],
    manifest_files: Mapping[str, Mapping[str, Any]],
    require_manifest_entry: bool = True,
) -> bytes:
    expected = checksums.get(relative_path)
    if expected is None:
        raise ScenePriorBuildError(f"{label} is absent from bundle SHA256SUMS")
    manifest_entry = manifest_files.get(relative_path)
    if require_manifest_entry and manifest_entry is None:
        raise ScenePriorBuildError(f"{label} is absent from the bundle manifest")
    if manifest_entry is not None:
        if str(manifest_entry.get("sha256") or "") != expected:
            raise ScenePriorBuildError(
                f"{label} hash disagrees between bundle inventories"
            )
        try:
            expected_size = int(manifest_entry.get("size_bytes"))
        except (TypeError, ValueError) as exc:
            raise ScenePriorBuildError(f"{label} has an invalid manifest size") from exc
    else:
        expected_size = None
    try:
        verified = read_scene_file(
            bundle_root / relative_path,
            label=label,
            max_bytes=max_bytes,
            expected_sha256=expected,
            expected_size=expected_size,
        )
    except SceneFileError as exc:
        raise ScenePriorBuildError(str(exc)) from exc
    return verified.data


def _bundle_inputs(bundle_root: Path) -> dict[str, Any]:
    try:
        sums_file = read_scene_file(
            bundle_root / "SHA256SUMS",
            label="bundle SHA256SUMS",
            max_bytes=MAX_SHA256SUMS_BYTES,
        )
    except SceneFileError as exc:
        raise ScenePriorBuildError(str(exc)) from exc
    checksums = _parse_sha256sums(sums_file.data)
    manifest_bytes = _verified_bundle_file(
        bundle_root,
        "bundle_manifest.json",
        label="bundle manifest",
        max_bytes=MAX_BUNDLE_MANIFEST_BYTES,
        checksums=checksums,
        manifest_files={},
        require_manifest_entry=False,
    )
    manifest = load_strict_json(manifest_bytes, label="bundle manifest")
    if not isinstance(manifest, Mapping):
        raise ScenePriorBuildError("bundle manifest must contain a JSON object")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ScenePriorBuildError("bundle manifest has no file inventory")
    manifest_files: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ScenePriorBuildError(
                "bundle manifest contains a non-object file entry"
            )
        relative = str(entry.get("path") or "")
        if relative in manifest_files:
            raise ScenePriorBuildError(f"bundle manifest repeats {relative!r}")
        manifest_files[relative] = entry

    reference_bytes = _verified_bundle_file(
        bundle_root,
        "reference.json",
        label="room-scan reference",
        max_bytes=MAX_REFERENCE_BYTES,
        checksums=checksums,
        manifest_files=manifest_files,
    )
    reference = load_strict_json(reference_bytes, label="room-scan reference")
    if not isinstance(reference, Mapping):
        raise ScenePriorBuildError("room-scan reference must contain a JSON object")
    if reference.get("schema") != "noesis.reference.room_scan_bundle.v1":
        raise ScenePriorBuildError("room-scan reference schema is unsupported")
    if manifest.get("schema") != "noesis.reference.room_scan_bundle_manifest.v1":
        raise ScenePriorBuildError("room-scan bundle manifest schema is unsupported")
    bundle_id = str(reference.get("bundle_id") or "")
    if not bundle_id or bundle_id != str(manifest.get("bundle_id") or ""):
        raise ScenePriorBuildError("room-scan bundle identities do not match")
    quality = reference.get("quality")
    if not isinstance(quality, Mapping) or quality.get("passed") is not True:
        raise ScenePriorBuildError("room-scan source quality was not admitted")
    coordinate = reference.get("coordinate_contract")
    if (
        not isinstance(coordinate, Mapping)
        or coordinate.get("target_frame") != "backend_world_m_stream_points"
    ):
        raise ScenePriorBuildError(
            "room-scan source is not aligned to backend world meters"
        )
    review = reference.get("review_assets")
    if not isinstance(review, Mapping):
        raise ScenePriorBuildError("room-scan reference has no review assets")
    aligned_path = _bundle_relative_path(
        review.get("aligned_rgb_glb"),
        label="aligned room points",
    )
    alignment_path = _bundle_relative_path(
        review.get("alignment_report"),
        label="alignment report",
    )
    transform_path = _bundle_relative_path(
        coordinate.get("transform"),
        label="alignment transform",
    )
    aligned_bytes = _verified_bundle_file(
        bundle_root,
        aligned_path,
        label="aligned room points",
        max_bytes=MAX_ALIGNED_POINTS_BYTES,
        checksums=checksums,
        manifest_files=manifest_files,
    )
    alignment_bytes = _verified_bundle_file(
        bundle_root,
        alignment_path,
        label="alignment report",
        max_bytes=MAX_ALIGNMENT_BYTES,
        checksums=checksums,
        manifest_files=manifest_files,
    )
    alignment = load_strict_json(alignment_bytes, label="alignment report")
    if (
        not isinstance(alignment, Mapping)
        or alignment.get("status") != "passed"
        or not isinstance(alignment.get("quality_gate"), Mapping)
        or alignment["quality_gate"].get("passed") is not True
    ):
        raise ScenePriorBuildError("room-scan alignment quality gate did not pass")
    transform_bytes = _verified_bundle_file(
        bundle_root,
        transform_path,
        label="alignment transform",
        max_bytes=MAX_ALIGNMENT_BYTES,
        checksums=checksums,
        manifest_files=manifest_files,
    )
    transform = load_strict_json(transform_bytes, label="alignment transform")
    if (
        not isinstance(transform, Mapping)
        or transform.get("target_coordinate_frame") != "backend_world_m_stream_points"
        or float(transform.get("scale", math.nan)) != 1.0
    ):
        raise ScenePriorBuildError(
            "alignment transform does not target metric backend world"
        )
    noesis_reference = reference.get("noesis_reference")
    if not isinstance(noesis_reference, Mapping):
        raise ScenePriorBuildError("room-scan reference has no Noesis camera reference")
    camera_id = str(noesis_reference.get("camera_id") or "")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", camera_id) is None:
        raise ScenePriorBuildError("room-scan reference camera ID is invalid")
    camera_calibration_path = _bundle_relative_path(
        noesis_reference.get("camera_calibration"),
        label="reference camera calibration",
    )
    target_revision_root = _bundle_relative_path(
        noesis_reference.get("revision"),
        label="reference target revision",
    )
    target_metadata_path = (
        PurePosixPath(target_revision_root) / "room_points_meta.json"
    ).as_posix()
    camera_calibration_bytes = _verified_bundle_file(
        bundle_root,
        camera_calibration_path,
        label="reference camera calibration",
        max_bytes=MAX_CAMERA_CALIBRATION_BYTES,
        checksums=checksums,
        manifest_files=manifest_files,
    )
    camera_calibration = load_strict_json(
        camera_calibration_bytes,
        label="reference camera calibration",
    )
    if not isinstance(camera_calibration, Mapping):
        raise ScenePriorBuildError(
            "reference camera calibration must contain a JSON object"
        )
    calibration_rows = camera_calibration.get("cameras", camera_calibration)
    calibration_row = (
        calibration_rows.get(camera_id)
        if isinstance(calibration_rows, Mapping)
        else None
    )
    if not isinstance(calibration_row, Mapping) or not isinstance(
        calibration_row.get("E"), list
    ):
        raise ScenePriorBuildError(
            f"reference camera {camera_id} has no calibrated E matrix"
        )
    target_metadata_bytes = _verified_bundle_file(
        bundle_root,
        target_metadata_path,
        label="reference target revision metadata",
        max_bytes=MAX_TARGET_REVISION_METADATA_BYTES,
        checksums=checksums,
        manifest_files=manifest_files,
    )
    target_metadata = load_strict_json(
        target_metadata_bytes,
        label="reference target revision metadata",
    )
    if not isinstance(target_metadata, Mapping):
        raise ScenePriorBuildError(
            "reference target revision metadata must contain a JSON object"
        )
    if target_metadata.get("camera") != camera_id:
        raise ScenePriorBuildError(
            "reference target revision belongs to another camera"
        )
    if target_metadata.get("coordinate_frame") != "backend_world_m_stream_points":
        raise ScenePriorBuildError(
            "reference target revision is not in backend world meters"
        )
    floor_alignment = target_metadata.get("floor_alignment")
    if not isinstance(floor_alignment, Mapping) or not isinstance(
        floor_alignment.get("world_correction_col_major"),
        list,
    ):
        raise ScenePriorBuildError(
            "reference target revision has no floor-world correction"
        )
    return {
        "bundle_id": bundle_id,
        "manifest": manifest,
        "manifest_bytes": manifest_bytes,
        "reference": reference,
        "reference_bytes": reference_bytes,
        "aligned_bytes": aligned_bytes,
        "alignment": alignment,
        "alignment_bytes": alignment_bytes,
        "transform": transform,
        "alignment_schema": str(alignment.get("schema") or "unknown"),
        "transform_bytes": transform_bytes,
        "reference_camera_id": camera_id,
        "camera_calibration_row": calibration_row,
        "camera_calibration": camera_calibration,
        "camera_calibration_bytes": camera_calibration_bytes,
        "target_revision_metadata": target_metadata,
        "target_revision_metadata_bytes": target_metadata_bytes,
    }


@dataclass(frozen=True)
class _CameraMapLockInput:
    evidence: ArtifactFingerprint
    yaw_correction_deg: float


@dataclass(frozen=True)
class _CameraPoseAnchorInput:
    evidence: ArtifactFingerprint
    camera_to_target: np.ndarray
    target_floor_y_m: float


def _load_camera_map_lock(
    path: Path | None,
    inputs: Mapping[str, Any],
) -> _CameraMapLockInput | None:
    if path is None:
        return None
    try:
        source = read_scene_file(
            path,
            label="scene-prior camera map lock",
            max_bytes=MAX_CAMERA_MAP_LOCK_BYTES,
        )
    except SceneFileError as exc:
        raise ScenePriorBuildError(str(exc)) from exc
    payload = load_strict_json(source.data, label="scene-prior camera map lock")
    expected_fields = {
        "contract",
        "contract_version",
        "camera_id",
        "source_camera_calibration_sha256",
        "target_revision_id",
        "target_revision_metadata_sha256",
        "yaw_correction_deg",
        "rotation_pivot",
        "evidence",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ScenePriorBuildError("camera map-lock fields do not match its contract")
    if (
        payload.get("contract") != "noesis.scene_prior.camera_map_lock.input"
        or payload.get("contract_version") != 1
        or payload.get("rotation_pivot")
        != "camera_optical_center_target_world_m"
    ):
        raise ScenePriorBuildError("camera map-lock contract is unsupported")
    camera_id = str(inputs["reference_camera_id"])
    calibration_sha256 = _sha256(inputs["camera_calibration_bytes"])
    target_metadata = inputs["target_revision_metadata"]
    target_metadata_sha256 = _sha256(inputs["target_revision_metadata_bytes"])
    if str(payload.get("camera_id") or "") != camera_id:
        raise ScenePriorBuildError("camera map lock belongs to another camera")
    if str(payload.get("source_camera_calibration_sha256") or "") != calibration_sha256:
        raise ScenePriorBuildError("camera map-lock calibration revision mismatch")
    if str(payload.get("target_revision_id") or "") != str(
        target_metadata.get("revision_id") or ""
    ):
        raise ScenePriorBuildError("camera map-lock target revision mismatch")
    if str(payload.get("target_revision_metadata_sha256") or "") != target_metadata_sha256:
        raise ScenePriorBuildError("camera map-lock target metadata mismatch")
    if not isinstance(payload.get("evidence"), Mapping):
        raise ScenePriorBuildError("camera map lock has no evidence")
    try:
        yaw_correction_deg = float(payload["yaw_correction_deg"])
    except (TypeError, ValueError) as exc:
        raise ScenePriorBuildError("camera map-lock yaw is invalid") from exc
    if (
        not math.isfinite(yaw_correction_deg)
        or abs(yaw_correction_deg) > 45.0
        or abs(yaw_correction_deg) < 0.01
    ):
        raise ScenePriorBuildError("camera map-lock yaw must be in [-45, 45] degrees")
    return _CameraMapLockInput(
        evidence=ArtifactFingerprint(
            role="camera_to_pcf_map_lock",
            sha256=source.sha256,
            version="noesis.scene_prior.camera_map_lock.input.v1",
            producer="noesis.static_pcf_video_fit",
        ),
        yaw_correction_deg=yaw_correction_deg,
    )


def _load_camera_pose_anchor(
    path: Path | None,
    inputs: Mapping[str, Any],
) -> _CameraPoseAnchorInput | None:
    if path is None:
        return None
    try:
        source = read_scene_file(
            path,
            label="PCF static-camera pose anchor",
            max_bytes=MAX_CAMERA_MAP_LOCK_BYTES,
        )
    except SceneFileError as exc:
        raise ScenePriorBuildError(str(exc)) from exc
    payload = load_strict_json(source.data, label="PCF static-camera pose anchor")
    if not isinstance(payload, Mapping):
        raise ScenePriorBuildError("PCF camera pose anchor is not an object")
    camera_id = str(inputs["reference_camera_id"])
    if (
        payload.get("schema") != "noesis.pcf.static_camera_anchor.v2"
        or payload.get("status") != "passed_review_anchor"
        or payload.get("accepted_for_canonical_use") is not True
        or payload.get("camera_id") != camera_id
        or payload.get("coordinate_frame") != "pcf_assembly_metric_world_m"
        or payload.get("pose_convention")
        != "opencv_cam2world_x_right_y_down_z_forward"
    ):
        raise ScenePriorBuildError("PCF camera pose anchor is not admitted")
    constraints = payload.get("constraints")
    estimate = payload.get("estimate")
    evidence = payload.get("evidence")
    floor = evidence.get("pcf_floor") if isinstance(evidence, Mapping) else None
    anchor_inputs = payload.get("inputs")
    if not all(
        isinstance(value, Mapping)
        for value in (constraints, estimate, evidence, floor, anchor_inputs)
    ):
        raise ScenePriorBuildError("PCF camera pose anchor evidence is incomplete")
    assert isinstance(constraints, Mapping)
    assert isinstance(estimate, Mapping)
    assert isinstance(evidence, Mapping)
    assert isinstance(floor, Mapping)
    assert isinstance(anchor_inputs, Mapping)
    if (
        constraints.get("metric_scale_fixed") is not True
        or constraints.get("gravity_fixed") is not True
        or constraints.get("full_pnp_translation_used") is not True
        or constraints.get("full_pnp_rotation_used") is not True
        or constraints.get("calibrated_pitch_roll_preserved") is not False
        or floor.get("status") != "passed"
        or evidence.get("early_view_supported") is not True
        or evidence.get("late_view_supported") is not True
        or int(evidence.get("admitted_view_count") or 0) < 8
    ):
        raise ScenePriorBuildError("PCF camera pose anchor gates are incomplete")
    calibration_sha256 = _sha256(inputs["camera_calibration_bytes"])
    if anchor_inputs.get("camera_calibration_sha256") != calibration_sha256:
        raise ScenePriorBuildError("PCF camera pose anchor calibration mismatch")
    target_metadata_sha256 = _sha256(inputs["target_revision_metadata_bytes"])
    if anchor_inputs.get("static_metadata_sha256") != target_metadata_sha256:
        raise ScenePriorBuildError("PCF camera pose anchor revision mismatch")
    try:
        camera_to_target = np.asarray(
            payload["camera_to_assembly_row_major"], dtype=np.float64
        )
        target_floor_y_m = float(constraints["floor_y_m"])
        translation_uncertainty_m = float(
            estimate["translation_uncertainty_p80_m"]
        )
        rotation_uncertainty_deg = float(
            estimate["rotation_uncertainty_p80_deg"]
        )
        floor_tilt_deg = float(floor["tilt_from_pcf_gravity_deg"])
        floor_residual_p90_m = float(floor["residual_p90_m"])
        camera_height_m = float(floor["camera_height_over_fitted_floor_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScenePriorBuildError("PCF camera pose anchor metrics are malformed") from exc
    if (
        camera_to_target.shape != (4, 4)
        or not np.isfinite(camera_to_target).all()
        or not np.allclose(camera_to_target[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
        or not np.allclose(
            camera_to_target[:3, :3].T @ camera_to_target[:3, :3],
            np.eye(3),
            atol=1e-5,
        )
        or not math.isclose(
            float(np.linalg.det(camera_to_target[:3, :3])),
            1.0,
            abs_tol=1e-5,
        )
        or not all(
            math.isfinite(value)
            for value in (
                target_floor_y_m,
                translation_uncertainty_m,
                rotation_uncertainty_deg,
                floor_tilt_deg,
                floor_residual_p90_m,
                camera_height_m,
            )
        )
        or abs(target_floor_y_m) > 0.05
        or translation_uncertainty_m > 0.25
        or rotation_uncertainty_deg > 5.0
        or floor_tilt_deg > 2.0
        or floor_residual_p90_m > 0.06
        or not 0.40 <= camera_height_m <= 3.50
    ):
        raise ScenePriorBuildError("PCF camera pose anchor failed geometry gates")
    return _CameraPoseAnchorInput(
        evidence=ArtifactFingerprint(
            role="pcf_static_camera_pose_anchor",
            sha256=source.sha256,
            version="noesis.pcf.static_camera_anchor.v2",
            producer="noesis.localize_pcf_static_camera",
        ),
        camera_to_target=camera_to_target,
        target_floor_y_m=target_floor_y_m,
    )


def _camera_preview_frame(
    inputs: Mapping[str, Any],
    *,
    camera_map_lock: _CameraMapLockInput | None = None,
    camera_pose_anchor: _CameraPoseAnchorInput | None = None,
) -> _CameraPreviewFrame:
    if camera_map_lock is not None and camera_pose_anchor is not None:
        raise ScenePriorBuildError(
            "camera yaw map lock and full PCF pose anchor are mutually exclusive"
        )
    calibration = inputs["camera_calibration_row"]
    target_metadata = inputs["target_revision_metadata"]
    try:
        camera_from_backend = np.asarray(calibration["E"], dtype=np.float64).reshape(
            (4, 4),
            order="F",
        )
        if camera_pose_anchor is None:
            world_correction = np.asarray(
                target_metadata["floor_alignment"]["world_correction_col_major"],
                dtype=np.float64,
            ).reshape((4, 4), order="F")
        else:
            world_correction = (
                np.asarray(camera_pose_anchor.camera_to_target, dtype=np.float64)
                @ camera_from_backend
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ScenePriorBuildError("reference camera transform is malformed") from exc
    if (
        not np.isfinite(camera_from_backend).all()
        or not np.isfinite(world_correction).all()
    ):
        raise ScenePriorBuildError("reference camera transform must be finite")
    try:
        camera_to_world = world_correction @ np.linalg.inv(camera_from_backend)
    except np.linalg.LinAlgError as exc:
        raise ScenePriorBuildError("reference camera transform is singular") from exc
    base_camera_frame = None
    map_lock_contract = None
    if camera_map_lock is not None:
        try:
            base_camera_frame = camera_ground_frame_from_camera_to_world(
                camera_to_world
            )
        except CoordinateFrameError as exc:
            raise ScenePriorBuildError(str(exc)) from exc
        pivot = np.asarray(base_camera_frame.camera_world_m, dtype=np.float64)
        radians = math.radians(float(camera_map_lock.yaw_correction_deg))
        cosine = math.cos(radians)
        sine = math.sin(radians)
        rotation = np.asarray(
            [
                [cosine, 0.0, sine, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-sine, 0.0, cosine, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        target_residual = rotation.copy()
        target_residual[:3, 3] = pivot - (rotation[:3, :3] @ pivot)
        world_correction = target_residual @ world_correction
        camera_to_world = world_correction @ np.linalg.inv(camera_from_backend)
    try:
        camera_frame = camera_ground_frame_from_camera_to_world(camera_to_world)
    except CoordinateFrameError as exc:
        raise ScenePriorBuildError(str(exc)) from exc
    forward = camera_frame.camera_forward_world[[0, 2]]
    right = camera_frame.camera_right_world[[0, 2]]
    position = camera_frame.camera_world_m
    camera_id = str(inputs["reference_camera_id"])
    if camera_map_lock is not None and base_camera_frame is not None:
        base_forward = base_camera_frame.camera_forward_world[[0, 2]]
        map_lock_contract = ScenePriorCameraMapLock(
            contract="noesis.scene_prior.camera_map_lock",
            contract_version=1,
            evidence=camera_map_lock.evidence,
            camera_id=camera_id,
            yaw_correction_deg=float(camera_map_lock.yaw_correction_deg),
            rotation_pivot="camera_optical_center_target_world_m",
            pivot_world_m=tuple(float(value) for value in position),
            base_camera_forward_world_xz=tuple(float(value) for value in base_forward),
            corrected_camera_forward_world_xz=tuple(float(value) for value in forward),
        )
    floor_alignment = target_metadata.get("floor_alignment")
    if not isinstance(floor_alignment, Mapping):
        raise ScenePriorBuildError("reference target floor alignment is missing")
    try:
        target_revision_id = str(target_metadata["revision_id"])
        target_floor_y_m = (
            float(camera_pose_anchor.target_floor_y_m)
            if camera_pose_anchor is not None
            else float(floor_alignment["target_floor_y"])
        )
        if camera_pose_anchor is not None:
            target_plane = np.asarray(
                [0.0, 1.0, 0.0, -target_floor_y_m], dtype=np.float64
            )
            source_plane = world_correction.T @ target_plane
            normal_norm = float(np.linalg.norm(source_plane[:3]))
            if not math.isfinite(normal_norm) or normal_norm <= 1e-8:
                raise ValueError("PCF anchor source floor normal is degenerate")
            source_floor_normal = tuple(
                float(value) / normal_norm for value in source_plane[:3]
            )
            source_floor_offset_m = float(source_plane[3]) / normal_norm
            has_source_normal = False
            has_source_offset = False
        else:
            has_source_normal = "source_floor_normal" in floor_alignment
            has_source_offset = "source_floor_offset" in floor_alignment
        if has_source_normal != has_source_offset:
            raise ValueError("source floor plane is partial")
        if camera_pose_anchor is not None:
            pass
        elif has_source_normal:
            source_floor_normal = tuple(
                float(value) for value in floor_alignment["source_floor_normal"]
            )
            source_floor_offset_m = float(
                floor_alignment["source_floor_offset"]
            )
        else:
            # Reference-locked targets historically omitted the redundant
            # source plane.  Derive it from the explicit source->target edge:
            # q_source = W^T q_target.  This keeps identity-aligned priors
            # revision-bound without inventing a second floor convention.
            target_plane = np.asarray(
                [0.0, 1.0, 0.0, -float(target_floor_y_m)],
                dtype=np.float64,
            )
            source_plane = world_correction.T @ target_plane
            normal_norm = float(np.linalg.norm(source_plane[:3]))
            if not math.isfinite(normal_norm) or normal_norm <= 1e-8:
                raise ValueError("derived source floor normal is degenerate")
            source_floor_normal = tuple(
                float(value) / normal_norm for value in source_plane[:3]
            )
            source_floor_offset_m = float(source_plane[3]) / normal_norm
    except (KeyError, TypeError, ValueError) as exc:
        raise ScenePriorBuildError(
            "reference target floor-frame metadata is malformed"
        ) from exc
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}", target_revision_id)
        is None
        or len(source_floor_normal) != 3
        or not all(math.isfinite(value) for value in source_floor_normal)
        or not math.isfinite(source_floor_offset_m)
        or not math.isfinite(target_floor_y_m)
    ):
        raise ScenePriorBuildError(
            "reference target floor-frame metadata is invalid"
        )
    return _CameraPreviewFrame(
        camera_id=camera_id,
        camera_position_world_m=tuple(float(value) for value in position),
        camera_right_world_xz=tuple(float(value) for value in right),
        camera_forward_world_xz=tuple(float(value) for value in forward),
        camera_calibration=ArtifactFingerprint(
            role="reference_camera_calibration",
            sha256=_sha256(inputs["camera_calibration_bytes"]),
            version="camera_from_backend_col_major_v1",
            producer="noesis",
        ),
        target_revision_metadata=ArtifactFingerprint(
            role="reference_target_revision_metadata",
            sha256=_sha256(inputs["target_revision_metadata_bytes"]),
            version=str(target_metadata.get("schema") or "unknown"),
            producer="noesis.virtual_twin",
        ),
        target_revision_id=target_revision_id,
        camera_map_lock=map_lock_contract,
        camera_pose_anchor=(
            camera_pose_anchor.evidence
            if camera_pose_anchor is not None
            else None
        ),
        target_from_source_col_major=tuple(
            float(value) for value in world_correction.flatten(order="F")
        ),
        source_floor_normal=source_floor_normal,
        source_floor_offset_m=source_floor_offset_m,
        target_floor_y_m=target_floor_y_m,
    )


def _load_points_glb(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    try:
        import trimesh

        loaded = trimesh.load(io.BytesIO(payload), file_type="glb", process=False)
    except Exception as exc:
        raise ScenePriorBuildError(
            f"aligned room points GLB cannot be decoded: {exc}"
        ) from exc
    geometries = list(getattr(loaded, "geometry", {}).values())
    if not geometries and hasattr(loaded, "vertices"):
        geometries = [loaded]
    candidates: list[tuple[int, Any]] = []
    for geometry in geometries:
        vertices = getattr(geometry, "vertices", None)
        if vertices is None:
            continue
        count = int(np.asarray(vertices).reshape((-1, 3)).shape[0])
        if count:
            candidates.append((count, geometry))
    if not candidates:
        raise ScenePriorBuildError("aligned room points GLB contains no vertices")
    _, geometry = max(candidates, key=lambda item: item[0])
    points = np.asarray(geometry.vertices, dtype=np.float64).reshape((-1, 3))
    raw_colors = getattr(getattr(geometry, "visual", None), "vertex_colors", None)
    if raw_colors is None or np.asarray(raw_colors).shape[0] != points.shape[0]:
        colors = np.full((points.shape[0], 3), [190, 215, 235], dtype=np.uint8)
    else:
        colors = np.asarray(raw_colors, dtype=np.uint8)[:, :3].copy()
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    colors = colors[finite]
    if points.shape[0] < 100:
        raise ScenePriorBuildError("aligned room point cloud has too few finite points")
    return points, colors


def _selected_authored_triangles(
    geometry: AuthoredSceneGeometry,
    authored_groups: Sequence[str],
    world_to_scene_matrix: np.ndarray,
) -> list[np.ndarray]:
    wanted = set(authored_groups)
    inverse = np.linalg.inv(np.asarray(world_to_scene_matrix, dtype=np.float64))
    triangles: list[np.ndarray] = []
    present: set[str] = set()
    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    for surface in geometry.walkable_triangles:
        if surface.authority_group not in wanted:
            continue
        present.add(surface.authority_group)
        scene = vertices[list(surface.triangle.vertex_indices)]
        homogeneous = np.concatenate(
            [scene, np.ones((scene.shape[0], 1), dtype=np.float64)], axis=1
        )
        world = (inverse @ homogeneous.T).T
        triangles.append(world[:, :3] / world[:, 3:4])
    missing = wanted.difference(present)
    if missing:
        raise ScenePriorBuildError(
            "authored room groups are not walkable surfaces: "
            + ", ".join(sorted(missing))
        )
    if not triangles:
        raise ScenePriorBuildError(
            "selected semantic rooms have no authored walkable geometry"
        )
    return triangles


def _points_in_triangle_xz(
    x: np.ndarray, z: np.ndarray, triangle: np.ndarray
) -> np.ndarray:
    ax, az = triangle[0, 0], triangle[0, 2]
    bx, bz = triangle[1, 0], triangle[1, 2]
    cx, cz = triangle[2, 0], triangle[2, 2]
    denominator = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
    if abs(float(denominator)) <= 1e-12:
        return np.zeros(x.shape, dtype=bool)
    alpha = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) / denominator
    beta = ((cz - az) * (x - cx) + (ax - cx) * (z - cz)) / denominator
    gamma = 1.0 - alpha - beta
    epsilon = 1e-8
    return (alpha >= -epsilon) & (beta >= -epsilon) & (gamma >= -epsilon)


def _authored_grid(
    triangles: Sequence[np.ndarray],
    resolution_m: float,
    *,
    evidence_points: np.ndarray | None = None,
) -> tuple[ScenePriorGrid, np.ndarray]:
    authored_vertices = np.concatenate(triangles, axis=0)
    extent_vertices = authored_vertices
    if evidence_points is not None:
        evidence = np.asarray(evidence_points, dtype=np.float64)
        if evidence.ndim != 2 or evidence.shape[1] != 3:
            raise ScenePriorBuildError("scene-prior evidence points must be Nx3")
        if evidence.shape[0] == 0 or not np.all(np.isfinite(evidence)):
            raise ScenePriorBuildError(
                "scene-prior evidence extent requires finite source points"
            )
        extent_vertices = np.concatenate((authored_vertices, evidence), axis=0)
    vertices = extent_vertices
    min_x = math.floor(float(np.min(vertices[:, 0])) / resolution_m) * resolution_m
    max_x = (
        math.floor(float(np.max(vertices[:, 0])) / resolution_m) + 1
    ) * resolution_m
    min_z = math.floor(float(np.min(vertices[:, 2])) / resolution_m) * resolution_m
    max_z = (
        math.floor(float(np.max(vertices[:, 2])) / resolution_m) + 1
    ) * resolution_m
    columns = int(round((max_x - min_x) / resolution_m))
    rows = int(round((max_z - min_z) / resolution_m))
    if columns < 1 or rows < 1 or columns * rows > 16 * 1024 * 1024:
        raise ScenePriorBuildError(
            "selected authored space produces an invalid grid shape"
        )
    center_x = min_x + (np.arange(columns, dtype=np.float64) + 0.5) * resolution_m
    center_z = min_z + (np.arange(rows, dtype=np.float64) + 0.5) * resolution_m
    authored = np.zeros((rows, columns), dtype=bool)
    for triangle in triangles:
        column_start = max(
            0, int(math.floor((float(np.min(triangle[:, 0])) - min_x) / resolution_m))
        )
        column_stop = min(
            columns,
            int(math.ceil((float(np.max(triangle[:, 0])) - min_x) / resolution_m)),
        )
        row_start = max(
            0, int(math.floor((float(np.min(triangle[:, 2])) - min_z) / resolution_m))
        )
        row_stop = min(
            rows, int(math.ceil((float(np.max(triangle[:, 2])) - min_z) / resolution_m))
        )
        if column_stop <= column_start or row_stop <= row_start:
            continue
        xx, zz = np.meshgrid(
            center_x[column_start:column_stop], center_z[row_start:row_stop]
        )
        authored[row_start:row_stop, column_start:column_stop] |= (
            _points_in_triangle_xz(xx, zz, triangle)
        )
    if not np.any(authored):
        raise ScenePriorBuildError(
            "selected authored space does not cover any grid cells"
        )
    grid = ScenePriorGrid(
        coordinate_frame="backend_world_m",
        units="meters",
        orientation="row_increases_positive_z_column_increases_positive_x",
        bounds=ScenePriorBounds(min_x=min_x, max_x=max_x, min_z=min_z, max_z=max_z),
        resolution_m=resolution_m,
        rows=rows,
        columns=columns,
    )
    return grid, authored


def _group_percentile(
    linear_indices: np.ndarray,
    values: np.ndarray,
    *,
    cell_count: int,
    percentile: float,
    default: float,
) -> np.ndarray:
    output = np.full(cell_count, default, dtype=np.float32)
    if linear_indices.size == 0:
        return output
    order = np.argsort(linear_indices, kind="stable")
    sorted_indices = linear_indices[order]
    sorted_values = values[order]
    starts = np.flatnonzero(np.r_[True, sorted_indices[1:] != sorted_indices[:-1]])
    stops = np.r_[starts[1:], sorted_indices.size]
    for start, stop in zip(starts.tolist(), stops.tolist()):
        output[int(sorted_indices[start])] = float(
            np.percentile(sorted_values[start:stop], percentile)
        )
    return output


def _derive_arrays(
    points: np.ndarray,
    *,
    grid: ScenePriorGrid,
    authored: np.ndarray,
    derivation: ScenePriorDerivation,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rows = int(grid.rows)
    columns = int(grid.columns)
    resolution = float(grid.resolution_m)
    point_rows = np.floor(
        (points[:, 2] - float(grid.bounds.min_z)) / resolution
    ).astype(np.int64)
    point_columns = np.floor(
        (points[:, 0] - float(grid.bounds.min_x)) / resolution
    ).astype(np.int64)
    in_grid = (
        (point_rows >= 0)
        & (point_rows < rows)
        & (point_columns >= 0)
        & (point_columns < columns)
    )
    safe_rows = np.clip(point_rows, 0, rows - 1)
    safe_columns = np.clip(point_columns, 0, columns - 1)
    vertically_admitted = (
        (
            points[:, 1]
            >= float(derivation.floor_y_m) - float(derivation.floor_support_band_m)
        )
        & (
            points[:, 1]
            <= float(derivation.floor_y_m) + float(derivation.max_source_height_m)
        )
    )
    full_evidence = (
        derivation.algorithm == "noesis_scene_prior_2_5d_full_evidence_v2"
    )
    selected = in_grid & vertically_admitted
    if not full_evidence:
        selected &= authored[safe_rows, safe_columns]
    selected_points = points[selected]
    selected_rows = point_rows[selected]
    selected_columns = point_columns[selected]
    if selected_points.shape[0] < 100:
        raise ScenePriorBuildError(
            "too few aligned points remain in the selected scene-prior extent"
        )
    linear = selected_rows * columns + selected_columns
    cell_count = rows * columns
    point_count = (
        np.bincount(linear, minlength=cell_count)
        .reshape((rows, columns))
        .astype(np.uint32)
    )
    height = selected_points[:, 1] - float(derivation.floor_y_m)
    floor_point = np.abs(height) <= float(derivation.floor_support_band_m)
    obstacle_point = (height >= float(derivation.obstacle_min_height_m)) & (
        height <= float(derivation.obstacle_max_height_m)
    )
    floor_count = (
        np.bincount(linear[floor_point], minlength=cell_count)
        .reshape((rows, columns))
        .astype(np.uint32)
    )
    obstacle_count = (
        np.bincount(linear[obstacle_point], minlength=cell_count)
        .reshape((rows, columns))
        .astype(np.uint32)
    )
    relevant_height = (height >= 0.0) & (
        height <= float(derivation.obstacle_max_height_m)
    )
    height_p95 = _group_percentile(
        linear[relevant_height],
        height[relevant_height],
        cell_count=cell_count,
        percentile=95.0,
        default=math.nan,
    ).reshape((rows, columns))
    floor_height = _group_percentile(
        linear[floor_point],
        selected_points[floor_point, 1],
        cell_count=cell_count,
        percentile=50.0,
        default=float(derivation.floor_y_m),
    ).reshape((rows, columns))
    if full_evidence:
        observed = point_count > 0
        floor_supported = floor_count > 0
        obstacle = obstacle_count >= int(derivation.obstacle_min_support)
        walkable_candidate = observed & floor_supported & ~obstacle
    else:
        observed = authored & (point_count > 0)
        floor_supported = authored & (floor_count > 0)
        obstacle = authored & (
            obstacle_count >= int(derivation.obstacle_min_support)
        )
        walkable_candidate = authored & observed & floor_supported & ~obstacle
    evidence_confidence = np.zeros((rows, columns), dtype=np.float32)
    density_confidence = 1.0 - np.exp(-point_count.astype(np.float32) / 8.0)
    floor_confidence = 1.0 - np.exp(-floor_count.astype(np.float32) / 3.0)
    confidence_extent = observed if full_evidence else authored
    evidence_confidence[confidence_extent] = (
        0.7 * density_confidence[confidence_extent]
        + 0.3 * floor_confidence[confidence_extent]
    )
    try:
        from scipy.ndimage import distance_transform_edt

        boundary_signed = (
            distance_transform_edt(authored) - distance_transform_edt(~authored)
        ) * resolution
        obstacle_clearance = (
            distance_transform_edt(~obstacle) - distance_transform_edt(obstacle)
        ) * resolution
    except Exception as exc:
        raise ScenePriorBuildError(
            f"scene-prior distance fields could not be derived: {exc}"
        ) from exc
    arrays = {
        "authored_walkable": authored.astype(np.uint8),
        "observed": observed.astype(np.uint8),
        "evidence_confidence": evidence_confidence.astype(np.float32),
        "floor_supported": floor_supported.astype(np.uint8),
        "obstacle_mask": obstacle.astype(np.uint8),
        "walkable_candidate": walkable_candidate.astype(np.uint8),
        "floor_height_m": floor_height.astype(np.float32),
        "height_agl_p95_m": height_p95.astype(np.float32),
        "boundary_signed_distance_m": boundary_signed.astype(np.float32),
        "obstacle_signed_clearance_m": obstacle_clearance.astype(np.float32),
        "point_count": point_count,
        "floor_support_count": floor_count,
        "obstacle_support_count": obstacle_count,
    }
    return arrays, selected


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for name in sorted(arrays):
            array_file = io.BytesIO()
            np.lib.format.write_array(
                array_file,
                np.ascontiguousarray(arrays[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                array_file.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
    return output.getvalue()


def _review_points_glb(points: np.ndarray, colors: np.ndarray) -> bytes:
    if points.shape[0] > MAX_REVIEW_POINTS:
        indices = np.linspace(0, points.shape[0] - 1, MAX_REVIEW_POINTS, dtype=np.int64)
        points = points[indices]
        colors = colors[indices]
    with TemporaryDirectory(prefix="noesis-scene-prior-") as directory:
        path = Path(directory) / "points.glb"
        write_points_glb(path, points.astype(np.float32), colors.astype(np.uint8))
        return path.read_bytes()


def _camera_local_preview_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    grid: ScenePriorGrid,
    camera_position_world_m: Sequence[float],
    camera_right_world_xz: Sequence[float],
    camera_forward_world_xz: Sequence[float],
) -> tuple[dict[str, np.ndarray], ScenePriorBounds]:
    resolution = float(grid.resolution_m)
    camera_xz = np.asarray(
        [camera_position_world_m[0], camera_position_world_m[2]],
        dtype=np.float64,
    )
    right = np.asarray(camera_right_world_xz, dtype=np.float64)
    forward = np.asarray(camera_forward_world_xz, dtype=np.float64)
    corners = np.asarray(
        [
            [grid.bounds.min_x, grid.bounds.min_z],
            [grid.bounds.min_x, grid.bounds.max_z],
            [grid.bounds.max_x, grid.bounds.min_z],
            [grid.bounds.max_x, grid.bounds.max_z],
        ],
        dtype=np.float64,
    )
    relative = corners - camera_xz
    local_x = relative @ right
    local_z = relative @ forward
    min_x = math.floor(float(np.min(local_x)) / resolution) * resolution
    max_x = math.ceil(float(np.max(local_x)) / resolution) * resolution
    min_z = math.floor(float(np.min(local_z)) / resolution) * resolution
    max_z = math.ceil(float(np.max(local_z)) / resolution) * resolution
    columns = int(round((max_x - min_x) / resolution))
    rows = int(round((max_z - min_z) / resolution))
    if columns < 1 or rows < 1 or columns * rows > 16 * 1024 * 1024:
        raise ScenePriorBuildError(
            "reference-camera preview produces an invalid grid shape"
        )
    center_x = min_x + (np.arange(columns, dtype=np.float64) + 0.5) * resolution
    center_z = min_z + (np.arange(rows, dtype=np.float64) + 0.5) * resolution
    xx, zz = np.meshgrid(center_x, center_z)
    world_x = camera_xz[0] + xx * right[0] + zz * forward[0]
    world_z = camera_xz[1] + xx * right[1] + zz * forward[1]
    source_columns = np.floor((world_x - float(grid.bounds.min_x)) / resolution).astype(
        np.int64
    )
    source_rows = np.floor((world_z - float(grid.bounds.min_z)) / resolution).astype(
        np.int64
    )
    valid = (
        (source_rows >= 0)
        & (source_rows < int(grid.rows))
        & (source_columns >= 0)
        & (source_columns < int(grid.columns))
    )
    transformed: dict[str, np.ndarray] = {}
    for name in ("authored_walkable", "observed", "floor_supported", "obstacle_mask"):
        source = np.asarray(arrays[name], dtype=bool)
        if source.shape != (int(grid.rows), int(grid.columns)):
            raise ScenePriorBuildError(
                f"scene-prior preview layer {name} has the wrong shape"
            )
        destination = np.zeros((rows, columns), dtype=bool)
        destination[valid] = source[
            source_rows[valid],
            source_columns[valid],
        ]
        transformed[name] = destination
    return transformed, ScenePriorBounds(
        min_x=min_x,
        max_x=max_x,
        min_z=min_z,
        max_z=max_z,
    )


def _camera_local_raster_to_preview_image(values: np.ndarray) -> np.ndarray:
    """Convert +Z-increasing numeric rows to row-zero-far image addressing."""

    return np.flip(np.asarray(values), axis=0).copy()


def _preview_png(arrays: Mapping[str, np.ndarray]) -> bytes:
    try:
        from PIL import Image
    except Exception as exc:
        raise ScenePriorBuildError(
            f"scene-prior preview requires Pillow: {exc}"
        ) from exc
    authored = np.asarray(arrays["authored_walkable"], dtype=bool)
    observed = np.asarray(arrays["observed"], dtype=bool)
    floor_supported = np.asarray(arrays["floor_supported"], dtype=bool)
    obstacle = np.asarray(arrays["obstacle_mask"], dtype=bool)
    preview = np.full((*authored.shape, 3), [20, 24, 30], dtype=np.uint8)
    preview[authored] = [72, 76, 84]
    preview[observed] = [62, 118, 151]
    preview[floor_supported] = [70, 160, 118]
    preview[obstacle] = [222, 103, 72]
    image = Image.fromarray(
        _camera_local_raster_to_preview_image(preview),
        mode="RGB",
    )
    scale = min(
        8,
        max(
            1,
            min(
                MAX_PREVIEW_DIMENSION // max(1, image.width),
                MAX_PREVIEW_DIMENSION // max(1, image.height),
            ),
        ),
    )
    if scale > 1:
        image = image.resize(
            (image.width * scale, image.height * scale),
            resample=Image.Resampling.NEAREST,
        )
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _artifact(role: str, relative_path: str, payload: bytes) -> ScenePriorArtifact:
    return ScenePriorArtifact(
        role=role,
        relative_path=relative_path,
        sha256=_sha256(payload),
        size_bytes=len(payload),
    )


def _atomic_write_catalog(path: Path, payload: bytes) -> None:
    parent = ensure_scene_directory(path.parent, label="scene-prior catalog parent")
    destination = parent / path.name
    try:
        existing = os.lstat(destination)
    except FileNotFoundError:
        existing = None
    if existing is not None and (
        not os.path.isfile(destination) or os.path.islink(destination)
    ):
        raise ScenePriorBuildError(
            "scene-prior catalog destination is not a regular file"
        )
    temporary = parent / f".{path.name}.tmp-{secrets.token_hex(12)}"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ScenePriorBuildError("scene-prior catalog write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, destination)
        parent_descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    try:
        read_scene_file(
            destination,
            label="scene-prior catalog",
            max_bytes=len(payload),
            expected_sha256=_sha256(payload),
            expected_size=len(payload),
        )
    except SceneFileError as exc:
        raise ScenePriorBuildError(str(exc)) from exc


def _load_acceptance_report(
    report_or_path: Mapping[str, Any] | Path | None,
) -> Mapping[str, Any]:
    if report_or_path is None:
        raise ScenePriorBuildError(
            "accepted room-to-home binding requires a registration acceptance report"
        )
    if isinstance(report_or_path, Mapping):
        report: Any = report_or_path
    else:
        try:
            _, report = _read_json_file(
                Path(report_or_path),
                label="room-to-home registration acceptance report",
                max_bytes=MAX_TARGET_REVISION_METADATA_BYTES,
            )
        except (OSError, SceneFileError) as exc:
            raise ScenePriorBuildError(str(exc)) from exc
    if not isinstance(report, Mapping):
        raise ScenePriorBuildError("registration acceptance report must be an object")
    return report


def _validate_room_to_home_acceptance_report(
    report_or_path: Mapping[str, Any] | Path | None,
    *,
    binding: ScenePriorFrameBinding,
) -> None:
    """Require an existing report to authorize one exact immutable edge."""

    report = _load_acceptance_report(report_or_path)
    if report.get("schema") not in _ACCEPTED_REGISTRATION_SCHEMAS:
        raise ScenePriorBuildError(
            "registration acceptance report has an unsupported producer schema"
        )
    if report.get("status") != "passed":
        raise ScenePriorBuildError(
            "registration acceptance report must have status=passed"
        )
    if report.get("accepted_for_canonical_use") is not True:
        raise ScenePriorBuildError(
            "registration acceptance report is not accepted for canonical use"
        )
    disposition = str(report.get("disposition") or "").strip().lower()
    if disposition not in _ACCEPTED_REGISTRATION_DISPOSITIONS:
        raise ScenePriorBuildError(
            "registration acceptance report has no canonical holdout disposition"
        )
    pose_graph = report.get("pose_graph")
    if (
        not isinstance(pose_graph, Mapping)
        or pose_graph.get("accepted") is not True
        or pose_graph.get("reason_codes") not in ([], ())
    ):
        raise ScenePriorBuildError(
            "registration acceptance report pose_graph did not pass"
        )
    # The registration producer owns this evidence.  Keep the import shape
    # exact so a caller cannot make an arbitrary disposition or alias satisfy
    # the canonical holdout gate.
    holdout = report.get("holdout")
    if (
        not isinstance(holdout, Mapping)
        or holdout.get("source") != "pose_graph"
        or holdout.get("status") != "passed"
    ):
        raise ScenePriorBuildError(
            "registration acceptance report must declare holdout.status=passed"
        )

    exact = None
    candidate = report.get("bound_transform")
    if isinstance(candidate, Mapping):
        exact = candidate
    if exact is None:
        raise ScenePriorBuildError(
            "registration acceptance report lacks an exact bound_transform"
        )

    def _frame(payload: Any, label: str) -> tuple[str, str]:
        if not isinstance(payload, Mapping):
            raise ScenePriorBuildError(f"registration report {label} is missing")
        return str(payload.get("frame_id") or ""), str(payload.get("revision") or "")

    if _frame(exact.get("source_frame"), "source_frame") != (
        binding.source_frame.frame_id,
        binding.source_frame.revision,
    ):
        raise ScenePriorBuildError("registration report source frame does not match binding")
    if _frame(exact.get("target_frame"), "target_frame") != (
        binding.target_frame.frame_id,
        binding.target_frame.revision,
    ):
        raise ScenePriorBuildError("registration report target frame does not match binding")
    try:
        report_matrix = tuple(
            float(value) for value in exact["target_from_source_col_major"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ScenePriorBuildError(
            "registration report bound transform matrix is malformed"
        ) from exc
    if report_matrix != binding.target_from_source_col_major:
        raise ScenePriorBuildError(
            "registration report bound transform does not match binding"
        )
    report_digest = str(exact.get("target_from_source_sha256") or "")
    if report_digest != binding.target_from_source_sha256:
        raise ScenePriorBuildError(
            "registration report transform digest does not match binding"
        )

    for key, expected in (
        ("source_floor_plane", binding.source_floor_plane),
        ("target_floor_plane", binding.target_floor_plane),
    ):
        declared = exact.get(key)
        if declared is None:
            raise ScenePriorBuildError(
                f"registration acceptance report lacks {key}"
            )
        if not isinstance(declared, Mapping):
            raise ScenePriorBuildError(f"registration report {key} is malformed")
        try:
            normal = tuple(float(value) for value in declared["normal"])
            offset = float(declared["offset_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ScenePriorBuildError(f"registration report {key} is malformed") from exc
        if normal != expected.normal or offset != expected.offset_m:
            raise ScenePriorBuildError(f"registration report {key} does not match binding")


def build_accepted_room_to_home_binding(
    *,
    artifact_revision_id: str,
    source_frame_revision: str,
    target_coordinate_revision: str | None,
    target_from_source_col_major: Sequence[float],
    source_floor_normal: Sequence[float],
    source_floor_offset_m: float,
    target_floor_normal: Sequence[float],
    target_floor_offset_m: float,
    source_camera_calibration_sha256: str,
    source_world_alignment_sha256: str,
    source_camera_calibration_physical_sha256: str,
    source_world_alignment_physical_sha256: str,
    target_revision_id: str,
    target_revision_metadata_sha256: str,
    metric_frame_provenance: Sequence[ArtifactFingerprint],
    acceptance_report: Mapping[str, Any] | Path | None = None,
    world_to_scene_col_major: Sequence[float] | None = None,
    world_to_scene_provenance: ArtifactFingerprint | None = None,
) -> ScenePriorFrameBinding:
    """Build one accepted room-to-home v2 frame edge from a report.

    ``acceptance_report`` is the registration authority.  It must declare a
    passed holdout, canonical acceptance, and an exact ``bound_transform``
    object containing source/target frame revisions, floor planes, matrix, and
    transform digest.  Geometry supplied separately is checked against that
    report; the helper never upgrades a review-only assembly or infers a common
    frame from room names.
    """

    try:
        provenance = tuple(metric_frame_provenance)
        if not provenance:
            raise ScenePriorBuildError(
                "accepted room-to-home binding requires physical provenance"
            )
        target_normal = tuple(float(value) for value in target_floor_normal)
        target_offset = float(target_floor_offset_m)
        target_revision = target_coordinate_revision or metric_frame_revision_sha256(
            BACKEND_WORLD_FRAME_ID,
            target_normal,
            target_offset,
            tuple(item.sha256 for item in provenance),
        )
        source_frame = ScenePriorFrameRef(
            frame_id=BACKEND_WORLD_FRAME_ID,
            revision=str(source_frame_revision),
        )
        target_frame = ScenePriorFrameRef(
            frame_id=BACKEND_WORLD_FRAME_ID,
            revision=str(target_revision),
        )
        source_plane = ScenePriorFloorPlane(
            frame=source_frame,
            normal=tuple(float(value) for value in source_floor_normal),
            offset_m=float(source_floor_offset_m),
        )
        target_plane = ScenePriorFloorPlane(
            frame=target_frame,
            normal=tuple(float(value) for value in target_floor_normal),
            offset_m=float(target_floor_offset_m),
        )
        metric_frame = ScenePriorMetricFrame(
            contract="noesis.scene_prior.metric_frame",
            contract_version=1,
            frame=target_frame,
            units="meters",
            floor_plane=target_plane,
            physical_provenance=provenance,
            accepted=True,
        )
        edge_digest = revisioned_transform_sha256(
            RevisionedFrame(
                frame_id=source_frame.frame_id,
                revision=source_frame.revision,
            ),
            RevisionedFrame(
                frame_id=target_frame.frame_id,
                revision=target_frame.revision,
            ),
            tuple(float(value) for value in target_from_source_col_major),
        )
        presentation = None
        if world_to_scene_col_major is not None:
            if world_to_scene_provenance is None:
                raise ScenePriorBuildError(
                    "world-to-scene mapping requires provenance"
                )
            presentation_values = tuple(
                float(value) for value in world_to_scene_col_major
            )
            presentation = ScenePriorWorldToScenePresentation(
                contract="noesis.scene_prior.world_to_scene_presentation",
                contract_version=1,
                world_frame=target_frame,
                render_frame="menon_scene",
                world_to_scene_col_major=presentation_values,
                world_to_scene_sha256=world_to_scene_presentation_sha256(
                    target_frame,
                    "menon_scene",
                    presentation_values,
                ),
                scene_revision_id=scene_revision_sha256(
                    target_frame,
                    "menon_scene",
                    presentation_values,
                ),
                source_transform_sha256s=(edge_digest,),
                provenance=world_to_scene_provenance,
                accepted=True,
            )
        binding = ScenePriorFrameBinding(
            contract="noesis.scene_prior.frame_binding",
            contract_version=2,
            source_frame=source_frame,
            target_frame=target_frame,
            source_camera_calibration_sha256=source_camera_calibration_sha256,
            source_world_alignment_sha256=source_world_alignment_sha256,
            target_revision_id=target_revision_id,
            target_revision_metadata_sha256=target_revision_metadata_sha256,
            target_from_source_col_major=tuple(
                float(value) for value in target_from_source_col_major
            ),
            target_from_source_sha256=edge_digest,
            source_floor_plane=source_plane,
            target_floor_plane=target_plane,
            metric_frame=metric_frame,
            artifact_revision_id=artifact_revision_id,
            source_camera_calibration_physical_sha256=(
                source_camera_calibration_physical_sha256
            ),
            source_world_alignment_physical_sha256=(
                source_world_alignment_physical_sha256
            ),
            presentation=presentation,
        )
        _validate_room_to_home_acceptance_report(
            acceptance_report,
            binding=binding,
        )
        return binding
    except (CoordinateFrameError, TypeError, ValueError) as exc:
        if isinstance(exc, ScenePriorBuildError):
            raise
        raise ScenePriorBuildError(
            f"accepted room-to-home binding is invalid: {exc}"
        ) from exc


def import_accepted_room_to_home_binding(
    payload_or_path: ScenePriorFrameBinding | Mapping[str, Any] | Path,
    acceptance_report: Mapping[str, Any] | Path | None = None,
) -> ScenePriorFrameBinding:
    """Load an accepted v2 room-to-home binding with its acceptance report."""

    if isinstance(payload_or_path, ScenePriorFrameBinding):
        payload: Any = payload_or_path.model_dump(mode="json")
    elif isinstance(payload_or_path, Mapping):
        payload: Any = dict(payload_or_path)
    else:
        try:
            _, payload = _read_json_file(
                Path(payload_or_path),
                label="accepted room-to-home frame binding",
                max_bytes=MAX_TARGET_REVISION_METADATA_BYTES,
            )
        except (OSError, SceneFileError) as exc:
            raise ScenePriorBuildError(str(exc)) from exc
    report = acceptance_report
    if isinstance(payload, Mapping):
        embedded = payload.pop("acceptance_report", None)
        if embedded is None:
            embedded = payload.pop("registration_acceptance", None)
        if report is None:
            report = embedded
        nested_binding = payload.pop("binding", None)
        if isinstance(nested_binding, Mapping):
            payload = nested_binding
    try:
        binding = ScenePriorFrameBinding.model_validate(payload)
    except Exception as exc:
        raise ScenePriorBuildError(
            f"accepted room-to-home frame binding is invalid: {exc}"
        ) from exc
    if binding.contract_version != 2 or binding.metric_frame is None:
        raise ScenePriorBuildError(
            "accepted room-to-home import requires frame binding v2"
        )
    _validate_room_to_home_acceptance_report(report, binding=binding)
    return binding


def _updated_catalog(
    path: Path,
    *,
    site_id: str,
    entry: ScenePriorCatalogEntry,
    camera_ids: Sequence[str],
    include_floorplan_layers: bool,
    frame_binding: ScenePriorFrameBinding,
) -> ScenePriorCatalog:
    if path.exists():
        _, payload = _read_json_file(
            path,
            label="existing scene-prior catalog",
            max_bytes=4 * 1024 * 1024,
        )
        try:
            existing = ScenePriorCatalog.model_validate(payload)
        except Exception as exc:
            raise ScenePriorBuildError(
                f"existing scene-prior catalog is invalid: {exc}"
            ) from exc
        if existing.site_id != site_id:
            raise ScenePriorBuildError(
                "existing scene-prior catalog belongs to another site"
            )
        entries = {item.prior_id: item for item in existing.revisions}
        previous = entries.get(entry.prior_id)
        if previous is not None and previous != entry:
            raise ScenePriorBuildError(
                "scene-prior ID already names different immutable content"
            )
        entries[entry.prior_id] = entry
        bindings = {binding.camera_id: binding for binding in existing.camera_bindings}
    else:
        entries = {entry.prior_id: entry}
        bindings: dict[str, ScenePriorCameraBinding] = {}
    for camera_id in camera_ids:
        bindings[camera_id] = ScenePriorCameraBinding(
            camera_id=camera_id,
            space_id=entry.space_id,
            prior_id=entry.prior_id,
            mode="shadow",
            include_floorplan_layers=include_floorplan_layers,
            frame_binding=frame_binding,
        )
    return ScenePriorCatalog(
        contract="noesis.scene_prior.catalog",
        contract_version=1,
        site_id=site_id,
        revisions=tuple(entries[key] for key in sorted(entries)),
        camera_bindings=tuple(bindings[key] for key in sorted(bindings)),
    )


def build_scene_prior(config: ScenePriorBuildConfig) -> ScenePriorBuildResult:
    cfg = config.validate()
    bundle_root = Path(os.path.abspath(os.fspath(cfg.source_bundle.expanduser())))
    output_root = Path(os.path.abspath(os.fspath(cfg.output_root.expanduser())))
    inputs = _bundle_inputs(bundle_root)
    room_map_bytes, room_map = _read_json_file(
        cfg.room_group_map,
        label="authored room-group map",
        max_bytes=MAX_ROOM_GROUP_MAP_BYTES,
    )
    if (
        room_map.get("contract") != "noesis.authored_scene.room_group_map"
        or room_map.get("contract_version") != 1
    ):
        raise ScenePriorBuildError("authored room-group map contract is unsupported")
    rooms = room_map.get("rooms")
    if not isinstance(rooms, Mapping):
        raise ScenePriorBuildError("authored room-group map has no room inventory")
    authored_groups: list[str] = []
    for label in cfg.semantic_rooms:
        groups = rooms.get(label)
        if (
            not isinstance(groups, list)
            or not groups
            or any(not isinstance(group, str) for group in groups)
        ):
            raise ScenePriorBuildError(
                f"semantic room {label!r} has no authored groups"
            )
        authored_groups.extend(groups)
    if len(set(authored_groups)) != len(authored_groups):
        raise ScenePriorBuildError("selected semantic rooms repeat an authored group")

    try:
        authored_file = read_scene_file(
            cfg.authored_scene,
            label="authored scene OBJ",
            max_bytes=64 * 1024 * 1024,
        )
        world_to_scene_file = read_scene_file(
            cfg.world_to_scene,
            label="world-to-scene alignment",
            max_bytes=MAX_WORLD_TO_SCENE_BYTES,
        )
    except SceneFileError as exc:
        raise ScenePriorBuildError(str(exc)) from exc
    if str(room_map.get("authored_scene_sha256") or "") != authored_file.sha256:
        raise ScenePriorBuildError(
            "room-group map does not bind the selected authored scene"
        )
    geometry = AuthoredSceneGeometry.from_obj(cfg.authored_scene)
    if geometry.source_sha256 != authored_file.sha256:
        raise ScenePriorBuildError("authored scene changed while geometry was loaded")
    try:
        similarity = load_similarity(cfg.world_to_scene)
    except (OSError, ValueError) as exc:
        raise ScenePriorBuildError(f"world-to-scene similarity is invalid: {exc}") from exc
    world_to_scene_payload = load_strict_json(
        world_to_scene_file.data,
        label="world-to-scene alignment",
    )
    if not isinstance(world_to_scene_payload, Mapping):
        raise ScenePriorBuildError(
            "world-to-scene alignment must contain a JSON object"
        )
    try:
        floor_y_m = float(world_to_scene_payload["floor_y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScenePriorBuildError(
            "world-to-scene alignment must declare floor_y"
        ) from exc
    if not math.isfinite(floor_y_m):
        raise ScenePriorBuildError("world-to-scene floor_y must be finite")

    camera_map_lock = _load_camera_map_lock(cfg.camera_map_lock, inputs)
    camera_pose_anchor = _load_camera_pose_anchor(cfg.camera_pose_anchor, inputs)
    preview_frame = _camera_preview_frame(
        inputs,
        camera_map_lock=camera_map_lock,
        camera_pose_anchor=camera_pose_anchor,
    )
    try:
        materialized_target_from_source = tuple(
            float(value)
            for value in inputs["target_revision_metadata"]["floor_alignment"][
                "world_correction_col_major"
            ]
        )
        _col_major_matrix(materialized_target_from_source)
    except (KeyError, TypeError, ValueError) as exc:
        raise ScenePriorBuildError(
            "reference target metadata does not bind materialized point geometry"
        ) from exc
    accepted_binding: ScenePriorFrameBinding | None = None
    effective_world_to_scene = np.asarray(similarity.matrix, dtype=np.float64)
    if cfg.accepted_frame_binding is not None:
        accepted_binding = import_accepted_room_to_home_binding(
            cfg.accepted_frame_binding,
            acceptance_report=cfg.registration_acceptance_report,
        )
        if accepted_binding.metric_frame is None:
            raise ScenePriorBuildError("accepted v2 binding has no metric frame")
        physical_source, physical_camera_sha256, physical_alignment_sha256 = (
            _source_physical_frame(inputs)
        )
        if accepted_binding.source_frame != ScenePriorFrameRef(
            frame_id=physical_source.frame_id,
            revision=physical_source.revision,
        ):
            raise ScenePriorBuildError(
                "accepted binding source frame does not match the physical bundle calibration"
            )
        if accepted_binding.source_camera_calibration_physical_sha256 != physical_camera_sha256:
            raise ScenePriorBuildError(
                "accepted binding camera physical identity does not match the bundle"
            )
        if accepted_binding.source_world_alignment_physical_sha256 != physical_alignment_sha256:
            raise ScenePriorBuildError(
                "accepted binding world-alignment physical identity does not match the bundle"
            )
        if accepted_binding.source_camera_calibration_sha256 != _sha256(
            inputs["camera_calibration_bytes"]
        ):
            raise ScenePriorBuildError(
                "accepted binding camera calibration artifact does not match the bundle"
            )
        if accepted_binding.source_world_alignment_sha256 != _sha256(
            inputs["transform_bytes"]
        ):
            raise ScenePriorBuildError(
                "accepted binding world-alignment artifact does not match the bundle"
            )
        if accepted_binding.target_revision_id != preview_frame.target_revision_id:
            raise ScenePriorBuildError(
                "accepted binding target revision does not match the bundle revision"
            )
        if accepted_binding.target_revision_metadata_sha256 != preview_frame.target_revision_metadata.sha256:
            raise ScenePriorBuildError(
                "accepted binding target metadata does not match the bundle revision"
            )
        source_plane = accepted_binding.source_floor_plane
        if (
            source_plane.normal != preview_frame.source_floor_normal
            or source_plane.offset_m != preview_frame.source_floor_offset_m
        ):
            raise ScenePriorBuildError(
                "accepted binding source floor plane does not match the bundle"
            )
        target_plane = accepted_binding.target_floor_plane
        target_normal = np.asarray(target_plane.normal, dtype=np.float64)
        if not np.allclose(target_normal, (0.0, 1.0, 0.0), atol=1e-6):
            raise ScenePriorBuildError(
                "accepted binding target floor plane must be horizontal and upward"
            )
        target_from_source = _col_major_matrix(
            accepted_binding.target_from_source_col_major
        )
        try:
            derived_world_to_scene = similarity.matrix @ np.linalg.inv(target_from_source)
        except np.linalg.LinAlgError as exc:
            raise ScenePriorBuildError(
                "accepted room-to-home transform cannot compose authored presentation"
            ) from exc
        if accepted_binding.presentation is not None:
            if cfg.world_presentation is not None:
                try:
                    configured_presentation = (
                        cfg.world_presentation
                        if isinstance(
                            cfg.world_presentation,
                            ScenePriorWorldToScenePresentation,
                        )
                        else ScenePriorWorldToScenePresentation.model_validate(
                            cfg.world_presentation
                        )
                    )
                except Exception as exc:
                    raise ScenePriorBuildError(
                        f"configured world presentation is invalid: {exc}"
                    ) from exc
                if configured_presentation != accepted_binding.presentation:
                    raise ScenePriorBuildError(
                        "configured world presentation does not equal accepted binding presentation"
                    )
            explicit_matrix = _col_major_matrix(
                accepted_binding.presentation.world_to_scene_col_major
            )
            if accepted_binding.presentation.provenance.sha256 != world_to_scene_file.sha256:
                raise ScenePriorBuildError(
                    "accepted presentation provenance does not match the current authored registration"
                )
            if not np.allclose(explicit_matrix, similarity.matrix, atol=1e-8, rtol=1e-8):
                raise ScenePriorBuildError(
                    "accepted presentation is stale for the current authored registration"
                )
            effective_world_to_scene = explicit_matrix
        else:
            if cfg.world_presentation is not None:
                raise ScenePriorBuildError(
                    "world_presentation must be carried by the accepted binding"
                )
            effective_world_to_scene = derived_world_to_scene
        floor_y_m = float(target_plane.offset_m) * -1.0
        preview_frame = _reframe_camera_preview(
            preview_frame,
            accepted_binding.target_from_source_col_major,
            source_floor_normal=source_plane.normal,
            source_floor_offset_m=source_plane.offset_m,
            target_floor_y_m=floor_y_m,
        )
    elif cfg.registration_acceptance_report is not None:
        raise ScenePriorBuildError(
            "registration acceptance report requires accepted_frame_binding"
        )
    triangles = _selected_authored_triangles(
        geometry, authored_groups, effective_world_to_scene
    )
    derivation = ScenePriorDerivation(
        algorithm="noesis_scene_prior_2_5d_full_evidence_v2",
        floor_y_m=floor_y_m,
        floor_support_band_m=float(cfg.floor_support_band_m),
        obstacle_min_height_m=float(cfg.obstacle_min_height_m),
        obstacle_max_height_m=float(cfg.obstacle_max_height_m),
        obstacle_min_support=int(cfg.obstacle_min_support),
        max_source_height_m=float(cfg.max_source_height_m),
    )
    points, colors = _load_points_glb(inputs["aligned_bytes"])
    if accepted_binding is not None:
        # The aligned GLB is materialized in the target revision named by the
        # room metadata.  The accepted binding starts in the raw physical
        # source frame, so convert the already-materialized points by the
        # exact old-target -> accepted-target delta.  Applying the accepted
        # source edge directly would apply the metadata floor correction a
        # second time (and would disagree with the camera preview).
        materialized_target = _col_major_matrix(materialized_target_from_source)
        accepted_target = _col_major_matrix(
            accepted_binding.target_from_source_col_major
        )
        try:
            materialized_to_accepted = accepted_target @ np.linalg.inv(
                materialized_target
            )
        except np.linalg.LinAlgError as exc:
            raise ScenePriorBuildError(
                "accepted room-to-home transform cannot reframe materialized points"
            ) from exc
        points = _apply_col_major_transform(
            points, materialized_to_accepted.flatten(order="F")
        )
    extent_points = points[
        (
            points[:, 1]
            >= float(derivation.floor_y_m) - float(derivation.floor_support_band_m)
        )
        & (
            points[:, 1]
            <= float(derivation.floor_y_m) + float(derivation.max_source_height_m)
        )
    ]
    grid, authored = _authored_grid(
        triangles,
        float(cfg.grid_resolution_m),
        evidence_points=extent_points,
    )
    arrays, selected = _derive_arrays(
        points,
        grid=grid,
        authored=authored,
        derivation=derivation,
    )
    selected_points = points[selected]
    selected_colors = colors[selected]
    authored_count = int(np.count_nonzero(arrays["authored_walkable"]))
    authored_observed_count = int(
        np.count_nonzero(
            np.asarray(arrays["authored_walkable"], dtype=bool)
            & np.asarray(arrays["observed"], dtype=bool)
        )
    )
    authored_floor_count = int(
        np.count_nonzero(
            np.asarray(arrays["authored_walkable"], dtype=bool)
            & np.asarray(arrays["floor_supported"], dtype=bool)
        )
    )
    authored_obstacle_count = int(
        np.count_nonzero(
            np.asarray(arrays["authored_walkable"], dtype=bool)
            & np.asarray(arrays["obstacle_mask"], dtype=bool)
        )
    )
    source_quality = inputs["reference"]["quality"]
    alignment_status = str(inputs["alignment"].get("status") or "unknown")
    quality = ScenePriorQuality(
        passed=bool(source_quality.get("passed"))
        and alignment_status == "passed"
        and authored_observed_count > 0,
        source_point_count=int(points.shape[0]),
        selected_point_count=int(selected_points.shape[0]),
        authored_cell_count=authored_count,
        observed_cell_count=authored_observed_count,
        floor_supported_cell_count=authored_floor_count,
        obstacle_cell_count=authored_obstacle_count,
        authored_observed_fraction=authored_observed_count / authored_count,
        authored_floor_supported_fraction=authored_floor_count / authored_count,
        alignment_status=alignment_status,
    )
    if not quality.passed:
        raise ScenePriorBuildError("scene-prior quality did not pass")

    capture = inputs["reference"].get("capture")
    if not isinstance(capture, Mapping):
        raise ScenePriorBuildError("room-scan reference has no capture identity")
    captured_at_us = _timestamp_us(
        capture.get("captured_at"), label="capture timestamp"
    )
    created_at_us = _timestamp_us(
        inputs["reference"].get("created_at"), label="bundle creation timestamp"
    )
    source_type = str(
        capture.get("source_type") or "mapanything_multiview_room_walk"
    )
    source = ScenePriorSource(
        source_type=source_type,
        bundle_id=inputs["bundle_id"],
        bundle_schema=str(inputs["reference"].get("schema")),
        bundle_manifest_sha256=_sha256(inputs["manifest_bytes"]),
        reference_sha256=_sha256(inputs["reference_bytes"]),
        capture_id=str(capture.get("scan_id") or ""),
        captured_at_us=captured_at_us,
        model=str(capture.get("model") or capture.get("mapanything_model") or ""),
    )
    alignment_fingerprint = ArtifactFingerprint(
        role="mapanything_alignment_report",
        sha256=_sha256(inputs["alignment_bytes"]),
        version=inputs["alignment_schema"],
        producer="mapanything_phone_scan",
    )
    authored_fingerprint = ArtifactFingerprint(
        role="authored_scene",
        sha256=authored_file.sha256,
        version=str(room_map.get("release_id") or "unknown"),
        producer="noesis.virtual_twin",
    )
    world_to_scene_fingerprint = ArtifactFingerprint(
        role="world_to_scene",
        sha256=world_to_scene_file.sha256,
        version="backend_world_to_authored_scene_similarity_v1",
        producer="noesis.validation.authored_scene",
    )
    semantic = ScenePriorSemanticBinding(
        room_labels=tuple(cfg.semantic_rooms),
        authored_groups=tuple(authored_groups),
        room_group_map=ArtifactFingerprint(
            role="authored_scene_room_group_map",
            sha256=_sha256(room_map_bytes),
            version="noesis.authored_scene.room_group_map.v1",
            producer="noesis",
        ),
    )
    preview_arrays, preview_bounds = _camera_local_preview_arrays(
        arrays,
        grid=grid,
        camera_position_world_m=preview_frame.camera_position_world_m,
        camera_right_world_xz=preview_frame.camera_right_world_xz,
        camera_forward_world_xz=preview_frame.camera_forward_world_xz,
    )
    preview = ScenePriorPreview(
        coordinate_frame="camera_local_ground_m",
        units="meters",
        orientation=CAMERA_LOCAL_RASTER_ORIENTATION,
        reference_camera_id=preview_frame.camera_id,
        camera_calibration=preview_frame.camera_calibration,
        target_revision_metadata=preview_frame.target_revision_metadata,
        camera_map_lock=preview_frame.camera_map_lock,
        camera_pose_anchor=preview_frame.camera_pose_anchor,
        camera_position_world_m=preview_frame.camera_position_world_m,
        camera_right_world_xz=preview_frame.camera_right_world_xz,
        camera_forward_world_xz=preview_frame.camera_forward_world_xz,
        bounds=preview_bounds,
        resolution_m=grid.resolution_m,
        rows=int(preview_arrays["authored_walkable"].shape[0]),
        columns=int(preview_arrays["authored_walkable"].shape[1]),
    )
    identity_payload = {
        "algorithm": derivation.algorithm,
        "site_id": cfg.site_id,
        "space_id": cfg.space_id,
        "source": source.model_dump(mode="json"),
        "alignment": alignment_fingerprint.model_dump(mode="json"),
        "authored_scene": authored_fingerprint.model_dump(mode="json"),
        "world_to_scene": world_to_scene_fingerprint.model_dump(mode="json"),
        "semantic_binding": semantic.model_dump(mode="json"),
        "grid": grid.model_dump(mode="json"),
        "preview": preview.model_dump(mode="json"),
        "derivation": derivation.model_dump(mode="json"),
    }
    supplied_metric_frame: ScenePriorMetricFrame | None = None
    supplied_presentation: ScenePriorWorldToScenePresentation | None = None
    if cfg.metric_frame is not None:
        try:
            supplied_metric_frame = (
                cfg.metric_frame
                if isinstance(cfg.metric_frame, ScenePriorMetricFrame)
                else ScenePriorMetricFrame.model_validate(cfg.metric_frame)
            )
        except Exception as exc:
            raise ScenePriorBuildError(
                f"configured metric frame is invalid: {exc}"
            ) from exc
        identity_payload["metric_frame"] = supplied_metric_frame.model_dump(
            mode="json"
        )
    if cfg.world_presentation is not None:
        try:
            supplied_presentation = (
                cfg.world_presentation
                if isinstance(
                    cfg.world_presentation,
                    ScenePriorWorldToScenePresentation,
                )
                else ScenePriorWorldToScenePresentation.model_validate(
                    cfg.world_presentation
                )
            )
        except Exception as exc:
            raise ScenePriorBuildError(
                f"configured world presentation is invalid: {exc}"
            ) from exc
        identity_payload["world_presentation"] = supplied_presentation.model_dump(
            mode="json"
        )
    if accepted_binding is not None:
        if (
            supplied_metric_frame is not None
            and supplied_metric_frame != accepted_binding.metric_frame
        ):
            raise ScenePriorBuildError(
                "configured metric frame does not equal accepted binding metric frame"
            )
        if (
            supplied_presentation is not None
            and supplied_presentation != accepted_binding.presentation
        ):
            raise ScenePriorBuildError(
                "configured world presentation does not equal accepted binding presentation"
            )
        identity_payload["accepted_frame_binding"] = accepted_binding.model_dump(
            mode="json"
        )
        # The source mapping is an input artifact; this composed mapping is
        # the actual target-frame geometry used for grid construction.
        identity_payload["effective_world_to_scene_col_major"] = [
            float(value) for value in effective_world_to_scene.flatten(order="F")
        ]
    identity_digest = _sha256(_compact_json(identity_payload))
    prior_id = f"sceneprior_{cfg.space_id}_{_timestamp_slug(captured_at_us)}_{identity_digest[:12]}"
    if accepted_binding is not None:
        # Keep the producer-owned physical edge and target coordinate revision
        # intact.  Only the scene-prior artifact identity is assigned here;
        # the catalog validator intentionally binds that identity separately.
        frame_payload = accepted_binding.model_dump(mode="json")
        frame_payload["artifact_revision_id"] = prior_id
        frame_binding = ScenePriorFrameBinding.model_validate(frame_payload)
        source_frame = RevisionedFrame(
            frame_id=frame_binding.source_frame.frame_id,
            revision=frame_binding.source_frame.revision,
        )
    else:
        source_frame = RevisionedFrame(
            frame_id=BACKEND_WORLD_FRAME_ID,
            revision=revisioned_frame_sha256(
                BACKEND_WORLD_FRAME_ID,
                artifact_sha256s=(
                    preview_frame.camera_calibration.sha256,
                    world_to_scene_fingerprint.sha256,
                ),
            ),
        )
        target_frame = RevisionedFrame(
            frame_id=BACKEND_WORLD_FRAME_ID,
            revision=prior_id,
        )
        transform_sha256 = revisioned_transform_sha256(
            source_frame,
            target_frame,
            preview_frame.target_from_source_col_major,
        )
        frame_binding = ScenePriorFrameBinding(
            contract="noesis.scene_prior.frame_binding",
            contract_version=1,
            source_frame=ScenePriorFrameRef(
                frame_id=source_frame.frame_id,
                revision=source_frame.revision,
            ),
            target_frame=ScenePriorFrameRef(
                frame_id=target_frame.frame_id,
                revision=target_frame.revision,
            ),
            source_camera_calibration_sha256=(
                preview_frame.camera_calibration.sha256
            ),
            source_world_alignment_sha256=world_to_scene_fingerprint.sha256,
            target_revision_id=preview_frame.target_revision_id,
            target_revision_metadata_sha256=(
                preview_frame.target_revision_metadata.sha256
            ),
            target_from_source_col_major=(
                preview_frame.target_from_source_col_major
            ),
            target_from_source_sha256=transform_sha256,
            source_floor_plane=ScenePriorFloorPlane(
                frame=ScenePriorFrameRef(
                    frame_id=source_frame.frame_id,
                    revision=source_frame.revision,
                ),
                normal=preview_frame.source_floor_normal,
                offset_m=preview_frame.source_floor_offset_m,
            ),
            target_floor_plane=ScenePriorFloorPlane(
                frame=ScenePriorFrameRef(
                    frame_id=target_frame.frame_id,
                    revision=target_frame.revision,
                ),
                normal=(0.0, 1.0, 0.0),
                offset_m=-float(preview_frame.target_floor_y_m),
            ),
        )

    grid_bytes = _deterministic_npz(arrays)
    metrics_payload = {
        "contract": "noesis.scene_prior.metrics",
        "contract_version": 1,
        "prior_id": prior_id,
        "site_id": cfg.site_id,
        "space_id": cfg.space_id,
        "coordinate_frame": "backend_world_m",
        "semantic_rooms": list(cfg.semantic_rooms),
        "authored_groups": authored_groups,
        "quality": quality.model_dump(mode="json"),
        "grid": grid.model_dump(mode="json"),
        "preview": preview.model_dump(mode="json"),
        "derivation": derivation.model_dump(mode="json"),
        "layers": {
            name: {
                "dtype": str(np.asarray(value).dtype),
                "shape": list(np.asarray(value).shape),
            }
            for name, value in sorted(arrays.items())
        },
        "interpretation": {
            "walkable_candidate": "advisory intersection of scan observation, floor support, and no static obstacle candidate; authored_walkable remains the separate semantic room authority",
            "obstacle_mask": "scan-derived candidate only; never semantic authority",
            "floorplan_composition": "live observed cells win; static evidence only fills live unknown cells",
        },
    }
    metrics_bytes = _canonical_json(metrics_payload)
    preview_bytes = _preview_png(preview_arrays)
    points_bytes = _review_points_glb(selected_points, selected_colors)
    artifacts = (
        _artifact("grid_npz", "grid.npz", grid_bytes),
        _artifact("metrics", "metrics.json", metrics_bytes),
        _artifact("preview", "preview.png", preview_bytes),
        _artifact("points_glb", "points.glb", points_bytes),
    )
    manifest = ScenePriorRevision(
        contract="noesis.scene_prior.revision",
        contract_version=1,
        prior_id=prior_id,
        site_id=cfg.site_id,
        space_id=cfg.space_id,
        created_at_us=created_at_us,
        created_by="noesis.scene_prior_builder",
        intended_use="shadow",
        source=source,
        alignment=alignment_fingerprint,
        authored_scene=authored_fingerprint,
        world_to_scene=world_to_scene_fingerprint,
        semantic_binding=semantic,
        grid=grid,
        preview=preview,
        derivation=derivation,
        quality=quality,
        artifacts=artifacts,
    )
    manifest_bytes = _canonical_json(manifest)
    revision_dir = output_root / "revisions" / prior_id
    files = {
        "manifest.json": manifest_bytes,
        "grid.npz": grid_bytes,
        "metrics.json": metrics_bytes,
        "preview.png": preview_bytes,
        "points.glb": points_bytes,
    }
    try:
        materialize_scene_tree(
            revision_dir, files, label=f"scene-prior revision {prior_id}"
        )
    except SceneFileError as exc:
        raise ScenePriorBuildError(str(exc)) from exc
    manifest_relative = (Path("revisions") / prior_id / "manifest.json").as_posix()
    entry = ScenePriorCatalogEntry(
        prior_id=prior_id,
        space_id=cfg.space_id,
        manifest_path=manifest_relative,
        manifest_sha256=_sha256(manifest_bytes),
        manifest_size_bytes=len(manifest_bytes),
    )
    catalog_path = output_root / "catalog.json"
    catalog = _updated_catalog(
        catalog_path,
        site_id=cfg.site_id,
        entry=entry,
        camera_ids=cfg.camera_ids,
        include_floorplan_layers=cfg.include_floorplan_layers,
        frame_binding=frame_binding,
    )
    catalog_bytes = _canonical_json(catalog)
    _atomic_write_catalog(catalog_path, catalog_bytes)
    return ScenePriorBuildResult(
        prior_id=prior_id,
        revision_dir=revision_dir,
        manifest_path=revision_dir / "manifest.json",
        catalog_path=catalog_path,
        manifest_sha256=_sha256(manifest_bytes),
        quality=quality,
    )


__all__ = [
    "ScenePriorBuildConfig",
    "ScenePriorBuildError",
    "ScenePriorBuildResult",
    "build_accepted_room_to_home_binding",
    "build_scene_prior",
    "import_accepted_room_to_home_binding",
]
