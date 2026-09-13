from __future__ import annotations

import hashlib
import json
import struct
import math
from pathlib import PurePosixPath
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from .base import ArtifactFingerprint, ContractModel, Sha256, TimestampUs
from noesis_core.coordinate_frames import (
    BACKEND_WORLD_FRAME_ID,
    CoordinateFrameError,
    MetricFloorPlane,
    RevisionedFrame,
    RevisionedFrameTransform,
)
from noesis_core.scene_files import SceneFileError, normalized_scene_relative_path


MAX_SCENE_PRIOR_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_SCENE_PRIOR_ARTIFACTS = 16
MAX_SCENE_PRIOR_REVISIONS = 256
MAX_SCENE_PRIOR_CAMERA_BINDINGS = 256
MAX_SCENE_PRIOR_GRID_DIMENSION = 16_384
MAX_SCENE_PRIOR_GRID_CELLS = 16 * 1024 * 1024


def _identifier(value: str, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text


def _relative_path(value: str, *, label: str) -> str:
    try:
        return normalized_scene_relative_path(value, label=label)
    except SceneFileError as exc:
        raise ValueError(str(exc)) from exc


class ScenePriorBounds(ContractModel):
    min_x: float
    max_x: float
    min_z: float
    max_z: float

    @model_validator(mode="after")
    def _finite_positive_extent(self) -> "ScenePriorBounds":
        values = (self.min_x, self.max_x, self.min_z, self.max_z)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("scene-prior bounds must be finite")
        if self.max_x <= self.min_x or self.max_z <= self.min_z:
            raise ValueError("scene-prior bounds must have positive X/Z extents")
        return self


class ScenePriorGrid(ContractModel):
    coordinate_frame: Literal["backend_world_m"]
    units: Literal["meters"]
    orientation: Literal["row_increases_positive_z_column_increases_positive_x"]
    bounds: ScenePriorBounds
    resolution_m: float = Field(gt=0.0, le=2.0)
    rows: int = Field(ge=1, le=MAX_SCENE_PRIOR_GRID_DIMENSION)
    columns: int = Field(ge=1, le=MAX_SCENE_PRIOR_GRID_DIMENSION)

    @model_validator(mode="after")
    def _shape_matches_bounds(self) -> "ScenePriorGrid":
        if self.rows * self.columns > MAX_SCENE_PRIOR_GRID_CELLS:
            raise ValueError("scene-prior grid exceeds the bounded cell count")
        expected_x = float(self.columns) * float(self.resolution_m)
        expected_z = float(self.rows) * float(self.resolution_m)
        actual_x = float(self.bounds.max_x) - float(self.bounds.min_x)
        actual_z = float(self.bounds.max_z) - float(self.bounds.min_z)
        tolerance = max(1e-6, float(self.resolution_m) * 1e-5)
        if (
            abs(actual_x - expected_x) > tolerance
            or abs(actual_z - expected_z) > tolerance
        ):
            raise ValueError("scene-prior grid shape does not match bounds/resolution")
        return self


class ScenePriorCameraMapLock(ContractModel):
    """Revision-bound static-camera registration residual applied in target world."""

    contract: Literal["noesis.scene_prior.camera_map_lock"]
    contract_version: Literal[1]
    evidence: ArtifactFingerprint
    camera_id: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$",
    )
    yaw_correction_deg: float = Field(ge=-45.0, le=45.0)
    rotation_pivot: Literal["camera_optical_center_target_world_m"]
    pivot_world_m: tuple[float, float, float]
    base_camera_forward_world_xz: tuple[float, float]
    corrected_camera_forward_world_xz: tuple[float, float]

    @model_validator(mode="after")
    def _valid_map_lock(self) -> "ScenePriorCameraMapLock":
        values = (
            self.yaw_correction_deg,
            *self.pivot_world_m,
            *self.base_camera_forward_world_xz,
            *self.corrected_camera_forward_world_xz,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("scene-prior camera map lock must be finite")
        for axis in (
            self.base_camera_forward_world_xz,
            self.corrected_camera_forward_world_xz,
        ):
            if abs(math.hypot(*axis) - 1.0) > 1e-6:
                raise ValueError(
                    "scene-prior camera map-lock forward axes must be unit vectors"
                )
        return self


class ScenePriorPreview(ContractModel):
    coordinate_frame: Literal["camera_local_ground_m"]
    units: Literal["meters"]
    # New revisions describe the actual PNG/raster addressing.  The legacy
    # value remains readable because deployed immutable revisions used it for
    # the pre-image numeric grid even though preview.png was vertically
    # converted to row-zero-far addressing.
    orientation: Literal[
        "row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x",
        "row_increases_camera_forward_column_increases_camera_right",
    ]
    reference_camera_id: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$",
    )
    camera_calibration: ArtifactFingerprint
    target_revision_metadata: ArtifactFingerprint
    camera_map_lock: ScenePriorCameraMapLock | None = None
    camera_pose_anchor: ArtifactFingerprint | None = None
    camera_position_world_m: tuple[float, float, float]
    camera_right_world_xz: tuple[float, float]
    camera_forward_world_xz: tuple[float, float]
    bounds: ScenePriorBounds
    resolution_m: float = Field(gt=0.0, le=2.0)
    rows: int = Field(ge=1, le=MAX_SCENE_PRIOR_GRID_DIMENSION)
    columns: int = Field(ge=1, le=MAX_SCENE_PRIOR_GRID_DIMENSION)

    @model_validator(mode="after")
    def _valid_camera_ground_frame(self) -> "ScenePriorPreview":
        values = (
            *self.camera_position_world_m,
            *self.camera_right_world_xz,
            *self.camera_forward_world_xz,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("scene-prior preview frame must be finite")
        right_norm = math.hypot(*self.camera_right_world_xz)
        forward_norm = math.hypot(*self.camera_forward_world_xz)
        dot = sum(
            float(right) * float(forward)
            for right, forward in zip(
                self.camera_right_world_xz,
                self.camera_forward_world_xz,
            )
        )
        if abs(right_norm - 1.0) > 1e-6 or abs(forward_norm - 1.0) > 1e-6:
            raise ValueError("scene-prior preview ground axes must be unit vectors")
        if abs(dot) > 1e-6:
            raise ValueError("scene-prior preview ground axes must be orthogonal")
        if self.rows * self.columns > MAX_SCENE_PRIOR_GRID_CELLS:
            raise ValueError("scene-prior preview exceeds the bounded cell count")
        expected_x = float(self.columns) * float(self.resolution_m)
        expected_z = float(self.rows) * float(self.resolution_m)
        actual_x = float(self.bounds.max_x) - float(self.bounds.min_x)
        actual_z = float(self.bounds.max_z) - float(self.bounds.min_z)
        tolerance = max(1e-6, float(self.resolution_m) * 1e-5)
        if (
            abs(actual_x - expected_x) > tolerance
            or abs(actual_z - expected_z) > tolerance
        ):
            raise ValueError(
                "scene-prior preview shape does not match bounds/resolution"
            )
        return self


class ScenePriorDerivation(ContractModel):
    algorithm: Literal[
        "noesis_scene_prior_2_5d_v1",
        "noesis_scene_prior_2_5d_full_evidence_v2",
    ]
    floor_y_m: float
    floor_support_band_m: float = Field(gt=0.0, le=1.0)
    obstacle_min_height_m: float = Field(gt=0.0, le=2.0)
    obstacle_max_height_m: float = Field(gt=0.0, le=10.0)
    obstacle_min_support: int = Field(ge=1, le=1_000_000)
    max_source_height_m: float = Field(gt=0.0, le=20.0)

    @model_validator(mode="after")
    def _coherent_thresholds(self) -> "ScenePriorDerivation":
        values = (
            self.floor_y_m,
            self.floor_support_band_m,
            self.obstacle_min_height_m,
            self.obstacle_max_height_m,
            self.max_source_height_m,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("scene-prior derivation thresholds must be finite")
        if self.obstacle_max_height_m <= self.obstacle_min_height_m:
            raise ValueError("obstacle_max_height_m must exceed obstacle_min_height_m")
        if self.max_source_height_m < self.obstacle_max_height_m:
            raise ValueError("max_source_height_m must cover obstacle_max_height_m")
        return self


class ScenePriorSource(ContractModel):
    source_type: Literal[
        "mapanything_multiview_room_walk",
        "conditioned_multimodel_room_walk",
    ]
    bundle_id: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$",
    )
    bundle_schema: str = Field(min_length=1, max_length=160)
    bundle_manifest_sha256: Sha256
    reference_sha256: Sha256
    capture_id: str = Field(min_length=1, max_length=200)
    captured_at_us: TimestampUs
    model: str = Field(min_length=1, max_length=200)


class ScenePriorSemanticBinding(ContractModel):
    room_labels: tuple[str, ...] = Field(min_length=1, max_length=64)
    authored_groups: tuple[str, ...] = Field(min_length=1, max_length=256)
    room_group_map: ArtifactFingerprint

    @model_validator(mode="after")
    def _unique_semantics(self) -> "ScenePriorSemanticBinding":
        labels = tuple(
            _identifier(value, label="room label") for value in self.room_labels
        )
        groups = tuple(
            _identifier(value, label="authored group") for value in self.authored_groups
        )
        if len(labels) != len(set(labels)) or len(groups) != len(set(groups)):
            raise ValueError("scene-prior semantic labels/groups must be unique")
        return self


class ScenePriorQuality(ContractModel):
    passed: bool
    source_point_count: int = Field(ge=0)
    selected_point_count: int = Field(ge=0)
    authored_cell_count: int = Field(ge=1)
    observed_cell_count: int = Field(ge=0)
    floor_supported_cell_count: int = Field(ge=0)
    obstacle_cell_count: int = Field(ge=0)
    authored_observed_fraction: float = Field(ge=0.0, le=1.0)
    authored_floor_supported_fraction: float = Field(ge=0.0, le=1.0)
    alignment_status: str = Field(min_length=1, max_length=80)

    @model_validator(mode="after")
    def _counts_are_bounded_by_grid(self) -> "ScenePriorQuality":
        for value in (
            self.observed_cell_count,
            self.floor_supported_cell_count,
            self.obstacle_cell_count,
        ):
            if value > self.authored_cell_count:
                raise ValueError("scene-prior quality counts exceed authored cells")
        if self.selected_point_count > self.source_point_count:
            raise ValueError("selected_point_count exceeds source_point_count")
        return self


class ScenePriorArtifact(ContractModel):
    role: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$",
    )
    relative_path: str = Field(min_length=1, max_length=500)
    sha256: Sha256
    size_bytes: int = Field(ge=1, le=MAX_SCENE_PRIOR_ARTIFACT_BYTES)

    @model_validator(mode="after")
    def _normalized_path(self) -> "ScenePriorArtifact":
        _relative_path(self.relative_path, label="scene-prior artifact path")
        return self


class ScenePriorRevision(ContractModel):
    contract: Literal["noesis.scene_prior.revision"]
    contract_version: Literal[1]
    prior_id: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$",
    )
    site_id: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$",
    )
    space_id: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$",
    )
    created_at_us: TimestampUs
    created_by: str = Field(min_length=1, max_length=160)
    intended_use: Literal["shadow"]
    source: ScenePriorSource
    alignment: ArtifactFingerprint
    authored_scene: ArtifactFingerprint
    world_to_scene: ArtifactFingerprint
    semantic_binding: ScenePriorSemanticBinding
    grid: ScenePriorGrid
    preview: ScenePriorPreview | None = None
    derivation: ScenePriorDerivation
    quality: ScenePriorQuality
    artifacts: tuple[ScenePriorArtifact, ...] = Field(
        min_length=1,
        max_length=MAX_SCENE_PRIOR_ARTIFACTS,
    )

    @model_validator(mode="after")
    def _artifact_inventory_is_complete(self) -> "ScenePriorRevision":
        roles = [artifact.role for artifact in self.artifacts]
        paths = [artifact.relative_path for artifact in self.artifacts]
        if len(roles) != len(set(roles)) or len(paths) != len(set(paths)):
            raise ValueError("scene-prior artifact roles and paths must be unique")
        required = {"grid_npz", "metrics", "preview", "points_glb"}
        missing = required.difference(roles)
        if missing:
            raise ValueError(
                "scene-prior artifacts are missing required roles: "
                + ", ".join(sorted(missing))
            )
        return self


class ScenePriorCatalogEntry(ContractModel):
    prior_id: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$",
    )
    space_id: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$",
    )
    manifest_path: str = Field(min_length=1, max_length=500)
    manifest_sha256: Sha256
    manifest_size_bytes: int = Field(ge=1, le=4 * 1024 * 1024)

    @model_validator(mode="after")
    def _manifest_is_catalog_relative(self) -> "ScenePriorCatalogEntry":
        path = _relative_path(self.manifest_path, label="scene-prior manifest path")
        if PurePosixPath(path).name != "manifest.json":
            raise ValueError("scene-prior catalog entries must reference manifest.json")
        return self


class ScenePriorFrameRef(ContractModel):
    frame_id: Literal["backend_world_m"]
    revision: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$",
    )


class ScenePriorFloorPlane(ContractModel):
    frame: ScenePriorFrameRef
    normal: tuple[float, float, float]
    offset_m: float

    @model_validator(mode="after")
    def _valid_metric_plane(self) -> "ScenePriorFloorPlane":
        try:
            MetricFloorPlane(
                frame=RevisionedFrame(
                    frame_id=self.frame.frame_id,
                    revision=self.frame.revision,
                ),
                normal=tuple(float(value) for value in self.normal),
                offset_m=float(self.offset_m),
            )
        except CoordinateFrameError as exc:
            raise ValueError(str(exc)) from exc
        return self


def metric_frame_revision_sha256(
    frame_id: str,
    floor_normal: tuple[float, float, float],
    floor_offset_m: float,
    artifact_sha256s: tuple[str, ...],
) -> str:
    """Derive a metric-frame revision from physical artifacts and its floor plane.

    The floor plane is part of the coordinate identity.  Two registrations
    cannot therefore reuse one revision while naming different physical floor
    planes, even when their artifact inventory happens to match.
    """

    normalized_frame = str(frame_id).strip()
    normal = tuple(float(value) for value in floor_normal)
    offset = float(floor_offset_m)
    if len(normal) != 3 or not all(math.isfinite(value) for value in normal):
        raise ValueError("metric frame floor normal must contain three finite values")
    if not math.isfinite(offset):
        raise ValueError("metric frame floor offset must be finite")
    payload = {
        "artifact_sha256s": [str(value).lower() for value in artifact_sha256s],
        "floor_normal": list(normal),
        "floor_offset_m": offset,
        "frame_id": normalized_frame,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class ScenePriorMetricFrame(ContractModel):
    """Accepted metric coordinate identity independent of scene artifacts."""

    contract: Literal["noesis.scene_prior.metric_frame"]
    contract_version: Literal[1]
    frame: ScenePriorFrameRef
    units: Literal["meters"]
    floor_plane: ScenePriorFloorPlane
    physical_provenance: tuple[ArtifactFingerprint, ...] = Field(
        min_length=1,
        max_length=16,
    )
    accepted: bool

    @model_validator(mode="after")
    def _valid_metric_identity(self) -> "ScenePriorMetricFrame":
        if self.frame.frame_id != BACKEND_WORLD_FRAME_ID:
            raise ValueError("metric frame must use backend_world_m")
        if self.floor_plane.frame != self.frame:
            raise ValueError("metric frame floor plane belongs to another frame")
        roles = [item.role for item in self.physical_provenance]
        if len(roles) != len(set(roles)):
            raise ValueError("metric frame physical provenance roles must be unique")
        required_roles = {
            "camera_calibration_physical",
            "world_alignment_physical",
        }
        if not required_roles.issubset(roles):
            raise ValueError(
                "metric frame physical provenance must include camera_calibration_physical and world_alignment_physical"
            )
        expected_revision = metric_frame_revision_sha256(
            self.frame.frame_id,
            self.floor_plane.normal,
            self.floor_plane.offset_m,
            tuple(item.sha256 for item in self.physical_provenance),
        )
        if expected_revision != self.frame.revision:
            raise ValueError(
                "metric frame revision does not match physical provenance"
            )
        if not self.accepted:
            raise ValueError("only accepted metric frames may enter a v2 binding")
        return self


def world_to_scene_presentation_sha256(
    world_frame: ScenePriorFrameRef,
    render_frame: str,
    world_to_scene_col_major: tuple[float, ...],
) -> str:
    """Return content identity for a target-frame presentation mapping."""

    return _presentation_sha256(
        world_frame,
        render_frame,
        world_to_scene_col_major,
        purpose="world_to_scene",
    )


def scene_revision_sha256(
    world_frame: ScenePriorFrameRef,
    render_frame: str,
    world_to_scene_col_major: tuple[float, ...],
) -> str:
    """Return the independent authored-scene revision for one mapping."""

    return _presentation_sha256(
        world_frame,
        render_frame,
        world_to_scene_col_major,
        purpose="scene_revision",
    )


def _presentation_sha256(
    world_frame: ScenePriorFrameRef,
    render_frame: str,
    world_to_scene_col_major: tuple[float, ...],
    *,
    purpose: str,
) -> str:
    """Hash frame identity plus exact IEEE-754 matrix bytes.

    The explicit little-endian Float64 representation avoids language-specific
    decimal rounding differences while preserving signed-zero normalization.
    ``digest_version`` is part of the bytes so a future canonical encoding can
    be introduced without silently accepting a different identity.
    """

    values = tuple(float(value) for value in world_to_scene_col_major)
    if len(values) != 16:
        raise ValueError("world-to-scene presentation requires 16 matrix values")
    if not all(np.isfinite(value) for value in values):
        raise ValueError("world-to-scene presentation matrix must be finite")
    header = {
        "digest_version": 2,
        "purpose": purpose,
        "render_frame": str(render_frame),
        "world_frame": {
            "frame_id": world_frame.frame_id,
            "revision": world_frame.revision,
        },
    }
    matrix_bytes = struct.pack(
        "<16d",
        *(0.0 if value == 0.0 else value for value in values),
    )
    payload = (
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\x00"
        + matrix_bytes
    )
    return hashlib.sha256(payload).hexdigest()


class ScenePriorWorldToScenePresentation(ContractModel):
    """Renderer mapping owned by a validated target metric frame revision."""

    contract: Literal["noesis.scene_prior.world_to_scene_presentation"]
    contract_version: Literal[1]
    world_frame: ScenePriorFrameRef
    render_frame: Literal["menon_scene"]
    world_to_scene_col_major: tuple[float, ...] = Field(
        min_length=16,
        max_length=16,
    )
    world_to_scene_sha256: Sha256
    scene_revision_id: Sha256
    source_transform_sha256s: tuple[Sha256, ...] = Field(
        min_length=1,
        max_length=256,
    )
    provenance: ArtifactFingerprint
    accepted: bool

    @model_validator(mode="after")
    def _valid_presentation_mapping(self) -> "ScenePriorWorldToScenePresentation":
        if self.world_frame.frame_id != BACKEND_WORLD_FRAME_ID:
            raise ValueError("presentation mapping must target backend_world_m")
        values = tuple(float(value) for value in self.world_to_scene_col_major)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("presentation mapping must be finite")
        matrix = np.asarray(values, dtype=np.float64).reshape((4, 4), order="F")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
            raise ValueError("presentation mapping must be affine")
        determinant = float(np.linalg.det(matrix[:3, :3]))
        if not math.isfinite(determinant) or determinant <= 1e-12:
            raise ValueError("presentation mapping must have a proper positive orientation")
        linear = matrix[:3, :3]
        gram = linear.T @ linear
        scale_squared = float(np.trace(gram) / 3.0)
        if not math.isfinite(scale_squared) or scale_squared <= 1e-12:
            raise ValueError("presentation mapping must have a positive uniform scale")
        if not np.allclose(
            gram,
            np.eye(3, dtype=np.float64) * scale_squared,
            atol=max(2e-5, scale_squared * 2e-5),
            rtol=2e-5,
        ):
            raise ValueError("presentation mapping must be a uniform similarity")
        expected = world_to_scene_presentation_sha256(
            self.world_frame,
            self.render_frame,
            values,
        )
        if self.world_to_scene_sha256 != expected:
            raise ValueError(
                "presentation mapping digest does not match its frame and matrix"
            )
        expected_scene_revision = scene_revision_sha256(
            self.world_frame,
            self.render_frame,
            values,
        )
        if self.scene_revision_id != expected_scene_revision:
            raise ValueError(
                "presentation scene revision does not match its frame and matrix"
            )
        if len(set(self.source_transform_sha256s)) != len(self.source_transform_sha256s):
            raise ValueError("presentation source transform digests must be unique")
        if not self.accepted:
            raise ValueError("only accepted presentation mappings may be transported")
        return self


class ScenePriorFrameBinding(ContractModel):
    """Immutable raw-to-active world revision edge for one camera binding."""

    contract: Literal["noesis.scene_prior.frame_binding"]
    contract_version: Literal[1, 2]
    source_frame: ScenePriorFrameRef
    target_frame: ScenePriorFrameRef
    source_camera_calibration_sha256: Sha256
    source_world_alignment_sha256: Sha256
    target_revision_id: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$",
    )
    target_revision_metadata_sha256: Sha256
    target_from_source_col_major: tuple[float, ...] = Field(
        min_length=16,
        max_length=16,
    )
    target_from_source_sha256: Sha256
    source_floor_plane: ScenePriorFloorPlane
    target_floor_plane: ScenePriorFloorPlane
    # v2 keeps artifact/provenance identities separate from the accepted
    # coordinate frame and its renderer mapping.  The v1 fields above remain
    # required so existing immutable catalogs retain their exact checks.
    metric_frame: ScenePriorMetricFrame | None = None
    artifact_revision_id: str | None = Field(default=None, max_length=200)
    source_camera_calibration_physical_sha256: Sha256 | None = None
    source_world_alignment_physical_sha256: Sha256 | None = None
    presentation: ScenePriorWorldToScenePresentation | None = None

    @model_validator(mode="after")
    def _valid_revisioned_transform(self) -> "ScenePriorFrameBinding":
        if self.source_frame.frame_id != BACKEND_WORLD_FRAME_ID:
            raise ValueError("scene-prior source frame must be backend_world_m")
        if self.target_frame.frame_id != BACKEND_WORLD_FRAME_ID:
            raise ValueError("scene-prior target frame must be backend_world_m")
        if self.contract_version == 2:
            if self.metric_frame is None:
                raise ValueError("v2 frame binding requires an accepted metric frame")
            if self.target_frame != self.metric_frame.frame:
                raise ValueError(
                    "v2 target frame must equal the accepted metric frame"
                )
            if self.target_floor_plane != self.metric_frame.floor_plane:
                raise ValueError(
                    "v2 target floor plane must equal the accepted metric frame floor"
                )
            if not self.artifact_revision_id:
                raise ValueError("v2 frame binding requires artifact_revision_id")
            if (
                self.source_camera_calibration_physical_sha256 is None
                or self.source_world_alignment_physical_sha256 is None
            ):
                raise ValueError(
                    "v2 frame binding requires physical calibration provenance"
                )
            if self.presentation is not None and (
                self.presentation.world_frame != self.target_frame
            ):
                raise ValueError(
                    "v2 presentation mapping must target the binding frame"
                )
        elif any(
            value is not None
            for value in (
                self.metric_frame,
                self.artifact_revision_id,
                self.source_camera_calibration_physical_sha256,
                self.source_world_alignment_physical_sha256,
                self.presentation,
            )
        ):
            raise ValueError("v1 frame binding cannot carry v2 identities")
        try:
            source = RevisionedFrame(
                frame_id=self.source_frame.frame_id,
                revision=self.source_frame.revision,
            )
            target = RevisionedFrame(
                frame_id=self.target_frame.frame_id,
                revision=self.target_frame.revision,
            )
            RevisionedFrameTransform(
                source_frame=source,
                target_frame=target,
                target_from_source_col_major=tuple(
                    float(value) for value in self.target_from_source_col_major
                ),
                transform_sha256=self.target_from_source_sha256,
                source_floor_plane=MetricFloorPlane(
                    frame=source,
                    normal=tuple(
                        float(value) for value in self.source_floor_plane.normal
                    ),
                    offset_m=float(self.source_floor_plane.offset_m),
                ),
                target_floor_plane=MetricFloorPlane(
                    frame=target,
                    normal=tuple(
                        float(value) for value in self.target_floor_plane.normal
                    ),
                    offset_m=float(self.target_floor_plane.offset_m),
                ),
            )
        except CoordinateFrameError as exc:
            raise ValueError(str(exc)) from exc
        return self

    def frame_transform(self) -> RevisionedFrameTransform:
        source = RevisionedFrame(
            frame_id=self.source_frame.frame_id,
            revision=self.source_frame.revision,
        )
        target = RevisionedFrame(
            frame_id=self.target_frame.frame_id,
            revision=self.target_frame.revision,
        )
        return RevisionedFrameTransform(
            source_frame=source,
            target_frame=target,
            target_from_source_col_major=tuple(
                float(value) for value in self.target_from_source_col_major
            ),
            transform_sha256=self.target_from_source_sha256,
            source_floor_plane=MetricFloorPlane(
                frame=source,
                normal=tuple(float(value) for value in self.source_floor_plane.normal),
                offset_m=float(self.source_floor_plane.offset_m),
            ),
            target_floor_plane=MetricFloorPlane(
                frame=target,
                normal=tuple(float(value) for value in self.target_floor_plane.normal),
                offset_m=float(self.target_floor_plane.offset_m),
            ),
        )


class ScenePriorCameraBinding(ContractModel):
    camera_id: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$",
    )
    space_id: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$",
    )
    prior_id: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$",
    )
    mode: Literal["shadow"]
    include_floorplan_layers: bool = True
    frame_binding: ScenePriorFrameBinding | None = None


class ScenePriorCatalog(ContractModel):
    contract: Literal["noesis.scene_prior.catalog"]
    contract_version: Literal[1]
    site_id: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$",
    )
    revisions: tuple[ScenePriorCatalogEntry, ...] = Field(
        min_length=1,
        max_length=MAX_SCENE_PRIOR_REVISIONS,
    )
    camera_bindings: tuple[ScenePriorCameraBinding, ...] = Field(
        max_length=MAX_SCENE_PRIOR_CAMERA_BINDINGS,
    )

    @model_validator(mode="after")
    def _bindings_reference_exact_revisions(self) -> "ScenePriorCatalog":
        entries = {entry.prior_id: entry for entry in self.revisions}
        if len(entries) != len(self.revisions):
            raise ValueError("scene-prior catalog revision IDs must be unique")
        paths = [entry.manifest_path for entry in self.revisions]
        if len(paths) != len(set(paths)):
            raise ValueError("scene-prior catalog manifest paths must be unique")
        camera_ids = [binding.camera_id for binding in self.camera_bindings]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("scene-prior camera bindings must be unique")
        for binding in self.camera_bindings:
            entry = entries.get(binding.prior_id)
            if entry is None:
                raise ValueError(
                    f"scene-prior camera {binding.camera_id} references an unknown revision"
                )
            if entry.space_id != binding.space_id:
                raise ValueError(
                    f"scene-prior camera {binding.camera_id} space does not match its revision"
                )
            frame_binding = binding.frame_binding
            if frame_binding is None:
                continue
            if frame_binding.contract_version == 1:
                if frame_binding.target_frame.revision != binding.prior_id:
                    raise ValueError(
                        f"scene-prior camera {binding.camera_id} frame revision does not match its active prior"
                    )
            elif frame_binding.artifact_revision_id != binding.prior_id:
                raise ValueError(
                    f"scene-prior camera {binding.camera_id} artifact revision does not match its active prior"
                )
        v2_frames = [
            binding.frame_binding.metric_frame
            for binding in self.camera_bindings
            if binding.frame_binding is not None
            and binding.frame_binding.contract_version == 2
        ]
        if v2_frames:
            first = v2_frames[0]
            assert first is not None
            for metric_frame in v2_frames[1:]:
                assert metric_frame is not None
                if metric_frame != first:
                    raise ValueError(
                        "v2 camera bindings must share one accepted metric frame"
                    )
        return self


__all__ = [
    "MAX_SCENE_PRIOR_ARTIFACT_BYTES",
    "ScenePriorArtifact",
    "ScenePriorBounds",
    "ScenePriorCameraMapLock",
    "ScenePriorCameraBinding",
    "ScenePriorCatalog",
    "ScenePriorCatalogEntry",
    "ScenePriorDerivation",
    "ScenePriorFloorPlane",
    "ScenePriorFrameBinding",
    "ScenePriorFrameRef",
    "ScenePriorGrid",
    "ScenePriorPreview",
    "ScenePriorQuality",
    "ScenePriorRevision",
    "ScenePriorSemanticBinding",
    "ScenePriorSource",
    "ScenePriorMetricFrame",
    "ScenePriorWorldToScenePresentation",
    "scene_revision_sha256",
    "metric_frame_revision_sha256",
    "world_to_scene_presentation_sha256",
]
