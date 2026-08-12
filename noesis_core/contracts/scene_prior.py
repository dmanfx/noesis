from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from .base import ArtifactFingerprint, ContractModel, Sha256, TimestampUs
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


class ScenePriorPreview(ContractModel):
    coordinate_frame: Literal["camera_local_ground_m"]
    units: Literal["meters"]
    orientation: Literal["row_increases_camera_forward_column_increases_camera_right"]
    reference_camera_id: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$",
    )
    camera_calibration: ArtifactFingerprint
    target_revision_metadata: ArtifactFingerprint
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
    algorithm: Literal["noesis_scene_prior_2_5d_v1"]
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
        return self


__all__ = [
    "MAX_SCENE_PRIOR_ARTIFACT_BYTES",
    "ScenePriorArtifact",
    "ScenePriorBounds",
    "ScenePriorCameraBinding",
    "ScenePriorCatalog",
    "ScenePriorCatalogEntry",
    "ScenePriorDerivation",
    "ScenePriorGrid",
    "ScenePriorPreview",
    "ScenePriorQuality",
    "ScenePriorRevision",
    "ScenePriorSemanticBinding",
    "ScenePriorSource",
]
