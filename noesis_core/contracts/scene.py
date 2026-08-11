from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from .base import ArtifactFingerprint, ContractModel, Sha256, TimestampUs
from noesis_core.scene_files import (
    MAX_AUTHORED_SCENE_BYTES,
    MAX_SCENE_ARTIFACT_BYTES,
    MAX_SCENE_AUTHORED_DEPENDENCIES,
    MAX_SCENE_CAMERAS,
    MAX_SCENE_CAMERA_ARTIFACTS,
    MAX_SCENE_RELEASE_BYTES,
    MAX_SCENE_RELEASE_FILES,
    MAX_SCENE_VALIDATION_BYTES,
    SceneFileError,
    normalized_scene_relative_path,
)


def _normalized_relative_path(value: str, *, label: str) -> str:
    try:
        return normalized_scene_relative_path(value, label=label)
    except SceneFileError as exc:
        raise ValueError(str(exc)) from exc


class SceneArtifact(ContractModel):
    role: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$",
    )
    relative_path: str = Field(min_length=1, max_length=500)
    sha256: Sha256
    size_bytes: int = Field(ge=1, le=MAX_SCENE_ARTIFACT_BYTES)

    @model_validator(mode="after")
    def _path_is_relative(self) -> "SceneArtifact":
        _normalized_relative_path(
            self.relative_path,
            label="scene artifact relative_path",
        )
        return self


class SceneCameraRevision(ContractModel):
    camera_id: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$",
    )
    revision_id: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$",
    )
    captured_at_us: TimestampUs
    calibration_bundle_sha256: Sha256
    calibration_sha256: Sha256
    model_bundle_sha256: Sha256
    model_sha256: Sha256
    manifest_sha256: Sha256
    artifact_path: str = Field(min_length=1, max_length=500)
    artifacts: tuple[SceneArtifact, ...] = Field(
        min_length=1,
        max_length=MAX_SCENE_CAMERA_ARTIFACTS,
    )

    @model_validator(mode="after")
    def _artifact_path_is_relative(self) -> "SceneCameraRevision":
        _normalized_relative_path(self.artifact_path, label="scene artifact_path")
        roles = [artifact.role for artifact in self.artifacts]
        paths = [artifact.relative_path for artifact in self.artifacts]
        if len(roles) != len(set(roles)):
            raise ValueError("scene camera artifact roles must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("scene camera artifact paths must be unique")
        return self


class SceneRelease(ContractModel):
    contract: Literal["noesis.scene.release"]
    contract_version: Literal[1]
    release_id: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$",
    )
    release_version: Literal[1]
    created_at_us: TimestampUs
    created_by: str = Field(min_length=1, max_length=160)
    calibration: ArtifactFingerprint
    model: ArtifactFingerprint
    config: ArtifactFingerprint
    authored_scene: ArtifactFingerprint
    authored_scene_path: str = Field(min_length=1, max_length=500)
    authored_scene_size_bytes: int = Field(ge=1, le=MAX_AUTHORED_SCENE_BYTES)
    authored_scene_dependencies: tuple[SceneArtifact, ...] = Field(
        max_length=MAX_SCENE_AUTHORED_DEPENDENCIES,
    )
    cohort_max_delta_us: int = Field(ge=0)
    cameras: tuple[SceneCameraRevision, ...] = Field(
        min_length=1,
        max_length=MAX_SCENE_CAMERAS,
    )
    validation_report_sha256: Sha256
    validation_report_path: str = Field(min_length=1, max_length=500)
    validation_report_size_bytes: int = Field(ge=1, le=MAX_SCENE_VALIDATION_BYTES)

    @model_validator(mode="after")
    def _cohort_is_coherent(self) -> "SceneRelease":
        _normalized_relative_path(
            self.authored_scene_path,
            label="authored_scene_path",
        )
        _normalized_relative_path(
            self.validation_report_path,
            label="validation_report_path",
        )
        dependency_roles = [item.role for item in self.authored_scene_dependencies]
        dependency_paths = [
            item.relative_path for item in self.authored_scene_dependencies
        ]
        if len(dependency_roles) != len(set(dependency_roles)):
            raise ValueError("authored scene dependency roles must be unique")
        if len(dependency_paths) != len(set(dependency_paths)):
            raise ValueError("authored scene dependency paths must be unique")
        release_prefix = ("releases", self.release_id)
        bundle_paths = [
            self.authored_scene_path,
            self.validation_report_path,
            *dependency_paths,
        ]
        if any(
            PurePosixPath(path).parts[:2] != release_prefix
            for path in bundle_paths
        ):
            raise ValueError(
                "authored scene bundle paths must be owned by the release directory"
            )
        if len(bundle_paths) != len(set(bundle_paths)):
            raise ValueError("authored scene bundle paths must be unique")
        camera_ids = [camera.camera_id for camera in self.cameras]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("scene release camera IDs must be unique")
        revision_ids = [camera.revision_id for camera in self.cameras]
        if len(revision_ids) != len(set(revision_ids)):
            raise ValueError("scene release revision IDs must be unique")
        artifact_paths = [camera.artifact_path for camera in self.cameras]
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ValueError("scene release camera artifact paths must be unique")
        if any(
            camera.calibration_bundle_sha256 != self.calibration.sha256
            for camera in self.cameras
        ):
            raise ValueError("scene camera revisions must share the release calibration bundle")
        if any(camera.model_bundle_sha256 != self.model.sha256 for camera in self.cameras):
            raise ValueError("scene camera revisions must share the release model bundle")
        captured = [camera.captured_at_us for camera in self.cameras]
        if max(captured) - min(captured) > self.cohort_max_delta_us:
            raise ValueError("scene camera revisions exceed cohort_max_delta_us")
        declared_files = (
            2
            + len(self.authored_scene_dependencies)
            + len(self.cameras)
            + sum(len(camera.artifacts) for camera in self.cameras)
        )
        if declared_files > MAX_SCENE_RELEASE_FILES:
            raise ValueError(
                f"scene release exceeds the {MAX_SCENE_RELEASE_FILES}-file limit"
            )
        declared_bytes = (
            self.authored_scene_size_bytes
            + self.validation_report_size_bytes
            + sum(item.size_bytes for item in self.authored_scene_dependencies)
            + sum(
                artifact.size_bytes
                for camera in self.cameras
                for artifact in camera.artifacts
            )
        )
        if declared_bytes > MAX_SCENE_RELEASE_BYTES:
            raise ValueError(
                f"scene release exceeds the {MAX_SCENE_RELEASE_BYTES}-byte limit"
            )
        return self
