"""Canonical coordinate conversions shared by Scene Prior and PCF tooling.

Metric registration stays in proper right-handed world frames.  The
``camera_local_ground_m`` frame is a presentation/addressing frame with +X to
camera-right, +Y above the floor, and +Z camera-forward.  For a calibrated
OpenCV camera (+X right, +Y down, +Z forward), changing down to world-up makes
the world-to-display linear map improper (normally determinant -1).  That map
must never be used as a registration transform or written back into backend
geometry.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np


BACKEND_GRID_ORIENTATION = "row_increases_positive_z_column_increases_positive_x"
CAMERA_LOCAL_RASTER_ORIENTATION = (
    "row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x"
)
BACKEND_WORLD_FRAME_ID = "backend_world_m"
_FRAME_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLANE_TRANSFORM_TOLERANCE_M = 0.01


class CoordinateFrameError(ValueError):
    """Raised when a coordinate transform cannot satisfy the frame contract."""


def _frame_token(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if _FRAME_TOKEN_RE.fullmatch(text) is None:
        raise CoordinateFrameError(f"{label} must be a portable frame token")
    return text


def _sha256(value: object, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(text) is None:
        raise CoordinateFrameError(f"{label} must be a lowercase SHA-256 digest")
    return text


@dataclass(frozen=True)
class RevisionedFrame:
    """Immutable identity for one metric coordinate frame revision."""

    frame_id: str
    revision: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frame_id",
            _frame_token(self.frame_id, label="frame_id"),
        )
        object.__setattr__(
            self,
            "revision",
            _frame_token(self.revision, label="frame revision"),
        )


@dataclass(frozen=True)
class MetricFloorPlane:
    """Unit-normal plane ``normal dot point + offset_m = 0`` in one frame."""

    frame: RevisionedFrame
    normal: tuple[float, float, float]
    offset_m: float

    def __post_init__(self) -> None:
        try:
            normal = tuple(float(value) for value in self.normal)
            offset = float(self.offset_m)
        except (TypeError, ValueError) as exc:
            raise CoordinateFrameError("floor plane must be numeric") from exc
        if len(normal) != 3 or not all(math.isfinite(value) for value in normal):
            raise CoordinateFrameError("floor plane normal must contain three finite values")
        if not math.isfinite(offset):
            raise CoordinateFrameError("floor plane offset must be finite")
        magnitude = math.sqrt(sum(value * value for value in normal))
        if not math.isclose(magnitude, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise CoordinateFrameError("floor plane normal must be unit length")
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "offset_m", offset)

    @property
    def horizontal_y_m(self) -> float:
        nx, ny, nz = self.normal
        if abs(nx) > 1e-6 or abs(nz) > 1e-6 or ny <= 1.0 - 1e-6:
            raise CoordinateFrameError("floor plane is not horizontal positive-Y")
        return float(-self.offset_m / ny)


def revisioned_frame_sha256(
    frame_id: str,
    *,
    artifact_sha256s: Sequence[str],
) -> str:
    """Derive a stable frame revision from the exact owning artifacts."""

    normalized_frame = _frame_token(frame_id, label="frame_id")
    normalized_artifacts = [
        _sha256(value, label="frame artifact") for value in artifact_sha256s
    ]
    if not normalized_artifacts:
        raise CoordinateFrameError("frame revision requires at least one artifact")
    payload = {
        "artifact_sha256s": normalized_artifacts,
        "frame_id": normalized_frame,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def revisioned_transform_sha256(
    source_frame: RevisionedFrame,
    target_frame: RevisionedFrame,
    target_from_source_col_major: Sequence[float],
) -> str:
    """Content identity for a directed, revision-bound metric transform."""

    values = np.asarray(target_from_source_col_major, dtype=np.float64)
    if values.size != 16 or not np.isfinite(values).all():
        raise CoordinateFrameError("frame transform must contain 16 finite values")
    payload = {
        "source_frame": {
            "frame_id": source_frame.frame_id,
            "revision": source_frame.revision,
        },
        "target_frame": {
            "frame_id": target_frame.frame_id,
            "revision": target_frame.revision,
        },
        "target_from_source_col_major": [float(value) for value in values],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_rigid_transform(
    values: Sequence[float] | np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.size != 16 or not np.isfinite(raw).all():
        raise CoordinateFrameError(f"{label} must contain 16 finite values")
    transform = raw.reshape((4, 4), order="F") if raw.ndim != 2 else raw
    if transform.shape != (4, 4):
        raise CoordinateFrameError(f"{label} must be a 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise CoordinateFrameError(f"{label} must be affine")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5):
        raise CoordinateFrameError(f"{label} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-5):
        raise CoordinateFrameError(f"{label} rotation must be proper")
    return transform


@dataclass(frozen=True)
class RevisionedFrameTransform:
    """Proper rigid edge and its floor-plane contract.

    The edge maps metric points from ``source_frame`` into ``target_frame``.
    Camera extrinsics stay calibration-owned; callers must explicitly request
    :meth:`camera_from_target_col_major` for a target-frame world estimator.
    """

    source_frame: RevisionedFrame
    target_frame: RevisionedFrame
    target_from_source_col_major: tuple[float, ...]
    transform_sha256: str
    source_floor_plane: MetricFloorPlane
    target_floor_plane: MetricFloorPlane

    def __post_init__(self) -> None:
        transform = _validated_rigid_transform(
            self.target_from_source_col_major,
            label="target-from-source transform",
        )
        values = tuple(float(value) for value in transform.flatten(order="F"))
        digest = _sha256(self.transform_sha256, label="frame transform digest")
        expected = revisioned_transform_sha256(
            self.source_frame,
            self.target_frame,
            values,
        )
        if digest != expected:
            raise CoordinateFrameError(
                "frame transform digest does not match its identities and matrix"
            )
        if self.source_floor_plane.frame != self.source_frame:
            raise CoordinateFrameError("source floor plane belongs to another frame")
        if self.target_floor_plane.frame != self.target_frame:
            raise CoordinateFrameError("target floor plane belongs to another frame")

        source_plane = np.asarray(
            [*self.source_floor_plane.normal, self.source_floor_plane.offset_m],
            dtype=np.float64,
        )
        try:
            transformed_plane = np.linalg.inv(transform).T @ source_plane
        except np.linalg.LinAlgError as exc:
            raise CoordinateFrameError("frame transform is singular") from exc
        magnitude = float(np.linalg.norm(transformed_plane[:3]))
        if not math.isfinite(magnitude) or magnitude <= 1e-9:
            raise CoordinateFrameError("transformed floor plane is degenerate")
        transformed_plane /= magnitude
        target_plane = np.asarray(
            [*self.target_floor_plane.normal, self.target_floor_plane.offset_m],
            dtype=np.float64,
        )
        if float(np.dot(transformed_plane[:3], target_plane[:3])) < 0.0:
            transformed_plane *= -1.0
        if not np.allclose(transformed_plane[:3], target_plane[:3], atol=2e-5):
            raise CoordinateFrameError(
                "frame transform does not map the declared source floor normal"
            )
        if abs(float(transformed_plane[3] - target_plane[3])) > _PLANE_TRANSFORM_TOLERANCE_M:
            raise CoordinateFrameError(
                "frame transform does not map the declared source floor offset"
            )
        object.__setattr__(self, "target_from_source_col_major", values)
        object.__setattr__(self, "transform_sha256", digest)

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(
            self.target_from_source_col_major,
            dtype=np.float64,
        ).reshape((4, 4), order="F")

    def camera_from_target_col_major(
        self,
        camera_from_source_col_major: Sequence[float],
    ) -> tuple[float, ...]:
        """Compose raw camera-from-source E with this explicit frame edge."""

        camera_from_source = _validated_rigid_transform(
            camera_from_source_col_major,
            label="camera-from-source extrinsics",
        )
        try:
            camera_from_target = camera_from_source @ np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError as exc:
            raise CoordinateFrameError("frame transform is singular") from exc
        camera_from_target = _validated_rigid_transform(
            camera_from_target,
            label="camera-from-target extrinsics",
        )
        return tuple(
            float(value) for value in camera_from_target.flatten(order="F")
        )


@dataclass(frozen=True)
class CameraGroundFrame:
    """Ground-projected OpenCV camera axes expressed in backend world."""

    camera_world_m: np.ndarray
    camera_right_world: np.ndarray
    camera_forward_world: np.ndarray

    def world_to_camera_local_display_matrix(self, floor_y_m: float) -> np.ndarray:
        """Return the presentation-only backend-world to camera-ground map."""

        floor_y = float(floor_y_m)
        if not math.isfinite(floor_y):
            raise CoordinateFrameError("camera-ground floor height must be finite")
        camera = np.asarray(self.camera_world_m, dtype=np.float64)
        right = np.asarray(self.camera_right_world, dtype=np.float64)
        forward = np.asarray(self.camera_forward_world, dtype=np.float64)
        return np.asarray(
            [
                [*right, -float(np.dot(right, camera))],
                [0.0, 1.0, 0.0, -floor_y],
                [*forward, -float(np.dot(forward, camera))],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def _validated_camera_to_world(camera_to_world: np.ndarray) -> np.ndarray:
    transform = np.asarray(camera_to_world, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise CoordinateFrameError("camera-to-world must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise CoordinateFrameError("camera-to-world must be affine")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5):
        raise CoordinateFrameError("camera-to-world rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-5):
        raise CoordinateFrameError("camera-to-world rotation must be proper")
    return transform


def camera_ground_frame_from_camera_to_world(
    camera_to_world: np.ndarray,
) -> CameraGroundFrame:
    """Project a proper OpenCV camera pose onto the backend-world ground plane."""

    transform = _validated_camera_to_world(camera_to_world)
    forward = np.asarray(transform[:3, 2], dtype=np.float64).copy()
    forward[1] = 0.0
    forward_norm = float(np.linalg.norm(forward))
    if not math.isfinite(forward_norm) or forward_norm <= 1e-8:
        raise CoordinateFrameError("camera forward has no stable ground projection")
    forward /= forward_norm

    # Preserve the actual OpenCV +X camera axis, then Gram-Schmidt it against
    # the projected +Z camera-forward axis.  Do not infer a room-specific yaw.
    right = np.asarray(transform[:3, 0], dtype=np.float64).copy()
    right[1] = 0.0
    right -= float(np.dot(right, forward)) * forward
    right_norm = float(np.linalg.norm(right))
    if not math.isfinite(right_norm) or right_norm <= 1e-8:
        raise CoordinateFrameError("camera right has no stable ground projection")
    right /= right_norm

    return CameraGroundFrame(
        camera_world_m=np.asarray(transform[:3, 3], dtype=np.float64).copy(),
        camera_right_world=right,
        camera_forward_world=forward,
    )


def camera_ground_frame_from_extrinsics_col_major(
    extrinsics_col_major: Sequence[float],
) -> CameraGroundFrame:
    """Build the camera-ground frame from column-major world-to-camera E."""

    values = np.asarray(extrinsics_col_major, dtype=np.float64)
    if values.size != 16 or not np.isfinite(values).all():
        raise CoordinateFrameError("camera extrinsics must contain 16 finite values")
    world_to_camera = values.reshape((4, 4), order="F")
    if not np.allclose(world_to_camera[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise CoordinateFrameError("camera extrinsics must be affine")
    try:
        camera_to_world = np.linalg.inv(world_to_camera)
    except np.linalg.LinAlgError as exc:
        raise CoordinateFrameError("camera extrinsics are singular") from exc
    return camera_ground_frame_from_camera_to_world(camera_to_world)


def transform_positions(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply an affine transform to positions only, never to pose rotations."""

    values = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise CoordinateFrameError("positions must be a finite Nx3 array")
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise CoordinateFrameError("position transform must be a finite 4x4 matrix")
    return (matrix[:3, :3] @ values.T).T + matrix[:3, 3]


def camera_local_raster_indices(
    x_m: np.ndarray,
    z_m: np.ndarray,
    *,
    min_x_m: float,
    min_z_m: float,
    resolution_m: float,
    rows: int,
    columns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map +X-right/+Z-forward coordinates to row-zero-far raster indices."""

    x = np.asarray(x_m, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    if x.shape != z.shape:
        raise CoordinateFrameError("camera-local X/Z arrays must have the same shape")
    resolution = float(resolution_m)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise CoordinateFrameError("raster resolution must be positive and finite")
    row_count = int(rows)
    column_count = int(columns)
    if row_count <= 0 or column_count <= 0:
        raise CoordinateFrameError("raster shape must be positive")

    increasing_z_row = np.floor((z - float(min_z_m)) / resolution).astype(np.int64)
    row = (row_count - 1) - increasing_z_row
    column = np.floor((x - float(min_x_m)) / resolution).astype(np.int64)
    valid = (
        np.isfinite(x)
        & np.isfinite(z)
        & (row >= 0)
        & (row < row_count)
        & (column >= 0)
        & (column < column_count)
    )
    return row, column, valid
