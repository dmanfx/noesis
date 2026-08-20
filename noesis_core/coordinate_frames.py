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

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


BACKEND_GRID_ORIENTATION = "row_increases_positive_z_column_increases_positive_x"
CAMERA_LOCAL_RASTER_ORIENTATION = (
    "row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x"
)


class CoordinateFrameError(ValueError):
    """Raised when a coordinate transform cannot satisfy the frame contract."""


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
