"""Pixel-to-world coordinate transforms."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _as_matrix4(candidate: Sequence[float], *, order: str = "C") -> np.ndarray:
    """Convert a flat sequence into a 4x4 matrix."""
    arr = np.array(candidate, dtype=float)
    if arr.size != 16:
        raise ValueError("Expected 16 elements for 4x4 matrix")
    return arr.reshape((4, 4), order=order)


def _as_intrinsics_matrix(K: Sequence[Sequence[float]]) -> np.ndarray:
    """Ensure a 3x3 intrinsics matrix."""
    arr = np.array(K, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError("Expected 3x3 intrinsics matrix")
    return arr


def build_align_matrix(align: Mapping[str, object] | None) -> np.ndarray:
    """Build a 4x4 alignment matrix with unit scaling."""
    if not isinstance(align, Mapping):
        return np.eye(4, dtype=float)
    mat = align.get("matrix") if isinstance(align, Mapping) else None
    matrix = None
    if isinstance(mat, Sequence):
        try:
            matrix = _as_matrix4(mat)
        except Exception:
            matrix = None
    if matrix is None:
        matrix = np.eye(4, dtype=float)
    units = align.get("units") if isinstance(align, Mapping) else None
    scale = None
    if isinstance(units, Mapping):
        s = units.get("s_obj_to_m")
        if isinstance(s, (int, float)):
            scale = float(s)
    if scale and abs(scale - 1.0) > 1e-9:
        scale_matrix = np.eye(4, dtype=float)
        scale_matrix[0, 0] = scale_matrix[1, 1] = scale_matrix[2, 2] = scale
        matrix = scale_matrix @ matrix
    return matrix


def pixel_to_world(
    u: float,
    v: float,
    depth_m: float,
    cam_id: str,
    K: Sequence[Sequence[float]],
    E_cam2world: Sequence[float] | Sequence[Sequence[float]],
    align: Mapping[str, object] | None,
) -> np.ndarray:
    """Project pixel coordinates with depth into aligned world space.

    Args:
        u: Horizontal pixel coordinate.
        v: Vertical pixel coordinate.
        depth_m: Metric depth along the camera forward axis.
        cam_id: Camera identifier (unused, but kept for logging/debug).
        K: 3x3 camera intrinsics matrix.
        E_cam2world: 4x4 camera-to-world transform (column-major or nested list).
        align: Alignment mapping that holds a transform matrix and unit scale.

    Returns:
        ``np.ndarray`` of shape ``(3,)`` with world coordinates in alignment frame.
    """
    if depth_m is None:
        raise ValueError("depth_m is required")
    depth = float(depth_m)
    if depth <= 0.0:
        raise ValueError("depth_m must be positive")

    K_mat = _as_intrinsics_matrix(K)

    if isinstance(E_cam2world, Mapping):  # type: ignore[unreachable]
        raise TypeError("E_cam2world must be a sequence, not mapping")

    try:
        if isinstance(E_cam2world, Sequence) and len(E_cam2world) == 16:
            T_cam2world = _as_matrix4(E_cam2world, order="F")
        else:
            T_cam2world = np.array(E_cam2world, dtype=float)
            if T_cam2world.shape != (4, 4):
                raise ValueError
    except Exception as exc:
        raise ValueError("Invalid E_cam2world") from exc

    fx = K_mat[0, 0]
    fy = K_mat[1, 1]
    cx = K_mat[0, 2]
    cy = K_mat[1, 2]

    x_cam = (float(u) - cx) / fx * depth
    y_cam = (float(v) - cy) / fy * depth
    z_cam = depth
    p_cam = np.array([x_cam, y_cam, z_cam, 1.0], dtype=float)

    p_world = T_cam2world @ p_cam
    align_matrix = build_align_matrix(align)
    p_aligned = align_matrix @ p_world
    return p_aligned[:3]


__all__ = ['pixel_to_world', 'build_align_matrix']
