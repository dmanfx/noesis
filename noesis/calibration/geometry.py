"""Geometry helpers for DS8 calibration.

Single implementation of pixel_to_world with correct semantics:
- DS8 does NOT apply align.matrix to outputs
- floor_y and unit_scale are applied consistently
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PixelToWorldResult:
    """Result of pixel_to_world conversion."""

    ok: bool
    world_point: Optional[List[float]] = None  # [x, y, z] in meters
    method: Optional[str] = None  # "depth" or "floor"
    error: Optional[str] = None


def pixel_to_world(
    K: np.ndarray,
    E_col_major: List[float],
    floor_y: float,
    unit_scale: float,
    u: float,
    v: float,
    depth_m: Optional[float] = None,
) -> PixelToWorldResult:
    """Convert pixel coordinates to world coordinates.

    Uses K (intrinsics) and E (extrinsics, world→camera, column-major) to project
    pixel (u, v) into world space. If depth_m is provided, uses depth projection;
    otherwise uses floor-plane intersection at floor_y.

    **Important:** This function does NOT apply align.matrix. Menon applies
    alignment client-side.

    Args:
        K: 3x3 intrinsics matrix
        E_col_major: 16 floats, column-major 4x4 world→camera extrinsics
        floor_y: Floor plane Y coordinate (in same units as E, typically meters)
        unit_scale: Scale factor to convert world units to meters (s_obj_to_m)
        u: Pixel x coordinate
        v: Pixel y coordinate
        depth_m: Optional depth in meters

    Returns:
        PixelToWorldResult with world_point, method, and ok status
    """
    # Validate inputs
    if K is None or K.shape != (3, 3):
        return PixelToWorldResult(ok=False, error="invalid_K")

    if E_col_major is None or len(E_col_major) != 16:
        return PixelToWorldResult(ok=False, error="invalid_E")

    # Parse extrinsics: E is world→camera, column-major
    try:
        E_mat = np.array(E_col_major, dtype=np.float64).reshape((4, 4), order="F")
        Twc = np.linalg.inv(E_mat)  # camera→world
        R_wc = Twc[:3, :3].copy()
        C_world = Twc[:3, 3].copy()
    except Exception:
        return PixelToWorldResult(ok=False, error="bad_extrinsics")

    # Apply unit scale
    scale = float(unit_scale) if unit_scale is not None and np.isfinite(unit_scale) and unit_scale > 0 else 1.0
    C_world = C_world * scale
    floor_y_m = float(floor_y) * scale if floor_y is not None else 0.0

    # Depth projection path
    if depth_m is not None and np.isfinite(depth_m) and depth_m > 0:
        try:
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])

            # Unproject pixel to camera coordinates (in meters)
            x_cam = (u - cx) / fx * depth_m
            y_cam = (v - cy) / fy * depth_m
            z_cam = depth_m

            # Apply unit scale to camera-space point (E translation is in world units)
            x_cam *= scale
            y_cam *= scale
            z_cam *= scale

            # Transform to world coordinates
            p_world = (R_wc @ np.array([x_cam, y_cam, z_cam], dtype=np.float64)) + C_world

            return PixelToWorldResult(
                ok=True,
                world_point=[float(p_world[0]), float(p_world[1]), float(p_world[2])],
                method="depth",
            )
        except Exception:
            # Fall through to floor-plane intersection
            pass

    # Floor-plane intersection path
    try:
        uv1 = np.array([u, v, 1.0], dtype=np.float64)
        Kinv = np.linalg.inv(K)
        dir_cam = Kinv @ uv1

        # Normalize direction
        norm = float(np.linalg.norm(dir_cam))
        if norm <= 1e-9:
            return PixelToWorldResult(ok=False, error="invalid_ray")
        dir_cam = dir_cam / norm

        # Transform to world direction
        dir_world = R_wc @ dir_cam

        # Intersect with floor plane (Y = floor_y_m)
        denom = float(dir_world[1])
        if abs(denom) < 1e-9:
            return PixelToWorldResult(ok=False, error="no_intersection")

        t = (floor_y_m - float(C_world[1])) / denom
        if t < 0:
            return PixelToWorldResult(ok=False, error="no_intersection")

        hit = C_world + float(t) * dir_world

        return PixelToWorldResult(
            ok=True,
            world_point=[float(hit[0]), float(hit[1]), float(hit[2])],
            method="floor",
        )
    except Exception:
        return PixelToWorldResult(ok=False, error="no_intersection")
