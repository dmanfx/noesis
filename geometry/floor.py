"""Utilities for fitting floor planes from MapAnything depth outputs."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from utils.rate_limited_logger import RateLimitedLogger


_logger = RateLimitedLogger(logging.getLogger(__name__), rate_limit_seconds=5.0)


@dataclass(frozen=True)
class PlaneModel:
    """Represents a plane defined by normal vector and offset."""

    normal: np.ndarray  # shape (3,)
    offset: float
    inlier_ratio: float

    @property
    def height(self) -> float:
        """Return floor height assuming plane normal has positive Y component."""
        denom = float(self.normal[1])
        if abs(denom) < 1e-6:
            return float("nan")
        return -self.offset / denom


def backproject_to_camera(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    conf: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    min_conf: float = 0.5,
    roi_ratio: float = 0.25,
    max_points: int = 20000,
) -> np.ndarray:
    """Back-project depth map to camera-space point cloud.

    Args:
        depth: Depth map (meters)
        intrinsics: 3x3 camera intrinsic matrix
        conf: Optional confidence map
        mask: Optional boolean mask
        min_conf: Minimum confidence threshold
        roi_ratio: Fraction of image height (from bottom) to consider for floor
        max_points: Maximum number of points to return

    Returns:
        Nx3 array of 3D points in camera coordinates
    """
    if depth.ndim != 2:
        raise ValueError("Depth map must be 2D")
    height, width = depth.shape
    roi_start = max(0, int(height * (1.0 - roi_ratio)))
    valid = np.ones_like(depth, dtype=bool)
    valid[:roi_start, :] = False
    if mask is not None:
        valid &= mask.astype(bool)
    if conf is not None:
        valid &= conf >= float(min_conf)
    valid &= np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    ys, xs = np.nonzero(valid)
    z = depth[valid]

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    points = np.stack((x, y, z), axis=1).astype(np.float32)

    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[idx]
    return points


def fit_floor_plane(
    points: np.ndarray,
    *,
    max_iterations: int = 200,
    distance_threshold: float = 0.03,
    min_inlier_ratio: float = 0.2,
) -> Optional[PlaneModel]:
    """Fit a plane to 3D points using RANSAC.

    Args:
        points: Nx3 array of 3D points
        max_iterations: Maximum RANSAC iterations
        distance_threshold: Inlier distance threshold in meters
        min_inlier_ratio: Minimum fraction of points required to accept a plane

    Returns:
        PlaneModel if successful, else None
    """
    if points.shape[0] < 3:
        _logger.warning("Not enough points to fit floor plane")
        return None

    best_inliers = None
    best_plane = None
    n_points = points.shape[0]

    for _ in range(max_iterations):
        sample_idx = np.random.choice(n_points, 3, replace=False)
        plane = _plane_from_points(points[sample_idx])
        if plane is None:
            continue
        normal, offset = plane
        distances = np.abs(points @ normal + offset)
        inliers = distances < distance_threshold
        inlier_count = int(inliers.sum())
        if best_inliers is None or inlier_count > int(best_inliers.sum()):
            best_inliers = inliers
            best_plane = (normal, offset)

    if best_plane is None or best_inliers is None:
        _logger.warning("Failed to find a floor plane candidate")
        return None

    inlier_ratio = float(best_inliers.sum() / n_points)
    if inlier_ratio < min_inlier_ratio:
        _logger.warning("Floor plane inlier ratio %.2f below threshold", inlier_ratio)
        return None

    refined_normal, refined_offset = _refine_plane(points[best_inliers])
    if refined_normal[1] < 0:
        refined_normal = -refined_normal
        refined_offset = -refined_offset

    return PlaneModel(normal=refined_normal, offset=refined_offset, inlier_ratio=inlier_ratio)


def camera_plane_to_world(plane: PlaneModel, extrinsic_world_to_cam: np.ndarray) -> PlaneModel:
    """Transform a plane from camera coordinates to world coordinates."""
    if extrinsic_world_to_cam.shape != (4, 4):
        raise ValueError("Extrinsic matrix must be 4x4")
    R = extrinsic_world_to_cam[:3, :3]
    t = extrinsic_world_to_cam[:3, 3]
    normal_world = R.T @ plane.normal
    offset_world = float(plane.normal @ t + plane.offset)
    normal_world_norm = np.linalg.norm(normal_world)
    if normal_world_norm == 0:
        raise ValueError("Invalid plane normal during transformation")
    normal_world = normal_world / normal_world_norm
    offset_world = offset_world / normal_world_norm
    if normal_world[1] < 0:
        normal_world = -normal_world
        offset_world = -offset_world
    return PlaneModel(normal=normal_world, offset=offset_world, inlier_ratio=plane.inlier_ratio)


def _plane_from_points(points: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    if points.shape[0] != 3:
        return None
    p0, p1, p2 = points
    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return None
    normal = normal / norm
    offset = -normal @ p0
    return normal, float(offset)


def _refine_plane(points: np.ndarray) -> Tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    offset = -normal @ centroid
    return normal.astype(np.float32), float(offset)


__all__ = [
    "PlaneModel",
    "backproject_to_camera",
    "fit_floor_plane",
    "camera_plane_to_world",
]
