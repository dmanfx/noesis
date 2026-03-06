"""
Homography helpers for mapping image pixels to a ground plane and into a BEV grid.

These utilities stay independent from the DeepStream runtime so they can be imported
from both the backend processing path and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import cv2
import math
import numpy as np


@dataclass(frozen=True)
class Plane:
    """Simple plane representation n^T x + d = 0."""

    normal: np.ndarray  # shape (3,)
    offset: float       # scalar d

    @staticmethod
    def horizontal(floor_y: float) -> "Plane":
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        offset = -float(floor_y)
        return Plane(normal=normal, offset=offset)


def parse_extrinsics(E_col_major_16: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a column-major world→camera 4x4 matrix into camera pose.

    Returns:
        R_wc: 3x3 rotation mapping camera axes into world coordinates.
        C_world: 3x1 camera origin in world coordinates.
    """
    if len(E_col_major_16) != 16:
        raise ValueError("Extrinsics must be a 16-element column-major list")
    E = np.array(E_col_major_16, dtype=np.float64).reshape((4, 4), order="F")
    Twc = np.linalg.inv(E)
    R_wc = Twc[:3, :3].copy()
    C_world = Twc[:3, 3].copy()
    return R_wc, C_world


def ray_from_pixel(u: float, v: float, K: np.ndarray, R_wc: np.ndarray, C_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (origin, direction) in world coordinates for an image pixel."""
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    Kinv = np.linalg.inv(K)
    dir_cam = Kinv @ uv1
    dir_cam = dir_cam / np.linalg.norm(dir_cam)
    dir_world = R_wc @ dir_cam
    dir_world = dir_world / np.linalg.norm(dir_world)
    return C_world.copy(), dir_world


def intersect_plane(origin: np.ndarray, direction: np.ndarray, plane: Plane) -> np.ndarray | None:
    """
    Intersect a ray with a plane. Returns xyz or None if parallel/backwards.
    """
    denom = float(np.dot(plane.normal, direction))
    if abs(denom) < 1e-9:
        return None
    t = -(np.dot(plane.normal, origin) + plane.offset) / denom
    if t < 0:
        return None
    return origin + t * direction


def project_world_to_image(
    point_world: Sequence[float],
    K: np.ndarray,
    E_col_major_16: Sequence[float],
    image_size: Tuple[int, int],
    unit_scale: float = 1.0,
    *,
    flip_u: bool = False,
    flip_v: bool = False,
) -> Tuple[float, float] | None:
    """Project a world-space point into image pixels."""
    if K.shape != (3, 3):
        raise ValueError("Intrinsics K must be 3x3")
    if len(point_world) < 3:
        raise ValueError("point_world must contain xyz")
    width, height = image_size
    scale = float(unit_scale or 1.0)
    R_wc, C_world = parse_extrinsics(E_col_major_16)
    C_world = C_world * scale
    world = np.array([float(point_world[0]), float(point_world[1]), float(point_world[2])], dtype=np.float64)
    R_cw = R_wc.T
    point_cam = R_cw @ (world - C_world)
    z_cam = float(point_cam[2])
    if not math.isfinite(z_cam) or z_cam <= 1e-9:
        return None
    uvw = K @ point_cam
    if abs(float(uvw[2])) < 1e-9:
        return None
    u = float(uvw[0] / uvw[2])
    v = float(uvw[1] / uvw[2])
    if flip_u:
        u = float(max(0, int(width) - 1)) - u
    if flip_v:
        v = float(max(0, int(height) - 1)) - v
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return float(u), float(v)


def estimate_upright_height_from_top_and_foot(
    u: float,
    v: float,
    foot_world: Sequence[float],
    K: np.ndarray,
    E_col_major_16: Sequence[float],
    floor_y: float,
    image_size: Tuple[int, int],
    unit_scale: float = 1.0,
    *,
    flip_u: bool = False,
    flip_v: bool = False,
) -> float | None:
    """Estimate an upright person's height from a top-of-box pixel and a floor footpoint."""
    if K.shape != (3, 3):
        raise ValueError("Intrinsics K must be 3x3")
    if len(foot_world) < 3:
        raise ValueError("foot_world must contain xyz")
    width, height = image_size
    scale = float(unit_scale or 1.0)
    floor_y_scaled = float(floor_y) * scale
    R_wc, C_world = parse_extrinsics(E_col_major_16)
    C_world = C_world * scale
    if flip_u:
        u = float(max(0, int(width) - 1)) - float(u)
    if flip_v:
        v = float(max(0, int(height) - 1)) - float(v)
    origin, direction = ray_from_pixel(float(u), float(v), K, R_wc, C_world)
    foot = np.array([float(foot_world[0]), float(foot_world[1]), float(foot_world[2])], dtype=np.float64)
    denom = float(direction[0] * direction[0] + direction[2] * direction[2])
    if denom <= 1e-9:
        return None
    t = (
        (float(foot[0]) - float(origin[0])) * float(direction[0])
        + (float(foot[2]) - float(origin[2])) * float(direction[2])
    ) / denom
    if not math.isfinite(t) or t <= 0.0:
        return None
    height = float(origin[1] + t * float(direction[1]) - floor_y_scaled)
    if not math.isfinite(height) or height <= 0.0:
        return None
    return float(height)


def compute_ground_frustum_aabb(
    K: np.ndarray,
    E_col_major_16: Iterable[float],
    floor_y: float,
    image_size: Tuple[int, int],
    unit_scale: float,
    max_distance_m: float,
    padding_m: float = 0.25,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute an axis-aligned bounding box (AABB) on the ground plane (Y = floor_y)
    that covers the camera's visible floor region, in *meters*.

    Returns:
        (x_min, x_max), (z_min, z_max)
    """
    R_wc, C_world = parse_extrinsics(E_col_major_16)
    scale = float(unit_scale or 1.0)
    C_world = C_world * scale
    floor_y_m = float(floor_y) * scale
    plane = Plane.horizontal(floor_y_m)

    width, height = image_size

    # Build sample pixels on the image that are likely to see the floor
    # Use a grid instead of fixed bottom-row samples to catch high-horizon/tilted views
    grid_steps_x = 8
    grid_steps_y = 8
    xs = np.linspace(0, width - 1, grid_steps_x)
    ys = np.linspace(0, height - 1, grid_steps_y)
    samples = [(float(x), float(y)) for x in xs for y in ys]

    xz_hits = []
    for u, v in samples:
        origin, direction = ray_from_pixel(float(u), float(v), K, R_wc, C_world)
        hit = intersect_plane(origin, direction, plane)
        if hit is None:
            continue
        dx = float(hit[0] - C_world[0])
        dz = float(hit[2] - C_world[2])
        dist = math.hypot(dx, dz)

        if dist > max_distance_m and dist > 1e-6:
            scale_d = max_distance_m / dist
            hit = np.array(
                [
                    C_world[0] + dx * scale_d,
                    hit[1],  # keep Y
                    C_world[2] + dz * scale_d,
                ],
                dtype=np.float64,
            )
        xz_hits.append((hit[2], hit[0]))

    if len(xz_hits) < 3:
        # Fallback extents
        return (-4.0, 4.0), (0.0, 12.0)

    xs = [p[0] for p in xz_hits]
    zs = [p[1] for p in xz_hits]
    x_min = min(xs) - padding_m
    x_max = max(xs) + padding_m
    z_min = min(zs) - padding_m
    z_max = max(zs) + padding_m

    # Ensure z_min is not negative
    z_min = max(0.0, z_min)

    # Ensure the extents are at least some minimal size
    if (x_max - x_min) < 1.0:
        cx = 0.5 * (x_min + x_max)
        x_min = cx - 0.5
        x_max = cx + 0.5
    if (z_max - z_min) < 1.0:
        cz = 0.5 * (z_min + z_max)
        z_min = cz - 0.5
        z_max = cz + 0.5

    return (x_min, x_max), (z_min, z_max)


def image_corners(width: int, height: int) -> Tuple[Tuple[float, float], ...]:
    return (
        (0.0, 0.0),
        (float(width), 0.0),
        (float(width), float(height)),
        (0.0, float(height)),
    )


def img_to_plane_homography(
    K: np.ndarray,
    E_col_major_16: Sequence[float],
    floor_y: float,
    image_size: Tuple[int, int],
    unit_scale: float = 1.0,
    *,
    flip_u: bool = False,
    flip_v: bool = False,
) -> np.ndarray:
    """
    Build a 3x3 homography that maps image pixels to ground plane XZ metres.
    """
    if K.shape != (3, 3):
        raise ValueError("Intrinsics K must be 3x3")
    width, height = image_size
    scale = float(unit_scale or 1.0)
    plane = Plane.horizontal(float(floor_y) * scale)
    R_wc, C_world = parse_extrinsics(E_col_major_16)
    C_world = C_world * scale

    def _apply_flip(u: float, v: float) -> Tuple[float, float]:
        if flip_u:
            u = float(width - 1) - float(u)
        if flip_v:
            v = float(height - 1) - float(v)
        return float(u), float(v)

    def try_build(points: Sequence[Tuple[float, float]]) -> np.ndarray | None:
        img_pts: list[list[float]] = []
        plane_pts: list[list[float]] = []
        for (u, v) in points:
            u_ray, v_ray = _apply_flip(u, v)
            origin, direction = ray_from_pixel(u_ray, v_ray, K, R_wc, C_world)
            hit = intersect_plane(origin, direction, plane)
            if hit is None:
                continue
            img_pts.append([u, v])
            plane_pts.append([hit[0], hit[2]])  # Map to world (X, Z)
        
        if len(img_pts) < 4:
            return None
            
        src = np.array(img_pts, dtype=np.float32)
        dst = np.array(plane_pts, dtype=np.float32)
        
        # Use RANSAC if we have enough points to handle outliers, though ray casting should be clean
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or H.shape != (3, 3):
            return None
        return H

    # Generate a grid of samples across the whole image to find valid ground plane projections.
    # This handles cases where the bottom band might be looking at the horizon or ceiling.
    grid_steps_x = 8
    grid_steps_y = 8
    xs = np.linspace(0, width - 1, grid_steps_x)
    ys = np.linspace(0, height - 1, grid_steps_y)
    grid_points = [(float(x), float(y)) for x in xs for y in ys]
    
    H = try_build(grid_points)
    if H is not None:
        return H

    raise RuntimeError("Failed to compute image→plane homography (no valid ray-plane intersections found in grid search)")


def plane_to_bev_affine(
    x_range: Tuple[float, float],
    z_range: Tuple[float, float],
    meters_per_px: float,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Build an affine matrix that maps plane coords (X,Z) to BEV pixels.

    Returns:
        (M, (width_px, height_px))
    """
    x_min, x_max = x_range
    z_min, z_max = z_range
    if x_max <= x_min or z_max <= z_min:
        raise ValueError("Invalid BEV ranges")
    if meters_per_px <= 0:
        raise ValueError("meters_per_px must be positive")

    width_px = int(np.ceil((x_max - x_min) / meters_per_px))
    height_px = int(np.ceil((z_max - z_min) / meters_per_px))
    width_px = max(1, width_px)
    height_px = max(1, height_px)

    sx = 1.0 / meters_per_px
    sz = 1.0 / meters_per_px
    tx = -x_min * sx
    ty = z_max * sz  # flip forward axis so +Z goes up

    A = np.array(
        [
            [sx, 0.0, tx],
            [0.0, -sz, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return A, (width_px, height_px)


def compose_bev_homography(
    K: np.ndarray,
    E_col_major_16: Sequence[float],
    floor_y: float,
    image_size: Tuple[int, int],
    x_range: Tuple[float, float],
    z_range: Tuple[float, float],
    meters_per_px: float,
    unit_scale: float = 1.0,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Compose the final image→BEV homography and output dimensions.
    """
    H_img2plane = img_to_plane_homography(K, E_col_major_16, floor_y, image_size, unit_scale=unit_scale)
    A_plane2bev, bev_size = plane_to_bev_affine(x_range, z_range, meters_per_px)
    M = A_plane2bev @ H_img2plane
    return M, bev_size


__all__ = [
    "Plane",
    "parse_extrinsics",
    "ray_from_pixel",
    "intersect_plane",
    "project_world_to_image",
    "estimate_upright_height_from_top_and_foot",
    "compute_ground_frustum_aabb",
    "img_to_plane_homography",
    "plane_to_bev_affine",
    "compose_bev_homography",
]
