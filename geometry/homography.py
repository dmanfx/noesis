"""
Homography helpers for mapping image pixels to a ground plane and into a BEV grid.

These utilities stay independent from the DeepStream runtime so they can be imported
from both the backend processing path and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
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
) -> np.ndarray:
    """
    Build a 3x3 homography that maps image pixels to ground plane XZ metres.
    """
    if K.shape != (3, 3):
        raise ValueError("Intrinsics K must be 3x3")
    width, height = image_size
    plane = Plane.horizontal(floor_y)
    R_wc, C_world = parse_extrinsics(E_col_major_16)

    img_pts = []
    plane_pts = []
    for (u, v) in image_corners(width, height):
        origin, direction = ray_from_pixel(u, v, K, R_wc, C_world)
        hit = intersect_plane(origin, direction, plane)
        if hit is None:
            raise ValueError("Image corner ray does not intersect the floor plane")
        img_pts.append([u, v])
        plane_pts.append([hit[0], hit[2]])  # map to (X,Z)

    src = np.array(img_pts, dtype=np.float32)
    dst = np.array(plane_pts, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    if H is None or H.shape != (3, 3):
        raise RuntimeError("Failed to compute image→plane homography")
    return H


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
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Compose the final image→BEV homography and output dimensions.
    """
    H_img2plane = img_to_plane_homography(K, E_col_major_16, floor_y, image_size)
    A_plane2bev, bev_size = plane_to_bev_affine(x_range, z_range, meters_per_px)
    M = A_plane2bev @ H_img2plane
    return M, bev_size


__all__ = [
    "Plane",
    "parse_extrinsics",
    "img_to_plane_homography",
    "plane_to_bev_affine",
    "compose_bev_homography",
]
