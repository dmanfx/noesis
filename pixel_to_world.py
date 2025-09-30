from typing import Tuple, Optional, Dict, Any
import numpy as np


def K_from_intrinsics(intr: Dict[str, Any]) -> Optional[np.ndarray]:
    if isinstance(intr, (list, tuple)) and len(intr) >= 4:
        try:
            fx = float(intr[0]); fy = float(intr[1])
            cx = float(intr[2]); cy = float(intr[3])
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
        except Exception:
            return None
    if not isinstance(intr, dict):
        return None
    if 'K3x3' in intr and isinstance(intr['K3x3'], list):
        try:
            K = np.array(intr['K3x3'], dtype=float)
            if K.shape == (3, 3):
                return K
        except Exception:
            pass
    keys = ('fx', 'fy', 'cx', 'cy')
    if all(k in intr for k in keys):
        try:
            fx = float(intr['fx']); fy = float(intr['fy'])
            cx = float(intr['cx']); cy = float(intr['cy'])
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
        except Exception:
            return None
    return None


def E_to_world_and_R(E_col_major_16: list) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Given E (world→camera) column-major, return camera center in world and world→camera rotation/translation.

    Returns (C_world, R_wc, t_wc) implicitly through matrix. We output (C_world, R_cw) as needed for ray computation.
    For ray tracing, we invert E to get camera pose in world: Twc = E^{-1}.
    """
    try:
        if not isinstance(E_col_major_16, list) or len(E_col_major_16) != 16:
            return None
        M = np.array(E_col_major_16, dtype=float).reshape((4, 4), order='F')  # column-major
        Twc = np.linalg.inv(M)  # camera pose in world
        C = Twc[:3, 3].copy()
        R = Twc[:3, :3].copy()
        return C, R
    except Exception:
        return None


def ray_from_pixel(u: float, v: float, K: np.ndarray, C_world: np.ndarray, R_wc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute world-space ray origin and direction for a pixel.

    Rationale: p_cam ~ K^{-1}[u v 1]^T, then transform dir to world via R_wc, origin = C_world.
    """
    uv1 = np.array([u, v, 1.0], dtype=float)
    Kinv = np.linalg.inv(K)
    dir_cam = Kinv @ uv1
    dir_cam = dir_cam / np.linalg.norm(dir_cam)
    # R_wc maps camera frame to world frame
    dir_world = R_wc @ dir_cam
    dir_world = dir_world / max(1e-12, np.linalg.norm(dir_world))
    return C_world, dir_world


def intersect_floor(origin_world: np.ndarray, dir_world: np.ndarray, floor_y: float) -> Optional[Tuple[float, float, float]]:
    """Intersect a ray with plane y = floor_y. Returns (x,y,z) or None if parallel/upwards.
    Plane normal = [0,1,0]. Ray: R(t) = O + t*D. Solve for y = floor_y.
    """
    dy = dir_world[1]
    if abs(dy) < 1e-9:
        return None
    t = (floor_y - origin_world[1]) / dy
    if t < 0:
        return None
    P = origin_world + t * dir_world
    return float(P[0]), float(P[1]), float(P[2])


def bbox_bottom_center(bbox_ltw_h: list) -> Optional[Tuple[float, float]]:
    """bbox as [left, top, width, height] -> (cx, y_bottom)."""
    try:
        left, top, width, height = bbox_ltw_h
        cx = float(left) + float(width) * 0.5
        yb = float(top) + float(height)
        return cx, yb
    except Exception:
        return None
