import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from geometry.floor import PlaneModel, backproject_to_camera, camera_plane_to_world, fit_floor_plane


def make_intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def test_backproject_plane_points():
    depth = np.full((3, 3), 2.0, dtype=np.float32)
    conf = np.ones_like(depth)
    mask = np.ones_like(depth, dtype=bool)
    intrinsics = make_intrinsics(100.0, 100.0, 1.0, 1.0)
    pts = backproject_to_camera(depth, intrinsics, conf=conf, mask=mask, min_conf=0.1, roi_ratio=1.0)
    assert pts.shape[1] == 3
    assert np.allclose(pts[:, 2], 2.0)


def test_fit_floor_plane_returns_expected_normal():
    depth = np.full((4, 4), 3.0, dtype=np.float32)
    conf = np.ones_like(depth)
    mask = np.ones_like(depth, dtype=bool)
    intrinsics = make_intrinsics(80.0, 80.0, 2.0, 2.0)
    pts = backproject_to_camera(depth, intrinsics, conf=conf, mask=mask, min_conf=0.0, roi_ratio=0.5)
    plane = fit_floor_plane(pts)
    assert plane is not None
    assert plane.normal.shape == (3,)
    assert np.isclose(np.linalg.norm(plane.normal), 1.0, atol=1e-2)


def test_camera_plane_to_world_identity():
    plane = PlaneModel(normal=np.array([0.0, 1.0, 0.0], dtype=np.float32), offset=-1.0, inlier_ratio=1.0)
    transformed = camera_plane_to_world(plane, np.eye(4, dtype=np.float32))
    assert np.allclose(transformed.normal, plane.normal)
    assert np.isclose(transformed.offset, plane.offset)
