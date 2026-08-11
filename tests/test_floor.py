import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.floor import backproject_to_camera, camera_plane_to_world, fit_floor_plane


def _load_intrinsics(width: int, height: int) -> np.ndarray:
    root = Path(__file__).resolve().parents[1]
    intr_data = json.loads((root / "intrinsics.json").read_text())
    model = intr_data["unifi_protect_g3_instant"]
    fx = float(model["intrinsics"]["fx"])
    fy = float(model["intrinsics"]["fy"])
    cx = float(model["intrinsics"]["cx"])
    cy = float(model["intrinsics"]["cy"])
    base_width, base_height = model["resolution"]
    sx = width / float(base_width)
    sy = height / float(base_height)
    return np.array(
        [[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _load_extrinsic_matrix() -> np.ndarray:
    root = Path(__file__).resolve().parents[1]
    extr_data = json.loads((root / "config" / "camera_calibration.json").read_text())
    E_list = extr_data["cameras"]["kitchen"]["E"]
    return np.array(E_list, dtype=np.float64).reshape((4, 4), order="F")


def _rotate_x(vec: np.ndarray, degrees: float) -> np.ndarray:
    rad = np.deg2rad(degrees)
    c = np.cos(rad)
    s = np.sin(rad)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    return rot @ vec


def _synthesize_depth(
    width: int,
    height: int,
    intrinsics: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> np.ndarray:
    xs = np.arange(width, dtype=np.float64)
    ys = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(xs, ys)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    rays = np.stack(((uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)), axis=-1)
    denom = rays @ normal
    depth = (-offset) / denom
    return depth.astype(np.float32)


@pytest.fixture(scope="module")
def synthetic_floor():
    width = height = 256
    intrinsics = _load_intrinsics(width, height)
    base_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    plane_normal = _rotate_x(base_normal, 10.0)
    plane_offset = -plane_normal @ np.array([0.0, 0.0, 2.0])
    depth = _synthesize_depth(width, height, intrinsics, plane_normal, plane_offset)
    points = backproject_to_camera(depth, intrinsics.astype(np.float32), roi_ratio=1.0, max_points=depth.size)
    assert points.shape[0] > 0
    rng_state = np.random.get_state()
    np.random.seed(1234)
    plane_model = fit_floor_plane(points, max_iterations=400, distance_threshold=0.01, min_inlier_ratio=0.5)
    np.random.set_state(rng_state)
    assert plane_model is not None
    expected_normal = plane_normal.copy()
    expected_offset = plane_offset
    if expected_normal[1] < 0:
        expected_normal = -expected_normal
        expected_offset = -expected_offset
    expected_normal = expected_normal / np.linalg.norm(expected_normal)
    extrinsic = _load_extrinsic_matrix()
    world_plane = camera_plane_to_world(plane_model, extrinsic)
    return {
        "depth": depth,
        "intrinsics": intrinsics,
        "plane_normal": plane_normal,
        "plane_offset": plane_offset,
        "expected_normal": expected_normal,
        "expected_offset": expected_offset,
        "points": points,
        "plane_model": plane_model,
        "extrinsic": extrinsic,
        "world_plane": world_plane,
        "width": width,
        "height": height,
    }


def test_backproject_points_lie_on_plane(synthetic_floor):
    points = synthetic_floor["points"].astype(np.float64)
    normal = synthetic_floor["plane_normal"]
    offset = synthetic_floor["plane_offset"]
    residual = points @ normal + offset
    assert np.abs(residual).max() < 1e-3


def test_fit_floor_plane_recovers_normal(synthetic_floor):
    plane_model = synthetic_floor["plane_model"]
    expected_normal = synthetic_floor["expected_normal"]
    dot = np.clip(float(np.dot(plane_model.normal.astype(np.float64), expected_normal)), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    assert angle < 3.0
    expected_offset = synthetic_floor["expected_offset"]
    assert abs(float(plane_model.offset) - expected_offset) < 0.05


def test_pixel_to_world_roundtrip(synthetic_floor):
    intr = synthetic_floor["intrinsics"]
    plane_model = synthetic_floor["plane_model"]
    extrinsic = synthetic_floor["extrinsic"]
    Twc = np.linalg.inv(extrinsic)
    R_wc = Twc[:3, :3]
    t_wc = Twc[:3, 3]
    R_cw = extrinsic[:3, :3]
    t_cw = extrinsic[:3, 3]
    world_plane = synthetic_floor["world_plane"]
    Kinv = np.linalg.inv(intr)
    rng = np.random.default_rng(2025)
    width = synthetic_floor["width"]
    height = synthetic_floor["height"]
    us = rng.integers(low=int(width * 0.2), high=width - 1, size=10)
    vs = rng.integers(low=int(height * 0.2), high=height - 1, size=10)
    for u, v in zip(us, vs):
        ray_cam = Kinv @ np.array([float(u), float(v), 1.0], dtype=np.float64)
        denom = float(np.dot(plane_model.normal.astype(np.float64), ray_cam))
        assert abs(denom) > 1e-6
        depth = -float(plane_model.offset) / denom
        assert depth > 0
        cam_point = ray_cam * depth
        world_point = R_wc @ cam_point + t_wc
        cam_back = R_cw @ world_point + t_cw
        assert np.allclose(cam_back, cam_point, atol=1e-5)
        residual_world = float(np.dot(world_plane.normal.astype(np.float64), world_point) + world_plane.offset)
        assert abs(residual_world) < 1e-3
        proj = intr @ cam_back
        u_proj = proj[0] / proj[2]
        v_proj = proj[1] / proj[2]
        pixel_error = np.hypot(u_proj - u, v_proj - v)
        assert pixel_error < 2.0
