from __future__ import annotations

import numpy as np

from tools.mapanything_phone_scan.run_mapanything_prior_variants import (
    _backproject_depth,
    _rotation_error_deg,
    _transform_poses,
    _zbuffer_depth,
)


def test_backproject_depth_uses_opencv_z_depth_and_cam2world() -> None:
    depth = np.full((2, 3), 2.0, dtype=np.float32)
    intrinsics = np.asarray(
        [[2.0, 0.0, 1.0], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [10.0, 20.0, 30.0]
    points = _backproject_depth(depth, intrinsics, pose)
    np.testing.assert_allclose(points[0, 1], [10.0, 19.5, 32.0], atol=1e-6)
    np.testing.assert_allclose(points[1, 2], [11.0, 20.5, 32.0], atol=1e-6)


def test_transform_poses_left_multiplies_world_frame() -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [4.0, -2.0, 7.0]
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    poses[1, :3, 3] = [1.0, 2.0, 3.0]
    transformed = _transform_poses(transform, poses)
    np.testing.assert_allclose(transformed[0, :3, 3], [4.0, -2.0, 7.0])
    np.testing.assert_allclose(transformed[1, :3, 3], [5.0, 0.0, 10.0])


def test_rotation_error_is_geodesic_angle() -> None:
    first = np.eye(4, dtype=np.float64)
    second = np.eye(4, dtype=np.float64)
    second[:3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    assert abs(_rotation_error_deg(first, second) - 90.0) < 1e-8


def test_zbuffer_keeps_nearest_point_per_pixel() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 1.5],
            [1.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    intrinsics = np.asarray(
        [[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    depth, count = _zbuffer_depth(
        points,
        np.eye(4, dtype=np.float64),
        intrinsics,
        (5, 5),
    )
    assert count == 2
    assert depth[2, 2] == 1.5
    assert depth[2, 3] == 2.0
