from __future__ import annotations

import numpy as np
import pytest

from tools.mapanything_phone_scan.render_phone_heatmap_diagnostics import (
    PhoneCloud,
    _camera_positions,
    _point_splat,
    _phone_floor_transform,
    _present_camera_ground,
    _rasterize,
)
from noesis_core.coordinate_frames import camera_ground_frame_from_camera_to_world


def test_floor_selection_rejects_dominant_wall_and_elevated_counter() -> None:
    pytest.importorskip("open3d")
    rng = np.random.default_rng(4)
    wall = rng.uniform([-1.0, -0.8, 0.0], [-1.0, 1.6, 3.0], (2400, 3))
    counter = rng.uniform([-0.8, 0.4, 0.0], [1.0, 0.4, 1.5], (1000, 3))
    floor = rng.uniform([-0.8, 1.6, 0.0], [1.0, 1.6, 3.0], (800, 3))
    cloud = _cloud()
    cloud.points = np.concatenate([wall, counter, floor])
    # An arbitrary world rotation must not turn this into a fixed-Y heuristic.
    rotation = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cloud.points = cloud.points @ rotation.T
    cloud.camera_to_world[0, :3, :3] = rotation
    original_points = cloud.points.copy()
    transform, report = _phone_floor_transform(cloud)
    leveled_floor = cloud.points[-len(floor):] @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(leveled_floor[:, 1], 0.0, atol=1e-5)
    assert report["camera_height_m"]["median"] == pytest.approx(1.6, abs=0.01)
    assert report["selected_candidate_index"] > 0
    assert "plane_not_aligned_with_camera_up" in report["candidates"][0]["rejection_reasons"]
    np.testing.assert_array_equal(cloud.points, original_points)


def test_floor_selection_rejects_room_without_floor_evidence() -> None:
    pytest.importorskip("open3d")
    rng = np.random.default_rng(4)
    cloud = _cloud()
    wall = rng.uniform([-1.0, -0.8, 0.0], [-1.0, 1.6, 3.0], (1200, 3))
    counter = rng.uniform([-0.8, 0.4, 0.0], [1.0, 0.4, 1.5], (800, 3))
    ceiling = rng.uniform([-0.8, -1.0, 0.0], [1.0, -1.0, 3.0], (600, 3))
    cloud.points = np.concatenate([wall, counter, ceiling])
    with pytest.raises(ValueError, match="no supported floor plane"):
        _phone_floor_transform(cloud)


def test_trajectory_preview_preserves_aspect_and_includes_camera_endpoints(tmp_path) -> None:
    import cv2

    from tools.mapanything_phone_scan.inference import _write_trajectory_preview

    # The dense cloud's percentiles exclude most of this narrow, L-shaped walk.
    points = np.random.default_rng(2).uniform(-0.1, 0.1, (20_000, 3))
    cameras = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 4.0]])
    output = tmp_path / "trajectory.png"
    _write_trajectory_preview(output, points, cameras)
    pixels = cv2.imread(str(output))
    orange = np.all(pixels == [44, 148, 255], axis=2)
    green = np.all(pixels == [96, 220, 96], axis=2)
    rows, columns = np.nonzero(orange | green)
    assert rows.min() >= 45 and rows.max() <= 1155
    assert columns.min() >= 45 and columns.max() <= 1155
    assert np.ptp(columns) / np.ptp(rows) == pytest.approx(0.5, abs=0.02)


def _cloud() -> PhoneCloud:
    return PhoneCloud(
        points=np.asarray(
            [
                [0.011, 0.10, 0.011],
                [0.012, 0.11, 0.012],
                [0.011, 0.90, 0.011],
            ],
            dtype=np.float32,
        ),
        colors=np.asarray(
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8
        ),
        weights=np.asarray([0.2, 0.9, 0.8], dtype=np.float32),
        ranges=np.ones(3, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float64)[None],
        frame_zero_depth=np.ones((1, 1), dtype=np.float32),
        frame_zero_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
    )


def test_static_crop_clips_drawn_path_without_clamping_or_mutating_geometry():
    cloud = _cloud()
    cloud.camera_to_world = np.repeat(np.eye(4)[None], 4, axis=0)
    cloud.camera_to_world[:, :3, 3] = [[-1, 1, -1], [0.5, 1, 0.5], [2, 1, 2], [2, 1, 3]]
    original_poses, original_points = cloud.camera_to_world.copy(), cloud.points.copy()
    grids = _rasterize(cloud, (0, 1, 0, 1), 0.05)
    assert grids["structural"].shape == (20, 20, 3)
    np.testing.assert_array_equal(cloud.camera_to_world, original_poses)
    np.testing.assert_array_equal(cloud.points, original_points)
    # No endpoint marker is invented where either real endpoint is outside.
    assert not np.any(np.all(grids["structural"] == [255, 90, 90], axis=-1))


def test_static_crop_still_rejects_nonfinite_camera_path():
    cloud = _cloud()
    cloud.camera_to_world[0, 0, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _rasterize(cloud, (0, 1, 0, 1), 0.05)


def test_point_splat_keeps_highest_confidence_sample_without_averaging() -> None:
    image, metrics = _point_splat(
        _cloud(), (0.0, 0.05, 0.0, 0.05), 0.025, (-0.15, 0.20)
    )
    np.testing.assert_array_equal(image[1, 0], [0, 255, 0])
    assert metrics == {"point_count": 2, "occupied_cell_count": 1}


def test_point_splat_keeps_vertical_bands_separate() -> None:
    floor, _ = _point_splat(
        _cloud(), (0.0, 0.05, 0.0, 0.05), 0.025, (-0.15, 0.20)
    )
    furniture, metrics = _point_splat(
        _cloud(), (0.0, 0.05, 0.0, 0.05), 0.025, (0.75, 1.40)
    )
    np.testing.assert_array_equal(floor[1, 0], [0, 255, 0])
    np.testing.assert_array_equal(furniture[1, 0], [0, 0, 255])
    assert metrics == {"point_count": 1, "occupied_cell_count": 1}


def test_asymmetric_landmarks_put_forward_up_and_camera_right_right() -> None:
    cloud = _cloud()
    cloud.points = np.asarray(
        [
            [0.5, 0.10, 0.5],   # near-left
            [0.5, 0.10, 2.5],   # forward-left
            [3.5, 0.10, 0.5],   # near-right
        ],
        dtype=np.float32,
    )
    cloud.colors = np.asarray(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        dtype=np.uint8,
    )
    cloud.weights = np.ones(3, dtype=np.float32)
    cloud.ranges = np.ones(3, dtype=np.float32)

    image, _ = _point_splat(cloud, (0.0, 4.0, 0.0, 3.0), 1.0, (-0.15, 0.20))

    np.testing.assert_array_equal(image[2, 0], [255, 0, 0])
    np.testing.assert_array_equal(image[0, 0], [0, 255, 0])
    np.testing.assert_array_equal(image[2, 3], [0, 0, 255])


def test_presentation_changes_positions_only_and_keeps_camera_at_bottom_origin() -> None:
    camera_to_world = np.asarray(
        [
            [-1.0, 0.0, 0.0, 2.0],
            [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    frame = camera_ground_frame_from_camera_to_world(camera_to_world)
    display = frame.world_to_camera_local_display_matrix(0.0)
    cloud = _cloud()
    cloud.camera_to_world = camera_to_world[None]
    cloud.points = np.asarray(
        [
            [2.0, 0.1, 3.5],  # near camera
            [2.0, 0.1, 5.5],  # camera-forward
            [0.5, 0.1, 3.5],  # camera-right
        ],
        dtype=np.float32,
    )
    cloud.colors = np.asarray(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        dtype=np.uint8,
    )
    cloud.weights = np.ones(3, dtype=np.float32)
    cloud.ranges = np.ones(3, dtype=np.float32)

    presented = _present_camera_ground(cloud, display)
    np.testing.assert_allclose(_camera_positions(presented)[0, [0, 2]], [0.0, 0.0])
    assert np.linalg.det(display[:3, :3]) == -1.0
    np.testing.assert_array_equal(presented.camera_to_world, camera_to_world[None])

    image, _ = _point_splat(
        presented,
        (-0.5, 2.0, -0.5, 3.0),
        0.5,
        (-0.15, 0.20),
    )
    # The green forward landmark is above the red near-camera landmark.
    green_row, green_column = np.argwhere(
        np.all(image == [0, 255, 0], axis=2)
    )[0]
    red_row, red_column = np.argwhere(np.all(image == [255, 0, 0], axis=2))[0]
    blue_row, blue_column = np.argwhere(np.all(image == [0, 0, 255], axis=2))[0]
    assert green_row < red_row
    assert blue_row == red_row
    assert blue_column > red_column
    assert green_column == red_column
