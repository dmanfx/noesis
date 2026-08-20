from __future__ import annotations

import numpy as np

from tools.mapanything_phone_scan.render_phone_heatmap_diagnostics import (
    PhoneCloud,
    _camera_positions,
    _point_splat,
    _present_camera_ground,
)
from noesis_core.coordinate_frames import camera_ground_frame_from_camera_to_world


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
