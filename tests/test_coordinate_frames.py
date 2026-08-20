from __future__ import annotations

import numpy as np
import pytest

from noesis_core.coordinate_frames import (
    CoordinateFrameError,
    camera_ground_frame_from_camera_to_world,
    camera_local_raster_indices,
    transform_positions,
)


def _family_reference_camera_to_world() -> np.ndarray:
    right = np.asarray([0.9297270684666662, 0.0, 0.36824934237603063])
    forward = np.asarray([0.36824934237603063, 0.0, -0.9297270684666662])
    down = np.cross(forward, right)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((right, down, forward))
    transform[:3, 3] = [15.419562592882805, 1.6297636700108056, 12.53002550330233]
    return transform


def test_family_display_basis_maps_asymmetric_axes_without_mutating_world() -> None:
    camera_to_world = _family_reference_camera_to_world()
    assert np.linalg.det(camera_to_world[:3, :3]) == pytest.approx(1.0)
    frame = camera_ground_frame_from_camera_to_world(camera_to_world)
    display = frame.world_to_camera_local_display_matrix(0.0)
    assert np.linalg.det(display[:3, :3]) == pytest.approx(-1.0)

    camera = camera_to_world[:3, 3]
    probes = np.stack(
        (
            camera,
            camera + (2.0 * frame.camera_right_world),
            camera + (3.0 * frame.camera_forward_world),
        )
    )
    before = probes.copy()
    local = transform_positions(probes, display)

    np.testing.assert_allclose(probes, before)
    np.testing.assert_allclose(local[0, [0, 2]], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(local[1, [0, 2]], [2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(local[2, [0, 2]], [0.0, 3.0], atol=1e-12)


def test_camera_local_raster_is_row_zero_far_and_column_zero_left() -> None:
    rows, columns, valid = camera_local_raster_indices(
        np.asarray([0.5, 0.5, 3.5]),
        np.asarray([0.5, 2.5, 0.5]),
        min_x_m=0.0,
        min_z_m=0.0,
        resolution_m=1.0,
        rows=3,
        columns=4,
    )
    np.testing.assert_array_equal(rows, [2, 0, 2])
    np.testing.assert_array_equal(columns, [0, 0, 3])
    np.testing.assert_array_equal(valid, [True, True, True])


def test_camera_ground_frame_rejects_improper_metric_pose() -> None:
    reflected = np.eye(4, dtype=np.float64)
    reflected[0, 0] = -1.0
    with pytest.raises(CoordinateFrameError, match="proper"):
        camera_ground_frame_from_camera_to_world(reflected)
