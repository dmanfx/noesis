from __future__ import annotations

import numpy as np

from tools.mapanything_phone_scan import alignment as a


def _camera_visibility():
    depth = np.full((125, 125), 5.0)
    camera_from_world = np.eye(4)
    intrinsics = np.asarray([[400., 0, 500], [0, 400., 500], [0, 0, 1]])

    def comparable(points):
        return a._fixed_camera_comparable_mask(
            points, depth, camera_from_world, intrinsics,
            cell_px=8, occlusion_tolerance_m=.30,
        )[0]

    return comparable


def test_hidden_vertical_surface_cannot_pull_visible_wall_fit():
    x, y = np.meshgrid(np.linspace(-1.5, 1.5, 30), np.linspace(-1.5, 1.5, 30))
    target = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 5.0)))
    normals = np.tile([0., 0., 1.], (len(target), 1))
    visible = target - np.asarray([0, 0, .1])
    hidden = np.repeat(target + np.asarray([0, 0, .4]), 3, axis=0)
    source = np.vstack((visible, hidden))
    initial = np.zeros(3)
    # Only depth translation is identifiable from this one-plane fixture.
    bounds = np.asarray([1e-6, 1e-6, .6])
    original, _ = a._structure_refine(
        source, target, normals, source, initial, bounds_delta=bounds,
    )
    corrected, metrics = a._structure_refine(
        source, target, normals, source, initial, bounds_delta=bounds,
        comparable_mask=_camera_visibility(),
    )
    assert abs(original[2] - .1) > .2
    assert abs(corrected[2] - .1) < 1e-4
    assert metrics["comparable_point_count"] == len(visible)
    assert metrics["source_overlap_0_30m"] == 1.0


def test_visibility_keeps_incorrect_foreground_for_scoring():
    points = np.asarray([[0., 0., 4.0], [0., 0., 5.1], [0., 0., 5.4], [0., 0., -1.]])
    np.testing.assert_array_equal(_camera_visibility()(points), [True, True, False, False])


def test_empty_visible_support_has_no_source_overlap():
    target = np.asarray([[0., 0., 5.0], [.1, 0., 5.0]])
    hidden = target + np.asarray([0, 0, 1.])
    score = a._structure_score(
        np.zeros(3), hidden, target, np.tile([0., 0., 1.], (2, 1)), hidden,
        comparable_mask=_camera_visibility(),
    )
    assert score["comparable_point_count"] == 0
    assert score["source_overlap_0_30m"] == 0.0
    assert score["plane_residual_median_m"] >= 1.0
