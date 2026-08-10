from __future__ import annotations

import numpy as np

from tools.mapanything_phone_scan.render_phone_heatmap_diagnostics import (
    PhoneCloud,
    _point_splat,
)


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
    np.testing.assert_array_equal(image[0, 0], [0, 255, 0])
    assert metrics == {"point_count": 2, "occupied_cell_count": 1}


def test_point_splat_keeps_vertical_bands_separate() -> None:
    floor, _ = _point_splat(
        _cloud(), (0.0, 0.05, 0.0, 0.05), 0.025, (-0.15, 0.20)
    )
    furniture, metrics = _point_splat(
        _cloud(), (0.0, 0.05, 0.0, 0.05), 0.025, (0.75, 1.40)
    )
    np.testing.assert_array_equal(floor[0, 0], [0, 255, 0])
    np.testing.assert_array_equal(furniture[0, 0], [0, 0, 255])
    assert metrics == {"point_count": 1, "occupied_cell_count": 1}
