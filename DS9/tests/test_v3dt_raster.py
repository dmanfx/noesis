from __future__ import annotations

import numpy as np
import pytest

from noesis.v3dt_raster import scale_projection_matrix, validate_raster


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ((1920, 1080), (1920, 1080)),
        ([640, 384], (640, 384)),
        ((np.int64(1920), np.int32(1088)), (1920, 1088)),
    ],
)
def test_validate_raster_accepts_positive_integer_pairs(value, expected) -> None:
    assert validate_raster(value, "raster") == expected


@pytest.mark.parametrize(
    "value",
    [
        (True, 1080),
        (1920, False),
        (1920.0, 1080),
        (1920, 1080.0),
        ("1920", 1080),
        (0, 1080),
        (1920, -1),
        (1920,),
        (1920, 1080, 1),
        "1920x1080",
        None,
    ],
)
def test_validate_raster_rejects_non_integer_or_invalid_pairs(value) -> None:
    with pytest.raises(ValueError, match="raster"):
        validate_raster(value, "raster")


def _project(projection: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points_xyz, np.ones(len(points_xyz))))
    pixels_h = (projection @ homogeneous.T).T
    return pixels_h[:, :2] / pixels_h[:, 2:3]


def test_scaled_projection_matches_affine_pixel_resize_and_inverts() -> None:
    projection = np.asarray(
        [
            [812.0, 3.0, 959.5, 12.0],
            [2.0, 806.0, 539.5, -8.0],
            [0.01, -0.02, 1.0, 0.5],
        ],
        dtype=np.float64,
    )
    points = np.asarray(
        [
            [0.2, -0.1, 3.0],
            [-1.4, 0.8, 6.5],
            [2.0, 1.2, 9.0],
            [-0.5, -2.1, 4.2],
        ],
        dtype=np.float64,
    )
    mux_size = (1920, 1080)
    tracker_size = (1920, 1088)

    tracker_projection = scale_projection_matrix(
        projection, mux_size, tracker_size
    )
    mux_pixels = _project(projection, points)
    tracker_pixels = _project(tracker_projection, points)
    expected = mux_pixels * np.asarray(
        (tracker_size[0] / mux_size[0], tracker_size[1] / mux_size[1])
    )

    np.testing.assert_allclose(tracker_projection[0], projection[0], rtol=0, atol=0)
    np.testing.assert_allclose(tracker_projection[2], projection[2], rtol=0, atol=0)
    np.testing.assert_allclose(tracker_pixels, expected, rtol=1e-12, atol=1e-10)
    restored_projection = scale_projection_matrix(
        tracker_projection, tracker_size, mux_size
    )
    np.testing.assert_allclose(
        restored_projection, projection, rtol=1e-12, atol=1e-10
    )
    np.testing.assert_allclose(
        _project(restored_projection, points), mux_pixels, rtol=1e-12, atol=1e-10
    )


def test_scaled_projection_accepts_flat_twelve_values_and_scales_both_axes() -> None:
    flat = [100.0, 0.0, 320.0, 0.0, 0.0, 120.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    scaled = scale_projection_matrix(flat, (640, 480), (320, 240))
    np.testing.assert_allclose(
        scaled,
        np.asarray(flat, dtype=np.float64).reshape(3, 4)
        * np.asarray((0.5, 0.5, 1.0))[:, np.newaxis],
    )


@pytest.mark.parametrize(
    "projection",
    [
        np.zeros((4, 3)),
        np.zeros((2, 6)),
        np.zeros(11),
        np.full((3, 4), np.nan),
        np.full((3, 4), np.inf),
    ],
)
def test_scale_projection_rejects_bad_shape_or_nonfinite_values(projection) -> None:
    with pytest.raises(ValueError, match="projection"):
        scale_projection_matrix(projection, (1920, 1080), (1920, 1088))


def test_scale_projection_validates_both_rasters() -> None:
    with pytest.raises(ValueError, match="from_size"):
        scale_projection_matrix(np.eye(3, 4), (1920.0, 1080), (1920, 1088))
    with pytest.raises(ValueError, match="to_size"):
        scale_projection_matrix(np.eye(3, 4), (1920, 1080), (1920, 0))
