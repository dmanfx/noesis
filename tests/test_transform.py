import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from geometry.transform import project_pixel_with_depth, floor_intersection_world
except Exception:
    project_pixel_with_depth = None  # type: ignore[assignment]
    floor_intersection_world = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    project_pixel_with_depth is None or floor_intersection_world is None,
    reason="geometry.transform legacy API helpers are unavailable in this build.",
)


def _identity4x4_col_major():
    mat = np.eye(4, dtype=float)
    return mat.reshape(-1, order='F').tolist()


def _basic_calibration_bundle():
    return {
        'cameras': {
            'K': {
                'cam': [100.0, 100.0, 0.0, 0.0],
            },
            'E': {
                'cam': _identity4x4_col_major(),
            },
        },
        'align': {
            'matrix': _identity4x4_col_major(),
            'floor_y': 0.0,
            'units': {'s_obj_to_m': 1.0},
        },
    }


def test_project_pixel_with_depth_identity():
    calib = _basic_calibration_bundle()
    point = project_pixel_with_depth('cam', 0.0, 0.0, 2.0, calib)
    assert point is not None
    assert np.allclose(point, np.array([0.0, 0.0, 2.0], dtype=float), atol=1e-6)


def test_project_pixel_with_depth_missing_camera_returns_none():
    calib = _basic_calibration_bundle()
    assert project_pixel_with_depth('missing', 0.0, 0.0, 1.0, calib) is None


def test_floor_intersection_world_identity():
    calib = _basic_calibration_bundle()
    hit = floor_intersection_world('cam', 0.0, 10.0, calib)
    assert hit is not None
    hx, hy, hz = hit
    assert pytest.approx(hy, abs=1e-6) == 0.0
    assert pytest.approx(hx, abs=1e-6) == 0.0


def test_floor_intersection_world_missing_returns_none():
    assert floor_intersection_world('cam', 1.0, 1.0, {}, floor_y=0.0) is None
