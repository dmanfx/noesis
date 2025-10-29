import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines import mapanything_depth_postprocess as ma_post


def test_select_layers_matches_expected_keys():
    layers = {
        "depth_z": np.ones((1, 1, 4, 4), dtype=np.float32),
        "confidence": np.linspace(0.2, 0.9, 16, dtype=np.float32).reshape(1, 1, 4, 4),
        "mask": np.ones((1, 1, 4, 4), dtype=np.float32),
        "scale": np.array([1.23], dtype=np.float32),
        "pose": np.arange(16, dtype=np.float32),
        "other": np.arange(5, dtype=np.float32),
    }
    bundle = ma_post.select_layers(layers)
    assert bundle.depth is not None and bundle.depth.shape == (1, 1, 4, 4)
    assert bundle.confidence is not None and np.isclose(bundle.confidence.min(), 0.2)
    assert bundle.mask is not None
    assert math.isclose(bundle.scale, 1.23, rel_tol=1e-6)
    assert bundle.pose is not None and bundle.pose.size == 16
    assert "other" in bundle.extras


def test_compute_depth_summary_applies_conf_and_mask():
    depth = np.array(
        [
            [1.0, 1.1, 1.2, 1.3],
            [1.4, 1.5, 1.6, 1.7],
            [1.8, 1.9, 2.0, np.inf],
            [0.0, 2.1, 2.2, 2.3],
        ],
        dtype=np.float32,
    )
    conf = np.linspace(0.1, 1.0, depth.size, dtype=np.float32).reshape(depth.shape)
    mask = np.ones_like(depth, dtype=bool)
    mask[0, 0] = False
    summary = ma_post.compute_depth_summary(depth, conf, mask, min_conf=0.5)
    assert summary["sample_count"] == 7
    assert 1.0 < summary["median"] < 2.1
    assert summary["valid_ratio"] < 1.0
    assert summary["conf_mean"] >= 0.5


@pytest.mark.parametrize(
    "anchor,bbox,depth_shape,expected",
    [
        ((10.0, 20.0), (0.0, 0.0, 100.0, 100.0), (518, 518), (51.7, 103.6)),
        ((0.0, 0.0), (10.0, 20.0, 50.0, 50.0), (100, 200), (0.0, 0.0)),
        ((60.0, 70.0), (10.0, 20.0, 50.0, 50.0), (100, 200), (199.0, 99.0)),
    ],
)
def test_anchor_to_depth_indices(anchor, bbox, depth_shape, expected):
    cx, cy = ma_post.anchor_to_depth_indices(anchor, bbox, depth_shape)
    ex, ey = expected
    assert 0.0 <= cx <= depth_shape[1] - 1
    assert 0.0 <= cy <= depth_shape[0] - 1
    assert cx == pytest.approx(ex, rel=1e-2)
    assert cy == pytest.approx(ey, rel=1e-2)


def test_sample_depth_window_prefers_valid_samples():
    depth = np.full((8, 8), 1.5, dtype=np.float32)
    depth[3:5, 3:5] = 2.4
    conf = np.full_like(depth, 0.6, dtype=np.float32)
    mask = np.zeros_like(depth, dtype=bool)
    mask[3:5, 3:5] = True
    depth_val, conf_val, count = ma_post.sample_depth_window(
        depth,
        center_xy=(3.8, 3.6),
        window_sizes=(5, 7),
        confidence=conf,
        mask=mask,
        min_conf=0.5,
    )
    assert pytest.approx(depth_val, rel=1e-6) == 2.4
    assert pytest.approx(conf_val, rel=1e-6) == 0.6
    assert count == 4


def test_sample_depth_window_returns_zero_on_no_valid_samples():
    depth = np.zeros((5, 5), dtype=np.float32)
    conf = np.zeros_like(depth, dtype=np.float32)
    depth_val, conf_val, count = ma_post.sample_depth_window(
        depth,
        center_xy=(2.0, 2.0),
        window_sizes=(3,),
        confidence=conf,
        mask=None,
        min_conf=0.9,
    )
    assert depth_val == 0.0
    assert conf_val == 0.0
    assert count == 0


def test_sanitize_helpers():
    assert ma_post.sanitize_scale(1.5) == pytest.approx(1.5)
    assert ma_post.sanitize_scale(float("nan")) is None
    assert ma_post.sanitize_pose(np.arange(4, dtype=np.float32)) == (0.0, 1.0, 2.0, 3.0)
    assert ma_post.sanitize_pose(np.array([0.0, np.nan])) is None
