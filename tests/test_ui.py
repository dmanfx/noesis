import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from geometry.depth_source import DepthSummary, MapAnythingDepthSource
from mapanything_config import load_service_config


def test_depth_summary_valid_ratio():
    source = MapAnythingDepthSource(load_service_config())
    depth = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
    conf = np.array([[0.9, 0.95], [0.4, 0.92]], dtype=np.float32)
    mask = np.array([[True, True], [True, False]], dtype=bool)
    summary = source._compute_summary(depth, conf, mask)
    assert isinstance(summary, DepthSummary)
    assert summary.valid_ratio > 0
    assert summary.conf_mean <= 1.0
