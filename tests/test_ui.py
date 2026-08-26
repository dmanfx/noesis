import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from geometry.depth_source import DepthSummary, MapAnythingDepthSource
from noesis.config.mapanything import load_service_config


def test_depth_summary_valid_ratio(tmp_path: Path):
    config = load_service_config()
    config = replace(
        config,
        storage=replace(
            config.storage,
            depth_base=str(tmp_path / "depth"),
            calib_base=str(tmp_path / "calibration"),
        ),
    )
    source = MapAnythingDepthSource(config)
    try:
        depth = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
        conf = np.array([[0.9, 0.95], [0.4, 0.92]], dtype=np.float32)
        mask = np.array([[True, True], [True, False]], dtype=bool)
        summary = source._compute_summary(depth, conf, mask)
        assert isinstance(summary, DepthSummary)
        assert summary.valid_ratio > 0
        assert summary.conf_mean <= 1.0
    finally:
        source.close()
