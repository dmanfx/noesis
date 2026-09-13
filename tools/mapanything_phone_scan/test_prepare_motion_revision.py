from __future__ import annotations

import copy
import math

import pytest

from tools.mapanything_phone_scan.prepare_motion_revision import select_motion_subset
from tools.mapanything_phone_scan.prepared_frame_identity import prepared_frame_identity


def _rows():
    return [{"index": i, "frame_id": prepared_frame_identity(i, str(i) * 64), "sha256": str(i) * 64,
             "timestamp_s": float(i), "quality": {"edge_from_previous": {"left": i - 1}},
             "imu_motion": {"status": "available", "rotation_during_exposure_estimate_rad": math.radians(deg)}}
            for i, deg in enumerate([0.1, 4, 0.2, 0.3])]


def test_motion_filter_keeps_parent_identity_and_does_not_relabel_gap_edge():
    rows = _rows()
    original = copy.deepcopy(rows)
    keep, reject = select_motion_subset(rows, 1.0)
    assert rows == original
    assert [r["parent_prepared_index"] for r in keep] == [0, 2, 3]
    assert reject[0]["original_frame"] == original[1]
    assert keep[1]["parent_prepared_frame_id"] == original[2]["frame_id"]
    assert keep[1]["frame_id"] == prepared_frame_identity(1, "2" * 64)
    assert keep[1]["quality"]["edge_from_previous"] is None
    assert keep[1]["quality"]["parent_edge_from_previous"] == {"left": 1}
    assert keep[2]["quality"]["edge_from_previous"] == {"left": 2}


def test_unknown_motion_is_retained_not_invented():
    rows = _rows()
    rows[1]["imu_motion"] = {"status": "unavailable", "reason": "coverage gap"}
    keep, reject = select_motion_subset(rows, 1.0)
    assert len(keep) == 4 and not reject


@pytest.mark.parametrize("limit", [0, 0.49, 11, math.inf, math.nan])
def test_motion_limit_is_explicit_finite_and_bounded(limit):
    with pytest.raises(ValueError):
        select_motion_subset(_rows(), limit)


def test_motion_filter_rejects_bad_identity_and_nonfinite_scores():
    rows = _rows()
    rows[1]["frame_id"] = rows[0]["frame_id"]
    with pytest.raises(ValueError, match="identity"):
        select_motion_subset(rows, 1)
    rows = _rows()
    rows[1]["imu_motion"]["rotation_during_exposure_estimate_rad"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        select_motion_subset(rows, 1)
