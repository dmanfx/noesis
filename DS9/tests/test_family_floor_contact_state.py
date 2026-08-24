"""Estimator-side Family contact geometry/state regressions."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = ROOT / "DS9"


def _run_state_probe() -> dict[str, object]:
    script = r'''
import json
import sys
from pathlib import Path

adapter = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
sys.path = [str(adapter), str(repo)] + [
    value
    for value in sys.path
    if value and Path(value).resolve() not in {adapter, repo}
]

from noesis.pipelines import hooks

processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
track = {}
range_rejected = not processor._floor_contact_bbox_gate(
    # Keep the anchor within the existing 20% bbox-gap allowance.  The
    # projected-floor range disagreement is the only reason this contact is
    # rejected.
    anchor_uv=(188.34, 449.0),
    bbox=(127.5, 286.5, 121.6875, 202.5),
    incidence_sin=0.0863,
    track=track,
    anchor_range_m=18.8,
    bbox_bottom_range_m=13.0,
)

state = hooks.PersonGroundState(
    last_good_world=(4.0, 0.0, 8.0),
    last_good_ts=10.0,
    world_x=4.0,
    world_z=8.0,
    filtered_ts=10.0,
    vel_world_x=0.25,
    vel_world_z=-0.10,
    trail_segment_id=7,
)
predicted = hooks.advance_human_cv_prediction(
    state,
    floor_y=0.0,
    now_ts=10.1,
    config=hooks.HumanGroundConfig(),
    reason="bbox_contact_range_disagreement_near_horizon",
)
print(json.dumps({
    "range_rejected": range_rejected,
    "reason": track.get("world_floor_contact_rejection_reason"),
    "range_delta_m": track.get("world_floor_contact_range_delta_m"),
    "predicted": predicted.tolist() if predicted is not None else None,
    "last_good_world": list(state.last_good_world) if state.last_good_world else None,
    "trail_segment_id": state.trail_segment_id,
    "trail_break_required": state.trail_break_required,
}, sort_keys=True))
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(DS9_ROOT), str(ROOT)],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_bbox_bottom_range_disagreement_rejects_far_pose_ray() -> None:
    result = _run_state_probe()

    assert result["range_rejected"] is True
    assert result["reason"] == "bbox_contact_range_disagreement_near_horizon"
    assert result["range_delta_m"] == pytest.approx(5.8)


def test_bad_contact_rejection_keeps_valid_prior_hold_and_trail_lifecycle() -> None:
    result = _run_state_probe()

    # The gate is estimator-side input rejection. A valid prior remains
    # available to the bounded hold and does not force a trail break merely
    # because one pose ankle was rejected.
    assert result["predicted"] is not None
    assert result["last_good_world"] == pytest.approx([4.0, 0.0, 8.0])
    assert result["trail_segment_id"] == 7
    assert result["trail_break_required"] is False
