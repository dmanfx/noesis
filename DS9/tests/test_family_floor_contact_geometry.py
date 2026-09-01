"""Focused estimator-side guard for distant Family Room pose contacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = ROOT / "DS9"


def _contact_gate_results() -> dict[str, object]:
    script = r'''
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

adapter = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
sys.path = [str(adapter), str(repo)] + [
    value
    for value in sys.path
    if value and Path(value).resolve() not in {adapter, repo}
]

from noesis.pipelines import hooks

processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
bbox = [116.625, 281.625, 122.25, 207.375]
cases = {
    # Retained Family replay frame 1053: pose ankle above the detector
    # silhouette, with a near-horizon floor ray.
    "pose_far_near_horizon": ((178.48, 439.68), 0.07877519),
    # The same elevated contact is not rejected solely because of its bbox
    # gap when incidence is healthy.
    "pose_far_healthy_incidence": ((178.48, 439.68), 0.123),
    # The detector bottom is the physically plausible cross-room contact.
    "bbox_bottom_near_horizon": ((177.75, 489.0), 0.123),
    # A hallucinated ankle materially below its detector silhouette is
    # impossible regardless of ray incidence.
    "pose_below_bbox_healthy_incidence": ((177.75, 516.0), 0.25),
}
result = {}
for name, (anchor_uv, incidence_sin) in cases.items():
    track = {}
    result[name] = {
        "admitted": bool(
            processor._floor_contact_bbox_gate(
                anchor_uv=anchor_uv,
                bbox=bbox,
                incidence_sin=incidence_sin,
                track=track,
            )
        ),
        "gap_px": track.get("world_floor_contact_gap_px"),
        "gap_ratio": track.get("world_floor_contact_gap_ratio"),
        "reason": track.get("world_floor_contact_rejection_reason"),
    }

class _Profile:
    floor_ray_max_range_m = 22.0

class _Policy:
    def profile(self, _camera_id):
        return _Profile()

processor.world_fusion_policy = _Policy()
calib = SimpleNamespace(
    # Active Family target-frame E, column-major.
    extrinsics_col_major=(
        0.92155567, -0.16023899, 0.35363628, 0.0,
        -0.05962848, -0.95846445, -0.27890921, 0.0,
        0.38363993, 0.23594357, -0.89283315, 0.0,
        -18.91982309, 1.07650679, 6.18886151, 1.0,
    ),
    intrinsics=np.array(
        [
            [759.8470119659571, 0.0, 937.3184106054618],
            [0.0, 758.693910792606, 524.9146561743939],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    image_size=(1920, 1080),
    floor_y=0.0,
    unit_scale=1.0,
)
for name, anchor_uv, candidate in (
    (
        "pose_far_near_horizon_full_gate",
        (178.48, 439.68),
        [7.41254053, 0.0, -6.47675511],
    ),
    (
        "bbox_bottom_full_gate",
        (177.75, 489.0),
        [10.17231028, 0.0, 0.46593501],
    ),
):
    track = {}
    result[name] = {
        "admitted": bool(
            processor._admit_floor_ray_range(
                "family-room",
                calib=calib,
                floor_candidate=candidate,
                track=track,
                anchor_uv=anchor_uv,
                bbox=bbox,
            )
        ),
        "range_m": track.get("world_floor_range_m"),
        "incidence_sin": track.get("world_floor_incidence_sin"),
        "reason": track.get("world_floor_rejection_reason"),
    }

# Retained frame 1217 passed the original 20% pixel-gap test, but the pose
# ray still landed 5+ metres farther away than the detector-bottom floor ray.
track = {}
result["pose_range_disagreement_full_gate"] = {
    "admitted": bool(
        processor._admit_floor_ray_range(
            "family-room",
            calib=calib,
            floor_candidate=[7.900197238971199, 0.0, -4.710416924121912],
            track=track,
            anchor_uv=(188.34, 449.0),
            bbox=[127.5, 286.5, 121.6875, 202.5],
        )
    ),
    "range_m": track.get("world_floor_range_m"),
    "bbox_bottom_range_m": track.get("world_floor_bbox_bottom_range_m"),
    "range_delta_m": track.get("world_floor_contact_range_delta_m"),
    "reason": track.get("world_floor_rejection_reason"),
}
print(json.dumps(result, sort_keys=True))
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(DS9_ROOT), str(ROOT)],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_family_pose_contact_gate_rejects_far_near_horizon_ankle() -> None:
    result = _contact_gate_results()

    rejected = result["pose_far_near_horizon"]
    assert rejected["admitted"] is False
    assert rejected["gap_px"] == pytest.approx(49.32)
    assert rejected["gap_ratio"] > 0.20
    assert rejected["reason"] == "bbox_contact_gap_near_horizon"

    # The guard is conditioned on ray incidence and does not turn ordinary
    # bbox padding into a global pose rejection.
    assert result["pose_far_healthy_incidence"]["admitted"] is True
    assert result["bbox_bottom_near_horizon"]["admitted"] is True

    below = result["pose_below_bbox_healthy_incidence"]
    assert below["admitted"] is False
    assert below["gap_px"] == pytest.approx(-27.0)
    assert below["reason"] == "floor_contact_below_detector_silhouette"

    range_rejected = result["pose_range_disagreement_full_gate"]
    assert range_rejected["admitted"] is False
    assert range_rejected["range_m"] == pytest.approx(18.8, abs=0.1)
    assert range_rejected["bbox_bottom_range_m"] == pytest.approx(13.0, abs=0.2)
    assert range_rejected["range_delta_m"] > 5.0
    assert (
        range_rejected["reason"]
        == "bbox_contact_range_disagreement_near_horizon"
    )
