from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = REPO_ROOT / "DS9"


def test_v3dt_missing_bbox3d_omits_world_without_baseline_projection() -> None:
    code = r'''
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

class ForbiddenCalibration:
    def snapshot(self, *_args, **_kwargs):
        raise AssertionError("V3DT missing-bbox path consulted baseline calibration")

processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
processor._tracking_mode = "v3dt"
processor.bev_calibration = ForbiddenCalibration()
track = {
    "bbox": [10.0, 20.0, 30.0, 40.0],
    "image_size": [1920, 1080],
    "world": [1.0, 2.0, 3.0],
    "world_valid": True,
    "world_quality": "good",
    "world_quality_reason": "stale-baseline",
    "world_frame": "backend_world_m",
    "world_source": "pose_floor_only",
}
processor._augment_track_with_world(0, "living-room", track)
for field in (
    "world",
    "world_valid",
    "world_quality",
    "world_quality_reason",
    "world_frame",
    "world_source",
):
    assert field not in track, (field, track)
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(DS9_ROOT), str(REPO_ROOT), env.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(DS9_ROOT), str(REPO_ROOT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_v3dt_and_baseline_preseeded_world_fail_closed_without_calibration() -> None:
    code = r'''
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
processor._tracking_mode = "v3dt"
processor.bev_calibration = None
v3dt = {
    "bbox3d": {"xCentre": 1.0},
    "world": [1.0, 2.0, 3.0],
    "world_valid": True,
    "world_frame": "backend_world_m",
    "world_source": "bbox3d",
}
processor._augment_track_with_world(0, "living-room", v3dt)
assert "world" not in v3dt
assert "world_source" not in v3dt
assert "world_frame" not in v3dt
assert v3dt["world_valid"] is False
assert v3dt["world_quality"] == "invalid"
assert v3dt["world_quality_reason"] == "calibration_unavailable"

processor._tracking_mode = "baseline"
baseline = {
    "world": [4.0, 5.0, 6.0],
    "world_valid": True,
    "world_frame": "backend_world_m",
    "world_source": "pose_floor_only",
}
processor._augment_track_with_world(0, "living-room", baseline)
assert "world" not in baseline
assert "world_source" not in baseline
assert "world_frame" not in baseline
assert baseline["world_valid"] is False
assert baseline["world_quality"] == "invalid"
assert baseline["world_quality_reason"] == "calibration_unavailable"
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(DS9_ROOT), str(REPO_ROOT), env.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(DS9_ROOT), str(REPO_ROOT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
