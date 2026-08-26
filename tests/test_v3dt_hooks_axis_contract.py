from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DS9_ROOT = REPO_ROOT / "DS9"


@pytest.mark.parametrize(
    ("adapter", "tracker_config"),
    (
        (DS9_ROOT, "DS9/config/v3dt/nvtracker_v3dt.yaml"),
    ),
)
def test_ds9_hooks_enforce_fail_closed_v3dt_axis_contract(
    adapter: Path,
    tracker_config: str,
) -> None:
    code = r'''
import sys
from pathlib import Path
from types import SimpleNamespace

adapter = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
tracker_config = sys.argv[3]
sys.path = [str(adapter), str(repo)] + [
    value
    for value in sys.path
    if value and Path(value).resolve() not in {adapter, repo}
]

from noesis.pipelines import hooks


def processor(v3dt):
    pipeline = SimpleNamespace(
        config={
            "models": {},
            "tracking_mode": "v3dt",
            "tracker": {"config-file": tracker_config},
            "v3dt": v3dt,
        }
    )
    return hooks._AnalyticsTelemetryProcessor(
        pipeline=pipeline,
        tracking_pub=object(),
        camera_labels={0: "living-room"},
        sensor_id_map={},
        tracking_mode="v3dt",
    )


active = processor({"world_frame": "backend_world_m", "caminfo_world_axes": "xzy"})
world = active._world_from_bbox3d(
    {"xCentre": 19.78, "yCentre": 21.493, "zCentre": 0.925, "zLen": 1.85}
)
assert world == [19.78, 0.0, 21.493], world

active._v3dt_caminfo_cache[0] = (
    "projectionMatrix_3x4_w2p",
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
)
image_base = active._image_base_from_bbox3d(
    0,
    {"xCentre": 2.0, "yCentre": 4.0, "zCentre": 1.0, "zLen": 2.0},
    (1920, 1080),
)
assert image_base == [0.5, 0.5], image_base

stale = {
    "bbox3d": {"xCentre": 1.0},
    "world": [9.0, 8.0, 7.0],
    "world_valid": True,
    "world_quality": "good",
    "world_frame": "backend_world_m",
    "world_source": "pose_floor_only",
}
active._augment_track_with_world(0, "living-room", stale)
assert stale == {"bbox3d": {"xCentre": 1.0}}, stale

for invalid in (
    {"world_frame": "camera_local", "caminfo_world_axes": "xzy"},
    {"world_frame": "backend_world_m"},
    {"world_frame": "backend_world_m", "caminfo_world_axes": "xxz"},
):
    try:
        processor(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid V3DT config was accepted: {invalid!r}")
'''
    env = dict(os.environ)
    env.pop("NOESIS_TRACKING_MODE", None)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(adapter), str(REPO_ROOT), env.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(adapter), str(REPO_ROOT), tracker_config],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30.0,
    )
    assert result.returncode == 0, result.stdout + result.stderr
