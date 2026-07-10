"""Regression tests for V3DT/BEV world-source recognition and pose gating."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np

from noesis.telemetry.bev import BevRenderer
from noesis.pipelines.hooks_v3dt_reimpl import (
    V3DT_WORLD_SOURCE_BBOX3D_FOOT,
    _AnalyticsTelemetryProcessor,
)


def test_bev_treats_v3dt_bbox3d_foot_as_live_tracking() -> None:
    assert BevRenderer._world_source_is_live_tracking("v3dt_bbox3d_foot")
    assert BevRenderer._world_source_is_live_tracking("bbox3d")
    assert BevRenderer._world_source_is_depth_fused("v3dt_bbox3d_foot")
    assert not BevRenderer._world_source_is_live_tracking("anchor_hold")


def test_world_track_key_prefers_stable_id() -> None:
    proc = object.__new__(_AnalyticsTelemetryProcessor)
    key = _AnalyticsTelemetryProcessor._world_track_key(
        proc,
        1,
        {"stable_id": 7, "tracker_id": 99, "track_id": 99},
    )
    assert key == (1, 7)
    key2 = _AnalyticsTelemetryProcessor._world_track_key(
        proc,
        2,
        {"tracker_id": 42},
    )
    assert key2 == (2, 42)


def test_extract_stable_id_pose_inputs_respects_needs_pose_update() -> None:
    proc = object.__new__(_AnalyticsTelemetryProcessor)
    calls = {"n": 0}

    def _payload(_obj: Any) -> Dict[str, Any]:
        calls["n"] += 1
        return {
            "features": {"h": 1.0},
            "kpt_mean_conf": 0.9,
        }

    proc._extract_pose_payload = _payload  # type: ignore[attr-defined]

    mgr = SimpleNamespace(pose_enabled=True)

    def needs_pose_update(_sid: int, _tid: int, _ts: float) -> bool:
        return False

    mgr.needs_pose_update = needs_pose_update

    feats, qual = _AnalyticsTelemetryProcessor._extract_stable_id_pose_inputs(
        proc,
        object(),
        mgr,
        sensor_id=0,
        track_id=1,
        now_ts=time.time(),
    )
    assert feats is None and qual is None
    assert calls["n"] == 0

    mgr.needs_pose_update = lambda *_a, **_k: True
    feats, qual = _AnalyticsTelemetryProcessor._extract_stable_id_pose_inputs(
        proc,
        object(),
        mgr,
        sensor_id=0,
        track_id=1,
        now_ts=time.time(),
    )
    assert feats == {"h": 1.0}
    assert qual is not None
    assert calls["n"] == 1


def test_refine_seeded_world_sets_trail_append_fields() -> None:
    proc = object.__new__(_AnalyticsTelemetryProcessor)
    proc.bev_calibration = None  # type: ignore[attr-defined]
    track: Dict[str, Any] = {
        "stable_id": 3,
        "tracker_id": 11,
        "world": [1.0, 0.0, 2.0],
        "world_valid": True,
        "world_source": V3DT_WORLD_SOURCE_BBOX3D_FOOT,
        "image_base": [100.0, 200.0],
    }
    # Without calibration, refine should still preserve validity/quality defaults.
    _AnalyticsTelemetryProcessor._refine_seeded_world_with_ground_state(
        proc,
        0,
        "kitchen",
        track,
        world_source_label=V3DT_WORLD_SOURCE_BBOX3D_FOOT,
    )
    assert track.get("world_valid") is True
    assert track.get("world_source") == V3DT_WORLD_SOURCE_BBOX3D_FOOT
    assert track.get("world_quality") == "good"
