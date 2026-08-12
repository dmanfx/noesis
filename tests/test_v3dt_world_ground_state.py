"""Regression tests for V3DT/BEV world-source recognition and pose gating."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from noesis.pipelines.hooks_v3dt_reimpl import (
    V3DT_WORLD_SOURCE_BBOX3D_FOOT,
    _AnalyticsTelemetryProcessor,
)
from noesis.telemetry.bev import BevPublicationReceipt, BevRenderer
from noesis.telemetry.publishers import TrackingPublicationReceipt


def test_bev_treats_v3dt_bbox3d_foot_as_live_tracking() -> None:
    assert BevRenderer._world_source_is_live_tracking("v3dt_bbox3d_foot")
    assert BevRenderer._world_source_is_live_tracking("bbox3d")
    assert BevRenderer._world_source_is_depth_fused("v3dt_bbox3d_foot")
    assert not BevRenderer._world_source_is_live_tracking("anchor_hold")


def test_world_track_key_is_owned_by_physical_tracker_not_stable_identity() -> None:
    proc = object.__new__(_AnalyticsTelemetryProcessor)
    key = _AnalyticsTelemetryProcessor._world_track_key(
        proc,
        1,
        {"stable_id": 7, "tracker_id": 99, "track_id": 99},
    )
    assert key == (1, 99)
    key2 = _AnalyticsTelemetryProcessor._world_track_key(
        proc,
        2,
        {"tracker_id": 42},
    )
    assert key2 == (2, 42)
    assert _AnalyticsTelemetryProcessor._world_track_key(  # type: ignore[attr-defined]
        proc,
        3,
        {"stable_id": 7},
    ) is None


def test_v3dt_tracking_gate_forces_zero_transition_then_bounds_heartbeat() -> None:
    proc = object.__new__(_AnalyticsTelemetryProcessor)
    proc._tracking_publish_interval_s = 1.0 / 15.0
    proc._tracking_empty_publish_interval_s = 0.5
    proc._last_tracking_publish_ts_by_sensor = {}
    proc._last_tracking_count_by_sensor = {}

    assert proc._tracking_publish_due(sensor_id=0, now_ts=100.0, count=1)
    proc._last_tracking_publish_ts_by_sensor[0] = 100.0
    proc._last_tracking_count_by_sensor[0] = 1
    assert proc._tracking_publish_due(sensor_id=0, now_ts=100.01, count=0)
    proc._last_tracking_publish_ts_by_sensor[0] = 100.01
    proc._last_tracking_count_by_sensor[0] = 0
    assert not proc._tracking_publish_due(sensor_id=0, now_ts=100.2, count=0)
    assert proc._tracking_publish_due(sensor_id=0, now_ts=100.6, count=0)


def test_v3dt_tracking_and_bev_share_pair_safe_cadence_above_15fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOESIS_REID_TEST_MODE", "1")
    monkeypatch.setenv("NOESIS_TRACKING_PUBLISH_MAX_HZ", "15")
    monkeypatch.setenv("NOESIS_BEV_PUBLISH_MAX_HZ", "12")
    events: list[tuple[str, int]] = []

    class Publisher:
        def __init__(self) -> None:
            self.sequence = 0
            self.submission_id = 0

        def publish(
            self,
            _source_id: int,
            _tracks: list[dict[str, Any]],
            *,
            frame_metadata: dict[str, Any] | None = None,
        ) -> TrackingPublicationReceipt:
            assert frame_metadata is not None
            frame_id = int(frame_metadata["frame_id"])
            observed_at_us = int(frame_metadata["observed_at_us"])
            events.append(("tracking", frame_id))
            self.submission_id += 2
            receipt = TrackingPublicationReceipt(
                source_id=int(_source_id),
                frame_id=frame_id,
                observed_at_us=observed_at_us,
                tracking_publication_sequence=self.sequence,
                outbound_submission_id=self.submission_id,
                outbound_message_count=1,
            )
            self.sequence += 1
            return receipt

    class Renderer:
        def render_and_publish(self, **kwargs: Any) -> BevPublicationReceipt:
            events.append(("bev", int(kwargs["frame_id"])))
            return BevPublicationReceipt(
                status="admitted",
                camera_id=str(kwargs["camera_id"]),
                source_id=int(kwargs["source_id"]),
                frame_id=int(kwargs["frame_id"]),
                observed_at_us=int(kwargs["observed_at_us"]),
                tracking_publication_sequence=int(
                    kwargs["tracking_publication_sequence"]
                ),
                tracking_outbound_submission_id=int(
                    kwargs["tracking_outbound_submission_id"]
                ),
                outbound_submission_id=int(
                    kwargs["tracking_outbound_submission_id"]
                )
                + 1,
            )

        def record_input_failure(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("paired V3DT BEV unexpectedly failed")

    class Calibration:
        def snapshot(self, _source_id: int, _camera_id: str) -> Any:
            return SimpleNamespace(image_size=(1280, 720))

    processor = _AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(config={}, stable_id_mgr=None),
        tracking_pub=Publisher(),
        camera_labels={0: "camera_0"},
        sensor_id_map={},
        bev_renderer=Renderer(),
        bev_calibration=Calibration(),
    )
    now = [100.0]
    monkeypatch.setattr(
        "noesis.pipelines.hooks_v3dt_reimpl.time.time",
        lambda: now[0],
    )

    def empty_frame(frame_id: int) -> SimpleNamespace:
        return SimpleNamespace(
            source_id=0,
            frame_number=frame_id,
            frame_width=1280,
            frame_height=720,
            buf_pts=frame_id * 1_000,
            object_items=[],
        )

    frame_times: dict[int, float] = {}
    for frame_id in range(13):
        now[0] = 100.0 + frame_id / 30.0
        frame_times[frame_id] = now[0]
        processor.handle_frame_ds8(empty_frame(frame_id))

    assert len(events) >= 8 and len(events) % 2 == 0
    assert all(
        events[index][0] == "tracking"
        and events[index + 1] == ("bev", events[index][1])
        for index in range(0, len(events), 2)
    )
    paired_frames = [events[index][1] for index in range(0, len(events), 2)]
    assert all(
        frame_times[right] - frame_times[left] >= (1.0 / 12.0) - 1e-9
        for left, right in zip(paired_frames, paired_frames[1:])
    )

    monkeypatch.delenv("NOESIS_REID_TEST_MODE")
    now[0] += 0.01
    processor.handle_frame_ds8(empty_frame(100))
    monkeypatch.setenv("NOESIS_REID_TEST_MODE", "1")
    now[0] += 0.01
    processor.handle_frame_ds8(empty_frame(101))
    assert events[-4:] == [
        ("tracking", 100),
        ("bev", 100),
        ("tracking", 101),
        ("bev", 101),
    ]


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
