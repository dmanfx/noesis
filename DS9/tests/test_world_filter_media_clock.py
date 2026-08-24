from __future__ import annotations

from types import SimpleNamespace

from noesis.pipelines import hooks
from noesis.telemetry.person_ground_state import PersonGroundState
from noesis_core.runtime_publication import RuntimePublicationGate


def _processor() -> hooks._AnalyticsTelemetryProcessor:
    return hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(config={}),
        tracking_pub=SimpleNamespace(),
        camera_labels={0: "camera-0", 1: "camera-1"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
    )


def test_world_filter_clock_uses_same_source_media_pts_delta() -> None:
    processor = _processor()
    processor._begin_source_frame_timeline(
        sensor_id=0,
        frame_id=1,
        observed_at_us=10_000_000,
        media_pts_ns=1_000_000_000,
    )
    first = processor._world_timestamp_for_frame(
        0,
        media_pts_ns=1_000_000_000,
        observed_ts=10.0,
    )

    # The callback is deliberately delayed by almost a second.  The world
    # filter must advance by the source's 30fps PTS, not callback scheduling.
    processor._begin_source_frame_timeline(
        sensor_id=0,
        frame_id=2,
        observed_at_us=10_900_000,
        media_pts_ns=1_033_333_333,
    )
    second = processor._world_timestamp_for_frame(
        0,
        media_pts_ns=1_033_333_333,
        observed_ts=10.9,
    )
    assert abs((second - first) - (1 / 30)) < 1e-9


def test_world_filter_clock_fallback_never_rewinds_after_missing_pts() -> None:
    processor = _processor()
    processor._begin_source_frame_timeline(
        sensor_id=0,
        frame_id=1,
        observed_at_us=20_000_000,
        media_pts_ns=1_000_000_000,
    )
    first = processor._world_timestamp_for_frame(
        0,
        media_pts_ns=1_000_000_000,
        observed_ts=20.0,
    )
    missing = processor._world_timestamp_for_frame(
        0,
        media_pts_ns=0,
        observed_ts=20.1,
    )
    assert missing == 20.1

    # PTS recovery is behind the observed fallback.  It is held at the last
    # logical timestamp until media time catches up, avoiding a negative dt.
    recovered = processor._world_timestamp_for_frame(
        0,
        media_pts_ns=1_050_000_000,
        observed_ts=20.2,
    )
    assert first == 20.0
    assert recovered == missing


def test_decreasing_media_pts_creates_new_source_epoch_and_clears_world_state() -> None:
    processor = _processor()
    processor._begin_source_frame_timeline(
        sensor_id=0,
        frame_id=10,
        observed_at_us=30_000_000,
        media_pts_ns=10_000_000_000,
    )
    processor._world_timestamp_for_frame(
        0,
        media_pts_ns=10_000_000_000,
        observed_ts=30.0,
    )
    processor._world_state_by_track[(0, 7)] = PersonGroundState(
        world_x=1.0,
        world_z=2.0,
        filtered_ts=30.0,
    )

    epoch = processor._begin_source_frame_timeline(
        sensor_id=0,
        frame_id=11,
        observed_at_us=30_100_000,
        media_pts_ns=100_000_000,
    )
    assert epoch == 1
    assert (0, 7) not in processor._world_state_by_track
    assert 0 not in processor._world_clock_by_sensor

    restarted = processor._world_timestamp_for_frame(
        0,
        media_pts_ns=100_000_000,
        observed_ts=30.1,
    )
    assert restarted == 30.1
