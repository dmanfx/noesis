from __future__ import annotations

import threading
import time

import pytest

from noesis_core.source_progress import (
    DecodedSourceProgressMonitor,
    SourceProgressFailure,
    SourceProgressPolicy,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _policy(**overrides: float | int) -> SourceProgressPolicy:
    values: dict[str, float | int] = {
        "stall_seconds": 2.0,
        "recovery_interval_seconds": 1.0,
        "max_recovery_attempts": 4,
        "startup_grace_seconds": 1.0,
        "poll_interval_seconds": 60.0,
        "fps_window_seconds": 5.0,
    }
    values.update(overrides)
    return SourceProgressPolicy(**values)


def test_policy_environment_is_strict_and_bounded() -> None:
    policy = SourceProgressPolicy.from_environment(
        {
            "NOESIS_SOURCE_PROGRESS_STALL_SECONDS": "12.5",
            "NOESIS_SOURCE_PROGRESS_RECOVERY_INTERVAL_SECONDS": "3",
            "NOESIS_SOURCE_PROGRESS_MAX_ATTEMPTS": "5",
            "NOESIS_SOURCE_PROGRESS_STARTUP_GRACE_SECONDS": "30",
            "NOESIS_SOURCE_PROGRESS_POLL_SECONDS": "0.5",
            "NOESIS_SOURCE_PROGRESS_FPS_WINDOW_SECONDS": "4",
        }
    )
    assert policy.stall_seconds == 12.5
    assert policy.recovery_interval_seconds == 3.0
    assert policy.max_recovery_attempts == 5

    with pytest.raises(ValueError, match="MAX_ATTEMPTS"):
        SourceProgressPolicy.from_environment(
            {"NOESIS_SOURCE_PROGRESS_MAX_ATTEMPTS": "forever"}
        )
    with pytest.raises(ValueError, match="STALL_SECONDS"):
        SourceProgressPolicy.from_environment(
            {"NOESIS_SOURCE_PROGRESS_STALL_SECONDS": "inf"}
        )


def test_brief_stall_recovers_only_when_decoded_progress_resumes() -> None:
    clock = _Clock()
    monitor = DecodedSourceProgressMonitor(
        {0: "living-room", 1: "kitchen"}, policy=_policy(), clock=clock
    )
    monitor.start(lambda _failure: None)
    try:
        monitor.record_progress(0)
        monitor.record_progress(1)
        clock.now = 2.1
        assert monitor.evaluate() is None
        stalled = monitor.snapshot()
        assert stalled["sources"]["0"]["recovery_attempts"] == 1
        assert stalled["sources"]["1"]["recovery_attempts"] == 1

        # Time and packet-side reconnect activity are not progress. The attempt
        # count advances until a decoded/dewarped buffer reaches this monitor.
        clock.now = 3.1
        assert monitor.evaluate() is None
        assert monitor.snapshot()["sources"]["0"]["recovery_attempts"] == 2

        clock.now = 3.2
        monitor.record_progress(0)
        recovered = monitor.snapshot()
        assert recovered["sources"]["0"]["status"] == "running"
        assert recovered["sources"]["0"]["recovery_attempts"] == 0
        assert recovered["sources"]["0"]["recoveries"] == 1
        assert recovered["sources"]["1"]["recovery_attempts"] == 2
    finally:
        assert monitor.stop()["quiesced"] is True


def test_one_stalled_source_escalates_while_other_sources_keep_progressing() -> None:
    clock = _Clock()
    monitor = DecodedSourceProgressMonitor(
        {0: "living-room", 1: "kitchen", 2: "family-room"},
        policy=_policy(),
        clock=clock,
    )
    monitor.start(lambda _failure: None)
    try:
        for source_id in range(3):
            monitor.record_progress(source_id)

        failure = None
        for attempt in range(1, 5):
            clock.now = 1.1 + (attempt * 1.1)
            monitor.record_progress(0)
            monitor.record_progress(1)
            failure = monitor.evaluate()
            assert monitor.snapshot()["sources"]["2"]["recovery_attempts"] == attempt

        assert isinstance(failure, SourceProgressFailure)
        assert failure.source_id == 2
        assert failure.camera_id == "family-room"
        assert failure.attempts == 4
        snapshot = monitor.snapshot()
        assert snapshot["healthy"] is False
        assert snapshot["sources"]["0"]["status"] == "running"
        assert snapshot["sources"]["1"]["status"] == "running"
        assert snapshot["fatal"]["source_id"] == 2
    finally:
        monitor.stop()


def test_monitor_thread_delivers_one_fatal_callback_and_quiesces() -> None:
    callback_event = threading.Event()
    failures: list[SourceProgressFailure] = []
    monitor = DecodedSourceProgressMonitor(
        {0: "living-room"},
        policy=_policy(
            stall_seconds=0.02,
            recovery_interval_seconds=0.01,
            max_recovery_attempts=2,
            startup_grace_seconds=0.0,
            poll_interval_seconds=0.01,
        ),
    )

    def _on_fatal(failure: SourceProgressFailure) -> None:
        failures.append(failure)
        callback_event.set()

    monitor.start(_on_fatal)
    assert callback_event.wait(1.0)
    time.sleep(0.04)
    assert len(failures) == 1
    assert failures[0].attempts == 2
    assert monitor.stop(timeout_seconds=1.0)["quiesced"] is True
