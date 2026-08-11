from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Sequence

import pytest

from noesis.capture_event_controller import (
    CanonicalCameraAliases,
    CaptureEventController,
    CaptureEventControllerError,
    CaptureEventRequest,
)
from noesis.depth_capture_event import DepthStorageFlushEvidence
from noesis_core.capture_event_fusion import (
    CaptureEventFusionError,
    FusedDepthSnapshot,
    RawDepthSnapshot,
    TimestampedRgbFrame,
)
from noesis_core.mapanything_lifecycle import MapAnythingIdleReceipt

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_capture_event_request_default_burst_matches_canonical_runtime() -> None:
    request = CaptureEventRequest(
        camera_alias="living-room",
        request_kind="depth",
    )

    assert request.burst_seconds == 20.0


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _raw(camera: str, index: int) -> RawDepthSnapshot:
    timestamp_us = 1_000_000 + index * 100_000
    return RawDepthSnapshot(
        camera_id=camera,
        storage_key=camera,
        timestamp_us=timestamp_us,
        snapshot_id=f"{camera}-raw-{index}",
        artifact_ref=f"depth-zarr:{camera}/{timestamp_us}.zarr",
        content_sha256=_digest(f"{camera}-content-{index}"),
        sequence=index + 1,
        manifest_sha256=_digest(f"{camera}-manifest-{index}"),
        source_id=0 if camera == "living-room" else 1,
        source_frame_number=40 + index,
        source_media_pts_ns=10_000_000 + index * 1_000_000,
    )


class _MapAnything:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls = 0
        self.fail_call: int | None = None

    def wait_idle(self, *, timeout_s: float = 5.0) -> MapAnythingIdleReceipt:
        self.calls += 1
        self.events.append(f"idle:{self.calls}:{timeout_s:.1f}")
        if self.fail_call == self.calls:
            raise TimeoutError("test barrier timeout")
        return MapAnythingIdleReceipt(
            active_captures=0,
            unfinished_tasks=0,
            worker_started=True,
            worker_alive=True,
            accepting=True,
        )


class _Storage:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.rows: dict[str, tuple[RawDepthSnapshot, ...]] = {
            "living-room": (),
            "kitchen": (),
        }
        self.list_calls = 0
        self.fuse_calls = 0
        self.flush_calls = 0
        self.expect_rgb = False

    def flush_capture_frontier(
        self,
        *,
        timeout_s: float,
    ) -> DepthStorageFlushEvidence:
        self.flush_calls += 1
        self.events.append(f"flush:{self.flush_calls}:{timeout_s:.1f}")
        return DepthStorageFlushEvidence(
            frontier_sequence=self.flush_calls * 10,
            completed=True,
            timed_out=False,
            pending_sequences=(),
            failed_sequences=(),
            poisoned=False,
        )

    def max_raw_timestamp(self, camera_id: str) -> int:
        self.events.append(f"baseline:{camera_id}")
        rows = self.rows[camera_id]
        return rows[-1].timestamp_us if rows else 0

    def list_raw_snapshots(
        self,
        storage_key: str,
        *,
        camera_id: str,
        after_timestamp_us: int,
        limit: int,
    ) -> Sequence[RawDepthSnapshot]:
        assert storage_key == camera_id
        assert not storage_key.isdigit()
        self.list_calls += 1
        self.events.append(f"list:{storage_key}:{after_timestamp_us}:{limit}")
        return tuple(
            row
            for row in self.rows[storage_key]
            if row.timestamp_us > after_timestamp_us
        )[-limit:]

    def fuse_raw_snapshots(
        self,
        storage_key: str,
        snapshots: Sequence[RawDepthSnapshot],
        *,
        rgb_frame: object | None,
        event_id: str,
        min_observations: int,
        depth_agreement_m: float,
    ) -> FusedDepthSnapshot:
        assert (rgb_frame is not None) is self.expect_rgb
        self.fuse_calls += 1
        self.events.append(f"fuse:{storage_key}:{len(snapshots)}")
        return FusedDepthSnapshot(
            camera_id=storage_key,
            storage_key=storage_key,
            timestamp_us=snapshots[-1].timestamp_us + 1,
            snapshot_id=f"{storage_key}-fused",
            artifact_ref=f"depth-zarr:{storage_key}/fused.zarr",
            content_sha256=_digest(f"{storage_key}-fused-content"),
            sequence=max(row.sequence for row in snapshots) + 1,
            manifest_sha256=_digest(f"{storage_key}-fused-manifest"),
            event_id=event_id,
            source_snapshot_ids=tuple(row.snapshot_id for row in snapshots),
        )

    def validate_fused_snapshot(
        self,
        snapshot: FusedDepthSnapshot,
    ) -> FusedDepthSnapshot:
        self.events.append(f"validate:{snapshot.camera_id}")
        return snapshot


class _Harness:
    def __init__(
        self,
        *,
        failure_callback=None,  # type: ignore[no-untyped-def]
        rgb_provider=None,  # type: ignore[no-untyped-def]
        require_rgb: bool = False,
        events: list[str] | None = None,
    ) -> None:
        self.events = events if events is not None else []
        self.gate_open = False
        self.mapanything = _MapAnything(self.events)
        self.storage = _Storage(self.events)
        self.on_wait = lambda: self.storage.rows.__setitem__(
            "living-room",
            (_raw("living-room", 1), _raw("living-room", 2)),
        )

        def set_gate(opened: bool) -> None:
            self.events.append(f"gate:{opened}")
            self.gate_open = opened

        def wait(seconds: float) -> bool:
            self.events.append(f"wait:{seconds}")
            self.on_wait()
            return False

        self.controller = CaptureEventController(
            aliases={0: "living-room", 1: "kitchen"},
            storage=self.storage,
            mapanything=self.mapanything,
            set_depth_gate=set_gate,
            depth_gate_is_open=lambda: self.gate_open,
            burst_waiter=wait,
            failure_callback=failure_callback,
            rgb_provider=rgb_provider,
            require_rgb=require_rgb,
        )


class _ExactRgbProvider:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.active: object | None = None

    def arm(self, camera_id: str) -> object:
        assert self.active is None
        self.active = ("arm", camera_id)
        self.events.append(f"rgb:arm:{camera_id}")
        return self.active

    def provide(
        self,
        camera_id: str,
        *,
        cohort: Sequence[RawDepthSnapshot],
    ) -> TimestampedRgbFrame:
        assert self.active == ("arm", camera_id)
        row = cohort[-1]
        self.events.append(f"rgb:provide:{camera_id}:{row.source_frame_number}")
        return TimestampedRgbFrame(
            camera_id=camera_id,
            source_id=int(row.source_id),
            batch_id=0,
            captured_at_us=1_500_000,
            frame_id=int(row.source_frame_number),
            source_media_pts_ns=int(row.source_media_pts_ns),
            width=1,
            height=1,
            content_sha256=_digest("rgb"),
            pixels=object(),
        )

    def disarm(self, arm: object) -> None:
        assert arm == self.active
        self.events.append("rgb:disarm")
        self.active = None


def _request(**changes: object) -> CaptureEventRequest:
    values: dict[str, object] = {
        "camera_alias": "0",
        "request_kind": "depth",
        "burst_seconds": 0.01,
        "drain_timeout_s": 1.0,
        "raw_limit": 4,
        "min_observations": 2,
    }
    values.update(changes)
    return CaptureEventRequest(**values)  # type: ignore[arg-type]


def _wait_until(predicate, *, timeout_s: float = 2.0) -> bool:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def test_capture_owns_strict_barriers_one_depth_only_fusion_and_exact_evidence() -> (
    None
):
    harness = _Harness()
    outcome = harness.controller.capture(_request())

    assert outcome.canonical_camera == "living-room"
    assert outcome.fused_snapshot.camera_id == "living-room"
    assert harness.storage.fuse_calls == 1
    assert harness.events == [
        "gate:False",
        "idle:1:1.0",
        "flush:1:1.0",
        "baseline:living-room",
        "gate:True",
        "wait:0.01",
        "gate:False",
        "idle:2:1.0",
        "flush:2:1.0",
        "list:living-room:0:5",
        "fuse:living-room:2",
        "validate:living-room",
    ]
    assert outcome.compact_evidence["capture_mode"] == "depth_only"
    assert outcome.compact_evidence["raw_snapshot_count"] == 2
    assert (
        outcome.compact_evidence["fused_snapshot"]["manifest_sha256"]
        == outcome.fused_snapshot.manifest_sha256
    )
    with pytest.raises(TypeError):
        outcome.compact_evidence["camera_id"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        outcome.compact_evidence["fused_snapshot"]["camera_id"] = "changed"  # type: ignore[index]
    json.dumps(outcome.compact_evidence_payload(), allow_nan=False)

    health = harness.controller.health_snapshot()
    assert health["counters"] == {
        "requests_total": 1,
        "admitted_total": 1,
        "completed_total": 1,
        "failed_total": 0,
        "busy_total": 0,
        "cache_only_rejected_total": 0,
        "fusion_total": 1,
        "fatal_barrier_failures_total": 0,
    }
    assert health["shared_gate_owned"] is False
    assert health["healthy"] is True
    assert health["last_fatal_error_code"] is None


def test_exact_rgb_is_armed_after_baseline_and_disarmed_after_fusion() -> None:
    events: list[str] = []
    provider = _ExactRgbProvider(events)
    harness = _Harness(
        rgb_provider=provider,
        require_rgb=True,
        events=events,
    )
    harness.storage.expect_rgb = True

    outcome = harness.controller.capture(_request())

    assert outcome.compact_evidence["capture_mode"] == "depth_rgb_exact"
    assert provider.active is None
    assert events.index("baseline:living-room") < events.index(
        "rgb:arm:living-room"
    )
    assert events.index("rgb:arm:living-room") < events.index("gate:True")
    assert events.index("gate:False") < events.index(
        "rgb:provide:living-room:42"
    )
    assert events.index("validate:living-room") < events.index("rgb:disarm")


def test_cache_only_rejects_before_alias_admission_gate_store_or_fusion() -> None:
    harness = _Harness()
    with pytest.raises(CaptureEventControllerError) as caught:
        harness.controller.capture(
            _request(camera_alias="not-a-camera", cache_only=True)
        )
    assert caught.value.code == "cache_only_forbids_capture_event"
    assert harness.events == []
    assert harness.storage.list_calls == 0
    assert harness.storage.fuse_calls == 0

    health = harness.controller.health_snapshot()
    assert health["healthy"] is True
    assert health["last_fatal_error_code"] is None


def test_aliases_share_camera_lock_and_all_cameras_share_global_gate_ownership() -> (
    None
):
    harness = _Harness()
    entered_wait = threading.Event()
    release_wait = threading.Event()
    failures: list[BaseException] = []

    def blocking_wait(_seconds: float) -> bool:
        harness.events.append("wait:blocking")
        entered_wait.set()
        if not release_wait.wait(timeout=2.0):
            raise TimeoutError("test did not release capture")
        harness.storage.rows["living-room"] = (
            _raw("living-room", 1),
            _raw("living-room", 2),
        )
        return False

    harness.controller._burst_waiter = blocking_wait

    def run_first() -> None:
        try:
            harness.controller.capture(_request(camera_alias="living-room"))
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    thread = threading.Thread(target=run_first, name="capture-owner")
    thread.start()
    assert entered_wait.wait(timeout=1.0)

    with pytest.raises(CaptureEventControllerError) as same_camera:
        harness.controller.capture(_request(camera_alias="0", request_kind="floorplan"))
    assert same_camera.value.code == "capture_event_busy"

    with pytest.raises(CaptureEventControllerError) as other_camera:
        harness.controller.capture(_request(camera_alias="1"))
    assert other_camera.value.code == "capture_event_busy"
    assert other_camera.value.details["shared_gate_owned"] is True

    release_wait.set()
    thread.join(timeout=2.0)
    assert thread.is_alive() is False
    assert failures == []
    assert harness.events.count("gate:True") == 1
    assert harness.storage.fuse_calls == 1


def test_n_plus_one_query_rejects_truncated_cohort_before_fusion() -> None:
    harness = _Harness()
    harness.on_wait = lambda: harness.storage.rows.__setitem__(
        "living-room",
        tuple(_raw("living-room", index) for index in range(1, 5)),
    )
    with pytest.raises(CaptureEventFusionError) as caught:
        harness.controller.capture(_request(raw_limit=3))
    assert caught.value.code == "raw_snapshot_limit_exceeded"
    assert "list:living-room:0:4" in harness.events
    assert harness.storage.fuse_calls == 0


def test_postburst_idle_failure_still_flushes_and_never_fuses() -> None:
    harness = _Harness()
    harness.mapanything.fail_call = 2
    with pytest.raises(CaptureEventControllerError) as caught:
        harness.controller.capture(_request())
    assert caught.value.code == "mapanything_idle_failed"
    assert caught.value.details["stage"] == "postburst_mapanything_idle"
    assert harness.gate_open is False
    assert "flush:2:1.0" in harness.events
    assert harness.storage.fuse_calls == 0
    health = harness.controller.health_snapshot()
    assert health["healthy"] is False
    assert health["last_fatal_error_code"] == "mapanything_idle_failed"


def test_fatal_exact_capture_closes_new_admission_before_releasing_gate() -> None:
    harness = _Harness()
    harness.mapanything.fail_call = 2
    inactive_entered = threading.Event()
    release_inactive = threading.Event()
    errors: list[BaseException] = []
    mark_active = harness.controller._mark_active

    def block_final_active(camera: str, request_kind: str, active: bool) -> None:
        if not active:
            inactive_entered.set()
            assert release_inactive.wait(timeout=2.0)
        mark_active(camera, request_kind, active)

    harness.controller._mark_active = block_final_active

    def capture() -> None:
        try:
            harness.controller.capture(_request())
        except BaseException as exc:
            errors.append(exc)

    owner = threading.Thread(target=capture, name="fatal-exact-capture")
    owner.start()
    assert inactive_entered.wait(timeout=1.0)

    with pytest.raises(CaptureEventControllerError) as cancelled:
        harness.controller.start_refresh(1)
    assert cancelled.value.code == "capture_event_cancelled"
    assert harness.controller.health_snapshot()["manual_refresh"][
        "admission_open"
    ] is False

    release_inactive.set()
    owner.join(timeout=2.0)
    assert owner.is_alive() is False
    assert len(errors) == 1
    assert isinstance(errors[0], CaptureEventControllerError)
    assert errors[0].code == "mapanything_idle_failed"  # type: ignore[attr-defined]


def test_total_barrier_deadline_rejects_late_idle_even_after_flush() -> None:
    harness = _Harness()
    wait_idle = harness.mapanything.wait_idle

    def late_idle(*, timeout_s: float) -> MapAnythingIdleReceipt:
        time.sleep(0.11)
        return wait_idle(timeout_s=timeout_s)

    harness.mapanything.wait_idle = late_idle
    with pytest.raises(CaptureEventControllerError) as caught:
        harness.controller.capture(_request(drain_timeout_s=0.1))

    assert caught.value.code == "depth_storage_flush_failed"
    assert caught.value.details["stage"] == "baseline_deadline_exhausted"
    assert harness.storage.flush_calls == 1
    assert harness.storage.fuse_calls == 0
    assert harness.controller.health_snapshot()["manual_refresh"][
        "admission_open"
    ] is False


def test_manual_refresh_rejects_runtime_stop_before_gate_open() -> None:
    harness = _Harness()
    harness.controller._stop_requested = lambda: True

    with pytest.raises(CaptureEventControllerError) as caught:
        harness.controller.start_refresh(1)

    assert caught.value.code == "capture_event_cancelled"
    assert harness.events == []
    assert harness.controller.health_snapshot()["shared_gate_owned"] is False


def test_compact_evidence_hash_is_stable_for_identical_exact_receipts() -> None:
    first = _Harness().controller.capture(_request())
    second = _Harness().controller.capture(_request())
    assert first.fusion_evidence_sha256 == second.fusion_evidence_sha256
    assert first.compact_evidence_sha256 == second.compact_evidence_sha256


def test_canonical_alias_contract_is_strict_and_controller_has_no_executor() -> None:
    aliases = CanonicalCameraAliases(
        {0: "living-room", 1: "kitchen"},
        extra_aliases={"family": "living-room"},
    )
    assert aliases.canonicalize("0") == "living-room"
    assert aliases.canonicalize("living-room") == "living-room"
    assert aliases.canonicalize("family") == "living-room"
    with pytest.raises(CaptureEventControllerError) as caught:
        aliases.canonicalize("unknown")
    assert caught.value.code == "unknown_camera_alias"
    with pytest.raises(ValueError, match="ambiguous"):
        CanonicalCameraAliases(
            {0: "living-room", 1: "kitchen"},
            extra_aliases={"0": "kitchen"},
        )

    source = (REPO_ROOT / "noesis" / "capture_event_controller.py").read_text(
        encoding="utf-8"
    )
    assert "ThreadPoolExecutor" not in source
    assert 'name="capture-event-manual-refresh"' in source
    assert "daemon=False" in source


@pytest.mark.parametrize(
    "seconds",
    [0, 301, -1, True, 1.5, float("nan"), float("inf"), "not-a-number"],
)
def test_manual_refresh_rejects_non_integer_non_finite_or_out_of_range_seconds(
    seconds: object,
) -> None:
    harness = _Harness()
    with pytest.raises(ValueError, match=r"seconds must be an integer in \[1, 300\]"):
        harness.controller.start_refresh(seconds)
    assert harness.events == []
    assert harness.controller.health_snapshot()["manual_refresh"]["counters"] == {
        "requests_total": 0,
        "admitted_total": 0,
        "extended_total": 0,
        "completed_total": 0,
        "failed_total": 0,
        "busy_total": 0,
        "shutdown_total": 0,
    }


def test_manual_refresh_returns_immediately_and_owner_closes_with_both_barriers() -> (
    None
):
    harness = _Harness()
    before = time.monotonic()
    payload = harness.controller.start_refresh(2)
    elapsed = time.monotonic() - before

    assert elapsed < 0.25
    assert payload == {
        "started_at": payload["started_at"],
        "will_disable_at": payload["will_disable_at"],
        "enabled": True,
        "seconds": 2,
    }
    assert payload["will_disable_at"] == payload["started_at"] + 2
    health = harness.controller.health_snapshot()
    assert health["shared_gate_owned"] is True
    assert health["manual_refresh"]["active"] is True
    assert health["manual_refresh"]["phase"] == "open"
    assert health["manual_refresh"]["thread_alive"] is True
    owner = harness.controller._manual_refresh_thread
    assert owner is not None
    assert owner.daemon is False

    harness.controller.shutdown(timeout_s=2.0)

    assert harness.events == [
        "gate:True",
        "gate:False",
        "idle:1:35.0",
        "flush:1:35.0",
        "gate:False",
        "idle:2:2.0",
        "flush:2:2.0",
    ]
    assert harness.gate_open is False
    assert owner.is_alive() is False
    health = harness.controller.health_snapshot()
    assert health["shared_gate_owned"] is False
    assert health["manual_refresh"]["active"] is False
    assert health["manual_refresh"]["phase"] == "shutdown"
    assert health["manual_refresh"]["counters"] == {
        "requests_total": 1,
        "admitted_total": 1,
        "extended_total": 0,
        "completed_total": 1,
        "failed_total": 0,
        "busy_total": 0,
        "shutdown_total": 1,
    }


def test_repeated_manual_refresh_extends_one_monotonic_owner() -> None:
    harness = _Harness()
    first = harness.controller.start_refresh(1)
    owner = harness.controller._manual_refresh_thread
    assert owner is not None

    second = harness.controller.start_refresh(2)

    assert harness.controller._manual_refresh_thread is owner
    assert second["will_disable_at"] >= first["will_disable_at"]
    assert second["will_disable_at"] == second["started_at"] + 2
    assert second["seconds"] == 2
    health = harness.controller.health_snapshot()["manual_refresh"]
    assert health["counters"]["requests_total"] == 2
    assert health["counters"]["admitted_total"] == 1
    assert health["counters"]["extended_total"] == 1
    assert harness.events.count("gate:True") == 1

    harness.controller.shutdown(timeout_s=2.0)
    assert owner.is_alive() is False
    # The owner proves the closing window, and shutdown independently proves
    # the now-closed controller frontier before runtime resources can be freed.
    assert harness.events.count("gate:False") == 2
    assert harness.mapanything.calls == 2
    assert harness.storage.flush_calls == 2


def test_manual_and_exact_capture_reject_cross_mode_contention() -> None:
    manual = _Harness()
    manual.controller.start_refresh(2)
    with pytest.raises(CaptureEventControllerError) as exact_busy:
        manual.controller.capture(_request())
    assert exact_busy.value.code == "capture_event_busy"
    assert exact_busy.value.details["shared_gate_owned"] is True
    manual.controller.shutdown(timeout_s=2.0)

    exact = _Harness()
    entered_wait = threading.Event()
    release_wait = threading.Event()
    failures: list[BaseException] = []

    def blocking_wait(_seconds: float) -> bool:
        entered_wait.set()
        if not release_wait.wait(timeout=2.0):
            raise TimeoutError("test did not release exact capture")
        exact.storage.rows["living-room"] = (
            _raw("living-room", 1),
            _raw("living-room", 2),
        )
        return False

    exact.controller._burst_waiter = blocking_wait

    def run_exact() -> None:
        try:
            exact.controller.capture(_request())
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    exact_owner = threading.Thread(target=run_exact, name="exact-capture-owner")
    exact_owner.start()
    assert entered_wait.wait(timeout=1.0)
    with pytest.raises(CaptureEventControllerError) as manual_busy:
        exact.controller.start_refresh(1)
    assert manual_busy.value.code == "capture_event_busy"
    assert manual_busy.value.details["shared_gate_owned"] is True

    release_wait.set()
    exact_owner.join(timeout=2.0)
    assert exact_owner.is_alive() is False
    assert failures == []


def test_manual_refresh_expiry_closes_gate_then_waits_idle_and_flushes() -> None:
    harness = _Harness()
    harness.controller.start_refresh(1)

    assert _wait_until(
        lambda: not harness.controller.health_snapshot()["manual_refresh"]["active"],
        timeout_s=2.0,
    )
    assert harness.events == [
        "gate:True",
        "gate:False",
        "idle:1:35.0",
        "flush:1:35.0",
    ]
    assert harness.gate_open is False
    health = harness.controller.health_snapshot()
    assert health["shared_gate_owned"] is False
    assert health["manual_refresh"]["phase"] == "idle"
    assert health["manual_refresh"]["counters"]["completed_total"] == 1
    harness.controller.shutdown(timeout_s=1.0)


def test_fatal_async_manual_refresh_failure_notifies_runtime_once_after_cleanup() -> (
    None
):
    callback_errors: list[BaseException] = []
    callback_states: list[tuple[bool, bool]] = []
    harness: _Harness

    def on_failure(error: BaseException) -> None:
        callback_errors.append(error)
        callback_states.append(
            (
                harness.gate_open,
                harness.controller.health_snapshot()["shared_gate_owned"],
            )
        )

    harness = _Harness(failure_callback=on_failure)
    harness.mapanything.fail_call = 1
    harness.controller.start_refresh(1)

    assert _wait_until(lambda: len(callback_errors) == 1, timeout_s=2.0)
    assert isinstance(callback_errors[0], CaptureEventControllerError)
    assert callback_errors[0].code == "mapanything_idle_failed"  # type: ignore[attr-defined]
    assert callback_states == [(False, False)]
    assert harness.gate_open is False
    health = harness.controller.health_snapshot()
    assert health["manual_refresh"]["active"] is False
    assert health["manual_refresh"]["last_error_code"] == "mapanything_idle_failed"
    assert health["manual_refresh"]["counters"]["failed_total"] == 1
    assert health["counters"]["fatal_barrier_failures_total"] == 1

    harness.controller.shutdown(timeout_s=1.0)
    harness.controller.shutdown(timeout_s=1.0)
    assert len(callback_errors) == 1
    assert health["manual_refresh"]["counters"]["admitted_total"] == 1


def test_fatal_synchronous_manual_start_failure_notifies_after_cleanup() -> None:
    callback_errors: list[BaseException] = []
    callback_states: list[tuple[bool, bool]] = []
    harness: _Harness

    def on_failure(error: BaseException) -> None:
        callback_errors.append(error)
        callback_states.append(
            (
                harness.gate_open,
                harness.controller.health_snapshot()["shared_gate_owned"],
            )
        )

    harness = _Harness(failure_callback=on_failure)

    def fail_open(opened: bool) -> None:
        harness.events.append(f"gate-attempt:{opened}")
        if opened:
            raise RuntimeError("gate fixture failed")
        harness.gate_open = False

    harness.controller._set_depth_gate = fail_open
    with pytest.raises(CaptureEventControllerError) as caught:
        harness.controller.start_refresh(1)

    assert caught.value.code == "depth_gate_transition_failed"
    assert callback_errors == [caught.value]
    assert callback_states == [(False, False)]
    assert harness.events == [
        "gate-attempt:True",
        "gate-attempt:False",
        "idle:1:35.0",
        "flush:1:35.0",
    ]
    health = harness.controller.health_snapshot()
    assert health["manual_refresh"]["phase"] == "shutdown"
    assert health["manual_refresh"]["admission_open"] is False
    assert health["manual_refresh"]["counters"]["shutdown_total"] == 1
    assert health["last_fatal_error_code"] == "depth_gate_transition_failed"
    harness.controller.shutdown(timeout_s=1.0)


def test_shutdown_is_idempotent_closes_admission_and_leaves_no_owner_thread() -> (
    None
):
    harness = _Harness()
    harness.controller.start_refresh(2)
    owner = harness.controller._manual_refresh_thread
    assert owner is not None

    harness.controller.shutdown(timeout_s=2.0)
    harness.controller.shutdown(timeout_s=2.0)

    assert owner.is_alive() is False
    assert harness.gate_open is False
    health = harness.controller.health_snapshot()
    assert health["shared_gate_owned"] is False
    assert health["manual_refresh"]["thread_alive"] is False
    assert health["manual_refresh"]["admission_open"] is False
    assert health["manual_refresh"]["counters"]["shutdown_total"] == 1
    with pytest.raises(CaptureEventControllerError) as refresh_cancelled:
        harness.controller.start_refresh(1)
    assert refresh_cancelled.value.code == "capture_event_cancelled"
    with pytest.raises(CaptureEventControllerError) as capture_cancelled:
        harness.controller.capture(_request())
    assert capture_cancelled.value.code == "capture_event_cancelled"
