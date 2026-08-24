from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest


def _receipt(hooks, source: int, frame: int, sequence: int):
    return hooks.TrackingPublicationReceipt(
        source_id=int(source),
        frame_id=int(frame),
        observed_at_us=1_000_000 + int(frame),
        tracking_publication_sequence=int(sequence),
        outbound_submission_id=(int(sequence) * 2) + 1,
        outbound_message_count=1,
    )


def _bev_receipt(hooks, kwargs):
    return hooks.BevPublicationReceipt(
        status="admitted",
        camera_id=str(kwargs["camera_id"]),
        source_id=int(kwargs["source_id"]),
        frame_id=int(kwargs["frame_id"]),
        observed_at_us=int(kwargs["observed_at_us"]),
        tracking_publication_sequence=int(kwargs["tracking_publication_sequence"]),
        tracking_outbound_submission_id=int(
            kwargs["tracking_outbound_submission_id"]
        ),
        outbound_submission_id=int(kwargs["tracking_outbound_submission_id"]) + 1,
    )


def _work(hooks, source: int, frame: int):
    return hooks._TrackingPublicationWork(
        source_id=int(source),
        camera_id=f"camera-{source}",
        frame_id=int(frame),
        observed_at_us=1_000_000 + int(frame),
        now_ts=100.0 + (float(frame) / 30.0),
        timestamp_us=2_000_000 + int(frame),
        tracks=[
            {
                "camera_id": f"camera-{source}",
                "tracker_id": 7,
                "class_id": 0,
            }
        ],
        footpoints=[],
        temporal_contract={"observed_at_us": 1_000_000 + int(frame)},
    )


def _shadow_work(
    hooks,
    source: int,
    frame: int,
    *,
    observed_at: float,
    sequence: int,
):
    return hooks._ShadowIdentityWork(
        source_id=int(source),
        camera_id=f"camera-{source}",
        frame_id=int(frame),
        observed_at=float(observed_at),
        primitives=(),
        admission_sequence=int(sequence),
    )


def test_media_side_lifecycle_rows_feed_exact_osd_ring_and_worker_packet() -> None:
    from noesis.pipelines import hooks
    from noesis_core.runtime_publication import RuntimePublicationGate

    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(config={}, ds_pipeline=object()),
        tracking_pub=SimpleNamespace(),
        camera_labels={0: "camera-0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
    )
    first = {
        "camera_id": "camera-0",
        "tracker_id": 7,
        "frame_id": 10,
        "class_id": 0,
    }
    first_continuity = processor._prepare_tracking_cohort(
        sensor_id=0,
        camera_id="camera-0",
        frame_id=10,
        observed_at_us=1_000_010,
        tracks=[first],
        footpoints=[],
    )
    processor._set_active_tracks_snapshot(0, 10, [first])
    second = {
        "camera_id": "camera-0",
        "tracker_id": 7,
        "frame_id": 11,
        "class_id": 0,
    }
    second_continuity = processor._prepare_tracking_cohort(
        sensor_id=0,
        camera_id="camera-0",
        frame_id=11,
        observed_at_us=1_000_011,
        tracks=[second],
        footpoints=[],
    )
    processor._set_active_tracks_snapshot(0, 11, [second])

    assert first_continuity.active_lifecycles[0].generation == 1
    assert second_continuity.active_lifecycles[0].generation == 1
    assert processor.get_active_track_map(0, 10)[7]["tracker_lifecycle_generation"] == 1
    assert processor.get_active_track_map(0, 11)[7]["tracker_lifecycle_generation"] == 1
    assert processor.get_active_track_map(0, 9) == {}


def test_worker_preserves_per_source_fifo_and_fairly_rotates_sources() -> None:
    from noesis.pipelines import hooks
    from noesis_core.runtime_publication import RuntimePublicationGate

    hooks.reset_core_path_instrumentation()

    first_started = threading.Event()
    release_first = threading.Event()
    events: list[tuple[str, int, int]] = []

    class Publisher:
        sequence = 0

        def publish(self, source_id, _tracks, *, frame_metadata=None):
            frame = int(frame_metadata["frame_id"])
            events.append(("tracking", int(source_id), frame))
            if frame == 1:
                first_started.set()
                assert release_first.wait(timeout=2.0)
            receipt = _receipt(hooks, int(source_id), frame, self.sequence)
            self.sequence += 1
            return receipt

    class Renderer:
        def render_and_publish(self, **kwargs):
            events.append(("bev", int(kwargs["source_id"]), int(kwargs["frame_id"])))
            return _bev_receipt(hooks, kwargs)

    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={},
            stable_id_mgr=None,
            ds_pipeline=object(),
        ),
        tracking_pub=Publisher(),
        camera_labels={0: "camera-0", 1: "camera-1"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
        bev_renderer=Renderer(),
        bev_calibration=SimpleNamespace(
        world_snapshot=lambda _source, camera: SimpleNamespace(
            image_size=(1280, 720),
            camera_id=camera,
        )
        ),
    )

    # Use the real worker with a minimal processor surface.  The first source
    # blocks in publication; source 1 is still admitted and source 0 remains
    # FIFO rather than being replaced/dropped.
    worker = hooks._TrackingPublicationWorker(
        processor,
        max_pending_per_source=4,
        max_pending_total=8,
    )
    worker.enqueue(_work(hooks, 0, 1))
    assert first_started.wait(timeout=2.0)
    worker.enqueue(_work(hooks, 0, 2))
    worker.enqueue(_work(hooks, 0, 3))
    worker.enqueue(_work(hooks, 1, 1))
    release_first.set()
    deadline = time.monotonic() + 3.0
    while len(events) < 8 and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.shutdown(wait=True, timeout_s=2.0)

    assert [(kind, source, frame) for kind, source, frame in events] == [
        ("tracking", 0, 1),
        ("bev", 0, 1),
        ("tracking", 0, 2),
        ("bev", 0, 2),
        ("tracking", 1, 1),
        ("bev", 1, 1),
        ("tracking", 0, 3),
        ("bev", 0, 3),
    ]
    snapshot = hooks.get_core_path_instrumentation_snapshot()
    counters = snapshot["counters"]
    assert counters["tracking.publication_worker.enqueued_total"] == 4
    assert counters["tracking.publication_worker.completed_total"] == 4
    assert counters["tracking.publication_worker.pending_total"] == 0
    assert counters["tracking.publication_worker.pending_high_watermark"] >= 3
    timing = snapshot["stage_timings"]["tracking.publication_worker_item"]
    assert timing["count"] == 4
    assert timing["max_ns"] > 0
    queue_wait = snapshot["stage_timings"][
        "tracking.publication_worker_queue_wait"
    ]
    assert queue_wait["count"] == 4
    assert queue_wait["max_ns"] > 0


def test_worker_overflow_is_terminal_and_does_not_drop_silently() -> None:
    from noesis.pipelines import hooks

    failures: list[BaseException] = []
    processor = SimpleNamespace()
    worker = hooks._TrackingPublicationWorker(
        processor,
        failure_callback=failures.append,
        max_pending_per_source=1,
        max_pending_total=1,
    )
    # Keep the worker from consuming the first item before the overflow check.
    worker._stopping = True
    assert worker.enqueue(_work(hooks, 0, 1)) is False
    worker.shutdown(wait=True, timeout_s=2.0)
    assert failures == []

    # A fresh worker demonstrates the actual bounded overflow transition.
    worker = hooks._TrackingPublicationWorker(
        processor,
        failure_callback=failures.append,
        max_pending_per_source=1,
        max_pending_total=1,
    )
    worker._stopping = True
    worker._stopping = False
    # Force a deterministic full queue without racing the daemon.
    with worker._condition:
        worker._pending[0] = __import__("collections").deque([_work(hooks, 0, 1)])
        worker._pending_total = 1
    assert worker.enqueue(_work(hooks, 0, 2)) is False
    assert isinstance(worker.terminal_failure, RuntimeError)
    assert len(failures) == 1
    worker.shutdown(wait=True, timeout_s=2.0)


def test_worker_publisher_failure_is_terminal_and_surfaces_once() -> None:
    from noesis.pipelines import hooks

    failures: list[BaseException] = []

    class Processor:
        def _publish_tracking_work(self, _work):
            raise RuntimeError("publisher unavailable")

    worker = hooks._TrackingPublicationWorker(
        Processor(),
        failure_callback=failures.append,
    )
    assert worker.enqueue(_work(hooks, 0, 1)) is True
    deadline = time.monotonic() + 2.0
    while worker.terminal_failure is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert isinstance(worker.terminal_failure, RuntimeError)
    assert len(failures) == 1
    assert worker.enqueue(_work(hooks, 0, 2)) is False
    worker.shutdown(wait=True, timeout_s=2.0)


def test_shadow_worker_coalesces_only_stale_optional_source_work_and_times_it() -> None:
    from noesis.pipelines import hooks

    first_started = threading.Event()
    release_first = threading.Event()
    processed: list[tuple[int, int, bool]] = []

    class Processor:
        def _process_identity_v2_source_frame(
            self,
            *,
            camera_id,
            frame_id,
            primitives,
            observed_at,
            fatal,
        ):
            source = int(str(camera_id).rsplit("-", 1)[1])
            processed.append((source, int(frame_id), bool(fatal)))
            assert primitives == ()
            assert observed_at > 0.0
            if source == 0 and int(frame_id) == 1:
                first_started.set()
                assert release_first.wait(timeout=2.0)

    hooks.reset_core_path_instrumentation()
    worker = hooks._ShadowIdentityWorker(
        Processor(),
        max_pending_sources=2,
    )
    assert worker.enqueue(
        _shadow_work(hooks, 0, 1, observed_at=100.0, sequence=0)
    )
    assert first_started.wait(timeout=2.0)
    assert worker.enqueue(
        _shadow_work(hooks, 0, 2, observed_at=102.0, sequence=1)
    )
    # The newest pending frame replaces only this source's stale optional
    # frame. The in-flight frame and the other source remain untouched.
    assert worker.enqueue(
        _shadow_work(hooks, 0, 3, observed_at=103.0, sequence=2)
    )
    assert worker.enqueue(
        _shadow_work(hooks, 1, 7, observed_at=101.0, sequence=3)
    )
    release_first.set()
    deadline = time.monotonic() + 2.0
    while len(processed) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.shutdown(wait=True, timeout_s=2.0)

    assert processed == [(0, 1, False), (1, 7, False), (0, 3, False)]
    snapshot = hooks.get_core_path_instrumentation_snapshot()
    counters = snapshot["counters"]
    assert counters["identity_v2.shadow.enqueued_total"] == 4
    assert counters["identity_v2.shadow.coalesced_total"] == 1
    assert counters["identity_v2.shadow.completed_total"] == 3
    assert counters["identity_v2.shadow.pending_sources"] == 0
    timing = snapshot["stage_timings"][
        "identity_v2.shadow_process_source_frame"
    ]
    assert timing["count"] == 3
    assert timing["max_ns"] > 0
    queue_wait = snapshot["stage_timings"]["identity_v2.shadow_queue_wait"]
    assert queue_wait["count"] == 3
    assert queue_wait["max_ns"] > 0


def test_shadow_worker_capacity_drop_never_touches_canonical_lane() -> None:
    from noesis.pipelines import hooks

    first_started = threading.Event()
    release_first = threading.Event()
    processed: list[tuple[int, int]] = []

    class Processor:
        def _process_identity_v2_source_frame(
            self,
            *,
            camera_id,
            frame_id,
            fatal,
            **_kwargs,
        ):
            source = int(str(camera_id).rsplit("-", 1)[1])
            processed.append((source, int(frame_id)))
            assert fatal is False
            if source == 0:
                first_started.set()
                assert release_first.wait(timeout=2.0)

    hooks.reset_core_path_instrumentation()
    worker = hooks._ShadowIdentityWorker(
        Processor(),
        max_pending_sources=1,
    )
    assert worker.enqueue(
        _shadow_work(hooks, 0, 1, observed_at=100.0, sequence=0)
    )
    assert first_started.wait(timeout=2.0)
    assert worker.enqueue(
        _shadow_work(hooks, 1, 1, observed_at=101.0, sequence=1)
    )
    assert not worker.enqueue(
        _shadow_work(hooks, 2, 1, observed_at=102.0, sequence=2)
    )
    release_first.set()
    deadline = time.monotonic() + 2.0
    while len(processed) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.shutdown(wait=True, timeout_s=2.0)

    assert processed == [(0, 1), (1, 1)]
    counters = hooks.get_core_path_instrumentation_snapshot()["counters"]
    assert counters["identity_v2.shadow.dropped_capacity_total"] == 1


def test_shadow_worker_yields_while_canonical_backlog_is_pending() -> None:
    from noesis.pipelines import hooks

    processed: list[int] = []
    processed_event = threading.Event()

    class Processor:
        canonical_backlogged = True

        def _canonical_publication_backlogged(self):
            return self.canonical_backlogged

        def _process_identity_v2_source_frame(self, *, frame_id, **_kwargs):
            processed.append(int(frame_id))
            processed_event.set()

    processor = Processor()
    hooks.reset_core_path_instrumentation()
    worker = hooks._ShadowIdentityWorker(processor, max_pending_sources=1)
    assert worker.enqueue(
        _shadow_work(hooks, 0, 1, observed_at=100.0, sequence=0)
    )
    time.sleep(0.05)
    assert processed == []
    processor.canonical_backlogged = False
    assert processed_event.wait(timeout=2.0)
    worker.shutdown(wait=True, timeout_s=2.0)

    counters = hooks.get_core_path_instrumentation_snapshot()["counters"]
    assert counters["identity_v2.shadow.yielded_canonical_backlog_total"] == 1
    assert counters["identity_v2.shadow.completed_total"] == 1


def test_shadow_worker_yields_while_canonical_item_is_inflight() -> None:
    from noesis.pipelines import hooks

    canonical_started = threading.Event()
    release_canonical = threading.Event()
    shadow_processed = threading.Event()
    processed: list[int] = []

    class Processor:
        canonical_worker = None

        def _canonical_publication_backlogged(self):
            return bool(self.canonical_worker and self.canonical_worker.busy)

        def _publish_tracking_work(self, _work):
            canonical_started.set()
            assert release_canonical.wait(timeout=2.0)

        def _process_identity_v2_source_frame(self, *, frame_id, **_kwargs):
            processed.append(int(frame_id))
            shadow_processed.set()

    processor = Processor()
    hooks.reset_core_path_instrumentation()
    canonical_worker = hooks._TrackingPublicationWorker(processor)
    processor.canonical_worker = canonical_worker
    assert canonical_worker.enqueue(hooks._TrackingPublicationWork(
        source_id=0,
        camera_id="camera-0",
        frame_id=1,
        observed_at_us=1_000_001,
        now_ts=100.0,
        timestamp_us=1_000_001,
        tracks=[],
        footpoints=[],
        temporal_contract={},
    ))
    assert canonical_started.wait(timeout=2.0)

    shadow_worker = hooks._ShadowIdentityWorker(processor, max_pending_sources=1)
    assert shadow_worker.enqueue(
        _shadow_work(hooks, 0, 2, observed_at=100.1, sequence=0)
    )
    time.sleep(0.05)
    assert not shadow_processed.is_set()

    release_canonical.set()
    assert shadow_processed.wait(timeout=2.0)
    shadow_worker.shutdown(wait=True, timeout_s=2.0)
    canonical_worker.shutdown(wait=True, timeout_s=2.0)

    assert processed == [2]
    counters = hooks.get_core_path_instrumentation_snapshot()["counters"]
    assert counters["identity_v2.shadow.yielded_canonical_backlog_total"] == 1


def test_shadow_admission_is_rate_limited_per_source() -> None:
    from noesis.pipelines import hooks
    from noesis_core.runtime_publication import RuntimePublicationGate

    published: list[int] = []

    class ShadowService:
        authoritative = False
        embedding_dim = 2

        def process_source_frame(self, **_kwargs):
            return None

    class Publisher:
        def publish(self, source_id, _tracks, *, frame_metadata=None):
            frame_id = int((frame_metadata or {})["frame_id"])
            published.append(frame_id)
            return _receipt(hooks, int(source_id), frame_id, len(published) - 1)

    hooks.reset_core_path_instrumentation()
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={},
            ds_pipeline=object(),
            identity_v2_service=ShadowService(),
        ),
        tracking_pub=Publisher(),
        camera_labels={0: "camera-0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
    )
    processor._shadow_identity_publish_interval_s = 1.0

    for frame_id, now_ts in ((1, 100.0), (2, 100.1)):
        track = {
            "camera_id": "camera-0",
            "tracker_id": 7,
            "frame_id": frame_id,
            "class_id": 0,
        }
        continuity = processor._prepare_tracking_cohort(
            sensor_id=0,
            camera_id="camera-0",
            frame_id=frame_id,
            observed_at_us=1_000_000 + frame_id,
            tracks=[track],
            footpoints=[],
        )
        assert processor._enqueue_tracking_publication(
            sensor_id=0,
            camera_id="camera-0",
            frame_meta=SimpleNamespace(
                frame_number=frame_id,
                buf_pts=(1_000_000 + frame_id) * 1_000,
            ),
            tracks=[track],
            footpoints=[],
            now_ts=now_ts,
            temporal_contract={"observed_at_us": 1_000_000 + frame_id},
            continuity=continuity,
            force=True,
            identity_v2_primitives=[],
        )
        if frame_id == 1:
            deadline = time.monotonic() + 2.0
            while (
                hooks.get_core_path_instrumentation_snapshot()["counters"].get(
                    "identity_v2.shadow.completed_total",
                    0,
                )
                < 1
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)

    processor.shutdown(wait=True, timeout_s=2.0)
    counters = hooks.get_core_path_instrumentation_snapshot()["counters"]
    assert published == [1, 2]
    assert counters["identity_v2.shadow.enqueued_total"] == 1
    assert counters["identity_v2.shadow.rate_limited_total"] == 1


def test_worker_overflow_rejects_only_new_work_and_drains_admitted_fifo() -> None:
    from noesis.pipelines import hooks

    first_started = threading.Event()
    release_first = threading.Event()
    published: list[int] = []
    failures: list[BaseException] = []

    class Processor:
        def _publish_tracking_work(self, work):
            published.append(int(work.frame_id))
            if int(work.frame_id) == 1:
                first_started.set()
                assert release_first.wait(timeout=2.0)

    worker = hooks._TrackingPublicationWorker(
        Processor(),
        failure_callback=failures.append,
        max_pending_per_source=1,
        max_pending_total=1,
    )
    assert worker.enqueue(_work(hooks, 0, 1)) is True
    assert first_started.wait(timeout=2.0)
    assert worker.enqueue(_work(hooks, 0, 2)) is True
    assert worker.enqueue(_work(hooks, 0, 3)) is False
    release_first.set()

    deadline = time.monotonic() + 2.0
    while worker.terminal_failure is None and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.shutdown(wait=True, timeout_s=2.0)

    assert published == [1, 2]
    assert len(failures) == 1


def test_worker_prepares_tombstone_after_queued_presence_publishes() -> None:
    """A queued absence must compare against the worker's published state."""

    from noesis.pipelines import hooks
    from noesis_core.runtime_publication import RuntimePublicationGate

    events: list[tuple[int, list[dict[str, object]]]] = []
    bev_events: list[tuple[int, list[dict[str, object]]]] = []

    class Publisher:
        sequence = 0

        def publish(self, source_id, tracks, *, frame_metadata=None):
            metadata = dict(frame_metadata or {})
            events.append(
                (
                    int(metadata["frame_id"]),
                    list(metadata["tracker_lifecycle_tombstones"]),
                )
            )
            receipt = _receipt(
                hooks,
                int(source_id),
                int(metadata["frame_id"]),
                self.sequence,
            )
            self.sequence += 1
            return receipt

    class Renderer:
        def render_and_publish(self, **kwargs):
            bev_events.append(
                (
                    int(kwargs["frame_id"]),
                    list(kwargs["tracker_lifecycle_tombstones"]),
                )
            )
            return _bev_receipt(hooks, kwargs)

    publisher = Publisher()
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(config={}, ds_pipeline=object()),
        tracking_pub=publisher,
        camera_labels={0: "camera-0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
        bev_renderer=Renderer(),
        bev_calibration=SimpleNamespace(
            world_snapshot=lambda _source, camera: SimpleNamespace(
                image_size=(1280, 720),
                camera_id=camera,
            )
        ),
    )

    present = {
        "camera_id": "camera-0",
        "tracker_id": 7,
        "frame_id": 10,
        "class_id": 0,
    }
    present_continuity = processor._prepare_tracking_cohort(
        sensor_id=0,
        camera_id="camera-0",
        frame_id=10,
        observed_at_us=1_000_010,
        tracks=[present],
        footpoints=[],
    )
    processor._set_active_tracks_snapshot(0, 10, [present])

    absent_continuity = processor._prepare_tracking_cohort(
        sensor_id=0,
        camera_id="camera-0",
        frame_id=11,
        observed_at_us=1_000_011,
        tracks=[],
        footpoints=[],
    )
    processor._set_active_tracks_snapshot(0, 11, [])

    # Both media callbacks completed before the worker publishes either row;
    # the media-time preview cannot see the not-yet-published presence.
    assert present_continuity.tombstones == ()
    assert absent_continuity.tombstones == ()

    def _work_for(
        frame_id: int,
        tracks: list[dict[str, object]],
        continuity,
    ):
        return hooks._TrackingPublicationWork(
            source_id=0,
            camera_id="camera-0",
            frame_id=int(frame_id),
            observed_at_us=1_000_000 + int(frame_id),
            now_ts=100.0 + (float(frame_id) / 30.0),
            timestamp_us=2_000_000 + int(frame_id),
            tracks=tracks,
            footpoints=[],
            temporal_contract={"observed_at_us": 1_000_000 + int(frame_id)},
            continuity=continuity,
            source_epoch=int(continuity.source_epoch),
        )

    processor._publish_tracking_work(
        _work_for(10, [dict(present)], present_continuity)
    )
    processor._publish_tracking_work(_work_for(11, [], absent_continuity))

    assert events[0] == (10, [])
    assert events[1][0] == 11
    assert events[1][1] == [
        {
            "camera_id": "camera-0",
            "tracker_id": 7,
            "tracker_lifecycle_generation": 1,
            "last_seen_frame_id": 10,
            "last_seen_observed_at_us": 1_000_010,
            "disappeared_at_frame_id": 11,
            "disappeared_at_observed_at_us": 1_000_011,
        }
    ]
    assert bev_events == events


def test_shadow_identity_isolated_from_exact_outbound_row() -> None:
    import numpy as np

    from noesis.identity_v2_service import IdentityFramePrimitive
    from noesis.pipelines import hooks
    from noesis_core.runtime_publication import RuntimePublicationGate

    scoring_started = threading.Event()
    release_scoring = threading.Event()
    published = threading.Event()
    outbound: list[dict[str, object]] = []
    failures: list[BaseException] = []
    callback_thread = threading.current_thread().name
    original_embedding = np.asarray([1.0, 0.0], dtype=np.float32)
    sdk_owned = object()

    class ShadowService:
        authoritative = False
        embedding_dim = 2
        thread_names: list[str] = []
        received: tuple[IdentityFramePrimitive, ...] = ()

        def process_source_frame(
            self,
            *,
            camera_id,
            frame_id,
            primitives,
            observed_at,
        ):
            assert camera_id == "camera-0"
            assert frame_id == 10
            assert observed_at == 100.0
            self.thread_names.append(threading.current_thread().name)
            self.received = tuple(primitives)
            scoring_started.set()
            assert release_scoring.wait(timeout=2.0)
            for primitive in self.received:
                primitive.public_track["identity_v2"] = {
                    "mode": "shadow",
                    "state": "resident",
                    "reason": "same_frame_test",
                    "fresh_embedding": True,
                }

    class Publisher:
        def _report_failure(self, error):
            failures.append(error)

        def publish(self, source_id, tracks, *, frame_metadata=None):
            outbound.extend(dict(track) for track in tracks)
            published.set()
            return _receipt(
                hooks,
                int(source_id),
                int(frame_metadata["frame_id"]),
                0,
            )

    service = ShadowService()
    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={},
            ds_pipeline=object(),
            identity_v2_service=service,
            identity_v2_shadow_failure_callback=failures.append,
        ),
        tracking_pub=Publisher(),
        camera_labels={0: "camera-0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
    )
    public_track: dict[str, object] = {
        "camera_id": "camera-0",
        "tracker_id": 7,
        "frame_id": 10,
        "class_id": 0,
        "opaque_sdk_meta": sdk_owned,
    }
    continuity = processor._prepare_tracking_cohort(
        sensor_id=0,
        camera_id="camera-0",
        frame_id=10,
        observed_at_us=1_000_010,
        tracks=[public_track],
        footpoints=[],
    )
    primitive = IdentityFramePrimitive(
        camera_id="camera-0",
        tracker_id="7",
        frame_id=10,
        public_track=public_track,
        diagnostic_track={"borrowed": sdk_owned},
        embedding=original_embedding,
        bbox=(1.0, 2.0, 3.0, 4.0),
        frame_size=(1280, 720),
        detection_confidence=0.9,
        tracker_confidence=0.8,
        world_xyz=(1.0, 0.0, 2.0),
        world_valid=True,
    )

    assert processor._enqueue_tracking_publication(
        sensor_id=0,
        camera_id="camera-0",
        frame_meta=SimpleNamespace(
            frame_number=10,
            frame_num=10,
            buf_pts=2_000_010_000,
        ),
        tracks=[public_track],
        footpoints=[],
        now_ts=100.0,
        temporal_contract={"observed_at_us": 1_000_010},
        continuity=continuity,
        force=True,
        identity_v2_primitives=[primitive],
    )
    assert scoring_started.wait(timeout=2.0)
    assert service.thread_names == ["NoesisShadowIdentity"]
    assert service.thread_names[0] != callback_thread
    # Slow optional scoring must not hold the canonical tracking publication.
    assert published.wait(timeout=2.0)

    # The queued primitive is a compact owned snapshot, not either callback
    # mapping or the mutable embedding sequence.
    original_embedding[0] = 99.0
    assert len(service.received) == 1
    detached = service.received[0]
    assert detached.public_track is not public_track
    assert detached.diagnostic_track is None
    assert detached.embedding == (1.0, 0.0)

    release_scoring.set()
    processor.shutdown(wait=True, timeout_s=2.0)

    assert published.is_set()
    assert failures == []
    assert len(outbound) == 1
    assert "identity_v2" not in outbound[0]
    assert outbound[0]["opaque_sdk_meta"] is None
    assert "embedding" not in outbound[0]
    assert "identity_v2" not in public_track


def test_shadow_identity_failure_does_not_stop_canonical_tracking() -> None:
    from noesis.identity_v2_service import IdentityFramePrimitive
    from noesis.pipelines import hooks
    from noesis_core.runtime_publication import RuntimePublicationGate

    identity_failures: list[BaseException] = []
    publication_failures: list[BaseException] = []
    publish_calls: list[int] = []

    class ShadowService:
        authoritative = False
        embedding_dim = 2

        def process_source_frame(self, **_kwargs):
            raise RuntimeError("shadow scoring failed")

    class Publisher:
        def _report_failure(self, error):
            publication_failures.append(error)

        def publish(self, _source_id, _tracks, *, frame_metadata=None):
            publish_calls.append(int(frame_metadata["frame_id"]))
            return _receipt(
                hooks,
                int(_source_id),
                int(frame_metadata["frame_id"]),
                len(publish_calls) - 1,
            )

    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={},
            ds_pipeline=object(),
            identity_v2_service=ShadowService(),
            identity_v2_shadow_failure_callback=identity_failures.append,
        ),
        tracking_pub=Publisher(),
        camera_labels={0: "camera-0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
    )
    track: dict[str, object] = {
        "camera_id": "camera-0",
        "tracker_id": 7,
        "frame_id": 10,
        "class_id": 0,
    }
    continuity = processor._prepare_tracking_cohort(
        sensor_id=0,
        camera_id="camera-0",
        frame_id=10,
        observed_at_us=1_000_010,
        tracks=[track],
        footpoints=[],
    )
    primitive = IdentityFramePrimitive(
        camera_id="camera-0",
        tracker_id="7",
        frame_id=10,
        public_track=track,
        embedding=(1.0, 0.0),
    )

    assert processor._enqueue_tracking_publication(
        sensor_id=0,
        camera_id="camera-0",
        frame_meta=SimpleNamespace(frame_number=10, buf_pts=2_000_010_000),
        tracks=[track],
        footpoints=[],
        now_ts=100.0,
        temporal_contract={"observed_at_us": 1_000_010},
        continuity=continuity,
        force=True,
        identity_v2_primitives=[primitive],
    )
    canonical_worker = processor._tracking_publication_worker
    shadow_worker = processor._shadow_identity_worker
    assert canonical_worker is not None
    assert shadow_worker is not None
    deadline = time.monotonic() + 2.0
    while shadow_worker.terminal_failure is None and time.monotonic() < deadline:
        time.sleep(0.01)

    next_track = {
        "camera_id": "camera-0",
        "tracker_id": 7,
        "frame_id": 11,
        "class_id": 0,
    }
    next_continuity = processor._prepare_tracking_cohort(
        sensor_id=0,
        camera_id="camera-0",
        frame_id=11,
        observed_at_us=1_000_011,
        tracks=[next_track],
        footpoints=[],
    )
    assert processor._enqueue_tracking_publication(
        sensor_id=0,
        camera_id="camera-0",
        frame_meta=SimpleNamespace(frame_number=11, buf_pts=2_000_011_000),
        tracks=[next_track],
        footpoints=[],
        now_ts=101.0,
        temporal_contract={"observed_at_us": 1_000_011},
        continuity=next_continuity,
        force=True,
        identity_v2_primitives=None,
    )
    deadline = time.monotonic() + 2.0
    while len(publish_calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    processor.shutdown(wait=True, timeout_s=2.0)

    assert isinstance(shadow_worker.terminal_failure, RuntimeError)
    assert str(shadow_worker.terminal_failure) == "shadow scoring failed"
    assert canonical_worker.terminal_failure is None
    assert len(identity_failures) == 1
    assert publication_failures == []
    assert publish_calls == [10, 11]


def test_authoritative_identity_remains_synchronous() -> None:
    from noesis.identity_v2_service import IdentityFramePrimitive
    from noesis.pipelines import hooks
    from noesis_core.runtime_publication import RuntimePublicationGate

    calls: list[str] = []

    class AuthoritativeService:
        authoritative = True

        def process_source_frame(self, *, primitives, **_kwargs):
            calls.append(threading.current_thread().name)
            primitives[0].public_track["stable_id"] = 42

    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={},
            identity_v2_service=AuthoritativeService(),
        ),
        tracking_pub=SimpleNamespace(),
        camera_labels={0: "camera-0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
    )
    track: dict[str, object] = {"tracker_id": 7}
    primitive = IdentityFramePrimitive(
        camera_id="camera-0",
        tracker_id="7",
        frame_id=10,
        public_track=track,
        embedding=(1.0, 0.0),
    )

    processor._process_identity_v2_source_frame(
        camera_id="camera-0",
        frame_id=10,
        primitives=[primitive],
        observed_at=100.0,
    )

    assert calls == [threading.current_thread().name]
    assert track["stable_id"] == 42


def test_empty_shadow_frame_still_advances_identity_independently() -> None:
    from noesis.pipelines import hooks
    from noesis_core.runtime_publication import RuntimePublicationGate

    events: list[tuple[str, int, int]] = []

    class ShadowService:
        authoritative = False
        embedding_dim = 256

        def process_source_frame(self, *, frame_id, primitives, **_kwargs):
            events.append(("identity", int(frame_id), len(primitives)))

    class Publisher:
        def publish(self, source_id, tracks, *, frame_metadata=None):
            frame_id = int(frame_metadata["frame_id"])
            events.append(("tracking", frame_id, len(tracks)))
            return _receipt(hooks, int(source_id), frame_id, 0)

    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={},
            ds_pipeline=object(),
            identity_v2_service=ShadowService(),
        ),
        tracking_pub=Publisher(),
        camera_labels={0: "camera-0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
    )
    continuity = processor._prepare_tracking_cohort(
        sensor_id=0,
        camera_id="camera-0",
        frame_id=11,
        observed_at_us=1_000_011,
        tracks=[],
        footpoints=[],
    )

    assert processor._enqueue_tracking_publication(
        sensor_id=0,
        camera_id="camera-0",
        frame_meta=SimpleNamespace(frame_number=11, buf_pts=2_000_011_000),
        tracks=[],
        footpoints=[],
        now_ts=101.0,
        temporal_contract={"observed_at_us": 1_000_011},
        continuity=continuity,
        force=True,
        identity_v2_primitives=[],
    )
    deadline = time.monotonic() + 2.0
    while len(events) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    processor.shutdown(wait=True, timeout_s=2.0)

    assert sorted(events) == [("identity", 11, 0), ("tracking", 11, 0)]


def test_shadow_primitive_tracker_mismatch_does_not_reject_canonical() -> None:
    from noesis.identity_v2_service import IdentityFramePrimitive
    from noesis.pipelines import hooks
    from noesis_core.runtime_publication import RuntimePublicationGate

    failures: list[BaseException] = []
    published = threading.Event()

    class ShadowService:
        authoritative = False
        embedding_dim = 2

        def process_source_frame(self, **_kwargs):
            raise AssertionError("mismatched work must never reach scoring")

    class Publisher:
        def publish(self, source_id, _tracks, *, frame_metadata=None):
            published.set()
            return _receipt(
                hooks,
                int(source_id),
                int(frame_metadata["frame_id"]),
                0,
            )

    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={},
            ds_pipeline=object(),
            identity_v2_service=ShadowService(),
            identity_v2_shadow_failure_callback=failures.append,
        ),
        tracking_pub=Publisher(),
        camera_labels={0: "camera-0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
    )
    track: dict[str, object] = {
        "camera_id": "camera-0",
        "tracker_id": 7,
        "frame_id": 10,
        "class_id": 0,
    }
    continuity = processor._prepare_tracking_cohort(
        sensor_id=0,
        camera_id="camera-0",
        frame_id=10,
        observed_at_us=1_000_010,
        tracks=[track],
        footpoints=[],
    )
    primitive = IdentityFramePrimitive(
        camera_id="camera-0",
        tracker_id="8",
        frame_id=10,
        public_track=track,
        embedding=(1.0, 0.0),
    )

    assert processor._enqueue_tracking_publication(
        sensor_id=0,
        camera_id="camera-0",
        frame_meta=SimpleNamespace(frame_number=10, buf_pts=2_000_010_000),
        tracks=[track],
        footpoints=[],
        now_ts=100.0,
        temporal_contract={"observed_at_us": 1_000_010},
        continuity=continuity,
        force=True,
        identity_v2_primitives=[primitive],
    )
    assert published.wait(timeout=2.0)
    processor.shutdown(wait=True, timeout_s=2.0)

    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    assert processor._tracking_publication_worker is not None
    assert processor._tracking_publication_worker.terminal_failure is None
    assert processor._shadow_identity_worker is None


@pytest.mark.parametrize(
    "service",
    [None, SimpleNamespace(authoritative=True, embedding_dim=2)],
    ids=["missing", "wrong-mode"],
)
def test_shadow_enqueue_reports_identity_service_contract_failure(service) -> None:
    from noesis.identity_v2_service import IdentityFramePrimitive
    from noesis.pipelines import hooks
    from noesis_core.runtime_publication import RuntimePublicationGate

    failures: list[BaseException] = []
    published = threading.Event()

    class Publisher:
        def publish(self, source_id, _tracks, *, frame_metadata=None):
            published.set()
            return _receipt(
                hooks,
                int(source_id),
                int(frame_metadata["frame_id"]),
                0,
            )

    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={},
            ds_pipeline=object(),
            identity_v2_service=service,
            identity_v2_shadow_failure_callback=failures.append,
        ),
        tracking_pub=Publisher(),
        camera_labels={0: "camera-0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
    )
    track: dict[str, object] = {
        "camera_id": "camera-0",
        "tracker_id": 7,
        "frame_id": 10,
        "class_id": 0,
    }
    continuity = processor._prepare_tracking_cohort(
        sensor_id=0,
        camera_id="camera-0",
        frame_id=10,
        observed_at_us=1_000_010,
        tracks=[track],
        footpoints=[],
    )
    primitive = IdentityFramePrimitive(
        camera_id="camera-0",
        tracker_id="7",
        frame_id=10,
        public_track=track,
        embedding=(1.0, 0.0),
    )

    assert processor._enqueue_tracking_publication(
        sensor_id=0,
        camera_id="camera-0",
        frame_meta=SimpleNamespace(frame_number=10, buf_pts=2_000_010_000),
        tracks=[track],
        footpoints=[],
        now_ts=100.0,
        temporal_contract={"observed_at_us": 1_000_010},
        continuity=continuity,
        force=True,
        identity_v2_primitives=[primitive],
    )
    assert published.wait(timeout=2.0)
    processor.shutdown(wait=True, timeout_s=2.0)

    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert processor._tracking_publication_worker is not None
    assert processor._tracking_publication_worker.terminal_failure is None
    assert processor._shadow_identity_worker is None


def test_shadow_tracker_key_is_source_epoch_scoped_across_reconnect() -> None:
    from noesis.identity_v2_service import IdentityFramePrimitive
    from noesis.pipelines import hooks
    from noesis_core.runtime_publication import RuntimePublicationGate

    identity_tracker_ids: list[str] = []
    last_frame_by_tracker: dict[str, int] = {}
    published_frames: list[int] = []
    failures: list[BaseException] = []

    class ShadowService:
        authoritative = False
        embedding_dim = 2

        def process_source_frame(self, *, frame_id, primitives, **_kwargs):
            for primitive in primitives:
                tracker_id = str(primitive.tracker_id)
                previous = last_frame_by_tracker.get(tracker_id)
                if previous is not None and int(frame_id) <= previous:
                    raise RuntimeError("identity coordinator saw a replay")
                last_frame_by_tracker[tracker_id] = int(frame_id)
                identity_tracker_ids.append(tracker_id)

    class Publisher:
        sequence = 0

        def _report_failure(self, error):
            failures.append(error)

        def publish(self, source_id, _tracks, *, frame_metadata=None):
            metadata = dict(frame_metadata or {})
            frame_id = int(metadata["frame_id"])
            published_frames.append(frame_id)
            receipt = hooks.TrackingPublicationReceipt(
                source_id=int(source_id),
                frame_id=frame_id,
                observed_at_us=int(metadata["observed_at_us"]),
                tracking_publication_sequence=self.sequence,
                outbound_submission_id=(self.sequence * 2) + 1,
                outbound_message_count=1,
            )
            self.sequence += 1
            return receipt

    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={},
            ds_pipeline=object(),
            identity_v2_service=ShadowService(),
            identity_v2_failure_callback=failures.append,
        ),
        tracking_pub=Publisher(),
        camera_labels={0: "camera-0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
    )

    def enqueue(frame_id: int, observed_at_us: int, now_ts: float) -> None:
        epoch = processor._begin_source_frame_timeline(
            sensor_id=0,
            frame_id=frame_id,
            observed_at_us=observed_at_us,
        )
        track: dict[str, object] = {
            "camera_id": "camera-0",
            "tracker_id": 7,
            "frame_id": frame_id,
            "class_id": 0,
        }
        continuity = processor._prepare_tracking_cohort(
            sensor_id=0,
            camera_id="camera-0",
            frame_id=frame_id,
            observed_at_us=observed_at_us,
            tracks=[track],
            footpoints=[],
        )
        assert int(continuity.source_epoch) == int(epoch)
        primitive = IdentityFramePrimitive(
            camera_id="camera-0",
            tracker_id=hooks._identity_v2_tracker_id_for_epoch(7, epoch),
            frame_id=frame_id,
            public_track=track,
            embedding=(1.0, 0.0),
        )
        assert processor._enqueue_tracking_publication(
            sensor_id=0,
            camera_id="camera-0",
            frame_meta=SimpleNamespace(
                frame_number=frame_id,
                buf_pts=observed_at_us * 1_000,
            ),
            tracks=[track],
            footpoints=[],
            now_ts=now_ts,
            temporal_contract={"observed_at_us": observed_at_us},
            continuity=continuity,
            force=True,
            identity_v2_primitives=[primitive],
        )

    enqueue(10, 1_000_010, 100.0)
    deadline = time.monotonic() + 2.0
    while len(identity_tracker_ids) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    # A lower frame number with a newer observed time is a source reconnect.
    # The raw tracker ID is reused, but the identity coordinator key is not.
    enqueue(1, 2_000_001, 101.0)
    deadline = time.monotonic() + 2.0
    while (
        (len(identity_tracker_ids) < 2 or len(published_frames) < 2)
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    processor.shutdown(wait=True, timeout_s=2.0)

    assert failures == []
    assert published_frames == [10, 1]
    assert identity_tracker_ids == ["7", "7@source_epoch:1"]


def test_shadow_runs_only_for_admitted_tracking_publication_cohorts() -> None:
    from noesis.identity_v2_service import IdentityFramePrimitive
    from noesis.pipelines import hooks
    from noesis_core.runtime_publication import RuntimePublicationGate

    identity_frames: list[int] = []
    published_frames: list[int] = []

    class ShadowService:
        authoritative = False
        embedding_dim = 2

        def process_source_frame(self, *, frame_id, **_kwargs):
            identity_frames.append(int(frame_id))

    class Publisher:
        sequence = 0

        def publish(self, source_id, _tracks, *, frame_metadata=None):
            metadata = dict(frame_metadata or {})
            frame_id = int(metadata["frame_id"])
            published_frames.append(frame_id)
            receipt = hooks.TrackingPublicationReceipt(
                source_id=int(source_id),
                frame_id=frame_id,
                observed_at_us=int(metadata["observed_at_us"]),
                tracking_publication_sequence=self.sequence,
                outbound_submission_id=(self.sequence * 2) + 1,
                outbound_message_count=1,
            )
            self.sequence += 1
            return receipt

    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={},
            ds_pipeline=object(),
            identity_v2_service=ShadowService(),
        ),
        tracking_pub=Publisher(),
        camera_labels={0: "camera-0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
    )
    processor._tracking_publish_interval_s = 1.0

    def enqueue(frame_id: int, observed_at_us: int, now_ts: float) -> bool:
        track: dict[str, object] = {
            "camera_id": "camera-0",
            "tracker_id": 7,
            "frame_id": frame_id,
            "class_id": 0,
        }
        continuity = processor._prepare_tracking_cohort(
            sensor_id=0,
            camera_id="camera-0",
            frame_id=frame_id,
            observed_at_us=observed_at_us,
            tracks=[track],
            footpoints=[],
        )
        primitive = IdentityFramePrimitive(
            camera_id="camera-0",
            tracker_id="7",
            frame_id=frame_id,
            public_track=track,
            embedding=(1.0, 0.0),
        )
        return processor._enqueue_tracking_publication(
            sensor_id=0,
            camera_id="camera-0",
            frame_meta=SimpleNamespace(
                frame_number=frame_id,
                buf_pts=observed_at_us * 1_000,
            ),
            tracks=[track],
            footpoints=[],
            now_ts=now_ts,
            temporal_contract={"observed_at_us": observed_at_us},
            continuity=continuity,
            force=False,
            identity_v2_primitives=[primitive],
        )

    assert enqueue(1, 1_000_001, 100.0) is True
    assert enqueue(2, 1_000_002, 100.1) is False
    deadline = time.monotonic() + 2.0
    while (
        (len(identity_frames) < 1 or len(published_frames) < 1)
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    processor.shutdown(wait=True, timeout_s=2.0)

    assert identity_frames == [1]
    assert published_frames == [1]


def test_handler_shadow_stable_id_none_dispatches_off_callback_without_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from noesis.pipelines import hooks
    from noesis_core.runtime_publication import RuntimePublicationGate

    scoring_started = threading.Event()
    release_scoring = threading.Event()
    published = threading.Event()
    received_stable_ids: list[object] = []
    outbound: list[dict[str, object]] = []
    callback_thread = threading.current_thread().name

    class ShadowService:
        authoritative = False
        embedding_dim = 2

        def process_source_frame(self, *, primitives, **_kwargs):
            assert threading.current_thread().name == "NoesisShadowIdentity"
            assert threading.current_thread().name != callback_thread
            received_stable_ids.extend(
                primitive.public_track.get("stable_id") for primitive in primitives
            )
            scoring_started.set()
            assert release_scoring.wait(timeout=2.0)
            for primitive in primitives:
                primitive.public_track["identity_v2"] = {
                    "mode": "shadow",
                    "state": "unknown",
                    "reason": "handler_dispatch_test",
                    "fresh_embedding": True,
                }

    class Publisher:
        def publish(self, source_id, tracks, *, frame_metadata=None):
            metadata = dict(frame_metadata or {})
            outbound.extend(dict(track) for track in tracks)
            published.set()
            return hooks.TrackingPublicationReceipt(
                source_id=int(source_id),
                frame_id=int(metadata["frame_id"]),
                observed_at_us=int(metadata["observed_at_us"]),
                tracking_publication_sequence=0,
                outbound_submission_id=1,
                outbound_message_count=1,
            )

    processor = hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={"models": {"reid": {"gie_id": 3}}},
            ds_pipeline=object(),
            stable_id_mgr=None,
            identity_v2_service=ShadowService(),
        ),
        tracking_pub=Publisher(),
        camera_labels={0: "camera-0"},
        sensor_id_map={},
        publication_gate=RuntimePublicationGate(),
    )
    monkeypatch.setattr(
        processor,
        "_build_track_dict_ds8",
        lambda _obj, _camera: {
            "track_id": 7,
            "bbox": [10.0, 20.0, 40.0, 80.0],
            "center": [30.0, 60.0],
            "class_id": 0,
            "confidence": 0.91,
            "tracker_confidence": 0.80,
            "analytics": {},
            "zone": "Kitchen",
        },
    )
    monkeypatch.setattr(
        processor,
        "_extract_reid_embedding_ds8",
        lambda _obj: np.asarray([1.0, 0.0], dtype=np.float32),
    )
    monkeypatch.setattr(processor, "_maybe_assign_stable_id", lambda **_kwargs: None)
    monkeypatch.setattr(
        processor,
        "_extract_pose_keypoints_for_anchor",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(processor, "_extract_object_depth_result", lambda _obj: None)
    monkeypatch.setattr(
        processor,
        "_augment_track_with_world",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        processor,
        "_apply_scene_prior_shadow",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        processor,
        "_apply_public_depth_fields",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(processor, "_stamp_osd_label_ds8", lambda *_a, **_k: None)
    monkeypatch.setattr(
        processor,
        "_apply_instance_mask_color_ds8",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(processor, "_footpoint_from_track", lambda *_a, **_k: None)
    monkeypatch.setattr(processor, "_publish_occupancy", lambda *_a, **_k: None)
    monkeypatch.setattr(processor, "_cleanup_zone_state", lambda *_a, **_k: None)
    monkeypatch.setattr(processor, "_maintain_stable_ids", lambda *_a, **_k: None)
    monkeypatch.setattr(
        processor,
        "_maybe_log_stable_id_metrics",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        processor,
        "_clear_absent_world_state",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(processor, "_log_diag_session_start", lambda: None)

    obj = SimpleNamespace(object_id=7, class_id=0)
    frame = SimpleNamespace(
        source_id=0,
        frame_number=10,
        frame_width=1280,
        frame_height=720,
        buf_pts=2_000_010_000,
        object_items=(obj,),
    )
    processor.handle_frame_ds8(frame)

    assert scoring_started.wait(timeout=2.0)
    assert received_stable_ids == [None]
    assert published.wait(timeout=2.0)
    release_scoring.set()
    processor.shutdown(wait=True, timeout_s=2.0)

    assert published.is_set()
    assert len(outbound) == 1
    assert outbound[0]["stable_id"] is None
    assert "identity_v2" not in outbound[0]
    assert "embedding" not in outbound[0]
