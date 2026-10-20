from __future__ import annotations

import time
from pathlib import Path

import pytest

from .companion_capture import (
    CompanionCamera,
    CompanionCaptureConflict,
    CompanionCaptureError,
    CompanionCaptureLimits,
    CompanionCaptureManager,
    GstCompressedRecorder,
)
from .static_capture_sources import resolve_active_source_authority


class _Recorder:
    def __init__(self, _source_uri: str, output_dir: Path, **_: object) -> None:
        self.output_dir = output_dir
        self.value = {
            "status": "created",
            "encoded_ready": False,
            "packet_count": 0,
            "error": None,
        }
        self.stopped = False

    def start(self) -> None:
        self.value.update({"status": "recording", "encoded_ready": True, "packet_count": 1})

    def stop(self, reason: str = "user") -> dict[str, object]:
        self.stopped = True
        (self.output_dir / "static_camera.mkv").write_bytes(b"encoded")
        (self.output_dir / "packet_timing.jsonl").write_text("{}\n", encoding="utf-8")
        self.value.update(
            {
                "status": "stopped",
                "stop_reason": reason,
                "video_path": "static_camera.mkv",
                "packet_timing_path": "packet_timing.jsonl",
            }
        )
        return dict(self.value)

    def status(self) -> dict[str, object]:
        return dict(self.value)


class _Tracking:
    def __init__(self, authority: object, output_dir: Path, _max_records: int) -> None:
        self.authority = authority
        self.output_dir = output_dir
        self.stopped = False
        self.value = {
            "status": "idle",
            "tracking_ready": False,
            "source_id": 0,
            "record_count": 0,
            "last_tracking_publication_sequence": None,
            "partial": False,
            "error": None,
        }

    def start(self) -> None:
        self.value.update(
            {
                "status": "recording",
                "tracking_ready": True,
                "record_count": 1,
                "last_tracking_publication_sequence": 41,
            }
        )

    def stop(self, reason: str = "user") -> dict[str, object]:
        self.stopped = True
        (self.output_dir / "tracking.ndjson").write_text("{}\n", encoding="utf-8")
        (self.output_dir / "provenance.json").write_text("{}\n", encoding="utf-8")
        self.value.update(
            {
                "status": "stopped",
                "stop_reason": reason,
                "artifacts": {"tracking": "tracking.ndjson", "provenance": "provenance.json"},
            }
        )
        return dict(self.value)

    def status(self) -> dict[str, object]:
        return dict(self.value)


class _SlowRecorder(_Recorder):
    def start(self) -> None:
        time.sleep(0.7)
        super().start()


def _authority(tmp_path: Path):
    pipeline = tmp_path / "infer.yaml"
    pipeline.write_text("version: 1\nsources:\n  - uri_secret: kitchen\n", encoding="utf-8")
    cameras = tmp_path / "cameras.yaml"
    cameras.write_text("version: 1\ncameras:\n  0:\n    name: kitchen\n", encoding="utf-8")
    return resolve_active_source_authority(
        "kitchen",
        pipeline_config=pipeline,
        cameras_config=cameras,
        camera_registry={"kitchen": "rtsp://example.test/live"},
    )


def test_manager_uses_one_authority_and_preserves_final_probe_marker(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    camera = CompanionCamera("kitchen", "Kitchen", "camera-secret:kitchen", authority.public_snapshot(), 0)
    seen: list[object] = []

    def tracking_factory(received: object, output_dir: Path, max_records: int) -> _Tracking:
        seen.append(received)
        return _Tracking(received, output_dir, max_records)

    manager = CompanionCaptureManager(
        tmp_path,
        camera_catalog=lambda: [camera],
        source_resolver=lambda _camera_id, require_process=True: authority,
        recorder_factory=_Recorder,
        tracking_observer_factory=tracking_factory,
        limits=CompanionCaptureLimits(readiness_timeout_s=1.0),
    )
    try:
        initial = manager.start_session(
            "kitchen",
            phone_capture_id="phone-1",
            clock_probes=[{"name": "start_1"}],
        )
        public = manager.public_state(initial["session_id"])
        assert public["status"] == "recording"
        assert public["video_status"] == "recording"
        assert public["tracking_status"] == "recording"
        assert seen == [authority]

        stopped = manager.stop_session(
            initial["session_id"],
            phone_capture_id="phone-1",
            clock_probes=[{"name": "stop_1"}],
            markers=[{"marker_id": "door", "client_monotonic_ms": 12, "client_epoch_ms": 13}],
        )
        assert stopped["status"] == "stopped"
        assert [row["client_probe"]["name"] for row in stopped["clock_exchanges"]] == ["start_1", "stop_1"]
        assert stopped["markers"][0]["marker_id"] == "door"
        public = manager.public_state(initial["session_id"])
        assert public["tracking_sequence"] == 41
        assert "tracking" in public["artifact_urls"]
        assert "tracking_provenance" in public["artifact_urls"]
    finally:
        manager.shutdown()


def test_duplicate_client_request_rejects_changed_phone_or_camera(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    camera = CompanionCamera("kitchen", "Kitchen", "camera-secret:kitchen", authority.public_snapshot(), 0)
    manager = CompanionCaptureManager(
        tmp_path,
        camera_catalog=lambda: [camera],
        source_resolver=lambda _camera_id, require_process=True: authority,
        recorder_factory=_Recorder,
        tracking_observer_factory=lambda _authority, output_dir, max_records: _Tracking(_authority, output_dir, max_records),
        limits=CompanionCaptureLimits(readiness_timeout_s=1.0),
    )
    try:
        manager.start_session("kitchen", client_request_id="retry-1", phone_capture_id="phone-1")
        with pytest.raises(CompanionCaptureConflict):
            manager.start_session("kitchen", client_request_id="retry-1", phone_capture_id="phone-2")
    finally:
        manager.shutdown()


def test_live_component_failure_stops_recorder_and_projects_failed(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    camera = CompanionCamera("kitchen", "Kitchen", "camera-secret:kitchen", authority.public_snapshot(), 0)
    recorders: list[_Recorder] = []
    trackers: list[_Tracking] = []

    def recorder_factory(source_uri: str, output_dir: Path, **kwargs: object) -> _Recorder:
        recorder = _Recorder(source_uri, output_dir, **kwargs)
        recorders.append(recorder)
        return recorder

    def tracking_factory(received: object, output_dir: Path, max_records: int) -> _Tracking:
        tracker = _Tracking(received, output_dir, max_records)
        trackers.append(tracker)
        return tracker

    manager = CompanionCaptureManager(
        tmp_path,
        camera_catalog=lambda: [camera],
        source_resolver=lambda _camera_id, require_process=True: authority,
        recorder_factory=recorder_factory,
        tracking_observer_factory=tracking_factory,
        limits=CompanionCaptureLimits(readiness_timeout_s=1.0),
    )
    try:
        started = manager.start_session("kitchen", phone_capture_id="phone-1")
        trackers[0].value.update(
            {
                "status": "failed",
                "tracking_ready": False,
                "partial": True,
                "error": "tracking_stale_timeout",
            }
        )
        heartbeat = manager.heartbeat(started["session_id"], phone_capture_id="phone-1")
        assert heartbeat["status"] == "partial"
        assert heartbeat["stop_reason"] == "tracking:tracking_stale_timeout"
        assert recorders[0].stopped is True
        public = manager.public_state(started["session_id"])
        assert public["status"] == "failed"
        assert public["tracking_status"] == "partial"
    finally:
        manager.shutdown()


def test_startup_watchdog_allows_not_ready_component_states(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    camera = CompanionCamera("kitchen", "Kitchen", "camera-secret:kitchen", authority.public_snapshot(), 0)
    manager = CompanionCaptureManager(
        tmp_path,
        camera_catalog=lambda: [camera],
        source_resolver=lambda _camera_id, require_process=True: authority,
        recorder_factory=_SlowRecorder,
        tracking_observer_factory=lambda _authority, output_dir, max_records: _Tracking(_authority, output_dir, max_records),
        limits=CompanionCaptureLimits(readiness_timeout_s=2.0),
    )
    try:
        started = manager.start_session("kitchen", phone_capture_id="phone-1")
        assert manager.public_state(started["session_id"])["status"] == "recording"
    finally:
        manager.shutdown()


def test_phone_upload_failure_does_not_poison_healthy_static_lifecycle(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    camera = CompanionCamera("kitchen", "Kitchen", "camera-secret:kitchen", authority.public_snapshot(), 0)
    manager = CompanionCaptureManager(
        tmp_path,
        camera_catalog=lambda: [camera],
        source_resolver=lambda _camera_id, require_process=True: authority,
        recorder_factory=_Recorder,
        tracking_observer_factory=lambda _authority, output_dir, max_records: _Tracking(_authority, output_dir, max_records),
        limits=CompanionCaptureLimits(readiness_timeout_s=1.0),
    )
    try:
        started = manager.start_session("kitchen", phone_capture_id="phone-1")
        stopped = manager.stop_session(started["session_id"])
        failed_archive = tmp_path / "failed-upload.tar"
        failed_archive.write_bytes(b"failed phone bundle")
        failure = manager.record_upload_failure(
            started["session_id"],
            "phone-1",
            failed_archive,
            "malformed manifest",
        )
        assert failure["status"] == "stopped"
        assert failure["phone"]["upload_error"] == "malformed manifest"
        assert failure["phone"]["failed_archive_path"] == "phone_uploads/phone-1.partial"

        # Reproduce an upload-only partial written by an earlier service
        # version; the static component records remain healthy and finalized.
        with manager._lock:
            partial = manager._read(started["session_id"])
            partial["status"] = "partial"
            manager._write(started["session_id"], partial)
        associated = manager.associate_phone_bundle(
            started["session_id"],
            capture_id="phone-1",
            archive_sha256="a" * 64,
            scan_id="20260906-010101-deadbeef",
        )
        assert associated["status"] == "complete"
        assert "upload_error" not in associated["phone"]
        assert "failed_archive_path" not in associated["phone"]
        assert associated["phone"]["previous_failed_archive_path"] == "phone_uploads/phone-1.partial"
    finally:
        manager.shutdown()


def test_lease_expiry_stops_both_components_and_projects_failed(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    camera = CompanionCamera("kitchen", "Kitchen", "camera-secret:kitchen", authority.public_snapshot(), 0)
    recorders: list[_Recorder] = []
    trackers: list[_Tracking] = []

    def recorder_factory(source_uri: str, output_dir: Path, **kwargs: object) -> _Recorder:
        recorder = _Recorder(source_uri, output_dir, **kwargs)
        recorders.append(recorder)
        return recorder

    def tracking_factory(received: object, output_dir: Path, max_records: int) -> _Tracking:
        tracker = _Tracking(received, output_dir, max_records)
        trackers.append(tracker)
        return tracker

    manager = CompanionCaptureManager(
        tmp_path,
        camera_catalog=lambda: [camera],
        source_resolver=lambda _camera_id, require_process=True: authority,
        recorder_factory=recorder_factory,
        tracking_observer_factory=tracking_factory,
        limits=CompanionCaptureLimits(readiness_timeout_s=1.0, lease_s=1.0),
    )
    try:
        started = manager.start_session("kitchen", phone_capture_id="phone-1")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and manager._active_session_id is not None:
            time.sleep(0.05)
        assert manager._active_session_id is None
        raw = manager._read(started["session_id"])
        assert raw["status"] == "partial"
        assert raw["stop_reason"] == "lease_expired"
        assert recorders[0].stopped is True
        assert trackers[0].stopped is True
        assert manager.public_state(started["session_id"])["status"] == "failed"
    finally:
        manager.shutdown()


def _multi_manager(tmp_path: Path, *, delayed: bool = False, fail_start: bool = False, lease_s: float = 45.0):
    pipeline = tmp_path / "multi-infer.yaml"
    pipeline.write_text("version: 1\nsources:\n  - uri_secret: kitchen\n  - uri_secret: family-room\n", encoding="utf-8")
    camera_file = tmp_path / "multi-cameras.yaml"
    camera_file.write_text("version: 1\ncameras:\n  0:\n    name: kitchen\n  1:\n    name: family-room\n", encoding="utf-8")
    authorities = {
        camera_id: resolve_active_source_authority(
            camera_id, pipeline_config=pipeline, cameras_config=camera_file,
            camera_registry={"kitchen": "rtsp://fixture/k", "family-room": "rtsp://fixture/f"},
        ) for camera_id in ("kitchen", "family-room")
    }
    cameras = [CompanionCamera(key, key, f"camera-secret:{key}", value.public_snapshot(), value.source_id)
               for key, value in authorities.items()]
    recorders = []
    trackers = []
    budgets = []

    class Recorder(_Recorder):
        def start(self):
            super().start()
            if fail_start and self is recorders[1]:
                raise RuntimeError("second camera failed")

    def recorder_factory(uri, output_dir, **kwargs):
        recorder = Recorder(uri, output_dir)
        recorders.append(recorder)
        budgets.append(kwargs)
        return recorder

    class Tracking(_Tracking):
        def start(self):
            super().start()
            self.value["source_id"] = self.authority.source_id
            if delayed and self.authority.source_id == 1:
                self.value.update(tracking_ready=False, record_count=0)

    def tracking_factory(authority, output_dir, max_records):
        observer = Tracking(authority, output_dir, max_records)
        observer.budget = max_records
        trackers.append(observer)
        return observer

    manager = CompanionCaptureManager(
        tmp_path, camera_catalog=lambda: cameras,
        source_resolver=lambda key, require_process=True: authorities[key],
        recorder_factory=recorder_factory, tracking_observer_factory=tracking_factory,
        limits=CompanionCaptureLimits(readiness_timeout_s=1, lease_s=lease_s,
                                      max_session_bytes=1000, max_packet_records=80, max_tracking_records=40),
    )
    return manager, recorders, trackers, budgets


def test_multi_camera_waits_for_both_and_retains_separate_evidence(tmp_path: Path) -> None:
    manager, recorders, trackers, budgets = _multi_manager(tmp_path, delayed=True)
    try:
        started = manager.start_session("kitchen", camera_ids=["family-room", "kitchen"],
                                        phone_capture_id="one-phone", client_request_id="one-request", await_ready=False)
        sid = started["session_id"]
        assert manager.public_state(sid)["status"] == "starting"
        assert manager.public_state(sid)["cameras"]["family-room"]["readiness"]["tracking_ready"] is False
        trackers[1].value.update(tracking_ready=True, record_count=1)
        deadline = time.monotonic() + 1
        while manager.public_state(sid)["status"] == "starting" and time.monotonic() < deadline:
            time.sleep(.01)
        public = manager.public_state(sid)
        assert public["status"] == "recording"
        assert public["camera_ids"] == ["kitchen", "family-room"]
        assert all(row["readiness"]["recorder_ready"] and row["readiness"]["tracking_ready"] for row in public["cameras"].values())
        assert sum(row["max_bytes"] for row in budgets) == 1000
        assert sum(row["max_packet_records"] for row in budgets) == 80
        assert sum(row["packet_queue_items"] for row in budgets) == 4096
        assert sum(row["packet_queue_bytes"] for row in budgets) == 32 * 1024 * 1024
        assert sum(observer.budget for observer in trackers) == 40
        assert all(row["max_duration_s"] == 900 for row in budgets)
        assert manager.start_session("kitchen", camera_ids=["kitchen", "family-room"],
                                     phone_capture_id="one-phone", client_request_id="one-request")["session_id"] == sid
        with pytest.raises(CompanionCaptureConflict):
            manager.start_session("kitchen", client_request_id="one-request", phone_capture_id="one-phone")
        stopped = manager.stop_session(sid, phone_capture_id="one-phone")
        assert stopped["status"] == "stopped"
        assert all(recorder.stopped for recorder in recorders) and all(observer.stopped for observer in trackers)
        public = manager.public_state(sid)
        assert public["artifacts"]["static_video"] == "static_camera.mkv"
        for key, prefix in (("kitchen", ""), ("family-room", "cameras/1/")):
            row = public["cameras"][key]
            assert row["artifacts"]["static_video"] == prefix + "static_camera.mkv"
            assert row["artifacts"]["tracking"] == prefix + "tracking.ndjson"
            assert row["authority"]["camera_id"] == key
            assert row["provenance"]["timing"]["phone_server_clock_exchange_verified"] is False
            assert row["artifact_urls"]["static_video"].endswith("/" + prefix + "static_camera.mkv")
        assert manager.stop_session(sid) == manager.status(sid)
        scan = manager.reserve_phone_scan_id(sid, "one-phone")
        assert manager.associate_phone_bundle(sid, capture_id="one-phone", archive_sha256="a" * 64, scan_id=scan)["status"] == "complete"
        assert manager.check_phone_archive(sid, "one-phone", "a" * 64)["camera_ids"] == ["kitchen", "family-room"]
    finally:
        manager.shutdown()


@pytest.mark.parametrize("selection", [[], ["kitchen", "kitchen"], ["family-room"], ["kitchen", "family-room", "third"],
                                      "kitchen", ["kitchen", " family-room"], ["kitchen", 1], ["kitchen", "unknown"]])
def test_multi_camera_selection_is_exact_and_bounded(tmp_path: Path, selection) -> None:
    manager, recorders, _, _ = _multi_manager(tmp_path)
    try:
        with pytest.raises(CompanionCaptureError):
            manager.start_session("kitchen", camera_ids=selection)
        assert not recorders
    finally:
        manager.shutdown()


@pytest.mark.parametrize("failure", ["tracking", "readiness", "start", "lease"])
def test_multi_camera_failure_stops_every_participant(tmp_path: Path, failure: str) -> None:
    manager, recorders, trackers, _ = _multi_manager(tmp_path, delayed=failure == "readiness",
                                                   fail_start=failure == "start", lease_s=.2 if failure == "lease" else 45)
    try:
        if failure == "start":
            with pytest.raises(CompanionCaptureError, match="could not start"):
                manager.start_session("kitchen", camera_ids=["kitchen", "family-room"])
        else:
            started = manager.start_session("kitchen", camera_ids=["kitchen", "family-room"], await_ready=failure != "readiness")
            if failure == "tracking":
                trackers[1].value.update(status="failed", error="tracking_stale_timeout", partial=True)
                assert manager.heartbeat(started["session_id"])["status"] == "partial"
            deadline = time.monotonic() + 3
            while manager._active_session_id and time.monotonic() < deadline:
                time.sleep(.02)
        assert manager._active_session_id is None
        assert all(recorder.stopped for recorder in recorders) and all(observer.stopped for observer in trackers)
        saved = manager.list_sessions()[0]
        assert saved["status"] == "failed"
        assert len(saved["cameras"]) == 2
        assert all(row["artifact_urls"].get("static_video") for row in saved["cameras"].values())
    finally:
        manager.shutdown()


def test_multi_camera_restart_preserves_members_and_legacy_sessions(tmp_path: Path) -> None:
    manager, _, _, _ = _multi_manager(tmp_path)
    started = manager.start_session("kitchen", camera_ids=["kitchen", "family-room"])
    sid = started["session_id"]
    manager.shutdown()
    interrupted = manager._read(sid)
    interrupted["status"] = "recording"
    manager._write(sid, interrupted)
    recovered, _, _, _ = _multi_manager(tmp_path)
    try:
        assert recovered.public_state(sid)["status"] == "failed"
        assert recovered.status(sid)["stop_reason"] == "service_restart"
        assert len(recovered.public_state(sid)["cameras"]) == 2
        # Existing durable v1 sessions have no cameras map or camera_ids.
        legacy = recovered._read(sid)
        legacy.pop("cameras")
        legacy.pop("camera_ids")
        legacy["status"] = "stopped"
        recovered._write(sid, legacy)
        public = recovered.public_state(sid)
        assert public["camera_ids"] == ["kitchen"]
        assert public["artifacts"]["static_video"] == "static_camera.mkv"
    finally:
        recovered.shutdown()


def test_multi_camera_component_start_uses_one_readiness_deadline(tmp_path: Path) -> None:
    manager, recorders, trackers, _ = _multi_manager(tmp_path)
    manager.limits = CompanionCaptureLimits(readiness_timeout_s=.05)
    original_factory = manager.recorder_factory

    def slow_factory(uri, output_dir, **kwargs):
        recorder = original_factory(uri, output_dir, **kwargs)
        start = recorder.start

        def slow_start():
            time.sleep(.08)
            start()

        recorder.start = slow_start
        return recorder

    manager.recorder_factory = slow_factory
    try:
        with pytest.raises(CompanionCaptureError, match="could not start"):
            manager.start_session("kitchen", camera_ids=["kitchen", "family-room"])
        assert manager._active_session_id is None
        assert all(recorder.stopped for recorder in recorders)
        assert all(observer.stopped for observer in trackers)
    finally:
        manager.shutdown()


def test_heartbeat_cannot_extend_multi_camera_readiness_timeout(tmp_path: Path) -> None:
    manager, recorders, trackers, _ = _multi_manager(tmp_path, delayed=True)
    manager.limits = CompanionCaptureLimits(readiness_timeout_s=.1)
    try:
        started = manager.start_session("kitchen", camera_ids=["kitchen", "family-room"], await_ready=False)
        manager.heartbeat(started["session_id"])
        deadline = time.monotonic() + 1
        while manager._active_session_id and time.monotonic() < deadline:
            time.sleep(.01)
        assert manager.status(started["session_id"])["stop_reason"] == "readiness_timeout"
        assert all(recorder.stopped for recorder in recorders) and all(observer.stopped for observer in trackers)
    finally:
        manager.shutdown()


def test_multi_camera_later_factory_failure_reconciles_allocated_participants(tmp_path: Path) -> None:
    manager, recorders, trackers, _ = _multi_manager(tmp_path)
    original = manager.recorder_factory

    def factory(uri, output_dir, **kwargs):
        if recorders:
            raise RuntimeError("second factory failed")
        return original(uri, output_dir, **kwargs)

    manager.recorder_factory = factory
    try:
        with pytest.raises(CompanionCaptureError, match="could not be constructed"):
            manager.start_session("kitchen", camera_ids=["kitchen", "family-room"])
        assert manager._active_session_id is None
        assert len(recorders) == len(trackers) == 1
        assert recorders[0].stopped and trackers[0].stopped
    finally:
        manager.shutdown()


def test_builtin_multi_camera_components_receive_partitioned_limits(tmp_path: Path) -> None:
    manager, _, _, _ = _multi_manager(tmp_path)
    try:
        manager.tracking_observer_factory = None
        camera = manager._camera("kitchen")
        _, authority = manager._resolve_source(camera)
        observer = manager._make_observer(camera, tmp_path / "bounded", authority, max_records=20, camera_count=2)
        assert observer._observer.limits.max_records == 20
        assert observer._observer.limits.max_file_bytes == 256 * 1024 * 1024
        assert observer._observer.limits.max_queue_bytes == 16 * 1024 * 1024
        assert observer._observer._queue.maxsize == 1024
        recorder = GstCompressedRecorder("rtsp://fixture/k", tmp_path / "bounded", packet_queue_items=2048,
                                         packet_queue_bytes=16 * 1024 * 1024)
        assert recorder._packet_queue.maxsize == 2048
        assert recorder._packet_queue_bytes_limit == 16 * 1024 * 1024
    finally:
        manager.shutdown()
