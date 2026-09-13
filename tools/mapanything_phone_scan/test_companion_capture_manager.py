from __future__ import annotations

import time
from pathlib import Path

import pytest

from .companion_capture import (
    CompanionCamera,
    CompanionCaptureConflict,
    CompanionCaptureLimits,
    CompanionCaptureManager,
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
