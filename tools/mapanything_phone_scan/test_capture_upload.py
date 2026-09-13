from __future__ import annotations

import asyncio
import hashlib
import io
import json
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from . import app as phone_app
from . import capture
from .capture_upload import ASYNC_VIDEO_PROBE_TIMEOUT_S, UPLOAD_RECEIPT_SCHEMA
from .test_android_capture import FRAME_TIMES, _manifest, _bundle, encoded_video
from .test_browser_capture import _settings
from .test_companion_upload import CAMERA_ID, CAPTURE_ID, SCAN_ID, SESSION_ID, _FakeCompanionManager


def _archive(capture_id: str = CAPTURE_ID, *, paired: bool = False, tag: str = "same") -> bytes:
    manifest = _manifest()
    manifest["capture_id"] = capture_id
    if paired:
        manifest["companion_capture"] = {
            "session_id": SESSION_ID, "camera_id": CAMERA_ID, "phone_capture_id": capture_id,
        }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("capture_manifest.json", json.dumps(manifest))
        archive.writestr("camera.mp4", tag.encode())
    return output.getvalue()


def _headers(*, paired: bool = False, asynchronous: bool = True) -> dict[str, str]:
    headers = {"Content-Type": "application/zip", "X-File-Name": "native.zip"}
    if asynchronous:
        headers["Prefer"] = "respond-async"
    if paired:
        headers.update({"X-Companion-Session": SESSION_ID, "X-Companion-Camera-ID": CAMERA_ID,
                        "X-Phone-Capture-ID": CAPTURE_ID})
    return headers


def _fake_import(path: Path, scan_dir: Path, *, limits: capture.CaptureImportLimits) -> dict[str, Any]:
    manifest = capture.validate_capture_manifest(phone_app._read_sensor_bundle_manifest(path, limits=limits))
    target = scan_dir / "capture"
    target.mkdir()
    (target / "camera.mp4").write_bytes(b"validated-video")
    return {
        "schema": "noesis.phone_capture.v1", "manifest": manifest,
        "video_path": "capture/camera.mp4", "metric_vio_allowed": False,
        "calibration": {}, "coverage": {}, "import_report_path": "capture/capture_import.json",
        "video_timestamps_path": "capture/video_timestamps_ns.json",
        "imu_normalized_path": "capture/imu_normalized.json",
    }


def _app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, importer: Any = _fake_import) -> Any:
    monkeypatch.setattr(phone_app, "import_capture_bundle", importer)
    app = phone_app.create_app(
        _settings(tmp_path), frame_processor=lambda *_: {"frame_count": 2, "frames": []},
    )
    app.state.phone_scan_service.companion_capture = _FakeCompanionManager()
    return app


def _wait(client: TestClient, scan_id: str, expected: str) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = client.get(f"/api/scans/{scan_id}").json()
        if state.get("status") == expected:
            return state
        time.sleep(0.01)
    raise AssertionError(state)


def test_receipt_precedes_import_and_paired_retry_keeps_exact_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started, release = threading.Event(), threading.Event()
    calls = []

    def blocked(path: Path, scan_dir: Path, *, limits: capture.CaptureImportLimits) -> dict[str, Any]:
        calls.append(limits.video_probe_timeout_s)
        started.set()
        assert release.wait(8)
        return _fake_import(path, scan_dir, limits=limits)

    app = _app(tmp_path, monkeypatch, blocked)
    service = app.state.phone_scan_service
    archive = _archive(paired=True)
    try:
        with TestClient(app) as client:
            response = client.post("/api/scans/sensor-bundle", content=archive, headers=_headers(paired=True))
            assert response.status_code == 202, response.text
            state = response.json()
            assert state["id"] == SCAN_ID and state["status"] == "importing_capture"
            assert state["validation_status"] == "pending"
            receipt = state["upload_receipt"]
            assert receipt.items() >= {
                "schema": UPLOAD_RECEIPT_SCHEMA, "status": "stored", "size_bytes": len(archive),
                "sha256": hashlib.sha256(archive).hexdigest(), "capture_id": CAPTURE_ID,
                "companion_session_id": SESSION_ID, "companion_camera_id": CAMERA_ID,
            }.items()
            assert receipt["receive_elapsed_ms"] > 0 and receipt["receive_mbps"] > 0
            assert (tmp_path / "scans" / SCAN_ID / "capture_upload.archive").read_bytes() == archive
            assert started.wait(2)
            assert service.companion_capture.phone["archive_sha256"] is None
            assert client.delete(f"/api/scans/{SCAN_ID}").status_code == 409
            duplicate = client.post("/api/scans/sensor-bundle", content=archive, headers=_headers(paired=True))
            assert duplicate.status_code == 202 and duplicate.json()["id"] == SCAN_ID
            assert duplicate.json()["upload_receipt"] == receipt
            changed = client.post("/api/scans/sensor-bundle", content=_archive(paired=True, tag="changed"), headers=_headers(paired=True))
            assert changed.status_code == 409
            legacy = client.post("/api/scans/sensor-bundle", content=archive, headers=_headers(paired=True, asynchronous=False))
            assert legacy.status_code == 409
            assert (tmp_path / "scans" / SCAN_ID / "capture_upload.archive").read_bytes() == archive
            assert calls == [ASYNC_VIDEO_PROBE_TIMEOUT_S]
            release.set()
            done = _wait(client, SCAN_ID, "ready")
            assert done["validation_status"] == "complete"
            assert service.companion_capture.phone["archive_sha256"] == hashlib.sha256(archive).hexdigest()
            duplicate = client.post("/api/scans/sensor-bundle", content=archive, headers=_headers(paired=True))
            assert duplicate.status_code == 200
            assert len(calls) == 1
    finally:
        release.set()


def test_import_queue_has_one_worker_and_two_pending_slots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started, release = threading.Event(), threading.Event()
    calls = []

    def blocked(path: Path, scan_dir: Path, *, limits: Any) -> dict[str, Any]:
        calls.append(scan_dir.name)
        started.set()
        assert release.wait(8)
        return _fake_import(path, scan_dir, limits=limits)

    app = _app(tmp_path, monkeypatch, blocked)
    try:
        with TestClient(app) as client:
            responses = [client.post("/api/scans/sensor-bundle", content=_archive(f"capture-{i}"), headers=_headers()) for i in range(4)]
            assert [r.status_code for r in responses] == [202, 202, 202, 503]
            assert responses[-1].headers["retry-after"] == "30"
            assert started.wait(2) and len(calls) == 1
            assert len(list((tmp_path / "scans").glob("20*/capture_upload.archive"))) == 3
            release.set()
            for response in responses[:3]:
                _wait(client, response.json()["id"], "ready")
            assert len(calls) == 3
    finally:
        release.set()


def test_failed_import_retains_archive_and_same_upload_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = []

    def fail_once(path: Path, scan_dir: Path, *, limits: Any) -> dict[str, Any]:
        attempts.append(path.read_bytes())
        if len(attempts) == 1:
            (scan_dir / ".capture-importing").mkdir()
            raise capture.CaptureImportError("invalid decoded timing evidence")
        assert not (scan_dir / ".capture-importing").exists()
        return _fake_import(path, scan_dir, limits=limits)

    app = _app(tmp_path, monkeypatch, fail_once)
    archive = _archive()
    with TestClient(app) as client:
        first = client.post("/api/scans/sensor-bundle", content=archive, headers=_headers())
        scan_id = first.json()["id"]
        failed = _wait(client, scan_id, "import_failed")
        assert "invalid decoded timing" in failed["error"]
        assert (tmp_path / "scans" / scan_id / "capture_upload.archive").read_bytes() == archive
        changed = client.post("/api/scans/sensor-bundle", content=_archive(tag="changed"), headers=_headers())
        assert changed.status_code == 409
        retry = client.post("/api/scans/sensor-bundle", content=archive, headers=_headers())
        assert retry.status_code == 202 and retry.json()["id"] == scan_id
        _wait(client, scan_id, "ready")
        assert attempts == [archive, archive]
        assert client.delete(f"/api/scans/{scan_id}").status_code == 204
        assert app.state.phone_scan_service.capture_uploads.lookup(CAPTURE_ID, None) is None
        new = client.post("/api/scans/sensor-bundle", content=archive, headers=_headers())
        assert new.status_code == 202 and new.json()["id"] != scan_id
        _wait(client, new.json()["id"], "ready")


def test_restart_marks_import_interrupted_and_exact_retry_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_app = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(first_app.state.phone_scan_service.capture_uploads, "submit", lambda *_: None)
    archive = _archive()
    with TestClient(first_app) as client:
        first = client.post("/api/scans/sensor-bundle", content=archive, headers=_headers())
        assert first.status_code == 202
        scan_id = first.json()["id"]
    second_app = _app(tmp_path, monkeypatch)
    with TestClient(second_app) as client:
        interrupted = client.get(f"/api/scans/{scan_id}").json()
        assert interrupted["status"] == "import_failed"
        assert interrupted["upload_import"]["phase"] == "interrupted"
        assert (tmp_path / "scans" / scan_id / "capture_upload.archive").read_bytes() == archive
        retry = client.post("/api/scans/sensor-bundle", content=archive, headers=_headers())
        assert retry.status_code == 202 and retry.json()["id"] == scan_id
        _wait(client, scan_id, "ready")


def test_request_cancellation_during_durable_acceptance_does_not_cancel_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path, monkeypatch)
    queue = app.state.phone_scan_service.capture_uploads
    real_persist = queue.persist
    persist_started, release = threading.Event(), threading.Event()

    def blocked_persist(*args: Any, **kwargs: Any) -> Path:
        persist_started.set()
        assert release.wait(8)
        return real_persist(*args, **kwargs)

    monkeypatch.setattr(queue, "persist", blocked_persist)

    async def run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            request = asyncio.create_task(client.post("/api/scans/sensor-bundle", content=_archive(), headers=_headers()))
            assert await asyncio.to_thread(persist_started.wait, 3)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            release.set()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                scan_id = queue.lookup(CAPTURE_ID, None)
                if scan_id:
                    result = await client.get(f"/api/scans/{scan_id}")
                    if result.json().get("status") == "ready":
                        assert (tmp_path / "scans" / scan_id / "capture_upload.archive").is_file()
                        return
                await asyncio.sleep(0.01)
            raise AssertionError("Cancelled HTTP request lost its admitted background import")

    try:
        asyncio.run(run())
    finally:
        release.set()
        app.state.phone_scan_service.shutdown()


def test_failed_durable_flush_never_acknowledges_or_starts_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from . import capture_upload

    imported = threading.Event()

    def importer(*args: Any, **kwargs: Any) -> dict[str, Any]:
        imported.set()
        return _fake_import(*args, **kwargs)

    app = _app(tmp_path, monkeypatch, importer)
    real_sync = capture_upload.sync_file

    def reject_archive_flush(path: Path) -> None:
        if path.suffix == ".uploading":
            raise OSError("disk flush failed")
        real_sync(path)

    monkeypatch.setattr(capture_upload, "sync_file", reject_archive_flush)
    with TestClient(app, raise_server_exceptions=False) as client:
        failed = client.post("/api/scans/sensor-bundle", content=_archive(), headers=_headers())
        assert failed.status_code == 500
        assert not imported.is_set()
        assert not list((tmp_path / "scans").glob("*/capture_upload.archive"))
        assert not app.state.phone_scan_service.capture_uploads._admitted
        assert not list((tmp_path / "scans").glob(".companion-phone-*.uploading"))


def test_async_native_real_import_preserves_timing_and_probe_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encoded_video: Path) -> None:
    timeouts = []
    original_run = subprocess.run

    def spy(command: Any, *args: Any, **kwargs: Any) -> Any:
        if command[0] == "ffprobe" and any("frame=best_effort_timestamp" in item for item in command):
            timeouts.append(kwargs.get("timeout"))
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(capture.subprocess, "run", spy)
    app = _app(tmp_path, monkeypatch, capture.import_capture_bundle)
    app.state.phone_scan_service.frame_processor = phone_app.prepare_video_frames
    with TestClient(app) as client:
        response = client.post("/api/scans/sensor-bundle", content=_bundle(encoded_video), headers=_headers())
        assert response.status_code == 202, response.text
        state = _wait(client, response.json()["id"], "ready")
        assert state["capture"]["camera_acquisition_timestamp_verified"] is True
        assert state["capture"]["metric_vio_allowed"] is False
        assert len(state["prepared"]["frames"]) >= 2
        assert all(
            frame["capture_time_ns"] == FRAME_TIMES[frame["source_frame_index"]]
            for frame in state["prepared"]["frames"]
        )
        assert (tmp_path / "scans" / state["id"] / "capture/camera.mp4").read_bytes() == encoded_video.read_bytes()
        assert state["upload_receipt"]["validation_status"] == "complete"
        assert timeouts == [ASYNC_VIDEO_PROBE_TIMEOUT_S]
        sync = client.post("/api/scans/sensor-bundle", content=_bundle(encoded_video), headers=_headers(asynchronous=False))
        assert sync.status_code == 201, sync.text
        _wait(client, sync.json()["id"], "ready")
        assert timeouts == [ASYNC_VIDEO_PROBE_TIMEOUT_S, 120]
