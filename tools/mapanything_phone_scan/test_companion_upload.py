from __future__ import annotations

import copy
import io
import json
import threading
import time
import tarfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from . import app as phone_app
from .browser_capture import BROWSER_CAPTURE_SCHEMA
from .companion_capture import CompanionCaptureConflict
from .test_browser_capture import _settings


SESSION_ID = "companion-20260905-010101-deadbeef"
CAPTURE_ID = "browser-upload-test"
SCAN_ID = "20260905-010101-deadbeef"
CAMERA_ID = "family-room"


class _FakeCompanionManager:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.camera_ids: list[str] | None = None
        self.phone: dict[str, Any] = {
            "capture_id": None,
            "scan_id": None,
            "archive_sha256": None,
        }

    def public_state(self, session_id: str) -> dict[str, Any]:
        self.calls.append(("public_state", session_id))
        assert session_id == SESSION_ID
        return {
            "session_id": SESSION_ID,
            "camera_id": CAMERA_ID,
            "phone_capture_id": self.phone.get("capture_id"),
            "phone": copy.deepcopy(self.phone),
            "status": "recording",
            **({"camera_ids": self.camera_ids} if self.camera_ids is not None else {}),
        }

    def check_phone_archive(
        self, session_id: str, capture_id: str, archive_sha256: str
    ) -> dict[str, Any] | None:
        self.calls.append(("check_phone_archive", session_id, capture_id, archive_sha256))
        assert session_id == SESSION_ID
        bound = self.phone.get("capture_id")
        if bound and bound != capture_id:
            raise CompanionCaptureConflict("phone capture ID conflicts with this companion session")
        prior = self.phone.get("archive_sha256")
        if prior and prior != archive_sha256:
            raise CompanionCaptureConflict("phone archive conflicts with the already associated archive")
        if prior == archive_sha256 and self.phone.get("scan_id"):
            return self.public_state(session_id)
        return None

    def reserve_phone_scan_id(self, session_id: str, capture_id: str) -> str:
        self.calls.append(("reserve_phone_scan_id", session_id, capture_id))
        assert session_id == SESSION_ID
        if self.phone.get("capture_id") not in {None, capture_id}:
            raise CompanionCaptureConflict("companion session is already bound to another phone capture")
        self.phone["capture_id"] = capture_id
        self.phone["scan_id"] = SCAN_ID
        return SCAN_ID

    def associate_phone_bundle(
        self,
        session_id: str,
        *,
        capture_id: str,
        archive_sha256: str,
        scan_id: str,
    ) -> dict[str, Any]:
        self.calls.append(("associate_phone_bundle", session_id, capture_id, archive_sha256, scan_id))
        assert session_id == SESSION_ID
        if self.phone.get("archive_sha256") not in {None, archive_sha256}:
            raise CompanionCaptureConflict("phone archive conflicts with the already associated archive")
        self.phone.update(
            {
                "capture_id": capture_id,
                "scan_id": scan_id,
                "archive_sha256": archive_sha256,
            }
        )
        return self.public_state(session_id)

    def record_upload_failure(
        self, session_id: str, capture_id: str, temporary_archive: Path, error: str
    ) -> dict[str, Any]:
        self.calls.append(("record_upload_failure", session_id, capture_id, error))
        temporary_archive.unlink(missing_ok=True)
        return self.public_state(session_id)

    def shutdown(self) -> None:
        return None


def _manifest(*, archive_tag: str = "same") -> dict[str, Any]:
    return {
        "schema": BROWSER_CAPTURE_SCHEMA,
        "capture_id": CAPTURE_ID,
        "video": {"path": "phone_walk.webm", "mime_type": "video/webm"},
        "device": {
            "id": f"browser-session:{CAPTURE_ID}",
            "model": "Test browser",
            "user_agent": "Mozilla/5.0 test",
        },
        "camera": {"settings": {}, "capabilities": {}},
        "timing": {"time_origin_ms": 1, "started_ms": 2, "stopped_ms": 3},
        "sensors": {
            "accelerometer": {"samples": []},
            "gyroscope": {"samples": []},
        },
        "video_frames": [],
        "events": [{"archive_tag": archive_tag}],
        "stop_reason": "user",
        "companion_capture": {
            "session_id": SESSION_ID,
            "camera_id": CAMERA_ID,
            "phone_capture_id": CAPTURE_ID,
        },
    }


def _archive(manifest: dict[str, Any], *, archive_tag: str) -> bytes:
    path = io.BytesIO()
    with tarfile.open(fileobj=path, mode="w") as archive:
        manifest_bytes = json.dumps(manifest).encode("utf-8")
        info = tarfile.TarInfo("capture_manifest.json")
        info.size = len(manifest_bytes)
        archive.addfile(info, io.BytesIO(manifest_bytes))
        video = (f"video-{archive_tag}".encode("ascii"))
        video_info = tarfile.TarInfo("phone_walk.webm")
        video_info.size = len(video)
        archive.addfile(video_info, io.BytesIO(video))
    return path.getvalue()


def _install_test_import(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[Path],
    limits: Any,
) -> None:
    def fake_import(
        archive_path: Path,
        scan_dir: Path,
        *,
        limits: Any,
    ) -> dict[str, Any]:
        calls.append(archive_path)
        manifest = phone_app._read_sensor_bundle_manifest(
            archive_path,
            limits=limits,
        )
        assert manifest is not None
        capture_dir = scan_dir / "capture"
        capture_dir.mkdir()
        (capture_dir / "phone_walk.webm").write_bytes(b"encoded-video")
        return {
            "schema": BROWSER_CAPTURE_SCHEMA,
            "manifest": manifest,
            "video_path": "capture/phone_walk.webm",
            "manifest_path": "capture/capture_manifest.json",
            "sensor_samples_path": "capture/browser_sensor_samples.json",
            "import_report_path": "capture/capture_import.json",
            "calibration": {"complete_for_metric_vio": False},
            "timing": {},
            "sensors": {},
            "video": {},
            "interruptions": [],
            "stop_reason": "user",
            "coverage": {},
        }

    monkeypatch.setattr(phone_app, "import_capture_bundle", fake_import)


def _test_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release: threading.Event,
    started: threading.Event,
) -> tuple[Any, _FakeCompanionManager, list[Path]]:
    import_calls: list[Path] = []
    settings = _settings(tmp_path)
    _install_test_import(monkeypatch, import_calls, settings.capture_limits)

    def fake_prepare(*_: Any) -> dict[str, Any]:
        started.set()
        assert release.wait(5.0)
        return {"frame_count": 2, "frames": [], "contact_sheet": "", "manifest": ""}

    app = phone_app.create_app(settings, frame_processor=fake_prepare)
    manager = _FakeCompanionManager()
    app.state.phone_scan_service.companion_capture = manager
    return app, manager, import_calls


def _headers() -> dict[str, str]:
    return {
        "Content-Type": "application/x-tar",
        "X-File-Name": "phone-capture.tar",
        "X-Companion-Session": SESSION_ID,
        "X-Phone-Capture-ID": CAPTURE_ID,
    }


def test_same_paired_archive_is_idempotent_while_processing_and_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    started = threading.Event()
    app, manager, import_calls = _test_app(tmp_path, monkeypatch, release, started)
    archive = _archive(_manifest(), archive_tag="same")

    with TestClient(app) as client:
        first = client.post("/api/scans/sensor-bundle", content=archive, headers=_headers())
        assert first.status_code == 201, first.text
        scan_id = first.json()["id"]
        assert scan_id == SCAN_ID
        assert started.wait(2.0)
        scan_dir = _settings(tmp_path).storage_root / scan_id
        assert scan_dir.is_dir()

        duplicate_processing = client.post(
            "/api/scans/sensor-bundle", content=archive, headers=_headers()
        )
        assert duplicate_processing.status_code == 200, duplicate_processing.text
        assert duplicate_processing.json()["id"] == scan_id
        assert len(import_calls) == 1
        assert scan_dir.is_dir()

        release.set()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            state = client.get(f"/api/scans/{scan_id}").json()
            if state["status"] == "ready":
                break
            time.sleep(0.02)
        assert state["status"] == "ready"
        duplicate_ready = client.post(
            "/api/scans/sensor-bundle", content=archive, headers=_headers()
        )
        assert duplicate_ready.status_code == 200, duplicate_ready.text
        assert duplicate_ready.json()["id"] == scan_id
        assert len(import_calls) == 1
        assert sum(call[0] == "associate_phone_bundle" for call in manager.calls) == 1


def test_different_paired_archive_conflicts_without_touching_successful_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    started = threading.Event()
    app, manager, import_calls = _test_app(tmp_path, monkeypatch, release, started)
    archive = _archive(_manifest(), archive_tag="same")
    mismatch = _archive(_manifest(archive_tag="different"), archive_tag="different")

    with TestClient(app) as client:
        first = client.post("/api/scans/sensor-bundle", content=archive, headers=_headers())
        assert first.status_code == 201, first.text
        scan_id = first.json()["id"]
        assert started.wait(2.0)
        before = client.get(f"/api/scans/{scan_id}").json()
        conflict = client.post(
            "/api/scans/sensor-bundle", content=mismatch, headers=_headers()
        )
        assert conflict.status_code == 409, conflict.text
        after = client.get(f"/api/scans/{scan_id}").json()
        assert after["id"] == before["id"] == scan_id
        assert after["status"] == before["status"] == "processing_frames"
        assert (tmp_path / "scans" / scan_id).is_dir()
        assert len(import_calls) == 1
        assert manager.phone["archive_sha256"]
        assert not any(call[0] == "record_upload_failure" for call in manager.calls)
        release.set()


def test_manual_tar_derives_companion_reference_and_associates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    started = threading.Event()
    app, manager, _ = _test_app(tmp_path, monkeypatch, release, started)
    archive = _archive(_manifest(), archive_tag="manual")

    with TestClient(app) as client:
        response = client.post(
            "/api/scans/sensor-bundle",
            content=archive,
            headers={
                "Content-Type": "application/x-tar",
                "X-File-Name": "manual-retry.tar",
            },
        )
        assert response.status_code == 201, response.text
        assert response.json()["id"] == SCAN_ID
        assert started.wait(2.0)
        assert manager.phone["capture_id"] == CAPTURE_ID
        assert manager.phone["scan_id"] == SCAN_ID
        assert any(call[0] == "reserve_phone_scan_id" for call in manager.calls)
        assert any(call[0] == "associate_phone_bundle" for call in manager.calls)
        release.set()


def test_header_manifest_session_mismatch_conflicts_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    started = threading.Event()
    app, manager, import_calls = _test_app(tmp_path, monkeypatch, release, started)
    archive = _archive(_manifest(), archive_tag="mismatch")
    headers = _headers()
    headers["X-Companion-Session"] = "companion-20260905-010101-facefeed"

    with TestClient(app) as client:
        response = client.post(
            "/api/scans/sensor-bundle", content=archive, headers=headers
        )
        assert response.status_code == 409, response.text
        assert "session" in response.json()["detail"].lower()
        assert import_calls == []
        assert manager.calls == []
        assert list((tmp_path / "scans").glob("20*")) == []
        release.set()


@pytest.mark.parametrize("declared", [None, [CAMERA_ID], [CAMERA_ID, "living-room"]])
def test_pair_upload_rejects_missing_or_wrong_camera_set_before_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: list[str] | None,
) -> None:
    release, started = threading.Event(), threading.Event()
    app, manager, imports = _test_app(tmp_path, monkeypatch, release, started)
    manager.camera_ids = [CAMERA_ID, "kitchen"]
    manifest = _manifest()
    if declared is not None:
        manifest["companion_capture"]["camera_ids"] = declared
    with TestClient(app) as client:
        response = client.post("/api/scans/sensor-bundle", content=_archive(manifest, archive_tag="pair"), headers=_headers())
        assert response.status_code == 409, response.text
        assert imports == []
        assert not any(call[0] == "reserve_phone_scan_id" for call in manager.calls)
    release.set()


def test_pair_upload_retains_exact_set_and_idempotent_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, started = threading.Event(), threading.Event()
    app, manager, imports = _test_app(tmp_path, monkeypatch, release, started)
    manager.camera_ids = [CAMERA_ID, "kitchen"]
    manifest = _manifest()
    manifest["companion_capture"]["camera_ids"] = ["kitchen", CAMERA_ID]
    archive = _archive(manifest, archive_tag="pair")
    headers = {**_headers(), "X-Companion-Camera-Ids": json.dumps(manager.camera_ids)}
    try:
        with TestClient(app) as client:
            response = client.post("/api/scans/sensor-bundle", content=archive, headers=headers)
            assert response.status_code == 201, response.text
            assert sorted(response.json()["companion_capture"]["camera_ids"]) == sorted(manager.camera_ids)
            duplicate = client.post("/api/scans/sensor-bundle", content=archive, headers=headers)
            assert duplicate.status_code == 200, duplicate.text
            assert len(imports) == 1
            wrong = client.post("/api/scans/sensor-bundle", content=archive, headers={**headers, "X-Companion-Camera-Ids": json.dumps([CAMERA_ID])})
            assert wrong.status_code == 409
            assert len(imports) == 1
            release.set()
    finally:
        release.set()


def test_camera_set_capability_and_start_request_reach_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, started = threading.Event(), threading.Event()
    app, manager, _ = _test_app(tmp_path, monkeypatch, release, started)
    service = app.state.phone_scan_service
    received = []
    monkeypatch.setattr(service, "companion_cameras", lambda: [{"camera_id": CAMERA_ID}, {"camera_id": "kitchen"}])

    def start(camera_id: str, **kwargs: Any) -> dict[str, Any]:
        received.append((camera_id, kwargs))
        return {"session_id": SESSION_ID, "camera_id": camera_id, "camera_ids": kwargs["camera_ids"]}

    monkeypatch.setattr(service, "start_companion_capture", start)
    with TestClient(app) as client:
        capability = client.get("/api/companion-captures/cameras").json()
        assert capability["multi_camera_capture"] is True
        assert capability["max_cameras"] == 2
        response = client.post("/api/companion-captures", json={"camera_id": CAMERA_ID, "camera_ids": [CAMERA_ID, "kitchen"], "phone_capture_id": CAPTURE_ID})
        assert response.status_code == 201
        assert received[0][0] == CAMERA_ID
        assert received[0][1]["camera_ids"] == [CAMERA_ID, "kitchen"]
        assert received[0][1]["phone_capture_id"] == CAPTURE_ID


def test_paired_alignment_selects_retained_member_without_rebinding_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, started = threading.Event(), threading.Event()
    app, manager, _ = _test_app(tmp_path, monkeypatch, release, started)
    service = app.state.phone_scan_service
    manager.camera_ids = [CAMERA_ID, "kitchen"]
    manager.phone.update(capture_id=CAPTURE_ID, scan_id=SCAN_ID, archive_sha256="a" * 64)
    saved = {**manager.public_state(SESSION_ID), "status": "stopped"}
    monkeypatch.setattr(manager, "public_state", lambda _: copy.deepcopy(saved))
    scan = {"id": SCAN_ID, "companion_capture": copy.deepcopy(saved)}
    selected = service._paired_alignment_companion(scan, "kitchen")
    assert selected["camera_id"] == "kitchen"
    assert scan["companion_capture"]["camera_id"] == CAMERA_ID
    saved["camera_ids"] = [CAMERA_ID]
    with pytest.raises(phone_app.HTTPException) as error:
        service._paired_alignment_companion(scan, "kitchen")
    assert error.value.status_code == 409
    service.shutdown()
