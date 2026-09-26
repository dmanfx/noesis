from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import time
import zipfile

from fastapi.testclient import TestClient
import pytest

from . import app as app_module
from .capture import CaptureImportError, import_capture_bundle, validate_capture_manifest
from .test_android_capture import _bundle, _manifest, encoded_video
from .test_browser_capture import _settings
from .walk_intent import effective_walk_intent, validate_walk_intent


TARGET = "20260919-120000-11111111"
SOURCE = "20260919-120100-22222222"
OTHER = "20260919-120200-33333333"


def intent(mode="reconstruction", target=None):
    return validate_walk_intent({"mode": mode, "target_scan_id": target})


def state(service, scan_id, *, capture=None, purpose=None):
    root = service.scan_dir(scan_id)
    root.mkdir(parents=True)
    value = {"schema": "noesis.phone_scan.state.v3", "id": scan_id, "name": scan_id,
             "status": "complete", "outputs": {"artifacts": {"manifest": "outputs/scan_outputs_manifest.json"}},
             "capture": capture or {}}
    if purpose is not None:
        value["walk_intent"] = purpose
    service._write_state_unlocked(scan_id, value)
    return value


def with_intent(bundle: bytes, value: dict, sidecar=None) -> bytes:
    result = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(bundle)) as source, zipfile.ZipFile(result, "w") as output:
        for name in source.namelist():
            raw = source.read(name)
            if name == "capture_manifest.json":
                manifest = json.loads(raw)
                manifest["walk_intent"] = value
                raw = json.dumps(manifest).encode()
            output.writestr(name, raw)
        output.writestr("walk_intent.json", json.dumps(value if sidecar is None else sidecar))
    return result.getvalue()


@pytest.mark.parametrize("value", [
    None, [], {"mode": []}, {"mode": {}}, {"mode": "walk"}, {"mode": "path_refinement"},
    {"mode": "reconstruction", "target_scan_id": "../room"},
    {"mode": "reconstruction", "carry_protocol": "close_body"},
    {"mode": "path_refinement", "target_scan_id": TARGET, "carry_protocol": "coverage"},
    {"mode": "reconstruction", "accuracy_target_m": True},
    {"mode": "reconstruction", "accuracy_target_m": 0.2},
    {"mode": "reconstruction", "accuracy_target_m": float("nan")},
    {"mode": "reconstruction", "accuracy_target_m": float("inf")},
    {"mode": "reconstruction", "accuracy_target_m": 10**400},
    {"mode": "reconstruction", "certified_accuracy": True},
])
def test_intent_rejects_invalid_or_accuracy_claims(value):
    with pytest.raises(ValueError):
        validate_walk_intent(value)


def test_legacy_intent_is_not_inferred():
    assert effective_walk_intent({"capture": {"capture_kind": "android_camera_imu"}}) is None
    assert intent()["carry_protocol"] == "coverage"
    assert intent("path_refinement", TARGET)["accuracy_target_m"] == 0.1


def test_large_accuracy_integer_is_a_controlled_http_rejection(tmp_path):
    application = app_module.create_app(_settings(tmp_path))
    service = application.state.phone_scan_service
    state(service, SOURCE)
    with TestClient(application) as client:
        response = client.post(f"/api/scans/{SOURCE}/walk-intent",
                               json={"mode": "reconstruction", "accuracy_target_m": 10**400})
        assert response.status_code == 422
        assert service.read_state(SOURCE).get("walk_intent") is None


def test_capture_normalizer_preserves_mode_without_calibration_admission():
    manifest = _manifest()
    before = validate_capture_manifest(manifest)
    manifest["walk_intent"] = intent("path_refinement", TARGET)
    after = validate_capture_manifest(manifest)
    assert after["walk_intent"] == manifest["walk_intent"]
    assert after["calibration"] == before["calibration"]
    assert after["camera"] == before["camera"]


def test_import_retains_raw_walk_sidecar_and_sensor_bytes(tmp_path, encoded_video):
    bundle = with_intent(_bundle(encoded_video), intent("path_refinement", TARGET))
    archive = tmp_path / "capture.zip"
    archive.write_bytes(bundle)
    root = tmp_path / "scan"
    root.mkdir()
    report = import_capture_bundle(archive, root)
    assert report["manifest"]["walk_intent"]["target_scan_id"] == TARGET
    assert report["metric_vio_allowed"] is False
    with zipfile.ZipFile(io.BytesIO(bundle)) as original:
        for name in ("accel.csv", "gyro.csv", "camera.mp4", "walk_intent.json", "capture_manifest.json"):
            assert (root / "capture" / name).read_bytes() == original.read(name)
    assert archive.read_bytes() == bundle


def test_import_rejects_contradictory_sidecar_without_altering_archive(tmp_path, encoded_video):
    bundle = with_intent(_bundle(encoded_video), intent("path_refinement", TARGET), sidecar=intent())
    archive = tmp_path / "capture.zip"
    archive.write_bytes(bundle)
    root = tmp_path / "scan"
    root.mkdir()
    with pytest.raises(CaptureImportError, match="disagrees"):
        import_capture_bundle(archive, root)
    assert archive.read_bytes() == bundle
    assert not (root / "capture").exists()


def test_explicit_review_declaration_keeps_legacy_raw_evidence(tmp_path):
    application = app_module.create_app(_settings(tmp_path))
    service = application.state.phone_scan_service
    state(service, TARGET)
    state(service, SOURCE)
    raw = service.scan_dir(SOURCE) / "raw-evidence.json"
    raw.write_bytes(b'{"old":"recording"}')
    with TestClient(application) as client:
        response = client.post(f"/api/scans/{SOURCE}/walk-intent", json=intent("path_refinement", TARGET))
        assert response.status_code == 200, response.text
        assert response.json()["walk_intent_source"] == "explicit_review_declaration"
        assert response.json()["capture"] == {}
        assert raw.read_bytes() == b'{"old":"recording"}'
        assert client.post(f"/api/scans/{SOURCE}/walk-intent", json=intent("path_refinement", SOURCE)).status_code == 409


def test_recorded_intent_and_calibration_cannot_be_reclassified(tmp_path):
    application = app_module.create_app(_settings(tmp_path))
    service = application.state.phone_scan_service
    purpose = intent("path_refinement", TARGET)
    state(service, TARGET)
    state(service, SOURCE, capture={"walk_intent": purpose}, purpose=purpose)
    state(service, OTHER, capture={"calibration_request": {"mode": "imu"}})
    with TestClient(application) as client:
        assert client.post(f"/api/scans/{SOURCE}/walk-intent", json=intent()).status_code == 409
        assert client.post(f"/api/scans/{OTHER}/walk-intent", json=intent()).status_code == 409


def test_native_add_views_links_retained_data_and_is_idempotent(tmp_path, monkeypatch):
    application = app_module.create_app(_settings(tmp_path))
    service = application.state.phone_scan_service
    calls = []
    monkeypatch.setattr(service, "submit_supplement_preparation", lambda *args: calls.append(args))
    state(service, TARGET)
    source = state(service, SOURCE, purpose=intent(target=TARGET))
    root = service.scan_dir(SOURCE) / "capture"
    root.mkdir()
    for name, data in (("camera.mp4", b"video"), ("accel.csv", b"accel"), ("gyro.csv", b"gyro"), ("capture_import.json", b"{}")):
        (root / name).write_bytes(data)
    source["video"] = {"path": "capture/camera.mp4"}
    service._write_state_unlocked(SOURCE, source)
    original_state = (service.scan_dir(SOURCE) / "scan_state.json").read_bytes()
    with TestClient(application) as client:
        response = client.post(f"/api/scans/{TARGET}/supplements/from-scan", params={"source_scan_id": SOURCE})
        assert response.status_code == 201, response.text
        added = response.json()["supplements"][0]
        assert added["source_capture"]["raw_streams_preserved"] is True
        retained = service.supplement_dir(TARGET, added["id"]) / "capture"
        for name in ("accel.csv", "gyro.csv", "camera.mp4"):
            assert (retained / name).read_bytes() == (root / name).read_bytes()
            assert (retained / name).stat().st_ino == (root / name).stat().st_ino
        again = client.post(f"/api/scans/{TARGET}/supplements/from-scan", params={"source_scan_id": SOURCE})
        assert len(again.json()["supplements"]) == 1
        assert len(calls) == 1
        assert (service.scan_dir(SOURCE) / "scan_state.json").read_bytes() == original_state


def test_path_capture_cannot_become_added_room_geometry(tmp_path):
    application = app_module.create_app(_settings(tmp_path))
    service = application.state.phone_scan_service
    state(service, TARGET)
    state(service, SOURCE, purpose=intent("path_refinement", TARGET))
    with TestClient(application) as client:
        response = client.post(f"/api/scans/{TARGET}/supplements/from-scan", params={"source_scan_id": SOURCE})
        assert response.status_code == 409
        assert not (service.scan_dir(TARGET) / "supplements").exists()


def test_import_api_keeps_capture_mode_in_public_state(tmp_path, encoded_video):
    application = app_module.create_app(_settings(tmp_path), frame_processor=lambda *_: {"frame_count": 2, "frames": []})
    with TestClient(application) as client:
        response = client.post("/api/scans/sensor-bundle", content=with_intent(_bundle(encoded_video), intent("path_refinement", TARGET)),
                               headers={"Content-Type": "application/zip", "X-File-Name": "walk.zip"})
        assert response.status_code == 201, response.text
        result = response.json()
        assert result["walk_intent"] == result["capture"]["walk_intent"] == intent("path_refinement", TARGET)
        assert result["walk_intent_source"] == "capture_manifest"
        # Target may be unavailable after import/transfer. Keep the capture; a
        # later review, not ingestion, checks the reference reconstruction.
        assert result["capture"]["metric_vio_allowed"] is False


def test_new_mode_does_not_apply_global_uploaded_video_calibration(tmp_path):
    calls = []
    application = app_module.create_app(_settings(tmp_path), frame_processor=lambda *args: calls.append(args[2]) or {"frame_count": 2})
    service = application.state.phone_scan_service
    value = state(service, SOURCE, purpose=intent())
    (service.scan_dir(SOURCE) / "camera.mp4").write_bytes(b"fixture")
    value.update(status="processing_frames", video={"path": "camera.mp4"})
    service._write_state_unlocked(SOURCE, value)
    service._prepare_worker(SOURCE)
    assert calls[0].phone_camera_calibration is None
    assert calls[0].phone_camera_capture_mode == "unbound"
    service.shutdown()


def test_path_walk_cannot_receive_geometry_supplements(tmp_path):
    application = app_module.create_app(_settings(tmp_path))
    service = application.state.phone_scan_service
    state(service, SOURCE, purpose=intent("path_refinement", TARGET))
    with pytest.raises(app_module.HTTPException) as rejected:
        service.add_supplement(SOURCE, {"id": "extra", "status": "processing_frames"})
    assert rejected.value.status_code == 409
    assert not service.read_state(SOURCE).get("supplements")
    service.shutdown()
