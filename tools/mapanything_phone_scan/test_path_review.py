from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

from fastapi.testclient import TestClient
import pytest

from . import app as app_module
from .path_review import build_path_review
from .test_browser_capture import _settings
from .test_trajectory_motion_review import capture
from .test_walk_modes import SOURCE, TARGET, OTHER, intent, state


def retained(capture):
    outputs = capture["scan"] / "outputs"
    outputs.mkdir()
    for filename in ("scan_outputs_manifest.json", "camera_trajectory.json"):
        shutil.copyfile(capture["provider"] / filename, outputs / filename)
    return {"id": SOURCE, "status": "complete", "name": "retained native walk",
            "outputs": {"artifacts": {"manifest": "outputs/scan_outputs_manifest.json"}},
            "capture": {"capture_kind": "android_camera_imu", "raw_streams_preserved": True,
                        "import_report": "capture/capture_import.json", "metric_vio_allowed": False},
            "companion_capture": {"session_id": "retained-session", "camera_id": "living-room",
                                  "artifacts": {"tracking": "tracking.ndjson"}}}


def evidence(capture):
    return {name: hashlib.sha256((capture["native"] / name).read_bytes()).hexdigest()
            for name in ("accel.csv", "gyro.csv", "capture_import.json", "timestamps.csv", "camera_results.jsonl")}


def test_retained_native_walk_exports_poses_and_motion_without_false_body_accuracy(capture):
    source = retained(capture)
    before = evidence(capture)
    original_state = (capture["scan"] / "scan_state.json").read_bytes()
    output = capture["tmp"] / "path-review"
    report = build_path_review(capture["scan"], source, capture["scan"], source, output)
    assert report["visual_path"]["status"] == "available"
    assert report["visual_path"]["pose_count"] == 40
    assert report["imu_consistency"]["status"] == "review_complete"
    assert report["imu_consistency"]["rotation"]["all_supported"]["n"] == 38
    assert report["accuracy"]["qualified"] is False
    assert report["accuracy"]["target_m"] == 0.1
    assert report["accuracy"]["measured_position_error_m"] is None
    assert report["body_ground_reference"]["status"] == "not_established"
    assert report["capture_intent"] is None
    assert report["paired_noesis"]["clocks_joined"] is False
    trajectory = json.loads((output / "camera_trajectory.json").read_text())
    assert trajectory["poses"][1]["capture_time_ns"] == str(capture["origin"] + 250_000_000)
    assert trajectory["interpolated"] is False
    assert report["imu_consistency"]["position_refinement"] is False
    assert evidence(capture) == before
    assert (capture["scan"] / "scan_state.json").read_bytes() == original_state


def test_recorded_close_body_protocol_does_not_establish_anatomical_offset(capture):
    source = retained(capture)
    source["walk_intent"] = intent("path_refinement", TARGET)
    source["walk_intent_source"] = "capture_manifest"
    target = {**deepcopy(source), "id": TARGET}
    result = build_path_review(capture["scan"], source, capture["scan"], target, capture["tmp"] / "review")
    assert result["body_ground_reference"]["carry_protocol_declared"] == "close_body"
    assert result["body_ground_reference"]["phone_to_body_offset_measured"] is False
    assert result["reference_reconstruction"]["manifest_sha256"]


def test_legacy_missing_pose_semantics_are_reported_not_invented(capture):
    source = retained(capture)
    manifest = capture["scan"] / "outputs/scan_outputs_manifest.json"
    value = json.loads(manifest.read_text())
    value.pop("coordinate_frame")
    manifest.write_text(json.dumps(value))
    before = manifest.read_bytes()
    result = build_path_review(capture["scan"], source, capture["scan"], source, capture["tmp"] / "review")
    assert result["visual_path"]["status"] == "needs_evidence"
    assert result["accuracy"]["qualified"] is False
    assert manifest.read_bytes() == before


def test_missing_sensor_evidence_does_not_discard_visual_path(capture):
    source = retained(capture)
    (capture["native"] / "gyro.csv").unlink()
    result = build_path_review(capture["scan"], source, capture["scan"], source, capture["tmp"] / "review")
    assert result["visual_path"]["status"] == "available"
    assert result["imu_consistency"]["status"] == "needs_evidence"
    assert result["sensor_refined_path"]["status"] == "not_available"


def test_wrong_target_and_existing_outputs_fail_without_overwrite(capture):
    source = retained(capture)
    source["walk_intent"] = intent("path_refinement", TARGET)
    output = capture["tmp"] / "review"
    with pytest.raises(ValueError, match="disagree"):
        build_path_review(capture["scan"], source, capture["scan"], source, output)
    assert not output.exists()
    output.mkdir()
    marker = output / "retained.json"
    marker.write_bytes(b"original")
    with pytest.raises(ValueError, match="new path-review"):
        build_path_review(capture["scan"], source, capture["scan"], {**source, "id": TARGET}, output)
    assert marker.read_bytes() == b"original"


def test_path_review_api_exports_artifacts_and_preserves_raw_scan(capture, monkeypatch):
    source = retained(capture)
    application = app_module.create_app(_settings(capture["tmp"] / "app"))
    service = application.state.phone_scan_service
    root = service.scan_dir(SOURCE)
    # Link/copy fixture-only inputs into the app's isolated test storage.
    shutil.copytree(capture["scan"], root)
    service._write_state_unlocked(SOURCE, source)
    with TestClient(application) as client:
        response = client.post(f"/api/scans/{SOURCE}/path-review")
        assert response.status_code == 202, response.text
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = client.get(f"/api/scans/{SOURCE}").json()
            if result["path_review"]["status"] in {"complete", "failed"}:
                break
            time.sleep(0.01)
        assert result["path_review"]["status"] == "complete", result
        link = result["path_review"]["results"]["artifact_url"]
        download = client.get(link)
        assert download.status_code == 200
        assert download.json()["accuracy"]["qualified"] is False
        assert result["outputs"] == {**source["outputs"], "artifact_urls": {"manifest": f"/assets/{SOURCE}/outputs/scan_outputs_manifest.json"}}
        assert result["capture"]["metric_vio_allowed"] is False


def test_review_slots_are_bounded_and_invalid_requests_release_capacity(tmp_path, monkeypatch):
    application = app_module.create_app(_settings(tmp_path))
    service = application.state.phone_scan_service
    for scan_id in (SOURCE, TARGET, OTHER):
        state(service, scan_id)
    submitted = []
    monkeypatch.setattr(service._executor, "submit", lambda *args: submitted.append(args))
    with TestClient(application) as client:
        assert client.post(f"/api/scans/{SOURCE}/path-review").status_code == 202
        assert client.post(f"/api/scans/{SOURCE}/path-review").status_code == 409
        assert client.post(f"/api/scans/{TARGET}/path-review").status_code == 202
        assert client.post(f"/api/scans/{OTHER}/path-review").status_code == 503
        assert len(submitted) == 2


def test_failed_review_submission_is_retryable_and_releases_capacity(tmp_path, monkeypatch):
    application = app_module.create_app(_settings(tmp_path))
    service = application.state.phone_scan_service
    state(service, SOURCE)
    def reject(*args):
        raise RuntimeError("executor is unavailable")
    monkeypatch.setattr(service._executor, "submit", reject)
    with TestClient(application) as client:
        for _ in range(2):
            assert client.post(f"/api/scans/{SOURCE}/path-review").status_code == 503
            assert service.read_state(SOURCE)["path_review"]["status"] == "failed"
        assert service._path_review_slots.acquire(blocking=False)
        assert service._path_review_slots.acquire(blocking=False)
        assert not service._path_review_slots.acquire(blocking=False)
        service._path_review_slots.release()
        service._path_review_slots.release()
