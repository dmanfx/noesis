from __future__ import annotations

import json
import threading
import time

from fastapi.testclient import TestClient
import pytest

from .app import create_app
from .calibration_jobs import CalibrationJobs, CalibrationJobBusy, CalibrationJobError, _read, _write
from .roomwalk_calibration import DEFAULT_BOARD, REQUEST_SCHEMA
from .test_browser_capture import _settings
from .test_imu_noise_calibration import noise_fixture


@pytest.mark.parametrize("mode,expected_timeout", [("camera", None), ("imu", 1500)])
def test_isolated_camera_worker_bounds_opencv_and_dense_motion_budget(tmp_path, monkeypatch, mode, expected_timeout):
    from . import calibration_jobs, roomwalk_calibration
    import cv2

    _write(tmp_path / "state.json", {"mode": mode, "_capture_dir": str(tmp_path / "capture"),
                                   "request": {"mode": mode}, "_settings": {}})
    threads = []
    monkeypatch.setattr(cv2, "setNumThreads", threads.append)
    def capture(source, request, output, *, settings, progress, cancelled):
        assert settings.get("timeout_s") == expected_timeout
        output.mkdir()
        return {"status": "completed", "accepted_for_metric_vio": False}
    monkeypatch.setattr(roomwalk_calibration, "run_calibration", capture)
    calibration_jobs._run_job(tmp_path)
    assert threads == [2]
    assert _read(tmp_path / "artifacts/report.json")["accepted_for_metric_vio"] is False


def request():
    return {"schema": REQUEST_SCHEMA, "mode": "camera", "board": DEFAULT_BOARD, "board_geometry_confirmed": False}


@pytest.mark.parametrize("extras,match", [
    ({}, "qualified camera"),
    ({"camera_calibration_id": "camera"}, "printed board"),
])
def test_motion_missing_prerequisites_rejected_before_creating_job(tmp_path, extras, match):
    source = tmp_path / "capture"
    source.mkdir()
    (source / "capture_import.json").write_text("{}")
    jobs = CalibrationJobs(tmp_path)
    try:
        with pytest.raises(CalibrationJobError, match=match):
            jobs.submit("scan", source, {**request(), "mode": "imu", **extras})
        assert jobs.list() == []
        assert (source / "capture_import.json").read_text() == "{}"
    finally:
        jobs.close()


def await_terminal(jobs, job_id, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = jobs.get(job_id)
        if row["status"] in {"completed", "failed", "cancelled"}:
            return row
        time.sleep(0.02)
    raise AssertionError(jobs.get(job_id))


def test_bounded_queue_cancel_and_source_retention(tmp_path):
    source = tmp_path / "capture"
    source.mkdir()
    (source / "capture_import.json").write_text("{}")
    started, release = threading.Event(), threading.Event()
    def execute(directory, cancel):
        started.set()
        release.wait(10)
        (directory / "artifacts").mkdir()
        _write(directory / "artifacts/report.json", {"status": "completed", "accepted_for_metric_vio": False})
    jobs = CalibrationJobs(tmp_path, execute=execute)
    try:
        first = jobs.submit("scan", source, request())
        assert started.wait(2)
        second = jobs.submit("scan", source, request())
        third = jobs.submit("scan", source, request())
        with pytest.raises(CalibrationJobBusy):
            jobs.submit("scan", source, request())
        assert jobs.cancel(second["id"])["status"] == "cancelling"
        jobs.cancel(third["id"])
        release.set()
        assert await_terminal(jobs, first["id"])["status"] == "completed"
        assert await_terminal(jobs, second["id"])["status"] == "cancelled"
        assert await_terminal(jobs, third["id"])["status"] == "cancelled"
        assert (source / "capture_import.json").read_text() == "{}"
        assert not any(key.startswith("_") for key in jobs.get(first["id"]))
    finally:
        release.set()
        jobs.close()


def test_real_noise_subprocess_and_artifact_path_containment(tmp_path):
    root = tmp_path / ".imu-calibration"
    root.mkdir()
    source = noise_fixture(root / "stationary")
    jobs = CalibrationJobs(tmp_path, timeout_s=30)
    try:
        job = jobs.submit_noise("stationary")
        result = await_terminal(jobs, job["id"], timeout=30)
        assert result["status"] == "completed", result
        assert result["result"]["imu_noise_calibrated"] is False
        assert "noise_allan_url" in result["artifacts"]
        assert jobs.artifact(job["id"], "noise_allan.png").read_bytes().startswith(b"\x89PNG")
        with pytest.raises(FileNotFoundError):
            jobs.artifact(job["id"], "../state.json")
        outside = tmp_path / "outside.json"
        outside.write_text("secret")
        (jobs.root / job["id"] / "artifacts/link.json").symlink_to(outside)
        with pytest.raises(FileNotFoundError):
            jobs.artifact(job["id"], "link.json")
        assert (source / "receipt.json").is_file()
        assert len(jobs.list_imu_recordings()) == 1
    finally:
        jobs.close()


def test_restart_marks_unfinished_jobs_without_rewriting_capture(tmp_path):
    jobs = CalibrationJobs(tmp_path)
    directory = jobs.root / "cal-20260914-120000-abcdef012345"
    directory.mkdir()
    _write(directory / "state.json", {"id": directory.name, "status": "running", "created_at": "now"})
    jobs.recover()
    assert jobs.get(directory.name)["status"] == "failed"
    jobs.close()


def test_routes_require_native_evidence_and_bound_request_body(tmp_path):
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.get("/api/calibration/jobs").json() == {"jobs": []}
        assert client.get("/api/calibration/imu-recordings").json() == {"recordings": []}
        assert client.get("/api/calibration/camera-selection").json() == {"selection": None}
        assert client.post("/api/calibration/camera-selection", json={"camera_calibration_id": None}).json() == {"selection": None}
        assert client.post("/api/calibration/camera-selection", json={"camera_calibration_id": "../../escape"}).status_code == 422
        assert client.post("/api/calibration/camera-selection", content=b"x" * 1025, headers={"Content-Type": "application/json"}).status_code == 413
        assert client.post("/api/calibration/jobs", json={"scan_id": "../../escape"}).status_code == 422
        assert client.post("/api/calibration/jobs", content=b"x" * 65537, headers={"Content-Type": "application/json"}).status_code == 413
        assert client.post("/api/calibration/noise-jobs", json={"imu_capture_id": "../../escape"}).status_code == 422
        assert client.post("/api/calibration/noise-jobs", content=b"x" * 1025, headers={"Content-Type": "application/json"}).status_code == 413
        assert client.get("/api/calibration/jobs/invalid").status_code == 404
        service = client.app.state.phone_scan_service
        scan_id = "20260914-120000-abcdef12"
        service.scan_dir(scan_id).mkdir()
        service._write_state_unlocked(scan_id, {"id": scan_id, "status": "ready", "capture": {"capture_kind": "browser_camera_imu"}})
        assert client.post("/api/calibration/jobs", json={**request(), "scan_id": scan_id}).status_code == 422


def test_running_calibration_protects_its_source_from_delete(tmp_path):
    with TestClient(create_app(_settings(tmp_path))) as client:
        service = client.app.state.phone_scan_service
        scan_id = "20260914-120000-abcdef12"
        directory = service.scan_dir(scan_id)
        (directory / "capture").mkdir(parents=True)
        (directory / "capture/capture_import.json").write_text("{}")
        service._write_state_unlocked(scan_id, {"id": scan_id, "status": "calibration_ready", "capture": {"capture_kind": "android_camera_imu"}})
        entered, finish = threading.Event(), threading.Event()
        def execute(directory, cancel):
            entered.set()
            while not finish.is_set() and not cancel.wait(0.05):
                pass
        service.calibration_jobs._execute = execute
        response = client.post("/api/calibration/jobs", json={**request(), "scan_id": scan_id})
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        assert entered.wait(2)
        try:
            assert client.delete(f"/api/scans/{scan_id}").status_code == 409
            assert (directory / "capture/capture_import.json").is_file()
            assert client.post(f"/api/calibration/jobs/{job_id}/cancel").status_code == 200
            assert await_terminal(service.calibration_jobs, job_id)["status"] == "cancelled"
        finally:
            finish.set()
