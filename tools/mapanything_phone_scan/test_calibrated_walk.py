import copy
import hashlib

import pytest

from .calibrated_walk import CameraSelection
from .calibration_jobs import CalibrationJobs, CalibrationJobError, _write
from .phone_calibration import load_phone_calibration
from .processing import FramePreparationSettings
from .roomwalk_calibration import camera_profile, fit_camera_observations, REPORT_SCHEMA
from .test_roomwalk_calibration import observations, request, binding


def fixture(tmp_path):
    jobs = CalibrationJobs(tmp_path)
    job_id = "cal-20260914-120000-abcdef012345"
    directory = jobs.root / job_id
    (directory / "artifacts").mkdir(parents=True)
    result = fit_camera_observations(observations()[0], request(), binding())
    assert result["quality"]["status"] == "qualified"
    _write(directory / "state.json", {"id": job_id, "mode": "camera", "status": "completed", "result": {"camera_intrinsics_calibrated": True}, "created_at": "now"})
    _write(directory / "artifacts/camera_result.json", result)
    _write(directory / "artifacts/camera_profile.json", camera_profile(result))
    data = (directory / "artifacts/camera_result.json").read_bytes()
    _write(directory / "artifacts/report.json", {"schema": REPORT_SCHEMA, "mode": "camera", "status": "completed", "artifacts": {"camera_result.json": {"path": "camera_result.json", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}})
    return jobs, job_id, result


def test_select_bind_and_rectification_profile_without_vio_admission(tmp_path, monkeypatch):
    jobs, job_id, result = fixture(tmp_path)
    try:
        selection = CameraSelection(jobs)
        assert selection.get() is None
        chosen = selection.select(job_id)
        assert selection.select(job_id)["profile_id"] == chosen["profile_id"]
        monkeypatch.setattr("tools.mapanything_phone_scan.calibrated_walk.camera_binding_for_capture", lambda _: result["binding"])
        settings, status = selection.frame_settings(tmp_path, chosen, FramePreparationSettings())
        assert status["status"] == "applied"
        profile = load_phone_calibration(settings.phone_camera_calibration)
        assert profile["D"] == result["D"] and len(profile["D"]) == 5
        assert profile["metric_vio_allowed"] is False
        assert settings.phone_camera_capture_mode == "native_sensor_bundle"
        assert selection.select(None) is None and selection.get() is None
        # A later user choice cannot change the immutable selection of an import.
        assert selection.frame_settings(tmp_path, chosen, FramePreparationSettings())[1]["status"] == "applied"
    finally:
        jobs.close()


def test_binding_mismatch_disables_selected_and_global_profiles(tmp_path, monkeypatch):
    jobs, job_id, result = fixture(tmp_path)
    try:
        selection = CameraSelection(jobs)
        chosen = selection.select(job_id)
        other = copy.deepcopy(result["binding"])
        other["sha256"] = "different-lens"
        monkeypatch.setattr("tools.mapanything_phone_scan.calibrated_walk.camera_binding_for_capture", lambda _: other)
        settings, status = selection.frame_settings(tmp_path, chosen, FramePreparationSettings(phone_camera_calibration=tmp_path / "unrelated.json"))
        assert settings.phone_camera_calibration is None
        assert status["status"] == "not_applied"
        assert "actual_camera_binding_mismatch" in status["reason_codes"]
    finally:
        jobs.close()


def test_changed_qualified_result_cannot_reuse_old_handoff(tmp_path):
    jobs, job_id, result = fixture(tmp_path)
    try:
        selection = CameraSelection(jobs)
        selection.select(job_id)
        result["K"][0][0] += 1
        _write(jobs.artifact(job_id, "camera_result.json"), result)
        with pytest.raises(ValueError, match="differs from its report hash"):
            selection.select(job_id)
        _write(jobs.artifact(job_id, "camera_profile.json"), camera_profile(result))
        with pytest.raises(ValueError, match="differs from its report hash"):
            selection.select(job_id)
    finally:
        jobs.close()
