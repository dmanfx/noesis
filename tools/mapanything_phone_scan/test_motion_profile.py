"""Short profiles keep modelled drift, fixed validation, and raw evidence distinct."""
from copy import deepcopy
import json
import shutil

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .calibration_jobs import _write
from .imu_noise_calibration import run_noise_calibration
from .motion_preprocessing import (CONSUMER_VERSION, PROFILE_SCHEMA, MotionProfileError,
                                   bind_profile_to_capture, checked_corrections, corrected_noise, json_sha)
from .motion_profile import (MotionSelection, _artifact, assemble_motion_profile,
                             run_motion_profile, score_heldout_motion, validate_short_walk_runtime)
from .calibrated_walk import CameraSelection
from .roomwalk_calibration import _load_noise_reference, _sha, _attach_point_timing
from .test_calibrated_walk import fixture as camera_fixture
from .test_imu_noise_calibration import noise_fixture
from .test_roomwalk_calibration import physical_result_fixture, unsupported_direct_physical_fixture


def corrections():
    return {"accelerometer_matrix": np.diag([1.01, 1.02, 1.03]).tolist(),
            "accelerometer_bias": [0.01, 0.02, 0.03],
            "gyroscope_matrix": np.diag([0.99, 1.01, 1.02]).tolist(),
            "gyroscope_bias": [0.001, 0.002, 0.003]}


def save_artifact(directory, name, value):
    directory.mkdir(exist_ok=True, parents=True)
    _write(directory / name, value)
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"status": "completed", "artifacts": {}}
    report["artifacts"][name] = {"path": name, "bytes": (directory / name).stat().st_size, "sha256": _sha(directory / name)}
    _write(report_path, report)


@pytest.fixture
def source_jobs(tmp_path):
    jobs, camera_id, camera = camera_fixture(tmp_path)
    noise_dir = tmp_path / "noise"
    noise = run_noise_calibration(noise_fixture(tmp_path / "stationary", duration_s=60), noise_dir)
    _write(noise_dir / "report.json", noise)
    reference = _load_noise_reference(noise_dir, {"device": noise["device"]}, {"recorder": {"sensors": noise["sensors"]}})
    imu = {"schema": "roomwalk.camera_imu_calibration.v1", "quality": {"status": "qualified"},
           "camera_imu_extrinsics_calibrated": True, "time_offset_calibrated": True,
           "physical_board_scale_verified": True, "camera_model_qualified": True,
           "camera_reference_sha256": json_sha(camera), "binding": camera["binding"],
           "noise_reference": reference, "point_timing": {"qualified": True},
           "T_imu_camera": np.eye(4).tolist(), "imu_to_camera_offset_ns": -5000000,
           "cam_time_offset_ns": 5000000, "imu_corrections": corrections()}
    imu_dir = tmp_path / "imu"
    save_artifact(imu_dir, "camera_imu_result.json", imu)
    dirs = {"camera_calibration_dir": jobs.artifact(camera_id, "report.json").parent,
            "imu_calibration_dir": imu_dir, "noise_calibration_dir": noise_dir}
    yield dirs, imu, jobs
    jobs.close()


def test_assembly_preserves_model_prior_without_claiming_allan_measurement(source_jobs):
    dirs, _, _ = source_jobs
    profile = assemble_motion_profile(**dirs)
    assert profile["status"] == "candidate"
    assert profile["imu_noise_calibrated"] is False
    assert profile["noise_provenance"]["gyroscope_random_walk"]["kind"] == "model_prior"
    assert profile["runtime_world_admission"] is False
    assert profile["profile_id"] == assemble_motion_profile(**dirs)["profile_id"]


@pytest.mark.parametrize("bad", ["camera", "noise", "timing", "scale", "correction"])
def test_profile_assembly_rejects_mixed_or_unqualified_sources(source_jobs, bad):
    dirs, imu, _ = source_jobs
    if bad == "camera":
        imu["camera_reference_sha256"] = "changed"
    elif bad == "noise":
        imu["noise_reference"]["source_sha256"] = "changed"
    elif bad == "timing":
        imu["time_offset_calibrated"] = False
    elif bad == "scale":
        imu["physical_board_scale_verified"] = False
    else:
        imu["imu_corrections"]["gyroscope_matrix"][0][0] = -1
    save_artifact(dirs["imu_calibration_dir"], "camera_imu_result.json", imu)
    with pytest.raises(ValueError):
        assemble_motion_profile(**dirs)


def trajectories(scale=1.0, drift=0.0):
    epoch = 18_900_000_000_000_789
    rotation = Rotation.from_euler("xyz", [0.2, -0.5, 0.7]).as_matrix()
    visual, poses, mapping = [], [], []
    for i in range(201):
        t = i * 0.05
        stamp = epoch + i * 50_000_000
        ref = np.eye(4)
        ref[:3, :3] = Rotation.from_euler("xyz", [0.1 * np.sin(t), 0.05 * t, 0.03 * t]).as_matrix()
        ref[:3, 3] = [0.3 * np.sin(t / 3), 0.15 * np.cos(t / 2), 0.02 * t]
        measured = ref.copy()
        measured[:3, :3] = rotation.T @ ref[:3, :3]
        measured[:3, 3] = rotation.T @ (ref[:3, 3] * scale) + [3, 5, 4]
        measured[0, 3] += drift * t
        visual.append({"timestamp_ns": stamp, "T_target_camera": ref.tolist()})
        poses.append({"capture_time_ns": stamp - 7000000, "pose_time_ns": stamp,
                      "T_vio_world_camera": measured.tolist()})
        mapping.append([stamp - 7000000, stamp])
    return {"poses": poses, "segments": [{"reset": False}]}, {"visual_only": True, "status": "converged", "visual_trajectory": visual}, mapping


def test_fixed_profile_score_is_rigid_only_and_keeps_scale_error_visible():
    assert score_heldout_motion(*trajectories())["passed"]
    result = score_heldout_motion(*trajectories(scale=1.2))
    assert not result["passed"]
    assert "heldout_metric_scale_mismatch" in result["reason_codes"]
    assert result["metrics"]["scale_ratio_diagnostic_only"] == pytest.approx(1.2)
    assert result["scale_fitted_or_applied"] is False
    assert not score_heldout_motion(*trajectories(drift=0.05))["passed"]


def test_heldout_score_rejects_gaps_wrong_clock_and_insufficient_motion():
    vio, visual, mapping = trajectories()
    vio["poses"] = vio["poses"][::10]
    result = score_heldout_motion(vio, visual, mapping)
    assert not result["passed"] and "insufficient_heldout_coverage" in result["reason_codes"]
    vio["poses"][0]["pose_time_ns"] += 1
    with pytest.raises(MotionProfileError, match="exposure times"):
        score_heldout_motion(vio, visual, mapping)


@pytest.mark.parametrize("bad", [None, "jump", "duration", "envelope", "provenance"])
def test_normal_walk_quality_does_not_admit_out_of_scope_or_divergent_motion(bad):
    vio, _, mapping = trajectories()
    envelope = {"maximum_rotation_during_exposure_rad": 0.005, "maximum_rotation_during_readout_rad": 0.03}
    short = {"consumer_version": CONSUMER_VERSION, "validation_run": False,
             "profile_validation_sha256": "a" * 64, "motion_envelope": deepcopy(envelope),
             "validated_motion_envelope": envelope}
    metadata = {"short_session": short, "frame_mapping": [{"capture_time_ns": a, "pose_time_ns": b} for a, b in mapping]}
    if bad == "jump":
        vio["poses"][40]["T_vio_world_camera"][0][3] += 1
    elif bad == "duration":
        metadata["frame_mapping"][-1]["pose_time_ns"] += 600_000_000_000
    elif bad == "envelope":
        short["motion_envelope"]["maximum_rotation_during_readout_rad"] = 0.3
    elif bad == "provenance":
        short["profile_validation_sha256"] = None
    result = validate_short_walk_runtime(vio, metadata)
    assert result["accepted"] is (bad is None)
    assert result["accuracy_validated"] is False


def capture_fixture(root, monkeypatch, *, unsupported_control=False):
    obs, rows, timing, bound = (unsupported_direct_physical_fixture() if unsupported_control
                                else physical_result_fixture())
    _attach_point_timing(obs, rows, timing, bound)
    bound.update(qualified=True, reason_codes=[], sha256="binding")
    bound["signature"]["resolution"] = obs["resolution"]
    monkeypatch.setattr("tools.mapanything_phone_scan.roomwalk_calibration.camera_binding_for_capture", lambda _: bound)
    second = deepcopy(rows[0])
    second["sensor_timestamp_ns"] += 50_000_000
    if "physical_capture_result" in second:
        second["physical_capture_result"]["sensor_timestamp_ns"] += 50_000_000
    rows.append(second)
    times = [r["sensor_timestamp_ns"] for r in rows]
    (root / "camera.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    raw = deepcopy(timing["recorder"])
    raw["device"] = raw.pop("raw_device", {})
    _write(root / "raw.json", raw)
    (root / "video_timestamps_ns.json").write_text(json.dumps(times))
    sensors = {k: [{"timestamp_ns": t, "si": [0.1, 0.2, 0.3]} for t in range(980000000, 1100000000, 5000000)] for k in ("accel", "gyro")}
    _write(root / "imu_normalized.json", sensors)
    report = {"schema": "noesis.phone_capture.v1", "manifest": {"camera": {}, "imu": {}, "clocks": {},
              "android_capture": {"camera_results_path": "camera.jsonl", "capture_result_path": "raw.json"}},
              "calibration": {}, "coverage": {}}
    profile = {"schema": PROFILE_SCHEMA, "consumer_version": CONSUMER_VERSION,
               "status": "ready_for_short_walks", "profile_id": "test", "binding": bound,
               "point_timing": obs["point_timing"], "imu_corrections": corrections(),
               "noise": {s + "_" + k: 0.01 for s in ("accelerometer", "gyroscope") for k in ("noise_density", "random_walk")},
               "noise_model_status": "short_session_candidate", "imu_to_camera_offset_ns": 0,
               "camera": {}, "T_imu_camera": np.eye(4).tolist(),
               "validated_motion_envelope": {"maximum_rotation_during_exposure_rad": 0.1, "maximum_rotation_during_readout_rad": 0.1}}
    return report, profile


@pytest.mark.parametrize("unsupported_control", [False, True])
def test_capture_binding_uses_physical_exposure_and_preserves_original_times(tmp_path, monkeypatch, unsupported_control):
    report, profile = capture_fixture(tmp_path, monkeypatch, unsupported_control=unsupported_control)
    original = deepcopy(report)
    derived = bind_profile_to_capture(tmp_path, report, profile)
    assert report == original
    assert derived["short_session"]["source_and_pose_times_ns"][0] == (1000000000, 1007000000)
    assert derived["short_session"]["motion_envelope"]["maximum_exposure_s"] == 0.002
    assert derived["short_session"]["motion_envelope"]["maximum_readout_s"] == 0.012
    assert derived["calibration"]["imu_noise_calibrated"] is False
    assert derived["metric_vio_allowed"] is True
    assert bind_profile_to_capture(tmp_path, report, profile, validation=True)["metric_vio_allowed"] is False
    profile["validated_motion_envelope"]["maximum_rotation_during_readout_rad"] = 1e-6
    with pytest.raises(MotionProfileError, match="exceeds"):
        bind_profile_to_capture(tmp_path, report, profile)


def test_bad_model_and_changed_artifact_fail_closed(source_jobs):
    dirs, _, jobs = source_jobs
    assert corrected_noise({s + "_" + k: 0.01 for s in ("accelerometer", "gyroscope") for k in ("noise_density", "random_walk")}, corrections())["accelerometer_noise_density"] == pytest.approx(0.0103)
    bad = corrections()
    bad["gyroscope_bias"][0] = float("nan")
    with pytest.raises(MotionProfileError):
        checked_corrections(bad)
    (dirs["imu_calibration_dir"] / "camera_imu_result.json").write_text("{}")
    with pytest.raises(MotionProfileError, match="retained report"):
        _artifact(dirs["imu_calibration_dir"], "camera_imu_result.json")
    selection = MotionSelection(jobs, None)
    assert selection.get() is None and selection.select(None) is None


@pytest.mark.parametrize("scale", [1.0, 1.25])
def test_job_and_selection_roundtrip_retains_failed_profile_and_exact_runtime_config(source_jobs, tmp_path, monkeypatch, scale):
    dirs, imu, jobs = source_jobs
    source_ids = {"camera": dirs["camera_calibration_dir"].parent.name,
                  "imu": "cal-20260915-120000-000000000001",
                  "noise": "cal-20260915-120000-000000000002"}
    for mode in ("imu", "noise"):
        directory = jobs.root / source_ids[mode]
        shutil.copytree(dirs[mode + "_calibration_dir"], directory / "artifacts")
        dirs[mode + "_calibration_dir"] = directory / "artifacts"
        _write(directory / "state.json", {"id": source_ids[mode], "mode": mode, "status": "completed", "created_at": "now"})
    vio, visual, mapping = trajectories(scale=scale)
    save_artifact(dirs["imu_calibration_dir"], "heldout_visual_result.json", visual)
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    noise_binding = imu["noise_reference"]["binding"]
    capture = {"schema": "noesis.phone_capture.v1", "manifest": {"device": noise_binding["device"],
               "android_capture": {"capture_result_path": "raw.json"}}}
    _write(capture_dir / "capture_import.json", capture)
    _write(capture_dir / "raw.json", noise_binding)
    envelope = {"maximum_rotation_during_readout_rad": 0.01, "maximum_rotation_during_exposure_rad": 0.002}
    monkeypatch.setattr("tools.mapanything_phone_scan.motion_profile.bind_profile_to_capture",
                        lambda *a, **k: {**capture, "short_session": {"source_and_pose_times_ns": mapping, "motion_envelope": envelope}})
    exe, config, runtime_config = [tmp_path / name for name in ("runner", "base.yaml", "runtime.yaml")]
    exe.write_text("native-fixture")
    config.write_text("base-config")
    runtime_config.write_text("materialized-config")
    monkeypatch.setenv("NOESIS_PHONE_SCAN_VIO_EXECUTABLE", str(exe))
    monkeypatch.setenv("NOESIS_PHONE_SCAN_VIO_CONFIG", str(config))
    calls = []
    def materialize(source, report, output, settings, progress):
        calls.append("materialize")
        assert source == capture_dir and report["short_session"]["source_and_pose_times_ns"] == mapping
        return output, runtime_config
    def run(source, output, settings, progress):
        calls.append("run")
        assert settings.config == runtime_config
        return vio
    monkeypatch.setattr("tools.mapanything_phone_scan.vio.materialize_openvins_profile_validation_input", materialize)
    monkeypatch.setattr("tools.mapanything_phone_scan.vio.run_openvins_profile_validation", run)
    job_id = "cal-20260915-120000-000000000003"
    output = jobs.root / job_id / "artifacts"
    request = {mode + "_calibration_id": job for mode, job in source_ids.items()}
    report = run_motion_profile(capture_dir, request, output,
                                settings={**dirs, "vio_executable": exe, "vio_config": config}, progress=lambda *a: None)
    assert calls == ["materialize", "run"]
    assert report["status"] == "completed", report
    assert report["motion_profile_ready"] is (scale == 1)
    assert json.loads((capture_dir / "capture_import.json").read_text()) == capture
    assert (output / "profile_vio_result.json").is_file()
    _write(output.parent / "state.json", {"id": job_id, "mode": "motion_profile", "status": "completed", "result": report, "created_at": "now"})
    camera_selection = CameraSelection(jobs)
    selection = MotionSelection(jobs, camera_selection)
    if scale != 1:
        with pytest.raises(ValueError, match="passing"):
            selection.select(job_id)
        assert _artifact(output, "motion_profile.json")["status"] == "check_failed"
        return
    snapshot = selection.select(job_id)
    assert snapshot["status"] == "ready_for_short_walks" and snapshot["maximum_duration_s"] == 300
    assert camera_selection.get() is None  # no unasked global intrinsic selection
    assert selection.select(None) is None
    assert selection.derived_report(capture_dir, capture, snapshot)["short_session"]["source_and_pose_times_ns"] == mapping
    config.write_text("changed-consumer-config")
    with pytest.raises(MotionProfileError, match="consumer changed"):
        selection.derived_report(capture_dir, capture, snapshot)
