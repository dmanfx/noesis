from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .roomwalk_calibration import (
    CAMERA_SCHEMA,
    DEFAULT_BOARD,
    REPORT_SCHEMA,
    REQUEST_SCHEMA,
    CalibrationError,
    CalibrationSettings,
    _Run,
    _board_object,
    _camera_model,
    _capture_binding,
    _digest,
    _imu_input,
    _load_camera_reference,
    _native_solve,
    _rotation_seed,
    _sample_indices,
    _sha,
    fit_camera_observations,
    _heldout_imu,
    _load_noise_reference,
    _visual_holdout,
    _attach_point_timing,
    camera_profile,
    camera_binding_for_capture,
    run_calibration,
    validate_calibration_request,
)


def request(mode="camera"):
    return {
        "schema": REQUEST_SCHEMA,
        "mode": mode,
        "board": copy.deepcopy(DEFAULT_BOARD),
    }


def binding():
    signature = {"test": "fixed qualified geometry"}
    return {
        "schema": "roomwalk.camera_binding.v1",
        "signature": signature,
        "sha256": _digest(signature),
        "qualified": True,
        "reason_codes": [],
    }


def observations():
    rng = np.random.default_rng(31)
    obj = _board_object(DEFAULT_BOARD).getChessboardCorners().astype(float)
    K = np.array([[980.0, 0, 640.0], [0, 975.0, 480.0], [0, 0, 1.0]])
    D = np.array([-0.06, 0.02, 0.002, -0.001, 0.004])
    rows = []
    for index in range(64):
        r = np.array(
            [rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-0.35, 0.35)]
        )
        t = np.array(
            [
                rng.uniform(-0.22, 0.04),
                rng.uniform(-0.25, 0.08),
                rng.uniform(0.43, 0.85),
            ]
        )
        px = cv2.projectPoints(obj, r, t, K, D)[0].reshape(-1, 2)
        ids = np.flatnonzero(
            (px[:, 0] > 8) & (px[:, 0] < 1272) & (px[:, 1] > 8) & (px[:, 1] < 952)
        )
        if len(ids) < 20:
            continue
        rows.append(
            {
                "frame_index": index,
                "timestamp_ns": 1_000_000_000 + index * 500_000_000,
                "time_s": index * 0.5,
                "ids": ids.tolist(),
                "points": (px[ids] + rng.normal(0, 0.035, (len(ids), 2))).tolist(),
                "rejection_reasons": [],
            }
        )
    return {"resolution": [1280, 960], "frames": rows}, K, D


def test_default_exact_board_and_normalization_idempotent():
    result = validate_calibration_request(request())
    assert result["board"]["marker_ids"] == list(range(300, 370))
    board = _board_object(result["board"])
    assert len(board.getChessboardCorners()) == 117
    assert len(board.getIds()) == 70
    assert board.getLegacyPattern() is False
    assert validate_calibration_request(result) == result
    assert (
        validate_calibration_request({"schema": REQUEST_SCHEMA, "mode": "imu"})["board"]
        == result["board"]
    )


@pytest.mark.parametrize("problem", ["missing_reference", "unconfirmed_board", "focus_mismatch", "unqualified_camera"])
def test_imu_preflight_rejects_before_native_detection(tmp_path, monkeypatch, problem):
    from . import roomwalk_calibration as module
    ref = {"schema": CAMERA_SCHEMA, "K": [[980, 0, 640], [0, 975, 480], [0, 0, 1]],
           "D": [0] * 5, "resolution": [1280, 960], "binding": binding(), "quality": {"status": "qualified"}}
    if problem == "focus_mismatch":
        ref["binding"]["sha256"] = "another_focus"
    if problem == "unqualified_camera":
        ref["quality"]["status"] = "rejected"
    monkeypatch.setattr(module, "_load_camera_reference", lambda *_: None if problem == "missing_reference" else ref)
    capture_calls = []
    def capture(*_):
        capture_calls.append(True)
        return {"capture_id": "kept"}, tmp_path / "camera.mp4", [1280, 960], [], [], {}, {}
    monkeypatch.setattr(module, "_capture", capture)
    monkeypatch.setattr(module, "_capture_binding", lambda *_: binding())
    monkeypatch.setattr(module, "_detect_capture", lambda *_: pytest.fail("preflight must run before native detection"))
    report = run_calibration(tmp_path / "source", {**request("imu"), "board_geometry_confirmed": problem != "unconfirmed_board"}, tmp_path / "output")
    assert report["status"] == "failed"
    assert report["accepted_for_metric_vio"] is False
    expected = {"missing_reference": "server-resolved", "unconfirmed_board": "printed board", "focus_mismatch": "lens/focus/crop", "unqualified_camera": "qualified matching"}
    assert expected[problem] in report["error"]
    assert bool(capture_calls) == (problem in {"focus_mismatch", "unqualified_camera"})




@pytest.mark.parametrize(
    "field,value",
    [
        ("squares_x", True),
        ("squares_y", 1),
        ("square_length_m", float("nan")),
        ("marker_length_m", 0.02),
        ("dictionary", "__dict__"),
        ("legacy_pattern", "false"),
        ("marker_ids", list(range(301, 370))),
        ("marker_ids", [300] * 70),
        ("marker_ids", list(range(950, 1020))),
    ],
)
def test_invalid_board_rejected(field, value):
    row = request()
    row["board"][field] = value
    with pytest.raises(CalibrationError):
        validate_calibration_request(row)


def test_changed_board_identity_and_contradictory_geometry():
    row = request()
    before = validate_calibration_request(row)["board_sha256"]
    row["board"]["legacy_pattern"] = True
    assert validate_calibration_request(row)["board_sha256"] != before
    row = request()
    row["board"]["charuco_corners"] = [{"point_m": [0, 0, 0]}] * 117
    with pytest.raises(CalibrationError, match="geometry"):
        validate_calibration_request(row)


def test_no_uploaded_inline_camera_or_paths():
    row = request("imu")
    row["camera_calibration"] = {"K": np.eye(3).tolist()}
    with pytest.raises(CalibrationError, match="server"):
        validate_calibration_request(row)
    row = request()
    row["camera_calibration_id"] = "../other"
    with pytest.raises(CalibrationError):
        validate_calibration_request(row)


def test_real_camera_fit_and_independent_block_holdouts():
    obs, K, D = observations()
    result = fit_camera_observations(obs, request(), binding())
    assert result["schema"] == CAMERA_SCHEMA
    assert len(result["D"]) == 5
    assert result["tangential_coefficients"] == "estimated"
    assert result["fit_uses_holdout"] is False
    assert not set(result["training_frame_indices"]) & set(
        result["holdout_frame_indices"]
    )
    np.testing.assert_allclose(
        np.array(result["K"])[[0, 1], [0, 1]], K[[0, 1], [0, 1]], rtol=0.003
    )
    assert result["quality"]["heldout"]["radial_px"]["rms"] < 0.15
    assert result["accepted_for_metric_vio"] is False
    # A repeat fitting the same training set cannot depend on perturbed withheld
    # points. Reverse-fit diagnostics change, but the returned K/D do not.
    bad = copy.deepcopy(obs)
    for row in bad["frames"]:
        if row["frame_index"] in result["holdout_frame_indices"]:
            row["points"] = (
                np.array(row["points"])
                + np.random.default_rng(row["frame_index"]).normal(
                    0, 2, (len(row["points"]), 2)
                )
            ).tolist()
    changed = fit_camera_observations(bad, request(), binding())
    np.testing.assert_array_equal(changed["K"], result["K"])
    np.testing.assert_array_equal(changed["D"], result["D"])
    assert "heldout_reprojection_exceeds_policy" in changed["quality"]["reason_codes"]


def test_camera_fit_does_not_qualify_unlocked_lens():
    obs, _, _ = observations()
    unqualified = {
        **binding(),
        "qualified": False,
        "reason_codes": ["focus_not_explicitly_locked"],
    }
    result = fit_camera_observations(obs, request(), unqualified)
    assert result["quality"]["status"] == "rejected"
    assert "focus_not_explicitly_locked" in result["quality"]["reason_codes"]


def test_binding_uses_actual_rows_and_ignores_session_counters():
    manifest = {"device": {"id": "same"}, "camera": {"id": "0", "orientation_deg": 0}}
    row = {
        "active_physical_camera_id": "5",
        "lens_focal_length_mm": 6.25,
        "lens_focus_distance_diopters": 1.2,
        "zoom_ratio": 1.0,
        "crop_region": [0, 0, 8192, 6144],
        "ois_mode": 0,
        "eis_mode": 0,
        "rotate_and_crop_mode": 0,
        "distortion_correction_mode": 0,
    }
    timing = {
        "recorder": {
            "camera": {"focus_control": {"mode": "locked", "confirmed_frame_count": 5}}
        }
    }
    a = _capture_binding(manifest, [row, row], [7680, 4320], timing)
    timing["recorder"]["camera"]["focus_control"]["confirmed_frame_count"] = 20
    b = _capture_binding(manifest, [row, row], [7680, 4320], timing)
    assert a["sha256"] == b["sha256"] and a["qualified"]
    pinned_manifest = {**manifest, "camera": {**manifest["camera"], "id": "5"}}
    pinned = _capture_binding(pinned_manifest, [row, row], [7680, 4320], timing)
    assert pinned["qualified"] and pinned["sha256"] != a["sha256"]
    changed = dict(row, active_physical_camera_id="2")
    assert not _capture_binding(manifest, [row, changed], [7680, 4320], timing)[
        "qualified"
    ]
    changed = dict(row, lens_focus_distance_diopters=1.3)
    assert (
        _capture_binding(manifest, [changed, changed], [7680, 4320], timing)["sha256"]
        != a["sha256"]
    )


def test_unsupported_distortion_control_can_bind_output_but_never_claims_off():
    manifest = {"device": {"id": "phone"}, "camera": {"id": "0", "orientation_deg": 0}}
    row = {
        "active_physical_camera_id": "5",
        "lens_focal_length_mm": 6.25,
        "lens_focus_distance_diopters": 1.2,
        "zoom_ratio": 1.0,
        "crop_region": [0, 0, 640, 480],
        "ois_mode": 0,
        "eis_mode": 0,
        "rotate_and_crop_mode": 0,
        "distortion_correction_mode": None,
    }
    camera = {
        "id": "0",
        "focus_control": {"mode": "locked", "build_fingerprint": "verified-build"},
        "distortion_correction_request_key_available": False,
        "distortion_correction_result_key_available": False,
        "distortion_correction_available_modes": None,
    }
    timing = {"recorder": {"camera": camera}}
    bound = _capture_binding(manifest, [row, row], [640, 480], timing)
    assert bound["qualified"]
    assert bound["signature"]["actual_camera2"]["distortion_correction_mode"] is None
    assert bound["signature"]["distortion_control"]["assumed_off"] is False
    assert bound["signature"]["distortion_control"]["row_mapping_qualified"] is False
    camera["distortion_correction_result_key_available"] = True
    assert not _capture_binding(manifest, [row], [640, 480], timing)["qualified"]
    camera["distortion_correction_result_key_available"] = False
    camera["distortion_correction_available_modes"] = [0, 1]
    assert not _capture_binding(manifest, [row], [640, 480], timing)["qualified"]
    camera["distortion_correction_available_modes"] = None
    camera["focus_control"].pop("build_fingerprint")
    assert not _capture_binding(manifest, [row], [640, 480], timing)["qualified"]


def test_sampling_uses_integer_acquisition_times_and_is_bounded():
    start = 18_900_000_000_000_789
    times = [start + i * 33_366_667 for i in range(1000)]
    chosen = _sample_indices(times, 10, 120)
    assert len(chosen) == 120 and len(set(chosen)) == 120
    assert chosen[0] == 0


def test_camera_reference_cannot_change_size_or_lens():
    model = {
        "schema": CAMERA_SCHEMA,
        "K": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        "D": [0, 0, 0, 0, 0.01],
        "resolution": [640, 480],
        "binding": binding(),
        "quality": {"status": "qualified"},
    }
    assert _camera_model(model, [640, 480], binding(), False)[2]
    with pytest.raises(CalibrationError, match="geometry"):
        _camera_model(model, [1280, 960], binding(), True)
    model["binding"]["sha256"] = "different"
    with pytest.raises(CalibrationError, match="binding"):
        _camera_model(model, [640, 480], binding(), True)


def test_server_resolved_camera_artifact_hash_is_verified(tmp_path):
    result = tmp_path / "camera_result.json"
    result.write_text(json.dumps({"schema": CAMERA_SCHEMA}))
    report = {
        "schema": REPORT_SCHEMA,
        "status": "completed",
        "mode": "camera",
        "artifacts": {
            "camera_result.json": {
                "path": "camera_result.json",
                "bytes": result.stat().st_size,
                "sha256": _sha(result),
            }
        },
    }
    (tmp_path / "report.json").write_text(json.dumps(report))
    assert _load_camera_reference(tmp_path)["schema"] == CAMERA_SCHEMA
    result.write_text(json.dumps({"schema": CAMERA_SCHEMA, "changed": True}))
    with pytest.raises(CalibrationError, match="hash"):
        _load_camera_reference(tmp_path)


def test_cancellation_retains_report_and_never_reads_or_changes_capture(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "untouched").write_bytes(b"original")
    result = run_calibration(
        capture, request(), tmp_path / "output", cancelled=lambda: True
    )
    assert result["status"] == "cancelled"
    assert result["accepted_for_metric_vio"] is False
    assert (capture / "untouched").read_bytes() == b"original"
    assert (tmp_path / "output/report.json").is_file()
    with pytest.raises(CalibrationError, match="already exists"):
        run_calibration(capture, request(), tmp_path / "output")


def test_output_cannot_be_inside_capture(tmp_path):
    with pytest.raises(CalibrationError, match="separate"):
        run_calibration(tmp_path, request(), tmp_path / "output")


def synthetic_fixture():
    path = Path(__file__).parent / "native/roomwalk_calibration/synthetic.py"
    spec = importlib.util.spec_from_file_location("roomwalk_synthetic_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixture


def test_real_rotation_time_initialization_and_independent_streams():
    data, truth = synthetic_fixture()(17_000_000)
    T, offset, excitation = _rotation_seed(data["frames"], data["gyro"])
    wanted = np.array(truth["T_imu_camera"])
    assert Rotation.from_matrix(T[:3, :3].T @ wanted[:3, :3]).magnitude() < 0.01
    assert abs(offset - truth["cam_time_offset_ns"]) < 4_000_000
    assert excitation[-1] > 0
    assert data["accel"][0]["timestamp_ns"] != data["gyro"][0]["timestamp_ns"]
    still = copy.deepcopy(data["frames"])
    for row in still:
        row["T_target_camera"] = still[0]["T_target_camera"]
    with pytest.raises(CalibrationError, match="excited"):
        _rotation_seed(still, data["gyro"])


@pytest.mark.parametrize("offset", [17_000_000, -23_000_000])
def test_native_basalt_real_synthetic_solve(tmp_path, offset):
    executable = os.environ.get("NOESIS_PHONE_SCAN_CALIBRATION_SOLVER")
    if not executable:
        pytest.skip(
            "set NOESIS_PHONE_SCAN_CALIBRATION_SOLVER to exercise the real pinned CPU solver"
        )
    data, truth = synthetic_fixture()(offset)
    run = _Run(
        CalibrationSettings(solver_path=Path(executable), timeout_s=300), None, None
    )
    result = _native_solve(data, tmp_path, run)
    got, wanted = np.array(result["T_imu_camera"]), np.array(truth["T_imu_camera"])
    assert result["status"] == "converged"
    assert Rotation.from_matrix(got[:3, :3].T @ wanted[:3, :3]).magnitude() < 0.01
    assert np.linalg.norm(got[:3, 3] - wanted[:3, 3]) < 0.01
    assert abs(result["cam_time_offset_ns"] - offset) < 1_000_000
    assert result["imu_to_camera_offset_ns"] == -result["cam_time_offset_ns"]
    assert result["accepted_for_metric_vio"] is False
    assert result["imu_noise_calibrated"] is False
    assert len(result["imu_corrections"]["accelerometer_parameters"]) == 9
    assert len(result["imu_corrections"]["gyroscope_parameters"]) == 12


def test_native_rejects_duplicate_target_id_before_optimization(tmp_path):
    executable = os.environ.get("NOESIS_PHONE_SCAN_CALIBRATION_SOLVER")
    if not executable:
        pytest.skip("native executable not configured")
    data, _ = synthetic_fixture()()
    data["frames"][0]["ids"][1] = data["frames"][0]["ids"][0]
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data))
    completed = subprocess.run(
        [executable, str(path), str(tmp_path / "result.json")],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode != 0
    assert "duplicate" in completed.stderr
    assert not (tmp_path / "result.json").exists()


def test_real_python_imu_preparation_preserves_holdout_and_native_corrections(tmp_path):
    executable = os.environ.get("NOESIS_PHONE_SCAN_CALIBRATION_SOLVER")
    if not executable:
        pytest.skip("native executable not configured")
    data, truth = synthetic_fixture()(-23_000_000)
    am = np.array([[1.015, 0, 0], [0.002, 0.991, 0], [0.003, -0.001, 1.008]])
    gm = np.array(
        [[1.008, 0.003, -0.001], [0.002, 0.992, 0.002], [0.001, -0.002, 1.012]]
    )
    ab = np.array([0.02, -0.03, 0.01])
    gb = np.array([0.001, -0.002, 0.0015])
    for kind, M, b in (("accel", am, ab), ("gyro", gm, gb)):
        for row in data[kind]:
            row["xyz"] = np.linalg.solve(M, np.array(row["xyz"]) + b).tolist()
    obs = {
        "resolution": data["resolution"],
        "frames": [
            {**r, "points": r["pixels"], "rejection_reasons": []}
            for r in data["frames"]
        ],
    }
    ref = {
        "schema": CAMERA_SCHEMA,
        "K": data["K"],
        "D": [0] * 5,
        "resolution": data["resolution"],
        "binding": binding(),
        "quality": {"status": "qualified"},
    }
    run = _Run(
        CalibrationSettings(solver_path=Path(executable), timeout_s=300), None, None
    )
    payload, heldout, qualified, _ = _imu_input(
        obs,
        validate_calibration_request(request("imu")),
        ref,
        binding(),
        {k: data[k] for k in ("accel", "gyro")},
        run,
    )
    assert qualified
    assert payload["accel"][-1]["timestamp_ns"] < heldout[0]["timestamp_ns"]
    assert payload["gyro"][-1]["timestamp_ns"] < heldout[0]["timestamp_ns"]
    assert not set(f["frame_index"] for f in payload["frames"]) & set(
        f["frame_index"] for f in heldout
    )
    result = _native_solve(payload, tmp_path, run)
    wanted = np.array(truth["T_imu_camera"])
    got = np.array(result["T_imu_camera"])
    assert np.linalg.norm(got[:3, 3] - wanted[:3, 3]) < 0.01
    assert Rotation.from_matrix(got[:3, :3].T @ wanted[:3, :3]).magnitude() < 0.01
    assert abs(result["cam_time_offset_ns"] - truth["cam_time_offset_ns"]) < 1_000_000
    np.testing.assert_allclose(
        result["imu_corrections"]["gyroscope_matrix"], gm, atol=0.002
    )
    np.testing.assert_allclose(
        result["imu_corrections"]["accelerometer_matrix"], am, atol=0.002
    )
    validation = _heldout_imu(heldout, {k: data[k] for k in ("accel", "gyro")}, result)
    assert validation["used_by_solver"] is False
    assert validation["gyro_vector_error_rad_s"]["rms"] < 0.005
    assert result["accepted_for_metric_vio"] is False


def test_noise_reference_is_separate_and_exactly_bound(tmp_path):
    assert not _load_noise_reference(None, {}, {})["imu_noise_calibrated"]
    device = {"id": "phone"}
    sensors = {"accelerometer": {"name": "a"}, "gyroscope": {"name": "g"}}
    result = {
        "imu_noise_calibrated": True,
        "binding": {"device": device, "sensors": sensors},
        "noise": {
            "gyroscope_noise_density": 0.001,
            "gyroscope_random_walk": 0.0001,
            "accelerometer_noise_density": 0.01,
            "accelerometer_random_walk": 0.001,
        },
        "unit_conventions": {
            "gyroscope_noise_density": "rad/s/sqrt(Hz)",
            "gyroscope_random_walk": "rad/s^2/sqrt(Hz)",
            "accelerometer_noise_density": "m/s^2/sqrt(Hz)",
            "accelerometer_random_walk": "m/s^3/sqrt(Hz)",
        },
    }
    path = tmp_path / "noise_result.json"
    path.write_text(json.dumps(result))
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "artifacts": {
                    "noise_result.json": {
                        "path": "noise_result.json",
                        "bytes": path.stat().st_size,
                        "sha256": _sha(path),
                    }
                },
            }
        )
    )
    assert _load_noise_reference(
        tmp_path, {"device": device}, {"recorder": {"sensors": sensors}}
    )["imu_noise_calibrated"]
    other = _load_noise_reference(
        tmp_path, {"device": {"id": "other"}}, {"recorder": {"sensors": sensors}}
    )
    assert not other["imu_noise_calibrated"]
    assert "noise_device_binding_mismatch" in other["reason_codes"]


@pytest.fixture(scope="module")
def short_noise_job(tmp_path_factory):
    from .imu_noise_calibration import run_noise_calibration
    from .test_imu_noise_calibration import noise_fixture

    root = tmp_path_factory.mktemp("short_noise")
    source = noise_fixture(root / "source", duration_s=60)
    directory = root / "output"
    result = run_noise_calibration(source, directory)
    assert result["noise_model_usable"], result["reason_codes"]
    (directory / "report.json").write_text(json.dumps(result))
    return directory, result


def _load_short_job(directory, result):
    return _load_noise_reference(directory, {"device": result["device"]},
                                 {"recorder": {"sensors": result["sensors"]}})


def test_real_short_noise_producer_to_bound_loader_keeps_measured_prior_distinction(short_noise_job):
    directory, result = short_noise_job
    reference = _load_short_job(directory, result)
    assert reference["noise_model_usable"] is True
    assert reference["noise_model_status"] == "short_session_candidate"
    assert reference["imu_noise_calibrated"] is False
    assert reference["noise"] == result["noise"]
    assert reference["noise_provenance"] == result["noise_provenance"]
    assert reference["source_sha256"] == result["artifacts"]["noise_result.json"]["sha256"]
    assert reference["source_bundle_sha256"] == result["source_bundle_sha256"]
    assert reference["reason_codes"] == []


@pytest.mark.parametrize("mutation", [None, "one_ulp", "sensor_id", "invalid", "missing"])
def test_noise_sensor_float32_serializations_match_without_tolerating_changed_sensor(tmp_path, short_noise_job, mutation):
    _, original = short_noise_job
    result = copy.deepcopy(original)
    values = {"accelerometer": (156.90640258789062, 0.004785645287483931),
              "gyroscope": (34.906036376953125, 0.001221729558892548)}
    for sensor, (maximum, resolution) in values.items():
        result["sensors"][sensor].update(maximum_range=maximum, resolution=resolution)
    path = tmp_path / "noise_result.json"
    path.write_text(json.dumps(result))
    (tmp_path / "report.json").write_text(json.dumps({"status": "completed", "artifacts": {
        "noise_result.json": {"path": path.name, "sha256": _sha(path), "bytes": path.stat().st_size}}}))
    actual = copy.deepcopy(result["sensors"])
    actual["accelerometer"].update(maximum_range=156.9064, resolution=0.0047856453)
    actual["gyroscope"].update(maximum_range=34.906036, resolution=0.0012217296)
    if mutation == "one_ulp":
        actual["gyroscope"]["resolution"] = float(np.nextafter(np.float32(actual["gyroscope"]["resolution"]), np.float32(np.inf)))
    elif mutation == "sensor_id": actual["gyroscope"]["sensor_id"] += "changed"
    elif mutation == "invalid": actual["accelerometer"]["maximum_range"] = float("nan")
    elif mutation == "missing": actual["accelerometer"].pop("resolution")
    before = copy.deepcopy(actual)
    reference = _load_noise_reference(tmp_path, {"device": result["device"]},
                                     {"recorder": {"sensors": actual}})
    assert reference["noise_model_usable"] is (mutation is None)
    assert reference["imu_noise_calibrated"] is False
    if mutation:
        assert "noise_sensor_binding_mismatch" in reference["reason_codes"]
    else:
        assert reference["noise"] == result["noise"]
        assert reference["binding"]["sensors"] == result["sensors"]
        assert actual == before


@pytest.mark.parametrize("bad", ["sensor", "device", "units", "missing_provenance", "prior_claimed_measured",
                                 "four_measured_claim", "failed_holdout", "short_duration", "missing_streams",
                                 "negative_term", "changed_prior", "unusable", "wrong_schema"])
def test_short_noise_loader_rejects_flags_without_matching_evidence(tmp_path, short_noise_job, bad):
    _, original = short_noise_job
    result = copy.deepcopy(original)
    if bad == "sensor":
        result["sensors"]["gyroscope"]["sensor_id"] = "another sensor"
    elif bad == "device":
        result["device"]["build_fingerprint"] = "new build"
    elif bad == "units":
        result["units"]["gyroscope_random_walk"] = "degrees/s"
    elif bad == "missing_provenance":
        result.pop("noise_provenance")
    elif bad == "prior_claimed_measured":
        result["noise_provenance"]["gyroscope_random_walk"]["measured"] = True
    elif bad == "four_measured_claim":
        result["imu_noise_calibrated"] = True
    elif bad == "failed_holdout":
        result["streams"]["gyroscope"]["axes"][0]["fits"]["white"]["heldout_passed"] = False
    elif bad == "short_duration":
        result["streams"]["accelerometer"]["duration_s"] = 57.9
    elif bad == "missing_streams":
        result.pop("streams")
    elif bad == "negative_term":
        result["noise"]["accelerometer_random_walk"] = -1
    elif bad == "changed_prior":
        result["streams"]["gyroscope"]["drift_prior"]["coefficient_xyz"][0] *= 0.01
    elif bad == "unusable":
        result["noise_model_usable"] = False
    else:
        result["schema"] = "unknown"
    path = tmp_path / "noise_result.json"
    path.write_text(json.dumps(result))
    # Rebind the artifact so this exercises semantic checks, not only hashing.
    (tmp_path / "report.json").write_text(json.dumps({"status": "completed", "artifacts": {
        "noise_result.json": {"path": path.name, "sha256": _sha(path), "bytes": path.stat().st_size}}}))
    reference = _load_short_job(tmp_path, original)
    assert not reference["noise_model_usable"]
    assert not reference["imu_noise_calibrated"]
    assert reference["reason_codes"]


def test_short_noise_loader_rejects_changed_artifact(tmp_path, short_noise_job):
    directory, original = short_noise_job
    (tmp_path / "report.json").write_bytes((directory / "report.json").read_bytes())
    (tmp_path / "noise_result.json").write_bytes((directory / "noise_result.json").read_bytes() + b" ")
    with pytest.raises(CalibrationError, match="report hash"):
        _load_short_job(tmp_path, original)


def test_short_noise_white_densities_reach_actual_imu_payload_with_independent_rates(short_noise_job):
    directory, result = short_noise_job
    noise = _load_short_job(directory, result)
    data, _ = synthetic_fixture()(-23_000_000)
    obs = {"resolution": data["resolution"], "frames": [
        {**r, "points": r["pixels"], "rejection_reasons": []} for r in data["frames"]]}
    ref = {"schema": CAMERA_SCHEMA, "K": data["K"], "D": [0]*5,
           "resolution": data["resolution"], "binding": binding(), "quality": {"status": "qualified"}}
    # Distinct rates must not be paired or replaced by one configured rate.
    streams = {"accel": data["accel"][::2], "gyro": data["gyro"]}
    payload, heldout, qualified, _ = _imu_input(
        obs, validate_calibration_request(request("imu")), ref, binding(), streams,
        _Run(CalibrationSettings(), None, None), noise)
    assert qualified and heldout
    weights = payload["weights"]
    assert weights["noise_reference"] == noise
    assert not weights["noise_reference"]["imu_noise_calibrated"]
    assert weights["observed_rates_hz"]["accel"] != weights["observed_rates_hz"]["gyro"]
    for stream, term, weight in (("gyro", "gyroscope_noise_density", "gyro_sample_sigma_rad_s"),
                                 ("accel", "accelerometer_noise_density", "accel_sample_sigma_m_s2")):
        rate = 1e9 / np.median(np.diff([r["timestamp_ns"] for r in payload[stream]]))
        assert weights[weight] == pytest.approx(noise["noise"][term] * np.sqrt(rate))
        assert payload[stream][-1]["timestamp_ns"] < heldout[0]["timestamp_ns"]


@pytest.mark.parametrize("gate", [None, "physical_rows", "board_scale", "translation", "timing"])
def test_practical_imu_quality_is_separate_from_noise_and_keeps_other_gates(tmp_path, monkeypatch, short_noise_job, gate):
    # This checks orchestration with controlled solver outputs, not accuracy.
    from . import roomwalk_calibration as module

    directory, result = short_noise_job
    manifest = {"capture_id": "board", "device": result["device"]}
    timing = {"recorder": {"sensors": result["sensors"]}}
    obs = {"resolution": [1280, 960], "frames": [], "point_timing": {
        "qualified": gate != "physical_rows", "reason_codes": ["physical_capture_result_row_mapping_unverified"] if gate == "physical_rows" else []}}
    reference = {"schema": CAMERA_SCHEMA, "K": [[980, 0, 640], [0, 975, 480], [0, 0, 1]],
                 "D": [0] * 5, "resolution": [1280, 960], "binding": binding(), "quality": {"status": "qualified"}}
    monkeypatch.setattr(module, "_capture", lambda *_: (
        manifest, tmp_path / "unused.mp4", [1280, 960], np.arange(64)*100_000_000, [], timing, {}))
    monkeypatch.setattr(module, "_capture_binding", lambda *_: binding())
    monkeypatch.setattr(module, "_detect_capture", lambda *_: obs)
    monkeypatch.setattr(module, "_attach_point_timing", lambda *_: None)
    monkeypatch.setattr(module, "_load_camera_reference", lambda *_: reference)
    monkeypatch.setattr(module, "_imu_arrays", lambda *_: {})
    monkeypatch.setattr(module, "_imu_input", lambda *_: ({}, [], True, [1, 1, 1]))
    monkeypatch.setattr(module, "_native_solve", lambda *_: {
        "status": "converged", "accepted_for_metric_vio": False, "T_imu_camera": np.eye(4).tolist(),
        "cam_time_offset_ns": 23_000_000, "imu_to_camera_offset_ns": -23_000_000, "imu_corrections": {}})
    monkeypatch.setattr(module, "_visual_holdout", lambda *_: {})
    monkeypatch.setattr(module, "_heldout_imu", lambda *_: {
        "gyro_vector_error_rad_s": {"rms": 0.001},
        "translation_validation": {"reason_codes": ["bad_translation"] if gate == "translation" else []},
        "time_offset_validation": {"reason_codes": ["bad_timing"] if gate == "timing" else []}})
    req = {**request("imu"), "board_geometry_confirmed": gate != "board_scale",
           "allow_provisional_camera": gate == "board_scale"}
    report = run_calibration(tmp_path / "capture", req, tmp_path / "output",
                             settings=CalibrationSettings(noise_calibration_dir=directory))
    assert report["status"] == "completed", report
    assert report["quality"]["status"] == ("qualified" if gate is None else "insufficient_evidence")
    assert report["noise_model_usable"] is True
    assert report["imu_noise_calibrated"] is False
    assert report["accepted_for_metric_vio"] is False
    saved = json.loads((tmp_path / "output/camera_imu_result.json").read_text())
    assert saved["noise_characterization"]["all_four_terms_measured"] is False
    assert saved["noise_reference"]["noise_provenance"] == result["noise_provenance"]
    assert saved["runtime_admission"]["accepted"] is False
    assert set(saved["runtime_admission"]["reason_codes"]) >= {
        "imu_correction_consumer_not_admitted", "camera_row_time_consumer_not_admitted",
        "short_noise_profile_validation_not_admitted"}
    assert "imu_noise_not_measured" not in saved["quality"]["reason_codes"]


@pytest.mark.parametrize("unsupported_camera2", [False, True])
def test_native_rolling_rows_and_independent_translation_holdout(tmp_path, unsupported_camera2):
    executable = os.environ.get("NOESIS_PHONE_SCAN_CALIBRATION_SOLVER")
    if not executable:
        pytest.skip("native executable not configured")
    data, truth = synthetic_fixture()(
        -23_000_000, duration=14, skew_ns=12_000_000, exposure_ns=2_000_000
    )
    obs = {
        "resolution": data["resolution"],
        "point_timing": {"model": data["point_timing_model"]},
        "frames": [
            {**r, "points": r["pixels"], "rejection_reasons": []}
            for r in data["frames"]
        ],
    }
    if unsupported_camera2:
        _, template_rows, timing, bound = unsupported_direct_physical_fixture()
        rect = [0, 0, *data["resolution"]]
        camera = timing["recorder"]["camera"]
        camera.update(sensor_active_array_size=rect, sensor_pre_correction_active_array_size=rect)
        bound["signature"]["actual_camera2"]["crop_region"] = rect
        rows = []
        for frame in obs["frames"]:
            frame["timestamp_ns"] = frame["sensor_timestamp_ns"]
            rows.append({**template_rows[0], "sensor_timestamp_ns": frame["timestamp_ns"],
                         "crop_region": rect, "exposure_time_ns": 2_000_000,
                         "rolling_shutter_skew_ns": 12_000_000})
        _attach_point_timing(obs, rows, timing, bound)
        assert obs["point_timing"]["qualified"]
        for source, mapped in zip(data["frames"], obs["frames"]):
            np.testing.assert_allclose(mapped["corner_timestamps_ns"], source["corner_timestamps_ns"], rtol=0, atol=1)
            assert mapped["exposure_midpoint_ns"] == source["timestamp_ns"]
    ref = {
        "schema": CAMERA_SCHEMA,
        "K": data["K"],
        "D": [0] * 5,
        "resolution": data["resolution"],
        "binding": binding(),
        "quality": {"status": "qualified"},
    }
    run = _Run(
        CalibrationSettings(solver_path=Path(executable), timeout_s=300), None, None
    )
    streams = {k: data[k] for k in ("accel", "gyro")}
    payload, heldout, _, _ = _imu_input(
        obs, validate_calibration_request(request("imu")), ref, binding(), streams, run
    )
    result = _native_solve(payload, tmp_path, run)
    wanted, got = np.array(truth["T_imu_camera"]), np.array(result["T_imu_camera"])
    assert np.linalg.norm(got[:3, 3] - wanted[:3, 3]) < 0.001
    assert Rotation.from_matrix(got[:3, :3].T @ wanted[:3, :3]).magnitude() < 0.001
    assert abs(result["cam_time_offset_ns"] - truth["cam_time_offset_ns"]) < 100_000
    visual = _visual_holdout(payload, heldout, tmp_path, run)
    assert visual["visual_only"] is True and visual["status"] == "converged"
    validation = _heldout_imu(heldout, streams, result, visual)
    assert validation["translation_validation"]["qualified"], validation
    assert validation["time_offset_validation"]["qualified"], validation
    poisoned = copy.deepcopy(result)
    poisoned["T_imu_camera"][0][3] += 0.05
    rejected = _heldout_imu(heldout, streams, poisoned, visual)
    assert not rejected["translation_validation"]["qualified"]
    assert (
        "heldout_translation_disagrees_over_15_mm"
        in rejected["translation_validation"]["reason_codes"]
    )
    poisoned = copy.deepcopy(result)
    poisoned["cam_time_offset_ns"] += 5_000_000
    assert not _heldout_imu(heldout, streams, poisoned, visual)[
        "time_offset_validation"
    ]["qualified"]
    source = json.loads((tmp_path / "heldout_visual_input.json").read_text())
    assert "gyro" not in source and "accel" not in source
    assert result["accepted_for_metric_vio"] is False


def test_point_timing_requires_physical_geometry_and_maps_aspect_crop():
    obs = {
        "resolution": [1600, 900],
        "frames": [
            {
                "timestamp_ns": 1_000_000_000,
                "points": [[800, 0], [800, 450], [800, 899]],
            }
        ],
    }
    raw = [
        {
            "sensor_timestamp_ns": 1_000_000_000,
            "exposure_time_ns": 2_000_000,
            "rolling_shutter_skew_ns": 12_000_000,
            "sensor_pixel_mode": 0,
        }
    ]
    actual = {
        "active_physical_camera_id": "5",
        "crop_region": [0, 0, 4000, 3000],
        "distortion_correction_mode": 0,
        "zoom_ratio": 1,
        "ois_mode": 0,
        "eis_mode": 0,
        "rotate_and_crop_mode": 0,
    }
    bound = {"signature": {"actual_camera2": actual}}
    camera = {
        "id": "0",
        "sensor_active_array_size": [0, 0, 4000, 3000],
        "sensor_pre_correction_active_array_size": [0, 0, 4000, 3000],
    }
    _attach_point_timing(obs, raw, {"recorder": {"camera": camera}}, bound)
    assert not obs["point_timing"]["qualified"]
    camera["sensor_geometry_camera_id"] = "5"
    raw[0]["result_camera_id"] = "5"
    _attach_point_timing(obs, raw, {"recorder": {"camera": camera}}, bound)
    assert obs["point_timing"]["qualified"]
    assert obs["point_timing"]["encoded_viewport_height_active_rows"] == 2250
    assert obs["frames"][0]["corner_timestamps_ns"][:2] == [
        1_002_500_000,
        1_007_000_000,
    ]
    assert obs["frames"][0]["exposure_midpoint_ns"] == 1_007_000_000


def physical_result_fixture():
    actual = {"active_physical_camera_id": "5", "crop_region": [0, 0, 4000, 3000],
              "distortion_correction_mode": 0, "zoom_ratio": 1, "ois_mode": 0,
              "eis_mode": 0, "rotate_and_crop_mode": 0}
    physical = {**actual, "result_camera_id": "5", "sensor_timestamp_ns": 1_000_000_000,
                "exposure_time_ns": 2_000_000, "rolling_shutter_skew_ns": 12_000_000,
                "sensor_pixel_mode": 0}
    logical = {**actual, "result_camera_id": "0", "sensor_timestamp_ns": 1_000_000_000,
               "exposure_time_ns": 4_000_000, "rolling_shutter_skew_ns": 16_000_000,
               "sensor_pixel_mode": 0, "physical_capture_result": physical,
               "physical_capture_result_matches_logical_timestamp": True}
    camera = {"id": "0", "sensor_geometry_camera_id": "5",
              "sensor_active_array_size": [0, 0, 4000, 3000],
              "sensor_pre_correction_active_array_size": [0, 0, 4000, 3000]}
    observations = {"resolution": [1600, 900], "frames": [{"timestamp_ns": 1_000_000_000,
                    "points": [[800, 0], [800, 450], [800, 899]]}]}
    return observations, [logical], {"recorder": {"camera": camera}}, {"signature": {"actual_camera2": actual}}


def test_direct_physical_output_uses_its_own_timestamp_and_retains_logical_metadata():
    obs, raw, timing, bound = physical_result_fixture()
    physical = copy.deepcopy(raw[0]["physical_capture_result"])
    physical.update(metadata_source="physical_output_capture_result", output_physical_camera_id="5",
                    logical_capture_result={**raw[0], "sensor_timestamp_ns": 1_000_100_000})
    timing["recorder"]["camera"].update(id="5", logical_camera_id="0")
    original = copy.deepcopy(physical)
    _attach_point_timing(obs, [physical], timing, bound)
    assert obs["point_timing"]["qualified"] is True
    assert obs["point_timing"]["physical_capture_result_row_count"] == 0
    assert obs["frames"][0]["exposure_midpoint_ns"] == 1_007_000_000
    assert physical == original


def test_exact_nested_physical_result_supplies_exposure_without_relabeling_logical_row():
    obs, raw, timing, bound = physical_result_fixture()
    original = copy.deepcopy(raw)
    # Advisory false cannot override the directly verified exact integer match.
    raw[0]["physical_capture_result_matches_logical_timestamp"] = False
    _attach_point_timing(obs, raw, timing, bound)
    assert obs["point_timing"]["qualified"] is True
    assert obs["point_timing"]["physical_capture_result_row_count"] == 1
    assert obs["frames"][0]["corner_timestamps_ns"][:2] == [1_002_500_000, 1_007_000_000]
    assert obs["frames"][0]["exposure_midpoint_ns"] == 1_007_000_000
    raw[0]["physical_capture_result_matches_logical_timestamp"] = True
    assert raw == original


def unsupported_direct_physical_fixture():
    obs, rows, timing, _ = physical_result_fixture()
    row = copy.deepcopy(rows[0]["physical_capture_result"])
    row.update(metadata_source="physical_output_capture_result", output_physical_camera_id="5",
               distortion_correction_mode=None, lens_focal_length_mm=6.25,
               lens_focus_distance_diopters=1.2)
    camera = timing["recorder"]["camera"]
    camera.update(id="5", distortion_correction_request_key_available=False,
                  distortion_correction_result_key_available=False, distortion_correction_available_modes=None,
                  focus_control={"mode": "manual_locked", "physical_camera_id": "5",
                                 "output_routing_policy": "physical_camera_output_v1",
                                 "build_fingerprint": "phone-build"})
    timing["recorder"]["raw_device"] = {"build_fingerprint": "phone-build", "android_api_level": 37}
    manifest = {"device": {"id": "phone"}, "camera": {"id": "5", "orientation_deg": 0}}
    bound = _capture_binding(manifest, [row], obs["resolution"], timing)
    assert bound["qualified"]
    return obs, [row], timing, bound


@pytest.mark.parametrize("modes", [None, []])
def test_unsupported_control_has_separate_verified_physical_row_contract(modes):
    obs, rows, timing, bound = unsupported_direct_physical_fixture()
    timing["recorder"]["camera"]["distortion_correction_available_modes"] = modes
    original = copy.deepcopy((rows, timing, bound))
    _attach_point_timing(obs, rows, timing, bound)
    assert obs["point_timing"]["qualified"]
    assert obs["frames"][0]["corner_timestamps_ns"][:2] == [1_002_500_000, 1_007_000_000]
    assert obs["frames"][0]["exposure_midpoint_ns"] == 1_007_000_000
    evidence = obs["point_timing"]["distortion_control_evidence"]
    assert evidence["contract"] == "camera2_unsupported_control_same_physical_array_v1"
    assert evidence["reported_mode"] is None and evidence["assumed_off"] is False
    assert evidence["verified_direct_physical_result_count"] == 1
    assert (rows, timing, bound) == original
    # The historical optical binding alone still does not admit row mapping.
    assert bound["signature"]["distortion_control"]["row_mapping_qualified"] is False


@pytest.mark.parametrize("bad", ["request_available", "result_available", "unknown_request",
    "missing_modes", "supported_modes", "wrong_characteristics", "wrong_geometry", "array_mismatch",
    "nonzero_array_origin", "float_array", "logical_output", "unproven_routing", "wrong_output_id",
    "missing_result", "changed_result", "changed_crop", "changed_ois", "bool_ois", "changed_eis",
    "changed_zoom", "changed_rotation", "changed_pixel_mode", "fingerprint", "device_fingerprint",
    "api_missing", "unqualified_binding", "logical_binding", "different_focus_lens"])
def test_unsupported_row_contract_rejects_missing_or_contradictory_evidence(bad):
    obs, rows, timing, bound = unsupported_direct_physical_fixture()
    _attach_point_timing(obs, rows, timing, bound)
    assert obs["point_timing"]["qualified"]
    camera = timing["recorder"]["camera"]
    row = rows[0]
    if bad in {"request_available", "result_available", "unknown_request"}:
        key = "result" if bad == "result_available" else "request"
        camera[f"distortion_correction_{key}_key_available"] = None if bad == "unknown_request" else True
    elif bad == "missing_modes": camera.pop("distortion_correction_available_modes")
    elif bad == "supported_modes": camera["distortion_correction_available_modes"] = [0, 1, 2]
    elif bad == "wrong_characteristics": camera["id"] = "0"
    elif bad == "wrong_geometry": camera["sensor_geometry_camera_id"] = "0"
    elif bad == "array_mismatch": camera["sensor_pre_correction_active_array_size"][2] -= 1
    elif bad == "nonzero_array_origin":
        camera["sensor_active_array_size"][0] = camera["sensor_pre_correction_active_array_size"][0] = 2
    elif bad == "float_array": camera["sensor_active_array_size"][0] = 0.0
    elif bad == "logical_output": row["result_camera_id"] = "0"
    elif bad == "unproven_routing": row["metadata_source"] = "logical_capture_result"
    elif bad == "wrong_output_id": row["output_physical_camera_id"] = "2"
    elif bad == "missing_result": row.pop("distortion_correction_mode")
    elif bad == "changed_result": row["distortion_correction_mode"] = 1
    elif bad == "changed_crop": row["crop_region"] = [0, 0, 2000, 1500]
    elif bad == "fingerprint": camera["focus_control"]["build_fingerprint"] = "different-build"
    elif bad == "device_fingerprint": timing["recorder"]["raw_device"]["build_fingerprint"] = "different-build"
    elif bad == "api_missing": timing["recorder"]["raw_device"].pop("android_api_level")
    elif bad == "unqualified_binding": bound["qualified"] = False
    elif bad == "logical_binding": bound["signature"]["camera_id"] = "0"
    elif bad == "different_focus_lens": camera["focus_control"]["physical_camera_id"] = "2"
    else:
        key, value = {"changed_ois": ("ois_mode", 1), "bool_ois": ("ois_mode", False),
                      "changed_eis": ("eis_mode", 1), "changed_zoom": ("zoom_ratio", 2),
                      "changed_rotation": ("rotate_and_crop_mode", 1),
                      "changed_pixel_mode": ("sensor_pixel_mode", 1)}[bad]
        row[key] = value
    _attach_point_timing(obs, rows, timing, bound)
    assert not obs["point_timing"]["qualified"], bad
    assert obs["point_timing"]["reason_codes"]
    assert "corner_timestamps_ns" not in obs["frames"][0]
    assert "exposure_midpoint_ns" not in obs["frames"][0]
    assert "distortion_control_evidence" not in obs["point_timing"]


@pytest.mark.parametrize("bad", ["missing", "wrong_id", "timestamp", "float_timestamp", "distortion", "missing_distortion",
                                 "ois", "eis", "rotation", "zoom", "crop", "pixel_mode", "exposure", "skew",
                                 "geometry", "logical_distortion_unknown"])
def test_nested_physical_metadata_cannot_bypass_exact_row_mapping_gates(bad):
    obs, raw, timing, bound = physical_result_fixture()
    _attach_point_timing(obs, raw, timing, bound)
    assert obs["point_timing"]["qualified"]
    physical = raw[0]["physical_capture_result"]
    if bad == "missing":
        raw[0].pop("physical_capture_result")
    elif bad == "wrong_id":
        physical["result_camera_id"] = "6"
    elif bad == "timestamp":
        physical["sensor_timestamp_ns"] += 1
    elif bad == "float_timestamp":
        physical["sensor_timestamp_ns"] = float(physical["sensor_timestamp_ns"])
    elif bad == "missing_distortion":
        physical.pop("distortion_correction_mode")
    elif bad == "geometry":
        timing["recorder"]["camera"]["sensor_geometry_camera_id"] = "0"
    elif bad == "logical_distortion_unknown":
        bound["signature"]["actual_camera2"]["distortion_correction_mode"] = None
    else:
        key, value = {
            "distortion": ("distortion_correction_mode", 1), "ois": ("ois_mode", 1),
            "eis": ("eis_mode", 1), "rotation": ("rotate_and_crop_mode", 1),
            "zoom": ("zoom_ratio", 2), "crop": ("crop_region", [0, 0, 4000, 2998]),
            "pixel_mode": ("sensor_pixel_mode", 1), "exposure": ("exposure_time_ns", None),
            "skew": ("rolling_shutter_skew_ns", -1)}[bad]
        physical[key] = value
    _attach_point_timing(obs, raw, timing, bound)
    assert not obs["point_timing"]["qualified"]
    assert obs["point_timing"]["reason_codes"]
    assert "corner_timestamps_ns" not in obs["frames"][0]
    assert "exposure_midpoint_ns" not in obs["frames"][0]


def test_camera_profile_and_explicit_physical_attestation():
    obs, _, _ = observations()
    req = request()
    req["board_geometry_confirmed"] = True
    result = fit_camera_observations(obs, req, binding())
    assert result["physical_board_scale_verified"] is True
    assert result["physical_board_scale_provenance"] == "user_attestation"
    profile = camera_profile(result)
    assert profile["distortion"] == result["D"]
    assert profile["intrinsics"]["fx"] == result["K"][0][0]
    assert profile["binding"] == result["binding"]
    result["quality"]["status"] = "rejected"
    with pytest.raises(CalibrationError):
        camera_profile(result)


def test_camera_binding_for_long_walk_checks_every_raw_frame_without_video(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: pytest.fail("binding helper must not probe/decode video"),
    )
    count = 6301
    settings = {
        "active_physical_camera_id": "5",
        "lens_focal_length_mm": 6.25,
        "lens_focus_distance_diopters": 1.2,
        "zoom_ratio": 1.0,
        "crop_region": [0, 0, 640, 480],
        "ois_mode": 0,
        "eis_mode": 0,
        "rotate_and_crop_mode": 0,
        "distortion_correction_mode": 0,
    }
    rows = [
        {
            **settings,
            "frame_number": i,
            "sensor_timestamp_ns": 1_000_000_000 + i * 33_333_333,
        }
        for i in range(count)
    ]
    (tmp_path / "camera.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (tmp_path / "encoder.csv").write_text(
        "encoded_index,encoded_pts_us,flags,size_bytes\n"
        + "".join(
            f"{i},{r['sensor_timestamp_ns'] // 1000},0,100\n"
            for i, r in enumerate(rows)
        )
    )
    (tmp_path / "times.csv").write_text(
        "encoded_index,encoded_pts_us,frame_number,timestamp_ns\n"
        + "".join(
            f"{i},{r['sensor_timestamp_ns'] // 1000},{i},{r['sensor_timestamp_ns']}\n"
            for i, r in enumerate(rows)
        )
    )
    raw = {
        "schema": "noesis.phone_capture.android_result.v1",
        "camera": {
            "id": "0",
            "width": 640,
            "height": 480,
            "focus_control": {"mode": "locked"},
        },
        "dropped_metadata_records": 0,
        "timing": {"exact_frame_association_verified": True},
    }
    (tmp_path / "result.json").write_text(json.dumps(raw))
    report = {
        "schema": "noesis.phone_capture.v1",
        "video": {
            "camera_acquisition_timestamp_verified": True,
            "frame_count": count,
            "duration_s": 210,
        },
        "android_capture": {"camera_acquisition_timestamp_verified": True},
        "manifest": {
            "device": {"id": "phone"},
            "camera": {"id": "0", "orientation_deg": 0},
            "video": {
                "encoded_resolution_px": [640, 480],
                "frame_timestamps_path": "times.csv",
            },
            "android_capture": {
                "capture_result_path": "result.json",
                "camera_results_path": "camera.jsonl",
                "encoder_pts_path": "encoder.csv",
            },
        },
    }
    (tmp_path / "capture_import.json").write_text(json.dumps(report))
    before = camera_binding_for_capture(tmp_path)
    assert before["qualified"]
    rows[-1]["lens_focus_distance_diopters"] = 1.3
    (tmp_path / "camera.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    after = camera_binding_for_capture(tmp_path)
    assert not after["qualified"]
    assert "changing_lens_focus_distance_diopters" in after["reason_codes"]
    rows[-1]["sensor_timestamp_ns"] += 1000
    (tmp_path / "camera.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    with pytest.raises(CalibrationError, match="association"):
        camera_binding_for_capture(tmp_path)
