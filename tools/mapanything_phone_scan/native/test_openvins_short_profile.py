"""Opt-in functional checks of the rebuilt bridge against retained EuRoC data.

Set NOESIS_OPENVINS_SHORT_TEST_EXECUTABLE and NOESIS_OPENVINS_TEST_EUROC_DIR
explicitly. This never selects a service binary or changes the source fixture.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tools.mapanything_phone_scan.vio import (
    SHORT_CONSUMER_VERSION,
    POSE_TIME_REFERENCE,
    VIOError,
    _replace_yaml_key,
    validate_vio_profile_validation_result,
    validate_vio_result,
)


@pytest.fixture(scope="module")
def native_fixture(tmp_path_factory):
    executable = os.environ.get("NOESIS_OPENVINS_SHORT_TEST_EXECUTABLE")
    source = os.environ.get("NOESIS_OPENVINS_TEST_EUROC_DIR")
    if not executable or not source:
        pytest.skip("explicit rebuilt bridge and retained EuRoC fixture not configured")
    executable, source = Path(executable), Path(source)
    assert executable.is_file() and source.is_dir()
    work = tmp_path_factory.mktemp("native-short-profile")
    root = work / "mav0"
    (root / "cam0").mkdir(parents=True)
    (root / "imu0").mkdir()
    (root / "logs").mkdir()
    # Reuse exactly the source image directory; no source file is rewritten.
    (root / "cam0/data").symlink_to(source / "cam0/data", target_is_directory=True)
    source_camera = source / "cam0/data.csv"
    before = hashlib.sha256(source_camera.read_bytes()).hexdigest()
    with source_camera.open() as handle:
        rows = [row for row in csv.reader(handle) if row and not row[0].startswith("#")][:500]
    assert len(rows) == 500
    shift = 7_000_007
    # This is an artificial clock-reference shift, not phone exposure evidence.
    # Its opposite calibrated offset preserves the actual camera/IMU interval.
    with (root / "cam0/data.csv").open("w") as handle:
        handle.writelines(f"{int(stamp)+shift},{name}\n" for stamp, name in rows)
    mapping = root / "source_frame_mapping.csv"
    with mapping.open("w") as handle:
        handle.write("source_frame_index,capture_time_ns,pose_time_ns\n")
        handle.writelines(f"{i},{int(stamp)},{int(stamp)+shift}\n" for i, (stamp, _) in enumerate(rows))
    (root / "imu0/data.csv").symlink_to(source / "imu0/data.csv")
    camera = (source / "kalibr_imucam_chain.yaml").read_text()
    assert "timeshift_cam_imu:" not in camera
    camera = camera.replace("cam0:\n", f"cam0:\n  timeshift_cam_imu: {-shift*1e-9:.17g}\n", 1)
    (root / "kalibr_imucam_chain.yaml").write_text(camera)
    (root / "kalibr_imu_chain.yaml").write_bytes((source / "kalibr_imu_chain.yaml").read_bytes())
    config = (source / "estimator_config.yaml").read_text()
    for key, value in {
        "calib_cam_extrinsics": "false", "calib_cam_intrinsics": "false", "calib_cam_timeoffset": "false",
        "calib_imu_intrinsics": "false", "calib_imu_g_sensitivity": "false", "init_dyn_mle_opt_calib": "false",
        "num_opencv_threads": "4", "init_dyn_mle_max_threads": "2", "multi_threading_subs": "false",
        "multi_threading_pubs": "false", "downsample_cameras": "false", "use_stereo": "false", "max_cameras": "1",
        "relative_config_imu": '"kalibr_imu_chain.yaml"', "relative_config_imucam": '"kalibr_imucam_chain.yaml"',
        "record_timing_filepath": '"logs/timing.txt"',
    }.items():
        config = _replace_yaml_key(config, key, value)
    (root / "estimator_config.yaml").write_text(config)
    command = [str(executable), "--capture-dir", str(root), "--config", str(root / "estimator_config.yaml"),
               "--capture-id", "benchmark-short-profile", "--camera-sensor-id", "cam0", "--time-domain", "euroc_ns",
               "--short-session-consumer", SHORT_CONSUMER_VERSION, "--profile-id", "synthetic-clock-reference-check",
               "--frame-mapping", str(mapping)]
    return {"work": work, "root": root, "command": command, "rows": rows, "shift": shift,
            "source_camera": source_camera, "before": before}


def _run(fixture, mode, label):
    result_path = fixture["work"] / f"{label}.json"
    command = fixture["command"] + ["--mode", mode, "--output", str(result_path)]
    with (fixture["work"] / f"{label}.stdout.log").open("w") as stdout, (fixture["work"] / f"{label}.stderr.log").open("w") as stderr:
        completed = subprocess.run(command, cwd=fixture["root"], stdout=stdout, stderr=stderr, timeout=120,
                                   env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
    return completed, result_path


def test_native_profile_validation_functionally_preserves_exact_source_and_pose_times(native_fixture):
    completed, path = _run(native_fixture, "profile_validation", "profile-validation")
    assert completed.returncode == 0, (native_fixture["work"] / "profile-validation.stderr.log").read_text()[-3000:]
    result = validate_vio_profile_validation_result(json.loads(path.read_text()))
    assert result["accepted_for_metric_vio"] is False
    assert result["frame"]["pose_time_reference"] == POSE_TIME_REFERENCE
    assert result["quality"]["initialized"] and len(result["poses"]) > 200
    assert result["fixed_calibration"]["imu_to_camera_offset_ns"] == native_fixture["shift"]
    original_times = [int(row[0]) for row in native_fixture["rows"]]
    for row in result["poses"]:
        assert row["capture_time_ns"] == original_times[row["source_frame_index"]]
        assert row["pose_time_ns"] == row["capture_time_ns"] + native_fixture["shift"]
        assert np.isfinite(row["T_vio_world_camera"]).all()
        assert np.isfinite(row["covariance"]).all()
    with pytest.raises(VIOError):
        validate_vio_result(result)
    assert hashlib.sha256(native_fixture["source_camera"].read_bytes()).hexdigest() == native_fixture["before"]


def test_native_ordinary_short_run_rejects_low_coverage_benchmark(native_fixture):
    # This retained subset initializes late. Execution is useful validation
    # evidence, but its ~63% full-camera coverage must not admit an ordinary walk.
    completed, path = _run(native_fixture, "metric_vio", "ordinary-quality-rejection")
    assert completed.returncode != 0
    assert "short-profile runtime quality failed" in (native_fixture["work"] / "ordinary-quality-rejection.stderr.log").read_text()
    assert not path.exists()


def test_native_short_consumer_rejects_incomplete_mapping_before_estimation(native_fixture):
    original = native_fixture["root"] / "source_frame_mapping.csv"
    changed = native_fixture["work"] / "incomplete.csv"
    changed.write_text("\n".join(original.read_text().splitlines()[:-1]) + "\n")
    fixture = dict(native_fixture, command=list(native_fixture["command"]))
    fixture["command"][fixture["command"].index("--frame-mapping") + 1] = str(changed)
    completed, path = _run(fixture, "profile_validation", "bad-mapping")
    assert completed.returncode != 0 and not path.exists()
    assert "frame mapping is incomplete" in (fixture["work"] / "bad-mapping.stderr.log").read_text()
