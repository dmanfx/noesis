from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.mapanything_phone_scan.imu_motion import load_native_motion


def _capture(root: Path, *, rotation: np.ndarray | None = None, gap: bool = False):
    rotation = np.eye(3) if rotation is None else rotation
    data = {}
    for name, vector in (("gyro", [0.0, 0.0, 1.0]), ("accel", [0.0, 9.81, 0.0])):
        data[name] = [{"timestamp_ns": t, "si": (rotation @ vector).tolist()}
                      for t in range(1_000_000_000, 1_500_000_001, 5_000_000)
                      if not (gap and name == "gyro" and 1_180_000_000 < t < 1_270_000_000)]
    (root / "imu_normalized.json").write_text(json.dumps(data))
    (root / "camera_results.jsonl").write_text(json.dumps({
        "sensor_timestamp_ns": 1_250_000_000, "exposure_time_ns": 10_000_000})+'\n')
    sensors = {name: {"units": unit, "timestamp_source": "android_elapsed_realtime_ns",
                      "axes": "android_device_x_right_y_up_z_out_of_screen"}
               for name, unit in (("accelerometer", "m/s^2"), ("gyroscope", "rad/s"))}
    return {"android_capture": {"camera_acquisition_timestamp_verified": True,
                                 "recorder": {"sensors": sensors}},
            "manifest": {"capture_id": "test", "clocks": {
                "camera_domain": "android.elapsedRealtimeNanos",
                "imu_domain": "android.elapsedRealtimeNanos"}}}


def test_motion_is_rotation_invariant_and_does_not_remove_gravity(tmp_path):
    report = _capture(tmp_path)
    first, _ = load_native_motion(tmp_path, report)
    first_row = first.frame(1_250_000_000)
    _capture(tmp_path, rotation=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))
    second, summary = load_native_motion(tmp_path, report)
    assert first_row == second.frame(1_250_000_000)
    assert first_row["rotation_during_exposure_estimate_rad"] == pytest.approx(.01)
    assert first_row["acceleration_m_s2"]["p50"] == pytest.approx(9.81)
    assert first_row["visual_quality_penalty"] == pytest.approx(.08)
    assert summary["metric_pose_authority"] is False
    assert summary["camera_imu_offset_measured"] is False


def test_missing_sample_coverage_never_changes_visual_score(tmp_path):
    report = _capture(tmp_path, gap=True)
    motion, _ = load_native_motion(tmp_path, report)
    row = motion.frame(1_250_000_000)
    assert row["status"] == "unavailable"
    assert "visual_quality_penalty" not in row
    assert motion.frame(1_500_000_000)["status"] == "unavailable"


def test_browser_or_unverified_clock_cannot_use_native_motion(tmp_path):
    assert load_native_motion(tmp_path, {})[0] is None
    report = _capture(tmp_path)
    report["android_capture"]["camera_acquisition_timestamp_verified"] = False
    motion, summary = load_native_motion(tmp_path, report)
    assert motion is None
    assert summary["status"] == "unavailable"


def test_nonfinite_sensor_input_is_preserved_but_not_used(tmp_path):
    report = _capture(tmp_path)
    path = tmp_path / 'imu_normalized.json'
    value = json.loads(path.read_text())
    value['gyro'][20]['si'][1] = float('nan')
    path.write_text(json.dumps(value))
    assert load_native_motion(tmp_path, report)[0] is None
    assert 'NaN' in path.read_text()
