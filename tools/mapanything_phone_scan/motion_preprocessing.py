"""Bound short-walk camera/IMU models to immutable native recordings.

The OpenVINS consumer uses each image's centre exposure time. It remains a
global-shutter approximation, not rolling-shutter compensation. Its short
profile must pass separate target-motion validation before ordinary VIO use.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CONSUMER_VERSION = "roomwalk.openvins.short_walk.v1"
PROFILE_SCHEMA = "roomwalk.motion_profile.v1"
MAXIMUM_WALK_SECONDS = 300
ANALYSIS_MAX_SIDE = 1280


class MotionProfileError(ValueError):
    pass


def json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def checked_corrections(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MotionProfileError("The camera–IMU result is missing sensor corrections")
    checked = {}
    for sensor in ("accelerometer", "gyroscope"):
        matrix = np.asarray(value.get(sensor + "_matrix"), dtype=float)
        bias = np.asarray(value.get(sensor + "_bias"), dtype=float)
        if (matrix.shape != (3, 3) or bias.shape != (3,)
                or not np.isfinite(matrix).all() or not np.isfinite(bias).all()
                or np.linalg.det(matrix) <= 0 or np.linalg.cond(matrix) > 100):
            raise MotionProfileError(f"The {sensor} correction is invalid")
        checked[sensor + "_matrix"] = matrix.tolist()
        checked[sensor + "_bias"] = bias.tolist()
    return checked


def correct_samples(values: Any, corrections: Mapping[str, Any], sensor: str) -> np.ndarray:
    checked = checked_corrections(corrections)
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise MotionProfileError("Sensor samples must be finite three-axis observations")
    return values @ np.asarray(checked[sensor + "_matrix"]).T - np.asarray(checked[sensor + "_bias"])


def corrected_noise(noise: Mapping[str, Any], corrections: Mapping[str, Any]) -> dict[str, float]:
    """Conservative scalar covariance bound after the actual matrix correction."""
    checked = checked_corrections(corrections)
    result = {}
    for sensor in ("accelerometer", "gyroscope"):
        gain = float(np.linalg.norm(np.asarray(checked[sensor + "_matrix"]), ord=2))
        for suffix in ("noise_density", "random_walk"):
            key = sensor + "_" + suffix
            value = noise.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise MotionProfileError("The profile needs four finite positive sensor-noise model parameters")
            result[key] = float(value) * gain
    return result


def _read(path: Path, maximum: int) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise MotionProfileError("Native calibration evidence is missing or exceeds its bound")
    def reject(value):
        raise MotionProfileError("Nonfinite JSON value in calibration evidence")
    return json.loads(path.read_text(), parse_constant=reject)


def bind_profile_to_capture(capture_dir: Path, capture_report: Mapping[str, Any], profile: Mapping[str, Any], *, validation: bool = False) -> dict[str, Any]:
    """Return a derived report, never overwrite the imported capture or samples."""
    from .roomwalk_calibration import camera_binding_for_capture, _attach_point_timing, _member

    if profile.get("schema") != PROFILE_SCHEMA or profile.get("consumer_version") != CONSUMER_VERSION:
        raise MotionProfileError("The motion profile needs to be checked by this version of RoomWalk")
    if not validation and profile.get("status") != "ready_for_short_walks":
        raise MotionProfileError("Check the motion profile before using it for a room walk")
    if capture_report.get("schema") != "noesis.phone_capture.v1":
        raise MotionProfileError("Motion profiles require an imported native recording")
    binding = camera_binding_for_capture(capture_dir)
    if not binding.get("qualified") or binding != profile.get("binding"):
        raise MotionProfileError("This recording's lens, focus, crop or device does not match the selected motion profile")
    source = capture_report["manifest"]
    evidence = source.get("android_capture")
    if not isinstance(evidence, Mapping):
        raise MotionProfileError("This profile requires the RoomWalk native camera and motion recorder")
    times = _read(capture_dir / "video_timestamps_ns.json", 512 * 1024)
    if not isinstance(times, list) or not 2 <= len(times) <= 12000 or any(type(t) is not int for t in times):
        raise MotionProfileError("The recording has invalid or excessive camera timestamps")
    if any(a >= b for a, b in zip(times, times[1:])):
        raise MotionProfileError("Camera timestamps are not strictly increasing")
    duration = (times[-1] - times[0]) * 1e-9
    # A little boundary allowance includes the frame on which the user stops.
    if duration > MAXIMUM_WALK_SECONDS + 2:
        raise MotionProfileError("This motion profile is for walks up to five minutes; keep this longer take as reconstruction evidence")
    rows = {}
    path = _member(capture_dir, evidence["camera_results_path"], 128 * 1024**2)
    if path.is_symlink() or path.stat().st_size > 128 * 1024**2:
        raise MotionProfileError("Camera timing evidence exceeds its bound")
    with path.open() as handle:
        for line in handle:
            if len(line) > 65536 or len(rows) >= 21000:
                raise MotionProfileError("Camera timing evidence exceeds its row bound")
            row = json.loads(line)
            stamp = row.get("sensor_timestamp_ns")
            if type(stamp) is not int or stamp in rows:
                raise MotionProfileError("Camera timing evidence contains ambiguous timestamps")
            rows[stamp] = row
    try:
        selected = [rows[t] for t in times]
    except KeyError as exc:
        raise MotionProfileError("Camera timing no longer matches the imported recording") from exc
    raw_result = _read(_member(capture_dir, evidence["capture_result_path"], 512 * 1024), 512 * 1024)
    timing = deepcopy(capture_report.get("android_capture") or {})
    timing.setdefault("recorder", {})["camera"] = raw_result["camera"]
    timing["recorder"]["raw_device"] = raw_result.get("device", {})
    width, height = binding["signature"]["resolution"]
    observations = {"resolution": [width, height], "frames": [
        {"timestamp_ns": t, "points": [[width / 2, height / 2]]} for t in times
    ]}
    _attach_point_timing(observations, selected, timing, binding)
    point_timing = observations["point_timing"]
    if not point_timing.get("qualified"):
        reasons = ", ".join(point_timing.get("reason_codes") or [])
        raise MotionProfileError("This recording cannot reproduce the calibrated camera exposure timing: " + reasons)
    expected_timing = profile["point_timing"]
    for key in ("model", "active_array", "crop", "encoded_viewport_top_active_rows", "encoded_viewport_height_active_rows", "sensor_geometry_camera_id"):
        if point_timing.get(key) != expected_timing.get(key):
            raise MotionProfileError("The camera readout geometry changed; repeat the short camera–IMU step")
    pose_times = [row["exposure_midpoint_ns"] for row in observations["frames"]]
    if any(a >= b for a, b in zip(pose_times, pose_times[1:])):
        raise MotionProfileError("Image exposure times do not preserve camera-frame order")
    corrections = checked_corrections(profile.get("imu_corrections"))
    noise = corrected_noise(profile["noise"], corrections)
    normalized = _read(capture_dir / "imu_normalized.json", 128 * 1024**2)
    if not isinstance(normalized, Mapping) or not {"accel", "gyro"}.issubset(normalized):
        raise MotionProfileError("The native recording needs separate accelerometer and gyroscope streams")
    start, end = [], []
    for sensor in ("accel", "gyro"):
        values = normalized.get(sensor)
        if not isinstance(values, list) or not 2 <= len(values) <= 300000:
            raise MotionProfileError("The motion recording is missing a bounded sensor stream")
        stamps = [row["timestamp_ns"] for row in values]
        if any(type(t) is not int for t in stamps) or any(a >= b for a, b in zip(stamps, stamps[1:])):
            raise MotionProfileError("Sensor timestamps are not strictly increasing")
        if max(b - a for a, b in zip(stamps, stamps[1:])) > 50_000_000:
            raise MotionProfileError("A motion-sensor gap is too large for this profile")
        start.append(stamps[0])
        end.append(stamps[-1])
    offset = profile["imu_to_camera_offset_ns"]
    if type(offset) is not int or abs(offset) > 100_000_000:
        raise MotionProfileError("The camera–IMU timing offset is invalid")
    camera_imu_start, camera_imu_end = pose_times[0] - offset, pose_times[-1] - offset
    if max(start) > camera_imu_start or min(end) < camera_imu_end:
        raise MotionProfileError("Motion samples do not cover the exposure-corrected video interval")
    gyro = correct_samples([r["si"] for r in normalized["gyro"]], corrections, "gyroscope")
    gyro_t = np.asarray([r["timestamp_ns"] for r in normalized["gyro"]], dtype=np.int64)
    # Interpolate relative to an integer epoch to avoid rounding hardware nanoseconds.
    epoch = int(gyro_t[0])
    speed = np.linalg.norm(gyro, axis=1)
    camera_speed = np.interp((np.asarray(pose_times, dtype=np.int64) - offset - epoch) * 1e-9, (gyro_t - epoch) * 1e-9, speed)
    physical_id = point_timing["sensor_geometry_camera_id"]
    physical_rows = [r if r.get("result_camera_id", raw_result["camera"].get("id")) == physical_id
                     else r["physical_capture_result"] for r in selected]
    exposures = np.asarray([r["exposure_time_ns"] for r in physical_rows], dtype=float) * 1e-9
    skews = np.asarray([r["rolling_shutter_skew_ns"] for r in physical_rows], dtype=float) * 1e-9
    motion_envelope = {
        "maximum_exposure_s": float(np.max(exposures)), "maximum_readout_s": float(np.max(skews)),
        "maximum_rotation_during_exposure_rad": float(np.max(camera_speed * exposures)),
        "maximum_rotation_during_readout_rad": float(np.max(camera_speed * skews)),
    }
    if not validation:
        accepted_envelope = profile.get("validated_motion_envelope") or {}
        for key in ("maximum_rotation_during_exposure_rad", "maximum_rotation_during_readout_rad"):
            maximum = accepted_envelope.get(key)
            if not isinstance(maximum, (float, int)) or not math.isfinite(maximum) or maximum <= 0:
                raise MotionProfileError("The motion profile has no validated camera-motion envelope")
            if motion_envelope[key] > maximum:
                raise MotionProfileError("Phone rotation or exposure blur exceeds the checked short-profile range; use slower movement and better light")
    report = deepcopy(capture_report)
    manifest = report["manifest"]
    camera = profile["camera"]
    manifest["camera"].update(deepcopy(camera), intrinsics_source="roomwalk_short_motion_profile")
    manifest["imu"].update(noise=noise, corrections=corrections)
    manifest["extrinsics"] = {"T_imu_camera": deepcopy(profile["T_imu_camera"]), "convention": "T_imu_camera maps camera coordinates to corrected IMU coordinates"}
    manifest["clocks"]["imu_to_camera_offset_ns"] = offset
    report["calibration"].update(camera_intrinsics=True, camera_to_imu_extrinsics=True, time_offset=True, imu_noise=True,
                                 distortion_supported=True, camera_geometry=True, complete_for_metric_vio=not validation,
                                 imu_noise_calibrated=profile.get("imu_noise_calibrated") is True,
                                 noise_model_status=profile["noise_model_status"])
    report["metric_vio_allowed"] = not validation
    report["coverage"].update(video_start_ns=pose_times[0], video_end_ns=pose_times[-1], imu_start_ns=max(start) + offset,
                               imu_end_ns=min(end) + offset, starts_before_video=True, ends_after_video=True)
    report["short_session"] = {
        "consumer_version": CONSUMER_VERSION, "profile_id": profile["profile_id"], "validation_run": validation,
        "maximum_duration_s": MAXIMUM_WALK_SECONDS, "analysis_max_side": ANALYSIS_MAX_SIDE,
        "source_and_pose_times_ns": list(zip(times, pose_times)), "point_timing": point_timing,
        "image_motion_model": "centre_timed_global_shutter_approximation", "rolling_shutter_compensated": False,
        "imu_correction_equation": "corrected = matrix * raw - bias", "vendor_bias_subtracted": False,
        "noise_transformation": "scalar density and drift multiplied by correction matrix spectral norm",
        "motion_envelope": motion_envelope, "noise_model_status": profile["noise_model_status"],
        "validated_motion_envelope": deepcopy(profile.get("validated_motion_envelope")),
        "profile_validation_sha256": profile.get("validation_sha256"),
        "noise_provenance": deepcopy(profile.get("noise_provenance", {})),
        "base_capture_report_sha256": json_sha(capture_report),
    }
    return report
