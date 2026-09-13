"""Use native IMU motion in view selection without inventing metric calibration.

Angular-speed and acceleration magnitudes are invariant to sensor rotation.
They can rank camera motion without camera/IMU extrinsics or gravity removal.
The generous timestamp neighborhood is explicitly a diagnostic window, not a
measured camera/IMU offset. No IMU position or camera pose is produced here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "noesis.phone_capture.imu_motion.v1"
MAX_EVIDENCE_BYTES = 128 * 1024 * 1024
MAX_ROWS = 200_000
WINDOW_NS = 50_000_000
MAX_GAP_NS = 50_000_000


class MotionEvidenceError(ValueError):
    pass


def _read(path: Path) -> bytes:
    if path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise MotionEvidenceError(f"{path.name} exceeds the motion-analysis size limit")
    return path.read_bytes()


class NativeMotion:
    def __init__(self, capture_dir: Path, report: dict[str, Any]):
        native = report.get("android_capture") or {}
        manifest = report.get("manifest") or {}
        clocks = manifest.get("clocks") or {}
        sensors = (native.get("recorder") or {}).get("sensors") or {}
        if (native.get("camera_acquisition_timestamp_verified") is not True
                or clocks.get("camera_domain") != "android.elapsedRealtimeNanos"
                or clocks.get("imu_domain") != "android.elapsedRealtimeNanos"):
            raise MotionEvidenceError("native camera/IMU acquisition clocks are not verified")
        for key, unit in (("accelerometer", "m/s^2"), ("gyroscope", "rad/s")):
            sensor = sensors.get(key) or {}
            if (sensor.get("units") != unit
                    or sensor.get("timestamp_source") != "android_elapsed_realtime_ns"
                    or sensor.get("axes") != "android_device_x_right_y_up_z_out_of_screen"):
                raise MotionEvidenceError(f"{key} units, axes, or clock are not verified")
        raw = _read(capture_dir / "imu_normalized.json")
        payload = json.loads(raw)
        self.streams = {}
        for key in ("gyro", "accel"):
            rows = payload[key]
            if not 2 <= len(rows) <= MAX_ROWS:
                raise MotionEvidenceError(f"{key} sample count is outside the analysis limit")
            ts = np.asarray([int(row["timestamp_ns"]) for row in rows], dtype=np.int64)
            values = np.asarray([row["si"] for row in rows], dtype=np.float64)
            if (values.shape != (len(rows), 3) or not np.isfinite(values).all()
                    or np.any(np.diff(ts) <= 0)):
                raise MotionEvidenceError(f"{key} samples are malformed")
            with np.errstate(over="ignore", invalid="ignore"):
                magnitudes = np.linalg.norm(values, axis=1)
            if not np.isfinite(magnitudes).all():
                raise MotionEvidenceError(f"{key} sample magnitudes are not finite")
            self.streams[key] = ts, magnitudes
        camera_raw = _read(capture_dir / "camera_results.jsonl")
        self.camera = {}
        for line in camera_raw.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            stamp = int(row["sensor_timestamp_ns"])
            if stamp in self.camera or len(self.camera) >= MAX_ROWS:
                raise MotionEvidenceError("camera result timestamps are duplicated or excessive")
            self.camera[stamp] = row
        self.summary = {
            "schema": SCHEMA, "status": "available", "capture_id": manifest.get("capture_id"),
            "use": "bounded_soft_preference_for_less_camera_rotation_during_exposure",
            "metric_pose_authority": False, "camera_imu_offset_measured": False,
            "clock_semantics": "native_acquisition_clock_with_unmeasured_residual_offset",
            "timing_neighborhood_ns": WINDOW_NS, "maximum_sample_gap_ns": MAX_GAP_NS,
            "acceleration_semantics": "magnitude_including_gravity_and_sensor_bias",
            "gyro_semantics": "raw_angular_speed_magnitude_including_sensor_bias",
            "maximum_quality_penalty": 0.10,
            "rotation_penalty_half_scale_rad": 0.005,
            "evidence": {
                "imu": "capture/imu_normalized.json",
                "imu_sha256": hashlib.sha256(raw).hexdigest(),
                "camera_results": "capture/camera_results.jsonl",
                "camera_results_sha256": hashlib.sha256(camera_raw).hexdigest(),
            },
        }

    def frame(self, timestamp_ns: int | None) -> dict[str, Any]:
        if timestamp_ns is None or timestamp_ns not in self.camera:
            return {"status": "unavailable", "reason": "no exact native camera result"}
        row = self.camera[timestamp_ns]
        exposure = row.get("exposure_time_ns")
        if not isinstance(exposure, int) or not 0 < exposure <= 1_000_000_000:
            return {"status": "unavailable", "reason": "camera exposure is unavailable"}
        start = timestamp_ns - WINDOW_NS
        end = timestamp_ns + exposure + WINDOW_NS
        stats: dict[str, Any] = {}
        for key, (ts, magnitudes) in self.streams.items():
            left = int(np.searchsorted(ts, start, side="right")) - 1
            right = int(np.searchsorted(ts, end, side="left"))
            if left < 0 or right >= len(ts):
                return {"status": "unavailable", "reason": f"{key} does not bracket the motion window"}
            times = ts[left:right + 1]
            if np.max(np.diff(times)) > MAX_GAP_NS:
                return {"status": "unavailable", "reason": f"{key} has a gap in the motion window"}
            values = magnitudes[left:right + 1]
            stats[key] = {"sample_count": len(values), "p50": float(np.median(values)),
                          "p90": float(np.percentile(values, 90)), "max": float(np.max(values))}
        exposure_rotation = stats["gyro"]["p90"] * exposure / 1e9
        # This bounded heuristic supplements visual sharpness and connectivity;
        # it never rejects a frame or overrides an overlap-repair requirement.
        penalty = 0.10 * (1.0 - 1.0 / (1.0 + (exposure_rotation / 0.005) ** 2))
        return {
            "status": "available", "capture_time_ns": timestamp_ns,
            "window_start_ns": start, "window_end_ns": end, "exposure_time_ns": exposure,
            "gyro_rad_s": stats["gyro"], "acceleration_m_s2": stats["accel"],
            "rotation_during_exposure_estimate_rad": exposure_rotation,
            "visual_quality_penalty": penalty, "metric_pose_authority": False,
        }


def load_native_motion(capture_dir: Path, report: dict[str, Any]) -> tuple[NativeMotion | None, dict[str, Any]]:
    if not isinstance(report.get("android_capture"), dict):
        return None, {"schema": SCHEMA, "status": "unavailable", "reason": "no native Android capture"}
    try:
        motion = NativeMotion(capture_dir, report)
        return motion, motion.summary
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        return None, {"schema": SCHEMA, "status": "unavailable", "reason": str(exc),
                      "metric_pose_authority": False}
