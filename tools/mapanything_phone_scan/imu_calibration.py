"""Bounded retention and integrity checks for native stationary IMU recordings.

This is acquisition evidence, not an IMU noise fit or metric-VIO admission.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Any

SCHEMA = "noesis.phone_imu_calibration.v1"
MAX_BUNDLE_BYTES = 512 * 1024 * 1024
MAX_STREAM_ROWS = 4_000_000
MAX_RETAINED_CAPTURES = 32
HEADER = ["timestamp_ns", "x", "y", "z", "bias_x", "bias_y", "bias_z", "accuracy", "received_elapsed_realtime_ns"]
FILES = {"imu_capture_manifest.json", "accel.csv", "gyro.csv"}


class ImuCalibrationError(ValueError):
    pass


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ImuCalibrationError(f"Invalid {label}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ImuCalibrationError(f"Invalid {label}")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ImuCalibrationError(f"Invalid {label}") from exc
    if not math.isfinite(result):
        raise ImuCalibrationError(f"Invalid {label}")
    return result


def _json(data: bytes) -> dict[str, Any]:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ImuCalibrationError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(data, object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ImuCalibrationError("Invalid IMU manifest JSON") from exc
    if not isinstance(value, dict):
        raise ImuCalibrationError("IMU manifest must be an object")
    return value


def _stream(path: Path, declared: dict[str, Any]) -> dict[str, Any]:
    count = 0
    first = last = max_interval = nonmonotonic = unreliable = future = 0
    mean = [0.0] * 3
    m2 = [0.0] * 3
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for raw in iter(lambda: handle.readline(2049), b""):
            if len(raw) > 2048:
                raise ImuCalibrationError(f"Oversized CSV record in {path.name}")
            digest.update(raw)
            size += len(raw)
            try:
                row = next(csv.reader([raw.decode("utf-8")], strict=True))
            except (csv.Error, UnicodeError) as exc:
                raise ImuCalibrationError(f"Invalid CSV in {path.name}") from exc
            if size == len(raw):
                if row != HEADER:
                    raise ImuCalibrationError(f"Invalid CSV header in {path.name}")
                continue
            if len(row) != len(HEADER) or count >= MAX_STREAM_ROWS:
                raise ImuCalibrationError(f"Invalid or excessive CSV records in {path.name}")
            try:
                if any(not re.fullmatch(r"[0-9]+", row[i]) for i in (0, 8)):
                    raise ValueError("timestamp")
                timestamp, received, accuracy = int(row[0]), int(row[8]), int(row[7])
                if not 0 < timestamp <= 2**63 - 1 or not 0 <= received <= 2**63 - 1 or accuracy not in {-1, 0, 1, 2, 3}:
                    raise ValueError("timestamp/accuracy")
            except ValueError as exc:
                raise ImuCalibrationError(f"Invalid timestamp/accuracy in {path.name}") from exc
            xyz = [_finite(value, path.name) for value in row[1:4]]
            for value in row[4:7]:
                if value or declared["uncalibrated"]:
                    _finite(value, "sensor bias estimate")
            if count:
                interval = timestamp - last
                max_interval = max(max_interval, interval)
                nonmonotonic += int(interval <= 0)
            else:
                first = timestamp
            last = timestamp
            count += 1
            unreliable += int(accuracy == 0)
            future += int(timestamp > received)
            for axis, value in enumerate(xyz):
                delta = value - mean[axis]
                mean[axis] += delta / count
                m2[axis] += delta * (value - mean[axis])
    if count < 2:
        raise ImuCalibrationError(f"Need at least two samples in {path.name}")
    if not all(math.isfinite(value) for value in mean + m2):
        raise ImuCalibrationError(f"Sensor values exceed numeric range in {path.name}")
    verified = {"sample_count": count, "first_timestamp_ns": first, "last_timestamp_ns": last,
                "maximum_interval_ns": max_interval, "nonmonotonic_timestamp_count": nonmonotonic,
                "unreliable_accuracy_sample_count": unreliable, "bytes": size, "sha256": digest.hexdigest()}
    for key, value in verified.items():
        if declared.get(key) != value or isinstance(declared.get(key), bool):
            raise ImuCalibrationError(f"{path.name} disagrees with declared {key}")
    duration = (last - first) / 1e9
    return {**verified, "duration_s": duration, "observed_rate_hz": (count - 1) / duration if duration > 0 else None,
            "timestamps_after_receipt": future, "mean_xyz": mean,
            "sample_std_xyz": [math.sqrt(max(0, value) / (count - 1)) for value in m2]}


def retain_imu_bundle(bundle: Path, storage_root: Path) -> dict[str, Any]:
    """Validate exact ZIP members and raw counts/hashes before atomic retention."""
    if not 0 < bundle.stat().st_size <= MAX_BUNDLE_BYTES:
        raise ImuCalibrationError("IMU bundle exceeds 512 MiB or is empty")
    storage_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".checking-", dir=storage_root) as temporary:
        work = Path(temporary)
        try:
            with zipfile.ZipFile(bundle) as archive:
                members = archive.infolist()
                if len(members) != 3 or {item.filename for item in members} != FILES:
                    raise ImuCalibrationError("IMU bundle must contain exactly the manifest, accel.csv, and gyro.csv")
                if sum(item.file_size for item in members) > MAX_BUNDLE_BYTES:
                    raise ImuCalibrationError("Expanded IMU bundle exceeds 512 MiB")
                for item in members:
                    if item.flag_bits & 1 or stat.S_ISLNK(item.external_attr >> 16) or item.is_dir():
                        raise ImuCalibrationError("Invalid IMU archive member")
                    if item.filename.endswith(".json") and item.file_size > 64 * 1024:
                        raise ImuCalibrationError("IMU manifest exceeds 64 KiB")
                    with archive.open(item) as source, (work / item.filename).open("wb") as destination:
                        shutil.copyfileobj(source, destination, 1024 * 1024)
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError) as exc:
            raise ImuCalibrationError("Invalid IMU ZIP archive") from exc
        manifest = _json((work / "imu_capture_manifest.json").read_bytes())
        if manifest.get("schema") != SCHEMA or manifest.get("clock") != "android.elapsedRealtimeNanos" or manifest.get("timestamp_unit") != "ns":
            raise ImuCalibrationError("Unsupported IMU schema, clock, or timestamp unit")
        if manifest.get("axes") != "android_device_x_right_y_up_z_out_of_screen":
            raise ImuCalibrationError("Unsupported IMU axes")
        capture_id = manifest.get("capture_id")
        if not isinstance(capture_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", capture_id):
            raise ImuCalibrationError("Invalid IMU capture ID")
        device = manifest.get("device")
        if not isinstance(device, dict) or any(not isinstance(device.get(key), str) or not device[key] for key in ("id", "model", "build_fingerprint")):
            raise ImuCalibrationError("Missing IMU device identity")
        _integer(device.get("android_api_level"), "Android API", minimum=1)
        recording_status = manifest.get("status")
        if recording_status not in {"complete", "partial", "failed"}:
            raise ImuCalibrationError("IMU recording has not been finalized")
        actual = _finite(manifest.get("actual_duration_s"), "actual duration")
        expected = _finite(manifest.get("expected_duration_s"), "expected duration")
        if not 0 < actual <= 86_400 or not 0 < expected <= 86_400:
            raise ImuCalibrationError("Invalid IMU recording duration")
        dropped = _integer(manifest.get("dropped_records"), "dropped record count")
        streams = manifest.get("streams")
        if not isinstance(streams, dict):
            raise ImuCalibrationError("Missing IMU streams")
        stats = {}
        for key, filename, units in (("accelerometer", "accel.csv", "m/s^2"), ("gyroscope", "gyro.csv", "rad/s")):
            declared = streams.get(key)
            if not isinstance(declared, dict) or declared.get("file") != filename or declared.get("units") != units:
                raise ImuCalibrationError(f"Invalid {key} declaration or units")
            if type(declared.get("uncalibrated")) is not bool or not isinstance(declared.get("sensor_id"), str) or not declared["sensor_id"]:
                raise ImuCalibrationError(f"Missing {key} sensor identity/type")
            stats[key] = _stream(work / filename, declared)
        issues = []
        if recording_status != "complete":
            issues.append(f"recording_{recording_status}")
        if min(value["duration_s"] for value in stats.values()) < 10_799:
            issues.append("less_than_three_hours")
        if dropped:
            issues.append("dropped_sensor_records")
        for key, value in stats.items():
            if value["nonmonotonic_timestamp_count"] or value["timestamps_after_receipt"]:
                issues.append(f"{key}_timestamp_failure")
            if value["maximum_interval_ns"] > 50_000_000:
                issues.append(f"{key}_sample_gap")
        digest = hashlib.sha256()
        with bundle.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        report = {"schema": "noesis.phone_imu_calibration.receipt.v1", "status": "stored", "capture_id": capture_id,
                  "recording_status": recording_status, "actual_duration_s": actual, "bundle_sha256": digest.hexdigest(),
                  "streams": stats, "acquisition_issues": issues, "imu_noise_calibrated": False,
                  "message": "IMU recording retained. Sensor-noise fitting and review are still required."}
        target = storage_root / capture_id
        if target.exists():
            old = _json((target / "receipt.json").read_bytes())
            if old.get("bundle_sha256") != report["bundle_sha256"]:
                raise ImuCalibrationError("This capture ID already contains a different recording")
            return old
        if sum(1 for item in storage_root.iterdir() if item.is_dir() and not item.name.startswith(".")) >= MAX_RETAINED_CAPTURES:
            raise ImuCalibrationError("IMU recording storage is full; retained recordings need review")
        (work / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        shutil.copyfile(bundle, work / "capture_bundle.zip")
        work.rename(target)
        return report
