"""Import and validate synchronized phone camera/IMU capture bundles.

The browser video upload remains intentionally separate from this format.  A
sensor bundle is a small, self describing archive produced by a recorder that
has access to acquisition timestamps and device calibration.  This module
keeps the original files and writes a normalized, bounded import report for
the frame preparation and VIO adapters.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


CAPTURE_SCHEMA = "noesis.phone_capture.v1"
CAPTURE_MANIFEST_NAMES = {"capture_manifest.json", "manifest.json"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".3gp"}
MAX_MANIFEST_BYTES = 512 * 1024
# Browser manifests intentionally carry bounded raw sensor observations.  The
# native schema keeps the smaller limit above; this larger limit is selected
# only after an explicit browser schema marker is found near the JSON root.
MAX_BROWSER_MANIFEST_BYTES = 32 * 1024 * 1024
MAX_IMU_ROWS = 2_000_000
MAX_VIDEO_TIMESTAMPS = 1_000_000
MAX_ARCHIVE_FILES = 512
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024
MAX_MEMBER_BYTES = 8 * 1024 * 1024 * 1024
MAX_TIME_GAP_S = 0.25


class CaptureImportError(ValueError):
    """Raised when a capture bundle cannot be admitted safely."""


@dataclass(frozen=True)
class CaptureImportLimits:
    max_archive_bytes: int = MAX_ARCHIVE_BYTES
    max_uncompressed_bytes: int = MAX_ARCHIVE_BYTES
    max_member_bytes: int = MAX_MEMBER_BYTES
    max_files: int = MAX_ARCHIVE_FILES
    max_imu_rows: int = MAX_IMU_ROWS
    max_video_timestamps: int = MAX_VIDEO_TIMESTAMPS
    max_time_gap_s: float = MAX_TIME_GAP_S
    video_probe_timeout_s: int = 120


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise CaptureImportError(f"{field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CaptureImportError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise CaptureImportError(f"{field} must be finite")
    return number


def _safe_member_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise CaptureImportError("archive contains an invalid member path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise CaptureImportError(f"archive contains an unsafe member path: {name}")
    normalized = path.as_posix()
    if normalized != name or normalized in {"", "."}:
        raise CaptureImportError(f"archive contains a non-normalized member path: {name}")
    return normalized


def _member_map(names: Iterable[str], *, limits: CaptureImportLimits) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in names:
        name = _safe_member_name(raw)
        if name in seen:
            raise CaptureImportError(f"archive contains duplicate member: {name}")
        seen.add(name)
        normalized.append(name)
    if len(normalized) > limits.max_files:
        raise CaptureImportError(
            f"archive contains too many files ({len(normalized)} > {limits.max_files})"
        )
    return normalized


def _read_json_bytes(data: bytes, *, label: str, maximum: int = MAX_MANIFEST_BYTES) -> dict[str, Any]:
    if len(data) > maximum:
        raise CaptureImportError(f"{label} exceeds {maximum} bytes")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureImportError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise CaptureImportError(f"{label} must be a JSON object")
    return payload


def _relative_path(raw: Any, *, label: str, allowed: set[str] | None = None) -> str:
    if not isinstance(raw, str):
        raise CaptureImportError(f"{label} must be a relative archive path")
    value = _safe_member_name(raw)
    if allowed is not None and Path(value).suffix.lower() not in allowed:
        raise CaptureImportError(f"{label} has an unsupported file type")
    return value


def _matrix(raw: Any, *, shape: tuple[int, int], field: str) -> list[list[float]]:
    if not isinstance(raw, list) or len(raw) != shape[0]:
        raise CaptureImportError(f"{field} must be a {shape[0]}x{shape[1]} matrix")
    result: list[list[float]] = []
    for row in raw:
        if not isinstance(row, list) or len(row) != shape[1]:
            raise CaptureImportError(f"{field} must be a {shape[0]}x{shape[1]} matrix")
        result.append([_finite_number(value, field=field) for value in row])
    return result


def _validate_rotation(matrix: list[list[float]], *, field: str) -> None:
    # A small tolerance is useful for JSON exports from mobile calibration APIs.
    import numpy as np

    rotation = np.asarray(matrix, dtype=np.float64)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise CaptureImportError(f"{field} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise CaptureImportError(f"{field} rotation is not right handed")


def validate_capture_manifest(
    manifest: Mapping[str, Any],
    *,
    member_names: set[str] | None = None,
) -> dict[str, Any]:
    """Validate and normalize the metadata portion of a capture bundle."""

    if manifest.get("schema") != CAPTURE_SCHEMA:
        raise CaptureImportError(f"capture manifest must use {CAPTURE_SCHEMA}")
    capture_id = str(manifest.get("capture_id") or "").strip()
    if not capture_id or len(capture_id) > 160:
        raise CaptureImportError("capture manifest has no valid capture_id")

    device = manifest.get("device")
    if not isinstance(device, Mapping):
        raise CaptureImportError("capture manifest has no device identity")
    device_id = str(device.get("id") or "").strip()
    if not device_id:
        raise CaptureImportError("capture manifest has no device.id")

    video = manifest.get("video")
    if not isinstance(video, Mapping):
        raise CaptureImportError("capture manifest has no video stream")
    video_path = _relative_path(video.get("path"), label="video.path", allowed=VIDEO_SUFFIXES)
    frame_timestamp_path_raw = video.get("frame_timestamps_path")
    frame_timestamp_path = (
        _relative_path(frame_timestamp_path_raw, label="video.frame_timestamps_path")
        if frame_timestamp_path_raw is not None
        else None
    )
    video_timestamp_unit = str(video.get("timestamp_unit") or "ns").strip().lower()
    if not _timestamp_scale(video_timestamp_unit):
        raise CaptureImportError("video.timestamp_unit is unsupported")
    imu = manifest.get("imu")
    if not isinstance(imu, Mapping):
        raise CaptureImportError("capture manifest has no IMU stream")
    imu_path_raw = imu.get("path")
    accel_path_raw = imu.get("accel_path")
    gyro_path_raw = imu.get("gyro_path")
    separate_imu_streams = accel_path_raw is not None or gyro_path_raw is not None
    if separate_imu_streams and (accel_path_raw is None or gyro_path_raw is None):
        raise CaptureImportError("imu.accel_path and imu.gyro_path must be supplied together")
    imu_path = _relative_path(imu_path_raw, label="imu.path") if not separate_imu_streams else None
    accel_path = _relative_path(accel_path_raw, label="imu.accel_path") if separate_imu_streams else None
    gyro_path = _relative_path(gyro_path_raw, label="imu.gyro_path") if separate_imu_streams else None
    camera = manifest.get("camera")
    if not isinstance(camera, Mapping):
        raise CaptureImportError("capture manifest has no camera calibration row")
    camera_id = str(camera.get("id") or camera.get("sensor_id") or "").strip()
    if not camera_id:
        raise CaptureImportError("capture manifest has no camera sensor identity")
    intrinsics_source = str(camera.get("intrinsics_source") or "provided").strip().lower()
    intrinsics_raw = camera.get("intrinsics")
    intrinsics_calibrated = intrinsics_source != "missing" and intrinsics_raw is not None
    if intrinsics_calibrated:
        intrinsics = _matrix(intrinsics_raw, shape=(3, 3), field="camera.intrinsics")
        if intrinsics[2] != [0.0, 0.0, 1.0]:
            raise CaptureImportError("camera.intrinsics must be a calibrated pinhole matrix")
        if intrinsics[0][0] <= 0.0 or intrinsics[1][1] <= 0.0:
            raise CaptureImportError("camera.intrinsics focal lengths must be positive")
    else:
        # A recorder export is useful raw evidence before a camera
        # calibration has been supplied.  Preserve the unknown value as null;
        # never put an identity K in a report that a downstream consumer could
        # mistake for metric calibration.
        intrinsics = None
    resolution = camera.get("resolution_px") or camera.get("resolution")
    if not isinstance(resolution, list) or len(resolution) != 2:
        if intrinsics_calibrated:
            raise CaptureImportError("camera.resolution_px must contain width and height")
        width, height = 0, 0
    else:
        width = int(_finite_number(resolution[0], field="camera.resolution_px[0]"))
        height = int(_finite_number(resolution[1], field="camera.resolution_px[1]"))
        if width <= 0 or height <= 0:
            raise CaptureImportError("camera resolution must be positive")
    distortion = camera.get("distortion", [])
    if not isinstance(distortion, list) or len(distortion) > 14:
        raise CaptureImportError("camera.distortion must be a finite coefficient list")
    distortion = [_finite_number(value, field="camera.distortion") for value in distortion]
    # D5 pinhole input is rectified with all coefficients before OpenVINS.
    # Other higher-order models remain evidence-only; never truncate them.
    distortion_model = str(camera.get("distortion_model") or "").strip().lower()
    known_distortion_model = distortion_model in {
        "plumb_bob", "radtan", "brown_conrady", "equidistant", "fisheye",
    }
    if distortion_model in {"unknown", "missing"} and not distortion:
        # Native acquisition can precede lens calibration.  An empty unknown
        # model preserves that evidence without inventing zero distortion.
        distortion_model = "unknown"
    elif not known_distortion_model:
        raise CaptureImportError(
            "camera.distortion_model must identify a supported calibrated model"
        )
    distortion_supported = known_distortion_model and (
        len(distortion) <= 4
        or (len(distortion) == 5 and distortion_model in {"plumb_bob", "radtan", "brown_conrady"})
    )

    extrinsics = manifest.get("extrinsics")
    if not isinstance(extrinsics, Mapping):
        extrinsics = {}
    t_imu_camera = extrinsics.get("T_imu_camera")
    if t_imu_camera is None:
        t_imu_camera = extrinsics.get("camera_to_imu")
    calibrated_extrinsics = t_imu_camera is not None
    if calibrated_extrinsics:
        t_imu_camera = _matrix(t_imu_camera, shape=(4, 4), field="extrinsics.T_imu_camera")
        _validate_rotation([row[:3] for row in t_imu_camera[:3]], field="extrinsics.T_imu_camera")
        if t_imu_camera[3] != [0.0, 0.0, 0.0, 1.0]:
            raise CaptureImportError("extrinsics.T_imu_camera must be homogeneous")
    else:
        t_imu_camera = None

    clocks = manifest.get("clocks")
    if not isinstance(clocks, Mapping):
        clocks = {}
    camera_domain = str(clocks.get("camera_domain") or video.get("time_domain") or "").strip()
    imu_domain = str(clocks.get("imu_domain") or imu.get("time_domain") or "").strip()
    offset_raw = clocks.get("imu_to_camera_offset_ns")
    if offset_raw is None:
        offset_raw = clocks.get("camera_minus_imu_offset_ns")
    time_offset_ns: float | None
    if offset_raw is None:
        time_offset_ns = None
    else:
        time_offset_ns = _finite_number(offset_raw, field="clocks.imu_to_camera_offset_ns")
    camera_epoch_ns = clocks.get("camera_start_time_ns")
    if camera_epoch_ns is not None:
        camera_epoch_ns = int(_finite_number(camera_epoch_ns, field="clocks.camera_start_time_ns"))
    clock_source = str(clocks.get("timestamp_source") or "").strip().lower()
    clock_source_verified = clock_source in {"realtime", "hardware", "ptp", "genlock"}

    android_capture = manifest.get("android_capture")
    if android_capture is not None:
        if not isinstance(android_capture, Mapping) or android_capture.get("schema") != "noesis.phone_capture.android.v1":
            raise CaptureImportError("android_capture must use noesis.phone_capture.android.v1")
        android_capture = {
            "schema": "noesis.phone_capture.android.v1",
            **{
                key: _relative_path(android_capture.get(key), label=f"android_capture.{key}")
                for key in ("capture_result_path", "encoder_pts_path", "camera_results_path")
            },
        }
        # The recorder's clock label is not proof of the encoded-frame mapping.
        # Import verifies its actual Camera2 and MediaCodec rows below.
        clock_source_verified = False

    stabilization = str(camera.get("stabilization") or "unknown").strip().lower()
    orientation_raw = _finite_number(camera.get("orientation_deg", 0), field="camera.orientation_deg")
    if orientation_raw % 90.0 != 0.0:
        raise CaptureImportError("camera.orientation_deg must be a quarter-turn orientation")
    orientation = int(orientation_raw) % 360
    crop = camera.get("crop")
    if crop is None:
        if width > 0 and height > 0:
            crop = {"left": 0, "top": 0, "width": width, "height": height}
        else:
            crop = {"left": 0, "top": 0, "width": 0, "height": 0}
    if not isinstance(crop, Mapping):
        raise CaptureImportError("camera.crop must be an object")
    crop_values = {
        key: int(_finite_number(crop.get(key, 0), field=f"camera.crop.{key}"))
        for key in ("left", "top", "width", "height")
    }
    if crop_values["width"] < 0 or crop_values["height"] < 0:
        raise CaptureImportError("camera crop width and height must be positive")
    if width > 0 and height > 0 and (crop_values["width"] <= 0 or crop_values["height"] <= 0):
        raise CaptureImportError("camera crop width and height must be positive")
    if width > 0 and height > 0 and (crop_values["left"] < 0 or crop_values["top"] < 0 or crop_values["left"] + crop_values["width"] > width or crop_values["top"] + crop_values["height"] > height):
        raise CaptureImportError("camera crop must be contained by the calibrated resolution")

    raw_accel_unit = str(imu.get("accel_unit") or imu.get("accelerometer_unit") or "").strip()
    raw_gyro_unit = str(imu.get("gyro_unit") or imu.get("gyroscope_unit") or "").strip()
    axes = str(imu.get("axes") or "x,y,z").strip().lower()
    if axes not in {"x,y,z", "xyz", "x y z"}:
        raise CaptureImportError("imu.axes must declare x,y,z order")
    if not raw_accel_unit or not raw_gyro_unit:
        raise CaptureImportError("imu accel_unit and gyro_unit are required")
    raw_noise = imu.get("noise")
    if not isinstance(raw_noise, Mapping):
        raw_noise = {}
    noise_aliases = {
        "gyroscope_noise_density": ("gyroscope_noise_density", "gyro_noise_density"),
        "gyroscope_random_walk": ("gyroscope_random_walk", "gyro_random_walk"),
        "accelerometer_noise_density": ("accelerometer_noise_density", "accel_noise_density"),
        "accelerometer_random_walk": ("accelerometer_random_walk", "accel_random_walk"),
    }
    noise: dict[str, float] = {}
    for key, aliases in noise_aliases.items():
        value = next((raw_noise.get(alias) for alias in aliases if raw_noise.get(alias) is not None), None)
        if value is not None:
            parsed = _finite_number(value, field=f"imu.noise.{key}")
            if parsed <= 0.0:
                raise CaptureImportError(f"imu.noise.{key} must be positive")
            noise[key] = parsed
    noise_complete = len(noise) == len(noise_aliases)

    stream_paths = [video_path]
    stream_paths.extend([imu_path] if imu_path else [accel_path, gyro_path])
    if frame_timestamp_path:
        stream_paths.append(frame_timestamp_path)
    if android_capture is not None:
        stream_paths.extend(android_capture[key] for key in ("capture_result_path", "encoder_pts_path", "camera_results_path"))
    if member_names is not None and any(path not in member_names for path in stream_paths):
        raise CaptureImportError("capture manifest references a missing stream file")

    normalized = {
        "schema": CAPTURE_SCHEMA,
        "capture_id": capture_id,
        "device": {
            "id": device_id,
            "model": str(device.get("model") or "unknown"),
            "os": str(device.get("os") or "unknown"),
        },
        "video": {
            "path": video_path,
            "sensor_id": str(video.get("sensor_id") or camera_id),
            "time_domain": camera_domain or "video_pts_relative",
            "timestamp_source": str(video.get("timestamp_source") or "container_pts"),
            "frame_timestamps_path": frame_timestamp_path,
            "timestamp_unit": video_timestamp_unit,
            "stabilization": stabilization,
            "orientation_deg": orientation,
            "crop": crop_values,
        },
        "imu": {
            "path": imu_path,
            "accel_path": accel_path,
            "gyro_path": gyro_path,
            "sample_clock_mode": "separate_streams" if separate_imu_streams else "shared_timestamp",
            "sensor_id": str(imu.get("sensor_id") or f"{device_id}:imu"),
            "time_domain": imu_domain or "imu_clock",
            "axes": "x,y,z",
            "accel_unit": raw_accel_unit,
            "gyro_unit": raw_gyro_unit,
            "timestamp_unit": str(imu.get("timestamp_unit") or "ns"),
            "noise": noise,
        },
        "camera": {
            "id": camera_id,
            "intrinsics": intrinsics,
            "intrinsics_source": intrinsics_source if intrinsics_calibrated else "missing",
            "distortion": distortion,
            "distortion_supported": distortion_supported,
            "distortion_model": distortion_model,
            "resolution_px": [width, height] if width > 0 and height > 0 else None,
            "orientation_deg": orientation,
            "stabilization": stabilization,
            "crop": crop_values if crop_values["width"] > 0 and crop_values["height"] > 0 else None,
        },
        "extrinsics": {
            "T_imu_camera": t_imu_camera,
            "convention": "T_imu_camera maps camera coordinates to IMU coordinates",
        },
        "clocks": {
            "camera_domain": camera_domain or "video_pts_relative",
            "imu_domain": imu_domain or "imu_clock",
            "imu_to_camera_offset_ns": time_offset_ns,
            "camera_start_time_ns": camera_epoch_ns,
            "timestamp_source": clock_source or "unverified",
        },
        "calibration": {
            "camera_intrinsics": intrinsics_calibrated,
            "camera_distortion": bool(distortion),
            "distortion_supported": distortion_supported,
            "camera_to_imu_extrinsics": calibrated_extrinsics,
            "time_offset": time_offset_ns is not None,
                "clock_source_verified": clock_source_verified,
                "imu_noise": noise_complete,
                "complete_for_metric_vio": bool(
                intrinsics_calibrated
                and calibrated_extrinsics
                and time_offset_ns is not None
                and clock_source_verified
                and camera_domain
                and imu_domain
                and stabilization == "off"
                and noise_complete
                and distortion_supported
            ),
        },
    }
    if android_capture is not None:
        normalized["android_capture"] = android_capture
    return normalized


def _timestamp_scale(unit: str) -> float:
    value = unit.strip().lower()
    return {
        "ns": 1e-9,
        "nanosecond": 1e-9,
        "nanoseconds": 1e-9,
        "us": 1e-6,
        "microsecond": 1e-6,
        "microseconds": 1e-6,
        "ms": 1e-3,
        "millisecond": 1e-3,
        "milliseconds": 1e-3,
        "s": 1.0,
        "sec": 1.0,
        "seconds": 1.0,
    }.get(value, 0.0)


def _timestamp_to_ns(raw: Any, unit: str, *, field: str) -> int:
    """Convert a timestamp with decimal arithmetic, preserving integer ns IDs."""

    factors = {
        "ns": Decimal("1"),
        "nanosecond": Decimal("1"),
        "nanoseconds": Decimal("1"),
        "us": Decimal("1000"),
        "microsecond": Decimal("1000"),
        "microseconds": Decimal("1000"),
        "ms": Decimal("1000000"),
        "millisecond": Decimal("1000000"),
        "milliseconds": Decimal("1000000"),
        "s": Decimal("1000000000"),
        "sec": Decimal("1000000000"),
        "seconds": Decimal("1000000000"),
    }
    factor = factors.get(str(unit).strip().lower())
    if factor is None:
        raise CaptureImportError(f"{field} timestamp unit is unsupported")
    try:
        value = Decimal(str(raw).strip()) * factor
    except (InvalidOperation, ValueError) as exc:
        raise CaptureImportError(f"{field} must be finite") from exc
    if not value.is_finite():
        raise CaptureImportError(f"{field} must be finite")
    return int(value.to_integral_value(rounding=ROUND_HALF_UP))


def _accel_scale(unit: str) -> float:
    value = unit.strip().lower().replace("²", "2")
    if value in {"m/s2", "m/s^2", "m s-2", "meter_per_second_squared", "meters_per_second_squared"}:
        return 1.0
    if value in {"g", "gravity"}:
        return 9.80665
    return 0.0


def _gyro_scale(unit: str) -> float:
    value = unit.strip().lower().replace("°", "deg")
    if value in {"rad/s", "rad s-1", "radians_per_second"}:
        return 1.0
    if value in {"deg/s", "degree/s", "degrees_per_second", "deg s-1"}:
        return math.pi / 180.0
    return 0.0


def _field_name(row: Mapping[str, Any], aliases: tuple[str, ...]) -> str | None:
    lowered = {str(key).strip().lower(): str(key) for key in row}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def _numeric_csv_rows(data: bytes, *, label: str) -> tuple[list[list[str]], bool]:
    try:
        rows = [row for row in csv.reader(io.StringIO(data.decode("utf-8-sig"))) if row]
    except (UnicodeDecodeError, csv.Error) as exc:
        raise CaptureImportError(f"{label} is not a readable CSV") from exc
    if not rows:
        raise CaptureImportError(f"{label} is empty")
    headerless = True
    for value in rows[0]:
        try:
            float(value)
        except ValueError:
            headerless = False
            break
    return rows, headerless


def parse_frame_timestamps_csv(
    data: bytes,
    *,
    timestamp_unit: str = "ns",
    limits: CaptureImportLimits = CaptureImportLimits(),
) -> list[int]:
    """Read OpenCamera Sensors ``*_timestamps.csv`` (normally one ns column)."""

    scale = _timestamp_scale(timestamp_unit)
    if not scale:
        raise CaptureImportError("video timestamp unit is unsupported")
    rows, headerless = _numeric_csv_rows(data, label="video timestamps")
    values = rows if headerless else rows[1:]
    timestamps: list[int] = []
    previous = None
    for index, row in enumerate(values):
        if index >= limits.max_video_timestamps:
            raise CaptureImportError("video timestamp stream exceeds its row limit")
        if not row:
            continue
        timestamp = _timestamp_to_ns(
            row[-1],
            timestamp_unit,
            field=f"video_timestamp[{index}]",
        )
        if previous is not None and timestamp <= previous:
            raise CaptureImportError("video capture timestamps are not strictly increasing")
        timestamps.append(timestamp)
        previous = timestamp
    if len(timestamps) < 2:
        raise CaptureImportError("video timestamp stream must contain at least two rows")
    return timestamps


def _derive_open_camera_manifest(names: set[str]) -> dict[str, Any] | None:
    """Create an incomplete import manifest for OpenCamera Sensors exports."""

    videos = [name for name in sorted(names) if Path(name).suffix.lower() in VIDEO_SUFFIXES]
    if len(videos) != 1:
        return None
    video_path = videos[0]
    stem = Path(video_path).stem
    parent = Path(video_path).parent
    candidates = {name.lower(): name for name in names}
    def sibling(suffix: str) -> str | None:
        wanted = (parent / f"{stem}{suffix}").as_posix().lower()
        if wanted in candidates:
            return candidates[wanted]
        # Some exports rename the mp4 while retaining the timestamp files.
        matches = [name for name in names if name.lower().endswith(suffix.lower())]
        return matches[0] if len(matches) == 1 else None
    accel_path = sibling("_accel.csv")
    gyro_path = sibling("_gyro.csv")
    if not accel_path or not gyro_path:
        return None
    timestamps_path = sibling("_timestamps.csv")
    return {
        "schema": CAPTURE_SCHEMA,
        "capture_id": f"opencamera:{Path(video_path).name}",
        "device": {"id": "unknown:opencamera-sensors", "model": "unknown", "os": "Android (unreported)"},
        "video": {
            "path": video_path,
            "sensor_id": "unknown:camera",
            "time_domain": "android_realtime_ns (unverified)",
            "timestamp_source": "opencamera_sensor_timestamps" if timestamps_path else "container_pts",
            "frame_timestamps_path": timestamps_path,
        },
        "imu": {
            "accel_path": accel_path,
            "gyro_path": gyro_path,
            "sensor_id": "unknown:imu",
            "time_domain": "android_realtime_ns (unverified)",
            "axes": "x,y,z",
            "accel_unit": "m/s^2",
            "gyro_unit": "rad/s",
            "timestamp_unit": "ns",
        },
        "camera": {
            "id": "unknown:camera",
            "intrinsics_source": "missing",
            "intrinsics": None,
            "distortion_model": "plumb_bob",
            "distortion": [],
            "resolution_px": None,
            "orientation_deg": 0,
            "stabilization": "unknown",
            "crop": None,
        },
        "clocks": {
            "camera_domain": "android_realtime_ns (unverified)",
            "imu_domain": "android_realtime_ns (unverified)",
            "imu_to_camera_offset_ns": 0,
        },
    }


def parse_imu_csv(
    data: bytes,
    imu: Mapping[str, Any],
    *,
    limits: CaptureImportLimits = CaptureImportLimits(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse a CSV IMU stream while retaining SI-normalized and raw values."""

    timestamp_scale = _timestamp_scale(str(imu.get("timestamp_unit") or "ns"))
    accel_scale = _accel_scale(str(imu.get("accel_unit") or ""))
    gyro_scale = _gyro_scale(str(imu.get("gyro_unit") or ""))
    if not timestamp_scale or not accel_scale or not gyro_scale:
        raise CaptureImportError("IMU units are unsupported; use explicit SI-compatible units")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CaptureImportError("IMU CSV is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise CaptureImportError("IMU CSV has no header")
    rows: list[dict[str, Any]] = []
    previous_ns: int | None = None
    gaps = 0
    max_gap_s = 0.0
    timestamp_aliases = ("timestamp_ns", "timestamp", "time_ns", "time", "t")
    accel_aliases = (("ax", "accel_x", "accelerometer_x", "a_x"), ("ay", "accel_y", "accelerometer_y", "a_y"), ("az", "accel_z", "accelerometer_z", "a_z"))
    gyro_aliases = (("gx", "gyro_x", "gyroscope_x", "g_x", "wx"), ("gy", "gyro_y", "gyroscope_y", "g_y", "wy"), ("gz", "gyro_z", "gyroscope_z", "g_z", "wz"))
    for index, raw in enumerate(reader):
        if index >= limits.max_imu_rows:
            raise CaptureImportError(f"IMU stream exceeds {limits.max_imu_rows} rows")
        timestamp_name = _field_name(raw, timestamp_aliases)
        value_names = [_field_name(raw, aliases) for aliases in accel_aliases + gyro_aliases]
        if timestamp_name is None or any(name is None for name in value_names):
            raise CaptureImportError("IMU CSV must contain timestamp, ax/ay/az, and gx/gy/gz columns")
        # Keep integer conversion explicit while allowing seconds,
        # milliseconds, microseconds, or nanoseconds input.
        timestamp_ns = _timestamp_to_ns(
            raw[timestamp_name],
            str(imu.get("timestamp_unit") or "ns"),
            field=f"imu[{index}].timestamp",
        )
        if previous_ns is not None:
            delta_s = (timestamp_ns - previous_ns) * 1e-9
            if delta_s <= 0.0:
                raise CaptureImportError(f"IMU timestamps are not strictly increasing at row {index}")
            max_gap_s = max(max_gap_s, delta_s)
            if delta_s > limits.max_time_gap_s:
                gaps += 1
        previous_ns = timestamp_ns
        accel_raw = [_finite_number(raw[name], field=f"imu[{index}].accel") for name in value_names[:3]]
        gyro_raw = [_finite_number(raw[name], field=f"imu[{index}].gyro") for name in value_names[3:]]
        rows.append({
            "timestamp_ns": timestamp_ns,
            "accel_raw": accel_raw,
            "gyro_raw": gyro_raw,
            "accel_mps2": [value * accel_scale for value in accel_raw],
            "gyro_rads": [value * gyro_scale for value in gyro_raw],
        })
    if len(rows) < 2:
        raise CaptureImportError("IMU stream must contain at least two samples")
    metrics = {
        "sample_count": len(rows),
        "start_time_ns": rows[0]["timestamp_ns"],
        "end_time_ns": rows[-1]["timestamp_ns"],
        "duration_s": (rows[-1]["timestamp_ns"] - rows[0]["timestamp_ns"]) * 1e-9,
        "max_gap_s": max_gap_s,
        "gap_count_over_limit": gaps,
        "strictly_increasing": True,
        "finite": True,
    }
    return rows, metrics


def parse_imu_axis_csv(
    data: bytes,
    imu: Mapping[str, Any],
    *,
    kind: str,
    limits: CaptureImportLimits = CaptureImportLimits(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse one independently timestamped accelerometer or gyro stream."""

    if kind not in {"accel", "gyro"}:
        raise CaptureImportError("IMU stream kind must be accel or gyro")
    scale = _accel_scale(str(imu.get("accel_unit") or "")) if kind == "accel" else _gyro_scale(str(imu.get("gyro_unit") or ""))
    if not scale:
        raise CaptureImportError(f"unsupported {kind} unit")
    raw_rows, headerless = _numeric_csv_rows(data, label=f"{kind} stream")
    if headerless:
        records: Iterable[Mapping[str, Any]] = (
            {"timestamp": row[-1], "x": row[0], "y": row[1], "z": row[2]}
            for row in raw_rows
            if len(row) >= 4
        )
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise CaptureImportError(f"{kind} CSV is not UTF-8") from exc
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise CaptureImportError(f"{kind} CSV has no header")
        records = reader
    timestamp_aliases = ("timestamp_ns", "timestamp", "timestamp (ns)", "time_ns", "time", "t")
    value_aliases = (("x", "x-data", f"{kind}_x", f"{kind}x", "a_x" if kind == "accel" else "w_x"), ("y", "y-data", f"{kind}_y", f"{kind}y", "a_y" if kind == "accel" else "w_y"), ("z", "z-data", f"{kind}_z", f"{kind}z", "a_z" if kind == "accel" else "w_z"))
    timestamp_scale = _timestamp_scale(str(imu.get("timestamp_unit") or "ns"))
    if not timestamp_scale:
        raise CaptureImportError("IMU timestamp_unit is unsupported")
    rows: list[dict[str, Any]] = []
    previous_ns: int | None = None
    gaps = 0
    max_gap_s = 0.0
    for index, raw in enumerate(records):
        if index >= limits.max_imu_rows:
            raise CaptureImportError(f"{kind} stream exceeds {limits.max_imu_rows} rows")
        timestamp_name = _field_name(raw, timestamp_aliases)
        names = [_field_name(raw, aliases) for aliases in value_aliases]
        if timestamp_name is None or any(name is None for name in names):
            raise CaptureImportError(f"{kind} CSV must contain timestamp and x/y/z columns")
        timestamp_ns = _timestamp_to_ns(
            raw[timestamp_name],
            str(imu.get("timestamp_unit") or "ns"),
            field=f"{kind}[{index}].timestamp",
        )
        if previous_ns is not None:
            delta_s = (timestamp_ns - previous_ns) * 1e-9
            if delta_s <= 0.0:
                raise CaptureImportError(f"{kind} timestamps are not strictly increasing at row {index}")
            max_gap_s = max(max_gap_s, delta_s)
            if delta_s > limits.max_time_gap_s:
                gaps += 1
        previous_ns = timestamp_ns
        raw_values = [_finite_number(raw[name], field=f"{kind}[{index}].value") for name in names]
        rows.append({"timestamp_ns": timestamp_ns, "raw": raw_values, "si": [value * scale for value in raw_values]})
    if len(rows) < 2:
        raise CaptureImportError(f"{kind} stream must contain at least two samples")
    return rows, {
        "sample_count": len(rows),
        "start_time_ns": rows[0]["timestamp_ns"],
        "end_time_ns": rows[-1]["timestamp_ns"],
        "duration_s": (rows[-1]["timestamp_ns"] - rows[0]["timestamp_ns"]) * 1e-9,
        "max_gap_s": max_gap_s,
        "gap_count_over_limit": gaps,
        "strictly_increasing": True,
        "finite": True,
    }


def _probe_video_metadata(path: Path) -> tuple[int, int]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    if completed.returncode != 0:
        raise CaptureImportError(f"ffprobe could not read video dimensions: {completed.stderr.strip()[-500:]}")
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CaptureImportError("ffprobe returned invalid video dimensions") from exc
    if width <= 0 or height <= 0:
        raise CaptureImportError("capture video dimensions must be positive")
    return width, height


def _probe_video_timestamps(
    path: Path, *, maximum: int, timestamp_details: dict[str, Any] | None = None,
    timeout_s: int = 120,
) -> list[float]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=time_base:frame=best_effort_timestamp,best_effort_timestamp_time,pkt_duration_time",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_s)
    if completed.returncode != 0:
        raise CaptureImportError(f"ffprobe could not read the capture video: {completed.stderr.strip()[-500:]}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CaptureImportError("ffprobe returned invalid frame timestamp JSON") from exc
    frames = payload.get("frames") if isinstance(payload, Mapping) else None
    if not isinstance(frames, list) or not frames:
        raise CaptureImportError("capture video has no frame timestamps")
    if len(frames) > maximum:
        raise CaptureImportError(f"capture video has too many frames ({len(frames)} > {maximum})")
    timestamps: list[float] = []
    previous = -math.inf
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise CaptureImportError(f"capture video frame {index} is invalid")
        value = frame.get("best_effort_timestamp_time")
        timestamp = _finite_number(value, field=f"video.frame[{index}].timestamp")
        if timestamp < previous:
            raise CaptureImportError("capture video frame timestamps are not monotonic")
        timestamps.append(timestamp)
        previous = timestamp
    if timestamp_details is not None:
        streams = payload.get("streams")
        timestamp_details["time_base"] = (
            streams[0].get("time_base")
            if isinstance(streams, list) and streams and isinstance(streams[0], Mapping)
            else None
        )
        timestamp_details["timestamps_ticks"] = [frame.get("best_effort_timestamp") for frame in frames]
    return timestamps


def _read_capture_manifest(path: Path) -> dict[str, Any]:
    """Read a manifest with schema-specific bounded size handling.

    Native manifests retain the historical 512 KiB ceiling.  Browser capture
    manifests may be larger because they preserve sensor samples, but the
    browser schema must be visible in a bounded prefix before that allowance
    is used.
    """

    try:
        size = path.stat().st_size
        maximum = MAX_MANIFEST_BYTES
        if size > MAX_MANIFEST_BYTES:
            with path.open("rb") as handle:
                prefix = handle.read(min(size, 128 * 1024))
            if re.search(
                rb'"schema"\s*:\s*"noesis\.phone_capture\.browser\.v1"',
                prefix,
            ):
                maximum = MAX_BROWSER_MANIFEST_BYTES
        if size > maximum:
            raise CaptureImportError(f"capture manifest exceeds {maximum} bytes")
        data = path.read_bytes()
        payload = _read_json_bytes(data, label="capture manifest", maximum=maximum)
        # The prefix is only a bounded admission hint.  A native manifest
        # containing a nested browser-looking string still receives the native
        # limit after the top-level schema is decoded.
        if payload.get("schema") != "noesis.phone_capture.browser.v1" and len(data) > MAX_MANIFEST_BYTES:
            raise CaptureImportError(f"capture manifest exceeds {MAX_MANIFEST_BYTES} bytes")
        return payload
    except OSError as exc:
        raise CaptureImportError("capture manifest is unreadable") from exc


def _safe_extract_zip(source: Path, destination: Path, *, limits: CaptureImportLimits) -> tuple[set[str], int]:
    try:
        archive = zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc:
        raise CaptureImportError("capture upload is not a valid ZIP archive") from exc
    names: list[str] = []
    total = 0
    with archive:
        for info in archive.infolist():
            name = _safe_member_name(info.filename)
            if info.is_dir():
                continue
            if info.create_system == 3 and ((info.external_attr >> 16) & 0o170000) in {stat.S_IFLNK, stat.S_IFDIR}:
                raise CaptureImportError(f"archive member is not a regular file: {name}")
            if info.file_size > limits.max_member_bytes:
                raise CaptureImportError(f"archive member exceeds size limit: {name}")
            names.append(name)
            total += int(info.file_size)
        normalized = _member_map(names, limits=limits)
        if total > limits.max_uncompressed_bytes:
            raise CaptureImportError("archive uncompressed size exceeds the import limit")
        for info, name in zip((item for item in archive.infolist() if not item.is_dir()), normalized, strict=True):
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with archive.open(info, "r") as source_handle, target.open("wb") as target_handle:
                while True:
                    chunk = source_handle.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > limits.max_member_bytes:
                        raise CaptureImportError(f"archive member expanded beyond size limit: {name}")
                    target_handle.write(chunk)
            target.chmod(0o600)
    return set(normalized), total


def _safe_extract_tar(source: Path, destination: Path, *, limits: CaptureImportLimits) -> tuple[set[str], int]:
    names: list[str] = []
    members: list[tarfile.TarInfo] = []
    total = 0
    try:
        archive = tarfile.open(source, mode="r:*")
    except tarfile.TarError as exc:
        raise CaptureImportError("capture upload is not a valid TAR archive") from exc
    with archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            name = _safe_member_name(member.name)
            if not member.isfile():
                raise CaptureImportError(f"archive member is not a regular file: {name}")
            if member.size > limits.max_member_bytes:
                raise CaptureImportError(f"archive member exceeds size limit: {name}")
            names.append(name)
            members.append(member)
            total += int(member.size)
        normalized = _member_map(names, limits=limits)
        if total > limits.max_uncompressed_bytes:
            raise CaptureImportError("archive uncompressed size exceeds the import limit")
        for member, name in zip(members, normalized, strict=True):
            extracted = archive.extractfile(member)
            if extracted is None:
                raise CaptureImportError(f"archive member could not be read: {name}")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with extracted, target.open("wb") as target_handle:
                shutil.copyfileobj(extracted, target_handle, 1024 * 1024)
            target.chmod(0o600)
    return set(normalized), total


def import_capture_bundle(
    archive_path: Path,
    scan_dir: Path,
    *,
    limits: CaptureImportLimits = CaptureImportLimits(),
) -> dict[str, Any]:
    """Safely import a ZIP/TAR sensor bundle into ``scan_dir/capture``."""

    if archive_path.stat().st_size > limits.max_archive_bytes:
        raise CaptureImportError("capture archive exceeds the upload limit")
    capture_root = scan_dir / "capture"
    temporary_root = scan_dir / ".capture-importing"
    if capture_root.exists() or temporary_root.exists():
        raise CaptureImportError("capture import destination already exists")
    temporary_root.mkdir(parents=True)
    try:
        try:
            names, archive_bytes = _safe_extract_zip(archive_path, temporary_root, limits=limits)
        except CaptureImportError:
            # A TAR upload should still receive the TAR-specific parser.  Do
            # not mask unsafe ZIP errors for files that are clearly ZIPs.
            with archive_path.open("rb") as handle:
                signature = handle.read(4)
            if signature[:2] == b"PK":
                raise
            names, archive_bytes = _safe_extract_tar(archive_path, temporary_root, limits=limits)
        manifest_name = next((name for name in CAPTURE_MANIFEST_NAMES if name in names), None)
        if manifest_name is None:
            manifest = _derive_open_camera_manifest(names)
            if manifest is None:
                raise CaptureImportError(
                    "capture archive must contain capture_manifest.json or an OpenCamera Sensors video with *_accel.csv, *_gyro.csv"
                )
        else:
            manifest = _read_capture_manifest(temporary_root / manifest_name)
        if manifest.get("schema") == "noesis.phone_capture.browser.v1":
            # Browser camera/IMU evidence has a distinct clock and calibration
            # contract.  Keep it out of the native CSV/OpenVINS path while
            # reusing the same bounded, symlink-safe archive extraction.
            from .browser_capture import import_browser_capture_from_extracted

            return import_browser_capture_from_extracted(
                temporary_root,
                scan_dir,
                archive_path,
                manifest_name=manifest_name or "capture_manifest.json",
                member_names=names,
                archive_uncompressed_bytes=archive_bytes,
                manifest=manifest,
                limits=limits,
            )
        normalized = validate_capture_manifest(manifest, member_names=names)
        normalized_imu = normalized["imu"]
        if normalized_imu["sample_clock_mode"] == "separate_streams":
            accel_rows, accel_metrics = parse_imu_axis_csv(
                (temporary_root / normalized_imu["accel_path"]).read_bytes(),
                normalized_imu,
                kind="accel",
                limits=limits,
            )
            gyro_rows, gyro_metrics = parse_imu_axis_csv(
                (temporary_root / normalized_imu["gyro_path"]).read_bytes(),
                normalized_imu,
                kind="gyro",
                limits=limits,
            )
            imu_rows = {"accel": accel_rows, "gyro": gyro_rows}
            imu_metrics = {
                "sample_count": min(accel_metrics["sample_count"], gyro_metrics["sample_count"]),
                "start_time_ns": min(accel_metrics["start_time_ns"], gyro_metrics["start_time_ns"]),
                "end_time_ns": max(accel_metrics["end_time_ns"], gyro_metrics["end_time_ns"]),
                "duration_s": (
                    max(accel_metrics["end_time_ns"], gyro_metrics["end_time_ns"])
                    - min(accel_metrics["start_time_ns"], gyro_metrics["start_time_ns"])
                ) * 1e-9,
                "max_gap_s": max(accel_metrics["max_gap_s"], gyro_metrics["max_gap_s"]),
                "gap_count_over_limit": accel_metrics["gap_count_over_limit"] + gyro_metrics["gap_count_over_limit"],
                "strictly_increasing": True,
                "finite": True,
                "streams": {"accel": accel_metrics, "gyro": gyro_metrics},
            }
        else:
            imu_path = temporary_root / normalized_imu["path"]
            imu_rows, imu_metrics = parse_imu_csv(imu_path.read_bytes(), normalized_imu, limits=limits)
        video_path = temporary_root / normalized["video"]["path"]
        encoded_width, encoded_height = _probe_video_metadata(video_path)
        camera = normalized["camera"]
        calibrated_resolution = camera.get("resolution_px")
        calibrated_width, calibrated_height = (
            calibrated_resolution if isinstance(calibrated_resolution, list) else (None, None)
        )
        if calibrated_width is None or calibrated_height is None:
            # Unknown calibration is retained as null in the manifest while
            # the actual encoded geometry is still recorded for later calibration.
            camera["resolution_px"] = [encoded_width, encoded_height]
            camera["crop"] = {"left": 0, "top": 0, "width": encoded_width, "height": encoded_height}
        else:
            expected = (calibrated_width, calibrated_height)
            rotated_expected = (calibrated_height, calibrated_width)
            orientation = int(camera.get("orientation_deg") or 0) % 360
            if (encoded_width, encoded_height) not in {expected, rotated_expected if orientation in {90, 270} else expected}:
                raise CaptureImportError(
                    "encoded video dimensions do not match calibrated camera resolution and orientation"
                )
        crop = camera.get("crop")
        geometry_ok = (
            int(camera.get("orientation_deg") or 0) % 360 == 0
            and isinstance(crop, Mapping)
            and int(crop.get("left") or 0) == 0
            and int(crop.get("top") or 0) == 0
            and int(crop.get("width") or 0) == encoded_width
            and int(crop.get("height") or 0) == encoded_height
        )
        normalized["video"]["encoded_resolution_px"] = [encoded_width, encoded_height]
        frame_timestamp_path = normalized["video"].get("frame_timestamps_path")
        android_timing = None
        if "android_capture" in normalized:
            from .android_capture import verify_android_capture

            container_timing: dict[str, Any] = {}
            container_timestamps = _probe_video_timestamps(
                video_path, maximum=limits.max_video_timestamps, timestamp_details=container_timing,
                timeout_s=limits.video_probe_timeout_s,
            )
            acquisition_timestamps, android_timing = verify_android_capture(
                temporary_root, normalized,
                frame_count=len(container_timestamps), container_timing=container_timing,
                maximum_rows=limits.max_video_timestamps,
            )
            acquisition_verified = acquisition_timestamps is not None
            normalized["calibration"]["clock_source_verified"] = acquisition_verified
            normalized["calibration"]["camera_acquisition_timestamp_verified"] = acquisition_verified
            normalized["clocks"]["timestamp_source"] = "realtime" if acquisition_verified else "unverified"
            stabilization = (
                (android_timing.get("camera_settings") or {}).get("stabilization", "unknown")
                if acquisition_verified else "unknown"
            )
            normalized["camera"]["stabilization"] = stabilization
            normalized["video"]["stabilization"] = stabilization
            if acquisition_verified:
                video_timestamps = acquisition_timestamps
                video_timestamp_source = "android_camera2_sensor_timestamp_exact_encoder_association"
            else:
                # Retain all original sidecars for review. A missing or failed
                # association still permits RGB preparation using encoded PTS.
                frame_timestamp_path = None
                video_timestamps = container_timestamps
                video_timestamp_source = "ffprobe.best_effort_timestamp_time"
        elif frame_timestamp_path:
            video_timestamps = parse_frame_timestamps_csv(
                (temporary_root / frame_timestamp_path).read_bytes(),
                timestamp_unit=str(normalized["video"].get("timestamp_unit") or "ns"),
                limits=limits,
            )
            video_timestamp_source = "recorder_frame_timestamps_csv"
            container_timestamps = _probe_video_timestamps(
                video_path, maximum=limits.max_video_timestamps, timeout_s=limits.video_probe_timeout_s,
            )
            if len(container_timestamps) != len(video_timestamps):
                raise CaptureImportError(
                    "recorder frame timestamp count does not match the encoded video frame count"
                )
        else:
            video_timestamps = _probe_video_timestamps(
                video_path, maximum=limits.max_video_timestamps, timeout_s=limits.video_probe_timeout_s,
            )
            video_timestamp_source = "ffprobe.best_effort_timestamp_time"
        camera_start_ns = (
            None if android_timing is not None
            else normalized["clocks"].get("camera_start_time_ns")
        )
        if frame_timestamp_path:
            # Recorder timestamps are already integer nanoseconds after the
            # parser conversion.  Do not multiply them by 1e9 again.
            video_time_ns = [int(value) + int(camera_start_ns or 0) for value in video_timestamps]
        else:
            # ffprobe reports seconds relative to the container timeline.
            video_time_ns = [int(round(float(value) * 1e9)) + int(camera_start_ns or 0) for value in video_timestamps]
        camera_start_for_coverage = video_time_ns[0]
        camera_end_for_coverage = video_time_ns[-1]
        offset_ns = normalized["clocks"].get("imu_to_camera_offset_ns")
        offset = int(round(offset_ns)) if offset_ns is not None else 0
        if normalized["imu"]["sample_clock_mode"] == "separate_streams":
            stream_metrics = imu_metrics.get("streams") or {}
            starts = [
                int(row["start_time_ns"]) + offset
                for row in stream_metrics.values()
                if isinstance(row, Mapping)
            ]
            ends = [
                int(row["end_time_ns"]) + offset
                for row in stream_metrics.values()
                if isinstance(row, Mapping)
            ]
            if len(starts) != 2 or len(ends) != 2:
                raise CaptureImportError("separate IMU stream metrics are incomplete")
            # Metric admission needs the interval shared by every sensor
            # stream.  Keep each raw range in the report, but never let one
            # stream's extra samples hide another stream's missing endpoint.
            mapped_imu_start = max(starts)
            mapped_imu_end = min(ends)
        else:
            mapped_imu_start = int(imu_metrics["start_time_ns"]) + offset
            mapped_imu_end = int(imu_metrics["end_time_ns"]) + offset
        coverage = {
            "video_start_ns": camera_start_for_coverage,
            "video_end_ns": camera_end_for_coverage,
            "imu_start_ns": mapped_imu_start,
            "imu_end_ns": mapped_imu_end,
            "starts_before_video": mapped_imu_start <= camera_start_for_coverage,
            "ends_after_video": mapped_imu_end >= camera_end_for_coverage,
        }
        calibration = normalized["calibration"]
        calibration["camera_geometry"] = geometry_ok
        calibration["complete_for_metric_vio"] = bool(
            # Android clock and actual stabilization evidence is established
            # after manifest normalization. Recompute from the verified fields.
            all(calibration.get(key) is True for key in (
                "camera_intrinsics", "camera_to_imu_extrinsics", "time_offset",
                "clock_source_verified", "imu_noise", "distortion_supported",
            ))
            and normalized["camera"]["stabilization"] == "off"
            and normalized["clocks"]["camera_domain"]
            and normalized["clocks"]["imu_domain"]
            and geometry_ok
            and coverage["starts_before_video"]
            and coverage["ends_after_video"]
            and imu_metrics["gap_count_over_limit"] == 0
        )
        report = {
            "schema": CAPTURE_SCHEMA,
            "imported_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "archive_size_bytes": archive_path.stat().st_size,
            "archive_uncompressed_bytes": archive_bytes,
            "manifest": normalized,
            "imu": imu_metrics,
            "video": {
                "frame_count": len(video_timestamps),
                "start_time_ns": video_time_ns[0],
                "end_time_ns": video_time_ns[-1],
                "duration_s": (video_time_ns[-1] - video_time_ns[0]) * 1e-9,
                "timestamp_source": video_timestamp_source,
            },
            "coverage": coverage,
            "calibration": calibration,
            "metric_vio_allowed": bool(calibration["complete_for_metric_vio"]),
            "raw_streams_preserved": True,
        }
        if android_timing is not None:
            report["android_capture"] = android_timing
            report["video"]["camera_acquisition_timestamp_verified"] = android_timing["camera_acquisition_timestamp_verified"]
        capture_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_root, capture_root)
        (capture_root / "video_timestamps_ns.json").write_text(
            json.dumps(video_time_ns, separators=(",", ":")), encoding="utf-8"
        )
        normalized_imu_path = capture_root / "imu_normalized.json"
        normalized_imu_path.write_text(
            json.dumps(imu_rows, separators=(",", ":")), encoding="utf-8"
        )
        (capture_root / "capture_import.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        for path in (capture_root / "video_timestamps_ns.json", normalized_imu_path, capture_root / "capture_import.json"):
            path.chmod(0o600)
        report["video_path"] = str(Path("capture") / normalized["video"]["path"])
        if manifest_name:
            report["manifest_path"] = str(Path("capture") / manifest_name)
        report["import_report_path"] = "capture/capture_import.json"
        report["video_timestamps_path"] = "capture/video_timestamps_ns.json"
        report["imu_normalized_path"] = "capture/imu_normalized.json"
        return report
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


__all__ = [
    "CAPTURE_SCHEMA",
    "CaptureImportError",
    "CaptureImportLimits",
    "import_capture_bundle",
    "parse_imu_csv",
    "validate_capture_manifest",
]
