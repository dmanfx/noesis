"""Bounded, output-only native RoomWalk ChArUco calibration.

The caller owns upload retention and scheduling. No function here modifies a
capture, selects a live calibration, launches VIO, or admits metric authority.
``camera_calibration`` is a server-resolved result, never an uploaded path.
"""

from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import re
import selectors
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import RotationSpline

from .corner_refinement import POLICY as CORNER_REFINEMENT_POLICY, refine_native_corners
from .imu_noise_calibration import short_noise_model_has_evidence

REQUEST_SCHEMA = "roomwalk.calibration_request.v1"
REPORT_SCHEMA = "roomwalk.calibration_report.v1"
CAMERA_SCHEMA = "roomwalk.camera_calibration.v1"
BASALT_COMMIT = "6d8637b9d68ea18a1a63c1baa72818779e156932"
DEFAULT_BOARD = {
    "squares_x": 10,
    "squares_y": 14,
    "square_length_m": 0.018,
    "marker_length_m": 0.0132,
    "dictionary": "DICT_4X4_1000",
    "marker_ids": list(range(300, 370)),
    "legacy_pattern": False,
}
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class CalibrationError(ValueError):
    """Invalid input or a bounded processing failure; never an admission."""


class CalibrationCancelled(CalibrationError):
    pass


@dataclass(frozen=True)
class CalibrationSettings:
    executable: Path | None = None
    camera_calibration_dir: Path | None = None
    solver_path: Path | None = None
    noise_calibration_dir: Path | None = None
    timeout_s: float = 1200
    max_duration_s: float = 120
    max_camera_frames: int = 120
    max_imu_frames: int = 900
    max_imu_rows: int = 200_000
    camera_sample_hz: float = 2
    imu_sample_hz: float = 10
    max_iterations: int = 60
    knot_spacing_ns: int = 60_000_000
    # Explicit numerical residual normalization, NOT measured sensor noise.
    gyro_sample_sigma_rad_s: float = 0.03
    accel_sample_sigma_m_s2: float = 0.3

    @classmethod
    def from_env(cls):
        path = (
            os.environ.get("NOESIS_PHONE_SCAN_CALIBRATION_SOLVER")
            or os.environ.get("NOESIS_ROOMWALK_CALIBRATION_EXECUTABLE", "")
        ).strip()
        return cls(executable=Path(path) if path else None)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise CalibrationError(f"{label} must be a finite number")
    return float(value)


def _integer(value, label, low, high):
    if type(value) is not int or not low <= value <= high:
        raise CalibrationError(f"{label} must be an integer in [{low}, {high}]")
    return value


def _read_json(path: Path, maximum=32 * 1024 * 1024):
    if path.stat().st_size > maximum:
        raise CalibrationError(f"{path.name} exceeds its byte limit")

    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise CalibrationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(),
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(
            CalibrationError("nonfinite JSON")
        ),
    )


def _write(path: Path, value):
    # output_dir is exclusively owned by this run. No capture writes occur.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _member(root: Path, name: Any, maximum: int) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise CalibrationError("invalid capture member path")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise CalibrationError("capture member escapes capture directory")
    path = root / relative
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
        or not path.is_file()
    ):
        raise CalibrationError(f"capture member is missing or unsafe: {name}")
    if not 0 < path.stat().st_size <= maximum:
        raise CalibrationError(f"capture member size outside limits: {name}")
    return path


def _board_object(board):
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, board["dictionary"])
    )
    result = cv2.aruco.CharucoBoard(
        (board["squares_x"], board["squares_y"]),
        board["square_length_m"],
        board["marker_length_m"],
        dictionary,
        ids=np.asarray(board["marker_ids"], np.int32),
    )
    result.setLegacyPattern(board["legacy_pattern"])
    return result


def validate_calibration_request(payload) -> dict:
    if not isinstance(payload, Mapping) or payload.get("schema") != REQUEST_SCHEMA:
        raise CalibrationError(f"request must use {REQUEST_SCHEMA}")
    if not isinstance(payload.get("mode"), str) or payload["mode"] not in {
        "camera",
        "imu",
    }:
        raise CalibrationError("mode must be camera or imu")
    raw = payload.get("board", DEFAULT_BOARD)
    if not isinstance(raw, Mapping):
        raise CalibrationError("board must be an object")
    sx = _integer(raw.get("squares_x", 10), "squares_x", 3, 30)
    sy = _integer(raw.get("squares_y", 14), "squares_y", 3, 30)
    square = _finite(raw.get("square_length_m", 0.018), "square_length_m")
    marker = _finite(raw.get("marker_length_m", 0.0132), "marker_length_m")
    if not 0.001 <= marker < square <= 0.2 or max(sx, sy) * square > 3:
        raise CalibrationError("invalid metric board dimensions")
    name = raw.get("dictionary", "DICT_4X4_1000")
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"DICT_[4567]X[4567]_(50|100|250|1000)", name)
        or not hasattr(cv2.aruco, name)
    ):
        raise CalibrationError("unsupported explicit ArUco dictionary")
    legacy = raw.get("legacy_pattern", False)
    if type(legacy) is not bool:
        raise CalibrationError("legacy_pattern must be boolean")
    count = sx * sy // 2
    ids = raw.get("marker_ids")
    if ids is None:
        first = _integer(raw.get("marker_start_id", 300), "marker_start_id", 0, 999)
        ids = list(range(first, first + count))
    maximum = int(name.rsplit("_", 1)[1])
    if (
        not isinstance(ids, list)
        or len(ids) != count
        or any(type(i) is not int or not 0 <= i < maximum for i in ids)
        or len(set(ids)) != count
    ):
        raise CalibrationError(
            "marker_ids must uniquely cover the exact board within the dictionary"
        )
    board = {
        "squares_x": sx,
        "squares_y": sy,
        "square_length_m": square,
        "marker_length_m": marker,
        "dictionary": name,
        "marker_ids": list(ids),
        "legacy_pattern": legacy,
    }
    try:
        native = _board_object(board)
    except cv2.error as exc:
        raise CalibrationError(f"OpenCV rejected board: {exc}") from exc
    points = native.getChessboardCorners().astype(float).tolist()
    if "charuco_corners" in raw:
        given = np.asarray([p["point_m"] for p in raw["charuco_corners"]], float)
        if given.shape != (len(points), 3) or not np.allclose(
            given, points, atol=5e-8, rtol=0
        ):
            raise CalibrationError(
                "supplied ChArUco geometry contradicts board definition"
            )
    if "markers" in raw:
        given = np.asarray([p["corners_m_tl_tr_br_bl"] for p in raw["markers"]], float)
        if given.shape != np.asarray(native.getObjPoints()).shape or not np.allclose(
            given, native.getObjPoints(), atol=5e-8, rtol=0
        ):
            raise CalibrationError(
                "supplied marker geometry contradicts board definition"
            )
    request_id = payload.get("request_id")
    if request_id is not None and (
        not isinstance(request_id, str) or not SAFE_ID.fullmatch(request_id)
    ):
        raise CalibrationError("invalid request_id")
    provisional = payload.get("allow_provisional_camera", False)
    if type(provisional) is not bool:
        raise CalibrationError("allow_provisional_camera must be boolean")
    confirmed = payload.get("board_geometry_confirmed", False)
    if type(confirmed) is not bool:
        raise CalibrationError("board_geometry_confirmed must be boolean")
    result = {
        "schema": REQUEST_SCHEMA,
        "mode": payload["mode"],
        "board": board,
        "board_sha256": _digest({"definition": board, "object_points": points}),
        "allow_provisional_camera": provisional,
        "board_geometry_confirmed": confirmed,
    }
    if request_id is not None:
        result["request_id"] = request_id
    for key in ("camera_calibration_id", "noise_calibration_id", "capture_id"):
        if key in payload:
            if not isinstance(payload[key], str) or not SAFE_ID.fullmatch(payload[key]):
                raise CalibrationError(f"invalid {key}")
            result[key] = payload[key]
    if "camera_calibration" in payload:
        raise CalibrationError(
            "camera reference must be resolved by the server via settings, not supplied inline"
        )
    return result


class _Run:
    def __init__(self, settings, progress, cancelled):
        self.settings = settings
        self.progress = progress or (lambda *_: None)
        self.cancelled = cancelled or (lambda: False)
        self.deadline = time.monotonic() + settings.timeout_s

    def check(self):
        if self.cancelled():
            raise CalibrationCancelled("calibration cancelled")
        if time.monotonic() > self.deadline:
            raise CalibrationError("calibration deadline exceeded")

    def update(self, fraction, message):
        self.check()
        self.progress(float(fraction), message)


def _settings(value):
    if value is None:
        value = CalibrationSettings.from_env()
    elif isinstance(value, Mapping):
        value = CalibrationSettings(**value)
    if not isinstance(value, CalibrationSettings):
        raise CalibrationError(
            "settings must be CalibrationSettings or a matching mapping"
        )
    if not 1 <= value.timeout_s <= 1800 or not 1 <= value.max_duration_s <= 120:
        raise CalibrationError("calibration time limits outside bounds")
    for number, low, high in (
        (value.max_camera_frames, 12, 240),
        (value.max_imu_frames, 12, 1500),
        (value.max_imu_rows, 100, 200000),
        (value.max_iterations, 1, 150),
    ):
        _integer(number, "computational limit", low, high)
    if not 0 < value.camera_sample_hz <= 10 or not 0 < value.imu_sample_hz <= 30:
        raise CalibrationError("invalid sampling rate")
    for sigma in (value.gyro_sample_sigma_rad_s, value.accel_sample_sigma_m_s2):
        if _finite(sigma, "residual weight") <= 0:
            raise CalibrationError("residual weights must be positive")
    return value


def _load_camera_reference(directory):
    if directory is None:
        return None
    root = Path(directory).resolve()
    report = _read_json(_member(root, "report.json", 8 * 1024 * 1024))
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("mode") != "camera"
        or report.get("status") != "completed"
    ):
        raise CalibrationError(
            "camera reference job did not complete camera calibration"
        )
    info = report.get("artifacts", {}).get("camera_result.json")
    if not isinstance(info, Mapping) or info.get("path") != "camera_result.json":
        raise CalibrationError(
            "camera reference report has no exact camera-result artifact"
        )
    path = _member(root, "camera_result.json", 8 * 1024 * 1024)
    if path.stat().st_size != info.get("bytes") or _sha(path) != info.get("sha256"):
        raise CalibrationError("camera reference artifact differs from its report hash")
    reference = _read_json(path)
    if reference.get("schema") != CAMERA_SCHEMA:
        raise CalibrationError("camera reference artifact schema mismatch")
    return reference


def _load_noise_reference(directory, manifest, timing):
    """Consume an optional separately qualified, hash-bound noise job.

    A noise result may contain other diagnostics. This consumer deliberately
    admits only an exact device/sensor binding; mismatches remain provisional.
    """
    if directory is None:
        return {
            "imu_noise_calibrated": False,
            "noise_model_usable": False,
            "noise_model_status": "not_supplied",
            "noise_provenance": {},
            "reason_codes": ["noise_reference_not_supplied"],
        }
    root = Path(directory).resolve()
    report = _read_json(_member(root, "report.json", 8 * 1024 * 1024))
    if report.get("status") != "completed":
        raise CalibrationError("noise reference job did not complete")
    info = report.get("artifacts", {}).get("noise_result.json")
    if not isinstance(info, Mapping) or info.get("path") != "noise_result.json":
        raise CalibrationError("noise job report lacks exact noise-result artifact")
    path = _member(root, "noise_result.json", 8 * 1024 * 1024)
    if path.stat().st_size != info.get("bytes") or _sha(path) != info.get("sha256"):
        raise CalibrationError("noise artifact differs from its report hash")
    reference = _read_json(path)
    binding = reference.get("binding", {})
    if not binding and reference.get("schema") == "roomwalk.imu_noise_calibration.v1":
        binding = {
            "device": reference.get("device"),
            "sensors": reference.get("sensors"),
        }
    raw_device = timing.get("recorder", {}).get("raw_device", {})
    expected_device = {**manifest.get("device", {})}
    for key in ("android_api_level", "build_fingerprint"):
        if key in raw_device:
            expected_device[key] = raw_device[key]
    expected_sensors = timing.get("recorder", {}).get("sensors", {})
    device_fields = ("id", "model", "android_api_level", "build_fingerprint")
    reference_device = binding.get("device") or {}
    same_device = bool(expected_device.get("id")) and all(
        reference_device.get(key) == expected_device.get(key) for key in device_fields
    )

    def sensor_identity(sensors):
        if not isinstance(sensors, Mapping):
            return None
        fields = (
            "sensor_id",
            "android_sensor_id",
            "type",
            "type_name",
            "name",
            "vendor",
            "version",
            "minimum_delay_us",
            "maximum_range",
            "resolution",
            "units",
            "axes",
            "uncalibrated",
            "bias_fields_are_sensor_estimates",
            "accelerometer_includes_gravity",
        )
        identities = {}
        for name, row in sensors.items():
            if not isinstance(row, Mapping):
                return None
            identity = {key: row.get(key) for key in fields}
            # Android Sensor getters return IEEE float32. JSONObject.put(double)
            # promotes that exact float, whereas put(Object) preserves Float and
            # emits its shortest decimal. Compare their native values, not JSON
            # decimal spellings. One float32 ULP still counts as a changed sensor.
            for key in ("maximum_range", "resolution"):
                value = identity[key]
                if value is not None:
                    if (type(value) not in (int, float) or not math.isfinite(value)
                            or not 0 < value <= np.finfo(np.float32).max):
                        return None
                    identity[key] = float(np.float32(value))
                    if identity[key] <= 0:
                        return None
            identities[name] = identity
        return identities

    expected_sensor_identity = sensor_identity(expected_sensors)
    same_sensors = (bool(expected_sensors) and expected_sensor_identity is not None
                    and sensor_identity(binding.get("sensors")) == expected_sensor_identity)
    keys = (
        "gyroscope_noise_density",
        "gyroscope_random_walk",
        "accelerometer_noise_density",
        "accelerometer_random_walk",
    )
    noise = reference.get("noise", {})
    positive = isinstance(noise, Mapping) and all(
        type(noise.get(k)) in (int, float) and math.isfinite(noise[k]) and noise[k] > 0
        for k in keys
    )
    reasons = []
    short_candidate = reference.get("noise_model_status") == "short_session_candidate"
    short_verified = short_candidate and short_noise_model_has_evidence(reference)
    provenance = reference.get("noise_provenance", {})
    all_measured = (
        reference.get("imu_noise_calibrated") is True and not short_candidate
        and reference.get("method") != "short_session"
        and isinstance(provenance, Mapping)
        and all(isinstance(item, Mapping) and item.get("measured") is True
                and item.get("kind") == "measured" for item in provenance.values())
    )
    if not (all_measured or short_verified):
        reasons.append("noise_job_not_qualified")
    if not same_device:
        reasons.append("noise_device_binding_mismatch")
    if not same_sensors:
        reasons.append("noise_sensor_binding_mismatch")
    if not positive:
        reasons.append("noise_coefficients_incomplete")
    units = reference.get("unit_conventions", reference.get("units", {}))
    expected_units = {
        "gyroscope_noise_density": "rad/s/sqrt(Hz)",
        "accelerometer_noise_density": "m/s^2/sqrt(Hz)",
        "gyroscope_random_walk": "rad/s^2/sqrt(Hz)",
        "accelerometer_random_walk": "m/s^3/sqrt(Hz)",
    }
    if units != expected_units:
        reasons.append("noise_unit_conventions_not_matched")
    return {
        "imu_noise_calibrated": all_measured and not reasons,
        "noise_model_usable": not reasons,
        "noise_model_status": ("short_session_candidate" if short_candidate else "full_allan_measured") if not reasons else "unusable_reference",
        "noise_provenance": reference.get("noise_provenance", {}),
        "reason_codes": reasons,
        "source_sha256": info["sha256"],
        "source_bundle_sha256": reference.get("source_bundle_sha256"),
        "noise": noise,
        "unit_conventions": units,
        "binding": binding,
    }


def _capture(root: Path, run: _Run):
    from .android_capture import verify_android_capture

    report_path = _member(root, "capture_import.json", 2 * 1024 * 1024)
    report = _read_json(report_path)
    if report.get("schema") != "noesis.phone_capture.v1":
        raise CalibrationError("requires an imported native capture report")
    manifest = report["manifest"]
    if "android_capture" not in manifest:
        raise CalibrationError(
            "calibration requires the native RoomWalk Camera2 capture path"
        )
    video = _member(root, manifest["video"]["path"], 8 * 1024**3)
    paths = [report_path, video]
    evidence = manifest["android_capture"]
    for key in ("camera_results_path", "capture_result_path", "encoder_pts_path"):
        paths.append(_member(root, evidence[key], 64 * 1024 * 1024))
    paths.append(
        _member(root, manifest["video"]["frame_timestamps_path"], 8 * 1024 * 1024)
    )
    for key in ("accel_path", "gyro_path"):
        paths.append(_member(root, manifest["imu"][key], 64 * 1024 * 1024))
    if report["video"]["duration_s"] > run.settings.max_duration_s:
        raise CalibrationError(
            "capture exceeds calibration duration limit; record a bounded target session"
        )
    run.update(0.03, "Verifying native capture timing and source files")
    # MP4 samples are the exact MediaMuxer/encoder access units retained by this
    # native recorder. Demux their PTS; do not decode full 8K once merely to probe
    # and again to localize corners. The prior importer already checked frames.
    # Cap at 4001 samples so an excessive recording cannot expand probe output.
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-read_intervals",
        "%+#4001",
        "-show_entries",
        "stream=width,height,time_base,nb_frames:stream_side_data=rotation:packet=pts",
        "-of",
        "json",
        str(video),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=min(30, max(1, int(run.deadline - time.monotonic()))),
    )
    if len(completed.stdout) > 2 * 1024 * 1024:
        raise CalibrationError("container probe exceeded byte bound")
    probe = json.loads(completed.stdout)
    stream = probe["streams"][0]
    packets = probe.get("packets", [])
    if not 2 <= len(packets) <= 4000 or int(stream.get("nb_frames", 0)) != len(packets):
        raise CalibrationError(
            "MP4 sample count is incomplete or exceeds calibration limit"
        )
    ticks = [int(p["pts"]) for p in packets]
    if any(a >= b for a, b in zip(ticks, ticks[1:])):
        raise CalibrationError("native encoded packet PTS are not strictly ordered")
    if any(
        float(s.get("rotation", 0)) % 360 != 0 for s in stream.get("side_data_list", [])
    ):
        raise CalibrationError(
            "encoded rotation requires a separately modeled camera geometry"
        )
    details = {"time_base": stream["time_base"], "timestamps_ticks": ticks}
    times, timing = verify_android_capture(
        root,
        manifest,
        frame_count=len(packets),
        container_timing=details,
        maximum_rows=5000,
    )
    if times is None:
        raise CalibrationError(
            "Camera2/encoder association failed: " + "; ".join(timing["errors"])
        )
    raw_result = _read_json(root / evidence["capture_result_path"], 512 * 1024)
    timing.setdefault("recorder", {})["raw_device"] = raw_result.get("device", {})
    if (times[-1] - times[0]) * 1e-9 > run.settings.max_duration_s:
        raise CalibrationError("verified capture duration exceeds limit")
    size = (int(stream["width"]), int(stream["height"]))
    if min(size) < 64 or max(size) > 8192 or size[0] * size[1] > 36_000_000:
        raise CalibrationError(
            "encoded resolution outside supported calibration limits"
        )
    camera_rows = {}
    with (root / evidence["camera_results_path"]).open() as stream:
        for line in stream:
            if len(line) > 65536 or len(camera_rows) >= 5000:
                raise CalibrationError("Camera2 evidence exceeds row limits")
            row = json.loads(line)
            camera_rows[row["sensor_timestamp_ns"]] = row
    actual = [camera_rows[t] for t in times]
    provenance = {
        p.relative_to(root).as_posix(): {"sha256": _sha(p), "bytes": p.stat().st_size}
        for p in paths
    }
    return manifest, video, size, times, actual, timing, provenance


def _capture_binding(manifest, rows, size, timing):
    fields = (
        "active_physical_camera_id",
        "lens_focal_length_mm",
        "lens_focus_distance_diopters",
        "zoom_ratio",
        "crop_region",
        "ois_mode",
        "eis_mode",
        "rotate_and_crop_mode",
        "distortion_correction_mode",
    )
    values = {}
    reasons = []
    for key in fields:
        distinct = {json.dumps(r.get(key), sort_keys=True) for r in rows}
        values[key] = json.loads(next(iter(distinct))) if len(distinct) == 1 else None
        if len(distinct) != 1:
            reasons.append(f"changing_{key}")
        elif values[key] is None:
            reasons.append(f"missing_{key}")
    for key in ("ois_mode", "eis_mode", "rotate_and_crop_mode"):
        if values[key] != 0:
            reasons.append(f"unsupported_{key}")
    recorder = timing.get("recorder", {}).get("camera", {})
    focus = recorder.get("focus_control", {})
    if focus.get("mode") not in {"locked", "manual", "manual_locked"}:
        reasons.append("focus_not_explicitly_locked")
    distortion_control = None
    # Intrinsics describe the *encoded output* and can include a fixed vendor
    # image transform. Explicitly unsupported control is not a missing report
    # from an available control, and it is emphatically not evidence of OFF.
    # Keep raw mode null and retain this capability/software binding. This alone
    # does not qualify row timing: the mapper separately rechecks physical output
    # routing, sensor arrays, per-frame controls and the Camera2 coordinate contract.
    if "missing_distortion_correction_mode" in reasons:
        request_available = recorder.get("distortion_correction_request_key_available")
        result_available = recorder.get("distortion_correction_result_key_available")
        available_modes = recorder.get("distortion_correction_available_modes")
        if (
            request_available is False
            and result_available is False
            and "distortion_correction_available_modes" in recorder
            and available_modes in (None, [])
            and str(recorder.get("id")) == manifest["camera"]["id"]
            and isinstance(focus.get("build_fingerprint"), str)
            and focus["build_fingerprint"]
        ):
            reasons.remove("missing_distortion_correction_mode")
            distortion_control = {
                "state": "unsupported_control_encoded_output_calibrated",
                "reported_mode": None,
                "assumed_off": False,
                "request_key_available": False,
                "result_key_available": False,
                "available_modes": available_modes,
                "characteristics_camera_id": recorder.get("id"),
                "build_fingerprint": focus["build_fingerprint"],
                "row_mapping_qualified": False,
            }
    # Per-session counts are evidence, not optical geometry; they must not make
    # two captures with the same actual lens settings appear incompatible.
    binding = {
        "device": manifest.get("device"),
        "camera_id": manifest["camera"]["id"],
        "resolution": list(size),
        "orientation_deg": manifest["camera"].get("orientation_deg", 0),
        "actual_camera2": values,
        "focus_mode": focus.get("mode"),
        "focus_build_fingerprint": focus.get("build_fingerprint"),
    }
    if distortion_control is not None:
        binding["distortion_control"] = distortion_control
    if binding["orientation_deg"] != 0:
        reasons.append("encoded_orientation_requires_separate_calibration")
    return {
        "schema": "roomwalk.camera_binding.v1",
        "signature": binding,
        "sha256": _digest(binding),
        "qualified": not reasons,
        "reason_codes": reasons,
    }


def camera_binding_for_capture(capture_dir: Path) -> dict:
    """Recheck imported native geometry for a <=10-minute walk, without video IO.

    Container decoding/timing remains the prior importer's responsibility. This
    checks every original encoded/Camera2 association and actual optical setting,
    independently of the calibration worker's shorter duration/frame limits.
    Unqualified changing/missing geometry is returned with reason codes; malformed
    or unverified evidence raises CalibrationError. No capture files are changed.
    """
    try:
        return _camera_binding_for_capture(capture_dir)
    except CalibrationError:
        raise
    except (KeyError, ValueError, TypeError, OSError, csv.Error) as exc:
        raise CalibrationError(
            f"invalid native camera binding evidence: {exc}"
        ) from exc


def _camera_binding_for_capture(capture_dir):
    root = Path(capture_dir).resolve()
    report = _read_json(_member(root, "capture_import.json", 2 * 1024**2))
    if (
        report.get("schema") != "noesis.phone_capture.v1"
        or report.get("video", {}).get("camera_acquisition_timestamp_verified")
        is not True
        or report.get("android_capture", {}).get(
            "camera_acquisition_timestamp_verified"
        )
        is not True
    ):
        raise CalibrationError("camera binding requires a verified native import")
    manifest = report["manifest"]
    evidence = manifest["android_capture"]
    count = _integer(report["video"]["frame_count"], "frame_count", 2, 20000)
    if not 0 < _finite(report["video"]["duration_s"], "duration_s") <= 605:
        raise CalibrationError("walk exceeds native ten-minute binding limit")
    raw = _read_json(_member(root, evidence["capture_result_path"], 512 * 1024))
    if raw.get("schema") != "noesis.phone_capture.android_result.v1":
        raise CalibrationError("unsupported raw native capture result")
    cam = raw["camera"]
    size = manifest["video"]["encoded_resolution_px"]
    if (
        [cam.get("width"), cam.get("height")] != size
        or str(cam.get("id")) != manifest["camera"]["id"]
        or raw.get("dropped_metadata_records") != 0
        or raw.get("timing", {}).get("exact_frame_association_verified") is not True
    ):
        raise CalibrationError(
            "raw recorder identity/timing disagrees with native import"
        )
    camera_rows = {}
    with _member(root, evidence["camera_results_path"], 128 * 1024**2).open() as stream:
        previous = -1
        for line in stream:
            if len(line) > 65536 or len(camera_rows) >= 21000:
                raise CalibrationError("camera binding row limit exceeded")
            row = json.loads(line)
            stamp = _integer(
                row.get("sensor_timestamp_ns"), "sensor timestamp", 0, 2**63 - 1
            )
            if stamp // 1000 <= previous:
                raise CalibrationError("Camera2 rows are not strictly ordered")
            previous = stamp // 1000
            camera_rows[stamp] = row

    def read_csv(name):
        with _member(root, name, 8 * 1024**2).open() as stream:
            result = []
            for row in csv.DictReader(stream):
                if len(result) >= 20000:
                    raise CalibrationError("encoded binding row limit exceeded")
                result.append({k: int(v) for k, v in row.items()})
            return result

    encoded = read_csv(evidence["encoder_pts_path"])
    association = read_csv(manifest["video"]["frame_timestamps_path"])
    if len(encoded) != count or len(association) != count:
        raise CalibrationError("binding association count differs from verified import")
    actual = []
    previous = -1
    for i, (enc, match) in enumerate(zip(encoded, association)):
        stamp = match["timestamp_ns"]
        row = camera_rows.get(stamp)
        if (
            row is None
            or stamp <= previous
            or enc["encoded_index"] != i
            or match["encoded_index"] != i
            or enc["encoded_pts_us"] != stamp // 1000
            or match["encoded_pts_us"] != enc["encoded_pts_us"]
            or match["frame_number"] != row["frame_number"]
            or enc["size_bytes"] <= 0
            or enc["flags"] & 2
        ):
            raise CalibrationError("original native frame association mismatch")
        actual.append(row)
        previous = stamp
    if (
        association[-1]["timestamp_ns"] - association[0]["timestamp_ns"]
        > 605_000_000_000
    ):
        raise CalibrationError("native association exceeds ten-minute binding limit")
    return _capture_binding(manifest, actual, size, {"recorder": raw})


def _sample_indices(times, hz, maximum):
    result = []
    last = None
    interval = round(1e9 / hz)
    for i, t in enumerate(times):
        if last is None or t - last >= interval:
            result.append(i)
            last = t
    if len(result) > maximum:
        result = [
            result[i]
            for i in np.linspace(0, len(result) - 1, maximum).round().astype(int)
        ]
    return result


def _detect_capture(video, size, times, indices, board, run, output):
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.markerBorderBits = 1
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.CharucoDetector(
        _board_object(board), detectorParams=detector_params
    )
    expression = "+".join(f"eq(n\\,{i})" for i in indices)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "2",
        "-noautorotate",
        "-i",
        str(video),
        "-vf",
        f"select={expression}",
        "-frames:v",
        str(len(indices)),
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-threads",
        "2",
        "pipe:1",
    ]
    rows = []
    w, h = size
    count = w * h
    # Nonblocking pipe reads make cancellation/deadlines effective during decode.
    with (output / "decode.stderr.log").open("x") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=log, bufsize=0
        )
        assert process.stdout is not None
        os.set_blocking(process.stdout.fileno(), False)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        buffer = bytearray()
        try:
            for index in indices:
                while len(buffer) < count:
                    run.check()
                    if not selector.select(0.2):
                        continue
                    chunk = os.read(
                        process.stdout.fileno(), min(1024 * 1024, count - len(buffer))
                    )
                    if not chunk:
                        raise CalibrationError(
                            "decoder ended before the selected frame"
                        )
                    buffer.extend(chunk)
                gray = np.frombuffer(buffer, dtype=np.uint8).reshape(h, w)
                corners, ids, marker_corners, markers = detector.detectBoard(gray)
                row = {
                    "frame_index": index,
                    "timestamp_ns": times[index],
                    "time_s": (times[index] - times[0]) * 1e-9,
                    "marker_count": 0 if markers is None else len(markers),
                    "ids": [],
                    "points": [],
                    "rejection_reasons": [],
                }
                if ids is not None:
                    ids = ids.reshape(-1)
                    pts, refinement, failures = refine_native_corners(
                        gray, corners, ids, marker_corners, board["squares_x"]
                    )
                    row["corner_refinement"] = refinement
                    row["rejection_reasons"].extend(failures)
                    row.update(
                        ids=ids.tolist(),
                        points=pts.astype(float).tolist(),
                        hull_fraction=float(
                            cv2.contourArea(cv2.convexHull(pts.astype(np.float32))) / (w * h)
                        ),
                    )
                if len(row["ids"]) < 20:
                    row["rejection_reasons"].append("fewer_than_20_corners")
                if row.get("hull_fraction", 0) < 0.02:
                    row["rejection_reasons"].append("target_hull_below_2_percent")
                if len(set(row["ids"])) != len(row["ids"]):
                    row["rejection_reasons"].append("duplicate_corner_id")
                rows.append(row)
                del gray
                buffer = bytearray()
                run.update(
                    0.1 + 0.4 * len(rows) / len(indices),
                    f"Detected target in {len(rows)}/{len(indices)} native frames",
                )
            process.stdout.close()
            try:
                code = process.wait(
                    timeout=min(10, max(0.1, run.deadline - time.monotonic()))
                )
            except subprocess.TimeoutExpired as exc:
                raise CalibrationError(
                    "decoder did not finish within its bound"
                ) from exc
            if code:
                raise CalibrationError("native video decoder failed")
        finally:
            selector.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
    return {
        "schema": "roomwalk.charuco_observations.v1",
        "opencv_version": cv2.__version__,
        "resolution": list(size),
        "detector": "full_native_CharucoDetector_no_K_D",
        "corner_refinement_policy": CORNER_REFINEMENT_POLICY,
        "decode_command": command,
        "frames": rows,
    }


def _points(row, objects):
    ids = np.asarray(row["ids"], np.int32)
    px = np.asarray(row["points"], np.float64)
    if (
        ids.ndim != 1
        or len(ids) < 6
        or px.shape != (len(ids), 2)
        or not np.isfinite(px).all()
        or np.any(ids < 0)
        or np.any(ids >= len(objects))
        or len(set(ids.tolist())) != len(ids)
    ):
        raise CalibrationError("invalid indexed ChArUco observations")
    return objects[ids], px, ids


def _fit_camera(rows, objects, size):
    pairs = [_points(r, objects) for r in rows]
    if len(pairs) < 8:
        raise CalibrationError("camera fit needs at least eight diverse training views")
    obj = [p[0].astype(np.float32) for p in pairs]
    img = [p[1].astype(np.float32) for p in pairs]
    result = cv2.calibrateCameraExtended(
        obj,
        img,
        tuple(size),
        None,
        None,
        flags=0,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-9),
    )
    rms, K, D, _, _, std, _, per = result
    if (
        not np.isfinite(K).all()
        or not np.isfinite(D).all()
        or not np.isfinite(std).all()
        or min(K[0, 0], K[1, 1]) <= 0
    ):
        raise CalibrationError("camera solve produced invalid parameters")
    return {
        "K": K.tolist(),
        "D": D.reshape(-1).tolist(),
        "resolution": list(size),
        "model": "opencv_pinhole",
        "training_rms_px": float(rms),
        "intrinsic_std": std.reshape(-1).tolist(),
        "per_view_rms_px": per.reshape(-1).tolist(),
        "flags": 0,
        "tangential_coefficients": "estimated",
    }


def _pose(row, objects, K, D, split=False):
    obj, img, ids = _points(row, objects)
    fit = (ids % 2 == 0) if split else np.ones(len(ids), bool)
    if fit.sum() < 6 or (split and (~fit).sum() < 6):
        raise CalibrationError("insufficient separate pose-fit and validation corners")
    ok, r, t = cv2.solvePnP(obj[fit], img[fit], K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or not np.isfinite(r).all() or not np.isfinite(t).all() or t[2, 0] <= 0:
        raise CalibrationError("target pose failed positive-depth validation")
    R = cv2.Rodrigues(r)[0]
    if np.any((R @ obj.T + t)[2] <= 0):
        raise CalibrationError("target point behind camera")
    predicted = cv2.projectPoints(obj, r, t, K, D)[0].reshape(-1, 2)
    e = np.linalg.norm(predicted - img, axis=1)
    camera = np.eye(4)
    camera[:3, :3] = R.T
    camera[:3, 3] = -R.T @ t.reshape(3)
    return (
        camera,
        e[~fit] if split else e,
        float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), 0, 1)))),
    )


def _error_summary(values):
    v = np.asarray(values, float)
    if v.size == 0 or not np.isfinite(v).all():
        raise CalibrationError("no finite validation residuals")
    return {
        "count": int(v.size),
        "rms": float(np.sqrt(np.mean(v * v))),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
    }


def _camera_score(rows, objects, model):
    K, D = np.array(model["K"]), np.array(model["D"])
    errors = []
    views = []
    for row in rows:
        _, e, tilt = _pose(row, objects, K, D, True)
        errors.extend(e)
        views.append(
            {
                "frame_index": row["frame_index"],
                "tilt_deg": tilt,
                "radial_px": _error_summary(e),
            }
        )
    return {"radial_px": _error_summary(errors), "views": views}


def fit_camera_observations(observations, request, binding, *, check=None):
    """Pure numerical entrypoint shared by CPU unit tests and the capture worker."""
    check = check or (lambda: None)
    request = validate_calibration_request(request)
    objects = _board_object(request["board"]).getChessboardCorners().astype(np.float64)
    rows = [r for r in observations["frames"] if not r.get("rejection_reasons")]
    if len(rows) < 16:
        raise CalibrationError("need at least sixteen usable native target views")
    # Blocked, not random frame leakage. Never exclude views by fitted error.
    first = rows[0]["timestamp_ns"]
    train = [r for r in rows if ((r["timestamp_ns"] - first) // 2_000_000_000) % 2 == 0]
    holdout = [
        r for r in rows if ((r["timestamp_ns"] - first) // 2_000_000_000) % 2 == 1
    ]
    if min(len(train), len(holdout)) < 8:
        raise CalibrationError(
            "insufficient views in independent two-second temporal blocks"
        )
    check()
    model = _fit_camera(train, objects, observations["resolution"])
    check()
    score = _camera_score(holdout, objects, model)
    check()
    reverse = _fit_camera(holdout, objects, observations["resolution"])
    check()
    reverse_score = _camera_score(train, objects, reverse)
    fx, fy = model["K"][0][0], model["K"][1][1]
    focal_delta = max(
        abs(reverse["K"][0][0] / fx - 1), abs(reverse["K"][1][1] / fy - 1)
    )
    allpx = np.concatenate([np.asarray(r["points"]) for r in rows])
    w, h = observations["resolution"]
    coverage = (allpx.max(0) - allpx.min(0)) / np.array([w, h])
    tilts = [v["tilt_deg"] for v in score["views"] + reverse_score["views"]]
    policy = {
        "id": "roomwalk.camera_quality.v1",
        "maximum_heldout_radial_rms_native_px": 2.0,
        "maximum_heldout_radial_p95_native_px": 4.0,
        "maximum_reverse_focal_fraction": 0.02,
        "minimum_coverage_fraction_xy": [0.55, 0.55],
        "minimum_tilt_span_deg": 15,
        "minimum_views_per_partition": 8,
    }
    reasons = list(binding["reason_codes"])
    for value in (score, reverse_score):
        if value["radial_px"]["rms"] > 2 or value["radial_px"]["p95"] > 4:
            reasons.append("heldout_reprojection_exceeds_policy")
    if focal_delta > 0.02:
        reasons.append("reverse_split_focal_instability")
    if np.any(coverage < 0.55):
        reasons.append("insufficient_image_coverage")
    if max(tilts) - min(tilts) < 15:
        reasons.append("insufficient_tilt_diversity")
    # Intrinsics can be estimated on an unmeasured flat target, but metric
    # camera-IMU translation needs separately confirmed physical board scale.
    return {
        "schema": CAMERA_SCHEMA,
        "status": "computed",
        "model": model["model"],
        **model,
        "board": request["board"],
        "board_sha256": request["board_sha256"],
        "binding": binding,
        "quality": {
            "status": "qualified" if not reasons else "rejected",
            "reason_codes": sorted(set(reasons)),
            "policy": policy,
            "heldout": score,
            "reverse_heldout": reverse_score,
            "reverse_model": reverse,
            "reverse_focal_fraction": focal_delta,
            "coverage_fraction_xy": coverage.tolist(),
            "tilt_span_deg": max(tilts) - min(tilts),
        },
        "training_frame_indices": [r["frame_index"] for r in train],
        "holdout_frame_indices": [r["frame_index"] for r in holdout],
        "fit_uses_holdout": False,
        "final_model_refitted_on_holdout": False,
        "physical_board_scale_verified": request["board_geometry_confirmed"],
        "physical_board_scale_provenance": "user_attestation"
        if request["board_geometry_confirmed"]
        else "unverified",
        "accepted_for_metric_vio": False,
        "camera_imu_extrinsics_calibrated": False,
        "time_offset_calibrated": False,
        "imu_noise_calibrated": False,
    }


def camera_profile(reference):
    """Export only a qualified, hash-bound model; never activate it globally."""
    if reference.get("quality", {}).get("status") != "qualified":
        raise CalibrationError(
            "camera profile requires qualified heldout camera evidence"
        )
    binding = reference["binding"]
    K, D, _ = _camera_model(reference, reference["resolution"], binding, False)
    return {
        "schema": "roomwalk.camera_profile.v1",
        "intrinsics": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        },
        "distortion": D.tolist(),
        "distortion_model": "brown_conrady",
        "resolution_px": reference["resolution"],
        "binding": binding,
        "camera_result_sha256": _digest(reference),
        "source_manifest_sha256": reference.get("source_manifest_sha256"),
        "camera_intrinsics_calibrated": True,
        "accepted_for_metric_vio": False,
        "review_only": True,
    }


def _camera_model(reference, size, binding, allow_provisional):
    if reference.get("schema") != CAMERA_SCHEMA:
        raise CalibrationError("camera reference has unsupported schema")
    K = np.asarray(reference.get("K"), float)
    D = np.asarray(reference.get("D"), float)
    if (
        K.shape != (3, 3)
        or D.shape != (5,)
        or not np.isfinite(K).all()
        or not np.isfinite(D).all()
        or min(K[0, 0], K[1, 1]) <= 0
    ):
        raise CalibrationError(
            "camera reference needs exact finite K and all five D coefficients"
        )
    if (
        not np.array_equal(K[2], [0, 0, 1])
        or K[0, 1] != 0
        or K[1, 0] != 0
        or reference.get("resolution") != list(size)
    ):
        raise CalibrationError(
            "camera reference does not match native encoded geometry"
        )
    same = reference.get("binding", {}).get("sha256") == binding["sha256"]
    if not same:
        raise CalibrationError(
            "camera reference actual lens/focus/crop binding differs from this capture"
        )
    qualified = (
        reference.get("quality", {}).get("status") == "qualified"
        and binding["qualified"]
    )
    if not qualified and not allow_provisional:
        raise CalibrationError(
            "IMU mode requires a qualified matching camera or explicit allow_provisional_camera"
        )
    return K, D, qualified


def _imu_arrays(root, manifest, run, timing=None):
    from .capture import CaptureImportLimits, parse_imu_axis_csv

    result = {}
    if timing is not None:
        sensors = timing.get("recorder", {}).get("sensors", {})
        for key, unit in (("accelerometer", "m/s^2"), ("gyroscope", "rad/s")):
            sensor = sensors.get(key, {})
            if (
                sensor.get("axes") != "android_device_x_right_y_up_z_out_of_screen"
                or sensor.get("units") != unit
                or sensor.get("timestamp_source") != "android_elapsed_realtime_ns"
            ):
                raise CalibrationError(
                    "native IMU axes, SI units or acquisition clock are unverified"
                )
        if sensors["accelerometer"].get("accelerometer_includes_gravity") is not True:
            raise CalibrationError(
                "IMU calibration requires accelerometer including gravity"
            )
    for kind in ("accel", "gyro"):
        path = _member(root, manifest["imu"][kind + "_path"], 64 * 1024 * 1024)
        data, metrics = parse_imu_axis_csv(
            path.read_bytes(),
            manifest["imu"],
            kind=kind,
            limits=CaptureImportLimits(
                max_imu_rows=run.settings.max_imu_rows, max_time_gap_s=0.05
            ),
        )
        if metrics["gap_count_over_limit"]:
            raise CalibrationError(f"{kind} has a gap over 50ms")
        result[kind] = [
            {"timestamp_ns": r["timestamp_ns"], "xyz": r["si"]} for r in data
        ]
    return result


def _rotation_seed(frames, gyro):
    # Finite camera rotation increments, robustly aligned with averaged gyro at
    # interval midpoints. Grid-search offset is ONLY an initialization.
    t0 = frames[0]["timestamp_ns"]
    times = np.array([(r["timestamp_ns"] - t0) * 1e-9 for r in frames])
    poses = np.array([r["T_target_camera"] for r in frames])
    mid = (times[:-1] + times[1:]) / 2
    omega = (
        Rotation.from_matrix(
            np.swapaxes(poses[:-1, :3, :3], 1, 2) @ poses[1:, :3, :3]
        ).as_rotvec()
        / np.diff(times)[:, None]
    )
    gt = np.array([(r["timestamp_ns"] - t0) * 1e-9 for r in gyro])
    gv = np.array([r["xyz"] for r in gyro])
    excitation = np.linalg.svd(omega - omega.mean(0), compute_uv=False)
    if (
        len(excitation) < 3
        or excitation[0] < 0.05
        or excitation[-1] / excitation[0] < 0.04
    ):
        raise CalibrationError(
            "camera/IMU rotation is insufficiently excited across three axes"
        )
    best = None
    for offset in np.linspace(-0.08, 0.08, 81):
        points = mid + offset
        valid = (points > gt[0]) & (points < gt[-1]) & (np.diff(times) < 0.25)
        if valid.sum() < 10:
            continue
        target = np.column_stack(
            [np.interp(points[valid], gt, gv[:, i]) for i in range(3)]
        )
        source = omega[valid]
        a = source - source.mean(0)
        b = target - target.mean(0)
        u, _, v = np.linalg.svd(b.T @ a)
        rot = u @ np.diag([1, 1, np.linalg.det(u @ v)]) @ v
        bias = target.mean(0) - rot @ source.mean(0)
        residual = np.linalg.norm(target - source @ rot.T - bias, axis=1)
        cost = float(np.median(residual))
        if best is None or cost < best[0]:
            best = (cost, rot, offset)
    if best is None:
        raise CalibrationError("not enough bracketed camera rotation intervals")
    T = np.eye(4)
    T[:3, :3] = best[1]
    return T, int(round(best[2] * 1e9)), excitation.tolist()


def _imu_input(
    observations, request, reference, binding, streams, run, noise_reference=None
):
    K, D, qualified = _camera_model(
        reference,
        observations["resolution"],
        binding,
        request["allow_provisional_camera"],
    )
    objects = _board_object(request["board"]).getChessboardCorners().astype(float)
    frames = []
    for row in observations["frames"]:
        if row.get("rejection_reasons"):
            continue
        pose, errors, _ = _pose(row, objects, K, D)
        rectified = cv2.undistortPoints(
            np.asarray(row["points"], float).reshape(-1, 1, 2), K, D, P=K
        ).reshape(-1, 2)
        if not np.isfinite(rectified).all():
            raise CalibrationError("nonfinite rectified corner")
        frames.append(
            {
                "frame_index": row["frame_index"],
                "timestamp_ns": row.get("exposure_midpoint_ns", row["timestamp_ns"]),
                "sensor_timestamp_ns": row["timestamp_ns"],
                "corner_timestamps_ns": row.get(
                    "corner_timestamps_ns", [row["timestamp_ns"]] * len(row["ids"])
                ),
                "ids": row["ids"],
                "pixels": rectified.tolist(),
                "raw_pixels": row["points"],
                "T_target_camera": pose.tolist(),
                "raw_pose_rms_px": _error_summary(errors)["rms"],
            }
        )
    low = max(streams[k][0]["timestamp_ns"] for k in streams) + 120_000_000
    high = min(streams[k][-1]["timestamp_ns"] for k in streams) - 120_000_000
    frames = [f for f in frames if low < f["timestamp_ns"] < high]
    if len(frames) < 30:
        raise CalibrationError(
            "IMU fit needs thirty usable target views with independent IMU brackets"
        )
    # Reserve the final quarter of target-visible motion from BOTH solver input
    # streams and camera observations. Validation below uses only this holdout.
    split = frames[0]["timestamp_ns"] + int(
        0.72 * (frames[-1]["timestamp_ns"] - frames[0]["timestamp_ns"])
    )
    train = [f for f in frames if f["timestamp_ns"] < split - 150_000_000]
    holdout = [f for f in frames if f["timestamp_ns"] > split + 150_000_000]
    if len(train) < 20 or len(holdout) < 8:
        raise CalibrationError("insufficient independently withheld target motion")
    training = {
        k: [
            r
            for r in rows
            if train[0]["timestamp_ns"] - 300_000_000
            <= r["timestamp_ns"]
            <= train[-1]["timestamp_ns"] + 130_000_000
        ]
        for k, rows in streams.items()
    }
    seed, offset, excitation = _rotation_seed(train, training["gyro"])
    camera = np.array(train[0]["T_target_camera"])
    accel = training["accel"]
    near = min(
        accel, key=lambda r: abs(r["timestamp_ns"] - train[0]["timestamp_ns"] - offset)
    )
    gravity = (camera @ np.linalg.inv(seed))[:3, :3] @ np.array(near["xyz"])
    gravity = gravity / np.linalg.norm(gravity) * 9.81
    payload = {
        "schema": "roomwalk.basalt_input.v1",
        "target_type": "charuco",
        "point_timing_model": observations.get("point_timing", {}).get(
            "model", "unmodeled_sensor_timestamp"
        ),
        "board_sha256": request["board_sha256"],
        "object_points": objects.tolist(),
        "K": K.tolist(),
        "D_native": D.tolist(),
        "resolution": observations["resolution"],
        "initial_T_imu_camera": seed.tolist(),
        "initial_cam_time_offset_ns": offset,
        "initial_gravity_target": gravity.tolist(),
        "frames": train,
        **training,
        "refine_imu_scale": True,
        "knot_spacing_ns": run.settings.knot_spacing_ns,
        "max_iterations": run.settings.max_iterations,
        "timeout_s": max(1, min(900, int(run.deadline - time.monotonic()))),
        "weights": {
            "gyro_sample_sigma_rad_s": run.settings.gyro_sample_sigma_rad_s,
            "accel_sample_sigma_m_s2": run.settings.accel_sample_sigma_m_s2,
            "provenance": "roomwalk numerical residual normalization v1; NOT measured phone noise",
        },
    }
    if noise_reference and noise_reference.get("noise_model_usable") is True:
        rates = {
            k: 1e9 / float(np.median(np.diff([r["timestamp_ns"] for r in rows])))
            for k, rows in training.items()
        }
        # The native adapter uses discrete residual weights; preserve the
        # original continuous densities and each independently measured rate.
        payload["weights"] = {
            "gyro_sample_sigma_rad_s": noise_reference["noise"][
                "gyroscope_noise_density"
            ]
            * math.sqrt(rates["gyro"]),
            "accel_sample_sigma_m_s2": noise_reference["noise"][
                "accelerometer_noise_density"
            ]
            * math.sqrt(rates["accel"]),
            "provenance": "verified bound stationary white density * sqrt(observed stream rate); drift characterization remains separate",
            "noise_reference": noise_reference,
            "observed_rates_hz": rates,
        }
    return payload, holdout, qualified, excitation


def _unsupported_distortion_row_contract(camera_rows, timing, binding):
    """Verify Camera2's unsupported-control active-array contract, not mode OFF.

    CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE defines equal active and
    pre-correction arrays when distortion control is unsupported. AOSP's
    DistortionMapper::isDistortionSupported also treats an absent mode list as
    unsupported. Restrict this path to explicitly pinned physical output; a
    logical-camera active ID or an unexplained missing result is insufficient.
    The measured encoded camera model remains separate from these sensor rows.
    """
    camera = timing.get("recorder", {}).get("camera", {})
    device = timing.get("recorder", {}).get("raw_device", {})
    signature = binding["signature"]
    actual = signature["actual_camera2"]
    physical = actual.get("active_physical_camera_id")
    focus = camera.get("focus_control", {})
    active = camera.get("sensor_active_array_size")
    if not (
        binding.get("qualified") is True
        and isinstance(physical, str) and physical
        and camera.get("id") == signature.get("camera_id") == physical
        and camera.get("sensor_geometry_camera_id") == physical
        and focus.get("physical_camera_id") == physical
        and focus.get("output_routing_policy") == "physical_camera_output_v1"
        and isinstance(focus.get("build_fingerprint"), str)
        and focus["build_fingerprint"]
        and focus["build_fingerprint"] == signature.get("focus_build_fingerprint")
        == device.get("build_fingerprint")
        and type(device.get("android_api_level")) is int
        and device["android_api_level"] >= 28
        and camera.get("distortion_correction_request_key_available") is False
        and camera.get("distortion_correction_result_key_available") is False
        and "distortion_correction_available_modes" in camera
        and camera["distortion_correction_available_modes"] in (None, [])
        and "distortion_correction_mode" in actual
        and actual["distortion_correction_mode"] is None
        and isinstance(active, list) and len(active) == 4
        and all(type(value) is int for value in active)
        and active[:2] == [0, 0] and min(active[2:]) > 0
        and active == camera.get("sensor_pre_correction_active_array_size")
        and camera_rows
    ):
        return None
    for row in camera_rows:
        if not (
            row.get("result_camera_id") == row.get("output_physical_camera_id") == physical
            and row.get("metadata_source") == "physical_output_capture_result"
            and row.get("active_physical_camera_id") == physical
            and "distortion_correction_mode" in row
            and row["distortion_correction_mode"] is None
            and all(type(row.get(k)) is int and row[k] == 0
                    for k in ("ois_mode", "eis_mode", "rotate_and_crop_mode", "sensor_pixel_mode"))
            and type(row.get("zoom_ratio")) in (int, float) and row["zoom_ratio"] == 1
            and row.get("crop_region") == actual.get("crop_region")
        ):
            return None
    return {
        "contract": "camera2_unsupported_control_same_physical_array_v1",
        "reported_mode": None,
        "assumed_off": False,
        "characteristics_camera_id": physical,
        "build_fingerprint": focus["build_fingerprint"],
        "request_key_available": False,
        "result_key_available": False,
        "available_modes": camera["distortion_correction_available_modes"],
        "verified_direct_physical_result_count": len(camera_rows),
    }


def _attach_point_timing(observations, camera_rows, timing, binding):
    """Map raw distorted encoded rows to active-array exposure midpoints.

    Camera2 skew covers the full active array, NOT the encoded 16:9 viewport.
    Restrict to a recorded physical sensor/default pixel mode, correction OFF or
    the verified unsupported-control coordinate contract, no zoom/stabilization/
    rotation, and matching active/pre-correction arrays.
    Missing physical geometry remains diagnostic-only, never guessed from K.
    """
    camera = timing.get("recorder", {}).get("camera", {})
    for frame in observations["frames"]:
        # Revalidation must not leave usable-looking point times from an older
        # successful mapping attached to newly rejected evidence.
        frame.pop("corner_timestamps_ns", None)
        frame.pop("exposure_midpoint_ns", None)
    actual = binding["signature"]["actual_camera2"]
    unsupported_contract = _unsupported_distortion_row_contract(camera_rows, timing, binding)
    active = camera.get("sensor_active_array_size")
    pre = camera.get("sensor_pre_correction_active_array_size")
    physical = actual.get("active_physical_camera_id")
    geometry_id = camera.get("sensor_geometry_camera_id", camera.get("id"))
    reasons = []
    if not physical or geometry_id != physical:
        reasons.append("physical_sensor_row_geometry_unverified")
    mapped_rows = {}
    nested_count = 0
    for row in camera_rows:
        selected = row
        if row.get("result_camera_id", camera.get("id")) != physical:
            selected = row.get("physical_capture_result")
            if (
                not isinstance(selected, Mapping)
                or selected.get("result_camera_id") != physical
                or type(selected.get("sensor_timestamp_ns")) is not int
                or selected["sensor_timestamp_ns"] != row["sensor_timestamp_ns"]
            ):
                reasons.append("physical_capture_result_row_mapping_unverified")
                continue
            # Do not trust the producer's matches-logical boolean. The physical
            # result must identify this exact exposure, with its own controls.
            # Logical output transforms are still checked below; a physical
            # OFF value cannot stand in for unreported encoded-output control.
            if (
                any(type(selected.get(k)) is not int or selected[k] != 0
                    for k in ("distortion_correction_mode", "ois_mode", "eis_mode", "rotate_and_crop_mode"))
                or type(selected.get("zoom_ratio")) not in (int, float)
                or selected["zoom_ratio"] != 1
            ):
                reasons.append("physical_row_mapping_has_unmodeled_image_transform")
            physical_crop = selected.get("crop_region")
            if (not isinstance(physical_crop, list) or len(physical_crop) != 4
                    or any(type(v) is not int for v in physical_crop)
                    or physical_crop != actual.get("crop_region")):
                reasons.append("physical_capture_result_crop_mapping_unverified")
            nested_count += 1
        mapped_rows[row["sensor_timestamp_ns"]] = selected
    if not isinstance(active, list) or len(active) != 4 or active != pre:
        reasons.append("active_array_row_geometry_unverified")
    if (
        (actual.get("distortion_correction_mode") != 0 and unsupported_contract is None)
        or actual.get("zoom_ratio") != 1
        or any(
            actual.get(k) != 0 for k in ("ois_mode", "eis_mode", "rotate_and_crop_mode")
        )
    ):
        reasons.append("row_mapping_has_unmodeled_image_transform")
    crop = actual.get("crop_region")
    if not isinstance(crop, list) or len(crop) != 4:
        reasons.append("row_mapping_crop_missing")
    if any(type(r.get("sensor_pixel_mode")) is not int or r["sensor_pixel_mode"] != 0 for r in mapped_rows.values()):
        reasons.append("sensor_pixel_mode_unverified")
    for frame in observations["frames"]:
        row = mapped_rows.get(frame["timestamp_ns"], {})
        if (
            type(row.get("exposure_time_ns")) is not int
            or not 0 < row["exposure_time_ns"] <= 100_000_000
            or type(row.get("rolling_shutter_skew_ns")) is not int
            or not 0 <= row["rolling_shutter_skew_ns"] <= 100_000_000
        ):
            reasons.append("exposure_or_skew_evidence_missing")
    if not reasons:
        if not (
            active[0] <= crop[0] < crop[2] <= active[2]
            and active[1] <= crop[1] < crop[3] <= active[3]
        ):
            reasons.append("crop_outside_active_array")
    if reasons:
        observations["point_timing"] = {
            "model": "unmodeled_sensor_timestamp",
            "qualified": False,
            "reason_codes": sorted(set(reasons)),
        }
        return
    w, h = observations["resolution"]
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    viewport_h = min(ch, cw * h / w)
    top = crop[1] + (ch - viewport_h) / 2 - active[1]
    ah = active[3] - active[1]
    for frame in observations["frames"]:
        row = mapped_rows[frame["timestamp_ns"]]
        skew, exposure = row["rolling_shutter_skew_ns"], row["exposure_time_ns"]
        points = np.asarray(frame["points"], float).reshape(-1, 2)
        fractions = (top + points[:, 1] / h * viewport_h) / ah
        frame["corner_timestamps_ns"] = [
            frame["timestamp_ns"] + int(round(exposure * 0.5 + f * skew))
            for f in fractions
        ]
        frame["exposure_midpoint_ns"] = frame["timestamp_ns"] + int(
            round(exposure * 0.5 + (top + viewport_h * 0.5) / ah * skew)
        )
    observations["point_timing"] = {
        "model": "camera2_active_array_row_exposure_midpoint",
        "qualified": True,
        "reason_codes": [],
        "active_array": active,
        "crop": crop,
        "encoded_viewport_top_active_rows": top,
        "encoded_viewport_height_active_rows": viewport_h,
        "sensor_geometry_camera_id": geometry_id,
        "physical_capture_result_row_count": nested_count,
        "capture_result_timestamp_binding": "direct_physical_sensor_timestamp_or_exact_nested_match_to_logical",
        "capture_result_source": "matching_nested_physical_or_direct_physical_result",
        "equation": "sensor_timestamp_ns + exposure_time_ns/2 + active_array_row_fraction * rolling_shutter_skew_ns",
        "source": "Camera2 active-array skew and centered per-stream aspect crop contract",
    }
    if unsupported_contract is not None:
        observations["point_timing"]["distortion_control_evidence"] = unsupported_contract


def _native_solve(payload, output, run, prefix="solver"):
    executable = run.settings.solver_path or run.settings.executable
    if executable is None:
        executable = CalibrationSettings.from_env().executable
    if executable is None or not Path(executable).is_file():
        raise CalibrationError(
            "configure NOESIS_PHONE_SCAN_CALIBRATION_SOLVER to the built pinned ChArUco Basalt adapter"
        )
    executable = Path(executable).resolve()
    version = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.strip()
    if version != f"roomwalk.basalt_adapter.v2 {BASALT_COMMIT}":
        raise CalibrationError("wrong calibration executable/version")
    _write(output / f"{prefix}_input.json", payload)
    run.update(0.6, "Solving fixed-camera ChArUco/IMU calibration with Basalt")
    with (
        (output / f"{prefix}.stdout.log").open("x") as stdout,
        (output / f"{prefix}.stderr.log").open("x") as stderr,
    ):
        process = subprocess.Popen(
            [
                str(executable),
                str(output / f"{prefix}_input.json"),
                str(output / f"{prefix}_result.json"),
            ],
            stdout=stdout,
            stderr=stderr,
        )
        try:
            while process.poll() is None:
                run.check()
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise CalibrationError(
                    "Basalt solve failed; see retained solver.stderr.log"
                )
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
    result = _read_json(output / f"{prefix}_result.json", 32 * 1024 * 1024)
    if (
        result.get("schema") != "roomwalk.basalt_result.v1"
        or result.get("basalt_commit") != BASALT_COMMIT
        or result.get("accepted_for_metric_vio") is not False
        or result.get("point_specific_spline_evaluation") is not True
        or result.get("point_timing_model") != payload["point_timing_model"]
    ):
        raise CalibrationError("native result contract mismatch")
    T = np.asarray(result["T_imu_camera"], float)
    if (
        T.shape != (4, 4)
        or not np.isfinite(T).all()
        or not np.allclose(T[3], [0, 0, 0, 1])
        or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
        or np.linalg.det(T[:3, :3]) < 0.99
        or np.linalg.norm(T[:3, 3]) > 0.5
    ):
        raise CalibrationError("native extrinsic result is invalid")
    offset = result["cam_time_offset_ns"]
    if (
        type(offset) is not int
        or abs(offset) > 100_000_000
        or result["imu_to_camera_offset_ns"] != -offset
    ):
        raise CalibrationError("native time offset result is invalid")
    for kind in ("accelerometer", "gyroscope"):
        M = np.asarray(result["imu_corrections"][kind + "_matrix"], float)
        b = np.asarray(result["imu_corrections"][kind + "_bias"], float)
        if (
            M.shape != (3, 3)
            or b.shape != (3,)
            or not np.isfinite(M).all()
            or not np.isfinite(b).all()
            or np.linalg.det(M) <= 0
            or np.linalg.cond(M) > 100
        ):
            raise CalibrationError("native IMU correction matrix is invalid")
    result["executable"] = {
        "name": executable.name,
        "sha256": _sha(executable),
        "version": version,
    }
    if "D_native" in payload:
        K = np.array(payload["K"])
        D = np.array(payload["D_native"])
        predicted = result.get("corner_predictions", [])
        if len(predicted) != len(payload["frames"]):
            raise CalibrationError("native corner prediction count mismatch")
        errors = []
        for source, prediction in zip(payload["frames"], predicted):
            px = np.array(prediction["pixels"], float)
            if prediction["frame_index"] != source["frame_index"] or px.shape != (
                len(source["ids"]),
                2,
            ):
                raise CalibrationError("native corner prediction identity mismatch")
            rays = np.column_stack(
                [
                    (px[:, 0] - K[0, 2]) / K[0, 0],
                    (px[:, 1] - K[1, 2]) / K[1, 1],
                    np.ones(len(px)),
                ]
            )
            raw = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K, D)[0].reshape(
                -1, 2
            )
            errors.extend(np.linalg.norm(raw - np.array(source["raw_pixels"]), axis=1))
        result["reprojection_native_px"] = _error_summary(errors)
    return result


def _visual_holdout(payload, frames, output, run):
    if len(frames) < 12:
        return None
    data = {
        k: payload[k]
        for k in (
            "schema",
            "target_type",
            "board_sha256",
            "object_points",
            "K",
            "D_native",
            "resolution",
            "weights",
            "point_timing_model",
        )
    }
    data.update(
        visual_only=True,
        frames=frames,
        initial_T_imu_camera=np.eye(4).tolist(),
        initial_cam_time_offset_ns=0,
        initial_gravity_target=[0, 9.81, 0],
        refine_imu_scale=False,
        knot_spacing_ns=100_000_000,
        max_iterations=run.settings.max_iterations,
        timeout_s=max(1, min(900, int(run.deadline - time.monotonic()))),
    )
    return _native_solve(data, output, run, prefix="heldout_visual")


def _translation_holdout(frames, streams, result, visual):
    """Independent visual displacement vs twice-integrated held-out acceleration.

    Symmetric triangular integration eliminates unknown initial velocity. The
    lever-arm term is (R0 - 2*Rmid + Rend)*t_imu_camera; no differentiated noisy
    PnP positions and no training trajectory are used. Diagnostic least-squares
    translation is NEVER substituted into the calibration returned to consumers.
    """
    unavailable = {
        "qualified": False,
        "reason_codes": ["insufficient_independent_visual_motion"],
    }
    if visual is None or visual.get("status") != "converged":
        return unavailable
    if (
        visual["reprojection_native_px"]["rms"] > 2
        or visual["reprojection_native_px"]["p95"] > 4
    ):
        return {
            "qualified": False,
            "reason_codes": ["independent_visual_reprojection_exceeds_policy"],
        }
    trajectory = visual["visual_trajectory"]
    epoch = trajectory[0]["timestamp_ns"]
    times = np.array([(r["timestamp_ns"] - epoch) * 1e-9 for r in trajectory])
    poses = np.array([r["T_target_camera"] for r in trajectory])
    if times[-1] < 2 or np.max(np.diff(times)) > 0.02:
        return unavailable
    position = CubicSpline(times, poses[:, :3, 3])
    rotation = RotationSpline(times, Rotation.from_matrix(poses[:, :3, :3]))
    transform = np.array(result["T_imu_camera"])
    correction = result["imu_corrections"]
    at = np.array([(r["timestamp_ns"] - epoch) * 1e-9 for r in streams["accel"]])
    av = np.array([r["xyz"] for r in streams["accel"]]) @ np.array(
        correction["accelerometer_matrix"]
    ).T - np.array(correction["accelerometer_bias"])
    offset = result["cam_time_offset_ns"] * 1e-9
    gravity = np.array(result["gravity_target"])
    design, measured = [], []
    half = 0.4
    for center in np.arange(half + 0.2, times[-1] - half - 0.2, 0.1):
        ts = np.linspace(center - half, center + half, 161)
        if ts[0] + offset < at[0] or ts[-1] + offset > at[-1]:
            continue
        ri = rotation(ts).as_matrix() @ transform[:3, :3].T
        accel = np.column_stack(
            [np.interp(ts + offset, at, av[:, a]) for a in range(3)]
        )
        world = np.einsum("nij,nj->ni", ri, accel) - gravity
        triangular = half - np.abs(ts - center)
        integral = trapezoid(world * triangular[:, None], ts, axis=0)
        endpoints = np.array([center - half, center, center + half])
        rr = rotation(endpoints).as_matrix() @ transform[:3, :3].T
        pp = position(endpoints)
        design.append(rr[0] - 2 * rr[1] + rr[2])
        measured.append(pp[0] - 2 * pp[1] + pp[2] - integral)
    if len(design) < 8:
        return unavailable
    A, y = np.concatenate(design), np.concatenate(measured)
    sv = np.linalg.svd(A / math.sqrt(len(design)), compute_uv=False)
    fitted, _, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    residual = (y - A @ transform[:3, 3]).reshape(-1, 3)
    errors = _error_summary(np.linalg.norm(residual, axis=1))
    disagreement = float(np.linalg.norm(fitted - transform[:3, 3]))
    sensitivity = float(errors["rms"] / max(float(sv[-1]), 1e-12))
    reasons = []
    if rank != 3 or sv[-1] < 0.003 or sv[0] / max(sv[-1], 1e-12) > 50:
        reasons.append("heldout_lever_arm_unobservable")
    if errors["rms"] > 0.005 or errors["p95"] > 0.01:
        reasons.append("heldout_integrated_acceleration_residual")
    if disagreement > 0.015:
        reasons.append("heldout_translation_disagrees_over_15_mm")
    if sensitivity > 0.01:
        reasons.append("heldout_translation_sensitivity_over_10_mm")
    return {
        "qualified": not reasons,
        "reason_codes": reasons,
        "method": "visual_only_row_timed_spline_vs_triangular_integral_acceleration",
        "visual_uses_imu": False,
        "calibration_refitted": False,
        "window_half_width_s": half,
        "window_count": len(design),
        "displacement_error_m": errors,
        "independent_translation_m": fitted.tolist(),
        "translation_disagreement_m": disagreement,
        "design_singular_values": sv.tolist(),
        "residual_sensitivity_bound_m": sensitivity,
        "sensitivity_is_statistical_covariance": False,
        "policy": {
            "id": "roomwalk.translation_holdout.v1",
            "maximum_rms_m": 0.005,
            "maximum_p95_m": 0.01,
            "maximum_disagreement_m": 0.015,
            "maximum_sensitivity_m": 0.01,
            "minimum_design_singular_value": 0.003,
            "maximum_condition": 50,
        },
    }


def _heldout_imu(frames, streams, result, visual=None):
    # The visual-only holdout uses no IMU observations or training trajectory.
    # Rotation, lever-arm and clock checks share this independent visual motion.
    rotation_frames = (
        visual["visual_trajectory"]
        if visual is not None and visual.get("status") == "converged"
        else frames
    )
    poses = np.array([r["T_target_camera"] for r in rotation_frames])
    epoch = rotation_frames[0]["timestamp_ns"]
    t = np.array([(r["timestamp_ns"] - epoch) * 1e-9 for r in rotation_frames])
    dt = np.diff(t)
    omega = (
        Rotation.from_matrix(
            np.swapaxes(poses[:-1, :3, :3], 1, 2) @ poses[1:, :3, :3]
        ).as_rotvec()
        / dt[:, None]
    )
    R = np.array(result["T_imu_camera"])[:3, :3]
    gt = np.array([(r["timestamp_ns"] - epoch) * 1e-9 for r in streams["gyro"]])
    raw = np.array([r["xyz"] for r in streams["gyro"]])
    correction = result["imu_corrections"]
    gyro = raw @ np.array(correction["gyroscope_matrix"]).T - np.array(
        correction["gyroscope_bias"]
    )
    offset = result["cam_time_offset_ns"] * 1e-9
    errors = []
    for i in range(len(dt)):
        if dt[i] > 0.25:
            continue
        times = np.linspace(t[i] + offset, t[i + 1] + offset, 11)
        if times[0] < gt[0] or times[-1] > gt[-1]:
            continue
        mean = np.mean(
            np.column_stack([np.interp(times, gt, gyro[:, a]) for a in range(3)]),
            axis=0,
        )
        errors.append(float(np.linalg.norm(mean - R @ omega[i])))
    translation = _translation_holdout(frames, streams, result, visual)
    # Independently re-estimate only an offset diagnostic over held-out motion;
    # the solver's returned offset is never overwritten.
    mids = (t[:-1] + t[1:]) / 2
    candidates = np.arange(-10_000_000, 10_000_001, 250_000)
    target = omega @ R.T
    objectives = []
    for delta in candidates:
        shifted = mids + offset + delta * 1e-9
        observed = np.column_stack(
            [np.interp(shifted, gt, gyro[:, a]) for a in range(3)]
        )
        objectives.append(float(np.mean((observed - target) ** 2)))
    best = int(candidates[int(np.argmin(objectives))])
    excitation = float(
        np.sqrt(
            np.mean(np.sum(np.diff(omega, axis=0) ** 2, axis=1) / np.diff(mids) ** 2)
        )
    )
    gyro_error = _error_summary(errors)
    time_reasons = []
    if visual is None or visual.get("status") != "converged":
        time_reasons.append("independent_row_timed_visual_fit_missing")
    elif (
        visual["reprojection_native_px"]["rms"] > 2
        or visual["reprojection_native_px"]["p95"] > 4
    ):
        time_reasons.append("independent_visual_reprojection_exceeds_policy")
    if excitation < 0.05:
        time_reasons.append("heldout_time_offset_unobservable")
    if abs(best) > 1_000_000:
        time_reasons.append("heldout_time_offset_disagrees_over_1_ms")
    if gyro_error["rms"] > 0.08:
        time_reasons.append("heldout_gyro_residual_exceeds_0_08_rad_s")
    return {
        "schema": "roomwalk.imu_holdout.v1",
        "frame_indices": [r["frame_index"] for r in frames],
        "used_by_solver": False,
        "gyro_vector_error_rad_s": gyro_error,
        "method": "independent_visual_rotation_increments_vs_corrected_interval_mean_gyro",
        "translation_validation": translation,
        "time_offset_validation": {
            "qualified": not time_reasons,
            "reason_codes": time_reasons,
            "independent_offset_difference_ns": best,
            "angular_acceleration_rms_rad_s2": excitation,
            "search_half_width_ns": 10_000_000,
            "search_step_ns": 250_000,
            "calibration_refitted": False,
        },
        "visual_reprojection_native_px": visual.get("reprojection_native_px")
        if visual
        else None,
        "noise_validation": "not_established",
        "rolling_shutter_model": result.get("point_timing_model"),
    }


def run_calibration(
    capture_dir: Path,
    request: dict,
    output_dir: Path,
    *,
    settings=None,
    progress=None,
    cancelled=None,
) -> dict:
    """Run real CPU calibration; preserve failures and all raw input evidence.

    ``progress`` receives (fraction, message); ``cancelled`` returns bool.
    Output is always review-only. The parent must not promote these flags to
    capture.py's metric admission without its separate measured-evidence gate.
    """
    request = validate_calibration_request(request)
    settings = _settings(settings)
    root = Path(capture_dir).resolve()
    output = Path(output_dir).resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise CalibrationError(
            "calibration output must be separate from its immutable capture"
        )
    if output.exists():
        raise CalibrationError("calibration output directory already exists")
    output.mkdir(parents=True, exist_ok=False)
    run = _Run(settings, progress, cancelled)
    report = {
        "schema": REPORT_SCHEMA,
        "mode": request["mode"],
        "status": "failed",
        "review_only": True,
        "accepted_for_metric_vio": False,
        "imu_noise_calibrated": False,
        "noise_model_usable": False,
        "noise_model_status": "not_supplied",
        "camera_imu_extrinsics_calibrated": False,
        "time_offset_calibrated": False,
        "reason_codes": [],
        "board_sha256": request["board_sha256"],
    }
    _write(output / "request.json", request)
    provenance = {}
    try:
        run.check()
        # Missing choices and invalid references must not consume native-video
        # detection first. The child repeats these checks even when its parent
        # server predates the submission preflight.
        reference = _load_camera_reference(settings.camera_calibration_dir)
        if request["mode"] == "imu" and not request["allow_provisional_camera"]:
            if reference is None:
                raise CalibrationError("IMU mode needs a server-resolved qualified camera calibration")
            if not request["board_geometry_confirmed"]:
                raise CalibrationError("Confirm the measured printed board dimensions before processing metric motion")
        manifest, video, size, times, rows, timing, provenance = _capture(root, run)
        report["capture_id"] = manifest["capture_id"]
        if request.get("capture_id", manifest["capture_id"]) != manifest["capture_id"]:
            raise CalibrationError("request capture_id differs from imported capture")
        binding = _capture_binding(manifest, rows, size, timing)
        if request["mode"] == "imu" and reference is not None:
            _camera_model(reference, size, binding, request["allow_provisional_camera"])
        _write(
            output / "source_manifest.json",
            {
                "schema": "roomwalk.calibration_sources.v1",
                "capture_id": manifest["capture_id"],
                "files": provenance,
                "binding": binding,
            },
        )
        hz = (
            settings.camera_sample_hz
            if request["mode"] == "camera"
            else settings.imu_sample_hz
        )
        maximum = (
            settings.max_camera_frames
            if request["mode"] == "camera"
            else settings.max_imu_frames
        )
        observations = _detect_capture(
            video,
            size,
            times,
            _sample_indices(times, hz, maximum),
            request["board"],
            run,
            output,
        )
        _attach_point_timing(observations, rows, timing, binding)
        _write(output / "observations.json", observations)
        if request["mode"] == "camera" or reference is None:
            if request["mode"] == "imu" and not request["allow_provisional_camera"]:
                raise CalibrationError(
                    "IMU mode needs a server-resolved qualified camera calibration"
                )
            run.update(0.52, "Fitting camera with blocked independent target holdouts")
            # IMU sampling is dense; bound the separate provisional intrinsic fit.
            selection = observations
            if len(observations["frames"]) > settings.max_camera_frames:
                indices = (
                    np.linspace(
                        0, len(observations["frames"]) - 1, settings.max_camera_frames
                    )
                    .round()
                    .astype(int)
                )
                selection = {
                    **observations,
                    "frames": [observations["frames"][i] for i in indices],
                }
            reference = fit_camera_observations(
                selection, request, binding, check=run.check
            )
            reference["source_manifest_sha256"] = _sha(output / "source_manifest.json")
            reference["observations_sha256"] = _sha(output / "observations.json")
            reference["capture_id"] = manifest["capture_id"]
            _write(output / "camera_result.json", reference)
            if reference["quality"]["status"] == "qualified":
                _write(output / "camera_profile.json", camera_profile(reference))
        report["camera_intrinsics_calibrated"] = (
            reference["quality"]["status"] == "qualified"
        )
        report["camera_result"] = (
            reference
            if request["mode"] == "camera"
            else {
                "schema": reference["schema"],
                "sha256": _digest(reference),
                "quality": reference["quality"]["status"],
            }
        )
        if request["mode"] == "imu":
            streams = _imu_arrays(root, manifest, run, timing)
            noise_reference = _load_noise_reference(
                settings.noise_calibration_dir, manifest, timing
            )
            report["imu_noise_calibrated"] = noise_reference["imu_noise_calibrated"]
            report["noise_model_usable"] = noise_reference["noise_model_usable"]
            report["noise_model_status"] = noise_reference["noise_model_status"]
            report["noise_reference"] = noise_reference
            payload, heldout, qualified, excitation = _imu_input(
                observations, request, reference, binding, streams, run, noise_reference
            )
            result = _native_solve(payload, output, run)
            run.update(
                0.86, "Validating independently withheld visual and inertial motion"
            )
            visual = _visual_holdout(payload, heldout, output, run)
            validation = _heldout_imu(heldout, streams, result, visual)
            result.update(
                {
                    "schema": "roomwalk.camera_imu_calibration.v1",
                    "capture_id": manifest["capture_id"],
                    "binding": binding,
                    "camera_reference_sha256": _digest(reference),
                    "board_sha256": request["board_sha256"],
                    "camera_model_qualified": qualified,
                    "imu_noise_calibrated": noise_reference["imu_noise_calibrated"],
                    "noise_model_usable": noise_reference["noise_model_usable"],
                    "noise_model_status": noise_reference["noise_model_status"],
                    "noise_reference": noise_reference,
                    "rotation_excitation_singular_values": excitation,
                    "heldout": validation,
                    "point_timing": observations["point_timing"],
                    "physical_board_scale_verified": request[
                        "board_geometry_confirmed"
                    ],
                    "physical_board_scale_provenance": "user_attestation"
                    if request["board_geometry_confirmed"]
                    else "unverified",
                    "quality": {"status": "insufficient_evidence", "reason_codes": []},
                    "runtime_admission": {
                        "accepted": False,
                        "reason_codes": [
                            "imu_correction_consumer_not_admitted",
                            "camera_row_time_consumer_not_admitted",
                        ],
                    },
                    "camera_time_reference": "point_exposure_midpoint_in_camera2_sensor_clock",
                }
            )
            common = []
            if not qualified:
                common.append("provisional_camera")
            if not observations["point_timing"]["qualified"]:
                common.extend(observations["point_timing"]["reason_codes"])
            if result["status"] != "converged":
                common.append("solver_iteration_limit")
            if validation["gyro_vector_error_rad_s"]["rms"] > 0.08:
                common.append("heldout_gyro_residual_exceeds_0_08_rad_s")
            extrinsic_reasons = (
                common + validation["translation_validation"]["reason_codes"]
            )
            if not request["board_geometry_confirmed"]:
                extrinsic_reasons.append("physical_board_scale_not_confirmed")
            time_reasons = common + validation["time_offset_validation"]["reason_codes"]
            result["camera_imu_extrinsics_calibrated"] = not extrinsic_reasons
            result["time_offset_calibrated"] = not time_reasons
            reasons = sorted(set(extrinsic_reasons + time_reasons))
            # Practical camera/IMU calibration and noise characterization are
            # separate capabilities. Missing long-term Allan terms must not
            # undo independent extrinsic/timing evidence. The parent still owns
            # candidate-profile validation and all runtime consumer admission.
            result["noise_characterization"] = {
                "status": noise_reference["noise_model_status"],
                "noise_model_usable": noise_reference["noise_model_usable"],
                "all_four_terms_measured": noise_reference["imu_noise_calibrated"],
                "reason_codes": noise_reference["reason_codes"],
            }
            if not noise_reference["noise_model_usable"]:
                result["runtime_admission"]["reason_codes"].append(
                    "imu_noise_model_unusable"
                )
            elif not noise_reference["imu_noise_calibrated"]:
                result["runtime_admission"]["reason_codes"].append(
                    "short_noise_profile_validation_not_admitted"
                )
            result["quality"] = {
                "status": "qualified" if not reasons else "insufficient_evidence",
                "reason_codes": reasons,
                "extrinsics_reason_codes": sorted(set(extrinsic_reasons)),
                "time_offset_reason_codes": sorted(set(time_reasons)),
            }
            report["camera_imu_extrinsics_calibrated"] = result[
                "camera_imu_extrinsics_calibrated"
            ]
            report["time_offset_calibrated"] = result["time_offset_calibrated"]
            _write(output / "camera_imu_result.json", result)
            _write(output / "validation.json", validation)
            report["quality"] = result["quality"]
            report["calibration_estimate"] = {
                k: result[k]
                for k in (
                    "T_imu_camera",
                    "cam_time_offset_ns",
                    "imu_to_camera_offset_ns",
                    "imu_corrections",
                )
            }
            report["solver_status"] = result["status"]
        else:
            report["quality"] = reference["quality"]
            _write(output / "validation.json", reference["quality"])
        run.check()
        for name, info in provenance.items():
            if _sha(root / name) != info["sha256"]:
                raise CalibrationError(
                    "source changed while calibration was processing"
                )
        report["status"] = "completed"
        report["reason_codes"] = report["quality"]["reason_codes"]
        run.update(1.0, "Calibration computed; review-only results retained")
    except CalibrationCancelled as exc:
        report.update(status="cancelled", reason_codes=["cancelled"], error=str(exc))
    except (
        CalibrationError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
        cv2.error,
        subprocess.SubprocessError,
    ) as exc:
        report.update(
            status="failed",
            reason_codes=["calibration_processing_failed"],
            error=str(exc),
        )
    if report["status"] != "completed":
        report.update(
            camera_intrinsics_calibrated=False,
            camera_imu_extrinsics_calibrated=False,
            time_offset_calibrated=False,
            quality={
                "status": "insufficient_evidence",
                "reason_codes": report["reason_codes"],
            },
        )
    report["artifacts"] = {
        p.name: {"path": p.name, "sha256": _sha(p), "bytes": p.stat().st_size}
        for p in sorted(output.iterdir())
        if p.is_file()
    }
    _write(output / "report.json", report)
    return report
