"""Fixed-estimator VIO boundary for synchronized phone captures.

OpenVINS is selected as the initial conventional estimator.  The adapter
deliberately does not contain a fallback integrator: if the native estimator
is unavailable, the operation fails with an actionable reason and the RGB
reconstruction remains usable.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from .phone_calibration import PhoneCalibrationError, rectification_maps


VIO_SCHEMA = "noesis.phone_capture.vio_result.v1"
VIO_CALIBRATION_PRIOR_SCHEMA = "noesis.phone_capture.vio_calibration_prior.v1"
VIO_CALIBRATION_SCHEMA = "noesis.phone_capture.vio_calibration_result.v1"
VIO_PROFILE_VALIDATION_SCHEMA = "noesis.phone_capture.vio_profile_validation_result.v1"
VIO_PROFILE_INPUT_SCHEMA = "noesis.phone_capture.openvins_profile_validation_input.v1"
SHORT_CONSUMER_VERSION = "roomwalk.openvins.short_walk.v1"
POSE_TIME_REFERENCE = "camera2_encoded_viewport_centre_exposure_midpoint"
SHORT_IMAGE_MODEL = "centre_timed_global_shutter_approximation"
SHORT_RUNTIME_QUALITY_POLICY = {"version": "roomwalk.short_walk_runtime_quality.v1",
                                "minimum_camera_coverage": 0.8, "maximum_pose_gap_s": 0.2,
                                "maximum_end_gap_s": 0.2, "maximum_resets": 0}
# Computational experiment limits. These never establish calibration admission.
CALIBRATION_BOUNDS = {
    "rotation_change_rad": 0.50,
    "translation_change_m": 0.20,
    "time_offset_change_s": 0.050,
    "translation_norm_m": 0.50,
    "absolute_time_offset_s": 0.100,
}
CALIBRATION_UNCERTAINTY = {
    "rotation_std_rad": (0.10, 0.50),
    "translation_std_m": (0.05, 0.20),
    "time_offset_std_s": (0.020, 0.050),
}
CALIBRATION_COVARIANCE_CONVENTION = "openvins_jpl_left_rotation_I_to_C_position_I_in_C_camera_to_imu_offset"
ProgressCallback = Callable[[float, str], None]
RuntimeQualityCheck = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


class VIOError(RuntimeError):
    """Raised when fixed-estimator VIO cannot run or its output is invalid."""


@dataclass(frozen=True)
class VIOSettings:
    estimator: str = "openvins"
    executable: Path | None = None
    config: Path | None = None
    timeout_s: int = 900
    max_input_frames: int = 12_000

    @classmethod
    def from_env(cls) -> "VIOSettings":
        executable = str(os.environ.get("NOESIS_PHONE_SCAN_VIO_EXECUTABLE") or "").strip()
        config = str(os.environ.get("NOESIS_PHONE_SCAN_VIO_CONFIG") or "").strip()
        return cls(
            estimator=os.environ.get("NOESIS_PHONE_SCAN_VIO_ESTIMATOR", "openvins").strip(),
            executable=Path(executable).expanduser() if executable else None,
            config=Path(config).expanduser() if config else None,
            timeout_s=max(30, int(os.environ.get("NOESIS_PHONE_SCAN_VIO_TIMEOUT_S", "900"))),
            max_input_frames=max(2, int(os.environ.get("NOESIS_PHONE_SCAN_VIO_MAX_FRAMES", "12000"))),
        )


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise VIOError(f"{field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VIOError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise VIOError(f"{field} must be finite")
    return number


def _matrix(raw: Any, field: str) -> list[list[float]]:
    if not isinstance(raw, list) or len(raw) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in raw):
        raise VIOError(f"{field} must be a 4x4 matrix")
    matrix = [[_finite(value, field) for value in row] for row in raw]
    if matrix[3] != [0.0, 0.0, 0.0, 1.0]:
        raise VIOError(f"{field} must be homogeneous")
    rotation = np.asarray([row[:3] for row in matrix[:3]], dtype=np.float64)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3) or float(np.linalg.det(rotation)) <= 0.0:
        raise VIOError(f"{field} rotation is not proper")
    return matrix


def validate_vio_input(capture_report: Mapping[str, Any], prepared: Mapping[str, Any]) -> None:
    """Require the calibrated, timestamped input needed for metric VIO."""

    if capture_report.get("schema") != "noesis.phone_capture.v1":
        raise VIOError("capture report is not a supported phone capture")
    if capture_report.get("metric_vio_allowed") is not True:
        raise VIOError("metric VIO is blocked because capture calibration or timing is incomplete")
    if "short_session" in capture_report:
        _checked_short_session(capture_report, validation=False)
    imu = capture_report.get("imu")
    video = capture_report.get("video")
    coverage = capture_report.get("coverage")
    if not isinstance(imu, Mapping) or int(imu.get("sample_count") or 0) < 2:
        raise VIOError("metric VIO requires timestamped IMU samples")
    if not isinstance(video, Mapping) or int(video.get("frame_count") or 0) < 2:
        raise VIOError("metric VIO requires timestamped video frames")
    if not isinstance(coverage, Mapping) or not coverage.get("starts_before_video") or not coverage.get("ends_after_video"):
        raise VIOError("IMU samples must cover the complete camera interval")
    frames = prepared.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise VIOError("metric VIO requires at least two prepared views")
    if any(not isinstance(row, Mapping) or not isinstance(row.get("capture_time_ns"), int) for row in frames):
        raise VIOError("prepared views are missing exact source capture timestamps")


def _yaml_matrix(matrix: list[list[float]]) -> str:
    return "[" + ", ".join("[" + ", ".join(f"{float(value):.17g}" for value in row) + "]" for row in matrix) + "]"


def _yaml_vector(values: list[float]) -> str:
    return "[" + ", ".join(f"{float(value):.17g}" for value in values) + "]"


def _replace_yaml_key(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s*:\s*).*$")
    replacement = rf"\g<1>{value}"
    text, count = pattern.subn(replacement, text, count=1)
    return text if count else text.rstrip() + f"\n{key}: {value}\n"


def _interpolate_stream(
    times: list[int],
    values: list[list[float]],
    timestamp_ns: int,
) -> list[float]:
    """Linearly sample one asynchronous stream using precomputed arrays.

    Separate phone sensors can contain hundreds of thousands of rows.  Keep
    the timestamp/value arrays outside this function so each bisect is O(log
    N), rather than rebuilding the timestamp list for every camera-aligned
    sample.
    """
    index = bisect.bisect_left(times, timestamp_ns)
    if index < len(times) and times[index] == timestamp_ns:
        return [float(value) for value in values[index]]
    if index <= 0 or index >= len(times):
        raise VIOError("separate IMU stream cannot be interpolated at the estimator timestamp")
    before_time, after_time = times[index - 1], times[index]
    fraction = (timestamp_ns - before_time) / (after_time - before_time)
    return [
        float(left) + fraction * (float(right) - float(left))
        for left, right in zip(values[index - 1], values[index], strict=True)
    ]


def _openvins_camera_projection(
    camera: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray] | None]:
    """Preserve a native model or rectify the complete OpenCV D5 model at fixed K."""
    try:
        matrix = np.asarray(camera.get("intrinsics"), dtype=np.float64)
        resolution = camera.get("resolution_px")
        if not isinstance(resolution, list) or len(resolution) != 2:
            raise ValueError("missing resolution")
        if any(type(value) is not int or not 1 <= value <= 16384 for value in resolution):
            raise ValueError("invalid resolution")
        width, height = resolution
        raw_distortion = camera.get("distortion")
        if not isinstance(raw_distortion, list):
            raise ValueError("missing distortion")
        distortion = [_finite(value, "camera.distortion") for value in raw_distortion]
    except (TypeError, ValueError) as exc:
        raise VIOError("OpenVINS camera calibration is invalid") from exc
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or not np.array_equal(matrix[2], [0.0, 0.0, 1.0])
        or matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
        or matrix[0, 1] != 0.0
        or matrix[1, 0] != 0.0
        or not 0.0 <= matrix[0, 2] < width
        or not 0.0 <= matrix[1, 2] < height
    ):
        raise VIOError("OpenVINS requires an exact finite zero-skew calibrated camera matrix")
    model = str(camera.get("distortion_model") or "").lower()
    pinhole = model in {"plumb_bob", "radtan", "brown_conrady"}
    fisheye = model in {"equidistant", "fisheye"}
    if not (pinhole or fisheye) or len(distortion) > (5 if pinhole else 4):
        raise VIOError("OpenVINS cannot preserve this calibrated distortion model")
    maps = None
    rectified = pinhole and len(distortion) == 5
    if rectified:
        # Use the same full-D5, same-K, no-crop implementation as prepared
        # calibrated RGB.  Dense VIO images retain their original resolution.
        try:
            map_x, map_y, output_matrix = rectification_maps(
                {
                    "model": "opencv_pinhole", "resolution": resolution,
                    "K": matrix.tolist(), "D": distortion,
                    "runtime": {"use_undistorted_frames": False},
                },
                (width, height),
            )
        except (PhoneCalibrationError, cv2.error) as exc:
            raise VIOError("OpenVINS full-D5 rectification could not be constructed") from exc
        if not np.array_equal(matrix, output_matrix) or not np.isfinite(map_x).all() or not np.isfinite(map_y).all():
            raise VIOError("OpenVINS rectification changed K or produced nonfinite image maps")
        maps = map_x, map_y
    estimator_distortion = [0.0] * 4 if rectified else distortion + [0.0] * (4 - len(distortion))
    return {
        "schema": "noesis.phone_capture.openvins_camera_preprocessing.v1",
        "operation": "opencv_pinhole_full_d5_undistort_same_K_no_crop" if rectified else "native_calibrated_image",
        "rectified": rectified,
        "source_camera": {
            "K": matrix.tolist(), "resolution_px": list(resolution),
            "distortion_model": model, "distortion": distortion,
        },
        "estimator_camera": {
            "K": matrix.tolist(), "resolution_px": list(resolution),
            "camera_model": "pinhole", "distortion_model": "equidistant" if fisheye else "radtan",
            "distortion": estimator_distortion,
        },
        "camera_axes_unchanged": True,
    }, maps


def _rectify_dense_camera_images(
    image_paths: list[Path],
    image_dir: Path,
    projection: dict[str, Any],
    maps: tuple[np.ndarray, np.ndarray],
    *,
    timeout_s: int = 900,
) -> list[dict[str, str]]:
    """Rewrite only newly decoded images and bind the full lens model and border mask."""
    width, height = projection["estimator_camera"]["resolution_px"]
    map_x, map_y = maps
    if map_x.shape != (height, width) or map_y.shape != (height, width):
        raise VIOError("OpenVINS rectification map dimensions do not match calibrated images")
    valid = (
        np.isfinite(map_x) & np.isfinite(map_y)
        & (map_x >= 0.0) & (map_x <= width - 1)
        & (map_y >= 0.0) & (map_y <= height - 1)
    )
    if not np.any(valid):
        raise VIOError("OpenVINS rectification has no valid source-image rays")
    invalid_mask = np.where(valid, 0, 255).astype(np.uint8)
    mask_path = image_dir.parent / "rectification_mask.png"
    ok, encoded_mask = cv2.imencode(".png", invalid_mask)
    if not ok:
        raise VIOError("OpenVINS rectification mask could not be encoded")
    mask_bytes = encoded_mask.tobytes()
    mask_path.write_bytes(mask_bytes)
    projection.update({
        "opencv_version": cv2.__version__,
        "interpolation": "INTER_LINEAR",
        "border_policy": "BORDER_CONSTANT_zero_excluded_by_camera_mask",
        "invalid_ray_mask": {
            "path": "cam0/rectification_mask.png",
            "sha256": hashlib.sha256(mask_bytes).hexdigest(),
            "valid_value": 0, "invalid_value": 255,
            "invalid_pixel_count": int(np.count_nonzero(invalid_mask)),
            "pixel_count": width * height,
        },
    })
    rows: list[dict[str, str]] = []
    deadline = monotonic() + timeout_s
    for path in image_paths:
        if monotonic() > deadline:
            raise VIOError("dense camera rectification exceeded its time limit")
        source_bytes = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[:2] != (height, width):
            raise VIOError(f"decoded image does not match calibrated rectification dimensions: {path.name}")
        rectified = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # Keep invalid pixels explicit even where interpolation rounding might
        # otherwise sample the source border.  The estimator excludes them.
        rectified[~valid] = 0
        ok, encoded = cv2.imencode(".png", rectified)
        if not ok:
            raise VIOError(f"rectified OpenVINS image could not be encoded: {path.name}")
        output_bytes = encoded.tobytes()
        path.write_bytes(output_bytes)
        rows.append({
            "decoded_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "estimator_image_sha256": hashlib.sha256(output_bytes).hexdigest(),
        })
    return rows


def materialize_openvins_input(
    capture_dir: Path,
    capture_report: Mapping[str, Any],
    output_dir: Path,
    settings: VIOSettings,
    progress: ProgressCallback,
) -> tuple[Path, Path]:
    """Materialize a bounded dense ASL layout using only imported calibration."""

    return _materialize_openvins_input(capture_dir, capture_report, output_dir, settings, progress)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_file(root: Path, name: str) -> Path:
    relative = Path(name)
    path = root / relative
    if (not name or relative.is_absolute() or ".." in relative.parts or "\\" in name
            or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve())):
        raise VIOError("short-profile input has an invalid source path")
    return path


def _checked_short_session(report, *, validation):
    from .motion_preprocessing import checked_corrections, MotionProfileError

    short = report.get("short_session")
    if (report.get("schema") != "noesis.phone_capture.v1" or not isinstance(short, Mapping)
            or short.get("consumer_version") != SHORT_CONSUMER_VERSION
            or short.get("validation_run") is not validation
            or report.get("metric_vio_allowed") is not (not validation)):
        raise VIOError("short-profile capture admission/mode or consumer version is invalid")
    if (not isinstance(short.get("profile_id"), str) or not 1 <= len(short["profile_id"]) <= 128
            or short.get("analysis_max_side") != 1280
            or short.get("image_motion_model") != SHORT_IMAGE_MODEL
            or short.get("rolling_shutter_compensated") is not False
            or short.get("imu_correction_equation") != "corrected = matrix * raw - bias"
            or short.get("vendor_bias_subtracted") is not False
            or not isinstance(short.get("point_timing"), Mapping)
            or short["point_timing"].get("qualified") is not True
            or short["point_timing"].get("model") != "camera2_active_array_row_exposure_midpoint"
            or not re.fullmatch(r"[0-9a-f]{64}", str(short.get("base_capture_report_sha256", "")))):
        raise VIOError("short-profile correction, centre timing, or provenance contract is invalid")
    pairs = short.get("source_and_pose_times_ns")
    if (not isinstance(pairs, (list, tuple)) or not 2 <= len(pairs) <= 12000
            or any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                   or any(type(t) is not int or not 0 < t < 2**63 for t in pair)
                   or not 0 <= pair[1] - pair[0] <= 150_000_000 for pair in pairs)
            or any(a[0] >= b[0] or a[1] >= b[1] for a, b in zip(pairs, pairs[1:]))):
        raise VIOError("short-profile source/pose timestamp mapping is invalid")
    try:
        checked_corrections(report["manifest"]["imu"]["corrections"])
        _matrix(report["manifest"]["extrinsics"]["T_imu_camera"], "short_profile.T_imu_camera")
        offset = report["manifest"]["clocks"]["imu_to_camera_offset_ns"]
        if type(offset) is not int or abs(offset) > 100_000_000:
            raise VIOError("short-profile offset must be measured integer nanoseconds")
    except (KeyError, TypeError, ValueError, MotionProfileError) as exc:
        raise VIOError("short-profile calibration/corrections are invalid") from exc
    return dict(short)


def materialize_openvins_profile_validation_input(
    capture_dir: Path,
    capture_report: Mapping[str, Any],
    output_dir: Path,
    settings: VIOSettings,
    progress: ProgressCallback,
) -> tuple[Path, Path]:
    """Use a bound fixed short-profile model without granting metric admission."""
    _checked_short_session(capture_report, validation=True)
    return _materialize_openvins_input(capture_dir, capture_report, output_dir, settings, progress,
                                      profile_validation=True)


def validate_vio_calibration_prior(prior: Mapping[str, Any], capture_id: str) -> dict[str, Any]:
    """Validate explicit provisional assumptions, without changing capture admission."""
    if prior.get("schema") != VIO_CALIBRATION_PRIOR_SCHEMA or prior.get("status") != "provisional":
        raise VIOError("online calibration requires an explicitly provisional prior")
    if not capture_id or prior.get("capture_id") != capture_id:
        raise VIOError("online calibration prior belongs to a different capture")
    if not isinstance(prior.get("provenance"), Mapping) or not prior["provenance"]:
        raise VIOError("online calibration prior must record its source assumptions")
    for key in ("camera", "extrinsics", "clocks", "imu"):
        if not isinstance(prior.get(key), Mapping):
            raise VIOError(f"online calibration prior is missing {key}")
    transform = np.asarray(_matrix(prior["extrinsics"].get("T_imu_camera"), "prior.T_imu_camera"))
    offset_s = _finite(prior["clocks"].get("imu_to_camera_offset_ns"), "prior.time_offset") * 1e-9
    if np.linalg.norm(transform[:3, 3]) > CALIBRATION_BOUNDS["translation_norm_m"]:
        raise VIOError("online calibration prior exceeds the lever-arm experiment bound")
    if abs(offset_s) > CALIBRATION_BOUNDS["absolute_time_offset_s"]:
        raise VIOError("online calibration prior exceeds the time-offset experiment bound")
    noise = prior["imu"].get("noise")
    if not isinstance(noise, Mapping):
        raise VIOError("online calibration requires explicit provisional IMU noise")
    for key in ("gyroscope_noise_density", "gyroscope_random_walk", "accelerometer_noise_density", "accelerometer_random_walk"):
        if _finite(noise.get(key), f"prior.imu.noise.{key}") <= 0:
            raise VIOError("online calibration IMU noise must be positive")
    uncertainty = prior.get("uncertainty", {})
    if not isinstance(uncertainty, Mapping):
        raise VIOError("online calibration prior uncertainty must be an object")
    normalized_uncertainty = {}
    for key, (default, maximum) in CALIBRATION_UNCERTAINTY.items():
        value = _finite(uncertainty.get(key, default), f"prior.uncertainty.{key}")
        if not 0 < value <= maximum:
            raise VIOError(f"online calibration prior {key} is outside the experiment bound")
        normalized_uncertainty[key] = value
    max_side = prior.get("analysis_max_side", 1280)
    if type(max_side) is not int or not 640 <= max_side <= 1920:
        raise VIOError("online calibration analysis_max_side must be an integer in [640, 1920]")
    return {**prior, "uncertainty": normalized_uncertainty, "bounds": dict(CALIBRATION_BOUNDS), "analysis_max_side": max_side}


def materialize_openvins_calibration_input(
    capture_dir: Path,
    capture_report: Mapping[str, Any],
    output_dir: Path,
    settings: VIOSettings,
    progress: ProgressCallback,
    *,
    prior: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Create isolated online-calibration input; provisional values cannot admit VIO."""
    manifest = capture_report.get("manifest")
    if not isinstance(manifest, Mapping) or capture_report.get("schema") != "noesis.phone_capture.v1":
        raise VIOError("online calibration requires a normalized phone capture report")
    checked = validate_vio_calibration_prior(prior, str(manifest.get("capture_id") or ""))
    return _materialize_openvins_input(capture_dir, capture_report, output_dir, settings, progress, calibration_prior=checked)


def _materialize_openvins_input(
    capture_dir: Path,
    capture_report: Mapping[str, Any],
    output_dir: Path,
    settings: VIOSettings,
    progress: ProgressCallback,
    *,
    calibration_prior: Mapping[str, Any] | None = None,
    profile_validation: bool = False,
) -> tuple[Path, Path]:

    short = None
    if "short_session" in capture_report or profile_validation:
        if calibration_prior is not None:
            raise VIOError("short-profile validation must not enable online calibration")
        short = _checked_short_session(capture_report, validation=profile_validation)
    if calibration_prior is None and not profile_validation and capture_report.get("metric_vio_allowed") is not True:
        raise VIOError("metric OpenVINS input is blocked by the capture report")
    manifest = capture_report.get("manifest")
    if not isinstance(manifest, Mapping):
        raise VIOError("capture report has no normalized manifest")
    video_meta = manifest.get("video")
    camera = manifest.get("camera")
    imu_meta = manifest.get("imu")
    clocks = manifest.get("clocks")
    extrinsics = manifest.get("extrinsics")
    if not all(isinstance(value, Mapping) for value in (video_meta, camera, imu_meta, clocks, extrinsics)):
        raise VIOError("capture report calibration rows are incomplete")
    if calibration_prior is not None:
        # Local overlay only. Never mutate or relabel the imported evidence.
        camera = {**camera, **calibration_prior["camera"]}
        imu_meta = {**imu_meta, "noise": calibration_prior["imu"]["noise"]}
        clocks = {**clocks, "imu_to_camera_offset_ns": calibration_prior["clocks"]["imu_to_camera_offset_ns"]}
        extrinsics = {"T_imu_camera": calibration_prior["extrinsics"]["T_imu_camera"]}
    orientation = int(camera.get("orientation_deg") or 0) % 360
    if orientation != 0:
        raise VIOError("OpenVINS materialization requires orientation_deg=0")
    video_path = capture_dir / str(video_meta.get("path") or "")
    timestamps_path = capture_dir / "video_timestamps_ns.json"
    try:
        raw_times = json.loads(timestamps_path.read_text(encoding="utf-8"))
        if short and (not isinstance(raw_times, list) or any(type(t) is not int for t in raw_times)):
            raise VIOError("short-profile source timestamps must be integer nanoseconds")
        frame_timestamps = [int(value) for value in raw_times]
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise VIOError("imported camera timestamps are unreadable") from exc
    if len(frame_timestamps) < 2 or any(
        frame_timestamps[index] >= frame_timestamps[index + 1]
        for index in range(len(frame_timestamps) - 1)
    ):
        raise VIOError("imported camera timestamps are not strictly increasing")
    if len(frame_timestamps) != int((capture_report.get("video") or {}).get("frame_count") or 0):
        raise VIOError("imported camera timestamp count does not match the capture report")
    if len(frame_timestamps) > settings.max_input_frames:
        raise VIOError(
            f"camera stream has {len(frame_timestamps)} frames; "
            f"OpenVINS limit is {settings.max_input_frames}"
        )
    if not video_path.is_file():
        raise VIOError(f"imported camera video is missing: {video_path}")
    source_timestamps = list(frame_timestamps)
    source_artifacts = {}
    if short:
        if source_timestamps != [pair[0] for pair in short["source_and_pose_times_ns"]]:
            raise VIOError("short-profile mapping differs from original camera timestamps")
        frame_timestamps = [pair[1] for pair in short["source_and_pose_times_ns"]]
        if output_dir.resolve() == capture_dir.resolve() or output_dir.resolve().is_relative_to(capture_dir.resolve()):
            raise VIOError("short-profile output must be separate from retained capture")
        source_names = [str(video_meta["path"]), "video_timestamps_ns.json", "imu_normalized.json"]
        android = manifest.get("android_capture")
        if isinstance(android, Mapping):
            source_names.extend(str(android[k]) for k in ("camera_results_path", "capture_result_path") if k in android)
        for name in source_names:
            path = _source_file(capture_dir, name)
            source_artifacts[name] = {"sha256": _file_sha(path), "bytes": path.stat().st_size}

    # The calibration is expressed in the encoded image axes.  ffmpeg's
    # default display-matrix autorotation would silently move those axes, so
    # probe and enforce the encoded geometry at the native-estimator boundary.
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=min(settings.timeout_s, 120),
    )
    try:
        streams = json.loads(probe.stdout).get("streams")
        encoded_width = int(streams[0]["width"])
        encoded_height = int(streams[0]["height"])
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VIOError("encoded camera dimensions could not be probed") from exc
    if probe.returncode != 0 or encoded_width <= 0 or encoded_height <= 0:
        raise VIOError("encoded camera dimensions are invalid")
    resolution = camera.get("resolution_px")
    crop = camera.get("crop")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or [int(resolution[0]), int(resolution[1])] != [encoded_width, encoded_height]
        or not isinstance(crop, Mapping)
        or int(crop.get("left") or 0) != 0
        or int(crop.get("top") or 0) != 0
        or int(crop.get("width") or 0) != encoded_width
        or int(crop.get("height") or 0) != encoded_height
    ):
        raise VIOError("encoded camera geometry does not match calibrated K/crop")
    analysis_resize = None
    if calibration_prior is not None or short is not None:
        max_side = (short if short is not None else calibration_prior)["analysis_max_side"]
        scale = min(1.0, max_side / max(encoded_width, encoded_height))
        analysis_width = max(1, int(round(encoded_width * scale)))
        analysis_height = max(1, int(round(encoded_height * scale)))
        if [analysis_width, analysis_height] != [encoded_width, encoded_height]:
            sx, sy = analysis_width / encoded_width, analysis_height / encoded_height
            pixel_transform = np.asarray([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
            source_k = np.asarray(camera.get("intrinsics"), dtype=np.float64)
            if source_k.shape != (3, 3):
                raise VIOError("online calibration camera prior has an invalid K")
            analysis_resize = {
                "operation": "ffmpeg_bilinear_pixel_center_resize",
                "source_resolution_px": [encoded_width, encoded_height],
                "source_intrinsics": source_k.tolist(),
                "resolution_px": [analysis_width, analysis_height],
                "T_analysis_pixel_source_pixel": pixel_transform.tolist(),
            }
            camera = {**camera, "intrinsics": (pixel_transform @ source_k).tolist(), "resolution_px": [analysis_width, analysis_height]}
    camera_projection, rectification = _openvins_camera_projection(camera)
    if analysis_resize is not None:
        camera_projection["analysis_resize"] = analysis_resize

    output_dir.mkdir(parents=True, exist_ok=True)
    input_root = output_dir / "openvins_input"
    image_dir = input_root / "cam0" / "data"
    imu_dir = input_root / "imu0"
    logs_dir = input_root / "logs"
    image_dir.mkdir(parents=True, exist_ok=False)
    imu_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_log = input_root / "ffmpeg_extract.log"
    video_filter = "showinfo"
    if analysis_resize is not None:
        video_filter = f"scale={analysis_width}:{analysis_height}:flags=bilinear,showinfo"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-nostdin",
        "-noautorotate",
        *(["-threads", "4", "-filter_threads", "2"] if calibration_prior is not None or short is not None else []),
        "-i", str(video_path), "-map", "0:v:0", "-vf", video_filter,
        "-fps_mode", "passthrough", "-start_number", "0",
        *(["-threads", "4"] if calibration_prior is not None or short is not None else []),
        str(image_dir / "frame-%08d.png"),
    ]
    progress(0.12, "Decoding the dense camera stream for OpenVINS")
    try:
        with ffmpeg_log.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=str(input_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log_handle,
            )
            return_code = process.wait(timeout=settings.timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            process.kill()
            process.wait(timeout=10)
        raise VIOError("dense camera extraction did not complete") from exc
    if return_code != 0:
        detail = ffmpeg_log.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise VIOError(f"dense camera extraction failed: {detail.strip()}")
    image_paths = sorted(image_dir.glob("frame-*.png"))
    showinfo_pattern = re.compile(r"\bn:\s*(\d+).*?\bpts_time:([^\s]+)")
    source_indices: list[int] = []
    for line in ffmpeg_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = showinfo_pattern.search(line)
        if match:
            source_indices.append(int(match.group(1)))
    if len(image_paths) != len(frame_timestamps) or source_indices != list(range(len(image_paths))):
        raise VIOError(
            "dense extraction changed the camera frame sequence; "
            "exact capture-time mapping was not admitted"
        )
    image_provenance: list[dict[str, str]] = [{} for _ in image_paths]
    if rectification is not None:
        progress(0.20, "Rectifying the full camera distortion model and excluding invalid border rays")
        image_provenance = _rectify_dense_camera_images(
            image_paths, image_dir, camera_projection, rectification, timeout_s=settings.timeout_s,
        )

    normalized_path = capture_dir / "imu_normalized.json"
    try:
        normalized_imu = json.loads(normalized_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise VIOError("normalized IMU streams are unreadable") from exc
    offset_ns = int(round(float(clocks.get("imu_to_camera_offset_ns") or 0.0)))
    imu_rows: list[tuple[int, list[float], list[float]]]
    stream_method = "shared_timestamp"
    if str(imu_meta.get("sample_clock_mode") or "") == "separate_streams":
        if not isinstance(normalized_imu, Mapping) or not isinstance(normalized_imu.get("accel"), list) or not isinstance(normalized_imu.get("gyro"), list):
            raise VIOError("separate normalized accelerometer and gyro streams are missing")
        accel_rows = list(normalized_imu["accel"])
        gyro_rows = list(normalized_imu["gyro"])
        if len(accel_rows) < 2 or len(gyro_rows) < 2:
            raise VIOError("separate IMU streams have too few samples")
        accel_times = [int(row["timestamp_ns"]) for row in accel_rows]
        gyro_times = [int(row["timestamp_ns"]) for row in gyro_rows]
        accel_values = [[float(value) for value in row["si"]] for row in accel_rows]
        gyro_values = [[float(value) for value in row["si"]] for row in gyro_rows]
        if short:
            from .motion_preprocessing import correct_samples, MotionProfileError
            for rows, times in ((accel_rows, accel_times), (gyro_rows, gyro_times)):
                if (len(rows) > 300000 or any(type(r.get("timestamp_ns")) is not int for r in rows)
                        or any(not 0 < b - a <= 50_000_000 for a, b in zip(times, times[1:]))):
                    raise VIOError("short-profile IMU timestamps/cadence exceed their bound")
            try:
                # Apply to each original stream exactly once BEFORE creating
                # the joint grid. Vendor bias estimates are not subtracted.
                accel_values = correct_samples(accel_values, imu_meta["corrections"], "accelerometer").tolist()
                gyro_values = correct_samples(gyro_values, imu_meta["corrections"], "gyroscope").tolist()
            except (ValueError, MotionProfileError) as exc:
                raise VIOError("short-profile IMU correction failed") from exc
        # Keep the common stream range, including the samples bracketing the
        # camera interval.  Dropping those rows forces the estimator to start
        # or end on an interpolated value and breaks nonzero time offsets.
        common_start = max(accel_times[0], gyro_times[0])
        common_end = min(accel_times[-1], gyro_times[-1])
        camera_imu_start = frame_timestamps[0] - offset_ns
        camera_imu_end = frame_timestamps[-1] - offset_ns
        if common_start > camera_imu_start or common_end < camera_imu_end:
            raise VIOError("separate IMU streams do not bracket the camera interval")
        grid = sorted({
            timestamp
            for timestamp in accel_times + gyro_times
            if common_start <= timestamp <= common_end
        })
        if len(grid) < 2:
            raise VIOError("separate IMU streams have no common camera-covered interval")
        imu_rows = [
            (
                timestamp,
                _interpolate_stream(accel_times, accel_values, timestamp),
                _interpolate_stream(gyro_times, gyro_values, timestamp),
            )
            for timestamp in grid
        ]
        stream_method = "linear_interpolation_to_union_timestamp_grid_with_brackets"
    elif short:
        raise VIOError("short profiles require original separate IMU streams")
    elif isinstance(normalized_imu, list):
        imu_rows = [
            (
                int(row["timestamp_ns"]),
                [float(value) for value in row["accel_mps2"]],
                [float(value) for value in row["gyro_rads"]],
            )
            for row in normalized_imu
        ]
    else:
        raise VIOError("shared normalized IMU samples are missing")
    if len(imu_rows) < 2:
        raise VIOError("materialized IMU stream has too few samples")
    if calibration_prior is not None:
        # The complete allowed offset range must be covered before optimization.
        margin_ns = int(CALIBRATION_BOUNDS["time_offset_change_s"] * 1e9)
        if imu_rows[0][0] > frame_timestamps[0] - offset_ns - margin_ns or imu_rows[-1][0] < frame_timestamps[-1] - offset_ns + margin_ns:
            raise VIOError("online calibration requires IMU brackets across the allowed time-offset range")

    camera_csv = input_root / "cam0" / "data.csv"
    with camera_csv.open("w", encoding="utf-8", newline="") as handle:
        for timestamp, image_path in zip(frame_timestamps, image_paths, strict=True):
            handle.write(f"{timestamp},{image_path.name}\n")
    imu_csv = imu_dir / "data.csv"
    with imu_csv.open("w", encoding="utf-8", newline="") as handle:
        for timestamp, accel, gyro in imu_rows:
            handle.write(
                f"{timestamp},{gyro[0]:.17g},{gyro[1]:.17g},{gyro[2]:.17g},"
                f"{accel[0]:.17g},{accel[1]:.17g},{accel[2]:.17g}\n"
            )

    estimator_camera = camera_projection["estimator_camera"]
    intrinsic_matrix = estimator_camera["K"]
    resolution = estimator_camera["resolution_px"]
    t_imu_camera = extrinsics.get("T_imu_camera")
    noise = imu_meta.get("noise")
    if (
        not isinstance(intrinsic_matrix, list)
        or not isinstance(resolution, list)
        or not isinstance(t_imu_camera, list)
        or not isinstance(noise, Mapping)
        or len(intrinsic_matrix) != 3
        or len(resolution) != 2
    ):
        raise VIOError("metric capture is missing camera or IMU calibration values")
    try:
        fx, fy = float(intrinsic_matrix[0][0]), float(intrinsic_matrix[1][1])
        cx, cy = float(intrinsic_matrix[0][2]), float(intrinsic_matrix[1][2])
        width, height = int(resolution[0]), int(resolution[1])
        distortion = estimator_camera["distortion"]
        offset_s = -offset_ns * 1e-9
        noise_values = {
            key: float(noise[key])
            for key in (
                "gyroscope_noise_density",
                "gyroscope_random_walk",
                "accelerometer_noise_density",
                "accelerometer_random_walk",
            )
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise VIOError("metric capture calibration values are invalid") from exc
    if not all(math.isfinite(value) and value > 0.0 for value in (fx, fy, *noise_values.values())):
        raise VIOError("camera focal lengths and IMU noise must be finite and positive")
    if width <= 0 or height <= 0:
        raise VIOError("camera resolution must be positive")
    imucam_text = (
        "%YAML:1.0\n"
        "cam0:\n"
        f"  T_imu_cam: {_yaml_matrix(t_imu_camera)}\n"
        "  cam_overlaps: [0]\n"
        "  camera_model: pinhole\n"
        f"  distortion_model: {estimator_camera['distortion_model']}\n"
        f"  distortion_coeffs: {_yaml_vector(distortion)}\n"
        f"  intrinsics: {_yaml_vector([fx, fy, cx, cy])}\n"
        f"  resolution: [{width}, {height}]\n"
        f"  timeshift_cam_imu: {offset_s:.17g}\n"
    )
    (input_root / "kalibr_imucam_chain.yaml").write_text(imucam_text, encoding="utf-8")
    imu_text = (
        "%YAML:1.0\n"
        "imu0:\n"
        "  T_i_b: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n"
        f"  accelerometer_noise_density: {noise_values['accelerometer_noise_density']:.17g}\n"
        f"  accelerometer_random_walk: {noise_values['accelerometer_random_walk']:.17g}\n"
        f"  gyroscope_noise_density: {noise_values['gyroscope_noise_density']:.17g}\n"
        f"  gyroscope_random_walk: {noise_values['gyroscope_random_walk']:.17g}\n"
        "  model: kalibr\n"
        "  Tw: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n"
        "  R_IMUtoGYRO: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n"
        "  Ta: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n"
        "  R_IMUtoACC: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n"
        "  Tg: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]\n"
    )
    (input_root / "kalibr_imu_chain.yaml").write_text(imu_text, encoding="utf-8")
    if settings.config is None or not settings.config.is_file():
        raise VIOError("OpenVINS base configuration is missing")
    config_text = settings.config.read_text(encoding="utf-8")
    config_text = _replace_yaml_key(config_text, "relative_config_imu", '"kalibr_imu_chain.yaml"')
    config_text = _replace_yaml_key(config_text, "relative_config_imucam", '"kalibr_imucam_chain.yaml"')
    config_text = _replace_yaml_key(config_text, "use_stereo", "false")
    config_text = _replace_yaml_key(config_text, "max_cameras", "1")
    config_text = _replace_yaml_key(config_text, "calib_cam_extrinsics", "false")
    config_text = _replace_yaml_key(config_text, "calib_cam_intrinsics", "false")
    config_text = _replace_yaml_key(config_text, "calib_cam_timeoffset", "false")
    if calibration_prior is not None:
        config_text = _replace_yaml_key(config_text, "calib_cam_extrinsics", "true")
        config_text = _replace_yaml_key(config_text, "calib_cam_timeoffset", "true")
        config_text = _replace_yaml_key(config_text, "init_dyn_use", "true")
        # The pinned dynamic initializer does not return optimized calibration.
        config_text = _replace_yaml_key(config_text, "init_dyn_mle_opt_calib", "false")
        config_text = _replace_yaml_key(config_text, "multi_threading_subs", "false")
    if short:
        for key, value in {
            "calib_imu_intrinsics": "false", "calib_imu_g_sensitivity": "false",
            "init_dyn_mle_opt_calib": "false", "downsample_cameras": "false",
            "multi_threading_subs": "false", "multi_threading_pubs": "false",
            "num_opencv_threads": "4", "init_dyn_mle_max_threads": "2",
        }.items():
            config_text = _replace_yaml_key(config_text, key, value)
    config_text = _replace_yaml_key(config_text, "record_timing_filepath", '"logs/openvins_timing.txt"')
    generated_config = input_root / "estimator_config.yaml"
    generated_config.write_text(config_text, encoding="utf-8")
    metadata = {
                "schema": VIO_PROFILE_INPUT_SCHEMA if profile_validation else "noesis.phone_capture.openvins_calibration_input.v1" if calibration_prior is not None else "noesis.phone_capture.openvins_input.v1",
                "mode": "profile_validation" if profile_validation else "calibration" if calibration_prior is not None else "metric_vio",
                "capture_metric_vio_allowed": capture_report.get("metric_vio_allowed") is True,
                **({"calibration_prior": calibration_prior, "accepted_for_metric_vio": False} if calibration_prior is not None else {}),
                "capture_id": str(manifest.get("capture_id") or ""),
                "camera_sensor_id": str(camera.get("id") or ""),
                "time_domain": str(clocks.get("camera_domain") or ""),
                "estimator_config_sha256": hashlib.sha256(generated_config.read_bytes()).hexdigest(),
                "frame_count": len(image_paths),
                "frame_mapping": [
                    {
                        "source_frame_index": index,
                        "capture_time_ns": source_timestamps[index],
                        **({"pose_time_ns": timestamp} if short else {}),
                        "filename": path.name,
                        **image_provenance[index],
                    }
                    for index, (timestamp, path) in enumerate(zip(frame_timestamps, image_paths, strict=True))
                ],
                "camera_preprocessing": camera_projection,
                "imu_sample_count": len(imu_rows),
                "imu_time_offset_applied_in_config_s": offset_s,
                "separate_stream_method": stream_method,
                "dense_source_timestamp": str((capture_report.get("video") or {}).get("timestamp_source") or ""),
            }
    if short:
        mapping_path = input_root / "source_frame_mapping.csv"
        with mapping_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["source_frame_index", "capture_time_ns", "pose_time_ns"])
            writer.writerows((row["source_frame_index"], row["capture_time_ns"], row["pose_time_ns"]) for row in metadata["frame_mapping"])
        report_path = input_root / "short_session_capture_report.json"
        report_path.write_text(json.dumps(capture_report, sort_keys=True, allow_nan=False), encoding="utf-8")
        for row, image_path in zip(metadata["frame_mapping"], image_paths, strict=True):
            row["estimator_image_sha256"] = _file_sha(image_path)
        names = ["source_frame_mapping.csv", "short_session_capture_report.json", "cam0/data.csv", "imu0/data.csv",
                 "kalibr_imucam_chain.yaml", "kalibr_imu_chain.yaml", "estimator_config.yaml"]
        metadata.update({
            "short_session": short, "accepted_for_metric_vio": False,
            "runtime_quality_control_required": not profile_validation,
            "pose_time_reference": POSE_TIME_REFERENCE,
            "source_capture_dir": str(capture_dir.resolve()), "source_artifacts": source_artifacts,
            "input_artifacts": {name: {"sha256": _file_sha(input_root / name), "bytes": (input_root / name).stat().st_size} for name in names},
            "imu_preprocessing": {"equation": "corrected = matrix * raw - bias", "applied_before_joint_interpolation": True,
                                  "application_count": 1, "corrections": imu_meta["corrections"],
                                  "vendor_bias_subtracted": False, "native_imu_corrections": "identity",
                                  "noise_already_transformed_by_profile": True},
            "fixed_calibration": {"T_imu_camera": t_imu_camera, "imu_to_camera_offset_ns": offset_ns},
        })
        for name, info in source_artifacts.items():
            if _file_sha(_source_file(capture_dir, name)) != info["sha256"]:
                raise VIOError("short-profile source changed during materialization")
    (input_root / "openvins_input.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    progress(0.28, f"Materialized {len(image_paths)} dense camera frames and {len(imu_rows)} IMU samples")
    return input_root, generated_config


def validate_vio_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the portable estimator output consumed by WO-3."""

    if payload.get("schema") != VIO_SCHEMA:
        raise VIOError(f"VIO output must use {VIO_SCHEMA}")
    if payload.get("estimator") != "openvins":
        raise VIOError("VIO output estimator identity is not OpenVINS")
    if payload.get("accepted_for_metric_vio") is not True:
        raise VIOError("VIO output was not accepted for metric use")
    if "short_session_consumer" in payload:
        if payload.get("status") != "completed":
            raise VIOError("short-profile metric result has no completed runtime quality admission")
        for field in ("direct_runtime_quality", "runtime_quality_control"):
            proof = payload.get(field)
            if not isinstance(proof, Mapping) or proof.get("accepted") is not True:
                raise VIOError(f"short-profile metric result has no retained {field} approval")
    return _validate_vio_trajectory(payload)


def _validate_vio_trajectory(payload: Mapping[str, Any], *, calibration: bool = False,
                             profile_validation: bool = False) -> dict[str, Any]:
    """Check pose semantics without converting provisional output to admission."""
    if payload.get("estimator") != "openvins":
        raise VIOError("VIO output estimator identity is not OpenVINS")
    frame = payload.get("frame")
    if (
        not isinstance(frame, Mapping)
        or frame.get("source") != "camera"
        or frame.get("target") != "vio_world"
        or frame.get("pose_convention") != "T_vio_world_camera"
        or frame.get("units") != "meters"
        or frame.get("camera_axes") != "x_right_y_down_z_forward"
        or frame.get("world_axes") != "z_up_gravity_up"
        or frame.get("pose_origin") != "camera_optical_center"
        or frame.get("velocity_origin") != "imu_center"
        or frame.get("gravity_frame") != "vio_world"
        or frame.get("gravity_semantics") != "physical_world_acceleration"
        or not frame.get("capture_id")
        or not frame.get("camera_sensor_id")
        or not frame.get("time_domain")
    ):
        raise VIOError("VIO output has no explicit camera/vio_world frame contract")
    scale = payload.get("scale")
    if calibration and profile_validation:
        raise VIOError("profile validation is not online calibration")
    scale_mode = "provisional_metric" if calibration or profile_validation else "metric"
    scale_source = "short_session_profile_validation" if profile_validation else "provisional_imu_camera_prior_online_calibration" if calibration else "imu_camera_calibration"
    if not isinstance(scale, Mapping) or scale.get("mode") != scale_mode or scale.get("source") != scale_source:
        raise VIOError("VIO output has no calibrated metric-scale declaration")
    rows = payload.get("poses")
    if not isinstance(rows, list) or len(rows) < 2:
        raise VIOError("VIO output contains too few poses")
    previous = None
    previous_pose_time = None
    short = payload.get("short_session_consumer")
    if profile_validation or short is not None:
        if (not isinstance(short, Mapping) or short.get("consumer_version") != SHORT_CONSUMER_VERSION
                or short.get("validation_run") is not profile_validation
                or short.get("fixed_camera_imu_calibration") is not True
                or short.get("native_imu_corrections") != "identity"
                or short.get("dense_frame_mapping_verified") is not True
                or short.get("image_motion_model") != SHORT_IMAGE_MODEL
                or short.get("rolling_shutter_compensated") is not False
                or not short.get("profile_id")
                or frame.get("pose_time_reference") != POSE_TIME_REFERENCE
                or frame.get("capture_time_reference") != "original_camera_sensor_timestamp"):
            raise VIOError("short-profile output does not confirm the fixed centre-timed consumer")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise VIOError(f"VIO pose {index} is invalid")
        timestamp = row.get("capture_time_ns")
        if type(timestamp) is not int:
            raise VIOError(f"VIO pose {index} has no integer capture_time_ns")
        if previous is not None and timestamp <= previous:
            raise VIOError("VIO pose timestamps are not strictly increasing")
        previous = timestamp
        if short is not None:
            pose_time = row.get("pose_time_ns")
            if (type(pose_time) is not int or not 0 <= pose_time - timestamp <= 150_000_000
                    or (previous_pose_time is not None and pose_time <= previous_pose_time)):
                raise VIOError("short-profile output has invalid centre-exposure pose timestamps")
            previous_pose_time = pose_time
        if not isinstance(row.get("prepared_frame_id"), str) or not row["prepared_frame_id"]:
            raise VIOError(f"VIO pose {index} has no prepared frame identity")
        _matrix(row.get("T_vio_world_camera"), f"poses[{index}].T_vio_world_camera")
        for key in ("velocity_mps", "gravity_mps2", "gyro_bias_rads", "accel_bias_mps2"):
            values = row.get(key)
            if not isinstance(values, list) or len(values) != 3:
                raise VIOError(f"poses[{index}].{key} must contain 3 values")
            for value in values:
                _finite(value, f"poses[{index}].{key}")
        covariance = row.get("covariance")
        if not isinstance(covariance, list) or len(covariance) != 6:
            raise VIOError(f"poses[{index}].covariance must be a 6x6 row-major matrix")
        for value in covariance:
            if not isinstance(value, list) or len(value) != 6:
                raise VIOError(f"poses[{index}].covariance must be a 6x6 row-major matrix")
            for item in value:
                _finite(item, f"poses[{index}].covariance")
        covariance_array = np.asarray(covariance, dtype=np.float64)
        if not np.allclose(covariance_array, covariance_array.T, atol=1e-9):
            raise VIOError(f"poses[{index}].covariance is not symmetric")
        if float(np.min(np.linalg.eigvalsh(covariance_array))) < -1e-9:
            raise VIOError(f"poses[{index}].covariance is not positive semidefinite")
    quality = payload.get("quality")
    if not isinstance(quality, Mapping):
        raise VIOError("VIO output has no quality report")
    if quality.get("initialized") is not True:
        raise VIOError("VIO output did not report successful initialization")
    if not isinstance(quality.get("tracking_ratio"), (int, float)) or not 0.0 < float(quality["tracking_ratio"]) <= 1.0:
        raise VIOError("VIO output has no usable tracking quality")
    if not isinstance(payload.get("segments"), list) or not payload["segments"]:
        raise VIOError("VIO output has no segment/reset report")
    for index, segment in enumerate(payload["segments"]):
        if not isinstance(segment, Mapping) or not isinstance(segment.get("id"), str) or not isinstance(segment.get("reset"), bool):
            raise VIOError(f"VIO segment {index} has no explicit reset identity")
    if payload.get("covariance_frame") != "camera_pose_tangent_se3_row_major" or payload.get("covariance_tangent_frame") != "vio_world_rotation_additive_position":
        raise VIOError("VIO covariance frame/order is not declared")
    return dict(payload)


def _validate_calibration_state(row: Any, field: str) -> tuple[np.ndarray, float]:
    if not isinstance(row, Mapping):
        raise VIOError(f"{field} is missing")
    transform = np.asarray(_matrix(row.get("T_imu_camera"), f"{field}.T_imu_camera"))
    offset = _finite(row.get("camera_to_imu_offset_s"), f"{field}.camera_to_imu_offset_s")
    other_offset = _finite(row.get("imu_to_camera_offset_ns"), f"{field}.imu_to_camera_offset_ns") * 1e-9
    if not math.isclose(offset, -other_offset, abs_tol=1e-12):
        raise VIOError(f"{field} time-offset conventions disagree")
    order = ["theta_x_rad", "theta_y_rad", "theta_z_rad", "p_I_in_C_x_m", "p_I_in_C_y_m", "p_I_in_C_z_m", "camera_to_imu_offset_s"]
    if row.get("covariance_convention") != CALIBRATION_COVARIANCE_CONVENTION or row.get("covariance_order") != order:
        raise VIOError(f"{field} calibration covariance convention is missing")
    raw_covariance = row.get("covariance")
    if not isinstance(raw_covariance, list) or len(raw_covariance) != 7 or any(not isinstance(item, list) or len(item) != 7 for item in raw_covariance):
        raise VIOError(f"{field}.covariance must be a 7x7 matrix")
    covariance = np.asarray([[_finite(item, f"{field}.covariance") for item in values] for values in raw_covariance])
    if not np.allclose(covariance, covariance.T, atol=1e-9) or np.min(np.linalg.eigvalsh(covariance)) < -1e-9:
        raise VIOError(f"{field}.covariance must be symmetric positive semidefinite")
    if np.linalg.norm(transform[:3, 3]) > CALIBRATION_BOUNDS["translation_norm_m"] or abs(offset) > CALIBRATION_BOUNDS["absolute_time_offset_s"]:
        raise VIOError(f"{field} exceeds an absolute experiment bound")
    return transform, offset


def validate_vio_calibration_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Require a complete provisional trace, including calibration uncertainty."""
    if payload.get("schema") != VIO_CALIBRATION_SCHEMA or payload.get("accepted_for_metric_vio") is not False or payload.get("status") != "provisional":
        raise VIOError("online calibration result must remain explicitly provisional")
    if payload.get("camera_intrinsics_optimized") is not False or payload.get("bounds") != CALIBRATION_BOUNDS:
        raise VIOError("online calibration result does not confirm the fixed-camera experiment bounds")
    result = _validate_vio_trajectory(payload, calibration=True)
    seed, seed_offset = _validate_calibration_state(payload.get("initial_calibration"), "initial_calibration")
    for index, row in enumerate([*payload["poses"], {"calibration": payload.get("calibration")}]):
        transform, offset = _validate_calibration_state(row.get("calibration"), f"calibration_history[{index}]")
        rotation_change = math.acos(float(np.clip((np.trace(transform[:3, :3] @ seed[:3, :3].T) - 1) / 2, -1, 1)))
        if (
            rotation_change > CALIBRATION_BOUNDS["rotation_change_rad"]
            or np.linalg.norm(transform[:3, 3] - seed[:3, 3]) > CALIBRATION_BOUNDS["translation_change_m"]
            or abs(offset - seed_offset) > CALIBRATION_BOUNDS["time_offset_change_s"]
        ):
            raise VIOError("online calibration history exceeds a seed-change experiment bound")
    if payload["calibration"] != payload["poses"][-1]["calibration"]:
        raise VIOError("final online calibration does not match its last camera state")
    return result


def validate_vio_profile_validation_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    if (payload.get("schema") != VIO_PROFILE_VALIDATION_SCHEMA
            or payload.get("accepted_for_metric_vio") is not False or payload.get("status") != "provisional"):
        raise VIOError("short-profile validation output must remain explicitly provisional")
    return _validate_vio_trajectory(payload, profile_validation=True)


def _read_result(path: Path, *, calibration: bool = False, profile_validation: bool = False,
                 pending_short_quality: bool = False) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VIOError(f"OpenVINS output is unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise VIOError("OpenVINS output must be a JSON object")
    if profile_validation:
        return validate_vio_profile_validation_result(payload)
    if pending_short_quality:
        if (payload.get("schema") != VIO_SCHEMA or payload.get("accepted_for_metric_vio") is not False
                or payload.get("status") != "pending_runtime_quality_control"):
            raise VIOError("short-profile metric result must await runtime quality control")
        return _validate_vio_trajectory(payload)
    return validate_vio_calibration_result(payload) if calibration else validate_vio_result(payload)


def run_openvins(
    capture_dir: Path,
    output_dir: Path,
    settings: VIOSettings,
    progress: ProgressCallback,
    *,
    runtime_quality_check: RuntimeQualityCheck | None = None,
) -> dict[str, Any]:
    """Run the pinned native OpenVINS bridge and validate its JSON result."""

    return _run_openvins(capture_dir, output_dir, settings, progress,
                         runtime_quality_check=runtime_quality_check)


def run_openvins_calibration(
    capture_dir: Path,
    output_dir: Path,
    settings: VIOSettings,
    progress: ProgressCallback,
) -> dict[str, Any]:
    """Run explicitly provisional online extrinsic/time calibration in isolation."""
    return _run_openvins(capture_dir, output_dir, settings, progress, calibration=True)


def run_openvins_profile_validation(
    capture_dir: Path,
    output_dir: Path,
    settings: VIOSettings,
    progress: ProgressCallback,
) -> dict[str, Any]:
    """Run a fixed short-session profile as non-admitted validation evidence."""
    return _run_openvins(capture_dir, output_dir, settings, progress, profile_validation=True)


def _verify_short_input(root, metadata, settings, *, validation):
    expected_schema = VIO_PROFILE_INPUT_SCHEMA if validation else "noesis.phone_capture.openvins_input.v1"
    if (metadata.get("schema") != expected_schema
            or metadata.get("mode") != ("profile_validation" if validation else "metric_vio")
            or metadata.get("accepted_for_metric_vio") is not False
            or metadata.get("capture_metric_vio_allowed") is not (not validation)
            or metadata.get("runtime_quality_control_required") is not (not validation)):
        raise VIOError("short-profile materialized input has the wrong mode/admission contract")
    artifacts = metadata.get("input_artifacts")
    required = {"source_frame_mapping.csv", "short_session_capture_report.json", "cam0/data.csv", "imu0/data.csv",
                "kalibr_imucam_chain.yaml", "kalibr_imu_chain.yaml", "estimator_config.yaml"}
    if not isinstance(artifacts, Mapping) or set(artifacts) != required:
        raise VIOError("short-profile input artifact binding is incomplete")
    for name, info in artifacts.items():
        path = _source_file(root, name)
        if not isinstance(info, Mapping) or path.stat().st_size != info.get("bytes") or _file_sha(path) != info.get("sha256"):
            raise VIOError("short-profile input artifact changed after materialization")
    if settings.config.resolve() != (root / "estimator_config.yaml").resolve() or _file_sha(settings.config) != metadata.get("estimator_config_sha256"):
        raise VIOError("short-profile config differs from its materialized input")
    report = json.loads((root / "short_session_capture_report.json").read_text())
    short = _checked_short_session(report, validation=validation)
    if metadata.get("short_session") != short or metadata.get("pose_time_reference") != POSE_TIME_REFERENCE:
        raise VIOError("short-profile consumer metadata differs from its bound report")
    for key, expected in (("capture_id", report["manifest"]["capture_id"]),
                          ("camera_sensor_id", report["manifest"]["camera"]["id"]),
                          ("time_domain", report["manifest"]["clocks"]["camera_domain"])):
        if metadata.get(key) != expected:
            raise VIOError("short-profile input changed its source identity")
    source = Path(metadata.get("source_capture_dir", ""))
    sources = metadata.get("source_artifacts")
    if not isinstance(sources, Mapping) or not {"video_timestamps_ns.json", "imu_normalized.json", report["manifest"]["video"]["path"]} <= set(sources):
        raise VIOError("short-profile original source binding is incomplete")
    for name, info in sources.items():
        path = _source_file(source, name)
        if not isinstance(info, Mapping) or path.stat().st_size != info.get("bytes") or _file_sha(path) != info.get("sha256"):
            raise VIOError("short-profile original source changed after materialization")
    source_times = json.loads((source / "video_timestamps_ns.json").read_text())
    if source_times != [pair[0] for pair in short["source_and_pose_times_ns"]]:
        raise VIOError("short-profile input changed the original timestamp mapping")
    mappings = metadata.get("frame_mapping")
    if not isinstance(mappings, list) or len(mappings) != len(source_times) or metadata.get("frame_count") != len(mappings):
        raise VIOError("short-profile dense camera mapping is incomplete")
    expected_csv = "source_frame_index,capture_time_ns,pose_time_ns\n"
    camera_csv = ""
    for index, (row, pair) in enumerate(zip(mappings, short["source_and_pose_times_ns"], strict=True)):
        if (not isinstance(row, Mapping) or row.get("source_frame_index") != index
                or row.get("capture_time_ns") != pair[0] or row.get("pose_time_ns") != pair[1]
                or row.get("filename") != f"frame-{index:08d}.png"):
            raise VIOError("short-profile dense frame mapping changed")
        image_path = _source_file(root, "cam0/data/" + row["filename"])
        if _file_sha(image_path) != row.get("estimator_image_sha256"):
            raise VIOError("short-profile analysis image changed")
        expected_csv += f"{index},{pair[0]},{pair[1]}\n"
        camera_csv += f"{pair[1]},{row['filename']}\n"
    if (root / "source_frame_mapping.csv").read_text() != expected_csv or (root / "cam0/data.csv").read_text() != camera_csv:
        raise VIOError("short-profile native timestamp mapping differs from source/pose identity")
    preprocessing = metadata.get("imu_preprocessing", {})
    if (preprocessing.get("corrections") != report["manifest"]["imu"]["corrections"]
            or preprocessing.get("application_count") != 1 or preprocessing.get("applied_before_joint_interpolation") is not True
            or preprocessing.get("native_imu_corrections") != "identity"
            or preprocessing.get("noise_already_transformed_by_profile") is not True):
        raise VIOError("short-profile IMU correction provenance changed")
    return short


def _short_runtime_quality(result, metadata):
    """Recompute bounded tracking checks from exact rows, not native ratios."""
    poses = result["poses"]
    total = len(metadata["frame_mapping"])
    coverage = len(poses) / total
    max_gap = max(b["pose_time_ns"] - a["pose_time_ns"] for a, b in zip(poses, poses[1:])) * 1e-9
    end_gap = (metadata["frame_mapping"][-1]["pose_time_ns"] - poses[-1]["pose_time_ns"]) * 1e-9
    reset = any(segment.get("reset") is not False for segment in result["segments"])
    accepted = (coverage >= SHORT_RUNTIME_QUALITY_POLICY["minimum_camera_coverage"]
                and max_gap <= SHORT_RUNTIME_QUALITY_POLICY["maximum_pose_gap_s"]
                and 0 <= end_gap <= SHORT_RUNTIME_QUALITY_POLICY["maximum_end_gap_s"]
                and not reset and result["quality"].get("reset_count") == 0)
    return {"accepted": accepted, "policy": dict(SHORT_RUNTIME_QUALITY_POLICY),
            "camera_coverage": coverage, "maximum_pose_gap_s": max_gap, "end_gap_s": end_gap,
            "reset_observed": reset, "accuracy_validated": False}


def _run_openvins(
    capture_dir: Path,
    output_dir: Path,
    settings: VIOSettings,
    progress: ProgressCallback,
    *,
    calibration: bool = False,
    profile_validation: bool = False,
    runtime_quality_check: RuntimeQualityCheck | None = None,
) -> dict[str, Any]:

    if settings.estimator != "openvins":
        raise VIOError(f"unsupported fixed estimator: {settings.estimator}")
    if settings.executable is None or not settings.executable.is_file() or not os.access(settings.executable, os.X_OK):
        raise VIOError("OpenVINS bridge executable is not installed; configure NOESIS_PHONE_SCAN_VIO_EXECUTABLE")
    if settings.config is None or not settings.config.is_file():
        raise VIOError("OpenVINS config is missing; configure NOESIS_PHONE_SCAN_VIO_CONFIG")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / ("vio_profile_validation_result.json" if profile_validation else "vio_calibration_result.json" if calibration else "vio_result.json")
    if result_path.exists():
        result_path.unlink()
    stdout_path = output_dir / "openvins.stdout.log"
    stderr_path = output_dir / "openvins.stderr.log"
    command = [
        str(settings.executable),
        "--capture-dir", str(capture_dir),
        "--config", str(settings.config),
        "--output", str(result_path),
    ]
    input_metadata_path = capture_dir / "openvins_input.json"
    required_mask: dict[str, Any] | None = None
    calibration_prior = None
    input_metadata: dict[str, Any] = {}
    short = None
    if input_metadata_path.is_file():
        try:
            input_metadata = json.loads(input_metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise VIOError("OpenVINS input metadata is unreadable") from exc
        if not isinstance(input_metadata, dict):
            raise VIOError("OpenVINS input metadata must be an object")
        if not calibration and not profile_validation and input_metadata.get("mode") in {"calibration", "profile_validation"}:
            raise VIOError("provisional calibration input cannot run as admitted metric VIO")
        if "short_session" in input_metadata or profile_validation:
            if calibration:
                raise VIOError("short-profile input cannot enable online calibration")
            short = _verify_short_input(capture_dir, input_metadata, settings, validation=profile_validation)
            if runtime_quality_check is not None and not callable(runtime_quality_check):
                raise VIOError("runtime quality check must be callable")
            command.extend(["--short-session-consumer", SHORT_CONSUMER_VERSION,
                            "--profile-id", short["profile_id"], "--frame-mapping", str(capture_dir / "source_frame_mapping.csv")])
        for key, option in (
            ("capture_id", "--capture-id"),
            ("camera_sensor_id", "--camera-sensor-id"),
            ("time_domain", "--time-domain"),
        ):
            value = input_metadata.get(key)
            if isinstance(value, str) and value:
                command.extend([option, value])
        preprocessing = input_metadata.get("camera_preprocessing")
        if isinstance(preprocessing, Mapping) and preprocessing.get("rectified") is True:
            mask = preprocessing.get("invalid_ray_mask")
            if not isinstance(mask, Mapping) or mask.get("path") != "cam0/rectification_mask.png":
                raise VIOError("rectified OpenVINS input is missing its required invalid-ray mask")
            mask_path = capture_dir / "cam0" / "rectification_mask.png"
            try:
                mask_bytes = mask_path.read_bytes()
            except OSError as exc:
                raise VIOError("rectified OpenVINS input mask is unreadable") from exc
            if hashlib.sha256(mask_bytes).hexdigest() != mask.get("sha256"):
                raise VIOError("rectified OpenVINS input mask does not match its provenance")
            required_mask = {
                "applied": True,
                "invalid_pixel_count": mask.get("invalid_pixel_count"),
                "resolution_px": preprocessing["estimator_camera"]["resolution_px"],
            }
            # Older bridges reject this explicit option rather than running
            # rectified images while silently admitting their invalid borders.
            command.extend(["--camera-mask", str(mask_path)])
    if profile_validation:
        if short is None:
            raise VIOError("profile validation requires materialized short-profile input")
        command.extend(["--mode", "profile_validation"])
    if calibration:
        if input_metadata.get("schema") != "noesis.phone_capture.openvins_calibration_input.v1" or input_metadata.get("mode") != "calibration" or input_metadata.get("accepted_for_metric_vio") is not False:
            raise VIOError("online calibration requires materialized provisional input")
        raw_prior = input_metadata.get("calibration_prior")
        if not isinstance(raw_prior, Mapping):
            raise VIOError("online calibration input has no bound prior")
        calibration_prior = validate_vio_calibration_prior(raw_prior, str(input_metadata.get("capture_id") or ""))
        if hashlib.sha256(settings.config.read_bytes()).hexdigest() != input_metadata.get("estimator_config_sha256"):
            raise VIOError("online calibration config differs from its materialized input")
        command.extend(["--mode", "calibration"])
        for key, option in (
            ("rotation_std_rad", "--calibration-rotation-std-rad"),
            ("translation_std_m", "--calibration-translation-std-m"),
            ("time_offset_std_s", "--calibration-time-offset-std-s"),
        ):
            command.extend([option, str(calibration_prior["uncertainty"][key])])
    progress(0.08, "Running OpenVINS on timestamped camera and IMU data")
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=str(capture_dir),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                **({"env": {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}} if short else {}),
            )
            try:
                return_code = process.wait(timeout=settings.timeout_s)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.wait(timeout=10)
                raise VIOError(f"OpenVINS timed out after {settings.timeout_s}s") from exc
    except OSError as exc:
        raise VIOError(f"OpenVINS process could not start: {exc}") from exc
    stdout_tail = stdout_path.read_text(encoding="utf-8", errors="replace")[-2000:]
    stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
    if return_code != 0:
        detail = (stderr_tail or stdout_tail or "unknown OpenVINS error").strip()
        raise VIOError(f"OpenVINS failed with exit code {return_code}: {detail}")
    progress(0.90, "Validating OpenVINS trajectory and reset boundaries")
    result = _read_result(result_path, calibration=calibration, profile_validation=profile_validation,
                          pending_short_quality=short is not None and not profile_validation)
    if required_mask is not None and result.get("camera_mask") != required_mask:
        raise VIOError("OpenVINS did not confirm the required rectification mask")
    if calibration_prior is not None:
        initial = result["initial_calibration"]
        if not np.allclose(initial["T_imu_camera"], calibration_prior["extrinsics"]["T_imu_camera"], atol=1e-9, rtol=0) or not math.isclose(initial["imu_to_camera_offset_ns"], calibration_prior["clocks"]["imu_to_camera_offset_ns"], abs_tol=1, rel_tol=0):
            raise VIOError("OpenVINS initial calibration differs from the supplied prior")
        sigma = calibration_prior["uncertainty"]
        expected_covariance = np.diag([sigma["rotation_std_rad"] ** 2] * 3 + [sigma["translation_std_m"] ** 2] * 3 + [sigma["time_offset_std_s"] ** 2])
        if not np.allclose(initial["covariance"], expected_covariance, atol=1e-12, rtol=1e-8):
            raise VIOError("OpenVINS did not apply the supplied calibration uncertainty")
        for key in ("capture_id", "camera_sensor_id", "time_domain"):
            if result["frame"].get(key) != input_metadata.get(key):
                raise VIOError("online calibration output changed its source identity")
        identities = {row["capture_time_ns"]: row["source_frame_index"] for row in input_metadata["frame_mapping"]}
        if any(identities.get(row["capture_time_ns"]) != row.get("source_frame_index") for row in result["poses"]):
            raise VIOError("online calibration output changed its camera-frame mapping")
        result["calibration_prior"] = calibration_prior
        result["camera_preprocessing"] = input_metadata.get("camera_preprocessing")
    if short is not None:
        confirmation = result.get("short_session_consumer")
        if (not isinstance(confirmation, Mapping) or confirmation.get("profile_id") != short["profile_id"]
                or confirmation.get("consumer_version") != SHORT_CONSUMER_VERSION):
            raise VIOError("OpenVINS did not confirm the selected short-profile consumer")
        for key in ("capture_id", "camera_sensor_id", "time_domain"):
            if result["frame"].get(key) != input_metadata.get(key):
                raise VIOError("short-profile output changed its source identity")
        identities = {row["capture_time_ns"]: (row["source_frame_index"], row["pose_time_ns"]) for row in input_metadata["frame_mapping"]}
        if any(identities.get(row["capture_time_ns"]) != (row.get("source_frame_index"), row.get("pose_time_ns"))
               or row.get("prepared_frame_id") != f"{input_metadata['capture_id']}:source:{row.get('source_frame_index')}" for row in result["poses"]):
            raise VIOError("short-profile output changed the source/pose timestamp mapping")
        fixed = result.get("fixed_calibration", {})
        expected = input_metadata["fixed_calibration"]
        if (np.shape(fixed.get("T_imu_camera")) != (4, 4)
                or not np.allclose(fixed["T_imu_camera"], expected["T_imu_camera"], atol=1e-9, rtol=0)
                or fixed.get("imu_to_camera_offset_ns") != expected["imu_to_camera_offset_ns"]):
            raise VIOError("short-profile native fixed calibration differs from its supplied model")
        _verify_short_input(capture_dir, input_metadata, settings, validation=profile_validation)
        result.update(short_session=short, camera_preprocessing=input_metadata["camera_preprocessing"],
                      imu_preprocessing=input_metadata["imu_preprocessing"])
        if not profile_validation:
            direct_quality = _short_runtime_quality(result, input_metadata)
            if not direct_quality["accepted"]:
                raise VIOError("short-profile direct runtime quality failed: coverage, gap or reset")
            # The ordinary API is safe without caller plumbing: always use
            # the parent's versioned envelope/trajectory policy by default.
            # An explicit callback can supply additional caller-owned evidence.
            from .motion_profile import validate_short_walk_runtime, MotionProfileError
            try:
                quality = validate_short_walk_runtime(result, input_metadata)
                extra_quality = runtime_quality_check(result, input_metadata) if runtime_quality_check else None
            except MotionProfileError as exc:
                raise VIOError("short-profile runtime quality input is invalid") from exc
            if not isinstance(quality, Mapping) or quality.get("accepted") is not True:
                raise VIOError("short-profile runtime quality control rejected this walk")
            if extra_quality is not None:
                if not isinstance(extra_quality, Mapping) or extra_quality.get("accepted") is not True:
                    raise VIOError("additional short-profile runtime quality control rejected this walk")
                result["additional_runtime_quality_control"] = dict(extra_quality)
            result.update(runtime_quality_control=dict(quality), direct_runtime_quality=direct_quality,
                          accepted_for_metric_vio=True, status="completed")
            validate_vio_result(result)
    result["command"] = command
    result["stdout_tail"] = stdout_tail
    result["stderr_tail"] = stderr_tail
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    progress(1.0, f"OpenVINS produced {len(result['poses'])} camera poses")
    return result


__all__ = [
    "VIOError",
    "VIO_SCHEMA",
    "VIO_CALIBRATION_SCHEMA",
    "VIO_PROFILE_VALIDATION_SCHEMA",
    "VIO_CALIBRATION_PRIOR_SCHEMA",
    "VIOSettings",
    "materialize_openvins_input",
    "materialize_openvins_calibration_input",
    "materialize_openvins_profile_validation_input",
    "run_openvins",
    "run_openvins_calibration",
    "run_openvins_profile_validation",
    "validate_vio_input",
    "validate_vio_result",
    "validate_vio_calibration_prior",
    "validate_vio_calibration_result",
    "validate_vio_profile_validation_result",
]
