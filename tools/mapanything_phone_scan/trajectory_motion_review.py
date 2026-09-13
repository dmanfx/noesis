"""Review a saved phone reconstruction against native IMU, without admission.

The provider manifest selects the trajectory and its exact prepared RGB views.
Relative rotation and 3D speed do not need a room/world alignment. This command
does not infer person heading, integrate inertial position, run a model, update
a saved scan, or turn a diagnostic rotation candidate into calibration.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import spearmanr

from .prepared_frame_identity import prepared_frame_identity


SCHEMA = "noesis.phone_scan.trajectory_motion_review.v1"
POSE_CONVENTION = "opencv_cam2world_x_right_y_down_z_forward"
MAX_BYTES = 128 * 1024 * 1024
MAX_ROWS = 200_000
MAX_VIEWS = 4096
MAX_DURATION_S = 1800.0
ACTIVITY_HZ = 100.0
METRIC_FRAMES = {
    "mapanything_metric_world_unaligned_to_noesis",
    "mapanything_metric_world_window_registered_unaligned_to_noesis",
    "da3_metric_world_unaligned_to_noesis",
    "da3_metric_world_window_registered_unaligned_to_noesis",
}


class TrajectoryMotionReviewError(ValueError):
    """Input evidence cannot support the requested offline review."""


@dataclass(frozen=True)
class TrajectoryMotionReviewSettings:
    heldout_start_s: float
    allow_partial: bool = False
    max_sample_gap_s: float = 0.05
    max_pose_interval_s: float = 2.0
    sensitivity_offsets_s: tuple[float, ...] = (-0.25, -0.1, 0.0, 0.1, 0.25)
    fit_rotation_candidate: bool = False

    def validate(self) -> None:
        if not math.isfinite(self.heldout_start_s) or self.heldout_start_s < 0:
            raise TrajectoryMotionReviewError("heldout start must be finite and nonnegative")
        if not 0.001 <= self.max_sample_gap_s <= 0.1:
            raise TrajectoryMotionReviewError("maximum IMU gap must be between 1 and 100 ms")
        if not 0.01 <= self.max_pose_interval_s <= 10:
            raise TrajectoryMotionReviewError("maximum pose interval must be between 0.01 and 10 seconds")
        offsets = self.sensitivity_offsets_s
        if (not 1 <= len(offsets) <= 11 or len(set(offsets)) != len(offsets)
                or 0.0 not in offsets or any(not math.isfinite(x) or abs(x) > 1 for x in offsets)):
            raise TrajectoryMotionReviewError("sensitivity needs 1–11 unique offsets within ±1 s, including zero")


def _read(path: Path) -> bytes:
    with path.open("rb") as handle:
        value = handle.read(MAX_BYTES + 1)
    if len(value) > MAX_BYTES:
        raise TrajectoryMotionReviewError(f"{path.name} exceeds the evidence size limit")
    return value


def _json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read(path)
    def invalid(value: str) -> None:
        raise TrajectoryMotionReviewError(f"{path.name} contains nonfinite JSON: {value}")
    value = json.loads(raw, parse_constant=invalid)
    if not isinstance(value, dict):
        raise TrajectoryMotionReviewError(f"{path.name} must contain a JSON object")
    return value, raw


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int, np.integer)):
        raise TrajectoryMotionReviewError(f"{label} must be an exact integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrajectoryMotionReviewError(f"{label} must be an exact integer") from exc
    if result < 0 or result > np.iinfo(np.int64).max:
        raise TrajectoryMotionReviewError(f"{label} is outside the native timestamp/index range")
    return result


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrajectoryMotionReviewError(f"{label} must be an object")
    return value


def _inside(root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise TrajectoryMotionReviewError(f"{label} needs a relative artifact path")
    rel = Path(relative)
    path = (root / rel).resolve()
    if rel.is_absolute() or ".." in rel.parts or not path.is_relative_to(root.resolve()):
        raise TrajectoryMotionReviewError(f"{label} escapes its declared artifact root")
    return path


def _evidence(path: Path, raw: bytes) -> dict[str, Any]:
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _csv(path: Path, required: set[str]) -> tuple[list[dict[str, str]], bytes]:
    raw = _read(path)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    fields = reader.fieldnames or []
    if len(set(fields)) != len(fields) or not required.issubset(fields):
        raise TrajectoryMotionReviewError(f"{path.name} has missing or duplicate CSV columns")
    rows = []
    for row in reader:
        if len(rows) >= MAX_ROWS or None in row or any(v is None for v in row.values()):
            raise TrajectoryMotionReviewError(f"{path.name} has malformed or excessive rows")
        rows.append(row)
    if len(rows) < 2:
        raise TrajectoryMotionReviewError(f"{path.name} needs at least two rows")
    return rows, raw


def _validate_poses(value: Any, count: int) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.shape != (count, 4, 4) or not np.isfinite(poses).all():
        raise TrajectoryMotionReviewError("trajectory must contain one finite 4×4 pose per provider view")
    rotations = poses[:, :3, :3]
    if (not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=2e-4, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=2e-4, rtol=0)):
        raise TrajectoryMotionReviewError("trajectory rotations must be proper SO(3), with homogeneous bottom rows")
    return poses


def _load_provider(scan: Path, manifest_path: Path, allow_partial: bool) -> dict[str, Any]:
    prepared_path = scan / "prepared_frames_manifest.json"
    prepared, prepared_raw = _json(prepared_path)
    manifest, manifest_raw = _json(manifest_path)
    frames, outputs = prepared.get("frames"), manifest.get("frames")
    if (not isinstance(frames, list) or not 2 <= len(frames) <= MAX_VIEWS
            or not isinstance(outputs, list) or not 2 <= len(outputs) <= len(frames)
            or _integer(manifest.get("view_count"), "provider view_count") != len(outputs)):
        raise TrajectoryMotionReviewError("prepared/provider frame counts are inconsistent or unsupported")
    if manifest.get("anchor_view_index") is not None:
        raise TrajectoryMotionReviewError("phone motion review cannot include a fixed-camera anchor")
    if manifest.get("coordinate_frame") not in METRIC_FRAMES:
        raise TrajectoryMotionReviewError("provider must declare its supported unaligned metric coordinate frame")
    if manifest.get("pose_convention") != POSE_CONVENTION:
        raise TrajectoryMotionReviewError("provider must explicitly declare the OpenCV camera-to-world convention")
    provider = manifest.get("provider")
    if provider is None and manifest.get("schema") in {"noesis.mapanything.phone_scan.outputs.v1", "noesis.mapanything.phone_scan.outputs.v2"}:
        provider = "mapanything"
    model_id = manifest.get("model_id") or _object(manifest.get("model", {}), "provider model").get("id")
    if (provider not in {"mapanything", "da3"} or not manifest["coordinate_frame"].startswith(provider + "_metric_world_")
            or not isinstance(model_id, str) or not model_id.strip()):
        raise TrajectoryMotionReviewError("provider/model identity is missing or disagrees with its coordinate frame")
    artifact = _object(manifest.get("artifacts", {}), "provider artifacts").get("trajectory_json")
    if not isinstance(artifact, str) or not artifact.startswith("outputs/"):
        raise TrajectoryMotionReviewError("provider must declare its outputs/ trajectory JSON artifact")
    trajectory_path = _inside(manifest_path.parent, artifact[len("outputs/"):], "trajectory")
    trajectory, trajectory_raw = _json(trajectory_path)
    if trajectory.get("pose_convention") != POSE_CONVENTION:
        raise TrajectoryMotionReviewError("trajectory convention is missing or disagrees with its provider")
    if trajectory.get("coordinate_frame", manifest["coordinate_frame"]) != manifest["coordinate_frame"]:
        raise TrajectoryMotionReviewError("trajectory coordinate frame disagrees with its provider")
    fields = [key for key in ("camera_poses", "camera_to_world") if key in trajectory]
    if len(fields) != 1:
        raise TrajectoryMotionReviewError("trajectory must declare exactly one supported pose array")
    poses = _validate_poses(trajectory[fields[0]], len(outputs))
    by_path = {}
    for index, row in enumerate(frames):
        if (not isinstance(row, dict) or _integer(row.get("index"), "prepared index") != index
                or row.get("frame_id") != prepared_frame_identity(index, row.get("sha256", ""))):
            raise TrajectoryMotionReviewError("prepared frame index/hash identity is inconsistent")
        source = row.get("frame")
        if not isinstance(source, str) or source in by_path:
            raise TrajectoryMotionReviewError("prepared source paths must be unique")
        by_path[source] = index
    selected, windows = [], []
    window_count = _integer(manifest.get("window_count", 1), "window_count")
    if not 1 <= window_count <= len(outputs):
        raise TrajectoryMotionReviewError("provider window_count is invalid")
    for index, row in enumerate(outputs):
        if (not isinstance(row, dict) or _integer(row.get("index"), "provider index") != index
                or row.get("fixed_camera_anchor") is not False or row.get("source_frame") not in by_path):
            raise TrajectoryMotionReviewError("provider row lacks exact ordered phone source identity")
        prepared_index = by_path[row["source_frame"]]
        if selected and prepared_index <= selected[-1]:
            raise TrajectoryMotionReviewError("provider prepared-view order is repeated or reversed")
        source = frames[prepared_index]
        if (isinstance(row.get("timestamp_s"), bool) or not isinstance(row.get("timestamp_s"), (float, int))
                or not math.isfinite(row["timestamp_s"])
                or row["timestamp_s"] != source.get("timestamp_s")):
            raise TrajectoryMotionReviewError("provider source timestamp disagrees with the prepared view")
        image = _inside(scan, source["frame"], "prepared RGB")
        if hashlib.sha256(_read(image)).hexdigest() != source["sha256"]:
            raise TrajectoryMotionReviewError("prepared RGB bytes do not match their declared SHA-256")
        window = _integer(row.get("adaptive_window_index", 0 if window_count == 1 else None), "adaptive_window_index")
        if window >= window_count or (windows and window < windows[-1]):
            raise TrajectoryMotionReviewError("provider window identities are inconsistent")
        windows.append(window)
        selected.append(prepared_index)
    omitted = sorted(set(range(len(frames))) - set(selected))
    if omitted and not allow_partial:
        raise TrajectoryMotionReviewError("provider covers a subset; explicitly allow a partial review")
    partial = _object(manifest.get("partial_reconstruction", {}), "partial reconstruction")
    if (partial.get("parent_prepared_manifest_sha256") is not None
            and partial["parent_prepared_manifest_sha256"] != hashlib.sha256(prepared_raw).hexdigest()):
        raise TrajectoryMotionReviewError("partial provider parent-manifest hash disagrees with the scan")
    for field, expected in (("included_global_indices", selected), ("omitted_global_indices", omitted), ("parent_view_count", len(frames))):
        if field in partial and partial[field] != expected:
            raise TrajectoryMotionReviewError("declared partial reconstruction scope disagrees with actual prepared identities")
    return {"prepared": prepared, "manifest": manifest, "poses": poses, "provider": provider, "model_id": model_id,
            "selected": selected, "windows": np.asarray(windows),
            "scope": {"status": "partial" if omitted else "full_prepared_view_set",
                      "parent_view_count": len(frames), "review_view_count": len(outputs),
                      "included_prepared_indices": selected, "omitted_prepared_indices": omitted},
            "evidence": {"prepared_manifest": _evidence(prepared_path, prepared_raw),
                         "provider_manifest": _evidence(manifest_path, manifest_raw),
                         "provider_trajectory": _evidence(trajectory_path, trajectory_raw)}}


@dataclass
class SensorStream:
    timestamps_ns: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    bias_applied: bool

    def segments(self, max_gap_ns: int) -> list[tuple[int, int]]:
        segments, start = [], None
        for index, valid in enumerate(self.valid):
            if start is not None and (not valid or self.timestamps_ns[index] - self.timestamps_ns[index - 1] > max_gap_ns):
                if index - start >= 2:
                    segments.append((start, index))
                start = None
            if valid and start is None:
                start = index
        if start is not None and len(self.valid) - start >= 2:
            segments.append((start, len(self.valid)))
        return segments


def _load_sensor(path: Path, descriptor: dict[str, Any]) -> tuple[SensorStream, bytes]:
    rows, raw = _csv(path, {"timestamp_ns", "x", "y", "z", "bias_x", "bias_y", "bias_z", "accuracy"})
    ts = np.asarray([_integer(r["timestamp_ns"], "sensor timestamp_ns") for r in rows], dtype=np.int64)
    if np.any(np.diff(ts) <= 0) or (int(ts[-1]) - int(ts[0])) / 1e9 > MAX_DURATION_S:
        raise TrajectoryMotionReviewError("sensor timestamps are not strictly increasing or exceed the duration limit")
    values = np.asarray([[float(row[k]) for k in ("x", "y", "z")] for row in rows])
    bias = np.asarray([[float(row["bias_" + k]) for k in ("x", "y", "z")] for row in rows])
    if not np.isfinite(values).all() or not np.isfinite(bias).all():
        raise TrajectoryMotionReviewError("sensor vectors and bias estimates must be finite")
    bias_applied = descriptor.get("uncalibrated") is True and descriptor.get("bias_fields_are_sensor_estimates") is True
    if not bias_applied and np.any(bias != 0):
        raise TrajectoryMotionReviewError("nonzero sensor bias lacks explicit uncalibrated-estimate semantics")
    if bias_applied:
        values = values - bias
    with np.errstate(over="ignore"):
        if not np.isfinite(np.linalg.norm(values, axis=1)).all():
            raise TrajectoryMotionReviewError("sensor magnitudes overflow")
    accuracy = np.asarray([_integer(row["accuracy"], "sensor accuracy") for row in rows])
    if np.any(accuracy > 3):
        raise TrajectoryMotionReviewError("sensor accuracy must be an Android status from 0 to 3")
    for field, actual in (("sample_count", len(rows)), ("first_timestamp_ns", int(ts[0])), ("last_timestamp_ns", int(ts[-1]))):
        if _integer(descriptor.get(field), f"sensor {field}") != actual:
            raise TrajectoryMotionReviewError(f"sensor {field} disagrees with its native capture report")
    return SensorStream(ts, values, accuracy > 0, bias_applied), raw


def _load_native(scan: Path, source: dict[str, Any]) -> dict[str, Any]:
    capture = (scan / "capture").resolve()
    report_path = capture / "capture_import.json"
    report, report_raw = _json(report_path)
    native = _object(report.get("android_capture", {}), "native capture report")
    manifest = _object(report.get("manifest", {}), "native capture manifest")
    clocks = _object(manifest.get("clocks", {}), "native clocks")
    _object(manifest.get("calibration", {}), "native calibration declaration")
    if (native.get("camera_acquisition_timestamp_verified") is not True
            or clocks.get("camera_domain") != "android.elapsedRealtimeNanos"
            or clocks.get("imu_domain") != "android.elapsedRealtimeNanos"
            or native.get("association_method") != "encoder_pts_us_equals_sensor_timestamp_ns_div_1000"):
        raise TrajectoryMotionReviewError("verified native Camera2/IMU acquisition association is required")
    descriptors = _object(_object(native.get("recorder", {}), "native recorder").get("sensors", {}), "native sensors")
    imu = _object(manifest.get("imu", {}), "native IMU declaration")
    evidence = {"capture_import": _evidence(report_path, report_raw)}
    streams = {}
    for key, label, unit in (("gyro", "gyroscope", "rad/s"), ("accel", "accelerometer", "m/s^2")):
        descriptor = _object(descriptors.get(label, {}), label)
        if (descriptor.get("units") != unit or imu.get(key + "_unit") != unit
                or descriptor.get("axes") != "android_device_x_right_y_up_z_out_of_screen"
                or descriptor.get("timestamp_source") != "android_elapsed_realtime_ns"):
            raise TrajectoryMotionReviewError(f"{label} units, axes, or timestamp source are not verified")
        declared = imu.get(key + "_path")
        if declared != descriptor.get("file"):
            raise TrajectoryMotionReviewError(f"{label} artifact identity is inconsistent")
        path = _inside(capture, declared, label)
        streams[key], raw = _load_sensor(path, descriptor)
        evidence[key] = _evidence(path, raw)
    time_path = _inside(capture, _object(manifest.get("video", {}), "native video").get("frame_timestamps_path"), "native frame times")
    associations, raw = _csv(time_path, {"encoded_index", "encoded_pts_us", "frame_number", "timestamp_ns"})
    evidence["frame_times"] = _evidence(time_path, raw)
    paths = _object(native.get("evidence_paths", {}), "native evidence paths")
    encoder_path = _inside(capture, paths.get("encoder_pts_path"), "encoder times")
    encoders, raw = _csv(encoder_path, {"encoded_index", "encoded_pts_us"})
    evidence["encoder_times"] = _evidence(encoder_path, raw)
    camera_path = _inside(capture, paths.get("camera_results_path"), "camera results")
    raw = _read(camera_path)
    evidence["camera_results"] = _evidence(camera_path, raw)
    camera = {}
    camera_numbers = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = _object(json.loads(line), "Camera2 result")
        stamp = _integer(row.get("sensor_timestamp_ns"), "Camera2 timestamp")
        number = _integer(row.get("frame_number"), "Camera2 frame number")
        if stamp in camera or number in camera_numbers or len(camera) >= MAX_ROWS:
            raise TrajectoryMotionReviewError("Camera2 results contain duplicate or excessive identities")
        camera[stamp] = row
        camera_numbers.add(number)
    if (len(encoders) != len(associations)
            or _integer(native.get("association_row_count"), "native association count") != len(associations)):
        raise TrajectoryMotionReviewError("native encoder/frame association counts disagree")
    stamps = []
    for index, (row, encoded) in enumerate(zip(associations, encoders)):
        stamp = _integer(row["timestamp_ns"], "associated timestamp_ns")
        pts = _integer(row["encoded_pts_us"], "associated encoded_pts_us")
        if (_integer(row["encoded_index"], "associated index") != index
                or _integer(encoded["encoded_index"], "encoder index") != index
                or _integer(encoded["encoded_pts_us"], "encoder PTS") != pts or pts != stamp // 1000
                or stamp not in camera or _integer(row["frame_number"], "associated frame number") != camera[stamp]["frame_number"]
                or (stamps and stamp <= stamps[-1])):
            raise TrajectoryMotionReviewError("native camera/encoder association no longer matches acquisition evidence")
        stamps.append(stamp)
    origin = stamps[0]
    if (stamps[-1] - origin) / 1e9 > MAX_DURATION_S:
        raise TrajectoryMotionReviewError("camera acquisition duration exceeds the analysis limit")
    selected_times, identities = [], []
    for output_index, prepared_index in enumerate(source["selected"]):
        frame = source["prepared"]["frames"][prepared_index]
        index = _integer(frame.get("source_frame_index"), "prepared encoded source index")
        stamp = _integer(frame.get("capture_time_ns"), "prepared capture_time_ns")
        if (index >= len(stamps) or stamp != stamps[index]
                or frame.get("timestamp_source") != "android_camera2_sensor_timestamp_exact_encoder_association"
                or abs(float(frame["timestamp_s"]) - (stamp - origin) / 1e9) > 1e-9):
            raise TrajectoryMotionReviewError("prepared view does not match its exact native acquisition time")
        selected_times.append(stamp)
        identities.append({"provider_index": output_index, "prepared_index": prepared_index,
                           "prepared_frame_id": frame["frame_id"], "source_frame": frame["frame"],
                           "rgb_sha256": frame["sha256"], "encoded_source_index": index,
                           "capture_time_ns": stamp, "phone_time_s": (stamp - origin) / 1e9})
    if np.any(np.diff(np.asarray(selected_times, dtype=np.int64)) <= 0):
        raise TrajectoryMotionReviewError("selected native acquisition times must be strictly increasing")
    return {"streams": streams, "report": report, "evidence": evidence, "capture_root": capture,
            "origin_ns": origin, "times_ns": np.asarray(selected_times, dtype=np.int64),
            "video_end_s": (stamps[-1] - origin) / 1e9, "identities": identities}


class GyroOrientations:
    """Independent SO(3) integrations per supported sensor segment; no gap fill."""

    def __init__(self, stream: SensorStream, origin_ns: int, max_gap_ns: int):
        self.origin_ns = origin_ns
        self.segments = []
        for start, end in stream.segments(max_gap_ns):
            timestamps = stream.timestamps_ns[start:end]
            times = (timestamps - origin_ns) / 1e9
            omega = stream.values[start:end]
            increments = (omega[1:] + omega[:-1]) * (0.5 * np.diff(times))[:, None]
            if np.any(np.linalg.norm(increments, axis=1) >= math.pi):
                raise TrajectoryMotionReviewError("gyro samples imply ambiguous ≥180-degree intersample increments")
            q = Rotation.identity()
            quaternions = [q.as_quat()]
            for delta in Rotation.from_rotvec(increments):
                q = q * delta
                quaternions.append(q.as_quat())
            self.segments.append((float(times[0]), float(times[-1]), Slerp(times, Rotation.from_quat(quaternions))))

    def relative(self, times_ns: np.ndarray, offset_s: float = 0) -> tuple[np.ndarray, np.ndarray]:
        times = (times_ns - self.origin_ns) / 1e9 + offset_s
        quaternions = np.full((len(times), 4), np.nan)
        segment_ids = np.full(len(times), -1)
        for index, (start, end, interpolate) in enumerate(self.segments):
            selected = (times >= start) & (times <= end)
            if np.any(selected):
                quaternions[selected] = interpolate(times[selected]).as_quat()
                segment_ids[selected] = index
        valid = (segment_ids[:-1] >= 0) & (segment_ids[:-1] == segment_ids[1:])
        vectors = np.full((len(times) - 1, 3), np.nan)
        if np.any(valid):
            vectors[valid] = (Rotation.from_quat(quaternions[:-1][valid]).inv()
                              * Rotation.from_quat(quaternions[1:][valid])).as_rotvec()
        return vectors, valid


def acceleration_activity(stream: SensorStream, origin_ns: int, query_s: np.ndarray,
                          max_gap_ns: int) -> np.ndarray:
    """0.7–3 Hz acceleration-norm RMS, independently filtered across gaps."""
    result = np.full(len(query_s), np.nan)
    sos = butter(3, [0.7, 3.0], btype="bandpass", fs=ACTIVITY_HZ, output="sos")
    for start, end in stream.segments(max_gap_ns):
        times = (stream.timestamps_ns[start:end] - origin_ns) / 1e9
        if times[-1] - times[0] < 3:
            continue
        # Keep one second clear of finite-filter edges and the centered RMS.
        eligible = (query_s >= times[0] + 1) & (query_s <= times[-1] - 1)
        if not np.any(eligible):
            continue
        grid = np.arange(math.ceil(times[0] * ACTIVITY_HZ), math.floor(times[-1] * ACTIVITY_HZ) + 1) / ACTIVITY_HZ
        if len(grid) > MAX_ROWS:
            raise TrajectoryMotionReviewError("acceleration resampling exceeds the bounded grid")
        # Interpolate vector samples first, matching the native diagnostic path.
        vector = np.column_stack([np.interp(grid, times, stream.values[start:end, axis]) for axis in range(3)])
        band = sosfiltfilt(sos, np.linalg.norm(vector, axis=1))
        activity = np.sqrt(np.maximum(0, uniform_filter1d(band * band, size=100, mode="nearest")))
        result[eligible] = np.interp(query_s[eligible], grid, activity)
    return result


def _correlation(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = mask & np.isfinite(first) & np.isfinite(second)
    count = int(np.count_nonzero(selected))
    if count < 5 or np.ptp(first[selected]) <= 1e-12 or np.ptp(second[selected]) <= 1e-12:
        return {"n": count, "spearman_rho": None, "reason": "insufficient_or_constant_samples"}
    return {"n": count, "spearman_rho": float(spearmanr(first[selected], second[selected]).statistic)}


def rotation_metrics(camera_vectors: np.ndarray, gyro_vectors: np.ndarray,
                     dt: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask = mask & np.isfinite(gyro_vectors).all(axis=1)
    camera = np.linalg.norm(camera_vectors, axis=1)
    gyro = np.linalg.norm(gyro_vectors, axis=1)
    result = _correlation(camera / dt, gyro / dt, mask)
    if result["n"]:
        error = np.rad2deg(camera[mask] - gyro[mask])
        result.update(absolute_increment_error_deg_median=float(np.median(abs(error))),
                      absolute_increment_error_deg_p80=float(np.percentile(abs(error), 80)),
                      absolute_increment_error_deg_p95=float(np.percentile(abs(error), 95)),
                      absolute_increment_error_deg_max=float(np.max(abs(error))),
                      signed_increment_error_deg_median=float(np.median(error)),
                      mean_abs_angular_rate_error_dps=float(np.mean(abs(error) / dt[mask])))
    return result


def rotation_candidate(camera_vectors: np.ndarray, gyro_vectors: np.ndarray,
                       train: np.ndarray, heldout: np.ndarray) -> dict[str, Any]:
    """Fit only training increments; refuse a numerically unobservable axis."""
    def excitation(mask: np.ndarray) -> dict[str, Any]:
        vectors = camera_vectors[mask]
        energy = vectors.T @ vectors
        eigenvalues = np.linalg.eigvalsh(energy)
        information = np.trace(energy) * np.eye(3) - energy
        info = np.linalg.eigvalsh(information)
        total = float(eigenvalues.sum())
        observable = bool(len(vectors) >= 6 and info[-1] > 1e-10 and info[0] > info[-1] * 1e-4)
        return {"n": len(vectors), "rotation_energy_fractions_ascending": (eigenvalues / total).tolist() if total else [0, 0, 0],
                "information_eigenvalues_ascending": info.tolist(),
                "information_condition_number": float(info[-1] / info[0]) if info[0] > 0 else None,
                "numerically_observable": observable}
    def fit(mask: np.ndarray) -> Rotation:
        return Rotation.align_vectors(gyro_vectors[mask], camera_vectors[mask])[0]
    def errors(rotation: Rotation, mask: np.ndarray) -> dict[str, Any]:
        if not np.any(mask):
            return {"n": 0}
        predicted = rotation * Rotation.from_rotvec(camera_vectors[mask]) * rotation.inv()
        values = np.rad2deg((predicted.inv() * Rotation.from_rotvec(gyro_vectors[mask])).magnitude())
        return {"n": len(values), "full_rotation_error_deg_median": float(np.median(values)),
                "full_rotation_error_deg_p80": float(np.percentile(values, 80)),
                "full_rotation_error_deg_p95": float(np.percentile(values, 95)),
                "full_rotation_error_deg_rms": float(np.sqrt(np.mean(values * values)))}
    result = {"status": "insufficient_rotational_excitation", "admitted_calibration": False,
              "transform_definition": "v_imu = X_camera_to_imu @ v_camera; B = X @ A @ X.T",
              "fit_method": "proper_SO3_rotation_vector_alignment_training_only",
              "training_excitation": excitation(train), "heldout_excitation": excitation(heldout)}
    if not result["training_excitation"]["numerically_observable"]:
        return result
    candidate = fit(train)
    result.update(status="diagnostic_candidate_only", camera_to_imu_rotation_row_major=candidate.as_matrix().tolist(),
                  determinant=float(np.linalg.det(candidate.as_matrix())),
                  training_residual=errors(candidate, train), heldout_residual=errors(candidate, heldout))
    if result["heldout_excitation"]["numerically_observable"]:
        independent = fit(heldout)
        result["independent_heldout_fit_difference_deg"] = float(np.rad2deg((candidate.inv() * independent).magnitude()))
    result["limitation"] = "Numerical excitation and small residuals do not establish precision; translation, noise, timing and stabilization remain unverified."
    return result


def review_trajectory_motion(scan_dir: Path, source_output_manifest: Path, output_dir: Path,
                             settings: TrajectoryMotionReviewSettings) -> dict[str, Any]:
    settings.validate()
    scan, manifest_path, output = scan_dir.resolve(), source_output_manifest.resolve(), output_dir.resolve()
    if output.exists() or output.is_relative_to(scan) or output.is_relative_to(manifest_path.parent):
        raise TrajectoryMotionReviewError("use a new output directory outside the scan and provider artifacts")
    source = _load_provider(scan, manifest_path, settings.allow_partial)
    native = _load_native(scan, source)
    if output.is_relative_to(native["capture_root"]):
        raise TrajectoryMotionReviewError("review output cannot modify the native capture directory")
    ts = native["times_ns"]
    times = (ts - native["origin_ns"]) / 1e9
    if not times[0] < settings.heldout_start_s < times[-1]:
        raise TrajectoryMotionReviewError("heldout start must lie inside the reviewed camera-time range")
    dt, midpoint = np.diff(times), (times[1:] + times[:-1]) / 2
    max_gap_ns = round(settings.max_sample_gap_s * 1e9)
    orientation = Rotation.from_matrix(source["poses"][:, :3, :3])
    camera_vectors = (orientation[:-1].inv() * orientation[1:]).as_rotvec()
    integrator = GyroOrientations(native["streams"]["gyro"], native["origin_ns"], max_gap_ns)
    gyro_vectors, gyro_valid = integrator.relative(ts)
    same_window = np.diff(source["windows"]) == 0
    consecutive = np.diff(source["selected"]) == 1
    pose_support = same_window & consecutive & (dt <= settings.max_pose_interval_s)
    supported = pose_support & gyro_valid
    train = times[1:] <= settings.heldout_start_s
    heldout = times[:-1] >= settings.heldout_start_s
    groups = {"all_supported": supported, "training": supported & train, "heldout": supported & heldout}
    rotations = {name: rotation_metrics(camera_vectors, gyro_vectors, dt, mask) for name, mask in groups.items()}
    sensitivity = []
    for offset in settings.sensitivity_offsets_s:
        shifted, shifted_valid = integrator.relative(ts, offset)
        common = supported & shifted_valid
        sensitivity.append({"additional_imu_offset_s": offset, "all_common_support": rotation_metrics(camera_vectors, shifted, dt, common),
                            "training": rotation_metrics(camera_vectors, shifted, dt, common & train),
                            "heldout": rotation_metrics(camera_vectors, shifted, dt, common & heldout)})
    activity = acceleration_activity(native["streams"]["accel"], native["origin_ns"], midpoint, max_gap_ns)
    with np.errstate(over="ignore", invalid="ignore"):
        speed = np.linalg.norm(np.diff(source["poses"][:, :3, 3], axis=0), axis=1) / dt
    if not np.isfinite(speed).all():
        raise TrajectoryMotionReviewError("camera displacement/speed is not finite")
    speed_metrics = {name: _correlation(speed, activity, pose_support & mask)
                     for name, mask in (("all_supported", np.ones(len(dt), dtype=bool)), ("training", train), ("heldout", heldout))}
    intervals = []
    for index in range(len(dt)):
        reasons = []
        if not same_window[index]: reasons.append("provider_window_transition")
        if not consecutive[index]: reasons.append("omitted_prepared_views")
        if dt[index] > settings.max_pose_interval_s: reasons.append("pose_interval_exceeds_review_limit")
        if not gyro_valid[index]: reasons.append("gyro_gap_unreliable_samples_or_missing_coverage")
        intervals.append({"start_provider_index": index, "end_provider_index": index + 1,
                          "start_prepared_index": source["selected"][index], "end_prepared_index": source["selected"][index + 1],
                          "start_capture_time_ns": int(ts[index]), "end_capture_time_ns": int(ts[index + 1]),
                          "start_phone_time_s": float(times[index]), "end_phone_time_s": float(times[index + 1]),
                          "split": "training" if train[index] else "heldout" if heldout[index] else "split_boundary",
                          "rotation_supported": bool(supported[index]), "exclusions": "|".join(reasons),
                          "camera_net_rotation_deg": float(np.rad2deg(np.linalg.norm(camera_vectors[index]))),
                          "gyro_net_rotation_deg": float(np.rad2deg(np.linalg.norm(gyro_vectors[index]))) if gyro_valid[index] else None,
                          "camera_speed_mps": float(speed[index]),
                          "speed_activity_supported": bool(pose_support[index] and np.isfinite(activity[index])),
                          "accel_activity_mps2": float(activity[index]) if np.isfinite(activity[index]) else None})
    calibration = (native["report"].get("manifest") or {}).get("calibration") or {}
    report = {"schema": SCHEMA, "status": "review_complete" if np.any(supported) else "insufficient_motion_support",
              "review_only": True, "updates_scan_state": False, "promotes_live_world": False,
              "metric_vio_admission_changed": False, "settings": asdict(settings), "scope": source["scope"],
              "source": {"provider": source["provider"], "model_id": source["model_id"], "coordinate_frame": source["manifest"]["coordinate_frame"],
                         "pose_convention": POSE_CONVENTION, "position_units": "provider_metric_meters", "room_alignment_required": False},
              "evidence": {**source["evidence"], **native["evidence"]}, "prepared_view_identities": native["identities"],
              "timing": {"phone_native_origin_ns": native["origin_ns"], "review_time_range_s": [float(times[0]), float(times[-1])],
                         "mapping": "(exact Camera2 acquisition_timestamp_ns - first_encoded_camera_timestamp_ns) / 1e9",
                         "applied_camera_imu_offset_s": 0.0, "offset_semantics": "nominal_native_clock_comparison_not_measured_offset",
                         "capture_declared_imu_to_camera_offset_ns": (native["report"].get("manifest") or {}).get("clocks", {}).get("imu_to_camera_offset_ns"),
                         "camera_encoder_association_rechecked": True, "video_not_redecoded": True},
              "calibration": {"imported_metric_vio_allowed": native["report"].get("metric_vio_allowed") is True,
                              "missing_or_unverified": [key for key in ("camera_intrinsics", "camera_distortion", "camera_to_imu_extrinsics", "imu_noise", "time_offset") if calibration.get(key) is not True]},
              "imu": {key: {"sample_count": len(stream.timestamps_ns), "unreliable_samples_excluded": int(np.count_nonzero(~stream.valid)),
                            "recorded_bias_estimates_subtracted": stream.bias_applied, "supported_segments": len(stream.segments(max_gap_ns))}
                      for key, stream in native["streams"].items()},
              "rotation_method": "SO3 midpoint gyro integration per supported segment; Slerp to exact camera times; compare magnitudes of relative rotations R_i^-1 R_j, invariant to a fixed camera/IMU rotation and global gauge.",
              "rotation": rotations, "timing_sensitivity": sensitivity,
              "speed_activity_method": "Adjacent camera 3D displacement / native interval, compared to offline zero-phase 0.7–3Hz accelerometer-norm 1s RMS, 100Hz interpolation; no filtering across IMU gaps and 1s segment-edge exclusion.",
              "speed_activity": speed_metrics,
              "limitations": [
                  "This is phone-camera motion, not person heading or a body/foot trajectory. Hand panning is shared camera/IMU motion.",
                  "Relative-angle agreement cannot establish axis-wise orientation, metric accuracy, or admitted extrinsics.",
                  "Acceleration activity includes gravity, bias and hand/lever-arm motion; it is not a velocity formula or inertial position/height estimate.",
                  "Camera/IMU exposure offset, rolling shutter and stabilization may affect the comparison; timing sensitivity is not calibration.",
                  "Temporal heldout samples remain one capture; overlapping filters and motion samples are correlated. No statistical confidence or generalization claim.",
                  "Prepared-view selection may already use IMU motion quality; independent pose/gyro evidence is conditional on those selected frames.",
                  "Provider geometry is model-derived. A partial review does not repair omitted views or relax reconstruction/alignment gates.",
                  "The activity filter is non-causal offline analysis; live use needs separate delay and quality validation.",
              ]}
    if settings.fit_rotation_candidate:
        report["rotation_candidate"] = rotation_candidate(camera_vectors, gyro_vectors, groups["training"], groups["heldout"])
    # Evidence errors above never create or modify the scan. Outputs are new and
    # deliberately separate from provider, capture, calibration and scan state.
    output.mkdir(parents=True, exist_ok=False)
    with (output / "intervals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(intervals[0]))
        writer.writeheader()
        writer.writerows(intervals)
    (output / "motion_review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", required=True, type=Path)
    parser.add_argument("--source-output-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--heldout-start-s", required=True, type=float)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--max-sample-gap-s", type=float, default=0.05)
    parser.add_argument("--max-pose-interval-s", type=float, default=2.0)
    parser.add_argument("--fit-rotation-candidate", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = review_trajectory_motion(args.scan_dir, args.source_output_manifest, args.output_dir,
            TrajectoryMotionReviewSettings(heldout_start_s=args.heldout_start_s, allow_partial=args.allow_partial,
                max_sample_gap_s=args.max_sample_gap_s, max_pose_interval_s=args.max_pose_interval_s,
                fit_rotation_candidate=args.fit_rotation_candidate))
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(f"Trajectory motion review rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "scope": result["scope"]["status"],
                      "rotation": result["rotation"], "output_dir": str(args.output_dir.resolve())}, allow_nan=False))
    return 0 if result["status"] == "review_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
