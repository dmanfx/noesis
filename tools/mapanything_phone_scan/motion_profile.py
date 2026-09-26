"""Reusable, evidence-bound short-walk profiles with fixed-consumer validation.

This is an offline camera-trajectory capability, not world/PCF admission or a
claim that a short stationary sample measured long-term sensor drift.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import threading
import math
import re
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .calibration_jobs import CalibrationJobError, _now, _read, _write
from .motion_preprocessing import (
    CONSUMER_VERSION, MAXIMUM_WALK_SECONDS, PROFILE_SCHEMA, MotionProfileError,
    bind_profile_to_capture, checked_corrections, corrected_noise, json_sha,
)
from .roomwalk_calibration import (
    _load_camera_reference, _load_noise_reference, _member, _read_json, _sha,
    camera_profile,
)

# Versioned application acceptance policy, not metrology or room accuracy.
VALIDATION_POLICY = {
    "version": "roomwalk.fixed_profile_board_check.v1",
    "minimum_heldout_seconds": 5.0, "minimum_heldout_poses": 20,
    "minimum_heldout_coverage": 0.8, "maximum_pose_gap_s": 0.2,
    "minimum_translation_extent_m": 0.15,
    "maximum_position_rmse_m": 0.10, "maximum_position_p95_m": 0.20,
    "maximum_orientation_rmse_deg": 3.0, "maximum_orientation_p95_deg": 6.0,
    "maximum_scale_error_fraction": 0.05,
}

WALK_SANITY_POLICY = {
    "version": "roomwalk.short_walk_consumer_sanity.v1",
    "maximum_duration_s": MAXIMUM_WALK_SECONDS + 2,
    "maximum_camera_speed_m_s": 5.0,
    "maximum_camera_angular_speed_rad_s": 5.0,
}


def validate_short_walk_runtime(result: Mapping[str, Any], input_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Last consumer check, in addition to native coverage/gap/reset validation.

    This rejects out-of-scope or grossly divergent camera trajectories. It is
    not a camera-position accuracy estimate or replacement for static alignment.
    """
    short = input_metadata.get("short_session") or {}
    reasons = []
    if (short.get("consumer_version") != CONSUMER_VERSION or short.get("validation_run") is not False
            or not re.fullmatch(r"[a-f0-9]{64}", str(short.get("profile_validation_sha256", "")))):
        reasons.append("checked_short_profile_provenance_missing")
    observed = short.get("motion_envelope") or {}
    checked = short.get("validated_motion_envelope") or {}
    for key in ("maximum_rotation_during_exposure_rad", "maximum_rotation_during_readout_rad"):
        got, limit = observed.get(key), checked.get(key)
        if (type(got) not in (int, float) or type(limit) not in (int, float)
                or not math.isfinite(got) or not math.isfinite(limit) or not 0 <= got <= limit or limit <= 0):
            reasons.append("recorded_motion_outside_checked_profile")
            break
    mapping = input_metadata.get("frame_mapping") or []
    poses = result.get("poses") or []
    if len(mapping) < 2 or len(poses) < 2:
        raise MotionProfileError("The short walk has insufficient timestamped camera poses")
    duration = (mapping[-1]["pose_time_ns"] - mapping[0]["pose_time_ns"]) * 1e-9
    if not 0 < duration <= WALK_SANITY_POLICY["maximum_duration_s"]:
        reasons.append("walk_exceeds_short_profile_duration")
    matrices = np.stack([_transform(r["T_vio_world_camera"]) for r in poses])
    dt = np.array([b["pose_time_ns"] - a["pose_time_ns"] for a, b in zip(poses, poses[1:])], float) * 1e-9
    if not np.all(dt > 0):
        raise MotionProfileError("Short-walk pose times must increase")
    speeds = np.linalg.norm(np.diff(matrices[:, :3, 3], axis=0), axis=1) / dt
    rotations = matrices[1:, :3, :3] @ np.transpose(matrices[:-1, :3, :3], (0, 2, 1))
    angular = Rotation.from_matrix(rotations).magnitude() / dt
    if np.max(speeds) > WALK_SANITY_POLICY["maximum_camera_speed_m_s"]:
        reasons.append("camera_trajectory_speed_exceeds_slow_walk_bound")
    if np.max(angular) > WALK_SANITY_POLICY["maximum_camera_angular_speed_rad_s"]:
        reasons.append("camera_trajectory_rotation_exceeds_slow_walk_bound")
    return {"accepted": not reasons, "reason_codes": sorted(set(reasons)),
            "policy": deepcopy(WALK_SANITY_POLICY), "duration_s": duration,
            "maximum_camera_speed_m_s": float(np.max(speeds)),
            "maximum_camera_angular_speed_rad_s": float(np.max(angular)),
            "accuracy_validated": False, "runtime_world_admission": False}


def _artifact(directory: Path, name: str) -> dict[str, Any]:
    directory = Path(directory).resolve()
    report = _read(_member(directory, "report.json", 8 * 1024**2), 8 * 1024**2)
    if report.get("status") != "completed":
        raise MotionProfileError("Finish the source calibration job before checking a profile")
    info = report.get("artifacts", {}).get(name)
    path = _member(directory, name, 32 * 1024**2)
    if (not isinstance(info, Mapping) or info.get("path") != name
            or info.get("bytes") != path.stat().st_size or info.get("sha256") != _sha(path)):
        raise MotionProfileError("Calibration evidence no longer matches its retained report")
    return _read(path, 32 * 1024**2)


def _transform(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (result.shape != (4, 4) or not np.isfinite(result).all()
            or not np.array_equal(result[3], [0, 0, 0, 1])
            or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=2e-3)
            or np.linalg.det(result[:3, :3]) <= 0):
        raise MotionProfileError("Calibration needs finite, proper rigid transforms")
    return result


def assemble_motion_profile(*, camera_calibration_dir: Path, imu_calibration_dir: Path,
                            noise_calibration_dir: Path) -> dict[str, Any]:
    camera = _load_camera_reference(camera_calibration_dir)
    exported = camera_profile(camera)
    imu = _artifact(imu_calibration_dir, "camera_imu_result.json")
    noise = _artifact(noise_calibration_dir, "noise_result.json")
    if (imu.get("schema") != "roomwalk.camera_imu_calibration.v1"
            or imu.get("quality", {}).get("status") != "qualified"
            or imu.get("camera_imu_extrinsics_calibrated") is not True
            or imu.get("time_offset_calibrated") is not True
            or imu.get("physical_board_scale_verified") is not True
            or imu.get("camera_model_qualified") is not True):
        raise MotionProfileError("Repeat the camera–IMU step until its measured geometry and timing checks pass")
    if imu.get("camera_reference_sha256") != json_sha(camera) or imu.get("binding") != camera["binding"]:
        raise MotionProfileError("Choose the same camera result used by this camera–IMU job")
    noise_ref = imu.get("noise_reference") or {}
    noise_sha = _sha(Path(noise_calibration_dir) / "noise_result.json")
    if noise_ref.get("source_sha256") != noise_sha or noise_ref.get("noise_model_usable") is not True:
        raise MotionProfileError("Choose the same usable stationary result used by this camera–IMU job")
    if noise_ref.get("noise") != noise.get("noise"):
        raise MotionProfileError("Stationary noise parameters differ from the camera–IMU fit")
    if not imu.get("point_timing", {}).get("qualified"):
        raise MotionProfileError("Camera exposure timing is not qualified on this device")
    corrections = checked_corrections(imu.get("imu_corrections"))
    corrected_noise(noise_ref["noise"], corrections)
    transform = _transform(imu.get("T_imu_camera"))
    if np.linalg.norm(transform[:3, 3]) > 0.5:
        raise MotionProfileError("Camera–IMU translation exceeds the handheld device bound")
    offset = imu.get("imu_to_camera_offset_ns")
    if type(offset) is not int or abs(offset) > 100_000_000 or imu.get("cam_time_offset_ns") != -offset:
        raise MotionProfileError("The measured camera–IMU offset has inconsistent conventions")
    # Recheck the measured-vs-modelled status and exact sensor/device identity.
    # The same check is repeated against each actual future capture below.
    noise_binding = noise_ref.get("binding") or {}
    manifest = {"device": noise_binding.get("device", {})}
    timing = {"recorder": {"raw_device": noise_binding.get("device", {}),
                            "sensors": noise_binding.get("sensors", {})}}
    verified_noise = _load_noise_reference(noise_calibration_dir, manifest, timing)
    if verified_noise.get("noise_model_usable") is not True or verified_noise != noise_ref:
        raise MotionProfileError("The short sensor model does not reproduce its original evidence")
    profile = {
        "schema": PROFILE_SCHEMA, "consumer_version": CONSUMER_VERSION,
        "status": "candidate", "maximum_duration_s": MAXIMUM_WALK_SECONDS,
        "binding": deepcopy(camera["binding"]), "point_timing": deepcopy(imu["point_timing"]),
        "camera": {"intrinsics": camera["K"], "distortion": exported["distortion"],
                   "distortion_model": "brown_conrady", "resolution_px": camera["resolution"],
                   "stabilization": "off"},
        "imu_corrections": corrections, "T_imu_camera": transform.tolist(),
        "imu_to_camera_offset_ns": offset, "noise": noise_ref["noise"],
        "noise_provenance": noise_ref["noise_provenance"],
        "noise_model_status": noise_ref["noise_model_status"],
        "imu_noise_calibrated": noise_ref["imu_noise_calibrated"],
        "source_results": {"camera_sha256": json_sha(camera), "imu_sha256": json_sha(imu),
                           "noise_artifact_sha256": noise_sha},
        "validation_policy": deepcopy(VALIDATION_POLICY),
        "image_motion_model": "centre_timed_global_shutter_approximation",
        "rolling_shutter_compensated": False,
        "physical_board_scale_provenance": "user_attestation",
        "runtime_world_admission": False,
    }
    profile["profile_id"] = "short-" + json_sha(profile)[:24]
    return profile


def score_heldout_motion(vio: Mapping[str, Any], visual: Mapping[str, Any],
                         source_and_pose_times_ns: list) -> dict[str, Any]:
    """Compare fixed VIO with withheld visual-only target motion; never fit scale.

    A single first-pose SE3 anchor resolves the unrelated coordinate origins.
    Its pose is excluded from scored errors. No VIO values fit the visual spline.
    """
    if visual.get("visual_only") is not True or visual.get("status") != "converged":
        raise MotionProfileError("Held-out visual-only motion did not converge")
    reference = visual.get("visual_trajectory")
    if not isinstance(reference, list) or not 2 <= len(reference) <= 100000:
        raise MotionProfileError("The camera–IMU job lacks bounded held-out visual motion")
    times = [r["timestamp_ns"] for r in reference]
    if any(type(t) is not int for t in times) or any(a >= b for a, b in zip(times, times[1:])):
        raise MotionProfileError("Held-out visual timestamps are invalid")
    transforms = np.stack([_transform(r["T_target_camera"]) for r in reference])
    expected = {a: b for a, b in source_and_pose_times_ns if times[0] <= b <= times[-1]}
    rows = [r for r in vio.get("poses", []) if r.get("capture_time_ns") in expected]
    if (len(rows) < 2 or any(r.get("pose_time_ns") != expected[r["capture_time_ns"]] for r in rows)
            or any(a["pose_time_ns"] >= b["pose_time_ns"] for a, b in zip(rows, rows[1:]))):
        raise MotionProfileError("VIO poses do not match the held-out exposure times")
    stamp = np.array([r["pose_time_ns"] - times[0] for r in rows], float) * 1e-9
    ref_stamp = np.array([t - times[0] for t in times], float) * 1e-9
    rotations = Slerp(ref_stamp, Rotation.from_matrix(transforms[:, :3, :3]))(stamp).as_matrix()
    positions = np.stack([np.interp(stamp, ref_stamp, transforms[:, k, 3]) for k in range(3)], axis=1)
    observed = np.stack([_transform(r["T_vio_world_camera"]) for r in rows])
    anchor_rotation = rotations[0] @ observed[0, :3, :3].T
    aligned = (observed[:, :3, 3] - observed[0, :3, 3]) @ anchor_rotation.T + positions[0]
    position_error = np.linalg.norm(aligned[1:] - positions[1:], axis=1)
    rotation_error = Rotation.from_matrix(
        rotations[1:] @ np.transpose(anchor_rotation @ observed[1:, :3, :3], (0, 2, 1))
    ).magnitude() * 180 / np.pi
    reference_displacement = positions[1:] - positions[0]
    actual_displacement = aligned[1:] - positions[0]
    denominator = float(np.sum(reference_displacement**2))
    scale = float(np.sum(actual_displacement * reference_displacement) / denominator) if denominator > 1e-12 else None
    metrics = {
        "heldout_seconds": float(stamp[-1] - stamp[0]), "heldout_poses": len(rows),
        "heldout_coverage": len(rows) / max(1, len(expected)),
        "maximum_pose_gap_s": float(np.max(np.diff(stamp))),
        "translation_extent_m": float(np.max(np.linalg.norm(reference_displacement, axis=1))),
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_p95_m": float(np.percentile(position_error, 95)),
        "orientation_rmse_deg": float(np.sqrt(np.mean(rotation_error**2))),
        "orientation_p95_deg": float(np.percentile(rotation_error, 95)),
        "scale_ratio_diagnostic_only": scale,
    }
    reasons = []
    for key in ("heldout_seconds", "heldout_poses", "heldout_coverage", "translation_extent_m"):
        if metrics[key] < VALIDATION_POLICY["minimum_" + key]:
            reasons.append("insufficient_" + key)
    for key in ("maximum_pose_gap_s", "position_rmse_m", "position_p95_m", "orientation_rmse_deg", "orientation_p95_deg"):
        threshold_key = key if key.startswith("maximum_") else "maximum_" + key
        if metrics[key] > VALIDATION_POLICY[threshold_key]:
            reasons.append(key + "_exceeds_check_limit")
    if scale is None or abs(scale - 1) > VALIDATION_POLICY["maximum_scale_error_fraction"]:
        reasons.append("heldout_metric_scale_mismatch")
    if any(segment.get("reset") is not False for segment in vio.get("segments", [])):
        reasons.append("vio_reset_during_profile_check")
    return {
        "schema": "roomwalk.motion_validation.v1", "passed": not reasons,
        "reason_codes": reasons, "metrics": metrics, "policy": deepcopy(VALIDATION_POLICY),
        "alignment": "single_first_pose_SE3_anchor_excluded_from_errors",
        "scale_fitted_or_applied": False, "calibration_refitted": False,
        "comparison": "withheld_visual_only_target_motion_vs_fixed_OpenVINS",
        "room_accuracy_certified": False,
    }


def run_motion_profile(capture_dir: Path, request: Mapping[str, Any], output: Path,
                       *, settings: Mapping[str, Any], progress) -> dict[str, Any]:
    from .vio import (VIOSettings, materialize_openvins_profile_validation_input,
                      run_openvins_profile_validation)
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema": "roomwalk.calibration_report.v1", "mode": "motion_profile",
              "status": "failed", "motion_profile_ready": False, "reason_codes": []}
    _write(output / "request.json", dict(request))
    try:
        progress(0.02, "Checking matching camera, sensor, and timing evidence")
        source_dirs = {k: Path(settings[k]) for k in
                       ("camera_calibration_dir", "imu_calibration_dir", "noise_calibration_dir")}
        profile = assemble_motion_profile(**source_dirs)
        _write(output / "motion_profile.json", profile)
        capture = _read(capture_dir / "capture_import.json")
        source_report_sha = json_sha(capture)
        derived = bind_profile_to_capture(capture_dir, capture, profile, validation=True)
        raw = _read_json(capture_dir / capture["manifest"]["android_capture"]["capture_result_path"])
        actual_noise = _load_noise_reference(source_dirs["noise_calibration_dir"], capture["manifest"], {"recorder": {**raw, "raw_device": raw.get("device", {})}})
        if not actual_noise.get("noise_model_usable") or actual_noise["source_sha256"] != profile["source_results"]["noise_artifact_sha256"]:
            raise MotionProfileError("This camera–IMU recording does not match the chosen motion sensors")
        _write(output / "profile_capture_report.json", derived)
        vio_settings = VIOSettings(executable=Path(settings["vio_executable"]), config=Path(settings["vio_config"]))
        input_dir = output / "openvins_input"
        materialized_dir, runtime_config = materialize_openvins_profile_validation_input(
            capture_dir, derived, input_dir, vio_settings, lambda f, m: progress(0.05 + 0.35 * f, m))
        result = run_openvins_profile_validation(materialized_dir, output / "openvins", replace(vio_settings, config=runtime_config),
                                                 lambda f, m: progress(0.4 + 0.5 * f, m))
        _write(output / "profile_vio_result.json", result)
        visual = _artifact(source_dirs["imu_calibration_dir"], "heldout_visual_result.json")
        validation = score_heldout_motion(result, visual, derived["short_session"]["source_and_pose_times_ns"])
        validation["visual_result_sha256"] = json_sha(visual)
        validation["vio_result_sha256"] = json_sha(result)
        _write(output / "motion_validation.json", validation)
        if assemble_motion_profile(**source_dirs) != profile or json_sha(_read(capture_dir / "capture_import.json")) != source_report_sha:
            raise MotionProfileError("Calibration evidence changed during the profile check")
        profile.update(status="ready_for_short_walks" if validation["passed"] else "check_failed",
                       validated_motion_envelope=derived["short_session"]["motion_envelope"],
                       validation_sha256=json_sha(validation), source_jobs={k: request[k] for k in
                       ("camera_calibration_id", "imu_calibration_id", "noise_calibration_id")},
                       consumer_executable_sha256=_sha(vio_settings.executable),
                       consumer_config_sha256=_sha(vio_settings.config))
        _write(output / "motion_profile.json", profile)
        report.update(status="completed", motion_profile_ready=validation["passed"],
                      reason_codes=validation["reason_codes"], quality={"status": profile["status"]},
                      profile_id=profile["profile_id"], maximum_duration_s=MAXIMUM_WALK_SECONDS,
                      imu_noise_calibrated=profile["imu_noise_calibrated"],
                      noise_model_status=profile["noise_model_status"], validation=validation)
        progress(1.0, "Profile ready for matching short walks" if validation["passed"] else
                 "Profile check needs attention; source recordings and results retained")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        report.update(error=str(exc), reason_codes=["motion_profile_check_failed"])
    report["artifacts"] = {name: {"path": name, "sha256": _sha(output / name),
                                      "bytes": (output / name).stat().st_size}
                           for name in ("motion_profile.json", "motion_validation.json", "profile_vio_result.json", "request.json", "profile_capture_report.json")
                           if (output / name).is_file()}
    _write(output / "report.json", report)
    return report


class MotionSelection:
    def __init__(self, jobs, camera_selection):
        self.jobs = jobs
        self.camera_selection = camera_selection
        self.path = jobs.root / "motion-selection.json"
        self.lock = threading.RLock()

    def get(self):
        with self.lock:
            return _read(self.path, 16384).get("selection") if self.path.is_file() else None

    def _profile(self, job_id):
        job = self.jobs.get(job_id)
        if job["mode"] != "motion_profile" or job["status"] != "completed" or (job.get("result") or {}).get("motion_profile_ready") is not True:
            raise CalibrationJobError("Finish a passing motion-profile check before selecting it")
        directory = self.jobs.artifact(job_id, "report.json").parent
        profile = _artifact(directory, "motion_profile.json")
        validation = _artifact(directory, "motion_validation.json")
        if (profile.get("status") != "ready_for_short_walks" or validation.get("passed") is not True
                or profile.get("validation_sha256") != json_sha(validation)
                or profile.get("consumer_version") != CONSUMER_VERSION
                or validation.get("policy") != VALIDATION_POLICY):
            raise MotionProfileError("This profile needs a new check by the current RoomWalk consumer")
        dirs = {mode + "_calibration_dir": self.jobs.artifact(profile["source_jobs"][mode + "_calibration_id"], "report.json").parent
                for mode in ("camera", "imu", "noise")}
        candidate = assemble_motion_profile(**dirs)
        if any(profile.get(key) != value for key, value in candidate.items() if key != "status"):
            raise MotionProfileError("Selected profile differs from its source calibration")
        return profile, dirs

    def select(self, job_id):
        with self.lock:
            if job_id is None:
                _write(self.path, {"selection": None})
                return None
            if not isinstance(job_id, str):
                raise CalibrationJobError("Select a completed motion-profile job")
            profile, dirs = self._profile(job_id)
            camera_id = profile["source_jobs"]["camera_calibration_id"]
            camera = _load_camera_reference(dirs["camera_calibration_dir"])
            path = self.camera_selection._materialize(camera_id, camera, camera_profile(camera))
            from .phone_calibration import load_phone_calibration
            selection = {"motion_calibration_id": job_id, "profile_id": profile["profile_id"],
                         "profile_sha256": json_sha(profile), "status": "ready_for_short_walks",
                         "maximum_duration_s": MAXIMUM_WALK_SECONDS, "selected_at": _now(),
                         "camera_selection": {"camera_calibration_id": camera_id,
                          "profile_id": load_phone_calibration(path)["profile_id"],
                          "binding_sha256": profile["binding"]["sha256"]}}
            _write(self.path, {"selection": selection})
            return selection

    def derived_report(self, capture_dir, capture_report, selection):
        if not isinstance(selection, Mapping):
            raise MotionProfileError("Choose a checked motion profile before recording the next walk")
        profile, dirs = self._profile(selection["motion_calibration_id"])
        if profile["profile_id"] != selection.get("profile_id") or json_sha(profile) != selection.get("profile_sha256"):
            raise MotionProfileError("Selected motion profile changed after this recording was imported")
        from .vio import VIOSettings
        settings = VIOSettings.from_env()
        if (settings.executable is None or settings.config is None
                or _sha(settings.executable) != profile.get("consumer_executable_sha256")
                or _sha(settings.config) != profile.get("consumer_config_sha256")):
            raise MotionProfileError("The OpenVINS consumer changed; rerun the retained motion-profile check")
        evidence = capture_report["manifest"]["android_capture"]
        raw = _read_json(_member(capture_dir, evidence["capture_result_path"], 512 * 1024))
        noise = _load_noise_reference(dirs["noise_calibration_dir"], capture_report["manifest"],
                                      {"recorder": {**raw, "raw_device": raw.get("device", {})}})
        if noise.get("noise_model_usable") is not True:
            raise MotionProfileError("The recording's motion sensors no longer match this profile")
        return bind_profile_to_capture(capture_dir, capture_report, profile)
