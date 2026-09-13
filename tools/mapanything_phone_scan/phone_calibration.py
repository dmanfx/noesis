"""Import and apply measured phone intrinsics without admitting metric VIO."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import stat
import tempfile
from typing import Any
import zipfile

import cv2
import numpy as np


PROFILE_SCHEMA = "noesis.phone_camera_calibration.v1"
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")


class PhoneCalibrationError(RuntimeError):
    """The calibration or its association with the recorded pixels is invalid."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _camera_parameters(camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if camera.get("model") != "opencv_pinhole":
        raise PhoneCalibrationError("phone calibration must declare opencv_pinhole")
    resolution = camera.get("resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(type(value) is not int or not 1 <= value <= 16384 for value in resolution)
    ):
        raise PhoneCalibrationError("calibration resolution must be [width, height]")
    size = (resolution[0], resolution[1])
    matrix = np.asarray(camera.get("K"), dtype=np.float64)
    distortion = np.asarray(camera.get("D"), dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-10)
        or matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
        or abs(matrix[0, 1]) > 1e-10
        or abs(matrix[1, 0]) > 1e-10
        or not 0 <= matrix[0, 2] < size[0]
        or not 0 <= matrix[1, 2] < size[1]
    ):
        raise PhoneCalibrationError("calibration K is not a valid OpenCV camera matrix")
    if distortion.shape != (5,) or not np.isfinite(distortion).all():
        raise PhoneCalibrationError("phone calibration currently requires all five OpenCV pinhole coefficients")
    if camera.get("runtime", {}).get("use_undistorted_frames") is not False:
        raise PhoneCalibrationError("calibration must explicitly describe raw distorted frames")
    return matrix, distortion, size


def import_calibration_bundle(bundle_path: Path, output_root: Path) -> Path:
    """Verify a flat handoff archive and retain the complete original evidence."""
    if bundle_path.stat().st_size > 32 * 1024 * 1024:
        raise PhoneCalibrationError("phone calibration bundle exceeds 32 MiB")
    archive_bytes = bundle_path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        members = archive.infolist()
        if len(members) > 64 or sum(row.file_size for row in members) > 32 * 1024 * 1024:
            raise PhoneCalibrationError("phone calibration archive exceeds its file/size limits")
        names = [row.filename for row in members]
        if len(set(names)) != len(names) or any(
            not SAFE_ID.fullmatch(row.filename)
            or row.is_dir()
            or stat.S_ISLNK(row.external_attr >> 16)
            for row in members
        ):
            raise PhoneCalibrationError("calibration archive must contain unique plain filenames")
        manifest = json.loads(archive.read("manifest.json"))
        declared = manifest.get("files")
        if not isinstance(declared, dict) or set(declared) != set(names) - {"manifest.json"}:
            raise PhoneCalibrationError("calibration manifest does not cover the exact archive")
        contents = {name: archive.read(name) for name in names}
        for name, expected in declared.items():
            actual = contents[name]
            if expected.get("bytes") != len(actual) or expected.get("sha256") != _digest(actual):
                raise PhoneCalibrationError(f"calibration checksum/size mismatch: {name}")
    camera = json.loads(contents["camera_noesis.json"])
    matrix, distortion, size = _camera_parameters(camera)
    camera_id = str(camera.get("camera_id") or "")
    session = str(manifest.get("session") or "")
    if not SAFE_ID.fullmatch(camera_id) or not SAFE_ID.fullmatch(session):
        raise PhoneCalibrationError("calibration camera/session identity is invalid")
    if manifest.get("camera_id") != camera_id:
        raise PhoneCalibrationError("calibration camera identity disagrees with manifest")
    mode = json.loads(contents["capture_mode.json"])
    if mode.get("camera_id") != camera_id or mode.get("calibration_resolution") != list(size):
        raise PhoneCalibrationError("capture mode disagrees with camera calibration")
    with np.load(io.BytesIO(contents["camera_opencv.npz"]), allow_pickle=False) as numeric:
        for key, expected in (
            ("camera_matrix", matrix),
            ("distortion_coefficients", distortion),
            ("image_size", np.asarray(size)),
        ):
            if not np.array_equal(numeric[key], expected):
                raise PhoneCalibrationError(f"JSON and NPZ calibration disagree: {key}")
    profile = {
        "schema": PROFILE_SCHEMA,
        "profile_id": f"{camera_id}-{session}",
        "camera_id": camera_id,
        "session": session,
        "model": "opencv_pinhole",
        "resolution": list(size),
        "K": matrix.tolist(),
        "D": distortion.tolist(),
        "runtime": {"use_undistorted_frames": False},
        "calibration_date": camera.get("calibration_date"),
        "quality": {
            "status": camera.get("status"),
            "training_mean_error_px": camera.get("reprojection_error_px"),
            "holdout_mean_error_px": camera.get("independent_video_validation", {}).get("mean_error_px"),
            "holdout_group_count": camera.get("independent_video_validation", {}).get("pose_group_count"),
            "warnings": camera.get("warnings", []),
        },
        "capture_mode": mode,
        "binding_policy": "explicit_capture_mode_match_required",
        "metric_vio_allowed": False,
        "source": {
            "archive": "source_bundle.zip",
            "archive_sha256": _digest(archive_bytes),
            "camera": "source/camera_noesis.json",
            "camera_sha256": _digest(contents["camera_noesis.json"]),
            "manifest": "source/manifest.json",
            "manifest_sha256": _digest(contents["manifest.json"]),
            "verified_file_count": len(declared),
        },
    }
    target = output_root / camera_id / session
    profile_path = target / "profile.json"
    if target.exists():
        if load_phone_calibration(profile_path)["source"] != profile["source"]:
            raise PhoneCalibrationError("existing calibration session has different source evidence")
        return profile_path
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phone-calibration-", dir=target.parent) as temporary:
        staging = Path(temporary) / "profile"
        (staging / "source").mkdir(parents=True)
        (staging / "source_bundle.zip").write_bytes(archive_bytes)
        for name, payload in contents.items():
            (staging / "source" / name).write_bytes(payload)
        profile["imported_at"] = datetime.now(timezone.utc).isoformat()
        (staging / "profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        staging.rename(target)
    return profile_path


def load_phone_calibration(profile_path: Path) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("schema") != PROFILE_SCHEMA:
        raise PhoneCalibrationError("unsupported phone camera calibration profile")
    _camera_parameters(profile)
    root = profile_path.parent.resolve()
    for field in ("archive", "camera", "manifest"):
        path = (root / profile["source"][field]).resolve()
        if not path.is_relative_to(root) or _digest(path.read_bytes()) != profile["source"][f"{field}_sha256"]:
            raise PhoneCalibrationError(f"phone calibration source changed: {field}")
    source_camera = json.loads((root / profile["source"]["camera"]).read_text(encoding="utf-8"))
    for field in ("camera_id", "model", "resolution", "K", "D"):
        if profile[field] != source_camera[field]:
            raise PhoneCalibrationError(f"phone calibration profile differs from its source: {field}")
    return profile


def calibration_summary(profile_path: Path | None) -> dict[str, Any] | None:
    if profile_path is None:
        return None
    profile = load_phone_calibration(profile_path)
    return {
        "profile_id": profile["profile_id"],
        "resolution": profile["resolution"],
        "model": profile["model"],
        "distortion_coefficient_count": len(profile["D"]),
        "source_sha256": profile["source"]["archive_sha256"],
        "quality": profile["quality"],
        "metric_vio_allowed": False,
    }


def calibration_for_video(
    profile_path: Path | None,
    *,
    configured_capture_mode: str,
    capture_mode: str,
    video: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Require the configured recorder mode and original encoded image geometry."""
    if profile_path is None:
        return None, None
    profile = load_phone_calibration(profile_path)
    summary = calibration_summary(profile_path)
    assert summary is not None
    reasons: list[str] = []
    if configured_capture_mode == "unbound":
        reasons.append("capture_mode_not_bound")
    elif capture_mode != configured_capture_mode:
        reasons.append("capture_mode_mismatch")
    if [video.get("width"), video.get("height")] != profile["resolution"]:
        reasons.append("native_resolution_mismatch")
    if abs(float(video.get("rotation_degrees") or 0.0)) % 360.0 > 1e-6:
        reasons.append("encoded_rotation_requires_separate_calibration_transform")
    summary.update(
        {
            "status": "not_applied" if reasons else "applied",
            "reason_codes": reasons,
            "configured_capture_mode": configured_capture_mode,
            "recorded_capture_mode": capture_mode,
            "recorded_resolution_px": [video.get("width"), video.get("height")],
            "profile_sha256": _digest(profile_path.read_bytes()),
            "calibrated_pose_estimation": False,
        }
    )
    return (None if reasons else profile), summary


def rectification_maps(profile: dict[str, Any], size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rectify a pure resize of the calibrated raw image, preserving its full K/D."""
    matrix, distortion, original_size = _camera_parameters(profile)
    width, height = size
    if width <= 0 or height <= 0:
        raise PhoneCalibrationError("rectification image size is invalid")
    # Exact resize dimensions, including any integer rounding by extraction.
    matrix[0] *= width / original_size[0]
    matrix[1] *= height / original_size[1]
    map_x, map_y = cv2.initUndistortRectifyMap(
        matrix, distortion, np.eye(3), matrix, (width, height), cv2.CV_32FC1
    )
    return map_x, map_y, matrix


def rectify_selected_frames(
    frame_paths: list[Path],
    profile: dict[str, Any],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Write rectified selected inputs before their final hashes are established."""
    rows: list[dict[str, Any]] = []
    maps: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    common_size: tuple[int, int] | None = None
    for path in frame_paths:
        source_sha = _digest(path.read_bytes())
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise PhoneCalibrationError(f"cannot read selected phone frame: {path.name}")
        size = (image.shape[1], image.shape[0])
        if common_size is None:
            common_size = size
            maps = rectification_maps(profile, size)
        elif size != common_size:
            raise PhoneCalibrationError("calibrated selected frames have inconsistent dimensions")
        # Selected images must be a pure resize of the native calibrated image.
        # Tolerate at most one pixel of extractor rounding, never a portrait turn.
        expected_height = size[0] * profile["resolution"][1] / profile["resolution"][0]
        if abs(size[1] - expected_height) > 1.0:
            raise PhoneCalibrationError("selected frame does not retain calibrated native framing")
        assert maps is not None
        map_x, map_y, matrix = maps
        rectified = cv2.remap(
            image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        if not cv2.imwrite(str(path), rectified, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise PhoneCalibrationError(f"cannot write rectified phone frame: {path.name}")
        rows.append(
            {
                "camera_intrinsics": {
                    "schema": "noesis.phone_scan.rectified_intrinsics.v1",
                    "profile_id": profile["profile_id"],
                    "profile_sha256": summary["profile_sha256"],
                    "K": matrix.tolist(),
                    "resolution_px": list(size),
                    "distortion_model": "none",
                    "calibration_applied": True,
                    "source_K": matrix.tolist(),
                    "source_distortion": list(profile["D"]),
                    "source_resolution_px": list(size),
                },
                "calibration_processing": {
                    "source_frame_sha256": source_sha,
                    "calibrated_native_resolution_px": profile["resolution"],
                    "operations": ["pure_resize_of_native_raw_pixels", "opencv_pinhole_undistort_same_K_no_crop"],
                    "native_to_resized_pixel_transform": [
                        [size[0] / profile["resolution"][0], 0.0, 0.0],
                        [0.0, size[1] / profile["resolution"][1], 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "border_policy": "reject_rays_outside_distorted_source_in_inference",
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = import_calibration_bundle(args.bundle, args.output_root)
    print(json.dumps({"profile": str(result), **calibration_summary(result)}, indent=2))


if __name__ == "__main__":
    main()
