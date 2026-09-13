"""Measured phone-camera geometry shared by the two reconstruction providers."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


RECTIFIED_INTRINSICS_SCHEMA = "noesis.phone_scan.rectified_intrinsics.v1"
_RAW_CALIBRATION_FIELDS = (
    "camera_intrinsics_json",
    "calibration_processing_json",
    "rectification_valid_mask",
    "network_intrinsics",
)


def _pinhole(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
        or not np.array_equal(matrix[2], [0.0, 0.0, 1.0])
        or matrix[0, 1] != 0.0
        or matrix[1, 0] != 0.0
    ):
        raise ValueError(f"{name} must be a finite, positive, zero-skew pinhole matrix")
    return matrix


def _resolution(value: Any, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise ValueError(f"{name} must contain positive integer width and height")
    return int(value[0]), int(value[1])


def calibration_specs(
    frame_rows: Sequence[dict[str, Any]], *, anchor_image: Path | None = None
) -> list[dict[str, Any]] | None:
    """Fail closed on partial calibration before loading either model."""
    present = ["camera_intrinsics" in row for row in frame_rows]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("mixed calibrated and uncalibrated phone views are unsupported")
    if anchor_image is not None:
        raise ValueError("calibrated phone views cannot use an unbound fixed-camera anchor")
    specs = []
    for index, row in enumerate(frame_rows):
        spec = row["camera_intrinsics"]
        if (
            not isinstance(spec, dict)
            or spec.get("schema") != RECTIFIED_INTRINSICS_SCHEMA
            or spec.get("calibration_applied") is not True
            or spec.get("distortion_model") != "none"
        ):
            raise ValueError(f"view {index} has no valid rectified camera-intrinsics binding")
        if not isinstance(spec.get("profile_id"), str) or not spec["profile_id"].strip():
            raise ValueError(f"view {index} has no camera calibration profile identity")
        digest = str(spec.get("profile_sha256") or "")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"view {index} has no valid camera calibration profile hash")
        _pinhole(spec.get("K"), "camera_intrinsics.K")
        _pinhole(spec.get("source_K"), "camera_intrinsics.source_K")
        width, height = _resolution(spec.get("resolution_px"), "camera_intrinsics.resolution_px")
        _resolution(spec.get("source_resolution_px"), "camera_intrinsics.source_resolution_px")
        distortion = np.asarray(spec.get("source_distortion"), dtype=np.float64)
        if distortion.shape != (5,) or not np.isfinite(distortion).all():
            raise ValueError("camera_intrinsics.source_distortion must retain all five OpenCV coefficients")
        if row.get("width", width) != width or row.get("height", height) != height:
            raise ValueError(f"view {index} dimensions disagree with its calibrated intrinsics")
        specs.append(spec)
    return specs


def calibrated_input_views(
    frame_paths: Sequence[Path], specs: Sequence[dict[str, Any]],
    *, frame_rows: Sequence[dict[str, Any]], scan_dir: Path,
) -> list[dict[str, Any]]:
    snapshot = scan_dir / "phone_camera_calibration.json"
    try:
        profile_bytes = snapshot.read_bytes()
        profile = json.loads(profile_bytes)
        profile_matrix = _pinhole(profile["K"], "retained calibration K")
        profile_width, profile_height = _resolution(profile["resolution"], "retained calibration resolution")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("retained phone camera calibration snapshot is unreadable") from exc
    profile_digest = hashlib.sha256(profile_bytes).hexdigest()
    views = []
    for path, spec, row in zip(frame_paths, specs, frame_rows, strict=True):
        if (
            profile.get("schema") != "noesis.phone_camera_calibration.v1"
            or profile.get("profile_id") != spec["profile_id"]
            or profile_digest != spec["profile_sha256"]
            or profile.get("D") != spec["source_distortion"]
        ):
            raise ValueError("prepared calibration binding disagrees with its retained profile")
        source_width, source_height = spec["source_resolution_px"]
        expected_matrix = profile_matrix.copy()
        expected_matrix[0] *= source_width / profile_width
        expected_matrix[1] *= source_height / profile_height
        if (
            spec["source_resolution_px"] != spec["resolution_px"]
            or not np.allclose(spec["source_K"], expected_matrix, atol=1e-8, rtol=1e-10)
            or not np.allclose(spec["K"], expected_matrix, atol=1e-8, rtol=1e-10)
        ):
            raise ValueError("prepared intrinsics do not match the retained profile's resize and same-K rectification")
        image_bytes = path.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != row.get("sha256"):
            raise ValueError(f"calibrated phone frame hash changed: {path.name}")
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"calibrated phone frame is unreadable: {path.name}")
        if image.shape[1::-1] != tuple(spec["resolution_px"]):
            raise ValueError(f"calibrated phone frame dimensions changed: {path.name}")
        views.append({
            "img": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "intrinsics": np.asarray(spec["K"], dtype=np.float32),
        })
    return views


def rectification_valid_mask(
    spec: dict[str, Any], model_intrinsics: np.ndarray, model_hw: tuple[int, int]
) -> np.ndarray:
    """Map model-grid rays back to raw distorted pixels, retaining full D5."""
    matrix = _pinhole(model_intrinsics, "model intrinsics")
    height, width = model_hw
    yy, xx = np.indices((height, width), dtype=np.float64)
    pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    rays = pixels.reshape(-1, 3) @ np.linalg.inv(matrix).T
    distorted, _ = cv2.projectPoints(
        rays,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.asarray(spec["source_K"], dtype=np.float64),
        np.asarray(spec["source_distortion"], dtype=np.float64),
    )
    raw_xy = distorted.reshape(height, width, 2)
    source_width, source_height = spec["source_resolution_px"]
    # Numerical roundoff at a pixel center must not invalidate an identity map.
    epsilon = 1e-5
    return (
        np.isfinite(raw_xy).all(axis=-1)
        & (raw_xy[..., 0] >= -epsilon)
        & (raw_xy[..., 0] <= source_width - 1 + epsilon)
        & (raw_xy[..., 1] >= -epsilon)
        & (raw_xy[..., 1] <= source_height - 1 + epsilon)
    )


def calibration_raw_fields(
    row: dict[str, Any], valid_mask: np.ndarray | None,
    *, network_intrinsics: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if valid_mask is None:
        return {}
    result = {
        "camera_intrinsics_json": np.asarray(json.dumps(row["camera_intrinsics"], sort_keys=True)),
        "calibration_processing_json": np.asarray(json.dumps(row.get("calibration_processing"), sort_keys=True)),
        "rectification_valid_mask": np.asarray(valid_mask, dtype=np.uint8),
    }
    if network_intrinsics is not None:
        result["network_intrinsics"] = np.asarray(network_intrinsics, dtype=np.float32)
    return result


def preserved_calibration_raw_fields(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: raw[key] for key in _RAW_CALIBRATION_FIELDS if key in raw}


def calibration_frame_fields(
    row: dict[str, Any], valid_mask: np.ndarray | None,
    *, network_conditioned: bool, network_intrinsics: np.ndarray | None = None,
) -> dict[str, Any]:
    if valid_mask is None:
        return {}
    result = {
        "camera_intrinsics": deepcopy(row["camera_intrinsics"]),
        "calibration_processing": deepcopy(row.get("calibration_processing")),
        "source_frame_id": row.get("frame_id"),
        "source_frame_sha256": row.get("sha256"),
        "intrinsics_source": "mapanything_calibration_conditioned" if network_conditioned else "measured_rectified",
        "network_calibration_conditioned": network_conditioned,
        "rectification_valid_fraction": float(np.mean(valid_mask)),
    }
    if network_intrinsics is not None:
        result["network_intrinsics"] = np.asarray(network_intrinsics).tolist()
    return result


def add_calibration_summary(
    result: dict[str, Any], prepared: dict[str, Any],
    specs: Sequence[dict[str, Any]] | None, *, network_conditioned: bool,
) -> None:
    if isinstance(prepared.get("camera_calibration"), dict):
        result["camera_calibration"] = deepcopy(prepared["camera_calibration"])
    if specs is None:
        return
    profiles = sorted({(spec["profile_id"], spec["profile_sha256"]) for spec in specs})
    result["inference_calibration"] = {
        "schema": "noesis.phone_scan.inference_calibration.v1",
        "profiles": [{"profile_id": name, "profile_sha256": digest} for name, digest in profiles],
        "network_calibration_conditioned": network_conditioned,
        "geometry_intrinsics_source": "mapanything_calibration_conditioned" if network_conditioned else "measured_rectified",
        "distortion_removed_during_preparation": True,
        "rectification_border_policy": "model_rays_project_inside_raw_source_with_full_opencv_D5",
    }
