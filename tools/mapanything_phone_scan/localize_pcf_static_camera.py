#!/usr/bin/env python3
"""Localize a fixed camera inside a PCF room without whole-cloud fitting.

The estimator uses mutual static/phone RGB features, a per-view fundamental
matrix gate, PCF depth-backed 3D points, and independent per-phone-view PnP.
It then takes a view-balanced consensus of the admitted camera poses.  The
legacy mode preserves calibrated camera height, pitch, and roll.  Full-PCF mode
uses the complete metric PnP pose and checks it against the fused PCF floor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.mapanything_phone_scan.calibration_replacement import (
    CalibrationReplacementError,
    write_calibration_replacement,
)


class StaticCameraLocalizationError(RuntimeError):
    """Raised when the preserved evidence cannot support a camera anchor."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StaticCameraLocalizationError(f"{path} is not a JSON object")
    return value


def _adapt_static_image_metadata(
    metadata: dict[str, Any],
    *,
    image_path: Path,
    image: np.ndarray,
    camera_id: str,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Adapt the existing stream-points producer without inventing acceptance.

    ``room_reconstruction.stream_points.v4`` predates the strict static-image
    contract.  It can still supply a review keyframe when its declared RGB
    keyframe is byte-identical to the requested image.  That producer did not
    record distortion provenance, so the adapted contract is explicitly
    marked review-only and cannot satisfy measured canonical acceptance.
    """
    if metadata.get("schema") != "noesis.room_reconstruction.stream_points.v4":
        return metadata
    if metadata.get("camera") != camera_id:
        raise StaticCameraLocalizationError(
            "legacy stream-points metadata camera does not match the requested camera"
        )
    keyframes = metadata.get("rgb_keyframes")
    if not isinstance(keyframes, dict) or not keyframes:
        raise StaticCameraLocalizationError(
            "legacy stream-points metadata has no declared RGB keyframe"
        )
    metadata_root = (metadata_path.parent if metadata_path is not None else image_path.parent).resolve()
    # The metadata is normally beside the revision directory.  Check each
    # declared path explicitly, without walking the artifact tree.
    matching_keyframe: tuple[str, Path] | None = None
    requested_sha = _sha256(image_path)
    for key, relative in keyframes.items():
        if not isinstance(key, str) or not isinstance(relative, str):
            continue
        candidate = (metadata_root / relative).resolve()
        if candidate.is_file() and _sha256(candidate) == requested_sha:
            matching_keyframe = (key, candidate)
            break
    if matching_keyframe is None:
        raise StaticCameraLocalizationError(
            "requested static keyframe is not the exact RGB keyframe declared by "
            "legacy stream-points metadata"
        )
    declared_contract = metadata.get("static_camera_image")
    if isinstance(declared_contract, dict):
        adapted = dict(declared_contract)
        adapted.setdefault("input_schema", metadata["schema"])
        return adapted
    revision = str(metadata.get("revision_id") or "").strip()
    coordinate_frame = str(metadata.get("coordinate_frame") or "").strip()
    intrinsics = metadata.get("intrinsics")
    if not revision or not coordinate_frame or intrinsics is None:
        raise StaticCameraLocalizationError(
            "legacy stream-points metadata lacks revision, frame, or intrinsics"
        )
    return {
        "schema": "noesis.pcf.static_camera_image.v1",
        "input_schema": metadata["schema"],
        "camera_id": camera_id,
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "intrinsics": intrinsics,
        "distortion_model": "none",
        "distortion": [],
        "rectification": {"status": "unknown", "source": "legacy_stream_points_v4"},
        "distortion_provenance": "missing_legacy_stream_points_metadata",
        "source_frame": {
            "frame_id": matching_keyframe[0],
            "revision": revision,
            "coordinate_frame": coordinate_frame,
            "units": "m",
            "image_sha256": requested_sha,
        },
    }


def _validate_static_image_contract(
    metadata: dict[str, Any],
    image: np.ndarray,
    *,
    image_path: Path,
    camera_id: str,
    metadata_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Validate the exact calibrated image consumed by PnP.

    A static image without a declared source frame, resolution, distortion
    state, and digest is useful review material but cannot establish an
    extrinsic replacement.
    """
    metadata = _adapt_static_image_metadata(
        metadata,
        image_path=image_path,
        image=image,
        camera_id=camera_id,
        metadata_path=metadata_path,
    )
    if metadata.get("schema") != "noesis.pcf.static_camera_image.v1":
        raise StaticCameraLocalizationError(
            "static metadata must use noesis.pcf.static_camera_image.v1"
        )
    declared_camera = str(metadata.get("camera_id") or "")
    if declared_camera != camera_id:
        raise StaticCameraLocalizationError("static metadata camera_id does not match the requested camera")
    image_size = metadata.get("image_size") or metadata.get("resolution_px")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(not isinstance(value, int) or value <= 0 for value in image_size)
    ):
        raise StaticCameraLocalizationError("static metadata image_size must be positive [width,height]")
    width, height = int(image_size[0]), int(image_size[1])
    if image.shape[:2] != (height, width):
        raise StaticCameraLocalizationError(
            f"static keyframe resolution {image.shape[1]}x{image.shape[0]} does not match calibrated {width}x{height}"
        )
    intrinsics = np.asarray(metadata.get("intrinsics"), dtype=np.float64)
    if (
        intrinsics.shape != (3, 3)
        or not np.isfinite(intrinsics).all()
        or not np.allclose(intrinsics[2], [0.0, 0.0, 1.0], atol=1e-7)
        or intrinsics[0, 0] <= 0.0
        or intrinsics[1, 1] <= 0.0
        or not (0.0 <= intrinsics[0, 2] < width)
        or not (0.0 <= intrinsics[1, 2] < height)
    ):
        raise StaticCameraLocalizationError("static-camera intrinsics are malformed or outside the calibrated image")
    distortion_model = str(metadata.get("distortion_model") or "").strip().lower()
    supported_models = {
        "none", "plumb_bob", "radtan", "brown_conrady", "opencv", "opencv_radtan",
        "fisheye", "equidistant",
    }
    if distortion_model not in supported_models:
        raise StaticCameraLocalizationError("static metadata has no supported distortion_model")
    distortion_raw = metadata.get("distortion")
    if not isinstance(distortion_raw, list) or len(distortion_raw) > 14:
        raise StaticCameraLocalizationError("static metadata distortion must be a finite coefficient list")
    distortion = np.asarray(distortion_raw, dtype=np.float64)
    if not np.isfinite(distortion).all() or distortion.size not in {0, 4, 5, 8, 12, 14}:
        raise StaticCameraLocalizationError("static metadata distortion has an unsupported shape")
    if distortion_model == "none" and np.any(np.abs(distortion) > 1e-12):
        raise StaticCameraLocalizationError("distortion_model=none cannot carry nonzero distortion")
    rectification = metadata.get("rectification")
    if isinstance(rectification, str):
        rectification_status = rectification.strip().lower()
    elif isinstance(rectification, dict):
        rectification_status = str(rectification.get("status") or "").strip().lower()
    else:
        rectification_status = ""
    if rectification_status not in {"raw", "rectified", "unknown"}:
        raise StaticCameraLocalizationError("static metadata must declare rectification status raw, rectified, or unknown")
    if rectification_status == "rectified":
        if np.any(np.abs(distortion) > 1e-12):
            raise StaticCameraLocalizationError("rectified static imagery must have zero effective distortion")
        distortion_for_pnp: np.ndarray | None = None
    elif distortion_model in {"fisheye", "equidistant"}:
        raise StaticCameraLocalizationError("raw fisheye imagery must be rectified before pinhole PnP")
    else:
        distortion_for_pnp = distortion if distortion.size else None
    source_frame = metadata.get("source_frame")
    if not isinstance(source_frame, dict):
        raise StaticCameraLocalizationError("static metadata has no exact source_frame identity")
    for key in ("frame_id", "revision", "coordinate_frame", "units", "image_sha256"):
        if not isinstance(source_frame.get(key), str) or not source_frame[key]:
            raise StaticCameraLocalizationError(f"static source_frame lacks {key}")
    if source_frame["units"] not in {"m", "meters"}:
        raise StaticCameraLocalizationError("static source_frame units must be meters")
    if source_frame["image_sha256"] != _sha256(image_path):
        raise StaticCameraLocalizationError("static source_frame image_sha256 does not match the keyframe")
    return intrinsics, distortion_for_pnp, {
        "schema": metadata["schema"],
        "input_schema": metadata.get("input_schema", metadata["schema"]),
        "camera_id": camera_id,
        "image_size": [width, height],
        "distortion_model": distortion_model,
        "rectification": rectification_status,
        "distortion_provenance": str(
            metadata.get("distortion_provenance") or "explicit_static_image_contract"
        ),
        "source_frame": dict(source_frame),
    }


def _validate_excluded_ranges(
    ranges: tuple[tuple[int, int], ...],
    view_count: int,
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    for start, end in ranges:
        if int(start) != start or int(end) != end or start < 0 or end <= start or end > view_count:
            raise StaticCameraLocalizationError("excluded view ranges must be bounded half-open intervals")
        normalized.append((int(start), int(end)))
    normalized.sort()
    for (_, previous_end), (start, _) in zip(normalized, normalized[1:]):
        if start < previous_end:
            raise StaticCameraLocalizationError("excluded view ranges overlap")
    return tuple(normalized)


def _load_independent_scale_evidence(
    path: Path | None,
    *,
    source_frame: dict[str, Any],
) -> dict[str, Any]:
    if path is None:
        return {
            "status": "unverified_no_independent_scale_evidence",
            "path": None,
            "measurement_count": 0,
        }
    if not path.is_file():
        raise StaticCameraLocalizationError(f"independent scale evidence is missing: {path}")
    payload = _json(path)
    if payload.get("schema") != "noesis.pcf.static_camera.scale_evidence.v1":
        raise StaticCameraLocalizationError("independent scale evidence schema is unsupported")
    if payload.get("units") not in {"m", "meters"} or payload.get("alignment_fit_used") is True:
        raise StaticCameraLocalizationError("scale evidence must declare meters and no alignment fitting")
    evidence_frame = payload.get("source_frame")
    if not isinstance(evidence_frame, dict) or any(
        evidence_frame.get(key) != source_frame.get(key)
        for key in ("frame_id", "revision", "coordinate_frame")
    ):
        raise StaticCameraLocalizationError("scale evidence source frame does not match static image frame")
    rows = payload.get("measurements")
    if not isinstance(rows, list) or len(rows) < 2:
        raise StaticCameraLocalizationError("independent scale evidence needs at least two measurements")
    ratios: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("independent") is not True:
            raise StaticCameraLocalizationError(f"scale measurement {index} is not declared independent")
        if not isinstance(row.get("provenance"), str) or not row["provenance"]:
            raise StaticCameraLocalizationError(f"scale measurement {index} has no provenance")
        measured = float(row.get("measured_distance_m"))
        reference = float(row.get("candidate_distance_m"))
        if not math.isfinite(measured) or not math.isfinite(reference) or measured <= 0.0 or reference <= 0.0:
            raise StaticCameraLocalizationError(f"scale measurement {index} is not finite positive")
        ratios.append(measured / reference)
    ratio_median = float(np.median(ratios))
    ratio_p20 = float(np.percentile(ratios, 20.0))
    ratio_p80 = float(np.percentile(ratios, 80.0))
    relative_spread = (ratio_p80 - ratio_p20) / max(abs(ratio_median), 1e-9)
    return {
        "status": "passed" if relative_spread <= 0.10 and 0.95 <= ratio_median <= 1.05 else "failed",
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "measurement_count": len(ratios),
        "ratio_median": ratio_median,
        "ratio_p20": ratio_p20,
        "ratio_p80": ratio_p80,
        "relative_spread": relative_spread,
        "alignment_fitted_on_measurements": False,
    }


def _load_calibration_frame_binding(path: Path | None) -> dict[str, Any] | None:
    """Load an explicit assembly-to-calibration frame edge for replacement."""
    if path is None:
        return None
    if not path.is_file():
        raise StaticCameraLocalizationError(f"calibration frame binding is missing: {path}")
    payload = _json(path)
    if payload.get("schema") != "noesis.pcf.static_camera_frame_binding.v1":
        raise StaticCameraLocalizationError("calibration frame binding schema is unsupported")
    source_frame = payload.get("source_frame")
    target_frame = payload.get("target_frame")
    for label, frame in (("source_frame", source_frame), ("target_frame", target_frame)):
        if not isinstance(frame, dict) or any(
            not isinstance(frame.get(key), str) or not frame[key]
            for key in ("frame_id", "revision", "coordinate_frame")
        ):
            raise StaticCameraLocalizationError(f"calibration frame binding {label} is incomplete")
    transform = np.asarray(payload.get("target_from_source_col_major"), dtype=np.float64)
    if transform.size != 16 or not np.isfinite(transform).all():
        raise StaticCameraLocalizationError("calibration frame binding transform is malformed")
    matrix = transform.reshape((4, 4), order="F")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4)
        or not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=2e-4)
    ):
        raise StaticCameraLocalizationError("calibration frame binding is not a proper rigid transform")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "source_frame": dict(source_frame),
        "target_frame": dict(target_frame),
        "target_from_source_col_major": transform.astype(float).tolist(),
        "target_from_source_row_major": matrix.tolist(),
    }


def _pose_spatial_quality(
    objects: np.ndarray,
    images: np.ndarray,
    admitted: np.ndarray,
    image_shape: tuple[int, int, int],
) -> dict[str, Any]:
    selected_objects = objects[admitted]
    selected_images = images[admitted]
    spread, grid_bins = _spread_metrics(selected_images, image_shape[:2])
    centered = selected_objects - selected_objects.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False) if centered.size else np.zeros(3)
    rank = int(np.count_nonzero(singular > max(float(singular[0]) if singular.size else 0.0, 1e-9) * 1e-3)) if singular.size else 0
    return {
        "image_hull_area_fraction": spread,
        "image_grid_bins": grid_bins,
        "object_point_rank": rank,
        "object_extent_m": np.ptp(selected_objects, axis=0).tolist() if selected_objects.size else [0.0, 0.0, 0.0],
        "passed": bool(spread >= 0.02 and grid_bins >= 4 and rank >= 2),
    }


def _spread_metrics(
    image_points: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[float, int]:
    """Return normalized convex-hull area and occupied coarse image bins."""
    points = np.asarray(image_points, dtype=np.float32).reshape((-1, 2))
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if points.shape[0] < 3 or width <= 0 or height <= 0:
        return 0.0, 0
    hull = cv2.convexHull(points)
    area = float(cv2.contourArea(hull))
    normalized_area = area / float(width * height)
    columns = np.clip((points[:, 0] / width * 4.0).astype(np.int64), 0, 3)
    rows = np.clip((points[:, 1] / height * 4.0).astype(np.int64), 0, 3)
    occupied = int(np.unique(rows * 4 + columns).size)
    return normalized_area, occupied


def _public_view_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _write_localization_failure_report(
    output_dir: Path,
    *,
    camera_id: str,
    full_pcf_pose: bool,
    excluded_ranges: tuple[tuple[int, int], ...],
    frame_count: int,
    evaluated_views: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    inputs: dict[str, Any],
    failure_reason: str,
) -> Path:
    """Persist bounded per-view evidence before surfacing an admission failure."""
    output_dir.mkdir(parents=True, exist_ok=False)
    reason_counts: dict[str, int] = {}
    for row in evaluated_views:
        for reason in row.get("rejection_reasons", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    report = {
        "schema": "noesis.pcf.static_camera_anchor.failure.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed_insufficient_independent_views",
        "failure_reason": failure_reason,
        "camera_id": camera_id,
        "full_pose_requested": bool(full_pcf_pose),
        "configured_minimum_independent_views": 4,
        "excluded_view_ranges": [list(item) for item in excluded_ranges],
        "evidence": {
            "phone_view_count": frame_count,
            "evaluated_view_count": len(evaluated_views),
            "unevaluated_view_count": max(0, frame_count - len(evaluated_views)),
            "fit_eligible_view_count": sum(
                bool(row.get("accepted_for_fit")) for row in evaluated_views
            ),
            "candidate_view_count": len(candidates),
            "candidate_view_indices": [
                int(row["view_index"]) for row in candidates
            ],
            "heldout_evaluated_view_count": sum(
                bool(row.get("fit_excluded")) for row in evaluated_views
            ),
            "rejection_reason_counts": reason_counts,
            "per_view": [_public_view_row(row) for row in evaluated_views],
        },
        "inputs": inputs,
    }
    path = output_dir / "static_camera_anchor_failure.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _heading_deg(forward: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(forward[0]), float(forward[2]))))


def _angle_delta_deg(left: float, right: float) -> float:
    return float((left - right + 180.0) % 360.0 - 180.0)


def _rotation_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left.T @ right
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _proper_rotation_mean(
    rotations: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise StaticCameraLocalizationError("camera rotations are malformed")
    normalized = np.asarray(weights, dtype=np.float64).reshape(-1)
    if normalized.shape[0] != rotations.shape[0]:
        raise StaticCameraLocalizationError("camera rotation weights are malformed")
    total = float(np.sum(normalized))
    if not math.isfinite(total) or total <= 0.0:
        raise StaticCameraLocalizationError("camera rotation weights are empty")
    matrix = np.einsum("n,nij->ij", normalized / total, rotations)
    left, _, right_t = np.linalg.svd(matrix)
    correction = np.eye(3, dtype=np.float64)
    correction[2, 2] = float(np.linalg.det(left @ right_t))
    result = left @ correction @ right_t
    if not np.isfinite(result).all() or not math.isclose(
        float(np.linalg.det(result)), 1.0, abs_tol=1e-6
    ):
        raise StaticCameraLocalizationError("camera rotation consensus is improper")
    return result


def _pcf_floor_diagnostics(
    *,
    pcf_points: Path,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> dict[str, Any]:
    if not pcf_points.is_file():
        raise StaticCameraLocalizationError(
            f"PCF floor evidence is missing: {pcf_points}"
        )
    with np.load(pcf_points, allow_pickle=False) as payload:
        if "points" not in payload.files:
            raise StaticCameraLocalizationError("PCF floor evidence has no points")
        local_points = np.asarray(payload["points"], dtype=np.float64)
    if (
        local_points.ndim != 2
        or local_points.shape[1] != 3
        or local_points.shape[0] < 1_000
        or not np.isfinite(local_points).all()
    ):
        raise StaticCameraLocalizationError("PCF floor points are malformed")
    points = scale * (rotation @ local_points.T).T + translation

    cell_m = 0.15
    columns = np.floor(points[:, 0] / cell_m).astype(np.int64)
    rows = np.floor(points[:, 2] / cell_m).astype(np.int64)
    keys = np.column_stack((columns, rows))
    order = np.lexsort((rows, columns))
    keys = keys[order]
    heights = points[order, 1]
    starts = np.flatnonzero(
        np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)]
    )
    stops = np.r_[starts[1:], keys.shape[0]]
    envelopes: list[tuple[float, float, float]] = []
    for start, stop in zip(starts.tolist(), stops.tolist()):
        if stop - start < 5:
            continue
        envelopes.append(
            (
                (float(keys[start, 0]) + 0.5) * cell_m,
                (float(keys[start, 1]) + 0.5) * cell_m,
                float(np.median(heights[start:stop])),
            )
        )
    cells = np.asarray(envelopes, dtype=np.float64)
    if cells.shape[0] < 500:
        raise StaticCameraLocalizationError(
            "PCF floor evidence has too little independent spatial support"
        )

    values = cells[:, 2]
    ordered = np.sort(values)
    band_m = 0.10
    best_start = 0
    best_count = 0
    stop = 0
    for start, low in enumerate(ordered):
        stop = max(stop, start)
        while stop < ordered.size and ordered[stop] <= low + band_m:
            stop += 1
        if stop - start > best_count:
            best_start = start
            best_count = stop - start
    band_low = float(ordered[best_start])
    selected = (values >= band_low) & (values <= band_low + band_m)
    if int(np.count_nonzero(selected)) < 300:
        raise StaticCameraLocalizationError(
            "PCF floor evidence has no dominant floor-height mode"
        )
    design = np.column_stack(
        (cells[selected, 0], cells[selected, 1], np.ones(np.count_nonzero(selected)))
    )
    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        values[selected],
        rcond=None,
    )
    residuals = values[selected] - design @ coefficients
    a, b, c = (float(value) for value in coefficients)
    raw_normal = np.asarray([-a, 1.0, -b], dtype=np.float64)
    normal_norm = float(np.linalg.norm(raw_normal))
    normal = raw_normal / normal_norm
    offset_m = -c / normal_norm
    tilt_deg = float(
        math.degrees(math.acos(float(np.clip(normal[1], -1.0, 1.0))))
    )
    residual_p90_m = float(np.percentile(np.abs(residuals), 90.0))
    passed = bool(
        tilt_deg <= 2.0
        and residual_p90_m <= 0.06
        and int(np.count_nonzero(selected)) >= 300
    )
    if not passed:
        raise StaticCameraLocalizationError(
            "PCF floor evidence is not sufficiently level and repeatable"
        )
    return {
        "method": "gravity_aligned_spatial_cell_dominant_floor_plane",
        "status": "passed",
        "source_point_count": int(points.shape[0]),
        "spatial_cell_size_m": cell_m,
        "spatial_cell_count": int(cells.shape[0]),
        "floor_cell_count": int(np.count_nonzero(selected)),
        "floor_mode_band_m": [band_low, band_low + band_m],
        "normal_assembly": normal.tolist(),
        "offset_m": offset_m,
        "tilt_from_pcf_gravity_deg": tilt_deg,
        "residual_median_m": float(np.median(np.abs(residuals))),
        "residual_p90_m": residual_p90_m,
    }


def _circular_median_deg(values: list[float]) -> float:
    if not values:
        raise StaticCameraLocalizationError("camera-heading consensus is empty")
    candidates = np.asarray(values, dtype=np.float64)
    costs = [
        float(np.sum(np.abs([_angle_delta_deg(value, item) for item in values])))
        for value in candidates
    ]
    return float(candidates[int(np.argmin(costs))])


def _camera_to_world_from_extrinsics(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.size != 16 or not np.isfinite(matrix).all():
        raise StaticCameraLocalizationError("calibrated camera extrinsics are malformed")
    world_to_camera = matrix.reshape((4, 4), order="F")
    if not np.allclose(world_to_camera[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise StaticCameraLocalizationError(
            "calibrated camera extrinsics have an invalid homogeneous row"
        )
    rotation = world_to_camera[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4) or not np.isclose(
        float(np.linalg.det(rotation)), 1.0, atol=2e-4
    ):
        raise StaticCameraLocalizationError(
            "calibrated camera extrinsics are not a proper rigid transform"
        )
    return np.linalg.inv(world_to_camera)


def _yaw_rotation(delta_deg: float) -> np.ndarray:
    angle = math.radians(delta_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _mutual_matches(
    matcher: cv2.BFMatcher,
    fixed_descriptors: np.ndarray,
    phone_descriptors: np.ndarray,
    ratio: float,
) -> list[tuple[int, int]]:
    forward_pairs = matcher.knnMatch(fixed_descriptors, phone_descriptors, k=2)
    reverse_pairs = matcher.knnMatch(phone_descriptors, fixed_descriptors, k=2)
    forward = {
        int(pair[0].queryIdx): int(pair[0].trainIdx)
        for pair in forward_pairs
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    }
    reverse = {
        int(pair[0].queryIdx): int(pair[0].trainIdx)
        for pair in reverse_pairs
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    }
    return [
        (fixed_index, phone_index)
        for fixed_index, phone_index in forward.items()
        if reverse.get(phone_index) == fixed_index
    ]


def _fundamental_inliers(
    pairs: list[tuple[int, int]],
    fixed_keypoints: list[cv2.KeyPoint],
    phone_keypoints: list[cv2.KeyPoint],
) -> list[tuple[int, int]]:
    if len(pairs) < 12:
        return []
    fixed_pixels = np.asarray(
        [fixed_keypoints[left].pt for left, _ in pairs], dtype=np.float64
    )
    phone_pixels = np.asarray(
        [phone_keypoints[right].pt for _, right in pairs], dtype=np.float64
    )
    _, mask = cv2.findFundamentalMat(
        fixed_pixels,
        phone_pixels,
        cv2.FM_RANSAC,
        2.0,
        0.999,
    )
    if mask is None:
        return []
    return [pair for pair, keep in zip(pairs, mask.reshape(-1)) if bool(keep)]


def _nearest_pcf_point(
    *,
    phone_pixel: tuple[float, float],
    phone_size: tuple[int, int],
    world_points: np.ndarray,
    mask: np.ndarray,
    confidence: np.ndarray,
    agreement: np.ndarray,
) -> np.ndarray | None:
    grid_height, grid_width = mask.shape
    phone_width, phone_height = phone_size
    x = int(round(phone_pixel[0] * (grid_width - 1) / (phone_width - 1)))
    y = int(round(phone_pixel[1] * (grid_height - 1) / (phone_height - 1)))
    best: tuple[float, int, int] | None = None
    for radius in range(4):
        for row in range(max(0, y - radius), min(grid_height, y + radius + 1)):
            for column in range(
                max(0, x - radius), min(grid_width, x + radius + 1)
            ):
                if (
                    not bool(mask[row, column])
                    or float(confidence[row, column]) <= 0.12
                    or not np.isfinite(world_points[row, column]).all()
                ):
                    continue
                score = float((column - x) ** 2 + (row - y) ** 2)
                if not bool(agreement[row, column]):
                    score += 0.10
                if best is None or score < best[0]:
                    best = (score, row, column)
        if best is not None:
            break
    if best is None:
        return None
    return np.asarray(world_points[best[1], best[2]], dtype=np.float64)


def _load_model_rgb(payload: Any, valid_mask: np.ndarray) -> np.ndarray:
    """Return the RGB image whose pixels index the serialized PCF grid."""
    model_rgb = np.asarray(payload.get("model_rgb"))
    if (
        model_rgb.ndim != 3
        or model_rgb.shape[2] != 3
        or model_rgb.shape[:2] != valid_mask.shape
    ):
        raise StaticCameraLocalizationError(
            "PCF raw view model_rgb does not match its depth grid"
        )
    if model_rgb.dtype != np.uint8:
        model_rgb = np.clip(
            model_rgb * 255.0
            if model_rgb.size and float(np.nanmax(model_rgb)) <= 1.5
            else model_rgb,
            0,
            255,
        ).astype(np.uint8)
    return model_rgb


def _render_reprojection(
    *,
    output_path: Path,
    static_image: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_assembly: np.ndarray,
    assembly_npz: Path,
    owner_room_id: int,
) -> dict[str, int]:
    with np.load(assembly_npz) as payload:
        points = np.asarray(payload["points"], dtype=np.float64)
        colors = np.asarray(payload["colors"], dtype=np.uint8)
        owners = (
            np.asarray(payload["owner_room_id"], dtype=np.uint8)
            if "owner_room_id" in payload.files
            else None
        )
    selected = (
        owners == int(owner_room_id)
        if owners is not None
        else np.ones(points.shape[0], dtype=bool)
    )
    points = points[selected][::2]
    colors = colors[selected][::2, ::-1]
    camera_from_assembly = np.linalg.inv(camera_to_assembly)
    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    camera_points = (camera_from_assembly @ homogeneous.T).T[:, :3]
    valid = (
        np.isfinite(camera_points).all(axis=1)
        & (camera_points[:, 2] > 0.10)
        & (camera_points[:, 2] < 12.0)
    )
    camera_points = camera_points[valid]
    colors = colors[valid]
    projected = (intrinsics @ camera_points.T).T
    pixels = projected[:, :2] / projected[:, 2, None]
    width = 960
    height = 540
    pixels *= np.asarray(
        [width / static_image.shape[1], height / static_image.shape[0]],
        dtype=np.float64,
    )
    columns = np.rint(pixels[:, 0]).astype(np.int32)
    rows = np.rint(pixels[:, 1]).astype(np.int32)
    inside = (
        (columns >= 0)
        & (columns < width)
        & (rows >= 0)
        & (rows < height)
    )
    columns = columns[inside]
    rows = rows[inside]
    depths = camera_points[inside, 2]
    colors = colors[inside]
    order = np.argsort(depths)[::-1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[rows[order], columns[order]] = colors[order]
    measured = np.any(canvas != 0, axis=2)
    canvas = cv2.dilate(canvas, np.ones((3, 3), dtype=np.uint8), iterations=1)
    reference = cv2.resize(static_image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.putText(
        reference,
        "STATIC CAMERA",
        (15, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "PCF THROUGH SOLVED STATIC POSE",
        (15, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    combined = np.vstack((reference, canvas))
    if not cv2.imwrite(str(output_path), combined):
        raise StaticCameraLocalizationError(f"could not write {output_path}")
    return {
        "projected_point_count": int(columns.size),
        "measured_pixel_count": int(np.count_nonzero(measured)),
    }


def localize(
    *,
    scan_dir: Path,
    raw_root: Path,
    static_keyframe: Path,
    static_metadata: Path,
    source_world_manifest: Path,
    camera_calibration: Path,
    camera_id: str,
    output_dir: Path,
    assembly_npz: Path | None,
    owner_room_id: int,
    pcf_points: Path | None = None,
    full_pcf_pose: bool = False,
    excluded_view_ranges: tuple[tuple[int, int], ...] = (),
    independent_scale_evidence: Path | None = None,
    calibration_frame_binding: Path | None = None,
    static_depth_input: Path | None = None,
    static_depth_output: Path | None = None,
    static_reference_template: Path | None = None,
    static_reference_output: Path | None = None,
    replacement_calibration_output: Path | None = None,
    backup_calibration_root: Path | None = None,
) -> dict[str, Any]:
    if replacement_calibration_output is None and (
        static_depth_input is not None or static_depth_output is not None
    ):
        raise StaticCameraLocalizationError(
            "static depth regeneration requires replacement_calibration_output"
        )
    if (static_depth_input is None) != (static_depth_output is None):
        raise StaticCameraLocalizationError(
            "static depth regeneration requires both input and output paths"
        )
    if (static_reference_template is None) != (static_reference_output is None):
        raise StaticCameraLocalizationError(
            "static reference materialization requires both template and output paths"
        )
    if static_reference_template is not None and static_depth_output is None:
        raise StaticCameraLocalizationError(
            "static reference materialization requires regenerated static depth"
        )
    required = [
        static_keyframe,
        static_metadata,
        source_world_manifest,
        camera_calibration,
    ]
    if assembly_npz is not None:
        required.append(assembly_npz)
    if calibration_frame_binding is not None:
        required.append(calibration_frame_binding)
    if full_pcf_pose:
        if pcf_points is None:
            raise StaticCameraLocalizationError(
                "full-PCF pose requires fused PCF floor points"
            )
        required.append(pcf_points)
    for path in required:
        if not path.is_file():
            raise StaticCameraLocalizationError(f"required input is missing: {path}")
    frame_paths = sorted((scan_dir / "frames").glob("frame_*.jpg"))
    raw_paths = sorted(raw_root.glob("view_*.npz"))
    if len(frame_paths) < 2 or len(frame_paths) != len(raw_paths):
        raise StaticCameraLocalizationError(
            "prepared phone frames and PCF raw views are incomplete or mismatched"
        )
    if output_dir.exists():
        raise StaticCameraLocalizationError(f"output already exists: {output_dir}")

    metadata = _json(static_metadata)
    static_bgr = cv2.imread(str(static_keyframe), cv2.IMREAD_COLOR)
    if static_bgr is None:
        raise StaticCameraLocalizationError("static keyframe is unreadable")
    intrinsics, distortion, static_contract = _validate_static_image_contract(
        metadata,
        static_bgr,
        image_path=static_keyframe,
        camera_id=camera_id,
        metadata_path=static_metadata,
    )
    calibration_binding = _load_calibration_frame_binding(calibration_frame_binding)
    excluded_ranges = _validate_excluded_ranges(excluded_view_ranges, len(frame_paths))
    excluded_indices = {
        index
        for start, end in excluded_ranges
        for index in range(start, end)
    }
    world_manifest = _json(source_world_manifest)
    alignment = world_manifest.get("alignment")
    if not isinstance(alignment, dict):
        raise StaticCameraLocalizationError("source world manifest has no alignment")
    rotation = np.asarray(alignment.get("rotation_row_major"), dtype=np.float64)
    translation = np.asarray(alignment.get("translation"), dtype=np.float64)
    scale = float(alignment.get("scale"))
    if (
        rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise StaticCameraLocalizationError("source local-to-assembly Sim3 is malformed")

    calibration = _json(camera_calibration).get("cameras", {})
    camera = calibration.get(camera_id) if isinstance(calibration, dict) else None
    if not isinstance(camera, dict):
        raise StaticCameraLocalizationError(f"camera calibration is missing {camera_id}")
    calibrated_camera_to_world = _camera_to_world_from_extrinsics(camera.get("E"))
    calibrated_image_size = camera.get("image_size")
    if calibrated_image_size is not None and calibrated_image_size != static_contract["image_size"]:
        raise StaticCameraLocalizationError(
            "static image resolution does not match calibrated camera image_size"
        )
    static_gray = cv2.cvtColor(static_bgr, cv2.COLOR_BGR2GRAY)
    cv2.setRNGSeed(7)
    sift = cv2.SIFT_create(nfeatures=10_000, contrastThreshold=0.01)
    fixed_keypoints, fixed_descriptors = sift.detectAndCompute(static_gray, None)
    if fixed_descriptors is None or len(fixed_keypoints) < 100:
        raise StaticCameraLocalizationError("static keyframe has too few SIFT features")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    candidates: list[dict[str, Any]] = []
    evaluated_views: list[dict[str, Any]] = []
    ratio = 0.82
    for view_index, (frame_path, raw_path) in enumerate(zip(frame_paths, raw_paths)):
        # The serialized model_rgb is the image that produced the depth grid.
        # Consensus fusion reprojects it onto common rays, so the prepared
        # phone frame is not pixel-aligned with world_points even when the
        # two images have the same aspect ratio.  Keep frame_path in the
        # one-to-one input contract, but use model_rgb for feature pixels and
        # nearest-point lookup so both coordinates share the raw grid.
        with np.load(raw_path) as payload:
            world_points = np.asarray(payload["world_points"], dtype=np.float64)
            valid_mask = np.asarray(payload["mask"], dtype=bool)
            confidence = np.asarray(payload["confidence"], dtype=np.float64)
            if "cross_model_agreement" not in payload.files:
                raise StaticCameraLocalizationError(
                    "PCF raw view lacks cross_model_agreement; use the conditioned consensus raw output"
                )
            agreement = np.asarray(payload["cross_model_agreement"], dtype=bool)
            model_rgb = _load_model_rgb(payload, valid_mask)
        phone_gray = cv2.cvtColor(model_rgb, cv2.COLOR_RGB2GRAY)
        phone_keypoints, phone_descriptors = sift.detectAndCompute(phone_gray, None)
        if phone_descriptors is None:
            continue
        mutual = _mutual_matches(
            matcher, fixed_descriptors, phone_descriptors, ratio
        )
        geometric = _fundamental_inliers(
            mutual, fixed_keypoints, phone_keypoints
        )
        if len(geometric) < 8:
            continue
        object_points: list[np.ndarray] = []
        image_points: list[tuple[float, float]] = []
        phone_size = (model_rgb.shape[1], model_rgb.shape[0])
        for fixed_index, phone_index in geometric:
            local_point = _nearest_pcf_point(
                phone_pixel=phone_keypoints[phone_index].pt,
                phone_size=phone_size,
                world_points=world_points,
                mask=valid_mask,
                confidence=confidence,
                agreement=agreement,
            )
            if local_point is None:
                continue
            object_points.append(scale * (rotation @ local_point) + translation)
            image_points.append(fixed_keypoints[fixed_index].pt)
        if len(object_points) < 8:
            continue
        objects = np.asarray(object_points, dtype=np.float64)
        images = np.asarray(image_points, dtype=np.float64)
        solved, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            objects,
            images,
            intrinsics,
            distortion,
            iterationsCount=20_000,
            reprojectionError=6.0,
            confidence=0.99999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not solved or inliers is None or len(inliers) < 8:
            continue
        admitted = np.asarray(inliers, dtype=np.int64).reshape(-1)
        rotation_vector, translation_vector = cv2.solvePnPRefineLM(
            objects[admitted],
            images[admitted],
            intrinsics,
            distortion,
            rotation_vector,
            translation_vector,
        )
        projected, _ = cv2.projectPoints(
            objects[admitted],
            rotation_vector,
            translation_vector,
            intrinsics,
            distortion,
        )
        errors = np.linalg.norm(
            projected.reshape((-1, 2)) - images[admitted], axis=1
        )
        camera_from_world = np.eye(4, dtype=np.float64)
        camera_from_world[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
        camera_from_world[:3, 3] = np.asarray(
            translation_vector, dtype=np.float64
        ).reshape(3)
        camera_to_world = np.linalg.inv(camera_from_world)
        center = camera_to_world[:3, 3]
        heading = _heading_deg(camera_to_world[:3, 2])
        median_error = float(np.median(errors))
        p80_error = float(np.percentile(errors, 80.0))
        spatial_quality = _pose_spatial_quality(objects, images, admitted, static_bgr.shape)
        camera_points = (
            camera_from_world[:3, :3] @ objects[admitted].T
        ).T + camera_from_world[:3, 3]
        cheirality_fraction = float(np.count_nonzero(camera_points[:, 2] > 0.05) / max(1, len(camera_points)))
        rejection_reasons: list[str] = []
        if len(admitted) < 8:
            rejection_reasons.append("pnp_inlier_count_below_8")
        if median_error > 5.0:
            rejection_reasons.append("reprojection_median_exceeded_5px")
        if p80_error > 6.0:
            rejection_reasons.append("reprojection_p80_exceeded_6px")
        if not 0.40 <= float(center[1]) <= 3.50:
            rejection_reasons.append("camera_center_height_outside_0.40_3.50m")
        if not spatial_quality["passed"]:
            rejection_reasons.append("spatial_support_gate_failed")
        if cheirality_fraction < 0.95:
            rejection_reasons.append("cheirality_fraction_below_0.95")
        if view_index in excluded_indices:
            rejection_reasons.append("excluded_from_fit")
        accepted = not rejection_reasons
        row = {
                    "view_index": view_index,
                    "mutual_match_count": len(mutual),
                    "fundamental_inlier_count": len(geometric),
                    "depth_supported_count": len(objects),
                    "pnp_inlier_count": int(len(admitted)),
                    "reprojection_median_px": median_error,
                    "reprojection_p80_px": p80_error,
                    "camera_center_assembly_m": center.tolist(),
                    "camera_heading_deg": heading,
                    "camera_to_assembly_row_major": camera_to_world.tolist(),
                    "fit_excluded": view_index in excluded_indices,
                    "spatial_support": spatial_quality,
                    "cheirality_fraction": cheirality_fraction,
                    "rejection_reasons": rejection_reasons,
                    "accepted_for_fit": bool(accepted and view_index not in excluded_indices),
                    "_objects": objects,
                    "_images": images,
                    "_camera_to_world": camera_to_world,
                }
        evaluated_views.append(row)
        if accepted and view_index not in excluded_indices:
            candidates.append(row)

    if len(candidates) < 4:
        failure_reason = (
            f"only {len(candidates)} independent phone views localized the static camera"
        )
        _write_localization_failure_report(
            output_dir,
            camera_id=camera_id,
            full_pcf_pose=full_pcf_pose,
            excluded_ranges=excluded_ranges,
            frame_count=len(frame_paths),
            evaluated_views=evaluated_views,
            candidates=candidates,
            inputs={
                "scan_dir": str(scan_dir.resolve()),
                "raw_root": str(raw_root.resolve()),
                "static_keyframe_sha256": _sha256(static_keyframe),
                "static_metadata_sha256": _sha256(static_metadata),
                "source_world_manifest_sha256": _sha256(source_world_manifest),
                "camera_calibration_sha256": _sha256(camera_calibration),
                "pcf_points_sha256": (
                    _sha256(pcf_points) if pcf_points is not None else None
                ),
            },
            failure_reason=failure_reason,
        )
        raise StaticCameraLocalizationError(failure_reason)
    centers = np.asarray(
        [row["camera_center_assembly_m"] for row in candidates], dtype=np.float64
    )
    camera_poses = np.asarray(
        [row["camera_to_assembly_row_major"] for row in candidates],
        dtype=np.float64,
    )
    rotations = camera_poses[:, :3, :3]
    headings = [float(row["camera_heading_deg"]) for row in candidates]
    if full_pcf_pose:
        translation_pairwise = np.linalg.norm(
            centers[:, None, :] - centers[None, :, :], axis=2
        )
        rotation_pairwise = np.asarray(
            [
                [
                    _rotation_error_deg(left, right)
                    for right in rotations
                ]
                for left in rotations
            ],
            dtype=np.float64,
        )
        medoid_cost = np.median(
            translation_pairwise / 0.20 + rotation_pairwise / 4.0,
            axis=1,
        )
        medoid_index = int(np.argmin(medoid_cost))
        admitted_indices = [
            index
            for index in range(len(candidates))
            if float(translation_pairwise[index, medoid_index]) <= 0.35
            and float(rotation_pairwise[index, medoid_index]) <= 6.0
        ]
    else:
        center_median = np.median(centers, axis=0)
        heading_median = _circular_median_deg(headings)
        horizontal_deviation = np.linalg.norm(
            centers[:, (0, 2)] - center_median[[0, 2]], axis=1
        )
        yaw_deviation = np.abs(
            np.asarray(
                [_angle_delta_deg(value, heading_median) for value in headings],
                dtype=np.float64,
            )
        )
        admitted_indices = [
            index
            for index, (horizontal, yaw) in enumerate(
                zip(horizontal_deviation, yaw_deviation)
            )
            if float(horizontal) <= 1.0 and float(yaw) <= 12.0
        ]
    admitted_candidates = [candidates[index] for index in admitted_indices]
    if len(admitted_candidates) < 4:
        raise StaticCameraLocalizationError(
            "static-camera pose candidates do not form a repeatable consensus"
        )
    admitted_centers = centers[admitted_indices]
    admitted_rotations = rotations[admitted_indices]
    admitted_headings = [headings[index] for index in admitted_indices]
    center = np.median(admitted_centers, axis=0)
    if full_pcf_pose:
        pose_weights = np.asarray(
            [
                math.sqrt(float(row["pnp_inlier_count"]))
                / max(0.50, float(row["reprojection_median_px"]))
                for row in admitted_candidates
            ],
            dtype=np.float64,
        )
        consensus_rotation = _proper_rotation_mean(
            admitted_rotations,
            pose_weights,
        )
        heading = _heading_deg(consensus_rotation[:, 2])
    else:
        consensus_rotation = np.eye(3, dtype=np.float64)
        heading = _circular_median_deg(admitted_headings)
    view_count = len(frame_paths)
    view_indices = [int(row["view_index"]) for row in admitted_candidates]
    if min(view_indices) > view_count // 4 or max(view_indices) < 3 * view_count // 4:
        raise StaticCameraLocalizationError(
            "static-camera localization lacks independent early/late walk support"
        )
    calibrated_heading = _heading_deg(calibrated_camera_to_world[:3, 2])
    camera_to_assembly = np.eye(4, dtype=np.float64)
    if full_pcf_pose:
        camera_to_assembly[:3, :3] = consensus_rotation
    else:
        center[1] = calibrated_camera_to_world[1, 3]
        yaw_delta = _angle_delta_deg(heading, calibrated_heading)
        camera_to_assembly[:3, :3] = (
            _yaw_rotation(yaw_delta) @ calibrated_camera_to_world[:3, :3]
        )
    camera_to_assembly[:3, 3] = center
    translation_deviation = np.linalg.norm(
        admitted_centers - center, axis=1
    )
    horizontal_deviation = np.linalg.norm(
        admitted_centers[:, (0, 2)] - center[[0, 2]], axis=1
    )
    yaw_deviation = np.abs(
        np.asarray(
            [_angle_delta_deg(value, heading) for value in admitted_headings],
            dtype=np.float64,
        )
    )
    old_displacement = float(
        np.linalg.norm(calibrated_camera_to_world[:3, 3] - center)
    )
    rotation_deviation = np.asarray(
        [
            _rotation_error_deg(value, camera_to_assembly[:3, :3])
            for value in admitted_rotations
        ],
        dtype=np.float64,
    )
    floor_evidence: dict[str, Any] | None = None
    if full_pcf_pose:
        assert pcf_points is not None
        floor_evidence = _pcf_floor_diagnostics(
            pcf_points=pcf_points,
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        floor_normal = np.asarray(
            floor_evidence["normal_assembly"], dtype=np.float64
        )
        floor_evidence["camera_height_over_fitted_floor_m"] = float(
            floor_normal @ center + float(floor_evidence["offset_m"])
        )
    excluded_evaluation: list[dict[str, Any]] = []
    excluded_camera_from_world = np.linalg.inv(camera_to_assembly)
    for row in evaluated_views:
        if not row.get("fit_excluded"):
            continue
        objects = np.asarray(row["_objects"], dtype=np.float64)
        images = np.asarray(row["_images"], dtype=np.float64)
        camera_points = (excluded_camera_from_world[:3, :3] @ objects.T).T + excluded_camera_from_world[:3, 3]
        projected, _ = cv2.projectPoints(
            objects,
            cv2.Rodrigues(excluded_camera_from_world[:3, :3])[0],
            excluded_camera_from_world[:3, 3],
            intrinsics,
            distortion,
        )
        errors = np.linalg.norm(projected.reshape((-1, 2)) - images, axis=1)
        excluded_evaluation.append(
            {
                "view_index": int(row["view_index"]),
                "correspondence_count": int(len(objects)),
                "reprojection_median_px": float(np.median(errors)) if errors.size else None,
                "reprojection_p80_px": float(np.percentile(errors, 80.0)) if errors.size else None,
                "cheirality_fraction": float(np.count_nonzero(camera_points[:, 2] > 0.05) / max(1, len(camera_points))),
                "passed": bool(
                    errors.size >= 8
                    and float(np.percentile(errors, 80.0)) <= 8.0
                    and float(np.count_nonzero(camera_points[:, 2] > 0.05) / max(1, len(camera_points))) >= 0.95
                ),
            }
        )
    independent_scale = _load_independent_scale_evidence(
        independent_scale_evidence,
        source_frame=static_contract["source_frame"],
    )
    excluded_status = (
        "passed"
        if excluded_evaluation and all(row["passed"] for row in excluded_evaluation)
        else "failed"
        if excluded_evaluation
        else "unavailable_no_excluded_views"
    )
    acceptance_checks = {
        "measured_pose_consensus": len(admitted_candidates) >= 4,
        "fit_support_spans_early_and_late": min(view_indices) <= view_count // 4 and max(view_indices) >= 3 * view_count // 4,
        "se3_dispersion_within_gate": bool(
            float(np.percentile(translation_deviation, 80.0)) <= 0.35
            and float(np.percentile(rotation_deviation, 80.0)) <= 6.0
        ),
        "cheirality_and_spatial_support": all(
            float(row.get("cheirality_fraction", 0.0)) >= 0.95
            and bool((row.get("spatial_support") or {}).get("passed"))
            for row in admitted_candidates
        ),
        "floor_geometry": bool(
            floor_evidence is not None
            and math.isfinite(float(floor_evidence.get("camera_height_over_fitted_floor_m", float("nan"))))
            and 0.30 <= float(floor_evidence.get("camera_height_over_fitted_floor_m", -1.0)) <= 4.0
        ) if full_pcf_pose else False,
        "excluded_view_evaluation": excluded_status == "passed",
        "independent_metric_scale": independent_scale.get("status") == "passed",
        "static_image_calibration_provenance": (
            static_contract["input_schema"] == "noesis.pcf.static_camera_image.v1"
            and static_contract["distortion_provenance"]
            == "explicit_static_image_contract"
        ),
        "calibration_frame_binding_for_replacement": (
            replacement_calibration_output is None or calibration_binding is not None
        ),
        "full_pose_requested": bool(full_pcf_pose),
    }
    measured_acceptance = bool(
        full_pcf_pose
        and all(acceptance_checks.values())
    )
    output_dir.mkdir(parents=True)
    reprojection: dict[str, Any] | None = None
    if assembly_npz is not None:
        reprojection_path = output_dir / "static_camera_reprojection_review.jpg"
        reprojection = {
            **_render_reprojection(
                output_path=reprojection_path,
                static_image=static_bgr,
                intrinsics=intrinsics,
                camera_to_assembly=camera_to_assembly,
                assembly_npz=assembly_npz,
                owner_room_id=owner_room_id,
            ),
            "relative_path": reprojection_path.name,
            "sha256": _sha256(reprojection_path),
        }
    report = {
        "schema": (
            "noesis.pcf.static_camera_anchor.v2"
            if full_pcf_pose
            else "noesis.pcf.static_camera_anchor.v1"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_review_anchor" if measured_acceptance else "review_only_unverified",
        "accepted_for_canonical_use": measured_acceptance,
        "camera_id": camera_id,
        "coordinate_frame": (
            "pcf_assembly_metric_world_m"
            if full_pcf_pose
            else "family_accepted_backend_world_m"
        ),
        "pose_convention": "opencv_cam2world_x_right_y_down_z_forward",
        "method": (
            "mutual_sift_fundamental_gate_pcf_depth_pnp_per_view_"
            + (
                "view_balanced_full_se3_consensus"
                if full_pcf_pose
                else "view_balanced_ground_pose_consensus"
            )
        ),
        "constraints": {
            "metric_scale_fixed": True,
            "gravity_fixed": True,
            "floor_y_m": 0.0,
            "floor_camera_height_fixed_m": (
                None if full_pcf_pose else float(center[1])
            ),
            "calibrated_pitch_roll_preserved": not full_pcf_pose,
            "full_pnp_translation_used": full_pcf_pose,
            "full_pnp_rotation_used": full_pcf_pose,
            "whole_cloud_icp_used": False,
            "bounding_box_anchor_used": False,
            "manual_scene_nudge_used": False,
        },
        "camera_to_assembly_row_major": camera_to_assembly.tolist(),
        "camera_to_assembly_col_major": camera_to_assembly.reshape(
            -1, order="F"
        ).tolist(),
        "static_image_source_frame": dict(static_contract["source_frame"]),
        "static_image_contract": dict(static_contract),
        "calibration_frame_binding": calibration_binding,
        "assembly_to_camera_row_major": np.linalg.inv(camera_to_assembly).tolist(),
        "device_reference_camera_to_assembly_col_major": (
            calibrated_camera_to_world.reshape(-1, order="F").tolist()
        ),
        "estimate": {
            "camera_center_assembly_m": center.tolist(),
            "camera_heading_deg": heading,
            "calibrated_heading_deg": calibrated_heading,
            "yaw_correction_from_legacy_backend_pose_deg": _angle_delta_deg(
                heading, calibrated_heading
            ),
            "legacy_camera_center_displacement_m": old_displacement,
            "translation_uncertainty_p80_m": float(
                np.percentile(
                    translation_deviation if full_pcf_pose else horizontal_deviation,
                    80.0,
                )
            ),
            "yaw_uncertainty_p80_deg": float(np.percentile(yaw_deviation, 80.0)),
            "rotation_uncertainty_p80_deg": float(
                np.percentile(rotation_deviation, 80.0)
            ),
        },
        "evidence": {
            "phone_view_count": view_count,
            "admitted_view_count": len(admitted_candidates),
            "admitted_view_indices": view_indices,
            "early_view_supported": min(view_indices) <= view_count // 4,
            "late_view_supported": max(view_indices) >= 3 * view_count // 4,
            "total_pnp_inlier_count": int(
                sum(int(row["pnp_inlier_count"]) for row in admitted_candidates)
            ),
            "per_view": [_public_view_row(row) for row in admitted_candidates],
            "excluded_view_evaluation": excluded_evaluation,
            "excluded_view_ranges": [list(item) for item in excluded_ranges],
            "all_evaluated_view_count": len(evaluated_views),
            "pcf_floor": floor_evidence,
            "independent_scale": independent_scale,
        },
        "acceptance": {
            "status": "passed" if measured_acceptance else "failed_or_unverified",
            "checks": acceptance_checks,
            "full_pcf_pose_is_request_only": True,
            "canonical_use_requires_all_measured_checks": True,
        },
        "inputs": {
            "static_keyframe_sha256": _sha256(static_keyframe),
            "static_metadata_sha256": _sha256(static_metadata),
            "source_world_manifest_sha256": _sha256(source_world_manifest),
            "camera_calibration_sha256": _sha256(camera_calibration),
            "assembly_npz_sha256": (
                _sha256(assembly_npz) if assembly_npz is not None else None
            ),
            "pcf_points_sha256": (
                _sha256(pcf_points) if pcf_points is not None else None
            ),
            "calibration_frame_binding_sha256": (
                _sha256(calibration_frame_binding)
                if calibration_frame_binding is not None
                else None
            ),
        },
        "reprojection_review": reprojection,
    }
    report_path = output_dir / "static_camera_anchor_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if replacement_calibration_output is not None:
        if backup_calibration_root is None:
            raise StaticCameraLocalizationError(
                "replacement output requires an explicit backup_calibration_root"
            )
        if calibration_binding is None:
            raise StaticCameraLocalizationError(
                "replacement output requires an explicit calibration_frame_binding"
            )
        try:
            replacement = write_calibration_replacement(
                camera_calibration,
                replacement_calibration_output,
                camera_id=camera_id,
                camera_to_world=camera_to_assembly,
                provenance={
                    **report,
                    "frame_identity": calibration_binding["target_frame"],
                    "assembly_frame_identity": calibration_binding["source_frame"],
                    "calibration_frame_binding_sha256": calibration_binding["sha256"],
                    "calibration_frame_revision": calibration_binding["target_frame"]["revision"],
                    "acceptance_report_sha256": _sha256(report_path),
                },
                calibration_from_assembly=calibration_binding["target_from_source_row_major"],
                backup_root=backup_calibration_root,
                static_depth_input=static_depth_input,
                static_depth_output=static_depth_output,
                static_reference_template=static_reference_template,
                static_reference_output=static_reference_output,
            )
        except CalibrationReplacementError as exc:
            raise StaticCameraLocalizationError(str(exc)) from exc
        report["replacement"] = replacement
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--static-keyframe", type=Path, required=True)
    parser.add_argument("--static-metadata", type=Path, required=True)
    parser.add_argument("--source-world-manifest", type=Path, required=True)
    parser.add_argument("--camera-calibration", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assembly-npz", type=Path)
    parser.add_argument("--owner-room-id", type=int, default=1)
    parser.add_argument("--pcf-points", type=Path)
    parser.add_argument("--full-pcf-pose", action="store_true")
    parser.add_argument(
        "--exclude-range",
        action="append",
        default=[],
        metavar="START:END",
        help="half-open temporal view range excluded from fitting but evaluated after consensus",
    )
    parser.add_argument("--independent-scale-evidence", type=Path)
    parser.add_argument("--calibration-frame-binding", type=Path)
    parser.add_argument("--static-depth-input", type=Path)
    parser.add_argument("--static-depth-output", type=Path)
    parser.add_argument("--static-reference-template", type=Path)
    parser.add_argument("--static-reference-output", type=Path)
    parser.add_argument("--replacement-calibration-output", type=Path)
    parser.add_argument("--backup-calibration-root", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        excluded_ranges = tuple(
            (int(raw.split(":", 1)[0]), int(raw.split(":", 1)[1]))
            for raw in args.exclude_range
            if ":" in raw
        )
        if len(excluded_ranges) != len(args.exclude_range):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise SystemExit("--exclude-range must use START:END") from exc
    report = localize(
        scan_dir=args.scan_dir,
        raw_root=args.raw_root,
        static_keyframe=args.static_keyframe,
        static_metadata=args.static_metadata,
        source_world_manifest=args.source_world_manifest,
        camera_calibration=args.camera_calibration,
        camera_id=args.camera_id,
        output_dir=args.output_dir,
        assembly_npz=args.assembly_npz,
        owner_room_id=args.owner_room_id,
        pcf_points=args.pcf_points,
        full_pcf_pose=args.full_pcf_pose,
        excluded_view_ranges=excluded_ranges,
        independent_scale_evidence=args.independent_scale_evidence,
        calibration_frame_binding=args.calibration_frame_binding,
        static_depth_input=args.static_depth_input,
        static_depth_output=args.static_depth_output,
        static_reference_template=args.static_reference_template,
        static_reference_output=args.static_reference_output,
        replacement_calibration_output=args.replacement_calibration_output,
        backup_calibration_root=args.backup_calibration_root,
    )
    print(
        json.dumps(
            {
                "camera_id": report["camera_id"],
                "status": report["status"],
                "admitted_view_count": report["evidence"]["admitted_view_count"],
                "camera_center_assembly_m": report["estimate"][
                    "camera_center_assembly_m"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
