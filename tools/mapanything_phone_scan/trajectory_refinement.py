"""Bounded, review-only trajectory refinement for retained phone-walk outputs.

This module owns the CPU side of WO-3.  It retrieves a small set of plausible
nonadjacent image revisits, verifies them with real image matches and retained
depth, and optionally materializes a corrected copy of the retained raw outputs.
The source raw directory is never changed.  A missing or rejected constraint
leaves the ordinary DA3 path available to the caller.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import cv2
import numpy as np

from .prepared_frame_identity import prepared_frame_identity


TRAJECTORY_REFINEMENT_SCHEMA = "noesis.phone_walk.trajectory_refinement.v1"
RAW_REQUIRED_KEYS = (
    "world_points",
    "depth_z",
    "confidence",
    "mask",
    "camera_pose",
    "intrinsics",
    "model_rgb",
)


class TrajectoryRefinementError(RuntimeError):
    """Raised when an explicit trajectory-refinement input is malformed."""


@dataclass(frozen=True)
class TrajectoryRefinementSettings:
    """Finite CPU budgets and conservative visual-constraint gates."""

    min_nonadjacent_gap: int = 8
    max_candidate_pairs: int = 64
    max_features: int = 1200
    ratio_test: float = 0.75
    min_matches: int = 24
    min_2d_inliers: int = 12
    min_inlier_fraction: float = 0.45
    min_inlier_spread_fraction: float = 0.02
    min_3d_matches: int = 8
    min_3d_inliers: int = 8
    min_3d_inlier_fraction: float = 0.55
    max_3d_residual_m: float = 0.20
    max_3d_p80_residual_m: float = 0.30
    ransac_iterations: int = 128
    max_pose_translation_change_m: float = 0.50
    max_pose_rotation_change_deg: float = 25.0
    max_verified_constraint_translation_residual_m: float = 0.35
    max_verified_constraint_rotation_residual_deg: float = 20.0
    max_withheld_translation_residual_m: float = 0.35
    max_withheld_rotation_residual_deg: float = 20.0
    # Retained PCF evidence is quantized at 4 cm; use half a voxel as the
    # meaningful absolute no-regression band instead of optimizer noise.
    withheld_no_regression_tolerance_m: float = 0.02
    min_scale_associations: int = 3
    min_scale_excitation_m: float = 0.10
    max_scale_relative_spread: float = 0.25
    withheld_ranges: tuple[tuple[int, int], ...] = ()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_similarity_matrix(value: Any, *, name: str) -> np.ndarray:
    """Validate an SE(3) matrix, or a uniform-scale Sim(3) if explicitly used."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise TrajectoryRefinementError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise TrajectoryRefinementError(f"{name} has an invalid homogeneous row")
    linear = matrix[:3, :3]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    if not np.isfinite(singular_values).all() or float(np.min(singular_values)) <= 1e-9:
        raise TrajectoryRefinementError(f"{name} has a singular linear part")
    scale = float(np.mean(singular_values))
    if scale <= 0.0 or not np.allclose(singular_values, scale, atol=2e-4, rtol=2e-4):
        raise TrajectoryRefinementError(f"{name} must have a uniform positive scale")
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise TrajectoryRefinementError(f"{name} linear part is not a similarity rotation")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-3):
        raise TrajectoryRefinementError(f"{name} rotation is not proper")
    return matrix


def _finite_matrix(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise TrajectoryRefinementError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise TrajectoryRefinementError(f"{name} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise TrajectoryRefinementError(f"{name} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-3):
        raise TrajectoryRefinementError(f"{name} rotation is not proper")
    return matrix


def _apply_pose(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    flat = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    result = (pose[:3, :3] @ flat.T).T + pose[:3, 3]
    return result.reshape(np.asarray(points).shape)


def _load_prepared_frames(scan_dir: Path) -> list[dict[str, Any]]:
    manifest_path = scan_dir / "prepared_frames_manifest.json"
    if not manifest_path.is_file():
        raise TrajectoryRefinementError(f"prepared-frame manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrajectoryRefinementError(f"prepared-frame manifest is unreadable: {manifest_path}") from exc
    rows = manifest.get("frames") if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or len(rows) < 2:
        raise TrajectoryRefinementError("prepared-frame manifest has too few frames")
    result: list[dict[str, Any]] = []
    for expected_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TrajectoryRefinementError(f"prepared frame {expected_index} is malformed")
        relative = row.get("frame")
        if not isinstance(relative, str) or not relative:
            raise TrajectoryRefinementError(f"prepared frame {expected_index} has no image path")
        image_path = (scan_dir / relative).resolve()
        try:
            image_path.relative_to(scan_dir.resolve())
        except ValueError as exc:
            raise TrajectoryRefinementError(
                f"prepared frame {expected_index} image escapes scan directory"
            ) from exc
        if not image_path.is_file():
            raise TrajectoryRefinementError(f"prepared frame image is missing: {image_path}")
        index = int(row.get("index", expected_index))
        if index != expected_index:
            raise TrajectoryRefinementError("prepared frame indexes must be contiguous")
        declared_sha256 = row.get("sha256")
        if not isinstance(declared_sha256, str) or len(declared_sha256) != 64:
            raise TrajectoryRefinementError(
                f"prepared frame {expected_index} has no declared SHA-256"
            )
        actual_sha256 = _sha256(image_path)
        if actual_sha256 != declared_sha256.lower():
            raise TrajectoryRefinementError(
                f"prepared frame {expected_index} image digest does not match its manifest"
            )
        try:
            canonical_frame_id = prepared_frame_identity(index, declared_sha256)
        except ValueError as exc:
            raise TrajectoryRefinementError(
                f"prepared frame {expected_index} has an invalid identity"
            ) from exc
        declared_frame_id = row.get("frame_id")
        if declared_frame_id is not None and declared_frame_id != canonical_frame_id:
            raise TrajectoryRefinementError(
                f"prepared frame {expected_index} identity is not index/hash bound"
            )
        result.append(
            {
                "index": index,
                "frame_id": canonical_frame_id,
                "timestamp_s": float(row.get("timestamp_s", index)),
                "capture_time_ns": row.get("capture_time_ns"),
                "image_path": image_path,
                "image_sha256": declared_sha256.lower(),
            }
        )
    return result


def _load_raw_views(raw_root: Path, frame_count: int) -> list[dict[str, Any]]:
    if not raw_root.is_dir():
        raise TrajectoryRefinementError(f"raw output directory is missing: {raw_root}")
    result: list[dict[str, Any]] = []
    for index in range(frame_count):
        path = raw_root / f"view_{index:04d}.npz"
        if not path.is_file():
            raise TrajectoryRefinementError(f"raw view {index} is missing: {path}")
        with np.load(path, allow_pickle=False) as archive:
            missing = [key for key in RAW_REQUIRED_KEYS if key not in archive.files]
            if missing:
                raise TrajectoryRefinementError(
                    f"raw view {index} lacks required keys: {', '.join(missing)}"
                )
            row = {key: np.asarray(archive[key]).copy() for key in archive.files}
        pose = _finite_matrix(row["camera_pose"], name=f"raw view {index} camera_pose")
        points = np.asarray(row["world_points"], dtype=np.float64)
        depth = np.asarray(row["depth_z"], dtype=np.float64)
        mask = np.asarray(row["mask"], dtype=bool)
        if points.ndim != 3 or points.shape[-1] != 3 or depth.shape != points.shape[:2] or mask.shape != depth.shape:
            raise TrajectoryRefinementError(f"raw view {index} has incompatible geometry shapes")
        row["camera_pose"] = pose
        row["world_points"] = points
        row["depth_z"] = depth
        row["mask"] = mask
        row["intrinsics"] = np.asarray(row["intrinsics"], dtype=np.float64)
        if row["intrinsics"].shape != (3, 3) or not np.isfinite(row["intrinsics"]).all():
            raise TrajectoryRefinementError(f"raw view {index} intrinsics are malformed")
        result.append({"path": path, "data": row})
    return result


def _raw_gray(raw: Mapping[str, np.ndarray]) -> np.ndarray:
    """Match landmarks on the exact pixel projection used by retained depth."""
    image = np.asarray(raw["model_rgb"])
    shape = np.asarray(raw["depth_z"]).shape
    if image.shape != (*shape, 3) or not np.isfinite(image).all():
        raise TrajectoryRefinementError("raw RGB must be finite HxWx3 on the depth grid")
    if image.dtype != np.uint8:
        if float(image.min()) < 0.0 or float(image.max()) > 255.0:
            raise TrajectoryRefinementError("raw RGB values are outside the supported range")
        image = np.clip(
            image * 255.0 if float(image.max()) <= 1.5 else image, 0, 255
        ).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _source_identity(raw_root: Path, frame_count: int) -> dict[str, Any]:
    manifest_path = raw_root.parent / "scan_outputs_manifest.json"
    if not manifest_path.is_file():
        return {"coordinate_frame": "da3_metric_world", "manifest": None}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        coordinate_frame = manifest["coordinate_frame"]
        view_count = int(manifest["view_count"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TrajectoryRefinementError("raw source manifest has invalid frame identity") from exc
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise TrajectoryRefinementError("raw source manifest has no coordinate frame")
    if view_count != frame_count:
        raise TrajectoryRefinementError("raw source manifest view count differs from prepared frames")
    return {
        "coordinate_frame": coordinate_frame,
        "provider": manifest.get("provider"),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }


def _image_signature(image: np.ndarray) -> np.ndarray:
    small = cv2.resize(image, (32, 24), interpolation=cv2.INTER_AREA)
    histogram = cv2.calcHist([small], [0], None, [16], [0, 256]).reshape(-1)
    histogram = histogram.astype(np.float64)
    histogram /= max(float(np.linalg.norm(histogram)), 1e-12)
    return histogram


def _retrieve_nonadjacent_pairs(
    images: list[np.ndarray], settings: TrajectoryRefinementSettings
) -> list[dict[str, Any]]:
    """Return a bounded, deterministic shortlist scored by image appearance."""
    if settings.min_nonadjacent_gap < 1 or settings.max_candidate_pairs < 1:
        raise TrajectoryRefinementError("nonadjacent retrieval settings are invalid")
    signatures = [_image_signature(image) for image in images]
    candidates: list[dict[str, Any]] = []
    withheld = settings.withheld_ranges
    for source in range(len(images)):
        for target in range(source + settings.min_nonadjacent_gap, len(images)):
            withheld_pair = any(
                start <= source < end or start <= target < end
                for start, end in withheld
            )
            score = float(signatures[source] @ signatures[target])
            candidates.append(
                {
                    "source_view": source,
                    "target_view": target,
                    "retrieval_score": score,
                    "withheld_from_fit": withheld_pair,
                }
            )
    candidates.sort(key=lambda row: (-row["retrieval_score"], row["source_view"], row["target_view"]))
    return candidates[: settings.max_candidate_pairs]


def _spread_metrics(points: np.ndarray, image_shape: tuple[int, int]) -> tuple[float, int]:
    if points.shape[0] < 3:
        return 0.0, 0
    height, width = image_shape
    normalized = points[:, [0, 1]] / np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float64)
    hull = cv2.convexHull(normalized.astype(np.float32))
    area = float(cv2.contourArea(hull))
    grid_x = np.clip((normalized[:, 0] * 4).astype(int), 0, 3)
    grid_y = np.clip((normalized[:, 1] * 4).astype(int), 0, 3)
    bins = len({(int(x), int(y)) for x, y in zip(grid_x, grid_y, strict=True)})
    return area, bins


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_transposed = np.linalg.svd(covariance)
    rotation = right_transposed.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_transposed[-1] *= -1.0
        rotation = right_transposed.T @ left.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def _sample_world_points(
    raw: Mapping[str, np.ndarray], keypoints: Iterable[cv2.KeyPoint]
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(raw["world_points"], dtype=np.float64)
    mask = np.asarray(raw["mask"], dtype=bool)
    height, width = mask.shape
    selected: list[np.ndarray] = []
    locations: list[np.ndarray] = []
    for keypoint in keypoints:
        x, y = np.rint(keypoint.pt).astype(int)
        if x < 0 or y < 0 or x >= width or y >= height or not mask[y, x]:
            continue
        point = points[y, x]
        if np.isfinite(point).all():
            selected.append(point)
            locations.append(np.asarray([x, y], dtype=np.float64))
    if not selected:
        return np.empty((0, 3)), np.empty((0, 2))
    return np.stack(selected), np.stack(locations)


def _fit_3d_rigid(
    source: np.ndarray,
    target: np.ndarray,
    settings: TrajectoryRefinementSettings,
    seed: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        return None, {"status": "malformed_correspondences", "match_count": 0}
    count = source.shape[0]
    if count < settings.min_3d_matches:
        return None, {"status": "insufficient_depth_backed_matches", "match_count": int(count)}
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(count, dtype=bool)
    best_transform: np.ndarray | None = None
    for _ in range(min(settings.ransac_iterations, max(1, count * 4))):
        sample = rng.choice(count, size=3, replace=False)
        if np.linalg.matrix_rank(source[sample] - source[sample].mean(axis=0)) < 2:
            continue
        transform = _kabsch(source[sample], target[sample])
        residual = np.linalg.norm(_apply_pose(transform, source) - target, axis=1)
        inliers = residual <= settings.max_3d_residual_m
        if int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers
            best_transform = transform
    if best_transform is None or int(best_inliers.sum()) < settings.min_3d_inliers:
        return None, {
            "status": "insufficient_3d_inliers",
            "match_count": int(count),
            "inlier_count": int(best_inliers.sum()),
        }
    transform = _kabsch(source[best_inliers], target[best_inliers])
    residual = np.linalg.norm(_apply_pose(transform, source) - target, axis=1)
    inliers = residual <= settings.max_3d_residual_m
    inlier_residual = residual[inliers]
    fraction = float(np.count_nonzero(inliers) / count)
    metrics = {
        "status": "ok" if fraction >= settings.min_3d_inlier_fraction and inlier_residual.size else "inlier_fraction_below_gate",
        "match_count": int(count),
        "inlier_count": int(np.count_nonzero(inliers)),
        "inlier_fraction": fraction,
        "residual_median_m": float(np.median(inlier_residual)) if inlier_residual.size else None,
        "residual_p80_m": float(np.percentile(inlier_residual, 80.0)) if inlier_residual.size else None,
        "residual_max_m": float(np.max(inlier_residual)) if inlier_residual.size else None,
    }
    if (
        metrics["status"] != "ok"
        or metrics["residual_p80_m"] is None
        or metrics["residual_p80_m"] > settings.max_3d_p80_residual_m
    ):
        return None, metrics
    return transform, metrics


def _verify_visual_revisit(
    source_index: int,
    target_index: int,
    source_image: np.ndarray,
    target_image: np.ndarray,
    source_raw: Mapping[str, np.ndarray],
    target_raw: Mapping[str, np.ndarray],
    settings: TrajectoryRefinementSettings,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Verify one candidate and return an explicit source-from-target edge."""
    detector = cv2.SIFT_create(nfeatures=settings.max_features)
    source_keypoints, source_descriptors = detector.detectAndCompute(source_image, None)
    target_keypoints, target_descriptors = detector.detectAndCompute(target_image, None)
    base = {
        "source_view": source_index,
        "target_view": target_index,
        "match_count": 0,
        "two_d_inlier_count": 0,
        "three_d_match_count": 0,
    }
    if source_descriptors is None or target_descriptors is None:
        base["rejection_reason"] = "missing_descriptors"
        return base, None
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    paired = matcher.knnMatch(source_descriptors, target_descriptors, k=2)
    matches = [row[0] for row in paired if len(row) == 2 and row[0].distance < settings.ratio_test * row[1].distance]
    base["match_count"] = int(len(matches))
    if len(matches) < settings.min_matches:
        base["rejection_reason"] = "insufficient_2d_matches"
        return base, None
    source_2d = np.float32([source_keypoints[m.queryIdx].pt for m in matches])
    target_2d = np.float32([target_keypoints[m.trainIdx].pt for m in matches])
    homography, inlier_mask = cv2.findHomography(
        source_2d,
        target_2d,
        cv2.RANSAC,
        3.0,
        maxIters=2000,
        confidence=0.995,
    )
    if homography is None or inlier_mask is None:
        base["rejection_reason"] = "no_2d_geometric_model"
        return base, None
    inliers_2d = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    base["two_d_inlier_count"] = int(np.count_nonzero(inliers_2d))
    base["two_d_inlier_fraction"] = float(np.count_nonzero(inliers_2d) / len(matches))
    spread, grid_bins = _spread_metrics(source_2d[inliers_2d], source_image.shape)
    base["two_d_inlier_spread_fraction"] = spread
    base["two_d_inlier_grid_bins"] = grid_bins
    if base["two_d_inlier_count"] < settings.min_2d_inliers:
        base["rejection_reason"] = "insufficient_2d_inliers"
        return base, None
    if base["two_d_inlier_fraction"] < settings.min_inlier_fraction:
        base["rejection_reason"] = "2d_inlier_fraction_below_gate"
        return base, None
    if spread < settings.min_inlier_spread_fraction or grid_bins < 4:
        base["rejection_reason"] = "2d_inliers_not_well_spread"
        return base, None

    selected_matches = [match for match, keep in zip(matches, inliers_2d, strict=True) if keep]
    source_points_world, _ = _sample_world_points(
        source_raw, [source_keypoints[m.queryIdx] for m in selected_matches]
    )
    target_points_world, _ = _sample_world_points(
        target_raw, [target_keypoints[m.trainIdx] for m in selected_matches]
    )
    # Sampling is performed in the same order but invalid pixels can differ.
    # Use the original keypoint coordinates to retain only pairs with valid
    # finite depth in both views.
    source_pose = _finite_matrix(source_raw["camera_pose"], name="source camera pose")
    target_pose = _finite_matrix(target_raw["camera_pose"], name="target camera pose")
    source_world = np.asarray(source_raw["world_points"], dtype=np.float64)
    target_world = np.asarray(target_raw["world_points"], dtype=np.float64)
    source_mask = np.asarray(source_raw["mask"], dtype=bool)
    target_mask = np.asarray(target_raw["mask"], dtype=bool)
    source_local: list[np.ndarray] = []
    target_local: list[np.ndarray] = []
    for match in selected_matches:
        sx, sy = np.rint(source_keypoints[match.queryIdx].pt).astype(int)
        tx, ty = np.rint(target_keypoints[match.trainIdx].pt).astype(int)
        if (
            sx < 0 or sy < 0 or tx < 0 or ty < 0
            or sy >= source_mask.shape[0] or sx >= source_mask.shape[1]
            or ty >= target_mask.shape[0] or tx >= target_mask.shape[1]
            or not source_mask[sy, sx] or not target_mask[ty, tx]
        ):
            continue
        source_point = source_world[sy, sx]
        target_point = target_world[ty, tx]
        if not np.isfinite(source_point).all() or not np.isfinite(target_point).all():
            continue
        source_local.append(_apply_pose(np.linalg.inv(source_pose), source_point))
        target_local.append(_apply_pose(np.linalg.inv(target_pose), target_point))
    if len(source_local) < settings.min_3d_matches:
        base["three_d_match_count"] = len(source_local)
        base["rejection_reason"] = "insufficient_depth_backed_matches"
        return base, None
    source_local_array = np.stack(source_local)
    target_local_array = np.stack(target_local)
    base["three_d_match_count"] = int(source_local_array.shape[0])
    transform, geometry = _fit_3d_rigid(
        source_local_array,
        target_local_array,
        settings,
        seed=source_index * 1_000_003 + target_index,
    )
    base["three_d_geometry"] = geometry
    if transform is None:
        base["rejection_reason"] = str(geometry.get("status") or "3d_geometry_gate")
        return base, None
    source_from_target = np.linalg.inv(transform)
    rotation_sigma_deg = max(1.0, min(15.0, math.degrees(math.atan2(
        max(float(geometry.get("residual_p80_m") or 0.0), 1e-6),
        max(float(np.median(np.linalg.norm(source_local_array, axis=1))), 1e-3),
    ))))
    translation_sigma = max(0.03, min(0.30, float(geometry["residual_p80_m"] or 0.03)))
    edge = {
        "source_view": source_index,
        "target_view": target_index,
        "transform": source_from_target.tolist(),
        "label": "verified_visual_loop",
        "translation_sigma_m": translation_sigma,
        "rotation_sigma_deg": rotation_sigma_deg,
        "source_from_target_semantics": "target_camera_coordinates_to_source_camera_coordinates",
    }
    base["status"] = "accepted"
    base["rejection_reason"] = None
    return base, edge


def _validate_vio_constraints(
    path: Path,
    frame_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt the validated capture VIO result to explicit camera-relative edges.

    ``vio.py`` emits camera-to-OpenVINS-world poses rather than relative rows.
    This adapter uses exact prepared-frame IDs and capture timestamps to derive
    only consecutive, same-segment camera-relative transforms.  The relative
    transform is expressed in the OpenCV camera axes shared by the retained
    DA3 raw outputs; OpenVINS' z-up global gauge is therefore not mixed into
    the carrier.  No absolute VIO pose or unverified gap is admitted.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrajectoryRefinementError(f"VIO constraint report is unreadable: {path}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != "noesis.phone_capture.vio_result.v1":
        raise TrajectoryRefinementError("VIO input must use the validated phone_capture.vio_result.v1 schema")
    try:
        from tools.mapanything_phone_scan.vio import validate_vio_result

        payload = validate_vio_result(payload)
    except Exception as exc:
        raise TrajectoryRefinementError(f"VIO result failed its calibrated OpenVINS contract: {exc}") from exc
    frame = payload["frame"]
    if frame.get("source") != "camera" or frame.get("target") != "vio_world" or frame.get("pose_convention") != "T_vio_world_camera":
        raise TrajectoryRefinementError("VIO frame is not T_vio_world_camera")
    frame_ids = {str(row["frame_id"]): int(row["index"]) for row in frame_rows}
    timestamp_by_index = {
        int(row["index"]): row.get("capture_time_ns") for row in frame_rows
    }
    pose_rows = payload["poses"]
    mapped: list[tuple[int, Mapping[str, Any]]] = []
    for index, row in enumerate(pose_rows):
        frame_id = row.get("prepared_frame_id")
        if not isinstance(frame_id, str) or frame_id not in frame_ids:
            raise TrajectoryRefinementError(
                f"VIO pose {index} does not identify a prepared frame exactly"
            )
        view_index = frame_ids[frame_id]
        capture_time_ns = timestamp_by_index.get(view_index)
        if not isinstance(capture_time_ns, int) or capture_time_ns != row.get("capture_time_ns"):
            raise TrajectoryRefinementError(
                f"VIO pose {index} timestamp does not match prepared frame {frame_id}"
            )
        mapped.append((view_index, row))
    mapped.sort(key=lambda item: item[0])
    if len({index for index, _ in mapped}) != len(mapped):
        raise TrajectoryRefinementError("VIO has duplicate prepared-frame identities")
    segments = payload.get("segments")
    if len(segments) > 1 and any("segment_id" not in row for _, row in mapped):
        raise TrajectoryRefinementError("VIO reset segments need per-pose segment_id values")
    result: list[dict[str, Any]] = []
    skipped_gaps = 0
    for (source, source_row), (target, target_row) in zip(mapped, mapped[1:], strict=False):
        if target != source + 1:
            skipped_gaps += 1
            continue
        if source_row.get("segment_id") != target_row.get("segment_id"):
            skipped_gaps += 1
            continue
        if source_row.get("gap_before") is True or target_row.get("gap_before") is True:
            skipped_gaps += 1
            continue
        source_covariance_value = source_row.get("covariance")
        target_covariance_value = target_row.get("covariance")
        if source_covariance_value is None or target_covariance_value is None:
            skipped_gaps += 1
            continue
        source_covariance = np.asarray(source_covariance_value, dtype=np.float64)
        target_covariance = np.asarray(target_covariance_value, dtype=np.float64)
        if (
            source_covariance.shape != (6, 6)
            or target_covariance.shape != (6, 6)
            or not np.isfinite(source_covariance).all()
            or not np.isfinite(target_covariance).all()
            or not np.allclose(source_covariance, source_covariance.T, atol=1e-8)
            or not np.allclose(target_covariance, target_covariance.T, atol=1e-8)
        ):
            raise TrajectoryRefinementError(f"VIO covariance for view pair {source},{target} is malformed")
        source_pose = _finite_matrix(source_row["T_vio_world_camera"], name=f"VIO pose {source}")
        target_pose = _finite_matrix(target_row["T_vio_world_camera"], name=f"VIO pose {target}")
        source_rotation = source_pose[:3, :3]
        displacement = target_pose[:3, 3] - source_pose[:3, 3]
        skew_displacement = np.asarray(
            [
                [0.0, -displacement[2], displacement[1]],
                [displacement[2], 0.0, -displacement[0]],
                [-displacement[1], displacement[0], 0.0],
            ],
            dtype=np.float64,
        )
        rotation_transpose = source_rotation.T
        zero = np.zeros((3, 3), dtype=np.float64)
        jacobian_source = np.block(
            [
                [-rotation_transpose, zero],
                [rotation_transpose @ skew_displacement, -rotation_transpose],
            ]
        )
        jacobian_target = np.block(
            [
                [rotation_transpose, zero],
                [zero, rotation_transpose],
            ]
        )
        marginal_covariance = (
            jacobian_source @ source_covariance @ jacobian_source.T
            + jacobian_target @ target_covariance @ jacobian_target.T
        )
        # OpenVINS does not provide source/target cross covariance.  Doubling
        # the marginal Jacobian sum is a declared bound used only to weight
        # this uncalibrated relative constraint, never a probability.
        covariance = 2.0 * marginal_covariance
        relative = np.linalg.inv(source_pose) @ target_pose
        rotation_sigma = math.sqrt(max(float(np.max(np.linalg.eigvalsh(covariance[:3, :3]))), 1e-8))
        translation_sigma = math.sqrt(max(float(np.max(np.linalg.eigvalsh(covariance[3:, 3:]))), 1e-8))
        result.append(
            {
                "source_view": source,
                "target_view": target,
                "transform": relative.tolist(),
                "label": "verified_vio_relative",
                "translation_sigma_m": max(0.03, translation_sigma),
                "rotation_sigma_deg": max(1.0, math.degrees(rotation_sigma)),
                "covariance": covariance.tolist(),
                "covariance_provenance": {
                    "source_absolute_covariance": True,
                    "target_absolute_covariance": True,
                    "cross_correlation_known": False,
                    "approximation": "two_times_relative_pose_jacobian_marginal_sum",
                    "jacobian_source": jacobian_source.tolist(),
                    "jacobian_target": jacobian_target.tolist(),
                },
                "source_from_target_semantics": "target_camera_coordinates_to_source_camera_coordinates",
                "source_time_ns": source_row["capture_time_ns"],
                "target_time_ns": target_row["capture_time_ns"],
                **({"source_pose_time_ns": source_row["pose_time_ns"],
                    "target_pose_time_ns": target_row["pose_time_ns"],
                    "pose_time_reference": frame["pose_time_reference"],
                    "source_time_reference": "original_camera_sensor_timestamp"}
                   if payload.get("short_session_consumer") else {}),
                "openvins_world_axes": "z_up",
                "noesis_camera_axes": "opencv_x_right_y_down_z_forward",
            }
        )
    return result, {
        "schema": "noesis.phone_capture.vio_result.v1",
        "estimator": "openvins",
        "capture_id": frame.get("capture_id"),
        "camera_sensor_id": frame.get("camera_sensor_id"),
        "source_frame": "camera",
        "target_frame": "vio_world",
        "pose_convention": "T_vio_world_camera",
        "world_axes": "openvins_z_up",
        "relative_constraint_axes": "opencv_x_right_y_down_z_forward",
        "relative_constraints_use_global_axis_map": False,
        **({"pose_time_reference": frame["pose_time_reference"],
            "capture_time_reference": frame["capture_time_reference"],
            "short_profile_id": payload["short_session_consumer"]["profile_id"],
            "image_motion_model": payload["short_session_consumer"]["image_motion_model"]}
           if payload.get("short_session_consumer") else {}),
        "skipped_gap_or_reset_count": skipped_gaps,
    }


def _materialize_refined_raw(
    raw_views: list[dict[str, Any]],
    corrected_poses: np.ndarray,
    output_dir: Path,
    *,
    scale: float = 1.0,
    source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_identity = source_identity or {"coordinate_frame": "da3_metric_world"}
    coordinate_frame = str(source_identity["coordinate_frame"])
    raw_output = output_dir / "raw"
    raw_output.mkdir(parents=True, exist_ok=False)
    for index, (raw_view, corrected_pose) in enumerate(zip(raw_views, corrected_poses, strict=True)):
        data = {key: np.asarray(value).copy() for key, value in raw_view["data"].items()}
        old_pose = _finite_matrix(data["camera_pose"], name=f"raw view {index} camera_pose")
        corrected_pose = _finite_matrix(corrected_pose, name=f"corrected pose {index}")
        old_points = np.asarray(data["world_points"], dtype=np.float64)
        old_local_points = _apply_pose(np.linalg.inv(old_pose), old_points)
        data["world_points"] = _apply_pose(
            corrected_pose, float(scale) * old_local_points
        ).astype(np.float32)
        if not math.isclose(float(scale), 1.0, abs_tol=1e-8):
            data["depth_z"] = (np.asarray(data["depth_z"], dtype=np.float64) * float(scale)).astype(np.float32)
        data["camera_pose"] = corrected_pose.astype(np.float32)
        np.savez_compressed(raw_output / f"view_{index:04d}.npz", **data)
    camera_solution = output_dir / "camera_solution.npz"
    np.savez_compressed(
        camera_solution,
        camera_to_world=np.asarray(corrected_poses, dtype=np.float32),
        source_camera_to_world=np.stack([row["data"]["camera_pose"] for row in raw_views]).astype(np.float32),
        intrinsics=np.stack([row["data"]["intrinsics"] for row in raw_views]).astype(np.float32),
        coordinate_frame=np.asarray(coordinate_frame),
        units=np.asarray("m", dtype="U2"),
    )
    manifest_path = output_dir / "scan_outputs_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "noesis.phone_walk.refined_source_outputs.v1",
                "provider": "trajectory_refinement",
                "coordinate_frame": coordinate_frame,
                "source_identity": dict(source_identity),
                "view_count": len(raw_views),
                "raw_dir": "raw",
                "geometry_scale_factor": float(scale),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "raw_dir": str(raw_output),
        "camera_solution": str(camera_solution),
        "camera_solution_sha256": _sha256(camera_solution),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "depth_rebuilt": not math.isclose(float(scale), 1.0, abs_tol=1e-8),
        "depth_semantics": (
            "retained_camera_local_depth_z_unchanged_under_pose_world_point_update"
            if math.isclose(float(scale), 1.0, abs_tol=1e-8)
            else "retained_camera_local_depth_z_scaled_with_metric_vio_association"
        ),
        "world_points_rebuilt": True,
        "poses_rebuilt": True,
        "metric_scale_preserved": math.isclose(float(scale), 1.0, abs_tol=1e-8),
        "metric_scale_factor_applied": float(scale),
    }


def _validate_scaled_alignment_result(
    value: Any,
    *,
    result_dir: Path,
    scale_factor: float,
) -> dict[str, Any] | None:
    """Accept only a passed alignment run over the materialized scaled carrier."""
    if not isinstance(value, Mapping):
        return None
    quality_gate = value.get("quality_gate")
    if not isinstance(quality_gate, Mapping) or quality_gate.get("passed") is not True:
        return None
    try:
        transform_value = value.get("transform_path")
        if transform_value is None and isinstance(value.get("artifact_paths"), Mapping):
            transform_value = value["artifact_paths"].get("transform")
        transform_path = Path(str(transform_value)).expanduser()
    except (KeyError, TypeError, ValueError):
        return None
    if not transform_path.is_absolute():
        transform_path = result_dir / transform_path
    transform_path = transform_path.resolve()
    try:
        transform_path.relative_to(result_dir.resolve())
    except ValueError:
        return None
    if not transform_path.is_file():
        return None
    try:
        payload = json.loads(transform_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or payload.get("schema") != "noesis.mapanything.phone_scan.world_alignment.v1":
        return None
    try:
        matrix = payload.get("world_from_mapanything_row_major")
        matrix = _validated_similarity_matrix(matrix, name="scaled-carrier world mapping")
        transform_scale = float(payload.get("scale", 1.0))
    except (TrajectoryRefinementError, TypeError, ValueError):
        return None
    matrix_scale = float(np.mean(np.linalg.svd(matrix[:3, :3], compute_uv=False)))
    if (
        not math.isclose(transform_scale, 1.0, abs_tol=2e-3)
        or not math.isclose(matrix_scale, 1.0, abs_tol=2e-3)
    ):
        return None
    result = dict(value)
    result["transform_path"] = str(transform_path)
    result["scale_factor"] = float(scale_factor)
    result["alignment_fit_used"] = True
    return result



def run_trajectory_refinement(
    scan_dir: Path,
    da3_raw: Path,
    output_dir: Path,
    settings: TrajectoryRefinementSettings | None = None,
    *,
    vio_constraints: Path | None = None,
    revalidate_scaled_carrier: Callable[[Path, Path, Path, float], Mapping[str, Any]] | None = None,
    prepared_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Run bounded visual revisit verification and optional raw materialization."""
    settings = settings or TrajectoryRefinementSettings()
    scan_dir = scan_dir.resolve()
    da3_raw = da3_raw.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise TrajectoryRefinementError(f"trajectory-refinement directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = _load_prepared_frames(scan_dir)
    parent_frame_count = len(frame_rows)
    if prepared_indices is not None:
        # A reviewed partial provider keeps original frame IDs/timestamps. It
        # must not be made to look contiguous by rewriting the prepared scan.
        from .trajectory_motion_review import _load_provider

        selected_source = _load_provider(scan_dir, da3_raw.parent / "scan_outputs_manifest.json", allow_partial=True)
        if (any(type(index) is not int for index in prepared_indices)
                or list(prepared_indices) != selected_source["selected"]):
            raise TrajectoryRefinementError("partial refinement indexes differ from exact provider/prepared identities")
        frame_rows = [frame_rows[index] for index in prepared_indices]
    if any(not (0 <= start < end <= len(frame_rows)) for start, end in settings.withheld_ranges):
        raise TrajectoryRefinementError("withheld ranges must be nonempty intervals within the prepared views")
    source_identity = _source_identity(da3_raw, len(frame_rows))
    raw_views = _load_raw_views(da3_raw, len(frame_rows))
    # Prepared images establish capture identity. Cropped or common-ray raw
    # outputs have a different projection, so resizing prepared RGB would
    # sample depth at unrelated pixels during three-dimensional verification.
    images = [_raw_gray(row["data"]) for row in raw_views]
    candidates = _retrieve_nonadjacent_pairs(images, settings)
    verified: list[dict[str, Any]] = []
    withheld_verified: list[dict[str, Any]] = []
    withheld_edges: list[dict[str, Any]] = []
    accepted_edges: list[dict[str, Any]] = []
    for candidate in candidates:
        metrics, edge = _verify_visual_revisit(
            candidate["source_view"],
            candidate["target_view"],
            images[candidate["source_view"]],
            images[candidate["target_view"]],
            raw_views[candidate["source_view"]]["data"],
            raw_views[candidate["target_view"]]["data"],
            settings,
        )
        row = {**candidate, **metrics}
        row.update(
            {
                "source_frame_id": frame_rows[candidate["source_view"]]["frame_id"],
                "target_frame_id": frame_rows[candidate["target_view"]]["frame_id"],
                "source_time_s": frame_rows[candidate["source_view"]]["timestamp_s"],
                "target_time_s": frame_rows[candidate["target_view"]]["timestamp_s"],
                "source_image_sha256": frame_rows[candidate["source_view"]]["image_sha256"],
                "target_image_sha256": frame_rows[candidate["target_view"]]["image_sha256"],
                "fit_eligible": not candidate["withheld_from_fit"],
            }
        )
        if candidate["withheld_from_fit"]:
            row["status"] = "withheld_from_fit"
            row["rejection_reason"] = "withheld_from_fit"
            if edge is not None:
                row["withheld_constraint"] = edge
                withheld_edges.append(edge)
            withheld_verified.append(row)
            verified.append(row)
            continue
        verified.append(row)
        if edge is not None:
            row["verified_constraint"] = edge
            accepted_edges.append(edge)
    vio_edges: list[dict[str, Any]] = []
    vio_info: dict[str, Any] = {"status": "not_supplied"}
    vio_status = "not_supplied"
    vio_rejection_reason: str | None = None
    if vio_constraints is not None:
        try:
            vio_edges, vio_info = _validate_vio_constraints(
                vio_constraints.resolve(), frame_rows
            )
            vio_status = "accepted" if vio_edges else "no_accepted_constraints"
        except TrajectoryRefinementError as exc:
            vio_status = "rejected"
            vio_rejection_reason = str(exc)
            verified.append({"constraint_family": "vio", "status": "rejected", "rejection_reason": str(exc)})
    all_edges = accepted_edges + vio_edges
    refinement: dict[str, Any] = {
        "status": "no_verified_constraints" if not all_edges else "verified_constraints_available",
        "accepted_constraint_count": len(all_edges),
        "accepted_visual_constraint_count": len(accepted_edges),
        "accepted_vio_constraint_count": len(vio_edges),
        "raw_materialized": False,
        "scale_change": 1.0,
        "gauge_change": False,
        "world_alignment_revalidation_required": False,
    }
    if all_edges:
        from tools.mapanything_phone_scan.build_consensus_fusion import _pose_graph

        source_poses = np.stack([row["data"]["camera_pose"] for row in raw_views])
        scale = 1.0
        scale_metrics: dict[str, Any] = {
            "status": "not_required",
            "factor": 1.0,
            "sample_count": 0,
        }
        if vio_edges:
            ratios: list[float] = []
            for edge in vio_edges:
                source = int(edge["source_view"])
                target = int(edge["target_view"])
                da3_relative = np.linalg.inv(source_poses[source]) @ source_poses[target]
                da3_norm = float(np.linalg.norm(da3_relative[:3, 3]))
                vio_norm = float(np.linalg.norm(np.asarray(edge["transform"], dtype=np.float64)[:3, 3]))
                if da3_norm > 0.02 and math.isfinite(vio_norm):
                    ratios.append(vio_norm / da3_norm)
            if len(ratios) < settings.min_scale_associations:
                refinement.update({
                    "status": "rejected_unobservable_metric_scale",
                    "rejection_reason": (
                        f"fewer than {settings.min_scale_associations} nonzero DA3/VIO "
                        "translation associations"
                    ),
                    "raw_materialized": False,
                })
                vio_rejection_reason = "fewer than three nonzero DA3/VIO translation associations"
                scale = 1.0
                all_edges = list(accepted_edges)
            else:
                scale = float(np.median(ratios))
                ratio_p20 = float(np.percentile(ratios, 20.0))
                ratio_p80 = float(np.percentile(ratios, 80.0))
                ratio_spread = (ratio_p80 - ratio_p20) / max(abs(scale), 1e-9)
                da3_motion = []
                for edge in vio_edges:
                    source_index = int(edge["source_view"])
                    target_index = int(edge["target_view"])
                    relative = np.linalg.inv(source_poses[source_index]) @ source_poses[target_index]
                    da3_motion.append(float(np.linalg.norm(relative[:3, 3])))
                scale_metrics = {
                    "status": "associated_from_vio_relative_translations",
                    "factor": scale,
                    "sample_count": len(ratios),
                    "ratio_p20": ratio_p20,
                    "ratio_p80": ratio_p80,
                    "relative_spread": ratio_spread,
                    "excitation_median_da3_translation": float(np.median(da3_motion)),
                }
                excitation = float(scale_metrics["excitation_median_da3_translation"])
                if (
                    not math.isfinite(scale)
                    or scale <= 0.0
                    or excitation < settings.min_scale_excitation_m
                    or not math.isfinite(ratio_spread)
                    or ratio_spread > settings.max_scale_relative_spread
                ):
                    refinement.update({
                        "status": "rejected_unobservable_metric_scale",
                        "rejection_reason": (
                            "VIO/DA3 scale association was nonfinite, weakly excited, "
                            "or too dispersed"
                        ),
                        "raw_materialized": False,
                    })
                    vio_rejection_reason = "VIO/DA3 scale association was not finite positive"
                    scale = 1.0
                    all_edges = list(accepted_edges)
        refinement["accepted_constraint_count"] = len(all_edges)
        if vio_rejection_reason is not None:
            refinement["vio_rejection_reason"] = vio_rejection_reason
        refinement["scale_association"] = scale_metrics
        refinement["scale_change"] = scale
        if all_edges:
            # DA3/visual translations are in the source metric gauge.  When a
            # calibrated VIO association supplies a non-unit physical scale,
            # scale both the source carrier and visual edge translations before
            # solving.  VIO edges already carry metric translations and remain
            # unchanged.  The first camera remains the gauge origin.
            solve_source_poses = source_poses.copy()
            if not math.isclose(float(scale), 1.0, abs_tol=1e-8):
                anchor = solve_source_poses[0, :3, 3].copy()
                solve_source_poses[:, :3, 3] = anchor + float(scale) * (
                    solve_source_poses[:, :3, 3] - anchor
                )
            solve_edges: list[dict[str, Any]] = []
            for edge in all_edges:
                row = dict(edge)
                if (
                    not str(row.get("label") or "").startswith("verified_vio")
                    and not math.isclose(float(scale), 1.0, abs_tol=1e-8)
                ):
                    transform = np.asarray(row["transform"], dtype=np.float64).copy()
                    transform[:3, 3] *= float(scale)
                    row["transform"] = transform.tolist()
                    row["translation_sigma_m"] = float(row.get("translation_sigma_m", 0.12)) * float(scale)
                solve_edges.append(row)
            optimized_local, pose_metrics = _pose_graph(
                solve_source_poses, solve_source_poses, solve_edges, single_carrier=True,
                single_carrier_name=str(source_identity.get("provider") or "da3"),
            )
            corrected_poses = np.einsum("ij,njk->nik", solve_source_poses[0], optimized_local)
            translation_changes = np.linalg.norm(
                corrected_poses[:, :3, 3] - solve_source_poses[:, :3, 3], axis=1
            )
            rotation_changes = np.asarray([
                math.degrees(math.acos(float(np.clip(
                    (np.trace(source[:3, :3].T @ corrected[:3, :3]) - 1.0) * 0.5,
                    -1.0,
                    1.0,
                ))))
                for source, corrected in zip(solve_source_poses, corrected_poses, strict=True)
            ])
            edge_translation_residuals: list[float] = []
            edge_rotation_residuals: list[float] = []
            for edge in solve_edges:
                source = int(edge["source_view"])
                target = int(edge["target_view"])
                observed = np.asarray(edge["transform"], dtype=np.float64)
                predicted = np.linalg.inv(optimized_local[source]) @ optimized_local[target]
                edge_translation_residuals.append(float(np.linalg.norm(predicted[:3, 3] - observed[:3, 3])))
                edge_rotation_residuals.append(math.degrees(math.acos(float(np.clip(
                    (np.trace(observed[:3, :3].T @ predicted[:3, :3]) - 1.0) * 0.5,
                    -1.0,
                    1.0,
                )))))
            residual_metrics = {
                "translation_p80_m": float(np.percentile(edge_translation_residuals, 80.0)),
                "rotation_p80_deg": float(np.percentile(edge_rotation_residuals, 80.0)),
                "constraint_count": len(all_edges),
            }
            valid_solver = bool(pose_metrics.get("solver_success")) and np.isfinite(corrected_poses).all()
            valid_deformation = bool(
                np.max(translation_changes) <= settings.max_pose_translation_change_m
                and np.max(rotation_changes) <= settings.max_pose_rotation_change_deg
                and residual_metrics["translation_p80_m"] <= settings.max_verified_constraint_translation_residual_m
                and residual_metrics["rotation_p80_deg"] <= settings.max_verified_constraint_rotation_residual_deg
            )
            refinement.update(
                {
                    "pose_graph": pose_metrics,
                    "pose_deformation": {
                        "translation_change_max_m": float(np.max(translation_changes)),
                        "rotation_change_max_deg": float(np.max(rotation_changes)),
                        "physical_scale_factor_applied": float(scale),
                        "total_translation_change_from_original_max_m": float(
                            np.max(
                                np.linalg.norm(
                                    corrected_poses[:, :3, 3]
                                    - source_poses[:, :3, 3],
                                    axis=1,
                                )
                            )
                        ),
                    },
                    "verified_constraint_residuals": residual_metrics,
                    "raw_materialized": bool(valid_solver and valid_deformation),
                    "source_pose_start_end_distance_m": float(np.linalg.norm(source_poses[-1, :3, 3] - source_poses[0, :3, 3])),
                }
            )
            refinement["corrected_pose_start_end_distance_m"] = float(np.linalg.norm(corrected_poses[-1, :3, 3] - corrected_poses[0, :3, 3]))
            if not (valid_solver and valid_deformation):
                refinement.update({
                    "status": "rejected_solver_or_deformation_gate",
                    "rejection_reason": "solver, pose deformation, or verified-edge residual gate failed",
                })
        else:
            refinement.setdefault("raw_materialized", False)
            if not accepted_edges and vio_rejection_reason is not None:
                refinement["status"] = "rejected_vio_constraint_gate"

    evaluation_poses = (
        corrected_poses
        if "corrected_poses" in locals() and refinement.get("raw_materialized") is True
        else np.stack([row["data"]["camera_pose"] for row in raw_views])
    )
    withheld_residuals: list[dict[str, Any]] = []
    evaluation_scale = float(refinement.get("scale_change") or 1.0)
    for edge in withheld_edges:
        source = int(edge["source_view"])
        target = int(edge["target_view"])
        observed = np.asarray(edge["transform"], dtype=np.float64).copy()
        if (
            not str(edge.get("label") or "").startswith("verified_vio")
            and not math.isclose(evaluation_scale, 1.0, abs_tol=1e-8)
        ):
            observed[:3, 3] *= evaluation_scale
        predicted = np.linalg.inv(
            np.linalg.inv(evaluation_poses[0]) @ evaluation_poses[source]
        ) @ (np.linalg.inv(evaluation_poses[0]) @ evaluation_poses[target])
        translation_error = float(np.linalg.norm(predicted[:3, 3] - observed[:3, 3]))
        rotation_error = math.degrees(math.acos(float(np.clip(
            (np.trace(observed[:3, :3].T @ predicted[:3, :3]) - 1.0) * 0.5,
            -1.0,
            1.0,
        ))))
        withheld_residuals.append(
            {
                "source_view": source,
                "target_view": target,
                "translation_residual_m": translation_error,
                "rotation_residual_deg": rotation_error,
                "observed_translation_scale": evaluation_scale,
            }
        )
    baseline_withheld_residuals: list[dict[str, Any]] = []
    baseline_poses = np.stack([row["data"]["camera_pose"] for row in raw_views])
    baseline_evaluation_poses = baseline_poses.copy()
    if not math.isclose(evaluation_scale, 1.0, abs_tol=1e-8):
        anchor = baseline_evaluation_poses[0, :3, 3].copy()
        baseline_evaluation_poses[:, :3, 3] = anchor + evaluation_scale * (
            baseline_evaluation_poses[:, :3, 3] - anchor
        )
    for edge in withheld_edges:
        source = int(edge["source_view"])
        target = int(edge["target_view"])
        observed = np.asarray(edge["transform"], dtype=np.float64).copy()
        if (
            not str(edge.get("label") or "").startswith("verified_vio")
            and not math.isclose(evaluation_scale, 1.0, abs_tol=1e-8)
        ):
            observed[:3, 3] *= evaluation_scale
        predicted = np.linalg.inv(
            np.linalg.inv(baseline_evaluation_poses[0])
            @ baseline_evaluation_poses[source]
        ) @ (
            np.linalg.inv(baseline_evaluation_poses[0])
            @ baseline_evaluation_poses[target]
        )
        baseline_withheld_residuals.append({
            "source_view": source,
            "target_view": target,
            "translation_residual_m": float(np.linalg.norm(predicted[:3, 3] - observed[:3, 3])),
            "rotation_residual_deg": math.degrees(math.acos(float(np.clip(
                (np.trace(observed[:3, :3].T @ predicted[:3, :3]) - 1.0) * 0.5,
                -1.0, 1.0,
            )))),
            "observed_translation_scale": evaluation_scale,
        })
    before_translation_p80 = (
        float(np.percentile([row["translation_residual_m"] for row in baseline_withheld_residuals], 80.0))
        if baseline_withheld_residuals else None
    )
    after_translation_p80 = (
        float(np.percentile([row["translation_residual_m"] for row in withheld_residuals], 80.0))
        if withheld_residuals else None
    )
    after_rotation_p80 = (
        float(np.percentile([row["rotation_residual_deg"] for row in withheld_residuals], 80.0))
        if withheld_residuals else None
    )
    holdout_status = "unavailable_no_verified_withheld_constraints"
    if withheld_residuals:
        holdout_status = "passed"
        if (
            after_translation_p80 is None
            or after_rotation_p80 is None
            or after_translation_p80 > settings.max_withheld_translation_residual_m
            or after_rotation_p80 > settings.max_withheld_rotation_residual_deg
            or (
                before_translation_p80 is not None
                and after_translation_p80
                > before_translation_p80 + settings.withheld_no_regression_tolerance_m
            )
        ):
            holdout_status = "failed"
    refinement["withheld_evaluation"] = {
        "status": holdout_status,
        "candidate_count": len(withheld_verified),
        "verified_constraint_count": len(withheld_residuals),
        "translation_residual_p50_m": float(np.median([row["translation_residual_m"] for row in withheld_residuals])) if withheld_residuals else None,
        "translation_residual_p80_m": after_translation_p80,
        "rotation_residual_p80_deg": after_rotation_p80,
        "baseline_translation_residual_p80_m": before_translation_p80,
        "no_regression_tolerance_m": settings.withheld_no_regression_tolerance_m,
        "residuals": withheld_residuals,
        "fit_excluded": True,
        "evaluation_scale_applied_to_visual_edges": evaluation_scale,
    }
    holdout_gate_passed = holdout_status == "passed"
    if refinement.get("raw_materialized") is True and not holdout_gate_passed:
        refinement["raw_materialized"] = False
        refinement["materialization_rejection_reason"] = (
            "withheld temporal evaluation is unavailable or failed; candidate remains review_only"
        )
    scale = float(refinement.get("scale_change") or 1.0)
    if abs(scale - 1.0) > 0.02 and not holdout_gate_passed:
        refinement["world_alignment_revalidation_required"] = True
        if revalidate_scaled_carrier is None:
            refinement.update(
                {
                    "status": "rejected_world_alignment_revalidation_required",
                    "rejection_reason": (
                        "metric VIO implies a scale change, but temporal holdout "
                        "evidence is unavailable or failed"
                    ),
                }
            )
    # Keep a stable candidate carrier whenever the solver/deformation gates
    # produced finite poses.  It is review evidence until the temporal gate
    # and, for non-unit metric scale, a fresh target alignment both pass.
    solver_candidate_ready = (
        "corrected_poses" in locals()
        and refinement.get("pose_deformation") is not None
        and refinement.get("status") != "rejected_solver_or_deformation_gate"
    )
    keep_candidate_evidence = solver_candidate_ready and (
        refinement.get("raw_materialized") is True or abs(scale - 1.0) > 0.02
    )
    if keep_candidate_evidence:
        materialized = _materialize_refined_raw(
            raw_views, evaluation_poses, output_dir, scale=scale,
            source_identity=source_identity,
        )
        if abs(scale - 1.0) > 0.02 and holdout_gate_passed:
            if revalidate_scaled_carrier is None:
                refinement.update(
                    {
                        "raw_materialized": False,
                        "status": "rejected_world_alignment_revalidation_required",
                        "rejection_reason": (
                            "metric VIO implies a scale change; the refined carrier "
                            "must be aligned against the target before conditioning"
                        ),
                        "world_alignment_revalidation_required": True,
                    }
                )
            else:
                revalidation_dir = output_dir / "world_alignment_revalidation"
                try:
                    alignment_result = revalidate_scaled_carrier(
                        Path(materialized["raw_dir"]),
                        Path(materialized["source_manifest"]),
                        revalidation_dir,
                        scale,
                    )
                    validated_result = _validate_scaled_alignment_result(
                        alignment_result,
                        result_dir=revalidation_dir,
                        scale_factor=scale,
                    )
                except Exception as exc:
                    validated_result = None
                    refinement["world_alignment_revalidation_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                if validated_result is None:
                    refinement.update(
                        {
                            "raw_materialized": False,
                            "status": "rejected_world_alignment_revalidation",
                            "rejection_reason": (
                                "scaled carrier alignment did not pass its target "
                                "quality gate"
                            ),
                            "world_alignment_revalidation_required": True,
                        }
                    )
                else:
                    refinement["world_alignment_revalidation"] = validated_result
        # Keep candidate files and diagnostics at stable paths even when a
        # later gate rejects the candidate. PCF selects them only when
        # raw_materialized remains true.
        refinement["materialized"] = materialized
    rejection_counts: dict[str, int] = {}
    for row in verified:
        reason = row.get("rejection_reason")
        if reason:
            rejection_counts[str(reason)] = rejection_counts.get(str(reason), 0) + 1
    accepted_surface_rows = [
        row.get("three_d_geometry")
        for row in verified
        if row.get("status") == "accepted" and isinstance(row.get("three_d_geometry"), dict)
    ]
    accepted_surface_p80 = [
        float(row["residual_p80_m"])
        for row in accepted_surface_rows
        if isinstance(row.get("residual_p80_m"), (int, float))
    ]
    accepted_source_views = {
        int(row["source_view"])
        for row in verified
        if row.get("status") == "accepted" and "source_view" in row
    }
    accepted_target_views = {
        int(row["target_view"])
        for row in verified
        if row.get("status") == "accepted" and "target_view" in row
    }
    valid_pixel_rows = [
        np.asarray(row["data"]["mask"], dtype=bool)
        & np.isfinite(np.asarray(row["data"]["depth_z"], dtype=np.float64))
        & (np.asarray(row["data"]["depth_z"], dtype=np.float64) > 0.05)
        for row in raw_views
    ]
    valid_pixel_fraction = float(
        np.count_nonzero(np.concatenate([row.reshape(-1) for row in valid_pixel_rows]))
        / max(1, sum(row.size for row in valid_pixel_rows))
    )
    report = {
        "schema": TRAJECTORY_REFINEMENT_SCHEMA,
        "generated_at": _utc_now(),
        "scan_id": scan_dir.name,
        "source_raw": str(da3_raw),
        "source_raw_view_count": len(raw_views),
        "coordinate_frame": source_identity["coordinate_frame"],
        "source_identity": source_identity,
        "camera_axes": "opencv_x_right_y_down_z_forward",
        "units": "m",
        "settings": asdict(settings),
        "retrieval": {
            "candidate_count": len(candidates),
            "withheld_candidate_count": len(withheld_verified),
            "min_nonadjacent_gap": settings.min_nonadjacent_gap,
            "max_candidate_pairs": settings.max_candidate_pairs,
        },
        "constraints": verified,
        "constraint_summary": {
            "accepted_count": int(refinement.get("accepted_constraint_count") or 0),
            "rejected_count": sum(1 for row in verified if row.get("rejection_reason")),
            "rejection_reasons": rejection_counts,
            "accepted_source_view_count": len(accepted_source_views),
            "accepted_target_view_count": len(accepted_target_views),
            "surface_residual_p80_m_median_across_constraints": float(np.median(accepted_surface_p80)) if accepted_surface_p80 else None,
        },
        "coverage": {
            "raw_valid_depth_pixel_fraction": valid_pixel_fraction,
            "raw_view_count": len(raw_views),
            "candidate_pair_count": len(candidates),
            "withheld_candidate_pair_count": len(withheld_verified),
            "accepted_constraint_source_view_count": len(accepted_source_views),
            "accepted_constraint_target_view_count": len(accepted_target_views),
        },
        "vio": {
            "path": str(vio_constraints.resolve()) if vio_constraints is not None else None,
            **vio_info,
            "status": vio_status,
        },
        "refinement": refinement,
        "provenance": {
            "trajectory_refinement_source_sha256": _sha256(Path(__file__)),
            "source_identity": "prepared_capture_identity_plus_retained_raw_rgb_depth_and_poses",
            "matching_image_projection": "raw_model_rgb_on_retained_depth_and_intrinsics_grid",
            "prepared_manifest_sha256": _sha256(scan_dir / "prepared_frames_manifest.json"),
            "parent_prepared_frame_count": parent_frame_count,
            "included_prepared_indices": list(prepared_indices) if prepared_indices is not None else list(range(parent_frame_count)),
            "raw_view_sha256": [
                {"index": index, "sha256": _sha256(row["path"])}
                for index, row in enumerate(raw_views)
            ],
            "alignment_fitted_on_validation_points": False,
            "visual_constraints_require_actual_2d_matches": True,
            "visual_constraints_require_depth_backed_3d_geometry": True,
            "endpoint_pseudo_loop_applied": False,
        },
    }
    report_path = output_dir / "trajectory_refinement_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = [
    "TRAJECTORY_REFINEMENT_SCHEMA",
    "TrajectoryRefinementError",
    "TrajectoryRefinementSettings",
    "run_trajectory_refinement",
    "_fit_3d_rigid",
    "_retrieve_nonadjacent_pairs",
    "_verify_visual_revisit",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify RGB/depth revisits and rebuild a retained review trajectory.")
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--source-raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--withheld-range", type=int, nargs=2, action="append", required=True,
        metavar=("START", "END"),
        help="Predeclared zero-based temporal interval [START, END), excluded from visual fitting; repeatable.",
    )
    args = parser.parse_args()
    report = run_trajectory_refinement(
        args.scan_dir, args.source_raw_root, args.output_dir,
        TrajectoryRefinementSettings(withheld_ranges=tuple(tuple(row) for row in args.withheld_range)),
    )
    print(json.dumps({
        "report": report["report_path"],
        "constraints": report["constraint_summary"],
        "refinement": report["refinement"],
    }, indent=2), flush=True)
    return 0 if report["refinement"].get("raw_materialized") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
