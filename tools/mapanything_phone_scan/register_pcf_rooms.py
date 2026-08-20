#!/usr/bin/env python3
"""Register two accepted PCF room walks from shared RGB-D observations.

This command intentionally does not run whole-cloud ICP.  It retrieves likely
cross-walk RGB overlap, verifies mutual local-feature matches, lifts those
matches through the accepted PCF depth, and estimates one gravity-preserving
SE(3) correction for the moving room in the fixed room's backend-world frame.

The accepted per-room PCFs remain immutable.  Results are written as a separate
multi-room review artifact with the exact source manifests and evidence needed
to reproduce or reject the registration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.optimize import least_squares


class PCFRoomRegistrationError(RuntimeError):
    """Raised when cross-room evidence cannot support a safe registration."""


@dataclass(frozen=True)
class Settings:
    retrieval_features: int = 1_600
    retrieval_ratio: float = 0.80
    retrieval_pairs: int = 180
    detailed_features: int = 4_500
    detailed_ratio: float = 0.72
    minimum_mutual_matches: int = 10
    minimum_pair_3d_inliers: int = 8
    minimum_pair_inlier_fraction: float = 0.30
    minimum_pair_span_m: float = 0.70
    pair_ransac_threshold_m: float = 0.18
    global_ransac_threshold_m: float = 0.15
    cluster_yaw_deg: float = 4.0
    cluster_translation_m: float = 0.60
    minimum_distinct_fixed_views: int = 3
    minimum_distinct_moving_views: int = 3
    minimum_global_3d_inliers: int = 30
    maximum_global_median_m: float = 0.10
    maximum_global_p80_m: float = 0.18
    maximum_heldout_median_m: float = 0.14
    voxel_m: float = 0.025
    random_seed: int = 19


@dataclass
class RawView:
    index: int
    path: Path
    rgb: np.ndarray
    feature_rgb: np.ndarray
    mask: np.ndarray
    confidence: np.ndarray
    uncertain: np.ndarray
    world_points_local: np.ndarray
    camera_pose_local: np.ndarray
    intrinsics: np.ndarray


@dataclass
class Room:
    name: str
    prior_id: str
    scan_id: str
    raw_root: Path
    scan_dir: Path
    world_manifest: Path
    world_from_local: np.ndarray
    views: list[RawView]


@dataclass
class PairEvidence:
    moving_view: int
    fixed_view: int
    retrieval_score: float
    mutual_match_count: int
    geometric_match_count: int
    moving_points_world: np.ndarray
    fixed_points_world: np.ndarray
    moving_pixels: np.ndarray
    fixed_pixels: np.ndarray
    transform: np.ndarray
    inlier_mask: np.ndarray
    yaw_deg: float
    translation: np.ndarray
    median_m: float
    p80_m: float
    span_m: float


@dataclass
class PnPCorrection:
    source_view: int
    target_view: int
    transform: np.ndarray
    object_points_source_world: np.ndarray
    image_points_target_model: np.ndarray
    inlier_mask: np.ndarray
    inlier_count: int
    inlier_fraction: float
    reprojection_median_px: float
    reprojection_p80_px: float
    non_yaw_rotation_deg: float
    support_span_m: float


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_u8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    scale = 255.0 if array.size and float(np.nanmax(array)) <= 1.5 else 1.0
    return np.clip(array * scale, 0.0, 255.0).astype(np.uint8)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    return (transform[:3, :3] @ array.T).T + transform[:3, 3]


def _manifest_transform(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    alignment = payload.get("alignment")
    if not isinstance(alignment, dict):
        raise PCFRoomRegistrationError(f"missing alignment in {path}")
    rotation = np.asarray(alignment.get("rotation_row_major"), dtype=np.float64)
    translation = np.asarray(alignment.get("translation"), dtype=np.float64)
    scale = float(alignment.get("scale", float("nan")))
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise PCFRoomRegistrationError(f"malformed alignment in {path}")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise PCFRoomRegistrationError(f"non-finite alignment in {path}")
    if not math.isfinite(scale) or abs(scale - 1.0) > 0.01:
        raise PCFRoomRegistrationError(
            f"accepted PCF transform is not rigid metric scale in {path}: {scale}"
        )
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise PCFRoomRegistrationError(
            f"accepted PCF transform contains scale or shear in {path}"
        )
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-3):
        raise PCFRoomRegistrationError(f"reflected or invalid rotation in {path}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform


def _load_view(path: Path, source_frame: Path, index: int) -> RawView:
    with np.load(path, allow_pickle=False) as row:
        required = {
            "model_rgb",
            "mask",
            "confidence",
            "world_points",
            "camera_pose",
            "intrinsics",
        }
        missing = sorted(required.difference(row.files))
        if missing:
            raise PCFRoomRegistrationError(f"{path} is missing {missing}")
        rgb = _as_u8(row["model_rgb"])
        mask = np.asarray(row["mask"], dtype=bool)
        confidence = np.asarray(row["confidence"], dtype=np.float32)
        uncertain = (
            np.asarray(row["cross_model_uncertain"], dtype=bool)
            if "cross_model_uncertain" in row.files
            else np.zeros_like(mask)
        )
        world_points = np.asarray(row["world_points"], dtype=np.float32)
        camera_pose = np.asarray(row["camera_pose"], dtype=np.float64)
        intrinsics = np.asarray(row["intrinsics"], dtype=np.float64)
    source_bgr = cv2.imread(str(source_frame), cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise PCFRoomRegistrationError(f"prepared RGB frame is unreadable: {source_frame}")
    feature_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    largest = max(feature_rgb.shape[:2])
    if largest > 1_200:
        scale = 1_200.0 / float(largest)
        feature_rgb = cv2.resize(
            feature_rgb,
            (
                int(round(feature_rgb.shape[1] * scale)),
                int(round(feature_rgb.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
    expected = rgb.shape[:2]
    if mask.shape != expected or confidence.shape != expected:
        raise PCFRoomRegistrationError(f"RGB/evidence shape mismatch in {path}")
    if world_points.shape != (*expected, 3):
        raise PCFRoomRegistrationError(f"world point shape mismatch in {path}")
    if camera_pose.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise PCFRoomRegistrationError(f"camera geometry is malformed in {path}")
    return RawView(
        index=index,
        path=path,
        rgb=rgb,
        feature_rgb=feature_rgb,
        mask=mask,
        confidence=confidence,
        uncertain=uncertain,
        world_points_local=world_points,
        camera_pose_local=camera_pose,
        intrinsics=intrinsics,
    )


def _load_room(
    *,
    name: str,
    prior_id: str,
    scan_id: str,
    scan_dir: Path,
    pcf_root: Path,
    world_manifest: Path,
) -> Room:
    raw_root = pcf_root / "raw"
    paths = sorted(raw_root.glob("view_*.npz"))
    if len(paths) < 2:
        raise PCFRoomRegistrationError(f"too few PCF raw views in {raw_root}")
    prepared = json.loads(
        (scan_dir / "prepared_frames_manifest.json").read_text(encoding="utf-8")
    )
    if len(prepared.get("frames", [])) != len(paths):
        raise PCFRoomRegistrationError(
            f"prepared/raw view count mismatch for {name}: "
            f"{len(prepared.get('frames', []))} != {len(paths)}"
        )
    prepared_rows = prepared["frames"]
    views = [
        _load_view(
            path,
            scan_dir / str(prepared_rows[index]["frame"]),
            index,
        )
        for index, path in enumerate(paths)
    ]
    return Room(
        name=name,
        prior_id=prior_id,
        scan_id=scan_id,
        raw_root=raw_root,
        scan_dir=scan_dir,
        world_manifest=world_manifest,
        world_from_local=_manifest_transform(world_manifest),
        views=views,
    )


def _orb_features(view: RawView, settings: Settings) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    gray = cv2.cvtColor(view.feature_rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.ORB_create(
        nfeatures=settings.retrieval_features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        fastThreshold=10,
    )
    return detector.detectAndCompute(gray, None)


def _sift_features(view: RawView, settings: Settings) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    gray = cv2.cvtColor(view.feature_rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.SIFT_create(
        nfeatures=settings.detailed_features,
        contrastThreshold=0.012,
    )
    return detector.detectAndCompute(gray, None)


def _ratio_matches(
    matcher: cv2.DescriptorMatcher,
    first: np.ndarray | None,
    second: np.ndarray | None,
    ratio: float,
) -> list[cv2.DMatch]:
    if first is None or second is None or len(first) < 4 or len(second) < 4:
        return []
    pairs = matcher.knnMatch(first, second, k=2)
    return [
        leading
        for pair in pairs
        if len(pair) == 2
        for leading, trailing in [pair]
        if leading.distance < ratio * trailing.distance
    ]


def _geometric_inlier_mask(
    first_xy: np.ndarray,
    second_xy: np.ndarray,
) -> np.ndarray:
    count = len(first_xy)
    if count < 6:
        return np.zeros(count, dtype=bool)
    masks: list[np.ndarray] = []
    if count >= 8:
        _, fundamental_mask = cv2.findFundamentalMat(
            first_xy,
            second_xy,
            cv2.FM_RANSAC,
            1.75,
            0.999,
        )
        if fundamental_mask is not None and fundamental_mask.size == count:
            masks.append(fundamental_mask.reshape(-1).astype(bool))
    _, homography_mask = cv2.findHomography(
        first_xy,
        second_xy,
        cv2.RANSAC,
        3.0,
    )
    if homography_mask is not None and homography_mask.size == count:
        masks.append(homography_mask.reshape(-1).astype(bool))
    if not masks:
        return np.zeros(count, dtype=bool)
    return max(masks, key=np.count_nonzero)


def _retrieve_pairs(
    moving: Room,
    fixed: Room,
    settings: Settings,
) -> list[tuple[int, int, float]]:
    moving_features = [_orb_features(view, settings) for view in moving.views]
    fixed_features = [_orb_features(view, settings) for view in fixed.views]
    def indexed_candidates(
        query_features: list[tuple[list[cv2.KeyPoint], np.ndarray | None]],
        train_features: list[tuple[list[cv2.KeyPoint], np.ndarray | None]],
    ) -> set[tuple[int, int]]:
        train_rows = [
            descriptors
            for _, descriptors in train_features
            if descriptors is not None and len(descriptors) >= 4
        ]
        if not train_rows:
            return set()
        owners = np.concatenate(
            [
                np.full(len(descriptors), index, dtype=np.int32)
                for index, (_, descriptors) in enumerate(train_features)
                if descriptors is not None and len(descriptors) >= 4
            ]
        )
        train_descriptors = np.concatenate(train_rows)
        matcher = cv2.FlannBasedMatcher(
            {
                "algorithm": 6,
                "table_number": 12,
                "key_size": 20,
                "multi_probe_level": 2,
            },
            {"checks": 64},
        )
        candidates: set[tuple[int, int]] = set()
        for query_index, (_, descriptors) in enumerate(query_features):
            if descriptors is None or len(descriptors) < 4:
                continue
            matches = _ratio_matches(
                matcher,
                descriptors,
                train_descriptors,
                settings.retrieval_ratio,
            )
            if not matches:
                continue
            counts = np.bincount(
                owners[[row.trainIdx for row in matches]],
                minlength=len(train_features),
            )
            for train_index in np.argsort(counts)[-6:]:
                if counts[train_index] >= 3:
                    candidates.add((query_index, int(train_index)))
        return candidates

    candidate_pairs = indexed_candidates(moving_features, fixed_features)
    candidate_pairs.update(
        (moving_index, fixed_index)
        for fixed_index, moving_index in indexed_candidates(
            fixed_features,
            moving_features,
        )
    )
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    scored: list[tuple[int, int, float]] = []
    for moving_index, fixed_index in sorted(candidate_pairs):
        moving_keypoints, moving_descriptors = moving_features[moving_index]
        fixed_keypoints, fixed_descriptors = fixed_features[fixed_index]
        matches = _ratio_matches(
            matcher,
            moving_descriptors,
            fixed_descriptors,
            settings.retrieval_ratio,
        )
        if len(matches) < 8:
            continue
        moving_xy = np.float32(
            [moving_keypoints[match.queryIdx].pt for match in matches]
        )
        fixed_xy = np.float32(
            [fixed_keypoints[match.trainIdx].pt for match in matches]
        )
        geometric = _geometric_inlier_mask(moving_xy, fixed_xy)
        inliers = int(np.count_nonzero(geometric))
        if inliers < 6:
            continue
        coverage = float(
            np.linalg.norm(np.ptp(moving_xy[geometric], axis=0))
            + np.linalg.norm(np.ptp(fixed_xy[geometric], axis=0))
        )
        score = float(inliers + 0.01 * coverage)
        scored.append((moving_index, fixed_index, score))
    scored.sort(key=lambda row: (-row[2], row[0], row[1]))
    selected = scored[: settings.retrieval_pairs]
    # Retain the best two candidates for every view represented in retrieval so
    # one visually dominant doorway frame cannot erase temporal diversity.
    for side in (0, 1):
        grouped: dict[int, list[tuple[int, int, float]]] = {}
        for row in scored:
            grouped.setdefault(row[side], []).append(row)
        for rows in grouped.values():
            selected.extend(rows[:2])
    unique = {(row[0], row[1]): row for row in selected}
    return sorted(unique.values(), key=lambda row: (-row[2], row[0], row[1]))


def _mutual_sift_matches(
    moving_features: tuple[list[cv2.KeyPoint], np.ndarray | None],
    fixed_features: tuple[list[cv2.KeyPoint], np.ndarray | None],
    ratio: float,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    moving_keypoints, moving_descriptors = moving_features
    fixed_keypoints, fixed_descriptors = fixed_features
    # Approximate KD-tree matching keeps the exhaustive 48x48 retrieval pass
    # practical without changing the later mutual/geometric/metric gates.
    matcher = cv2.FlannBasedMatcher(
        {"algorithm": 1, "trees": 5},
        {"checks": 96},
    )
    forward_rows = _ratio_matches(
        matcher,
        moving_descriptors,
        fixed_descriptors,
        ratio,
    )
    reverse_rows = _ratio_matches(
        matcher,
        fixed_descriptors,
        moving_descriptors,
        ratio,
    )
    forward = {int(row.queryIdx): int(row.trainIdx) for row in forward_rows}
    reverse = {int(row.queryIdx): int(row.trainIdx) for row in reverse_rows}
    mutual = [
        (moving_index, fixed_index)
        for moving_index, fixed_index in forward.items()
        if reverse.get(fixed_index) == moving_index
    ]
    if len(mutual) < 6:
        return mutual, np.zeros(len(mutual), dtype=bool)
    moving_xy = np.float32(
        [moving_keypoints[moving_index].pt for moving_index, _ in mutual]
    )
    fixed_xy = np.float32(
        [fixed_keypoints[fixed_index].pt for _, fixed_index in mutual]
    )
    return mutual, _geometric_inlier_mask(moving_xy, fixed_xy)


def _point_at_feature(view: RawView, pixel: tuple[float, float]) -> np.ndarray | None:
    height, width = view.mask.shape
    feature_height, feature_width = view.feature_rgb.shape[:2]
    x = int(round(pixel[0] * width / feature_width))
    y = int(round(pixel[1] * height / feature_height))
    candidates: list[tuple[float, float, np.ndarray]] = []
    for radius in range(0, 3):
        for row in range(max(0, y - radius), min(height, y + radius + 1)):
            for column in range(max(0, x - radius), min(width, x + radius + 1)):
                if max(abs(row - y), abs(column - x)) != radius:
                    continue
                if not view.mask[row, column] or view.uncertain[row, column]:
                    continue
                confidence = float(view.confidence[row, column])
                point = np.asarray(view.world_points_local[row, column], dtype=np.float64)
                if confidence < 0.08 or not np.isfinite(point).all():
                    continue
                candidates.append((float(radius), -confidence, point))
        if candidates:
            break
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row[0], row[1]))[2]


def _feature_to_model_pixels(view: RawView, pixels: np.ndarray) -> np.ndarray:
    array = np.asarray(pixels, dtype=np.float64)
    feature_height, feature_width = view.feature_rgb.shape[:2]
    model_height, model_width = view.mask.shape
    scale = np.asarray(
        [model_width / feature_width, model_height / feature_height],
        dtype=np.float64,
    )
    return array * scale


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _pnp_correction(
    source: Room,
    target: Room,
    source_index: int,
    target_index: int,
    source_feature_pixels: np.ndarray,
    target_feature_pixels: np.ndarray,
    *,
    minimum_inliers: int = 10,
) -> PnPCorrection | None:
    source_view = source.views[source_index]
    target_view = target.views[target_index]
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    target_model_pixels = _feature_to_model_pixels(
        target_view,
        target_feature_pixels,
    )
    for source_pixel, target_pixel in zip(
        source_feature_pixels,
        target_model_pixels,
    ):
        source_point = _point_at_feature(
            source_view,
            (float(source_pixel[0]), float(source_pixel[1])),
        )
        if source_point is None:
            continue
        object_points.append(
            _transform_points(source_point[None, :], source.world_from_local)[0]
        )
        image_points.append(np.asarray(target_pixel, dtype=np.float64))
    if len(object_points) < minimum_inliers:
        return None
    object_array = np.asarray(object_points, dtype=np.float64)
    image_array = np.asarray(image_points, dtype=np.float64)
    span = float(np.linalg.norm(np.ptp(object_array, axis=0)))
    if span < 0.8:
        return None
    # OpenCV's PnP RANSAC consumes a process-global RNG.  Reset it per directed
    # view pair so a valid loop closure cannot appear or disappear merely
    # because unrelated candidates were evaluated first.
    cv2.setRNGSeed(
        int(((source_index + 1) * 1_009 + (target_index + 1) * 9_176) % 2_147_483_647)
    )
    solved, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        object_array,
        image_array,
        target_view.intrinsics,
        None,
        iterationsCount=1_500,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    inlier_count = int(len(inliers)) if solved and inliers is not None else 0
    if not solved or inliers is None or inlier_count < minimum_inliers:
        return None
    inlier_indices = np.asarray(inliers, dtype=np.int64).reshape(-1)
    if hasattr(cv2, "solvePnPRefineLM"):
        rotation_vector, translation_vector = cv2.solvePnPRefineLM(
            object_array[inlier_indices],
            image_array[inlier_indices],
            target_view.intrinsics,
            None,
            rotation_vector,
            translation_vector,
        )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    target_camera_from_source_world = np.eye(4, dtype=np.float64)
    target_camera_from_source_world[:3, :3] = rotation
    target_camera_from_source_world[:3, 3] = np.asarray(
        translation_vector,
        dtype=np.float64,
    ).reshape(3)
    target_camera_to_world = (
        target.world_from_local @ target_view.camera_pose_local
    )
    full_transform = target_camera_to_world @ target_camera_from_source_world
    yaw = math.atan2(float(full_transform[0, 2]), float(full_transform[0, 0]))
    initial_parameters = np.asarray([yaw, *full_transform[:3, 3]], dtype=np.float64)
    non_yaw = _rotation_angle_deg(
        _yaw_transform(initial_parameters)[:3, :3].T @ full_transform[:3, :3]
    )
    target_camera_from_world = np.linalg.inv(target_camera_to_world)
    inlier_objects = object_array[inlier_indices]
    inlier_images = image_array[inlier_indices]

    def gravity_constrained_reprojection(parameters: np.ndarray) -> np.ndarray:
        corrected_points = _transform_points(
            inlier_objects,
            _yaw_transform(parameters),
        )
        camera_coordinates = _transform_points(
            corrected_points,
            target_camera_from_world,
        )
        safe_z = np.maximum(camera_coordinates[:, 2], 0.05)
        projected_points = (target_view.intrinsics @ camera_coordinates.T).T
        projected_points = projected_points[:, :2] / safe_z[:, None]
        residual = projected_points - inlier_images
        behind = camera_coordinates[:, 2] <= 0.05
        residual[behind] = 100.0
        return residual.reshape(-1)

    optimized = least_squares(
        gravity_constrained_reprojection,
        initial_parameters,
        loss="huber",
        f_scale=2.0,
        max_nfev=300,
    )
    transform = _yaw_transform(optimized.x)
    corrected = _transform_points(inlier_objects, transform)
    camera_points = _transform_points(corrected, target_camera_from_world)
    visible = camera_points[:, 2] > 0.05
    if int(np.count_nonzero(visible)) < minimum_inliers:
        return None
    projected = (target_view.intrinsics @ camera_points[visible].T).T
    projected = projected[:, :2] / projected[:, 2, None]
    errors = np.linalg.norm(
        projected - inlier_images[visible],
        axis=1,
    )
    median = float(np.median(errors))
    p80 = float(np.percentile(errors, 80.0))
    if non_yaw > 6.0 or median > 4.0 or p80 > 7.0:
        return None
    inlier_mask = np.zeros(len(object_array), dtype=bool)
    inlier_mask[inlier_indices] = True
    return PnPCorrection(
        source_view=source_index,
        target_view=target_index,
        transform=transform,
        object_points_source_world=object_array,
        image_points_target_model=image_array,
        inlier_mask=inlier_mask,
        inlier_count=inlier_count,
        inlier_fraction=float(inlier_count / len(object_array)),
        reprojection_median_px=median,
        reprojection_p80_px=p80,
        non_yaw_rotation_deg=non_yaw,
        support_span_m=span,
    )


def _pair_evidence_from_pixels(
    moving: Room,
    fixed: Room,
    moving_index: int,
    fixed_index: int,
    retrieval_score: float,
    moving_pixels_input: np.ndarray,
    fixed_pixels_input: np.ndarray,
    *,
    mutual_match_count: int,
    geometric_match_count: int,
    settings: Settings,
) -> PairEvidence | None:
    moving_view = moving.views[moving_index]
    fixed_view = fixed.views[fixed_index]
    moving_points: list[np.ndarray] = []
    fixed_points: list[np.ndarray] = []
    moving_pixels: list[tuple[float, float]] = []
    fixed_pixels: list[tuple[float, float]] = []
    for moving_pixel_array, fixed_pixel_array in zip(
        moving_pixels_input,
        fixed_pixels_input,
    ):
        moving_pixel = (float(moving_pixel_array[0]), float(moving_pixel_array[1]))
        fixed_pixel = (float(fixed_pixel_array[0]), float(fixed_pixel_array[1]))
        moving_point = _point_at_feature(moving_view, moving_pixel)
        fixed_point = _point_at_feature(fixed_view, fixed_pixel)
        if moving_point is None or fixed_point is None:
            continue
        moving_points.append(
            _transform_points(moving_point[None, :], moving.world_from_local)[0]
        )
        fixed_points.append(
            _transform_points(fixed_point[None, :], fixed.world_from_local)[0]
        )
        moving_pixels.append(moving_pixel)
        fixed_pixels.append(fixed_pixel)
    if len(moving_points) < settings.minimum_pair_3d_inliers:
        return None
    moving_array = np.asarray(moving_points, dtype=np.float64)
    fixed_array = np.asarray(fixed_points, dtype=np.float64)
    try:
        transform, inliers = _ransac_yaw_translation(
            moving_array,
            fixed_array,
            threshold_m=settings.pair_ransac_threshold_m,
            random_seed=settings.random_seed + 97 * moving_index + fixed_index,
            iterations=700,
        )
    except PCFRoomRegistrationError:
        return None
    inlier_count = int(np.count_nonzero(inliers))
    fraction = float(inlier_count / len(moving_array))
    if (
        inlier_count < settings.minimum_pair_3d_inliers
        or fraction < settings.minimum_pair_inlier_fraction
    ):
        return None
    residual = np.linalg.norm(
        _transform_points(moving_array[inliers], transform) - fixed_array[inliers],
        axis=1,
    )
    combined = np.concatenate((moving_array[inliers], fixed_array[inliers]), axis=0)
    span = float(np.linalg.norm(np.ptp(combined, axis=0)))
    if span < settings.minimum_pair_span_m:
        return None
    parameters = _parameters_from_transform(transform)
    return PairEvidence(
        moving_view=moving_index,
        fixed_view=fixed_index,
        retrieval_score=retrieval_score,
        mutual_match_count=mutual_match_count,
        geometric_match_count=geometric_match_count,
        moving_points_world=moving_array,
        fixed_points_world=fixed_array,
        moving_pixels=np.asarray(moving_pixels, dtype=np.float32),
        fixed_pixels=np.asarray(fixed_pixels, dtype=np.float32),
        transform=transform,
        inlier_mask=inliers,
        yaw_deg=math.degrees(float(parameters[0])),
        translation=np.asarray(parameters[1:], dtype=np.float64),
        median_m=float(np.median(residual)),
        p80_m=float(np.percentile(residual, 80.0)),
        span_m=span,
    )


def _yaw_transform(parameters: Iterable[float]) -> np.ndarray:
    yaw, tx, ty, tz = [float(value) for value in parameters]
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )
    transform[:3, 3] = [tx, ty, tz]
    return transform


def _parameters_from_transform(transform: np.ndarray) -> np.ndarray:
    yaw = math.atan2(float(transform[0, 2]), float(transform[0, 0]))
    return np.asarray([yaw, *transform[:3, 3]], dtype=np.float64)


def _fit_yaw_translation_closed(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_xz = source[:, [0, 2]]
    target_xz = target[:, [0, 2]]
    source_center = np.mean(source_xz, axis=0)
    target_center = np.mean(target_xz, axis=0)
    covariance = (source_xz - source_center).T @ (target_xz - target_center)
    left, _, right = np.linalg.svd(covariance)
    rotation_2d = right.T @ left.T
    if float(np.linalg.det(rotation_2d)) < 0.0:
        right[-1, :] *= -1.0
        rotation_2d = right.T @ left.T
    yaw = math.atan2(float(rotation_2d[0, 1]), float(rotation_2d[0, 0]))
    rotation = _yaw_transform([yaw, 0.0, 0.0, 0.0])[:3, :3]
    translation = np.median(target - (rotation @ source.T).T, axis=0)
    return _yaw_transform(np.asarray([yaw, *translation], dtype=np.float64))


def _fit_yaw_translation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    initial = _parameters_from_transform(_fit_yaw_translation_closed(source, target))

    def residual(parameters: np.ndarray) -> np.ndarray:
        predicted = _transform_points(source, _yaw_transform(parameters))
        return (predicted - target).reshape(-1)

    optimized = least_squares(
        residual,
        initial,
        loss="huber",
        f_scale=0.08,
        max_nfev=200,
    )
    return _yaw_transform(optimized.x)


def _ransac_yaw_translation(
    source: np.ndarray,
    target: np.ndarray,
    *,
    threshold_m: float,
    random_seed: int,
    iterations: int = 1_500,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must be matching Nx3 arrays")
    if len(source) < 3:
        raise PCFRoomRegistrationError("too few points for rigid registration")
    generator = np.random.default_rng(random_seed)
    best_transform: np.ndarray | None = None
    best_mask = np.zeros(len(source), dtype=bool)
    best_key = (0, -float("inf"))
    for _ in range(iterations):
        indices = generator.choice(len(source), size=3, replace=False)
        sample_source = source[indices]
        if float(np.linalg.norm(np.ptp(sample_source[:, [0, 2]], axis=0))) < 0.15:
            continue
        transform = _fit_yaw_translation_closed(sample_source, target[indices])
        residual = np.linalg.norm(_transform_points(source, transform) - target, axis=1)
        mask = residual <= threshold_m
        count = int(np.count_nonzero(mask))
        median = float(np.median(residual[mask])) if count else float("inf")
        key = (count, -median)
        if key > best_key:
            best_key = key
            best_transform = transform
            best_mask = mask
    if best_transform is None or int(np.count_nonzero(best_mask)) < 3:
        raise PCFRoomRegistrationError("no rigid cross-room hypothesis was found")
    refined = _fit_yaw_translation(source[best_mask], target[best_mask])
    residual = np.linalg.norm(_transform_points(source, refined) - target, axis=1)
    return refined, residual <= threshold_m


def _pair_evidence(
    moving: Room,
    fixed: Room,
    moving_index: int,
    fixed_index: int,
    retrieval_score: float,
    moving_features: tuple[list[cv2.KeyPoint], np.ndarray | None],
    fixed_features: tuple[list[cv2.KeyPoint], np.ndarray | None],
    settings: Settings,
) -> PairEvidence | None:
    mutual, geometric_mask = _mutual_sift_matches(
        moving_features,
        fixed_features,
        settings.detailed_ratio,
    )
    if len(mutual) < settings.minimum_mutual_matches:
        return None
    moving_keypoints = moving_features[0]
    fixed_keypoints = fixed_features[0]
    moving_pixels = np.asarray(
        [
            moving_keypoints[moving_feature].pt
            for keep, (moving_feature, _) in zip(geometric_mask, mutual)
            if keep
        ],
        dtype=np.float32,
    )
    fixed_pixels = np.asarray(
        [
            fixed_keypoints[fixed_feature].pt
            for keep, (_, fixed_feature) in zip(geometric_mask, mutual)
            if keep
        ],
        dtype=np.float32,
    )
    return _pair_evidence_from_pixels(
        moving,
        fixed,
        moving_index,
        fixed_index,
        retrieval_score,
        moving_pixels,
        fixed_pixels,
        mutual_match_count=len(mutual),
        geometric_match_count=int(np.count_nonzero(geometric_mask)),
        settings=settings,
    )


def _angle_distance_deg(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _select_consensus_cluster(
    pairs: list[PairEvidence],
    settings: Settings,
) -> list[PairEvidence]:
    best: list[PairEvidence] = []
    best_key = (0, 0, 0, -float("inf"))
    for seed in pairs:
        members = [
            row
            for row in pairs
            if _angle_distance_deg(row.yaw_deg, seed.yaw_deg)
            <= settings.cluster_yaw_deg
            and float(np.linalg.norm(row.translation - seed.translation))
            <= settings.cluster_translation_m
        ]
        moving_views = len({row.moving_view for row in members})
        fixed_views = len({row.fixed_view for row in members})
        support = sum(int(np.count_nonzero(row.inlier_mask)) for row in members)
        median = float(np.median([row.median_m for row in members]))
        key = (min(moving_views, fixed_views), len(members), support, -median)
        if key > best_key:
            best_key = key
            best = members
    return sorted(best, key=lambda row: (row.moving_view, row.fixed_view))


def _deduplicate_correspondences(
    moving: np.ndarray,
    fixed: np.ndarray,
    pair_ids: np.ndarray,
    voxel_m: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    moving_key = np.floor(moving / voxel_m).astype(np.int64)
    fixed_key = np.floor(fixed / voxel_m).astype(np.int64)
    keys = np.concatenate((moving_key, fixed_key), axis=1)
    _, indices = np.unique(keys, axis=0, return_index=True)
    indices.sort()
    return moving[indices], fixed[indices], pair_ids[indices]


def _reprojection_error(
    points_world: np.ndarray,
    pixels: np.ndarray,
    room: Room,
    view_index: int,
) -> np.ndarray:
    camera_to_world = room.world_from_local @ room.views[view_index].camera_pose_local
    world_to_camera = np.linalg.inv(camera_to_world)
    camera_points = _transform_points(points_world, world_to_camera)
    valid = camera_points[:, 2] > 0.05
    errors = np.full(len(points_world), np.inf, dtype=np.float64)
    if not np.any(valid):
        return errors
    projected = (room.views[view_index].intrinsics @ camera_points[valid].T).T
    projected = projected[:, :2] / projected[:, 2, None]
    errors[valid] = np.linalg.norm(projected - pixels[valid], axis=1)
    return errors


def _pair_report(row: PairEvidence) -> dict[str, Any]:
    return {
        "moving_view": row.moving_view,
        "fixed_view": row.fixed_view,
        "retrieval_score": row.retrieval_score,
        "mutual_match_count": row.mutual_match_count,
        "geometric_match_count": row.geometric_match_count,
        "depth_backed_count": int(len(row.moving_points_world)),
        "rigid_inlier_count": int(np.count_nonzero(row.inlier_mask)),
        "rigid_inlier_fraction": float(np.mean(row.inlier_mask)),
        "yaw_correction_deg": row.yaw_deg,
        "translation_correction_m": row.translation.tolist(),
        "residual_median_m": row.median_m,
        "residual_p80_m": row.p80_m,
        "support_span_m": row.span_m,
    }


def _write_match_sheet(
    moving: Room,
    fixed: Room,
    pairs: list[PairEvidence],
    path: Path,
) -> None:
    selected = sorted(
        pairs,
        key=lambda row: (-int(np.count_nonzero(row.inlier_mask)), row.median_m),
    )[:8]
    if not selected:
        return
    tile_width = (
        moving.views[0].feature_rgb.shape[1]
        + fixed.views[0].feature_rgb.shape[1]
    )
    tile_height = max(
        moving.views[0].feature_rgb.shape[0],
        fixed.views[0].feature_rgb.shape[0],
    )
    canvas = np.full((tile_height * len(selected), tile_width, 3), 24, dtype=np.uint8)
    palette = [(255, 96, 96), (96, 220, 255), (160, 255, 128), (255, 210, 96)]
    for row_index, evidence in enumerate(selected):
        moving_rgb = moving.views[evidence.moving_view].feature_rgb
        fixed_rgb = fixed.views[evidence.fixed_view].feature_rgb
        top = row_index * tile_height
        canvas[top : top + tile_height, : moving_rgb.shape[1]] = moving_rgb
        canvas[top : top + tile_height, moving_rgb.shape[1] :] = fixed_rgb
        indices = np.flatnonzero(evidence.inlier_mask)
        for point_index, match_index in enumerate(indices[:80]):
            color = palette[point_index % len(palette)]
            moving_xy = tuple(
                int(round(value))
                for value in evidence.moving_pixels[match_index]
            )
            fixed_xy = evidence.fixed_pixels[match_index].copy()
            fixed_xy[0] += moving_rgb.shape[1]
            fixed_xy_tuple = tuple(int(round(value)) for value in fixed_xy)
            cv2.line(
                canvas,
                (moving_xy[0], top + moving_xy[1]),
                (fixed_xy_tuple[0], top + fixed_xy_tuple[1]),
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            f"{moving.name} {evidence.moving_view:02d} -> {fixed.name} {evidence.fixed_view:02d} | "
            f"{len(indices)} 3D inliers | {evidence.median_m * 100:.1f} cm median",
            (8, top + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _surfel_path(room: Room) -> Path:
    return room.raw_root.parent / "surfel_points.npz"


def _load_surfels(room: Room) -> tuple[np.ndarray, np.ndarray]:
    path = _surfel_path(room)
    with np.load(path, allow_pickle=False) as row:
        points = np.asarray(row["points"], dtype=np.float64)
        colors = np.asarray(row["colors"], dtype=np.uint8)
    return _transform_points(points, room.world_from_local), colors


def _write_topdown(
    moving: Room,
    fixed: Room,
    correction: np.ndarray,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    moving_points, moving_colors = _load_surfels(moving)
    fixed_points, fixed_colors = _load_surfels(fixed)
    corrected_moving = _transform_points(moving_points, correction)
    figure, axes = plt.subplots(1, 2, figsize=(18, 9), constrained_layout=True)
    for axis, transformed, title in (
        (axes[0], moving_points, "Accepted per-room backend transforms"),
        (axes[1], corrected_moving, "Cross-session RGB-D registration"),
    ):
        axis.scatter(
            fixed_points[::2, 0],
            fixed_points[::2, 2],
            s=0.35,
            c=fixed_colors[::2] / 255.0,
            alpha=0.72,
            rasterized=True,
        )
        axis.scatter(
            transformed[::2, 0],
            transformed[::2, 2],
            s=0.35,
            c=moving_colors[::2] / 255.0,
            alpha=0.72,
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("backend world X (m)")
        axis.set_ylabel("backend world Z (m)")
        axis.grid(alpha=0.15)
    figure.suptitle(
        f"PCF multi-room registration: {moving.name} into {fixed.name} (fixed)",
        fontsize=15,
    )
    figure.savefig(output, dpi=180, facecolor="#11151c")
    plt.close(figure)


def _write_glb(
    moving: Room,
    fixed: Room,
    correction: np.ndarray,
    output: Path,
) -> None:
    import trimesh

    moving_points, moving_colors = _load_surfels(moving)
    fixed_points, fixed_colors = _load_surfels(fixed)
    moving_points = _transform_points(moving_points, correction)
    moving_rgba = np.column_stack(
        (moving_colors, np.full(len(moving_colors), 225, dtype=np.uint8))
    )
    fixed_rgba = np.column_stack(
        (fixed_colors, np.full(len(fixed_colors), 225, dtype=np.uint8))
    )
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.points.PointCloud(fixed_points, colors=fixed_rgba),
        node_name=f"fixed_{fixed.name}",
    )
    scene.add_geometry(
        trimesh.points.PointCloud(moving_points, colors=moving_rgba),
        node_name=f"moving_{moving.name}_registered",
    )
    scene.export(output)


def register(moving: Room, fixed: Room, output_dir: Path, settings: Settings) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    print("[1/5] Retrieving cross-walk image pairs", flush=True)
    retrieval = _retrieve_pairs(moving, fixed, settings)
    if not retrieval:
        raise PCFRoomRegistrationError("no cross-walk image retrieval candidates were found")
    moving_indices = sorted({row[0] for row in retrieval})
    fixed_indices = sorted({row[1] for row in retrieval})
    moving_features = {
        index: _sift_features(moving.views[index], settings) for index in moving_indices
    }
    fixed_features = {
        index: _sift_features(fixed.views[index], settings) for index in fixed_indices
    }
    print(
        f"[2/5] Verifying {len(retrieval)} retrieved pairs with mutual SIFT and PCF depth",
        flush=True,
    )
    accepted_pairs: list[PairEvidence] = []
    for pair_number, (moving_index, fixed_index, score) in enumerate(retrieval, start=1):
        evidence = _pair_evidence(
            moving,
            fixed,
            moving_index,
            fixed_index,
            score,
            moving_features[moving_index],
            fixed_features[fixed_index],
            settings,
        )
        if evidence is not None:
            accepted_pairs.append(evidence)
        if pair_number % 30 == 0 or pair_number == len(retrieval):
            print(
                f"      checked {pair_number}/{len(retrieval)}; "
                f"depth-rigid candidates={len(accepted_pairs)}",
                flush=True,
            )
    print("[3/5] Clustering independent room-transform hypotheses", flush=True)
    cluster = _select_consensus_cluster(accepted_pairs, settings)
    moving_view_count = len({row.moving_view for row in cluster})
    fixed_view_count = len({row.fixed_view for row in cluster})
    if (
        moving_view_count < settings.minimum_distinct_moving_views
        or fixed_view_count < settings.minimum_distinct_fixed_views
    ):
        rejection = {
            "schema": "noesis.pcf.multiroom_cross_session_registration_rejection.v1",
            "generated_at": _utc_now(),
            "status": "rejected",
            "reason": "insufficient_independent_temporal_support",
            "whole_cloud_icp_used": False,
            "retrieval_candidate_pair_count": len(retrieval),
            "depth_verified_pair_count": len(accepted_pairs),
            "consensus_pair_count": len(cluster),
            "consensus_distinct_moving_views": moving_view_count,
            "consensus_distinct_fixed_views": fixed_view_count,
            "depth_verified_pairs": [_pair_report(row) for row in accepted_pairs],
            "consensus_pairs": [_pair_report(row) for row in cluster],
        }
        (output_dir / "rejection_report.json").write_text(
            json.dumps(rejection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if accepted_pairs:
            _write_match_sheet(
                moving,
                fixed,
                accepted_pairs,
                output_dir / "depth_verified_candidates.png",
            )
        raise PCFRoomRegistrationError(
            "cross-walk RGB-D hypotheses lack independent temporal support: "
            f"moving_views={moving_view_count}, fixed_views={fixed_view_count}"
        )
    moving_rows: list[np.ndarray] = []
    fixed_rows: list[np.ndarray] = []
    pair_ids: list[np.ndarray] = []
    for pair_index, row in enumerate(cluster):
        moving_rows.append(row.moving_points_world[row.inlier_mask])
        fixed_rows.append(row.fixed_points_world[row.inlier_mask])
        pair_ids.append(
            np.full(int(np.count_nonzero(row.inlier_mask)), pair_index, dtype=np.int32)
        )
    moving_all, fixed_all, ids_all = _deduplicate_correspondences(
        np.concatenate(moving_rows),
        np.concatenate(fixed_rows),
        np.concatenate(pair_ids),
    )
    unique_pair_ids = sorted(set(int(value) for value in ids_all))
    heldout_pair_ids = set(unique_pair_ids[::5]) if len(unique_pair_ids) >= 5 else set()
    train_mask = ~np.isin(ids_all, list(heldout_pair_ids))
    if int(np.count_nonzero(train_mask)) < settings.minimum_global_3d_inliers:
        train_mask = np.ones(len(ids_all), dtype=bool)
        heldout_pair_ids = set()
    correction, inliers_train = _ransac_yaw_translation(
        moving_all[train_mask],
        fixed_all[train_mask],
        threshold_m=settings.global_ransac_threshold_m,
        random_seed=settings.random_seed,
        iterations=2_500,
    )
    train_indices = np.flatnonzero(train_mask)
    global_inlier_mask = np.zeros(len(moving_all), dtype=bool)
    global_inlier_mask[train_indices[inliers_train]] = True
    train_residual = np.linalg.norm(
        _transform_points(moving_all[global_inlier_mask], correction)
        - fixed_all[global_inlier_mask],
        axis=1,
    )
    heldout_mask = ~train_mask
    heldout_residual = np.linalg.norm(
        _transform_points(moving_all[heldout_mask], correction) - fixed_all[heldout_mask],
        axis=1,
    )
    parameters = _parameters_from_transform(correction)
    passed = bool(
        len(train_residual) >= settings.minimum_global_3d_inliers
        and float(np.median(train_residual)) <= settings.maximum_global_median_m
        and float(np.percentile(train_residual, 80.0)) <= settings.maximum_global_p80_m
        and (
            not len(heldout_residual)
            or float(np.median(heldout_residual)) <= settings.maximum_heldout_median_m
        )
    )
    np.savez_compressed(
        output_dir / "cross_room_correspondences.npz",
        moving_points_backend_world=moving_all.astype(np.float32),
        fixed_points_backend_world=fixed_all.astype(np.float32),
        pair_ids=ids_all,
        train_mask=train_mask,
        global_inlier_mask=global_inlier_mask,
    )
    _write_match_sheet(moving, fixed, cluster, output_dir / "verified_rgbd_matches.png")
    print("[4/5] Evaluating held-out cross-session geometry", flush=True)
    if passed:
        _write_topdown(moving, fixed, correction, output_dir / "before_after_topdown.png")
        _write_glb(moving, fixed, correction, output_dir / "registered_room_surfels.glb")
    print("[5/5] Writing immutable-input registration report", flush=True)
    report: dict[str, Any] = {
        "schema": "noesis.pcf.multiroom_cross_session_registration.v1",
        "generated_at": _utc_now(),
        "status": "passed" if passed else "failed",
        "method": "cross_walk_mutual_sift_depth_backprojection_gravity_constrained_ransac",
        "whole_cloud_icp_used": False,
        "fixed_room": {
            "name": fixed.name,
            "prior_id": fixed.prior_id,
            "scan_id": fixed.scan_id,
            "world_manifest": str(fixed.world_manifest),
            "world_manifest_sha256": _sha256(fixed.world_manifest),
            "pcf_raw_root": str(fixed.raw_root),
            "view_count": len(fixed.views),
        },
        "moving_room": {
            "name": moving.name,
            "prior_id": moving.prior_id,
            "scan_id": moving.scan_id,
            "world_manifest": str(moving.world_manifest),
            "world_manifest_sha256": _sha256(moving.world_manifest),
            "pcf_raw_root": str(moving.raw_root),
            "view_count": len(moving.views),
        },
        "retrieval": {
            "candidate_pair_count": len(retrieval),
            "depth_verified_pair_count": len(accepted_pairs),
            "consensus_pair_count": len(cluster),
            "distinct_moving_views": moving_view_count,
            "distinct_fixed_views": fixed_view_count,
        },
        "consensus_pairs": [_pair_report(row) for row in cluster],
        "solution": {
            "moving_correction_in_backend_world_row_major": correction.tolist(),
            "yaw_correction_deg": math.degrees(float(parameters[0])),
            "translation_correction_m": parameters[1:].tolist(),
            "training_correspondence_count": int(np.count_nonzero(train_mask)),
            "training_inlier_count": int(len(train_residual)),
            "training_inlier_fraction": float(
                len(train_residual) / max(1, int(np.count_nonzero(train_mask)))
            ),
            "training_residual_median_m": float(np.median(train_residual)),
            "training_residual_p80_m": float(np.percentile(train_residual, 80.0)),
            "heldout_correspondence_count": int(len(heldout_residual)),
            "heldout_residual_median_m": (
                float(np.median(heldout_residual)) if len(heldout_residual) else None
            ),
            "heldout_residual_p80_m": (
                float(np.percentile(heldout_residual, 80.0))
                if len(heldout_residual)
                else None
            ),
        },
        "quality_gate": {
            "passed": passed,
            "checks": {
                "global_support": len(train_residual)
                >= settings.minimum_global_3d_inliers,
                "training_median": float(np.median(train_residual))
                <= settings.maximum_global_median_m,
                "training_p80": float(np.percentile(train_residual, 80.0))
                <= settings.maximum_global_p80_m,
                "heldout_median": (
                    not len(heldout_residual)
                    or float(np.median(heldout_residual))
                    <= settings.maximum_heldout_median_m
                ),
            },
        },
        "artifacts": {
            "correspondences": "cross_room_correspondences.npz",
            "verified_matches": "verified_rgbd_matches.png",
            "before_after_topdown": "before_after_topdown.png" if passed else None,
            "registered_surfels": "registered_room_surfels.glb" if passed else None,
        },
    }
    (output_dir / "registration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise PCFRoomRegistrationError(
            "cross-session solution failed its held-out geometric gate; see "
            f"{output_dir / 'registration_report.json'}"
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moving-name", required=True)
    parser.add_argument("--moving-prior-id", required=True)
    parser.add_argument("--moving-scan-id", required=True)
    parser.add_argument("--moving-scan-dir", type=Path, required=True)
    parser.add_argument("--moving-pcf-root", type=Path, required=True)
    parser.add_argument("--moving-world-manifest", type=Path, required=True)
    parser.add_argument("--fixed-name", required=True)
    parser.add_argument("--fixed-prior-id", required=True)
    parser.add_argument("--fixed-scan-id", required=True)
    parser.add_argument("--fixed-scan-dir", type=Path, required=True)
    parser.add_argument("--fixed-pcf-root", type=Path, required=True)
    parser.add_argument("--fixed-world-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    settings = Settings()
    moving = _load_room(
        name=arguments.moving_name,
        prior_id=arguments.moving_prior_id,
        scan_id=arguments.moving_scan_id,
        scan_dir=arguments.moving_scan_dir,
        pcf_root=arguments.moving_pcf_root,
        world_manifest=arguments.moving_world_manifest,
    )
    fixed = _load_room(
        name=arguments.fixed_name,
        prior_id=arguments.fixed_prior_id,
        scan_id=arguments.fixed_scan_id,
        scan_dir=arguments.fixed_scan_dir,
        pcf_root=arguments.fixed_pcf_root,
        world_manifest=arguments.fixed_world_manifest,
    )
    report = register(moving, fixed, arguments.output_dir, settings)
    print(json.dumps(report["solution"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
