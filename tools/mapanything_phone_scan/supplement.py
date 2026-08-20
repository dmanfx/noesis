from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .inference import _write_reconstruction_glb, _write_trajectory_preview


ProgressCallback = Callable[[float, str], None]


class SupplementIntegrationError(RuntimeError):
    """Raised when an added walk cannot be safely registered to its parent scan."""


@dataclass(frozen=True)
class SupplementIntegrationSettings:
    max_total_views: int = 200
    point_budget: int = 600_000
    voxel_size_m: float = 0.04
    min_bridge_views: int = 2
    max_bridge_views: int = 8
    min_visual_inliers: int = 10
    min_visual_inlier_ratio: float = 0.12
    min_reference_pose_consensus: int = 2
    max_reference_pose_distance_m: float = 1.25
    max_reference_pose_orientation_deg: float = 35.0
    overlap_support_fraction: float = 0.30
    max_overlap_frame_gap: int = 2
    max_bridges_per_new_view: int = 4
    max_scale_ratio: float = 1.35
    max_bridge_position_p80_m: float = 0.40
    max_bridge_orientation_p80_deg: float = 16.0
    min_pnp_inliers: int = 10
    max_pnp_position_error_m: float = 0.55
    max_pnp_orientation_error_deg: float = 20.0
    max_pnp_translation_refinement_m: float = 1.50

    @property
    def bridge_view_target(self) -> int:
        available = max(0, int(self.max_total_views) - 2)
        return min(
            int(self.max_bridge_views),
            max(int(self.min_bridge_views), int(self.max_total_views) // 6),
            available,
        )

    @property
    def new_view_limit(self) -> int:
        return max(2, int(self.max_total_views) - self.bridge_view_target)


@dataclass(frozen=True)
class _ReferenceView:
    catalog_id: str
    source_frame: str
    raw_npz: str
    capture_id: str
    source_kind: str
    scale_to_base: float
    rotation_to_base: np.ndarray
    translation_to_base: np.ndarray


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_scan_path(scan_dir: Path, raw: str, *, label: str) -> Path:
    value = Path(str(raw))
    path = value.resolve() if value.is_absolute() else (scan_dir / value).resolve()
    try:
        path.relative_to(scan_dir.resolve())
    except ValueError as exc:
        raise SupplementIntegrationError(f"{label} escapes the saved walk") from exc
    if not path.is_file():
        raise SupplementIntegrationError(f"{label} is missing: {raw}")
    return path


def _scan_relative(scan_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(scan_dir.resolve()).as_posix()
    except ValueError as exc:
        raise SupplementIntegrationError(f"artifact escaped the saved walk: {path}") from exc


def _as_u8(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.dtype == np.uint8:
        return value
    finite = value[np.isfinite(value)]
    scale = 255.0 if finite.size and float(np.max(finite)) <= 1.5 else 1.0
    return np.clip(value * scale, 0.0, 255.0).astype(np.uint8)


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    trace = float(np.trace(np.asarray(rotation, dtype=np.float64)))
    cosine = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _, right_t = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation


def _transform_pose(
    pose: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    source = np.asarray(pose, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation @ source[:3, :3]
    result[:3, 3] = scale * (rotation @ source[:3, 3]) + translation
    return result


def _transform_points(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != 3:
        raise SupplementIntegrationError(
            f"transform points must end in XYZ coordinates, got {values.shape}"
        )
    shape = values.shape
    flat = values.reshape((-1, 3))
    transformed = (scale * (rotation @ flat.T)).T + translation
    return transformed.reshape(shape)


def _load_raw(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = (
                "world_points",
                "depth_z",
                "confidence",
                "mask",
                "camera_pose",
                "intrinsics",
                "model_rgb",
            )
            missing = [name for name in required if name not in payload]
            if missing:
                raise SupplementIntegrationError(
                    f"raw reconstruction view {path.name} is missing {', '.join(missing)}"
                )
            return {name: np.asarray(payload[name]) for name in payload.files}
    except SupplementIntegrationError:
        raise
    except Exception as exc:
        raise SupplementIntegrationError(
            f"raw reconstruction view is unreadable: {path}"
        ) from exc


def _transform_from_result(result: dict[str, Any]) -> tuple[float, np.ndarray, np.ndarray]:
    transform = result.get("append_to_base")
    if not isinstance(transform, dict):
        raise SupplementIntegrationError("an earlier added walk has no base transform")
    try:
        scale = float(transform["scale"])
        rotation = np.asarray(transform["rotation_row_major"], dtype=np.float64)
        translation = np.asarray(transform["translation_m"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise SupplementIntegrationError("an earlier added walk has an invalid base transform") from exc
    if rotation.shape != (3, 3) or translation.shape != (3,) or not math.isfinite(scale):
        raise SupplementIntegrationError("an earlier added walk has an invalid base transform")
    return scale, rotation, translation


def _reference_catalog(scan_dir: Path, state: dict[str, Any]) -> list[_ReferenceView]:
    outputs = state.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get("frames"), list):
        raise SupplementIntegrationError("the original reconstruction has no per-view outputs")
    catalog: list[_ReferenceView] = []
    for row in outputs["frames"]:
        if not isinstance(row, dict) or row.get("fixed_camera_anchor"):
            continue
        source = row.get("source_frame")
        raw = row.get("raw_npz")
        if not isinstance(source, str) or not isinstance(raw, str):
            continue
        _resolve_scan_path(scan_dir, source, label="original source frame")
        _resolve_scan_path(scan_dir, raw, label="original raw view")
        catalog.append(
            _ReferenceView(
                catalog_id=f"base:{int(row.get('index', len(catalog)))}",
                source_frame=source,
                raw_npz=raw,
                capture_id="base",
                source_kind="base",
                scale_to_base=1.0,
                rotation_to_base=np.eye(3, dtype=np.float64),
                translation_to_base=np.zeros(3, dtype=np.float64),
            )
        )

    for supplement in state.get("supplements") or []:
        if not isinstance(supplement, dict) or supplement.get("status") != "complete":
            continue
        result = supplement.get("results")
        if not isinstance(result, dict):
            continue
        scale, rotation, translation = _transform_from_result(result)
        for row in result.get("capture_views") or []:
            if not isinstance(row, dict):
                continue
            source = row.get("source_frame")
            raw = row.get("raw_npz")
            if not isinstance(source, str) or not isinstance(raw, str):
                continue
            _resolve_scan_path(scan_dir, source, label="added source frame")
            _resolve_scan_path(scan_dir, raw, label="added raw view")
            catalog.append(
                _ReferenceView(
                    catalog_id=f"{supplement.get('id')}:{int(row.get('index', 0))}",
                    source_frame=source,
                    raw_npz=raw,
                    capture_id=str(supplement.get("id")),
                    source_kind="supplement",
                    scale_to_base=scale,
                    rotation_to_base=rotation,
                    translation_to_base=translation,
                )
            )
    if len(catalog) < 2:
        raise SupplementIntegrationError(
            "the current reconstruction does not contain enough reusable source views"
        )
    return catalog


def _selection_features(path: Path) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SupplementIntegrationError(f"overlap image is unreadable: {path.name}")
    largest = max(image.shape)
    if largest > 960:
        scale = 960.0 / float(largest)
        image = cv2.resize(
            image,
            (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    sift = cv2.SIFT_create(nfeatures=1_600, contrastThreshold=0.02)
    return sift.detectAndCompute(image, None)


def _visual_pair_score(
    left: tuple[list[cv2.KeyPoint], np.ndarray | None],
    right: tuple[list[cv2.KeyPoint], np.ndarray | None],
) -> tuple[int, float, int]:
    left_keypoints, left_descriptors = left
    right_keypoints, right_descriptors = right
    if left_descriptors is None or right_descriptors is None:
        return 0, 0.0, 0
    if len(left_descriptors) < 8 or len(right_descriptors) < 8:
        return 0, 0.0, 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
    good = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    if len(good) < 6:
        return 0, 0.0, len(good)
    left_xy = np.float32([left_keypoints[match.queryIdx].pt for match in good])
    right_xy = np.float32([right_keypoints[match.trainIdx].pt for match in good])
    _, mask = cv2.findHomography(left_xy, right_xy, cv2.RANSAC, 4.0)
    inliers = int(np.count_nonzero(mask)) if mask is not None else 0
    return inliers, float(inliers / max(1, len(good))), len(good)


def _select_bridges(
    scan_dir: Path,
    catalog: list[_ReferenceView],
    new_rows: list[dict[str, Any]],
    settings: SupplementIntegrationSettings,
    progress: ProgressCallback,
) -> list[dict[str, Any]]:
    # Additional walks are already bounded by ``new_view_limit`` (40 views with
    # the default 48-view inference budget).  Search every prepared view here:
    # sparse temporal probes can miss the short overlap at the start/end of a
    # walk and leave the new camera path connected through only one moment.
    probe_indices = list(range(len(new_rows)))
    new_features: dict[int, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}
    for index in probe_indices:
        path = _resolve_scan_path(
            scan_dir, str(new_rows[index]["frame"]), label="new prepared frame"
        )
        new_features[index] = _selection_features(path)

    reference_features: dict[str, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}
    scored: list[tuple[int, float, int, _ReferenceView, int]] = []
    for reference_index, reference in enumerate(catalog):
        reference_path = _resolve_scan_path(
            scan_dir, reference.source_frame, label="reference source frame"
        )
        features = _selection_features(reference_path)
        reference_features[reference.catalog_id] = features
        for new_index in probe_indices:
            inliers, ratio, match_count = _visual_pair_score(
                features, new_features[new_index]
            )
            if (
                inliers >= settings.min_visual_inliers
                and ratio >= settings.min_visual_inlier_ratio
            ):
                scored.append((inliers, ratio, match_count, reference, new_index))
        if reference_index % 12 == 0:
            progress(
                0.03 + 0.08 * ((reference_index + 1) / len(catalog)),
                "Finding visual overlap with the current reconstruction",
            )

    # A repeated rug, floor, or brick pattern can pass a homography check even
    # when the images look in opposite directions.  Genuine matches for one
    # new frame should point to a compact neighborhood of camera poses in the
    # already reconstructed walk.  Keep only the strongest such neighborhood
    # for each new frame before reasoning about temporal overlap.
    scored_by_new: dict[
        int, list[tuple[int, float, int, _ReferenceView, int]]
    ] = {}
    for row in scored:
        scored_by_new.setdefault(int(row[4]), []).append(row)

    reference_poses: dict[str, np.ndarray] = {}

    def reference_pose(reference: _ReferenceView) -> np.ndarray:
        cached = reference_poses.get(reference.catalog_id)
        if cached is not None:
            return cached
        raw = _load_raw(
            _resolve_scan_path(
                scan_dir, reference.raw_npz, label="bridge reference raw view"
            )
        )
        pose = _transform_pose(
            raw["camera_pose"],
            reference.scale_to_base,
            reference.rotation_to_base,
            reference.translation_to_base,
        )
        reference_poses[reference.catalog_id] = pose
        return pose

    pose_consistent: list[tuple[int, float, int, _ReferenceView, int]] = []
    pose_consensus_count: dict[int, int] = {}
    for new_index, candidates in scored_by_new.items():
        best_cluster: list[tuple[int, float, int, _ReferenceView, int]] = []
        best_key = (-1, -1, -1.0)
        for pivot in candidates:
            pivot_pose = reference_pose(pivot[3])
            cluster: list[tuple[int, float, int, _ReferenceView, int]] = []
            for candidate in candidates:
                candidate_pose = reference_pose(candidate[3])
                position_delta = float(
                    np.linalg.norm(
                        pivot_pose[:3, 3] - candidate_pose[:3, 3]
                    )
                )
                orientation_delta = _rotation_angle_deg(
                    pivot_pose[:3, :3] @ candidate_pose[:3, :3].T
                )
                if (
                    position_delta <= settings.max_reference_pose_distance_m
                    and orientation_delta
                    <= settings.max_reference_pose_orientation_deg
                ):
                    cluster.append(candidate)
            key = (
                sum(row[0] for row in cluster),
                len(cluster),
                sum(row[1] for row in cluster),
            )
            if key > best_key:
                best_key = key
                best_cluster = cluster
        if len(best_cluster) >= settings.min_reference_pose_consensus:
            pose_consistent.extend(best_cluster)
            pose_consensus_count[new_index] = len(best_cluster)

    if not pose_consistent:
        raise SupplementIntegrationError(
            "visual overlap candidates did not agree with any camera neighborhood "
            "in the current reconstruction"
        )

    # Capture overlap is a short contiguous segment (normally the deliberate
    # opening hold), not isolated texture matches sprinkled through the walk.
    # Select the strongest temporal component and exclude later one-off hits.
    support_by_new: dict[int, int] = {}
    for new_index in pose_consensus_count:
        candidates = sorted(
            (row for row in pose_consistent if row[4] == new_index),
            key=lambda row: (row[0], row[1], row[2]),
            reverse=True,
        )
        support_by_new[new_index] = sum(
            row[0] for row in candidates[: settings.max_bridges_per_new_view]
        )
    peak_support = max(support_by_new.values())
    support_floor = max(
        settings.min_visual_inliers * settings.min_reference_pose_consensus,
        int(math.ceil(peak_support * settings.overlap_support_fraction)),
    )
    strong_indices = sorted(
        index for index, support in support_by_new.items() if support >= support_floor
    )
    components: list[list[int]] = []
    for index in strong_indices:
        if (
            not components
            or index - components[-1][-1] > settings.max_overlap_frame_gap
        ):
            components.append([index])
        else:
            components[-1].append(index)
    overlap_component = max(
        components,
        key=lambda component: (
            sum(support_by_new[index] for index in component),
            max(support_by_new[index] for index in component),
            len(component),
        ),
    )
    overlap_start = min(overlap_component)
    overlap_end = max(overlap_component)
    scored = [
        row
        for row in pose_consistent
        if overlap_start <= int(row[4]) <= overlap_end
    ]
    scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    selected: list[dict[str, Any]] = []
    used_catalog: set[str] = set()
    new_use_count: dict[int, int] = {}

    def admit(max_uses_per_new_view: int) -> None:
        for inliers, ratio, match_count, reference, new_index in scored:
            if len(selected) >= settings.bridge_view_target:
                return
            if reference.catalog_id in used_catalog:
                continue
            if new_use_count.get(new_index, 0) >= max_uses_per_new_view:
                continue
            selected.append(
                {
                    "reference": reference,
                    "matched_new_index": int(new_index),
                    "visual_inliers": int(inliers),
                    "visual_inlier_ratio": float(ratio),
                    "visual_match_count": int(match_count),
                    "reference_pose_consensus_count": int(
                        pose_consensus_count[new_index]
                    ),
                    "overlap_frame_start": int(overlap_start),
                    "overlap_frame_end": int(overlap_end),
                }
            )
            used_catalog.add(reference.catalog_id)
            new_use_count[new_index] = new_use_count.get(new_index, 0) + 1

    # Several old views of the same deliberate overlap moment are useful here:
    # they constrain the old neighborhood and give joint inference multiple
    # ways to attach the continuous new path.  Cap each exact new frame so one
    # still image cannot consume the entire bridge budget.
    admit(settings.max_bridges_per_new_view)

    if len(selected) < settings.min_bridge_views:
        best = scored[0][:3] if scored else (0, 0.0, 0)
        raise SupplementIntegrationError(
            "the added video does not have enough visual overlap with the current "
            f"reconstruction: found {len(selected)} bridge views, need "
            f"{settings.min_bridge_views}; best pair had {best[0]} inliers and "
            f"{best[1]:.2f} inlier ratio"
        )
    return selected


def _estimate_bridge_similarity(
    base_poses: np.ndarray,
    append_poses: np.ndarray,
    settings: SupplementIntegrationSettings,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    base = np.asarray(base_poses, dtype=np.float64)
    append = np.asarray(append_poses, dtype=np.float64)
    if base.shape != append.shape or base.ndim != 3 or base.shape[1:] != (4, 4):
        raise SupplementIntegrationError("bridge camera poses are malformed")
    if base.shape[0] < settings.min_bridge_views:
        raise SupplementIntegrationError("too few bridge camera poses were reconstructed")

    relative_rotations = np.stack(
        [base[index, :3, :3] @ append[index, :3, :3].T for index in range(len(base))]
    )
    pair_angles = np.zeros((len(base), len(base)), dtype=np.float64)
    for left in range(len(base)):
        for right in range(left + 1, len(base)):
            angle = _rotation_angle_deg(
                relative_rotations[left] @ relative_rotations[right].T
            )
            pair_angles[left, right] = angle
            pair_angles[right, left] = angle
    medoid = int(np.argmin(np.median(pair_angles, axis=1)))
    initial_errors = pair_angles[medoid]
    rotation_inliers = initial_errors <= max(
        8.0, float(np.median(initial_errors) + 2.5 * np.median(np.abs(initial_errors - np.median(initial_errors))))
    )
    if int(np.count_nonzero(rotation_inliers)) < settings.min_bridge_views:
        rotation_inliers = np.argsort(initial_errors)[: settings.min_bridge_views]
        mask = np.zeros(len(base), dtype=bool)
        mask[rotation_inliers] = True
        rotation_inliers = mask
    rotation = _project_rotation(np.mean(relative_rotations[rotation_inliers], axis=0))

    base_centers = base[:, :3, 3]
    append_centers = append[:, :3, 3]
    scale_ratios: list[float] = []
    for left in range(len(base)):
        for right in range(left + 1, len(base)):
            base_distance = float(np.linalg.norm(base_centers[left] - base_centers[right]))
            append_distance = float(np.linalg.norm(append_centers[left] - append_centers[right]))
            if base_distance >= 0.10 and append_distance >= 0.10:
                scale_ratios.append(base_distance / append_distance)
    scale = float(np.median(scale_ratios)) if scale_ratios else 1.0
    translations = base_centers - scale * (rotation @ append_centers.T).T
    translation = np.median(translations[rotation_inliers], axis=0)

    predicted_centers = scale * (rotation @ append_centers.T).T + translation
    position_errors = np.linalg.norm(predicted_centers - base_centers, axis=1)
    orientation_errors = np.asarray(
        [
            _rotation_angle_deg(
                (rotation @ append[index, :3, :3]) @ base[index, :3, :3].T
            )
            for index in range(len(base))
        ],
        dtype=np.float64,
    )
    position_p80 = float(np.percentile(position_errors, 80.0))
    orientation_p80 = float(np.percentile(orientation_errors, 80.0))
    checks = {
        "bridge_count": len(base) >= settings.min_bridge_views,
        "metric_scale": (1.0 / settings.max_scale_ratio) <= scale <= settings.max_scale_ratio,
        "bridge_position": position_p80 <= settings.max_bridge_position_p80_m,
        "bridge_orientation": orientation_p80 <= settings.max_bridge_orientation_p80_deg,
    }
    metrics = {
        "checks": checks,
        "bridge_count": int(len(base)),
        "scale": scale,
        "position_error_median_m": float(np.median(position_errors)),
        "position_error_p80_m": position_p80,
        "position_error_max_m": float(np.max(position_errors)),
        "orientation_error_median_deg": float(np.median(orientation_errors)),
        "orientation_error_p80_deg": orientation_p80,
        "orientation_error_max_deg": float(np.max(orientation_errors)),
        "scale_pair_count": int(len(scale_ratios)),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise SupplementIntegrationError(
            "duplicate bridge views did not produce a stable append-to-base transform: "
            f"{failed}; scale={scale:.3f}, position_p80={position_p80:.3f}m, "
            f"orientation_p80={orientation_p80:.1f}deg"
        )
    return scale, rotation, translation, metrics


def _raw_sift(image_rgb: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    image = _as_u8(image_rgb)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=4_000, contrastThreshold=0.015)
    return sift.detectAndCompute(gray, None)


def _pnp_refine_translation(
    bridge_rows: list[dict[str, Any]],
    new_raw_rows: list[dict[str, np.ndarray]],
    append_to_base: tuple[float, np.ndarray, np.ndarray],
    settings: SupplementIntegrationSettings,
) -> tuple[np.ndarray, dict[str, Any]]:
    scale, rotation, initial_translation = append_to_base
    validations: list[dict[str, Any]] = []
    translation_candidates: list[tuple[int, np.ndarray]] = []
    for bridge in bridge_rows:
        reference_raw = bridge["reference_raw"]
        new_index = int(bridge["matched_new_index"])
        if new_index < 0 or new_index >= len(new_raw_rows):
            continue
        new_raw = new_raw_rows[new_index]
        ref_keypoints, ref_descriptors = _raw_sift(reference_raw["model_rgb"])
        new_keypoints, new_descriptors = _raw_sift(new_raw["model_rgb"])
        if ref_descriptors is None or new_descriptors is None:
            continue
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        pairs = matcher.knnMatch(ref_descriptors, new_descriptors, k=2)
        good = [first for first, second in pairs if first.distance < 0.72 * second.distance]
        object_points: list[np.ndarray] = []
        image_points: list[tuple[float, float]] = []
        world_map = np.asarray(reference_raw["world_points"], dtype=np.float64)
        mask = np.asarray(reference_raw["mask"], dtype=bool)
        reference = bridge["reference"]
        for match in good:
            x, y = ref_keypoints[match.queryIdx].pt
            column = int(round(x))
            row = int(round(y))
            if not (0 <= row < mask.shape[0] and 0 <= column < mask.shape[1]):
                continue
            if not mask[row, column]:
                continue
            point = world_map[row, column]
            if not np.isfinite(point).all():
                continue
            transformed = _transform_points(
                point[None, :],
                reference.scale_to_base,
                reference.rotation_to_base,
                reference.translation_to_base,
            )[0]
            object_points.append(transformed)
            image_points.append(new_keypoints[match.trainIdx].pt)
        row_result: dict[str, Any] = {
            "reference_catalog_id": reference.catalog_id,
            "new_view_index": new_index,
            "candidate_correspondence_count": int(len(object_points)),
            "accepted": False,
        }
        if len(object_points) < max(8, settings.min_pnp_inliers):
            validations.append(row_result)
            continue
        object_array = np.asarray(object_points, dtype=np.float32)
        image_array = np.asarray(image_points, dtype=np.float32)
        intrinsics = np.asarray(new_raw["intrinsics"], dtype=np.float64)
        solved, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            object_array,
            image_array,
            intrinsics,
            None,
            iterationsCount=300,
            reprojectionError=4.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        inlier_count = int(len(inliers)) if solved and inliers is not None else 0
        row_result["inlier_count"] = inlier_count
        if not solved or inliers is None or inlier_count < settings.min_pnp_inliers:
            validations.append(row_result)
            continue
        inlier_indices = inliers.reshape(-1)
        if hasattr(cv2, "solvePnPRefineLM"):
            rotation_vector, translation_vector = cv2.solvePnPRefineLM(
                object_array[inlier_indices],
                image_array[inlier_indices],
                intrinsics,
                None,
                rotation_vector,
                translation_vector,
            )
        world_to_camera_rotation, _ = cv2.Rodrigues(rotation_vector)
        camera_to_base = np.eye(4, dtype=np.float64)
        camera_to_base[:3, :3] = world_to_camera_rotation.T
        camera_to_base[:3, 3] = -world_to_camera_rotation.T @ translation_vector.reshape(3)
        expected_orientation = rotation @ new_raw["camera_pose"][:3, :3]
        orientation_error = _rotation_angle_deg(
            camera_to_base[:3, :3] @ expected_orientation.T
        )
        projected, _ = cv2.projectPoints(
            object_array[inlier_indices],
            rotation_vector,
            translation_vector,
            intrinsics,
            None,
        )
        reprojection = np.linalg.norm(
            projected.reshape(-1, 2) - image_array[inlier_indices], axis=1
        )
        translation_candidate = camera_to_base[:3, 3] - scale * (
            rotation @ new_raw["camera_pose"][:3, 3]
        )
        row_result.update(
            {
                "orientation_error_deg": orientation_error,
                "reprojection_error_median_px": float(np.median(reprojection)),
                "pnp_camera_center_base_m": camera_to_base[:3, 3].tolist(),
                "translation_candidate_m": translation_candidate.tolist(),
            }
        )
        if orientation_error <= settings.max_pnp_orientation_error_deg:
            translation_candidates.append((len(validations), translation_candidate))
        validations.append(row_result)

    required = max(1, min(2, len(bridge_rows) // 3))
    if len(translation_candidates) < required:
        best = max(
            (int(row.get("inlier_count", 0)) for row in validations), default=0
        )
        raise SupplementIntegrationError(
            "the independent RGB/3D check could not solve enough orientation-consistent "
            f"overlap anchors: found {len(translation_candidates)}, need {required}; "
            f"best had {best} inliers"
        )

    candidate_values = np.stack([row[1] for row in translation_candidates])
    pair_distances = np.linalg.norm(
        candidate_values[:, None, :] - candidate_values[None, :, :], axis=2
    )
    medoid_index = int(np.argmin(np.median(pair_distances, axis=1)))
    consensus_mask = (
        pair_distances[medoid_index] <= settings.max_pnp_position_error_m
    )
    if int(np.count_nonzero(consensus_mask)) < required:
        raise SupplementIntegrationError(
            "the independent RGB/3D overlap anchors did not agree on the added "
            f"path position: found {int(np.count_nonzero(consensus_mask))} "
            f"consistent anchors, need {required}"
        )
    refined_translation = np.median(candidate_values[consensus_mask], axis=0)
    translation_refinement = float(
        np.linalg.norm(refined_translation - initial_translation)
    )

    accepted = 0
    accepted_position_errors: list[float] = []
    for validation_index, translation_candidate in translation_candidates:
        row_result = validations[validation_index]
        position_error = float(
            np.linalg.norm(translation_candidate - refined_translation)
        )
        passed = position_error <= settings.max_pnp_position_error_m
        row_result["position_error_m"] = position_error
        row_result["accepted"] = bool(passed)
        if passed:
            accepted += 1
            accepted_position_errors.append(position_error)

    position_p80 = (
        float(np.percentile(accepted_position_errors, 80.0))
        if accepted_position_errors
        else float("inf")
    )
    checks = {
        "anchor_count": accepted >= required,
        "translation_refinement": (
            translation_refinement <= settings.max_pnp_translation_refinement_m
        ),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "accepted_anchor_count": int(accepted),
        "required_anchor_count": int(required),
        "translation_initial_m": initial_translation.tolist(),
        "translation_refined_m": refined_translation.tolist(),
        "translation_refinement_m": translation_refinement,
        "translation_consensus_p80_m": position_p80,
        "anchors": validations,
    }
    if not result["passed"]:
        failed = ", ".join(
            name for name, passed in checks.items() if not passed
        )
        raise SupplementIntegrationError(
            "the independent RGB/3D overlap could not safely place the added "
            f"camera path: {failed}; accepted={accepted}/{required}, "
            f"translation_refinement={translation_refinement:.3f}m, "
            f"position_p80={position_p80:.3f}m"
        )
    return refined_translation, result


def _sample_raw_view(
    raw: dict[str, np.ndarray],
    *,
    budget: int,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_map = np.asarray(raw["world_points"], dtype=np.float64)
    depth = np.asarray(raw["depth_z"], dtype=np.float64)
    confidence = np.asarray(raw["confidence"], dtype=np.float64)
    mask = np.asarray(raw["mask"], dtype=bool)
    image = _as_u8(raw["model_rgb"])
    valid = (
        mask
        & np.isfinite(depth)
        & (depth > 0.0)
        & np.isfinite(confidence)
        & np.isfinite(points_map).all(axis=2)
    )
    if not np.any(valid):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0,), dtype=np.float32),
        )
    values = confidence[valid]
    threshold = float(np.percentile(values, 35.0))
    selected = np.flatnonzero((valid & (confidence >= threshold)).reshape(-1))
    if selected.size > budget:
        positions = np.linspace(0, selected.size - 1, budget, dtype=np.int64)
        selected = selected[positions]
    points = points_map.reshape(-1, 3)[selected]
    points = _transform_points(points, scale, rotation, translation)
    colors = image.reshape(-1, 3)[selected]
    selected_confidence = confidence.reshape(-1)[selected]
    low, high = np.percentile(values, (10.0, 90.0))
    weights = 0.25 + 0.75 * np.clip(
        (selected_confidence - low) / max(float(high - low), 1e-8), 0.0, 1.0
    )
    finite = np.isfinite(points).all(axis=1) & (np.max(np.abs(points), axis=1) < 1_000.0)
    return (
        points[finite].astype(np.float32),
        colors[finite].astype(np.uint8),
        weights[finite].astype(np.float32),
    )


def _write_comparison_glb(
    path: Path,
    points: np.ndarray,
    base_mask: np.ndarray,
    supplement_mask: np.ndarray,
) -> None:
    import trimesh

    scene = trimesh.Scene()
    if np.any(base_mask):
        blue = np.tile(np.asarray([74, 150, 255, 225], dtype=np.uint8), (int(np.count_nonzero(base_mask)), 1))
        scene.add_geometry(
            trimesh.points.PointCloud(points[base_mask], colors=blue),
            geom_name="original_and_parent_evidence",
        )
    if np.any(supplement_mask):
        orange = np.tile(np.asarray([255, 142, 55, 235], dtype=np.uint8), (int(np.count_nonzero(supplement_mask)), 1))
        scene.add_geometry(
            trimesh.points.PointCloud(points[supplement_mask], colors=orange),
            geom_name="added_video_evidence",
        )
    scene.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    )
    scene.export(str(path))


def _fuse_sources(
    scan_dir: Path,
    sources: list[dict[str, Any]],
    output_dir: Path,
    settings: SupplementIntegrationSettings,
    current_camera_poses: np.ndarray,
    progress: ProgressCallback,
) -> dict[str, Any]:
    per_view_budget = max(800, int(settings.point_budget) // max(1, len(sources)))
    point_rows: list[np.ndarray] = []
    color_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    source_rows: list[np.ndarray] = []
    view_rows: list[np.ndarray] = []
    for index, source in enumerate(sources):
        raw = _load_raw(
            _resolve_scan_path(scan_dir, source["raw_npz"], label="fusion raw view")
        )
        points, colors, weights = _sample_raw_view(
            raw,
            budget=per_view_budget,
            scale=float(source["scale"]),
            rotation=np.asarray(source["rotation"], dtype=np.float64),
            translation=np.asarray(source["translation"], dtype=np.float64),
        )
        if points.size:
            if source["source_kind"] == "base":
                weights = weights * 1.35
            point_rows.append(points)
            color_rows.append(colors)
            weight_rows.append(weights)
            source_rows.append(
                np.full(len(points), 0 if source["source_kind"] == "base" else 1, dtype=np.uint8)
            )
            view_rows.append(np.full(len(points), index, dtype=np.int32))
        if index % 12 == 0:
            progress(
                0.78 + 0.08 * ((index + 1) / len(sources)),
                "Fusing the original and added room evidence",
            )
    if not point_rows:
        raise SupplementIntegrationError("no valid points were available for the merged revision")
    points = np.concatenate(point_rows)
    colors = np.concatenate(color_rows)
    weights = np.concatenate(weight_rows).astype(np.float64)
    source_ids = np.concatenate(source_rows)
    view_ids = np.concatenate(view_rows)
    keys = np.floor(points / float(settings.voxel_size_m)).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    voxel_count = len(unique_keys)
    weight_sum = np.bincount(inverse, weights=weights, minlength=voxel_count)
    fused_points = np.stack(
        [
            np.bincount(inverse, weights=weights * points[:, axis], minlength=voxel_count)
            / np.maximum(weight_sum, 1e-9)
            for axis in range(3)
        ],
        axis=1,
    )
    fused_colors = np.stack(
        [
            np.bincount(inverse, weights=weights * colors[:, axis], minlength=voxel_count)
            / np.maximum(weight_sum, 1e-9)
            for axis in range(3)
        ],
        axis=1,
    )
    base_samples = np.bincount(
        inverse, weights=(source_ids == 0), minlength=voxel_count
    ).astype(np.int32)
    supplement_samples = np.bincount(
        inverse, weights=(source_ids == 1), minlength=voxel_count
    ).astype(np.int32)
    unique_view_voxels = np.unique(np.column_stack((inverse, view_ids)), axis=0)
    base_view_ids = np.asarray(
        [index for index, row in enumerate(sources) if row["source_kind"] == "base"],
        dtype=np.int32,
    )
    base_view_pairs = unique_view_voxels[
        np.isin(unique_view_voxels[:, 1], base_view_ids)
    ]
    supplement_view_pairs = unique_view_voxels[
        ~np.isin(unique_view_voxels[:, 1], base_view_ids)
    ]
    base_support = np.bincount(
        base_view_pairs[:, 0] if len(base_view_pairs) else np.empty(0, dtype=np.int64),
        minlength=voxel_count,
    ).astype(np.int16)
    supplement_support = np.bincount(
        supplement_view_pairs[:, 0] if len(supplement_view_pairs) else np.empty(0, dtype=np.int64),
        minlength=voxel_count,
    ).astype(np.int16)
    keep = (base_samples > 0) | (supplement_support >= 2)
    fused_points = fused_points[keep].astype(np.float32)
    fused_colors = np.clip(fused_colors[keep], 0, 255).astype(np.uint8)
    base_samples = base_samples[keep]
    supplement_samples = supplement_samples[keep]
    base_support = base_support[keep]
    supplement_support = supplement_support[keep]
    if len(fused_points) == 0:
        raise SupplementIntegrationError("voxel fusion rejected every reconstructed point")

    merged_glb = output_dir / "merged_reconstruction.glb"
    added_glb = output_dir / "added_evidence.glb"
    comparison_glb = output_dir / "source_comparison.glb"
    trajectory_preview = output_dir / "added_camera_trajectory_topdown.png"
    revision_npz = output_dir / "merged_surfels.npz"
    _write_reconstruction_glb(
        merged_glb,
        fused_points,
        fused_colors,
        np.asarray(current_camera_poses, dtype=np.float64)[:, :3, 3],
    )
    added_mask = supplement_samples > 0
    _write_reconstruction_glb(
        added_glb,
        fused_points[added_mask],
        fused_colors[added_mask],
        np.asarray(current_camera_poses, dtype=np.float64)[:, :3, 3],
    )
    _write_comparison_glb(
        comparison_glb,
        fused_points,
        base_samples > 0,
        added_mask,
    )
    _write_trajectory_preview(
        trajectory_preview,
        fused_points,
        np.asarray(current_camera_poses, dtype=np.float64)[:, :3, 3],
    )
    np.savez_compressed(
        revision_npz,
        points=fused_points,
        colors=fused_colors,
        base_sample_count=base_samples,
        supplement_sample_count=supplement_samples,
        base_view_support=base_support,
        supplement_view_support=supplement_support,
        voxel_size_m=np.asarray([settings.voxel_size_m], dtype=np.float32),
    )
    return {
        "point_count": int(len(fused_points)),
        "base_supported_voxel_count": int(np.count_nonzero(base_samples > 0)),
        "supplement_supported_voxel_count": int(np.count_nonzero(added_mask)),
        "new_only_voxel_count": int(np.count_nonzero((base_samples == 0) & added_mask)),
        "shared_voxel_count": int(np.count_nonzero((base_samples > 0) & added_mask)),
        "voxel_size_m": float(settings.voxel_size_m),
        "minimum_new_only_view_support": 2,
        "points": fused_points,
        "colors": fused_colors,
        "artifact_names": {
            "merged_reconstruction_glb": merged_glb.name,
            "added_evidence_glb": added_glb.name,
            "source_comparison_glb": comparison_glb.name,
            "trajectory_preview": trajectory_preview.name,
            "merged_surfels_npz": revision_npz.name,
        },
    }


def _noesis_transform(scan_dir: Path, state: dict[str, Any]) -> np.ndarray | None:
    alignment = state.get("alignment")
    if not isinstance(alignment, dict) or alignment.get("status") != "complete":
        return None
    results = alignment.get("results")
    artifacts = results.get("artifacts") if isinstance(results, dict) else None
    transform_path = artifacts.get("transform") if isinstance(artifacts, dict) else None
    if not isinstance(transform_path, str):
        return None
    payload_path = _resolve_scan_path(
        scan_dir, transform_path, label="phone-to-Noesis transform"
    )
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        matrix = np.asarray(payload["world_from_mapanything_row_major"], dtype=np.float64)
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        raise SupplementIntegrationError("the saved phone-to-Noesis transform is invalid") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise SupplementIntegrationError("the saved phone-to-Noesis transform is invalid")
    return matrix


def _apply_rigid_matrix(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return (
        np.asarray(matrix[:3, :3], dtype=np.float64) @ np.asarray(points).T
    ).T + np.asarray(matrix[:3, 3], dtype=np.float64)


def _rebase_joint_result(
    result: dict[str, Any],
    prefix: str,
    joint_dir: Path,
) -> dict[str, Any]:
    rebased = json.loads(json.dumps(result))

    def rewrite(raw: str) -> str:
        path = str(raw)
        return f"{prefix}/{path.removeprefix('outputs/')}"

    artifacts = rebased.get("artifacts")
    if isinstance(artifacts, dict):
        for key, value in list(artifacts.items()):
            if isinstance(value, str):
                artifacts[key] = rewrite(value)
    for row in rebased.get("frames") or []:
        if not isinstance(row, dict):
            continue
        for key in (
            "model_rgb",
            "depth_preview",
            "confidence_preview",
            "mask_preview",
            "raw_npz",
        ):
            if isinstance(row.get(key), str):
                row[key] = rewrite(row[key])
    for row in rebased.get("files") or []:
        if isinstance(row, dict) and isinstance(row.get("path"), str):
            row["path"] = rewrite(row["path"])
    manifest = joint_dir / "scan_outputs_manifest.json"
    if manifest.is_file():
        manifest.write_text(json.dumps(rebased, indent=2) + "\n", encoding="utf-8")
        for row in rebased.get("files") or []:
            if isinstance(row, dict) and row.get("path", "").endswith(
                "/scan_outputs_manifest.json"
            ):
                row["size_bytes"] = int(manifest.stat().st_size)
        manifest.write_text(json.dumps(rebased, indent=2) + "\n", encoding="utf-8")
    return rebased


def materialize_noesis_revision(
    scan_dir: Path,
    result: dict[str, Any],
    alignment_results: dict[str, Any],
) -> dict[str, Any]:
    updated = json.loads(json.dumps(result))
    artifacts = updated.get("artifacts")
    alignment_artifacts = alignment_results.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(alignment_artifacts, dict):
        raise SupplementIntegrationError("revision or alignment artifacts are missing")
    surfel_path = _resolve_scan_path(
        scan_dir,
        str(artifacts.get("merged_surfels_npz") or ""),
        label="merged revision surfels",
    )
    transform_path = _resolve_scan_path(
        scan_dir,
        str(alignment_artifacts.get("transform") or ""),
        label="phone-to-Noesis transform",
    )
    try:
        transform_payload = json.loads(transform_path.read_text(encoding="utf-8"))
        matrix = np.asarray(
            transform_payload["world_from_mapanything_row_major"], dtype=np.float64
        )
        with np.load(surfel_path, allow_pickle=False) as surfels:
            points = np.asarray(surfels["points"], dtype=np.float32)
            colors = np.asarray(surfels["colors"], dtype=np.uint8)
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        raise SupplementIntegrationError(
            "the merged revision could not be transformed into the Noesis world"
        ) from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise SupplementIntegrationError("the phone-to-Noesis transform is invalid")
    poses = np.asarray(
        [row["camera_pose_base"] for row in updated.get("capture_views") or []],
        dtype=np.float64,
    )
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise SupplementIntegrationError("the merged revision has no valid camera poses")
    points_noesis = _apply_rigid_matrix(points, matrix).astype(np.float32)
    poses[:, :3, :3] = matrix[:3, :3][None, :, :] @ poses[:, :3, :3]
    poses[:, :3, 3] = _apply_rigid_matrix(poses[:, :3, 3], matrix)
    revision_dir = surfel_path.parent
    glb_path = revision_dir / "merged_noesis_reconstruction.glb"
    _write_reconstruction_glb(
        glb_path, points_noesis, colors, poses[:, :3, 3]
    )
    relative = _scan_relative(scan_dir, glb_path)
    artifacts["noesis_aligned_glb"] = relative
    updated["coordinate_frame"] = "base_phone_frame_with_noesis_aligned_derivative"
    files = updated.get("files")
    if isinstance(files, list):
        files[:] = [
            row
            for row in files
            if not isinstance(row, dict) or row.get("path") != relative
        ]
        files.append({"path": relative, "size_bytes": int(glb_path.stat().st_size)})
    for key in ("report", "manifest"):
        path_value = artifacts.get(key)
        if isinstance(path_value, str):
            path = _resolve_scan_path(scan_dir, path_value, label=f"revision {key}")
            path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    return updated


def run_supplement_integration(
    scan_dir: Path,
    supplement_dir: Path,
    output_dir: Path,
    state: dict[str, Any],
    supplement: dict[str, Any],
    provider: str,
    inference_runner: Callable[..., dict[str, Any]],
    provider_settings: Any,
    settings: SupplementIntegrationSettings,
    progress: ProgressCallback,
) -> dict[str, Any]:
    prepared = supplement.get("prepared")
    new_rows = prepared.get("frames") if isinstance(prepared, dict) else None
    if not isinstance(new_rows, list) or len(new_rows) < 2:
        raise SupplementIntegrationError("the added video has no prepared multi-view frame set")
    if len(new_rows) > settings.new_view_limit:
        raise SupplementIntegrationError(
            f"the added video has {len(new_rows)} frames, above its {settings.new_view_limit}-view allowance"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    progress(0.01, "Checking overlap with the current reconstruction")
    catalog = _reference_catalog(scan_dir, state)
    selected = _select_bridges(scan_dir, catalog, new_rows, settings, progress)

    bridge_dir = supplement_dir / "bridge_inputs"
    if bridge_dir.exists():
        shutil.rmtree(bridge_dir)
    bridge_dir.mkdir()
    bridge_records: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for index, selected_row in enumerate(selected):
        reference: _ReferenceView = selected_row["reference"]
        source_path = _resolve_scan_path(
            scan_dir, reference.source_frame, label="bridge source frame"
        )
        suffix = source_path.suffix.lower() if source_path.suffix else ".jpg"
        copied = bridge_dir / f"bridge_{index:04d}{suffix}"
        shutil.copy2(source_path, copied)
        copied_relative = _scan_relative(scan_dir, copied)
        combined_rows.append(
            {
                "index": index,
                "timestamp_s": None,
                "frame": copied_relative,
                "bridge_view": True,
                "bridge_catalog_id": reference.catalog_id,
            }
        )
        bridge_records.append(
            {
                **selected_row,
                "copied_frame": copied_relative,
                "reference_raw": _load_raw(
                    _resolve_scan_path(
                        scan_dir, reference.raw_npz, label="bridge reference raw view"
                    )
                ),
            }
        )
    for index, row in enumerate(new_rows):
        combined_rows.append(
            {
                **row,
                "index": len(selected) + index,
                "supplement_view": True,
                "supplement_source_index": index,
            }
        )
    if len(combined_rows) > settings.max_total_views:
        raise SupplementIntegrationError("bridge and added views exceed the inference view budget")
    combined_prepared = {
        "frame_count": len(combined_rows),
        "frames": combined_rows,
        "bridge_view_count": len(selected),
        "supplement_view_count": len(new_rows),
    }
    joint_dir = output_dir / "joint_inference"
    joint_dir.mkdir()
    clean_provider_settings = replace(provider_settings, anchor_image=None)
    progress(0.12, f"Starting joint {provider.upper()} bridge reconstruction")
    joint_result = inference_runner(
        scan_dir,
        joint_dir,
        combined_prepared,
        clean_provider_settings,
        lambda fraction, message: progress(0.12 + 0.52 * float(fraction), message),
    )
    joint_result = _rebase_joint_result(
        joint_result,
        f"supplements/{supplement['id']}/revision/joint_inference",
        joint_dir,
    )
    joint_frames = joint_result.get("frames")
    if not isinstance(joint_frames, list) or len(joint_frames) != len(combined_rows):
        raise SupplementIntegrationError("joint inference returned an unexpected view set")

    reference_poses: list[np.ndarray] = []
    append_bridge_poses: list[np.ndarray] = []
    for index, bridge in enumerate(bridge_records):
        reference: _ReferenceView = bridge["reference"]
        reference_pose = _transform_pose(
            bridge["reference_raw"]["camera_pose"],
            reference.scale_to_base,
            reference.rotation_to_base,
            reference.translation_to_base,
        )
        append_raw = _load_raw(joint_dir / "raw" / f"view_{index:04d}.npz")
        reference_poses.append(reference_pose)
        append_bridge_poses.append(np.asarray(append_raw["camera_pose"], dtype=np.float64))
    progress(0.66, "Registering duplicate bridge cameras to the current room frame")
    scale, rotation, translation, bridge_metrics = _estimate_bridge_similarity(
        np.stack(reference_poses),
        np.stack(append_bridge_poses),
        settings,
    )

    new_raw_rows = [
        _load_raw(joint_dir / "raw" / f"view_{len(selected) + index:04d}.npz")
        for index in range(len(new_rows))
    ]
    progress(0.71, "Refining the added camera path from RGB and reconstructed 3D")
    translation, pnp_validation = _pnp_refine_translation(
        bridge_records,
        new_raw_rows,
        (scale, rotation, translation),
        settings,
    )

    prefix = f"supplements/{supplement['id']}/revision"
    capture_views: list[dict[str, Any]] = []
    current_camera_poses: list[np.ndarray] = []
    for index, (row, raw) in enumerate(zip(new_rows, new_raw_rows, strict=True)):
        pose_base = _transform_pose(raw["camera_pose"], scale, rotation, translation)
        current_camera_poses.append(pose_base)
        capture_views.append(
            {
                "index": int(index),
                "source_frame": str(row["frame"]),
                "raw_npz": f"{prefix}/joint_inference/raw/view_{len(selected) + index:04d}.npz",
                "joint_view_index": int(len(selected) + index),
                "timestamp_s": row.get("timestamp_s"),
                "camera_pose_base": pose_base.tolist(),
            }
        )

    fusion_sources: list[dict[str, Any]] = []
    for reference in _reference_catalog(scan_dir, state):
        fusion_sources.append(
            {
                "raw_npz": reference.raw_npz,
                "source_kind": reference.source_kind,
                "scale": reference.scale_to_base,
                "rotation": reference.rotation_to_base,
                "translation": reference.translation_to_base,
            }
        )
    for row in capture_views:
        fusion_sources.append(
            {
                "raw_npz": _scan_relative(
                    scan_dir,
                    joint_dir / "raw" / f"view_{int(row['joint_view_index']):04d}.npz",
                ),
                "source_kind": "supplement",
                "scale": scale,
                "rotation": rotation,
                "translation": translation,
            }
        )
    fusion = _fuse_sources(
        scan_dir,
        fusion_sources,
        output_dir,
        settings,
        np.stack(current_camera_poses),
        progress,
    )

    trajectory_json = output_dir / "added_camera_trajectory.json"
    trajectory_json.write_text(
        json.dumps(
            {
                "schema": "noesis.phone_scan.supplement.trajectory.v1",
                "coordinate_frame": "original_phone_reconstruction_base",
                "pose_convention": "opencv_cam2world_x_right_y_down_z_forward",
                "camera_poses": np.stack(current_camera_poses).tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    transform_matrix = np.eye(4, dtype=np.float64)
    transform_matrix[:3, :3] = scale * rotation
    transform_matrix[:3, 3] = translation
    transform_path = output_dir / "append_to_base_transform.json"
    transform_payload = {
        "schema": "noesis.phone_scan.supplement.transform.v1",
        "generated_at": _utc_now(),
        "source_coordinate_frame": "added_joint_reconstruction",
        "target_coordinate_frame": "original_phone_reconstruction_base",
        "scale": scale,
        "rotation_row_major": rotation.tolist(),
        "translation_m": translation.tolist(),
        "base_from_append_row_major": transform_matrix.tolist(),
        "bridge_quality": bridge_metrics,
        "independent_pnp_validation": pnp_validation,
    }
    transform_path.write_text(
        json.dumps(transform_payload, indent=2) + "\n", encoding="utf-8"
    )

    artifact_names = dict(fusion.pop("artifact_names"))
    artifact_names.update(
        {
            "trajectory_json": trajectory_json.name,
            "append_to_base_transform": transform_path.name,
            "joint_reconstruction_glb": "joint_inference/reconstruction_points.glb",
            "joint_trajectory_preview": "joint_inference/camera_trajectory_topdown.png",
        }
    )
    noesis_matrix = _noesis_transform(scan_dir, state)
    coordinate_frame = "original_phone_reconstruction_base"
    if noesis_matrix is not None:
        noesis_points = _apply_rigid_matrix(fusion["points"], noesis_matrix)
        noesis_poses = np.stack(current_camera_poses).copy()
        noesis_poses[:, :3, :3] = (
            noesis_matrix[:3, :3][None, :, :] @ noesis_poses[:, :3, :3]
        )
        noesis_poses[:, :3, 3] = _apply_rigid_matrix(
            noesis_poses[:, :3, 3], noesis_matrix
        )
        noesis_glb = output_dir / "merged_noesis_reconstruction.glb"
        _write_reconstruction_glb(
            noesis_glb,
            noesis_points.astype(np.float32),
            fusion["colors"],
            noesis_poses[:, :3, 3],
        )
        artifact_names["noesis_aligned_glb"] = noesis_glb.name
        coordinate_frame = "base_phone_frame_with_noesis_aligned_derivative"

    del fusion["points"]
    del fusion["colors"]
    parent = state.get("active_revision")
    parent_revision_id = (
        str(parent.get("revision_id"))
        if isinstance(parent, dict) and parent.get("revision_id")
        else f"{state['id']}:base"
    )
    bridge_public = [
        {
            "catalog_id": row["reference"].catalog_id,
            "capture_id": row["reference"].capture_id,
            "source_frame": row["reference"].source_frame,
            "matched_new_index": row["matched_new_index"],
            "visual_inliers": row["visual_inliers"],
            "visual_inlier_ratio": row["visual_inlier_ratio"],
        }
        for row in bridge_records
    ]
    artifacts = {key: f"{prefix}/{value}" for key, value in artifact_names.items()}
    revision_id = f"{state['id']}:add:{supplement['id']}"
    result: dict[str, Any] = {
        "schema": "noesis.phone_scan.supplement.revision.v1",
        "revision_id": revision_id,
        "parent_revision_id": parent_revision_id,
        "generated_at": _utc_now(),
        "provider": provider,
        "coordinate_frame": coordinate_frame,
        "base_scan_id": state["id"],
        "supplement_id": supplement["id"],
        "input_lineage": {
            "base_manifest_sha256": _sha256(
                _resolve_scan_path(
                    scan_dir,
                    str(state["outputs"]["artifacts"]["manifest"]),
                    label="base output manifest",
                )
            ),
            "added_video_sha256": str(
                (prepared.get("video") or {}).get("sha256")
                or (supplement.get("video") or {}).get("sha256")
                or ""
            ),
        },
        "bridge": {
            "selected_count": len(bridge_public),
            "selected_views": bridge_public,
            "transform_quality": bridge_metrics,
        },
        "independent_pnp_validation": pnp_validation,
        "append_to_base": {
            "scale": scale,
            "rotation_row_major": rotation.tolist(),
            "translation_m": translation.tolist(),
            "base_from_append_row_major": transform_matrix.tolist(),
        },
        "fusion": fusion,
        "capture_views": capture_views,
        "artifacts": artifacts,
        "joint_inference": {
            "provider": provider,
            "view_count": int(joint_result.get("view_count", len(combined_rows))),
            "bridge_view_count": len(selected),
            "added_view_count": len(new_rows),
            "artifacts": joint_result.get("artifacts", {}),
        },
    }
    report_path = output_dir / "integration_report.json"
    manifest_path = output_dir / "revision_manifest.json"
    result["artifacts"]["report"] = f"{prefix}/{report_path.name}"
    result["artifacts"]["manifest"] = f"{prefix}/{manifest_path.name}"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["files"] = [
        {
            "path": f"{prefix}/{path.relative_to(output_dir).as_posix()}",
            "size_bytes": int(path.stat().st_size),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["files"].append(
        {
            "path": f"{prefix}/{manifest_path.name}",
            "size_bytes": int(manifest_path.stat().st_size),
        }
    )
    progress(1.0, "Added video registered and merged into a new saved revision")
    return result


def public_active_revision(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "revision_id": result["revision_id"],
        "parent_revision_id": result["parent_revision_id"],
        "supplement_id": result["supplement_id"],
        "generated_at": result["generated_at"],
        "provider": result["provider"],
        "coordinate_frame": result["coordinate_frame"],
        "fusion": result["fusion"],
        "artifacts": result["artifacts"],
    }


__all__ = [
    "SupplementIntegrationError",
    "SupplementIntegrationSettings",
    "_estimate_bridge_similarity",
    "materialize_noesis_revision",
    "public_active_revision",
    "run_supplement_integration",
]
