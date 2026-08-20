from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .processing import _analyze_candidates, _visual_edge
from .supplement import _load_raw, _rotation_angle_deg


def _distribution(values: list[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"min": None, "p50": None, "p80": None, "p95": None, "max": None}
    return {
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50.0)),
        "p80": float(np.percentile(array, 80.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest is malformed: {path}")
    return payload


def _frame_path(scan_dir: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else scan_dir / path


def _sequence_metrics(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    scan_dir = manifest_path.parent
    rows = manifest.get("frames")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError(f"prepared manifest has fewer than two views: {manifest_path}")
    paths = [_frame_path(scan_dir, str(row["frame"])) for row in rows]
    candidates = _analyze_candidates(paths, 1.0, 640, lambda *_: None)
    edges = [
        _visual_edge(candidates[index - 1], candidates[index])
        for index in range(1, len(candidates))
    ]
    timestamps = np.asarray([float(row["timestamp_s"]) for row in rows])
    intervals = np.diff(timestamps)
    return {
        "view_count": len(rows),
        "walk_span_s": float(timestamps[-1] - timestamps[0]),
        "interval_s": _distribution(intervals),
        "quality_score": _distribution(
            [candidate.quality_score for candidate in candidates]
        ),
        "feature_count": _distribution(
            [float(candidate.feature_count) for candidate in candidates]
        ),
        "adjacent_visual_inliers": _distribution(
            [float(edge["inlier_count"]) for edge in edges]
        ),
        "adjacent_visual_displacement_norm": _distribution(
            [float(edge["median_displacement_norm"]) for edge in edges]
        ),
        "adjacent_connectivity_pass_fraction": float(
            np.mean([bool(edge["passes_connectivity"]) for edge in edges])
        ),
        "weak_adjacent_pair_count": int(
            sum(not bool(edge["passes_connectivity"]) for edge in edges)
        ),
    }


def _raw_paths(output_manifest_path: Path, manifest: dict[str, Any]) -> list[Path]:
    scan_dir = output_manifest_path.parent.parent
    paths = []
    for row in manifest.get("frames") or []:
        raw = row.get("raw_npz")
        if not isinstance(raw, str):
            raise ValueError(f"output frame is missing raw_npz: {output_manifest_path}")
        paths.append(_frame_path(scan_dir, raw))
    return paths


def _trajectory_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = manifest.get("frames") or []
    poses = np.asarray([row["camera_pose"] for row in rows], dtype=np.float64)
    timestamps = np.asarray(
        [float(row.get("timestamp_s") or index) for index, row in enumerate(rows)],
        dtype=np.float64,
    )
    positions = poses[:, :3, 3]
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    intervals = np.maximum(np.diff(timestamps), 1e-6)
    speeds = steps / intervals
    rotations = np.asarray(
        [
            _rotation_angle_deg(
                poses[index - 1, :3, :3].T @ poses[index, :3, :3]
            )
            for index in range(1, len(poses))
        ]
    )
    return {
        "path_length_m": float(np.sum(steps)),
        "start_to_end_m": float(np.linalg.norm(positions[-1] - positions[0])),
        "step_m": _distribution(steps),
        "speed_mps": _distribution(speeds),
        "orientation_step_deg": _distribution(rotations),
        "position_jump_over_1m_count": int(np.count_nonzero(steps > 1.0)),
        "speed_over_2mps_count": int(np.count_nonzero(speeds > 2.0)),
    }


def _repeatability_metrics(raw_paths: list[Path], maximum_pairs: int = 64) -> dict[str, Any]:
    pair_indices = np.linspace(
        0, len(raw_paths) - 2, min(maximum_pairs, len(raw_paths) - 1), dtype=np.int64
    )
    pair_indices = np.unique(pair_indices)
    detector = cv2.ORB_create(nfeatures=1400, fastThreshold=10)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    residual_rows: list[np.ndarray] = []
    accepted_matches: list[int] = []
    for left_index in pair_indices:
        left = _load_raw(raw_paths[int(left_index)])
        right = _load_raw(raw_paths[int(left_index) + 1])
        left_image = np.asarray(left["model_rgb"])
        right_image = np.asarray(right["model_rgb"])
        if left_image.dtype != np.uint8:
            left_image = np.clip(
                left_image * 255.0 if float(np.nanmax(left_image)) <= 1.5 else left_image,
                0,
                255,
            ).astype(np.uint8)
        if right_image.dtype != np.uint8:
            right_image = np.clip(
                right_image * 255.0 if float(np.nanmax(right_image)) <= 1.5 else right_image,
                0,
                255,
            ).astype(np.uint8)
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
        left_keypoints, left_descriptors = detector.detectAndCompute(left_gray, None)
        right_keypoints, right_descriptors = detector.detectAndCompute(right_gray, None)
        if left_descriptors is None or right_descriptors is None:
            accepted_matches.append(0)
            continue
        matches = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
        good = [first for first, second in matches if first.distance < 0.75 * second.distance]
        values: list[float] = []
        left_points = np.asarray(left["world_points"])
        right_points = np.asarray(right["world_points"])
        left_mask = np.asarray(left["mask"], dtype=bool)
        right_mask = np.asarray(right["mask"], dtype=bool)
        for match in good:
            lx, ly = left_keypoints[match.queryIdx].pt
            rx, ry = right_keypoints[match.trainIdx].pt
            left_x = min(left_points.shape[1] - 1, max(0, int(round(lx))))
            left_y = min(left_points.shape[0] - 1, max(0, int(round(ly))))
            right_x = min(right_points.shape[1] - 1, max(0, int(round(rx))))
            right_y = min(right_points.shape[0] - 1, max(0, int(round(ry))))
            if not left_mask[left_y, left_x] or not right_mask[right_y, right_x]:
                continue
            left_point = left_points[left_y, left_x]
            right_point = right_points[right_y, right_x]
            if np.isfinite(left_point).all() and np.isfinite(right_point).all():
                values.append(float(np.linalg.norm(left_point - right_point)))
        accepted_matches.append(len(values))
        if values:
            residual_rows.append(np.asarray(values, dtype=np.float64))
    residuals = np.concatenate(residual_rows) if residual_rows else np.empty(0)
    return {
        "evaluated_adjacent_pair_count": int(len(pair_indices)),
        "pairs_with_3d_matches": int(len(residual_rows)),
        "accepted_3d_matches": _distribution(accepted_matches),
        "matched_world_residual_m": _distribution(residuals),
        "matched_world_overlap_0_10m": (
            float(np.mean(residuals <= 0.10)) if residuals.size else None
        ),
        "matched_world_overlap_0_20m": (
            float(np.mean(residuals <= 0.20)) if residuals.size else None
        ),
    }


def _sample_cloud(raw_paths: list[Path], budget: int = 220_000) -> np.ndarray:
    per_view = max(500, budget // len(raw_paths))
    rows: list[np.ndarray] = []
    for path in raw_paths:
        raw = _load_raw(path)
        points = np.asarray(raw["world_points"], dtype=np.float64)
        depth = np.asarray(raw["depth_z"], dtype=np.float64)
        confidence = np.asarray(raw["confidence"], dtype=np.float64)
        valid = (
            np.asarray(raw["mask"], dtype=bool)
            & np.isfinite(depth)
            & (depth > 0.0)
            & np.isfinite(confidence)
            & np.isfinite(points).all(axis=2)
        )
        if not np.any(valid):
            continue
        threshold = float(np.percentile(confidence[valid], 35.0))
        indices = np.flatnonzero((valid & (confidence >= threshold)).reshape(-1))
        positions = np.linspace(
            0, indices.size - 1, min(per_view, indices.size), dtype=np.int64
        )
        rows.append(points.reshape(-1, 3)[indices[positions]])
    if not rows:
        raise ValueError("reconstruction contains no sampleable world points")
    result = np.concatenate(rows)
    return result[
        np.isfinite(result).all(axis=1) & (np.max(np.abs(result), axis=1) < 100.0)
    ]


def _umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = target_zero.T @ source_zero / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(left @ right_t) < 0.0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right_t
    variance = float(np.mean(np.sum(source_zero**2, axis=1)))
    scale = float(np.sum(singular * sign) / max(variance, 1e-12))
    translation = target_center - scale * (rotation @ source_center)
    return scale, rotation, translation


def _agreement_metrics(
    baseline_manifest: dict[str, Any],
    adaptive_manifest: dict[str, Any],
    baseline_cloud: np.ndarray,
    adaptive_cloud: np.ndarray,
) -> dict[str, Any]:
    baseline_rows = baseline_manifest.get("frames") or []
    adaptive_rows = adaptive_manifest.get("frames") or []
    baseline_times = np.asarray([float(row.get("timestamp_s") or 0.0) for row in baseline_rows])
    adaptive_times = np.asarray([float(row.get("timestamp_s") or 0.0) for row in adaptive_rows])
    baseline_poses = np.asarray([row["camera_pose"] for row in baseline_rows], dtype=np.float64)
    adaptive_poses = np.asarray([row["camera_pose"] for row in adaptive_rows], dtype=np.float64)
    nearest = np.asarray(
        [int(np.argmin(np.abs(adaptive_times - timestamp))) for timestamp in baseline_times]
    )
    source = adaptive_poses[nearest, :3, 3]
    target = baseline_poses[:, :3, 3]
    keep = np.ones(len(source), dtype=bool)
    for _ in range(4):
        scale, rotation, translation = _umeyama(source[keep], target[keep])
        transformed = scale * (rotation @ source.T).T + translation
        residuals = np.linalg.norm(transformed - target, axis=1)
        threshold = float(np.percentile(residuals, 82.0))
        keep = residuals <= threshold
    transformed = scale * (rotation @ source.T).T + translation
    position_residuals = np.linalg.norm(transformed - target, axis=1)
    orientation_residuals = np.asarray(
        [
            _rotation_angle_deg(
                baseline_poses[index, :3, :3]
                @ (rotation @ adaptive_poses[adaptive_index, :3, :3]).T
            )
            for index, adaptive_index in enumerate(nearest)
        ]
    )
    aligned_adaptive_cloud = (
        scale * (rotation @ adaptive_cloud.T).T + translation
    )
    baseline_tree = cKDTree(baseline_cloud)
    adaptive_tree = cKDTree(aligned_adaptive_cloud)
    adaptive_to_baseline = baseline_tree.query(aligned_adaptive_cloud, k=1)[0]
    baseline_to_adaptive = adaptive_tree.query(baseline_cloud, k=1)[0]
    voxel_size = 0.10
    baseline_voxels = {
        tuple(row) for row in np.floor(baseline_cloud / voxel_size).astype(np.int64)
    }
    adaptive_voxels = {
        tuple(row)
        for row in np.floor(aligned_adaptive_cloud / voxel_size).astype(np.int64)
    }
    shared = baseline_voxels & adaptive_voxels
    return {
        "authority": "agreement_with_existing_48_view_result_not_survey_ground_truth",
        "adaptive_to_baseline": {
            "scale": scale,
            "rotation_row_major": rotation.tolist(),
            "translation": translation.tolist(),
        },
        "nearest_timestamp_delta_s": _distribution(
            np.abs(adaptive_times[nearest] - baseline_times)
        ),
        "camera_position_residual_m": _distribution(position_residuals),
        "camera_orientation_residual_deg": _distribution(orientation_residuals),
        "baseline_to_adaptive_cloud_distance_m": _distribution(baseline_to_adaptive),
        "adaptive_to_baseline_cloud_distance_m": _distribution(adaptive_to_baseline),
        "baseline_point_overlap_0_20m": float(np.mean(baseline_to_adaptive <= 0.20)),
        "adaptive_point_overlap_0_20m": float(np.mean(adaptive_to_baseline <= 0.20)),
        "voxel_0_10m": {
            "baseline_count": len(baseline_voxels),
            "adaptive_count": len(adaptive_voxels),
            "shared_count": len(shared),
            "baseline_covered_fraction": float(
                len(shared) / max(1, len(baseline_voxels))
            ),
            "adaptive_new_fraction": float(
                len(adaptive_voxels - baseline_voxels)
                / max(1, len(adaptive_voxels))
            ),
        },
    }


def _reconstruction_metrics(manifest_path: Path) -> tuple[dict[str, Any], np.ndarray]:
    manifest = _load_manifest(manifest_path)
    raw_paths = _raw_paths(manifest_path, manifest)
    frames = manifest.get("frames") or []
    cloud = _sample_cloud(raw_paths)
    return {
        "view_count": int(manifest.get("view_count") or len(frames)),
        "review_point_count": int(manifest.get("review_point_count") or 0),
        "valid_fraction": _distribution(
            [float((row.get("depth") or {}).get("valid_fraction")) for row in frames]
        ),
        "confidence_p50": _distribution(
            [float((row.get("confidence") or {}).get("p50")) for row in frames]
        ),
        "trajectory": _trajectory_metrics(manifest),
        "adjacent_3d_repeatability": _repeatability_metrics(raw_paths),
        "sampled_cloud_point_count": int(len(cloud)),
        "runtime": manifest.get("runtime"),
    }, cloud


def _markdown(report: dict[str, Any]) -> str:
    baseline_sequence = report["frame_selection"]["baseline"]
    adaptive_sequence = report["frame_selection"]["adaptive"]
    baseline_reconstruction = report["reconstruction"]["baseline"]
    adaptive_reconstruction = report["reconstruction"]["adaptive"]
    agreement = report["agreement_proxy"]
    return "\n".join(
        [
            "# Adaptive phone-walk comparison",
            "",
            "The existing 48-view reconstruction is the comparison baseline, not survey ground truth.",
            "",
            "| Metric | Existing | Adaptive |",
            "|---|---:|---:|",
            f"| Selected views | {baseline_sequence['view_count']} | {adaptive_sequence['view_count']} |",
            f"| Median frame interval | {baseline_sequence['interval_s']['p50']:.3f}s | {adaptive_sequence['interval_s']['p50']:.3f}s |",
            f"| Maximum frame interval | {baseline_sequence['interval_s']['max']:.3f}s | {adaptive_sequence['interval_s']['max']:.3f}s |",
            f"| Connected adjacent pairs | {baseline_sequence['adjacent_connectivity_pass_fraction']:.1%} | {adaptive_sequence['adjacent_connectivity_pass_fraction']:.1%} |",
            f"| Median adjacent visual inliers | {baseline_sequence['adjacent_visual_inliers']['p50']:.1f} | {adaptive_sequence['adjacent_visual_inliers']['p50']:.1f} |",
            f"| Median valid depth fraction | {baseline_reconstruction['valid_fraction']['p50']:.1%} | {adaptive_reconstruction['valid_fraction']['p50']:.1%} |",
            f"| Median matched-3D residual | {baseline_reconstruction['adjacent_3d_repeatability']['matched_world_residual_m']['p50']:.3f}m | {adaptive_reconstruction['adjacent_3d_repeatability']['matched_world_residual_m']['p50']:.3f}m |",
            f"| Camera path length | {baseline_reconstruction['trajectory']['path_length_m']:.2f}m | {adaptive_reconstruction['trajectory']['path_length_m']:.2f}m |",
            "",
            "## Baseline-agreement proxy",
            "",
            f"- Existing points recovered within 20 cm: {agreement['baseline_point_overlap_0_20m']:.1%}",
            f"- Adaptive points within 20 cm of existing geometry: {agreement['adaptive_point_overlap_0_20m']:.1%}",
            f"- Existing 10 cm voxels represented by adaptive output: {agreement['voxel_0_10m']['baseline_covered_fraction']:.1%}",
            f"- Adaptive 10 cm voxels not present in the sparse baseline: {agreement['voxel_0_10m']['adaptive_new_fraction']:.1%}",
            "",
        ]
    )


def evaluate(
    baseline_prepared: Path,
    adaptive_prepared: Path,
    baseline_outputs: Path,
    adaptive_outputs: Path,
    output_json: Path,
) -> dict[str, Any]:
    baseline_output_manifest = _load_manifest(baseline_outputs)
    adaptive_output_manifest = _load_manifest(adaptive_outputs)
    baseline_metrics, baseline_cloud = _reconstruction_metrics(baseline_outputs)
    adaptive_metrics, adaptive_cloud = _reconstruction_metrics(adaptive_outputs)
    report = {
        "schema": "noesis.mapanything.phone_scan.adaptive_comparison.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "baseline_prepared": str(baseline_prepared.resolve()),
            "adaptive_prepared": str(adaptive_prepared.resolve()),
            "baseline_outputs": str(baseline_outputs.resolve()),
            "adaptive_outputs": str(adaptive_outputs.resolve()),
        },
        "frame_selection": {
            "baseline": _sequence_metrics(baseline_prepared),
            "adaptive": _sequence_metrics(adaptive_prepared),
        },
        "reconstruction": {
            "baseline": baseline_metrics,
            "adaptive": adaptive_metrics,
        },
        "agreement_proxy": _agreement_metrics(
            baseline_output_manifest,
            adaptive_output_manifest,
            baseline_cloud,
            adaptive_cloud,
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare adaptive phone-walk selection and reconstruction with an existing result."
    )
    parser.add_argument("baseline_prepared", type=Path)
    parser.add_argument("adaptive_prepared", type=Path)
    parser.add_argument("baseline_outputs", type=Path)
    parser.add_argument("adaptive_outputs", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    report = evaluate(
        args.baseline_prepared,
        args.adaptive_prepared,
        args.baseline_outputs,
        args.adaptive_outputs,
        args.output_json,
    )
    print(_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
