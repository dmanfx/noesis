#!/usr/bin/env python3
"""Transfer an accepted DA3 world alignment to a denser run of the same video."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mapanything_phone_scan.supplement import (  # noqa: E402
    SupplementIntegrationSettings,
    _estimate_bridge_similarity,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _load_pose(raw_root: Path, index: int) -> np.ndarray:
    with np.load(raw_root / f"view_{index:04d}.npz") as row:
        pose = np.asarray(row["camera_pose"], dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"view {index} has a malformed camera pose")
    return pose


def _match_timestamps(
    baseline_rows: list[dict[str, Any]],
    adaptive_rows: list[dict[str, Any]],
    max_delta_s: float,
) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    matches: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    used_adaptive: set[int] = set()
    for baseline in baseline_rows:
        baseline_time = float(baseline["timestamp_s"])
        adaptive = min(
            adaptive_rows,
            key=lambda row: abs(float(row["timestamp_s"]) - baseline_time),
        )
        adaptive_index = int(adaptive["index"])
        delta = abs(float(adaptive["timestamp_s"]) - baseline_time)
        if delta <= max_delta_s and adaptive_index not in used_adaptive:
            used_adaptive.add(adaptive_index)
            matches.append((baseline, adaptive, delta))
    return matches


def _sample_cloud(
    raw_root: Path,
    view_count: int,
    samples_per_view: int,
    seed: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for index in range(view_count):
        with np.load(raw_root / f"view_{index:04d}.npz") as raw:
            points = np.asarray(raw["world_points"], dtype=np.float64).reshape(-1, 3)
            confidence = np.asarray(raw["confidence"], dtype=np.float64).reshape(-1)
            mask = np.asarray(raw["mask"], dtype=bool).reshape(-1)
        valid = mask & np.isfinite(points).all(axis=1) & np.isfinite(confidence)
        indices = np.flatnonzero(valid)
        if indices.size > samples_per_view:
            indices = rng.choice(indices, samples_per_view, replace=False)
        rows.append(points[indices])
    if not rows:
        raise ValueError(f"{raw_root} has no valid DA3 points")
    return np.concatenate(rows, axis=0)


def _voxel_first(points: np.ndarray, voxel_m: float) -> np.ndarray:
    keys = np.floor(np.asarray(points) / voxel_m).astype(np.int32)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def _proper_rigid(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _cloud_refine(
    baseline: np.ndarray,
    adaptive: np.ndarray,
    initial: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    baseline_tree = cKDTree(baseline)
    transformed = (
        initial[:3, :3] @ adaptive.T
    ).T + initial[:3, 3]
    result = initial.copy()
    history: list[dict[str, Any]] = []
    for iteration in range(iterations):
        distances, indices = baseline_tree.query(transformed, k=1, workers=-1)
        threshold = max(0.10, min(0.45, float(np.percentile(distances, 55.0))))
        keep = distances <= threshold
        source = transformed[keep]
        target = baseline[indices[keep]]
        if source.shape[0] < 2_000:
            raise ValueError("too few cloud correspondences survived robust trimming")
        if source.shape[0] > 120_000:
            selected = np.random.default_rng(100 + iteration).choice(
                source.shape[0], 120_000, replace=False
            )
            source = source[selected]
            target = target[selected]
        rotation, translation = _proper_rigid(source, target)
        transformed = (rotation @ transformed.T).T + translation
        increment = np.eye(4, dtype=np.float64)
        increment[:3, :3] = rotation
        increment[:3, 3] = translation
        result = increment @ result
        angle = _rotation_angle_deg(rotation)
        distance = float(np.linalg.norm(translation))
        history.append(
            {
                "iteration": iteration,
                "correspondence_count": int(source.shape[0]),
                "trim_distance_m": threshold,
                "increment_rotation_deg": angle,
                "increment_translation_m": distance,
            }
        )
        if angle < 0.005 and distance < 0.0005:
            break

    adaptive_to_baseline = baseline_tree.query(transformed, k=1, workers=-1)[0]
    baseline_to_adaptive = cKDTree(transformed).query(baseline, k=1, workers=-1)[0]
    metrics = {
        "iterations": history,
        "adaptive_to_baseline": {
            "median_m": float(np.median(adaptive_to_baseline)),
            "p80_m": float(np.percentile(adaptive_to_baseline, 80.0)),
            "overlap_0_20m": float(np.mean(adaptive_to_baseline <= 0.20)),
            "overlap_0_30m": float(np.mean(adaptive_to_baseline <= 0.30)),
        },
        "baseline_to_adaptive": {
            "median_m": float(np.median(baseline_to_adaptive)),
            "p80_m": float(np.percentile(baseline_to_adaptive, 80.0)),
            "overlap_0_20m": float(np.mean(baseline_to_adaptive <= 0.20)),
            "overlap_0_30m": float(np.mean(baseline_to_adaptive <= 0.30)),
        },
    }
    return result, metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-scan", type=Path, required=True)
    parser.add_argument("--adaptive-scan", type=Path, required=True)
    parser.add_argument("--baseline-world-transform", type=Path, required=True)
    parser.add_argument(
        "--adaptive-world-transform-seed",
        type=Path,
        help=(
            "Optional independently accepted adaptive static alignment used only "
            "to seed the fixed-scale cloud refinement."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-timestamp-delta-s", type=float, default=0.35)
    parser.add_argument("--minimum-matches", type=int, default=24)
    parser.add_argument("--voxel-m", type=float, default=0.07)
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--max-pose-position-p80-m", type=float, default=1.25)
    parser.add_argument("--max-pose-orientation-p80-deg", type=float, default=35.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    baseline_scan = args.baseline_scan.resolve()
    adaptive_scan = args.adaptive_scan.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    baseline_manifest_path = baseline_scan / "prepared_frames_manifest.json"
    adaptive_manifest_path = adaptive_scan / "prepared_frames_manifest.json"
    baseline_outputs_path = baseline_scan / "outputs/scan_outputs_manifest.json"
    adaptive_outputs_path = adaptive_scan / "outputs/scan_outputs_manifest.json"
    baseline_manifest = _load_json(baseline_manifest_path)
    adaptive_manifest = _load_json(adaptive_manifest_path)
    baseline_outputs = _load_json(baseline_outputs_path)
    adaptive_outputs = _load_json(adaptive_outputs_path)
    baseline_rows = baseline_manifest.get("frames")
    adaptive_rows = adaptive_manifest.get("frames")
    if not isinstance(baseline_rows, list) or not isinstance(adaptive_rows, list):
        raise ValueError("prepared frame manifests are malformed")
    matches = _match_timestamps(
        baseline_rows, adaptive_rows, args.max_timestamp_delta_s
    )
    if len(matches) < args.minimum_matches:
        raise ValueError(
            f"only {len(matches)} same-video timestamp bridges passed the "
            f"{args.max_timestamp_delta_s:.2f}s gate"
        )

    baseline_raw = baseline_scan / "outputs/raw"
    adaptive_raw = adaptive_scan / "outputs/raw"
    baseline_poses = np.stack(
        [_load_pose(baseline_raw, int(row[0]["index"])) for row in matches]
    )
    adaptive_poses = np.stack(
        [_load_pose(adaptive_raw, int(row[1]["index"])) for row in matches]
    )
    settings = SupplementIntegrationSettings(
        max_total_views=512,
        min_bridge_views=args.minimum_matches,
        max_bridge_views=len(matches),
        max_scale_ratio=1.20,
        max_bridge_position_p80_m=args.max_pose_position_p80_m,
        max_bridge_orientation_p80_deg=args.max_pose_orientation_p80_deg,
    )
    scale_diagnostic, rotation, _, pose_metrics = _estimate_bridge_similarity(
        baseline_poses, adaptive_poses, settings
    )
    translation = np.median(
        baseline_poses[:, :3, 3]
        - (rotation @ adaptive_poses[:, :3, 3].T).T,
        axis=0,
    )
    initial = np.eye(4, dtype=np.float64)
    initial[:3, :3] = rotation
    initial[:3, 3] = translation

    baseline_transform_path = args.baseline_world_transform.resolve()
    baseline_transform = _load_json(baseline_transform_path)
    matrix_row = baseline_transform.get("world_from_mapanything_row_major")
    if matrix_row is None:
        matrix_row = baseline_transform.get("world_from_da3_row_major")
    world_from_baseline = np.asarray(matrix_row, dtype=np.float64)
    if world_from_baseline.shape != (4, 4):
        raise ValueError("baseline world transform is malformed")
    seed_transform_path: Path | None = None
    seed_report_path: Path | None = None
    seed_method = "same_video_timestamp_pose_bridge"
    if args.adaptive_world_transform_seed is not None:
        seed_transform_path = args.adaptive_world_transform_seed.resolve()
        seed_payload = _load_json(seed_transform_path)
        seed_matrix_row = seed_payload.get("world_from_mapanything_row_major")
        if seed_matrix_row is None:
            seed_matrix_row = seed_payload.get("world_from_da3_row_major")
        world_from_adaptive_seed = np.asarray(seed_matrix_row, dtype=np.float64)
        if world_from_adaptive_seed.shape != (4, 4):
            raise ValueError("adaptive world-transform seed is malformed")
        seed_report_path = seed_transform_path.parent / "alignment_report.json"
        seed_report = _load_json(seed_report_path)
        if not bool(seed_report.get("quality_gate", {}).get("passed")):
            raise ValueError("adaptive world-transform seed did not pass its alignment gate")
        initial = np.linalg.inv(world_from_baseline) @ world_from_adaptive_seed
        seed_method = "independently_accepted_dual_static_alignment"

    baseline_cloud = _voxel_first(
        _sample_cloud(baseline_raw, len(baseline_rows), 2_500, 17),
        args.voxel_m,
    )
    adaptive_cloud = _voxel_first(
        _sample_cloud(adaptive_raw, len(adaptive_rows), 800, 23),
        args.voxel_m,
    )
    baseline_from_adaptive, cloud_metrics = _cloud_refine(
        baseline_cloud, adaptive_cloud, initial, args.iterations
    )
    correction = baseline_from_adaptive @ np.linalg.inv(initial)
    correction_rotation_deg = _rotation_angle_deg(correction[:3, :3])
    correction_translation_m = float(np.linalg.norm(correction[:3, 3]))

    world_from_adaptive = world_from_baseline @ baseline_from_adaptive
    inverse = np.linalg.inv(world_from_adaptive)
    round_trip = float(
        np.max(np.abs(world_from_adaptive @ inverse - np.eye(4)))
    )
    checks = {
        "timestamp_bridge_count": len(matches) >= args.minimum_matches,
        "metric_scale_diagnostic": 0.85 <= scale_diagnostic <= 1.15,
        "bounded_icp_correction_rotation": correction_rotation_deg <= 5.0,
        "bounded_icp_correction_translation": correction_translation_m <= 0.30,
        "adaptive_cloud_p80": cloud_metrics["adaptive_to_baseline"]["p80_m"] <= 0.15,
        "baseline_cloud_p80": cloud_metrics["baseline_to_adaptive"]["p80_m"] <= 0.15,
        "symmetric_overlap_0_30m": min(
            cloud_metrics["adaptive_to_baseline"]["overlap_0_30m"],
            cloud_metrics["baseline_to_adaptive"]["overlap_0_30m"],
        )
        >= 0.90,
        "proper_rigid_transform": abs(np.linalg.det(world_from_adaptive[:3, :3]) - 1.0)
        <= 1e-5,
        "round_trip": round_trip <= 1e-8,
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise ValueError(f"transferred DA3 alignment did not clear quality gate: {failed}")

    generated_at = datetime.now(timezone.utc).isoformat()
    transform = {
        "schema": "noesis.mapanything.phone_scan.world_alignment.v1",
        "generated_at": generated_at,
        "source_coordinate_frame": adaptive_outputs.get("coordinate_frame"),
        "target_coordinate_frame": baseline_transform.get(
            "target_coordinate_frame", "backend_world_m_stream_points"
        ),
        "scale": 1.0,
        "world_from_mapanything_row_major": world_from_adaptive.tolist(),
        "world_from_mapanything_col_major": world_from_adaptive.reshape(
            -1, order="F"
        ).tolist(),
        "mapanything_from_world_row_major": inverse.tolist(),
        "round_trip_max_abs_error": round_trip,
        "transfer": {
            "method": f"{seed_method}_then_fixed_scale_symmetric_cloud_icp",
            "metric_scale_diagnostic": scale_diagnostic,
            "baseline_from_adaptive_row_major": baseline_from_adaptive.tolist(),
        },
    }
    transform_path = output_dir / "phone_ma_to_noesis_world.json"
    transform_path.write_text(json.dumps(transform, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": "noesis.phone_scan.transferred_da3_alignment.v1",
        "generated_at": generated_at,
        "status": "passed",
        "admission": "comparison_candidate_derived_from_accepted_same_video_alignment",
        "method": transform["transfer"]["method"],
        "quality_gate": {"passed": True, "checks": checks},
        "timestamp_bridges": {
            "count": len(matches),
            "maximum_delta_s": max(row[2] for row in matches),
            "median_delta_s": float(np.median([row[2] for row in matches])),
        },
        "pose_bridge": pose_metrics,
        "metric_scale_diagnostic": scale_diagnostic,
        "fixed_scale_cloud_refinement": {
            "correction_rotation_deg": correction_rotation_deg,
            "correction_translation_m": correction_translation_m,
            **cloud_metrics,
        },
        "transform": transform,
        "inputs": {
            "baseline_prepared_manifest_sha256": _sha256(baseline_manifest_path),
            "adaptive_prepared_manifest_sha256": _sha256(adaptive_manifest_path),
            "baseline_outputs_manifest_sha256": _sha256(baseline_outputs_path),
            "adaptive_outputs_manifest_sha256": _sha256(adaptive_outputs_path),
            "baseline_world_transform_sha256": _sha256(baseline_transform_path),
            "adaptive_world_transform_seed_sha256": (
                _sha256(seed_transform_path) if seed_transform_path is not None else None
            ),
            "adaptive_world_transform_seed_report_sha256": (
                _sha256(seed_report_path) if seed_report_path is not None else None
            ),
            "baseline_provider": baseline_outputs.get("provider"),
            "adaptive_provider": adaptive_outputs.get("provider"),
        },
    }
    report_path = output_dir / "alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
