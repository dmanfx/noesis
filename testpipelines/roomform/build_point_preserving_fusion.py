#!/usr/bin/env python3
"""Rebuild a dense point cloud from cached consensus-fusion frame evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mapanything_phone_scan.inference import (  # noqa: E402
    _write_reconstruction_glb,
)


def _weighted_columns(
    inverse: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    weight_sum: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.bincount(
                inverse,
                weights=values[:, axis] * weights,
                minlength=len(weight_sum),
            )
            / np.maximum(weight_sum, 1e-8)
            for axis in range(values.shape[1])
        ]
    )


def _load_evidence(
    raw_dir: Path,
    minimum_quality: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted(raw_dir.glob("view_*.npz"))
    if len(paths) < 2:
        raise RuntimeError(f"consensus frame evidence is missing from {raw_dir}")

    point_rows: list[np.ndarray] = []
    color_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    view_rows: list[np.ndarray] = []
    cameras: list[np.ndarray] = []
    for view_index, path in enumerate(paths):
        with np.load(path) as row:
            points = np.asarray(row["world_points"], dtype=np.float32)
            colors = np.asarray(row["model_rgb"], dtype=np.uint8)
            weights = np.asarray(row["confidence"], dtype=np.float32)
            selected = np.asarray(row["mask"], dtype=bool)
            selected &= weights >= minimum_quality
            selected &= np.asarray(row["depth_z"], dtype=np.float32) <= 12.0
            selected &= np.isfinite(points).all(axis=-1)
            point_rows.append(points[selected])
            color_rows.append(colors[selected])
            weight_rows.append(weights[selected])
            view_rows.append(
                np.full(np.count_nonzero(selected), view_index, dtype=np.int16)
            )
            cameras.append(
                np.asarray(row["camera_pose"], dtype=np.float32)[:3, 3]
            )

    return (
        np.concatenate(point_rows),
        np.concatenate(color_rows),
        np.concatenate(weight_rows),
        np.concatenate(view_rows),
        np.stack(cameras),
    )


def build(
    raw_dir: Path,
    output_dir: Path,
    *,
    voxel_m: float = 0.02,
    minimum_quality: float = 0.35,
    minimum_samples: int = 2,
    minimum_weight: float = 0.80,
) -> dict[str, Any]:
    if voxel_m <= 0:
        raise ValueError("voxel size must be positive")
    if minimum_samples < 1:
        raise ValueError("minimum samples must be at least one")

    print("Loading full-resolution cached fusion evidence", flush=True)
    points, colors, weights, view_ids, cameras = _load_evidence(
        raw_dir, minimum_quality
    )
    print(f"Aggregating {len(points):,} accepted samples at {voxel_m:.3f} m", flush=True)

    voxel_keys = np.floor(points / voxel_m).astype(np.int32)
    _, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    voxel_count = int(np.max(inverse)) + 1
    sample_counts = np.bincount(inverse, minlength=voxel_count)
    weight_sum = np.bincount(
        inverse, weights=weights, minlength=voxel_count
    )
    fused_points = _weighted_columns(inverse, points, weights, weight_sum)
    fused_colors = _weighted_columns(
        inverse, colors.astype(np.float32), weights, weight_sum
    )

    view_stride = int(np.max(view_ids)) + 1
    view_voxel_pairs = inverse.astype(np.int64) * view_stride + view_ids
    unique_pairs = np.unique(view_voxel_pairs)
    view_counts = np.bincount(
        unique_pairs // view_stride, minlength=voxel_count
    )

    keep = (sample_counts >= minimum_samples) & (weight_sum >= minimum_weight)
    fused_points = fused_points[keep].astype(np.float32)
    fused_colors = np.clip(fused_colors[keep], 0, 255).astype(np.uint8)
    fused_weights = weight_sum[keep].astype(np.float32)
    fused_sample_counts = sample_counts[keep].astype(np.uint32)
    fused_view_counts = view_counts[keep].astype(np.uint16)

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "point_preserving_fusion_2cm.npz"
    np.savez_compressed(
        npz_path,
        points=fused_points,
        colors=fused_colors,
        weights=fused_weights,
        sample_counts=fused_sample_counts,
        view_counts=fused_view_counts,
    )
    glb_path = output_dir / "point_preserving_fusion_2cm.glb"
    _write_reconstruction_glb(glb_path, fused_points, fused_colors, cameras)
    ply_path = output_dir / "point_preserving_fusion_2cm.ply"
    trimesh.PointCloud(fused_points, colors=fused_colors).export(ply_path)

    report: dict[str, Any] = {
        "schema": "noesis.phone_walk.point_preserving_fusion.v1",
        "source_raw_evidence": str(raw_dir),
        "method": "full_resolution_consistency_gated_weighted_voxel_fusion",
        "voxel_m": voxel_m,
        "minimum_quality": minimum_quality,
        "minimum_samples": minimum_samples,
        "minimum_accumulated_weight": minimum_weight,
        "view_count": int(len(cameras)),
        "accepted_input_sample_count": int(len(points)),
        "candidate_voxel_count": voxel_count,
        "output_point_count": int(len(fused_points)),
        "multi_view_point_count": int(np.count_nonzero(fused_view_counts >= 2)),
        "multi_view_point_fraction": float(np.mean(fused_view_counts >= 2)),
        "bounds_m": {
            "min": fused_points.min(axis=0).tolist(),
            "max": fused_points.max(axis=0).tolist(),
        },
        "artifacts": {
            "glb": glb_path.name,
            "ply": ply_path.name,
            "npz": npz_path.name,
        },
    }
    report_path = output_dir / "point_preserving_fusion_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--voxel-m", type=float, default=0.02)
    parser.add_argument("--minimum-quality", type=float, default=0.35)
    parser.add_argument("--minimum-samples", type=int, default=2)
    parser.add_argument("--minimum-weight", type=float, default=0.80)
    args = parser.parse_args()
    build(
        args.raw_dir.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        voxel_m=args.voxel_m,
        minimum_quality=args.minimum_quality,
        minimum_samples=args.minimum_samples,
        minimum_weight=args.minimum_weight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
