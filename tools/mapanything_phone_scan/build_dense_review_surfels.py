#!/usr/bin/env python3
"""Export all accepted cached consensus pixels through canonical surfel fusion.

The raw source, alignment, model outputs and acceptance masks are immutable.
This derives denser review geometry without rerunning or replacing either model.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tools.mapanything_phone_scan.build_consensus_fusion import _surfel_fusion
from tools.mapanything_phone_scan.build_single_room_review_surface import (
    _record, _validate_alignment,
)


def build(consensus_dir: Path, alignment_dir: Path, output_dir: Path,
          *, camera_id: str, voxel_m: float = .02,
          max_total_pixels: int = 40_000_000) -> dict:
    started = time.monotonic()
    consensus_dir, alignment_dir, output_dir = (path.resolve() for path in (consensus_dir, alignment_dir, output_dir))
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite dense export: {output_dir}")
    if not .01 <= voxel_m <= .04:
        raise ValueError("dense review voxel size must be between 1 and 4 cm")
    if not 1 <= max_total_pixels <= 64_000_000:
        raise ValueError("dense total-pixel memory budget must be within 1..64,000,000")
    _, _, _, rows = _validate_alignment(consensus_dir, alignment_dir, expected_camera_id=camera_id)
    if len(rows) > 256:
        raise ValueError("dense export exceeds the 256-view budget")
    fused = {key: [] for key in ("depth", "quality", "mask", "rgb")}
    intrinsics, poses = [], []
    total_pixels, shape = 0, None
    for index, row in enumerate(rows):
        with np.load(row["path"], allow_pickle=False) as arrays:
            depth = np.asarray(arrays["depth_z"], dtype=np.float32).copy()
            quality = np.asarray(arrays["confidence"], dtype=np.float32).copy()
            mask = np.asarray(arrays["mask"], dtype=bool).copy()
            rgb = np.asarray(arrays["model_rgb"], dtype=np.uint8).copy()
            k = np.asarray(arrays["intrinsics"], dtype=np.float64).copy()
            pose = np.asarray(arrays["camera_pose"], dtype=np.float64).copy()
        shape = depth.shape if shape is None else shape
        if (depth.ndim != 2 or depth.shape != shape or quality.shape != shape
                or mask.shape != shape or rgb.shape != (*shape, 3)
                or not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 1))
                or k.shape != (3, 3) or pose.shape != (4, 4)
                or not np.isfinite(k).all() or not np.isfinite(pose).all()):
            raise ValueError(f"malformed accepted consensus view {index}")
        if np.any(mask & (~np.isfinite(depth) | (depth <= 0))):
            raise ValueError(f"accepted mask contains invalid depth in view {index}")
        total_pixels += depth.size
        if total_pixels > max_total_pixels:
            raise ValueError("dense export exceeds the total-pixel memory budget")
        for key, value in (("depth", depth), ("quality", quality), ("mask", mask), ("rgb", rgb)):
            fused[key].append(value)
        intrinsics.append(k)
        poses.append(pose)
    fused = {key: np.stack(value) for key, value in fused.items()}
    print(f"Canonical full-grid surfel fusion: {len(rows)} views, {total_pixels:,} pixels", flush=True)
    points, colors, weights, metrics = _surfel_fusion(
        fused, np.stack(intrinsics), np.stack(poses), voxel_m, pixel_stride=1,
    )
    if not len(points):
        raise ValueError("no dense surfels passed the unchanged acceptance gates")
    output_dir.mkdir(parents=True, exist_ok=False)
    accepted_path, withheld_path = output_dir / "surfel_points.npz", output_dir / "surfel_points_single_view.npz"
    np.savez_compressed(accepted_path, points=points, colors=colors, weights=weights,
                        support_view_count=metrics["_support_view_count"],
                        support_view_ids=metrics["_support_view_ids"])
    np.savez_compressed(withheld_path, points=metrics["_single_view_points"],
                        colors=metrics["_single_view_colors"], weights=metrics["_single_view_weights"],
                        support_view_count=metrics["_single_view_count"],
                        support_view_ids=metrics["_single_view_ids"])
    report = {
        "schema": "noesis.pcf.dense_surfel_export.v1", "status": "review_only",
        "accepted_for_canonical_use": False, "phone_walk_only": True,
        "static_camera_points_included": False, "raw_geometry_or_mask_changed": False,
        "method": "canonical_surfel_fusion_full_accepted_grid",
        "parameters": {"voxel_m": voxel_m, "pixel_stride": 1, "max_total_pixels": max_total_pixels},
        "source": {
            "consensus_output_manifest": _record(consensus_dir / "scan_outputs_manifest.json"),
            "consensus_manifest": _record(consensus_dir / "consensus_manifest.json"),
            "alignment_report": _record(alignment_dir / "alignment_report.json"),
            "alignment_transform": _record(alignment_dir / "phone_ma_to_noesis_world.json"),
            "raw_views": rows,
        },
        "metrics": {key: value for key, value in metrics.items() if not key.startswith("_")},
        "artifacts": {"accepted_surfels": _record(accepted_path), "withheld_surfels": _record(withheld_path)},
        "elapsed_s": time.monotonic() - started,
    }
    (output_dir / "dense_surfel_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metrics": report["metrics"], "elapsed_s": report["elapsed_s"]}, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("consensus-dir", "alignment-dir", "output-dir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--voxel-m", type=float, default=.02)
    parser.add_argument("--max-total-pixels", type=int, default=40_000_000,
                        help="Explicit bounded memory allowance; default unchanged, maximum 64 million pixels.")
    args = parser.parse_args()
    build(args.consensus_dir, args.alignment_dir, args.output_dir,
          camera_id=args.camera_id, voxel_m=args.voxel_m, max_total_pixels=args.max_total_pixels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
