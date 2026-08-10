#!/usr/bin/env python3
"""Evaluate and render DA3-prior MapAnything variants in static-camera world."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mapanything_phone_scan.alignment import (  # noqa: E402
    _full_cloud_metrics,
    _write_reprojection,
    _write_topdown,
)
from tools.mapanything_phone_scan.build_consensus_fusion import (  # noqa: E402
    Sequence,
    _heldout_reprojection,
    _load_sequence,
    _multiview_consistency,
    _umeyama,
)
from tools.mapanything_phone_scan.render_phone_heatmap_diagnostics import (  # noqa: E402
    PhoneCloud,
    _apply_rigid,
    _apply_sim3,
    _colorize,
    _depth_panel,
    _load_raw_phone_cloud,
    _overlay_panel,
    _point_splat,
    _rasterize,
    _render_point_preserving_layers,
    _save_panel,
)
from tools.mapanything_phone_scan.run_mapanything_prior_variants import (  # noqa: E402
    _load_static_reference,
    _load_world_from_da3,
    _rotation_error_deg,
    _transform_poses,
)


@dataclass
class Candidate:
    slug: str
    label: str
    raw_root: Path
    cloud: PhoneCloud
    sequence: Sequence
    aligned_poses: np.ndarray
    alignment_scale: float
    alignment_rotation: np.ndarray
    alignment_translation: np.ndarray
    alignment_method: str


def _camera_oriented_bev(image: np.ndarray) -> np.ndarray:
    """Put the foyer/hallway at the review convention's top-left."""
    return np.rot90(np.asarray(image), 2).copy()


def _load_target_points(revision: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(revision / "room_points.npz") as row:
        points = np.asarray(row["points"], dtype=np.float64)
        colors = np.asarray(row["colors"], dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("static target point cloud is malformed")
    return points, colors


def _identity_candidate(
    slug: str,
    label: str,
    raw_root: Path,
    point_budget: int,
) -> Candidate:
    cloud = _load_raw_phone_cloud(raw_root, point_budget, label)
    sequence = _load_sequence(raw_root, label)
    return Candidate(
        slug=slug,
        label=label,
        raw_root=raw_root,
        cloud=cloud,
        sequence=sequence,
        aligned_poses=sequence.poses.copy(),
        alignment_scale=1.0,
        alignment_rotation=np.eye(3),
        alignment_translation=np.zeros(3),
        alignment_method="already_backend_world_pose_carrier",
    )


def _rigid_candidate(
    slug: str,
    label: str,
    raw_root: Path,
    transform: np.ndarray,
    point_budget: int,
) -> Candidate:
    cloud = _load_raw_phone_cloud(raw_root, point_budget, label)
    sequence = _load_sequence(raw_root, label)
    aligned_cloud = _apply_rigid(cloud, transform)
    aligned_poses = _transform_poses(transform, sequence.poses)
    return Candidate(
        slug=slug,
        label=label,
        raw_root=raw_root,
        cloud=aligned_cloud,
        sequence=sequence,
        aligned_poses=aligned_poses,
        alignment_scale=1.0,
        alignment_rotation=transform[:3, :3].copy(),
        alignment_translation=transform[:3, 3].copy(),
        alignment_method="validated_da3_to_backend_rigid_transform",
    )


def _trajectory_sim3_candidate(
    slug: str,
    label: str,
    raw_root: Path,
    target_poses: np.ndarray,
    point_budget: int,
) -> Candidate:
    cloud = _load_raw_phone_cloud(raw_root, point_budget, label)
    sequence = _load_sequence(raw_root, label)
    if sequence.poses.shape[0] != target_poses.shape[0]:
        raise ValueError(
            f"{label} has {sequence.poses.shape[0]} poses; target has {target_poses.shape[0]}"
        )
    scale, rotation, translation = _umeyama(
        sequence.poses[:, :3, 3], target_poses[:, :3, 3]
    )
    aligned_cloud = _apply_sim3(cloud, scale, rotation, translation)
    aligned_poses = sequence.poses.copy()
    aligned_poses[:, :3, :3] = np.einsum(
        "ij,njk->nik", rotation, aligned_poses[:, :3, :3]
    )
    aligned_poses[:, :3, 3] = (
        scale * (rotation @ aligned_poses[:, :3, 3].T).T + translation
    )
    return Candidate(
        slug=slug,
        label=label,
        raw_root=raw_root,
        cloud=aligned_cloud,
        sequence=sequence,
        aligned_poses=aligned_poses,
        alignment_scale=float(scale),
        alignment_rotation=rotation,
        alignment_translation=translation,
        alignment_method="all_phone_pose_correspondence_sim3_to_validated_da3_backend_path",
    )


def _bounded_source(
    points: np.ndarray,
    target_points: np.ndarray,
    margin_m: float,
) -> np.ndarray:
    low = np.min(target_points, axis=0) - margin_m
    high = np.max(target_points, axis=0) + margin_m
    keep = np.all((points >= low) & (points <= high), axis=1)
    bounded = points[keep]
    if bounded.shape[0] < 1_000:
        raise ValueError("fewer than 1,000 source points overlap the static room bounds")
    return bounded


def _pose_metrics(poses: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    position = np.linalg.norm(poses[:, :3, 3] - target[:, :3, 3], axis=1)
    rotation = np.asarray(
        [_rotation_error_deg(pose, expected) for pose, expected in zip(poses, target)]
    )
    return {
        "position_error_m": {
            "median": float(np.median(position)),
            "p80": float(np.percentile(position, 80.0)),
            "max": float(np.max(position)),
        },
        "rotation_error_deg": {
            "median": float(np.median(rotation)),
            "p80": float(np.percentile(rotation, 80.0)),
            "max": float(np.max(rotation)),
        },
        "start_end_distance_m": float(
            np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3])
        ),
    }


def _render_diagnostic_montage(
    candidate: Candidate,
    grids: dict[str, np.ndarray],
    output_dir: Path,
) -> Path:
    observed = grids["observed"]
    panels: list[tuple[str, np.ndarray]] = [
        ("Camera Depth · Primary", _depth_panel(candidate.cloud.frame_zero_depth)),
        (
            "Exact Phone Frame RGB + Depth",
            _overlay_panel(
                candidate.cloud.frame_zero_rgb, candidate.cloud.frame_zero_depth
            ),
        ),
        (
            "Observed Height Floorplan · Primary",
            _camera_oriented_bev(_colorize(grids["height"], "inferno", observed)),
        ),
        ("Structural Composite (Diagnostic)", _camera_oriented_bev(grids["structural"])),
        (
            "Density (Grayscale)",
            _camera_oriented_bev(
                _colorize(grids["density"], "gray", observed, (0.0, 1.0))
            ),
        ),
        (
            "Raw Height (Inferno)",
            _camera_oriented_bev(_colorize(grids["height"], "inferno", observed)),
        ),
        (
            "Raw Height (Contrast)",
            _camera_oriented_bev(_colorize(grids["height"], "turbo", observed)),
        ),
        (
            "Height Above Floor",
            _camera_oriented_bev(
                _colorize(grids["height_agl"], "turbo", observed, (0.0, 1.2))
            ),
        ),
        (
            "Distance (Viridis)",
            _camera_oriented_bev(
                _colorize(grids["distance"], "viridis", observed)
            ),
        ),
        (
            "Obstacle Height (Clean)",
            _camera_oriented_bev(
                _colorize(
                    grids["obstacle_height"],
                    "inferno",
                    grids["obstacle_height"] > 0,
                    (0.0, 1.8),
                )
            ),
        ),
        (
            "Walkable (Binary)",
            _camera_oriented_bev(
                _colorize(
                    grids["walkable"],
                    "gray",
                    grids["walkable"] >= 0,
                    (0.0, 1.0),
                )
            ),
        ),
        (
            "Gradient (Edges)",
            _camera_oriented_bev(
                _colorize(grids["gradient"], "viridis", observed, (0.0, 1.0))
            ),
        ),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for title, image in panels:
        slug = (
            title.lower()
            .replace(" · ", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        _save_panel(output_dir / f"{slug}.png", image)
    figure, axes = plt.subplots(4, 3, figsize=(15, 18), facecolor="#0d1014")
    figure.suptitle(
        f"{candidate.label} · phone views in calibrated static-camera world",
        color="white",
        fontsize=18,
        y=0.992,
    )
    for axis, (title, image) in zip(axes.ravel(), panels):
        axis.imshow(image)
        axis.set_title(title, color="white", fontsize=11)
        axis.axis("off")
    figure.text(
        0.018,
        0.715,
        "Diagnostic layers",
        color="#7bdcff",
        fontsize=15,
        fontweight="bold",
        ha="left",
    )
    figure.text(
        0.5,
        0.008,
        "Shared backend-world crop · 5 cm diagnostics · green path is supplied/aligned phone trajectory",
        color="#bcc7d1",
        fontsize=10,
        ha="center",
    )
    figure.tight_layout(rect=(0.01, 0.025, 0.99, 0.975), h_pad=2.0)
    montage = output_dir / "static_world_heatmap_diagnostics.png"
    figure.savefig(montage, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    return montage


def _render_overview(
    candidates: list[Candidate],
    grids: dict[str, dict[str, np.ndarray]],
    splats: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(candidates),
        4,
        figsize=(17, 3.55 * len(candidates)),
        facecolor="#0d1014",
        squeeze=False,
    )
    figure.suptitle(
        "Horizontal phone walk · shared static-camera world crop",
        color="white",
        fontsize=20,
        y=0.988,
    )
    headers = ("Structural", "Height AGL", "Density", "2.5 cm point-preserving")
    for row, candidate in enumerate(candidates):
        candidate_grids = grids[candidate.slug]
        observed = candidate_grids["observed"]
        images = (
            _camera_oriented_bev(candidate_grids["structural"]),
            _camera_oriented_bev(
                _colorize(
                    candidate_grids["height_agl"],
                    "turbo",
                    observed,
                    (0.0, 1.2),
                )
            ),
            _camera_oriented_bev(
                _colorize(
                    candidate_grids["density"],
                    "gray",
                    observed,
                    (0.0, 1.0),
                )
            ),
            _camera_oriented_bev(splats[candidate.slug]),
        )
        for column, (axis, image) in enumerate(zip(axes[row], images)):
            axis.imshow(image, interpolation="nearest")
            axis.axis("off")
            if row == 0:
                axis.set_title(headers[column], color="#7bdcff", fontsize=12)
            if column == 0:
                axis.text(
                    -0.03,
                    0.5,
                    candidate.label,
                    color="white",
                    fontsize=11,
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=axis.transAxes,
                )
    figure.text(
        0.5,
        0.006,
        "Every row uses identical bounds and raster rules; static cloud is a reference/evaluation target, not drawn into phone layers.",
        color="#bcc7d1",
        fontsize=10,
        ha="center",
    )
    figure.tight_layout(rect=(0.055, 0.02, 0.995, 0.972), h_pad=1.25, w_pad=0.8)
    figure.savefig(output_path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--world-from-da3", type=Path, required=True)
    parser.add_argument("--target-revision", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--camera", default="living-room")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-budget", type=int, default=600_000)
    parser.add_argument("--grid-res-m", type=float, default=0.05)
    parser.add_argument("--point-splat-res-m", type=float, default=0.025)
    parser.add_argument(
        "--reuse-metrics-from",
        type=Path,
        help="Reuse an unchanged evaluation JSON and regenerate only presentation artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    scan_dir = args.scan_dir.resolve()
    suite_root = args.suite_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite existing evaluation: {output_dir}")
    output_dir.mkdir(parents=True)
    target_revision = args.target_revision.resolve()
    target_points, _ = _load_target_points(target_revision)
    static = _load_static_reference(
        target_revision,
        args.calibration.resolve(),
        args.camera,
    )
    world_from_da3 = _load_world_from_da3(args.world_from_da3.resolve())
    da3_raw = scan_dir / "da3_outputs" / "raw"
    da3_sequence = _load_sequence(da3_raw, "DA3")
    backend_da3_poses = _transform_poses(world_from_da3, da3_sequence.poses)

    candidates = [
        _trajectory_sim3_candidate(
            "mapanything_image_only",
            "MA image-only baseline",
            scan_dir / "outputs" / "raw",
            backend_da3_poses,
            args.point_budget,
        ),
        _rigid_candidate(
            "da3",
            "DA3 baseline",
            da3_raw,
            world_from_da3,
            args.point_budget,
        ),
        _trajectory_sim3_candidate(
            "consensus_da3_carrier",
            "MA+DA3 prior consensus",
            scan_dir / "consensus_da3_carrier_20260809" / "raw",
            backend_da3_poses,
            args.point_budget,
        ),
        _trajectory_sim3_candidate(
            "prior_conditioned_consensus",
            "Prior-conditioned MA+DA3 fusion",
            suite_root / "prior_conditioned_consensus_da3_carrier" / "raw",
            backend_da3_poses,
            args.point_budget,
        ),
        _identity_candidate(
            "ma_da3_pose",
            "MA + DA3 pose",
            suite_root / "mapanything_da3_pose" / "raw",
            args.point_budget,
        ),
        _trajectory_sim3_candidate(
            "ma_da3_sparse_depth",
            "MA + sparse DA3 depth",
            suite_root / "mapanything_da3_sparse_depth" / "raw",
            backend_da3_poses,
            args.point_budget,
        ),
        _identity_candidate(
            "ma_da3_pose_depth",
            "MA + DA3 pose/depth",
            suite_root / "mapanything_da3_pose_sparse_depth" / "raw",
            args.point_budget,
        ),
        _identity_candidate(
            "ma_da3_pose_depth_static",
            "MA + DA3 pose/depth + static",
            suite_root / "mapanything_da3_pose_sparse_depth_static" / "phone_raw",
            args.point_budget,
        ),
    ]

    low = np.min(target_points[:, [0, 2]], axis=0) - 0.35
    high = np.max(target_points[:, [0, 2]], axis=0) + 0.35
    bounds = (float(low[0]), float(high[0]), float(low[1]), float(high[1]))
    reused_metrics: dict[str, Any] | None = None
    if args.reuse_metrics_from is not None:
        reuse_path = args.reuse_metrics_from.resolve()
        reused_metrics = json.loads(reuse_path.read_text(encoding="utf-8"))
        if reused_metrics.get("schema") != "noesis.mapanything.prior_variant_evaluation.v1":
            raise ValueError(f"unsupported reused metrics schema in {reuse_path}")
        if reused_metrics.get("shared_bev_bounds_xz_m") != list(bounds):
            raise ValueError("reused metrics used different static-world bounds")
        if float(reused_metrics.get("grid_res_m")) != float(args.grid_res_m):
            raise ValueError("reused metrics used a different diagnostic grid resolution")
        if float(reused_metrics.get("point_splat_res_m")) != float(
            args.point_splat_res_m
        ):
            raise ValueError("reused metrics used a different point-splat resolution")
        metrics = copy.deepcopy(reused_metrics)
        metrics["generated_at"] = datetime.now(timezone.utc).isoformat()
        metrics["metrics_reused_from"] = str(reuse_path)
        metrics["presentation"] = {
            "bev_rotation_deg": 180,
            "reason": "camera-oriented review convention with foyer/hallway at top-left",
        }
        metrics["candidates"] = {}
    else:
        metrics = {
            "schema": "noesis.mapanything.prior_variant_evaluation.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "coordinate_frame": "backend_world_m_stream_points",
            "alignment_reference": "validated DA3 phone trajectory mapped into static-camera backend world",
            "static_target_revision": str(target_revision),
            "shared_bev_bounds_xz_m": list(bounds),
            "grid_res_m": float(args.grid_res_m),
            "point_splat_res_m": float(args.point_splat_res_m),
            "presentation": {
                "bev_rotation_deg": 180,
                "reason": "camera-oriented review convention with foyer/hallway at top-left",
            },
            "candidates": {},
        }
    grids_by_slug: dict[str, dict[str, np.ndarray]] = {}
    splats_by_slug: dict[str, np.ndarray] = {}

    for index, candidate in enumerate(candidates):
        cached_row: dict[str, Any] | None = None
        if reused_metrics is not None:
            cached_candidates = reused_metrics.get("candidates")
            if isinstance(cached_candidates, dict) and candidate.slug in cached_candidates:
                cached_row = copy.deepcopy(cached_candidates[candidate.slug])
        needs_evaluation = cached_row is None
        action = "evaluating" if needs_evaluation else "rendering"
        print(f"[{index + 1}/{len(candidates)}] {action} {candidate.label}", flush=True)
        candidate_dir = output_dir / candidate.slug
        candidate_dir.mkdir()
        reprojection_path = candidate_dir / "fixed_camera_reprojection.jpg"
        topdown_path = candidate_dir / "static_alignment_topdown.png"
        if needs_evaluation:
            bounded = _bounded_source(candidate.cloud.points, target_points, 0.25)
            full_metrics = _full_cloud_metrics(candidate.cloud.points, target_points)
            bounded_metrics = _full_cloud_metrics(bounded, target_points)
            consistency = _multiview_consistency(
                candidate.sequence.depth * candidate.alignment_scale,
                candidate.sequence.mask,
                candidate.sequence.intrinsics,
                candidate.aligned_poses,
            )
            heldout = _heldout_reprojection(
                candidate.sequence.depth * candidate.alignment_scale,
                candidate.sequence.mask,
                candidate.sequence.intrinsics,
                candidate.aligned_poses,
            )
            reprojection = _write_reprojection(
                reprojection_path,
                static.source_image,
                candidate.cloud.points,
                candidate.cloud.colors,
                target_points,
                static.camera_from_world,
                static.intrinsics,
            )
            _write_topdown(
                topdown_path,
                candidate.cloud.points,
                target_points,
                candidate.cloud.camera_to_world[:, :3, 3],
                static.camera_to_world[:3, 3],
            )
        else:
            reuse_root = args.reuse_metrics_from.resolve().parent
            for filename, target in (
                ("fixed_camera_reprojection.jpg", reprojection_path),
                ("static_alignment_topdown.png", topdown_path),
            ):
                source = reuse_root / candidate.slug / filename
                if not source.is_file():
                    raise ValueError(f"reused artifact is missing: {source}")
                os.link(source, target)
        grids = _rasterize(candidate.cloud, bounds, args.grid_res_m)
        grids_by_slug[candidate.slug] = grids
        montage = _render_diagnostic_montage(candidate, grids, candidate_dir)
        point_layers, point_layer_metrics = _render_point_preserving_layers(
            candidate.label,
            candidate.cloud,
            bounds,
            args.point_splat_res_m,
            candidate_dir,
            rotate_180=True,
        )
        splat, splat_metrics = _point_splat(
            candidate.cloud,
            bounds,
            args.point_splat_res_m,
            (-0.15, 2.70),
        )
        splats_by_slug[candidate.slug] = splat
        _save_panel(
            candidate_dir / "point_splat_all_2p5cm.png",
            _camera_oriented_bev(splat),
        )
        artifact_row = {
            "diagnostic_layout": str(montage.relative_to(output_dir)),
            "point_preserving_layout": str(Path(point_layers).relative_to(output_dir)),
            "static_alignment_topdown": f"{candidate.slug}/static_alignment_topdown.png",
            "fixed_camera_reprojection": str(reprojection_path.relative_to(output_dir)),
        }
        if cached_row is not None:
            cached_row["artifacts"] = artifact_row
            cached_row["render_counts"]["point_splat_2p5cm"] = splat_metrics
            cached_row["render_counts"]["point_layers_2p5cm"] = point_layer_metrics
            metrics["candidates"][candidate.slug] = cached_row
        else:
            metrics["candidates"][candidate.slug] = {
                "label": candidate.label,
                "raw_root": str(candidate.raw_root),
                "alignment": {
                    "method": candidate.alignment_method,
                    "scale": candidate.alignment_scale,
                    "rotation_row_major": candidate.alignment_rotation.tolist(),
                    "translation": candidate.alignment_translation.tolist(),
                },
                "trajectory_vs_validated_da3_backend": _pose_metrics(
                    candidate.aligned_poses, backend_da3_poses
                ),
                "internal_multiview_reprojection": {
                    "median_error_m": consistency.median_error_m,
                    "p80_error_m": consistency.p80_error_m,
                },
                "heldout_even_to_odd_reprojection": heldout,
                "static_cloud_full_metrics": full_metrics,
                "static_cloud_room_bounds_metrics": bounded_metrics,
                "fixed_camera_reprojection": reprojection,
                "render_counts": {
                    "sampled_phone_point_count": int(candidate.cloud.points.shape[0]),
                    "bounded_phone_point_count": int(bounded.shape[0]),
                    "observed_5cm_cells": int(np.count_nonzero(grids["observed"])),
                    "walkable_5cm_cells": int(
                        np.count_nonzero(grids["walkable"] > 0.5)
                    ),
                    "obstacle_5cm_cells": int(
                        np.count_nonzero(grids["obstacle_height"] > 0)
                    ),
                    "point_splat_2p5cm": splat_metrics,
                    "point_layers_2p5cm": point_layer_metrics,
                },
                "artifacts": artifact_row,
            }

    overview_path = output_dir / "prior_variant_static_world_overview.png"
    _render_overview(candidates, grids_by_slug, splats_by_slug, overview_path)
    metrics["artifacts"] = {
        "overview": overview_path.name,
        "metrics": "evaluation_metrics.json",
    }
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
