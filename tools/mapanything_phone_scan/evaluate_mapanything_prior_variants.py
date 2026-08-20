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
import cv2
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mapanything_phone_scan.alignment import (  # noqa: E402
    _fixed_camera_visible_cloud_metrics,
    _fixed_camera_visible_structure_metrics,
    _full_cloud_metrics,
    _project_depth_grid,
    _resolve_target_cloud_for_calibrated_camera,
    _vertical_structure,
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
    _camera_positions,
    _colorize,
    _depth_panel,
    _load_raw_phone_cloud,
    _overlay_panel,
    _point_splat,
    _present_camera_ground,
    _rasterize,
    _render_point_preserving_layers,
    _save_panel,
)
from noesis_core.coordinate_frames import (  # noqa: E402
    CAMERA_LOCAL_RASTER_ORIENTATION,
    camera_ground_frame_from_camera_to_world,
    transform_positions,
)
from tools.mapanything_phone_scan.run_mapanything_prior_variants import (  # noqa: E402
    VARIANT_SPECS,
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


def _pose_carrier_alignment_mode(variant_root: Path) -> str:
    manifest_path = variant_root / "variant_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"conditioned variant manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coordinate_frame = str(manifest.get("coordinate_frame") or "")
    if coordinate_frame == "backend_world_m_stream_points":
        return "identity"
    if coordinate_frame == "da3_metric_world_unaligned_to_noesis":
        return "world_from_da3"
    raise ValueError(
        f"conditioned variant has unsupported coordinate frame {coordinate_frame!r}: "
        f"{manifest_path}"
    )


def _pose_carrier_candidate(
    slug: str,
    label: str,
    variant_root: Path,
    raw_relative: str,
    world_from_da3: np.ndarray,
    point_budget: int,
) -> Candidate:
    raw_root = variant_root / raw_relative
    alignment_mode = _pose_carrier_alignment_mode(variant_root)
    if alignment_mode == "identity":
        return _identity_candidate(slug, label, raw_root, point_budget)
    return _rigid_candidate(
        slug,
        label,
        raw_root,
        world_from_da3,
        point_budget,
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
            _colorize(grids["height"], "inferno", observed),
        ),
        ("Structural Composite (Diagnostic)", grids["structural"]),
        (
            "Density (Grayscale)",
            _colorize(grids["density"], "gray", observed, (0.0, 1.0)),
        ),
        (
            "Raw Height (Inferno)",
            _colorize(grids["height"], "inferno", observed),
        ),
        (
            "Raw Height (Contrast)",
            _colorize(grids["height"], "turbo", observed),
        ),
        (
            "Height Above Floor",
            _colorize(grids["height_agl"], "turbo", observed, (0.0, 1.2)),
        ),
        (
            "Distance (Viridis)",
            _colorize(grids["distance"], "viridis", observed),
        ),
        (
            "Obstacle Height (Clean)",
            _colorize(
                grids["obstacle_height"],
                "inferno",
                grids["obstacle_height"] > 0,
                (0.0, 1.8),
            ),
        ),
        (
            "Walkable (Binary)",
            _colorize(
                grids["walkable"],
                "gray",
                grids["walkable"] >= 0,
                (0.0, 1.0),
            ),
        ),
        (
            "Gradient (Edges)",
            _colorize(grids["gradient"], "viridis", observed, (0.0, 1.0)),
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
        "Shared camera-local crop · row zero is forward · green path is supplied/aligned phone trajectory",
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
        "Horizontal phone walk · shared static-camera presentation crop",
        color="white",
        fontsize=20,
        y=0.988,
    )
    headers = ("Structural", "Height AGL", "Density", "2.5 cm point-preserving")
    for row, candidate in enumerate(candidates):
        candidate_grids = grids[candidate.slug]
        observed = candidate_grids["observed"]
        images = (
            candidate_grids["structural"],
            _colorize(
                candidate_grids["height_agl"],
                "turbo",
                observed,
                (0.0, 1.2),
            ),
            _colorize(
                candidate_grids["density"],
                "gray",
                observed,
                (0.0, 1.0),
            ),
            splats[candidate.slug],
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
    parser.add_argument(
        "--da3-raw",
        type=Path,
        required=True,
        help="Raw DA3 output used to construct the sparse depth and pose priors.",
    )
    parser.add_argument(
        "--prior-consensus-raw",
        type=Path,
        required=True,
        help="Raw prior-conditioned MapAnything plus DA3 consensus output.",
    )
    parser.add_argument(
        "--image-only-raw",
        type=Path,
        help="Optional image-only MapAnything control from the same prepared views.",
    )
    parser.add_argument(
        "--comparison-consensus-raw",
        type=Path,
        help="Optional earlier image-only MapAnything plus DA3 consensus control.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANT_SPECS),
        default=list(VARIANT_SPECS),
        help="Conditioned MapAnything variants present under --suite-root.",
    )
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
    prepared_manifest = scan_dir / "prepared_frames_manifest.json"
    if not prepared_manifest.is_file():
        raise ValueError(f"prepared frame manifest is missing: {prepared_manifest}")
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
    target_points, _, _ = _resolve_target_cloud_for_calibrated_camera(
        target_points,
        static.camera_to_world,
    )
    target_image = cv2.imread(str(static.source_image), cv2.IMREAD_COLOR)
    if target_image is None:
        raise ValueError(f"static-camera keyframe is unreadable: {static.source_image}")
    target_depth_grid = _project_depth_grid(
        target_points,
        static.camera_from_world,
        static.intrinsics,
        target_image.shape[1],
        target_image.shape[0],
        8,
    )
    target_structure, target_structure_normals = _vertical_structure(
        target_points,
        0.10,
    )
    world_from_da3 = _load_world_from_da3(args.world_from_da3.resolve())
    da3_raw = args.da3_raw.resolve()
    da3_sequence = _load_sequence(da3_raw, "DA3")
    backend_da3_poses = _transform_poses(world_from_da3, da3_sequence.poses)

    candidates: list[Candidate] = []
    if args.image_only_raw is not None:
        candidates.append(
            _trajectory_sim3_candidate(
                "mapanything_image_only",
                "MA image-only baseline",
                args.image_only_raw.resolve(),
                backend_da3_poses,
                args.point_budget,
            )
        )
    candidates.append(
        _rigid_candidate(
            "da3",
            "DA3 baseline",
            da3_raw,
            world_from_da3,
            args.point_budget,
        )
    )
    if args.comparison_consensus_raw is not None:
        candidates.append(
            _trajectory_sim3_candidate(
                "consensus_da3_carrier",
                "MA+DA3 image-only consensus control",
                args.comparison_consensus_raw.resolve(),
                backend_da3_poses,
                args.point_budget,
            )
        )
    candidates.append(
        _trajectory_sim3_candidate(
            "prior_conditioned_consensus",
            "Prior-conditioned MA+DA3 fusion",
            args.prior_consensus_raw.resolve(),
            backend_da3_poses,
            args.point_budget,
        )
    )
    if "da3_pose" in args.variants:
        candidates.append(
            _pose_carrier_candidate(
                "ma_da3_pose",
                "MA + DA3 pose",
                suite_root / "mapanything_da3_pose",
                "raw",
                world_from_da3,
                args.point_budget,
            )
        )
    if "da3_sparse_depth" in args.variants:
        candidates.append(
            _trajectory_sim3_candidate(
                "ma_da3_sparse_depth",
                "MA + sparse DA3 depth",
                suite_root / "mapanything_da3_sparse_depth" / "raw",
                backend_da3_poses,
                args.point_budget,
            )
        )
    if "da3_pose_sparse_depth" in args.variants:
        candidates.append(
            _pose_carrier_candidate(
                "ma_da3_pose_depth",
                "MA + DA3 pose/depth",
                suite_root / "mapanything_da3_pose_sparse_depth",
                "raw",
                world_from_da3,
                args.point_budget,
            )
        )
    if "da3_pose_sparse_depth_static" in args.variants:
        candidates.append(
            _pose_carrier_candidate(
                "ma_da3_pose_depth_static",
                "MA + DA3 pose/depth + static",
                suite_root / "mapanything_da3_pose_sparse_depth_static",
                "phone_raw",
                world_from_da3,
                args.point_budget,
            )
        )

    metric_low = np.min(target_points[:, [0, 2]], axis=0) - 0.35
    metric_high = np.max(target_points[:, [0, 2]], axis=0) + 0.35
    metric_bounds = (
        float(metric_low[0]),
        float(metric_high[0]),
        float(metric_low[1]),
        float(metric_high[1]),
    )
    target_metadata_path = target_revision / "room_points_meta.json"
    target_metadata = json.loads(target_metadata_path.read_text(encoding="utf-8"))
    floor_y_m = float(target_metadata.get("floor_y", 0.0))
    presentation_frame = camera_ground_frame_from_camera_to_world(
        static.camera_to_world
    )
    presentation_matrix = presentation_frame.world_to_camera_local_display_matrix(
        floor_y_m
    )
    presented_target = transform_positions(target_points, presentation_matrix)
    presented_static_camera = transform_positions(
        static.camera_to_world[None, :3, 3],
        presentation_matrix,
    )[0]
    presentation_low = np.min(presented_target[:, [0, 2]], axis=0) - 0.35
    presentation_high = np.max(presented_target[:, [0, 2]], axis=0) + 0.35
    presentation_bounds = (
        float(presentation_low[0]),
        float(presentation_high[0]),
        float(presentation_low[1]),
        float(presentation_high[1]),
    )
    presentation_metadata = {
        "source_coordinate_frame": "backend_world_m",
        "target_coordinate_frame": "camera_local_ground_m",
        "presentation_only": True,
        "backend_geometry_mutated": False,
        "reference_camera_id": args.camera,
        "world_to_camera_local_row_major": presentation_matrix.tolist(),
        "linear_determinant": float(np.linalg.det(presentation_matrix[:3, :3])),
        "shared_bounds_xz_m": list(presentation_bounds),
        "raster_orientation": CAMERA_LOCAL_RASTER_ORIENTATION,
        "screen_right": "camera_right_positive_x",
        "screen_up": "camera_forward_positive_z",
        "room_specific_rotation_deg": 0,
    }
    reused_metrics: dict[str, Any] | None = None
    if args.reuse_metrics_from is not None:
        reuse_path = args.reuse_metrics_from.resolve()
        reused_metrics = json.loads(reuse_path.read_text(encoding="utf-8"))
        if reused_metrics.get("schema") != "noesis.mapanything.prior_variant_evaluation.v1":
            raise ValueError(f"unsupported reused metrics schema in {reuse_path}")
        if reused_metrics.get("shared_bev_bounds_xz_m") != list(metric_bounds):
            raise ValueError("reused metrics used different static-world bounds")
        if float(reused_metrics.get("grid_res_m")) != float(args.grid_res_m):
            raise ValueError("reused metrics used a different diagnostic grid resolution")
        if float(reused_metrics.get("point_splat_res_m")) != float(
            args.point_splat_res_m
        ):
            raise ValueError("reused metrics used a different point-splat resolution")
        metrics = copy.deepcopy(reused_metrics)
        metrics["generated_at"] = datetime.now(timezone.utc).isoformat()
        metrics["scan_dir"] = str(scan_dir)
        metrics["metrics_reused_from"] = str(reuse_path)
        metrics["presentation"] = presentation_metadata
        metrics["candidates"] = {}
    else:
        metrics = {
            "schema": "noesis.mapanything.prior_variant_evaluation.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scan_dir": str(scan_dir),
            "coordinate_frame": "backend_world_m_stream_points",
            "alignment_reference": "validated DA3 phone trajectory mapped into static-camera backend world",
            "static_target_revision": str(target_revision),
            "shared_bev_bounds_xz_m": list(metric_bounds),
            "grid_res_m": float(args.grid_res_m),
            "point_splat_res_m": float(args.point_splat_res_m),
            "presentation": presentation_metadata,
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
        presentation_cloud = _present_camera_ground(
            candidate.cloud,
            presentation_matrix,
        )
        if needs_evaluation:
            bounded = _bounded_source(candidate.cloud.points, target_points, 0.25)
            full_metrics = _full_cloud_metrics(candidate.cloud.points, target_points)
            bounded_metrics = _full_cloud_metrics(bounded, target_points)
            visible_cloud_metrics = _fixed_camera_visible_cloud_metrics(
                candidate.cloud.points,
                target_points,
                target_depth_grid,
                static.camera_from_world,
                static.intrinsics,
            )
            candidate_structure, _ = _vertical_structure(
                candidate.cloud.points,
                0.10,
            )
            visible_structure_metrics = _fixed_camera_visible_structure_metrics(
                candidate_structure,
                target_structure,
                target_structure_normals,
                target_depth_grid,
                static.camera_from_world,
                static.intrinsics,
            )
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
                args.camera.replace("-", " "),
            )
        else:
            reuse_root = args.reuse_metrics_from.resolve().parent
            source = reuse_root / candidate.slug / "fixed_camera_reprojection.jpg"
            if not source.is_file():
                raise ValueError(f"reused artifact is missing: {source}")
            os.link(source, reprojection_path)
        # This file is a presentation diagnostic, so regenerate it even when
        # metric results are reused.  Older copies may contain the retired
        # backend-X/Z or Living-Room-specific orientation.
        _write_topdown(
            topdown_path,
            presentation_cloud.points,
            presented_target,
            _camera_positions(presentation_cloud),
            presented_static_camera,
        )
        grids = _rasterize(
            presentation_cloud,
            presentation_bounds,
            args.grid_res_m,
        )
        grids_by_slug[candidate.slug] = grids
        montage = _render_diagnostic_montage(candidate, grids, candidate_dir)
        point_layers, point_layer_metrics = _render_point_preserving_layers(
            candidate.label,
            presentation_cloud,
            presentation_bounds,
            args.point_splat_res_m,
            candidate_dir,
        )
        splat, splat_metrics = _point_splat(
            presentation_cloud,
            presentation_bounds,
            args.point_splat_res_m,
            (-0.15, 2.70),
        )
        splats_by_slug[candidate.slug] = splat
        _save_panel(
            candidate_dir / "point_splat_all_2p5cm.png",
            splat,
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
                "fixed_camera_visible_cloud_metrics": visible_cloud_metrics,
                "fixed_camera_visible_structure_metrics": visible_structure_metrics,
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
