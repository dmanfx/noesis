#!/usr/bin/env python3
"""Render a matched 48-view versus adaptive-view PCF visual comparison."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis_core.coordinate_frames import (  # noqa: E402
    CAMERA_LOCAL_RASTER_ORIENTATION,
    camera_ground_frame_from_camera_to_world,
    transform_positions,
)
from tools.mapanything_phone_scan.alignment import (  # noqa: E402
    _resolve_target_cloud_for_calibrated_camera,
)
from tools.mapanything_phone_scan.evaluate_mapanything_prior_variants import (  # noqa: E402
    Candidate,
    _load_sequence,
    _load_target_points,
    _render_diagnostic_montage,
    _trajectory_sim3_candidate,
    _transform_poses,
)
from tools.mapanything_phone_scan.render_phone_heatmap_diagnostics import (  # noqa: E402
    _camera_positions,
    _colorize,
    _point_splat,
    _present_camera_ground,
    _rasterize,
    _render_point_preserving_layers,
)
from tools.mapanything_phone_scan.run_mapanything_prior_variants import (  # noqa: E402
    _load_static_reference,
    _load_world_from_da3,
)


def _compose_pair(
    first: Path,
    second: Path,
    output: Path,
    labels: tuple[str, str],
) -> None:
    images = [Image.open(path).convert("RGB") for path in (first, second)]
    target_height = min(image.height for image in images)
    resized = [
        image.resize(
            (round(image.width * target_height / image.height), target_height),
            Image.Resampling.LANCZOS,
        )
        for image in images
    ]
    header = 72
    gutter = 12
    canvas = Image.new(
        "RGB",
        (sum(image.width for image in resized) + gutter, target_height + header),
        "#0d1014",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=24)
    x = 0
    for image, label in zip(resized, labels, strict=True):
        draw.text((x + 16, 20), label, fill="white", font=font)
        canvas.paste(image, (x, header))
        x += image.width + gutter
    canvas.save(output)


def _key_layers(
    candidates: list[Candidate],
    grids: dict[str, dict[str, np.ndarray]],
    splats: dict[str, np.ndarray],
    output: Path,
) -> None:
    rows = (
        "Structural composite",
        "Observed height",
        "Obstacle height",
        "2.5 cm point preserving",
    )
    figure, axes = plt.subplots(
        len(rows), 2, figsize=(14, 22), facecolor="#0d1014", squeeze=False
    )
    figure.suptitle(
        "PCF frame-density comparison · identical camera-ground crop and render parameters",
        color="white",
        fontsize=19,
        y=0.995,
    )
    for column, candidate in enumerate(candidates):
        candidate_grids = grids[candidate.slug]
        observed = candidate_grids["observed"]
        images = (
            candidate_grids["structural"],
            _colorize(candidate_grids["height"], "inferno", observed),
            _colorize(
                candidate_grids["obstacle_height"],
                "inferno",
                candidate_grids["obstacle_height"] > 0,
                (0.0, 1.8),
            ),
            splats[candidate.slug],
        )
        for row, (label, image) in enumerate(zip(rows, images, strict=True)):
            axis = axes[row, column]
            axis.imshow(image)
            axis.axis("off")
            if row == 0:
                axis.set_title(candidate.label, color="white", fontsize=15)
            if column == 0:
                axis.text(
                    -0.03,
                    0.5,
                    label,
                    color="#7bdcff",
                    fontsize=12,
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=axis.transAxes,
                )
    figure.text(
        0.5,
        0.005,
        "Same target-derived bounds · 5 cm diagnostic grid · 2.5 cm point splat",
        color="#bcc7d1",
        ha="center",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.02, 0.018, 0.99, 0.98), h_pad=1.2, w_pad=0.4)
    figure.savefig(output, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-raw", type=Path, required=True)
    parser.add_argument("--baseline-world-from-da3", type=Path, required=True)
    parser.add_argument("--adaptive-raw", type=Path, required=True)
    parser.add_argument("--adaptive-world-from-da3", type=Path, required=True)
    parser.add_argument("--target-revision", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-budget", type=int, default=900_000)
    parser.add_argument("--grid-res-m", type=float, default=0.05)
    parser.add_argument("--point-splat-res-m", type=float, default=0.025)
    parser.add_argument(
        "--reference-evaluation",
        type=Path,
        help=(
            "Reuse the saved 48-view evaluation bounds and presentation so the "
            "comparison is pixel-for-pixel parameter matched to that baseline."
        ),
    )
    return parser.parse_args()


def _pcf_candidate(
    slug: str,
    label: str,
    raw_root: Path,
    world_from_da3_path: Path,
    point_budget: int,
) -> Candidate:
    manifest_path = raw_root.parent / "consensus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("pose_carrier") != "da3":
        raise ValueError(f"{manifest_path} is not a DA3-carried PCF result")
    da3_raw_value = manifest.get("inputs", {}).get("da3_raw")
    if not da3_raw_value:
        raise ValueError(f"{manifest_path} does not identify its DA3 raw input")
    da3_raw = Path(str(da3_raw_value)).resolve()
    da3_sequence = _load_sequence(da3_raw, f"{label} DA3 carrier")
    target_poses = _transform_poses(
        _load_world_from_da3(world_from_da3_path.resolve()),
        da3_sequence.poses,
    )
    return _trajectory_sim3_candidate(
        slug,
        label,
        raw_root,
        target_poses,
        point_budget,
    )


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    target_revision = args.target_revision.resolve()
    reference_evaluation: Path | None = None
    if args.reference_evaluation is not None:
        reference_evaluation = args.reference_evaluation.resolve()
        reference = json.loads(reference_evaluation.read_text(encoding="utf-8"))
        bounds = tuple(float(value) for value in reference["shared_bev_bounds_xz_m"])
        if len(bounds) != 4:
            raise ValueError("reference evaluation has malformed shared BEV bounds")
        args.grid_res_m = float(reference.get("grid_res_m", args.grid_res_m))
        args.point_splat_res_m = float(
            reference.get("point_splat_res_m", args.point_splat_res_m)
        )
        rotation_deg = int(reference.get("presentation", {}).get("bev_rotation_deg", 0))
        if rotation_deg not in {0, 180}:
            raise ValueError(
                f"unsupported reference BEV rotation {rotation_deg}; expected 0 or 180"
            )
        presentation = np.eye(4, dtype=np.float64)
        if rotation_deg == 180:
            center_x = 0.5 * (bounds[0] + bounds[1])
            center_z = 0.5 * (bounds[2] + bounds[3])
            presentation[0, 0] = -1.0
            presentation[2, 2] = -1.0
            presentation[0, 3] = 2.0 * center_x
            presentation[2, 3] = 2.0 * center_z
    else:
        static = _load_static_reference(
            target_revision, args.calibration.resolve(), args.camera
        )
        target_points, _ = _load_target_points(target_revision)
        target_points, _, _ = _resolve_target_cloud_for_calibrated_camera(
            target_points, static.camera_to_world
        )
        target_meta = json.loads(
            (target_revision / "room_points_meta.json").read_text(encoding="utf-8")
        )
        floor_y_m = float(target_meta.get("floor_y", 0.0))
        presentation = camera_ground_frame_from_camera_to_world(
            static.camera_to_world
        ).world_to_camera_local_display_matrix(floor_y_m)
        presented_target = transform_positions(target_points, presentation)
        low = np.min(presented_target[:, [0, 2]], axis=0) - 0.35
        high = np.max(presented_target[:, [0, 2]], axis=0) + 0.35
        bounds = (float(low[0]), float(high[0]), float(low[1]), float(high[1]))

    baseline_raw = args.baseline_raw.resolve()
    adaptive_raw = args.adaptive_raw.resolve()
    baseline_count = len(list(baseline_raw.glob("view_*.npz")))
    adaptive_count = len(list(adaptive_raw.glob("view_*.npz")))
    candidates = [
        _pcf_candidate(
            "baseline_48",
            f"Baseline PCF · {baseline_count} views",
            baseline_raw,
            args.baseline_world_from_da3,
            args.point_budget,
        ),
        _pcf_candidate(
            "adaptive",
            f"Adaptive PCF · {adaptive_count} views",
            adaptive_raw,
            args.adaptive_world_from_da3,
            args.point_budget,
        ),
    ]
    grids: dict[str, dict[str, np.ndarray]] = {}
    splats: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {
        "schema": "noesis.pcf.frame_density_comparison.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "camera_id": args.camera,
        "target_revision": str(target_revision),
        "shared_bounds_camera_local_ground_m": list(bounds),
        "grid_res_m": float(args.grid_res_m),
        "point_splat_res_m": float(args.point_splat_res_m),
        "raster_orientation": CAMERA_LOCAL_RASTER_ORIENTATION,
        "reference_evaluation": (
            str(reference_evaluation) if reference_evaluation is not None else None
        ),
        "candidates": {},
    }
    labels = tuple(candidate.label for candidate in candidates)
    montage_paths: list[Path] = []
    layer_paths: list[Path] = []
    for candidate in candidates:
        candidate_dir = output_dir / candidate.slug
        candidate_dir.mkdir()
        presented = _present_camera_ground(candidate.cloud, presentation)
        camera_positions = _camera_positions(presented).copy()
        camera_outside = (
            (camera_positions[:, 0] < bounds[0])
            | (camera_positions[:, 0] >= bounds[1])
            | (camera_positions[:, 2] < bounds[2])
            | (camera_positions[:, 2] >= bounds[3])
        )
        if np.any(camera_outside):
            # A saved baseline crop may intentionally exclude part of a wider
            # walk. Keep the crop exact and clip only the green trajectory
            # overlay; reconstruction points remain untouched and bounded by
            # the normal rasterizer.
            epsilon = min(float(args.grid_res_m), 0.01)
            camera_positions[:, 0] = np.clip(
                camera_positions[:, 0], bounds[0] + epsilon, bounds[1] - epsilon
            )
            camera_positions[:, 2] = np.clip(
                camera_positions[:, 2], bounds[2] + epsilon, bounds[3] - epsilon
            )
            presented.presentation_camera_positions = camera_positions
        candidate_grids = _rasterize(presented, bounds, args.grid_res_m)
        grids[candidate.slug] = candidate_grids
        presented_candidate = Candidate(
            slug=candidate.slug,
            label=candidate.label,
            raw_root=candidate.raw_root,
            cloud=presented,
            sequence=candidate.sequence,
            aligned_poses=candidate.aligned_poses,
            alignment_scale=candidate.alignment_scale,
            alignment_rotation=candidate.alignment_rotation,
            alignment_translation=candidate.alignment_translation,
            alignment_method=candidate.alignment_method,
        )
        montage_paths.append(
            _render_diagnostic_montage(
                presented_candidate, candidate_grids, candidate_dir
            )
        )
        point_layers, point_metrics = _render_point_preserving_layers(
            candidate.label,
            presented,
            bounds,
            args.point_splat_res_m,
            candidate_dir,
        )
        layer_paths.append(Path(point_layers))
        splat, splat_metrics = _point_splat(
            presented,
            bounds,
            args.point_splat_res_m,
            (-0.15, 2.70),
        )
        splats[candidate.slug] = splat
        metrics["candidates"][candidate.slug] = {
            "label": candidate.label,
            "raw_root": str(candidate.raw_root),
            "view_count": int(candidate.sequence.depth.shape[0]),
            "sampled_point_count": int(candidate.cloud.points.shape[0]),
            "pcf_to_da3_world_alignment": {
                "method": candidate.alignment_method,
                "scale": float(candidate.alignment_scale),
            },
            "camera_path_positions_clipped_to_reference_crop": int(
                np.count_nonzero(camera_outside)
            ),
            "observed_5cm_cells": int(np.count_nonzero(candidate_grids["observed"])),
            "walkable_5cm_cells": int(
                np.count_nonzero(candidate_grids["walkable"] > 0.5)
            ),
            "obstacle_5cm_cells": int(
                np.count_nonzero(candidate_grids["obstacle_height"] > 0.0)
            ),
            "point_preserving": point_metrics,
            "point_splat": splat_metrics,
        }

    key_path = output_dir / "pcf_48_vs_adaptive_key_layers.png"
    _key_layers(candidates, grids, splats, key_path)
    full_path = output_dir / "pcf_48_vs_adaptive_full_diagnostics.png"
    _compose_pair(montage_paths[0], montage_paths[1], full_path, labels)
    points_path = output_dir / "pcf_48_vs_adaptive_point_layers.png"
    _compose_pair(layer_paths[0], layer_paths[1], points_path, labels)
    metrics["artifacts"] = {
        "key_layers": key_path.name,
        "full_diagnostics": full_path.name,
        "point_layers": points_path.name,
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
