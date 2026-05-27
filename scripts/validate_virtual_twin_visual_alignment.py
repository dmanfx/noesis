#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.virtual_twin.artifacts import write_json
from noesis.virtual_twin.builder import (
    VirtualTwinBuildError,
    _structural_visibility_buffer,
    _structural_visibility_surfaces,
)
from noesis.virtual_twin.menon_obj import parse_menon_obj_surfaces
from noesis.virtual_twin.store import VirtualTwinStore, validate_revision_id
from scripts.refresh_virtual_twin_render_artifacts import _load_frames, _model_path, _read_json, _replace_json


def _safe_stem(value: str) -> str:
    raw = str(value or "").strip() or "frame"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)[:96]


def _surface_boundary(surface_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(surface_ids, dtype=np.int32)
    boundary = np.zeros(ids.shape, dtype=bool)
    visible = ids >= 0
    boundary[:-1, :] |= visible[:-1, :] & (ids[:-1, :] != ids[1:, :])
    boundary[1:, :] |= visible[1:, :] & (ids[1:, :] != ids[:-1, :])
    boundary[:, :-1] |= visible[:, :-1] & (ids[:, :-1] != ids[:, 1:])
    boundary[:, 1:] |= visible[:, 1:] & (ids[:, 1:] != ids[:, :-1])
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1).astype(bool) & visible


def _label_color(label: str) -> np.ndarray:
    text = str(label or "").lower()
    if text == "floor":
        return np.asarray([60, 190, 255], dtype=np.float32)
    if text == "wall":
        return np.asarray([255, 190, 60], dtype=np.float32)
    return np.asarray([180, 120, 255], dtype=np.float32)


def _frame_visual(
    *,
    frame: Any,
    buffer: dict[str, Any],
    visibility_surfaces: list[tuple[Any, str]],
    output_dir: Path,
    canny_low: float,
    canny_high: float,
) -> dict[str, Any]:
    surface_ids = np.asarray(buffer["surface_ids"], dtype=np.int32)
    scale = max(1, int(buffer.get("downsample", 1)))
    h, w = int(surface_ids.shape[0]), int(surface_ids.shape[1])
    image = cv2.resize(np.asarray(frame.image_bgr, dtype=np.uint8), (w, h), interpolation=cv2.INTER_AREA)
    visible = surface_ids >= 0
    color_layer = np.zeros_like(image, dtype=np.float32)
    label_pixels: dict[str, int] = {}
    unique_ids = np.unique(surface_ids[visible]) if np.any(visible) else np.asarray([], dtype=np.int32)
    for idx in unique_ids:
        label = visibility_surfaces[int(idx)][1] if 0 <= int(idx) < len(visibility_surfaces) else "unknown"
        mask = surface_ids == int(idx)
        label_pixels[label] = int(label_pixels.get(label, 0) + np.count_nonzero(mask))
        color_layer[mask] = _label_color(label)

    overlay = image.copy().astype(np.float32)
    overlay[visible] = (overlay[visible] * 0.62) + (color_layer[visible] * 0.38)
    boundary = _surface_boundary(surface_ids)
    overlay[boundary] = np.asarray([0, 255, 255], dtype=np.float32)
    overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, float(canny_low), float(canny_high))
    distance_input = np.where(edges > 0, 0, 255).astype(np.uint8)
    distance = cv2.distanceTransform(distance_input, cv2.DIST_L2, 3)
    boundary_count = int(np.count_nonzero(boundary))
    samples = distance[boundary].astype(np.float64) * float(scale) if boundary_count else np.asarray([], dtype=np.float64)
    edge_panel = image.copy()
    edge_panel[edges > 0] = np.asarray([255, 255, 255], dtype=np.uint8)
    edge_panel[boundary] = np.asarray([0, 255, 255], dtype=np.uint8)

    side_by_side = np.concatenate([image, overlay_u8, edge_panel], axis=1)
    stem = _safe_stem(str(frame.frame_id))
    overlay_rel = Path("visual_validation") / f"{stem}_model_overlay.png"
    side_rel = Path("visual_validation") / f"{stem}_camera_model_compare.png"
    cv2.imwrite(str(output_dir / overlay_rel.name), overlay_u8)
    cv2.imwrite(str(output_dir / side_rel.name), side_by_side)

    def pct(value: float) -> float | None:
        return float(np.percentile(samples, value)) if samples.size else None

    return {
        "frame_id": str(frame.frame_id),
        "downsample": int(scale),
        "image_size": [int(frame.image_bgr.shape[1]), int(frame.image_bgr.shape[0])],
        "validation_image_size": [w, h],
        "visible_model_pixel_ratio": float(np.count_nonzero(visible) / max(1, visible.size)),
        "visible_model_pixel_count": int(np.count_nonzero(visible)),
        "model_boundary_pixel_count": boundary_count,
        "image_edge_pixel_count": int(np.count_nonzero(edges)),
        "boundary_to_image_edge_median_px": pct(50.0),
        "boundary_to_image_edge_p90_px": pct(90.0),
        "boundary_to_image_edge_p95_px": pct(95.0),
        "boundary_edge_support_ratio_6px": float(np.mean(samples <= 6.0)) if samples.size else 0.0,
        "boundary_edge_support_ratio_12px": float(np.mean(samples <= 12.0)) if samples.size else 0.0,
        "surface_label_visible_pixels": label_pixels,
        "overlay": str(overlay_rel),
        "camera_model_compare": str(side_rel),
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    store = VirtualTwinStore(Path(args.output_root) if args.output_root else None)
    revision_id = validate_revision_id(args.revision or store.latest_revision_id() or "")
    revision_dir = store.revision_dir(revision_id)
    manifest = _read_json(revision_dir / "manifest.json")
    metrics = _read_json(revision_dir / "metrics.json")
    tracking = _read_json(revision_dir / "tracking_alignment.json")
    registration = tracking.get("pose_correction")
    if not isinstance(registration, dict):
        raise VirtualTwinBuildError("tracking_alignment.json has no pose_correction")

    surfaces = parse_menon_obj_surfaces(_model_path(manifest, args.model_obj))
    visibility_surfaces = _structural_visibility_surfaces(surfaces)
    frames = _load_frames(revision_dir, manifest)
    output_dir = revision_dir / "visual_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_rows: list[dict[str, Any]] = []
    for frame in frames:
        buffer = _structural_visibility_buffer(
            frame,
            visibility_surfaces,
            registration,
            downsample=int(args.downsample),
        )
        frame_rows.append(
            _frame_visual(
                frame=frame,
                buffer=buffer,
                visibility_surfaces=visibility_surfaces,
                output_dir=output_dir,
                canny_low=float(args.canny_low),
                canny_high=float(args.canny_high),
            )
        )

    medians = np.asarray(
        [row["boundary_to_image_edge_median_px"] for row in frame_rows if row["boundary_to_image_edge_median_px"] is not None],
        dtype=np.float64,
    )
    p90s = np.asarray(
        [row["boundary_to_image_edge_p90_px"] for row in frame_rows if row["boundary_to_image_edge_p90_px"] is not None],
        dtype=np.float64,
    )
    visible = np.asarray([row["visible_model_pixel_ratio"] for row in frame_rows], dtype=np.float64)
    support6 = np.asarray([row["boundary_edge_support_ratio_6px"] for row in frame_rows], dtype=np.float64)
    validation = {
        "schema": "noesis.virtual_twin.camera_visual_alignment.v1",
        "revision_id": revision_id,
        "generated_ts_us": int(time.time() * 1_000_000),
        "method": "structural_model_projection_over_keyframe_with_edge_distance_diagnostics",
        "artifact_dir": "visual_validation",
        "frames": frame_rows,
        "aggregate": {
            "frame_count": len(frame_rows),
            "median_boundary_to_image_edge_px": float(np.median(medians)) if medians.size else None,
            "p90_boundary_to_image_edge_px": float(np.median(p90s)) if p90s.size else None,
            "median_visible_model_pixel_ratio": float(np.median(visible)) if visible.size else 0.0,
            "median_boundary_edge_support_ratio_6px": float(np.median(support6)) if support6.size else 0.0,
        },
        "notes": [
            "Yellow lines are projected Menon structural boundaries.",
            "White lines are Canny edges from the camera keyframe.",
            "Lower boundary-to-edge distance means the model projection visually agrees better with the camera image.",
        ],
    }
    write_json(output_dir / "visual_validation.json", validation)

    if not args.no_update_metrics:
        metrics["camera_visual_alignment"] = validation
        metrics.setdefault("gates", {})["camera_visual_alignment_visible_projection_pass"] = bool(
            float(validation["aggregate"]["median_visible_model_pixel_ratio"]) >= float(args.min_visible_ratio)
        )
        metrics["gates"]["camera_visual_alignment_edge_support_pass"] = bool(
            float(validation["aggregate"]["median_boundary_edge_support_ratio_6px"]) >= float(args.min_edge_support_ratio_6px)
        )
        manifest.setdefault("artifacts", {})["visual_validation_json"] = "visual_validation/visual_validation.json"
        if frame_rows:
            manifest["artifacts"]["visual_validation_first_compare_png"] = str(frame_rows[0]["camera_model_compare"])
        _replace_json(revision_dir / "metrics.json", metrics)
        _replace_json(revision_dir / "manifest.json", manifest)
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render camera-vs-model visual validation overlays for a virtual twin revision.")
    parser.add_argument("--revision", default=None, help="Revision id; defaults to latest.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model-obj", type=Path, default=None)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--canny-low", type=float, default=55.0)
    parser.add_argument("--canny-high", type=float, default=150.0)
    parser.add_argument("--min-visible-ratio", type=float, default=0.05)
    parser.add_argument("--min-edge-support-ratio-6px", type=float, default=0.12)
    parser.add_argument("--no-update-metrics", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        result = validate(parse_args())
    except Exception as exc:
        print(f"virtual-twin visual alignment validation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
