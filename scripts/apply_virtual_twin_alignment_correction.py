#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.virtual_twin.artifacts import write_json, write_points_npz
from noesis.virtual_twin.builder import (
    VirtualTwinBuildError,
    _model_acceptance_mask,
    _registration_matrix,
    revision_id_from_clock,
)
from noesis.virtual_twin.geometry import transform_points
from noesis.virtual_twin.menon_obj import parse_menon_obj_surfaces
from noesis.virtual_twin.store import VirtualTwinStore, validate_revision_id
from scripts.refresh_virtual_twin_render_artifacts import (
    _clone_revision,
    _model_path,
    _read_json,
    _replace_json,
    _update_revision_ids,
    refresh as refresh_render_artifacts,
)


def _correction_matrix(
    *,
    center: np.ndarray,
    translation: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    scale: float,
) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    roll = math.radians(float(roll_deg))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    rz = np.asarray([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    correction = np.eye(4, dtype=np.float64)
    correction[:3, :3] = (ry @ rx @ rz) * float(scale)
    to_origin = np.eye(4, dtype=np.float64)
    to_origin[:3, 3] = -np.asarray(center, dtype=np.float64).reshape(3)
    from_origin = np.eye(4, dtype=np.float64)
    from_origin[:3, 3] = np.asarray(center, dtype=np.float64).reshape(3) + np.asarray(translation, dtype=np.float64).reshape(3)
    return from_origin @ correction @ to_origin


def _scene_per_m(matrix: np.ndarray) -> float:
    linear = np.asarray(matrix, dtype=np.float64).reshape(4, 4)[:3, :3]
    det = abs(float(np.linalg.det(linear)))
    scale = det ** (1.0 / 3.0)
    return float(scale) if math.isfinite(scale) and scale > 1e-9 else 1.0


def _update_pose_payload(payload: dict[str, Any], *, matrix: np.ndarray, correction: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    scene_per_m = _scene_per_m(matrix)
    out["world_to_menon_scene_col_major"] = [float(x) for x in matrix.flatten(order="F")]
    out["matrix_row_major"] = [float(x) for x in matrix.reshape(-1)]
    out["scene_per_m"] = float(scene_per_m)
    out["s_obj_to_m"] = float(1.0 / scene_per_m) if scene_per_m > 1e-9 else 1.0
    method = str(out.get("method") or "virtual_twin_registration")
    if "visual_alignment_correction" not in method:
        method = f"{method}+visual_alignment_correction"
    out["method"] = method
    out["visual_alignment_correction"] = correction
    return out


def apply(args: argparse.Namespace) -> dict[str, Any]:
    store = VirtualTwinStore(Path(args.output_root) if args.output_root else None)
    source_revision = validate_revision_id(args.revision or store.latest_revision_id() or "")
    output_revision = validate_revision_id(args.output_revision or revision_id_from_clock("vt_living_room_visual_align"))
    revision_dir = _clone_revision(store, source_revision, output_revision)
    _update_revision_ids(revision_dir, output_revision, source_revision)

    manifest = _read_json(revision_dir / "manifest.json")
    metrics = _read_json(revision_dir / "metrics.json")
    tracking = _read_json(revision_dir / "tracking_alignment.json")
    pose = tracking.get("pose_correction")
    if not isinstance(pose, dict):
        raise VirtualTwinBuildError("tracking_alignment.json has no pose_correction")
    prior_matrix = _registration_matrix(pose)
    center = np.asarray([args.center_x, args.center_y, args.center_z], dtype=np.float64)
    translation = np.asarray([args.tx, args.ty, args.tz], dtype=np.float64)
    correction_matrix = _correction_matrix(
        center=center,
        translation=translation,
        yaw_deg=float(args.yaw_deg),
        pitch_deg=float(args.pitch_deg),
        roll_deg=float(args.roll_deg),
        scale=float(args.scale),
    )
    matrix = correction_matrix @ prior_matrix
    correction = {
        "status": "ok",
        "method": "visual_edge_alignment_with_depth_surface_regularization",
        "source_revision_id": source_revision,
        "generated_ts_us": int(time.time() * 1_000_000),
        "center_scene": [float(x) for x in center],
        "translation_scene": [float(x) for x in translation],
        "rotation_deg": {
            "yaw": float(args.yaw_deg),
            "pitch": float(args.pitch_deg),
            "roll": float(args.roll_deg),
        },
        "scale": float(args.scale),
        "correction_col_major": [float(x) for x in correction_matrix.flatten(order="F")],
    }

    tracking["revision_id"] = output_revision
    tracking["source_revision_id"] = source_revision
    tracking["pose_correction"] = _update_pose_payload(pose, matrix=matrix, correction=correction)
    if isinstance(tracking.get("depth_ray_correction_diagnostics"), dict):
        tracking["depth_ray_correction_diagnostics"]["visual_alignment_correction_status"] = "applied"
    _replace_json(revision_dir / "tracking_alignment.json", tracking)

    points_payload = np.load(revision_dir / "points.npz")
    points_world = np.asarray(points_payload["points"], dtype=np.float32).reshape((-1, 3))
    colors = np.asarray(points_payload["colors"], dtype=np.uint8).reshape((-1, 3))
    scene_points = transform_points(points_world, matrix).astype(np.float32)
    surfaces = parse_menon_obj_surfaces(_model_path(manifest, args.model_obj))
    scene_per_m = _scene_per_m(matrix)
    accepted_mask, leakage = _model_acceptance_mask(
        scene_points,
        surfaces,
        scene_margin=max(50.0, scene_per_m * 2.0),
    )
    points_npz = revision_dir / "points.npz"
    if points_npz.exists():
        points_npz.unlink()
    write_points_npz(
        points_npz,
        points_world,
        colors,
        scene_points=scene_points,
        accepted_scene_mask=accepted_mask.astype(np.uint8),
    )

    metrics["revision_id"] = output_revision
    metrics["source_revision_id"] = source_revision
    metrics["room_model_leakage_ratio"] = float(leakage["ratio"])
    metrics["room_model_leakage"] = leakage
    metrics.setdefault("registration", {})["scene_per_m"] = float(scene_per_m)
    metrics["registration"]["method"] = tracking["pose_correction"].get("method")
    metrics["registration"]["visual_alignment_correction"] = correction
    metrics.setdefault("gates", {})["room_model_leakage_pass"] = bool(
        float(leakage["ratio"]) <= float(args.max_room_model_leakage)
    )
    _replace_json(revision_dir / "metrics.json", metrics)

    manifest["revision_id"] = output_revision
    manifest["source_revision_id"] = source_revision
    manifest["visual_alignment_correction"] = correction
    _replace_json(revision_dir / "manifest.json", manifest)

    refresh_result = refresh_render_artifacts(
        Namespace(
            revision=output_revision,
            output_revision=None,
            in_place=True,
            update_latest=bool(args.update_latest),
            output_root=args.output_root,
            model_obj=args.model_obj,
            browser_point_budget=int(args.browser_point_budget),
            texture_tile_px=int(args.texture_tile_px),
            min_texture_coverage=float(args.min_texture_coverage),
            texture_exposure=float(args.texture_exposure),
            texture_gamma=float(args.texture_gamma),
            texture_contrast=float(args.texture_contrast),
        )
    )
    return {
        **refresh_result,
        "visual_alignment_correction": correction,
        "room_model_leakage_ratio": float(leakage["ratio"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a persisted visual-alignment correction to a virtual-twin revision.")
    parser.add_argument("--revision", default=None, help="Source revision id; defaults to latest.")
    parser.add_argument("--output-revision", default=None)
    parser.add_argument("--update-latest", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model-obj", type=Path, default=None)
    parser.add_argument("--center-x", type=float, required=True)
    parser.add_argument("--center-y", type=float, required=True)
    parser.add_argument("--center-z", type=float, required=True)
    parser.add_argument("--tx", type=float, required=True)
    parser.add_argument("--ty", type=float, required=True)
    parser.add_argument("--tz", type=float, required=True)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-room-model-leakage", type=float, default=0.03)
    parser.add_argument("--browser-point-budget", type=int, default=0)
    parser.add_argument("--texture-tile-px", type=int, default=96)
    parser.add_argument("--min-texture-coverage", type=float, default=0.02)
    parser.add_argument("--texture-exposure", type=float, default=2.2)
    parser.add_argument("--texture-gamma", type=float, default=0.65)
    parser.add_argument("--texture-contrast", type=float, default=1.08)
    return parser.parse_args()


def main() -> int:
    try:
        result = apply(parse_args())
    except Exception as exc:
        print(f"virtual-twin alignment correction failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
