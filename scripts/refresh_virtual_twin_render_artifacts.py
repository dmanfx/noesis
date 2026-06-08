#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.calibration.manager import CalibrationSnapshot
from noesis.virtual_twin.artifacts import write_json, write_points_glb, write_textured_mesh_glb
from noesis.virtual_twin.builder import (
    VirtualTwinBuildError,
    VirtualTwinFrameInput,
    _assign_support_to_surfaces,
    _bake_model_surface_texture,
    _structural_visibility_surfaces,
    _surface_candidates_for_points,
    revision_id_from_clock,
)
from noesis.virtual_twin.menon_obj import parse_menon_obj_surfaces
from noesis.virtual_twin.store import VirtualTwinStore, validate_revision_id


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise VirtualTwinBuildError(f"expected JSON object at {path}")
    return payload


def _replace_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        path.unlink()
    write_json(path, payload)


def _model_path(manifest: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        path = Path(override).expanduser()
    else:
        raw = (
            (manifest.get("source_file_refs") or {}).get("menon_structural_obj")
            or ((manifest.get("model_fingerprints") or {}).get("menon_structural_obj") or {}).get("path")
        )
        path = Path(str(raw or "")).expanduser()
    if not str(path):
        raise VirtualTwinBuildError("Menon structural OBJ is required via --model-obj or manifest source_file_refs")
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise VirtualTwinBuildError(f"Menon structural OBJ does not exist: {path}")
    return path


def _snapshot(payload: dict[str, Any]) -> CalibrationSnapshot:
    return CalibrationSnapshot(
        camera_id=str(payload.get("camera_id") or ""),
        intrinsics=np.asarray(payload.get("intrinsics"), dtype=np.float64).reshape(3, 3),
        extrinsics_col_major=[float(x) for x in payload.get("extrinsics_col_major")],
        floor_y=float(payload.get("floor_y") or 0.0),
        image_size=(int(payload.get("image_size")[0]), int(payload.get("image_size")[1])),
        unit_scale=float(payload.get("unit_scale") or 1.0),
    )


def _load_frames(revision_dir: Path, manifest: dict[str, Any]) -> list[VirtualTwinFrameInput]:
    rows = ((manifest.get("source_file_refs") or {}).get("frames") or [])
    if not isinstance(rows, list) or not rows:
        raise VirtualTwinBuildError("manifest has no source_file_refs.frames entries")
    frames: list[VirtualTwinFrameInput] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        artifacts = row.get("revision_artifacts") or {}
        rgb_rel = artifacts.get("rgb")
        depth_rel = artifacts.get("mapanything_npz")
        calib_payload = artifacts.get("calibration")
        if not isinstance(rgb_rel, str) or not isinstance(depth_rel, str) or not isinstance(calib_payload, dict):
            raise VirtualTwinBuildError(f"frame row lacks RGB/depth/calibration artifacts: {row.get('frame_id')}")
        rgb_path = revision_dir / rgb_rel
        depth_path = revision_dir / depth_rel
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise VirtualTwinBuildError(f"failed to read RGB keyframe: {rgb_path}")
        depth_payload = np.load(depth_path)
        normals_camera = (
            np.asarray(depth_payload["normals_camera"], dtype=np.float32)
            if "normals_camera" in depth_payload.files
            else None
        )
        frames.append(
            VirtualTwinFrameInput(
                frame_id=str(row.get("frame_id") or rgb_path.stem),
                camera_id=str(row.get("camera_id") or manifest.get("camera") or ""),
                image_bgr=image,
                map_depth=np.asarray(depth_payload["depth"], dtype=np.float32),
                map_confidence=np.asarray(depth_payload["confidence"], dtype=np.float32),
                map_mask=np.asarray(depth_payload["mask"], dtype=bool),
                calibration=_snapshot(calib_payload),
                plane_candidates=(),
                source_ref=str(row.get("source_ref") or rgb_rel),
                map_normals_camera=normals_camera,
            )
        )
    if not frames:
        raise VirtualTwinBuildError("no usable frame artifacts were loaded")
    return frames


def _clone_revision(store: VirtualTwinStore, source_revision: str, output_revision: str) -> Path:
    source_dir = store.revision_dir(source_revision)
    output_dir = store.revision_dir(output_revision)
    if output_dir.exists():
        raise VirtualTwinBuildError(f"output revision already exists: {output_revision}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, output_dir, copy_function=os.link)
    return output_dir


def _update_revision_ids(revision_dir: Path, revision_id: str, source_revision_id: str | None) -> None:
    for name in ("manifest.json", "metrics.json", "tracking_alignment.json", "planes.json"):
        path = revision_dir / name
        if not path.exists():
            continue
        payload = _read_json(path)
        payload["revision_id"] = revision_id
        if source_revision_id and name in {"manifest.json", "metrics.json", "tracking_alignment.json"}:
            payload.setdefault("source_revision_id", source_revision_id)
        _replace_json(path, payload)


def refresh(args: argparse.Namespace) -> dict[str, Any]:
    store = VirtualTwinStore(Path(args.output_root) if args.output_root else None)
    source_revision = validate_revision_id(args.revision or store.latest_revision_id() or "")
    if args.in_place:
        output_revision = source_revision
        revision_dir = store.revision_dir(source_revision)
        source_revision_for_payload = None
    else:
        output_revision = validate_revision_id(args.output_revision or revision_id_from_clock("vt_living_room_render_refresh"))
        revision_dir = _clone_revision(store, source_revision, output_revision)
        _update_revision_ids(revision_dir, output_revision, source_revision)
        source_revision_for_payload = source_revision

    manifest = _read_json(revision_dir / "manifest.json")
    metrics = _read_json(revision_dir / "metrics.json")
    tracking = _read_json(revision_dir / "tracking_alignment.json")
    model_path = _model_path(manifest, args.model_obj)
    surfaces = parse_menon_obj_surfaces(model_path)
    frames = _load_frames(revision_dir, manifest)

    points_payload = np.load(revision_dir / "points.npz")
    scene_points = np.asarray(points_payload["scene_points"], dtype=np.float32).reshape((-1, 3))
    colors = np.asarray(points_payload["colors"], dtype=np.uint8).reshape((-1, 3))
    accepted_mask = np.asarray(points_payload["accepted_scene_mask"], dtype=np.uint8).astype(bool).reshape((-1,))
    if accepted_mask.shape[0] != scene_points.shape[0] or accepted_mask.shape[0] != colors.shape[0]:
        raise VirtualTwinBuildError("points.npz accepted_scene_mask does not match scene_points/colors")
    accepted_scene = scene_points[accepted_mask]
    accepted_colors = colors[accepted_mask]
    if accepted_scene.shape[0] == 0:
        raise VirtualTwinBuildError("points.npz has no accepted scene points")

    registration = tracking.get("pose_correction")
    if not isinstance(registration, dict):
        registration = (metrics.get("registration") or {}).get("pose_correction")
    if not isinstance(registration, dict):
        raise VirtualTwinBuildError("tracking_alignment.json has no pose_correction")
    scene_per_m = float(registration.get("scene_per_m") or (metrics.get("registration") or {}).get("scene_per_m") or 1.0)
    browser_point_budget = int(
        args.browser_point_budget
        or ((metrics.get("browser_render_budget") or {}).get("point_budget") or 150_000)
    )

    surface_candidates = _surface_candidates_for_points(accepted_scene, surfaces, scene_per_m=scene_per_m)
    mesh_vertices, mesh_indices, _mesh_colors, mesh_triangle_surface_indices, surface_metrics = _assign_support_to_surfaces(
        accepted_scene,
        accepted_colors,
        surface_candidates,
        scene_per_m=scene_per_m,
        browser_point_budget=browser_point_budget,
    )
    texcoords, atlas, texture_metrics = _bake_model_surface_texture(
        mesh_vertices,
        mesh_indices,
        mesh_triangle_surface_indices,
        surface_candidates,
        _structural_visibility_surfaces(surfaces),
        frames,
        registration,
        texture_tile_px=int(args.texture_tile_px),
        min_texture_coverage=float(args.min_texture_coverage),
        texture_exposure=float(args.texture_exposure),
        texture_gamma=float(args.texture_gamma),
        texture_contrast=float(args.texture_contrast),
    )

    point_stride = (
        int(np.ceil(accepted_scene.shape[0] / float(max(1, browser_point_budget))))
        if accepted_scene.shape[0] > browser_point_budget
        else 1
    )
    for rel_name in ("surfaces.glb", "points.glb"):
        target = revision_dir / rel_name
        if target.exists():
            target.unlink()
    write_textured_mesh_glb(revision_dir / "surfaces.glb", mesh_vertices, mesh_indices, texcoords, atlas)
    write_points_glb(revision_dir / "points.glb", accepted_scene[::point_stride], accepted_colors[::point_stride])

    browser_budget = metrics.setdefault("browser_render_budget", {})
    surface_metrics = {
        **surface_metrics,
        "artifact_type": "textured_model_surface_mesh",
        "texture": texture_metrics,
    }
    browser_budget.update(
        {
            "point_budget": int(browser_point_budget),
            "accepted_scene_point_count": int(accepted_scene.shape[0]),
            "served_glb_point_count": int(accepted_scene[::point_stride].shape[0]),
            "served_glb_vertex_count": int(mesh_vertices.shape[0]),
            "served_glb_triangle_count": int(mesh_indices.shape[0] // 3),
            "decimation_stride": int(surface_metrics["source_sample_stride"]),
            "artifact_type": "textured_model_surface_mesh",
            "texture": texture_metrics,
            "model_surface_projection": surface_metrics,
            "point_cloud": {
                "artifact_type": "scene_point_cloud",
                "coordinate_frame": "menon_scene_units",
                "source_point_count": int(accepted_scene.shape[0]),
                "served_glb_point_count": int(accepted_scene[::point_stride].shape[0]),
                "source_sample_stride": int(point_stride),
            },
        }
    )
    metrics["revision_id"] = output_revision
    if source_revision_for_payload:
        metrics["source_revision_id"] = source_revision_for_payload
    metrics["render_artifact_refresh"] = {
        "generated_ts_us": int(time.time() * 1_000_000),
        "source_revision_id": source_revision_for_payload or output_revision,
        "method": "structural_visibility_texture_bake_with_depth_occlusion_reject",
    }
    metrics.setdefault("gates", {})["rgb_surface_texture_coverage_pass"] = bool(
        float(texture_metrics.get("texture_coverage_ratio") or 0.0) >= float(args.min_texture_coverage)
    )

    manifest["revision_id"] = output_revision
    manifest["created_ts_us"] = int(time.time() * 1_000_000)
    if source_revision_for_payload:
        manifest["source_revision_id"] = source_revision_for_payload
    manifest.setdefault("artifacts", {}).update(
        {
            "surfaces_glb": "surfaces.glb",
            "points_glb": "points.glb",
            "points_ply": "points.ply",
            "points_npz": "points.npz",
        }
    )
    manifest.setdefault("coordinate_frames", {}).update(
        {
            "surfaces_glb": "menon_scene_units_textured_model_surface_mesh",
            "points_glb": "menon_scene_units_decimated_point_cloud",
        }
    )
    manifest["render_artifact_refresh"] = metrics["render_artifact_refresh"]
    _replace_json(revision_dir / "manifest.json", manifest)
    _replace_json(revision_dir / "metrics.json", metrics)

    if args.update_latest:
        store.write_latest_revision_id(output_revision)
    return {
        "revision_id": output_revision,
        "source_revision_id": source_revision,
        "updated_latest": bool(args.update_latest),
        "surface_texture_coverage": float(texture_metrics.get("texture_coverage_ratio") or 0.0),
        "served_glb_triangle_count": int(mesh_indices.shape[0] // 3),
        "served_glb_point_count": int(accepted_scene[::point_stride].shape[0]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh Menon render artifacts for an existing virtual-twin revision.")
    parser.add_argument("--revision", default=None, help="Source revision id; defaults to latest.")
    parser.add_argument("--output-revision", default=None, help="New revision id. Defaults to a generated render-refresh id.")
    parser.add_argument("--in-place", action="store_true", help="Mutate the source revision instead of cloning it.")
    parser.add_argument("--update-latest", action="store_true", help="Point data/virtual_twin/latest at the output revision.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model-obj", type=Path, default=None)
    parser.add_argument("--browser-point-budget", type=int, default=0)
    parser.add_argument("--texture-tile-px", type=int, default=96)
    parser.add_argument("--min-texture-coverage", type=float, default=0.02)
    parser.add_argument("--texture-exposure", type=float, default=2.2)
    parser.add_argument("--texture-gamma", type=float, default=0.65)
    parser.add_argument("--texture-contrast", type=float, default=1.08)
    return parser.parse_args()


def main() -> int:
    try:
        result = refresh(parse_args())
    except Exception as exc:
        print(f"virtual-twin render artifact refresh failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
