#!/usr/bin/env python3
"""Bound exact phone cameras to CPU z-buffers of a review surface.

This is presentation visibility, not agreement with provider depth or a new
geometry measurement. It neither fills images nor changes reconstruction.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from tools.mapanything_phone_scan import texture_single_room_review_surface as texture


SCHEMA = "noesis.pcf.mesh_texture_visibility.v1"


def camera_binding(view):
    return {key: view.provenance[key] for key in (
        "frame_id", "source_frame_index", "consensus_raw", "conditioned_provider_raw",
        "common_intrinsics", "provider_intrinsics", "camera_to_world", "photo_size",
        "model_to_photo_row_major")}


def camera_signature(view):
    return hashlib.sha256(json.dumps(camera_binding(view), sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_bound_visibility(directory, views, mesh_sha256):
    directory = Path(directory).resolve()
    manifest = texture._json(directory / "mesh_texture_visibility_manifest.json")
    if (manifest.get("schema") != SCHEMA or manifest.get("mesh_sha256") != mesh_sha256
            or manifest.get("status") != "complete_review_surface_zbuffers"
            or len(manifest["views"]) != len(views)):
        raise texture.TextureSurfaceError("surface visibility mesh/view binding mismatch")
    for row, view in zip(manifest["views"], views, strict=True):
        if row.get("index") != view.index or row.get("camera_signature") != camera_signature(view):
            raise texture.TextureSurfaceError("surface visibility camera/photo mapping changed")
        if view.model_to_photo is None:
            if row.get("status") != "texture_only_excluded":
                raise texture.TextureSurfaceError("surface visibility exclusion changed")
            continue
        path = (directory / row["depth"]["path"]).resolve()
        if not path.is_relative_to(directory) or row["depth"]["sha256"] != texture._sha(path):
            raise texture.TextureSurfaceError("surface visibility depth hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            depth = data["depth"].copy()
        if (depth.ndim != 2 or depth.shape != (row["height"], row["width"])
                or max(depth.shape) > 1280 or np.isnan(depth).any() or np.any(depth <= 0)):
            raise texture.TextureSurfaceError("surface visibility depth is malformed or unbounded")
        view.surface_depth = depth
        view.provenance["surface_visibility_zbuffer"] = {"sha256": row["depth"]["sha256"],
            "width": row["width"], "height": row["height"], "mesh_sha256": mesh_sha256}
    return {"manifest": texture._record(directory / "mesh_texture_visibility_manifest.json"),
            "evidence_type": "self_visibility_of_reconstructed_surface_not_provider_depth_agreement",
            "geometry_support_classes_changed": False}


def build(args):
    if args.output_dir.exists() or not 320 <= args.visibility_edge_px <= 1280:
        raise texture.TextureSurfaceError("use a new bounded visibility directory")
    mesh_path = args.mesh_dir / "surface_mesh.glb"
    max_faces, max_vertices = texture.mesh_work_bounds(args)
    mesh_manifest = texture._json(args.mesh_dir / "surface_mesh_manifest.json")
    if (mesh_manifest.get("status") != "review_only" or mesh_manifest.get("accepted_for_canonical_use") is not False
            or mesh_manifest["output"]["sha256"] != texture._sha(mesh_path)
            or mesh_manifest["output"]["triangle_count"] > max_faces
            or mesh_manifest["output"]["vertex_count"] > max_vertices):
        raise texture.TextureSurfaceError("surface visibility requires a bound review-only mesh")
    views, source = texture.load_bound_views(args)
    spec = importlib.util.spec_from_file_location("explicit_surface_visibility_renderer", args.renderer)
    renderer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = renderer
    spec.loader.exec_module(renderer)
    parts, consumer = renderer.load_parts(mesh_path)
    if sum(len(part.faces) for part in parts) != mesh_manifest["output"]["triangle_count"]:
        raise texture.TextureSurfaceError("visibility renderer lost mesh triangles")
    raster = renderer.raster_library(args.rasterizer)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for view in views:
        row = {"index": view.index, "camera_signature": camera_signature(view), "camera_binding": camera_binding(view)}
        if view.model_to_photo is None:
            rows.append({**row, "status": "texture_only_excluded"})
            continue
        width, height = texture._size_at_edge(view.photo_size, args.visibility_edge_px)
        affine = np.vstack((view.model_to_photo, [0., 0., 1.]))
        K = np.diag([width / view.photo_size[0], height / view.photo_size[1], 1.]) @ affine @ view.model_intrinsics
        if not np.allclose(K[[0, 1], [1, 0]], 0, atol=1e-7):
            raise texture.TextureSurfaceError("explicit CPU rasterizer does not support a skewed photo map")
        buffers, metrics = renderer.render(parts, K, np.linalg.inv(view.from_world), width=width, height=height,
                                           raster=raster, flip_texture_v=False)
        path = args.output_dir / f"view_{view.index:04d}.npz"
        np.savez_compressed(path, depth=np.asarray(buffers["depth"], dtype=np.float32))
        rows.append({**row, "status": "rendered", "width": width, "height": height,
                     "depth": {"path": path.name, "sha256": texture._sha(path)},
                     "coverage_fraction": metrics["coverage_fraction"]})
        print(f"surface visibility camera {view.index + 1}/{len(views)}", flush=True)
    report = {"schema": SCHEMA, "status": "complete_review_surface_zbuffers", "mesh_sha256": texture._sha(mesh_path),
              "source": source, "renderer": texture._record(args.renderer), "rasterizer": texture._record(args.rasterizer),
              "consumer": consumer, "views": rows, "work_bounds": {"max_faces": max_faces, "max_vertices": max_vertices},
              "policy": {"geometry_changed": False, "provider_depth_agreement": False,
              "visibility_is_not_independent_physical_accuracy": True, "image_inpainting": False}}
    path = args.output_dir / "mesh_texture_visibility_manifest.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = texture._parser()
    parser.set_defaults(max_faces=3_000_000, max_vertices=2_000_000)
    parser.description = __doc__
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--rasterizer", type=Path, required=True)
    parser.add_argument("--visibility-edge-px", type=int, default=960)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
