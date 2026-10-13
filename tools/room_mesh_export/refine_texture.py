#!/usr/bin/env python3
"""Compare texture-only derivatives of an exact existing browser geometry.

No simplification, inference, reconstruction, scene publication or source edit.
The existing saved geometry is verified against its bound GLB before reuse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import trimesh

from tools.room_mesh_export.build_browser_mesh import (
    bake, export_glb, normals, ray_scene, read_source, record, unwrap,
)


def geometry_digest(vertices, faces):
    """Exact oriented world-space triangle digest, independent of UV seams."""
    points, inverse = np.unique(np.asarray(vertices, dtype="<f4")[faces.reshape(-1)],
                                axis=0, return_inverse=True)
    rows = inverse.reshape(-1, 3)
    rows = np.take_along_axis(rows, (np.argmin(rows, axis=1)[:, None] + np.arange(3)) % 3, axis=1)
    rows = rows[np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))]
    digest = hashlib.sha256(points.astype("<f4").tobytes())
    digest.update(rows.astype("<u4").tobytes())
    return digest.hexdigest()


def load_geometry(directory):
    manifest_path = directory / "mesh_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = directory / "room.glb"
    actual = record(path)
    if (actual["sha256"] != manifest["output"]["sha256"]
            or actual["size_bytes"] != manifest["output"]["size_bytes"]):
        raise ValueError("baseline GLB binding mismatch")
    scene = trimesh.load(path, force="scene", process=False)
    if len(scene.graph.nodes_geometry) != 1:
        raise ValueError("baseline must contain one bound geometry")
    transform, name = scene.graph[scene.graph.nodes_geometry[0]]
    if not np.array_equal(transform, np.eye(4)):
        raise ValueError("baseline has an unexpected world transform")
    mesh = scene.geometry[name]
    with np.load(directory / "simplified_geometry.npz", allow_pickle=False) as raw:
        vertices, faces = raw["vertices"].copy(), raw["faces"].copy()
    digest = geometry_digest(vertices, faces)
    if (len(faces) != manifest["output"]["triangle_count"]
            or digest != geometry_digest(mesh.vertices, mesh.faces)):
        raise ValueError("saved simplified geometry differs from the bound baseline")
    return vertices, faces, manifest, digest, record(manifest_path)


def build(args):
    started = time.monotonic()
    if args.output_dir.exists() or args.texture_edge not in (2048, 4096):
        raise ValueError("use a new output directory and a bounded texture edge")
    vertices, faces, base, geometry_sha, base_record = load_geometry(args.base_dir)
    source_path = Path(base["source"]["mesh"]["path"])
    source_manifest_path = Path(base["source"]["manifest"]["path"])
    if record(source_manifest_path)["sha256"] != base["source"]["manifest"]["sha256"]:
        raise ValueError("dense source manifest binding changed")
    source_manifest = json.loads(source_manifest_path.read_text())
    source_vertices, source_faces, source_colours = read_source(source_path, source_manifest,
        max_faces=base.get("source_work_bounds", {}).get("max_faces", 3_000_000),
        max_vertices=base.get("source_work_bounds", {}).get("max_vertices", 2_000_000))
    source_scene = ray_scene(source_vertices, source_faces, args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / "simplified_geometry.npz", vertices=vertices, faces=faces)
    vertex_normals = normals(vertices, faces)
    if args.layout_from:
        layout_manifest = json.loads((args.layout_from / "mesh_manifest.json").read_text())
        if layout_manifest["texture_refinement"]["world_geometry_sha256"] != geometry_sha:
            raise ValueError("reused layout belongs to different world geometry")
        with np.load(args.layout_from / "unwrapped_geometry.npz", allow_pickle=False) as raw:
            vertices, faces, uvs, vertex_normals = (raw[key].copy() for key in ("vertices", "faces", "uvs", "normals"))
        uv_report = layout_manifest["uv_atlas"]
    else:
        vertices, faces, uvs, mapping, uv_report = unwrap(vertices, faces, args.texture_edge,
            padding_pixels=args.atlas_padding, chart_iterations=getattr(args, "chart_iterations", 1),
            chart_area=getattr(args, "chart_area", .75))
        vertex_normals = vertex_normals[mapping]
    if geometry_digest(vertices, faces) != geometry_sha:
        raise ValueError("texture layout changed oriented world geometry")
    np.savez_compressed(args.output_dir / "unwrapped_geometry.npz", vertices=vertices,
                        faces=faces, uvs=uvs, normals=vertex_normals)
    (args.output_dir / "uv_layout.json").write_text(json.dumps(uv_report, indent=2) + "\n")
    if args.mode == "vertex":
        image, bake_report = bake(source_scene, source_faces, source_colours, vertices, faces, uvs,
                                  args.texture_edge, args.threads, padding_pixels=args.bake_padding)
    else:
        from tools.room_mesh_export.photo_texture import bake_photos
        image, bake_report = bake_photos(source_scene, source_faces, source_colours, source_manifest,
            vertices, faces, uvs, args.texture_edge, args.threads, args.bake_padding, mode=args.mode)
    image.save(args.output_dir / "colour_atlas.png")
    metadata = {key: base[key] for key in ("review_only", "accepted_for_canonical_use", "coordinate_frame",
                 "source_sha256", "inherited_source_inferred_area_fraction", "source_scan_id")}
    metadata.update(geometry_is_simplified_presentation=True, texture_only_derivative=True,
                    world_geometry_sha256=geometry_sha)
    export_glb(args.output_dir / "room.glb", vertices, faces, uvs, vertex_normals, image, metadata)
    report = {**base, **metadata, "source": base["source"],
        "texture_refinement": {"baseline_manifest": base_record, "world_geometry_sha256": geometry_sha,
            "mode": args.mode, "geometry_changed": False, "new_depth_inference": False,
            "atlas_padding_pixels": uv_report["padding_pixels"], "outside_chart_bake_padding_pixels": args.bake_padding},
        "output": {**record(args.output_dir / "room.glb"), "triangle_count": len(faces),
            "vertex_count": len(vertices), "material_count": 1, "texture_count": 1, "draw_calls": 1,
            "decoded_texture_bytes_rgba_with_mips": int(args.texture_edge**2*4*4/3)},
        "uv_atlas": uv_report, "texture_bake": bake_report,
        "elapsed_seconds": time.monotonic() - started}
    report["parameters"] = {**base["parameters"], "texture_edge": args.texture_edge}
    (args.output_dir / "mesh_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": report["output"], "texture_coverage": bake_report["fraction"],
                      "geometry_sha256": geometry_sha, "elapsed_seconds": report["elapsed_seconds"]}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--texture-edge", type=int, default=4096)
    parser.add_argument("--atlas-padding", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--bake-padding", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--layout-from", type=Path)
    parser.add_argument("--mode", choices=("vertex", "photo-best", "photo-blend"), default="vertex")
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--chart-iterations", type=int, choices=(1, 2), default=1)
    parser.add_argument("--chart-area", type=float, default=.75)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
