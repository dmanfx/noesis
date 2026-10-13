#!/usr/bin/env python3
"""Bake bounded real-video samples into continuous review-mesh vertex color.

Geometry, provider depth/confidence, poses and source images are unchanged.
This explicit alternative uses robust multiview color blending. Vertices with
no eligible photo retain separately labelled Poisson-interpolated input color.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import trimesh

from tools.mapanything_phone_scan import texture_single_room_review_surface as texture
from tools.mapanything_phone_scan.build_closed_room_review_surface import edge_topology


def unlit_vertex_color(tree):
    """Make COLOR_0 usable without a consumer-specific lighting override."""
    primitives = [primitive for mesh in tree["meshes"] for primitive in mesh["primitives"]]
    if not primitives or any("COLOR_0" not in primitive["attributes"] for primitive in primitives):
        raise ValueError("unlit vertex export requires explicit COLOR_0 on every primitive")
    material = len(tree.setdefault("materials", []))
    tree["materials"].append({
        "name": "verified_video_vertex_color",
        "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 1},
        "doubleSided": True, "extensions": {"KHR_materials_unlit": {}},
    })
    for primitive in primitives:
        primitive["material"] = material
    tree["extensionsUsed"] = sorted(set(tree.get("extensionsUsed", [])) | {"KHR_materials_unlit"})


def finalize_bound_bake(source_dir: Path, output_dir: Path):
    """Reuse an exact verified bake; change only its glTF material binding."""
    if output_dir.exists():
        raise ValueError("use a new final color mesh directory")
    source_manifest = source_dir / "surface_mesh_manifest.json"
    source = texture._json(source_manifest)
    path = source_dir / "surface_mesh.glb"
    bake = source["color_bake"]
    if (source.get("status") != "review_only" or source.get("accepted_for_canonical_use") is not False
            or source["output"]["sha256"] != texture._sha(path)
            or bake.get("vertex_color_encoding") != "gltf_linear_uint8"):
        raise ValueError("material finalization requires an exact bound linear-color review bake")
    sidecar = Path(bake["color_sources"]["path"])
    texture._bound_file(bake["color_sources"], sidecar, "color sources")
    scene = trimesh.load(path, force="scene", process=False)
    if len(scene.graph.nodes_geometry) != 1:
        raise ValueError("final color export requires one bound source geometry")
    transform, name = scene.graph[scene.graph.nodes_geometry[0]]
    if not np.array_equal(transform, np.eye(4)):
        raise ValueError("final color export has an unbound geometry transform")
    mesh = scene.geometry[name]
    def bound_colors(value):
        # Trimesh loads COLOR_0 + a material as TextureVisuals, including an
        # image-less unlit material. Read the explicit serialized colour
        # attribute instead of converting visuals or inventing white colors.
        if getattr(value.visual, "kind", None) == "vertex":
            colors = np.asarray(value.visual.vertex_colors)
        else:
            colors = np.asarray(getattr(value.visual, "vertex_attributes", {}).get("color"))
        if colors.dtype != np.uint8 or colors.shape != (len(value.vertices), 4):
            raise ValueError("final color export requires explicit bound uint8 RGBA")
        return colors.copy()
    original_colors = bound_colors(mesh)
    output_dir.mkdir(parents=True, exist_ok=False)
    target = output_dir / "surface_mesh.glb"
    target.write_bytes(trimesh.exchange.gltf.export_glb(scene, include_normals=True, tree_postprocessor=unlit_vertex_color))
    direct_scene = trimesh.load(target, force="scene", process=False)
    direct = direct_scene.geometry[name]
    direct_colors = bound_colors(direct)
    if (not np.array_equal(direct.vertices, mesh.vertices) or not np.array_equal(direct.faces, mesh.faces)
            or not np.array_equal(direct_colors, original_colors)):
        raise ValueError("material finalization changed bound vertices, faces or linear colors")
    copied_sidecar = output_dir / "vertex_video_color_sources.npz"
    shutil.copyfile(sidecar, copied_sidecar)
    report = copy.deepcopy(source)
    report["source"]["parent_color_bake_manifest"] = texture._record(source_manifest)
    report["source"]["parent_color_bake_mesh"] = texture._record(path)
    report["color_bake"].update(color_sources=texture._record(copied_sidecar),
        gltf_unlit_material_bound=True, serialized_positions_faces_and_linear_colors_unchanged=True)
    report["output"].update(glb=str(target), sha256=texture._sha(target), size_bytes=target.stat().st_size)
    (output_dir / "surface_mesh_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": report["output"], "material": "bound_unlit_vertex_color"}, indent=2), flush=True)
    return report


def robust_video_colors(samples, weights, fallback):
    if (samples.shape != (*weights.shape, 3) or fallback.shape != (len(samples), 3)
            or not np.isfinite(samples).all() or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError("malformed bounded video color samples")
    valid = weights > 0
    available = valid.any(1)
    output = np.asarray(fallback, dtype=np.uint8).copy()
    rows = np.flatnonzero(available)
    for start in range(0, len(rows), 8192):
        part = rows[start:start + 8192]
        linear = texture._linear_rgb(samples[part])
        order = np.argsort(np.where(valid[part, :, None], linear, np.inf), axis=1)
        sorted_color = np.take_along_axis(linear, order, axis=1)
        rank = np.maximum(0, (valid[part].sum(1) - 1) // 2)
        median = sorted_color[np.arange(len(part)), rank]
        delta = np.mean(np.abs(linear - median[:, None]), axis=2)
        # Smoothly downweight exposure/parallax outliers; do not invent pixels
        # or force colors across geometry. At least the measured medoid remains.
        robust = weights[part] / (1 + (delta / .06) ** 2)
        combined = np.sum(linear * robust[:, :, None], axis=1) / robust.sum(1)[:, None]
        output[part] = texture._encoded_rgb(combined)
    return output, valid.sum(1).astype(np.uint8)


def build(args):
    if args.output_dir.exists() or not 1 <= args.color_top_k <= 8:
        raise ValueError("use a new bounded continuous color directory")
    if args.surface_visibility_dir is not None:
        raise ValueError("continuous video color uses the original provider depth gate, not same-mesh visibility")
    manifest_path = args.mesh_dir / "surface_mesh_manifest.json"
    manifest = texture._json(manifest_path)
    mesh_path = args.mesh_dir / "surface_mesh.glb"
    if (manifest.get("status") != "review_only" or manifest.get("accepted_for_canonical_use") is not False
            or manifest["output"]["sha256"] != texture._sha(mesh_path)):
        raise ValueError("continuous bake requires a bound review-only surface")
    scene = trimesh.load(mesh_path, force="scene", process=False)
    if len(scene.graph.nodes_geometry) != 1:
        raise ValueError("continuous bake requires one exact source surface")
    transform, name = scene.graph[scene.graph.nodes_geometry[0]]
    if not np.allclose(transform, np.eye(4)):
        raise ValueError("source has an unbound node transform")
    mesh = scene.geometry[name]
    max_faces, max_vertices = texture.mesh_work_bounds(args)
    if len(mesh.vertices) > max_vertices or len(mesh.faces) > max_faces:
        raise ValueError("continuous bake exceeds finite vertex/face bounds")
    support = texture.completed_geometry_support(manifest, str(name), len(mesh.faces))
    views, provenance = texture.load_bound_views(args)
    normalization = texture._json(args.photometric_manifest)
    texture._bound_file(normalization["source"]["mesh"], mesh_path, "color gain geometry")
    if len(normalization["views"]) != len(views):
        raise ValueError("color gain source cohort changed")
    gains = normalization["photometric_normalization"].get("applied_gains", {})
    for old, view in zip(normalization["views"], views, strict=True):
        for field in ("frame_id", "source_frame_index", "prepared_photo", "texture_photo", "conditioned_provider_raw",
                      "consensus_raw", "provider_intrinsics", "common_intrinsics", "camera_to_world", "model_to_photo_row_major"):
            if old.get(field) != view.provenance.get(field):
                raise ValueError("color gain exact frame/ray/photo binding changed")
    count = len(mesh.vertices)
    samples = np.zeros((count, args.color_top_k, 3), dtype=np.uint8)
    weights = np.zeros((count, args.color_top_k), dtype=np.float32)
    source_ids = np.full(weights.shape, -1, dtype=np.int16)
    normal = mesh.vertex_normals
    for view in views:
        if view.model_to_photo is None:
            continue
        gain = np.asarray(gains.get(str(view.index), [1., 1., 1.]), dtype=float)
        if gain.shape != (3,) or not np.isfinite(gain).all() or np.any((gain < .67) | (gain > 1.5)):
            raise ValueError("color gain exceeds unchanged fitted bounds")
        with Image.open(view.photo_path) as photo:
            rgb = np.asarray(photo.convert("RGB").resize(view.photo_size, Image.Resampling.LANCZOS))
        camera_center = np.linalg.inv(view.from_world)[:3, 3]
        for start in range(0, count, 8192):
            rows = np.arange(start, min(count, start + 8192))
            valid, quality = texture.depth_visible(mesh.vertices[rows], view)
            toward = camera_center - mesh.vertices[rows]
            distance = np.linalg.norm(toward, axis=1)
            cosine = np.abs(np.einsum("ij,ij->i", normal[rows], toward)) / np.maximum(distance, 1e-6)
            score = quality * view.quality * np.sqrt(np.maximum(cosine, .1)) / np.maximum(distance, .4)
            score[~valid] = 0
            worst = weights[rows].argmin(1)
            keep = score > weights[rows, worst]
            chosen, slots = rows[keep], worst[keep]
            if not len(chosen):
                continue
            _, _, uv = texture.project_photo(mesh.vertices[chosen], view)
            integer = np.rint(uv).astype(int)
            values = rgb[integer[:, 1], integer[:, 0]]
            values = texture._encoded_rgb(texture._linear_rgb(values) * gain)
            samples[chosen, slots] = values
            weights[chosen, slots] = score[keep]
            source_ids[chosen, slots] = view.index
        print(f"continuous video color {view.index + 1}/{len(views)}", flush=True)
    colors, sample_count = robust_video_colors(samples, weights, mesh.visual.vertex_colors[:, :3])
    # glTF COLOR_0 is linear light (unlike its sRGB photo textures). Preserve
    # that distinction in both the serialized artifact and CPU review consumer.
    linear_colors = np.rint(texture._linear_rgb(colors) * 255).clip(0, 255).astype(np.uint8)
    result = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), vertex_normals=mesh.vertex_normals,
        vertex_colors=np.column_stack((linear_colors, np.full(count, 255, dtype=np.uint8))), process=False,
        metadata={"review_only": True, "color_method": "robust_real_video_vertex_blend_with_interpolated_fallback",
                  "geometry_support_sidecar_preserved": True, "vertex_color_encoding": "gltf_linear_uint8"})
    output_scene = trimesh.Scene()
    output_scene.add_geometry(result, geom_name=name, node_name=name)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / "surface_mesh.glb"
    path.write_bytes(trimesh.exchange.gltf.export_glb(output_scene, include_normals=True, tree_postprocessor=unlit_vertex_color))
    sidecar = args.output_dir / "vertex_video_color_sources.npz"
    np.savez_compressed(sidecar, source_view_index=source_ids, source_weight=weights, sample_count=sample_count,
                        face_support_class=support, geometry_name=np.asarray(name))
    direct = trimesh.load(path, force="mesh", process=False)
    if not np.array_equal(direct.faces, mesh.faces) or not np.array_equal(direct.vertices, mesh.vertices):
        raise ValueError("continuous color export changed source geometry or local-fan identities")
    report = {**manifest, "method": "continuous_robust_verified_video_vertex_color_on_unchanged_closed_mesh",
        "source": {**manifest["source"], "color_source_surface": texture._record(mesh_path), **provenance,
                   "photometric_manifest": texture._record(args.photometric_manifest)},
        "color_bake": {"top_k": args.color_top_k, "cross_view_colour_blending": True, "image_inpainting": False,
            "work_bounds": {"max_faces": max_faces, "max_vertices": max_vertices},
            "vertex_color_encoding": "gltf_linear_uint8",
            "geometry_changed": False, "photo_visibility": "original_provider_depth_gate",
            "direct_photo_sample_vertex_count": int(np.count_nonzero(sample_count)),
            "multi_photo_sample_vertex_count": int(np.count_nonzero(sample_count >= 2)),
            "interpolated_input_color_vertex_count": int(np.count_nonzero(sample_count == 0)),
            "unobserved_color_policy": "retain_explicitly_interpolated_source_vertex_color_not_measured_photo_color",
            "color_sources": texture._record(sidecar)},
        "output": {"glb": str(path), "sha256": texture._sha(path), "size_bytes": path.stat().st_size,
                   "vertex_count": len(direct.vertices), "triangle_count": len(direct.faces)},
        "direct_color_consumer_topology": edge_topology(direct.vertices, direct.faces)}
    (args.output_dir / "surface_mesh_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": report["output"], "color_bake": report["color_bake"]}, indent=2), flush=True)
    return report


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "finalize":
        parser = argparse.ArgumentParser(description="Finalize an existing exact video color bake without resampling")
        parser.add_argument("--source-dir", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        finalize_bound_bake(args.source_dir, args.output_dir)
        return
    parser = texture._parser()
    parser.set_defaults(max_faces=3_000_000, max_vertices=2_000_000)
    parser.description = __doc__
    parser.add_argument("--photometric-manifest", type=Path, required=True)
    parser.add_argument("--color-top-k", type=int, default=4)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
