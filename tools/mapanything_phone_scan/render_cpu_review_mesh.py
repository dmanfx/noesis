#!/usr/bin/env python3
"""Bounded direct-GLB review renders using an explicitly supplied CPU renderer.

The retained task renderer supplies rasterization. This wrapper verifies source
bindings, handles native vertex-colour visuals without silently whitening them,
and records the camera and serialized mesh used in every image.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from tools.mapanything_phone_scan.build_pcf_review_surface_mesh import _json, _sha256


def color_rgba(visual, count: int, *, encoding=None) -> np.ndarray:
    if getattr(visual, "kind", None) == "texture":
        # An image-less PBR material already has a baseColorFactor in the
        # rasterizer. White vertex colour preserves it without multiplying
        # the factor into itself or inventing a missing photo.
        attributes = getattr(visual, "vertex_attributes", {})
        if "color" not in attributes:
            return np.full((count, 4), 255, dtype=np.uint8)
        rgba = np.asarray(attributes["color"], dtype=np.uint8)
    else:
        color = visual if getattr(visual, "kind", None) == "vertex" else visual.to_color()
        rgba = np.asarray(color.vertex_colors, dtype=np.uint8)
    if rgba.shape == (count, 3):
        rgba = np.column_stack((rgba, np.full(count, 255, dtype=np.uint8)))
    if rgba.shape != (count, 4):
        raise ValueError("direct GLB consumer has invalid native vertex colours")
    if encoding == "gltf_linear_uint8":
        from tools.mapanything_phone_scan.texture_single_room_review_surface import _encoded_rgb
        rgba = rgba.copy()
        rgba[:, :3] = _encoded_rgb(rgba[:, :3].astype(float) / 255)
    elif encoding is not None:
        raise ValueError("unrecognized explicit vertex color encoding")
    return np.ascontiguousarray(rgba)


def build(args):
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("use a new render directory")
    if not (32 <= args.width <= 2048 and 32 <= args.height <= 2048
            and len(args.phone_view) <= 8):
        raise ValueError("render work exceeds bounded dimensions or camera count")
    manifest_path = args.mesh_manifest.resolve()
    manifest = _json(manifest_path)
    texture_manifest = getattr(args, "texture_manifest", None)
    if texture_manifest is None:
        glb_hash = manifest["output"]["sha256"]
    else:
        texture = _json(texture_manifest)
        if (texture.get("status") != "review_only" or texture.get("accepted_for_canonical_use") is not False
                or texture["source"]["mesh_manifest"]["sha256"] != _sha256(manifest_path)
                or Path(texture["source"]["mesh_manifest"]["path"]).resolve() != manifest_path
                or texture["source"]["mesh"]["sha256"] != manifest["output"]["sha256"]):
            raise ValueError("texture is not bound to the supplied measured/completed geometry manifest")
        glb_hash = texture["output"]["glb"]["sha256"]
    if (glb_hash != _sha256(args.glb) or manifest.get("status") != "review_only"
            or manifest.get("accepted_for_canonical_use") is not False
            or manifest.get("coordinate_frame") != "backend_world_m_stream_points"):
        raise ValueError("render producer binding does not match review-only backend-world GLB")
    spec = importlib.util.spec_from_file_location("explicit_review_cpu_renderer", args.renderer)
    renderer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = renderer
    spec.loader.exec_module(renderer)
    parts, consumer = renderer.load_parts(args.glb)
    if sum(len(part.faces) for part in parts) > 5_000_000:
        raise ValueError("mesh exceeds bounded render triangle work")
    scene = trimesh.load(args.glb, force="scene", process=False)
    corrected = []
    for part in parts:
        if not part.has_texture:
            geometry = scene.geometry[part.geometry_name]
            part.rgba = color_rgba(geometry.visual, len(part.vertices_world),
                                  encoding=geometry.metadata.get("vertex_color_encoding"))
            corrected.append(part.node_name)
    with np.load(args.reference_visibility, allow_pickle=False) as reference:
        cameras = [("fixed", np.asarray(reference["K"], dtype=float),
                    np.asarray(reference["camera_to_world"], dtype=float),
                    reference["depth"].shape[1], reference["depth"].shape[0])]
    input_manifest = _json(Path(manifest["source"]["surfels_manifest"]))
    owner = input_manifest["room_name"]
    rows = input_manifest["sources"][owner]["raw_views"]
    for index in args.phone_view:
        row = rows[index]
        path = Path(row["path"])
        if row.get("sha256") != _sha256(path):
            raise ValueError("phone render camera raw binding mismatch")
        with np.load(path, allow_pickle=False) as data:
            transform = np.asarray(row["effective_world_from_local_row_major"], dtype=float)
            cameras.append((f"phone_{index:04d}", np.asarray(data["intrinsics"], dtype=float),
                            transform @ data["camera_pose"], data["depth_z"].shape[1], data["depth_z"].shape[0]))
    output.mkdir(parents=True, exist_ok=False)
    raster = renderer.raster_library(args.rasterizer)
    metrics = {}
    for name, K, pose, width, height in cameras:
        # Model/source integer image coordinates are scaled consistently with
        # their published K. The CPU renderer samples output pixel centres.
        K = np.diag([args.width / width, args.height / height, 1.]) @ K
        buffers, values = renderer.render(parts, K, pose, width=args.width, height=args.height,
                                          raster=raster, flip_texture_v=False)
        for key in ("rgb", "flat", "smooth"):
            Image.fromarray(buffers[key]).save(output / f"{name}_{key}.png")
        Image.fromarray(renderer.wire_image(buffers["wire"])).save(output / f"{name}_wireframe.png")
        np.savez_compressed(output / f"{name}_framebuffer.npz", depth=buffers["depth"],
                            part=buffers["mesh"], face=buffers["face"])
        metrics[name] = values
        print(json.dumps({"camera": name, "coverage_fraction": values["coverage_fraction"],
                          "elapsed_s": values["elapsed_seconds"]}), flush=True)
    report = {"status": "offline_direct_glb_cpu_render_complete", "accepted_for_canonical_use": False,
              "source": {"glb": str(args.glb.resolve()), "glb_sha256": _sha256(args.glb),
                         "mesh_manifest": str(manifest_path), "mesh_manifest_sha256": _sha256(manifest_path),
                         "renderer": str(args.renderer.resolve()), "renderer_sha256": _sha256(args.renderer),
                         "rasterizer_sha256": _sha256(args.rasterizer),
                         "reference_camera_sha256": _sha256(args.reference_visibility)},
              "direct_consumer": consumer, "native_vertex_colour_nodes": corrected, "cameras": metrics,
              "texture_manifest_sha256": _sha256(texture_manifest) if texture_manifest is not None else None,
              "policy": {"image_inpainting": False, "geometry_changed": False, "texture_v_origin": "gltf_upper_left",
                         "camera_render_coverage_is_not_physical_accuracy": True}}
    (output / "render_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("glb", "mesh-manifest", "renderer", "rasterizer", "reference-visibility", "output-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--phone-view", type=int, action="append", default=[])
    parser.add_argument("--texture-manifest", type=Path)
    return parser


if __name__ == "__main__":
    build(_parser().parse_args())
