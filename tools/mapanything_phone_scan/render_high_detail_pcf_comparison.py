#!/usr/bin/env python3
"""CPU rasterize matched phone-view samples of low and high PCF review meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from tools.mapanything_phone_scan.build_high_detail_pcf_mesh import (
    _manifest_source_for_owner,
    checked_matrix,
    read_json,
    world_from_local,
)


def raster_mesh(
    mesh: trimesh.Trimesh, image: np.ndarray, z_buffer: np.ndarray,
    camera_from_assembly: np.ndarray, k: np.ndarray,
) -> dict[str, int]:
    height, width = z_buffer.shape
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    camera = vertices @ camera_from_assembly[:3, :3].T + camera_from_assembly[:3, 3]
    z = camera[:, 2]
    safe_z = np.where(z > 1e-4, z, 1.0)
    projected = np.column_stack((
        k[0, 0] * camera[:, 0] / safe_z + k[0, 2],
        k[1, 1] * camera[:, 1] / safe_z + k[1, 2],
    ))
    valid = (z[faces] > 0.08).all(1) & (z[faces] < 12.0).all(1)
    bounds = projected[faces]
    valid &= (bounds[:, :, 0].max(1) >= 0) & (bounds[:, :, 0].min(1) < width)
    valid &= (bounds[:, :, 1].max(1) >= 0) & (bounds[:, :, 1].min(1) < height)
    indices = np.flatnonzero(valid)
    # Painter ordering reduces avoidable z-buffer updates without changing the
    # direct depth test. Rasterization remains CPU-only and view bounded.
    indices = indices[np.argsort(z[faces[indices]].mean(1))[::-1]]
    texture = mesh.visual.kind == "texture"
    if texture:
        uv = np.asarray(mesh.visual.uv, dtype=np.float64)
        tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"), dtype=np.uint8)
        tex_height, tex_width = tex.shape[:2]
    else:
        rgba = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)
        if len(rgba) != len(vertices):
            raise ValueError("vertex-colored mesh lacks a complete RGB source")
    painted = 0
    skipped_huge = 0
    for index in indices:
        face = faces[index]
        xy = projected[face]
        x0 = max(0, int(np.floor(xy[:, 0].min())))
        x1 = min(width - 1, int(np.ceil(xy[:, 0].max())))
        y0 = max(0, int(np.floor(xy[:, 1].min())))
        y1 = min(height - 1, int(np.ceil(xy[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 30000:
            skipped_huge += 1
            continue
        a, b, c = xy
        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denom) < 1e-5:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        pixel_x, pixel_y = xs + 0.5, ys + 0.5
        w0 = ((b[1] - c[1]) * (pixel_x - c[0]) + (c[0] - b[0]) * (pixel_y - c[1])) / denom
        w1 = ((c[1] - a[1]) * (pixel_x - c[0]) + (a[0] - c[0]) * (pixel_y - c[1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not np.any(inside):
            continue
        inv_z = w0 / z[face[0]] + w1 / z[face[1]] + w2 / z[face[2]]
        depth = 1.0 / np.maximum(inv_z, 1e-6)
        patch_z = z_buffer[y0:y1 + 1, x0:x1 + 1]
        draw = inside & (depth < patch_z)
        if not np.any(draw):
            continue
        perspective = np.stack((w0 / z[face[0]], w1 / z[face[1]], w2 / z[face[2]]), axis=2)
        perspective /= np.maximum(inv_z[:, :, None], 1e-8)
        if texture:
            uv_face = uv[face]
            pixel_uv = np.einsum("hwc,cd->hwd", perspective, uv_face)
            pixel_uv[~inside] = 0.0
            tx = np.clip(np.rint(pixel_uv[:, :, 0] * (tex_width - 1)).astype(np.int32), 0, tex_width - 1)
            ty = np.clip(np.rint((1.0 - pixel_uv[:, :, 1]) * (tex_height - 1)).astype(np.int32), 0, tex_height - 1)
            color = tex[ty, tx]
        else:
            color = np.uint8(np.clip(np.rint(np.einsum("hwc,cd->hwd", perspective, rgba[face, :3])), 0, 255))
        patch_image = image[y0:y1 + 1, x0:x1 + 1]
        patch_image[draw] = color[draw]
        patch_z[draw] = depth[draw]
        painted += int(np.count_nonzero(draw))
    return {"candidate_triangle_count": int(len(indices)), "painted_pixel_count": painted, "large_triangle_skipped_count": skipped_huge}


def render(scene: trimesh.Scene, owner: int, camera_from_assembly: np.ndarray, k: np.ndarray, width: int, height: int) -> tuple[Image.Image, dict]:
    image = np.full((height, width, 3), [24, 26, 30], dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)
    room = {1: "family", 2: "kitchen", 3: "living"}[owner]
    metrics = {}
    # Only one owner is rendered; every source geometry is already in the
    # exact recorded assembly gauge. High fallback and photo pieces share z.
    for name, mesh in sorted(scene.geometry.items(), key=lambda row: "photo" not in row[0]):
        if not name.startswith(f"pcf_surface_{room}"):
            continue
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        metrics[name] = raster_mesh(mesh, image, z_buffer, camera_from_assembly, k)
    metrics["covered_pixel_count"] = int(np.isfinite(z_buffer).sum())
    return Image.fromarray(image), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--high-dir", type=Path, required=True)
    parser.add_argument("--view-index", type=int, default=35)
    parser.add_argument("--width", type=int, default=1008)
    parser.add_argument("--height", type=int, default=560)
    args = parser.parse_args()
    if not 0 <= args.view_index < 48 or args.width > 1400 or args.height > 800:
        raise ValueError("comparison view or image exceeds bound")
    binding = read_json(args.binding)
    descriptor = read_json(Path(binding["source_descriptor"]["path"]))
    multiroom = read_json(Path(binding["reintegration"]["path"]))
    prior_join = read_json(Path(binding["family_kitchen"]["path"]))
    registration = read_json(Path(binding["endpoint_registration"]["path"]))
    low = trimesh.load(Path(binding["mesh_manifest"]["path"]).parent / "multiroom_surface_mesh.glb", force="scene")
    high = trimesh.load(args.high_dir / "multiroom_surface_mesh_high_detail.glb", force="scene")
    output_dir = args.high_dir / "comparison"
    output_dir.mkdir(exist_ok=True)
    report = {"view_index": args.view_index, "width": args.width, "height": args.height, "owners": {}}
    for owner in (1, 2, 3):
        raw_root, correction, rows = _manifest_source_for_owner(
            owner, multiroom, prior_join, registration, binding, args.repo_root,
        )
        world_ref = binding["rooms"][{1:"family-room",2:"kitchen",3:"living-room"}[owner]]["source_world_manifest"]
        edge = (checked_matrix(rows[args.view_index]["effective_world_from_local_row_major"], "view edge")
                if rows is not None else correction @ world_from_local(read_json(Path(world_ref["path"]))))
        with np.load(raw_root / f"view_{args.view_index:04d}.npz", allow_pickle=False) as row:
            pose = checked_matrix(row["camera_pose"], "phone pose")
            k = np.asarray(row["intrinsics"], dtype=np.float64).copy()
        camera_from_assembly = np.linalg.inv(edge @ pose)
        k[0] *= args.width / 504
        k[1] *= args.height / 280
        room = {1:"family",2:"kitchen",3:"living"}[owner]
        result = {}
        for label, scene in (("low", low), ("high", high)):
            image, metrics = render(scene, owner, camera_from_assembly, k, args.width, args.height)
            target = output_dir / f"{room}_{label}_view_{args.view_index:04d}.png"
            image.save(target)
            result[label] = {"path": str(target), "metrics": metrics}
            print(f"{room} {label}: {metrics['covered_pixel_count']} covered pixels", flush=True)
        report["owners"][str(owner)] = result
    (output_dir / "comparison_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
