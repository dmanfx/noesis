#!/usr/bin/env python3
"""Derive one compact, textured renderer mesh from a bound room surface.

This offline exporter consumes existing geometry and linear vertex colours.
It does not run reconstruction, change RoomWalk, or publish scene authority.
The dense colour field is baked into a UV atlas after geometric simplification,
so removing triangles does not also remove the original surface colour detail.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import time
from pathlib import Path

import fast_simplification
import numpy as np
import open3d as o3d
import xatlas
from PIL import Image
from scipy.ndimage import distance_transform_edt


def record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(),
            "size_bytes": path.stat().st_size}


def read_source(path: Path, manifest: dict, *, max_faces=3_000_000,
                max_vertices=2_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (isinstance(max_faces, bool) or not isinstance(max_faces, int) or not 1 <= max_faces <= 5_000_000
            or isinstance(max_vertices, bool) or not isinstance(max_vertices, int) or not 1 <= max_vertices <= 3_000_000):
        raise ValueError("source mesh work bounds exceed supported limits")
    expected = manifest["output"]
    actual = record(path)
    if (expected.get("sha256") != actual["sha256"]
            or expected.get("size_bytes") != actual["size_bytes"]):
        raise ValueError("source mesh does not match its manifest")
    if (manifest.get("accepted_for_canonical_use") is not False
            or manifest.get("status") != "review_only"):
        raise ValueError("this exporter requires an explicitly bound review surface")
    if manifest.get("color_bake", {}).get("vertex_color_encoding") != "gltf_linear_uint8":
        raise ValueError("source colour encoding must explicitly be glTF linear uint8")
    with path.open("rb") as stream:
        magic, version, size = struct.unpack("<III", stream.read(12))
        if magic != 0x46546C67 or version != 2 or size != actual["size_bytes"]:
            raise ValueError("invalid source GLB header")
        length, kind = struct.unpack("<II", stream.read(8))
        if kind != 0x4E4F534A or length > 4 * 1024 * 1024:
            raise ValueError("invalid source GLB JSON chunk")
        document = json.loads(stream.read(length))
        length, kind = struct.unpack("<II", stream.read(8))
        if kind != 0x004E4942 or length > 512 * 1024 * 1024:
            raise ValueError("invalid source GLB binary chunk")
        binary = stream.read(length)
    primitives = [p for mesh in document.get("meshes", []) for p in mesh["primitives"]]
    nodes = [node for node in document.get("nodes", []) if "mesh" in node]
    if len(primitives) != 1 or len(nodes) != 1:
        raise ValueError("source must contain one explicitly bound mesh primitive")
    if any(key in nodes[0] for key in ("matrix", "translation", "rotation", "scale")):
        raise ValueError("source mesh has a transform; bake it explicitly before export")
    primitive = primitives[0]
    if primitive.get("mode", 4) != 4:
        raise ValueError("source is not a triangle surface")
    types = {5121: np.dtype("u1"), 5123: np.dtype("<u2"),
             5125: np.dtype("<u4"), 5126: np.dtype("<f4")}
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def accessor(index: int) -> tuple[np.ndarray, dict]:
        definition = document["accessors"][index]
        if "sparse" in definition:
            raise ValueError("sparse source accessors are unsupported")
        view = document["bufferViews"][definition["bufferView"]]
        if view.get("buffer", 0) != 0 or "extensions" in view:
            raise ValueError("source accessor must use plain embedded bytes")
        dtype = types[definition["componentType"]]
        width = widths[definition["type"]]
        stride = view.get("byteStride", dtype.itemsize * width)
        offset = view.get("byteOffset", 0) + definition.get("byteOffset", 0)
        array = np.ndarray((definition["count"], width), dtype=dtype, buffer=binary,
                           offset=offset, strides=(stride, dtype.itemsize)).copy()
        return array, definition

    vertices, _ = accessor(primitive["attributes"]["POSITION"])
    indices, _ = accessor(primitive["indices"])
    colours, colour_def = accessor(primitive["attributes"]["COLOR_0"])
    faces = indices.reshape(-1, 3).astype(np.int32)
    if colours.dtype != np.uint8 or not colour_def.get("normalized"):
        raise ValueError("source colours disagree with the declared linear uint8 encoding")
    colours = colours[:, :3].astype(np.float32) / 255
    if (len(vertices) > max_vertices or len(faces) > max_faces
            or vertices.shape[1] != 3 or colours.shape != vertices.shape
            or not np.isfinite(vertices).all()
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("source mesh shape, indices, or work bounds are invalid")
    if len(faces) != expected["triangle_count"] or len(vertices) != expected["vertex_count"]:
        raise ValueError("source geometry counts disagree with its manifest")
    return vertices.astype(np.float32), faces, colours


def ray_scene(vertices: np.ndarray, faces: np.ndarray, threads: int) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene(nthreads=threads)
    scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(vertices), o3d.core.Dtype.Float32),
                        o3d.core.Tensor(np.ascontiguousarray(faces, dtype=np.uint32), o3d.core.Dtype.UInt32))
    return scene


def surface_samples(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    corners = vertices[faces]
    areas = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("source surface has no finite nonzero area")
    selected = rng.choice(len(faces), count, p=areas / areas.sum())
    bary = rng.random((count, 2)).astype(np.float32)
    outside = bary.sum(axis=1) > 1
    bary[outside] = 1 - bary[outside]
    return (corners[selected, 0] * (1 - bary.sum(axis=1))[:, None]
            + corners[selected, 1] * bary[:, :1] + corners[selected, 2] * bary[:, 1:]).astype(np.float32)


def distance_summary(scene, points: np.ndarray, threads: int) -> dict:
    result = scene.compute_closest_points(o3d.core.Tensor(points), nthreads=threads)
    distance = np.linalg.norm(points - result["points"].numpy(), axis=1)
    return {"sample_count": len(points), "median_m": float(np.median(distance)),
            "p95_m": float(np.quantile(distance, .95)), "p99_m": float(np.quantile(distance, .99)),
            "maximum_m": float(distance.max())}


def unwrap(vertices: np.ndarray, faces: np.ndarray, edge: int, *,
           chart_iterations: int = 1, chart_area: float = .75,
           cell_m: float = 1, maximum_patch_faces: int = 1024,
           padding_pixels: int = 8) -> tuple:
    if not 1 <= padding_pixels <= 8:
        raise ValueError("UV padding must be between one and eight pixels")
    atlas = xatlas.Atlas()
    # Bound chart-solving work spatially. Only UV seams are added; world-space
    # corners and triangle winding remain identical across these submeshes.
    cell_keys = np.floor(vertices[faces].mean(axis=1) / cell_m).astype(np.int32)
    _, groups = np.unique(cell_keys, axis=0, return_inverse=True)
    order = np.argsort(groups, kind="stable")
    cuts = np.flatnonzero(np.diff(groups[order])) + 1
    patches, input_face_rows = [], []
    for group in np.split(order, cuts):
        for start in range(0, len(group), maximum_patch_faces):
            rows = faces[group[start:start+maximum_patch_faces]]
            global_vertices, local_faces = np.unique(rows, return_inverse=True)
            patches.append(global_vertices)
            input_face_rows.append(rows)
            atlas.add_mesh(vertices[global_vertices], local_faces.reshape(-1, 3).astype(np.uint32))
    print(f"UV layout: {len(patches):,} spatial patches, at most {maximum_patch_faces} faces each", flush=True)
    charts = xatlas.ChartOptions()
    charts.fix_winding = True
    charts.max_iterations = chart_iterations
    charts.max_chart_area = chart_area
    pack = xatlas.PackOptions()
    pack.resolution = edge
    pack.padding = padding_pixels
    pack.bilinear = True
    pack.blockAlign = True
    atlas.generate(chart_options=charts, pack_options=pack, verbose=True)
    if atlas.atlas_count != 1:
        raise ValueError("UV unwrap did not fit into one finite atlas")
    mappings, indices, coordinates = [], [], []
    offset = 0
    for index, patch in enumerate(patches):
        mapping, triangles, uvs = atlas[index]
        if not np.isfinite(uvs).all():
            raise ValueError("UV layout has non-finite coordinates")
        mappings.append(patch[mapping])
        indices.append(triangles + offset)
        coordinates.append(uvs)
        offset += len(mapping)
    mapping = np.concatenate(mappings)
    triangles = np.concatenate(indices)
    uvs = np.concatenate(coordinates)
    def canonical_rows(rows):
        rotated = np.take_along_axis(rows, (np.argmin(rows, axis=1)[:, None] + np.arange(3)) % 3, axis=1)
        return rotated[np.lexsort((rotated[:, 2], rotated[:, 1], rotated[:, 0]))]
    if not np.array_equal(canonical_rows(mapping[triangles]), canonical_rows(np.concatenate(input_face_rows))):
        raise ValueError("UV layout changed source triangle corners or winding")
    # xatlas UVs are bottom-left. glTF and this image raster use top-left.
    uvs = np.asarray(uvs, dtype=np.float32).copy()
    uvs[:, 1] = 1 - uvs[:, 1]
    metadata = {"chart_count": sum(atlas.get_mesh_chart_count(i) for i in range(len(patches))),
                "spatial_patch_count": len(patches), "cell_m": cell_m,
                "maximum_patch_faces": maximum_patch_faces,
                "world_space_corners_and_winding_changed": False,
                "packing_utilization": atlas.utilization,
                "packed_width": atlas.width, "packed_height": atlas.height,
                "texture_edge": edge, "padding_pixels": pack.padding,
                "chart_iterations": chart_iterations, "maximum_chart_area_m2": chart_area,
                "uv_winding_enforced": True,
                "uv_origin": "gltf_top_left"}
    return vertices[mapping], np.asarray(triangles, dtype=np.uint32), uvs, mapping, metadata


def uv_raster(uvs: np.ndarray, faces: np.ndarray, edge: int, *, diagnostics: bool = False) -> tuple:
    face_map = np.full((edge, edge), -1, np.int32)
    bary_map = np.zeros((edge, edge, 2), np.float32)
    # Subpixel thin triangles need float64 edge equations. Float32 determinant
    # cancellation can make adjacent faces appear to have overlapping interiors.
    coords = uvs.astype(np.float64) * edge - .5
    ambiguous = np.zeros((edge, edge), bool)
    for index, triangle in enumerate(faces):
        p = coords[triangle]
        low = np.maximum(np.ceil(p.min(axis=0)).astype(int), 0)
        high = np.minimum(np.floor(p.max(axis=0)).astype(int), edge - 1)
        if np.any(high < low):
            continue
        a, b, c = p
        determinant = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(determinant) < 1e-12:
            continue
        yy, xx = np.mgrid[low[1]:high[1]+1, low[0]:high[0]+1]
        w0 = ((b[1]-c[1])*(xx-c[0]) + (c[0]-b[0])*(yy-c[1])) / determinant
        w1 = ((c[1]-a[1])*(xx-c[0]) + (a[0]-c[0])*(yy-c[1])) / determinant
        inside = (w0 >= -1e-10) & (w1 >= -1e-10) & (w0 + w1 <= 1 + 1e-10)
        sy, sx = yy[inside], xx[inside]
        # Shared triangle edges can touch the same sample; chart interiors must not overlap.
        interior = inside & (w0 > 1e-4) & (w1 > 1e-4) & (w0 + w1 < 1 - 1e-4)
        conflicting = face_map[yy[interior], xx[interior]] >= 0
        ambiguous[yy[interior][conflicting], xx[interior][conflicting]] = True
        face_map[sy, sx] = index
        bary_map[sy, sx, 0] = w0[inside]
        bary_map[sy, sx, 1] = w1[inside]
        if index % 50_000 == 0:
            print(f"UV raster: {index:,}/{len(faces):,} triangles", flush=True)
    covered = face_map >= 0
    count = int(ambiguous.sum())
    fraction = count / max(1, int(covered.sum()))
    if fraction > .001:
        raise ValueError(f"UV layout has excessive ambiguous texture coverage: {fraction:.3%}")
    # Small folded UV regions remain explicitly neutral, never mixed between
    # unrelated world surfaces. This does not change or remove mesh triangles.
    report = {"neutral_uv_overlap_texels": count, "neutral_uv_overlap_fraction": fraction,
              "maximum_ambiguous_fraction": .001,
              "policy": "neutral_colour_for_uv_ambiguity_no_geometry_removal"}
    print(f"UV raster: {int(covered.sum()):,} covered texels, {count:,} explicitly neutral overlaps", flush=True)
    if diagnostics:
        return face_map, bary_map, ambiguous, report
    return face_map, bary_map


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = value.clip(0, 1)
    return np.where(value <= .0031308, value * 12.92, 1.055 * np.power(value, 1/2.4) - .055)


def bake(source_scene, source_faces: np.ndarray, source_colours: np.ndarray,
         vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray, edge: int, threads: int,
         *, padding_pixels: int = 8) -> tuple:
    face_map, bary_map, ambiguous, raster_report = uv_raster(uvs, faces, edge, diagnostics=True)
    covered = face_map >= 0
    flat_valid = np.flatnonzero((covered & ~ambiguous).reshape(-1))
    image = np.full((edge * edge, 3), 128, np.uint8)
    offsets = np.zeros(len(flat_valid), np.float32)
    flat_bary = bary_map.reshape(-1, 2)
    flat_faces = face_map.reshape(-1)
    for start in range(0, len(flat_valid), 131_072):
        pixels = flat_valid[start:start+131_072]
        target_bary = flat_bary[pixels]
        target_triangles = vertices[faces[flat_faces[pixels]]]
        positions = (target_triangles[:, 0] * target_bary[:, :1]
                     + target_triangles[:, 1] * target_bary[:, 1:]
                     + target_triangles[:, 2] * (1-target_bary.sum(axis=1))[:, None]).astype(np.float32)
        hit = source_scene.compute_closest_points(o3d.core.Tensor(positions), nthreads=threads)
        source_triangles = source_faces[hit["primitive_ids"].numpy()]
        uv = hit["primitive_uvs"].numpy()
        weights = np.column_stack((1 - uv.sum(axis=1), uv))
        colour = (source_colours[source_triangles] * weights[:, :, None]).sum(axis=1)
        image[pixels] = np.rint(linear_to_srgb(colour) * 255).clip(0, 255).astype(np.uint8)
        offsets[start:start+len(pixels)] = np.linalg.norm(positions - hit["points"].numpy(), axis=1)
        if start % (131_072 * 16) == 0:
            print(f"Colour bake: {start:,}/{len(flat_valid):,} texels", flush=True)
    image = image.reshape(edge, edge, 3)
    known = covered & ~ambiguous
    del bary_map
    distance, nearest = distance_transform_edt(~known, return_indices=True)
    padding = (~covered) & (distance <= padding_pixels)
    image[padding] = image[nearest[0][padding], nearest[1][padding]]
    report = {"filled_texels": len(flat_valid), "fraction": len(flat_valid)/(edge*edge),
              "padding_texels": int(padding.sum()),
              "source_projection_distance_m": {"median": float(np.median(offsets)),
                  "p95": float(np.quantile(offsets, .95)), "p99": float(np.quantile(offsets, .99)),
                  "maximum": float(offsets.max()), "fraction_over_3cm": float(np.mean(offsets > .03))},
              "colour_transfer": "closest_dense_source_triangle_barycentric_linear_rgb_then_srgb",
              "unobserved_geometry_added": False, "image_inpainting": False}
    report["uv_raster"] = raster_report
    return Image.fromarray(image), report


def normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    corners = vertices[faces]
    face_normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    result = np.zeros(vertices.shape, np.float32)
    for corner in range(3):
        np.add.at(result, faces[:, corner], face_normal)
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-20)
    return result


def export_glb(path: Path, vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray,
               vertex_normals: np.ndarray, image: Image.Image, metadata: dict) -> None:
    payload = bytearray()
    views, accessors = [], []

    def add_bytes(value: bytes, target: int | None = None) -> int:
        payload.extend(b"\x00" * ((-len(payload)) % 4))
        item = {"buffer": 0, "byteOffset": len(payload), "byteLength": len(value)}
        if target is not None:
            item["target"] = target
        views.append(item)
        payload.extend(value)
        return len(views)-1

    def add_array(value: np.ndarray, kind: str, component: int, target: int, bounds=False) -> int:
        item = {"bufferView": add_bytes(value.tobytes(), target), "componentType": component,
                "count": len(value), "type": kind}
        if bounds:
            item.update(min=value.min(axis=0).tolist(), max=value.max(axis=0).tolist())
        accessors.append(item)
        return len(accessors)-1

    position = add_array(np.asarray(vertices, dtype="<f4"), "VEC3", 5126, 34962, True)
    normal = add_array(np.asarray(vertex_normals, dtype="<f4"), "VEC3", 5126, 34962)
    uv = add_array(np.asarray(uvs, dtype="<f4"), "VEC2", 5126, 34962)
    indices = add_array(np.asarray(faces.reshape(-1), dtype="<u4"), "SCALAR", 5125, 34963)
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=92, subsampling=0, optimize=True)
    image_view = add_bytes(encoded.getvalue())
    document = {"asset": {"version": "2.0", "generator": "Noesis offline room renderer mesh exporter"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scenes": [{"nodes": [0], "extras": metadata}], "scene": 0,
        "nodes": [{"mesh": 0, "name": "Living Room reconstruction trial"}],
        "meshes": [{"name": "room_surface", "primitives": [{"attributes": {
            "POSITION": position, "NORMAL": normal, "TEXCOORD_0": uv}, "indices": indices, "material": 0}]}],
        "materials": [{"name": "Baked room surface colour", "doubleSided": True,
            "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0, "roughnessFactor": 1},
            "extensions": {"KHR_materials_unlit": {}}}],
        "images": [{"bufferView": image_view, "mimeType": "image/jpeg", "name": "room_colour_atlas"}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "bufferViews": views, "accessors": accessors, "buffers": [{"byteLength": len(payload)}]}
    encoded_json = json.dumps(document, separators=(",", ":")).encode()
    encoded_json += b" " * ((-len(encoded_json)) % 4)
    payload.extend(b"\x00" * ((-len(payload)) % 4))
    total = 12 + 8 + len(encoded_json) + 8 + len(payload)
    with path.open("xb") as stream:
        stream.write(struct.pack("<III", 0x46546C67, 2, total))
        stream.write(struct.pack("<II", len(encoded_json), 0x4E4F534A))
        stream.write(encoded_json)
        stream.write(struct.pack("<II", len(payload), 0x004E4942))
        stream.write(payload)


def build(args) -> dict:
    started = time.monotonic()
    if args.output_dir.exists():
        raise ValueError("refusing to overwrite a previous renderer trial")
    if not 10_000 <= args.triangles <= 600_000 or args.texture_edge not in (1024, 2048, 4096):
        raise ValueError("triangle or texture budget is outside the bounded trial range")
    if not 1 <= args.threads <= 4:
        raise ValueError("CPU query work must use one to four threads")
    manifest = json.loads(args.source_manifest.read_text())
    source_vertices, source_faces, source_colours = read_source(args.source_glb, manifest,
        max_faces=getattr(args, "max_source_faces", 3_000_000),
        max_vertices=getattr(args, "max_source_vertices", 2_000_000))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Loaded {len(source_faces):,} dense source triangles", flush=True)
    vertices, faces = fast_simplification.simplify(source_vertices, source_faces,
                                                  target_count=args.triangles, agg=5)
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int32)
    print(f"Simplified to {len(faces):,} triangles in {time.monotonic()-started:.1f}s", flush=True)
    source_scene = ray_scene(source_vertices, source_faces, args.threads)
    target_scene = ray_scene(vertices, faces, args.threads)
    errors = {"sample_method": "area_weighted_deterministic_surface_samples",
        "simplified_to_dense": distance_summary(source_scene, surface_samples(vertices, faces, 100_000, 73), args.threads),
        "dense_to_simplified": distance_summary(target_scene, surface_samples(source_vertices, source_faces, 100_000, 91), args.threads),
        "interpretation": "export_loss_relative_to_saved_mesh_not_independent_physical_accuracy"}
    print(json.dumps(errors, indent=2), flush=True)
    np.savez_compressed(args.output_dir / "simplified_geometry.npz", vertices=vertices, faces=faces)
    base_normals = normals(vertices, faces)
    vertices, faces, uvs, mapping, uv_report = unwrap(vertices, faces, args.texture_edge,
        chart_iterations=args.chart_iterations, chart_area=args.chart_area,
        padding_pixels=getattr(args, "atlas_padding", 8))
    print(f"Unwrapped {uv_report['chart_count']:,} charts in {time.monotonic()-started:.1f}s", flush=True)
    np.savez_compressed(args.output_dir / "unwrapped_geometry.npz", vertices=vertices,
                        faces=faces, uvs=uvs, normals=base_normals[mapping])
    (args.output_dir / "uv_layout.json").write_text(json.dumps(uv_report, indent=2) + "\n")
    atlas, bake_report = bake(source_scene, source_faces, source_colours,
                             vertices, faces, uvs, args.texture_edge, args.threads,
                             padding_pixels=getattr(args, "bake_padding", 8))
    atlas.save(args.output_dir / "colour_atlas.png")
    completion = manifest.get("completion", {})
    measured_area = completion.get("class_surface_area_m2", {})
    inferred_fraction = (sum(float(value) for key, value in measured_area.items() if key != "multi_view_supported")
                         / sum(measured_area.values())) if measured_area else None
    metadata = {"review_only": True, "accepted_for_canonical_use": False,
        "coordinate_frame": manifest.get("coordinate_frame"),
        "source_sha256": record(args.source_glb)["sha256"],
        "inherited_source_inferred_area_fraction": inferred_fraction,
        "geometry_is_simplified_presentation": True, "source_scan_id": args.scan_id}
    export_glb(args.output_dir / "room.glb", vertices, faces, uvs, base_normals[mapping], atlas, metadata)
    report = {"schema": "noesis.room_renderer_mesh_trial.v1", "status": "offline_presentation_trial",
        **metadata, "source": {"mesh": record(args.source_glb), "manifest": record(args.source_manifest),
            "reconstruction_source": {key: value for key, value in manifest.get("source", {}).items()
                if key in ("consensus_manifest", "alignment_report", "alignment_transform", "bound_native_texture_photos")}},
        "parameters": {"triangle_budget": args.triangles, "texture_edge": args.texture_edge,
            "simplification": "quadric_edge_collapse", "aggressiveness": 5,
            "cpu_query_threads": args.threads, "jpeg_quality": 92,
            "atlas_padding_pixels": getattr(args, "atlas_padding", 8),
            "outside_chart_bake_padding_pixels": getattr(args, "bake_padding", 8)},
        "source_work_bounds": {"max_faces": getattr(args, "max_source_faces", 3_000_000),
            "max_vertices": getattr(args, "max_source_vertices", 2_000_000)},
        "output": {**record(args.output_dir / "room.glb"), "triangle_count": len(faces),
            "vertex_count": len(vertices), "material_count": 1, "texture_count": 1,
            "draw_calls": 1, "decoded_texture_bytes_rgba_with_mips": int(args.texture_edge**2*4*4/3)},
        "export_geometry_error": errors, "uv_atlas": uv_report, "texture_bake": bake_report,
        "elapsed_seconds": time.monotonic()-started,
        "limitations": ["source_reconstruction_errors_and_inferred_completion_are_inherited",
            "no_new_geometry_or_reconstruction_accuracy_claim", "not_admitted_to_live_menon_or_noesis"]}
    (args.output_dir / "mesh_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": report["output"], "elapsed_seconds": report["elapsed_seconds"]}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-glb", "source-manifest", "output-dir"):
        parser.add_argument("--"+flag, required=True, type=Path)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--triangles", type=int, default=150_000)
    parser.add_argument("--texture-edge", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--chart-iterations", type=int, choices=(1, 2), default=1)
    parser.add_argument("--atlas-padding", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--bake-padding", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--max-source-faces", type=int, default=3_000_000)
    parser.add_argument("--max-source-vertices", type=int, default=2_000_000)
    parser.add_argument("--chart-area", type=float, default=.75,
                        help="Optional maximum UV chart area in square metres; zero is unlimited")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
