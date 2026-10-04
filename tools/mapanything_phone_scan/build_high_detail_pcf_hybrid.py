#!/usr/bin/env python3
"""Add measured low-prior gap faces to a frozen high-detail PCF review GLB.

The source high-detail geometry is retained byte-for-byte. The only image
change replaces its exporter-reencoded JPEGs with exact saved phone RGB atlases.
Low-prior faces are copied only into gaps with same-owner surfel support. This
is a review presentation derivative, not a geometry or calibration admission.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree


OWNER_NAMES = {"1": "family", "2": "kitchen", "3": "living"}
GLB_MAX_BYTES = 64 * 1024 * 1024
GLTF_JSON = 0x4E4F534A
GLTF_BIN = 0x004E4942


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return data


def unpack_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    if len(data) < 28 or data[:4] != b"glTF" or struct.unpack_from("<II", data, 4) != (2, len(data)):
        raise ValueError("invalid GLB header")
    json_size, json_type = struct.unpack_from("<II", data, 12)
    if json_type != GLTF_JSON:
        raise ValueError("GLB has no JSON first chunk")
    tree = json.loads(data[20:20 + json_size])
    binary_offset = 20 + json_size
    binary_size, binary_type = struct.unpack_from("<II", data, binary_offset)
    if binary_type != GLTF_BIN or binary_offset + 8 + binary_size != len(data):
        raise ValueError("GLB has unexpected BIN layout")
    if len(tree.get("buffers", [])) != 1:
        raise ValueError("expected one embedded GLB buffer")
    return tree, data[binary_offset + 8:]


def pack_glb(tree: dict, binary: bytes) -> bytes:
    tree["buffers"][0]["byteLength"] = len(binary)
    encoded = json.dumps(tree, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    size = 12 + 8 + len(encoded) + 8 + len(binary)
    if size > GLB_MAX_BYTES:
        raise ValueError(f"hybrid GLB exceeds 64 MiB bound: {size} bytes")
    return (struct.pack("<4sII", b"glTF", 2, size)
            + struct.pack("<II", len(encoded), GLTF_JSON) + encoded
            + struct.pack("<II", len(binary), GLTF_BIN) + binary)


def image_index_for_owner(tree: dict, owner: str) -> int:
    group = f"pcf_surface_{OWNER_NAMES[owner]}"
    photo = next(
        (node for node in tree["nodes"] if node.get("name") == f"{group}_photo"),
        None,
    )
    if photo is None or "mesh" not in photo:
        raise ValueError(f"source GLB lacks {group} photo mesh")
    primitive = tree["meshes"][photo["mesh"]]["primitives"]
    if len(primitive) != 1:
        raise ValueError(f"source GLB {group} photo primitive is ambiguous")
    material = tree["materials"][primitive[0]["material"]]
    texture_id = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    return int(tree["textures"][texture_id]["source"])


def exact_atlas_buffer_repack(
    tree: dict, binary: bytes, v2_dir: Path, compact: dict,
) -> tuple[bytearray, dict[str, dict], dict[int, str]]:
    replacements: dict[int, bytes] = {}
    room_images: dict[str, dict] = {}
    for owner in OWNER_NAMES:
        image_index = image_index_for_owner(tree, owner)
        image = tree["images"][image_index]
        if image["mimeType"] != "image/jpeg":
            raise ValueError(f"owner {owner} source texture is not JPEG")
        view_index = int(image["bufferView"])
        atlas_path = v2_dir / compact["owners"][owner]["texture_path"]
        image_bytes = atlas_path.read_bytes()
        if (sha_bytes(image_bytes) != compact["owners"][owner]["texture_sha256"]
                or len(image_bytes) != compact["owners"][owner]["texture_size_bytes"]):
            raise ValueError(f"owner {owner} exact saved atlas changed")
        if view_index in replacements:
            raise ValueError("source owners unexpectedly share an atlas bufferView")
        replacements[view_index] = image_bytes
        room_images[owner] = {
            "atlas_file": atlas_path.name,
            "atlas_sha256": sha_bytes(image_bytes),
            "atlas_size_bytes": len(image_bytes),
            "embedded_image_index": image_index,
            "embedded_buffer_view": view_index,
        }
    new_binary = bytearray()
    geometry_hashes: dict[int, str] = {}
    for index, view in enumerate(tree["bufferViews"]):
        if view.get("buffer", 0) != 0:
            raise ValueError("source GLB bufferView is not embedded")
        start, length = int(view.get("byteOffset", 0)), int(view["byteLength"])
        original = binary[start:start + length]
        if len(original) != length:
            raise ValueError("source GLB bufferView exceeds BIN chunk")
        while len(new_binary) % 4:
            new_binary.append(0)
        view["byteOffset"] = len(new_binary)
        selected = replacements.get(index, original)
        view["byteLength"] = len(selected)
        new_binary.extend(selected)
        if index not in replacements:
            geometry_hashes[index] = sha_bytes(original)
    return new_binary, room_images, geometry_hashes


def raycasting_scene(meshes: list[trimesh.Trimesh]) -> o3d.t.geometry.RaycastingScene:
    vertices = np.concatenate([np.asarray(mesh.vertices, dtype=np.float32) for mesh in meshes])
    offsets = np.cumsum([0] + [len(mesh.vertices) for mesh in meshes[:-1]])
    faces = np.concatenate([
        np.asarray(mesh.faces, dtype=np.uint32) + offset
        for mesh, offset in zip(meshes, offsets, strict=True)
    ])
    if not len(faces):
        raise ValueError("high-detail owner mesh is empty")
    source = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices), o3d.core.Tensor(faces)
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(source)
    return scene


def high_distance(scene: o3d.t.geometry.RaycastingScene, points: np.ndarray) -> np.ndarray:
    # Batches bound RaycastingScene scratch allocation on large owners.
    output = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), 100000):
        stop = min(start + 100000, len(points))
        output[start:stop] = scene.compute_distance(
            o3d.core.Tensor(np.asarray(points[start:stop], dtype=np.float32))
        ).numpy()
    return output


def low_gap_face_selection(
    low: trimesh.Trimesh, high_meshes: list[trimesh.Trimesh], surfels: np.ndarray,
    *, gap_centroid_m: float, gap_vertex_m: float, support_m: float,
    max_edge_m: float,
) -> tuple[np.ndarray, dict]:
    vertices = np.asarray(low.vertices, dtype=np.float32)
    faces = np.asarray(low.faces, dtype=np.int64)
    centers = vertices[faces].mean(1)
    high = raycasting_scene(high_meshes)
    centroid_gap = high_distance(high, centers)
    vertex_gap = high_distance(high, vertices)
    gap = (centroid_gap >= gap_centroid_m) & (vertex_gap[faces].min(1) >= gap_vertex_m)
    tree = cKDTree(np.asarray(surfels, dtype=np.float32))
    center_support, _ = tree.query(centers, k=1, workers=1)
    vertex_support, _ = tree.query(vertices, k=1, workers=1)
    supported = (center_support <= support_m) & (vertex_support[faces].max(1) <= support_m)
    edge = vertices[faces] - vertices[faces[:, [1, 2, 0]]]
    edge_length = np.sqrt((edge * edge).sum(2)).max(1)
    bounded = edge_length <= max_edge_m
    selected = gap & supported & bounded
    metrics = {
        "low_source_triangle_count": int(len(faces)),
        "centroid_gap_candidate_triangle_count": int(np.count_nonzero(centroid_gap >= gap_centroid_m)),
        "centroid_and_vertex_gap_candidate_triangle_count": int(np.count_nonzero(gap)),
        "gap_candidate_unsupported_triangle_count": int(np.count_nonzero(gap & ~supported)),
        "gap_candidate_long_edge_triangle_count": int(np.count_nonzero(gap & supported & ~bounded)),
        "selected_gap_triangle_count": int(np.count_nonzero(selected)),
        "selected_gap_centroid_distance_to_high_p50_m": float(np.median(centroid_gap[selected])) if np.any(selected) else None,
        "selected_gap_centroid_distance_to_source_p95_m": float(np.percentile(center_support[selected], 95)) if np.any(selected) else None,
        "selected_gap_max_edge_p95_m": float(np.percentile(edge_length[selected], 95)) if np.any(selected) else None,
    }
    return selected, metrics


def append_blob(tree: dict, binary: bytearray, data: bytes, target: int) -> int:
    while len(binary) % 4:
        binary.append(0)
    index = len(tree["bufferViews"])
    tree["bufferViews"].append({
        "buffer": 0, "byteOffset": len(binary), "byteLength": len(data), "target": target,
    })
    binary.extend(data)
    return index


def append_accessor(
    tree: dict, view_index: int, component_type: int, count: int,
    value_type: str, *, normalized: bool = False,
    minimum: list[float] | None = None, maximum: list[float] | None = None,
) -> int:
    row = {"bufferView": view_index, "componentType": component_type,
           "count": count, "type": value_type}
    if normalized:
        row["normalized"] = True
    if minimum is not None:
        row["min"] = minimum
    if maximum is not None:
        row["max"] = maximum
    index = len(tree["accessors"])
    tree["accessors"].append(row)
    return index


def append_gap_mesh(
    tree: dict, binary: bytearray, owner: str, source: trimesh.Trimesh,
    selected: np.ndarray,
) -> tuple[int, int]:
    if not np.any(selected):
        return 0, 0
    original_faces = np.asarray(source.faces[selected], dtype=np.int64)
    unique, inverse = np.unique(original_faces.reshape(-1), return_inverse=True)
    faces = inverse.reshape(-1, 3).astype("<u4")
    positions = np.asarray(source.vertices[unique], dtype="<f4")
    normals = np.asarray(source.vertex_normals[unique], dtype="<f4")
    colors = np.asarray(source.visual.vertex_colors[unique], dtype=np.uint8)
    if colors.shape != (len(unique), 4):
        raise ValueError(f"owner {owner} low mesh lacks RGBA vertex colors")
    if not np.isfinite(positions).all() or not np.isfinite(normals).all():
        raise ValueError(f"owner {owner} low mesh has nonfinite geometry")
    position_view = append_blob(tree, binary, positions.tobytes(order="C"), 34962)
    normal_view = append_blob(tree, binary, normals.tobytes(order="C"), 34962)
    color_view = append_blob(tree, binary, colors.tobytes(order="C"), 34962)
    index_view = append_blob(tree, binary, faces.tobytes(order="C"), 34963)
    position_accessor = append_accessor(
        tree, position_view, 5126, len(positions), "VEC3",
        minimum=positions.min(0).astype(float).tolist(),
        maximum=positions.max(0).astype(float).tolist(),
    )
    normal_accessor = append_accessor(tree, normal_view, 5126, len(normals), "VEC3")
    color_accessor = append_accessor(tree, color_view, 5121, len(colors), "VEC4", normalized=True)
    index_accessor = append_accessor(tree, index_view, 5125, faces.size, "SCALAR")
    name = f"pcf_surface_{OWNER_NAMES[owner]}_low_prior_gap_fill"
    mesh_index = len(tree["meshes"])
    tree["meshes"].append({
        "name": name,
        "primitives": [{
            "attributes": {"POSITION": position_accessor, "NORMAL": normal_accessor,
                           "COLOR_0": color_accessor},
            "indices": index_accessor,
            "mode": 4,
        }],
    })
    group_name = f"pcf_surface_{OWNER_NAMES[owner]}"
    group = next((node for node in tree["nodes"] if node.get("name") == group_name), None)
    if group is None or "mesh" in group:
        raise ValueError(f"source GLB has no owner group {group_name}")
    node_index = len(tree["nodes"])
    tree["nodes"].append({"name": name, "mesh": mesh_index})
    group.setdefault("children", []).append(node_index)
    return len(positions), len(faces)


def finalize_existing_hybrid(source_dir: Path, output_dir: Path) -> None:
    """Reissue a frozen hybrid with the review-loader provenance contract.

    This copies the GLB and saved atlases without opening, rendering, or
    recomputing geometry. The original manifest remains available for audit.
    """
    source_dir = source_dir.resolve()
    compact_path = source_dir / "tracking_high.json"
    full_path = source_dir / "mesh_manifest.json"
    compact = read_json(compact_path)
    full = read_json(full_path)
    if (full.get("contract") != "noesis.pcf.high_detail_review_surface_mesh_hybrid"
            or full.get("contract_version") != 1
            or full.get("accepted_for_canonical_use") is not False
            or sha_file(full_path) != compact["source"]["detail_manifest_sha256"]
            or full.get("output") != compact.get("output")
            or full.get("assembly_gauge") != compact.get("assembly_gauge")):
        raise ValueError("source hybrid provenance or assembly differs")
    artifact = compact["output"]
    if artifact["path"] != "multiroom_surface_mesh_high_detail.glb":
        raise ValueError("source hybrid artifact path differs")
    copy_names = [artifact["path"]]
    for owner in OWNER_NAMES:
        texture = compact["owners"][owner]
        embedded = full["owners"][owner]["embedded_atlas"]
        if (texture["texture_path"] != embedded["atlas_file"]
                or texture["texture_sha256"] != embedded["atlas_sha256"]
                or texture["texture_size_bytes"] != embedded["atlas_size_bytes"]):
            raise ValueError(f"owner {owner} atlas provenance differs")
        copy_names.append(texture["texture_path"])
    if len(set(copy_names)) != len(copy_names):
        raise ValueError("source hybrid file names overlap")
    for name in copy_names:
        path = source_dir / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise ValueError(f"invalid source hybrid file: {name}")
    if (sha_file(source_dir / artifact["path"]) != artifact["sha256"]
            or (source_dir / artifact["path"]).stat().st_size != artifact["size_bytes"]):
        raise ValueError("source hybrid GLB changed")
    for owner in OWNER_NAMES:
        row = compact["owners"][owner]
        atlas = source_dir / row["texture_path"]
        if sha_file(atlas) != row["texture_sha256"] or atlas.stat().st_size != row["texture_size_bytes"]:
            raise ValueError(f"owner {owner} source atlas changed")

    output_dir.mkdir(parents=True, exist_ok=False)
    for name in copy_names:
        shutil.copyfile(source_dir / name, output_dir / name)
        if sha_file(source_dir / name) != sha_file(output_dir / name):
            raise ValueError(f"copied hybrid file changed: {name}")
    updated_full = copy.deepcopy(full)
    updated_full["contract"] = "noesis.pcf.high_detail_review_surface_mesh"
    updated_full["coordinate_frame"] = compact["assembly_gauge"]
    updated_full["source"].update({
        "reissued_from_hybrid_glb_sha256": artifact["sha256"],
        "reissued_from_hybrid_compact_descriptor_sha256": sha_file(compact_path),
        "reissued_from_hybrid_detail_manifest_sha256": sha_file(full_path),
    })
    for owner in OWNER_NAMES:
        row = compact["owners"][owner]
        updated_full["owners"][owner]["atlas"] = {
            "path": row["texture_path"],
            "sha256": row["texture_sha256"],
            "size_bytes": row["texture_size_bytes"],
        }
    new_full_path = output_dir / "mesh_manifest.json"
    new_full_path.write_text(json.dumps(updated_full, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    updated_compact = copy.deepcopy(compact)
    updated_compact["source"]["detail_manifest_sha256"] = sha_file(new_full_path)
    new_compact_path = output_dir / "tracking_high.json"
    new_compact_path.write_text(json.dumps(updated_compact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": artifact, "descriptor_sha256": sha_file(new_compact_path),
                      "detail_manifest_sha256": sha_file(new_full_path)}, indent=2), flush=True)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--v2-dir", type=Path)
    parser.add_argument("--finalize-from", type=Path,
                        help="reissue a frozen hybrid without recomputing geometry")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gap-centroid-m", type=float, default=0.06)
    parser.add_argument("--gap-vertex-m", type=float, default=0.025)
    parser.add_argument("--max-edge-m", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse()
    if args.finalize_from is not None:
        if args.binding is not None or args.v2_dir is not None:
            raise ValueError("finalization does not accept build inputs")
        finalize_existing_hybrid(args.finalize_from, args.output_dir.resolve())
        return
    if args.binding is None or args.v2_dir is None:
        raise ValueError("hybrid build requires --binding and --v2-dir")
    if (not 0.04 <= args.gap_centroid_m <= 0.12
            or not 0.02 <= args.gap_vertex_m <= 0.06
            or args.gap_vertex_m >= args.gap_centroid_m
            or not 0.10 <= args.max_edge_m <= 0.20):
        raise ValueError("gap support or separation is outside the bounded review range")
    v2_dir, output_dir = args.v2_dir.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    binding = read_json(args.binding)
    descriptor_path = Path(binding["source_descriptor"]["path"])
    low_descriptor = read_json(descriptor_path)
    if sha_file(descriptor_path) != binding["source_descriptor"]["sha256"]:
        raise ValueError("low assembly descriptor changed")
    low_path = descriptor_path.parent.parent / low_descriptor["artifact"]["relative_path"]
    if (sha_file(low_path) != low_descriptor["artifact"]["sha256"]
            or low_path.stat().st_size != low_descriptor["artifact"]["size_bytes"]):
        raise ValueError("low review mesh changed")
    low_manifest_path = Path(binding["mesh_manifest"]["path"])
    if sha_file(low_manifest_path) != binding["mesh_manifest"]["sha256"]:
        raise ValueError("low mesh support contract changed")
    low_manifest = read_json(low_manifest_path)
    support_m = float(low_manifest["parameters"]["support_distance_m"])
    if not 0.01 <= support_m <= 0.10:
        raise ValueError("low mesh support bound is outside review policy")
    v2_compact_path = v2_dir / "tracking_high.json"
    v2_compact = read_json(v2_compact_path)
    v2_full_path = v2_dir / "mesh_manifest.json"
    if sha_file(v2_full_path) != v2_compact["source"]["detail_manifest_sha256"]:
        raise ValueError("v2 detailed provenance changed")
    v2_path = v2_dir / v2_compact["output"]["path"]
    if (sha_file(v2_path) != v2_compact["output"]["sha256"]
            or v2_path.stat().st_size != v2_compact["output"]["size_bytes"]):
        raise ValueError("v2 high-detail mesh changed")
    if (v2_compact["source"]["low_glb_sha256"] != low_descriptor["artifact"]["sha256"]
            or v2_compact["assembly_gauge"] != low_descriptor["assembly_gauge"]
            or v2_compact["source"]["binding_sha256"] != sha_file(args.binding)):
        raise ValueError("v2 and low source chain or assembly gauge differ")
    source_manifest_path = Path(binding["reintegration"]["path"])
    source_manifest = read_json(source_manifest_path)
    if sha_file(source_manifest_path) != binding["reintegration"]["sha256"]:
        raise ValueError("source reintegration manifest changed")
    surfel_path = source_manifest_path.parent / source_manifest["artifacts"]["surfels_npz"]["path"]
    if sha_file(surfel_path) != source_manifest["artifacts"]["surfels_npz"]["sha256"]:
        raise ValueError("source measured surfels changed")
    tree, source_binary = unpack_glb(v2_path)
    tree = copy.deepcopy(tree)
    binary, room_images, retained_buffer_hashes = exact_atlas_buffer_repack(
        tree, source_binary, v2_dir, v2_compact
    )
    low_scene = trimesh.load(low_path, force="scene")
    high_scene = trimesh.load(v2_path, force="scene")
    with np.load(surfel_path, allow_pickle=False) as arrays:
        all_points = np.asarray(arrays["points"], dtype=np.float32).copy()
        all_owners = np.asarray(arrays["owner_room_id"], dtype=np.uint8).copy()
        all_view_count = np.asarray(arrays["view_count"], dtype=np.uint16).copy()
    metrics: dict[str, dict] = {}
    total_vertices = int(v2_compact["output"]["vertex_count"])
    total_triangles = int(v2_compact["output"]["triangle_count"])
    compact = copy.deepcopy(v2_compact)
    for owner, room in OWNER_NAMES.items():
        name = f"pcf_surface_{room}"
        low_mesh = low_scene.geometry[name]
        high_meshes = [high_scene.geometry[f"{name}_photo"],
                       high_scene.geometry[f"{name}_rgb_fallback"]]
        source_mask = all_owners == int(owner)
        owner_surfels = all_points[source_mask]
        selected, owner_metrics = low_gap_face_selection(
            low_mesh, high_meshes, owner_surfels,
            gap_centroid_m=args.gap_centroid_m, gap_vertex_m=args.gap_vertex_m,
            support_m=support_m, max_edge_m=args.max_edge_m,
        )
        added_vertices, added_triangles = append_gap_mesh(
            tree, binary, owner, low_mesh, selected
        )
        tree["buffers"][0]["byteLength"] = len(binary)
        total_vertices += added_vertices
        total_triangles += added_triangles
        nearest = cKDTree(owner_surfels).query(
            np.asarray(low_mesh.triangles_center[selected], dtype=np.float32),
            k=1, workers=1,
        )[1] if added_triangles else np.empty(0, dtype=np.int64)
        supported_views = all_view_count[source_mask][nearest]
        owner_metrics.update({
            "room": room,
            "owner_group": name,
            "gap_mesh_name": f"{name}_low_prior_gap_fill",
            "selected_gap_vertex_count": added_vertices,
            "selected_gap_triangle_count": added_triangles,
            "nearest_surfel_two_or_more_view_fraction": float(np.mean(supported_views >= 2)) if len(supported_views) else None,
            "retained_high_photo_triangle_count": v2_compact["owners"][owner]["direct_photo_triangle_count"],
            "retained_high_rgb_fallback_triangle_count": v2_compact["owners"][owner]["fallback_triangle_count"],
            "embedded_atlas": room_images[owner],
            "atlas": {
                "path": compact["owners"][owner]["texture_path"],
                "sha256": room_images[owner]["atlas_sha256"],
                "size_bytes": room_images[owner]["atlas_size_bytes"],
            },
        })
        metrics[owner] = owner_metrics
        compact["output"]["owner_triangle_counts"][owner] += added_triangles
        compact["owners"][owner]["fallback_triangle_count"] += added_triangles
        print(f"{room}: {added_triangles} supported low-prior gap faces, {added_vertices} vertices", flush=True)
    packed = pack_glb(tree, bytes(binary))
    glb_path = output_dir / "multiroom_surface_mesh_high_detail.glb"
    glb_path.write_bytes(packed)
    compact["output"].update({
        "path": glb_path.name, "sha256": sha_bytes(packed),
        "size_bytes": len(packed), "vertex_count": total_vertices,
        "triangle_count": total_triangles,
    })
    for owner, row in room_images.items():
        atlas_path = v2_dir / row["atlas_file"]
        target = output_dir / row["atlas_file"]
        target.write_bytes(atlas_path.read_bytes())
        if sha_file(target) != row["atlas_sha256"]:
            raise ValueError(f"owner {owner} copied atlas differs")
    full = {
        "contract": "noesis.pcf.high_detail_review_surface_mesh",
        "contract_version": 1,
        "status": "review_only", "accepted_for_canonical_use": False,
        "assembly_id": low_descriptor["assembly_id"],
        "assembly_gauge": low_descriptor["assembly_gauge"],
        "coordinate_frame": low_descriptor["assembly_gauge"],
        "units": "meters",
        "method": "exact_v2_high_geometry_and_phone_atlases_plus_supported_existing_low_mesh_gap_faces",
        "source": {
            "binding_sha256": sha_file(args.binding),
            "v2_glb_sha256": v2_compact["output"]["sha256"],
            "v2_compact_descriptor_sha256": sha_file(v2_compact_path),
            "v2_detail_manifest_sha256": sha_file(v2_full_path),
            "low_glb_sha256": low_descriptor["artifact"]["sha256"],
            "low_mesh_manifest_sha256": binding["mesh_manifest"]["sha256"],
            "low_mesh_manifest_path": str(low_manifest_path),
            "reintegration_manifest_sha256": binding["reintegration"]["sha256"],
            "surfel_npz_sha256": source_manifest["artifacts"]["surfels_npz"]["sha256"],
            "retained_v2_non_image_buffer_view_count": len(retained_buffer_hashes),
            "retained_v2_non_image_buffer_views_sha256": sha_bytes(
                json.dumps(retained_buffer_hashes, sort_keys=True).encode("utf-8")
            ),
            "phone_walk_only": True,
            "static_camera_points_included": False,
            "registration_status": source_manifest["registration"]["status"],
        },
        "parameters": {
            "minimum_gap_centroid_to_high_surface_m": args.gap_centroid_m,
            "minimum_gap_vertex_to_high_surface_m": args.gap_vertex_m,
            "maximum_same_owner_surfel_support_m": support_m,
            "support_bound_source": "frozen_low_mesh_manifest.parameters.support_distance_m",
            "maximum_low_triangle_edge_m": args.max_edge_m,
            "surfel_support_includes_one_view_measurements": True,
            "global_alignment_changed": False,
            "v2_existing_geometry_buffer_bytes_unchanged": True,
            "low_mesh_vertices_copied_without_offset": True,
        },
        "owners": metrics,
        "output": compact["output"],
        "limitations": [
            "existing_low_mesh_gap_fill_is_a_source_backed_presentation_prior_not_new_physical_measurement",
            "gap selection does not independently prove free-space validity",
            "low-prior gaps may inherit smoothness or shape error from the prior mesh",
            "phone pose, depth, visibility and room registration uncertainty remain unchanged",
            "living room registration remains rejected for canonical use",
            "this artifact does not change tracking, calibration, Scene Prior or world authority",
        ],
    }
    full_path = output_dir / "mesh_manifest.json"
    full_path.write_text(json.dumps(full, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact["source"]["detail_manifest_sha256"] = sha_file(full_path)
    compact_path = output_dir / "tracking_high.json"
    compact_path.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": compact["output"], "descriptor_sha256": sha_file(compact_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
