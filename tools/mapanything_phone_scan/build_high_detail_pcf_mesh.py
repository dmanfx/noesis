#!/usr/bin/env python3
"""Offline, review-only high-detail RGB derivative of the retained PCF assembly.

The 3-D source is the recorded, ownership-filtered 2.5 cm multiroom surfel set.
Each owner is meshed separately. Texture pixels come only from prepared phone
frames whose exact identities are bound by the conditioned MapAnything suite.
The raw fused RGB-D views provide camera/depth/mask visibility, never new geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals


OWNER_NAMES = {1: "family", 2: "kitchen", 3: "living"}
SCAN_IDS = {
    1: "20260810-215847-571c6efe",
    2: "20260814-190412-ea26c040",
    3: "20260802-162254-8bcc7dd7",
}
MODEL_GRID = (518, 294)
MAX_ATLAS_SIDE = 8192


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def checked_matrix(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid {label}")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"invalid affine {label}")
    if np.linalg.det(matrix[:3, :3]) <= 0:
        raise ValueError(f"invalid handedness {label}")
    return matrix


def world_from_local(manifest: dict) -> np.ndarray:
    alignment = manifest["alignment"]
    matrix = np.eye(4)
    matrix[:3, :3] = float(alignment["scale"]) * np.asarray(
        alignment["rotation_row_major"], dtype=np.float64
    )
    matrix[:3, 3] = alignment["translation"]
    return checked_matrix(matrix, "accepted phone-to-world edge")


@dataclass
class View:
    index: int
    depth: np.ndarray
    mask: np.ndarray
    confidence: np.ndarray
    k: np.ndarray
    from_assembly: np.ndarray
    model_k: np.ndarray
    model_to_atlas: np.ndarray | None
    tile_xy: tuple[int, int]
    raw_path: Path
    raw_sha256: str
    model_path: Path
    prepared_path: Path
    prepared_sha256: str
    image_matches: int
    image_inliers: int
    image_error_px_p50: float


def image_match(model_rgb: np.ndarray, source_rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Measure the saved provider image to prepared-image map; reject weak pairs."""
    detector = cv2.SIFT_create(nfeatures=2200)
    gray_model = cv2.cvtColor(model_rgb, cv2.COLOR_RGB2GRAY)
    gray_source = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    key_model, desc_model = detector.detectAndCompute(gray_model, None)
    key_source, desc_source = detector.detectAndCompute(gray_source, None)
    if desc_model is None or desc_source is None:
        raise ValueError("prepared/provider image has no repeatable features")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_model, desc_source, k=2)
    matches = [first for first, second in pairs if first.distance < 0.70 * second.distance]
    if len(matches) < 100:
        raise ValueError(f"too few image matches: {len(matches)}")
    src = np.float32([key_model[m.queryIdx].pt for m in matches])
    dst = np.float32([key_source[m.trainIdx].pt for m in matches])
    training = np.arange(len(matches)) % 4 != 0
    matrix, inlier = cv2.estimateAffinePartial2D(
        src[training], dst[training], method=cv2.RANSAC, ransacReprojThreshold=2.5,
        maxIters=3000, confidence=0.995,
    )
    if matrix is None or inlier is None:
        raise ValueError("provider/prepared image map could not be measured")
    keep = inlier[:, 0].astype(bool)
    predicted = src[training][keep] @ matrix[:, :2].T + matrix[:, 2]
    errors = np.linalg.norm(predicted - dst[training][keep], axis=1)
    withheld_predicted = src[~training] @ matrix[:, :2].T + matrix[:, 2]
    withheld_error = np.linalg.norm(withheld_predicted - dst[~training], axis=1)
    withheld_good = withheld_error <= 2.5
    scale = math.sqrt(float(np.linalg.det(matrix[:, :2])))
    expected_scale = source_rgb.shape[1] / model_rgb.shape[1]
    # Exact frame pairing plus this narrow check prevents a drifting affine
    # registration from substituting for the recorded preprocessing geometry.
    if (keep.sum() < 75 or keep.mean() < 0.35
            or withheld_good.sum() < 25 or withheld_good.mean() < 0.30
            or not 0.96 * expected_scale <= scale <= 1.05 * expected_scale):
        raise ValueError("provider/prepared image map violates scale/inlier bounds")
    p50 = float(np.median(errors))
    heldout_p50 = float(np.median(withheld_error[withheld_good]))
    if p50 > 2.5 or heldout_p50 > 2.5:
        raise ValueError(f"provider/prepared image residual {p50:.2f}/{heldout_p50:.2f}px exceeds 2.5px")
    return matrix, {
        "match_count": len(matches), "inlier_count": int(keep.sum()),
        "inlier_fraction": float(keep.mean()), "residual_px_p50": p50,
        "withheld_match_count": int((~training).sum()),
        "withheld_inlier_count": int(withheld_good.sum()),
        "withheld_residual_px_p50": heldout_p50,
        "scale": scale, "expected_resize_scale": expected_scale,
    }


def _manifest_source_for_owner(
    owner: int, multiroom: dict, prior_join: dict, registration: dict,
    recipe: dict, repo_root: Path,
) -> tuple[Path, np.ndarray, list[dict] | None]:
    if owner in (1, 2):
        role = "fixed_room" if owner == 1 else "moving_room"
        block = prior_join[role]
        if block["accepted_world_manifest_sha256"] != recipe["rooms"]["family-room" if owner == 1 else "kitchen"]["source_world_manifest"]["sha256"]:
            raise ValueError(f"owner {owner} source world manifest differs from binding")
        return Path(block["raw_root"]), np.asarray(block["global_world_correction_row_major"], dtype=np.float64), block["raw_views"]
    block = multiroom["sources"]["living"]
    if block["world_manifest_sha256"] != recipe["rooms"]["living-room"]["source_world_manifest"]["sha256"]:
        raise ValueError("living source world manifest differs from binding")
    transform = checked_matrix(registration["derived_endpoint_transforms"]["living_to_family_row_major"], "living-to-family edge")
    return Path(block["raw_root"]), transform, None


def load_views(
    owner: int, *, multiroom: dict, prior_join: dict, registration: dict,
    recipe: dict, repo_root: Path, tile_width: int, tile_height: int,
    atlas: Image.Image,
) -> tuple[list[View], list[dict]]:
    raw_root, correction, recorded_rows = _manifest_source_for_owner(
        owner, multiroom, prior_join, registration, recipe, repo_root
    )
    scan_dir = repo_root / "data/mapanything_phone_scans" / SCAN_IDS[owner]
    prepared_manifest_path = scan_dir / "prepared_frames_manifest.json"
    prepared_manifest = read_json(prepared_manifest_path)
    consensus_manifest = read_json(raw_root.parent / "consensus_manifest.json")
    model_raw = Path(consensus_manifest["inputs"]["mapanything_raw"])
    variant_dir = model_raw.parent
    variant_manifest = read_json(variant_dir / "variant_manifest.json")
    suite_manifest = read_json(variant_dir.parent / "variant_suite_manifest.json")
    if suite_manifest["prepared_frames_manifest_sha256"] != digest(prepared_manifest_path):
        raise ValueError(f"owner {owner} prepared frame manifest changed")
    if variant_manifest["inputs"]["phone_view_count"] != 48:
        raise ValueError(f"owner {owner} conditioned variant view count changed")
    frames = prepared_manifest["frames"]
    model_frames = variant_manifest["frames"]
    paths = sorted(raw_root.glob("view_*.npz"))
    if len(paths) != 48 or len(frames) != 48 or len(model_frames) != 48:
        raise ValueError(f"owner {owner} must have 48 exact phone views")
    if recorded_rows is not None and len(recorded_rows) != 48:
        raise ValueError(f"owner {owner} source view manifest is incomplete")
    world_manifest_ref = recipe["rooms"][{1:"family-room",2:"kitchen",3:"living-room"}[owner]]["source_world_manifest"]
    world_manifest_path = Path(world_manifest_ref["path"])
    if digest(world_manifest_path) != world_manifest_ref["sha256"]:
        raise ValueError(f"owner {owner} source world manifest changed")
    source_world = world_from_local(read_json(world_manifest_path))
    views: list[View] = []
    provenance: list[dict] = []
    for index, raw_path in enumerate(paths):
        frame = frames[index]
        model_frame = model_frames[index]
        if (frame["index"] != index or model_frame["index"] != index
                or model_frame["source_frame"] != frame["frame"]
                or raw_path.name != f"view_{index:04d}.npz"):
            raise ValueError(f"owner {owner} view {index} identity mismatch")
        prepared_path = scan_dir / frame["frame"]
        if digest(prepared_path) != frame["sha256"]:
            raise ValueError(f"owner {owner} prepared frame {index} changed")
        model_path = model_raw / f"view_{index:04d}.npz"
        if not model_path.is_file():
            raise ValueError(f"owner {owner} missing conditioned RGB view {index}")
        raw_sha = digest(raw_path)
        if recorded_rows is not None:
            row = recorded_rows[index]
            if row["path"] != str(raw_path) or row["sha256"] != raw_sha:
                raise ValueError(f"owner {owner} raw view {index} changed")
            edge = checked_matrix(row["effective_world_from_local_row_major"], "recorded source view edge")
            expected_edge = correction @ source_world
            if not np.allclose(edge, expected_edge, atol=2e-5):
                raise ValueError(f"owner {owner} view {index} source edge differs")
        else:
            edge = correction @ source_world
        with np.load(raw_path, allow_pickle=False) as raw:
            depth = np.asarray(raw["depth_z"], dtype=np.float32).copy()
            mask = np.asarray(raw["mask"], dtype=bool).copy()
            confidence = np.asarray(raw["confidence"], dtype=np.float32).copy()
            k = np.asarray(raw["intrinsics"], dtype=np.float64).copy()
            pose = checked_matrix(raw["camera_pose"], "fused phone pose")
        with np.load(model_path, allow_pickle=False) as model:
            model_k = np.asarray(model["intrinsics"], dtype=np.float64).copy()
            model_rgb = np.asarray(model["model_rgb"], dtype=np.float32)
            model_rgb = np.uint8(np.clip(np.rint(model_rgb * 255 if model_rgb.max() <= 1.5 else model_rgb), 0, 255))
        if depth.shape != (280, 504) or mask.shape != depth.shape or confidence.shape != depth.shape:
            raise ValueError(f"owner {owner} view {index} depth grid changed")
        if model_rgb.shape != (294, 518, 3) or model_k.shape != (3, 3) or k.shape != (3, 3):
            raise ValueError(f"owner {owner} view {index} provider grid changed")
        if not np.isfinite(k).all() or not np.isfinite(model_k).all():
            raise ValueError(f"owner {owner} view {index} invalid intrinsics")
        with Image.open(prepared_path) as image:
            prepared = image.convert("RGB")
            if prepared.size != (frame["width"], frame["height"]):
                raise ValueError(f"owner {owner} view {index} prepared size differs")
            tile = prepared.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        tile_rgb = np.asarray(tile)
        try:
            measured, match = image_match(model_rgb, tile_rgb)
            image_status = "accepted"
        except ValueError as exc:
            # A weak image pair never supplies texture, while its retained
            # RGB-D depth remains available to contradict a false surface.
            measured, match = None, None
            image_status = f"rejected: {exc}"
            print(f"owner {owner} view {index} texture rejected: {exc}", flush=True)
        tile_x = (index % 8) * tile_width
        tile_y = (index // 8) * tile_height
        map_to_atlas = None
        if measured is not None:
            atlas.paste(tile, (tile_x, tile_y))
            map_to_atlas = measured.copy()
            map_to_atlas[:, 2] += [tile_x, tile_y]
        camera_to_assembly = edge @ pose
        views.append(View(
            index=index, depth=depth, mask=mask, confidence=confidence, k=k,
            from_assembly=np.linalg.inv(camera_to_assembly), model_k=model_k,
            model_to_atlas=map_to_atlas, tile_xy=(tile_x, tile_y),
            raw_path=raw_path, raw_sha256=raw_sha, model_path=model_path,
            prepared_path=prepared_path, prepared_sha256=frame["sha256"],
            image_matches=match["match_count"] if match else 0,
            image_inliers=match["inlier_count"] if match else 0,
            image_error_px_p50=match["residual_px_p50"] if match else float("nan"),
        ))
        provenance.append({
            "view_index": index, "raw_sha256": raw_sha,
            "prepared_frame_sha256": frame["sha256"],
            "conditioned_model_rgb_path": str(model_path),
            "prepared_frame_path": str(prepared_path),
            "model_to_atlas_row_major": measured.tolist() if measured is not None else None,
            "image_match": match, "image_status": image_status,
        })
        if index % 12 == 11:
            print(f"owner {owner} verified RGB-D/phone image views {index + 1}/48", flush=True)
    return views, provenance


def orient_normals(cloud: o3d.geometry.PointCloud, cameras: np.ndarray) -> None:
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    for start in range(0, len(points), 20000):
        stop = min(start + 20000, len(points))
        block = points[start:stop]
        distances = np.sum((block[:, None, :] - cameras[None, :, :]) ** 2, axis=2)
        toward = cameras[np.argmin(distances, axis=1)] - block
        flip = np.einsum("ij,ij->i", normals[start:stop], toward) < 0
        normals[start:stop][flip] *= -1


def geometry_from_surfels(
    points: np.ndarray, colors: np.ndarray, cameras: np.ndarray,
    *, depth: int, poisson_threads: int,
) -> tuple[o3d.geometry.TriangleMesh, dict]:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.09, max_nn=48))
    orient_normals(cloud, cameras)
    start = time.monotonic()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=depth, scale=1.05, linear_fit=True, n_threads=poisson_threads,
    )
    elapsed = time.monotonic() - start
    vertices = np.asarray(mesh.vertices)
    density = np.asarray(densities)
    tree = cKDTree(points)
    distance, _ = tree.query(vertices, k=1, workers=1)
    minimum, maximum = points.min(0) - 0.075, points.max(0) + 0.075
    maximum[1] = min(maximum[1], 1.88)
    accepted = ((density >= np.percentile(density, 2)) & (distance <= 0.075)
                & np.all(vertices >= minimum, axis=1) & np.all(vertices <= maximum, axis=1))
    rejected_vertices = int(np.count_nonzero(~accepted))
    mesh.remove_vertices_by_mask(~accepted)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    centroid = verts[faces].mean(1)
    dist, _ = tree.query(centroid, k=1, workers=1)
    edges = verts[faces[:, [0, 1, 2]]] - verts[faces[:, [1, 2, 0]]]
    max_edge = np.sqrt(np.sum(edges * edges, axis=2)).max(1)
    keep = (dist <= 0.065) & (max_edge <= 0.15)
    rejected_faces = int(np.count_nonzero(~keep))
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh, {
        "input_surfel_count": int(len(points)),
        "poisson_seconds": elapsed,
        "density_vertex_rejected_count": rejected_vertices,
        "centroid_or_edge_rejected_triangle_count": rejected_faces,
        "pre_visibility_vertex_count": int(len(mesh.vertices)),
        "pre_visibility_triangle_count": int(len(mesh.triangles)),
    }


def project(points: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = points @ view.from_assembly[:3, :3].T + view.from_assembly[:3, 3]
    z = camera[:, 2]
    safe_z = np.where(z > 1e-5, z, 1.0)
    uv = np.empty((len(points), 2), dtype=np.float32)
    uv[:, 0] = (view.k[0, 0] * camera[:, 0] + view.k[0, 2] * z) / safe_z
    uv[:, 1] = (view.k[1, 1] * camera[:, 1] + view.k[1, 2] * z) / safe_z
    return camera, z, uv


def sample_depth(points: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera, z, uv = project(points, view)
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    inside = (z > 0.05) & (z < 12.0) & (u >= 1) & (u < 503) & (v >= 1) & (v < 279)
    indices = np.flatnonzero(inside)
    visible = np.zeros(len(points), dtype=bool)
    free_space = np.zeros(len(points), dtype=bool)
    quality = np.zeros(len(points), dtype=np.float32)
    if len(indices):
        pixel_depth = view.depth[v[indices], u[indices]]
        pixel_mask = view.mask[v[indices], u[indices]]
        confidence = view.confidence[v[indices], u[indices]]
        tolerance = np.maximum(0.05, 0.03 * pixel_depth)
        valid = pixel_mask & np.isfinite(pixel_depth) & (confidence >= 0.20)
        distance = np.abs(z[indices] - pixel_depth)
        visible[indices] = valid & (distance <= tolerance)
        free_space[indices] = valid & (z[indices] + tolerance < pixel_depth)
        quality[indices] = np.where(visible[indices], np.clip(confidence, 0, 1), 0)
    return visible, free_space, quality


def texture_mesh(
    mesh: o3d.geometry.TriangleMesh, views: list[View], atlas: Image.Image,
    *, tile_width: int, tile_height: int,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh | None, dict]:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.uint8(np.clip(np.rint(np.asarray(mesh.vertex_colors) * 255), 0, 255))
    centers = vertices[faces].mean(1)
    normals = np.asarray(mesh.triangle_normals)
    best = np.full(len(faces), -1, dtype=np.int16)
    score = np.zeros(len(faces), dtype=np.float32)
    free_contradiction_count = np.zeros(len(faces), dtype=np.uint8)
    atlas_w, atlas_h = atlas.size
    # Each view is considered independently; only O(vertices + faces) scratch is
    # live. All triangle corners and the center must match its measured depth.
    for view in views:
        vertex_visible, vertex_free, vertex_quality = sample_depth(vertices, view)
        center_visible, center_free, center_quality = sample_depth(centers, view)
        free_contradiction_count += center_free.astype(np.uint8)
        if view.model_to_atlas is None:
            continue
        possible = center_visible & vertex_visible[faces].all(1)
        if not np.any(possible):
            continue
        camera_center = np.linalg.inv(view.from_assembly)[:3, 3]
        toward = camera_center[None, :] - centers
        distance = np.linalg.norm(toward, axis=1)
        cos_angle = np.abs(np.einsum("ij,ij->i", normals, toward)) / np.maximum(distance, 1e-5)
        possible &= cos_angle >= 0.17
        possible &= ~vertex_free[faces].any(1)
        if not np.any(possible):
            continue
        candidate = (np.minimum(center_quality, vertex_quality[faces].min(1))
                     * np.sqrt(cos_angle) / np.maximum(distance, 0.4))
        candidate[~possible] = 0
        better = candidate > score
        best[better] = view.index
        score[better] = candidate[better]
        if view.index % 12 == 11:
            print(f"  visibility and texture view {view.index + 1}/48", flush=True)
    contradictory = free_contradiction_count >= 2
    supported = ~contradictory & (best >= 0)
    # The fallback retains directly measured surfel RGB where no individual
    # image covers every triangle corner, but still requires centroid support.
    fallback = ~contradictory & (best < 0)
    if np.count_nonzero(supported) < 10000:
        raise ValueError("too few directly visible textured triangles")
    original_faces = faces[supported]
    face_view = best[supported]
    # A view seam needs a separate UV at the same 3-D vertex. Indexing the
    # (original vertex, view) pairs makes this duplication explicit and bounded.
    pair = np.stack((original_faces.reshape(-1), np.repeat(face_view, 3)), axis=1)
    unique, inverse = np.unique(pair, axis=0, return_inverse=True)
    output_faces = inverse.reshape(-1, 3).astype(np.int32)
    output_vertices = vertices[unique[:, 0]].astype(np.float32)
    uv = np.zeros((len(unique), 2), dtype=np.float32)
    for view in views:
        if view.model_to_atlas is None:
            continue
        selection = unique[:, 1] == view.index
        if not np.any(selection):
            continue
        camera, z, _ = project(output_vertices[selection], view)
        model_uv = np.empty((len(z), 2), dtype=np.float64)
        model_uv[:, 0] = view.model_k[0, 0] * camera[:, 0] / z + view.model_k[0, 2]
        model_uv[:, 1] = view.model_k[1, 1] * camera[:, 1] / z + view.model_k[1, 2]
        atlas_xy = model_uv @ view.model_to_atlas[:, :2].T + view.model_to_atlas[:, 2]
        atlas_xy[:, 0] = np.clip(atlas_xy[:, 0], view.tile_xy[0] + 1, view.tile_xy[0] + tile_width - 2)
        atlas_xy[:, 1] = np.clip(atlas_xy[:, 1], view.tile_xy[1] + 1, view.tile_xy[1] + tile_height - 2)
        uv[selection, 0] = atlas_xy[:, 0] / atlas_w
        uv[selection, 1] = 1.0 - atlas_xy[:, 1] / atlas_h
    material = PBRMaterial(baseColorTexture=atlas, metallicFactor=0.0, roughnessFactor=1.0)
    output = trimesh.Trimesh(
        vertices=output_vertices, faces=output_faces, process=False,
    )
    output.visual = TextureVisuals(uv=uv, material=material)
    fallback_mesh = None
    if np.any(fallback):
        fallback_colors = np.column_stack((colors, np.full(len(colors), 255, dtype=np.uint8)))
        fallback_mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces[fallback],
            vertex_colors=fallback_colors, process=False,
        )
        fallback_mesh.remove_unreferenced_vertices()
    return output, fallback_mesh, {
        "source_triangle_count": int(len(faces)),
        "direct_photo_triangle_count": int(np.count_nonzero(supported)),
        "surfel_rgb_fallback_triangle_count": int(np.count_nonzero(fallback)),
        "free_space_rejected_triangle_count": int(np.count_nonzero(contradictory)),
        "single_free_space_contradiction_triangle_count": int(np.count_nonzero(free_contradiction_count == 1)),
        "output_triangle_count": int(len(output_faces) + (len(fallback_mesh.faces) if fallback_mesh is not None else 0)),
        "output_vertex_count_with_uv_seams": int(len(output_vertices) + (len(fallback_mesh.vertices) if fallback_mesh is not None else 0)),
        "photo_mesh_triangle_count": int(len(output_faces)),
        "photo_mesh_vertex_count_with_uv_seams": int(len(output_vertices)),
        "fallback_mesh_triangle_count": int(len(fallback_mesh.faces) if fallback_mesh is not None else 0),
        "fallback_mesh_vertex_count": int(len(fallback_mesh.vertices) if fallback_mesh is not None else 0),
        "texel_count": int(atlas_w * atlas_h),
    }


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-width", type=int, default=1024)
    parser.add_argument("--tile-height", type=int, default=576)
    parser.add_argument("--poisson-depth", type=int, default=9)
    parser.add_argument("--poisson-threads", type=int, default=1)
    parser.add_argument("--owners", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3])
    return parser.parse_args()


def main() -> None:
    args = parse()
    if args.tile_width * 8 > MAX_ATLAS_SIDE or args.tile_height * 6 > MAX_ATLAS_SIDE:
        raise ValueError("atlas exceeds bounded texture side")
    if args.poisson_depth < 8 or args.poisson_depth > 9 or args.poisson_threads != 1:
        raise ValueError("offline build is bounded to Poisson depth 8-9 and one thread")
    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and (output_dir / "mesh_manifest.json").exists():
        raise ValueError("refusing to overwrite a completed high-detail artifact")
    output_dir.mkdir(parents=True, exist_ok=True)
    root = args.repo_root.resolve()
    recipe = read_json(args.binding)
    descriptor_path = Path(recipe["source_descriptor"]["path"])
    if digest(descriptor_path) != recipe["source_descriptor"]["sha256"]:
        raise ValueError("current assembly descriptor differs from frozen binding")
    descriptor = read_json(descriptor_path)
    if descriptor["accepted_for_canonical_use"] is not False:
        raise ValueError("review assembly source changed authority")
    low_path = descriptor_path.parent.parent / descriptor["artifact"]["relative_path"]
    if (digest(low_path) != descriptor["artifact"]["sha256"]
            or low_path.stat().st_size != descriptor["artifact"]["size_bytes"]):
        raise ValueError("frozen low-detail RGB mesh bytes changed")
    low_manifest_path = Path(recipe["mesh_manifest"]["path"])
    if digest(low_manifest_path) != recipe["mesh_manifest"]["sha256"]:
        raise ValueError("frozen low-detail mesh manifest changed")
    source_manifest_path = Path(recipe["reintegration"]["path"])
    if digest(source_manifest_path) != recipe["reintegration"]["sha256"]:
        raise ValueError("multiroom reintegration manifest differs from binding")
    multiroom = read_json(source_manifest_path)
    source_npz = source_manifest_path.parent / multiroom["artifacts"]["surfels_npz"]["path"]
    if digest(source_npz) != multiroom["artifacts"]["surfels_npz"]["sha256"]:
        raise ValueError("ownership-filtered surfels changed")
    prior_path = Path(recipe["family_kitchen"]["path"])
    registration_path = Path(recipe["endpoint_registration"]["path"])
    if digest(prior_path) != recipe["family_kitchen"]["sha256"] or digest(registration_path) != recipe["endpoint_registration"]["sha256"]:
        raise ValueError("recorded room transform chain changed")
    prior_join, registration = read_json(prior_path), read_json(registration_path)
    if multiroom.get("phone_walk_only") is not True or multiroom.get("static_camera_points_included") is not False:
        raise ValueError("source is not phone-only review evidence")
    if multiroom["output_coordinate_frame"] != descriptor["assembly_gauge"]:
        raise ValueError("source and assembly gauge differ")
    with np.load(source_npz, allow_pickle=False) as arrays:
        points = np.asarray(arrays["points"], dtype=np.float32).copy()
        colors = np.asarray(arrays["colors"], dtype=np.uint8).copy()
        owners = np.asarray(arrays["owner_room_id"], dtype=np.uint8).copy()
        view_count = np.asarray(arrays["view_count"], dtype=np.uint16).copy()
        cameras = {
            owner: np.asarray(arrays[{1:"fixed_camera_positions",2:"moving_camera_positions",3:"living_camera_positions"}[owner]], dtype=np.float64).copy()
            for owner in args.owners
        }
    scene = trimesh.Scene(metadata={"review_only": True, "assembly_gauge": descriptor["assembly_gauge"]})
    results = {}
    for owner in args.owners:
        started = time.monotonic()
        room = OWNER_NAMES[owner]
        print(f"building owner {owner} ({room})", flush=True)
        selected = (owners == owner) & (view_count >= 2) & (points[:, 1] <= 1.85)
        if np.count_nonzero(selected) < 10000:
            raise ValueError(f"owner {owner} lacks measured support")
        atlas = Image.new("RGB", (args.tile_width * 8, args.tile_height * 6), (255, 255, 255))
        views, image_sources = load_views(
            owner, multiroom=multiroom, prior_join=prior_join, registration=registration,
            recipe=recipe, repo_root=root, tile_width=args.tile_width,
            tile_height=args.tile_height, atlas=atlas,
        )
        mesh, geometric = geometry_from_surfels(
            points[selected], colors[selected], cameras[owner],
            depth=args.poisson_depth, poisson_threads=args.poisson_threads,
        )
        output, fallback_mesh, texture = texture_mesh(
            mesh, views, atlas, tile_width=args.tile_width, tile_height=args.tile_height,
        )
        name = f"pcf_surface_{room}"
        scene.graph.update(frame_from=scene.graph.base_frame, frame_to=name, matrix=np.eye(4))
        output.metadata.update({"owner_room_id": owner, "room": room, "source_role": "phone_only_review"})
        scene.add_geometry(output, geom_name=f"{name}_photo", node_name=f"{name}_photo", parent_node_name=name)
        if fallback_mesh is not None:
            fallback_name = f"{name}_rgb_fallback"
            fallback_mesh.metadata.update({"owner_room_id": owner, "room": room, "source_role": "measured_surfel_rgb_fallback"})
            scene.add_geometry(fallback_mesh, geom_name=fallback_name, node_name=fallback_name, parent_node_name=name)
        atlas_path = output_dir / f"{name}_atlas.jpg"
        atlas.save(atlas_path, "JPEG", quality=91, subsampling=0, optimize=True)
        # Re-open so the GLB exporter embeds the already bounded JPEG encoding.
        output.visual.material.baseColorTexture = Image.open(atlas_path)
        results[str(owner)] = {
            "room": room, "mesh_name": name,
            "fallback_mesh_name": f"{name}_rgb_fallback" if fallback_mesh is not None else None,
            "geometry": geometric, "texture": texture,
            "atlas": {"path": str(atlas_path), "sha256": digest(atlas_path),
                      "size_bytes": atlas_path.stat().st_size, "width": atlas.width,
                      "height": atlas.height},
            "image_sources": image_sources,
            "elapsed_s": time.monotonic() - started,
        }
        print(f"owner {owner}: {texture['output_triangle_count']} tris, {texture['direct_photo_triangle_count']} photo projected", flush=True)
    glb_path = output_dir / "multiroom_surface_mesh_high_detail.glb"
    glb_path.write_bytes(trimesh.exchange.gltf.export_glb(scene))
    report = {
        "contract": "noesis.pcf.high_detail_review_surface_mesh",
        "contract_version": 1,
        "status": "review_only", "accepted_for_canonical_use": False,
        "coordinate_frame": descriptor["assembly_gauge"], "units": "meters",
        "source": {
            "binding_sha256": digest(args.binding),
            "source_descriptor_sha256": recipe["source_descriptor"]["sha256"],
            "low_glb_sha256": descriptor["artifact"]["sha256"],
            "low_mesh_manifest_sha256": recipe["mesh_manifest"]["sha256"],
            "source_multiroom_manifest_sha256": recipe["reintegration"]["sha256"],
            "source_surfels_npz_sha256": multiroom["artifacts"]["surfels_npz"]["sha256"],
            "family_kitchen_manifest_sha256": recipe["family_kitchen"]["sha256"],
            "endpoint_registration_sha256": recipe["endpoint_registration"]["sha256"],
            "static_camera_points_included": False,
            "phone_walk_only": True,
            "registration_status": multiroom["registration"]["status"],
        },
        "method": "per_owner_2_5cm_measured_surfel_poisson_depth9_with_depth_visible_prepared_phone_rgb_atlases",
        "parameters": {
            "poisson_depth": args.poisson_depth, "poisson_threads": args.poisson_threads,
            "ceiling_cutaway_m": 1.85, "normal_radius_m": 0.09,
            "minimum_source_surfel_view_count": 2,
            "vertex_support_distance_m": 0.075, "triangle_centroid_support_distance_m": 0.065,
            "maximum_triangle_edge_m": 0.15, "depth_tolerance_m": "max(0.05, 0.03*source_depth)",
            "minimum_source_confidence": 0.20,
            "minimum_distinct_view_free_space_contradictions_to_reject": 2,
            "atlas_tile_width": args.tile_width, "atlas_tile_height": args.tile_height,
        },
        "owners": results,
        "output": {"glb": str(glb_path), "sha256": digest(glb_path),
                   "size_bytes": glb_path.stat().st_size,
                   "triangle_count": int(sum(r["texture"]["output_triangle_count"] for r in results.values())),
                   "vertex_count": int(sum(r["texture"]["output_vertex_count_with_uv_seams"] for r in results.values()))},
        "limitations": [
            "derived presentation surface, not measured geometry authority",
            "2.5 cm surfel support and 280x504 RGB-D depth limit geometric detail",
            "phone JPEG atlases retain higher-frequency RGB but do not add true geometric detail",
            "image-plane registration and uncertain phone poses can leave texture seams or ghosting",
            "living registration rejected; room-to-room placement remains review only",
            "global structural alignment is applied by presentation consumer",
        ],
    }
    report_path = output_dir / "mesh_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if set(args.owners) == {1, 2, 3}:
        compact = {
            "contract": "noesis.pcf.tracking_review_high_detail",
            "contract_version": 1,
            "status": "review_only", "accepted_for_canonical_use": False,
            "assembly_id": descriptor["assembly_id"],
            "assembly_gauge": descriptor["assembly_gauge"],
            "coordinate_frame": descriptor["coordinate_frame"], "units": "meters",
            "source": {
                "binding_sha256": digest(args.binding),
                "detail_manifest_sha256": digest(report_path),
                "low_glb_sha256": descriptor["artifact"]["sha256"],
                "low_mesh_manifest_sha256": recipe["mesh_manifest"]["sha256"],
                "reintegration_manifest_sha256": recipe["reintegration"]["sha256"],
                "surfels_npz_sha256": multiroom["artifacts"]["surfels_npz"]["sha256"],
            },
            "output": {
                "path": glb_path.name, "sha256": report["output"]["sha256"],
                "size_bytes": report["output"]["size_bytes"],
                "vertex_count": report["output"]["vertex_count"],
                "triangle_count": report["output"]["triangle_count"],
                "owner_triangle_counts": {
                    owner: result["texture"]["output_triangle_count"]
                    for owner, result in results.items()
                },
            },
            "owners": {
                owner: {
                    "room": {"1": "family-room", "2": "kitchen", "3": "living-room"}[owner],
                    "mesh_name": result["mesh_name"],
                    "texture_path": Path(result["atlas"]["path"]).name,
                    "texture_sha256": result["atlas"]["sha256"],
                    "texture_size_bytes": result["atlas"]["size_bytes"],
                    "direct_photo_triangle_count": result["texture"]["direct_photo_triangle_count"],
                    "fallback_triangle_count": result["texture"]["surfel_rgb_fallback_triangle_count"],
                    "accepted_image_count": sum(source["image_status"] == "accepted" for source in result["image_sources"]),
                }
                for owner, result in results.items()
            },
        }
        (output_dir / "tracking_high.json").write_text(
            json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report["output"], indent=2), flush=True)


if __name__ == "__main__":
    main()
