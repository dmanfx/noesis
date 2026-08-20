#!/usr/bin/env python3
"""Build a cutaway RGB surface mesh from a retained multi-room PCF.

The operation is presentation-only. It preserves the PCF assembly gauge and
does not alter room registration, camera calibration, or canonical world state.
Each room owner is reconstructed independently so uncertain cross-room joins do
not create triangles between unrelated surfaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree


OWNER_CAMERAS = {
    1: "fixed_camera_positions",
    2: "moving_camera_positions",
    3: "living_camera_positions",
}
OWNER_NAMES = {1: "family", 2: "kitchen", 3: "living"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _orient_normals_to_nearest_camera(
    cloud: o3d.geometry.PointCloud,
    cameras: np.ndarray,
) -> None:
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    camera_values = np.asarray(cameras, dtype=np.float64)
    if camera_values.ndim != 2 or camera_values.shape[1] != 3:
        raise ValueError("camera positions must have shape (N, 3)")
    for start in range(0, len(points), 20_000):
        end = min(start + 20_000, len(points))
        block = points[start:end]
        distance_sq = np.sum(
            (block[:, None, :] - camera_values[None, :, :]) ** 2,
            axis=2,
        )
        nearest = camera_values[np.argmin(distance_sq, axis=1)]
        toward_camera = nearest - block
        flip = np.einsum("ij,ij->i", normals[start:end], toward_camera) < 0.0
        normals[start:end][flip] *= -1.0


def _remove_tiny_components(
    mesh: o3d.geometry.TriangleMesh,
    minimum_triangles: int,
) -> int:
    if not mesh.has_triangles():
        return 0
    labels, counts, _ = mesh.cluster_connected_triangles()
    label_values = np.asarray(labels)
    count_values = np.asarray(counts)
    remove = count_values[label_values] < minimum_triangles
    removed = int(np.count_nonzero(remove))
    mesh.remove_triangles_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    return removed


def _reconstruct_owner(
    points: np.ndarray,
    colors: np.ndarray,
    cameras: np.ndarray,
    *,
    voxel_size_m: float,
    normal_radius_m: float,
    support_distance_m: float,
    ceiling_cutaway_m: float,
    poisson_depth: int,
    density_percentile: float,
    minimum_component_triangles: int,
) -> tuple[o3d.geometry.TriangleMesh, dict[str, Any]]:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    cloud = cloud.voxel_down_sample(voxel_size_m)
    if len(cloud.points) < 1_000:
        raise ValueError("owner cloud is too small for surface reconstruction")
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius_m,
            max_nn=48,
        )
    )
    _orient_normals_to_nearest_camera(cloud, cameras)

    started = time.monotonic()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud,
        depth=poisson_depth,
        scale=1.05,
        linear_fit=True,
        n_threads=0,
    )
    poisson_seconds = time.monotonic() - started
    density_values = np.asarray(densities)
    density_threshold = float(np.percentile(density_values, density_percentile))
    vertices = np.asarray(mesh.vertices)
    support_distance, _ = cKDTree(np.asarray(cloud.points)).query(
        vertices,
        k=1,
        workers=-1,
    )
    cloud_points = np.asarray(cloud.points)
    lower = np.min(cloud_points, axis=0) - support_distance_m
    upper = np.max(cloud_points, axis=0) + support_distance_m
    upper[1] = min(upper[1], ceiling_cutaway_m + 0.03)
    supported = (
        (density_values >= density_threshold)
        & (support_distance <= support_distance_m)
        & np.all(vertices >= lower[None, :], axis=1)
        & np.all(vertices <= upper[None, :], axis=1)
    )
    rejected_vertices = int(np.count_nonzero(~supported))
    mesh.remove_vertices_by_mask(~supported)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    removed_component_triangles = _remove_tiny_components(
        mesh,
        minimum_component_triangles,
    )
    mesh.compute_vertex_normals()
    if not mesh.has_vertex_colors():
        raise ValueError("Poisson reconstruction did not retain RGB vertex colors")
    return mesh, {
        "input_point_count": int(len(points)),
        "downsampled_point_count": int(len(cloud.points)),
        "output_vertex_count": int(len(mesh.vertices)),
        "output_triangle_count": int(len(mesh.triangles)),
        "density_threshold": density_threshold,
        "support_rejected_vertex_count": rejected_vertices,
        "small_component_removed_triangle_count": removed_component_triangles,
        "poisson_seconds": poisson_seconds,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_npz = args.surfels_npz.resolve()
    source_manifest = args.surfels_manifest.resolve()
    manifest = _json(source_manifest)
    expected_sha = manifest.get("artifacts", {}).get("surfels_npz", {}).get("sha256")
    actual_sha = _sha256(source_npz)
    if expected_sha != actual_sha:
        raise ValueError("surfel NPZ digest does not match its reintegration manifest")
    if manifest.get("accepted_for_canonical_use") is not False:
        raise ValueError("surface meshing requires an explicitly non-canonical source")

    with np.load(source_npz, allow_pickle=False) as arrays:
        required = {"points", "colors", "owner_room_id", *OWNER_CAMERAS.values()}
        if not required.issubset(arrays.files):
            raise ValueError(f"surfel NPZ is missing {sorted(required - set(arrays.files))}")
        all_points = np.asarray(arrays["points"], dtype=np.float64)
        all_colors = np.asarray(arrays["colors"], dtype=np.uint8)
        owners = np.asarray(arrays["owner_room_id"], dtype=np.uint8)
        cameras = {
            owner: np.asarray(arrays[field], dtype=np.float64)
            for owner, field in OWNER_CAMERAS.items()
        }

    finite = np.all(np.isfinite(all_points), axis=1)
    cutaway = all_points[:, 1] <= args.ceiling_cutaway_m
    scene = trimesh.Scene(metadata={"review_only": True})
    owner_metrics: dict[str, Any] = {}
    total_vertices = 0
    total_triangles = 0
    for owner in sorted(OWNER_CAMERAS):
        selected = finite & cutaway & (owners == owner)
        mesh, metrics = _reconstruct_owner(
            all_points[selected],
            all_colors[selected],
            cameras[owner],
            voxel_size_m=args.voxel_size_m,
            normal_radius_m=args.normal_radius_m,
            support_distance_m=args.support_distance_m,
            ceiling_cutaway_m=args.ceiling_cutaway_m,
            poisson_depth=args.poisson_depth,
            density_percentile=args.density_percentile,
            minimum_component_triangles=args.minimum_component_triangles,
        )
        vertex_colors = np.clip(
            np.rint(np.asarray(mesh.vertex_colors) * 255.0),
            0,
            255,
        ).astype(np.uint8)
        alpha = np.full((len(vertex_colors), 1), 255, dtype=np.uint8)
        trimesh_mesh = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.triangles),
            vertex_normals=np.asarray(mesh.vertex_normals),
            vertex_colors=np.concatenate([vertex_colors, alpha], axis=1),
            process=False,
            metadata={"owner_room_id": owner, "room": OWNER_NAMES[owner]},
        )
        scene.add_geometry(
            trimesh_mesh,
            geom_name=f"pcf_surface_{OWNER_NAMES[owner]}",
            node_name=f"pcf_surface_{OWNER_NAMES[owner]}",
        )
        owner_metrics[str(owner)] = {"room": OWNER_NAMES[owner], **metrics}
        total_vertices += len(trimesh_mesh.vertices)
        total_triangles += len(trimesh_mesh.faces)

    args.output_glb.parent.mkdir(parents=True, exist_ok=True)
    glb_bytes = trimesh.exchange.gltf.export_glb(scene)
    args.output_glb.write_bytes(glb_bytes)
    output_sha = _sha256(args.output_glb)
    report = {
        "contract": "noesis.pcf.review_surface_mesh",
        "contract_version": 1,
        "status": "review_only",
        "accepted_for_canonical_use": False,
        "coordinate_frame": manifest.get("output_coordinate_frame"),
        "method": "per_owner_camera_oriented_screened_poisson_with_support_trim",
        "source": {
            "surfels_npz": str(source_npz),
            "surfels_npz_sha256": actual_sha,
            "surfels_manifest": str(source_manifest),
            "surfels_manifest_sha256": _sha256(source_manifest),
            "registration_status": manifest.get("registration", {}).get("status"),
        },
        "parameters": {
            "voxel_size_m": args.voxel_size_m,
            "normal_radius_m": args.normal_radius_m,
            "support_distance_m": args.support_distance_m,
            "ceiling_cutaway_m": args.ceiling_cutaway_m,
            "poisson_depth": args.poisson_depth,
            "density_percentile": args.density_percentile,
            "minimum_component_triangles": args.minimum_component_triangles,
            "room_owners_reconstructed_independently": True,
        },
        "owners": owner_metrics,
        "output": {
            "glb": str(args.output_glb.resolve()),
            "sha256": output_sha,
            "size_bytes": args.output_glb.stat().st_size,
            "vertex_count": int(total_vertices),
            "triangle_count": int(total_triangles),
        },
        "limitations": [
            "surface_mesh_is_derived_review_presentation_not_measured_geometry_authority",
            "global_structural_alignment_is_applied_later_by_menon",
            "internal_room_registration_uncertainty_is_unchanged",
        ],
    }
    args.output_manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surfels-npz", type=Path, required=True)
    parser.add_argument("--surfels-manifest", type=Path, required=True)
    parser.add_argument("--output-glb", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--voxel-size-m", type=float, default=0.04)
    parser.add_argument("--normal-radius-m", type=float, default=0.14)
    parser.add_argument("--support-distance-m", type=float, default=0.10)
    parser.add_argument("--ceiling-cutaway-m", type=float, default=1.85)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--density-percentile", type=float, default=2.0)
    parser.add_argument("--minimum-component-triangles", type=int, default=64)
    return parser


def main() -> None:
    report = build(_parser().parse_args())
    print(json.dumps(report["output"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
