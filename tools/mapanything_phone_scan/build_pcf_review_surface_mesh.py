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
from typing import Any, Mapping, Sequence

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

from tools.mapanything_phone_scan.reintegrate_pcf_rooms import (  # noqa: E402
    ViewEvidence,
    _depth_visibility_evidence,
)


OWNER_CAMERAS = {
    1: "fixed_camera_positions",
    2: "moving_camera_positions",
    3: "living_camera_positions",
    4: "connector_camera_positions",
}
OWNER_NAMES = {1: "family", 2: "kitchen", 3: "living", 4: "connector"}


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


def _owner_visibility_rows(
    manifest: Mapping[str, Any],
    owner: int,
    *,
    manifest_dir: Path,
    visited: set[Path] | None = None,
) -> list[Mapping[str, Any]]:
    """Resolve only the manifest-named RGB-D view rows for one owner."""

    seen = set() if visited is None else visited
    rows: list[Mapping[str, Any]] = []
    sources = manifest.get("sources")
    sources = sources if isinstance(sources, Mapping) else {}
    if owner == 1:
        blocks = [manifest.get("fixed_room")]
    elif owner == 2:
        blocks = [manifest.get("moving_room")]
    elif owner == 3:
        blocks = [sources.get("living")]
    else:
        blocks = [sources.get("connector")]
    for block in blocks:
        if isinstance(block, Mapping):
            values = block.get("raw_views") or block.get("visibility_views")
            if isinstance(values, list):
                rows.extend(item for item in values if isinstance(item, Mapping))
    nested = manifest.get("existing_kitchen_family_authority")
    if owner in (1, 2) and isinstance(nested, Mapping):
        nested_path = Path(str(nested.get("manifest") or ""))
        if not nested_path.is_absolute():
            nested_path = (manifest_dir / nested_path).resolve()
        else:
            nested_path = nested_path.resolve()
        if nested_path not in seen and nested_path.is_file():
            seen.add(nested_path)
            rows.extend(
                _owner_visibility_rows(
                    _json(nested_path),
                    owner,
                    manifest_dir=nested_path.parent,
                    visited=seen,
                )
            )
    unique: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("path") or ""), int(row.get("view_index", -1)))
        unique.setdefault(key, row)
    return list(unique.values())


def _load_visibility_evidence(
    rows: Sequence[Mapping[str, Any]],
) -> list[ViewEvidence]:
    evidence: list[ViewEvidence] = []
    for item in rows:
        path = Path(str(item.get("path") or ""))
        try:
            view_index = int(item["view_index"])
            transform = np.asarray(
                item["effective_world_from_local_row_major"], dtype=np.float64
            )
            with np.load(path, allow_pickle=False) as row:
                depth = np.asarray(row["depth_z"], dtype=np.float32).copy()
                mask = np.asarray(row["mask"], dtype=bool).copy()
                intrinsics = np.asarray(row["intrinsics"], dtype=np.float64).copy()
                camera_pose = np.asarray(row["camera_pose"], dtype=np.float64)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"surface source view visibility evidence cannot be read: {path}"
            ) from exc
        if (
            transform.shape != (4, 4)
            or camera_pose.shape != (4, 4)
            or intrinsics.shape != (3, 3)
            or depth.ndim != 2
            or mask.shape != depth.shape
            or not np.isfinite(transform).all()
            or not np.isfinite(camera_pose).all()
            or not np.isfinite(intrinsics).all()
        ):
            raise ValueError(f"surface source view visibility evidence is malformed: {path}")
        evidence.append(
            ViewEvidence(
                view_index=view_index,
                intrinsics=intrinsics,
                depth_z=depth,
                mask=mask,
                camera_to_world=transform @ camera_pose,
            )
        )
    return evidence


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


def _triangle_support_mask(
    mesh: o3d.geometry.TriangleMesh,
    cloud_points: np.ndarray,
    view_evidence: Sequence[ViewEvidence],
    support_distance_m: float,
    *,
    minimum_depth_m: float = 0.05,
    maximum_depth_m: float = 12.0,
    depth_tolerance_m: float = 0.05,
    depth_tolerance_fraction: float = 0.03,
) -> tuple[np.ndarray, int, int]:
    """Keep triangles supported by observed points and source-camera rays.

    Poisson can span a gap while both endpoints remain close to the input
    cloud.  Testing triangle centroids and edge midpoints catches those
    unsupported bridges.  Every sample must be supported by at least one
    retained RGB-D view under its explicit mask/depth tolerance; a free-space
    contradiction rejects the triangle even when its endpoints are supported.
    """

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    if not len(triangles):
        return np.zeros(0, dtype=bool), 0, 0
    centroids = vertices[triangles].mean(axis=1)
    support_tree = cKDTree(np.asarray(cloud_points, dtype=np.float64))
    support_distance, _ = support_tree.query(centroids, k=1, workers=1)
    supported = np.isfinite(support_distance) & (
        support_distance <= float(support_distance_m)
    )
    if not view_evidence:
        return np.zeros(len(triangles), dtype=bool), int(
            np.count_nonzero(~supported)
        ), int(len(triangles))
    triangle_vertices = vertices[triangles]
    samples = np.concatenate(
        (
            centroids[:, None, :],
            (triangle_vertices[:, 0] + triangle_vertices[:, 1])[:, None, :] / 2.0,
            (triangle_vertices[:, 1] + triangle_vertices[:, 2])[:, None, :] / 2.0,
            (triangle_vertices[:, 2] + triangle_vertices[:, 0])[:, None, :] / 2.0,
        ),
        axis=1,
    ).reshape((-1, 3))
    sample_supported, _, free_space = _depth_visibility_evidence(
        samples,
        view_evidence,
        minimum_depth_m=minimum_depth_m,
        maximum_depth_m=maximum_depth_m,
        absolute_tolerance_m=depth_tolerance_m,
        relative_tolerance=depth_tolerance_fraction,
    )
    sample_supported = sample_supported.reshape((-1, 4))
    free_space = free_space.reshape((-1, 4))
    visible = sample_supported[:, 0] & (np.sum(sample_supported, axis=1) >= 2)
    visible &= ~np.any(free_space, axis=1)
    retained = supported & visible
    return (
        retained,
        int(np.count_nonzero(~supported)),
        int(np.count_nonzero(supported & ~visible)),
    )


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
    poisson_threads: int = 1,
    view_evidence: Sequence[ViewEvidence] = (),
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
        n_threads=poisson_threads,
    )
    poisson_seconds = time.monotonic() - started
    density_values = np.asarray(densities)
    density_threshold = float(np.percentile(density_values, density_percentile))
    vertices = np.asarray(mesh.vertices)
    vertex_support_distance, _ = cKDTree(np.asarray(cloud.points)).query(
        vertices,
        k=1,
        workers=1,
    )
    cloud_points = np.asarray(cloud.points)
    lower = np.min(cloud_points, axis=0) - support_distance_m
    upper = np.max(cloud_points, axis=0) + support_distance_m
    upper[1] = min(upper[1], ceiling_cutaway_m + 0.03)
    supported = (
        (density_values >= density_threshold)
        & (vertex_support_distance <= support_distance_m)
        & np.all(vertices >= lower[None, :], axis=1)
        & np.all(vertices <= upper[None, :], axis=1)
    )
    rejected_vertices = int(np.count_nonzero(~supported))
    mesh.remove_vertices_by_mask(~supported)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    triangle_support, rejected_triangles, ray_rejected_triangles = (
        _triangle_support_mask(
            mesh,
            np.asarray(cloud.points),
            view_evidence,
            support_distance_m,
        )
    )
    mesh.remove_triangles_by_mask(~triangle_support)
    mesh.remove_unreferenced_vertices()
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
        "support_rejected_triangle_count": rejected_triangles,
        "ray_visibility_rejected_triangle_count": ray_rejected_triangles,
        "small_component_removed_triangle_count": removed_component_triangles,
        "poisson_seconds": poisson_seconds,
        "poisson_threads": int(poisson_threads),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_npz = args.surfels_npz.resolve()
    source_manifest = args.surfels_manifest.resolve()
    manifest = _json(source_manifest)
    artifact_entry = manifest.get("artifacts", {}).get("surfels_npz")
    if isinstance(artifact_entry, Mapping):
        expected_sha = artifact_entry.get("sha256")
    elif isinstance(artifact_entry, str):
        # v1 reintegration manifests emit artifact names in ``artifacts`` and
        # their digests in the sibling ``artifact_sha256`` object.
        expected_sha = manifest.get("artifact_sha256", {}).get("surfels_npz")
    else:
        expected_sha = None
    actual_sha = _sha256(source_npz)
    if expected_sha != actual_sha:
        raise ValueError("surfel NPZ digest does not match its reintegration manifest")
    accepted_for_canonical_use = manifest.get("accepted_for_canonical_use")
    if accepted_for_canonical_use is None:
        accepted_for_canonical_use = manifest.get("registration", {}).get(
            "accepted_for_canonical_use"
        )
    if accepted_for_canonical_use is not False:
        raise ValueError("surface meshing requires an explicitly non-canonical source")
    owner_view_evidence = {
        owner: _load_visibility_evidence(
            _owner_visibility_rows(
                manifest,
                owner,
                manifest_dir=source_manifest.parent,
            )
        )
        for owner in OWNER_CAMERAS
    }
    support_view_mask_fields: list[str] = []

    with np.load(source_npz, allow_pickle=False) as arrays:
        required = {"points", "colors", "owner_room_id"}
        if not required.issubset(arrays.files):
            raise ValueError(f"surfel NPZ is missing {sorted(required - set(arrays.files))}")
        all_points = np.asarray(arrays["points"], dtype=np.float64)
        all_colors = np.asarray(arrays["colors"], dtype=np.uint8)
        owners = np.asarray(arrays["owner_room_id"], dtype=np.uint8)
        cameras: dict[int, np.ndarray] = {}
        for owner, field in OWNER_CAMERAS.items():
            if np.any(owners == owner):
                if field not in arrays.files:
                    raise ValueError(f"surfel NPZ is missing camera provenance {field}")
                cameras[owner] = np.asarray(arrays[field], dtype=np.float64)
        support_view_count = (
            np.asarray(arrays["view_count"], dtype=np.uint16).copy()
            if "view_count" in arrays.files
            else None
        )
        geometry_status = (
            np.asarray(arrays["geometry_status"], dtype=np.uint8).copy()
            if "geometry_status" in arrays.files
            else np.zeros(len(all_points), dtype=np.uint8)
        )
        support_view_mask_fields = [
            field
            for field in (
                "fixed_view_mask_words",
                "moving_view_mask_words",
                "living_view_mask_words",
                "connector_view_mask_words",
            )
            if field in arrays.files
        ]

    finite = np.all(np.isfinite(all_points), axis=1)
    cutaway = all_points[:, 1] <= args.ceiling_cutaway_m
    scene = trimesh.Scene(metadata={"review_only": True})
    owner_metrics: dict[str, Any] = {}
    total_vertices = 0
    total_triangles = 0
    class_tints = {
        "observed": None,
        "uncertain": np.asarray([255, 180, 0], dtype=np.float64),
        "unknown": np.asarray([150, 150, 150], dtype=np.float64),
    }
    for owner in sorted(OWNER_CAMERAS):
        selected = finite & cutaway & (owners == owner)
        if not np.any(selected):
            continue
        selected_points = all_points[selected]
        selected_colors = all_colors[selected]
        selected_status = geometry_status[selected]
        support_counts = (
            support_view_count[selected]
            if support_view_count is not None
            else np.zeros(np.count_nonzero(selected), dtype=np.uint16)
        )
        class_masks = {
            "observed": (support_counts >= 2) & (selected_status == 0),
            "uncertain": (selected_status == 1)
            | ((selected_status == 0) & (support_counts == 1)),
            "unknown": (selected_status == 2)
            | ((selected_status == 0) & (support_counts == 0)),
        }
        metrics: dict[str, Any] = {
            "input_point_count": int(len(selected_points)),
            "output_vertex_count": 0,
            "output_triangle_count": 0,
            "support_class_counts": {
                name: int(np.count_nonzero(mask))
                for name, mask in class_masks.items()
            },
            "geometry_uncertain_count": int(np.count_nonzero(selected_status == 1)),
            "classes": {},
        }
        for class_name, class_mask in class_masks.items():
            if not np.any(class_mask):
                continue
            class_points = selected_points[class_mask]
            class_colors = selected_colors[class_mask]
            tinted_colors = class_colors.astype(np.float64)
            tint = class_tints[class_name]
            if tint is not None:
                tinted_colors = 0.55 * tinted_colors + 0.45 * tint[None, :]
            class_metrics: dict[str, Any] = {
                "input_point_count": int(len(class_points)),
                "support_class": class_name,
            }
            class_mesh = None
            if len(class_points) >= 1_000:
                class_mesh, reconstructed = _reconstruct_owner(
                    class_points,
                    np.rint(tinted_colors).astype(np.uint8),
                    cameras[owner],
                    voxel_size_m=args.voxel_size_m,
                    normal_radius_m=args.normal_radius_m,
                    support_distance_m=args.support_distance_m,
                    ceiling_cutaway_m=args.ceiling_cutaway_m,
                    poisson_depth=args.poisson_depth,
                    density_percentile=args.density_percentile,
                    minimum_component_triangles=args.minimum_component_triangles,
                    poisson_threads=args.poisson_threads,
                    view_evidence=owner_view_evidence.get(owner, ()),
                )
                class_metrics.update(reconstructed)
            if class_mesh is not None and class_mesh.has_triangles():
                vertex_colors = np.clip(
                    np.rint(np.asarray(class_mesh.vertex_colors) * 255.0),
                    0,
                    255,
                ).astype(np.uint8)
                alpha = np.full((len(vertex_colors), 1), 255, dtype=np.uint8)
                trimesh_mesh = trimesh.Trimesh(
                    vertices=np.asarray(class_mesh.vertices),
                    faces=np.asarray(class_mesh.triangles),
                    vertex_normals=np.asarray(class_mesh.vertex_normals),
                    vertex_colors=np.concatenate([vertex_colors, alpha], axis=1),
                    process=False,
                    metadata={
                        "owner_room_id": owner,
                        "room": OWNER_NAMES[owner],
                        "support_class": class_name,
                    },
                )
                node_name = f"pcf_surface_{OWNER_NAMES[owner]}_{class_name}"
                scene.add_geometry(
                    trimesh_mesh,
                    geom_name=node_name,
                    node_name=node_name,
                )
                class_metrics["geometry_mode"] = "screened_poisson_surface"
                metrics["output_vertex_count"] += len(trimesh_mesh.vertices)
                metrics["output_triangle_count"] += len(trimesh_mesh.faces)
                total_vertices += len(trimesh_mesh.vertices)
                total_triangles += len(trimesh_mesh.faces)
            else:
                # Keep small or ray-rejected uncertainty visible as points so
                # it cannot disappear or be presented as an observed surface.
                point_cloud = trimesh.points.PointCloud(
                    class_points,
                    colors=np.column_stack(
                        (
                            np.rint(tinted_colors).clip(0, 255).astype(np.uint8),
                            np.full(len(class_points), 230, dtype=np.uint8),
                        )
                    ),
                )
                node_name = f"pcf_surface_{OWNER_NAMES[owner]}_{class_name}_points"
                point_cloud.metadata.update(
                    {
                        "owner_room_id": owner,
                        "room": OWNER_NAMES[owner],
                        "support_class": class_name,
                    }
                )
                scene.add_geometry(
                    point_cloud,
                    geom_name=node_name,
                    node_name=node_name,
                )
                class_metrics["geometry_mode"] = "review_points"
                class_metrics["output_vertex_count"] = int(len(class_points))
                metrics["output_vertex_count"] += len(class_points)
                total_vertices += len(class_points)
            metrics["classes"][class_name] = class_metrics
        owner_metrics[str(owner)] = {"room": OWNER_NAMES[owner], **metrics}

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
        "method": "per_owner_support_class_split_screened_poisson_with_ray_triangle_support_trim",
        "frame_identity": {
            "coordinate_frame": manifest.get("output_coordinate_frame"),
            "registration": manifest.get("registration"),
            "support_view_count_field": "view_count" if support_view_count is not None else None,
            "support_view_mask_fields": support_view_mask_fields,
            "geometry_status_field": "geometry_status",
            "support_geometry_name_pattern": "pcf_surface_{room}_{support_class}",
            "uncertain_point_tint_rgb": [255, 180, 0],
            "unknown_point_tint_rgb": [150, 150, 150],
        },
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
            "poisson_threads": args.poisson_threads,
            "room_owners_reconstructed_independently": True,
            "triangle_support_and_camera_ray_checks": True,
            "support_classes_rendered_separately": True,
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
    parser.add_argument("--poisson-threads", type=int, default=1)
    return parser


def main() -> None:
    report = build(_parser().parse_args())
    print(json.dumps(report["output"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
