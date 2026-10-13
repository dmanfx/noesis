#!/usr/bin/env python3
"""Make an explicitly completed, closed presentation mesh beside measured evidence.

This uses the same accepted surfels and screened Poisson reconstruction, not
new depth or registration. Unmeasured Poisson extension and boundary caps are
classified per face. Closure is not a physical-accuracy or live-authority gate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

from tools.mapanything_phone_scan.build_pcf_review_surface_mesh import (
    OWNER_CAMERAS, OWNER_NAMES, _json, _load_visibility_evidence,
    _orient_normals_to_supported_cameras, _owner_visibility_rows, _sha256,
    _triangle_support_mask,
)


def edge_topology(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Check indexed topology without an unbounded pairwise intersection test."""
    if not len(faces):
        raise ValueError("cannot qualify an empty mesh")
    directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    signed = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    direction_sum = np.bincount(inverse, weights=signed)
    triangle = vertices[faces]
    area2 = np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0],
                                    triangle[:, 2] - triangle[:, 0]), axis=1)
    return {
        "boundary_edge_count": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edge_count": int(np.count_nonzero(counts > 2)),
        "inconsistent_winding_edge_count": int(np.count_nonzero((counts == 2) & (direction_sum != 0))),
        "degenerate_triangle_count": int(np.count_nonzero(area2 <= 1e-12)),
        "closed_two_face_edge_incidence": bool(np.all(counts == 2)),
        "self_intersections": "not_assessed_not_part_of_edge_incidence_claim",
    }


def _oriented_boundary_caps(mesh: o3d.geometry.TriangleMesh, radius: float):
    """Follow each boundary's incident face fan, including pinched vertices.

    One new inferred centre per simple loop avoids triangulation diagonals
    colliding with existing interior edges. Positions of source vertices and
    triangles are preserved. Concave/nonplanar caps are presentation closure,
    not a measured surface or a no-intersection qualification.
    """
    vertices, faces = np.asarray(mesh.vertices).copy(), np.asarray(mesh.triangles).copy()
    colors = np.asarray(mesh.vertex_colors).copy()
    directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("boundary fan requires an edge-manifold source")
    offsets = np.flatnonzero(counts[inverse] == 1)
    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(counts) - counts
    paired = starts[counts == 2]
    peer = np.full(len(directed), -1, dtype=np.int32)
    peer[order[paired]], peer[order[paired + 1]] = order[paired + 1], order[paired]
    following = offsets // 3 * 3 + (offsets % 3 + 1) % 3
    for _ in range(4096):
        crossing = peer[following] >= 0
        if not crossing.any():
            break
        other = peer[following[crossing]]
        following[crossing] = other // 3 * 3 + (other % 3 + 1) % 3
    else:
        raise ValueError("boundary incident-face traversal exceeds finite star bound")
    lookup = np.full(len(directed), -1, dtype=np.int32)
    lookup[offsets] = np.arange(len(offsets), dtype=np.int32)
    successors = lookup[following]
    if np.any(successors < 0) or not np.array_equal(np.sort(successors), np.arange(len(offsets))):
        raise ValueError("oriented boundary loops are not a one-to-one partition")
    visited = np.zeros(len(offsets), dtype=bool)
    loops = []
    for initial in range(len(offsets)):
        if visited[initial]:
            continue
        current, ring = initial, []
        while not visited[current]:
            visited[current] = True
            ring.append(int(directed[offsets[current], 0]))
            current = int(successors[current])
        if current != initial:
            raise ValueError("boundary traversal entered a different visited loop")
        # Split self-touching loops at repeated vertex identities. Their edges
        # remain exactly partitioned, with a distinct centre for each cycle.
        stack, location = [], {}
        for vertex in ring + ring[:1]:
            if vertex in location:
                index = location[vertex]
                cycle = stack[index:]
                if len(cycle) < 3:
                    raise ValueError("boundary cycle has fewer than three distinct vertices")
                loops.append(cycle)
                for old in stack[index:]:
                    location.pop(old)
                stack = stack[:index]
            location[vertex] = len(stack)
            stack.append(vertex)
    new_vertices, new_colors, new_faces = [], [], []
    for ring in loops:
        xyz = vertices[ring]
        center = xyz.mean(axis=0)
        if np.max(np.linalg.norm(xyz - center, axis=1)) > radius:
            raise ValueError("boundary loop exceeds explicitly bounded cap radius")
        shifted = center.copy()
        cross = np.cross(xyz - shifted, np.roll(xyz, -1, axis=0) - shifted)
        if np.any(np.linalg.norm(cross, axis=1) <= 1e-12):
            _, _, basis = np.linalg.svd(xyz - center, full_matrices=False)
            shifted = center + basis[-1] * 1e-4
        center_index = len(vertices) + len(new_vertices)
        new_vertices.append(shifted)
        new_colors.append(colors[ring].mean(axis=0) if len(colors) else np.full(3, .5))
        new_faces.extend([[ring[(index + 1) % len(ring)], vertex, center_index]
                          for index, vertex in enumerate(ring)])
    completed = o3d.geometry.TriangleMesh()
    completed.vertices = o3d.utility.Vector3dVector(np.concatenate((vertices, np.asarray(new_vertices).reshape(-1, 3))))
    completed.triangles = o3d.utility.Vector3iVector(np.concatenate((faces, np.asarray(new_faces, dtype=np.int32).reshape(-1, 3))))
    original_colors = colors if len(colors) else np.full(vertices.shape, .5)
    completed.vertex_colors = o3d.utility.Vector3dVector(np.concatenate((original_colors, np.asarray(new_colors).reshape(-1, 3))))
    return completed, len(loops)


def close_boundary(mesh: o3d.geometry.TriangleMesh, maximum_hole_radius_m: float) -> tuple[o3d.geometry.TriangleMesh, np.ndarray, dict[str, Any]]:
    """Bounded CPU cap operation; fail closed if holes remain or topology breaks."""
    if not np.isfinite(maximum_hole_radius_m) or not 0 < maximum_hole_radius_m <= 32:
        raise ValueError("hole radius must be finite and in (0, 32] meters")
    before_vertices = np.asarray(mesh.vertices).copy()
    before = edge_topology(before_vertices, np.asarray(mesh.triangles))
    if before["nonmanifold_edge_count"]:
        raise ValueError("cannot cap a nonmanifold source envelope")
    if not mesh.orient_triangles():
        raise ValueError("source envelope is not orientable")
    before_faces = np.asarray(mesh.triangles).copy()
    source = o3d.geometry.TriangleMesh(mesh)
    cap_method, rejected_cap_topology = "none_required", None
    if before["boundary_edge_count"]:
        tensor = o3d.t.geometry.TriangleMesh.from_legacy(
            mesh, vertex_dtype=o3d.core.Dtype.Float64)
        mesh = tensor.fill_holes(hole_size=maximum_hole_radius_m).to_legacy()
        cap_method = "cpu_library_boundary_triangulation"
    # VTK's cap filter preserves its input vertices and prepends input faces.
    # Verify that installed behavior before relying on its provenance mapping.
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if (not np.array_equal(vertices, before_vertices)
            or len(faces) < len(before_faces)
            or not np.array_equal(faces[:len(before_faces)], before_faces)):
        raise ValueError("hole filler changed source topology; cap provenance is not exact")
    after = edge_topology(vertices, faces)
    if not after["closed_two_face_edge_incidence"] or after["degenerate_triangle_count"]:
        rejected_cap_topology = after
        mesh, loop_count = _oriented_boundary_caps(source, maximum_hole_radius_m)
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        cap_method = "oriented_incident_face_boundary_loops_with_explicit_inferred_centres"
        after = edge_topology(vertices, faces)
    cap_mask = np.arange(len(faces)) >= len(before_faces)
    if not after["closed_two_face_edge_incidence"] or after["degenerate_triangle_count"]:
        raise ValueError(f"boundary completion did not produce a closed mesh: {after}")
    # Caps can arrive oppositely wound. Orient the existing indexed manifold;
    # no position, connectivity, or supported/inferred classification changes.
    if not mesh.orient_triangles():
        raise ValueError("completed surface is not orientable")
    after = edge_topology(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    if after["inconsistent_winding_edge_count"]:
        raise ValueError("completed surface has inconsistent winding")
    return mesh, cap_mask, {"before": before, "after": after,
                            "boundary_cap_triangle_count": int(cap_mask.sum()),
                            "cap_method": cap_method, "rejected_library_cap_topology": rejected_cap_topology,
                            "maximum_hole_radius_m": maximum_hole_radius_m}


def orientable_envelope(mesh: o3d.geometry.TriangleMesh, *, maximum_removed_fraction: float = .005):
    """Remove only bounded degenerate/nonorientable extraction defects.

    Removed areas may be capped only as inferred geometry. The measured mesh
    is separate and unchanged. This is not a relaxation of the evidence gate.
    """
    initial = len(mesh.triangles)
    removed = 0
    for attempt in range(3):
        faces, vertices = np.asarray(mesh.triangles), np.asarray(mesh.vertices)
        xyz = vertices[faces]
        bad_area = np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1) <= 1e-12
        if bad_area.any():
            removed += int(bad_area.sum())
            if removed > initial * maximum_removed_fraction:
                raise ValueError("degenerate envelope exceeds bounded extraction-defect repair")
            mesh.remove_triangles_by_mask(bad_area)
        if mesh.orient_triangles():
            mesh.remove_unreferenced_vertices()
            return mesh, {"removed_extraction_defect_triangle_count": removed,
                          "source_triangle_count": initial, "passes": attempt + 1,
                          "maximum_removed_fraction": maximum_removed_fraction,
                          "replacement_geometry_is_inferred": True}
        faces = np.asarray(mesh.triangles)
        directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
        _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
        directions = np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
        conflict = (counts == 2) & (directions != 0)
        bad_faces = np.unique(np.flatnonzero(conflict[inverse]) // 3)
        if not len(bad_faces) or removed + len(bad_faces) > initial * maximum_removed_fraction:
            raise ValueError("nonorientable envelope exceeds bounded extraction-defect repair")
        removed += len(bad_faces)
        mesh.remove_triangles_by_index(bad_faces.tolist())
    raise ValueError("envelope remains nonorientable after bounded repair")


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("use a new completed-surface output directory")
    if not 6 <= args.poisson_depth <= 11 or not 1 <= args.poisson_threads <= 8:
        raise ValueError("Poisson work exceeds bounded review limits")
    source_dir = args.measured_surface_dir.resolve()
    source_path = source_dir / "surface_mesh_manifest.json"
    source = _json(source_path)
    if (source.get("status") != "review_only" or source.get("accepted_for_canonical_use") is not False
            or source.get("coordinate_frame") != "backend_world_m_stream_points"):
        raise ValueError("completion requires an explicitly registered review-only mesh")
    if source["output"]["sha256"] != _sha256(source_dir / "surface_mesh.glb"):
        raise ValueError("measured mesh digest mismatch")
    input_path = Path(source["source"]["surfels_manifest"]).resolve()
    surfel_path = Path(source["source"]["surfels_npz"]).resolve()
    if (_sha256(input_path) != source["source"]["surfels_manifest_sha256"]
            or _sha256(surfel_path) != source["source"]["surfels_npz_sha256"]):
        raise ValueError("measured source binding mismatch")
    manifest = _json(input_path)
    if (manifest.get("schema") != "noesis.pcf.single_room_surface_input.v2"
            or manifest.get("phone_walk_only") is not True
            or manifest.get("static_camera_points_included") is not False
            or manifest.get("accepted_for_canonical_use") is not False
            or manifest.get("registration", {}).get("status") != "passed"
            or manifest.get("ceiling", {}).get("mode") != "full_height"):
        raise ValueError("completion requires passed, full-height, phone-only single-room evidence")
    owner = int(manifest["owner_room_id"])
    rows = _owner_visibility_rows(manifest, owner, manifest_dir=input_path.parent)
    if not 2 <= len(rows) <= 256:
        raise ValueError("source view count exceeds completion work bounds")
    for row in rows:
        path = Path(row["path"])
        if row.get("sha256") != _sha256(path):
            raise ValueError("completion raw view binding mismatch")
    evidence = _load_visibility_evidence(rows)
    with np.load(surfel_path, allow_pickle=False) as data:
        points = np.asarray(data["points"], dtype=np.float64).copy()
        colors = np.asarray(data["colors"], dtype=np.uint8).copy()
        cameras = np.asarray(data[OWNER_CAMERAS[owner]], dtype=np.float64).copy()
        if (points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3
                or not 1_000 <= len(points) <= 2_000_000 or not np.isfinite(points).all()
                or np.any(data["view_count"] < 2) or np.any(data["geometry_status"] != 0)
                or np.any(data["owner_room_id"] != owner)):
            raise ValueError("source surfels are not bounded multi-view observed single-room evidence")
    parameters = source["parameters"]
    cloud = o3d.geometry.PointCloud()
    checkpoint = getattr(args, "envelope_checkpoint", None)
    if checkpoint is not None and checkpoint.is_file():
        binding = _json(checkpoint.with_suffix(".json"))
        if (binding.get("checkpoint_sha256") != _sha256(checkpoint)
                or binding.get("source_manifest_sha256") != _sha256(source_path)
                or binding.get("poisson_depth") != args.poisson_depth):
            raise ValueError("envelope checkpoint source/hash/parameter binding mismatch")
        with np.load(checkpoint, allow_pickle=False) as data:
            cloud.points = o3d.utility.Vector3dVector(data["cloud_points"])
            cloud.colors = o3d.utility.Vector3dVector(data["cloud_colors"])
            cloud.normals = o3d.utility.Vector3dVector(data["cloud_normals"])
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(data["vertices"])
            mesh.triangles = o3d.utility.Vector3iVector(data["faces"])
            mesh.vertex_colors = o3d.utility.Vector3dVector(data["colors"])
            density = data["density"].copy()
        orientation = binding["normal_orientation"]
    else:
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255)
        cloud = cloud.voxel_down_sample(float(parameters["voxel_size_m"]))
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(parameters["normal_radius_m"]), max_nn=48))
        orientation = _orient_normals_to_supported_cameras(cloud, cameras, evidence)
        mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=args.poisson_depth, scale=1.05, linear_fit=True, n_threads=args.poisson_threads)
        if checkpoint is not None:
            if checkpoint.exists() or checkpoint.with_suffix(".json").exists():
                raise ValueError("checkpoint artifact already exists")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(checkpoint, vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles),
                                colors=np.asarray(mesh.vertex_colors), density=np.asarray(density),
                                cloud_points=np.asarray(cloud.points), cloud_colors=np.asarray(cloud.colors),
                                cloud_normals=np.asarray(cloud.normals))
            binding = {"checkpoint_sha256": _sha256(checkpoint), "source_manifest_sha256": _sha256(source_path),
                       "poisson_depth": args.poisson_depth, "normal_orientation": orientation}
            checkpoint.with_suffix(".json").write_text(json.dumps(binding, indent=2) + "\n")
    print(json.dumps({"stage": "poisson_envelope_created", "vertices": len(mesh.vertices),
                      "triangles": len(mesh.triangles)}), flush=True)
    if not 1 <= len(mesh.triangles) <= 5_000_000:
        raise ValueError("Poisson envelope exceeds bounded triangle work")
    density_vertices = np.asarray(mesh.vertices).copy()
    density_values = np.asarray(density).copy()
    mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_duplicated_vertices()
    mesh.remove_non_manifold_edges().remove_unreferenced_vertices()
    mesh, extraction_repair = orientable_envelope(mesh)
    print(json.dumps({"stage": "extraction_defects_checked", "repair": extraction_repair}), flush=True)
    # All small closed components remain: removal is not needed for edge
    # closure and can remove genuine furniture detail. The sidecar classifies
    # every output face, including unsupported floating components.
    mesh, caps, topology = close_boundary(mesh, args.maximum_hole_radius_m)
    print(json.dumps({"stage": "boundary_closure_checked", "topology": topology}), flush=True)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    _, density_indices = cKDTree(density_vertices).query(vertices, workers=1)
    distance, _ = cKDTree(np.asarray(cloud.points)).query(vertices, workers=1)
    support_distance = float(parameters["support_distance_m"])
    density_cutoff = float(np.percentile(density_values, float(parameters["density_percentile"])))
    vertex_supported = (density_values[density_indices] >= density_cutoff) & (distance <= support_distance)
    ray_supported, _, _ = _triangle_support_mask(mesh, np.asarray(cloud.points), evidence,
                                                support_distance, visibility_policy="qualified_consensus")
    supported = np.all(vertex_supported[faces], axis=1) & ray_supported & ~caps
    face_class = np.where(caps, 2, np.where(supported, 0, 1)).astype(np.uint8)
    mesh.compute_vertex_normals()
    rgb = np.clip(np.rint(np.asarray(mesh.vertex_colors) * 255), 0, 255).astype(np.uint8)
    completed = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=np.asarray(mesh.vertex_normals),
                                vertex_colors=np.column_stack((rgb, np.full(len(rgb), 255, dtype=np.uint8))),
                                process=False, metadata={"review_only": True, "support_class": "completed_with_face_support_sidecar"})
    name = f"pcf_surface_{OWNER_NAMES[owner]}_completed"
    scene = trimesh.Scene(metadata={"review_only": True, "measured_and_inferred_faces_separately_classified": True})
    scene.add_geometry(completed, geom_name=name, node_name=name)
    output.mkdir(parents=True, exist_ok=False)
    glb_path = output / "surface_mesh.glb"
    glb_path.write_bytes(trimesh.exchange.gltf.export_glb(scene, include_normals=True))
    support_path = output / "face_geometry_support.npz"
    np.savez_compressed(support_path, face_support_class=face_class,
                        face_local_index=np.arange(len(faces), dtype=np.uint32), geometry_name=np.asarray(name),
                        class_names=np.asarray(["multi_view_supported", "poisson_inferred", "boundary_cap_inferred"]))
    # Qualify the actual serialized consumer, not only the pre-export mesh.
    direct = trimesh.load(glb_path, force="mesh", process=False)
    consumer = edge_topology(direct.vertices, direct.faces)
    if not consumer["closed_two_face_edge_incidence"] or consumer["inconsistent_winding_edge_count"]:
        raise ValueError("exported completed mesh is not closed and consistently wound")
    areas = completed.area_faces
    report = {**source, "method": "same_accepted_surfel_screened_poisson_envelope_with_explicit_inferred_closure",
              "source": {**source["source"], "measured_surface_manifest": _record(source_path),
                         "measured_surface_glb": _record(source_dir / "surface_mesh.glb")},
              "parameters": {**parameters, "poisson_depth": args.poisson_depth, "poisson_threads": args.poisson_threads,
                             "completion_mode": "closed_presentation_separate_from_measured", "normal_orientation": orientation},
              "owners": {str(owner): {"room": OWNER_NAMES[owner], "input_point_count": len(points),
                                      "output_vertex_count": len(direct.vertices), "output_triangle_count": len(direct.faces),
                                      "classes": {"completed": {"support_class": "see_face_geometry_support_sidecar",
                                                                "geometry_mode": "completed_screened_poisson_envelope"}}}},
              "completion": {"geometry_support": _record(support_path), "geometry_name": name,
                             "class_counts": {label: int(np.count_nonzero(face_class == index)) for index, label in enumerate(("multi_view_supported", "poisson_inferred", "boundary_cap_inferred"))},
                             "class_surface_area_m2": {label: float(areas[face_class == index].sum()) for index, label in enumerate(("multi_view_supported", "poisson_inferred", "boundary_cap_inferred"))},
                             "topology": topology, "exported_consumer_topology": consumer,
                             "extraction_defect_repair": extraction_repair,
                             "physical_completeness_proven": False, "measured_support_gates_unchanged": True},
              "output": {"glb": str(glb_path), "sha256": _sha256(glb_path), "size_bytes": glb_path.stat().st_size,
                         "vertex_count": len(direct.vertices), "triangle_count": len(direct.faces)},
              "elapsed_s": time.monotonic() - started,
              "limitations": source["limitations"] + ["closure_and_unmeasured_poisson_extension_are_inferred_not_video_measurements", "closed_edge_incidence_does_not_prove_no_self_intersections_or_physical_accuracy"]}
    (output / "surface_mesh_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": report["output"], "completion": report["completion"]}, indent=2), flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured-surface-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poisson-depth", type=int, default=10)
    parser.add_argument("--poisson-threads", type=int, default=1)
    parser.add_argument("--maximum-hole-radius-m", type=float, required=True)
    parser.add_argument("--envelope-checkpoint", type=Path,
                        help="Optional hash/source-bound reusable Poisson envelope, never a new depth source.")
    return parser


if __name__ == "__main__":
    build(_parser().parse_args())
