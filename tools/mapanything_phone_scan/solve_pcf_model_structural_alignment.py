#!/usr/bin/env python3
"""Solve a review-only PCF-to-Menon alignment from measured room planes.

The static-camera pose is used only to seed a bounded correspondence search.
The final planar similarity is determined by the four dominant Family Room
wall planes and the reviewed Family Room floor polygon in the authored OBJ.
Nothing in this tool publishes calibration or canonical world state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from noesis.validation.authored_scene import AuthoredSceneGeometry


CONTRACT = "noesis.pcf.menon_structural_alignment"
CONTRACT_VERSION = 1


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix(values: Any, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (16,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain 16 finite column-major values")
    matrix = array.reshape((4, 4), order="F")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{label} is not affine")
    return matrix


def _camera_seed_matrix(manifest: dict[str, Any]) -> tuple[np.ndarray, float, np.ndarray]:
    binding = manifest["scene_binding"]
    anchor = manifest["camera_anchor"]
    world_to_scene = _matrix(binding["world_to_scene_col_major"], "world_to_scene")
    solved = _matrix(anchor["camera_to_assembly_col_major"], "camera_to_assembly")
    reference = _matrix(
        anchor["device_reference_camera_to_assembly_col_major"],
        "device_reference_camera_to_assembly",
    )
    solved_forward = solved[:3, 2].copy()
    reference_forward = reference[:3, 2].copy()
    solved_forward[1] = 0.0
    reference_forward[1] = 0.0
    solved_forward /= np.linalg.norm(solved_forward)
    reference_forward /= np.linalg.norm(reference_forward)
    source_yaw = math.atan2(float(solved_forward[0]), float(solved_forward[2]))
    reference_yaw = math.atan2(float(reference_forward[0]), float(reference_forward[2]))
    yaw_delta = math.atan2(
        math.sin(reference_yaw - source_yaw),
        math.cos(reference_yaw - source_yaw),
    )
    rotation = np.array(
        [
            [math.cos(yaw_delta), 0.0, math.sin(yaw_delta)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw_delta), 0.0, math.cos(yaw_delta)],
        ],
        dtype=np.float64,
    )
    correction = np.eye(4, dtype=np.float64)
    correction[:3, :3] = rotation
    correction[:3, 3] = reference[:3, 3] - rotation @ solved[:3, 3]
    singular_values = np.linalg.svd(world_to_scene[:3, :3], compute_uv=False)
    scene_units_per_m = float(np.mean(singular_values))
    if scene_units_per_m <= 0.0:
        raise ValueError("world_to_scene has invalid scale")
    pivot_scene_m = (
        world_to_scene
        @ correction
        @ np.r_[np.asarray(anchor["camera_center_assembly_m"], dtype=np.float64), 1.0]
    )[:3] / scene_units_per_m
    return world_to_scene @ correction, scene_units_per_m, pivot_scene_m


def _review_matrix(
    base: np.ndarray,
    *,
    scene_units_per_m: float,
    pivot_scene_m: np.ndarray,
    yaw_deg: float,
    scale: float = 1.0,
    x_m: float = 0.0,
    z_m: float = 0.0,
) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    planar = np.array(
        [
            [scale * math.cos(yaw), 0.0, scale * math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-scale * math.sin(yaw), 0.0, scale * math.cos(yaw)],
        ],
        dtype=np.float64,
    )
    pivot = pivot_scene_m * scene_units_per_m
    adjustment = np.eye(4, dtype=np.float64)
    adjustment[:3, :3] = planar
    adjustment[:3, 3] = (
        pivot
        + np.array([x_m, 0.0, z_m], dtype=np.float64) * scene_units_per_m
        - planar @ pivot
    )
    return adjustment @ base


def _line_from_points(points_xz: np.ndarray, axis: str) -> tuple[np.ndarray, float]:
    center = np.mean(points_xz, axis=0)
    _, _, basis = np.linalg.svd(points_xz - center, full_matrices=False)
    direction = basis[0]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    if (axis == "x" and normal[0] < 0.0) or (axis == "z" and normal[1] < 0.0):
        normal = -normal
    return normal, float(normal @ center)


def _intersection(
    first: tuple[np.ndarray, float],
    second: tuple[np.ndarray, float],
) -> np.ndarray:
    normals = np.stack([first[0], second[0]])
    if abs(float(np.linalg.det(normals))) < 0.5:
        raise ValueError("selected wall planes do not form stable corners")
    return np.linalg.solve(normals, np.array([first[1], second[1]], dtype=np.float64))


def _extract_family_lines(
    *,
    surfels: Any,
    seed_matrix: np.ndarray,
    scene_units_per_m: float,
    family_owner_id: int,
) -> list[dict[str, Any]]:
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - host dependency gate
        raise RuntimeError("Open3D is required for structural plane extraction") from exc

    points = np.asarray(surfels["points"], dtype=np.float64)
    owner = np.asarray(surfels["owner_room_id"])
    view_count = np.asarray(surfels["view_count"])
    confidence = np.asarray(surfels["mean_confidence"], dtype=np.float64)
    admitted = (
        (owner == int(family_owner_id))
        & (points[:, 1] > 0.12)
        & (points[:, 1] < 1.90)
        & (view_count >= 2)
        & (confidence >= 0.50)
    )
    selected = points[admitted]
    if len(selected) < 10_000:
        raise ValueError("insufficient Family Room structural surfels")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(selected))
    work = cloud.voxel_down_sample(0.05)
    o3d.utility.random.seed(41)
    lines: list[dict[str, Any]] = []
    for _ in range(28):
        if len(work.points) < 300:
            break
        plane, indices = work.segment_plane(
            distance_threshold=0.035,
            ransac_n=3,
            num_iterations=700,
            probability=0.999,
        )
        plane_points = np.asarray(work.points)[indices]
        normal = np.asarray(plane[:3], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        y_span = float(np.ptp(plane_points[:, 1]))
        xz = plane_points[:, [0, 2]]
        center = np.mean(xz, axis=0)
        _, _, basis = np.linalg.svd(xz - center, full_matrices=False)
        projected = (xz - center) @ basis[0]
        xz_span = float(np.percentile(projected, 98) - np.percentile(projected, 2))
        if (
            abs(float(normal[1])) < 0.25
            and y_span >= 0.90
            and xz_span >= 0.80
            and len(indices) >= 120
        ):
            homogeneous = np.c_[plane_points, np.ones(len(plane_points))]
            scene_xz_m = (seed_matrix @ homogeneous.T).T[:, [0, 2]] / scene_units_per_m
            local_center = np.mean(scene_xz_m, axis=0)
            _, _, local_basis = np.linalg.svd(
                scene_xz_m - local_center,
                full_matrices=False,
            )
            local_normal = np.array(
                [-local_basis[0, 1], local_basis[0, 0]],
                dtype=np.float64,
            )
            local_normal /= np.linalg.norm(local_normal)
            axis = "x" if abs(float(local_normal[0])) >= abs(float(local_normal[1])) else "z"
            coordinate = float(local_center[0] if axis == "x" else local_center[1])
            lines.append(
                {
                    "axis": axis,
                    "coordinate_m": coordinate,
                    "points_xz_m": scene_xz_m,
                    "inlier_count": int(len(indices)),
                    "vertical_span_m": y_span,
                    "horizontal_span_m": xz_span,
                }
            )
        work = work.select_by_index(indices, invert=True)
    return lines


def _cluster_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for axis in ("x", "z"):
        rows = sorted(
            (row for row in lines if row["axis"] == axis),
            key=lambda row: row["coordinate_m"],
        )
        for row in rows:
            if (
                clusters
                and clusters[-1]["axis"] == axis
                and abs(row["coordinate_m"] - clusters[-1]["coordinate_m"]) < 0.32
            ):
                clusters[-1]["rows"].append(row)
            else:
                clusters.append({"axis": axis, "rows": [row]})
            cluster = clusters[-1]
            total = sum(item["inlier_count"] for item in cluster["rows"])
            cluster["coordinate_m"] = sum(
                item["coordinate_m"] * item["inlier_count"]
                for item in cluster["rows"]
            ) / total
            cluster["inlier_count"] = int(total)
            cluster["horizontal_span_m"] = max(
                item["horizontal_span_m"] for item in cluster["rows"]
            )
    return clusters


def _family_bounds_m(
    geometry: AuthoredSceneGeometry,
    *,
    family_group: str,
    scene_units_per_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            point
            for edge in geometry.boundary_edges
            if edge.authority_group == family_group
            for point in (edge.start, edge.end)
        ],
        dtype=np.float64,
    )
    if len(points) < 8:
        raise ValueError(f"authored Family Room group {family_group!r} has no stable boundary")
    xz_m = points[:, [0, 2]] / scene_units_per_m
    return np.min(xz_m, axis=0), np.max(xz_m, axis=0)


def _select_boundary_clusters(
    clusters: list[dict[str, Any]],
    *,
    target_min: np.ndarray,
    target_max: np.ndarray,
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    chosen: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for index, axis in enumerate(("x", "z")):
        candidates = sorted(
            (
                cluster
                for cluster in clusters
                if cluster["axis"] == axis
                and target_min[index] - 1.6 <= cluster["coordinate_m"] <= target_max[index] + 1.6
                and cluster["inlier_count"] >= 500
                and cluster["horizontal_span_m"] >= 1.8
            ),
            key=lambda cluster: cluster["coordinate_m"],
        )
        if len(candidates) < 2:
            raise ValueError(f"could not isolate two stable Family Room {axis}-wall planes")
        chosen[axis] = (candidates[0], candidates[-1])
    return chosen


def _fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    left, singular_values, right_t = np.linalg.svd(
        (target_zero.T @ source_zero) / len(source),
    )
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    variance = float(np.mean(np.sum(source_zero * source_zero, axis=1)))
    scale = float(np.sum(singular_values) / variance)
    translation = target_center - scale * rotation @ source_center
    return scale, rotation, translation


def solve(
    *,
    review_manifest: Path,
    surfels_npz: Path,
    surfels_manifest: Path,
    authored_scene: Path,
    room_group_map: Path,
    seed_yaw_deg: float,
    family_owner_id: int,
) -> dict[str, Any]:
    manifest = _json(review_manifest)
    reintegration = _json(surfels_manifest)
    room_map = _json(room_group_map)
    if manifest.get("status") != "review_only" or manifest.get("accepted_for_canonical_use") is not False:
        raise ValueError("review manifest is not fail-closed")
    authored_sha = _sha256(authored_scene)
    if authored_sha != manifest["scene_binding"]["authored_scene_sha256"]:
        raise ValueError("authored OBJ does not match the review assembly scene binding")
    if room_map.get("authored_scene_sha256") != authored_sha:
        raise ValueError("room-group map does not match the authored OBJ")
    declared_surfels = reintegration.get("artifacts", {}).get("surfels_npz", {})
    if declared_surfels.get("sha256") != _sha256(surfels_npz):
        raise ValueError("surfel NPZ does not match its reintegration manifest")
    family_groups = room_map.get("rooms", {}).get("Family Room")
    if not isinstance(family_groups, list) or len(family_groups) != 1:
        raise ValueError("Family Room must map to one reviewed authored floor group")
    family_group = str(family_groups[0])

    base, scene_units_per_m, pivot_scene_m = _camera_seed_matrix(manifest)
    seed_matrix = _review_matrix(
        base,
        scene_units_per_m=scene_units_per_m,
        pivot_scene_m=pivot_scene_m,
        yaw_deg=seed_yaw_deg,
    )
    geometry = AuthoredSceneGeometry.from_obj(authored_scene)
    target_min, target_max = _family_bounds_m(
        geometry,
        family_group=family_group,
        scene_units_per_m=scene_units_per_m,
    )
    with np.load(surfels_npz) as surfels:
        lines = _extract_family_lines(
            surfels=surfels,
            seed_matrix=seed_matrix,
            scene_units_per_m=scene_units_per_m,
            family_owner_id=family_owner_id,
        )
    clusters = _cluster_lines(lines)
    chosen = _select_boundary_clusters(
        clusters,
        target_min=target_min,
        target_max=target_max,
    )
    x_low, x_high = (
        _line_from_points(
            np.concatenate([row["points_xz_m"] for row in cluster["rows"]]),
            "x",
        )
        for cluster in chosen["x"]
    )
    z_low, z_high = (
        _line_from_points(
            np.concatenate([row["points_xz_m"] for row in cluster["rows"]]),
            "z",
        )
        for cluster in chosen["z"]
    )
    source_corners = np.asarray(
        [
            _intersection(x_low, z_low),
            _intersection(x_high, z_low),
            _intersection(x_high, z_high),
            _intersection(x_low, z_high),
        ],
        dtype=np.float64,
    )
    target_corners = np.asarray(
        [
            [target_min[0], target_min[1]],
            [target_max[0], target_min[1]],
            [target_max[0], target_max[1]],
            [target_min[0], target_max[1]],
        ],
        dtype=np.float64,
    )
    scale, delta_rotation, delta_translation = _fit_similarity(
        source_corners,
        target_corners,
    )
    predicted = (scale * (delta_rotation @ source_corners.T)).T + delta_translation
    residuals = np.linalg.norm(predicted - target_corners, axis=1)
    delta_yaw_deg = math.degrees(
        math.atan2(float(delta_rotation[0, 1]), float(delta_rotation[0, 0])),
    )
    pivot_xz = pivot_scene_m[[0, 2]]
    review_translation = (
        scale * delta_rotation @ pivot_xz + delta_translation - pivot_xz
    )
    review_alignment = {
        "yawDeg": float(seed_yaw_deg + delta_yaw_deg),
        "xM": float(review_translation[0]),
        "zM": float(review_translation[1]),
        "scale": float(scale),
    }
    gates = {
        "four_wall_clusters_found": True,
        "corner_residual_max_le_0_20m": bool(float(np.max(residuals)) <= 0.20),
        "corner_residual_median_le_0_15m": bool(float(np.median(residuals)) <= 0.15),
        "uniform_scale_in_0_80_1_30": bool(0.80 <= scale <= 1.30),
        "seed_yaw_delta_abs_le_10deg": bool(abs(delta_yaw_deg) <= 10.0),
        "camera_seed_displacement_le_1_50m": bool(float(np.linalg.norm(review_translation)) <= 1.50),
    }
    solver_status = "passed" if all(gates.values()) else "rejected"
    return {
        "contract": CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "status": "review_only",
        "accepted_for_canonical_use": False,
        "solver_status": solver_status,
        "method": "camera_seed_then_family_floor_and_four_vertical_wall_planes_uniform_sim2",
        "assembly_id": manifest["assembly_id"],
        "coordinate_frame": "menon_scene_review",
        "units": "meters",
        "source": {
            "review_manifest_sha256": _sha256(review_manifest),
            "review_artifact_sha256": manifest["artifact"]["sha256"],
            "surfels_npz_sha256": declared_surfels["sha256"],
            "surfels_manifest_sha256": _sha256(surfels_manifest),
            "authored_scene_sha256": authored_sha,
            "room_group_map_sha256": _sha256(room_group_map),
            "camera_anchor_report_sha256": manifest["camera_anchor"]["source_report_sha256"],
        },
        "seed": {
            "role": "bounded_correspondence_initialization_only",
            "camera_id": manifest["camera_anchor"]["camera_id"],
            "review_yaw_deg": float(seed_yaw_deg),
        },
        "review_alignment": review_alignment,
        "structural_fit": {
            "family_owner_id": int(family_owner_id),
            "family_authored_group": family_group,
            "scene_units_per_meter": scene_units_per_m,
            "candidate_plane_count": len(lines),
            "cluster_count": len(clusters),
            "selected_clusters": {
                axis: [
                    {
                        "coordinate_m": float(cluster["coordinate_m"]),
                        "inlier_count": int(cluster["inlier_count"]),
                        "horizontal_span_m": float(cluster["horizontal_span_m"]),
                    }
                    for cluster in pair
                ]
                for axis, pair in chosen.items()
            },
            "source_corners_xz_m": source_corners.tolist(),
            "target_corners_xz_m": target_corners.tolist(),
            "corner_residual_m": residuals.tolist(),
            "corner_residual_median_m": float(np.median(residuals)),
            "corner_residual_max_m": float(np.max(residuals)),
            "delta_yaw_deg": float(delta_yaw_deg),
            "uniform_scale": float(scale),
            "camera_seed_displacement_m": float(np.linalg.norm(review_translation)),
        },
        "gates": gates,
        "limitations": [
            "family_room_structure_is_the_global_alignment_authority",
            "kitchen_and_living_residuals_include_rejected_cross_session_registration_error",
            "result_is_review_only_and_does_not_mutate_noesis_or_tracking_calibration",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--surfels-npz", type=Path, required=True)
    parser.add_argument("--surfels-manifest", type=Path, required=True)
    parser.add_argument("--authored-scene", type=Path, required=True)
    parser.add_argument("--room-group-map", type=Path, required=True)
    parser.add_argument("--seed-yaw-deg", type=float, required=True)
    parser.add_argument("--family-owner-id", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = solve(
        review_manifest=args.review_manifest.resolve(),
        surfels_npz=args.surfels_npz.resolve(),
        surfels_manifest=args.surfels_manifest.resolve(),
        authored_scene=args.authored_scene.resolve(),
        room_group_map=args.room_group_map.resolve(),
        seed_yaw_deg=args.seed_yaw_deg,
        family_owner_id=args.family_owner_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "solver_status": report["solver_status"],
        "review_alignment": report["review_alignment"],
        "structural_fit": report["structural_fit"],
    }, indent=2, sort_keys=True))
    return 0 if report["solver_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
