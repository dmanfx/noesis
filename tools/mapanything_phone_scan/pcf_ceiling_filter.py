"""Conservative owner-local ceiling classification for presentation derivatives.

Height nominates candidates, never removes a point. Removal needs a connected,
overhead planar patch, a compatible local normal and qualified source RGB-D
support. Mixed wall/ceiling junctions abstain. No coordinates are corrected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class CeilingPatch:
    normal: np.ndarray
    offset: float
    seeds: np.ndarray
    metrics: dict


def plane_distance(points: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    return np.abs(np.asarray(points) @ normal + offset)


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = np.asarray(points, dtype=float).mean(0)
    _, _, vectors = np.linalg.svd(np.asarray(points, dtype=float) - center, full_matrices=False)
    normal = vectors[-1]
    if normal[1] < 0:
        normal = -normal
    return normal, -float(center @ normal)


def classify_points(
    points: np.ndarray, normals: np.ndarray, patches: list[CeilingPatch],
    *, normal_cosine: float = .94, plane_tolerance_m: float = .035,
    patch_radius_m: float = .10, junction_guard_m: float = .08,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ceiling and ambiguous masks; vertical/unsupported points survive."""
    points, normals = np.asarray(points), np.asarray(normals)
    if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("point and normal arrays must be matching Nx3")
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError("ceiling classification requires finite geometry")
    removed = np.zeros(len(points), dtype=bool)
    ambiguous = np.zeros(len(points), dtype=bool)
    vertical = np.abs(normals[:, 1]) < .55
    vertical_distance = np.full(len(points), np.inf)
    if np.any(vertical):
        vertical_distance = cKDTree(points[vertical]).query(points, workers=1)[0]
    for patch in patches:
        if not len(patch.seeds):
            continue
        near_plane = plane_distance(points, patch.normal, patch.offset) <= plane_tolerance_m
        inside_patch = cKDTree(patch.seeds).query(points, workers=1)[0] <= patch_radius_m
        candidate = near_plane & inside_patch
        aligned = np.abs(normals @ patch.normal) >= normal_cosine
        junction = vertical_distance <= junction_guard_m
        selected = candidate & aligned & ~junction
        removed |= selected
        ambiguous |= candidate & ~selected
    ambiguous &= ~removed
    return removed, ambiguous


def ceiling_triangle_mask(
    vertices: np.ndarray, faces: np.ndarray, patches: list[CeilingPatch],
    *, plane_tolerance_m: float = .055, patch_radius_m: float = .12,
    normal_cosine: float = .90,
    junction_points: np.ndarray | None = None, junction_guard_m: float = .08,
) -> np.ndarray:
    """Reject only roof-oriented triangles wholly inside qualified patch support.

    Four corner/centroid samples plus three edge midpoints protect junctions and
    prevent a Poisson cap from extending beyond the evidenced ceiling patch.
    """
    corners = np.asarray(vertices)[np.asarray(faces)]
    result = np.zeros(len(corners), dtype=bool)
    if not len(corners):
        return result
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    samples = np.concatenate((corners, corners.mean(1)[:, None],
                              (corners + corners[:, [1, 2, 0]]) / 2), axis=1)
    protected = np.zeros(len(corners),dtype=bool)
    if junction_points is not None and len(junction_points):
        distance = cKDTree(junction_points).query(samples.reshape(-1,3),workers=1)[0].reshape(-1,7)
        protected = (distance<=junction_guard_m).any(1)
    for patch in patches:
        if not len(patch.seeds):
            continue
        aligned = np.abs(normals @ patch.normal) >= normal_cosine
        near = plane_distance(samples.reshape(-1, 3), patch.normal, patch.offset).reshape(-1, 7)
        distance = cKDTree(patch.seeds).query(samples.reshape(-1, 3), workers=1)[0].reshape(-1, 7)
        result |= aligned & ~protected & (near <= plane_tolerance_m).all(1) & (distance <= patch_radius_m).all(1)
    return result


def discover_ceiling_patches(
    points: np.ndarray, cameras: np.ndarray,
    qualify: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, dict]],
    *, voxel_m: float = .04, normal_radius_m: float = .18,
    min_area_m2: float = .45, min_span_m: float = .6,
) -> tuple[np.ndarray, np.ndarray, list[CeilingPatch], dict]:
    """Estimate normals on the FULL owner cloud before nominating upper patches.

    The callback must enforce distinct, confidence-qualified RGB-D support. A
    missing/weak plane produces an empty mask, never a new height-cut fallback.
    """
    import open3d as o3d

    if not len(points) or not len(cameras):
        raise ValueError("ceiling discovery needs owner geometry and source cameras")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=float)))
    cloud = cloud.voxel_down_sample(voxel_m)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_m, max_nn=64))
    samples, normals = np.asarray(cloud.points).copy(), np.asarray(cloud.normals).copy()
    # Search above the typical phone, but this nominates only candidate planes.
    search_y = float(np.median(np.asarray(cameras)[:, 1]) + .10)
    candidate = (samples[:, 1] >= search_y) & (np.abs(normals[:, 1]) >= np.cos(np.deg2rad(25)))
    remaining = np.flatnonzero(candidate)
    patches, rejected = [], []
    o3d.utility.random.seed(19)
    for _ in range(16):
        if len(remaining) < 100:
            break
        upper = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(samples[remaining]))
        plane, indices = upper.segment_plane(distance_threshold=.035, ransac_n=3, num_iterations=600)
        indices = np.asarray(indices, dtype=int)
        if len(indices) < 100:
            break
        selected = remaining[indices]
        remaining = np.delete(remaining, indices)
        normal, offset = fit_plane(samples[selected])
        if normal[1] < np.cos(np.deg2rad(20)):
            continue
        compatible = np.abs(normals[selected] @ normal) >= .94
        selected = selected[compatible]
        if len(selected) < 100:
            continue
        connected = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(samples[selected]))
        labels = np.asarray(connected.cluster_dbscan(eps=.12, min_points=5, print_progress=False))
        for label in sorted(set(labels).difference({-1})):
            rows = selected[labels == label]
            seeds = samples[rows]
            cells = np.unique(np.floor(seeds[:, [0, 2]] / .10).astype(int), axis=0)
            area, span = len(cells) * .01, np.ptp(seeds[:, [0, 2]], axis=0)
            if area < min_area_m2 or np.min(span) < min_span_m:
                continue
            supported, votes = qualify(seeds,np.tile(normal,(len(seeds),1)))
            # Use only supporting source cameras for this hemisphere test.
            # A cabinet TOP seen from cameras above it is not a ceiling, even
            # when most other phone stations happen to be below its plane.
            direction = votes.get("normal_direction_vote")
            if direction is None:
                raise ValueError("ceiling qualification needs recorded camera-normal votes")
            supported &= np.asarray(direction) < 0
            fraction = float(np.mean(supported))
            # Test coverage on connected QUALIFIED seeds, not a permissive
            # extrapolation over the whole original patch. Unsupported seeds
            # abstain even if a fitted plane looks visually plausible.
            if np.count_nonzero(supported) < 50:
                rejected.append({"seed_count":len(seeds), "qualified_fraction":fraction,
                                 "occupied_area_proxy_m2":area, "reason":"weak_rgbd_support"})
                continue
            qualified = seeds[supported]
            qcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(qualified))
            qlabels = np.asarray(qcloud.cluster_dbscan(eps=.12, min_points=5, print_progress=False))
            valid_labels = [value for value in set(qlabels).difference({-1})
                            if np.count_nonzero(qlabels == value) >= 50]
            # Keep individual connected seed components; never span an evidence
            # gap by fitting a ceiling plane across distant clusters.
            if not valid_labels:
                continue
            seeds = qualified[np.isin(qlabels, valid_labels)]
            cells = np.unique(np.floor(seeds[:, [0, 2]] / .10).astype(int), axis=0)
            area, span = len(cells) * .01, np.ptp(seeds[:, [0, 2]], axis=0)
            if area < min_area_m2 or np.min(span) < min_span_m:
                continue
            normal, offset = fit_plane(seeds)
            below = np.asarray(cameras) @ normal + offset < -.10
            if float(np.mean(below)) < .75 or normal[1] < np.cos(np.deg2rad(20)):
                rejected.append({"seed_count":len(seeds), "reason":"not_broadly_overhead"})
                continue
            metrics = {"normal":normal.tolist(), "offset_m":offset, "seed_count":len(seeds),
                       "qualified_fraction":fraction, "occupied_area_proxy_m2":area,
                       "XZ_span_m":span.tolist(), "XYZ_min_m":seeds.min(0).tolist(),
                       "XYZ_max_m":seeds.max(0).tolist(),
                       "residual_p95_m":float(np.percentile(plane_distance(seeds,normal,offset),95)),
                       "qualified_support_count_min":int(np.min(votes["support_count"][supported]))}
            patches.append(CeilingPatch(normal, offset, seeds, metrics))
    return samples, normals, patches, {"sample_count":len(samples), "height_nomination_m":search_y,
        "horizontal_upper_candidate_count":int(np.count_nonzero(candidate)),
        "patches":[patch.metrics for patch in patches], "rejected_patches":rejected}
