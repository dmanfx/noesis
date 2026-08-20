#!/usr/bin/env python3
"""Evidence-gated local geometric refinement for two PCF reconstructions.

This module deliberately is *not* a room-cloud registration initializer.  It
only refines an already verified RGB-D transform inside the region observed by
the accepted cross-session feature matches.  The solved correction is rigid,
metric, gravity preserving (yaw + XYZ), and regularized to the visual estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RefinementSettings:
    voxel_size_m: float = 0.05
    normal_radius_m: float = 0.15
    maximum_correspondence_m: float = 0.25
    normal_cosine_minimum: float = 0.72
    maximum_iterations: int = 12
    yaw_bound_deg: float = 3.0
    horizontal_translation_bound_m: float = 0.50
    vertical_translation_bound_m: float = 0.25
    yaw_prior_sigma_deg: float = 1.5
    horizontal_prior_sigma_m: float = 0.25
    vertical_prior_sigma_m: float = 0.10
    point_plane_sigma_m: float = 0.035
    tangential_weight: float = 0.10
    minimum_correspondences: int = 250
    overlap_horizontal_margin_m: float = 0.75
    overlap_vertical_margin_m: float = 0.35
    lock_vertical_translation_to_floor: bool = True
    floor_histogram_bin_m: float = 0.005
    random_seed: int = 20260815


@dataclass
class SurfaceCloud:
    points: np.ndarray
    normals: np.ndarray


def yaw_transform(parameters: np.ndarray) -> np.ndarray:
    yaw, tx, ty, tz = np.asarray(parameters, dtype=np.float64)
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )
    transform[:3, 3] = [tx, ty, tz]
    return transform


def transform_parameters(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    return np.asarray(
        [
            math.atan2(float(matrix[0, 2]), float(matrix[0, 0])),
            *matrix[:3, 3],
        ],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    return (transform[:3, :3] @ array.T).T + transform[:3, 3]


def _voxel_points(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    finite = np.asarray(points, dtype=np.float64)
    finite = finite[np.isfinite(finite).all(axis=1)]
    keys = np.floor(finite / voxel_size_m).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    centroids = np.column_stack(
        [
            np.bincount(inverse, weights=finite[:, axis]) / counts
            for axis in range(3)
        ]
    )
    return centroids


def prepare_surface(
    points: np.ndarray,
    settings: RefinementSettings,
) -> SurfaceCloud:
    reduced = _voxel_points(points, settings.voxel_size_m)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reduced))
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=settings.normal_radius_m,
            max_nn=40,
        )
    )
    normals = np.asarray(cloud.normals, dtype=np.float64)
    valid = np.isfinite(normals).all(axis=1)
    valid &= np.linalg.norm(normals, axis=1) > 0.9
    return SurfaceCloud(points=reduced[valid], normals=normals[valid])


def visual_overlap_bounds(
    fixed_visual_points: np.ndarray,
    *,
    horizontal_margin_m: float = 0.75,
    vertical_margin_m: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(fixed_visual_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
        raise ValueError("at least eight fixed RGB-D points are required")
    lower = np.percentile(points, 2.0, axis=0)
    upper = np.percentile(points, 98.0, axis=0)
    margin = np.asarray(
        [horizontal_margin_m, vertical_margin_m, horizontal_margin_m],
        dtype=np.float64,
    )
    return lower - margin, upper + margin


def _inside(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.all((points >= lower) & (points <= upper), axis=1)


def _reciprocal_correspondences(
    moving: SurfaceCloud,
    fixed: SurfaceCloud,
    parameters: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    settings: RefinementSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    transform = yaw_transform(parameters)
    moving_world = transform_points(moving.points, transform)
    moving_normals = (transform[:3, :3] @ moving.normals.T).T
    moving_mask = _inside(moving_world, lower, upper)
    fixed_mask = _inside(fixed.points, lower, upper)
    moving_indices = np.flatnonzero(moving_mask)
    fixed_indices = np.flatnonzero(fixed_mask)
    if not len(moving_indices) or not len(fixed_indices):
        raise ValueError("the visually supported overlap region contains no surfels")
    moving_roi = moving_world[moving_indices]
    fixed_roi = fixed.points[fixed_indices]
    fixed_tree = cKDTree(fixed_roi)
    distance, nearest_fixed = fixed_tree.query(moving_roi, k=1)
    _, nearest_moving = cKDTree(moving_roi).query(fixed_roi, k=1)
    reciprocal = np.arange(len(moving_roi)) == nearest_moving[nearest_fixed]
    compatible = distance <= settings.maximum_correspondence_m
    source_normals = moving_normals[moving_indices]
    target_normals = fixed.normals[fixed_indices[nearest_fixed]]
    normal_cosine = np.abs(np.sum(source_normals * target_normals, axis=1))
    compatible &= normal_cosine >= settings.normal_cosine_minimum
    accepted = reciprocal & compatible
    source_index = moving_indices[accepted]
    target_index = fixed_indices[nearest_fixed[accepted]]
    if len(source_index) < settings.minimum_correspondences:
        raise ValueError(
            "too few reciprocal, normal-compatible overlap correspondences: "
            f"{len(source_index)} < {settings.minimum_correspondences}"
        )
    diagnostics = {
        "moving_points_in_roi": int(len(moving_indices)),
        "fixed_points_in_roi": int(len(fixed_indices)),
        "reciprocal_normal_compatible_count": int(len(source_index)),
        "euclidean_median_m": float(np.median(distance[accepted])),
        "euclidean_p80_m": float(np.percentile(distance[accepted], 80.0)),
    }
    return (
        moving.points[source_index],
        fixed.points[target_index],
        fixed.normals[target_index],
        fixed.points[target_index, 1],
        diagnostics,
    )


def _balanced_weights(normals: np.ndarray) -> np.ndarray:
    vertical_component = np.abs(normals[:, 1])
    groups = np.where(vertical_component >= 0.72, 0, np.where(vertical_component <= 0.45, 1, 2))
    counts = np.bincount(groups, minlength=3).astype(np.float64)
    populated = counts > 0
    target = float(np.mean(counts[populated])) if np.any(populated) else 1.0
    group_weight = np.ones(3, dtype=np.float64)
    group_weight[populated] = target / counts[populated]
    return np.sqrt(group_weight[groups])


def estimate_floor_height(
    points: np.ndarray,
    *,
    bin_size_m: float = 0.005,
) -> float:
    """Estimate the dominant lowest horizontal support level in world Y."""

    vertical = np.asarray(points, dtype=np.float64)[:, 1]
    vertical = vertical[np.isfinite(vertical)]
    if len(vertical) < 100:
        raise ValueError("too few finite points to estimate the floor height")
    ceiling = float(np.percentile(vertical, 35.0))
    lower = float(np.percentile(vertical, 0.5))
    candidates = vertical[(vertical >= lower) & (vertical <= ceiling)]
    keys = np.floor(candidates / bin_size_m).astype(np.int64)
    unique, counts = np.unique(keys, return_counts=True)
    mode = float(unique[int(np.argmax(counts))] * bin_size_m + 0.5 * bin_size_m)
    near = vertical[np.abs(vertical - mode) <= 0.04]
    if len(near) < 100:
        raise ValueError("the dominant low support level is not floor-like")
    return float(np.median(near))


def refine_visual_overlap(
    moving_points: np.ndarray,
    fixed_points: np.ndarray,
    fixed_visual_points: np.ndarray,
    initial_transform: np.ndarray,
    *,
    settings: RefinementSettings | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine a verified transform using only the visually observed overlap.

    The returned transform maps the original moving backend-world coordinates
    into the fixed backend-world coordinates.
    """

    active = settings or RefinementSettings()
    moving = prepare_surface(moving_points, active)
    fixed = prepare_surface(fixed_points, active)
    lower, upper = visual_overlap_bounds(
        fixed_visual_points,
        horizontal_margin_m=active.overlap_horizontal_margin_m,
        vertical_margin_m=active.overlap_vertical_margin_m,
    )
    initial = transform_parameters(initial_transform)
    moving_floor = estimate_floor_height(
        moving_points,
        bin_size_m=active.floor_histogram_bin_m,
    )
    fixed_floor = estimate_floor_height(
        fixed_points,
        bin_size_m=active.floor_histogram_bin_m,
    )
    floor_translation = fixed_floor - moving_floor
    if active.lock_vertical_translation_to_floor:
        initial[2] = floor_translation
    parameters = initial.copy()
    lower_parameters = initial - np.asarray(
        [
            math.radians(active.yaw_bound_deg),
            active.horizontal_translation_bound_m,
            active.vertical_translation_bound_m,
            active.horizontal_translation_bound_m,
        ]
    )
    upper_parameters = initial + np.asarray(
        [
            math.radians(active.yaw_bound_deg),
            active.horizontal_translation_bound_m,
            active.vertical_translation_bound_m,
            active.horizontal_translation_bound_m,
        ]
    )
    history: list[dict[str, Any]] = []
    for iteration in range(active.maximum_iterations):
        source, target, normals, _, diagnostics = _reciprocal_correspondences(
            moving,
            fixed,
            parameters,
            lower,
            upper,
            active,
        )
        weights = _balanced_weights(normals)

        def residual(candidate_free: np.ndarray) -> np.ndarray:
            candidate = parameters.copy()
            if active.lock_vertical_translation_to_floor:
                candidate[[0, 1, 3]] = candidate_free
                candidate[2] = floor_translation
            else:
                candidate = candidate_free
            transformed = transform_points(source, yaw_transform(candidate))
            difference = transformed - target
            point_plane = np.sum(difference * normals, axis=1)
            geometry = weights * point_plane / active.point_plane_sigma_m
            tangential = (
                active.tangential_weight
                * weights[:, None]
                * difference
                / active.point_plane_sigma_m
            ).reshape(-1)
            prior = np.asarray(
                [
                    (candidate[0] - initial[0])
                    / math.radians(active.yaw_prior_sigma_deg),
                    (candidate[1] - initial[1])
                    / active.horizontal_prior_sigma_m,
                    (candidate[2] - initial[2])
                    / active.vertical_prior_sigma_m,
                    (candidate[3] - initial[3])
                    / active.horizontal_prior_sigma_m,
                ],
                dtype=np.float64,
            )
            return np.concatenate([geometry, tangential, prior])

        if active.lock_vertical_translation_to_floor:
            free_parameters = parameters[[0, 1, 3]]
            free_lower = lower_parameters[[0, 1, 3]]
            free_upper = upper_parameters[[0, 1, 3]]
        else:
            free_parameters = parameters
            free_lower = lower_parameters
            free_upper = upper_parameters
        optimized = least_squares(
            residual,
            free_parameters,
            bounds=(free_lower, free_upper),
            loss="huber",
            f_scale=1.0,
            max_nfev=150,
        )
        previous = parameters.copy()
        if active.lock_vertical_translation_to_floor:
            parameters[[0, 1, 3]] = optimized.x
            parameters[2] = floor_translation
        else:
            parameters = optimized.x
        step = parameters - previous
        diagnostics.update(
            {
                "iteration": iteration,
                "yaw_deg": math.degrees(float(parameters[0])),
                "translation_m": parameters[1:].tolist(),
                "step_yaw_deg": math.degrees(float(step[0])),
                "step_translation_m": step[1:].tolist(),
            }
        )
        history.append(diagnostics)
        if (
            abs(math.degrees(float(step[0]))) < 0.002
            and float(np.linalg.norm(step[1:])) < 0.0005
        ):
            break
    final_transform = yaw_transform(parameters)
    source, target, normals, _, final_diagnostics = _reciprocal_correspondences(
        moving,
        fixed,
        parameters,
        lower,
        upper,
        active,
    )
    difference = transform_points(source, final_transform) - target
    point_plane = np.abs(np.sum(difference * normals, axis=1))
    euclidean = np.linalg.norm(difference, axis=1)
    report = {
        "method": "visual_overlap_gated_reciprocal_point_to_plane_yaw_xyz",
        "whole_room_icp_used": False,
        "metric_scale_fixed": True,
        "reflection_allowed": False,
        "overlap_bounds_backend_world_m": {
            "minimum": lower.tolist(),
            "maximum": upper.tolist(),
        },
        "initial": {
            "yaw_deg": math.degrees(float(initial[0])),
            "translation_m": initial[1:].tolist(),
        },
        "floor_constraint": {
            "locked": active.lock_vertical_translation_to_floor,
            "moving_floor_height_m": moving_floor,
            "fixed_floor_height_m": fixed_floor,
            "vertical_translation_m": floor_translation,
        },
        "final": {
            "yaw_deg": math.degrees(float(parameters[0])),
            "translation_m": parameters[1:].tolist(),
            "correspondence_count": int(len(source)),
            "point_plane_median_m": float(np.median(point_plane)),
            "point_plane_p80_m": float(np.percentile(point_plane, 80.0)),
            "euclidean_median_m": float(np.median(euclidean)),
            "euclidean_p80_m": float(np.percentile(euclidean, 80.0)),
        },
        "settings": active.__dict__,
        "iterations": history,
        "final_correspondence_diagnostics": final_diagnostics,
    }
    return final_transform, report
