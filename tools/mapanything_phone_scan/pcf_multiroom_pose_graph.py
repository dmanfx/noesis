#!/usr/bin/env python3
"""Gravity-constrained cross-session pose graph for accepted PCF room walks.

This module registers one moving RoomWalk/PCF session into an anchored PCF
session.  It consumes *already geometrically verified* bidirectional PnP
observations; feature extraction and matching deliberately live elsewhere.

Transform convention
--------------------

All transforms use column vectors and map points as ``p_out = T @ p_in``.
``moving_camera_poses`` and ``fixed_camera_poses`` map camera coordinates into
their respective baseline PCF world frames.  Every observation transform maps
the moving baseline world into the fixed baseline world.  The fixed session is
never optimized and therefore remains the world anchor.

The optimized state is a planar session transform ``G`` and one planar
effective transform ``T_i`` per moving view.  Only yaw, X, and Z are optimized;
the vertical translation is a supplied floor-to-floor offset shared by every
state.  A PnP
observation on pair ``(i, j)`` constrains

``inv(F_j) @ T_i @ M_i``

against the same relative camera pose formed with its observed transform.  A
weak global tether plus first- and second-order temporal smoothness regularize
the per-view correction field.  This permits gradual RoomWalk drift without
allowing independent per-frame alignment.  Robust loss handles isolated PnP
noise; support, bidirectional agreement, deformation, and leave-one-whole-
moving-view-out gates fail the result closed.

There is intentionally no point-cloud input and no ICP fallback in this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PnPTransformObservation:
    """One direction of a depth-backed PnP cross-session observation.

    ``direction`` is an opaque but stable label.  A view pair is considered
    bidirectional only when it has at least two distinct direction labels.
    The transform must be a rigid, gravity-preserving moving-world to
    fixed-world transform.
    """

    moving_view: int
    fixed_view: int
    direction: str
    transform_moving_to_fixed: np.ndarray
    inlier_count: int
    reprojection_median_px: float
    reprojection_p80_px: float


@dataclass(frozen=True)
class IntraSessionPoseObservation:
    """Optional RGB-D/VIO edge between two moving-session camera views.

    ``target_camera_from_source_camera`` maps source-camera coordinates into
    target-camera coordinates.  These edges constrain *trajectory shape* but
    do not set the fixed-world gauge; only cross-session PnP observations can
    do that.  The public type is present now so verified intra-session RGB-D
    edges can be added without changing the solver contract later.
    """

    source_view: int
    target_view: int
    target_camera_from_source_camera: np.ndarray
    translation_sigma_m: float = 0.10
    rotation_sigma_deg: float = 1.5
    weight: float = 1.0


@dataclass(frozen=True)
class PoseGraphSettings:
    """Numerical scales and fail-closed admission thresholds.

    Observation sigmas affect the robust least-squares fit.  Gate values are
    evaluated in physical units after optimization and are deliberately
    separate from those sigmas.
    """

    minimum_inliers_per_observation: int = 20
    maximum_observation_reprojection_median_px: float = 3.0
    maximum_observation_reprojection_p80_px: float = 5.0
    maximum_observation_non_yaw_deg: float = 2.0
    consensus_yaw_deg: float = 5.0
    consensus_translation_m: float = 1.75
    maximum_bidirectional_yaw_disagreement_deg: float = 2.5
    maximum_bidirectional_translation_disagreement_m: float = 0.65
    minimum_distinct_moving_views: int = 3
    minimum_distinct_fixed_views: int = 3
    minimum_distinct_pairs: int = 5
    minimum_bidirectional_pairs: int = 3
    minimum_moving_camera_span_m: float = 0.40
    minimum_fixed_camera_span_m: float = 0.40

    observation_translation_sigma_m: float = 0.14
    observation_yaw_sigma_deg: float = 1.25
    global_position_tether_sigma_m: float = 0.80
    global_yaw_tether_sigma_deg: float = 7.0
    smooth_position_sigma_m: float = 0.12
    smooth_yaw_sigma_deg: float = 1.8
    curvature_position_sigma_m: float = 0.22
    curvature_yaw_sigma_deg: float = 2.8
    robust_loss_scale: float = 1.5
    maximum_optimizer_evaluations: int = 650

    maximum_train_translation_median_m: float = 0.15
    maximum_train_translation_p80_m: float = 0.25
    maximum_train_yaw_median_deg: float = 0.75
    maximum_train_yaw_p80_deg: float = 1.25
    maximum_heldout_translation_median_m: float = 0.20
    maximum_heldout_translation_p80_m: float = 0.25
    maximum_heldout_yaw_median_deg: float = 0.40
    maximum_heldout_yaw_p80_deg: float = 0.50
    maximum_leaveout_session_translation_p80_m: float = 0.25
    maximum_leaveout_session_yaw_p80_deg: float = 0.5
    temporal_segment_gap_views: int = 6
    maximum_segment_heldout_translation_median_m: float = 0.25
    maximum_segment_heldout_translation_p80_m: float = 0.35
    maximum_segment_heldout_yaw_median_deg: float = 0.75
    maximum_segment_heldout_yaw_p80_deg: float = 1.25
    maximum_segment_leaveout_session_translation_m: float = 0.20
    maximum_segment_leaveout_session_yaw_deg: float = 0.50
    maximum_view_deformation_p80_m: float = 0.40
    maximum_view_deformation_max_m: float = 0.75
    maximum_view_yaw_deformation_p80_deg: float = 2.0
    maximum_view_yaw_deformation_max_deg: float = 3.0


@dataclass
class PoseGraphResult:
    """Fail-closed solver result.

    Authoritative transforms are populated only when ``accepted`` is true.
    Rejected candidates remain available solely as JSON-friendly diagnostics
    under ``report["candidate"]``.
    """

    accepted: bool
    reason_codes: tuple[str, ...]
    global_transform_moving_to_fixed: np.ndarray | None
    per_view_transforms_moving_to_fixed: dict[int, np.ndarray] = field(
        default_factory=dict
    )
    corrected_moving_camera_poses: dict[int, np.ndarray] = field(
        default_factory=dict
    )
    report: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _PreparedObservation:
    source: PnPTransformObservation
    parameters: np.ndarray
    weight: float


@dataclass
class _NumericSolution:
    success: bool
    message: str
    global_parameters: np.ndarray
    per_view_parameters: dict[int, np.ndarray]
    cost: float
    evaluations: int


def _yaw_transform(parameters: Sequence[float]) -> np.ndarray:
    yaw, tx, ty, tz = [float(value) for value in parameters]
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )
    transform[:3, 3] = [tx, ty, tz]
    return transform


def _planar_transform(
    parameters: Sequence[float], vertical_translation_m: float
) -> np.ndarray:
    yaw, tx, tz = [float(value) for value in parameters]
    return _yaw_transform([yaw, tx, vertical_translation_m, tz])


def _wrapped_radians(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _parameters_from_transform(transform: np.ndarray) -> tuple[np.ndarray, float]:
    rotation = transform[:3, :3]
    yaw = math.atan2(float(rotation[0, 2]), float(rotation[0, 0]))
    yaw_rotation = _yaw_transform([yaw, 0.0, 0.0, 0.0])[:3, :3]
    relative = yaw_rotation.T @ rotation
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    non_yaw_deg = math.degrees(math.acos(cosine))
    parameters = np.asarray([yaw, *transform[:3, 3]], dtype=np.float64)
    return parameters, non_yaw_deg


def _validate_transform(transform: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"{label} contains scale or shear")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, abs_tol=2e-3):
        raise ValueError(f"{label} contains a reflection or invalid rotation")
    return value


def _normalize_poses(
    poses: Mapping[int, np.ndarray] | Sequence[np.ndarray],
    label: str,
) -> dict[int, np.ndarray]:
    items = poses.items() if isinstance(poses, Mapping) else enumerate(poses)
    normalized: dict[int, np.ndarray] = {}
    for key, transform in items:
        view = int(key)
        if view in normalized:
            raise ValueError(f"duplicate {label} view {view}")
        normalized[view] = _validate_transform(transform, f"{label}[{view}]")
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    return dict(sorted(normalized.items()))


def _normalize_intra_session_edges(
    edges: Sequence[IntraSessionPoseObservation],
    moving_poses: Mapping[int, np.ndarray],
) -> list[IntraSessionPoseObservation]:
    normalized: list[IntraSessionPoseObservation] = []
    for index, edge in enumerate(edges):
        if edge.source_view not in moving_poses or edge.target_view not in moving_poses:
            raise ValueError(
                f"intra-session edge {index} references a missing moving view"
            )
        if edge.source_view == edge.target_view:
            raise ValueError(f"intra-session edge {index} is a self edge")
        if (
            edge.translation_sigma_m <= 0.0
            or edge.rotation_sigma_deg <= 0.0
            or edge.weight <= 0.0
            or not all(
                math.isfinite(value)
                for value in (
                    edge.translation_sigma_m,
                    edge.rotation_sigma_deg,
                    edge.weight,
                )
            )
        ):
            raise ValueError(f"intra-session edge {index} has invalid uncertainty")
        transform = _validate_transform(
            edge.target_camera_from_source_camera,
            f"intra_session_edges[{index}].target_camera_from_source_camera",
        )
        normalized.append(
            IntraSessionPoseObservation(
                source_view=edge.source_view,
                target_view=edge.target_view,
                target_camera_from_source_camera=transform,
                translation_sigma_m=edge.translation_sigma_m,
                rotation_sigma_deg=edge.rotation_sigma_deg,
                weight=edge.weight,
            )
        )
    return normalized


def _camera_center(pose: np.ndarray) -> np.ndarray:
    return pose[:3, 3]


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return transform[:3, :3] @ point + transform[:3, 3]


def _span(points: Sequence[np.ndarray]) -> float:
    if len(points) < 2:
        return 0.0
    values = np.stack(points)
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    return float(np.max(distances))


def _weighted_parameters(rows: Sequence[_PreparedObservation]) -> np.ndarray:
    weights = np.asarray([row.weight for row in rows], dtype=np.float64)
    values = np.stack([row.parameters for row in rows])
    yaw = math.atan2(
        float(np.sum(weights * np.sin(values[:, 0]))),
        float(np.sum(weights * np.cos(values[:, 0]))),
    )
    translation = np.average(values[:, 1:], axis=0, weights=weights)
    return np.asarray([yaw, *translation], dtype=np.float64)


def _weighted_planar_parameters(
    rows: Sequence[_PreparedObservation],
) -> np.ndarray:
    full = _weighted_parameters(rows)
    return np.asarray([full[0], full[1], full[3]], dtype=np.float64)


def _parameter_distance(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    yaw = math.degrees(abs(float(_wrapped_radians(first[0] - second[0]))))
    translation = float(np.linalg.norm(first[1:] - second[1:]))
    return yaw, translation


def _quality_weight(observation: PnPTransformObservation) -> float:
    inlier_factor = float(np.clip(observation.inlier_count / 24.0, 0.45, 2.25))
    reprojection_factor = float(
        np.clip(1.5 / max(0.35, observation.reprojection_median_px), 0.35, 2.0)
    )
    return math.sqrt(inlier_factor * reprojection_factor)


def _prepare_observations(
    observations: Sequence[PnPTransformObservation],
    moving_poses: Mapping[int, np.ndarray],
    fixed_poses: Mapping[int, np.ndarray],
    settings: PoseGraphSettings,
) -> tuple[list[_PreparedObservation], dict[str, Any]]:
    prepared: list[_PreparedObservation] = []
    quality_rejected: list[dict[str, Any]] = []
    for index, observation in enumerate(observations):
        if observation.moving_view not in moving_poses:
            raise ValueError(
                f"observation {index} references missing moving view "
                f"{observation.moving_view}"
            )
        if observation.fixed_view not in fixed_poses:
            raise ValueError(
                f"observation {index} references missing fixed view "
                f"{observation.fixed_view}"
            )
        transform = _validate_transform(
            observation.transform_moving_to_fixed,
            f"observations[{index}].transform_moving_to_fixed",
        )
        parameters, non_yaw_deg = _parameters_from_transform(transform)
        reasons: list[str] = []
        if observation.inlier_count < settings.minimum_inliers_per_observation:
            reasons.append("low_inlier_count")
        if (
            not math.isfinite(observation.reprojection_median_px)
            or observation.reprojection_median_px
            > settings.maximum_observation_reprojection_median_px
        ):
            reasons.append("high_reprojection_median_error")
        if (
            not math.isfinite(observation.reprojection_p80_px)
            or observation.reprojection_p80_px
            > settings.maximum_observation_reprojection_p80_px
        ):
            reasons.append("high_reprojection_p80_error")
        if non_yaw_deg > settings.maximum_observation_non_yaw_deg:
            reasons.append("non_gravity_rotation")
        if not observation.direction:
            reasons.append("missing_direction")
        if reasons:
            quality_rejected.append(
                {
                    "index": index,
                    "moving_view": observation.moving_view,
                    "fixed_view": observation.fixed_view,
                    "reason_codes": reasons,
                }
            )
            continue
        prepared.append(
            _PreparedObservation(
                source=observation,
                parameters=parameters,
                weight=_quality_weight(observation),
            )
        )

    pair_groups: dict[tuple[int, int], list[_PreparedObservation]] = {}
    for row in prepared:
        key = (row.source.moving_view, row.source.fixed_view)
        pair_groups.setdefault(key, []).append(row)

    pair_rejected: list[dict[str, Any]] = []
    direction_metrics: dict[str, dict[str, float | int | bool]] = {}
    accepted_pairs: set[tuple[int, int]] = set()
    for key, rows in sorted(pair_groups.items()):
        direction_groups: dict[str, list[_PreparedObservation]] = {}
        for row in rows:
            direction_groups.setdefault(row.source.direction, []).append(row)
        direction_centers = [
            _weighted_parameters(group) for group in direction_groups.values()
        ]
        yaw_disagreement = 0.0
        translation_disagreement = 0.0
        for first_index, first in enumerate(direction_centers):
            for second in direction_centers[first_index + 1 :]:
                yaw, translation = _parameter_distance(first, second)
                yaw_disagreement = max(yaw_disagreement, yaw)
                translation_disagreement = max(translation_disagreement, translation)
        bidirectional = len(direction_groups) >= 2
        pair_label = f"moving_{key[0]:02d}_fixed_{key[1]:02d}"
        direction_metrics[pair_label] = {
            "direction_count": len(direction_groups),
            "bidirectional": bidirectional,
            "yaw_disagreement_deg": yaw_disagreement,
            "translation_disagreement_m": translation_disagreement,
        }
        if bidirectional and (
            yaw_disagreement
            > settings.maximum_bidirectional_yaw_disagreement_deg
            or translation_disagreement
            > settings.maximum_bidirectional_translation_disagreement_m
        ):
            pair_rejected.append(
                {
                    "moving_view": key[0],
                    "fixed_view": key[1],
                    "reason": "bidirectional_disagreement",
                    "yaw_disagreement_deg": yaw_disagreement,
                    "translation_disagreement_m": translation_disagreement,
                }
            )
        else:
            accepted_pairs.add(key)

    screened = [
        row
        for row in prepared
        if (row.source.moving_view, row.source.fixed_view) in accepted_pairs
    ]
    return screened, {
        "input_observation_count": len(observations),
        "quality_rejected": quality_rejected,
        "bidirectional_pair_metrics": direction_metrics,
        "pair_rejected": pair_rejected,
    }


def _select_consensus(
    rows: Sequence[_PreparedObservation],
    settings: PoseGraphSettings,
) -> list[_PreparedObservation]:
    """Select the strongest broad transform mode before nonlinear fitting."""

    best: list[_PreparedObservation] = []
    best_key = (0, 0, 0, 0.0)
    for seed in rows:
        members = []
        for row in rows:
            yaw, translation = _parameter_distance(seed.parameters, row.parameters)
            if (
                yaw <= settings.consensus_yaw_deg
                and translation <= settings.consensus_translation_m
            ):
                members.append(row)
        moving_count = len({row.source.moving_view for row in members})
        fixed_count = len({row.source.fixed_view for row in members})
        pair_count = len(
            {(row.source.moving_view, row.source.fixed_view) for row in members}
        )
        support = float(sum(row.weight for row in members))
        key = (min(moving_count, fixed_count), pair_count, len(members), support)
        if key > best_key:
            best_key = key
            best = members
    return sorted(
        best,
        key=lambda row: (
            row.source.moving_view,
            row.source.fixed_view,
            row.source.direction,
        ),
    )


def _support_metrics(
    rows: Sequence[_PreparedObservation],
    moving_poses: Mapping[int, np.ndarray],
    fixed_poses: Mapping[int, np.ndarray],
) -> dict[str, Any]:
    moving_views = sorted({row.source.moving_view for row in rows})
    fixed_views = sorted({row.source.fixed_view for row in rows})
    pair_groups: dict[tuple[int, int], set[str]] = {}
    for row in rows:
        pair_groups.setdefault(
            (row.source.moving_view, row.source.fixed_view), set()
        ).add(row.source.direction)
    bidirectional = sum(len(directions) >= 2 for directions in pair_groups.values())
    return {
        "observation_count": len(rows),
        "moving_views": moving_views,
        "fixed_views": fixed_views,
        "distinct_pair_count": len(pair_groups),
        "bidirectional_pair_count": bidirectional,
        "moving_camera_span_m": _span(
            [_camera_center(moving_poses[index]) for index in moving_views]
        ),
        "fixed_camera_span_m": _span(
            [_camera_center(fixed_poses[index]) for index in fixed_views]
        ),
    }


def _support_failures(
    metrics: Mapping[str, Any], settings: PoseGraphSettings
) -> list[str]:
    failures: list[str] = []
    if len(metrics["moving_views"]) < settings.minimum_distinct_moving_views:
        failures.append("insufficient_moving_view_support")
    if len(metrics["fixed_views"]) < settings.minimum_distinct_fixed_views:
        failures.append("insufficient_fixed_view_support")
    if metrics["distinct_pair_count"] < settings.minimum_distinct_pairs:
        failures.append("insufficient_cross_view_pairs")
    if metrics["bidirectional_pair_count"] < settings.minimum_bidirectional_pairs:
        failures.append("insufficient_bidirectional_pairs")
    if metrics["moving_camera_span_m"] < settings.minimum_moving_camera_span_m:
        failures.append("insufficient_moving_camera_baseline")
    if metrics["fixed_camera_span_m"] < settings.minimum_fixed_camera_span_m:
        failures.append("insufficient_fixed_camera_baseline")
    return failures


def _pack_parameters(
    global_parameters: np.ndarray,
    per_view_parameters: Mapping[int, np.ndarray],
    moving_view_ids: Sequence[int],
) -> np.ndarray:
    return np.concatenate(
        [global_parameters]
        + [np.asarray(per_view_parameters[index]) for index in moving_view_ids]
    )


def _unpack_parameters(
    values: np.ndarray, moving_view_ids: Sequence[int]
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    global_parameters = values[:3]
    per_view = {
        index: values[3 + offset * 3 : 6 + offset * 3]
        for offset, index in enumerate(moving_view_ids)
    }
    return global_parameters, per_view


def _observation_residual_components(
    row: _PreparedObservation,
    effective_parameters: np.ndarray,
    moving_poses: Mapping[int, np.ndarray],
    fixed_poses: Mapping[int, np.ndarray],
    vertical_translation_m: float,
) -> tuple[np.ndarray, float, float]:
    """Return fixed-world XZ, yaw, and diagnostic vertical disagreement."""

    moving_pose = moving_poses[row.source.moving_view]
    fixed_from_world = np.linalg.inv(fixed_poses[row.source.fixed_view])
    predicted_relative = (
        fixed_from_world
        @ _planar_transform(effective_parameters, vertical_translation_m)
        @ moving_pose
    )
    observed_relative = (
        fixed_from_world
        @ row.source.transform_moving_to_fixed
        @ moving_pose
    )
    translation_camera = predicted_relative[:3, 3] - observed_relative[:3, 3]
    translation_world = (
        fixed_poses[row.source.fixed_view][:3, :3] @ translation_camera
    )
    horizontal_translation = translation_world[[0, 2]]
    yaw = float(_wrapped_radians(effective_parameters[0] - row.parameters[0]))
    return horizontal_translation, yaw, float(translation_world[1])


def _numeric_solve(
    rows: Sequence[_PreparedObservation],
    moving_poses: Mapping[int, np.ndarray],
    fixed_poses: Mapping[int, np.ndarray],
    intra_session_edges: Sequence[IntraSessionPoseObservation],
    vertical_translation_m: float,
    settings: PoseGraphSettings,
) -> _NumericSolution:
    moving_view_ids = sorted(moving_poses)
    global_initial = _weighted_planar_parameters(rows)
    per_view_initial = {index: global_initial.copy() for index in moving_view_ids}
    initial = _pack_parameters(global_initial, per_view_initial, moving_view_ids)
    row_view_weight_energy: dict[int, float] = {}
    for row in rows:
        view = row.source.moving_view
        row_view_weight_energy[view] = (
            row_view_weight_energy.get(view, 0.0) + row.weight**2
        )

    yaw_observation_sigma = math.radians(settings.observation_yaw_sigma_deg)
    yaw_tether_sigma = math.radians(settings.global_yaw_tether_sigma_deg)
    yaw_smooth_sigma = math.radians(settings.smooth_yaw_sigma_deg)
    yaw_curvature_sigma = math.radians(settings.curvature_yaw_sigma_deg)

    def residual(values: np.ndarray) -> np.ndarray:
        global_parameters, per_view = _unpack_parameters(values, moving_view_ids)
        global_transform = _planar_transform(
            global_parameters, vertical_translation_m
        )
        result: list[np.ndarray] = []
        for row in rows:
            translation, yaw, _ = _observation_residual_components(
                row,
                per_view[row.source.moving_view],
                moving_poses,
                fixed_poses,
                vertical_translation_m,
            )
            # Balance complete moving views.  A high-texture Kitchen view that
            # matches many Family frames must not outweigh an independently
            # observed Kitchen view solely by contributing more PnP records.
            view_energy = math.sqrt(
                row_view_weight_energy[row.source.moving_view]
            )
            weight = row.weight / view_energy
            result.append(
                weight
                * np.asarray(
                    [
                        *(translation / settings.observation_translation_sigma_m),
                        yaw / yaw_observation_sigma,
                    ],
                    dtype=np.float64,
                )
            )

        for edge in intra_session_edges:
            source_pose = (
                _planar_transform(
                    per_view[edge.source_view], vertical_translation_m
                )
                @ moving_poses[edge.source_view]
            )
            target_pose = (
                _planar_transform(
                    per_view[edge.target_view], vertical_translation_m
                )
                @ moving_poses[edge.target_view]
            )
            predicted = np.linalg.inv(target_pose) @ source_pose
            observed = edge.target_camera_from_source_camera
            rotation_error = Rotation.from_matrix(
                observed[:3, :3].T @ predicted[:3, :3]
            ).as_rotvec()
            result.append(
                edge.weight
                * np.asarray(
                    [
                        *((predicted[:3, 3] - observed[:3, 3])
                          / edge.translation_sigma_m),
                        *(rotation_error / math.radians(edge.rotation_sigma_deg)),
                    ],
                    dtype=np.float64,
                )
            )

        deviations: dict[int, np.ndarray] = {}
        yaw_deviations: dict[int, float] = {}
        for index in moving_view_ids:
            center = _camera_center(moving_poses[index])
            effective_center = _transform_point(
                _planar_transform(per_view[index], vertical_translation_m),
                center,
            )
            global_center = _transform_point(global_transform, center)
            deviations[index] = effective_center - global_center
            yaw_deviations[index] = float(
                _wrapped_radians(per_view[index][0] - global_parameters[0])
            )
            result.append(
                np.asarray(
                    [
                        *(deviations[index] / settings.global_position_tether_sigma_m),
                        yaw_deviations[index] / yaw_tether_sigma,
                    ]
                )
            )

        for first, second in zip(moving_view_ids, moving_view_ids[1:]):
            gap_scale = math.sqrt(max(1, second - first))
            result.append(
                np.asarray(
                    [
                        *((deviations[second] - deviations[first])
                          / (settings.smooth_position_sigma_m * gap_scale)),
                        float(
                            _wrapped_radians(
                                yaw_deviations[second] - yaw_deviations[first]
                            )
                        )
                        / (yaw_smooth_sigma * gap_scale),
                    ]
                )
            )

        for first, middle, last in zip(
            moving_view_ids,
            moving_view_ids[1:],
            moving_view_ids[2:],
        ):
            left_gap = max(1, middle - first)
            right_gap = max(1, last - middle)
            fraction = left_gap / (left_gap + right_gap)
            linear_position = (
                deviations[first]
                + fraction * (deviations[last] - deviations[first])
            )
            left_yaw = yaw_deviations[first]
            yaw_delta = float(
                _wrapped_radians(yaw_deviations[last] - yaw_deviations[first])
            )
            linear_yaw = left_yaw + fraction * yaw_delta
            result.append(
                np.asarray(
                    [
                        *((deviations[middle] - linear_position)
                          / settings.curvature_position_sigma_m),
                        float(
                            _wrapped_radians(yaw_deviations[middle] - linear_yaw)
                        )
                        / yaw_curvature_sigma,
                    ]
                )
            )
        return np.concatenate(result)

    yaw_bound = math.radians(12.0)
    lower = initial.copy()
    upper = initial.copy()
    for offset in range(0, len(initial), 3):
        lower[offset] -= yaw_bound
        upper[offset] += yaw_bound
        lower[offset + 1 : offset + 3] -= 3.0
        upper[offset + 1 : offset + 3] += 3.0
    optimized = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=settings.robust_loss_scale,
        max_nfev=settings.maximum_optimizer_evaluations,
    )
    global_parameters, per_view_parameters = _unpack_parameters(
        optimized.x, moving_view_ids
    )
    return _NumericSolution(
        success=bool(optimized.success and np.all(np.isfinite(optimized.x))),
        message=str(optimized.message),
        global_parameters=global_parameters,
        per_view_parameters=per_view_parameters,
        cost=float(optimized.cost),
        evaluations=int(optimized.nfev),
    )


def _observation_metrics(
    rows: Sequence[_PreparedObservation],
    solution: _NumericSolution,
    moving_poses: Mapping[int, np.ndarray],
    fixed_poses: Mapping[int, np.ndarray],
    vertical_translation_m: float,
) -> dict[str, float]:
    translations: list[float] = []
    yaws: list[float] = []
    vertical: list[float] = []
    for row in rows:
        translation, yaw, vertical_error = _observation_residual_components(
            row,
            solution.per_view_parameters[row.source.moving_view],
            moving_poses,
            fixed_poses,
            vertical_translation_m,
        )
        translations.append(float(np.linalg.norm(translation)))
        yaws.append(math.degrees(abs(yaw)))
        vertical.append(abs(vertical_error))
    if not translations:
        translations = [float("inf")]
        yaws = [float("inf")]
        vertical = [float("inf")]
    return {
        "translation_domain": "fixed_world_horizontal_xz",
        "translation_median_m": float(np.median(translations)),
        "translation_p80_m": float(np.percentile(translations, 80.0)),
        "translation_max_m": float(np.max(translations)),
        "yaw_median_deg": float(np.median(yaws)),
        "yaw_p80_deg": float(np.percentile(yaws, 80.0)),
        "yaw_max_deg": float(np.max(yaws)),
        "vertical_disagreement_median_m": float(np.median(vertical)),
        "vertical_disagreement_p80_m": float(np.percentile(vertical, 80.0)),
    }


def _deformation_metrics(
    solution: _NumericSolution,
    moving_poses: Mapping[int, np.ndarray],
    vertical_translation_m: float,
) -> dict[str, float]:
    global_transform = _planar_transform(
        solution.global_parameters, vertical_translation_m
    )
    positions: list[float] = []
    yaws: list[float] = []
    for index, pose in moving_poses.items():
        center = _camera_center(pose)
        global_center = _transform_point(global_transform, center)
        effective_center = _transform_point(
            _planar_transform(
                solution.per_view_parameters[index], vertical_translation_m
            ),
            center,
        )
        positions.append(float(np.linalg.norm(effective_center - global_center)))
        yaws.append(
            math.degrees(
                abs(
                    float(
                        _wrapped_radians(
                            solution.per_view_parameters[index][0]
                            - solution.global_parameters[0]
                        )
                    )
                )
            )
        )
    return {
        "position_median_m": float(np.median(positions)),
        "position_p80_m": float(np.percentile(positions, 80.0)),
        "position_max_m": float(np.max(positions)),
        "yaw_median_deg": float(np.median(yaws)),
        "yaw_p80_deg": float(np.percentile(yaws, 80.0)),
        "yaw_max_deg": float(np.max(yaws)),
    }


def _leave_one_moving_view_out(
    rows: Sequence[_PreparedObservation],
    moving_poses: Mapping[int, np.ndarray],
    fixed_poses: Mapping[int, np.ndarray],
    intra_session_edges: Sequence[IntraSessionPoseObservation],
    vertical_translation_m: float,
    full_solution: _NumericSolution,
    settings: PoseGraphSettings,
) -> tuple[dict[str, Any], list[str]]:
    observed_views = sorted({row.source.moving_view for row in rows})
    folds: list[dict[str, Any]] = []
    all_translation: list[float] = []
    all_yaw: list[float] = []
    all_vertical: list[float] = []
    session_translation: list[float] = []
    session_yaw: list[float] = []
    failures: list[str] = []
    for heldout_view in observed_views:
        train = [row for row in rows if row.source.moving_view != heldout_view]
        heldout = [row for row in rows if row.source.moving_view == heldout_view]
        train_support = _support_metrics(train, moving_poses, fixed_poses)
        # A fold may have one fewer fixed view than the full graph, but it must
        # still be constrained by at least two moving views and three pairs.
        if (
            len(train_support["moving_views"]) < 2
            or len(train_support["fixed_views"]) < 2
            or train_support["distinct_pair_count"] < 3
        ):
            failures.append("leaveout_training_support_collapsed")
            folds.append(
                {
                    "heldout_moving_view": heldout_view,
                    "passed": False,
                    "reason": "training_support_collapsed",
                }
            )
            continue
        candidate = _numeric_solve(
            train,
            moving_poses,
            fixed_poses,
            intra_session_edges,
            vertical_translation_m,
            settings,
        )
        metrics = _observation_metrics(
            heldout,
            candidate,
            moving_poses,
            fixed_poses,
            vertical_translation_m,
        )
        global_yaw, global_translation = _parameter_distance(
            candidate.global_parameters, full_solution.global_parameters
        )
        session_translation.append(global_translation)
        session_yaw.append(global_yaw)
        for row in heldout:
            translation, yaw, vertical_error = _observation_residual_components(
                row,
                candidate.per_view_parameters[heldout_view],
                moving_poses,
                fixed_poses,
                vertical_translation_m,
            )
            all_translation.append(float(np.linalg.norm(translation)))
            all_yaw.append(math.degrees(abs(yaw)))
            all_vertical.append(abs(vertical_error))
        fold_passed = bool(candidate.success)
        folds.append(
            {
                "heldout_moving_view": heldout_view,
                "passed": fold_passed,
                "optimizer_success": candidate.success,
                "translation_median_m": metrics["translation_median_m"],
                "translation_p80_m": metrics["translation_p80_m"],
                "yaw_median_deg": metrics["yaw_median_deg"],
                "yaw_p80_deg": metrics["yaw_p80_deg"],
                "session_translation_delta_m": global_translation,
                "session_yaw_delta_deg": global_yaw,
            }
        )
        if not candidate.success:
            failures.append("leaveout_optimizer_failed")

    if not all_translation:
        failures.append("leaveout_validation_unavailable")
        all_translation = [float("inf")]
        all_yaw = [float("inf")]
        all_vertical = [float("inf")]
        session_translation = [float("inf")]
        session_yaw = [float("inf")]
    aggregate = {
        "translation_domain": "fixed_world_horizontal_xz",
        "translation_median_m": float(np.median(all_translation)),
        "translation_p80_m": float(np.percentile(all_translation, 80.0)),
        "yaw_median_deg": float(np.median(all_yaw)),
        "yaw_p80_deg": float(np.percentile(all_yaw, 80.0)),
        "vertical_disagreement_median_m": float(np.median(all_vertical)),
        "vertical_disagreement_p80_m": float(
            np.percentile(all_vertical, 80.0)
        ),
        "session_translation_delta_p80_m": float(
            np.percentile(session_translation, 80.0)
        ),
        "session_yaw_delta_p80_deg": float(np.percentile(session_yaw, 80.0)),
        "folds": folds,
    }
    if (
        aggregate["translation_median_m"]
        > settings.maximum_heldout_translation_median_m
        or aggregate["translation_p80_m"]
        > settings.maximum_heldout_translation_p80_m
        or aggregate["yaw_median_deg"] > settings.maximum_heldout_yaw_median_deg
        or aggregate["yaw_p80_deg"] > settings.maximum_heldout_yaw_p80_deg
    ):
        failures.append("heldout_observation_error_exceeded")
    if (
        aggregate["session_translation_delta_p80_m"]
        > settings.maximum_leaveout_session_translation_p80_m
        or aggregate["session_yaw_delta_p80_deg"]
        > settings.maximum_leaveout_session_yaw_p80_deg
    ):
        failures.append("leaveout_session_transform_unstable")
    return aggregate, sorted(set(failures))


def _temporal_segments(
    moving_views: Sequence[int], gap_threshold: int
) -> list[list[int]]:
    if not moving_views:
        return []
    segments = [[moving_views[0]]]
    for view in moving_views[1:]:
        if view - segments[-1][-1] > gap_threshold:
            segments.append([view])
        else:
            segments[-1].append(view)
    return segments


def _leave_one_temporal_segment_out(
    rows: Sequence[_PreparedObservation],
    moving_poses: Mapping[int, np.ndarray],
    fixed_poses: Mapping[int, np.ndarray],
    intra_session_edges: Sequence[IntraSessionPoseObservation],
    vertical_translation_m: float,
    full_solution: _NumericSolution,
    settings: PoseGraphSettings,
) -> tuple[dict[str, Any], list[str]]:
    """Validate early/late overlap segments as independent evidence blocks."""

    observed_views = sorted({row.source.moving_view for row in rows})
    segments = _temporal_segments(
        observed_views, settings.temporal_segment_gap_views
    )
    if len(segments) < 2:
        return (
            {
                "applicable": False,
                "reason": "only_one_temporal_overlap_segment",
                "segments": segments,
                "folds": [],
            },
            [],
        )

    folds: list[dict[str, Any]] = []
    failures: list[str] = []
    for segment_index, segment in enumerate(segments):
        heldout_set = set(segment)
        train = [row for row in rows if row.source.moving_view not in heldout_set]
        heldout = [row for row in rows if row.source.moving_view in heldout_set]
        support = _support_metrics(train, moving_poses, fixed_poses)
        if (
            len(support["moving_views"]) < 2
            or len(support["fixed_views"]) < 2
            or support["distinct_pair_count"] < 3
        ):
            failures.append("segment_holdout_training_support_collapsed")
            folds.append(
                {
                    "heldout_segment": segment_index,
                    "heldout_moving_views": segment,
                    "passed": False,
                    "reason": "training_support_collapsed",
                }
            )
            continue
        candidate = _numeric_solve(
            train,
            moving_poses,
            fixed_poses,
            intra_session_edges,
            vertical_translation_m,
            settings,
        )
        metrics = _observation_metrics(
            heldout,
            candidate,
            moving_poses,
            fixed_poses,
            vertical_translation_m,
        )
        global_yaw, global_translation = _parameter_distance(
            candidate.global_parameters, full_solution.global_parameters
        )
        passed = bool(
            candidate.success
            and metrics["translation_median_m"]
            <= settings.maximum_segment_heldout_translation_median_m
            and metrics["translation_p80_m"]
            <= settings.maximum_segment_heldout_translation_p80_m
            and metrics["yaw_median_deg"]
            <= settings.maximum_segment_heldout_yaw_median_deg
            and metrics["yaw_p80_deg"]
            <= settings.maximum_segment_heldout_yaw_p80_deg
            and global_translation
            <= settings.maximum_segment_leaveout_session_translation_m
            and global_yaw <= settings.maximum_segment_leaveout_session_yaw_deg
        )
        if not passed:
            failures.append("temporal_segment_holdout_failed")
        folds.append(
            {
                "heldout_segment": segment_index,
                "heldout_moving_views": segment,
                "passed": passed,
                "optimizer_success": candidate.success,
                "translation_median_m": metrics["translation_median_m"],
                "translation_p80_m": metrics["translation_p80_m"],
                "yaw_median_deg": metrics["yaw_median_deg"],
                "yaw_p80_deg": metrics["yaw_p80_deg"],
                "session_translation_delta_m": global_translation,
                "session_yaw_delta_deg": global_yaw,
            }
        )
    return (
        {
            "applicable": True,
            "segment_gap_threshold_views": settings.temporal_segment_gap_views,
            "segments": segments,
            "folds": folds,
        },
        sorted(set(failures)),
    )


def _settings_report(settings: PoseGraphSettings) -> dict[str, Any]:
    return {
        name: value
        for name, value in settings.__dict__.items()
    }


def solve_cross_session_pose_graph(
    observations: Sequence[PnPTransformObservation],
    moving_camera_poses: Mapping[int, np.ndarray] | Sequence[np.ndarray],
    fixed_camera_poses: Mapping[int, np.ndarray] | Sequence[np.ndarray],
    *,
    intra_session_edges: Sequence[IntraSessionPoseObservation] = (),
    vertical_translation_m: float = 0.0,
    settings: PoseGraphSettings | None = None,
) -> PoseGraphResult:
    """Solve and validate a moving-session to fixed-session PCF alignment.

    The returned transforms are safe to consume only when ``accepted`` is
    true.  Invalid matrices or view references raise ``ValueError``; weak or
    inconsistent geometric evidence returns a rejected result with reason
    codes and diagnostics.
    """

    active_settings = settings or PoseGraphSettings()
    if not math.isfinite(vertical_translation_m):
        raise ValueError("vertical_translation_m must be finite")
    moving_poses = _normalize_poses(moving_camera_poses, "moving_camera_poses")
    fixed_poses = _normalize_poses(fixed_camera_poses, "fixed_camera_poses")
    normalized_edges = _normalize_intra_session_edges(
        intra_session_edges, moving_poses
    )
    screened, screening_report = _prepare_observations(
        observations,
        moving_poses,
        fixed_poses,
        active_settings,
    )
    consensus = _select_consensus(screened, active_settings)
    support = _support_metrics(consensus, moving_poses, fixed_poses)
    failures = _support_failures(support, active_settings)
    base_report: dict[str, Any] = {
        "schema": "noesis.pcf_multiroom_pose_graph.v1",
        "transform_convention": (
            "column-vector; observation maps moving baseline world to fixed "
            "baseline world; fixed session is immutable anchor"
        ),
        "method": (
            "floor-locked planar SE(2) session transform plus smoothly "
            "regularized per-moving-view planar drift; no point-cloud ICP"
        ),
        "settings": _settings_report(active_settings),
        "screening": screening_report,
        "consensus_support": support,
        "consensus_observation_count": len(consensus),
        "intra_session_edge_count": len(normalized_edges),
        "vertical_translation_m": vertical_translation_m,
        "vertical_translation_authority": "supplied_floor_to_floor_offset_locked",
    }
    if failures:
        base_report["accepted"] = False
        base_report["reason_codes"] = sorted(set(failures))
        return PoseGraphResult(
            accepted=False,
            reason_codes=tuple(sorted(set(failures))),
            global_transform_moving_to_fixed=None,
            report=base_report,
        )

    solution = _numeric_solve(
        consensus,
        moving_poses,
        fixed_poses,
        normalized_edges,
        vertical_translation_m,
        active_settings,
    )
    train = _observation_metrics(
        consensus,
        solution,
        moving_poses,
        fixed_poses,
        vertical_translation_m,
    )
    deformation = _deformation_metrics(
        solution, moving_poses, vertical_translation_m
    )
    if not solution.success:
        failures.append("optimizer_failed")
    if (
        train["translation_median_m"]
        > active_settings.maximum_train_translation_median_m
        or train["translation_p80_m"]
        > active_settings.maximum_train_translation_p80_m
        or train["yaw_median_deg"] > active_settings.maximum_train_yaw_median_deg
        or train["yaw_p80_deg"] > active_settings.maximum_train_yaw_p80_deg
    ):
        failures.append("training_observation_error_exceeded")
    if (
        deformation["position_p80_m"]
        > active_settings.maximum_view_deformation_p80_m
        or deformation["position_max_m"]
        > active_settings.maximum_view_deformation_max_m
        or deformation["yaw_p80_deg"]
        > active_settings.maximum_view_yaw_deformation_p80_deg
        or deformation["yaw_max_deg"]
        > active_settings.maximum_view_yaw_deformation_max_deg
    ):
        failures.append("per_view_deformation_exceeded")

    leaveout, leaveout_failures = _leave_one_moving_view_out(
        consensus,
        moving_poses,
        fixed_poses,
        normalized_edges,
        vertical_translation_m,
        solution,
        active_settings,
    )
    failures.extend(leaveout_failures)
    segment_leaveout, segment_failures = _leave_one_temporal_segment_out(
        consensus,
        moving_poses,
        fixed_poses,
        normalized_edges,
        vertical_translation_m,
        solution,
        active_settings,
    )
    failures.extend(segment_failures)
    failures = sorted(set(failures))

    global_candidate = _planar_transform(
        solution.global_parameters, vertical_translation_m
    )
    per_view_candidate = {
        index: _planar_transform(parameters, vertical_translation_m)
        for index, parameters in solution.per_view_parameters.items()
    }
    candidate_report = {
        "optimizer_success": solution.success,
        "optimizer_message": solution.message,
        "optimizer_cost": solution.cost,
        "optimizer_evaluations": solution.evaluations,
        "global_transform_moving_to_fixed_row_major": global_candidate.tolist(),
        "global_yaw_deg": math.degrees(float(solution.global_parameters[0])),
        "global_translation_m": global_candidate[:3, 3].tolist(),
        "optimized_degrees_of_freedom": ["yaw", "translation_x", "translation_z"],
        "locked_vertical_translation_m": vertical_translation_m,
        "per_view_transforms_moving_to_fixed_row_major": {
            str(index): transform.tolist()
            for index, transform in per_view_candidate.items()
        },
        "training_observation_error": train,
        "per_view_deformation_from_global": deformation,
        "leave_one_whole_moving_view_out": leaveout,
        "leave_one_temporal_overlap_segment_out": segment_leaveout,
    }
    base_report.update(
        {
            "accepted": not failures,
            "reason_codes": failures,
            "candidate": candidate_report,
        }
    )
    if failures:
        return PoseGraphResult(
            accepted=False,
            reason_codes=tuple(failures),
            global_transform_moving_to_fixed=None,
            report=base_report,
        )

    corrected_poses = {
        index: per_view_candidate[index] @ pose
        for index, pose in moving_poses.items()
    }
    return PoseGraphResult(
        accepted=True,
        reason_codes=(),
        global_transform_moving_to_fixed=global_candidate,
        per_view_transforms_moving_to_fixed=per_view_candidate,
        corrected_moving_camera_poses=corrected_poses,
        report=base_report,
    )


__all__ = [
    "IntraSessionPoseObservation",
    "PnPTransformObservation",
    "PoseGraphResult",
    "PoseGraphSettings",
    "solve_cross_session_pose_graph",
]
