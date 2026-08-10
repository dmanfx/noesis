#!/usr/bin/env python3
"""Build a conservative shared reconstruction from MapAnything and DA3.

This is an offline review builder. It preserves both model outputs and writes a
third result assembled from a common pose graph, common camera rays, separately
calibrated confidence, multiview depth consistency, disagreement gating, and
weighted surfel fusion. Static-camera data is intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mapanything_phone_scan.inference import (  # noqa: E402
    _write_reconstruction_glb,
)


@dataclass
class Sequence:
    name: str
    depth: np.ndarray
    confidence: np.ndarray
    mask: np.ndarray
    rgb: np.ndarray
    poses: np.ndarray
    intrinsics: np.ndarray


@dataclass
class Consistency:
    score: np.ndarray
    support: np.ndarray
    median_error_m: float
    p80_error_m: float


def _load_sequence(raw_root: Path, name: str) -> Sequence:
    paths = sorted(raw_root.glob("view_*.npz"))
    if len(paths) < 2:
        raise ValueError(f"{name} raw views are missing from {raw_root}")
    depth: list[np.ndarray] = []
    confidence: list[np.ndarray] = []
    mask: list[np.ndarray] = []
    rgb: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    for path in paths:
        with np.load(path) as row:
            depth.append(np.asarray(row["depth_z"], dtype=np.float32))
            confidence.append(np.asarray(row["confidence"], dtype=np.float32))
            mask.append(np.asarray(row["mask"], dtype=bool))
            image = np.asarray(row["model_rgb"])
            if image.dtype != np.uint8:
                image = np.clip(
                    image * 255.0 if float(np.nanmax(image)) <= 1.5 else image,
                    0,
                    255,
                ).astype(np.uint8)
            rgb.append(image)
            poses.append(np.asarray(row["camera_pose"], dtype=np.float64))
            intrinsics.append(np.asarray(row["intrinsics"], dtype=np.float64))
    return Sequence(
        name=name,
        depth=np.stack(depth),
        confidence=np.stack(confidence),
        mask=np.stack(mask),
        rgb=np.stack(rgb),
        poses=np.stack(poses),
        intrinsics=np.stack(intrinsics),
    )


def _scaled_intrinsics(
    intrinsics: np.ndarray,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> np.ndarray:
    source_height, source_width = source_hw
    target_height, target_width = target_hw
    scale = np.asarray(
        [
            [target_width / source_width, 0.0, 0.0],
            [0.0, target_height / source_height, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return np.einsum("ij,njk->nik", scale, intrinsics)


def _common_intrinsics(
    first: Sequence,
    second: Sequence,
    target_hw: tuple[int, int],
) -> np.ndarray:
    first_scaled = _scaled_intrinsics(first.intrinsics, first.depth.shape[1:], target_hw)
    second_scaled = _scaled_intrinsics(second.intrinsics, second.depth.shape[1:], target_hw)
    common = 0.5 * (first_scaled + second_scaled)
    common[:, 0, 1:] = np.stack(
        [np.zeros(common.shape[0]), common[:, 0, 2]], axis=1
    )
    common[:, 1, 0] = 0.0
    common[:, 1, 2] = 0.5 * (
        first_scaled[:, 1, 2] + second_scaled[:, 1, 2]
    )
    common[:, 2] = np.asarray([0.0, 0.0, 1.0])
    return common


def _remap_to_common_rays(
    sequence: Sequence,
    common_intrinsics: np.ndarray,
    target_hw: tuple[int, int],
) -> Sequence:
    target_height, target_width = target_hw
    uu, vv = np.meshgrid(
        np.arange(target_width, dtype=np.float64),
        np.arange(target_height, dtype=np.float64),
    )
    pixels = np.stack((uu, vv, np.ones_like(uu)), axis=0).reshape(3, -1)
    depth: list[np.ndarray] = []
    confidence: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    images: list[np.ndarray] = []
    for index in range(sequence.depth.shape[0]):
        rays = np.linalg.inv(common_intrinsics[index]) @ pixels
        projected = sequence.intrinsics[index] @ rays
        map_x = (projected[0] / projected[2]).reshape(target_hw).astype(np.float32)
        map_y = (projected[1] / projected[2]).reshape(target_hw).astype(np.float32)
        remapped_depth = cv2.remap(
            sequence.depth[index],
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        remapped_confidence = cv2.remap(
            sequence.confidence[index],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        remapped_mask = cv2.remap(
            sequence.mask[index].astype(np.uint8),
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        remapped_rgb = cv2.remap(
            sequence.rgb[index],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        remapped_mask &= np.isfinite(remapped_depth) & (remapped_depth > 0.05)
        depth.append(remapped_depth.astype(np.float32))
        confidence.append(remapped_confidence.astype(np.float32))
        masks.append(remapped_mask)
        images.append(remapped_rgb)
    return Sequence(
        name=sequence.name,
        depth=np.stack(depth),
        confidence=np.stack(confidence),
        mask=np.stack(masks),
        rgb=np.stack(images),
        poses=sequence.poses.copy(),
        intrinsics=common_intrinsics.copy(),
    )


def _umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1
    rotation = u @ sign @ vt
    variance = float(np.sum(source_centered * source_centered) / source.shape[0])
    scale = float(np.sum(singular * np.diag(sign)) / max(variance, 1e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _align_poses_to_reference(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    scale, rotation, translation = _umeyama(
        source[:, :3, 3], target[:, :3, 3]
    )
    aligned = source.copy()
    aligned[:, :3, :3] = np.einsum("ij,njk->nik", rotation, source[:, :3, :3])
    aligned[:, :3, 3] = (
        scale * (rotation @ source[:, :3, 3].T).T + translation
    )
    residual = np.linalg.norm(aligned[:, :3, 3] - target[:, :3, 3], axis=1)
    return aligned, {
        "scale": scale,
        "rotation_row_major": rotation.tolist(),
        "translation": translation.tolist(),
        "position_residual_median_m": float(np.median(residual)),
        "position_residual_p80_m": float(np.percentile(residual, 80.0)),
    }


def _relative_to_first(poses: np.ndarray) -> np.ndarray:
    origin_from_world = np.linalg.inv(poses[0])
    return np.stack([origin_from_world @ pose for pose in poses])


def _average_pose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    relative_rotation = first[:3, :3].T @ second[:3, :3]
    half_delta = Rotation.from_rotvec(
        0.5 * Rotation.from_matrix(relative_rotation).as_rotvec()
    ).as_matrix()
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = first[:3, :3] @ half_delta
    result[:3, 3] = 0.5 * (first[:3, 3] + second[:3, 3])
    return result


def _pose_graph(
    first_poses: np.ndarray,
    second_poses: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    first = _relative_to_first(first_poses)
    second = _relative_to_first(second_poses)
    count = first.shape[0]
    initial = np.stack([_average_pose(first[i], second[i]) for i in range(count)])
    initial[0] = np.eye(4)

    edges: list[tuple[int, int, np.ndarray, float, float, str]] = []
    for model_name, poses in (("mapanything", first), ("da3", second)):
        for span, translation_sigma, rotation_sigma_deg in (
            (1, 0.10, 3.0),
            (2, 0.16, 5.0),
            (4, 0.26, 8.0),
        ):
            for index in range(count - span):
                observed = np.linalg.inv(poses[index]) @ poses[index + span]
                edges.append(
                    (
                        index,
                        index + span,
                        observed,
                        translation_sigma,
                        math.radians(rotation_sigma_deg),
                        model_name,
                    )
                )

    def encode(poses: np.ndarray) -> np.ndarray:
        rows = []
        for pose in poses[1:]:
            rows.append(
                np.concatenate(
                    [Rotation.from_matrix(pose[:3, :3]).as_rotvec(), pose[:3, 3]]
                )
            )
        return np.concatenate(rows)

    def decode(parameters: np.ndarray) -> np.ndarray:
        result = np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))
        rows = parameters.reshape(count - 1, 6)
        result[1:, :3, :3] = Rotation.from_rotvec(rows[:, :3]).as_matrix()
        result[1:, :3, 3] = rows[:, 3:]
        return result

    def residual(parameters: np.ndarray) -> np.ndarray:
        poses = decode(parameters)
        values: list[np.ndarray] = []
        for i, j, observed, translation_sigma, rotation_sigma, _ in edges:
            predicted_rotation = poses[i, :3, :3].T @ poses[j, :3, :3]
            predicted_translation = poses[i, :3, :3].T @ (
                poses[j, :3, 3] - poses[i, :3, 3]
            )
            rotation_error = Rotation.from_matrix(
                observed[:3, :3].T @ predicted_rotation
            ).as_rotvec()
            values.append(rotation_error / rotation_sigma)
            values.append(
                (predicted_translation - observed[:3, 3]) / translation_sigma
            )
        for index in range(1, count):
            rotation_error = Rotation.from_matrix(
                initial[index, :3, :3].T @ poses[index, :3, :3]
            ).as_rotvec()
            values.append(rotation_error / math.radians(15.0))
            values.append((poses[index, :3, 3] - initial[index, :3, 3]) / 0.60)
        # The capture returned near its start. Position is constrained softly;
        # orientation is deliberately not forced because the phone faced away.
        values.append(poses[-1, :3, 3] / 0.16)
        return np.concatenate(values)

    residual_count = len(edges) * 6 + (count - 1) * 6 + 3
    variable_count = (count - 1) * 6
    jacobian = sparse.lil_matrix((residual_count, variable_count), dtype=np.int8)
    row = 0
    for i, j, *_ in edges:
        if i > 0:
            jacobian[row : row + 6, (i - 1) * 6 : i * 6] = 1
        if j > 0:
            jacobian[row : row + 6, (j - 1) * 6 : j * 6] = 1
        row += 6
    for index in range(1, count):
        jacobian[row : row + 6, (index - 1) * 6 : index * 6] = 1
        row += 6
    jacobian[row : row + 3, (count - 2) * 6 : (count - 1) * 6] = 1

    before = float(np.linalg.norm(initial[-1, :3, 3]))
    result = least_squares(
        residual,
        encode(initial),
        jac_sparsity=jacobian.tocsr(),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=90,
        verbose=0,
    )
    optimized = decode(result.x)
    after = float(np.linalg.norm(optimized[-1, :3, 3]))
    return optimized, {
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "edge_count": len(edges),
        "start_end_before_m": before,
        "start_end_after_m": after,
        "loop_closure_kind": "soft_position_only",
        "loop_translation_sigma_m": 0.16,
    }


def _confidence_cdf(sequence: Sequence) -> tuple[np.ndarray, dict[str, Any]]:
    samples = []
    for index in range(sequence.depth.shape[0]):
        values = sequence.confidence[index][sequence.mask[index]]
        if values.size:
            samples.append(values[:: max(1, values.size // 12_000)])
    combined = np.concatenate(samples)
    probabilities = np.linspace(0.01, 0.99, 33)
    quantiles = np.quantile(combined, probabilities)
    quantiles = np.maximum.accumulate(quantiles)
    calibrated = np.zeros_like(sequence.confidence, dtype=np.float32)
    for index in range(sequence.depth.shape[0]):
        calibrated[index] = np.interp(
            sequence.confidence[index],
            quantiles,
            probabilities,
            left=0.01,
            right=0.99,
        ).astype(np.float32)
        calibrated[index][~sequence.mask[index]] = 0.0
    return calibrated, {
        "method": "within_model_empirical_cdf_33_quantiles",
        "sample_count": int(combined.size),
        "raw_quantiles": {
            "p02": float(np.quantile(combined, 0.02)),
            "p50": float(np.quantile(combined, 0.50)),
            "p98": float(np.quantile(combined, 0.98)),
        },
    }


def _normalize_depth_scale(
    reference: Sequence,
    candidate: Sequence,
    reference_confidence: np.ndarray,
    candidate_confidence: np.ndarray,
    global_scale: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    candidate_depth = candidate.depth * global_scale
    ratios: list[float] = []
    for index in range(reference.depth.shape[0]):
        valid = (
            reference.mask[index]
            & candidate.mask[index]
            & (reference_confidence[index] >= 0.45)
            & (candidate_confidence[index] >= 0.45)
            & (reference.depth[index] > 0.2)
            & (candidate_depth[index] > 0.2)
        )
        values = reference.depth[index][valid] / candidate_depth[index][valid]
        values = values[np.isfinite(values) & (values >= 0.70) & (values <= 1.30)]
        ratios.append(float(np.median(values)) if values.size >= 500 else 1.0)
    log_ratios = np.log(np.asarray(ratios, dtype=np.float64))
    from scipy.ndimage import gaussian_filter1d, median_filter

    smooth = gaussian_filter1d(median_filter(log_ratios, size=5, mode="nearest"), 1.2)
    # Preserve the global trajectory-derived metric scale and permit only a
    # small, smooth correction for per-view depth bias.
    smooth = np.clip(0.70 * smooth, math.log(0.90), math.log(1.10))
    per_view = np.exp(smooth).astype(np.float32)
    normalized = candidate_depth * per_view[:, None, None]
    return normalized.astype(np.float32), per_view, {
        "reference": reference.name,
        "candidate": candidate.name,
        "trajectory_sim3_scale": float(global_scale),
        "per_view_scale_min": float(np.min(per_view)),
        "per_view_scale_median": float(np.median(per_view)),
        "per_view_scale_max": float(np.max(per_view)),
        "regularization": "five_frame_median_then_gaussian_sigma_1.2_and_70_percent_shrinkage",
    }


def _neighbor_indices(index: int, count: int) -> list[int]:
    neighbors = {
        candidate
        for offset in (-2, -1, 1, 2)
        if 0 <= (candidate := index + offset) < count
    }
    if index == 0:
        neighbors.add(count - 1)
    elif index == count - 1:
        neighbors.add(0)
    return sorted(neighbors)


def _multiview_consistency(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    poses: np.ndarray,
) -> Consistency:
    count, height, width = depth.shape
    uu, vv = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    pixels = np.stack((uu, vv, np.ones_like(uu)), axis=-1).reshape(-1, 3)
    score = np.zeros_like(depth, dtype=np.float32)
    support = np.zeros_like(depth, dtype=np.uint8)
    sampled_errors: list[np.ndarray] = []
    for index in range(count):
        valid_flat = np.flatnonzero(mask[index].reshape(-1))
        if valid_flat.size == 0:
            continue
        rays = (np.linalg.inv(intrinsics[index]) @ pixels[valid_flat].T).T
        local = rays * depth[index].reshape(-1)[valid_flat, None]
        world = (
            poses[index, :3, :3] @ local.T
        ).T + poses[index, :3, 3]
        score_flat = score[index].reshape(-1)
        support_flat = support[index].reshape(-1)
        for neighbor in _neighbor_indices(index, count):
            target = (
                poses[neighbor, :3, :3].T
                @ (world - poses[neighbor, :3, 3]).T
            ).T
            positive = target[:, 2] > 0.05
            projected = (intrinsics[neighbor] @ target.T).T
            map_x = (projected[:, 0] / np.maximum(projected[:, 2], 1e-8)).astype(
                np.float32
            )
            map_y = (projected[:, 1] / np.maximum(projected[:, 2], 1e-8)).astype(
                np.float32
            )
            inside = (
                positive
                & (map_x >= 0)
                & (map_x <= width - 1)
                & (map_y >= 0)
                & (map_y <= height - 1)
            )
            finite_coordinates = np.isfinite(map_x) & np.isfinite(map_y)
            inside &= finite_coordinates
            safe_x = np.clip(
                np.nan_to_num(map_x, nan=0.0, posinf=0.0, neginf=0.0),
                -4.0 * width,
                4.0 * width,
            )
            safe_y = np.clip(
                np.nan_to_num(map_y, nan=0.0, posinf=0.0, neginf=0.0),
                -4.0 * height,
                4.0 * height,
            )
            sample_x = np.clip(np.rint(safe_x).astype(np.int32), 0, width - 1)
            sample_y = np.clip(np.rint(safe_y).astype(np.int32), 0, height - 1)
            sampled_depth = np.zeros(map_x.shape, dtype=np.float32)
            sampled_mask = np.zeros(map_x.shape, dtype=bool)
            sampled_depth[inside] = depth[neighbor, sample_y[inside], sample_x[inside]]
            sampled_mask[inside] = mask[neighbor, sample_y[inside], sample_x[inside]]
            tolerance = 0.08 + 0.02 * np.clip(target[:, 2], 0.0, 8.0)
            occluded = sampled_depth + tolerance < target[:, 2]
            comparable = inside & sampled_mask & ~occluded
            errors = np.abs(sampled_depth - target[:, 2])
            local_score = np.exp(-np.square(errors / np.maximum(tolerance, 1e-6)))
            selected = valid_flat[comparable]
            np.add.at(score_flat, selected, local_score[comparable].astype(np.float32))
            np.add.at(support_flat, selected, 1)
            if np.count_nonzero(comparable):
                sample = errors[comparable]
                sampled_errors.append(sample[:: max(1, sample.size // 8_000)])
    valid_support = support > 0
    score[valid_support] /= support[valid_support]
    score[~valid_support & mask] = 0.25
    all_errors = np.concatenate(sampled_errors) if sampled_errors else np.asarray([math.inf])
    return Consistency(
        score=score,
        support=support,
        median_error_m=float(np.median(all_errors)),
        p80_error_m=float(np.percentile(all_errors, 80.0)),
    )


def _depth_boundary_weight(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(depth, dtype=np.float32)
    for index in range(depth.shape[0]):
        log_depth = np.log(np.maximum(depth[index], 0.05))
        gx = cv2.Sobel(log_depth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(log_depth, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.hypot(gx, gy)
        result[index] = np.exp(-np.clip(gradient, 0.0, 3.0) / 0.55)
        result[index][~mask[index]] = 0.0
    return result


def _fuse_depth(
    first: Sequence,
    second: Sequence,
    first_depth: np.ndarray,
    second_depth: np.ndarray,
    first_confidence: np.ndarray,
    second_confidence: np.ndarray,
    first_consistency: Consistency,
    second_consistency: Consistency,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    first_reliability = np.sqrt(
        np.clip(first_confidence, 0.01, 1.0)
        * np.clip(first_consistency.score, 0.05, 1.0)
        * np.clip(_depth_boundary_weight(first_depth, first.mask), 0.05, 1.0)
    ).astype(np.float32)
    second_reliability = np.sqrt(
        np.clip(second_confidence, 0.01, 1.0)
        * np.clip(second_consistency.score, 0.05, 1.0)
        * np.clip(_depth_boundary_weight(second_depth, second.mask), 0.05, 1.0)
    ).astype(np.float32)
    both = first.mask & second.mask
    delta = np.abs(first_depth - second_depth)
    agreement_threshold = np.clip(
        0.08 + 0.015 * np.minimum(first_depth, second_depth), 0.10, 0.15
    )
    disagreement_threshold = np.clip(
        0.22 + 0.025 * np.minimum(first_depth, second_depth), 0.25, 0.35
    )
    agreement = both & (delta <= agreement_threshold)
    moderate = both & (delta > agreement_threshold) & (delta <= disagreement_threshold)
    uncertain = both & (delta > disagreement_threshold)

    fused = np.zeros_like(first_depth, dtype=np.float32)
    quality = np.zeros_like(first_depth, dtype=np.float32)
    source = np.zeros_like(first_depth, dtype=np.uint8)

    # With two observations, the weighted median is the observation whose
    # calibrated reliability crosses half the total weight. It cannot create a
    # synthetic surface between two edges.
    choose_first_agreement = agreement & (first_reliability >= second_reliability)
    choose_second_agreement = agreement & ~choose_first_agreement
    fused[choose_first_agreement] = first_depth[choose_first_agreement]
    fused[choose_second_agreement] = second_depth[choose_second_agreement]
    quality[agreement] = np.clip(
        0.5 * (first_reliability[agreement] + second_reliability[agreement]) + 0.20,
        0.0,
        1.0,
    )
    source[agreement] = 1

    choose_first_moderate = moderate & (
        (first_consistency.score > second_consistency.score)
        | (
            np.isclose(first_consistency.score, second_consistency.score, atol=0.02)
            & (first_reliability >= second_reliability)
        )
    )
    choose_second_moderate = moderate & ~choose_first_moderate
    fused[choose_first_moderate] = first_depth[choose_first_moderate]
    fused[choose_second_moderate] = second_depth[choose_second_moderate]
    quality[choose_first_moderate] = first_reliability[choose_first_moderate]
    quality[choose_second_moderate] = second_reliability[choose_second_moderate]
    source[choose_first_moderate] = 2
    source[choose_second_moderate] = 3

    first_only = first.mask & ~second.mask
    second_only = second.mask & ~first.mask
    accepted_first_only = (
        first_only
        & (first_reliability >= 0.48)
        & (first_consistency.score >= 0.42)
        & (first_consistency.support >= 1)
    )
    accepted_second_only = (
        second_only
        & (second_reliability >= 0.48)
        & (second_consistency.score >= 0.42)
        & (second_consistency.support >= 1)
    )
    fused[accepted_first_only] = first_depth[accepted_first_only]
    fused[accepted_second_only] = second_depth[accepted_second_only]
    quality[accepted_first_only] = first_reliability[accepted_first_only]
    quality[accepted_second_only] = second_reliability[accepted_second_only]
    source[accepted_first_only] = 4
    source[accepted_second_only] = 5

    valid = source > 0
    rgb = np.round(
        0.5 * first.rgb.astype(np.float32) + 0.5 * second.rgb.astype(np.float32)
    ).astype(np.uint8)
    valid_both_values = delta[both]
    metrics = {
        "pixel_count": int(fused.size),
        "valid_fraction": float(np.count_nonzero(valid) / valid.size),
        "both_valid_fraction": float(np.count_nonzero(both) / both.size),
        "agreement_fraction_of_both": float(
            np.count_nonzero(agreement) / max(1, np.count_nonzero(both))
        ),
        "moderate_selection_fraction_of_both": float(
            np.count_nonzero(moderate) / max(1, np.count_nonzero(both))
        ),
        "large_disagreement_fraction_of_both": float(
            np.count_nonzero(uncertain) / max(1, np.count_nonzero(both))
        ),
        "single_model_fill_fraction": float(
            np.count_nonzero(accepted_first_only | accepted_second_only) / valid.size
        ),
        "absolute_depth_disagreement_median_m": float(np.median(valid_both_values)),
        "absolute_depth_disagreement_p80_m": float(
            np.percentile(valid_both_values, 80.0)
        ),
        "source_counts": {
            "consensus_weighted_median": int(np.count_nonzero(source == 1)),
            "moderate_mapanything": int(np.count_nonzero(source == 2)),
            "moderate_da3": int(np.count_nonzero(source == 3)),
            "mapanything_only_fill": int(np.count_nonzero(source == 4)),
            "da3_only_fill": int(np.count_nonzero(source == 5)),
            "rejected_or_unknown": int(np.count_nonzero(source == 0)),
        },
    }
    return {
        "depth": fused,
        "quality": quality,
        "mask": valid,
        "source": source,
        "uncertain": uncertain,
        "agreement": agreement,
        "delta": delta,
        "rgb": rgb,
        "first_reliability": first_reliability,
        "second_reliability": second_reliability,
    }, metrics


def _world_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    pose: np.ndarray,
) -> np.ndarray:
    height, width = depth.shape
    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack((uu, vv, np.ones_like(uu)), axis=-1).reshape(-1, 3)
    rays = (np.linalg.inv(intrinsics) @ pixels.T).T
    local = rays * depth.reshape(-1, 1)
    world = (pose[:3, :3] @ local.T).T + pose[:3, 3]
    return world.reshape(height, width, 3).astype(np.float32)


def _surfel_fusion(
    fused: dict[str, np.ndarray],
    intrinsics: np.ndarray,
    poses: np.ndarray,
    voxel_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    point_rows: list[np.ndarray] = []
    color_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    for index in range(fused["depth"].shape[0]):
        points = _world_points(fused["depth"][index], intrinsics[index], poses[index])
        selected = fused["mask"][index] & (fused["quality"][index] >= 0.35)
        selected[1::2, :] = False
        selected[:, 1::2] = False
        selected &= np.isfinite(points).all(axis=-1)
        selected &= fused["depth"][index] <= 12.0
        point_rows.append(points[selected])
        color_rows.append(fused["rgb"][index][selected])
        weight_rows.append(fused["quality"][index][selected])
    points = np.concatenate(point_rows).astype(np.float64)
    colors = np.concatenate(color_rows).astype(np.float64)
    weights = np.concatenate(weight_rows).astype(np.float64)
    keys = np.floor(points / voxel_m).astype(np.int32)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    voxel_count = int(np.max(inverse)) + 1
    weight_sum = np.bincount(inverse, weights=weights, minlength=voxel_count)
    counts = np.bincount(inverse, minlength=voxel_count)
    fused_points = np.column_stack(
        [
            np.bincount(inverse, weights=points[:, axis] * weights, minlength=voxel_count)
            / np.maximum(weight_sum, 1e-8)
            for axis in range(3)
        ]
    )
    fused_colors = np.column_stack(
        [
            np.bincount(inverse, weights=colors[:, axis] * weights, minlength=voxel_count)
            / np.maximum(weight_sum, 1e-8)
            for axis in range(3)
        ]
    )
    keep = (counts >= 2) & (weight_sum >= 0.80)
    fused_points = fused_points[keep].astype(np.float32)
    fused_colors = np.clip(fused_colors[keep], 0, 255).astype(np.uint8)
    fused_weights = weight_sum[keep].astype(np.float32)
    return fused_points, fused_colors, fused_weights, {
        "method": "confidence_weighted_4cm_surfel_voxels",
        "voxel_m": float(voxel_m),
        "input_sample_count": int(points.shape[0]),
        "surfel_count": int(fused_points.shape[0]),
        "minimum_samples_per_surfel": 2,
        "minimum_accumulated_weight": 0.80,
    }


def _heldout_reprojection(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    poses: np.ndarray,
) -> dict[str, float]:
    count, height, width = depth.shape
    source_points: list[np.ndarray] = []
    for index in range(0, count, 2):
        points = _world_points(depth[index], intrinsics[index], poses[index])
        selected = mask[index].copy()
        selected[1::3, :] = False
        selected[2::3, :] = False
        selected[:, 1::3] = False
        selected[:, 2::3] = False
        selected &= np.isfinite(points).all(axis=-1) & (depth[index] <= 12.0)
        source_points.append(points[selected])
    world = np.concatenate(source_points).astype(np.float64)
    residuals: list[np.ndarray] = []
    coverages: list[float] = []
    for index in range(1, count, 2):
        local = (
            poses[index, :3, :3].T
            @ (world - poses[index, :3, 3]).T
        ).T
        positive = local[:, 2] > 0.05
        projected = (intrinsics[index] @ local.T).T
        u = np.round(projected[:, 0] / np.maximum(projected[:, 2], 1e-8)).astype(int)
        v = np.round(projected[:, 1] / np.maximum(projected[:, 2], 1e-8)).astype(int)
        inside = positive & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        flat_index = v[inside] * width + u[inside]
        zbuffer = np.full(height * width, np.inf, dtype=np.float32)
        np.minimum.at(zbuffer, flat_index, local[inside, 2].astype(np.float32))
        zbuffer = zbuffer.reshape(height, width)
        comparable = mask[index] & np.isfinite(zbuffer)
        if np.count_nonzero(comparable) < 100:
            continue
        error = np.abs(zbuffer[comparable] - depth[index][comparable])
        error = error[np.isfinite(error) & (error <= 2.0)]
        if error.size:
            residuals.append(error)
        coverages.append(
            float(np.count_nonzero(comparable) / max(1, np.count_nonzero(mask[index])))
        )
    values = np.concatenate(residuals) if residuals else np.asarray([math.inf])
    return {
        "even_frame_map_to_odd_frame_depth_median_m": float(np.median(values)),
        "even_frame_map_to_odd_frame_depth_p80_m": float(np.percentile(values, 80.0)),
        "odd_frame_valid_pixel_coverage_fraction": float(np.mean(coverages)) if coverages else 0.0,
    }


def _save_raw_frames(
    output_dir: Path,
    fused: dict[str, np.ndarray],
    intrinsics: np.ndarray,
    poses: np.ndarray,
) -> None:
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for index in range(fused["depth"].shape[0]):
        points = _world_points(fused["depth"][index], intrinsics[index], poses[index])
        np.savez_compressed(
            raw_dir / f"view_{index:04d}.npz",
            world_points=points,
            depth_z=fused["depth"][index].astype(np.float32),
            confidence=fused["quality"][index].astype(np.float32),
            mask=fused["mask"][index],
            camera_pose=poses[index].astype(np.float32),
            intrinsics=intrinsics[index].astype(np.float32),
            metric_scaling_factor=np.asarray([1.0], dtype=np.float32),
            model_rgb=fused["rgb"][index],
            source_selection=fused["source"][index],
            cross_model_uncertain=fused["uncertain"][index],
            cross_model_agreement=fused["agreement"][index],
            absolute_depth_disagreement_m=fused["delta"][index].astype(np.float32),
            mapanything_reliability=fused["first_reliability"][index].astype(np.float32),
            da3_reliability=fused["second_reliability"][index].astype(np.float32),
        )


def _colorize(values: np.ndarray, cmap: str, valid: np.ndarray) -> np.ndarray:
    finite = values[valid & np.isfinite(values)]
    low, high = np.percentile(finite, (2.0, 98.0)) if finite.size else (0.0, 1.0)
    normalized = np.clip((values - low) / max(float(high - low), 1e-8), 0.0, 1.0)
    image = np.round(plt.get_cmap(cmap)(normalized)[..., :3] * 255).astype(np.uint8)
    image[~valid] = (7, 8, 10)
    return image


def _collaboration_sheet(
    output_path: Path,
    first_depth: np.ndarray,
    second_depth: np.ndarray,
    fused: dict[str, np.ndarray],
    first_consistency: Consistency,
    second_consistency: Consistency,
    first_poses: np.ndarray,
    second_poses: np.ndarray,
    optimized_poses: np.ndarray,
) -> None:
    index = 0
    valid_first = first_depth[index] > 0
    valid_second = second_depth[index] > 0
    source_colors = np.asarray(
        [
            [8, 9, 12],
            [83, 220, 118],
            [63, 147, 255],
            [255, 151, 55],
            [108, 179, 255],
            [255, 205, 92],
        ],
        dtype=np.uint8,
    )
    source_image = source_colors[fused["source"][index]]
    uncertainty = np.zeros((*fused["uncertain"][index].shape, 3), dtype=np.uint8)
    uncertainty[fused["uncertain"][index]] = (255, 68, 105)
    uncertainty[fused["agreement"][index]] = (69, 224, 120)
    panels: list[tuple[str, np.ndarray]] = [
        ("MapAnything depth · common rays", _colorize(first_depth[index], "turbo_r", valid_first)),
        ("DA3 depth · normalized common rays", _colorize(second_depth[index], "turbo_r", valid_second)),
        ("Consensus fused depth", _colorize(fused["depth"][index], "turbo_r", fused["mask"][index])),
        ("Absolute model disagreement", _colorize(fused["delta"][index], "magma", valid_first & valid_second)),
        ("MapAnything calibrated reliability", _colorize(fused["first_reliability"][index], "viridis", valid_first)),
        ("DA3 calibrated reliability", _colorize(fused["second_reliability"][index], "viridis", valid_second)),
        ("MapAnything multiview consistency", _colorize(first_consistency.score[index], "viridis", valid_first)),
        ("DA3 multiview consistency", _colorize(second_consistency.score[index], "viridis", valid_second)),
        ("Selected source", source_image),
        ("Green agreement · pink rejected", uncertainty),
    ]
    trajectory = np.full((700, 700, 3), 10, dtype=np.uint8)
    paths = [
        (first_poses[:, :3, 3][:, [0, 2]], (82, 158, 255)),
        (second_poses[:, :3, 3][:, [0, 2]], (255, 155, 67)),
        (optimized_poses[:, :3, 3][:, [0, 2]], (73, 235, 119)),
    ]
    all_points = np.concatenate([row[0] for row in paths])
    low = np.min(all_points, axis=0) - 0.15
    high = np.max(all_points, axis=0) + 0.15
    span = np.maximum(high - low, 1e-6)
    for points, color in paths:
        pixels = np.round((points - low) / span * 620 + 40).astype(np.int32)
        cv2.polylines(trajectory, [pixels.reshape(-1, 1, 2)], False, color, 3, cv2.LINE_AA)
        cv2.circle(trajectory, tuple(pixels[0]), 7, color, -1)
        cv2.circle(trajectory, tuple(pixels[-1]), 7, (255, 80, 90), -1)
    panels.append(("Pose graph · blue MA · orange DA3 · green fused", trajectory))
    legend = np.full((700, 700, 3), 10, dtype=np.uint8)
    labels = [
        (1, "cross-model consensus"),
        (2, "MA selected by consistency"),
        (3, "DA3 selected by consistency"),
        (4, "MA-only validated fill"),
        (5, "DA3-only validated fill"),
        (0, "rejected / unknown"),
    ]
    for row, (code, label) in enumerate(labels):
        y = 90 + row * 90
        cv2.rectangle(legend, (70, y - 28), (120, y + 22), source_colors[code].tolist(), -1)
        cv2.putText(legend, label, (145, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (235, 240, 244), 2, cv2.LINE_AA)
    panels.append(("Source-selection legend", legend))

    figure, axes = plt.subplots(3, 4, figsize=(18, 14), facecolor="#0d1014")
    figure.suptitle(
        "Consensus Fusion · common rays, joint pose graph, reliability gating, surfel-ready depth",
        color="white",
        fontsize=18,
        y=0.992,
    )
    for axis, (title, image) in zip(axes.ravel(), panels):
        axis.imshow(image)
        axis.set_title(title, color="white", fontsize=10)
        axis.axis("off")
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.972), h_pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def build(
    scan_dir: Path,
    output_dir: Path,
    voxel_m: float = 0.04,
    mapanything_raw: Path | None = None,
    da3_raw: Path | None = None,
    pose_carrier: str = "joint",
) -> dict[str, Any]:
    started = time.perf_counter()
    mapanything_raw = (
        mapanything_raw.resolve()
        if mapanything_raw is not None
        else scan_dir / "outputs" / "raw"
    )
    da3_raw = (
        da3_raw.resolve()
        if da3_raw is not None
        else scan_dir / "da3_outputs" / "raw"
    )
    print("[1/8] Loading MapAnything and integrated DA3 raw outputs", flush=True)
    mapanything = _load_sequence(mapanything_raw, "MapAnything")
    da3 = _load_sequence(da3_raw, "DA3 Integrated")
    if mapanything.depth.shape[0] != da3.depth.shape[0]:
        raise ValueError("MapAnything and DA3 view counts differ")

    print("[2/8] Reprojecting both depth sets onto identical camera rays", flush=True)
    # DA3's processed grid preserves the capture orientation: 504x280 for the
    # validated portrait walk and 280x504 for a landscape walk.
    target_hw = tuple(int(value) for value in da3.depth.shape[1:])
    common_k = _common_intrinsics(mapanything, da3, target_hw)
    mapanything = _remap_to_common_rays(mapanything, common_k, target_hw)
    da3 = _remap_to_common_rays(da3, common_k, target_hw)

    print("[3/8] Aligning trajectory scales and optimizing the shared pose graph", flush=True)
    da3_aligned_poses, sim3 = _align_poses_to_reference(
        da3.poses, mapanything.poses
    )
    map_relative = _relative_to_first(mapanything.poses)
    da3_relative = _relative_to_first(da3_aligned_poses)
    optimized_poses, pose_graph_metrics = _pose_graph(
        mapanything.poses, da3_aligned_poses
    )

    print("[4/8] Calibrating each model confidence distribution independently", flush=True)
    map_confidence, map_confidence_metrics = _confidence_cdf(mapanything)
    da3_confidence, da3_confidence_metrics = _confidence_cdf(da3)
    da3_depth, da3_per_view_scale, depth_scale_metrics = _normalize_depth_scale(
        mapanything,
        da3,
        map_confidence,
        da3_confidence,
        float(sim3["scale"]),
    )

    if pose_carrier == "joint":
        output_poses = optimized_poses
        map_depth = mapanything.depth
        output_da3_depth = da3_depth
    elif pose_carrier == "da3":
        # Keep DA3's metric trajectory intact when it is the stronger static-
        # world registration carrier. Convert MapAnything and the already
        # normalized DA3 depths back into DA3 metric units before evaluating
        # consistency or fusing surfaces.
        output_poses = _relative_to_first(da3.poses)
        da3_metric_scale = (
            float(sim3["scale"]) * da3_per_view_scale
        ).astype(np.float32)
        map_depth = mapanything.depth / da3_metric_scale[:, None, None]
        output_da3_depth = da3_depth / da3_metric_scale[:, None, None]
    else:
        raise ValueError(f"unsupported pose carrier: {pose_carrier}")

    print("[5/8] Measuring multiview reprojection consistency", flush=True)
    map_consistency = _multiview_consistency(
        map_depth, mapanything.mask, common_k, output_poses
    )
    da3_consistency = _multiview_consistency(
        output_da3_depth, da3.mask, common_k, output_poses
    )

    print("[6/8] Gating agreement, selecting supported surfaces, and rejecting conflicts", flush=True)
    fused, fusion_metrics = _fuse_depth(
        mapanything,
        da3,
        map_depth,
        output_da3_depth,
        map_confidence,
        da3_confidence,
        map_consistency,
        da3_consistency,
    )
    fused_consistency = _multiview_consistency(
        fused["depth"], fused["mask"], common_k, output_poses
    )

    print("[7/8] Fusing accepted depths into weighted surfels", flush=True)
    surfel_points, surfel_colors, surfel_weights, surfel_metrics = _surfel_fusion(
        fused, common_k, output_poses, voxel_m
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "surfel_points.npz",
        points=surfel_points,
        colors=surfel_colors,
        weights=surfel_weights,
    )
    _write_reconstruction_glb(
        output_dir / "consensus_surfel_reconstruction.glb",
        surfel_points,
        surfel_colors,
        output_poses[:, :3, 3],
    )
    np.savez_compressed(
        output_dir / "camera_solution.npz",
        camera_poses=output_poses.astype(np.float32),
        intrinsics=common_k.astype(np.float32),
        da3_per_view_scale=da3_per_view_scale,
    )
    _save_raw_frames(output_dir, fused, common_k, output_poses)

    print("[8/8] Rendering collaboration evidence and validating held-out reprojection", flush=True)
    collaboration_path = output_dir / "consensus_collaboration_diagnostics.png"
    _collaboration_sheet(
        collaboration_path,
        map_depth,
        output_da3_depth,
        fused,
        map_consistency,
        da3_consistency,
        map_relative,
        da3_relative,
        output_poses,
    )
    heldout = {
        "mapanything": _heldout_reprojection(
            map_depth, mapanything.mask, common_k, output_poses
        ),
        "da3": _heldout_reprojection(
            output_da3_depth, da3.mask, common_k, output_poses
        ),
        "consensus": _heldout_reprojection(
            fused["depth"], fused["mask"], common_k, output_poses
        ),
    }
    metrics: dict[str, Any] = {
        "schema": "noesis.phone_walk.consensus_fusion.v1",
        "phone_walk_only": True,
        "static_camera_data_used": False,
        "pose_carrier": pose_carrier,
        "view_count": int(mapanything.depth.shape[0]),
        "inputs": {
            "mapanything_raw": str(mapanything_raw),
            "da3_raw": str(da3_raw),
        },
        "common_ray_grid": {
            "height": target_hw[0],
            "width": target_hw[1],
            "intrinsics_source": "per_frame_mean_of_model_intrinsics_after_resolution_scaling",
        },
        "trajectory_alignment": sim3,
        "pose_graph": pose_graph_metrics,
        "confidence_calibration": {
            "mapanything": map_confidence_metrics,
            "da3": da3_confidence_metrics,
        },
        "depth_scale_normalization": depth_scale_metrics,
        "multiview_consistency": {
            "mapanything": {
                "median_error_m": map_consistency.median_error_m,
                "p80_error_m": map_consistency.p80_error_m,
            },
            "da3": {
                "median_error_m": da3_consistency.median_error_m,
                "p80_error_m": da3_consistency.p80_error_m,
            },
            "consensus": {
                "median_error_m": fused_consistency.median_error_m,
                "p80_error_m": fused_consistency.p80_error_m,
            },
        },
        "fusion": fusion_metrics,
        "surfel_fusion": surfel_metrics,
        "heldout_even_to_odd_reprojection": heldout,
        "elapsed_s": float(time.perf_counter() - started),
        "artifacts": {
            "surfel_glb": "consensus_surfel_reconstruction.glb",
            "surfel_points": "surfel_points.npz",
            "camera_solution": "camera_solution.npz",
            "raw_frames": "raw",
            "collaboration_diagnostics": "consensus_collaboration_diagnostics.png",
        },
    }
    (output_dir / "consensus_manifest.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    output_contract = {
        "schema": "noesis.phone_scan.outputs.v2",
        "provider": "consensus_fusion",
        "model_id": "mapanything_plus_da3_consistency_gated_surfel_fusion",
        "mode": (
            "joint_pose_graph_common_rays_and_multiview_consensus"
            if pose_carrier == "joint"
            else "da3_pose_carried_common_rays_and_multiview_consensus"
        ),
        "coordinate_frame": "consensus_phone_metric_world_unaligned_to_noesis",
        "view_count": int(mapanything.depth.shape[0]),
        "phone_view_count": int(mapanything.depth.shape[0]),
        "anchor_view_index": None,
        "review_point_count": int(surfel_points.shape[0]),
        "artifacts": metrics["artifacts"],
        "consensus_manifest": "consensus_manifest.json",
    }
    (output_dir / "scan_outputs_manifest.json").write_text(
        json.dumps(output_contract, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mapanything-raw", type=Path)
    parser.add_argument("--da3-raw", type=Path)
    parser.add_argument("--voxel-m", type=float, default=0.04)
    parser.add_argument("--pose-carrier", choices=("joint", "da3"), default="joint")
    args = parser.parse_args()
    scan_dir = args.scan_dir.resolve()
    output_dir = (
        args.output_dir or scan_dir / "consensus_fusion"
    ).resolve()
    build(
        scan_dir,
        output_dir,
        args.voxel_m,
        mapanything_raw=args.mapanything_raw,
        da3_raw=args.da3_raw,
        pose_carrier=args.pose_carrier,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
