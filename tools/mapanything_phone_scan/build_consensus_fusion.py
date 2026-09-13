#!/usr/bin/env python3
"""Build a conservative shared reconstruction from MapAnything and DA3.

This is an offline review builder. It preserves both model outputs and writes a
third result assembled from a common pose graph, common camera rays, explicit
confidence rank scores, multiview depth consistency, disagreement gating, and
weighted surfel fusion. Static-camera data is intentionally excluded.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
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
from tools.mapanything_phone_scan.calibrated_inference import (  # noqa: E402
    calibration_raw_fields,
    calibration_specs,
    rectification_valid_mask,
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
    calibration_rows: list[dict[str, Any]] | None = None


@dataclass
class Consistency:
    score: np.ndarray
    support: np.ndarray
    median_error_m: float | None
    p80_error_m: float | None
    comparable_count: int = 0
    valid_pixel_count: int = 0


EVIDENCE_RELATIONSHIPS = (
    "da3_conditioned_mapanything",
    "same_capture_distinct_estimators",
    "independent",
    "unspecified",
)
SURFEL_SUPPORT_VIEW_ID_LIMIT = 16


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
    calibration_rows: list[dict[str, Any]] = []
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
            calibration_row = {}
            if "camera_intrinsics_json" in row:
                calibration_row = {
                    "camera_intrinsics": json.loads(str(row["camera_intrinsics_json"].item())),
                    "calibration_processing": json.loads(str(row["calibration_processing_json"].item())),
                }
                valid = np.asarray(row["rectification_valid_mask"], dtype=bool)
                if valid.shape != mask[-1].shape or np.any(mask[-1] & ~valid):
                    raise ValueError(f"{name} has invalid rectification borders: {path.name}")
            calibration_rows.append(calibration_row)
    specs = calibration_specs(calibration_rows)
    if specs is not None:
        manifest_path = raw_root.parent / "scan_outputs_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_rows = manifest.get("frames", [])
        if len(source_rows) != len(paths):
            raise ValueError(f"{name} calibrated raw views lack matching source identities")
        for index, (path, raw, source) in enumerate(zip(paths, calibration_rows, source_rows, strict=True)):
            if source.get("index") != index or Path(str(source.get("raw_npz"))).name != path.name:
                raise ValueError(f"{name} calibrated raw view order differs from its manifest")
            for key in ("camera_intrinsics", "calibration_processing"):
                if source.get(key) != raw.get(key):
                    raise ValueError(f"{name} raw {key} differs from its output manifest")
            for key in ("source_frame_id", "source_frame_sha256", "timestamp_s"):
                if source.get(key) is None:
                    raise ValueError(f"{name} calibrated output is missing {key}")
                raw[key] = source[key]
    return Sequence(
        name=name,
        depth=np.stack(depth),
        confidence=np.stack(confidence),
        mask=np.stack(mask),
        rgb=np.stack(rgb),
        poses=np.stack(poses),
        intrinsics=np.stack(intrinsics),
        calibration_rows=calibration_rows if specs is not None else None,
    )


def _consensus_calibration(
    scan_dir: Path, first: Sequence, second: Sequence,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    """Require matching retained calibration and preserve its authority limits."""
    prepared_path = scan_dir / "prepared_frames_manifest.json"
    prepared_bytes = prepared_path.read_bytes() if prepared_path.is_file() else b"{}"
    prepared = json.loads(prepared_bytes)
    frames = prepared.get("frames", [])
    specs = calibration_specs(frames)
    present = (specs is not None, first.calibration_rows is not None, second.calibration_rows is not None)
    if not any(present):
        return {}, None
    if not all(present):
        raise ValueError("consensus requires the same calibration on prepared views and both providers")
    for sequence in (first, second):
        if len(frames) != len(sequence.calibration_rows):
            raise ValueError("calibrated consensus view counts differ from prepared views")
        for index, (frame, raw) in enumerate(zip(frames, sequence.calibration_rows, strict=True)):
            for key in ("camera_intrinsics", "calibration_processing"):
                if raw.get(key) != frame.get(key):
                    raise ValueError(f"{sequence.name} view {index} {key} differs from prepared calibration")
            for source_key, prepared_key in (
                ("source_frame_id", "frame_id"), ("source_frame_sha256", "sha256"),
                ("timestamp_s", "timestamp_s"),
            ):
                if raw.get(source_key) != frame.get(prepared_key):
                    raise ValueError(f"{sequence.name} view {index} differs from prepared source identity")
    profile_path = scan_dir / "phone_camera_calibration.json"
    profile_bytes = profile_path.read_bytes()
    profile = json.loads(profile_bytes)
    profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
    if any(spec["profile_sha256"] != profile_sha256 or spec["profile_id"] != profile.get("profile_id") for spec in specs):
        raise ValueError("consensus calibration snapshot differs from prepared profile binding")
    return {
        "camera_calibration": deepcopy(prepared.get("camera_calibration", {})),
        "calibration_provenance": {
            "schema": "noesis.phone_walk.consensus_calibration.v1",
            "prepared_manifest_sha256": hashlib.sha256(prepared_bytes).hexdigest(),
            "profile_id": profile["profile_id"],
            "profile_sha256": profile_sha256,
            "evidence_kind": profile.get("evidence_kind"),
            "capture_binding_status": profile.get("capture_binding_status"),
            "geometry_intrinsics_source": "mean_of_provider_geometry_intrinsics_on_common_rays",
            "rectification_border_policy": "common_rays_project_inside_raw_source_with_full_opencv_D5",
            "calibration_admission_performed": False,
            "metric_vio_admission_performed": False,
        },
    }, deepcopy(frames)


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
        if sequence.calibration_rows is not None:
            remapped_mask &= rectification_valid_mask(
                sequence.calibration_rows[index]["camera_intrinsics"],
                common_intrinsics[index], target_hw,
            )
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
        calibration_rows=sequence.calibration_rows,
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
    verified_constraints: list[dict[str, Any]] | None = None,
    *,
    single_carrier: bool = False,
    single_carrier_name: str = "da3",
) -> tuple[np.ndarray, dict[str, Any]]:
    first = _relative_to_first(first_poses)
    second = _relative_to_first(second_poses)
    count = first.shape[0]
    initial = np.stack([_average_pose(first[i], second[i]) for i in range(count)])
    initial[0] = np.eye(4)

    edges: list[tuple[int, int, np.ndarray, float, float, str]] = []
    model_sequences = [("mapanything", first), ("da3", second)]
    if single_carrier:
        # A trajectory refinement has one retained carrier. Feeding that same
        # sequence twice would silently double its odometry weight while
        # pretending there are two estimators.
        model_sequences = [(single_carrier_name, second)]
    for model_name, poses in model_sequences:
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

    verified_count = 0
    for constraint_index, constraint in enumerate(verified_constraints or ()):
        if not isinstance(constraint, dict):
            raise ValueError(f"verified constraint {constraint_index} is not an object")
        try:
            source = int(constraint["source_view"])
            target = int(constraint["target_view"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"verified constraint {constraint_index} has invalid view IDs"
            ) from exc
        if source < 0 or target < 0 or source >= count or target >= count or source == target:
            raise ValueError(
                f"verified constraint {constraint_index} has out-of-range or equal views"
            )
        observed_value = constraint.get("observed", constraint.get("transform"))
        observed = np.asarray(observed_value, dtype=np.float64)
        if observed.shape != (4, 4) or not np.isfinite(observed).all():
            raise ValueError(
                f"verified constraint {constraint_index} transform is not finite 4x4"
            )
        if not np.allclose(observed[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError(
                f"verified constraint {constraint_index} has invalid homogeneous row"
            )
        rotation = observed[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
            raise ValueError(
                f"verified constraint {constraint_index} rotation is not orthonormal"
            )
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-4):
            raise ValueError(
                f"verified constraint {constraint_index} rotation is not proper"
            )
        translation_sigma = float(constraint.get("translation_sigma_m", 0.12))
        rotation_sigma_deg = float(constraint.get("rotation_sigma_deg", 5.0))
        if (
            not math.isfinite(translation_sigma)
            or translation_sigma <= 0.0
            or not math.isfinite(rotation_sigma_deg)
            or rotation_sigma_deg <= 0.0
        ):
            raise ValueError(
                f"verified constraint {constraint_index} has invalid uncertainty"
            )
        edges.append(
            (
                source,
                target,
                observed,
                translation_sigma,
                math.radians(rotation_sigma_deg),
                str(constraint.get("label") or "verified_constraint"),
            )
        )
        verified_count += 1

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
        return np.concatenate(values)

    residual_count = len(edges) * 6 + (count - 1) * 6
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
        "verified_constraint_count": verified_count,
        "start_end_before_m": before,
        "start_end_after_m": after,
        "loop_closure_kind": "verified_constraints_only",
        "endpoint_constraint_applied": False,
        "carrier_mode": f"single_{single_carrier_name}" if single_carrier else "joint_model_sequences",
    }


def _confidence_cdf(sequence: Sequence) -> tuple[np.ndarray, dict[str, Any]]:
    samples = []
    for index in range(sequence.depth.shape[0]):
        values = sequence.confidence[index][sequence.mask[index]]
        values = values[np.isfinite(values)]
        if values.size:
            samples.append(values[:: max(1, values.size // 12_000)])
    if not samples:
        return np.zeros_like(sequence.confidence, dtype=np.float32), {
            "method": "within_model_empirical_cdf_rank_score_33_quantiles",
            "semantics": (
                "uncalibrated_within_model_percentile_rank_score; "
                "not_a_probability_or_calibrated_reliability"
            ),
            "status": "empty_finite_masked_input",
            "sample_count": 0,
            "raw_quantiles": {},
        }
    combined = np.concatenate(samples)
    probabilities = np.linspace(0.01, 0.99, 33)
    quantiles = np.quantile(combined, probabilities)
    quantiles = np.maximum.accumulate(quantiles)
    rank_scores = np.zeros_like(sequence.confidence, dtype=np.float32)
    for index in range(sequence.depth.shape[0]):
        rank_scores[index] = np.interp(
            sequence.confidence[index],
            quantiles,
            probabilities,
            left=0.01,
            right=0.99,
        ).astype(np.float32)
        rank_scores[index][~sequence.mask[index]] = 0.0
        rank_scores[index][~np.isfinite(sequence.confidence[index])] = 0.0
    return rank_scores, {
        "method": "within_model_empirical_cdf_rank_score_33_quantiles",
        "semantics": (
            "uncalibrated_within_model_percentile_rank_score; "
            "not_a_probability_or_calibrated_reliability"
        ),
        "status": "ok",
        "sample_count": int(combined.size),
        "raw_quantiles": {
            "p02": float(np.quantile(combined, 0.02)),
            "p50": float(np.quantile(combined, 0.50)),
            "p98": float(np.quantile(combined, 0.98)),
        },
        "rank_score_range": [0.01, 0.99],
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
        valid_depth = (
            mask[index]
            & np.isfinite(depth[index])
            & (depth[index] > 0.05)
        )
        valid_flat = np.flatnonzero(valid_depth.reshape(-1))
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
            sampled_mask[inside] = (
                mask[neighbor, sample_y[inside], sample_x[inside]]
                & np.isfinite(sampled_depth[inside])
                & (sampled_depth[inside] > 0.05)
            )
            tolerance = 0.08 + 0.02 * np.clip(target[:, 2], 0.0, 8.0)
            occluded = sampled_depth + tolerance < target[:, 2]
            comparable = (
                inside
                & sampled_mask
                & np.isfinite(target[:, 2])
                & ~occluded
            )
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
    score[~valid_support & np.isfinite(depth) & (depth > 0.05) & mask] = 0.25
    all_errors = np.concatenate(sampled_errors) if sampled_errors else np.asarray([], dtype=np.float32)
    return Consistency(
        score=score,
        support=support,
        median_error_m=float(np.median(all_errors)) if all_errors.size else None,
        p80_error_m=float(np.percentile(all_errors, 80.0)) if all_errors.size else None,
        comparable_count=int(all_errors.size),
        valid_pixel_count=int(
            np.count_nonzero(mask & np.isfinite(depth) & (depth > 0.05))
        ),
    )


def _depth_boundary_weight(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(depth, dtype=np.float32)
    for index in range(depth.shape[0]):
        finite_depth = np.isfinite(depth[index])
        safe_depth = np.nan_to_num(
            depth[index], nan=0.05, posinf=0.05, neginf=0.05
        )
        log_depth = np.log(np.maximum(safe_depth, 0.05))
        gx = cv2.Sobel(log_depth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(log_depth, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.hypot(gx, gy)
        result[index] = np.exp(-np.clip(gradient, 0.0, 3.0) / 0.55)
        result[index][~mask[index] | ~finite_depth] = 0.0
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
    evidence_relationship: str = "da3_conditioned_mapanything",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if evidence_relationship not in EVIDENCE_RELATIONSHIPS:
        raise ValueError(
            f"unsupported evidence relationship: {evidence_relationship!r}"
        )
    first_valid = first.mask & np.isfinite(first_depth) & (first_depth > 0.05)
    second_valid = second.mask & np.isfinite(second_depth) & (second_depth > 0.05)
    first_evidence_weight = np.sqrt(
        np.clip(first_confidence, 0.01, 1.0)
        * np.clip(first_consistency.score, 0.05, 1.0)
        * np.clip(_depth_boundary_weight(first_depth, first_valid), 0.05, 1.0)
    ).astype(np.float32)
    second_evidence_weight = np.sqrt(
        np.clip(second_confidence, 0.01, 1.0)
        * np.clip(second_consistency.score, 0.05, 1.0)
        * np.clip(_depth_boundary_weight(second_depth, second_valid), 0.05, 1.0)
    ).astype(np.float32)
    both = first_valid & second_valid
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
    # source evidence weight crosses half the total weight. It cannot create a
    # synthetic surface between two edges.
    choose_first_agreement = agreement & (
        first_evidence_weight >= second_evidence_weight
    )
    choose_second_agreement = agreement & ~choose_first_agreement
    fused[choose_first_agreement] = first_depth[choose_first_agreement]
    fused[choose_second_agreement] = second_depth[choose_second_agreement]
    # These providers share the capture and, for the operational path, DA3 is
    # supplied to MapAnything. Agreement therefore does not create an
    # independent-evidence bonus. A max keeps the quality gate from treating
    # two correlated estimates as stronger merely because they agree.
    quality[agreement] = np.maximum(
        first_evidence_weight[agreement], second_evidence_weight[agreement]
    )
    source[agreement] = 1

    choose_first_moderate = moderate & (
        (first_consistency.score > second_consistency.score)
        | (
            np.isclose(first_consistency.score, second_consistency.score, atol=0.02)
            & (first_evidence_weight >= second_evidence_weight)
        )
    )
    choose_second_moderate = moderate & ~choose_first_moderate
    fused[choose_first_moderate] = first_depth[choose_first_moderate]
    fused[choose_second_moderate] = second_depth[choose_second_moderate]
    quality[choose_first_moderate] = first_evidence_weight[choose_first_moderate]
    quality[choose_second_moderate] = second_evidence_weight[choose_second_moderate]
    source[choose_first_moderate] = 2
    source[choose_second_moderate] = 3

    first_only = first_valid & ~second_valid
    second_only = second_valid & ~first_valid
    accepted_first_only = (
        first_only
        & (first_evidence_weight >= 0.48)
        & (first_consistency.score >= 0.42)
        & (first_consistency.support >= 1)
    )
    accepted_second_only = (
        second_only
        & (second_evidence_weight >= 0.48)
        & (second_consistency.score >= 0.42)
        & (second_consistency.support >= 1)
    )
    fused[accepted_first_only] = first_depth[accepted_first_only]
    fused[accepted_second_only] = second_depth[accepted_second_only]
    quality[accepted_first_only] = first_evidence_weight[accepted_first_only]
    quality[accepted_second_only] = second_evidence_weight[accepted_second_only]
    source[accepted_first_only] = 4
    source[accepted_second_only] = 5

    valid = source > 0
    # Both images come from the same exposure, but their inferred intrinsics
    # produce different warps into the common grid. Averaging those warps
    # duplicates edges and destroys the landmarks used by static-camera PnP.
    # Keep one explicit image projection; depth agreement remains independent.
    rgb = first.rgb.copy()
    valid_both_values = delta[both]
    valid_both_values = valid_both_values[np.isfinite(valid_both_values)]
    metrics = {
        "rgb_projection_policy": "single_reference_rgb_on_common_rays",
        "rgb_projection_source": first.name,
        "evidence_relationship": evidence_relationship,
        "agreement_quality_combination": (
            "maximum_correlated_source_weight_without_independence_bonus"
            if evidence_relationship != "independent"
            else "maximum_source_weight_no_bonus"
        ),
        "pixel_count": int(fused.size),
        "valid_fraction": float(np.count_nonzero(valid) / max(1, valid.size)),
        "both_valid_fraction": float(np.count_nonzero(both) / max(1, both.size)),
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
        "absolute_depth_disagreement_median_m": (
            float(np.median(valid_both_values)) if valid_both_values.size else None
        ),
        "absolute_depth_disagreement_p80_m": (
            float(np.percentile(valid_both_values, 80.0))
            if valid_both_values.size
            else None
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
        "first_evidence_weight": first_evidence_weight,
        "second_evidence_weight": second_evidence_weight,
        # Keep the old in-memory keys for callers that only consume the
        # pre-serialization fusion result. They are weights, not probabilities.
        # Preserve the historical array names for readers, but make their
        # evidence-weight semantics explicit; these are not calibrated
        # reliability probabilities.
        "first_reliability": first_evidence_weight,
        "second_reliability": second_evidence_weight,
        "reliability_semantics": "uncalibrated_evidence_weight_not_probability",
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
    view_rows: list[np.ndarray] = []
    for index in range(fused["depth"].shape[0]):
        points = _world_points(fused["depth"][index], intrinsics[index], poses[index])
        selected = fused["mask"][index] & (fused["quality"][index] >= 0.35)
        selected[1::2, :] = False
        selected[:, 1::2] = False
        selected &= np.isfinite(points).all(axis=-1)
        selected &= np.isfinite(fused["depth"][index]) & (
            fused["depth"][index] <= 12.0
        )
        point_rows.append(points[selected])
        color_rows.append(fused["rgb"][index][selected])
        weight_rows.append(fused["quality"][index][selected])
        view_rows.append(np.full(np.count_nonzero(selected), index, dtype=np.int32))
    if not point_rows or not any(row.size for row in point_rows):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0,), dtype=np.float32),
            {
                "method": "confidence_weighted_4cm_surfel_voxels",
                "voxel_m": float(voxel_m),
                "input_sample_count": 0,
                "pre_filter_voxel_count": 0,
                "surfel_count": 0,
                "minimum_samples_per_surfel": 2,
                "minimum_distinct_views_per_surfel": 2,
                "minimum_accumulated_weight": 0.80,
                "status": "empty_accepted_input",
                "support_view_id_limit": SURFEL_SUPPORT_VIEW_ID_LIMIT,
                "support_view_id_semantics": "zero_based_raw_view_index",
                "single_view_support_class": "single_view_withheld",
                "_support_view_count": np.empty((0,), dtype=np.uint16),
                "_support_view_ids": np.empty(
                    (0, SURFEL_SUPPORT_VIEW_ID_LIMIT), dtype=np.int32
                ),
                "_single_view_points": np.empty((0, 3), dtype=np.float32),
                "_single_view_colors": np.empty((0, 3), dtype=np.uint8),
                "_single_view_weights": np.empty((0,), dtype=np.float32),
                "_single_view_count": np.empty((0,), dtype=np.uint16),
                "_single_view_ids": np.empty(
                    (0, SURFEL_SUPPORT_VIEW_ID_LIMIT), dtype=np.int32
                ),
            },
        )
    points = np.concatenate(point_rows).astype(np.float64)
    colors = np.concatenate(color_rows).astype(np.float64)
    weights = np.concatenate(weight_rows).astype(np.float64)
    source_views = np.concatenate(view_rows)
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

    # ``counts`` counts pixels and can therefore be satisfied by one image.
    # Build support from (voxel, source-view) pairs so dense samples from one
    # camera never masquerade as multi-view geometric evidence.
    order = np.lexsort((source_views, inverse))
    ordered_voxels = inverse[order]
    ordered_views = source_views[order]
    pair_start = np.r_[
        True,
        (ordered_voxels[1:] != ordered_voxels[:-1])
        | (ordered_views[1:] != ordered_views[:-1]),
    ]
    unique_voxels = ordered_voxels[pair_start]
    unique_views = ordered_views[pair_start]
    voxel_start = np.flatnonzero(
        np.r_[True, unique_voxels[1:] != unique_voxels[:-1]]
    )
    run_lengths = np.diff(np.r_[voxel_start, unique_voxels.size])
    view_rank = np.arange(unique_views.size, dtype=np.int32) - np.repeat(
        voxel_start, run_lengths
    )
    distinct_view_count = np.bincount(
        unique_voxels, minlength=voxel_count
    ).astype(np.uint16)
    support_view_ids = np.full(
        (voxel_count, SURFEL_SUPPORT_VIEW_ID_LIMIT), -1, dtype=np.int32
    )
    retained_support = view_rank < SURFEL_SUPPORT_VIEW_ID_LIMIT
    support_view_ids[unique_voxels[retained_support], view_rank[retained_support]] = (
        unique_views[retained_support]
    )

    base_keep = (counts >= 2) & (weight_sum >= 0.80)
    single_view_keep = base_keep & (distinct_view_count < 2)
    keep = (
        (counts >= 2)
        & (distinct_view_count >= 2)
        & (weight_sum >= 0.80)
    )
    all_fused_points = fused_points
    all_fused_colors = fused_colors
    fused_points = all_fused_points[keep].astype(np.float32)
    fused_colors = np.clip(all_fused_colors[keep], 0, 255).astype(np.uint8)
    fused_weights = weight_sum[keep].astype(np.float32)
    kept_view_count = distinct_view_count[keep]
    kept_view_ids = support_view_ids[keep]
    single_view_points = all_fused_points[single_view_keep].astype(np.float32)
    single_view_colors = np.clip(
        all_fused_colors[single_view_keep], 0, 255
    ).astype(np.uint8)
    single_view_weights = weight_sum[~keep & single_view_keep].astype(np.float32)
    single_view_count = distinct_view_count[single_view_keep]
    single_view_ids = support_view_ids[single_view_keep]
    truncated_support_count = int(
        np.count_nonzero(distinct_view_count[keep] > SURFEL_SUPPORT_VIEW_ID_LIMIT)
    )
    return fused_points, fused_colors, fused_weights, {
        "method": "confidence_weighted_4cm_surfel_voxels",
        "voxel_m": float(voxel_m),
        "input_sample_count": int(points.shape[0]),
        "pre_filter_voxel_count": voxel_count,
        "pre_filter_single_view_voxel_count": int(
            np.count_nonzero(distinct_view_count < 2)
        ),
        "pre_filter_multi_view_voxel_count": int(
            np.count_nonzero(distinct_view_count >= 2)
        ),
        "single_view_rejected_voxel_count": int(single_view_points.shape[0]),
        "single_view_rejected_sample_count": int(
            np.sum(counts[single_view_keep], dtype=np.int64)
        ),
        "surfel_count": int(fused_points.shape[0]),
        "accepted_sample_fraction": float(
            np.sum(counts[keep], dtype=np.int64) / max(1, points.shape[0])
        ),
        "minimum_samples_per_surfel": 2,
        "minimum_distinct_views_per_surfel": 2,
        "minimum_accumulated_weight": 0.80,
        "source_view_count": int(np.unique(source_views).size),
        "support_view_count_min": int(np.min(kept_view_count))
        if kept_view_count.size
        else 0,
        "support_view_count_median": float(np.median(kept_view_count))
        if kept_view_count.size
        else 0.0,
        "support_view_count_max": int(np.max(kept_view_count))
        if kept_view_count.size
        else 0,
        "support_view_id_limit": SURFEL_SUPPORT_VIEW_ID_LIMIT,
        "support_view_id_semantics": "zero_based_raw_view_index",
        "support_view_ids_truncated_count": truncated_support_count,
        "single_view_support_class": "single_view_withheld",
        "status": "ok" if fused_points.size else "no_voxel_passed_support_gate",
        "_support_view_count": kept_view_count,
        "_support_view_ids": kept_view_ids,
        "_single_view_points": single_view_points,
        "_single_view_colors": single_view_colors,
        "_single_view_weights": single_view_weights,
        "_single_view_count": single_view_count,
        "_single_view_ids": single_view_ids,
    }


def _heldout_reprojection(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    poses: np.ndarray,
) -> dict[str, Any]:
    """Measure even-to-odd consistency without calling it independent data.

    Both parity sets are outputs of the same inference run and the supplied
    trajectory may have used every view. This is useful for detecting internal
    instability, but it is not an external accuracy or calibration estimate.
    """
    count, height, width = depth.shape
    source_points: list[np.ndarray] = []
    for index in range(0, count, 2):
        points = _world_points(depth[index], intrinsics[index], poses[index])
        selected = mask[index].copy()
        selected[1::3, :] = False
        selected[2::3, :] = False
        selected[:, 1::3] = False
        selected[:, 2::3] = False
        selected &= (
            np.isfinite(points).all(axis=-1)
            & np.isfinite(depth[index])
            & (depth[index] > 0.05)
            & (depth[index] <= 12.0)
        )
        source_points.append(points[selected])
    world = (
        np.concatenate(source_points).astype(np.float64)
        if any(row.size for row in source_points)
        else np.empty((0, 3), dtype=np.float64)
    )
    residuals: list[np.ndarray] = []
    coverages: list[float] = []
    frame_rows: list[dict[str, Any]] = []
    for index in range(1, count, 2):
        target_valid = mask[index] & np.isfinite(depth[index]) & (depth[index] > 0.05)
        target_count = int(np.count_nonzero(target_valid))
        frame_row: dict[str, Any] = {
            "frame_index": index,
            "target_valid_pixel_count": target_count,
            "comparable_pixel_count": 0,
            "status": "empty_source" if world.size == 0 else "no_comparable_pixels",
        }
        if world.size == 0:
            frame_rows.append(frame_row)
            continue
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
        comparable = target_valid & np.isfinite(zbuffer)
        comparable_count = int(np.count_nonzero(comparable))
        frame_row["comparable_pixel_count"] = comparable_count
        if target_count:
            frame_row["coverage_fraction"] = comparable_count / target_count
            coverages.append(comparable_count / target_count)
        if comparable_count == 0:
            frame_rows.append(frame_row)
            continue
        error = np.abs(zbuffer[comparable] - depth[index][comparable])
        error = error[np.isfinite(error)]
        if error.size:
            residuals.append(error)
            frame_row["finite_error_count"] = int(error.size)
            frame_row["large_error_count_gt_2m"] = int(np.count_nonzero(error > 2.0))
            frame_row["large_error_fraction_gt_2m"] = float(
                np.mean(error > 2.0)
            )
            frame_row["status"] = "ok"
        frame_rows.append(frame_row)
    values = np.concatenate(residuals) if residuals else np.asarray([], dtype=np.float32)
    finite_count = int(values.size)
    large_error_count = int(np.count_nonzero(values > 2.0)) if finite_count else 0
    evaluated_count = len(residuals)
    odd_frame_count = count // 2
    return {
        "evaluation_type": "internal_same_inference_even_to_odd_consistency",
        "independent": False,
        "alignment_fitted_on_validation_points": False,
        "source_parity": "even",
        "target_parity": "odd",
        "status": "ok" if finite_count else "empty_no_finite_comparisons",
        "large_error_threshold_m": 2.0,
        "finite_comparison_count": finite_count,
        "large_error_count_gt_2m": large_error_count,
        "large_error_fraction_gt_2m": (
            float(large_error_count / finite_count) if finite_count else None
        ),
        "even_frame_map_to_odd_frame_depth_median_m": (
            float(np.median(values)) if finite_count else None
        ),
        "even_frame_map_to_odd_frame_depth_p80_m": (
            float(np.percentile(values, 80.0)) if finite_count else None
        ),
        "odd_frame_valid_pixel_coverage_fraction": (
            float(np.mean(coverages)) if coverages else 0.0
        ),
        "odd_frame_count": odd_frame_count,
        "evaluated_odd_frame_count": evaluated_count,
        "skipped_odd_frame_count": odd_frame_count - evaluated_count,
        "frame_results": frame_rows,
    }


def _save_raw_frames(
    output_dir: Path,
    fused: dict[str, np.ndarray],
    intrinsics: np.ndarray,
    poses: np.ndarray,
    calibration_rows: list[dict[str, Any]] | None = None,
) -> None:
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for index in range(fused["depth"].shape[0]):
        points = _world_points(fused["depth"][index], intrinsics[index], poses[index])
        calibrated_fields = {}
        if calibration_rows is not None:
            valid = rectification_valid_mask(
                calibration_rows[index]["camera_intrinsics"], intrinsics[index],
                fused["depth"][index].shape,
            )
            if np.any(fused["mask"][index] & ~valid):
                raise ValueError("consensus mask includes rays outside calibrated source")
            calibrated_fields = calibration_raw_fields(calibration_rows[index], valid)
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
            mapanything_evidence_weight=fused["first_evidence_weight"][index].astype(
                np.float32
            ),
            da3_evidence_weight=fused["second_evidence_weight"][index].astype(
                np.float32
            ),
            # Retain the historical names for raw/reintegration readers. These
            # are evidence weights, not calibrated probabilities.
            mapanything_reliability=fused["first_evidence_weight"][index].astype(
                np.float32
            ),
            da3_reliability=fused["second_evidence_weight"][index].astype(
                np.float32
            ),
            **calibrated_fields,
        )


def _infer_evidence_relationship(raw_root: Path) -> tuple[str, dict[str, Any]]:
    """Read small sidecar manifests to state whether model evidence is shared."""
    manifest_candidates = (
        raw_root.parent / "variant_manifest.json",
        raw_root.parent / "scan_outputs_manifest.json",
    )
    for manifest_path in manifest_candidates:
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid raw-output manifest: {manifest_path}") from exc
        inputs = manifest.get("inputs")
        if isinstance(inputs, dict) and (
            inputs.get("uses_da3_pose_prior") is True
            or inputs.get("uses_da3_sparse_depth_prior") is True
        ):
            return "da3_conditioned_mapanything", {
                "manifest": str(manifest_path),
                "conditioned_by_da3": True,
            }
        return "same_capture_distinct_estimators", {
            "manifest": str(manifest_path),
            "conditioned_by_da3": False,
        }
    return "unspecified", {
        "manifest": None,
        "conditioned_by_da3": None,
    }


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
        ("MapAnything evidence weight", _colorize(fused["first_evidence_weight"][index], "viridis", valid_first)),
        ("DA3 evidence weight", _colorize(fused["second_evidence_weight"][index], "viridis", valid_second)),
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
        "Consensus Fusion · common rays, joint pose graph, evidence gating, surfel-ready depth",
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


def _write_support_diagnostics(
    output_path: Path,
    accepted_points: np.ndarray,
    single_view_points: np.ndarray,
) -> None:
    """Render accepted and withheld support classes over one shared X/Z box."""
    all_points = np.concatenate(
        [points for points in (accepted_points, single_view_points) if points.size],
        axis=0,
    ) if accepted_points.size or single_view_points.size else np.empty((0, 3))
    if all_points.size:
        low = np.min(all_points[:, [0, 2]], axis=0) - 0.05
        high = np.max(all_points[:, [0, 2]], axis=0) + 0.05
    else:
        low = np.asarray([-1.0, -1.0])
        high = np.asarray([1.0, 1.0])
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="#0d1014")
    for axis, points, title, color in (
        (axes[0], accepted_points, "Accepted · 2+ distinct views", "#4fdd91"),
        (axes[1], single_view_points, "Withheld · single-view support", "#ffae57"),
    ):
        if points.size:
            stride = max(1, int(points.shape[0] / 100_000))
            axis.scatter(
                points[::stride, 0],
                points[::stride, 2],
                s=1.0,
                c=color,
                alpha=0.45,
                linewidths=0,
            )
        axis.set_xlim(float(low[0]), float(high[0]))
        axis.set_ylim(float(low[1]), float(high[1]))
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title, color="white")
        axis.set_xlabel("world X (m)", color="#bcc7d1")
        axis.set_ylabel("world Z (m)", color="#bcc7d1")
        axis.tick_params(colors="#bcc7d1")
        axis.set_facecolor("#111820")
    figure.suptitle(
        "Consensus surfel support classes · shared bounds · single-view evidence withheld",
        color="white",
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def _load_refined_consensus_poses(
    report_path: Path,
    scan_dir: Path,
    original_poses: np.ndarray,
    common_intrinsics: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Admit a pose-only refinement of this exact consensus carrier."""
    from tools.mapanything_phone_scan.trajectory_refinement import (
        TRAJECTORY_REFINEMENT_SCHEMA,
        _finite_matrix,
        _sha256,
    )

    report_path = report_path.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    refinement = report.get("refinement") or {}
    source = report.get("source_identity") or {}
    holdout = refinement.get("withheld_evaluation") or {}
    if (
        report.get("schema") != TRAJECTORY_REFINEMENT_SCHEMA
        or source.get("provider") != "consensus_fusion"
        or source.get("coordinate_frame") != "consensus_phone_metric_world_unaligned_to_noesis"
        or refinement.get("raw_materialized") is not True
        or holdout.get("status") != "passed"
        or not int(holdout.get("verified_constraint_count") or 0)
        or not math.isclose(float(refinement.get("scale_change", 0.0)), 1.0, abs_tol=1e-8)
        or refinement.get("gauge_change") is not False
    ):
        raise ValueError("fusion requires an admitted pose-only consensus refinement with temporal holdouts")
    prepared_sha256 = _sha256(scan_dir / "prepared_frames_manifest.json")
    if report.get("provenance", {}).get("prepared_manifest_sha256") != prepared_sha256:
        raise ValueError("trajectory refinement belongs to a different prepared capture")
    source_manifest = Path(source["manifest"])
    if _sha256(source_manifest) != source.get("manifest_sha256"):
        raise ValueError("trajectory source manifest changed after refinement")
    raw_root = Path(report["source_raw"])
    rows = report.get("provenance", {}).get("raw_view_sha256") or []
    if len(rows) != len(original_poses):
        raise ValueError("trajectory refinement has incomplete raw-view provenance")
    for index, row in enumerate(rows):
        if row.get("index") != index or _sha256(raw_root / f"view_{index:04d}.npz") != row.get("sha256"):
            raise ValueError(f"trajectory source raw view {index} changed after refinement")
    solution_path = Path(refinement["materialized"]["camera_solution"])
    if _sha256(solution_path) != refinement["materialized"].get("camera_solution_sha256"):
        raise ValueError("refined camera solution changed after trajectory validation")
    with np.load(solution_path, allow_pickle=False) as solution:
        refined = np.asarray(solution["camera_to_world"], dtype=np.float64)
        source_poses = np.asarray(solution["source_camera_to_world"], dtype=np.float64)
        source_intrinsics = np.asarray(solution["intrinsics"], dtype=np.float64)
        frame = str(solution["coordinate_frame"].item())
    if (
        frame != source["coordinate_frame"]
        or refined.shape != original_poses.shape
        or source_poses.shape != original_poses.shape
        or source_intrinsics.shape != common_intrinsics.shape
        or not np.allclose(source_poses, original_poses, atol=2e-5, rtol=2e-5)
        or not np.allclose(source_intrinsics, common_intrinsics, atol=2e-5, rtol=2e-5)
    ):
        raise ValueError("refinement source poses or rays differ from the rebuilt consensus")
    for index, pose in enumerate(refined):
        _finite_matrix(pose, name=f"refined consensus camera {index}")
    if not np.allclose(refined[0], original_poses[0], atol=2e-5, rtol=2e-5):
        raise ValueError("refinement changed the consensus origin gauge")
    return refined, {
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "camera_solution_sha256": _sha256(solution_path),
        "source_manifest_sha256": source["manifest_sha256"],
        "prepared_manifest_sha256": prepared_sha256,
        "accepted_constraint_count": int(refinement.get("accepted_constraint_count") or 0),
        "withheld_evaluation": {key: value for key, value in holdout.items() if key != "residuals"},
        "scale_change": 1.0,
        "recomputed_after_pose_change": ["multiview_consistency", "depth_selection", "evidence_weights", "raw_world_points", "distinct_view_surfel_support"],
    }


def build(
    scan_dir: Path,
    output_dir: Path,
    voxel_m: float = 0.04,
    mapanything_raw: Path | None = None,
    da3_raw: Path | None = None,
    pose_carrier: str = "joint",
    evidence_relationship: str | None = None,
    trajectory_refinement_report: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if trajectory_refinement_report is not None and pose_carrier != "joint":
        raise ValueError("consensus trajectory refinement requires the joint pose carrier")
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
    calibration_provenance, calibrated_frames = _consensus_calibration(scan_dir, mapanything, da3)
    if evidence_relationship is None:
        evidence_relationship, evidence_provenance = _infer_evidence_relationship(
            mapanything_raw
        )
    elif evidence_relationship not in EVIDENCE_RELATIONSHIPS:
        raise ValueError(
            f"unsupported evidence relationship: {evidence_relationship!r}"
        )
    else:
        evidence_provenance = {
            "manifest": None,
            "conditioned_by_da3": evidence_relationship
            == "da3_conditioned_mapanything",
            "declared_by_cli": True,
        }

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
    trajectory_refinement = None
    if trajectory_refinement_report is not None:
        optimized_poses, trajectory_refinement = _load_refined_consensus_poses(
            trajectory_refinement_report, scan_dir, optimized_poses, common_k
        )

    print("[4/8] Computing per-model empirical confidence rank scores", flush=True)
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
        evidence_relationship=evidence_relationship,
    )
    fused_consistency = _multiview_consistency(
        fused["depth"], fused["mask"], common_k, output_poses
    )

    print("[7/8] Fusing accepted depths into weighted surfels", flush=True)
    surfel_points, surfel_colors, surfel_weights, surfel_metrics = _surfel_fusion(
        fused, common_k, output_poses, voxel_m
    )
    support_view_count = surfel_metrics.pop("_support_view_count")
    support_view_ids = surfel_metrics.pop("_support_view_ids")
    single_view_points = surfel_metrics.pop("_single_view_points")
    single_view_colors = surfel_metrics.pop("_single_view_colors")
    single_view_weights = surfel_metrics.pop("_single_view_weights")
    single_view_count = surfel_metrics.pop("_single_view_count")
    single_view_ids = surfel_metrics.pop("_single_view_ids")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "surfel_points.npz",
        points=surfel_points,
        colors=surfel_colors,
        weights=surfel_weights,
        support_view_count=support_view_count,
        support_view_ids=support_view_ids,
        support_view_id_limit=np.asarray(
            [SURFEL_SUPPORT_VIEW_ID_LIMIT], dtype=np.int32
        ),
        support_view_ids_truncated=(
            support_view_count > SURFEL_SUPPORT_VIEW_ID_LIMIT
        ),
        support_view_id_semantics=np.asarray("zero_based_raw_view_index"),
        support_class=np.asarray("multi_view_accepted"),
    )
    np.savez_compressed(
        output_dir / "surfel_points_single_view.npz",
        points=single_view_points,
        colors=single_view_colors,
        weights=single_view_weights,
        support_view_count=single_view_count,
        support_view_ids=single_view_ids,
        support_view_id_limit=np.asarray(
            [SURFEL_SUPPORT_VIEW_ID_LIMIT], dtype=np.int32
        ),
        support_view_ids_truncated=(
            single_view_count > SURFEL_SUPPORT_VIEW_ID_LIMIT
        ),
        support_view_id_semantics=np.asarray("zero_based_raw_view_index"),
        support_class=np.asarray("single_view_withheld"),
    )
    support_diagnostics_path = output_dir / "surfel_support_diagnostics.png"
    _write_support_diagnostics(
        support_diagnostics_path,
        surfel_points,
        single_view_points,
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
    _save_raw_frames(output_dir, fused, common_k, output_poses, calibrated_frames)

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
            "intrinsics_source": "per_frame_mean_of_provider_geometry_intrinsics_after_resolution_scaling",
        },
        "trajectory_alignment": sim3,
        "pose_graph": pose_graph_metrics,
        "pose_graph_output_used": trajectory_refinement is None,
        "trajectory_refinement": trajectory_refinement,
        "confidence_rank_scores": {
            "mapanything": map_confidence_metrics,
            "da3": da3_confidence_metrics,
        },
        "evidence_relationship": evidence_relationship,
        "evidence_provenance": evidence_provenance,
        "depth_scale_normalization": depth_scale_metrics,
        "multiview_consistency": {
            "mapanything": {
                "median_error_m": map_consistency.median_error_m,
                "p80_error_m": map_consistency.p80_error_m,
                "comparable_count": map_consistency.comparable_count,
                "valid_pixel_count": map_consistency.valid_pixel_count,
            },
            "da3": {
                "median_error_m": da3_consistency.median_error_m,
                "p80_error_m": da3_consistency.p80_error_m,
                "comparable_count": da3_consistency.comparable_count,
                "valid_pixel_count": da3_consistency.valid_pixel_count,
            },
            "consensus": {
                "median_error_m": fused_consistency.median_error_m,
                "p80_error_m": fused_consistency.p80_error_m,
                "comparable_count": fused_consistency.comparable_count,
                "valid_pixel_count": fused_consistency.valid_pixel_count,
            },
        },
        "fusion": fusion_metrics,
        "surfel_fusion": surfel_metrics,
        "heldout_even_to_odd_reprojection": heldout,
        "elapsed_s": float(time.perf_counter() - started),
        "artifacts": {
            "surfel_glb": "consensus_surfel_reconstruction.glb",
            "surfel_points": "surfel_points.npz",
            "single_view_surfels": "surfel_points_single_view.npz",
            "surfel_support_diagnostics": "surfel_support_diagnostics.png",
            "camera_solution": "camera_solution.npz",
            "raw_frames": "raw",
            "collaboration_diagnostics": "consensus_collaboration_diagnostics.png",
        },
    }
    metrics.update(calibration_provenance)
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
    output_contract.update(calibration_provenance)
    if calibrated_frames is not None:
        output_contract["frames"] = [{
            "index": index,
            "raw_npz": f"raw/view_{index:04d}.npz",
            "source_frame_id": frame.get("frame_id"),
            "source_frame_sha256": frame.get("sha256"),
            "timestamp_s": frame.get("timestamp_s"),
            "camera_intrinsics": frame["camera_intrinsics"],
            "calibration_processing": frame.get("calibration_processing"),
            "intrinsics": common_k[index].tolist(),
            "camera_pose": output_poses[index].tolist(),
        } for index, frame in enumerate(calibrated_frames)]
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
    parser.add_argument(
        "--trajectory-refinement-report", type=Path,
        help="Passed pose-only refinement of this consensus capture; rebuilds depth selection and surfels.",
    )
    parser.add_argument(
        "--evidence-relationship",
        choices=EVIDENCE_RELATIONSHIPS,
        help=(
            "Relationship between MapAnything and DA3 evidence. By default it "
            "is read from the small raw-output sidecar manifest."
        ),
    )
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
        evidence_relationship=args.evidence_relationship,
        trajectory_refinement_report=args.trajectory_refinement_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
