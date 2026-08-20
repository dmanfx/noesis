#!/usr/bin/env python3
"""Exhaustive learned cross-walk matcher for PCF multi-room registration.

This is the wide-baseline companion to ``register_pcf_rooms.py``.  It searches
all retained view pairs with official SuperPoint + LightGlue, then hands only
geometrically verified, PCF-depth-backed matches to the same gravity-preserving
registration gates.  The command fails closed and writes an insufficiency
report when the walks do not contain enough shared rigid observations.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

import pcf_multiroom_pose_graph as pose_graph
import register_pcf_rooms as core


@dataclass
class MatchDiagnostic:
    moving_view: int
    fixed_view: int
    match_count: int
    geometric_count: int
    depth_backed_count: int
    score_median: float | None
    score_p20: float | None
    baseline_3d_median_m: float | None
    moving_pixels: np.ndarray
    fixed_pixels: np.ndarray
    accepted: bool


@dataclass
class CorrectionRecord:
    moving_view: int
    fixed_view: int
    direction: str
    transform_moving_to_fixed: np.ndarray
    evidence: core.PnPCorrection


def _correction_parameters(record: CorrectionRecord) -> np.ndarray:
    return core._parameters_from_transform(record.transform_moving_to_fixed)


def _select_pnp_cluster(
    records: list[CorrectionRecord],
) -> list[CorrectionRecord]:
    best: list[CorrectionRecord] = []
    best_key = (0, 0, 0, -float("inf"))
    for seed in records:
        seed_parameters = _correction_parameters(seed)
        seed_yaw = math.degrees(float(seed_parameters[0]))
        members: list[CorrectionRecord] = []
        for row in records:
            parameters = _correction_parameters(row)
            yaw = math.degrees(float(parameters[0]))
            if (
                core._angle_distance_deg(yaw, seed_yaw) <= 1.75
                and float(np.linalg.norm(parameters[1:] - seed_parameters[1:]))
                <= 0.55
            ):
                members.append(row)
        moving_views = len({row.moving_view for row in members})
        fixed_views = len({row.fixed_view for row in members})
        pair_count = len({(row.moving_view, row.fixed_view) for row in members})
        support = sum(row.evidence.inlier_count for row in members)
        reprojection = float(
            np.median([row.evidence.reprojection_median_px for row in members])
        )
        key = (min(moving_views, fixed_views), pair_count, support, -reprojection)
        if key > best_key:
            best_key = key
            best = members
    return sorted(
        best,
        key=lambda row: (row.moving_view, row.fixed_view, row.direction),
    )


def _initial_pnp_transform(records: list[CorrectionRecord]) -> np.ndarray:
    parameters = np.stack([_correction_parameters(row) for row in records])
    weights = np.asarray(
        [
            row.evidence.inlier_count
            / max(0.75, row.evidence.reprojection_median_px)
            for row in records
        ],
        dtype=np.float64,
    )
    reference_yaw = float(parameters[np.argmax(weights), 0])
    unwrapped = reference_yaw + np.unwrap(parameters[:, 0] - reference_yaw)
    yaw = float(np.average(unwrapped, weights=weights))
    translation = np.asarray(
        [np.median(parameters[:, column]) for column in range(1, 4)],
        dtype=np.float64,
    )
    return core._yaw_transform([yaw, *translation])


def _record_reprojection_components(
    record: CorrectionRecord,
    correction: np.ndarray,
    moving: core.Room,
    fixed: core.Room,
) -> np.ndarray:
    evidence = record.evidence
    objects = evidence.object_points_source_world[evidence.inlier_mask]
    images = evidence.image_points_target_model[evidence.inlier_mask]
    if record.direction == "moving_depth_to_fixed_image":
        corrected = core._transform_points(objects, correction)
        camera_to_world = (
            fixed.world_from_local
            @ fixed.views[record.fixed_view].camera_pose_local
        )
        intrinsics = fixed.views[record.fixed_view].intrinsics
        camera_points = core._transform_points(corrected, np.linalg.inv(camera_to_world))
    elif record.direction == "fixed_depth_to_moving_image":
        moving_camera_to_baseline = (
            moving.world_from_local
            @ moving.views[record.moving_view].camera_pose_local
        )
        corrected_camera_to_world = correction @ moving_camera_to_baseline
        intrinsics = moving.views[record.moving_view].intrinsics
        camera_points = core._transform_points(
            objects,
            np.linalg.inv(corrected_camera_to_world),
        )
    else:
        raise ValueError(f"unknown PnP direction: {record.direction}")
    residual = np.full((len(objects), 2), 100.0, dtype=np.float64)
    visible = camera_points[:, 2] > 0.05
    if np.any(visible):
        projected = (intrinsics @ camera_points[visible].T).T
        projected = projected[:, :2] / projected[:, 2, None]
        residual[visible] = projected - images[visible]
    return residual


def _record_reprojection(
    record: CorrectionRecord,
    correction: np.ndarray,
    moving: core.Room,
    fixed: core.Room,
) -> np.ndarray:
    return np.linalg.norm(
        _record_reprojection_components(record, correction, moving, fixed),
        axis=1,
    )


def _optimize_reprojection(
    records: list[CorrectionRecord],
    initial: np.ndarray,
    moving: core.Room,
    fixed: core.Room,
) -> np.ndarray:
    initial_parameters = core._parameters_from_transform(initial)

    def residual(parameters: np.ndarray) -> np.ndarray:
        correction = core._yaw_transform(parameters)
        rows: list[np.ndarray] = []
        for record in records:
            components = _record_reprojection_components(
                record,
                correction,
                moving,
                fixed,
            )
            # Equalize complete image pairs so a single high-texture view cannot
            # dominate the session transform merely by contributing more pixels.
            rows.append(
                components.reshape(-1) / math.sqrt(max(1, len(components)))
            )
        return np.concatenate(rows)

    yaw_margin = math.radians(8.0)
    lower = initial_parameters - np.asarray([yaw_margin, 2.0, 1.0, 2.0])
    upper = initial_parameters + np.asarray([yaw_margin, 2.0, 1.0, 2.0])
    optimized = core.least_squares(
        residual,
        initial_parameters,
        bounds=(lower, upper),
        loss="huber",
        f_scale=1.5,
        max_nfev=500,
    )
    return core._yaw_transform(optimized.x)


def _aggregate_reprojection(
    records: list[CorrectionRecord],
    correction: np.ndarray,
    moving: core.Room,
    fixed: core.Room,
) -> tuple[float, float, dict[str, dict[str, float]]]:
    all_errors: list[np.ndarray] = []
    pair_rows: dict[str, list[np.ndarray]] = {}
    for record in records:
        errors = _record_reprojection(record, correction, moving, fixed)
        all_errors.append(errors)
        key = f"K{record.moving_view:02d}_F{record.fixed_view:02d}"
        pair_rows.setdefault(key, []).append(errors)
    combined = np.concatenate(all_errors) if all_errors else np.asarray([np.inf])
    per_pair = {
        key: {
            "count": int(sum(len(row) for row in rows)),
            "median_px": float(np.median(np.concatenate(rows))),
            "p80_px": float(np.percentile(np.concatenate(rows), 80.0)),
        }
        for key, rows in sorted(pair_rows.items())
    }
    return (
        float(np.median(combined)),
        float(np.percentile(combined, 80.0)),
        per_pair,
    )


def _finalize_pnp_solution(
    records: list[CorrectionRecord],
    moving: core.Room,
    fixed: core.Room,
) -> tuple[dict[str, Any], np.ndarray | None]:
    cluster = _select_pnp_cluster(records)
    moving_views = sorted({row.moving_view for row in cluster})
    fixed_views = sorted({row.fixed_view for row in cluster})
    pair_keys = sorted({(row.moving_view, row.fixed_view) for row in cluster})
    if len(moving_views) < 3 or len(fixed_views) < 3 or len(pair_keys) < 5:
        return (
            {
                "passed": False,
                "reason": "insufficient_bidirectional_pnp_consensus",
                "candidate_count": len(records),
                "consensus_candidate_count": len(cluster),
                "distinct_moving_views": moving_views,
                "distinct_fixed_views": fixed_views,
                "distinct_pair_count": len(pair_keys),
            },
            None,
        )
    initial = _initial_pnp_transform(cluster)
    final = _optimize_reprojection(cluster, initial, moving, fixed)
    median, p80, per_pair = _aggregate_reprojection(
        cluster,
        final,
        moving,
        fixed,
    )
    cross_validation: list[dict[str, Any]] = []
    heldout_errors: list[np.ndarray] = []
    leaveout_transforms: list[np.ndarray] = []
    for heldout_view in moving_views:
        train = [row for row in cluster if row.moving_view != heldout_view]
        heldout = [row for row in cluster if row.moving_view == heldout_view]
        if len({row.moving_view for row in train}) < 2 or not heldout:
            continue
        candidate = _optimize_reprojection(train, initial, moving, fixed)
        errors = np.concatenate(
            [_record_reprojection(row, candidate, moving, fixed) for row in heldout]
        )
        heldout_errors.append(errors)
        leaveout_transforms.append(candidate)
        parameters = core._parameters_from_transform(candidate)
        cross_validation.append(
            {
                "heldout": f"moving_view_{heldout_view:02d}",
                "constraint_count": len(heldout),
                "median_px": float(np.median(errors)),
                "p80_px": float(np.percentile(errors, 80.0)),
                "yaw_deg": math.degrees(float(parameters[0])),
                "translation_m": parameters[1:].tolist(),
            }
        )
    heldout = np.concatenate(heldout_errors) if heldout_errors else np.asarray([np.inf])
    final_parameters = core._parameters_from_transform(final)
    stability_translation: list[float] = []
    stability_yaw: list[float] = []
    for candidate in leaveout_transforms:
        parameters = core._parameters_from_transform(candidate)
        stability_translation.append(
            float(np.linalg.norm(parameters[1:] - final_parameters[1:]))
        )
        stability_yaw.append(
            core._angle_distance_deg(
                math.degrees(float(parameters[0])),
                math.degrees(float(final_parameters[0])),
            )
        )
    heldout_median = float(np.median(heldout))
    heldout_p80 = float(np.percentile(heldout, 80.0))
    translation_p80 = (
        float(np.percentile(stability_translation, 80.0))
        if stability_translation
        else float("inf")
    )
    yaw_p80 = (
        float(np.percentile(stability_yaw, 80.0))
        if stability_yaw
        else float("inf")
    )
    passed = bool(
        median <= 3.0
        and p80 <= 6.0
        and heldout_median <= 4.0
        and heldout_p80 <= 8.0
        and translation_p80 <= 0.40
        and yaw_p80 <= 1.5
    )
    candidate_rows = []
    for row in cluster:
        parameters = _correction_parameters(row)
        candidate_rows.append(
            {
                "moving_view": row.moving_view,
                "fixed_view": row.fixed_view,
                "direction": row.direction,
                "inlier_count": row.evidence.inlier_count,
                "reprojection_median_px": row.evidence.reprojection_median_px,
                "reprojection_p80_px": row.evidence.reprojection_p80_px,
                "yaw_deg": math.degrees(float(parameters[0])),
                "translation_m": parameters[1:].tolist(),
            }
        )
    return (
        {
            "passed": passed,
            "reason": None if passed else "joint_reprojection_or_leaveout_gate_failed",
            "moving_correction_in_backend_world_row_major": final.tolist(),
            "yaw_correction_deg": math.degrees(float(final_parameters[0])),
            "translation_correction_m": final_parameters[1:].tolist(),
            "consensus_candidate_count": len(cluster),
            "distinct_moving_views": moving_views,
            "distinct_fixed_views": fixed_views,
            "distinct_pair_count": len(pair_keys),
            "joint_reprojection_median_px": median,
            "joint_reprojection_p80_px": p80,
            "heldout_moving_view_median_px": heldout_median,
            "heldout_moving_view_p80_px": heldout_p80,
            "leaveout_translation_p80_m": translation_p80,
            "leaveout_yaw_p80_deg": yaw_p80,
            "per_pair_reprojection": per_pair,
            "leave_one_moving_view_out": cross_validation,
            "consensus_candidates": candidate_rows,
        },
        final if passed else None,
    )


def _image_tensor(view: core.RawView, device: torch.device) -> torch.Tensor:
    image = torch.from_numpy(np.ascontiguousarray(view.feature_rgb))
    return image.permute(2, 0, 1).float().div(255.0).to(device)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _depth_backed_baseline(
    moving: core.Room,
    fixed: core.Room,
    moving_index: int,
    fixed_index: int,
    moving_pixels: np.ndarray,
    fixed_pixels: np.ndarray,
) -> tuple[int, float | None]:
    distances: list[float] = []
    for moving_pixel, fixed_pixel in zip(moving_pixels, fixed_pixels):
        moving_point = core._point_at_feature(
            moving.views[moving_index],
            (float(moving_pixel[0]), float(moving_pixel[1])),
        )
        fixed_point = core._point_at_feature(
            fixed.views[fixed_index],
            (float(fixed_pixel[0]), float(fixed_pixel[1])),
        )
        if moving_point is None or fixed_point is None:
            continue
        moving_world = core._transform_points(
            moving_point[None, :], moving.world_from_local
        )[0]
        fixed_world = core._transform_points(
            fixed_point[None, :], fixed.world_from_local
        )[0]
        distances.append(float(np.linalg.norm(moving_world - fixed_world)))
    return len(distances), float(np.median(distances)) if distances else None


def _write_candidate_sheet(
    moving: core.Room,
    fixed: core.Room,
    diagnostics: list[MatchDiagnostic],
    path: Path,
) -> None:
    selected = sorted(
        diagnostics,
        key=lambda row: (
            -row.geometric_count,
            -(row.depth_backed_count),
            -row.match_count,
        ),
    )[:10]
    if not selected:
        return
    moving_width = moving.views[0].feature_rgb.shape[1]
    fixed_width = fixed.views[0].feature_rgb.shape[1]
    tile_width = moving_width + fixed_width
    tile_height = max(
        moving.views[0].feature_rgb.shape[0],
        fixed.views[0].feature_rgb.shape[0],
    )
    canvas = np.full((tile_height * len(selected), tile_width, 3), 18, dtype=np.uint8)
    for row_index, row in enumerate(selected):
        moving_image = moving.views[row.moving_view].feature_rgb
        fixed_image = fixed.views[row.fixed_view].feature_rgb
        top = row_index * tile_height
        canvas[top : top + moving_image.shape[0], :moving_width] = moving_image
        canvas[top : top + fixed_image.shape[0], moving_width:] = fixed_image
        for match_index, (moving_pixel, fixed_pixel) in enumerate(
            zip(row.moving_pixels[:100], row.fixed_pixels[:100])
        ):
            color = (96, 220, 255) if match_index % 2 else (255, 180, 96)
            left = (
                int(round(float(moving_pixel[0]))),
                top + int(round(float(moving_pixel[1]))),
            )
            right = (
                moving_width + int(round(float(fixed_pixel[0]))),
                top + int(round(float(fixed_pixel[1]))),
            )
            cv2.line(canvas, left, right, color, 1, cv2.LINE_AA)
        baseline = (
            f"{row.baseline_3d_median_m:.2f} m baseline median"
            if row.baseline_3d_median_m is not None
            else "no paired PCF depth"
        )
        cv2.putText(
            canvas,
            f"K{row.moving_view:02d} / F{row.fixed_view:02d}: "
            f"matches={row.match_count}, geometry={row.geometric_count}, "
            f"depth={row.depth_backed_count}, {baseline}",
            (8, top + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _input_report(room: core.Room) -> dict[str, Any]:
    return {
        "name": room.name,
        "prior_id": room.prior_id,
        "scan_id": room.scan_id,
        "view_count": len(room.views),
        "pcf_raw_root": str(room.raw_root),
        "world_manifest": str(room.world_manifest),
        "world_manifest_sha256": core._sha256(room.world_manifest),
        "prepared_manifest": str(room.scan_dir / "prepared_frames_manifest.json"),
        "prepared_manifest_sha256": core._sha256(
            room.scan_dir / "prepared_frames_manifest.json"
        ),
    }


def _diagnostic_report(row: MatchDiagnostic) -> dict[str, Any]:
    return {
        "moving_view": row.moving_view,
        "fixed_view": row.fixed_view,
        "match_count": row.match_count,
        "geometric_inlier_count": row.geometric_count,
        "pcf_depth_backed_count": row.depth_backed_count,
        "match_score_median": row.score_median,
        "match_score_p20": row.score_p20,
        "accepted_rigid_hypothesis": row.accepted,
        "baseline_3d_distance_median_m": row.baseline_3d_median_m,
    }


def _finalize_solution(
    moving: core.Room,
    fixed: core.Room,
    pairs: list[core.PairEvidence],
    output_dir: Path,
    settings: core.Settings,
) -> tuple[dict[str, Any], np.ndarray | None]:
    cluster = core._select_consensus_cluster(pairs, settings)
    moving_view_count = len({row.moving_view for row in cluster})
    fixed_view_count = len({row.fixed_view for row in cluster})
    if (
        moving_view_count < settings.minimum_distinct_moving_views
        or fixed_view_count < settings.minimum_distinct_fixed_views
    ):
        return (
            {
                "passed": False,
                "reason": "insufficient_independent_temporal_support",
                "consensus_pair_count": len(cluster),
                "distinct_moving_views": moving_view_count,
                "distinct_fixed_views": fixed_view_count,
                "required_distinct_views_per_room": settings.minimum_distinct_moving_views,
                "consensus_pairs": [core._pair_report(row) for row in cluster],
            },
            None,
        )
    moving_rows: list[np.ndarray] = []
    fixed_rows: list[np.ndarray] = []
    pair_ids: list[np.ndarray] = []
    for pair_index, row in enumerate(cluster):
        moving_rows.append(row.moving_points_world[row.inlier_mask])
        fixed_rows.append(row.fixed_points_world[row.inlier_mask])
        pair_ids.append(
            np.full(int(np.count_nonzero(row.inlier_mask)), pair_index, dtype=np.int32)
        )
    moving_all, fixed_all, ids_all = core._deduplicate_correspondences(
        np.concatenate(moving_rows),
        np.concatenate(fixed_rows),
        np.concatenate(pair_ids),
    )
    pair_identity = sorted(set(int(value) for value in ids_all))
    heldout_ids = set(pair_identity[::4]) if len(pair_identity) >= 5 else set()
    train_mask = ~np.isin(ids_all, list(heldout_ids))
    if int(np.count_nonzero(train_mask)) < settings.minimum_global_3d_inliers:
        return (
            {
                "passed": False,
                "reason": "insufficient_deduplicated_3d_support",
                "deduplicated_correspondence_count": len(moving_all),
                "required": settings.minimum_global_3d_inliers,
                "consensus_pairs": [core._pair_report(row) for row in cluster],
            },
            None,
        )
    correction, inlier_mask = core._ransac_yaw_translation(
        moving_all[train_mask],
        fixed_all[train_mask],
        threshold_m=settings.global_ransac_threshold_m,
        random_seed=settings.random_seed,
        iterations=3_000,
    )
    training_source = moving_all[train_mask]
    training_target = fixed_all[train_mask]
    training_residual = np.linalg.norm(
        core._transform_points(training_source[inlier_mask], correction)
        - training_target[inlier_mask],
        axis=1,
    )
    heldout_mask = ~train_mask
    heldout_residual = np.linalg.norm(
        core._transform_points(moving_all[heldout_mask], correction)
        - fixed_all[heldout_mask],
        axis=1,
    )
    parameters = core._parameters_from_transform(correction)
    passed = bool(
        len(training_residual) >= settings.minimum_global_3d_inliers
        and float(np.median(training_residual)) <= settings.maximum_global_median_m
        and float(np.percentile(training_residual, 80.0))
        <= settings.maximum_global_p80_m
        and (
            not len(heldout_residual)
            or float(np.median(heldout_residual)) <= settings.maximum_heldout_median_m
        )
    )
    np.savez_compressed(
        output_dir / "cross_room_correspondences.npz",
        moving_points_backend_world=moving_all.astype(np.float32),
        fixed_points_backend_world=fixed_all.astype(np.float32),
        pair_ids=ids_all,
        train_mask=train_mask,
    )
    solution = {
        "passed": passed,
        "reason": None if passed else "global_or_heldout_geometric_gate_failed",
        "moving_correction_in_backend_world_row_major": correction.tolist(),
        "yaw_correction_deg": math.degrees(float(parameters[0])),
        "translation_correction_m": parameters[1:].tolist(),
        "training_correspondence_count": int(np.count_nonzero(train_mask)),
        "training_inlier_count": len(training_residual),
        "training_inlier_fraction": float(
            len(training_residual) / max(1, int(np.count_nonzero(train_mask)))
        ),
        "training_residual_median_m": float(np.median(training_residual)),
        "training_residual_p80_m": float(np.percentile(training_residual, 80.0)),
        "heldout_correspondence_count": len(heldout_residual),
        "heldout_residual_median_m": (
            float(np.median(heldout_residual)) if len(heldout_residual) else None
        ),
        "heldout_residual_p80_m": (
            float(np.percentile(heldout_residual, 80.0))
            if len(heldout_residual)
            else None
        ),
        "consensus_pairs": [core._pair_report(row) for row in cluster],
    }
    return solution, correction if passed else None


def run(
    moving: core.Room,
    fixed: core.Room,
    output_dir: Path,
    settings: core.Settings,
    *,
    vertical_translation_m: float = 0.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SuperPoint(max_num_keypoints=1_024).eval().to(device)
    matcher = LightGlue(
        features="superpoint",
        depth_confidence=0.90,
        width_confidence=0.95,
    ).eval().to(device)
    print(f"[1/4] Extracting SuperPoint features on {device}", flush=True)
    moving_features = [
        extractor.extract(_image_tensor(view, device), resize=960)
        for view in moving.views
    ]
    fixed_features = [
        extractor.extract(_image_tensor(view, device), resize=960)
        for view in fixed.views
    ]
    diagnostics: list[MatchDiagnostic] = []
    accepted_pairs: list[core.PairEvidence] = []
    pnp_records: list[CorrectionRecord] = []
    total_pairs = len(moving.views) * len(fixed.views)
    print(f"[2/4] Exhaustively matching all {total_pairs} cross-walk view pairs", flush=True)
    pair_number = 0
    with torch.inference_mode():
        for moving_index, moving_feature in enumerate(moving_features):
            for fixed_index, fixed_feature in enumerate(fixed_features):
                pair_number += 1
                result = matcher(
                    {
                        "image0": _to_device(moving_feature, device),
                        "image1": _to_device(fixed_feature, device),
                    }
                )
                moving_unbatched, fixed_unbatched, result_unbatched = map(
                    rbd,
                    (moving_feature, fixed_feature, result),
                )
                matches = result_unbatched["matches"].detach().cpu().numpy()
                scores = result_unbatched["scores"].detach().cpu().numpy()
                if len(matches) < 8:
                    if pair_number % 192 == 0:
                        print(
                            f"      matched {pair_number}/{total_pairs}; "
                            f"depth-rigid candidates={len(accepted_pairs)}",
                            flush=True,
                        )
                    continue
                moving_keypoints = (
                    moving_unbatched["keypoints"].detach().cpu().numpy()
                )
                fixed_keypoints = fixed_unbatched["keypoints"].detach().cpu().numpy()
                moving_pixels = moving_keypoints[matches[:, 0]].astype(np.float32)
                fixed_pixels = fixed_keypoints[matches[:, 1]].astype(np.float32)
                geometric_mask = core._geometric_inlier_mask(
                    moving_pixels,
                    fixed_pixels,
                )
                geometric_count = int(np.count_nonzero(geometric_mask))
                if geometric_count < 6:
                    if pair_number % 192 == 0:
                        print(
                            f"      matched {pair_number}/{total_pairs}; "
                            f"depth-rigid candidates={len(accepted_pairs)}",
                            flush=True,
                        )
                    continue
                geometric_moving = moving_pixels[geometric_mask]
                geometric_fixed = fixed_pixels[geometric_mask]
                depth_count, baseline_median = _depth_backed_baseline(
                    moving,
                    fixed,
                    moving_index,
                    fixed_index,
                    geometric_moving,
                    geometric_fixed,
                )
                score_median = float(np.median(scores)) if len(scores) else 0.0
                score_p20 = (
                    float(np.percentile(scores, 20.0)) if len(scores) else 0.0
                )
                evidence = core._pair_evidence_from_pixels(
                    moving,
                    fixed,
                    moving_index,
                    fixed_index,
                    float(geometric_count),
                    geometric_moving,
                    geometric_fixed,
                    mutual_match_count=len(matches),
                    geometric_match_count=geometric_count,
                    settings=settings,
                )
                if evidence is not None:
                    accepted_pairs.append(evidence)
                strong_for_pnp = bool(
                    len(matches) >= 30
                    and geometric_count >= 12
                    and score_median >= 0.12
                    and score_p20 >= 0.03
                )
                forward_pnp = (
                    core._pnp_correction(
                        moving,
                        fixed,
                        moving_index,
                        fixed_index,
                        geometric_moving,
                        geometric_fixed,
                        minimum_inliers=8,
                    )
                    if strong_for_pnp
                    else None
                )
                if forward_pnp is not None:
                    pnp_records.append(
                        CorrectionRecord(
                            moving_view=moving_index,
                            fixed_view=fixed_index,
                            direction="moving_depth_to_fixed_image",
                            transform_moving_to_fixed=forward_pnp.transform,
                            evidence=forward_pnp,
                        )
                    )
                reverse_pnp = (
                    core._pnp_correction(
                        fixed,
                        moving,
                        fixed_index,
                        moving_index,
                        geometric_fixed,
                        geometric_moving,
                        minimum_inliers=8,
                    )
                    if strong_for_pnp
                    else None
                )
                if reverse_pnp is not None:
                    pnp_records.append(
                        CorrectionRecord(
                            moving_view=moving_index,
                            fixed_view=fixed_index,
                            direction="fixed_depth_to_moving_image",
                            transform_moving_to_fixed=np.linalg.inv(
                                reverse_pnp.transform
                            ),
                            evidence=reverse_pnp,
                        )
                    )
                diagnostics.append(
                    MatchDiagnostic(
                        moving_view=moving_index,
                        fixed_view=fixed_index,
                        match_count=len(matches),
                        geometric_count=geometric_count,
                        depth_backed_count=depth_count,
                        score_median=score_median,
                        score_p20=score_p20,
                        baseline_3d_median_m=baseline_median,
                        moving_pixels=geometric_moving,
                        fixed_pixels=geometric_fixed,
                        accepted=evidence is not None,
                    )
                )
                if pair_number % 192 == 0:
                    print(
                        f"      matched {pair_number}/{total_pairs}; "
                        f"depth-rigid candidates={len(accepted_pairs)}",
                        flush=True,
                    )
    print("[3/4] Solving multi-view transform consensus", flush=True)
    depth_depth_solution, _ = _finalize_solution(
        moving,
        fixed,
        accepted_pairs,
        output_dir,
        settings,
    )
    reprojection_diagnostic, _ = _finalize_pnp_solution(
        pnp_records,
        moving,
        fixed,
    )
    pose_observations = [
        pose_graph.PnPTransformObservation(
            moving_view=row.moving_view,
            fixed_view=row.fixed_view,
            direction=row.direction,
            transform_moving_to_fixed=row.transform_moving_to_fixed,
            inlier_count=row.evidence.inlier_count,
            reprojection_median_px=row.evidence.reprojection_median_px,
            reprojection_p80_px=row.evidence.reprojection_p80_px,
        )
        for row in pnp_records
    ]
    moving_camera_poses = {
        view.index: moving.world_from_local @ view.camera_pose_local
        for view in moving.views
    }
    fixed_camera_poses = {
        view.index: fixed.world_from_local @ view.camera_pose_local
        for view in fixed.views
    }
    pose_result = pose_graph.solve_cross_session_pose_graph(
        pose_observations,
        moving_camera_poses,
        fixed_camera_poses,
        vertical_translation_m=vertical_translation_m,
    )
    correction = pose_result.global_transform_moving_to_fixed
    solution: dict[str, Any] = {
        "passed": pose_result.accepted,
        "reason_codes": list(pose_result.reason_codes),
        "moving_correction_in_backend_world_row_major": (
            correction.tolist() if correction is not None else None
        ),
        "per_view_world_corrections_row_major": (
            {
                str(index): transform.tolist()
                for index, transform in (
                    (
                        index,
                        transform @ np.linalg.inv(correction),
                    )
                    for index, transform in pose_result.per_view_transforms_moving_to_fixed.items()
                )
            }
            if pose_result.accepted
            else {}
        ),
    }
    _write_candidate_sheet(
        moving,
        fixed,
        diagnostics,
        output_dir / "learned_match_candidates.png",
    )
    if correction is not None:
        core._write_match_sheet(
            moving,
            fixed,
            core._select_consensus_cluster(accepted_pairs, settings),
            output_dir / "verified_rgbd_matches.png",
        )
        core._write_topdown(
            moving,
            fixed,
            correction,
            output_dir / "before_after_topdown.png",
        )
        core._write_glb(
            moving,
            fixed,
            correction,
            output_dir / "registered_room_surfels.glb",
        )
    top_diagnostics = sorted(
        diagnostics,
        key=lambda row: (-row.geometric_count, -row.depth_backed_count),
    )[:40]
    superpoint_weight = Path(torch.hub.get_dir()) / "checkpoints" / "superpoint_v1.pth"
    lightglue_weight = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / "superpoint_lightglue_v0-1_arxiv.pth"
    )
    passed = bool(solution.get("passed"))
    report: dict[str, Any] = {
        "schema": "noesis.pcf.multiroom_cross_session_registration.v1",
        "generated_at": core._utc_now(),
        "status": "passed" if passed else "insufficient_cross_session_overlap",
        "method": "exhaustive_superpoint_lightglue_geometry_pcf_depth_gravity_constrained_ransac",
        "whole_cloud_icp_used": False,
        "fixed_room": _input_report(fixed),
        "moving_room": _input_report(moving),
        "matcher": {
            "package": "lightglue",
            "package_version": importlib.metadata.version("lightglue"),
            "feature_model": "SuperPoint",
            "maximum_keypoints": 1_024,
            "device": str(device),
            "view_pair_count": total_pairs,
            "pairs_with_geometric_support": len(diagnostics),
            "pairs_with_depth_rigid_hypothesis": len(accepted_pairs),
            "bidirectional_pnp_candidate_count": len(pnp_records),
            "superpoint_weight_sha256": (
                core._sha256(superpoint_weight) if superpoint_weight.exists() else None
            ),
            "lightglue_weight_sha256": (
                core._sha256(lightglue_weight) if lightglue_weight.exists() else None
            ),
        },
        "top_pair_diagnostics": [
            _diagnostic_report(row) for row in top_diagnostics
        ],
        "solution": solution,
        "pose_graph_validation": pose_result.report,
        "joint_reprojection_diagnostic": reprojection_diagnostic,
        "depth_to_depth_diagnostic": depth_depth_solution,
        "recommendation": (
            None
            if passed
            else {
                "action": "capture_short_connector_walk",
                "reason": (
                    "the retained independent walks do not provide three temporally "
                    "independent, spatially distributed RGB-D loop closures"
                ),
                "do_not_use": [
                    "whole_room_icp",
                    "nearest_neighbor_surface_pairs",
                    "manual_visual_nudging_without_correspondence_evidence",
                ],
            }
        ),
        "artifacts": {
            "learned_match_candidates": "learned_match_candidates.png",
            "verified_rgbd_matches": (
                "verified_rgbd_matches.png" if correction is not None else None
            ),
            "before_after_topdown": (
                "before_after_topdown.png" if correction is not None else None
            ),
            "registered_room_surfels": (
                "registered_room_surfels.glb" if correction is not None else None
            ),
        },
    }
    (output_dir / "registration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("[4/4] Wrote the cross-session evidence report", flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moving-name", required=True)
    parser.add_argument("--moving-prior-id", required=True)
    parser.add_argument("--moving-scan-id", required=True)
    parser.add_argument("--moving-scan-dir", type=Path, required=True)
    parser.add_argument("--moving-pcf-root", type=Path, required=True)
    parser.add_argument("--moving-world-manifest", type=Path, required=True)
    parser.add_argument("--fixed-name", required=True)
    parser.add_argument("--fixed-prior-id", required=True)
    parser.add_argument("--fixed-scan-id", required=True)
    parser.add_argument("--fixed-scan-dir", type=Path, required=True)
    parser.add_argument("--fixed-pcf-root", type=Path, required=True)
    parser.add_argument("--fixed-world-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--vertical-translation-m",
        type=float,
        default=0.0,
        help=(
            "fixed-floor minus moving-floor world Y; keep at zero when both "
            "accepted room worlds already share the measured floor plane"
        ),
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    settings = core.Settings()
    moving = core._load_room(
        name=arguments.moving_name,
        prior_id=arguments.moving_prior_id,
        scan_id=arguments.moving_scan_id,
        scan_dir=arguments.moving_scan_dir,
        pcf_root=arguments.moving_pcf_root,
        world_manifest=arguments.moving_world_manifest,
    )
    fixed = core._load_room(
        name=arguments.fixed_name,
        prior_id=arguments.fixed_prior_id,
        scan_id=arguments.fixed_scan_id,
        scan_dir=arguments.fixed_scan_dir,
        pcf_root=arguments.fixed_pcf_root,
        world_manifest=arguments.fixed_world_manifest,
    )
    report = run(
        moving,
        fixed,
        arguments.output_dir,
        settings,
        vertical_translation_m=arguments.vertical_translation_m,
    )
    print(json.dumps(report["solution"], indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
