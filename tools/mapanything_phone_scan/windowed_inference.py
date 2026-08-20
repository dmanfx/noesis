from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .inference import (
    MapAnythingScanError,
    MapAnythingScanSettings,
    _write_reconstruction_glb,
    _write_trajectory_preview,
    run_mapanything_scan,
)
from .supplement import (
    SupplementIntegrationError,
    SupplementIntegrationSettings,
    _estimate_bridge_similarity,
    _load_raw,
    _transform_points,
    _transform_pose,
)


ProgressCallback = Callable[[float, str], None]
WindowRunner = Callable[
    [Path, Path, dict[str, Any], MapAnythingScanSettings, ProgressCallback],
    dict[str, Any],
]


def _window_ranges(total: int, maximum: int, overlap: int) -> list[tuple[int, int]]:
    if total < 2:
        raise MapAnythingScanError("MapAnything needs at least two prepared views")
    if maximum < 2:
        raise MapAnythingScanError("MapAnything joint-view capacity must be at least two")
    if overlap < 2 or overlap >= maximum:
        raise MapAnythingScanError(
            "MapAnything window overlap must be at least two and below the joint-view capacity"
        )
    if total <= maximum:
        return [(0, total)]
    stride = maximum - overlap
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(total, start + maximum)
        ranges.append((start, end))
        if end >= total:
            break
        start += stride
    return ranges


def _window_overlap_residual(
    base_records: list[dict[str, Any]],
    append_raw: list[dict[str, np.ndarray]],
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> dict[str, Any]:
    residual_rows: list[np.ndarray] = []
    common_fractions: list[float] = []
    for record, new_raw in zip(base_records, append_raw, strict=True):
        base_raw = _load_raw(record["raw_path"])
        base_points = _transform_points(
            base_raw["world_points"],
            float(record["scale"]),
            np.asarray(record["rotation"], dtype=np.float64),
            np.asarray(record["translation"], dtype=np.float64),
        )
        new_points = _transform_points(
            new_raw["world_points"], scale, rotation, translation
        )
        base_mask = np.asarray(base_raw["mask"], dtype=bool)
        new_mask = np.asarray(new_raw["mask"], dtype=bool)
        if base_mask.shape != new_mask.shape or base_points.shape != new_points.shape:
            raise MapAnythingScanError(
                "overlapping MapAnything windows returned incompatible geometry shapes"
            )
        common = (
            base_mask
            & new_mask
            & np.isfinite(base_points).all(axis=2)
            & np.isfinite(new_points).all(axis=2)
        )
        common_fractions.append(float(np.count_nonzero(common) / common.size))
        values = np.linalg.norm(new_points[common] - base_points[common], axis=1)
        if values.size:
            stride = max(1, values.size // 20_000)
            residual_rows.append(values[::stride])
    if not residual_rows:
        raise MapAnythingScanError(
            "overlapping MapAnything windows had no common valid geometry"
        )
    residuals = np.concatenate(residual_rows)
    result = {
        "sample_count": int(residuals.size),
        "common_valid_fraction_p50": float(np.median(common_fractions)),
        "residual_m_p50": float(np.percentile(residuals, 50.0)),
        "residual_m_p80": float(np.percentile(residuals, 80.0)),
        "residual_m_p95": float(np.percentile(residuals, 95.0)),
    }
    result["checks"] = {
        "common_geometry": result["common_valid_fraction_p50"] >= 0.45,
        "median_residual": result["residual_m_p50"] <= 0.20,
        "p80_residual": result["residual_m_p80"] <= 0.40,
    }
    if not all(result["checks"].values()):
        failed = ", ".join(
            name for name, passed in result["checks"].items() if not passed
        )
        raise MapAnythingScanError(
            "overlapping MapAnything windows did not clear the geometry gate: "
            f"{failed}; median={result['residual_m_p50']:.3f}m, "
            f"p80={result['residual_m_p80']:.3f}m"
        )
    return result


def _refine_window_scale_translation(
    base_records: list[dict[str, Any]],
    append_raw: list[dict[str, np.ndarray]],
    base_poses: np.ndarray,
    append_poses: np.ndarray,
    initial_scale: float,
    rotation: np.ndarray,
    initial_translation: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    source_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    for record, new_raw in zip(base_records, append_raw, strict=True):
        base_raw = _load_raw(record["raw_path"])
        target = _transform_points(
            base_raw["world_points"],
            float(record["scale"]),
            np.asarray(record["rotation"], dtype=np.float64),
            np.asarray(record["translation"], dtype=np.float64),
        )
        source = np.asarray(new_raw["world_points"], dtype=np.float64)
        base_confidence = np.asarray(base_raw["confidence"], dtype=np.float64)
        new_confidence = np.asarray(new_raw["confidence"], dtype=np.float64)
        common = (
            np.asarray(base_raw["mask"], dtype=bool)
            & np.asarray(new_raw["mask"], dtype=bool)
            & np.isfinite(target).all(axis=2)
            & np.isfinite(source).all(axis=2)
            & np.isfinite(base_confidence)
            & np.isfinite(new_confidence)
        )
        indices = np.flatnonzero(common.reshape(-1))
        if indices.size == 0:
            continue
        positions = np.linspace(
            0, indices.size - 1, min(6000, indices.size), dtype=np.int64
        )
        chosen = indices[positions]
        source_rows.append(source.reshape(-1, 3)[chosen])
        target_rows.append(target.reshape(-1, 3)[chosen])
        confidence = np.minimum(
            base_confidence.reshape(-1)[chosen],
            new_confidence.reshape(-1)[chosen],
        )
        low, high = np.percentile(confidence, (10.0, 90.0))
        weight_rows.append(
            0.25
            + 0.75
            * np.clip(
                (confidence - low) / max(float(high - low), 1e-8), 0.0, 1.0
            )
        )
    if not source_rows:
        raise MapAnythingScanError(
            "overlapping MapAnything windows had no geometry for scale refinement"
        )
    source = np.concatenate(source_rows)
    target = np.concatenate(target_rows)
    weights = np.concatenate(weight_rows)
    rotated = (np.asarray(rotation, dtype=np.float64) @ source.T).T
    scale = float(initial_scale)
    translation = np.asarray(initial_translation, dtype=np.float64)
    initial_residual = np.linalg.norm(
        scale * rotated + translation - target, axis=1
    )
    keep = initial_residual <= min(float(np.percentile(initial_residual, 80.0)), 0.80)
    for _ in range(4):
        if int(np.count_nonzero(keep)) < 1000:
            raise MapAnythingScanError(
                "overlapping MapAnything windows had too few robust 3D correspondences"
            )
        selected_weights = weights[keep]
        selected_weights = selected_weights / np.maximum(
            np.sum(selected_weights), 1e-12
        )
        source_selected = rotated[keep]
        target_selected = target[keep]
        source_center = np.sum(
            selected_weights[:, None] * source_selected, axis=0
        )
        target_center = np.sum(
            selected_weights[:, None] * target_selected, axis=0
        )
        source_centered = source_selected - source_center
        target_centered = target_selected - target_center
        numerator = float(
            np.sum(
                selected_weights
                * np.sum(source_centered * target_centered, axis=1)
            )
        )
        denominator = float(
            np.sum(selected_weights * np.sum(source_centered**2, axis=1))
        )
        candidate_scale = numerator / max(denominator, 1e-12)
        candidate_translation = target_center - candidate_scale * source_center
        residual = np.linalg.norm(
            candidate_scale * rotated + candidate_translation - target, axis=1
        )
        threshold = min(float(np.percentile(residual, 72.0)), 0.60)
        next_keep = residual <= threshold
        scale = float(candidate_scale)
        translation = candidate_translation
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep

    final_residual = np.linalg.norm(scale * rotated + translation - target, axis=1)
    transformed_centers = (
        scale
        * (rotation @ np.asarray(append_poses, dtype=np.float64)[:, :3, 3].T).T
        + translation
    )
    position_errors = np.linalg.norm(
        transformed_centers - np.asarray(base_poses, dtype=np.float64)[:, :3, 3],
        axis=1,
    )
    metrics = {
        "correspondence_count": int(len(source)),
        "robust_correspondence_count": int(np.count_nonzero(keep)),
        "initial_residual_m_p50": float(np.percentile(initial_residual, 50.0)),
        "initial_residual_m_p80": float(np.percentile(initial_residual, 80.0)),
        "final_residual_m_p50": float(np.percentile(final_residual, 50.0)),
        "final_residual_m_p80": float(np.percentile(final_residual, 80.0)),
        "scale_ratio_to_pose_solution": float(scale / initial_scale),
        "translation_refinement_m": float(
            np.linalg.norm(translation - initial_translation)
        ),
        "camera_position_error_m_p80": float(
            np.percentile(position_errors, 80.0)
        ),
    }
    metrics["checks"] = {
        "scale_refinement": 0.65 <= metrics["scale_ratio_to_pose_solution"] <= 1.35,
        "translation_refinement": metrics["translation_refinement_m"] <= 0.75,
        "camera_position": metrics["camera_position_error_m_p80"] <= 0.55,
        "surface_improvement": (
            metrics["final_residual_m_p50"]
            <= metrics["initial_residual_m_p50"] * 0.90
            or metrics["final_residual_m_p50"] <= 0.12
        ),
    }
    if not all(metrics["checks"].values()):
        failed = ", ".join(
            name for name, passed in metrics["checks"].items() if not passed
        )
        raise MapAnythingScanError(
            "duplicate-window 3D refinement did not clear its bounded gate: "
            f"{failed}; scale_ratio={metrics['scale_ratio_to_pose_solution']:.3f}, "
            f"translation={metrics['translation_refinement_m']:.3f}m, "
            f"camera_p80={metrics['camera_position_error_m_p80']:.3f}m"
        )
    return scale, translation, metrics


def _as_u8(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.dtype == np.uint8:
        return value
    if value.size and float(np.nanmax(value)) <= 1.5:
        value = value * 255.0
    return np.clip(value, 0, 255).astype(np.uint8)


def _depth_stats(depth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    values = depth[valid]
    if values.size == 0:
        raise MapAnythingScanError("a registered window view has no valid depth")
    return {
        "min_m": float(np.min(values)),
        "p02_m": float(np.percentile(values, 2.0)),
        "p50_m": float(np.percentile(values, 50.0)),
        "p98_m": float(np.percentile(values, 98.0)),
        "max_m": float(np.max(values)),
        "valid_fraction": float(np.count_nonzero(valid) / valid.size),
    }


def _confidence_stats(confidence: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    values = confidence[mask & np.isfinite(confidence)]
    if values.size == 0:
        raise MapAnythingScanError("a registered window view has no valid confidence")
    return {
        "min": float(np.min(values)),
        "p02": float(np.percentile(values, 2.0)),
        "p50": float(np.percentile(values, 50.0)),
        "p98": float(np.percentile(values, 98.0)),
        "max": float(np.max(values)),
    }


def _fuse_review_points(
    point_rows: list[np.ndarray],
    color_rows: list[np.ndarray],
    weight_rows: list[np.ndarray],
    view_rows: list[np.ndarray],
    *,
    voxel_size_m: float = 0.035,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.concatenate(point_rows).astype(np.float64)
    colors = np.concatenate(color_rows).astype(np.float64)
    weights = np.concatenate(weight_rows).astype(np.float64)
    view_ids = np.concatenate(view_rows)
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    voxel_count = int(np.max(inverse)) + 1
    weight_sum = np.bincount(inverse, weights=weights, minlength=voxel_count)
    fused_points = np.stack(
        [
            np.bincount(
                inverse, weights=weights * points[:, axis], minlength=voxel_count
            )
            / np.maximum(weight_sum, 1e-9)
            for axis in range(3)
        ],
        axis=1,
    )
    fused_colors = np.stack(
        [
            np.bincount(
                inverse, weights=weights * colors[:, axis], minlength=voxel_count
            )
            / np.maximum(weight_sum, 1e-9)
            for axis in range(3)
        ],
        axis=1,
    )
    unique_view_voxels = np.unique(np.column_stack((inverse, view_ids)), axis=0)
    support = np.bincount(
        unique_view_voxels[:, 0], minlength=voxel_count
    ).astype(np.int16)
    return (
        fused_points.astype(np.float32),
        np.clip(fused_colors, 0, 255).astype(np.uint8),
        support,
    )


def run_windowed_mapanything_scan(
    scan_dir: Path,
    output_dir: Path,
    prepared: dict[str, Any],
    settings: MapAnythingScanSettings,
    progress: ProgressCallback,
    *,
    window_runner: WindowRunner = run_mapanything_scan,
) -> dict[str, Any]:
    started = time.monotonic()
    frame_rows = prepared.get("frames")
    if not isinstance(frame_rows, list) or len(frame_rows) < 2:
        raise MapAnythingScanError("the scan has no valid prepared multi-view frame set")
    ranges = _window_ranges(
        len(frame_rows), settings.max_joint_views, settings.window_overlap_views
    )
    if len(ranges) == 1:
        return window_runner(scan_dir, output_dir, prepared, settings, progress)
    if settings.anchor_image is not None:
        raise MapAnythingScanError(
            "a fixed-camera anchor cannot be silently repeated across adaptive windows"
        )

    windows_root = output_dir / ".window-inference"
    windows_root.mkdir(parents=True, exist_ok=False)
    canonical: dict[int, dict[str, Any]] = {}
    window_reports: list[dict[str, Any]] = []
    peak_allocated = 0
    peak_reserved = 0
    joint_seconds = 0.0
    registration_settings = SupplementIntegrationSettings(
        max_total_views=settings.max_joint_views,
        min_bridge_views=max(4, min(8, settings.window_overlap_views)),
        max_bridge_views=settings.window_overlap_views,
        max_scale_ratio=1.50,
        max_bridge_position_p80_m=0.60,
    )

    for window_index, (start, end) in enumerate(ranges):
        window_dir = windows_root / f"window_{window_index:03d}"
        window_dir.mkdir()
        rows = [
            {**frame_rows[global_index], "adaptive_global_index": global_index}
            for global_index in range(start, end)
        ]
        window_prepared = {
            "frame_count": len(rows),
            "frames": rows,
            "adaptive_window": {
                "index": window_index,
                "global_start": start,
                "global_end_exclusive": end,
            },
        }
        progress(
            0.02 + 0.62 * (window_index / len(ranges)),
            f"Running MapAnything window {window_index + 1} of {len(ranges)}",
        )
        result = window_runner(
            scan_dir,
            window_dir,
            window_prepared,
            settings,
            lambda fraction, message, window_index=window_index: progress(
                0.02
                + 0.62 * ((window_index + min(1.0, float(fraction))) / len(ranges)),
                message,
            ),
        )
        runtime = result.get("runtime") if isinstance(result.get("runtime"), dict) else {}
        peak_allocated = max(peak_allocated, int(runtime.get("cuda_peak_allocated_bytes") or 0))
        peak_reserved = max(peak_reserved, int(runtime.get("cuda_peak_reserved_bytes") or 0))
        joint_seconds += float(runtime.get("joint_inference_s") or 0.0)

        scale = 1.0
        rotation = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)
        bridge_metrics: dict[str, Any] | None = None
        refinement_metrics: dict[str, Any] | None = None
        overlap_metrics: dict[str, Any] | None = None
        overlap_count = 0
        if window_index > 0:
            overlap_indices = [
                global_index
                for global_index in range(start, end)
                if global_index in canonical
            ]
            overlap_count = len(overlap_indices)
            if overlap_count < registration_settings.min_bridge_views:
                raise MapAnythingScanError(
                    f"adaptive window {window_index} has only {overlap_count} bridge views"
                )
            base_records = [canonical[index] for index in overlap_indices]
            append_raw = [
                _load_raw(window_dir / "raw" / f"view_{index - start:04d}.npz")
                for index in overlap_indices
            ]
            base_poses = np.stack(
                [
                    _transform_pose(
                        _load_raw(record["raw_path"])["camera_pose"],
                        float(record["scale"]),
                        np.asarray(record["rotation"], dtype=np.float64),
                        np.asarray(record["translation"], dtype=np.float64),
                    )
                    for record in base_records
                ]
            )
            append_poses = np.stack([raw["camera_pose"] for raw in append_raw])
            try:
                scale, rotation, translation, bridge_metrics = _estimate_bridge_similarity(
                    base_poses, append_poses, registration_settings
                )
            except SupplementIntegrationError as exc:
                raise MapAnythingScanError(
                    f"adaptive window {window_index} registration failed: {exc}"
                ) from exc
            scale, translation, refinement_metrics = _refine_window_scale_translation(
                base_records,
                append_raw,
                base_poses,
                append_poses,
                scale,
                rotation,
                translation,
            )
            overlap_metrics = _window_overlap_residual(
                base_records, append_raw, scale, rotation, translation
            )

        for global_index in range(start, end):
            if global_index in canonical:
                continue
            local_index = global_index - start
            canonical[global_index] = {
                "window_index": window_index,
                "local_index": local_index,
                "window_dir": window_dir,
                "raw_path": window_dir / "raw" / f"view_{local_index:04d}.npz",
                "scale": scale,
                "rotation": rotation,
                "translation": translation,
            }
        window_reports.append(
            {
                "window_index": window_index,
                "global_start": start,
                "global_end_exclusive": end,
                "view_count": end - start,
                "overlap_view_count": overlap_count,
                "window_to_base": {
                    "scale": scale,
                    "rotation_row_major": rotation.tolist(),
                    "translation_m": translation.tolist(),
                },
                "duplicate_pose_gate": bridge_metrics,
                "duplicate_surface_refinement": refinement_metrics,
                "duplicate_geometry_gate": overlap_metrics,
                "runtime": runtime,
            }
        )

    if len(canonical) != len(frame_rows):
        raise MapAnythingScanError(
            f"adaptive windows retained {len(canonical)} of {len(frame_rows)} prepared views"
        )

    progress(0.66, "Materializing the registered adaptive reconstruction")
    views_dir = output_dir / "views"
    raw_dir = output_dir / "raw"
    views_dir.mkdir()
    raw_dir.mkdir()
    final_frames: list[dict[str, Any]] = []
    camera_poses: list[np.ndarray] = []
    intrinsics_rows: list[np.ndarray] = []
    scales: list[float] = []
    point_rows: list[np.ndarray] = []
    color_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    view_id_rows: list[np.ndarray] = []
    per_view_budget = max(1000, int(settings.point_budget) // len(frame_rows))

    for global_index, source_row in enumerate(frame_rows):
        record = canonical[global_index]
        local_index = int(record["local_index"])
        window_dir = Path(record["window_dir"])
        scale = float(record["scale"])
        rotation = np.asarray(record["rotation"], dtype=np.float64)
        translation = np.asarray(record["translation"], dtype=np.float64)
        raw = _load_raw(Path(record["raw_path"]))
        world_points = _transform_points(
            raw["world_points"], scale, rotation, translation
        ).astype(np.float32)
        depth = (np.asarray(raw["depth_z"], dtype=np.float32) * scale).astype(np.float32)
        confidence = np.asarray(raw["confidence"], dtype=np.float32)
        mask = np.asarray(raw["mask"], dtype=bool)
        pose = _transform_pose(
            raw["camera_pose"], scale, rotation, translation
        ).astype(np.float32)
        intrinsics = np.asarray(raw["intrinsics"], dtype=np.float32)
        effective_scale = float(np.asarray(raw["metric_scaling_factor"]).reshape(-1)[0] * scale)
        model_rgb = np.asarray(raw["model_rgb"], dtype=np.float32)
        raw_path = raw_dir / f"view_{global_index:04d}.npz"
        np.savez_compressed(
            raw_path,
            world_points=world_points,
            depth_z=depth,
            confidence=confidence,
            mask=mask.astype(np.uint8),
            camera_pose=pose,
            intrinsics=intrinsics,
            metric_scaling_factor=np.asarray([effective_scale], dtype=np.float32),
            model_rgb=model_rgb,
        )

        preview_paths: dict[str, str] = {}
        for kind in ("rgb", "depth", "confidence", "mask"):
            source = window_dir / "views" / f"view_{local_index:04d}_{kind}.png"
            target = views_dir / f"view_{global_index:04d}_{kind}.png"
            shutil.copy2(source, target)
            preview_paths[kind] = f"outputs/views/{target.name}"

        valid = (
            mask
            & np.isfinite(depth)
            & (depth > 0.0)
            & np.isfinite(confidence)
            & np.isfinite(world_points).all(axis=2)
        )
        valid_indices = np.flatnonzero(valid.reshape(-1))
        if valid_indices.size:
            positions = np.linspace(
                0,
                valid_indices.size - 1,
                min(per_view_budget, valid_indices.size),
                dtype=np.int64,
            )
            chosen = valid_indices[positions]
            values = confidence[valid]
            low, high = np.percentile(values, (10.0, 90.0))
            weights = 0.25 + 0.75 * np.clip(
                (confidence.reshape(-1)[chosen] - low) / max(float(high - low), 1e-8),
                0.0,
                1.0,
            )
            point_rows.append(world_points.reshape(-1, 3)[chosen])
            color_rows.append(_as_u8(model_rgb).reshape(-1, 3)[chosen])
            weight_rows.append(weights.astype(np.float32))
            view_id_rows.append(
                np.full(chosen.size, global_index, dtype=np.int32)
            )

        camera_poses.append(pose)
        intrinsics_rows.append(intrinsics)
        scales.append(effective_scale)
        final_frames.append(
            {
                "index": global_index,
                "source_frame": str(source_row["frame"]),
                "fixed_camera_anchor": False,
                "timestamp_s": source_row.get("timestamp_s"),
                "adaptive_window_index": int(record["window_index"]),
                "model_rgb": preview_paths["rgb"],
                "depth_preview": preview_paths["depth"],
                "confidence_preview": preview_paths["confidence"],
                "mask_preview": preview_paths["mask"],
                "raw_npz": f"outputs/raw/{raw_path.name}",
                "depth": _depth_stats(depth, mask),
                "confidence": _confidence_stats(confidence, mask),
                "camera_pose": pose.tolist(),
                "intrinsics": intrinsics.tolist(),
                "metric_scaling_factor": effective_scale,
            }
        )
        if global_index % 16 == 0:
            progress(
                0.66 + 0.20 * ((global_index + 1) / len(frame_rows)),
                f"Saving adaptive view {global_index + 1} of {len(frame_rows)}",
            )

    if not point_rows:
        raise MapAnythingScanError("adaptive windows produced no valid review points")
    points, colors, support = _fuse_review_points(
        point_rows, color_rows, weight_rows, view_id_rows
    )
    poses_array = np.stack(camera_poses).astype(np.float32)
    intrinsics_array = np.stack(intrinsics_rows).astype(np.float32)
    scales_array = np.asarray(scales, dtype=np.float32)
    glb_path = output_dir / "reconstruction_points.glb"
    trajectory_preview = output_dir / "camera_trajectory_topdown.png"
    camera_solution = output_dir / "camera_solution.npz"
    surfels_path = output_dir / "reconstruction_surfels.npz"
    _write_reconstruction_glb(glb_path, points, colors, poses_array[:, :3, 3])
    _write_trajectory_preview(
        trajectory_preview, points, poses_array[:, :3, 3]
    )
    np.savez_compressed(
        camera_solution,
        camera_poses=poses_array,
        intrinsics=intrinsics_array,
        metric_scaling_factors=scales_array,
    )
    np.savez_compressed(
        surfels_path,
        points=points,
        colors=colors,
        view_support=support,
        voxel_size_m=np.asarray([0.035], dtype=np.float32),
    )
    trajectory_json = output_dir / "camera_trajectory.json"
    trajectory_json.write_text(
        json.dumps(
            {
                "schema": "noesis.mapanything.phone_scan.camera_trajectory.v1",
                "pose_convention": "opencv_cam2world_x_right_y_down_z_forward",
                "camera_poses": poses_array.tolist(),
                "intrinsics": intrinsics_array.tolist(),
                "metric_scaling_factors": scales_array.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    registration_report = output_dir / "window_registration_report.json"
    registration_report.write_text(
        json.dumps(
            {
                "schema": "noesis.mapanything.phone_scan.window_registration.v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "view_count": len(frame_rows),
                "max_joint_views": settings.max_joint_views,
                "window_overlap_views": settings.window_overlap_views,
                "windows": window_reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.rmtree(windows_root)

    bounds_min = np.min(points, axis=0)
    bounds_max = np.max(points, axis=0)
    summary: dict[str, Any] = {
        "schema": "noesis.mapanything.phone_scan.outputs.v2",
        "model": {
            "id": settings.model_id,
            "device": settings.device,
            "amp_dtype": settings.amp_dtype,
            "memory_efficient_inference": True,
            "minibatch_size": 1,
            "apply_mask": True,
            "mask_edges": True,
            "use_multiview_confidence": False,
            "adaptive_windowing": True,
            "max_joint_views": settings.max_joint_views,
            "window_overlap_views": settings.window_overlap_views,
        },
        "runtime": {
            "joint_inference_s": joint_seconds,
            "total_processing_s": float(time.monotonic() - started),
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
        },
        "view_count": len(final_frames),
        "phone_view_count": len(final_frames),
        "window_count": len(ranges),
        "anchor_view_index": None,
        "review_point_count": int(points.shape[0]),
        "coordinate_frame": "mapanything_metric_world_window_registered_unaligned_to_noesis",
        "pose_convention": "opencv_cam2world_x_right_y_down_z_forward",
        "bounds": {"min": bounds_min.tolist(), "max": bounds_max.tolist()},
        "scale": {
            "min": float(np.min(scales_array)),
            "median": float(np.median(scales_array)),
            "max": float(np.max(scales_array)),
        },
        "artifacts": {
            "reconstruction_glb": "outputs/reconstruction_points.glb",
            "trajectory_preview": "outputs/camera_trajectory_topdown.png",
            "trajectory_json": "outputs/camera_trajectory.json",
            "camera_solution_npz": "outputs/camera_solution.npz",
            "reconstruction_surfels_npz": "outputs/reconstruction_surfels.npz",
            "window_registration_report": "outputs/window_registration_report.json",
            "manifest": "outputs/scan_outputs_manifest.json",
        },
        "window_registration": {
            "window_count": len(window_reports),
            "all_duplicate_pose_gates_passed": True,
            "all_duplicate_geometry_gates_passed": True,
        },
        "frames": final_frames,
    }
    manifest_path = output_dir / "scan_outputs_manifest.json"
    summary["files"] = [
        {
            "path": f"outputs/{path.relative_to(output_dir).as_posix()}",
            "size_bytes": int(path.stat().st_size),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    summary["files"].append(
        {"path": "outputs/scan_outputs_manifest.json", "size_bytes": 0}
    )
    for _ in range(3):
        manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        size = int(manifest_path.stat().st_size)
        if summary["files"][-1]["size_bytes"] == size:
            break
        summary["files"][-1]["size_bytes"] = size
    progress(1.0, "Adaptive MapAnything windows are registered and saved")
    return summary


def run_adaptive_mapanything_scan(
    scan_dir: Path,
    output_dir: Path,
    prepared: dict[str, Any],
    settings: MapAnythingScanSettings,
    progress: ProgressCallback,
) -> dict[str, Any]:
    return run_windowed_mapanything_scan(
        scan_dir, output_dir, prepared, settings, progress
    )


__all__ = [
    "run_adaptive_mapanything_scan",
    "run_windowed_mapanything_scan",
]
