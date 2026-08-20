from __future__ import annotations

import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .da3_inference import (
    DA3PhoneScanError,
    DA3PhoneScanSettings,
    run_da3_phone_scan,
)
from .inference import _write_reconstruction_glb, _write_trajectory_preview
from .supplement import (
    SupplementIntegrationError,
    SupplementIntegrationSettings,
    _estimate_bridge_similarity,
    _load_raw,
    _transform_points,
    _transform_pose,
)
from .windowed_inference import (
    _confidence_stats,
    _depth_stats,
    _fuse_review_points,
    _refine_window_scale_translation,
    _window_overlap_residual,
    _window_ranges,
)


ProgressCallback = Callable[[float, str], None]
WindowRunner = Callable[
    [Path, Path, dict[str, Any], DA3PhoneScanSettings, ProgressCallback],
    dict[str, Any],
]


def _as_u8(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.dtype == np.uint8:
        return value
    if value.size and float(np.nanmax(value)) <= 1.5:
        value = value * 255.0
    return np.clip(value, 0, 255).astype(np.uint8)


def run_windowed_da3_phone_scan(
    scan_dir: Path,
    output_dir: Path,
    prepared: dict[str, Any],
    settings: DA3PhoneScanSettings,
    progress: ProgressCallback,
    *,
    window_runner: WindowRunner = run_da3_phone_scan,
) -> dict[str, Any]:
    """Run every adaptive view through bounded, overlap-registered DA3 windows."""

    started = time.monotonic()
    frame_rows = prepared.get("frames")
    if not isinstance(frame_rows, list) or len(frame_rows) < 2:
        raise DA3PhoneScanError("the scan has no valid prepared multi-view frame set")
    try:
        ranges = _window_ranges(
            len(frame_rows), settings.max_joint_views, settings.window_overlap_views
        )
    except Exception as exc:
        raise DA3PhoneScanError(str(exc)) from exc
    if len(ranges) == 1:
        return window_runner(scan_dir, output_dir, prepared, settings, progress)
    if settings.anchor_image is not None:
        raise DA3PhoneScanError(
            "a fixed-camera anchor cannot be silently repeated across adaptive DA3 windows"
        )

    windows_root = output_dir / ".window-inference"
    windows_root.mkdir(parents=True, exist_ok=False)
    canonical: dict[int, dict[str, Any]] = {}
    reports: list[dict[str, Any]] = []
    registration_settings = SupplementIntegrationSettings(
        max_total_views=settings.max_joint_views,
        min_bridge_views=max(6, min(10, settings.window_overlap_views)),
        max_bridge_views=settings.window_overlap_views,
        max_scale_ratio=1.50,
        max_bridge_position_p80_m=0.80,
    )

    for window_index, (start, end) in enumerate(ranges):
        window_dir = windows_root / f"window_{window_index:03d}"
        window_dir.mkdir()
        rows = [
            {**frame_rows[index], "adaptive_global_index": index}
            for index in range(start, end)
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
            f"Running DA3 window {window_index + 1} of {len(ranges)}",
        )
        result = window_runner(
            scan_dir,
            window_dir,
            window_prepared,
            settings,
            lambda fraction, message, window_index=window_index: progress(
                0.02
                + 0.62
                * ((window_index + min(1.0, float(fraction))) / len(ranges)),
                message,
            ),
        )

        scale = 1.0
        rotation = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)
        pose_gate: dict[str, Any] | None = None
        refinement: dict[str, Any] | None = None
        geometry_gate: dict[str, Any] | None = None
        overlap_indices = [index for index in range(start, end) if index in canonical]
        if window_index > 0:
            if len(overlap_indices) < registration_settings.min_bridge_views:
                raise DA3PhoneScanError(
                    f"adaptive DA3 window {window_index} has only "
                    f"{len(overlap_indices)} bridge views"
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
            append_poses = np.stack([row["camera_pose"] for row in append_raw])
            try:
                scale, rotation, translation, pose_gate = _estimate_bridge_similarity(
                    base_poses, append_poses, registration_settings
                )
            except SupplementIntegrationError as exc:
                raise DA3PhoneScanError(
                    f"adaptive DA3 window {window_index} registration failed: {exc}"
                ) from exc
            try:
                scale, translation, refinement = _refine_window_scale_translation(
                    base_records,
                    append_raw,
                    base_poses,
                    append_poses,
                    scale,
                    rotation,
                    translation,
                )
                geometry_gate = _window_overlap_residual(
                    base_records, append_raw, scale, rotation, translation
                )
            except Exception as exc:
                raise DA3PhoneScanError(
                    f"adaptive DA3 window {window_index} surface registration failed: {exc}"
                ) from exc

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
        reports.append(
            {
                "window_index": window_index,
                "global_start": start,
                "global_end_exclusive": end,
                "view_count": end - start,
                "overlap_view_count": len(overlap_indices),
                "window_to_base": {
                    "scale": scale,
                    "rotation_row_major": rotation.tolist(),
                    "translation_m": translation.tolist(),
                },
                "duplicate_pose_gate": pose_gate,
                "duplicate_surface_refinement": refinement,
                "duplicate_geometry_gate": geometry_gate,
                "runtime": {
                    "elapsed_s": float(result.get("elapsed_s") or 0.0),
                    "view_count": int(result.get("view_count") or end - start),
                },
            }
        )

    if len(canonical) != len(frame_rows):
        raise DA3PhoneScanError(
            f"adaptive DA3 windows retained {len(canonical)} of "
            f"{len(frame_rows)} prepared views"
        )

    progress(0.66, "Materializing the registered adaptive DA3 reconstruction")
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
    per_view_budget = max(1_000, int(settings.point_budget) // len(frame_rows))

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
        depth = (np.asarray(raw["depth_z"], dtype=np.float32) * scale).astype(
            np.float32
        )
        confidence = np.asarray(raw["confidence"], dtype=np.float32)
        mask = np.asarray(raw["mask"], dtype=bool)
        pose = _transform_pose(
            raw["camera_pose"], scale, rotation, translation
        ).astype(np.float32)
        intrinsics = np.asarray(raw["intrinsics"], dtype=np.float32)
        effective_scale = float(
            np.asarray(raw["metric_scaling_factor"]).reshape(-1)[0] * scale
        )
        model_rgb = np.asarray(raw["model_rgb"])
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

        previews: dict[str, str] = {}
        for kind in ("rgb", "depth", "confidence", "mask"):
            source = window_dir / "views" / f"view_{local_index:04d}_{kind}.png"
            target = views_dir / f"view_{global_index:04d}_{kind}.png"
            shutil.copy2(source, target)
            previews[kind] = f"outputs/views/{target.name}"

        valid = (
            mask
            & np.isfinite(depth)
            & (depth > 0.0)
            & np.isfinite(confidence)
            & np.isfinite(world_points).all(axis=2)
        )
        selected = np.flatnonzero(valid.reshape(-1))
        if selected.size:
            threshold = float(np.percentile(confidence[valid], 55.0))
            selected = np.flatnonzero(
                (valid & (confidence >= threshold)).reshape(-1)
            )
            stride = max(1, int(math.ceil(selected.size / per_view_budget)))
            selected = selected[::stride][:per_view_budget]
            conf_values = confidence[valid]
            low, high = np.percentile(conf_values, (10.0, 90.0))
            weights = 0.25 + 0.75 * np.clip(
                (confidence.reshape(-1)[selected] - low)
                / max(float(high - low), 1e-8),
                0.0,
                1.0,
            )
            point_rows.append(world_points.reshape(-1, 3)[selected])
            color_rows.append(_as_u8(model_rgb).reshape(-1, 3)[selected])
            weight_rows.append(weights.astype(np.float32))
            view_id_rows.append(
                np.full(selected.size, global_index, dtype=np.int32)
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
                "model_rgb": previews["rgb"],
                "depth_preview": previews["depth"],
                "confidence_preview": previews["confidence"],
                "mask_preview": previews["mask"],
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
                f"Saving adaptive DA3 view {global_index + 1} of {len(frame_rows)}",
            )

    if not point_rows:
        raise DA3PhoneScanError("adaptive DA3 windows produced no valid review points")
    points, colors, support = _fuse_review_points(
        point_rows, color_rows, weight_rows, view_id_rows
    )
    poses_array = np.stack(camera_poses).astype(np.float32)
    intrinsics_array = np.stack(intrinsics_rows).astype(np.float32)
    scales_array = np.asarray(scales, dtype=np.float32)
    _write_reconstruction_glb(
        output_dir / "reconstruction_points.glb",
        points,
        colors,
        poses_array[:, :3, 3],
    )
    _write_trajectory_preview(
        output_dir / "camera_trajectory_topdown.png",
        points,
        poses_array[:, :3, 3],
    )
    np.savez_compressed(
        output_dir / "camera_solution.npz",
        camera_poses=poses_array,
        intrinsics=intrinsics_array,
        metric_scaling_factors=scales_array,
    )
    np.savez_compressed(
        output_dir / "reconstruction_surfels.npz",
        points=points,
        colors=colors,
        view_support=support,
        voxel_size_m=np.asarray([0.035], dtype=np.float32),
    )
    (output_dir / "camera_trajectory.json").write_text(
        json.dumps(
            {
                "schema": "noesis.da3.phone_scan.camera_trajectory.v1",
                "provider": "da3",
                "model_id": settings.model_id,
                "pose_convention": "opencv_cam2world_x_right_y_down_z_forward",
                "camera_to_world": poses_array.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "window_registration_report.json").write_text(
        json.dumps(
            {
                "schema": "noesis.da3.phone_scan.window_registration.v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "view_count": len(frame_rows),
                "max_joint_views": settings.max_joint_views,
                "window_overlap_views": settings.window_overlap_views,
                "windows": reports,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(windows_root)

    bounds_min = np.min(points, axis=0)
    bounds_max = np.max(points, axis=0)
    summary: dict[str, Any] = {
        "schema": "noesis.phone_scan.outputs.v2",
        "provider": "da3",
        "model_id": settings.model_id,
        "mode": (
            "DA3-BASE overlapping any-view windows registered by duplicate RGB-D "
            "views and metricized by DA3Metric-Large TensorRT"
        ),
        "view_count": len(frame_rows),
        "phone_view_count": len(frame_rows),
        "window_count": len(ranges),
        "anchor_view_index": None,
        "review_point_count": int(points.shape[0]),
        "coordinate_frame": "da3_metric_world_window_registered_unaligned_to_noesis",
        "pose_convention": "opencv_cam2world_x_right_y_down_z_forward",
        "bounds": {"min": bounds_min.tolist(), "max": bounds_max.tolist()},
        "scale": {
            "min": float(np.min(scales_array)),
            "median": float(np.median(scales_array)),
            "max": float(np.max(scales_array)),
        },
        "window_registration": {
            "max_joint_views": settings.max_joint_views,
            "window_overlap_views": settings.window_overlap_views,
            "all_duplicate_pose_gates_passed": True,
            "all_duplicate_geometry_gates_passed": True,
        },
        "elapsed_s": float(time.monotonic() - started),
        "artifacts": {
            "reconstruction_glb": "outputs/reconstruction_points.glb",
            "trajectory_preview": "outputs/camera_trajectory_topdown.png",
            "trajectory_json": "outputs/camera_trajectory.json",
            "camera_solution_npz": "outputs/camera_solution.npz",
            "reconstruction_surfels_npz": "outputs/reconstruction_surfels.npz",
            "window_registration_report": "outputs/window_registration_report.json",
            "manifest": "outputs/scan_outputs_manifest.json",
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
        manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        size = int(manifest_path.stat().st_size)
        if summary["files"][-1]["size_bytes"] == size:
            break
        summary["files"][-1]["size_bytes"] = size
    progress(1.0, "Adaptive DA3 windows are registered and saved")
    return summary


def run_adaptive_da3_phone_scan(
    scan_dir: Path,
    output_dir: Path,
    prepared: dict[str, Any],
    settings: DA3PhoneScanSettings,
    progress: ProgressCallback,
) -> dict[str, Any]:
    return run_windowed_da3_phone_scan(
        scan_dir, output_dir, prepared, settings, progress
    )


__all__ = ["run_adaptive_da3_phone_scan", "run_windowed_da3_phone_scan"]
