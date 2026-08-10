from __future__ import annotations

import gc
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


ProgressCallback = Callable[[float, str], None]


class MapAnythingScanError(RuntimeError):
    """Raised when the canonical MapAnything scan cannot be completed."""


def _prepare_fixed_camera_anchor(
    anchor_path: Path,
    phone_frame_path: Path,
    output_path: Path,
) -> Path:
    """Letterbox the full static view to the phone frame aspect without cropping it."""
    anchor = cv2.imread(str(anchor_path), cv2.IMREAD_COLOR)
    phone = cv2.imread(str(phone_frame_path), cv2.IMREAD_COLOR)
    if anchor is None or phone is None:
        raise MapAnythingScanError("fixed-camera anchor or phone reference frame is unreadable")
    target_height, target_width = phone.shape[:2]
    scale = min(target_width / anchor.shape[1], target_height / anchor.shape[0])
    resized = cv2.resize(
        anchor,
        (
            max(1, int(round(anchor.shape[1] * scale))),
            max(1, int(round(anchor.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((target_height, target_width, 3), 12, dtype=np.uint8)
    left = (target_width - resized.shape[1]) // 2
    top = (target_height - resized.shape[0]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise MapAnythingScanError("failed to save the fixed-camera anchor input")
    return output_path


@dataclass(frozen=True)
class MapAnythingScanSettings:
    model_id: str = "facebook/map-anything-apache"
    device: str = "cuda:0"
    amp_dtype: str = "bf16"
    point_budget: int = 600_000
    local_files_only: bool = True
    anchor_image: Path | None = None


def _tensor_numpy(value: Any, *, name: str) -> np.ndarray:
    if value is None:
        raise MapAnythingScanError(f"MapAnything output is missing {name}")
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def _first_batch(value: Any, *, name: str) -> np.ndarray:
    array = _tensor_numpy(value, name=name)
    if array.ndim == 0 or array.shape[0] != 1:
        raise MapAnythingScanError(
            f"MapAnything output {name} must have a one-item batch, got {array.shape}"
        )
    return np.asarray(array[0])


def _map2d(value: Any, *, name: str) -> np.ndarray:
    array = _first_batch(value, name=name)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise MapAnythingScanError(f"MapAnything output {name} is not a 2D map: {array.shape}")
    return array


def _map3d(value: Any, *, name: str) -> np.ndarray:
    array = _first_batch(value, name=name)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise MapAnythingScanError(f"MapAnything output {name} is not HxWx3: {array.shape}")
    return array


def _matrix(value: Any, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    array = _first_batch(value, name=name)
    if array.shape != shape:
        raise MapAnythingScanError(
            f"MapAnything output {name} has shape {array.shape}, expected {shape}"
        )
    return array


def _scalar(value: Any, *, name: str) -> float:
    array = _tensor_numpy(value, name=name).reshape(-1)
    if array.size != 1:
        raise MapAnythingScanError(f"MapAnything output {name} is not scalar")
    result = float(array[0])
    if not math.isfinite(result):
        raise MapAnythingScanError(f"MapAnything output {name} is non-finite")
    return result


def _write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    image = np.asarray(image_rgb)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 if float(np.nanmax(image)) <= 1.5 else image, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
        raise MapAnythingScanError(f"failed to write {path.name}")


def _write_depth_preview(path: Path, depth: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        raise MapAnythingScanError("MapAnything produced no finite positive depth")
    values = depth[valid]
    low, median, high = (float(value) for value in np.percentile(values, (2.0, 50.0, 98.0)))
    span = max(high - low, 1e-6)
    normalized = np.clip((depth - low) / span, 0.0, 1.0)
    gray = np.where(valid, np.round(normalized * 255.0), 0).astype(np.uint8)
    colored = cv2.applyColorMap(255 - gray, cv2.COLORMAP_TURBO)
    colored[~valid] = (18, 18, 18)
    if not cv2.imwrite(str(path), colored, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
        raise MapAnythingScanError(f"failed to write {path.name}")
    return {
        "min_m": float(np.min(values)),
        "p02_m": low,
        "p50_m": median,
        "p98_m": high,
        "max_m": float(np.max(values)),
        "valid_fraction": float(np.count_nonzero(valid) / valid.size),
    }


def _write_confidence_preview(path: Path, confidence: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(confidence)
    if not np.any(valid):
        raise MapAnythingScanError("MapAnything produced no finite confidence")
    values = confidence[valid]
    low, high = (float(value) for value in np.percentile(values, (2.0, 98.0)))
    span = max(high - low, 1e-8)
    gray = np.where(valid, np.clip((confidence - low) / span, 0.0, 1.0) * 255.0, 0).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    colored[~valid] = (18, 18, 18)
    if not cv2.imwrite(str(path), colored, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
        raise MapAnythingScanError(f"failed to write {path.name}")
    return {
        "min": float(np.min(values)),
        "p02": low,
        "p50": float(np.percentile(values, 50.0)),
        "p98": high,
        "max": float(np.max(values)),
    }


def _write_mask(path: Path, mask: np.ndarray) -> None:
    image = np.where(mask, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
        raise MapAnythingScanError(f"failed to write {path.name}")


def _cylinder_between(start: np.ndarray, end: np.ndarray, radius: float) -> Any | None:
    import trimesh

    vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length <= 1e-8:
        return None
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=8)
    transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], vector / length)
    if transform is None:
        transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = (np.asarray(start, dtype=np.float64) + np.asarray(end, dtype=np.float64)) * 0.5
    mesh.apply_transform(transform)
    mesh.visual.vertex_colors = np.tile(np.asarray([255, 148, 44, 255], dtype=np.uint8), (len(mesh.vertices), 1))
    return mesh


def _write_reconstruction_glb(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    camera_positions: np.ndarray,
) -> None:
    import trimesh

    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise MapAnythingScanError("no valid world points are available for the GLB review asset")
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.points.PointCloud(points, colors=colors), geom_name="mapanything_points")
    finite_positions = camera_positions[np.isfinite(camera_positions).all(axis=1)]
    if finite_positions.shape[0] >= 2:
        scene_extent = np.ptp(points, axis=0)
        radius = max(float(np.linalg.norm(scene_extent)) * 0.0015, 0.002)
        segments = [
            segment
            for segment in (
                _cylinder_between(finite_positions[index], finite_positions[index + 1], radius)
                for index in range(finite_positions.shape[0] - 1)
            )
            if segment is not None
        ]
        if segments:
            scene.add_geometry(trimesh.util.concatenate(segments), geom_name="camera_trajectory")
    rotation_x = trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    scene.apply_transform(rotation_x)
    try:
        scene.export(str(path))
    except Exception as exc:
        raise MapAnythingScanError(f"failed to export reconstruction GLB: {exc}") from exc


def _write_trajectory_preview(
    path: Path,
    points: np.ndarray,
    camera_positions: np.ndarray,
) -> None:
    canvas_size = 1200
    canvas = np.full((canvas_size, canvas_size, 3), 18, dtype=np.uint8)
    finite_points = points[np.isfinite(points).all(axis=1)]
    finite_cameras = camera_positions[np.isfinite(camera_positions).all(axis=1)]
    if finite_points.shape[0] == 0 or finite_cameras.shape[0] == 0:
        raise MapAnythingScanError("cannot render camera trajectory without finite points and poses")
    cloud_xz = finite_points[:, [0, 2]]
    cameras_xz = finite_cameras[:, [0, 2]]
    all_xz = np.concatenate([cloud_xz, cameras_xz], axis=0)
    low = np.percentile(all_xz, 1.0, axis=0)
    high = np.percentile(all_xz, 99.0, axis=0)
    span = np.maximum(high - low, 1e-6)
    margin = 60

    def project(values: np.ndarray) -> np.ndarray:
        normalized = (values - low) / span
        xy = normalized * (canvas_size - 2 * margin) + margin
        xy[:, 1] = canvas_size - xy[:, 1]
        return np.round(xy).astype(np.int32)

    cloud_stride = max(1, int(math.ceil(cloud_xz.shape[0] / 80_000.0)))
    for x, y in project(cloud_xz[::cloud_stride]):
        if 0 <= x < canvas_size and 0 <= y < canvas_size:
            canvas[y, x] = (78, 78, 78)
    path_xy = project(cameras_xz)
    if path_xy.shape[0] >= 2:
        cv2.polylines(canvas, [path_xy.reshape((-1, 1, 2))], False, (44, 148, 255), 4, cv2.LINE_AA)
    for index, (x, y) in enumerate(path_xy):
        color = (96, 220, 96) if index == 0 else (44, 148, 255)
        cv2.circle(canvas, (int(x), int(y)), 7, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, "Top-down MapAnything world frame", (34, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
        raise MapAnythingScanError("failed to write camera trajectory preview")


def run_mapanything_scan(
    scan_dir: Path,
    output_dir: Path,
    prepared: dict[str, Any],
    settings: MapAnythingScanSettings,
    progress: ProgressCallback,
) -> dict[str, Any]:
    if not settings.model_id.endswith("-apache"):
        raise MapAnythingScanError("the phone-scan tool only permits the Apache-licensed MapAnything model")
    frame_rows = prepared.get("frames")
    if not isinstance(frame_rows, list) or len(frame_rows) < 2:
        raise MapAnythingScanError("the scan has no valid prepared multi-view frame set")
    phone_frame_paths = [scan_dir / str(row["frame"]) for row in frame_rows]
    inference_rows = list(frame_rows)
    anchor_view_index: int | None = None
    if settings.anchor_image is not None:
        anchor_path = settings.anchor_image.resolve()
        if not anchor_path.is_file():
            raise MapAnythingScanError(f"fixed-camera anchor image is missing: {anchor_path}")
        prepared_anchor = _prepare_fixed_camera_anchor(
            anchor_path,
            phone_frame_paths[0],
            output_dir / "fixed_camera_anchor_input.jpg",
        )
        inference_rows = [
            {
                "frame": str(prepared_anchor),
                "timestamp_s": None,
                "fixed_camera_anchor": True,
            },
            *inference_rows,
        ]
        anchor_view_index = 0
    frame_paths = [
        Path(str(row["frame"]))
        if Path(str(row["frame"])).is_absolute()
        else scan_dir / str(row["frame"])
        for row in inference_rows
    ]
    missing = [path.name for path in frame_paths if not path.is_file()]
    if missing:
        raise MapAnythingScanError(f"prepared MapAnything frames are missing: {missing[:4]}")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    progress(0.02, "Loading the Apache MapAnything model")
    model: Any | None = None
    outputs: Any | None = None
    try:
        import torch
        from mapanything.models import MapAnything
        from mapanything.utils.image import load_images

        if settings.device.startswith("cuda") and not torch.cuda.is_available():
            raise MapAnythingScanError(
                f"MapAnything device {settings.device} was requested but CUDA is unavailable"
            )
        device = torch.device(settings.device)
        model = MapAnything.from_pretrained(
            settings.model_id,
            local_files_only=settings.local_files_only,
        ).to(device)
        model.eval()
        progress(0.14, f"Loading and normalizing {len(frame_paths)} prepared views")
        views = load_images([str(path) for path in frame_paths])
        if len(views) != len(frame_paths):
            raise MapAnythingScanError(
                f"MapAnything loaded {len(views)} views, expected {len(frame_paths)}"
            )
        progress(0.20, f"Running joint MapAnything inference on {len(views)} views")
        with torch.inference_mode():
            outputs = model.infer(
                views,
                memory_efficient_inference=True,
                minibatch_size=1,
                use_amp=True,
                amp_dtype=settings.amp_dtype,
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
                confidence_percentile=10,
                use_multiview_confidence=False,
            )
        if not isinstance(outputs, list) or len(outputs) != len(frame_paths):
            raise MapAnythingScanError(
                f"MapAnything returned {len(outputs) if isinstance(outputs, list) else 'invalid'} views"
            )

        preview_root = output_dir / "views"
        raw_root = output_dir / "raw"
        preview_root.mkdir(parents=True, exist_ok=False)
        raw_root.mkdir(parents=True, exist_ok=False)
        frame_results: list[dict[str, Any]] = []
        camera_poses: list[np.ndarray] = []
        intrinsics_rows: list[np.ndarray] = []
        scales: list[float] = []
        sampled_points: list[np.ndarray] = []
        sampled_colors: list[np.ndarray] = []
        per_view_budget = max(1000, int(settings.point_budget) // len(outputs))

        for index, pred in enumerate(outputs):
            if not isinstance(pred, dict):
                raise MapAnythingScanError(f"MapAnything prediction {index} is malformed")
            world_points = _map3d(pred.get("pts3d"), name="pts3d").astype(np.float32)
            depth = _map2d(pred.get("depth_z"), name="depth_z").astype(np.float32)
            confidence = _map2d(pred.get("conf"), name="conf").astype(np.float32)
            mask = _map2d(pred.get("mask"), name="mask").astype(bool)
            image_rgb = _map3d(pred.get("img_no_norm"), name="img_no_norm")
            pose = _matrix(pred.get("camera_poses"), name="camera_poses", shape=(4, 4)).astype(np.float32)
            intrinsics = _matrix(pred.get("intrinsics"), name="intrinsics", shape=(3, 3)).astype(np.float32)
            scale = _scalar(pred.get("metric_scaling_factor"), name="metric_scaling_factor")
            if depth.shape != mask.shape or confidence.shape != mask.shape or world_points.shape[:2] != mask.shape:
                raise MapAnythingScanError(f"MapAnything prediction {index} has inconsistent map shapes")

            stem = f"view_{index:04d}"
            model_rgb_path = preview_root / f"{stem}_rgb.png"
            depth_path = preview_root / f"{stem}_depth.png"
            confidence_path = preview_root / f"{stem}_confidence.png"
            mask_path = preview_root / f"{stem}_mask.png"
            raw_path = raw_root / f"{stem}.npz"
            _write_rgb(model_rgb_path, image_rgb)
            depth_stats = _write_depth_preview(depth_path, depth, mask)
            confidence_stats = _write_confidence_preview(confidence_path, confidence, mask)
            _write_mask(mask_path, mask)
            np.savez_compressed(
                raw_path,
                world_points=world_points,
                depth_z=depth,
                confidence=confidence,
                mask=mask.astype(np.uint8),
                camera_pose=pose,
                intrinsics=intrinsics,
                metric_scaling_factor=np.asarray([scale], dtype=np.float32),
                model_rgb=np.asarray(image_rgb, dtype=np.float32),
            )

            valid = mask & np.isfinite(depth) & (depth > 0.0) & np.isfinite(world_points).all(axis=2)
            valid_indices = np.flatnonzero(valid.reshape(-1))
            if valid_indices.size:
                sample_stride = max(1, int(math.ceil(valid_indices.size / per_view_budget)))
                chosen = valid_indices[::sample_stride][:per_view_budget]
                image_u8 = np.clip(image_rgb * 255.0 if float(np.nanmax(image_rgb)) <= 1.5 else image_rgb, 0, 255).astype(np.uint8)
                sampled_points.append(world_points.reshape((-1, 3))[chosen])
                sampled_colors.append(image_u8.reshape((-1, 3))[chosen])
            camera_poses.append(pose)
            intrinsics_rows.append(intrinsics)
            scales.append(scale)
            frame_results.append(
                {
                    "index": index,
                    "source_frame": (
                        None
                        if inference_rows[index].get("fixed_camera_anchor")
                        else str(inference_rows[index]["frame"])
                    ),
                    "fixed_camera_anchor": bool(
                        inference_rows[index].get("fixed_camera_anchor")
                    ),
                    "timestamp_s": inference_rows[index].get("timestamp_s"),
                    "model_rgb": f"outputs/views/{model_rgb_path.name}",
                    "depth_preview": f"outputs/views/{depth_path.name}",
                    "confidence_preview": f"outputs/views/{confidence_path.name}",
                    "mask_preview": f"outputs/views/{mask_path.name}",
                    "raw_npz": f"outputs/raw/{raw_path.name}",
                    "depth": depth_stats,
                    "confidence": confidence_stats,
                    "camera_pose": pose.tolist(),
                    "intrinsics": intrinsics.tolist(),
                    "metric_scaling_factor": scale,
                }
            )
            outputs[index] = {}
            progress(
                0.55 + 0.30 * ((index + 1) / len(frame_paths)),
                f"Saving MapAnything view {index + 1} of {len(frame_paths)}",
            )

        if not sampled_points:
            raise MapAnythingScanError("MapAnything produced no valid world points")
        points = np.concatenate(sampled_points, axis=0).astype(np.float32)
        colors = np.concatenate(sampled_colors, axis=0).astype(np.uint8)
        poses_array = np.stack(camera_poses, axis=0).astype(np.float32)
        intrinsics_array = np.stack(intrinsics_rows, axis=0).astype(np.float32)
        scales_array = np.asarray(scales, dtype=np.float32)
        progress(0.88, "Building the phone-review reconstruction and camera path")
        glb_path = output_dir / "reconstruction_points.glb"
        trajectory_preview = output_dir / "camera_trajectory_topdown.png"
        camera_solution = output_dir / "camera_solution.npz"
        _write_reconstruction_glb(glb_path, points, colors, poses_array[:, :3, 3])
        _write_trajectory_preview(trajectory_preview, points, poses_array[:, :3, 3])
        np.savez_compressed(
            camera_solution,
            camera_poses=poses_array,
            intrinsics=intrinsics_array,
            metric_scaling_factors=scales_array,
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
        finite_points = points[np.isfinite(points).all(axis=1)]
        bounds_min = np.min(finite_points, axis=0)
        bounds_max = np.max(finite_points, axis=0)
        summary = {
            "schema": "noesis.mapanything.phone_scan.outputs.v1",
            "model": {
                "id": settings.model_id,
                "device": settings.device,
                "amp_dtype": settings.amp_dtype,
                "memory_efficient_inference": True,
                "minibatch_size": 1,
                "apply_mask": True,
                "mask_edges": True,
                "use_multiview_confidence": False,
            },
            "view_count": len(frame_results),
            "phone_view_count": len(frame_rows),
            "anchor_view_index": anchor_view_index,
            "review_point_count": int(points.shape[0]),
            "coordinate_frame": "mapanything_metric_world_unaligned_to_noesis",
            "pose_convention": "opencv_cam2world_x_right_y_down_z_forward",
            "bounds": {
                "min": bounds_min.tolist(),
                "max": bounds_max.tolist(),
            },
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
            },
            "frames": frame_results,
        }
        manifest_path = output_dir / "scan_outputs_manifest.json"
        summary["artifacts"]["manifest"] = "outputs/scan_outputs_manifest.json"
        files = []
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path != manifest_path:
                files.append(
                    {
                        "path": f"outputs/{path.relative_to(output_dir).as_posix()}",
                        "size_bytes": int(path.stat().st_size),
                    }
                )
        manifest_row = {
            "path": "outputs/scan_outputs_manifest.json",
            "size_bytes": 0,
        }
        files.append(manifest_row)
        summary["files"] = files
        for _ in range(3):
            manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            actual_size = int(manifest_path.stat().st_size)
            if manifest_row["size_bytes"] == actual_size:
                break
            manifest_row["size_bytes"] = actual_size
        progress(1.0, "MapAnything reconstruction and review outputs are saved")
        return summary
    except MapAnythingScanError:
        raise
    except Exception as exc:
        if type(exc).__name__ == "OutOfMemoryError" and type(exc).__module__.startswith("torch"):
            raise MapAnythingScanError(
                "CUDA ran out of memory while processing this view set; the saved video and frames remain intact"
            ) from exc
        raise MapAnythingScanError(f"MapAnything failed: {type(exc).__name__}: {exc}") from exc
    finally:
        outputs = None
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
        model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


__all__ = [
    "MapAnythingScanError",
    "MapAnythingScanSettings",
    "run_mapanything_scan",
]
