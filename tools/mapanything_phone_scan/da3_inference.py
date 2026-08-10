from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .inference import (
    _prepare_fixed_camera_anchor,
    _write_confidence_preview,
    _write_depth_preview,
    _write_mask,
    _write_reconstruction_glb,
    _write_rgb,
    _write_trajectory_preview,
)


ProgressCallback = Callable[[float, str], None]


class DA3PhoneScanError(RuntimeError):
    """Raised when the DA3 any-view phone reconstruction cannot be completed."""


@dataclass(frozen=True)
class DA3PhoneScanSettings:
    model_id: str = "depth-anything/DA3-BASE"
    device: str = "cuda:0"
    process_res: int = 504
    ref_view_strategy: str = "middle"
    point_budget: int = 600_000
    local_files_only: bool = True
    metric_engine_path: Path = Path(
        "data/ds9_artifacts/models/engines/"
        "da3metric_large_294x518_b3_fp16_trt10.13.engine"
    )
    metric_focal_denominator: float = 300.0
    anchor_image: Path | None = None


def _cuda_result(result: Any, operation: str) -> Any:
    from cuda.bindings import runtime as cudart

    if not isinstance(result, tuple) or not result:
        raise DA3PhoneScanError(f"{operation} returned an invalid CUDA result")
    if result[0] != cudart.cudaError_t.cudaSuccess:
        raise DA3PhoneScanError(f"{operation} failed with CUDA status {result[0]}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


class _TensorRTRunner:
    def __init__(self, engine_path: Path) -> None:
        from cuda.bindings import runtime as cudart
        import tensorrt as trt

        if not engine_path.is_file():
            raise DA3PhoneScanError(
                "DA3Metric-Large TensorRT engine is missing; set "
                f"NOESIS_PHONE_SCAN_DA3_ENGINE to the validated engine: {engine_path}"
            )
        self._cudart = cudart
        self._trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise DA3PhoneScanError(f"cannot deserialize TensorRT engine {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise DA3PhoneScanError("cannot create DA3Metric-Large TensorRT context")
        self.stream = _cuda_result(cudart.cudaStreamCreate(), "cudaStreamCreate")
        self.host: dict[str, np.ndarray] = {}
        self.device: dict[str, Any] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.input_name = ""
        self.output_names: list[str] = []
        for index in range(self.engine.num_io_tensors):
            name = str(self.engine.get_tensor_name(index))
            shape = tuple(int(value) for value in self.engine.get_tensor_shape(name))
            if not shape or any(value <= 0 for value in shape):
                raise DA3PhoneScanError(f"TensorRT binding {name} is not static: {shape}")
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            host = np.empty(shape, dtype=dtype, order="C")
            device = _cuda_result(cudart.cudaMalloc(host.nbytes), f"cudaMalloc({name})")
            if not self.context.set_tensor_address(name, int(device)):
                raise DA3PhoneScanError(f"cannot bind TensorRT tensor {name}")
            self.host[name] = host
            self.device[name] = device
            self.shapes[name] = shape
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if self.input_name:
                    raise DA3PhoneScanError("DA3 metric engine has multiple inputs")
                self.input_name = name
            else:
                self.output_names.append(name)
        if self.input_name != "images" or self.output_names != ["depth", "conf", "mask"]:
            raise DA3PhoneScanError(
                f"unexpected DA3 metric bindings: input={self.input_name}, outputs={self.output_names}"
            )

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.shapes[self.input_name]

    def infer(self, batch: np.ndarray) -> dict[str, np.ndarray]:
        cudart = self._cudart
        expected = self.host[self.input_name]
        value = np.ascontiguousarray(batch, dtype=expected.dtype)
        if value.shape != expected.shape:
            raise DA3PhoneScanError(
                f"DA3 metric input shape mismatch: expected {expected.shape}, got {value.shape}"
            )
        np.copyto(expected, value, casting="no")
        _cuda_result(
            cudart.cudaMemcpyAsync(
                int(self.device[self.input_name]),
                int(expected.ctypes.data),
                expected.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            ),
            "cudaMemcpyAsync(H2D)",
        )
        if not self.context.execute_async_v3(stream_handle=int(self.stream)):
            raise DA3PhoneScanError("DA3 metric TensorRT execution failed")
        for name in self.output_names:
            host = self.host[name]
            _cuda_result(
                cudart.cudaMemcpyAsync(
                    int(host.ctypes.data),
                    int(self.device[name]),
                    host.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                ),
                f"cudaMemcpyAsync({name})",
            )
        _cuda_result(cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
        return {name: self.host[name].copy() for name in self.output_names}

    def close(self) -> None:
        for pointer in self.device.values():
            _cuda_result(self._cudart.cudaFree(pointer), "cudaFree")
        self.device.clear()
        if self.stream is not None:
            _cuda_result(self._cudart.cudaStreamDestroy(self.stream), "cudaStreamDestroy")
            self.stream = None

    def __enter__(self) -> "_TensorRTRunner":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _metric_input(image_rgb: np.ndarray, model_hw: tuple[int, int]) -> tuple[np.ndarray, dict[str, int | float]]:
    model_height, model_width = model_hw
    source_height, source_width = image_rgb.shape[:2]
    scale = min(model_width / source_width, model_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        image_rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    canvas = np.zeros((model_height, model_width, 3), dtype=np.uint8)
    left = (model_width - resized_width) // 2
    top = (model_height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    chw = np.ascontiguousarray(np.transpose(canvas.astype(np.float32), (2, 0, 1)))
    return chw, {
        "scale": float(scale),
        "left": int(left),
        "top": int(top),
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
        "source_width": int(source_width),
        "source_height": int(source_height),
    }


def _restore_metric_map(value: np.ndarray, transform: dict[str, int | float]) -> np.ndarray:
    top = int(transform["top"])
    left = int(transform["left"])
    resized_height = int(transform["resized_height"])
    resized_width = int(transform["resized_width"])
    source_height = int(transform["source_height"])
    source_width = int(transform["source_width"])
    crop = np.asarray(value)[top : top + resized_height, left : left + resized_width]
    return cv2.resize(crop, (source_width, source_height), interpolation=cv2.INTER_LINEAR)


def _run_metric_branch(
    images_rgb: np.ndarray,
    intrinsics: np.ndarray,
    engine_path: Path,
    focal_denominator: float,
    progress: ProgressCallback,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    count = images_rgb.shape[0]
    raw_depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    transforms: list[dict[str, int | float]] = []
    with _TensorRTRunner(engine_path) as runner:
        batch_size, channels, model_height, model_width = runner.input_shape
        if (batch_size, channels, model_height, model_width) != (3, 3, 294, 518):
            raise DA3PhoneScanError(
                f"validated DA3 metric engine profile drifted: {runner.input_shape}"
            )
        prepared = []
        for image in images_rgb:
            tensor, transform = _metric_input(image, (model_height, model_width))
            prepared.append(tensor)
            transforms.append(transform)
        for start in range(0, count, batch_size):
            rows = prepared[start : start + batch_size]
            actual_count = len(rows)
            while len(rows) < batch_size:
                rows.append(rows[-1])
            outputs = runner.infer(np.stack(rows))
            for offset in range(actual_count):
                index = start + offset
                raw = _restore_metric_map(outputs["depth"][offset, 0], transforms[index])
                non_sky = _restore_metric_map(outputs["mask"][offset, 0], transforms[index]) >= 0.5
                focal_in_metric_input = (
                    float(intrinsics[index, 0, 0] + intrinsics[index, 1, 1])
                    * 0.5
                    * float(transforms[index]["scale"])
                )
                raw_depths.append(
                    (raw * focal_in_metric_input / float(focal_denominator)).astype(np.float32)
                )
                masks.append(non_sky)
            progress(
                0.58 + 0.14 * min(1.0, (start + actual_count) / count),
                f"Metricizing DA3 views {start + 1}-{start + actual_count} of {count}",
            )
    return np.stack(raw_depths), np.stack(masks), {
        "engine": str(engine_path),
        "profile": [3, 3, 294, 518],
        "focal_denominator": float(focal_denominator),
        "confidence_semantics": "official non-sky binary validity; not learned confidence",
    }


def _world_points(depth: np.ndarray, intrinsics: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack((uu, vv, np.ones_like(uu)), axis=-1).reshape(-1, 3)
    rays = (np.linalg.inv(intrinsics) @ pixels.T).T
    camera = rays * depth.reshape(-1, 1)
    world = (camera_to_world[:3, :3] @ camera.T).T + camera_to_world[:3, 3]
    return world.reshape(height, width, 3).astype(np.float32)


def _files(output_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": f"outputs/{path.relative_to(output_dir).as_posix()}",
            "size_bytes": int(path.stat().st_size),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    ]


def run_da3_phone_scan(
    scan_dir: Path,
    output_dir: Path,
    prepared: dict[str, Any],
    settings: DA3PhoneScanSettings,
    progress: ProgressCallback,
) -> dict[str, Any]:
    if settings.model_id != "depth-anything/DA3-BASE":
        raise DA3PhoneScanError(
            "phone any-view mode permits only the Apache-2.0 DA3-BASE checkpoint"
        )
    frame_rows = prepared.get("frames")
    if not isinstance(frame_rows, list) or len(frame_rows) < 2:
        raise DA3PhoneScanError("the scan has no prepared multi-view frame set")
    phone_frame_paths = [scan_dir / str(row["frame"]) for row in frame_rows]
    inference_rows = list(frame_rows)
    anchor_view_index: int | None = None
    if settings.anchor_image is not None:
        anchor_path = settings.anchor_image.resolve()
        if not anchor_path.is_file():
            raise DA3PhoneScanError(f"fixed-camera anchor image is missing: {anchor_path}")
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
    if any(not path.is_file() for path in frame_paths):
        raise DA3PhoneScanError("one or more prepared phone frames are missing")

    progress(0.02, "Loading official DA3-BASE any-view model")
    prediction = None
    model = None
    started = time.perf_counter()
    try:
        import torch
        from depth_anything_3.api import DepthAnything3

        model = DepthAnything3.from_pretrained(
            settings.model_id,
            local_files_only=settings.local_files_only,
        ).to(settings.device)
        progress(0.10, f"Jointly reconstructing {len(frame_paths)} phone views with DA3")
        prediction = model.inference(
            [str(path) for path in frame_paths],
            process_res=settings.process_res,
            process_res_method="upper_bound_resize",
            ref_view_strategy=settings.ref_view_strategy,
        )
        if prediction.extrinsics is None or prediction.intrinsics is None or prediction.conf is None:
            raise DA3PhoneScanError("DA3 any-view output omitted poses, intrinsics, or confidence")
        depth_relative = np.asarray(prediction.depth, dtype=np.float32)
        confidence = np.asarray(prediction.conf, dtype=np.float32)
        extrinsics = np.asarray(prediction.extrinsics, dtype=np.float64)
        intrinsics = np.asarray(prediction.intrinsics, dtype=np.float64)
        images_rgb = np.asarray(prediction.processed_images, dtype=np.uint8)
    finally:
        del model
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    progress(0.56, "Loading validated DA3Metric-Large FP16 TensorRT engine")
    metric_depth, non_sky, metric_meta = _run_metric_branch(
        images_rgb,
        intrinsics,
        settings.metric_engine_path,
        settings.metric_focal_denominator,
        progress,
    )
    finite = (
        np.isfinite(depth_relative)
        & (depth_relative > 1e-3)
        & np.isfinite(metric_depth)
        & (metric_depth > 1e-2)
        & np.isfinite(confidence)
        & non_sky
    )
    confidence_median = float(np.median(confidence[finite]))
    alignment_mask = finite & (confidence >= confidence_median)
    if int(np.count_nonzero(alignment_mask)) < 1_000:
        raise DA3PhoneScanError("too few valid pixels to metricize DA3 any-view depth")
    relative_values = depth_relative[alignment_mask].astype(np.float64)
    metric_values = metric_depth[alignment_mask].astype(np.float64)
    scale = float(np.dot(metric_values, relative_values) / np.dot(relative_values, relative_values))
    if not math.isfinite(scale) or scale <= 0:
        raise DA3PhoneScanError(f"invalid DA3 metric alignment scale {scale}")
    depth = depth_relative * scale
    extrinsics = extrinsics.copy()
    extrinsics[:, :3, 3] *= scale
    camera_to_world = np.linalg.inv(
        np.concatenate(
            [extrinsics, np.tile(np.array([[[0.0, 0.0, 0.0, 1.0]]]), (extrinsics.shape[0], 1, 1))],
            axis=1,
        )
        if extrinsics.shape[-2:] == (3, 4)
        else extrinsics
    )

    views_dir = output_dir / "views"
    raw_dir = output_dir / "raw"
    views_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    review_points: list[np.ndarray] = []
    review_colors: list[np.ndarray] = []
    frame_outputs: list[dict[str, Any]] = []
    per_view_budget = max(1_000, settings.point_budget // len(frame_paths))
    progress(0.74, "Saving metric DA3 phone reconstruction")
    for index, row in enumerate(inference_rows):
        mask = non_sky[index] & np.isfinite(depth[index]) & (depth[index] > 0)
        points = _world_points(depth[index], intrinsics[index], camera_to_world[index])
        rgb_path = views_dir / f"view_{index:04d}_rgb.png"
        depth_path = views_dir / f"view_{index:04d}_depth.png"
        confidence_path = views_dir / f"view_{index:04d}_confidence.png"
        mask_path = views_dir / f"view_{index:04d}_mask.png"
        raw_path = raw_dir / f"view_{index:04d}.npz"
        _write_rgb(rgb_path, images_rgb[index])
        depth_stats = _write_depth_preview(depth_path, depth[index], mask)
        confidence_stats = _write_confidence_preview(confidence_path, confidence[index], mask)
        _write_mask(mask_path, mask)
        np.savez_compressed(
            raw_path,
            world_points=points,
            depth_z=depth[index].astype(np.float32),
            confidence=confidence[index].astype(np.float32),
            mask=mask.astype(np.uint8),
            camera_pose=camera_to_world[index].astype(np.float32),
            intrinsics=intrinsics[index].astype(np.float32),
            metric_scaling_factor=np.asarray([scale], dtype=np.float32),
            model_rgb=images_rgb[index],
        )
        valid_indices = np.flatnonzero(mask.reshape(-1))
        if valid_indices.size:
            threshold = float(np.percentile(confidence[index][mask], 55.0))
            selected = np.flatnonzero((mask & (confidence[index] >= threshold)).reshape(-1))
            stride = max(1, int(math.ceil(selected.size / per_view_budget)))
            selected = selected[::stride][:per_view_budget]
            review_points.append(points.reshape(-1, 3)[selected])
            review_colors.append(images_rgb[index].reshape(-1, 3)[selected])
        frame_outputs.append(
            {
                "index": index,
                "timestamp_s": float(row.get("timestamp_s") or 0.0),
                "source_frame": (
                    None if row.get("fixed_camera_anchor") else str(row["frame"])
                ),
                "fixed_camera_anchor": bool(row.get("fixed_camera_anchor")),
                "model_rgb": f"outputs/views/{rgb_path.name}",
                "depth_preview": f"outputs/views/{depth_path.name}",
                "confidence_preview": f"outputs/views/{confidence_path.name}",
                "mask_preview": f"outputs/views/{mask_path.name}",
                "raw_npz": f"outputs/raw/{raw_path.name}",
                "depth": depth_stats,
                "confidence": confidence_stats,
            }
        )
        progress(0.74 + 0.18 * (index + 1) / len(inference_rows), f"Saved DA3 view {index + 1} of {len(inference_rows)}")

    all_points = np.concatenate(review_points)
    all_colors = np.concatenate(review_colors)
    camera_positions = camera_to_world[:, :3, 3]
    reconstruction_path = output_dir / "reconstruction_points.glb"
    trajectory_preview_path = output_dir / "camera_trajectory_topdown.png"
    trajectory_json_path = output_dir / "camera_trajectory.json"
    camera_solution_path = output_dir / "camera_solution.npz"
    manifest_path = output_dir / "scan_outputs_manifest.json"
    _write_reconstruction_glb(reconstruction_path, all_points, all_colors, camera_positions)
    _write_trajectory_preview(trajectory_preview_path, all_points, camera_positions)
    trajectory_json_path.write_text(
        json.dumps(
            {
                "schema": "noesis.da3.phone_scan.camera_trajectory.v1",
                "provider": "da3",
                "model_id": settings.model_id,
                "camera_to_world": camera_to_world.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        camera_solution_path,
        camera_poses=camera_to_world.astype(np.float32),
        intrinsics=intrinsics.astype(np.float32),
        metric_scaling_factors=np.full(len(inference_rows), scale, dtype=np.float32),
    )
    result: dict[str, Any] = {
        "schema": "noesis.phone_scan.outputs.v2",
        "provider": "da3",
        "model_id": settings.model_id,
        "mode": "DA3-BASE any-view poses and consistent depth metricized by DA3Metric-Large TensorRT",
        "view_count": len(inference_rows),
        "phone_view_count": len(frame_rows),
        "anchor_view_index": anchor_view_index,
        "review_point_count": int(all_points.shape[0]),
        "scale": {"median": scale, "global": scale},
        "metric_branch": metric_meta,
        "confidence_semantics": "DA3-BASE learned any-view depth confidence",
        "validity_semantics": "DA3Metric-Large official non-sky binary mask",
        "elapsed_s": float(time.perf_counter() - started),
        "artifacts": {
            "reconstruction_glb": "outputs/reconstruction_points.glb",
            "trajectory_preview": "outputs/camera_trajectory_topdown.png",
            "trajectory_json": "outputs/camera_trajectory.json",
            "camera_solution_npz": "outputs/camera_solution.npz",
            "manifest": "outputs/scan_outputs_manifest.json",
        },
        "frames": frame_outputs,
    }
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["files"] = _files(output_dir)
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    progress(1.0, "DA3 phone reconstruction complete")
    return result


__all__ = ["DA3PhoneScanError", "DA3PhoneScanSettings", "run_da3_phone_scan"]
