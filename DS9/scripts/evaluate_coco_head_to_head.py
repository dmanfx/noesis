#!/usr/bin/env python3
"""Evaluate exact DS9 TensorRT engines on COCO 2017 person tasks.

This is a local accuracy evaluator, not a published-checkpoint score importer.
It runs static-batch-3 TensorRT engines, applies the reviewed model-output
quality preprocessing and postprocessing lane, and evaluates the resulting
person predictions with pycocotools. This is intentionally distinct from
byte-exact DeepStream preprocessing and deployed parser/clustering behavior.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import cv2
import numpy as np
from cuda.bindings import runtime as cudart
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.special import logsumexp

try:
    import tensorrt as trt
except ImportError as exc:  # pragma: no cover - environment failure
    raise SystemExit(
        "TensorRT Python bindings are required. Run this evaluator with the "
        "TensorRT 10.16.0.72 toolchain used to build the DS9 engines."
    ) from exc


BATCH_SIZE = 3
PERSON_CATEGORY_ID = 1
RF_RELEASE = "1.8.3"
RF_FP16_PROFILE = "fp16_tf32"
RF_FP32_PROFILE = "fp32_no_tf32"
RF_MATRIX_SCHEMA = "noesis.ds9.rfdetr-model-matrix.v1"
YOLO_MATRIX_SCHEMA = "noesis.ds9.yolo26-performance-model-matrix.v2"
YOLO_PERFORMANCE_SCHEMA = "noesis.ds9.yolo26-performance-benchmark.v2"
REPORT_SCHEMA = "noesis.ds9.local-coco-person-evaluation.v2"
EXPECTED_TRT_VERSION = "10.16.0.72"
DEFAULT_SCORE_FLOOR = 0.001
DEFAULT_TOPK = 100
COCO_STAT_NAMES = {
    "bbox": (
        "ap",
        "ap50",
        "ap75",
        "ap_small",
        "ap_medium",
        "ap_large",
        "ar_1",
        "ar_10",
        "ar_100",
        "ar_small",
        "ar_medium",
        "ar_large",
    ),
    "segm": (
        "ap",
        "ap50",
        "ap75",
        "ap_small",
        "ap_medium",
        "ap_large",
        "ar_1",
        "ar_10",
        "ar_100",
        "ar_small",
        "ar_medium",
        "ar_large",
    ),
    "keypoints": (
        "ap",
        "ap50",
        "ap75",
        "ap_medium",
        "ap_large",
        "ar",
        "ar50",
        "ar75",
        "ar_medium",
        "ar_large",
    ),
}


class EvaluationError(RuntimeError):
    """Raised when an evaluation contract is invalid."""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    source_id: str
    architecture: str
    family: str
    variant: str
    precision_profile: str
    engine: str
    engine_receipt: str | None
    expected_engine_sha256: str | None
    expected_input_name: str
    expected_outputs: Mapping[str, tuple[int, ...]]
    resolution: int
    preprocess: str
    interpolation: str
    operating_score_threshold: float
    mask_threshold: float | None
    keypoint_threshold: float | None


@dataclass(frozen=True)
class ImageTransform:
    source_width: int
    source_height: int
    model_width: int
    model_height: int
    gain_x: float
    gain_y: float
    pad_x: float
    pad_y: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


class JsonArrayWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        self._descriptor = os.open(path, flags, 0o600)
        self._handle = os.fdopen(self._descriptor, "w", encoding="utf-8")
        self._descriptor = -1
        self._first = True
        self.count = 0
        self._handle.write("[")

    def write(self, row: Mapping[str, Any]) -> None:
        if not self._first:
            self._handle.write(",")
        self._first = False
        json.dump(dict(row), self._handle, separators=(",", ":"), allow_nan=False)
        self.count += 1

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle.write("]\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._handle = None

    def __enter__(self) -> "JsonArrayWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
            return
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _cuda_result(result: Any, operation: str) -> Any:
    if not isinstance(result, tuple) or not result:
        raise EvaluationError(f"{operation} returned an invalid CUDA result")
    status = result[0]
    if status != cudart.cudaError_t.cudaSuccess:
        raise EvaluationError(f"{operation} failed with CUDA status {status}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


class TensorRTRunner:
    def __init__(self, engine_path: Path) -> None:
        if not engine_path.is_file():
            raise EvaluationError(f"TensorRT engine is missing: {engine_path}")
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise EvaluationError(f"cannot deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise EvaluationError(f"cannot create execution context: {engine_path}")
        self.stream = _cuda_result(cudart.cudaStreamCreate(), "cudaStreamCreate")
        self.input_name = ""
        self.host: dict[str, np.ndarray] = {}
        self.device: dict[str, Any] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.dtypes: dict[str, np.dtype[Any]] = {}
        self.output_names: list[str] = []
        for index in range(self.engine.num_io_tensors):
            name = str(self.engine.get_tensor_name(index))
            shape = tuple(int(value) for value in self.engine.get_tensor_shape(name))
            if not shape or any(value <= 0 for value in shape):
                raise EvaluationError(f"{name} is not a static positive tensor: {shape}")
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            host = np.empty(shape, dtype=dtype, order="C")
            device = _cuda_result(cudart.cudaMalloc(host.nbytes), f"cudaMalloc({name})")
            if not self.context.set_tensor_address(name, int(device)):
                raise EvaluationError(f"cannot bind TensorRT tensor {name}")
            self.host[name] = host
            self.device[name] = device
            self.shapes[name] = shape
            self.dtypes[name] = dtype
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                if self.input_name:
                    raise EvaluationError("evaluator requires exactly one input tensor")
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)
            else:
                raise EvaluationError(f"unsupported TensorRT I/O mode for {name}: {mode}")
        if not self.input_name or not self.output_names:
            raise EvaluationError("engine must expose one input and at least one output")

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.shapes[self.input_name]

    def infer(self, batch: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
        expected = self.host[self.input_name]
        value = np.ascontiguousarray(batch, dtype=expected.dtype)
        if value.shape != expected.shape:
            raise EvaluationError(
                f"input shape mismatch: expected {expected.shape}, found {value.shape}"
            )
        np.copyto(expected, value, casting="no")
        started = time.perf_counter()
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
            raise EvaluationError("TensorRT execute_async_v3 returned false")
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
                f"cudaMemcpyAsync(D2H:{name})",
            )
        _cuda_result(cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {name: self.host[name].copy() for name in self.output_names}, elapsed_ms

    def close(self) -> None:
        for pointer in self.device.values():
            _cuda_result(cudart.cudaFree(pointer), "cudaFree")
        self.device.clear()
        if self.stream is not None:
            _cuda_result(cudart.cudaStreamDestroy(self.stream), "cudaStreamDestroy")
            self.stream = None
        self.context = None
        self.engine = None
        self.runtime = None

    def __enter__(self) -> "TensorRTRunner":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read JSON {path}: {exc}") from exc


def _rf_specs(args: argparse.Namespace, profile: str) -> list[ModelSpec]:
    matrix_path = Path(args.rf_matrix).resolve()
    matrix = _load_json(matrix_path)
    if matrix.get("schema") != RF_MATRIX_SCHEMA:
        raise EvaluationError(f"unexpected RF model matrix schema: {matrix_path}")
    root = Path(args.artifact_root).resolve()
    engine_root = root / "models" / "engines" / "rfdetr" / RF_RELEASE / "runtime" / profile
    receipt_root = (
        root / "models" / "provenance" / "rfdetr" / RF_RELEASE / "runtime" / profile
    )
    specs = []
    for row in matrix.get("models", []):
        if row.get("enabled") is not True:
            continue
        runtime = row.get("runtime") or {}
        onnx_name = str(runtime.get("onnx_filename") or "")
        if not onnx_name.endswith(".onnx"):
            raise EvaluationError(f"{row.get('id')} lacks a runtime ONNX filename")
        engine_name = f"{Path(onnx_name).stem}_{profile}.engine"
        family = str(row["family"])
        outputs = {
            str(name): tuple(int(value) for value in shape)
            for name, shape in (row.get("outputs") or {}).items()
        }
        if not outputs:
            raise EvaluationError(f"{row.get('id')} lacks an output contract")
        specs.append(
            ModelSpec(
                id=f"rfdetr_{profile}_{row['id']}",
                source_id=str(row["id"]),
                architecture="rfdetr",
                family=family,
                variant=str(row["variant"]),
                precision_profile=profile,
                engine=str((engine_root / engine_name).resolve()),
                engine_receipt=str((receipt_root / f"{row['id']}.engine.json").resolve()),
                expected_engine_sha256=None,
                expected_input_name="input",
                expected_outputs=outputs,
                resolution=int(row["resolution"]),
                preprocess="rgb01_direct_square_bilinear",
                interpolation="bilinear",
                operating_score_threshold=0.4,
                mask_threshold=0.5 if family == "segmentation" else None,
                keypoint_threshold=0.35 if family == "keypoint" else None,
            )
        )
    return specs


def _yolo_specs(args: argparse.Namespace) -> list[ModelSpec]:
    matrix_path = Path(args.yolo_matrix).resolve()
    matrix = _load_json(matrix_path)
    schema = str(matrix.get("schema") or "")
    if schema != YOLO_MATRIX_SCHEMA:
        raise EvaluationError(f"unexpected YOLO model matrix schema: {schema}")
    if int(matrix.get("batch_size") or 0) != BATCH_SIZE:
        raise EvaluationError(f"unexpected YOLO model matrix batch size: {matrix_path}")
    performance_path = (
        Path(args.yolo_performance_report).resolve()
        if str(args.yolo_performance_report).strip()
        else matrix_path.parent / "performance_report.json"
    )
    performance = _load_json(performance_path)
    if (
        performance.get("schema") != YOLO_PERFORMANCE_SCHEMA
        or performance.get("status") != "passed"
        or int(performance.get("batch_size") or 0) != BATCH_SIZE
    ):
        raise EvaluationError(
            f"unexpected YOLO performance report contract: {performance_path}"
        )
    performance_rows = {
        str(row.get("model_id")): row
        for row in performance.get("models", [])
        if isinstance(row, Mapping)
    }
    root = Path(args.artifact_root).resolve()
    specs = []
    for row in matrix.get("models", []):
        model_id = str(row["id"])
        performance_row = performance_rows.get(model_id)
        if not isinstance(performance_row, Mapping):
            raise EvaluationError(
                f"{model_id} is missing from YOLO performance evidence"
            )
        engine_sha256 = str(performance_row.get("engine_sha256") or "")
        if len(engine_sha256) != 64:
            raise EvaluationError(f"{model_id} lacks a YOLO engine SHA-256")
        matrix_sha256 = str(row.get("engine_sha256") or "")
        if matrix_sha256 and matrix_sha256 != engine_sha256:
            raise EvaluationError(f"{model_id} YOLO engine identity drifted")
        outputs = {
            str(name): tuple(int(value) for value in shape)
            for name, shape in (row.get("outputs") or {}).items()
        }
        performance_outputs = {
            str(name): tuple(int(value) for value in shape)
            for name, shape in (performance_row.get("outputs") or {}).items()
        }
        if not outputs or outputs != performance_outputs:
            raise EvaluationError(f"{model_id} YOLO output contract drifted")
        family = str(row["family"])
        relative = Path(str(row["engine_relative_path"]))
        engine = (root / relative).resolve()
        if family == "detection":
            threshold = 0.25
            interpolation = "nearest"
        elif family == "segmentation":
            threshold = 0.35
            interpolation = "nearest"
        elif family == "keypoint":
            threshold = 0.25
            interpolation = "bilinear"
        else:
            raise EvaluationError(f"unsupported YOLO family: {family}")
        specs.append(
            ModelSpec(
                id=f"yolo26_fp16_tf32_{row['id']}",
                source_id=str(row["id"]),
                architecture="yolo26",
                family=family,
                variant=str(row["variant"]),
                precision_profile="fp16_tf32",
                engine=str(engine),
                engine_receipt=None,
                expected_engine_sha256=engine_sha256,
                expected_input_name="images",
                expected_outputs=outputs,
                resolution=int(row["resolution"]),
                preprocess=f"rgb01_zero_symmetric_letterbox_{interpolation}",
                interpolation=interpolation,
                operating_score_threshold=threshold,
                mask_threshold=0.5 if family == "segmentation" else None,
                keypoint_threshold=0.35 if family == "keypoint" else None,
            )
        )
    return specs


def _select_specs(args: argparse.Namespace) -> list[ModelSpec]:
    if args.suite == "rf_fp16":
        specs = _rf_specs(args, RF_FP16_PROFILE)
    elif args.suite == "rf_fp32":
        specs = _rf_specs(args, RF_FP32_PROFILE)
    elif args.suite == "yolo_fp16":
        specs = _yolo_specs(args)
    else:  # pragma: no cover - argparse protects this
        raise EvaluationError(f"unsupported suite: {args.suite}")
    selected = {str(value) for value in args.model}
    if selected:
        specs = [
            spec
            for spec in specs
            if spec.id in selected or spec.source_id in selected
        ]
        found = {spec.id for spec in specs} | {spec.source_id for spec in specs}
        missing = sorted(selected - found)
        if missing:
            raise EvaluationError(f"unknown selected models: {', '.join(missing)}")
    if not specs:
        raise EvaluationError("model selection is empty")
    return specs


def _image_batches(rows: Sequence[Mapping[str, Any]]) -> Iterator[list[Mapping[str, Any]]]:
    for start in range(0, len(rows), BATCH_SIZE):
        yield list(rows[start : start + BATCH_SIZE])


def _direct_square(
    image_bgr: np.ndarray, resolution: int
) -> tuple[np.ndarray, ImageTransform]:
    height, width = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32).transpose(2, 0, 1) / np.float32(255.0)
    return np.ascontiguousarray(tensor), ImageTransform(
        source_width=width,
        source_height=height,
        model_width=resolution,
        model_height=resolution,
        gain_x=resolution / width,
        gain_y=resolution / height,
        pad_x=0.0,
        pad_y=0.0,
    )


def _letterbox(
    image_bgr: np.ndarray, resolution: int, interpolation: str
) -> tuple[np.ndarray, ImageTransform]:
    height, width = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gain = min(resolution / width, resolution / height)
    resized_width = max(1, min(resolution, int(round(width * gain))))
    resized_height = max(1, min(resolution, int(round(height * gain))))
    interpolation_flag = (
        cv2.INTER_NEAREST if interpolation == "nearest" else cv2.INTER_LINEAR
    )
    resized = cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=interpolation_flag,
    )
    horizontal = resolution - resized_width
    vertical = resolution - resized_height
    left = int(round(horizontal / 2.0 - 0.1))
    top = int(round(vertical / 2.0 - 0.1))
    canvas = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    tensor = canvas.astype(np.float32).transpose(2, 0, 1) / np.float32(255.0)
    return np.ascontiguousarray(tensor), ImageTransform(
        source_width=width,
        source_height=height,
        model_width=resolution,
        model_height=resolution,
        gain_x=resized_width / width,
        gain_y=resized_height / height,
        pad_x=float(left),
        pad_y=float(top),
    )


def _prepare_image(
    path: Path, spec: ModelSpec
) -> tuple[np.ndarray, ImageTransform]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise EvaluationError(f"cannot decode COCO image: {path}")
    if spec.architecture == "rfdetr":
        return _direct_square(image, spec.resolution)
    return _letterbox(image, spec.resolution, spec.interpolation)


def _sigmoid(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(array, -80.0, 80.0)))


def _clip_xyxy(
    box: Sequence[float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    x1 = min(max(float(box[0]), 0.0), float(width))
    y1 = min(max(float(box[1]), 0.0), float(height))
    x2 = min(max(float(box[2]), 0.0), float(width))
    y2 = min(max(float(box[3]), 0.0), float(height))
    if x2 - x1 <= 0.0 or y2 - y1 <= 0.0:
        return None
    return x1, y1, x2, y2


def _rf_box(box: Sequence[float], transform: ImageTransform) -> tuple[float, ...] | None:
    cx, cy, width, height = [float(value) for value in box]
    if not all(math.isfinite(value) for value in (cx, cy, width, height)):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return _clip_xyxy(
        (
            (cx - width / 2.0) * transform.source_width,
            (cy - height / 2.0) * transform.source_height,
            (cx + width / 2.0) * transform.source_width,
            (cy + height / 2.0) * transform.source_height,
        ),
        transform.source_width,
        transform.source_height,
    )


def _yolo_box(
    box: Sequence[float], transform: ImageTransform, *, normalized: bool
) -> tuple[float, ...] | None:
    values = np.asarray(box, dtype=np.float64).copy()
    if not np.isfinite(values).all():
        return None
    if normalized:
        values[[0, 2]] *= transform.model_width
        values[[1, 3]] *= transform.model_height
    values[[0, 2]] = (values[[0, 2]] - transform.pad_x) / transform.gain_x
    values[[1, 3]] = (values[[1, 3]] - transform.pad_y) / transform.gain_y
    return _clip_xyxy(
        values,
        transform.source_width,
        transform.source_height,
    )


def _bbox_result(
    image_id: int, box: Sequence[float], score: float
) -> dict[str, Any]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return {
        "image_id": int(image_id),
        "category_id": PERSON_CATEGORY_ID,
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "score": float(score),
    }


def _metric_result(row: Mapping[str, Any], metric: str) -> dict[str, Any]:
    result = {
        "image_id": int(row["image_id"]),
        "category_id": int(row["category_id"]),
        "score": float(row["score"]),
    }
    if metric == "bbox":
        result["bbox"] = list(row["bbox"])
    elif metric == "segm":
        result["segmentation"] = dict(row["segmentation"])
    elif metric == "keypoints":
        result["keypoints"] = list(row["keypoints"])
    else:
        raise EvaluationError(f"unsupported prediction metric: {metric}")
    return result


def _rle(binary_mask: np.ndarray) -> dict[str, Any]:
    encoded = mask_utils.encode(
        np.asfortranarray(binary_mask.astype(np.uint8, copy=False))
    )
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(value) for value in encoded["size"]], "counts": counts}


def _rf_fused_keypoint_scores(
    labels: np.ndarray, keypoints: np.ndarray
) -> np.ndarray:
    active = np.asarray(keypoints[:, 17:34, :], dtype=np.float64)
    log_l11 = active[..., 4]
    l21 = active[..., 5]
    log_l22 = active[..., 6]
    findable = _sigmoid(active[..., 2])
    log_t1 = -2.0 * log_l11
    log_t2 = -2.0 * log_l22
    log_t3 = (
        2.0 * np.log(np.maximum(np.abs(l21), 1e-12))
        + log_t1
        + log_t2
    )
    log_trace = np.logaddexp(np.logaddexp(log_t1, log_t2), log_t3)
    log_weights = np.log(np.maximum(findable, 1e-12))
    log_mean_trace = logsumexp(log_trace + log_weights, axis=-1) - logsumexp(
        log_weights, axis=-1
    )
    return _sigmoid(labels[:, 1]) * np.exp(-0.2 * log_mean_trace)


def _top_indexes(scores: np.ndarray, score_floor: float, topk: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    indexes = np.flatnonzero(np.isfinite(values) & (values >= score_floor))
    if indexes.size == 0:
        return indexes
    order = np.argsort(-values[indexes], kind="stable")
    return indexes[order[:topk]]


def _postprocess_rf(
    spec: ModelSpec,
    image_id: int,
    transform: ImageTransform,
    outputs: Mapping[str, np.ndarray],
    batch_index: int,
    score_floor: float,
    topk: int,
) -> list[dict[str, Any]]:
    boxes = np.asarray(outputs["dets"][batch_index])
    labels = np.asarray(outputs["labels"][batch_index])
    if spec.family == "keypoint":
        raw_keypoints = np.asarray(outputs["keypoints"][batch_index])
        scores = _rf_fused_keypoint_scores(labels, raw_keypoints)
    else:
        scores = _sigmoid(labels[:, PERSON_CATEGORY_ID])
    results = []
    for query in _top_indexes(scores, score_floor, topk):
        box = _rf_box(boxes[query], transform)
        if box is None:
            continue
        row = _bbox_result(image_id, box, float(scores[query]))
        if spec.family == "segmentation":
            logits = np.asarray(outputs["masks"][batch_index, query], dtype=np.float32)
            resized = cv2.resize(
                logits,
                (transform.source_width, transform.source_height),
                interpolation=cv2.INTER_LINEAR,
            )
            row["segmentation"] = _rle(resized > 0.0)
        elif spec.family == "keypoint":
            active = raw_keypoints[query, 17:34]
            keypoints = []
            for point in active:
                keypoints.extend(
                    (
                        float(
                            min(
                                max(float(point[0]) * transform.source_width, 0.0),
                                float(transform.source_width),
                            )
                        ),
                        float(
                            min(
                                max(float(point[1]) * transform.source_height, 0.0),
                                float(transform.source_height),
                            )
                        ),
                        float(_sigmoid(point[2])),
                    )
                )
            row["keypoints"] = keypoints
        results.append(row)
    return results


def _paste_letterboxed_box_mask(
    mask: np.ndarray,
    network_box: Sequence[float],
    transform: ImageTransform,
) -> np.ndarray:
    """Project a bbox-relative network mask through the letterbox transform."""
    x1, y1, x2, y2 = [float(value) for value in network_box]
    output = np.zeros(
        (transform.source_height, transform.source_width),
        dtype=np.uint8,
    )
    if (
        not all(math.isfinite(value) for value in (x1, y1, x2, y2))
        or x2 <= x1
        or y2 <= y1
    ):
        return output
    content_x1 = transform.pad_x
    content_y1 = transform.pad_y
    content_x2 = transform.pad_x + transform.gain_x * transform.source_width
    content_y2 = transform.pad_y + transform.gain_y * transform.source_height
    visible_x1 = max(x1, content_x1, 0.0)
    visible_y1 = max(y1, content_y1, 0.0)
    visible_x2 = min(x2, content_x2, float(transform.model_width))
    visible_y2 = min(y2, content_y2, float(transform.model_height))
    if visible_x2 <= visible_x1 or visible_y2 <= visible_y1:
        return output
    source_x1 = (visible_x1 - transform.pad_x) / transform.gain_x
    source_y1 = (visible_y1 - transform.pad_y) / transform.gain_y
    source_x2 = (visible_x2 - transform.pad_x) / transform.gain_x
    source_y2 = (visible_y2 - transform.pad_y) / transform.gain_y
    left = max(0, min(transform.source_width, int(math.floor(source_x1))))
    top = max(0, min(transform.source_height, int(math.floor(source_y1))))
    right = max(0, min(transform.source_width, int(math.ceil(source_x2))))
    bottom = max(0, min(transform.source_height, int(math.ceil(source_y2))))
    if right <= left or bottom <= top:
        return output
    mask_values = np.asarray(mask, dtype=np.float32)
    if mask_values.ndim != 2 or not np.isfinite(mask_values).all():
        return output
    mask_height, mask_width = mask_values.shape
    source_x = np.arange(left, right, dtype=np.float32) + np.float32(0.5)
    source_y = np.arange(top, bottom, dtype=np.float32) + np.float32(0.5)
    network_x = source_x * np.float32(transform.gain_x) + np.float32(
        transform.pad_x
    )
    network_y = source_y * np.float32(transform.gain_y) + np.float32(
        transform.pad_y
    )
    map_x = (
        (network_x - np.float32(x1))
        * np.float32(mask_width / (x2 - x1))
        - np.float32(0.5)
    )
    map_y = (
        (network_y - np.float32(y1))
        * np.float32(mask_height / (y2 - y1))
        - np.float32(0.5)
    )
    grid_x, grid_y = np.meshgrid(map_x, map_y)
    projected = cv2.remap(
        mask_values,
        grid_x,
        grid_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    output[top:bottom, left:right] = (projected > 0.5).astype(np.uint8)
    return output


def _postprocess_yolo(
    spec: ModelSpec,
    image_id: int,
    transform: ImageTransform,
    outputs: Mapping[str, np.ndarray],
    batch_index: int,
    score_floor: float,
    topk: int,
) -> list[dict[str, Any]]:
    if "output0" not in outputs:
        raise EvaluationError(f"{spec.id} does not expose output0")
    rows = np.asarray(outputs["output0"][batch_index])
    if rows.ndim != 2 or rows.shape[1] < 6:
        raise EvaluationError(f"{spec.id} output0 shape is invalid: {rows.shape}")
    person = np.flatnonzero(
        np.isfinite(rows[:, 4])
        & (rows[:, 4] >= score_floor)
        & (np.rint(rows[:, 5]).astype(np.int64) == 0)
    )
    if person.size:
        person = person[np.argsort(-rows[person, 4], kind="stable")[:topk]]
    results = []
    for index in person:
        network_box = rows[index, :4].astype(np.float64, copy=True)
        box = _yolo_box(network_box, transform, normalized=False)
        if box is None:
            continue
        row = _bbox_result(image_id, box, float(rows[index, 4]))
        if spec.family == "segmentation":
            mask_values = rows[index, 6:]
            side = int(round(math.sqrt(mask_values.size)))
            if side * side != mask_values.size:
                raise EvaluationError(
                    f"{spec.id} fused mask is not square: {mask_values.size}"
                )
            binary = _paste_letterboxed_box_mask(
                mask_values.reshape(side, side),
                network_box,
                transform,
            )
            row["segmentation"] = _rle(binary)
        elif spec.family == "keypoint":
            values = rows[index, 6:]
            if values.size < 51:
                raise EvaluationError(f"{spec.id} lacks 17 keypoints")
            points = values[-51:].reshape(17, 3).astype(np.float64)
            points[:, 0] = (points[:, 0] - transform.pad_x) / transform.gain_x
            points[:, 1] = (points[:, 1] - transform.pad_y) / transform.gain_y
            points[:, 0] = np.clip(points[:, 0], 0.0, transform.source_width)
            points[:, 1] = np.clip(points[:, 1], 0.0, transform.source_height)
            row["keypoints"] = [float(value) for value in points.reshape(-1)]
        results.append(row)
    return results


def _postprocess(
    spec: ModelSpec,
    image_id: int,
    transform: ImageTransform,
    outputs: Mapping[str, np.ndarray],
    batch_index: int,
    score_floor: float,
    topk: int,
) -> list[dict[str, Any]]:
    if spec.architecture == "rfdetr":
        return _postprocess_rf(
            spec,
            image_id,
            transform,
            outputs,
            batch_index,
            score_floor,
            topk,
        )
    return _postprocess_yolo(
        spec,
        image_id,
        transform,
        outputs,
        batch_index,
        score_floor,
        topk,
    )


def _evaluate_predictions(
    annotation_path: Path,
    prediction_path: Path,
    image_ids: Sequence[int],
    metric: str,
) -> dict[str, Any]:
    if metric not in COCO_STAT_NAMES:
        raise EvaluationError(f"unsupported COCO metric: {metric}")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        ground_truth = COCO(str(annotation_path))
        prediction_is_empty = (
            prediction_path.stat().st_size <= 4
            and _load_json(prediction_path) == []
        )
        if prediction_is_empty:
            detections = COCO()
            detections.dataset = {
                "info": copy.deepcopy(ground_truth.dataset.get("info", {})),
                "images": copy.deepcopy(ground_truth.dataset["images"]),
                "categories": copy.deepcopy(ground_truth.dataset["categories"]),
                "annotations": [],
            }
            detections.createIndex()
        else:
            detections = ground_truth.loadRes(str(prediction_path))
        evaluator = COCOeval(ground_truth, detections, metric)
        evaluator.params.imgIds = [int(value) for value in image_ids]
        evaluator.params.catIds = [PERSON_CATEGORY_ID]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = [float(value) for value in evaluator.stats.tolist()]
    names = COCO_STAT_NAMES[metric]
    if len(stats) != len(names):
        raise EvaluationError(
            f"COCO {metric} returned {len(stats)} stats, expected {len(names)}"
        )
    return {
        "metric": metric,
        "person_category_id": PERSON_CATEGORY_ID,
        "image_count": len(image_ids),
        "stats": dict(zip(names, stats, strict=True)),
        "summary": output.getvalue(),
    }


def _engine_receipt(spec: ModelSpec, engine_sha256: str) -> dict[str, Any] | None:
    if spec.engine_receipt is None:
        return None
    receipt_path = Path(spec.engine_receipt)
    receipt = _load_json(receipt_path)
    recorded = str((receipt.get("engine") or {}).get("sha256") or "")
    if recorded != engine_sha256:
        raise EvaluationError(
            f"{spec.id} engine receipt hash mismatch: {recorded} != {engine_sha256}"
        )
    build = receipt.get("build_contract") or {}
    if str(build.get("profile") or "") != spec.precision_profile:
        raise EvaluationError(f"{spec.id} precision profile receipt mismatch")
    return {
        "path": str(receipt_path),
        "sha256": _sha256(receipt_path),
        "recorded_engine_sha256": recorded,
    }


def _evaluate_model(
    args: argparse.Namespace,
    spec: ModelSpec,
    image_rows: Sequence[Mapping[str, Any]],
    instance_annotation: Path,
    keypoint_annotation: Path,
) -> dict[str, Any]:
    model_dir = Path(args.run_dir).resolve() / "models" / spec.id
    model_dir.mkdir(parents=True, exist_ok=False)
    metric_names = ["bbox"]
    if spec.family == "segmentation":
        metric_names.append("segm")
    elif spec.family == "keypoint":
        metric_names.append("keypoints")
    prediction_paths = {
        metric: model_dir / f"{metric}_predictions.json" for metric in metric_names
    }
    engine_path = Path(spec.engine)
    engine_sha256 = _sha256(engine_path)
    if (
        spec.expected_engine_sha256 is not None
        and engine_sha256 != spec.expected_engine_sha256
    ):
        raise EvaluationError(
            f"{spec.id} engine SHA-256 mismatch: "
            f"{engine_sha256} != {spec.expected_engine_sha256}"
        )
    receipt = _engine_receipt(spec, engine_sha256)
    inference_ms = []
    prediction_count = 0
    started = time.monotonic()
    with TensorRTRunner(engine_path) as runner:
        if runner.input_shape[0] != BATCH_SIZE:
            raise EvaluationError(
                f"{spec.id} is not static batch {BATCH_SIZE}: {runner.input_shape}"
            )
        if runner.input_shape != (
            BATCH_SIZE,
            3,
            spec.resolution,
            spec.resolution,
        ):
            raise EvaluationError(
                f"{spec.id} input shape does not match its model contract: "
                f"{runner.input_shape}"
            )
        if runner.input_name != spec.expected_input_name:
            raise EvaluationError(
                f"{spec.id} input tensor name mismatch: "
                f"{runner.input_name} != {spec.expected_input_name}"
            )
        if runner.dtypes[runner.input_name] != np.dtype(np.float32):
            raise EvaluationError(f"{spec.id} input tensor is not float32")
        if set(runner.output_names) != set(spec.expected_outputs):
            raise EvaluationError(
                f"{spec.id} output tensor names mismatch: "
                f"{sorted(runner.output_names)} != "
                f"{sorted(spec.expected_outputs)}"
            )
        for name, expected_shape in spec.expected_outputs.items():
            if runner.shapes[name] != expected_shape:
                raise EvaluationError(
                    f"{spec.id} output {name} shape mismatch: "
                    f"{runner.shapes[name]} != {expected_shape}"
                )
            if runner.dtypes[name] != np.dtype(np.float32):
                raise EvaluationError(
                    f"{spec.id} output tensor {name} is not float32"
                )
        with contextlib.ExitStack() as stack:
            writers = {
                metric: stack.enter_context(JsonArrayWriter(path))
                for metric, path in prediction_paths.items()
            }
            for batch_number, batch_rows in enumerate(_image_batches(image_rows), start=1):
                tensors = []
                transforms = []
                for image_row in batch_rows:
                    image_path = Path(args.dataset_root) / "val2017" / str(
                        image_row["file_name"]
                    )
                    tensor, transform = _prepare_image(image_path, spec)
                    tensors.append(tensor)
                    transforms.append(transform)
                while len(tensors) < BATCH_SIZE:
                    tensors.append(tensors[-1].copy())
                    transforms.append(transforms[-1])
                batch = np.ascontiguousarray(np.stack(tensors), dtype=np.float32)
                if batch_number == 1:
                    runner.infer(batch)
                outputs, elapsed_ms = runner.infer(batch)
                inference_ms.append(elapsed_ms)
                for batch_index, (image_row, transform) in enumerate(
                    zip(batch_rows, transforms, strict=False)
                ):
                    rows = _postprocess(
                        spec,
                        int(image_row["id"]),
                        transform,
                        outputs,
                        batch_index,
                        float(args.score_floor),
                        int(args.topk),
                    )
                    for row in rows:
                        for metric, writer in writers.items():
                            writer.write(_metric_result(row, metric))
                    prediction_count += len(rows)
                if batch_number == 1 or batch_number % int(args.progress_every) == 0:
                    print(
                        f"[{spec.id}] batch {batch_number}/"
                        f"{math.ceil(len(image_rows) / BATCH_SIZE)} "
                        f"predictions={prediction_count}",
                        flush=True,
                    )
    image_ids = [int(row["id"]) for row in image_rows]
    metrics = {}
    for metric in metric_names:
        annotation = (
            keypoint_annotation if metric == "keypoints" else instance_annotation
        )
        metrics[metric] = _evaluate_predictions(
            annotation,
            prediction_paths[metric],
            image_ids,
            metric,
        )
    report = {
        "schema": REPORT_SCHEMA,
        "model": asdict(spec),
        "engine": {
            "path": str(engine_path),
            "sha256": engine_sha256,
            "size_bytes": engine_path.stat().st_size,
            "receipt": receipt,
        },
        "dataset": {
            "root": str(Path(args.dataset_root).resolve()),
            "instance_annotation": str(instance_annotation),
            "instance_annotation_sha256": _sha256(instance_annotation),
            "keypoint_annotation": str(keypoint_annotation),
            "keypoint_annotation_sha256": _sha256(keypoint_annotation),
            "image_count": len(image_rows),
            "person_only": True,
        },
        "evaluation": {
            "lane": "local_coco_model_output_quality",
            "runtime_parser_exact": False,
            "keypoint_context": (
                "standalone_full_frame_engine"
                if spec.architecture == "yolo26" and spec.family == "keypoint"
                else "native_full_frame_model"
            ),
            "score_floor": float(args.score_floor),
            "topk_per_image": int(args.topk),
            "prediction_count": prediction_count,
            "predictions": {
                metric: {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for metric, path in prediction_paths.items()
            },
            "metrics": metrics,
        },
        "timing": {
            "wall_seconds": time.monotonic() - started,
            "inference_batch_count": len(inference_ms),
            "inference_ms_mean": float(np.mean(inference_ms)),
            "inference_ms_p50": float(np.percentile(inference_ms, 50)),
            "inference_ms_p95": float(np.percentile(inference_ms, 95)),
        },
        "runtime": {
            "python": sys.version,
            "tensorrt": str(trt.__version__),
            "numpy": str(np.__version__),
        },
    }
    report_path = model_dir / "evaluation.json"
    _write_json_exclusive(report_path, report)
    print(f"[OK] {spec.id}: {report_path}", flush=True)
    return report


def _platform() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    probe = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "nvidia_smi_command": command,
        "nvidia_smi_returncode": probe.returncode,
        "nvidia_smi": probe.stdout.strip(),
        "tensorrt": str(trt.__version__),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("rf_fp16", "rf_fp32", "yolo_fp16"),
        required=True,
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument(
        "--rf-matrix",
        default=str(Path(__file__).resolve().parents[1] / "config" / "rfdetr_1_8_3_models.json"),
    )
    parser.add_argument("--yolo-matrix", required=True)
    parser.add_argument("--yolo-performance-report", default="")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--score-floor", type=float, default=DEFAULT_SCORE_FLOOR)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    trt_version = str(trt.__version__)
    if not (
        trt_version == EXPECTED_TRT_VERSION
        or trt_version.startswith(f"{EXPECTED_TRT_VERSION}.post")
    ):
        raise EvaluationError(
            f"TensorRT {EXPECTED_TRT_VERSION} is required, found {trt_version}"
        )
    if not (0.0 <= float(args.score_floor) < 1.0):
        raise EvaluationError("--score-floor must be in [0,1)")
    if int(args.topk) <= 0 or int(args.topk) > 300:
        raise EvaluationError("--topk must be in [1,300]")
    if int(args.progress_every) <= 0:
        raise EvaluationError("--progress-every must be positive")
    dataset_root = Path(args.dataset_root).resolve()
    instance_annotation = dataset_root / "annotations" / "instances_val2017.json"
    keypoint_annotation = (
        dataset_root / "annotations" / "person_keypoints_val2017.json"
    )
    if not instance_annotation.is_file() or not keypoint_annotation.is_file():
        raise EvaluationError("COCO instance/keypoint annotations are missing")
    image_index = _load_json(instance_annotation)
    image_rows = sorted(image_index.get("images", []), key=lambda row: int(row["id"]))
    if int(args.limit) > 0:
        image_rows = image_rows[: int(args.limit)]
    if not image_rows:
        raise EvaluationError("COCO image selection is empty")
    specs = _select_specs(args)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    run_contract = {
        "schema": "noesis.ds9.local-coco-head-to-head-run.v1",
        "suite": args.suite,
        "models": [asdict(spec) for spec in specs],
        "dataset": {
            "root": str(dataset_root),
            "instance_annotation": str(instance_annotation),
            "instance_annotation_sha256": _sha256(instance_annotation),
            "keypoint_annotation": str(keypoint_annotation),
            "keypoint_annotation_sha256": _sha256(keypoint_annotation),
            "image_count": len(image_rows),
            "first_image_id": int(image_rows[0]["id"]),
            "last_image_id": int(image_rows[-1]["id"]),
        },
        "score_floor": float(args.score_floor),
        "topk_per_image": int(args.topk),
        "platform": _platform(),
    }
    _write_json_exclusive(run_dir / "run_contract.json", run_contract)
    reports = []
    for spec in specs:
        reports.append(
            _evaluate_model(
                args,
                spec,
                image_rows,
                instance_annotation,
                keypoint_annotation,
            )
        )
    summary = {
        "schema": "noesis.ds9.local-coco-head-to-head-suite.v1",
        "suite": args.suite,
        "model_count": len(reports),
        "models": [
            {
                "id": report["model"]["id"],
                "family": report["model"]["family"],
                "variant": report["model"]["variant"],
                "precision_profile": report["model"]["precision_profile"],
                "prediction_count": report["evaluation"]["prediction_count"],
                "metrics": {
                    metric: values["stats"]
                    for metric, values in report["evaluation"]["metrics"].items()
                },
                "wall_seconds": report["timing"]["wall_seconds"],
            }
            for report in reports
        ],
    }
    _write_json_exclusive(run_dir / "suite_summary.json", summary)
    print(f"[DONE] {run_dir / 'suite_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
