"""Deterministic manual MapAnything inference from an exact RGB frame.

The live nvinfer branch remains responsible for the capture barrier and exact
RGB association.  This runner is invoked only after a manual capture closes
that branch, so the floorplan geometry is produced from the same reviewed
TensorRT engine with explicit, reproducible image preprocessing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


_BATCH_SIZE = 3
_INPUT_HEIGHT = 294
_INPUT_WIDTH = 518
_INPUT_SHAPE = (_BATCH_SIZE, 3, _INPUT_HEIGHT, _INPUT_WIDTH)
_OUTPUT_SHAPE = (_BATCH_SIZE, 1, _INPUT_HEIGHT, _INPUT_WIDTH)
_OUTPUT_NAMES = frozenset({"depth", "conf", "mask"})
_MAX_OUTPUT_JSON_BYTES = 32 * 1024 * 1024
_SCENE_TOP_EXCLUSION_FRACTION = 0.15
_SCENE_BLACK_FOV_MAX_VALUE = 4


class MapAnythingManualInferenceError(RuntimeError):
    """Raised when deterministic manual inference cannot be trusted."""


@dataclass(frozen=True)
class MapAnythingManualInferenceResult:
    """One exact RGB-aligned inference result plus portable evidence."""

    depth: np.ndarray
    confidence: np.ndarray
    mask: np.ndarray
    evidence: Mapping[str, Any]


def _resize_mapanything_rgb(
    rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    image = np.asarray(rgb)
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or int(image.shape[2]) != 3
    ):
        raise MapAnythingManualInferenceError(
            "manual MapAnything input must be uint8 RGB HxWx3"
        )
    source_height, source_width = (int(value) for value in image.shape[:2])
    if source_height < _INPUT_HEIGHT or source_width < _INPUT_WIDTH:
        raise MapAnythingManualInferenceError(
            "manual MapAnything input cannot require source upscaling"
        )
    scale = min(
        _INPUT_WIDTH / float(source_width),
        _INPUT_HEIGHT / float(source_height),
    )
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    preprocess = {
        "source_width": source_width,
        "source_height": source_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
    }
    return image, resized, preprocess


def measure_mapanything_scene_luminance(rgb: np.ndarray) -> dict[str, Any]:
    """Measure visible scene luminance on the exact model-resized RGB content."""

    _image, resized, preprocess = _resize_mapanything_rgb(rgb)
    resized_height, resized_width = (int(value) for value in resized.shape[:2])
    excluded_top_rows = min(
        resized_height,
        int(math.ceil(resized_height * _SCENE_TOP_EXCLUSION_FRACTION)),
    )
    scene_mask = np.max(resized, axis=2) > _SCENE_BLACK_FOV_MAX_VALUE
    scene_mask[:excluded_top_rows, :] = False
    sampled_pixels = int(np.count_nonzero(scene_mask))
    sample_capacity = max(1, (resized_height - excluded_top_rows) * resized_width)
    if sampled_pixels:
        image_float = resized.astype(np.float32)
        luminance = (
            0.2126 * image_float[:, :, 0]
            + 0.7152 * image_float[:, :, 1]
            + 0.0722 * image_float[:, :, 2]
        )
        p50, p90, p99 = (
            float(value)
            for value in np.percentile(luminance[scene_mask], (50.0, 90.0, 99.0))
        )
    else:
        p50 = 0.0
        p90 = 0.0
        p99 = 0.0
    return {
        "contract": "noesis.mapanything.manual_scene_luminance.v1",
        "resized_width": int(preprocess["resized_width"]),
        "resized_height": int(preprocess["resized_height"]),
        "top_exclusion_fraction": _SCENE_TOP_EXCLUSION_FRACTION,
        "excluded_top_rows": excluded_top_rows,
        "black_fov_max_value": _SCENE_BLACK_FOV_MAX_VALUE,
        "sampled_pixels": sampled_pixels,
        "sampled_fraction": float(sampled_pixels / sample_capacity),
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "p99_p50_range": float(max(0.0, p99 - p50)),
    }


def preprocess_mapanything_rgb(
    rgb: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply the canonical HR-0 resize/pad/RGB/NCHW input contract."""

    _image, resized, preprocess = _resize_mapanything_rgb(rgb)
    resized_width = int(preprocess["resized_width"])
    resized_height = int(preprocess["resized_height"])
    pad_x = _INPUT_WIDTH - resized_width
    pad_y = _INPUT_HEIGHT - resized_height
    pad_left = pad_x // 2
    pad_top = pad_y // 2
    canvas = np.zeros((_INPUT_HEIGHT, _INPUT_WIDTH, 3), dtype=np.uint8)
    canvas[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = resized
    chw = np.ascontiguousarray(
        np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1)),
        dtype=np.dtype("<f4"),
    )
    batch = np.ascontiguousarray(
        np.broadcast_to(chw, _INPUT_SHAPE),
        dtype=np.dtype("<f4"),
    )
    return batch, {
        **preprocess,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": int(pad_x - pad_left),
        "pad_bottom": int(pad_y - pad_top),
    }


def _parse_output_json(path: Path) -> dict[str, np.ndarray]:
    try:
        size = int(path.stat().st_size)
    except OSError as exc:
        raise MapAnythingManualInferenceError(
            "trtexec did not publish an output file"
        ) from exc
    if size <= 0 or size > _MAX_OUTPUT_JSON_BYTES:
        raise MapAnythingManualInferenceError(
            f"trtexec output size is outside its contract: {size}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MapAnythingManualInferenceError(
            "trtexec output is not valid JSON"
        ) from exc
    if not isinstance(payload, list) or len(payload) != len(_OUTPUT_NAMES):
        raise MapAnythingManualInferenceError(
            "trtexec output must contain exactly depth, conf, and mask"
        )
    tensors: dict[str, np.ndarray] = {}
    for item in payload:
        if not isinstance(item, Mapping):
            raise MapAnythingManualInferenceError(
                "trtexec output tensor record is malformed"
            )
        name = str(item.get("name") or "")
        if name not in _OUTPUT_NAMES or name in tensors:
            raise MapAnythingManualInferenceError(
                f"unexpected or duplicate trtexec output tensor: {name!r}"
            )
        if str(item.get("dimensions") or "") != "3x1x294x518":
            raise MapAnythingManualInferenceError(
                f"trtexec tensor {name} has an unexpected shape"
            )
        values = item.get("values")
        if not isinstance(values, list) or len(values) != math.prod(_OUTPUT_SHAPE):
            raise MapAnythingManualInferenceError(
                f"trtexec tensor {name} has an unexpected element count"
            )
        try:
            tensor = np.asarray(values, dtype=np.float32).reshape(_OUTPUT_SHAPE)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MapAnythingManualInferenceError(
                f"trtexec tensor {name} contains invalid values"
            ) from exc
        if not np.all(np.isfinite(tensor)):
            raise MapAnythingManualInferenceError(
                f"trtexec tensor {name} contains non-finite values"
            )
        if any(
            tensor[0].tobytes(order="C")
            != tensor[index].tobytes(order="C")
            for index in range(1, _BATCH_SIZE)
        ):
            raise MapAnythingManualInferenceError(
                f"repeated-input trtexec tensor {name} is not deterministic"
            )
        tensors[name] = tensor
    if set(tensors) != _OUTPUT_NAMES:
        raise MapAnythingManualInferenceError(
            "trtexec output tensor set is incomplete"
        )
    return tensors


def _align_output_to_rgb(
    tensors: Mapping[str, np.ndarray],
    preprocess: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_width = int(preprocess["source_width"])
    source_height = int(preprocess["source_height"])
    resized_width = int(preprocess["resized_width"])
    resized_height = int(preprocess["resized_height"])
    left = int(preprocess["pad_left"])
    top = int(preprocess["pad_top"])
    y_slice = slice(top, top + resized_height)
    x_slice = slice(left, left + resized_width)

    depth_native = np.asarray(tensors["depth"][0, 0, y_slice, x_slice])
    conf_native = np.asarray(tensors["conf"][0, 0, y_slice, x_slice])
    mask_native = np.asarray(tensors["mask"][0, 0, y_slice, x_slice])
    depth = cv2.resize(
        depth_native,
        (source_width, source_height),
        interpolation=cv2.INTER_CUBIC,
    ).astype(np.float32, copy=False)
    confidence = cv2.resize(
        conf_native,
        (source_width, source_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32, copy=False)
    model_mask = cv2.resize(
        mask_native,
        (source_width, source_height),
        interpolation=cv2.INTER_NEAREST,
    ) >= 0.5
    valid = model_mask & np.isfinite(depth) & (depth > 0.0)
    aligned_depth = np.where(valid, depth, np.nan).astype(np.float32)
    aligned_confidence = np.where(
        valid & np.isfinite(confidence),
        confidence,
        0.0,
    ).astype(np.float32)
    return aligned_depth, aligned_confidence, valid.astype(np.uint8)


class TrtexecMapAnythingManualInferencer:
    """Run the fixed canonical engine only for an explicit manual capture."""

    def __init__(
        self,
        *,
        engine_path: str | os.PathLike[str],
        work_root: str | os.PathLike[str] | None = None,
        trtexec_path: str | os.PathLike[str] | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        engine = Path(engine_path).expanduser()
        if not engine.is_absolute() or not engine.is_file():
            raise MapAnythingManualInferenceError(
                "manual MapAnything engine must be an existing absolute file"
            )
        executable = (
            Path(trtexec_path).expanduser()
            if trtexec_path is not None
            else Path(shutil.which("trtexec") or "")
        )
        if not executable.is_absolute() or not os.access(executable, os.X_OK):
            raise MapAnythingManualInferenceError(
                "manual MapAnything inference requires executable trtexec"
            )
        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("manual MapAnything timeout must be positive")
        root = (
            Path(work_root).expanduser()
            if work_root is not None
            else Path(tempfile.gettempdir())
        )
        if not root.is_absolute():
            raise MapAnythingManualInferenceError(
                "manual MapAnything work root must be absolute"
            )
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not root.is_dir():
            raise MapAnythingManualInferenceError(
                "manual MapAnything work root is unavailable"
            )
        self._engine_path = engine
        self._trtexec_path = executable
        self._work_root = root
        self._timeout_s = timeout
        self._lock = threading.Lock()

    def infer(self, rgb: np.ndarray) -> MapAnythingManualInferenceResult:
        image = np.ascontiguousarray(np.asarray(rgb))
        batch, preprocess = preprocess_mapanything_rgb(image)
        input_sha256 = hashlib.sha256(batch.tobytes(order="C")).hexdigest()
        started = time.perf_counter()
        with self._lock, tempfile.TemporaryDirectory(
            prefix="mapanything-manual-",
            dir=str(self._work_root),
        ) as temporary:
            temp_root = Path(temporary)
            input_path = temp_root / "images-b3.raw"
            output_path = temp_root / "output.json"
            transcript_path = temp_root / "trtexec.log"
            batch.tofile(input_path)
            command = [
                str(self._trtexec_path),
                f"--loadEngine={self._engine_path}",
                f"--loadInputs=images:{input_path}",
                "--iterations=1",
                "--duration=0",
                "--warmUp=0",
                "--avgRuns=1",
                "--dumpOutput",
                f"--exportOutput={output_path}",
            ]
            try:
                with transcript_path.open("wb") as transcript:
                    completed = subprocess.run(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=transcript,
                        stderr=subprocess.STDOUT,
                        timeout=self._timeout_s,
                        check=False,
                        start_new_session=True,
                    )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise MapAnythingManualInferenceError(
                    f"trtexec manual inference failed: {type(exc).__name__}"
                ) from exc
            if completed.returncode != 0:
                try:
                    transcript = transcript_path.read_bytes()[-4096:].decode(
                        "utf-8",
                        errors="replace",
                    )
                except OSError:
                    transcript = ""
                raise MapAnythingManualInferenceError(
                    "trtexec manual inference returned "
                    f"{completed.returncode}: {transcript.strip()}"
                )
            tensors = _parse_output_json(output_path)
        depth, confidence, mask = _align_output_to_rgb(tensors, preprocess)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        valid_fraction = float(np.count_nonzero(mask) / max(1, mask.size))
        return MapAnythingManualInferenceResult(
            depth=depth,
            confidence=confidence,
            mask=mask,
            evidence={
                "contract": "noesis.mapanything.manual_inference.v1",
                "engine_name": self._engine_path.name,
                "runner": self._trtexec_path.name,
                "input_color_space": "rgb8",
                "input_content_sha256": hashlib.sha256(
                    image.tobytes(order="C")
                ).hexdigest(),
                "input_tensor_sha256": input_sha256,
                "input_shape": list(_INPUT_SHAPE),
                "output_shape": list(_OUTPUT_SHAPE),
                "preprocess": dict(preprocess),
                "valid_fraction": valid_fraction,
                "elapsed_ms": float(elapsed_ms),
            },
        )


__all__ = [
    "MapAnythingManualInferenceError",
    "MapAnythingManualInferenceResult",
    "TrtexecMapAnythingManualInferencer",
    "measure_mapanything_scene_luminance",
    "preprocess_mapanything_rgb",
]
