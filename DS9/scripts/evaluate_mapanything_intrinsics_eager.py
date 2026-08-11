#!/usr/bin/env python3
"""Run a locked eager MapAnything calibration-conditioning diagnostic.

This is an isolated measurement tool. It does not build or select an engine,
start DS9, or mutate canonical artifacts.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np


CONTRACT = "noesis.ds9.mapanything_intrinsics_eager.v2"
COMPARISON_MANIFEST_CONTRACT = (
    "noesis.ds9.mapanything_fixed_corpus_comparison.v1"
)
TRT_REFERENCE_CANDIDATE = "v11-control-294x518"
EXPECTED_SHAPE = (3, 3, 294, 518)
RAY_HELPER_ABS_TOLERANCE = 2e-7
ARM_NAMES = (
    "image_only",
    "authored_rays",
    "recovered_rays",
    "swapped_authored_rays_control",
)


class DiagnosticError(RuntimeError):
    """Raised when diagnostic provenance or model output is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_intrinsics_pixel_center(
    intrinsics: np.ndarray,
    *,
    source_size: tuple[int, int],
    resized_size: tuple[int, int],
    pad_left: int,
    pad_top: int,
) -> np.ndarray:
    """Transform K through an OpenCV-style half-pixel resize plus padding."""

    source_width, source_height = source_size
    resized_width, resized_height = resized_size
    if min(source_width, source_height, resized_width, resized_height) <= 0:
        raise DiagnosticError("source and resized dimensions must be positive")
    if pad_left < 0 or pad_top < 0:
        raise DiagnosticError("padding must be non-negative")
    source_k = np.asarray(intrinsics, dtype=np.float64)
    if source_k.shape != (3, 3) or not np.all(np.isfinite(source_k)):
        raise DiagnosticError("intrinsics must be one finite 3x3 matrix")
    if abs(float(source_k[2, 2]) - 1.0) > 1e-9:
        raise DiagnosticError("intrinsics homogeneous scale must be one")
    sx = float(resized_width) / float(source_width)
    sy = float(resized_height) / float(source_height)
    transform = np.asarray(
        [
            [sx, 0.0, float(pad_left) + ((sx - 1.0) / 2.0)],
            [0.0, sy, float(pad_top) + ((sy - 1.0) / 2.0)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transformed = transform @ source_k
    return transformed.astype(np.float32)


def rays_from_intrinsics_numpy(
    intrinsics: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Construct unit camera rays with MapAnything's integer-pixel convention."""

    matrices = np.asarray(intrinsics, dtype=np.float64)
    if matrices.ndim == 2:
        matrices = matrices[None, ...]
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise DiagnosticError("intrinsics batch must have shape Nx3x3")
    if height <= 0 or width <= 0:
        raise DiagnosticError("ray dimensions must be positive")
    u_grid, v_grid = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
        indexing="xy",
    )
    rays: list[np.ndarray] = []
    for matrix in matrices:
        fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
        cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
        if abs(fx) <= 1e-12 or abs(fy) <= 1e-12:
            raise DiagnosticError("intrinsics focal lengths must be non-zero")
        raw = np.stack(
            (
                (u_grid - cx) / fx,
                (v_grid - cy) / fy,
                np.ones_like(u_grid),
            ),
            axis=-1,
        )
        norm = np.linalg.norm(raw, axis=-1, keepdims=True)
        rays.append(raw / np.maximum(norm, 1e-12))
    return np.asarray(rays, dtype=np.float32)


def fit_pinhole_intrinsics(
    rays: np.ndarray,
    *,
    sample_stride: int = 8,
) -> dict[str, float]:
    """Least-squares recover fx/fy/cx/cy from predicted camera rays."""

    values = np.asarray(rays, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise DiagnosticError("one ray map must have shape HxWx3")
    if sample_stride <= 0:
        raise DiagnosticError("sample stride must be positive")
    height, width = values.shape[:2]
    v_grid, u_grid = np.meshgrid(
        np.arange(0, height, sample_stride, dtype=np.float64),
        np.arange(0, width, sample_stride, dtype=np.float64),
        indexing="ij",
    )
    sampled = values[::sample_stride, ::sample_stride]
    z = sampled[..., 2]
    valid = (
        np.all(np.isfinite(sampled), axis=-1)
        & (np.abs(z) > 1e-8)
    )
    if int(np.count_nonzero(valid)) < 16:
        raise DiagnosticError("too few finite rays to recover intrinsics")
    x_ratio = (sampled[..., 0] / z)[valid]
    y_ratio = (sampled[..., 1] / z)[valid]
    u = u_grid[valid]
    v = v_grid[valid]
    fx, cx = np.linalg.lstsq(
        np.stack((x_ratio, np.ones_like(x_ratio)), axis=1),
        u,
        rcond=None,
    )[0]
    fy, cy = np.linalg.lstsq(
        np.stack((y_ratio, np.ones_like(y_ratio)), axis=1),
        v,
        rcond=None,
    )[0]
    u_residual = u - ((fx * x_ratio) + cx)
    v_residual = v - ((fy * y_ratio) + cy)
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "fit_residual_px_p50": float(
            np.percentile(np.hypot(u_residual, v_residual), 50)
        ),
        "fit_residual_px_p95": float(
            np.percentile(np.hypot(u_residual, v_residual), 95)
        ),
    }


def angular_error_metrics(
    observed: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    left = np.asarray(observed, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape or left.shape[-1] != 3:
        raise DiagnosticError("ray maps must have equal ...x3 shapes")
    left /= np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1e-12)
    right /= np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-12)
    dots = np.clip(np.sum(left * right, axis=-1), -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    finite = angles[np.isfinite(angles)]
    if finite.size == 0:
        raise DiagnosticError("ray comparison has no finite samples")
    return {
        "degrees_p50": float(np.percentile(finite, 50)),
        "degrees_p95": float(np.percentile(finite, 95)),
        "degrees_max": float(np.max(finite)),
    }


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise DiagnosticError(f"non-finite JSON constant {value!r} in {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    path.chmod(0o600)


def _write_trt_output(
    path: Path,
    *,
    depth: np.ndarray,
    conf: np.ndarray,
    mask: np.ndarray,
) -> None:
    arrays = {
        "conf": np.asarray(conf, dtype=np.float32),
        "depth": np.asarray(depth, dtype=np.float32),
        "mask": np.asarray(mask, dtype=np.float32),
    }
    shape: tuple[int, ...] | None = None
    payload: list[dict[str, Any]] = []
    for name in ("conf", "depth", "mask"):
        array = arrays[name]
        if array.ndim != 3:
            raise DiagnosticError(f"{name} must have shape NxHxW")
        if not np.all(np.isfinite(array)):
            raise DiagnosticError(f"{name} contains non-finite values")
        current_shape = (array.shape[0], 1, array.shape[1], array.shape[2])
        if shape is None:
            shape = current_shape
        elif shape != current_shape:
            raise DiagnosticError("model output shapes differ")
        payload.append(
            {
                "name": name,
                "dimensions": "x".join(str(value) for value in current_shape),
                "values": array.reshape(-1).tolist(),
            }
        )
    _write_json(path, payload)


def _parse_scene_output(path: Path) -> dict[str, np.ndarray]:
    payload = _strict_json(path)
    if not isinstance(payload, list):
        raise DiagnosticError(f"scene output must be a JSON list: {path}")
    tensors: dict[str, np.ndarray] = {}
    for item in payload:
        if not isinstance(item, Mapping):
            raise DiagnosticError(f"scene output item must be an object: {path}")
        name = str(item.get("name") or "")
        if name not in {"depth", "conf", "mask"} or name in tensors:
            raise DiagnosticError(f"unexpected or duplicate output layer {name!r}")
        dimensions = str(item.get("dimensions") or "")
        try:
            shape = tuple(int(value) for value in dimensions.split("x"))
        except ValueError as exc:
            raise DiagnosticError(
                f"invalid dimensions for output layer {name!r}"
            ) from exc
        if len(shape) != 4 or shape[1] != 1 or min(shape) <= 0:
            raise DiagnosticError(
                f"output layer {name!r} must have shape Nx1xHxW"
            )
        values = np.asarray(item.get("values"), dtype=np.float32)
        if values.size != math.prod(shape):
            raise DiagnosticError(
                f"output layer {name!r} value count does not match dimensions"
            )
        values = values.reshape(shape)[:, 0]
        if not np.all(np.isfinite(values)):
            raise DiagnosticError(
                f"output layer {name!r} contains non-finite values"
            )
        tensors[name] = values
    if set(tensors) != {"depth", "conf", "mask"}:
        raise DiagnosticError("scene output must contain depth, conf, and mask")
    return tensors


def layer_parity_metrics(
    actual: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    left = np.asarray(actual, dtype=np.float32)
    right = np.asarray(reference, dtype=np.float32)
    if left.shape != right.shape:
        raise DiagnosticError(
            f"parity layer shapes differ: {left.shape} versus {right.shape}"
        )
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise DiagnosticError("parity layers must contain only finite values")
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {
        "exact_equal": bool(np.array_equal(left, right)),
        "abs_delta": {
            "p50": float(np.percentile(delta, 50)),
            "p95": float(np.percentile(delta, 95)),
            "p99": float(np.percentile(delta, 99)),
            "max": float(np.max(delta)),
        },
    }


def verify_model_ray_helper_parity(
    constructed: np.ndarray,
    model_helper: np.ndarray,
) -> dict[str, Any]:
    """Require our authored rays to match MapAnything's pinned helper."""

    metrics = layer_parity_metrics(constructed, model_helper)
    metrics["angular_error_degrees"] = angular_error_metrics(
        constructed,
        model_helper,
    )
    if metrics["abs_delta"]["max"] > RAY_HELPER_ABS_TOLERANCE:
        raise DiagnosticError(
            "authored rays differ materially from MapAnything's geometry helper: "
            f"max abs delta {metrics['abs_delta']['max']}"
        )
    return metrics


def _require_absolute_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise DiagnosticError(f"{label} path must be absolute: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise DiagnosticError(f"{label} is not a file: {resolved}")
    return resolved


def _require_sha_record(
    value: Any,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise DiagnosticError(f"{label} identity must be an object")
    path = _require_absolute_file(Path(str(value.get("path") or "")), label)
    expected_sha = str(value.get("sha256") or "")
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise DiagnosticError(
            f"{label} SHA mismatch: expected {expected_sha}, observed {observed_sha}"
        )
    expected_size = value.get("size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise DiagnosticError(f"{label} byte count does not match its identity")
    return path, {
        "path": str(path),
        "sha256": observed_sha,
        "size_bytes": path.stat().st_size,
    }


def load_trt_reference(
    *,
    comparison_manifest_path: Path,
    input_raw: Path,
    fixture_receipt_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = _strict_json(comparison_manifest_path)
    if not isinstance(manifest, Mapping):
        raise DiagnosticError("comparison manifest must be a JSON object")
    if manifest.get("contract") != COMPARISON_MANIFEST_CONTRACT:
        raise DiagnosticError(
            f"comparison manifest contract must be {COMPARISON_MANIFEST_CONTRACT}"
        )
    candidates = manifest.get("candidates")
    if not isinstance(candidates, Mapping):
        raise DiagnosticError("comparison manifest has no candidates object")
    candidate = candidates.get(TRT_REFERENCE_CANDIDATE)
    if not isinstance(candidate, Mapping):
        raise DiagnosticError(
            f"comparison manifest has no {TRT_REFERENCE_CANDIDATE!r} candidate"
        )
    if candidate.get("include_intrinsics") is not False:
        raise DiagnosticError("TRT reference must be the image-only v1.1 control")

    input_identity = candidate.get("fixture_tensor")
    if not isinstance(input_identity, Mapping):
        raise DiagnosticError("TRT reference has no fixture tensor identity")
    input_sha = sha256_file(input_raw)
    if str(input_identity.get("sha256") or "") != input_sha:
        raise DiagnosticError("TRT reference fixture tensor differs from eager input")

    receipt_identity = candidate.get("fixture_receipt")
    if not isinstance(receipt_identity, Mapping):
        raise DiagnosticError("TRT reference has no fixture receipt identity")
    receipt_sha = sha256_file(fixture_receipt_path)
    if str(receipt_identity.get("sha256") or "") != receipt_sha:
        raise DiagnosticError(
            "TRT reference fixture receipt differs from eager fixture receipt"
        )

    output_path, output_identity = _require_sha_record(
        candidate.get("model_output"),
        label="TRT reference output",
    )
    engine = candidate.get("engine")
    if not isinstance(engine, Mapping):
        raise DiagnosticError("TRT reference has no engine identity")
    engine_identity = {
        "path": str(engine.get("path") or ""),
        "sha256": str(engine.get("sha256") or ""),
        "size_bytes": int(engine.get("size_bytes") or 0),
        "verification": "content identity bound by comparison manifest",
    }
    if (
        not engine_identity["path"]
        or len(engine_identity["sha256"]) != 64
        or engine_identity["size_bytes"] <= 0
    ):
        raise DiagnosticError("TRT reference engine identity is incomplete")

    manifest_identity = {
        "path": str(comparison_manifest_path),
        "sha256": sha256_file(comparison_manifest_path),
        "size_bytes": comparison_manifest_path.stat().st_size,
    }
    return _parse_scene_output(output_path), {
        "comparison_manifest": manifest_identity,
        "candidate_id": TRT_REFERENCE_CANDIDATE,
        "reference_output": output_identity,
        "reference_engine": engine_identity,
    }


def resolve_hf_artifacts(
    *,
    exporter: Any,
    model_id: str,
    revision: str,
) -> dict[str, dict[str, Any]]:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise DiagnosticError("Hugging Face revision must be one full 40-hex commit")
    records: dict[str, dict[str, Any]] = {}
    for key, filename in (
        ("config", "config.json"),
        ("model_safetensors", "model.safetensors"),
    ):
        try:
            logical = Path(
                exporter.hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    revision=revision,
                    local_files_only=True,
                )
            )
        except Exception as exc:
            raise DiagnosticError(
                f"locked Hugging Face artifact is unavailable: {filename}"
            ) from exc
        if not logical.is_absolute():
            raise DiagnosticError(
                f"Hugging Face {filename} path must be absolute: {logical}"
            )
        if not logical.is_file():
            raise DiagnosticError(
                f"Hugging Face {filename} is not a file: {logical}"
            )
        resolved = logical.resolve(strict=True)
        records[key] = {
            "logical_path": str(logical),
            "resolved_path": str(resolved),
            "sha256": sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        }
    return records


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in (
        "huggingface-hub",
        "hydra-core",
        "numpy",
        "safetensors",
        "transformers",
        "uniception",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise DiagnosticError(
                f"required dependency metadata is missing: {distribution}"
            ) from exc
    return versions


def _gpu_compute_owners() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and "no running processes" not in line.lower()
    ]


@contextmanager
def _artifact_lock(path: Path) -> Iterator[None]:
    resolved = _require_absolute_file(path, "artifact lock")
    descriptor = os.open(resolved, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DiagnosticError(f"artifact lock is already owned: {resolved}") from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _load_corpus(
    *,
    repo_root: Path,
    input_raw: Path,
    fixture_receipt_path: Path,
    annotations_path: Path,
) -> tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any], list[str], np.ndarray]:
    scripts_dir = repo_root / "DS9" / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import evaluate_mapanything_fixed_corpus as fixed  # type: ignore

    receipt = _strict_json(fixture_receipt_path)
    annotations = _strict_json(annotations_path)
    if not isinstance(receipt, Mapping) or not isinstance(annotations, Mapping):
        raise DiagnosticError("receipt and annotations must be JSON objects")
    tensor_meta = receipt.get("tensor")
    if not isinstance(tensor_meta, Mapping):
        raise DiagnosticError("fixture receipt has no tensor object")
    shape = tuple(int(value) for value in tensor_meta.get("shape", ()))
    if shape != EXPECTED_SHAPE:
        raise DiagnosticError(
            f"diagnostic requires exact input shape {EXPECTED_SHAPE}; observed {shape}"
        )
    expected_sha = str(tensor_meta.get("sha256") or "")
    observed_sha = sha256_file(input_raw)
    if observed_sha != expected_sha:
        raise DiagnosticError(
            f"input tensor SHA mismatch: expected {expected_sha}, observed {observed_sha}"
        )
    expected_bytes = math.prod(shape) * np.dtype("<f4").itemsize
    if input_raw.stat().st_size != expected_bytes:
        raise DiagnosticError("input tensor byte count does not match receipt")
    images = np.fromfile(input_raw, dtype="<f4").reshape(shape)
    if not np.all(np.isfinite(images)):
        raise DiagnosticError("input tensor contains non-finite values")
    if float(images.min()) < 0.0 or float(images.max()) > 1.0:
        raise DiagnosticError("input tensor must be RGB in the 0..1 range")

    _, _, calibration_arrays = fixed._validate_calibration(
        annotations,
        annotation_path=annotations_path,
        repo_root=repo_root,
    )
    frames = annotations.get("frames")
    sources = receipt.get("sources")
    if not isinstance(frames, list) or not isinstance(sources, list):
        raise DiagnosticError("annotations.frames and receipt.sources must be lists")
    camera_by_sha: dict[str, str] = {}
    for frame in frames:
        if not isinstance(frame, Mapping) or not isinstance(frame.get("rgb"), Mapping):
            raise DiagnosticError("annotation frame has no RGB identity")
        camera_by_sha[str(frame["rgb"].get("sha256") or "")] = str(
            frame.get("camera_id") or ""
        )

    camera_order: list[str] = []
    transformed: list[np.ndarray] = []
    for source in sources:
        if not isinstance(source, Mapping):
            raise DiagnosticError("fixture source must be an object")
        camera_id = camera_by_sha.get(str(source.get("sha256") or ""), "")
        source_key = f"{camera_id}:K"
        if not camera_id or source_key not in calibration_arrays:
            raise DiagnosticError("fixture source has no authored camera intrinsics")
        camera_order.append(camera_id)
        transformed.append(
            transform_intrinsics_pixel_center(
                calibration_arrays[source_key],
                source_size=(
                    int(source.get("source_width") or 0),
                    int(source.get("source_height") or 0),
                ),
                resized_size=(
                    int(source.get("resized_width") or 0),
                    int(source.get("resized_height") or 0),
                ),
                pad_left=int(source.get("pad_left") or 0),
                pad_top=int(source.get("pad_top") or 0),
            )
        )
    if len(set(camera_order)) != len(camera_order):
        raise DiagnosticError("camera batch order must contain unique cameras")
    return images, receipt, annotations, camera_order, np.stack(transformed)


def _run_model_arm(
    *,
    model: Any,
    normalized_images: Any,
    norm_type: str,
    ray_directions: Any | None,
) -> dict[str, np.ndarray]:
    import torch

    configure = getattr(model, "_configure_geometric_input_config", None)
    restore = getattr(model, "_restore_original_geometric_input_config", None)
    if not callable(configure) or not callable(restore):
        raise DiagnosticError(
            "MapAnything model lacks deterministic inference geometry controls"
        )
    use_calibration = ray_directions is not None
    configure(
        use_calibration=use_calibration,
        use_depth=False,
        use_pose=False,
        use_depth_scale=False,
        use_pose_scale=False,
    )
    expected_config = {
        "overall_prob": 1.0 if use_calibration else 0.0,
        "dropout_prob": 0.0 if use_calibration else 1.0,
        "ray_dirs_prob": 1.0 if use_calibration else 0.0,
        "depth_prob": 0.0,
        "cam_prob": 0.0,
    }
    observed_config = getattr(model, "geometric_input_config", None)
    if not isinstance(observed_config, Mapping) or any(
        float(observed_config.get(key, float("nan"))) != value
        for key, value in expected_config.items()
    ):
        restore()
        raise DiagnosticError(
            "MapAnything deterministic inference geometry configuration failed"
        )

    torch.manual_seed(0)
    view: dict[str, Any] = {
        "img": normalized_images,
        "data_norm_type": [norm_type],
    }
    if ray_directions is not None:
        view["ray_directions_cam"] = ray_directions
    try:
        with torch.inference_mode():
            predictions = model.forward([view], memory_efficient_inference=False)
    finally:
        restore()
    if not isinstance(predictions, (list, tuple)) or len(predictions) != 1:
        raise DiagnosticError("MapAnything forward did not return one view")
    output = predictions[0]
    if not isinstance(output, Mapping):
        raise DiagnosticError("MapAnything output is not an object")
    required = ("pts3d_cam", "ray_directions", "conf", "non_ambiguous_mask")
    missing = [name for name in required if name not in output]
    if missing:
        raise DiagnosticError(
            "MapAnything output is missing required real layer(s): "
            + ", ".join(missing)
        )
    arrays = {
        "depth": output["pts3d_cam"][..., 2].detach().float().cpu().numpy(),
        "rays": output["ray_directions"].detach().float().cpu().numpy(),
        "conf": output["conf"].detach().float().cpu().numpy(),
        "mask": output["non_ambiguous_mask"].detach().float().cpu().numpy(),
    }
    expected = (EXPECTED_SHAPE[0], EXPECTED_SHAPE[2], EXPECTED_SHAPE[3])
    for name in ("depth", "conf", "mask"):
        if arrays[name].shape != expected:
            raise DiagnosticError(
                f"{name} shape {arrays[name].shape} does not match {expected}"
            )
        if not np.all(np.isfinite(arrays[name])):
            raise DiagnosticError(f"{name} contains non-finite values")
    if arrays["rays"].shape != (*expected, 3):
        raise DiagnosticError("predicted ray shape does not match the input")
    if not np.all(np.isfinite(arrays["rays"])):
        raise DiagnosticError("predicted rays contain non-finite values")
    return arrays


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--hf-model-id", required=True)
    parser.add_argument("--hf-revision", required=True)
    parser.add_argument("--input-raw", type=Path, required=True)
    parser.add_argument("--fixture-receipt", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-lock", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve(strict=True)
    source_repo = args.source_repo.resolve(strict=True)
    input_raw = _require_absolute_file(args.input_raw, "input tensor")
    fixture_receipt = _require_absolute_file(
        args.fixture_receipt, "fixture receipt"
    )
    annotations_path = _require_absolute_file(args.annotations, "annotations")
    comparison_manifest_path = _require_absolute_file(
        args.comparison_manifest,
        "comparison manifest",
    )
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        raise DiagnosticError("--output-dir must be absolute")
    if output_dir.exists() or output_dir.is_symlink():
        raise DiagnosticError(f"refusing to replace output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = output_dir.with_name(f".{output_dir.name}.partial-{os.getpid()}")
    if partial.exists() or partial.is_symlink():
        raise DiagnosticError(f"partial output already exists: {partial}")

    source_head = subprocess.check_output(
        ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if source_head != args.source_commit:
        raise DiagnosticError(
            f"source commit mismatch: expected {args.source_commit}, observed {source_head}"
        )
    source_status = subprocess.check_output(
        ["git", "-C", str(source_repo), "status", "--porcelain"],
        text=True,
    )
    if source_status:
        raise DiagnosticError("MapAnything source repository is dirty")

    with _artifact_lock(args.artifact_lock):
        owners = _gpu_compute_owners()
        if owners:
            raise DiagnosticError(f"GPU is not exclusive: {owners}")
        partial.mkdir(mode=0o700)
        try:
            images, receipt, annotations, camera_order, network_k = _load_corpus(
                repo_root=repo_root,
                input_raw=input_raw,
                fixture_receipt_path=fixture_receipt,
                annotations_path=annotations_path,
            )
            trt_reference, trt_reference_identity = load_trt_reference(
                comparison_manifest_path=comparison_manifest_path,
                input_raw=input_raw,
                fixture_receipt_path=fixture_receipt,
            )
            constructed_authored_rays_np = rays_from_intrinsics_numpy(
                network_k,
                height=EXPECTED_SHAPE[2],
                width=EXPECTED_SHAPE[3],
            )

            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            sys.path.insert(0, str(repo_root))
            from utils.onnx2trt.export_ma_onnx import export_to_onnx as exporter  # type: ignore
            import torch

            expected_exporter_path = (
                repo_root
                / "utils"
                / "onnx2trt"
                / "export_ma_onnx"
                / "export_to_onnx.py"
            ).resolve(strict=True)
            exporter_path = Path(exporter.__file__).resolve(strict=True)
            if exporter_path != expected_exporter_path:
                raise DiagnosticError(
                    "imported MapAnything exporter does not match repository source"
                )
            hf_artifacts = resolve_hf_artifacts(
                exporter=exporter,
                model_id=args.hf_model_id,
                revision=args.hf_revision,
            )
            config = exporter.ExportConfig(
                repo_path=source_repo,
                outdir=partial,
                output_name=None,
                report_name=None,
                fail_if_output_exists=True,
                height=EXPECTED_SHAPE[2],
                width=EXPECTED_SHAPE[3],
                opset=17,
                checkpoint_path=None,
                repo_url=None,
                repo_branch=None,
                fused_input=False,
                hf_model_id=args.hf_model_id,
                hf_revision=args.hf_revision,
                include_intrinsics=False,
                return_conf_mask=True,
                export_device="cuda",
                skip_eager_smoke=True,
                skip_ort=True,
                skip_simplify=True,
                skip_shape_inference=True,
            )
            started = time.monotonic()
            artifacts = exporter.load_model(config)
            wrapper = artifacts["wrapper"]
            model = artifacts["model"]
            from mapanything.utils import geometry as mapanything_geometry  # type: ignore

            expected_geometry_path = (
                source_repo / "mapanything" / "utils" / "geometry.py"
            ).resolve(strict=True)
            geometry_path = Path(mapanything_geometry.__file__).resolve(strict=True)
            if geometry_path != expected_geometry_path:
                raise DiagnosticError(
                    "imported MapAnything geometry helper does not match locked source"
                )
            expected_model_path = (
                source_repo
                / "mapanything"
                / "models"
                / "mapanything"
                / "model.py"
            ).resolve(strict=True)
            model_source_path = Path(
                str(artifacts["model_module_path"])
            ).resolve(strict=True)
            if model_source_path != expected_model_path:
                raise DiagnosticError(
                    "loaded MapAnything model does not match locked source"
                )
            _, helper_authored_rays = (
                mapanything_geometry.get_rays_in_camera_frame(
                    intrinsics=torch.from_numpy(np.array(network_k, copy=True)),
                    height=EXPECTED_SHAPE[2],
                    width=EXPECTED_SHAPE[3],
                    normalize_to_unit_sphere=True,
                )
            )
            authored_rays_np = (
                helper_authored_rays.detach().float().cpu().numpy()
            )
            ray_helper_parity = verify_model_ray_helper_parity(
                constructed_authored_rays_np,
                authored_rays_np,
            )
            tensor = torch.from_numpy(np.array(images, copy=True)).to(
                device="cuda",
                dtype=torch.float32,
            )
            normalized = (tensor - wrapper.mean) / wrapper.std
            authored_rays = torch.from_numpy(authored_rays_np).to(
                device="cuda",
                dtype=torch.float32,
            )

            arm_outputs: dict[str, dict[str, np.ndarray]] = {}
            arm_outputs["image_only"] = _run_model_arm(
                model=model,
                normalized_images=normalized,
                norm_type=artifacts["norm_type"],
                ray_directions=None,
            )
            recovered_rays = torch.from_numpy(
                arm_outputs["image_only"]["rays"]
            ).to(device="cuda", dtype=torch.float32)
            arm_outputs["authored_rays"] = _run_model_arm(
                model=model,
                normalized_images=normalized,
                norm_type=artifacts["norm_type"],
                ray_directions=authored_rays,
            )
            arm_outputs["recovered_rays"] = _run_model_arm(
                model=model,
                normalized_images=normalized,
                norm_type=artifacts["norm_type"],
                ray_directions=recovered_rays,
            )
            swapped_indices = [2, 1, 0]
            swapped_rays_np = authored_rays_np[swapped_indices]
            swapped_rays = torch.from_numpy(swapped_rays_np).to(
                device="cuda",
                dtype=torch.float32,
            )
            arm_outputs["swapped_authored_rays_control"] = _run_model_arm(
                model=model,
                normalized_images=normalized,
                norm_type=artifacts["norm_type"],
                ray_directions=swapped_rays,
            )
            elapsed_s = time.monotonic() - started

            sys.path.insert(0, str(repo_root / "DS9" / "scripts"))
            import evaluate_mapanything_fixed_corpus as fixed  # type: ignore

            supplied_rays_by_arm: dict[str, np.ndarray | None] = {
                "image_only": None,
                "authored_rays": authored_rays_np,
                "recovered_rays": arm_outputs["image_only"]["rays"],
                "swapped_authored_rays_control": swapped_rays_np,
            }
            supplied_ray_labels = {
                "image_only": "none",
                "authored_rays": "authored_calibration",
                "recovered_rays": "image_only_predicted_output",
                "swapped_authored_rays_control": "cross_camera_authored_control",
            }
            output_records: dict[str, Any] = {}
            for arm_name in ARM_NAMES:
                values = arm_outputs[arm_name]
                output_path = partial / f"{arm_name}.scene-output.json"
                _write_trt_output(
                    output_path,
                    depth=values["depth"],
                    conf=values["conf"],
                    mask=values["mask"],
                )
                evaluation = fixed.evaluate(
                    annotations_path=annotations_path,
                    fixture_receipt_path=fixture_receipt,
                    model_output_path=output_path,
                    model_output_identity_path=output_dir / output_path.name,
                    candidate_id=f"eager-{arm_name}",
                    repo_root=repo_root,
                )
                evaluation_path = partial / f"{arm_name}.evaluation.json"
                _write_json(evaluation_path, evaluation)
                per_camera: dict[str, Any] = {}
                for index, camera_id in enumerate(camera_order):
                    camera_record: dict[str, Any] = {
                        "output_ray_vs_authored": angular_error_metrics(
                            values["rays"][index],
                            authored_rays_np[index],
                        ),
                        "recovered_output_intrinsics": fit_pinhole_intrinsics(
                            values["rays"][index]
                        ),
                    }
                    supplied_rays_np = supplied_rays_by_arm[arm_name]
                    if supplied_rays_np is not None:
                        camera_record.update(
                            {
                                "output_ray_vs_supplied": angular_error_metrics(
                                    values["rays"][index],
                                    supplied_rays_np[index],
                                ),
                                "supplied_ray_vs_authored": angular_error_metrics(
                                    supplied_rays_np[index],
                                    authored_rays_np[index],
                                ),
                            }
                        )
                    per_camera[camera_id] = camera_record
                supplied_rays_np = supplied_rays_by_arm[arm_name]
                output_records[arm_name] = {
                    "conditioning_input": {
                        "kind": supplied_ray_labels[arm_name],
                        "ray_sha256": (
                            hashlib.sha256(
                                supplied_rays_np.tobytes(order="C")
                            ).hexdigest()
                            if supplied_rays_np is not None
                            else None
                        ),
                    },
                    "scene_output": {
                        "path": output_path.name,
                        "sha256": sha256_file(output_path),
                        "size_bytes": output_path.stat().st_size,
                    },
                    "evaluation": {
                        "path": evaluation_path.name,
                        "sha256": sha256_file(evaluation_path),
                        "size_bytes": evaluation_path.stat().st_size,
                    },
                    "valid_mask_fraction": float(np.mean(values["mask"] >= 0.5)),
                    "confidence": {
                        "p05": float(np.percentile(values["conf"], 5)),
                        "p50": float(np.percentile(values["conf"], 50)),
                        "p95": float(np.percentile(values["conf"], 95)),
                    },
                    "depth_abs_delta_vs_image_only_m": {
                        "p50": float(
                            np.percentile(
                                np.abs(
                                    values["depth"]
                                    - arm_outputs["image_only"]["depth"]
                                ),
                                50,
                            )
                        ),
                        "p95": float(
                            np.percentile(
                                np.abs(
                                    values["depth"]
                                    - arm_outputs["image_only"]["depth"]
                                ),
                                95,
                            )
                        ),
                    },
                    "per_camera": per_camera,
                }

            script_path = Path(__file__).resolve()
            parity_layers: dict[str, Any] = {}
            for layer_name in ("depth", "conf", "mask"):
                parity_layers[layer_name] = layer_parity_metrics(
                    arm_outputs["image_only"][layer_name],
                    trt_reference[layer_name],
                )
            parity_layers["depth"]["unit"] = "m"
            trt_b_parity = {
                **trt_reference_identity,
                "layers": parity_layers,
            }
            run_receipt = {
                "contract": CONTRACT,
                "status": "passed",
                "interpretation": (
                    "Offline same-checkpoint conditioning diagnostic only; "
                    "authored-K evaluation is not independent metric truth."
                ),
                "source": {
                    "repo": str(source_repo),
                    "commit": source_head,
                    "hf_model_id": args.hf_model_id,
                    "hf_revision": args.hf_revision,
                    "checkpoint_status": artifacts["checkpoint_status"],
                    "hf_artifacts": hf_artifacts,
                },
                "implementation": {
                    "script": str(script_path.relative_to(repo_root)),
                    "script_sha256": sha256_file(script_path),
                    "fixed_corpus_evaluator_sha256": sha256_file(
                        repo_root
                        / "DS9"
                        / "scripts"
                        / "evaluate_mapanything_fixed_corpus.py"
                    ),
                    "exporter": {
                        "path": str(exporter_path.relative_to(repo_root)),
                        "sha256": sha256_file(exporter_path),
                    },
                    "mapanything_geometry": {
                        "path": str(geometry_path),
                        "sha256": sha256_file(geometry_path),
                    },
                    "mapanything_model": {
                        "path": str(model_source_path),
                        "sha256": sha256_file(model_source_path),
                    },
                },
                "inputs": {
                    "raw": {
                        "path": str(input_raw),
                        "sha256": sha256_file(input_raw),
                    },
                    "fixture_receipt": {
                        "path": str(fixture_receipt),
                        "sha256": sha256_file(fixture_receipt),
                    },
                    "annotations": {
                        "path": str(annotations_path),
                        "sha256": sha256_file(annotations_path),
                    },
                    "camera_order": camera_order,
                    "network_intrinsics_float32": network_k.tolist(),
                    "authored_rays_sha256": hashlib.sha256(
                        authored_rays_np.tobytes(order="C")
                    ).hexdigest(),
                    "ray_helper_parity": ray_helper_parity,
                    "fixture_profile": receipt.get("profile"),
                    "annotation_contract": annotations.get("contract"),
                },
                "execution": {
                    "device": str(next(model.parameters()).device),
                    "elapsed_s_including_load": elapsed_s,
                    "python_version": sys.version,
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "dependencies": dependency_versions(),
                },
                "ray_conditioning_contract": {
                    "classification": "optional_context_not_fixed_geometry",
                    "input_behavior": (
                        "The pinned model encodes supplied camera-frame unit rays "
                        "with ray_dirs_encoder and additively fuses those features."
                    ),
                    "output_behavior": (
                        "The dense prediction head emits a new unit ray map and "
                        "depth; the source has no copy or preservation constraint "
                        "tying output rays to supplied rays."
                    ),
                    "inference_configuration": (
                        "Each arm explicitly uses the model's deterministic "
                        "_configure_geometric_input_config inference controls."
                    ),
                },
                "trt_b_parity": trt_b_parity,
                "arms": output_records,
            }
            _write_json(partial / "run-receipt.json", run_receipt)
            os.rename(partial, output_dir)
        except Exception:
            if partial.exists():
                failure_path = partial / "FAILED"
                if not failure_path.exists():
                    failure_path.write_text("diagnostic failed\n", encoding="utf-8")
                    failure_path.chmod(0o600)
            raise
        finally:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    print(output_dir / "run-receipt.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DiagnosticError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
