#!/usr/bin/env python3
"""Validate a DA3Metric-Large TensorRT result against its PyTorch reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


EXPECTED_OUTPUTS = ("depth", "conf", "mask")
EXPECTED_SHAPE = (3, 1, 294, 518)
DEPTH_MAE_LIMIT = 0.03
DEPTH_RMSE_LIMIT = 0.04
DEPTH_MAX_ABS_LIMIT = 0.25
DEPTH_CORRELATION_MIN = 0.999
BINARY_AGREEMENT_MIN = 0.999


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _load_trt_outputs(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("TensorRT output JSON must be a list")
    outputs: dict[str, np.ndarray] = {}
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("TensorRT output row must be a mapping")
        name = str(row.get("name") or "")
        dimensions = tuple(int(value) for value in str(row["dimensions"]).split("x"))
        values = np.asarray(row.get("values"), dtype=np.float32)
        outputs[name] = values.reshape(dimensions)
    if set(outputs) != set(EXPECTED_OUTPUTS):
        raise ValueError(
            f"TensorRT output names mismatch: expected={EXPECTED_OUTPUTS} "
            f"actual={tuple(outputs)}"
        )
    return outputs


def _comparison(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    error = np.abs(reference - actual)
    denominator = np.maximum(np.abs(reference), 1e-6)
    return {
        "shape": list(actual.shape),
        "finite_count": int(np.count_nonzero(np.isfinite(actual))),
        "positive_count": int(np.count_nonzero(actual > 0.0)),
        "min": float(np.min(actual)),
        "max": float(np.max(actual)),
        "mae": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "max_abs": float(np.max(error)),
        "mean_relative": float(np.mean(error / denominator)),
        "exact_fraction": float(np.mean(reference == actual)),
    }


def validate(args: argparse.Namespace) -> bool:
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in vars(args).items()
        if name
        in {
            "engine",
            "onnx",
            "export_receipt",
            "reference",
            "trt_output",
            "trt_times",
            "layer_info",
            "receipt",
        }
    }
    for name, path in paths.items():
        if name == "receipt":
            continue
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{name} is not a regular file: {path}")

    export_receipt = json.loads(paths["export_receipt"].read_text(encoding="utf-8"))
    expected_onnx_sha = str((export_receipt.get("onnx") or {}).get("sha256") or "")
    actual_onnx_sha = _sha256_file(paths["onnx"])
    if not expected_onnx_sha or actual_onnx_sha != expected_onnx_sha:
        raise ValueError("ONNX digest does not match the official export receipt")

    with np.load(paths["reference"], allow_pickle=False) as loaded:
        reference = {
            name: np.asarray(loaded[name], dtype=np.float32)
            for name in EXPECTED_OUTPUTS
        }
    actual = _load_trt_outputs(paths["trt_output"])
    comparisons = {
        name: _comparison(reference[name], actual[name])
        for name in EXPECTED_OUTPUTS
    }
    depth_correlation = float(
        np.corrcoef(reference["depth"].ravel(), actual["depth"].ravel())[0, 1]
    )
    binary_agreement = {
        name: float(np.mean(reference[name] == actual[name]))
        for name in ("conf", "mask")
    }

    timing_rows = json.loads(paths["trt_times"].read_text(encoding="utf-8"))
    if not isinstance(timing_rows, list) or not timing_rows:
        raise ValueError("TensorRT timing JSON has no inference rows")
    timing_summary: dict[str, Any] = {"queries": len(timing_rows)}
    for key in ("computeMs", "latencyMs", "h2dMs", "d2hMs"):
        values = np.asarray([float(row[key]) for row in timing_rows], dtype=np.float64)
        timing_summary[key] = {
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        }

    layer_info = json.loads(paths["layer_info"].read_text(encoding="utf-8"))
    layers = layer_info.get("Layers") if isinstance(layer_info, dict) else None
    bindings = layer_info.get("Bindings") if isinstance(layer_info, dict) else None
    if not isinstance(layers, list):
        raise ValueError("TensorRT layer-info JSON has no Layers list")
    compute_layers = [
        row
        for row in layers
        if isinstance(row, dict)
        and str(row.get("LayerType") or "") not in {"Constant", "Reformat"}
    ]
    half_output_layers = sum(
        1
        for row in compute_layers
        if any(
            str(output.get("Format/Datatype") or "") == "Half"
            for output in row.get("Outputs", [])
            if isinstance(output, dict)
        )
    )

    checks = {
        "bindings_exact": bindings == ["images", "depth", "conf", "mask"],
        "shapes_exact": all(actual[name].shape == EXPECTED_SHAPE for name in EXPECTED_OUTPUTS),
        "depth_all_finite": bool(np.all(np.isfinite(actual["depth"]))),
        "depth_all_positive": bool(np.all(actual["depth"] > 0.0)),
        "depth_mae": comparisons["depth"]["mae"] <= DEPTH_MAE_LIMIT,
        "depth_rmse": comparisons["depth"]["rmse"] <= DEPTH_RMSE_LIMIT,
        "depth_max_abs": comparisons["depth"]["max_abs"] <= DEPTH_MAX_ABS_LIMIT,
        "depth_correlation": depth_correlation >= DEPTH_CORRELATION_MIN,
        "conf_binary": bool(np.all(np.isin(actual["conf"], (0.0, 1.0)))),
        "mask_binary": bool(np.all(np.isin(actual["mask"], (0.0, 1.0)))),
        "conf_mask_identical": bool(np.array_equal(actual["conf"], actual["mask"])),
        "conf_agreement": binary_agreement["conf"] >= BINARY_AGREEMENT_MIN,
        "mask_agreement": binary_agreement["mask"] >= BINARY_AGREEMENT_MIN,
        "fp16_compute_present": half_output_layers > 0,
    }
    passed = all(checks.values())
    receipt = {
        "contract": "noesis.da3metric_large.tensorrt_validation.v1",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "target": {
            "gpu": args.gpu,
            "compute_capability": args.compute_capability,
            "tensorrt": args.tensorrt,
            "build_image": args.build_image,
            "build_image_id": args.build_image_id,
            "profile": "images:3x3x294x518",
            "precision": "FP32_IO_FP16_ENABLED",
        },
        "artifacts": {
            "engine": _file_record(paths["engine"]),
            "onnx": _file_record(paths["onnx"]),
            "export_receipt": _file_record(paths["export_receipt"]),
            "pytorch_reference": _file_record(paths["reference"]),
            "trt_output": _file_record(paths["trt_output"]),
            "trt_times": _file_record(paths["trt_times"]),
            "trt_layer_info": _file_record(paths["layer_info"]),
        },
        "comparison": {
            "outputs": comparisons,
            "depth_correlation": depth_correlation,
            "binary_agreement": binary_agreement,
        },
        "engine_inspection": {
            "bindings": bindings,
            "layer_count": len(layers),
            "compute_layer_count": len(compute_layers),
            "compute_layers_with_fp16_output": half_output_layers,
        },
        "timing": timing_summary,
        "thresholds": {
            "depth_mae_max": DEPTH_MAE_LIMIT,
            "depth_rmse_max": DEPTH_RMSE_LIMIT,
            "depth_max_abs_max": DEPTH_MAX_ABS_LIMIT,
            "depth_correlation_min": DEPTH_CORRELATION_MIN,
            "binary_agreement_min": BINARY_AGREEMENT_MIN,
        },
        "checks": checks,
    }
    receipt_path = paths["receipt"]
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(receipt_path, 0o600)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return passed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--trt-output", type=Path, required=True)
    parser.add_argument("--trt-times", type=Path, required=True)
    parser.add_argument("--layer-info", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gpu", default="NVIDIA GeForce RTX 3060 12GB")
    parser.add_argument("--compute-capability", default="8.6")
    parser.add_argument("--tensorrt", default="10.14.1.48")
    parser.add_argument("--build-image", default="noesis-ds9-dev:9.0-20260710")
    parser.add_argument(
        "--build-image-id",
        default=(
            "sha256:7476b1021376cd67793c95d949cdc7d46eef7704ab98a5a76feed461e4f907a4"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return 0 if validate(_parse_args(argv)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
