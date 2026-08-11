#!/usr/bin/env python3
"""Join exact-engine local COCO quality and TensorRT B3 performance evidence.

Only local COCO evaluator metrics are selected. Vendor or checkpoint accuracy
metadata present in a source performance report is never copied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


COCO_SUMMARY_SCHEMA = "noesis.ds9.local-coco-head-to-head-suite.v1"
COCO_RUN_SCHEMA = "noesis.ds9.local-coco-head-to-head-run.v1"
COCO_EVALUATION_SCHEMA = "noesis.ds9.local-coco-person-evaluation.v2"
RF_PERFORMANCE_REPORT_SCHEMA = "noesis.ds9.rfdetr-performance-benchmark.v1"
RF_PERFORMANCE_MATRIX_SCHEMA = "noesis.ds9.rfdetr-performance-matrix.v1"
RF_ENGINE_BINDING_SCHEMA = "noesis.ds9.rfdetr-media-validation-trt.v1"
YOLO_PERFORMANCE_REPORT_SCHEMA = "noesis.ds9.yolo26-performance-benchmark.v2"
OUTPUT_SCHEMA = "noesis.ds9.local-coco-performance-comparison.v1"
BATCH_SIZE = 3
RF_AP_PARITY_THRESHOLD_POINTS = 0.1
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

EXPECTED_RF_MODEL_IDS = (
    "detect_nano",
    "detect_small",
    "detect_medium",
    "detect_large",
    "seg_nano",
    "seg_small",
    "seg_medium",
    "seg_large",
    "seg_xlarge",
    "seg_2xlarge",
    "keypoint_preview",
)

SUITES = {
    "rf_fp32": {
        "architecture": "rfdetr",
        "profile": "fp32_no_tf32",
        "prefix": "rfdetr_fp32_no_tf32_",
    },
    "rf_fp16": {
        "architecture": "rfdetr",
        "profile": "fp16_tf32",
        "prefix": "rfdetr_fp16_tf32_",
    },
    "yolo_fp16": {
        "architecture": "yolo26",
        "profile": "fp16_tf32",
        "prefix": "yolo26_fp16_tf32_",
    },
}

PRIMARY_METRIC = {
    "detection": "bbox",
    "segmentation": "segm",
    "keypoint": "keypoints",
}
FAMILY_ORDER = {"detection": 0, "segmentation": 1, "keypoint": 2}
VARIANT_ORDER = {
    "nano": 0,
    "n": 0,
    "small": 1,
    "s": 1,
    "medium": 2,
    "m": 2,
    "large": 3,
    "l": 3,
    "xlarge": 4,
    "x": 4,
    "2xlarge": 5,
    "preview": 6,
}

DELTA_FIELDS = (
    "rf_fp16_minus_fp32_ap_points",
    "rf_fp16_minus_fp32_ap50_points",
    "rf_fp16_minus_fp32_ap75_points",
    "rf_fp16_minus_fp32_b3_latency_p50_ms",
    "rf_fp16_latency_speedup_x",
    "rf_fp16_minus_fp32_throughput_work_items_s",
    "rf_fp16_throughput_gain_x",
    "rf_fp16_minus_fp32_vram_mib",
    "rf_fp16_minus_fp32_power_w",
    "rf_fp16_primary_ap_parity_pass",
)
CSV_FIELDS = (
    "architecture",
    "model_id",
    "family",
    "variant",
    "precision_profile",
    "engine_sha256",
    "accuracy_evaluation_sha256",
    "performance_evidence_sha256",
    "engine_binding_evidence_sha256",
    "resolution",
    "batch_size",
    "coco_metric",
    "coco_ap_pct",
    "coco_ap50_pct",
    "coco_ap75_pct",
    "b3_compute_latency_p50_ms",
    "b3_compute_latency_p95_ms",
    "engine_throughput_work_items_s",
    "throughput_unit",
    "process_vram_peak_mib",
    "power_mean_w",
    *DELTA_FIELDS,
)


class ComparisonError(RuntimeError):
    """Raised when evidence is incomplete, mismatched, or incomparable."""


@dataclass(frozen=True)
class Performance:
    rows: dict[str, dict[str, Any]]
    metadata: dict[str, Any]
    sources: tuple[dict[str, Any], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ComparisonError(f"{label} must be a JSON object: {path}")
    return payload


def _source(
    role: str,
    path: Path,
    payload: Mapping[str, Any],
    *,
    engine_bindings: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    result = {
        "role": role,
        "path": str(path),
        "sha256": _sha256(path),
        "schema": str(payload.get("schema") or ""),
    }
    if engine_bindings is not None:
        result["engine_sha256_by_model"] = dict(sorted(engine_bindings.items()))
    return result


def _sha_value(value: Any, label: str) -> str:
    result = str(value or "")
    if not SHA256_PATTERN.fullmatch(result):
        raise ComparisonError(f"{label} must be a lowercase SHA-256")
    return result


def _number(
    value: Any,
    label: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ComparisonError(f"{label} must be finite and >= {minimum}")
    if maximum is not None and result > maximum:
        raise ComparisonError(f"{label} must be <= {maximum}")
    return result


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ComparisonError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _metric_pct(stats: Mapping[str, Any], key: str, label: str) -> float:
    return _rounded(
        100.0 * _number(stats.get(key), label, maximum=1.0),
        4,
    )


def _canonical_id(suite: str, evaluation_id: str) -> str:
    prefix = str(SUITES[suite]["prefix"])
    if not evaluation_id.startswith(prefix) or evaluation_id == prefix:
        raise ComparisonError(
            f"{suite} evaluation id lacks prefix {prefix!r}: {evaluation_id!r}"
        )
    return evaluation_id[len(prefix) :]


def _evaluation_contract(
    *,
    path: Path,
    suite: str,
    evaluation_id: str,
    model_id: str,
    family: str,
    variant: str,
    resolution: int,
    corpus: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    report = _load(path, f"{suite}/{model_id} COCO evaluation")
    if report.get("schema") != COCO_EVALUATION_SCHEMA:
        raise ComparisonError(f"{suite}/{model_id} evaluation schema drifted")
    model = report.get("model")
    engine = report.get("engine")
    dataset = report.get("dataset")
    if not all(isinstance(value, Mapping) for value in (model, engine, dataset)):
        raise ComparisonError(f"{suite}/{model_id} evaluation contract is incomplete")
    assert isinstance(model, Mapping)
    assert isinstance(engine, Mapping)
    assert isinstance(dataset, Mapping)
    if (
        model.get("id") != evaluation_id
        or model.get("source_id") != model_id
        or model.get("architecture") != SUITES[suite]["architecture"]
        or model.get("family") != family
        or model.get("variant") != variant
        or model.get("precision_profile") != SUITES[suite]["profile"]
        or model.get("resolution") != resolution
    ):
        raise ComparisonError(f"{suite}/{model_id} evaluation identity drifted")
    engine_sha = _sha_value(
        engine.get("sha256"), f"{suite}/{model_id} evaluation engine"
    )
    expected_sha = model.get("expected_engine_sha256")
    if (
        expected_sha is not None
        and _sha_value(expected_sha, f"{suite}/{model_id} expected engine")
        != engine_sha
    ):
        raise ComparisonError(f"{suite}/{model_id} expected engine SHA mismatch")
    receipt = engine.get("receipt")
    if isinstance(receipt, Mapping):
        recorded = receipt.get("recorded_engine_sha256")
        if (
            recorded is not None
            and _sha_value(recorded, f"{suite}/{model_id} receipt engine") != engine_sha
        ):
            raise ComparisonError(f"{suite}/{model_id} receipt engine SHA mismatch")
    for key in (
        "instance_annotation_sha256",
        "keypoint_annotation_sha256",
        "image_count",
    ):
        if dataset.get(key) != corpus.get(key):
            raise ComparisonError(f"{suite}/{model_id} evaluation dataset drifted")
    return engine_sha, report


def _load_accuracy(
    summary_path: Path,
    suite: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], tuple[dict[str, Any], ...]]:
    summary = _load(summary_path, f"{suite} COCO summary")
    run_path = summary_path.parent / "run_contract.json"
    run = _load(run_path, f"{suite} COCO run contract")
    if (
        summary.get("schema") != COCO_SUMMARY_SCHEMA
        or summary.get("suite") != suite
        or run.get("schema") != COCO_RUN_SCHEMA
        or run.get("suite") != suite
    ):
        raise ComparisonError(f"{suite} COCO summary/run contract drifted")
    raw_models = summary.get("models")
    run_models = run.get("models")
    dataset = run.get("dataset")
    if not isinstance(raw_models, list) or not isinstance(run_models, list):
        raise ComparisonError(f"{suite} COCO model rows are missing")
    if not isinstance(dataset, Mapping):
        raise ComparisonError(f"{suite} COCO dataset contract is missing")
    if not raw_models or summary.get("model_count") != len(raw_models):
        raise ComparisonError(f"{suite} COCO model count drifted")
    run_by_id = {
        str(row.get("id")): row for row in run_models if isinstance(row, Mapping)
    }
    if len(run_by_id) != len(run_models):
        raise ComparisonError(f"{suite} COCO run model ids are invalid")
    corpus = {
        "instance_annotation_sha256": _sha_value(
            dataset.get("instance_annotation_sha256"),
            f"{suite} instance annotations",
        ),
        "keypoint_annotation_sha256": _sha_value(
            dataset.get("keypoint_annotation_sha256"),
            f"{suite} keypoint annotations",
        ),
        "image_count": _integer(dataset.get("image_count"), f"{suite} images", 1),
        "first_image_id": _integer(
            dataset.get("first_image_id"), f"{suite} first image", 1
        ),
        "last_image_id": _integer(
            dataset.get("last_image_id"), f"{suite} last image", 1
        ),
        "score_floor": _number(
            run.get("score_floor"), f"{suite} score floor", maximum=1.0
        ),
        "topk_per_image": _integer(run.get("topk_per_image"), f"{suite} topk", 1),
    }
    rows: dict[str, dict[str, Any]] = {}
    sources = [
        _source(f"{suite}_coco_summary", summary_path, summary),
        _source(f"{suite}_coco_run_contract", run_path, run),
    ]
    for raw in raw_models:
        if not isinstance(raw, Mapping):
            raise ComparisonError(f"{suite} COCO row must be an object")
        evaluation_id = str(raw.get("id") or "")
        model_id = _canonical_id(suite, evaluation_id)
        run_row = run_by_id.get(evaluation_id)
        if not isinstance(run_row, Mapping):
            raise ComparisonError(f"{suite}/{model_id} is absent from run contract")
        family = str(raw.get("family") or "")
        variant = str(raw.get("variant") or "")
        if family not in PRIMARY_METRIC or not variant:
            raise ComparisonError(f"{suite}/{model_id} identity is incomplete")
        resolution = _integer(
            run_row.get("resolution"), f"{suite}/{model_id} resolution", 1
        )
        if (
            run_row.get("source_id") != model_id
            or run_row.get("family") != family
            or run_row.get("variant") != variant
            or raw.get("precision_profile") != SUITES[suite]["profile"]
            or run_row.get("precision_profile") != SUITES[suite]["profile"]
        ):
            raise ComparisonError(f"{suite}/{model_id} run identity drifted")
        metric = PRIMARY_METRIC[family]
        metrics = raw.get("metrics")
        if not isinstance(metrics, Mapping) or not {"bbox", metric}.issubset(metrics):
            raise ComparisonError(f"{suite}/{model_id} lacks required COCO metrics")
        stats = metrics.get(metric)
        if not isinstance(stats, Mapping):
            raise ComparisonError(f"{suite}/{model_id}/{metric} stats are invalid")
        evaluation_path = (
            summary_path.parent / "models" / evaluation_id / "evaluation.json"
        )
        engine_sha, evaluation = _evaluation_contract(
            path=evaluation_path,
            suite=suite,
            evaluation_id=evaluation_id,
            model_id=model_id,
            family=family,
            variant=variant,
            resolution=resolution,
            corpus=corpus,
        )
        evaluation_metrics = (
            (evaluation.get("evaluation") or {}).get("metrics")
            if isinstance(evaluation.get("evaluation"), Mapping)
            else None
        )
        if not isinstance(evaluation_metrics, Mapping):
            raise ComparisonError(f"{suite}/{model_id} evaluation metrics are missing")
        evaluation_task = evaluation_metrics.get(metric)
        if (
            not isinstance(evaluation_task, Mapping)
            or evaluation_task.get("stats") != stats
        ):
            raise ComparisonError(
                f"{suite}/{model_id} summary/evaluation metrics differ"
            )
        if model_id in rows:
            raise ComparisonError(f"{suite} repeats model id: {model_id}")
        evaluation_digest = _sha256(evaluation_path)
        rows[model_id] = {
            "model_id": model_id,
            "family": family,
            "variant": variant,
            "resolution": resolution,
            "engine_sha256": engine_sha,
            "accuracy_evaluation_sha256": evaluation_digest,
            "coco_metric": metric,
            "coco_ap_pct": _metric_pct(stats, "ap", f"{suite}/{model_id}/{metric}/ap"),
            "coco_ap50_pct": _metric_pct(
                stats, "ap50", f"{suite}/{model_id}/{metric}/ap50"
            ),
            "coco_ap75_pct": _metric_pct(
                stats, "ap75", f"{suite}/{model_id}/{metric}/ap75"
            ),
        }
        sources.append(
            {
                **_source(
                    f"{suite}_{model_id}_coco_evaluation",
                    evaluation_path,
                    evaluation,
                ),
                "engine_sha256": engine_sha,
            }
        )
    if set(run_by_id) != {str(SUITES[suite]["prefix"]) + model_id for model_id in rows}:
        raise ComparisonError(f"{suite} run contains unevaluated models")
    return rows, corpus, tuple(sources)


def _companion_report(
    matrix_path: Path, matrix: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    report_path = matrix_path.parent / "performance_report.json"
    report = _load(report_path, "performance matrix companion report")
    if _sha_value(
        matrix.get("source_report_sha256"), "performance source report"
    ) != _sha256(report_path):
        raise ComparisonError("performance matrix companion report digest differs")
    return report_path, report


def _rf32_engine_bindings(
    path: Path,
    report: Mapping[str, Any],
) -> tuple[dict[str, str], tuple[dict[str, Any], ...], str]:
    manifest = _load(path, "RF FP32 engine binding manifest")
    if (
        manifest.get("schema") != RF_ENGINE_BINDING_SCHEMA
        or manifest.get("runtime_engine_profile") != "fp32_no_tf32"
    ):
        raise ComparisonError("RF FP32 engine binding manifest drifted")
    sources = report.get("sources")
    if not isinstance(sources, Mapping):
        raise ComparisonError("RF FP32 performance sources are missing")
    media_root = Path(str(sources.get("media_run") or "")).resolve()
    if media_root != path.parent.resolve():
        raise ComparisonError(
            "RF FP32 engine bindings are not from the benchmark media run"
        )
    run_path = media_root / "run.json"
    run = _load(run_path, "RF FP32 benchmark media manifest")
    if _sha_value(
        sources.get("media_manifest_sha256"), "RF FP32 media manifest"
    ) != _sha256(run_path):
        raise ComparisonError("RF FP32 benchmark media manifest digest differs")
    raw_models = manifest.get("models")
    if not isinstance(raw_models, list):
        raise ComparisonError("RF FP32 binding models are missing")
    bindings: dict[str, str] = {}
    for row in raw_models:
        if not isinstance(row, Mapping) or not isinstance(row.get("engine"), Mapping):
            raise ComparisonError("RF FP32 engine binding row is invalid")
        model_id = str(row.get("id") or "")
        engine = row["engine"]
        assert isinstance(engine, Mapping)
        if not model_id or model_id in bindings:
            raise ComparisonError("RF FP32 engine binding id is invalid")
        bindings[model_id] = _sha_value(
            engine.get("sha256"), f"RF FP32/{model_id} binding engine"
        )
    if tuple(bindings) != EXPECTED_RF_MODEL_IDS:
        raise ComparisonError("RF FP32 engine binding model order/coverage drifted")
    binding_digest = _sha256(path)
    return (
        bindings,
        (
            _source(
                "rf_fp32_engine_bindings",
                path,
                manifest,
                engine_bindings=bindings,
            ),
            _source("rf_fp32_benchmark_media_manifest", run_path, run),
        ),
        binding_digest,
    )


def _performance_number(
    row: Mapping[str, Any],
    nested: str,
    flattened: str,
    label: str,
) -> float:
    compute = row.get("compute")
    value = compute.get(nested) if isinstance(compute, Mapping) else row.get(flattened)
    return _number(value, label)


def _normalize_performance(
    row: Mapping[str, Any],
    *,
    suite: str,
    input_digest: str,
    binding_sha: str | None,
) -> dict[str, Any]:
    model_id = str(row.get("model_id") or "")
    family = str(row.get("family") or "")
    variant = str(row.get("variant") or "")
    if not model_id or family not in PRIMARY_METRIC or not variant:
        raise ComparisonError(f"{suite} performance model identity is invalid")
    workload = str(row.get("workload_unit") or "full_frames")
    unit = "person_crops/s" if workload == "person_crops" else "frames/s"
    return {
        "model_id": model_id,
        "family": family,
        "variant": variant,
        "resolution": _integer(
            row.get("resolution"), f"{suite}/{model_id} performance resolution", 1
        ),
        "batch_size": _integer(
            row.get("batch_size"), f"{suite}/{model_id} performance batch", 1
        ),
        "engine_sha256": _sha_value(
            row.get("engine_sha256"), f"{suite}/{model_id} performance engine"
        ),
        "performance_evidence_sha256": input_digest,
        "engine_binding_evidence_sha256": binding_sha,
        "b3_compute_latency_p50_ms": _rounded(
            _performance_number(
                row,
                "batch_latency_p50_ms",
                "compute_batch_latency_p50_ms",
                f"{suite}/{model_id} p50 latency",
            )
        ),
        "b3_compute_latency_p95_ms": _rounded(
            _performance_number(
                row,
                "batch_latency_p95_ms",
                "compute_batch_latency_p95_ms",
                f"{suite}/{model_id} p95 latency",
            )
        ),
        "engine_throughput_work_items_s": _rounded(
            _performance_number(
                row,
                "throughput_images_s",
                "compute_throughput_images_s",
                f"{suite}/{model_id} throughput",
            )
        ),
        "throughput_unit": unit,
        "process_vram_peak_mib": _rounded(
            _performance_number(
                row,
                "process_vram_peak_mib",
                "compute_process_vram_peak_mib",
                f"{suite}/{model_id} VRAM",
            )
        ),
        "power_mean_w": _rounded(
            _performance_number(
                row,
                "power_mean_w",
                "compute_power_mean_w",
                f"{suite}/{model_id} power",
            )
        ),
    }


def _performance_metadata(report: Mapping[str, Any], suite: str) -> dict[str, Any]:
    if report.get("status") != "passed" or report.get("batch_size") != BATCH_SIZE:
        raise ComparisonError(f"{suite} performance report did not pass B3")
    environment = report.get("environment")
    protocol = report.get("protocol")
    if not isinstance(environment, Mapping) or not isinstance(protocol, Mapping):
        raise ComparisonError(f"{suite} performance metadata is incomplete")
    gpu = environment.get("gpu")
    if not isinstance(gpu, Mapping):
        raise ComparisonError(f"{suite} GPU metadata is incomplete")
    return {
        "environment": {
            "gpu_name": str(gpu.get("name") or ""),
            "gpu_uuid": str(gpu.get("uuid") or ""),
            "driver_version": str(gpu.get("driver_version") or ""),
            "image_id": str(environment.get("image_id") or ""),
            "tensorrt_version": str(environment.get("tensorrt_version") or ""),
        },
        "protocol": {
            key: protocol.get(key)
            for key in (
                "compute_lane",
                "inference_streams",
                "primary_duration_s",
                "primary_rounds",
                "spin_wait",
                "warmup_ms",
            )
        },
    }


def _validate_precision(
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
    suite: str,
) -> None:
    if suite == "rf_fp32":
        if report.get("precision") != "fp32" or report.get("tf32_enabled") is not False:
            raise ComparisonError("RF FP32 performance precision drifted")
    elif suite == "rf_fp16":
        if report.get("precision") != "fp16" or report.get("tf32_enabled") is not True:
            raise ComparisonError("RF FP16 performance precision drifted")
        if payload.get("schema") == RF_PERFORMANCE_MATRIX_SCHEMA:
            rows = payload.get("rows")
            if not isinstance(rows, list) or any(
                not isinstance(row, Mapping)
                or row.get("precision") != "fp16"
                or row.get("tf32_enabled") is not True
                or row.get("runtime_engine_profile") != "fp16_tf32"
                for row in rows
            ):
                raise ComparisonError("RF FP16 performance matrix precision drifted")
    else:
        if (
            "fp16" not in str(report.get("precision") or "").lower()
            or "default" not in str(report.get("tf32_build_setting") or "").lower()
        ):
            raise ComparisonError("YOLO FP16+TF32 performance precision drifted")


def _load_performance(
    path: Path,
    suite: str,
    *,
    rf32_binding_path: Path | None = None,
) -> Performance:
    payload = _load(path, f"{suite} performance input")
    schema = payload.get("schema")
    sources = [_source(f"{suite}_performance", path, payload)]
    if schema == RF_PERFORMANCE_MATRIX_SCHEMA:
        if not suite.startswith("rf_"):
            raise ComparisonError("RF performance matrix used for YOLO")
        report_path, report = _companion_report(path, payload)
        sources.append(_source(f"{suite}_performance_report", report_path, report))
        raw_rows = payload.get("rows")
    elif schema in {RF_PERFORMANCE_REPORT_SCHEMA, YOLO_PERFORMANCE_REPORT_SCHEMA}:
        report = payload
        raw_rows = report.get("models")
        if suite.startswith("rf_") != (schema == RF_PERFORMANCE_REPORT_SCHEMA):
            raise ComparisonError(f"{suite} performance family/schema mismatch")
    else:
        raise ComparisonError(f"unsupported performance schema: {schema!r}")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ComparisonError(f"{suite} performance rows are missing")
    _validate_precision(payload, report, suite)
    binding_sha = None
    if suite == "rf_fp32":
        if rf32_binding_path is None:
            raise ComparisonError("RF FP32 engine binding manifest is required")
        bindings, binding_sources, binding_sha = _rf32_engine_bindings(
            rf32_binding_path, report
        )
        sources.extend(binding_sources)
    else:
        bindings = {}
    input_digest = _sha256(path)
    rows: dict[str, dict[str, Any]] = {}
    for original in raw_rows:
        if not isinstance(original, Mapping):
            raise ComparisonError(f"{suite} performance row must be an object")
        row = dict(original)
        model_id = str(row.get("model_id") or "")
        if suite == "rf_fp32":
            recorded = row.get("engine_sha256")
            bound = bindings.get(model_id)
            if bound is None:
                raise ComparisonError(f"RF FP32/{model_id} lacks an engine binding")
            if (
                recorded is not None
                and _sha_value(recorded, f"RF FP32/{model_id} report engine") != bound
            ):
                raise ComparisonError(f"RF FP32/{model_id} engine bindings disagree")
            row["engine_sha256"] = bound
        normalized = _normalize_performance(
            row,
            suite=suite,
            input_digest=input_digest,
            binding_sha=binding_sha,
        )
        if normalized["model_id"] in rows:
            raise ComparisonError(f"{suite} repeats performance model: {model_id}")
        rows[model_id] = normalized
    return Performance(rows, _performance_metadata(report, suite), tuple(sources))


def _equal(label: str, values: Mapping[str, Any]) -> Any:
    encoded = {json.dumps(value, sort_keys=True) for value in values.values()}
    if len(encoded) != 1:
        raise ComparisonError(f"{label} does not match across suites")
    return next(iter(values.values()))


def _join(
    suite: str,
    accuracy: Mapping[str, Mapping[str, Any]],
    performance: Performance,
) -> list[dict[str, Any]]:
    if set(accuracy) != set(performance.rows):
        raise ComparisonError(f"{suite} quality/performance model coverage differs")
    rows = []
    for model_id, quality in accuracy.items():
        perf = performance.rows[model_id]
        if (
            quality["family"] != perf["family"]
            or quality["variant"] != perf["variant"]
            or quality["resolution"] != perf["resolution"]
            or perf["batch_size"] != BATCH_SIZE
        ):
            raise ComparisonError(f"{suite}/{model_id} identity differs")
        if quality["engine_sha256"] != perf["engine_sha256"]:
            raise ComparisonError(
                f"{suite}/{model_id} accuracy/performance engine SHA mismatch: "
                f"{quality['engine_sha256']} != {perf['engine_sha256']}"
            )
        rows.append(
            {
                "architecture": SUITES[suite]["architecture"],
                "model_id": model_id,
                "family": quality["family"],
                "variant": quality["variant"],
                "precision_profile": SUITES[suite]["profile"],
                "engine_sha256": quality["engine_sha256"],
                "accuracy_evaluation_sha256": quality["accuracy_evaluation_sha256"],
                "performance_evidence_sha256": perf["performance_evidence_sha256"],
                "engine_binding_evidence_sha256": perf[
                    "engine_binding_evidence_sha256"
                ],
                "resolution": quality["resolution"],
                "batch_size": BATCH_SIZE,
                "coco_metric": quality["coco_metric"],
                "coco_ap_pct": quality["coco_ap_pct"],
                "coco_ap50_pct": quality["coco_ap50_pct"],
                "coco_ap75_pct": quality["coco_ap75_pct"],
                "b3_compute_latency_p50_ms": perf["b3_compute_latency_p50_ms"],
                "b3_compute_latency_p95_ms": perf["b3_compute_latency_p95_ms"],
                "engine_throughput_work_items_s": perf[
                    "engine_throughput_work_items_s"
                ],
                "throughput_unit": perf["throughput_unit"],
                "process_vram_peak_mib": perf["process_vram_peak_mib"],
                "power_mean_w": perf["power_mean_w"],
                **{field: None for field in DELTA_FIELDS},
            }
        )
    return rows


def _rf_deltas(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = {}
    for profile in ("fp32_no_tf32", "fp16_tf32"):
        pairs[profile] = {
            str(row["model_id"]): row
            for row in rows
            if row["architecture"] == "rfdetr" and row["precision_profile"] == profile
        }
    if (
        tuple(pairs["fp32_no_tf32"]) != EXPECTED_RF_MODEL_IDS
        or tuple(pairs["fp16_tf32"]) != EXPECTED_RF_MODEL_IDS
    ):
        raise ComparisonError("RF parity requires all current 11 models in order")
    deltas = []
    for model_id in EXPECTED_RF_MODEL_IDS:
        fp32 = pairs["fp32_no_tf32"][model_id]
        fp16 = pairs["fp16_tf32"][model_id]
        if (
            fp32["family"] != fp16["family"]
            or fp32["variant"] != fp16["variant"]
            or fp32["coco_metric"] != fp16["coco_metric"]
        ):
            raise ComparisonError(f"RF precision pair identity differs: {model_id}")
        ap_delta = _rounded(fp16["coco_ap_pct"] - fp32["coco_ap_pct"], 4)
        parity = abs(ap_delta) <= RF_AP_PARITY_THRESHOLD_POINTS + 1e-12
        delta = {
            "model_id": model_id,
            "family": fp16["family"],
            "variant": fp16["variant"],
            "coco_metric": fp16["coco_metric"],
            "rf_fp16_minus_fp32_ap_points": ap_delta,
            "rf_fp16_minus_fp32_ap50_points": _rounded(
                fp16["coco_ap50_pct"] - fp32["coco_ap50_pct"], 4
            ),
            "rf_fp16_minus_fp32_ap75_points": _rounded(
                fp16["coco_ap75_pct"] - fp32["coco_ap75_pct"], 4
            ),
            "rf_fp16_minus_fp32_b3_latency_p50_ms": _rounded(
                fp16["b3_compute_latency_p50_ms"] - fp32["b3_compute_latency_p50_ms"]
            ),
            "rf_fp16_latency_speedup_x": _rounded(
                fp32["b3_compute_latency_p50_ms"] / fp16["b3_compute_latency_p50_ms"]
            ),
            "rf_fp16_minus_fp32_throughput_work_items_s": _rounded(
                fp16["engine_throughput_work_items_s"]
                - fp32["engine_throughput_work_items_s"]
            ),
            "rf_fp16_throughput_gain_x": _rounded(
                fp16["engine_throughput_work_items_s"]
                / fp32["engine_throughput_work_items_s"]
            ),
            "rf_fp16_minus_fp32_vram_mib": _rounded(
                fp16["process_vram_peak_mib"] - fp32["process_vram_peak_mib"]
            ),
            "rf_fp16_minus_fp32_power_w": _rounded(
                fp16["power_mean_w"] - fp32["power_mean_w"]
            ),
            "rf_fp16_primary_ap_parity_pass": parity,
        }
        for field in DELTA_FIELDS:
            fp16[field] = delta[field]
        deltas.append(delta)
    passed = sum(bool(row["rf_fp16_primary_ap_parity_pass"]) for row in deltas)
    acceptance = {
        "status": "passed" if passed == len(EXPECTED_RF_MODEL_IDS) else "failed",
        "criterion": "abs(FP16 primary AP - FP32 primary AP) <= threshold",
        "threshold_ap_points": RF_AP_PARITY_THRESHOLD_POINTS,
        "required_model_count": len(EXPECTED_RF_MODEL_IDS),
        "evaluated_model_count": len(deltas),
        "passed_model_count": passed,
        "failed_model_count": len(deltas) - passed,
        "failed_models": [
            row["model_id"]
            for row in deltas
            if not row["rf_fp16_primary_ap_parity_pass"]
        ],
    }
    return deltas, acceptance


def _sort_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            FAMILY_ORDER[str(row["family"])],
            0 if row["architecture"] == "rfdetr" else 1,
            VARIANT_ORDER.get(str(row["variant"]), 99),
            str(row["model_id"]),
            0 if row["precision_profile"] == "fp32_no_tf32" else 1,
        ),
    )


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _markdown(
    rows: Sequence[Mapping[str, Any]],
    deltas: Sequence[Mapping[str, Any]],
    corpus: Mapping[str, Any],
    parity: Mapping[str, Any],
    caveats: Sequence[str],
) -> str:
    lines = [
        "# Local RF-DETR vs YOLO26 comparison",
        "",
        (
            "**Construction status: passed.** This means source, schema, corpus, "
            "engine-SHA, model-coverage, precision, B3, environment, and protocol "
            "contracts validated. It is not an accuracy acceptance or a claim "
            "that one architecture dominates."
        ),
        "",
        (
            f"Accuracy uses local COCO person-only evaluation over "
            f"{corpus['image_count']} images. AP values are percentage points. "
            "No vendor or published-checkpoint accuracy is used."
        ),
        "",
        (
            f"**RF FP16 primary-AP parity: {str(parity['status']).upper()}** — "
            f"{parity['passed_model_count']}/{parity['required_model_count']} "
            f"models are within ±{parity['threshold_ap_points']} AP point."
        ),
        "",
    ]
    names = {"rfdetr": "RF-DETR", "yolo26": "YOLO26"}
    labels = {"bbox": "BBox", "segm": "Mask", "keypoints": "Keypoint"}
    for family in ("detection", "segmentation", "keypoint"):
        family_rows = [row for row in rows if row["family"] == family]
        label = labels[PRIMARY_METRIC[family]]
        lines.extend(
            [
                f"## {family.title()}",
                "",
                (
                    f"| Model | Precision | Input | {label} AP | AP50 | AP75 | "
                    "B3 p50 ms | Throughput | VRAM MiB | Power W |"
                ),
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in family_rows:
            precision = (
                "FP32 no TF32"
                if row["precision_profile"] == "fp32_no_tf32"
                else "FP16 + TF32"
            )
            throughput = (
                f"{_fmt(row['engine_throughput_work_items_s'], 1)} "
                f"{row['throughput_unit']}"
            )
            values = (
                f"{names[str(row['architecture'])]} {row['variant']}",
                precision,
                str(row["resolution"]),
                _fmt(row["coco_ap_pct"]),
                _fmt(row["coco_ap50_pct"]),
                _fmt(row["coco_ap75_pct"]),
                _fmt(row["b3_compute_latency_p50_ms"], 3),
                throughput,
                _fmt(row["process_vram_peak_mib"], 1),
                _fmt(row["power_mean_w"], 1),
            )
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    lines.extend(
        [
            "## RF-DETR FP16 vs FP32",
            "",
            (
                "| Model | Metric | ΔAP pt | Parity | ΔB3 p50 ms | "
                "Latency speedup | Throughput gain | ΔVRAM MiB | ΔPower W |"
            ),
            "|---|---|---:|:---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in deltas:
        values = (
            f"RF-DETR {row['variant']}",
            str(row["coco_metric"]),
            _fmt(row["rf_fp16_minus_fp32_ap_points"]),
            "pass" if row["rf_fp16_primary_ap_parity_pass"] else "FAIL",
            _fmt(row["rf_fp16_minus_fp32_b3_latency_p50_ms"], 3),
            f"{_fmt(row['rf_fp16_latency_speedup_x'], 3)}x",
            f"{_fmt(row['rf_fp16_throughput_gain_x'], 3)}x",
            _fmt(row["rf_fp16_minus_fp32_vram_mib"], 1),
            _fmt(row["rf_fp16_minus_fp32_power_w"], 1),
        )
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {item}" for item in caveats)
    lines.append("")
    return "\n".join(lines)


def _write(path: Path, content: str) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf-fp32-coco-summary", type=Path, required=True)
    parser.add_argument("--rf-fp16-coco-summary", type=Path, required=True)
    parser.add_argument("--yolo-fp16-coco-summary", type=Path, required=True)
    parser.add_argument("--rf-fp32-performance", type=Path, required=True)
    parser.add_argument("--rf-fp32-engine-bindings", type=Path, required=True)
    parser.add_argument("--rf-fp16-performance", type=Path, required=True)
    parser.add_argument("--yolo-fp16-performance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    os.umask(0o077)
    summaries = {
        "rf_fp32": args.rf_fp32_coco_summary.resolve(),
        "rf_fp16": args.rf_fp16_coco_summary.resolve(),
        "yolo_fp16": args.yolo_fp16_coco_summary.resolve(),
    }
    performance_paths = {
        "rf_fp32": args.rf_fp32_performance.resolve(),
        "rf_fp16": args.rf_fp16_performance.resolve(),
        "yolo_fp16": args.yolo_fp16_performance.resolve(),
    }
    try:
        accuracy = {}
        corpora = {}
        performance = {}
        sources = []
        for suite in ("rf_fp32", "rf_fp16", "yolo_fp16"):
            accuracy[suite], corpora[suite], accuracy_sources = _load_accuracy(
                summaries[suite], suite
            )
            performance[suite] = _load_performance(
                performance_paths[suite],
                suite,
                rf32_binding_path=(
                    args.rf_fp32_engine_bindings.resolve()
                    if suite == "rf_fp32"
                    else None
                ),
            )
            sources.extend(accuracy_sources)
            sources.extend(performance[suite].sources)
        corpus = _equal("local COCO corpus/protocol", corpora)
        environment = _equal(
            "performance environment",
            {
                suite: item.metadata["environment"]
                for suite, item in performance.items()
            },
        )
        protocol = _equal(
            "performance protocol",
            {suite: item.metadata["protocol"] for suite, item in performance.items()},
        )
        rows = []
        for suite in ("rf_fp32", "rf_fp16", "yolo_fp16"):
            rows.extend(_join(suite, accuracy[suite], performance[suite]))
        deltas, parity = _rf_deltas(rows)
        rows = _sort_rows(rows)
        caveats = [
            (
                "Comparison status=passed means matrix construction and contract "
                "validation passed; RF precision acceptance is reported separately."
            ),
            (
                "Every row is joined only when the per-model COCO evaluation and "
                "performance evidence carry the same exact engine SHA-256."
            ),
            (
                "Accuracy is local COCO person-category quality from exact TensorRT "
                "engine outputs, not vendor checkpoint accuracy; it excludes the "
                "complete DeepStream parser, tracker, and camera pipeline."
            ),
            (
                "Performance is the engine-only compute lane with real B3 inputs; "
                "it excludes transfers, runtime preprocessing, clustering/NMS, "
                "tracking, OSD, metadata, and transport. It is not camera FPS."
            ),
            (
                "Input resolutions and architectures differ. The RF parity gate "
                "does not evaluate or assert YOLO-vs-RF dominance."
            ),
            (
                "YOLO26 pose quality is standalone full-frame inference, while "
                "performance measures its deployed SGIE B3 person-crop role; "
                "person_crops/s is not full-frame keypoint FPS."
            ),
            (
                "RF latency speedup is FP32 p50 divided by FP16 p50; above 1 means "
                "FP16 is faster. Throughput gain is FP16 divided by FP32."
            ),
            (
                "RF AP deltas are FP16 minus FP32 percentage points; parity uses "
                "the absolute primary-task AP delta."
            ),
        ]
        payload = {
            "schema": OUTPUT_SCHEMA,
            "status": "passed",
            "status_scope": "matrix_construction_and_contract_validation_only",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "accuracy": {
                "source": "local_coco_engine_evaluation",
                "published_accuracy_imported": False,
                "category": "person",
                "unit": "percentage_points",
                "corpus_contract": corpus,
            },
            "performance": {
                "lane": "engine_only_compute",
                "batch_size": BATCH_SIZE,
                "latency_unit": "milliseconds_per_batch",
                "environment": environment,
                "protocol": protocol,
            },
            "acceptance": {
                "rf_fp16_primary_ap_parity": parity,
                "yolo_vs_rfdetr_dominance": {
                    "status": "not_evaluated",
                    "reason": (
                        "RF precision parity is independent of cross-architecture "
                        "quality/performance tradeoffs."
                    ),
                },
            },
            "columns": list(CSV_FIELDS),
            "rows": rows,
            "rf_fp16_vs_fp32": deltas,
            "sources": sources,
            "caveats": caveats,
        }
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        json_path = output_dir / "comparison_matrix.json"
        csv_path = output_dir / "comparison_matrix.csv"
        markdown_path = output_dir / "comparison_matrix.md"
        _write(
            json_path,
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        _write(csv_path, buffer.getvalue())
        _write(markdown_path, _markdown(rows, deltas, corpus, parity, caveats))
        print(f"[PASS] construction_json={json_path}")
        print(f"[{str(parity['status']).upper()}] rf_fp16_primary_ap_parity")
        print(f"[PASS] csv={csv_path}")
        print(f"[PASS] markdown={markdown_path}")
        return 0
    except (ComparisonError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
