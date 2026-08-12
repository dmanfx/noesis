from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "DS9" / "scripts" / "build_local_model_comparison.py"
SPEC = importlib.util.spec_from_file_location("build_local_model_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


RF_MODELS = (
    ("detect_nano", "detection", "nano", 384),
    ("detect_small", "detection", "small", 512),
    ("detect_medium", "detection", "medium", 576),
    ("detect_large", "detection", "large", 704),
    ("seg_nano", "segmentation", "nano", 312),
    ("seg_small", "segmentation", "small", 384),
    ("seg_medium", "segmentation", "medium", 432),
    ("seg_large", "segmentation", "large", 504),
    ("seg_xlarge", "segmentation", "xlarge", 624),
    ("seg_2xlarge", "segmentation", "2xlarge", 768),
    ("keypoint_preview", "keypoint", "preview", 576),
)
YOLO_MODELS = (
    ("yolo26_detect_n", "detection", "n", 640),
    ("yolo26_seg_n", "segmentation", "n", 640),
    ("yolo26_pose_n", "keypoint", "n", 640),
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coco_stats(ap: float) -> dict[str, float]:
    return {"ap": ap, "ap50": ap + 0.1, "ap75": ap - 0.1}


def _engine_sha(suite: str, model_id: str) -> str:
    return hashlib.sha256(f"{suite}:{model_id}".encode()).hexdigest()


def _write_accuracy_suite(
    root: Path,
    suite: str,
    models: tuple[tuple[str, str, str, int], ...],
    *,
    ap: float,
) -> Path:
    prefixes = {
        "rf_fp32": "rfdetr_fp32_no_tf32_",
        "rf_fp16": "rfdetr_fp16_tf32_",
        "yolo_fp16": "yolo26_fp16_tf32_",
    }
    profiles = {
        "rf_fp32": "fp32_no_tf32",
        "rf_fp16": "fp16_tf32",
        "yolo_fp16": "fp16_tf32",
    }
    run_models = []
    summary_models = []
    for model_id, family, variant, resolution in models:
        evaluation_id = prefixes[suite] + model_id
        engine_sha = _engine_sha(suite, model_id)
        run_model = {
            "id": evaluation_id,
            "source_id": model_id,
            "architecture": "rfdetr" if suite.startswith("rf_") else "yolo26",
            "family": family,
            "variant": variant,
            "precision_profile": profiles[suite],
            "resolution": resolution,
            "expected_engine_sha256": (engine_sha if suite == "yolo_fp16" else None),
        }
        run_models.append(run_model)
        metrics = {"bbox": _coco_stats(ap)}
        if family == "segmentation":
            metrics["segm"] = _coco_stats(ap)
        elif family == "keypoint":
            metrics["keypoints"] = _coco_stats(ap)
        summary_models.append(
            {
                "id": evaluation_id,
                "family": family,
                "variant": variant,
                "precision_profile": profiles[suite],
                "prediction_count": 10,
                "metrics": metrics,
                "wall_seconds": 1.0,
            }
        )
        evaluation_metrics = {
            metric: {"stats": stats} for metric, stats in metrics.items()
        }
        _write_json(
            root / suite / "models" / evaluation_id / "evaluation.json",
            {
                "schema": "noesis.ds9.local-coco-person-evaluation.v2",
                "model": run_model,
                "engine": {
                    "sha256": engine_sha,
                    "size_bytes": 123,
                    "receipt": (
                        {
                            "recorded_engine_sha256": engine_sha,
                            "sha256": hashlib.sha256(
                                f"receipt:{suite}:{model_id}".encode()
                            ).hexdigest(),
                        }
                        if suite.startswith("rf_")
                        else None
                    ),
                },
                "dataset": {
                    "instance_annotation_sha256": "a" * 64,
                    "keypoint_annotation_sha256": "b" * 64,
                    "image_count": 5000,
                    "person_only": True,
                },
                "evaluation": {"metrics": evaluation_metrics},
            },
        )
    suite_root = root / suite
    _write_json(
        suite_root / "run_contract.json",
        {
            "schema": "noesis.ds9.local-coco-head-to-head-run.v1",
            "suite": suite,
            "models": run_models,
            "dataset": {
                "instance_annotation_sha256": "a" * 64,
                "keypoint_annotation_sha256": "b" * 64,
                "image_count": 5000,
                "first_image_id": 139,
                "last_image_id": 581929,
            },
            "score_floor": 0.001,
            "topk_per_image": 100,
        },
    )
    summary_path = suite_root / "suite_summary.json"
    _write_json(
        summary_path,
        {
            "schema": "noesis.ds9.local-coco-head-to-head-suite.v1",
            "suite": suite,
            "model_count": len(summary_models),
            "models": summary_models,
        },
    )
    return summary_path


def _environment() -> dict[str, object]:
    return {
        "gpu": {
            "name": "fixture GPU",
            "uuid": "GPU-fixture",
            "driver_version": "1.2.3",
        },
        "image_id": "sha256:fixture",
        "tensorrt_version": "10.16.0.72",
    }


def _protocol() -> dict[str, object]:
    return {
        "compute_lane": "real B3 input; no H2D/D2H timing",
        "inference_streams": 1,
        "primary_duration_s": 10,
        "primary_rounds": 3,
        "spin_wait": True,
        "warmup_ms": 1000,
    }


def _performance_model(
    identity: tuple[str, str, str, int],
    *,
    suite: str,
    fp16: bool,
    yolo: bool = False,
    include_engine_sha: bool = True,
) -> dict[str, object]:
    model_id, family, variant, resolution = identity
    latency = 5.0 if fp16 else 10.0
    throughput = 300.0 if fp16 else 150.0
    row: dict[str, object] = {
        "model_id": model_id,
        "family": family,
        "variant": variant,
        "resolution": resolution,
        "batch_size": 3,
        **(
            {"engine_sha256": _engine_sha(suite, model_id)}
            if include_engine_sha
            else {}
        ),
        "compute": {
            "batch_latency_p50_ms": latency,
            "batch_latency_p95_ms": latency + 1.0,
            "throughput_images_s": throughput,
            "process_vram_peak_mib": 180.0 if fp16 else 200.0,
            "power_mean_w": 90.0 if fp16 else 100.0,
        },
    }
    if yolo:
        row["workload_unit"] = "person_crops" if family == "keypoint" else "full_frames"
        row["published_accuracy"] = {
            "value": 98.765,
            "source": "fixture vendor page",
        }
    return row


def _write_performance_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    rf_fp32_media = root / "rf_fp32_media"
    media_run_path = rf_fp32_media / "run.json"
    _write_json(
        media_run_path,
        {
            "schema": "noesis.ds9.rfdetr-media-validation-run.v1",
            "runtime_engine_profile": "fp32_no_tf32",
        },
    )
    rf_fp32_bindings = rf_fp32_media / "trt-runtime-fp32_no_tf32.json"
    _write_json(
        rf_fp32_bindings,
        {
            "schema": "noesis.ds9.rfdetr-media-validation-trt.v1",
            "runtime_engine_profile": "fp32_no_tf32",
            "models": [
                {
                    "id": model_id,
                    "engine": {"sha256": _engine_sha("rf_fp32", model_id)},
                }
                for model_id, _family, _variant, _resolution in RF_MODELS
            ],
        },
    )
    rf_fp32_path = root / "rf_fp32_performance.json"
    _write_json(
        rf_fp32_path,
        {
            "schema": "noesis.ds9.rfdetr-performance-benchmark.v1",
            "status": "passed",
            "precision": "fp32",
            "tf32_enabled": False,
            "batch_size": 3,
            "environment": _environment(),
            "protocol": _protocol(),
            "sources": {
                "media_run": str(rf_fp32_media),
                "media_manifest_sha256": _sha256(media_run_path),
            },
            "models": [
                _performance_model(
                    model,
                    suite="rf_fp32",
                    fp16=False,
                    include_engine_sha=False,
                )
                for model in RF_MODELS
            ],
        },
    )

    rf_fp16_root = root / "rf_fp16_performance"
    rf_fp16_report = rf_fp16_root / "performance_report.json"
    fp16_models = [
        _performance_model(model, suite="rf_fp16", fp16=True) for model in RF_MODELS
    ]
    _write_json(
        rf_fp16_report,
        {
            "schema": "noesis.ds9.rfdetr-performance-benchmark.v1",
            "status": "passed",
            "precision": "fp16",
            "tf32_enabled": True,
            "batch_size": 3,
            "environment": _environment(),
            "protocol": _protocol(),
            "models": fp16_models,
        },
    )
    matrix_rows = []
    for row in fp16_models:
        compute = row["compute"]
        assert isinstance(compute, dict)
        matrix_rows.append(
            {
                **{key: value for key, value in row.items() if key != "compute"},
                "framework": "rfdetr",
                "runtime_engine_profile": "fp16_tf32",
                "precision": "fp16",
                "tf32_enabled": True,
                **{f"compute_{key}": value for key, value in compute.items()},
            }
        )
    rf_fp16_matrix = rf_fp16_root / "performance_matrix.json"
    _write_json(
        rf_fp16_matrix,
        {
            "schema": "noesis.ds9.rfdetr-performance-matrix.v1",
            "status": "passed",
            "source_report": str(rf_fp16_report),
            "source_report_sha256": _sha256(rf_fp16_report),
            "rows": matrix_rows,
        },
    )

    yolo_path = root / "yolo_fp16_performance.json"
    _write_json(
        yolo_path,
        {
            "schema": "noesis.ds9.yolo26-performance-benchmark.v2",
            "status": "passed",
            "precision": "FP16 enabled, TensorRT mixed precision",
            "tf32_build_setting": "TensorRT default fallback",
            "batch_size": 3,
            "environment": _environment(),
            "protocol": _protocol(),
            "models": [
                _performance_model(
                    model,
                    suite="yolo_fp16",
                    fp16=True,
                    yolo=True,
                )
                for model in YOLO_MODELS
            ],
        },
    )
    return rf_fp32_path, rf_fp32_bindings, rf_fp16_matrix, yolo_path


def _fixture_arguments(
    tmp_path: Path,
    *,
    output_name: str,
) -> tuple[list[str], Path, Path]:
    rf_fp32_coco = _write_accuracy_suite(tmp_path, "rf_fp32", RF_MODELS, ap=0.50)
    rf_fp16_coco = _write_accuracy_suite(tmp_path, "rf_fp16", RF_MODELS, ap=0.5005)
    yolo_fp16_coco = _write_accuracy_suite(tmp_path, "yolo_fp16", YOLO_MODELS, ap=0.55)
    (
        rf_fp32_perf,
        rf_fp32_bindings,
        rf_fp16_perf,
        yolo_fp16_perf,
    ) = _write_performance_inputs(tmp_path)
    output = tmp_path / output_name
    arguments = [
        "--rf-fp32-coco-summary",
        str(rf_fp32_coco),
        "--rf-fp16-coco-summary",
        str(rf_fp16_coco),
        "--yolo-fp16-coco-summary",
        str(yolo_fp16_coco),
        "--rf-fp32-performance",
        str(rf_fp32_perf),
        "--rf-fp32-engine-bindings",
        str(rf_fp32_bindings),
        "--rf-fp16-performance",
        str(rf_fp16_perf),
        "--yolo-fp16-performance",
        str(yolo_fp16_perf),
        "--output-dir",
        str(output),
    ]
    return arguments, output, yolo_fp16_perf


def test_builds_local_only_sha_bound_matrices(tmp_path: Path) -> None:
    arguments, output, _yolo_performance = _fixture_arguments(
        tmp_path, output_name="output"
    )
    result = MODULE.main(arguments)

    assert result == 0
    payload = json.loads(
        (output / "comparison_matrix.json").read_text(encoding="utf-8")
    )
    assert payload["schema"] == ("noesis.ds9.local-coco-performance-comparison.v1")
    assert payload["status"] == "passed"
    assert payload["status_scope"] == (
        "matrix_construction_and_contract_validation_only"
    )
    assert payload["accuracy"]["published_accuracy_imported"] is False
    assert payload["acceptance"]["rf_fp16_primary_ap_parity"]["status"] == "passed"
    assert (
        payload["acceptance"]["rf_fp16_primary_ap_parity"]["passed_model_count"] == 11
    )
    assert payload["acceptance"]["yolo_vs_rfdetr_dominance"]["status"] == (
        "not_evaluated"
    )
    assert len(payload["rows"]) == 25
    assert {(row["family"], row["coco_metric"]) for row in payload["rows"]} == {
        ("detection", "bbox"),
        ("segmentation", "segm"),
        ("keypoint", "keypoints"),
    }
    fp16_detection = next(
        row
        for row in payload["rows"]
        if row["architecture"] == "rfdetr"
        and row["model_id"] == "detect_nano"
        and row["precision_profile"] == "fp16_tf32"
    )
    assert fp16_detection["engine_sha256"] == _engine_sha("rf_fp16", "detect_nano")
    assert fp16_detection["rf_fp16_minus_fp32_ap_points"] == 0.05
    assert fp16_detection["rf_fp16_primary_ap_parity_pass"] is True
    assert fp16_detection["rf_fp16_latency_speedup_x"] == 2.0
    assert fp16_detection["rf_fp16_throughput_gain_x"] == 2.0
    serialized = json.dumps(payload)
    assert '"published_accuracy":' not in serialized
    assert "98.765" not in serialized

    with (output / "comparison_matrix.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 25
    markdown = (output / "comparison_matrix.md").read_text(encoding="utf-8")
    assert "No vendor or published-checkpoint accuracy is used." in markdown
    assert "person_crops/s" in markdown
    assert "Construction status: passed" in markdown
    assert "RF FP16 primary-AP parity: PASSED" in markdown


def test_rejects_accuracy_performance_engine_sha_mismatch(
    tmp_path: Path,
) -> None:
    arguments, output, yolo_performance = _fixture_arguments(
        tmp_path, output_name="mismatch-output"
    )
    report = json.loads(yolo_performance.read_text(encoding="utf-8"))
    report["models"][0]["engine_sha256"] = "f" * 64
    _write_json(yolo_performance, report)

    assert MODULE.main(arguments) == 2
    assert not output.exists()


def test_rejects_missing_performance_engine_sha(tmp_path: Path) -> None:
    arguments, output, yolo_performance = _fixture_arguments(
        tmp_path, output_name="missing-output"
    )
    report = json.loads(yolo_performance.read_text(encoding="utf-8"))
    report["models"][0].pop("engine_sha256")
    _write_json(yolo_performance, report)

    assert MODULE.main(arguments) == 2
    assert not output.exists()
