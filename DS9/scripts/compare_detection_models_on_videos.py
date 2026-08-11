#!/usr/bin/env python3
"""Compare two exact DS9 detection engines on identical decoded video frames.

This is an unlabeled, model-output-quality comparison. It reports confidence,
detection continuity, box/count agreement, and isolated TensorRT timing. It
does not claim AP, tracker quality, or byte-exact DeepStream parser behavior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_coco_head_to_head as evaluator  # noqa: E402


SCHEMA = "noesis.ds9.identical-video-detection-comparison.v1"
MODEL_SCORE_FLOOR = 0.05
COMMON_THRESHOLDS = (0.25, 0.40, 0.50, 0.70)
MATCH_IOU_MIN = 0.50


class ComparisonError(RuntimeError):
    """Raised when the deterministic comparison contract is invalid."""


@dataclass(frozen=True)
class VideoSpec:
    cohort: str
    camera: str
    path: Path


@dataclass
class DetectionStats:
    frame_count: int = 0
    detection_count: int = 0
    frames_with_detection: int = 0
    singleton_frames: int = 0
    multi_detection_frames: int = 0
    scores: list[float] = field(default_factory=list)
    per_frame_counts: list[int] = field(default_factory=list)
    per_present_frame_max_scores: list[float] = field(default_factory=list)
    box_area_fractions: list[float] = field(default_factory=list)
    current_gap_frames: int = 0
    max_gap_frames: int = 0
    max_gap_seconds: float = 0.0

    def observe(
        self,
        detections: Sequence[Mapping[str, Any]],
        *,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        count = len(detections)
        self.frame_count += 1
        self.detection_count += count
        self.per_frame_counts.append(count)
        if count == 0:
            self.current_gap_frames += 1
            self.max_gap_frames = max(self.max_gap_frames, self.current_gap_frames)
            if fps > 0.0:
                self.max_gap_seconds = max(
                    self.max_gap_seconds, self.current_gap_frames / fps
                )
            return
        self.current_gap_frames = 0
        self.frames_with_detection += 1
        self.singleton_frames += int(count == 1)
        self.multi_detection_frames += int(count >= 2)
        frame_scores = [float(row["score"]) for row in detections]
        self.scores.extend(frame_scores)
        self.per_present_frame_max_scores.append(max(frame_scores))
        image_area = float(width * height)
        for row in detections:
            box = row["bbox"]
            self.box_area_fractions.append(
                max(0.0, float(box[2])) * max(0.0, float(box[3])) / image_area
            )


@dataclass
class AgreementStats:
    frame_count: int = 0
    matched_count: int = 0
    yolo_count: int = 0
    rf_count: int = 0
    yolo_unmatched_count: int = 0
    rf_unmatched_count: int = 0
    frames_both: int = 0
    frames_yolo_only: int = 0
    frames_rf_only: int = 0
    frames_neither: int = 0
    matched_ious: list[float] = field(default_factory=list)
    absolute_count_deltas: list[int] = field(default_factory=list)

    def observe(
        self,
        yolo: Sequence[Mapping[str, Any]],
        rf: Sequence[Mapping[str, Any]],
    ) -> None:
        self.frame_count += 1
        self.yolo_count += len(yolo)
        self.rf_count += len(rf)
        self.absolute_count_deltas.append(abs(len(yolo) - len(rf)))
        if yolo and rf:
            self.frames_both += 1
        elif yolo:
            self.frames_yolo_only += 1
        elif rf:
            self.frames_rf_only += 1
        else:
            self.frames_neither += 1
        matched = _match_detections(yolo, rf)
        self.matched_count += len(matched)
        self.matched_ious.extend(matched)
        self.yolo_unmatched_count += len(yolo) - len(matched)
        self.rf_unmatched_count += len(rf) - len(matched)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(value)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _parse_video(raw: str) -> VideoSpec:
    parts = raw.split(":", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "--video must be COHORT:CAMERA:/absolute/path.mp4"
        )
    path = Path(parts[2]).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("video paths must be absolute")
    return VideoSpec(parts[0].strip(), parts[1].strip(), path.resolve())


def _percentile(values: Sequence[float | int], percentile: float) -> float | None:
    if len(values) == 0:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _distribution(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "p95": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p10": _percentile(array, 10),
        "p50": _percentile(array, 50),
        "p90": _percentile(array, 90),
        "p95": _percentile(array, 95),
    }


def _stats_payload(stats: DetectionStats) -> dict[str, Any]:
    frames = max(stats.frame_count, 1)
    present = max(stats.frames_with_detection, 1)
    return {
        "frame_count": stats.frame_count,
        "detection_count": stats.detection_count,
        "frames_with_detection": stats.frames_with_detection,
        "frame_presence_rate": stats.frames_with_detection / frames,
        "detections_per_frame": stats.detection_count / frames,
        "singleton_frame_rate_among_present": stats.singleton_frames / present,
        "multi_detection_frame_rate_among_present": (
            stats.multi_detection_frames / present
        ),
        "scores": _distribution(stats.scores),
        "per_frame_counts": _distribution(stats.per_frame_counts),
        "per_present_frame_max_scores": _distribution(
            stats.per_present_frame_max_scores
        ),
        "box_area_fraction": _distribution(stats.box_area_fractions),
        "max_no_detection_gap_frames": stats.max_gap_frames,
        "max_no_detection_gap_seconds": stats.max_gap_seconds,
    }


def _agreement_payload(stats: AgreementStats) -> dict[str, Any]:
    yolo_denominator = max(stats.yolo_count, 1)
    rf_denominator = max(stats.rf_count, 1)
    return {
        "frame_count": stats.frame_count,
        "matched_count": stats.matched_count,
        "yolo_detection_count": stats.yolo_count,
        "rf_detection_count": stats.rf_count,
        "yolo_matched_fraction": stats.matched_count / yolo_denominator,
        "rf_matched_fraction": stats.matched_count / rf_denominator,
        "yolo_unmatched_count": stats.yolo_unmatched_count,
        "rf_unmatched_count": stats.rf_unmatched_count,
        "frames_both": stats.frames_both,
        "frames_yolo_only": stats.frames_yolo_only,
        "frames_rf_only": stats.frames_rf_only,
        "frames_neither": stats.frames_neither,
        "matched_iou": _distribution(stats.matched_ious),
        "absolute_count_delta": _distribution(stats.absolute_count_deltas),
    }


def _xywh_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lw, lh = [float(value) for value in left]
    rx1, ry1, rw, rh = [float(value) for value in right]
    lx2, ly2 = lx1 + lw, ly1 + lh
    rx2, ry2 = rx1 + rw, ry1 + rh
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = max(0.0, lw * lh) + max(0.0, rw * rh) - intersection
    return intersection / union if union > 0.0 else 0.0


def _match_detections(
    yolo: Sequence[Mapping[str, Any]],
    rf: Sequence[Mapping[str, Any]],
) -> list[float]:
    if not yolo or not rf:
        return []
    ious = np.asarray(
        [
            [_xywh_iou(left["bbox"], right["bbox"]) for right in rf]
            for left in yolo
        ],
        dtype=np.float64,
    )
    left_indexes, right_indexes = linear_sum_assignment(1.0 - ious)
    return [
        float(ious[left, right])
        for left, right in zip(left_indexes, right_indexes, strict=True)
        if float(ious[left, right]) >= MATCH_IOU_MIN
    ]


def _prepare_frame(
    frame_bgr: np.ndarray, spec: evaluator.ModelSpec
) -> tuple[np.ndarray, evaluator.ImageTransform]:
    if spec.architecture == "rfdetr":
        return evaluator._direct_square(frame_bgr, spec.resolution)
    return evaluator._letterbox(frame_bgr, spec.resolution, spec.interpolation)


def _select_models(args: argparse.Namespace) -> dict[str, evaluator.ModelSpec]:
    spec_args = argparse.Namespace(
        artifact_root=args.artifact_root,
        rf_matrix=args.rf_matrix,
        yolo_matrix=args.yolo_matrix,
        yolo_performance_report=args.yolo_performance_report,
    )
    rf = [
        row
        for row in evaluator._rf_specs(spec_args, evaluator.RF_FP16_PROFILE)
        if row.source_id == "detect_medium"
    ]
    yolo = [
        row
        for row in evaluator._yolo_specs(spec_args)
        if row.source_id == "yolo26_detect_m"
    ]
    if len(rf) != 1 or len(yolo) != 1:
        raise ComparisonError(
            f"expected exactly one RF Medium and YOLO26-M spec, found {len(rf)} and {len(yolo)}"
        )
    return {"yolo26_m": yolo[0], "rfdetr_medium": rf[0]}


def _validate_runner(
    runner: evaluator.TensorRTRunner, spec: evaluator.ModelSpec
) -> None:
    expected_input = (3, 3, spec.resolution, spec.resolution)
    if runner.input_shape != expected_input:
        raise ComparisonError(
            f"{spec.id} input shape {runner.input_shape} != {expected_input}"
        )
    if runner.input_name != spec.expected_input_name:
        raise ComparisonError(
            f"{spec.id} input name {runner.input_name} != {spec.expected_input_name}"
        )
    if set(runner.output_names) != set(spec.expected_outputs):
        raise ComparisonError(f"{spec.id} output names drifted")
    for name, shape in spec.expected_outputs.items():
        if runner.shapes[name] != shape:
            raise ComparisonError(
                f"{spec.id} output {name} shape {runner.shapes[name]} != {shape}"
            )


def _filter(
    detections: Sequence[Mapping[str, Any]], threshold: float
) -> list[Mapping[str, Any]]:
    return [row for row in detections if float(row["score"]) >= threshold]


def _video_metadata(video: VideoSpec) -> dict[str, Any]:
    if not video.path.is_file():
        raise ComparisonError(f"video is missing: {video.path}")
    capture = cv2.VideoCapture(str(video.path))
    if not capture.isOpened():
        raise ComparisonError(f"cannot open video: {video.path}")
    try:
        return {
            "cohort": video.cohort,
            "camera": video.camera,
            "path": str(video.path),
            "sha256": _sha256(video.path),
            "size_bytes": video.path.stat().st_size,
            "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "declared_frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        }
    finally:
        capture.release()


def _make_detection_stats() -> dict[str, dict[str, DetectionStats]]:
    return {
        model: {
            f"{threshold:.2f}": DetectionStats()
            for threshold in COMMON_THRESHOLDS
        }
        for model in ("yolo26_m", "rfdetr_medium")
    }


def _make_agreement_stats() -> dict[str, AgreementStats]:
    return {
        f"{threshold:.2f}": AgreementStats() for threshold in COMMON_THRESHOLDS
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if str(evaluator.trt.__version__) not in {
        evaluator.EXPECTED_TRT_VERSION,
        f"{evaluator.EXPECTED_TRT_VERSION}.post1",
    }:
        raise ComparisonError(
            f"TensorRT {evaluator.EXPECTED_TRT_VERSION} is required; "
            f"found {evaluator.trt.__version__}"
        )
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    os.chmod(run_dir, 0o700)
    videos = list(args.video)
    if len(videos) < 2:
        raise ComparisonError("at least two videos are required")
    keys = [(row.cohort, row.camera) for row in videos]
    if len(set(keys)) != len(keys):
        raise ComparisonError("cohort/camera video keys must be unique")
    metadata = [_video_metadata(row) for row in videos]
    models = _select_models(args)
    model_rows = {}
    for name, spec in models.items():
        engine = Path(spec.engine)
        observed_sha = _sha256(engine)
        if spec.expected_engine_sha256 and observed_sha != spec.expected_engine_sha256:
            raise ComparisonError(f"{name} engine SHA-256 drifted")
        if spec.engine_receipt:
            evaluator._engine_receipt(spec, observed_sha)
        model_rows[name] = {
            "spec": asdict(spec),
            "engine_sha256": observed_sha,
            "engine_size_bytes": engine.stat().st_size,
        }

    contract = {
        "schema": SCHEMA,
        "scope": "unlabeled_model_output_quality",
        "ground_truth_available": False,
        "ap_computed": False,
        "runtime_parser_exact": False,
        "same_decoded_frame_object_for_both_models": True,
        "model_score_floor": MODEL_SCORE_FLOOR,
        "common_thresholds": list(COMMON_THRESHOLDS),
        "match_iou_min": MATCH_IOU_MIN,
        "models": model_rows,
        "videos": metadata,
        "runtime": {
            "python": sys.version,
            "tensorrt": str(evaluator.trt.__version__),
            "opencv": str(cv2.__version__),
            "numpy": str(np.__version__),
        },
    }
    _write_json_exclusive(run_dir / "run_contract.json", contract)

    cohort_detection: dict[str, dict[str, dict[str, DetectionStats]]] = {}
    cohort_agreement: dict[str, dict[str, AgreementStats]] = {}
    video_results: dict[str, Any] = {}
    timing: dict[str, dict[str, list[float]]] = {
        name: {"inference_ms": [], "preprocess_ms": []} for name in models
    }
    started = time.monotonic()
    predictions_path = run_dir / "frame_predictions.jsonl"
    descriptor = os.open(
        predictions_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    prediction_handle = os.fdopen(descriptor, "w", encoding="utf-8")
    descriptor = -1
    try:
        with evaluator.TensorRTRunner(
            Path(models["yolo26_m"].engine)
        ) as yolo_runner, evaluator.TensorRTRunner(
            Path(models["rfdetr_medium"].engine)
        ) as rf_runner:
            runners = {"yolo26_m": yolo_runner, "rfdetr_medium": rf_runner}
            for name, runner in runners.items():
                _validate_runner(runner, models[name])
                zeros = np.zeros(runner.input_shape, dtype=np.float32)
                runner.infer(zeros)

            for video_number, (video, media) in enumerate(
                zip(videos, metadata, strict=True), start=1
            ):
                print(
                    f"[VIDEO {video_number}/{len(videos)}] "
                    f"{video.cohort}/{video.camera}: {video.path}",
                    flush=True,
                )
                capture = cv2.VideoCapture(str(video.path))
                if not capture.isOpened():
                    raise ComparisonError(f"cannot open video: {video.path}")
                video_detection = _make_detection_stats()
                video_agreement = _make_agreement_stats()
                cohort_detection.setdefault(video.cohort, _make_detection_stats())
                cohort_agreement.setdefault(video.cohort, _make_agreement_stats())
                frame_index = 0
                batch_number = 0
                decoded_fingerprint = hashlib.sha256()
                try:
                    while True:
                        frames: list[np.ndarray] = []
                        frame_indexes: list[int] = []
                        for _ in range(3):
                            ok, frame = capture.read()
                            if not ok:
                                break
                            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                                raise ComparisonError(
                                    f"invalid decoded frame {frame_index}: {video.path}"
                                )
                            frames.append(frame)
                            frame_indexes.append(frame_index)
                            frame_index += 1
                        if not frames:
                            break
                        decoded_fingerprint.update(
                            np.asarray(
                                [
                                    frame_indexes[0],
                                    len(frames),
                                    int(frames[0].shape[1]),
                                    int(frames[0].shape[0]),
                                    int(frames[0][0, 0, 0]),
                                    int(frames[-1][-1, -1, -1]),
                                ],
                                dtype="<i8",
                            ).tobytes()
                        )
                        prepared: dict[str, tuple[np.ndarray, list[Any]]] = {}
                        for name, spec in models.items():
                            prep_started = time.perf_counter()
                            tensors = []
                            transforms = []
                            for frame in frames:
                                tensor, transform = _prepare_frame(frame, spec)
                                tensors.append(tensor)
                                transforms.append(transform)
                            while len(tensors) < 3:
                                tensors.append(tensors[-1].copy())
                                transforms.append(transforms[-1])
                            prepared[name] = (
                                np.ascontiguousarray(
                                    np.stack(tensors), dtype=np.float32
                                ),
                                transforms,
                            )
                            timing[name]["preprocess_ms"].append(
                                (time.perf_counter() - prep_started) * 1000.0
                            )
                        order = (
                            ("yolo26_m", "rfdetr_medium")
                            if batch_number % 2 == 0
                            else ("rfdetr_medium", "yolo26_m")
                        )
                        outputs = {}
                        for name in order:
                            outputs[name], elapsed_ms = runners[name].infer(
                                prepared[name][0]
                            )
                            timing[name]["inference_ms"].append(elapsed_ms)
                        for batch_index, source_frame_index in enumerate(frame_indexes):
                            raw: dict[str, list[dict[str, Any]]] = {}
                            for name, spec in models.items():
                                raw[name] = evaluator._postprocess(
                                    spec,
                                    source_frame_index,
                                    prepared[name][1][batch_index],
                                    outputs[name],
                                    batch_index,
                                    MODEL_SCORE_FLOOR,
                                    100,
                                )
                            prediction_handle.write(
                                json.dumps(
                                    {
                                        "cohort": video.cohort,
                                        "camera": video.camera,
                                        "video_sha256": media["sha256"],
                                        "frame_index": source_frame_index,
                                        "pts_seconds": (
                                            source_frame_index / float(media["fps"])
                                            if float(media["fps"]) > 0.0
                                            else None
                                        ),
                                        "detections": raw,
                                    },
                                    separators=(",", ":"),
                                    allow_nan=False,
                                )
                                + "\n"
                            )
                            for threshold in COMMON_THRESHOLDS:
                                key = f"{threshold:.2f}"
                                filtered = {
                                    name: _filter(rows, threshold)
                                    for name, rows in raw.items()
                                }
                                for name in models:
                                    kwargs = {
                                        "width": int(media["width"]),
                                        "height": int(media["height"]),
                                        "fps": float(media["fps"]),
                                    }
                                    video_detection[name][key].observe(
                                        filtered[name], **kwargs
                                    )
                                    cohort_detection[video.cohort][name][key].observe(
                                        filtered[name], **kwargs
                                    )
                                video_agreement[key].observe(
                                    filtered["yolo26_m"],
                                    filtered["rfdetr_medium"],
                                )
                                cohort_agreement[video.cohort][key].observe(
                                    filtered["yolo26_m"],
                                    filtered["rfdetr_medium"],
                                )
                        batch_number += 1
                        if batch_number % 250 == 0:
                            print(
                                f"  frames={frame_index} batches={batch_number}",
                                flush=True,
                            )
                finally:
                    capture.release()
                if frame_index != int(media["declared_frame_count"]):
                    raise ComparisonError(
                        f"decoded frame count drifted for {video.path}: "
                        f"{frame_index} != {media['declared_frame_count']}"
                    )
                video_key = f"{video.cohort}/{video.camera}"
                video_results[video_key] = {
                    "media": media,
                    "decoded_frame_count": frame_index,
                    "decoded_sequence_fingerprint": decoded_fingerprint.hexdigest(),
                    "models": {
                        name: {
                            key: _stats_payload(value)
                            for key, value in thresholds.items()
                        }
                        for name, thresholds in video_detection.items()
                    },
                    "agreement": {
                        key: _agreement_payload(value)
                        for key, value in video_agreement.items()
                    },
                }
    finally:
        prediction_handle.close()

    report = {
        **contract,
        "status": "passed",
        "wall_seconds": time.monotonic() - started,
        "frame_predictions": {
            "path": str(predictions_path),
            "sha256": _sha256(predictions_path),
            "size_bytes": predictions_path.stat().st_size,
        },
        "timing": {
            name: {
                "batch_size": 3,
                "batch_count": len(values["inference_ms"]),
                "engine_work_items": len(values["inference_ms"]) * 3,
                "inference_batch_ms": _distribution(values["inference_ms"]),
                "preprocess_batch_ms": _distribution(values["preprocess_ms"]),
                "engine_throughput_work_items_s": (
                    len(values["inference_ms"])
                    * 3
                    / (sum(values["inference_ms"]) / 1000.0)
                ),
            }
            for name, values in timing.items()
        },
        "cohorts": {
            cohort: {
                "models": {
                    name: {
                        key: _stats_payload(value)
                        for key, value in thresholds.items()
                    }
                    for name, thresholds in model_stats.items()
                },
                "agreement": {
                    key: _agreement_payload(value)
                    for key, value in cohort_agreement[cohort].items()
                },
            }
            for cohort, model_stats in cohort_detection.items()
        },
        "videos": video_results,
        "limitations": [
            "The videos have no human ground-truth annotations, so AP, precision, recall, and calibration error are not computed.",
            "The lane uses reviewed engine-output preprocessing and postprocessing, not byte-exact DeepStream preprocessing or deployed parser/clustering.",
            "Detection continuity is not tracker identity continuity and cannot distinguish an empty frame from a missed person without labels.",
        ],
    }
    _write_json_exclusive(run_dir / "report.json", report)
    _write_outputs(run_dir, report)
    return report


def _write_outputs(run_dir: Path, report: Mapping[str, Any]) -> None:
    csv_path = run_dir / "comparison_matrix.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cohort",
                "threshold",
                "model",
                "frames",
                "detections",
                "presence_rate",
                "detections_per_frame",
                "mean_score",
                "median_score",
                "mean_max_score",
                "engine_batch_p50_ms",
                "engine_batch_p95_ms",
                "engine_work_items_s",
            ],
        )
        writer.writeheader()
        for cohort, cohort_row in report["cohorts"].items():
            for model, thresholds in cohort_row["models"].items():
                timing = report["timing"][model]
                for threshold, row in thresholds.items():
                    writer.writerow(
                        {
                            "cohort": cohort,
                            "threshold": threshold,
                            "model": model,
                            "frames": row["frame_count"],
                            "detections": row["detection_count"],
                            "presence_rate": row["frame_presence_rate"],
                            "detections_per_frame": row["detections_per_frame"],
                            "mean_score": row["scores"]["mean"],
                            "median_score": row["scores"]["p50"],
                            "mean_max_score": row[
                                "per_present_frame_max_scores"
                            ]["mean"],
                            "engine_batch_p50_ms": timing[
                                "inference_batch_ms"
                            ]["p50"],
                            "engine_batch_p95_ms": timing[
                                "inference_batch_ms"
                            ]["p95"],
                            "engine_work_items_s": timing[
                                "engine_throughput_work_items_s"
                            ],
                        }
                    )

    lines = [
        "# Identical-video YOLO26-M vs RF-DETR Medium",
        "",
        "Unlabeled local camera-video comparison. AP is intentionally not computed.",
        "",
        "| Cohort | Threshold | Model | Presence | Det/frame | Mean score | Median score |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for cohort, cohort_row in report["cohorts"].items():
        for threshold in sorted(cohort_row["agreement"]):
            for model in ("yolo26_m", "rfdetr_medium"):
                row = cohort_row["models"][model][threshold]
                lines.append(
                    f"| {cohort} | {threshold} | {model} | "
                    f"{row['frame_presence_rate']:.3f} | "
                    f"{row['detections_per_frame']:.3f} | "
                    f"{row['scores']['mean']:.3f} | "
                    f"{row['scores']['p50']:.3f} |"
                )
    lines.extend(
        [
            "",
            "| Model | B3 p50 ms | B3 p95 ms | Engine work items/s |",
            "|---|---:|---:|---:|",
        ]
    )
    for model in ("yolo26_m", "rfdetr_medium"):
        row = report["timing"][model]
        lines.append(
            f"| {model} | {row['inference_batch_ms']['p50']:.3f} | "
            f"{row['inference_batch_ms']['p95']:.3f} | "
            f"{row['engine_throughput_work_items_s']:.1f} |"
        )
    lines.extend(["", "See `report.json` for per-camera continuity and IoU agreement.", ""])
    _write_text_exclusive(run_dir / "comparison_matrix.md", "\n".join(lines))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--rf-matrix", required=True)
    parser.add_argument("--yolo-matrix", required=True)
    parser.add_argument("--yolo-performance-report", required=True)
    parser.add_argument(
        "--video",
        action="append",
        type=_parse_video,
        required=True,
        help="COHORT:CAMERA:/absolute/path.mp4; repeat for each source",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = _run(args)
    except (ComparisonError, evaluator.EvaluationError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    print(f"[PASS] {Path(args.run_dir).resolve() / 'report.json'}")
    print(
        f"frames={sum(row['decoded_frame_count'] for row in report['videos'].values())} "
        f"wall_seconds={report['wall_seconds']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
