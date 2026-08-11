#!/usr/bin/env python3
"""Run an owner-private, unpromoted RF-DETR 1.8.3 media-quality gate.

The workflow is intentionally split into CPU reference preparation, isolated
TensorRT execution, comparison/rendering, and an explicit visual-review seal.
It never changes the DS9 asset manifest, realization, runtime configuration,
deployment selector, or selected production model.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
MATRIX_PATH = DS9_ROOT / "config" / "rfdetr_1_8_3_models.json"
BUILDER_PATH = DS9_ROOT / "scripts" / "build_rfdetr_1_8_3_engines.py"
RUNNER_SOURCE = DS9_ROOT / "native" / "rfdetr_1_8_3_media_runner.cpp"

RELEASE_VERSION = "1.8.3"
MATRIX_SCHEMA = "noesis.ds9.rfdetr-model-matrix.v1"
RUN_SCHEMA = "noesis.ds9.rfdetr-media-validation-run.v1"
TRT_SCHEMA = "noesis.ds9.rfdetr-media-validation-trt.v1"
REPORT_SCHEMA = "noesis.ds9.rfdetr-media-quality-report.v2"
SUPPORTED_REPORT_SCHEMAS = frozenset(
    {
        "noesis.ds9.rfdetr-media-quality-report.v1",
        REPORT_SCHEMA,
    }
)
REVIEW_SCHEMA = "noesis.ds9.rfdetr-media-visual-review.v1"
RUNNER_SCHEMA = "noesis.rfdetr_1_8_3.media_runner.v1"

FRAME_INDICES = (180, 360, 630, 810, 1080, 1260, 1530, 1710)
ROOMS = ("family-room", "kitchen", "living-room")
VIDEO_ROWS = {
    "family-room": {
        "filename": "family-room-calibration-occupied-v3-68s.mp4",
        "sha256": "3906092f2f650a8a8dd50f73d8aba94b5d3c14f573ae7ccea21813d2b9e9b53c",
        "width": 1280,
        "height": 720,
    },
    "kitchen": {
        "filename": "kitchen-calibration-occupied-v3-68s.mp4",
        "sha256": "747726df51577072ccb8666f025c8881f04a31499007805e37fcc1c1b9236194",
        "width": 1920,
        "height": 1080,
    },
    "living-room": {
        "filename": "living-room-calibration-occupied-v3-68s.mp4",
        "sha256": "57c3a96fb4d17072f71c54ca35330abb4518c2ab121196897820df895992e564",
        "width": 1920,
        "height": 1080,
    },
}
OCCUPANCY = {
    "family-room": {index: True for index in FRAME_INDICES},
    "kitchen": {index: True for index in FRAME_INDICES},
    "living-room": {
        180: False,
        360: True,
        630: False,
        810: True,
        1080: False,
        1260: True,
        1530: False,
        1710: True,
    },
}

SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_LOG_BYTES = 8 * 1024 * 1024
COMPILE_TIMEOUT_SECONDS = 600
INFERENCE_TIMEOUT_SECONDS = 1800
PRECISION_CANARY_PROFILES = (
    "fp32_no_tf32",
    "fp32_tf32",
    "fp16_no_tf32",
    "fp16_fp32_heads_no_tf32",
)
BASELINE_RUNTIME_ENGINE_PROFILE = "fp16_tf32"
RUNTIME_ENGINE_PROFILES = (
    "fp32_no_tf32",
    BASELINE_RUNTIME_ENGINE_PROFILE,
)
RUNTIME_VALIDATION_ARTIFACT_ROLE = "runtime_input_engine_validation"
RUNTIME_INPUT_CONTRACT = "rgb01_imagenet"
RUNTIME_ADAPTER_REVISION = "sub_div_float32_v1"
RUNTIME_ADAPTER_SPEC = {
    "input_contract": RUNTIME_INPUT_CONTRACT,
    "adapter_revision": RUNTIME_ADAPTER_REVISION,
    "input_dtype": "float32",
    "input_range": [0.0, 1.0],
    "operation": "(input - mean) / std",
    "channel_order": "RGB",
    "layout": "NCHW",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "nodes": [
        "NoesisInputNormalize/Sub",
        "NoesisInputNormalize/DivStd",
    ],
}

DETECTION_GATES = {
    "class_agreement_min": 1.0,
    "retention_min": 1.0,
    "box_iou_median_min": 0.98,
    "box_iou_p05_min": 0.90,
    "score_abs_error_p95_max": 0.03,
}
PERSON_TASK_GATES = {
    "reference_retention_min": 1.0,
    "candidate_precision_min": 1.0,
    "box_iou_median_min": 0.98,
    "box_iou_p05_min": 0.90,
    "score_abs_error_p95_max": 0.03,
}
SEGMENTATION_GATES = {
    "mask_iou_median_min": 0.95,
    "mask_iou_p05_min": 0.85,
    "mask_area_relative_error_p95_max": 0.10,
}
KEYPOINT_GATES = {
    "base_score_abs_error_p95_max": 0.03,
    "fused_score_relative_error_p95_max": 0.03,
    "oks_median_min": 0.97,
    "oks_min_min": 0.90,
    "coordinate_error_box_diagonal_p95_max": 0.01,
    "visible_jaccard_min": 0.90,
}
STRONG_REFERENCE_SCORE_MIN = 0.7
PERSON_TASK_SCORE_MIN = 0.5
SEMANTIC_MATCH_IOU_MIN = 0.5
SEMANTIC_ASSIGNMENT_CONTRACT = (
    "rfdetr_semantic_hungarian_lexicographic_v1"
)
PERSON_ASSIGNMENT_CONTRACT = (
    "rfdetr_person_thresholded_symmetric_hungarian_v1"
)

COCO_KEYPOINT_SIGMAS = (
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
)
COCO_SKELETON = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)


class ValidationError(RuntimeError):
    """Raised when the fail-closed media-validation contract is violated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_run_id() -> str:
    return "rfdetr-media-" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    ).lower()


def _safe_token(raw: object, label: str = "token") -> str:
    value = str(raw or "")
    if SAFE_TOKEN.fullmatch(value) is None or value in {".", ".."}:
        raise ValidationError(f"unsafe {label}: {value!r}")
    return value


def _safe_filename(raw: object, label: str = "filename") -> str:
    value = str(raw or "")
    if (
        SAFE_FILENAME.fullmatch(value) is None
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise ValidationError(f"unsafe {label}: {value!r}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValidationError(f"path escaped validation run: {path}") from exc


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValidationError(f"{label} must be a nonempty regular file: {path}")


def _require_private_directory(path: Path, label: str = "private directory") -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValidationError(f"{label} is unavailable: {path}")
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValidationError(
            f"{label} must be caller-owned and mode 0700: {path}"
        )
    return path


def _ensure_private_directory(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        return _require_private_directory(path)
    path.mkdir(parents=True, mode=0o700)
    os.chmod(path, 0o700)
    return _require_private_directory(path)


def _explicit_external_root(raw: str, label: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise ValidationError(f"{label} must be set explicitly")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValidationError(f"{label} must be absolute: {value}")
    resolved = path.resolve()
    if resolved == Path("/") or resolved in {
        REPO_ROOT.resolve(),
        DS9_ROOT.resolve(),
    }:
        raise ValidationError(f"refusing unsafe {label}: {resolved}")
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return resolved
    raise ValidationError(f"{label} must be outside the checkout: {resolved}")


def _open_exclusive(path: Path, mode: int = 0o600) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for name in ("O_CLOEXEC", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    try:
        return os.open(path, flags, mode)
    except OSError as exc:
        raise ValidationError(f"cannot create validation evidence: {path}") from exc


def _write_bytes_exclusive(path: Path, value: bytes) -> None:
    descriptor = _open_exclusive(path)
    try:
        offset = 0
        while offset < len(value):
            written = os.write(descriptor, value[offset:])
            if written <= 0:
                raise ValidationError(f"short write while creating {path}")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    os.close(descriptor)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_JSON_BYTES:
        raise ValidationError(f"JSON evidence exceeds {MAX_JSON_BYTES} bytes")
    _write_bytes_exclusive(path, encoded)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValidationError(f"{label} exceeds {MAX_JSON_BYTES} bytes: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"{label} root must be a JSON object: {path}")
    return payload


def _load_matrix(path: Path = MATRIX_PATH) -> dict[str, Any]:
    payload = _read_json(path, "RF-DETR model matrix")
    release = payload.get("release")
    models = payload.get("models")
    if (
        payload.get("schema") != MATRIX_SCHEMA
        or not isinstance(release, dict)
        or release.get("version") != RELEASE_VERSION
        or release.get("batch_size") != 3
        or release.get("dynamic_batch") is not False
        or not isinstance(models, list)
        or not models
    ):
        raise ValidationError("RF-DETR matrix is not the reviewed static-B3 1.8.3 matrix")
    seen: set[str] = set()
    for row in models:
        if not isinstance(row, dict):
            raise ValidationError("RF-DETR matrix contains a non-object model row")
        model_id = _safe_token(row.get("id"), "model ID")
        if model_id in seen:
            raise ValidationError(f"duplicate RF-DETR model ID: {model_id}")
        seen.add(model_id)
        _safe_filename(row.get("onnx_filename"), "ONNX filename")
        _safe_filename(row.get("engine_filename"), "engine filename")
        runtime = row.get("runtime")
        if (
            not isinstance(runtime, Mapping)
            or runtime.get("input_contract") != RUNTIME_INPUT_CONTRACT
            or runtime.get("adapter_revision") != RUNTIME_ADAPTER_REVISION
        ):
            raise ValidationError(
                f"invalid runtime-input contract for {model_id}"
            )
        runtime_onnx = _safe_filename(
            runtime.get("onnx_filename"), "runtime ONNX filename"
        )
        if not runtime_onnx.endswith(".onnx"):
            raise ValidationError(
                f"invalid runtime ONNX filename for {model_id}"
            )
        outputs = row.get("outputs")
        if (
            row.get("family") not in {"detection", "segmentation", "keypoint"}
            or not isinstance(outputs, dict)
            or not outputs
            or any(
                not isinstance(shape, list)
                or not shape
                or any(type(value) is not int or value <= 0 for value in shape)
                for shape in outputs.values()
            )
        ):
            raise ValidationError(f"invalid tensor contract for {model_id}")
    return payload


def _split_models(raw: Sequence[str]) -> list[str]:
    return [
        part.strip()
        for item in raw
        for part in str(item).split(",")
        if part.strip()
    ]


def _select_models(
    matrix: Mapping[str, Any], requested: Sequence[str]
) -> list[dict[str, Any]]:
    requested_ids = _split_models(requested)
    if not requested_ids:
        raise ValidationError("at least one explicit --model ID is required")
    if len(set(requested_ids)) != len(requested_ids):
        raise ValidationError("duplicate RF-DETR model selection")
    by_id = {str(row["id"]): row for row in matrix["models"]}
    unknown = sorted(set(requested_ids) - set(by_id))
    if unknown:
        raise ValidationError(
            "unknown RF-DETR model ID(s): " + ", ".join(unknown)
        )
    disabled = [item for item in requested_ids if not by_id[item].get("enabled")]
    if disabled:
        raise ValidationError(
            "licensed/disabled RF-DETR rows are not authorized: "
            + "; ".join(
                f"{item}: {by_id[item].get('blocked_reason', 'disabled')}"
                for item in disabled
            )
        )
    selected = set(requested_ids)
    return [dict(row) for row in matrix["models"] if row["id"] in selected]


def _runtime_engine_profile(raw: object) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    if value not in RUNTIME_ENGINE_PROFILES:
        raise ValidationError(
            f"unsupported runtime engine profile {value!r}; expected one of "
            + ", ".join(RUNTIME_ENGINE_PROFILES)
        )
    return _safe_token(value, "runtime engine profile")


def _prepared_runtime_profile(run: Mapping[str, Any]) -> str:
    profile_raw = run.get("runtime_engine_profile")
    role = run.get("artifact_role")
    workflow = run.get("workflow")
    preprocess = (
        workflow.get("preprocess")
        if isinstance(workflow, Mapping)
        else None
    )
    runtime_fields_present = (
        profile_raw is not None
        or role is not None
        or (
            isinstance(preprocess, Mapping)
            and (
                "input_contract" in preprocess
                or "runtime_adapter_revision" in preprocess
                or "normalization_location" in preprocess
            )
        )
    )
    if not runtime_fields_present:
        return ""
    profile = _runtime_engine_profile(profile_raw)
    if (
        not profile
        or role != RUNTIME_VALIDATION_ARTIFACT_ROLE
        or not isinstance(preprocess, Mapping)
        or preprocess.get("resize") != "direct_square_bilinear"
        or preprocess.get("color") != "RGB"
        or preprocess.get("scale") != "uint8_to_float32_[0,1]"
        or preprocess.get("mean") is not None
        or preprocess.get("std") is not None
        or preprocess.get("letterbox") is not False
        or preprocess.get("input_contract") != RUNTIME_INPUT_CONTRACT
        or preprocess.get("runtime_adapter_revision")
        != RUNTIME_ADAPTER_REVISION
        or preprocess.get("normalization_location")
        != "revision_bound_runtime_onnx"
    ):
        raise ValidationError(
            "runtime-input validation run contract drifted"
        )
    models = run.get("models")
    if not isinstance(models, list) or not models:
        raise ValidationError(
            "runtime-input validation run has no prepared models"
        )
    for model in models:
        onnx = model.get("onnx") if isinstance(model, Mapping) else None
        cases = model.get("cases") if isinstance(model, Mapping) else None
        if (
            not isinstance(onnx, Mapping)
            or onnx.get("artifact_kind") != "runtime_onnx"
            or onnx.get("input_contract") != RUNTIME_INPUT_CONTRACT
            or onnx.get("adapter_revision") != RUNTIME_ADAPTER_REVISION
            or onnx.get("adapter") != RUNTIME_ADAPTER_SPEC
            or not isinstance(cases, list)
            or not cases
        ):
            raise ValidationError(
                "prepared runtime ONNX/input contract drifted"
            )
        for case in cases:
            input_row = (
                case.get("input") if isinstance(case, Mapping) else None
            )
            if (
                not isinstance(input_row, Mapping)
                or input_row.get("dtype") != "float32"
                or input_row.get("finite") is not True
                or type(input_row.get("min")) not in {int, float}
                or type(input_row.get("max")) not in {int, float}
                or not math.isfinite(float(input_row["min"]))
                or not math.isfinite(float(input_row["max"]))
                or float(input_row["min"]) < 0.0
                or float(input_row["max"]) > 1.0
            ):
                raise ValidationError(
                    "prepared runtime RGB01 tensor contract drifted"
                )
    return profile


def _sigmoid(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values)
    clipped = np.clip(array, -88.0, 88.0)
    one = np.asarray(1.0, dtype=clipped.dtype)
    return one / (one + np.exp(-clipped))


def _cxcywh_iou(left: Any, right: Any) -> Any:
    import numpy as np

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape[-1:] != (4,) or b.shape[-1:] != (4,):
        raise ValidationError("cxcywh IoU inputs must end in four coordinates")
    a_xyxy = np.concatenate((a[..., :2] - a[..., 2:] / 2, a[..., :2] + a[..., 2:] / 2), axis=-1)
    b_xyxy = np.concatenate((b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2), axis=-1)
    intersection_min = np.maximum(a_xyxy[..., :2], b_xyxy[..., :2])
    intersection_max = np.minimum(a_xyxy[..., 2:], b_xyxy[..., 2:])
    intersection_size = np.maximum(intersection_max - intersection_min, 0.0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    a_size = np.maximum(a_xyxy[..., 2:] - a_xyxy[..., :2], 0.0)
    b_size = np.maximum(b_xyxy[..., 2:] - b_xyxy[..., :2], 0.0)
    union = (
        a_size[..., 0] * a_size[..., 1]
        + b_size[..., 0] * b_size[..., 1]
        - intersection
    )
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )


def _semantic_assignment_weights(reference_count: int) -> dict[str, int]:
    """Return integer tier weights with proof of batch-level dominance."""
    if type(reference_count) is not int or reference_count <= 0:
        raise ValidationError(
            "semantic assignment reference count must be a positive integer"
        )
    iou_weight = 1
    spatial_weight = reference_count * iou_weight + 1
    class_weight = (
        reference_count * (spatial_weight + iou_weight) + 1
    )
    retained_weight = (
        reference_count
        * (class_weight + spatial_weight + iou_weight)
        + 1
    )
    maximum_edge_utility = (
        retained_weight
        + class_weight
        + spatial_weight
        + iou_weight
    )
    maximum_assignment_utility = (
        reference_count * maximum_edge_utility
    )
    if maximum_assignment_utility > 2**52:
        raise ValidationError(
            "semantic assignment objective exceeds exact float64 integer range"
        )
    return {
        "retained_correct": retained_weight,
        "correct_class": class_weight,
        "spatial_match": spatial_weight,
        "iou": iou_weight,
    }


def _match_queries_semantically(
    reference_boxes: Any,
    candidate_boxes: Any,
    reference_classes: Any,
    candidate_classes: Any,
    candidate_reference_class_scores: Any,
    *,
    iou_floor: float = SEMANTIC_MATCH_IOU_MIN,
    retention_floor: float = STRONG_REFERENCE_SCORE_MIN,
) -> list[dict[str, Any]]:
    """Return a unique, transparent lexicographic semantic assignment.

    Semantic class and retention priority applies only to spatially plausible
    edges. The integer weights make one improvement at a higher tier dominate
    every possible lower-tier change across the complete reference batch.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    reference = np.asarray(reference_boxes, dtype=np.float64)
    candidate = np.asarray(candidate_boxes, dtype=np.float64)
    reference_class = np.asarray(reference_classes)
    candidate_class = np.asarray(candidate_classes)
    reference_class_scores = np.asarray(
        candidate_reference_class_scores, dtype=np.float64
    )
    if (
        reference.ndim != 2
        or candidate.ndim != 2
        or reference.shape[1:] != (4,)
        or candidate.shape[1:] != (4,)
        or reference.shape[0] == 0
        or candidate.shape[0] < reference.shape[0]
        or not np.isfinite(reference).all()
        or not np.isfinite(candidate).all()
        or np.any(reference[:, 2:] <= 0.0)
        or np.any(candidate[:, 2:] <= 0.0)
    ):
        raise ValidationError(
            "semantic query matching requires finite positive-size [N,4] "
            "reference and [M,4] candidate boxes with 0 < N <= M"
        )
    reference_count = int(reference.shape[0])
    candidate_count = int(candidate.shape[0])
    if (
        reference_class.shape != (reference_count,)
        or candidate_class.shape != (candidate_count,)
        or reference_class.dtype.kind not in {"i", "u"}
        or candidate_class.dtype.kind not in {"i", "u"}
        or np.any(reference_class < 0)
        or np.any(candidate_class < 0)
        or reference_class_scores.shape
        != (reference_count, candidate_count)
        or not np.isfinite(reference_class_scores).all()
    ):
        raise ValidationError(
            "semantic query matching class/score contract drifted"
        )
    if (
        type(iou_floor) not in {int, float}
        or not math.isfinite(float(iou_floor))
        or not 0.0 <= float(iou_floor) <= 1.0
        or type(retention_floor) not in {int, float}
        or not math.isfinite(float(retention_floor))
    ):
        raise ValidationError("semantic query matching thresholds are invalid")

    pairwise_iou = _cxcywh_iou(
        reference[:, None, :],
        candidate[None, :, :],
    )
    spatial = pairwise_iou >= float(iou_floor)
    predicted_class_match = (
        reference_class[:, None] == candidate_class[None, :]
    )
    score_retained = reference_class_scores >= float(retention_floor)
    semantic_class_edge = spatial & predicted_class_match
    semantic_retained_edge = semantic_class_edge & score_retained
    weights = _semantic_assignment_weights(reference_count)
    utility = (
        pairwise_iou * weights["iou"]
        + spatial.astype(np.float64) * weights["spatial_match"]
        + semantic_class_edge.astype(np.float64)
        * weights["correct_class"]
        + semantic_retained_edge.astype(np.float64)
        * weights["retained_correct"]
    )
    reference_indexes, candidate_indexes = linear_sum_assignment(-utility)
    if len(reference_indexes) != reference_count:
        raise ValidationError(
            "semantic query matching did not assign every reference"
        )
    assignments = []
    for reference_index, candidate_index in sorted(
        zip(reference_indexes, candidate_indexes, strict=True)
    ):
        ref_index = int(reference_index)
        cand_index = int(candidate_index)
        assignments.append(
            {
                "reference_index": ref_index,
                "candidate_index": cand_index,
                "reference_class": int(reference_class[ref_index]),
                "candidate_class": int(candidate_class[cand_index]),
                "candidate_reference_class_score": float(
                    reference_class_scores[ref_index, cand_index]
                ),
                "iou": float(pairwise_iou[ref_index, cand_index]),
                "spatial_match": bool(spatial[ref_index, cand_index]),
                "class_match": bool(
                    predicted_class_match[ref_index, cand_index]
                ),
                "score_retained": bool(
                    score_retained[ref_index, cand_index]
                ),
                "semantic_class_edge": bool(
                    semantic_class_edge[ref_index, cand_index]
                ),
                "semantic_retained_edge": bool(
                    semantic_retained_edge[ref_index, cand_index]
                ),
                "objective_utility": float(
                    utility[ref_index, cand_index]
                ),
            }
        )
    if [row["reference_index"] for row in assignments] != list(
        range(reference_count)
    ):
        raise ValidationError(
            "semantic query matching reference order drifted"
        )
    if len({row["candidate_index"] for row in assignments}) != len(
        assignments
    ):
        raise ValidationError("semantic query matching reused a candidate")
    return assignments


def _match_person_prediction_sets(
    reference_boxes: Any,
    candidate_boxes: Any,
    reference_scores: Any,
    candidate_scores: Any,
    *,
    score_floor: float = PERSON_TASK_SCORE_MIN,
    iou_floor: float = SEMANTIC_MATCH_IOU_MIN,
) -> dict[str, Any]:
    """Match thresholded person predictions symmetrically.

    RF-DETR is sigmoid multi-label and the Noesis person task consumes class
    index 1 directly, independently of the query's all-class argmax.  Both
    reference and candidate sets are thresholded before matching so the audit
    exposes TensorRT-only predictions as well as dropped ONNX predictions.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    reference = np.asarray(reference_boxes, dtype=np.float64)
    candidate = np.asarray(candidate_boxes, dtype=np.float64)
    ref_score = np.asarray(reference_scores, dtype=np.float64)
    cand_score = np.asarray(candidate_scores, dtype=np.float64)
    if (
        reference.ndim != 2
        or candidate.ndim != 2
        or reference.shape[1:] != (4,)
        or candidate.shape[1:] != (4,)
        or ref_score.shape != (reference.shape[0],)
        or cand_score.shape != (candidate.shape[0],)
        or not np.isfinite(reference).all()
        or not np.isfinite(candidate).all()
        or not np.isfinite(ref_score).all()
        or not np.isfinite(cand_score).all()
        or np.any(reference[:, 2:] <= 0.0)
        or np.any(candidate[:, 2:] <= 0.0)
        or type(score_floor) not in {int, float}
        or not math.isfinite(float(score_floor))
        or type(iou_floor) not in {int, float}
        or not math.isfinite(float(iou_floor))
        or not 0.0 <= float(iou_floor) <= 1.0
    ):
        raise ValidationError("person prediction-set matching contract drifted")

    reference_queries = np.flatnonzero(ref_score >= float(score_floor))
    candidate_queries = np.flatnonzero(cand_score >= float(score_floor))
    matches: list[dict[str, Any]] = []
    matched_reference: set[int] = set()
    matched_candidate: set[int] = set()
    if reference_queries.size and candidate_queries.size:
        pairwise_iou = _cxcywh_iou(
            reference[reference_queries, None, :],
            candidate[None, candidate_queries, :],
        )
        spatial = pairwise_iou >= float(iou_floor)
        maximum_pairs = min(
            int(reference_queries.size), int(candidate_queries.size)
        )
        spatial_weight = maximum_pairs + 1
        utility = (
            spatial.astype(np.float64) * spatial_weight + pairwise_iou
        )
        reference_indexes, candidate_indexes = linear_sum_assignment(-utility)
        for reference_index, candidate_index in sorted(
            zip(reference_indexes, candidate_indexes, strict=True)
        ):
            ref_query = int(reference_queries[int(reference_index)])
            cand_query = int(candidate_queries[int(candidate_index)])
            is_spatial = bool(
                spatial[int(reference_index), int(candidate_index)]
            )
            matches.append(
                {
                    "reference_query": ref_query,
                    "candidate_query": cand_query,
                    "reference_score": float(ref_score[ref_query]),
                    "candidate_score": float(cand_score[cand_query]),
                    "iou": float(
                        pairwise_iou[
                            int(reference_index), int(candidate_index)
                        ]
                    ),
                    "spatial_match": is_spatial,
                }
            )
            if is_spatial:
                matched_reference.add(ref_query)
                matched_candidate.add(cand_query)

    return {
        "score_floor": float(score_floor),
        "iou_floor": float(iou_floor),
        "reference_queries": [
            int(value) for value in reference_queries.tolist()
        ],
        "candidate_queries": [
            int(value) for value in candidate_queries.tolist()
        ],
        "matches": matches,
        "unmatched_reference_queries": [
            int(value)
            for value in reference_queries.tolist()
            if int(value) not in matched_reference
        ],
        "unmatched_candidate_queries": [
            int(value)
            for value in candidate_queries.tolist()
            if int(value) not in matched_candidate
        ],
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValidationError("percentile input must contain finite values")
    if percentile < 0.0 or percentile > 100.0:
        raise ValidationError("percentile must be in [0,100]")
    return float(np.percentile(array, percentile))


def _summarize_distribution(values: Sequence[float]) -> dict[str, float | int]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValidationError("distribution must contain finite values")
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p05": _percentile(array, 5.0),
        "median": _percentile(array, 50.0),
        "p95": _percentile(array, 95.0),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _apply_quality_gates(
    family: str,
    metrics: Mapping[str, Any],
    *,
    scope: str = "all_classes",
) -> list[dict[str, Any]]:
    if scope == "all_classes":
        if family == "keypoint":
            raise ValidationError(
                "keypoint models do not have an all-class conversion gate"
            )
        gates: list[tuple[str, str, float, str]] = [
            ("class_agreement", "class_agreement_min", DETECTION_GATES["class_agreement_min"], "min"),
            ("retention", "retention_min", DETECTION_GATES["retention_min"], "min"),
            ("box_iou.median", "box_iou_median_min", DETECTION_GATES["box_iou_median_min"], "min"),
            ("box_iou.p05", "box_iou_p05_min", DETECTION_GATES["box_iou_p05_min"], "min"),
            ("score_abs_error.p95", "score_abs_error_p95_max", DETECTION_GATES["score_abs_error_p95_max"], "max"),
        ]
    elif scope == "person_task":
        gates = [
            (
                "reference_retention",
                "reference_retention_min",
                PERSON_TASK_GATES["reference_retention_min"],
                "min",
            ),
            (
                "candidate_precision",
                "candidate_precision_min",
                PERSON_TASK_GATES["candidate_precision_min"],
                "min",
            ),
            (
                "box_iou.median",
                "box_iou_median_min",
                PERSON_TASK_GATES["box_iou_median_min"],
                "min",
            ),
            (
                "box_iou.p05",
                "box_iou_p05_min",
                PERSON_TASK_GATES["box_iou_p05_min"],
                "min",
            ),
        ]
        if family == "keypoint":
            gates.extend(
                [
                    (
                        "base_score_abs_error.p95",
                        "base_score_abs_error_p95_max",
                        KEYPOINT_GATES[
                            "base_score_abs_error_p95_max"
                        ],
                        "max",
                    ),
                    (
                        "fused_score_relative_error.p95",
                        "fused_score_relative_error_p95_max",
                        KEYPOINT_GATES[
                            "fused_score_relative_error_p95_max"
                        ],
                        "max",
                    ),
                ]
            )
        else:
            gates.append(
                (
                    "score_abs_error.p95",
                    "score_abs_error_p95_max",
                    PERSON_TASK_GATES["score_abs_error_p95_max"],
                    "max",
                )
            )
    else:
        raise ValidationError(f"unsupported quality-gate scope: {scope}")

    if family == "segmentation":
        gates.extend(
            [
                ("mask_iou.median", "mask_iou_median_min", SEGMENTATION_GATES["mask_iou_median_min"], "min"),
                ("mask_iou.p05", "mask_iou_p05_min", SEGMENTATION_GATES["mask_iou_p05_min"], "min"),
                (
                    "mask_area_relative_error.p95",
                    "mask_area_relative_error_p95_max",
                    SEGMENTATION_GATES["mask_area_relative_error_p95_max"],
                    "max",
                ),
            ]
        )
    elif family == "keypoint":
        if scope != "person_task":
            raise ValidationError(
                "keypoint geometry gates require the person-task scope"
            )
        gates.extend(
            [
                ("oks.median", "oks_median_min", KEYPOINT_GATES["oks_median_min"], "min"),
                ("oks.min", "oks_min_min", KEYPOINT_GATES["oks_min_min"], "min"),
                (
                    "coordinate_error_box_diagonal.p95",
                    "coordinate_error_box_diagonal_p95_max",
                    KEYPOINT_GATES["coordinate_error_box_diagonal_p95_max"],
                    "max",
                ),
                (
                    "visible_jaccard.min",
                    "visible_jaccard_min",
                    KEYPOINT_GATES["visible_jaccard_min"],
                    "min",
                ),
            ]
        )

    def lookup(path: str) -> float:
        value: Any = metrics
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise ValidationError(f"missing quality metric: {path}")
            value = value[part]
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise ValidationError(f"quality metric is not finite: {path}")
        return float(value)

    results = []
    for metric_path, gate_name, threshold, direction in gates:
        observed = lookup(metric_path)
        passed = observed >= threshold if direction == "min" else observed <= threshold
        results.append(
            {
                "name": gate_name,
                "metric": metric_path,
                "direction": direction,
                "threshold": threshold,
                "observed": observed,
                "passed": passed,
            }
        )
    return results


def _builder_module() -> Any:
    spec = importlib.util.spec_from_file_location("rfdetr_1_8_3_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise ValidationError(f"cannot load engine builder: {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_checked(
    command: Sequence[str],
    *,
    label: str,
    timeout_seconds: int,
    cwd: Path | None = None,
) -> str:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValidationError(f"{label} failed: {exc}") from exc
    output = result.stdout or ""
    if len(output.encode("utf-8", errors="replace")) > MAX_LOG_BYTES:
        raise ValidationError(f"{label} transcript exceeded {MAX_LOG_BYTES} bytes")
    if result.returncode != 0:
        raise ValidationError(
            f"{label} exited {result.returncode}:\n{output.rstrip()}"
        )
    return output


def _copy_private(source: Path, target: Path) -> None:
    _require_regular_file(source, "media source")
    if target.exists() or target.is_symlink():
        raise ValidationError(f"media destination already exists: {target}")
    descriptor = _open_exclusive(target)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb", closefd=False) as writer:
            shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
    except BaseException:
        os.close(descriptor)
        target.unlink(missing_ok=True)
        raise
    os.close(descriptor)


def _extract_media(media_dir: Path, run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from PIL import Image

    media_out = _ensure_private_directory(run_dir / "media")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValidationError("ffmpeg is required to extract validation frames")
    videos: list[dict[str, Any]] = []
    frames_by_room: dict[str, dict[int, dict[str, Any]]] = {}
    selected = "+".join(f"eq(n\\,{index})" for index in FRAME_INDICES)

    for room in ROOMS:
        row = VIDEO_ROWS[room]
        source = media_dir / str(row["filename"])
        _require_regular_file(source, f"{room} validation clip")
        observed_sha = _sha256(source)
        if observed_sha != row["sha256"]:
            raise ValidationError(
                f"{room} clip SHA-256 mismatch: expected={row['sha256']} observed={observed_sha}"
            )
        videos.append(
            {
                "room": room,
                "filename": row["filename"],
                "sha256": observed_sha,
                "size_bytes": source.stat().st_size,
                "expected_resolution": [row["width"], row["height"]],
            }
        )
        staging = _ensure_private_directory(media_out / f".{room}-extract")
        pattern = staging / "frame-%02d.png"
        command = [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(source),
            "-vf",
            f"select={selected},format=rgb24",
            "-fps_mode",
            "vfr",
            "-frames:v",
            str(len(FRAME_INDICES)),
            "-c:v",
            "png",
            "-compression_level",
            "9",
            str(pattern),
        ]
        _run_checked(command, label=f"{room} deterministic frame extraction", timeout_seconds=300)
        extracted = sorted(staging.glob("frame-*.png"))
        if len(extracted) != len(FRAME_INDICES):
            raise ValidationError(
                f"{room} extraction returned {len(extracted)} frames, expected {len(FRAME_INDICES)}"
            )
        frames_by_room[room] = {}
        for frame_index, temporary in zip(FRAME_INDICES, extracted, strict=True):
            target = media_out / f"{room}-f{frame_index:04d}.png"
            if target.exists() or target.is_symlink():
                raise ValidationError(f"frame destination already exists: {target}")
            os.replace(temporary, target)
            os.chmod(target, 0o600)
            with Image.open(target) as image:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                width, height = image.size
            if [width, height] != [row["width"], row["height"]]:
                raise ValidationError(
                    f"{room} frame resolution drifted: {(width, height)}"
                )
            frames_by_room[room][frame_index] = {
                "room": room,
                "frame_index": frame_index,
                "occupied": OCCUPANCY[room][frame_index],
                "path": _relative(target, run_dir),
                "sha256": _sha256(target),
                "size_bytes": target.stat().st_size,
                "width": width,
                "height": height,
            }
        staging.rmdir()

    cases = []
    for frame_index in FRAME_INDICES:
        cases.append(
            {
                "id": f"f{frame_index:04d}",
                "frame_index": frame_index,
                "images": [
                    frames_by_room[room][frame_index] for room in ROOMS
                ],
            }
        )
    return videos, cases


def _write_float_array(path: Path, array: Any) -> dict[str, Any]:
    import numpy as np

    value = np.asarray(array, dtype="<f4", order="C")
    if not np.isfinite(value).all():
        raise ValidationError(f"refusing to write non-finite tensor: {path}")
    _write_bytes_exclusive(path, value.tobytes(order="C"))
    return {
        "path": path,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "shape": [int(item) for item in value.shape],
        "dtype": "float32",
        "finite": True,
        "min": float(value.min()),
        "max": float(value.max()),
    }


def _prepare_references(
    run_dir: Path,
    cases: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    *,
    runtime_engine_profile: str = "",
) -> list[dict[str, Any]]:
    import gc

    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image
    from rfdetr.datasets.transforms import Normalize
    from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

    if sys.byteorder != "little":
        raise ValidationError("TensorRT media evidence requires a little-endian host")
    builder = _builder_module()
    matrix = _load_matrix()
    release = dict(matrix["release"])
    inputs_root = _ensure_private_directory(run_dir / "inputs")
    reference_root = _ensure_private_directory(run_dir / "reference")
    model_records: list[dict[str, Any]] = []

    for model in selected:
        model_id = str(model["id"])
        if runtime_engine_profile:
            paths = builder._runtime_engine_artifact_paths(
                artifact_root, model, runtime_engine_profile
            )
            try:
                source = builder._validate_runtime_onnx_receipt(
                    artifact_root,
                    paths,
                    model,
                    release,
                )
            except Exception as exc:
                raise ValidationError(
                    f"{model_id} runtime ONNX provenance validation failed: "
                    f"{exc}"
                ) from exc
        else:
            paths = builder._artifact_paths(artifact_root, model)
            try:
                source = builder._validate_onnx_receipt(
                    paths, model, release
                )
            except Exception as exc:
                raise ValidationError(
                    f"{model_id} ONNX provenance validation failed: {exc}"
                ) from exc
        input_dir = _ensure_private_directory(inputs_root / model_id)
        output_dir = _ensure_private_directory(reference_root / model_id)
        resolution = int(model["resolution"])
        transform_steps: list[Any] = [
            Resize((resolution, resolution)),
            ToImage(),
            ToDtype(torch.float32, scale=True),
        ]
        if not runtime_engine_profile:
            transform_steps.append(Normalize())
        transform = Compose(transform_steps)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = max(1, min(8, os.cpu_count() or 1))
        started = time.monotonic()
        session = ort.InferenceSession(
            str(paths.onnx),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise ValidationError(
                f"{model_id} ONNX Runtime provider drifted: {session.get_providers()}"
            )
        input_meta = session.get_inputs()
        output_meta = session.get_outputs()
        expected_input = [3, 3, resolution, resolution]
        expected_outputs = {
            str(name): [int(value) for value in shape]
            for name, shape in dict(model["outputs"]).items()
        }
        if (
            len(input_meta) != 1
            or input_meta[0].name != "input"
            or input_meta[0].shape != expected_input
            or input_meta[0].type != "tensor(float)"
            or [item.name for item in output_meta] != list(expected_outputs)
            or any(
                item.shape != expected_outputs[item.name]
                or item.type != "tensor(float)"
                for item in output_meta
            )
        ):
            raise ValidationError(f"{model_id} ONNX tensor contract drifted")

        case_records = []
        for case in cases:
            tensors = []
            for image_row in case["images"]:
                image_path = run_dir / str(image_row["path"])
                with Image.open(image_path) as image:
                    rgb = image.convert("RGB")
                    tensor, _ = transform(rgb, None)
                tensors.append(tensor)
            batch = torch.stack(tensors).contiguous().numpy().astype("<f4", copy=False)
            if runtime_engine_profile and (
                float(batch.min()) < 0.0 or float(batch.max()) > 1.0
            ):
                raise ValidationError(
                    f"{model_id} {case['id']} runtime input escaped RGB [0,1]"
                )
            input_path = input_dir / f"{case['id']}.bin"
            input_record = _write_float_array(input_path, batch)
            output_started = time.monotonic()
            raw_outputs = session.run(None, {"input": batch})
            output_seconds = time.monotonic() - output_started
            outputs = {}
            for metadata, raw in zip(output_meta, raw_outputs, strict=True):
                array = np.asarray(raw, dtype="<f4", order="C")
                if list(array.shape) != expected_outputs[metadata.name]:
                    raise ValidationError(
                        f"{model_id} {case['id']} {metadata.name} shape drifted"
                    )
                output_path = output_dir / f"{case['id']}__{metadata.name}.bin"
                record = _write_float_array(output_path, array)
                record["path"] = _relative(record["path"], run_dir)
                outputs[metadata.name] = record
            input_record["path"] = _relative(input_record["path"], run_dir)
            case_records.append(
                {
                    "id": case["id"],
                    "input": input_record,
                    "outputs": outputs,
                    "inference_seconds": output_seconds,
                }
            )
            print(
                f"[REFERENCE] {model_id} {case['id']} {output_seconds:.2f}s",
                flush=True,
            )
        model_record: dict[str, Any] = {
            "id": model_id,
            "family": model["family"],
            "variant": model["variant"],
            "resolution": resolution,
            "onnx": {
                "path": _relative(paths.onnx, artifact_root),
                "size_bytes": source["size_bytes"],
                "sha256": source["sha256"],
                "receipt": _relative(paths.onnx_receipt, artifact_root),
                "receipt_sha256": source["receipt_sha256"],
            },
            "expected_input": expected_input,
            "expected_outputs": expected_outputs,
            "cases": case_records,
            "session_initialization_and_inference_seconds": (
                time.monotonic() - started
            ),
        }
        if runtime_engine_profile:
            model_record["onnx"]["artifact_kind"] = "runtime_onnx"
            model_record["onnx"]["input_contract"] = RUNTIME_INPUT_CONTRACT
            model_record["onnx"][
                "adapter_revision"
            ] = RUNTIME_ADAPTER_REVISION
            model_record["onnx"]["adapter"] = dict(source["adapter"])
        model_records.append(model_record)
        del session, transform
        gc.collect()
    return model_records


def _prepare(args: argparse.Namespace) -> int:
    if os.getuid() == 0:
        raise ValidationError("RF-DETR media validation must run as a non-root caller")
    artifact_root = _explicit_external_root(
        os.environ.get("NOESIS_DS9_ARTIFACT_ROOT", ""),
        "NOESIS_DS9_ARTIFACT_ROOT",
    )
    media_dir = _explicit_external_root(args.media_dir, "--media-dir")
    if not artifact_root.is_dir() or not media_dir.is_dir():
        raise ValidationError("artifact root and media directory must already exist")
    matrix = _load_matrix()
    selected = _select_models(matrix, args.model)
    runtime_engine_profile = _runtime_engine_profile(
        getattr(args, "runtime_engine_profile", "")
    )
    run_id = _safe_token(args.run_id or _default_run_id(), "run ID")
    evidence_root_path = (
        artifact_root / "models" / "validation" / "rfdetr" / RELEASE_VERSION
    )
    if runtime_engine_profile:
        evidence_root_path = (
            evidence_root_path / "runtime" / runtime_engine_profile
        )
    evidence_root = _ensure_private_directory(evidence_root_path)
    run_dir = evidence_root / run_id
    if run_dir.exists() or run_dir.is_symlink():
        raise ValidationError(f"validation run already exists: {run_dir}")
    run_dir.mkdir(mode=0o700)
    os.chmod(run_dir, 0o700)
    _require_private_directory(run_dir, "validation run")
    try:
        videos, cases = _extract_media(media_dir, run_dir)
        models = _prepare_references(
            run_dir,
            cases,
            selected,
            artifact_root,
            runtime_engine_profile=runtime_engine_profile,
        )
        preprocess: dict[str, Any] = {
            "resize": "direct_square_bilinear",
            "color": "RGB",
            "scale": "uint8_to_float32_[0,1]",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "letterbox": False,
        }
        if runtime_engine_profile:
            preprocess.update(
                {
                    "mean": None,
                    "std": None,
                    "input_contract": RUNTIME_INPUT_CONTRACT,
                    "runtime_adapter_revision": (
                        RUNTIME_ADAPTER_REVISION
                    ),
                    "normalization_location": (
                        "revision_bound_runtime_onnx"
                    ),
                }
            )
        manifest: dict[str, Any] = {
            "schema": RUN_SCHEMA,
            "release": dict(matrix["release"]),
            "run_id": run_id,
            "created_at_utc": _utc_now(),
            "promotion_status": "unpromoted",
            "runtime_selected": False,
            "privacy": {
                "classification": "owner_private_identifiable_home_media",
                "directory_mode": "0700",
                "file_mode": "0600",
                "commit_or_upload": "forbidden",
            },
            "workflow": {
                "batch_mode": "static",
                "batch_size": 3,
                "batch_order": list(ROOMS),
                "frame_indices": list(FRAME_INDICES),
                "preprocess": preprocess,
                "reference": {
                    "backend": "onnxruntime",
                    "provider": "CPUExecutionProvider",
                    "version": __import__("onnxruntime").__version__,
                },
            },
            "source_files": {
                "matrix": {
                    "path": "DS9/config/rfdetr_1_8_3_models.json",
                    "sha256": _sha256(MATRIX_PATH),
                },
                "runner": {
                    "path": "DS9/native/rfdetr_1_8_3_media_runner.cpp",
                    "sha256": _sha256(RUNNER_SOURCE),
                },
                "validator": {
                    "path": "DS9/scripts/validate_rfdetr_1_8_3_media.py",
                    "sha256": _sha256(Path(__file__)),
                },
            },
            "videos": videos,
            "cases": cases,
            "models": models,
        }
        if runtime_engine_profile:
            manifest["artifact_role"] = RUNTIME_VALIDATION_ARTIFACT_ROLE
            manifest["runtime_engine_profile"] = runtime_engine_profile
        manifest["manifest_sha256"] = _json_digest(manifest)
        _write_json_exclusive(run_dir / "run.json", manifest)
        print(f"[OK] CPU/ONNX reference preparation complete: {run_dir}")
        print(f"[RUN_DIR] {run_dir}")
        return 0
    except BaseException:
        failure = {
            "schema": "noesis.ds9.rfdetr-media-validation-failure.v1",
            "phase": "prepare",
            "recorded_at_utc": _utc_now(),
        }
        if runtime_engine_profile:
            failure["runtime_engine_profile"] = runtime_engine_profile
        if not (run_dir / "failure.json").exists():
            try:
                _write_json_exclusive(run_dir / "failure.json", failure)
            except BaseException:
                pass
        raise


def _validated_manifest_digest(payload: Mapping[str, Any], label: str) -> None:
    recorded = payload.get("manifest_sha256")
    if not isinstance(recorded, str) or SHA256_RE.fullmatch(recorded) is None:
        raise ValidationError(f"{label} lacks a valid manifest digest")
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    if _json_digest(unsigned) != recorded:
        raise ValidationError(f"{label} manifest digest mismatch")


def _load_run(
    run_dir: Path,
    *,
    allow_validator_drift: bool = False,
    allow_prepared_source_drift: bool = False,
) -> dict[str, Any]:
    _require_private_directory(run_dir, "validation run")
    payload = _read_json(run_dir / "run.json", "validation run manifest")
    if (
        payload.get("schema") != RUN_SCHEMA
        or payload.get("release", {}).get("version") != RELEASE_VERSION
        or payload.get("promotion_status") != "unpromoted"
        or payload.get("runtime_selected") is not False
    ):
        raise ValidationError("validation run manifest contract drifted")
    _prepared_runtime_profile(payload)
    _validated_manifest_digest(payload, "validation run")
    if payload.get("run_id") != run_dir.name:
        raise ValidationError("validation run directory identity mismatch")
    source_files = payload.get("source_files")
    if not isinstance(source_files, Mapping):
        raise ValidationError("validation run lacks source-file bindings")
    for key, path in (
        ("matrix", MATRIX_PATH),
        ("runner", RUNNER_SOURCE),
        ("validator", Path(__file__)),
    ):
        row = source_files.get(key)
        if not isinstance(row, Mapping):
            raise ValidationError(f"validation run lacks source binding: {key}")
        expected_sha256 = str(row.get("sha256") or "")
        if SHA256_RE.fullmatch(expected_sha256) is None:
            raise ValidationError(f"validation run has invalid source digest: {key}")
        if allow_prepared_source_drift or (
            key == "validator" and allow_validator_drift
        ):
            continue
        if expected_sha256 != _sha256(path):
            raise ValidationError(f"validation source changed since prepare: {key}")
    return payload


def _comparison_models_from_run(
    run: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Reconstruct the immutable prepared tensor contract for analysis.

    Successor reports must not silently adopt a newer live model matrix.  The
    preparation manifest already binds the selected model identity, resolution,
    and exact input/output shapes, which are the only model fields consumed by
    the comparison and overlay paths.
    """
    raw_models = run.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValidationError("validation run has no prepared models")
    models = []
    seen: set[str] = set()
    for row in raw_models:
        if not isinstance(row, Mapping):
            raise ValidationError("prepared model row must be an object")
        model_id = _safe_token(row.get("id"), "prepared model ID")
        family = str(row.get("family") or "")
        variant = _safe_token(row.get("variant"), "prepared model variant")
        resolution = row.get("resolution")
        outputs = row.get("expected_outputs")
        expected_input = row.get("expected_input")
        if (
            model_id in seen
            or family not in {"detection", "segmentation", "keypoint"}
            or type(resolution) is not int
            or resolution <= 0
            or expected_input != [3, 3, resolution, resolution]
            or not isinstance(outputs, Mapping)
            or not outputs
        ):
            raise ValidationError("prepared model tensor contract drifted")
        normalized_outputs: dict[str, list[int]] = {}
        for name, shape in outputs.items():
            safe_name = _safe_token(name, "prepared output tensor")
            if (
                not isinstance(shape, list)
                or not shape
                or any(type(value) is not int or value <= 0 for value in shape)
            ):
                raise ValidationError(
                    "prepared output tensor contract drifted"
                )
            normalized_outputs[safe_name] = list(shape)
        seen.add(model_id)
        models.append(
            {
                "id": model_id,
                "family": family,
                "variant": variant,
                "resolution": resolution,
                "outputs": normalized_outputs,
            }
        )
    return models


def _validate_runner_manifest(
    payload: Mapping[str, Any],
    model: Mapping[str, Any],
    case: str | Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    case_id = str(case["id"]) if isinstance(case, Mapping) else str(case)
    expected_input = [3, 3, int(model["resolution"]), int(model["resolution"])]
    expected_outputs = {
        str(name): [int(value) for value in shape]
        for name, shape in dict(model["outputs"]).items()
    }
    expected_root = {
        "schema",
        "case",
        "repeat_stable",
        "inference_elapsed_ms",
        "tensors",
        "input",
        "outputs",
    }
    if set(payload) != expected_root:
        raise ValidationError(f"{model['id']} {case_id} runner root schema drifted")
    if (
        payload.get("schema") != RUNNER_SCHEMA
        or payload.get("case") != case_id
        or payload.get("repeat_stable") is not True
    ):
        raise ValidationError(f"{model['id']} {case_id} runner envelope drifted")
    elapsed = payload.get("inference_elapsed_ms")
    if (
        not isinstance(elapsed, Mapping)
        or set(elapsed) != {"first", "repeat"}
        or any(
            type(elapsed.get(key)) not in {int, float}
            or not math.isfinite(float(elapsed[key]))
            or float(elapsed[key]) < 0
            for key in ("first", "repeat")
        )
    ):
        raise ValidationError(f"{model['id']} {case_id} runner timing is invalid")

    tensors = payload.get("tensors")
    if not isinstance(tensors, list) or len(tensors) != 1 + len(expected_outputs):
        raise ValidationError(f"{model['id']} {case_id} runner tensor list drifted")
    tensor_rows: dict[str, Mapping[str, Any]] = {}
    for row in tensors:
        if not isinstance(row, Mapping):
            raise ValidationError("runner tensor row must be an object")
        expected_keys = {
            "name",
            "mode",
            "location",
            "data_type",
            "shape",
            "element_count",
            "byte_count",
        }
        name = str(row.get("name") or "")
        if (
            set(row) != expected_keys
            or name in tensor_rows
            or _safe_token(name, "runner tensor") != name
            or row.get("data_type") != "fp32"
            or row.get("location") not in {"device", "host"}
        ):
            raise ValidationError(f"{model['id']} {case_id} invalid runner tensor row")
        tensor_rows[name] = row
    if list(tensor_rows) != ["input", *expected_outputs]:
        raise ValidationError(f"{model['id']} {case_id} runner tensor names drifted")

    def validate_tensor(row: Mapping[str, Any], shape: list[int], mode: str) -> None:
        elements = math.prod(shape)
        if (
            row.get("mode") != mode
            or row.get("shape") != shape
            or row.get("element_count") != elements
            or row.get("byte_count") != elements * 4
        ):
            raise ValidationError(
                f"{model['id']} {case_id} {row.get('name')} tensor contract drifted"
            )

    validate_tensor(tensor_rows["input"], expected_input, "input")
    for name, shape in expected_outputs.items():
        validate_tensor(tensor_rows[name], shape, "output")

    input_row = payload.get("input")
    if (
        not isinstance(input_row, Mapping)
        or set(input_row)
        != {"name", "shape", "byte_count", "finite", "min", "max"}
        or input_row.get("name") != "input"
        or input_row.get("shape") != expected_input
        or input_row.get("byte_count") != math.prod(expected_input) * 4
        or input_row.get("finite") is not True
        or any(
            type(input_row.get(key)) not in {int, float}
            or not math.isfinite(float(input_row[key]))
            for key in ("min", "max")
        )
    ):
        raise ValidationError(f"{model['id']} {case_id} runner input drifted")

    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(expected_outputs):
        raise ValidationError(f"{model['id']} {case_id} runner outputs drifted")
    by_name: dict[str, Mapping[str, Any]] = {}
    for row in outputs:
        if not isinstance(row, Mapping):
            raise ValidationError("runner output row must be an object")
        expected_keys = {
            "name",
            "shape",
            "byte_count",
            "file",
            "repeat_stable",
            "finite",
            "min",
            "max",
        }
        name = str(row.get("name") or "")
        expected_shape = expected_outputs.get(name)
        expected_file = f"{case_id}__{name}.bin"
        if (
            set(row) != expected_keys
            or expected_shape is None
            or name in by_name
            or row.get("shape") != expected_shape
            or row.get("byte_count") != math.prod(expected_shape) * 4
            or row.get("file") != expected_file
            or row.get("repeat_stable") is not True
            or row.get("finite") is not True
            or any(
                type(row.get(key)) not in {int, float}
                or not math.isfinite(float(row[key]))
                for key in ("min", "max")
            )
        ):
            raise ValidationError(
                f"{model['id']} {case_id} {name or '<unnamed>'} runner output drifted"
            )
        by_name[name] = row
    if list(by_name) != list(expected_outputs):
        raise ValidationError(f"{model['id']} {case_id} runner output order drifted")
    return by_name


def _validate_engine_receipt(
    artifact_root: Path,
    builder: Any,
    model: Mapping[str, Any],
    precision_canary_profile: Any = None,
    runtime_engine_profile: str = "",
) -> dict[str, Any]:
    profile = precision_canary_profile
    runtime_profile = _runtime_engine_profile(runtime_engine_profile)
    if profile is not None and runtime_profile:
        raise ValidationError(
            "precision-canary and runtime-engine profiles are mutually exclusive"
        )
    runtime_source: Mapping[str, Any] | None = None
    if runtime_profile:
        runtime_precision = builder._runtime_precision_profile(
            runtime_profile
        )
        baseline_runtime = (
            runtime_profile == BASELINE_RUNTIME_ENGINE_PROFILE
        )
        if runtime_precision is None and not baseline_runtime:
            raise ValidationError(
                "runtime media validation received an unknown baseline profile"
            )
        paths = builder._runtime_engine_artifact_paths(
            artifact_root, model, runtime_profile
        )
        expected_schema = builder.RUNTIME_ENGINE_RECEIPT_SCHEMA
        expected_precision = (
            "fp16"
            if runtime_precision is None
            else runtime_precision.precision
        )
        try:
            runtime_source = builder._validate_runtime_onnx_receipt(
                artifact_root,
                paths,
                model,
                _load_matrix()["release"],
            )
        except Exception as exc:
            raise ValidationError(
                f"{model['id']} runtime ONNX provenance validation failed: "
                f"{exc}"
            ) from exc
    elif profile is None:
        paths = builder._artifact_paths(artifact_root, model)
        expected_schema = builder.ENGINE_RECEIPT_SCHEMA
        expected_precision = "fp16"
    else:
        paths = builder._precision_canary_artifact_paths(
            artifact_root, model, profile
        )
        expected_schema = builder.CANARY_ENGINE_RECEIPT_SCHEMA
        expected_precision = profile.precision
    _require_regular_file(paths.engine, f"{model['id']} TensorRT engine")
    _require_regular_file(paths.engine_receipt, f"{model['id']} engine receipt")
    receipt = _read_json(paths.engine_receipt, f"{model['id']} engine receipt")
    engine = receipt.get("engine")
    build = receipt.get("build_contract")
    platform = receipt.get("platform")
    source = receipt.get("source")
    expected_outputs = {
        str(name): [int(value) for value in shape]
        for name, shape in dict(model["outputs"]).items()
    }
    expected_input = [3, 3, int(model["resolution"]), int(model["resolution"])]
    if (
        receipt.get("schema") != expected_schema
        or receipt.get("promotion_status") != "unpromoted"
        or receipt.get("runtime_selected") is not False
        or receipt.get("model_id") != model["id"]
        or receipt.get("family") != model["family"]
        or not isinstance(engine, Mapping)
        or engine.get("path") != _relative(paths.engine, artifact_root)
        or engine.get("size_bytes") != paths.engine.stat().st_size
        or engine.get("sha256") != _sha256(paths.engine)
        or not isinstance(build, Mapping)
        or build.get("precision") != expected_precision
        or build.get("batch") != {"mode": "static", "size": 3}
        or not isinstance(platform, Mapping)
        or platform.get("tensorrt_version") != builder.REQUIRED_TRT_VERSION
        or platform.get("image_id") != builder.REQUIRED_IMAGE_ID
        or not isinstance(source, Mapping)
        or source.get("tensor_contract", {}).get("input") != {"input": expected_input}
        or source.get("tensor_contract", {}).get("outputs") != expected_outputs
    ):
        raise ValidationError(f"{model['id']} engine receipt contract drifted")
    if profile is not None:
        if (
            receipt.get("artifact_role") != "precision_canary"
            or build.get("profile") != profile.id
            or build.get("fp16_enabled") is not profile.fp16_enabled
            or build.get("tf32_enabled") is not profile.tf32_enabled
            or not isinstance(receipt.get("baseline_engine"), Mapping)
            or not isinstance(receipt.get("layer_info"), Mapping)
        ):
            raise ValidationError(
                f"{model['id']} precision canary receipt contract drifted"
            )
    if runtime_profile:
        assert runtime_source is not None
        runtime_precision = builder._runtime_precision_profile(
            runtime_profile
        )
        baseline_runtime = (
            runtime_profile == BASELINE_RUNTIME_ENGINE_PROFILE
        )
        if runtime_precision is None and not baseline_runtime:
            raise ValidationError(
                f"{model['id']} runtime precision profile drifted"
            )
        expected_source = {
            "runtime_onnx": _relative(paths.onnx, artifact_root),
            "runtime_onnx_size_bytes": runtime_source["size_bytes"],
            "runtime_onnx_sha256": runtime_source["sha256"],
            "runtime_onnx_receipt": _relative(
                paths.onnx_receipt, artifact_root
            ),
            "runtime_onnx_receipt_sha256": runtime_source[
                "receipt_sha256"
            ],
            "tensor_contract": runtime_source["tensor_contract"],
        }
        normalized = runtime_source["normalized_source"]
        expected_runtime_contract = {
            "input_contract": builder.RUNTIME_INPUT_CONTRACT,
            "adapter_revision": builder.RUNTIME_ADAPTER_REVISION,
            "adapter": dict(builder.RUNTIME_ADAPTER_SPEC),
            "normalized_source": {
                "onnx": _relative(
                    Path(normalized["path"]), artifact_root
                ),
                "onnx_size_bytes": normalized["size_bytes"],
                "onnx_sha256": normalized["sha256"],
                "onnx_receipt": _relative(
                    Path(normalized["receipt_path"]), artifact_root
                ),
                "onnx_receipt_sha256": normalized["receipt_sha256"],
            },
        }
        expected_build = builder._runtime_build_contract(
            model,
            runtime_profile,
            runtime_precision,
        )
        expected_commands = [
            list(
                builder._inner_build_command(
                    f"{paths.engine.name}.candidate",
                    model=model,
                    precision_profile=runtime_precision,
                    layer_info_name=(
                        None
                        if baseline_runtime
                        else f"{paths.engine.name}.layers.json"
                    ),
                )
            ),
            list(builder._inner_load_command()),
            list(builder._inner_load_command()),
        ]
        layer_info = receipt.get("layer_info")
        commands = receipt.get("commands")
        if (
            receipt.get("artifact_role") != "runtime_input_engine"
            or "baseline_engine" in receipt
            or build != expected_build
            or source != expected_source
            or receipt.get("runtime_input_contract")
            != expected_runtime_contract
            or (
                baseline_runtime
                and (
                    build.get("layer_info_exported") is not False
                    or "layer_info" in receipt
                )
            )
            or (
                not baseline_runtime
                and (
                    build.get("layer_info_exported") is not True
                    or not isinstance(layer_info, Mapping)
                )
            )
            or not isinstance(commands, list)
            or len(commands) != 3
            or any(not isinstance(row, Mapping) for row in commands)
            or [row.get("label") for row in commands]
            != ["build", "load-candidate", "load-installed"]
            or [row.get("command") for row in commands]
            != expected_commands
            or any(
                row.get("status") != "passed"
                or row.get("returncode") != 0
                for row in commands
            )
            or platform.get("image_ref") != builder.REQUIRED_IMAGE_REF
            or platform.get("base_digest") != builder.REQUIRED_BASE_DIGEST
            or platform.get("image_tensorrt_version")
            != builder.REQUIRED_IMAGE_TRT_VERSION
            or platform.get("cuda_version") != builder.REQUIRED_CUDA_VERSION
            or platform.get("trtexec_banner") != builder.REQUIRED_TRT_BANNER
            or platform.get("expected_trtexec_banner")
            != builder.REQUIRED_TRT_BANNER
        ):
            raise ValidationError(
                f"{model['id']} runtime engine receipt contract drifted"
            )
        for row in commands:
            relative_log = str(row["log"])
            log_path = artifact_root / relative_log
            if _relative(log_path, artifact_root) != relative_log:
                raise ValidationError(
                    f"{model['id']} runtime engine command path drifted"
                )
            _require_regular_file(
                log_path, f"{model['id']} runtime engine command log"
            )
            if row.get("log_sha256") != _sha256(log_path):
                raise ValidationError(
                    f"{model['id']} runtime engine command evidence drifted"
                )
        if not baseline_runtime:
            assert isinstance(layer_info, Mapping)
            relative_layer = str(layer_info.get("path") or "")
            layer_path = artifact_root / relative_layer
            if _relative(layer_path, artifact_root) != relative_layer:
                raise ValidationError(
                    f"{model['id']} runtime engine layer-info path drifted"
                )
            _require_regular_file(
                layer_path, f"{model['id']} runtime engine layer info"
            )
            if (
                layer_info.get("size_bytes") != layer_path.stat().st_size
                or layer_info.get("sha256") != _sha256(layer_path)
            ):
                raise ValidationError(
                    f"{model['id']} runtime engine layer info drifted"
                )
        receipt_digest = receipt.get("receipt_sha256")
        unsigned = dict(receipt)
        unsigned.pop("receipt_sha256", None)
        if (
            not isinstance(receipt_digest, str)
            or receipt_digest != _json_digest(unsigned)
        ):
            raise ValidationError(
                f"{model['id']} runtime engine receipt digest drifted"
            )
    return {
        "path": paths.engine,
        "receipt_path": paths.engine_receipt,
        "size_bytes": paths.engine.stat().st_size,
        "sha256": _sha256(paths.engine),
        "receipt_sha256": _sha256(paths.engine_receipt),
    }


def _run_container(
    context: Any,
    builder: Any,
    *,
    name: str,
    gpu: bool,
    mounts: Sequence[tuple[Path, str, bool]],
    entrypoint: str,
    arguments: Sequence[str],
    labels: Mapping[str, str],
    timeout_seconds: int,
) -> str:
    safe_name = re.sub(r"[^a-z0-9_.-]+", "-", name.lower())[:128]
    command = [
        *context.command,
        "create",
        "--name",
        safe_name,
        *builder._container_security_args(gpu=gpu),
    ]
    for key, value in labels.items():
        command.extend(["--label", f"{key}={value}"])
    for source, destination, writable in mounts:
        _require_regular_file(source, "container mount source") if source.is_file() else None
        mount = f"type=bind,src={source},dst={destination}"
        if not writable:
            mount += ",readonly"
        command.extend(["--mount", mount])
    command.extend(
        [
            "--entrypoint",
            entrypoint,
            builder.REQUIRED_IMAGE_ID,
            *arguments,
        ]
    )
    if gpu:
        builder._assert_no_compute_owners(f"before {safe_name} creation")
    raw_id = _run_checked(
        command,
        label=f"{safe_name} container creation",
        timeout_seconds=120,
    ).strip()
    if re.fullmatch(r"[0-9a-f]{64}", raw_id) is None:
        raise ValidationError(f"Docker returned an invalid container ID: {raw_id}")
    try:
        if gpu:
            builder._assert_no_compute_owners(f"before {safe_name} start")
        return _run_checked(
            [*context.command, "start", "--attach", raw_id],
            label=f"{safe_name} container execution",
            timeout_seconds=timeout_seconds,
        )
    finally:
        cleanup = subprocess.run(
            [*context.command, "rm", "-f", raw_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
        if cleanup.returncode != 0:
            raise ValidationError(
                f"cannot remove owned validation container {raw_id}: "
                f"{(cleanup.stdout or '').strip()}"
            )


def _compile_runner(
    run_dir: Path,
    context: Any,
    builder: Any,
    run_id: str,
    *,
    directory_name: str = "bin",
) -> dict[str, Any]:
    binary_dir = _ensure_private_directory(
        run_dir / _safe_token(directory_name, "runner directory")
    )
    binary = binary_dir / "rfdetr_1_8_3_media_runner"
    if binary.exists() or binary.is_symlink():
        raise ValidationError(f"runner binary already exists: {binary}")
    compile_command = (
        "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror "
        "-I/usr/local/cuda/include /src/runner.cpp -o /work/runner "
        "-L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 "
        "-lnvinfer -lnvinfer_plugin -lcudart"
    )
    transcript = _run_container(
        context,
        builder,
        name=f"noesis-rfdetr-media-compile-{hashlib.sha256(run_id.encode()).hexdigest()[:12]}",
        gpu=False,
        mounts=(
            (RUNNER_SOURCE, "/src/runner.cpp", False),
            (binary_dir, "/work", True),
        ),
        entrypoint="/bin/sh",
        arguments=("-c", compile_command),
        labels={
            "noesis.ds9.role": "rfdetr-1.8.3-media-runner-compile",
            "noesis.ds9.run": run_id,
        },
        timeout_seconds=COMPILE_TIMEOUT_SECONDS,
    )
    compiled = binary_dir / "runner"
    _require_regular_file(compiled, "compiled TensorRT media runner")
    info = compiled.stat()
    if info.st_uid != os.getuid() or not (info.st_mode & stat.S_IXUSR):
        raise ValidationError("compiled media runner is not caller-owned/executable")
    os.replace(compiled, binary)
    os.chmod(binary, 0o700)
    log = run_dir / (
        "compile.log"
        if directory_name == "bin"
        else f"compile-{directory_name}.log"
    )
    _write_bytes_exclusive(log, transcript.encode("utf-8", errors="replace"))
    return {
        "path": binary,
        "sha256": _sha256(binary),
        "size_bytes": binary.stat().st_size,
        "source_sha256": _sha256(RUNNER_SOURCE),
        "command": compile_command,
        "log": _relative(log, run_dir),
        "log_sha256": _sha256(log),
    }


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    flags = os.O_RDWR | os.O_CREAT
    for name in ("O_CLOEXEC", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ValidationError("validation lock must be caller-owned mode 0600")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValidationError("another RF-DETR media validation owns the lock") from exc
        yield
    finally:
        os.close(descriptor)


def _trt(args: argparse.Namespace) -> int:
    if os.getuid() == 0:
        raise ValidationError("RF-DETR media validation must run as a non-root caller")
    artifact_root = _explicit_external_root(
        os.environ.get("NOESIS_DS9_ARTIFACT_ROOT", ""),
        "NOESIS_DS9_ARTIFACT_ROOT",
    )
    docker_root = _explicit_external_root(
        os.environ.get("NOESIS_DS9_DOCKER_ROOT", ""),
        "NOESIS_DS9_DOCKER_ROOT",
    )
    if (
        artifact_root == docker_root
        or artifact_root in docker_root.parents
        or docker_root in artifact_root.parents
    ):
        raise ValidationError("artifact and Docker roots must be disjoint")
    builder = _builder_module()
    precision_profile = builder._precision_profile(
        getattr(args, "precision_canary_profile", "")
    )
    runtime_engine_profile = _runtime_engine_profile(
        getattr(args, "runtime_engine_profile", "")
    )
    if precision_profile is not None and runtime_engine_profile:
        raise ValidationError(
            "precision-canary and runtime-engine profiles are mutually exclusive"
        )
    profile_id = precision_profile.id if precision_profile is not None else ""
    if runtime_engine_profile:
        execution_scope = f"runtime-{runtime_engine_profile}"
        receipt_filename = f"trt-{execution_scope}.json"
        output_directory = f"trt-{execution_scope}"
        runner_directory = f"bin-{execution_scope}"
    elif profile_id:
        execution_scope = f"canary-{profile_id}"
        receipt_filename = f"trt-{execution_scope}.json"
        output_directory = f"trt-{execution_scope}"
        runner_directory = f"bin-{execution_scope}"
    else:
        execution_scope = "baseline"
        receipt_filename = "trt.json"
        output_directory = "trt"
        runner_directory = "bin"
    run_dir = Path(args.run_dir).expanduser().resolve()
    run = _load_run(
        run_dir,
        allow_validator_drift=precision_profile is not None,
    )
    prepared_runtime_profile = _prepared_runtime_profile(run)
    if prepared_runtime_profile != runtime_engine_profile:
        raise ValidationError(
            "prepared input contract/runtime engine profile mismatch: "
            f"prepared={prepared_runtime_profile or 'normalized'} "
            f"requested={runtime_engine_profile or 'normalized'}"
        )
    if (run_dir / receipt_filename).exists() or (
        run_dir / receipt_filename
    ).is_symlink():
        raise ValidationError("TensorRT phase already exists for this validation run")
    matrix = _load_matrix()
    requested = [str(row["id"]) for row in run["models"]]
    selected = _select_models(matrix, requested)
    context = builder._docker_context(docker_root)
    platform = builder._inspect_secondary_docker(context)
    probe = builder._probe_trtexec(context)
    if builder.REQUIRED_TRT_BANNER not in probe:
        raise ValidationError("pinned TensorRT banner probe failed")
    builder._assert_no_compute_owners("before RF-DETR media validation")
    platform.update(builder._gpu_identity())
    platform["tensorrt_version"] = builder.REQUIRED_TRT_VERSION
    runner = _compile_runner(
        run_dir,
        context,
        builder,
        str(run["run_id"]),
        directory_name=runner_directory,
    )
    trt_root = _ensure_private_directory(run_dir / output_directory)
    records = []
    lock = artifact_root / ".noesis-ds9-rfdetr-1.8.3-media-validation.lock"
    with _exclusive_lock(lock):
        for model, run_model in zip(selected, run["models"], strict=True):
            if model["id"] != run_model["id"]:
                raise ValidationError("model order drifted since reference preparation")
            engine = _validate_engine_receipt(
                artifact_root,
                builder,
                model,
                precision_canary_profile=precision_profile,
                runtime_engine_profile=runtime_engine_profile,
            )
            model_dir = _ensure_private_directory(trt_root / str(model["id"]))
            mounts: list[tuple[Path, str, bool]] = [
                (runner["path"], "/runner", False),
                (engine["path"], "/engine/model.engine", False),
                (model_dir, "/output", True),
            ]
            arguments: list[str] = [
                "--engine",
                "/engine/model.engine",
                "--output-dir",
                "/output",
            ]
            for case in run_model["cases"]:
                case_id = _safe_token(case["id"], "case ID")
                input_path = run_dir / str(case["input"]["path"])
                _require_regular_file(input_path, f"{model['id']} {case_id} input")
                if (
                    input_path.stat().st_size != case["input"]["size_bytes"]
                    or _sha256(input_path) != case["input"]["sha256"]
                ):
                    raise ValidationError(f"{model['id']} {case_id} input changed")
                if runtime_engine_profile:
                    runtime_input = _load_float_tensor(
                        input_path,
                        case["input"]["shape"],
                        expected_sha256=case["input"]["sha256"],
                        label=(
                            f"{model['id']} {case_id} runtime RGB01 input"
                        ),
                    )
                    if (
                        float(runtime_input.min()) < 0.0
                        or float(runtime_input.max()) > 1.0
                    ):
                        raise ValidationError(
                            f"{model['id']} {case_id} runtime input "
                            "escaped RGB [0,1]"
                        )
                    del runtime_input
                destination = f"/inputs/{case_id}.bin"
                mounts.append((input_path, destination, False))
                arguments.extend(["--input", f"{case_id}={destination}"])
            name_hash = hashlib.sha256(
                f"{run['run_id']}:{execution_scope}:{model['id']}".encode()
            ).hexdigest()[:12]
            started = time.monotonic()
            transcript = _run_container(
                context,
                builder,
                name=(
                    f"noesis-rfdetr-media-{execution_scope}-"
                    f"{model['id']}-{name_hash}"
                ),
                gpu=True,
                mounts=mounts,
                entrypoint="/runner",
                arguments=arguments,
                labels={
                    "noesis.ds9.role": "rfdetr-1.8.3-media-validation",
                    "noesis.ds9.model": str(model["id"]),
                    "noesis.ds9.run": str(run["run_id"]),
                    "noesis.ds9.precision": (
                        runtime_engine_profile
                        or profile_id
                        or "fp16_tf32"
                    ),
                    "noesis.ds9.input-contract": (
                        RUNTIME_INPUT_CONTRACT
                        if runtime_engine_profile
                        else "imagenet_normalized"
                    ),
                },
                timeout_seconds=INFERENCE_TIMEOUT_SECONDS,
            )
            log = model_dir / "runner.log"
            _write_bytes_exclusive(log, transcript.encode("utf-8", errors="replace"))
            case_records = []
            for case in run_model["cases"]:
                case_id = str(case["id"])
                manifest_path = model_dir / f"{case_id}.runner.json"
                payload = _read_json(
                    manifest_path, f"{model['id']} {case_id} runner manifest"
                )
                outputs = _validate_runner_manifest(payload, model, case_id)
                output_records = {}
                for name, row in outputs.items():
                    output_path = model_dir / str(row["file"])
                    _require_regular_file(
                        output_path, f"{model['id']} {case_id} {name} TRT output"
                    )
                    if output_path.stat().st_size != row["byte_count"]:
                        raise ValidationError(
                            f"{model['id']} {case_id} {name} TRT output size drifted"
                        )
                    output_records[name] = {
                        "path": _relative(output_path, run_dir),
                        "size_bytes": output_path.stat().st_size,
                        "sha256": _sha256(output_path),
                    }
                case_records.append(
                    {
                        "id": case_id,
                        "runner_manifest": _relative(manifest_path, run_dir),
                        "runner_manifest_sha256": _sha256(manifest_path),
                        "repeat_stable": True,
                        "inference_elapsed_ms": payload["inference_elapsed_ms"],
                        "outputs": output_records,
                    }
                )
            records.append(
                {
                    "id": model["id"],
                    "engine": {
                        "path": _relative(engine["path"], artifact_root),
                        "size_bytes": engine["size_bytes"],
                        "sha256": engine["sha256"],
                        "receipt": _relative(engine["receipt_path"], artifact_root),
                        "receipt_sha256": engine["receipt_sha256"],
                    },
                    "runner_log": _relative(log, run_dir),
                    "runner_log_sha256": _sha256(log),
                    "elapsed_seconds": time.monotonic() - started,
                    "cases": case_records,
                }
            )
            print(
                f"[TENSORRT] {model['id']} validated {len(case_records)} B3 cases",
                flush=True,
            )
    builder._assert_no_compute_owners("after RF-DETR media validation")
    receipt: dict[str, Any] = {
        "schema": TRT_SCHEMA,
        "run_id": run["run_id"],
        "recorded_at_utc": _utc_now(),
        "promotion_status": "unpromoted",
        "runtime_selected": False,
        "container_contract": {
            "network": "none",
            "root_filesystem": "read_only",
            "capabilities": "drop_all",
            "no_new_privileges": True,
            "caller_uid_gid": f"{os.getuid()}:{os.getgid()}",
            "gpu_device": builder.GPU_DEVICE_INDEX,
            "image_id": builder.REQUIRED_IMAGE_ID,
        },
        "platform": platform,
        "runner": {
            **runner,
            "path": _relative(runner["path"], run_dir),
        },
        "models": records,
    }
    if precision_profile is not None:
        receipt["artifact_role"] = "precision_canary"
        receipt["precision_canary_profile"] = profile_id
        receipt["prepared_validator_sha256"] = run["source_files"][
            "validator"
        ]["sha256"]
        receipt["execution_validator_sha256"] = _sha256(Path(__file__))
    if runtime_engine_profile:
        receipt["artifact_role"] = RUNTIME_VALIDATION_ARTIFACT_ROLE
        receipt["runtime_engine_profile"] = runtime_engine_profile
        receipt["input_contract"] = {
            "name": RUNTIME_INPUT_CONTRACT,
            "adapter_revision": RUNTIME_ADAPTER_REVISION,
            "tensor_range": [0.0, 1.0],
            "layout": "NCHW",
            "batch_size": 3,
        }
    receipt["manifest_sha256"] = _json_digest(receipt)
    destination = run_dir / receipt_filename
    _write_json_exclusive(destination, receipt)
    print(f"[OK] TensorRT media execution complete: {destination}")
    return 0


def _load_float_tensor(
    path: Path,
    shape: Sequence[int],
    *,
    expected_sha256: str,
    label: str,
) -> Any:
    import numpy as np

    _require_regular_file(path, label)
    expected_bytes = math.prod(shape) * 4
    if path.stat().st_size != expected_bytes:
        raise ValidationError(
            f"{label} byte count drifted: expected={expected_bytes} observed={path.stat().st_size}"
        )
    if SHA256_RE.fullmatch(expected_sha256 or "") is None or _sha256(path) != expected_sha256:
        raise ValidationError(f"{label} SHA-256 drifted")
    array = np.memmap(path, mode="r", dtype="<f4", shape=tuple(shape), order="C")
    if not np.isfinite(array).all():
        raise ValidationError(f"{label} contains non-finite values")
    return array


def _raw_error_stats(reference: Any, candidate: Any) -> dict[str, Any]:
    import numpy as np

    left = np.asarray(reference).reshape(-1)
    right = np.asarray(candidate).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValidationError("raw parity tensors must have equal nonempty shapes")
    count = int(left.size)
    maximum = 0.0
    absolute_sum = 0.0
    chunk_size = 1_000_000
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        difference = np.abs(
            left[start:stop].astype(np.float64)
            - right[start:stop].astype(np.float64)
        )
        maximum = max(maximum, float(difference.max()))
        absolute_sum += float(difference.sum(dtype=np.float64))
    stride = max(1, count // 1_000_000)
    sample = np.abs(
        left[::stride].astype(np.float64) - right[::stride].astype(np.float64)
    )
    return {
        "element_count": count,
        "mean_abs_error": absolute_sum / count,
        "max_abs_error": maximum,
        "sample_stride": stride,
        "sample_count": int(sample.size),
        "sample_p50_abs_error": float(np.percentile(sample, 50)),
        "sample_p95_abs_error": float(np.percentile(sample, 95)),
        "sample_p99_abs_error": float(np.percentile(sample, 99)),
    }


def _logsumexp(values: Any, axis: int) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.sum(np.exp(array - maximum), axis=axis, keepdims=True)
    )
    return np.squeeze(result, axis=axis)


def _keypoint_score_components(
    labels: Any, keypoints: Any
) -> tuple[Any, Any]:
    import numpy as np

    logits = np.asarray(labels)
    raw = np.asarray(keypoints)
    if (
        logits.ndim != 3
        or logits.shape[-1] != 2
        or raw.ndim != 4
        or raw.shape[:2] != logits.shape[:2]
        or raw.shape[2:] != (34, 8)
    ):
        raise ValidationError("keypoint preview raw tensor contract drifted")
    active = raw[:, :, 17:34, :]
    log_l11 = active[..., 4].astype(np.float64)
    l21 = active[..., 5].astype(np.float64)
    log_l22 = active[..., 6].astype(np.float64)
    findable = _sigmoid(active[..., 2]).astype(np.float64)
    log_t1 = -2.0 * log_l11
    log_t2 = -2.0 * log_l22
    log_t3 = (
        2.0 * np.log(np.maximum(np.abs(l21), 1e-12))
        + log_t1
        + log_t2
    )
    log_trace = _logsumexp(
        np.stack((log_t1, log_t2, log_t3), axis=-1), axis=-1
    )
    log_weights = np.log(np.maximum(findable, 1e-12))
    log_mean_trace = _logsumexp(log_trace + log_weights, axis=-1) - _logsumexp(
        log_weights, axis=-1
    )
    base_scores = _sigmoid(logits[..., 1]).astype(np.float64)
    fused_scores = base_scores * np.exp(
        -0.2 * log_mean_trace
    )
    if not np.isfinite(base_scores).all() or not np.isfinite(
        fused_scores
    ).all():
        raise ValidationError("keypoint preview score component is not finite")
    return base_scores, fused_scores


def _keypoint_fused_scores(labels: Any, keypoints: Any) -> Any:
    return _keypoint_score_components(labels, keypoints)[1]


def _mask_iou(left: Any, right: Any) -> float:
    import numpy as np

    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return 1.0 if union == 0 else intersection / union


def _keypoint_pair_metrics(
    ref_keypoints: Any,
    trt_keypoints: Any,
    ref_box: Any,
) -> tuple[float, list[float], float]:
    import numpy as np

    reference = np.asarray(ref_keypoints, dtype=np.float64)
    candidate = np.asarray(trt_keypoints, dtype=np.float64)
    if reference.shape != (17, 8) or candidate.shape != (17, 8):
        raise ValidationError("active keypoint pair must have shape [17,8]")
    reference_visible = _sigmoid(reference[:, 2]) >= 0.5
    candidate_visible = _sigmoid(candidate[:, 2]) >= 0.5
    union = int(np.logical_or(reference_visible, candidate_visible).sum())
    intersection = int(np.logical_and(reference_visible, candidate_visible).sum())
    jaccard = 1.0 if union == 0 else intersection / union
    if not reference_visible.any():
        raise ValidationError("strong keypoint reference has no findable keypoints")
    box = np.asarray(ref_box, dtype=np.float64)
    box_diagonal = max(float(np.hypot(box[2], box[3])), 1e-12)
    delta = candidate[reference_visible, :2] - reference[reference_visible, :2]
    distance = np.linalg.norm(delta, axis=1)
    normalized_errors = (distance / box_diagonal).tolist()
    area = max(float(box[2] * box[3]), 1e-12)
    sigmas = np.asarray(COCO_KEYPOINT_SIGMAS, dtype=np.float64)[reference_visible]
    oks = float(
        np.exp(
            -(distance**2)
            / np.maximum(2.0 * (sigmas**2) * area, 1e-12)
        ).mean()
    )
    return oks, normalized_errors, jaccard


def _compare_model(
    run_dir: Path,
    model: Mapping[str, Any],
    run_model: Mapping[str, Any],
    trt_model: Mapping[str, Any],
    corpus_cases: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[Any]]:
    import numpy as np
    from PIL import Image

    if model["id"] != run_model["id"] or model["id"] != trt_model["id"]:
        raise ValidationError("comparison model identities differ")
    expected_outputs = {
        str(name): [int(value) for value in shape]
        for name, shape in dict(model["outputs"]).items()
    }
    trt_cases = {str(row["id"]): row for row in trt_model["cases"]}
    class_agreement: list[float] = []
    retention: list[float] = []
    box_ious: list[float] = []
    score_errors: list[float] = []
    mask_ious: list[float] = []
    mask_area_errors: list[float] = []
    keypoint_oks: list[float] = []
    keypoint_coordinate_errors: list[float] = []
    keypoint_jaccards: list[float] = []
    person_box_ious: list[float] = []
    person_score_errors: list[float] = []
    person_mask_ious: list[float] = []
    person_mask_area_errors: list[float] = []
    person_keypoint_base_score_errors: list[float] = []
    person_keypoint_fused_score_abs_errors: list[float] = []
    person_keypoint_fused_score_relative_errors: list[float] = []
    person_keypoint_oks: list[float] = []
    person_keypoint_coordinate_errors: list[float] = []
    person_keypoint_jaccards: list[float] = []
    raw_parity: list[dict[str, Any]] = []
    semantic_cases: list[dict[str, Any]] = []
    assignment_batches: list[dict[str, Any]] = []
    person_assignment_batches: list[dict[str, Any]] = []
    sheet_rows: list[Any] = []
    strong_reference_count = 0
    person_reference_count = 0
    person_candidate_count = 0
    person_spatial_match_count = 0
    run_cases = {str(row["id"]): row for row in corpus_cases}

    for case, run_case in zip(
        run_model["cases"],
        run_model["cases"],
        strict=True,
    ):
        case_id = str(case["id"])
        case_metadata = run_cases.get(case_id)
        if not isinstance(case_metadata, Mapping):
            raise ValidationError(f"validation corpus lacks case {case_id}")
        trt_case = trt_cases.get(case_id)
        if not isinstance(trt_case, Mapping) or trt_case.get("repeat_stable") is not True:
            raise ValidationError(f"{model['id']} {case_id} TRT case is unavailable")
        reference_outputs: dict[str, Any] = {}
        candidate_outputs: dict[str, Any] = {}
        for name, shape in expected_outputs.items():
            ref_row = run_case["outputs"][name]
            trt_row = trt_case["outputs"][name]
            reference = _load_float_tensor(
                run_dir / str(ref_row["path"]),
                shape,
                expected_sha256=str(ref_row["sha256"]),
                label=f"{model['id']} {case_id} {name} reference",
            )
            candidate = _load_float_tensor(
                run_dir / str(trt_row["path"]),
                shape,
                expected_sha256=str(trt_row["sha256"]),
                label=f"{model['id']} {case_id} {name} TensorRT",
            )
            reference_outputs[name] = reference
            candidate_outputs[name] = candidate
            raw_parity.append(
                {
                    "case": case_id,
                    "tensor": name,
                    **_raw_error_stats(reference, candidate),
                }
            )

        ref_boxes = reference_outputs["dets"]
        trt_boxes = candidate_outputs["dets"]
        ref_labels = reference_outputs["labels"]
        trt_labels = candidate_outputs["labels"]
        if model["family"] == "keypoint":
            ref_base_scores, ref_scores = _keypoint_score_components(
                ref_labels, reference_outputs["keypoints"]
            )
            trt_base_scores, trt_scores = _keypoint_score_components(
                trt_labels, candidate_outputs["keypoints"]
            )
            ref_classes = np.ones(ref_scores.shape, dtype=np.int64)
            trt_classes = np.argmax(_sigmoid(trt_labels), axis=-1)
        else:
            ref_probabilities = _sigmoid(ref_labels)
            trt_probabilities = _sigmoid(trt_labels)
            ref_scores = ref_probabilities.max(axis=-1)
            trt_scores = trt_probabilities.max(axis=-1)
            ref_base_scores = ref_probabilities[..., 1]
            trt_base_scores = trt_probabilities[..., 1]
            ref_classes = ref_probabilities.argmax(axis=-1)
            trt_classes = trt_probabilities.argmax(axis=-1)

        reference_person_counts = []
        candidate_person_counts = []
        for batch_index in range(3):
            if model["family"] == "keypoint":
                person_scores = ref_scores[batch_index]
                candidate_person_scores = trt_scores[batch_index]
            else:
                person_scores = _sigmoid(ref_labels[batch_index, :, 1])
                candidate_person_scores = _sigmoid(
                    trt_labels[batch_index, :, 1]
                )
            reference_person_counts.append(
                int((person_scores >= PERSON_TASK_SCORE_MIN).sum())
            )
            candidate_person_counts.append(
                int(
                    (
                        candidate_person_scores
                        >= PERSON_TASK_SCORE_MIN
                    ).sum()
                )
            )

            person_assignment = _match_person_prediction_sets(
                ref_boxes[batch_index],
                trt_boxes[batch_index],
                person_scores,
                candidate_person_scores,
            )
            reference_queries = person_assignment["reference_queries"]
            candidate_queries = person_assignment["candidate_queries"]
            person_reference_count += len(reference_queries)
            person_candidate_count += len(candidate_queries)
            audited_person_matches = []
            for match in person_assignment["matches"]:
                audit_match = dict(match)
                if not match["spatial_match"]:
                    audited_person_matches.append(audit_match)
                    continue
                reference_query = int(match["reference_query"])
                candidate_query = int(match["candidate_query"])
                person_spatial_match_count += 1
                person_box_ious.append(float(match["iou"]))
                if model["family"] == "keypoint":
                    reference_base_score = float(
                        ref_base_scores[batch_index, reference_query]
                    )
                    candidate_base_score = float(
                        trt_base_scores[batch_index, candidate_query]
                    )
                    fused_abs_error = abs(
                        float(match["candidate_score"])
                        - float(match["reference_score"])
                    )
                    person_keypoint_base_score_errors.append(
                        abs(candidate_base_score - reference_base_score)
                    )
                    person_keypoint_fused_score_abs_errors.append(
                        fused_abs_error
                    )
                    person_keypoint_fused_score_relative_errors.append(
                        fused_abs_error
                        / max(abs(float(match["reference_score"])), 1e-12)
                    )
                    audit_match.update(
                        {
                            "reference_base_score": reference_base_score,
                            "candidate_base_score": candidate_base_score,
                            "fused_score_abs_error": fused_abs_error,
                            "fused_score_relative_error": (
                                fused_abs_error
                                / max(
                                    abs(float(match["reference_score"])),
                                    1e-12,
                                )
                            ),
                        }
                    )
                    ref_active = reference_outputs["keypoints"][
                        batch_index, reference_query, 17:34
                    ]
                    trt_active = candidate_outputs["keypoints"][
                        batch_index, candidate_query, 17:34
                    ]
                    oks, coordinate_errors, jaccard = (
                        _keypoint_pair_metrics(
                            ref_active,
                            trt_active,
                            ref_boxes[batch_index, reference_query],
                        )
                    )
                    person_keypoint_oks.append(oks)
                    person_keypoint_coordinate_errors.extend(
                        coordinate_errors
                    )
                    person_keypoint_jaccards.append(jaccard)
                else:
                    person_score_errors.append(
                        abs(
                            float(match["candidate_score"])
                            - float(match["reference_score"])
                        )
                    )

                if model["family"] == "segmentation":
                    image_row = case_metadata["images"][batch_index]
                    source_size = (
                        int(image_row["width"]),
                        int(image_row["height"]),
                    )
                    ref_resized = Image.fromarray(
                        np.asarray(
                            reference_outputs["masks"][
                                batch_index, reference_query
                            ],
                            dtype=np.float32,
                        ),
                        mode="F",
                    ).resize(source_size, Image.Resampling.BILINEAR)
                    trt_resized = Image.fromarray(
                        np.asarray(
                            candidate_outputs["masks"][
                                batch_index, candidate_query
                            ],
                            dtype=np.float32,
                        ),
                        mode="F",
                    ).resize(source_size, Image.Resampling.BILINEAR)
                    ref_mask = np.asarray(ref_resized) > 0
                    trt_mask = np.asarray(trt_resized) > 0
                    ref_area = int(ref_mask.sum())
                    if ref_area >= 64:
                        trt_area = int(trt_mask.sum())
                        person_mask_ious.append(
                            _mask_iou(ref_mask, trt_mask)
                        )
                        person_mask_area_errors.append(
                            abs(trt_area - ref_area) / ref_area
                        )
                audited_person_matches.append(audit_match)
            person_assignment_batches.append(
                {
                    "case": case_id,
                    "batch_index": batch_index,
                    "room": ROOMS[batch_index],
                    "reference_count": len(reference_queries),
                    "candidate_count": len(candidate_queries),
                    "spatial_match_count": sum(
                        int(row["spatial_match"])
                        for row in person_assignment["matches"]
                    ),
                    "matches": audited_person_matches,
                    "unmatched_reference_queries": person_assignment[
                        "unmatched_reference_queries"
                    ],
                    "unmatched_candidate_queries": person_assignment[
                        "unmatched_candidate_queries"
                    ],
                }
            )

            strong = np.flatnonzero(
                (
                    ref_scores[batch_index]
                    >= STRONG_REFERENCE_SCORE_MIN
                )
                & (ref_classes[batch_index] != 0)
            )
            if strong.size == 0:
                assignment_batches.append(
                    {
                        "case": case_id,
                        "batch_index": batch_index,
                        "room": ROOMS[batch_index],
                        "reference_count": 0,
                        "candidate_count": int(
                            trt_boxes[batch_index].shape[0]
                        ),
                        "weights": None,
                        "matches": [],
                    }
                )
                continue
            strong_classes = np.asarray(
                ref_classes[batch_index, strong], dtype=np.int64
            )
            if model["family"] == "keypoint":
                candidate_reference_scores = np.broadcast_to(
                    np.asarray(
                        trt_scores[batch_index], dtype=np.float64
                    )[None, :],
                    (
                        int(strong.size),
                        int(trt_scores[batch_index].shape[0]),
                    ),
                )
            else:
                candidate_reference_scores = np.asarray(
                    trt_probabilities[batch_index][
                        :, strong_classes
                    ].T,
                    dtype=np.float64,
                )
            assignments = _match_queries_semantically(
                ref_boxes[batch_index, strong],
                trt_boxes[batch_index],
                strong_classes,
                np.asarray(
                    trt_classes[batch_index], dtype=np.int64
                ),
                candidate_reference_scores,
            )
            audit_matches = []
            for assignment in assignments:
                strong_index = int(assignment["reference_index"])
                candidate_query = int(assignment["candidate_index"])
                reference_query = int(strong[strong_index])
                reference_class = int(assignment["reference_class"])
                strong_reference_count += 1
                class_agreement.append(
                    float(assignment["semantic_class_edge"])
                )
                if model["family"] == "keypoint":
                    reference_score = float(
                        ref_scores[batch_index, reference_query]
                    )
                else:
                    reference_score = float(
                        _sigmoid(
                            ref_labels[
                                batch_index,
                                reference_query,
                                reference_class,
                            ]
                        )
                    )
                candidate_score = float(
                    assignment["candidate_reference_class_score"]
                )
                retention.append(
                    float(assignment["semantic_retained_edge"])
                )
                score_errors.append(abs(candidate_score - reference_score))
                box_ious.append(float(assignment["iou"]))
                audit_matches.append(
                    {
                        **assignment,
                        "reference_query": reference_query,
                        "candidate_query": candidate_query,
                        "reference_score": reference_score,
                        "same_query": (
                            reference_query == candidate_query
                        ),
                    }
                )

                if model["family"] == "segmentation":
                    image_row = case_metadata["images"][batch_index]
                    source_size = (
                        int(image_row["width"]),
                        int(image_row["height"]),
                    )
                    ref_resized = Image.fromarray(
                        np.asarray(
                            reference_outputs["masks"][
                                batch_index, reference_query
                            ],
                            dtype=np.float32,
                        ),
                        mode="F",
                    ).resize(source_size, Image.Resampling.BILINEAR)
                    trt_resized = Image.fromarray(
                        np.asarray(
                            candidate_outputs["masks"][
                                batch_index, candidate_query
                            ],
                            dtype=np.float32,
                        ),
                        mode="F",
                    ).resize(source_size, Image.Resampling.BILINEAR)
                    ref_mask = np.asarray(ref_resized) > 0
                    trt_mask = np.asarray(trt_resized) > 0
                    ref_area = int(ref_mask.sum())
                    if ref_area >= 64:
                        trt_area = int(trt_mask.sum())
                        mask_ious.append(_mask_iou(ref_mask, trt_mask))
                        mask_area_errors.append(abs(trt_area - ref_area) / ref_area)
                elif model["family"] == "keypoint":
                    if reference_class != 1:
                        continue
                    ref_active = reference_outputs["keypoints"][
                        batch_index, reference_query, 17:34
                    ]
                    trt_active = candidate_outputs["keypoints"][
                        batch_index, candidate_query, 17:34
                    ]
                    oks, coordinate_errors, jaccard = _keypoint_pair_metrics(
                        ref_active,
                        trt_active,
                        ref_boxes[batch_index, reference_query],
                    )
                    keypoint_oks.append(oks)
                    keypoint_coordinate_errors.extend(coordinate_errors)
                    keypoint_jaccards.append(jaccard)
            assignment_batches.append(
                {
                    "case": case_id,
                    "batch_index": batch_index,
                    "room": ROOMS[batch_index],
                    "reference_count": int(strong.size),
                    "candidate_count": int(
                        trt_boxes[batch_index].shape[0]
                    ),
                    "weights": _semantic_assignment_weights(
                        int(strong.size)
                    ),
                    "matches": audit_matches,
                }
            )

        semantic_cases.append(
            {
                "id": case_id,
                "onnx_person_predictions_at_or_above_task_floor": dict(
                    zip(ROOMS, reference_person_counts, strict=True)
                ),
                "tensorrt_person_predictions_at_or_above_task_floor": dict(
                    zip(ROOMS, candidate_person_counts, strict=True)
                ),
                "occupancy_labels": {
                    room: OCCUPANCY[room][int(case["id"][1:])]
                    for room in ROOMS
                },
            }
        )
        images = []
        for batch_index, image_row in enumerate(
            case_metadata["images"]
        ):
            with Image.open(run_dir / str(image_row["path"])) as source_image:
                source = source_image.convert("RGB")
            outputs_for_image = {
                name: np.asarray(candidate_outputs[name][batch_index])
                for name in expected_outputs
            }
            images.append(
                _render_overlay(
                    source,
                    model,
                    outputs_for_image,
                    f"{case_id} | {image_row['room']} | occupied={str(image_row['occupied']).lower()}",
                )
            )
        row_height = max(image.height for image in images)
        row_width = sum(image.width for image in images)
        row = Image.new("RGB", (row_width, row_height), "black")
        offset = 0
        for image in images:
            row.paste(image, (offset, 0))
            offset += image.width
        sheet_rows.append(row)
        del reference_outputs, candidate_outputs

    if strong_reference_count == 0:
        raise ValidationError(f"{model['id']} produced no strong reference detections")
    if person_reference_count == 0 or not person_box_ious:
        raise ValidationError(
            f"{model['id']} produced no spatial person-task reference matches"
        )

    all_class_metrics: dict[str, Any] = {
        "strong_reference_count": strong_reference_count,
        "class_agreement": float(sum(class_agreement) / len(class_agreement)),
        "retention": float(sum(retention) / len(retention)),
        "box_iou": _summarize_distribution(box_ious),
        "score_abs_error": _summarize_distribution(score_errors),
    }
    if model["family"] == "segmentation":
        all_class_metrics["mask_iou"] = _summarize_distribution(mask_ious)
        all_class_metrics["mask_area_relative_error"] = _summarize_distribution(
            mask_area_errors
        )

    person_metrics: dict[str, Any] = {
        "score_floor": PERSON_TASK_SCORE_MIN,
        "reference_count": person_reference_count,
        "candidate_count": person_candidate_count,
        "spatial_match_count": person_spatial_match_count,
        "unmatched_reference_count": (
            person_reference_count - person_spatial_match_count
        ),
        "unmatched_candidate_count": (
            person_candidate_count - person_spatial_match_count
        ),
        "reference_retention": (
            person_spatial_match_count / person_reference_count
        ),
        "candidate_precision": (
            person_spatial_match_count / person_candidate_count
            if person_candidate_count
            else 1.0
        ),
        "box_iou": _summarize_distribution(person_box_ious),
    }
    if model["family"] == "keypoint":
        person_metrics.update(
            {
                "base_score_abs_error": _summarize_distribution(
                    person_keypoint_base_score_errors
                ),
                "fused_score_abs_error": _summarize_distribution(
                    person_keypoint_fused_score_abs_errors
                ),
                "fused_score_relative_error": _summarize_distribution(
                    person_keypoint_fused_score_relative_errors
                ),
                "oks": _summarize_distribution(person_keypoint_oks),
                "coordinate_error_box_diagonal": _summarize_distribution(
                    person_keypoint_coordinate_errors
                ),
                "visible_jaccard": _summarize_distribution(
                    person_keypoint_jaccards
                ),
            }
        )
    else:
        person_metrics["score_abs_error"] = _summarize_distribution(
            person_score_errors
        )
    if model["family"] == "segmentation":
        person_metrics["mask_iou"] = _summarize_distribution(
            person_mask_ious
        )
        person_metrics["mask_area_relative_error"] = _summarize_distribution(
            person_mask_area_errors
        )

    audited_matches = [
        match
        for batch in assignment_batches
        for match in batch["matches"]
    ]
    if len(audited_matches) != strong_reference_count:
        raise ValidationError(
            f"{model['id']} semantic assignment audit count drifted"
        )
    all_class_semantic_assignment = {
        "contract": SEMANTIC_ASSIGNMENT_CONTRACT,
        "iou_floor": SEMANTIC_MATCH_IOU_MIN,
        "retention_score_floor": STRONG_REFERENCE_SCORE_MIN,
        "objective_order": [
            "spatial_correct_class_and_retained",
            "spatial_correct_class",
            "spatial_match",
            "total_iou",
        ],
        "weight_proof": {
            "iou": "1",
            "spatial_match": "N*iou+1",
            "correct_class": "N*(spatial_match+iou)+1",
            "retained_correct": (
                "N*(correct_class+spatial_match+iou)+1"
            ),
            "dominance_scope": (
                "each higher-tier unit exceeds every possible lower-tier "
                "change across N batch references"
            ),
        },
        "aggregate": {
            "match_count": len(audited_matches),
            "spatial_match_count": sum(
                int(row["spatial_match"]) for row in audited_matches
            ),
            "class_match_count": sum(
                int(row["class_match"]) for row in audited_matches
            ),
            "score_retained_count": sum(
                int(row["score_retained"]) for row in audited_matches
            ),
            "semantic_class_edge_count": sum(
                int(row["semantic_class_edge"])
                for row in audited_matches
            ),
            "semantic_retained_edge_count": sum(
                int(row["semantic_retained_edge"])
                for row in audited_matches
            ),
            "same_query_count": sum(
                int(row["same_query"]) for row in audited_matches
            ),
        },
        "batches": assignment_batches,
    }
    person_semantic_assignment = {
        "contract": PERSON_ASSIGNMENT_CONTRACT,
        "score_source": (
            "official_uncertainty_fused_person_score"
            if model["family"] == "keypoint"
            else "sigmoid_logit_class_1"
        ),
        "score_floor": PERSON_TASK_SCORE_MIN,
        "iou_floor": SEMANTIC_MATCH_IOU_MIN,
        "objective_order": ["spatial_match", "total_iou"],
        "aggregate": {
            "reference_count": person_reference_count,
            "candidate_count": person_candidate_count,
            "spatial_match_count": person_spatial_match_count,
            "unmatched_reference_count": (
                person_reference_count - person_spatial_match_count
            ),
            "unmatched_candidate_count": (
                person_candidate_count - person_spatial_match_count
            ),
        },
        "batches": person_assignment_batches,
    }
    all_class_gates = (
        []
        if model["family"] == "keypoint"
        else _apply_quality_gates(
            str(model["family"]),
            all_class_metrics,
            scope="all_classes",
        )
    )
    person_gates = _apply_quality_gates(
        str(model["family"]),
        person_metrics,
        scope="person_task",
    )
    all_class_status = (
        "not_applicable"
        if model["family"] == "keypoint"
        else (
            "passed"
            if all(row["passed"] for row in all_class_gates)
            else "failed"
        )
    )
    person_status = (
        "passed" if all(row["passed"] for row in person_gates) else "failed"
    )
    required_statuses = [person_status]
    if all_class_status != "not_applicable":
        required_statuses.append(all_class_status)
    automated_status = (
        "passed"
        if all(status == "passed" for status in required_statuses)
        else "failed"
    )
    return (
        {
            "id": model["id"],
            "family": model["family"],
            "variant": model["variant"],
            "automated_status": automated_status,
            "scope_status": {
                "all_class_conversion": all_class_status,
                "person_task_conversion": person_status,
            },
            "metrics": {
                "all_classes": (
                    None
                    if model["family"] == "keypoint"
                    else all_class_metrics
                ),
                "person_task": person_metrics,
            },
            "gates": {
                "all_classes": all_class_gates,
                "person_task": person_gates,
            },
            "raw_parity": raw_parity,
            "semantic_assignment": {
                "all_classes": (
                    None
                    if model["family"] == "keypoint"
                    else all_class_semantic_assignment
                ),
                "person_task": person_semantic_assignment,
            },
            "semantic_cases": semantic_cases,
        },
        sheet_rows,
    )


def _render_overlay(
    image: Any,
    model: Mapping[str, Any],
    outputs: Mapping[str, Any],
    caption: str,
) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw

    canvas = image.copy().convert("RGB")
    boxes = np.asarray(outputs["dets"])
    logits = np.asarray(outputs["labels"])
    family = str(model["family"])
    if family == "keypoint":
        keypoints = np.asarray(outputs["keypoints"])
        scores = _keypoint_fused_scores(logits[None, ...], keypoints[None, ...])[0]
        classes = np.ones(scores.shape, dtype=np.int64)
    else:
        probabilities = _sigmoid(logits)
        scores = probabilities.max(axis=-1)
        classes = probabilities.argmax(axis=-1)
    selected = [
        int(index)
        for index in np.argsort(scores)[::-1]
        if scores[index] >= 0.5 and int(classes[index]) != 0
    ][:12]
    palette = (
        (255, 70, 70),
        (70, 220, 90),
        (70, 140, 255),
        (255, 210, 70),
        (220, 70, 255),
        (70, 230, 230),
    )
    width, height = canvas.size
    if family == "segmentation":
        masks = np.asarray(outputs["masks"])
        for order, query in enumerate(selected):
            raw_mask = Image.fromarray(masks[query].astype(np.float32), mode="F")
            resized = raw_mask.resize((width, height), Image.Resampling.BILINEAR)
            binary = np.asarray(resized) > 0
            alpha = Image.fromarray((binary.astype(np.uint8) * 72), mode="L")
            color = palette[order % len(palette)]
            layer = Image.new("RGB", canvas.size, color)
            canvas.paste(layer, (0, 0), alpha)
    draw = ImageDraw.Draw(canvas)
    for order, query in enumerate(selected):
        color = palette[order % len(palette)]
        cx, cy, box_width, box_height = [float(value) for value in boxes[query]]
        left = max(0.0, min(width, (cx - box_width / 2.0) * width))
        top = max(0.0, min(height, (cy - box_height / 2.0) * height))
        right = max(0.0, min(width, (cx + box_width / 2.0) * width))
        bottom = max(0.0, min(height, (cy + box_height / 2.0) * height))
        draw.rectangle((left, top, right, bottom), outline=color, width=max(2, width // 640))
        class_id = int(classes[query])
        class_name = "person" if class_id == 1 else ("dog" if class_id == 18 else f"c{class_id}")
        draw.text(
            (left + 2, max(0, top - 14)),
            f"{class_name} {float(scores[query]):.2f}",
            fill=color,
            stroke_width=2,
            stroke_fill="black",
        )
        if family == "keypoint" and class_id == 1:
            active = np.asarray(outputs["keypoints"])[query, 17:34]
            visible = _sigmoid(active[:, 2]) >= 0.5
            points = [
                (float(active[index, 0] * width), float(active[index, 1] * height))
                for index in range(17)
            ]
            for start, end in COCO_SKELETON:
                if visible[start] and visible[end]:
                    draw.line((points[start], points[end]), fill=color, width=max(2, width // 640))
            radius = max(2, width // 480)
            for index, point in enumerate(points):
                if visible[index]:
                    draw.ellipse(
                        (
                            point[0] - radius,
                            point[1] - radius,
                            point[0] + radius,
                            point[1] + radius,
                        ),
                        fill=color,
                        outline="black",
                    )
    target_width = 480
    target_height = max(1, round(height * target_width / width))
    canvas.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
    framed = Image.new("RGB", (target_width, canvas.height + 24), "black")
    framed.paste(canvas, ((target_width - canvas.width) // 2, 24))
    ImageDraw.Draw(framed).text((4, 5), caption, fill="white")
    return framed


def _save_contact_sheet(path: Path, rows: Sequence[Any]) -> dict[str, Any]:
    from PIL import Image

    if not rows:
        raise ValidationError("contact sheet requires at least one row")
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    sheet = Image.new("RGB", (width, height), "black")
    offset = 0
    for row in rows:
        sheet.paste(row, (0, offset))
        offset += row.height
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_raw)
    try:
        sheet.save(temporary, format="JPEG", quality=92, optimize=True)
        os.chmod(temporary, 0o600)
        if path.exists() or path.is_symlink():
            raise ValidationError(f"contact sheet already exists: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": path,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "width": sheet.width,
        "height": sheet.height,
    }


def _report_successor_lineage(
    run_dir: Path,
    run_id: str,
    successor_filename: str,
    supersedes_report: str,
) -> dict[str, Any] | None:
    raw = str(supersedes_report or "").strip()
    if not raw:
        return None
    predecessor_filename = _safe_filename(
        raw, "superseded report filename"
    )
    if (
        not predecessor_filename.startswith("report")
        or not predecessor_filename.endswith(".json")
    ):
        raise ValidationError(
            "superseded report must be a validation report JSON"
        )
    if predecessor_filename == successor_filename:
        raise ValidationError("successor report cannot supersede itself")
    predecessor_path = run_dir / predecessor_filename
    predecessor = _read_json(
        predecessor_path, "superseded media-quality report"
    )
    if (
        predecessor.get("schema") not in SUPPORTED_REPORT_SCHEMAS
        or predecessor.get("run_id") != run_id
        or predecessor.get("promotion_status") != "unpromoted"
        or predecessor.get("runtime_selected") is not False
    ):
        raise ValidationError("superseded report contract drifted")
    _validated_manifest_digest(predecessor, "superseded report")
    return {
        "path": predecessor_filename,
        "sha256": _sha256(predecessor_path),
        "manifest_sha256": predecessor["manifest_sha256"],
        "automated_status": predecessor.get("automated_status"),
        "recorded_at_utc": predecessor.get("recorded_at_utc"),
        "reason": "quality_scope_and_keypoint_score_correction",
    }


def _compare(args: argparse.Namespace) -> int:
    import gc

    run_dir = Path(args.run_dir).expanduser().resolve()
    revision = (
        _safe_token(args.revision, "analysis revision")
        if str(args.revision or "")
        else ""
    )
    precision_canary_profile = str(
        getattr(args, "precision_canary_profile", "") or ""
    )
    runtime_engine_profile = _runtime_engine_profile(
        getattr(args, "runtime_engine_profile", "")
    )
    if precision_canary_profile and runtime_engine_profile:
        raise ValidationError(
            "precision-canary and runtime-engine profiles are mutually exclusive"
        )
    if precision_canary_profile and not revision:
        raise ValidationError(
            "--precision-canary-profile requires an explicit --revision"
        )
    supersedes_report = str(args.supersedes_report or "").strip()
    if supersedes_report and not revision:
        raise ValidationError(
            "--supersedes-report requires an explicit --revision"
        )
    run = _load_run(
        run_dir,
        allow_validator_drift=bool(
            supersedes_report or precision_canary_profile
        ),
        allow_prepared_source_drift=bool(supersedes_report),
    )
    prepared_runtime_profile = _prepared_runtime_profile(run)
    if prepared_runtime_profile != runtime_engine_profile:
        raise ValidationError(
            "prepared input contract/runtime comparison profile mismatch: "
            f"prepared={prepared_runtime_profile or 'normalized'} "
            f"requested={runtime_engine_profile or 'normalized'}"
        )
    if revision:
        report_filename = f"report-{revision}.json"
    elif runtime_engine_profile:
        report_filename = (
            f"report-runtime-{runtime_engine_profile}.json"
        )
    else:
        report_filename = "report.json"
    lineage = _report_successor_lineage(
        run_dir,
        str(run["run_id"]),
        report_filename,
        supersedes_report,
    )
    report_path = run_dir / report_filename
    if report_path.exists() or report_path.is_symlink():
        raise ValidationError("comparison report already exists for this run")
    if runtime_engine_profile:
        trt_filename = f"trt-runtime-{runtime_engine_profile}.json"
    elif precision_canary_profile:
        trt_filename = (
            f"trt-canary-{precision_canary_profile}.json"
        )
    else:
        trt_filename = "trt.json"
    trt = _read_json(run_dir / trt_filename, "TensorRT phase receipt")
    if (
        trt.get("schema") != TRT_SCHEMA
        or trt.get("run_id") != run["run_id"]
        or trt.get("promotion_status") != "unpromoted"
        or trt.get("runtime_selected") is not False
    ):
        raise ValidationError("TensorRT phase receipt contract drifted")
    if runtime_engine_profile:
        if (
            trt.get("artifact_role")
            != RUNTIME_VALIDATION_ARTIFACT_ROLE
            or trt.get("runtime_engine_profile")
            != runtime_engine_profile
            or trt.get("precision_canary_profile") is not None
            or trt.get("input_contract")
            != {
                "name": RUNTIME_INPUT_CONTRACT,
                "adapter_revision": RUNTIME_ADAPTER_REVISION,
                "tensor_range": [0.0, 1.0],
                "layout": "NCHW",
                "batch_size": 3,
            }
        ):
            raise ValidationError(
                "TensorRT runtime-input receipt contract drifted"
            )
    elif precision_canary_profile:
        if (
            trt.get("artifact_role") != "precision_canary"
            or trt.get("precision_canary_profile")
            != precision_canary_profile
        ):
            raise ValidationError(
                "TensorRT precision-canary receipt contract drifted"
            )
    elif (
        trt.get("artifact_role") is not None
        or trt.get("precision_canary_profile") is not None
    ):
        raise ValidationError("baseline TensorRT receipt claims a canary role")
    _validated_manifest_digest(trt, "TensorRT phase")
    selected = _comparison_models_from_run(run)
    if [row["id"] for row in trt["models"]] != [row["id"] for row in run["models"]]:
        raise ValidationError("TensorRT/reference model order differs")
    overlay_root = _ensure_private_directory(
        run_dir
        / (
            f"overlays-{revision}"
            if revision
            else (
                f"overlays-runtime-{runtime_engine_profile}"
                if runtime_engine_profile
                else "overlays"
            )
        )
    )
    model_reports = []
    for model, run_model, trt_model in zip(
        selected, run["models"], trt["models"], strict=True
    ):
        started = time.monotonic()
        model_report, rows = _compare_model(
            run_dir,
            model,
            run_model,
            trt_model,
            run["cases"],
        )
        sheet_path = overlay_root / f"{model['id']}.jpg"
        sheet = _save_contact_sheet(sheet_path, rows)
        model_report["overlay_contact_sheet"] = {
            **sheet,
            "path": _relative(sheet["path"], run_dir),
        }
        model_report["comparison_seconds"] = time.monotonic() - started
        model_reports.append(model_report)
        all_class_count = (
            model_report["metrics"]["all_classes"] or {}
        ).get("strong_reference_count")
        reference_summary = (
            f"all={all_class_count}, "
            if all_class_count is not None
            else ""
        )
        reference_summary += (
            "person="
            f"{model_report['metrics']['person_task']['reference_count']}"
        )
        print(
            f"[COMPARE] {model['id']} {model_report['automated_status']} "
            f"({reference_summary})",
            flush=True,
        )
        del rows, sheet
        gc.collect()
    automated_status = (
        "passed"
        if all(row["automated_status"] == "passed" for row in model_reports)
        else "failed"
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "run_id": run["run_id"],
        "recorded_at_utc": _utc_now(),
        "promotion_status": "unpromoted",
        "runtime_selected": False,
        "automated_status": automated_status,
        "visual_review_status": "pending",
        "analysis": {
            "revision": revision or "initial",
            "matching": SEMANTIC_ASSIGNMENT_CONTRACT,
            "matching_iou_floor": SEMANTIC_MATCH_IOU_MIN,
            "matching_retention_score_floor": (
                STRONG_REFERENCE_SCORE_MIN
            ),
            "person_task_matching": PERSON_ASSIGNMENT_CONTRACT,
            "person_task_score_floor": PERSON_TASK_SCORE_MIN,
            "status_policy": (
                "all applicable conversion scopes must pass; all-class "
                "failures are never overridden by person-task success"
            ),
            "matching_objective_order": [
                "spatial_correct_class_and_retained",
                "spatial_correct_class",
                "spatial_match",
                "total_iou",
            ],
            "raw_query_aligned_error_role": "diagnostic_only",
            "validator_sha256": _sha256(Path(__file__)),
            "prepared_validator_sha256": run["source_files"]["validator"][
                "sha256"
            ],
            "prepared_matrix_sha256": run["source_files"]["matrix"][
                "sha256"
            ],
            "analysis_matrix_sha256": _sha256(MATRIX_PATH),
            "prepared_source_reuse": bool(
                supersedes_report or precision_canary_profile
            ),
            "supersedes_report": lineage,
            "tensor_rt_evidence": {
                "receipt": trt_filename,
                "receipt_sha256": _sha256(run_dir / trt_filename),
                "precision_canary_profile": (
                    precision_canary_profile or None
                ),
                "runtime_engine_profile": (
                    runtime_engine_profile or None
                ),
                "input_contract": (
                    RUNTIME_INPUT_CONTRACT
                    if runtime_engine_profile
                    else "imagenet_normalized"
                ),
            },
        },
        "gates": {
            "all_classes": {
                "detection": DETECTION_GATES,
                "segmentation": SEGMENTATION_GATES,
                "strong_reference_score_min": (
                    STRONG_REFERENCE_SCORE_MIN
                ),
            },
            "person_task": {
                "detection": PERSON_TASK_GATES,
                "segmentation": SEGMENTATION_GATES,
                "keypoint": KEYPOINT_GATES,
                "score_min": PERSON_TASK_SCORE_MIN,
                "class_index": 1,
            },
            "tiny_mask_exclusion_pixels": 64,
        },
        "corpus": {
            "batch_count": len(FRAME_INDICES),
            "batch_size": 3,
            "frame_count": len(FRAME_INDICES) * 3,
            "frame_indices": list(FRAME_INDICES),
            "occupied_frames": 20,
            "negative_frames": 4,
        },
        "models": model_reports,
        "no_promotion_or_runtime_change": True,
    }
    report["manifest_sha256"] = _json_digest(report)
    _write_json_exclusive(report_path, report)
    print(f"[{automated_status.upper()}] automated media-quality report: {report_path}")
    return 0 if automated_status == "passed" else 3


def _review(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    run = _load_run(
        run_dir,
        allow_validator_drift=True,
        allow_prepared_source_drift=True,
    )
    report_filename = _safe_filename(args.report, "report filename")
    if not report_filename.startswith("report") or not report_filename.endswith(
        ".json"
    ):
        raise ValidationError("visual review requires a validation report JSON")
    report_path = run_dir / report_filename
    report = _read_json(report_path, "media-quality report")
    if (
        report.get("schema") not in SUPPORTED_REPORT_SCHEMAS
        or report.get("run_id") != run["run_id"]
        or report.get("visual_review_status") != "pending"
    ):
        raise ValidationError("media-quality report is not awaiting visual review")
    _validated_manifest_digest(report, "media-quality report")
    decision = str(args.decision)
    if decision not in {"passed", "failed"}:
        raise ValidationError("visual review decision must be passed or failed")
    if decision == "passed" and report.get("automated_status") != "passed":
        raise ValidationError("cannot pass visual review over failed automated gates")
    reviewer = _safe_token(args.reviewer, "reviewer")
    sheets = []
    for model in report.get("models", []):
        row = model.get("overlay_contact_sheet")
        if not isinstance(row, Mapping):
            raise ValidationError("report lacks an overlay contact sheet")
        path = run_dir / str(row.get("path"))
        _require_regular_file(path, "overlay contact sheet")
        if (
            path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("sha256")
        ):
            raise ValidationError(f"overlay contact sheet changed: {path}")
        sheets.append(
            {
                "model_id": model["id"],
                "path": row["path"],
                "sha256": row["sha256"],
            }
        )
    notes = [str(value).strip() for value in args.note if str(value).strip()]
    if not notes:
        raise ValidationError("at least one explicit visual-review --note is required")
    payload: dict[str, Any] = {
        "schema": REVIEW_SCHEMA,
        "run_id": run["run_id"],
        "report": report_filename,
        "report_sha256": _sha256(report_path),
        "decision": decision,
        "reviewer": reviewer,
        "reviewed_at_utc": _utc_now(),
        "review_scope": {
            "models": len(sheets),
            "frames_per_model": len(FRAME_INDICES) * 3,
            "checks": [
                "box_alignment",
                "mask_silhouette_alignment",
                "keypoint_anatomy_and_left_right_topology",
                "duplicate_or_false_positive_stressors",
                "occupied_and_negative_scene_coverage",
            ],
        },
        "contact_sheets": sheets,
        "notes": notes,
        "overall_status": (
            "passed"
            if decision == "passed" and report["automated_status"] == "passed"
            else "failed"
        ),
        "no_promotion_or_runtime_change": True,
    }
    payload["manifest_sha256"] = _json_digest(payload)
    report_stem = _safe_token(Path(report_filename).stem, "report stem")
    destination = run_dir / f"visual-review-{report_stem}.json"
    _write_json_exclusive(destination, payload)
    print(f"[{payload['overall_status'].upper()}] visual review: {destination}")
    return 0 if payload["overall_status"] == "passed" else 3


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Extract the reviewed media corpus and run CPU ONNX references.",
    )
    prepare.add_argument(
        "--model",
        action="append",
        default=[],
        help="Explicit model ID; repeat or pass a comma-separated list.",
    )
    prepare.add_argument(
        "--media-dir",
        required=True,
        help="Absolute directory containing the three pinned 68-second clips.",
    )
    prepare.add_argument(
        "--run-id",
        default="",
        help="Optional safe evidence run ID; an UTC ID is generated by default.",
    )
    prepare.add_argument(
        "--runtime-engine-profile",
        choices=RUNTIME_ENGINE_PROFILES,
        default="",
        help=(
            "Prepare raw RGB float32 [0,1] B3 tensors against the "
            "revision-bound runtime ONNX for this final engine profile. "
            "Evidence is isolated below runtime/<profile>/<run-id>."
        ),
    )

    trt = subparsers.add_parser(
        "trt",
        help="Run the prepared cases through the pinned isolated TensorRT engines.",
    )
    trt.add_argument("--run-dir", required=True)
    trt_mode = trt.add_mutually_exclusive_group()
    trt_mode.add_argument(
        "--precision-canary-profile",
        choices=PRECISION_CANARY_PROFILES,
        default="",
        help=(
            "Use an isolated precision-canary engine set and write "
            "profile-scoped TensorRT evidence."
        ),
    )
    trt_mode.add_argument(
        "--runtime-engine-profile",
        choices=RUNTIME_ENGINE_PROFILES,
        default="",
        help=(
            "Use the matching runtime-input engine/receipt and raw RGB01 "
            "prepared inputs. Evidence is isolated by runtime profile."
        ),
    )

    compare = subparsers.add_parser(
        "compare",
        help="Compare TensorRT outputs with the CPU ONNX references and render overlays.",
    )
    compare.add_argument("--run-dir", required=True)
    compare_mode = compare.add_mutually_exclusive_group()
    compare_mode.add_argument(
        "--precision-canary-profile",
        choices=PRECISION_CANARY_PROFILES,
        default="",
        help=(
            "Compare the profile-scoped precision-canary TensorRT evidence. "
            "Requires --revision."
        ),
    )
    compare_mode.add_argument(
        "--runtime-engine-profile",
        choices=RUNTIME_ENGINE_PROFILES,
        default="",
        help=(
            "Compare matching runtime-input TensorRT evidence against its "
            "runtime ONNX RGB01 reference."
        ),
    )
    compare.add_argument(
        "--revision",
        default="",
        help=(
            "Optional safe successor-analysis ID. This preserves every "
            "existing report; use --supersedes-report to record predecessor "
            "digest lineage."
        ),
    )
    compare.add_argument(
        "--supersedes-report",
        default="",
        help=(
            "Optional existing report filename in the same run. Requires "
            "--revision and records exact predecessor digest lineage."
        ),
    )

    review = subparsers.add_parser(
        "review",
        help="Seal an explicit inspection of all model contact sheets.",
    )
    review.add_argument("--run-dir", required=True)
    review.add_argument("--report", default="report.json")
    review.add_argument("--decision", required=True, choices=("passed", "failed"))
    review.add_argument("--reviewer", required=True)
    review.add_argument("--note", action="append", default=[])
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    try:
        if args.phase == "prepare":
            return _prepare(args)
        if args.phase == "trt":
            return _trt(args)
        if args.phase == "compare":
            return _compare(args)
        if args.phase == "review":
            return _review(args)
        raise ValidationError(f"unsupported validation phase: {args.phase}")
    except ValidationError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
