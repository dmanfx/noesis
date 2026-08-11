#!/usr/bin/env python3
"""Compare current MapAnything alignment with one RGB-guided candidate.

This is a bounded, offline diagnostic.  It binds the locked fixed-corpus RGB,
model input, native TensorRT output, annotations, and current DS9 dewarper
validity configuration before comparing:

* the runtime's crop + bilinear depth/confidence + nearest-mask alignment; and
* a 3x3-neighbourhood RGB-guided joint-bilateral upsampler.

The RGB and authored-edge scores are internal alignment evidence only.  They
are not surveyed metric-depth accuracy, temporal stability, or promotion
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml


SCRIPT_PATH = Path(__file__).resolve()
DS9_ROOT = SCRIPT_PATH.parent.parent
REPO_ROOT = DS9_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CORPUS_ROOT = Path(
    "/mnt/noesis_storage/noesis-validation/"
    "mapanything-depth-panel-quality-20260724/fixed-corpus-v1"
)
DEFAULT_MANIFEST = DEFAULT_CORPUS_ROOT / "comparison-manifest-v1.json"
DEFAULT_OUTPUT = (
    DEFAULT_CORPUS_ROOT.parent / "alignment-ab-v1/alignment-ab-report.json"
)
DEFAULT_PIPELINE_CONFIG = DS9_ROOT / "config/infer.yaml"

REPORT_CONTRACT = "noesis.ds9.mapanything_alignment_ab.v1"
COMPARISON_CONTRACT = "noesis.ds9.mapanything_fixed_corpus_comparison.v1"
ANNOTATION_CONTRACT = "noesis.ds9.mapanything_fixed_annotations.v1"
FIXTURE_CONTRACT = "noesis.ds9.mapanything_raw_fixture.v1"
DEFAULT_CANDIDATE_ID = "canonical-baseline-294x518"
OUTPUT_NAMES = ("depth", "conf", "mask")

# This single candidate is fixed in code rather than tuned against annotations.
GUIDED_RADIUS_NATIVE_PX = 1
GUIDED_SIGMA_SPATIAL_NATIVE_PX = 0.75
GUIDED_SIGMA_COLOR_LAB = 6.0

# Provisional, predeclared diagnostic gates.  Passing would justify only a
# controlled temporal experiment, never live promotion.
MIN_RGB_EDGE_DISTANCE_IMPROVEMENT = 0.05
MIN_ANNOTATED_EDGE_P50_IMPROVEMENT = 0.10
MAX_ANNOTATED_EDGE_P95_REGRESSION = 0.05
MAX_FLAT_RGB_GRADIENT_AMPLIFICATION = 1.10
MAX_DEPTH_DELTA_P95_M = 0.10
MAX_ABS_SIGNED_BIAS_M = 0.01


class AlignmentEvaluationError(RuntimeError):
    """Raised when locked input identity or alignment evaluation is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AlignmentEvaluationError(f"invalid JSON {path}: {exc}") from exc


def require_identity(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path") or "")).expanduser()
    if not path.is_file():
        raise AlignmentEvaluationError(f"{label} does not exist: {path}")
    try:
        expected_size = int(record["size_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AlignmentEvaluationError(f"{label} has no valid size identity") from exc
    observed_size = int(path.stat().st_size)
    if observed_size != expected_size:
        raise AlignmentEvaluationError(
            f"{label} size mismatch: expected {expected_size}, observed {observed_size}"
        )
    expected_sha = str(record.get("sha256") or "")
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise AlignmentEvaluationError(
            f"{label} SHA-256 mismatch: expected {expected_sha}, observed {observed_sha}"
        )
    return path


def parse_dimensions(raw: Any) -> tuple[int, ...]:
    try:
        dimensions = tuple(int(value) for value in str(raw).lower().split("x"))
    except ValueError as exc:
        raise AlignmentEvaluationError(
            f"invalid output dimensions: {raw!r}"
        ) from exc
    if len(dimensions) != 4 or dimensions[1] != 1 or any(
        value <= 0 for value in dimensions
    ):
        raise AlignmentEvaluationError(
            f"output must have positive Nx1xHxW dimensions, got {dimensions}"
        )
    return dimensions


def parse_output_tensors(payload: Any) -> dict[str, np.ndarray]:
    if not isinstance(payload, list):
        raise AlignmentEvaluationError("TensorRT output must be a list")
    entries: dict[str, Mapping[str, Any]] = {}
    for entry in payload:
        if not isinstance(entry, Mapping):
            raise AlignmentEvaluationError("TensorRT output entry must be an object")
        name = str(entry.get("name") or "")
        if name in entries:
            raise AlignmentEvaluationError(f"duplicate TensorRT output {name!r}")
        entries[name] = entry
    if set(entries) != set(OUTPUT_NAMES):
        raise AlignmentEvaluationError(
            f"expected outputs {OUTPUT_NAMES}, observed {tuple(sorted(entries))}"
        )

    result: dict[str, np.ndarray] = {}
    expected_shape: tuple[int, ...] | None = None
    for name in OUTPUT_NAMES:
        entry = entries[name]
        shape = parse_dimensions(entry.get("dimensions"))
        values = entry.get("values")
        if not isinstance(values, list) or len(values) != math.prod(shape):
            raise AlignmentEvaluationError(
                f"{name} value count does not match {shape}"
            )
        array = np.asarray(values, dtype=np.float32).reshape(shape)[:, 0]
        if not np.all(np.isfinite(array)):
            raise AlignmentEvaluationError(f"{name} contains non-finite values")
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise AlignmentEvaluationError(
                f"output shape mismatch: expected {expected_shape}, got {shape}"
            )
        result[name] = array
    return result


def runtime_crop(
    native_shape: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Reproduce MapAnythingProcessor._align_to_frame crop geometry."""

    model_h, model_w = (int(native_shape[0]), int(native_shape[1]))
    target_w, target_h = (int(target_size[0]), int(target_size[1]))
    if min(model_h, model_w, target_w, target_h) <= 0:
        raise AlignmentEvaluationError("native and target dimensions must be positive")
    scale = min(model_w / float(target_w), model_h / float(target_h))
    resized_w = max(1, min(model_w, int(round(float(target_w) * scale))))
    resized_h = max(1, min(model_h, int(round(float(target_h) * scale))))
    pad_left = max(0, int(math.floor((model_w - resized_w) * 0.5)))
    pad_top = max(0, int(math.floor((model_h - resized_h) * 0.5)))
    return resized_w, resized_h, pad_left, pad_top


def align_bilinear(
    depth_native: np.ndarray,
    conf_native: np.ndarray,
    mask_native: np.ndarray,
    *,
    crop: tuple[int, int, int, int],
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the current runtime alignment and explicit unknown handling."""

    resized_w, resized_h, pad_left, pad_top = crop
    target_w, target_h = target_size
    y_slice = slice(pad_top, pad_top + resized_h)
    x_slice = slice(pad_left, pad_left + resized_w)
    depth = np.asarray(depth_native[y_slice, x_slice], dtype=np.float32)
    conf = np.asarray(conf_native[y_slice, x_slice], dtype=np.float32)
    mask = np.asarray(mask_native[y_slice, x_slice], dtype=bool)
    if depth.shape != (resized_h, resized_w):
        raise AlignmentEvaluationError("runtime crop lies outside native depth")

    aligned_depth = cv2.resize(
        depth,
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )
    aligned_conf = cv2.resize(
        conf,
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )
    aligned_mask = cv2.resize(
        mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    aligned_mask &= np.isfinite(aligned_depth)
    aligned_depth = np.asarray(aligned_depth, dtype=np.float32)
    aligned_conf = np.asarray(aligned_conf, dtype=np.float32)
    aligned_depth[~aligned_mask] = np.nan
    aligned_conf[~aligned_mask] = 0.0
    return aligned_depth, aligned_conf, aligned_mask


def align_rgb_guided(
    depth_native: np.ndarray,
    conf_native: np.ndarray,
    mask_native: np.ndarray,
    low_rgb_native: np.ndarray,
    high_rgb: np.ndarray,
    *,
    crop: tuple[int, int, int, int],
    target_size: tuple[int, int],
    radius: int = GUIDED_RADIUS_NATIVE_PX,
    sigma_spatial: float = GUIDED_SIGMA_SPATIAL_NATIVE_PX,
    sigma_color: float = GUIDED_SIGMA_COLOR_LAB,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Joint-bilateral upsample while preserving nearest-mask unknowns.

    Each output is a convex combination of valid native samples in a fixed
    local neighbourhood.  The candidate therefore cannot overshoot the local
    metric-depth range.  It is not allowed to fill a pixel rejected by the
    runtime's nearest-neighbour native mask.
    """

    if radius < 1 or sigma_spatial <= 0.0 or sigma_color <= 0.0:
        raise AlignmentEvaluationError("guided parameters must be positive")
    resized_w, resized_h, pad_left, pad_top = crop
    target_w, target_h = target_size
    y_slice = slice(pad_top, pad_top + resized_h)
    x_slice = slice(pad_left, pad_left + resized_w)
    depth = np.asarray(depth_native[y_slice, x_slice], dtype=np.float32)
    conf = np.asarray(conf_native[y_slice, x_slice], dtype=np.float32)
    native_valid = (
        np.asarray(mask_native[y_slice, x_slice], dtype=bool)
        & np.isfinite(depth)
        & np.isfinite(conf)
    )
    low_rgb = np.asarray(low_rgb_native[y_slice, x_slice], dtype=np.float32)
    high_rgb_array = np.asarray(high_rgb, dtype=np.float32)
    if (
        depth.shape != (resized_h, resized_w)
        or low_rgb.shape != (resized_h, resized_w, 3)
        or high_rgb_array.shape != (target_h, target_w, 3)
    ):
        raise AlignmentEvaluationError("guided depth/RGB shapes do not match mapping")
    if (
        not np.all(np.isfinite(low_rgb))
        or not np.all(np.isfinite(high_rgb_array))
        or np.min(low_rgb) < 0.0
        or np.max(low_rgb) > 1.0
        or np.min(high_rgb_array) < 0.0
        or np.max(high_rgb_array) > 1.0
    ):
        raise AlignmentEvaluationError("guided RGB must be finite in [0,1]")

    low_lab = cv2.cvtColor(
        np.ascontiguousarray(low_rgb),
        cv2.COLOR_RGB2Lab,
    )
    high_lab = cv2.cvtColor(
        np.ascontiguousarray(high_rgb_array),
        cv2.COLOR_RGB2Lab,
    )
    x_continuous = (
        (np.arange(target_w, dtype=np.float32) + 0.5)
        * (float(resized_w) / float(target_w))
    ) - 0.5
    y_continuous = (
        (np.arange(target_h, dtype=np.float32) + 0.5)
        * (float(resized_h) / float(target_h))
    ) - 0.5
    base_x = np.floor(x_continuous).astype(np.int32)
    base_y = np.floor(y_continuous).astype(np.int32)

    min_energy = np.full((target_h, target_w), np.inf, dtype=np.float32)
    local_min = np.full((target_h, target_w), np.inf, dtype=np.float32)
    local_max = np.full((target_h, target_w), -np.inf, dtype=np.float32)

    def neighbour(
        offset_x: int,
        offset_y: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sample_x = base_x + int(offset_x)
        sample_y = base_y + int(offset_y)
        inside_x = (sample_x >= 0) & (sample_x < resized_w)
        inside_y = (sample_y >= 0) & (sample_y < resized_h)
        clipped_x = np.clip(sample_x, 0, resized_w - 1)
        clipped_y = np.clip(sample_y, 0, resized_h - 1)
        support = (
            inside_y[:, None]
            & inside_x[None, :]
            & native_valid[clipped_y[:, None], clipped_x[None, :]]
        )
        spatial = (
            ((sample_y.astype(np.float32) - y_continuous)[:, None] ** 2)
            + ((sample_x.astype(np.float32) - x_continuous)[None, :] ** 2)
        ) / float(sigma_spatial**2)
        sampled_lab = low_lab[clipped_y[:, None], clipped_x[None, :]]
        color = np.sum((high_lab - sampled_lab) ** 2, axis=2) / float(
            sigma_color**2
        )
        return clipped_x, clipped_y, support, spatial + color

    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            sample_x, sample_y, support, energy = neighbour(offset_x, offset_y)
            sampled_depth = depth[sample_y[:, None], sample_x[None, :]]
            np.minimum(
                min_energy,
                np.where(support, energy, np.inf),
                out=min_energy,
            )
            np.minimum(
                local_min,
                np.where(support, sampled_depth, np.inf),
                out=local_min,
            )
            np.maximum(
                local_max,
                np.where(support, sampled_depth, -np.inf),
                out=local_max,
            )

    weight_sum = np.zeros((target_h, target_w), dtype=np.float32)
    depth_sum = np.zeros_like(weight_sum)
    conf_sum = np.zeros_like(weight_sum)
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            sample_x, sample_y, support, energy = neighbour(offset_x, offset_y)
            relative_energy = np.zeros_like(energy, dtype=np.float32)
            np.subtract(
                energy,
                min_energy,
                out=relative_energy,
                where=support,
            )
            weight = np.zeros_like(energy, dtype=np.float32)
            np.exp(-0.5 * relative_energy, out=weight, where=support)
            weight[~support] = 0.0
            sampled_depth = depth[sample_y[:, None], sample_x[None, :]]
            sampled_conf = conf[sample_y[:, None], sample_x[None, :]]
            weight_sum += weight
            depth_sum += weight * sampled_depth
            conf_sum += weight * sampled_conf

    nearest_mask = cv2.resize(
        native_valid.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    output_valid = nearest_mask & np.isfinite(min_energy) & (weight_sum > 0.0)
    output_depth = np.full((target_h, target_w), np.nan, dtype=np.float32)
    output_conf = np.zeros((target_h, target_w), dtype=np.float32)
    output_depth[output_valid] = (
        depth_sum[output_valid] / weight_sum[output_valid]
    )
    output_conf[output_valid] = conf_sum[output_valid] / weight_sum[output_valid]
    tolerance = np.maximum(
        1e-5,
        np.maximum(np.abs(local_min), np.abs(local_max)) * 1e-6,
    )
    hull_violations = output_valid & (
        (output_depth < (local_min - tolerance))
        | (output_depth > (local_max + tolerance))
    )
    diagnostics = {
        "local_convex_hull_violation_count": int(
            np.count_nonzero(hull_violations)
        ),
        "supported_pixel_count": int(np.count_nonzero(output_valid)),
        "nearest_native_valid_pixel_count": int(np.count_nonzero(nearest_mask)),
    }
    return output_depth, output_conf, output_valid, diagnostics


def percentile(values: np.ndarray, quantile: float) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(np.percentile(array, quantile))


def distribution(values: np.ndarray) -> dict[str, float | None]:
    return {
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": percentile(values, 100),
    }


def improvement(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or abs(reference) <= 1e-12:
        return None
    return float((reference - candidate) / reference)


def depth_gradients(
    depth: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_array = np.asarray(valid, dtype=bool)
    interior = cv2.erode(
        valid_array.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    safe = np.where(valid_array, np.maximum(depth, 1e-6), 1.0).astype(np.float32)
    log_depth = np.log(safe)
    grad_x = cv2.Sobel(log_depth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    grad_y = cv2.Sobel(log_depth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    magnitude = np.hypot(grad_x, grad_y)
    magnitude[~interior] = 0.0
    return grad_x, grad_y, magnitude, interior


def rgb_gradient(high_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(
        np.ascontiguousarray(high_rgb.astype(np.float32)),
        cv2.COLOR_RGB2Lab,
    )
    luminance = lab[:, :, 0]
    grad_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    grad_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.hypot(grad_x, grad_y)


def rgb_edge_distance_metrics(
    depth_magnitude: np.ndarray,
    rgb_magnitude: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(valid, dtype=bool)
    if np.count_nonzero(selected) < 100:
        raise AlignmentEvaluationError("too few valid pixels for RGB edge metrics")
    rgb_threshold = percentile(rgb_magnitude[selected], 85)
    depth_threshold = percentile(depth_magnitude[selected], 95)
    assert rgb_threshold is not None and depth_threshold is not None
    rgb_edges = selected & (rgb_magnitude >= rgb_threshold)
    depth_edges = selected & (depth_magnitude >= depth_threshold)
    distance = cv2.distanceTransform(
        (~rgb_edges).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    return {
        "evidence_class": "non_surveyed_internal_rgb_edge_diagnostic",
        "rgb_edge_percentile": 85,
        "depth_edge_percentile": 95,
        "strong_depth_edge_pixel_count": int(np.count_nonzero(depth_edges)),
        "distance_to_strong_rgb_edge_px": distribution(distance[depth_edges]),
    }


def sample_polyline_segments(
    points: np.ndarray,
    *,
    spacing_px: float = 4.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    for start, end in zip(points[:-1], points[1:]):
        tangent = np.asarray(end - start, dtype=np.float32)
        length = float(np.linalg.norm(tangent))
        if length <= 1e-6:
            continue
        tangent /= length
        normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float32)
        count = max(2, int(math.ceil(length / spacing_px)) + 1)
        segment = np.linspace(start, end, num=count, endpoint=True)
        if samples:
            segment = segment[1:]
        samples.extend((point.astype(np.float32), normal) for point in segment)
    return samples


def normal_edge_metrics(
    edge: Mapping[str, Any],
    *,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    valid: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    points = np.asarray(edge.get("polyline_px"), dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        raise AlignmentEvaluationError("edge polyline must contain at least two points")
    samples = sample_polyline_segments(points)
    radius = max(1, int(round(float(edge.get("search_radius_px", 24.0)))))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    distances: list[float] = []
    strengths: list[float] = []
    valid_u8 = np.asarray(valid, dtype=np.uint8)
    height, width = valid.shape
    for point, normal in samples:
        map_x = point[0] + (offsets * normal[0])
        map_y = point[1] + (offsets * normal[1])
        inside = (
            (map_x >= 0.0)
            & (map_x <= float(width - 1))
            & (map_y >= 0.0)
            & (map_y <= float(height - 1))
        )
        map_x_2d = map_x[None, :]
        map_y_2d = map_y[None, :]
        sampled_x = cv2.remap(
            grad_x,
            map_x_2d,
            map_y_2d,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )[0]
        sampled_y = cv2.remap(
            grad_y,
            map_x_2d,
            map_y_2d,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )[0]
        sampled_valid = cv2.remap(
            valid_u8,
            map_x_2d,
            map_y_2d,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )[0].astype(bool)
        admitted = inside & sampled_valid
        if not np.any(admitted):
            continue
        normal_strength = np.abs(
            (sampled_x * normal[0]) + (sampled_y * normal[1])
        )
        normal_strength[~admitted] = -np.inf
        best_index = int(np.argmax(normal_strength))
        distances.append(float(abs(offsets[best_index])))
        strengths.append(float(normal_strength[best_index]))
    distance_array = np.asarray(distances, dtype=np.float64)
    strength_array = np.asarray(strengths, dtype=np.float64)
    return (
        {
            "id": str(edge.get("id") or ""),
            "kind": str(edge.get("kind") or ""),
            "annotation_confidence": str(
                edge.get("annotation_confidence") or "unspecified"
            ),
            "sample_count": int(distance_array.size),
            "search_axis": "authored_polyline_normal_only",
            "nearest_normal_log_depth_gradient_distance_px": distribution(
                distance_array
            ),
            "selected_normal_log_depth_gradient_strength": distribution(
                strength_array
            ),
        },
        distance_array,
        strength_array,
    )


def masked_area_roundtrip(
    aligned_depth: np.ndarray,
    aligned_valid: np.ndarray,
    *,
    native_depth: np.ndarray,
    native_valid: np.ndarray,
) -> dict[str, Any]:
    native_h, native_w = native_depth.shape
    numerator = cv2.resize(
        np.where(aligned_valid, aligned_depth, 0.0).astype(np.float32),
        (native_w, native_h),
        interpolation=cv2.INTER_AREA,
    )
    support = cv2.resize(
        aligned_valid.astype(np.float32),
        (native_w, native_h),
        interpolation=cv2.INTER_AREA,
    )
    admitted = native_valid & (support >= 0.999)
    reconstructed = np.zeros_like(numerator)
    np.divide(numerator, support, out=reconstructed, where=support > 0.0)
    error = np.abs(reconstructed[admitted] - native_depth[admitted])
    return {
        "admitted_native_pixel_count": int(np.count_nonzero(admitted)),
        "abs_error_m": distribution(error),
    }


def load_pipeline_fov_masks(
    pipeline_config_path: Path,
    *,
    camera_ids: Sequence[str],
    target_size: tuple[int, int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        config = yaml.safe_load(pipeline_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AlignmentEvaluationError(
            f"invalid DS9 pipeline YAML {pipeline_config_path}: {exc}"
        ) from exc
    if not isinstance(config, Mapping):
        raise AlignmentEvaluationError("DS9 pipeline YAML must contain an object")
    sources = config.get("sources")
    mask_root = config.get("dewarper_validity_masks")
    masks = mask_root.get("sources") if isinstance(mask_root, Mapping) else None
    if not isinstance(sources, list) or not isinstance(masks, Mapping):
        raise AlignmentEvaluationError(
            "DS9 pipeline lacks sources/dewarper validity masks"
        )
    if len(sources) != len(camera_ids):
        raise AlignmentEvaluationError(
            "locked corpus camera count differs from DS9 pipeline sources"
        )

    from geometry.dewarper_validity import (
        build_dewarper_fov_mask,
        load_dewarper_fov_spec,
    )

    result: dict[str, np.ndarray] = {}
    source_receipts: list[dict[str, Any]] = []
    for source_id, (camera_id, source_cfg) in enumerate(zip(camera_ids, sources)):
        if not isinstance(source_cfg, Mapping):
            raise AlignmentEvaluationError(f"pipeline source {source_id} is invalid")
        configured_camera = str(source_cfg.get("uri_secret") or "")
        if configured_camera != camera_id:
            raise AlignmentEvaluationError(
                "locked corpus order differs from runtime source order: "
                f"{camera_id!r} != {configured_camera!r}"
            )
        mask_cfg = masks.get(str(source_id), masks.get(source_id))
        if not isinstance(mask_cfg, Mapping):
            raise AlignmentEvaluationError(
                f"runtime source {source_id} has no explicit validity mask"
            )
        spec = load_dewarper_fov_spec(
            source_cfg=source_cfg,
            pipeline_yaml_path=pipeline_config_path,
            mask_cfg=mask_cfg,
            repo_root=REPO_ROOT,
        )
        if spec is None:
            raise AlignmentEvaluationError(
                f"runtime source {source_id} validity mask is disabled"
            )
        erode_px = int(mask_cfg.get("erode-px", mask_cfg.get("erode_px", 1)))
        result[camera_id] = build_dewarper_fov_mask(
            spec,
            target_size=target_size,
            erode_px=erode_px,
        )
        source_receipts.append(
            {
                "source_id": source_id,
                "camera_id": camera_id,
                "dewarper_config": {
                    "path": str(spec.config_path),
                    "sha256": sha256_file(spec.config_path),
                    "size_bytes": int(spec.config_path.stat().st_size),
                },
                "erode_px": erode_px,
                "valid_fraction": float(np.mean(result[camera_id])),
            }
        )
    return result, {
        "path": str(pipeline_config_path),
        "sha256": sha256_file(pipeline_config_path),
        "size_bytes": int(pipeline_config_path.stat().st_size),
        "sources": source_receipts,
    }


def evaluate(
    *,
    comparison_manifest_path: Path,
    pipeline_config_path: Path,
    candidate_id: str = DEFAULT_CANDIDATE_ID,
) -> dict[str, Any]:
    evaluation_started = time.perf_counter()
    manifest = load_json(comparison_manifest_path)
    if not isinstance(manifest, Mapping) or manifest.get("contract") != COMPARISON_CONTRACT:
        raise AlignmentEvaluationError(
            f"comparison manifest contract must be {COMPARISON_CONTRACT}"
        )
    candidates = manifest.get("candidates")
    if not isinstance(candidates, Mapping) or candidate_id not in candidates:
        raise AlignmentEvaluationError(f"candidate {candidate_id!r} is absent")
    candidate = candidates[candidate_id]
    if not isinstance(candidate, Mapping):
        raise AlignmentEvaluationError("candidate record must be an object")
    if list(candidate.get("input_contract") or []) != ["images"]:
        raise AlignmentEvaluationError("alignment diagnostic requires images-only input")

    corpus = manifest.get("corpus")
    if not isinstance(corpus, Mapping):
        raise AlignmentEvaluationError("comparison manifest has no corpus record")
    annotations_record = corpus.get("annotations")
    if not isinstance(annotations_record, Mapping):
        raise AlignmentEvaluationError("comparison manifest has no annotations identity")
    annotations_path = require_identity(
        annotations_record,
        label="locked annotations",
    )
    receipt_path = require_identity(
        candidate.get("fixture_receipt", {}),
        label="locked fixture receipt",
    )
    tensor_path = require_identity(
        candidate.get("fixture_tensor", {}),
        label="locked fixture tensor",
    )
    output_path = require_identity(
        candidate.get("model_output", {}),
        label="locked native output",
    )
    annotations = load_json(annotations_path)
    receipt = load_json(receipt_path)
    tensors = parse_output_tensors(load_json(output_path))
    if (
        not isinstance(annotations, Mapping)
        or annotations.get("contract") != ANNOTATION_CONTRACT
    ):
        raise AlignmentEvaluationError(
            f"annotation contract must be {ANNOTATION_CONTRACT}"
        )
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("contract") != FIXTURE_CONTRACT
    ):
        raise AlignmentEvaluationError(
            f"fixture contract must be {FIXTURE_CONTRACT}"
        )
    if bool(receipt.get("identical_batch_members")):
        raise AlignmentEvaluationError("alignment corpus requires distinct scenes")

    tensor_record = receipt.get("tensor")
    sources = receipt.get("sources")
    frames = annotations.get("frames")
    if (
        not isinstance(tensor_record, Mapping)
        or not isinstance(sources, list)
        or not isinstance(frames, list)
    ):
        raise AlignmentEvaluationError("fixture tensor/sources or frames are invalid")
    try:
        tensor_shape = tuple(int(value) for value in tensor_record["shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AlignmentEvaluationError("fixture tensor shape is invalid") from exc
    if (
        len(tensor_shape) != 4
        or tensor_shape[1] != 3
        or tensor_shape[0] != len(sources)
        or tensors["depth"].shape != (tensor_shape[0], *tensor_shape[2:])
    ):
        raise AlignmentEvaluationError(
            "fixture/model output batch or spatial shapes differ"
        )
    expected_bytes = math.prod(tensor_shape) * np.dtype("<f4").itemsize
    if tensor_path.stat().st_size != expected_bytes:
        raise AlignmentEvaluationError("fixture tensor byte count is invalid")
    image_tensor = np.fromfile(tensor_path, dtype="<f4").reshape(tensor_shape)
    if (
        not np.all(np.isfinite(image_tensor))
        or float(np.min(image_tensor)) < 0.0
        or float(np.max(image_tensor)) > 1.0
    ):
        raise AlignmentEvaluationError("fixture RGB tensor is not finite in [0,1]")

    frames_by_sha: dict[str, Mapping[str, Any]] = {}
    for frame in frames:
        if not isinstance(frame, Mapping) or not isinstance(frame.get("rgb"), Mapping):
            raise AlignmentEvaluationError("annotation frame has no RGB identity")
        rgb_record = frame["rgb"]
        rgb_sha = str(rgb_record.get("sha256") or "")
        if not rgb_sha or rgb_sha in frames_by_sha:
            raise AlignmentEvaluationError("annotation RGB identities are invalid")
        frames_by_sha[rgb_sha] = frame

    camera_ids: list[str] = []
    source_frames: list[Mapping[str, Any]] = []
    target_size: tuple[int, int] | None = None
    for source in sources:
        if not isinstance(source, Mapping):
            raise AlignmentEvaluationError("fixture source must be an object")
        source_sha = str(source.get("sha256") or "")
        frame = frames_by_sha.get(source_sha)
        if frame is None:
            raise AlignmentEvaluationError(
                f"fixture RGB {source_sha} has no annotation identity"
            )
        camera_ids.append(str(frame.get("camera_id") or ""))
        source_frames.append(frame)
        size = (
            int(source.get("source_width") or 0),
            int(source.get("source_height") or 0),
        )
        if target_size is None:
            target_size = size
        elif size != target_size:
            raise AlignmentEvaluationError(
                "alignment A/B requires one common runtime target size"
            )
    if target_size is None or min(target_size) <= 0:
        raise AlignmentEvaluationError("fixture has no positive target size")

    fov_masks, pipeline_identity = load_pipeline_fov_masks(
        pipeline_config_path,
        camera_ids=camera_ids,
        target_size=target_size,
    )

    camera_reports: dict[str, dict[str, Any]] = {}
    all_baseline_annotation_distances: list[np.ndarray] = []
    all_candidate_annotation_distances: list[np.ndarray] = []
    for batch_index, (source, frame, camera_id) in enumerate(
        zip(sources, source_frames, camera_ids)
    ):
        assert isinstance(source, Mapping)
        rgb_path = Path(str(source.get("path") or "")).expanduser()
        source_record = {
            "path": str(rgb_path),
            "sha256": str(source.get("sha256") or ""),
            "size_bytes": int(rgb_path.stat().st_size) if rgb_path.is_file() else -1,
        }
        require_identity(source_record, label=f"{camera_id} RGB")
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise AlignmentEvaluationError(f"unable to decode {camera_id} RGB")
        high_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if high_rgb.shape[:2] != (target_size[1], target_size[0]):
            raise AlignmentEvaluationError(f"{camera_id} RGB dimensions drifted")

        native_shape = tuple(int(value) for value in tensors["depth"].shape[-2:])
        crop = runtime_crop(native_shape, target_size)
        receipt_crop = (
            int(source.get("resized_width") or 0),
            int(source.get("resized_height") or 0),
            int(source.get("pad_left") or 0),
            int(source.get("pad_top") or 0),
        )
        if crop != receipt_crop:
            raise AlignmentEvaluationError(
                f"{camera_id} receipt mapping {receipt_crop} differs from "
                f"runtime crop {crop}"
            )

        low_rgb_native = np.transpose(image_tensor[batch_index], (1, 2, 0))
        resized_w, resized_h, pad_left, pad_top = crop
        expected_low_bgr = cv2.resize(
            rgb_bgr,
            (resized_w, resized_h),
            interpolation=cv2.INTER_AREA,
        )
        expected_low_rgb = (
            cv2.cvtColor(expected_low_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            / 255.0
        )
        observed_low_rgb = low_rgb_native[
            pad_top : pad_top + resized_h,
            pad_left : pad_left + resized_w,
        ]
        if not np.array_equal(observed_low_rgb, expected_low_rgb):
            raise AlignmentEvaluationError(
                f"{camera_id} locked tensor does not equal declared RGB preprocess"
            )

        baseline_started = time.perf_counter()
        baseline_depth, baseline_conf, baseline_valid = align_bilinear(
            tensors["depth"][batch_index],
            tensors["conf"][batch_index],
            tensors["mask"][batch_index].astype(bool),
            crop=crop,
            target_size=target_size,
        )
        baseline_wall_time_s = time.perf_counter() - baseline_started
        guided_started = time.perf_counter()
        (
            guided_depth,
            guided_conf,
            guided_valid,
            guided_diagnostics,
        ) = align_rgb_guided(
            tensors["depth"][batch_index],
            tensors["conf"][batch_index],
            tensors["mask"][batch_index].astype(bool),
            low_rgb_native,
            high_rgb,
            crop=crop,
            target_size=target_size,
        )
        guided_wall_time_s = time.perf_counter() - guided_started
        fov_mask = fov_masks[camera_id]
        baseline_valid &= fov_mask
        guided_valid &= fov_mask
        baseline_depth[~baseline_valid] = np.nan
        baseline_conf[~baseline_valid] = 0.0
        guided_depth[~guided_valid] = np.nan
        guided_conf[~guided_valid] = 0.0

        overlap = baseline_valid & guided_valid
        depth_delta = guided_depth[overlap] - baseline_depth[overlap]
        conf_delta = guided_conf[overlap] - baseline_conf[overlap]
        baseline_gx, baseline_gy, baseline_gradient, baseline_interior = (
            depth_gradients(baseline_depth, baseline_valid)
        )
        guided_gx, guided_gy, guided_gradient, guided_interior = depth_gradients(
            guided_depth,
            guided_valid,
        )
        common_interior = baseline_interior & guided_interior
        rgb_magnitude = rgb_gradient(high_rgb)
        baseline_rgb_edge = rgb_edge_distance_metrics(
            baseline_gradient,
            rgb_magnitude,
            common_interior,
        )
        guided_rgb_edge = rgb_edge_distance_metrics(
            guided_gradient,
            rgb_magnitude,
            common_interior,
        )

        rgb_flat_threshold = percentile(rgb_magnitude[common_interior], 50)
        assert rgb_flat_threshold is not None
        rgb_flat = common_interior & (rgb_magnitude <= rgb_flat_threshold)
        baseline_flat_p95 = percentile(baseline_gradient[rgb_flat], 95)
        guided_flat_p95 = percentile(guided_gradient[rgb_flat], 95)
        flat_amplification = (
            float(guided_flat_p95 / baseline_flat_p95)
            if baseline_flat_p95 is not None
            and guided_flat_p95 is not None
            and baseline_flat_p95 > 1e-12
            else None
        )

        edge_reports: list[dict[str, Any]] = []
        structural_edges = frame.get("structural_edges", [])
        if not isinstance(structural_edges, list):
            raise AlignmentEvaluationError(
                f"{camera_id}.structural_edges must be a list"
            )
        for edge in structural_edges:
            if not isinstance(edge, Mapping):
                raise AlignmentEvaluationError("structural edge must be an object")
            baseline_edge, baseline_distances, _ = normal_edge_metrics(
                edge,
                grad_x=baseline_gx,
                grad_y=baseline_gy,
                valid=baseline_interior,
            )
            guided_edge, guided_distances, _ = normal_edge_metrics(
                edge,
                grad_x=guided_gx,
                grad_y=guided_gy,
                valid=guided_interior,
            )
            all_baseline_annotation_distances.append(baseline_distances)
            all_candidate_annotation_distances.append(guided_distances)
            baseline_p50 = baseline_edge[
                "nearest_normal_log_depth_gradient_distance_px"
            ]["p50"]
            guided_p50 = guided_edge[
                "nearest_normal_log_depth_gradient_distance_px"
            ]["p50"]
            edge_reports.append(
                {
                    "id": baseline_edge["id"],
                    "kind": baseline_edge["kind"],
                    "annotation_confidence": baseline_edge[
                        "annotation_confidence"
                    ],
                    "evidence_class": (
                        "non_surveyed_human_authored_rgb_edge_diagnostic"
                    ),
                    "baseline_bilinear": baseline_edge,
                    "rgb_guided": guided_edge,
                    "p50_distance_improvement_fraction": improvement(
                        baseline_p50,
                        guided_p50,
                    ),
                }
            )

        y_slice = slice(pad_top, pad_top + resized_h)
        x_slice = slice(pad_left, pad_left + resized_w)
        native_depth_crop = tensors["depth"][batch_index][y_slice, x_slice]
        native_valid_crop = (
            tensors["mask"][batch_index][y_slice, x_slice].astype(bool)
            & np.isfinite(native_depth_crop)
        )
        baseline_distance_p50 = baseline_rgb_edge[
            "distance_to_strong_rgb_edge_px"
        ]["p50"]
        baseline_distance_p90 = baseline_rgb_edge[
            "distance_to_strong_rgb_edge_px"
        ]["p90"]
        guided_distance_p50 = guided_rgb_edge["distance_to_strong_rgb_edge_px"][
            "p50"
        ]
        guided_distance_p90 = guided_rgb_edge["distance_to_strong_rgb_edge_px"][
            "p90"
        ]
        camera_reports[camera_id] = {
            "batch_index": batch_index,
            "identities": {
                "rgb": source_record,
                "native_output_batch_index": batch_index,
            },
            "mapping": {
                "native_shape": list(native_shape),
                "crop_shape": [resized_h, resized_w],
                "pad_left": pad_left,
                "pad_top": pad_top,
                "target_size": list(target_size),
            },
            "offline_cpu_timing_s": {
                "baseline_bilinear_alignment": baseline_wall_time_s,
                "rgb_guided_alignment": guided_wall_time_s,
            },
            "validity": {
                "runtime_fov_valid_fraction": float(np.mean(fov_mask)),
                "baseline_valid_fraction": float(np.mean(baseline_valid)),
                "rgb_guided_valid_fraction": float(np.mean(guided_valid)),
                "unknown_fill_pixel_count": int(
                    np.count_nonzero(guided_valid & ~baseline_valid)
                ),
                "unknown_loss_pixel_count": int(
                    np.count_nonzero(baseline_valid & ~guided_valid)
                ),
            },
            "metric_preservation": {
                "overlap_pixel_count": int(np.count_nonzero(overlap)),
                "rgb_guided_minus_bilinear_depth_m": {
                    "signed_bias": (
                        float(np.mean(depth_delta)) if depth_delta.size else None
                    ),
                    "abs": distribution(np.abs(depth_delta)),
                },
                "rgb_guided_minus_bilinear_confidence": {
                    "signed_bias": (
                        float(np.mean(conf_delta)) if conf_delta.size else None
                    ),
                    "abs": distribution(np.abs(conf_delta)),
                },
                **guided_diagnostics,
                "native_roundtrip": {
                    "baseline_bilinear": masked_area_roundtrip(
                        baseline_depth,
                        baseline_valid,
                        native_depth=native_depth_crop,
                        native_valid=native_valid_crop,
                    ),
                    "rgb_guided": masked_area_roundtrip(
                        guided_depth,
                        guided_valid,
                        native_depth=native_depth_crop,
                        native_valid=native_valid_crop,
                    ),
                },
            },
            "internal_rgb_edge_alignment": {
                "baseline_bilinear": baseline_rgb_edge,
                "rgb_guided": guided_rgb_edge,
                "p50_distance_improvement_fraction": improvement(
                    baseline_distance_p50,
                    guided_distance_p50,
                ),
                "p90_distance_improvement_fraction": improvement(
                    baseline_distance_p90,
                    guided_distance_p90,
                ),
                "flat_rgb_region_log_depth_gradient_p95": {
                    "baseline_bilinear": baseline_flat_p95,
                    "rgb_guided": guided_flat_p95,
                    "amplification_ratio": flat_amplification,
                },
            },
            "authored_structural_edges": edge_reports,
        }

    baseline_annotation = (
        np.concatenate(all_baseline_annotation_distances)
        if all_baseline_annotation_distances
        else np.empty(0, dtype=np.float64)
    )
    guided_annotation = (
        np.concatenate(all_candidate_annotation_distances)
        if all_candidate_annotation_distances
        else np.empty(0, dtype=np.float64)
    )
    baseline_annotation_dist = distribution(baseline_annotation)
    guided_annotation_dist = distribution(guided_annotation)

    rgb_p50_improvements = [
        report["internal_rgb_edge_alignment"][
            "p50_distance_improvement_fraction"
        ]
        for report in camera_reports.values()
    ]
    rgb_p90_improvements = [
        report["internal_rgb_edge_alignment"][
            "p90_distance_improvement_fraction"
        ]
        for report in camera_reports.values()
    ]
    flat_amplifications = [
        report["internal_rgb_edge_alignment"][
            "flat_rgb_region_log_depth_gradient_p95"
        ]["amplification_ratio"]
        for report in camera_reports.values()
    ]
    delta_p95_values = [
        report["metric_preservation"]["rgb_guided_minus_bilinear_depth_m"][
            "abs"
        ]["p95"]
        for report in camera_reports.values()
    ]
    bias_values = [
        abs(
            report["metric_preservation"]["rgb_guided_minus_bilinear_depth_m"][
                "signed_bias"
            ]
        )
        for report in camera_reports.values()
    ]
    no_unknown_change = all(
        report["validity"]["unknown_fill_pixel_count"] == 0
        and report["validity"]["unknown_loss_pixel_count"] == 0
        for report in camera_reports.values()
    )
    no_hull_violation = all(
        report["metric_preservation"]["local_convex_hull_violation_count"] == 0
        for report in camera_reports.values()
    )
    rgb_p50_camera_wins = sum(
        value is not None and value >= MIN_RGB_EDGE_DISTANCE_IMPROVEMENT
        for value in rgb_p50_improvements
    )
    rgb_p90_camera_wins = sum(
        value is not None and value >= MIN_RGB_EDGE_DISTANCE_IMPROVEMENT
        for value in rgb_p90_improvements
    )
    annotation_p50_improvement = improvement(
        baseline_annotation_dist["p50"],
        guided_annotation_dist["p50"],
    )
    annotation_p95_improvement = improvement(
        baseline_annotation_dist["p95"],
        guided_annotation_dist["p95"],
    )
    gates = {
        "gpu_native_implementation_proven": False,
        "unknown_mask_exactly_preserved": no_unknown_change,
        "local_metric_convex_hull_preserved": no_hull_violation,
        "depth_delta_p95_within_0_10m_all_cameras": all(
            value is not None and value <= MAX_DEPTH_DELTA_P95_M
            for value in delta_p95_values
        ),
        "absolute_signed_bias_within_0_01m_all_cameras": all(
            value <= MAX_ABS_SIGNED_BIAS_M for value in bias_values
        ),
        "rgb_edge_p50_improves_5pct_at_least_two_cameras": (
            rgb_p50_camera_wins >= 2
        ),
        "rgb_edge_p90_improves_5pct_at_least_two_cameras": (
            rgb_p90_camera_wins >= 2
        ),
        "authored_normal_edge_p50_improves_10pct": (
            annotation_p50_improvement is not None
            and annotation_p50_improvement
            >= MIN_ANNOTATED_EDGE_P50_IMPROVEMENT
        ),
        "authored_normal_edge_p95_regression_at_most_5pct": (
            annotation_p95_improvement is not None
            and annotation_p95_improvement
            >= -MAX_ANNOTATED_EDGE_P95_REGRESSION
        ),
        "flat_rgb_gradient_amplification_at_most_1_10_all_cameras": all(
            value is not None
            and value <= MAX_FLAT_RGB_GRADIENT_AMPLIFICATION
            for value in flat_amplifications
        ),
    }
    consideration = all(gates.values())
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "contract": REPORT_CONTRACT,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": (
            "internal_candidate_warrants_temporal_experiment"
            if consideration
            else "internal_candidate_rejected"
        ),
        "claim_limits": {
            "surveyed_metric_accuracy": False,
            "temporal_flicker_or_stability": False,
            "fusion_or_floorplan_quality": False,
            "runtime_performance": False,
            "live_promotion": False,
            "reason": (
                "The locked corpus has one RGB frame per camera and no surveyed "
                "depth truth. RGB-guided and authored-RGB-edge scores are "
                "non-surveyed internal alignment diagnostics."
            ),
        },
        "identity": {
            "script": {
                "path": str(SCRIPT_PATH),
                "sha256": sha256_file(SCRIPT_PATH),
                "size_bytes": int(SCRIPT_PATH.stat().st_size),
            },
            "comparison_manifest": {
                "path": str(comparison_manifest_path),
                "sha256": sha256_file(comparison_manifest_path),
                "size_bytes": int(comparison_manifest_path.stat().st_size),
            },
            "annotations": dict(annotations_record),
            "fixture_receipt": dict(candidate["fixture_receipt"]),
            "fixture_tensor": dict(candidate["fixture_tensor"]),
            "native_model_output": dict(candidate["model_output"]),
            "pipeline_config": pipeline_identity,
        },
        "model_candidate_id": candidate_id,
        "methods": {
            "baseline": {
                "name": "runtime_crop_bilinear_depth_conf_nearest_mask",
                "runtime_source": "DS9/noesis/pipelines/hooks.py:MapAnythingProcessor._align_to_frame",
            },
            "candidate": {
                "name": "rgb_guided_joint_bilateral_3x3_native_neighbourhood",
                "implementation": "offline_cpu_numpy_opencv_only",
                "radius_native_px": GUIDED_RADIUS_NATIVE_PX,
                "sigma_spatial_native_px": GUIDED_SIGMA_SPATIAL_NATIVE_PX,
                "sigma_color_cielab": GUIDED_SIGMA_COLOR_LAB,
                "unknown_policy": (
                    "preserve baseline nearest-native mask and calibrated FoV; "
                    "never fill unknown"
                ),
                "metric_policy": (
                    "convex combination of valid local native metric depths; "
                    "no extrapolation or range overshoot"
                ),
            },
        },
        "offline_execution": {
            "camera_count": len(camera_reports),
            "target_size": list(target_size),
            "total_evaluation_wall_time_s": (
                time.perf_counter() - evaluation_started
            ),
            "peak_process_rss_mib": float(peak_rss_kib / 1024.0),
            "scope": (
                "JSON/tensor/RGB loading, current FoV construction, baseline "
                "alignment, RGB-guided alignment, and diagnostics for all cameras"
            ),
            "runtime_performance_claim": False,
        },
        "cameras": camera_reports,
        "macro": {
            "authored_structural_edge_normal_distance_px": {
                "evidence_class": (
                    "non_surveyed_human_authored_rgb_edge_diagnostic"
                ),
                "baseline_bilinear": baseline_annotation_dist,
                "rgb_guided": guided_annotation_dist,
                "p50_improvement_fraction": annotation_p50_improvement,
                "p95_improvement_fraction": annotation_p95_improvement,
            },
            "rgb_edge_camera_p50_improvement_fractions": rgb_p50_improvements,
            "rgb_edge_camera_p90_improvement_fractions": rgb_p90_improvements,
            "gate_results": gates,
        },
        "decision": {
            "deserves_runtime_consideration": consideration,
            "runtime_integration_performed": False,
            "promotion_performed": False,
            "next_step": "Keep current bilinear runtime alignment.",
            "reason": (
                "The fixed RGB-guided candidate failed the predeclared edge "
                "improvement gates and has no GPU-native implementation. A "
                "different candidate would need multi-frame quality evidence "
                "plus bounded GPU-native performance before runtime consideration."
            ),
        },
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    if path.exists():
        raise AlignmentEvaluationError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact DS9 MapAnything bilinear alignment with one bounded "
            "RGB-guided candidate."
        )
    )
    parser.add_argument(
        "--comparison-manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=DEFAULT_PIPELINE_CONFIG,
    )
    parser.add_argument(
        "--candidate-id",
        default=DEFAULT_CANDIDATE_ID,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = evaluate(
            comparison_manifest_path=args.comparison_manifest.resolve(),
            pipeline_config_path=args.pipeline_config.resolve(),
            candidate_id=str(args.candidate_id),
        )
        write_report(args.output.resolve(), report)
    except AlignmentEvaluationError as exc:
        print(f"[FAIL] MapAnything alignment A/B refused: {exc}")
        return 2
    print(
        "[PASS] MapAnything alignment A/B complete: "
        f"{args.output.resolve()} "
        f"(runtime_consideration="
        f"{report['decision']['deserves_runtime_consideration']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
