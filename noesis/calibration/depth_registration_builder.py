from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from noesis.calibration.depth_registration import (
    DepthRegistrationEntry,
    REGISTRATION_SCOPE,
    SOURCE_SPACE,
    TARGET_SPACE,
    TRANSFORM_TYPE_PIECEWISE,
)

DEFAULT_NUM_KNOTS = 8
DEFAULT_MIN_SAMPLES = 512


class DepthRegistrationBuildError(Exception):
    """Raised when offline registration fitting cannot produce a usable artifact."""


@dataclass(frozen=True, slots=True)
class DepthRegistrationFitResult:
    knots_raw_m: tuple[float, ...]
    knots_registered_m: tuple[float, ...]
    fit_metrics: Mapping[str, Any]
    sample_counts: Mapping[str, Any]


def _robust_percentile_clip(values: np.ndarray, low_q: float = 2.5, high_q: float = 97.5) -> np.ndarray:
    lo = float(np.percentile(values, low_q))
    hi = float(np.percentile(values, high_q))
    return values[(values >= lo) & (values <= hi)]


def _enforce_monotonic(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted isotonic least squares, not a one-sided cumulative maximum.

    A noisy high near-depth bin must not lift every farther reference depth to
    that value. Pool adjacent violating bins using their observed support. A
    genuinely unobservable plateau stays flat and remains runtime-ineligible.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.shape != values.shape or not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise DepthRegistrationBuildError("isotonic values/positive weights must be finite and aligned")
    blocks: list[list[float | int]] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([index, index + 1, float(value * weight), float(weight)])
        while len(blocks) >= 2 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    out = np.empty_like(values)
    for start, end, weighted_sum, total_weight in blocks:
        out[int(start):int(end)] = weighted_sum / total_weight
    return out


def fit_monotonic_piecewise_mapping(
    raw_depth_m: Sequence[float],
    registered_depth_m: Sequence[float],
    *,
    num_knots: int = DEFAULT_NUM_KNOTS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> DepthRegistrationFitResult:
    raw = np.asarray(raw_depth_m, dtype=np.float64).reshape(-1)
    target = np.asarray(registered_depth_m, dtype=np.float64).reshape(-1)
    if raw.size != target.size:
        raise DepthRegistrationBuildError("raw_depth_m and registered_depth_m must have the same length")
    finite = np.isfinite(raw) & np.isfinite(target) & (raw > 0.0) & (target > 0.0)
    raw = raw[finite]
    target = target[finite]
    if raw.size < max(32, int(min_samples)):
        raise DepthRegistrationBuildError(
            f"Need at least {max(32, int(min_samples))} valid sample pairs; got {raw.size}"
        )
    order = np.argsort(raw)
    raw = raw[order]
    target = target[order]

    clipped_raw = _robust_percentile_clip(raw)
    lo = float(clipped_raw.min())
    hi = float(clipped_raw.max())
    keep = (raw >= lo) & (raw <= hi)
    raw = raw[keep]
    target = target[keep]

    knot_count = max(4, int(num_knots))
    quantiles = np.linspace(0.0, 1.0, knot_count)
    raw_knots: list[float] = []
    target_knots: list[float] = []
    knot_support: list[int] = []
    retained_total = 0
    for idx in range(knot_count - 1):
        q_lo = float(quantiles[idx])
        q_hi = float(quantiles[idx + 1])
        raw_lo = float(np.quantile(raw, q_lo))
        raw_hi = float(np.quantile(raw, q_hi))
        if idx == knot_count - 2:
            mask = (raw >= raw_lo) & (raw <= raw_hi)
        else:
            mask = (raw >= raw_lo) & (raw < raw_hi)
        bin_raw = raw[mask]
        bin_target = target[mask]
        if bin_raw.size == 0 or bin_target.size == 0:
            continue
        med = float(np.median(bin_target))
        mad = float(np.median(np.abs(bin_target - med)))
        if mad > 1e-6:
            robust = np.abs(bin_target - med) <= (3.5 * mad)
            bin_raw = bin_raw[robust]
            bin_target = bin_target[robust]
        if bin_raw.size == 0:
            continue
        retained_total += int(bin_raw.size)
        raw_knots.append(float(np.median(bin_raw)))
        target_knots.append(float(np.median(bin_target)))
        knot_support.append(int(bin_raw.size))
    if len(raw_knots) < 2:
        raise DepthRegistrationBuildError("Unable to construct a monotonic mapping from the provided samples")

    raw_knots[0] = lo
    raw_knots[-1] = hi
    reg_knots = _enforce_monotonic(
        np.asarray(target_knots, dtype=np.float64), np.asarray(knot_support, dtype=np.float64)
    )
    if reg_knots.size != len(raw_knots):
        raise DepthRegistrationBuildError("Internal knot vector mismatch")

    corrected = np.interp(raw, raw_knots, reg_knots)
    residual = corrected - target
    fit_metrics = {
        "monotonic_fit_method": "weighted_pool_adjacent_violators",
        "median_abs_error_m": float(np.median(np.abs(residual))),
        "mean_abs_error_m": float(np.mean(np.abs(residual))),
        "p90_abs_error_m": float(np.percentile(np.abs(residual), 90.0)),
        "raw_domain_m": [float(lo), float(hi)],
    }
    sample_counts = {
        "input_pairs": int(len(raw_depth_m)),
        "retained_pairs": int(raw.size),
        "retained_after_bin_filter": int(retained_total),
        "num_knots": int(len(raw_knots)),
    }
    return DepthRegistrationFitResult(
        knots_raw_m=tuple(float(v) for v in raw_knots),
        knots_registered_m=tuple(float(v) for v in reg_knots.tolist()),
        fit_metrics=fit_metrics,
        sample_counts=sample_counts,
    )


def build_registration_entry(
    *,
    camera_id: str,
    calibration_fingerprint: Mapping[str, Any],
    dav2_profile: Mapping[str, Any],
    mapanything_profile: Mapping[str, Any],
    provenance: Mapping[str, Any],
    raw_depth_m: Sequence[float],
    registered_depth_m: Sequence[float],
    generation_tool_version: str = "depth_registration_builder_v2_isotonic",
    created_ts_us: Optional[int] = None,
    num_knots: int = DEFAULT_NUM_KNOTS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> DepthRegistrationEntry:
    fit = fit_monotonic_piecewise_mapping(
        raw_depth_m,
        registered_depth_m,
        num_knots=num_knots,
        min_samples=min_samples,
    )
    now_us = int(created_ts_us) if created_ts_us is not None else int(time.time() * 1_000_000)
    return DepthRegistrationEntry(
        camera_id=str(camera_id),
        created_ts_us=now_us,
        transform_type=TRANSFORM_TYPE_PIECEWISE,
        source_space=SOURCE_SPACE,
        target_space=TARGET_SPACE,
        scope=REGISTRATION_SCOPE,
        raw_range_domain_m=(float(fit.knots_raw_m[0]), float(fit.knots_raw_m[-1])),
        knots_raw_m=fit.knots_raw_m,
        knots_registered_m=fit.knots_registered_m,
        calibration_fingerprint=dict(calibration_fingerprint),
        dav2_profile=dict(dav2_profile),
        mapanything_profile=dict(mapanything_profile),
        fit_metrics=dict(fit.fit_metrics),
        sample_counts=dict(fit.sample_counts),
        generation_tool_version=str(generation_tool_version),
        provenance=dict(provenance),
    )


def flatten_sample_pairs(
    frames: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    min_map_depth_m: float = 0.1,
    max_depth_m: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    raw_samples: list[np.ndarray] = []
    ref_samples: list[np.ndarray] = []
    for da2_depth, ma_depth in frames:
        da = np.asarray(da2_depth, dtype=np.float64)
        ma = np.asarray(ma_depth, dtype=np.float64)
        if da.shape != ma.shape:
            raise DepthRegistrationBuildError("DAv2 and MapAnything frame samples must share shape")
        valid = np.isfinite(da) & np.isfinite(ma) & (da > 0.0) & (ma >= float(min_map_depth_m))
        valid &= (da <= float(max_depth_m)) & (ma <= float(max_depth_m))
        if not np.any(valid):
            continue
        raw_samples.append(da[valid].reshape(-1))
        ref_samples.append(ma[valid].reshape(-1))
    if not raw_samples or not ref_samples:
        raise DepthRegistrationBuildError("No valid dense depth pairs available for fitting")
    return np.concatenate(raw_samples, axis=0), np.concatenate(ref_samples, axis=0)
