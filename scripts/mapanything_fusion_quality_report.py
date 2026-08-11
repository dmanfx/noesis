#!/usr/bin/env python3
"""Compare MapAnything temporal-fusion candidates on one exact raw cohort."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi

try:
    import zarr
except Exception as exc:  # pragma: no cover - command-line dependency guard
    raise SystemExit(f"zarr is required: {exc}") from exc


def _finite_percentiles(
    values: np.ndarray,
    quantiles: Sequence[float],
) -> list[float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [None for _ in quantiles]
    return [float(value) for value in np.percentile(finite, quantiles)]


def _pearson_sampled(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_samples: int = 1_000_000,
) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    left_values = np.asarray(left[valid], dtype=np.float64)
    right_values = np.asarray(right[valid], dtype=np.float64)
    if left_values.size < 2:
        return None
    if left_values.size > max_samples:
        stride = int(math.ceil(left_values.size / max_samples))
        left_values = left_values[::stride]
        right_values = right_values[::stride]
    left_values -= np.mean(left_values)
    right_values -= np.mean(right_values)
    denominator = float(
        np.sqrt(np.sum(left_values * left_values) * np.sum(right_values * right_values))
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    return float(np.sum(left_values * right_values) / denominator)


def _edge_metrics(depth: np.ndarray, valid: np.ndarray) -> dict[str, float | None]:
    horizontal_valid = valid[:, 1:] & valid[:, :-1]
    vertical_valid = valid[1:, :] & valid[:-1, :]
    horizontal = np.abs(depth[:, 1:] - depth[:, :-1])[horizontal_valid]
    vertical = np.abs(depth[1:, :] - depth[:-1, :])[vertical_valid]
    if horizontal.size and vertical.size:
        gradients = np.concatenate((horizontal, vertical))
    elif horizontal.size:
        gradients = horizontal
    else:
        gradients = vertical
    p50, p95 = _finite_percentiles(gradients, (50.0, 95.0))
    return {
        "neighbor_count": int(gradients.size),
        "gradient_p50_m": p50,
        "gradient_p95_m": p95,
    }


def _binary_mask_component_evidence(mask: np.ndarray) -> dict[str, int | float]:
    """Mirror the runtime's compact 8-connected fragment/hole evidence."""

    binary = np.asarray(mask, dtype=bool)
    pixels = int(np.count_nonzero(binary))
    if pixels <= 0:
        return {
            "connectivity": 8,
            "component_count": 0,
            "largest_component_pixels": 0,
            "largest_component_fraction": 0.0,
            "fragment_pixels": 0,
            "fragment_fraction": 0.0,
            "hole_count": 0,
            "hole_pixels": 0,
        }
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, component_count = ndi.label(binary, structure=structure)
    counts = np.bincount(labels.ravel())
    sizes = counts[1:] if counts.size > 1 else np.empty(0, dtype=np.int64)
    largest = int(sizes.max()) if sizes.size else 0
    fragments = max(0, pixels - largest)
    holes = np.asarray(ndi.binary_fill_holes(binary), dtype=bool) & ~binary
    _hole_labels, hole_count = ndi.label(holes, structure=structure)
    return {
        "connectivity": 8,
        "component_count": int(component_count),
        "largest_component_pixels": largest,
        "largest_component_fraction": float(largest / pixels),
        "fragment_pixels": fragments,
        "fragment_fraction": float(fragments / pixels),
        "hole_count": int(hole_count),
        "hole_pixels": int(np.count_nonzero(holes)),
    }


def _candidate_metrics(
    fused: np.ndarray,
    fused_valid: np.ndarray,
    depth_stack: np.ndarray,
    source_valid: np.ndarray,
) -> dict[str, Any]:
    residual_valid = source_valid & fused_valid[None, :, :]
    residual = np.abs(depth_stack - fused[None, :, :])[residual_valid]
    residual_p50, residual_p95 = _finite_percentiles(residual, (50.0, 95.0))
    depth_p02, depth_p50, depth_p98 = _finite_percentiles(
        fused[fused_valid],
        (2.0, 50.0, 98.0),
    )
    return {
        "valid_fraction": float(np.count_nonzero(fused_valid) / fused_valid.size),
        "depth_p02_m": depth_p02,
        "depth_p50_m": depth_p50,
        "depth_p98_m": depth_p98,
        "temporal_residual_p50_m": residual_p50,
        "temporal_residual_p95_m": residual_p95,
        "edge": _edge_metrics(fused, fused_valid),
    }


def evaluate_fusion_cohort(
    depth_stack: np.ndarray,
    confidence_stack: np.ndarray,
    mask_stack: np.ndarray,
    *,
    min_confidence: float = 0.1,
    min_observations: int = 3,
    support_ratio: float = 0.5,
    depth_agreement_m: float = 0.18,
    normalize_frame_scale: bool = True,
    scale_deadband: float = 0.05,
    scale_factor_min: float = 0.50,
    scale_factor_max: float = 2.00,
    confidence_cap_percentile: float = 98.0,
    confidence_weight_floor: float = 0.05,
) -> dict[str, Any]:
    """Evaluate robust fusion reducers without writing or changing runtime state."""

    started = time.perf_counter()
    depth = np.asarray(depth_stack, dtype=np.float32)
    confidence = np.asarray(confidence_stack, dtype=np.float32)
    mask = np.asarray(mask_stack, dtype=bool)
    if (
        depth.ndim != 3
        or confidence.shape != depth.shape
        or mask.shape != depth.shape
        or depth.shape[0] < 1
    ):
        raise ValueError("depth, confidence, and mask must be same-shape [N,H,W]")
    if not math.isfinite(min_confidence):
        raise ValueError("min_confidence must be finite")
    if not 0.0 < support_ratio <= 1.0:
        raise ValueError("support_ratio must be in (0, 1]")
    if not math.isfinite(depth_agreement_m) or depth_agreement_m <= 0.0:
        raise ValueError("depth_agreement_m must be finite and positive")
    if (
        not math.isfinite(scale_factor_min)
        or not math.isfinite(scale_factor_max)
        or scale_factor_min <= 0.0
        or scale_factor_max < scale_factor_min
    ):
        raise ValueError("scale factor bounds must be finite, positive, and ordered")
    if not 0.0 < confidence_cap_percentile <= 100.0:
        raise ValueError("confidence_cap_percentile must be in (0, 100]")
    if (
        not math.isfinite(confidence_weight_floor)
        or not 0.0 <= confidence_weight_floor <= 1.0
    ):
        raise ValueError("confidence_weight_floor must be in [0, 1]")

    source_valid = (
        mask
        & np.isfinite(depth)
        & (depth > 0.0)
        & np.isfinite(confidence)
        & (confidence >= float(min_confidence))
    )
    frame_medians = np.full(depth.shape[0], np.nan, dtype=np.float64)
    for index in range(depth.shape[0]):
        values = depth[index][source_valid[index]]
        if values.size:
            frame_medians[index] = float(np.median(values))
    usable_medians = np.isfinite(frame_medians) & (frame_medians > 0.0)
    scale_baseline = (
        float(np.median(frame_medians[usable_medians]))
        if np.any(usable_medians)
        else float("nan")
    )
    scale_factors = np.ones(depth.shape[0], dtype=np.float32)
    proposed_scale_factors = np.ones(depth.shape[0], dtype=np.float64)
    scale_applied = np.zeros(depth.shape[0], dtype=bool)
    scale_rejected = np.zeros(depth.shape[0], dtype=bool)
    if normalize_frame_scale and math.isfinite(scale_baseline) and scale_baseline > 0.0:
        relative_change = np.zeros(depth.shape[0], dtype=np.float64)
        relative_change[usable_medians] = (
            np.abs(frame_medians[usable_medians] - scale_baseline) / scale_baseline
        )
        scale_candidates = usable_medians & (
            relative_change >= float(scale_deadband)
        )
        proposed_scale_factors[scale_candidates] = (
            scale_baseline / frame_medians[scale_candidates]
        )
        scale_rejected = scale_candidates & (
            (proposed_scale_factors < float(scale_factor_min))
            | (proposed_scale_factors > float(scale_factor_max))
        )
        scale_applied = scale_candidates & ~scale_rejected
        scale_factors[scale_applied] = (
            proposed_scale_factors[scale_applied]
        ).astype(np.float32)
        source_valid[scale_rejected, :, :] = False
    evaluated_depth = depth * scale_factors[:, None, None]

    masked_depth = np.ma.array(evaluated_depth, mask=~source_valid)
    temporal_median = (
        np.ma.median(masked_depth, axis=0).filled(np.nan).astype(np.float32)
    )
    tolerance = float(depth_agreement_m) + np.nan_to_num(
        temporal_median,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ) * 0.025
    agreeing = (
        source_valid
        & np.isfinite(temporal_median)[None, :, :]
        & (
            np.abs(evaluated_depth - temporal_median[None, :, :])
            <= tolerance[None, :, :]
        )
    )
    support = np.count_nonzero(agreeing, axis=0)
    cohort_size = int(depth.shape[0])
    strict_majority_support = (cohort_size // 2) + 1
    required_support = max(
        1,
        min(
            cohort_size,
            max(
                int(min_observations),
                min(3, cohort_size),
                int(math.ceil(cohort_size * support_ratio)),
                strict_majority_support,
            ),
        ),
    )
    fused_valid = support >= required_support
    eligible = np.count_nonzero(source_valid, axis=0) >= required_support

    confidence_values = confidence[source_valid]
    confidence_p02, confidence_p50, confidence_p98, confidence_p995 = (
        _finite_percentiles(confidence_values, (2.0, 50.0, 98.0, 99.5))
    )
    cap = float(confidence_p98) if confidence_p98 is not None else 1.0
    if confidence_p50 is not None and confidence_p50 > 0.0:
        cap = min(cap, float(confidence_p50) * 4.0)
    cap = max(cap, 1e-6)

    frame_confidence_caps = np.ones(cohort_size, dtype=np.float64)
    runtime_normalized_confidence = np.zeros_like(confidence, dtype=np.float32)
    for index in range(cohort_size):
        frame_values = confidence[index][source_valid[index]]
        if frame_values.size:
            frame_cap = float(
                np.percentile(frame_values, confidence_cap_percentile)
            )
            if math.isfinite(frame_cap) and frame_cap > 0.0:
                frame_confidence_caps[index] = frame_cap
        normalized = np.divide(
            confidence[index],
            frame_confidence_caps[index],
            out=np.zeros_like(confidence[index], dtype=np.float32),
            where=np.isfinite(confidence[index]),
        )
        runtime_normalized_confidence[index] = np.where(
            source_valid[index],
            np.clip(normalized, confidence_weight_floor, 1.0),
            0.0,
        )

    def weighted_candidate(
        weight_cap: float | None,
        *,
        weight_values: np.ndarray | None = None,
    ) -> np.ndarray:
        if weight_values is None:
            weight_values = np.maximum(confidence, 0.0)
            if weight_cap is not None:
                weight_values = np.minimum(weight_values, float(weight_cap))
        weights = np.where(agreeing, weight_values, 0.0)
        weight_sum = np.sum(weights, axis=0, dtype=np.float64)
        depth_sum = np.sum(
            np.where(agreeing, evaluated_depth, 0.0) * weights,
            axis=0,
            dtype=np.float64,
        )
        candidate = np.zeros(depth.shape[1:], dtype=np.float32)
        np.divide(depth_sum, weight_sum, out=candidate, where=weight_sum > 0.0)
        candidate[~fused_valid] = 0.0
        return candidate

    median_candidate = np.where(
        fused_valid & np.isfinite(temporal_median),
        temporal_median,
        0.0,
    ).astype(np.float32)
    agreeing_sum = np.sum(
        np.where(agreeing, evaluated_depth, 0.0),
        axis=0,
        dtype=np.float64,
    )
    agreeing_min = np.min(
        np.where(agreeing, evaluated_depth, np.inf),
        axis=0,
    )
    agreeing_max = np.max(
        np.where(agreeing, evaluated_depth, -np.inf),
        axis=0,
    )
    trimmed_candidate = median_candidate.copy()
    trim_valid = fused_valid & (support >= 3)
    trimmed_values = np.zeros(depth.shape[1:], dtype=np.float32)
    trimmed_numerator = (
        agreeing_sum
        - np.where(support > 0, agreeing_min, 0.0)
        - np.where(support > 0, agreeing_max, 0.0)
    )
    np.divide(
        trimmed_numerator,
        support - 2,
        out=trimmed_values,
        where=support > 2,
    )
    trimmed_candidate[trim_valid] = trimmed_values[trim_valid]

    raw_weighted = weighted_candidate(None)
    capped_weighted = weighted_candidate(cap)
    runtime_bounded_weighted = weighted_candidate(
        None,
        weight_values=runtime_normalized_confidence,
    )
    reference_residual = np.abs(
        evaluated_depth - temporal_median[None, :, :]
    )
    reference_residual_p50, reference_residual_p95 = _finite_percentiles(
        reference_residual[source_valid],
        (50.0, 95.0),
    )
    confidence_error_correlation = _pearson_sampled(
        np.log1p(np.maximum(confidence[source_valid], 0.0)),
        reference_residual[source_valid],
    )
    support_histogram = np.bincount(
        support.reshape(-1),
        minlength=cohort_size + 1,
    )
    support_values = support[fused_valid]
    support_p10, support_p50, support_p90 = _finite_percentiles(
        support_values,
        (10.0, 50.0, 90.0),
    )
    return {
        "contract": "noesis.mapanything.fusion_quality_report",
        "contract_version": 1,
        "cohort": {
            "frame_count": cohort_size,
            "shape": [int(depth.shape[1]), int(depth.shape[2])],
            "required_support": int(required_support),
            "strict_majority_support": int(strict_majority_support),
            "min_confidence": float(min_confidence),
            "support_ratio": float(support_ratio),
            "depth_agreement_m": float(depth_agreement_m),
        },
        "scale_normalization": {
            "enabled": bool(normalize_frame_scale),
            "deadband": float(scale_deadband),
            "accepted_factor_bounds": [
                float(scale_factor_min),
                float(scale_factor_max),
            ],
            "baseline_median_depth": (
                float(scale_baseline) if math.isfinite(scale_baseline) else None
            ),
            "frame_medians": [
                float(value) if math.isfinite(value) else None
                for value in frame_medians
            ],
            "proposed_factors": [
                float(value) if math.isfinite(value) else None
                for value in proposed_scale_factors
            ],
            "factors": [float(value) for value in scale_factors],
            "applied": [bool(value) for value in scale_applied],
            "rejected": [bool(value) for value in scale_rejected],
            "rejection_policy": "quarantine_entire_frame",
        },
        "support": {
            "eligible_fraction": float(np.count_nonzero(eligible) / eligible.size),
            "retained_full_frame_fraction": float(
                np.count_nonzero(fused_valid) / fused_valid.size
            ),
            "retained_over_eligible_fraction": float(
                np.count_nonzero(fused_valid)
                / max(1, np.count_nonzero(eligible))
            ),
            "histogram": {
                str(index): int(value)
                for index, value in enumerate(support_histogram.tolist())
            },
            "retained_p10": support_p10,
            "retained_p50": support_p50,
            "retained_p90": support_p90,
            "quarantined_frame_count": int(np.count_nonzero(scale_rejected)),
            "quarantined_frame_indices": [
                int(index)
                for index, rejected in enumerate(scale_rejected)
                if rejected
            ],
            "component_evidence": _binary_mask_component_evidence(fused_valid),
        },
        "confidence": {
            "p02": confidence_p02,
            "p50": confidence_p50,
            "p98": confidence_p98,
            "p995": confidence_p995,
            "robust_cap": cap,
            "runtime_cap_percentile": float(confidence_cap_percentile),
            "runtime_weight_floor": float(confidence_weight_floor),
            "runtime_frame_caps": [
                float(value) for value in frame_confidence_caps
            ],
            "log_confidence_vs_temporal_error_pearson": confidence_error_correlation,
        },
        "temporal_consensus_reference": {
            "definition": "absolute_source_residual_to_temporal_median",
            "residual_p50_m": reference_residual_p50,
            "residual_p95_m": reference_residual_p95,
        },
        "candidates": {
            "confidence_weighted_raw": _candidate_metrics(
                raw_weighted,
                fused_valid,
                evaluated_depth,
                source_valid,
            ),
            "confidence_weighted_robust_capped": _candidate_metrics(
                capped_weighted,
                fused_valid,
                evaluated_depth,
                source_valid,
            ),
            "confidence_weighted_runtime_bounded": _candidate_metrics(
                runtime_bounded_weighted,
                fused_valid,
                evaluated_depth,
                source_valid,
            ),
            "temporal_median": _candidate_metrics(
                median_candidate,
                fused_valid,
                evaluated_depth,
                source_valid,
            ),
            "one_sample_trimmed_mean": _candidate_metrics(
                trimmed_candidate,
                fused_valid,
                evaluated_depth,
                source_valid,
            ),
        },
        "elapsed_ms": float((time.perf_counter() - started) * 1000.0),
    }


def _attrs(group: Any) -> dict[str, Any]:
    attrs = group.attrs
    return dict(attrs.asdict() if hasattr(attrs, "asdict") else dict(attrs))


def load_depth_snapshot(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group = zarr.open_group(str(path), mode="r")
    depth_key = "depth" if "depth" in group else "depth_z"
    if depth_key not in group or "conf" not in group or "mask" not in group:
        raise ValueError(f"snapshot lacks depth/conf/mask datasets: {path}")
    return (
        np.asarray(group[depth_key], dtype=np.float32),
        np.asarray(group["conf"], dtype=np.float32),
        np.asarray(group["mask"], dtype=np.uint8),
    )


def resolve_fused_source_paths(path: Path) -> list[Path]:
    group = zarr.open_group(str(path), mode="r")
    attrs = _attrs(group)
    rows: object = attrs.get("source_snapshots")
    if isinstance(rows, str):
        rows = json.loads(rows)
    if not isinstance(rows, list):
        fusion_meta = attrs.get("fusion_meta")
        if isinstance(fusion_meta, str):
            fusion_meta = json.loads(fusion_meta)
        rows = (
            fusion_meta.get("source_snapshots")
            if isinstance(fusion_meta, Mapping)
            else None
        )
    if not isinstance(rows, list) or not rows:
        raise ValueError("fused snapshot has no source_snapshots evidence")
    references = [
        str(row.get("storage_ref") or "").strip()
        for row in rows
        if isinstance(row, Mapping)
    ]
    if len(references) != len(rows) or any(not reference for reference in references):
        raise ValueError("fused snapshot source evidence is incomplete")
    resolved: list[Path] = []
    for reference in references:
        relative = Path(reference)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"non-portable source snapshot reference: {reference}")
        candidate = next(
            (
                ancestor / relative
                for ancestor in path.parents
                if (ancestor / relative).is_dir()
            ),
            None,
        )
        if candidate is None:
            raise FileNotFoundError(
                f"source snapshot was pruned or is unavailable: {reference}"
            )
        resolved.append(candidate)
    return resolved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MapAnything fusion reducers on an immutable raw cohort."
    )
    parser.add_argument(
        "--snapshot",
        action="append",
        default=[],
        type=Path,
        help="Raw Zarr snapshot path; repeat for every frame.",
    )
    parser.add_argument(
        "--fused-snapshot",
        type=Path,
        help="Resolve the exact raw cohort from a fused snapshot's evidence.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.1)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--support-ratio", type=float, default=0.5)
    parser.add_argument("--depth-agreement-m", type=float, default=0.18)
    parser.add_argument(
        "--without-scale-normalization",
        action="store_true",
        help="Evaluate raw frame scale instead of median-normalized frame scale.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = [
        path if path.is_absolute() else (Path.cwd() / path)
        for path in args.snapshot
    ]
    if args.fused_snapshot is not None:
        fused_path = (
            args.fused_snapshot
            if args.fused_snapshot.is_absolute()
            else Path.cwd() / args.fused_snapshot
        )
        paths.extend(resolve_fused_source_paths(fused_path))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise SystemExit("provide --snapshot or --fused-snapshot")
    loaded = [load_depth_snapshot(path) for path in paths]
    shapes = {tuple(depth.shape) for depth, _confidence, _mask in loaded}
    if len(shapes) != 1:
        raise SystemExit(f"cohort shape mismatch: {sorted(shapes)}")
    report = evaluate_fusion_cohort(
        np.stack([row[0] for row in loaded]),
        np.stack([row[1] for row in loaded]),
        np.stack([row[2] for row in loaded]),
        min_confidence=args.min_confidence,
        min_observations=args.min_observations,
        support_ratio=args.support_ratio,
        depth_agreement_m=args.depth_agreement_m,
        normalize_frame_scale=not args.without_scale_normalization,
    )
    report["sources"] = [str(path) for path in paths]
    encoded = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
