"""Bounded offline registration review; never fits or publishes a mapping.

Run ``python3 -m noesis.calibration.depth_registration_review --help``.
Qualification samples use ``noesis.depth_registration.review_samples.v1``:
the document binds camera_id, registration_id, source_space and target_space,
and contains a ``samples`` list. Each row supplies sample_id, partition
(fit/holdout), capture_id, recording_id, source_frame_id, raw_timestamp_us,
reference_timestamp_us, raw_depth_m, target_range_m, anchor_support_count,
anchor_id, reference_anchor_id, reference_kind=independent_same_anchor_range,
reference_evidence_ref, and reference_evidence_sha256. The reference is an
independently established range in TARGET_SPACE for that exact observed anchor,
not an output of the mapping, person-world fusion, or the phone camera center.

Provenance fields are supplied attestations, not verification of external asset
contents or physical accuracy. Passing means review-ready under the reported
offline policy only; it never admits a runtime artifact. Existing tracking CSV
or companion tracking NDJSON is diagnostic input, never qualification labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from noesis.calibration.depth_registration import (
    MAX_DEPTH_REGISTRATION_BYTES,
    MAX_OCCUPIED_ABS_ERROR_M,
    MAX_OCCUPIED_ANCHOR_TIME_DELTA_MS,
    MAX_OCCUPIED_MEDIAN_ABS_ERROR_M,
    MAX_OCCUPIED_P95_ABS_ERROR_M,
    MIN_OCCUPIED_ANCHOR_SUPPORT,
    MIN_OCCUPIED_CAPTURES_PER_PARTITION,
    MIN_OCCUPIED_FIT_OBSERVATIONS,
    MIN_OCCUPIED_HOLDOUT_OBSERVATIONS,
    SOURCE_SPACE,
    TARGET_SPACE,
    DepthRegistrationEntry,
    DepthRegistrationError,
)
from noesis_core.strict_json import strict_json_loads

SAMPLE_CONTRACT = "noesis.depth_registration.review_samples.v1"
REPORT_CONTRACT = "noesis.depth_registration.offline_review.v1"
MAX_INPUT_BYTES = MAX_DEPTH_REGISTRATION_BYTES
MAX_ROWS = 100_000
_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ReviewPolicy:
    """Offline coverage policy; residual/count minima reuse runtime contracts."""

    min_fit_samples: int = MIN_OCCUPIED_FIT_OBSERVATIONS
    min_holdout_samples: int = MIN_OCCUPIED_HOLDOUT_OBSERVATIONS
    min_captures_per_partition: int = MIN_OCCUPIED_CAPTURES_PER_PARTITION
    min_anchor_support: int = MIN_OCCUPIED_ANCHOR_SUPPORT
    max_timestamp_delta_ms: float = MAX_OCCUPIED_ANCHOR_TIME_DELTA_MS
    max_median_abs_error_m: float = MAX_OCCUPIED_MEDIAN_ABS_ERROR_M
    max_p95_abs_error_m: float = MAX_OCCUPIED_P95_ABS_ERROR_M
    max_abs_error_m: float = MAX_OCCUPIED_ABS_ERROR_M
    min_usable_fraction: float = 0.8
    min_holdout_domain_span_fraction: float = 0.8
    range_bins: int = 4
    min_capture_span_us: int = 1_000_000
    min_distinct_frames_per_capture: int = 8

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"policy {key} must be finite and positive")
        for key in ("min_fit_samples", "min_holdout_samples", "min_captures_per_partition",
                    "min_anchor_support", "range_bins", "min_capture_span_us",
                    "min_distinct_frames_per_capture"):
            if not isinstance(getattr(self, key), int):
                raise ValueError(f"policy {key} must be an integer")
        if self.min_usable_fraction > 1 or self.min_holdout_domain_span_fraction > 1:
            raise ValueError("coverage fractions cannot exceed one")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 2**63 - 1


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 2048


def _range(values: Sequence[float]) -> list[float] | None:
    return [float(min(values)), float(max(values))] if values else None


def _residuals(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median_abs_error_m": None, "p95_abs_error_m": None,
                "max_abs_error_m": None, "mean_signed_error_m": None}
    errors = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "median_abs_error_m": float(np.median(abs(errors))),
            "p95_abs_error_m": float(np.percentile(abs(errors), 95)),
            "max_abs_error_m": float(max(abs(errors))),
            "mean_signed_error_m": float(np.mean(errors))}


def _status(entry: DepthRegistrationEntry, raw: Any) -> tuple[str, float | None]:
    depth = _number(raw)
    if depth is None or depth <= 0:
        return "missing_or_invalid_raw_depth", None
    registered = entry.apply(depth)
    if registered is None:
        return "out_of_domain", None
    if not entry.has_observable_local_slope(depth):
        return "unobservable_plateau", registered
    return "usable", registered


def mapping_review(entry: DepthRegistrationEntry) -> dict[str, Any]:
    """Use runtime slope decisions, including its flat-interval tolerance."""
    if not all(math.isfinite(value) and value > 0 for value in entry.raw_range_domain_m):
        raise ValueError("registration domain must be finite and positive")
    if entry.raw_range_domain_m[1] <= entry.raw_range_domain_m[0]:
        raise ValueError("registration domain must have positive width")
    intervals = []
    total_span = entry.raw_range_domain_m[1] - entry.raw_range_domain_m[0]
    observable_span = 0.0
    for left, right, target_left, target_right in zip(
        entry.knots_raw_m[:-1], entry.knots_raw_m[1:],
        entry.knots_registered_m[:-1], entry.knots_registered_m[1:],
    ):
        observable = entry.has_observable_local_slope((left + right) / 2)
        span = max(0.0, min(right, entry.raw_range_domain_m[1]) -
                   max(left, entry.raw_range_domain_m[0]))
        observable_span += span if observable else 0.0
        intervals.append({"raw_range_m": [left, right],
                          "registered_range_m": [target_left, target_right],
                          "slope": (target_right - target_left) / (right - left),
                          "observable": observable})
    return {"raw_domain_m": list(entry.raw_range_domain_m), "intervals": intervals,
            "observable_domain_fraction": observable_span / total_span if total_span > 0 else 0.0,
            "historical_fit_metrics": dict(entry.fit_metrics),
            "historical_occupied_validation": dict(entry.occupied_anchor_validation),
            "historical_summary_is_current_qualification": False}


def _sample_reasons(row: Mapping[str, Any], policy: ReviewPolicy) -> list[str]:
    reasons = []
    if row.get("partition") not in ("fit", "holdout"):
        reasons.append("partition_invalid")
    for key in ("sample_id", "capture_id", "recording_id", "anchor_id", "reference_anchor_id"):
        if not _text(row.get(key)):
            reasons.append(key + "_missing_or_invalid")
    if row.get("reference_kind") != "independent_same_anchor_range":
        reasons.append("independent_reference_label_unsupported")
    if not _text(row.get("reference_evidence_ref")) or not _DIGEST.fullmatch(
        str(row.get("reference_evidence_sha256", ""))
    ):
        reasons.append("independent_reference_provenance_missing")
    if row.get("anchor_id") != row.get("reference_anchor_id"):
        reasons.append("reference_anchor_mismatch")
    for key in ("raw_depth_m", "target_range_m"):
        value = row.get(key)
        number = _number(value)
        if not isinstance(value, (int, float)) or number is None or number <= 0:
            reasons.append(key + "_invalid")
    for key in ("source_frame_id", "raw_timestamp_us", "reference_timestamp_us"):
        if not _integer(row.get(key)):
            reasons.append(key + "_invalid")
    if _integer(row.get("raw_timestamp_us")) and _integer(row.get("reference_timestamp_us")):
        if abs(row["raw_timestamp_us"] - row["reference_timestamp_us"]) / 1000 > policy.max_timestamp_delta_ms:
            reasons.append("anchor_timestamp_mismatch")
    if not _integer(row.get("anchor_support_count")) or row["anchor_support_count"] < policy.min_anchor_support:
        reasons.append("anchor_support_insufficient")
    return reasons


def qualification_review(
    entry: DepthRegistrationEntry, evidence: Mapping[str, Any] | None,
    *, policy: ReviewPolicy = ReviewPolicy(),
) -> dict[str, Any]:
    reasons: list[str] = []
    if evidence is None:
        return {"ready": False, "reasons": ["independent_qualification_samples_missing"],
                "partitions": {}, "rejected_sample_count": 0}
    for key, expected in (("contract", SAMPLE_CONTRACT), ("camera_id", entry.camera_id),
                          ("registration_id", entry.registration_id),
                          ("source_space", SOURCE_SPACE), ("target_space", TARGET_SPACE)):
        if evidence.get(key) != expected:
            reasons.append(key + "_mismatch")
    rows = evidence.get("samples")
    if not isinstance(rows, list) or len(rows) > MAX_ROWS:
        raise ValueError(f"samples must be a list with at most {MAX_ROWS} rows")
    accepted: dict[str, list[Mapping[str, Any]]] = {"fit": [], "holdout": []}
    rejected = Counter()
    sample_ids: set[str] = set()
    frame_keys: set[tuple[Any, ...]] = set()
    for row in rows:
        errors = _sample_reasons(row, policy) if isinstance(row, Mapping) else ["sample_not_mapping"]
        if not errors:
            frame_key = (row["recording_id"], row["source_frame_id"], row["anchor_id"])
            if row["sample_id"] in sample_ids or frame_key in frame_keys:
                errors.append("duplicate_sample_or_source_anchor")
            sample_ids.add(row["sample_id"])
            frame_keys.add(frame_key)
        if errors:
            rejected.update(errors)
        else:
            accepted[row["partition"]].append(row)
    if rejected:
        reasons.append("sample_provenance_or_identity_rejected")
    partitions = {}
    lo, hi = entry.raw_range_domain_m
    for partition, selected in accepted.items():
        captures: dict[str, list[int]] = defaultdict(list)
        capture_frames: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
        counts: Counter[str] = Counter()
        residuals, usable_residuals, bins = [], [], set()
        for row in selected:
            captures[row["capture_id"]].append(row["raw_timestamp_us"])
            capture_frames[row["capture_id"]].add((row["recording_id"], row["source_frame_id"]))
            status, registered = _status(entry, row["raw_depth_m"])
            counts[status] += 1
            if registered is not None:
                residuals.append(registered - row["target_range_m"])
                if status == "usable":
                    usable_residuals.append(registered - row["target_range_m"])
                bins.add(min(policy.range_bins - 1, int(
                    (row["raw_depth_m"] - lo) / (hi - lo) * policy.range_bins)))
        raw_range = _range([row["raw_depth_m"] for row in selected])
        span = (max(0.0, min(hi, raw_range[1]) - max(lo, raw_range[0])) / (hi - lo)) if raw_range else 0.0
        capture_spans = {key: {"timestamp_range_us": [min(times), max(times)], "span_us": max(times) - min(times),
                               "sample_count": len(times), "distinct_frame_count": len(capture_frames[key])}
                         for key, times in captures.items()}
        partitions[partition] = {
            "sample_count": len(selected), "capture_count": len(captures),
            "captures": capture_spans, "raw_range_m": raw_range,
            "target_range_m": _range([row["target_range_m"] for row in selected]),
            "domain_span_fraction": span, "occupied_range_bins": sorted(bins),
            "status_counts": dict(counts), "usable_fraction": counts["usable"] / len(selected) if selected else 0.0,
            "mapped_residuals": _residuals(residuals), "usable_residuals": _residuals(usable_residuals),
            "max_timestamp_delta_ms": max((abs(row["raw_timestamp_us"] - row["reference_timestamp_us"]) / 1000 for row in selected), default=None),
        }
        minimum = policy.min_fit_samples if partition == "fit" else policy.min_holdout_samples
        if len(selected) < minimum:
            reasons.append(partition + "_samples_insufficient")
        if len(captures) < policy.min_captures_per_partition:
            reasons.append(partition + "_captures_insufficient")
        if any(value["span_us"] < policy.min_capture_span_us for value in capture_spans.values()):
            reasons.append(partition + "_capture_time_coverage_insufficient")
        if any(value["distinct_frame_count"] < policy.min_distinct_frames_per_capture for value in capture_spans.values()):
            reasons.append(partition + "_capture_frame_coverage_insufficient")
    fit_captures = set(partitions["fit"]["captures"])
    if fit_captures & set(partitions["holdout"]["captures"]):
        reasons.append("fit_holdout_capture_leakage")
    # Different capture names cannot conceal overlapping source-recording time.
    recording_times: dict[tuple[str, str], list[int]] = defaultdict(list)
    for partition, selected in accepted.items():
        for row in selected:
            recording_times[(row["recording_id"], partition)].append(row["raw_timestamp_us"])
    for recording, partition in recording_times:
        fit = recording_times.get((recording, "fit"), [])
        holdout = recording_times.get((recording, "holdout"), [])
        if partition == "fit" and fit and holdout and max(min(fit), min(holdout)) <= min(max(fit), max(holdout)):
            reasons.append("fit_holdout_recording_time_overlap")
    holdout = partitions["holdout"]
    fit_range, holdout_range = partitions["fit"]["raw_range_m"], holdout["raw_range_m"]
    if fit_range and holdout_range and (holdout_range[0] < fit_range[0] or holdout_range[1] > fit_range[1]):
        reasons.append("holdout_requires_extrapolation_from_fit")
    if partitions["fit"]["status_counts"].get("out_of_domain", 0):
        reasons.append("fit_samples_outside_mapping_domain")
    if holdout["domain_span_fraction"] < policy.min_holdout_domain_span_fraction or len(holdout["occupied_range_bins"]) < policy.range_bins:
        reasons.append("holdout_range_coverage_insufficient")
    if holdout["usable_fraction"] < policy.min_usable_fraction:
        reasons.append("holdout_usable_fraction_insufficient")
    errors = holdout["mapped_residuals"]
    for key, maximum in (("median_abs_error_m", policy.max_median_abs_error_m),
                         ("p95_abs_error_m", policy.max_p95_abs_error_m),
                         ("max_abs_error_m", policy.max_abs_error_m)):
        if errors[key] is None or errors[key] > maximum:
            reasons.append("holdout_" + key + "_unqualified")
    return {"ready": not reasons, "reasons": sorted(set(reasons)), "partitions": partitions,
            "input_sample_count": len(rows), "rejected_sample_count": len(rows) - sum(map(len, accepted.values())),
            "rejection_counts": dict(rejected),
            "reference_evidence": [{"ref": ref, "sha256": digest} for ref, digest in sorted({
                (row["reference_evidence_ref"], row["reference_evidence_sha256"])
                for selected in accepted.values() for row in selected})],
            "provenance_status": "supplied_attestations_external_content_not_verified"}


def tracking_review(entry: DepthRegistrationEntry, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    with_status: Counter[str] = Counter()
    accepted_contributions = 0
    for row in rows:
        status, _ = _status(entry, row.get("depth_anchor_m"))
        counts[status] += 1
        if row.get("depth_registration_status"):
            with_status[status] += 1
        if row.get("world_measurement_accepted") in (True, "True", "true") and row.get("depth_registration_status") == "ok":
            # A valid registration can still lose to a floor-only resolver.
            selected = row.get("world_resolver_selected_id", "")
            if row.get("world_source") in {"depth", "pose_depth", "pose_depth_fused", "depth_registered"} or "depth" in str(selected):
                accepted_contributions += 1
    return {"row_count": len(rows), "raw_anchor_status_counts": dict(counts),
            "rows_with_recorded_registration_status": sum(with_status.values()),
            "recorded_registration_rows_current_status_counts": dict(with_status),
            "raw_mapping_usable_fraction_all_rows": counts["usable"] / len(rows) if rows else None,
            "status_scope": "Mapping domain/slope only; raw-anchor rows may still fail contact, cache-age, source-time, or world-admission gates.",
            "accepted_depth_contribution_rows": accepted_contributions,
            "qualification_label_status": "unsupported_independent_reference_labels",
            "reason": "Tracking world outputs and phone camera centers are not independent same-anchor range labels."}


def review_registration(entry: DepthRegistrationEntry, *, evidence: Mapping[str, Any] | None = None,
                        tracking_rows: Sequence[Mapping[str, Any]] = (), policy: ReviewPolicy = ReviewPolicy()) -> dict[str, Any]:
    if len(tracking_rows) > MAX_ROWS:
        raise ValueError(f"tracking rows exceed {MAX_ROWS}")
    mapping = mapping_review(entry)
    qualification = qualification_review(entry, evidence, policy=policy)
    reasons = list(qualification["reasons"])
    if mapping["observable_domain_fraction"] < policy.min_usable_fraction:
        reasons.append("mapping_observable_domain_fraction_insufficient")
    return {"contract": REPORT_CONTRACT, "camera_id": entry.camera_id,
            "registration_id": entry.registration_id, "qualification_ready": not reasons,
            "readiness_scope": "offline_review_only_no_runtime_admission", "reasons": reasons,
            "policy": asdict(policy), "mapping": mapping, "qualification": qualification,
            "tracking_diagnostics": tracking_review(entry, tracking_rows),
            "limitations": ["No fitting, model execution, mapping replacement, or runtime changes.",
                            "Reference provenance is attested by input; source contents and surveyed accuracy are not independently verified."]}


def _read(path: Path) -> tuple[str, dict[str, Any]]:
    with path.open("rb") as handle:
        content = handle.read(MAX_INPUT_BYTES + 1)
    if len(content) > MAX_INPUT_BYTES:
        raise ValueError(f"input exceeds {MAX_INPUT_BYTES} bytes: {path}")
    return content.decode("utf-8"), {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}


def _tracking_rows(path: Path, camera_id: str, tracker_id: int | None) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    content, source = _read(path)
    rows = []
    inputs = csv.DictReader(io.StringIO(content)) if path.suffix.lower() == ".csv" else content.splitlines()
    for index, item in enumerate(inputs):
        if index >= MAX_ROWS:
            raise ValueError(f"tracking input exceeds {MAX_ROWS} records")
        if isinstance(item, str):
            message = strict_json_loads(item, label=f"{path}:{index + 1}")
            if not isinstance(message, Mapping):
                raise ValueError("companion record must be a mapping")
            if message.get("message_type") != "tracking":
                continue
            payload = message.get("message")
            if not isinstance(payload, Mapping) or not isinstance(payload.get("tracks"), list):
                raise ValueError("companion tracking record must contain a tracks list")
            candidates = payload["tracks"]
        else:
            candidates = [item]
        for row in candidates:
            if not isinstance(row, Mapping):
                raise ValueError("tracking row must be a mapping")
            if row.get("camera_id") != camera_id:
                continue
            if tracker_id is not None and str(row.get("tracker_id")) != str(tracker_id):
                continue
            rows.append(row)
            if len(rows) > MAX_ROWS:
                raise ValueError(f"tracking input exceeds {MAX_ROWS} selected rows")
    return rows, source


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--samples", type=Path, help="Independent review_samples.v1 JSON; never inferred from telemetry")
    parser.add_argument("--tracking", type=Path, help="Diagnostic-only tracking CSV or companion NDJSON")
    parser.add_argument("--tracker-id", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() in {path.resolve() for path in (args.registration, args.samples, args.tracking) if path}:
            raise ValueError("review output must not overwrite an input artifact")
        text, registration_source = _read(args.registration)
        bundle = strict_json_loads(text, label=str(args.registration))
        if not isinstance(bundle, Mapping) or not isinstance(bundle.get("cameras"), Mapping):
            raise ValueError("registration must contain a cameras mapping")
        entry = DepthRegistrationEntry.from_dict(bundle["cameras"][args.camera])
        if entry.camera_id != args.camera:
            raise ValueError("registration camera identity does not match selected key")
        sources = {"registration": registration_source}
        evidence = None
        if args.samples:
            text, sources["samples"] = _read(args.samples)
            evidence = strict_json_loads(text, label=str(args.samples))
            if not isinstance(evidence, Mapping):
                raise ValueError("sample document must be a mapping")
        tracking = []
        if args.tracking:
            tracking, sources["tracking"] = _tracking_rows(args.tracking, args.camera, args.tracker_id)
        report = review_registration(entry, evidence=evidence, tracking_rows=tracking)
        report["input_evidence"] = sources
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (ValueError, TypeError, KeyError, OSError, csv.Error, DepthRegistrationError) as exc:
        parser.exit(2, f"review input error: {exc}\n")
    print(json.dumps({"output": str(args.output), "qualification_ready": report["qualification_ready"], "reasons": report["reasons"]}))
    return 0 if report["qualification_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
