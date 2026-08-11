#!/usr/bin/env python3
"""Audit what cadence/fusion conclusions an immutable depth cohort supports.

This tool deliberately separates an observed cadence from window/tolerance
sensitivity.  A cohort produced at one nvinfer interval cannot supply the
frames that a shorter interval would have inferred, so the report never
declares a cadence winner unless every requested cadence has its own sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.mapanything_fusion_quality_report import (  # noqa: E402
    evaluate_fusion_cohort,
    load_depth_snapshot,
)
from geometry.depth_source import DepthStorageManager  # noqa: E402

CONTRACT = "noesis.mapanything.cadence_fusion_offline_report"
CONTRACT_VERSION = 1
SOURCE_REPORT_CONTRACT = "noesis.mapanything.fusion_quality_report"
RUNTIME_CANDIDATE = "confidence_weighted_runtime_bounded"
SOURCE_FRAME_CONTRACT = "noesis.mapanything.source_frame.v1"
FIXED_SOURCE_FPS = 30.0
FIXED_OBSERVED_INTERVAL_FRAMES = 89
FIXED_COHORT_FRAME_COUNT = 6
FIXED_WINDOW_SIZES = (4, 5)


@dataclass(frozen=True)
class SourceDescriptor:
    path: Path
    camera_id: str
    timestamp_us: int
    sequence: int
    write_id: str
    manifest_file_sha256: str
    component_sha256s: Mapping[str, str]
    source_frame_contract: str | None = None
    source_frame_number: int | None = None
    source_media_pts_ns: int | None = None
    storage_timestamp_basis: str | None = None


def _strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _required_text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _load_descriptor(path: Path, expected_camera: str) -> SourceDescriptor:
    resolved = path.resolve(strict=True)
    try:
        manifest = DepthStorageManager._read_and_validate_commit_manifest(
            resolved
        )
    except Exception as exc:
        raise ValueError(
            f"source snapshot commit validation failed: {resolved}"
        ) from exc
    if int(manifest.get("version") or 0) != 2:
        raise ValueError(f"source snapshot must use commit manifest v2: {resolved}")

    group = zarr.open_group(str(resolved), mode="r")
    attrs = dict(group.attrs.asdict())
    camera_id = _required_text(attrs.get("camera_id"), "camera_id")
    if camera_id != expected_camera:
        raise ValueError(
            f"source camera mismatch: expected {expected_camera}, got {camera_id}"
        )
    if str(attrs.get("snapshot_role") or "").strip() or str(
        attrs.get("fusion_level") or ""
    ).strip():
        raise ValueError(f"derived snapshot cannot be an audit source: {resolved}")

    timestamp_us = int(attrs.get("timestamp_us"))
    sequence = int(attrs.get("sequence"))
    write_id = _required_text(attrs.get("write_id"), "write_id")
    if timestamp_us <= 0 or sequence <= 0:
        raise ValueError(f"invalid source timestamp/sequence: {resolved}")

    if (
        manifest.get("state") != "committed"
        or str(manifest.get("camera_id") or "") != camera_id
        or int(manifest.get("timestamp_us") or 0) != timestamp_us
        or int(manifest.get("sequence") or 0) != sequence
        or str(manifest.get("write_id") or "") != write_id
    ):
        raise ValueError(f"source commit manifest identity mismatch: {resolved}")
    components = manifest.get("components")
    if not isinstance(components, Mapping):
        raise ValueError(f"source commit components missing: {resolved}")
    actual_components = DepthStorageManager._raw_component_records(
        depth=np.asarray(group["depth_z"], dtype=np.float32),
        conf=np.asarray(group["conf"], dtype=np.float32),
        mask=np.asarray(group["mask"], dtype=np.uint8),
        rgb=None,
    )
    component_sha256s: dict[str, str] = {}
    for canonical in ("depth", "conf", "mask"):
        row = components.get(canonical)
        if row is None:
            raise ValueError(f"source component {canonical} missing: {resolved}")
        if not isinstance(row, Mapping) or dict(row) != actual_components[canonical]:
            raise ValueError(
                f"source component {canonical} content digest mismatch: {resolved}"
            )
        digest = str(row.get("sha256") or "")
        component_sha256s[canonical] = digest
    source_frame_contract = str(attrs.get("source_frame_contract") or "").strip()
    source_frame_fields_present = any(
        attrs.get(key) is not None
        for key in (
            "source_frame_number",
            "source_media_pts_ns",
            "storage_timestamp_basis",
        )
    )
    source_frame_number: int | None = None
    source_media_pts_ns: int | None = None
    storage_timestamp_basis: str | None = None
    if source_frame_contract or source_frame_fields_present:
        if source_frame_contract != SOURCE_FRAME_CONTRACT:
            raise ValueError(f"source frame contract invalid: {resolved}")
        try:
            source_frame_number = int(attrs["source_frame_number"])
            source_media_pts_ns = int(attrs["source_media_pts_ns"])
            storage_timestamp_basis = str(attrs["storage_timestamp_basis"])
        except Exception as exc:
            raise ValueError(f"source frame evidence incomplete: {resolved}") from exc
        if (
            source_frame_number < 0
            or source_media_pts_ns < 0
            or storage_timestamp_basis
            not in {"source_epoch_pts", "wall_clock_fallback"}
        ):
            raise ValueError(f"source frame evidence invalid: {resolved}")

    return SourceDescriptor(
        path=resolved,
        camera_id=camera_id,
        timestamp_us=timestamp_us,
        sequence=sequence,
        write_id=write_id,
        manifest_file_sha256=str(manifest["manifest_sha256"]),
        component_sha256s=component_sha256s,
        source_frame_contract=source_frame_contract or None,
        source_frame_number=source_frame_number,
        source_media_pts_ns=source_media_pts_ns,
        storage_timestamp_basis=storage_timestamp_basis,
    )


def load_source_report(
    camera_id: str,
    report_path: Path,
) -> tuple[list[SourceDescriptor], list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    report = _strict_json_load(report_path.resolve(strict=True))
    if (
        not isinstance(report, Mapping)
        or report.get("contract") != SOURCE_REPORT_CONTRACT
        or int(report.get("contract_version") or 0) != 1
    ):
        raise ValueError(f"invalid fusion source report: {report_path}")
    sources = report.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"fusion source report has no sources: {report_path}")
    descriptors = [
        _load_descriptor(Path(_required_text(source, "source path")), camera_id)
        for source in sources
    ]
    timestamps = [row.timestamp_us for row in descriptors]
    if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
        raise ValueError(f"source timestamps are not strictly ordered: {report_path}")
    identities = {
        (
            row.camera_id,
            row.timestamp_us,
            row.sequence,
            row.write_id,
            row.manifest_file_sha256,
        )
        for row in descriptors
    }
    if len(identities) != len(descriptors):
        raise ValueError(f"source identities are not unique: {report_path}")
    loaded = [load_depth_snapshot(row.path) for row in descriptors]
    for descriptor, (depth, confidence, mask) in zip(
        descriptors,
        loaded,
        strict=True,
    ):
        actual_components = DepthStorageManager._raw_component_records(
            depth=depth,
            conf=confidence,
            mask=mask,
            rgb=None,
        )
        for component in ("depth", "conf", "mask"):
            if (
                actual_components[component]["sha256"]
                != descriptor.component_sha256s[component]
            ):
                raise ValueError(
                    f"loaded source component {component} changed after "
                    f"validation: {descriptor.path}"
                )
    shapes = {tuple(depth.shape) for depth, _confidence, _mask in loaded}
    if len(shapes) != 1:
        raise ValueError(f"source shapes differ for {camera_id}: {sorted(shapes)}")
    return descriptors, loaded


def summarize_timestamps(
    timestamps_us: Sequence[int],
    *,
    source_fps: float,
    interval_frames: int,
) -> dict[str, Any]:
    if (
        len(timestamps_us) < 2
        or not math.isfinite(source_fps)
        or source_fps <= 0.0
        or interval_frames < 0
    ):
        raise ValueError("cadence timing needs two timestamps, positive FPS, and interval")
    gaps_s = [
        (right - left) / 1_000_000.0
        for left, right in zip(timestamps_us, timestamps_us[1:])
    ]
    if any(gap <= 0.0 or not math.isfinite(gap) for gap in gaps_s):
        raise ValueError("source timestamp gaps must be finite and positive")
    expected_period_s = (interval_frames + 1) / source_fps
    tolerance_s = max(0.15, expected_period_s * 0.08)
    errors = [abs(gap - expected_period_s) for gap in gaps_s]
    return {
        "frame_count": len(timestamps_us),
        "first_timestamp_us": int(timestamps_us[0]),
        "last_timestamp_us": int(timestamps_us[-1]),
        "span_s": (timestamps_us[-1] - timestamps_us[0]) / 1_000_000.0,
        "gaps_s": gaps_s,
        "gap_min_s": min(gaps_s),
        "gap_median_s": statistics.median(gaps_s),
        "gap_max_s": max(gaps_s),
        "expected_period_s": expected_period_s,
        "allowed_period_error_s": tolerance_s,
        "max_period_error_s": max(errors),
        "matches_declared_interval": all(error <= tolerance_s for error in errors),
    }


def contiguous_windows(frame_count: int, window_size: int) -> list[tuple[int, ...]]:
    if frame_count < 1 or not 1 <= window_size <= frame_count:
        raise ValueError("window size must be within the cohort")
    return [
        tuple(range(start, start + window_size))
        for start in range(frame_count - window_size + 1)
    ]


def _compact_evaluation(
    report: Mapping[str, Any],
    *,
    source_indices: Sequence[int],
    timestamps_us: Sequence[int],
) -> dict[str, Any]:
    candidate = report["candidates"][RUNTIME_CANDIDATE]
    return {
        "source_indices": list(source_indices),
        "source_timestamps_us": [int(timestamps_us[index]) for index in source_indices],
        "source_span_s": (
            timestamps_us[source_indices[-1]] - timestamps_us[source_indices[0]]
        )
        / 1_000_000.0,
        "cohort": report["cohort"],
        "scale_normalization": report["scale_normalization"],
        "support": report["support"],
        "confidence": report["confidence"],
        "runtime_candidate": candidate,
    }


def evaluate_indices(
    loaded: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    descriptors: Sequence[SourceDescriptor],
    indices: Sequence[int],
    *,
    min_observations: int,
    depth_agreement_m: float,
) -> dict[str, Any]:
    depth = np.stack([loaded[index][0] for index in indices])
    confidence = np.stack([loaded[index][1] for index in indices])
    mask = np.stack([loaded[index][2] for index in indices])
    report = evaluate_fusion_cohort(
        depth,
        confidence,
        mask,
        min_observations=min_observations,
        depth_agreement_m=depth_agreement_m,
        normalize_frame_scale=True,
    )
    return _compact_evaluation(
        report,
        source_indices=indices,
        timestamps_us=[row.timestamp_us for row in descriptors],
    )


def _finite_values(rows: Iterable[Mapping[str, Any]], path: Sequence[str]) -> list[float]:
    values: list[float] = []
    for row in rows:
        value: object = row
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def metric_envelope(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = {
        "retained_full_frame_fraction": (
            "support",
            "retained_full_frame_fraction",
        ),
        "retained_over_eligible_fraction": (
            "support",
            "retained_over_eligible_fraction",
        ),
        "temporal_residual_p50_m": (
            "runtime_candidate",
            "temporal_residual_p50_m",
        ),
        "temporal_residual_p95_m": (
            "runtime_candidate",
            "temporal_residual_p95_m",
        ),
        "edge_gradient_p95_m": (
            "runtime_candidate",
            "edge",
            "gradient_p95_m",
        ),
    }
    output: dict[str, Any] = {"sample_count": len(rows)}
    for name, path in metrics.items():
        values = _finite_values(rows, path)
        output[name] = (
            {
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
            }
            if values
            else None
        )
    return output


def _source_evidence(descriptors: Sequence[SourceDescriptor]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(row.path),
            "camera_id": row.camera_id,
            "timestamp_us": row.timestamp_us,
            "sequence": row.sequence,
            "write_id": row.write_id,
            "manifest_file_sha256": row.manifest_file_sha256,
            "component_sha256s": dict(row.component_sha256s),
            "source_frame_contract": row.source_frame_contract,
            "source_frame_number": row.source_frame_number,
            "source_media_pts_ns": row.source_media_pts_ns,
            "storage_timestamp_basis": row.storage_timestamp_basis,
        }
        for row in descriptors
    ]


def build_offline_report(
    cohorts: Mapping[
        str,
        tuple[
            Sequence[SourceDescriptor],
            Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
        ],
    ],
    *,
    source_fps: float,
    observed_interval_frames: int,
    window_sizes: Sequence[int] = (4, 5),
    agreements_m: Sequence[float] = (0.18, 0.12),
) -> dict[str, Any]:
    if not cohorts:
        raise ValueError("at least one camera cohort is required")
    if not math.isclose(
        source_fps,
        FIXED_SOURCE_FPS,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("this fixed-corpus report requires source_fps=30")
    if observed_interval_frames != FIXED_OBSERVED_INTERVAL_FRAMES:
        raise ValueError(
            "this fixed-corpus report requires observed_interval_frames=89"
        )
    if tuple(window_sizes) != FIXED_WINDOW_SIZES:
        raise ValueError(
            "this fixed-corpus report requires window_sizes=(4, 5)"
        )
    if len(set(agreements_m)) != len(agreements_m) or any(
        not math.isfinite(value) or value <= 0.0 for value in agreements_m
    ):
        raise ValueError("agreement values must be unique, finite, and positive")

    cameras: dict[str, Any] = {}
    envelope_rows: dict[str, list[Mapping[str, Any]]] = {}
    observed_periods: list[float] = []
    source_frame_evidence_complete = True
    for camera_id, (descriptors, loaded) in sorted(cohorts.items()):
        if len(descriptors) != len(loaded):
            raise ValueError(f"descriptor/data count mismatch for {camera_id}")
        if len(descriptors) != FIXED_COHORT_FRAME_COUNT:
            raise ValueError(
                f"{camera_id} requires exactly "
                f"{FIXED_COHORT_FRAME_COUNT} source frames"
            )
        timing = summarize_timestamps(
            [row.timestamp_us for row in descriptors],
            source_fps=source_fps,
            interval_frames=observed_interval_frames,
        )
        if not timing["matches_declared_interval"]:
            raise ValueError(
                f"{camera_id} source timing does not match declared interval="
                f"{observed_interval_frames}"
            )
        observed_periods.append(float(timing["gap_median_s"]))
        source_frame_evidence_complete = source_frame_evidence_complete and all(
            row.source_frame_contract == SOURCE_FRAME_CONTRACT
            and row.source_frame_number is not None
            and row.source_media_pts_ns is not None
            and bool(row.storage_timestamp_basis)
            for row in descriptors
        )
        full_indices = tuple(range(len(descriptors)))
        baseline: dict[str, Any] = {}
        for agreement in agreements_m:
            key = f"{agreement:.6g}"
            result = evaluate_indices(
                loaded,
                descriptors,
                full_indices,
                min_observations=2,
                depth_agreement_m=agreement,
            )
            baseline[key] = result
            envelope_rows.setdefault(f"full6_agreement_{key}", []).append(result)

        windows: dict[str, Any] = {}
        for window_size in window_sizes:
            if window_size > len(descriptors):
                continue
            group: dict[str, Any] = {}
            for agreement in agreements_m:
                key = f"{agreement:.6g}"
                rows = [
                    evaluate_indices(
                        loaded,
                        descriptors,
                        indices,
                        min_observations=3,
                        depth_agreement_m=agreement,
                    )
                    for indices in contiguous_windows(
                        len(descriptors),
                        window_size,
                    )
                ]
                group[key] = rows
                envelope_rows.setdefault(
                    f"window{window_size}_agreement_{key}",
                    [],
                ).extend(rows)
            windows[str(window_size)] = group
        cameras[camera_id] = {
            "source_timing": timing,
            "sources": _source_evidence(descriptors),
            "full_cohort_min_observations_2": baseline,
            (
                f"interval{observed_interval_frames}_window_sensitivity_"
                "min_observations_3"
            ): windows,
        }

    expected_primary_period_s = 60.0 / source_fps
    primary_period_present = any(
        abs(period - expected_primary_period_s)
        <= max(0.15, expected_primary_period_s * 0.08)
        for period in observed_periods
    )
    return {
        "contract": CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "evidence_scope": {
            "source_kind": "preserved_raw_depth_cohorts",
            "source_fps": source_fps,
            "observed_interval_frames": observed_interval_frames,
            "observed_interval_expected_period_s": (
                observed_interval_frames + 1
            )
            / source_fps,
            "requested_comparison": {
                "baseline": {
                    "interval_frames": 89,
                    "burst_seconds": 20,
                    "min_observations": 2,
                    "depth_agreement_m": 0.18,
                },
                "primary": {
                    "interval_frames": 59,
                    "burst_seconds": 12,
                    "min_observations": 3,
                    "depth_agreement_m": 0.18,
                },
                "control": {
                    "interval_frames": 89,
                    "burst_seconds": 12,
                    "min_observations": 3,
                    "depth_agreement_m": 0.18,
                },
            },
        },
        "cadence_conclusion": {
            "status": "blocked_requires_distinct_fresh_arm_captures",
            "winner": None,
            "cadence_ab_conclusive": False,
            "primary_period_present_in_sources": primary_period_present,
            "exact_source_frame_evidence_complete": (
                source_frame_evidence_complete
            ),
            "reasons": [
                "all preserved sources were inferred at the declared interval=89 cadence",
                "interval=59 requires approximately two-second observations that do not exist in these cohorts",
                "subsampling or interpolating interval=89 depth cannot recreate missing model outputs",
                "four/five-frame windows test cohort-size sensitivity only, not a fresh 12-second capture fence",
                *(
                    []
                    if source_frame_evidence_complete
                    else [
                        "preserved snapshots predate exact source frame/PTS evidence and cannot be paired to hashed RGB frames"
                    ]
                ),
            ],
        },
        "agreement_conclusion": {
            "status": "preliminary_sensitivity_only",
            "winner": None,
            "reason": (
                "agreement can be compared on identical raw cohorts, but cadence "
                "is not selected and fixed boundary/obstacle labels are absent"
            ),
        },
        "cameras": cameras,
        "metric_envelopes": {
            key: metric_envelope(rows)
            for key, rows in sorted(envelope_rows.items())
        },
        "fresh_capture_execution_plan": {
            "required_inputs": [
                "three synchronized constant-frame-rate 30 fps RGB clips with per-file SHA-256",
                "raw depth receipts containing source frame number, media PTS, and timestamp basis",
                "fixed calibration, dewarper, engine, preprocessing, and camera-order digests",
                "a clean isolated DS9 measurement session for each arm",
            ],
            "arms": [
                {
                    "id": "baseline",
                    "interval_frames": 89,
                    "burst_seconds": 20,
                    "min_observations": 2,
                    "depth_agreement_m": 0.18,
                },
                {
                    "id": "primary",
                    "interval_frames": 59,
                    "burst_seconds": 12,
                    "min_observations": 3,
                    "depth_agreement_m": 0.18,
                },
                {
                    "id": "control",
                    "interval_frames": 89,
                    "burst_seconds": 12,
                    "min_observations": 3,
                    "depth_agreement_m": 0.18,
                },
            ],
            "bounded_steps": [
                "copy the canonical pipeline and MapAnything infer INI into an external per-arm run directory",
                "change only nvinfer interval and the three capture-event environment values declared by that arm",
                "reset all three fixed clips to the identical source frame before each isolated launch",
                "perform one explicit manual capture and preserve its compact capture-event receipt plus every raw commit manifest",
                "repeat three times per arm, resetting the clips each time; do not reuse a raw snapshot across arms",
                "verify actual source count, timestamp gaps, source identities, engine/config digests, GPU load, and burst wall time before comparing quality",
                "select cadence only with fixed floor/boundary/obstacle labels; then rerun 0.18 versus 0.12 offline on each winning-cadence raw cohort",
            ],
            "promotion_policy": (
                "no selector, runtime asset, or service changes until all three "
                "arms have comparable evidence and labeled acceptance passes"
            ),
        },
    }


def _parse_source_report(value: str) -> tuple[str, Path]:
    camera, separator, raw_path = value.partition("=")
    camera_id = camera.strip()
    if not separator or not camera_id or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected CAMERA=REPORT.json")
    return camera_id, Path(raw_path.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit preserved MapAnything cohorts without pretending one cadence "
            "contains another cadence's missing inference outputs."
        )
    )
    parser.add_argument(
        "--source-report",
        action="append",
        required=True,
        type=_parse_source_report,
        metavar="CAMERA=REPORT.json",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        choices=(FIXED_SOURCE_FPS,),
        default=FIXED_SOURCE_FPS,
    )
    parser.add_argument(
        "--observed-interval-frames",
        type=int,
        choices=(FIXED_OBSERVED_INTERVAL_FRAMES,),
        default=FIXED_OBSERVED_INTERVAL_FRAMES,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_reports = dict(args.source_report)
    if len(source_reports) != len(args.source_report):
        raise SystemExit("duplicate --source-report camera")
    cohorts = {
        camera: load_source_report(camera, path)
        for camera, path in source_reports.items()
    }
    report = build_offline_report(
        cohorts,
        source_fps=float(args.source_fps),
        observed_interval_frames=int(args.observed_interval_frames),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
    print(f"[OK] wrote {output} sha256={_sha256_bytes(encoded.encode('utf-8'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
