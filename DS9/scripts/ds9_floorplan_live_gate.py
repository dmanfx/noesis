#!/usr/bin/env python3
"""Prove exact DS9 capture-event, active-floorplan, BEV, and cache contracts."""

from __future__ import annotations

import argparse
import array
import asyncio
import base64
import binascii
import hashlib
import json
import math
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis_core.private_paths import (  # noqa: E402
    PrivatePathError,
    atomic_create_private_file,
    read_private_file,
    require_fresh_private_file_bundle,
)
from noesis_core.strict_json import strict_json_loads  # noqa: E402
from scripts.internal_auth_client import (  # noqa: E402
    RequiredInternalAuth,
    add_auth_token_file_argument,
    connect_required_websocket,
    load_required_internal_auth,
)

CONTRACT = "noesis.ds9.floorplan-live-gate"
SCHEMA_VERSION = 4
CONTRACT_VERSION = 4
CANONICAL_REPORT_FILENAME = "mapanything-depth-quality.json"
SESSION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{5,47}$")
RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PORTABLE_REF_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
FLOORPLAN_CONTRACT_VERSION = 10
FLOORPLAN_FRAME = "camera_local_ground_m"
FLOORPLAN_ORIENTATION = "camera_ground_right_forward"
MAX_WS_MESSAGE_BYTES = 64 * 1024 * 1024
MAX_WS_MESSAGES_SEEN = 100_000
MAX_GRID_CELLS = 16 * 1024 * 1024
GST_CLOCK_TIME_NONE = (1 << 64) - 1
MAX_CAPTURE_RGB_DIMENSION = 16_384
MAX_CAPTURE_RGB_FRAME_BYTES = 64 * 1024 * 1024
MAX_REPORT_BYTES = 256 * 1024
MAX_SOURCE_TRANSCRIPT_BYTES = 2 * 1024 * 1024
MAX_SOURCE_MESSAGES = 256
CANONICAL_SOURCE_TRANSCRIPT_FILENAME = "mapanything-depth-quality-source.json"
SOURCE_TRANSCRIPT_CONTRACT = "noesis.ds9.floorplan-source-transcript"
SOURCE_TRANSCRIPT_VERSION = 4
SOURCE_PRIVACY_POLICY = {
    "payload_policy": "minimal_exact_contract_evidence",
    "raw_depth_grids": "digest_only",
    "raw_embedding_vectors": "absent",
    "image_frames": "absent",
    "secrets": "absent",
}
REPORT_KEYS = frozenset(
    {
        "schema_version",
        "contract",
        "contract_version",
        "session_id",
        "runtime_lane",
        "runtime_instance_id",
        "runtime_run_id",
        "ok",
        "max_snapshot_age_s",
        "configured_camera_count",
        "validated_camera_count",
        "all_configured_camera_floorplans_validated",
        "exact_capture_event_camera_count",
        "cache_only_validated_camera_count",
        "cache_only_zero_mutation",
        "bev_renderer_ready",
        "bev_active_camera_count",
        "bev_inactive_ready_camera_count",
        "bev_failed_camera_count",
        "all_configured_cameras_bev_ready",
        "cameras",
        "cache_only_cameras",
        "runtime_health",
        "source_evidence",
        "errors",
    }
)
SOURCE_DOCUMENT_KEYS = frozenset(
    {
        "schema_version",
        "contract",
        "contract_version",
        "session_id",
        "runtime_lane",
        "runtime_instance_id",
        "runtime_run_id",
        "camera_ids",
        "max_snapshot_age_s",
        "privacy",
        "message_count",
        "messages",
    }
)
SOURCE_EVIDENCE_KEYS = frozenset(
    {
        "filename",
        "sha256",
        "message_count",
        "first_observed_at_us",
        "last_observed_at_us",
    }
)
RETRIABLE_ERRORS = frozenset(
    {
        "no_raw_snapshots_for_capture_event",
        "insufficient_raw_observations",
        "no_depth",
        "stale_depth",
        "timeout_waiting_for_depth",
    }
)
REQUIRED_LAYERS = (
    "density",
    "observed",
    "unknown",
    "height",
    "height_agl",
    "distance",
)
OPTIONAL_IMMUTABLE_LAYERS = (
    "gradient",
    "obstacle_height",
    "walkable",
    "inferred_walkable",
)
OBSERVATION_META_KEYS = frozenset(
    {
        "contract",
        "observed_definition",
        "unknown_definition",
        "observed_cells",
        "unknown_cells",
        "total_cells",
    }
)
CANONICAL_FLOAT32_QNAN_LE = b"\x00\x00\xc0\x7f"
FRESH_RESULT_KEYS = frozenset(
    {
        "camera_id",
        "snapshot_age_s",
        "snapshot_ts_us",
        "snapshot_ref",
        "snapshot_id",
        "snapshot_content_sha256",
        "floorplan_ts_us",
        "point_count",
        "grid_shape",
        "grid_res_m",
        "bounds",
        "ray_to_floorplan_alignment",
        "served_from_cache",
        "floorplan_contract_version",
        "calibration_fingerprint",
        "floorplan_payload_sha256",
        "layer_sha256s",
        "observation_meta",
        "inferred_walkable_present",
        "capture_event_evidence_sha256",
        "fusion_evidence_sha256",
        "capture_event_id",
        "capture_event_source_snapshot_count",
        "capture_event_rgb_status",
        "capture_event_rgb_evidence",
    }
)
CACHE_RESULT_KEYS = frozenset(
    {
        "camera_id",
        "snapshot_age_s",
        "snapshot_ts_us",
        "snapshot_ref",
        "snapshot_id",
        "snapshot_content_sha256",
        "floorplan_ts_us",
        "point_count",
        "grid_shape",
        "grid_res_m",
        "bounds",
        "ray_to_floorplan_alignment",
        "served_from_cache",
        "floorplan_contract_version",
        "calibration_fingerprint",
        "floorplan_payload_sha256",
        "layer_sha256s",
        "observation_meta",
        "inferred_walkable_present",
        "identity_unchanged",
        "payload_unchanged",
    }
)
HEALTH_RESULT_KEYS = frozenset(
    {
        "bev_frame",
        "bev_health",
        "active_floorplan_health",
        "capture_event_controller_health",
    }
)
CAPTURE_EVENT_KEYS = frozenset(
    {
        "contract",
        "contract_version",
        "camera_id",
        "request_kind",
        "capture_mode",
        "baseline_raw_timestamp_us",
        "baseline_mapanything_idle",
        "baseline_storage_flush",
        "postburst_mapanything_idle",
        "postburst_storage_flush",
        "parameters",
        "fusion_evidence_sha256",
        "raw_snapshot_count",
        "rgb",
        "fusion_quality",
        "fused_snapshot",
    }
)
FUSION_QUALITY_KEYS = frozenset(
    {
        "contract",
        "support_valid_fraction",
        "median_support",
        "support_evidence",
        "support_quality_gate",
        "frame_scale_normalization",
    }
)
SUPPORT_EVIDENCE_KEYS = frozenset(
    {
        "contract",
        "cohort_size",
        "full_frame_pixels",
        "required_observations",
        "fixed_min_observations",
        "cohort_floor_observations",
        "strict_majority_observations",
        "tie_policy",
        "min_observation_ratio",
        "ratio_required_observations",
        "quarantined_frame_count",
        "quarantined_frame_indices",
        "eligible_pixels",
        "eligible_full_frame_fraction",
        "consensus_pixels",
        "consensus_full_frame_fraction",
        "consensus_retained_eligible_fraction",
        "support_count_histogram",
        "component_evidence",
        "temporal_absolute_residual_median_m",
        "temporal_absolute_residual_p95_m",
    }
)
SUPPORT_COMPONENT_KEYS = frozenset(
    {
        "connectivity",
        "component_count",
        "largest_component_pixels",
        "largest_component_fraction",
        "fragment_pixels",
        "fragment_fraction",
        "hole_count",
        "hole_pixels",
    }
)
SUPPORT_QUALITY_GATE_KEYS = frozenset(
    {"contract", "metric", "observed", "required", "passed"}
)
FRAME_SCALE_NORMALIZATION_KEYS = frozenset(
    {
        "contract",
        "enabled",
        "algorithm",
        "statistic",
        "baseline",
        "minimum_relative_change",
        "accepted_factor_bounds",
        "frame_medians",
        "proposed_factors",
        "factors",
        "applied",
        "rejected",
        "rejection_policy",
    }
)
FUSED_SNAPSHOT_KEYS = frozenset(
    {
        "camera_id",
        "storage_key",
        "timestamp_us",
        "snapshot_id",
        "artifact_ref",
        "content_sha256",
        "sequence",
        "manifest_sha256",
        "event_id",
        "source_snapshot_ids",
        "snapshot_role",
        "fusion_level",
    }
)
IDLE_RECEIPT_KEYS = frozenset(
    {
        "active_captures",
        "unfinished_tasks",
        "worker_started",
        "worker_alive",
        "accepting",
    }
)
FLUSH_RECEIPT_KEYS = frozenset(
    {
        "frontier_sequence",
        "completed",
        "timed_out",
        "pending_sequences",
        "failed_sequences",
        "poisoned",
    }
)
CAPTURE_PARAMETER_KEYS = frozenset(
    {
        "burst_seconds",
        "raw_limit",
        "min_observations",
        "depth_agreement_m",
        "max_cohort_span_us",
    }
)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: object, label: str) -> int:
    if not _is_int(value) or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_int(value: object, label: str) -> int:
    if not _is_int(value) or int(value) < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(value)


def _finite_float(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or (positive and parsed <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return parsed


def _nonempty_text(value: object, label: str, *, max_bytes: int = 1024) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    if len(text.encode("utf-8")) > max_bytes:
        raise ValueError(f"{label} exceeds the byte bound")
    return text


def _sha256(value: object, label: str) -> str:
    text = _nonempty_text(value, label)
    if SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _exact_mapping(value: object, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} schema drifted")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("evidence is outside the canonical JSON domain") from exc


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _portable_snapshot_ref(value: object) -> str:
    text = _nonempty_text(value, "snapshot_ref")
    if text.startswith("/") or "\\" in text:
        raise ValueError("snapshot_ref must be a portable relative reference")
    parts = text.split("/")
    if (
        not parts
        or any(
            part in {"", ".", ".."}
            or PORTABLE_REF_COMPONENT_RE.fullmatch(part) is None
            for part in parts
        )
    ):
        raise ValueError("snapshot_ref must be a portable relative reference")
    return text


def _load_yaml_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"unable to load {label} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} root must be a mapping")
    return payload


def _active_camera_ids(pipeline_config: Path, cameras_config: Path) -> tuple[str, ...]:
    pipeline = _load_yaml_mapping(pipeline_config, "pipeline config")
    cameras_payload = _load_yaml_mapping(cameras_config, "camera config")
    sources = pipeline.get("sources")
    cameras = cameras_payload.get("cameras")
    if not isinstance(sources, list) or not sources:
        raise ValueError("pipeline config must contain at least one source")
    if not isinstance(cameras, Mapping):
        raise ValueError("camera config must contain a cameras mapping")
    active: list[str] = []
    for source_index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ValueError(f"sources[{source_index}] must be a mapping")
        enabled = source.get("enable", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"sources[{source_index}].enable must be boolean")
        if not enabled:
            continue
        camera = cameras.get(source_index, cameras.get(str(source_index)))
        if not isinstance(camera, Mapping):
            raise ValueError(
                f"active source {source_index} has no matching camera metadata"
            )
        active.append(
            _nonempty_text(camera.get("name"), f"cameras[{source_index}].name")
        )
    if not active:
        raise ValueError("pipeline config has no active camera sources")
    if len(active) != len(set(active)):
        raise ValueError("active camera names must be unique")
    return tuple(active)


def _decode_grid(
    layer: object,
    label: str,
    *,
    allow_canonical_quiet_nan: bool = False,
) -> tuple[tuple[int, int], array.array, str, dict[str, object]]:
    if not isinstance(layer, Mapping):
        raise ValueError(f"{label} layer must be an object")
    shape = layer.get("grid_shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"{label}.grid_shape must be [rows, cols]")
    rows = _positive_int(shape[0], f"{label}.grid_shape[0]")
    cols = _positive_int(shape[1], f"{label}.grid_shape[1]")
    cells = rows * cols
    if cells > MAX_GRID_CELLS:
        raise ValueError(f"{label} grid exceeds the cell bound")
    encoded = _nonempty_text(layer.get("grid_b64"), f"{label}.grid_b64", max_bytes=MAX_WS_MESSAGE_BYTES)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"{label}.grid_b64 is invalid base64") from exc
    expected_bytes = cells * 4
    if len(raw) != expected_bytes:
        raise ValueError(
            f"{label} byte length mismatch: {len(raw)} != {expected_bytes}"
        )
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != cells:
        raise ValueError(f"{label} grid contains invalid float32 values")
    for index, value in enumerate(values):
        if math.isfinite(float(value)):
            continue
        start = index * 4
        if (
            allow_canonical_quiet_nan
            and math.isnan(float(value))
            and raw[start : start + 4] == CANONICAL_FLOAT32_QNAN_LE
        ):
            continue
        raise ValueError(f"{label} grid contains invalid float32 values")
    value_min = _finite_float(layer.get("value_min"), f"{label}.value_min")
    value_max = _finite_float(layer.get("value_max"), f"{label}.value_max")
    if value_max < value_min:
        raise ValueError(f"{label} value range is inverted")
    digest = hashlib.sha256(raw).hexdigest()
    digest_record = {
        "grid_shape": [rows, cols],
        "value_min": value_min,
        "value_max": value_max,
        "raw_float32_sha256": digest,
    }
    return (rows, cols), values, digest, digest_record


def _validate_height_agl_observation_policy(
    height_agl: array.array,
    observed: array.array,
    unknown: array.array,
) -> None:
    if len(height_agl) != len(observed) or len(height_agl) != len(unknown):
        raise ValueError("height_agl and observation cell counts differ")
    for index, height_value in enumerate(height_agl):
        if not math.isnan(float(height_value)):
            continue
        if float(observed[index]) != 0.0 or float(unknown[index]) != 1.0:
            raise ValueError(
                "height_agl canonical NaN is allowed only in unknown cells"
            )


def _validate_idle_receipt(value: object, label: str) -> None:
    receipt = _exact_mapping(value, IDLE_RECEIPT_KEYS, label)
    if (
        _nonnegative_int(receipt.get("active_captures"), f"{label}.active_captures")
        != 0
        or _nonnegative_int(
            receipt.get("unfinished_tasks"), f"{label}.unfinished_tasks"
        )
        != 0
        or receipt.get("worker_started") is not True
        or receipt.get("worker_alive") is not True
        or receipt.get("accepting") is not True
    ):
        raise ValueError(f"{label} does not prove an idle accepting worker")


def _validate_flush_receipt(value: object, label: str) -> int:
    receipt = _exact_mapping(value, FLUSH_RECEIPT_KEYS, label)
    frontier = _nonnegative_int(
        receipt.get("frontier_sequence"), f"{label}.frontier_sequence"
    )
    if (
        receipt.get("completed") is not True
        or receipt.get("timed_out") is not False
        or receipt.get("poisoned") is not False
        or receipt.get("pending_sequences") != []
        or receipt.get("failed_sequences") != []
    ):
        raise ValueError(f"{label} does not prove a clean storage frontier")
    return frontier


def _fraction(value: object, label: str) -> float:
    parsed = _finite_float(value, label)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return parsed


def _validate_fusion_quality(value: object) -> int:
    quality = _exact_mapping(
        value,
        FUSION_QUALITY_KEYS,
        "capture_event.fusion_quality",
    )
    if quality.get("contract") != "noesis.capture_event.depth_quality.v1":
        raise ValueError("capture_event fusion quality contract drifted")
    support_fraction = _fraction(
        quality.get("support_valid_fraction"),
        "capture_event.fusion_quality.support_valid_fraction",
    )
    median_support = _finite_float(
        quality.get("median_support"),
        "capture_event.fusion_quality.median_support",
    )
    if median_support < 0.0:
        raise ValueError("capture_event fusion median support is negative")

    support = _exact_mapping(
        quality.get("support_evidence"),
        SUPPORT_EVIDENCE_KEYS,
        "capture_event.fusion_quality.support_evidence",
    )
    if (
        support.get("contract") != "noesis.depth.fusion.support.v1"
        or support.get("tie_policy") != "reject_exact_half_support"
    ):
        raise ValueError("capture_event fusion support contract drifted")
    cohort_size = _positive_int(
        support.get("cohort_size"),
        "capture_event.fusion_quality.support_evidence.cohort_size",
    )
    full_pixels = _positive_int(
        support.get("full_frame_pixels"),
        "capture_event.fusion_quality.support_evidence.full_frame_pixels",
    )
    required = _positive_int(
        support.get("required_observations"),
        "capture_event.fusion_quality.support_evidence.required_observations",
    )
    count_fields = (
        "fixed_min_observations",
        "cohort_floor_observations",
        "strict_majority_observations",
        "ratio_required_observations",
    )
    counts = {
        field: _positive_int(
            support.get(field),
            f"capture_event.fusion_quality.support_evidence.{field}",
        )
        for field in count_fields
    }
    if (
        required > cohort_size
        or any(value > cohort_size for value in counts.values())
        or median_support > cohort_size
    ):
        raise ValueError("capture_event fusion support exceeds cohort size")
    _fraction(
        support.get("min_observation_ratio"),
        "capture_event.fusion_quality.support_evidence.min_observation_ratio",
    )

    quarantined_count = _nonnegative_int(
        support.get("quarantined_frame_count"),
        "capture_event.fusion_quality.support_evidence.quarantined_frame_count",
    )
    quarantined = support.get("quarantined_frame_indices")
    if (
        not isinstance(quarantined, list)
        or len(quarantined) != quarantined_count
        or len(quarantined) != len(set(quarantined))
        or any(
            not _is_int(index) or int(index) < 0 or int(index) >= cohort_size
            for index in quarantined
        )
    ):
        raise ValueError("capture_event fusion quarantined-frame evidence is invalid")

    eligible_pixels = _nonnegative_int(
        support.get("eligible_pixels"),
        "capture_event.fusion_quality.support_evidence.eligible_pixels",
    )
    consensus_pixels = _nonnegative_int(
        support.get("consensus_pixels"),
        "capture_event.fusion_quality.support_evidence.consensus_pixels",
    )
    if consensus_pixels > eligible_pixels or eligible_pixels > full_pixels:
        raise ValueError("capture_event fusion pixel counts do not reconcile")
    eligible_fraction = _fraction(
        support.get("eligible_full_frame_fraction"),
        "capture_event.fusion_quality.support_evidence.eligible_full_frame_fraction",
    )
    consensus_fraction = _fraction(
        support.get("consensus_full_frame_fraction"),
        "capture_event.fusion_quality.support_evidence.consensus_full_frame_fraction",
    )
    retained_fraction = _fraction(
        support.get("consensus_retained_eligible_fraction"),
        "capture_event.fusion_quality.support_evidence.consensus_retained_eligible_fraction",
    )
    if (
        not math.isclose(
            eligible_fraction,
            eligible_pixels / full_pixels,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            consensus_fraction,
            consensus_pixels / full_pixels,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            retained_fraction,
            consensus_pixels / max(1, eligible_pixels),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            support_fraction,
            consensus_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("capture_event fusion support fractions do not reconcile")

    histogram = support.get("support_count_histogram")
    expected_histogram_keys = {str(index) for index in range(cohort_size + 1)}
    if (
        not isinstance(histogram, Mapping)
        or set(histogram) != expected_histogram_keys
        or any(not _is_int(value) or int(value) < 0 for value in histogram.values())
        or sum(int(value) for value in histogram.values()) != full_pixels
    ):
        raise ValueError("capture_event fusion support histogram is invalid")

    components = _exact_mapping(
        support.get("component_evidence"),
        SUPPORT_COMPONENT_KEYS,
        "capture_event.fusion_quality.support_evidence.component_evidence",
    )
    if components.get("connectivity") != 8:
        raise ValueError("capture_event fusion component connectivity drifted")
    component_count = _nonnegative_int(
        components.get("component_count"),
        "capture_event.fusion_quality.support_evidence.component_evidence.component_count",
    )
    largest_pixels = _nonnegative_int(
        components.get("largest_component_pixels"),
        "capture_event.fusion_quality.support_evidence.component_evidence.largest_component_pixels",
    )
    fragment_pixels = _nonnegative_int(
        components.get("fragment_pixels"),
        "capture_event.fusion_quality.support_evidence.component_evidence.fragment_pixels",
    )
    _nonnegative_int(
        components.get("hole_count"),
        "capture_event.fusion_quality.support_evidence.component_evidence.hole_count",
    )
    hole_pixels = _nonnegative_int(
        components.get("hole_pixels"),
        "capture_event.fusion_quality.support_evidence.component_evidence.hole_pixels",
    )
    largest_fraction = _fraction(
        components.get("largest_component_fraction"),
        "capture_event.fusion_quality.support_evidence.component_evidence.largest_component_fraction",
    )
    fragment_fraction = _fraction(
        components.get("fragment_fraction"),
        "capture_event.fusion_quality.support_evidence.component_evidence.fragment_fraction",
    )
    if (
        largest_pixels + fragment_pixels != consensus_pixels
        or hole_pixels > full_pixels
        or (consensus_pixels == 0 and component_count != 0)
        or (
            consensus_pixels > 0
            and (
                component_count <= 0
                or not math.isclose(
                    largest_fraction,
                    largest_pixels / consensus_pixels,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    fragment_fraction,
                    fragment_pixels / consensus_pixels,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        )
    ):
        raise ValueError("capture_event fusion component evidence does not reconcile")
    residual_median = _finite_float(
        support.get("temporal_absolute_residual_median_m"),
        "capture_event.fusion_quality.support_evidence.temporal_absolute_residual_median_m",
    )
    residual_p95 = _finite_float(
        support.get("temporal_absolute_residual_p95_m"),
        "capture_event.fusion_quality.support_evidence.temporal_absolute_residual_p95_m",
    )
    if residual_median < 0.0 or residual_p95 < residual_median:
        raise ValueError("capture_event fusion residual evidence is invalid")

    gate = _exact_mapping(
        quality.get("support_quality_gate"),
        SUPPORT_QUALITY_GATE_KEYS,
        "capture_event.fusion_quality.support_quality_gate",
    )
    observed = _fraction(
        gate.get("observed"),
        "capture_event.fusion_quality.support_quality_gate.observed",
    )
    required_fraction = _fraction(
        gate.get("required"),
        "capture_event.fusion_quality.support_quality_gate.required",
    )
    if (
        gate.get("contract") != "noesis.depth.fusion.quality_gate.v1"
        or gate.get("metric") != "consensus_full_frame_fraction"
        or gate.get("passed") is not True
        or observed < required_fraction
        or not math.isclose(
            observed,
            support_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("capture_event fusion quality gate is invalid")

    normalization = _exact_mapping(
        quality.get("frame_scale_normalization"),
        FRAME_SCALE_NORMALIZATION_KEYS,
        "capture_event.fusion_quality.frame_scale_normalization",
    )
    if (
        normalization.get("contract")
        != "noesis.depth.fusion.frame_scale_normalization.v1"
        or normalization.get("enabled") is not True
        or normalization.get("algorithm")
        != "depth_times_cohort_median_over_frame_median"
        or normalization.get("statistic") != "median_valid_depth"
        or normalization.get("rejection_policy") != "quarantine_entire_frame"
    ):
        raise ValueError("capture_event frame-scale normalization contract drifted")
    baseline = _finite_float(
        normalization.get("baseline"),
        "capture_event.fusion_quality.frame_scale_normalization.baseline",
        positive=True,
    )
    if baseline <= 0.0:
        raise ValueError("capture_event frame-scale normalization baseline is invalid")
    _fraction(
        normalization.get("minimum_relative_change"),
        "capture_event.fusion_quality.frame_scale_normalization.minimum_relative_change",
    )
    factor_bounds = normalization.get("accepted_factor_bounds")
    if (
        not isinstance(factor_bounds, list)
        or len(factor_bounds) != 2
        or any(
            _finite_float(
                value,
                "capture_event.fusion_quality.frame_scale_normalization.accepted_factor_bounds",
                positive=True,
            )
            <= 0.0
            for value in factor_bounds
        )
        or float(factor_bounds[0]) > float(factor_bounds[1])
    ):
        raise ValueError("capture_event frame-scale factor bounds are invalid")
    for field in ("frame_medians", "proposed_factors"):
        values = normalization.get(field)
        if (
            not isinstance(values, list)
            or len(values) != cohort_size
            or any(
                item is not None
                and (
                    not isinstance(item, (int, float))
                    or isinstance(item, bool)
                    or not math.isfinite(float(item))
                    or float(item) <= 0.0
                )
                for item in values
            )
        ):
            raise ValueError(f"capture_event frame-scale {field} is invalid")
    factors = normalization.get("factors")
    applied = normalization.get("applied")
    rejected = normalization.get("rejected")
    if (
        not isinstance(factors, list)
        or len(factors) != cohort_size
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            or float(item) <= 0.0
            for item in factors
        )
        or not isinstance(applied, list)
        or len(applied) != cohort_size
        or any(type(item) is not bool for item in applied)  # noqa: E721
        or not isinstance(rejected, list)
        or len(rejected) != cohort_size
        or any(type(item) is not bool for item in rejected)  # noqa: E721
        or any(bool(applied[index]) and bool(rejected[index]) for index in range(cohort_size))
        or [index for index, item in enumerate(rejected) if item] != quarantined
    ):
        raise ValueError("capture_event frame-scale decisions are invalid")
    return cohort_size


def _validate_capture_event(
    value: object,
    *,
    evidence_sha256: object,
    camera_id: str,
    snapshot_ref: str,
    snapshot_id: str,
    snapshot_content_sha256: str,
    snapshot_ts_us: int,
) -> dict[str, object]:
    event = _exact_mapping(value, CAPTURE_EVENT_KEYS, "capture_event")
    if (
        event.get("contract") != "noesis.capture_event_controller"
        or event.get("contract_version") != 1
        or event.get("camera_id") != camera_id
        or event.get("request_kind") != "floorplan"
        or event.get("capture_mode") not in {"depth_only", "depth_rgb_exact"}
    ):
        raise ValueError("capture_event contract or scope mismatch")
    compact_digest = _sha256(evidence_sha256, "capture_event_evidence_sha256")
    if _canonical_json_sha256(event) != compact_digest:
        raise ValueError("capture_event evidence digest mismatch")

    baseline_ts = _nonnegative_int(
        event.get("baseline_raw_timestamp_us"),
        "capture_event.baseline_raw_timestamp_us",
    )
    if baseline_ts >= snapshot_ts_us:
        raise ValueError("capture_event fused snapshot is not newer than its baseline")
    _validate_idle_receipt(
        event.get("baseline_mapanything_idle"),
        "capture_event.baseline_mapanything_idle",
    )
    _validate_idle_receipt(
        event.get("postburst_mapanything_idle"),
        "capture_event.postburst_mapanything_idle",
    )
    baseline_frontier = _validate_flush_receipt(
        event.get("baseline_storage_flush"),
        "capture_event.baseline_storage_flush",
    )
    post_frontier = _validate_flush_receipt(
        event.get("postburst_storage_flush"),
        "capture_event.postburst_storage_flush",
    )
    if post_frontier < baseline_frontier:
        raise ValueError("capture_event storage frontier regressed")

    parameters = _exact_mapping(
        event.get("parameters"), CAPTURE_PARAMETER_KEYS, "capture_event.parameters"
    )
    _finite_float(parameters.get("burst_seconds"), "capture_event.parameters.burst_seconds", positive=True)
    raw_limit = _positive_int(parameters.get("raw_limit"), "capture_event.parameters.raw_limit")
    minimum = _positive_int(
        parameters.get("min_observations"),
        "capture_event.parameters.min_observations",
    )
    if minimum > raw_limit:
        raise ValueError("capture_event minimum observations exceeds its raw limit")
    _finite_float(
        parameters.get("depth_agreement_m"),
        "capture_event.parameters.depth_agreement_m",
        positive=True,
    )
    _positive_int(
        parameters.get("max_cohort_span_us"),
        "capture_event.parameters.max_cohort_span_us",
    )
    quality_cohort_size = _validate_fusion_quality(event.get("fusion_quality"))

    rgb_value = event.get("rgb")
    if not isinstance(rgb_value, Mapping):
        raise ValueError("capture_event RGB evidence must be an object")
    rgb_status = rgb_value.get("status")
    if rgb_status == "not_requested":
        rgb = _exact_mapping(
            rgb_value,
            frozenset({"status", "provider_configured"}),
            "capture_event.rgb",
        )
        if type(rgb.get("provider_configured")) is not bool:  # noqa: E721
            raise ValueError(
                "capture_event not-requested RGB provider evidence is invalid"
            )
        if event.get("capture_mode") != "depth_only":
            raise ValueError("capture_event RGB status and capture mode disagree")
    elif rgb_status == "available":
        rgb = _exact_mapping(
            rgb_value,
            frozenset(
                {
                    "status",
                    "provider_configured",
                    "source_id",
                    "batch_id",
                    "captured_at_us",
                    "frame_id",
                    "source_media_pts_ns",
                    "width",
                    "height",
                    "color_space",
                    "content_sha256",
                }
            ),
            "capture_event.rgb",
        )
        if rgb.get("provider_configured") is not True:
            raise ValueError("capture_event exact RGB provider is not configured")
        _nonnegative_int(rgb.get("source_id"), "capture_event.rgb.source_id")
        _nonnegative_int(rgb.get("batch_id"), "capture_event.rgb.batch_id")
        _positive_int(rgb.get("captured_at_us"), "capture_event.rgb.captured_at_us")
        _nonnegative_int(rgb.get("frame_id"), "capture_event.rgb.frame_id")
        source_media_pts_ns = _nonnegative_int(
            rgb.get("source_media_pts_ns"),
            "capture_event.rgb.source_media_pts_ns",
        )
        if source_media_pts_ns == GST_CLOCK_TIME_NONE:
            raise ValueError("capture_event exact RGB source PTS is CLOCK_TIME_NONE")
        width = _positive_int(rgb.get("width"), "capture_event.rgb.width")
        height = _positive_int(rgb.get("height"), "capture_event.rgb.height")
        if (
            width > MAX_CAPTURE_RGB_DIMENSION
            or height > MAX_CAPTURE_RGB_DIMENSION
            or width * height * 3 > MAX_CAPTURE_RGB_FRAME_BYTES
        ):
            raise ValueError("capture_event exact RGB dimensions exceed provider bounds")
        if rgb.get("color_space") != "rgb8":
            raise ValueError("capture_event exact RGB color space is invalid")
        _sha256(rgb.get("content_sha256"), "capture_event.rgb.content_sha256")
        if event.get("capture_mode") != "depth_rgb_exact":
            raise ValueError("capture_event RGB status and capture mode disagree")
    else:
        raise ValueError("capture_event RGB status is invalid")

    fused = _exact_mapping(
        event.get("fused_snapshot"),
        FUSED_SNAPSHOT_KEYS,
        "capture_event.fused_snapshot",
    )
    source_snapshot_ids = fused.get("source_snapshot_ids")
    if (
        not isinstance(source_snapshot_ids, list)
        or not source_snapshot_ids
        or len(source_snapshot_ids) != len(set(source_snapshot_ids))
        or any(not isinstance(item, str) or not item for item in source_snapshot_ids)
    ):
        raise ValueError("capture_event source snapshot IDs are invalid")
    raw_snapshot_count = _positive_int(
        event.get("raw_snapshot_count"), "capture_event.raw_snapshot_count"
    )
    if (
        raw_snapshot_count != len(source_snapshot_ids)
        or raw_snapshot_count < minimum
        or raw_snapshot_count != quality_cohort_size
    ):
        raise ValueError("capture_event raw snapshot count does not reconcile")
    event_id = _nonempty_text(fused.get("event_id"), "capture_event.fused_snapshot.event_id")
    if not event_id.startswith("capture-event-sha256:") or SHA256_RE.fullmatch(event_id.removeprefix("capture-event-sha256:")) is None:
        raise ValueError("capture_event event ID is invalid")
    if (
        fused.get("camera_id") != camera_id
        or _nonempty_text(fused.get("storage_key"), "capture_event.fused_snapshot.storage_key") != camera_id
        or _positive_int(fused.get("timestamp_us"), "capture_event.fused_snapshot.timestamp_us") != snapshot_ts_us
        or fused.get("snapshot_id") != snapshot_id
        or fused.get("artifact_ref") != f"depth-zarr:{snapshot_ref}"
        or fused.get("content_sha256") != snapshot_content_sha256
        or fused.get("snapshot_role") != "capture_event_fused"
        or fused.get("fusion_level") != "intra_capture"
    ):
        raise ValueError("capture_event fused identity differs from the floorplan")
    _positive_int(fused.get("sequence"), "capture_event.fused_snapshot.sequence")
    _sha256(fused.get("manifest_sha256"), "capture_event.fused_snapshot.manifest_sha256")
    fusion_digest = _sha256(
        event.get("fusion_evidence_sha256"), "capture_event.fusion_evidence_sha256"
    )
    return {
        "capture_event_evidence_sha256": compact_digest,
        "fusion_evidence_sha256": fusion_digest,
        "capture_event_id": event_id,
        "capture_event_source_snapshot_count": raw_snapshot_count,
        "capture_event_rgb_status": rgb_status,
        "capture_event_rgb_evidence": deepcopy(dict(rgb)),
    }


def _floorplan_immutable_digest(
    payload: Mapping[str, Any],
    *,
    layer_records: Mapping[str, Mapping[str, object]],
) -> str:
    immutable: dict[str, object] = {
        "camera_id": payload.get("camera_id"),
        "ts": payload.get("ts"),
        "snapshot_ts": payload.get("snapshot_ts"),
        "snapshot_ref": payload.get("snapshot_ref"),
        "snapshot_id": payload.get("snapshot_id"),
        "snapshot_content_sha256": payload.get("snapshot_content_sha256"),
        "frame": payload.get("frame"),
        "orientation": payload.get("orientation"),
        "floorplan_contract_version": payload.get("floorplan_contract_version"),
        "units": payload.get("units"),
        "s_obj_to_m": payload.get("s_obj_to_m"),
        "bounds": payload.get("bounds"),
        "scale_m_per_px": payload.get("scale_m_per_px"),
        "scale_scene_per_px": payload.get("scale_scene_per_px"),
        "grid_res_m": payload.get("grid_res_m"),
        "grid_res_scene": payload.get("grid_res_scene"),
        "max_extent_m": payload.get("max_extent_m"),
        "point_count": payload.get("point_count"),
        "image_flip": payload.get("image_flip"),
        "calibration_fingerprint": payload.get("calibration_fingerprint"),
        "observation_meta": payload.get("observation_meta"),
        "layers": {
            name: dict(layer_records[name])
            for name in sorted(layer_records)
        },
    }
    if "ray_to_floorplan_alignment" in payload:
        immutable["ray_to_floorplan_alignment"] = payload[
            "ray_to_floorplan_alignment"
        ]
    return _canonical_json_sha256(immutable)


def _validate_floorplan_payload(
    payload: object,
    *,
    camera_id: str,
    request_id: str,
    max_age_sec: float,
    now_us: int | None = None,
    mode: str = "fresh",
    expected_fresh: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError("floorplan response must be an object")
    if payload.get("type") != "floorplan_response":
        raise ValueError("floorplan response type mismatch")
    if payload.get("request_id") != request_id:
        raise ValueError("floorplan response request_id mismatch")
    if payload.get("camera_id") != camera_id:
        raise ValueError("floorplan response camera_id mismatch")
    if payload.get("error"):
        raise ValueError(f"floorplan response error: {payload.get('error')}")
    if payload.get("frame") != FLOORPLAN_FRAME:
        raise ValueError("floorplan frame contract mismatch")
    if payload.get("orientation") != FLOORPLAN_ORIENTATION:
        raise ValueError("floorplan orientation contract mismatch")
    if payload.get("floorplan_contract_version") != FLOORPLAN_CONTRACT_VERSION:
        raise ValueError("floorplan contract version mismatch")
    if payload.get("units") != "meters":
        raise ValueError("floorplan units must be meters")
    snapshot_ts = _positive_int(payload.get("snapshot_ts"), "snapshot_ts")
    current_us = int(time.time() * 1_000_000) if now_us is None else int(now_us)
    age_us = current_us - snapshot_ts
    if age_us < -5_000_000:
        raise ValueError("floorplan snapshot timestamp is in the future")
    if age_us > int(max_age_sec * 1_000_000):
        raise ValueError("floorplan snapshot exceeds the requested age bound")
    floorplan_ts = _positive_int(payload.get("ts"), "floorplan ts")
    if floorplan_ts < snapshot_ts or floorplan_ts > current_us + 5_000_000:
        raise ValueError("floorplan timestamp is outside its exact snapshot window")
    point_count = _positive_int(payload.get("point_count"), "point_count")
    _finite_float(payload.get("scale_m_per_px"), "scale_m_per_px", positive=True)
    _finite_float(payload.get("scale_scene_per_px"), "scale_scene_per_px", positive=True)
    grid_res_m = _finite_float(
        payload.get("grid_res_m"), "grid_res_m", positive=True
    )
    _finite_float(payload.get("grid_res_scene"), "grid_res_scene", positive=True)
    _finite_float(payload.get("max_extent_m"), "max_extent_m", positive=True)
    _finite_float(payload.get("s_obj_to_m"), "s_obj_to_m", positive=True)
    fingerprint = _sha256(
        payload.get("calibration_fingerprint"), "calibration_fingerprint"
    )
    snapshot_ref = _portable_snapshot_ref(payload.get("snapshot_ref"))
    snapshot_id = _nonempty_text(payload.get("snapshot_id"), "snapshot_id")
    snapshot_content_sha256 = _sha256(
        payload.get("snapshot_content_sha256"), "snapshot_content_sha256"
    )
    served_from_cache = payload.get("served_from_cache")
    if type(served_from_cache) is not bool:  # noqa: E721
        raise ValueError("served_from_cache must be boolean")
    image_flip = payload.get("image_flip")
    if (
        not isinstance(image_flip, Mapping)
        or set(image_flip) != {"u", "v"}
        or type(image_flip.get("u")) is not bool  # noqa: E721
        or type(image_flip.get("v")) is not bool  # noqa: E721
    ):
        raise ValueError("image_flip must contain exactly boolean u/v fields")
    bounds = payload.get("bounds")
    if not isinstance(bounds, Mapping) or set(bounds) != {"min_x", "max_x", "min_z", "max_z"}:
        raise ValueError("bounds must contain exactly min/max X/Z")
    min_x = _finite_float(bounds.get("min_x"), "bounds.min_x")
    max_x = _finite_float(bounds.get("max_x"), "bounds.max_x")
    min_z = _finite_float(bounds.get("min_z"), "bounds.min_z")
    max_z = _finite_float(bounds.get("max_z"), "bounds.max_z")
    if max_x <= min_x or max_z <= min_z:
        raise ValueError("floorplan bounds must have positive extents")
    normalized_bounds = {
        "min_x": min_x,
        "max_x": max_x,
        "min_z": min_z,
        "max_z": max_z,
    }
    alignment = payload.get("ray_to_floorplan_alignment")
    if alignment is not None and not isinstance(alignment, Mapping):
        raise ValueError("ray_to_floorplan_alignment must be an object or null")

    shapes: dict[str, tuple[int, int]] = {}
    decoded: dict[str, array.array] = {}
    layer_sha256s: dict[str, str] = {}
    layer_records: dict[str, Mapping[str, object]] = {}
    for layer_name in REQUIRED_LAYERS:
        shape, values, digest, digest_record = _decode_grid(
            payload.get(layer_name),
            layer_name,
            allow_canonical_quiet_nan=layer_name == "height_agl",
        )
        shapes[layer_name] = shape
        decoded[layer_name] = values
        layer_sha256s[layer_name] = digest
        layer_records[layer_name] = digest_record
    for layer_name in OPTIONAL_IMMUTABLE_LAYERS:
        if layer_name not in payload:
            continue
        shape, values, digest, digest_record = _decode_grid(
            payload.get(layer_name), layer_name
        )
        shapes[layer_name] = shape
        decoded[layer_name] = values
        layer_sha256s[layer_name] = digest
        layer_records[layer_name] = digest_record
    if len(set(shapes.values())) != 1:
        raise ValueError("required floorplan layers must use one grid shape")
    grid_shape = shapes["density"]
    if grid_shape == (1, 1):
        raise ValueError("floorplan must not use the zero-density sentinel grid")
    if not any(float(value) > 0.0 for value in decoded["density"]):
        raise ValueError("floorplan density grid must contain occupied cells")
    for layer_name in ("observed", "unknown"):
        if (
            layer_records[layer_name]["value_min"] != 0.0
            or layer_records[layer_name]["value_max"] != 1.0
            or any(float(value) not in {0.0, 1.0} for value in decoded[layer_name])
        ):
            raise ValueError(f"{layer_name} must be an exact binary grid")
    if any(
        float(observed) + float(unknown) != 1.0
        for observed, unknown in zip(decoded["observed"], decoded["unknown"])
    ):
        raise ValueError("observed and unknown grids must be exact complements")
    _validate_height_agl_observation_policy(
        decoded["height_agl"],
        decoded["observed"],
        decoded["unknown"],
    )
    if any(
        (float(density) > 0.0) != (float(observed) == 1.0)
        for density, observed in zip(decoded["density"], decoded["observed"])
    ):
        raise ValueError(
            "observed must identify exactly the cells with projected-point density"
        )
    observation_meta = _exact_mapping(
        payload.get("observation_meta"),
        OBSERVATION_META_KEYS,
        "observation_meta",
    )
    observed_cells = sum(float(value) == 1.0 for value in decoded["observed"])
    unknown_cells = sum(float(value) == 1.0 for value in decoded["unknown"])
    total_cells = grid_shape[0] * grid_shape[1]
    if (
        observation_meta.get("contract") != "noesis.floorplan.observation.v1"
        or observation_meta.get("observed_definition")
        != "one_or_more_valid_projected_depth_points"
        or observation_meta.get("unknown_definition")
        != "zero_valid_projected_depth_points_within_grid_bounds"
        or _nonnegative_int(
            observation_meta.get("observed_cells"),
            "observation_meta.observed_cells",
        )
        != observed_cells
        or _nonnegative_int(
            observation_meta.get("unknown_cells"),
            "observation_meta.unknown_cells",
        )
        != unknown_cells
        or _positive_int(
            observation_meta.get("total_cells"),
            "observation_meta.total_cells",
        )
        != total_cells
        or observed_cells + unknown_cells != total_cells
    ):
        raise ValueError("floorplan observation metadata does not match its grids")
    has_walkable = "walkable" in decoded
    has_inferred_walkable = "inferred_walkable" in decoded
    if has_walkable != has_inferred_walkable:
        raise ValueError(
            "inferred_walkable must be present exactly when walkable is present"
        )
    if has_walkable:
        for layer_name in ("walkable", "inferred_walkable"):
            if (
                layer_records[layer_name]["value_min"] != 0.0
                or layer_records[layer_name]["value_max"] != 1.0
                or any(
                    float(value) not in {0.0, 1.0}
                    for value in decoded[layer_name]
                )
            ):
                raise ValueError(f"{layer_name} must be an exact binary grid")
        for walkable, unknown, inferred in zip(
            decoded["walkable"],
            decoded["unknown"],
            decoded["inferred_walkable"],
        ):
            expected_inferred = float(walkable) == 1.0 and float(unknown) == 1.0
            if (float(inferred) == 1.0) != expected_inferred:
                raise ValueError(
                    "inferred_walkable must equal walkable intersect unknown"
                )
    floorplan_digest = _floorplan_immutable_digest(
        payload, layer_records=layer_records
    )

    common: dict[str, object] = {
        "camera_id": camera_id,
        "snapshot_age_s": max(0.0, age_us / 1_000_000.0),
        "snapshot_ts_us": snapshot_ts,
        "snapshot_ref": snapshot_ref,
        "snapshot_id": snapshot_id,
        "snapshot_content_sha256": snapshot_content_sha256,
        "floorplan_ts_us": floorplan_ts,
        "point_count": point_count,
        "grid_shape": [grid_shape[0], grid_shape[1]],
        "grid_res_m": grid_res_m,
        "bounds": normalized_bounds,
        "ray_to_floorplan_alignment": deepcopy(alignment),
        "served_from_cache": served_from_cache,
        "floorplan_contract_version": FLOORPLAN_CONTRACT_VERSION,
        "calibration_fingerprint": fingerprint,
        "floorplan_payload_sha256": floorplan_digest,
        "layer_sha256s": layer_sha256s,
        "observation_meta": deepcopy(dict(observation_meta)),
        "inferred_walkable_present": has_inferred_walkable,
    }
    if mode == "fresh":
        if served_from_cache is not False or payload.get("cache_only") is not False:
            raise ValueError("fresh floorplan must be a non-cache capture")
        if (
            payload.get("depth_burst_triggered") is not True
            or payload.get("depth_burst_fresh") is not True
        ):
            raise ValueError("fresh floorplan lacks exact depth-burst evidence")
        capture_summary = _validate_capture_event(
            payload.get("capture_event"),
            evidence_sha256=payload.get("capture_event_evidence_sha256"),
            camera_id=camera_id,
            snapshot_ref=snapshot_ref,
            snapshot_id=snapshot_id,
            snapshot_content_sha256=snapshot_content_sha256,
            snapshot_ts_us=snapshot_ts,
        )
        common.update(capture_summary)
        if set(common) != FRESH_RESULT_KEYS:
            raise ValueError("fresh floorplan evidence schema drifted")
        return common
    if mode != "cache_only" or expected_fresh is None:
        raise ValueError("floorplan validation mode is invalid")
    if served_from_cache is not True or payload.get("cache_only") is not True:
        raise ValueError("cache-only floorplan was not served from cache")
    for forbidden in (
        "capture_event",
        "capture_event_evidence_sha256",
        "depth_burst_triggered",
        "depth_burst_fresh",
    ):
        if forbidden in payload:
            raise ValueError("cache-only floorplan contains fresh-capture evidence")
    identity_keys = (
        "camera_id",
        "snapshot_ts_us",
        "snapshot_ref",
        "snapshot_id",
        "snapshot_content_sha256",
        "floorplan_ts_us",
        "calibration_fingerprint",
        "grid_res_m",
        "bounds",
        "ray_to_floorplan_alignment",
    )
    identity_unchanged = all(
        common.get(key) == expected_fresh.get(key) for key in identity_keys
    )
    payload_unchanged = (
        common["floorplan_payload_sha256"]
        == expected_fresh.get("floorplan_payload_sha256")
        and common["layer_sha256s"] == expected_fresh.get("layer_sha256s")
    )
    if not identity_unchanged or not payload_unchanged:
        raise ValueError("cache-only floorplan mutated the active immutable payload")
    common["identity_unchanged"] = True
    common["payload_unchanged"] = True
    if set(common) != CACHE_RESULT_KEYS:
        raise ValueError("cache-only floorplan evidence schema drifted")
    return common


def _copy_json_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    encoded = _canonical_json_bytes(value)
    if len(encoded) > MAX_REPORT_BYTES:
        raise ValueError(f"{label} exceeds its byte bound")
    decoded = strict_json_loads(encoded, label=label)
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be an object")
    return decoded


def _validate_health_result(
    result: object,
    *,
    camera_ids: Sequence[str],
    fresh_results: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    health = _exact_mapping(result, HEALTH_RESULT_KEYS, "runtime health result")
    expected = set(camera_ids)
    if health.get("bev_frame") != FLOORPLAN_FRAME:
        raise ValueError("runtime BEV frame is not camera_local_ground_m")

    active = health.get("active_floorplan_health")
    if not isinstance(active, Mapping):
        raise ValueError("active floorplan health is missing")
    active_cameras = active.get("cameras")
    if (
        active.get("contract") != "noesis.active_floorplan.health"
        or active.get("contract_version") != 1
        or active.get("healthy") is not True
        or not _is_int(active.get("configured_camera_count"))
        or active.get("configured_camera_count") != len(camera_ids)
        or not _is_int(active.get("active_camera_count"))
        or active.get("active_camera_count") != len(camera_ids)
        or active.get("missing_cameras") != []
        or not isinstance(active_cameras, Mapping)
        or set(active_cameras) != expected
    ):
        raise ValueError("active floorplan health lacks configured-camera N/N coverage")
    for camera_id in camera_ids:
        record = active_cameras.get(camera_id)
        fresh = fresh_results.get(camera_id)
        if not isinstance(record, Mapping) or not isinstance(fresh, Mapping):
            raise ValueError("active floorplan health camera evidence is missing")
        if (
            record.get("snapshot_ts_us") != fresh.get("snapshot_ts_us")
            or record.get("floorplan_ts_us") != fresh.get("floorplan_ts_us")
            or record.get("snapshot_ref") != fresh.get("snapshot_ref")
            or record.get("snapshot_id") != fresh.get("snapshot_id")
            or record.get("snapshot_content_sha256")
            != fresh.get("snapshot_content_sha256")
            or record.get("calibration_fingerprint")
            != fresh.get("calibration_fingerprint")
            or record.get("frame") != FLOORPLAN_FRAME
            or record.get("units") != "meters"
        ):
            raise ValueError("active floorplan health differs from exact capture evidence")

    controller = health.get("capture_event_controller_health")
    if not isinstance(controller, Mapping):
        raise ValueError("capture-event controller health is missing")
    controller_cameras = controller.get("cameras")
    counters = controller.get("counters")
    if (
        controller.get("contract") != "noesis.capture_event_controller_health"
        or controller.get("contract_version") != 1
        or controller.get("healthy") is not True
        or controller.get("canonical_camera_count") != len(camera_ids)
        or controller.get("shared_gate_scope") != "process"
        or controller.get("shared_gate_owned") is not False
        or controller.get("last_fatal_error_code") is not None
        or controller.get("request_kinds") != ["depth", "floorplan"]
        or not isinstance(controller_cameras, Mapping)
        or set(controller_cameras) != expected
        or not isinstance(counters, Mapping)
        or counters.get("fatal_barrier_failures_total") != 0
        or counters.get("cache_only_rejected_total") != 0
    ):
        raise ValueError("capture-event controller health is not quiescent and clean")
    for camera_id in camera_ids:
        camera = controller_cameras.get(camera_id)
        fresh = fresh_results[camera_id]
        if (
            not isinstance(camera, Mapping)
            or camera.get("active") is not False
            or camera.get("active_request_kind") is not None
            or camera.get("last_fused_timestamp_us") != fresh.get("snapshot_ts_us")
            or camera.get("last_fused_snapshot_id") != fresh.get("snapshot_id")
            or not _is_int(camera.get("completed_total"))
            or int(camera.get("completed_total")) <= 0
        ):
            raise ValueError("capture-event controller camera evidence is incomplete")

    bev = health.get("bev_health")
    if not isinstance(bev, Mapping):
        raise ValueError("BEV health is missing")
    bev_cameras = bev.get("cameras")
    bev_active_cameras = bev.get("active_cameras")
    bev_inactive_cameras = bev.get("inactive_cameras")
    if (
        bev.get("contract") != "noesis.bev.health"
        or bev.get("contract_version") != 2
        or bev.get("healthy") is not True
        or bev.get("config_ready") is not True
        or bev.get("renderer_ready") is not True
        or bev.get("configured_camera_count") != len(camera_ids)
        or not _is_int(bev.get("active_camera_count"))
        or not _is_int(bev.get("inactive_camera_count"))
        or int(bev.get("active_camera_count"))
        + int(bev.get("inactive_camera_count"))
        != len(camera_ids)
        or not _is_int(bev.get("failed_camera_count"))
        or bev.get("failed_camera_count") != 0
        or set(bev.get("configured_cameras") or []) != expected
        or not isinstance(bev_active_cameras, list)
        or not isinstance(bev_inactive_cameras, list)
        or len(bev_active_cameras) != len(set(bev_active_cameras))
        or len(bev_inactive_cameras) != len(set(bev_inactive_cameras))
        or set(bev_active_cameras).intersection(
            set(bev_inactive_cameras)
        )
        or set(bev_active_cameras).union(set(bev_inactive_cameras))
        != expected
        or bev.get("failed_cameras") != []
        or bev.get("unexpected_cameras") != []
        or bev.get("missing_cameras") != []
        or bev.get("unhealthy_cameras") != []
        or not isinstance(bev_cameras, Mapping)
        or set(bev_cameras) != expected
        or any(
            not isinstance(value, Mapping)
            or value.get("healthy") is not True
            or value.get("configured") is not True
            or value.get("state") not in {"active_ready", "inactive_ready"}
            for value in bev_cameras.values()
        )
    ):
        raise ValueError("local BEV readiness/activity classification is invalid")
    for camera_id, value in bev_cameras.items():
        expected_active = camera_id in set(bev_active_cameras)
        success_count = value.get("success_count")
        failure_count = value.get("failure_count")
        if (
            value.get("active") is not expected_active
            or value.get("state")
            != ("active_ready" if expected_active else "inactive_ready")
            or not _is_int(success_count)
            or int(success_count) < 0
            or not _is_int(failure_count)
            or int(failure_count) < 0
            or (expected_active and int(success_count) <= 0)
            or (not expected_active and int(success_count) != 0)
            or (
                not expected_active
                and (
                    int(failure_count) != 0
                    or value.get("last_failure_ts_us") is not None
                    or value.get("last_failure_stage") is not None
                    or value.get("last_failure_type") is not None
                )
            )
        ):
            raise ValueError("local BEV camera activity row is inconsistent")
    return _copy_json_mapping(health, "runtime health result")


def _validate_stats_payload(
    message: object,
    *,
    camera_ids: Sequence[str],
    fresh_results: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if not isinstance(message, Mapping) or message.get("type") != "stats":
        raise ValueError("runtime health message must be stats")
    payload = message.get("payload")
    pipeline = payload.get("pipeline") if isinstance(payload, Mapping) else None
    if not isinstance(pipeline, Mapping):
        raise ValueError("runtime stats pipeline payload is missing")
    bev = pipeline.get("bev")
    if not isinstance(bev, Mapping):
        raise ValueError("runtime stats BEV payload is missing")
    result = {
        "bev_frame": bev.get("frame"),
        "bev_health": deepcopy(bev.get("health")),
        "active_floorplan_health": deepcopy(pipeline.get("active_floorplan")),
        "capture_event_controller_health": deepcopy(
            pipeline.get("capture_event_fusion")
        ),
    }
    return _validate_health_result(
        result,
        camera_ids=camera_ids,
        fresh_results=fresh_results,
    )


def _health_zero_mutation(
    before: Mapping[str, object], after: Mapping[str, object]
) -> bool:
    return (
        before.get("active_floorplan_health")
        == after.get("active_floorplan_health")
        and before.get("capture_event_controller_health")
        == after.get("capture_event_controller_health")
    )


async def _next_json(
    websocket: Any,
    *,
    deadline: float,
    counter: list[int],
) -> Mapping[str, Any]:
    while time.monotonic() < deadline:
        timeout = min(2.0, max(0.05, deadline - time.monotonic()))
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            continue
        counter[0] += 1
        if counter[0] > MAX_WS_MESSAGES_SEEN:
            raise ValueError("WebSocket observation message bound exceeded")
        if not isinstance(raw, str):
            continue
        try:
            payload = strict_json_loads(raw, label="floorplan WebSocket message")
        except ValueError as exc:
            raise ValueError("floorplan WebSocket emitted invalid strict JSON") from exc
        if isinstance(payload, Mapping):
            return payload
    raise TimeoutError("WebSocket evidence deadline expired")


async def _request_floorplan(
    websocket: Any,
    *,
    camera_id: str,
    camera_index: int,
    cache_only: bool,
    max_age_sec: float,
    max_attempts: int,
    deadline: float,
    counter: list[int],
) -> tuple[Mapping[str, Any], str, int]:
    for attempt in range(1, max_attempts + 1):
        mode = "cache" if cache_only else "fresh"
        request_id = f"ds9-floorplan-v2-{mode}-{camera_index}-{attempt}"
        await websocket.send(
            json.dumps(
                {
                    "type": "get_floorplan",
                    "camera": camera_id,
                    "request_id": request_id,
                    "max_age_sec": float(max_age_sec),
                    "cache_only": cache_only,
                },
                separators=(",", ":"),
            )
        )
        while time.monotonic() < deadline:
            payload = await _next_json(
                websocket, deadline=deadline, counter=counter
            )
            if (
                payload.get("type") != "floorplan_response"
                or payload.get("request_id") != request_id
            ):
                continue
            error = str(payload.get("error") or "").strip()
            if error and error in RETRIABLE_ERRORS and attempt < max_attempts:
                break
            return payload, request_id, attempt
    raise TimeoutError(f"{camera_id}: no floorplan response before deadline")


async def _wait_for_health(
    websocket: Any,
    *,
    camera_ids: Sequence[str],
    fresh_results: Mapping[str, Mapping[str, object]],
    deadline: float,
    counter: list[int],
    baseline_health: Mapping[str, object] | None = None,
) -> dict[str, object]:
    last_error = "no stats message observed"
    while time.monotonic() < deadline:
        try:
            payload = await _next_json(
                websocket, deadline=deadline, counter=counter
            )
        except TimeoutError:
            break
        if payload.get("type") != "stats":
            continue
        try:
            health = _validate_stats_payload(
                payload,
                camera_ids=camera_ids,
                fresh_results=fresh_results,
            )
            if baseline_health is not None and not _health_zero_mutation(
                baseline_health, health
            ):
                raise ValueError(
                    "cache-only request mutated active registry or capture controller"
                )
            return health
        except Exception as exc:
            last_error = str(exc)
    raise ValueError(f"no qualifying runtime health snapshot: {last_error}")


async def _collect_floorplans(
    uri: str,
    auth: RequiredInternalAuth,
    *,
    camera_ids: tuple[str, ...],
    max_age_sec: float,
    timeout_s: float,
    max_attempts: int,
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, object] | None,
    dict[str, object] | None,
    list[str],
    list[dict[str, object]],
]:
    deadline = time.monotonic() + timeout_s
    fresh_results: dict[str, dict[str, object]] = {}
    cache_results: dict[str, dict[str, object]] = {}
    source_events: list[dict[str, object]] = []
    errors: list[str] = []
    before_cache_health: dict[str, object] | None = None
    after_cache_health: dict[str, object] | None = None
    message_counter = [0]
    last_event_timestamp_us = 0

    def observed_at_us() -> int:
        nonlocal last_event_timestamp_us
        last_event_timestamp_us = max(
            time.time_ns() // 1_000,
            last_event_timestamp_us + 1,
        )
        return last_event_timestamp_us

    async with connect_required_websocket(
        uri,
        auth,
        compression=None,
        max_size=MAX_WS_MESSAGE_BYTES,
        open_timeout=10.0,
        close_timeout=5.0,
    ) as websocket:
        # Fresh requests are deliberately serialized: all cameras share one
        # process-owned MapAnything valve and overlapping captures are invalid.
        for camera_index, camera_id in enumerate(camera_ids):
            try:
                payload, request_id, attempt = await _request_floorplan(
                    websocket,
                    camera_id=camera_id,
                    camera_index=camera_index,
                    cache_only=False,
                    max_age_sec=max_age_sec,
                    max_attempts=max_attempts,
                    deadline=deadline,
                    counter=message_counter,
                )
                result = _validate_floorplan_payload(
                    payload,
                    camera_id=camera_id,
                    request_id=request_id,
                    max_age_sec=max_age_sec,
                    mode="fresh",
                )
                fresh_results[camera_id] = result
                source_events.append(
                    {
                        "type": "validated_exact_floorplan_capture",
                        "observed_at_us": observed_at_us(),
                        "request_id": request_id,
                        "attempt": attempt,
                        "camera_id": camera_id,
                        "result": dict(result),
                        "capture_event": deepcopy(payload["capture_event"]),
                    }
                )
            except Exception as exc:
                errors.append(f"{camera_id}: {exc}")
                break

        if len(fresh_results) == len(camera_ids):
            try:
                before_cache_health = await _wait_for_health(
                    websocket,
                    camera_ids=camera_ids,
                    fresh_results=fresh_results,
                    deadline=deadline,
                    counter=message_counter,
                )
                source_events.append(
                    {
                        "type": "validated_floorplan_runtime_health",
                        "observed_at_us": observed_at_us(),
                        "phase": "after_fresh",
                        "result": deepcopy(before_cache_health),
                    }
                )
            except Exception as exc:
                errors.append(f"runtime health after fresh captures: {exc}")

        if before_cache_health is not None:
            for camera_index, camera_id in enumerate(camera_ids):
                try:
                    payload, request_id, attempt = await _request_floorplan(
                        websocket,
                        camera_id=camera_id,
                        camera_index=camera_index,
                        cache_only=True,
                        max_age_sec=max_age_sec,
                        max_attempts=1,
                        deadline=deadline,
                        counter=message_counter,
                    )
                    result = _validate_floorplan_payload(
                        payload,
                        camera_id=camera_id,
                        request_id=request_id,
                        max_age_sec=max_age_sec,
                        mode="cache_only",
                        expected_fresh=fresh_results[camera_id],
                    )
                    cache_results[camera_id] = result
                    source_events.append(
                        {
                            "type": "validated_cache_only_floorplan",
                            "observed_at_us": observed_at_us(),
                            "request_id": request_id,
                            "attempt": attempt,
                            "camera_id": camera_id,
                            "result": dict(result),
                        }
                    )
                except Exception as exc:
                    errors.append(f"{camera_id} cache-only: {exc}")
                    break

        if len(cache_results) == len(camera_ids) and before_cache_health is not None:
            try:
                after_cache_health = await _wait_for_health(
                    websocket,
                    camera_ids=camera_ids,
                    fresh_results=fresh_results,
                    deadline=deadline,
                    counter=message_counter,
                    baseline_health=before_cache_health,
                )
                source_events.append(
                    {
                        "type": "validated_floorplan_runtime_health",
                        "observed_at_us": observed_at_us(),
                        "phase": "after_cache_only",
                        "result": deepcopy(after_cache_health),
                    }
                )
            except Exception as exc:
                errors.append(f"runtime health after cache-only checks: {exc}")

    if len(source_events) > MAX_SOURCE_MESSAGES:
        errors.append("floorplan source transcript message bound exceeded")
    return (
        fresh_results,
        cache_results,
        before_cache_health,
        after_cache_health,
        errors,
        source_events,
    )


def _replay_source_events(
    events: Sequence[object],
    *,
    camera_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    if len(events) > MAX_SOURCE_MESSAGES:
        raise ValueError("floorplan source transcript message bound exceeded")
    expected_camera_ids = tuple(camera_ids or ())
    fresh_results: dict[str, dict[str, object]] = {}
    cache_results: dict[str, dict[str, object]] = {}
    before_health: dict[str, object] | None = None
    after_health: dict[str, object] | None = None
    phase = "fresh"
    last_observed_at_us = 0
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"floorplan source event {index} must be an object")
        observed_at_us = event.get("observed_at_us")
        if not _is_int(observed_at_us) or int(observed_at_us) <= 0:
            raise ValueError(f"floorplan source event {index} timestamp is invalid")
        if int(observed_at_us) <= last_observed_at_us:
            raise ValueError("floorplan source event timestamps must strictly advance")
        last_observed_at_us = int(observed_at_us)
        event_type = event.get("type")
        if event_type == "validated_exact_floorplan_capture":
            if phase != "fresh" or set(event) != {
                "type", "observed_at_us", "request_id", "attempt", "camera_id", "result", "capture_event"
            }:
                raise ValueError(f"floorplan source event {index} sequence/schema drifted")
            camera_id = _nonempty_text(event.get("camera_id"), f"floorplan source event {index}.camera_id")
            if (
                expected_camera_ids
                and (
                    len(fresh_results) >= len(expected_camera_ids)
                    or camera_id != expected_camera_ids[len(fresh_results)]
                )
            ):
                raise ValueError("fresh floorplan source camera order drifted")
            _nonempty_text(event.get("request_id"), f"floorplan source event {index}.request_id")
            _positive_int(event.get("attempt"), f"floorplan source event {index}.attempt")
            result = _exact_mapping(event.get("result"), FRESH_RESULT_KEYS, f"floorplan source event {index}.result")
            if result.get("camera_id") != camera_id or camera_id in fresh_results:
                raise ValueError("floorplan source transcript fresh camera drifted")
            summary = _validate_capture_event(
                event.get("capture_event"),
                evidence_sha256=result.get("capture_event_evidence_sha256"),
                camera_id=camera_id,
                snapshot_ref=_portable_snapshot_ref(result.get("snapshot_ref")),
                snapshot_id=_nonempty_text(result.get("snapshot_id"), "snapshot_id"),
                snapshot_content_sha256=_sha256(result.get("snapshot_content_sha256"), "snapshot_content_sha256"),
                snapshot_ts_us=_positive_int(result.get("snapshot_ts_us"), "snapshot_ts_us"),
            )
            if any(result.get(key) != value for key, value in summary.items()):
                raise ValueError("floorplan source capture summary drifted")
            fresh_results[camera_id] = dict(result)
        elif event_type == "validated_floorplan_runtime_health":
            if set(event) != {"type", "observed_at_us", "phase", "result"}:
                raise ValueError(f"floorplan source event {index} health schema drifted")
            health_phase = event.get("phase")
            derived_cameras = expected_camera_ids or tuple(fresh_results)
            if health_phase == "after_fresh" and phase == "fresh":
                if set(fresh_results) != set(derived_cameras):
                    raise ValueError("floorplan health precedes complete fresh evidence")
                before_health = _validate_health_result(
                    event.get("result"),
                    camera_ids=derived_cameras,
                    fresh_results=fresh_results,
                )
                phase = "cache"
            elif health_phase == "after_cache_only" and phase == "cache_done":
                after_health = _validate_health_result(
                    event.get("result"),
                    camera_ids=derived_cameras,
                    fresh_results=fresh_results,
                )
                if before_health is None or not _health_zero_mutation(
                    before_health, after_health
                ):
                    raise ValueError("cache-only health mutation detected during replay")
                phase = "done"
            else:
                raise ValueError("floorplan source health phase ordering drifted")
        elif event_type == "validated_cache_only_floorplan":
            if phase != "cache" or set(event) != {
                "type", "observed_at_us", "request_id", "attempt", "camera_id", "result"
            }:
                raise ValueError(f"floorplan source event {index} cache schema drifted")
            camera_id = _nonempty_text(event.get("camera_id"), f"floorplan source event {index}.camera_id")
            if (
                expected_camera_ids
                and (
                    len(cache_results) >= len(expected_camera_ids)
                    or camera_id != expected_camera_ids[len(cache_results)]
                )
            ):
                raise ValueError("cache-only floorplan source camera order drifted")
            _nonempty_text(event.get("request_id"), f"floorplan source event {index}.request_id")
            if _positive_int(event.get("attempt"), f"floorplan source event {index}.attempt") != 1:
                raise ValueError("cache-only evidence must use exactly one request")
            result = _exact_mapping(event.get("result"), CACHE_RESULT_KEYS, f"floorplan source event {index}.result")
            fresh = fresh_results.get(camera_id)
            if not isinstance(fresh, Mapping) or camera_id in cache_results:
                raise ValueError("cache-only source camera drifted")
            for key in (
                "snapshot_ts_us", "snapshot_ref", "snapshot_id", "snapshot_content_sha256",
                "floorplan_ts_us", "calibration_fingerprint", "grid_res_m", "bounds",
                "ray_to_floorplan_alignment", "floorplan_payload_sha256", "layer_sha256s",
                "observation_meta", "inferred_walkable_present",
            ):
                if result.get(key) != fresh.get(key):
                    raise ValueError("cache-only source result mutated immutable evidence")
            if (
                result.get("served_from_cache") is not True
                or result.get("identity_unchanged") is not True
                or result.get("payload_unchanged") is not True
            ):
                raise ValueError("cache-only source result does not prove zero mutation")
            cache_results[camera_id] = dict(result)
            if set(cache_results) == set(derived_cameras := (expected_camera_ids or tuple(fresh_results))):
                phase = "cache_done"
        else:
            raise ValueError(f"floorplan source event {index} type drifted")
    derived = expected_camera_ids or tuple(fresh_results)
    if (
        phase != "done"
        or set(fresh_results) != set(derived)
        or set(cache_results) != set(derived)
        or before_health is None
        or after_health is None
    ):
        raise ValueError("floorplan source transcript is incomplete")
    return {
        "fresh_results": fresh_results,
        "cache_results": cache_results,
        "after_fresh_health": before_health,
        "after_cache_only_health": after_health,
    }


def _source_transcript_document(
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    camera_ids: Sequence[str],
    max_snapshot_age_s: float,
    events: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": SOURCE_TRANSCRIPT_VERSION,
        "contract": SOURCE_TRANSCRIPT_CONTRACT,
        "contract_version": SOURCE_TRANSCRIPT_VERSION,
        "session_id": str(session_id).strip().lower(),
        "runtime_lane": str(runtime_lane).strip().lower(),
        "runtime_instance_id": str(runtime_instance_id),
        "runtime_run_id": str(runtime_run_id),
        "camera_ids": [str(value) for value in camera_ids],
        "max_snapshot_age_s": float(max_snapshot_age_s),
        "privacy": dict(SOURCE_PRIVACY_POLICY),
        "message_count": len(events),
        "messages": [deepcopy(dict(event)) for event in events],
    }


def _encoded_private_json(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _source_evidence_metadata(
    *,
    encoded: bytes,
    document: Mapping[str, object],
) -> dict[str, object]:
    raw_messages = document.get("messages")
    messages = raw_messages if isinstance(raw_messages, list) else []
    timestamps = [
        int(event["observed_at_us"])
        for event in messages
        if isinstance(event, Mapping)
        and _is_int(event.get("observed_at_us"))
    ]
    return {
        "filename": CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "message_count": document.get("message_count"),
        "first_observed_at_us": min(timestamps) if timestamps else None,
        "last_observed_at_us": max(timestamps) if timestamps else None,
    }


def _read_private_json_document(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> tuple[dict[str, object], bytes]:
    raw = read_private_file(path, label=label, max_bytes=max_bytes)
    if not raw:
        raise ValueError(f"{label} is empty")
    decoded = strict_json_loads(raw, label=label)
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be an object")
    if raw != _encoded_private_json(decoded):
        raise ValueError(f"{label} is not in canonical private JSON form")
    return decoded, raw


def load_and_validate_sealed_authority(
    report_path: Path,
    source_path: Path,
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
) -> dict[str, object]:
    """Replay an owner-private floorplan report/source pair exactly."""

    report, _report_raw = _read_private_json_document(
        report_path,
        label="floorplan report",
        max_bytes=MAX_REPORT_BYTES,
    )
    source, source_raw = _read_private_json_document(
        source_path,
        label="floorplan source transcript",
        max_bytes=MAX_SOURCE_TRANSCRIPT_BYTES,
    )
    _exact_mapping(report, REPORT_KEYS, "floorplan report")
    _exact_mapping(source, SOURCE_DOCUMENT_KEYS, "floorplan source transcript")
    expected_session = str(session_id).strip().lower()
    expected_lane = str(runtime_lane).strip().lower()
    expected_identity = {
        "session_id": expected_session,
        "runtime_lane": expected_lane,
        "runtime_instance_id": str(runtime_instance_id),
        "runtime_run_id": str(runtime_run_id),
    }
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("contract") != CONTRACT
        or report.get("contract_version") != CONTRACT_VERSION
        or report.get("ok") is not True
        or report.get("errors") != []
        or any(report.get(key) != value for key, value in expected_identity.items())
    ):
        raise ValueError("floorplan report header or runtime identity drifted")
    if (
        source.get("schema_version") != SOURCE_TRANSCRIPT_VERSION
        or source.get("contract") != SOURCE_TRANSCRIPT_CONTRACT
        or source.get("contract_version") != SOURCE_TRANSCRIPT_VERSION
        or source.get("privacy") != SOURCE_PRIVACY_POLICY
        or any(source.get(key) != value for key, value in expected_identity.items())
    ):
        raise ValueError("floorplan source header, privacy, or runtime identity drifted")
    camera_values = source.get("camera_ids")
    if (
        not isinstance(camera_values, list)
        or not camera_values
        or any(not isinstance(value, str) or not value for value in camera_values)
        or len(camera_values) != len(set(camera_values))
    ):
        raise ValueError("floorplan source camera inventory is invalid")
    camera_ids = tuple(camera_values)
    max_age = _finite_float(
        source.get("max_snapshot_age_s"),
        "floorplan source max_snapshot_age_s",
        positive=True,
    )
    if report.get("max_snapshot_age_s") != max_age:
        raise ValueError("floorplan report/source freshness bound drifted")
    messages = source.get("messages")
    if (
        not isinstance(messages, list)
        or source.get("message_count") != len(messages)
    ):
        raise ValueError("floorplan source message count drifted")
    replay = _replay_source_events(messages, camera_ids=camera_ids)
    source_metadata = _source_evidence_metadata(
        encoded=source_raw,
        document=source,
    )
    _exact_mapping(
        report.get("source_evidence"),
        SOURCE_EVIDENCE_KEYS,
        "floorplan report source_evidence",
    )
    if report.get("source_evidence") != source_metadata:
        raise ValueError("floorplan source checksum/metadata drifted")
    recomputed = _build_report(
        session_id=expected_session,
        runtime_lane=expected_lane,
        runtime_instance_id=str(runtime_instance_id),
        runtime_run_id=str(runtime_run_id),
        camera_ids=camera_ids,
        results=replay["fresh_results"],
        cache_results=replay["cache_results"],
        after_fresh_health=replay["after_fresh_health"],
        after_cache_only_health=replay["after_cache_only_health"],
        max_snapshot_age_s=max_age,
        errors=[],
        source_evidence=source_metadata,
    )
    if _canonical_json_bytes(report) != _canonical_json_bytes(recomputed):
        raise ValueError("floorplan report does not exactly recompute from its source")
    return deepcopy(report)


def _build_report(
    *,
    session_id: str,
    runtime_lane: str,
    runtime_instance_id: str,
    runtime_run_id: str,
    camera_ids: tuple[str, ...],
    results: Mapping[str, Mapping[str, object]],
    cache_results: Mapping[str, Mapping[str, object]],
    after_fresh_health: Mapping[str, object] | None,
    after_cache_only_health: Mapping[str, object] | None,
    max_snapshot_age_s: float,
    errors: list[str],
    source_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    session_id = str(session_id).strip().lower()
    runtime_lane = str(runtime_lane).strip().lower()
    if SESSION_RE.fullmatch(session_id) is None:
        raise ValueError("session_id must match [a-z0-9][a-z0-9-]{5,47}")
    if runtime_lane != "baseline":
        raise ValueError("floorplan depth-quality evidence requires runtime lane 'baseline'")
    if (
        RUNTIME_ID_RE.fullmatch(str(runtime_instance_id)) is None
        or RUNTIME_ID_RE.fullmatch(str(runtime_run_id)) is None
    ):
        raise ValueError("runtime instance/run identity is invalid")
    freshness_bound = float(max_snapshot_age_s)
    combined_errors = list(errors)
    expected = set(camera_ids)
    if not math.isfinite(freshness_bound) or freshness_bound <= 0.0:
        combined_errors.append("max snapshot age must be finite and positive")
    if set(results) != expected:
        combined_errors.append("fresh exact capture evidence does not cover every camera")
    if set(cache_results) != expected:
        combined_errors.append("cache-only evidence does not cover every camera")
    for camera_id, result in results.items():
        age = result.get("snapshot_age_s")
        if (
            isinstance(age, bool)
            or not isinstance(age, (int, float))
            or not math.isfinite(float(age))
            or not 0.0 <= float(age) <= freshness_bound
        ):
            combined_errors.append(f"{camera_id}: snapshot age is invalid")
    zero_mutation = bool(
        after_fresh_health is not None
        and after_cache_only_health is not None
        and _health_zero_mutation(after_fresh_health, after_cache_only_health)
        and all(
            value.get("identity_unchanged") is True
            and value.get("payload_unchanged") is True
            for value in cache_results.values()
        )
    )
    if not zero_mutation:
        combined_errors.append("cache-only zero-mutation proof is incomplete")
    bev_active_count = 0
    bev_inactive_count = 0
    bev_failed_count = 0
    bev_renderer_ready = False
    if isinstance(after_cache_only_health, Mapping):
        bev = after_cache_only_health.get("bev_health")
        if isinstance(bev, Mapping):
            bev_active_count = int(bev.get("active_camera_count", 0) or 0)
            bev_inactive_count = int(bev.get("inactive_camera_count", 0) or 0)
            bev_failed_count = int(bev.get("failed_camera_count", 0) or 0)
            bev_renderer_ready = bool(bev.get("renderer_ready"))
    all_bev_ready = bool(
        len(camera_ids) > 0
        and bev_renderer_ready
        and bev_failed_count == 0
        and bev_active_count + bev_inactive_count == len(camera_ids)
    )
    if not all_bev_ready:
        combined_errors.append("configured-camera local BEV readiness is incomplete")
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "session_id": session_id,
        "runtime_lane": runtime_lane,
        "runtime_instance_id": str(runtime_instance_id),
        "runtime_run_id": str(runtime_run_id),
        "ok": not combined_errors,
        "max_snapshot_age_s": freshness_bound,
        "configured_camera_count": len(camera_ids),
        "validated_camera_count": len(results),
        "all_configured_camera_floorplans_validated": set(results) == expected,
        "exact_capture_event_camera_count": len(results),
        "cache_only_validated_camera_count": len(cache_results),
        "cache_only_zero_mutation": zero_mutation,
        "bev_renderer_ready": bev_renderer_ready,
        "bev_active_camera_count": bev_active_count,
        "bev_inactive_ready_camera_count": bev_inactive_count,
        "bev_failed_camera_count": bev_failed_count,
        "all_configured_cameras_bev_ready": all_bev_ready,
        "cameras": [dict(results[camera_id]) for camera_id in camera_ids if camera_id in results],
        "cache_only_cameras": [dict(cache_results[camera_id]) for camera_id in camera_ids if camera_id in cache_results],
        "runtime_health": {
            "after_fresh": deepcopy(after_fresh_health),
            "after_cache_only": deepcopy(after_cache_only_health),
        },
        "source_evidence": dict(source_evidence or {}),
        "errors": combined_errors,
    }
    return report


def _write_private_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    max_bytes: int = MAX_REPORT_BYTES,
) -> None:
    path = path.expanduser().absolute()
    if path.name not in {
        CANONICAL_REPORT_FILENAME,
        CANONICAL_SOURCE_TRANSCRIPT_FILENAME,
    }:
        raise ValueError("floorplan evidence output filename is not canonical")
    encoded = _encoded_private_json(payload)
    try:
        atomic_create_private_file(
            path,
            encoded,
            label=f"immutable floorplan evidence {path.name}",
            max_bytes=max_bytes,
        )
    except PrivatePathError as exc:
        raise RuntimeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prove exact capture-event floorplans, active registry/BEV health, "
            "and cache-only zero mutation for every configured DS9 camera."
        )
    )
    parser.add_argument("--ws", default="ws://127.0.0.1:6008")
    parser.add_argument("--pipeline-config", type=Path, default=Path("DS9/config/infer.yaml"))
    parser.add_argument("--cameras-config", type=Path, default=Path("config/cameras.yaml"))
    parser.add_argument("--max-age-sec", type=float, default=120.0)
    parser.add_argument("--timeout-s", type=float, default=75.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--runtime-lane", choices=("baseline",), required=True)
    parser.add_argument("--runtime-instance-id", required=True)
    parser.add_argument("--runtime-run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-out", type=Path, required=True)
    add_auth_token_file_argument(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require_fresh_private_file_bundle(
            (args.source_out, args.out),
            label="floorplan behavior evidence bundle",
        )
    except (PrivatePathError, ValueError) as exc:
        print(f"[FAIL] floorplan evidence session is not fresh: {exc}", file=sys.stderr)
        return 1
    camera_ids: tuple[str, ...] = ()
    results: dict[str, dict[str, object]] = {}
    cache_results: dict[str, dict[str, object]] = {}
    after_fresh_health: dict[str, object] | None = None
    after_cache_only_health: dict[str, object] | None = None
    errors: list[str] = []
    source_events: list[dict[str, object]] = []
    source_evidence: dict[str, object] = {}
    source_published = False
    max_age = 0.0
    try:
        max_age = _finite_float(args.max_age_sec, "max-age-sec", positive=True)
        timeout = _finite_float(args.timeout_s, "timeout-s", positive=True)
        attempts = _positive_int(args.max_attempts, "max-attempts")
        if args.out.expanduser().absolute() == args.source_out.expanduser().absolute():
            raise ValueError("report and source transcript paths must be distinct")
        camera_ids = _active_camera_ids(args.pipeline_config, args.cameras_config)
        auth = load_required_internal_auth(args.auth_token_file)
        (
            results,
            cache_results,
            after_fresh_health,
            after_cache_only_health,
            errors,
            source_events,
        ) = asyncio.run(
            _collect_floorplans(
                args.ws,
                auth,
                camera_ids=camera_ids,
                max_age_sec=max_age,
                timeout_s=timeout,
                max_attempts=attempts,
            )
        )
        if not errors:
            _replay_source_events(source_events, camera_ids=camera_ids)
        source_document = _source_transcript_document(
            session_id=args.session_id,
            runtime_lane=args.runtime_lane,
            runtime_instance_id=args.runtime_instance_id,
            runtime_run_id=args.runtime_run_id,
            camera_ids=camera_ids,
            max_snapshot_age_s=max_age,
            events=source_events,
        )
        source_encoded = _encoded_private_json(source_document)
        if len(source_encoded) > MAX_SOURCE_TRANSCRIPT_BYTES:
            raise ValueError("floorplan source transcript byte bound exceeded")
        _write_private_json(
            args.source_out,
            source_document,
            max_bytes=MAX_SOURCE_TRANSCRIPT_BYTES,
        )
        source_published = True
        source_evidence = _source_evidence_metadata(
            encoded=source_encoded,
            document=source_document,
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    if not source_published:
        print(
            "[FAIL] floorplan source evidence was not published; use a fresh session",
            file=sys.stderr,
        )
        return 1
    report = _build_report(
        session_id=args.session_id,
        runtime_lane=args.runtime_lane,
        runtime_instance_id=args.runtime_instance_id,
        runtime_run_id=args.runtime_run_id,
        camera_ids=camera_ids,
        results=results,
        cache_results=cache_results,
        after_fresh_health=after_fresh_health,
        after_cache_only_health=after_cache_only_health,
        max_snapshot_age_s=max_age,
        errors=errors,
        source_evidence=source_evidence,
    )
    try:
        _write_private_json(args.out, report)
    except Exception as exc:
        print(f"[FAIL] unable to write floorplan gate report: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
