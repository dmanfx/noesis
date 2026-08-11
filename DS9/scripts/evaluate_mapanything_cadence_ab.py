#!/usr/bin/env python3
"""Compare two exact fresh MapAnything cadence sessions without runtime access.

Each arm root must contain a self-hashed cadence arm receipt, the sealed DS9
floorplan live-gate report/source pair, a byte-exact copy of the active
MapAnything nvinfer config, and a self-contained committed depth store.  The
analyzer revalidates every manifest and array before recomputing fusion.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DS9.scripts import ds9_floorplan_live_gate as live_gate  # noqa: E402
from geometry.depth_source import DepthStorageManager  # noqa: E402
from noesis_core.capture_event_fusion import canonical_json_sha256  # noqa: E402
from scripts.mapanything_fusion_quality_report import (  # noqa: E402
    evaluate_fusion_cohort,
)

CONTRACT = "noesis.mapanything.cadence_ab_analysis"
CONTRACT_VERSION = 1
ARM_RECEIPT_CONTRACT = "noesis.mapanything.cadence_arm_receipt"
ARM_RECEIPT_VERSION = 1
ARM_RECEIPT_FILENAME = "cadence-arm-receipt.json"
SOURCE_FRAME_CONTRACT = "noesis.mapanything.source_frame.v1"
EXPECTED_INTERVALS = (89, 59)
EXPECTED_SOURCE_FPS = 30.0
EXPECTED_BURST_SECONDS = 20.0
EXPECTED_MIN_OBSERVATIONS = 3
EXPECTED_AGREEMENT_M = 0.18
AGREEMENT_CANDIDATES_M = (0.12, 0.18)
REQUIRED_CONTROL_FINGERPRINTS = frozenset(
    {
        "engine",
        "mapanything_input_contract",
        "calibration_bundle",
        "dewarper_bundle",
        "source_corpus",
    }
)
ARM_RECEIPT_KEYS = frozenset(
    {
        "contract",
        "contract_version",
        "arm_id",
        "interval_frames",
        "source_fps",
        "burst_seconds",
        "min_observations",
        "depth_agreement_m",
        "infer_config",
        "live_gate",
        "snapshot_store",
        "controlled_input_sha256s",
        "receipt_sha256",
    }
)
INFER_CONFIG_KEYS = frozenset({"path", "sha256"})
LIVE_GATE_KEYS = frozenset(
    {
        "report_path",
        "report_sha256",
        "source_path",
        "source_sha256",
    }
)


@dataclass(frozen=True)
class SnapshotIdentity:
    path: Path
    camera_id: str
    timestamp_us: int
    sequence: int
    write_id: str
    manifest_sha256: str
    content_sha256: str
    component_sha256s: Mapping[str, str]
    snapshot_role: str
    fusion_level: str
    attrs: Mapping[str, Any]


@dataclass(frozen=True)
class LoadedSnapshot:
    identity: SnapshotIdentity
    depth: np.ndarray
    confidence: np.ndarray
    mask: np.ndarray


@dataclass
class ArmAnalysis:
    arm_id: str
    root: Path
    interval_frames: int
    session_id: str
    runtime_lane: str
    runtime_instance_id: str
    runtime_run_id: str
    camera_order: tuple[str, ...]
    controlled_input_sha256s: dict[str, str]
    config_semantics_without_interval: dict[str, dict[str, str]]
    cameras: dict[str, dict[str, Any]]
    cohorts: dict[str, tuple[LoadedSnapshot, ...]]


def _strict_json_load(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    decoded = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(decoded, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return decoded


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    text = str(value or "").strip()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _require_exact_keys(
    value: object,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{label} schema mismatch")
    return value


def _require_text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _require_exact_int(value: object, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:  # noqa: E721
        raise ValueError(f"{label} must equal {expected}")
    return value


def _require_exact_float(value: object, expected: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError(f"{label} must equal {expected}")
    return float(value)


def _resolve_relative(
    root: Path,
    value: object,
    label: str,
    *,
    require_file: bool | None = None,
) -> Path:
    raw = _require_text(value, label)
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a portable relative path")
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its arm root") from exc
    if require_file is True and not candidate.is_file():
        raise ValueError(f"{label} must name a regular file")
    if require_file is False and not candidate.is_dir():
        raise ValueError(f"{label} must name a directory")
    return candidate


def _load_nvinfer_semantics(
    path: Path,
    *,
    expected_interval: int,
) -> dict[str, dict[str, str]]:
    parser = configparser.ConfigParser(
        interpolation=None,
        strict=True,
        delimiters=("=",),
        comment_prefixes=("#", ";"),
    )
    parser.optionxform = str
    try:
        with path.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
    except (OSError, UnicodeError, configparser.Error) as exc:
        raise ValueError(f"invalid MapAnything nvinfer config: {path}") from exc
    if not parser.has_section("property"):
        raise ValueError("MapAnything nvinfer config lacks [property]")
    property_keys = {key.lower(): value.strip() for key, value in parser["property"].items()}
    try:
        interval = int(property_keys["interval"], 10)
    except (KeyError, ValueError) as exc:
        raise ValueError("MapAnything nvinfer interval is invalid") from exc
    if interval != expected_interval:
        raise ValueError(
            f"MapAnything nvinfer config interval {interval} != {expected_interval}"
        )
    semantics = {
        section.lower(): {
            key.lower(): value.strip()
            for key, value in parser[section].items()
        }
        for section in parser.sections()
    }
    semantics["property"].pop("interval", None)
    return semantics


def _json_attr(attrs: Mapping[str, Any], key: str) -> Any:
    value = attrs.get(key)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"snapshot attribute {key} is invalid JSON") from exc
    return value


def _load_snapshot(path: Path, *, expected_camera: str) -> LoadedSnapshot:
    resolved = path.resolve(strict=True)
    try:
        manifest = DepthStorageManager._read_and_validate_commit_manifest(resolved)
    except Exception as exc:
        raise ValueError(
            f"snapshot commit validation failed: {resolved}"
        ) from exc
    if int(manifest.get("version") or 0) != 2:
        raise ValueError(f"snapshot must use commit manifest v2: {resolved}")
    try:
        group = zarr.open_group(str(resolved), mode="r")
        attrs = dict(group.attrs.asdict())
        depth_key = "depth_z" if "depth_z" in group else "depth"
        if depth_key not in group or "conf" not in group or "mask" not in group:
            raise ValueError("snapshot lacks depth/conf/mask")
        depth = np.asarray(group[depth_key], dtype=np.float32)
        confidence = np.asarray(group["conf"], dtype=np.float32)
        mask = np.asarray(group["mask"], dtype=np.uint8)
        rgb = np.asarray(group["rgb"], dtype=np.uint8) if "rgb" in group else None
    except Exception as exc:
        raise ValueError(f"snapshot array load failed: {resolved}") from exc
    if depth.ndim != 2 or confidence.shape != depth.shape or mask.shape != depth.shape:
        raise ValueError(f"snapshot depth/conf/mask shape mismatch: {resolved}")

    camera_id = _require_text(attrs.get("camera_id"), "snapshot camera_id")
    write_id = _require_text(attrs.get("write_id"), "snapshot write_id")
    try:
        timestamp_us = int(attrs["timestamp_us"])
        sequence = int(attrs["sequence"])
    except Exception as exc:
        raise ValueError(f"snapshot identity attributes missing: {resolved}") from exc
    if (
        camera_id != expected_camera
        or timestamp_us <= 0
        or sequence <= 0
        or manifest.get("state") != "committed"
        or manifest.get("camera_id") != camera_id
        or int(manifest.get("timestamp_us") or 0) != timestamp_us
        or int(manifest.get("sequence") or 0) != sequence
        or manifest.get("write_id") != write_id
    ):
        raise ValueError(f"snapshot identity mismatch: {resolved}")

    actual_components = DepthStorageManager._raw_component_records(
        depth=depth,
        conf=confidence,
        mask=mask,
        rgb=rgb,
    )
    committed_components = manifest.get("components")
    if (
        not isinstance(committed_components, Mapping)
        or set(committed_components) != set(actual_components)
        or any(
            not isinstance(committed_components[name], Mapping)
            or dict(committed_components[name]) != actual_components[name]
            for name in actual_components
        )
    ):
        raise ValueError(f"snapshot component content mismatch: {resolved}")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError(f"snapshot file identity missing: {resolved}")
    content_sha256 = hashlib.sha256(
        DepthStorageManager._canonical_json_bytes(
            {
                "files": files,
                "components": dict(committed_components),
            }
        )
    ).hexdigest()
    identity = SnapshotIdentity(
        path=resolved,
        camera_id=camera_id,
        timestamp_us=timestamp_us,
        sequence=sequence,
        write_id=write_id,
        manifest_sha256=_require_sha256(
            manifest.get("manifest_sha256"),
            "snapshot manifest_sha256",
        ),
        content_sha256=content_sha256,
        component_sha256s={
            name: _require_sha256(row.get("sha256"), f"{name} sha256")
            for name, row in committed_components.items()
        },
        snapshot_role=str(attrs.get("snapshot_role") or "").strip(),
        fusion_level=str(attrs.get("fusion_level") or "").strip(),
        attrs=attrs,
    )
    return LoadedSnapshot(
        identity=identity,
        depth=depth,
        confidence=confidence,
        mask=mask,
    )


def _resolve_storage_ref(store_root: Path, value: object, label: str) -> Path:
    reference = Path(_require_text(value, label))
    if reference.is_absolute() or ".." in reference.parts:
        raise ValueError(f"{label} must be a portable storage reference")
    resolved = (store_root / reference).resolve(strict=True)
    try:
        resolved.relative_to(store_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the sealed snapshot store") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} does not name a snapshot directory")
    return resolved


def _source_timing(
    snapshots: Sequence[LoadedSnapshot],
    *,
    interval_frames: int,
    source_fps: float,
    expected_source_id: int | None = None,
) -> dict[str, Any]:
    if len(snapshots) < 2:
        raise ValueError("cadence analysis requires at least two raw snapshots")
    timestamps = [row.identity.timestamp_us for row in snapshots]
    frame_numbers: list[int] = []
    media_pts_ns: list[int] = []
    timestamp_bases: list[str] = []
    source_ids: list[int | None] = []
    for row in snapshots:
        attrs = row.identity.attrs
        if attrs.get("source_frame_contract") != SOURCE_FRAME_CONTRACT:
            raise ValueError(
                f"raw snapshot lacks exact source-frame contract: {row.identity.path}"
            )
        if (
            type(attrs.get("source_frame_number")) is not int  # noqa: E721
            or type(attrs.get("source_media_pts_ns")) is not int  # noqa: E721
            or int(attrs["source_frame_number"]) < 0
            or int(attrs["source_media_pts_ns"]) < 0
        ):
            raise ValueError(
                f"raw snapshot source frame/PTS evidence is invalid: {row.identity.path}"
            )
        raw_source_id = attrs.get("source_id")
        if raw_source_id is not None and (
            type(raw_source_id) is not int or raw_source_id < 0  # noqa: E721
        ):
            raise ValueError(
                f"raw snapshot source_id is invalid: {row.identity.path}"
            )
        source_ids.append(raw_source_id)
        basis = str(attrs.get("storage_timestamp_basis") or "")
        if basis not in {"capture_wall_clock", "source_epoch_pts"}:
            raise ValueError(
                "raw snapshot timestamp basis is not hardened capture-time "
                f"evidence: {row.identity.path}"
            )
        frame_numbers.append(int(attrs["source_frame_number"]))
        media_pts_ns.append(int(attrs["source_media_pts_ns"]))
        timestamp_bases.append(basis)
    if len(set(timestamp_bases)) != 1:
        raise ValueError(
            "raw snapshot cohort mixes storage timestamp bases"
        )
    present_source_ids = [value for value in source_ids if value is not None]
    if present_source_ids and len(present_source_ids) != len(source_ids):
        raise ValueError("raw snapshot cohort mixes present and missing source_id")
    if len(set(present_source_ids)) > 1:
        raise ValueError("raw snapshot cohort mixes source identities")
    source_id = present_source_ids[0] if present_source_ids else None
    if (
        expected_source_id is not None
        and source_id is not None
        and source_id != expected_source_id
    ):
        raise ValueError(
            "raw snapshot source_id does not match exact RGB camera binding"
        )

    def gaps(values: Sequence[int]) -> list[int]:
        output = [
            int(right) - int(left)
            for left, right in zip(values[:-1], values[1:], strict=True)
        ]
        if any(value <= 0 for value in output):
            raise ValueError("raw snapshot timing identities must strictly advance")
        return output

    frame_gaps = gaps(frame_numbers)
    pts_gaps_ns = gaps(media_pts_ns)
    storage_gaps_us = gaps(timestamps)
    period_frames = interval_frames + 1
    pts_expected_from_frames_s = [value / source_fps for value in frame_gaps]
    pts_gaps_s = [value / 1_000_000_000.0 for value in pts_gaps_ns]
    pts_errors_s = [
        abs(observed - expected)
        for observed, expected in zip(
            pts_gaps_s,
            pts_expected_from_frames_s,
            strict=True,
        )
    ]
    period_on_lattice = all(value % period_frames == 0 for value in frame_gaps)
    every_opportunity = all(value == period_frames for value in frame_gaps)
    return {
        "snapshot_count": len(snapshots),
        "source_id": source_id,
        "source_frame_numbers": frame_numbers,
        "source_frame_gaps": frame_gaps,
        "expected_source_frame_period": period_frames,
        "on_declared_interval_lattice": period_on_lattice,
        "every_inference_opportunity_persisted": every_opportunity,
        "missing_opportunity_count": (
            sum(max(0, value // period_frames - 1) for value in frame_gaps)
            if period_on_lattice
            else None
        ),
        "source_media_pts_ns": media_pts_ns,
        "media_pts_gaps_s": pts_gaps_s,
        "media_pts_gap_median_s": statistics.median(pts_gaps_s),
        "media_pts_matches_frame_numbers": all(
            error <= max(0.05, 2.0 / source_fps) for error in pts_errors_s
        ),
        "storage_timestamps_us": timestamps,
        "storage_timestamp_gaps_s": [
            value / 1_000_000.0 for value in storage_gaps_us
        ],
        "storage_timestamp_basis": timestamp_bases,
        "cohort_span_s": (timestamps[-1] - timestamps[0]) / 1_000_000.0,
    }


def _compact_evaluation(report: Mapping[str, Any]) -> dict[str, Any]:
    candidate = report["candidates"]["confidence_weighted_runtime_bounded"]
    support = report["support"]
    temporal_reference = report["temporal_consensus_reference"]
    return {
        "cohort": report["cohort"],
        "support": support,
        "scale_normalization": report["scale_normalization"],
        "confidence": report["confidence"],
        "temporal_residual_definition": temporal_reference["definition"],
        "temporal_residual_p50_m": temporal_reference["residual_p50_m"],
        "temporal_residual_p95_m": temporal_reference["residual_p95_m"],
        "fused_candidate_residual_p50_m": candidate[
            "temporal_residual_p50_m"
        ],
        "fused_candidate_residual_p95_m": candidate[
            "temporal_residual_p95_m"
        ],
        "edge": candidate["edge"],
    }


def _close_float(left: object, right: object, *, tolerance: float = 1e-6) -> bool:
    return bool(
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
        and math.isfinite(float(left))
        and math.isfinite(float(right))
        and math.isclose(
            float(left),
            float(right),
            rel_tol=1e-6,
            abs_tol=tolerance,
        )
    )


def _validate_stored_fusion_evidence(
    fused: LoadedSnapshot,
    evaluation: Mapping[str, Any],
    *,
    source_count: int,
) -> None:
    fusion_meta = _json_attr(fused.identity.attrs, "fusion_meta")
    if not isinstance(fusion_meta, Mapping):
        raise ValueError("fused snapshot lacks fusion_meta")
    stored_support = fusion_meta.get("support_evidence")
    if not isinstance(stored_support, Mapping):
        raise ValueError("fused snapshot lacks support evidence")
    cohort = evaluation["cohort"]
    support = evaluation["support"]
    checks = (
        (
            fusion_meta.get("source_snapshot_count"),
            source_count,
            "source snapshot count",
        ),
        (
            stored_support.get("required_observations"),
            cohort.get("required_support"),
            "required support",
        ),
        (
            stored_support.get("strict_majority_observations"),
            cohort.get("strict_majority_support"),
            "strict-majority support",
        ),
    )
    if any(left != right for left, right, _label in checks):
        labels = ", ".join(
            label for left, right, label in checks if left != right
        )
        raise ValueError(f"stored fusion evidence differs from replay: {labels}")
    float_checks = (
        (
            stored_support.get("eligible_full_frame_fraction"),
            support.get("eligible_fraction"),
            "eligible coverage",
        ),
        (
            stored_support.get("consensus_full_frame_fraction"),
            support.get("retained_full_frame_fraction"),
            "retained coverage",
        ),
        (
            stored_support.get("consensus_retained_eligible_fraction"),
            support.get("retained_over_eligible_fraction"),
            "retained/eligible coverage",
        ),
        (
            stored_support.get("temporal_absolute_residual_median_m"),
            evaluation.get("temporal_residual_p50_m"),
            "temporal residual p50",
        ),
        (
            stored_support.get("temporal_absolute_residual_p95_m"),
            evaluation.get("temporal_residual_p95_m"),
            "temporal residual p95",
        ),
    )
    if any(not _close_float(left, right) for left, right, _label in float_checks):
        labels = ", ".join(
            label
            for left, right, label in float_checks
            if not _close_float(left, right)
        )
        raise ValueError(f"stored fusion evidence differs from replay: {labels}")
    stored_components = stored_support.get("component_evidence")
    replay_components = support.get("component_evidence")
    if (
        not isinstance(stored_components, Mapping)
        or not isinstance(replay_components, Mapping)
        or set(stored_components) != set(replay_components)
        or any(
            (
                not _close_float(stored_components[key], replay_components[key])
                if isinstance(replay_components[key], float)
                else stored_components[key] != replay_components[key]
            )
            for key in replay_components
        )
    ):
        raise ValueError("stored component/hole/fragment evidence differs from replay")


def _analyze_capture(
    *,
    event: Mapping[str, Any],
    result: Mapping[str, Any],
    store_root: Path,
    interval_frames: int,
    source_fps: float,
) -> tuple[dict[str, Any], tuple[LoadedSnapshot, ...]]:
    camera_id = _require_text(event.get("camera_id"), "capture camera_id")
    if result.get("camera_id") != camera_id:
        raise ValueError("live-gate capture camera identity mismatch")
    parameters = event.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("capture event parameters are missing")
    _require_exact_float(
        parameters.get("burst_seconds"),
        EXPECTED_BURST_SECONDS,
        "capture burst_seconds",
    )
    _require_exact_int(
        parameters.get("min_observations"),
        EXPECTED_MIN_OBSERVATIONS,
        "capture min_observations",
    )
    _require_exact_float(
        parameters.get("depth_agreement_m"),
        EXPECTED_AGREEMENT_M,
        "capture depth_agreement_m",
    )
    fused_evidence = event.get("fused_snapshot")
    if not isinstance(fused_evidence, Mapping):
        raise ValueError("capture event fused snapshot is missing")
    artifact_ref = _require_text(
        fused_evidence.get("artifact_ref"),
        "fused artifact_ref",
    )
    if not artifact_ref.startswith("depth-zarr:"):
        raise ValueError("fused artifact_ref must use depth-zarr")
    fused = _load_snapshot(
        _resolve_storage_ref(
            store_root,
            artifact_ref.removeprefix("depth-zarr:"),
            "fused artifact_ref",
        ),
        expected_camera=camera_id,
    )
    fused_identity = fused.identity
    if (
        fused_identity.snapshot_role != "capture_event_fused"
        or fused_identity.fusion_level != "intra_capture"
        or fused_identity.timestamp_us != fused_evidence.get("timestamp_us")
        or fused_identity.sequence != fused_evidence.get("sequence")
        or fused_identity.write_id != fused_evidence.get("snapshot_id")
        or fused_identity.manifest_sha256
        != fused_evidence.get("manifest_sha256")
        or fused_identity.content_sha256 != fused_evidence.get("content_sha256")
        or fused_identity.attrs.get("event_id") != fused_evidence.get("event_id")
    ):
        raise ValueError("fused snapshot content identity differs from capture receipt")

    source_rows = _json_attr(fused_identity.attrs, "source_snapshots")
    source_ids = fused_evidence.get("source_snapshot_ids")
    if (
        not isinstance(source_rows, list)
        or not source_rows
        or not isinstance(source_ids, list)
        or len(source_rows) != len(source_ids)
        or int(event.get("raw_snapshot_count") or 0) != len(source_rows)
    ):
        raise ValueError("capture raw source count does not reconcile")
    raw_snapshots: list[LoadedSnapshot] = []
    for index, row in enumerate(source_rows):
        if not isinstance(row, Mapping):
            raise ValueError("fused source row is invalid")
        raw = _load_snapshot(
            _resolve_storage_ref(
                store_root,
                row.get("storage_ref"),
                f"raw source {index} storage_ref",
            ),
            expected_camera=camera_id,
        )
        identity = raw.identity
        if (
            identity.snapshot_role
            or identity.fusion_level
            or identity.write_id != source_ids[index]
            or identity.write_id != row.get("write_id")
            or identity.timestamp_us != row.get("timestamp_us")
            or identity.sequence != row.get("sequence")
            or identity.manifest_sha256 != row.get("manifest_sha256")
            or identity.content_sha256 != row.get("content_sha256")
            or row.get("camera_id") != camera_id
            or str(row.get("snapshot_role") or "")
            or str(row.get("fusion_level") or "")
        ):
            raise ValueError("raw snapshot identity differs from fused cohort receipt")
        raw_snapshots.append(raw)
    raw_snapshots.sort(key=lambda row: row.identity.timestamp_us)
    if [row.identity.write_id for row in raw_snapshots] != list(source_ids):
        raise ValueError("capture receipt source order differs from committed timestamps")
    baseline = event.get("baseline_raw_timestamp_us")
    max_span_us = parameters.get("max_cohort_span_us")
    if (
        type(baseline) is not int  # noqa: E721
        or type(max_span_us) is not int  # noqa: E721
        or any(row.identity.timestamp_us <= baseline for row in raw_snapshots)
        or (
            raw_snapshots[-1].identity.timestamp_us
            - raw_snapshots[0].identity.timestamp_us
            > max_span_us
        )
    ):
        raise ValueError("raw snapshot cohort violates its capture fence")
    shapes = {tuple(row.depth.shape) for row in raw_snapshots}
    if len(shapes) != 1:
        raise ValueError("raw snapshot cohort shapes differ")

    expected_source_id: int | None = None
    exact_rgb_identity: tuple[int, int, int] | None = None
    rgb = event.get("rgb")
    if isinstance(rgb, Mapping) and rgb.get("status") == "available":
        rgb_identity_values = (
            rgb.get("source_id"),
            rgb.get("frame_id"),
            rgb.get("source_media_pts_ns"),
        )
        if any(
            type(value) is not int or value < 0  # noqa: E721
            for value in rgb_identity_values
        ):
            raise ValueError("exact RGB source/frame/PTS identity is invalid")
        exact_rgb_identity = tuple(rgb_identity_values)
        expected_source_id = exact_rgb_identity[0]

    timing = _source_timing(
        raw_snapshots,
        interval_frames=interval_frames,
        source_fps=source_fps,
        expected_source_id=expected_source_id,
    )
    if (
        exact_rgb_identity is not None
        and timing["source_id"] is not None
        and exact_rgb_identity
        not in {
            (
                int(row.identity.attrs["source_id"]),
                int(row.identity.attrs["source_frame_number"]),
                int(row.identity.attrs["source_media_pts_ns"]),
            )
            for row in raw_snapshots
        }
    ):
        raise ValueError("exact RGB identity is absent from the raw depth cohort")
    report = evaluate_fusion_cohort(
        np.stack([row.depth for row in raw_snapshots]),
        np.stack([row.confidence for row in raw_snapshots]),
        np.stack([row.mask for row in raw_snapshots]),
        min_observations=EXPECTED_MIN_OBSERVATIONS,
        depth_agreement_m=EXPECTED_AGREEMENT_M,
        normalize_frame_scale=True,
    )
    evaluation = _compact_evaluation(report)
    _validate_stored_fusion_evidence(
        fused,
        evaluation,
        source_count=len(raw_snapshots),
    )
    return (
        {
            "camera_id": camera_id,
            "calibration_fingerprint": _require_sha256(
                result.get("calibration_fingerprint"),
                "live-gate calibration_fingerprint",
            ),
            "fused_snapshot": {
                "storage_ref": artifact_ref.removeprefix("depth-zarr:"),
                "snapshot_id": fused_identity.write_id,
                "timestamp_us": fused_identity.timestamp_us,
                "manifest_sha256": fused_identity.manifest_sha256,
                "content_sha256": fused_identity.content_sha256,
            },
            "raw_snapshot_identities": [
                {
                    "storage_ref": str(
                        row.identity.path.relative_to(store_root)
                    ),
                    "timestamp_us": row.identity.timestamp_us,
                    "sequence": row.identity.sequence,
                    "write_id": row.identity.write_id,
                    "manifest_sha256": row.identity.manifest_sha256,
                    "content_sha256": row.identity.content_sha256,
                    "component_sha256s": dict(
                        row.identity.component_sha256s
                    ),
                }
                for row in raw_snapshots
            ],
            "timing": timing,
            "fusion_at_0.18_m": evaluation,
        },
        tuple(raw_snapshots),
    )


def _load_live_gate_session(
    *,
    root: Path,
    row: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[tuple[Mapping[str, Any], Mapping[str, Any]]],
]:
    _require_exact_keys(row, LIVE_GATE_KEYS, "live_gate")
    report_path = _resolve_relative(
        root,
        row.get("report_path"),
        "live_gate.report_path",
        require_file=True,
    )
    source_path = _resolve_relative(
        root,
        row.get("source_path"),
        "live_gate.source_path",
        require_file=True,
    )
    if report_path.name != live_gate.CANONICAL_REPORT_FILENAME:
        raise ValueError("live-gate report filename is not canonical")
    if source_path.name != live_gate.CANONICAL_SOURCE_TRANSCRIPT_FILENAME:
        raise ValueError("live-gate source filename is not canonical")
    if _sha256_file(report_path) != _require_sha256(
        row.get("report_sha256"),
        "live_gate.report_sha256",
    ):
        raise ValueError("live-gate report digest mismatch")
    if _sha256_file(source_path) != _require_sha256(
        row.get("source_sha256"),
        "live_gate.source_sha256",
    ):
        raise ValueError("live-gate source digest mismatch")
    source = _strict_json_load(source_path)
    report = live_gate.load_and_validate_sealed_authority(
        report_path,
        source_path,
        session_id=_require_text(source.get("session_id"), "session_id"),
        runtime_lane=_require_text(source.get("runtime_lane"), "runtime_lane"),
        runtime_instance_id=_require_text(
            source.get("runtime_instance_id"),
            "runtime_instance_id",
        ),
        runtime_run_id=_require_text(
            source.get("runtime_run_id"),
            "runtime_run_id",
        ),
    )
    messages = source.get("messages")
    if not isinstance(messages, list):
        raise ValueError("live-gate source messages are missing")
    captures: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for message in messages:
        if (
            isinstance(message, Mapping)
            and message.get("type") == "validated_exact_floorplan_capture"
        ):
            event = message.get("capture_event")
            result = message.get("result")
            if not isinstance(event, Mapping) or not isinstance(result, Mapping):
                raise ValueError("live-gate exact capture evidence is malformed")
            captures.append((event, result))
    camera_ids = source.get("camera_ids")
    if (
        not isinstance(camera_ids, list)
        or len(captures) != len(camera_ids)
        or [event.get("camera_id") for event, _result in captures] != camera_ids
    ):
        raise ValueError("live-gate capture inventory does not cover camera order")
    return source, report, captures


def load_arm(root: Path, *, expected_interval: int) -> ArmAnalysis:
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError("cadence arm root must be a directory")
    receipt_path = resolved_root / ARM_RECEIPT_FILENAME
    receipt = _strict_json_load(receipt_path.resolve(strict=True))
    _require_exact_keys(receipt, ARM_RECEIPT_KEYS, "cadence arm receipt")
    if (
        receipt.get("contract") != ARM_RECEIPT_CONTRACT
        or receipt.get("contract_version") != ARM_RECEIPT_VERSION
    ):
        raise ValueError("cadence arm receipt contract mismatch")
    expected_receipt_sha = _require_sha256(
        receipt.get("receipt_sha256"),
        "receipt_sha256",
    )
    hashable_receipt = dict(receipt)
    hashable_receipt.pop("receipt_sha256")
    if canonical_json_sha256(hashable_receipt) != expected_receipt_sha:
        raise ValueError("cadence arm receipt self-digest mismatch")

    _require_exact_int(
        receipt.get("interval_frames"),
        expected_interval,
        "interval_frames",
    )
    expected_arm_id = f"interval-{expected_interval}"
    if receipt.get("arm_id") != expected_arm_id:
        raise ValueError(f"arm_id must equal {expected_arm_id}")
    source_fps = _require_exact_float(
        receipt.get("source_fps"),
        EXPECTED_SOURCE_FPS,
        "source_fps",
    )
    _require_exact_float(
        receipt.get("burst_seconds"),
        EXPECTED_BURST_SECONDS,
        "burst_seconds",
    )
    _require_exact_int(
        receipt.get("min_observations"),
        EXPECTED_MIN_OBSERVATIONS,
        "min_observations",
    )
    _require_exact_float(
        receipt.get("depth_agreement_m"),
        EXPECTED_AGREEMENT_M,
        "depth_agreement_m",
    )

    infer = _require_exact_keys(
        receipt.get("infer_config"),
        INFER_CONFIG_KEYS,
        "infer_config",
    )
    infer_path = _resolve_relative(
        resolved_root,
        infer.get("path"),
        "infer_config.path",
        require_file=True,
    )
    if _sha256_file(infer_path) != _require_sha256(
        infer.get("sha256"),
        "infer_config.sha256",
    ):
        raise ValueError("MapAnything nvinfer config digest mismatch")
    config_semantics = _load_nvinfer_semantics(
        infer_path,
        expected_interval=expected_interval,
    )

    controls = receipt.get("controlled_input_sha256s")
    if (
        not isinstance(controls, Mapping)
        or not REQUIRED_CONTROL_FINGERPRINTS.issubset(controls)
        or any(not isinstance(key, str) or not key for key in controls)
    ):
        raise ValueError(
            "controlled_input_sha256s lacks required comparison controls"
        )
    controlled = {
        key: _require_sha256(value, f"controlled input {key}")
        for key, value in controls.items()
    }
    live_source, _live_report, captures = _load_live_gate_session(
        root=resolved_root,
        row=_require_exact_keys(
            receipt.get("live_gate"),
            LIVE_GATE_KEYS,
            "live_gate",
        ),
    )
    store_root = _resolve_relative(
        resolved_root,
        receipt.get("snapshot_store"),
        "snapshot_store",
        require_file=False,
    )
    cameras: dict[str, dict[str, Any]] = {}
    cohorts: dict[str, tuple[LoadedSnapshot, ...]] = {}
    for event, result in captures:
        camera_id = _require_text(event.get("camera_id"), "capture camera")
        if camera_id in cameras:
            raise ValueError(f"duplicate live-gate capture camera: {camera_id}")
        camera, cohort = _analyze_capture(
            event=event,
            result=result,
            store_root=store_root,
            interval_frames=expected_interval,
            source_fps=source_fps,
        )
        cameras[camera_id] = camera
        cohorts[camera_id] = cohort
    camera_order = tuple(
        _require_text(value, "live-gate camera_id")
        for value in live_source.get("camera_ids", [])
    )
    if tuple(cameras) != camera_order:
        raise ValueError("analyzed camera order differs from live-gate order")
    arm_timestamp_bases = {
        camera["timing"]["storage_timestamp_basis"][0]
        for camera in cameras.values()
    }
    if len(arm_timestamp_bases) != 1:
        raise ValueError("cadence arm mixes storage timestamp bases across cameras")
    return ArmAnalysis(
        arm_id=expected_arm_id,
        root=resolved_root,
        interval_frames=expected_interval,
        session_id=_require_text(live_source.get("session_id"), "session_id"),
        runtime_lane=_require_text(
            live_source.get("runtime_lane"),
            "runtime_lane",
        ),
        runtime_instance_id=_require_text(
            live_source.get("runtime_instance_id"),
            "runtime_instance_id",
        ),
        runtime_run_id=_require_text(
            live_source.get("runtime_run_id"),
            "runtime_run_id",
        ),
        camera_order=camera_order,
        controlled_input_sha256s=controlled,
        config_semantics_without_interval=config_semantics,
        cameras=cameras,
        cohorts=cohorts,
    )


METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "raw_snapshot_count": ("timing", "snapshot_count"),
    "eligible_full_frame_fraction": (
        "fusion_at_0.18_m",
        "support",
        "eligible_fraction",
    ),
    "retained_full_frame_fraction": (
        "fusion_at_0.18_m",
        "support",
        "retained_full_frame_fraction",
    ),
    "retained_over_eligible_fraction": (
        "fusion_at_0.18_m",
        "support",
        "retained_over_eligible_fraction",
    ),
    "retained_support_p50": (
        "fusion_at_0.18_m",
        "support",
        "retained_p50",
    ),
    "temporal_residual_p50_m": (
        "fusion_at_0.18_m",
        "temporal_residual_p50_m",
    ),
    "temporal_residual_p95_m": (
        "fusion_at_0.18_m",
        "temporal_residual_p95_m",
    ),
    "component_count": (
        "fusion_at_0.18_m",
        "support",
        "component_evidence",
        "component_count",
    ),
    "hole_count": (
        "fusion_at_0.18_m",
        "support",
        "component_evidence",
        "hole_count",
    ),
    "hole_pixels": (
        "fusion_at_0.18_m",
        "support",
        "component_evidence",
        "hole_pixels",
    ),
    "fragment_fraction": (
        "fusion_at_0.18_m",
        "support",
        "component_evidence",
        "fragment_fraction",
    ),
}


def _path_number(value: Mapping[str, Any], path: Sequence[str]) -> float | None:
    current: object = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    if (
        isinstance(current, (int, float))
        and not isinstance(current, bool)
        and math.isfinite(float(current))
    ):
        return float(current)
    return None


def _camera_balanced_metrics(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    *,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    if tuple(left) != tuple(right):
        raise ValueError("camera-balanced comparison requires identical camera order")
    output: dict[str, Any] = {}
    for metric, path in METRIC_PATHS.items():
        per_camera: dict[str, Any] = {}
        left_values: list[float] = []
        right_values: list[float] = []
        for camera in left:
            left_value = _path_number(left[camera], path)
            right_value = _path_number(right[camera], path)
            if left_value is None or right_value is None:
                per_camera[camera] = {
                    left_label: left_value,
                    right_label: right_value,
                    "delta_right_minus_left": None,
                }
                continue
            left_values.append(left_value)
            right_values.append(right_value)
            per_camera[camera] = {
                left_label: left_value,
                right_label: right_value,
                "delta_right_minus_left": right_value - left_value,
            }
        output[metric] = {
            "per_camera": per_camera,
            "camera_balanced_mean": {
                left_label: (
                    statistics.fmean(left_values) if left_values else None
                ),
                right_label: (
                    statistics.fmean(right_values) if right_values else None
                ),
                "delta_right_minus_left": (
                    statistics.fmean(right_values)
                    - statistics.fmean(left_values)
                    if left_values
                    and len(left_values) == len(right_values)
                    else None
                ),
                "camera_count": len(left_values),
            },
        }
    return output


def _public_arm(arm: ArmAnalysis) -> dict[str, Any]:
    return {
        "arm_id": arm.arm_id,
        "arm_root": str(arm.root),
        "interval_frames": arm.interval_frames,
        "source_fps": EXPECTED_SOURCE_FPS,
        "burst_seconds": EXPECTED_BURST_SECONDS,
        "min_observations": EXPECTED_MIN_OBSERVATIONS,
        "depth_agreement_m": EXPECTED_AGREEMENT_M,
        "session_id": arm.session_id,
        "runtime_lane": arm.runtime_lane,
        "runtime_instance_id": arm.runtime_instance_id,
        "runtime_run_id": arm.runtime_run_id,
        "camera_order": list(arm.camera_order),
        "controlled_input_sha256s": arm.controlled_input_sha256s,
        "cameras": arm.cameras,
    }


def _agreement_camera_payload(
    arm: ArmAnalysis,
    camera_id: str,
) -> dict[str, Any]:
    cohort = arm.cohorts[camera_id]
    identities = arm.cameras[camera_id]["raw_snapshot_identities"]
    evaluations: dict[str, Any] = {}
    required_support: set[int] = set()
    for agreement in AGREEMENT_CANDIDATES_M:
        report = evaluate_fusion_cohort(
            np.stack([row.depth for row in cohort]),
            np.stack([row.confidence for row in cohort]),
            np.stack([row.mask for row in cohort]),
            min_observations=EXPECTED_MIN_OBSERVATIONS,
            depth_agreement_m=agreement,
            normalize_frame_scale=True,
        )
        compact = _compact_evaluation(report)
        required_support.add(int(compact["cohort"]["required_support"]))
        evaluations[f"{agreement:.2f}"] = compact
    if len(required_support) != 1:
        raise ValueError("agreement replay changed strict-majority support")
    return {
        "camera_id": camera_id,
        "exact_raw_snapshot_identities": identities,
        "same_cohort_for_all_agreements": True,
        "required_support": required_support.pop(),
        "evaluations": evaluations,
    }


def build_report(
    interval89: ArmAnalysis,
    interval59: ArmAnalysis,
    *,
    agreement_arm: str | None = None,
) -> dict[str, Any]:
    if (
        interval89.interval_frames != 89
        or interval59.interval_frames != 59
        or interval89.camera_order != interval59.camera_order
    ):
        raise ValueError("cadence arms do not match the required camera-balanced design")
    if interval89.runtime_lane != interval59.runtime_lane:
        raise ValueError("cadence arms changed runtime lane")
    if interval89.runtime_instance_id != interval59.runtime_instance_id:
        raise ValueError("cadence arms changed runtime instance")
    if interval89.controlled_input_sha256s != interval59.controlled_input_sha256s:
        raise ValueError("cadence arms changed controlled input fingerprints")
    if (
        interval89.config_semantics_without_interval
        != interval59.config_semantics_without_interval
    ):
        raise ValueError("MapAnything nvinfer configs differ beyond interval")
    if (
        interval89.session_id == interval59.session_id
        or interval89.runtime_run_id == interval59.runtime_run_id
    ):
        raise ValueError("cadence arms must be distinct fresh runtime sessions")
    for camera in interval89.camera_order:
        left = interval89.cameras[camera]
        right = interval59.cameras[camera]
        if left["calibration_fingerprint"] != right["calibration_fingerprint"]:
            raise ValueError(f"{camera} calibration changed between cadence arms")
        left_shape = left["fusion_at_0.18_m"]["cohort"]["shape"]
        right_shape = right["fusion_at_0.18_m"]["cohort"]["shape"]
        if left_shape != right_shape:
            raise ValueError(f"{camera} depth shape changed between cadence arms")
        left_ids = {
            row["write_id"] for row in left["raw_snapshot_identities"]
        }
        right_ids = {
            row["write_id"] for row in right["raw_snapshot_identities"]
        }
        if left_ids & right_ids:
            raise ValueError(f"{camera} raw snapshots were reused across cadence arms")
        if left["timing"]["source_id"] != right["timing"]["source_id"]:
            raise ValueError(f"{camera} source identity changed between cadence arms")

    for arm in (interval89, interval59):
        source_bindings = [
            arm.cameras[camera]["timing"]["source_id"]
            for camera in arm.camera_order
            if arm.cameras[camera]["timing"]["source_id"] is not None
        ]
        if len(source_bindings) != len(set(source_bindings)):
            raise ValueError(f"{arm.arm_id} reuses one source identity across cameras")

    timing_blockers: list[str] = []
    for arm in (interval89, interval59):
        for camera in arm.camera_order:
            timing = arm.cameras[camera]["timing"]
            if not timing["on_declared_interval_lattice"]:
                timing_blockers.append(
                    f"{arm.arm_id}/{camera}: source frames are off the declared interval lattice"
                )
            if not timing["media_pts_matches_frame_numbers"]:
                timing_blockers.append(
                    f"{arm.arm_id}/{camera}: media PTS gaps disagree with source frame gaps"
                )
    comparison = _camera_balanced_metrics(
        interval89.cameras,
        interval59.cameras,
        left_label="interval_89",
        right_label="interval_59",
    )
    report: dict[str, Any] = {
        "contract": CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "design": {
            "comparison_variable": "nvinfer_property_interval_only",
            "source_fps": EXPECTED_SOURCE_FPS,
            "burst_seconds": EXPECTED_BURST_SECONDS,
            "min_observations": EXPECTED_MIN_OBSERVATIONS,
            "depth_agreement_m": EXPECTED_AGREEMENT_M,
            "strict_majority_policy": (
                "max(configured_minimum, cohort_floor_3, "
                "ceil(0.5*cohort), floor(cohort/2)+1)"
            ),
            "camera_weighting": "one_equal_weight_per_camera",
        },
        "comparison_ready": not timing_blockers,
        "comparison_blockers": timing_blockers,
        "arms": {
            "interval_89": _public_arm(interval89),
            "interval_59": _public_arm(interval59),
        },
        "camera_balanced_cadence_comparison": comparison,
        "cadence_selection": {
            "winner": None,
            "status": (
                "measured_comparison_ready_for_labeled_review"
                if not timing_blockers
                else "blocked_by_timing_evidence"
            ),
            "reason": (
                "coverage, temporal residual, components, holes, and fragments "
                "are reported without inventing scene ground truth or an "
                "unreviewed scalar quality score"
            ),
        },
        "agreement_ab": {
            "status": "not_requested",
            "selected_cadence_arm": None,
            "winner": None,
        },
    }
    if agreement_arm is not None:
        selected = {
            "interval-89": interval89,
            "interval-59": interval59,
        }.get(agreement_arm)
        if selected is None:
            raise ValueError("agreement_arm must be interval-89 or interval-59")
        agreement_cameras = {
            camera: _agreement_camera_payload(selected, camera)
            for camera in selected.camera_order
        }
        left = {
            camera: {
                "fusion_at_0.18_m": payload["evaluations"]["0.18"],
                "timing": selected.cameras[camera]["timing"],
            }
            for camera, payload in agreement_cameras.items()
        }
        right = {
            camera: {
                "fusion_at_0.18_m": payload["evaluations"]["0.12"],
                "timing": selected.cameras[camera]["timing"],
            }
            for camera, payload in agreement_cameras.items()
        }
        report["agreement_ab"] = {
            "status": (
                "exact_explicit_cadence_cohort_replay_with_ready_"
                "cadence_comparison"
                if report["comparison_ready"]
                else (
                    "exact_explicit_cadence_cohort_replay_with_blocked_"
                    "cadence_comparison"
                )
            ),
            "selected_cadence_arm": agreement_arm,
            "selection_source": (
                "explicit_cli_with_ready_cadence_comparison"
                if report["comparison_ready"]
                else "explicit_cli_with_blocked_cadence_comparison"
            ),
            "agreements_m": [0.18, 0.12],
            "same_raw_cohort_for_both_arms": True,
            "strict_majority_unchanged": True,
            "cameras": agreement_cameras,
            "camera_balanced_comparison": _camera_balanced_metrics(
                left,
                right,
                left_label="agreement_0.18_m",
                right_label="agreement_0.12_m",
            ),
            "winner": None,
            "reason": (
                "threshold effects are isolated on identical arrays; final "
                "selection still requires labeled boundary/ghost review"
            ),
        }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and compare exact interval=89 and interval=59 fresh "
            "MapAnything cadence sessions without touching runtime or GPU."
        )
    )
    parser.add_argument("--interval89-root", required=True, type=Path)
    parser.add_argument("--interval59-root", required=True, type=Path)
    parser.add_argument(
        "--agreement-arm",
        choices=("interval-89", "interval-59"),
        help=(
            "Replay this explicitly selected arm at 0.18 m and 0.12 m on "
            "the exact same raw cohorts."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    interval89 = load_arm(args.interval89_root, expected_interval=89)
    interval59 = load_arm(args.interval59_root, expected_interval=59)
    report = build_report(
        interval89,
        interval59,
        agreement_arm=args.agreement_arm,
    )
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise SystemExit(f"refusing to replace existing report: {output}") from exc
    print(
        f"[OK] wrote {output} "
        f"sha256={hashlib.sha256(encoded).hexdigest()} "
        f"comparison_ready={report['comparison_ready']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
