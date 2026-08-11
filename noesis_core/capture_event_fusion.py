"""Runtime-neutral capture-event fusion coordination.

The coordinator owns cohort selection and evidence. Runtime adapters own the
depth store and may optionally expose a frame already captured by the active
pipeline. This module deliberately has no camera-opening or transport logic.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RAW_ROLE = "raw"
_FUSED_ROLE = "capture_event_fused"
_INTRA_CAPTURE = "intra_capture"


def freeze_json_evidence(value: Any, *, _path: str = "$") -> Any:
    """Return a deeply immutable, JSON-domain copy with deterministic key order."""

    if value is None or type(value) in {bool, int, str}:  # noqa: E721
        return value
    if type(value) is float:  # noqa: E721
        if not math.isfinite(value):
            raise ValueError(f"non-finite evidence value at {_path}")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        keys = tuple(value.keys())
        for key in keys:
            if type(key) is not str or not key:  # noqa: E721
                raise ValueError(f"evidence keys must be non-empty strings at {_path}")
        for key in sorted(keys):
            normalized[key] = freeze_json_evidence(
                value[key],
                _path=f"{_path}.{key}",
            )
        return MappingProxyType(normalized)
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_json_evidence(item, _path=f"{_path}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValueError(f"unsupported evidence value at {_path}: {type(value).__name__}")


def thaw_json_evidence(value: Any) -> Any:
    """Return a plain JSON-serializable copy of deeply frozen evidence."""

    if isinstance(value, Mapping):
        return {str(key): thaw_json_evidence(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json_evidence(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-domain evidence canonically for event and receipt hashes."""

    frozen = freeze_json_evidence(value)
    return json.dumps(
        thaw_json_evidence(frozen),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _required_text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text


def _required_sha256(value: object, label: str) -> str:
    text = _required_text(value, label)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


@dataclass(frozen=True)
class RawDepthSnapshot:
    """One immutable, non-derived depth snapshot admitted to a cohort."""

    camera_id: str
    storage_key: str
    timestamp_us: int
    snapshot_id: str
    artifact_ref: str
    content_sha256: str
    sequence: int
    manifest_sha256: str
    source_id: int | None = None
    source_frame_number: int | None = None
    source_media_pts_ns: int | None = None
    snapshot_role: str = _RAW_ROLE
    fusion_level: None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "camera_id", _required_text(self.camera_id, "camera_id")
        )
        object.__setattr__(
            self, "storage_key", _required_text(self.storage_key, "storage_key")
        )
        object.__setattr__(
            self, "snapshot_id", _required_text(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self, "artifact_ref", _required_text(self.artifact_ref, "artifact_ref")
        )
        object.__setattr__(
            self,
            "content_sha256",
            _required_sha256(self.content_sha256, "content_sha256"),
        )
        object.__setattr__(
            self,
            "manifest_sha256",
            _required_sha256(self.manifest_sha256, "manifest_sha256"),
        )
        if type(self.timestamp_us) is not int or self.timestamp_us <= 0:  # noqa: E721
            raise ValueError("timestamp_us must be a positive integer")
        if type(self.sequence) is not int or self.sequence <= 0:  # noqa: E721
            raise ValueError("sequence must be a positive integer")
        source_identity = (
            self.source_id,
            self.source_frame_number,
            self.source_media_pts_ns,
        )
        if any(value is not None for value in source_identity):
            if not all(type(value) is int for value in source_identity):  # noqa: E721
                raise ValueError(
                    "source_id, source_frame_number, and source_media_pts_ns "
                    "must be provided together as integers"
                )
            if (
                int(self.source_id) < 0
                or int(self.source_frame_number) < 0
                or int(self.source_media_pts_ns) < 0
                or int(self.source_media_pts_ns) == (1 << 64) - 1
            ):
                raise ValueError("raw source-frame identity is invalid")
        if self.snapshot_role != _RAW_ROLE or self.fusion_level is not None:
            raise ValueError("capture-event cohorts accept raw snapshots only")


@dataclass(frozen=True)
class TimestampedRgbFrame:
    """A pipeline-owned RGB frame plus non-pixel evidence used for association."""

    camera_id: str
    source_id: int
    batch_id: int
    captured_at_us: int
    frame_id: int
    source_media_pts_ns: int
    width: int
    height: int
    content_sha256: str
    pixels: Any = field(repr=False, compare=False)
    color_space: str = "rgb8"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "camera_id", _required_text(self.camera_id, "camera_id")
        )
        object.__setattr__(
            self,
            "content_sha256",
            _required_sha256(self.content_sha256, "content_sha256"),
        )
        if type(self.source_id) is not int or self.source_id < 0:  # noqa: E721
            raise ValueError("source_id must be a non-negative integer")
        if type(self.batch_id) is not int or self.batch_id < 0:  # noqa: E721
            raise ValueError("batch_id must be a non-negative integer")
        if type(self.captured_at_us) is not int or self.captured_at_us <= 0:  # noqa: E721
            raise ValueError("captured_at_us must be a positive integer")
        if type(self.frame_id) is not int or self.frame_id < 0:  # noqa: E721
            raise ValueError("frame_id must be a non-negative integer")
        if (
            type(self.source_media_pts_ns) is not int  # noqa: E721
            or self.source_media_pts_ns < 0
            or self.source_media_pts_ns == (1 << 64) - 1
        ):
            raise ValueError(
                "source_media_pts_ns must be a valid non-negative GStreamer timestamp"
            )
        if type(self.width) is not int or type(self.height) is not int:  # noqa: E721
            raise ValueError("width and height must be integers")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.color_space != "rgb8":
            raise ValueError("color_space must be rgb8")
        if self.pixels is None:
            raise ValueError("pixels are required")


class TimestampedRgbProvider(Protocol):
    """Returns an already-produced frame associated with the depth cohort."""

    def provide(
        self,
        camera_id: str,
        *,
        cohort: Sequence[RawDepthSnapshot],
    ) -> TimestampedRgbFrame | None: ...


@dataclass(frozen=True)
class FusedDepthSnapshot:
    """Exact persisted result returned by a runtime storage adapter."""

    camera_id: str
    storage_key: str
    timestamp_us: int
    snapshot_id: str
    artifact_ref: str
    content_sha256: str
    sequence: int
    manifest_sha256: str
    event_id: str
    source_snapshot_ids: tuple[str, ...]
    quality_evidence: Mapping[str, Any] = field(default_factory=dict)
    snapshot_role: str = _FUSED_ROLE
    fusion_level: str = _INTRA_CAPTURE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "camera_id", _required_text(self.camera_id, "camera_id")
        )
        object.__setattr__(
            self, "storage_key", _required_text(self.storage_key, "storage_key")
        )
        object.__setattr__(
            self, "snapshot_id", _required_text(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self, "artifact_ref", _required_text(self.artifact_ref, "artifact_ref")
        )
        object.__setattr__(self, "event_id", _required_text(self.event_id, "event_id"))
        object.__setattr__(
            self,
            "content_sha256",
            _required_sha256(self.content_sha256, "content_sha256"),
        )
        object.__setattr__(
            self,
            "manifest_sha256",
            _required_sha256(self.manifest_sha256, "manifest_sha256"),
        )
        object.__setattr__(
            self,
            "source_snapshot_ids",
            tuple(
                _required_text(value, "source_snapshot_id")
                for value in self.source_snapshot_ids
            ),
        )
        quality_evidence = freeze_json_evidence(self.quality_evidence)
        if not isinstance(quality_evidence, Mapping):
            raise ValueError("quality_evidence must be a mapping")
        object.__setattr__(self, "quality_evidence", quality_evidence)
        if type(self.timestamp_us) is not int or self.timestamp_us <= 0:  # noqa: E721
            raise ValueError("timestamp_us must be a positive integer")
        if type(self.sequence) is not int or self.sequence <= 0:  # noqa: E721
            raise ValueError("sequence must be a positive integer")
        if self.snapshot_role != _FUSED_ROLE or self.fusion_level != _INTRA_CAPTURE:
            raise ValueError("fused snapshot role or level is invalid")
        if not self.source_snapshot_ids:
            raise ValueError("source_snapshot_ids are required")
        if len(set(self.source_snapshot_ids)) != len(self.source_snapshot_ids):
            raise ValueError("source_snapshot_ids must be unique")


class CaptureEventStore(Protocol):
    """Minimal persistence boundary required by the pure coordinator."""

    def list_raw_snapshots(
        self,
        storage_key: str,
        *,
        camera_id: str,
        after_timestamp_us: int,
        limit: int,
    ) -> Sequence[RawDepthSnapshot]: ...

    def fuse_raw_snapshots(
        self,
        storage_key: str,
        snapshots: Sequence[RawDepthSnapshot],
        *,
        rgb_frame: TimestampedRgbFrame | None,
        event_id: str,
        min_observations: int,
        depth_agreement_m: float,
    ) -> FusedDepthSnapshot: ...


@dataclass(frozen=True)
class CaptureEventFusionRequest:
    camera_id: str
    storage_keys: tuple[str, ...]
    baseline_timestamp_us: Mapping[str, int]
    cache_only: bool = False
    require_rgb: bool = True
    raw_limit: int = 24
    min_observations: int = 2
    depth_agreement_m: float = 0.18
    max_cohort_span_us: int = 20_000_000
    max_rgb_skew_us: int = 2_000_000

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "camera_id", _required_text(self.camera_id, "camera_id")
        )
        keys = tuple(
            _required_text(value, "storage_key") for value in self.storage_keys
        )
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("storage_keys must be non-empty and unique")
        object.__setattr__(self, "storage_keys", keys)
        baseline = dict(self.baseline_timestamp_us)
        if set(baseline) != set(keys):
            raise ValueError("baseline_timestamp_us must exactly cover storage_keys")
        normalized_baseline: dict[str, int] = {}
        for key in keys:
            value = baseline[key]
            if type(value) is not int or value < 0:  # noqa: E721
                raise ValueError("baseline timestamps must be non-negative integers")
            normalized_baseline[key] = value
        object.__setattr__(
            self, "baseline_timestamp_us", MappingProxyType(normalized_baseline)
        )
        if type(self.cache_only) is not bool or type(self.require_rgb) is not bool:  # noqa: E721
            raise ValueError("cache_only and require_rgb must be booleans")
        if type(self.raw_limit) is not int or not 1 <= self.raw_limit <= 256:  # noqa: E721
            raise ValueError("raw_limit must be in [1, 256]")
        if (  # noqa: E721
            type(self.min_observations) is not int
            or not 1 <= self.min_observations <= self.raw_limit
        ):
            raise ValueError("min_observations must be in [1, raw_limit]")
        if not math.isfinite(self.depth_agreement_m) or self.depth_agreement_m <= 0.0:
            raise ValueError("depth_agreement_m must be finite and positive")
        if (  # noqa: E721
            type(self.max_cohort_span_us) is not int
            or type(self.max_rgb_skew_us) is not int
            or self.max_cohort_span_us <= 0
            or self.max_rgb_skew_us < 0
        ):
            raise ValueError(
                "cohort span must be positive and RGB skew cannot be negative"
            )


@dataclass(frozen=True)
class CaptureEventFusionOutcome:
    fused_snapshot: FusedDepthSnapshot
    evidence: Mapping[str, Any]
    evidence_sha256: str

    def __post_init__(self) -> None:
        frozen = freeze_json_evidence(self.evidence)
        if not isinstance(frozen, Mapping):
            raise ValueError("evidence must be a mapping")
        object.__setattr__(self, "evidence", frozen)
        object.__setattr__(
            self,
            "evidence_sha256",
            _required_sha256(self.evidence_sha256, "evidence_sha256"),
        )
        if canonical_json_sha256(frozen) != self.evidence_sha256:
            raise ValueError("evidence_sha256 does not match immutable evidence")

    def evidence_payload(self) -> dict[str, Any]:
        return thaw_json_evidence(self.evidence)


class CaptureEventFusionError(RuntimeError):
    def __init__(self, code: str, *, details: Mapping[str, Any] | None = None) -> None:
        self.code = _required_text(code, "code")
        frozen = freeze_json_evidence(details or {})
        if not isinstance(frozen, Mapping):
            raise ValueError("error details must be a mapping")
        self.details = frozen
        super().__init__(self.code)


def _snapshot_evidence(snapshot: RawDepthSnapshot) -> dict[str, Any]:
    evidence = {
        "snapshot_id": snapshot.snapshot_id,
        "timestamp_us": snapshot.timestamp_us,
        "artifact_ref": snapshot.artifact_ref,
        "content_sha256": snapshot.content_sha256,
        "sequence": snapshot.sequence,
        "manifest_sha256": snapshot.manifest_sha256,
        "snapshot_role": snapshot.snapshot_role,
        "fusion_level": snapshot.fusion_level,
    }
    if snapshot.source_id is not None:
        evidence["source_id"] = snapshot.source_id
        evidence["source_frame_number"] = snapshot.source_frame_number
        evidence["source_media_pts_ns"] = snapshot.source_media_pts_ns
    return evidence


def _rgb_evidence(
    frame: TimestampedRgbFrame | None,
    *,
    requested: bool,
    provider_configured: bool,
) -> dict[str, Any]:
    if frame is None:
        return {
            "status": "unavailable" if requested else "not_requested",
            "provider_configured": provider_configured,
        }
    return {
        "status": "available",
        "provider_configured": provider_configured,
        "source_id": frame.source_id,
        "batch_id": frame.batch_id,
        "captured_at_us": frame.captured_at_us,
        "frame_id": frame.frame_id,
        "source_media_pts_ns": frame.source_media_pts_ns,
        "width": frame.width,
        "height": frame.height,
        "color_space": frame.color_space,
        "content_sha256": frame.content_sha256,
    }


def _event_id(payload: Mapping[str, Any]) -> str:
    return f"capture-event-sha256:{canonical_json_sha256(payload)}"


class CaptureEventFusionCoordinator:
    """Select one raw cohort, optionally bind RGB, fuse once, and seal evidence."""

    def __init__(
        self,
        store: CaptureEventStore,
        *,
        rgb_provider: TimestampedRgbProvider | None = None,
        failure_callback: Callable[[CaptureEventFusionError], None] | None = None,
    ) -> None:
        self._store = store
        self._rgb_provider = rgb_provider
        self._failure_callback = failure_callback

    def _fail(self, code: str, **details: Any) -> CaptureEventFusionError:
        error = CaptureEventFusionError(code, details=details)
        if self._failure_callback is not None:
            try:
                self._failure_callback(error)
            except Exception:
                # The original coordinator failure remains authoritative.
                pass
        return error

    def fuse(self, request: CaptureEventFusionRequest) -> CaptureEventFusionOutcome:
        if request.cache_only:
            raise self._fail(
                "cache_only_forbids_capture_event_fusion",
                camera_id=request.camera_id,
            )

        selected_key: str | None = None
        cohort: tuple[RawDepthSnapshot, ...] = ()
        for key in request.storage_keys:
            try:
                rows = tuple(
                    self._store.list_raw_snapshots(
                        key,
                        camera_id=request.camera_id,
                        after_timestamp_us=request.baseline_timestamp_us[key],
                        # N+1 proves the bounded cohort was not silently
                        # truncated by the storage adapter.
                        limit=request.raw_limit + 1,
                    )
                )
            except Exception as exc:
                raise self._fail(
                    "raw_snapshot_listing_failed",
                    camera_id=request.camera_id,
                    storage_key=key,
                    exception_type=type(exc).__name__,
                ) from exc
            if not rows:
                continue
            selected_key = key
            cohort = self._validate_cohort(request, key, rows)
            break

        if selected_key is None:
            raise self._fail(
                "no_raw_snapshots_for_capture_event",
                camera_id=request.camera_id,
                storage_keys=list(request.storage_keys),
            )
        if len(cohort) < request.min_observations:
            raise self._fail(
                "insufficient_raw_observations",
                camera_id=request.camera_id,
                storage_key=selected_key,
                observed=len(cohort),
                required=request.min_observations,
            )

        start_us = cohort[0].timestamp_us
        end_us = cohort[-1].timestamp_us
        rgb_frame: TimestampedRgbFrame | None = None
        if request.require_rgb and self._rgb_provider is not None:
            if any(row.source_id is None for row in cohort):
                raise self._fail(
                    "rgb_depth_identity_unavailable",
                    camera_id=request.camera_id,
                    storage_key=selected_key,
                    snapshot_ids=[row.snapshot_id for row in cohort],
                )
            try:
                rgb_frame = self._rgb_provider.provide(
                    request.camera_id,
                    cohort=cohort,
                )
            except Exception as exc:
                raise self._fail(
                    "rgb_provider_failed",
                    camera_id=request.camera_id,
                    exception_type=type(exc).__name__,
                ) from exc
        if rgb_frame is None and request.require_rgb:
            raise self._fail(
                "rgb_frame_unavailable",
                camera_id=request.camera_id,
                provider_configured=self._rgb_provider is not None,
            )
        if rgb_frame is not None:
            if rgb_frame.camera_id != request.camera_id:
                raise self._fail(
                    "rgb_camera_mismatch",
                    expected=request.camera_id,
                    observed=rgb_frame.camera_id,
                )
            matching_rows = tuple(
                row
                for row in cohort
                if (
                    row.source_id == rgb_frame.source_id
                    and row.source_frame_number == rgb_frame.frame_id
                    and row.source_media_pts_ns == rgb_frame.source_media_pts_ns
                )
            )
            if not matching_rows:
                raise self._fail(
                    "rgb_frame_identity_mismatch",
                    camera_id=request.camera_id,
                    source_id=rgb_frame.source_id,
                    source_frame_number=rgb_frame.frame_id,
                    source_media_pts_ns=rgb_frame.source_media_pts_ns,
                )
            if len(matching_rows) != 1:
                raise self._fail(
                    "rgb_frame_identity_ambiguous",
                    camera_id=request.camera_id,
                    source_id=rgb_frame.source_id,
                    source_frame_number=rgb_frame.frame_id,
                    source_media_pts_ns=rgb_frame.source_media_pts_ns,
                )

        pre_fusion_evidence = {
            "contract": "noesis.capture_event_fusion",
            "contract_version": 1,
            "camera_id": request.camera_id,
            "storage_key": selected_key,
            "cache_only": False,
            "baseline_timestamp_us": request.baseline_timestamp_us[selected_key],
            "cohort_start_us": start_us,
            "cohort_end_us": end_us,
            "raw_snapshot_count": len(cohort),
            "raw_snapshots": [_snapshot_evidence(row) for row in cohort],
            "rgb": _rgb_evidence(
                rgb_frame,
                requested=request.require_rgb,
                provider_configured=self._rgb_provider is not None,
            ),
            "parameters": {
                "raw_limit": request.raw_limit,
                "min_observations": request.min_observations,
                "depth_agreement_m": request.depth_agreement_m,
                "max_cohort_span_us": request.max_cohort_span_us,
                "max_rgb_skew_us": request.max_rgb_skew_us,
            },
        }
        event_id = _event_id(pre_fusion_evidence)
        try:
            fused = self._store.fuse_raw_snapshots(
                selected_key,
                cohort,
                rgb_frame=rgb_frame,
                event_id=event_id,
                min_observations=request.min_observations,
                depth_agreement_m=request.depth_agreement_m,
            )
        except Exception as exc:
            # Storage owns dense-depth quality measurement. Preserve its
            # retryable rejection code instead of converting it into the
            # fatal generic fusion failure used for integrity defects.
            if getattr(exc, "code", None) == "capture_event_fusion_quality_rejected":
                details: dict[str, Any] = {
                    "camera_id": request.camera_id,
                    "storage_key": selected_key,
                    "event_id": event_id,
                }
                for key in ("observed", "required"):
                    value = getattr(exc, key, None)
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        details[key] = float(value)
                metric = getattr(exc, "metric", None)
                if isinstance(metric, str) and metric:
                    details["metric"] = metric
                quality_evidence = getattr(exc, "evidence", None)
                if isinstance(quality_evidence, Mapping):
                    details["quality_evidence"] = dict(quality_evidence)
                raise self._fail(
                    "capture_event_fusion_quality_rejected",
                    **details,
                ) from exc
            raise self._fail(
                "capture_event_fusion_failed",
                camera_id=request.camera_id,
                storage_key=selected_key,
                event_id=event_id,
                exception_type=type(exc).__name__,
            ) from exc

        if not isinstance(fused, FusedDepthSnapshot):
            raise self._fail(
                "untyped_fused_snapshot",
                camera_id=request.camera_id,
                storage_key=selected_key,
                event_id=event_id,
                observed_type=type(fused).__name__,
            )
        expected_ids = tuple(row.snapshot_id for row in cohort)
        if (
            fused.camera_id != request.camera_id
            or fused.storage_key != selected_key
            or fused.event_id != event_id
            or fused.source_snapshot_ids != expected_ids
            or fused.timestamp_us <= end_us
            or fused.sequence <= max(row.sequence for row in cohort)
        ):
            raise self._fail(
                "fused_snapshot_evidence_mismatch",
                camera_id=request.camera_id,
                storage_key=selected_key,
                event_id=event_id,
            )

        fused_snapshot_evidence = {
            "snapshot_id": fused.snapshot_id,
            "timestamp_us": fused.timestamp_us,
            "artifact_ref": fused.artifact_ref,
            "content_sha256": fused.content_sha256,
            "sequence": fused.sequence,
            "manifest_sha256": fused.manifest_sha256,
            "snapshot_role": fused.snapshot_role,
            "fusion_level": fused.fusion_level,
            "source_snapshot_ids": list(fused.source_snapshot_ids),
        }
        if fused.quality_evidence:
            fused_snapshot_evidence["quality_evidence"] = thaw_json_evidence(
                fused.quality_evidence
            )
        evidence = {
            **pre_fusion_evidence,
            "event_id": event_id,
            "fused_snapshot": fused_snapshot_evidence,
        }
        evidence_sha256 = canonical_json_sha256(evidence)
        return CaptureEventFusionOutcome(
            fused_snapshot=fused,
            evidence=evidence,
            evidence_sha256=evidence_sha256,
        )

    def _validate_cohort(
        self,
        request: CaptureEventFusionRequest,
        storage_key: str,
        rows: Sequence[RawDepthSnapshot],
    ) -> tuple[RawDepthSnapshot, ...]:
        if len(rows) > request.raw_limit:
            raise self._fail(
                "raw_snapshot_limit_exceeded",
                storage_key=storage_key,
                observed=len(rows),
                limit=request.raw_limit,
            )
        for row in rows:
            if not isinstance(row, RawDepthSnapshot):
                raise self._fail("untyped_raw_snapshot", storage_key=storage_key)
        ordered = tuple(sorted(rows, key=lambda row: row.timestamp_us))
        snapshot_ids: set[str] = set()
        timestamps: set[int] = set()
        artifact_refs: set[str] = set()
        sequences: set[int] = set()
        manifests: set[str] = set()
        source_identities: set[tuple[int, int, int]] = set()
        for row in ordered:
            if row.camera_id != request.camera_id or row.storage_key != storage_key:
                raise self._fail(
                    "raw_snapshot_scope_mismatch",
                    storage_key=storage_key,
                    snapshot_id=row.snapshot_id,
                )
            if row.timestamp_us <= request.baseline_timestamp_us[storage_key]:
                raise self._fail(
                    "raw_snapshot_precedes_baseline",
                    storage_key=storage_key,
                    snapshot_id=row.snapshot_id,
                )
            if (
                row.snapshot_id in snapshot_ids
                or row.timestamp_us in timestamps
                or row.artifact_ref in artifact_refs
                or row.sequence in sequences
                or row.manifest_sha256 in manifests
            ):
                raise self._fail(
                    "duplicate_raw_snapshot_evidence",
                    storage_key=storage_key,
                    snapshot_id=row.snapshot_id,
                )
            snapshot_ids.add(row.snapshot_id)
            timestamps.add(row.timestamp_us)
            artifact_refs.add(row.artifact_ref)
            sequences.add(row.sequence)
            manifests.add(row.manifest_sha256)
            if row.source_id is not None:
                source_identity = (
                    int(row.source_id),
                    int(row.source_frame_number),
                    int(row.source_media_pts_ns),
                )
                if source_identity in source_identities:
                    raise self._fail(
                        "duplicate_raw_source_frame_identity",
                        storage_key=storage_key,
                        source_id=source_identity[0],
                        source_frame_number=source_identity[1],
                        source_media_pts_ns=source_identity[2],
                    )
                source_identities.add(source_identity)
        span_us = ordered[-1].timestamp_us - ordered[0].timestamp_us
        if span_us > request.max_cohort_span_us:
            raise self._fail(
                "raw_snapshot_cohort_too_wide",
                storage_key=storage_key,
                span_us=span_us,
                limit_us=request.max_cohort_span_us,
            )
        return ordered


__all__ = [
    "CaptureEventFusionCoordinator",
    "CaptureEventFusionError",
    "CaptureEventFusionOutcome",
    "CaptureEventFusionRequest",
    "CaptureEventStore",
    "FusedDepthSnapshot",
    "RawDepthSnapshot",
    "TimestampedRgbFrame",
    "TimestampedRgbProvider",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "freeze_json_evidence",
    "thaw_json_evidence",
]
