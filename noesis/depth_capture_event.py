"""Typed adapter from transactional depth storage to capture-event fusion."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from geometry.depth_source import (
    DepthFusionQualityError,
    DepthStorageError,
    DepthStorageManager,
    SnapshotDescriptor,
)
from noesis_core.capture_event_fusion import (
    FusedDepthSnapshot,
    RawDepthSnapshot,
    TimestampedRgbFrame,
)
from noesis.mapanything_manual_inference import (
    MapAnythingManualInferenceResult,
    measure_mapanything_scene_luminance,
)

logger = logging.getLogger(__name__)

DEFAULT_MANUAL_SCENE_SEVERE_DARK_P90 = 25.0
DEFAULT_MANUAL_SCENE_SEVERE_DARK_P99_P50_RANGE = 20.0


def _artifact_ref(descriptor: SnapshotDescriptor) -> str:
    return f"depth-zarr:{descriptor.storage_ref}"


def _raw_role(descriptor: SnapshotDescriptor) -> bool:
    role = descriptor.snapshot_role.strip().lower()
    level = descriptor.fusion_level.strip().lower()
    return role in {"", "raw"} and not level


@dataclass(frozen=True)
class DepthStorageFlushEvidence:
    """Portable exact storage-frontier receipt used by capture barriers."""

    frontier_sequence: int
    completed: bool
    timed_out: bool
    pending_sequences: tuple[int, ...]
    failed_sequences: tuple[int, ...]
    poisoned: bool

    @property
    def clean(self) -> bool:
        return bool(
            self.completed
            and not self.timed_out
            and not self.pending_sequences
            and not self.failed_sequences
            and not self.poisoned
        )


class ManualDepthInferencer(Protocol):
    """Minimal deterministic inference boundary used by manual capture."""

    def infer(self, rgb: np.ndarray) -> MapAnythingManualInferenceResult: ...


class DepthStorageCaptureEventAdapter:
    """Expose only exact committed raw snapshots to the pure coordinator."""

    def __init__(
        self,
        storage: DepthStorageManager,
        *,
        manual_inferencer: ManualDepthInferencer | None = None,
        severe_dark_p90_threshold: float = DEFAULT_MANUAL_SCENE_SEVERE_DARK_P90,
        severe_dark_p99_p50_range_threshold: float = (
            DEFAULT_MANUAL_SCENE_SEVERE_DARK_P99_P50_RANGE
        ),
    ) -> None:
        if not isinstance(storage, DepthStorageManager):
            raise TypeError("storage must be DepthStorageManager")
        dark_p90 = float(severe_dark_p90_threshold)
        dark_range = float(severe_dark_p99_p50_range_threshold)
        if (
            not math.isfinite(dark_p90)
            or not 0.0 < dark_p90 <= 255.0
            or not math.isfinite(dark_range)
            or not 0.0 < dark_range <= 255.0
        ):
            raise ValueError(
                "manual scene severe-dark thresholds must be finite in (0, 255]"
            )
        self._storage = storage
        self._manual_inferencer = manual_inferencer
        self._severe_dark_p90_threshold = dark_p90
        self._severe_dark_p99_p50_range_threshold = dark_range

    def flush_capture_frontier(
        self,
        *,
        timeout_s: float,
    ) -> DepthStorageFlushEvidence:
        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("capture-event storage flush timeout must be positive")
        receipt = self._storage.flush(timeout=timeout)
        return DepthStorageFlushEvidence(
            frontier_sequence=int(receipt.frontier_sequence),
            completed=bool(receipt.completed),
            timed_out=bool(receipt.timed_out),
            pending_sequences=tuple(int(value) for value in receipt.pending_sequences),
            failed_sequences=tuple(int(value) for value in receipt.failed_sequences),
            poisoned=receipt.poison is not None,
        )

    def max_raw_timestamp(self, camera_id: str) -> int:
        camera = str(camera_id or "").strip()
        if not camera:
            raise DepthStorageError("canonical camera id is required")
        rows = self.list_raw_snapshots(
            camera,
            camera_id=camera,
            after_timestamp_us=0,
            limit=1,
        )
        return int(rows[-1].timestamp_us) if rows else 0

    def list_raw_snapshots(
        self,
        storage_key: str,
        *,
        camera_id: str,
        after_timestamp_us: int,
        limit: int,
    ) -> Sequence[RawDepthSnapshot]:
        key = str(storage_key or "").strip()
        camera = str(camera_id or "").strip()
        if not key or not camera or key != camera:
            raise DepthStorageError(
                "capture-event storage key must equal the canonical camera id"
            )
        rows = self._storage.list_snapshot_entries(
            key,
            ts_min_exclusive=int(after_timestamp_us),
            include_derived=False,
            limit=int(limit),
        )
        snapshots: list[RawDepthSnapshot] = []
        for timestamp_us, path in rows:
            descriptor = self._storage.describe_snapshot(path)
            if (
                descriptor.camera_id != camera
                or descriptor.ts_us != int(timestamp_us)
                or not _raw_role(descriptor)
            ):
                raise DepthStorageError(
                    f"capture-event source is not exact canonical raw depth: {descriptor.storage_ref}"
                )
            source_identity = (
                descriptor.source_id,
                descriptor.source_frame_number,
                descriptor.source_media_pts_ns,
            )
            exact_source_identity = all(
                value is not None for value in source_identity
            )
            snapshots.append(
                RawDepthSnapshot(
                    camera_id=camera,
                    storage_key=key,
                    timestamp_us=descriptor.ts_us,
                    snapshot_id=descriptor.write_id,
                    artifact_ref=_artifact_ref(descriptor),
                    content_sha256=descriptor.content_sha256,
                    sequence=descriptor.sequence,
                    manifest_sha256=descriptor.manifest_sha256,
                    source_id=(
                        int(descriptor.source_id)
                        if exact_source_identity
                        else None
                    ),
                    source_frame_number=(
                        int(descriptor.source_frame_number)
                        if exact_source_identity
                        else None
                    ),
                    source_media_pts_ns=(
                        int(descriptor.source_media_pts_ns)
                        if exact_source_identity
                        else None
                    ),
                )
            )
        return tuple(snapshots)

    def _resolve_raw_snapshot(
        self,
        storage_key: str,
        snapshot: RawDepthSnapshot,
    ) -> tuple[int, Any]:
        rows = self._storage.list_snapshot_entries(
            storage_key,
            ts_min_exclusive=snapshot.timestamp_us - 1,
            ts_max_us=snapshot.timestamp_us,
            include_derived=True,
            limit=1,
        )
        if len(rows) != 1 or int(rows[0][0]) != snapshot.timestamp_us:
            raise DepthStorageError(
                f"sealed capture-event source is no longer committed: {snapshot.snapshot_id}"
            )
        timestamp_us, path = rows[0]
        descriptor = self._storage.describe_snapshot(path)
        if (
            not _raw_role(descriptor)
            or descriptor.camera_id != snapshot.camera_id
            or descriptor.write_id != snapshot.snapshot_id
            or descriptor.content_sha256 != snapshot.content_sha256
            or _artifact_ref(descriptor) != snapshot.artifact_ref
            or (
                snapshot.source_id is not None
                and (
                    descriptor.source_id != snapshot.source_id
                    or descriptor.source_frame_number
                    != snapshot.source_frame_number
                    or descriptor.source_media_pts_ns
                    != snapshot.source_media_pts_ns
                )
            )
        ):
            raise DepthStorageError(
                f"sealed capture-event source identity changed: {snapshot.snapshot_id}"
            )
        return int(timestamp_us), path

    @staticmethod
    def _rgb_pixels(frame: TimestampedRgbFrame | None) -> np.ndarray | None:
        if frame is None:
            return None
        pixels = np.asarray(frame.pixels)
        if (
            pixels.dtype != np.uint8
            or pixels.ndim != 3
            or pixels.shape != (frame.height, frame.width, 3)
        ):
            raise DepthStorageError(
                "timestamped RGB pixels do not match their declared uint8 HxWx3 contract"
            )
        contiguous = np.ascontiguousarray(pixels)
        digest = hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()
        if digest != frame.content_sha256:
            raise DepthStorageError("timestamped RGB content digest mismatch")
        return contiguous

    def fuse_raw_snapshots(
        self,
        storage_key: str,
        snapshots: Sequence[RawDepthSnapshot],
        *,
        rgb_frame: TimestampedRgbFrame | None,
        event_id: str,
        min_observations: int,
        depth_agreement_m: float,
    ) -> FusedDepthSnapshot:
        if not snapshots:
            raise DepthStorageError("capture-event fusion requires raw snapshots")
        key = str(storage_key or "").strip()
        if key != snapshots[0].camera_id:
            raise DepthStorageError(
                "capture-event storage key must equal the canonical camera id"
            )
        entries: list[tuple[int, Any]] = []
        for snapshot in snapshots:
            if not isinstance(snapshot, RawDepthSnapshot):
                raise DepthStorageError(
                    "capture-event fusion received an untyped source"
                )
            if snapshot.storage_key != key:
                raise DepthStorageError("capture-event source storage key changed")
            entries.append(self._resolve_raw_snapshot(key, snapshot))
        rgb = self._rgb_pixels(rgb_frame)
        scene_quality_evidence: dict[str, Any] | None = None
        if rgb is not None:
            scene_luminance = measure_mapanything_scene_luminance(rgb)
            observed_p90 = float(scene_luminance["p90"])
            observed_range = float(scene_luminance["p99_p50_range"])
            severe_dark = bool(
                observed_p90 < self._severe_dark_p90_threshold
                and observed_range
                < self._severe_dark_p99_p50_range_threshold
            )
            visibility_ratio = max(
                observed_p90 / self._severe_dark_p90_threshold,
                observed_range
                / self._severe_dark_p99_p50_range_threshold,
            )
            scene_quality_evidence = {
                **dict(scene_luminance),
                "gate": {
                    "contract": "noesis.mapanything.manual_scene_quality_gate.v1",
                    "metric": "manual_rgb_scene_visibility_ratio",
                    "p90_threshold": self._severe_dark_p90_threshold,
                    "p99_p50_range_threshold": (
                        self._severe_dark_p99_p50_range_threshold
                    ),
                    "visibility_ratio": float(visibility_ratio),
                    "passed": not severe_dark,
                    "reason": (
                        "severe_dark_scene"
                        if severe_dark
                        else "scene_visibility_accepted"
                    ),
                },
            }
            log = logger.warning if severe_dark else logger.info
            log(
                "Manual MapAnything scene quality camera=%s p50=%.3f "
                "p90=%.3f p99=%.3f p99_p50_range=%.3f "
                "sampled_fraction=%.6f reason=%s",
                key,
                float(scene_luminance["p50"]),
                observed_p90,
                float(scene_luminance["p99"]),
                observed_range,
                float(scene_luminance["sampled_fraction"]),
                scene_quality_evidence["gate"]["reason"],
            )
            if severe_dark:
                raise DepthFusionQualityError(
                    observed=float(visibility_ratio),
                    required=1.0,
                    metric="manual_rgb_scene_visibility_ratio",
                    evidence=scene_quality_evidence,
                )
        if self._manual_inferencer is None:
            path, meta = self._storage.fuse_snapshot_entries(
                key,
                entries,
                rgb=rgb,
                min_observations=int(min_observations),
                depth_agreement_m=float(depth_agreement_m),
                snapshot_role="capture_event_fused",
                fusion_level="intra_capture",
                event_id=str(event_id),
            )
        else:
            if rgb is None or rgb_frame is None:
                raise DepthStorageError(
                    "deterministic manual inference requires exact RGB"
                )
            result = self._manual_inferencer.infer(rgb)
            expected_shape = (int(rgb_frame.height), int(rgb_frame.width))
            if (
                not isinstance(result, MapAnythingManualInferenceResult)
                or result.depth.dtype != np.float32
                or result.confidence.dtype != np.float32
                or result.mask.dtype != np.uint8
                or result.depth.shape != expected_shape
                or result.confidence.shape != expected_shape
                or result.mask.shape != expected_shape
            ):
                raise DepthStorageError(
                    "deterministic manual inference returned an invalid result"
                )
            source_descriptors = [
                self._storage.describe_snapshot(path)
                for _timestamp_us, path in entries
            ]
            source_evidence = [
                {
                    "camera_id": descriptor.camera_id,
                    "timestamp_us": descriptor.ts_us,
                    "write_id": descriptor.write_id,
                    "sequence": descriptor.sequence,
                    "path": str(descriptor.path),
                    "storage_ref": descriptor.storage_ref,
                    "manifest_sha256": descriptor.manifest_sha256,
                    "content_sha256": descriptor.content_sha256,
                    "snapshot_role": descriptor.snapshot_role,
                    "fusion_level": descriptor.fusion_level,
                }
                for descriptor in source_descriptors
            ]
            source_timestamps = [
                int(descriptor.ts_us) for descriptor in source_descriptors
            ]
            source_paths = [
                str(descriptor.path) for descriptor in source_descriptors
            ]
            latest_rows = self._storage.list_snapshot_entries(
                key,
                include_derived=True,
                limit=1,
            )
            latest_timestamp = int(latest_rows[-1][0]) if latest_rows else 0
            fused_timestamp = max(
                time.time_ns() // 1_000,
                max(source_timestamps) + 1,
                latest_timestamp + 1,
            )
            valid_fraction = float(
                np.count_nonzero(result.mask) / max(1, result.mask.size)
            )
            deterministic_inference_evidence = {
                **dict(result.evidence),
                "scene_quality": dict(scene_quality_evidence or {}),
            }
            meta = {
                "source_snapshot_paths": source_paths,
                "source_timestamps_us": source_timestamps,
                "source_snapshot_count": len(source_descriptors),
                "source_snapshots": source_evidence,
                "event_id": str(event_id),
                "support_valid_fraction": valid_fraction,
                "median_support": 1.0,
                "support_evidence": {
                    "contract": "noesis.mapanything.manual_inference.support.v1",
                    "mode": "exact_rgb_deterministic",
                    "output_full_frame_fraction": valid_fraction,
                },
                "support_quality_gate": {
                    "contract": "noesis.depth.fusion.quality_gate.v1",
                    "metric": "deterministic_valid_fraction",
                    "observed": valid_fraction,
                    "required": 0.0,
                    "passed": True,
                },
                "frame_scale_normalization": {
                    "contract": "noesis.mapanything.manual_scale.v1",
                    "applied": False,
                    "reason": "single_exact_rgb_inference",
                },
                "deterministic_inference": deterministic_inference_evidence,
            }
            attrs = {
                "snapshot_role": "capture_event_fused",
                "fusion_level": "intra_capture",
                "fusion_meta": json.dumps(meta, separators=(",", ":")),
                "source_snapshot_paths": json.dumps(
                    source_paths,
                    separators=(",", ":"),
                ),
                "source_timestamps_us": json.dumps(
                    source_timestamps,
                    separators=(",", ":"),
                ),
                "source_snapshot_count": len(source_descriptors),
                "source_snapshots": json.dumps(
                    source_evidence,
                    separators=(",", ":"),
                ),
                "event_id": str(event_id),
                "event_start_ts_us": min(source_timestamps),
                "event_end_ts_us": max(source_timestamps),
                "capture_event_depth_mode": "exact_rgb_deterministic",
                "deterministic_inference": json.dumps(
                    deterministic_inference_evidence,
                    separators=(",", ":"),
                ),
            }
            handle = self._storage.store(
                key,
                fused_timestamp,
                result.depth,
                result.confidence,
                result.mask,
                rgb=rgb,
                attrs=attrs,
            )
            receipt = handle.wait(timeout=self._storage.public_commit_timeout_s)
            path = receipt.path
            self._storage.invalidate_floorplan_cache(key)
        descriptor = self._storage.describe_snapshot(path)
        source_rows = meta.get("source_snapshots")
        if not isinstance(source_rows, list):
            raise DepthStorageError("fused snapshot omitted exact source evidence")
        source_ids = tuple(str(row.get("write_id") or "") for row in source_rows)
        expected_ids = tuple(snapshot.snapshot_id for snapshot in snapshots)
        if (
            source_ids != expected_ids
            or descriptor.camera_id != snapshots[0].camera_id
            or descriptor.snapshot_role != "capture_event_fused"
            or descriptor.fusion_level != "intra_capture"
            or str(meta.get("event_id") or "") != str(event_id)
        ):
            raise DepthStorageError(
                "fused snapshot identity does not match the sealed cohort"
            )
        quality_evidence = {
            "contract": "noesis.capture_event.depth_quality.v1",
            "support_valid_fraction": float(
                meta.get("support_valid_fraction") or 0.0
            ),
            "median_support": float(meta.get("median_support") or 0.0),
            "support_evidence": dict(meta.get("support_evidence") or {}),
            "support_quality_gate": dict(meta.get("support_quality_gate") or {}),
            "frame_scale_normalization": dict(
                meta.get("frame_scale_normalization") or {}
            ),
        }
        return FusedDepthSnapshot(
            camera_id=descriptor.camera_id,
            storage_key=key,
            timestamp_us=descriptor.ts_us,
            snapshot_id=descriptor.write_id,
            artifact_ref=_artifact_ref(descriptor),
            content_sha256=descriptor.content_sha256,
            sequence=descriptor.sequence,
            manifest_sha256=descriptor.manifest_sha256,
            event_id=str(event_id),
            source_snapshot_ids=source_ids,
            quality_evidence=quality_evidence,
        )

    def validate_fused_snapshot(
        self,
        snapshot: FusedDepthSnapshot,
    ) -> FusedDepthSnapshot:
        """Re-read and match every public field of an exact fused descriptor."""

        if not isinstance(snapshot, FusedDepthSnapshot):
            raise DepthStorageError("fused snapshot evidence is untyped")
        rows = self._storage.list_snapshot_entries(
            snapshot.camera_id,
            ts_min_exclusive=snapshot.timestamp_us - 1,
            ts_max_us=snapshot.timestamp_us,
            include_derived=True,
            limit=2,
        )
        if len(rows) != 1 or int(rows[0][0]) != snapshot.timestamp_us:
            raise DepthStorageError(
                "fused snapshot is no longer an exact committed entry"
            )
        descriptor = self._storage.describe_snapshot(rows[0][1])
        if (
            descriptor.camera_id != snapshot.camera_id
            or descriptor.ts_us != snapshot.timestamp_us
            or descriptor.write_id != snapshot.snapshot_id
            or descriptor.sequence != snapshot.sequence
            or descriptor.manifest_sha256 != snapshot.manifest_sha256
            or descriptor.content_sha256 != snapshot.content_sha256
            or _artifact_ref(descriptor) != snapshot.artifact_ref
            or descriptor.snapshot_role != snapshot.snapshot_role
            or descriptor.fusion_level != snapshot.fusion_level
        ):
            raise DepthStorageError("fused snapshot descriptor evidence changed")
        return snapshot


__all__ = [
    "DEFAULT_MANUAL_SCENE_SEVERE_DARK_P90",
    "DEFAULT_MANUAL_SCENE_SEVERE_DARK_P99_P50_RANGE",
    "DepthStorageCaptureEventAdapter",
    "DepthStorageFlushEvidence",
]
