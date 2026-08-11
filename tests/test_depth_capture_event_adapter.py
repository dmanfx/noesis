from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest

from geometry.depth_source import (
    DepthFusionQualityError,
    DepthStorageError,
    DepthStorageManager,
)
from noesis.depth_capture_event import DepthStorageCaptureEventAdapter
from noesis.mapanything_manual_inference import (
    MapAnythingManualInferenceResult,
    preprocess_mapanything_rgb,
)
from noesis_core.capture_event_fusion import (
    CaptureEventFusionCoordinator,
    CaptureEventFusionError,
    CaptureEventFusionRequest,
    RawDepthSnapshot,
    TimestampedRgbFrame,
)


def _manager(tmp_path) -> DepthStorageManager:  # type: ignore[no-untyped-def]
    return DepthStorageManager(
        tmp_path / "depth",
        max_snapshots_per_camera=16,
        retention_minutes=60.0,
        enable_async=False,
        worker_count=1,
        max_worker_count=1,
        enforce_async=False,
        zarr_clevel=0,
    )


def _arrays(offset: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.full((3, 4), 2.0 + offset, dtype=np.float32)
    confidence = np.full((3, 4), 0.9, dtype=np.float32)
    mask = np.ones((3, 4), dtype=np.uint8)
    return depth, confidence, mask


def _timestamps() -> tuple[int, int, int, int]:
    base = time.time_ns() // 1_000
    return base - 1, base, base + 100_000, base + 200_000


def _close(manager: DepthStorageManager) -> None:
    receipt = manager.shutdown(wait=True, timeout=5.0)
    assert receipt.completed
    assert receipt.flush.poison is None


class _RecordingManualInferencer:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, rgb: np.ndarray) -> MapAnythingManualInferenceResult:
        self.calls += 1
        height, width = rgb.shape[:2]
        return MapAnythingManualInferenceResult(
            depth=np.full((height, width), 2.0, dtype=np.float32),
            confidence=np.full((height, width), 0.75, dtype=np.float32),
            mask=np.ones((height, width), dtype=np.uint8),
            evidence={
                "contract": "test.manual_inference.v1",
                "valid_fraction": 1.0,
            },
        )


def _manual_capture_rows(
    manager: DepthStorageManager,
) -> tuple[tuple[RawDepthSnapshot, ...], int]:
    baseline, first_ts, second_ts, third_ts = _timestamps()
    source_identity = {
        "source_frame_contract": "noesis.mapanything.source_frame.v1",
        "source_id": 0,
        "source_frame_number": 7,
        "source_media_pts_ns": 123_000_000,
    }
    for timestamp_us in (first_ts, second_ts, third_ts):
        manager.store(
            "cam0",
            timestamp_us,
            *_arrays(),
            attrs=source_identity,
        ).wait(5.0)
    rows = DepthStorageCaptureEventAdapter(manager).list_raw_snapshots(
        "cam0",
        camera_id="cam0",
        after_timestamp_us=baseline,
        limit=16,
    )
    return tuple(rows), second_ts


def _manual_rgb_frame(rgb: np.ndarray, *, captured_at_us: int) -> TimestampedRgbFrame:
    height, width = rgb.shape[:2]
    return TimestampedRgbFrame(
        camera_id="cam0",
        source_id=0,
        batch_id=0,
        captured_at_us=captured_at_us,
        frame_id=7,
        source_media_pts_ns=123_000_000,
        width=int(width),
        height=int(height),
        content_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        pixels=rgb,
    )


def test_adapter_fuses_exact_committed_raw_cohort_with_portable_evidence(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    manager = _manager(tmp_path)
    try:
        baseline, first_ts, second_ts, third_ts = _timestamps()
        for index, timestamp_us in enumerate((first_ts, second_ts, third_ts)):
            manager.store("cam0", timestamp_us, *_arrays(index * 0.01)).wait(5.0)
        outcome = CaptureEventFusionCoordinator(
            DepthStorageCaptureEventAdapter(manager)
        ).fuse(
            CaptureEventFusionRequest(
                camera_id="cam0",
                storage_keys=("cam0",),
                baseline_timestamp_us={"cam0": baseline},
                require_rgb=False,
            )
        )

        assert outcome.evidence["rgb"] == {
            "status": "not_requested",
            "provider_configured": False,
        }
        assert outcome.fused_snapshot.timestamp_us > third_ts
        assert outcome.fused_snapshot.artifact_ref.startswith("depth-zarr:cam0/")
        assert outcome.fused_snapshot.sequence > 0
        assert len(outcome.fused_snapshot.manifest_sha256) == 64
        assert len(outcome.fused_snapshot.source_snapshot_ids) == 3
        quality = outcome.fused_snapshot.quality_evidence
        assert quality["support_valid_fraction"] == pytest.approx(1.0)
        assert quality["median_support"] == pytest.approx(3.0)
        assert (
            quality["support_evidence"]["contract"]
            == "noesis.depth.fusion.support.v1"
        )
        assert (
            outcome.evidence["fused_snapshot"]["quality_evidence"][
                "support_valid_fraction"
            ]
            == pytest.approx(1.0)
        )
        adapter = DepthStorageCaptureEventAdapter(manager)
        assert adapter.max_raw_timestamp("cam0") == third_ts
        assert adapter.flush_capture_frontier(timeout_s=1.0).clean is True
        assert (
            adapter.validate_fused_snapshot(outcome.fused_snapshot)
            is outcome.fused_snapshot
        )
        assert (
            manager.describe_snapshot(
                manager.latest_entry("cam0", outcome.fused_snapshot.timestamp_us)
            ).snapshot_role
            == "capture_event_fused"
        )
    finally:
        _close(manager)


def test_adapter_propagates_exact_mapanything_source_frame_identity(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    manager = _manager(tmp_path)
    try:
        _baseline, timestamp_us, _second, _third = _timestamps()
        receipt = manager.store(
            "cam0",
            timestamp_us,
            *_arrays(),
            attrs={
                "source_frame_contract": "noesis.mapanything.source_frame.v1",
                "source_id": 2,
                "source_frame_number": 321,
                "source_media_pts_ns": 456_000_000,
            },
        ).wait(5.0)
        descriptor = manager.describe_snapshot(receipt.path)
        assert (
            descriptor.source_id,
            descriptor.source_frame_number,
            descriptor.source_media_pts_ns,
        ) == (2, 321, 456_000_000)

        rows = DepthStorageCaptureEventAdapter(manager).list_raw_snapshots(
            "cam0",
            camera_id="cam0",
            after_timestamp_us=timestamp_us - 1,
            limit=2,
        )
        assert len(rows) == 1
        assert (
            rows[0].source_id,
            rows[0].source_frame_number,
            rows[0].source_media_pts_ns,
        ) == (2, 321, 456_000_000)
    finally:
        _close(manager)


def test_sparse_capture_event_is_retryable_and_does_not_publish_a_fused_artifact(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    manager = _manager(tmp_path)
    try:
        camera_id = "cam0"
        base = time.time_ns() // 1_000
        confidence = np.ones((10, 10), dtype=np.float32)
        mask = np.ones((10, 10), dtype=np.uint8)

        accepted_rows = []
        for offset in (1, 2):
            receipt = manager.store(
                camera_id,
                base + offset,
                np.full((10, 10), 2.0, dtype=np.float32),
                confidence,
                mask,
            ).wait(5.0)
            accepted_rows.append((receipt.ts_us, receipt.path))
        accepted_path, _meta = manager.fuse_snapshot_entries(
            camera_id,
            accepted_rows,
            event_id="accepted",
        )
        accepted = manager.describe_snapshot(accepted_path)

        checker = (np.indices((10, 10)).sum(axis=0) % 2).astype(bool)
        depths = (
            np.where(checker, 1.0, 3.0).astype(np.float32),
            np.where(checker, 3.0, 1.0).astype(np.float32),
            np.full((10, 10), 2.0, dtype=np.float32),
        )
        for offset, depth in enumerate(depths, start=3):
            manager.store(camera_id, base + offset, depth, confidence, mask).wait(
                5.0
            )

        adapter = DepthStorageCaptureEventAdapter(manager)
        with pytest.raises(CaptureEventFusionError) as caught:
            CaptureEventFusionCoordinator(adapter).fuse(
                CaptureEventFusionRequest(
                    camera_id=camera_id,
                    storage_keys=(camera_id,),
                    baseline_timestamp_us={camera_id: base + 2},
                    require_rgb=False,
                )
            )

        assert caught.value.code == "capture_event_fusion_quality_rejected"
        assert caught.value.details["observed"] == pytest.approx(0.0)
        assert caught.value.details["required"] == pytest.approx(0.40)
        assert (
            caught.value.details["metric"]
            == "consensus_full_frame_fraction"
        )
        support = caught.value.details["quality_evidence"]
        assert support["eligible_full_frame_fraction"] == pytest.approx(1.0)
        assert support["consensus_retained_eligible_fraction"] == pytest.approx(
            0.0
        )
        descriptors = [
            manager.describe_snapshot(path)
            for _timestamp_us, path in manager.list_snapshot_entries(
                camera_id,
                include_derived=True,
            )
        ]
        public_fused = [
            descriptor
            for descriptor in descriptors
            if descriptor.snapshot_role == "capture_event_fused"
        ]
        assert [descriptor.write_id for descriptor in public_fused] == [
            accepted.write_id
        ]
    finally:
        _close(manager)


def test_adapter_never_reuses_a_derived_snapshot_as_raw_input(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manager = _manager(tmp_path)
    try:
        baseline, first_ts, second_ts, _third_ts = _timestamps()
        first = manager.store("cam0", first_ts, *_arrays()).wait(5.0)
        second = manager.store("cam0", second_ts, *_arrays(0.01)).wait(5.0)
        manager.fuse_snapshot_entries(
            "cam0",
            ((first_ts, first.path), (second_ts, second.path)),
            event_id="first-fusion",
        )
        rows = DepthStorageCaptureEventAdapter(manager).list_raw_snapshots(
            "cam0",
            camera_id="cam0",
            after_timestamp_us=baseline,
            limit=16,
        )
        assert len(rows) == 2
        assert (
            DepthStorageCaptureEventAdapter(manager).max_raw_timestamp("cam0")
            == second_ts
        )
    finally:
        _close(manager)


def test_real_adapter_returns_n_plus_one_and_blocks_truncated_fusion(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manager = _manager(tmp_path)
    try:
        baseline, first_ts, second_ts, third_ts = _timestamps()
        for index, timestamp_us in enumerate((first_ts, second_ts, third_ts)):
            manager.store("cam0", timestamp_us, *_arrays(index * 0.01)).wait(5.0)

        with pytest.raises(CaptureEventFusionError) as caught:
            CaptureEventFusionCoordinator(
                DepthStorageCaptureEventAdapter(manager)
            ).fuse(
                CaptureEventFusionRequest(
                    camera_id="cam0",
                    storage_keys=("cam0",),
                    baseline_timestamp_us={"cam0": baseline},
                    require_rgb=False,
                    raw_limit=2,
                    min_observations=2,
                )
            )

        assert caught.value.code == "raw_snapshot_limit_exceeded"
        rows = manager.list_snapshot_entries("cam0", include_derived=True)
        assert [timestamp_us for timestamp_us, _path in rows] == [
            first_ts,
            second_ts,
            third_ts,
        ]
    finally:
        _close(manager)


def test_adapter_rejects_rgb_pixels_that_do_not_match_declared_digest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manager = _manager(tmp_path)
    try:
        baseline, first_ts, second_ts, _third_ts = _timestamps()
        manager.store("cam0", first_ts, *_arrays()).wait(5.0)
        manager.store("cam0", second_ts, *_arrays(0.01)).wait(5.0)
        adapter = DepthStorageCaptureEventAdapter(manager)
        rows = adapter.list_raw_snapshots(
            "cam0",
            camera_id="cam0",
            after_timestamp_us=baseline,
            limit=16,
        )
        rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        frame = TimestampedRgbFrame(
            camera_id="cam0",
            source_id=0,
            batch_id=0,
            captured_at_us=first_ts + 50_000,
            frame_id=1,
            source_media_pts_ns=123_000_000,
            width=4,
            height=3,
            content_sha256=hashlib.sha256(b"different").hexdigest(),
            pixels=rgb,
        )
        with pytest.raises(DepthStorageError, match="digest mismatch"):
            adapter.fuse_raw_snapshots(
                "cam0",
                rows,
                rgb_frame=frame,
                event_id="event",
                min_observations=2,
                depth_agreement_m=0.18,
            )
    finally:
        _close(manager)


def test_adapter_rejects_alias_relabeling_and_non_storage_objects(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manager = _manager(tmp_path)
    try:
        with pytest.raises(DepthStorageError, match="canonical camera"):
            DepthStorageCaptureEventAdapter(manager).list_raw_snapshots(
                "0",
                camera_id="living-room",
                after_timestamp_us=0,
                limit=16,
            )
    finally:
        _close(manager)

    class NotAStorage:
        pass

    with pytest.raises(TypeError, match="DepthStorageManager"):
        DepthStorageCaptureEventAdapter(NotAStorage())  # type: ignore[arg-type]


def test_canonical_manual_preprocess_preserves_rgb_and_repeats_exact_batch() -> None:
    rgb = np.empty((1080, 1920, 3), dtype=np.uint8)
    rgb[:, :, 0] = 11
    rgb[:, :, 1] = 22
    rgb[:, :, 2] = 33

    batch, meta = preprocess_mapanything_rgb(rgb)

    assert batch.shape == (3, 3, 294, 518)
    assert batch.dtype == np.dtype("<f4")
    assert batch.flags.c_contiguous
    assert meta == {
        "source_width": 1920,
        "source_height": 1080,
        "resized_width": 518,
        "resized_height": 291,
        "pad_left": 0,
        "pad_top": 1,
        "pad_right": 0,
        "pad_bottom": 2,
    }
    np.testing.assert_array_equal(batch[0], batch[1])
    np.testing.assert_array_equal(batch[0], batch[2])
    np.testing.assert_allclose(
        batch[0, :, 100, 100],
        np.asarray([11, 22, 33], dtype=np.float32) / 255.0,
    )
    assert np.count_nonzero(batch[:, :, 0, :]) == 0
    assert np.count_nonzero(batch[:, :, -1, :]) == 0


def test_adapter_publishes_deterministic_exact_rgb_result_as_fused_snapshot(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    manager = _manager(tmp_path)

    class FakeManualInferencer:
        def infer(self, rgb: np.ndarray) -> MapAnythingManualInferenceResult:
            height, width = rgb.shape[:2]
            depth = np.arange(height * width, dtype=np.float32).reshape(
                height,
                width,
            ) + 1.0
            confidence = np.full((height, width), 0.75, dtype=np.float32)
            mask = np.ones((height, width), dtype=np.uint8)
            return MapAnythingManualInferenceResult(
                depth=depth,
                confidence=confidence,
                mask=mask,
                evidence={
                    "contract": "test.manual_inference.v1",
                    "valid_fraction": 1.0,
                },
            )

    try:
        baseline, first_ts, second_ts, third_ts = _timestamps()
        source_identity = {
            "source_frame_contract": "noesis.mapanything.source_frame.v1",
            "source_id": 0,
            "source_frame_number": 7,
            "source_media_pts_ns": 123_000_000,
        }
        for timestamp_us in (first_ts, second_ts, third_ts):
            manager.store(
                "cam0",
                timestamp_us,
                *_arrays(),
                attrs=source_identity,
            ).wait(5.0)
        adapter = DepthStorageCaptureEventAdapter(
            manager,
            manual_inferencer=FakeManualInferencer(),
        )
        rows = adapter.list_raw_snapshots(
            "cam0",
            camera_id="cam0",
            after_timestamp_us=baseline,
            limit=16,
        )
        rgb = np.full((294, 518, 3), 40, dtype=np.uint8)
        rgb[180:, 360:, :] = 100
        frame = TimestampedRgbFrame(
            camera_id="cam0",
            source_id=0,
            batch_id=0,
            captured_at_us=second_ts,
            frame_id=7,
            source_media_pts_ns=123_000_000,
            width=518,
            height=294,
            content_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            pixels=rgb,
        )

        fused = adapter.fuse_raw_snapshots(
            "cam0",
            rows,
            rgb_frame=frame,
            event_id="deterministic-event",
            min_observations=3,
            depth_agreement_m=0.18,
        )

        assert fused.source_snapshot_ids == tuple(row.snapshot_id for row in rows)
        assert fused.quality_evidence["support_valid_fraction"] == 1.0
        assert (
            fused.quality_evidence["support_evidence"]["mode"]
            == "exact_rgb_deterministic"
        )
        descriptor = manager.describe_snapshot(
            manager.latest_entry("cam0", fused.timestamp_us)
        )
        assert descriptor.snapshot_role == "capture_event_fused"
        datasets = manager.load_datasets(descriptor.path)
        np.testing.assert_array_equal(
            datasets["depth"],
            np.arange(294 * 518, dtype=np.float32).reshape(294, 518) + 1.0,
        )
        np.testing.assert_array_equal(datasets["rgb"], rgb)
    finally:
        _close(manager)


def test_adapter_rejects_severely_dark_manual_rgb_before_inference(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    manager = _manager(tmp_path)
    inferencer = _RecordingManualInferencer()
    try:
        rows, captured_at_us = _manual_capture_rows(manager)
        rgb = np.full((294, 518, 3), 14, dtype=np.uint8)
        rgb[:, :64, :] = 0
        rgb[:45, :, :] = 255
        frame = _manual_rgb_frame(rgb, captured_at_us=captured_at_us)
        before = tuple(
            manager.list_snapshot_entries("cam0", include_derived=True)
        )
        adapter = DepthStorageCaptureEventAdapter(
            manager,
            manual_inferencer=inferencer,
        )

        with pytest.raises(DepthFusionQualityError) as caught:
            adapter.fuse_raw_snapshots(
                "cam0",
                rows,
                rgb_frame=frame,
                event_id="severe-dark-event",
                min_observations=3,
                depth_agreement_m=0.18,
            )

        assert inferencer.calls == 0
        assert caught.value.metric == "manual_rgb_scene_visibility_ratio"
        assert caught.value.observed < caught.value.required
        evidence = caught.value.evidence
        assert evidence["p90"] < 25.0
        assert evidence["p99_p50_range"] < 20.0
        assert evidence["excluded_top_rows"] == 45
        assert evidence["gate"]["reason"] == "severe_dark_scene"
        assert evidence["gate"]["passed"] is False
        assert tuple(
            manager.list_snapshot_entries("cam0", include_derived=True)
        ) == before
    finally:
        _close(manager)


def test_adapter_rejects_severely_dark_rgb_before_temporal_fusion(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    manager = _manager(tmp_path)
    try:
        rows, captured_at_us = _manual_capture_rows(manager)
        rgb = np.full((294, 518, 3), 14, dtype=np.uint8)
        rgb[:, :64, :] = 0
        rgb[:45, :, :] = 255
        frame = _manual_rgb_frame(rgb, captured_at_us=captured_at_us)
        before = tuple(
            manager.list_snapshot_entries("cam0", include_derived=True)
        )
        adapter = DepthStorageCaptureEventAdapter(manager)

        with pytest.raises(DepthFusionQualityError) as caught:
            adapter.fuse_raw_snapshots(
                "cam0",
                rows,
                rgb_frame=frame,
                event_id="severe-dark-fusion-event",
                min_observations=3,
                depth_agreement_m=0.18,
            )

        assert caught.value.metric == "manual_rgb_scene_visibility_ratio"
        assert caught.value.evidence["gate"]["reason"] == "severe_dark_scene"
        assert tuple(
            manager.list_snapshot_entries("cam0", include_derived=True)
        ) == before
    finally:
        _close(manager)


@pytest.mark.parametrize(
    ("scene_name", "base_luminance", "highlight_luminance"),
    (
        pytest.param("kitchen", 40, 90, id="kitchen"),
        pytest.param("family-room", 50, 180, id="family-room"),
    ),
)
def test_adapter_accepts_visible_manual_rgb_scenes(
    tmp_path,
    scene_name: str,
    base_luminance: int,
    highlight_luminance: int,
) -> None:  # type: ignore[no-untyped-def]
    manager = _manager(tmp_path)
    inferencer = _RecordingManualInferencer()
    try:
        rows, captured_at_us = _manual_capture_rows(manager)
        rgb = np.full((294, 518, 3), base_luminance, dtype=np.uint8)
        rgb[:, :64, :] = 0
        rgb[:45, :, :] = 255
        rgb[150:, 300:, :] = highlight_luminance
        adapter = DepthStorageCaptureEventAdapter(
            manager,
            manual_inferencer=inferencer,
        )

        fused = adapter.fuse_raw_snapshots(
            "cam0",
            rows,
            rgb_frame=_manual_rgb_frame(rgb, captured_at_us=captured_at_us),
            event_id=f"{scene_name}-visible-event",
            min_observations=3,
            depth_agreement_m=0.18,
        )

        assert inferencer.calls == 1
        assert fused.camera_id == "cam0"
        assert fused.snapshot_role == "capture_event_fused"
        descriptor = manager.describe_snapshot(
            manager.latest_entry("cam0", fused.timestamp_us)
        )
        assert descriptor.write_id == fused.snapshot_id
        assert descriptor.snapshot_role == "capture_event_fused"
    finally:
        _close(manager)
