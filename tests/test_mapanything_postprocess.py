from __future__ import annotations

import base64
import time
from pathlib import Path

import numpy as np
import pytest

from geometry.depth_source import (
    DepthFusionQualityError,
    DepthStorageError,
    DepthStorageManager,
    _binary_mask_component_evidence,
)
from noesis.pipelines import deepstream_pipeline as pipeline
from noesis.pipelines import hooks


@pytest.fixture(autouse=True)
def _reset_pipeline_singleton(monkeypatch: pytest.MonkeyPatch):
    pipeline._PIPELINE_SINGLETON = None  # type: ignore[attr-defined]
    monkeypatch.setenv("NOESIS_DS9_STUB_PIPELINE", "1")
    monkeypatch.setattr(
        pipeline,
        "materialize_nvinfer_engine_only_config",
        lambda **kwargs: Path(kwargs["source_config"]),
    )
    monkeypatch.setattr(
        pipeline,
        "materialize_nvtracker_engine_only_config",
        lambda **kwargs: Path(kwargs["source_config"]),
    )
    yield
    pipeline._PIPELINE_SINGLETON = None  # type: ignore[attr-defined]


def _build_pipeline() -> pipeline.DeepStreamPipeline:
    config_path = Path("DS9/config/infer.yaml")
    return pipeline.build_pipeline(config_path)


def test_mapanything_processor_emits_depth_result(tmp_path: Path):
    graph = _build_pipeline()
    graph.mark_depth_enabled(True)

    storage = DepthStorageManager(
        base_path=tmp_path,
        max_snapshots_per_camera=10,
        retention_minutes=5,
        enable_async=False,
        max_queue_size=1,
        worker_count=1,
        max_worker_count=1,
    )

    published: list = []

    class _StubDepthPublisher:
        def publish(self, result):
            published.append(result)

    hooks.attach_mapanything_postprocess_hook(
        graph,
        storage=storage,
        depth_pub=_StubDepthPublisher(),
    )

    component = graph.components["mapanything_fullframe"]
    processor = component.config["_mapanything_processor"]
    assert graph.mapanything_processor is processor

    width, height = 64, 48
    depth = np.linspace(0.5, 4.5, num=width * height, dtype=np.float32).reshape(
        height, width
    )
    confidence = np.ones_like(depth, dtype=np.float32)
    mask = np.ones_like(depth, dtype=bool)
    pts_ns = int(time.time() * 1_000_000_000)

    result = processor.handle_numpy_arrays(  # type: ignore[attr-defined]
        source_id=1,
        frame_id=42,
        pts_ns=pts_ns,
        tensors={
            "depth": depth,
            "confidence": confidence,
            "mask": mask,
        },
    )

    storage.flush()
    storage.shutdown(wait=True)

    assert result is not None
    assert published and published[0] == result
    assert graph.depth_frame_samples, "Depth FPS tracking did not record the burst"
    expected_w, expected_h = graph.frame_size
    assert result.width == expected_w
    assert result.height == expected_h
    assert result.source_id == 1
    assert result.frame_id == 42
    assert result.minmax[0] >= float(depth.min())
    assert result.minmax[1] <= float(depth.max())

    zarr_path = Path(result.depth_map_ref)
    assert zarr_path.exists()

    datasets = storage.load_datasets(zarr_path)
    assert datasets is not None
    stored_depth = datasets["depth"]
    stored_conf = datasets["conf"]
    stored_mask = datasets["mask"]

    assert stored_depth.shape == (expected_h, expected_w)
    assert stored_conf.shape == (expected_h, expected_w)
    assert stored_mask.shape == (expected_h, expected_w)
    valid_mask = stored_mask.astype(bool)
    assert np.any(valid_mask)
    assert np.all(valid_mask)
    assert np.isfinite(stored_depth).all()
    assert np.all(stored_conf > 0.0)
    assert np.any(stored_conf < 1.0)
    valid = stored_depth[valid_mask]
    assert valid.size > 0
    assert result.minmax[0] == pytest.approx(float(np.nanmin(valid)))
    assert result.minmax[1] == pytest.approx(float(np.nanmax(valid)))


def test_depth_storage_persists_rgb_only_as_part_of_atomic_snapshot(tmp_path: Path):
    storage = DepthStorageManager(
        base_path=tmp_path,
        max_snapshots_per_camera=10,
        retention_minutes=5,
        enable_async=False,
        enforce_async=False,
        zarr_clevel=0,
        zarr_chunk_px=0,
    )

    camera_id = "living-room"
    ts_us = int(time.time() * 1_000_000)
    depth = np.ones((4, 6), dtype=np.float32)
    confidence = np.ones_like(depth, dtype=np.float32)
    mask = np.ones_like(depth, dtype=np.uint8)
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[:, :, 0] = 10
    rgb[:, :, 1] = 20
    rgb[:, :, 2] = 30

    zarr_path = storage.store(camera_id, ts_us, depth, confidence, mask, rgb=rgb)
    datasets = storage.load_datasets(zarr_path)
    assert datasets is not None
    assert np.array_equal(datasets["rgb"], rgb)

    payload = storage.load_latest_depth(camera_id)
    assert payload is not None
    assert payload["rgb_shape"] == [4, 6, 3]
    decoded = np.frombuffer(
        base64.b64decode(payload["rgb_b64"]), dtype=np.uint8
    ).reshape((4, 6, 3))
    assert np.array_equal(decoded, rgb)

    replacement = np.full((4, 6, 3), 220, dtype=np.uint8)
    with pytest.raises(DepthStorageError, match="immutable"):
        storage.attach_rgb_to_snapshot(camera_id, ts_us, replacement)

    payload = storage.load_latest_depth(camera_id)
    assert payload is not None
    decoded = np.frombuffer(
        base64.b64decode(payload["rgb_b64"]), dtype=np.uint8
    ).reshape((4, 6, 3))
    assert np.array_equal(decoded, rgb)


def test_depth_storage_fuses_raw_snapshots_into_capture_event(tmp_path: Path):
    storage = DepthStorageManager(
        base_path=tmp_path,
        max_snapshots_per_camera=10,
        retention_minutes=5,
        enable_async=False,
        enforce_async=False,
        zarr_clevel=0,
        zarr_chunk_px=0,
    )

    camera_id = "living-room"
    base_ts = int(time.time() * 1_000_000)
    depth_a = np.array([[1.0, 2.0], [3.0, 20.0]], dtype=np.float32)
    depth_b = np.array([[1.1, 2.1], [3.1, 4.0]], dtype=np.float32)
    depth_c = np.array([[0.9, 1.9], [2.9, 4.1]], dtype=np.float32)
    confidence = np.ones((2, 2), dtype=np.float32)
    mask = np.ones((2, 2), dtype=np.uint8)
    rgb = np.full((2, 2, 3), 80, dtype=np.uint8)

    entries = []
    for offset, depth in enumerate((depth_a, depth_b, depth_c)):
        ts_us = base_ts + offset
        path = storage.store(camera_id, ts_us, depth, confidence, mask)
        entries.append((ts_us, path))

    fused_path, meta = storage.fuse_snapshot_entries(
        camera_id,
        entries,
        rgb=rgb,
        min_confidence=0.1,
        min_observations=2,
        depth_agreement_m=0.25,
        snapshot_role="capture_event_fused",
        fusion_level="intra_capture",
    )

    datasets = storage.load_datasets(fused_path)
    assert datasets is not None
    fused_depth = datasets["depth"]
    fused_mask = datasets["mask"].astype(bool)
    assert fused_depth.shape == (2, 2)
    assert int(np.count_nonzero(fused_mask)) == 4
    assert fused_depth[0, 0] == pytest.approx(1.0, abs=0.11)
    assert fused_depth[1, 1] == pytest.approx(4.1, abs=0.11)
    assert np.array_equal(datasets["rgb"], rgb)
    assert meta["source_snapshot_count"] == 3
    assert meta["min_observations"] == 3
    assert meta["support_evidence"]["required_observations"] == 3
    assert meta["support_evidence"]["consensus_full_frame_fraction"] == pytest.approx(
        0.75
    )
    assert meta["support_evidence"]["continuity_full_frame_fraction"] == 0.0
    assert meta["output_valid_fraction"] == pytest.approx(1.0)
    assert meta["fusion_output"]["contract"] == "noesis.depth.fusion.output.v3"
    assert (
        meta["fusion_output"]["algorithm"]
        == "coherent_pairwise_medoid_reference_surface"
    )
    assert meta["fusion_output"]["continuity_pixels"] == 0
    assert meta["support_quality_gate"] == {
        "contract": "noesis.depth.fusion.quality_gate.v1",
        "metric": "consensus_full_frame_fraction",
        "observed": pytest.approx(0.75),
        "required": pytest.approx(0.40),
        "passed": True,
    }
    assert meta["snapshot_role"] == "capture_event_fused"
    assert meta["fusion_level"] == "intra_capture"

    entries_without_derived = storage.list_snapshot_entries(
        camera_id, include_derived=False
    )
    assert len(entries_without_derived) == 3


def test_capture_event_fusion_normalizes_coherent_monocular_scale_drift(
    tmp_path: Path,
) -> None:
    storage = DepthStorageManager(
        base_path=tmp_path,
        max_snapshots_per_camera=10,
        retention_minutes=5,
        enable_async=False,
        enforce_async=False,
        zarr_clevel=0,
        zarr_chunk_px=0,
    )
    try:
        camera_id = "living-room"
        base_ts = int(time.time() * 1_000_000)
        geometry = np.array(
            [[1.0, 2.0, 3.0], [1.5, 2.5, 4.0]],
            dtype=np.float32,
        )
        confidence = np.ones_like(geometry, dtype=np.float32)
        mask = np.ones_like(geometry, dtype=np.uint8)
        entries = []
        for offset, scale in enumerate((2.0, 0.5, 1.0)):
            receipt = storage.store(
                camera_id,
                base_ts + offset,
                geometry * scale,
                confidence,
                mask,
            ).wait(timeout=5.0)
            entries.append((receipt.ts_us, receipt.path))

        with pytest.raises(DepthFusionQualityError) as unnormalized:
            storage.fuse_snapshot_entries(
                camera_id,
                entries,
                min_confidence=0.1,
                min_observations=2,
                depth_agreement_m=0.18,
                normalize_frame_scale=False,
                snapshot_role="capture_event_fused",
                fusion_level="intra_capture",
            )
        assert unnormalized.value.observed == pytest.approx(0.0)

        fused_path, meta = storage.fuse_snapshot_entries(
            camera_id,
            entries,
            min_confidence=0.1,
            min_observations=2,
            depth_agreement_m=0.18,
            snapshot_role="capture_event_fused",
            fusion_level="intra_capture",
        )

        datasets = storage.load_datasets(fused_path)
        assert datasets is not None
        assert np.array_equal(datasets["mask"], mask)
        assert np.allclose(datasets["depth"], geometry, atol=1e-5)
        normalization = meta["frame_scale_normalization"]
        assert (
            normalization["contract"]
            == "noesis.depth.fusion.frame_scale_normalization.v1"
        )
        assert normalization["enabled"] is True
        assert (
            normalization["algorithm"]
            == "depth_times_cohort_median_over_frame_median"
        )
        assert normalization["factors"] == pytest.approx([0.5, 2.0, 1.0])
        assert meta["support_valid_fraction"] == pytest.approx(1.0)
    finally:
        storage.shutdown(wait=True, timeout=5.0)


def test_fusion_support_ratio_scales_with_the_admitted_cohort(
    tmp_path: Path,
) -> None:
    storage = DepthStorageManager(
        base_path=tmp_path,
        max_snapshots_per_camera=0,
        retention_minutes=0.0,
        enable_async=False,
        enforce_async=False,
        zarr_clevel=0,
        zarr_chunk_px=0,
    )
    try:
        confidence = np.ones((1, 1), dtype=np.float32)

        def fuse_mask_for(
            valid_count: int,
            cohort_size: int,
            *,
            min_observation_ratio: float = 0.5,
        ):
            snapshots = []
            for index in range(cohort_size):
                snapshots.append(
                    (
                        index + 1,
                        tmp_path / "synthetic" / f"{index}.zarr",
                        {
                            "depth": np.full((1, 1), 2.0, dtype=np.float32),
                            "conf": confidence,
                            "mask": np.full(
                                (1, 1),
                                1 if index < valid_count else 0,
                                dtype=np.uint8,
                            ),
                        },
                    )
                )
            _depth, _conf, mask, _rgb, meta = storage._fuse_depth_datasets(
                snapshots,
                min_confidence=0.1,
                min_observations=2,
                min_observation_ratio=min_observation_ratio,
                depth_agreement_m=0.18,
                normalize_frame_scale=False,
            )
            return mask, meta

        mask_3_of_6, meta_3_of_6 = fuse_mask_for(3, 6)
        assert bool(mask_3_of_6[0, 0]) is True
        assert meta_3_of_6["min_observations"] == 4
        assert (
            meta_3_of_6["support_evidence"][
                "strict_majority_observations"
            ]
            == 4
        )
        assert (
            meta_3_of_6["support_evidence"]["tie_policy"]
            == "reject_exact_half_support"
        )
        assert (
            meta_3_of_6["support_evidence"]["consensus_pixels"]
            == 0
        )
        assert meta_3_of_6["support_evidence"]["continuity_pixels"] == 0
        assert (
            meta_3_of_6["support_evidence"]["support_count_histogram"]["3"]
            == 1
        )

        mask_4_of_6, meta_4_of_6 = fuse_mask_for(4, 6)
        assert bool(mask_4_of_6[0, 0]) is True
        assert meta_4_of_6["min_observations"] == 4

        mask_3_of_8, meta_3_of_8 = fuse_mask_for(3, 8)
        assert bool(mask_3_of_8[0, 0]) is True
        assert meta_3_of_8["min_observations"] == 5
        assert (
            meta_3_of_8["support_evidence"][
                "consensus_retained_eligible_fraction"
            ]
            == 0.0
        )

        mask_4_of_8, meta_4_of_8 = fuse_mask_for(4, 8)
        assert bool(mask_4_of_8[0, 0]) is True
        assert meta_4_of_8["min_observations"] == 5

        mask_5_of_8, meta_5_of_8 = fuse_mask_for(5, 8)
        assert bool(mask_5_of_8[0, 0]) is True
        assert meta_5_of_8["min_observations"] == 5
        assert (
            meta_5_of_8["support_evidence"]["component_evidence"][
                "largest_component_fraction"
            ]
            == 1.0
        )

        mask_5_of_8_high_ratio, meta_high_ratio = fuse_mask_for(
            5,
            8,
            min_observation_ratio=0.75,
        )
        assert bool(mask_5_of_8_high_ratio[0, 0]) is True
        assert meta_high_ratio["min_observations"] == 6
        assert meta_high_ratio["support_evidence"]["consensus_pixels"] == 0
        assert meta_high_ratio["support_evidence"]["continuity_pixels"] == 0
        assert (
            meta_high_ratio["support_evidence"][
                "ratio_required_observations"
            ]
            == 6
        )
    finally:
        storage.shutdown(wait=True, timeout=5.0)


def test_fusion_component_evidence_separates_primary_support_and_fragments() -> None:
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:4, 1:4] = True
    mask[5, 5] = True

    evidence = _binary_mask_component_evidence(mask)

    assert evidence["component_count"] == 2
    assert evidence["largest_component_pixels"] == 9
    assert evidence["fragment_pixels"] == 1
    assert evidence["largest_component_fraction"] == pytest.approx(0.9)


def test_fusion_outputs_one_coherent_pairwise_medoid_surface(
    tmp_path: Path,
) -> None:
    storage = DepthStorageManager(
        base_path=tmp_path,
        max_snapshots_per_camera=0,
        retention_minutes=0.0,
        enable_async=False,
        enforce_async=False,
        zarr_clevel=0,
        zarr_chunk_px=0,
    )
    try:
        frames = (
            np.full((2, 3), 1.0, dtype=np.float32),
            np.full((2, 3), 1.1, dtype=np.float32),
            np.full((2, 3), 9.0, dtype=np.float32),
        )
        snapshots = [
            (
                index + 1,
                tmp_path / "synthetic" / f"{index}.zarr",
                {
                    "depth": frame,
                    "conf": np.ones_like(frame, dtype=np.float32),
                    "mask": np.ones_like(frame, dtype=np.uint8),
                },
            )
            for index, frame in enumerate(frames)
        ]

        depth, _conf, mask, _rgb, meta = storage._fuse_depth_datasets(
            snapshots,
            min_confidence=0.1,
            min_observations=2,
            min_observation_ratio=0.5,
            depth_agreement_m=0.18,
            normalize_frame_scale=False,
        )

        assert np.array_equal(mask, np.ones((2, 3), dtype=np.uint8))
        assert np.array_equal(depth, frames[1])
        output = meta["fusion_output"]
        assert output["selection"] == "pairwise_residual_medoid"
        assert output["reference_frame_index"] == 1
        assert output["depth_source_policy"] == "selected_reference_frame_only"
        assert output["continuity_pixels"] == 0
    finally:
        storage.shutdown(wait=True, timeout=5.0)


def test_fusion_medoid_rejects_tiny_perfect_overlap_frame(
    tmp_path: Path,
) -> None:
    storage = DepthStorageManager(
        base_path=tmp_path,
        max_snapshots_per_camera=0,
        retention_minutes=0.0,
        enable_async=False,
        enforce_async=False,
        zarr_clevel=0,
        zarr_chunk_px=0,
    )
    try:
        region = np.zeros((10, 10), dtype=bool)
        region[:5, :] = True
        isolated = np.zeros((10, 10), dtype=bool)
        isolated[9, 9] = True
        snapshots = []
        for index, depth_value in enumerate(
            (1.00, 1.04, 1.08, 1.12, 1.16)
        ):
            mask = region.copy()
            if index == 0:
                mask |= isolated
            depth = np.full((10, 10), depth_value, dtype=np.float32)
            depth[9, 9] = 7.0
            snapshots.append(
                (
                    index + 1,
                    tmp_path / "synthetic" / f"{index}.zarr",
                    {
                        "depth": depth,
                        "conf": mask.astype(np.float32),
                        "mask": mask.astype(np.uint8),
                    },
                )
            )
        sparse_depth = np.full((10, 10), 7.0, dtype=np.float32)
        snapshots.append(
            (
                6,
                tmp_path / "synthetic" / "5.zarr",
                {
                    "depth": sparse_depth,
                    "conf": isolated.astype(np.float32),
                    "mask": isolated.astype(np.uint8),
                },
            )
        )

        _depth, _conf, mask, _rgb, meta = storage._fuse_depth_datasets(
            snapshots,
            min_confidence=0.1,
            min_observations=2,
            min_observation_ratio=0.5,
            depth_agreement_m=0.18,
            normalize_frame_scale=False,
        )

        output = meta["fusion_output"]
        assert output["reference_frame_index"] != 5
        assert output["reference_coverage_eligible"][5] is False
        assert output["frame_valid_pixels"][5] == 1
        assert int(np.count_nonzero(mask)) >= 50
        assert meta["output_valid_fraction"] >= 0.50
    finally:
        storage.shutdown(wait=True, timeout=5.0)


def test_fusion_medoid_tie_uses_newest_timestamp_not_input_order(
    tmp_path: Path,
) -> None:
    storage = DepthStorageManager(
        base_path=tmp_path,
        max_snapshots_per_camera=0,
        retention_minutes=0.0,
        enable_async=False,
        enforce_async=False,
        zarr_clevel=0,
        zarr_chunk_px=0,
    )
    try:
        frame = np.ones((3, 3), dtype=np.float32)
        snapshots = [
            (
                200,
                tmp_path / "synthetic" / "newer.zarr",
                {
                    "depth": frame,
                    "conf": frame,
                    "mask": np.ones_like(frame, dtype=np.uint8),
                },
            ),
            (
                100,
                tmp_path / "synthetic" / "older.zarr",
                {
                    "depth": frame,
                    "conf": frame,
                    "mask": np.ones_like(frame, dtype=np.uint8),
                },
            ),
        ]

        _depth, _conf, _mask, _rgb, meta = storage._fuse_depth_datasets(
            snapshots,
            min_confidence=0.1,
            min_observations=2,
            min_observation_ratio=0.5,
            depth_agreement_m=0.18,
            normalize_frame_scale=False,
        )

        output = meta["fusion_output"]
        assert output["reference_frame_index"] == 0
        assert output["reference_timestamp_us"] == 200
    finally:
        storage.shutdown(wait=True, timeout=5.0)


def test_fusion_normalizes_and_caps_per_frame_confidence_tails(
    tmp_path: Path,
) -> None:
    storage = DepthStorageManager(
        base_path=tmp_path,
        max_snapshots_per_camera=0,
        retention_minutes=0.0,
        enable_async=False,
        enforce_async=False,
        zarr_clevel=0,
        zarr_chunk_px=0,
    )
    try:
        snapshots = []
        for index, (depth_value, confidence_value) in enumerate(
            ((1.0, 1_000.0), (1.2, 1.0), (1.2, 1.0))
        ):
            snapshots.append(
                (
                    index + 1,
                    tmp_path / "synthetic" / f"{index}.zarr",
                    {
                        "depth": np.array([[depth_value]], dtype=np.float32),
                        "conf": np.array(
                            [[confidence_value]],
                            dtype=np.float32,
                        ),
                        "mask": np.ones((1, 1), dtype=np.uint8),
                    },
                )
            )

        depth, _conf, mask, _rgb, meta = storage._fuse_depth_datasets(
            snapshots,
            min_confidence=0.1,
            min_observations=2,
            min_observation_ratio=0.5,
            depth_agreement_m=0.18,
            normalize_frame_scale=False,
        )

        assert bool(mask[0, 0]) is True
        assert depth[0, 0] == pytest.approx(1.2)
        weighting = meta["confidence_weighting"]
        assert (
            weighting["algorithm"]
            == "per_frame_percentile_cap_diagnostic_only"
        )
        assert weighting["frame_caps"] == pytest.approx([1_000.0, 1.0, 1.0])
        assert weighting["weight_max"] == pytest.approx(1.0)
    finally:
        storage.shutdown(wait=True, timeout=5.0)


def test_fusion_quarantines_unreasonable_frame_scale_instead_of_clipping(
    tmp_path: Path,
) -> None:
    storage = DepthStorageManager(
        base_path=tmp_path,
        max_snapshots_per_camera=0,
        retention_minutes=0.0,
        enable_async=False,
        enforce_async=False,
        zarr_clevel=0,
        zarr_chunk_px=0,
    )
    try:
        snapshots = []
        for index, depth_value in enumerate((0.1, 1.0, 1.0, 1.0)):
            snapshots.append(
                (
                    index + 1,
                    tmp_path / "synthetic" / f"{index}.zarr",
                    {
                        "depth": np.array([[depth_value]], dtype=np.float32),
                        "conf": np.ones((1, 1), dtype=np.float32),
                        "mask": np.ones((1, 1), dtype=np.uint8),
                    },
                )
            )

        depth, _conf, mask, _rgb, meta = storage._fuse_depth_datasets(
            snapshots,
            min_confidence=0.05,
            min_observations=2,
            min_observation_ratio=0.5,
            depth_agreement_m=0.18,
            normalize_frame_scale=True,
        )

        assert bool(mask[0, 0]) is True
        assert depth[0, 0] == pytest.approx(1.0)
        normalization = meta["frame_scale_normalization"]
        assert normalization["proposed_factors"][0] == pytest.approx(10.0)
        assert normalization["factors"][0] == pytest.approx(1.0)
        assert normalization["applied"][0] is False
        assert normalization["rejected"] == [True, False, False, False]
        assert normalization["rejection_policy"] == "quarantine_entire_frame"
        support = meta["support_evidence"]
        assert support["quarantined_frame_count"] == 1
        assert support["quarantined_frame_indices"] == [0]
        assert support["effective_cohort_size"] == 3
        assert support["minimum_effective_cohort_size"] == 3
        assert support["effective_cohort_sufficient"] is True
        assert support["required_observations_basis"] == "non_quarantined_frames"
    finally:
        storage.shutdown(wait=True, timeout=5.0)


def test_fusion_majority_is_computed_from_non_quarantined_frames(
    tmp_path: Path,
) -> None:
    storage = DepthStorageManager(
        base_path=tmp_path,
        max_snapshots_per_camera=0,
        retention_minutes=0.0,
        enable_async=False,
        enforce_async=False,
        zarr_clevel=0,
        zarr_chunk_px=0,
    )
    try:
        snapshots = []
        for index, depth_value in enumerate((0.1, 0.2, 1.0, 1.0, 1.0, 1.0)):
            snapshots.append(
                (
                    index + 1,
                    tmp_path / "synthetic" / f"{index}.zarr",
                    {
                        "depth": np.full((2, 2), depth_value, dtype=np.float32),
                        "conf": np.ones((2, 2), dtype=np.float32),
                        "mask": np.ones((2, 2), dtype=np.uint8),
                    },
                )
            )

        _depth, _conf, mask, _rgb, meta = storage._fuse_depth_datasets(
            snapshots,
            min_confidence=0.05,
            min_observations=3,
            min_observation_ratio=0.5,
            depth_agreement_m=0.18,
            normalize_frame_scale=True,
        )

        support = meta["support_evidence"]
        assert support["cohort_size"] == 6
        assert support["quarantined_frame_indices"] == [0, 1]
        assert support["effective_cohort_size"] == 4
        assert support["minimum_effective_cohort_size"] == 3
        assert support["effective_cohort_sufficient"] is True
        assert support["required_observations"] == 3
        assert np.all(mask)
    finally:
        storage.shutdown(wait=True, timeout=5.0)


def test_fusion_rejects_when_quarantine_leaves_too_few_frames(
    tmp_path: Path,
) -> None:
    storage = DepthStorageManager(
        base_path=tmp_path,
        max_snapshots_per_camera=0,
        retention_minutes=0.0,
        enable_async=False,
        enforce_async=False,
        zarr_clevel=0,
        zarr_chunk_px=0,
    )
    try:
        entries = []
        for index, depth_value in enumerate((0.1, 1.0, 1.0)):
            ts_us = 10_000 + index
            path = storage.store(
                "living-room",
                ts_us,
                np.full((2, 2), depth_value, dtype=np.float32),
                np.ones((2, 2), dtype=np.float32),
                np.ones((2, 2), dtype=np.uint8),
            )
            entries.append((ts_us, path))

        with pytest.raises(DepthFusionQualityError) as caught:
            storage.fuse_snapshot_entries(
                "living-room",
                entries,
                min_confidence=0.05,
                min_observations=3,
                depth_agreement_m=0.18,
                snapshot_role="capture_event_fused",
                fusion_level="intra_capture",
            )

        assert caught.value.metric == "effective_cohort_size"
        assert caught.value.observed == pytest.approx(2.0)
        assert caught.value.required == pytest.approx(3.0)
    finally:
        storage.shutdown(wait=True, timeout=5.0)
