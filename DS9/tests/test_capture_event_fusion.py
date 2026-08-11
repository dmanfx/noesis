from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from noesis_core.capture_event_fusion import (
    CaptureEventFusionCoordinator,
    CaptureEventFusionError,
    CaptureEventFusionRequest,
    FusedDepthSnapshot,
    RawDepthSnapshot,
    TimestampedRgbFrame,
)

from DS9.noesis.capture_event_rgb_provider import PipelineRgbFrameProvider

REPO_ROOT = Path(__file__).resolve().parents[2]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _raw(
    index: int,
    *,
    camera: str = "cam0",
    key: str = "cam0",
    source_id: int | None = 0,
) -> RawDepthSnapshot:
    timestamp_us = 1_000_000 + index * 100_000
    exact = source_id is not None
    return RawDepthSnapshot(
        camera_id=camera,
        storage_key=key,
        timestamp_us=timestamp_us,
        snapshot_id=f"raw-{index}",
        artifact_ref=f"depth-artifact:{key}:{timestamp_us}",
        content_sha256=_digest(f"raw-{index}"),
        sequence=index + 1,
        manifest_sha256=_digest(f"manifest-raw-{index}"),
        source_id=source_id,
        source_frame_number=40 + index if exact else None,
        source_media_pts_ns=10_000_000 + index * 1_000_000 if exact else None,
    )


def _rgb(
    captured_at_us: int = 1_150_000,
    *,
    camera: str = "cam0",
    source_id: int = 0,
    frame_id: int = 42,
    source_media_pts_ns: int = 12_000_000,
    pixels: np.ndarray | None = None,
    width: int = 4,
    height: int = 3,
    content_sha256: str | None = None,
) -> TimestampedRgbFrame:
    if pixels is None:
        pixels = np.full(
            (height, width, 3),
            fill_value=captured_at_us % 251,
            dtype=np.uint8,
        )
    digest = hashlib.sha256(pixels.tobytes(order="C")).hexdigest()
    return TimestampedRgbFrame(
        camera_id=camera,
        source_id=source_id,
        batch_id=0,
        captured_at_us=captured_at_us,
        frame_id=frame_id,
        source_media_pts_ns=source_media_pts_ns,
        width=width,
        height=height,
        content_sha256=content_sha256 or digest,
        pixels=pixels,
    )


class _Store:
    def __init__(self, rows: Sequence[object]) -> None:
        self.rows = tuple(rows)
        self.list_calls = 0
        self.fuse_calls = 0
        self.last_event_id: str | None = None

    def list_raw_snapshots(
        self,
        storage_key: str,
        *,
        camera_id: str,
        after_timestamp_us: int,
        limit: int,
    ) -> Sequence[object]:
        self.list_calls += 1
        return self.rows

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
        self.fuse_calls += 1
        self.last_event_id = event_id
        return FusedDepthSnapshot(
            camera_id="cam0",
            storage_key=storage_key,
            timestamp_us=snapshots[-1].timestamp_us + 1,
            snapshot_id="fused-1",
            artifact_ref="depth-artifact:cam0:fused-1",
            content_sha256=_digest("fused-1"),
            sequence=max(row.sequence for row in snapshots) + 1,
            manifest_sha256=_digest("manifest-fused-1"),
            event_id=event_id,
            source_snapshot_ids=tuple(row.snapshot_id for row in snapshots),
        )


class _RgbProvider:
    def __init__(self, frame: TimestampedRgbFrame | None) -> None:
        self.frame = frame
        self.calls = 0
        self.cohort: tuple[RawDepthSnapshot, ...] = ()

    def provide(
        self,
        camera_id: str,
        *,
        cohort: Sequence[RawDepthSnapshot],
    ) -> TimestampedRgbFrame | None:
        self.calls += 1
        self.cohort = tuple(cohort)
        return self.frame


def _request(**changes: object) -> CaptureEventFusionRequest:
    values: dict[str, object] = {
        "camera_id": "cam0",
        "storage_keys": ("cam0",),
        "baseline_timestamp_us": {"cam0": 900_000},
    }
    values.update(changes)
    return CaptureEventFusionRequest(**values)  # type: ignore[arg-type]


def test_coordinator_binds_rgb_to_one_exact_admitted_source_frame() -> None:
    rows = (_raw(1), _raw(2), _raw(3))
    store = _Store(rows)
    provider = _RgbProvider(_rgb())
    outcome = CaptureEventFusionCoordinator(store, rgb_provider=provider).fuse(
        _request()
    )

    assert provider.cohort == rows
    assert store.fuse_calls == 1
    assert outcome.evidence["rgb"] == {
        "status": "available",
        "provider_configured": True,
        "source_id": 0,
        "batch_id": 0,
        "captured_at_us": 1_150_000,
        "frame_id": 42,
        "source_media_pts_ns": 12_000_000,
        "width": 4,
        "height": 3,
        "color_space": "rgb8",
        "content_sha256": _rgb().content_sha256,
    }
    exact = outcome.evidence["raw_snapshots"][1]
    assert (
        exact["source_id"],
        exact["source_frame_number"],
        exact["source_media_pts_ns"],
    ) == (0, 42, 12_000_000)


@pytest.mark.parametrize(
    ("frame", "code"),
    [
        (_rgb(camera="other"), "rgb_camera_mismatch"),
        (_rgb(source_id=1), "rgb_frame_identity_mismatch"),
        (_rgb(frame_id=999), "rgb_frame_identity_mismatch"),
        (_rgb(source_media_pts_ns=999), "rgb_frame_identity_mismatch"),
    ],
)
def test_camera_source_frame_or_pts_mismatch_fails_before_fused_write(
    frame: TimestampedRgbFrame,
    code: str,
) -> None:
    store = _Store((_raw(1), _raw(2)))
    with pytest.raises(CaptureEventFusionError) as caught:
        CaptureEventFusionCoordinator(
            store,
            rgb_provider=_RgbProvider(frame),
        ).fuse(_request())
    assert caught.value.code == code
    assert store.fuse_calls == 0


def test_required_rgb_rejects_legacy_depth_without_exact_source_identity() -> None:
    store = _Store((_raw(1, source_id=None), _raw(2, source_id=None)))
    provider = _RgbProvider(_rgb())
    with pytest.raises(CaptureEventFusionError) as caught:
        CaptureEventFusionCoordinator(store, rgb_provider=provider).fuse(_request())
    assert caught.value.code == "rgb_depth_identity_unavailable"
    assert provider.calls == 0
    assert store.fuse_calls == 0


def test_required_rgb_unavailable_is_explicit_and_blocks_write() -> None:
    store = _Store((_raw(1), _raw(2)))
    with pytest.raises(CaptureEventFusionError) as caught:
        CaptureEventFusionCoordinator(store).fuse(_request())
    assert caught.value.code == "rgb_frame_unavailable"
    assert store.fuse_calls == 0


def test_depth_only_request_does_not_touch_rgb_provider() -> None:
    store = _Store((_raw(1), _raw(2)))
    provider = _RgbProvider(_rgb())
    outcome = CaptureEventFusionCoordinator(store, rgb_provider=provider).fuse(
        _request(require_rgb=False)
    )
    assert outcome.evidence["rgb"] == {
        "status": "not_requested",
        "provider_configured": True,
    }
    assert provider.calls == 0


def test_cache_only_rejects_before_store_or_provider() -> None:
    store = _Store((_raw(1), _raw(2)))
    provider = _RgbProvider(_rgb())
    with pytest.raises(CaptureEventFusionError) as caught:
        CaptureEventFusionCoordinator(store, rgb_provider=provider).fuse(
            _request(cache_only=True)
        )
    assert caught.value.code == "cache_only_forbids_capture_event_fusion"
    assert store.list_calls == 0
    assert provider.calls == 0


def test_event_identity_is_deterministic_for_exact_same_inputs() -> None:
    first = CaptureEventFusionCoordinator(
        _Store((_raw(1), _raw(2))),
        rgb_provider=_RgbProvider(_rgb()),
    ).fuse(_request())
    second = CaptureEventFusionCoordinator(
        _Store((_raw(1), _raw(2))),
        rgb_provider=_RgbProvider(_rgb()),
    ).fuse(_request())
    assert first.evidence["event_id"] == second.evidence["event_id"]


def _armed_provider(
    *,
    camera_sources: dict[int, str] | None = None,
    frames_per_camera: int = 8,
    max_total_bytes: int = 144,
) -> tuple[PipelineRgbFrameProvider, object]:
    provider = PipelineRgbFrameProvider(
        camera_sources=camera_sources or {0: "cam0"},
        frames_per_camera=frames_per_camera,
        max_width=4,
        max_height=3,
        max_frame_bytes=36,
        max_total_bytes=max_total_bytes,
    )
    return provider, provider.arm("cam0")


def test_pipeline_provider_is_dormant_before_and_after_manual_arm() -> None:
    provider = PipelineRgbFrameProvider(camera_sources={0: "cam0"})
    assert provider.health_snapshot()["armed"] is False
    assert provider.capture_arm(source_id=0, camera_id="cam0") is None

    arm = provider.arm("cam0")
    assert provider.capture_arm(source_id=1, camera_id="cam0") is None
    assert provider.capture_arm(source_id=0, camera_id="other") is None
    assert provider.capture_arm(source_id=0, camera_id="cam0") == arm
    provider.disarm(arm)

    health = provider.health_snapshot()
    assert health["armed"] is False
    assert health["frame_count"] == 0
    assert health["retained_bytes"] == 0
    assert provider.capture_arm(source_id=0, camera_id="cam0") is None


def test_pipeline_provider_admits_only_exact_arm_and_cohort_identity() -> None:
    provider, arm = _armed_provider()
    offered = _rgb()
    provider.offer(offered, arm=arm)
    selected = provider.provide("cam0", cohort=(_raw(1), _raw(2)))
    assert selected is not None
    assert (
        selected.source_id,
        selected.frame_id,
        selected.source_media_pts_ns,
    ) == (0, 42, 12_000_000)

    with pytest.raises(ValueError, match="duplicated"):
        provider.offer(offered, arm=arm)
    provider.disarm(arm)
    with pytest.raises(RuntimeError, match="not_armed"):
        provider.provide("cam0", cohort=(_raw(1), _raw(2)))


def test_pipeline_provider_rejects_cohort_from_another_source() -> None:
    provider, arm = _armed_provider()
    provider.offer(_rgb(), arm=arm)

    with pytest.raises(ValueError, match="configured camera source"):
        provider.provide(
            "cam0",
            cohort=(_raw(1), _raw(2, source_id=1)),
        )


def test_pipeline_provider_owns_immutable_bytes_and_checks_digest() -> None:
    pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    expected = pixels.copy()
    provider, arm = _armed_provider(max_total_bytes=36)
    provider.offer(_rgb(pixels=pixels), arm=arm)
    pixels.fill(255)

    selected = provider.provide("cam0", cohort=(_raw(1), _raw(2)))
    assert selected is not None
    np.testing.assert_array_equal(selected.pixels, expected)
    assert selected.pixels.flags.c_contiguous
    assert not selected.pixels.flags.writeable

    provider.disarm(arm)
    provider, arm = _armed_provider(max_total_bytes=36)
    with pytest.raises(ValueError, match="content_sha256"):
        provider.offer(
            _rgb(content_sha256=_digest("not-pixels")),
            arm=arm,
        )


def test_pipeline_provider_is_bounded_and_clears_previous_generation() -> None:
    provider, arm = _armed_provider(frames_per_camera=2, max_total_bytes=72)
    provider.offer(
        _rgb(
            captured_at_us=1_000_000,
            frame_id=41,
            source_media_pts_ns=11_000_000,
        ),
        arm=arm,
    )
    provider.offer(_rgb(), arm=arm)
    provider.offer(
        _rgb(
            captured_at_us=1_200_000,
            frame_id=43,
            source_media_pts_ns=13_000_000,
        ),
        arm=arm,
    )
    health = provider.health_snapshot()
    assert health["frame_count"] == 2
    assert health["retained_bytes"] == 72
    assert health["evicted_frames"] == 1
    provider.disarm(arm)

    next_arm = provider.arm("cam0")
    assert provider.provide("cam0", cohort=(_raw(1), _raw(2))) is None
    provider.disarm(next_arm)


def test_provider_configuration_is_bounded_and_exact() -> None:
    with pytest.raises(TypeError, match="map numeric source"):
        PipelineRgbFrameProvider(camera_sources="cam0")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        PipelineRgbFrameProvider(camera_sources={})
    with pytest.raises(ValueError, match="non-negative"):
        PipelineRgbFrameProvider(camera_sources={-1: "cam0"})
    with pytest.raises(ValueError, match="unique"):
        PipelineRgbFrameProvider(camera_sources={0: "cam0", 1: "cam0"})


def test_shared_and_ds9_provider_paths_open_no_second_camera_source() -> None:
    paths = (
        REPO_ROOT / "noesis_core" / "capture_event_fusion.py",
        REPO_ROOT / "DS9" / "noesis" / "capture_event_rgb_provider.py",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            str(node.module or "").split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        assert not (imports & {"cv2", "av", "gi"}), path
        assert "VideoCapture" not in names | attrs, path
