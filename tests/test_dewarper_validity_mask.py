from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from noesis.pipelines.hooks import (
    MapAnythingProcessor,
    _build_dewarper_fov_mask,
    _load_dewarper_fov_spec,
)
from noesis.pipelines.hooks_v3dt_reimpl import MapAnythingProcessor as V3DTMapAnythingProcessor


ROOT = Path(__file__).resolve().parents[1]


class _PipelineStub:
    def __init__(self, config: dict, frame_size: tuple[int, int]) -> None:
        self.yaml_path = ROOT / "config" / "infer.yaml"
        self.config = config
        self.frame_size = frame_size
        self.depth_enabled = True
        self.recorded_depth_frames = 0

    def record_depth_frame(self, _timestamp: float) -> None:
        self.recorded_depth_frames += 1


class _StorageStub:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def store(
        self,
        camera_id: str,
        ts_us: int,
        depth: np.ndarray,
        conf: np.ndarray,
        mask: np.ndarray,
    ) -> object:
        self.records.append(
            {
                "camera_id": camera_id,
                "ts_us": int(ts_us),
                "depth": np.array(depth, copy=True),
                "conf": np.array(conf, copy=True),
                "mask": np.array(mask, copy=True),
            }
        )
        path = ROOT / "tmp" / f"{camera_id}_{ts_us}.zarr"

        class _WriteHandle:
            @staticmethod
            def wait(timeout: float | None = None) -> object:
                assert timeout == 30.0
                return SimpleNamespace(path=path)

        return _WriteHandle()


def _load_infer_config() -> dict:
    return yaml.safe_load((ROOT / "config" / "infer.yaml").read_text(encoding="utf-8"))


def test_dewarper_fov_mask_uses_configured_circular_source_region() -> None:
    cfg = _load_infer_config()
    source_cfg = cfg["sources"][2]
    mask_cfg = cfg["dewarper_validity_masks"]["sources"]["2"]

    spec = _load_dewarper_fov_spec(
        source_cfg=source_cfg,
        pipeline_yaml_path=ROOT / "config" / "infer.yaml",
        mask_cfg=mask_cfg,
    )

    assert spec is not None
    assert spec.source_size == (1280, 720)
    assert spec.source_valid_region == "circle"

    mask = _build_dewarper_fov_mask(spec, target_size=(1920, 1080), erode_px=1)

    assert mask.shape == (1080, 1920)
    assert mask.dtype == bool
    assert 0.75 < float(mask.mean()) < 0.85
    assert int(mask[:, -1].sum()) == 0


def test_mapanything_depth_store_keeps_dense_depth_and_downweights_outside_fov() -> None:
    cfg = _load_infer_config()
    storage = _StorageStub()
    pipeline = _PipelineStub(cfg, frame_size=(96, 54))
    processor = MapAnythingProcessor(
        pipeline=pipeline,  # type: ignore[arg-type]
        storage=storage,  # type: ignore[arg-type]
        depth_pub=None,
        gie_id=2,
        camera_labels={2: "kitchen"},
    )

    result = processor.handle_numpy_arrays(
        source_id=2,
        frame_id=7,
        pts_ns=1_780_000_000_000_000_000,
        tensors={
            "depth": np.ones((54, 96), dtype=np.float32),
            "confidence": np.ones((54, 96), dtype=np.float32),
        },
    )

    assert result is not None
    assert len(storage.records) == 1
    record = storage.records[0]
    depth = record["depth"]
    conf = record["conf"]
    mask = record["mask"]
    assert isinstance(depth, np.ndarray)
    assert isinstance(conf, np.ndarray)
    assert isinstance(mask, np.ndarray)
    assert depth.shape == (54, 96)
    assert mask.shape == (54, 96)
    assert int(mask.sum()) == mask.size
    assert np.isfinite(depth).all()
    assert float(np.min(conf)) == pytest.approx(0.25)
    assert float(np.max(conf)) == pytest.approx(1.0)
    assert pipeline.recorded_depth_frames == 1


def test_mapanything_depth_store_treats_degenerate_tensor_mask_as_uncertainty() -> None:
    cfg = _load_infer_config()
    storage = _StorageStub()
    pipeline = _PipelineStub(cfg, frame_size=(96, 54))
    processor = MapAnythingProcessor(
        pipeline=pipeline,  # type: ignore[arg-type]
        storage=storage,  # type: ignore[arg-type]
        depth_pub=None,
        gie_id=2,
        camera_labels={2: "kitchen"},
    )

    depth = np.linspace(0.5, 6.0, 54 * 96, dtype=np.float32).reshape((54, 96))
    result = processor.handle_numpy_arrays(
        source_id=2,
        frame_id=9,
        pts_ns=1_780_000_000_000_000_000,
        tensors={
            "depth": depth,
            "confidence": np.ones((54, 96), dtype=np.float32),
            "mask": np.zeros((54, 96), dtype=np.float32),
        },
    )

    assert result is not None
    assert len(storage.records) == 1
    record = storage.records[0]
    stored_depth = record["depth"]
    stored_conf = record["conf"]
    stored_mask = record["mask"]
    assert isinstance(stored_depth, np.ndarray)
    assert isinstance(stored_conf, np.ndarray)
    assert isinstance(stored_mask, np.ndarray)
    assert int(stored_mask.sum()) == stored_mask.size
    assert np.isfinite(stored_depth).all()
    assert float(np.min(stored_conf)) == pytest.approx(0.125)
    assert float(np.max(stored_conf)) == pytest.approx(0.5)
    assert pipeline.recorded_depth_frames == 1


def test_mapanything_depth_store_does_not_hard_drop_empty_fov() -> None:
    cfg = _load_infer_config()
    storage = _StorageStub()
    pipeline = _PipelineStub(cfg, frame_size=(8, 8))
    processor = MapAnythingProcessor(
        pipeline=pipeline,  # type: ignore[arg-type]
        storage=storage,  # type: ignore[arg-type]
        depth_pub=None,
        gie_id=2,
    )
    processor._dewarper_fov_masks[(0, 8, 8)] = np.zeros((8, 8), dtype=bool)

    result = processor.handle_numpy_arrays(
        source_id=0,
        frame_id=8,
        pts_ns=1_780_000_000_000_000_000,
        tensors={"depth": np.ones((8, 8), dtype=np.float32)},
    )

    assert result is not None
    assert len(storage.records) == 1
    record = storage.records[0]
    assert np.all(np.asarray(record["mask"], dtype=np.uint8) == 1)
    assert np.isfinite(np.asarray(record["depth"], dtype=np.float32)).all()
    assert pipeline.recorded_depth_frames == 1


def test_v3dt_mapanything_uses_the_same_calibrated_fov_mask() -> None:
    cfg = _load_infer_config()
    outputs: list[np.ndarray] = []
    for processor_type in (MapAnythingProcessor, V3DTMapAnythingProcessor):
        storage = _StorageStub()
        pipeline = _PipelineStub(cfg, frame_size=(96, 54))
        processor = processor_type(
            pipeline=pipeline,  # type: ignore[arg-type]
            storage=storage,  # type: ignore[arg-type]
            depth_pub=None,
            gie_id=2,
            camera_labels={2: "kitchen"},
        )
        result = processor.handle_numpy_arrays(
            source_id=2,
            frame_id=10,
            pts_ns=1_780_000_000_000_000_000,
            tensors={"depth": np.ones((54, 96), dtype=np.float32)},
        )
        assert result is not None
        outputs.append(np.asarray(storage.records[0]["mask"], dtype=np.uint8))
    assert np.array_equal(outputs[0], outputs[1])
