from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from noesis.manual_depth_models import (
    build_manual_depth_overlay,
    normalize_manual_depth_model,
)
from noesis.pipelines import hooks


def _mapanything_reference() -> dict[str, object]:
    return {
        "enable": True,
        "name": "mapanything_fullframe",
        "config-file-path": "DS9/pipelines/config_infer_secondary_mapanything.ini",
        "engine": "DS9/models/engines/mapanything_images_294x518_b3_fp32.plan",
        "batch_size": 3,
        "gie_id": 2,
        "attach_tensor_meta": True,
    }


def test_da3_selector_replaces_only_manual_lane_and_preserves_registration() -> None:
    base = {
        "depth_registration": {"path": "DS9/config/depth_registration.json"},
        "models": {
            "depth_tracking": {
                "engine": (
                    "DS9/models/engines/"
                    "depth_anything_v2_metric_hypersim_vits_294x518_b3_fp16.engine"
                ),
                "config-file-path": (
                    "DS9/pipelines/config_infer_secondary_depth_tracking_da2.ini"
                ),
            },
            "mapanything": _mapanything_reference(),
        },
    }
    overlay = build_manual_depth_overlay(base, "da3metric-large")

    active = overlay["models"]["mapanything"]
    assert active["backend"] == "da3metric-large"
    assert active["network_mode"] == "fp16"
    assert active["engine"].endswith("da3metric_large_294x518_b3_fp16.engine")
    assert (
        overlay["depth_registration"]["mapanything_reference"]
        == base["models"]["mapanything"]
    )


def test_manual_depth_selector_defaults_and_fails_closed() -> None:
    assert normalize_manual_depth_model(None) == "mapanything"
    assert normalize_manual_depth_model("da3metric-large") == "da3metric-large"
    with pytest.raises(ValueError, match="unsupported manual depth model"):
        normalize_manual_depth_model("da3-small")


def test_da3_metric_scale_uses_calibrated_model_input_focal_length() -> None:
    calibration = SimpleNamespace(
        intrinsics=np.array(
            [[1000.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        image_size=(1920, 1080),
    )
    provider = SimpleNamespace(snapshot=lambda _source_id, _camera_id: calibration)
    processor = hooks.MapAnythingProcessor(
        pipeline=SimpleNamespace(
            depth_enabled=True,
            frame_size=(1920, 1080),
            bev_calibration=provider,
        ),
        storage=SimpleNamespace(),
        depth_pub=None,
        gie_id=2,
        depth_backend="da3metric-large",
    )

    metric_scale, focal_px = processor._da3_metric_scale(
        source_id=0,
        camera_id="living-room",
        target_size=(1920, 1080),
        model_size=(518, 294),
    )
    expected_focal = 950.0 * min(518.0 / 1920.0, 294.0 / 1080.0)
    assert math.isclose(focal_px, expected_focal, rel_tol=1e-12)
    assert math.isclose(metric_scale, expected_focal / 300.0, rel_tol=1e-12)


def test_da3_metric_scale_requires_calibration() -> None:
    processor = hooks.MapAnythingProcessor(
        pipeline=SimpleNamespace(depth_enabled=True, frame_size=(1920, 1080)),
        storage=SimpleNamespace(),
        depth_pub=None,
        gie_id=2,
        depth_backend="da3metric-large",
    )
    with pytest.raises(RuntimeError, match="requires the DS9 calibration provider"):
        processor._da3_metric_scale(
            source_id=0,
            camera_id="living-room",
            target_size=(1920, 1080),
            model_size=(518, 294),
        )


def test_da3_depth_is_scaled_to_meters_before_storage() -> None:
    observed: dict[str, object] = {}

    class _Handle:
        def wait(self, timeout=None):
            observed["timeout"] = timeout
            return SimpleNamespace(path="/validated/da3-depth.zarr")

    class _Storage:
        def store(self, camera_id, _timestamp_us, depth, _conf, _mask, *, attrs):
            observed["camera_id"] = camera_id
            observed["depth"] = np.asarray(depth).copy()
            observed["attrs"] = dict(attrs)
            return _Handle()

    calibration = SimpleNamespace(
        intrinsics=np.array(
            [[600.0, 0.0, 2.0], [0.0, 600.0, 1.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        image_size=(4, 3),
    )
    pipeline = SimpleNamespace(
        depth_enabled=True,
        frame_size=(4, 3),
        bev_calibration=SimpleNamespace(
            snapshot=lambda _source_id, _camera_id: calibration
        ),
        record_depth_frame=lambda _now: None,
    )
    processor = hooks.MapAnythingProcessor(
        pipeline=pipeline,
        storage=_Storage(),
        depth_pub=None,
        gie_id=2,
        camera_labels={0: "living-room"},
        depth_backend="da3metric-large",
    )
    result = processor.handle_numpy_arrays(
        source_id=0,
        frame_id=7,
        pts_ns=1,
        tensors={
            "depth": np.full((3, 4), 2.0, dtype=np.float32),
            "conf": np.ones((3, 4), dtype=np.float32),
            "mask": np.ones((3, 4), dtype=np.float32),
        },
        captured_while_enabled=True,
    )

    assert result is not None
    assert observed["camera_id"] == "living-room"
    assert np.allclose(observed["depth"], 4.0)
    attrs = observed["attrs"]
    assert attrs["manual_depth_backend"] == "da3metric-large"
    assert attrs["depth_units"] == "meters"
    assert attrs["model_input_focal_px"] == 600.0
    assert attrs["metric_scale"] == 2.0
