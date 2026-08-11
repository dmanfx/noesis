from __future__ import annotations

import base64
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import geometry.depth_source as depth_source
from geometry.depth_source import DepthStorageManager, MapAnythingDepthSource
from mapanything_config import (
    InferenceSettings,
    PerformanceSettings,
    ServiceConfig,
    ServiceSettings,
    StorageSettings,
)


def _service_config(depth_base: Path) -> ServiceConfig:
    return ServiceConfig(
        service=ServiceSettings(host="127.0.0.1", port=1, api_key="test-only"),
        inference=InferenceSettings(
            model_id="test-only",
            device="cpu",
            amp_dtype_name="fp32",
            memory_efficient_mono=False,
            memory_efficient_multi=False,
            apply_mask=True,
            mask_edges=True,
            confidence_percentile=10,
        ),
        performance=PerformanceSettings(
            max_res=64,
            mono_freq_hz=1.0,
            multi_batch_size=1,
            multi_interval_s=1.0,
            min_conf=0.0,
        ),
        storage=StorageSettings(
            depth_base=str(depth_base),
            calib_base=str(depth_base.parent / "calib"),
            max_snapshots_per_camera=0,
            snapshot_retention_minutes=0.0,
            max_total_bytes=None,
            async_enabled=False,
            queue_size=8,
            async_workers=1,
            async_max_workers=1,
            enforce_async=False,
            enforce_interval_s=1.0,
            quota_hysteresis_ratio=0.9,
            zarr_clevel=0,
            zarr_chunk_px=0,
        ),
    )


def _decode_grid(layer: dict[str, Any]) -> np.ndarray:
    return np.frombuffer(
        base64.b64decode(layer["grid_b64"]),
        dtype=np.float32,
    ).reshape(layer["grid_shape"])


@pytest.mark.parametrize("implementation", ["storage", "source"])
def test_floorplan_agl_offset_is_finite_and_applied_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    implementation: str,
) -> None:
    camera_id = f"agl_{implementation}"
    if implementation == "storage":
        target: Any = DepthStorageManager(
            tmp_path / implementation / "depth",
            max_snapshots_per_camera=0,
            retention_minutes=0.0,
            max_total_bytes=None,
            enable_async=False,
            enforce_async=False,
            zarr_clevel=0,
            zarr_chunk_px=0,
            min_conf=0.0,
        )
        storage = target
    else:
        target = MapAnythingDepthSource(
            _service_config(tmp_path / implementation / "depth")
        )
        storage = target.storage

    target.calibration_bundle = {
        "cameras": {
            "K": {camera_id: [4.0, 4.0, 1.5, 1.5]},
            "E": {
                camera_id: np.eye(4, dtype=np.float32)
                .reshape(-1, order="F")
                .astype(float)
                .tolist()
            },
        }
    }
    depth = np.array(
        [
            [1.0, 1.2, 1.4, 1.6],
            [1.8, 2.0, 2.2, 2.4],
            [2.6, 2.8, 3.0, 3.2],
            [3.4, 3.6, 3.8, 4.0],
        ],
        dtype=np.float32,
    )
    confidence = np.ones_like(depth, dtype=np.float32)
    mask = np.ones_like(depth, dtype=np.uint8)

    original_normalizer = depth_source._normalize_floorplan_agl_heights
    normalizer_inputs: list[np.ndarray] = []

    def _record_normalization(
        heights: np.ndarray,
        quality_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        normalizer_inputs.append(np.asarray(heights, dtype=np.float32).copy())
        return original_normalizer(heights, quality_mask)

    monkeypatch.setattr(
        depth_source,
        "_estimate_floor_y_from_horizontal_points",
        lambda *_args, **_kwargs: (-2.0, {"mode": "test_bias", "floor_y": -2.0}),
    )
    monkeypatch.setattr(
        depth_source,
        "_normalize_floorplan_agl_heights",
        _record_normalization,
    )

    try:
        storage.store(
            camera_id,
            int(time.time() * 1_000_000),
            depth,
            confidence,
            mask,
        )
        payload = target.generate_topdown_floorplan(
            camera_id,
            max_age_sec=60.0,
            grid_res_m=0.01,
            max_extent_m=10.0,
        )

        assert payload.get("error") is None
        assert isinstance(payload.get("snapshot_ref"), str)
        assert isinstance(payload.get("snapshot_id"), str)
        assert isinstance(payload.get("snapshot_content_sha256"), str)
        cached = target.generate_topdown_floorplan(
            camera_id,
            max_age_sec=60.0,
            grid_res_m=0.01,
            max_extent_m=10.0,
            cache_only=True,
        )
        assert cached.get("error") is None
        assert cached["served_from_cache"] is True
        assert cached["snapshot_id"] == payload["snapshot_id"]
        assert len(normalizer_inputs) == 1

        normalized_once, expected_offset_m = original_normalizer(
            normalizer_inputs[0],
            np.ones_like(normalizer_inputs[0], dtype=bool),
        )
        reported_offset_m = payload["height_agl_meta"]["floor_offset_m"]
        assert math.isfinite(reported_offset_m)
        assert reported_offset_m == pytest.approx(expected_offset_m)
        assert reported_offset_m > 0.5

        output_values = np.sort(
            _decode_grid(payload["height_agl"])[
                np.isfinite(_decode_grid(payload["height_agl"]))
            ]
        )
        np.testing.assert_allclose(
            output_values,
            np.sort(normalized_once),
            rtol=0.0,
            atol=1e-6,
        )
    finally:
        if implementation == "storage":
            target.shutdown(wait=True, timeout=2.0)
        else:
            target.close()
