from __future__ import annotations

import base64
import time
from pathlib import Path

import numpy as np

from geometry.depth_source import DepthStorageManager, _infer_image_flips_from_extrinsics


def _decode_grid(layer: dict) -> np.ndarray:
    shape = layer.get("grid_shape")
    assert isinstance(shape, list)
    rows, cols = int(shape[0]), int(shape[1])
    raw = base64.b64decode(layer["grid_b64"])
    return np.frombuffer(raw, dtype=np.float32).reshape(rows, cols)


def _roll_180_z_extrinsics() -> list[float]:
    twc = np.eye(4, dtype=np.float32)
    twc[:3, :3] = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return np.linalg.inv(twc).reshape(-1, order="F").astype(float).tolist()


def test_floorplan_stays_camera_local_when_image_flip_hint_is_true(tmp_path: Path) -> None:
    camera_id = "living_test"
    extrinsics = _roll_180_z_extrinsics()
    expected_flip = _infer_image_flips_from_extrinsics(extrinsics)
    assert expected_flip == (True, False)

    mgr = DepthStorageManager(
        base_path=tmp_path / "depth",
        max_snapshots_per_camera=0,
        retention_minutes=0.0,
        max_total_bytes=None,
        enable_async=False,
        enforce_async=False,
        zarr_clevel=0,
        zarr_chunk_px=0,
        min_conf=0.0,
    )
    mgr.calibration_bundle = {
        "cameras": {
            "K": {camera_id: [1.0, 1.0, 2.0, 1.0]},
            "E": {camera_id: extrinsics},
        }
    }

    depth = np.zeros((3, 5), dtype=np.float32)
    conf = np.zeros_like(depth, dtype=np.float32)
    mask = np.zeros_like(depth, dtype=np.uint8)

    # Put a single observed point on the far-right side of the image.
    depth[1, 4] = 1.0
    conf[1, 4] = 1.0
    mask[1, 4] = 1

    ts_us = int(time.time() * 1_000_000)
    mgr.store(camera_id, ts_us, depth, conf, mask)

    payload = mgr.generate_topdown_floorplan(
        camera_id,
        max_age_sec=60.0,
        grid_res_m=1.0,
        max_extent_m=4.0,
    )

    assert payload.get("error") is None
    assert payload["image_flip"] == {"u": True, "v": False}

    density = _decode_grid(payload["density"])
    occupied = np.argwhere(density > 0)
    assert occupied.shape[0] == 1

    _, occupied_x = occupied[0]
    assert occupied_x >= density.shape[1] // 2

    height = _decode_grid(payload["height"])
    assert np.isfinite(height[occupied[0][0], occupied_x])


def test_floorplan_bounds_follow_observed_room_extent_not_request_extent(tmp_path: Path) -> None:
    camera_id = "living_test"
    mgr = DepthStorageManager(
        base_path=tmp_path / "depth",
        max_snapshots_per_camera=0,
        retention_minutes=0.0,
        max_total_bytes=None,
        enable_async=False,
        enforce_async=False,
        zarr_clevel=0,
        zarr_chunk_px=0,
        min_conf=0.0,
    )
    mgr.calibration_bundle = {
        "cameras": {
            "K": {camera_id: [1.0, 1.0, 2.0, 1.0]},
            "E": {camera_id: np.eye(4, dtype=np.float32).reshape(-1, order="F").astype(float).tolist()},
        }
    }

    depth = np.zeros((3, 5), dtype=np.float32)
    conf = np.zeros_like(depth, dtype=np.float32)
    mask = np.zeros_like(depth, dtype=np.uint8)
    depth[1, 1:4] = 2.0
    conf[1, 1:4] = 1.0
    mask[1, 1:4] = 1

    ts_us = int(time.time() * 1_000_000)
    mgr.store(camera_id, ts_us, depth, conf, mask)

    payload = mgr.generate_topdown_floorplan(
        camera_id,
        max_age_sec=60.0,
        grid_res_m=0.25,
        max_extent_m=20.0,
    )

    assert payload.get("error") is None
    bounds = payload["bounds"]
    assert bounds["max_x"] - bounds["min_x"] < 10.0
    assert bounds["max_z"] < 10.0
