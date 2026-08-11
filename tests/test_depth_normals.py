from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from geometry.depth_source import DepthStorageManager


def _encode_base64(array: np.ndarray) -> str:
    return base64.b64encode(array.tobytes()).decode("ascii")


def test_attach_normals_camera_space(tmp_path: Path) -> None:
    storage = DepthStorageManager(
        base_path=tmp_path,
        max_snapshots_per_camera=4,
        retention_minutes=1,
        enable_async=False,
        max_queue_size=1,
        worker_count=1,
        max_worker_count=1,
    )

    fx = 100.0
    fy = 100.0
    width, height = 32, 24
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    storage.calibration_bundle = {
        "cameras": {
            "K": {
                "cam1": [fx, fy, cx, cy],
            }
        }
    }

    depth = np.full((height, width), 2.0, dtype=np.float32)
    conf = np.ones_like(depth, dtype=np.float32)
    mask = np.ones_like(depth, dtype=np.uint8)

    payload = {
        "ts": 123456,
        "depth_b64": _encode_base64(depth),
        "conf_b64": _encode_base64(conf),
        "mask_b64": _encode_base64(mask),
        "shape": [height, width],
    }

    storage.attach_normals_to_payload("cam1", payload, space="camera", dtype="float16")

    assert "normals_b64" in payload
    assert payload.get("normals_error") is None
    assert payload.get("normals_shape") == [height, width, 3]
    assert payload.get("normals_dtype") == "float16"
    assert payload.get("normals_space") == "camera"

    raw = base64.b64decode(payload["normals_b64"])
    normals = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    normals = normals.reshape((height, width, 3))

    mag = np.linalg.norm(normals, axis=-1)
    valid = mag > 0.1
    assert np.any(valid)

    mean = normals[valid].mean(axis=0)
    assert abs(float(mean[0])) < 0.05
    assert abs(float(mean[1])) < 0.05
    assert float(mean[2]) < -0.9

    storage.shutdown(wait=True)
