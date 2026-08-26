from __future__ import annotations

import base64
import os

import cv2
import numpy as np
import pytest
import requests

from noesis.config.mapanything import load_service_config
from noesis_core.runtime_secrets import load_camera_uri_registry


CAMERA_SECRET_REF = os.environ.get("NOESIS_MA_FRAME_CAMERA_SECRET", "living-room")
RUN_INTEGRATION = os.environ.get("NOESIS_RUN_MA_FRAME_TEST", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


@pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="Manual integration test (requires RTSP camera + running mapanything service). Set NOESIS_RUN_MA_FRAME_TEST=1 to run.",
)
def test_ma_frame_infer_mono_roundtrip() -> None:
    service_config = load_service_config()
    camera_registry = load_camera_uri_registry()
    rtsp_url = camera_registry[CAMERA_SECRET_REF]
    service_url = f"{service_config.service.base_url}/infer_mono"
    cap = cv2.VideoCapture(rtsp_url)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        pytest.skip(f"No frame captured from camera secret: {CAMERA_SECRET_REF}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    payload = {
        "view": {
            "img_b64": base64.b64encode(rgb.tobytes()).decode("ascii"),
            "shape": [h, w, c],
            "cam_id": "test",
            "intrinsics": None,
        }
    }

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": service_config.service.api_key,
    }
    try:
        resp = requests.post(service_url, json=payload, headers=headers, timeout=8)
    except requests.RequestException as exc:
        pytest.skip(f"MapAnything service unavailable: {exc}")

    assert resp.ok, f"infer_mono failed ({resp.status_code}): {resp.text}"
    data = resp.json()

    shape = data.get("shape") or [0, 0]
    assert isinstance(shape, list) and len(shape) == 2
    out_h, out_w = int(shape[0]), int(shape[1])
    assert out_h > 0 and out_w > 0

    depth_b64 = data.get("depth_b64")
    conf_b64 = data.get("conf_b64")
    mask_b64 = data.get("mask_b64")
    assert depth_b64 and conf_b64 and mask_b64

    depth = np.frombuffer(base64.b64decode(depth_b64), dtype=np.float32).reshape((out_h, out_w))
    conf = np.frombuffer(base64.b64decode(conf_b64), dtype=np.float32).reshape((out_h, out_w))
    mask = np.frombuffer(base64.b64decode(mask_b64), dtype=np.uint8).reshape((out_h, out_w)).astype(bool)

    assert depth.shape == (out_h, out_w)
    assert conf.shape == (out_h, out_w)
    assert mask.shape == (out_h, out_w)
