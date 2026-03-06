#!/usr/bin/env python3
"""Smoke test for DS8 calibration RPC wiring over WebSocket.

Validates that DS8 runtime wires handlers for:
- set_extrinsics in strict pose-only mode (legacy E/Twc-only payloads rejected)
- set_align (should not return error=no_handler)
- pixel_to_world (should not return error=no_handler)

This script intentionally avoids persisting extrinsics by sending legacy
set_extrinsics payloads without a valid pose while strict mode is enabled.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

try:
    import websockets  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"[FAIL] websockets package required: {exc}")
    sys.exit(1)


def _spawn_runtime(args: argparse.Namespace) -> subprocess.Popen:
    ws_url = urlparse(str(args.ws))
    ws_port = ws_url.port or 6008
    cmd = [
        sys.executable,
        "noesis/ds8_runtime.py",
        "--pipeline-config",
        str(args.pipeline_config),
        "--cameras-config",
        str(args.cameras_config),
        "--ws-port",
        str(ws_port),
        "--depth-enable-seconds",
        "0",
        "--disable-rest",
    ]
    env = os.environ.copy()
    env.setdefault("NOESIS_MOSAIC_RTSP_ENABLED", "0")
    env.setdefault("NOESIS_MOSAIC_WEBRTC_ENABLED", "0")
    env.setdefault("NOESIS_MAPANYTHING_POSTPROCESS_ENABLED", "0")
    env.setdefault("NOESIS_CALIBRATION_POSE_ONLY", "1")
    return subprocess.Popen(cmd, env=env)


async def _recv_json(ws, *, timeout_s: float) -> Optional[Dict[str, Any]]:
    try:
        msg = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return None
    if isinstance(msg, (bytes, bytearray)):
        return None
    try:
        payload = json.loads(msg)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


async def _wait_for_type(ws, expected: str, *, timeout_s: float = 8.0) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        payload = await _recv_json(ws, timeout_s=2.0)
        if not payload:
            continue
        if payload.get("type") == expected:
            return payload
    raise TimeoutError(f"Timed out waiting for {expected}")


async def _run(uri: str) -> None:
    async with websockets.connect(uri, max_size=None) as ws:
        # 1) set_align should be wired and succeed.
        # Try to reuse the current scale from calibration-bundle to minimise side effects.
        s_obj_to_m = 1.0
        camera_id_for_identity_extrinsics: Optional[str] = None
        try:
            bundle_msg = await _wait_for_type(ws, "calibration-bundle", timeout_s=2.0)
            data = bundle_msg.get("data") if isinstance(bundle_msg, dict) else None
            align = data.get("align") if isinstance(data, dict) else None
            units = align.get("units") if isinstance(align, dict) else None
            s_val = units.get("s_obj_to_m") if isinstance(units, dict) else None
            if isinstance(s_val, (int, float)) and float(s_val) > 0:
                s_obj_to_m = float(s_val)
            cams = data.get("cameras") if isinstance(data, dict) else None
            if isinstance(cams, dict):
                k_table = cams.get("K")
                e_table = cams.get("E")
                if isinstance(k_table, dict) and k_table:
                    camera_id_for_identity_extrinsics = next(iter(k_table.keys()))
                elif isinstance(e_table, dict) and e_table:
                    camera_id_for_identity_extrinsics = next(iter(e_table.keys()))
        except Exception:
            pass

        await ws.send(json.dumps({"type": "set_align", "align": {"units": {"s_obj_to_m": s_obj_to_m}}}))
        align_res = await _wait_for_type(ws, "set_align_result", timeout_s=10.0)
        if not align_res.get("ok"):
            raise RuntimeError(f"set_align_result ok=false error={align_res.get('error')!r}")

        # 2) strict pose-only mode: reject legacy E-only set_extrinsics payloads.
        identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        strict_cam = camera_id_for_identity_extrinsics or "__smoke_test__"
        await ws.send(json.dumps({"type": "set_extrinsics", "cameraId": strict_cam, "E": identity}))
        extr_e_only = await _wait_for_type(ws, "set_extrinsics_result", timeout_s=10.0)
        if extr_e_only.get("error") == "no_handler":
            raise RuntimeError("set_extrinsics_result error=no_handler")
        if extr_e_only.get("ok"):
            raise RuntimeError("strict pose-only mode unexpectedly accepted legacy E-only set_extrinsics")
        if extr_e_only.get("error") != "pose_required":
            raise RuntimeError(
                f"strict pose-only mode should reject legacy E-only payload with pose_required; got {extr_e_only.get('error')!r}"
            )

        # 3) strict pose-only mode: reject legacy Twc-only set_extrinsics payloads.
        await ws.send(json.dumps({"type": "set_extrinsics", "cameraId": strict_cam, "Twc": identity}))
        extr_twc_only = await _wait_for_type(ws, "set_extrinsics_result", timeout_s=10.0)
        if extr_twc_only.get("error") == "no_handler":
            raise RuntimeError("set_extrinsics_result error=no_handler")
        if extr_twc_only.get("ok"):
            raise RuntimeError("strict pose-only mode unexpectedly accepted legacy Twc-only set_extrinsics")
        if extr_twc_only.get("error") != "pose_required":
            raise RuntimeError(
                f"strict pose-only mode should reject legacy Twc-only payload with pose_required; got {extr_twc_only.get('error')!r}"
            )

        # 4) pixel_to_world should be wired (likely calibration_missing for dummy cam).
        await ws.send(json.dumps({"type": "pixel_to_world", "camId": "__smoke_test__", "u": 0, "v": 0, "reqId": "smoke"}))
        p2w_res = await _wait_for_type(ws, "pixel_to_world_response", timeout_s=10.0)
        if p2w_res.get("error") == "no_handler":
            raise RuntimeError("pixel_to_world_response error=no_handler")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibration RPC smoke test (DS8 runtime)")
    parser.add_argument("--ws", default="ws://127.0.0.1:6008", help="WebSocket URL")
    parser.add_argument("--pipeline-config", default="config/infer.yaml")
    parser.add_argument("--cameras-config", default="config/cameras.yaml")
    parser.add_argument("--no-spawn", action="store_true", help="Do not spawn ds8_runtime; assume external runtime")
    args = parser.parse_args()

    proc = None
    if not args.no_spawn:
        proc = _spawn_runtime(args)
        time.sleep(10.0)

    try:
        asyncio.run(_run(str(args.ws)))
        print("[PASS] DS8 calibration RPC strict pose-only expectations satisfied")
        return 0
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1
    finally:
        if proc is not None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
