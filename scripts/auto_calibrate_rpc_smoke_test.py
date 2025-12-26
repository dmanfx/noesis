#!/usr/bin/env python3
"""Smoke test for auto-calibration RPC over WebSocket (DS8 runtime)."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse
from typing import Any, Dict

try:
    import websockets  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"[FAIL] websockets package required: {exc}")
    sys.exit(1)

DEFAULT_SMOKE_CAMERA = "__smoke_test__"


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
    ]
    if args.enable_rest:
        cmd.append("--enable-rest")
    else:
        cmd.append("--disable-rest")
    env = os.environ.copy()
    env.setdefault("NOESIS_MOSAIC_JPEG_ENABLED", "0")
    env.setdefault("NOESIS_MOSAIC_RTSP_ENABLED", "0")
    env.setdefault("NOESIS_MOSAIC_WEBRTC_ENABLED", "0")
    env.setdefault("NOESIS_AUTOCALIB_ENABLE_SECONDS", "1")
    return subprocess.Popen(cmd, env=env)


async def _wait_for_result(uri: str, camera: str | None, timeout_s: float = 25.0) -> Dict[str, Any]:
    async with websockets.connect(uri, max_size=None) as ws:
        req: Dict[str, Any] = {"type": "auto_calibrate_pose"}
        if camera is not None:
            req["camera"] = camera
        await ws.send(json.dumps(req))
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            if isinstance(msg, (bytes, bytearray)):
                continue
            try:
                payload = json.loads(msg)
            except Exception:
                continue
            if isinstance(payload, dict) and payload.get("type") == "auto_calibrate_result":
                return payload
        raise TimeoutError("No auto_calibrate_result received")


def _validate_result(payload: Dict[str, Any]) -> str:
    if payload.get("type") != "auto_calibrate_result":
        return f"unexpected type: {payload.get('type')!r}"
    if not isinstance(payload.get("ok"), bool):
        return "ok is not boolean"
    if not isinstance(payload.get("updated"), list):
        return "updated is not a list"
    if payload.get("error") == "no_handler":
        return "error is no_handler"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-calibrate RPC smoke test")
    parser.add_argument("--ws", default="ws://127.0.0.1:6008", help="WebSocket URL")
    parser.add_argument("--pipeline-config", default="config/infer.yaml")
    parser.add_argument("--cameras-config", default="config/cameras.yaml")
    parser.add_argument(
        "--camera",
        default=DEFAULT_SMOKE_CAMERA,
        help=(
            "Camera ID to request. Defaults to a non-existent ID to avoid persisting calibration; "
            "use an actual camera ID to exercise the full auto-calibration path."
        ),
    )
    parser.add_argument(
        "--calibrate-all",
        action="store_true",
        help="Send auto_calibrate_pose without a camera id (may persist camera_calibration.json).",
    )
    parser.add_argument("--enable-rest", action="store_true", help="Start REST server when spawning runtime")
    parser.add_argument("--no-spawn", action="store_true", help="Do not spawn ds8_runtime; assume external runtime")
    args = parser.parse_args()

    proc = None
    if not args.no_spawn:
        proc = _spawn_runtime(args)
        time.sleep(10.0)

    try:
        payload = None
        last_err: Exception | None = None
        for _ in range(8):
            try:
                camera = None if args.calibrate_all else str(args.camera)
                payload = asyncio.run(_wait_for_result(args.ws, camera=camera))
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1.0)
        if payload is None:
            raise RuntimeError(f"auto_calibrate_result not received: {last_err}")
        error = _validate_result(payload)
        if error:
            print(f"[FAIL] {error}")
            return 1
        print("[PASS] auto_calibrate_result received with valid fields")
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
