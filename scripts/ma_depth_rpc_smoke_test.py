#!/usr/bin/env python3
"""Smoke test for MapAnything depth RPC over WebSocket (DS8 runtime)."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - import guard
    print(f"[FAIL] numpy package required: {exc}")
    sys.exit(1)

try:
    import websockets  # type: ignore
except Exception as exc:  # pragma: no cover - import guard
    print(f"[FAIL] websockets package required: {exc}")
    sys.exit(1)


def _first_camera(cameras_path: Path) -> str:
    try:
        with cameras_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cams = data.get("cameras", {})
        if isinstance(cams, dict):
            first = next(iter(cams.values()))
            name = (first or {}).get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        return "living-room"
    except Exception:
        return "living-room"


async def _wait_for_ma_depth(
    uri: str,
    *,
    camera: str,
    request_id: str,
    ts_max_us: Optional[int] = None,
    timeout_s: float = 8.0,
) -> Dict[str, Any]:
    async with websockets.connect(uri, max_size=None) as ws:
        req: Dict[str, Any] = {
            "type": "get_ma_depth",
            "camera": camera,
            "request_id": request_id,
        }
        if ts_max_us is not None:
            req["ts_max_us"] = int(ts_max_us)
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
            if not isinstance(payload, dict):
                continue
            if payload.get("type") != "ma_depth_response":
                continue
            if payload.get("request_id") != request_id:
                continue
            return payload
        raise TimeoutError("No ma_depth_response received within timeout")


def _decode_depth(payload: Dict[str, Any]) -> Tuple[Tuple[int, int], np.ndarray]:
    container = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
    shape = container.get("shape")
    if not (isinstance(shape, list) and len(shape) == 2):
        raise ValueError(f"Missing/invalid shape: {shape!r}")
    height = int(shape[0])
    width = int(shape[1])
    depth_b64 = container.get("depth_b64") or container.get("depth_z_b64")
    if not isinstance(depth_b64, str) or not depth_b64:
        raise ValueError("Missing depth_b64")
    raw = base64.b64decode(depth_b64)
    depth = np.frombuffer(raw, dtype=np.float32)
    if depth.size < height * width:
        raise ValueError(f"Depth buffer too small: {depth.size} < {height*width}")
    depth = depth[: height * width].reshape((height, width))
    return (height, width), depth


def _decode_normals(payload: Dict[str, Any]) -> Tuple[Tuple[int, int], np.ndarray]:
    container = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
    shape = container.get("normals_shape")
    if not (isinstance(shape, list) and len(shape) == 3):
        raise ValueError(f"Missing/invalid normals_shape: {shape!r}")
    height = int(shape[0])
    width = int(shape[1])
    channels = int(shape[2])
    if channels != 3:
        raise ValueError(f"Unexpected normals channels: {channels}")
    normals_b64 = container.get("normals_b64")
    if not isinstance(normals_b64, str) or not normals_b64:
        raise ValueError("Missing normals_b64")
    dtype = container.get("normals_dtype") or "float16"
    raw = base64.b64decode(normals_b64)
    if dtype == "float16":
        normals = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dtype == "float32":
        normals = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported normals dtype: {dtype}")
    needed = height * width * channels
    if normals.size < needed:
        raise ValueError(f"Normals buffer too small: {normals.size} < {needed}")
    normals = normals[:needed].reshape((height, width, channels))
    return (height, width), normals


def _spawn_runtime(args: argparse.Namespace) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "noesis/ds8_runtime.py",
        "--pipeline-config",
        str(args.pipeline_config),
        "--cameras-config",
        str(args.cameras_config),
    ]
    if args.enable_rest:
        cmd.append("--enable-rest")
    env = os.environ.copy()
    env.setdefault("NOESIS_MOSAIC_RTSP_ENABLED", "0")
    env.setdefault("NOESIS_MOSAIC_WEBRTC_ENABLED", "0")
    env.setdefault("NOESIS_DEPTH_RPC_ENABLE_SECONDS", str(args.depth_enable_seconds))
    return subprocess.Popen(cmd, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="MapAnything depth RPC smoke test")
    parser.add_argument("--ws", default="ws://127.0.0.1:6008", help="WebSocket URL")
    parser.add_argument("--pipeline-config", type=Path, default=Path("config/infer.yaml"))
    parser.add_argument("--cameras-config", type=Path, default=Path("config/cameras.yaml"))
    parser.add_argument("--enable-rest", action="store_true", help="Start REST server when spawning runtime")
    parser.add_argument("--no-spawn", action="store_true", help="Do not spawn ds8_runtime; assume external runtime")
    parser.add_argument("--camera", default="", help="Camera id to request (default: first in cameras.yaml)")
    parser.add_argument(
        "--depth-enable-seconds",
        type=int,
        default=2,
        help="Depth burst duration for RPC-triggered enables (env NOESIS_DEPTH_RPC_ENABLE_SECONDS).",
    )
    parser.add_argument("--check-normals", action="store_true", help="Require normals payload in ma_depth_response")
    args = parser.parse_args()

    proc = None
    if not args.no_spawn:
        proc = _spawn_runtime(args)
        time.sleep(10.0)  # allow pipeline + WS to come up

    try:
        camera = args.camera.strip() or _first_camera(args.cameras_config)

        # Cache-first: should return cached if present; otherwise may trigger a fresh burst.
        now_us = int(time.time() * 1_000_000)
        cache_resp = None
        last_err: Optional[Exception] = None
        for attempt in range(1, 11):
            try:
                cache_resp = asyncio.run(
                    _wait_for_ma_depth(
                        args.ws,
                        camera=camera,
                        request_id=f"ma-depth-cache-{attempt}",
                        ts_max_us=now_us,
                    )
                )
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1.0)
        if cache_resp is None:
            raise RuntimeError(f"Unable to retrieve cache-first ma_depth_response: {last_err}")
        if cache_resp.get("ok") is False:
            raise RuntimeError(f"cache-first ma_depth_response error: {cache_resp.get('error')}")

        (h0, w0), depth0 = _decode_depth(cache_resp)
        finite0 = np.isfinite(depth0)
        pos0 = finite0 & (depth0 > 0)
        if not pos0.any():
            raise RuntimeError(f"cache-first depth has no positive finite samples (shape={h0}x{w0})")
        ts0 = int(cache_resp.get("ts_us") or 0)

        time.sleep(0.7)  # avoid per-camera RPC throttle window

        # Fresh: should return a newer snapshot (served_from_cache=False).
        fresh_resp = None
        last_err = None
        for attempt in range(1, 11):
            try:
                fresh_resp = asyncio.run(
                    _wait_for_ma_depth(
                        args.ws,
                        camera=camera,
                        request_id=f"ma-depth-fresh-{attempt}",
                        ts_max_us=None,
                    )
                )
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1.0)
        if fresh_resp is None:
            raise RuntimeError(f"Unable to retrieve fresh ma_depth_response: {last_err}")
        if fresh_resp.get("ok") is False:
            raise RuntimeError(f"fresh ma_depth_response error: {fresh_resp.get('error')}")
        if fresh_resp.get("served_from_cache") is True:
            raise RuntimeError("fresh ma_depth_response unexpectedly served_from_cache=true")

        (h1, w1), depth1 = _decode_depth(fresh_resp)
        if (h1, w1) != (h0, w0):
            raise RuntimeError(f"shape changed unexpectedly: {(h0,w0)} -> {(h1,w1)}")
        finite1 = np.isfinite(depth1)
        pos1 = finite1 & (depth1 > 0)
        if not pos1.any():
            raise RuntimeError(f"fresh depth has no positive finite samples (shape={h1}x{w1})")
        ts1 = int(fresh_resp.get("ts_us") or 0)
        if ts0 and ts1 and ts1 <= ts0:
            raise RuntimeError(f"fresh ts_us did not increase: {ts0} -> {ts1}")

        if args.check_normals:
            container = fresh_resp.get("payload") if isinstance(fresh_resp.get("payload"), dict) else fresh_resp
            normals_error = container.get("normals_error")
            if normals_error:
                raise RuntimeError(f"normals_error present: {normals_error}")
            (_, _), normals = _decode_normals(fresh_resp)
            mag = np.linalg.norm(normals, axis=-1)
            if not np.any(mag > 0.1):
                raise RuntimeError("normals payload has no valid samples")

        print("[PASS] ma_depth RPC returned depth payload (cache-first + fresh)")
        return 0
    finally:
        if proc is not None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=6.0)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
