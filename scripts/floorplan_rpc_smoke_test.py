#!/usr/bin/env python3
"""Smoke test for DS8 floorplan RPC over WebSocket."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml

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


async def _wait_for_floorplan(uri: str, camera: str, max_age_sec: float, timeout_s: float = 20.0) -> Dict[str, Any]:
    async with websockets.connect(uri) as ws:
        req = {
            "type": "get_floorplan",
            "camera": camera,
            "request_id": "floorplan-smoke",
            "max_age_sec": float(max_age_sec),
        }
        await ws.send(json.dumps(req))
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            try:
                payload = json.loads(msg)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("type") == "floorplan_response" and payload.get("request_id") == req["request_id"]:
                return payload
        raise TimeoutError("No floorplan_response received within timeout")


def _spawn_runtime(args: argparse.Namespace) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "noesis/ds8_runtime.py",
        "--pipeline-config",
        str(args.pipeline_config),
        "--cameras-config",
        str(args.cameras_config),
    ]
    if args.depth_seconds is not None:
        cmd.extend(["--depth-enable-seconds", str(args.depth_seconds)])
    env = os.environ.copy()
    env.setdefault("NOESIS_MOSAIC_RTSP_ENABLED", "0")
    env.setdefault("NOESIS_MOSAIC_WEBRTC_ENABLED", "0")
    if args.enable_rest:
        cmd.append("--enable-rest")
    proc = subprocess.Popen(cmd, env=env)
    return proc


def main() -> int:
    parser = argparse.ArgumentParser(description="Floorplan RPC smoke test")
    parser.add_argument("--ws", default="ws://127.0.0.1:6008", help="WebSocket URL")
    parser.add_argument("--pipeline-config", type=Path, default=Path("config/infer.yaml"))
    parser.add_argument("--cameras-config", type=Path, default=Path("config/cameras.yaml"))
    parser.add_argument("--camera", default="", help="Camera id override (default: first camera in cameras.yaml)")
    parser.add_argument("--enable-rest", action="store_true", help="Start REST server when spawning runtime")
    parser.add_argument("--no-spawn", action="store_true", help="Do not spawn ds8_runtime; assume external runtime")
    parser.add_argument(
        "--depth-seconds",
        type=int,
        default=120,
        help="Number of seconds to enable depth on startup (omit to use runtime default).",
    )
    parser.add_argument(
        "--max-age-sec",
        type=float,
        default=1200.0,
        help="Accept floorplans computed from snapshots up to this age (seconds).",
    )
    args = parser.parse_args()

    proc = None
    if not args.no_spawn:
        proc = _spawn_runtime(args)
        time.sleep(5.0)  # allow startup

    try:
        camera = str(args.camera or "").strip() or _first_camera(args.cameras_config)
        payload = None
        last_err = None
        for _ in range(10):
            try:
                payload = asyncio.run(_wait_for_floorplan(args.ws, camera, args.max_age_sec))
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1.0)
        if payload is None:
            raise RuntimeError(f"Unable to retrieve floorplan_response: {last_err}")
        retries = 0
        retriable = {"no_depth", "stale_depth", "missing_calibration", "no_points"}
        while payload.get("error") in retriable and retries < 5:
            time.sleep(2.0)
            try:
                payload = asyncio.run(_wait_for_floorplan(args.ws, camera, args.max_age_sec))
            except Exception as exc:
                last_err = exc
                payload = None
            retries += 1
            if payload is None:
                continue
        if payload is None or payload.get("error"):
            print(f"[FAIL] floorplan_response error: {payload.get('error') if payload else last_err}")
            return 1
        required = ("camera_id", "bounds", "density", "height", "distance")
        missing = [k for k in required if k not in payload]
        if missing:
            print(f"[FAIL] missing fields in response: {missing}")
            return 1
        if "kitchen" in camera.lower():
            for key in ("obstacle_height", "walkable"):
                if key not in payload:
                    print(f"[FAIL] missing kitchen clean layer: {key}")
                    return 1
        print("[PASS] floorplan RPC returned payload without error")
        return 0
    finally:
        if proc is not None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
