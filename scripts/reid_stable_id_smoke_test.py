#!/usr/bin/env python3
"""Smoke test to ensure stable IDs are present and persistent."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time

try:
    import websockets  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"[FAIL] websockets package required: {exc}")
    sys.exit(1)


async def _collect_tracking(uri: str, duration: float = 10.0) -> dict:
    seen = {}
    non_null = set()
    start = time.time()
    async with websockets.connect(uri) as ws:
        while time.time() - start < duration:
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
            if payload.get("type") != "tracking":
                continue
            tracks = payload.get("tracks") or []
            for track in tracks:
                sid = track.get("stable_id")
                cam = track.get("camera_id")
                if sid is None:
                    continue
                try:
                    sid_int = int(sid)
                except Exception:
                    continue
                if sid_int <= 0:
                    continue
                key = f"{cam}:{sid_int}" if cam else str(sid_int)
                seen[key] = seen.get(key, 0) + 1
                non_null.add(key)
        return {"seen": seen, "non_null": non_null}


def _spawn_runtime(args: argparse.Namespace) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "noesis/ds8_runtime.py",
        "--pipeline-config",
        args.pipeline_config,
        "--cameras-config",
        args.cameras_config,
        "--disable-rest",
    ]
    if args.depth_seconds is not None:
        cmd.extend(["--depth-enable-seconds", str(args.depth_seconds)])
    env = os.environ.copy()
    env.setdefault("NOESIS_REID_ENABLED", "1")
    env.setdefault("NOESIS_MOSAIC_JPEG_ENABLED", "0")
    env.setdefault("NOESIS_MOSAIC_RTSP_ENABLED", "0")
    env.setdefault("NOESIS_MOSAIC_WEBRTC_ENABLED", "0")
    if args.synthetic:
        env.setdefault("NOESIS_REID_TEST_MODE", "1")
    proc = subprocess.Popen(cmd, env=env)
    return proc


def main() -> int:
    parser = argparse.ArgumentParser(description="Stable ID smoke test")
    parser.add_argument("--ws", default="ws://127.0.0.1:6008", help="WebSocket URL")
    parser.add_argument("--pipeline-config", default="config/infer_smoke_reid.yaml")
    parser.add_argument("--cameras-config", default="config/cameras.yaml")
    parser.add_argument("--no-spawn", action="store_true", help="Do not spawn runtime")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Inject a synthetic stable_id track when no detections are present (NOESIS_REID_TEST_MODE=1).",
    )
    parser.add_argument(
        "--depth-seconds",
        type=int,
        default=0,
        help="Enable depth on startup for this many seconds (omit to use runtime default).",
    )
    args = parser.parse_args()

    proc = None
    if not args.no_spawn:
        proc = _spawn_runtime(args)
        time.sleep(5.0)

    try:
        result = None
        last_err = None
        for _ in range(10):
            try:
                result = asyncio.run(_collect_tracking(args.ws, duration=args.duration))
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1.0)
        if result is None:
            print(f"[FAIL] could not collect tracking: {last_err}")
            return 1
        if not result["non_null"]:
            print("[FAIL] no stable_id values observed")
            return 1
        for key, count in result["seen"].items():
            if count >= 2:
                print(f"[PASS] stable_id persisted across frames for {key}")
                return 0
        print("[FAIL] stable_id did not persist across frames")
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
