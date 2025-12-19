#!/usr/bin/env python3
"""Smoke test for ROI hot-reload via REST + WS stats."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict

import requests

try:
    import websockets  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"[FAIL] websockets package required: {exc}")
    sys.exit(1)


def _spawn_runtime(args: argparse.Namespace) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "noesis/ds8_runtime.py",
        "--pipeline-config",
        args.pipeline_config,
        "--cameras-config",
        args.cameras_config,
        "--enable-rest",
        "--depth-enable-seconds",
        "0",
    ]
    env = os.environ.copy()
    env.setdefault("NOESIS_MOSAIC_JPEG_ENABLED", "0")
    env.setdefault("NOESIS_MOSAIC_RTSP_ENABLED", "0")
    env.setdefault("NOESIS_MOSAIC_WEBRTC_ENABLED", "0")
    proc = subprocess.Popen(cmd, env=env)
    return proc


async def _read_stats(uri: str, max_wait: float = 10.0) -> Dict[str, Any]:
    async with websockets.connect(uri) as ws:
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            try:
                payload = json.loads(msg)
            except Exception:
                continue
            if isinstance(payload, dict) and payload.get("type") == "stats":
                return payload.get("payload") or {}
        raise TimeoutError("No stats payload received")


def _bump_roi(cfg: Dict[str, Any]) -> Dict[str, Any]:
    streams = (cfg.get("streams") or [])[:]
    if not streams:
        return cfg
    first = streams[0]
    rois = first.get("rois") or []
    if not rois:
        rois = [{"id": "smoke", "points_px": [[10.0, 10.0], [100.0, 10.0], [100.0, 100.0]]}]
    else:
        roi = rois[0]
        pts = roi.get("points_px") or []
        if pts:
            pts = [[p[0] + 1.0, p[1] + 1.0] for p in pts]
            roi["points_px"] = pts
        rois[0] = roi
    first["rois"] = rois
    streams[0] = first
    cfg["streams"] = streams
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="ROI reload smoke test")
    parser.add_argument("--ws", default="ws://127.0.0.1:6008", help="WebSocket URL")
    parser.add_argument("--rest", default="http://127.0.0.1:8080", help="REST base URL")
    parser.add_argument("--pipeline-config", default="config/infer.yaml")
    parser.add_argument("--cameras-config", default="config/cameras.yaml")
    parser.add_argument("--no-spawn", action="store_true", help="Do not spawn runtime")
    args = parser.parse_args()

    proc = None
    if not args.no_spawn:
        proc = _spawn_runtime(args)
        time.sleep(5.0)

    try:
        stats_before = None
        for _ in range(10):
            try:
                stats_before = asyncio.run(_read_stats(args.ws))
                break
            except Exception:
                time.sleep(1.0)
        if stats_before is None:
            print("[FAIL] unable to read initial stats from WS")
            return 1
        reload_before = int(
            ((stats_before.get("pipeline") or {}).get("analytics_reload_count")) or 0
        )
        resp = None
        for _ in range(5):
            try:
                resp = requests.get(f"{args.rest}/api/v1/analytics/rois", params={"stage": "exclude"}, timeout=5.0)
                resp.raise_for_status()
                break
            except Exception:
                time.sleep(1.0)
        if resp is None:
            print("[FAIL] unable to fetch initial ROIs")
            return 1
        roi_cfg = resp.json()
        roi_cfg = _bump_roi(roi_cfg)
        post_payload = {"stage": roi_cfg.get("stage", "exclude"), "streams": roi_cfg.get("streams", [])}
        post_resp = None
        for _ in range(5):
            try:
                post_resp = requests.post(f"{args.rest}/api/v1/analytics/rois", json=post_payload, timeout=5.0)
                post_resp.raise_for_status()
                break
            except Exception:
                time.sleep(1.0)
        if post_resp is None:
            print("[FAIL] ROI POST failed")
            return 1
        stats_after = None
        for _ in range(10):
            try:
                stats_after = asyncio.run(_read_stats(args.ws))
                break
            except Exception:
                time.sleep(1.0)
        if stats_after is None:
            print("[FAIL] unable to read stats after ROI POST")
            return 1
        reload_after = int(
            ((stats_after.get("pipeline") or {}).get("analytics_reload_count")) or 0
        )
        if reload_after <= reload_before:
            print(f"[FAIL] reload counter did not increment (before={reload_before}, after={reload_after})")
            return 1
        print(f"[PASS] ROI reload counter incremented to {reload_after}")
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
