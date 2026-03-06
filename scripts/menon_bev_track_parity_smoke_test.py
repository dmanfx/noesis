#!/usr/bin/env python3
"""Smoke test: BEV/track parity in menon_scene world frame.

Runs DS8 runtime, listens to WS telemetry, and validates:
- tracking world points are valid and labeled menon_scene
- bev-frame metadata is in world mode / menon_scene
- BEV footpoints (x,y -> world x,z) are close to tracking world coordinates
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

try:
    import websockets  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"[FAIL] websockets package required: {exc}")
    sys.exit(1)


def _spawn_runtime(args: argparse.Namespace) -> subprocess.Popen:
    ws_url = urlparse(str(args.ws))
    ws_port = ws_url.port or 6040
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


async def _recv_json(ws, *, timeout_s: float) -> Optional[Dict[str, object]]:
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


async def _run(uri: str, duration_s: float) -> Dict[str, object]:
    track_total = 0
    track_world_valid = 0
    track_world_frame_ok = 0
    bev_total = 0
    bev_world_frame_ok = 0
    comparisons: List[float] = []
    last_bev_sid: Dict[Tuple[str, int], Tuple[float, float]] = {}
    last_bev_tracker: Dict[Tuple[str, int], Tuple[float, float]] = {}

    async with websockets.connect(uri, max_size=None) as ws:
        end_at = time.time() + max(5.0, float(duration_s))
        while time.time() < end_at:
            payload = await _recv_json(ws, timeout_s=2.0)
            if not payload:
                continue
            msg_type = payload.get("type")
            if msg_type == "bev-frame":
                bev_total += 1
                if payload.get("world_frame") == "menon_scene" and payload.get("frame_mode") == "world":
                    bev_world_frame_ok += 1
                camera_id = payload.get("cameraId")
                if not isinstance(camera_id, str):
                    continue
                for fp in payload.get("footpoints") or []:
                    if not isinstance(fp, dict):
                        continue
                    try:
                        x = float(fp.get("x"))
                        y = float(fp.get("y"))
                        if not (math.isfinite(x) and math.isfinite(y)):
                            continue
                    except Exception:
                        continue
                    sid = fp.get("stableId")
                    if sid not in (None, "", -1):
                        try:
                            key_sid = (camera_id, int(sid))
                            last_bev_sid[key_sid] = (x, y)
                        except Exception:
                            pass
                    tid = fp.get("trackerId")
                    if tid not in (None, "", -1):
                        try:
                            key_tid = (camera_id, int(tid))
                            last_bev_tracker[key_tid] = (x, y)
                        except Exception:
                            pass
            elif msg_type == "tracking":
                tracks = payload.get("tracks")
                if not isinstance(tracks, list):
                    continue
                for tr in tracks:
                    if not isinstance(tr, dict):
                        continue
                    track_total += 1
                    if tr.get("world_valid") is not True:
                        continue
                    world = tr.get("world")
                    if not (isinstance(world, list) and len(world) == 3):
                        continue
                    track_world_valid += 1
                    if tr.get("world_frame") == "menon_scene":
                        track_world_frame_ok += 1
                    camera_id = tr.get("camera_id")
                    sid = tr.get("stable_id")
                    tid = tr.get("tracker_id", tr.get("track_id"))
                    if not isinstance(camera_id, str):
                        continue
                    bev_pt = None
                    if tid not in (None, "", -1):
                        try:
                            bev_pt = last_bev_tracker.get((camera_id, int(tid)))
                        except Exception:
                            bev_pt = None
                    if bev_pt is None and sid not in (None, "", -1):
                        try:
                            bev_pt = last_bev_sid.get((camera_id, int(sid)))
                        except Exception:
                            bev_pt = None
                    if bev_pt is None:
                        continue
                    bx, bz = bev_pt
                    wx = float(world[0])
                    wz = float(world[2])
                    comparisons.append(math.hypot(bx - wx, bz - wz))

    result: Dict[str, object] = {
        "track_total": track_total,
        "track_world_valid": track_world_valid,
        "track_world_frame_menon_scene": track_world_frame_ok,
        "bev_total": bev_total,
        "bev_world_frame_menon_scene": bev_world_frame_ok,
        "comparisons": len(comparisons),
    }
    if comparisons:
        vals = sorted(comparisons)
        result["mean_err_m"] = float(sum(vals) / len(vals))
        result["p95_err_m"] = float(vals[int(0.95 * (len(vals) - 1))])
    else:
        result["mean_err_m"] = None
        result["p95_err_m"] = None
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="BEV/track parity smoke test for menon_scene.")
    parser.add_argument("--ws", default="ws://127.0.0.1:6040", help="WebSocket URL")
    parser.add_argument("--pipeline-config", default="config/infer.yaml")
    parser.add_argument("--cameras-config", default="config/cameras.yaml")
    parser.add_argument("--duration", type=float, default=20.0, help="Collection window seconds")
    parser.add_argument(
        "--p95-threshold-m",
        type=float,
        default=15.0,
        help="Max accepted p95 error in scene units (legacy flag name kept for compatibility)",
    )
    parser.add_argument("--no-spawn", action="store_true", help="Do not spawn ds8_runtime")
    args = parser.parse_args()

    proc: Optional[subprocess.Popen] = None
    if not args.no_spawn:
        proc = _spawn_runtime(args)
        time.sleep(8.0)

    try:
        summary = asyncio.run(_run(str(args.ws), float(args.duration)))
        print(json.dumps(summary, indent=2))

        if int(summary.get("track_world_valid", 0)) <= 0:
            print("[FAIL] no world-valid tracking samples observed")
            return 1
        if int(summary.get("track_world_frame_menon_scene", 0)) <= 0:
            print("[FAIL] no tracking samples labeled world_frame=menon_scene")
            return 1
        if int(summary.get("bev_world_frame_menon_scene", 0)) <= 0:
            print("[FAIL] no BEV samples labeled world_frame=menon_scene/frame_mode=world")
            return 1
        if int(summary.get("comparisons", 0)) <= 0:
            print("[FAIL] no comparable BEV/track samples (stableId + camera overlap)")
            return 1

        p95 = summary.get("p95_err_m")
        if not isinstance(p95, (int, float)):
            print("[FAIL] invalid p95 error metric")
            return 1
        if float(p95) > float(args.p95_threshold_m):
            print(
                f"[FAIL] BEV/track parity p95 error too high (scene units): "
                f"{p95:.6f} > {float(args.p95_threshold_m):.6f}"
            )
            return 1

        print(
            "[PASS] BEV/track parity validated "
            f"(p95_err_scene={float(p95):.6f}, threshold={float(args.p95_threshold_m):.6f})"
        )
        return 0
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1
    finally:
        if proc is not None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
