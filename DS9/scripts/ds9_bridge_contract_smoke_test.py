#!/usr/bin/env python3
"""Focused DS9 native bridge contract smoke.

Run against an already-running DS9 runtime. This checks bridge-level evidence
instead of only end-to-end track presence:

- depth tensor native capture produced aligned device frames
- object-depth bridge copied object ROIs and attached NOESIS.OBJECT_DEPTH
- ReID native extraction produced host embeddings for StableID
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any, Dict

try:
    import websockets  # type: ignore
except Exception as exc:  # pragma: no cover
    print(json.dumps({"ok": False, "error": f"websockets_unavailable:{exc}"}))
    raise SystemExit(1)


def _counter(counters: Dict[str, Any], key: str) -> int:
    try:
        return int(counters.get(key, 0) or 0)
    except Exception:
        return 0


async def _collect(args: argparse.Namespace) -> Dict[str, Any]:
    max_counters: Dict[str, int] = {}
    stats_samples = 0
    tracking_messages = 0
    tracks_seen = 0
    depth_ok_tracks = 0
    embedding_tracks = 0
    last_errors: list[str] = []

    async with websockets.connect(args.ws, max_size=None) as ws:
        if not args.no_clear:
            try:
                await ws.send(json.dumps({"type": "clear_stats"}, separators=(",", ":")))
            except Exception:
                pass

        deadline = time.monotonic() + max(1.0, float(args.duration))
        while time.monotonic() < deadline:
            timeout_s = min(2.0, max(0.05, deadline - time.monotonic()))
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
            except asyncio.TimeoutError:
                continue
            if not isinstance(raw, str):
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            msg_type = str(payload.get("type") or "")
            if msg_type == "stats":
                stats_samples += 1
                stats = payload.get("payload") or {}
                pipe = stats.get("pipeline") if isinstance(stats, dict) else {}
                if isinstance(pipe, dict):
                    errors = pipe.get("errors") or []
                    if isinstance(errors, list):
                        last_errors = [str(x) for x in errors[-8:]]
                    zero = pipe.get("zero_copy_core") or {}
                    counters = zero.get("counters") if isinstance(zero, dict) else {}
                    if isinstance(counters, dict):
                        for key, value in counters.items():
                            try:
                                value_i = int(value or 0)
                            except Exception:
                                continue
                            current = int(max_counters.get(str(key), 0))
                            if value_i > current:
                                max_counters[str(key)] = value_i
                continue

            if msg_type == "tracking":
                tracking_messages += 1
                tracks = payload.get("tracks") or []
                if not isinstance(tracks, list):
                    continue
                tracks_seen += len(tracks)
                for track in tracks:
                    if not isinstance(track, dict):
                        continue
                    try:
                        sample_count = int(track.get("depth_sample_count") or 0)
                    except Exception:
                        sample_count = 0
                    if str(track.get("depth_status") or "") == "ok" and sample_count > 0:
                        depth_ok_tracks += 1
                    if track.get("embedding_present") is True:
                        embedding_tracks += 1

    depth_tensor_frames = _counter(max_counters, "depth_tracking_device_frames_total")
    object_depth_roi_copies = _counter(max_counters, "object_depth_gpu_roi_copies_total")
    object_depth_attaches = _counter(max_counters, "object_depth_attach_total")
    object_depth_ok = _counter(max_counters, "object_depth_status_total.ok")
    reid_host_copies = _counter(max_counters, "tensor_host_copies_total.reid")
    zero_copy_violations = _counter(max_counters, "core_path.cpu_copy_violation.total")

    checks = {
        "depth_tensor_frames": depth_tensor_frames >= int(args.min_depth_tensor_frames),
        "object_depth_roi_copies": object_depth_roi_copies >= int(args.min_object_depth_roi_copies),
        "object_depth_attaches": object_depth_attaches >= int(args.min_object_depth_attaches),
        "object_depth_ok": object_depth_ok >= int(args.min_object_depth_ok) or depth_ok_tracks >= int(args.min_object_depth_ok),
        "reid_native_extractions": reid_host_copies >= int(args.min_reid_extractions),
        "zero_copy_violations": zero_copy_violations <= int(args.max_zero_copy_violations),
        "stats_samples": stats_samples > 0,
    }
    if args.require_embedding_track:
        checks["embedding_tracks"] = embedding_tracks > 0

    ok = all(bool(v) for v in checks.values())
    return {
        "ok": ok,
        "ws": args.ws,
        "duration_s": float(args.duration),
        "checks": checks,
        "stats_samples": stats_samples,
        "tracking_messages": tracking_messages,
        "tracks_seen": tracks_seen,
        "depth_ok_tracks": depth_ok_tracks,
        "embedding_tracks": embedding_tracks,
        "max_counters": {
            "depth_tracking_device_frames_total": depth_tensor_frames,
            "object_depth_gpu_roi_copies_total": object_depth_roi_copies,
            "object_depth_attach_total": object_depth_attaches,
            "object_depth_status_total.ok": object_depth_ok,
            "tensor_host_copies_total.reid": reid_host_copies,
            "core_path.cpu_copy_violation.total": zero_copy_violations,
        },
        "pipeline_errors": last_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DS9 object-depth/depth-tensor/ReID native bridge smoke")
    parser.add_argument("--ws", default="ws://127.0.0.1:6008")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--no-clear", action="store_true", help="Do not send clear_stats before collection.")
    parser.add_argument("--min-depth-tensor-frames", type=int, default=1)
    parser.add_argument("--min-object-depth-roi-copies", type=int, default=1)
    parser.add_argument("--min-object-depth-attaches", type=int, default=1)
    parser.add_argument("--min-object-depth-ok", type=int, default=1)
    parser.add_argument("--min-reid-extractions", type=int, default=1)
    parser.add_argument("--max-zero-copy-violations", type=int, default=0)
    parser.add_argument(
        "--require-embedding-track",
        action="store_true",
        help="Also require at least one tracking payload with embedding_present=true.",
    )
    args = parser.parse_args()

    result = asyncio.run(_collect(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if bool(result.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
