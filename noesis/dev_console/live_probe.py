from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional


def _stats_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    stats = payload.get("payload") if isinstance(payload.get("payload"), Mapping) else payload
    if not isinstance(stats, Mapping):
        return {}
    application = stats.get("application") if isinstance(stats.get("application"), Mapping) else {}
    pipeline = stats.get("pipeline") if isinstance(stats.get("pipeline"), Mapping) else {}
    cameras = stats.get("cameras") if isinstance(stats.get("cameras"), Mapping) else {}
    return {
        "timestamp": stats.get("timestamp"),
        "uptime": stats.get("uptime"),
        "stack": stats.get("stack"),
        "application": {
            "running": application.get("running"),
            "cameras_active": application.get("cameras_active"),
            "processors_active": application.get("processors_active"),
        },
        "pipeline": {
            "prepared": pipeline.get("prepared"),
            "activated": pipeline.get("activated"),
            "depth_enabled": pipeline.get("depth_enabled"),
            "depth_fps": pipeline.get("depth_fps"),
            "zero_copy_profile": pipeline.get("zero_copy_profile"),
            "zero_copy_violations": pipeline.get("zero_copy_violations"),
            "stableid_backend_mode": pipeline.get("stableid_backend_mode"),
            "stableid_gallery_size": pipeline.get("stableid_gallery_size"),
            "analytics_reload_count": pipeline.get("analytics_reload_count"),
        },
        "camera_count": len(cameras),
        "camera_ids": sorted(str(key) for key in cameras.keys())[:8],
    }


async def _probe_ws(ws_url: str, *, timeout_s: float = 3.0) -> Dict[str, Any]:
    try:
        import websockets
    except Exception as exc:
        return {"ok": False, "connected": False, "error": f"websockets unavailable: {exc}"}

    messages: List[Dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    stats: Dict[str, Any] = {}
    pong_seen = False
    trail_enabled: Optional[bool] = None
    started = time.time()
    try:
        async with websockets.connect(ws_url, open_timeout=2.0, max_size=8 * 1024 * 1024) as ws:
            await ws.send(json.dumps({"type": "ping", "timestamp": started}, separators=(",", ":")))
            deadline = time.time() + max(0.5, float(timeout_s))
            while time.time() < deadline and len(messages) < 24:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, min(0.75, deadline - time.time())))
                except asyncio.TimeoutError:
                    break
                if not isinstance(raw, str):
                    type_counts["binary"] += 1
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    type_counts["unparsed"] += 1
                    continue
                msg_type = str(payload.get("type", "unknown"))
                type_counts[msg_type] += 1
                if msg_type == "stats" and not stats:
                    stats = _stats_summary(payload)
                if msg_type == "pong":
                    pong_seen = True
                if msg_type == "trail_visualization_enabled_update":
                    trail_enabled = bool(payload.get("enabled"))
                messages.append(
                    {
                        "type": msg_type,
                        "keys": sorted(str(key) for key in payload.keys())[:12],
                    }
                )
                if stats and pong_seen:
                    break
    except Exception as exc:
        return {"ok": False, "connected": False, "error": str(exc), "url": ws_url}

    return {
        "ok": True,
        "connected": True,
        "url": ws_url,
        "pong": pong_seen,
        "trail_enabled": trail_enabled,
        "message_types": dict(sorted(type_counts.items())),
        "messages": messages,
        "stats": stats,
    }


async def _send_control(ws_url: str, message: Mapping[str, Any], *, timeout_s: float = 2.0) -> Dict[str, Any]:
    try:
        import websockets
    except Exception as exc:
        return {"ok": False, "error": f"websockets unavailable: {exc}"}
    try:
        async with websockets.connect(ws_url, open_timeout=2.0, max_size=4 * 1024 * 1024) as ws:
            await ws.send(json.dumps(dict(message), separators=(",", ":")))
            replies: List[Dict[str, Any]] = []
            deadline = time.time() + max(0.5, float(timeout_s))
            while time.time() < deadline and len(replies) < 6:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, min(0.5, deadline - time.time())))
                except asyncio.TimeoutError:
                    break
                if not isinstance(raw, str):
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                replies.append({"type": payload.get("type"), "payload": payload})
                msg_type = payload.get("type")
                if msg_type in {"pong", "toggle_update", "stats"}:
                    break
    except Exception as exc:
        return {"ok": False, "error": str(exc), "url": ws_url}
    return {"ok": True, "url": ws_url, "sent": dict(message), "replies": replies}


def probe_live_ws(ws_host: str, ws_port: int, *, timeout_s: float = 3.0) -> Dict[str, Any]:
    url = f"ws://{ws_host}:{int(ws_port)}"
    return asyncio.run(_probe_ws(url, timeout_s=timeout_s))


def send_live_control(ws_host: str, ws_port: int, *, action: str, enabled: Optional[bool] = None) -> Dict[str, Any]:
    action = str(action or "").strip()
    if action == "ping":
        message = {"type": "ping", "timestamp": time.time()}
    elif action == "clear_stats":
        message = {"type": "clear_stats"}
    elif action == "trail_visualization_enabled":
        message = {
            "type": "set_vis_toggle",
            "toggle_name": "trail_visualization_enabled",
            "enabled": bool(enabled),
        }
    else:
        return {"ok": False, "error": f"unsupported live control action: {action}"}
    return asyncio.run(_send_control(f"ws://{ws_host}:{int(ws_port)}", message))
