from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping


def _indicator(label: str, value: Any, status: str, detail: str = "") -> Dict[str, Any]:
    return {"label": label, "value": value, "status": status, "detail": detail}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def summarize_live_health(probe: Mapping[str, Any]) -> Dict[str, Any]:
    if not probe.get("connected"):
        return {
            "status": "down",
            "score": 0,
            "summary": probe.get("error") or "Live WebSocket is not connected.",
            "sampled_at": time.time(),
            "url": probe.get("url"),
            "indicators": [_indicator("WebSocket", "down", "block", str(probe.get("error") or ""))],
            "cameras": [],
            "message_types": probe.get("message_types", {}),
            "raw": dict(probe),
        }

    stats = probe.get("stats") if isinstance(probe.get("stats"), Mapping) else {}
    app = stats.get("application") if isinstance(stats.get("application"), Mapping) else {}
    pipe = stats.get("pipeline") if isinstance(stats.get("pipeline"), Mapping) else {}
    camera_ids = [str(item) for item in stats.get("camera_ids", [])] if isinstance(stats.get("camera_ids"), list) else []
    active_cameras = app.get("cameras_active")
    camera_count = stats.get("camera_count")
    score = 100
    indicators: List[Dict[str, Any]] = []

    pong = bool(probe.get("pong"))
    indicators.append(_indicator("WebSocket", "pong" if pong else "connected", "ok" if pong else "warn", str(probe.get("url") or "")))
    if not pong:
        score -= 10

    running = _truthy(app.get("running"))
    indicators.append(_indicator("Application", "running" if running else "not running", "ok" if running else "block"))
    if not running:
        score -= 30

    activated = _truthy(pipe.get("activated"))
    prepared = _truthy(pipe.get("prepared"))
    indicators.append(_indicator("Pipeline", "activated" if activated else "not activated", "ok" if activated else "block", "prepared" if prepared else "not prepared"))
    if not activated:
        score -= 30
    elif not prepared:
        score -= 10

    cameras_ok = isinstance(active_cameras, int) and active_cameras > 0
    indicators.append(_indicator("Cameras", active_cameras if active_cameras is not None else "-", "ok" if cameras_ok else "warn", f"{camera_count or len(camera_ids)} discovered"))
    if not cameras_ok:
        score -= 20

    zero_copy_profile = pipe.get("zero_copy_profile") or "-"
    zero_copy_violations = int(pipe.get("zero_copy_violations") or 0)
    zc_status = "ok" if zero_copy_violations == 0 else "warn"
    indicators.append(_indicator("Zero copy", zero_copy_profile, zc_status, f"{zero_copy_violations} violation(s)"))
    if zero_copy_violations:
        score -= 20

    stableid_mode = pipe.get("stableid_backend_mode") or "-"
    stableid_status = "ok" if stableid_mode == "gpu" else "warn"
    indicators.append(_indicator("StableID", stableid_mode, stableid_status, f"gallery {pipe.get('stableid_gallery_size', '-')}"))
    if stableid_status != "ok":
        score -= 10

    depth_enabled = _truthy(pipe.get("depth_enabled"))
    depth_fps = float(pipe.get("depth_fps") or 0.0)
    depth_status = "ok" if not depth_enabled or depth_fps > 0 else "warn"
    indicators.append(_indicator("Depth gate", "on" if depth_enabled else "off", depth_status, f"{depth_fps:.2f} fps"))
    if depth_enabled and depth_fps <= 0:
        score -= 10

    score = max(0, min(100, score))
    status = "healthy" if score >= 85 else "attention" if score >= 60 else "down"
    if status == "healthy":
        summary = f"Live DS8 is activated with {active_cameras or 0} active camera(s)."
    elif status == "attention":
        summary = f"Live DS8 is reachable but {sum(1 for item in indicators if item['status'] == 'warn')} indicator(s) need attention."
    else:
        summary = "Live DS8 is reachable but not healthy enough for operator confidence."

    return {
        "status": status,
        "score": score,
        "summary": summary,
        "sampled_at": time.time(),
        "url": probe.get("url"),
        "uptime_s": stats.get("uptime"),
        "stack": stats.get("stack"),
        "trail_enabled": probe.get("trail_enabled"),
        "indicators": indicators,
        "cameras": [{"id": camera_id, "status": "active"} for camera_id in camera_ids],
        "message_types": probe.get("message_types", {}),
        "raw": dict(probe),
    }
