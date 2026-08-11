"""DS9 metadata adapters for the canonical observation/world contracts."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Mapping, Optional


def _meta_value(frame_meta: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(frame_meta, Mapping) and name in frame_meta:
            value = frame_meta.get(name)
        else:
            value = getattr(frame_meta, name, None)
        if value is not None:
            return value
    return default


def frame_temporal_contract(
    frame_meta: Any,
    *,
    observed_at_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Return truthful processing-time and media-PTS evidence for one frame.

    DeepStream buffer PTS is stream-relative on the current DS9 graph, so it is
    never presented as an epoch capture timestamp. The wall-clock observation
    time is explicitly marked estimated until camera clock synchronization is
    available.
    """

    try:
        observed_s = float(observed_at_s if observed_at_s is not None else time.time())
    except Exception:
        observed_s = time.time()
    if not math.isfinite(observed_s) or observed_s <= 0.0:
        observed_s = time.time()
    observed_at_us = max(1, int(observed_s * 1_000_000.0))

    raw_pts = _meta_value(frame_meta, "buf_pts", "buffer_pts", "pts", default=0)
    try:
        media_pts_ns = int(raw_pts or 0)
    except Exception:
        media_pts_ns = 0
    media_pts_ns = max(0, media_pts_ns)
    return {
        "captured_at_us": observed_at_us,
        "observed_at_us": observed_at_us,
        "capture_time_status": "estimated",
        "media_pts_ns": media_pts_ns,
    }


def stable_identity_contract(
    manager: Any,
    *,
    sensor_id: int,
    tracker_id: int,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize shared stable-ID diagnostics for canonical world subjects."""

    diag: Mapping[str, Any] = diagnostics if isinstance(diagnostics, Mapping) else {}
    if not diag and manager is not None:
        getter = getattr(manager, "get_track_diagnostics", None)
        if callable(getter):
            try:
                candidate = getter(int(sensor_id), int(tracker_id))
                if isinstance(candidate, Mapping):
                    diag = candidate
            except Exception:
                diag = {}

    result: Dict[str, Any] = {}
    for key in ("identity_kind", "identity_state", "resident_uuid", "display_name"):
        raw = diag.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            result[key] = text

    generation = diag.get("visitor_generation")
    if generation is not None:
        try:
            parsed_generation = int(generation)
        except Exception:
            parsed_generation = -1
        if parsed_generation >= 0:
            result["visitor_generation"] = parsed_generation
    return result


__all__ = ["frame_temporal_contract", "stable_identity_contract"]
