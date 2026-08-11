"""Shared producer-side rules for usable registered person depth."""

from __future__ import annotations

import math
from typing import Any, Mapping


def usable_registered_depth_m(track: Mapping[str, Any]) -> float | None:
    """Return coherent registered depth using the canonical observation rules."""

    if str(track.get("depth_status") or "").strip().lower() != "ok":
        return None
    if str(track.get("depth_registration_status") or "").strip().lower() != "ok":
        return None
    try:
        registered = float(track.get("depth_registered_m"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(registered) or registered <= 0.0:
        return None
    used_raw = track.get("depth_used_m")
    if used_raw is None:
        return registered
    try:
        used = float(used_raw)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(used)
        or used <= 0.0
        or not math.isclose(
            used,
            registered,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    ):
        return None
    return registered


__all__ = ["usable_registered_depth_m"]
