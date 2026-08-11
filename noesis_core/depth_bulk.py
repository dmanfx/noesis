"""Runtime-neutral resource limits for the dense-depth bulk wire contract."""

from __future__ import annotations

import json
from pathlib import Path

_LIMITS_PATH = Path(__file__).with_name("depth_bulk_limits.json")
_LIMITS = json.loads(_LIMITS_PATH.read_text(encoding="ascii"))

DEPTH_BULK_MAX_PIXELS = int(_LIMITS["max_pixels"])
DEPTH_BULK_MAX_COMPONENT_BYTES = int(_LIMITS["max_component_bytes"])
DEPTH_BULK_MAX_SNAPSHOT_BYTES = int(_LIMITS["max_snapshot_bytes"])

if not (
    0 < DEPTH_BULK_MAX_PIXELS <= (1 << 53) - 1
    and 0 < DEPTH_BULK_MAX_COMPONENT_BYTES <= (1 << 53) - 1
    and DEPTH_BULK_MAX_COMPONENT_BYTES
    <= DEPTH_BULK_MAX_SNAPSHOT_BYTES
    <= (1 << 53) - 1
):
    raise RuntimeError("dense-depth bulk limits are invalid")

__all__ = [
    "DEPTH_BULK_MAX_COMPONENT_BYTES",
    "DEPTH_BULK_MAX_PIXELS",
    "DEPTH_BULK_MAX_SNAPSHOT_BYTES",
]
