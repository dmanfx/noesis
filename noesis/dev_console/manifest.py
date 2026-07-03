from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List

from noesis.ds8_preflight import REPO_ROOT


ENV_PATTERN = re.compile(r"\bNOESIS_[A-Z0-9_]+\b")


SOURCE_PATHS = [
    "noesis/ds8_runtime.py",
    "noesis/pipelines/hooks.py",
    "noesis/pipelines/ds8_pipeline.py",
    "websocket_server.py",
    "config/infer.yaml",
    "docs/DS8_README_FOR_AGENTS.md",
    "docs/DS8_testing_guide.md",
]


CATEGORY_HINTS = {
    "REID": "identity",
    "POSE": "pose",
    "V3DT": "v3dt",
    "BEV": "bev",
    "MOSAIC": "mosaic",
    "WEBRTC": "mosaic",
    "RTSP": "mosaic",
    "DEPTH": "depth",
    "MAPANYTHING": "depth",
    "ANALYTICS": "analytics",
    "CALIBRATION": "calibration",
    "TRACKING": "tracking",
    "TRAIL": "tracking",
    "REST": "runtime",
    "WS": "runtime",
    "CUDA": "runtime",
}


DEPRECATED = {
    "NOESIS_REID_RESER_SID_POOL": "Typo alias retained for old environments; use NOESIS_REID_RESET_SID_POOL.",
}


DESCRIPTIONS = {
    "NOESIS_REID_ENABLED": "Enable StableID/ReID runtime integration.",
    "NOESIS_TRACKING_MODE": "Select baseline or v3dt tracking mode.",
    "NOESIS_PGIE_PROFILE": "Select the primary inference profile overlay.",
    "NOESIS_DEPTH_ENABLE_SECONDS": "Open the gated MapAnything branch on startup for this many seconds.",
    "NOESIS_MOSAIC_WEBRTC_ENABLED": "Enable the RTSP-to-WebRTC mosaic gateway at build time.",
    "NOESIS_DEV_CONSOLE_LAUNCH_DIR": "Console-owned launch artifact directory.",
}


def _category(key: str) -> str:
    for needle, category in CATEGORY_HINTS.items():
        if needle in key:
            return category
    return "general"


def _restart_required(key: str) -> bool:
    hot_runtime = {"DEPTH_ENABLE_SECONDS", "TRAILS", "BEV"}
    if any(item in key for item in hot_runtime):
        return False
    return True


def _scan_keys() -> Dict[str, str]:
    keys: Dict[str, str] = {}
    for rel in SOURCE_PATHS:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in ENV_PATTERN.finditer(text):
            keys.setdefault(match.group(0), rel)
    for key in DESCRIPTIONS:
        keys.setdefault(key, "manifest")
    return keys


def load_knobs() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for key, source in sorted(_scan_keys().items()):
        entry = {
            "key": key,
            "category": _category(key),
            "current": os.environ.get(key),
            "default": os.environ.get(key, ""),
            "restart_required": _restart_required(key),
            "source": source,
            "description": DESCRIPTIONS.get(key, key.replace("NOESIS_", "").replace("_", " ").title()),
            "deprecated": key in DEPRECATED,
            "deprecation_note": DEPRECATED.get(key, ""),
        }
        entries.append(entry)
    return entries
