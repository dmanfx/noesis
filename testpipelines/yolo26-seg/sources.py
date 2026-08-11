"""Source selection helpers for YOLO26 segmentation test pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from noesis_core.runtime_secrets import materialize_pipeline_config

DEFAULT_SOURCES_PATH = Path(__file__).resolve().parent / "sources.yaml"
_CAMERA_SECRET_PREFIX = "camera-secret:"


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def load_sources_yaml(path: Path = DEFAULT_SOURCES_PATH) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Sources YAML not found: {path}")
    payload = yaml.safe_load(path.read_text()) or {}
    return payload


def _resolve_uris(uris: List[str]) -> List[str]:
    sources = []
    for value in uris:
        uri = str(value or "").strip()
        if uri.startswith(_CAMERA_SECRET_PREFIX):
            sources.append({"uri_secret": uri.removeprefix(_CAMERA_SECRET_PREFIX)})
        else:
            sources.append({"uri": uri})
    materialized = materialize_pipeline_config({"sources": sources})
    return [str(source["uri"]) for source in materialized["sources"]]


def select_sources(
    payload: Dict[str, object],
    *,
    camera: Optional[str] = None,
    required_count: int = 3,
) -> Dict[str, object]:
    uris = _resolve_uris([str(value) for value in (payload.get("uris", []) or [])])
    sensor_ids: List[str] = [str(x) for x in (payload.get("sensor_ids", []) or [])]
    sensor_names: List[str] = [str(x) for x in (payload.get("sensor_names", []) or [])]
    if not uris:
        raise RuntimeError("sources.yaml has no URIs configured")

    width = int(payload.get("width", 1920) or 1920)
    height = int(payload.get("height", 1080) or 1080)

    entries = list(zip(uris, sensor_ids, sensor_names))
    if camera:
        target = _normalize_name(camera)
        match_idx = None
        for idx, entry in enumerate(entries):
            name = entry[2] or f"source_{idx}"
            if _normalize_name(str(name)) == target:
                match_idx = idx
                break
        if match_idx is None:
            available = ", ".join(name for _, _, name in entries)
            raise RuntimeError(f"Camera '{camera}' not found. Available: {available}")
        selected = entries[match_idx]
        remainder = [entry for idx, entry in enumerate(entries) if idx != match_idx]
        entries = [selected] + remainder

    replicated = False
    if len(entries) < required_count:
        last = entries[-1]
        replicated = True
        while len(entries) < required_count:
            entries.append(last)
    else:
        entries = entries[:required_count]

    new_uris = [entry[0] for entry in entries]
    new_ids = [str(i) for i in range(len(entries))]
    new_names: List[str] = []
    for i, entry in enumerate(entries):
        name = entry[2] or f"source_{i}"
        if replicated and len(entries) > 1:
            name = f"{name} #{i + 1}"
        new_names.append(str(name))

    return {
        "uris": new_uris,
        "sensor_ids": new_ids,
        "sensor_names": new_names,
        "width": width,
        "height": height,
    }
