"""Source selection helpers for the room-layout segmentation utility."""

from __future__ import annotations

from pathlib import PurePosixPath
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES_PATH = Path(__file__).resolve().parent / "sources.yaml"
DEFAULT_INFER_CONFIG_PATH = REPO_ROOT / "config" / "infer.yaml"


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _default_sensor_name(uri: str, idx: int) -> str:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        stem = PurePosixPath(parsed.path).stem
        if stem:
            return stem
    return f"source_{idx}"


def load_sources_yaml(path: Path = DEFAULT_SOURCES_PATH) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Sources YAML not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected mapping in {path}, got {type(payload).__name__}")
    return payload


def generate_sources_yaml_from_infer_config(
    output_path: Path = DEFAULT_SOURCES_PATH,
    infer_config_path: Path = DEFAULT_INFER_CONFIG_PATH,
    *,
    force: bool = False,
) -> Dict[str, object]:
    if not infer_config_path.exists():
        raise FileNotFoundError(f"DS8 infer config not found: {infer_config_path}")
    payload = yaml.safe_load(infer_config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected mapping in {infer_config_path}")

    sources = list(payload.get("sources", []) or [])
    uris: List[str] = []
    sensor_ids: List[str] = []
    sensor_names: List[str] = []
    for idx, source in enumerate(sources):
        if not isinstance(source, dict):
            continue
        uri = str(source.get("uri", "") or "").strip()
        if not uri:
            continue
        sensor_id = str(source.get("sensor-id", idx))
        sensor_name = str(source.get("name") or _default_sensor_name(uri, idx))
        uris.append(uri)
        sensor_ids.append(sensor_id)
        sensor_names.append(sensor_name)

    if not uris:
        raise RuntimeError(f"No usable sources found in {infer_config_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = {
        "width": 1920,
        "height": 1080,
        "uris": uris,
        "sensor_ids": sensor_ids,
        "sensor_names": sensor_names,
    }
    if output_path.exists() and not force:
        existing = yaml.safe_load(output_path.read_text(encoding="utf-8")) or {}
        if existing == generated:
            return generated
    output_path.write_text(yaml.safe_dump(generated, sort_keys=False), encoding="utf-8")
    return generated


def select_sources(
    payload: Dict[str, object],
    *,
    camera: Optional[str] = None,
    max_sources: int = 1,
) -> Dict[str, object]:
    uris: List[str] = [str(x) for x in (payload.get("uris", []) or [])]
    sensor_ids: List[str] = [str(x) for x in (payload.get("sensor_ids", []) or [])]
    sensor_names: List[str] = [str(x) for x in (payload.get("sensor_names", []) or [])]
    if not uris:
        raise RuntimeError("sources payload has no URIs configured")

    width = int(payload.get("width", 1920) or 1920)
    height = int(payload.get("height", 1080) or 1080)
    entries = list(zip(uris, sensor_ids, sensor_names))
    if camera:
        target = _normalize_name(camera)
        matched = [entry for entry in entries if _normalize_name(entry[2]) == target]
        if not matched:
            available = ", ".join(name for _, _, name in entries)
            raise RuntimeError(f"Camera '{camera}' not found. Available: {available}")
        entries = matched
    if max_sources > 0:
        entries = entries[:max_sources]
    if not entries:
        raise RuntimeError("Source selection produced zero entries")

    return {
        "uris": [entry[0] for entry in entries],
        "sensor_ids": [entry[1] for entry in entries],
        "sensor_names": [entry[2] for entry in entries],
        "width": width,
        "height": height,
    }
