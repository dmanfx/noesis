"""Source selection helpers for the DEIMv2 Wholebody49 prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES_PATH = Path(__file__).resolve().parent / "sources.yaml"
DEFAULT_INFER_CONFIG_PATH = REPO_ROOT / "config" / "infer.yaml"
DEFAULT_SENSOR_NAMES = ("Living Room Camera", "Kitchen Camera", "Family Room Camera")


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _default_sensor_name(idx: int, uri: str) -> str:
    if idx < len(DEFAULT_SENSOR_NAMES):
        return DEFAULT_SENSOR_NAMES[idx]
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        stem = Path(parsed.path).stem
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

    uris: List[str] = []
    sensor_ids: List[str] = []
    sensor_names: List[str] = []
    for idx, source in enumerate(list(payload.get("sources", []) or [])):
        if not isinstance(source, dict):
            continue
        uri = str(source.get("uri", "") or "").strip()
        if not uri:
            continue
        uris.append(uri)
        sensor_ids.append(str(source.get("sensor-id", idx)))
        sensor_names.append(str(source.get("name") or _default_sensor_name(idx, uri)))

    if len(uris) < 3:
        raise RuntimeError(f"Expected at least 3 sources in {infer_config_path}, found {len(uris)}")

    generated = {
        "width": int(payload.get("streammux", {}).get("width", 1920) if isinstance(payload.get("streammux"), dict) else 1920),
        "height": int(payload.get("streammux", {}).get("height", 1080) if isinstance(payload.get("streammux"), dict) else 1080),
        "uris": uris[:3],
        "sensor_ids": sensor_ids[:3],
        "sensor_names": sensor_names[:3],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    required_count: int = 3,
    require_rtsp: bool = True,
) -> Dict[str, object]:
    uris = [str(x) for x in (payload.get("uris", []) or [])]
    sensor_ids = [str(x) for x in (payload.get("sensor_ids", []) or [])]
    sensor_names = [str(x) for x in (payload.get("sensor_names", []) or [])]
    if not uris:
        raise RuntimeError("sources payload has no URIs configured")
    if len(sensor_ids) < len(uris):
        sensor_ids.extend(str(idx) for idx in range(len(sensor_ids), len(uris)))
    if len(sensor_names) < len(uris):
        sensor_names.extend(_default_sensor_name(idx, uri) for idx, uri in enumerate(uris[len(sensor_names):], start=len(sensor_names)))

    entries = list(zip(uris, sensor_ids, sensor_names))
    if camera:
        target = _normalize_name(camera)
        match_idx = None
        for idx, (_, _, name) in enumerate(entries):
            if _normalize_name(name) == target:
                match_idx = idx
                break
        if match_idx is None:
            available = ", ".join(name for _, _, name in entries)
            raise RuntimeError(f"Camera '{camera}' not found. Available: {available}")
        selected = entries[match_idx]
        entries = [selected] + [entry for idx, entry in enumerate(entries) if idx != match_idx]

    entries = entries[:required_count]
    if len(entries) != required_count:
        raise RuntimeError(f"Expected {required_count} sources, got {len(entries)}")
    if require_rtsp:
        non_rtsp = [uri for uri, _, _ in entries if not uri.lower().startswith("rtsp://")]
        if non_rtsp:
            raise RuntimeError(f"DEIMv2 prototype requires RTSP sources; non-RTSP entries: {non_rtsp}")

    return {
        "uris": [entry[0] for entry in entries],
        "sensor_ids": [entry[1] for entry in entries],
        "sensor_names": [entry[2] for entry in entries],
        "width": int(payload.get("width", 1920) or 1920),
        "height": int(payload.get("height", 1080) or 1080),
    }
