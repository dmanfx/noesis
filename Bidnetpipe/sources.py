"""Load camera sources from config.py and emit a DS8-friendly YAML list."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

DEFAULT_SOURCES_PATH = Path("config/bisenetpipe_sources.yaml")


@dataclass
class SourceSpec:
    uri: str
    sensor_id: str
    sensor_name: str


def _load_config_rtsp_streams() -> List[Dict[str, Any]]:
    try:
        import config as app_config  # local config.py
    except Exception as exc:  # pragma: no cover - import failure should be explicit
        raise RuntimeError(f"Failed to import config.py: {exc}") from exc

    cfg = getattr(app_config, "config", None)
    if cfg is None:
        cfg = app_config.AppConfig()
    streams = getattr(cfg.cameras, "RTSP_STREAMS", None)
    if not streams:
        raise RuntimeError("config.py has no RTSP_STREAMS configured")
    return list(streams)


def extract_enabled_sources(required_count: int = 3) -> List[SourceSpec]:
    streams = _load_config_rtsp_streams()
    enabled = [stream for stream in streams if stream.get("enabled", True)]
    if len(enabled) < required_count:
        raise RuntimeError(
            f"Need {required_count} enabled RTSP streams, found {len(enabled)}"
        )
    selected = enabled[:required_count]
    sources: List[SourceSpec] = []
    for idx, stream in enumerate(selected):
        uri = stream.get("url")
        if not uri:
            raise RuntimeError(f"RTSP stream entry {idx} missing url")
        name = stream.get("name") or f"source_{idx}"
        sources.append(SourceSpec(uri=uri, sensor_id=str(idx), sensor_name=name))
    return sources


def generate_sources_yaml(
    output_path: Path = DEFAULT_SOURCES_PATH,
    required_count: int = 3,
    force: bool = False,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sources = extract_enabled_sources(required_count=required_count)

    payload = {
        "uris": [s.uri for s in sources],
        "sensor_ids": [s.sensor_id for s in sources],
        "sensor_names": [s.sensor_name for s in sources],
    }

    if output_path.exists() and not force:
        existing = yaml.safe_load(output_path.read_text())
        if existing == payload:
            return payload

    output_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return payload


def load_sources_yaml(path: Path = DEFAULT_SOURCES_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Sources YAML not found: {path}")
    return yaml.safe_load(path.read_text())
