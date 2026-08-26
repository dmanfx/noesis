"""Resolve the three canonical Noesis camera sources in memory."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from noesis_core.runtime_secrets import load_pipeline_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "DS9" / "config" / "infer.yaml"


def load_three_room_sources(
    pipeline_config: Path = DEFAULT_PIPELINE_CONFIG,
    *,
    camera_registry: Mapping[str, str] | None = None,
) -> dict[str, object]:
    config = load_pipeline_config(pipeline_config, camera_registry=camera_registry)
    raw_sources = config.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != 3:
        raise RuntimeError(f"Expected exactly three canonical sources in {pipeline_config}")
    streammux = config.get("streammux")
    streammux = dict(streammux) if isinstance(streammux, dict) else {}
    uris: list[str] = []
    sensor_ids: list[str] = []
    sensor_names: list[str] = []
    for index, source in enumerate(raw_sources):
        if not isinstance(source, dict):
            raise RuntimeError(f"Canonical source {index} must be a mapping")
        uri = str(source.get("uri") or "").strip()
        if not uri.lower().startswith(("rtsp://", "rtsps://")):
            raise RuntimeError(f"Canonical source {index} did not resolve to RTSP")
        secret_ref = str(source.get("uri_secret") or f"room-{index}")
        uris.append(uri)
        sensor_ids.append(str(index))
        sensor_names.append(secret_ref.replace("-", " ").title())
    return {
        "uris": uris,
        "sensor_ids": sensor_ids,
        "sensor_names": sensor_names,
        "width": int(streammux.get("width", 1920) or 1920),
        "height": int(streammux.get("height", 1080) or 1080),
    }
