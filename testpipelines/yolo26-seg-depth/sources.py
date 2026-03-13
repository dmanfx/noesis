"""Single-source selection helpers for the seg+depth DS8 prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES_PATH = Path(__file__).resolve().parent / "sources.yaml"
DEFAULT_INFER_CONFIG_PATH = REPO_ROOT / "config" / "infer.yaml"
DEFAULT_CAMERA = "Family Room Camera"


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def load_sources_yaml(path: Path = DEFAULT_SOURCES_PATH) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Sources YAML not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected mapping in sources YAML: {path}")
    return payload


def _load_infer_source_candidates(path: Optional[Path]) -> List[Dict[str, Optional[str]]]:
    if path is None or not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        return []
    sources = list(payload.get("sources", []) or [])
    candidates: List[Dict[str, Optional[str]]] = []
    for src in sources:
        uri = ""
        if isinstance(src, dict):
            uri = str(src.get("uri", "") or "").strip()
        candidates.append(
            {
                "active_uri": uri or None,
                "stream_uri": None,
                "file_uri": None,
            }
        )

    current_index = -1
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("- element:"):
            current_index += 1
            continue
        if current_index < 0 or current_index >= len(candidates):
            continue
        if not (stripped.startswith("uri:") or stripped.startswith("#uri:")):
            continue
        _, raw_uri = stripped.split(":", 1)
        uri = str(raw_uri or "").strip()
        if not uri:
            continue
        uri_l = uri.lower()
        if uri_l.startswith("rtsp://"):
            candidates[current_index]["stream_uri"] = uri
        elif uri_l.startswith("file://"):
            candidates[current_index]["file_uri"] = uri

    for candidate in candidates:
        active_uri = candidate.get("active_uri") or ""
        active_uri_l = active_uri.lower()
        if active_uri_l.startswith("rtsp://") and not candidate.get("stream_uri"):
            candidate["stream_uri"] = active_uri
        if active_uri_l.startswith("file://") and not candidate.get("file_uri"):
            candidate["file_uri"] = active_uri
    return candidates


def _resolve_uri_for_mode(
    *,
    idx: int,
    source_mode: str,
    infer_candidates: List[Dict[str, Optional[str]]],
    yaml_uris: List[str],
) -> str:
    infer_entry = infer_candidates[idx] if idx < len(infer_candidates) else {}
    active_uri = str(infer_entry.get("active_uri") or "").strip()
    stream_uri = str(infer_entry.get("stream_uri") or "").strip()
    file_uri = str(infer_entry.get("file_uri") or "").strip()
    yaml_uri = str(yaml_uris[idx] or "").strip() if idx < len(yaml_uris) else ""

    if source_mode == "active":
        resolved = active_uri or yaml_uri
    elif source_mode == "file":
        resolved = file_uri or (active_uri if active_uri.lower().startswith("file://") else "")
    elif source_mode == "stream":
        resolved = stream_uri or (active_uri if active_uri.lower().startswith("rtsp://") else "") or yaml_uri
    else:
        raise ValueError(f"Unsupported source_mode: {source_mode}")

    if resolved:
        return resolved
    raise RuntimeError(
        f"No {source_mode} URI available for source index {idx}. "
        f"infer_active={active_uri or 'n/a'} infer_stream={stream_uri or 'n/a'} "
        f"infer_file={file_uri or 'n/a'} yaml_uri={yaml_uri or 'n/a'}"
    )


def select_source(
    payload: Dict[str, object],
    *,
    camera: str = DEFAULT_CAMERA,
    frame_size: Tuple[int, int] = (1920, 1080),
    infer_config_path: Optional[Path] = DEFAULT_INFER_CONFIG_PATH,
    source_mode: str = "stream",
) -> Dict[str, object]:
    uris: List[str] = list(payload.get("uris", []) or [])
    sensor_ids: List[str] = [str(x) for x in (payload.get("sensor_ids", []) or [])]
    sensor_names: List[str] = [str(x) for x in (payload.get("sensor_names", []) or [])]
    if not uris:
        raise RuntimeError("sources.yaml has no URIs configured")

    entries = list(zip(uris, sensor_ids, sensor_names))
    target = _normalize_name(camera)
    frame_w = max(1, int(frame_size[0] or 0))
    frame_h = max(1, int(frame_size[1] or 0))
    mode = str(source_mode or "stream").strip().lower()
    infer_candidates = _load_infer_source_candidates(infer_config_path)
    for idx, (uri, sensor_id, sensor_name) in enumerate(entries):
        if _normalize_name(str(sensor_name)) != target:
            continue
        resolved_uri = _resolve_uri_for_mode(
            idx=idx,
            source_mode=mode,
            infer_candidates=infer_candidates,
            yaml_uris=uris,
        )
        return {
            "uri": resolved_uri,
            "sensor_id": str(sensor_id or idx),
            "sensor_name": str(sensor_name or f"source_{idx}"),
            "width": frame_w,
            "height": frame_h,
        }

    available = ", ".join(name for _, _, name in entries)
    raise RuntimeError(f"Camera '{camera}' not found. Available: {available}")
