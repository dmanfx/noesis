from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlparse

import yaml

from noesis.ds8_preflight import REPO_ROOT
from noesis.dev_console.launch_spec import LaunchSpec


@dataclass(frozen=True)
class MediaRoot:
    id: str
    label: str
    path: Path

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "label": self.label, "path": str(self.path)}


def _load_mapping(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"YAML must be a mapping: {path}")
    return payload


def _relpath(path: Path, root: Path = REPO_ROOT) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _display_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme == "rtsp":
        host = parsed.hostname or "host"
        port = f":{parsed.port}" if parsed.port else ""
        return f"rtsp://{host}{port}/..."
    if parsed.scheme == "file":
        path = Path(parsed.path or "")
        return f"file://.../{path.name}" if path.name else "file://..."
    return uri[:120] if len(uri) <= 120 else f"{uri[:120]}..."


def _source_kind(uri: str) -> str:
    value = str(uri or "").strip().lower()
    parsed = urlparse(value)
    if parsed.scheme == "rtsp":
        return "rtsp"
    if parsed.scheme == "file" and value.endswith(".mp4"):
        return "mp4"
    if parsed.scheme == "file":
        return "file"
    if value.endswith(".mp4"):
        return "mp4"
    return parsed.scheme or "uri"


def _camera_by_index(cameras: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    raw = cameras.get(index)
    if raw is None:
        raw = cameras.get(str(index))
    return raw if isinstance(raw, Mapping) else {}


def _source_enabled(raw: Mapping[str, Any]) -> bool:
    value = raw.get("enabled", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _source_from_row(raw: Mapping[str, Any]) -> Dict[str, Any]:
    nested = raw.get("source")
    source = deepcopy(nested) if isinstance(nested, Mapping) else deepcopy(raw)
    for key in ("enabled", "id", "kind", "label", "selected", "source", "source_id", "uri_display"):
        source.pop(key, None)
    uri = str(raw.get("uri") or source.get("uri") or "").strip()
    if uri:
        source["uri"] = uri
    uri_secret = str(raw.get("uri_secret") or source.get("uri_secret") or "").strip()
    if uri_secret:
        source["uri_secret"] = uri_secret
    return source


def build_source_catalog(spec: LaunchSpec) -> Dict[str, Any]:
    pipeline_cfg = _load_mapping(spec.pipeline_path)
    cameras_cfg = _load_mapping(spec.cameras_path)
    cameras = cameras_cfg.get("cameras") if isinstance(cameras_cfg.get("cameras"), Mapping) else {}

    if spec.source_overrides:
        raw_sources: Iterable[Mapping[str, Any]] = [
            item for item in spec.source_overrides if isinstance(item, Mapping)
        ]
        using_overrides = True
    else:
        raw = pipeline_cfg.get("sources")
        raw_sources = raw if isinstance(raw, list) else []
        using_overrides = False

    rows: List[Dict[str, Any]] = []
    for index, raw_source in enumerate(raw_sources):
        source = _source_from_row(raw_source if isinstance(raw_source, Mapping) else {})
        uri = str(source.get("uri") or "").strip()
        uri_secret = str(source.get("uri_secret") or "").strip()
        camera = _camera_by_index(cameras, index)
        camera_id = str(raw_source.get("camera_id") or camera.get("name") or f"camera_{index}")
        label = str(raw_source.get("label") or camera_id)
        rows.append(
            {
                "id": str(raw_source.get("id") or f"source-{index}"),
                "source_id": int(raw_source.get("source_id", index) or index),
                "camera_id": camera_id,
                "label": label,
                "kind": str(raw_source.get("kind") or ("rtsp" if uri_secret else _source_kind(uri))),
                "enabled": _source_enabled(raw_source),
                "uri": uri,
                "uri_secret": uri_secret,
                "uri_display": f"camera-secret:{uri_secret}" if uri_secret else _display_uri(uri),
                "source": source,
            }
        )

    selected = sum(1 for row in rows if row.get("enabled"))
    return {
        "summary": {
            "total": len(rows),
            "selected": selected,
            "using_overrides": using_overrides,
        },
        "rows": rows,
        "pipeline_config": spec.pipeline_config,
        "cameras_config": spec.cameras_config,
    }


def default_media_roots() -> List[MediaRoot]:
    home = Path.home()
    candidates = [
        MediaRoot("workspace", "Workspace", REPO_ROOT),
        MediaRoot("videos", "Videos", home / "Videos"),
        MediaRoot("downloads", "Downloads", home / "Downloads"),
        MediaRoot("documents", "Documents", home / "Documents"),
    ]
    roots: List[MediaRoot] = []
    seen: set[Path] = set()
    for item in candidates:
        path = item.path.expanduser().resolve()
        if path in seen or not path.is_dir():
            continue
        seen.add(path)
        roots.append(MediaRoot(item.id, item.label, path))
    return roots or [MediaRoot("workspace", "Workspace", REPO_ROOT)]


def browse_mp4_media(
    *,
    root_id: Optional[str] = None,
    relative_path: str = "",
    roots: Optional[Iterable[MediaRoot]] = None,
    max_entries: int = 240,
) -> Dict[str, Any]:
    available_roots = list(roots or default_media_roots())
    if not available_roots:
        available_roots = [MediaRoot("workspace", "Workspace", REPO_ROOT)]
    root_map = {root.id: root for root in available_roots}
    active_root = root_map.get(str(root_id or "")) or available_roots[0]
    root_path = active_root.path.expanduser().resolve()
    requested_rel = str(relative_path or "").strip().lstrip("/")
    current = (root_path / requested_rel).resolve()
    if not _is_relative_to(current, root_path):
        current = root_path
        requested_rel = ""
    if not current.exists() or not current.is_dir():
        raise ValueError(f"media browser path is not a directory: {requested_rel or active_root.label}")

    entries: List[Dict[str, Any]] = []
    truncated = False
    try:
        children = sorted(current.iterdir(), key=lambda path: (path.is_file(), path.name.lower()))
    except OSError as exc:
        raise ValueError(f"media browser cannot read directory: {exc}") from exc

    for child in children:
        if len(entries) >= int(max_entries):
            truncated = True
            break
        try:
            if child.is_dir():
                child_rel = _relpath(child.resolve(), root_path)
                entries.append(
                    {
                        "kind": "dir",
                        "name": child.name,
                        "relative_path": "" if child_rel == "." else child_rel,
                        "path": str(child),
                    }
                )
            elif child.is_file() and child.suffix.lower() == ".mp4":
                stat = child.stat()
                child_rel = _relpath(child.resolve(), root_path)
                entries.append(
                    {
                        "kind": "file",
                        "name": child.name,
                        "relative_path": child_rel,
                        "path": str(child),
                        "uri": child.resolve().as_uri(),
                        "size_bytes": int(stat.st_size),
                        "mtime": float(stat.st_mtime),
                    }
                )
        except OSError:
            continue

    parent = ""
    if current != root_path:
        parent = _relpath(current.parent.resolve(), root_path)
        if parent == ".":
            parent = ""
    current_rel = _relpath(current.resolve(), root_path)
    if current_rel == ".":
        current_rel = ""
    file_count = sum(1 for item in entries if item.get("kind") == "file")
    dir_count = sum(1 for item in entries if item.get("kind") == "dir")
    return {
        "root_id": active_root.id,
        "root_label": active_root.label,
        "root_path": str(root_path),
        "relative_path": current_rel,
        "parent_path": parent,
        "roots": [root.to_dict() for root in available_roots],
        "entries": entries,
        "summary": {"files": file_count, "dirs": dir_count, "truncated": truncated},
    }
