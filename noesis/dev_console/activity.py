from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from noesis.ds8_preflight import REPO_ROOT


ACTIVITY_LOG = REPO_ROOT / "build" / "dev_console" / "activity" / "activity.jsonl"
_SENSITIVE_KEY_TOKENS = ("SECRET", "TOKEN", "PASSWORD", "PASSWD", "API_KEY", "PRIVATE_KEY", "CREDENTIAL")
_MAX_STRING = 320
_MAX_ITEMS = 40
_MAX_DEPTH = 5


def _is_sensitive_key(key: str) -> bool:
    upper = key.upper()
    return any(token in upper for token in _SENSITIVE_KEY_TOKENS)


def _jsonable(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if _is_sensitive_key(key):
        return "<redacted>"
    if depth > _MAX_DEPTH:
        return "<truncated>"
    if isinstance(value, Mapping):
        items = list(value.items())[:_MAX_ITEMS]
        return {str(child_key): _jsonable(child_value, key=str(child_key), depth=depth + 1) for child_key, child_value in items}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item, key=key, depth=depth + 1) for item in list(value)[:_MAX_ITEMS]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = str(value) if isinstance(value, str) else value
        if isinstance(text, str) and len(text) > _MAX_STRING:
            return f"{text[:_MAX_STRING]}..."
        return text
    if isinstance(value, Path):
        return str(value)
    text = str(value)
    return f"{text[:_MAX_STRING]}..." if len(text) > _MAX_STRING else text


def _event_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def _clean_severity(severity: str) -> str:
    normalized = str(severity or "info").strip().lower()
    return normalized if normalized in {"info", "ok", "warn", "block"} else "info"


def record_activity(
    event_type: str,
    title: str,
    *,
    severity: str = "info",
    detail: str = "",
    payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    ts = time.time()
    event = {
        "id": uuid.uuid4().hex[:12],
        "ts": ts,
        "ts_text": _event_time(ts),
        "type": str(event_type or "activity"),
        "severity": _clean_severity(severity),
        "title": str(title or event_type or "Activity"),
        "detail": str(detail or ""),
        "payload": _jsonable(dict(payload or {})),
    }
    ACTIVITY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ACTIVITY_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    return event


def record_activity_safe(*args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
    try:
        return record_activity(*args, **kwargs)
    except Exception:
        return None


def list_activity(*, limit: int = 100) -> Dict[str, Any]:
    capped = max(1, min(500, int(limit)))
    items: List[Dict[str, Any]] = []
    total = 0
    if ACTIVITY_LOG.exists():
        lines = ACTIVITY_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        for line in reversed(lines):
            if len(items) >= capped:
                break
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, Mapping):
                items.append(dict(item))
    return {"items": items, "limit": capped, "total": total, "path": str(ACTIVITY_LOG)}
