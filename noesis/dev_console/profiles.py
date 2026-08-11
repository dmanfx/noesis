from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from noesis.dev_console.launch_spec import LaunchSpec
from noesis.dev_console.paths import dev_console_root


# Test/operator override. The default follows NOESIS_BUILD_DIR at access time.
PROFILE_ROOT: Path | None = None


def profile_root() -> Path:
    if PROFILE_ROOT is not None:
        return Path(PROFILE_ROOT).expanduser().resolve(strict=False)
    return dev_console_root() / "profiles"


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(text or "").strip().lower()).strip("-._")
    return cleaned[:80] or f"profile-{int(time.time())}"


def _profile_path(profile_id: str) -> Path:
    safe_id = _slug(profile_id)
    return profile_root() / f"{safe_id}.json"


def _profile_summary(profile: Mapping[str, Any]) -> Dict[str, Any]:
    spec = profile.get("spec") if isinstance(profile.get("spec"), Mapping) else {}
    return {
        "id": profile.get("id"),
        "name": profile.get("name"),
        "notes": profile.get("notes", ""),
        "created_at": profile.get("created_at"),
        "updated_at": profile.get("updated_at"),
        "pipeline_config": spec.get("pipeline_config"),
        "pgie_profile": spec.get("pgie_profile"),
        "size": spec.get("size"),
        "tracking_mode": spec.get("tracking_mode"),
        "ws_port": spec.get("ws_port"),
        "rest_port": spec.get("rest_port"),
        "rtsp_port": spec.get("rtsp_port"),
        "depth_enable_seconds": spec.get("depth_enable_seconds"),
        "env_count": len(spec.get("env") or {}),
    }


def list_profiles() -> List[Dict[str, Any]]:
    root = profile_root()
    root.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(profile, Mapping):
            items.append(_profile_summary(profile))
    return sorted(items, key=lambda item: str(item.get("updated_at") or ""), reverse=True)


def save_profile(*, name: str, notes: str = "", spec: LaunchSpec, profile_id: Optional[str] = None) -> Dict[str, Any]:
    profile_root().mkdir(parents=True, exist_ok=True)
    clean_name = str(name or "").strip() or f"{spec.pgie_profile} {spec.tracking_mode}"
    clean_id = _slug(profile_id or clean_name)
    path = _profile_path(clean_id)
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    created_at = now
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, Mapping) and existing.get("created_at"):
                created_at = str(existing.get("created_at"))
        except Exception:
            pass
    payload = {
        "schema_version": 1,
        "id": clean_id,
        "name": clean_name,
        "notes": str(notes or ""),
        "created_at": created_at,
        "updated_at": now,
        "spec": spec.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"profile": payload, "summary": _profile_summary(payload), "path": str(path)}


def load_profile(profile_id: str) -> Dict[str, Any]:
    path = _profile_path(profile_id)
    if not path.exists():
        raise FileNotFoundError(f"Unknown launch profile: {profile_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Launch profile is not a JSON object: {profile_id}")
    return {"profile": dict(payload), "summary": _profile_summary(payload), "path": str(path)}


def delete_profile(profile_id: str) -> Dict[str, Any]:
    path = _profile_path(profile_id)
    if not path.exists():
        raise FileNotFoundError(f"Unknown launch profile: {profile_id}")
    path.unlink()
    return {"ok": True, "profile_id": _slug(profile_id)}
