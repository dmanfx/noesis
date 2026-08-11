from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIRTUAL_TWIN_ROOT = REPO_ROOT / "data" / "virtual_twin"
REVISION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class VirtualTwinStoreError(RuntimeError):
    """Raised when a virtual-twin artifact cannot be read safely."""


def _root_from_env() -> Path:
    raw = os.environ.get("NOESIS_VIRTUAL_TWIN_ROOT", "").strip()
    if not raw:
        return DEFAULT_VIRTUAL_TWIN_ROOT
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)


def validate_revision_id(revision_id: str) -> str:
    value = str(revision_id or "").strip()
    if not REVISION_ID_RE.match(value):
        raise VirtualTwinStoreError(f"invalid virtual-twin revision id: {revision_id!r}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VirtualTwinStoreError(f"virtual-twin artifact missing: {path}") from exc
    except Exception as exc:
        raise VirtualTwinStoreError(f"unable to parse virtual-twin artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VirtualTwinStoreError(f"virtual-twin artifact root must be an object: {path}")
    return payload


@dataclass(frozen=True)
class VirtualTwinStore:
    """Filesystem-backed access to versioned virtual-twin revisions."""

    root: Path | None = None

    def __post_init__(self) -> None:
        chosen = Path(self.root) if self.root is not None else _root_from_env()
        # Preserve path components so security-sensitive scene readers can
        # reject symlinked roots instead of having resolve() hide them.
        object.__setattr__(
            self,
            "root",
            Path(os.path.abspath(os.fspath(chosen.expanduser()))),
        )

    @property
    def revisions_root(self) -> Path:
        return Path(self.root) / "revisions"

    @property
    def latest_pointer_path(self) -> Path:
        return Path(self.root) / "latest"

    def ensure_root(self) -> None:
        self.revisions_root.mkdir(parents=True, exist_ok=True)

    def revision_dir(self, revision_id: str) -> Path:
        safe_id = validate_revision_id(revision_id)
        return self.revisions_root / safe_id

    def list_revisions(self) -> list[dict[str, Any]]:
        if not self.revisions_root.exists():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.revisions_root.iterdir()):
            if not path.is_dir():
                continue
            try:
                revision_id = validate_revision_id(path.name)
                manifest = self.read_manifest(revision_id)
            except Exception:
                continue
            rows.append(
                {
                    "revision_id": revision_id,
                    "camera": manifest.get("camera"),
                    "created_ts_us": int(manifest.get("created_ts_us") or 0),
                    "label": manifest.get("label") or revision_id,
                }
            )
        rows.sort(key=lambda item: int(item.get("created_ts_us") or 0), reverse=True)
        return rows

    def latest_revision_id(self) -> str | None:
        try:
            raw = self.latest_pointer_path.read_text(encoding="utf-8").strip()
            if raw:
                return validate_revision_id(raw)
        except FileNotFoundError:
            pass
        except Exception as exc:
            raise VirtualTwinStoreError(f"unable to read virtual-twin latest pointer: {exc}") from exc
        revisions = self.list_revisions()
        if not revisions:
            return None
        return str(revisions[0]["revision_id"])

    def write_latest_revision_id(self, revision_id: str) -> None:
        safe_id = validate_revision_id(revision_id)
        self.ensure_root()
        self.latest_pointer_path.write_text(f"{safe_id}\n", encoding="utf-8")

    def read_manifest(self, revision_id: str) -> dict[str, Any]:
        return _read_json(self.revision_dir(revision_id) / "manifest.json")

    def read_metrics(self, revision_id: str) -> dict[str, Any]:
        return _read_json(self.revision_dir(revision_id) / "metrics.json")

    def read_tracking_alignment(self, revision_id: str) -> dict[str, Any]:
        return _read_json(self.revision_dir(revision_id) / "tracking_alignment.json")

    def artifact_path(self, revision_id: str, artifact_rel_path: str) -> Path:
        revision_dir = self.revision_dir(revision_id).resolve()
        rel = Path(str(artifact_rel_path or "").strip())
        if rel.is_absolute() or any(part == ".." for part in rel.parts):
            raise VirtualTwinStoreError("virtual-twin artifact path must be revision-relative")
        path = (revision_dir / rel).resolve()
        if revision_dir not in path.parents and path != revision_dir:
            raise VirtualTwinStoreError("virtual-twin artifact path escaped revision directory")
        if not path.exists() or not path.is_file():
            raise VirtualTwinStoreError(f"virtual-twin artifact missing: {artifact_rel_path}")
        return path

    def artifact_urls(self, revision_id: str, manifest: Mapping[str, Any] | None = None) -> dict[str, str]:
        payload = dict(manifest or self.read_manifest(revision_id))
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, Mapping):
            return {}
        urls: dict[str, str] = {}
        for name, rel_path in artifacts.items():
            if not isinstance(rel_path, str) or not rel_path.strip():
                continue
            urls[str(name)] = f"/api/v1/virtual-twin/revisions/{revision_id}/artifacts/{rel_path}"
        return urls

    def latest_payload(self) -> dict[str, Any]:
        revision_id = self.latest_revision_id()
        if revision_id is None:
            raise VirtualTwinStoreError("no virtual-twin revision has been built")
        manifest = self.read_manifest(revision_id)
        metrics = self.read_metrics(revision_id)
        tracking_alignment = self.read_tracking_alignment(revision_id)
        return {
            "revision_id": revision_id,
            "manifest": manifest,
            "metrics": metrics,
            "tracking_alignment": tracking_alignment,
            "artifact_urls": self.artifact_urls(revision_id, manifest),
        }


__all__ = [
    "DEFAULT_VIRTUAL_TWIN_ROOT",
    "VirtualTwinStore",
    "VirtualTwinStoreError",
    "validate_revision_id",
]
