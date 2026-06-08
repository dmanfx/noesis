from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class FixtureRegistryEntry:
    fixture_id: str
    path: Path
    description: str = ""
    tiers: tuple[str, ...] = ()
    rooms: tuple[str, ...] = ()
    cameras: tuple[str, ...] = ()
    expected: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class FixtureRegistry:
    path: Path
    entries: dict[str, FixtureRegistryEntry]

    @classmethod
    def load(cls, path: str | Path) -> "FixtureRegistry":
        registry_path = Path(path)
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"fixture registry must be a JSON object: {registry_path}")
        raw_fixtures = payload.get("fixtures")
        if not isinstance(raw_fixtures, list):
            raise RuntimeError(f"fixture registry requires fixtures[]: {registry_path}")
        entries: dict[str, FixtureRegistryEntry] = {}
        for raw in raw_fixtures:
            if not isinstance(raw, Mapping):
                continue
            fixture_id = str(raw.get("id") or "").strip()
            rel_path = str(raw.get("path") or "").strip()
            if not fixture_id or not rel_path:
                continue
            fixture_path = Path(rel_path)
            if not fixture_path.is_absolute():
                fixture_path = registry_path.parent / fixture_path
            entries[fixture_id] = FixtureRegistryEntry(
                fixture_id=fixture_id,
                path=fixture_path,
                description=str(raw.get("description") or ""),
                tiers=tuple(str(v) for v in raw.get("tiers") or ()),
                rooms=tuple(str(v) for v in raw.get("rooms") or ()),
                cameras=tuple(str(v) for v in raw.get("cameras") or ()),
                expected=raw.get("expected") if isinstance(raw.get("expected"), Mapping) else None,
            )
        return cls(path=registry_path, entries=entries)

    def resolve(self, fixture_id: str) -> Path:
        entry = self.entries.get(str(fixture_id))
        if entry is None:
            known = ", ".join(sorted(self.entries)) or "<none>"
            raise RuntimeError(f"unknown fixture id {fixture_id!r}; known fixtures: {known}")
        return entry.path
