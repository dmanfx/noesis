from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class ArtifactIndex:
    root: Path
    artifacts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative_path: str | Path) -> Path:
        rel = Path(relative_path)
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def add(self, key: str, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> str:
        target = Path(path)
        try:
            value = str(target.relative_to(self.root))
        except Exception:
            value = str(target)
        if metadata:
            self.artifacts[key] = {"path": value, **dict(metadata)}
        else:
            self.artifacts[key] = value
        return value

    def write(self, relative_path: str | Path = "visual/index.json") -> Path:
        target = self.path(relative_path)
        target.write_text(json.dumps(self.artifacts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target
