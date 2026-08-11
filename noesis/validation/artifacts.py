from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Mapping

from noesis_core.private_paths import (
    PRIVATE_DIRECTORY_MODE,
    PrivatePathError,
    ensure_private_directory,
    validate_private_file,
)


def private_artifact_writer(function):
    """Run one synchronous artifact writer with owner-only creation modes."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        previous_umask = os.umask(0o077)
        try:
            return function(*args, **kwargs)
        finally:
            os.umask(previous_umask)

    return wrapped


def ensure_private_artifact_tree(root: str | Path, *, label: str) -> Path:
    """Create or validate an owner-only validation-artifact tree."""

    resolved = ensure_private_directory(root, label=label)
    for path in resolved.rglob("*"):
        info = path.lstat()
        item_label = f"{label} item"
        if stat.S_ISLNK(info.st_mode):
            raise PrivatePathError(f"{item_label} must not be a symlink")
        if info.st_uid != os.geteuid():
            raise PrivatePathError(f"{item_label} must be owned by the service user")
        if stat.S_ISDIR(info.st_mode):
            mode = stat.S_IMODE(info.st_mode)
            if mode != PRIVATE_DIRECTORY_MODE:
                raise PrivatePathError(
                    f"{item_label} directory mode must be {PRIVATE_DIRECTORY_MODE:04o}; found {mode:04o}"
                )
            continue
        if stat.S_ISREG(info.st_mode):
            validate_private_file(path, label=item_label)
            continue
        raise PrivatePathError(f"{item_label} must be a regular file or directory")
    return resolved


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
