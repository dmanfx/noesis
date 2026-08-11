from __future__ import annotations

import os
from pathlib import Path

from noesis.ds8_preflight import REPO_ROOT


def noesis_build_root(override: str | Path | None = None) -> Path:
    """Return the configured mutable build root used by runtime materializers."""

    raw = str(override or os.environ.get("NOESIS_BUILD_DIR", "")).strip()
    candidate = Path(raw).expanduser() if raw else REPO_ROOT / "build"
    return candidate.resolve(strict=False)


def dev_console_root(override: str | Path | None = None) -> Path:
    """Return the mutable dev-console root below the canonical build root."""

    return noesis_build_root(override) / "dev_console"
