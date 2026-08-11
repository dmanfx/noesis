#!/usr/bin/env python3
"""Strict compatibility launcher for the canonical ROI hot-restore gate."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NoReturn, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_GATE = REPO_ROOT / "scripts" / "roi_reload_smoke_test.py"


def main(argv: Sequence[str] | None = None) -> NoReturn:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    os.execv(
        sys.executable,
        [sys.executable, str(CANONICAL_GATE), *forwarded],
    )
    raise AssertionError("os.execv returned unexpectedly")


if __name__ == "__main__":
    main()
