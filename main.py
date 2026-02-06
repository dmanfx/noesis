#!/usr/bin/env python3
"""Legacy DS7 entrypoint removed.

Use the DS8 runtime harness (`noesis/ds8_runtime.py`) for all runtime launches.
"""

from __future__ import annotations

import sys
from textwrap import dedent

_REMOVED_MESSAGE = dedent(
    """
    ERROR: DS7 entrypoint `main.py` has been removed and will not run.

    Use the DS8 runtime harness instead. Example commands:

      python3 noesis/ds8_runtime.py --pipeline-config config/infer.yaml --cameras-config config/cameras.yaml

      python3 noesis/ds8_runtime.py --pipeline-config config/infer_v3dt_baseline.yaml --cameras-config config/cameras.yaml --tracking-mode v3dt --disable-rest
    """
).strip()


def main() -> int:
    print(_REMOVED_MESSAGE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
