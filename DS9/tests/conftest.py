"""Load DS9 adapter modules before shared Noesis modules during DS9 tests."""

from __future__ import annotations

import sys
from pathlib import Path


DS9_ROOT = Path(__file__).resolve().parents[1]
if str(DS9_ROOT) in sys.path:
    sys.path.remove(str(DS9_ROOT))
sys.path.insert(0, str(DS9_ROOT))
