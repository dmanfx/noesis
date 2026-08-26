"""Load DS9 adapter modules before shared Noesis modules during DS9 tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


DS9_ROOT = Path(__file__).resolve().parents[1]
if str(DS9_ROOT) in sys.path:
    sys.path.remove(str(DS9_ROOT))
sys.path.insert(0, str(DS9_ROOT))

DS9_NATIVE_EXTENSION_MODULES = (
    "noesis_pose_meta_ext",
    "noesis_analytics_meta_ext",
    "noesis_v3dt_meta_ext",
    "noesis_reid_meta_ext",
    "noesis_latency_ext",
    "noesis_depth_meta_ext",
    "noesis_depth_tracking_tensor_ext",
)
_MISSING = object()


@pytest.fixture(autouse=True)
def _isolate_native_extension_module_cache():  # type: ignore[no-untyped-def]
    """Prevent native-extension cache state from leaking between test modules."""

    before = {
        name: sys.modules.get(name, _MISSING)
        for name in DS9_NATIVE_EXTENSION_MODULES
    }
    yield
    for name, module in before.items():
        if module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module  # type: ignore[assignment]
