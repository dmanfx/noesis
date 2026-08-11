from __future__ import annotations

import importlib

import pytest


def test_object_depth_native_module_importable_if_built() -> None:
    module = pytest.importorskip("noesis_depth_meta_ext")
    assert callable(getattr(module, "attach_object_depth", None))
    assert callable(getattr(module, "extract_object_depth", None))
    assert callable(getattr(module, "extract_object_mask", None))
    assert callable(getattr(module, "object_depth_meta_type", None))


def test_object_depth_native_module_can_be_reimported_if_built() -> None:
    module = pytest.importorskip("noesis_depth_meta_ext")
    reloaded = importlib.reload(module)
    assert reloaded is module
