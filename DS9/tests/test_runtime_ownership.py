from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "DS9/scripts/validate_runtime_ownership.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("native_runtime_ownership", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ownership = _load_validator()


def _matrix():
    return yaml.safe_load(
        (REPO_ROOT / "DS9/docs/runtime_ownership.yaml").read_text(encoding="utf-8")
    )


def test_current_native_ownership_contract_is_valid() -> None:
    result = ownership.validate_matrix(_matrix())
    assert result["ok"], result["errors"]
    assert result["canonical_runtime"] == "ds9_native_host"


def test_ds8_cannot_be_declared_canonical() -> None:
    matrix = copy.deepcopy(_matrix())
    matrix["policy"]["canonical_runtime"] = "ds8"
    result = ownership.validate_matrix(matrix)
    assert not result["ok"]
    assert any("canonical_runtime" in error for error in result["errors"])


def test_adapter_owner_must_be_ds9_owned() -> None:
    matrix = copy.deepcopy(_matrix())
    matrix["modules"][0]["owner_path"] = "noesis/runtime.py"
    result = ownership.validate_matrix(matrix)
    assert not result["ok"]
    assert any("DS9 adapters" in error for error in result["errors"])


def test_shared_owner_cannot_be_a_ds9_copy() -> None:
    matrix = copy.deepcopy(_matrix())
    shared = next(
        row for row in matrix["modules"] if row["classification"] == "shared_single_source"
    )
    shared["owner_path"] = "DS9/noesis/server"
    result = ownership.validate_matrix(matrix)
    assert not result["ok"]
    assert any("shared owners" in error for error in result["errors"])
