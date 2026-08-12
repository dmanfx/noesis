from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from DS9.noesis import v3dt_assets
from DS9.scripts import engine_maintenance_common
from noesis_core.strict_json import StrictJSONError


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_asset_validator():
    path = REPO_ROOT / "DS9" / "scripts" / "validate_asset_manifest.py"
    spec = importlib.util.spec_from_file_location(
        "strict_artifact_json_asset_validator",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_engine_source_contracts_reject_duplicate_selector(tmp_path: Path) -> None:
    path = tmp_path / "engine_source_contracts.json"
    path.write_text(
        '{"schema_version":1,"contracts":{"mapanything":{}},'
        '"contracts":{"alternate":{}}}',
        encoding="utf-8",
    )

    with pytest.raises(
        engine_maintenance_common.EngineMaintenanceError,
        match="invalid engine source contract file",
    ):
        engine_maintenance_common.load_source_contracts(path)


def test_engine_maintenance_receipt_rejects_nested_duplicate(tmp_path: Path) -> None:
    validator = _load_asset_validator()
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"contract":"noesis.ds9.engine_maintenance",'
        '"installed":{"sha256":"first","sha256":"second"}}',
        encoding="utf-8",
    )

    with pytest.raises(StrictJSONError, match="duplicate JSON object key"):
        validator._load_maintenance_manifest(path, "engine.mapanything")


def test_v3dt_source_provenance_rejects_duplicate_contract_field(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"reviewed-model")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    provenance = tmp_path / "model.provenance.json"
    provenance.write_text(
        "{"
        f'"source_sha256":"{digest}",'
        f'"output_sha256":"{digest}",'
        '"output":"wrong","output":"models/model.onnx",'
        f'"output_bytes":{source.stat().st_size}'
        "}",
        encoding="utf-8",
    )
    errors: list[str] = []

    result = v3dt_assets._validate_source_provenance(
        provenance,
        source_path=source,
        expected_sha256=digest,
        expected_output="models/model.onnx",
        label="BodyPose source",
        errors=errors,
    )

    assert result is None
    assert any("duplicate JSON object key" in error for error in errors)
    assert all("wrong" not in error for error in errors)
