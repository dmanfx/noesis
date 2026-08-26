from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "DS9/scripts/rebase_asset_realization.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("native_manifest_rebase", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rebaser = _load_module()


def _old_manifest() -> bytes:
    return subprocess.run(
        ["git", "show", "HEAD:DS9/asset_manifest.yaml"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _fixture(tmp_path: Path) -> tuple[Path, bytes, str]:
    root = tmp_path / "artifacts"
    root.mkdir(mode=0o700)
    old_raw = _old_manifest()
    payload = {
        "schema_version": 1,
        "contract": "noesis.ds9.asset_realization",
        "base_manifest": {
            "path": "DS9/asset_manifest.yaml",
            "sha256": hashlib.sha256(old_raw).hexdigest(),
        },
        "source_contracts": {
            "path": "DS9/config/engine_source_contracts.json",
            "sha256": "a" * 64,
        },
        "created_at_utc": "2026-08-12T00:00:00Z",
        "updated_at_utc": "2026-08-12T00:00:00Z",
        "artifacts": {
            "engine.example": {
                "state": "validated",
                "provenance": {"output_sha256": "b" * 64},
            }
        },
    }
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path = root / "asset_realization.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    return root, raw, hashlib.sha256(raw).hexdigest()


def test_native_manifest_rebase_dry_run_preserves_artifacts(tmp_path, monkeypatch) -> None:
    root, raw, digest = _fixture(tmp_path)
    monkeypatch.setattr(rebaser, "_git_manifest", lambda _ref: _old_manifest())
    result = rebaser.rebase(
        artifact_root=root,
        old_manifest_ref="HEAD",
        expected_realization_sha256=digest,
        apply=False,
    )
    assert result["ok"] is True
    assert result["applied"] is False
    assert result["realized_artifact_count"] == 1
    assert (root / "asset_realization.json").read_bytes() == raw


def test_native_manifest_rebase_apply_is_atomic_and_evidenced(
    tmp_path, monkeypatch
) -> None:
    root, raw, digest = _fixture(tmp_path)
    monkeypatch.setattr(rebaser, "_git_manifest", lambda _ref: _old_manifest())
    result = rebaser.rebase(
        artifact_root=root,
        old_manifest_ref="HEAD",
        expected_realization_sha256=digest,
        apply=True,
    )
    assert result["applied"] is True
    updated = json.loads((root / "asset_realization.json").read_text())
    before = json.loads(raw)
    assert updated["artifacts"] == before["artifacts"]
    assert updated["base_manifest"]["sha256"] == hashlib.sha256(
        (REPO_ROOT / "DS9/asset_manifest.yaml").read_bytes()
    ).hexdigest()
    evidence = Path(result["evidence"])
    assert evidence.is_file()
    assert evidence.stat().st_mode & 0o777 == 0o600


def test_manifest_rebase_rejects_realized_artifact_changes() -> None:
    old = rebaser._mapping_from_yaml(_old_manifest(), "old")
    new = rebaser._mapping_from_yaml(
        (REPO_ROOT / "DS9/asset_manifest.yaml").read_bytes(), "new"
    )
    try:
        rebaser._validate_rebase_change(
            old,
            new,
            realized_artifact_ids={"native.latency"},
        )
    except rebaser.RebaseError as exc:
        assert "realized engine authority" in str(exc)
    else:
        raise AssertionError("realized artifact mutation was accepted")
