from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "DS9/scripts/transition_mapanything_fp32_authority.py"


def _load_transitioner():
    spec = importlib.util.spec_from_file_location("ds9_mapanything_transition", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transitioner = _load_transitioner()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, raw: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)


def _fixture(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "artifacts"
    root.mkdir(mode=0o750, parents=True)
    _write(root / transitioner.LOCK_FILENAME, b"", 0o600)
    old_output = root / "models/engines/mapanything_images_294x518_b3_fp16.plan"
    old_output_raw = b"reviewed-fp16-engine-fixture"
    _write(old_output, old_output_raw, 0o600)

    other = {
        "id": "engine.other",
        "kind": "tensorrt_engine",
        "output": "DS9/models/engines/other.plan",
        "compatibility": {"precision": "fp16"},
    }
    old_map = {
        "id": transitioner.MAP_ARTIFACT_ID,
        "kind": "tensorrt_engine",
        "output": transitioner.OLD_MAP_OUTPUT,
        "compatibility": {"precision": "fp16", "batch": 3},
    }
    new_map = copy.deepcopy(old_map)
    new_map["output"] = transitioner.NEW_MAP_OUTPUT
    new_map["compatibility"]["precision"] = "fp32"
    old_manifest = {
        "schema_version": 2,
        "target": {"fixture": True},
        "artifacts": [other, old_map],
    }
    new_manifest = copy.deepcopy(old_manifest)
    new_manifest["artifacts"][1] = new_map

    old_source = {
        "schema_version": 1,
        "contracts": {
            "other": {"token": "same"},
            transitioner.MAP_CONTRACT_NAME: {
                "maintenance_build": {"precision_arg": "--fp16"}
            },
        },
    }
    new_source = copy.deepcopy(old_source)
    new_source["contracts"][transitioner.MAP_CONTRACT_NAME] = {
        "maintenance_build": {"precision": "fp32"},
        "quality_gate": {"contract": "fixture"},
    }

    checkpoint = tmp_path / "checkpoint"
    old_manifest_path = checkpoint / "DS9/asset_manifest.yaml"
    old_source_path = checkpoint / "DS9/config/engine_source_contracts.json"
    repo = tmp_path / "repo"
    new_manifest_path = repo / "DS9/asset_manifest.yaml"
    new_source_path = repo / "DS9/config/engine_source_contracts.json"
    old_manifest_raw = yaml.safe_dump(old_manifest, sort_keys=False).encode()
    new_manifest_raw = yaml.safe_dump(new_manifest, sort_keys=False).encode()
    old_source_raw = _json_bytes(old_source)
    new_source_raw = _json_bytes(new_source)
    _write(old_manifest_path, old_manifest_raw, 0o644)
    _write(new_manifest_path, new_manifest_raw, 0o644)
    _write(old_source_path, old_source_raw, 0o644)
    _write(new_source_path, new_source_raw, 0o644)

    map_record = {
        "state": "staged_unverified",
        "provenance": {
            "output_sha256": _sha(old_output_raw),
            "source_sha256": "a" * 64,
            "maintenance": {
                "output_size_bytes": len(old_output_raw),
                "precision": "fp16",
            },
        },
    }
    other_record = {
        "state": "validated",
        "provenance": {"opaque": "must-remain-byte-equivalent"},
    }
    realization = {
        "schema_version": 1,
        "contract": transitioner.REALIZATION_CONTRACT,
        "base_manifest": {
            "path": transitioner.BASE_MANIFEST_PATH,
            "sha256": _sha(old_manifest_raw),
        },
        "source_contracts": {
            "path": transitioner.SOURCE_CONTRACT_PATH,
            "sha256": _sha(old_source_raw),
        },
        "created_at_utc": "2026-07-11T00:00:00Z",
        "updated_at_utc": "2026-07-11T01:00:00Z",
        "artifacts": {
            transitioner.MAP_ARTIFACT_ID: map_record,
            "engine.other": other_record,
        },
    }
    realization_raw = _json_bytes(realization)
    realization_path = root / transitioner.REALIZATION_FILENAME
    _write(realization_path, realization_raw, 0o600)
    return {
        "root": root,
        "repo": repo,
        "old_manifest_path": old_manifest_path,
        "new_manifest_path": new_manifest_path,
        "old_source_path": old_source_path,
        "new_source_path": new_source_path,
        "old_manifest_raw": old_manifest_raw,
        "new_manifest_raw": new_manifest_raw,
        "old_source_raw": old_source_raw,
        "new_source_raw": new_source_raw,
        "realization": realization,
        "realization_raw": realization_raw,
        "realization_path": realization_path,
        "old_output": old_output,
        "old_output_raw": old_output_raw,
        "map_record": map_record,
        "old_map_record_sha256": _sha(_json_bytes(map_record)),
        "other_record": other_record,
    }


def _constants(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "REPO_ROOT": fixture["repo"],
        "NEW_MANIFEST_AUTHORITY": fixture["new_manifest_path"],
        "NEW_SOURCE_CONTRACT_AUTHORITY": fixture["new_source_path"],
        "OLD_MANIFEST_SHA256": _sha(fixture["old_manifest_path"].read_bytes()),
        "NEW_MANIFEST_SHA256": _sha(fixture["new_manifest_path"].read_bytes()),
        "OLD_SOURCE_CONTRACTS_SHA256": _sha(fixture["old_source_path"].read_bytes()),
        "NEW_SOURCE_CONTRACTS_SHA256": _sha(fixture["new_source_path"].read_bytes()),
        "OLD_MAP_RECORD_SHA256": fixture["old_map_record_sha256"],
        "OLD_FP16_OUTPUT_SHA256": _sha(fixture["old_output_raw"]),
        "OLD_FP16_OUTPUT_SIZE_BYTES": len(fixture["old_output_raw"]),
    }


def _arguments(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_root": fixture["root"],
        "old_manifest_snapshot": fixture["old_manifest_path"],
        "old_source_contract_snapshot": fixture["old_source_path"],
        "expected_old_realization_sha256": _sha(
            fixture["realization_path"].read_bytes()
        ),
        "expected_new_realization_sha256": None,
        "expected_plan_sha256": None,
        "updated_at_utc": "2026-07-11T02:00:00Z",
        "dry_run": True,
    }


def _call(fixture: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    arguments = _arguments(fixture)
    arguments.update(overrides)
    with mock.patch.multiple(transitioner, **_constants(fixture)):
        return transitioner.transition_mapanything_authority(**arguments)


def test_dry_run_is_write_free_and_apply_is_exact(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    before_realization = fixture["realization_path"].read_bytes()
    before_output = fixture["old_output"].read_bytes()

    dry = _call(fixture)

    assert dry["dry_run"] is True
    assert dry["realized_artifact_ids_before"] == [
        "engine.mapanything",
        "engine.other",
    ]
    assert dry["realized_artifact_ids_after"] == ["engine.other"]
    assert fixture["realization_path"].read_bytes() == before_realization
    assert not (fixture["root"] / transitioner.EVIDENCE_ROOT_RELATIVE).exists()

    applied = _call(
        fixture,
        dry_run=False,
        expected_new_realization_sha256=dry["new_realization_sha256"],
        expected_plan_sha256=dry["transition_plan_sha256"],
    )

    assert applied["state"] == "committed"
    current = json.loads(fixture["realization_path"].read_text())
    assert current["base_manifest"]["sha256"] == _sha(fixture["new_manifest_raw"])
    assert current["source_contracts"]["sha256"] == _sha(fixture["new_source_raw"])
    assert current["artifacts"] == {"engine.other": fixture["other_record"]}
    assert fixture["old_output"].read_bytes() == before_output
    run_dir = (fixture["root"] / applied["evidence"]).parent
    assert stat.S_IMODE(run_dir.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
    assert {
        path.name for path in run_dir.iterdir()
    } == transitioner._TRANSACTION_INVENTORY
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in run_dir.iterdir())
    evidence = json.loads((run_dir / transitioner.EVIDENCE_FILENAME).read_text())
    assert evidence["state"] == "committed"
    assert evidence["verification_stages"] == [
        "initial",
        "snapshots_written",
        "pre_cas",
        "post_cas",
        "pre_evidence_commit",
    ]


@pytest.mark.parametrize("field", ("realization", "plan"))
def test_apply_requires_exact_dry_run_authorization(tmp_path: Path, field: str) -> None:
    fixture = _fixture(tmp_path)
    dry = _call(fixture)
    kwargs = {
        "dry_run": False,
        "expected_new_realization_sha256": dry["new_realization_sha256"],
        "expected_plan_sha256": dry["transition_plan_sha256"],
    }
    kwargs[
        "expected_new_realization_sha256"
        if field == "realization"
        else "expected_plan_sha256"
    ] = "0" * 64
    with pytest.raises(transitioner.TransitionError, match="differs from dry-run"):
        _call(fixture, **kwargs)
    assert fixture["realization_path"].read_bytes() == fixture["realization_raw"]


@pytest.mark.parametrize(
    ("damage", "message"),
    (
        ("missing", "missing"),
        ("symlink", "symlink|safely opened"),
        ("hardlink", "single-link"),
        ("record", "record hash"),
        ("manifest_diff", "non-MapAnything"),
        ("source_diff", "not exactly"),
    ),
)
def test_unsafe_or_broadened_transition_is_rejected(
    tmp_path: Path, damage: str, message: str
) -> None:
    fixture = _fixture(tmp_path)
    if damage == "missing":
        fixture["old_output"].unlink()
    elif damage == "symlink":
        target = fixture["old_output"].with_name("target.plan")
        fixture["old_output"].rename(target)
        fixture["old_output"].symlink_to(target)
    elif damage == "hardlink":
        os.link(fixture["old_output"], fixture["old_output"].with_name("alias.plan"))
    elif damage == "record":
        fixture["realization"]["artifacts"][transitioner.MAP_ARTIFACT_ID]["provenance"][
            "unexpected"
        ] = True
        fixture["realization_raw"] = _json_bytes(fixture["realization"])
        _write(fixture["realization_path"], fixture["realization_raw"], 0o600)
    elif damage == "manifest_diff":
        payload = yaml.safe_load(fixture["new_manifest_path"].read_text())
        payload["artifacts"][0]["unexpected"] = True
        raw = yaml.safe_dump(payload, sort_keys=False).encode()
        _write(fixture["new_manifest_path"], raw, 0o644)
        fixture["new_manifest_raw"] = raw
    else:
        payload = json.loads(fixture["new_source_path"].read_text())
        payload["contracts"]["other"]["token"] = "drift"
        raw = _json_bytes(payload)
        _write(fixture["new_source_path"], raw, 0o644)
        fixture["new_source_raw"] = raw

    with pytest.raises(transitioner.TransitionError, match=message):
        _call(fixture)


def test_input_drift_after_preparation_emits_terminal_abort(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    dry = _call(fixture)
    with mock.patch.multiple(transitioner, **_constants(fixture)):
        with mock.patch.object(
            transitioner,
            "_revalidate_originals_and_snapshots",
            side_effect=transitioner.TransitionError("authority drifted"),
        ):
            with pytest.raises(transitioner.TransitionError, match="authority drifted"):
                transitioner.transition_mapanything_authority(
                    **{
                        **_arguments(fixture),
                        "dry_run": False,
                        "expected_new_realization_sha256": dry[
                            "new_realization_sha256"
                        ],
                        "expected_plan_sha256": dry["transition_plan_sha256"],
                    }
                )
    assert fixture["realization_path"].read_bytes() == fixture["realization_raw"]
    evidence_paths = list(
        (fixture["root"] / transitioner.EVIDENCE_ROOT_RELATIVE).glob(
            f"*/{transitioner.EVIDENCE_FILENAME}"
        )
    )
    assert len(evidence_paths) == 1
    assert json.loads(evidence_paths[0].read_text())["state"] == "aborted_before_commit"


def test_exclusive_snapshot_write_removes_only_partial_created_inode(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    target = parent / "snapshot.json"
    with mock.patch.object(transitioner.os, "fsync", side_effect=OSError("fault")):
        with pytest.raises(transitioner.TransitionError, match="cannot create"):
            transitioner._write_private_exclusive(target, b"payload\n", "snapshot")
    assert not target.exists()


def _prepare_recovery_fixture(
    fixture: dict[str, Any], *, install_after: bool
) -> tuple[str, dict[str, Any]]:
    arguments = _arguments(fixture)
    root = fixture["root"]
    with mock.patch.multiple(transitioner, **_constants(fixture)):
        with transitioner.transaction._artifact_transaction_lock(root):
            raw, proof = transitioner._read_transition_inputs(
                artifact_root=root,
                old_manifest_snapshot=fixture["old_manifest_path"],
                old_source_contract_snapshot=fixture["old_source_path"],
                expected_old_realization_sha256=arguments[
                    "expected_old_realization_sha256"
                ],
                updated_at_utc=arguments["updated_at_utc"],
            )
            transaction_id = "20990101T000000000000Z"
            run_dir = transitioner._create_evidence_directory(root, transaction_id)
            transitioner._write_snapshots(run_dir, raw, proof["proposal_raw"])
            evidence = transitioner._new_evidence(
                transaction_id=transaction_id,
                proof=proof,
                prepared_at_utc="2026-07-11T01:30:00Z",
            )
            transitioner._write_private_exclusive(
                run_dir / transitioner.EVIDENCE_FILENAME,
                _json_bytes(evidence),
                "test recovery evidence",
            )
            if install_after:
                _write(fixture["realization_path"], proof["proposal_raw"], 0o600)
    return transaction_id, proof


@pytest.mark.parametrize(
    ("install_after", "expected_state"),
    ((False, "aborted_by_recovery"), (True, "committed")),
)
def test_recovery_terminalizes_only_exact_before_or_after(
    tmp_path: Path, install_after: bool, expected_state: str
) -> None:
    fixture = _fixture(tmp_path)
    transaction_id, proof = _prepare_recovery_fixture(
        fixture, install_after=install_after
    )
    with mock.patch.multiple(transitioner, **_constants(fixture)):
        result = transitioner.recover_mapanything_authority(
            artifact_root=fixture["root"], transaction_id=transaction_id
        )
    assert result["state"] == expected_state
    expected = proof["proposal_raw"] if install_after else fixture["realization_raw"]
    assert fixture["realization_path"].read_bytes() == expected


def test_recovery_rejects_tampered_snapshot_and_ambiguous_realization(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    transaction_id, _proof = _prepare_recovery_fixture(fixture, install_after=False)
    run_dir = fixture["root"] / transitioner.EVIDENCE_ROOT_RELATIVE / transaction_id
    snapshot = run_dir / transitioner.OLD_SOURCE_SNAPSHOT_FILENAME
    _write(snapshot, snapshot.read_bytes() + b" ", 0o600)
    with mock.patch.multiple(transitioner, **_constants(fixture)):
        with pytest.raises(transitioner.TransitionError, match="hash mismatch"):
            transitioner.recover_mapanything_authority(
                artifact_root=fixture["root"], transaction_id=transaction_id
            )

    fixture = _fixture(tmp_path / "ambiguous")
    transaction_id, _proof = _prepare_recovery_fixture(fixture, install_after=False)
    _write(fixture["realization_path"], b"{}\n", 0o600)
    with mock.patch.multiple(transitioner, **_constants(fixture)):
        with pytest.raises(transitioner.TransitionError, match="neither exact"):
            transitioner.recover_mapanything_authority(
                artifact_root=fixture["root"], transaction_id=transaction_id
            )
