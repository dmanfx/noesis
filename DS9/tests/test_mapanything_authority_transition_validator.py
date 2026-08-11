from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest import mock

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transitioner = _load(
    "ds9_mapanything_transition_for_validator",
    "DS9/scripts/transition_mapanything_fp32_authority.py",
)
validator = _load(
    "ds9_mapanything_transition_validator",
    "DS9/scripts/validate_asset_manifest.py",
)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, raw: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)


def _engine_row(artifact_id: str) -> dict[str, Any]:
    if artifact_id == validator.MAPANYTHING_ARTIFACT_ID:
        return {
            "id": artifact_id,
            "kind": "tensorrt_engine",
            "output": validator.MAPANYTHING_OLD_OUTPUT,
            "compatibility": {"precision": "fp16", "batch": 3},
        }
    return {
        "id": artifact_id,
        "kind": "tensorrt_engine",
        "output": f"DS9/models/engines/{artifact_id.removeprefix('engine.')}.plan",
        "compatibility": {"precision": "fp16"},
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "artifacts"
    root.mkdir(mode=0o750, parents=True)
    _write(root / transitioner.LOCK_FILENAME, b"", 0o600)
    (root / validator.SOURCE_CONTRACT_REBASE_ROOT).mkdir(mode=0o700)
    (root / validator.MANIFEST_REBASE_ROOT).mkdir(mode=0o755)

    output = root / "models/engines/mapanything_images_294x518_b3_fp16.plan"
    output_raw = b"historic-mapanything-fp16"
    _write(output, output_raw, 0o600)

    old_manifest = {
        "schema_version": 2,
        "target": {"platform": "fixture"},
        "artifacts": [
            _engine_row(artifact_id)
            for artifact_id in validator.ENGINE_NAME_BY_ARTIFACT_ID
        ],
    }
    new_manifest = copy.deepcopy(old_manifest)
    new_map = next(
        row
        for row in new_manifest["artifacts"]
        if row["id"] == validator.MAPANYTHING_ARTIFACT_ID
    )
    new_map["output"] = validator.MAPANYTHING_NEW_OUTPUT
    new_map["compatibility"]["precision"] = "fp32"
    old_source = {
        "schema_version": 1,
        "contracts": {
            name: {"token": "unchanged"}
            for name in validator.ENGINE_NAME_BY_ARTIFACT_ID.values()
        },
    }
    old_source["contracts"][validator.MAPANYTHING_CONTRACT_NAME] = {
        "maintenance_build": {"precision_arg": "--fp16"}
    }
    new_source = copy.deepcopy(old_source)
    new_source["contracts"][validator.MAPANYTHING_CONTRACT_NAME] = {
        "maintenance_build": {"precision": "fp32"},
        "quality_gate": {"contract": "fixture"},
    }

    checkpoint = tmp_path / "checkpoint"
    repo = tmp_path / "repo"
    old_manifest_path = checkpoint / validator.BASE_MANIFEST_PATH
    old_source_path = checkpoint / validator.SOURCE_CONTRACT_PATH
    new_manifest_path = repo / validator.BASE_MANIFEST_PATH
    new_source_path = repo / validator.SOURCE_CONTRACT_PATH
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
            "output_sha256": _sha(output_raw),
            "source_sha256": "a" * 64,
            "maintenance": {
                "output_size_bytes": len(output_raw),
                "precision": "fp16",
            },
        },
    }
    unchanged_id = "engine.yolo26_detect_m"
    unchanged_record = {
        "state": "validated",
        "provenance": {"opaque": "unchanged"},
    }
    realization = {
        "schema_version": 1,
        "contract": validator.REALIZATION_CONTRACT,
        "base_manifest": {
            "path": validator.BASE_MANIFEST_PATH,
            "sha256": _sha(old_manifest_raw),
        },
        "source_contracts": {
            "path": validator.SOURCE_CONTRACT_PATH,
            "sha256": _sha(old_source_raw),
        },
        "created_at_utc": "2026-07-11T00:00:00Z",
        "updated_at_utc": "2026-07-11T01:00:00Z",
        "artifacts": {
            validator.MAPANYTHING_ARTIFACT_ID: map_record,
            unchanged_id: unchanged_record,
        },
    }
    realization_raw = _json_bytes(realization)
    realization_path = root / validator.REALIZATION_FILENAME
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
        "realization_path": realization_path,
        "realization_raw": realization_raw,
        "map_record": map_record,
        "map_record_hash": _sha(_json_bytes(map_record)),
        "unchanged_id": unchanged_id,
        "output": output,
        "output_raw": output_raw,
    }


def _transition_constants(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "REPO_ROOT": fixture["repo"],
        "NEW_MANIFEST_AUTHORITY": fixture["new_manifest_path"],
        "NEW_SOURCE_CONTRACT_AUTHORITY": fixture["new_source_path"],
        "OLD_MANIFEST_SHA256": _sha(fixture["old_manifest_raw"]),
        "NEW_MANIFEST_SHA256": _sha(fixture["new_manifest_raw"]),
        "OLD_SOURCE_CONTRACTS_SHA256": _sha(fixture["old_source_raw"]),
        "NEW_SOURCE_CONTRACTS_SHA256": _sha(fixture["new_source_raw"]),
        "OLD_MAP_RECORD_SHA256": fixture["map_record_hash"],
        "OLD_FP16_OUTPUT_SHA256": _sha(fixture["output_raw"]),
        "OLD_FP16_OUTPUT_SIZE_BYTES": len(fixture["output_raw"]),
    }


def _validator_constants(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "MAPANYTHING_OLD_MANIFEST_SHA256": _sha(fixture["old_manifest_raw"]),
        "MAPANYTHING_NEW_MANIFEST_SHA256": _sha(fixture["new_manifest_raw"]),
        "MAPANYTHING_OLD_SOURCE_CONTRACTS_SHA256": _sha(fixture["old_source_raw"]),
        "MAPANYTHING_NEW_SOURCE_CONTRACTS_SHA256": _sha(fixture["new_source_raw"]),
        "MAPANYTHING_OLD_REALIZED_RECORD_SHA256": fixture["map_record_hash"],
        "MAPANYTHING_OLD_OUTPUT_SHA256": _sha(fixture["output_raw"]),
        "MAPANYTHING_OLD_OUTPUT_SIZE_BYTES": len(fixture["output_raw"]),
    }


def _patches(fixture: dict[str, Any]) -> Iterator[None]:
    stack = ExitStack()
    stack.enter_context(
        mock.patch.multiple(transitioner, **_transition_constants(fixture))
    )
    stack.enter_context(mock.patch.multiple(validator, **_validator_constants(fixture)))
    return stack


def _apply(fixture: dict[str, Any]) -> Path:
    kwargs = {
        "artifact_root": fixture["root"],
        "old_manifest_snapshot": fixture["old_manifest_path"],
        "old_source_contract_snapshot": fixture["old_source_path"],
        "expected_old_realization_sha256": _sha(fixture["realization_raw"]),
        "expected_new_realization_sha256": None,
        "expected_plan_sha256": None,
        "updated_at_utc": "2026-07-11T02:00:00Z",
        "dry_run": True,
    }
    with _patches(fixture):
        dry = transitioner.transition_mapanything_authority(**kwargs)
        result = transitioner.transition_mapanything_authority(
            **{
                **kwargs,
                "dry_run": False,
                "expected_new_realization_sha256": dry["new_realization_sha256"],
                "expected_plan_sha256": dry["transition_plan_sha256"],
            }
        )
    return (fixture["root"] / result["evidence"]).parent


def _call(fixture: dict[str, Any]) -> dict[str, frozenset[str]]:
    realization_raw = fixture["realization_path"].read_bytes()
    realization = json.loads(realization_raw)
    with _patches(fixture):
        return validator._validate_source_contract_rebase_chain(
            artifact_root=fixture["root"],
            current_source_contracts_path=fixture["new_source_path"],
            current_source_contracts_sha256=_sha(
                fixture["new_source_path"].read_bytes()
            ),
            current_manifest_path=fixture["new_manifest_path"],
            current_manifest_sha256=_sha(fixture["new_manifest_path"].read_bytes()),
            current_realization=realization,
            current_realization_sha256=_sha(realization_raw),
        )


def _rewrite_evidence(run_dir: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = run_dir / validator.MAPANYTHING_TRANSITION_EVIDENCE_FILENAME
    payload = json.loads(path.read_text())
    mutate(payload)
    _write(path, _json_bytes(payload), 0o600)


def test_combined_edge_preserves_only_unchanged_historic_source_equivalence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _apply(fixture)

    accepted = _call(fixture)

    assert accepted[fixture["unchanged_id"]] == frozenset(
        {_sha(fixture["old_source_raw"]), _sha(fixture["new_source_raw"])}
    )
    assert validator.MAPANYTHING_ARTIFACT_ID not in accepted

    current = json.loads(fixture["realization_path"].read_text())
    current["artifacts"][validator.MAPANYTHING_ARTIFACT_ID] = {
        "state": "validated",
        "provenance": {},
    }
    current["updated_at_utc"] = "2026-07-11T03:00:00Z"
    _write(fixture["realization_path"], _json_bytes(current), 0o600)

    accepted = _call(fixture)

    assert accepted[validator.MAPANYTHING_ARTIFACT_ID] == frozenset(
        {_sha(fixture["new_source_raw"])}
    )
    assert accepted[fixture["unchanged_id"]] == frozenset(
        {_sha(fixture["old_source_raw"]), _sha(fixture["new_source_raw"])}
    )


@pytest.mark.parametrize(
    ("damage", "message"),
    (
        ("unresolved", "unresolved"),
        ("wrong_diff", "manifest diff"),
        ("wrong_record_hash", "retired output|retirement|record|plan"),
        ("wrong_authority_hash", "authority endpoints"),
        ("wrong_inventory", "retirement inventory"),
        ("duplicate_json", "duplicate"),
        ("tampered_snapshot", "snapshot hash"),
    ),
)
def test_tampered_or_unresolved_combined_evidence_is_rejected(
    tmp_path: Path, damage: str, message: str
) -> None:
    fixture = _fixture(tmp_path)
    run_dir = _apply(fixture)
    evidence_path = run_dir / validator.MAPANYTHING_TRANSITION_EVIDENCE_FILENAME
    if damage == "unresolved":
        _rewrite_evidence(run_dir, lambda payload: payload.update(state="prepared"))
    elif damage == "wrong_diff":
        _rewrite_evidence(
            run_dir,
            lambda payload: payload["manifest_diff"].update(changed_paths=["output"]),
        )
    elif damage == "wrong_record_hash":
        _rewrite_evidence(
            run_dir,
            lambda payload: payload["retired_artifact"].update(
                realized_record_sha256="0" * 64
            ),
        )
    elif damage == "wrong_authority_hash":
        _rewrite_evidence(
            run_dir,
            lambda payload: payload["authorities"]["manifest"].update(
                old_sha256="0" * 64
            ),
        )
    elif damage == "wrong_inventory":
        _rewrite_evidence(
            run_dir,
            lambda payload: payload["realization"].update(
                realized_artifact_ids_after=[]
            ),
        )
    elif damage == "duplicate_json":
        raw = evidence_path.read_bytes().replace(
            b'{\n  "aborted_at_utc"',
            b'{\n  "schema_version": 1,\n  "aborted_at_utc"',
            1,
        )
        _write(evidence_path, raw, 0o600)
    else:
        snapshot = run_dir / "old_asset_manifest.yaml"
        _write(snapshot, snapshot.read_bytes() + b"\n", 0o600)

    with pytest.raises(ValueError, match=message):
        _call(fixture)


@pytest.mark.parametrize("damage", ("missing", "symlink", "hardlink"))
def test_historic_fp16_bytes_remain_mandatory_and_single_link(
    tmp_path: Path, damage: str
) -> None:
    fixture = _fixture(tmp_path)
    _apply(fixture)
    if damage == "missing":
        fixture["output"].unlink()
    elif damage == "symlink":
        target = fixture["output"].with_name("target.plan")
        fixture["output"].rename(target)
        fixture["output"].symlink_to(target)
    else:
        os.link(fixture["output"], fixture["output"].with_name("alias.plan"))
    with pytest.raises(ValueError, match="missing|unsafe|ownership|link|symlink"):
        _call(fixture)


def test_duplicate_authority_edge_branches_and_disconnected_graph_refuses(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    run_dir = _apply(fixture)
    duplicate = run_dir.with_name("20990101T000000000000Z")
    shutil.copytree(run_dir, duplicate)
    duplicate.chmod(0o700)
    _rewrite_evidence(
        duplicate,
        lambda payload: payload.update(transaction_id=duplicate.name),
    )
    with pytest.raises(ValueError, match="branches|predecessors"):
        _call(fixture)

    shutil.rmtree(duplicate)
    with _patches(fixture):
        real_edges = validator._load_mapanything_transition_edges(fixture["root"])
        fake = dict(real_edges[0])
        fake.update(
            {
                "edge_uid": "mapanything:disconnected",
                "transaction_id": "disconnected",
                "source_old_hash": "1" * 64,
                "source_new_hash": "2" * 64,
            }
        )
        realization_raw = fixture["realization_path"].read_bytes()
        with mock.patch.object(
            validator,
            "_load_mapanything_transition_edges",
            return_value=(*real_edges, fake),
        ):
            with pytest.raises(ValueError, match="ambiguous roots|disconnected"):
                validator._validate_source_contract_rebase_chain(
                    artifact_root=fixture["root"],
                    current_source_contracts_path=fixture["new_source_path"],
                    current_source_contracts_sha256=_sha(fixture["new_source_raw"]),
                    current_manifest_path=fixture["new_manifest_path"],
                    current_manifest_sha256=_sha(fixture["new_manifest_raw"]),
                    current_realization=json.loads(realization_raw),
                    current_realization_sha256=_sha(realization_raw),
                )


def test_current_authority_drift_and_preserved_record_tampering_are_rejected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _apply(fixture)
    current_source = json.loads(fixture["new_source_path"].read_text())
    current_source["contracts"]["yolo26_m"]["later"] = True
    drifted_raw = _json_bytes(current_source)
    _write(fixture["new_source_path"], drifted_raw, 0o644)
    with pytest.raises(
        ValueError,
        match="source-contract binding is wrong|does not reach current authority",
    ):
        _call(fixture)

    fixture = _fixture(tmp_path / "record")
    run_dir = _apply(fixture)
    snapshot = run_dir / "asset_realization.after.json"
    after = json.loads(snapshot.read_text())
    after["artifacts"][fixture["unchanged_id"]]["provenance"]["opaque"] = "tampered"
    _write(snapshot, _json_bytes(after), 0o600)
    with pytest.raises(ValueError, match="snapshot hash|preserved realization"):
        _call(fixture)
