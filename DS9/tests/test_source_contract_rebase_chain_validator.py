from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "DS9/scripts/validate_asset_manifest.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("ds9_chain_validator", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _yaml_bytes(payload: dict[str, Any]) -> bytes:
    return yaml.safe_dump(payload, sort_keys=False).encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _private(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for parent in [path.parent, *path.parent.parents]:
        if parent.name in {"artifacts", ""}:
            break
        parent.chmod(0o700)
    path.write_bytes(raw)
    path.chmod(0o600)


def _contract_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contracts": {
            name: {"token": f"{name}:v0"}
            for name in validator.ENGINE_NAME_BY_ARTIFACT_ID.values()
        },
    }


def _manifest(parser_token: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "updated_at": "2026-07-11",
        "target": {"platform": "fixture"},
        "runtime": {
            "entrypoint": "DS9/noesis/ds9_runtime.py",
            "image": copy.deepcopy(validator.RUNTIME_IMAGE_AUTHORITY),
        },
        "policy": {"fixture": True},
        "artifacts": [
            {"id": artifact_id, "kind": "tensorrt_engine", "token": "fixed"}
            for artifact_id in validator.ENGINE_NAME_BY_ARTIFACT_ID
        ]
        + [
            {
                "id": "parser.fixture",
                "kind": "nvinfer_parser",
                "token": parser_token,
            }
        ],
    }


def _realization(
    *, source_hash: str, manifest_hash: str, updated_at: str
) -> dict[str, Any]:
    realized_ids = ("engine.mapanything", "engine.yolo26_detect_m")
    return {
        "schema_version": 1,
        "contract": validator.REALIZATION_CONTRACT,
        "base_manifest": {
            "path": validator.BASE_MANIFEST_PATH,
            "sha256": manifest_hash,
        },
        "source_contracts": {
            "path": validator.SOURCE_CONTRACT_PATH,
            "sha256": source_hash,
        },
        "created_at_utc": "2026-07-11T00:00:00Z",
        "updated_at_utc": updated_at,
        "artifacts": {
            artifact_id: {"state": "validated", "provenance": {}}
            for artifact_id in realized_ids
        },
    }


def _source_checks() -> dict[str, bool]:
    return {key: True for key in validator._SOURCE_REBASE_SEMANTIC_CHECK_KEYS}


def _write_source_edge(
    fixture: dict[str, Any],
    *,
    index: int,
    old_document: dict[str, Any],
    new_hash: str,
    changed: list[str],
    base_hash: str,
    old_realization_hash: str,
    new_realization_hash: str,
    updated_before: str,
    updated_after: str,
) -> Path:
    root = fixture["artifact_root"]
    snapshot = (
        root
        / validator.SOURCE_CONTRACT_REBASE_ROOT
        / validator.SOURCE_CONTRACT_REBASE_INPUT_ROOT
        / f"snapshot-{index}"
        / "old.json"
    )
    old_raw = _json_bytes(old_document)
    _private(snapshot, old_raw)
    transaction_id = f"20990101T0000{index:02d}000000Z"
    run_dir = root / validator.SOURCE_CONTRACT_REBASE_ROOT / transaction_id
    run_dir.mkdir(mode=0o700)
    artifact_by_engine = {
        name: artifact_id
        for artifact_id, name in validator.ENGINE_NAME_BY_ARTIFACT_ID.items()
    }
    evidence = {
        "schema_version": 1,
        "contract": validator.SOURCE_CONTRACT_REBASE_CONTRACT,
        "transaction_id": transaction_id,
        "state": "committed",
        "prepared_at_utc": f"2026-07-11T05:{index:02d}:00Z",
        "committed_at_utc": f"2026-07-11T05:{index:02d}:01Z",
        "old_source_contracts": {
            "path": snapshot.relative_to(root).as_posix(),
            "sha256": _sha(old_raw),
        },
        "new_source_contracts": {
            "path": validator.SOURCE_CONTRACT_PATH,
            "sha256": new_hash,
        },
        "base_manifest": {
            "path": validator.BASE_MANIFEST_PATH,
            "sha256": base_hash,
        },
        "realization": {
            "path": validator.REALIZATION_FILENAME,
            "old_sha256": old_realization_hash,
            "new_sha256": new_realization_hash,
            "updated_at_utc_before": updated_before,
            "updated_at_utc_after": updated_after,
        },
        "changed_contracts": sorted(changed),
        "mapped_unrealized_artifact_ids": sorted(
            artifact_by_engine[name] for name in changed
        ),
        "realized_engine_ids": sorted(fixture["realization"]["artifacts"]),
        "mutation_paths": ["source_contracts.sha256", "updated_at_utc"],
        "semantic_checks": _source_checks(),
    }
    evidence_path = run_dir / validator.SOURCE_CONTRACT_REBASE_EVIDENCE_FILENAME
    _private(evidence_path, _json_bytes(evidence))
    fixture["source_evidence"].append(evidence_path)
    fixture["source_snapshots"].append(snapshot)
    return evidence_path


def _fixture(tmp_path: Path, *, source_edges: int = 1) -> dict[str, Any]:
    root = tmp_path / "artifacts"
    root.mkdir(mode=0o750, parents=True)
    source_root = root / validator.SOURCE_CONTRACT_REBASE_ROOT
    source_root.mkdir(mode=0o700)
    (source_root / validator.SOURCE_CONTRACT_REBASE_INPUT_ROOT).mkdir(mode=0o700)

    documents = [_contract_document()]
    changed_names: list[str] = []
    for index in range(source_edges):
        following = copy.deepcopy(documents[-1])
        changed_name = (
            "wholebody49_x_boxes" if index % 2 == 0 else "wholebody49_s_masks"
        )
        following["contracts"][changed_name]["token"] = f"changed:{index}"
        documents.append(following)
        changed_names.append(changed_name)

    source_path = tmp_path / "repo/DS9/config/engine_source_contracts.json"
    source_path.parent.mkdir(parents=True)
    source_raw = _json_bytes(documents[-1])
    source_path.write_bytes(source_raw)
    source_path.chmod(0o644)

    manifest_path = tmp_path / "repo/DS9/asset_manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_document = _manifest("current")
    manifest_raw = _yaml_bytes(manifest_document)
    manifest_path.write_bytes(manifest_raw)
    manifest_path.chmod(0o644)

    updated_at = f"2026-07-11T02:{source_edges:02d}:00Z"
    realization = _realization(
        source_hash=_sha(source_raw),
        manifest_hash=_sha(manifest_raw),
        updated_at=updated_at,
    )
    realization_raw = _json_bytes(realization)
    fixture: dict[str, Any] = {
        "artifact_root": root,
        "source_path": source_path,
        "source_document": documents[-1],
        "source_hash": _sha(source_raw),
        "manifest_path": manifest_path,
        "manifest_document": manifest_document,
        "manifest_hash": _sha(manifest_raw),
        "realization": realization,
        "realization_hash": _sha(realization_raw),
        "documents": documents,
        "source_evidence": [],
        "source_snapshots": [],
    }
    realization_boundaries = [f"{index + 1:064x}" for index in range(source_edges)]
    realization_boundaries.append(fixture["realization_hash"])
    for index in range(source_edges):
        _write_source_edge(
            fixture,
            index=index,
            old_document=documents[index],
            new_hash=_sha(_json_bytes(documents[index + 1])),
            changed=[changed_names[index]],
            base_hash=fixture["manifest_hash"],
            old_realization_hash=realization_boundaries[index],
            new_realization_hash=realization_boundaries[index + 1],
            updated_before=f"2026-07-11T02:{index:02d}:00Z",
            updated_after=f"2026-07-11T02:{index + 1:02d}:00Z",
        )
    return fixture


def _call(fixture: dict[str, Any]) -> dict[str, frozenset[str]]:
    return validator._validate_source_contract_rebase_chain(
        artifact_root=fixture["artifact_root"],
        current_source_contracts_path=fixture["source_path"],
        current_source_contracts_sha256=fixture["source_hash"],
        current_manifest_path=fixture["manifest_path"],
        current_manifest_sha256=fixture["manifest_hash"],
        current_realization=fixture["realization"],
        current_realization_sha256=fixture["realization_hash"],
    )


def _rewrite_json(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(path.read_text())
    mutate(payload)
    _private(path, _json_bytes(payload))


def _install_manifest_bridge(fixture: dict[str, Any], *, edge_count: int) -> list[Path]:
    root = fixture["artifact_root"]
    manifest_root = root / validator.MANIFEST_REBASE_ROOT
    manifest_root.mkdir(mode=0o755)
    documents = [_manifest(f"historic-{index}") for index in range(edge_count)]
    documents.append(fixture["manifest_document"])
    hashes = [_sha(_yaml_bytes(document)) for document in documents]
    fixture["realization"]["updated_at_utc"] = "2026-07-11T04:00:00Z"
    fixture["realization_hash"] = _sha(_json_bytes(fixture["realization"]))

    source_evidence = fixture["source_evidence"][-1]
    bridge_start_hash = "a" * 64
    _rewrite_json(
        source_evidence,
        lambda payload: (
            payload["base_manifest"].update(sha256=hashes[0]),
            payload["realization"].update(
                new_sha256=bridge_start_hash,
                updated_at_utc_after="2026-07-11T03:00:00Z",
            ),
        ),
    )
    prior_realization_hash = bridge_start_hash
    prior_time = "2026-07-11T03:00:00Z"
    result: list[Path] = []
    for index in range(edge_count):
        snapshot = manifest_root / "inputs" / f"snapshot-{index}" / "old.yaml"
        old_raw = _yaml_bytes(documents[index])
        _private(snapshot, old_raw)
        transaction_id = f"20990202T0000{index:02d}000000Z"
        run_dir = manifest_root / transaction_id
        run_dir.mkdir(mode=0o700)
        next_realization_hash = (
            fixture["realization_hash"]
            if index + 1 == edge_count
            else f"{index + 11:064x}"
        )
        next_time = (
            fixture["realization"]["updated_at_utc"]
            if index + 1 == edge_count
            else f"2026-07-11T03:{index + 1:02d}:00Z"
        )
        diff, runtime = validator._recompute_manifest_rebase_semantics(
            documents[index], documents[index + 1]
        )
        evidence = {
            "schema_version": 1,
            "contract": validator.MANIFEST_REBASE_CONTRACT,
            "transaction_id": transaction_id,
            "state": "committed",
            "prepared_at_utc": f"2026-07-11T06:{index:02d}:00Z",
            "committed_at_utc": f"2026-07-11T06:{index:02d}:01Z",
            "old_manifest": {
                "path": snapshot.relative_to(root).as_posix(),
                "sha256": hashes[index],
            },
            "new_manifest": {
                "path": validator.BASE_MANIFEST_PATH,
                "sha256": hashes[index + 1],
            },
            "source_contracts": {
                "path": validator.SOURCE_CONTRACT_PATH,
                "sha256": fixture["source_hash"],
            },
            "realization": {
                "path": validator.REALIZATION_FILENAME,
                "old_sha256": prior_realization_hash,
                "new_sha256": next_realization_hash,
                "updated_at_utc_before": prior_time,
                "updated_at_utc_after": next_time,
            },
            "mutation_paths": ["base_manifest.sha256", "updated_at_utc"],
            "realized_engine_ids": sorted(fixture["realization"]["artifacts"]),
            "semantic_checks": {
                "target_unchanged": True,
                "source_contracts_unchanged": True,
                "all_engine_artifacts_unchanged": True,
                "realized_artifacts_unchanged": True,
                "runtime_fields_outside_image_unchanged": True,
                "runtime_image_change_is_exact_reviewed_authority": True,
            },
            "manifest_diff": diff,
            "runtime_image_authority": runtime,
        }
        evidence_path = run_dir / validator.MANIFEST_REBASE_EVIDENCE_FILENAME
        _private(evidence_path, _json_bytes(evidence))
        result.append(evidence_path)
        prior_realization_hash = next_realization_hash
        prior_time = next_time
    return result


def test_valid_single_and_multi_source_chains_accept_only_equivalent_hashes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, source_edges=2)
    accepted = _call(fixture)
    expected = {
        _sha(_json_bytes(document)) for document in fixture["documents"]
    }
    assert accepted["engine.mapanything"] == expected
    assert accepted["engine.yolo26_detect_m"] == expected


def test_current_realization_may_monotonically_add_engine_after_rebase(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["realization"]["artifacts"]["engine.wholebody49_x_boxes"] = {
        "state": "validated",
        "provenance": {},
    }
    fixture["realization"]["updated_at_utc"] = "2026-07-11T03:00:00Z"
    fixture["realization_hash"] = _sha(_json_bytes(fixture["realization"]))

    accepted = _call(fixture)

    assert accepted["engine.wholebody49_x_boxes"] == frozenset(
        {fixture["source_hash"]}
    )


def _install_manifest_bridge_after_monotonic_engine_gap(
    fixture: dict[str, Any],
) -> Path:
    fixture["realization"]["artifacts"]["engine.wholebody49_x_boxes"] = {
        "state": "validated",
        "provenance": {},
    }
    evidence = _install_manifest_bridge(fixture, edge_count=1)[0]
    _rewrite_json(
        evidence,
        lambda payload: payload["realization"].update(
            old_sha256="b" * 64,
            updated_at_utc_before="2026-07-11T03:30:00Z",
        ),
    )
    return evidence


def test_later_manifest_rebase_accepts_monotonic_engine_maintenance_gap(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _install_manifest_bridge_after_monotonic_engine_gap(fixture)

    accepted = _call(fixture)

    assert accepted["engine.wholebody49_x_boxes"] == frozenset(
        {fixture["source_hash"]}
    )


@pytest.mark.parametrize(
    ("damage", "message"),
    (
        ("dropped_engine", "drops previously realized engines"),
        ("overlapping_time", "gap timestamps are not strictly monotonic"),
        ("backdated_evidence", "gap evidence chronology is not monotonic"),
    ),
)
def test_engine_maintenance_gap_is_fail_closed(
    tmp_path: Path,
    damage: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = _install_manifest_bridge_after_monotonic_engine_gap(fixture)
    if damage == "dropped_engine":
        _rewrite_json(
            evidence,
            lambda payload: payload.update(
                realized_engine_ids=["engine.mapanything"]
            ),
        )
    elif damage == "overlapping_time":
        _rewrite_json(
            evidence,
            lambda payload: payload["realization"].update(
                updated_at_utc_before="2026-07-11T02:59:00Z"
            ),
        )
    else:
        _rewrite_json(
            evidence,
            lambda payload: payload.update(
                prepared_at_utc="2026-07-11T04:59:59Z"
            ),
        )

    with pytest.raises(ValueError, match=message):
        _call(fixture)


@pytest.mark.parametrize("damage", ("stale_timestamp", "dropped_realized_engine"))
def test_current_realization_must_monotonically_succeed_rebase_terminal(
    tmp_path: Path,
    damage: str,
) -> None:
    fixture = _fixture(tmp_path)
    if damage == "stale_timestamp":
        fixture["realization"]["created_at_utc"] = "2026-07-10T23:59:59Z"
    else:
        del fixture["realization"]["artifacts"]["engine.mapanything"]
        fixture["realization"]["updated_at_utc"] = "2026-07-11T03:00:00Z"
    fixture["realization_hash"] = _sha(_json_bytes(fixture["realization"]))

    with pytest.raises(
        ValueError, match="unknown current engines|monotonically succeed"
    ):
        _call(fixture)


def test_legitimate_multi_manifest_rebase_bridge_is_verified(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _install_manifest_bridge(fixture, edge_count=2)
    accepted = _call(fixture)
    assert _sha(_json_bytes(fixture["documents"][0])) in accepted[
        "engine.yolo26_detect_m"
    ]


@pytest.mark.parametrize("damage", ("missing_root", "deleted_snapshot", "bad_mode", "symlink"))
def test_private_source_evidence_damage_is_fail_closed(
    tmp_path: Path, damage: str
) -> None:
    fixture = _fixture(tmp_path)
    if damage == "missing_root":
        os.rename(
            fixture["artifact_root"] / validator.SOURCE_CONTRACT_REBASE_ROOT,
            fixture["artifact_root"] / "removed",
        )
    elif damage == "deleted_snapshot":
        fixture["source_snapshots"][0].unlink()
    elif damage == "bad_mode":
        fixture["source_evidence"][0].chmod(0o644)
    else:
        snapshot = fixture["source_snapshots"][0]
        target = snapshot.with_name("target.json")
        snapshot.rename(target)
        snapshot.symlink_to(target)
    with pytest.raises(ValueError, match="missing|mode 0600|symlink"):
        _call(fixture)


def test_duplicate_and_unresolved_source_evidence_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = fixture["source_evidence"][0]
    evidence.write_bytes(
        evidence.read_bytes().replace(
            b'{\n  "base_manifest"',
            b'{\n  "schema_version": 1,\n  "base_manifest"',
            1,
        )
    )
    evidence.chmod(0o600)
    with pytest.raises(ValueError, match="duplicate"):
        _call(fixture)

    fixture = _fixture(tmp_path / "unresolved")
    _rewrite_json(
        fixture["source_evidence"][0],
        lambda payload: payload.update(state="prepared"),
    )
    with pytest.raises(ValueError, match="unresolved"):
        _call(fixture)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda payload: payload["base_manifest"].update(sha256="0" * 64),
            "manifest",
        ),
        (
            lambda payload: payload["realization"].update(new_sha256="0" * 64),
            "disconnected|terminal|monotonically succeed",
        ),
        (
            lambda payload: payload.update(
                realized_engine_ids=["engine.mapanything"]
            ),
            "exactly bound|inconsistent",
        ),
        (
            lambda payload: payload["realization"].update(
                updated_at_utc_after="2026-07-11T09:00:00Z"
            ),
            "exactly bound|timestamp",
        ),
    ),
)
def test_base_realization_subset_and_timestamp_forgery_are_rejected(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    _rewrite_json(fixture["source_evidence"][0], mutate)
    with pytest.raises(ValueError, match=message):
        _call(fixture)


def test_changed_realized_contract_and_false_allowlist_are_rejected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    artifact_by_name = {
        name: artifact_id
        for artifact_id, name in validator.ENGINE_NAME_BY_ARTIFACT_ID.items()
    }
    _rewrite_json(
        fixture["source_evidence"][0],
        lambda payload: payload.update(
            changed_contracts=["yolo26_m"],
            mapped_unrealized_artifact_ids=[artifact_by_name["yolo26_m"]],
        ),
    )
    with pytest.raises(ValueError, match="changed a realized"):
        _call(fixture)

    fixture = _fixture(tmp_path / "allowlist")
    _rewrite_json(
        fixture["source_evidence"][0],
        lambda payload: payload.update(
            changed_contracts=["wholebody49_s_masks"],
            mapped_unrealized_artifact_ids=[artifact_by_name["wholebody49_s_masks"]],
        ),
    )
    with pytest.raises(ValueError, match="allowlist"):
        _call(fixture)


@pytest.mark.parametrize("topology", ("branch", "predecessor", "cycle"))
def test_source_chain_topology_is_unambiguous(tmp_path: Path, topology: str) -> None:
    fixture = _fixture(tmp_path)
    old_document = fixture["documents"][0]
    current_document = fixture["documents"][1]
    if topology == "branch":
        branch = copy.deepcopy(old_document)
        branch["contracts"]["wholebody49_s_masks"]["token"] = "branch"
        edge_old, edge_new, changed = old_document, _sha(_json_bytes(branch)), [
            "wholebody49_s_masks"
        ]
    elif topology == "predecessor":
        predecessor = copy.deepcopy(current_document)
        predecessor["contracts"]["wholebody49_s_masks"]["token"] = "prior"
        edge_old, edge_new, changed = predecessor, fixture["source_hash"], [
            "wholebody49_s_masks"
        ]
    else:
        edge_old = current_document
        edge_new = _sha(_json_bytes(old_document))
        changed = ["wholebody49_x_boxes"]
    _write_source_edge(
        fixture,
        index=8,
        old_document=edge_old,
        new_hash=edge_new,
        changed=changed,
        base_hash=fixture["manifest_hash"],
        old_realization_hash="e" * 64,
        new_realization_hash="f" * 64,
        updated_before="2026-07-11T04:00:00Z",
        updated_after="2026-07-11T04:01:00Z",
    )
    with pytest.raises(ValueError, match="branches|predecessors|cyclic|ambiguous"):
        _call(fixture)


@pytest.mark.parametrize("topology", ("duplicate_successor", "ambiguous_predecessor"))
def test_combined_source_manifest_realization_graph_is_unambiguous(
    tmp_path: Path, topology: str
) -> None:
    fixture = _fixture(tmp_path)
    manifest_evidence = _install_manifest_bridge(fixture, edge_count=1)[0]
    source_evidence = fixture["source_evidence"][0]
    source_payload = json.loads(source_evidence.read_text())
    if topology == "duplicate_successor":
        duplicate_old = source_payload["realization"]["old_sha256"]
        _rewrite_json(
            manifest_evidence,
            lambda payload: payload["realization"].update(
                old_sha256=duplicate_old
            ),
        )
        message = "branches"
    else:
        _rewrite_json(
            source_evidence,
            lambda payload: payload["realization"].update(
                new_sha256=fixture["realization_hash"]
            ),
        )
        message = "predecessors"
    with pytest.raises(ValueError, match=message):
        _call(fixture)


def test_manifest_semantic_diff_and_runtime_evidence_are_recomputed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = _install_manifest_bridge(fixture, edge_count=1)[0]
    _rewrite_json(
        evidence,
        lambda payload: payload["manifest_diff"].update(changed_paths=["forged"]),
    )
    with pytest.raises(ValueError, match="diff evidence"):
        _call(fixture)

    fixture = _fixture(tmp_path / "runtime")
    evidence = _install_manifest_bridge(fixture, edge_count=1)[0]
    _rewrite_json(
        evidence,
        lambda payload: payload["runtime_image_authority"].update(changed=True),
    )
    with pytest.raises(ValueError, match="runtime-image evidence"):
        _call(fixture)


def test_manifest_duplicate_yaml_and_bad_private_mode_are_rejected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = _install_manifest_bridge(fixture, edge_count=1)[0]
    manifest_evidence = json.loads(evidence.read_text())
    snapshot = fixture["artifact_root"] / manifest_evidence["old_manifest"]["path"]
    duplicate = snapshot.read_bytes() + b"target: {platform: fixture}\n"
    _private(snapshot, duplicate)
    duplicate_hash = _sha(duplicate)
    _rewrite_json(
        evidence,
        lambda payload: payload["old_manifest"].update(sha256=duplicate_hash),
    )
    _rewrite_json(
        fixture["source_evidence"][0],
        lambda payload: payload["base_manifest"].update(sha256=duplicate_hash),
    )
    with pytest.raises(ValueError, match="duplicate YAML"):
        _call(fixture)

    fixture = _fixture(tmp_path / "mode")
    evidence = _install_manifest_bridge(fixture, edge_count=1)[0]
    evidence.chmod(0o644)
    with pytest.raises(ValueError, match="mode 0600"):
        _call(fixture)
