from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable
from unittest import mock

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "DS9/scripts/rebase_source_contract_realization.py"


def _load_rebaser():
    spec = importlib.util.spec_from_file_location(
        "ds9_source_contract_realization_rebaser",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rebaser = _load_rebaser()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_private(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)


def _contracts() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contracts": {
            key: {
                "raw_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                "staged": f"models/{key}.onnx",
            }
            for key in rebaser.ENGINE_ARTIFACT_IDS
        },
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True, mode=0o750)
    _write_private(
        artifact_root / rebaser.transaction.LOCK_FILENAME,
        b"canonical-lock\n",
    )
    evidence_root = artifact_root / rebaser.EVIDENCE_ROOT_RELATIVE
    evidence_root.mkdir(mode=0o700)
    snapshot_dir = evidence_root / "reviewed-old-source-contracts"
    snapshot_dir.mkdir(mode=0o700)

    old_contracts = _contracts()
    new_contracts = copy.deepcopy(old_contracts)
    new_contracts["contracts"]["wholebody49_x_boxes"]["raw_sha256"] = "f" * 64
    old_contracts_path = snapshot_dir / "engine_source_contracts.before.json"
    _write_private(old_contracts_path, _canonical_json(old_contracts))

    repo = tmp_path / "repo"
    source_contracts_path = repo / "DS9/config/engine_source_contracts.json"
    source_contracts_path.parent.mkdir(parents=True)
    source_contracts_path.write_bytes(_canonical_json(new_contracts))
    source_contracts_path.chmod(0o644)

    manifest = {
        "schema_version": 2,
        "artifacts": [
            {"id": artifact_id, "kind": "tensorrt_engine"}
            for artifact_id in rebaser.ENGINE_ARTIFACT_IDS.values()
        ],
    }
    base_manifest_path = repo / "DS9/asset_manifest.yaml"
    base_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    base_manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    base_manifest_path.chmod(0o644)

    realization = {
        "schema_version": 1,
        "contract": rebaser.REALIZATION_CONTRACT,
        "base_manifest": {
            "path": rebaser.BASE_MANIFEST_PATH,
            "sha256": _sha256(base_manifest_path),
        },
        "source_contracts": {
            "path": rebaser.SOURCE_CONTRACT_PATH,
            "sha256": _sha256(old_contracts_path),
        },
        "created_at_utc": "2026-07-11T01:00:00Z",
        "updated_at_utc": "2026-07-11T02:00:00Z",
        "artifacts": {
            "engine.yolo26_detect_m": {
                "state": "staged_unverified",
                "provenance": {
                    "output_sha256": "7" * 64,
                    "private_marker": "must-not-enter-evidence",
                },
            }
        },
    }
    realization_path = artifact_root / rebaser.REALIZATION_FILENAME
    _write_private(realization_path, _canonical_json(realization))
    return {
        "artifact_root": artifact_root,
        "evidence_root": evidence_root,
        "repo": repo,
        "old_contracts_path": old_contracts_path,
        "source_contracts_path": source_contracts_path,
        "base_manifest_path": base_manifest_path,
        "realization_path": realization_path,
        "old_contracts": old_contracts,
        "new_contracts": new_contracts,
        "manifest": manifest,
        "realization": realization,
    }


def _arguments(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_root": fixture["artifact_root"],
        "old_source_contract_snapshot": fixture["old_contracts_path"],
        "expected_old_source_contracts_sha256": _sha256(fixture["old_contracts_path"]),
        "expected_new_source_contracts_sha256": _sha256(
            fixture["source_contracts_path"]
        ),
        "expected_base_manifest_sha256": _sha256(fixture["base_manifest_path"]),
        "expected_old_realization_sha256": _sha256(fixture["realization_path"]),
        "expected_new_realization_sha256": None,
        "updated_at_utc": "2026-07-11T04:00:00Z",
        "changed_contracts": ["wholebody49_x_boxes"],
        "dry_run": True,
    }


def _call(
    fixture: dict[str, Any],
    **overrides: Any,
) -> dict[str, Any]:
    arguments = _arguments(fixture)
    arguments.update(overrides)
    with mock.patch.multiple(
        rebaser,
        REPO_ROOT=fixture["repo"],
        SOURCE_CONTRACTS_AUTHORITY=fixture["source_contracts_path"],
        BASE_MANIFEST_AUTHORITY=fixture["base_manifest_path"],
    ):
        return rebaser.rebase_source_contract_realization(**arguments)


def _rewrite_new_contracts(
    fixture: dict[str, Any],
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    payload = copy.deepcopy(fixture["old_contracts"])
    mutate(payload)
    fixture["new_contracts"] = payload
    fixture["source_contracts_path"].write_bytes(_canonical_json(payload))
    fixture["source_contracts_path"].chmod(0o644)


def _rewrite_realization(
    fixture: dict[str, Any],
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    payload = copy.deepcopy(fixture["realization"])
    mutate(payload)
    fixture["realization"] = payload
    _write_private(fixture["realization_path"], _canonical_json(payload))


def _evidence_payloads(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in fixture["evidence_root"].glob(f"*/{rebaser.EVIDENCE_FILENAME}"):
        result.append(json.loads(path.read_text(encoding="utf-8")))
    return result


def test_deterministic_dry_run_and_successful_apply(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    realization_before = fixture["realization_path"].read_bytes()
    inventory_before = sorted(
        path.relative_to(fixture["artifact_root"]).as_posix()
        for path in fixture["evidence_root"].rglob("*")
    )

    first = _call(fixture)
    second = _call(fixture)
    assert first == second
    assert first["ok"] is True
    assert first["dry_run"] is True
    assert first["changed_contracts"] == ["wholebody49_x_boxes"]
    assert first["mapped_unrealized_artifact_ids"] == ["engine.wholebody49_x_boxes"]
    assert fixture["realization_path"].read_bytes() == realization_before
    assert (
        sorted(
            path.relative_to(fixture["artifact_root"]).as_posix()
            for path in fixture["evidence_root"].rglob("*")
        )
        == inventory_before
    )

    applied = _call(
        fixture,
        dry_run=False,
        expected_new_realization_sha256=first["new_realization_sha256"],
    )
    assert applied["state"] == "committed"
    assert _sha256(fixture["realization_path"]) == first["new_realization_sha256"]
    realized = json.loads(fixture["realization_path"].read_text(encoding="utf-8"))
    assert realized["source_contracts"]["sha256"] == _sha256(
        fixture["source_contracts_path"]
    )
    assert realized["base_manifest"] == fixture["realization"]["base_manifest"]
    assert realized["artifacts"] == fixture["realization"]["artifacts"]
    evidence_path = fixture["artifact_root"] / applied["evidence"]
    assert stat.S_IMODE(evidence_path.stat().st_mode) == 0o600
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["state"] == "committed"
    assert "must-not-enter-evidence" not in json.dumps(evidence)


def test_changed_realized_contract_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _rewrite_new_contracts(
        fixture,
        lambda payload: payload["contracts"]["yolo26_m"].update(raw_sha256="e" * 64),
    )
    with pytest.raises(rebaser.RebaseError, match="map to realized engines"):
        _call(fixture, changed_contracts=["yolo26_m"])


def test_unallowlisted_contract_change_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["new_contracts"]["contracts"]["v3dt_tracker_reid"]["raw_sha256"] = "d" * 64
    fixture["source_contracts_path"].write_bytes(
        _canonical_json(fixture["new_contracts"])
    )
    with pytest.raises(rebaser.RebaseError, match="do not equal"):
        _call(fixture)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.update(schema_version=2),
        lambda payload: payload.update(unreviewed_top_level=True),
        lambda payload: payload["contracts"].pop("v3dt_tracker_reid"),
    ),
)
def test_source_contract_schema_and_membership_drift_are_rejected(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    fixture = _fixture(tmp_path)
    _rewrite_new_contracts(fixture, mutate)
    with pytest.raises(rebaser.RebaseError, match="schema|membership"):
        _call(fixture)


@pytest.mark.parametrize(
    "changed_contracts",
    (
        ["wholebody49_x_boxes", "wholebody49_x_boxes"],
        ["unregistered_engine"],
    ),
)
def test_changed_contract_allowlist_must_be_unique_and_registered(
    tmp_path: Path,
    changed_contracts: list[str],
) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(rebaser.RebaseError, match="duplicate|unregistered"):
        _call(fixture, changed_contracts=changed_contracts)


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"expected_old_source_contracts_sha256": "0" * 64}, "hash mismatch"),
        ({"expected_base_manifest_sha256": "0" * 64}, "hash mismatch"),
        ({"expected_new_realization_sha256": "0" * 64}, "new realization hash"),
        ({"updated_at_utc": "2026-07-11T02:00:00Z"}, "must advance"),
    ),
)
def test_hash_cas_and_timestamp_authority_are_exact(
    tmp_path: Path,
    override: dict[str, Any],
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(rebaser.RebaseError, match=message):
        _call(fixture, **override)


def test_old_snapshot_requires_private_mode_and_no_links(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["old_contracts_path"].chmod(0o644)
    with pytest.raises(rebaser.RebaseError, match="mode 0600"):
        _call(fixture)

    fixture = _fixture(tmp_path / "linked")
    target = fixture["old_contracts_path"].with_name("target.json")
    _write_private(target, fixture["old_contracts_path"].read_bytes())
    fixture["old_contracts_path"].unlink()
    fixture["old_contracts_path"].symlink_to(target)
    with pytest.raises(rebaser.RebaseError, match="symlink"):
        _call(fixture)


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    duplicate = (
        b'{"schema_version":1,"schema_version":1,"contracts":'
        + json.dumps(fixture["new_contracts"]["contracts"], sort_keys=True).encode(
            "utf-8"
        )
        + b"}\n"
    )
    fixture["source_contracts_path"].write_bytes(duplicate)
    with pytest.raises(rebaser.RebaseError, match="duplicate"):
        _call(fixture)


def test_canonical_artifact_lock_contention_is_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    descriptor = os.open(
        fixture["artifact_root"] / rebaser.transaction.LOCK_FILENAME,
        os.O_RDWR,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(rebaser.RebaseError, match="owns the lock"):
            _call(fixture)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_apply_cas_change_aborts_before_commit_with_terminal_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    dry = _call(fixture)
    realization_before = fixture["realization_path"].read_bytes()
    original_revalidate = rebaser._revalidate_inputs

    def mutate_then_revalidate(**kwargs: Any) -> None:
        fixture["source_contracts_path"].write_bytes(
            fixture["source_contracts_path"].read_bytes() + b" "
        )
        original_revalidate(**kwargs)

    with (
        mock.patch.object(
            rebaser,
            "_revalidate_inputs",
            side_effect=mutate_then_revalidate,
        ),
        pytest.raises(rebaser.RebaseError, match="changed during"),
    ):
        _call(
            fixture,
            dry_run=False,
            expected_new_realization_sha256=dry["new_realization_sha256"],
        )
    assert fixture["realization_path"].read_bytes() == realization_before
    evidence = _evidence_payloads(fixture)
    assert len(evidence) == 1
    assert evidence[0]["state"] == "aborted_before_commit"


def test_evidence_interruption_rolls_back_realization_and_records_recovery(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    dry = _call(fixture)
    realization_before = fixture["realization_path"].read_bytes()
    real_transition = rebaser._atomic_private_transition
    injected = False

    def fail_committed_evidence(
        path: Path,
        after_raw: bytes,
        *,
        before_raw: bytes,
        label: str,
    ):
        nonlocal injected
        if label == "source-contract-rebase evidence" and not injected:
            payload = json.loads(after_raw)
            if payload.get("state") == "committed":
                injected = True
                raise rebaser._AtomicTransitionError(
                    "injected evidence interruption",
                    observed_state="before",
                )
        return real_transition(
            path,
            after_raw,
            before_raw=before_raw,
            label=label,
        )

    with (
        mock.patch.object(
            rebaser,
            "_atomic_private_transition",
            side_effect=fail_committed_evidence,
        ),
        pytest.raises(rebaser.RebaseError, match="restored to its old hash"),
    ):
        _call(
            fixture,
            dry_run=False,
            expected_new_realization_sha256=dry["new_realization_sha256"],
        )
    assert injected is True
    assert fixture["realization_path"].read_bytes() == realization_before
    evidence = _evidence_payloads(fixture)
    assert len(evidence) == 1
    assert evidence[0]["state"] == "rolled_back_after_evidence_failure"


def test_unresolved_prepared_transaction_requires_recovery(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    unresolved = fixture["evidence_root"] / "unresolved"
    unresolved.mkdir(mode=0o700)
    _write_private(
        unresolved / rebaser.EVIDENCE_FILENAME,
        _canonical_json(
            {
                "schema_version": 1,
                "contract": rebaser.REBASE_CONTRACT,
                "state": "prepared",
            }
        ),
    )
    with pytest.raises(rebaser.RebaseError, match="requires recovery"):
        _call(fixture)


def test_repeatable_cli_changed_contract_contract() -> None:
    args = rebaser.parse_args(
        [
            "--dry-run",
            "--artifact-root",
            "/tmp/artifacts",
            "--old-source-contract-snapshot",
            "/tmp/artifacts/source_contract_rebase/old.json",
            "--expected-old-source-contracts-sha256",
            "1" * 64,
            "--expected-new-source-contracts-sha256",
            "2" * 64,
            "--expected-base-manifest-sha256",
            "3" * 64,
            "--expected-old-realization-sha256",
            "4" * 64,
            "--updated-at-utc",
            "2026-07-11T04:00:00Z",
            "--changed-contract",
            "wholebody49_s_masks",
            "--changed-contract",
            "wholebody49_x_boxes",
        ]
    )
    assert args.changed_contracts == [
        "wholebody49_s_masks",
        "wholebody49_x_boxes",
    ]


def test_engine_registry_matches_reconciliation_authority() -> None:
    path = REPO_ROOT / "DS9/scripts/reconcile_engine_provenance.py"
    spec = importlib.util.spec_from_file_location("ds9_engine_reconciler", path)
    assert spec is not None and spec.loader is not None
    reconciler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reconciler)
    assert rebaser.ENGINE_ARTIFACT_IDS == reconciler.ENGINE_ARTIFACT_IDS
