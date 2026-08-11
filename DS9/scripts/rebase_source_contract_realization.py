#!/usr/bin/env python3
"""Rebase a DS9 realization across reviewed unrealized-engine source changes.

This transaction is intentionally narrower than a general source-contract or
realization migration.  Every changed contract key must be named explicitly,
must map through the reviewed engine registry, and must still be absent from
the external realization.  The only permitted realization mutations are the
source-contract SHA-256 anchor and an advancing caller-fixed UTC timestamp.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import rebase_asset_realization as transaction  # noqa: E402


RebaseError = transaction.RebaseError
_AtomicTransitionError = transaction._AtomicTransitionError
_artifact_transaction_lock = transaction._artifact_transaction_lock
_atomic_private_replace = transaction._atomic_private_replace
_atomic_private_transition = transaction._atomic_private_transition
_bounded_path = transaction._bounded_path
_canonical_json = transaction._canonical_json
_fsync_directory = transaction._fsync_directory
_lexical_absolute = transaction._lexical_absolute
_parse_json_mapping = transaction._parse_json_mapping
_parse_utc = transaction._parse_utc
_parse_yaml_mapping = transaction._parse_yaml_mapping
_read_file = transaction._read_file
_require_real_owned_directory = transaction._require_real_owned_directory
_require_sha256 = transaction._require_sha256
_run_id = transaction._run_id
_semantic_diff_paths = transaction._semantic_diff_paths
_sha256_bytes = transaction._sha256_bytes
_utc_now = transaction._utc_now


BASE_MANIFEST_AUTHORITY = REPO_ROOT / "DS9/asset_manifest.yaml"
SOURCE_CONTRACTS_AUTHORITY = REPO_ROOT / "DS9/config/engine_source_contracts.json"
REALIZATION_FILENAME = "asset_realization.json"
EVIDENCE_ROOT_RELATIVE = Path("source_contract_rebase")
EVIDENCE_FILENAME = "source_contract_rebase_evidence.json"
REALIZATION_CONTRACT = "noesis.ds9.asset_realization"
REBASE_CONTRACT = "noesis.ds9.source_contract_realization_rebase"
SOURCE_CONTRACT_PATH = "DS9/config/engine_source_contracts.json"
BASE_MANIFEST_PATH = "DS9/asset_manifest.yaml"
_REALIZATION_KEYS = {
    "schema_version",
    "contract",
    "base_manifest",
    "source_contracts",
    "created_at_utc",
    "updated_at_utc",
    "artifacts",
}
_SOURCE_CONTRACT_KEYS = {"schema_version", "contracts"}

# This registry is deliberately explicit and mirrors the reviewed engine
# reconciliation authority.  Adding a source contract must be a code review,
# never an inferred name transformation.
ENGINE_ARTIFACT_IDS = {
    "yolo11_seg": "engine.yolo11_seg_alternate",
    "yolo26_m": "engine.yolo26_detect_m",
    "yolo26_seg_s": "engine.yolo26_seg_s",
    "reid_swin": "engine.reid_swin_tiny",
    "yolo26_pose_n": "engine.pose_yolo26",
    "depth_anything_v2_tracking": "engine.depth_tracking_dav2",
    "mapanything": "engine.mapanything",
    "wholebody49_s_masks": "engine.wholebody49_s_masks",
    "wholebody49_x_boxes": "engine.wholebody49_x_boxes",
    "bodypose3dnet": "engine.v3dt_bodypose",
    "v3dt_tracker_reid": "engine.v3dt_tracker_reid",
}


def _normalize_changed_contracts(values: Sequence[str]) -> tuple[str, ...]:
    if len(set(ENGINE_ARTIFACT_IDS.values())) != len(ENGINE_ARTIFACT_IDS):
        raise RebaseError("reviewed engine registry is not one-to-one")
    normalized = tuple(str(value or "").strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise RebaseError("at least one nonempty changed contract is required")
    if len(set(normalized)) != len(normalized):
        raise RebaseError("changed contract allowlist contains a duplicate")
    unknown = sorted(set(normalized) - set(ENGINE_ARTIFACT_IDS))
    if unknown:
        raise RebaseError(
            "changed contract allowlist contains an unregistered key: "
            + ", ".join(unknown)
        )
    return tuple(sorted(normalized))


def _manifest_engine_ids(payload: Mapping[str, Any]) -> set[str]:
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise RebaseError("base manifest artifacts must be a nonempty list")
    seen: set[str] = set()
    engines: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise RebaseError("base manifest contains a malformed artifact record")
        artifact_id = str(row.get("id") or "").strip()
        if not artifact_id or artifact_id in seen:
            raise RebaseError("base manifest has a missing or duplicate artifact ID")
        seen.add(artifact_id)
        if row.get("kind") == "tensorrt_engine":
            engines.add(artifact_id)
    missing = sorted(set(ENGINE_ARTIFACT_IDS.values()) - engines)
    if missing:
        raise RebaseError(
            "base manifest lacks registered TensorRT engine artifacts: "
            + ", ".join(missing)
        )
    return engines


def _validate_source_contract_documents(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    changed_contracts: tuple[str, ...],
) -> dict[str, Any]:
    if set(old) != _SOURCE_CONTRACT_KEYS or set(new) != _SOURCE_CONTRACT_KEYS:
        raise RebaseError("source-contract top-level schema keys drifted")
    if old.get("schema_version") != 1 or new.get("schema_version") != 1:
        raise RebaseError("source-contract schema_version must remain 1")
    old_contracts = old.get("contracts")
    new_contracts = new.get("contracts")
    if not isinstance(old_contracts, Mapping) or not isinstance(new_contracts, Mapping):
        raise RebaseError("source-contract contracts must be mappings")
    if set(old_contracts) != set(ENGINE_ARTIFACT_IDS) or set(new_contracts) != set(
        ENGINE_ARTIFACT_IDS
    ):
        raise RebaseError(
            "source-contract membership must exactly match the reviewed engine registry"
        )
    for key, value in (*old_contracts.items(), *new_contracts.items()):
        if not isinstance(value, Mapping) or not value:
            raise RebaseError(f"source contract {key!r} must be a nonempty mapping")
    observed_changes = tuple(
        sorted(
            key
            for key in ENGINE_ARTIFACT_IDS
            if old_contracts[key] != new_contracts[key]
        )
    )
    if observed_changes != changed_contracts:
        raise RebaseError(
            "semantic source-contract changes do not equal the explicit allowlist: "
            f"allowed={list(changed_contracts)} observed={list(observed_changes)}"
        )
    return {
        "changed_contracts": list(observed_changes),
        "unchanged_contract_count": len(ENGINE_ARTIFACT_IDS) - len(observed_changes),
    }


def _validate_realized_artifacts(
    realization: Mapping[str, Any], manifest_engine_ids: set[str]
) -> dict[str, Mapping[str, Any]]:
    raw = realization.get("artifacts")
    if not isinstance(raw, dict):
        raise RebaseError("asset realization artifacts must be a mapping")
    result: dict[str, Mapping[str, Any]] = {}
    for artifact_id, record in raw.items():
        if artifact_id not in manifest_engine_ids:
            raise RebaseError(
                f"realized artifact is not a base-manifest TensorRT engine: {artifact_id}"
            )
        if not isinstance(record, Mapping) or set(record) != {
            "state",
            "provenance",
        }:
            raise RebaseError(
                f"realized engine record has an invalid shape: {artifact_id}"
            )
        if record.get("state") not in {
            "staged_unverified",
            "validated",
        } or not isinstance(record.get("provenance"), Mapping):
            raise RebaseError(
                f"realized engine record has invalid state/provenance: {artifact_id}"
            )
        result[str(artifact_id)] = record
    return result


def _create_evidence_directory(root: Path, transaction_id: str) -> Path:
    evidence_root = root / EVIDENCE_ROOT_RELATIVE
    if not evidence_root.exists() and not evidence_root.is_symlink():
        evidence_root.mkdir(mode=0o700)
        _fsync_directory(root)
    evidence_root = _require_real_owned_directory(
        evidence_root, "source-contract-rebase evidence root"
    )
    run_dir = evidence_root / transaction_id
    try:
        run_dir.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise RebaseError(
            "source-contract-rebase evidence transaction already exists"
        ) from exc
    os.chmod(run_dir, 0o700)
    _fsync_directory(evidence_root)
    return run_dir


def _assert_no_unresolved_rebase(root: Path) -> None:
    evidence_root = root / EVIDENCE_ROOT_RELATIVE
    if not evidence_root.exists() and not evidence_root.is_symlink():
        return
    evidence_root = _require_real_owned_directory(
        evidence_root, "source-contract-rebase evidence root"
    )
    with os.scandir(evidence_root) as inventory:
        entries = list(inventory)
    if len(entries) > 4096:
        raise RebaseError(
            "source-contract-rebase evidence inventory exceeds safety bound"
        )
    terminal_states = {
        "committed",
        "aborted_before_commit",
        "rolled_back_after_evidence_failure",
    }
    for entry in entries:
        if entry.is_symlink():
            raise RebaseError(
                "source-contract-rebase evidence inventory contains a symlink"
            )
        if not entry.is_dir(follow_symlinks=False):
            continue
        evidence_path = Path(entry.path) / EVIDENCE_FILENAME
        if not evidence_path.exists() and not evidence_path.is_symlink():
            continue
        payload = _parse_json_mapping(
            _read_file(
                evidence_path,
                "prior source-contract-rebase evidence",
                require_private=True,
            ),
            "prior source-contract-rebase evidence",
        )
        if payload.get("contract") != REBASE_CONTRACT:
            raise RebaseError(
                "prior source-contract-rebase evidence has an invalid contract"
            )
        if payload.get("state") not in terminal_states:
            raise RebaseError(
                "unresolved prepared source-contract rebase requires recovery"
            )


def _revalidate_inputs(
    *,
    old_contracts_path: Path,
    new_contracts_path: Path,
    base_manifest_path: Path,
    realization_path: Path,
    expected_old_source_contracts_sha256: str,
    expected_new_source_contracts_sha256: str,
    expected_base_manifest_sha256: str,
    expected_old_realization_sha256: str,
) -> None:
    checks = (
        (
            old_contracts_path,
            "old source-contract snapshot",
            True,
            expected_old_source_contracts_sha256,
        ),
        (
            new_contracts_path,
            "new tracked source-contract authority",
            False,
            expected_new_source_contracts_sha256,
        ),
        (
            base_manifest_path,
            "unchanged base-manifest authority",
            False,
            expected_base_manifest_sha256,
        ),
        (
            realization_path,
            "asset realization",
            True,
            expected_old_realization_sha256,
        ),
    )
    for path, label, require_private, expected_hash in checks:
        observed_hash = _sha256_bytes(
            _read_file(path, label, require_private=require_private)
        )
        if observed_hash != expected_hash:
            raise RebaseError(f"{label} changed during rebase authorization")


def rebase_source_contract_realization(
    *,
    artifact_root: Path,
    old_source_contract_snapshot: Path,
    expected_old_source_contracts_sha256: str,
    expected_new_source_contracts_sha256: str,
    expected_base_manifest_sha256: str,
    expected_old_realization_sha256: str,
    expected_new_realization_sha256: str | None,
    updated_at_utc: str,
    changed_contracts: Sequence[str],
    dry_run: bool,
) -> dict[str, Any]:
    """Prove and optionally commit one unrealized-engine contract rebase."""

    expected_old_source_contracts_sha256 = _require_sha256(
        expected_old_source_contracts_sha256,
        "expected old source-contract hash",
    )
    expected_new_source_contracts_sha256 = _require_sha256(
        expected_new_source_contracts_sha256,
        "expected new source-contract hash",
    )
    expected_base_manifest_sha256 = _require_sha256(
        expected_base_manifest_sha256,
        "expected unchanged base-manifest hash",
    )
    expected_old_realization_sha256 = _require_sha256(
        expected_old_realization_sha256,
        "expected old realization hash",
    )
    if expected_old_source_contracts_sha256 == expected_new_source_contracts_sha256:
        raise RebaseError("old and new source-contract hashes must differ")
    if expected_new_realization_sha256 is not None:
        expected_new_realization_sha256 = _require_sha256(
            expected_new_realization_sha256,
            "expected new realization hash",
        )
    if not dry_run and expected_new_realization_sha256 is None:
        raise RebaseError("apply requires an explicit expected new realization hash")
    allowed_changes = _normalize_changed_contracts(changed_contracts)

    root = _require_real_owned_directory(artifact_root, "artifact root")
    evidence_root = root / EVIDENCE_ROOT_RELATIVE
    old_contracts_path = _bounded_path(
        old_source_contract_snapshot,
        evidence_root,
        "old source-contract snapshot",
    )
    new_contracts_path = _lexical_absolute(SOURCE_CONTRACTS_AUTHORITY)
    if new_contracts_path != _lexical_absolute(
        REPO_ROOT / "DS9/config/engine_source_contracts.json"
    ):
        raise RebaseError(
            "new source-contract authority is not the tracked DS9 contract"
        )
    base_manifest_path = _lexical_absolute(BASE_MANIFEST_AUTHORITY)
    if base_manifest_path != _lexical_absolute(REPO_ROOT / "DS9/asset_manifest.yaml"):
        raise RebaseError(
            "base-manifest authority is not tracked DS9/asset_manifest.yaml"
        )
    realization_path = _bounded_path(
        root / REALIZATION_FILENAME,
        root,
        "asset realization",
    )

    with _artifact_transaction_lock(root):
        _assert_no_unresolved_rebase(root)
        old_contracts_raw = _read_file(
            old_contracts_path,
            "old source-contract snapshot",
            require_private=True,
        )
        new_contracts_raw = _read_file(
            new_contracts_path,
            "new tracked source-contract authority",
            require_private=False,
        )
        base_manifest_raw = _read_file(
            base_manifest_path,
            "unchanged base-manifest authority",
            require_private=False,
        )
        realization_raw = _read_file(
            realization_path,
            "asset realization",
            require_private=True,
        )
        observed_hashes = {
            "old_source_contracts": _sha256_bytes(old_contracts_raw),
            "new_source_contracts": _sha256_bytes(new_contracts_raw),
            "base_manifest": _sha256_bytes(base_manifest_raw),
            "old_realization": _sha256_bytes(realization_raw),
        }
        expected_hashes = {
            "old_source_contracts": expected_old_source_contracts_sha256,
            "new_source_contracts": expected_new_source_contracts_sha256,
            "base_manifest": expected_base_manifest_sha256,
            "old_realization": expected_old_realization_sha256,
        }
        for key, expected_hash in expected_hashes.items():
            if observed_hashes[key] != expected_hash:
                raise RebaseError(
                    f"{key.replace('_', ' ')} hash mismatch: "
                    f"expected={expected_hash} observed={observed_hashes[key]}"
                )

        old_contracts = _parse_json_mapping(
            old_contracts_raw,
            "old source-contract snapshot",
        )
        new_contracts = _parse_json_mapping(
            new_contracts_raw,
            "new tracked source-contract authority",
        )
        base_manifest = _parse_yaml_mapping(
            base_manifest_raw,
            "unchanged base-manifest authority",
        )
        realization = _parse_json_mapping(realization_raw, "asset realization")
        if _canonical_json(realization) != realization_raw:
            raise RebaseError(
                "asset realization is not canonical JSON; refusing incidental byte drift"
            )
        contract_diff = _validate_source_contract_documents(
            old_contracts,
            new_contracts,
            allowed_changes,
        )
        manifest_engine_ids = _manifest_engine_ids(base_manifest)
        if set(realization) != _REALIZATION_KEYS:
            raise RebaseError("asset realization has unexpected or missing keys")
        if (
            realization.get("schema_version") != 1
            or realization.get("contract") != REALIZATION_CONTRACT
        ):
            raise RebaseError("asset realization contract is invalid")
        expected_manifest_record = {
            "path": BASE_MANIFEST_PATH,
            "sha256": expected_base_manifest_sha256,
        }
        if realization.get("base_manifest") != expected_manifest_record:
            raise RebaseError("asset realization base-manifest authority drifted")
        expected_old_contract_record = {
            "path": SOURCE_CONTRACT_PATH,
            "sha256": expected_old_source_contracts_sha256,
        }
        if realization.get("source_contracts") != expected_old_contract_record:
            raise RebaseError(
                "asset realization does not bind the old source-contract authority"
            )

        created_at = _parse_utc(
            realization.get("created_at_utc"),
            "created_at_utc",
        )
        previous_updated_at = _parse_utc(
            realization.get("updated_at_utc"),
            "current updated_at_utc",
        )
        proposed_updated_at = _parse_utc(updated_at_utc, "updated_at_utc")
        if (
            proposed_updated_at <= previous_updated_at
            or proposed_updated_at < created_at
        ):
            raise RebaseError("updated_at_utc must advance the realization timestamp")

        realized_artifacts = _validate_realized_artifacts(
            realization,
            manifest_engine_ids,
        )
        mapped_artifacts = {
            ENGINE_ARTIFACT_IDS[contract_key] for contract_key in allowed_changes
        }
        already_realized = sorted(mapped_artifacts & set(realized_artifacts))
        if already_realized:
            raise RebaseError(
                "changed source contracts map to realized engines: "
                + ", ".join(already_realized)
            )

        proposal = copy.deepcopy(realization)
        proposal["source_contracts"]["sha256"] = expected_new_source_contracts_sha256
        proposal["updated_at_utc"] = updated_at_utc
        semantic_changes = _semantic_diff_paths(realization, proposal)
        if semantic_changes != ["source_contracts.sha256", "updated_at_utc"]:
            raise RebaseError(
                "proposed realization mutation escaped the narrow source-contract contract"
            )
        if proposal["base_manifest"] != realization["base_manifest"]:
            raise RebaseError("proposal changed the base-manifest authority")
        if proposal["artifacts"] != realization["artifacts"]:
            raise RebaseError("proposal changed realized artifact records")
        proposal_raw = _canonical_json(proposal)
        proposal_hash = _sha256_bytes(proposal_raw)
        if (
            expected_new_realization_sha256 is not None
            and proposal_hash != expected_new_realization_sha256
        ):
            raise RebaseError(
                "new realization hash mismatch: "
                f"expected={expected_new_realization_sha256} observed={proposal_hash}"
            )

        semantic_checks = {
            "source_contract_schema_unchanged": True,
            "source_contract_membership_unchanged": True,
            "changed_contracts_exact_allowlist": True,
            "changed_contracts_map_only_to_unrealized_engines": True,
            "base_manifest_unchanged": True,
            "realized_artifact_records_unchanged": True,
            "proposal_mutation_is_exact": True,
        }
        result: dict[str, Any] = {
            "ok": True,
            "dry_run": dry_run,
            "base_manifest_sha256": expected_base_manifest_sha256,
            "old_source_contracts_sha256": expected_old_source_contracts_sha256,
            "new_source_contracts_sha256": expected_new_source_contracts_sha256,
            "old_realization_sha256": expected_old_realization_sha256,
            "new_realization_sha256": proposal_hash,
            "changed_contracts": list(allowed_changes),
            "mapped_unrealized_artifact_ids": sorted(mapped_artifacts),
            "realized_engine_count": len(realized_artifacts),
            "contract_diff": contract_diff,
            "mutation_paths": semantic_changes,
            "semantic_checks": semantic_checks,
        }
        if dry_run:
            return result

        transaction_id = _run_id()
        run_dir = _create_evidence_directory(root, transaction_id)
        evidence_path = run_dir / EVIDENCE_FILENAME
        evidence: dict[str, Any] = {
            "schema_version": 1,
            "contract": REBASE_CONTRACT,
            "transaction_id": transaction_id,
            "state": "prepared",
            "prepared_at_utc": _utc_now(),
            "committed_at_utc": None,
            "old_source_contracts": {
                "path": old_contracts_path.relative_to(root).as_posix(),
                "sha256": expected_old_source_contracts_sha256,
            },
            "new_source_contracts": {
                "path": SOURCE_CONTRACT_PATH,
                "sha256": expected_new_source_contracts_sha256,
            },
            "base_manifest": expected_manifest_record,
            "realization": {
                "path": REALIZATION_FILENAME,
                "old_sha256": expected_old_realization_sha256,
                "new_sha256": proposal_hash,
                "updated_at_utc_before": realization["updated_at_utc"],
                "updated_at_utc_after": updated_at_utc,
            },
            "changed_contracts": list(allowed_changes),
            "mapped_unrealized_artifact_ids": sorted(mapped_artifacts),
            "realized_engine_ids": sorted(realized_artifacts),
            "mutation_paths": semantic_changes,
            "semantic_checks": copy.deepcopy(semantic_checks),
        }
        prepared_raw = _canonical_json(evidence)
        _atomic_private_replace(
            evidence_path,
            prepared_raw,
            expected_current_sha256=None,
            label="source-contract-rebase evidence",
        )
        try:
            _revalidate_inputs(
                old_contracts_path=old_contracts_path,
                new_contracts_path=new_contracts_path,
                base_manifest_path=base_manifest_path,
                realization_path=realization_path,
                expected_old_source_contracts_sha256=(
                    expected_old_source_contracts_sha256
                ),
                expected_new_source_contracts_sha256=(
                    expected_new_source_contracts_sha256
                ),
                expected_base_manifest_sha256=expected_base_manifest_sha256,
                expected_old_realization_sha256=expected_old_realization_sha256,
            )
        except Exception:
            evidence["state"] = "aborted_before_commit"
            evidence["aborted_at_utc"] = _utc_now()
            try:
                _atomic_private_transition(
                    evidence_path,
                    _canonical_json(evidence),
                    before_raw=prepared_raw,
                    label="source-contract-rebase evidence",
                )
            except _AtomicTransitionError as evidence_error:
                raise RebaseError(
                    "source-contract authorization failed before commit, but "
                    "terminal evidence could not be proven"
                ) from evidence_error
            raise

        realization_replace_recovered = False
        try:
            _, realization_replace_recovered = _atomic_private_transition(
                realization_path,
                proposal_raw,
                before_raw=realization_raw,
                label="asset realization",
            )
        except _AtomicTransitionError as realization_error:
            if realization_error.observed_state == "before":
                evidence["state"] = "aborted_before_commit"
                evidence["aborted_at_utc"] = _utc_now()
                try:
                    _atomic_private_transition(
                        evidence_path,
                        _canonical_json(evidence),
                        before_raw=prepared_raw,
                        label="source-contract-rebase evidence",
                    )
                except _AtomicTransitionError as evidence_error:
                    raise RebaseError(
                        "realization remained at its old exact hash, but terminal "
                        "abort evidence could not be proven"
                    ) from evidence_error
                raise RebaseError(
                    "asset realization replacement failed before commit; the old "
                    "exact bytes remain installed"
                ) from realization_error

            evidence["state"] = "recovery_required_after_realization_replace"
            evidence["recovery_required_at_utc"] = _utc_now()
            evidence["realization_replace_observed_state"] = "ambiguous"
            try:
                _atomic_private_transition(
                    evidence_path,
                    _canonical_json(evidence),
                    before_raw=prepared_raw,
                    label="source-contract-rebase evidence",
                )
            except _AtomicTransitionError:
                pass
            raise RebaseError(
                "asset realization replacement has an ambiguous exact-byte "
                "outcome; nonterminal recovery evidence requires inspection"
            ) from realization_error

        if realization_replace_recovered:
            evidence["realization_replace_recovery"] = {
                "outcome": "proposal_exact_bytes_refsynced",
                "recovered_at_utc": _utc_now(),
            }

        evidence["state"] = "committed"
        evidence["committed_at_utc"] = _utc_now()
        try:
            _atomic_private_transition(
                evidence_path,
                _canonical_json(evidence),
                before_raw=prepared_raw,
                label="source-contract-rebase evidence",
            )
        except _AtomicTransitionError as evidence_error:
            if evidence_error.observed_state == "ambiguous":
                raise RebaseError(
                    "evidence finalization outcome is ambiguous; the realization "
                    "remains at its proposed exact hash and recovery is required"
                ) from evidence_error
            try:
                _atomic_private_transition(
                    realization_path,
                    realization_raw,
                    before_raw=proposal_raw,
                    label="asset realization rollback",
                )
            except _AtomicTransitionError as rollback_error:
                evidence["state"] = "recovery_required_after_evidence_failure"
                evidence["committed_at_utc"] = None
                evidence["recovery_required_at_utc"] = _utc_now()
                evidence["realization_rollback_observed_state"] = (
                    rollback_error.observed_state
                )
                try:
                    _atomic_private_transition(
                        evidence_path,
                        _canonical_json(evidence),
                        before_raw=prepared_raw,
                        label="source-contract-rebase evidence",
                    )
                except _AtomicTransitionError:
                    pass
                raise RebaseError(
                    "evidence finalization failed after realization commit; "
                    "automatic rollback was not proven and nonterminal evidence "
                    "requires recovery"
                ) from rollback_error
            evidence["state"] = "rolled_back_after_evidence_failure"
            evidence["committed_at_utc"] = None
            evidence["rolled_back_at_utc"] = _utc_now()
            try:
                _atomic_private_transition(
                    evidence_path,
                    _canonical_json(evidence),
                    before_raw=prepared_raw,
                    label="source-contract-rebase evidence",
                )
            except _AtomicTransitionError as rollback_evidence_error:
                raise RebaseError(
                    "evidence finalization failed and the realization was restored, "
                    "but terminal rollback evidence could not be proven"
                ) from rollback_evidence_error
            raise RebaseError(
                "evidence finalization failed; realization was restored to its old hash"
            ) from evidence_error

        result["evidence"] = evidence_path.relative_to(root).as_posix()
        result["transaction_id"] = transaction_id
        result["state"] = "committed"
        return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Use one fixed --updated-at-utc and the same repeated "
            "--changed-contract values for dry-run and apply. Review the dry-run "
            "new_realization_sha256, then authorize that exact digest on apply."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--old-source-contract-snapshot", type=Path, required=True)
    parser.add_argument("--expected-old-source-contracts-sha256", required=True)
    parser.add_argument("--expected-new-source-contracts-sha256", required=True)
    parser.add_argument("--expected-base-manifest-sha256", required=True)
    parser.add_argument("--expected-old-realization-sha256", required=True)
    parser.add_argument("--expected-new-realization-sha256")
    parser.add_argument("--updated-at-utc", required=True)
    parser.add_argument(
        "--changed-contract",
        dest="changed_contracts",
        action="append",
        required=True,
        help="Reviewed source-contract key; repeat once per changed unrealized engine.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = rebase_source_contract_realization(
            artifact_root=args.artifact_root,
            old_source_contract_snapshot=args.old_source_contract_snapshot,
            expected_old_source_contracts_sha256=(
                args.expected_old_source_contracts_sha256
            ),
            expected_new_source_contracts_sha256=(
                args.expected_new_source_contracts_sha256
            ),
            expected_base_manifest_sha256=args.expected_base_manifest_sha256,
            expected_old_realization_sha256=args.expected_old_realization_sha256,
            expected_new_realization_sha256=args.expected_new_realization_sha256,
            updated_at_utc=args.updated_at_utc,
            changed_contracts=args.changed_contracts,
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        print(
            f"[FAIL] source-contract realization rebase refused: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
