#!/usr/bin/env python3
"""Atomically retire the realized MapAnything FP16 record and adopt FP32 authority.

This is deliberately a one-time, MapAnything-only transaction.  It advances the
asset realization across the exact reviewed manifest and source-contract hashes,
removes only ``engine.mapanything`` from the realized inventory, and leaves the
old FP16 engine bytes in place as continuously verified historical evidence.

Dry-run is write-free and produces both the proposed realization hash and a plan
hash.  Apply requires explicit authorization of both exact hashes.  Recovery is
limited to securely classifying a previously prepared transaction as the exact
before or exact after state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import rebase_asset_realization as transaction  # noqa: E402


REPO_ROOT = SCRIPT_DIR.parents[1]
NEW_MANIFEST_AUTHORITY = REPO_ROOT / "DS9/asset_manifest.yaml"
NEW_SOURCE_CONTRACT_AUTHORITY = REPO_ROOT / "DS9/config/engine_source_contracts.json"
REALIZATION_FILENAME = "asset_realization.json"
LOCK_FILENAME = transaction.LOCK_FILENAME
EVIDENCE_ROOT_RELATIVE = Path("mapanything_authority_transition")
EVIDENCE_FILENAME = "transition_evidence.json"
OLD_MANIFEST_SNAPSHOT_FILENAME = "old_asset_manifest.yaml"
NEW_MANIFEST_SNAPSHOT_FILENAME = "new_asset_manifest.yaml"
OLD_SOURCE_SNAPSHOT_FILENAME = "old_engine_source_contracts.json"
NEW_SOURCE_SNAPSHOT_FILENAME = "new_engine_source_contracts.json"
OLD_REALIZATION_SNAPSHOT_FILENAME = "asset_realization.before.json"
NEW_REALIZATION_SNAPSHOT_FILENAME = "asset_realization.after.json"
TRANSITION_CONTRACT = "noesis.ds9.mapanything_authority_transition"
PLAN_CONTRACT = "noesis.ds9.mapanything_authority_transition.plan.v1"
REALIZATION_CONTRACT = "noesis.ds9.asset_realization"
BASE_MANIFEST_PATH = "DS9/asset_manifest.yaml"
SOURCE_CONTRACT_PATH = "DS9/config/engine_source_contracts.json"
MAP_ARTIFACT_ID = "engine.mapanything"
MAP_CONTRACT_NAME = "mapanything"
OLD_MAP_OUTPUT = "DS9/models/engines/mapanything_images_294x518_b3_fp16.plan"
NEW_MAP_OUTPUT = "DS9/models/engines/mapanything_images_294x518_b3_fp32.plan"

# Exact reviewed one-time authority edge.  Any later authority edit makes this
# tool refuse rather than silently broadening the migration contract.
OLD_MANIFEST_SHA256 = "eed4340c1af587f903541f364824e643377b6149c5249e8314fc1d77fa12e789"
NEW_MANIFEST_SHA256 = "10e15351382e8a693a6acc048a05df3adff96a906fde50f0fa371fb0a8455198"
OLD_SOURCE_CONTRACTS_SHA256 = (
    "7ddd449c82e80d5c3195c0ccfb4ba4c3d4d542c654234f5d1f31d4ea549a20e0"
)
NEW_SOURCE_CONTRACTS_SHA256 = (
    "94144007b6e59f2eddd3239a6f0af02ce7d37224089e5f07a7a95d884f752b5a"
)
OLD_MAP_RECORD_SHA256 = (
    "24504cc670bbeeb325f361c05ba360b749e1c4bb98465aeb636c5d7800fbd6d0"
)
OLD_FP16_OUTPUT_SHA256 = (
    "aeb7140a56c31b8e420c7a1d31fb21ef4590c38d85e9eb41299dfe0d55b6891d"
)
OLD_FP16_OUTPUT_SIZE_BYTES = 1_850_829_956
MAX_EVIDENCE_TRANSACTIONS = 4096

_REALIZATION_KEYS = {
    "schema_version",
    "contract",
    "base_manifest",
    "source_contracts",
    "created_at_utc",
    "updated_at_utc",
    "artifacts",
}
_SNAPSHOT_FILENAMES = {
    OLD_MANIFEST_SNAPSHOT_FILENAME,
    NEW_MANIFEST_SNAPSHOT_FILENAME,
    OLD_SOURCE_SNAPSHOT_FILENAME,
    NEW_SOURCE_SNAPSHOT_FILENAME,
    OLD_REALIZATION_SNAPSHOT_FILENAME,
    NEW_REALIZATION_SNAPSHOT_FILENAME,
}
_TRANSACTION_INVENTORY = _SNAPSHOT_FILENAMES | {EVIDENCE_FILENAME}
_TERMINAL_STATES = {
    "committed",
    "aborted_before_commit",
    "aborted_by_recovery",
    "rolled_back_after_evidence_failure",
}
_RECOVERABLE_STATES = {
    "prepared",
    "recovery_required_after_realization_replace",
    "recovery_required_after_evidence_failure",
    "recovery_required_after_evidence_finalize",
}


class TransitionError(transaction.RebaseError):
    """Raised when the narrow MapAnything transition cannot be proven."""


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_exact_hash(observed: str, expected: str, label: str) -> None:
    if observed != expected:
        raise TransitionError(
            f"{label} hash mismatch: expected={expected} observed={observed}"
        )


def _manifest_rows(
    payload: Mapping[str, Any], label: str
) -> dict[str, Mapping[str, Any]]:
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise TransitionError(f"{label}.artifacts must be a nonempty list")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TransitionError(f"{label}.artifacts contains a non-mapping")
        artifact_id = str(row.get("id") or "").strip()
        if not artifact_id or artifact_id in result:
            raise TransitionError(f"{label}.artifacts has a missing or duplicate ID")
        result[artifact_id] = row
    return result


def _prove_manifest_edge(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> dict[str, Any]:
    old_without_artifacts = dict(old)
    new_without_artifacts = dict(new)
    old_without_artifacts.pop("artifacts", None)
    new_without_artifacts.pop("artifacts", None)
    if old_without_artifacts != new_without_artifacts:
        raise TransitionError("manifest drifted outside the artifact inventory")
    old_rows = _manifest_rows(old, "old manifest")
    new_rows = _manifest_rows(new, "new manifest")
    if set(old_rows) != set(new_rows) or MAP_ARTIFACT_ID not in old_rows:
        raise TransitionError("manifest artifact membership drifted")
    for artifact_id in sorted(set(old_rows) - {MAP_ARTIFACT_ID}):
        if old_rows[artifact_id] != new_rows[artifact_id]:
            raise TransitionError(
                f"manifest changed a non-MapAnything artifact: {artifact_id}"
            )
    old_map = old_rows[MAP_ARTIFACT_ID]
    new_map = new_rows[MAP_ARTIFACT_ID]
    changed_paths = transaction._semantic_diff_paths(old_map, new_map)
    if changed_paths != ["compatibility.precision", "output"]:
        raise TransitionError(
            "MapAnything manifest delta is not exactly precision and output"
        )
    old_compatibility = old_map.get("compatibility")
    new_compatibility = new_map.get("compatibility")
    if (
        old_map.get("kind") != "tensorrt_engine"
        or old_map.get("output") != OLD_MAP_OUTPUT
        or new_map.get("output") != NEW_MAP_OUTPUT
        or not isinstance(old_compatibility, Mapping)
        or not isinstance(new_compatibility, Mapping)
        or old_compatibility.get("precision") != "fp16"
        or new_compatibility.get("precision") != "fp32"
    ):
        raise TransitionError(
            "MapAnything manifest endpoints are not reviewed FP16/FP32"
        )
    return {
        "artifact_id": MAP_ARTIFACT_ID,
        "changed_paths": changed_paths,
        "old_record_sha256": _sha256_bytes(_canonical_json(old_map)),
        "new_record_sha256": _sha256_bytes(_canonical_json(new_map)),
        "old_output": OLD_MAP_OUTPUT,
        "new_output": NEW_MAP_OUTPUT,
    }


def _prove_source_contract_edge(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> dict[str, Any]:
    if set(old) != {"schema_version", "contracts"} or set(new) != set(old):
        raise TransitionError("source-contract top-level schema drifted")
    if old.get("schema_version") != 1 or new.get("schema_version") != 1:
        raise TransitionError("source-contract schema_version must remain 1")
    old_contracts = old.get("contracts")
    new_contracts = new.get("contracts")
    if (
        not isinstance(old_contracts, Mapping)
        or not isinstance(new_contracts, Mapping)
        or set(old_contracts) != set(new_contracts)
        or MAP_CONTRACT_NAME not in old_contracts
    ):
        raise TransitionError("source-contract membership drifted")
    changed = sorted(
        name for name in old_contracts if old_contracts[name] != new_contracts[name]
    )
    if changed != [MAP_CONTRACT_NAME]:
        raise TransitionError(
            "source-contract delta is not exactly the MapAnything contract"
        )
    old_map = old_contracts[MAP_CONTRACT_NAME]
    new_map = new_contracts[MAP_CONTRACT_NAME]
    if not isinstance(old_map, Mapping) or not isinstance(new_map, Mapping):
        raise TransitionError("MapAnything source contracts must be mappings")
    old_build = old_map.get("maintenance_build")
    new_build = new_map.get("maintenance_build")
    if (
        not isinstance(old_build, Mapping)
        or old_build.get("precision_arg") != "--fp16"
        or not isinstance(new_build, Mapping)
        or new_build.get("precision") != "fp32"
        or "precision_arg" in new_build
        or not isinstance(new_map.get("quality_gate"), Mapping)
    ):
        raise TransitionError("MapAnything source-contract endpoints are not FP16/FP32")
    return {
        "changed_contracts": [MAP_CONTRACT_NAME],
        "old_contract_sha256": _sha256_bytes(_canonical_json(old_map)),
        "new_contract_sha256": _sha256_bytes(_canonical_json(new_map)),
    }


def _engine_physical_path(artifact_root: Path, relative_raw: object) -> Path:
    relative = Path(str(relative_raw or "").strip())
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or tuple(relative.parts[:2]) != ("DS9", "models")
    ):
        raise TransitionError("old MapAnything output path is outside DS9/models")
    try:
        return transaction._bounded_path(
            artifact_root / Path(*relative.parts[1:]),
            artifact_root,
            "old MapAnything FP16 output",
        )
    except transaction.RebaseError as exc:
        raise TransitionError(str(exc)) from exc


def _verify_engine_file(path: Path, *, expected_hash: str, expected_size: int) -> None:
    transaction._reject_symlink_components(path, "old MapAnything FP16 output")
    try:
        lexical = path.lstat()
    except FileNotFoundError as exc:
        raise TransitionError("old MapAnything FP16 output is missing") from exc
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TransitionError(
            "old MapAnything FP16 output cannot be safely opened"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            stat.S_ISLNK(lexical.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino) != (lexical.st_dev, lexical.st_ino)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o022
        ):
            raise TransitionError(
                "old MapAnything FP16 output must be owned, single-link, and nonwritable"
            )
        if before.st_size != expected_size:
            raise TransitionError("old MapAnything FP16 output size drifted")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        if remaining or (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise TransitionError("old MapAnything FP16 output changed while read")
        if digest.hexdigest() != expected_hash:
            raise TransitionError("old MapAnything FP16 output digest drifted")
    finally:
        os.close(descriptor)


def _verify_retired_record(
    artifact_root: Path,
    old_map_manifest: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    record_digest = _sha256_bytes(_canonical_json(record))
    _require_exact_hash(
        record_digest, OLD_MAP_RECORD_SHA256, "old MapAnything realization record"
    )
    if set(record) != {"state", "provenance"} or not isinstance(
        record.get("provenance"), Mapping
    ):
        raise TransitionError("old MapAnything realization record shape drifted")
    provenance = record["provenance"]
    maintenance = provenance.get("maintenance")
    if (
        provenance.get("output_sha256") != OLD_FP16_OUTPUT_SHA256
        or not isinstance(maintenance, Mapping)
        or maintenance.get("output_size_bytes") != OLD_FP16_OUTPUT_SIZE_BYTES
        or maintenance.get("precision") != "fp16"
        or old_map_manifest.get("output") != OLD_MAP_OUTPUT
    ):
        raise TransitionError("old MapAnything realization proof drifted")
    path = _engine_physical_path(artifact_root, old_map_manifest.get("output"))
    _verify_engine_file(
        path,
        expected_hash=OLD_FP16_OUTPUT_SHA256,
        expected_size=OLD_FP16_OUTPUT_SIZE_BYTES,
    )
    return {
        "artifact_id": MAP_ARTIFACT_ID,
        "realized_record_sha256": record_digest,
        "output": {
            "path": OLD_MAP_OUTPUT,
            "sha256": OLD_FP16_OUTPUT_SHA256,
            "size_bytes": OLD_FP16_OUTPUT_SIZE_BYTES,
        },
    }


def _plan_payload(
    *,
    old_realization_sha256: str,
    new_realization_sha256: str,
    updated_before: str,
    updated_after: str,
    inventory_before: list[str],
    inventory_after: list[str],
    retired_artifact: Mapping[str, Any],
    manifest_diff: Mapping[str, Any],
    source_contract_diff: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract": PLAN_CONTRACT,
        "authorities": {
            "manifest": {
                "old_sha256": OLD_MANIFEST_SHA256,
                "new_sha256": NEW_MANIFEST_SHA256,
            },
            "source_contracts": {
                "old_sha256": OLD_SOURCE_CONTRACTS_SHA256,
                "new_sha256": NEW_SOURCE_CONTRACTS_SHA256,
            },
        },
        "realization": {
            "old_sha256": old_realization_sha256,
            "new_sha256": new_realization_sha256,
            "updated_at_utc_before": updated_before,
            "updated_at_utc_after": updated_after,
            "realized_artifact_ids_before": inventory_before,
            "realized_artifact_ids_after": inventory_after,
        },
        "retired_artifact": copy.deepcopy(dict(retired_artifact)),
        "manifest_diff": copy.deepcopy(dict(manifest_diff)),
        "source_contract_diff": copy.deepcopy(dict(source_contract_diff)),
        "mutation_paths": [
            "artifacts.engine.mapanything",
            "base_manifest.sha256",
            "source_contracts.sha256",
            "updated_at_utc",
        ],
    }


def _prove_transition(
    *,
    artifact_root: Path,
    old_manifest_raw: bytes,
    new_manifest_raw: bytes,
    old_source_raw: bytes,
    new_source_raw: bytes,
    realization_raw: bytes,
    expected_old_realization_sha256: str,
    updated_at_utc: str,
) -> dict[str, Any]:
    observed = {
        "old_manifest": _sha256_bytes(old_manifest_raw),
        "new_manifest": _sha256_bytes(new_manifest_raw),
        "old_source": _sha256_bytes(old_source_raw),
        "new_source": _sha256_bytes(new_source_raw),
        "realization": _sha256_bytes(realization_raw),
    }
    for key, expected in {
        "old_manifest": OLD_MANIFEST_SHA256,
        "new_manifest": NEW_MANIFEST_SHA256,
        "old_source": OLD_SOURCE_CONTRACTS_SHA256,
        "new_source": NEW_SOURCE_CONTRACTS_SHA256,
        "realization": expected_old_realization_sha256,
    }.items():
        _require_exact_hash(observed[key], expected, key.replace("_", " "))

    old_manifest = transaction._parse_yaml_mapping(old_manifest_raw, "old manifest")
    new_manifest = transaction._parse_yaml_mapping(new_manifest_raw, "new manifest")
    old_source = transaction._parse_json_mapping(old_source_raw, "old source contracts")
    new_source = transaction._parse_json_mapping(new_source_raw, "new source contracts")
    realization = transaction._parse_json_mapping(realization_raw, "asset realization")
    if _canonical_json(realization) != realization_raw:
        raise TransitionError("asset realization is not canonical JSON")
    if set(realization) != _REALIZATION_KEYS or (
        realization.get("schema_version") != 1
        or realization.get("contract") != REALIZATION_CONTRACT
    ):
        raise TransitionError("asset realization contract drifted")
    if realization.get("base_manifest") != {
        "path": BASE_MANIFEST_PATH,
        "sha256": OLD_MANIFEST_SHA256,
    }:
        raise TransitionError("asset realization does not bind the old manifest")
    if realization.get("source_contracts") != {
        "path": SOURCE_CONTRACT_PATH,
        "sha256": OLD_SOURCE_CONTRACTS_SHA256,
    }:
        raise TransitionError("asset realization does not bind the old source contract")
    created_at = transaction._parse_utc(
        realization.get("created_at_utc"), "created_at_utc"
    )
    previous_time = transaction._parse_utc(
        realization.get("updated_at_utc"), "current updated_at_utc"
    )
    proposed_time = transaction._parse_utc(updated_at_utc, "updated_at_utc")
    if proposed_time <= previous_time or proposed_time < created_at:
        raise TransitionError("updated_at_utc must strictly advance the realization")

    artifacts = realization.get("artifacts")
    if not isinstance(artifacts, dict) or MAP_ARTIFACT_ID not in artifacts:
        raise TransitionError("asset realization lacks the old MapAnything record")
    for artifact_id, record in artifacts.items():
        if (
            not isinstance(record, Mapping)
            or set(record) != {"state", "provenance"}
            or record.get("state") not in {"staged_unverified", "validated"}
            or not isinstance(record.get("provenance"), Mapping)
        ):
            raise TransitionError(f"realized record shape drifted: {artifact_id}")

    manifest_diff = _prove_manifest_edge(old_manifest, new_manifest)
    source_contract_diff = _prove_source_contract_edge(old_source, new_source)
    old_map_manifest = _manifest_rows(old_manifest, "old manifest")[MAP_ARTIFACT_ID]
    retired_artifact = _verify_retired_record(
        artifact_root, old_map_manifest, artifacts[MAP_ARTIFACT_ID]
    )

    proposal = copy.deepcopy(realization)
    proposal["base_manifest"]["sha256"] = NEW_MANIFEST_SHA256
    proposal["source_contracts"]["sha256"] = NEW_SOURCE_CONTRACTS_SHA256
    proposal["updated_at_utc"] = updated_at_utc
    del proposal["artifacts"][MAP_ARTIFACT_ID]
    mutation_paths = transaction._semantic_diff_paths(realization, proposal)
    expected_mutations = [
        "artifacts.engine.mapanything",
        "base_manifest.sha256",
        "source_contracts.sha256",
        "updated_at_utc",
    ]
    if mutation_paths != expected_mutations:
        raise TransitionError(
            "proposed realization escaped the exact transition contract"
        )
    for artifact_id, record in artifacts.items():
        if (
            artifact_id != MAP_ARTIFACT_ID
            and proposal["artifacts"].get(artifact_id) != record
        ):
            raise TransitionError(f"proposal changed realized record: {artifact_id}")
    proposal_raw = _canonical_json(proposal)
    proposal_hash = _sha256_bytes(proposal_raw)
    inventory_before = sorted(artifacts)
    inventory_after = sorted(proposal["artifacts"])
    if set(inventory_before) - set(inventory_after) != {MAP_ARTIFACT_ID}:
        raise TransitionError(
            "proposal retirement inventory is not exactly MapAnything"
        )
    plan = _plan_payload(
        old_realization_sha256=expected_old_realization_sha256,
        new_realization_sha256=proposal_hash,
        updated_before=str(realization["updated_at_utc"]),
        updated_after=updated_at_utc,
        inventory_before=inventory_before,
        inventory_after=inventory_after,
        retired_artifact=retired_artifact,
        manifest_diff=manifest_diff,
        source_contract_diff=source_contract_diff,
    )
    return {
        "proposal": proposal,
        "proposal_raw": proposal_raw,
        "proposal_sha256": proposal_hash,
        "plan": plan,
        "plan_sha256": _sha256_bytes(_canonical_json(plan)),
        "manifest_diff": manifest_diff,
        "source_contract_diff": source_contract_diff,
        "retired_artifact": retired_artifact,
        "inventory_before": inventory_before,
        "inventory_after": inventory_after,
        "realization": realization,
    }


def _read_transition_inputs(
    *,
    artifact_root: Path,
    old_manifest_snapshot: Path,
    old_source_contract_snapshot: Path,
    expected_old_realization_sha256: str,
    updated_at_utc: str,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    realization_path = transaction._bounded_path(
        artifact_root / REALIZATION_FILENAME, artifact_root, "asset realization"
    )
    paths = {
        "old_manifest": old_manifest_snapshot,
        "new_manifest": NEW_MANIFEST_AUTHORITY,
        "old_source": old_source_contract_snapshot,
        "new_source": NEW_SOURCE_CONTRACT_AUTHORITY,
        "realization": realization_path,
    }
    raw = {
        "old_manifest": transaction._read_file(
            paths["old_manifest"], "old manifest checkpoint", require_private=False
        ),
        "new_manifest": transaction._read_file(
            paths["new_manifest"], "new tracked manifest", require_private=False
        ),
        "old_source": transaction._read_file(
            paths["old_source"], "old source-contract checkpoint", require_private=False
        ),
        "new_source": transaction._read_file(
            paths["new_source"], "new tracked source contracts", require_private=False
        ),
        "realization": transaction._read_file(
            paths["realization"], "asset realization", require_private=True
        ),
    }
    proof = _prove_transition(
        artifact_root=artifact_root,
        old_manifest_raw=raw["old_manifest"],
        new_manifest_raw=raw["new_manifest"],
        old_source_raw=raw["old_source"],
        new_source_raw=raw["new_source"],
        realization_raw=raw["realization"],
        expected_old_realization_sha256=expected_old_realization_sha256,
        updated_at_utc=updated_at_utc,
    )
    return raw, proof


def _write_private_exclusive(path: Path, raw: bytes, label: str) -> None:
    parent = transaction._require_real_owned_directory(path.parent, f"{label} parent")
    candidate = transaction._bounded_path(path, parent, label)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    created_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(candidate, flags, 0o600)
        opened = os.fstat(descriptor)
        created_identity = (opened.st_dev, opened.st_ino)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(candidate, 0o600)
        transaction._fsync_directory(parent)
        observed = transaction._read_file(candidate, label, require_private=True)
        if observed != raw:
            raise TransitionError(f"{label} post-write bytes drifted")
    except Exception as exc:
        # Remove only the inode this call created.  If the directory entry was
        # swapped or linked, leave it visible and fail closed for inspection.
        if created_identity is not None:
            try:
                current = candidate.lstat()
                if (
                    not stat.S_ISLNK(current.st_mode)
                    and stat.S_ISREG(current.st_mode)
                    and (current.st_dev, current.st_ino) == created_identity
                    and current.st_uid == os.geteuid()
                    and current.st_nlink == 1
                ):
                    candidate.unlink()
                    try:
                        transaction._fsync_directory(parent)
                    except OSError:
                        pass
            except FileNotFoundError:
                pass
        if isinstance(exc, TransitionError):
            raise
        raise TransitionError(f"cannot create exact private {label}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _create_evidence_directory(root: Path, transaction_id: str) -> Path:
    evidence_root = root / EVIDENCE_ROOT_RELATIVE
    if not evidence_root.exists() and not evidence_root.is_symlink():
        evidence_root.mkdir(mode=0o700)
        os.chmod(evidence_root, 0o700)
        transaction._fsync_directory(root)
    evidence_root = transaction._require_real_owned_directory(
        evidence_root, "MapAnything transition evidence root"
    )
    if stat.S_IMODE(evidence_root.stat().st_mode) != 0o700:
        raise TransitionError(
            "MapAnything transition evidence root must have mode 0700"
        )
    run_dir = evidence_root / transaction_id
    try:
        run_dir.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise TransitionError(
            "MapAnything transition transaction already exists"
        ) from exc
    os.chmod(run_dir, 0o700)
    transaction._fsync_directory(evidence_root)
    return run_dir


def _assert_no_unresolved_transition(root: Path) -> None:
    evidence_root = root / EVIDENCE_ROOT_RELATIVE
    if not evidence_root.exists() and not evidence_root.is_symlink():
        return
    evidence_root = transaction._require_real_owned_directory(
        evidence_root, "MapAnything transition evidence root"
    )
    if stat.S_IMODE(evidence_root.stat().st_mode) != 0o700:
        raise TransitionError(
            "MapAnything transition evidence root must have mode 0700"
        )
    entries = list(evidence_root.iterdir())
    if len(entries) > MAX_EVIDENCE_TRANSACTIONS:
        raise TransitionError("MapAnything transition evidence exceeds safety bound")
    for entry in entries:
        if entry.is_symlink() or not entry.is_dir():
            raise TransitionError("MapAnything transition inventory is unsafe")
        if stat.S_IMODE(entry.stat().st_mode) != 0o700:
            raise TransitionError(
                "MapAnything transaction directory must have mode 0700"
            )
        inventory = {path.name for path in entry.iterdir()}
        if inventory != _TRANSACTION_INVENTORY:
            raise TransitionError(
                "incomplete MapAnything transition evidence requires recovery"
            )
        payload = transaction._parse_json_mapping(
            transaction._read_file(
                entry / EVIDENCE_FILENAME,
                "prior MapAnything transition evidence",
                require_private=True,
            ),
            "prior MapAnything transition evidence",
        )
        if payload.get("contract") != TRANSITION_CONTRACT:
            raise TransitionError(
                "prior MapAnything transition evidence contract drifted"
            )
        if payload.get("state") not in _TERMINAL_STATES:
            raise TransitionError(
                "unresolved MapAnything transition evidence requires recovery"
            )
        if payload.get("state") == "committed":
            raise TransitionError(
                "the one-time MapAnything authority transition is already committed"
            )


def _snapshot_records() -> dict[str, dict[str, str]]:
    return {
        "manifest": {
            "old_path": OLD_MANIFEST_SNAPSHOT_FILENAME,
            "old_sha256": OLD_MANIFEST_SHA256,
            "new_path": NEW_MANIFEST_SNAPSHOT_FILENAME,
            "new_authority_path": BASE_MANIFEST_PATH,
            "new_sha256": NEW_MANIFEST_SHA256,
        },
        "source_contracts": {
            "old_path": OLD_SOURCE_SNAPSHOT_FILENAME,
            "old_sha256": OLD_SOURCE_CONTRACTS_SHA256,
            "new_path": NEW_SOURCE_SNAPSHOT_FILENAME,
            "new_authority_path": SOURCE_CONTRACT_PATH,
            "new_sha256": NEW_SOURCE_CONTRACTS_SHA256,
        },
    }


def _new_evidence(
    *, transaction_id: str, proof: Mapping[str, Any], prepared_at_utc: str
) -> dict[str, Any]:
    realization = proof["realization"]
    return {
        "schema_version": 1,
        "contract": TRANSITION_CONTRACT,
        "transaction_id": transaction_id,
        "state": "prepared",
        "prepared_at_utc": prepared_at_utc,
        "committed_at_utc": None,
        "aborted_at_utc": None,
        "rolled_back_at_utc": None,
        "recovery": None,
        "realization_replace_recovery": None,
        "plan_sha256": proof["plan_sha256"],
        "authorities": _snapshot_records(),
        "realization": {
            "path": REALIZATION_FILENAME,
            "old_snapshot": OLD_REALIZATION_SNAPSHOT_FILENAME,
            "new_snapshot": NEW_REALIZATION_SNAPSHOT_FILENAME,
            "old_sha256": _sha256_bytes(_canonical_json(realization)),
            "new_sha256": proof["proposal_sha256"],
            "updated_at_utc_before": realization["updated_at_utc"],
            "updated_at_utc_after": proof["proposal"]["updated_at_utc"],
            "realized_artifact_ids_before": proof["inventory_before"],
            "realized_artifact_ids_after": proof["inventory_after"],
        },
        "retired_artifact": copy.deepcopy(dict(proof["retired_artifact"])),
        "manifest_diff": copy.deepcopy(dict(proof["manifest_diff"])),
        "source_contract_diff": copy.deepcopy(dict(proof["source_contract_diff"])),
        "mutation_paths": list(proof["plan"]["mutation_paths"]),
        "semantic_checks": {
            "exact_reviewed_authority_hashes": True,
            "manifest_change_is_mapanything_fp32_only": True,
            "source_contract_change_is_mapanything_fp32_only": True,
            "realization_authorities_advance_together": True,
            "only_mapanything_realization_is_retired": True,
            "other_realized_records_are_unchanged": True,
            "old_fp16_bytes_are_preserved_and_verified": True,
        },
        "verification_stages": ["initial", "snapshots_written"],
    }


def _write_snapshots(
    run_dir: Path, raw: Mapping[str, bytes], proposal_raw: bytes
) -> None:
    snapshots = {
        OLD_MANIFEST_SNAPSHOT_FILENAME: raw["old_manifest"],
        NEW_MANIFEST_SNAPSHOT_FILENAME: raw["new_manifest"],
        OLD_SOURCE_SNAPSHOT_FILENAME: raw["old_source"],
        NEW_SOURCE_SNAPSHOT_FILENAME: raw["new_source"],
        OLD_REALIZATION_SNAPSHOT_FILENAME: raw["realization"],
        NEW_REALIZATION_SNAPSHOT_FILENAME: proposal_raw,
    }
    for name, value in snapshots.items():
        _write_private_exclusive(run_dir / name, value, f"transition snapshot {name}")


def _revalidate_originals_and_snapshots(
    *,
    artifact_root: Path,
    old_manifest_snapshot: Path,
    old_source_contract_snapshot: Path,
    expected_old_realization_sha256: str,
    updated_at_utc: str,
    run_dir: Path,
    proof: Mapping[str, Any],
) -> None:
    raw, repeated = _read_transition_inputs(
        artifact_root=artifact_root,
        old_manifest_snapshot=old_manifest_snapshot,
        old_source_contract_snapshot=old_source_contract_snapshot,
        expected_old_realization_sha256=expected_old_realization_sha256,
        updated_at_utc=updated_at_utc,
    )
    if repeated["plan_sha256"] != proof["plan_sha256"]:
        raise TransitionError("transition plan changed during apply authorization")
    expected_snapshots = {
        OLD_MANIFEST_SNAPSHOT_FILENAME: raw["old_manifest"],
        NEW_MANIFEST_SNAPSHOT_FILENAME: raw["new_manifest"],
        OLD_SOURCE_SNAPSHOT_FILENAME: raw["old_source"],
        NEW_SOURCE_SNAPSHOT_FILENAME: raw["new_source"],
        OLD_REALIZATION_SNAPSHOT_FILENAME: raw["realization"],
        NEW_REALIZATION_SNAPSHOT_FILENAME: proof["proposal_raw"],
    }
    for name, expected in expected_snapshots.items():
        observed = transaction._read_file(
            run_dir / name, f"transition snapshot {name}", require_private=True
        )
        if observed != expected:
            raise TransitionError(f"transition snapshot changed: {name}")


def transition_mapanything_authority(
    *,
    artifact_root: Path,
    old_manifest_snapshot: Path,
    old_source_contract_snapshot: Path,
    expected_old_realization_sha256: str,
    expected_new_realization_sha256: str | None,
    expected_plan_sha256: str | None,
    updated_at_utc: str,
    dry_run: bool,
) -> dict[str, Any]:
    expected_old_realization_sha256 = transaction._require_sha256(
        expected_old_realization_sha256, "expected old realization hash"
    )
    if expected_new_realization_sha256 is not None:
        expected_new_realization_sha256 = transaction._require_sha256(
            expected_new_realization_sha256, "expected new realization hash"
        )
    if expected_plan_sha256 is not None:
        expected_plan_sha256 = transaction._require_sha256(
            expected_plan_sha256, "expected transition plan hash"
        )
    if not dry_run and (
        expected_new_realization_sha256 is None or expected_plan_sha256 is None
    ):
        raise TransitionError(
            "apply requires exact dry-run realization and transition-plan hashes"
        )
    root = transaction._require_real_owned_directory(artifact_root, "artifact root")
    if transaction._lexical_absolute(
        NEW_MANIFEST_AUTHORITY
    ) != transaction._lexical_absolute(REPO_ROOT / BASE_MANIFEST_PATH):
        raise TransitionError("new manifest authority is not tracked DS9 authority")
    if transaction._lexical_absolute(
        NEW_SOURCE_CONTRACT_AUTHORITY
    ) != transaction._lexical_absolute(REPO_ROOT / SOURCE_CONTRACT_PATH):
        raise TransitionError(
            "new source-contract authority is not tracked DS9 authority"
        )

    with transaction._artifact_transaction_lock(root):
        _assert_no_unresolved_transition(root)
        raw, proof = _read_transition_inputs(
            artifact_root=root,
            old_manifest_snapshot=old_manifest_snapshot,
            old_source_contract_snapshot=old_source_contract_snapshot,
            expected_old_realization_sha256=expected_old_realization_sha256,
            updated_at_utc=updated_at_utc,
        )
        result: dict[str, Any] = {
            "ok": True,
            "dry_run": dry_run,
            "old_manifest_sha256": OLD_MANIFEST_SHA256,
            "new_manifest_sha256": NEW_MANIFEST_SHA256,
            "old_source_contracts_sha256": OLD_SOURCE_CONTRACTS_SHA256,
            "new_source_contracts_sha256": NEW_SOURCE_CONTRACTS_SHA256,
            "old_realization_sha256": expected_old_realization_sha256,
            "new_realization_sha256": proof["proposal_sha256"],
            "transition_plan_sha256": proof["plan_sha256"],
            "retired_artifact": copy.deepcopy(dict(proof["retired_artifact"])),
            "realized_artifact_ids_before": proof["inventory_before"],
            "realized_artifact_ids_after": proof["inventory_after"],
            "mutation_paths": list(proof["plan"]["mutation_paths"]),
        }
        if expected_new_realization_sha256 is not None and (
            proof["proposal_sha256"] != expected_new_realization_sha256
        ):
            raise TransitionError(
                "authorized new realization hash differs from dry-run"
            )
        if (
            expected_plan_sha256 is not None
            and proof["plan_sha256"] != expected_plan_sha256
        ):
            raise TransitionError(
                "authorized transition plan hash differs from dry-run"
            )
        if dry_run:
            return result

        transaction_id = transaction._run_id()
        run_dir = _create_evidence_directory(root, transaction_id)
        evidence_path = run_dir / EVIDENCE_FILENAME
        try:
            _write_snapshots(run_dir, raw, proof["proposal_raw"])
            evidence = _new_evidence(
                transaction_id=transaction_id,
                proof=proof,
                prepared_at_utc=_utc_now(),
            )
            prepared_raw = _canonical_json(evidence)
            _write_private_exclusive(
                evidence_path, prepared_raw, "MapAnything transition evidence"
            )
        except Exception:
            # No prepared record exists until every immutable snapshot is durable.
            # Best-effort cleanup is safe only before evidence creation.
            if not evidence_path.exists() and not evidence_path.is_symlink():
                for name in _SNAPSHOT_FILENAMES:
                    (run_dir / name).unlink(missing_ok=True)
                try:
                    run_dir.rmdir()
                    transaction._fsync_directory(run_dir.parent)
                except OSError:
                    pass
            raise

        try:
            _revalidate_originals_and_snapshots(
                artifact_root=root,
                old_manifest_snapshot=old_manifest_snapshot,
                old_source_contract_snapshot=old_source_contract_snapshot,
                expected_old_realization_sha256=expected_old_realization_sha256,
                updated_at_utc=updated_at_utc,
                run_dir=run_dir,
                proof=proof,
            )
            evidence["verification_stages"].append("pre_cas")
        except Exception:
            evidence["state"] = "aborted_before_commit"
            evidence["aborted_at_utc"] = _utc_now()
            transaction._atomic_private_transition(
                evidence_path,
                _canonical_json(evidence),
                before_raw=prepared_raw,
                label="MapAnything transition evidence",
            )
            raise

        realization_path = root / REALIZATION_FILENAME
        try:
            _committed_hash, recovered = transaction._atomic_private_transition(
                realization_path,
                proof["proposal_raw"],
                before_raw=raw["realization"],
                label="asset realization",
            )
        except transaction._AtomicTransitionError as exc:
            if exc.observed_state == "before":
                evidence["state"] = "aborted_before_commit"
                evidence["aborted_at_utc"] = _utc_now()
            else:
                evidence["state"] = "recovery_required_after_realization_replace"
                evidence["recovery"] = {
                    "outcome": "exact_realization_state_requires_classification",
                    "recorded_at_utc": _utc_now(),
                }
            try:
                transaction._atomic_private_transition(
                    evidence_path,
                    _canonical_json(evidence),
                    before_raw=prepared_raw,
                    label="MapAnything transition evidence",
                )
            except transaction._AtomicTransitionError:
                pass
            raise TransitionError(
                "asset realization CAS did not reach a proven commit"
            ) from exc

        if recovered:
            evidence["realization_replace_recovery"] = {
                "outcome": "proposal_exact_bytes_refsynced",
                "recovered_at_utc": _utc_now(),
            }
        old_output = _engine_physical_path(root, OLD_MAP_OUTPUT)
        _verify_engine_file(
            old_output,
            expected_hash=OLD_FP16_OUTPUT_SHA256,
            expected_size=OLD_FP16_OUTPUT_SIZE_BYTES,
        )
        for name in _SNAPSHOT_FILENAMES:
            transaction._read_file(
                run_dir / name, f"transition snapshot {name}", require_private=True
            )
        evidence["verification_stages"].extend(["post_cas", "pre_evidence_commit"])
        evidence["state"] = "committed"
        evidence["committed_at_utc"] = _utc_now()
        committed_evidence_raw = _canonical_json(evidence)
        try:
            transaction._atomic_private_transition(
                evidence_path,
                committed_evidence_raw,
                before_raw=prepared_raw,
                label="MapAnything transition evidence",
            )
        except transaction._AtomicTransitionError as evidence_error:
            if evidence_error.observed_state == "before":
                try:
                    transaction._atomic_private_transition(
                        realization_path,
                        raw["realization"],
                        before_raw=proof["proposal_raw"],
                        label="asset realization rollback",
                    )
                except transaction._AtomicTransitionError as rollback_error:
                    evidence["state"] = "recovery_required_after_evidence_failure"
                    evidence["committed_at_utc"] = None
                    evidence["recovery"] = {
                        "outcome": "realization_rollback_requires_classification",
                        "recorded_at_utc": _utc_now(),
                    }
                    try:
                        transaction._atomic_private_transition(
                            evidence_path,
                            _canonical_json(evidence),
                            before_raw=prepared_raw,
                            label="MapAnything transition evidence",
                        )
                    except transaction._AtomicTransitionError:
                        pass
                    raise TransitionError(
                        "evidence finalization failed and rollback is unresolved"
                    ) from rollback_error
                evidence["state"] = "rolled_back_after_evidence_failure"
                evidence["committed_at_utc"] = None
                evidence["rolled_back_at_utc"] = _utc_now()
                evidence["recovery"] = {
                    "outcome": "exact_before_realization_restored",
                    "recorded_at_utc": evidence["rolled_back_at_utc"],
                }
                transaction._atomic_private_transition(
                    evidence_path,
                    _canonical_json(evidence),
                    before_raw=prepared_raw,
                    label="MapAnything transition evidence",
                )
                raise TransitionError(
                    "evidence finalization failed; exact prior realization was restored"
                ) from evidence_error
            # Do not alter the realization if the evidence inode cannot be
            # classified; recovery will classify both exact snapshots.
            recovery_evidence = copy.deepcopy(evidence)
            recovery_evidence["state"] = "recovery_required_after_evidence_finalize"
            recovery_evidence["committed_at_utc"] = None
            recovery_evidence["recovery"] = {
                "outcome": "evidence_and_realization_require_classification",
                "recorded_at_utc": _utc_now(),
            }
            try:
                state = transaction._classify_exact_file_state(
                    evidence_path,
                    before_raw=prepared_raw,
                    after_raw=committed_evidence_raw,
                    label="MapAnything transition evidence",
                )
                if state == "before":
                    transaction._atomic_private_transition(
                        evidence_path,
                        _canonical_json(recovery_evidence),
                        before_raw=prepared_raw,
                        label="MapAnything transition evidence",
                    )
            except Exception:
                pass
            raise TransitionError(
                "evidence finalization outcome requires recovery"
            ) from evidence_error

        result.update(
            {
                "state": "committed",
                "transaction_id": transaction_id,
                "evidence": evidence_path.relative_to(root).as_posix(),
            }
        )
        return result


def _load_recovery_material(
    artifact_root: Path, transaction_id: str
) -> tuple[Path, dict[str, Any], bytes, dict[str, Any], bytes, bytes]:
    if not transaction_id or any(
        token in transaction_id for token in ("/", "\\", "..")
    ):
        raise TransitionError("recovery transaction ID is invalid")
    evidence_root = transaction._bounded_path(
        artifact_root / EVIDENCE_ROOT_RELATIVE,
        artifact_root,
        "MapAnything transition evidence root",
    )
    run_dir = transaction._bounded_path(
        evidence_root / transaction_id,
        evidence_root,
        "MapAnything transition transaction",
    )
    run_dir = transaction._require_real_owned_directory(
        run_dir, "MapAnything transition transaction"
    )
    if stat.S_IMODE(run_dir.stat().st_mode) != 0o700:
        raise TransitionError("recovery transaction directory must have mode 0700")
    if {path.name for path in run_dir.iterdir()} != _TRANSACTION_INVENTORY:
        raise TransitionError("recovery transaction inventory drifted")
    evidence_path = run_dir / EVIDENCE_FILENAME
    evidence_raw = transaction._read_file(
        evidence_path, "MapAnything recovery evidence", require_private=True
    )
    evidence = transaction._parse_json_mapping(
        evidence_raw, "MapAnything recovery evidence"
    )
    if _canonical_json(evidence) != evidence_raw:
        raise TransitionError("recovery evidence is not canonical JSON")
    if (
        evidence.get("schema_version") != 1
        or evidence.get("contract") != TRANSITION_CONTRACT
        or evidence.get("transaction_id") != transaction_id
        or evidence.get("state") not in _RECOVERABLE_STATES
    ):
        raise TransitionError("transaction is not valid recoverable evidence")
    snapshots = {
        name: transaction._read_file(
            run_dir / name, f"recovery snapshot {name}", require_private=True
        )
        for name in _SNAPSHOT_FILENAMES
    }
    realization_record = evidence.get("realization")
    if not isinstance(realization_record, Mapping):
        raise TransitionError("recovery realization evidence is malformed")
    updated_at = str(realization_record.get("updated_at_utc_after") or "")
    expected_old_hash = transaction._require_sha256(
        str(realization_record.get("old_sha256") or ""),
        "recovery old realization hash",
    )
    proof = _prove_transition(
        artifact_root=artifact_root,
        old_manifest_raw=snapshots[OLD_MANIFEST_SNAPSHOT_FILENAME],
        new_manifest_raw=snapshots[NEW_MANIFEST_SNAPSHOT_FILENAME],
        old_source_raw=snapshots[OLD_SOURCE_SNAPSHOT_FILENAME],
        new_source_raw=snapshots[NEW_SOURCE_SNAPSHOT_FILENAME],
        realization_raw=snapshots[OLD_REALIZATION_SNAPSHOT_FILENAME],
        expected_old_realization_sha256=expected_old_hash,
        updated_at_utc=updated_at,
    )
    if snapshots[NEW_REALIZATION_SNAPSHOT_FILENAME] != proof["proposal_raw"]:
        raise TransitionError("recovery after-realization snapshot drifted")
    expected_evidence = _new_evidence(
        transaction_id=transaction_id,
        proof=proof,
        prepared_at_utc=str(evidence.get("prepared_at_utc") or ""),
    )
    immutable_keys = {
        "schema_version",
        "contract",
        "transaction_id",
        "prepared_at_utc",
        "plan_sha256",
        "authorities",
        "realization",
        "retired_artifact",
        "manifest_diff",
        "source_contract_diff",
        "mutation_paths",
        "semantic_checks",
    }
    for key in immutable_keys:
        if evidence.get(key) != expected_evidence.get(key):
            raise TransitionError(f"recovery evidence immutable field drifted: {key}")
    current_manifest = transaction._read_file(
        NEW_MANIFEST_AUTHORITY, "current tracked manifest", require_private=False
    )
    current_source = transaction._read_file(
        NEW_SOURCE_CONTRACT_AUTHORITY,
        "current tracked source contracts",
        require_private=False,
    )
    if (
        current_manifest != snapshots[NEW_MANIFEST_SNAPSHOT_FILENAME]
        or current_source != snapshots[NEW_SOURCE_SNAPSHOT_FILENAME]
    ):
        raise TransitionError("tracked FP32 authority drifted during recovery")
    return (
        evidence_path,
        evidence,
        evidence_raw,
        proof,
        snapshots[OLD_REALIZATION_SNAPSHOT_FILENAME],
        snapshots[NEW_REALIZATION_SNAPSHOT_FILENAME],
    )


def recover_mapanything_authority(
    *, artifact_root: Path, transaction_id: str
) -> dict[str, Any]:
    root = transaction._require_real_owned_directory(artifact_root, "artifact root")
    with transaction._artifact_transaction_lock(root):
        (
            evidence_path,
            evidence,
            evidence_raw,
            proof,
            before_raw,
            after_raw,
        ) = _load_recovery_material(root, transaction_id)
        realization_path = root / REALIZATION_FILENAME
        state = transaction._classify_exact_file_state(
            realization_path,
            before_raw=before_raw,
            after_raw=after_raw,
            label="asset realization recovery",
        )
        if state == "ambiguous":
            raise TransitionError(
                "recovery refused: realization is neither exact before nor exact after"
            )
        _verify_engine_file(
            _engine_physical_path(root, OLD_MAP_OUTPUT),
            expected_hash=OLD_FP16_OUTPUT_SHA256,
            expected_size=OLD_FP16_OUTPUT_SIZE_BYTES,
        )
        recovered_at = _utc_now()
        if state == "before":
            evidence["state"] = "aborted_by_recovery"
            evidence["aborted_at_utc"] = recovered_at
            outcome = "exact_before_realization_terminalized"
        else:
            evidence["state"] = "committed"
            evidence["committed_at_utc"] = recovered_at
            outcome = "exact_after_realization_terminalized"
        evidence["recovery"] = {
            "outcome": outcome,
            "recorded_at_utc": recovered_at,
        }
        if "recovery_final" not in evidence.get("verification_stages", []):
            evidence.setdefault("verification_stages", []).append("recovery_final")
        transaction._atomic_private_transition(
            evidence_path,
            _canonical_json(evidence),
            before_raw=evidence_raw,
            label="MapAnything transition recovery evidence",
        )
        return {
            "ok": True,
            "state": evidence["state"],
            "transaction_id": transaction_id,
            "realization_state": state,
            "new_realization_sha256": proof["proposal_sha256"],
            "transition_plan_sha256": proof["plan_sha256"],
            "evidence": evidence_path.relative_to(root).as_posix(),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--recover", metavar="TRANSACTION_ID")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--old-manifest-snapshot", type=Path)
    parser.add_argument("--old-source-contract-snapshot", type=Path)
    parser.add_argument("--expected-old-realization-sha256")
    parser.add_argument("--expected-new-realization-sha256")
    parser.add_argument("--expected-transition-plan-sha256")
    parser.add_argument("--updated-at-utc")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.recover:
            result = recover_mapanything_authority(
                artifact_root=args.artifact_root,
                transaction_id=args.recover,
            )
        else:
            required = {
                "--old-manifest-snapshot": args.old_manifest_snapshot,
                "--old-source-contract-snapshot": args.old_source_contract_snapshot,
                "--expected-old-realization-sha256": args.expected_old_realization_sha256,
                "--updated-at-utc": args.updated_at_utc,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise TransitionError(
                    "missing required arguments: " + ", ".join(missing)
                )
            result = transition_mapanything_authority(
                artifact_root=args.artifact_root,
                old_manifest_snapshot=args.old_manifest_snapshot,
                old_source_contract_snapshot=args.old_source_contract_snapshot,
                expected_old_realization_sha256=args.expected_old_realization_sha256,
                expected_new_realization_sha256=args.expected_new_realization_sha256,
                expected_plan_sha256=args.expected_transition_plan_sha256,
                updated_at_utc=args.updated_at_utc,
                dry_run=bool(args.dry_run),
            )
    except Exception as exc:
        print(
            f"[FAIL] MapAnything authority transition refused: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
