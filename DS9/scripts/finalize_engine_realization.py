#!/usr/bin/env python3
"""Commit or roll back a DS9 engine and its external realization as one unit."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import reconcile_engine_provenance as reconciler  # noqa: E402
import validate_asset_manifest as validator  # noqa: E402
import nvml_gpu_memory_sampler as gpu_sampler  # noqa: E402
from engine_maintenance_common import (  # noqa: E402
    _copy_private,
    _write_json_atomic,
    fsync_directory,
    fsync_file,
    mapanything_quality_gate_from_source_contracts,
    new_run_id,
    validate_mapanything_functional_quality_receipt,
)


CONTRACT = "noesis.ds9.engine_finalize_transaction"
TRANSACTION_ROOT_RELATIVE = Path("models/engine_finalize")
BASE_MANIFEST_AUTHORITY = REPO_ROOT / "DS9/asset_manifest.yaml"
SOURCE_CONTRACTS_AUTHORITY = reconciler.SOURCE_CONTRACTS
_RUN_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{12}Z$")
GPU_MEMORY_SAMPLE_INTERVAL_MS = gpu_sampler.REVIEWED_SAMPLE_INTERVAL_MS
GPU_MEMORY_MAX_GAP_MS = gpu_sampler.REVIEWED_MAX_GAP_MS
GPU_MEMORY_GUARD_MIB = gpu_sampler.REVIEWED_GUARD_MIB_BY_ENGINE


class FinalizeError(RuntimeError):
    """Raised when a host-side engine transaction cannot commit safely."""


class PriorAgreementError(FinalizeError):
    """Raised when builder evidence did not start from the host snapshot."""


def _validate_gpu_guard_evidence(
    *,
    transaction_path: Path,
    transaction: Mapping[str, Any],
    prepared_transaction_sha256: str,
    maintenance: Mapping[str, Any],
    evidence_path: Path,
    expected_evidence_sha256: str,
    container_id: str,
    wrapper_pid: int,
    wrapper_start_time_ticks: int,
) -> dict[str, Any]:
    root = Path(str(transaction["artifact_root"]))
    transaction_cohort = transaction_path.parent
    bounded = validator._bounded_private_path(
        evidence_path, transaction_cohort, "NVML GPU-memory guard evidence"
    )
    if bounded != transaction_cohort / "gpu-memory.jsonl":
        raise FinalizeError(
            "NVML GPU-memory guard evidence is outside its transaction cohort"
        )
    expected_hash = validator._require_sha256(
        expected_evidence_sha256, "NVML GPU-memory guard evidence digest"
    )
    engine = str(transaction["engine"])
    guard_mib = gpu_sampler.reviewed_guard_mib(engine)
    platform = _required_mapping(
        _required_mapping(maintenance.get("metadata"), "maintenance metadata").get(
            "platform"
        ),
        "maintenance platform",
    )
    expected_uuid = str(platform.get("gpu_uuid") or "")
    if not expected_uuid.startswith("GPU-"):
        raise FinalizeError("maintenance evidence lacks a reviewed GPU UUID")
    artifact_root_id = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    summary_args = argparse.Namespace(
        evidence=bounded,
        device_index=0,
        expected_uuid=expected_uuid,
        engine=engine,
        transaction_id=str(transaction["transaction_id"]),
        prepared_transaction_sha256=prepared_transaction_sha256,
        artifact_root_id=artifact_root_id,
        container_id=container_id,
        guard_mib=guard_mib,
        interval_ms=GPU_MEMORY_SAMPLE_INTERVAL_MS,
        max_gap_ms=GPU_MEMORY_MAX_GAP_MS,
        parent_pid=wrapper_pid,
        parent_start_time_ticks=wrapper_start_time_ticks,
        allow_active=False,
    )
    try:
        summary = gpu_sampler.summarize(summary_args)
    except Exception as exc:
        raise FinalizeError(f"NVML GPU-memory guard proof is invalid: {exc}") from exc
    if (
        summary.get("state") != "stopped"
        or summary.get("guard_ok") is not True
        or summary.get("evidence_sha256") != expected_hash
    ):
        raise FinalizeError("NVML GPU-memory guard proof is not a clean terminal run")
    relative = bounded.relative_to(root)
    record = {
        "contract": gpu_sampler.CONTRACT,
        "path": relative.as_posix(),
        "sha256": expected_hash,
        "engine": engine,
        "transaction_id": str(transaction["transaction_id"]),
        "prepared_transaction_sha256": prepared_transaction_sha256,
        "artifact_root_id": artifact_root_id,
        "container_id": container_id,
        "wrapper_pid": wrapper_pid,
        "wrapper_start_time_ticks": wrapper_start_time_ticks,
        "device_index": 0,
        "gpu_uuid": expected_uuid,
        "guard_mib": guard_mib,
        "sample_interval_ms": GPU_MEMORY_SAMPLE_INTERVAL_MS,
        "maximum_gap_limit_ms": GPU_MEMORY_MAX_GAP_MS,
        "summary": summary,
    }
    try:
        return validator._validate_gpu_memory_guard_proof(
            artifact_root=root,
            expected_engine=engine,
            maintenance_payload=maintenance,
            guard_record=record,
        )
    except Exception as exc:
        raise FinalizeError(
            f"NVML GPU-memory guard realization proof is invalid: {exc}"
        ) from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalizeError(f"{label} must be a mapping")
    return value


def _artifact_root(path: Path) -> Path:
    resolved = validator._resolve_artifact_root(path)
    if resolved is None:
        raise FinalizeError("artifact root is required")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved.resolve(strict=True)


def _require_inherited_transaction_lock(root: Path, descriptor: int) -> None:
    expected = root / ".noesis-ds9-artifact-transaction.lock"
    try:
        info = os.fstat(descriptor)
        linked = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
        expected_info = expected.lstat()
    except (OSError, ValueError) as exc:
        raise FinalizeError(
            f"invalid inherited artifact transaction lock fd: {exc}"
        ) from exc
    if linked != expected:
        raise FinalizeError(
            f"transaction lock fd resolves to {linked}, expected {expected}"
        )
    if (
        not stat.S_ISREG(info.st_mode)
        or (info.st_dev, info.st_ino) != (expected_info.st_dev, expected_info.st_ino)
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise FinalizeError(
            "artifact transaction lock must be a private owned regular file"
        )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise FinalizeError(
            "artifact transaction lock fd is not exclusively held"
        ) from exc


def _engine_artifact(
    base: Mapping[str, Any], engine_name: str
) -> tuple[str, Mapping[str, Any]]:
    artifact_id = reconciler.ENGINE_ARTIFACT_IDS.get(engine_name)
    if artifact_id is None:
        raise FinalizeError(f"unknown reviewed engine: {engine_name}")
    rows = [row for row in base.get("artifacts", []) if row.get("id") == artifact_id]
    if len(rows) != 1 or rows[0].get("kind") != "tensorrt_engine":
        raise FinalizeError(f"invalid base artifact mapping for {engine_name}")
    return artifact_id, rows[0]


def _validate_transaction_scope(
    transaction: Mapping[str, Any],
    supplied_root: Path,
    transaction_path: Path,
    *,
    require_authority_digests: bool = True,
) -> None:
    """Re-derive every mutable path from pinned base authority."""

    root = _artifact_root(supplied_root)
    if transaction.get("artifact_root") != str(root):
        raise FinalizeError("transaction artifact root differs from recovery root")
    base_record = _required_mapping(transaction.get("base_manifest"), "base manifest")
    base_path = (
        Path(str(base_record.get("path") or "")).expanduser().resolve(strict=True)
    )
    if base_path != BASE_MANIFEST_AUTHORITY.expanduser().resolve(strict=True):
        raise FinalizeError(
            "transaction base-manifest path is not the tracked authority"
        )
    if require_authority_digests and _sha256(base_path) != base_record.get("sha256"):
        raise FinalizeError("transaction base-manifest digest drifted")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, Mapping):
        raise FinalizeError("transaction base manifest must be a mapping")
    engine_name = str(transaction.get("engine") or "")
    artifact_id, artifact = _engine_artifact(base, engine_name)
    if transaction.get("artifact_id") != artifact_id:
        raise FinalizeError("transaction artifact ID differs from base authority")
    virtual_output = validator._relative_path(artifact.get("output"))
    host_output = validator._physical_path(virtual_output, root)
    try:
        host_output.relative_to(root / "models/engines")
    except ValueError as exc:
        raise FinalizeError("transaction engine output escapes engine root") from exc
    if transaction.get("engine_output") != str(host_output):
        raise FinalizeError("transaction host engine path differs from base authority")
    if transaction.get("container_engine_output") != str(
        Path("/workspace") / virtual_output
    ):
        raise FinalizeError(
            "transaction container engine path differs from base authority"
        )
    contract_record = _required_mapping(
        transaction.get("source_contracts"), "source contracts"
    )
    contract_path = (
        Path(str(contract_record.get("path") or "")).expanduser().resolve(strict=True)
    )
    if contract_path != SOURCE_CONTRACTS_AUTHORITY.expanduser().resolve(strict=True):
        raise FinalizeError(
            "transaction source-contract path is not the tracked authority"
        )
    if require_authority_digests and _sha256(contract_path) != contract_record.get(
        "sha256"
    ):
        raise FinalizeError("transaction source-contract digest drifted")
    cohort_pruned = isinstance(transaction.get("retention"), Mapping) and transaction[
        "retention"
    ].get("state") in {"cohort_pruning", "cohort_pruned"}
    for key in ("prior_engine", "prior_realization"):
        record = _required_mapping(transaction.get(key), key.replace("_", " "))
        if not bool(record.get("exists")):
            if record.get("snapshot"):
                raise FinalizeError(f"absent {key} must not declare a snapshot")
            continue
        snapshot_name = str(record.get("snapshot") or "")
        snapshot_path = validator._bounded_private_path(
            transaction_path.parent / snapshot_name,
            transaction_path.parent,
            f"{key} snapshot",
        )
        if (
            cohort_pruned
            and not snapshot_path.exists()
            and not snapshot_path.is_symlink()
        ):
            continue
        if (
            snapshot_path.parent != transaction_path.parent
            or snapshot_path.is_symlink()
            or not snapshot_path.is_file()
            or snapshot_path.stat().st_size != record.get("size_bytes")
            or _sha256(snapshot_path) != record.get("sha256")
            or stat.S_IMODE(snapshot_path.stat().st_mode) != 0o600
        ):
            raise FinalizeError(f"{key} snapshot is missing, unsafe, or changed")


def _maintenance_inventory(root: Path) -> list[dict[str, str]]:
    evidence_root = root / "models/engine_maintenance"
    if not evidence_root.exists():
        return []
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise FinalizeError(f"maintenance evidence root is unsafe: {evidence_root}")
    manifests = sorted(evidence_root.glob("*/manifest.json"))
    if len(manifests) > 4096:
        raise FinalizeError("maintenance evidence inventory exceeds safety bound")
    records: list[dict[str, str]] = []
    for path in manifests:
        bounded = validator._bounded_private_path(
            path, evidence_root, "maintenance manifest"
        )
        validator._load_private_json(bounded, "maintenance manifest")
        records.append(
            {
                "path": bounded.relative_to(evidence_root).as_posix(),
                "sha256": _sha256(bounded),
            }
        )
    return records


def _candidate_prefix(engine: Path) -> str:
    return f".{engine.name}.building-"


def _hidden_candidate_record(path: Path) -> dict[str, Any]:
    """Describe one hidden candidate without following a symlink."""

    before = path.lstat()
    if stat.S_ISREG(before.st_mode):
        kind = "regular"
    elif stat.S_ISLNK(before.st_mode):
        kind = "symlink"
    else:
        kind = "other"
    record: dict[str, Any] = {
        "name": path.name,
        "kind": kind,
        "device": before.st_dev,
        "inode": before.st_ino,
        "uid": before.st_uid,
        "gid": before.st_gid,
        "nlink": before.st_nlink,
        "mode": f"{stat.S_IMODE(before.st_mode):04o}",
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
    }
    if kind == "regular":
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise FinalizeError(
                f"hidden candidate cannot be opened without following links: {path}: {exc}"
            ) from exc
        try:
            opened_before = os.fstat(descriptor)
            if not stat.S_ISREG(opened_before.st_mode) or (
                opened_before.st_dev,
                opened_before.st_ino,
            ) != (before.st_dev, before.st_ino):
                raise FinalizeError(
                    f"hidden candidate changed before secure open: {path}"
                )
            digest = hashlib.sha256()
            while block := os.read(descriptor, 4 * 1024 * 1024):
                digest.update(block)
            opened_after = os.fstat(descriptor)
            if (
                opened_before.st_dev,
                opened_before.st_ino,
                opened_before.st_size,
                opened_before.st_mtime_ns,
                opened_before.st_ctime_ns,
                stat.S_IMODE(opened_before.st_mode),
                opened_before.st_uid,
                opened_before.st_gid,
                opened_before.st_nlink,
            ) != (
                opened_after.st_dev,
                opened_after.st_ino,
                opened_after.st_size,
                opened_after.st_mtime_ns,
                opened_after.st_ctime_ns,
                stat.S_IMODE(opened_after.st_mode),
                opened_after.st_uid,
                opened_after.st_gid,
                opened_after.st_nlink,
            ):
                raise FinalizeError(f"hidden candidate changed while hashed: {path}")
            record["size_bytes"] = opened_before.st_size
            record["sha256"] = digest.hexdigest()
        finally:
            os.close(descriptor)
        after = path.lstat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            stat.S_IMODE(before.st_mode),
            before.st_uid,
            before.st_gid,
            before.st_nlink,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            stat.S_IMODE(after.st_mode),
            after.st_uid,
            after.st_gid,
            after.st_nlink,
        ):
            raise FinalizeError(f"hidden candidate changed while inventoried: {path}")
    elif kind == "symlink":
        record["link_target"] = os.readlink(path)
    return record


def _hidden_candidate_inventory(engine: Path) -> list[dict[str, Any]]:
    parent = engine.parent
    if parent.is_symlink() or not parent.is_dir():
        raise FinalizeError(f"engine directory is missing or unsafe: {parent}")
    prefix = _candidate_prefix(engine)
    paths = sorted(
        (path for path in parent.iterdir() if path.name.startswith(prefix)),
        key=lambda path: path.name,
    )
    if len(paths) > 4096:
        raise FinalizeError("hidden engine candidate inventory exceeds safety bound")
    return [_hidden_candidate_record(path) for path in paths]


def _parse_utc_timestamp_ns(value: object, label: str) -> int:
    raw = str(value or "")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FinalizeError(f"{label} is not an ISO UTC timestamp: {raw!r}") from exc
    if parsed.tzinfo is None:
        raise FinalizeError(f"{label} lacks a timezone: {raw!r}")
    return int(parsed.timestamp() * 1_000_000_000)


def _container_transaction_manifest(transaction_path: Path, root: Path) -> str:
    relative = transaction_path.relative_to(root / "models")
    return str(Path("/workspace/DS9/models") / relative)


def _new_target_maintenance_manifests(
    *,
    transaction_path: Path,
    transaction: Mapping[str, Any],
    prepared_transaction_sha256: str | None,
    explicit_residue_authority: Mapping[str, Any] | None = None,
    require_prior_integrity: bool = False,
) -> tuple[list[tuple[Path, Mapping[str, Any]]], list[str]]:
    root = Path(str(transaction["artifact_root"]))
    evidence_root = root / "models/engine_maintenance"
    prior_rows = transaction.get("maintenance_inventory", [])
    if not isinstance(prior_rows, list):
        raise FinalizeError("transaction maintenance inventory is malformed")
    prior: dict[str, str] = {}
    for row in prior_rows:
        if not isinstance(row, Mapping):
            raise FinalizeError("transaction maintenance inventory row is malformed")
        relative = str(row.get("path") or "")
        digest = str(row.get("sha256") or "")
        if (
            not relative
            or relative in prior
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise FinalizeError("transaction maintenance inventory is ambiguous")
        prior[relative] = digest

    if not evidence_root.exists():
        return [], []
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise FinalizeError(f"maintenance evidence root is unsafe: {evidence_root}")
    raw_manifests = sorted(evidence_root.glob("*/manifest.json"))
    if len(raw_manifests) > 4096:
        raise FinalizeError("maintenance evidence inventory exceeds safety bound")

    matches: list[tuple[Path, Mapping[str, Any]]] = []
    blockers: list[str] = []
    seen_prior: set[str] = set()
    for raw in raw_manifests:
        try:
            path = validator._bounded_private_path(
                raw, evidence_root, "maintenance manifest"
            )
            relative = path.relative_to(evidence_root).as_posix()
            if relative in prior:
                seen_prior.add(relative)
                if require_prior_integrity and _sha256(path) != prior[relative]:
                    blockers.append(
                        f"pre-snapshot maintenance manifest changed: {path}"
                    )
                continue
            payload = validator._load_private_json(path, "maintenance manifest")
            if payload.get("engine") != transaction.get("engine") or payload.get(
                "target"
            ) != transaction.get("container_engine_output"):
                continue
            if (
                payload.get("schema_version") != 1
                or payload.get("contract") != "noesis.ds9.engine_maintenance"
            ):
                blockers.append(
                    f"new target maintenance manifest has wrong contract: {path}"
                )
                continue
            run_id = str(payload.get("run_id") or "")
            safe_engine = re.sub(
                r"[^A-Za-z0-9_.-]+", "-", str(transaction.get("engine") or "")
            ).strip("-")
            if (
                not _RUN_ID_RE.fullmatch(run_id)
                or path.parent.name != f"{run_id}-{safe_engine}"
            ):
                blockers.append(
                    f"new target maintenance manifest has unsafe run identity: {path}"
                )
                continue

            host_binding = (payload.get("metadata") or {}).get("host_transaction")
            expected_container_manifest = _container_transaction_manifest(
                transaction_path, root
            )
            expected_binding = {
                "transaction_id": transaction.get("transaction_id"),
                "transaction_sha256": prepared_transaction_sha256,
                "transaction_manifest": expected_container_manifest,
            }
            if isinstance(host_binding, Mapping) and prepared_transaction_sha256:
                if dict(host_binding) != expected_binding:
                    blockers.append(
                        f"new target maintenance manifest host binding differs: {path}"
                    )
                    continue
            elif isinstance(explicit_residue_authority, Mapping):
                if not (
                    path == explicit_residue_authority.get("maintenance_manifest")
                    and _sha256(path)
                    == explicit_residue_authority.get("maintenance_sha256")
                ):
                    blockers.append(
                        f"unbound maintenance manifest differs from explicit residue authority: {path}"
                    )
                    continue
            else:
                blockers.append(
                    f"new target maintenance manifest lacks exact host binding: {path}"
                )
                continue
            matches.append((path, payload))
        except Exception as exc:
            blockers.append(f"{raw}: {type(exc).__name__}: {exc}")
    if require_prior_integrity:
        blockers.extend(
            f"pre-snapshot maintenance manifest disappeared: {relative}"
            for relative in sorted(set(prior) - seen_prior)
        )
    return matches, blockers


def _verify_candidate_plan_maintenance(
    *,
    transaction: Mapping[str, Any],
    row: Mapping[str, Any],
) -> None:
    root = Path(str(transaction["artifact_root"]))
    evidence_root = root / "models/engine_maintenance"
    manifest = validator._bounded_private_path(
        Path(str(row.get("maintenance_manifest") or "")),
        evidence_root,
        "candidate cleanup maintenance manifest",
    )
    expected_sha = str(row.get("maintenance_manifest_sha256") or "")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
        or _sha256(manifest) != expected_sha
    ):
        raise FinalizeError(
            f"candidate cleanup maintenance manifest changed: {manifest}"
        )
    payload = validator._load_private_json(
        manifest, "candidate cleanup maintenance manifest"
    )
    if (
        payload.get("engine") != transaction.get("engine")
        or payload.get("target") != transaction.get("container_engine_output")
        or payload.get("run_id") != row.get("run_id")
    ):
        raise FinalizeError(
            f"candidate cleanup maintenance manifest identity changed: {manifest}"
        )


def _resume_prepared_candidate_cleanup(
    *,
    transaction_path: Path,
    transaction: dict[str, Any],
    cleanup: Mapping[str, Any],
) -> dict[str, Any]:
    engine = Path(str(transaction["engine_output"]))
    prefix = _candidate_prefix(engine)
    planned_raw = cleanup.get("planned")
    if not isinstance(planned_raw, list):
        raise FinalizeError("prepared candidate cleanup has no sealed plan")
    planned: list[Mapping[str, Any]] = []
    planned_names: set[str] = set()
    for raw in planned_raw:
        if not isinstance(raw, Mapping):
            raise FinalizeError("prepared candidate cleanup plan row is malformed")
        path = validator._lexical_absolute(Path(str(raw.get("path") or "")))
        if path.parent != engine.parent or not path.name.startswith(prefix):
            raise FinalizeError("prepared candidate cleanup path escaped target scope")
        if path.name != f"{prefix}{raw.get('run_id')}":
            raise FinalizeError("prepared candidate cleanup path/run ID differs")
        _verify_candidate_plan_maintenance(transaction=transaction, row=raw)
        if path.name in planned_names:
            raise FinalizeError("prepared candidate cleanup plan is ambiguous")
        planned_names.add(path.name)
        planned.append(raw)

    snapshot_rows = transaction.get("hidden_candidate_inventory", [])
    if not isinstance(snapshot_rows, list) or any(
        not isinstance(row, Mapping) for row in snapshot_rows
    ):
        raise FinalizeError("prepared cleanup has malformed candidate inventory")
    snapshot = {str(row.get("name")): row for row in snapshot_rows}
    if "" in snapshot or len(snapshot) != len(snapshot_rows):
        raise FinalizeError("prepared cleanup candidate inventory is ambiguous")
    snapshot_names = set(snapshot)
    current = {str(row["name"]): row for row in _hidden_candidate_inventory(engine)}
    for name, expected in snapshot.items():
        observed = current.get(name)
        if observed is None:
            raise FinalizeError(
                f"pre-snapshot hidden candidate disappeared during cleanup: {name}"
            )
        if dict(observed) != dict(expected):
            raise FinalizeError(
                f"pre-snapshot hidden candidate changed during cleanup: {name}"
            )
    unknown = sorted(set(current) - snapshot_names - planned_names)
    if unknown:
        raise FinalizeError(
            "unowned candidate appeared while resuming cleanup: " + ", ".join(unknown)
        )

    removed: list[dict[str, Any]] = []
    identity_keys = (
        "kind",
        "device",
        "inode",
        "uid",
        "gid",
        "nlink",
        "mode",
        "size_bytes",
        "sha256",
        "mtime_ns",
        "ctime_ns",
    )
    for planned_row in planned:
        path = Path(str(planned_row["path"]))
        if path.exists() or path.is_symlink():
            observed = _hidden_candidate_record(path)
            for key in identity_keys:
                if observed.get(key) != planned_row.get(key):
                    raise FinalizeError(
                        f"hidden candidate changed while resuming cleanup ({key}): {path}"
                    )
            path.unlink()
            fsync_directory(path.parent)
            status = "removed_during_recovery"
        else:
            status = "recovered_after_interrupted_unlink"
        if path.exists() or path.is_symlink():
            raise FinalizeError(
                f"hidden candidate remains after resumed cleanup: {path}"
            )
        removed.append(
            {
                **dict(planned_row),
                "removal_status": status,
                "removed_at_utc": _utc_now(),
            }
        )

    result = {
        "state": "complete",
        "completed_at_utc": _utc_now(),
        "resumed_from_prepared": True,
        "authority": cleanup.get("authority"),
        "reason": cleanup.get("reason"),
        "removed": removed,
        "preserved_preexisting": list(cleanup.get("preserved_preexisting", [])),
        "blockers": [],
    }
    transaction["candidate_cleanup"] = result
    transaction["updated_at_utc"] = _utc_now()
    _write_json_atomic(transaction_path, transaction)
    return result


def _cleanup_hidden_candidates(
    *,
    transaction_path: Path,
    transaction: dict[str, Any],
    prepared_transaction_sha256: str | None,
    explicit_residue_authority: Mapping[str, Any] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Remove only transaction-created hidden candidates and persist evidence."""

    engine = Path(str(transaction["engine_output"]))
    prefix = _candidate_prefix(engine)
    snapshot_rows = transaction.get("hidden_candidate_inventory")
    legacy_mode = snapshot_rows is None
    if legacy_mode:
        snapshot_rows = []
    if not isinstance(snapshot_rows, list):
        raise FinalizeError("hidden candidate snapshot inventory is malformed")
    snapshot: dict[str, Mapping[str, Any]] = {}
    for row in snapshot_rows:
        if not isinstance(row, Mapping):
            raise FinalizeError("hidden candidate snapshot row is malformed")
        name = str(row.get("name") or "")
        if not name.startswith(prefix) or name in snapshot:
            raise FinalizeError("hidden candidate snapshot inventory is ambiguous")
        snapshot[name] = row

    current_rows = _hidden_candidate_inventory(engine)
    current = {str(row["name"]): row for row in current_rows}
    previous = transaction.get("candidate_cleanup")
    if isinstance(previous, Mapping) and previous.get("state") == "prepared":
        try:
            return _resume_prepared_candidate_cleanup(
                transaction_path=transaction_path,
                transaction=transaction,
                cleanup=previous,
            )
        except Exception as exc:
            result = {
                **dict(previous),
                "state": "manual_recovery_required",
                "failed_at_utc": _utc_now(),
                "blockers": [f"{type(exc).__name__}: {exc}"],
            }
            transaction["candidate_cleanup"] = result
            transaction["updated_at_utc"] = _utc_now()
            _write_json_atomic(transaction_path, transaction)
            return result
    if (
        isinstance(previous, Mapping)
        and previous.get("state") == "manual_recovery_required"
    ):
        return dict(previous)
    if isinstance(previous, Mapping) and previous.get("state") == "complete":
        removed_names = {
            Path(str(row.get("path") or "")).name
            for row in previous.get("removed", [])
            if isinstance(row, Mapping)
        }
        if any(name in current for name in removed_names):
            raise FinalizeError("a previously removed transaction candidate reappeared")
        for row in previous.get("removed", []):
            if not isinstance(row, Mapping):
                raise FinalizeError(
                    "completed candidate cleanup removal evidence is malformed"
                )
            _verify_candidate_plan_maintenance(transaction=transaction, row=row)
        preserved_rows = previous.get("preserved_preexisting", [])
        if not isinstance(preserved_rows, list) or any(
            not isinstance(row, Mapping) for row in preserved_rows
        ):
            raise FinalizeError(
                "completed candidate cleanup has malformed preserved evidence"
            )
        preserved = {str(row.get("name")): row for row in preserved_rows}
        if "" in preserved or len(preserved) != len(preserved_rows):
            raise FinalizeError(
                "completed candidate cleanup preserved evidence is ambiguous"
            )
        for name, expected in preserved.items():
            observed = current.get(name)
            if observed is None or dict(observed) != dict(expected):
                raise FinalizeError(
                    f"preserved pre-snapshot candidate changed after cleanup: {name}"
                )
        unowned = sorted(set(current) - set(preserved))
        if unowned:
            raise FinalizeError(
                "unowned hidden candidate exists after completed cleanup: "
                + ", ".join(unowned)
            )
        return dict(previous)
    blockers: list[str] = []
    preserved: list[dict[str, Any]] = []
    for name, prior in snapshot.items():
        observed = current.get(name)
        if observed is None:
            blockers.append(f"pre-snapshot hidden candidate disappeared: {name}")
        elif dict(observed) != dict(prior):
            blockers.append(f"pre-snapshot hidden candidate changed: {name}")
        else:
            preserved.append(dict(observed))

    matches, manifest_blockers = _new_target_maintenance_manifests(
        transaction_path=transaction_path,
        transaction=transaction,
        prepared_transaction_sha256=prepared_transaction_sha256,
        explicit_residue_authority=explicit_residue_authority,
        require_prior_integrity=True,
    )
    blockers.extend(manifest_blockers)
    if len(matches) > 1:
        blockers.append(
            "multiple post-snapshot maintenance manifests match one host transaction"
        )

    authorized: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for manifest_path, payload in matches:
        run_id = str(payload["run_id"])
        name = f"{prefix}{run_id}"
        if name in authorized:
            blockers.append(
                f"duplicate maintenance run authorizes hidden candidate: {name}"
            )
        else:
            authorized[name] = (manifest_path, payload)

    new_names = sorted(set(current) - set(snapshot))
    for name in new_names:
        if name not in authorized:
            blockers.append(f"unowned post-snapshot hidden candidate: {name}")

    planned: list[dict[str, Any]] = []
    transaction_created_ns = _parse_utc_timestamp_ns(
        transaction.get("created_at_utc"), "transaction creation time"
    )
    for name, (manifest_path, payload) in authorized.items():
        observed = current.get(name)
        if observed is None:
            continue
        if name in snapshot:
            blockers.append(
                f"maintenance run collides with a preexisting candidate: {name}"
            )
            continue
        if (
            observed.get("kind") != "regular"
            or observed.get("uid") != os.getuid()
            or observed.get("nlink") != 1
        ):
            blockers.append(
                f"authorized hidden candidate is symlink, nonregular, foreign-owned, or hardlinked: {name}"
            )
            continue
        manifest_created_ns = _parse_utc_timestamp_ns(
            payload.get("created_at_utc"), "maintenance manifest creation time"
        )
        if manifest_created_ns < transaction_created_ns:
            blockers.append(
                f"maintenance manifest predates host transaction: {manifest_path}"
            )
            continue
        if (
            int(observed.get("mtime_ns") or 0) < manifest_created_ns
            or int(observed.get("ctime_ns") or 0) < manifest_created_ns
        ):
            blockers.append(
                f"hidden candidate predates its maintenance manifest: {name}"
            )
            continue
        if legacy_mode:
            authority = explicit_residue_authority
            if not isinstance(authority, Mapping):
                blockers.append(
                    f"pre-inventory transaction requires explicit residue authority: {name}"
                )
                continue
            if (
                observed.get("sha256") != authority.get("candidate_sha256")
                or observed.get("size_bytes") != authority.get("candidate_size_bytes")
                or int(observed.get("mtime_ns") or 0) < transaction_created_ns
                or int(observed.get("ctime_ns") or 0) < transaction_created_ns
            ):
                blockers.append(
                    f"legacy hidden candidate does not match reviewed bytes/time: {name}"
                )
                continue
        planned.append(
            {
                **dict(observed),
                "path": str(engine.parent / name),
                "maintenance_manifest": str(manifest_path),
                "maintenance_manifest_sha256": _sha256(manifest_path),
                "run_id": payload["run_id"],
                "proof": "explicit_residue_cas"
                if legacy_mode
                else "host_transaction_binding",
            }
        )

    if blockers:
        result = {
            "state": "manual_recovery_required",
            "audited_at_utc": _utc_now(),
            "reason": reason
            or (
                explicit_residue_authority.get("reason")
                if isinstance(explicit_residue_authority, Mapping)
                else None
            ),
            "removed": [],
            "preserved_preexisting": preserved,
            "blockers": blockers,
        }
        transaction["candidate_cleanup"] = result
        transaction["updated_at_utc"] = _utc_now()
        _write_json_atomic(transaction_path, transaction)
        return result

    if isinstance(explicit_residue_authority, Mapping):
        authority_evidence: dict[str, Any] = {
            "kind": "explicit_residue_cas",
            "rolled_back_transaction_sha256": str(
                explicit_residue_authority.get("rolled_back_transaction_sha256") or ""
            ),
            "maintenance_manifest": str(
                explicit_residue_authority.get("maintenance_manifest") or ""
            ),
            "maintenance_sha256": explicit_residue_authority.get("maintenance_sha256"),
            "candidate_sha256": explicit_residue_authority.get("candidate_sha256"),
            "candidate_size_bytes": explicit_residue_authority.get(
                "candidate_size_bytes"
            ),
        }
    else:
        authority_evidence = {
            "kind": "host_transaction_binding",
            "transaction_id": transaction.get("transaction_id"),
            "transaction_sha256": prepared_transaction_sha256,
            "transaction_manifest": _container_transaction_manifest(
                transaction_path, Path(str(transaction["artifact_root"]))
            ),
        }
    cleanup_reason = reason or (
        str(explicit_residue_authority.get("reason") or "")
        if isinstance(explicit_residue_authority, Mapping)
        else ""
    )
    transaction["candidate_cleanup"] = {
        "state": "prepared",
        "prepared_at_utc": _utc_now(),
        "reason": cleanup_reason,
        "authority": authority_evidence,
        "planned": planned,
        "preserved_preexisting": preserved,
        "blockers": [],
    }
    transaction["updated_at_utc"] = _utc_now()
    _write_json_atomic(transaction_path, transaction)

    removed: list[dict[str, Any]] = []
    try:
        for planned_row in planned:
            path = Path(str(planned_row["path"]))
            observed = _hidden_candidate_record(path)
            for key in (
                "kind",
                "device",
                "inode",
                "uid",
                "gid",
                "nlink",
                "mode",
                "size_bytes",
                "sha256",
                "mtime_ns",
                "ctime_ns",
            ):
                if observed.get(key) != planned_row.get(key):
                    raise FinalizeError(
                        f"hidden candidate changed before removal ({key}): {path}"
                    )
            path.unlink()
            fsync_directory(path.parent)
            if path.exists() or path.is_symlink():
                raise FinalizeError(f"hidden candidate remains after removal: {path}")
            removed.append(
                {
                    **planned_row,
                    "removed_at_utc": _utc_now(),
                }
            )
    except Exception as exc:
        result = {
            "state": "manual_recovery_required",
            "failed_at_utc": _utc_now(),
            "reason": cleanup_reason,
            "authority": authority_evidence,
            "planned": planned,
            "removed": removed,
            "preserved_preexisting": preserved,
            "blockers": [f"{type(exc).__name__}: {exc}"],
        }
        transaction["candidate_cleanup"] = result
        transaction["updated_at_utc"] = _utc_now()
        _write_json_atomic(transaction_path, transaction)
        return result

    result = {
        "state": "complete",
        "completed_at_utc": _utc_now(),
        "reason": cleanup_reason,
        "authority": authority_evidence,
        "removed": removed,
        "preserved_preexisting": preserved,
        "blockers": [],
    }
    transaction["candidate_cleanup"] = result
    transaction["updated_at_utc"] = _utc_now()
    _write_json_atomic(transaction_path, transaction)
    return result


def cleanup_residue(
    *,
    transaction_manifest: Path,
    expected_transaction_sha256: str,
    maintenance_manifest: Path,
    expected_maintenance_sha256: str,
    expected_candidate_sha256: str,
    expected_candidate_size_bytes: int,
    reason: str,
) -> dict[str, Any]:
    """CAS-authorize one pre-binding residue cleanup without a fallback rule."""

    transaction_path = validator._lexical_absolute(transaction_manifest)
    if _sha256(transaction_path) != expected_transaction_sha256:
        raise FinalizeError(
            "engine finalize transaction changed before residue cleanup"
        )
    transaction_path, transaction = _load_transaction(
        transaction_path,
        expected_transaction_sha256,
        allow_rolled_back=True,
    )
    if transaction.get("state") != "rolled_back":
        raise FinalizeError(
            "explicit residue cleanup is limited to a rolled-back transaction"
        )
    sealed_cleanup = transaction.get("candidate_cleanup")
    if (
        isinstance(sealed_cleanup, Mapping)
        and sealed_cleanup.get("state") == "complete"
    ):
        raise FinalizeError(
            "explicit residue cleanup is already sealed; use recover for idempotent audit"
        )
    root = Path(str(transaction["artifact_root"]))
    evidence_root = root / "models/engine_maintenance"
    maintenance_path = validator._bounded_private_path(
        maintenance_manifest, evidence_root, "explicit maintenance manifest"
    )
    if _sha256(maintenance_path) != expected_maintenance_sha256:
        raise FinalizeError("maintenance manifest changed before residue cleanup")
    maintenance = validator._load_private_json(
        maintenance_path, "explicit maintenance manifest"
    )
    if (
        maintenance.get("schema_version") != 1
        or maintenance.get("contract") != "noesis.ds9.engine_maintenance"
        or maintenance.get("engine") != transaction.get("engine")
        or maintenance.get("target") != transaction.get("container_engine_output")
    ):
        raise FinalizeError(
            "explicit maintenance manifest does not belong to the transaction"
        )
    run_id = str(maintenance.get("run_id") or "")
    if not _RUN_ID_RE.fullmatch(run_id):
        raise FinalizeError("explicit maintenance manifest has an unsafe run ID")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_candidate_sha256):
        raise FinalizeError("expected candidate SHA-256 is invalid")
    if expected_candidate_size_bytes <= 0:
        raise FinalizeError("expected candidate size must be positive")
    authority = {
        "rolled_back_transaction_sha256": expected_transaction_sha256,
        "maintenance_manifest": maintenance_path,
        "maintenance_sha256": expected_maintenance_sha256,
        "candidate_sha256": expected_candidate_sha256,
        "candidate_size_bytes": expected_candidate_size_bytes,
        "reason": reason,
    }
    result = _cleanup_hidden_candidates(
        transaction_path=transaction_path,
        transaction=transaction,
        prepared_transaction_sha256=None,
        explicit_residue_authority=authority,
        reason=reason,
    )
    if result.get("state") != "complete":
        raise FinalizeError(f"explicit residue cleanup refused: {result}")
    return {
        "status": "residue_cleaned",
        "transaction_manifest": str(transaction_path),
        "transaction_sha256": _sha256(transaction_path),
        "candidate_cleanup": result,
    }


def _copy_snapshot(source: Path, destination: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_file():
        raise FinalizeError(f"snapshot source is missing or unsafe: {source}")
    before = source.stat()
    _copy_private(source, destination)
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise FinalizeError(f"snapshot source changed while copied: {source}")
    source_hash = _sha256(source)
    if (
        destination.stat().st_size != before.st_size
        or _sha256(destination) != source_hash
    ):
        raise FinalizeError(f"snapshot copy verification failed: {source}")
    return {
        "exists": True,
        "sha256": source_hash,
        "size_bytes": before.st_size,
        "mode": f"{stat.S_IMODE(before.st_mode):04o}",
        "snapshot": destination.name,
    }


def snapshot(
    *,
    engine_name: str,
    artifact_root: Path,
    base_manifest_path: Path,
    validation_profile: str,
) -> dict[str, Any]:
    root = _artifact_root(artifact_root)
    base_manifest_path = base_manifest_path.expanduser().resolve(strict=True)
    if base_manifest_path != BASE_MANIFEST_AUTHORITY.expanduser().resolve(strict=True):
        raise FinalizeError(
            "base manifest must be the tracked DS9 asset-manifest authority"
        )
    base = yaml.safe_load(base_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(base, Mapping):
        raise FinalizeError("base manifest must be a mapping")
    artifact_id, artifact = _engine_artifact(base, engine_name)
    if validation_profile == "engine_finalize":
        validation_profile = f"artifact:{artifact_id}"
    source_contracts = SOURCE_CONTRACTS_AUTHORITY.resolve(strict=True)
    virtual_output = validator._relative_path(artifact["output"])
    engine = validator._physical_path(virtual_output, root)
    try:
        engine.relative_to(root / "models/engines")
    except ValueError as exc:
        raise FinalizeError(
            f"engine output escapes the DS9 engine root: {engine}"
        ) from exc
    if engine.is_symlink():
        raise FinalizeError(f"engine output must not be a symlink: {engine}")

    transaction_root = validator._bounded_private_path(
        root / TRANSACTION_ROOT_RELATIVE, root, "engine finalize transaction root"
    )
    if transaction_root.exists():
        info = transaction_root.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise FinalizeError(
                "existing engine finalize transaction root must already be "
                f"owner-private mode 0700: {transaction_root}"
            )
    else:
        transaction_root.parent.mkdir(parents=True, exist_ok=True)
        transaction_root.mkdir(mode=0o700)
    run_id = new_run_id()
    run_dir = transaction_root / f"{run_id}-{engine_name}"
    run_dir.mkdir(mode=0o700)
    os.chmod(run_dir, 0o700)
    engine_record: dict[str, Any] = {"exists": False}
    if engine.exists():
        engine_record = _copy_snapshot(engine, run_dir / "prior_engine.bin")

    realization = root / reconciler.REALIZATION_FILENAME
    realization_record: dict[str, Any] = {"exists": False, "expected": "missing"}
    if realization.exists() or realization.is_symlink():
        validator._bounded_private_path(realization, root, "asset realization")
        validator._load_private_json(realization, "asset realization")
        realization_record = _copy_snapshot(
            realization, run_dir / "prior_realization.json"
        )
        realization_record["expected"] = realization_record["sha256"]

    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": CONTRACT,
        "transaction_id": run_id,
        "state": "prepared",
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "engine": engine_name,
        "artifact_id": artifact_id,
        "artifact_root": str(root),
        "engine_output": str(engine),
        "container_engine_output": str(Path("/workspace") / virtual_output),
        "base_manifest": {
            "path": str(base_manifest_path),
            "sha256": _sha256(base_manifest_path),
        },
        "source_contracts": {
            "path": str(source_contracts),
            "sha256": _sha256(source_contracts),
        },
        "validation_profile": validation_profile,
        "prior_engine": engine_record,
        "prior_realization": realization_record,
        "maintenance_inventory": _maintenance_inventory(root),
        "hidden_candidate_inventory": _hidden_candidate_inventory(engine),
    }
    manifest = run_dir / "transaction.json"
    _write_json_atomic(manifest, payload)
    fsync_directory(run_dir)
    fsync_directory(transaction_root)
    result = {
        "transaction_manifest": str(manifest),
        "transaction_sha256": _sha256(manifest),
        "transaction_id": run_id,
        "engine": engine_name,
        "prior_realization": realization_record["expected"],
    }
    return result


def _load_transaction(
    transaction_manifest: Path,
    expected_sha256: str,
    *,
    allow_rolled_back: bool = False,
    required_state: str = "prepared",
) -> tuple[Path, dict[str, Any]]:
    path = validator._lexical_absolute(transaction_manifest)
    payload = dict(validator._load_private_json(path, "engine finalize transaction"))
    if payload.get("schema_version") != 1 or payload.get("contract") != CONTRACT:
        raise FinalizeError("engine finalize transaction contract is invalid")
    root = _artifact_root(Path(str(payload.get("artifact_root") or "")))
    expected_root = root / TRANSACTION_ROOT_RELATIVE
    validator._bounded_private_path(path, expected_root, "engine finalize transaction")
    if path.parent.parent != expected_root:
        raise FinalizeError("engine finalize transaction has an unexpected path")
    _validate_transaction_scope(payload, root, path, require_authority_digests=False)
    if allow_rolled_back and payload.get("state") == "rolled_back":
        return path, payload
    if _sha256(path) != expected_sha256:
        raise FinalizeError("engine finalize transaction changed before use")
    if payload.get("state") != required_state:
        raise FinalizeError(f"engine finalize transaction is not {required_state}")
    return path, payload


def _restore_file(
    snapshot_path: Path, target: Path, expected: Mapping[str, Any]
) -> None:
    if (
        snapshot_path.is_symlink()
        or not snapshot_path.is_file()
        or snapshot_path.stat().st_size != expected.get("size_bytes")
        or _sha256(snapshot_path) != expected.get("sha256")
    ):
        raise FinalizeError(f"prior snapshot is missing or changed: {snapshot_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.host-rollback-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        _copy_private(snapshot_path, temporary)
        os.chmod(temporary, int(str(expected.get("mode") or "0600"), 8))
        fsync_file(temporary)
        os.replace(temporary, target)
        fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
    if target.stat().st_size != expected.get("size_bytes") or _sha256(
        target
    ) != expected.get("sha256"):
        raise FinalizeError(f"restored target failed verification: {target}")


def _candidate_record(maintenance: Mapping[str, Any]) -> Mapping[str, Any] | None:
    installed = maintenance.get("installed")
    if isinstance(installed, Mapping):
        return installed
    transaction = maintenance.get("install_transaction")
    if isinstance(transaction, Mapping) and isinstance(
        transaction.get("candidate"), Mapping
    ):
        return transaction["candidate"]
    candidate = maintenance.get("candidate")
    return candidate if isinstance(candidate, Mapping) else None


def _require_prior_agreement(
    transaction: Mapping[str, Any], maintenance: Mapping[str, Any]
) -> None:
    host_prior = _required_mapping(transaction.get("prior_engine"), "host prior engine")
    builder_prior = _required_mapping(maintenance.get("prior"), "builder prior engine")
    if bool(host_prior.get("exists")) != bool(builder_prior.get("exists")):
        raise PriorAgreementError(
            "builder prior presence differs from the prelaunch host snapshot"
        )
    if bool(host_prior.get("exists")):
        for key in ("sha256", "size_bytes", "mode"):
            if host_prior.get(key) != builder_prior.get(key):
                raise PriorAgreementError(
                    f"builder prior {key} differs from the prelaunch host snapshot"
                )


def _resolve_maintenance_manifest(
    transaction_path: Path,
    transaction: Mapping[str, Any],
    requested: Path | None,
    prepared_transaction_sha256: str,
) -> tuple[Path, Mapping[str, Any]]:
    root = Path(str(transaction["artifact_root"]))
    evidence_root = root / "models/engine_maintenance"
    matches, blockers = _new_target_maintenance_manifests(
        transaction_path=transaction_path,
        transaction=transaction,
        prepared_transaction_sha256=prepared_transaction_sha256,
        require_prior_integrity=True,
    )
    if blockers:
        raise FinalizeError("maintenance evidence audit failed: " + "; ".join(blockers))
    if len(matches) != 1:
        raise FinalizeError(
            "unable to identify exactly one new transaction-bound maintenance "
            f"manifest for {transaction.get('engine')}: found={len(matches)}"
        )
    path, payload = matches[0]
    if requested is not None:
        requested_path = validator._bounded_private_path(
            requested, evidence_root, "requested maintenance manifest"
        )
        if requested_path != path:
            raise FinalizeError(
                "requested maintenance manifest is not the sole post-snapshot authority"
            )
    return path, payload


def _prune_committed_engine_snapshots(
    transaction_path: Path,
    transaction: Mapping[str, Any],
    maintenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep one host rollback cohort and remove only redundant/older copies."""

    removed: list[str] = []
    freed = 0
    errors: list[str] = []

    # The host transaction retains the atomic engine+realization rollback cohort.
    # The builder's byte-identical prior is redundant after host commit.
    preserved = maintenance.get("preserved_prior")
    if isinstance(preserved, Mapping):
        container_path = Path(str(preserved.get("path") or ""))
        evidence_root = (
            Path(str(transaction["artifact_root"])) / "models/engine_maintenance"
        )
        try:
            container_root = Path("/workspace/DS9/models/engine_maintenance")
            relative = container_path.relative_to(container_root)
            path = evidence_root / relative
            path = validator._bounded_private_path(
                path, evidence_root, "builder preserved prior"
            )
            if path.is_symlink() or not path.is_file():
                raise FinalizeError(f"builder preserved prior is unsafe: {path}")
            if path.stat().st_size != preserved.get("size_bytes") or _sha256(
                path
            ) != preserved.get("sha256"):
                raise FinalizeError(
                    f"builder preserved prior hash/size drifted: {path}"
                )
            size = path.stat().st_size
            path.unlink()
            fsync_directory(path.parent)
            removed.append(str(path))
            freed += size
        except Exception as exc:
            errors.append(str(exc))

    # Bound full rollback cohorts to the newest committed transaction per engine.
    transaction_root = transaction_path.parent.parent
    for old_manifest in sorted(transaction_root.glob("*/transaction.json")):
        if old_manifest == transaction_path:
            continue
        try:
            old_manifest = validator._bounded_private_path(
                old_manifest, transaction_root, "older engine finalize transaction"
            )
            old = dict(
                validator._load_private_json(
                    old_manifest, "older engine finalize transaction"
                )
            )
            if (
                old.get("contract") != CONTRACT
                or old.get("state") != "committed"
                or old.get("engine") != transaction.get("engine")
            ):
                continue
            for key in ("prior_engine", "prior_realization"):
                record = old.get(key)
                if not isinstance(record, Mapping) or not bool(record.get("exists")):
                    continue
                path = old_manifest.parent / str(record.get("snapshot") or "")
                if path.is_symlink() or not path.is_file():
                    raise FinalizeError(f"older rollback snapshot is unsafe: {path}")
                if _sha256(path) != record.get("sha256"):
                    raise FinalizeError(f"older rollback snapshot hash drifted: {path}")
                size = path.stat().st_size
                path.unlink()
                removed.append(str(path))
                freed += size
            old["retention"] = {
                "state": "cohort_pruned",
                "pruned_at_utc": _utc_now(),
                "superseded_by": transaction.get("transaction_id"),
            }
            old["updated_at_utc"] = _utc_now()
            _write_json_atomic(old_manifest, old)
            fsync_directory(old_manifest.parent)
        except Exception as exc:
            errors.append(str(exc))
    return {
        "completed_at_utc": _utc_now(),
        "retained_host_rollback_cohort": True,
        "removed": removed,
        "freed_bytes": freed,
        "errors": errors,
    }


def _prune_completed_rollback_cohort(
    transaction_path: Path,
    transaction: dict[str, Any],
) -> dict[str, Any]:
    """Release rollback bytes once those exact bytes are current again."""

    planned: list[tuple[Path, int, str]] = []
    for key in ("prior_engine", "prior_realization"):
        record = transaction.get(key)
        if not isinstance(record, Mapping) or not bool(record.get("exists")):
            continue
        path = validator._bounded_private_path(
            transaction_path.parent / str(record.get("snapshot") or ""),
            transaction_path.parent,
            f"completed {key} snapshot",
        )
        if (
            path.parent != transaction_path.parent
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise FinalizeError(f"completed {key} snapshot is unsafe or changed")
        planned.append((path, path.stat().st_size, key))

    transaction["retention"] = {
        "state": "cohort_pruning",
        "started_at_utc": _utc_now(),
        "reason": "rollback cohort now duplicates restored current bytes",
        "planned": [key for _path, _size, key in planned],
    }
    transaction["updated_at_utc"] = _utc_now()
    _write_json_atomic(transaction_path, transaction)

    removed: list[str] = []
    freed = 0
    errors: list[str] = []
    for path, size, _key in planned:
        try:
            path.unlink()
            fsync_directory(path.parent)
            removed.append(str(path))
            freed += size
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    transaction["retention"] = {
        "state": "cohort_pruned",
        "completed_at_utc": _utc_now(),
        "reason": "rollback cohort now duplicates restored current bytes",
        "removed": removed,
        "freed_bytes": freed,
        "errors": errors,
    }
    transaction["updated_at_utc"] = _utc_now()
    _write_json_atomic(transaction_path, transaction)
    return dict(transaction["retention"])


def _rollback(
    *,
    transaction_path: Path,
    transaction: dict[str, Any],
    maintenance_manifest: Path | None,
    accepted_realization_sha256: str | None,
    reason: str,
    prepared_transaction_sha256: str | None = None,
    refuse_engine_restore: bool = False,
) -> dict[str, Any]:
    root = Path(transaction["artifact_root"])
    _validate_transaction_scope(
        transaction,
        root,
        transaction_path,
        require_authority_digests=False,
    )
    recorded_prepared_sha = str(transaction.get("prepared_transaction_sha256") or "")
    if prepared_transaction_sha256:
        if (
            recorded_prepared_sha
            and recorded_prepared_sha != prepared_transaction_sha256
        ):
            raise FinalizeError(
                "recorded prepared transaction digest differs from rollback authority"
            )
        if not recorded_prepared_sha:
            transaction["prepared_transaction_sha256"] = prepared_transaction_sha256
            transaction["updated_at_utc"] = _utc_now()
            _write_json_atomic(transaction_path, transaction)
            recorded_prepared_sha = prepared_transaction_sha256
    engine = Path(transaction["engine_output"])
    prior_engine = _required_mapping(transaction.get("prior_engine"), "prior engine")
    rollback: dict[str, Any] = {
        "requested_at_utc": _utc_now(),
        "reason": reason,
        "engine": "pending",
        "realization": "pending",
        "status": "manual_recovery_required",
    }

    current_engine_hash = (
        _sha256(engine) if engine.is_file() and not engine.is_symlink() else None
    )
    prior_engine_hash = (
        str(prior_engine.get("sha256")) if bool(prior_engine.get("exists")) else None
    )
    if refuse_engine_restore:
        rollback["engine"] = "tamper_refused"
    elif engine.is_symlink() or (engine.exists() and not engine.is_file()):
        rollback["engine"] = "tamper_refused"
    elif bool(prior_engine.get("exists")) and current_engine_hash == prior_engine_hash:
        rollback["engine"] = "already_prior"
    elif bool(prior_engine.get("exists")):
        _restore_file(
            transaction_path.parent / str(prior_engine["snapshot"]),
            engine,
            prior_engine,
        )
        rollback["engine"] = "restored_prior"
    elif not engine.exists():
        rollback["engine"] = "already_absent"
    else:
        engine.unlink()
        fsync_directory(engine.parent)
        rollback["engine"] = "removed_candidate_no_prior"

    realization = root / reconciler.REALIZATION_FILENAME
    prior_realization = _required_mapping(
        transaction.get("prior_realization"), "prior realization"
    )
    current_realization_hash = (
        _sha256(realization)
        if realization.is_file() and not realization.is_symlink()
        else None
    )
    prior_hash = (
        str(prior_realization.get("sha256"))
        if bool(prior_realization.get("exists"))
        else None
    )
    if realization.is_symlink() or (realization.exists() and not realization.is_file()):
        rollback["realization"] = "tamper_refused"
    elif current_realization_hash == prior_hash:
        rollback["realization"] = "already_prior"
    elif prior_hash:
        _restore_file(
            transaction_path.parent / str(prior_realization["snapshot"]),
            realization,
            prior_realization,
        )
        rollback["realization"] = "restored_prior"
    elif not realization.exists():
        rollback["realization"] = "already_absent"
    else:
        realization.unlink()
        fsync_directory(realization.parent)
        rollback["realization"] = "removed_new_no_prior"

    try:
        candidate_cleanup = _cleanup_hidden_candidates(
            transaction_path=transaction_path,
            transaction=transaction,
            prepared_transaction_sha256=recorded_prepared_sha or None,
            reason=reason,
        )
    except Exception as exc:
        candidate_cleanup = {
            "state": "manual_recovery_required",
            "blockers": [f"{type(exc).__name__}: {exc}"],
        }
    rollback["candidate_cleanup"] = candidate_cleanup

    if (
        not any(value == "tamper_refused" for value in rollback.values())
        and candidate_cleanup.get("state") == "complete"
    ):
        rollback["status"] = "rolled_back"
        rollback["completed_at_utc"] = _utc_now()
    transaction["state"] = (
        "rolled_back"
        if rollback["status"] == "rolled_back"
        else "manual_recovery_required"
    )
    transaction["rollback"] = rollback
    transaction["updated_at_utc"] = _utc_now()
    _write_json_atomic(transaction_path, transaction)
    if rollback["status"] == "rolled_back":
        rollback["retention_cleanup"] = _prune_completed_rollback_cohort(
            transaction_path,
            transaction,
        )
        transaction["rollback"] = rollback
        transaction["updated_at_utc"] = _utc_now()
        _write_json_atomic(transaction_path, transaction)
    return rollback


def commit(
    *,
    transaction_manifest: Path,
    expected_transaction_sha256: str,
    maintenance_manifest: Path | None,
    gpu_guard_evidence: Path,
    expected_gpu_guard_sha256: str,
    gpu_guard_container_id: str,
    gpu_guard_wrapper_pid: int,
    gpu_guard_wrapper_start_time_ticks: int,
) -> dict[str, Any]:
    transaction_path, transaction = _load_transaction(
        transaction_manifest, expected_transaction_sha256
    )
    base = _required_mapping(transaction["base_manifest"], "base manifest")
    contracts = _required_mapping(transaction["source_contracts"], "source contracts")
    root = Path(transaction["artifact_root"])
    accepted_realization: str | None = None
    functional_quality: dict[str, Any] | None = None
    try:
        maintenance_path, maintenance = _resolve_maintenance_manifest(
            transaction_path,
            transaction,
            maintenance_manifest,
            expected_transaction_sha256,
        )
        if transaction["engine"] == "mapanything":
            quality_authority = mapanything_quality_gate_from_source_contracts(
                Path(str(contracts["path"]))
            )
            quality_fixture = root / Path(
                str(
                    quality_authority["fixture"]["artifact_relative_path"]
                )
            )
            functional_quality = validate_mapanything_functional_quality_receipt(
                maintenance,
                authority=quality_authority,
                expected_engine_path=Path(str(transaction["engine_output"])),
                evidence_directory=maintenance_path.parent,
                fixture_path=quality_fixture,
            )
        guard_validation = _validate_gpu_guard_evidence(
            transaction_path=transaction_path,
            transaction=transaction,
            prepared_transaction_sha256=expected_transaction_sha256,
            maintenance=maintenance,
            evidence_path=gpu_guard_evidence,
            expected_evidence_sha256=expected_gpu_guard_sha256,
            container_id=gpu_guard_container_id,
            wrapper_pid=gpu_guard_wrapper_pid,
            wrapper_start_time_ticks=gpu_guard_wrapper_start_time_ticks,
        )
        _require_prior_agreement(transaction, maintenance)
        result = reconciler.reconcile(
            engine_name=str(transaction["engine"]),
            maintenance_manifest=maintenance_path,
            gpu_memory_guard=guard_validation,
            artifact_root=root,
            base_manifest_path=Path(str(base["path"])),
            expected_base_manifest_sha256=str(base["sha256"]),
            expected_source_contracts_sha256=str(contracts["sha256"]),
            expected_realization_sha256=str(
                _required_mapping(
                    transaction["prior_realization"], "prior realization"
                )["expected"]
            ),
            dry_run=False,
        )
        accepted_realization = str(result["realization_sha256"])
        validation = validator.validate_asset_realization(
            Path(str(base["path"])),
            Path(str(result["realization"])),
            root,
            profile=str(transaction["validation_profile"]),
            check_files=True,
            require_provenance=True,
        )
        if not validation.get("ok"):
            raise FinalizeError(
                "authoritative realized validation failed: "
                + "; ".join(
                    str(value)
                    for value in validation.get("errors", [])
                    + validation.get("blockers", [])
                )
            )
        realized = validator._load_private_json(
            Path(str(result["realization"])), "asset realization"
        )
        if transaction["artifact_id"] not in _required_mapping(
            realized.get("artifacts"), "realized artifacts"
        ):
            raise FinalizeError("committed realization lacks the selected engine")
    except BaseException as exc:
        rollback = _rollback(
            transaction_path=transaction_path,
            transaction=transaction,
            maintenance_manifest=maintenance_manifest,
            accepted_realization_sha256=accepted_realization,
            reason=f"{type(exc).__name__}: {exc}",
            prepared_transaction_sha256=expected_transaction_sha256,
        )
        raise FinalizeError(
            f"engine realization commit failed; rollback={rollback['status']}: {exc}"
        ) from exc

    transaction["state"] = "committed"
    transaction["prepared_transaction_sha256"] = expected_transaction_sha256
    transaction["commit"] = {
        "completed_at_utc": _utc_now(),
        "maintenance_manifest": str(maintenance_path),
        "gpu_memory_guard": guard_validation,
        "engine_output_sha256": _sha256(Path(transaction["engine_output"])),
        "engine_output_size_bytes": Path(transaction["engine_output"]).stat().st_size,
        "realization_sha256": accepted_realization,
        "validation_profile": transaction["validation_profile"],
        "realized_artifact_count": validation.get("realized_artifact_count"),
        **(
            {"functional_quality": functional_quality}
            if functional_quality is not None
            else {}
        ),
    }
    transaction["updated_at_utc"] = _utc_now()
    _write_json_atomic(transaction_path, transaction)
    transaction["cleanup"] = _prune_committed_engine_snapshots(
        transaction_path, transaction, maintenance
    )
    transaction["updated_at_utc"] = _utc_now()
    _write_json_atomic(transaction_path, transaction)
    return {
        "status": "committed",
        "transaction_manifest": str(transaction_path),
        "transaction_sha256": _sha256(transaction_path),
        "realization_sha256": accepted_realization,
        "artifact_id": transaction["artifact_id"],
    }


def rollback(
    *,
    transaction_manifest: Path,
    expected_transaction_sha256: str,
    maintenance_manifest: Path | None,
    reason: str,
) -> dict[str, Any]:
    transaction_path, transaction = _load_transaction(
        transaction_manifest,
        expected_transaction_sha256,
        allow_rolled_back=True,
    )
    if transaction.get("state") == "rolled_back":
        return {
            "status": "rolled_back",
            "idempotent": True,
            "transaction_manifest": str(transaction_path),
            "transaction_sha256": _sha256(transaction_path),
            "rollback": transaction.get("rollback"),
        }
    result = _rollback(
        transaction_path=transaction_path,
        transaction=transaction,
        maintenance_manifest=maintenance_manifest,
        accepted_realization_sha256=None,
        reason=reason,
        prepared_transaction_sha256=expected_transaction_sha256,
    )
    if result["status"] != "rolled_back":
        raise FinalizeError(f"automatic host rollback refused: {result}")
    return {
        "status": "rolled_back",
        "transaction_manifest": str(transaction_path),
        "transaction_sha256": _sha256(transaction_path),
        "rollback": result,
    }


def revert_committed(
    *,
    transaction_manifest: Path,
    expected_transaction_sha256: str,
    reason: str,
) -> dict[str, Any]:
    """CAS-revert the newest committed engine after a failed live canary."""

    transaction_path, transaction = _load_transaction(
        transaction_manifest,
        expected_transaction_sha256,
        required_state="committed",
    )
    commit_record = _required_mapping(transaction.get("commit"), "commit record")
    if (
        isinstance(transaction.get("retention"), Mapping)
        and transaction["retention"].get("state") == "cohort_pruned"
    ):
        raise FinalizeError("committed rollback cohort was superseded and pruned")
    engine = Path(str(transaction["engine_output"]))
    realization = (
        Path(str(transaction["artifact_root"])) / reconciler.REALIZATION_FILENAME
    )
    if (
        engine.is_symlink()
        or not engine.is_file()
        or engine.stat().st_size != commit_record.get("engine_output_size_bytes")
        or _sha256(engine) != commit_record.get("engine_output_sha256")
    ):
        raise FinalizeError("committed engine changed; refusing canary revert")
    if (
        realization.is_symlink()
        or not realization.is_file()
        or _sha256(realization) != commit_record.get("realization_sha256")
    ):
        raise FinalizeError("committed realization changed; refusing canary revert")
    transaction["state"] = "revert_in_progress"
    transaction["revert"] = {
        "requested_at_utc": _utc_now(),
        "reason": reason,
        "expected_engine_sha256": commit_record["engine_output_sha256"],
        "expected_realization_sha256": commit_record["realization_sha256"],
    }
    transaction["updated_at_utc"] = _utc_now()
    _write_json_atomic(transaction_path, transaction)
    result = _rollback(
        transaction_path=transaction_path,
        transaction=transaction,
        maintenance_manifest=None,
        accepted_realization_sha256=str(commit_record["realization_sha256"]),
        reason=f"committed canary revert: {reason}",
        prepared_transaction_sha256=str(
            transaction.get("prepared_transaction_sha256") or ""
        )
        or None,
    )
    if result["status"] != "rolled_back":
        raise FinalizeError(f"committed canary revert requires recovery: {result}")
    transaction["state"] = "reverted"
    transaction["revert"]["completed_at_utc"] = _utc_now()
    transaction["revert"]["rollback"] = result
    transaction["updated_at_utc"] = _utc_now()
    _write_json_atomic(transaction_path, transaction)
    return {
        "status": "reverted",
        "transaction_manifest": str(transaction_path),
        "transaction_sha256": _sha256(transaction_path),
        "rollback": result,
    }


def recover_transactions(
    *, artifact_root: Path, apply: bool, reason: str
) -> dict[str, Any]:
    """Audit or recover interrupted host transactions before new GPU work."""

    root = _artifact_root(artifact_root)
    transaction_root = root / TRANSACTION_ROOT_RELATIVE
    if not transaction_root.exists():
        return {"ok": True, "mode": "apply" if apply else "audit", "transactions": []}
    info = transaction_root.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise FinalizeError("engine finalize transaction root is not private and safe")
    manifests = sorted(transaction_root.glob("*/transaction.json"))
    if len(manifests) > 4096:
        raise FinalizeError("engine finalize recovery inventory exceeds safety bound")
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for raw in manifests:
        try:
            path = validator._bounded_private_path(
                raw, transaction_root, "engine finalize transaction"
            )
            transaction = dict(
                validator._load_private_json(path, "engine finalize transaction")
            )
            if transaction.get("contract") != CONTRACT:
                raise FinalizeError(f"unexpected transaction contract: {path}")
            state = str(transaction.get("state") or "")
            # Recovery may mutate only active transactions.  Those must still
            # agree with the current tracked authorities before rollback.  A
            # sealed historical transaction is evidence, not current build
            # authority; later reviewed manifest/contract changes must not make
            # old committed or completed records permanent startup blockers.
            if state in {"prepared", "revert_in_progress"}:
                _validate_transaction_scope(transaction, root, path)
            row: dict[str, Any] = {
                "transaction_manifest": str(path),
                "engine": transaction.get("engine"),
                "state_before": state,
                "action": "none",
            }
            if state == "prepared":
                if apply:
                    result = _rollback(
                        transaction_path=path,
                        transaction=transaction,
                        maintenance_manifest=None,
                        accepted_realization_sha256=None,
                        reason=f"startup recovery: {reason}",
                        prepared_transaction_sha256=str(
                            transaction.get("prepared_transaction_sha256") or ""
                        )
                        or _sha256(path),
                    )
                    row["action"] = "rolled_back"
                    row["result"] = result
                    if result["status"] != "rolled_back":
                        blockers.append(f"{path}: {result['status']}")
                        cleanup_result = result.get("candidate_cleanup")
                        if isinstance(cleanup_result, Mapping):
                            blockers.extend(
                                f"{path}: candidate cleanup: {detail}"
                                for detail in cleanup_result.get("blockers", [])
                            )
                else:
                    row["action"] = "recovery_required"
                    blockers.append(f"{path}: prepared transaction requires recovery")
            elif state == "revert_in_progress":
                if apply:
                    result = _rollback(
                        transaction_path=path,
                        transaction=transaction,
                        maintenance_manifest=None,
                        accepted_realization_sha256=None,
                        reason=f"startup completion of canary revert: {reason}",
                        prepared_transaction_sha256=str(
                            transaction.get("prepared_transaction_sha256") or ""
                        )
                        or None,
                    )
                    if result["status"] == "rolled_back":
                        transaction["state"] = "reverted"
                        transaction.setdefault("revert", {})["recovered_at_utc"] = (
                            _utc_now()
                        )
                        transaction["updated_at_utc"] = _utc_now()
                        _write_json_atomic(path, transaction)
                    row["action"] = "revert_completed"
                    row["result"] = result
                    if result["status"] != "rolled_back":
                        blockers.append(f"{path}: {result['status']}")
                else:
                    row["action"] = "recovery_required"
                    blockers.append(
                        f"{path}: interrupted canary revert requires recovery"
                    )
            elif state == "manual_recovery_required":
                row["action"] = "manual_recovery_required"
                blockers.append(f"{path}: manual recovery required")
            elif state == "rolled_back":
                cleanup = transaction.get("candidate_cleanup")
                if isinstance(cleanup, Mapping) and cleanup.get("state") == "complete":
                    cleanup = _cleanup_hidden_candidates(
                        transaction_path=path,
                        transaction=transaction,
                        prepared_transaction_sha256=str(
                            transaction.get("prepared_transaction_sha256") or ""
                        )
                        or None,
                        reason=f"sealed cleanup audit: {reason}",
                    )
                    row["action"] = "candidate_cleanup_complete_verified"
                    row["candidate_cleanup"] = cleanup
                elif (
                    isinstance(cleanup, Mapping)
                    and cleanup.get("state") == "manual_recovery_required"
                ):
                    row["action"] = "manual_recovery_required"
                    row["candidate_cleanup"] = cleanup
                    blockers.append(
                        f"{path}: candidate cleanup requires manual recovery"
                    )
                elif (
                    isinstance(cleanup, Mapping) and cleanup.get("state") == "prepared"
                ):
                    if apply:
                        cleanup = _cleanup_hidden_candidates(
                            transaction_path=path,
                            transaction=transaction,
                            prepared_transaction_sha256=str(
                                transaction.get("prepared_transaction_sha256") or ""
                            )
                            or None,
                            reason=f"startup recovery audit: {reason}",
                        )
                        row["action"] = "candidate_cleanup_resumed"
                        row["candidate_cleanup"] = cleanup
                        if cleanup.get("state") != "complete":
                            blockers.append(
                                f"{path}: candidate cleanup requires manual recovery"
                            )
                            blockers.extend(
                                f"{path}: candidate cleanup: {detail}"
                                for detail in cleanup.get("blockers", [])
                            )
                    else:
                        row["action"] = "candidate_cleanup_resume_required"
                        blockers.append(
                            f"{path}: prepared candidate cleanup requires recovery"
                        )
                elif "hidden_candidate_inventory" not in transaction:
                    engine = Path(str(transaction.get("engine_output") or ""))
                    residue = _hidden_candidate_inventory(engine)
                    if residue:
                        row["action"] = "explicit_residue_cleanup_required"
                        blockers.append(
                            f"{path}: pre-inventory rolled-back transaction has hidden residue; "
                            "use cleanup-residue with exact CAS authority"
                        )
                    else:
                        row["action"] = "no_hidden_candidate_residue"
                elif apply:
                    cleanup = _cleanup_hidden_candidates(
                        transaction_path=path,
                        transaction=transaction,
                        prepared_transaction_sha256=str(
                            transaction.get("prepared_transaction_sha256") or ""
                        )
                        or None,
                        reason=f"startup recovery audit: {reason}",
                    )
                    row["action"] = "candidate_cleanup_recovered"
                    row["candidate_cleanup"] = cleanup
                    if cleanup.get("state") != "complete":
                        blockers.append(
                            f"{path}: candidate cleanup requires manual recovery"
                        )
                        blockers.extend(
                            f"{path}: candidate cleanup: {detail}"
                            for detail in cleanup.get("blockers", [])
                        )
                else:
                    row["action"] = "candidate_cleanup_audit_required"
                    blockers.append(
                        f"{path}: candidate cleanup requires recovery audit"
                    )
            elif state not in {"committed", "rolled_back", "reverted"}:
                row["action"] = "unknown_state"
                blockers.append(f"{path}: unknown transaction state {state!r}")
            rows.append(row)
        except Exception as exc:
            blockers.append(f"{raw}: {type(exc).__name__}: {exc}")
    fsync_directory(transaction_root)
    return {
        "ok": not blockers,
        "mode": "apply" if apply else "audit",
        "transactions": rows,
        "blockers": blockers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    prepare = subparsers.add_parser("snapshot")
    prepare.add_argument(
        "--engine", choices=sorted(reconciler.ENGINE_ARTIFACT_IDS), required=True
    )
    prepare.add_argument("--artifact-root", type=Path, required=True)
    prepare.add_argument(
        "--base-manifest",
        type=Path,
        default=REPO_ROOT / "DS9/asset_manifest.yaml",
    )
    prepare.add_argument("--validation-profile", default="engine_finalize")
    prepare.add_argument("--lock-fd", type=int, required=True)
    for name in ("commit", "rollback"):
        command = subparsers.add_parser(name)
        command.add_argument("--transaction-manifest", type=Path, required=True)
        command.add_argument("--expected-transaction-sha256", required=True)
        command.add_argument("--maintenance-manifest", type=Path)
        command.add_argument("--lock-fd", type=int, required=True)
        if name == "rollback":
            command.add_argument("--reason", required=True)
        else:
            command.add_argument("--gpu-guard-evidence", type=Path, required=True)
            command.add_argument("--expected-gpu-guard-sha256", required=True)
            command.add_argument("--gpu-guard-container-id", required=True)
            command.add_argument("--gpu-guard-wrapper-pid", type=int, required=True)
            command.add_argument(
                "--gpu-guard-wrapper-start-time-ticks", type=int, required=True
            )
    revert = subparsers.add_parser("revert-committed")
    revert.add_argument("--transaction-manifest", type=Path, required=True)
    revert.add_argument("--expected-transaction-sha256", required=True)
    revert.add_argument("--reason", required=True)
    revert.add_argument("--lock-fd", type=int, required=True)
    recover = subparsers.add_parser("recover")
    recover.add_argument("--artifact-root", type=Path, required=True)
    recover.add_argument("--apply", action="store_true")
    recover.add_argument("--reason", default="pre-maintenance recovery audit")
    recover.add_argument("--lock-fd", type=int, required=True)
    residue = subparsers.add_parser("cleanup-residue")
    residue.add_argument("--transaction-manifest", type=Path, required=True)
    residue.add_argument("--expected-transaction-sha256", required=True)
    residue.add_argument("--maintenance-manifest", type=Path, required=True)
    residue.add_argument("--expected-maintenance-sha256", required=True)
    residue.add_argument("--expected-candidate-sha256", required=True)
    residue.add_argument("--expected-candidate-size-bytes", type=int, required=True)
    residue.add_argument("--reason", required=True)
    residue.add_argument("--lock-fd", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.operation in {"snapshot", "recover"}:
            lock_root = _artifact_root(args.artifact_root)
        else:
            transaction_preview = validator._load_private_json(
                args.transaction_manifest, "engine finalize transaction"
            )
            lock_root = _artifact_root(
                Path(str(transaction_preview.get("artifact_root") or ""))
            )
        _require_inherited_transaction_lock(lock_root, int(args.lock_fd))
        if args.operation == "snapshot":
            result = snapshot(
                engine_name=args.engine,
                artifact_root=args.artifact_root,
                base_manifest_path=args.base_manifest,
                validation_profile=args.validation_profile,
            )
        elif args.operation == "commit":
            result = commit(
                transaction_manifest=args.transaction_manifest,
                expected_transaction_sha256=args.expected_transaction_sha256,
                maintenance_manifest=args.maintenance_manifest,
                gpu_guard_evidence=args.gpu_guard_evidence,
                expected_gpu_guard_sha256=args.expected_gpu_guard_sha256,
                gpu_guard_container_id=args.gpu_guard_container_id,
                gpu_guard_wrapper_pid=args.gpu_guard_wrapper_pid,
                gpu_guard_wrapper_start_time_ticks=(
                    args.gpu_guard_wrapper_start_time_ticks
                ),
            )
        elif args.operation == "rollback":
            result = rollback(
                transaction_manifest=args.transaction_manifest,
                expected_transaction_sha256=args.expected_transaction_sha256,
                maintenance_manifest=args.maintenance_manifest,
                reason=args.reason,
            )
        elif args.operation == "revert-committed":
            result = revert_committed(
                transaction_manifest=args.transaction_manifest,
                expected_transaction_sha256=args.expected_transaction_sha256,
                reason=args.reason,
            )
        elif args.operation == "cleanup-residue":
            result = cleanup_residue(
                transaction_manifest=args.transaction_manifest,
                expected_transaction_sha256=args.expected_transaction_sha256,
                maintenance_manifest=args.maintenance_manifest,
                expected_maintenance_sha256=args.expected_maintenance_sha256,
                expected_candidate_sha256=args.expected_candidate_sha256,
                expected_candidate_size_bytes=args.expected_candidate_size_bytes,
                reason=args.reason,
            )
        else:
            result = recover_transactions(
                artifact_root=args.artifact_root,
                apply=bool(args.apply),
                reason=args.reason,
            )
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("ok", True) else 3


if __name__ == "__main__":
    raise SystemExit(main())
