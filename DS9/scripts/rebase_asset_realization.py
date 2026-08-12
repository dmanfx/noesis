#!/usr/bin/env python3
"""Rebase a DS9 engine realization across a non-engine manifest change.

This tool is intentionally narrower than a general manifest migration.  It
requires byte-level hash authority for both manifests and the current
realization, then proves that target/build authority, source-contract authority,
and every TensorRT engine record are unchanged.  The sole permitted runtime
change is adoption of the exact reviewed runtime-image authority.  Apply mode
is an explicit compare-and-swap transaction protected by the canonical DS9
artifact lock.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis_core.strict_json import (  # noqa: E402
    StrictJSONError,
    strict_json_loads,
)

NEW_MANIFEST_AUTHORITY = REPO_ROOT / "DS9/asset_manifest.yaml"
SOURCE_CONTRACTS_AUTHORITY = REPO_ROOT / "DS9/config/engine_source_contracts.json"
REALIZATION_FILENAME = "asset_realization.json"
LOCK_FILENAME = ".noesis-ds9-artifact-transaction.lock"
EVIDENCE_ROOT_RELATIVE = Path("manifest_rebase")
REALIZATION_CONTRACT = "noesis.ds9.asset_realization"
REBASE_CONTRACT = "noesis.ds9.asset_realization_rebase"
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_DIFF_PATHS = 4096
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
_REALIZATION_KEYS = {
    "schema_version",
    "contract",
    "base_manifest",
    "source_contracts",
    "created_at_utc",
    "updated_at_utc",
    "artifacts",
}
REVIEWED_RUNTIME_IMAGE_AUTHORITY = {
    "reference": "noesis-ds9-runtime:9.1-20260812",
    "image_id": "sha256:b97a32b082e74265c15e767bcaafa4dc1d8947e53feb36adb9baafdf69ba762e",
    "parent_reference": "noesis-ds9-dev:9.1-20260812",
    "parent_image_id": "sha256:88d80ad35f12ec3a574cf2555a8242d33ac4110abdcc5f88a6cbdee40dfcf872",
    "base_digest": "sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994",
    "tensorrt_version": "10.16.0.72",
    "cuda_version": "13.2.0.046",
    "dockerfile": "DS9/docker/Dockerfile.runtime",
    "dockerfile_sha256": "4061b2dd98298aa07f0438ccf8ea9e1ae6e238621ba4b5ed363d4ce87d6cc133",
}


class RebaseError(RuntimeError):
    """Raised when a realization rebase cannot be proven or committed safely."""


class _AtomicTransitionError(RebaseError):
    """An atomic replacement failed with a securely classified file outcome."""

    def __init__(self, message: str, *, observed_state: str) -> None:
        super().__init__(message)
        self.observed_state = observed_state


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader which rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    explicit_keys: set[Any] = set()
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in explicit_keys
        except TypeError as exc:
            raise RebaseError("YAML mapping key is not hashable") from exc
        if duplicate:
            raise RebaseError("YAML document contains a duplicate mapping key")
        explicit_keys.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: str, label: str) -> str:
    result = str(value or "").strip()
    if SHA256_RE.fullmatch(result) is None:
        raise RebaseError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _reject_symlink_components(path: Path, label: str) -> None:
    candidate = _lexical_absolute(path)
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(info.st_mode):
            raise RebaseError(f"{label} contains a symlink")


def _require_real_owned_directory(path: Path, label: str) -> Path:
    candidate = _lexical_absolute(path)
    _reject_symlink_components(candidate, label)
    try:
        info = candidate.lstat()
    except FileNotFoundError as exc:
        raise RebaseError(f"{label} is missing: {candidate}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RebaseError(f"{label} must be a real directory: {candidate}")
    if info.st_uid != os.geteuid():
        raise RebaseError(f"{label} must be owned by the current user")
    if stat.S_IMODE(info.st_mode) & 0o022:
        raise RebaseError(f"{label} must not be group/world writable")
    return candidate


def _bounded_path(path: str | Path, root: Path, label: str) -> Path:
    candidate = _lexical_absolute(path)
    root = _require_real_owned_directory(root, f"{label} root")
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise RebaseError(f"{label} escapes its authority root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise RebaseError(f"{label} contains a symlink")
    return candidate


def _read_file(
    path: Path,
    label: str,
    *,
    require_private: bool,
) -> bytes:
    candidate = _lexical_absolute(path)
    _reject_symlink_components(candidate, label)
    try:
        lexical = candidate.lstat()
    except FileNotFoundError as exc:
        raise RebaseError(f"{label} is missing") from exc
    if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISREG(lexical.st_mode):
        raise RebaseError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise RebaseError(f"{label} cannot be opened without following links") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino) != (lexical.st_dev, lexical.st_ino)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
        ):
            raise RebaseError(
                f"{label} must be an owned, single-link regular file"
            )
        if require_private and stat.S_IMODE(before.st_mode) != 0o600:
            raise RebaseError(f"{label} must have mode 0600")
        if before.st_size <= 0 or before.st_size > MAX_INPUT_BYTES:
            raise RebaseError(f"{label} size is outside the accepted bound")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(raw) != before.st_size
            or (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            != (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
        ):
            raise RebaseError(f"{label} changed while it was read")
        return raw
    finally:
        os.close(descriptor)


def _parse_json_mapping(raw: bytes, label: str) -> dict[str, Any]:
    try:
        payload = strict_json_loads(raw, label=label)
    except StrictJSONError as exc:
        if exc.reason == "duplicate_key":
            message = f"{label} contains a duplicate mapping key"
        elif exc.reason == "nonfinite_number":
            message = f"{label} contains a non-finite number"
        else:
            message = f"{label} is not valid UTF-8 JSON"
        raise RebaseError(message) from exc
    if not isinstance(payload, dict):
        raise RebaseError(f"{label} must be a JSON mapping")
    return payload


def _parse_yaml_mapping(raw: bytes, label: str) -> dict[str, Any]:
    try:
        payload = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RebaseError(f"{label} is not valid UTF-8 YAML") from exc
    if not isinstance(payload, dict):
        raise RebaseError(f"{label} must be a YAML mapping")
    return payload


def _parse_utc(value: object, label: str) -> datetime:
    text = str(value or "")
    if UTC_RE.fullmatch(text) is None:
        raise RebaseError(f"{label} must be an explicit RFC3339 UTC timestamp")
    try:
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RebaseError(f"{label} is not a valid UTC timestamp") from exc
    if result.utcoffset() != timezone.utc.utcoffset(result):
        raise RebaseError(f"{label} must use UTC")
    return result


def _manifest_artifacts(
    payload: Mapping[str, Any], label: str
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise RebaseError(f"{label}.artifacts must be a nonempty list")
    all_rows: dict[str, Mapping[str, Any]] = {}
    engines: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RebaseError(f"{label}.artifacts contains a non-mapping record")
        artifact_id = str(row.get("id") or "").strip()
        if not artifact_id or artifact_id in all_rows:
            raise RebaseError(f"{label}.artifacts has a missing or duplicate ID")
        all_rows[artifact_id] = row
        if row.get("kind") == "tensorrt_engine":
            engines[artifact_id] = row
    if not engines:
        raise RebaseError(f"{label} declares no TensorRT engine artifacts")
    return all_rows, engines


def _append_diff(
    paths: list[str], path: Sequence[str], left: object, right: object
) -> None:
    if len(paths) >= MAX_DIFF_PATHS:
        raise RebaseError("manifest semantic diff exceeds the accepted bound")
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                paths.append(".".join((*path, str(key))))
            else:
                _append_diff(paths, (*path, str(key)), left[key], right[key])
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            paths.append(".".join((*path, "length")))
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _append_diff(
                paths, (*path, str(index)), left_item, right_item
            )
        return
    if left != right:
        paths.append(".".join(path))


def _manifest_diff_summary(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    old_rows: Mapping[str, Mapping[str, Any]],
    new_rows: Mapping[str, Mapping[str, Any]],
    engine_ids: set[str],
) -> dict[str, Any]:
    paths: list[str] = []
    for key in sorted((set(old) | set(new)) - {"artifacts"}, key=str):
        if key not in old or key not in new:
            paths.append(str(key))
        else:
            _append_diff(paths, (str(key),), old[key], new[key])
    for artifact_id in sorted(set(old_rows) | set(new_rows)):
        if artifact_id not in old_rows or artifact_id not in new_rows:
            paths.append(f"artifacts.{artifact_id}")
        else:
            _append_diff(
                paths,
                ("artifacts", artifact_id),
                old_rows[artifact_id],
                new_rows[artifact_id],
            )
    changed_ids = sorted(
        artifact_id
        for artifact_id in set(old_rows) | set(new_rows)
        if old_rows.get(artifact_id) != new_rows.get(artifact_id)
    )
    return {
        "changed_path_count": len(paths),
        "changed_paths": paths,
        "changed_artifact_ids": changed_ids,
        "changed_non_engine_artifact_ids": [
            artifact_id for artifact_id in changed_ids if artifact_id not in engine_ids
        ],
    }


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _semantic_diff_paths(left: object, right: object) -> list[str]:
    result: list[str] = []
    _append_diff(result, (), left, right)
    return result


def _runtime_image_transition(
    old_manifest: Mapping[str, Any], new_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove that runtime drift is limited to the reviewed image authority."""

    old_runtime = old_manifest.get("runtime")
    new_runtime = new_manifest.get("runtime")
    if not isinstance(old_runtime, Mapping) or not isinstance(new_runtime, Mapping):
        raise RebaseError("runtime authority must be a mapping in both manifests")
    old_runtime_without_image = dict(old_runtime)
    new_runtime_without_image = dict(new_runtime)
    old_image = old_runtime_without_image.pop("image", None)
    new_image = new_runtime_without_image.pop("image", None)
    if old_runtime_without_image != new_runtime_without_image:
        raise RebaseError("unsupported runtime authority drift outside runtime.image")
    changed = old_image != new_image
    if (new_image is not None or changed) and (
        not isinstance(new_image, Mapping)
        or dict(new_image) != REVIEWED_RUNTIME_IMAGE_AUTHORITY
    ):
        raise RebaseError(
            "runtime.image drift does not adopt the exact reviewed runtime-image authority"
        )
    if old_image is not None and not isinstance(old_image, Mapping):
        raise RebaseError("old runtime.image authority must be a mapping when present")
    return {
        "changed": changed,
        "before": copy.deepcopy(dict(old_image))
        if isinstance(old_image, Mapping)
        else None,
        "after": copy.deepcopy(dict(new_image))
        if isinstance(new_image, Mapping)
        else None,
    }


@contextmanager
def _artifact_transaction_lock(root: Path) -> Iterator[None]:
    lock_path = _bounded_path(root / LOCK_FILENAME, root, "artifact transaction lock")
    try:
        lexical = lock_path.lstat()
    except FileNotFoundError as exc:
        raise RebaseError("canonical artifact transaction lock is missing") from exc
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags)
    except OSError as exc:
        raise RebaseError("canonical artifact transaction lock cannot be opened") from exc
    try:
        info = os.fstat(descriptor)
        if (
            stat.S_ISLNK(lexical.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or (info.st_dev, info.st_ino) != (lexical.st_dev, lexical.st_ino)
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise RebaseError(
                "artifact transaction lock must be an owned single-link mode-0600 file"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RebaseError("another DS9 artifact transaction owns the lock") from exc
        yield
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_private_replace(
    path: Path,
    raw: bytes,
    *,
    expected_current_sha256: str | None,
    label: str,
) -> str:
    parent = _require_real_owned_directory(path.parent, f"{label} parent")
    candidate = _bounded_path(path, parent, label)
    if expected_current_sha256 is not None:
        current = _read_file(candidate, label, require_private=True)
        if _sha256_bytes(current) != expected_current_sha256:
            raise RebaseError(f"{label} changed before atomic replacement")
    temporary = parent / f".{candidate.name}.writing-{os.getpid()}-{_run_id()}"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if expected_current_sha256 is not None:
            current = _read_file(candidate, label, require_private=True)
            if _sha256_bytes(current) != expected_current_sha256:
                raise RebaseError(f"{label} changed during atomic replacement")
        os.replace(temporary, candidate)
        _fsync_directory(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    committed = _read_file(candidate, label, require_private=True)
    committed_hash = _sha256_bytes(committed)
    expected_hash = _sha256_bytes(raw)
    if committed_hash != expected_hash:
        raise RebaseError(f"{label} failed post-write verification")
    return committed_hash


def _classify_exact_file_state(
    path: Path,
    *,
    before_raw: bytes,
    after_raw: bytes,
    label: str,
) -> str:
    """Classify a securely read file as the exact before, after, or unknown state."""

    try:
        observed_raw = _read_file(path, label, require_private=True)
    except Exception:
        return "ambiguous"
    observed_hash = _sha256_bytes(observed_raw)
    if observed_raw == before_raw and observed_hash == _sha256_bytes(before_raw):
        return "before"
    if observed_raw == after_raw and observed_hash == _sha256_bytes(after_raw):
        return "after"
    return "ambiguous"


def _atomic_private_transition(
    path: Path,
    after_raw: bytes,
    *,
    before_raw: bytes,
    label: str,
) -> tuple[str, bool]:
    """Replace an exact private file and recover a proven post-rename success.

    ``_atomic_private_replace`` can report an error after ``os.replace`` has
    installed the new inode.  On that path this helper securely rereads the
    file, repeats the parent-directory durability barrier, and rereads it
    again before accepting the transition.  Every other outcome is exposed as
    either the exact pre-transition state or an ambiguous state which requires
    operator recovery.
    """

    before_hash = _sha256_bytes(before_raw)
    after_hash = _sha256_bytes(after_raw)
    try:
        committed_hash = _atomic_private_replace(
            path,
            after_raw,
            expected_current_sha256=before_hash,
            label=label,
        )
    except Exception as replace_error:
        observed_state = _classify_exact_file_state(
            path,
            before_raw=before_raw,
            after_raw=after_raw,
            label=label,
        )
        if observed_state == "after":
            try:
                parent = _require_real_owned_directory(
                    path.parent, f"{label} recovery parent"
                )
                _fsync_directory(parent)
            except Exception as durability_error:
                raise _AtomicTransitionError(
                    f"{label} post-replace durability recovery failed",
                    observed_state="ambiguous",
                ) from durability_error
            confirmed_state = _classify_exact_file_state(
                path,
                before_raw=before_raw,
                after_raw=after_raw,
                label=label,
            )
            if confirmed_state == "after":
                return after_hash, True
            raise _AtomicTransitionError(
                f"{label} changed during post-replace recovery",
                observed_state="ambiguous",
            ) from replace_error
        if observed_state == "before":
            raise _AtomicTransitionError(
                f"{label} replacement failed before commit",
                observed_state="before",
            ) from replace_error
        raise _AtomicTransitionError(
            f"{label} replacement outcome is ambiguous",
            observed_state="ambiguous",
        ) from replace_error
    if committed_hash != after_hash:
        raise _AtomicTransitionError(
            f"{label} returned an unexpected committed hash",
            observed_state="ambiguous",
        )
    return committed_hash, False


def _create_evidence_directory(root: Path, transaction_id: str) -> Path:
    evidence_root = root / EVIDENCE_ROOT_RELATIVE
    if not evidence_root.exists() and not evidence_root.is_symlink():
        evidence_root.mkdir(mode=0o700)
        _fsync_directory(root)
    evidence_root = _require_real_owned_directory(
        evidence_root, "manifest-rebase evidence root"
    )
    run_dir = evidence_root / transaction_id
    try:
        run_dir.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise RebaseError("manifest-rebase evidence transaction already exists") from exc
    os.chmod(run_dir, 0o700)
    _fsync_directory(evidence_root)
    return run_dir


def _assert_no_unresolved_rebase(root: Path) -> None:
    evidence_root = root / EVIDENCE_ROOT_RELATIVE
    if not evidence_root.exists() and not evidence_root.is_symlink():
        return
    evidence_root = _require_real_owned_directory(
        evidence_root, "manifest-rebase evidence root"
    )
    with os.scandir(evidence_root) as inventory:
        entries = list(inventory)
    if len(entries) > 4096:
        raise RebaseError("manifest-rebase evidence inventory exceeds safety bound")
    terminal_states = {
        "committed",
        "aborted_before_commit",
        "rolled_back_after_evidence_failure",
    }
    for entry in entries:
        if entry.is_symlink():
            raise RebaseError("manifest-rebase evidence inventory contains a symlink")
        if not entry.is_dir(follow_symlinks=False):
            continue
        evidence_path = Path(entry.path) / "rebase_evidence.json"
        if not evidence_path.exists() and not evidence_path.is_symlink():
            continue
        payload = _parse_json_mapping(
            _read_file(
                evidence_path,
                "prior manifest-rebase evidence",
                require_private=True,
            ),
            "prior manifest-rebase evidence",
        )
        if payload.get("contract") != REBASE_CONTRACT:
            raise RebaseError("prior manifest-rebase evidence has an invalid contract")
        if payload.get("state") not in terminal_states:
            raise RebaseError(
                "unresolved prepared manifest-rebase evidence requires recovery"
            )


def _revalidate_inputs(
    *,
    old_manifest_path: Path,
    new_manifest_path: Path,
    source_contracts_path: Path,
    realization_path: Path,
    expected_old_manifest_sha256: str,
    expected_new_manifest_sha256: str,
    expected_source_contracts_sha256: str,
    expected_old_realization_sha256: str,
) -> None:
    checks = (
        (
            old_manifest_path,
            "old manifest snapshot",
            True,
            expected_old_manifest_sha256,
        ),
        (
            new_manifest_path,
            "new tracked manifest",
            False,
            expected_new_manifest_sha256,
        ),
        (
            source_contracts_path,
            "source-contract authority",
            False,
            expected_source_contracts_sha256,
        ),
        (
            realization_path,
            "asset realization",
            True,
            expected_old_realization_sha256,
        ),
    )
    for path, label, require_private, expected in checks:
        observed = _sha256_bytes(
            _read_file(path, label, require_private=require_private)
        )
        if observed != expected:
            raise RebaseError(f"{label} changed during rebase authorization")


def rebase_realization(
    *,
    artifact_root: Path,
    old_manifest_snapshot: Path,
    expected_old_manifest_sha256: str,
    expected_new_manifest_sha256: str,
    expected_source_contracts_sha256: str,
    expected_old_realization_sha256: str,
    expected_new_realization_sha256: str | None,
    updated_at_utc: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Prove and optionally commit one narrow realization rebase."""

    expected_old_manifest_sha256 = _require_sha256(
        expected_old_manifest_sha256, "expected old manifest hash"
    )
    expected_new_manifest_sha256 = _require_sha256(
        expected_new_manifest_sha256, "expected new manifest hash"
    )
    expected_source_contracts_sha256 = _require_sha256(
        expected_source_contracts_sha256, "expected source-contract hash"
    )
    expected_old_realization_sha256 = _require_sha256(
        expected_old_realization_sha256, "expected old realization hash"
    )
    if expected_old_manifest_sha256 == expected_new_manifest_sha256:
        raise RebaseError("old and new manifest hashes must differ")
    if expected_new_realization_sha256 is not None:
        expected_new_realization_sha256 = _require_sha256(
            expected_new_realization_sha256, "expected new realization hash"
        )
    if not dry_run and expected_new_realization_sha256 is None:
        raise RebaseError("apply requires an explicit expected new realization hash")

    root = _require_real_owned_directory(artifact_root, "artifact root")
    old_root = root / EVIDENCE_ROOT_RELATIVE
    old_manifest_path = _bounded_path(
        old_manifest_snapshot, old_root, "old manifest snapshot"
    )
    new_manifest_path = _lexical_absolute(NEW_MANIFEST_AUTHORITY)
    if new_manifest_path != _lexical_absolute(REPO_ROOT / "DS9/asset_manifest.yaml"):
        raise RebaseError("new manifest authority is not tracked DS9/asset_manifest.yaml")
    source_contracts_path = _lexical_absolute(SOURCE_CONTRACTS_AUTHORITY)
    if source_contracts_path != _lexical_absolute(
        REPO_ROOT / "DS9/config/engine_source_contracts.json"
    ):
        raise RebaseError("source-contract authority is not the tracked DS9 contract")
    realization_path = _bounded_path(
        root / REALIZATION_FILENAME, root, "asset realization"
    )

    with _artifact_transaction_lock(root):
        _assert_no_unresolved_rebase(root)
        old_raw = _read_file(
            old_manifest_path, "old manifest snapshot", require_private=True
        )
        new_raw = _read_file(
            new_manifest_path, "new tracked manifest", require_private=False
        )
        contracts_raw = _read_file(
            source_contracts_path,
            "source-contract authority",
            require_private=False,
        )
        realization_raw = _read_file(
            realization_path, "asset realization", require_private=True
        )
        observed = {
            "old_manifest": _sha256_bytes(old_raw),
            "new_manifest": _sha256_bytes(new_raw),
            "source_contracts": _sha256_bytes(contracts_raw),
            "old_realization": _sha256_bytes(realization_raw),
        }
        expected = {
            "old_manifest": expected_old_manifest_sha256,
            "new_manifest": expected_new_manifest_sha256,
            "source_contracts": expected_source_contracts_sha256,
            "old_realization": expected_old_realization_sha256,
        }
        for key in expected:
            if observed[key] != expected[key]:
                raise RebaseError(
                    f"{key.replace('_', ' ')} hash mismatch: "
                    f"expected={expected[key]} observed={observed[key]}"
                )

        old_manifest = _parse_yaml_mapping(old_raw, "old manifest snapshot")
        new_manifest = _parse_yaml_mapping(new_raw, "new tracked manifest")
        source_contracts = _parse_json_mapping(
            contracts_raw, "source-contract authority"
        )
        realization = _parse_json_mapping(realization_raw, "asset realization")
        if _canonical_json(realization) != realization_raw:
            raise RebaseError(
                "asset realization is not canonical JSON; refusing incidental byte drift"
            )
        if source_contracts.get("schema_version") != 1 or not isinstance(
            source_contracts.get("contracts"), Mapping
        ):
            raise RebaseError("source-contract authority has an invalid contract")
        if set(realization) != _REALIZATION_KEYS:
            raise RebaseError("asset realization has unexpected or missing keys")
        if (
            realization.get("schema_version") != 1
            or realization.get("contract") != REALIZATION_CONTRACT
        ):
            raise RebaseError("asset realization contract is invalid")
        if realization.get("base_manifest") != {
            "path": "DS9/asset_manifest.yaml",
            "sha256": expected_old_manifest_sha256,
        }:
            raise RebaseError("asset realization does not bind the old manifest")
        expected_contract_record = {
            "path": "DS9/config/engine_source_contracts.json",
            "sha256": expected_source_contracts_sha256,
        }
        if realization.get("source_contracts") != expected_contract_record:
            raise RebaseError("asset realization source-contract authority drifted")

        created_at = _parse_utc(realization.get("created_at_utc"), "created_at_utc")
        previous_updated_at = _parse_utc(
            realization.get("updated_at_utc"), "current updated_at_utc"
        )
        proposed_updated_at = _parse_utc(updated_at_utc, "updated_at_utc")
        if proposed_updated_at <= previous_updated_at or proposed_updated_at < created_at:
            raise RebaseError("updated_at_utc must advance the realization timestamp")

        if old_manifest.get("target") != new_manifest.get("target"):
            raise RebaseError("DS9 target authority drifted between manifests")
        runtime_image_transition = _runtime_image_transition(
            old_manifest, new_manifest
        )
        ignored_top_level = {"artifacts", "runtime", "updated_at"}
        old_top_level = set(old_manifest) - ignored_top_level
        new_top_level = set(new_manifest) - ignored_top_level
        if old_top_level != new_top_level:
            raise RebaseError("unsupported top-level manifest key drift")
        for key in sorted(old_top_level):
            if old_manifest[key] != new_manifest[key]:
                raise RebaseError(
                    f"unsupported top-level manifest drift outside artifacts: {key}"
                )

        old_rows, old_engines = _manifest_artifacts(old_manifest, "old manifest")
        new_rows, new_engines = _manifest_artifacts(new_manifest, "new manifest")
        if set(old_engines) != set(new_engines):
            raise RebaseError("TensorRT engine artifact membership drifted")
        for artifact_id in sorted(old_engines):
            if old_engines[artifact_id] != new_engines[artifact_id]:
                raise RebaseError(
                    f"TensorRT engine artifact record drifted: {artifact_id}"
                )

        realized_artifacts = realization.get("artifacts")
        if not isinstance(realized_artifacts, dict):
            raise RebaseError("asset realization artifacts must be a mapping")
        for artifact_id, record in realized_artifacts.items():
            if artifact_id not in old_engines or artifact_id not in new_engines:
                raise RebaseError(
                    f"realized artifact is not an unchanged TensorRT engine: {artifact_id}"
                )
            if not isinstance(record, Mapping) or set(record) != {
                "state",
                "provenance",
            }:
                raise RebaseError(
                    f"realized engine record has an invalid shape: {artifact_id}"
                )
            if record.get("state") not in {"staged_unverified", "validated"} or not isinstance(
                record.get("provenance"), Mapping
            ):
                raise RebaseError(
                    f"realized engine record has invalid state/provenance: {artifact_id}"
                )

        proposal = copy.deepcopy(realization)
        proposal["base_manifest"]["sha256"] = expected_new_manifest_sha256
        proposal["updated_at_utc"] = updated_at_utc
        semantic_changes = _semantic_diff_paths(realization, proposal)
        if semantic_changes != ["base_manifest.sha256", "updated_at_utc"]:
            raise RebaseError("proposed realization mutation escaped the narrow contract")
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

        diff_summary = _manifest_diff_summary(
            old_manifest,
            new_manifest,
            old_rows,
            new_rows,
            set(old_engines),
        )
        result: dict[str, Any] = {
            "ok": True,
            "dry_run": dry_run,
            "old_manifest_sha256": expected_old_manifest_sha256,
            "new_manifest_sha256": expected_new_manifest_sha256,
            "source_contracts_sha256": expected_source_contracts_sha256,
            "old_realization_sha256": expected_old_realization_sha256,
            "new_realization_sha256": proposal_hash,
            "realized_engine_count": len(realized_artifacts),
            "engine_artifact_count": len(old_engines),
            "manifest_diff": diff_summary,
            "runtime_image_authority": runtime_image_transition,
            "semantic_checks": {
                "target_unchanged": True,
                "source_contracts_unchanged": True,
                "all_engine_artifacts_unchanged": True,
                "realized_artifacts_unchanged": True,
                "runtime_fields_outside_image_unchanged": True,
                "runtime_image_change_is_exact_reviewed_authority": True,
            },
        }
        if dry_run:
            return result

        transaction_id = _run_id()
        run_dir = _create_evidence_directory(root, transaction_id)
        evidence_path = run_dir / "rebase_evidence.json"
        prepared_at = _utc_now()
        evidence: dict[str, Any] = {
            "schema_version": 1,
            "contract": REBASE_CONTRACT,
            "transaction_id": transaction_id,
            "state": "prepared",
            "prepared_at_utc": prepared_at,
            "committed_at_utc": None,
            "old_manifest": {
                "path": old_manifest_path.relative_to(root).as_posix(),
                "sha256": expected_old_manifest_sha256,
            },
            "new_manifest": {
                "path": "DS9/asset_manifest.yaml",
                "sha256": expected_new_manifest_sha256,
            },
            "source_contracts": expected_contract_record,
            "realization": {
                "path": REALIZATION_FILENAME,
                "old_sha256": expected_old_realization_sha256,
                "new_sha256": proposal_hash,
                "updated_at_utc_before": realization["updated_at_utc"],
                "updated_at_utc_after": updated_at_utc,
            },
            "mutation_paths": semantic_changes,
            "realized_engine_ids": sorted(realized_artifacts),
            "semantic_checks": copy.deepcopy(result["semantic_checks"]),
            "manifest_diff": copy.deepcopy(diff_summary),
            "runtime_image_authority": copy.deepcopy(
                runtime_image_transition
            ),
        }
        prepared_raw = _canonical_json(evidence)
        _atomic_private_replace(
            evidence_path,
            prepared_raw,
            expected_current_sha256=None,
            label="manifest-rebase evidence",
        )
        try:
            _revalidate_inputs(
                old_manifest_path=old_manifest_path,
                new_manifest_path=new_manifest_path,
                source_contracts_path=source_contracts_path,
                realization_path=realization_path,
                expected_old_manifest_sha256=expected_old_manifest_sha256,
                expected_new_manifest_sha256=expected_new_manifest_sha256,
                expected_source_contracts_sha256=expected_source_contracts_sha256,
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
                    label="manifest-rebase evidence",
                )
            except _AtomicTransitionError as evidence_error:
                raise RebaseError(
                    "rebase authorization failed before realization commit, but "
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
                        label="manifest-rebase evidence",
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
                    label="manifest-rebase evidence",
                )
            except _AtomicTransitionError:
                # The original prepared record is itself nonterminal.  Do not
                # relabel either file when its exact state cannot be proven.
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
                label="manifest-rebase evidence",
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
                        label="manifest-rebase evidence",
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
                    label="manifest-rebase evidence",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Use one fixed --updated-at-utc value for both phases. Run --dry-run "
            "first, review its new_realization_sha256, then pass that digest as "
            "--expected-new-realization-sha256 to --apply."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Prove the rebase and print the deterministic proposed hash without writes.",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Commit the previously reviewed proposal and private evidence.",
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--old-manifest-snapshot", type=Path, required=True)
    parser.add_argument("--expected-old-manifest-sha256", required=True)
    parser.add_argument("--expected-new-manifest-sha256", required=True)
    parser.add_argument("--expected-source-contracts-sha256", required=True)
    parser.add_argument("--expected-old-realization-sha256", required=True)
    parser.add_argument(
        "--expected-new-realization-sha256",
        help="Required in apply mode; obtain it from the matching dry run.",
    )
    parser.add_argument(
        "--updated-at-utc",
        required=True,
        help="Explicit advancing RFC3339 UTC timestamp used to make the proposal deterministic.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = rebase_realization(
            artifact_root=args.artifact_root,
            old_manifest_snapshot=args.old_manifest_snapshot,
            expected_old_manifest_sha256=args.expected_old_manifest_sha256,
            expected_new_manifest_sha256=args.expected_new_manifest_sha256,
            expected_source_contracts_sha256=args.expected_source_contracts_sha256,
            expected_old_realization_sha256=args.expected_old_realization_sha256,
            expected_new_realization_sha256=args.expected_new_realization_sha256,
            updated_at_utc=args.updated_at_utc,
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        print(f"[FAIL] realization rebase refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
