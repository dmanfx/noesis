#!/usr/bin/env python3
"""Owner-private hash-chained DS9 runtime-ownership promotion registry."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


REGISTRY_DIRECTORY = "runtime-ownership"
REGISTRY_FILENAME = "promotions.jsonl"
REGISTRY_HEAD_FILENAME = "promotions.head.json"
REGISTRY_LOCK_FILENAME = ".promotions.lock"
REGISTRY_CONTRACT = "noesis.ds9.runtime-ownership-promotion-registry"
REGISTRY_HEAD_CONTRACT = "noesis.ds9.runtime-ownership-promotion-head"
EVENT_CONTRACT = "noesis.ds9.runtime-ownership-promotion"
EVENT_DIGEST_DOMAIN = b"noesis-ds9-runtime-ownership-promotion-v1\0"
MAX_REGISTRY_BYTES = 64 * 1024 * 1024
MAX_EVENT_BYTES = 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SUBJECT_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
RFC3339_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,9})?Z$"
)
EVENT_TYPES = frozenset({"promote", "revoke"})
EVIDENCE_TYPES = frozenset({"asset_realization", "runtime_session"})
EVENT_KEYS = frozenset(
    {
        "schema_version",
        "contract",
        "contract_version",
        "sequence",
        "event_type",
        "recorded_at_utc",
        "matrix_id",
        "matrix_sha256",
        "capability_id",
        "evidence_type",
        "subject",
        "selector",
        "runtime_binding",
        "checkout_sha256",
        "artifact_binding",
        "evidence_sha256s_digest",
        "previous_event_digest",
        "supersedes_event_digest",
        "event_digest",
    }
)
HEAD_KEYS = frozenset(
    {
        "schema_version",
        "contract",
        "contract_version",
        "event_count",
        "tail_event_digest",
        "registry_sha256",
        "updated_at_utc",
    }
)


class RegistryError(ValueError):
    """The external ownership-promotion registry is unsafe or inconsistent."""


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RegistryError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RegistryError(f"registry JSON is not canonicalizable: {exc}") from exc


def _parse_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RegistryError) as exc:
        raise RegistryError(f"{label} is invalid unique-key UTF-8 JSON: {exc}") from exc


def _event_digest(event: Mapping[str, Any]) -> str:
    body = dict(event)
    body.pop("event_digest", None)
    return hashlib.sha256(EVENT_DIGEST_DOMAIN + canonical_json_bytes(body)).hexdigest()


def _parse_timestamp(value: object, label: str) -> datetime:
    text = str(value or "")
    if RFC3339_UTC_RE.fullmatch(text) is None:
        raise RegistryError(f"{label} must be RFC3339 UTC")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise RegistryError(f"{label} must be RFC3339 UTC") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RegistryError(f"{label} must be explicit UTC")
    return parsed


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def event_key(event: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(event.get("capability_id") or ""),
        str(event.get("evidence_type") or ""),
        str(event.get("subject") or ""),
    )


@dataclass(frozen=True)
class RegistrySnapshot:
    events: tuple[Mapping[str, Any], ...]
    raw: bytes
    registry_sha256: str | None
    tail_event_digest: str | None
    key_heads: Mapping[tuple[str, str, str], Mapping[str, Any]]
    active: Mapping[tuple[str, str, str], Mapping[str, Any]]

    def active_for_matrix(
        self, matrix_id: str, matrix_sha256: str
    ) -> dict[tuple[str, str, str], Mapping[str, Any]]:
        return {
            key: event
            for key, event in self.active.items()
            if event.get("matrix_id") == matrix_id
            and event.get("matrix_sha256") == matrix_sha256
        }


EMPTY_SNAPSHOT = RegistrySnapshot((), b"", None, None, {}, {})


def _validate_events(events: Sequence[Mapping[str, Any]]) -> RegistrySnapshot:
    previous_digest: str | None = None
    previous_time: datetime | None = None
    key_heads: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    active: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    raw_rows: list[bytes] = []
    for index, event in enumerate(events, start=1):
        if set(event) != EVENT_KEYS:
            raise RegistryError(f"registry event {index} keys differ from the contract")
        if (
            type(event.get("schema_version")) is not int  # noqa: E721
            or event.get("schema_version") != 1
            or event.get("contract") != EVENT_CONTRACT
            or type(event.get("contract_version")) is not int  # noqa: E721
            or event.get("contract_version") != 1
            or type(event.get("sequence")) is not int  # noqa: E721
            or event.get("sequence") != index
            or event.get("event_type") not in EVENT_TYPES
            or not str(event.get("matrix_id") or "").strip()
            or SHA256_RE.fullmatch(str(event.get("matrix_sha256") or "")) is None
            or not str(event.get("capability_id") or "").strip()
            or event.get("evidence_type") not in EVIDENCE_TYPES
            or SUBJECT_RE.fullmatch(str(event.get("subject") or "")) is None
            or SHA256_RE.fullmatch(str(event.get("checkout_sha256") or "")) is None
        ):
            raise RegistryError(f"registry event {index} envelope is invalid")
        recorded_at = _parse_timestamp(
            event.get("recorded_at_utc"), f"registry event {index}.recorded_at_utc"
        )
        if previous_time is not None and recorded_at < previous_time:
            raise RegistryError("registry event timestamps move backwards")
        if event.get("previous_event_digest") != previous_digest:
            raise RegistryError(f"registry event {index} hash chain is not contiguous")
        digest = str(event.get("event_digest") or "")
        if SHA256_RE.fullmatch(digest) is None or digest != _event_digest(event):
            raise RegistryError(f"registry event {index} digest mismatch")
        key = event_key(event)
        prior_for_key = key_heads.get(key)
        expected_supersedes = (
            prior_for_key.get("event_digest") if prior_for_key is not None else None
        )
        if event.get("supersedes_event_digest") != expected_supersedes:
            raise RegistryError(
                f"registry event {index} does not explicitly supersede its key head"
            )
        if event.get("event_type") == "promote":
            if (
                not isinstance(event.get("selector"), Mapping)
                or not isinstance(event.get("artifact_binding"), Mapping)
            ):
                raise RegistryError(f"registry promotion event {index} lacks bindings")
            if event.get("evidence_type") == "runtime_session":
                if not isinstance(event.get("runtime_binding"), Mapping):
                    raise RegistryError(
                        f"registry runtime promotion event {index} lacks runtime binding"
                    )
                if SHA256_RE.fullmatch(
                    str(event.get("evidence_sha256s_digest") or "")
                ) is None:
                    raise RegistryError(
                        f"registry runtime promotion event {index} lacks evidence digest"
                    )
            elif (
                event.get("runtime_binding") is not None
                or event.get("evidence_sha256s_digest") is not None
            ):
                raise RegistryError(
                    f"registry asset promotion event {index} has runtime-only bindings"
                )
            active[key] = event
        else:
            if prior_for_key is None or key not in active:
                raise RegistryError(f"registry revoke event {index} has no active promotion")
            if any(
                event.get(field) is not None
                for field in (
                    "selector",
                    "runtime_binding",
                    "artifact_binding",
                    "evidence_sha256s_digest",
                )
            ):
                raise RegistryError(f"registry revoke event {index} carries evidence")
            active.pop(key, None)
        key_heads[key] = event
        previous_digest = digest
        previous_time = recorded_at
        raw_rows.append(canonical_json_bytes(event) + b"\n")
    raw = b"".join(raw_rows)
    if len(raw) > MAX_REGISTRY_BYTES:
        raise RegistryError("registry exceeds its byte bound")
    return RegistrySnapshot(
        events=tuple(events),
        raw=raw,
        registry_sha256=hashlib.sha256(raw).hexdigest() if raw else None,
        tail_event_digest=previous_digest,
        key_heads=key_heads,
        active=active,
    )


def validate_event_chain(
    events: Sequence[Mapping[str, Any]],
) -> RegistrySnapshot:
    return _validate_events(events)


def build_event(
    snapshot: RegistrySnapshot,
    *,
    event_type: str,
    matrix_id: str,
    matrix_sha256: str,
    capability_id: str,
    evidence_type: str,
    subject: str,
    selector: Mapping[str, Any] | None,
    runtime_binding: Mapping[str, Any] | None,
    checkout_sha256: str,
    artifact_binding: Mapping[str, Any] | None,
    evidence_sha256s_digest: str | None,
    supersedes_event_digest: str | None,
    recorded_at_utc: str | None = None,
) -> dict[str, Any]:
    key = (capability_id, evidence_type, subject)
    prior = snapshot.key_heads.get(key)
    expected_supersedes = prior.get("event_digest") if prior is not None else None
    if supersedes_event_digest != expected_supersedes:
        raise RegistryError(
            "promotion supersession must exactly name the current key head"
        )
    event: dict[str, Any] = {
        "schema_version": 1,
        "contract": EVENT_CONTRACT,
        "contract_version": 1,
        "sequence": len(snapshot.events) + 1,
        "event_type": event_type,
        "recorded_at_utc": recorded_at_utc or utc_now_text(),
        "matrix_id": matrix_id,
        "matrix_sha256": matrix_sha256,
        "capability_id": capability_id,
        "evidence_type": evidence_type,
        "subject": subject,
        "selector": dict(selector) if selector is not None else None,
        "runtime_binding": (
            dict(runtime_binding) if runtime_binding is not None else None
        ),
        "checkout_sha256": checkout_sha256,
        "artifact_binding": (
            dict(artifact_binding) if artifact_binding is not None else None
        ),
        "evidence_sha256s_digest": evidence_sha256s_digest,
        "previous_event_digest": snapshot.tail_event_digest,
        "supersedes_event_digest": supersedes_event_digest,
        "event_digest": None,
    }
    event["event_digest"] = _event_digest(event)
    _validate_events([*snapshot.events, event])
    return event


@dataclass
class _LockedRegistry:
    runtime_root: Path
    root_fd: int
    directory_fd: int
    lock_fd: int
    root_identity: tuple[int, int]
    directory_identity: tuple[int, int]


def _identity(info: os.stat_result) -> tuple[int, int]:
    return int(info.st_dev), int(info.st_ino)


def _validate_directory_info(info: os.stat_result, label: str) -> None:
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise RegistryError(f"{label} must be an owned mode-0700 directory")


def _validate_file_info(info: os.stat_result, label: str) -> None:
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise RegistryError(f"{label} must be an owned single-link mode-0600 file")


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            raise RegistryError(f"registry path component is missing: {current}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise RegistryError(f"registry path contains a symlink component: {current}")


def _verify_locked_identity(locked: _LockedRegistry) -> None:
    if _identity(os.fstat(locked.root_fd)) != locked.root_identity:
        raise RegistryError("runtime root descriptor identity changed")
    current_root = locked.runtime_root.lstat()
    if _identity(current_root) != locked.root_identity:
        raise RegistryError("runtime root pathname identity changed")
    current_directory = os.stat(
        REGISTRY_DIRECTORY,
        dir_fd=locked.root_fd,
        follow_symlinks=False,
    )
    if _identity(current_directory) != locked.directory_identity:
        raise RegistryError("registry directory pathname identity changed")
    if _identity(os.fstat(locked.directory_fd)) != locked.directory_identity:
        raise RegistryError("registry directory descriptor identity changed")


@contextmanager
def _locked_registry(
    runtime_root: Path,
    *,
    create: bool,
    exclusive: bool,
) -> Iterator[_LockedRegistry | None]:
    root = runtime_root.expanduser().absolute()
    _reject_symlink_components(root)
    try:
        root_info = root.lstat()
    except FileNotFoundError as exc:
        raise RegistryError(f"runtime root is missing: {root}") from exc
    _validate_directory_info(root_info, "runtime root")
    root_fd = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    directory_fd = -1
    lock_fd = -1
    try:
        if _identity(os.fstat(root_fd)) != _identity(root_info):
            raise RegistryError("runtime root changed while opening")
        try:
            directory_info = os.stat(
                REGISTRY_DIRECTORY, dir_fd=root_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            if not create:
                if _identity(root.lstat()) != _identity(root_info):
                    raise RegistryError("runtime root changed while checking registry")
                yield None
                return
            os.mkdir(REGISTRY_DIRECTORY, 0o700, dir_fd=root_fd)
            os.fsync(root_fd)
            directory_info = os.stat(
                REGISTRY_DIRECTORY, dir_fd=root_fd, follow_symlinks=False
            )
        _validate_directory_info(directory_info, "registry directory")
        directory_fd = os.open(
            REGISTRY_DIRECTORY,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=root_fd,
        )
        if _identity(os.fstat(directory_fd)) != _identity(directory_info):
            raise RegistryError("registry directory changed while opening")
        lock_flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(
            os, "O_CLOEXEC", 0
        )
        if create:
            lock_flags |= os.O_CREAT
        try:
            lock_fd = os.open(
                REGISTRY_LOCK_FILENAME, lock_flags, 0o600, dir_fd=directory_fd
            )
        except FileNotFoundError as exc:
            raise RegistryError("registry lock is missing") from exc
        lock_info = os.fstat(lock_fd)
        _validate_file_info(lock_info, "registry lock")
        if lock_info.st_size != 0:
            raise RegistryError("registry lock file must remain empty")
        named_lock = os.stat(
            REGISTRY_LOCK_FILENAME,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if _identity(named_lock) != _identity(lock_info):
            raise RegistryError("registry lock pathname changed while opening")
        try:
            fcntl.flock(
                lock_fd,
                (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RegistryError("another registry transaction owns the lock") from exc
        locked = _LockedRegistry(
            runtime_root=root,
            root_fd=root_fd,
            directory_fd=directory_fd,
            lock_fd=lock_fd,
            root_identity=_identity(root_info),
            directory_identity=_identity(directory_info),
        )
        _verify_locked_identity(locked)
        yield locked
        _verify_locked_identity(locked)
    finally:
        if lock_fd >= 0:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        if directory_fd >= 0:
            os.close(directory_fd)
        os.close(root_fd)


def _read_named_file(
    locked: _LockedRegistry,
    name: str,
    *,
    required: bool,
    max_bytes: int,
) -> bytes | None:
    try:
        named_before = os.stat(
            name, dir_fd=locked.directory_fd, follow_symlinks=False
        )
    except FileNotFoundError:
        if required:
            raise RegistryError(f"registry file is missing: {name}")
        return None
    _validate_file_info(named_before, f"registry file {name}")
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=locked.directory_fd,
    )
    try:
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(named_before):
            raise RegistryError(f"registry file {name} changed while opening")
        payload = bytearray()
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > max_bytes:
                raise RegistryError(f"registry file {name} exceeds its byte bound")
        named_after = os.stat(
            name, dir_fd=locked.directory_fd, follow_symlinks=False
        )
        if _identity(named_after) != _identity(opened):
            raise RegistryError(f"registry file {name} changed while reading")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _snapshot_from_locked(locked: _LockedRegistry) -> RegistrySnapshot:
    entries = set(os.listdir(locked.directory_fd))
    allowed = {
        REGISTRY_LOCK_FILENAME,
        REGISTRY_FILENAME,
        REGISTRY_HEAD_FILENAME,
    }
    if entries - allowed or REGISTRY_LOCK_FILENAME not in entries:
        raise RegistryError(
            "registry directory entry set drifted: "
            f"unexpected={sorted(entries - allowed)}"
        )
    registry_raw = _read_named_file(
        locked,
        REGISTRY_FILENAME,
        required=False,
        max_bytes=MAX_REGISTRY_BYTES,
    )
    head_raw = _read_named_file(
        locked,
        REGISTRY_HEAD_FILENAME,
        required=False,
        max_bytes=64 * 1024,
    )
    if registry_raw is None and head_raw is None:
        return EMPTY_SNAPSHOT
    if registry_raw is None or head_raw is None:
        raise RegistryError("registry and persisted head must either both exist or both be absent")
    if not registry_raw or not registry_raw.endswith(b"\n"):
        raise RegistryError("registry JSONL is empty or has an incomplete final row")
    events: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(registry_raw.splitlines(), start=1):
        if not line or len(line) > MAX_EVENT_BYTES:
            raise RegistryError(f"registry event line {line_number} exceeds its bound")
        event = _parse_json(line, f"registry event line {line_number}")
        if not isinstance(event, Mapping):
            raise RegistryError(f"registry event line {line_number} must be an object")
        if canonical_json_bytes(event) != line:
            raise RegistryError(f"registry event line {line_number} is not canonical JSON")
        events.append(event)
    snapshot = _validate_events(events)
    if snapshot.raw != registry_raw:
        raise RegistryError("registry JSONL canonical replay drifted")
    head = _parse_json(head_raw, "registry persisted head")
    if not isinstance(head, Mapping) or set(head) != HEAD_KEYS:
        raise RegistryError("registry persisted head keys differ from the contract")
    if head_raw != canonical_json_bytes(head) + b"\n":
        raise RegistryError("registry persisted head is not canonical JSON")
    if (
        type(head.get("schema_version")) is not int  # noqa: E721
        or head.get("schema_version") != 1
        or head.get("contract") != REGISTRY_HEAD_CONTRACT
        or type(head.get("contract_version")) is not int  # noqa: E721
        or head.get("contract_version") != 1
        or type(head.get("event_count")) is not int  # noqa: E721
        or head.get("event_count") != len(snapshot.events)
        or head.get("tail_event_digest") != snapshot.tail_event_digest
        or head.get("registry_sha256") != snapshot.registry_sha256
        or head.get("updated_at_utc") != snapshot.events[-1].get("recorded_at_utc")
    ):
        raise RegistryError("registry persisted head does not match the JSONL chain")
    return snapshot


def read_registry(runtime_root: str | Path) -> RegistrySnapshot:
    with _locked_registry(Path(runtime_root), create=False, exclusive=False) as locked:
        if locked is None:
            return EMPTY_SNAPSHOT
        return _snapshot_from_locked(locked)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RegistryError("registry append made no progress")
        view = view[written:]


def _replace_head(
    locked: _LockedRegistry,
    snapshot: RegistrySnapshot,
) -> None:
    assert snapshot.events and snapshot.registry_sha256 and snapshot.tail_event_digest
    head = {
        "schema_version": 1,
        "contract": REGISTRY_HEAD_CONTRACT,
        "contract_version": 1,
        "event_count": len(snapshot.events),
        "tail_event_digest": snapshot.tail_event_digest,
        "registry_sha256": snapshot.registry_sha256,
        "updated_at_utc": snapshot.events[-1]["recorded_at_utc"],
    }
    encoded = canonical_json_bytes(head) + b"\n"
    temporary = f".{REGISTRY_HEAD_FILENAME}.tmp-{os.getpid()}-{time.time_ns()}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=locked.directory_fd,
    )
    try:
        _validate_file_info(os.fstat(descriptor), "registry temporary head")
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(
            temporary,
            REGISTRY_HEAD_FILENAME,
            src_dir_fd=locked.directory_fd,
            dst_dir_fd=locked.directory_fd,
        )
        os.fsync(locked.directory_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=locked.directory_fd)
        except FileNotFoundError:
            pass


def append_registry_event(
    runtime_root: str | Path,
    builder: Callable[[RegistrySnapshot], Mapping[str, Any]],
) -> Mapping[str, Any]:
    with _locked_registry(Path(runtime_root), create=True, exclusive=True) as locked:
        if locked is None:  # pragma: no cover - create=True always returns a handle
            raise RegistryError("registry lock was not created")
        before = _snapshot_from_locked(locked)
        event = dict(builder(before))
        after = _validate_events([*before.events, event])
        if after.raw[: len(before.raw)] != before.raw:
            raise RegistryError("registry append would rewrite existing history")
        row = after.raw[len(before.raw) :]
        if len(row) > MAX_EVENT_BYTES + 1 or not row.endswith(b"\n"):
            raise RegistryError("registry append row exceeds its bound")
        terminal_before = _snapshot_from_locked(locked)
        if terminal_before.raw != before.raw:
            raise RegistryError("registry changed while the promotion was validated")
        descriptor = os.open(
            REGISTRY_FILENAME,
            os.O_RDWR
            | os.O_APPEND
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=locked.directory_fd,
        )
        try:
            opened = os.fstat(descriptor)
            _validate_file_info(opened, "registry JSONL")
            existing = bytearray()
            os.lseek(descriptor, 0, os.SEEK_SET)
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                existing.extend(chunk)
                if len(existing) > MAX_REGISTRY_BYTES:
                    raise RegistryError("registry JSONL exceeds its byte bound")
            if bytes(existing) != before.raw:
                raise RegistryError("registry JSONL changed before append")
            _write_all(descriptor, row)
            os.fsync(descriptor)
            named = os.stat(
                REGISTRY_FILENAME,
                dir_fd=locked.directory_fd,
                follow_symlinks=False,
            )
            if _identity(named) != _identity(opened):
                raise RegistryError("registry JSONL pathname changed during append")
        finally:
            os.close(descriptor)
        _replace_head(locked, after)
        os.fsync(locked.directory_fd)
        verified = _snapshot_from_locked(locked)
        if verified.raw != after.raw:
            raise RegistryError("registry append failed its final compare-and-swap")
        _verify_locked_identity(locked)
        return event


__all__ = [
    "EMPTY_SNAPSHOT",
    "EVENT_CONTRACT",
    "REGISTRY_DIRECTORY",
    "REGISTRY_FILENAME",
    "REGISTRY_HEAD_FILENAME",
    "REGISTRY_HEAD_CONTRACT",
    "RegistryError",
    "RegistrySnapshot",
    "append_registry_event",
    "build_event",
    "canonical_json_bytes",
    "event_key",
    "read_registry",
    "utc_now_text",
    "validate_event_chain",
]
