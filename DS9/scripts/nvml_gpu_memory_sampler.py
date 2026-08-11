#!/usr/bin/env python3
"""Continuously enforce a DS9 maintenance GPU-memory guard through NVML."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import signal
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, TextIO


CONTRACT = "noesis.ds9.nvml_gpu_memory_guard.v1"
MIB = 1024 * 1024
MAX_EVIDENCE_BYTES = 32 * 1024 * 1024
MAX_SAMPLE_ROWS = 100_000
REVIEWED_SAMPLE_INTERVAL_MS = 25
REVIEWED_MAX_GAP_MS = 250
REVIEWED_GUARD_MIB_BY_ENGINE = {
    "mapanything": 9000,
    "wholebody49_s_masks": 10000,
    "wholebody49_x_boxes": 11000,
    "bodypose3dnet": 9000,
    "v3dt_tracker_reid": 9000,
}
REVIEWED_DEFAULT_GUARD_MIB = 11000
NVML_SUCCESS = 0
EXIT_GUARD_BREACH = 42
EXIT_PARENT_LOST = 43
EXIT_CADENCE_GAP = 44
EXIT_SAMPLER_ERROR = 45
EXIT_SAMPLE_LIMIT = 46
_RFC3339_UTC = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$"
)


class SamplerError(RuntimeError):
    """Raised when sampler evidence or NVML state is unsafe."""


def reviewed_guard_mib(engine: str) -> int:
    return REVIEWED_GUARD_MIB_BY_ENGINE.get(engine, REVIEWED_DEFAULT_GUARD_MIB)


class NvmlMemoryV2(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _proc_start_time_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    closing = raw.rfind(")")
    if closing <= 0:
        raise SamplerError(f"malformed process identity for PID {pid}")
    fields = raw[closing + 1 :].split()
    if len(fields) <= 19:
        raise SamplerError(f"incomplete process identity for PID {pid}")
    value = int(fields[19])
    if value <= 0:
        raise SamplerError(f"invalid process start time for PID {pid}")
    return value


def _parent_is_exact(pid: int, start_time_ticks: int) -> bool:
    try:
        return _proc_start_time_ticks(pid) == start_time_ticks
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return False


def _notify_exact_parent(pid: int, start_time_ticks: int) -> None:
    if not _parent_is_exact(pid, start_time_ticks):
        return
    try:
        pidfd = os.pidfd_open(pid)
    except (AttributeError, ProcessLookupError, PermissionError, OSError) as exc:
        raise SamplerError(
            f"unable to open identity-bound parent pidfd: {exc}"
        ) from exc
    try:
        if not _parent_is_exact(pid, start_time_ticks):
            raise SamplerError("parent identity changed before guard notification")
        signal.pidfd_send_signal(pidfd, signal.SIGUSR1)
    finally:
        os.close(pidfd)


class NvmlBackend:
    def __init__(self, device_index: int, expected_uuid: str) -> None:
        self._library = ctypes.CDLL("libnvidia-ml.so.1")
        self._configure()
        self._check(self._library.nvmlInit_v2(), "nvmlInit_v2")
        self._initialized = True
        handle = ctypes.c_void_p()
        self._check(
            self._library.nvmlDeviceGetHandleByIndex_v2(
                ctypes.c_uint(device_index), ctypes.byref(handle)
            ),
            "nvmlDeviceGetHandleByIndex_v2",
        )
        if not handle.value:
            raise SamplerError("NVML returned a null device handle")
        self._handle = handle
        uuid_buffer = ctypes.create_string_buffer(96)
        self._check(
            self._library.nvmlDeviceGetUUID(
                self._handle, uuid_buffer, ctypes.c_uint(len(uuid_buffer))
            ),
            "nvmlDeviceGetUUID",
        )
        observed_uuid = uuid_buffer.value.decode("ascii", errors="strict")
        if observed_uuid != expected_uuid:
            raise SamplerError(
                "NVML GPU UUID differs from reviewed authority: "
                f"expected={expected_uuid} observed={observed_uuid}"
            )

    def _configure(self) -> None:
        library = self._library
        library.nvmlInit_v2.argtypes = []
        library.nvmlInit_v2.restype = ctypes.c_int
        library.nvmlShutdown.argtypes = []
        library.nvmlShutdown.restype = ctypes.c_int
        library.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        library.nvmlDeviceGetUUID.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        library.nvmlDeviceGetUUID.restype = ctypes.c_int
        library.nvmlDeviceGetMemoryInfo_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(NvmlMemoryV2),
        ]
        library.nvmlDeviceGetMemoryInfo_v2.restype = ctypes.c_int
        library.nvmlErrorString.argtypes = [ctypes.c_int]
        library.nvmlErrorString.restype = ctypes.c_char_p

    def _check(self, result: int, operation: str) -> None:
        if result == NVML_SUCCESS:
            return
        raw = self._library.nvmlErrorString(result)
        description = raw.decode("utf-8", errors="replace") if raw else "unknown"
        raise SamplerError(f"{operation} failed: code={result} desc={description}")

    def sample(self) -> tuple[int, int, int]:
        memory = NvmlMemoryV2()
        memory.version = ctypes.sizeof(NvmlMemoryV2) | (2 << 24)
        self._check(
            self._library.nvmlDeviceGetMemoryInfo_v2(
                self._handle, ctypes.byref(memory)
            ),
            "nvmlDeviceGetMemoryInfo_v2",
        )
        return int(memory.total), int(memory.reserved), int(memory.used)

    def close(self) -> None:
        if getattr(self, "_initialized", False):
            result = self._library.nvmlShutdown()
            self._initialized = False
            self._check(result, "nvmlShutdown")


def _open_evidence(path: Path) -> TextIO:
    parent_before = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or stat.S_ISLNK(parent_before.st_mode)
        or stat.S_IMODE(parent_before.st_mode) != 0o700
        or parent_before.st_uid != os.getuid()
    ):
        raise SamplerError("sampler evidence parent directory is not private")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    opened = os.fstat(descriptor)
    path_after = path.lstat()
    parent_after = path.parent.lstat()
    if (
        not stat.S_ISREG(opened.st_mode)
        or stat.S_IMODE(opened.st_mode) != 0o600
        or opened.st_uid != os.getuid()
        or opened.st_nlink != 1
        or (path_after.st_dev, path_after.st_ino) != (opened.st_dev, opened.st_ino)
        or (parent_after.st_dev, parent_after.st_ino)
        != (parent_before.st_dev, parent_before.st_ino)
    ):
        os.close(descriptor)
        raise SamplerError("sampler evidence path identity is unsafe after creation")
    return os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)


def _write_row(
    handle: TextIO, row: Mapping[str, Any], *, durable: bool = False
) -> None:
    encoded = json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
    if handle.tell() + len(encoded.encode("utf-8")) > MAX_EVIDENCE_BYTES:
        raise SamplerError("sampler evidence would exceed its size bound")
    handle.write(encoded)
    handle.flush()
    if durable:
        os.fsync(handle.fileno())


def _mib_ceil(value: int) -> int:
    return (value + MIB - 1) // MIB


def sample(
    args: argparse.Namespace, *, backend_type: type[NvmlBackend] = NvmlBackend
) -> int:
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    interval_ns = args.interval_ms * 1_000_000
    max_gap_ns = args.max_gap_ms * 1_000_000
    count = 0
    peak_mib = 0
    peak_reserved_mib = 0
    first_utc: str | None = None
    last_utc: str | None = None
    previous_monotonic_ns: int | None = None
    maximum_gap_ns = 0
    backend: NvmlBackend | None = None
    with _open_evidence(args.evidence) as evidence:

        def fail_closed(row: Mapping[str, Any], code: int) -> int:
            _write_row(evidence, row, durable=True)
            _notify_exact_parent(args.parent_pid, args.parent_start_time_ticks)
            return code

        try:
            backend = backend_type(args.device_index, args.expected_uuid)
            header = {
                "kind": "header",
                "schema_version": 1,
                "contract": CONTRACT,
                "device_index": args.device_index,
                "expected_uuid": args.expected_uuid,
                "engine": args.engine,
                "transaction_id": args.transaction_id,
                "prepared_transaction_sha256": args.prepared_transaction_sha256,
                "artifact_root_id": args.artifact_root_id,
                "container_id": args.container_id,
                "guard_mib": args.guard_mib,
                "guard_bytes": args.guard_mib * MIB,
                "interval_ms": args.interval_ms,
                "max_gap_ms": args.max_gap_ms,
                "parent_pid": args.parent_pid,
                "parent_start_time_ticks": args.parent_start_time_ticks,
                "sampler_pid": os.getpid(),
                "sampler_start_time_ticks": _proc_start_time_ticks(os.getpid()),
                "started_at_utc": _utc_now(),
            }
            _write_row(evidence, header, durable=True)
            next_deadline = time.monotonic_ns()
            while not stop_requested:
                if not _parent_is_exact(args.parent_pid, args.parent_start_time_ticks):
                    return fail_closed(
                        {
                            "kind": "footer",
                            "state": "parent_lost",
                            "sample_count": count,
                            "peak_mib": peak_mib,
                            "ended_at_utc": _utc_now(),
                        },
                        EXIT_PARENT_LOST,
                    )
                sampled_monotonic_ns = time.monotonic_ns()
                if previous_monotonic_ns is not None:
                    gap_ns = sampled_monotonic_ns - previous_monotonic_ns
                    maximum_gap_ns = max(maximum_gap_ns, gap_ns)
                    if gap_ns > max_gap_ns:
                        return fail_closed(
                            {
                                "kind": "footer",
                                "state": "cadence_gap",
                                "sample_count": count,
                                "peak_mib": peak_mib,
                                "gap_ms": gap_ns / 1_000_000,
                                "ended_at_utc": _utc_now(),
                            },
                            EXIT_CADENCE_GAP,
                        )
                total, reserved, used = backend.sample()
                sampled_at_utc = _utc_now()
                used_mib = _mib_ceil(used)
                reserved_mib = _mib_ceil(reserved)
                count += 1
                peak_mib = max(peak_mib, used_mib)
                peak_reserved_mib = max(peak_reserved_mib, reserved_mib)
                first_utc = first_utc or sampled_at_utc
                last_utc = sampled_at_utc
                _write_row(
                    evidence,
                    {
                        "kind": "sample",
                        "sequence": count,
                        "sampled_at_utc": sampled_at_utc,
                        "monotonic_ns": sampled_monotonic_ns,
                        "total_mib": _mib_ceil(total),
                        "total_bytes": total,
                        "reserved_mib": reserved_mib,
                        "reserved_bytes": reserved,
                        "used_mib": used_mib,
                        "used_bytes": used,
                    },
                )
                previous_monotonic_ns = sampled_monotonic_ns
                if used > args.guard_mib * MIB:
                    return fail_closed(
                        {
                            "kind": "footer",
                            "state": "guard_breached",
                            "sample_count": count,
                            "peak_mib": peak_mib,
                            "peak_reserved_mib": peak_reserved_mib,
                            "breach_mib": used_mib,
                            "breach_bytes": used,
                            "breach_at_utc": sampled_at_utc,
                            "first_sample_at_utc": first_utc,
                            "last_sample_at_utc": last_utc,
                            "maximum_gap_ms": maximum_gap_ns / 1_000_000,
                            "ended_at_utc": _utc_now(),
                        },
                        EXIT_GUARD_BREACH,
                    )
                if count >= args.max_samples:
                    return fail_closed(
                        {
                            "kind": "footer",
                            "state": "sample_limit",
                            "sample_count": count,
                            "peak_mib": peak_mib,
                            "ended_at_utc": _utc_now(),
                        },
                        EXIT_SAMPLE_LIMIT,
                    )
                next_deadline += interval_ns
                remaining_ns = next_deadline - time.monotonic_ns()
                if remaining_ns > 0:
                    time.sleep(remaining_ns / 1_000_000_000)
            backend.close()
            backend = None
            _write_row(
                evidence,
                {
                    "kind": "footer",
                    "state": "stopped",
                    "sample_count": count,
                    "peak_mib": peak_mib,
                    "peak_reserved_mib": peak_reserved_mib,
                    "first_sample_at_utc": first_utc,
                    "last_sample_at_utc": last_utc,
                    "maximum_gap_ms": maximum_gap_ns / 1_000_000,
                    "ended_at_utc": _utc_now(),
                },
                durable=True,
            )
            return 0
        except BaseException as exc:
            try:
                return fail_closed(
                    {
                        "kind": "footer",
                        "state": "sampler_error",
                        "error": f"{type(exc).__name__}: {exc}"[:1024],
                        "sample_count": count,
                        "peak_mib": peak_mib,
                        "ended_at_utc": _utc_now(),
                    },
                    EXIT_SAMPLER_ERROR,
                )
            except BaseException:
                return EXIT_SAMPLER_ERROR
        finally:
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    pass


def _canonical_utc(value: object) -> bool:
    if not isinstance(value, str) or _RFC3339_UTC.fullmatch(value) is None:
        return False
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return True


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SamplerError(f"sampler evidence contains duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_rows(path: Path) -> tuple[list[Mapping[str, Any]], bytes]:
    parent = path.parent
    parent_metadata = parent.lstat()
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or stat.S_ISLNK(parent_metadata.st_mode)
        or stat.S_IMODE(parent_metadata.st_mode) != 0o700
        or parent_metadata.st_uid != os.getuid()
    ):
        raise SamplerError("sampler evidence parent directory is not private")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, MAX_EVIDENCE_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_EVIDENCE_BYTES:
                raise SamplerError("sampler evidence exceeds its size bound")
        after = os.fstat(descriptor)
        path_after = path.lstat()
        parent_after = parent.lstat()
    finally:
        os.close(descriptor)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, key) != getattr(after, key) for key in identity_fields):
        raise SamplerError("sampler evidence changed during its authoritative read")
    if (
        stat.S_ISLNK(path_after.st_mode)
        or (path_after.st_dev, path_after.st_ino) != (before.st_dev, before.st_ino)
        or (parent_after.st_dev, parent_after.st_ino)
        != (parent_metadata.st_dev, parent_metadata.st_ino)
        or stat.S_IMODE(parent_after.st_mode) != 0o700
        or parent_after.st_uid != os.getuid()
    ):
        raise SamplerError(
            "sampler evidence pathname changed during authoritative read"
        )
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_uid != os.getuid()
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size != size
    ):
        raise SamplerError("sampler evidence ownership/mode/size is unsafe")
    raw = b"".join(chunks)
    if not raw.endswith(b"\n"):
        raise SamplerError("sampler evidence lacks a complete final line")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SamplerError("sampler evidence is not UTF-8") from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise SamplerError(
                f"sampler evidence line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, Mapping):
            raise SamplerError(f"sampler evidence line {line_number} is not a mapping")
        rows.append(row)
    return rows, raw


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    rows, evidence_raw = _load_rows(args.evidence)
    if len(rows) < 2 or rows[0].get("kind") != "header":
        raise SamplerError("sampler evidence lacks header and samples")
    header = rows[0]
    header_keys = {
        "kind",
        "schema_version",
        "contract",
        "device_index",
        "expected_uuid",
        "engine",
        "transaction_id",
        "prepared_transaction_sha256",
        "artifact_root_id",
        "container_id",
        "guard_mib",
        "guard_bytes",
        "interval_ms",
        "max_gap_ms",
        "parent_pid",
        "parent_start_time_ticks",
        "sampler_pid",
        "sampler_start_time_ticks",
        "started_at_utc",
    }
    if set(header) != header_keys or not _canonical_utc(header.get("started_at_utc")):
        raise SamplerError("sampler header keys/timestamp are invalid")
    if (
        isinstance(header.get("device_index"), bool)
        or not isinstance(header.get("device_index"), int)
        or int(header["device_index"]) < 0
        or isinstance(header.get("guard_mib"), bool)
        or not isinstance(header.get("guard_mib"), int)
        or int(header["guard_mib"]) <= 0
        or isinstance(header.get("guard_bytes"), bool)
        or not isinstance(header.get("guard_bytes"), int)
        or int(header["guard_bytes"]) <= 0
        or isinstance(header.get("interval_ms"), bool)
        or not isinstance(header.get("interval_ms"), int)
        or int(header["interval_ms"]) <= 0
        or isinstance(header.get("max_gap_ms"), bool)
        or not isinstance(header.get("max_gap_ms"), int)
        or int(header["max_gap_ms"]) < int(header["interval_ms"])
        or isinstance(header.get("parent_pid"), bool)
        or not isinstance(header.get("parent_pid"), int)
        or int(header["parent_pid"]) <= 1
        or isinstance(header.get("parent_start_time_ticks"), bool)
        or not isinstance(header.get("parent_start_time_ticks"), int)
        or int(header["parent_start_time_ticks"]) <= 0
        or isinstance(header.get("sampler_pid"), bool)
        or not isinstance(header.get("sampler_pid"), int)
        or int(header["sampler_pid"]) <= 1
        or isinstance(header.get("sampler_start_time_ticks"), bool)
        or not isinstance(header.get("sampler_start_time_ticks"), int)
        or int(header["sampler_start_time_ticks"]) <= 0
    ):
        raise SamplerError("sampler header process identity is invalid")
    expected_header = {
        "schema_version": 1,
        "contract": CONTRACT,
        "device_index": args.device_index,
        "expected_uuid": args.expected_uuid,
        "engine": args.engine,
        "transaction_id": args.transaction_id,
        "prepared_transaction_sha256": args.prepared_transaction_sha256,
        "artifact_root_id": args.artifact_root_id,
        "container_id": args.container_id,
        "guard_mib": args.guard_mib,
        "guard_bytes": args.guard_mib * MIB,
        "interval_ms": args.interval_ms,
        "max_gap_ms": args.max_gap_ms,
        "parent_pid": args.parent_pid,
        "parent_start_time_ticks": args.parent_start_time_ticks,
    }
    for key, expected in expected_header.items():
        if header.get(key) != expected:
            raise SamplerError(f"sampler header differs from authority: {key}")
    footer = rows[-1] if rows[-1].get("kind") == "footer" else None
    sample_rows = rows[1:-1] if footer is not None else rows[1:]
    if any(row.get("kind") != "sample" for row in sample_rows):
        raise SamplerError("sampler evidence row order/shape is invalid")
    samples = list(sample_rows)
    if not samples:
        raise SamplerError("sampler evidence contains zero samples")
    previous = None
    peak = 0
    peak_bytes = 0
    peak_reserved = 0
    maximum_gap_ns = 0
    for sequence, row in enumerate(samples, start=1):
        if set(row) != {
            "kind",
            "sequence",
            "sampled_at_utc",
            "monotonic_ns",
            "total_mib",
            "total_bytes",
            "reserved_mib",
            "reserved_bytes",
            "used_mib",
            "used_bytes",
        }:
            raise SamplerError("sampler sample keys are invalid")
        if row.get("sequence") != sequence:
            raise SamplerError("sampler evidence sequence is not contiguous")
        monotonic_ns = row.get("monotonic_ns")
        used_mib = row.get("used_mib")
        used_bytes = row.get("used_bytes")
        reserved_mib = row.get("reserved_mib")
        reserved_bytes = row.get("reserved_bytes")
        total_mib = row.get("total_mib")
        total_bytes = row.get("total_bytes")
        if (
            isinstance(monotonic_ns, bool)
            or not isinstance(monotonic_ns, int)
            or monotonic_ns <= 0
            or isinstance(used_mib, bool)
            or not isinstance(used_mib, int)
            or used_mib < 0
            or isinstance(used_bytes, bool)
            or not isinstance(used_bytes, int)
            or used_bytes < 0
            or used_mib != _mib_ceil(used_bytes)
            or isinstance(reserved_mib, bool)
            or not isinstance(reserved_mib, int)
            or reserved_mib < 0
            or isinstance(reserved_bytes, bool)
            or not isinstance(reserved_bytes, int)
            or reserved_bytes < 0
            or reserved_mib != _mib_ceil(reserved_bytes)
            or isinstance(total_bytes, bool)
            or not isinstance(total_bytes, int)
            or total_bytes <= 0
            or isinstance(total_mib, bool)
            or not isinstance(total_mib, int)
            or total_mib != _mib_ceil(total_bytes)
            or used_bytes > total_bytes
            or reserved_bytes > total_bytes
            or not _canonical_utc(row.get("sampled_at_utc"))
        ):
            raise SamplerError("sampler sample fields are invalid")
        if previous is not None:
            gap_ns = monotonic_ns - previous
            if gap_ns <= 0 or gap_ns > args.max_gap_ms * 1_000_000:
                raise SamplerError("sampler evidence contains an excessive cadence gap")
            maximum_gap_ns = max(maximum_gap_ns, gap_ns)
        previous = monotonic_ns
        peak = max(peak, used_mib)
        peak_bytes = max(peak_bytes, used_bytes)
        peak_reserved = max(peak_reserved, reserved_mib)
    if footer is None:
        if not args.allow_active:
            raise SamplerError("sampler evidence lacks a terminal footer")
        state = "active"
    else:
        state = str(footer.get("state") or "")
        if state not in {
            "stopped",
            "guard_breached",
            "parent_lost",
            "cadence_gap",
            "sampler_error",
            "sample_limit",
        }:
            raise SamplerError("sampler footer state is invalid")
        footer_keys_by_state = {
            "stopped": {
                "kind",
                "state",
                "sample_count",
                "peak_mib",
                "peak_reserved_mib",
                "first_sample_at_utc",
                "last_sample_at_utc",
                "maximum_gap_ms",
                "ended_at_utc",
            },
            "guard_breached": {
                "kind",
                "state",
                "sample_count",
                "peak_mib",
                "peak_reserved_mib",
                "breach_mib",
                "breach_bytes",
                "breach_at_utc",
                "first_sample_at_utc",
                "last_sample_at_utc",
                "maximum_gap_ms",
                "ended_at_utc",
            },
            "parent_lost": {
                "kind",
                "state",
                "sample_count",
                "peak_mib",
                "ended_at_utc",
            },
            "cadence_gap": {
                "kind",
                "state",
                "sample_count",
                "peak_mib",
                "gap_ms",
                "ended_at_utc",
            },
            "sampler_error": {
                "kind",
                "state",
                "error",
                "sample_count",
                "peak_mib",
                "ended_at_utc",
            },
            "sample_limit": {
                "kind",
                "state",
                "sample_count",
                "peak_mib",
                "ended_at_utc",
            },
        }
        if set(footer) != footer_keys_by_state[state] or not _canonical_utc(
            footer.get("ended_at_utc")
        ):
            raise SamplerError("sampler footer keys/timestamp are invalid")
        if footer.get("sample_count") != len(samples) or footer.get("peak_mib") != peak:
            raise SamplerError("sampler footer count/peak differs from samples")
        if state in {"stopped", "guard_breached"} and (
            footer.get("first_sample_at_utc") != samples[0]["sampled_at_utc"]
            or footer.get("last_sample_at_utc") != samples[-1]["sampled_at_utc"]
            or footer.get("peak_reserved_mib") != peak_reserved
            or isinstance(footer.get("maximum_gap_ms"), bool)
            or not isinstance(footer.get("maximum_gap_ms"), (int, float))
            or float(footer["maximum_gap_ms"]) != maximum_gap_ns / 1_000_000
        ):
            raise SamplerError("sampler footer sample summary drifted")
        if state == "guard_breached" and (
            footer.get("breach_bytes") != peak_bytes
            or footer.get("breach_mib") != peak
            or footer.get("breach_at_utc") != samples[-1]["sampled_at_utc"]
        ):
            raise SamplerError("sampler breach footer differs from breaching sample")
    guard_breached = peak_bytes > args.guard_mib * MIB
    if state == "stopped" and guard_breached:
        raise SamplerError("sampler stopped footer conceals a guard breach")
    if state == "guard_breached" and not guard_breached:
        raise SamplerError("sampler guard-breach footer lacks a breaching sample")
    return {
        "ok": True,
        "state": state,
        "sample_count": len(samples),
        "peak_mib": peak,
        "maximum_observed_bytes": peak_bytes,
        "maximum_observed_mib": peak,
        "peak_reserved_mib": peak_reserved,
        "first_sample_at_utc": samples[0]["sampled_at_utc"],
        "last_sample_at_utc": samples[-1]["sampled_at_utc"],
        "maximum_gap_ms": maximum_gap_ns / 1_000_000,
        "breach_at_utc": footer.get("breach_at_utc") if footer else None,
        "breach_mib": footer.get("breach_mib") if footer else None,
        "guard_ok": state == "stopped" and not guard_breached,
        "evidence_sha256": hashlib.sha256(evidence_raw).hexdigest(),
    }


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--transaction-id", required=True)
    parser.add_argument("--prepared-transaction-sha256", required=True)
    parser.add_argument("--artifact-root-id", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--guard-mib", type=int, required=True)
    parser.add_argument("--interval-ms", type=int, required=True)
    parser.add_argument("--max-gap-ms", type=int, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--parent-start-time-ticks", type=int, required=True)


def signal_exact_process(pid: int, start_time_ticks: int, signum: int) -> None:
    if not _parent_is_exact(pid, start_time_ticks):
        raise SamplerError("sampler process identity no longer matches")
    pidfd = os.pidfd_open(pid)
    try:
        if not _parent_is_exact(pid, start_time_ticks):
            raise SamplerError("sampler process identity changed before signal")
        signal.pidfd_send_signal(pidfd, signum)
    finally:
        os.close(pidfd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    sample_parser = subparsers.add_parser("sample")
    _common_arguments(sample_parser)
    sample_parser.add_argument("--max-samples", type=int, required=True)
    summarize_parser = subparsers.add_parser("summarize")
    _common_arguments(summarize_parser)
    summarize_parser.add_argument("--allow-active", action="store_true")
    signal_parser = subparsers.add_parser("signal")
    signal_parser.add_argument("--pid", type=int, required=True)
    signal_parser.add_argument("--start-time-ticks", type=int, required=True)
    signal_parser.add_argument("--signal", choices=("TERM",), required=True)
    args = parser.parse_args()
    if args.operation == "signal":
        if args.pid <= 1 or args.start_time_ticks <= 0:
            parser.error("invalid sampler process identity")
        return args
    if (
        args.device_index < 0
        or args.guard_mib <= 0
        or not 10 <= args.interval_ms <= 1000
        or args.max_gap_ms < args.interval_ms
        or args.parent_pid <= 1
        or args.parent_start_time_ticks <= 0
        or not args.expected_uuid.startswith("GPU-")
        or not args.engine.replace("_", "").isalnum()
        or not args.transaction_id.replace("-", "").replace("_", "").isalnum()
        or len(args.prepared_transaction_sha256) != 64
        or any(
            value not in "0123456789abcdef"
            for value in args.prepared_transaction_sha256
        )
        or len(args.artifact_root_id) != 64
        or any(value not in "0123456789abcdef" for value in args.artifact_root_id)
        or len(args.container_id) != 64
        or any(value not in "0123456789abcdef" for value in args.container_id)
        or (args.operation == "sample" and not 0 < args.max_samples <= MAX_SAMPLE_ROWS)
    ):
        parser.error("invalid sampler contract arguments")
    return args


def main() -> int:
    args = parse_args()
    if args.operation == "sample":
        return sample(args)
    if args.operation == "signal":
        try:
            signal_exact_process(args.pid, args.start_time_ticks, signal.SIGTERM)
        except Exception as exc:
            print(f"[FAIL] {exc}", file=os.sys.stderr)
            return 2
        return 0
    try:
        result = summarize(args)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
