"""Read and verify immutable container-era NVML guard receipts.

This module is compatibility-only. Native-host engine maintenance does not
start an NVML sampler or emit this receipt shape; the verifier remains so the
asset validator can authenticate already-realized historical engine receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


CONTRACT = "noesis.ds9.nvml_gpu_memory_guard.v1"
MIB = 1024 * 1024
MAX_EVIDENCE_BYTES = 32 * 1024 * 1024
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
_RFC3339_UTC = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$"
)


class SamplerError(RuntimeError):
    """Raised when a historical NVML guard receipt is unsafe."""


def reviewed_guard_mib(engine: str) -> int:
    return REVIEWED_GUARD_MIB_BY_ENGINE.get(engine, REVIEWED_DEFAULT_GUARD_MIB)


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


def _mib_ceil(value: int) -> int:
    return (value + MIB - 1) // MIB


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
