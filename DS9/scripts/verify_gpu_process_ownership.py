#!/usr/bin/env python3
"""Prove that GPU process IDs are descendants of one container init process.

The host maintenance wrapper deliberately does not trust a ``docker top``
snapshot: a builder can exec or fork between that snapshot and the NVML query.
Instead, this helper walks the host ``/proc`` parent chain and binds every PID
edge to Linux's process start-time identity so PID reuse cannot silently turn a
foreign process into an accepted owner.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


class ProcessVanished(RuntimeError):
    """Raised when a process exits while its ownership is being proved."""


class ProcessProofError(RuntimeError):
    """Raised when process ancestry is unreadable, malformed, or foreign."""


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    parent_pid: int
    start_time_ticks: int


def _read_process_identity(proc_root: Path, pid: int) -> ProcessIdentity:
    if pid <= 0:
        raise ProcessProofError(f"invalid process ID: {pid}")
    path = proc_root / str(pid) / "stat"
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ProcessVanished(f"process {pid} vanished") from exc
    except OSError as exc:
        raise ProcessProofError(f"cannot read process {pid} identity: {exc}") from exc

    # proc_pid_stat(5) permits spaces and ')' in comm, so split after the final
    # closing parenthesis.  The remaining fields begin with state (field 3).
    closing = raw.rfind(")")
    opening = raw.find("(")
    if opening <= 0 or closing <= opening:
        raise ProcessProofError(f"process {pid} has malformed /proc stat data")
    try:
        observed_pid = int(raw[:opening].strip())
        fields = raw[closing + 1 :].split()
        parent_pid = int(fields[1])  # field 4
        start_time_ticks = int(fields[19])  # field 22
    except (IndexError, ValueError) as exc:
        raise ProcessProofError(
            f"process {pid} has malformed identity fields"
        ) from exc
    if observed_pid != pid or parent_pid < 0 or start_time_ticks <= 0:
        raise ProcessProofError(f"process {pid} has invalid identity fields")
    return ProcessIdentity(pid, parent_pid, start_time_ticks)


def _same_identity(
    proc_root: Path, expected: ProcessIdentity, *, label: str
) -> ProcessIdentity:
    observed = _read_process_identity(proc_root, expected.pid)
    if observed != expected:
        raise ProcessProofError(
            f"{label} PID identity changed during ancestry proof: "
            f"expected={asdict(expected)} observed={asdict(observed)}"
        )
    return observed


def prove_descendant(
    *,
    owner_pid: int,
    container_init_pid: int,
    expected_container_init_start_time_ticks: int,
    proc_root: Path = Path("/proc"),
    maximum_depth: int = 4096,
) -> dict[str, Any]:
    """Return a start-time-bound ancestry proof for one GPU owner PID."""

    if container_init_pid <= 1:
        raise ProcessProofError(
            f"container init PID must be greater than one: {container_init_pid}"
        )
    if maximum_depth <= 0:
        raise ProcessProofError("maximum ancestry depth must be positive")

    init_identity = _read_process_identity(proc_root, container_init_pid)
    if init_identity.start_time_ticks != expected_container_init_start_time_ticks:
        raise ProcessProofError(
            "container init PID identity differs from the inspected start time: "
            f"expected={expected_container_init_start_time_ticks} "
            f"observed={init_identity.start_time_ticks}"
        )
    owner_identity = _read_process_identity(proc_root, owner_pid)
    current = owner_identity
    visited: set[tuple[int, int]] = set()
    chain: list[dict[str, int]] = []

    for _depth in range(maximum_depth):
        identity_key = (current.pid, current.start_time_ticks)
        if identity_key in visited:
            raise ProcessProofError(
                f"process ancestry loop while proving GPU owner {owner_pid}"
            )
        visited.add(identity_key)
        chain.append(asdict(current))

        if current.pid == container_init_pid:
            if current.start_time_ticks != init_identity.start_time_ticks:
                raise ProcessProofError(
                    "container init PID was reused during ancestry proof"
                )
            _same_identity(proc_root, init_identity, label="container init")
            _same_identity(proc_root, owner_identity, label="GPU owner")
            return {
                "owner_pid": owner_pid,
                "owner_start_time_ticks": owner_identity.start_time_ticks,
                "container_init_pid": container_init_pid,
                "container_init_start_time_ticks": init_identity.start_time_ticks,
                "chain": chain,
                "status": "descendant",
            }

        if current.parent_pid <= 1:
            raise ProcessProofError(
                f"GPU owner {owner_pid} escapes container init ancestry at "
                f"pid={current.pid} parent={current.parent_pid}"
            )

        # Bind the edge before following it.  If the child exits or its PID is
        # recycled, the caller must requery NVML instead of accepting stale
        # ancestry.
        _same_identity(proc_root, current, label=f"ancestry process {current.pid}")
        current = _read_process_identity(proc_root, current.parent_pid)

    raise ProcessProofError(
        f"GPU owner {owner_pid} ancestry exceeds {maximum_depth} processes"
    )


def prove_owners(
    *,
    owner_pids: Iterable[int],
    container_init_pid: int,
    expected_container_init_start_time_ticks: int,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    init_identity = _read_process_identity(proc_root, container_init_pid)
    if init_identity.start_time_ticks != expected_container_init_start_time_ticks:
        raise ProcessProofError(
            "container init PID identity differs from the inspected start time: "
            f"expected={expected_container_init_start_time_ticks} "
            f"observed={init_identity.start_time_ticks}"
        )
    proofs: list[dict[str, Any]] = []
    vanished: list[dict[str, Any]] = []
    for owner_pid in sorted(set(owner_pids)):
        try:
            proofs.append(
                prove_descendant(
                    owner_pid=owner_pid,
                    container_init_pid=container_init_pid,
                    expected_container_init_start_time_ticks=(
                        expected_container_init_start_time_ticks
                    ),
                    proc_root=proc_root,
                )
            )
        except ProcessVanished as exc:
            vanished.append({"owner_pid": owner_pid, "reason": str(exc)})
    return {
        "ok": not vanished,
        "container_init_pid": container_init_pid,
        "proofs": proofs,
        "vanished": vanished,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-init-pid", type=int, required=True)
    parser.add_argument("--container-init-start-time-ticks", type=int)
    parser.add_argument("--snapshot-init", action="store_true")
    parser.add_argument("--owner-pid", type=int, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.snapshot_init:
            if args.owner_pid or args.container_init_start_time_ticks is not None:
                raise ProcessProofError(
                    "--snapshot-init does not accept owners or a prior start time"
                )
            identity = _read_process_identity(Path("/proc"), args.container_init_pid)
            print(
                json.dumps(
                    {
                        "container_init_pid": identity.pid,
                        "container_init_start_time_ticks": identity.start_time_ticks,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if (
            args.container_init_start_time_ticks is None
            or args.container_init_start_time_ticks <= 0
        ):
            raise ProcessProofError(
                "--container-init-start-time-ticks must be a positive inspected identity"
            )
        result = prove_owners(
            owner_pids=args.owner_pid,
            container_init_pid=args.container_init_pid,
            expected_container_init_start_time_ticks=(
                args.container_init_start_time_ticks
            ),
        )
    except ProcessProofError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    except ProcessVanished as exc:
        # This can occur while reading the init PID before per-owner handling.
        print(json.dumps({"ok": False, "vanished": [{"reason": str(exc)}]}, sort_keys=True))
        return 4
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
