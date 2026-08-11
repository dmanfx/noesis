"""Strict runtime integration helpers for transactional depth storage."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from geometry.depth_source import (
    DepthStorageError,
    DepthStorageManager,
    FlushReceipt,
    ShutdownReceipt,
    StorageLifecycle,
)


@dataclass(frozen=True)
class RuntimeStorageCloseReceipt:
    flush: FlushReceipt
    shutdown: ShutdownReceipt
    completed: bool


def close_depth_storage(
    storage: DepthStorageManager,
    *,
    timeout_s: float,
) -> RuntimeStorageCloseReceipt:
    """Close admission, drain the exact frontier, and reject poison or timeout."""

    timeout = float(timeout_s)
    if timeout <= 0.0:
        raise ValueError("depth storage close timeout must be positive")
    deadline = time.monotonic() + timeout
    flush = storage.flush(timeout=max(0.0, deadline - time.monotonic()))
    shutdown = storage.shutdown(
        wait=True,
        timeout=max(0.0, deadline - time.monotonic()),
    )
    completed = bool(
        flush.completed
        and not flush.timed_out
        and not flush.failed_sequences
        and flush.poison is None
        and shutdown.completed
        and not shutdown.timed_out
        and shutdown.state is StorageLifecycle.CLOSED
        and shutdown.flush.completed
        and not shutdown.flush.failed_sequences
        and shutdown.flush.poison is None
        and not shutdown.alive_writer_names
        and not shutdown.enforcer_alive
    )
    receipt = RuntimeStorageCloseReceipt(
        flush=flush,
        shutdown=shutdown,
        completed=completed,
    )
    if not completed:
        raise DepthStorageError(
            "depth storage did not close cleanly "
            f"(flush_completed={flush.completed} flush_timed_out={flush.timed_out} "
            f"failed_sequences={list(flush.failed_sequences)} poison={flush.poison is not None} "
            f"shutdown_completed={shutdown.completed} state={shutdown.state.value} "
            f"alive_writers={list(shutdown.alive_writer_names)} "
            f"enforcer_alive={shutdown.enforcer_alive})"
        )
    return receipt


def storage_close_evidence(receipt: RuntimeStorageCloseReceipt) -> dict[str, Any]:
    return {
        "completed": bool(receipt.completed),
        "frontier_sequence": int(receipt.flush.frontier_sequence),
        "state": receipt.shutdown.state.value,
        "alive_writer_names": list(receipt.shutdown.alive_writer_names),
        "enforcer_alive": bool(receipt.shutdown.enforcer_alive),
        "poisoned": receipt.flush.poison is not None,
        "failed_sequences": list(receipt.flush.failed_sequences),
    }


__all__ = [
    "RuntimeStorageCloseReceipt",
    "close_depth_storage",
    "storage_close_evidence",
]
