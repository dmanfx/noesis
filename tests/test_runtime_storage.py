from __future__ import annotations

from types import SimpleNamespace

import pytest

from geometry.depth_source import DepthStorageError, StorageLifecycle
from noesis.runtime_storage import close_depth_storage, storage_close_evidence


def _flush(*, completed: bool = True, poison=None):  # type: ignore[no-untyped-def]
    return SimpleNamespace(
        frontier_sequence=7,
        completed=completed,
        timed_out=not completed,
        pending_sequences=() if completed else (7,),
        failed_sequences=(),
        poison=poison,
    )


def _shutdown(*, completed: bool = True, flush=None):  # type: ignore[no-untyped-def]
    return SimpleNamespace(
        state=StorageLifecycle.CLOSED if completed else StorageLifecycle.CLOSING,
        completed=completed,
        timed_out=not completed,
        flush=flush or _flush(completed=completed),
        alive_writer_names=() if completed else ("writer",),
        enforcer_alive=False,
    )


class _Storage:
    def __init__(self, flush, shutdown) -> None:  # type: ignore[no-untyped-def]
        self.flush_receipt = flush
        self.shutdown_receipt = shutdown
        self.calls: list[str] = []

    def flush(self, timeout):  # type: ignore[no-untyped-def]
        assert timeout >= 0.0
        self.calls.append("flush")
        return self.flush_receipt

    def shutdown(self, *, wait, timeout):  # type: ignore[no-untyped-def]
        assert wait is True
        assert timeout >= 0.0
        self.calls.append("shutdown")
        return self.shutdown_receipt


def test_runtime_storage_close_requires_both_exact_receipts() -> None:
    flush = _flush()
    storage = _Storage(flush, _shutdown(flush=flush))

    receipt = close_depth_storage(storage, timeout_s=5.0)  # type: ignore[arg-type]

    assert storage.calls == ["flush", "shutdown"]
    assert storage_close_evidence(receipt) == {
        "completed": True,
        "frontier_sequence": 7,
        "state": "closed",
        "alive_writer_names": [],
        "enforcer_alive": False,
        "poisoned": False,
        "failed_sequences": [],
    }


def test_runtime_storage_close_still_attempts_shutdown_after_flush_failure() -> None:
    flush = _flush(completed=False)
    storage = _Storage(flush, _shutdown())

    with pytest.raises(DepthStorageError, match="did not close cleanly"):
        close_depth_storage(storage, timeout_s=5.0)  # type: ignore[arg-type]

    assert storage.calls == ["flush", "shutdown"]


def test_runtime_storage_close_rejects_poison_even_after_threads_stop() -> None:
    poison = object()
    flush = _flush(poison=poison)
    storage = _Storage(flush, _shutdown(flush=flush))

    with pytest.raises(DepthStorageError, match="poison=True"):
        close_depth_storage(storage, timeout_s=5.0)  # type: ignore[arg-type]
