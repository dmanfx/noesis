from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimePublicationReceipt:
    """Exact admission/drain state for native runtime publication callbacks."""

    contract: str
    contract_version: int
    admission_closed: bool
    quiesced: bool
    admitted: int
    completed: int
    rejected: int
    active: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimePublicationQuiescenceError(RuntimeError):
    """Raised when admitted native publication callbacks do not drain in time."""

    def __init__(self, message: str, receipt: RuntimePublicationReceipt) -> None:
        super().__init__(message)
        self.receipt = receipt


class RuntimePublicationLease:
    """One exactly-once lease owned by an admitted publication callback."""

    __slots__ = ("_gate", "_released")

    def __init__(self, gate: "RuntimePublicationGate") -> None:
        self._gate = gate
        self._released = False

    def __enter__(self) -> "RuntimePublicationLease":
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc: object,
        _traceback: object,
    ) -> None:
        self.release()

    def release(self) -> None:
        if self._released:
            raise RuntimeError("runtime publication lease released more than once")
        self._released = True
        self._gate._release()


class RuntimePublicationGate:
    """Close and drain native callbacks before WebSocket egress shuts down."""

    CONTRACT = "noesis.runtime-publication-gate"
    CONTRACT_VERSION = 1

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._admission_closed = False
        self._admitted = 0
        self._completed = 0
        self._rejected = 0
        self._active = 0

    def acquire(self) -> RuntimePublicationLease | None:
        with self._condition:
            if self._admission_closed:
                self._rejected += 1
                return None
            self._admitted += 1
            self._active += 1
            return RuntimePublicationLease(self)

    def _release(self) -> None:
        with self._condition:
            if self._active <= 0 or self._completed >= self._admitted:
                raise RuntimeError("runtime publication gate lease accounting underflow")
            self._active -= 1
            self._completed += 1
            if self._active == 0:
                self._condition.notify_all()

    def _receipt_locked(self) -> RuntimePublicationReceipt:
        quiesced = bool(
            self._admission_closed
            and self._active == 0
            and self._completed == self._admitted
        )
        return RuntimePublicationReceipt(
            contract=self.CONTRACT,
            contract_version=self.CONTRACT_VERSION,
            admission_closed=self._admission_closed,
            quiesced=quiesced,
            admitted=self._admitted,
            completed=self._completed,
            rejected=self._rejected,
            active=self._active,
        )

    def snapshot(self) -> RuntimePublicationReceipt:
        with self._condition:
            return self._receipt_locked()

    def close_and_wait(self, *, timeout_s: float) -> RuntimePublicationReceipt:
        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError(
                "runtime publication shutdown timeout must be finite and positive"
            )
        deadline = time.monotonic() + timeout
        with self._condition:
            self._admission_closed = True
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    receipt = self._receipt_locked()
                    raise RuntimePublicationQuiescenceError(
                        "native runtime publication callbacks did not quiesce",
                        receipt,
                    )
                self._condition.wait(timeout=remaining)
            return self._receipt_locked()


__all__ = [
    "RuntimePublicationGate",
    "RuntimePublicationLease",
    "RuntimePublicationQuiescenceError",
    "RuntimePublicationReceipt",
]
