from __future__ import annotations

import threading
import time

import pytest

from noesis_core.runtime_publication import (
    RuntimePublicationGate,
    RuntimePublicationQuiescenceError,
)


def _wait_until_closed(gate: RuntimePublicationGate) -> None:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if gate.snapshot().admission_closed:
            return
        threading.Event().wait(0.005)
    raise AssertionError("publication gate did not close admission")


def test_close_waits_for_inflight_lease_then_rejects_new_work() -> None:
    gate = RuntimePublicationGate()
    lease = gate.acquire()
    assert lease is not None
    receipts: list[object] = []

    shutdown = threading.Thread(
        target=lambda: receipts.append(gate.close_and_wait(timeout_s=1.0))
    )
    shutdown.start()
    _wait_until_closed(gate)

    assert shutdown.is_alive()
    assert gate.acquire() is None
    lease.release()
    shutdown.join(timeout=1.0)

    assert not shutdown.is_alive()
    receipt = receipts[0]
    assert receipt.quiesced is True
    assert receipt.admitted == receipt.completed == 1
    assert receipt.rejected == 1
    assert receipt.active == 0


def test_timeout_is_fail_closed_and_gate_never_reopens() -> None:
    gate = RuntimePublicationGate()
    lease = gate.acquire()
    assert lease is not None

    with pytest.raises(RuntimePublicationQuiescenceError) as raised:
        gate.close_and_wait(timeout_s=0.01)

    assert raised.value.receipt.admission_closed is True
    assert raised.value.receipt.active == 1
    assert gate.acquire() is None
    lease.release()
    assert gate.close_and_wait(timeout_s=0.1).quiesced is True


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_shutdown_timeout_must_be_finite_and_positive(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        RuntimePublicationGate().close_and_wait(timeout_s=timeout)


def test_lease_release_is_exactly_once() -> None:
    gate = RuntimePublicationGate()
    lease = gate.acquire()
    assert lease is not None
    lease.release()
    with pytest.raises(RuntimeError, match="more than once"):
        lease.release()
