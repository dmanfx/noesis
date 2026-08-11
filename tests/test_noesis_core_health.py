from __future__ import annotations

import pytest

from noesis_core.contracts.health import CapabilityStatus
from noesis_core.health import CapabilityMonitor, CapabilityPolicy, CapabilityProgressError


def _monitor() -> CapabilityMonitor:
    return CapabilityMonitor(
        instance_id="appliance",
        run_id="runtime-1",
        policies={
            "tracking": CapabilityPolicy(stale_after_us=100, fail_after_us=300),
            "identity": CapabilityPolicy(stale_after_us=200, fail_after_us=400),
        },
    )


def test_health_requires_real_progress_and_ages_truthfully() -> None:
    monitor = _monitor()
    initial = monitor.snapshot(generated_at_us=1_000)
    assert all(row.status == CapabilityStatus.UNKNOWN for row in initial.capabilities)

    monitor.record_success(
        "tracking",
        producer_run_id="source-run",
        sequence=1,
        observed_at_us=990,
        checked_at_us=1_000,
        contract_compatible=True,
        evidence={"open_port": True},
    )
    healthy = monitor.snapshot(generated_at_us=1_050)
    tracking = next(row for row in healthy.capabilities if row.capability == "tracking")
    assert tracking.status == CapabilityStatus.HEALTHY
    assert tracking.evidence["sequence"] == 1
    assert monitor.snapshot(generated_at_us=1_150).capabilities[1].status == CapabilityStatus.DEGRADED
    assert monitor.snapshot(generated_at_us=1_350).capabilities[1].status == CapabilityStatus.FAILED


def test_health_rejects_non_monotonic_progress_but_allows_new_run() -> None:
    monitor = _monitor()
    kwargs = dict(observed_at_us=900, checked_at_us=1_000, contract_compatible=True)
    monitor.record_success("tracking", producer_run_id="a", sequence=2, **kwargs)
    with pytest.raises(CapabilityProgressError, match="non-monotonic"):
        monitor.record_success("tracking", producer_run_id="a", sequence=2, **kwargs)
    monitor.record_success(
        "tracking",
        producer_run_id="b",
        sequence=0,
        observed_at_us=1_100,
        checked_at_us=1_101,
        contract_compatible=True,
    )
    assert monitor.snapshot(generated_at_us=1_102).capabilities[1].evidence["producer_run_id"] == "b"


def test_contract_mismatch_and_explicit_blockers_fail_closed() -> None:
    monitor = _monitor()
    monitor.record_success(
        "identity",
        producer_run_id="identity-run",
        sequence=1,
        observed_at_us=900,
        checked_at_us=1_000,
        contract_compatible=False,
    )
    identity = next(row for row in monitor.snapshot(generated_at_us=1_001).capabilities if row.capability == "identity")
    assert identity.status == CapabilityStatus.FAILED
    assert identity.blockers == ("contract_incompatible",)

    monitor.record_failure(
        "tracking",
        checked_at_us=1_001,
        blockers=("artifact_missing",),
        blocked=True,
    )
    tracking = next(row for row in monitor.snapshot(generated_at_us=1_002).capabilities if row.capability == "tracking")
    assert tracking.status == CapabilityStatus.BLOCKED
