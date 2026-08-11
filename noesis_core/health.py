from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Mapping

from noesis_core.contracts.health import CapabilityHealth, CapabilityState, CapabilityStatus


class CapabilityProgressError(ValueError):
    pass


@dataclass(frozen=True)
class CapabilityPolicy:
    stale_after_us: int
    fail_after_us: int

    def __post_init__(self) -> None:
        if self.stale_after_us <= 0 or self.fail_after_us < self.stale_after_us:
            raise ValueError("expected 0 < stale_after_us <= fail_after_us")


@dataclass
class _Progress:
    producer_run_id: str
    sequence: int
    observed_at_us: int
    success_at_us: int
    evidence: dict[str, Any]
    forced_status: CapabilityStatus | None = None
    blockers: tuple[str, ...] = ()


class CapabilityMonitor:
    """Readiness based on compatible producer progress, never socket reachability."""

    def __init__(
        self,
        *,
        instance_id: str,
        run_id: str,
        policies: Mapping[str, CapabilityPolicy],
    ) -> None:
        if not str(instance_id).strip() or not str(run_id).strip():
            raise ValueError("instance_id and run_id are required")
        if not policies:
            raise ValueError("at least one capability policy is required")
        self.instance_id = str(instance_id)
        self.run_id = str(run_id)
        self.policies = dict(policies)
        self._progress: dict[str, _Progress] = {}
        self._lock = RLock()

    def record_success(
        self,
        capability: str,
        *,
        producer_run_id: str,
        sequence: int,
        observed_at_us: int,
        checked_at_us: int,
        contract_compatible: bool,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        name = self._require_capability(capability)
        if not contract_compatible:
            self.record_failure(
                name,
                checked_at_us=checked_at_us,
                blockers=("contract_incompatible",),
                evidence=evidence,
            )
            return
        if not str(producer_run_id).strip():
            raise CapabilityProgressError("producer_run_id is required")
        if int(sequence) < 0 or int(observed_at_us) <= 0 or int(checked_at_us) <= 0:
            raise CapabilityProgressError("sequence/timestamps are invalid")
        if int(observed_at_us) > int(checked_at_us):
            raise CapabilityProgressError("observed_at_us cannot follow checked_at_us")
        with self._lock:
            previous = self._progress.get(name)
            if previous is not None and previous.producer_run_id == str(producer_run_id):
                if int(sequence) <= previous.sequence:
                    raise CapabilityProgressError(
                        f"non-monotonic {name} sequence: received={sequence} last={previous.sequence}"
                    )
                if int(observed_at_us) < previous.observed_at_us:
                    raise CapabilityProgressError(f"non-monotonic {name} observation time")
            self._progress[name] = _Progress(
                producer_run_id=str(producer_run_id),
                sequence=int(sequence),
                observed_at_us=int(observed_at_us),
                success_at_us=int(checked_at_us),
                evidence={
                    **dict(evidence or {}),
                    "producer_run_id": str(producer_run_id),
                    "sequence": int(sequence),
                    "observed_at_us": int(observed_at_us),
                    "contract_compatible": True,
                },
            )

    def record_failure(
        self,
        capability: str,
        *,
        checked_at_us: int,
        blockers: tuple[str, ...],
        evidence: Mapping[str, Any] | None = None,
        blocked: bool = False,
    ) -> None:
        name = self._require_capability(capability)
        clean_blockers = tuple(str(item).strip() for item in blockers if str(item).strip())
        if not clean_blockers:
            raise CapabilityProgressError("failure requires blocker evidence")
        with self._lock:
            previous = self._progress.get(name)
            self._progress[name] = _Progress(
                producer_run_id=previous.producer_run_id if previous else "none",
                sequence=previous.sequence if previous else 0,
                observed_at_us=previous.observed_at_us if previous else int(checked_at_us),
                success_at_us=previous.success_at_us if previous else 0,
                evidence={**(previous.evidence if previous else {}), **dict(evidence or {})},
                forced_status=CapabilityStatus.BLOCKED if blocked else CapabilityStatus.FAILED,
                blockers=clean_blockers,
            )

    def snapshot(self, *, generated_at_us: int) -> CapabilityHealth:
        if int(generated_at_us) <= 0:
            raise ValueError("generated_at_us must be positive")
        rows: list[CapabilityState] = []
        with self._lock:
            progress = dict(self._progress)
        for name, policy in sorted(self.policies.items()):
            current = progress.get(name)
            if current is None:
                rows.append(
                    CapabilityState(
                        capability=name,
                        status=CapabilityStatus.UNKNOWN,
                        checked_at_us=int(generated_at_us),
                        last_success_at_us=None,
                        evidence={"producer_progress": False},
                        blockers=("no_producer_progress",),
                    )
                )
                continue
            if current.forced_status is not None:
                rows.append(
                    CapabilityState(
                        capability=name,
                        status=current.forced_status,
                        checked_at_us=int(generated_at_us),
                        last_success_at_us=(current.success_at_us if current.success_at_us > 0 else None),
                        evidence=dict(current.evidence),
                        blockers=current.blockers,
                    )
                )
                continue
            age_us = max(0, int(generated_at_us) - current.success_at_us)
            evidence = {**current.evidence, "progress_age_us": age_us}
            if age_us <= policy.stale_after_us:
                status = CapabilityStatus.HEALTHY
                blockers: tuple[str, ...] = ()
            elif age_us <= policy.fail_after_us:
                status = CapabilityStatus.DEGRADED
                blockers = ("producer_progress_stale",)
            else:
                status = CapabilityStatus.FAILED
                blockers = ("producer_progress_expired",)
            rows.append(
                CapabilityState(
                    capability=name,
                    status=status,
                    checked_at_us=int(generated_at_us),
                    last_success_at_us=current.success_at_us,
                    evidence=evidence,
                    blockers=blockers,
                )
            )
        return CapabilityHealth(
            contract="noesis.capability.health",
            contract_version=1,
            instance_id=self.instance_id,
            run_id=self.run_id,
            generated_at_us=int(generated_at_us),
            capabilities=tuple(rows),
        )

    def _require_capability(self, capability: str) -> str:
        name = str(capability).strip()
        if name not in self.policies:
            raise CapabilityProgressError(f"unknown capability: {name}")
        return name
