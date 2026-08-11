from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from .base import ContractModel, TimestampUs


class CapabilityStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class CapabilityState(ContractModel):
    capability: str = Field(min_length=1, max_length=160)
    status: CapabilityStatus
    checked_at_us: TimestampUs
    last_success_at_us: TimestampUs | None = None
    evidence: dict[str, Any]
    blockers: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _status_has_evidence(self) -> "CapabilityState":
        if self.status == CapabilityStatus.HEALTHY and self.last_success_at_us is None:
            raise ValueError("healthy capability requires last_success_at_us")
        if self.status in {CapabilityStatus.FAILED, CapabilityStatus.BLOCKED} and not self.blockers:
            raise ValueError("failed/blocked capability requires blocker evidence")
        return self


class CapabilityHealth(ContractModel):
    contract: Literal["noesis.capability.health"]
    contract_version: Literal[1]
    instance_id: str = Field(min_length=1, max_length=160)
    run_id: str = Field(min_length=1, max_length=160)
    generated_at_us: TimestampUs
    capabilities: tuple[CapabilityState, ...]

    @model_validator(mode="after")
    def _capabilities_are_unique(self) -> "CapabilityHealth":
        names = [item.capability for item in self.capabilities]
        if len(names) != len(set(names)):
            raise ValueError("capability names must be unique")
        return self
