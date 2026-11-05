"""Private persisted-event inspection; never current world or replay authority."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import Field, StrictInt, field_validator, model_validator
from .base import ContractModel, Sha256

MAX_HISTORY_ROWS = 50
MAX_HISTORY_BYTES = 256 * 1024
HISTORY_LEASE_US = 30_000_000


class WorldHistoryRecord(ContractModel):
    sequence: StrictInt = Field(ge=0, le=2**53 - 1)
    recorded_at_us: StrictInt = Field(ge=1, le=2**53 - 1)
    previous_sha256: Sha256
    record_sha256: Sha256
    payload: dict[str, Any]

    @field_validator("payload")
    @classmethod
    def _persisted_event(cls, value):
        from noesis_core.replay import validate_stored_contract_payload
        if value.get("contract") != "noesis.world.event":
            raise ValueError("history accepts only original WorldEvent rows")
        return validate_stored_contract_payload(value)


class WorldHistoryInspection(ContractModel):
    anchor_sequence: StrictInt = Field(ge=-1, le=2**53 - 1)
    anchor_sha256: Sha256
    retained_first_sequence: StrictInt = Field(ge=0, le=2**53 - 1)
    retained_first_previous_sha256: Sha256
    created_at_us: StrictInt = Field(ge=1, le=2**53 - 1)
    expires_at_us: StrictInt = Field(ge=1, le=2**53 - 1)

    @model_validator(mode="after")
    def _finite_original_lease(self):
        if self.expires_at_us - self.created_at_us != HISTORY_LEASE_US:
            raise ValueError("invalid original inspection lease")
        return self


class WorldHistoryPersistence(ContractModel):
    status: Literal["synchronous", "healthy", "failed", "closed"]
    pending_batches: StrictInt | None = Field(default=None, ge=0)
    inflight_batches: StrictInt | None = Field(default=None, ge=0)
    worker_alive: bool | None = None
    rejected_cohorts: StrictInt = Field(ge=0)
    error_reported: bool


class WorldHistoryRead(ContractModel):
    contract: Literal["noesis.world.history"] = "noesis.world.history"
    version: Literal[1] = 1
    status: Literal["available", "partial", "unavailable"]
    reason: str = Field(min_length=1, max_length=96)
    coverage: Literal["retained_persisted_events_only"] = "retained_persisted_events_only"
    inspection: WorldHistoryInspection | None = None
    persistence: WorldHistoryPersistence | None = None
    events: tuple[WorldHistoryRecord, ...] = Field(default=(), max_length=MAX_HISTORY_ROWS)
    next_cursor: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{43}$")

    @model_validator(mode="after")
    def _qualified(self):
        if self.status == "unavailable" and (self.events or self.next_cursor or self.inspection):
            raise ValueError("unavailable history cannot expose a page")
        if self.status != "unavailable" and self.persistence is None:
            raise ValueError("history requires persistence evidence")
        if self.events and self.inspection is None:
            raise ValueError("events require an inspection anchor")
        return self
