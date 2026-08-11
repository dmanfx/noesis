from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from .base import ContractModel, TimestampUs


class ActorRole(StrEnum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    OWNER = "owner"
    AGENT = "agent"
    AUTOMATION = "automation"


class ActionRisk(StrEnum):
    ROUTINE = "routine"
    SENSITIVE = "sensitive"
    DESTRUCTIVE = "destructive"


class ActionStatus(StrEnum):
    PROPOSED = "proposed"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    REJECTED = "rejected"


class ActorRef(ContractModel):
    actor_id: str = Field(min_length=1, max_length=160)
    role: ActorRole
    session_id: str | None = Field(default=None, max_length=200)


class ActionIntent(ContractModel):
    contract: Literal["noesis.action.intent"]
    contract_version: Literal[1]
    action_id: str = Field(min_length=1, max_length=200)
    idempotency_key: str = Field(min_length=16, max_length=240)
    actor: ActorRef
    capability: str = Field(min_length=1, max_length=160)
    target: str = Field(min_length=1, max_length=240)
    parameters: dict[str, Any]
    requested_at_us: TimestampUs
    risk: ActionRisk
    precondition: dict[str, Any] | None = None


class ActionApproval(ContractModel):
    contract: Literal["noesis.action.approval"]
    contract_version: Literal[1]
    approval_id: str = Field(min_length=1, max_length=200)
    action_id: str = Field(min_length=1, max_length=200)
    approver: ActorRef
    issued_at_us: TimestampUs
    expires_at_us: TimestampUs
    decision: Literal["approve", "reject"]
    one_use: Literal[True] = True

    @model_validator(mode="after")
    def _approval_is_valid(self) -> "ActionApproval":
        if self.expires_at_us <= self.issued_at_us:
            raise ValueError("approval must expire after it is issued")
        if self.approver.role in {ActorRole.AGENT, ActorRole.AUTOMATION}:
            raise ValueError("agents and automations may not approve actions")
        return self


class ExecutionReceipt(ContractModel):
    contract: Literal["noesis.action.receipt"]
    contract_version: Literal[1]
    receipt_id: str = Field(min_length=1, max_length=200)
    action_id: str = Field(min_length=1, max_length=200)
    status: ActionStatus
    started_at_us: TimestampUs
    completed_at_us: TimestampUs
    provider: str = Field(min_length=1, max_length=160)
    provider_result: dict[str, Any] | None = None
    readback: dict[str, Any] | None = None
    error: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _receipt_is_truthful(self) -> "ExecutionReceipt":
        if self.completed_at_us < self.started_at_us:
            raise ValueError("receipt completion cannot precede start")
        if self.status == ActionStatus.SUCCEEDED and self.readback is None:
            raise ValueError("successful actions require provider readback")
        if self.status in {ActionStatus.FAILED, ActionStatus.UNCERTAIN, ActionStatus.REJECTED} and not self.error:
            raise ValueError("non-success terminal actions require an explanation")
        return self


class AuditEntry(ContractModel):
    contract: Literal["noesis.audit.entry"]
    contract_version: Literal[1]
    audit_id: str = Field(min_length=1, max_length=200)
    occurred_at_us: TimestampUs
    actor: ActorRef
    event: str = Field(min_length=1, max_length=200)
    action_id: str | None = Field(default=None, max_length=200)
    receipt_id: str | None = Field(default=None, max_length=200)
    redacted_details: dict[str, Any]
    previous_entry_sha256: str | None = None
    entry_sha256: str
