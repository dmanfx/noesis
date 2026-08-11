from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .base import Confidence, ContractModel, NonNegativeFloat, TimestampUs


class IdentityKind(StrEnum):
    RESIDENT = "resident"
    VISITOR = "visitor"
    UNKNOWN = "unknown"
    PROVISIONAL = "provisional"


class IdentityOutcome(StrEnum):
    RESIDENT = "resident"
    VISITOR = "visitor"
    UNKNOWN = "unknown"


class TrackletRef(ContractModel):
    run_id: str = Field(min_length=1, max_length=160)
    camera_id: str = Field(min_length=1, max_length=160)
    source_id: int = Field(ge=0)
    tracker_id: int = Field(ge=0)
    frame_id: int = Field(ge=0)
    observed_at_us: TimestampUs


class SubjectRef(ContractModel):
    subject_id: str = Field(min_length=1, max_length=200)
    kind: IdentityKind
    generation: int = Field(default=0, ge=0)
    resident_uuid: str | None = Field(default=None, max_length=160)
    display_name: str | None = Field(default=None, max_length=160)
    stable_id: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _resident_fields_are_coherent(self) -> "SubjectRef":
        if self.kind == IdentityKind.RESIDENT and not self.resident_uuid:
            raise ValueError("resident subjects require resident_uuid")
        if self.kind != IdentityKind.RESIDENT and self.resident_uuid is not None:
            raise ValueError("only resident subjects may carry resident_uuid")
        if self.kind in {IdentityKind.UNKNOWN, IdentityKind.PROVISIONAL} and self.display_name is not None:
            raise ValueError("unknown/provisional subjects may not expose display_name")
        return self


class IdentityEvidenceSummary(ContractModel):
    embedding_present: bool
    embedding_sequence: int | None = Field(default=None, ge=0)
    embedding_model_sha256: str | None = None
    embedding_dimension: int | None = Field(default=None, ge=1)
    appearance_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    calibrated_confidence: Confidence | None = None
    ambiguity_margin: NonNegativeFloat | None = None
    prior_contribution: float = Field(default=0.0, ge=0.0)
    independent_observation_count: int = Field(default=0, ge=0)
    pose_present: bool = False
    world_present: bool = False
    quality_accepted: bool = False


class IdentityDecision(ContractModel):
    contract: Literal["noesis.identity.decision"]
    contract_version: Literal[1]
    decision_id: str = Field(min_length=1, max_length=200)
    tracklets: tuple[TrackletRef, ...] = Field(min_length=1)
    outcome: IdentityOutcome
    subject: SubjectRef | None = None
    decided_at_us: TimestampUs
    evidence: IdentityEvidenceSummary
    required_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    absolute_open_set_floor: float | None = Field(default=None, ge=-1.0, le=1.0)
    reject_reason: str | None = Field(default=None, max_length=200)
    prior_changed_winner: bool = False
    overlap_permit: bool = False

    @model_validator(mode="after")
    def _outcome_is_coherent(self) -> "IdentityDecision":
        if self.outcome == IdentityOutcome.UNKNOWN:
            if self.subject is not None:
                raise ValueError("unknown decisions must not expose a subject")
            if not self.reject_reason:
                raise ValueError("unknown decisions require reject_reason")
            if self.prior_changed_winner:
                raise ValueError("a resident prior cannot turn rejection into acceptance")
            return self
        if self.subject is None:
            raise ValueError("accepted decisions require a subject")
        if self.outcome.value != self.subject.kind.value:
            raise ValueError("decision outcome and subject kind disagree")
        if self.absolute_open_set_floor is None or self.evidence.appearance_similarity is None:
            raise ValueError("accepted decisions require similarity and absolute open-set floor")
        if self.evidence.appearance_similarity < self.absolute_open_set_floor:
            raise ValueError("accepted decision is below the absolute open-set floor")
        return self
