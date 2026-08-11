"""Runtime-neutral contracts for household identity scoring and assignment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class IdentityKind(str, Enum):
    RESIDENT = "resident"
    VISITOR = "visitor"


@dataclass(frozen=True)
class CandidateEvidence:
    """Appearance evidence for assigning one identity to one tracklet."""

    identity_id: str
    identity_kind: IdentityKind
    raw_similarity: float
    hard_allowed: bool = True
    hard_constraint_reason: Optional[str] = None
    evidence: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        identity_id = str(self.identity_id).strip()
        if not identity_id:
            raise ValueError("identity_id must be non-empty")
        object.__setattr__(self, "identity_id", identity_id)
        try:
            kind = IdentityKind(self.identity_kind)
        except Exception as exc:
            raise ValueError(
                f"unsupported identity kind: {self.identity_kind!r}"
            ) from exc
        object.__setattr__(self, "identity_kind", kind)
        similarity = float(self.raw_similarity)
        if not math.isfinite(similarity) or similarity < -1.0 or similarity > 1.0:
            raise ValueError("raw_similarity must be finite and within [-1, 1]")
        object.__setattr__(self, "raw_similarity", similarity)
        object.__setattr__(self, "evidence", tuple(str(item) for item in self.evidence))


@dataclass(frozen=True)
class TrackletObservation:
    """Quality and candidate evidence for a single tracker-local person."""

    tracklet_id: str
    quality: float
    candidates: Tuple[CandidateEvidence, ...] = field(default_factory=tuple)
    evidence: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        tracklet_id = str(self.tracklet_id).strip()
        if not tracklet_id:
            raise ValueError("tracklet_id must be non-empty")
        object.__setattr__(self, "tracklet_id", tracklet_id)
        quality = float(self.quality)
        if not math.isfinite(quality) or quality < 0.0 or quality > 1.0:
            raise ValueError("quality must be finite and within [0, 1]")
        object.__setattr__(self, "quality", quality)
        candidates = tuple(self.candidates)
        ids = [candidate.identity_id for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError(
                f"duplicate identity candidate for tracklet {tracklet_id!r}"
            )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "evidence", tuple(str(item) for item in self.evidence))


@dataclass(frozen=True)
class HardConstraint:
    """A resolver-level allow/deny override for one tracklet/identity edge."""

    tracklet_id: str
    identity_id: str
    allowed: bool
    reason: str = "hard_constraint"

    def __post_init__(self) -> None:
        tracklet_id = str(self.tracklet_id).strip()
        identity_id = str(self.identity_id).strip()
        if not tracklet_id or not identity_id:
            raise ValueError("constraint tracklet_id and identity_id must be non-empty")
        object.__setattr__(self, "tracklet_id", tracklet_id)
        object.__setattr__(self, "identity_id", identity_id)
        object.__setattr__(self, "reason", str(self.reason or "hard_constraint"))


@dataclass(frozen=True)
class OverlapSharePermit:
    """Permit exactly two tracklets to share one identity concurrently."""

    identity_id: str
    tracklet_a: str
    tracklet_b: str
    reason: str = "overlap_permit"

    def __post_init__(self) -> None:
        identity_id = str(self.identity_id).strip()
        tracklet_a = str(self.tracklet_a).strip()
        tracklet_b = str(self.tracklet_b).strip()
        if not identity_id or not tracklet_a or not tracklet_b:
            raise ValueError("overlap permit fields must be non-empty")
        if tracklet_a == tracklet_b:
            raise ValueError("overlap permit requires two distinct tracklets")
        if tracklet_b < tracklet_a:
            tracklet_a, tracklet_b = tracklet_b, tracklet_a
        object.__setattr__(self, "identity_id", identity_id)
        object.__setattr__(self, "tracklet_a", tracklet_a)
        object.__setattr__(self, "tracklet_b", tracklet_b)
        object.__setattr__(self, "reason", str(self.reason or "overlap_permit"))

    @property
    def pair(self) -> Tuple[str, str]:
        return self.tracklet_a, self.tracklet_b


@dataclass(frozen=True)
class ScoredCandidate:
    tracklet_id: str
    identity_id: str
    identity_kind: IdentityKind
    eligible: bool
    raw_similarity: float
    quality: float
    calibrated_confidence: float
    base_utility: float
    adjusted_utility: float
    prior_contribution: float
    reason: str
    evidence: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TrackletScores:
    observation: TrackletObservation
    candidates: Tuple[ScoredCandidate, ...]
    assignment_eligible: bool
    reason: str
    pre_prior_winner_id: Optional[str]
    pre_prior_margin: Optional[float]


@dataclass(frozen=True)
class IdentityDecision:
    tracklet_id: str
    identity_id: Optional[str]
    identity_kind: Optional[IdentityKind]
    is_unknown: bool
    raw_similarity: Optional[float]
    calibrated_confidence: float
    ambiguity_margin: Optional[float]
    prior_contribution: float
    assignment_utility: float
    reason: str
    evidence: Tuple[str, ...] = field(default_factory=tuple)
