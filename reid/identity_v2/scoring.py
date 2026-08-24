"""Resident-biased open-set scoring with prior-independent rejection gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

from .models import (
    CandidateEvidence,
    IdentityKind,
    ScoredCandidate,
    TrackletObservation,
    TrackletScores,
)


# BLAS-backed gallery scoring and the former scalar reduction can differ by a
# few ULPs.  Keep policy decisions stable at an exact configured boundary
# without making the tolerance large enough to hide a meaningful score gap.
_POLICY_COMPARISON_EPSILON = 1e-12


def _below_policy_floor(value: float, floor: float) -> bool:
    return float(value) < float(floor) - _POLICY_COMPARISON_EPSILON


@dataclass(frozen=True)
class OpenSetPolicy:
    """Open-set scoring and conservative household-prior policy.

    ``calibration_slope`` and ``calibration_midpoint`` describe a logistic
    mapping. The numeric defaults are heuristics, not a probability calibration;
    they remain explicitly ``uncalibrated_default`` until an external validation
    artifact ID is supplied. The resident bonus is assignment utility, not
    confidence, and is bounded independently.
    """

    appearance_floor: float = 0.70
    quality_floor: float = 0.50
    calibrated_confidence_floor: float = 0.55
    ambiguity_margin_floor: float = 0.025
    calibration_slope: float = 12.0
    calibration_midpoint: float = 0.70
    resident_prior_bonus: float = 0.025
    resident_prior_cap: float = 0.05
    unknown_utility: float = 0.0
    calibration_artifact_id: Optional[str] = None
    model_semantic_profile_sha256: Optional[str] = None
    maximum_resident_candidates: Optional[int] = None
    maximum_visitor_candidates: Optional[int] = None
    maximum_total_candidates: Optional[int] = None
    maximum_exemplars_per_candidate: Optional[int] = None

    def __post_init__(self) -> None:
        for name in (
            "appearance_floor",
            "quality_floor",
            "calibrated_confidence_floor",
            "ambiguity_margin_floor",
            "calibration_slope",
            "calibration_midpoint",
            "resident_prior_bonus",
            "resident_prior_cap",
            "unknown_utility",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not -1.0 <= self.appearance_floor <= 1.0:
            raise ValueError("appearance_floor must be within [-1, 1]")
        if not 0.0 <= self.quality_floor <= 1.0:
            raise ValueError("quality_floor must be within [0, 1]")
        if not 0.0 <= self.calibrated_confidence_floor <= 1.0:
            raise ValueError("calibrated_confidence_floor must be within [0, 1]")
        if self.ambiguity_margin_floor < 0.0:
            raise ValueError("ambiguity_margin_floor must be non-negative")
        if self.calibration_slope <= 0.0:
            raise ValueError("calibration_slope must be positive")
        if self.resident_prior_cap < 0.0:
            raise ValueError("resident_prior_cap must be non-negative")
        if not 0.0 <= self.resident_prior_bonus <= self.resident_prior_cap:
            raise ValueError(
                "resident_prior_bonus must be within [0, resident_prior_cap]"
            )
        if not 0.0 <= self.unknown_utility <= 1.0:
            raise ValueError("unknown_utility must be within [0, 1]")
        artifact_id = self.calibration_artifact_id
        if artifact_id is not None:
            artifact_id = str(artifact_id).strip()
            if not artifact_id:
                raise ValueError("calibration_artifact_id must be non-empty")
        object.__setattr__(self, "calibration_artifact_id", artifact_id)
        semantic_profile = self.model_semantic_profile_sha256
        if semantic_profile is not None:
            semantic_profile = str(semantic_profile).strip().lower()
            if len(semantic_profile) != 64 or any(
                value not in "0123456789abcdef" for value in semantic_profile
            ):
                raise ValueError(
                    "model_semantic_profile_sha256 must be lowercase SHA-256"
                )
        object.__setattr__(self, "model_semantic_profile_sha256", semantic_profile)
        for name in (
            "maximum_resident_candidates",
            "maximum_visitor_candidates",
            "maximum_total_candidates",
            "maximum_exemplars_per_candidate",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            normalized = int(value)
            if normalized < (0 if name == "maximum_visitor_candidates" else 1):
                raise ValueError(f"{name} is below its minimum")
            object.__setattr__(self, name, normalized)

    @property
    def calibration_status(self) -> str:
        return (
            "artifact_backed"
            if self.calibration_artifact_id is not None
            else "uncalibrated_default"
        )


class OpenSetScorer:
    """Score candidates while keeping every rejection gate prior-independent."""

    def __init__(self, policy: Optional[OpenSetPolicy] = None) -> None:
        self.policy = policy or OpenSetPolicy()

    def calibrate(self, raw_similarity: float) -> float:
        z = self.policy.calibration_slope * (
            float(raw_similarity) - self.policy.calibration_midpoint
        )
        if z >= 0.0:
            return float(1.0 / (1.0 + math.exp(-z)))
        exp_z = math.exp(z)
        return float(exp_z / (1.0 + exp_z))

    def score_candidate(
        self,
        observation: TrackletObservation,
        candidate: CandidateEvidence,
    ) -> ScoredCandidate:
        raw = float(candidate.raw_similarity)
        quality = float(observation.quality)
        confidence = self.calibrate(raw)
        eligible = False
        reason = "eligible"

        # Gate order is deliberate. No prior is computed until every absolute
        # open-set and quality requirement has passed.
        if quality < self.policy.quality_floor:
            reason = "quality_below_floor"
        elif _below_policy_floor(raw, self.policy.appearance_floor):
            reason = "appearance_below_floor"
        elif not bool(candidate.hard_allowed):
            reason = str(candidate.hard_constraint_reason or "hard_constraint")
        elif _below_policy_floor(
            confidence,
            self.policy.calibrated_confidence_floor,
        ):
            reason = "confidence_below_floor"
        else:
            eligible = True

        prior = 0.0
        if eligible and candidate.identity_kind is IdentityKind.RESIDENT:
            prior = min(
                self.policy.resident_prior_bonus,
                self.policy.resident_prior_cap,
            )
        base_utility = confidence if eligible else float("-inf")
        adjusted_utility = base_utility + prior if eligible else float("-inf")
        evidence = (
            tuple(observation.evidence)
            + tuple(candidate.evidence)
            + (
                f"quality={quality:.6f}",
                f"raw_similarity={raw:.6f}",
                f"calibrated_confidence={confidence:.6f}",
                f"calibration_status={self.policy.calibration_status}",
                f"candidate_reason={reason}",
                f"resident_prior={prior:.6f}",
            )
        )
        return ScoredCandidate(
            tracklet_id=observation.tracklet_id,
            identity_id=candidate.identity_id,
            identity_kind=candidate.identity_kind,
            eligible=eligible,
            raw_similarity=raw,
            quality=quality,
            calibrated_confidence=confidence,
            base_utility=base_utility,
            adjusted_utility=adjusted_utility,
            prior_contribution=prior,
            reason=reason,
            evidence=evidence,
        )

    def score_tracklet(self, observation: TrackletObservation) -> TrackletScores:
        scored = tuple(
            self.score_candidate(observation, candidate)
            for candidate in sorted(
                observation.candidates, key=lambda item: item.identity_id
            )
        )
        eligible = sorted(
            (candidate for candidate in scored if candidate.eligible),
            key=lambda item: (
                -item.base_utility,
                -item.raw_similarity,
                item.identity_id,
            ),
        )

        if not eligible:
            reason = self._unknown_reason(scored)
            return TrackletScores(
                observation=observation,
                candidates=scored,
                assignment_eligible=False,
                reason=reason,
                pre_prior_winner_id=None,
                pre_prior_margin=None,
            )

        best = eligible[0]
        runner_utility = self.policy.unknown_utility
        if len(eligible) > 1:
            runner_utility = max(runner_utility, eligible[1].base_utility)
        margin = float(best.base_utility - runner_utility)
        if best.base_utility <= self.policy.unknown_utility:
            return TrackletScores(
                observation=observation,
                candidates=scored,
                assignment_eligible=False,
                reason="unknown_preferred",
                pre_prior_winner_id=best.identity_id,
                pre_prior_margin=margin,
            )
        if _below_policy_floor(margin, self.policy.ambiguity_margin_floor):
            return TrackletScores(
                observation=observation,
                candidates=scored,
                assignment_eligible=False,
                reason="ambiguous",
                pre_prior_winner_id=best.identity_id,
                pre_prior_margin=margin,
            )
        return TrackletScores(
            observation=observation,
            candidates=scored,
            assignment_eligible=True,
            reason="eligible",
            pre_prior_winner_id=best.identity_id,
            pre_prior_margin=margin,
        )

    @staticmethod
    def _unknown_reason(candidates: Iterable[ScoredCandidate]) -> str:
        rows = tuple(candidates)
        if not rows:
            return "no_candidates"
        best = max(rows, key=lambda item: (item.raw_similarity, item.identity_id))
        return str(best.reason or "no_eligible_candidates")
