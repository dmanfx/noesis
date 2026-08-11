from __future__ import annotations

import pytest

from reid.identity_v2 import (
    BatchIdentityResolver,
    CandidateEvidence,
    IdentityKind,
    OpenSetPolicy,
    OpenSetScorer,
    TrackletObservation,
)


def candidate(
    identity_id: str,
    raw: float,
    *,
    kind: IdentityKind = IdentityKind.RESIDENT,
    allowed: bool = True,
) -> CandidateEvidence:
    return CandidateEvidence(
        identity_id=identity_id,
        identity_kind=kind,
        raw_similarity=raw,
        hard_allowed=allowed,
        hard_constraint_reason="geometry_blocked" if not allowed else None,
        evidence=(f"raw={raw:.3f}",),
    )


def observation(
    tracklet_id: str,
    *candidates: CandidateEvidence,
    quality: float = 0.9,
) -> TrackletObservation:
    return TrackletObservation(
        tracklet_id=tracklet_id,
        quality=quality,
        candidates=tuple(candidates),
        evidence=("quality_measured",),
    )


def test_all_candidates_below_floor_resolve_unknown() -> None:
    resolver = BatchIdentityResolver()
    decisions = resolver.resolve(
        (
            observation("track-a", candidate("resident-a", 0.50)),
            observation(
                "track-b",
                candidate("resident-b", 0.65),
                candidate("visitor-a", 0.40, kind=IdentityKind.VISITOR),
            ),
        )
    )
    assert all(decision.is_unknown for decision in decisions)
    assert {decision.reason for decision in decisions} == {"appearance_below_floor"}
    assert all(decision.prior_contribution == 0.0 for decision in decisions)


def test_tracklet_without_candidates_has_its_own_unknown_option() -> None:
    decision = BatchIdentityResolver().resolve((observation("track"),))[0]
    assert decision.is_unknown
    assert decision.identity_id is None
    assert decision.reason == "no_candidates"
    assert decision.calibrated_confidence == pytest.approx(1.0)


def test_low_quality_gate_happens_before_resident_prior() -> None:
    scorer = OpenSetScorer()
    obs = observation("track", candidate("resident", 0.99), quality=0.20)
    scored = scorer.score_tracklet(obs)
    row = scored.candidates[0]
    assert row.eligible is False
    assert row.reason == "quality_below_floor"
    assert row.prior_contribution == 0.0
    decision = BatchIdentityResolver(scorer).resolve((obs,))[0]
    assert decision.is_unknown
    assert decision.reason == "quality_below_floor"


def test_below_floor_resident_cannot_be_rescued_by_large_allowed_prior() -> None:
    policy = OpenSetPolicy(
        appearance_floor=0.70,
        resident_prior_bonus=0.05,
        resident_prior_cap=0.05,
    )
    scorer = OpenSetScorer(policy)
    obs = observation("track", candidate("resident", 0.699))
    row = scorer.score_tracklet(obs).candidates[0]
    assert row.eligible is False
    assert row.prior_contribution == 0.0
    decision = BatchIdentityResolver(scorer).resolve((obs,))[0]
    assert decision.is_unknown
    assert decision.reason == "appearance_below_floor"


def test_resident_prior_reranks_only_already_acceptable_candidate() -> None:
    policy = OpenSetPolicy(
        ambiguity_margin_floor=0.02,
        resident_prior_bonus=0.05,
        resident_prior_cap=0.05,
    )
    scorer = OpenSetScorer(policy)
    obs = observation(
        "track",
        candidate("resident", 0.80),
        candidate("visitor", 0.82, kind=IdentityKind.VISITOR),
    )
    scores = scorer.score_tracklet(obs)
    assert scores.assignment_eligible
    assert scores.pre_prior_winner_id == "visitor"
    resident_score = next(
        row for row in scores.candidates if row.identity_id == "resident"
    )
    assert resident_score.eligible
    assert resident_score.prior_contribution == pytest.approx(0.05)

    decision = BatchIdentityResolver(scorer).resolve((obs,))[0]
    assert decision.identity_id == "resident"
    assert decision.reason == "resident_prior_rerank"
    assert decision.prior_contribution == pytest.approx(0.05)
    assert decision.raw_similarity == pytest.approx(0.80)
    assert 0.0 < decision.calibrated_confidence < 1.0
    assert decision.ambiguity_margin is not None
    assert "bounded_resident_prior_changed_winner" in decision.evidence


def test_prior_cannot_turn_ambiguous_rejection_into_acceptance() -> None:
    policy = OpenSetPolicy(
        ambiguity_margin_floor=0.02,
        resident_prior_bonus=0.05,
        resident_prior_cap=0.05,
    )
    obs = observation(
        "track",
        candidate("resident", 0.800),
        candidate("visitor", 0.799, kind=IdentityKind.VISITOR),
    )
    scores = OpenSetScorer(policy).score_tracklet(obs)
    assert scores.assignment_eligible is False
    assert scores.reason == "ambiguous"
    decision = BatchIdentityResolver(policy=policy).resolve((obs,))[0]
    assert decision.is_unknown
    assert decision.reason == "ambiguous"
    assert decision.prior_contribution == 0.0
    assert decision.ambiguity_margin is not None
    assert decision.ambiguity_margin < policy.ambiguity_margin_floor


def test_resident_prior_is_strictly_bounded_by_policy() -> None:
    with pytest.raises(ValueError, match="resident_prior_bonus"):
        OpenSetPolicy(resident_prior_bonus=0.20, resident_prior_cap=0.05)


def test_default_logistic_mapping_is_explicitly_uncalibrated() -> None:
    default_policy = OpenSetPolicy()
    assert default_policy.calibration_status == "uncalibrated_default"
    decision = BatchIdentityResolver(policy=default_policy).resolve(
        (observation("track", candidate("resident", 0.95)),)
    )[0]
    assert "calibration_status=uncalibrated_default" in decision.evidence

    artifact_policy = OpenSetPolicy(calibration_artifact_id="eval-2026-07-10-sha256")
    assert artifact_policy.calibration_status == "artifact_backed"
    with pytest.raises(ValueError, match="calibration_artifact_id"):
        OpenSetPolicy(calibration_artifact_id="  ")
