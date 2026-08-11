from __future__ import annotations

from itertools import permutations
import time

import pytest

from reid.identity_v2 import (
    BatchIdentityResolver,
    CandidateEvidence,
    HardConstraint,
    IdentityKind,
    OpenSetPolicy,
    OverlapSharePermit,
    ResolverCapacityError,
    TrackletObservation,
)


def resident(identity_id: str, raw: float) -> CandidateEvidence:
    return CandidateEvidence(identity_id, IdentityKind.RESIDENT, raw)


def visitor(identity_id: str, raw: float) -> CandidateEvidence:
    return CandidateEvidence(identity_id, IdentityKind.VISITOR, raw)


def track(tracklet_id: str, *rows: CandidateEvidence) -> TrackletObservation:
    return TrackletObservation(tracklet_id, quality=0.95, candidates=tuple(rows))


def by_track(decisions):
    return {
        decision.tracklet_id: (
            decision.identity_id,
            decision.is_unknown,
            decision.reason,
            decision.raw_similarity,
            decision.prior_contribution,
        )
        for decision in decisions
    }


def test_similar_residents_are_assigned_jointly_one_to_one() -> None:
    resolver = BatchIdentityResolver(
        policy=OpenSetPolicy(resident_prior_bonus=0.02, resident_prior_cap=0.05)
    )
    decisions = resolver.resolve(
        (
            track(
                "track-a", resident("resident-a", 0.90), resident("resident-b", 0.78)
            ),
            track(
                "track-b", resident("resident-a", 0.84), resident("resident-b", 0.89)
            ),
        )
    )
    assigned = {decision.tracklet_id: decision.identity_id for decision in decisions}
    assert assigned == {"track-a": "resident-a", "track-b": "resident-b"}
    assert len(set(assigned.values())) == 2


def test_one_to_one_conflict_uses_explicit_unknown_for_loser() -> None:
    resolver = BatchIdentityResolver()
    decisions = resolver.resolve(
        (
            track("track-a", resident("resident", 0.92)),
            track("track-b", resident("resident", 0.85)),
        )
    )
    rows = by_track(decisions)
    assert rows["track-a"][0] == "resident"
    assert rows["track-b"][0] is None
    assert rows["track-b"][1] is True
    assert rows["track-b"][2] == "assignment_conflict"


def test_explicit_overlap_permit_allows_identity_sharing() -> None:
    observations = (
        track("camera-a:7", resident("resident", 0.92)),
        track("camera-b:9", resident("resident", 0.88)),
    )
    decisions = BatchIdentityResolver().resolve(
        observations,
        overlap_permits=(OverlapSharePermit("resident", "camera-b:9", "camera-a:7"),),
    )
    assert [decision.identity_id for decision in decisions] == ["resident", "resident"]
    assert all(not decision.is_unknown for decision in decisions)
    assert all("overlap_share_permitted" in decision.evidence for decision in decisions)


def test_overlap_permit_is_identity_specific() -> None:
    observations = (
        track("camera-a", resident("resident-a", 0.90)),
        track("camera-b", resident("resident-a", 0.89)),
    )
    decisions = BatchIdentityResolver().resolve(
        observations,
        overlap_permits=(OverlapSharePermit("resident-b", "camera-a", "camera-b"),),
    )
    assert sum(not decision.is_unknown for decision in decisions) == 1


def test_hard_constraint_mask_is_applied_before_scoring_and_assignment() -> None:
    observations = (
        track(
            "track",
            resident("resident-a", 0.94),
            resident("resident-b", 0.82),
        ),
    )
    decisions = BatchIdentityResolver().resolve(
        observations,
        constraints=(HardConstraint("track", "resident-a", False, "active_elsewhere"),),
    )
    decision = decisions[0]
    assert decision.identity_id == "resident-b"
    assert decision.raw_similarity == pytest.approx(0.82)
    assert decision.is_unknown is False
    assert "rejected_candidate=resident-a:active_elsewhere" in decision.evidence


def test_all_hard_constrained_candidates_resolve_unknown() -> None:
    observations = (track("track", resident("resident", 0.99)),)
    decision = BatchIdentityResolver().resolve(
        observations,
        constraints=(HardConstraint("track", "resident", False, "exclusive_claim"),),
    )[0]
    assert decision.is_unknown
    assert decision.reason == "exclusive_claim"
    assert decision.prior_contribution == 0.0


def test_batch_result_is_invariant_to_observation_candidate_and_permit_order() -> None:
    policy = OpenSetPolicy(
        ambiguity_margin_floor=0.01,
        resident_prior_bonus=0.04,
        resident_prior_cap=0.05,
    )
    original = (
        track("z-track", visitor("visitor", 0.84), resident("resident-a", 0.83)),
        track("a-track", resident("resident-a", 0.91), resident("resident-b", 0.81)),
        track("m-track", resident("resident-b", 0.90), visitor("visitor", 0.76)),
    )
    constraints = (
        HardConstraint("m-track", "visitor", False, "topology_blocked"),
        HardConstraint("z-track", "visitor", True, "topology_allowed"),
    )
    permits = (
        OverlapSharePermit("resident-a", "z-track", "a-track"),
        OverlapSharePermit("resident-b", "a-track", "m-track"),
    )
    expected = None
    resolver = BatchIdentityResolver(policy=policy)
    for order in permutations(original):
        reversed_candidates = tuple(
            TrackletObservation(
                item.tracklet_id,
                item.quality,
                tuple(reversed(item.candidates)),
            )
            for item in order
        )
        current = by_track(
            resolver.resolve(
                reversed_candidates,
                constraints=tuple(reversed(constraints)),
                overlap_permits=tuple(reversed(permits)),
            )
        )
        if expected is None:
            expected = current
        assert current == expected


def test_equal_score_tie_break_is_deterministic() -> None:
    observations = (
        track("track-b", resident("resident", 0.90)),
        track("track-a", resident("resident", 0.90)),
    )
    decisions = by_track(BatchIdentityResolver().resolve(observations))
    assert decisions["track-a"][0] == "resident"
    assert decisions["track-b"][0] is None


def test_invalid_constraint_and_overlap_references_fail_loudly() -> None:
    observations = (track("track", resident("resident", 0.90)),)
    resolver = BatchIdentityResolver()
    with pytest.raises(ValueError, match="unknown candidate edges"):
        resolver.resolve(
            observations,
            constraints=(HardConstraint("track", "missing", False),),
        )
    with pytest.raises(ValueError, match="outside batch"):
        resolver.resolve(
            observations,
            overlap_permits=(OverlapSharePermit("resident", "track", "missing"),),
        )


def test_hungarian_adversarial_32_track_64_identity_gallery_latency() -> None:
    observations = tuple(
        track(
            f"track-{track_index:03d}",
            *tuple(
                resident(
                    f"resident-{identity_index:03d}",
                    (
                        0.96
                        if identity_index == track_index
                        else 0.74 - ((track_index + identity_index) % 5) * 0.001
                    ),
                )
                for identity_index in reversed(range(64))
            ),
        )
        for track_index in range(32)
    )
    resolver = BatchIdentityResolver()
    expected = {f"track-{index:03d}": f"resident-{index:03d}" for index in range(32)}
    assert {
        row.tracklet_id: row.identity_id
        for row in resolver.resolve(tuple(reversed(observations)))
    } == expected

    samples_ms = []
    for _ in range(25):
        started = time.perf_counter()
        decisions = resolver.resolve(observations)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
        assert len(decisions) == 32
    samples_ms.sort()
    p95_ms = samples_ms[int(0.95 * (len(samples_ms) - 1))]
    assert p95_ms < 100.0, f"Hungarian 32x64 p95 was {p95_ms:.2f} ms"


def test_assignment_capacity_bounds_fail_without_candidate_truncation() -> None:
    with pytest.raises(ResolverCapacityError, match="tracklets"):
        BatchIdentityResolver(max_tracklets=2).resolve(
            (track("a"), track("b"), track("c"))
        )
    with pytest.raises(ResolverCapacityError, match="identities"):
        BatchIdentityResolver(max_identities=2).resolve(
            (
                track(
                    "track",
                    resident("resident-a", 0.95),
                    resident("resident-b", 0.80),
                    resident("resident-c", 0.70),
                ),
            )
        )
