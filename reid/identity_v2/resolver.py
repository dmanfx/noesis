"""Deterministic bounded batch assignment for household identities."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .models import (
    CandidateEvidence,
    HardConstraint,
    IdentityDecision,
    OverlapSharePermit,
    ScoredCandidate,
    TrackletObservation,
    TrackletScores,
)
from .scoring import OpenSetPolicy, OpenSetScorer

_EPSILON = 1e-12
_FORBIDDEN_UTILITY = -1.0e9


class ResolverCapacityError(RuntimeError):
    """The configured assignment surface was exceeded and was not truncated."""


class BatchIdentityResolver:
    """Resolve simultaneous tracklets jointly with explicit unknown options.

    The normal one-to-one path is a rectangular Hungarian assignment with an
    independent unknown column for every tracklet. Explicit overlap permits use
    the exact permit-aware solver because sharing is not a bipartite matching.
    Both paths fail loudly at configured capacity bounds; candidate evidence is
    never silently truncated.
    """

    def __init__(
        self,
        scorer: Optional[OpenSetScorer] = None,
        *,
        policy: Optional[OpenSetPolicy] = None,
        max_tracklets: int = 64,
        max_identities: int = 128,
        max_permit_solver_tracklets: int = 8,
        max_permit_solver_identities: int = 16,
    ) -> None:
        if scorer is not None and policy is not None:
            raise ValueError("provide scorer or policy, not both")
        self.scorer = scorer or OpenSetScorer(policy)
        self.max_tracklets = self._positive_bound(max_tracklets, "max_tracklets")
        self.max_identities = self._positive_bound(max_identities, "max_identities")
        self.max_permit_solver_tracklets = self._positive_bound(
            max_permit_solver_tracklets,
            "max_permit_solver_tracklets",
        )
        self.max_permit_solver_identities = self._positive_bound(
            max_permit_solver_identities,
            "max_permit_solver_identities",
        )

    @staticmethod
    def _positive_bound(value: int, name: str) -> int:
        bound = int(value)
        if bound < 1:
            raise ValueError(f"{name} must be positive")
        return bound

    def resolve(
        self,
        observations: Sequence[TrackletObservation],
        *,
        constraints: Sequence[HardConstraint] = (),
        overlap_permits: Sequence[OverlapSharePermit] = (),
    ) -> Tuple[IdentityDecision, ...]:
        normalized = self.apply_constraints(observations, constraints)
        ordered = tuple(sorted(normalized, key=lambda item: item.tracklet_id))
        track_ids = [item.tracklet_id for item in ordered]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("tracklet_id values must be unique within a batch")
        if len(track_ids) > self.max_tracklets:
            raise ResolverCapacityError(
                f"batch has {len(track_ids)} tracklets; configured maximum is "
                f"{self.max_tracklets}"
            )
        identity_ids = sorted(
            {
                candidate.identity_id
                for observation in ordered
                for candidate in observation.candidates
            }
        )
        if len(identity_ids) > self.max_identities:
            raise ResolverCapacityError(
                f"batch exposes {len(identity_ids)} identities; configured maximum is "
                f"{self.max_identities}"
            )
        known_tracks = set(track_ids)
        permit_pairs = self._permit_pairs(overlap_permits, known_tracks)
        score_rows = tuple(self.scorer.score_tracklet(item) for item in ordered)

        options: Dict[str, Tuple[Optional[ScoredCandidate], ...]] = {}
        for row in score_rows:
            candidates: List[Optional[ScoredCandidate]] = []
            if row.assignment_eligible:
                candidates.extend(
                    sorted(
                        (
                            candidate
                            for candidate in row.candidates
                            if candidate.eligible
                        ),
                        key=lambda item: (
                            -item.adjusted_utility,
                            item.identity_id,
                        ),
                    )
                )
            # Every tracklet owns an explicit, independent unknown option.
            candidates.append(None)
            options[row.observation.tracklet_id] = tuple(candidates)

        eligible_identity_ids = {
            candidate.identity_id
            for candidates in options.values()
            for candidate in candidates
            if candidate is not None
        }
        effective_permits = {
            permit for permit in permit_pairs if permit[0] in eligible_identity_ids
        }
        if effective_permits:
            best_assignment = self._resolve_with_permits(
                score_rows,
                options,
                effective_permits,
                eligible_identity_ids,
            )
        else:
            best_assignment = self._resolve_one_to_one(score_rows, options)

        selected_identity_counts: Dict[str, int] = {}
        for selected_candidate in best_assignment.values():
            if selected_candidate is not None:
                selected_identity_counts[selected_candidate.identity_id] = (
                    selected_identity_counts.get(selected_candidate.identity_id, 0) + 1
                )
        return tuple(
            self._decision_for(
                row,
                best_assignment.get(row.observation.tracklet_id),
                overlap_shared=(
                    best_assignment.get(row.observation.tracklet_id) is not None
                    and selected_identity_counts.get(
                        best_assignment[row.observation.tracklet_id].identity_id, 0
                    )
                    > 1
                ),
            )
            for row in score_rows
        )

    def _resolve_one_to_one(
        self,
        score_rows: Sequence[TrackletScores],
        options: Dict[str, Tuple[Optional[ScoredCandidate], ...]],
    ) -> Dict[str, Optional[ScoredCandidate]]:
        """Solve the common no-sharing case in polynomial time."""

        if not score_rows:
            return {}
        identity_ids = sorted(
            {
                candidate.identity_id
                for candidates in options.values()
                for candidate in candidates
                if candidate is not None
            }
        )
        identity_column = {
            identity_id: index for index, identity_id in enumerate(identity_ids)
        }
        track_count = len(score_rows)
        identity_count = len(identity_ids)
        column_count = identity_count + track_count
        utilities = np.full(
            (track_count, column_count),
            _FORBIDDEN_UTILITY,
            dtype=np.float64,
        )
        candidate_by_edge: Dict[Tuple[int, int], ScoredCandidate] = {}
        for row_index, row in enumerate(score_rows):
            tracklet_id = row.observation.tracklet_id
            for candidate in options[tracklet_id]:
                if candidate is None:
                    continue
                column = identity_column[candidate.identity_id]
                utilities[row_index, column] = candidate.adjusted_utility
                candidate_by_edge[(row_index, column)] = candidate
            # A tracklet may only consume its own unknown dummy. The rectangular
            # assignment therefore cannot make unknown capacity a shared global
            # resource.
            utilities[row_index, identity_count + row_index] = (
                self.scorer.policy.unknown_utility
            )

        # Inputs and columns are sorted. A bounded sub-epsilon secondary term
        # gives SciPy stable ordering on exactly tied utility surfaces without
        # changing any comparison the resolver otherwise treats as non-tied.
        tie_budget = _EPSILON / 8.0
        denominator = max(1, track_count * max(1, identity_count))
        for row_index in range(track_count):
            for column in np.flatnonzero(utilities[row_index] > _FORBIDDEN_UTILITY):
                row_priority = track_count - row_index
                # All private unknown columns represent the same lexical token;
                # their physical matrix indices must not influence the tie.
                token_priority = (
                    identity_count - int(column) if int(column) < identity_count else 0
                )
                utilities[row_index, column] += tie_budget * (
                    row_priority * token_priority / denominator
                )

        row_indices, selected_columns = linear_sum_assignment(
            utilities,
            maximize=True,
        )
        if len(row_indices) != track_count:
            raise RuntimeError("Hungarian assignment did not cover every tracklet")
        assignment: Dict[str, Optional[ScoredCandidate]] = {}
        for row_index, column in zip(row_indices.tolist(), selected_columns.tolist()):
            tracklet_id = score_rows[row_index].observation.tracklet_id
            if utilities[row_index, column] <= _FORBIDDEN_UTILITY:
                raise RuntimeError(
                    f"identity assignment has no option for {tracklet_id!r}"
                )
            assignment[tracklet_id] = candidate_by_edge.get((row_index, column))
        return assignment

    def _resolve_with_permits(
        self,
        score_rows: Sequence[TrackletScores],
        options: Dict[str, Tuple[Optional[ScoredCandidate], ...]],
        permit_pairs: set[Tuple[str, str, str]],
        eligible_identity_ids: set[str],
    ) -> Dict[str, Optional[ScoredCandidate]]:
        """Use the bounded exact solver only when sharing is explicitly enabled."""

        if len(score_rows) > self.max_permit_solver_tracklets:
            raise ResolverCapacityError(
                "explicit overlap assignment has "
                f"{len(score_rows)} tracklets; exact permit-aware maximum is "
                f"{self.max_permit_solver_tracklets}"
            )
        if len(eligible_identity_ids) > self.max_permit_solver_identities:
            raise ResolverCapacityError(
                "explicit overlap assignment exposes "
                f"{len(eligible_identity_ids)} eligible identities; exact "
                "permit-aware maximum is "
                f"{self.max_permit_solver_identities}"
            )
        permit_identity_ids = {identity_id for identity_id, _a, _b in permit_pairs}

        # Dynamic programming keeps the exact solver practical when several
        # tracklets have the same resident shortlist. The state records only
        # existing identity claims; without overlap permits this reduces to the
        # familiar identity-used bitset shape, while explicit sharing retains
        # the track IDs needed to verify each permitted pair.
        ClaimState = Tuple[Tuple[str, Tuple[str, ...]], ...]

        @lru_cache(maxsize=None)
        def solve(index: int, claim_state: ClaimState) -> Tuple[float, Tuple[str, ...]]:
            if index >= len(score_rows):
                return 0.0, ()
            row = score_rows[index]
            track_id = row.observation.tracklet_id
            claims_by_identity = {
                identity_id: list(tracklets) for identity_id, tracklets in claim_state
            }
            best_total = float("-inf")
            best_tokens: Optional[Tuple[str, ...]] = None

            for candidate in options[track_id]:
                next_claims = claims_by_identity
                token = "~unknown"
                utility = self.scorer.policy.unknown_utility
                if candidate is not None:
                    existing = claims_by_identity.get(candidate.identity_id, [])
                    if existing and not self._sharing_allowed(
                        candidate.identity_id,
                        track_id,
                        existing,
                        permit_pairs,
                    ):
                        continue
                    next_claims = {
                        identity_id: list(tracklets)
                        for identity_id, tracklets in claims_by_identity.items()
                    }
                    if candidate.identity_id in permit_identity_ids:
                        next_claims.setdefault(candidate.identity_id, []).append(
                            track_id
                        )
                    else:
                        # No future track needs to know which track owns a
                        # non-shareable identity; a sentinel collapses equivalent
                        # DP states to the ordinary used-identity bitset.
                        next_claims[candidate.identity_id] = ["*"]
                    token = candidate.identity_id
                    utility = candidate.adjusted_utility

                next_state: ClaimState = tuple(
                    (identity_id, tuple(sorted(tracklets)))
                    for identity_id, tracklets in sorted(next_claims.items())
                    if tracklets
                )
                suffix_total, suffix_tokens = solve(index + 1, next_state)
                total = float(utility + suffix_total)
                tokens = (token,) + suffix_tokens
                if total > best_total + _EPSILON or (
                    abs(total - best_total) <= _EPSILON
                    and (best_tokens is None or tokens < best_tokens)
                ):
                    best_total = total
                    best_tokens = tokens

            if best_tokens is None:
                raise RuntimeError(
                    f"identity assignment has no option for {track_id!r}"
                )
            return best_total, best_tokens

        _best_total, selected_tokens = solve(0, ())
        best_assignment: Dict[str, Optional[ScoredCandidate]] = {}
        for row, token in zip(score_rows, selected_tokens):
            if token == "~unknown":
                best_assignment[row.observation.tracklet_id] = None
                continue
            selected_candidate = next(
                (
                    candidate
                    for candidate in options[row.observation.tracklet_id]
                    if candidate is not None and candidate.identity_id == token
                ),
                None,
            )
            if selected_candidate is None:
                raise RuntimeError(f"selected candidate {token!r} disappeared")
            best_assignment[row.observation.tracklet_id] = selected_candidate
        return best_assignment

    def apply_constraints(
        self,
        observations: Sequence[TrackletObservation],
        constraints: Sequence[HardConstraint],
    ) -> Tuple[TrackletObservation, ...]:
        """Return the exact hard-masked candidate surface used by resolution."""

        overrides: Dict[Tuple[str, str], HardConstraint] = {}
        for constraint in constraints:
            key = (constraint.tracklet_id, constraint.identity_id)
            if key in overrides:
                raise ValueError(f"duplicate hard constraint for {key!r}")
            overrides[key] = constraint

        known_edges = {
            (observation.tracklet_id, candidate.identity_id)
            for observation in observations
            for candidate in observation.candidates
        }
        unknown_edges = sorted(set(overrides) - known_edges)
        if unknown_edges:
            raise ValueError(
                f"hard constraints reference unknown candidate edges: {unknown_edges!r}"
            )

        out: List[TrackletObservation] = []
        for observation in observations:
            candidates: List[CandidateEvidence] = []
            for candidate in observation.candidates:
                override = overrides.get(
                    (observation.tracklet_id, candidate.identity_id)
                )
                if override is None:
                    candidates.append(candidate)
                    continue
                allowed = bool(candidate.hard_allowed and override.allowed)
                reason = candidate.hard_constraint_reason
                if not allowed:
                    reason = override.reason if not override.allowed else reason
                candidates.append(
                    replace(
                        candidate,
                        hard_allowed=allowed,
                        hard_constraint_reason=reason,
                    )
                )
            out.append(replace(observation, candidates=tuple(candidates)))
        return tuple(out)

    @staticmethod
    def _permit_pairs(
        permits: Sequence[OverlapSharePermit],
        known_tracks: set[str],
    ) -> set[Tuple[str, str, str]]:
        out: set[Tuple[str, str, str]] = set()
        for permit in permits:
            if (
                permit.tracklet_a not in known_tracks
                or permit.tracklet_b not in known_tracks
            ):
                raise ValueError(
                    f"overlap permit references tracklet outside batch: {permit.pair!r}"
                )
            out.add((permit.identity_id, permit.tracklet_a, permit.tracklet_b))
        return out

    @staticmethod
    def _sharing_allowed(
        identity_id: str,
        tracklet_id: str,
        existing_claims: Iterable[str],
        permits: set[Tuple[str, str, str]],
    ) -> bool:
        for existing in existing_claims:
            a, b = sorted((str(existing), str(tracklet_id)))
            if (identity_id, a, b) not in permits:
                return False
        return True

    def _decision_for(
        self,
        row: TrackletScores,
        selected: Optional[ScoredCandidate],
        *,
        overlap_shared: bool = False,
    ) -> IdentityDecision:
        if selected is not None:
            reason = "assigned"
            evidence = list(selected.evidence)
            evidence.append(f"pre_prior_winner={row.pre_prior_winner_id}")
            if row.pre_prior_margin is not None:
                evidence.append(f"pre_prior_margin={row.pre_prior_margin:.6f}")
            for rejected in row.candidates:
                if not rejected.eligible:
                    evidence.append(
                        f"rejected_candidate={rejected.identity_id}:{rejected.reason}"
                    )
            if overlap_shared:
                evidence.append("overlap_share_permitted")
            if selected.identity_id != row.pre_prior_winner_id:
                base_winner = next(
                    (
                        candidate
                        for candidate in row.candidates
                        if candidate.identity_id == row.pre_prior_winner_id
                    ),
                    None,
                )
                if (
                    selected.identity_kind.value == "resident"
                    and selected.prior_contribution > 0.0
                    and base_winner is not None
                    and selected.adjusted_utility > base_winner.adjusted_utility
                ):
                    reason = "resident_prior_rerank"
                    evidence.append("bounded_resident_prior_changed_winner")
                else:
                    reason = "joint_assignment"
                    evidence.append("joint_assignment_changed_local_winner")
            return IdentityDecision(
                tracklet_id=row.observation.tracklet_id,
                identity_id=selected.identity_id,
                identity_kind=selected.identity_kind,
                is_unknown=False,
                raw_similarity=selected.raw_similarity,
                calibrated_confidence=selected.calibrated_confidence,
                ambiguity_margin=row.pre_prior_margin,
                prior_contribution=selected.prior_contribution,
                assignment_utility=selected.adjusted_utility,
                reason=reason,
                evidence=tuple(evidence),
            )

        ranked = sorted(
            row.candidates,
            key=lambda item: (-item.raw_similarity, item.identity_id),
        )
        best = ranked[0] if ranked else None
        if not row.assignment_eligible:
            reason = row.reason
        else:
            reason = "assignment_conflict"
        evidence = list(row.observation.evidence)
        if best is not None:
            evidence.extend(best.evidence)
            evidence.append(f"best_candidate_reason={best.reason}")
        evidence.append(f"tracklet_reason={reason}")
        unknown_confidence = 1.0
        if best is not None:
            unknown_confidence = max(0.0, min(1.0, 1.0 - best.calibrated_confidence))
            evidence.append("unknown_confidence=one_minus_best_candidate_confidence")
        return IdentityDecision(
            tracklet_id=row.observation.tracklet_id,
            identity_id=None,
            identity_kind=None,
            is_unknown=True,
            raw_similarity=best.raw_similarity if best is not None else None,
            calibrated_confidence=unknown_confidence,
            ambiguity_margin=row.pre_prior_margin,
            prior_contribution=0.0,
            assignment_utility=self.scorer.policy.unknown_utility,
            reason=reason,
            evidence=tuple(evidence),
        )
