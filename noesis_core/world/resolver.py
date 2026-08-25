"""Universal, camera-agnostic world measurement resolver.

This module adjudicates a bounded set of already-extracted metric hypotheses.
It deliberately does not know about rooms, cameras, models, image buffers, or
runtime state.  The caller supplies exact-cohort evidence and an optional
process continuation; the resolver returns one immutable ground-footprint
measurement with covariance and a compact explanation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from noesis_core.contracts.base import Matrix3, Vector3
from noesis_core.contracts.world_measurement import (
    MeasurementQuality,
    ResolvedGroundMeasurement,
    WorldMeasurementCandidateDiagnostic,
    WorldMeasurementHypothesis,
    WorldMeasurementSet,
    WorldPriorEvidence,
)


_AXES = 3
_MIN_VARIANCE_M2 = 1e-6
_MAX_VARIANCE_M2 = 1e8


@dataclass(frozen=True)
class WorldMeasurementResolverConfig:
    """Bounded numeric policy for the universal resolver.

    These values are source- and site-independent.  Calibration and camera
    differences belong in the candidate covariance/evidence, not here.
    """

    max_hypotheses: int = 4
    agreement_mahalanobis_sq: float = 9.0
    max_compatible_distance_m: float = 1.25
    covariance_regularization_m2: float = 1e-5
    incompatible_covariance_scale: float = 2.0
    disagreement_inflation: float = 0.50
    pcf_boundary_scale_m: float = 0.60
    pcf_floor_scale_m: float = 0.45
    pcf_strong_conflict_margin_m: float = 0.50
    pcf_strong_conflict_sigma: float = 3.0
    pcf_strong_conflict_weight: float = 0.85
    uncertainty_scale_m: float = 0.75
    good_score: float = 0.72
    estimated_score: float = 0.43

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_hypotheses", max(1, min(4, int(self.max_hypotheses))))
        object.__setattr__(
            self,
            "agreement_mahalanobis_sq",
            max(1e-6, float(self.agreement_mahalanobis_sq)),
        )
        object.__setattr__(
            self,
            "max_compatible_distance_m",
            max(1e-3, float(self.max_compatible_distance_m)),
        )
        object.__setattr__(
            self,
            "covariance_regularization_m2",
            max(_MIN_VARIANCE_M2, float(self.covariance_regularization_m2)),
        )
        object.__setattr__(
            self,
            "incompatible_covariance_scale",
            max(0.0, float(self.incompatible_covariance_scale)),
        )
        object.__setattr__(
            self,
            "disagreement_inflation",
            max(0.0, float(self.disagreement_inflation)),
        )
        object.__setattr__(self, "pcf_boundary_scale_m", max(1e-3, float(self.pcf_boundary_scale_m)))
        object.__setattr__(self, "pcf_floor_scale_m", max(1e-3, float(self.pcf_floor_scale_m)))
        object.__setattr__(
            self,
            "pcf_strong_conflict_margin_m",
            max(0.0, float(self.pcf_strong_conflict_margin_m)),
        )
        object.__setattr__(
            self,
            "pcf_strong_conflict_sigma",
            max(1.0, float(self.pcf_strong_conflict_sigma)),
        )
        object.__setattr__(
            self,
            "pcf_strong_conflict_weight",
            min(0.95, max(0.40, float(self.pcf_strong_conflict_weight))),
        )
        object.__setattr__(self, "uncertainty_scale_m", max(1e-3, float(self.uncertainty_scale_m)))
        object.__setattr__(self, "good_score", min(1.0, max(0.0, float(self.good_score))))
        object.__setattr__(self, "estimated_score", min(self.good_score, max(0.0, float(self.estimated_score))))


@dataclass(frozen=True)
class _ScoredHypothesis:
    hypothesis: WorldMeasurementHypothesis
    score: float
    pcf_score: float
    innovation_m: float | None
    innovation_mahalanobis_sq: float | None
    pcf_strong_conflict: bool


class UniversalWorldMeasurementResolver:
    """Resolve one exact-cohort set of metric world hypotheses.

    The resolver is stateless.  That is intentional: PersonGroundState owns
    temporal filtering and lifecycle state, while this class owns only the
    current measurement decision.  Statelessness also makes legacy-vs-new
    replay comparisons safe because both resolvers can consume the same
    immutable candidate set.
    """

    def __init__(self, *, config: WorldMeasurementResolverConfig | None = None) -> None:
        self.config = config or WorldMeasurementResolverConfig()

    def resolve(self, measurement_set: WorldMeasurementSet) -> ResolvedGroundMeasurement:
        """Resolve at most four candidates without performing external work."""

        candidates = tuple(measurement_set.hypotheses[: self.config.max_hypotheses])
        prediction = measurement_set.prediction
        scored = tuple(
            self._score(candidate, prediction.position if prediction is not None else None, prediction.covariance if prediction is not None else None)
            for candidate in candidates
            if candidate.valid
        )
        if not scored:
            return self._continuation_or_rejection(measurement_set)

        ranked = tuple(sorted(scored, key=self._ranking_key))
        selected = ranked[0]
        contributors: list[_ScoredHypothesis] = [selected]
        alternate: _ScoredHypothesis | None = None
        disagreement_m: float | None = None
        disagreement_mahalanobis_sq: float | None = None
        excluded_reasons: dict[str, str] = {}

        for candidate in ranked[1:]:
            pairwise = tuple(
                (
                    contributor,
                    self._distance(
                        contributor.hypothesis.position,
                        candidate.hypothesis.position,
                    ),
                    self._mahalanobis_sq(
                        contributor.hypothesis.position,
                        candidate.hypothesis.position,
                        contributor.hypothesis.covariance,
                        candidate.hypothesis.covariance,
                    ),
                )
                for contributor in contributors
            )
            incompatible = tuple(
                row
                for row in pairwise
                if (
                    row[1] > self.config.max_compatible_distance_m
                    or row[2] > self.config.agreement_mahalanobis_sq
                )
            )
            if not incompatible:
                # Covariance intersection is safe when the relationship is
                # unknown, but every contributor must be mutually compatible.
                # Pairwise-to-primary admission is insufficient: two secondary
                # candidates can each agree with the primary while strongly
                # disagreeing with each other.  Only a literal duplicate (same
                # correlation group, source kind, and anatomical anchor) is
                # excluded from the contributor set; floor and depth candidates
                # sharing camera calibration can still contribute.
                literal_duplicate = any(
                    contributor.hypothesis.correlation_group
                    and contributor.hypothesis.correlation_group
                    == candidate.hypothesis.correlation_group
                    and contributor.hypothesis.kind == candidate.hypothesis.kind
                    and contributor.hypothesis.anchor == candidate.hypothesis.anchor
                    for contributor in contributors
                )
                if not literal_duplicate:
                    contributors.append(candidate)
                else:
                    excluded_reasons[candidate.hypothesis.candidate_id] = (
                        "literal_duplicate_of_contributor"
                    )
                continue

            # Even when candidates share a correlation group, disagreement is
            # valuable evidence.  Preserve the highest-ranked incompatible
            # candidate as an alternate instead of averaging a non-coherent
            # contributor set.  Record the worst conflicting pair because it
            # is the actual reason mutual compatibility failed.
            _conflicting_contributor, candidate_distance_m, mahalanobis_sq = max(
                incompatible,
                key=lambda row: (row[2], row[1], row[0].hypothesis.candidate_id),
            )
            excluded_reasons[candidate.hypothesis.candidate_id] = (
                "not_mutually_compatible_with_contributors"
            )
            if alternate is None:
                alternate = candidate
                disagreement_mahalanobis_sq = mahalanobis_sq
                disagreement_m = candidate_distance_m

        # Combine only genuinely compatible candidates.  The
        # covariance-intersection update makes no independence assumption and
        # therefore cannot manufacture overconfident precision.
        position = selected.hypothesis.position
        covariance = selected.hypothesis.covariance
        fused = False
        for contributor in contributors[1:]:
            position, covariance = self._covariance_intersection(
                position,
                covariance,
                contributor.hypothesis.position,
                contributor.hypothesis.covariance,
            )
            fused = True

        if alternate is not None:
            covariance = self._inflate_for_disagreement(
                covariance,
                position,
                alternate.hypothesis.position,
            )

        selected_score = selected.score
        quality = self._quality(selected_score)
        shared_contributor_pcf_conflict = bool(
            fused
            and all(
                self._pcf_conflict_distance_m(item.hypothesis.pcf) is not None
                for item in contributors
            )
            and any(item.pcf_strong_conflict for item in contributors)
        )
        # A strong, revision-matched geometric contradiction is an admission
        # fact, not merely another score term.  Compatible floor/depth
        # hypotheses can otherwise reinforce the same wrong side of a wall
        # and lift a fused result back above the measured threshold.  Preserve
        # the hypotheses and fused point for diagnostics, but require the
        # temporal estimator to quarantine that current measurement.
        result_pcf_strong_conflict = bool(
            selected.pcf_strong_conflict or shared_contributor_pcf_conflict
        )
        if result_pcf_strong_conflict:
            quality = "weak"
        diagnostics = self._diagnostics(
            ranked,
            selected=selected,
            contributors=contributors,
            alternate=alternate,
            excluded_reasons=excluded_reasons,
        )
        if result_pcf_strong_conflict:
            reason = "selected_hypothesis_strongly_conflicts_with_pcf"
        elif alternate is not None:
            reason = "selected_best_supported_candidate_with_incompatible_alternate"
        elif fused:
            reason = "fused_compatible_hypotheses"
        elif len(ranked) > 1:
            reason = "selected_best_supported_candidate_without_independent_fusion"
        else:
            reason = "selected_single_valid_hypothesis"

        return ResolvedGroundMeasurement(
            cohort=measurement_set.cohort,
            status="measured",
            position=position,
            covariance=covariance,
            selected_candidate_id=selected.hypothesis.candidate_id,
            selected_kind=selected.hypothesis.kind,
            posture=selected.hypothesis.posture,
            support_state=selected.hypothesis.support_state,
            contributor_ids=tuple(item.hypothesis.candidate_id for item in contributors),
            alternate_candidate_id=alternate.hypothesis.candidate_id if alternate else None,
            fused=fused,
            confidence=selected_score,
            quality=quality,
            pcf_score=selected.pcf_score,
            disagreement_m=disagreement_m,
            agreement_mahalanobis_sq=disagreement_mahalanobis_sq,
            reason=reason,
            diagnostics=diagnostics,
        )

    def _score(
        self,
        candidate: WorldMeasurementHypothesis,
        prediction_position: Vector3 | None,
        prediction_covariance: Matrix3 | None,
    ) -> _ScoredHypothesis:
        prior_score = self._pcf_score(candidate.position, candidate.pcf)
        innovation_m: float | None = None
        innovation_mahalanobis_sq: float | None = None
        motion_score = candidate.motion_consistency
        if prediction_position is not None and prediction_covariance is not None:
            innovation_m = self._distance(candidate.position, prediction_position)
            innovation_mahalanobis_sq = self._mahalanobis_sq(
                candidate.position,
                prediction_position,
                candidate.covariance,
                prediction_covariance,
            )
            # A prediction is a soft continuity cue, not an admission gate.
            motion_score *= math.exp(-0.5 * min(16.0, innovation_mahalanobis_sq))

        kind_prior = {
            "floor_ray": 0.82,
            "registered_depth": 0.82,
            "pose_scale": 0.66,
            "gravity_reconstruction": 0.58,
        }[candidate.kind]
        incidence_score = (
            0.60 + 0.40 * candidate.ray_incidence_sin
            if candidate.ray_incidence_sin is not None
            else 1.0
        )
        depth_score = (
            0.60 + 0.40 * candidate.depth_support_fraction
            if candidate.depth_support_fraction is not None
            else 1.0
        )
        occlusion_score = 1.0 - 0.72 * candidate.occlusion
        uncertainty_score = self._uncertainty_score(candidate.covariance)
        evidence = (
            0.38 * candidate.confidence
            + 0.14 * candidate.support_score
            + 0.12 * candidate.posture_compatibility
            + 0.10 * motion_score
            + 0.07 * occlusion_score
            + 0.07 * kind_prior
            + 0.12 * uncertainty_score
        )
        score = evidence * candidate.source_reliability
        score *= 0.72 + 0.28 * incidence_score
        score *= 0.72 + 0.28 * depth_score
        # PCF remains a soft likelihood factor: it never clamps a point or
        # manufactures a replacement. A revision-matched candidate several
        # of its own sigmas through an authored boundary is nevertheless much
        # stronger evidence than an unobserved/near-boundary cell. Give that
        # explicit geometric conflict enough weight to make a lone bad
        # measurement weak (and therefore quarantinable by the caller), while
        # retaining the historical gentle weighting everywhere else.
        pcf_strong_conflict = self._pcf_strong_conflict(candidate)
        pcf_weight = (
            self.config.pcf_strong_conflict_weight
            if pcf_strong_conflict
            else 0.40
        )
        score *= (1.0 - pcf_weight) + pcf_weight * prior_score
        return _ScoredHypothesis(
            hypothesis=candidate,
            score=self._clamp01(score),
            pcf_score=prior_score,
            innovation_m=innovation_m,
            innovation_mahalanobis_sq=innovation_mahalanobis_sq,
            pcf_strong_conflict=pcf_strong_conflict,
        )

    def _continuation_or_rejection(
        self,
        measurement_set: WorldMeasurementSet,
    ) -> ResolvedGroundMeasurement:
        continuation = measurement_set.continuation
        if continuation is not None:
            quality: MeasurementQuality = "prediction" if continuation.kind == "prediction" else "held"
            return ResolvedGroundMeasurement(
                cohort=measurement_set.cohort,
                status=continuation.kind,
                position=continuation.position,
                covariance=continuation.covariance,
                confidence=continuation.confidence,
                posture="unknown",
                support_state="unknown",
                quality=quality,
                reason=f"no_valid_measurement_{continuation.kind}",
            )
        return ResolvedGroundMeasurement(
            cohort=measurement_set.cohort,
            status="rejected",
            confidence=0.0,
            quality="rejected",
            reason="no_valid_measurement_or_process_continuation",
        )

    def _quality(self, score: float) -> MeasurementQuality:
        if score >= self.config.good_score:
            return "good"
        if score >= self.config.estimated_score:
            return "estimated"
        return "weak"

    def _ranking_key(self, scored: _ScoredHypothesis) -> tuple[float, float, str]:
        # Candidate id is the deterministic final tie-breaker.  No room or
        # camera name enters the ordering.
        candidate = scored.hypothesis
        return (-scored.score, -candidate.confidence, candidate.candidate_id)

    def _diagnostics(
        self,
        ranked: Iterable[_ScoredHypothesis],
        *,
        selected: _ScoredHypothesis,
        contributors: Iterable[_ScoredHypothesis],
        alternate: _ScoredHypothesis | None,
        excluded_reasons: dict[str, str],
    ) -> tuple[WorldMeasurementCandidateDiagnostic, ...]:
        contributor_ids = {item.hypothesis.candidate_id for item in contributors}
        alternate_id = alternate.hypothesis.candidate_id if alternate else None
        rows: list[WorldMeasurementCandidateDiagnostic] = []
        for item in ranked:
            candidate = item.hypothesis
            pairwise_nis = None
            if candidate.candidate_id != selected.hypothesis.candidate_id:
                pairwise_nis = self._mahalanobis_sq(
                    selected.hypothesis.position,
                    candidate.position,
                    selected.hypothesis.covariance,
                    candidate.covariance,
                )
            rows.append(
                WorldMeasurementCandidateDiagnostic(
                    candidate_id=candidate.candidate_id,
                    kind=candidate.kind,
                    score=item.score,
                    pcf_score=item.pcf_score,
                    innovation_m=item.innovation_m,
                    agreement_mahalanobis_sq=pairwise_nis,
                    compatible_with_selected=(
                        candidate.candidate_id in contributor_ids
                        or (
                            candidate.candidate_id != alternate_id
                            and pairwise_nis is not None
                            and pairwise_nis <= self.config.agreement_mahalanobis_sq
                            and self._distance(
                                selected.hypothesis.position,
                                candidate.position,
                            )
                            <= self.config.max_compatible_distance_m
                        )
                    ),
                    selected=(candidate.candidate_id == selected.hypothesis.candidate_id),
                    retained_as_alternate=(candidate.candidate_id == alternate_id),
                    rejection_reason=(
                        excluded_reasons.get(candidate.candidate_id)
                        or (
                            "strong_pcf_boundary_conflict"
                            if item.pcf_strong_conflict
                            else None
                        )
                        or candidate.rejection_reason
                    ),
                )
            )
        return tuple(rows[: self.config.max_hypotheses])

    def _pcf_score(self, position: Vector3, prior: WorldPriorEvidence | None) -> float:
        if prior is None:
            return 1.0
        # Lack of observed reconstruction is neutral by itself.  Authored
        # extent and boundary evidence remain useful even where the scan has
        # no dense samples, so do not return early for an unobserved cell.
        score = (
            0.65 + 0.35 * prior.observed_confidence
            if prior.evidence_observed
            else 1.0
        )
        if prior.inside_extent is False:
            score *= 0.55
        if prior.inside_authored_space is False:
            score *= 0.25
        if prior.boundary_signed_distance_m is not None and prior.boundary_signed_distance_m < 0.0:
            score *= math.exp(
                max(-8.0, prior.boundary_signed_distance_m / self.config.pcf_boundary_scale_m)
            )
        if prior.floor_height_m is not None:
            floor_error = abs(float(position.y) - prior.floor_height_m)
            score *= math.exp(-min(8.0, floor_error / self.config.pcf_floor_scale_m))
        # Obstacle clearance is intentionally not included.  The current PCF
        # obstacle layer is not a semantic support-surface authority.
        return self._clamp01(score)

    def _uncertainty_score(self, covariance: Matrix3) -> float:
        """Map horizontal covariance to a bounded generic evidence score.

        The position is a ground footprint, so X/Z uncertainty controls the
        resolver decision.  Using the largest eigenvalue of the symmetric X/Z
        block prevents a long, shallow-incidence error ellipse from looking
        precise merely because its orthogonal axis is tight.
        """

        sigma_major_m = self._horizontal_sigma_major(covariance)
        ratio = sigma_major_m / self.config.uncertainty_scale_m
        return self._clamp01(1.0 / (1.0 + ratio * ratio))

    def _pcf_strong_conflict(
        self,
        candidate: WorldMeasurementHypothesis,
    ) -> bool:
        prior = candidate.pcf
        outside_m = self._pcf_conflict_distance_m(prior)
        if outside_m is None:
            return False
        uncertainty_margin_m = (
            self.config.pcf_strong_conflict_sigma
            * self._horizontal_sigma_major(candidate.covariance)
        )
        return outside_m > max(
            self.config.pcf_strong_conflict_margin_m,
            uncertainty_margin_m,
        )

    @staticmethod
    def _pcf_conflict_distance_m(
        prior: WorldPriorEvidence | None,
    ) -> float | None:
        if prior is None:
            return None
        if (
            prior.inside_authored_space is False
            and prior.boundary_signed_distance_m is not None
            and float(prior.boundary_signed_distance_m) < 0.0
        ):
            return abs(float(prior.boundary_signed_distance_m))
        if (
            prior.inside_extent is False
            and prior.extent_outside_distance_m is not None
            and float(prior.extent_outside_distance_m) > 0.0
        ):
            return float(prior.extent_outside_distance_m)
        return None

    @staticmethod
    def _horizontal_sigma_major(covariance: Matrix3) -> float:
        values = covariance.values
        variance_x = max(_MIN_VARIANCE_M2, float(values[0]))
        variance_z = max(_MIN_VARIANCE_M2, float(values[8]))
        covariance_xz = 0.5 * (float(values[2]) + float(values[6]))
        half_trace = 0.5 * (variance_x + variance_z)
        radius = math.sqrt(
            max(
                0.0,
                ((0.5 * (variance_x - variance_z)) ** 2)
                + covariance_xz * covariance_xz,
            )
        )
        return math.sqrt(max(_MIN_VARIANCE_M2, half_trace + radius))

    def _mahalanobis_sq(
        self,
        left: Vector3,
        right: Vector3,
        left_covariance: Matrix3,
        right_covariance: Matrix3,
    ) -> float:
        delta = self._vector_sub(left, right)
        covariance = self._matrix_add(
            self._regularize(self._matrix_from_contract(left_covariance)),
            self._regularize(self._matrix_from_contract(right_covariance)),
        )
        inverse = self._inverse3(covariance)
        if inverse is None:
            return float("inf")
        return max(0.0, self._quadratic(delta, inverse))

    def _covariance_intersection(
        self,
        left_position: Vector3,
        left_covariance: Matrix3,
        right_position: Vector3,
        right_covariance: Matrix3,
    ) -> tuple[Vector3, Matrix3]:
        left = self._regularize(self._matrix_from_contract(left_covariance))
        right = self._regularize(self._matrix_from_contract(right_covariance))
        left_inverse = self._inverse3(left)
        right_inverse = self._inverse3(right)
        if left_inverse is None or right_inverse is None:
            return left_position, left_covariance

        left_vector = self._vector_tuple(left_position)
        right_vector = self._vector_tuple(right_position)
        best: tuple[float, float, tuple[tuple[float, ...], ...], tuple[float, ...]] | None = None
        # A fixed grid is bounded, deterministic, and avoids a numerical
        # optimizer in the per-track hot path.
        for index in range(1, 20):
            weight = index / 20.0
            information = self._matrix_add(
                self._matrix_scale(left_inverse, weight),
                self._matrix_scale(right_inverse, 1.0 - weight),
            )
            covariance = self._inverse3(information)
            if covariance is None:
                continue
            information_vector = self._vector_add(
                self._matrix_vector(left_inverse, left_vector, scale=weight),
                self._matrix_vector(right_inverse, right_vector, scale=1.0 - weight),
            )
            position = self._matrix_vector(covariance, information_vector)
            trace = covariance[0][0] + covariance[1][1] + covariance[2][2]
            rank = (trace, abs(weight - 0.5), covariance, position)
            if best is None or rank < best:
                best = rank
        if best is None:
            return left_position, left_covariance
        _trace, _weight_tie, covariance, position = best
        return self._vector_contract(position), self._matrix_contract(covariance)

    def _inflate_for_disagreement(
        self,
        covariance: Matrix3,
        selected_position: Vector3,
        alternate_position: Vector3,
    ) -> Matrix3:
        base = self._matrix_scale(
            self._regularize(self._matrix_from_contract(covariance)),
            1.0 + self.config.incompatible_covariance_scale,
        )
        delta = self._vector_sub(selected_position, alternate_position)
        base = self._matrix_add(
            base,
            self._matrix_scale(self._outer(delta, delta), self.config.disagreement_inflation),
        )
        return self._matrix_contract(base)

    def _regularize(self, matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
        symmetric = tuple(
            tuple(
                0.5 * (float(matrix[row][column]) + float(matrix[column][row]))
                for column in range(_AXES)
            )
            for row in range(_AXES)
        )
        max_diagonal = max(float(symmetric[index][index]) for index in range(_AXES))
        scale = (
            _MAX_VARIANCE_M2 / max_diagonal
            if max_diagonal > _MAX_VARIANCE_M2
            else 1.0
        )
        regularization = self.config.covariance_regularization_m2
        return tuple(
            tuple(
                (
                    float(symmetric[row][column]) * scale + regularization
                    if row == column
                    else float(symmetric[row][column]) * scale
                )
                for column in range(_AXES)
            )
            for row in range(_AXES)
        )

    @staticmethod
    def _matrix_from_contract(matrix: Matrix3) -> tuple[tuple[float, ...], ...]:
        values = matrix.values
        return (
            (float(values[0]), float(values[1]), float(values[2])),
            (float(values[3]), float(values[4]), float(values[5])),
            (float(values[6]), float(values[7]), float(values[8])),
        )

    @staticmethod
    def _matrix_contract(matrix: tuple[tuple[float, ...], ...]) -> Matrix3:
        values = tuple(float(matrix[row][column]) for row in range(_AXES) for column in range(_AXES))
        return Matrix3(values=values)  # type: ignore[arg-type]

    @staticmethod
    def _vector_tuple(vector: Vector3) -> tuple[float, ...]:
        return float(vector.x), float(vector.y), float(vector.z)

    @staticmethod
    def _vector_contract(vector: tuple[float, ...]) -> Vector3:
        return Vector3(x=float(vector[0]), y=float(vector[1]), z=float(vector[2]))

    @staticmethod
    def _vector_sub(left: Vector3, right: Vector3) -> tuple[float, ...]:
        return (
            float(left.x - right.x),
            float(left.y - right.y),
            float(left.z - right.z),
        )

    @staticmethod
    def _distance(left: Vector3, right: Vector3) -> float:
        delta = UniversalWorldMeasurementResolver._vector_sub(left, right)
        return math.sqrt(sum(value * value for value in delta))

    @staticmethod
    def _outer(vector: tuple[float, ...], other: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(vector[row] * other[column] for column in range(_AXES)) for row in range(_AXES))

    @staticmethod
    def _quadratic(vector: tuple[float, ...], matrix: tuple[tuple[float, ...], ...]) -> float:
        projected = UniversalWorldMeasurementResolver._matrix_vector(matrix, vector)
        return sum(vector[index] * projected[index] for index in range(_AXES))

    @staticmethod
    def _matrix_vector(
        matrix: tuple[tuple[float, ...], ...],
        vector: tuple[float, ...],
        *,
        scale: float = 1.0,
    ) -> tuple[float, ...]:
        return tuple(
            float(scale) * sum(matrix[row][column] * vector[column] for column in range(_AXES))
            for row in range(_AXES)
        )

    @staticmethod
    def _vector_add(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(left[index] + right[index] for index in range(_AXES))

    @staticmethod
    def _matrix_add(
        left: tuple[tuple[float, ...], ...],
        right: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(left[row][column] + right[row][column] for column in range(_AXES))
            for row in range(_AXES)
        )

    @staticmethod
    def _matrix_scale(
        matrix: tuple[tuple[float, ...], ...],
        scale: float,
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(float(scale) * matrix[row][column] for column in range(_AXES))
            for row in range(_AXES)
        )

    @staticmethod
    def _inverse3(
        matrix: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[float, ...], ...] | None:
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        g, h, i = matrix[2]
        determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        if not math.isfinite(determinant) or abs(determinant) <= 1e-15:
            return None
        inverse_determinant = 1.0 / determinant
        return (
            ((e * i - f * h) * inverse_determinant, (c * h - b * i) * inverse_determinant, (b * f - c * e) * inverse_determinant),
            ((f * g - d * i) * inverse_determinant, (a * i - c * g) * inverse_determinant, (c * d - a * f) * inverse_determinant),
            ((d * h - e * g) * inverse_determinant, (b * g - a * h) * inverse_determinant, (a * e - b * d) * inverse_determinant),
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        return min(1.0, max(0.0, float(value)))


# Short alias for callers that do not need to spell out the implementation
# adjective.  Both names are intentionally public and refer to the same
# stateless resolver.
WorldMeasurementResolver = UniversalWorldMeasurementResolver


__all__ = [
    "UniversalWorldMeasurementResolver",
    "WorldMeasurementResolver",
    "WorldMeasurementResolverConfig",
]
