from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from noesis_core.contracts.base import Matrix3, Vector3
from noesis_core.contracts.world_measurement import (
    WorldMeasurementCohort,
    WorldMeasurementHypothesis,
    WorldMeasurementSet,
    WorldPriorEvidence,
    WorldProcessContinuation,
)
from noesis_core.world.resolver import (
    UniversalWorldMeasurementResolver,
    WorldMeasurementResolverConfig,
)


def _cohort(*, frame_id: int = 10, pcf_revision: str | None = None) -> WorldMeasurementCohort:
    return WorldMeasurementCohort(
        track_key="camera-a:source-0:tracker-7",
        tracker_lifecycle_generation=1,
        camera_id="camera-a",
        source_id=0,
        tracker_id=7,
        frame_id=frame_id,
        observed_at_us=1_000_000 + frame_id,
        world_revision="world-revision-1",
        calibration_revision="calibration-revision-1",
        pcf_revision=pcf_revision,
    )


def _covariance(x: float = 0.20, y: float = 0.04, z: float = 0.20) -> Matrix3:
    return Matrix3(values=(x, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, z))


def _hypothesis(
    cohort: WorldMeasurementCohort,
    candidate_id: str,
    kind: str,
    position: tuple[float, float, float],
    *,
    covariance: Matrix3 | None = None,
    confidence: float = 0.9,
    correlation_group: str | None = None,
    pcf: WorldPriorEvidence | None = None,
    anchor: str = "lower_body_contact",
    support_state: str = "floor",
    valid: bool = True,
    rejection_reason: str | None = None,
) -> WorldMeasurementHypothesis:
    return WorldMeasurementHypothesis(
        candidate_id=candidate_id,
        cohort=cohort,
        kind=kind,  # type: ignore[arg-type]
        position=Vector3(x=position[0], y=position[1], z=position[2]),
        covariance=covariance or _covariance(),
        anchor=anchor,
        support_state=support_state,  # type: ignore[arg-type]
        confidence=confidence,
        correlation_group=correlation_group,
        pcf=pcf,
        valid=valid,
        rejection_reason=rejection_reason,
        ray_incidence_sin=0.9 if kind == "floor_ray" else None,
        depth_support_fraction=0.9 if kind == "registered_depth" else None,
    )


def test_covariance_contract_requires_finite_symmetric_psd_values() -> None:
    with pytest.raises(ValidationError, match="positive semidefinite"):
        _hypothesis(
            _cohort(),
            "bad-negative",
            "floor_ray",
            (0.0, 0.0, 0.0),
            covariance=Matrix3(values=(-0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1)),
        )
    with pytest.raises(ValidationError, match="symmetric"):
        _hypothesis(
            _cohort(),
            "bad-asymmetric",
            "floor_ray",
            (0.0, 0.0, 0.0),
            covariance=Matrix3(values=(0.1, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1)),
        )
    with pytest.raises(ValidationError, match="finite"):
        _hypothesis(
            _cohort(),
            "bad-nan",
            "floor_ray",
            (0.0, 0.0, 0.0),
            covariance=Matrix3(values=(math.nan, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1)),
        )


def test_measurement_set_rejects_mixed_cohort_and_is_bounded() -> None:
    cohort = _cohort()
    mismatched = _hypothesis(_cohort(frame_id=11), "other", "floor_ray", (0.0, 0.0, 0.0))
    with pytest.raises(ValidationError, match="exact cohort identity"):
        WorldMeasurementSet(cohort=cohort, hypotheses=(mismatched,))

    candidates = tuple(
        _hypothesis(cohort, f"candidate-{index}", "floor_ray", (0.0, 0.0, 0.0))
        for index in range(5)
    )
    with pytest.raises(ValidationError, match="at most 4"):
        WorldMeasurementSet(cohort=cohort, hypotheses=candidates)


@pytest.mark.parametrize("kind,quality", [("prediction", "prediction"), ("hold", "held")])
def test_prediction_and_hold_are_explicit_non_measurement_results(kind: str, quality: str) -> None:
    cohort = _cohort()
    continuation = WorldProcessContinuation(
        kind=kind,  # type: ignore[arg-type]
        cohort=cohort,
        position=Vector3(x=2.0, y=0.0, z=3.0),
        covariance=_covariance(0.8, 0.2, 0.8),
        confidence=0.4,
    )
    result = UniversalWorldMeasurementResolver().resolve(
        WorldMeasurementSet(cohort=cohort, continuation=continuation)
    )
    assert result.status == kind
    assert result.quality == quality
    assert result.selected_candidate_id is None
    assert result.position == continuation.position


def test_agreeing_floor_and_depth_are_genuinely_fused_with_ci() -> None:
    cohort = _cohort()
    floor = _hypothesis(
        cohort,
        "floor-1",
        "floor_ray",
        (1.0, 0.0, 2.0),
        covariance=_covariance(0.20, 0.04, 0.20),
        confidence=0.92,
        correlation_group="camera-ray",
    )
    depth = _hypothesis(
        cohort,
        "depth-1",
        "registered_depth",
        (1.12, 0.0, 2.08),
        covariance=_covariance(0.24, 0.05, 0.24),
        confidence=0.89,
        correlation_group="camera-ray",
    )
    result = UniversalWorldMeasurementResolver().resolve(
        WorldMeasurementSet(cohort=cohort, hypotheses=(floor, depth))
    )
    assert result.status == "measured"
    assert result.fused is True
    assert result.contributor_ids == ("floor-1", "depth-1")
    assert result.alternate_candidate_id is None
    assert result.position is not None
    assert result.position != floor.position
    assert result.position != depth.position
    assert result.covariance is not None
    assert result.quality in {"good", "estimated"}
    assert result.legacy_source_label == "pose_depth_fused"
    assert result.legacy_quality_label in {"good", "estimated"}


def test_incompatible_candidates_choose_strongest_retain_alternate_and_inflate_covariance() -> None:
    cohort = _cohort()
    selected = _hypothesis(
        cohort,
        "floor-strong",
        "floor_ray",
        (0.0, 0.0, 0.0),
        covariance=_covariance(0.04, 0.02, 0.04),
        confidence=0.98,
        correlation_group="same-anchor",
    )
    alternate = _hypothesis(
        cohort,
        "depth-weak",
        "registered_depth",
        (2.0, 0.0, 0.0),
        covariance=_covariance(0.04, 0.02, 0.04),
        confidence=0.55,
        correlation_group="same-anchor",
    )
    result = UniversalWorldMeasurementResolver().resolve(
        WorldMeasurementSet(cohort=cohort, hypotheses=(selected, alternate))
    )
    assert result.selected_candidate_id == "floor-strong"
    assert result.alternate_candidate_id == "depth-weak"
    assert result.fused is False
    assert result.disagreement_m == pytest.approx(2.0)
    assert result.covariance is not None
    assert result.covariance.values[0] > selected.covariance.values[0]
    assert any(item.retained_as_alternate for item in result.diagnostics)


def test_observed_floor_outranks_incompatible_precise_unknown_body_range() -> None:
    cohort = _cohort()
    floor = _hypothesis(
        cohort,
        "floor-contact",
        "floor_ray",
        (0.0, 0.0, 6.0),
        covariance=_covariance(0.18, 0.04, 0.18),
        confidence=0.75,
        support_state="floor",
    )
    torso_range = _hypothesis(
        cohort,
        "torso-range",
        "registered_depth",
        (0.0, 0.0, 0.1),
        covariance=_covariance(0.001, 0.02, 0.02),
        confidence=0.90,
        anchor="torso_core",
        support_state="unknown",
    )

    result = UniversalWorldMeasurementResolver().resolve(
        WorldMeasurementSet(cohort=cohort, hypotheses=(floor, torso_range))
    )

    assert result.selected_candidate_id == "floor-contact"
    assert result.support_state == "floor"
    assert result.alternate_candidate_id is None
    torso_diagnostic = next(
        row for row in result.diagnostics if row.candidate_id == "torso-range"
    )
    assert torso_diagnostic.rejection_reason == (
        "support_state_not_equivalent_to_selected"
    )


def test_compatible_precise_unknown_body_range_cannot_pull_floor_result() -> None:
    cohort = _cohort()
    floor = _hypothesis(
        cohort,
        "floor-contact",
        "floor_ray",
        (0.0, 0.0, 6.0),
        covariance=_covariance(0.18, 0.04, 0.18),
        confidence=0.75,
        support_state="floor",
    )
    torso_range = _hypothesis(
        cohort,
        "torso-range",
        "registered_depth",
        (0.25, 0.0, 6.0),
        covariance=_covariance(0.001, 0.02, 0.001),
        confidence=0.99,
        anchor="torso_core",
        support_state="unknown",
    )

    result = UniversalWorldMeasurementResolver().resolve(
        WorldMeasurementSet(cohort=cohort, hypotheses=(floor, torso_range))
    )

    assert result.selected_candidate_id == "floor-contact"
    assert result.contributor_ids == ("floor-contact",)
    assert result.fused is False
    assert result.position == floor.position
    torso_diagnostic = next(
        row for row in result.diagnostics if row.candidate_id == "torso-range"
    )
    assert torso_diagnostic.rejection_reason == (
        "support_state_not_equivalent_to_selected"
    )


def test_large_metric_disagreement_never_fuses_even_with_broad_covariance() -> None:
    cohort = _cohort()
    floor = _hypothesis(
        cohort,
        "floor-broad",
        "floor_ray",
        (0.0, 0.0, 0.0),
        covariance=_covariance(4.0, 0.2, 4.0),
        confidence=0.95,
    )
    depth = _hypothesis(
        cohort,
        "depth-broad",
        "registered_depth",
        (1.5, 0.0, 0.0),
        covariance=_covariance(4.0, 0.2, 4.0),
        confidence=0.80,
    )
    resolver = UniversalWorldMeasurementResolver(
        config=WorldMeasurementResolverConfig(max_compatible_distance_m=1.25)
    )

    result = resolver.resolve(
        WorldMeasurementSet(cohort=cohort, hypotheses=(floor, depth))
    )

    assert result.fused is False
    assert result.alternate_candidate_id == "depth-broad"
    assert result.disagreement_m == pytest.approx(1.5)
    assert any(
        item.candidate_id == "depth-broad" and not item.compatible_with_selected
        for item in result.diagnostics
    )


def test_fusion_requires_every_contributor_to_be_mutually_compatible() -> None:
    cohort = _cohort()
    primary = _hypothesis(
        cohort,
        "primary",
        "floor_ray",
        (0.0, 0.0, 0.0),
        covariance=_covariance(1.0, 0.04, 1.0),
        confidence=0.99,
    )
    positive = _hypothesis(
        cohort,
        "positive",
        "registered_depth",
        (1.2, 0.0, 0.0),
        covariance=_covariance(1.0, 0.04, 1.0),
        confidence=0.90,
    )
    negative = _hypothesis(
        cohort,
        "negative",
        "pose_scale",
        (-1.2, 0.0, 0.0),
        covariance=_covariance(1.0, 0.04, 1.0),
        confidence=0.80,
    )

    result = UniversalWorldMeasurementResolver().resolve(
        WorldMeasurementSet(
            cohort=cohort,
            hypotheses=(primary, positive, negative),
        )
    )

    assert result.contributor_ids == ("primary", "positive")
    assert result.alternate_candidate_id == "negative"
    assert result.disagreement_m == pytest.approx(2.4)
    assert result.position is not None
    assert result.position.x > 0.0
    negative_diagnostic = next(
        item for item in result.diagnostics if item.candidate_id == "negative"
    )
    assert negative_diagnostic.rejection_reason == (
        "not_mutually_compatible_with_contributors"
    )


def test_literal_duplicate_is_not_reported_as_false_fusion() -> None:
    cohort = _cohort()
    first = _hypothesis(
        cohort,
        "floor-a",
        "floor_ray",
        (1.0, 0.0, 1.0),
        correlation_group="same-ray",
    )
    duplicate = _hypothesis(
        cohort,
        "floor-b",
        "floor_ray",
        (1.01, 0.0, 1.01),
        confidence=0.8,
        correlation_group="same-ray",
    )
    result = UniversalWorldMeasurementResolver().resolve(
        WorldMeasurementSet(cohort=cohort, hypotheses=(first, duplicate))
    )
    assert result.fused is False
    assert result.contributor_ids == ("floor-a",)
    assert result.alternate_candidate_id is None
    assert result.legacy_source_label == "pose_floor_only"


def test_pcf_is_soft_evidence_and_obstacle_clearance_is_diagnostic_only() -> None:
    cohort = _cohort(pcf_revision="pcf-1")
    prior = WorldPriorEvidence(
        prior_id="scene-prior",
        revision_id="pcf-1",
        status="evaluated",
        inside_extent=False,
        inside_authored_space=False,
        evidence_observed=True,
        observed_confidence=1.0,
        boundary_signed_distance_m=-1.0,
        floor_height_m=0.0,
        obstacle_clearance_m=-10.0,
    )
    candidate = _hypothesis(
        cohort,
        "outside",
        "registered_depth",
        (4.0, 0.0, 4.0),
        confidence=0.99,
        pcf=prior,
    )
    result = UniversalWorldMeasurementResolver().resolve(
        WorldMeasurementSet(cohort=cohort, hypotheses=(candidate,))
    )
    assert result.position == candidate.position
    assert result.pcf_score is not None and result.pcf_score < 0.2
    assert result.status == "measured"


def test_precise_candidate_far_through_authored_boundary_becomes_weak_without_clamping() -> None:
    cohort = _cohort(pcf_revision="pcf-1")
    prior = WorldPriorEvidence(
        prior_id="scene-prior",
        revision_id="pcf-1",
        status="fail",
        inside_extent=True,
        inside_authored_space=False,
        evidence_observed=False,
        observed_confidence=0.0,
        boundary_signed_distance_m=-2.375,
        floor_height_m=0.0,
        reasons=("outside_authored_space",),
    )
    candidate = _hypothesis(
        cohort,
        "floor-through-wall",
        "floor_ray",
        (4.0, 0.0, 4.0),
        covariance=_covariance(0.04, 0.04, 0.04),
        confidence=0.99,
        pcf=prior,
    )

    result = UniversalWorldMeasurementResolver().resolve(
        WorldMeasurementSet(cohort=cohort, hypotheses=(candidate,))
    )

    # The resolver preserves the measured hypothesis for diagnostics.  It
    # neither snaps nor fabricates an inside-room replacement, but its weak
    # quality tells PersonGroundState to quarantine the impossible update.
    assert result.status == "measured"
    assert result.position == candidate.position
    assert result.quality == "weak"
    assert result.reason == "selected_hypothesis_strongly_conflicts_with_pcf"
    assert result.diagnostics[0].rejection_reason == "strong_pcf_boundary_conflict"


def test_unobserved_outside_extent_still_reduces_pcf_plausibility_without_clamping() -> None:
    cohort = _cohort(pcf_revision="pcf-1")
    outside = WorldPriorEvidence(
        prior_id="scene-prior",
        revision_id="pcf-1",
        status="unknown",
        inside_extent=False,
        inside_authored_space=False,
        evidence_observed=False,
        observed_confidence=0.0,
        reasons=("outside_prior_extent",),
    )
    candidate = _hypothesis(
        cohort,
        "outside-unobserved",
        "floor_ray",
        (8.0, 0.0, 8.0),
        pcf=outside,
    )
    result = UniversalWorldMeasurementResolver().resolve(
        WorldMeasurementSet(cohort=cohort, hypotheses=(candidate,))
    )
    assert result.position == candidate.position
    assert result.pcf_score == pytest.approx(0.55 * 0.25)


def test_precise_candidate_far_beyond_revision_bound_extent_becomes_weak() -> None:
    cohort = _cohort(pcf_revision="pcf-1")
    outside = WorldPriorEvidence(
        prior_id="scene-prior",
        revision_id="pcf-1",
        status="unknown",
        inside_extent=False,
        inside_authored_space=False,
        extent_outside_distance_m=1.47,
        evidence_observed=False,
        observed_confidence=0.0,
        reasons=("outside_prior_extent",),
    )
    candidate = _hypothesis(
        cohort,
        "outside-authoritative-extent",
        "floor_ray",
        (4.0, 0.0, 8.0),
        covariance=_covariance(0.04, 0.04, 0.04),
        confidence=0.99,
        pcf=outside,
    )

    result = UniversalWorldMeasurementResolver().resolve(
        WorldMeasurementSet(cohort=cohort, hypotheses=(candidate,))
    )

    assert result.position == candidate.position
    assert result.quality == "weak"
    assert result.reason == "selected_hypothesis_strongly_conflicts_with_pcf"
    assert result.diagnostics[0].rejection_reason == "strong_pcf_boundary_conflict"


def test_compatible_candidates_cannot_fuse_away_shared_strong_pcf_conflict() -> None:
    cohort = _cohort(pcf_revision="pcf-1")
    outside = WorldPriorEvidence(
        prior_id="scene-prior",
        revision_id="pcf-1",
        status="fail",
        inside_extent=True,
        inside_authored_space=False,
        evidence_observed=False,
        observed_confidence=0.0,
        boundary_signed_distance_m=-2.0,
        reasons=("outside_authored_space",),
    )
    floor = _hypothesis(
        cohort,
        "floor-outside",
        "floor_ray",
        (4.0, 0.0, 4.0),
        covariance=_covariance(0.04, 0.04, 0.04),
        confidence=0.99,
        pcf=outside,
    )
    depth = _hypothesis(
        cohort,
        "depth-outside",
        "registered_depth",
        (4.05, 0.0, 4.02),
        covariance=_covariance(0.09, 0.04, 0.09),
        confidence=0.99,
        pcf=outside,
    )

    result = UniversalWorldMeasurementResolver().resolve(
        WorldMeasurementSet(cohort=cohort, hypotheses=(floor, depth))
    )

    assert result.fused is True
    assert result.quality == "weak"
    assert result.reason == "selected_hypothesis_strongly_conflicts_with_pcf"


def test_uncertain_selected_candidate_cannot_hide_precise_contributor_pcf_conflict() -> None:
    cohort = _cohort(pcf_revision="pcf-1")
    outside = WorldPriorEvidence(
        prior_id="scene-prior",
        revision_id="pcf-1",
        status="fail",
        inside_extent=True,
        inside_authored_space=False,
        evidence_observed=False,
        observed_confidence=0.0,
        boundary_signed_distance_m=-2.7,
        reasons=("outside_authored_space",),
    )
    precise_floor = _hypothesis(
        cohort,
        "precise-floor-outside",
        "floor_ray",
        (4.0, 0.0, 4.0),
        covariance=_covariance(0.04, 0.04, 0.04),
        confidence=0.62,
        pcf=outside,
    )
    uncertain_depth = _hypothesis(
        cohort,
        "uncertain-depth-outside",
        "registered_depth",
        (4.05, 0.0, 4.02),
        covariance=_covariance(0.64, 0.04, 0.64),
        confidence=0.99,
        pcf=outside,
    )

    result = UniversalWorldMeasurementResolver().resolve(
        WorldMeasurementSet(
            cohort=cohort,
            hypotheses=(precise_floor, uncertain_depth),
        )
    )

    assert result.fused is True
    assert any(
        row.rejection_reason == "strong_pcf_boundary_conflict"
        for row in result.diagnostics
    )
    assert result.quality == "weak"
    assert result.reason == "selected_hypothesis_strongly_conflicts_with_pcf"


def test_candidate_covariance_influences_selection_not_only_fusion() -> None:
    cohort = _cohort()
    uncertain = _hypothesis(
        cohort,
        "uncertain-high-confidence",
        "floor_ray",
        (0.0, 0.0, 0.0),
        covariance=_covariance(4.0, 0.04, 4.0),
        confidence=0.92,
    )
    precise = _hypothesis(
        cohort,
        "precise-slightly-lower-confidence",
        "registered_depth",
        (10.0, 0.0, 0.0),
        covariance=_covariance(0.04, 0.04, 0.04),
        confidence=0.88,
    )
    result = UniversalWorldMeasurementResolver().resolve(
        WorldMeasurementSet(cohort=cohort, hypotheses=(uncertain, precise))
    )
    assert result.selected_candidate_id == "precise-slightly-lower-confidence"
    assert result.alternate_candidate_id == "uncertain-high-confidence"


def test_large_psd_covariance_remains_psd_after_bounded_regularization() -> None:
    cohort = _cohort()
    direction = np.asarray([2.0e9, 1.0e9, 2.0e9], dtype=np.float64)
    rank_one = np.outer(direction, direction)
    covariance = Matrix3(values=tuple(float(value) for value in rank_one.reshape(-1)))
    candidates = (
        _hypothesis(
            cohort,
            "large-a",
            "floor_ray",
            (0.0, 0.0, 0.0),
            covariance=covariance,
        ),
        _hypothesis(
            cohort,
            "large-b",
            "registered_depth",
            (0.1, 0.0, 0.1),
            covariance=covariance,
        ),
    )

    result = UniversalWorldMeasurementResolver().resolve(
        WorldMeasurementSet(cohort=cohort, hypotheses=candidates)
    )

    assert result.covariance is not None
    matrix = np.asarray(result.covariance.values, dtype=np.float64).reshape(3, 3)
    assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-6


def test_resolver_is_deterministic_for_tied_candidates() -> None:
    cohort = _cohort()
    candidates = (
        _hypothesis(cohort, "z-candidate", "gravity_reconstruction", (1.0, 0.0, 1.0), confidence=0.8),
        _hypothesis(cohort, "a-candidate", "pose_scale", (1.0, 0.0, 1.0), confidence=0.8),
    )
    resolver = UniversalWorldMeasurementResolver(
        config=WorldMeasurementResolverConfig(agreement_mahalanobis_sq=9.0)
    )
    first = resolver.resolve(WorldMeasurementSet(cohort=cohort, hypotheses=candidates))
    second = resolver.resolve(WorldMeasurementSet(cohort=cohort, hypotheses=candidates))
    assert first.model_dump() == second.model_dump()
    assert first.selected_candidate_id == "a-candidate"
    assert tuple(item.candidate_id for item in first.diagnostics) == (
        "a-candidate",
        "z-candidate",
    )
