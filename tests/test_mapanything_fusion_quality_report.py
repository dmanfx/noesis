from __future__ import annotations

import numpy as np

from scripts.mapanything_fusion_quality_report import evaluate_fusion_cohort


def _cohort(values: list[float], *, confidence_tail: bool = False):
    depth = np.stack(
        [np.full((4, 5), value, dtype=np.float32) for value in values]
    )
    confidence = np.ones_like(depth)
    if confidence_tail:
        confidence[-1] = 1_000.0
    mask = np.ones_like(depth, dtype=np.uint8)
    return depth, confidence, mask


def test_scale_normalization_recovers_consensus_from_frame_scale_drift() -> None:
    depth, confidence, mask = _cohort([1.0, 2.0, 4.0])

    raw = evaluate_fusion_cohort(
        depth,
        confidence,
        mask,
        normalize_frame_scale=False,
    )
    normalized = evaluate_fusion_cohort(
        depth,
        confidence,
        mask,
        normalize_frame_scale=True,
    )

    assert raw["support"]["retained_full_frame_fraction"] == 0.0
    assert normalized["support"]["retained_full_frame_fraction"] == 1.0
    assert normalized["scale_normalization"]["factors"] == [2.0, 1.0, 0.5]


def test_confidence_cap_prevents_one_high_tail_frame_from_dominating() -> None:
    depth, confidence, mask = _cohort(
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.15],
        confidence_tail=True,
    )

    report = evaluate_fusion_cohort(
        depth,
        confidence,
        mask,
        normalize_frame_scale=False,
    )
    raw = report["candidates"]["confidence_weighted_raw"]
    capped = report["candidates"]["confidence_weighted_robust_capped"]
    runtime_bounded = report["candidates"][
        "confidence_weighted_runtime_bounded"
    ]
    median = report["candidates"]["temporal_median"]

    assert raw["temporal_residual_p50_m"] > capped["temporal_residual_p50_m"]
    assert capped["temporal_residual_p50_m"] >= median["temporal_residual_p50_m"]
    assert (
        runtime_bounded["temporal_residual_p50_m"]
        < raw["temporal_residual_p50_m"]
    )
    assert report["confidence"]["runtime_frame_caps"][-1] == 1_000.0
    assert report["support"]["retained_p50"] == 6.0


def test_unreasonable_scale_factor_is_quarantined_not_clipped() -> None:
    depth, confidence, mask = _cohort([0.1, 1.0, 1.0, 1.0, 1.0, 1.0])

    report = evaluate_fusion_cohort(
        depth,
        confidence,
        mask,
        normalize_frame_scale=True,
    )

    normalization = report["scale_normalization"]
    assert np.isclose(normalization["proposed_factors"][0], 10.0)
    assert normalization["factors"][0] == 1.0
    assert normalization["rejected"] == [True, False, False, False, False, False]
    assert normalization["rejection_policy"] == "quarantine_entire_frame"
    assert report["support"]["quarantined_frame_indices"] == [0]
    assert report["support"]["retained_p50"] == 5.0


def test_even_cohort_ties_are_rejected_to_avoid_moving_edge_ghosts() -> None:
    depth = np.ones((6, 1, 2), dtype=np.float32)
    confidence = np.ones_like(depth)
    tied_mask = np.zeros_like(depth, dtype=np.uint8)
    tied_mask[:3, 0, 0] = 1
    tied_mask[3:, 0, 1] = 1

    tied = evaluate_fusion_cohort(
        depth,
        confidence,
        tied_mask,
        normalize_frame_scale=False,
    )

    assert tied["cohort"]["strict_majority_support"] == 4
    assert tied["cohort"]["required_support"] == 4
    assert tied["support"]["retained_full_frame_fraction"] == 0.0

    majority_mask = tied_mask.copy()
    majority_mask[3, 0, 0] = 1
    majority = evaluate_fusion_cohort(
        depth,
        confidence,
        majority_mask,
        normalize_frame_scale=False,
    )
    assert majority["support"]["retained_full_frame_fraction"] == 0.5
