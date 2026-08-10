from __future__ import annotations

import numpy as np

from tools.mapanything_phone_scan.build_consensus_fusion import (
    Consistency,
    Sequence,
    _common_intrinsics,
    _fuse_depth,
    _pose_graph,
    _remap_to_common_rays,
)


def _sequence(
    depth: np.ndarray,
    mask: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
) -> Sequence:
    count, height, width = depth.shape
    poses = np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))
    intrinsics = np.tile(
        np.asarray(
            [[80.0, 0.0, (width - 1) / 2], [0.0, 80.0, (height - 1) / 2], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        (count, 1, 1),
    )
    return Sequence(
        name="synthetic",
        depth=depth.astype(np.float32),
        confidence=(
            np.ones_like(depth, dtype=np.float32)
            if confidence is None
            else confidence.astype(np.float32)
        ),
        mask=(depth > 0 if mask is None else mask.astype(bool)),
        rgb=np.full((count, height, width, 3), 128, dtype=np.uint8),
        poses=poses,
        intrinsics=intrinsics,
    )


def test_common_ray_remap_is_identity_for_matching_intrinsics() -> None:
    depth = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
    sequence = _sequence(depth)
    common = _common_intrinsics(sequence, sequence, (3, 4))
    remapped = _remap_to_common_rays(sequence, common, (3, 4))
    np.testing.assert_allclose(remapped.depth, depth)
    np.testing.assert_array_equal(remapped.mask, sequence.mask)


def test_fusion_selects_consensus_consistent_surface_and_rejects_conflict() -> None:
    first_depth = np.asarray([[[1.00, 1.00, 1.00, 1.00]]], dtype=np.float32)
    second_depth = np.asarray([[[1.04, 1.20, 2.00, 0.00]]], dtype=np.float32)
    first_mask = np.asarray([[[True, True, True, True]]])
    second_mask = np.asarray([[[True, True, True, False]]])
    first = _sequence(first_depth, first_mask)
    second = _sequence(second_depth, second_mask)
    first_confidence = np.asarray([[[0.9, 0.7, 0.9, 0.9]]], dtype=np.float32)
    second_confidence = np.asarray([[[0.6, 0.9, 0.9, 0.0]]], dtype=np.float32)
    first_consistency = Consistency(
        score=np.asarray([[[0.8, 0.3, 0.9, 0.9]]], dtype=np.float32),
        support=np.ones((1, 1, 4), dtype=np.uint8),
        median_error_m=0.1,
        p80_error_m=0.2,
    )
    second_consistency = Consistency(
        score=np.asarray([[[0.7, 0.9, 0.9, 0.0]]], dtype=np.float32),
        support=np.ones((1, 1, 4), dtype=np.uint8),
        median_error_m=0.1,
        p80_error_m=0.2,
    )
    fused, _ = _fuse_depth(
        first,
        second,
        first_depth,
        second_depth,
        first_confidence,
        second_confidence,
        first_consistency,
        second_consistency,
    )
    assert fused["source"][0, 0, 0] == 1
    assert fused["source"][0, 0, 1] == 3
    assert fused["depth"][0, 0, 1] == second_depth[0, 0, 1]
    assert fused["source"][0, 0, 2] == 0
    assert fused["uncertain"][0, 0, 2]
    assert fused["source"][0, 0, 3] == 4


def test_pose_graph_soft_loop_closure_reduces_endpoint_drift() -> None:
    count = 7
    first = np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))
    second = first.copy()
    first[:, 0, 3] = np.asarray([0.0, 0.8, 1.7, 2.5, 1.8, 1.0, 0.75])
    second[:, 0, 3] = np.asarray([0.0, 0.9, 1.8, 2.6, 1.9, 1.1, 0.65])
    optimized, metrics = _pose_graph(first, second)
    assert metrics["solver_success"]
    assert metrics["start_end_after_m"] < metrics["start_end_before_m"]
    assert np.linalg.norm(optimized[-1, :3, 3]) < 0.75
