from __future__ import annotations

import numpy as np

from tools.mapanything_phone_scan.build_consensus_fusion import (
    Consistency,
    Sequence,
    _common_intrinsics,
    _confidence_cdf,
    _fuse_depth,
    _heldout_reprojection,
    _multiview_consistency,
    _pose_graph,
    _remap_to_common_rays,
    _surfel_fusion,
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


def test_pose_graph_has_no_endpoint_pseudoloop_and_accepts_verified_edges() -> None:
    count = 7
    first = np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))
    second = first.copy()
    first[:, 0, 3] = np.asarray([0.0, 0.8, 1.7, 2.5, 1.8, 1.0, 0.75])
    second[:, 0, 3] = np.asarray([0.0, 0.9, 1.8, 2.6, 1.9, 1.1, 0.65])
    baseline, baseline_metrics = _pose_graph(first, second)
    optimized, metrics = _pose_graph(
        first,
        second,
        [
            {
                "source_view": 0,
                "target_view": 6,
                "transform": np.linalg.inv(second[6]) @ second[0],
                "translation_sigma_m": 0.05,
                "rotation_sigma_deg": 1.0,
                "label": "test_verified_loop",
            }
        ],
    )
    assert baseline_metrics["solver_success"]
    assert metrics["solver_success"]
    assert baseline_metrics["endpoint_constraint_applied"] is False
    assert metrics["endpoint_constraint_applied"] is False
    assert metrics["verified_constraint_count"] == 1
    assert np.linalg.norm(optimized[-1, :3, 3]) < np.linalg.norm(baseline[-1, :3, 3])


def test_fusion_preserves_one_sharp_rgb_projection_for_landmarks() -> None:
    depth = np.ones((1, 32, 48), dtype=np.float32)
    first, second = _sequence(depth), _sequence(depth)
    first.name = "reference-rgb"
    first.rgb[:] = 0
    second.rgb[:] = 0
    # Different inferred camera rays put the same recorded edge at different
    # pixels. Averaging would create two half-contrast landmarks for PnP.
    first.rgb[:, :, 12:14] = 255
    second.rgb[:, :, 20:22] = 255
    consistency = Consistency(
        score=np.ones_like(depth),
        support=np.ones_like(depth, dtype=np.uint8),
        median_error_m=0.0,
        p80_error_m=0.0,
    )
    fused, metrics = _fuse_depth(
        first, second, depth, depth, np.ones_like(depth), np.ones_like(depth),
        consistency, consistency,
    )
    assert fused["rgb"][0, 16, 12, 0] == 255
    assert fused["rgb"][0, 16, 20, 0] == 0
    np.testing.assert_array_equal(fused["depth"], depth)
    assert fused["mask"].all()
    assert metrics["rgb_projection_source"] == "reference-rgb"


def test_confidence_cdf_is_an_uncalibrated_rank_score() -> None:
    sequence = _sequence(np.ones((1, 2, 3), dtype=np.float32))
    sequence.confidence[0] = np.asarray([[1.0, 2.0, 4.0], [1.5, 3.0, 5.0]])
    scores, metrics = _confidence_cdf(sequence)

    assert metrics["status"] == "ok"
    assert metrics["method"] == "within_model_empirical_cdf_rank_score_33_quantiles"
    assert "not_a_probability" in metrics["semantics"]
    assert np.all(np.diff(scores[0]) >= 0)
    assert np.all((scores >= 0.01) & (scores <= 0.99))


def test_correlated_agreement_has_no_independence_bonus() -> None:
    depth = np.ones((1, 1, 4), dtype=np.float32)
    first = _sequence(depth)
    second = _sequence(depth)
    first_consistency = Consistency(
        score=np.full_like(depth, 0.8),
        support=np.ones_like(depth, dtype=np.uint8),
        median_error_m=0.01,
        p80_error_m=0.02,
    )
    second_consistency = Consistency(
        score=np.full_like(depth, 0.2),
        support=np.ones_like(depth, dtype=np.uint8),
        median_error_m=0.03,
        p80_error_m=0.04,
    )
    fused, metrics = _fuse_depth(
        first,
        second,
        depth,
        depth,
        np.full_like(depth, 0.9),
        np.full_like(depth, 0.2),
        first_consistency,
        second_consistency,
    )

    maximum_source_weight = float(np.max(fused["first_evidence_weight"]))
    assert np.max(fused["quality"]) <= maximum_source_weight + 1e-6
    second_weight = float(np.max(fused["second_evidence_weight"]))
    legacy_bonus_reference = 0.5 * (maximum_source_weight + second_weight) + 0.2
    assert maximum_source_weight > second_weight + 0.4
    assert float(np.max(fused["quality"])) > legacy_bonus_reference
    assert metrics["evidence_relationship"] == "da3_conditioned_mapanything"
    assert "without_independence_bonus" in metrics["agreement_quality_combination"]


def test_surfel_support_requires_two_distinct_views() -> None:
    depth = np.ones((1, 1, 4), dtype=np.float32)
    intrinsics = np.tile(
        np.asarray(
            [[80.0, 0.0, 1.5], [0.0, 80.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        (1, 1, 1),
    )
    poses = np.tile(np.eye(4, dtype=np.float64), (1, 1, 1))
    fused = {
        "depth": depth,
        "mask": np.ones_like(depth, dtype=bool),
        "quality": np.ones_like(depth, dtype=np.float32),
        "rgb": np.full((1, 1, 4, 3), 128, dtype=np.uint8),
    }
    one_view, one_colors, one_weights, one_metrics = _surfel_fusion(
        fused, intrinsics, poses, 0.04
    )
    assert one_view.shape[0] == 0
    assert one_colors.shape == (0, 3)
    assert one_weights.shape == (0,)
    assert one_metrics["pre_filter_single_view_voxel_count"] > 0
    assert one_metrics["minimum_distinct_views_per_surfel"] == 2

    two_view_fused = {
        key: np.concatenate((value, value), axis=0) for key, value in fused.items()
    }
    two_view_intrinsics = np.concatenate((intrinsics, intrinsics), axis=0)
    two_view_poses = np.concatenate((poses, poses), axis=0)
    two_view, _, _, two_metrics = _surfel_fusion(
        two_view_fused, two_view_intrinsics, two_view_poses, 0.04
    )
    assert two_view.shape[0] > 0
    assert two_metrics["support_view_count_min"] >= 2
    assert two_metrics["support_view_count_max"] == 2


def test_heldout_reprojection_counts_large_errors_and_reports_coverage() -> None:
    depth = np.ones((2, 1, 12), dtype=np.float32)
    depth[1] = 5.0
    mask = np.ones_like(depth, dtype=bool)
    intrinsics = np.tile(
        np.asarray(
            [[80.0, 0.0, 5.5], [0.0, 80.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        (2, 1, 1),
    )
    poses = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
    result = _heldout_reprojection(depth, mask, intrinsics, poses)

    assert result["evaluation_type"] == "internal_same_inference_even_to_odd_consistency"
    assert result["finite_comparison_count"] > 0
    assert result["large_error_count_gt_2m"] == result["finite_comparison_count"]
    assert result["large_error_fraction_gt_2m"] == 1.0
    assert result["odd_frame_valid_pixel_coverage_fraction"] > 0.0


def test_heldout_reprojection_empty_input_is_explicit() -> None:
    depth = np.ones((2, 2, 2), dtype=np.float32)
    mask = np.zeros_like(depth, dtype=bool)
    intrinsics = np.tile(np.eye(3, dtype=np.float64), (2, 1, 1))
    poses = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
    result = _heldout_reprojection(depth, mask, intrinsics, poses)

    assert result["status"] == "empty_no_finite_comparisons"
    assert result["finite_comparison_count"] == 0
    assert result["even_frame_map_to_odd_frame_depth_median_m"] is None
    assert result["skipped_odd_frame_count"] == 1


def test_nonfinite_depth_is_excluded_from_consistency_and_fusion() -> None:
    depth = np.ones((2, 2, 2), dtype=np.float32)
    depth[0, 0, 0] = np.nan
    mask = np.ones_like(depth, dtype=bool)
    intrinsics = np.tile(np.eye(3, dtype=np.float64), (2, 1, 1))
    poses = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
    consistency = _multiview_consistency(depth, mask, intrinsics, poses)

    assert consistency.valid_pixel_count == 7
    assert consistency.median_error_m is not None
    assert np.isfinite(consistency.median_error_m)

    first = _sequence(depth)
    second = _sequence(np.nan_to_num(depth, nan=1.0))
    first_consistency = Consistency(
        score=np.ones_like(depth),
        support=np.ones_like(depth, dtype=np.uint8),
        median_error_m=0.0,
        p80_error_m=0.0,
    )
    second_consistency = first_consistency
    fused, metrics = _fuse_depth(
        first,
        second,
        depth,
        second.depth,
        np.ones_like(depth),
        np.ones_like(depth),
        first_consistency,
        second_consistency,
    )
    assert np.isfinite(fused["quality"]).all()
    assert metrics["absolute_depth_disagreement_median_m"] == 0.0


def test_refined_fusion_requires_capture_pose_and_file_identity(tmp_path) -> None:
    import hashlib
    import json
    import pytest
    from tools.mapanything_phone_scan.build_consensus_fusion import _load_refined_consensus_poses

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    scan = tmp_path / 'scan'
    scan.mkdir()
    prepared = scan / 'prepared_frames_manifest.json'
    prepared.write_text('{}')
    source_root = tmp_path / 'source'
    raw = source_root / 'raw'
    raw.mkdir(parents=True)
    frame = 'consensus_phone_metric_world_unaligned_to_noesis'
    source_manifest = source_root / 'scan_outputs_manifest.json'
    source_manifest.write_text(json.dumps({'provider': 'consensus_fusion', 'coordinate_frame': frame}))
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[1, 0, 3] = 0.2
    intrinsics = np.tile(np.eye(3), (2, 1, 1))
    raw_hashes = []
    for index in range(2):
        path = raw / f'view_{index:04d}.npz'
        np.savez(path, camera_pose=poses[index], intrinsics=intrinsics[index])
        raw_hashes.append({'index': index, 'sha256': sha(path)})
    refined = poses.copy()
    refined[1, 0, 3] = 0.21
    solution = tmp_path / 'camera_solution.npz'
    np.savez(solution, camera_to_world=refined, source_camera_to_world=poses,
             intrinsics=intrinsics, coordinate_frame=np.asarray(frame))
    report = {
        'schema': 'noesis.phone_walk.trajectory_refinement.v1',
        'source_identity': {'provider': 'consensus_fusion', 'coordinate_frame': frame,
                            'manifest': str(source_manifest), 'manifest_sha256': sha(source_manifest)},
        'source_raw': str(raw),
        'provenance': {'prepared_manifest_sha256': sha(prepared), 'raw_view_sha256': raw_hashes},
        'refinement': {'raw_materialized': True, 'scale_change': 1.0, 'gauge_change': False,
                       'accepted_constraint_count': 1,
                       'withheld_evaluation': {'status': 'passed', 'verified_constraint_count': 1},
                       'materialized': {'camera_solution': str(solution), 'camera_solution_sha256': sha(solution)}},
    }
    report_path = tmp_path / 'report.json'
    report_path.write_text(json.dumps(report))
    result, provenance = _load_refined_consensus_poses(report_path, scan, poses, intrinsics)
    np.testing.assert_array_equal(result, refined)
    assert 'depth_selection' in provenance['recomputed_after_pose_change']
    wrong_poses = poses.copy()
    wrong_poses[1, 0, 3] += 0.5
    with pytest.raises(ValueError, match='poses or rays'):
        _load_refined_consensus_poses(report_path, scan, wrong_poses, intrinsics)
    report['refinement']['withheld_evaluation']['status'] = 'failed'
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='temporal holdouts'):
        _load_refined_consensus_poses(report_path, scan, poses, intrinsics)
    report['refinement']['withheld_evaluation']['status'] = 'passed'
    report_path.write_text(json.dumps(report))
    with solution.open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='solution changed'):
        _load_refined_consensus_poses(report_path, scan, poses, intrinsics)
