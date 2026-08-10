from __future__ import annotations

import sys

import numpy as np

from tools.mapanything_phone_scan.evaluate_mapanything_prior_variants import (
    _bounded_source,
    _camera_oriented_bev,
    _parse_args,
    _pose_metrics,
)


def test_bounded_source_uses_expanded_target_box() -> None:
    target = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    source = np.asarray(
        [[-0.2, 1.0, 0.0]] * 1_000 + [[10.0, 1.0, 0.0]] * 100,
        dtype=np.float64,
    )
    bounded = _bounded_source(source, target, 0.25)
    assert bounded.shape == (1_000, 3)


def test_pose_metrics_reports_translation_and_closure() -> None:
    target = np.repeat(np.eye(4)[None], 3, axis=0)
    poses = target.copy()
    poses[:, 0, 3] = [0.0, 0.2, 0.5]
    metrics = _pose_metrics(poses, target)
    assert metrics["position_error_m"]["median"] == 0.2
    assert metrics["position_error_m"]["max"] == 0.5
    assert metrics["start_end_distance_m"] == 0.5


def test_camera_oriented_bev_rotates_hallway_convention_180_degrees() -> None:
    image = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(
        _camera_oriented_bev(image),
        np.asarray([[11, 10, 9, 8], [7, 6, 5, 4], [3, 2, 1, 0]]),
    )


def test_parse_args_accepts_room_specific_inputs_and_selected_variant(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_mapanything_prior_variants.py",
            "scan",
            "--suite-root",
            "suite",
            "--da3-raw",
            "da3/raw",
            "--prior-consensus-raw",
            "consensus/raw",
            "--variants",
            "da3_pose_sparse_depth",
            "--world-from-da3",
            "alignment.json",
            "--target-revision",
            "revision",
            "--calibration",
            "calibration.json",
            "--camera",
            "foyer",
            "--output-dir",
            "evaluation",
        ],
    )

    args = _parse_args()

    assert str(args.da3_raw) == "da3/raw"
    assert str(args.prior_consensus_raw) == "consensus/raw"
    assert args.variants == ["da3_pose_sparse_depth"]
    assert args.camera == "foyer"
