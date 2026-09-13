from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tools.mapanything_phone_scan.evaluate_mapanything_prior_variants import (
    _bounded_source,
    _independent_evaluation_metrics,
    _load_independent_evaluation,
    _parse_args,
    _pose_carrier_alignment_mode,
    _pose_metrics,
    Candidate,
)
from tools.mapanything_phone_scan.render_phone_heatmap_diagnostics import PhoneCloud
from tools.mapanything_phone_scan.build_conditioned_scene_prior_bundle import (
    _conditioned_admission_checks,
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


def test_evaluator_has_no_room_specific_bev_rotation() -> None:
    source = Path(
        "tools/mapanything_phone_scan/evaluate_mapanything_prior_variants.py"
    ).read_text(encoding="utf-8")
    assert "np.rot90" not in source
    assert "rotate_180" not in source
    assert "foyer/hallway" not in source


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


def test_pose_carrier_alignment_mode_accepts_backend_world(tmp_path: Path) -> None:
    variant = tmp_path / "variant"
    variant.mkdir()
    (variant / "variant_manifest.json").write_text(
        '{"coordinate_frame":"backend_world_m_stream_points"}',
        encoding="utf-8",
    )

    assert _pose_carrier_alignment_mode(variant) == "identity"


def test_pose_carrier_alignment_mode_applies_late_da3_alignment(
    tmp_path: Path,
) -> None:
    variant = tmp_path / "variant"
    variant.mkdir()
    (variant / "variant_manifest.json").write_text(
        '{"coordinate_frame":"da3_metric_world_unaligned_to_noesis"}',
        encoding="utf-8",
    )

    assert _pose_carrier_alignment_mode(variant) == "world_from_da3"


def test_independent_evaluation_requires_declared_frame_units_and_provenance(
    tmp_path: Path,
) -> None:
    points_path = tmp_path / "withheld.npy"
    np.save(points_path, np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    manifest_path = tmp_path / "independent.json"
    manifest_path.write_text(
        """{
          "schema": "noesis.phone_walk.independent_evaluation.v1",
          "measurement_id": "survey-1",
          "coordinate_frame": "backend_world_m_stream_points",
          "units": "m",
          "source_identity": "survey-lidar-1",
          "provenance": {"capture": "withheld-visit"},
          "frame_identity": {"frame_id": "backend_world_m_stream_points", "revision": "living-room-r1", "registration_fingerprint": "sha256:test"},
          "points_file": "withheld.npy",
          "alignment_fitted_on_evaluation_points": false
        }""",
        encoding="utf-8",
    )

    measurements = _load_independent_evaluation(manifest_path)
    assert len(measurements) == 1
    assert measurements[0].measurement_id == "survey-1"
    np.testing.assert_allclose(measurements[0].points, [[0, 0, 0], [1, 0, 0]])

    with np.testing.assert_raises(ValueError):
        _load_independent_evaluation(
            manifest_path,
            expected_frame_identity={
                "frame_id": "backend_world_m_stream_points",
                "revision": "kitchen-r1",
                "registration_fingerprint": "sha256:test",
            },
        )

    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            '"coordinate_frame": "backend_world_m_stream_points"',
            '"coordinate_frame": "unknown_frame"',
        ),
        encoding="utf-8",
    )
    with np.testing.assert_raises(ValueError):
        _load_independent_evaluation(manifest_path)


def test_independent_evaluation_rejects_validation_alignment_fit(tmp_path: Path) -> None:
    points_path = tmp_path / "withheld.npz"
    np.savez(points_path, points=np.zeros((2, 3), dtype=np.float64))
    manifest_path = tmp_path / "independent.json"
    manifest_path.write_text(
        """{
          "schema": "noesis.phone_walk.independent_evaluation.v1",
          "measurement_id": "survey-1",
          "coordinate_frame": "backend_world_m_stream_points",
          "units": "m",
          "source_identity": "survey-lidar-1",
          "provenance": {"capture": "withheld-visit"},
          "frame_identity": {"frame_id": "backend_world_m_stream_points", "revision": "living-room-r1", "registration_fingerprint": "sha256:test"},
          "points_file": "withheld.npz",
          "alignment_fitted_on_evaluation_points": true
        }""",
        encoding="utf-8",
    )
    with np.testing.assert_raises(ValueError):
        _load_independent_evaluation(manifest_path)


def test_independent_evaluation_reports_accuracy_and_coverage(tmp_path: Path) -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    cloud = PhoneCloud(
        points=points.copy(),
        colors=np.zeros((2, 3), dtype=np.uint8),
        weights=np.ones(2, dtype=np.float32),
        ranges=np.ones(2, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float64),
        frame_zero_depth=np.ones((1, 1), dtype=np.float32),
        frame_zero_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
    )
    candidate = Candidate(
        slug="candidate",
        label="candidate",
        raw_root=tmp_path,
        cloud=cloud,
        sequence=object(),  # type: ignore[arg-type]
        aligned_poses=np.eye(4, dtype=np.float64)[None],
        alignment_scale=1.0,
        alignment_rotation=np.eye(3, dtype=np.float64),
        alignment_translation=np.zeros(3, dtype=np.float64),
        alignment_method="test",
    )
    measurement = type("Measurement", (), {
        "measurement_id": "survey-1",
        "points": points,
        "coordinate_frame": "backend_world_m_stream_points",
        "units": "m",
            "source_identity": "survey-lidar-1",
            "provenance": {"capture": "withheld-visit"},
            "frame_identity": {"frame_id": "backend_world_m_stream_points", "revision": "living-room-r1", "registration_fingerprint": "sha256:test"},
            "manifest_path": tmp_path / "independent.json",
    })()

    metrics = _independent_evaluation_metrics(candidate, [measurement])
    assert metrics["independent"] is True
    assert metrics["alignment_fitted_on_evaluation_points"] is False
    assert metrics["measurements"][0]["candidate_metrics"]["source_overlap_0_30m"] == 1.0


def _candidate_admission_metrics() -> dict[str, object]:
    return {
        "static_cloud_room_bounds_metrics": {"target_overlap_0_30m": 0.41},
        "fixed_camera_visible_cloud_metrics": {
            "source_point_count": 100_000,
            "comparable_point_count": 8_000,
            "source_overlap_0_30m": 0.90,
        },
        "fixed_camera_visible_structure_metrics": {
            "source_point_count": 4_000,
            "comparable_point_count": 600,
            "source_overlap_0_30m": 0.80,
            "plane_residual_median_m": 0.08,
        },
    }


def test_conditioned_admission_uses_static_camera_visible_domain() -> None:
    checks = _conditioned_admission_checks(
        _candidate_admission_metrics(),
        metric_scale_preserved=True,
    )

    assert all(checks.values())


def test_conditioned_admission_rejects_low_visible_support() -> None:
    candidate = _candidate_admission_metrics()
    candidate["fixed_camera_visible_cloud_metrics"]["comparable_point_count"] = 4_999  # type: ignore[index]

    checks = _conditioned_admission_checks(
        candidate,
        metric_scale_preserved=True,
    )

    assert checks["fixed_camera_visible_support_admitted"] is False


def test_conditioned_admission_rejects_bad_vertical_geometry() -> None:
    candidate = _candidate_admission_metrics()
    candidate["fixed_camera_visible_structure_metrics"]["plane_residual_median_m"] = 0.11  # type: ignore[index]

    checks = _conditioned_admission_checks(
        candidate,
        metric_scale_preserved=True,
    )

    assert checks["fixed_camera_vertical_structure_admitted"] is False
