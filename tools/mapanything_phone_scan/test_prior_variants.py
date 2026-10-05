from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import pytest

from tools.mapanything_phone_scan.run_mapanything_prior_variants import (
    _backproject_depth,
    _resolve_carrier_frame,
    _rotation_error_deg,
    _top_up_reliable_samples,
    _top_up_preprocessed_depth_priors,
    _run_conditioned_mapanything_windows,
    _transform_poses,
    _zbuffer_depth,
)
from tools.mapanything_phone_scan.inference import MapAnythingScanError


def _crop_prior_fixture(views, *, verbose=False):
    return [{"img": row["img"][:, 1:].copy(),
             "intrinsics": row["intrinsics"].copy(),
             "depth_z": row["depth_z"][None, :, 1:].copy()}
            for row in views]


def test_processed_prior_top_up_restores_crop_losses_from_reliable_only():
    reliable = np.asarray([[2, 2, 0, 2], [2, 0, 2, 0]], dtype=np.float32)
    sampled = np.asarray([[2, 2, 0, 0], [0, 0, 2, 0]], dtype=np.float32)
    raw = [{"img": np.zeros((2, 4, 3), np.uint8),
            "intrinsics": np.eye(3), "depth_z": sampled}]
    first, second = _crop_prior_fixture(raw), _crop_prior_fixture(raw)
    before = first[0]["depth_z"].copy()
    kwargs = dict(reliable_depth_priors=[reliable], preprocess_inputs=_crop_prior_fixture,
                  global_start=13, random_seed=42, minimum_per_view=3)
    report = _top_up_preprocessed_depth_priors(first, raw, **kwargs)
    _top_up_preprocessed_depth_priors(second, raw, **kwargs)
    np.testing.assert_array_equal(first[0]["depth_z"], second[0]["depth_z"])
    np.testing.assert_array_equal(first[0]["depth_z"][before > 0], before[before > 0])
    assert np.all((first[0]["depth_z"] == 0) | (reliable[None, :, 1:] > 0))
    assert report[0]["samples_before"] == 2
    assert report[0]["added_reliable_samples"] == 1
    assert report[0]["samples_after"] == 3
    assert report[0]["global_view_index"] == 13


def test_processed_prior_top_up_fails_closed_when_crop_has_insufficient_evidence():
    raw = [{"img": np.zeros((2, 4, 3), np.uint8), "intrinsics": np.eye(3),
            "depth_z": np.asarray([[2, 2, 0, 0], [2, 0, 0, 0]], np.float32)}]
    processed = _crop_prior_fixture(raw)
    original = processed[0]["depth_z"].copy()
    with pytest.raises(MapAnythingScanError, match="after official preprocessing"):
        _top_up_preprocessed_depth_priors(
            processed, raw, reliable_depth_priors=[raw[0]["depth_z"]],
            preprocess_inputs=_crop_prior_fixture, global_start=0, random_seed=42,
            minimum_per_view=3)
    np.testing.assert_array_equal(processed[0]["depth_z"], original)


def test_processed_prior_top_up_rejects_changed_camera_or_existing_depth():
    raw = [{"img": np.zeros((2, 4, 3), np.uint8), "intrinsics": np.eye(3),
            "depth_z": np.asarray([[2, 2, 0, 0], [0, 0, 2, 0]], np.float32)}]
    def changed_camera(views, *, verbose=False):
        result = _crop_prior_fixture(views)
        result[0]["intrinsics"][0, 0] = 3
        return result
    with pytest.raises(MapAnythingScanError, match="changed intrinsics"):
        _top_up_preprocessed_depth_priors(
            _crop_prior_fixture(raw), raw, reliable_depth_priors=[np.full((2, 4), 2)],
            preprocess_inputs=changed_camera, global_start=0, random_seed=42,
            minimum_per_view=3)
    with pytest.raises(MapAnythingScanError, match="existing sampled depths"):
        _top_up_preprocessed_depth_priors(
            _crop_prior_fixture(raw), raw, reliable_depth_priors=[np.full((2, 4), 4)],
            preprocess_inputs=_crop_prior_fixture, global_start=0, random_seed=42,
            minimum_per_view=3)


def test_processed_prior_top_up_does_not_replay_already_sufficient_views():
    rows = [{"depth_z": np.ones((1, 2, 3), np.float32)}]
    def unexpected(*args, **kwargs):
        raise AssertionError("sufficient priors must remain unchanged")
    assert _top_up_preprocessed_depth_priors(
        rows, [], reliable_depth_priors=None, preprocess_inputs=unexpected,
        global_start=0, random_seed=42, minimum_per_view=3) == []


def test_window_consumer_receives_and_retains_restored_500_sample_minimum():
    depth = np.zeros((20, 32), np.float32)
    depth.reshape(-1)[:500] = 2
    raw = [{"img": np.zeros((20, 32, 3), np.uint8), "intrinsics": np.eye(3),
            "depth_z": depth},
           {"img": np.zeros((20, 32, 3), np.uint8), "intrinsics": np.eye(3),
            "depth_z": np.full((20, 32), 2, np.float32)}]
    class Model:
        def infer(self, processed, **kwargs):
            assert [np.count_nonzero(row["depth_z"]) for row in processed] == [500, 620]
            return [{"depth_z": row["depth_z"].copy(), "mask": row["depth_z"] > 0}
                    for row in processed]
    fake_torch = SimpleNamespace(inference_mode=nullcontext,
                                 cuda=SimpleNamespace(is_available=lambda: False))
    _, retained, report = _run_conditioned_mapanything_windows(
        Model(), raw, preprocess_inputs=_crop_prior_fixture, torch=fake_torch,
        amp_dtype="bf16", max_joint_views=16, window_overlap_views=4,
        require_depth_prior=True,
        reliable_depth_priors=[np.full((20, 32), 2, np.float32) for _ in raw])
    assert np.count_nonzero(retained[0]["depth_z"]) == 500
    top_up = report["windows"][0]["processed_sparse_prior_top_up"]
    assert top_up[0]["samples_after"] == 500
    assert top_up[0]["added_reliable_samples"] > 0


def test_missing_world_alignment_selects_explicit_da3_native_carrier() -> None:
    transform, frame, source = _resolve_carrier_frame(None)
    np.testing.assert_array_equal(transform, np.eye(4, dtype=np.float64))
    assert frame == "da3_metric_world_unaligned_to_noesis"
    assert source is None


def test_sparse_prior_top_up_uses_only_reliable_samples_deterministically() -> None:
    reliable = np.asarray(
        [
            [[True, True, True], [False, False, False]],
            [[True, False, True], [True, False, True]],
        ]
    )
    sampled = np.zeros_like(reliable)
    first, views, added = _top_up_reliable_samples(
        reliable,
        sampled,
        rng=np.random.default_rng(42),
        minimum_per_view=2,
    )
    second, _, _ = _top_up_reliable_samples(
        reliable,
        sampled,
        rng=np.random.default_rng(42),
        minimum_per_view=2,
    )
    np.testing.assert_array_equal(first, second)
    assert views == [0, 1]
    assert added == 4
    assert np.all(~first | reliable)
    np.testing.assert_array_equal(np.count_nonzero(first, axis=(1, 2)), [2, 2])


def test_backproject_depth_uses_opencv_z_depth_and_cam2world() -> None:
    depth = np.full((2, 3), 2.0, dtype=np.float32)
    intrinsics = np.asarray(
        [[2.0, 0.0, 1.0], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [10.0, 20.0, 30.0]
    points = _backproject_depth(depth, intrinsics, pose)
    np.testing.assert_allclose(points[0, 1], [10.0, 19.5, 32.0], atol=1e-6)
    np.testing.assert_allclose(points[1, 2], [11.0, 20.5, 32.0], atol=1e-6)


def test_transform_poses_left_multiplies_world_frame() -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [4.0, -2.0, 7.0]
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    poses[1, :3, 3] = [1.0, 2.0, 3.0]
    transformed = _transform_poses(transform, poses)
    np.testing.assert_allclose(transformed[0, :3, 3], [4.0, -2.0, 7.0])
    np.testing.assert_allclose(transformed[1, :3, 3], [5.0, 0.0, 10.0])


def test_rotation_error_is_geodesic_angle() -> None:
    first = np.eye(4, dtype=np.float64)
    second = np.eye(4, dtype=np.float64)
    second[:3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    assert abs(_rotation_error_deg(first, second) - 90.0) < 1e-8


def test_zbuffer_keeps_nearest_point_per_pixel() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 1.5],
            [1.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    intrinsics = np.asarray(
        [[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    depth, count = _zbuffer_depth(
        points,
        np.eye(4, dtype=np.float64),
        intrinsics,
        (5, 5),
    )
    assert count == 2
    assert depth[2, 2] == 1.5
    assert depth[2, 3] == 2.0


def test_zbuffer_returns_empty_depth_when_all_points_are_behind_camera() -> None:
    depth, count = _zbuffer_depth(
        np.asarray([[0.0, 0.0, -1.0]], dtype=np.float64),
        np.eye(4, dtype=np.float64),
        np.eye(3, dtype=np.float64),
        (5, 5),
    )

    assert count == 0
    assert not np.any(depth)
