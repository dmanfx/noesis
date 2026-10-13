"""Surface-specific vote aggregation must not change strict room-join semantics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from tools.mapanything_phone_scan import build_pcf_review_surface_mesh as surface
from tools.mapanything_phone_scan.build_single_room_review_surface import _parser as single_room_parser
from tools.mapanything_phone_scan.reintegrate_pcf_rooms import ViewEvidence, _depth_visibility_evidence


POINTS = np.asarray([[0.0, 0.0, 2.0]])
K = np.asarray([[20.0, 0.0, 31.5], [0.0, 20.0, 31.5], [0.0, 0.0, 1.0]])


def _view(index: int, depth=2.0, quality=.9, *, mask=True, x=0.0, z=0.0):
    pose = np.eye(4)
    pose[0, 3], pose[2, 3] = x, z
    depth_array = np.full((64, 64), depth, dtype=np.float32)
    quality_array = (None if quality is None else np.full(depth_array.shape, quality, dtype=float)
                     if np.isscalar(quality) else np.asarray(quality))
    return surface.SurfaceViewEvidence(index, K.copy(), depth_array,
                                       np.full(depth_array.shape, mask, dtype=bool), pose, quality_array)


def _vote(views, points=POINTS, **kwargs):
    return surface._qualified_surface_depth_votes(points, views, **kwargs)


def test_two_qualified_supports_outweigh_one_qualified_farther_outlier():
    views = [_view(0, quality=.8), _view(1, quality=.9), _view(2, depth=3.0, quality=.7)]
    accepted, votes = _vote(views)
    assert accepted.tolist() == [True]
    np.testing.assert_array_equal(votes["support_count"], [2])
    np.testing.assert_array_equal(votes["free_count"], [1])
    np.testing.assert_allclose(votes["support_weight"], [1.7])
    np.testing.assert_allclose(votes["free_weight"], [.7])
    # The shared cross-room helper retains its original any-view contradiction.
    supported, _, free = _depth_visibility_evidence(
        POINTS, views, minimum_depth_m=.05, maximum_depth_m=12.0,
        absolute_tolerance_m=.05, relative_tolerance=.03,
    )
    assert supported.tolist() == free.tolist() == [True]


def test_qualified_free_majority_rejects_even_when_support_weight_is_larger():
    views = [_view(0, quality=.9), _view(1, quality=.9)]
    views.extend(_view(index, depth=3.0, quality=.35) for index in (2, 3, 4))
    accepted, votes = _vote(views)
    assert accepted.tolist() == [False]
    assert votes["support_count"][0] == 2 and votes["free_count"][0] == 3
    assert votes["support_weight"][0] > votes["free_weight"][0]


def test_weights_not_merely_counts_must_favor_support():
    accepted, votes = _vote([_view(0, quality=.35), _view(1, quality=.35),
                             _view(2, depth=3.0, quality=.95)])
    assert votes["support_count"][0] > votes["free_count"][0]
    assert votes["support_weight"][0] < votes["free_weight"][0]
    assert accepted.tolist() == [False]


def test_tied_count_or_weight_is_not_an_acceptance():
    accepted, votes = _vote([_view(0, quality=.8), _view(1, quality=.8),
                             _view(2, depth=3.0, quality=.35), _view(3, depth=3.0, quality=.35)])
    assert votes["support_count"][0] == votes["free_count"][0] == 2
    assert accepted.tolist() == [False]
    accepted, votes = _vote([_view(0, quality=.4), _view(1, quality=.4),
                             _view(2, depth=3.0, quality=.8)])
    assert votes["support_count"][0] > votes["free_count"][0]
    assert votes["support_weight"][0] == votes["free_weight"][0]
    assert accepted.tolist() == [False]


@pytest.mark.parametrize("extra", [
    _view(2, depth=3.0, quality=.349),
    _view(2, depth=3.0, quality=.9, mask=False),
    _view(2, depth=1.0, quality=.9),  # Foreground occludes the proposed point.
    _view(2, depth=3.0, quality=.9, x=100.0),  # Outside this view's image.
    _view(2, depth=3.0, quality=.9, z=3.0),  # Point is behind the camera.
    _view(2, depth=0.0, quality=.9),
    _view(2, depth=-1.0, quality=.9),
    _view(2, depth=13.0, quality=.9),
    _view(2, depth=np.nan, quality=.9),
    _view(2, depth=np.inf, quality=.9),
], ids=["weak", "masked", "occluded", "out_of_frustum", "behind_camera", "zero_depth",
        "negative_depth", "beyond_depth_bound", "nan_depth", "infinite_depth"])
def test_uncertain_or_unobserved_rays_abstain_not_veto(extra):
    accepted, votes = _vote([_view(0), _view(1), extra])
    assert accepted.tolist() == [True]
    assert votes["support_count"].tolist() == [2]
    assert votes["free_count"].tolist() == [0]
    np.testing.assert_allclose(votes["support_weight"], [1.8])
    np.testing.assert_array_equal(votes["free_weight"], [0.0])


def test_weak_positive_rays_cannot_supply_the_second_qualified_view():
    accepted, votes = _vote([_view(0), _view(1, quality=.349)])
    assert accepted.tolist() == [False]
    assert votes["support_count"].tolist() == [1]
    accepted, votes = _vote([_view(0, quality=.35), _view(1, quality=.35)])
    assert accepted.tolist() == [True]  # Preserve the existing inclusive 0.35 threshold.
    np.testing.assert_allclose(votes["support_weight"], [.7])


def test_duplicate_view_ids_are_not_extra_support_and_fail_closed():
    with pytest.raises(ValueError, match="distinct source-view identities"):
        _vote([_view(0), _view(0)])


@pytest.mark.parametrize("quality", [None, np.ones((63, 64)), np.full((64, 64), np.nan),
                                      np.full((64, 64), np.inf), -.01, 1.01],
                         ids=["missing", "wrong_shape", "nan", "infinite", "negative", "above_one"])
def test_missing_or_malformed_confidence_fails_closed(quality):
    with pytest.raises(ValueError, match="confidence"):
        _vote([_view(0), _view(1, quality=quality)])
    legacy = ViewEvidence(1, K.copy(), np.full((64, 64), 2.0), np.ones((64, 64), dtype=bool), np.eye(4))
    with pytest.raises(ValueError, match="recorded source confidence"):
        _vote([_view(0), legacy])


def test_loader_retains_actual_confidence_and_missing_confidence_is_not_fabricated(tmp_path: Path):
    raw = tmp_path / "source.npz"
    fields = {"depth_z": np.full((64, 64), 2.0), "mask": np.ones((64, 64), dtype=bool),
              "intrinsics": K, "camera_pose": np.eye(4)}
    confidence = np.full((64, 64), .42, dtype=np.float32)
    np.savez(raw, **fields, confidence=confidence)
    rows = [{"path": str(raw), "view_index": 0, "effective_world_from_local_row_major": np.eye(4).tolist()}]
    view = surface._load_visibility_evidence(rows)[0]
    np.testing.assert_array_equal(view.quality, confidence)
    np.savez(raw, **fields)
    view = surface._load_visibility_evidence(rows)[0]
    assert view.quality is None  # Legacy strict processing can still read the source.
    with pytest.raises(ValueError, match="recorded source confidence"):
        _vote([view, _view(1)])


def test_loader_rejects_malformed_confidence_even_if_depth_and_pose_are_valid(tmp_path: Path):
    raw = tmp_path / "source.npz"
    np.savez(raw, depth_z=np.full((64, 64), 2.0), mask=np.ones((64, 64), dtype=bool),
             confidence=np.full((64, 64), np.nan), intrinsics=K, camera_pose=np.eye(4))
    rows = [{"path": str(raw), "view_index": 0, "effective_world_from_local_row_major": np.eye(4).tolist()}]
    with pytest.raises(ValueError, match="malformed"):
        surface._load_visibility_evidence(rows)


def test_chunked_voting_is_bounded_and_has_identical_per_point_votes():
    points = np.tile(POINTS, (7, 1))
    views = [_view(0), _view(1), _view(2, depth=3.0, quality=.4)]
    first, counts = _vote(views, points, chunk_size=1)
    second, repeated = _vote(views, points, chunk_size=3)
    np.testing.assert_array_equal(first, second)
    for name in counts:
        np.testing.assert_array_equal(counts[name], repeated[name])
    for invalid in (0, 65537):
        with pytest.raises(ValueError, match="chunk size"):
            _vote(views, points, chunk_size=invalid)


def _triangle():
    vertices = np.asarray([[-.1, -.1, 2.0], [.1, -.1, 2.0], [0.0, .1, 2.0]])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                    o3d.utility.Vector3iVector([[0, 1, 2]]))
    return mesh, np.vstack((vertices, vertices.mean(0)))


def test_triangle_qualified_policy_recovers_measured_support_but_strict_default_is_unchanged():
    mesh, cloud = _triangle()
    views = [_view(0), _view(1), _view(2, depth=3.0, quality=.4)]
    strict, support_rejected, ray_rejected = surface._triangle_support_mask(mesh, cloud, views, .075)
    assert strict.tolist() == [False] and support_rejected == 0 and ray_rejected == 1
    explicit_strict, _, _ = surface._triangle_support_mask(mesh, cloud, views, .075, visibility_policy="strict_any_view")
    np.testing.assert_array_equal(strict, explicit_strict)
    qualified, support_rejected, ray_rejected = surface._triangle_support_mask(
        mesh, cloud, views, .075, visibility_policy="qualified_consensus",
    )
    assert qualified.tolist() == [True] and support_rejected == ray_rejected == 0


def test_triangle_voting_does_not_fill_without_nearby_measured_points_or_multiview_support():
    mesh, cloud = _triangle()
    supported = [_view(0), _view(1)]
    qualified, support_rejected, ray_rejected = surface._triangle_support_mask(
        mesh, cloud + [1.0, 0.0, 0.0], supported, .075, visibility_policy="qualified_consensus",
    )
    assert qualified.tolist() == [False] and support_rejected == 1 and ray_rejected == 0
    qualified, _, ray_rejected = surface._triangle_support_mask(
        mesh, cloud, [_view(0)], .075, visibility_policy="qualified_consensus",
    )
    assert qualified.tolist() == [False] and ray_rejected == 1
    strict, _, _ = surface._triangle_support_mask(mesh, cloud, [_view(0)], .075)
    assert strict.tolist() == [True]  # Existing default contract is not tightened globally.


def test_triangle_rejects_qualified_conflicts_missing_evidence_and_unknown_policy():
    mesh, cloud = _triangle()
    views = [_view(0), _view(1), _view(2, depth=3.0), _view(3, depth=3.0)]
    qualified, _, ray_rejected = surface._triangle_support_mask(
        mesh, cloud, views, .075, visibility_policy="qualified_consensus",
    )
    assert qualified.tolist() == [False] and ray_rejected == 1
    qualified, _, _ = surface._triangle_support_mask(mesh, cloud, [], .075, visibility_policy="qualified_consensus")
    assert qualified.tolist() == [False]
    with pytest.raises(ValueError, match="visibility policy"):
        surface._triangle_support_mask(mesh, cloud, views, .075, visibility_policy="unbound_fallback")


def test_single_room_cli_keeps_strict_policy_as_default_and_requires_explicit_opt_in():
    required = ["--consensus-dir", "unused", "--alignment-dir", "unused",
                "--room-name", "living", "--owner", "3", "--output-dir", "unused"]
    assert single_room_parser().parse_args(required).visibility_policy == "strict_any_view"
    assert single_room_parser().parse_args(required + ["--visibility-policy", "qualified_consensus"]).visibility_policy == "qualified_consensus"


def test_normal_orientation_uses_supporting_views_not_nearest_unseen_camera():
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(POINTS))
    cloud.normals = o3d.utility.Vector3dVector([[0., 0., 1.]])
    cameras = np.asarray([[0., 0., 2.1], [0., 0., 0.]])
    report = surface._orient_normals_to_supported_cameras(cloud, cameras, [_view(0), _view(1)])
    np.testing.assert_array_equal(np.asarray(cloud.normals), [[0., 0., -1.]])
    assert report["qualified_camera_normal_flip_count"] == 1
    assert report["qualified_support_oriented_point_count"] == 1
    assert report["unresolved_direction_fallback_count"] == 0


def test_normal_orientation_counts_unresolved_support_instead_of_inventing_it():
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(POINTS))
    cloud.normals = o3d.utility.Vector3dVector([[0., 0., 1.]])
    report = surface._orient_normals_to_supported_cameras(cloud, np.asarray([[0., 0., 2.1]]),
                                                         [_view(0), _view(1, quality=.34)])
    np.testing.assert_array_equal(np.asarray(cloud.normals), [[0., 0., 1.]])
    assert report["qualified_support_oriented_point_count"] == 0
    assert report["unresolved_direction_fallback_count"] == 1
