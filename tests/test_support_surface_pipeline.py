"""Baseline world-projection and BEV transport of admitted support planes."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from noesis.pipelines import hooks
from noesis.telemetry.person_ground_state import PersonGroundState, align_inferred_ground_observation
from noesis_core.contracts.base import Matrix3
from noesis_core.contracts.support_surface import (
    SupportSurface,
    SupportSurfaceReference,
    SupportSurfaceSet,
)
from noesis_core.contracts.world_measurement import WorldMeasurementCohort
from noesis_core.world.resolver import UniversalWorldMeasurementResolver


def _processor():
    return hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(config={}), tracking_pub=None,
        camera_labels={}, sensor_id_map={},
    )


def _collection(*, overlap=False):
    def plane(surface_id, elevation):
        return SupportSurface(
            surface_id=surface_id, revision="survey-1",
            normal_world=(0.0, 1.0, 0.0), offset_m=-elevation,
            extent_xz=((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)),
            residual_sigma_m=0.02, source_observation_count=8,
            source_kind="independent_static_metric_depth",
            source_artifact_sha256="a" * 64,
        )
    return SupportSurfaceSet(
        camera_id="test-camera", world_frame_revision="world-1",
        calibration_revision="cal-1", world_transform_sha256="b" * 64,
        evidence_sha256="a" * 64, authority="measured_calibration",
        surfaces=(plane("lower", 0.25), plane("upper", 0.5)) if overlap else (plane("lower", 0.25),),
    )


def _calibration(collection):
    return SimpleNamespace(
        support_surfaces=collection, camera_id="test-camera",
        world_frame_revision="world-1", camera_calibration_sha256="cal-1",
        frame_transform_sha256="b" * 64,
        image_localization_basis="revision_target_floor",
        image_size=(100, 100), extrinsics_col_major=tuple(np.eye(4).ravel(order="F")),
        intrinsics=np.eye(3), floor_y=0.0,
    )


def test_floor_ray_prefers_unique_measured_surface_and_rejects_overlap(monkeypatch):
    processor = _processor()
    monkeypatch.setattr(hooks, "parse_extrinsics", lambda *_: (np.eye(3), np.zeros(3)))
    monkeypatch.setattr(hooks, "ray_from_pixel", lambda *_: ((1.0, 2.0, 1.0), (0.0, -1.0, 0.0)))
    processor._project_pixel_to_world_observation = lambda *args, **kwargs: np.array([1.0, 0.0, 1.0])
    hit = processor._project_pixel_to_floor_world(
        _calibration(_collection()), 50, 50, flip_u=False, flip_v=False,
    )
    assert hit == pytest.approx((1.0, 0.25, 1.0))
    scaled = _calibration(_collection())
    scaled.unit_scale = 2.0
    assert processor._project_pixel_to_floor_world(
        scaled, 50, 50, flip_u=False, flip_v=False,
    ) == pytest.approx((2.0, 0.25, 2.0))
    assert processor._project_pixel_to_floor_world(
        _calibration(_collection(overlap=True)), 50, 50,
        flip_u=False, flip_v=False,
    ) is None
    stale = _calibration(_collection())
    stale.frame_transform_sha256 = "c" * 64
    assert processor._project_pixel_to_floor_world(
        stale, 50, 50, flip_u=False, flip_v=False,
    ) is None
    review = _collection().model_copy(update={"authority": "review_only"})
    assert processor._project_pixel_to_floor_world(
        _calibration(review), 50, 50, flip_u=False, flip_v=False,
    ) == pytest.approx((1.0, 0.0, 1.0))


def test_canonical_footpoint_transports_height_covariance_and_surface():
    processor = _processor()
    reference = SupportSurfaceReference.from_surface(_collection().surfaces[0])
    footpoint = processor._footpoint_from_track(
        {
            "class_id": 0, "tracker_id": 7, "world_valid": True,
            "world": [1.0, 0.25, 1.0], "world_frame": "backend_world_m",
            "world_frame_revision": "world-1", "image_foot": [40, 80],
            "world_covariance": [0.1, 0, 0, 0, 0.02, 0, 0, 0, 0.1],
            "world_support_surface": reference.model_dump(),
        }, (100, 100),
    )
    assert footpoint is not None
    assert footpoint.world_y == pytest.approx(0.25)
    assert footpoint.world_covariance[4] == pytest.approx(0.02)
    assert footpoint.support_surface["surface_id"] == "lower"


def test_queue_watermark_retains_surface_independently_of_local_filter():
    processor = _processor()
    key = (0, 7, 1, "backend_world_m", "world-1", "b" * 64, "a" * 64)
    reference = SupportSurfaceReference.from_surface(_collection().surfaces[0])
    processor._set_world_output_watermark(
        key, world_x=1., world_z=1., media_pts_ns=1_000_000_000,
        filter_ts=1., trail_segment_id=0, metric_authoritative=True,
        committed_world_source="pose_floor_only", support_surface=reference,
    )
    assert processor._world_output_watermarks[key].support_surface == reference
    # An accepted lower-floor transition replaces the reference explicitly;
    # the latest filter's plane must not be used to relabel old queue history.
    processor._set_world_output_watermark(
        key, world_x=3., world_z=1., media_pts_ns=1_100_000_000,
        filter_ts=1.1, trail_segment_id=1, metric_authoritative=True,
        committed_world_source="pose_floor_only", support_surface=None,
    )
    assert processor._world_output_watermarks[key].support_surface is None


@pytest.mark.parametrize("slope", [0., .1])
def test_inferred_process_preserves_named_surface_in_full_xyz_proof(slope):
    normal = np.array([-slope, 1., 0.])
    normal /= np.linalg.norm(normal)
    surface = SupportSurfaceReference.from_surface(_collection().surfaces[0]).model_copy(update={
        "normal_world": tuple(normal), "offset_m": -.25 * normal[1],
        "extent_xz": ((-30., -30.), (30., -30.), (30., 30.), (-30., 30.)),
    })
    state = PersonGroundState()
    common = dict(floor_y=0., lifecycle_generation=1, height_ref_scene=1.8,
                  world_frame_id="backend_world_m", world_frame_revision="world-1",
                  world_transform_sha256="b" * 64, support_surface=surface,
                  trusted_output_reference=(1., 2., 900_000_000, .9, 3))
    first = align_inferred_ground_observation(
        state, raw_measurement=np.array([10., 0., 20.]), now_ts=1.,
        observed_at_us=1_000_000, media_pts_ns=1_000_000_000, **common)
    second = align_inferred_ground_observation(
        state, raw_measurement=np.array([10.2, 0., 20.1]), now_ts=1.1,
        observed_at_us=1_100_000, media_pts_ns=1_100_000_000, **common)
    assert first is not None and second is not None
    point, anchor, delta, raw = second[:4]
    assert anchor is first[1]
    assert anchor.support_surface == surface
    assert anchor.trusted_world_origin == pytest.approx((1., .25 + slope, 2.))
    assert point == pytest.approx(np.array(anchor.trusted_world_origin) + delta)
    assert delta == pytest.approx(np.array(raw) - anchor.raw_origin)
    assert point[1] == pytest.approx(.25 + slope * point[0])


def test_upright_body_geometry_is_resolved_on_the_selected_floor(monkeypatch):
    processor = _processor()
    calib = _calibration(_collection())
    floors = []
    covariance = Matrix3(values=(.1, 0, 0, 0, .1, 0, 0, 0, .1))
    def solve(view, **kwargs):
        floors.append(view.floor_y)
        return (np.array([1., view.floor_y, 1.]), covariance, 1.7, .1, 5, True)
    monkeypatch.setattr(processor, "_upright_pose_ground_projection_on_floor", solve)
    kwargs = dict(kpts_abs=np.zeros((17, 3)), bbox=(0, 0, 50, 90),
                  state=None, flip_u=False, flip_v=False)
    result = processor._upright_pose_ground_projection(calib, **kwargs)
    assert floors == [0., .25]
    assert result[0] == pytest.approx((1., .25, 1.))
    assert calib.floor_y == 0.  # no mutation of shared camera calibration
    # If distinct bounded levels both explain the current body rays, the
    # previous track or a convenient nearest level must not break that tie.
    monkeypatch.setattr(processor, "_upright_pose_ground_projection_on_floor",
                        lambda view, **kw: (np.array([3. if view.floor_y == 0 else 1., view.floor_y, 1.]),
                                            covariance, 1.7, .1, 5, True))
    assert processor._upright_pose_ground_projection(calib, **kwargs) is None


def test_seated_measured_torso_plane_does_not_add_floor_rise_twice(monkeypatch):
    processor = _processor()
    calib = _calibration(_collection())
    state = SimpleNamespace(seated_torso_height_scene=1.1,
                            seated_torso_height_uncertainty_scene=0.)
    heights = []
    def project(view, u, v, height, **kwargs):
        heights.append(height)
        return np.array([1., 0., 1.])
    monkeypatch.setattr(processor, "_reference_plane_floor_world", project)
    result = processor._seated_pose_ground_projection(
        calib, torso_uv=(50., 50.), state=state, flip_u=False, flip_v=False)
    assert result is not None
    assert set(heights) == {1.1}
    assert result[0] == pytest.approx((1., .25, 1.))


def test_floor_covariance_uses_observed_plane_sigma_and_same_surface_differences():
    processor = _processor()
    collection = _collection()
    reference = SupportSurfaceReference.from_surface(collection.surfaces[0])
    center = np.array([1.0, 0.25, 1.0])
    processor._project_pixel_to_floor_world = lambda *args, **kwargs: center.copy()
    kwargs = dict(anchor_uv=(50.0, 50.0), floor_candidate=center,
                  incidence_sin=0.5, support_surface=reference,
                  posture="standing", occlusion_fraction=0.0,
                  flip_u=False, flip_v=False)
    low = processor._floor_candidate_covariance(_calibration(collection), **kwargs)
    assert low is not None
    noisier = collection.model_copy(update={"surfaces": (
        collection.surfaces[0].model_copy(update={"residual_sigma_m": 0.08}),
    )})
    high = processor._floor_candidate_covariance(_calibration(noisier), **kwargs)
    assert high is not None
    assert high.values[0] > low.values[0]
    processor._project_pixel_to_floor_world = (
        lambda *args, **kwargs: np.array([3.0, 0.25, 1.0])
    )
    assert processor._floor_candidate_covariance(_calibration(collection), **kwargs) is None


def test_registered_depth_covariance_keeps_image_height_and_support_height_distinct():
    processor = _processor()
    collection = _collection()
    surface = SupportSurfaceReference.from_surface(collection.surfaces[0])
    processor._project_pixel_to_world_observation = (
        lambda calib, u, v, **kw: np.array([u / 50., .8, v / 50.])
    )
    covariance = processor._depth_candidate_covariance(
        _calibration(collection), anchor_uv=(50., 50.), registered_depth_m=3.,
        depth_result=SimpleNamespace(anchor_valid_fraction=1., anchor_depth_spread_m=.1),
        depth_candidate=np.array([1., .25, 1.]), support_surface=surface,
        posture="sitting", occlusion_fraction=0., flip_u=False, flip_v=False,
    )
    assert covariance is not None
    assert all(math.isfinite(value) for value in covariance.values)


def test_floor_range_uses_observed_plane_incidence_not_global_y():
    processor = _processor()
    base = _collection()
    sloped = base.surfaces[0].model_copy(update={
        "normal_world": (0.0, 0.8, -0.6), "offset_m": 0.4,
    })
    calib = _calibration(base.model_copy(update={"surfaces": (sloped,)}))
    track = {}
    point = np.array([1.0, 0.25, 1.0])
    assert processor._admit_floor_ray_range(
        "test-camera", calib=calib, floor_candidate=point, track=track,
    )
    delta = point / np.linalg.norm(point)
    assert track["world_floor_incidence_sin"] == pytest.approx(
        abs(float(np.dot(sloped.normal_world, delta)))
    )
    assert track["world_floor_incidence_sin"] != pytest.approx(abs(float(delta[1])))


def test_resolver_diagnostics_keep_selected_surface_identity():
    processor = _processor()
    processor._world_resolver = UniversalWorldMeasurementResolver()
    processor._world_resolver_diagnostics_enabled = True
    cohort = WorldMeasurementCohort(
        track_key="test-camera:7:1", tracker_lifecycle_generation=1,
        camera_id="test-camera", source_id=0, tracker_id=7,
        frame_id=10, observed_at_us=1_000_010,
        world_revision="world-1", calibration_revision="cal-1",
        world_transform_sha256="b" * 64,
    )
    reference = SupportSurfaceReference.from_surface(_collection().surfaces[0])
    measurements = processor._build_world_measurement_set(
        cohort=cohort, candidates=({
            "candidate_id": "floor_ray", "kind": "floor_ray",
            "position": np.array([1.0, 0.25, 1.0]),
            "covariance": Matrix3(values=(0.05, 0, 0, 0, 0.02, 0, 0, 0, 0.05)),
            "anchor": "pose_ankle_floor", "confidence": 0.95,
            "support_state": "floor", "support_score": 0.95,
            "ray_incidence_sin": 0.9, "support_surface": reference,
        },),
    )
    assert measurements is not None
    assert measurements.hypotheses[0].support_surface == reference
    track = {}
    point, resolved = processor._apply_resolved_world_measurement(track, measurements)
    assert point is not None and resolved is not None
    assert resolved.support_surface == reference
    diagnostics = track["world_resolver"]
    assert diagnostics["support_surface"]["surface_id"] == "lower"
    assert diagnostics["resolved"]["support_surface"]["surface_id"] == "lower"
    assert diagnostics["candidates"][0]["support_surface"]["surface_id"] == "lower"


def test_bounded_proof_uses_plane_elevation_at_each_position():
    processor = _processor()
    ny = 1.0 / math.sqrt(1.01)
    reference = SupportSurfaceReference(
        surface_id="slope", revision="survey-1",
        normal_world=(0.0, ny, -0.1 * ny), offset_m=-0.2 * ny,
        extent_xz=((-20.0, -20.0), (20.0, -20.0), (20.0, 20.0), (-20.0, 20.0)),
    )
    proof = processor._bounded_process_output_provenance(
        {"media_pts_ns": 1_100_000_000, "tracker_lifecycle_generation": 1},
        provenance_type="bounded_cv_process", provenance_origin="last_metric_process",
        reason="measurement_unavailable",
        prior_output=(0.0, 0.0, 1_000_000_000, 1.0, 0),
        metric_output=(0.0, 0.0, 1_000_000_000, 1.0, 0),
        process_observation=(0.0, 0.0, 1.0),
        emitted_point=(0.0, 0.3, 1.0), output_continuous=True,
        emitted_trail_segment_id=0, world_frame_id="backend_world_m",
        world_frame_revision="world-1", world_transform_sha256="b" * 64,
        support_surface=reference,
    )
    assert proof is not None
    transition = proof["filter_transition"]
    assert transition["origin_world"][1] == pytest.approx(0.2)
    assert transition["metric_origin_world"][1] == pytest.approx(0.2)
    assert proof["process_observation"][1] == pytest.approx(0.3)
