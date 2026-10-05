"""Canonical world-service proof checks for a named sloped support plane."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from noesis.pipelines.hooks import _AnalyticsTelemetryProcessor
from noesis_core.contracts.support_surface import SupportSurfaceReference

_source = Path(__file__).with_name("test_noesis_core_world_service.py")
_spec = importlib.util.spec_from_file_location("_world_service_test_helpers", _source)
assert _spec is not None and _spec.loader is not None
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_bbox_affine_process_track = _helpers._bbox_affine_process_track
_bounded_process_track = _helpers._bounded_process_track
_retimed_bounded_output_hold = _helpers._retimed_bounded_output_hold
_service = _helpers._service
_set_image_motion_origin = _helpers._set_image_motion_origin
_track = _helpers._track


SLOPE = SupportSurfaceReference(
    surface_id="foyer-survey", revision="survey-1",
    normal_world=(-0.09950371902099892, 0.9950371902099892, 0.0),
    offset_m=0.0,
    extent_xz=((-20.0, -20.0), (20.0, -20.0), (20.0, 20.0), (-20.0, 20.0)),
)


def _on_slope(track, surface=SLOPE):
    track["world_support_surface"] = surface.model_dump()

    def elevate(vector):
        if isinstance(vector, list) and len(vector) >= 3:
            vector[1] = 0.1 * float(vector[0])

    for key in ("world", "world_filter_prediction", "world_floor_candidate"):
        elevate(track.get(key))
    provenance = track.get("world_prediction_provenance")
    if isinstance(provenance, dict):
        for key in ("process_observation", "projective_origin_world"):
            elevate(provenance.get(key))
        transition = provenance.get("filter_transition")
        if isinstance(transition, dict):
            for key in (
                "origin_world", "metric_origin_world", "position_base",
                "raw_position_target", "filter_observation", "visible_origin_world",
            ):
                elevate(transition.get(key))
    return track


def _origin():
    return _on_slope(_set_image_motion_origin(_track(
        camera_id="kitchen", tracker_id=7, frame_id=1,
        observed_at_us=1_900_000, x=1.0,
    )))


def test_named_slope_fresh_metric_to_cv_and_exact_hold():
    service = _service()
    assert service.publish(0, [_origin()], metadata={}).observations[0].payload.world is not None
    cv = _on_slope(_bounded_process_track())
    cv_publication = service.publish(0, [cv], metadata={})
    assert cv_publication.observations[0].payload.world is not None
    assert cv_publication.observations[0].payload.world.position.y == pytest.approx(0.12)
    hold = _retimed_bounded_output_hold(
        origin_x=1.2, origin_media_pts_ns=1_100_000_000,
        current_media_pts_ns=1_200_000_000,
    )
    hold["frame_id"] = 3
    hold["observed_at_us"] = 1_960_000
    hold = _on_slope(hold)
    held = service.publish(0, [hold], metadata={})
    assert held.observations[0].payload.world is not None
    assert held.observations[0].payload.world.position.y == pytest.approx(0.12)


def test_named_slope_image_motion_keeps_exact_plane_algebra():
    service = _service()
    service.publish(0, [_origin()], metadata={})
    image_motion = _on_slope(_bbox_affine_process_track())
    publication = service.publish(0, [image_motion], metadata={})
    assert publication.observations[0].payload.world is not None
    assert publication.observations[0].payload.world.position.y == pytest.approx(0.12)


def test_named_slope_inferred_process_requires_consistent_xyz_everywhere():
    service = _service()
    origin = _on_slope(_set_image_motion_origin(_track(
        camera_id="kitchen", tracker_id=7, frame_id=1,
        observed_at_us=1_900_000, x=1.0,
    ), trail_segment_id=4))
    service.publish(0, [origin], metadata={})
    track = _on_slope(_helpers._inferred_ground_process_track())
    proof = track["world_prediction_provenance"]
    for vector in (
        track["world_inferred_raw_observation"], track["world_inferred_process_observation"],
        proof["raw_sample"], proof["raw_consensus"], proof["raw_origin"],
        proof["trusted_world_origin"],
    ):
        vector[1] = .1 * vector[0]
    proof["raw_delta"][1] = .1 * proof["raw_delta"][0]
    publication = service.publish(0, [track], metadata={})
    assert publication.observations[0].payload.world is not None
    assert publication.observations[0].payload.world.position.y == pytest.approx(.12)
    # The old producer changed only emitted Y while its trusted origin and
    # raw/process proof remained on the global floor. The gate must still reject it.
    other = _service()
    other.publish(0, [origin], metadata={})
    proof["trusted_world_origin"][1] = 0.
    assert other.publish(0, [track], metadata={}).observations[0].payload.world is None


def test_producer_bounded_proof_enters_canonical_service_on_slope():
    processor = _AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(config={}), tracking_pub=None,
        camera_labels={}, sensor_id_map={},
    )
    service = _service()
    service.publish(0, [_origin()], metadata={})
    cv = _on_slope(_bounded_process_track())
    proof = processor._bounded_process_output_provenance(
        cv, provenance_type="bounded_cv_process",
        provenance_origin="last_metric_process",
        reason="physical_measurement_rejected",
        prior_output=(1.0, 2.0, 1_000_000_000, 1.0, 0),
        metric_output=(1.0, 2.0, 1_000_000_000, 1.0, 0),
        process_observation=(1.2, 0.12, 2.0),
        emitted_point=(1.2, 0.12, 2.0), output_continuous=True,
        emitted_trail_segment_id=0, world_frame_id="backend_world_m",
        world_frame_revision="world-rev", world_transform_sha256="d" * 64,
        support_surface=SLOPE,
    )
    assert proof is not None
    cv["world_prediction_provenance"] = proof
    publication = service.publish(0, [cv], metadata={})
    assert publication.observations[0].payload.world is not None
    assert publication.observations[0].payload.world.position.y == pytest.approx(0.12)


@pytest.mark.parametrize("failure", ["wrong_surface", "off_plane", "unknown_to_named"])
def test_continuation_cannot_switch_or_fake_a_support_plane(failure):
    service = _service()
    origin = _origin()
    if failure == "unknown_to_named":
        origin.pop("world_support_surface")
        origin["world"][1] = 0.0
    service.publish(0, [origin], metadata={})
    cv = _on_slope(_bounded_process_track())
    if failure == "wrong_surface":
        cv["world_support_surface"]["surface_id"] = "another-survey"
    elif failure == "off_plane":
        cv["world_prediction_provenance"]["process_observation"][1] += 0.03
    publication = service.publish(0, [cv], metadata={})
    assert publication.observations[0].payload.world is None


def test_named_metric_point_must_itself_lie_on_declared_plane():
    service = _service()
    origin = _origin()
    origin["world"][1] = 0.0
    publication = service.publish(0, [origin], metadata={})
    assert publication.observations[0].payload.world is None


def test_named_continuation_cannot_leave_observed_extent():
    service = _service()
    surface = SLOPE.model_copy(update={
        "extent_xz": ((0.0, 0.0), (1.1, 0.0), (1.1, 3.0), (0.0, 3.0)),
    })
    origin = _on_slope(_origin(), surface)
    assert service.publish(0, [origin], metadata={}).observations[0].payload.world is not None
    cv = _on_slope(_bounded_process_track(), surface)
    # X=1.2 has a correct Y on the same plane, but is beyond measured support.
    assert service.publish(0, [cv], metadata={}).observations[0].payload.world is None


def test_producer_gain_zero_hold_inside_extent_enters_service_after_rejected_outside_step():
    processor = _AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(config={}), tracking_pub=None,
        camera_labels={}, sensor_id_map={},
    )
    surface = SLOPE.model_copy(update={
        "extent_xz": ((0.0, 0.0), (1.1, 0.0), (1.1, 3.0), (0.0, 3.0)),
    })
    service = _service()
    assert service.publish(0, [_on_slope(_origin(), surface)], metadata={}).observations[0].payload.world is not None
    hold = _on_slope(_retimed_bounded_output_hold(
        origin_x=1.0, origin_media_pts_ns=1_000_000_000,
        current_media_pts_ns=1_100_000_000,
    ), surface)
    # The proposed X=1.4 step is outside measured support and had gain zero.
    # The producer's canonical proof must describe the emitted X=1.0 hold.
    rejected_proposal = (1.4, 0.14, 2.0)
    emitted_hold = (1.0, 0.1, 2.0)
    assert rejected_proposal[0] > max(x for x, _ in surface.extent_xz)
    proof = processor._bounded_process_output_provenance(
        hold, provenance_type="bounded_output_hold",
        provenance_origin="last_published_output",
        reason="physical_output_continuity_exceeded",
        prior_output=(1.0, 2.0, 1_000_000_000, 1.0, 0),
        metric_output=(1.0, 2.0, 1_000_000_000, 1.0, 0),
        process_observation=emitted_hold, emitted_point=emitted_hold,
        output_continuous=False, emitted_trail_segment_id=0,
        world_frame_id="backend_world_m", world_frame_revision="world-rev",
        world_transform_sha256="d" * 64, support_surface=surface,
    )
    assert proof is not None
    assert proof["process_observation"] == pytest.approx(emitted_hold)
    hold["world_prediction_provenance"] = proof
    hold["world_filter_prediction"] = list(emitted_hold)
    hold["world_quality_reason"] = "physical_output_continuity_exceeded"
    publication = service.publish(0, [hold], metadata={})
    assert publication.observations[0].payload.world is not None
    assert publication.observations[0].payload.world.position.y == pytest.approx(.1)


def test_producer_positive_gain_cannot_prove_outside_support_step():
    processor = _AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(config={}), tracking_pub=None,
        camera_labels={}, sensor_id_map={},
    )
    surface = SLOPE.model_copy(update={
        "extent_xz": ((0.0, 0.0), (1.1, 0.0), (1.1, 3.0), (0.0, 3.0)),
    })
    service = _service()
    service.publish(0, [_on_slope(_origin(), surface)], metadata={})
    cv = _on_slope(_bounded_process_track(), surface)
    proof = processor._bounded_process_output_provenance(
        cv, provenance_type="bounded_cv_process",
        provenance_origin="last_metric_process",
        reason="physical_measurement_rejected",
        prior_output=(1.0, 2.0, 1_000_000_000, 1.0, 0),
        metric_output=(1.0, 2.0, 1_000_000_000, 1.0, 0),
        process_observation=(1.2, .12, 2.0), emitted_point=(1.2, .12, 2.0),
        output_continuous=True, emitted_trail_segment_id=0,
        world_frame_id="backend_world_m", world_frame_revision="world-rev",
        world_transform_sha256="d" * 64, support_surface=surface,
    )
    assert proof is not None
    cv["world_prediction_provenance"] = proof
    assert service.publish(0, [cv], metadata={}).observations[0].payload.world is None
