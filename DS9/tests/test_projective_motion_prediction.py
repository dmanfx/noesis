from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from noesis.pipelines import hooks
from noesis.telemetry.person_ground_state import (
    HumanGroundConfig,
    PersonGroundState,
    integrate_projective_ground_observation,
    observe_bbox_stationarity,
    record_accepted_image_geometry,
    transport_accepted_image_foot,
    transport_accepted_image_foot_from_pose,
)


def _state() -> PersonGroundState:
    state = PersonGroundState(
        last_good_world=(0.0, 0.0, 5.0),
        last_good_ts=10.0,
        motion_mode="walk",
    )
    assert record_accepted_image_geometry(
        state,
        image_foot_uv=(100.0, 200.0),
        bbox=(80.0, 100.0, 40.0, 100.0),
        now_ts=10.0,
        lifecycle_generation=4,
        motion_anchor_uv=(100.0, 150.0),
        motion_basis="pose:torso_motion",
    )
    return state


def test_bbox_affine_transport_follows_motion_without_integrating_prediction() -> None:
    state = _state()
    first = transport_accepted_image_foot(
        state,
        bbox=(90.0, 100.0, 40.0, 100.0),
        now_ts=10.1,
        lifecycle_generation=4,
    )
    second = transport_accepted_image_foot(
        state,
        bbox=(100.0, 100.0, 40.0, 100.0),
        now_ts=10.2,
        lifecycle_generation=4,
    )
    assert first is not None and first[:2] == (110.0, 200.0)
    # The second result is from the original accepted foot (100, 200), not
    # from the first predicted foot (110, 200).
    assert second is not None and second[:2] == (120.0, 200.0)
    assert state.last_accepted_image_foot == (100.0, 200.0)
    assert state.last_accepted_bbox_geometry == (80.0, 100.0, 40.0, 100.0)


def test_pose_transport_follows_current_torso_without_integrating_prediction() -> None:
    state = _state()
    first = transport_accepted_image_foot_from_pose(
        state,
        motion_anchor_uv=(110.0, 150.0),
        motion_basis="pose:torso_motion",
        bbox=(90.0, 100.0, 40.0, 100.0),
        now_ts=10.1,
        lifecycle_generation=4,
    )
    second = transport_accepted_image_foot_from_pose(
        state,
        motion_anchor_uv=(130.0, 150.0),
        motion_basis="pose:torso_motion",
        bbox=(110.0, 100.0, 40.0, 100.0),
        now_ts=12.0,
        lifecycle_generation=4,
    )

    assert first is not None and first[:2] == (110.0, 200.0)
    assert first[5] == "pose_torso_translation"
    assert second is not None and second[:2] == (130.0, 200.0)
    assert state.last_accepted_image_foot == (100.0, 200.0)
    assert state.last_accepted_motion_anchor == (100.0, 150.0)


def test_bbox_only_acceptance_preserves_independent_pose_origin() -> None:
    state = _state()
    pose_origin = state.last_accepted_pose_projective_origin
    metric_before = (
        state.last_good_world,
        state.last_good_ts,
        state.world_x,
        state.world_z,
    )
    assert pose_origin is not None

    # A newer physically accepted bbox floor point refreshes the generic
    # bbox-affine origin. It must not erase or partially overwrite the older
    # complete torso/foot bundle used by pose transport.
    assert record_accepted_image_geometry(
        state,
        image_foot_uv=(210.0, 220.0),
        bbox=(180.0, 100.0, 60.0, 120.0),
        now_ts=11.0,
        lifecycle_generation=4,
        world_point=(1.0, 0.0, 6.0),
    )

    assert state.last_accepted_image_foot == (210.0, 220.0)
    assert state.last_accepted_bbox_geometry == (180.0, 100.0, 60.0, 120.0)
    assert state.last_accepted_image_world == (1.0, 0.0, 6.0)
    assert state.last_accepted_motion_anchor is None
    assert state.last_accepted_pose_projective_origin == pose_origin
    assert (
        state.last_good_world,
        state.last_good_ts,
        state.world_x,
        state.world_z,
    ) == metric_before

    transported = transport_accepted_image_foot_from_pose(
        state,
        motion_anchor_uv=(120.0, 150.0),
        motion_basis="pose:torso_motion",
        bbox=(100.0, 100.0, 40.0, 100.0),
        now_ts=12.0,
        lifecycle_generation=4,
    )
    assert transported is not None
    assert transported[:2] == (120.0, 200.0)
    assert transported[2] == pytest.approx(2.0)


def test_preserved_pose_origin_cannot_cross_lifecycle_generation() -> None:
    state = _state()
    pose_origin = state.last_accepted_pose_projective_origin
    assert pose_origin is not None

    assert record_accepted_image_geometry(
        state,
        image_foot_uv=(110.0, 200.0),
        bbox=(90.0, 100.0, 40.0, 100.0),
        now_ts=10.1,
        lifecycle_generation=5,
        world_point=(0.1, 0.0, 5.0),
    )
    assert state.last_accepted_pose_projective_origin == pose_origin
    assert transport_accepted_image_foot_from_pose(
        state,
        motion_anchor_uv=(110.0, 150.0),
        motion_basis="pose:torso_motion",
        bbox=(90.0, 100.0, 40.0, 100.0),
        now_ts=10.2,
        lifecycle_generation=5,
    ) is None


def test_pose_transport_rejects_wrong_basis_lifecycle_and_expiry() -> None:
    state = _state()
    common = {
        "state": state,
        "motion_anchor_uv": (110.0, 150.0),
        "bbox": (90.0, 100.0, 40.0, 100.0),
    }
    assert transport_accepted_image_foot_from_pose(
        **common,
        motion_basis="pose:left_shoulder",
        now_ts=10.1,
        lifecycle_generation=4,
    ) is None
    assert transport_accepted_image_foot_from_pose(
        **common,
        motion_basis="pose:torso_motion",
        now_ts=10.1,
        lifecycle_generation=5,
    ) is None
    assert transport_accepted_image_foot_from_pose(
        **common,
        motion_basis="pose:torso_motion",
        now_ts=12.6,
        lifecycle_generation=4,
    ) is None


def test_pose_transport_rejects_incoherent_bbox_or_impossible_speed() -> None:
    state = _state()
    assert transport_accepted_image_foot_from_pose(
        state,
        motion_anchor_uv=(120.0, 150.0),
        motion_basis="pose:torso_motion",
        bbox=(60.0, 100.0, 40.0, 100.0),
        now_ts=10.2,
        lifecycle_generation=4,
    ) is None


def test_pose_transport_rejects_large_pose_bbox_magnitude_mismatch() -> None:
    state = _state()

    # Both vectors point right, but a 90 px anatomy jump corroborated by only
    # one pixel of detector motion is a pose swap, not physical translation.
    assert transport_accepted_image_foot_from_pose(
        state,
        motion_anchor_uv=(190.0, 150.0),
        motion_basis="pose:torso_motion",
        bbox=(81.0, 100.0, 40.0, 100.0),
        now_ts=10.2,
        lifecycle_generation=4,
    ) is None


def test_pose_transport_uses_translation_not_bbox_occlusion_scale() -> None:
    state = _state()

    transported = transport_accepted_image_foot_from_pose(
        state,
        motion_anchor_uv=(110.0, 155.0),
        motion_basis="pose:torso_motion",
        bbox=(90.0, 95.0, 40.0, 120.0),
        now_ts=10.2,
        lifecycle_generation=4,
    )

    assert transported is not None
    assert transported[:2] == pytest.approx((110.0, 205.0))
    assert transported[4] == pytest.approx(1.0)


def test_pose_transport_rejects_uncorroborated_bbox_ground_edge_recovery() -> None:
    state = _state()

    transported = transport_accepted_image_foot_from_pose(
        state,
        motion_anchor_uv=(110.0, 155.0),
        motion_basis="pose:torso_motion",
        bbox=(90.0, 20.0, 40.0, 260.0),
        now_ts=10.2,
        lifecycle_generation=4,
    )

    assert transported is None


def test_pose_transport_hard_rejects_sit_to_stand_articulation() -> None:
    state = _state()
    diagnostics: dict[str, object] = {}

    # The bbox centre follows the rising torso exactly, but the bottom edge
    # remains planted. Copying the torso's vertical displacement to the foot
    # would move a stationary ground contact upward by 20 pixels.
    transported = transport_accepted_image_foot_from_pose(
        state,
        motion_anchor_uv=(110.0, 130.0),
        motion_basis="pose:torso_motion",
        bbox=(90.0, 60.0, 40.0, 140.0),
        now_ts=10.2,
        lifecycle_generation=4,
        rejection_diagnostics=diagnostics,
    )

    assert transported is None
    assert diagnostics["hard_rejection"] == (
        "pose_bbox_ground_articulation_conflict"
    )


def test_invalid_new_geometry_preserves_atomic_predictor_bundle() -> None:
    state = _state()
    original = (
        state.last_accepted_image_foot,
        state.last_accepted_image_world,
        state.last_accepted_bbox_geometry,
        state.last_accepted_motion_anchor,
        state.last_accepted_motion_basis,
        state.last_accepted_image_ts,
    )

    assert record_accepted_image_geometry(
        state,
        image_foot_uv=(97.0, 251.3),
        bbox=(100.0, 100.0, 44.0, 100.0),
        now_ts=10.5,
        lifecycle_generation=4,
        motion_anchor_uv=(120.0, 150.0),
        motion_basis="pose:torso_motion",
        world_point=(1.0, 0.0, 6.0),
    ) is False
    assert (
        state.last_accepted_image_foot,
        state.last_accepted_image_world,
        state.last_accepted_bbox_geometry,
        state.last_accepted_motion_anchor,
        state.last_accepted_motion_basis,
        state.last_accepted_image_ts,
    ) == original


def test_pose_transport_rejects_transported_foot_outside_current_silhouette() -> None:
    state = _state()
    assert record_accepted_image_geometry(
        state,
        image_foot_uv=(100.0, 225.0),
        bbox=(80.0, 100.0, 40.0, 100.0),
        now_ts=10.0,
        lifecycle_generation=4,
        motion_anchor_uv=(100.0, 150.0),
        motion_basis="pose:torso_motion",
    )

    assert transport_accepted_image_foot_from_pose(
        state,
        motion_anchor_uv=(100.0, 140.0),
        motion_basis="pose:torso_motion",
        bbox=(80.0, 90.0, 40.0, 80.0),
        now_ts=10.2,
        lifecycle_generation=4,
    ) is None
    assert transport_accepted_image_foot_from_pose(
        state,
        motion_anchor_uv=(180.0, 150.0),
        motion_basis="pose:torso_motion",
        bbox=(160.0, 100.0, 40.0, 100.0),
        now_ts=10.033,
        lifecycle_generation=4,
    ) is None


def test_projective_processor_applies_floor_projection_and_provenance() -> None:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    processor._human_ground_cfg = HumanGroundConfig(
        projective_world_slack_m=0.35,
        max_speed_mps=4.0,
    )
    processor._world_anchor_hold_ttl_s = 0.40
    processor._admit_floor_ray_range = MethodType(
        lambda self, *_args, **_kwargs: True,
        processor,
    )
    processor._project_pixel_to_floor_world = MethodType(
        lambda self, _calib, u, v, **_kwargs: np.asarray(
            [(float(u) - 100.0) * 0.01, 0.0, 5.0], dtype=np.float64
        ),
        processor,
    )
    track: dict[str, object] = {}
    state = _state()
    # A newer metric observation can advance canonical last-good state even
    # when its reprojected foot is not transportable. The predictor must keep
    # using the older atomic image+world bundle, never the new world with the
    # old image geometry.
    state.last_good_world = (9.0, 0.0, 9.0)
    result = processor._project_image_motion_prediction(
        camera_id="cam0",
        state=state,
        bbox_project=(90.0, 100.0, 40.0, 100.0),
        calib=SimpleNamespace(image_size=(320, 240)),
        now_ts=10.1,
        lifecycle_generation=4,
        flip_u=False,
        flip_v=False,
        track=track,
    )
    assert result is not None
    point, provenance = result
    assert point.tolist() == [0.1, 0.0, 5.0]
    assert provenance["non_authoritative"] is True
    assert provenance["origin"] == "last_accepted_image_foot"
    assert provenance["transport"] == "bbox_affine"


def test_projective_processor_prefers_exact_pose_transport() -> None:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    processor._human_ground_cfg = HumanGroundConfig(
        projective_world_slack_m=0.35,
        max_speed_mps=4.0,
    )
    processor._world_anchor_hold_ttl_s = 0.40
    processor._admit_floor_ray_range = MethodType(
        lambda self, *_args, **_kwargs: True,
        processor,
    )
    processor._project_pixel_to_floor_world = MethodType(
        lambda self, _calib, u, v, **_kwargs: np.asarray(
            [(float(u) - 100.0) * 0.01, 0.0, 5.0], dtype=np.float64
        ),
        processor,
    )

    result = processor._project_image_motion_prediction(
        camera_id="cam0",
        state=_state(),
        bbox_project=(110.0, 100.0, 40.0, 100.0),
        calib=SimpleNamespace(image_size=(320, 240)),
        now_ts=12.0,
        lifecycle_generation=4,
        flip_u=False,
        flip_v=False,
        track={},
        motion_anchor_uv=(130.0, 150.0),
        motion_basis="pose:torso_motion",
    )

    assert result is not None
    point, provenance = result
    assert point.tolist() == [0.3, 0.0, 5.0]
    assert provenance["transport"] == "pose_torso_translation"
    assert provenance["origin"] == "last_accepted_pose_projective_origin"


def test_pose_projective_world_gate_uses_its_atomic_pose_origin() -> None:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    processor._human_ground_cfg = HumanGroundConfig(
        projective_world_slack_m=0.35,
        max_speed_mps=0.1,
    )
    processor._world_anchor_hold_ttl_s = 0.40
    processor._admit_floor_ray_range = MethodType(
        lambda self, *_args, **_kwargs: True,
        processor,
    )
    processor._project_pixel_to_floor_world = MethodType(
        lambda self, _calib, u, v, **_kwargs: np.asarray(
            [(float(u) - 100.0) * 0.01, 0.0, 5.0], dtype=np.float64
        ),
        processor,
    )
    state = _state()
    pose_origin = state.last_accepted_pose_projective_origin
    assert pose_origin is not None
    assert record_accepted_image_geometry(
        state,
        image_foot_uv=(210.0, 220.0),
        bbox=(180.0, 100.0, 60.0, 120.0),
        now_ts=11.0,
        lifecycle_generation=4,
        world_point=(9.0, 0.0, 9.0),
    )

    result = processor._project_image_motion_prediction(
        camera_id="cam0",
        state=state,
        bbox_project=(110.0, 100.0, 40.0, 100.0),
        calib=SimpleNamespace(image_size=(320, 240)),
        now_ts=12.0,
        lifecycle_generation=4,
        flip_u=False,
        flip_v=False,
        track={},
        motion_anchor_uv=(130.0, 150.0),
        motion_basis="pose:torso_motion",
    )

    assert result is not None
    point, provenance = result
    assert point.tolist() == [0.3, 0.0, 5.0]
    assert provenance["origin"] == "last_accepted_pose_projective_origin"
    assert provenance["world_delta_m"] == pytest.approx(0.3)


def test_projective_processor_does_not_bypass_articulation_rejection() -> None:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    processor._human_ground_cfg = HumanGroundConfig(
        projective_world_slack_m=0.35,
        max_speed_mps=4.0,
    )
    processor._world_anchor_hold_ttl_s = 0.40
    processor._admit_floor_ray_range = MethodType(
        lambda self, *_args, **_kwargs: True,
        processor,
    )
    processor._project_pixel_to_floor_world = MethodType(
        lambda self, _calib, u, v, **_kwargs: np.asarray(
            [(float(u) - 100.0) * 0.01, 0.0, 5.0], dtype=np.float64
        ),
        processor,
    )
    state = _state()
    before = (
        state.world_x,
        state.world_z,
        state.last_accepted_image_foot,
        state.last_accepted_image_world,
        state.last_accepted_bbox_geometry,
        state.last_accepted_motion_anchor,
    )
    track: dict[str, object] = {}

    result = processor._project_image_motion_prediction(
        camera_id="cam0",
        state=state,
        bbox_project=(90.0, 60.0, 40.0, 140.0),
        calib=SimpleNamespace(image_size=(320, 240)),
        now_ts=10.2,
        lifecycle_generation=4,
        flip_u=False,
        flip_v=False,
        track=track,
        motion_anchor_uv=(110.0, 130.0),
        motion_basis="pose:torso_motion",
    )

    assert result is None
    assert track["world_projective_rejection_reason"] == (
        "pose_bbox_ground_articulation_conflict"
    )
    assert (
        state.world_x,
        state.world_z,
        state.last_accepted_image_foot,
        state.last_accepted_image_world,
        state.last_accepted_bbox_geometry,
        state.last_accepted_motion_anchor,
    ) == before


def test_recent_projective_process_bridge_chains_within_immutable_root() -> None:
    state = PersonGroundState(
        last_good_world=(0.0, 0.0, 5.0),
        last_good_ts=10.0,
        world_x=0.5,
        world_z=5.0,
        filtered_ts=12.0,
        rejection_anchor_x=0.5,
        rejection_anchor_z=5.0,
        rejection_anchor_ts=12.0,
        projective_bridge_origin_ts=12.0,
        projective_bridge_process_ts=12.0,
        projective_bridge_rows_remaining=1,
    )

    assert hooks._AnalyticsTelemetryProcessor._recent_process_bridge_is_current(
        state,
        now_ts=12.1,
        ttl_s=0.4,
    ) is True

    # A bounded CV callback advances only the dedicated process timestamp.
    # Neither publication nor the legacy row counter may renew or terminate
    # the immutable root episode.
    state.filtered_ts = 12.1
    state.projective_bridge_process_ts = 12.1
    state.projective_bridge_rows_remaining = 0
    assert hooks._AnalyticsTelemetryProcessor._recent_process_bridge_is_current(
        state,
        now_ts=12.2,
        ttl_s=0.4,
    ) is True

    # A second descendant remains eligible without moving the original image
    # root. This is the producer half of the service-owned root lineage.
    state.filtered_ts = 12.2
    state.projective_bridge_process_ts = 12.2
    assert hooks._AnalyticsTelemetryProcessor._recent_process_bridge_is_current(
        state,
        now_ts=12.3,
        ttl_s=0.4,
    ) is True
    assert state.projective_bridge_origin_ts == pytest.approx(12.0)

    # The fixed 5ms cadence tolerance includes a nominal 400ms row despite
    # epoch-float subtraction, but no descendant can extend the root beyond
    # 405ms.
    state.filtered_ts = 12.3
    state.projective_bridge_process_ts = 12.3
    assert hooks._AnalyticsTelemetryProcessor._recent_process_bridge_is_current(
        state,
        now_ts=12.4,
        ttl_s=0.4,
    ) is True
    assert hooks._AnalyticsTelemetryProcessor._recent_process_bridge_is_current(
        state,
        now_ts=12.405_001,
        ttl_s=0.4,
    ) is False

    # A metric mutation cannot impersonate a process descendant even while
    # the immutable root is still within its wall-clock budget.
    state.filtered_ts = 12.35
    assert hooks._AnalyticsTelemetryProcessor._recent_process_bridge_is_current(
        state,
        now_ts=12.39,
        ttl_s=0.4,
    ) is False


def test_hold_ttl_allows_only_bounded_media_clock_epsilon() -> None:
    current = hooks._AnalyticsTelemetryProcessor._hold_age_is_current

    assert current(0.4000, ttl_s=0.4) is True
    assert current(0.4018, ttl_s=0.4) is True
    assert current(0.4050, ttl_s=0.4) is True
    assert current(0.4051, ttl_s=0.4) is False
    assert current(0.5000, ttl_s=0.4) is False


@pytest.mark.parametrize("image_size", ((640, 360), (1280, 720), (1920, 1080)))
def test_bbox_floor_candidate_size_gate_is_resolution_normalized(
    image_size: tuple[int, int],
) -> None:
    gate = (
        hooks._AnalyticsTelemetryProcessor
        ._bbox_floor_candidate_has_resolved_silhouette
    )
    image_height = float(image_size[1])
    threshold_height = image_height * 48.0 / 1080.0

    assert gate(
        bbox_width_px=threshold_height * 0.45,
        bbox_height_px=threshold_height,
        image_size=image_size,
    ) is True
    assert gate(
        bbox_width_px=threshold_height * 0.45,
        bbox_height_px=threshold_height - 0.01,
        image_size=image_size,
    ) is False
    assert gate(
        bbox_width_px=threshold_height * 0.90,
        bbox_height_px=threshold_height,
        image_size=image_size,
    ) is False


def test_one_frame_impossible_bbox_jump_fails_closed() -> None:
    state = _state()
    assert transport_accepted_image_foot(
        state,
        bbox=(180.0, 100.0, 40.0, 100.0),
        now_ts=10.033,
        lifecycle_generation=4,
    ) is None


def _rejected_state() -> PersonGroundState:
    state = _state()
    state.world_x = 0.0
    state.world_z = 5.0
    state.filtered_ts = 10.1
    state.measurement_accepted = False
    state.measurement_rejection_reason = "physical_output_speed_exceeded"
    state.rejection_anchor_x = 0.0
    state.rejection_anchor_z = 5.0
    state.rejection_anchor_ts = 10.0
    return state


def test_rejected_metric_frame_can_integrate_projective_continuation() -> None:
    state = _rejected_state()
    result = integrate_projective_ground_observation(
        state,
        measurement=np.asarray([0.30, 0.0, 5.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=10.1,
        config=HumanGroundConfig(max_speed_mps=4.0),
    )

    assert result is not None
    assert float(result[0]) > 0.0
    assert state.measurement_accepted is False
    assert state.measurement_rejection_reason == "projective_weak_observation"
    assert state.last_good_world == (0.0, 0.0, 5.0)
    assert state.last_good_ts == 10.0
    assert state.projective_bridge_origin_ts == pytest.approx(state.filtered_ts)
    assert state.projective_bridge_process_ts == pytest.approx(state.filtered_ts)
    assert state.projective_bridge_rows_remaining == 1


def test_projective_continuation_preserves_metric_reacquire_consensus() -> None:
    state = _rejected_state()
    state.reacquire_candidate_x = 2.0
    state.reacquire_candidate_z = 5.0
    state.reacquire_candidate_ts = 10.1
    state.reacquire_candidate_basis = "pose_floor"
    state.reacquire_count = 1

    result = integrate_projective_ground_observation(
        state,
        measurement=np.asarray([0.30, 0.0, 5.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=10.1,
        config=HumanGroundConfig(max_speed_mps=4.0),
    )

    assert result is not None
    assert state.reacquire_candidate_x == pytest.approx(2.0)
    assert state.reacquire_candidate_z == pytest.approx(5.0)
    assert state.reacquire_candidate_ts == pytest.approx(10.1)
    assert state.reacquire_candidate_basis == "pose_floor"
    assert state.reacquire_count == 1


def test_stationary_or_pose_anchor_jump_still_fails_physical_projective_gate() -> None:
    state = _rejected_state()
    config = HumanGroundConfig(static_exit_frames=3, max_speed_mps=4.0)
    for frame_id in (1, 2, 3):
        observe_bbox_stationarity(
            state,
            frame_id=frame_id,
            observation_ts=float(frame_id) / 10.0,
            bbox=(80.0, 100.0, 40.0, 100.0),
            config=config,
        )
    assert state.bbox_stationary_supported is True

    # A pose/depth anchor jump is not allowed to become a clipped projective
    # relocation, even when detector-box stationarity suggests a hold.
    result = integrate_projective_ground_observation(
        state,
        measurement=np.asarray([3.0, 0.0, 5.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=10.1,
        config=config,
    )

    assert result is None
    assert state.world_x == 0.0
    assert state.world_z == 5.0
    assert state.measurement_accepted is False
    assert state.measurement_rejection_reason == "physical_output_speed_exceeded"


def test_stationary_seated_hold_evidence_remains_bounded() -> None:
    state = PersonGroundState()
    config = HumanGroundConfig(static_exit_frames=3)
    assert not observe_bbox_stationarity(
        state,
        frame_id=1,
        observation_ts=0.1,
        bbox=(80.0, 100.0, 40.0, 100.0),
        config=config,
    )
    assert not observe_bbox_stationarity(
        state,
        frame_id=2,
        observation_ts=0.2,
        bbox=(81.0, 100.0, 40.0, 100.0),
        config=config,
    )
    assert observe_bbox_stationarity(
        state,
        frame_id=3,
        observation_ts=0.3,
        bbox=(80.0, 100.0, 40.0, 100.0),
        config=config,
    )
    assert state.bbox_stationary_supported is True
