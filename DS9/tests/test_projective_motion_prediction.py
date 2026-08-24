from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np

from noesis.pipelines import hooks
from noesis.telemetry.person_ground_state import (
    HumanGroundConfig,
    PersonGroundState,
    integrate_projective_ground_observation,
    observe_bbox_stationarity,
    record_accepted_image_geometry,
    transport_accepted_image_foot,
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
    result = processor._project_image_motion_prediction(
        camera_id="cam0",
        state=_state(),
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


def test_stationary_or_pose_anchor_jump_still_fails_physical_projective_gate() -> None:
    state = _rejected_state()
    config = HumanGroundConfig(static_exit_frames=3, max_speed_mps=4.0)
    for frame_id in (1, 2, 3):
        observe_bbox_stationarity(
            state,
            frame_id=frame_id,
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
        bbox=(80.0, 100.0, 40.0, 100.0),
        config=config,
    )
    assert not observe_bbox_stationarity(
        state,
        frame_id=2,
        bbox=(81.0, 100.0, 40.0, 100.0),
        config=config,
    )
    assert observe_bbox_stationarity(
        state,
        frame_id=3,
        bbox=(80.0, 100.0, 40.0, 100.0),
        config=config,
    )
    assert state.bbox_stationary_supported is True
