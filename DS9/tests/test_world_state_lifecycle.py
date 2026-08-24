from __future__ import annotations

from types import SimpleNamespace

from noesis.pipelines import hooks


def _processor() -> object:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    processor._world_state_by_track = {}
    processor._world_state_ghost_by_track = {}
    processor._world_state_ghost_ttl_s = 0.75
    return processor


def test_absent_tracker_lifecycle_moves_world_filter_state_to_quarantine() -> None:
    processor = _processor()
    old_family_state = SimpleNamespace(last_good_world=(4.0, 0.0, 8.0))
    current_family_state = SimpleNamespace(last_good_world=(5.0, 0.0, 9.0))
    other_camera_state = SimpleNamespace(last_good_world=(1.0, 0.0, 2.0))
    processor._world_state_by_track = {
        (2, 19): old_family_state,
        (2, 20): current_family_state,
        (1, 19): other_camera_state,
    }

    processor._clear_absent_world_state(2, [20], now_ts=10.0)

    assert processor._world_state_by_track == {
        (2, 20): current_family_state,
        (1, 19): other_camera_state,
    }
    assert processor._world_state_ghost_by_track == {
        (2, 19): (old_family_state, 10.0)
    }


def test_absent_world_state_clear_accepts_tracker_id_strings_and_ignores_bad_values() -> None:
    processor = _processor()
    keep = object()
    drop = object()
    processor._world_state_by_track = {(2, 7): keep, (2, 8): drop}

    processor._clear_absent_world_state(
        2,
        (value for value in ("7", "bad")),
        now_ts=10.0,
    )

    assert processor._world_state_by_track == {(2, 7): keep}
    assert processor._world_state_ghost_by_track == {(2, 8): (drop, 10.0)}


def _ground_state_for_bbox() -> hooks._WorldAnchorState:
    state = hooks._WorldAnchorState()
    state.bbox_center_u = 120.0
    state.bbox_bottom_v = 300.0
    state.bbox_geometry = (20.0, 100.0, 200.0, 200.0)
    state.last_bbox_height_px = 200.0
    state.last_good_world = (1.0, 0.0, 2.0)
    return state


def test_short_absence_restores_only_matching_bbox_geometry() -> None:
    processor = _processor()
    prior = _ground_state_for_bbox()
    processor._world_state_by_track[(2, 7)] = prior
    processor._clear_absent_world_state(2, [], now_ts=10.0)

    restored, was_restored = processor._world_state_for_observation(
        (2, 7),
        now_ts=10.2,
        bbox=[24.0, 102.0, 200.0, 200.0],
    )

    assert was_restored is True
    assert restored is prior
    assert processor._world_state_by_track[(2, 7)] is prior
    assert processor._world_state_ghost_by_track == {}


def test_reused_tracker_id_with_distant_bbox_starts_cold() -> None:
    processor = _processor()
    prior = _ground_state_for_bbox()
    processor._world_state_by_track[(2, 7)] = prior
    processor._clear_absent_world_state(2, [], now_ts=10.0)

    replacement, was_restored = processor._world_state_for_observation(
        (2, 7),
        now_ts=10.2,
        bbox=[700.0, 100.0, 200.0, 200.0],
    )

    assert was_restored is False
    assert replacement is not prior
    assert replacement.last_good_world is None


def test_reused_tracker_id_with_new_lifecycle_generation_starts_cold() -> None:
    processor = _processor()
    prior = _ground_state_for_bbox()
    prior.tracker_lifecycle_generation = 1
    processor._world_state_by_track[(2, 7)] = prior

    replacement, was_restored = processor._world_state_for_observation(
        (2, 7),
        now_ts=10.2,
        bbox=[24.0, 102.0, 200.0, 200.0],
        lifecycle_generation=2,
    )

    assert was_restored is False
    assert replacement is not prior
    assert replacement.last_good_world is None


def test_world_frame_revision_change_resets_filter_before_rejected_hold() -> None:
    state = hooks._WorldAnchorState()
    assert hooks.bind_world_frame(
        state,
        world_frame_id="backend_world_m",
        world_frame_revision="rev-a",
        world_transform_sha256="transform-a",
    ) is False
    state.last_good_world = (4.0, 0.0, 8.0)
    state.last_good_ts = 10.0
    state.world_x = 4.0
    state.world_z = 8.0
    state.filtered_ts = 10.0
    state.vel_world_x = 1.0
    state.vel_world_z = 0.5

    changed = hooks.bind_world_frame(
        state,
        world_frame_id="backend_world_m",
        world_frame_revision="rev-b",
        world_transform_sha256="transform-b",
    )

    assert changed is True
    assert state.world_frame_revision == "rev-b"
    assert state.world_transform_sha256 == "transform-b"
    assert state.last_good_world is None
    assert state.world_x is None
    assert state.world_z is None
    assert state.filtered_ts < 0.0
    assert state.trail_break_required is True
    hooks.mark_world_measurement_unavailable(
        state,
        reason="depth_measurement_not_current",
    )
    assert state.last_good_world is None
    assert state.world_x is None
    assert state.world_z is None
