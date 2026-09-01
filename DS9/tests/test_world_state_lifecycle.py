from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from noesis.pipelines import hooks


def _processor() -> object:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    processor._world_state_by_track = {}
    processor._world_state_ghost_by_track = {}
    processor._world_state_ghost_ttl_s = 0.75
    processor._world_output_watermarks = hooks.OrderedDict()
    processor._world_output_watermark_capacity = 8
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


def test_same_lifecycle_generation_restores_ground_state_without_second_bbox_authority() -> None:
    processor = _processor()
    prior = _ground_state_for_bbox()
    prior.tracker_lifecycle_generation = 7
    prior.last_output_world_x = 1.0
    prior.last_output_world_z = 2.0
    processor._world_state_by_track[(2, 7)] = prior
    processor._clear_absent_world_state(2, [], now_ts=10.0)

    restored, was_restored = processor._world_state_for_observation(
        (2, 7),
        now_ts=10.2,
        bbox=[100.0, 100.0, 200.0, 200.0],
        lifecycle_generation=7,
    )

    assert was_restored is True
    assert restored is prior
    assert restored.last_output_world_x == 1.0
    assert restored.last_output_world_z == 2.0


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


def test_same_public_lifecycle_retains_output_gate_when_measurement_state_is_recreated() -> None:
    processor = _processor()
    track = {
        "tracker_id": 7,
        "tracker_lifecycle_generation": 3,
        "world_calibration_sha256": "a" * 64,
    }
    key = processor._world_output_watermark_key(
        2,
        track,
        world_frame_id="backend_world_m",
        world_frame_revision="rev-a",
        world_transform_sha256="transform-a",
    )
    assert key is not None

    prior = hooks._WorldAnchorState()
    prior.last_output_world_x = 1.0
    prior.last_output_world_z = 2.0
    prior.last_output_media_pts_ns = 1_000_000_000
    prior.last_output_filter_ts = 10.0
    prior.last_output_trail_segment_id = 0
    processor._save_world_output_watermark(prior, key)

    recreated = hooks._WorldAnchorState()
    assert processor._restore_world_output_watermark(recreated, key) is True
    emitted, accepted = hooks.admit_human_ground_output(
        recreated,
        candidate=np.array([2.0, 0.0, 2.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=10.033,
        media_pts_ns=1_033_000_000,
        config=hooks.HumanGroundConfig(max_speed_mps=4.0),
    )

    assert accepted is False
    assert emitted.tolist() == [1.0, 0.0, 2.0]
    assert recreated.measurement_rejection_reason == "physical_output_continuity_exceeded"


def test_rate_suppressed_internal_output_cannot_replace_queue_reference() -> None:
    processor = _processor()
    track = {
        "tracker_id": 7,
        "tracker_lifecycle_generation": 3,
        "world_calibration_sha256": "a" * 64,
    }
    key = processor._world_output_watermark_key(
        2,
        track,
        world_frame_id="backend_world_m",
        world_frame_revision="rev-a",
        world_transform_sha256="transform-a",
    )
    published = hooks._WorldAnchorState(
        last_output_world_x=1.0,
        last_output_world_z=2.0,
        last_output_media_pts_ns=1_000_000_000,
        last_output_filter_ts=10.0,
        last_output_trail_segment_id=0,
    )
    processor._save_world_output_watermark(published, key)

    # A later raw callback reanchored internally but was rate-suppressed and
    # never entered the ordered tracking/BEV queue.
    internal = hooks._WorldAnchorState(
        last_output_world_x=3.0,
        last_output_world_z=4.0,
        last_output_media_pts_ns=1_020_000_000,
        last_output_filter_ts=10.02,
        last_output_trail_segment_id=1,
        trail_segment_id=1,
        trail_break_required=True,
    )
    reference = processor._world_output_reference(key)
    assert reference == (1.0, 2.0, 1_000_000_000, 10.0, 0)

    emitted, accepted = hooks.admit_human_ground_output(
        internal,
        candidate=np.array([3.1, 0.0, 4.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=10.1,
        media_pts_ns=1_100_000_000,
        config=hooks.HumanGroundConfig(max_speed_mps=4.0),
        allow_segment_break=False,
        prior_output=reference,
    )

    assert accepted is False
    assert emitted.tolist() == [1.0, 0.0, 2.0]
    assert processor._world_output_reference(key) == reference

    internal.tracker_lifecycle_generation = 3
    processor._world_state_by_track[(2, 7)] = internal
    processor._commit_enqueued_world_output_watermarks(
        2,
        [
            {
                **track,
                "world_valid": True,
                "world": emitted.tolist(),
                "world_frame": "backend_world_m",
                "world_frame_revision": "rev-a",
                "world_transform_sha256": "transform-a",
                "media_pts_ns": 1_100_000_000,
                # Internal metric evaluation exposed segment 1, but the held
                # coordinate still belongs to visible segment 0.
                "trail_segment_id": 1,
            }
        ],
        filter_ts=10.1,
    )
    assert processor._world_output_reference(key) == (
        1.0,
        2.0,
        1_100_000_000,
        10.1,
        0,
    )


def test_service_admitted_projective_descendants_preserve_root_episode() -> None:
    processor = _processor()
    state = hooks._WorldAnchorState(
        last_good_world=(1.0, 0.0, 2.0),
        last_good_ts=9.9,
        world_x=1.2,
        world_z=2.2,
        filtered_ts=10.1,
        last_output_world_x=1.2,
        last_output_world_z=2.2,
        last_output_media_pts_ns=1_100_000_000,
        last_output_filter_ts=10.1,
        last_output_trail_segment_id=1,
        projective_bridge_origin_ts=10.0,
        projective_bridge_process_ts=10.1,
        projective_bridge_rows_remaining=1,
    )
    state.tracker_lifecycle_generation = 3
    processor._world_state_by_track[(2, 7)] = state
    common = {
        "tracker_id": 7,
        "tracker_lifecycle_generation": 3,
        "world_calibration_sha256": "a" * 64,
        "world_valid": True,
        "world_frame": "backend_world_m",
        "world_frame_revision": "rev-a",
        "world_transform_sha256": "transform-a",
        "trail_segment_id": 1,
        "world_quality": "held",
        "world_measurement_accepted": False,
    }
    image_track = {
        **common,
        "frame_id": 10,
        "world": [1.1, 0.0, 2.1],
        "media_pts_ns": 1_000_000_000,
        "world_source": "image_motion_prediction",
        "world_prediction_provenance": {
            "type": "bbox_affine_floor_projection",
            "origin": "last_accepted_image_foot",
            "state_integrated": True,
        },
    }
    processor._commit_enqueued_world_output_watermarks(
        2,
        [image_track],
        filter_ts=10.0,
        admitted_world_track_keys={(7, 3, 10)},
    )
    cv_track = {
        **common,
        "frame_id": 11,
        "world": [1.2, 0.0, 2.2],
        "media_pts_ns": 1_100_000_000,
        "world_source": "cv_prediction",
        "world_prediction_provenance": {
            "type": "bounded_cv_process",
            "origin": "recent_projective_process",
            "state_integrated": True,
        },
    }

    processor._commit_enqueued_world_output_watermarks(
        2,
        [cv_track],
        filter_ts=10.1,
        admitted_world_track_keys={(7, 3, 11)},
    )

    key = processor._world_output_watermark_key(
        2,
        cv_track,
        world_frame_id="backend_world_m",
        world_frame_revision="rev-a",
        world_transform_sha256="transform-a",
    )
    assert processor._world_output_reference(key) == (
        1.2,
        2.2,
        1_100_000_000,
        10.1,
        1,
    )
    assert processor._world_output_projective_root(
        key,
        current_media_pts_ns=1_200_000_000,
    ) == 1_000_000_000
    assert state.last_output_world_x == 1.2
    assert state.last_output_world_z == 2.2
    assert state.projective_bridge_origin_ts == 10.0
    assert state.projective_bridge_rows_remaining == 0
    assert processor._recent_process_bridge_is_current(
        state,
        now_ts=10.2,
        ttl_s=0.4,
    ) is True

    # The next bounded process callback advances only the process timestamp;
    # the image-root timestamp remains immutable. A service-admitted exact
    # hold advances the visible cohort but must not replace the last
    # motion-bearing reference used by a later recovery.
    state.filtered_ts = 10.2
    state.projective_bridge_process_ts = 10.2
    held_track = {
        **common,
        "frame_id": 12,
        "world": [1.2, 0.0, 2.2],
        "media_pts_ns": 1_200_000_000,
        "world_source": "anchor_hold",
        "world_prediction_provenance": {
            "type": "bounded_output_hold",
            "origin": "last_published_output",
            "state_integrated": True,
        },
    }
    processor._commit_enqueued_world_output_watermarks(
        2,
        [held_track],
        filter_ts=10.2,
        admitted_world_track_keys={(7, 3, 12)},
    )

    assert processor._world_output_reference(key) == (
        1.2,
        2.2,
        1_200_000_000,
        10.2,
        1,
    )
    assert processor._world_output_reference(key, kinematic_only=True) == (
        1.2,
        2.2,
        1_100_000_000,
        10.1,
        1,
    )
    assert processor._world_output_commit_provenance(key) == (
        "anchor_hold",
        "bounded_output_hold",
    )
    assert processor._world_output_projective_root(
        key,
        current_media_pts_ns=2_655_000_000,
    ) == 1_000_000_000
    assert processor._world_output_projective_root(
        key,
        current_media_pts_ns=2_655_001_000,
    ) is None
    assert state.projective_bridge_origin_ts == 10.0
    assert processor._recent_process_bridge_is_current(
        state,
        now_ts=10.4,
        ttl_s=0.4,
    ) is True
    assert processor._recent_process_bridge_is_current(
        state,
        now_ts=10.405_001,
        ttl_s=0.4,
    ) is False


def test_canonical_rejection_cannot_advance_producer_output_watermark() -> None:
    processor = _processor()
    track = {
        "tracker_id": 7,
        "tracker_lifecycle_generation": 3,
        "world_calibration_sha256": "a" * 64,
        "frame_id": 10,
        "world_valid": True,
        "world": [1.2, 0.0, 2.2],
        "world_frame": "backend_world_m",
        "world_frame_revision": "rev-a",
        "world_transform_sha256": "transform-a",
        "media_pts_ns": 1_100_000_000,
        "trail_segment_id": 1,
        "world_source": "pose_floor_only",
        "world_measurement_accepted": True,
    }

    processor._commit_enqueued_world_output_watermarks(
        2,
        [track],
        filter_ts=10.1,
        admitted_world_track_keys=set(),
    )
    key = processor._world_output_watermark_key(
        2,
        track,
        world_frame_id="backend_world_m",
        world_frame_revision="rev-a",
        world_transform_sha256="transform-a",
    )
    assert processor._world_output_reference(key) is None

    processor._commit_enqueued_world_output_watermarks(
        2,
        [track],
        filter_ts=10.1,
        admitted_world_track_keys={(7, 3, 10)},
    )
    assert processor._world_output_reference(key) == (
        1.2,
        2.2,
        1_100_000_000,
        10.1,
        1,
    )


def test_service_admitted_history_origin_advances_producer_watermark() -> None:
    processor = _processor()
    common = {
        "tracker_id": 7,
        "tracker_lifecycle_generation": 3,
        "world_calibration_sha256": "a" * 64,
        "world_valid": True,
        "world_frame": "backend_world_m",
        "world_frame_revision": "rev-a",
        "world_transform_sha256": "transform-a",
        "trail_segment_id": 0,
    }
    metric = {
        **common,
        "frame_id": 1,
        "world": [1.0, 0.0, 2.0],
        "media_pts_ns": 1_000_000_000,
        "world_source": "pose_floor_only",
        "world_quality": "good",
        "world_measurement_accepted": True,
    }
    processor._commit_enqueued_world_output_watermarks(
        2,
        [metric],
        filter_ts=10.0,
    )
    candidate = {
        **common,
        "frame_id": 3,
        "world": [1.25, 0.0, 2.0],
        "media_pts_ns": 1_200_000_000,
        "world_source": "cv_prediction",
        "world_quality": "held",
        "world_measurement_accepted": False,
        # The canonical service has already checked the full proof and names
        # this exact row in its receipt. This compact fixture intentionally
        # omits the proof so a latest-only producer recheck would reject it.
        "world_prediction_provenance": {
            "type": "bounded_cv_process",
            "origin": "recent_projective_process",
        },
    }
    processor._commit_enqueued_world_output_watermarks(
        2,
        [candidate],
        filter_ts=10.2,
        admitted_world_track_keys={(7, 3, 3)},
    )

    key = processor._world_output_watermark_key(
        2,
        candidate,
        world_frame_id="backend_world_m",
        world_frame_revision="rev-a",
        world_transform_sha256="transform-a",
    )
    assert processor._world_output_reference(key) == (
        1.25,
        2.0,
        1_200_000_000,
        10.2,
        0,
    )
    assert processor._world_output_commit_provenance(key) == (
        "cv_prediction",
        "bounded_cv_process",
    )
    assert processor._world_output_reference(key, metric_only=True) == (
        1.0,
        2.0,
        1_000_000_000,
        10.0,
        0,
    )
    # A CV row descended directly from metric process state cannot create a
    # projective episode merely by claiming a recent-projective label.
    assert processor._world_output_projective_root(
        key,
        current_media_pts_ns=1_300_000_000,
    ) is None


def test_output_watermark_retains_last_metric_origin_across_held_rows() -> None:
    processor = _processor()
    common = {
        "tracker_id": 7,
        "tracker_lifecycle_generation": 3,
        "world_calibration_sha256": "a" * 64,
        "world_valid": True,
        "world_frame": "backend_world_m",
        "world_frame_revision": "rev-a",
        "world_transform_sha256": "transform-a",
        "trail_segment_id": 0,
    }
    processor._commit_enqueued_world_output_watermarks(
        2,
        [
            {
                **common,
                "world": [1.0, 0.0, 2.0],
                "media_pts_ns": 1_000_000_000,
                "world_source": "pose_floor_only",
                "world_quality": "good",
                "world_measurement_accepted": True,
            }
        ],
        filter_ts=10.0,
    )
    held_world = [1.2, 0.0, 2.1]
    world_delta_m = float(np.hypot(0.2, 0.1))
    processor._commit_enqueued_world_output_watermarks(
        2,
        [
            {
                **common,
                "world": held_world,
                "world_filter_prediction": held_world,
                "media_pts_ns": 1_100_000_000,
                "world_source": "image_motion_prediction",
                "world_quality": "held",
                "world_measurement_accepted": False,
                "world_prediction_provenance": {
                    "type": "bbox_affine_floor_projection",
                    "non_authoritative": True,
                    "state_integrated": True,
                    "origin": "last_accepted_image_foot",
                    "transport": "bbox_affine",
                    "image_foot": [100.0, 200.0],
                    "projective_origin_world": [1.0, 0.0, 2.0],
                    "process_observation": held_world,
                    "age_s": 0.1,
                    "world_delta_m": world_delta_m,
                    "world_delta_limit_m": 0.75,
                    "filter_transition": {
                        "version": 1,
                        "kind": "innovation_update",
                        "origin_kind": "queue_admitted_world_output",
                        "origin_world": [1.0, 0.0, 2.0],
                        "origin_media_pts_ns": 1_000_000_000,
                        "current_media_pts_ns": 1_100_000_000,
                        "origin_trail_segment_id": 0,
                        "gate_dt_s": 0.1,
                        "position_base": [1.0, 0.0, 2.0],
                        "position_gain": 1.0,
                        "max_speed_mps": 4.0,
                        "max_jump_m": 0.75,
                        "reset_after_s": 1.25,
                    },
                },
            }
        ],
        filter_ts=10.1,
    )
    key = processor._world_output_watermark_key(
        2,
        common,
        world_frame_id="backend_world_m",
        world_frame_revision="rev-a",
        world_transform_sha256="transform-a",
    )

    assert processor._world_output_reference(key) == (
        1.2,
        2.1,
        1_100_000_000,
        10.1,
        0,
    )
    assert processor._world_output_reference(key, metric_only=True) == (
        1.0,
        2.0,
        1_000_000_000,
        10.0,
        0,
    )


def test_repeated_gain_zero_holds_preserve_kinematic_reference() -> None:
    processor = _processor()
    common = {
        "tracker_id": 7,
        "tracker_lifecycle_generation": 3,
        "world_calibration_sha256": "a" * 64,
        "world_valid": True,
        "world_frame": "backend_world_m",
        "world_frame_revision": "rev-a",
        "world_transform_sha256": "transform-a",
        "trail_segment_id": 0,
    }

    def commit(track: dict[str, object], *, filter_ts: float) -> None:
        processor._commit_enqueued_world_output_watermarks(
            2,
            [track],
            filter_ts=filter_ts,
            admitted_world_track_keys={
                (
                    int(track["tracker_id"]),
                    int(track["tracker_lifecycle_generation"]),
                    int(track["frame_id"]),
                )
            },
        )

    metric = {
        **common,
        "frame_id": 1,
        "world": [1.0, 0.0, 2.0],
        "media_pts_ns": 1_000_000_000,
        "world_source": "pose_floor_only",
        "world_quality": "good",
        "world_measurement_accepted": True,
    }
    commit(metric, filter_ts=10.0)
    moving = {
        **common,
        "frame_id": 2,
        "world": [1.2, 0.0, 2.1],
        "media_pts_ns": 1_100_000_000,
        "world_source": "cv_prediction",
        "world_quality": "held",
        "world_measurement_accepted": False,
        "world_prediction_provenance": {
            "type": "bounded_cv_process",
            "origin": "recent_projective_process",
        },
    }
    commit(moving, filter_ts=10.1)

    for frame_id, media_pts_ns, filter_ts in (
        (3, 1_200_000_000, 10.2),
        (4, 1_300_000_000, 10.3),
    ):
        commit(
            {
                **common,
                "frame_id": frame_id,
                "world": [1.2, 0.0, 2.1],
                "media_pts_ns": media_pts_ns,
                "world_source": "anchor_hold",
                "world_quality": "held",
                "world_measurement_accepted": False,
                "world_prediction_provenance": {
                    "type": "bounded_output_hold",
                    "origin": "last_published_output",
                },
            },
            filter_ts=filter_ts,
        )

    key = processor._world_output_watermark_key(
        2,
        common,
        world_frame_id="backend_world_m",
        world_frame_revision="rev-a",
        world_transform_sha256="transform-a",
    )
    assert processor._world_output_reference(key) == (
        1.2,
        2.1,
        1_300_000_000,
        10.3,
        0,
    )
    assert processor._world_output_reference(key, kinematic_only=True) == (
        1.2,
        2.1,
        1_100_000_000,
        10.1,
        0,
    )
    assert processor._world_output_reference(key, metric_only=True) == (
        1.0,
        2.0,
        1_000_000_000,
        10.0,
        0,
    )

    recovered = {
        **common,
        "frame_id": 5,
        "world": [1.4, 0.0, 2.2],
        "media_pts_ns": 1_400_000_000,
        "world_source": "cv_prediction",
        "world_quality": "held",
        "world_measurement_accepted": False,
        "world_prediction_provenance": {
            "type": "bounded_cv_process",
            "origin": "recent_projective_process",
        },
    }
    commit(recovered, filter_ts=10.4)
    assert processor._world_output_reference(key, kinematic_only=True) == (
        1.4,
        2.2,
        1_400_000_000,
        10.4,
        0,
    )


def test_output_watermark_does_not_cross_generation_or_world_revision() -> None:
    processor = _processor()
    prior_key = processor._world_output_watermark_key(
        2,
        {
            "tracker_id": 7,
            "tracker_lifecycle_generation": 3,
            "world_calibration_sha256": "a" * 64,
        },
        world_frame_id="backend_world_m",
        world_frame_revision="rev-a",
        world_transform_sha256="transform-a",
    )
    assert prior_key is not None
    prior = hooks._WorldAnchorState()
    prior.last_output_world_x = 1.0
    prior.last_output_world_z = 2.0
    prior.last_output_filter_ts = 10.0
    processor._save_world_output_watermark(prior, prior_key)

    next_generation = processor._world_output_watermark_key(
        2,
        {
            "tracker_id": 7,
            "tracker_lifecycle_generation": 4,
            "world_calibration_sha256": "a" * 64,
        },
        world_frame_id="backend_world_m",
        world_frame_revision="rev-a",
        world_transform_sha256="transform-a",
    )
    next_revision = processor._world_output_watermark_key(
        2,
        {
            "tracker_id": 7,
            "tracker_lifecycle_generation": 3,
            "world_calibration_sha256": "a" * 64,
        },
        world_frame_id="backend_world_m",
        world_frame_revision="rev-b",
        world_transform_sha256="transform-b",
    )
    next_calibration = processor._world_output_watermark_key(
        2,
        {
            "tracker_id": 7,
            "tracker_lifecycle_generation": 3,
            "world_calibration_sha256": "b" * 64,
        },
        world_frame_id="backend_world_m",
        world_frame_revision="rev-a",
        world_transform_sha256="transform-a",
    )
    assert processor._restore_world_output_watermark(
        hooks._WorldAnchorState(),
        next_generation,
    ) is False
    assert processor._restore_world_output_watermark(
        hooks._WorldAnchorState(),
        next_revision,
    ) is False
    assert processor._restore_world_output_watermark(
        hooks._WorldAnchorState(),
        next_calibration,
    ) is False


def test_late_output_watermark_cannot_mutate_new_authority_state() -> None:
    current_authority = {
        "world_frame": "backend_world_m",
        "world_frame_revision": "rev-b",
        "world_transform_sha256": "transform-b",
        "world_calibration_sha256": "b" * 64,
        "source_epoch": 7,
        "tracker_lifecycle_generation": 3,
    }
    stale_variants = {
        "world_frame": "calibration_world_m",
        "world_frame_revision": "rev-a",
        "world_transform_sha256": "transform-a",
        "world_calibration_sha256": "a" * 64,
        "source_epoch": 6,
        "tracker_lifecycle_generation": 2,
    }

    for changed_field, stale_value in stale_variants.items():
        processor = _processor()
        processor._world_frame = "backend_world_m"
        processor._source_epoch_by_sensor = {2: 7}
        processor.bev_calibration = SimpleNamespace(
            world_snapshot=lambda _source_id, _camera_id: SimpleNamespace(
                world_frame_id="backend_world_m",
                world_frame_revision="rev-b",
                frame_transform_sha256="transform-b",
                camera_calibration_sha256="b" * 64,
            )
        )
        state = hooks._WorldAnchorState(
            world_frame_id="backend_world_m",
            world_frame_revision="rev-b",
            world_transform_sha256="transform-b",
            last_output_world_x=9.0,
            last_output_world_z=8.0,
            last_output_media_pts_ns=900_000_000,
            last_output_filter_ts=9.0,
            last_output_trail_segment_id=4,
        )
        state.tracker_lifecycle_generation = 3
        processor._world_state_by_track[(2, 7)] = state
        track = {
            "camera_id": "kitchen",
            "tracker_id": 7,
            "frame_id": 10,
            "world_valid": True,
            "world": [1.0, 0.0, 2.0],
            "media_pts_ns": 1_000_000_000,
            "trail_segment_id": 5,
            "world_source": "pose_depth_fused",
            "world_measurement_accepted": True,
            **current_authority,
        }
        track[changed_field] = stale_value

        processor._commit_enqueued_world_output_watermarks(
            2,
            [track],
            filter_ts=10.0,
            admitted_world_track_keys={
                (
                    7,
                    int(track["tracker_lifecycle_generation"]),
                    10,
                )
            },
        )

        assert (
            state.last_output_world_x,
            state.last_output_world_z,
            state.last_output_media_pts_ns,
            state.last_output_filter_ts,
            state.last_output_trail_segment_id,
        ) == (9.0, 8.0, 900_000_000, 9.0, 4), changed_field


def test_current_output_authority_can_advance_live_state() -> None:
    processor = _processor()
    processor._world_frame = "backend_world_m"
    processor._source_epoch_by_sensor = {2: 7}
    processor.bev_calibration = SimpleNamespace(
        world_snapshot=lambda _source_id, _camera_id: SimpleNamespace(
            world_frame_id="backend_world_m",
            world_frame_revision="rev-b",
            frame_transform_sha256="transform-b",
            camera_calibration_sha256="b" * 64,
        )
    )
    state = hooks._WorldAnchorState(
        world_frame_id="backend_world_m",
        world_frame_revision="rev-b",
        world_transform_sha256="transform-b",
    )
    state.tracker_lifecycle_generation = 3
    processor._world_state_by_track[(2, 7)] = state
    track = {
        "camera_id": "kitchen",
        "tracker_id": 7,
        "tracker_lifecycle_generation": 3,
        "source_epoch": 7,
        "frame_id": 10,
        "world_valid": True,
        "world": [1.0, 0.0, 2.0],
        "media_pts_ns": 1_000_000_000,
        "trail_segment_id": 5,
        "world_frame": "backend_world_m",
        "world_frame_revision": "rev-b",
        "world_transform_sha256": "transform-b",
        "world_calibration_sha256": "b" * 64,
        "world_source": "pose_depth_fused",
        "world_measurement_accepted": True,
    }

    processor._commit_enqueued_world_output_watermarks(
        2,
        [track],
        filter_ts=10.0,
        admitted_world_track_keys={(7, 3, 10)},
    )

    assert (
        state.last_output_world_x,
        state.last_output_world_z,
        state.last_output_media_pts_ns,
        state.last_output_filter_ts,
        state.last_output_trail_segment_id,
    ) == (1.0, 2.0, 1_000_000_000, 10.0, 5)


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
