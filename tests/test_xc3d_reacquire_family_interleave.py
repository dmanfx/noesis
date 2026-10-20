"""Public PGS transitions with synthetic exact-original receipts; no native proof."""
import numpy as np
import pytest

from noesis.telemetry.person_ground_state import (
    HumanGroundConfig, PersonGroundState, RegisteredAnkleOriginal,
    bind_world_frame, mark_world_measurement_unavailable, update_human_cv_filter,
)


BINDING = ('run', 'graph', 0, 'source', '2:3:3', 3, 'family-room', 2, 3,
           'backend_world_m', 'revision', 'calibration', 'transform', None)
ENTRY = (12.302637219096829, 5.7345975115216525)


def _cursor(state):
    return tuple(getattr(state, name) for name in (
        'reacquire_candidate_x', 'reacquire_candidate_z', 'reacquire_candidate_ts',
        'reacquire_candidate_basis', 'reacquire_candidate_exact_basis',
        'reacquire_count', 'reacquire_unavailable_rows'))


def _update(state, cfg, frame, point, *, depth=False, quality=None, binding=BINDING,
            support=True, basis=None, typed=True):
    now = frame / 30.
    proof = RegisteredAnkleOriginal(BINDING + ('registration', 'model'), frame,
        round(now * 1e9), round(now * 1e6), now, tuple(point)) if depth and typed else None
    return update_human_cv_filter(state, measurement=np.array([point[0], 0., point[1]]),
        floor_y=0., now_ts=now, quality=quality or ('good' if depth else 'weak'), config=cfg,
        contact_basis=basis or ('depth:pose_ankle_support' if depth else 'pose_ankle_floor'),
        selected_geometry_kind='registered_depth' if depth else 'floor_ray',
        registered_ankle_original=proof, registered_ankle_binding=binding,
        media_pts_ns=round(now * 1e9), verified_reacquire_support=support)


def _two_current_depth_votes(*, typed=True):
    state = PersonGroundState(posture='standing', motion_mode='walk')
    cfg = HumanGroundConfig()
    _update(state, cfg, 1129, ENTRY, depth=True)
    assert state.measurement_accepted
    for frame in (1130, 1142):
        _update(state, cfg, frame, ENTRY)
    _update(state, cfg, 1143, (11.941131393967467, 5.78549603921268), depth=True, typed=typed)
    assert not state.measurement_accepted and state.reacquire_count == 1
    for frame in (1144, 1145, 1147, 1148):
        _update(state, cfg, frame, ENTRY)
    _update(state, cfg, 1149, (11.9877011084226, 5.792471157262457), depth=True, typed=typed)
    assert not state.measurement_accepted and state.reacquire_count == 2
    assert state.reacquire_candidate_basis == 'depth:pose_ankle_support'
    for frame in (1150, 1151):
        _update(state, cfg, frame, ENTRY)
    return state, cfg


def test_rejected_weak_floor_cannot_replace_two_fresh_registered_depth_votes():
    state, cfg = _two_current_depth_votes()
    before = _cursor(state)
    original = state.ankle_original_cursor
    old_good = (state.last_good_world, state.last_good_ts, state.last_output_media_pts_ns)
    for frame in (1152, 1153, 1154):
        output = _update(state, cfg, frame, (11.94639919149628, 5.693029791162329))
        assert not state.measurement_accepted and not state.reacquired
        assert state.accepted_ankle_original is None  # Accepted1129 proof really expired.
        assert _cursor(state) == before
        assert state.ankle_original_cursor is original
        assert np.allclose(output, [ENTRY[0], 0., ENTRY[1]])
        assert (state.last_good_world, state.last_good_ts, state.last_output_media_pts_ns) == old_good
    _update(state, cfg, 1155, (11.95, 5.78), depth=True)
    assert state.measurement_accepted and state.reacquired
    assert (state.world_x, state.world_z) == pytest.approx((11.95, 5.78))
    assert state.trail_break_required and state.trail_segment_id == 1


def test_admitted_floor_still_clears_pending_depth_without_becoming_depth():
    state, cfg = _two_current_depth_votes()
    original = state.ankle_original_cursor
    _update(state, cfg, 1152, (ENTRY[0] + .30, ENTRY[1]))
    assert state.measurement_accepted and not state.reacquired
    assert state.reacquire_count == 0
    assert state.accepted_metric_contact_basis == 'pose_ankle_floor'
    assert state.ankle_original_cursor is original
    assert state.accepted_ankle_original is None


@pytest.mark.parametrize('quality', ['estimated', 'good'])
def test_qualified_floor_keeps_existing_family_transition(quality):
    state, cfg = _two_current_depth_votes()
    _update(state, cfg, 1152, (11.94639919149628, 5.693029791162329), quality=quality)
    assert not state.measurement_accepted
    assert state.reacquire_count == 1 and state.reacquire_candidate_basis == 'pose_floor'


@pytest.mark.parametrize('binding', [None,
    BINDING[:2] + (1,) + BINDING[3:],  # Source epoch.
    BINDING[:4] + ('2:3:4',) + BINDING[5:],  # Native lifecycle.
    BINDING[:10] + ('foreign-revision',) + BINDING[11:]])
def test_missing_or_foreign_current_binding_cannot_preserve_old_votes(binding):
    state, cfg = _two_current_depth_votes()
    _update(state, cfg, 1152, (11.94639919149628, 5.693029791162329), binding=binding)
    assert not state.measurement_accepted and state.reacquire_candidate_basis == 'pose_floor'
    _update(state, cfg, 1153, (11.95, 5.78), depth=True)
    assert not state.measurement_accepted and state.reacquire_count == 1


def test_untyped_depth_cursor_gets_no_new_rejection_preservation():
    state, cfg = _two_current_depth_votes(typed=False)
    _update(state, cfg, 1152, (11.94639919149628, 5.693029791162329))
    assert not state.measurement_accepted
    assert state.reacquire_count == 1 and state.reacquire_candidate_basis == 'pose_floor'


@pytest.mark.parametrize('basis', ['depth:pose_torso_support', 'depth:pose_shoulder_support'])
def test_sibling_anatomy_is_not_an_ankle_vote(basis):
    state, cfg = _two_current_depth_votes()
    update_human_cv_filter(state, measurement=np.array([11.95, 0., 5.78]), floor_y=0.,
        now_ts=1152/30., quality='estimated', config=cfg, contact_basis=basis,
        selected_geometry_kind='registered_depth', verified_reacquire_support=True,
        registered_ankle_binding=BINDING, media_pts_ns=38_400_000_000)
    assert not state.measurement_accepted and not state.reacquired
    assert state.reacquire_count == 1 and state.reacquire_candidate_basis != 'depth:pose_ankle_support'


def test_rejected_floor_cannot_refresh_pending_deadline_or_bootstrap_after_expiry():
    state, cfg = _two_current_depth_votes()
    original_ts = state.reacquire_candidate_ts
    _update(state, cfg, 1152, (11.94639919149628, 5.693029791162329))
    assert state.reacquire_candidate_ts == original_ts
    _update(state, cfg, 1173, (30., 30.))
    assert not state.measurement_accepted and not state.reacquired
    assert state.reacquire_candidate_basis == 'pose_floor' and state.reacquire_count == 1
    _update(state, cfg, 1174, (11.95, 5.78), depth=True)
    assert not state.measurement_accepted and state.reacquire_count == 1


def test_replayed_current_depth_is_not_a_third_vote_or_time_renewal():
    state, cfg = _two_current_depth_votes()
    before = _cursor(state)
    _update(state, cfg, 1149, (11.9877011084226, 5.792471157262457), depth=True)
    assert not state.measurement_accepted and state.measurement_rejection_reason == 'registered_ankle_original_invalid'
    assert _cursor(state) == before


def test_incoherent_new_current_depth_restarts_its_own_consensus():
    state, cfg = _two_current_depth_votes()
    _update(state, cfg, 1152, (30., 30.), depth=True)
    assert not state.measurement_accepted and not state.reacquired
    assert state.reacquire_count == 1
    assert (state.reacquire_candidate_x, state.reacquire_candidate_z) == (30., 30.)


@pytest.mark.parametrize('bad_time', [1151/30., float('nan')])
def test_invalid_or_nonmonotone_time_cannot_preserve_pending_depth(bad_time):
    state, cfg = _two_current_depth_votes()
    update_human_cv_filter(state, measurement=np.array([11.95, 0., 5.78]), floor_y=0.,
        now_ts=bad_time, quality='weak', config=cfg, contact_basis='pose_ankle_floor',
        selected_geometry_kind='floor_ray', registered_ankle_binding=BINDING)
    assert not state.measurement_accepted and state.reacquire_count == 0


def test_unavailable_callbacks_and_frame_reset_never_renew_or_reuse_votes():
    state, cfg = _two_current_depth_votes()
    original_ts = state.reacquire_candidate_ts
    mark_world_measurement_unavailable(state, reason='no_current_measurement', now_ts=1152/30., config=cfg)
    assert not state.measurement_accepted
    assert state.reacquire_count == 2 and state.reacquire_candidate_ts == original_ts
    mark_world_measurement_unavailable(state, reason='no_current_measurement', now_ts=1173/30., config=cfg)
    assert state.reacquire_count == 0
    bind_world_frame(state, world_frame_id='backend_world_m', world_frame_revision='revision',
        world_transform_sha256='transform')
    assert bind_world_frame(state, world_frame_id='backend_world_m', world_frame_revision='new-revision',
        world_transform_sha256='new-transform')
    assert state.reacquire_count == 0 and state.ankle_original_cursor is None
