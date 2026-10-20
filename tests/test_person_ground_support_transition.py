"""Literal R45 same-torso upgrade, separate from floor/column relocations."""
import numpy as np
import pytest
from dataclasses import replace
from noesis.telemetry.person_ground_state import (
    HumanGroundConfig, PersonGroundState, update_human_cv_filter,
    RegisteredBodyColumnOriginal, RegisteredTorsoUpgradeOriginal,
    mark_world_measurement_unavailable, bind_world_frame,
    advance_human_cv_prediction,
)

F56 = (11.72348655363943, 5.2046049228839735)
F62 = (11.720459386755074, 5.204580050233755)
F68 = (11.70147498911205, 5.16356353286445)
F80 = (12.182727563479446, 6.155684695672381)
F81 = (12.187721197979894, 6.176749633033857)
BINDING = ('run', 'graph', 0, 'source_sr', 'F4:g1', 'calibration', 'registration')
COV = (.14489056175392007, .2864793573553259, .2864793573553259, .5677773001145622)


def original(frame, point, *, covariance=COV):
    return RegisteredBodyColumnOriginal(BINDING, frame, int(round(frame/30*1e9)),
        1792130840000000+frame*33333, frame/30, point, covariance)


def seeded():
    s = PersonGroundState(world_x=F56[0], world_z=F56[1], filtered_ts=56/30,
        accepted_metric_contact_basis='depth:registered_body_column', posture='sitting')
    s.accepted_column_original = original(56, F56)
    s.column_original_cursor = s.accepted_column_original
    return s


def apply(s, frame, point, *, proof=True, basis='depth:pose_torso_support', quality='estimated'):
    p = RegisteredTorsoUpgradeOriginal(original(frame, point), 'sitting', 'seat', 9.) if proof else None
    out = update_human_cv_filter(s, measurement=np.array([point[0], 0., point[1]]),
        floor_y=0., now_ts=frame/30, quality=quality, config=HumanGroundConfig(),
        contact_basis=basis, verified_reacquire_support=True,
        registered_torso_upgrade=p, media_pts_ns=int(round(frame/30*1e9)))
    return out


def test_r89_current_qualified_same_torso_upgrade():
    state = seeded()
    apply(state, 62, F62)
    assert state.measurement_accepted
    assert state.trail_break_required and state.trail_segment_id == 1
    assert not state.trail_append_allowed and state.vel_world_x == state.vel_world_z == 0.
    assert state.accepted_metric_contact_basis == 'depth:pose_torso_support'
    apply(state, 68, F68, proof=False, quality='weak')
    assert state.measurement_accepted and state.trail_segment_id == 1
    stamp=state.filtered_ts
    mark_world_measurement_unavailable(state, reason='pose_anatomy_unavailable', now_ts=74/30, config=HumanGroundConfig())
    assert state.filtered_ts == stamp and not state.measurement_accepted
    # The floor is >1m away: not a same-coordinate support upgrade.
    assert np.linalg.norm(np.array(F80)-np.array(F56)) > 1.
    advance_human_cv_prediction(state,floor_y=0.,now_ts=79/30,config=HumanGroundConfig())
    apply(state, 80, F80, proof=False, basis='pose_ankle_floor')
    assert not state.measurement_accepted and state.reacquire_count == 1


@pytest.mark.parametrize('defect', ['missing','foreign','stale','future','covariance','expired','shoulder','floor','large','weak','mahalanobis','posture','missing-prior'])
def test_r89_no_generic_basis_or_boolean_upgrade(defect):
    s=seeded(); point=F62; frame=62; basis='depth:pose_torso_support'
    o=original(frame,point)
    if defect=='foreign': o=replace(o,binding=('foreign',))
    elif defect=='stale': o=replace(o,frame_id=56)
    elif defect=='future': o=replace(o,media_pts_ns=o.media_pts_ns+1)
    elif defect=='covariance': o=replace(o,covariance_xz=(0.,0.,0.,0.))
    elif defect=='expired': frame=90; o=original(frame,point)
    elif defect=='shoulder': basis='depth:pose_shoulder_support'
    elif defect=='floor': basis='pose_ankle_floor';point=F80;o=original(frame,point)
    elif defect=='large': point=(15.,7.);o=original(frame,point)
    elif defect=='mahalanobis':
        point=(F56[0]+.5,F56[1]);o=original(frame,point,covariance=(1e-6,0.,0.,1e-6))
        s.accepted_column_original=replace(s.accepted_column_original,covariance_xz=(1e-6,0.,0.,1e-6))
    elif defect=='posture':s.posture='unknown'
    elif defect=='missing-prior':s.accepted_column_original=None
    proof=None if defect=='missing' else RegisteredTorsoUpgradeOriginal(o,'sitting','seat',9.)
    update_human_cv_filter(s,measurement=np.array([point[0],0.,point[1]]),floor_y=0.,now_ts=frame/30,
        quality='weak' if defect=='weak' else 'estimated',config=HumanGroundConfig(),contact_basis=basis,verified_reacquire_support=True,
        registered_torso_upgrade=proof,media_pts_ns=int(round(frame/30*1e9)))
    assert not s.measurement_accepted and s.trail_segment_id==0


def test_r89_k299_to303_is_not_a_covariance_consistent_coordinate_upgrade():
    old=(16.384453156998077,7.348514858705489); new=(15.422930787504207,4.687455241765517)
    s=PersonGroundState(world_x=old[0],world_z=old[1],filtered_ts=299/30,
        accepted_metric_contact_basis='pose_single_ankle_floor',posture='standing')
    token=replace(original(303,new),sample_ts=303/30)
    update_human_cv_filter(s,measurement=np.array([new[0],0.,new[1]]),floor_y=0.,now_ts=303/30,
        quality='estimated',config=HumanGroundConfig(),contact_basis='depth:registered_body_column',
        registered_column_original=token,media_pts_ns=token.media_pts_ns)
    assert not s.measurement_accepted and s.reacquire_count==1
    assert s.measurement_innovation_m > 2.8


def test_r89_real_frame_reset_discards_upgrade_custody():
    s=seeded()
    bind_world_frame(s,world_frame_id='backend_world_m',world_frame_revision='old',world_transform_sha256='a'*64)
    bind_world_frame(s,world_frame_id='backend_world_m',world_frame_revision='new',world_transform_sha256='b'*64)
    assert s.accepted_column_original is None


def test_r89_actual_receipt_caller_builds_bound_upgrade_and_rejects_cached_foreign():
    from DS9.tests.test_registered_body_column import _r75_current_column, _r75_apply
    from types import SimpleNamespace as NS
    from noesis.pipelines import hooks
    from noesis_core.world.resolver import UniversalWorldMeasurementResolver
    s=PersonGroundState()
    for frame in (44,50,56):
        p,t,c,d=_r75_current_column(frame,int(frame/30*1e9))
        _r75_apply(p,t,c,d,s)
        assert s.measurement_accepted == (frame==56)
    assert s.accepted_column_original is s.column_original_cursor
    assert s.accepted_column_original.covariance_xz is not None
    p,t,c,d=_r75_current_column(62,2_066_666_666)
    p._world_resolver=UniversalWorldMeasurementResolver()
    p.depth_registration=NS();p.scene_priors=None
    p._depth_candidate_covariance=lambda *args,**kwargs:c.covariance
    candidates=[]
    p._append_universal_world_candidates(candidates,sensor_id=2,track=t,camera_id='family-room',
        calib=NS(floor_y=0.,camera_id='family-room'),floor_candidate=None,floor_ray_admitted=False,
        anchor_candidate=None,pose_uv=None,contact_basis=None,posture='sitting',observed_posture='sitting',
        occlusion_fraction=0.,image_motion_supported=False,
        depth_obs=np.array([c.position.x,.414,c.position.z]),depth_weight=.9,
        depth_observation=hooks._DepthObservationResult(np.array([c.position.x,.414,c.position.z]),.9,'ok',
            registered_depth_m=7.2,registration_status='ok',registration_id='reg'),
        depth_result=d,person_anchor=p._resolve_person_depth_anchor(d),reject_current_geometry=False,
        flip_u=False,flip_v=False)
    p._world_resolver_diag_max_candidates=4
    typed=p._build_world_measurement_set(cohort=c.cohort,candidates=candidates)
    c=next(x for x in typed.hypotheses if x.candidate_id=='registered_depth')
    assert c.anchor=='person_body_projection' and c.support_state=='seat' and c.posture=='sitting'
    point=np.array([c.position.x,c.position.y,c.position.z])
    token=p._registered_column_original(track=t,selected=c,depth_result=d,measurement=point,now_ts=t['media_pts_ns']/1e9,
        torso_upgrade=True)
    assert type(token) is RegisteredTorsoUpgradeOriginal
    s.posture='sitting'
    p._update_track_world_state(t,s,measurement=point,floor_y=0.,now_ts=t['media_pts_ns']/1e9,alpha=.3,beta=.15,
        quality='estimated',contact_basis='depth:pose_torso_support',verified_reacquire_support=True,
        registered_torso_upgrade=token,media_pts_ns=t['media_pts_ns'])
    assert s.measurement_accepted and s.trail_break_required
    assert p._registered_column_original(track=t,selected=c,depth_result=replace(d,measurement_cached=True),
        measurement=point,now_ts=62/30,torso_upgrade=True) is None
    t['_xc3d_body_depth']=replace(t['_xc3d_body_depth'],registration_status='foreign')
    assert p._registered_column_original(track=t,selected=c,depth_result=d,
        measurement=point,now_ts=62/30,torso_upgrade=True) is None
