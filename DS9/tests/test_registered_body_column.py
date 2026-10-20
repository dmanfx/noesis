from types import SimpleNamespace as NS
import ast
from pathlib import Path
import numpy as np
import pytest
from noesis.pipelines import hooks
from noesis.v3dt_assets import registered_body_column_enabled, V3DTAssetError
from noesis.metadata.object_depth import ObjectDepthResult
from noesis_core.contracts.base import Matrix3
from noesis.telemetry.person_ground_state import HumanGroundConfig, PersonGroundState, update_human_cv_filter, mark_world_measurement_unavailable


def _r75_current_column(frame=179, pts_ns=6_000_000_000):
    from noesis.metadata.object_depth import NativePosePatchSupport,PoseDepthSample
    from noesis.pipelines.source_acquisition import OriginalCohort
    c=_column();c=c.model_copy(update={'cohort':c.cohort.model_copy(update={
        'frame_id':frame,'observed_at_us':pts_ns//1000,'world_transform_sha256':'a'*64})})
    p=object.__new__(hooks._AnalyticsTelemetryProcessor)
    p.pipeline=NS(config={'registered_body_column':True,'xc3d_world_quality':{
        'contract':'noesis.xc3d.world_quality.v1','enabled':True}})
    p._human_ground_cfg=HumanGroundConfig()
    patch=NativePosePatchSupport('pose_torso_support',(492.,465.),
        ((492.,450.,492.,480.,5.,5.),),(1920,1080),frame,pts_ns//1000)
    sample=PoseDepthSample('pose_torso_support',(492.,465.),44,1.,44,12.6,12.5,12.7,'good')
    depth=ObjectDepthResult(2,frame,4,0,(448.,431.,95.,114.),.9,'pose_capsule_native','ok','m',True,44,1.,
        anchor_source='pose_torso_support',anchor_uv=(492.,465.),anchor_depth_m=12.6,
        anchor_sample_count=44,anchor_valid_fraction=1.,pose_depth_samples=(sample,),native_pose_support=(patch,),
        ts_us=pts_ns//1000,measurement_frame_id=frame,measurement_ts_us=pts_ns//1000,
        measurement_cached=False,depth_tensor_frame_id=frame,depth_tensor_ts_us=pts_ns//1000,
        depth_tensor_age_frames=0,depth_tensor_age_us=0)
    original=OriginalCohort('run','graph','family-room',2,0,frame,pts_ns,0,'source_sr')
    track=dict(source_id=2,tracker_id=4,frame_id=frame,media_pts_ns=pts_ns,image_size=(1920,1080),
        observed_at_us=pts_ns//1000,tracker_lifecycle_generation=4,_xc3d_original_frame_id=frame,
        _xc3d_current_original=original)
    observation=hooks._DepthObservationResult(np.array([c.position.x,.414,c.position.z]),.9,'ok',
        raw_depth_m=12.6,registered_depth_m=7.2,registration_status='ok',registration_id='reg')
    anchor=p._resolve_person_depth_anchor(depth)
    track['_xc3d_body_depth']=p._original_body_depth_receipt(cohort=c.cohort,depth_result=depth,
        selected_observation=observation,selected_anchor=anchor,calib=NS(image_size=(1920,1080)),
        track=track,pose_is_current=True,flip_u=False,flip_v=False)
    assert track['_xc3d_body_depth'] is not None
    return p,track,c,depth


def _r75_apply(p,track,c,depth,state):
    point=np.array([c.position.x,c.position.y,c.position.z]);ts=track['media_pts_ns']/1e9
    proof=p._registered_column_original(track=track,selected=c,depth_result=depth,measurement=point,now_ts=ts)
    assert proof is not None
    out=p._update_track_world_state(track,state,measurement=point,floor_y=0.,now_ts=ts,
        alpha=.3,beta=.15,quality='weak',contact_basis='depth:registered_body_column',
        media_pts_ns=track['media_pts_ns'],registered_column_original=proof)
    return out,proof


def _r75_occluded_state():
    s=PersonGroundState();s.world_x=1.;s.world_z=2.;s.filtered_ts=5.8
    s.accepted_metric_contact_basis='pose_ankle_floor';s.post_occlusion_reacquire_support_required=True
    return s


def test_r75_actual_receipt_producer_pgs_three_columns_reanchor_without_floor_claim():
    s=_r75_occluded_state();segment=s.trail_segment_id
    for i in range(3):
        p,t,c,d=_r75_current_column(179+i*6,6_000_000_000+i*200_000_000)
        out,_=_r75_apply(p,t,c,d,s)
        assert s.measurement_accepted==(i==2)
        if i<2:
            assert s.reacquire_count==i+1
            stamp=s.reacquire_candidate_ts
            mark_world_measurement_unavailable(s,reason='depth_measurement_not_current',now_ts=stamp+.05,config=p._human_ground_cfg)
            p._update_track_world_state({},s,measurement=np.array([8.,0.,3.]),floor_y=0.,now_ts=stamp+.1,
                alpha=.3,beta=.15,quality='weak',contact_basis='bbox:bottom_center')
            assert s.reacquire_count==i+1 and s.reacquire_candidate_ts==stamp
    assert s.reacquired and s.trail_break_required and s.trail_segment_id==segment+1
    assert tuple(out[[0,2]])==(c.position.x,c.position.z)
    assert s.posture=='unknown' and s.height_ref_scene is None and s.seated_torso_height_scene is None


@pytest.mark.parametrize('defect',['cached','old','future','birth','registration','original-pts','disabled'])
def test_r75_current_original_binding_refusals(defect):
    from dataclasses import replace
    p,t,c,d=_r75_current_column();s=_r75_occluded_state();_r75_apply(p,t,c,d,s)
    p,t,c,d=_r75_current_column(185,6_200_000_000)
    if defect=='cached':d=replace(d,measurement_cached=True)
    elif defect=='old':d=replace(d,measurement_frame_id=179)
    elif defect=='future':d=replace(d,measurement_frame_id=191)
    elif defect=='birth':t['tracker_lifecycle_generation']=5
    elif defect=='registration':t['_xc3d_body_depth']=replace(t['_xc3d_body_depth'],registration_id='foreign')
    elif defect=='original-pts':t['_xc3d_current_original']=replace(t['_xc3d_current_original'],pts_ns=6_400_000_000)
    else:p.pipeline.config['registered_body_column']=False
    point=np.array([c.position.x,0.,c.position.z])
    proof=p._registered_column_original(track=t,selected=c,depth_result=d,measurement=point,now_ts=6.2)
    if defect=='registration':
        assert proof is not None
        p._update_track_world_state(t,s,measurement=point,floor_y=0.,now_ts=6.2,alpha=.3,beta=.15,
            quality='weak',contact_basis='depth:registered_body_column',media_pts_ns=t['media_pts_ns'],registered_column_original=proof)
        assert not s.measurement_accepted and s.measurement_rejection_reason=='registered_body_column_original_invalid'
    else:assert proof is None
    assert not s.reacquired


def test_r75_duplicate_and_replay_do_not_count_or_refresh_expired_run():
    s=_r75_occluded_state();p,t,c,d=_r75_current_column();_,proof=_r75_apply(p,t,c,d,s)
    stamp=s.reacquire_candidate_ts
    p._update_track_world_state(t,s,measurement=np.array([c.position.x,0.,c.position.z]),floor_y=0.,now_ts=6.,
        alpha=.3,beta=.15,quality='weak',contact_basis='depth:registered_body_column',
        media_pts_ns=t['media_pts_ns'],registered_column_original=proof)
    assert not s.measurement_accepted and s.reacquire_count==1 and s.reacquire_candidate_ts==stamp
    mark_world_measurement_unavailable(s,reason='no_current_depth',now_ts=6.8,config=p._human_ground_cfg)
    assert s.reacquire_count==0
    p,t,c,d=_r75_current_column(209,7_000_000_000);_r75_apply(p,t,c,d,s)
    assert s.reacquire_count==1 and not s.measurement_accepted


def test_r75_qualified_contradiction_resets_column_consensus_and_untyped_remains_strict():
    s=_r75_occluded_state();p,t,c,d=_r75_current_column();_r75_apply(p,t,c,d,s)
    p._update_track_world_state({},s,measurement=np.array([1.01,0.,2.]),floor_y=0.,now_ts=6.1,
        alpha=.3,beta=.15,quality='good',contact_basis='pose_ankle_floor',verified_reacquire_support=True)
    assert s.reacquire_count==0
    p,t,c,d=_r75_current_column(185,6_200_000_000);_r75_apply(p,t,c,d,s)
    assert s.reacquire_count==1 and not s.measurement_accepted
    s=_r75_occluded_state()
    p._update_track_world_state({},s,measurement=np.array([8.,0.,3.]),floor_y=0.,now_ts=6.,
        alpha=.3,beta=.15,quality='weak',contact_basis='depth:registered_body_column')
    assert s.reacquire_count==0 and s.measurement_rejection_reason=='post_occlusion_reacquire_requires_verified_support'


@pytest.mark.parametrize('changed',['calibration','run','epoch','birth'])
def test_r75_foreign_binding_cannot_carry_pending_consensus_and_reset_clears_cursor(changed):
    from dataclasses import replace
    s=_r75_occluded_state();p,t,c,d=_r75_current_column();_r75_apply(p,t,c,d,s)
    p,t,c,d=_r75_current_column(185,6_200_000_000)
    if changed=='calibration':
        c=c.model_copy(update={'cohort':c.cohort.model_copy(update={'calibration_revision':'other'})})
        t['_xc3d_body_depth']=replace(t['_xc3d_body_depth'],cohort=c.cohort)
    elif changed=='birth':
        c=c.model_copy(update={'cohort':c.cohort.model_copy(update={'tracker_lifecycle_generation':5})})
        t['tracker_lifecycle_generation']=5
        t['_xc3d_body_depth']=replace(t['_xc3d_body_depth'],cohort=c.cohort)
    else:
        t['_xc3d_current_original']=replace(t['_xc3d_current_original'],
            **({'run_id':'other-run'} if changed=='run' else {'source_clock_epoch':1}))
    _r75_apply(p,t,c,d,s)
    assert not s.measurement_accepted and s.reacquire_count==0
    assert s.measurement_rejection_reason=='registered_body_column_original_invalid'
    # A real reset/new state has no old cursor or old consensus to inherit.
    fresh=_r75_occluded_state();_r75_apply(p,t,c,d,fresh)
    assert fresh.reacquire_count==1 and not fresh.measurement_accepted


def test_r75_inconsistent_current_column_starts_new_run_without_speed_waiver():
    from dataclasses import replace
    s=_r75_occluded_state();p,t,c,d=_r75_current_column();_r75_apply(p,t,c,d,s)
    p,t,c,d=_r75_current_column(185,6_200_000_000)
    c=c.model_copy(update={'position':c.position.model_copy(update={'x':100.})})
    body=t['_xc3d_body_depth'];t['_xc3d_body_depth']=replace(body,
        original_point=(100.,body.original_point[1],body.original_point[2]))
    _r75_apply(p,t,c,d,s)
    assert s.reacquire_count==1 and not s.measurement_accepted


def test_r75_older_foreign_replay_cannot_clear_newer_proof_and_real_frame_reset_clears():
    from dataclasses import replace
    from noesis.telemetry.person_ground_state import bind_world_frame
    s=_r75_occluded_state();p,t,c,d=_r75_current_column();_,old=_r75_apply(p,t,c,d,s)
    p,t,c,d=_r75_current_column(185,6_200_000_000);_r75_apply(p,t,c,d,s)
    stamp=s.reacquire_candidate_ts;cursor=s.column_original_cursor
    old=replace(old,binding=('foreign',)+old.binding[1:])
    p._update_track_world_state({},s,measurement=np.array([c.position.x,0.,c.position.z]),floor_y=0.,
        now_ts=6.,alpha=.3,beta=.15,quality='weak',contact_basis='depth:registered_body_column',
        media_pts_ns=old.media_pts_ns,registered_column_original=old)
    assert s.reacquire_count==2 and s.reacquire_candidate_ts==stamp and s.column_original_cursor is cursor
    bind_world_frame(s,world_frame_id='backend_world_m',world_frame_revision='old',world_transform_sha256='a'*64)
    assert bind_world_frame(s,world_frame_id='backend_world_m',world_frame_revision='new',world_transform_sha256='b'*64)
    assert s.column_original_cursor is None and s.reacquire_count==0


def test_r75_no_typed_cursor_no_new_weak_bbox_preservation():
    s=PersonGroundState();cfg=HumanGroundConfig()
    update_human_cv_filter(s,measurement=np.array([1.,0.,2.]),floor_y=0.,now_ts=1.,
        quality='weak',config=cfg,contact_basis='depth:registered_body_column')
    assert s.column_original_cursor is None
    update_human_cv_filter(s,measurement=np.array([1.,0.,2.]),floor_y=0.,now_ts=1.1,
        quality='weak',config=cfg,contact_basis='bbox:bottom_center')
    assert s.reacquire_candidate_basis!='depth:registered_body_column'


def test_r75_untyped_column_cannot_join_or_finalize_exact_original_run():
    s=_r75_occluded_state();p,t,c,d=_r75_current_column();_r75_apply(p,t,c,d,s)
    stamp=s.reacquire_candidate_ts
    p._update_track_world_state({},s,measurement=np.array([c.position.x,0.,c.position.z]),floor_y=0.,
        now_ts=6.1,alpha=.3,beta=.15,quality='weak',contact_basis='depth:registered_body_column',
        image_motion_supported=True,verified_reacquire_support=True)
    assert not s.measurement_accepted and s.reacquire_count==1 and s.reacquire_candidate_ts==stamp
    # The inverse splice is also prohibited: two old cold string-only rows
    # cannot be promoted by the first typed original.
    s=PersonGroundState()
    for ts in (5.6,5.8):
        update_human_cv_filter(s,measurement=np.array([c.position.x,0.,c.position.z]),floor_y=0.,
            now_ts=ts,quality='weak',config=p._human_ground_cfg,contact_basis='depth:registered_body_column')
    assert s.reacquire_count==2
    _r75_apply(p,t,c,d,s)
    assert not s.measurement_accepted and s.reacquire_count==1


def _r77_strong_fixture(frame=242,pts_ns=8_074_733_333):
    from dataclasses import replace
    from noesis.metadata.object_depth import PoseDepthSample,NativePosePatchSupport
    p,t,c,d=_r75_current_column(frame,pts_ns)
    t.update(camera_id='family-room',class_id=0,confidence=.43408203125,tracker_confidence=.5876390338,
        identity_lifecycle_disposition='continuous',bbox=tuple(d.bbox),_xc3d_native_pose_measured=True)
    joints=np.zeros((17,3),dtype=np.float64)
    for i,xy in ((5,(470.,450.)),(6,(520.,450.)),(11,(475.,500.)),(12,(515.,500.))):
        joints[i]=(*xy,.95)
    capsule,uv=hooks._ObjectDepthFusionProcessor._pose_torso_capsule_geometry(joints,bbox=tuple(d.bbox))
    assert capsule is not None
    sample=PoseDepthSample('pose_torso_support',uv,45,1.,45,12.6,12.5,12.7,'estimated')
    patch=NativePosePatchSupport('pose_torso_support',uv,(capsule,),(1920,1080),frame,pts_ns//1000)
    d=replace(d,anchor_uv=uv,pose_depth_samples=(sample,),native_pose_support=(patch,),
        pose_joint_confidences=tuple(float(v)for v in joints[:,2]))
    p._min_unposed_detection_confidence=.55;p.depth_registration=NS();p.scene_priors=None
    p._depth_candidate_covariance=lambda *a,**k:Matrix3(values=(.1,0.,0.,0.,.02,0.,0.,0.,.1))
    observation=hooks._DepthObservationResult(np.array([c.position.x,.414,c.position.z]),.9,'ok',
        raw_depth_m=12.6,registered_depth_m=7.2,registration_status='ok',registration_id='reg')
    # Explicit synthetic calibrated geometry boundary; no native/metric claim.
    p._depth_observation_from_anchor=lambda **kw:observation
    p._admit_world_observation_range=lambda *a,**k:True
    return p,t,c,d,joints


def _r77_resolved(p,t,c,d,joints):
    from noesis_core.world.resolver import UniversalWorldMeasurementResolver,WorldMeasurementResolverConfig
    proof=p._strong_current_column_pose(track=t,keypoints=joints,pose_is_current=True,depth_result=d)
    assert proof is not None
    rows=[]
    p._append_strong_posed_column(rows,proof=proof,track=t,calib=NS(camera_id='family-room',floor_y=0.),
        posture='unknown',flip_u=False,flip_v=False,occlusion_fraction=.35,image_motion_supported=False)
    assert len(rows)==1 and rows[0]['method']=='registered_body_column'
    assert rows[0]['support_state']=='unknown' and rows[0]['posture']=='unknown'
    p._world_resolver_diag_max_candidates=4
    p._world_resolver=UniversalWorldMeasurementResolver(config=WorldMeasurementResolverConfig(registered_body_column=True))
    typed=p._build_world_measurement_set(cohort=c.cohort,candidates=rows)
    result=p._world_resolver.resolve(typed)
    selected=next(h for h in typed.hypotheses if h.candidate_id==result.selected_candidate_id)
    assert selected.method=='registered_body_column'
    t['_xc3d_body_depth']=p._original_body_depth_receipt(cohort=c.cohort,depth_result=d,
        selected_observation=p._depth_observation_from_anchor(),selected_anchor=p._resolve_person_depth_anchor(d),
        calib=NS(image_size=(1920,1080)),track=t,pose_is_current=True,flip_u=False,flip_v=False)
    return selected


@pytest.mark.parametrize('detector,tracker',[(.43408203125,.5876390338),(.53515625,.5245158672)])
def test_r77_weak_unposed_gate_unchanged_strong_posed_emits_only_named_column(detector,tracker):
    p,t,c,d,joints=_r77_strong_fixture();t.update(confidence=detector,tracker_confidence=tracker)
    assert p._person_admission_confidence(t,trusted_lifecycle=True)<.60
    selected=_r77_resolved(p,t,c,d,joints)
    assert selected.kind=='registered_depth' and selected.support_state=='unknown'


@pytest.mark.parametrize('defect',['weak','partial','cached','foreign-joints','foreign-frame','foreign-bbox',
    'absent-detector','negative-detector','nan-detector','native-shadow','lost','class','geometry','off'])
def test_r77_strong_posed_provenance_semantic_and_geometry_negatives(defect):
    from dataclasses import replace
    p,t,c,d,joints=_r77_strong_fixture()
    if defect=='weak':joints[5,2]=.749;d=replace(d,pose_joint_confidences=tuple(joints[:,2]))
    elif defect=='partial':joints=joints[:16]
    elif defect=='cached':d=replace(d,measurement_cached=True)
    elif defect=='foreign-joints':joints[5,0]+=3.
    elif defect=='foreign-frame':d=replace(d,measurement_frame_id=d.frame_id+1)
    elif defect=='foreign-bbox':t['bbox']=(0.,0.,95.,114.)
    elif defect=='absent-detector':t.pop('confidence')
    elif defect=='negative-detector':t['confidence']=-.1
    elif defect=='nan-detector':t['confidence']=float('nan')
    elif defect=='native-shadow':t['_xc3d_native_pose_measured']=False
    elif defect=='lost':t['identity_lifecycle_disposition']='lost'
    elif defect=='class':t['class_id']=1
    elif defect=='geometry':joints[6]=joints[5];d=replace(d,pose_joint_confidences=tuple(joints[:,2]))
    else:p.pipeline.config['registered_body_column']=False
    assert p._strong_current_column_pose(track=t,keypoints=joints,pose_is_current=True,depth_result=d)is None


def test_r77_current_flag_and_malformed_posture_no_semantic_fallback():
    p,t,c,d,joints=_r77_strong_fixture()
    assert p._strong_current_column_pose(track=t,keypoints=joints,pose_is_current=False,depth_result=d)is None
    proof=p._strong_current_column_pose(track=t,keypoints=joints,pose_is_current=True,depth_result=d)
    rows=[]
    p._append_strong_posed_column(rows,proof=proof,track=t,calib=NS(camera_id='family-room',floor_y=0.),
        posture='fabricated',flip_u=False,flip_v=False,occlusion_fraction=0.,image_motion_supported=False)
    assert not rows


@pytest.mark.parametrize('refusal',['registration','range'])
def test_r77_strong_pose_cannot_replace_failed_registration_or_geometry(refusal):
    p,t,c,d,joints=_r77_strong_fixture()
    proof=p._strong_current_column_pose(track=t,keypoints=joints,pose_is_current=True,depth_result=d)
    if refusal=='registration':
        p._depth_observation_from_anchor=lambda **kw:hooks._DepthObservationResult(None,0.,'registration_unavailable')
    else:p._admit_world_observation_range=lambda *a,**k:False
    rows=[]
    p._append_strong_posed_column(rows,proof=proof,track=t,calib=NS(camera_id='family-room',floor_y=0.),
        posture='standing',flip_u=False,flip_v=False,occlusion_fraction=.35,image_motion_supported=False)
    assert not rows


def _column():
    from noesis_core.contracts.world_measurement import WorldMeasurementHypothesis,WorldMeasurementCohort,WorldPriorEvidence
    return WorldMeasurementHypothesis(candidate_id='registered_body_column',method='registered_body_column',
        kind='registered_depth',anchor='registered_body_column',posture='unknown',support_state='unknown',
        cohort=WorldMeasurementCohort(track_key='2:4:4',tracker_lifecycle_generation=4,camera_id='family-room',
            source_id=2,tracker_id=4,frame_id=36,observed_at_us=1200000,world_revision='bound',calibration_revision='cal',pcf_revision='bound'),
        position=dict(x=11.6678,y=0.,z=5.1446),covariance=dict(values=(.1,0.,0.,0.,.02,0.,0.,0.,.1)),
        pcf=WorldPriorEvidence(prior_id='prior',revision_id='bound',status='fail',inside_extent=True,
            inside_authored_space=False,evidence_observed=False,observed_confidence=0.,boundary_signed_distance_m=-3.229745,
            reasons=('outside_authored_space',)))


def test_r65_flag_default_off_strict_boolean_task_only():
    assert not registered_body_column_enabled({})
    with pytest.raises(V3DTAssetError):registered_body_column_enabled({'registered_body_column':1})
    with pytest.raises(V3DTAssetError):registered_body_column_enabled({'registered_body_column':True})


@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('raw_posture',['unknown','standing','sitting','lying'])
def test_r65_actual_body_candidate_no_floor_raw_unknown_never_seat(enabled,raw_posture):
    p=object.__new__(hooks._AnalyticsTelemetryProcessor)
    p.pipeline=NS(config={'registered_body_column':enabled,'xc3d_world_quality':{
        'contract':'noesis.xc3d.world_quality.v1','enabled':True}})
    p.depth_registration=NS();p.scene_priors=None
    p._depth_candidate_covariance=lambda *a,**kw:Matrix3(values=(.1,0.,0.,0.,.02,0.,0.,0.,.1))
    depth=ObjectDepthResult(source_id=2,frame_id=36,object_id=4,class_id=0,
        bbox=(448.,431.,95.,114.),score=.9,sampling_mode='pose_capsule_native',status='ok',unit='m',is_metric=True,
        sample_count=141,valid_fraction=1.,anchor_source='pose_torso_support',anchor_uv=(492.3476,465.2891),
        anchor_depth_m=12.6132,anchor_valid_fraction=1.,anchor_depth_spread_m=.1304569)
    anchor=p._resolve_person_depth_anchor(depth);assert anchor is not None
    candidates=[]
    p._append_universal_world_candidates(candidates,sensor_id=2,track={'tracker_id':4},camera_id='family-room',
        calib=NS(floor_y=0.,camera_id='family-room'),floor_candidate=None,floor_ray_admitted=False,
        anchor_candidate=None,pose_uv=None,contact_basis=None,
        posture='sitting' if raw_posture=='unknown' else raw_posture,observed_posture=raw_posture,
        occlusion_fraction=0.,image_motion_supported=False,depth_obs=np.array([11.6678,.414,5.1446]),depth_weight=.9,
        depth_observation=hooks._DepthObservationResult(np.array([11.6678,.414,5.1446]),.9,'ok',
            registered_depth_m=7.2082,registration_status='ok',registration_id='bound'),depth_result=depth,
        person_anchor=anchor,reject_current_geometry=False,flip_u=False,flip_v=False)
    assert len(candidates)==(2 if enabled else 1) and all(c['posture']==raw_posture for c in candidates)
    assert candidates[0]['method'] is None
    if enabled:
        assert candidates[1]['method']=='registered_body_column' and candidates[1]['support_state']=='unknown'
        assert candidates[1]['contact_basis']=='depth:registered_body_column'
        assert candidates[1]['position'][1]==0.
        # Actual appender -> typed set -> eligibility priority -> real resolver.
        from noesis_core.world.resolver import UniversalWorldMeasurementResolver,WorldMeasurementResolverConfig
        p._world_resolver_diag_max_candidates=4;p._min_unposed_detection_confidence=.55
        p._world_resolver=UniversalWorldMeasurementResolver(config=WorldMeasurementResolverConfig(registered_body_column=True))
        typed=p._build_world_measurement_set(cohort=_column().cohort,candidates=candidates)
        bases={c['candidate_id']:c['contact_basis'] for c in candidates}
        typed=p._body_column_fallback_set(typed,contact_basis=bases,track={},state=None,detector_confidence=.9)
        result=p._world_resolver.resolve(typed)
        selected=next(h for h in typed.hypotheses if h.candidate_id==result.selected_candidate_id)
        assert p._resolver_candidate_is_ground_footprint_supported(selected,contact_basis=bases[selected.candidate_id])
        assert selected.method==('registered_body_column' if raw_posture in {'standing','unknown'} else None)


@pytest.mark.parametrize('posture',['unknown','standing','sitting','lying'])
def test_r65_three_original_consensus_no_unavailable_renewal_and_basis_break(posture):
    state=PersonGroundState();state.posture=posture
    cfg=HumanGroundConfig();basis='depth:registered_body_column'
    for i in range(3):
        update_human_cv_filter(state,measurement=np.array([1.+i*.01,0.,2.]),floor_y=0.,
            now_ts=i*.2,quality='weak',config=cfg,contact_basis=basis)
        assert state.measurement_accepted==(i==2)
        if i<2:
            stamp=state.reacquire_candidate_ts
            mark_world_measurement_unavailable(state,reason='depth_measurement_not_current',now_ts=i*.2+.1,config=cfg)
            assert state.reacquire_candidate_ts==stamp
    assert state.height_ref_scene is None and state.seated_torso_height_scene is None
    update_human_cv_filter(state,measurement=np.array([1.02,0.,2.]),floor_y=0.,
        now_ts=.6,quality='estimated',config=cfg,contact_basis='pose_ankle_floor')
    assert not state.measurement_accepted and state.measurement_rejection_reason=='estimated_body_column_basis_changed'


def test_r65_exact_independent_dispatch_condition_no_floor_anchor():
    tree=ast.parse(Path(hooks.__file__).read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and 'force_occlusion_gravity or' in ast.unparse(n.test)
        and 'registered_body_column_enabled' in ast.unparse(n.test))
    env={'force_occlusion_gravity':False,'person_anchor':object(),'anchor_candidate':None,
        'self':NS(pipeline=NS(config={'registered_body_column':True,'xc3d_world_quality':{
            'contract':'noesis.xc3d.world_quality.v1','enabled':True}})),
        'registered_body_column_enabled':registered_body_column_enabled}
    assert eval(compile(ast.Expression(node.test),'actual_dispatch','eval'),env)
    env['self'].pipeline.config['registered_body_column']=False
    assert not eval(compile(ast.Expression(node.test),'actual_dispatch','eval'),env)


def test_r65_actual_earlier_torso_context_unknown_current_only():
    tree=ast.parse(Path(hooks.__file__).read_text())
    expr=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='torso_context_supported' for t in n.targets))
    config={'registered_body_column':True,'xc3d_world_quality':{'contract':'noesis.xc3d.world_quality.v1','enabled':True}}
    env=dict(raw_classified_posture='unknown',partial_body_without_floor_contact=False,
        pose_is_current=True,person_anchor=NS(contact_basis='depth:pose_torso_support'),
        self=NS(pipeline=NS(config=config),_depth_measurement_is_current=lambda *a,**k:True),
        depth_result=object(),track={},occlusion_assessment=None,
        registered_body_column_enabled=registered_body_column_enabled)
    assert eval(compile(ast.Expression(expr),'actual_context','eval'),env)
    env['pose_is_current']=False
    assert not eval(compile(ast.Expression(expr),'actual_context','eval'),env)
    env['pose_is_current']=True;config['registered_body_column']=False
    assert not eval(compile(ast.Expression(expr),'actual_context','eval'),env)


@pytest.mark.parametrize('reason',['no_valid_measurement','depth_measurement_not_current'])
def test_r65_unavailable_original_expires_pending_consensus_without_new_point(reason):
    state=PersonGroundState();cfg=HumanGroundConfig()
    update_human_cv_filter(state,measurement=np.array([1.,0.,2.]),floor_y=0.,now_ts=0.,
        quality='weak',config=cfg,contact_basis='depth:registered_body_column')
    mark_world_measurement_unavailable(state,reason=reason,now_ts=.8,config=cfg)
    assert state.reacquire_count==0 and state.world_x is None and not state.measurement_accepted


def test_r68_transition_keeps_three_samples_and_requires_genuine_verified_flag():
    state=PersonGroundState();cfg=HumanGroundConfig()
    for i in range(3):
        update_human_cv_filter(state,measurement=np.array([1.,0.,2.]),floor_y=0.,now_ts=i*.2,
            quality='weak',config=cfg,contact_basis='depth:registered_body_column')
    assert state.measurement_accepted
    segment=state.trail_segment_id
    for i in range(3,6):
        update_human_cv_filter(state,measurement=np.array([1.02,0.,2.]),floor_y=0.,now_ts=i*.2,
            quality='estimated',config=cfg,contact_basis='pose:upright_body_planes',verified_reacquire_support=False)
        assert not state.measurement_accepted
    assert state.reacquire_count==3
    update_human_cv_filter(state,measurement=np.array([1.02,0.,2.]),floor_y=0.,now_ts=1.2,
        quality='estimated',config=cfg,contact_basis='pose:upright_body_planes',verified_reacquire_support=True)
    assert state.measurement_accepted and state.trail_break_required and state.trail_segment_id==segment+1


@pytest.mark.parametrize('ordinary_kind',['floor','strong-floor','unsupported-gravity','upright4','upright5'])
def test_r68_only_eligible_ordinary_beats_real_fallback(ordinary_kind):
    from noesis_core.contracts.world_measurement import WorldMeasurementHypothesis,WorldMeasurementSet
    from noesis_core.world.resolver import UniversalWorldMeasurementResolver,WorldMeasurementResolverConfig
    c=_column();data=c.model_dump();data.update(candidate_id='ordinary',method=None,pcf=None,confidence=.99,support_score=.99)
    if ordinary_kind in {'floor','strong-floor'}:
        data.update(kind='floor_ray',anchor='pose_ankle_floor',support_state='floor',ray_incidence_sin=.5)
        if ordinary_kind=='strong-floor':data['pcf']=c.pcf.model_copy(update={'evidence_observed':True}).model_dump()
    else:data.update(kind='pose_scale',posture='standing',anchor='upright_body_plane' if ordinary_kind.startswith('upright') else 'learned_height')
    h=WorldMeasurementHypothesis.model_validate(data);rows=WorldMeasurementSet(cohort=c.cohort,hypotheses=(h,c))
    p=object.__new__(hooks._AnalyticsTelemetryProcessor);p._min_unposed_detection_confidence=.55
    p._world_resolver=UniversalWorldMeasurementResolver(config=WorldMeasurementResolverConfig(registered_body_column=True))
    track=dict(world_upright_body_anchor_count=5 if ordinary_kind=='upright5' else 4,
        world_upright_body_scatter_m=.1,world_upright_body_height_m=1.7)
    state=PersonGroundState();state.world_x=1.;state.world_z=2.
    chosen=p._body_column_fallback_set(rows,contact_basis={'ordinary':'pose_ankle_floor'},track=track,
        state=state,detector_confidence=.9)
    result=p._world_resolver.resolve(chosen)
    assert result.selected_candidate_id==('ordinary' if ordinary_kind in {'floor','upright5'} else 'registered_body_column')
    assert len(chosen.hypotheses)==2
