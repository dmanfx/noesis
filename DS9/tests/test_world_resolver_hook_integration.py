from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from noesis_core.contracts.base import Matrix3, Vector3
from geometry.homography import project_world_to_image
from noesis_core.contracts.world_measurement import (
    WorldMeasurementCohort,
    WorldMeasurementHypothesis,
    WorldMeasurementSet,
    WorldPriorEvidence,
)
from noesis.metadata.object_depth import ObjectDepthResult
from noesis.pipelines import hooks


def _processor(config: dict[str, object] | None = None):
    return hooks._AnalyticsTelemetryProcessor(  # type: ignore[attr-defined]
        pipeline=SimpleNamespace(
            config=config or {
                "canonical_world": {
                    "measurement_resolver": {
                        "enabled": True,
                        "max_range_m": 22.0,
                        "max_disagreement_m": 1.25,
                        "max_candidates": 4,
                    }
                }
            },
            frame_size=(1920, 1080),
        ),
        tracking_pub=SimpleNamespace(),
        camera_labels={0: "camera-0"},
        sensor_id_map={0: 0},
    )


@pytest.mark.parametrize('mode',['negative','anchor_negative','geometry_negative','current','budget','off'])
def test_r141_same_original_native_diagnostics_and_rendezvous(monkeypatch,mode):
    hooks.reset_core_path_instrumentation()
    monkeypatch.setenv('NOESIS_XC3D_BODY_DIAGNOSTIC_TARGETS_S','6')
    if mode!='off':monkeypatch.setenv('NOESIS_XC3D_POSE_DIAGNOSTIC_RANGE','1:6:420:470')
    else:monkeypatch.delenv('NOESIS_XC3D_POSE_DIAGNOSTIC_RANGE',raising=False)
    negative=mode in ('negative','anchor_negative','geometry_negative')
    monkeypatch.setenv('NOESIS_POSE_FEATURES_PER_FRAME_MAX','0' if mode in ('budget','anchor_negative') else '1')
    monkeypatch.setenv('NOESIS_POSE_ANCHOR_NATIVE_EXTRACTS_PER_FRAME_MAX','1' if mode=='anchor_negative' else '0')
    calls=[]
    def extract(obj,gie,w,h,score,letterbox,*optional):
        calls.append(len(optional))
        if optional:
            optional[0].update(tensor_count=1,context_present=True,context_valid=True,
                shape_valid=True,rows=300,cols=57,qualifying_rows_seen=0 if negative else 1,
                result='no_qualifying_row' if negative else 'decoded')
            if mode=='geometry_negative':
                optional[0].update(_i156_geometry())
                optional[0]['result']='context_no_matching_row'
        if negative:return None
        joints=[[float(i),float(i),.9] for i in range(17)]
        return dict(score=.9,keypoints_roi=joints,keypoints_abs=joints)
    monkeypatch.setattr(hooks,'noesis_pose_meta_ext',SimpleNamespace(
        extract_pose_keypoints=extract,attach_pose_features=lambda *args:True))
    pipeline=SimpleNamespace(config={
        'xc3d_world_quality':{'contract':'noesis.xc3d.world_quality.v1','enabled':True},
        'xc3d_image_regions':{'enabled':True}},stable_id_mgr=SimpleNamespace(active_tracks={}))
    producer=hooks.PoseFeatureProcessor(pipeline=pipeline,gie_id=4)
    obj=SimpleNamespace(object_id=6,class_id=0,rect_params=SimpleNamespace(
        left=30.,top=40.,width=100.,height=200.))
    frame=420;pts=frame*33_333_333+777
    receipt=dict(source_id=1,original_source_id=1,receipt_valid=True,
        sdk_source_id=1,original_sdk_source_id=1,slot=0,original_batch_id=0,
        original_buffer_scope=421,original_ntp_ns=0,post_tracker_ntp_ns=0,
        original_frame=frame,original_pts_ns=pts,post_tracker_pts_ns=pts)
    producer.handle_servicemaker_frame(object(),SimpleNamespace(source_id=1,
        frame_number=1320,buf_pts=pts,object_items=iter([obj])),source_receipt=receipt)
    found,value=producer._owned_pose_result((1,6,frame,pts),[30.,40.,100.,200.],(4,(640,640),.25,True))
    assert calls==([] if mode=='budget' else [0 if mode=='off' else 1])
    assert found==(mode!='budget')
    assert (value is not None)==(mode in ('current','off'))
    summary=hooks.get_core_path_instrumentation_snapshot()['registered_ankle_priority']
    if mode=='off':
        assert not summary['records'];return
    row=summary['records'][0]
    assert row['frame']==420 and row['pts_ns']==pts and row['generation'] is None
    assert row['lifecycle_available'] is False
    assert row['rendezvous_status']==('owned_negative' if negative else
        'object_or_bbox_missing' if mode=='budget' else 'owned_current')
    if negative:
        assert row['result']==('context_no_matching_row' if mode=='geometry_negative' else 'no_qualifying_row')
        assert row['tensor_count']==1
        assert row['first_pose_failure_frame']==420
        if mode=='geometry_negative':
            assert row['first_pre_iou_rejection']['frame']==420
            assert row['first_pre_iou_rejection']['pts_ns']==pts
            assert row['pre_iou_candidates'][0]['torso_unclipped'][0][0]==31.
    producer._owned_pose_result((1,6,421,pts+1),[30.,40.,100.,200.],(4,(640,640),.25,True))
    later=hooks.get_core_path_instrumentation_snapshot()['registered_ankle_priority']['records'][0]
    assert later['frame']==420 and later['rendezvous_frame']==421
    if negative:assert later['first_pose_failure_frame']==420
    assert len(calls)==(0 if mode=='budget' else 1)


def test_r141_pose_probe_shares_final_record_and_byte_bounds():
    import json
    observer=hooks._CorePathInstrumentation()
    for raw in range(10):
        observer.record_pose_probe(dict(source=1,raw=raw,frame=420,pts_ns=1,
            producer_status='anchor_attempt',native_returned=False,result='x'*80))
    result=observer.snapshot()['registered_ankle_priority']
    assert len(result['records'])==8 and result['omitted_records']==2
    assert len(json.dumps(result,sort_keys=True,allow_nan=False,separators=(',',':')).encode())<=8192
    observer.record_pose_probe(dict(source=1,raw=0,frame=421,pts_ns=2,producer_status='budget_denied'))
    row=observer.snapshot()['registered_ankle_priority']['records'][0]
    assert row['frame']==421 and 'result' not in row and 'native_returned' not in row
    assert row['first_pose_failure_frame']==420


@pytest.mark.parametrize('native_current',[True,False])
def test_r142_actual_projective_call_forwards_current_original_binding(monkeypatch,native_current):
    import ast
    p,t,proof=_r138_factory(monkeypatch)
    state=hooks._WorldAnchorState()
    point=np.array([proof.position_xz[0],0.,proof.position_xz[1]])
    p._update_track_world_state(t,state,measurement=point,floor_y=0.,now_ts=proof.sample_ts,
        alpha=.3,beta=.15,quality='good',contact_basis='depth:pose_ankle_support',
        registered_ankle_original=proof,registered_ankle_binding=proof.binding[:-2],
        selected_geometry_kind='registered_depth',media_pts_ns=proof.media_pts_ns)
    _,current,selected,_=_r131_ankle_original(frame=75)
    current['_xc3d_native_pose_measured']=native_current
    selected=selected.model_copy(update={'kind':'floor_ray'})
    hooks.mark_world_measurement_unavailable(state,reason='depth_measurement_not_current',
        now_ts=proof.sample_ts+1/30,config=p._human_ground_cfg)
    assert state.accepted_ankle_original is proof
    calls=[n for n in ast.walk(ast.parse(Path(hooks.__file__).read_text()))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
        and n.func.id=='integrate_projective_ground_observation']
    assert len(calls)==1
    namespace=dict(integrate_projective_ground_observation=hooks.integrate_projective_ground_observation,
        state=state,projective_point=point+np.array([.01,0.,.01]),calib=SimpleNamespace(floor_y=0.),
        now_ts=proof.sample_ts+1/30,self=p,track=current,selected_resolver_candidate=selected,
        kinematic_output_reference_at_entry=(point[0],point[2],proof.media_pts_ns,proof.sample_ts,0),
        output_reference_at_entry=None,display_recovery_reference_at_entry=None,filter_transition={})
    result=eval(compile(ast.Expression(calls[0]),'<actual projective caller>','eval'),namespace)
    assert result is not None and not state.measurement_accepted
    assert (state.accepted_ankle_original is proof)==native_current
    assert proof.sample_ts==proof.media_pts_ns/1e9 and state.ankle_original_cursor is proof


def test_r143_same_pass_native_subpredicate_scalars_are_bounded_and_not_grafted():
    observer=hooks._CorePathInstrumentation()
    observer.record_pose_probe(dict(source=1,raw=6,frame=420,pts_ns=1,producer_status='feature_attempt',
        native_returned=False,result='context_no_matching_row',finite_rows_seen=300,padding_rows_seen=298,
        score_pass_rows=2,class_pass_rows=2,bbox_pass_rows=2,center_pass_rows=0,iou_pass_rows=0,
        max_score=.9,max_iou=None))
    row=observer.snapshot()['registered_ankle_priority']['records'][0]
    assert row['score_pass_rows']==2 and row['max_score']==.9 and row['max_iou'] is None
    observer.record_pose_probe(dict(source=1,raw=6,frame=421,pts_ns=2,producer_status='budget_denied'))
    row=observer.snapshot()['registered_ankle_priority']['records'][0]
    assert 'score_pass_rows' not in row and 'max_score' not in row
    observer.record_pose_probe(dict(source=1,raw=6,frame=422,pts_ns=3,max_iou=float('nan')))
    assert observer.snapshot()['registered_ankle_priority']['records'][0]['frame']==421


def _i156_geometry():
    return dict(pre_iou_parent_bbox_xywh=[30.,40.,100.,200.],pre_iou_candidates_seen=3,
        pre_iou_scan_complete=True,pre_iou_invalid_row_index=None,
        pre_iou_candidates=[dict(row_index=i,score=.8,class_id=0.,iou=.49,
            bbox_xyxy=[20.,30.,140.,250.],torso_unclipped=[[31.,60.,.9],[80.,60.,.9],
                [31.,180.,.9],[80.,180.,.9]]) for i in range(2)])


def test_i156_first_geometry_rejection_is_detached_and_keeps_its_own_original():
    observer=hooks._CorePathInstrumentation()
    observer.record_pose_probe(dict(source=1,raw=6,frame=420,pts_ns=4200,
        native_returned=False,result='tensor_missing'))
    geometry=_i156_geometry()
    observer.record_pose_probe(dict(source=1,raw=6,frame=431,pts_ns=4310,
        native_returned=False,result='context_no_matching_row',**geometry))
    geometry['pre_iou_candidates'][0]['torso_unclipped'][0][0]=999.
    first=observer.snapshot()['registered_ankle_priority']['records'][0]
    assert first['first_pose_failure_frame']==420
    pinned=first['first_pre_iou_rejection']
    assert (pinned['frame'],pinned['pts_ns'])==(431,4310)
    assert pinned['pre_iou_candidates_seen']==3 and len(pinned['pre_iou_candidates'])==2
    assert pinned['pre_iou_candidates'][0]['torso_unclipped'][0][0]==31.
    pinned['pre_iou_candidates'][0]['iou']=999.
    # Later success/no-context and denied-original records cannot erase or
    # restamp the first real geometry failure, nor graft latest coordinates.
    observer.record_pose_probe(dict(source=1,raw=6,frame=432,pts_ns=4320,
        native_returned=True,result='decoded'))
    observer.record_pose_probe(dict(source=1,raw=6,frame=433,pts_ns=4330,producer_status='budget_denied'))
    row=observer.snapshot()['registered_ankle_priority']['records'][0]
    assert row['frame']==433 and 'pre_iou_candidates' not in row and 'result' not in row
    assert row['first_pre_iou_rejection']['pre_iou_candidates'][0]['iou']==.49
    assert (row['first_pre_iou_rejection']['frame'],row['first_pre_iou_rejection']['pts_ns'])==(431,4310)


@pytest.mark.parametrize('fault',['rows','torso','nan','bool-coordinate','unknown-key','bool-count','partial','invalid-index'])
def test_i156_nested_geometry_rejects_unbounded_or_malformed_values(fault):
    g=_i156_geometry()
    if fault=='rows':g['pre_iou_candidates'].append(g['pre_iou_candidates'][0].copy())
    elif fault=='torso':g['pre_iou_candidates'][0]['torso_unclipped'].append(None)
    elif fault=='nan':g['pre_iou_candidates'][0]['torso_unclipped'][0][0]=float('nan')
    elif fault=='bool-coordinate':g['pre_iou_parent_bbox_xywh'][0]=True
    elif fault=='unknown-key':g['pre_iou_candidates'][0]['unbounded_tensor']=[]
    elif fault=='bool-count':g['pre_iou_candidates_seen']=True
    elif fault=='partial':g.pop('pre_iou_scan_complete')
    elif fault=='invalid-index':g['pre_iou_invalid_row_index']=300
    observer=hooks._CorePathInstrumentation()
    observer.record_pose_probe(dict(source=1,raw=6,frame=431,pts_ns=4310,**g))
    assert observer.snapshot()['registered_ankle_priority']['records']==[]


def test_i156_geometry_keeps_existing_shared_record_and_byte_caps_with_honest_omissions():
    import json
    observer=hooks._CorePathInstrumentation()
    for raw in range(10):
        observer.record_pose_probe(dict(source=1,raw=raw,frame=431,pts_ns=4310,
            native_returned=False,result='context_no_matching_row',**_i156_geometry()))
    result=observer.snapshot()['registered_ankle_priority']
    assert result['record_limit']==8 and result['byte_limit']==8192
    assert len(result['records'])<=8 and result['omitted_records']==10-len(result['records'])
    assert len(json.dumps(result,allow_nan=False,sort_keys=True,separators=(',',':')).encode())<=8192
    # Explicit unavailable point/IoU from a malformed mapping remains JSON None.
    g=_i156_geometry();g['pre_iou_candidates'][0]['torso_unclipped'][0]=None
    g['pre_iou_candidates'][0]['iou']=None
    observer=hooks._CorePathInstrumentation()
    observer.record_pose_probe(dict(source=1,raw=6,frame=431,pts_ns=4310,**g))
    row=observer.snapshot()['registered_ankle_priority']['records'][0]
    assert row['pre_iou_candidates'][0]['torso_unclipped'][0] is None
    assert row['pre_iou_candidates'][0]['iou'] is None


def test_r146_pending_weak_floor_cannot_suppress_current_column():
    import copy
    from DS9.tests.test_registered_body_column import _r75_current_column,_r75_apply
    from noesis_core.world.resolver import UniversalWorldMeasurementResolver,WorldMeasurementResolverConfig
    state=hooks._WorldAnchorState(world_x=11.5604794722,world_z=5.7414193952,
        filtered_ts=4.4,last_good_world=(11.5604794722,0.,5.7414193952),last_good_ts=4.4,
        last_output_world_x=11.5604794722,last_output_world_z=5.7414193952,
        accepted_metric_contact_basis='pose_ankle_floor',posture='standing')
    for i in range(1):
        p,t,column,depth=_r75_current_column(134+i*6,4_466_666_666+i*200_000_000)
        p._min_unposed_detection_confidence=.55
        p._world_resolver=UniversalWorldMeasurementResolver(config=WorldMeasurementResolverConfig(registered_body_column=True))
        floor=column.model_copy(update=dict(candidate_id='floor_ray',method=None,kind='floor_ray',
            anchor='pose_ankle_floor',support_state='floor',confidence=.99,support_score=.99,
            source_reliability=1.,ray_incidence_sin=.3,
            position=Vector3(x=12.3412776064,y=0.,z=6.5337807356)))
        rows=WorldMeasurementSet(cohort=column.cohort,hypotheses=(floor,column))
        trial=copy.deepcopy(state)
        floor_result=p._world_resolver.resolve(rows.model_copy(update={'hypotheses':(floor,)}))
        assert floor_result.quality=='weak'
        hooks.update_human_cv_filter(trial,measurement=np.array([floor.position.x,0.,floor.position.z]),
            floor_y=0.,now_ts=t['media_pts_ns']/1e9,quality='weak',config=p._human_ground_cfg,
            contact_basis='pose_ankle_floor',selected_geometry_kind='floor_ray')
        assert not trial.measurement_accepted  # Actual temporal/physical admission, not just support.
        before=copy.deepcopy(state.__dict__)
        kwargs=dict(contact_basis={'floor_ray':'pose_ankle_floor','registered_body_column':'depth:registered_body_column'},
            track=t,state=state,detector_confidence=.9,now_ts=t['media_pts_ns']/1e9)
        chosen=p._body_column_fallback_set(rows,**kwargs)
        assert state.__dict__==before
        assert next(h for h in chosen.hypotheses if h.method=='registered_body_column').valid
        assert p._world_resolver.resolve(chosen).selected_candidate_id=='registered_body_column'
        _r75_apply(p,t,column,depth,state)
        assert not state.measurement_accepted and state.reacquire_count==1
    assert not state.reacquired  # Selection is not a one-sample measurement grant.


@pytest.mark.parametrize('mode',['near-weak','third-floor-vote','good-floor','cold-floor','no-column'])
def test_r146_actual_admitted_ordinary_retains_priority_without_state_mutation(mode):
    import copy
    from DS9.tests.test_registered_body_column import _r75_current_column
    from noesis_core.world.resolver import UniversalWorldMeasurementResolver,WorldMeasurementResolverConfig
    p,t,column,_=_r75_current_column(146,4_866_666_666)
    p._min_unposed_detection_confidence=.55
    p._world_resolver=UniversalWorldMeasurementResolver(config=WorldMeasurementResolverConfig(registered_body_column=True))
    state=hooks._WorldAnchorState(world_x=11.56,world_z=5.74,filtered_ts=4.8,
        last_good_world=(11.56,0.,5.74),last_good_ts=4.8,last_output_world_x=11.56,
        last_output_world_z=5.74,accepted_metric_contact_basis='pose_ankle_floor')
    far=mode=='third-floor-vote'
    point=Vector3(x=12.34 if far else 11.57,y=0.,z=6.53 if far else 5.75)
    floor=column.model_copy(update=dict(candidate_id='floor_ray',method=None,kind='floor_ray',
        anchor='pose_ankle_floor',support_state='floor',confidence=.99,support_score=.99,
        source_reliability=1.,ray_incidence_sin=.3,position=point,
        pcf=None if mode in ('good-floor','cold-floor') else column.pcf))
    if far:
        state.reacquire_candidate_x=12.34;state.reacquire_candidate_z=6.53
        state.reacquire_candidate_ts=4.8;state.reacquire_candidate_basis='pose_floor'
        state.reacquire_candidate_exact_basis='pose_ankle_floor';state.reacquire_count=2
    if mode=='cold-floor':state=hooks._WorldAnchorState()
    if mode=='no-column':column=column.model_copy(update={'valid':False,'rejection_reason':'no_current_registered_original'})
    before=copy.deepcopy(state.__dict__)
    rows=WorldMeasurementSet(cohort=column.cohort,hypotheses=(floor,column))
    chosen=p._body_column_fallback_set(rows,contact_basis={'floor_ray':'pose_ankle_floor'},
        track=t,state=state,detector_confidence=.9,now_ts=t['media_pts_ns']/1e9)
    assert p._world_resolver.resolve(chosen).selected_candidate_id=='floor_ray'
    assert state.__dict__==before


def _r131_ankle_original(frame=62):
    from noesis.metadata.object_depth import NativePosePatchSupport, PoseDepthSample
    from noesis.pipelines.source_acquisition import OriginalCohort
    p = _processor()
    p.pipeline.config['xc3d_image_regions'] = {'enabled': True}
    pts = frame*1_000_000_000//30
    c = _cohort().model_copy(update={'frame_id': frame, 'observed_at_us': pts//1000,
        'world_transform_sha256': 'a'*64})
    selected = _hypothesis(c, candidate_id='registered_depth', kind='registered_depth',
        x=10.6101953969).model_copy(update={'anchor':'person_mask_floor',
            'position':Vector3(x=10.6101953969,y=0.,z=5.5591718368)})
    patch = NativePosePatchSupport('pose_ankle_support',(400.,500.),
        ((390.,500.,410.,500.,5.,5.),),(1920,1080),frame,pts//1000)
    sample = PoseDepthSample('pose_ankle_support',(400.,500.),32,1.,32,12.,11.9,12.1,'good')
    depth = ObjectDepthResult(5,frame,17,0,(350.,300.,100.,220.),.9,'pose_capsule_native','ok','m',True,32,1.,
        anchor_source='pose_ankle_support',anchor_uv=(400.,500.),anchor_depth_m=12.,
        anchor_sample_count=32,anchor_valid_fraction=1.,pose_depth_samples=(sample,),native_pose_support=(patch,),
        ts_us=pts//1000,measurement_frame_id=frame,measurement_ts_us=pts//1000,measurement_cached=False,
        depth_tensor_frame_id=frame,depth_tensor_ts_us=pts//1000,depth_tensor_age_frames=0,depth_tensor_age_us=0)
    track=dict(source_id=5,tracker_id=17,frame_id=frame,observed_at_us=pts//1000,
        tracker_lifecycle_generation=3,media_pts_ns=pts,image_size=(1920,1080),
        _xc3d_native_pose_measured=True,
        _xc3d_original_frame_id=frame,
        _xc3d_current_original=OriginalCohort('run','graph','camera-0',5,0,frame,pts,0,'source_sr'))
    return p,track,selected,depth


def _r138_factory(monkeypatch, defect=None, *, enabled=True):
    from dataclasses import replace
    monkeypatch.setenv('NOESIS_XC3D_BODY_DIAGNOSTIC_TARGETS_S','5.333333')
    _,t,c,d=_r131_ankle_original(frame=74)
    p=_processor({'xc3d_image_regions':{'enabled':enabled}})
    if defect=='binding':t['_xc3d_native_pose_measured']=False
    if defect=='patch':d=replace(d,native_pose_support=())
    if defect=='patch-frame':d=replace(d,native_pose_support=(replace(d.native_pose_support[0],measurement_frame_id=73),))
    point=np.array([c.position.x,0.,c.position.z])
    resolved=point.copy()
    if defect=='xz':resolved[0]+=.001
    proof=p._registered_ankle_original(track=t,selected=c,depth_result=d,
        depth_observation=hooks._DepthObservationResult(point,.9,'ok',raw_depth_m=12.,
            registered_depth_m=7.2,registration_status='ok',registration_id='reg'),
        person_anchor=p._resolve_person_depth_anchor(d),measurement=resolved,
        now_ts=t['media_pts_ns']/1e9,calib=SimpleNamespace(image_size=(1280,720) if defect=='raster' else (1920,1080)),
        pose_is_current=defect!='pose')
    return p,t,proof


@pytest.mark.parametrize('defect,stage',[(None,'ready'),('binding','binding'),('pose','pose_not_current'),
    ('patch','private_patch_count'),('patch-frame','private_patch_original'),('raster','private_patch_raster'),
    ('xz','resolved_xz_mismatch')])
def test_r138_exact_factory_failure_stage_outside_fixed_targets(monkeypatch,defect,stage):
    hooks.reset_core_path_instrumentation()
    _,_,proof=_r138_factory(monkeypatch,defect)
    rows=hooks.get_core_path_instrumentation_snapshot()['registered_ankle_priority']['records']
    assert len(rows)==1 and rows[0]['factory_stage']==stage and rows[0]['frame']==74
    assert (proof is not None)==(defect is None)
    assert rows[0]['pts_ns']==74*1_000_000_000//30  # Not the configured 5.333s target.


def test_r138_diagnostic_default_disabled_and_observer_failure_parity(monkeypatch):
    hooks.reset_core_path_instrumentation()
    _r138_factory(monkeypatch,'pose',enabled=False)
    assert not hooks.get_core_path_instrumentation_snapshot()['registered_ankle_priority']['records']
    monkeypatch.delenv('NOESIS_XC3D_BODY_DIAGNOSTIC_TARGETS_S')
    p=_processor({'xc3d_image_regions':{'enabled':True}})
    assert p._ankle_priority_diagnostic_enabled is False
    def broken(*a,**kw):raise RuntimeError('optional diagnostic unavailable')
    monkeypatch.setattr(hooks._CORE_PATH_INSTRUMENTATION,'record_ankle_priority',broken)
    assert _r138_factory(monkeypatch)[2] is not None
    assert _r138_factory(monkeypatch,'pose')[2] is None


def test_r138_first_failure_latest_proof_and_final_byte_bound():
    import json
    observer=hooks._CorePathInstrumentation()
    for raw in range(40):
        observer.record_ankle_priority({'source':2,'raw':raw,'generation':2,'frame':74,
            'factory_stage':'pose_not_current','pose_current':False,
            'factory_geometry':'registered_depth','factory_relevant':True})
    observer.record_ankle_priority({'source':2,'raw':0,'generation':2,'frame':80,
        'factory_stage':'ready','proof_present':True,'proof_frame':80,
        'factory_geometry':'registered_depth','factory_relevant':True})
    view=observer.snapshot()['registered_ankle_priority']
    assert len(view['records'])==8 and view['omitted_records']==32
    first=view['records'][0]
    assert first['first_failure']=='pose_not_current' and first['first_failure_frame']==74
    assert first['factory_stage']=='ready' and first['proof_frame']==80
    first['factory_stage']='foreign'
    assert observer.snapshot()['registered_ankle_priority']['records'][0]['factory_stage']=='ready'
    # All accepted scalar fields may be long; FINAL encoded bytes still bound.
    for raw in range(8):
        observer.record_ankle_priority(dict(source=2,raw=raw,generation=2,
            **{k:'x'*80 for k in ('frame','pts_ns','factory_stage','pose_current','native_measured','original_frame',
            'depth_frame','patch_frame','patch_pts_us','patch_raster_match','depth_current','filter_frame',
            'local_accepted','cursor_frame','proof_frame','proof_sample_ts','proof_age_s','proof_present',
            'proof_expired','binding_present','geometry','proof_transition')},factory_relevant=True))
    encoded=json.dumps(observer.snapshot()['registered_ankle_priority'],sort_keys=True,allow_nan=False,separators=(',',':')).encode()
    assert len(encoded)<=8192 and json.loads(encoded)['omitted_records']>32


def test_r138_floor_support_skip_never_hides_real_registered_factory_failure():
    observer=hooks._CorePathInstrumentation()
    key={'source':2,'raw':3,'generation':2}
    for frame in (66,68):
        observer.record_ankle_priority(dict(key,frame=frame,factory_stage='not_invoked_quality',
            factory_geometry='floor_ray',factory_relevant=False))
    assert not observer.snapshot()['registered_ankle_priority']['records']
    observer.record_ankle_priority(dict(key,frame=74,factory_stage='private_patch_original',
        factory_geometry='registered_depth',factory_relevant=True))
    observer.record_ankle_priority(dict(key,frame=76,factory_stage='selected_anatomy_or_support',
        factory_geometry='floor_ray',factory_relevant=False))
    observer.record_ankle_priority(dict(key,filter_frame=76,geometry='floor_ray',proof_present=False))
    row=observer.snapshot()['registered_ankle_priority']['records'][0]
    assert row['first_failure']=='private_patch_original' and row['first_failure_frame']==74
    assert row['factory_stage']=='private_patch_original' and row['frame']==74
    assert row['factory_geometry']=='registered_depth' and row['factory_relevant'] is True
    assert row['filter_frame']==76 and row['geometry']=='floor_ray'
    clean=hooks._CorePathInstrumentation()
    clean.record_ankle_priority(dict(key,frame=74,factory_stage='ready',factory_geometry='registered_depth',factory_relevant=True))
    clean.record_ankle_priority(dict(key,frame=76,factory_stage='not_invoked_quality',factory_geometry='floor_ray',factory_relevant=False))
    assert 'first_failure' not in clean.snapshot()['registered_ankle_priority']['records'][0]


def test_r138_actual_filter_proof_harvest_and_expiry_scalars(monkeypatch):
    hooks.reset_core_path_instrumentation()
    p,t,proof=_r138_factory(monkeypatch)
    state=hooks._WorldAnchorState()
    for frame,quality,basis,geometry in ((74,'good','depth:pose_ankle_support','registered_depth'),
            (76,'weak','pose_ankle_floor','floor_ray'),(100,'weak','pose_ankle_floor','floor_ray')):
        t['frame_id']=frame
        p._update_track_world_state(t,state,measurement=np.array([*proof.position_xz[:1],0.,proof.position_xz[1]]),
            floor_y=0.,now_ts=proof.sample_ts if frame==74 else frame/30.,alpha=.3,beta=.15,quality=quality,
            contact_basis=basis,selected_geometry_kind=geometry,registered_ankle_binding=proof.binding[:-2],
            registered_ankle_original=proof if frame==74 else None,media_pts_ns=proof.media_pts_ns if frame==74 else None)
        row=hooks.get_core_path_instrumentation_snapshot()['registered_ankle_priority']['records'][0]
        assert row['filter_frame']==frame and row['cursor_frame']==74
        if frame==76:assert row['proof_transition']=='harvested' and row['proof_frame']==74
        if frame==100:assert row['proof_transition']=='cleared' and row['proof_expired'] is True
    assert row['first_clear_frame']==100 and row['first_clear_proof_frame']==74
    assert row['first_clear_quality']=='weak' and row['first_clear_contact']=='pose_ankle_floor'
    assert row['first_clear_geometry']=='floor_ray' and row['first_clear_binding_equal'] is True
    assert row['first_clear_age_s']>.75 and row['first_clear_native_measured'] is True
    # Another actual clear must not replace the first; ordinary latest state can advance.
    p._record_ankle_filter_state(t,state,proof,proof.sample_ts+.1,None,'floor_ray','weak','pose_ankle_floor')
    latest=hooks.get_core_path_instrumentation_snapshot()['registered_ankle_priority']['records'][0]
    assert latest['first_clear_binding_equal'] is True and latest['first_clear_age_s']==row['first_clear_age_s']


def test_r140_actual_filter_pins_native_binding_clear_before_later_updates(monkeypatch):
    hooks.reset_core_path_instrumentation()
    p,t,proof=_r138_factory(monkeypatch)
    state=hooks._WorldAnchorState()
    point=np.array([proof.position_xz[0],0.,proof.position_xz[1]])
    for frame in (74,76,78,80):
        t['frame_id']=frame;t['_xc3d_native_pose_measured']=frame!=78
        p._update_track_world_state(t,state,measurement=point,floor_y=0.,
            now_ts=proof.sample_ts if frame==74 else frame/30.,alpha=.3,beta=.15,
            quality='good' if frame==74 else 'weak',
            contact_basis='depth:pose_ankle_support' if frame==74 else 'pose_ankle_floor',
            selected_geometry_kind='registered_depth' if frame==74 else 'floor_ray',
            registered_ankle_original=proof if frame==74 else None,
            registered_ankle_binding=None if frame==78 else proof.binding[:-2],
            media_pts_ns=proof.media_pts_ns if frame==74 else None)
    row=hooks.get_core_path_instrumentation_snapshot()['registered_ankle_priority']['records'][0]
    assert row['first_clear_frame']==78 and row['first_clear_proof_frame']==74
    assert row['first_clear_binding_present'] is False and row['first_clear_binding_equal'] is False
    assert row['first_clear_native_measured'] is False and 0<row['first_clear_age_s']<.75
    assert row['filter_frame']==80 and row['binding_present'] is True


def test_r138_actual_runtime_stats_payload_forwards_only_opted_in_collected_probe(monkeypatch):
    import ast
    import copy
    import json
    import os
    root=ast.parse((REPO_ROOT/'DS9/noesis/ds9_runtime_core.py').read_text())
    stats=next(n for n in ast.walk(root) if isinstance(n,ast.FunctionDef) and n.name=='_stats')
    assignment=next(n for n in ast.walk(stats) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='stats_payload' for t in n.targets))
    def field(node,name):
        return next(v for k,v in zip(node.keys,node.values) if isinstance(k,ast.Constant) and k.value==name)
    # Execute the ACTUAL producer payload expression, without importing SDK/runtime.
    node=copy.deepcopy(field(field(assignment.value,'pipeline'),'zero_copy_core'))
    expression=compile(ast.fix_missing_locations(ast.Expression(node)),'actual_stats_zero_copy_core','eval')
    hooks.reset_core_path_instrumentation()
    _r138_factory(monkeypatch,'pose')
    snapshot=hooks.get_core_path_instrumentation_snapshot()
    namespace=dict(core_instr=snapshot,core_counters={'existing':1},core_stage_timings={'existing':{}},
        ws_boundary_metrics={'ws':1},rest_boundary_metrics={'rest':1},os=os,
        pipeline=SimpleNamespace(config={'xc3d_image_regions':{'enabled':True}}))
    payload=eval(expression,namespace)
    assert payload['registered_ankle_priority']==snapshot['registered_ankle_priority']
    assert payload['registered_ankle_priority']['records'][0]['factory_stage']=='pose_not_current'
    assert len(json.dumps(payload['registered_ankle_priority'],sort_keys=True,separators=(',',':')).encode())<=8192
    expected={'counters':{'existing':1},'stage_timings':{'existing':{}},
        'boundary_serialization_metrics':{'ws':{'ws':1},'rest':{'rest':1}}}
    assert {k:v for k,v in payload.items() if k!='registered_ankle_priority'}==expected
    monkeypatch.delenv('NOESIS_XC3D_BODY_DIAGNOSTIC_TARGETS_S')
    assert eval(expression,namespace)==expected
    monkeypatch.setenv('NOESIS_XC3D_BODY_DIAGNOSTIC_TARGETS_S','5.333333')
    namespace['pipeline'].config={}
    assert eval(expression,namespace)==expected
    namespace['pipeline'].config={'xc3d_image_regions':{'enabled':True}}
    namespace['core_instr']={'registered_ankle_priority':{'records':[],'omitted_records':0}}
    assert eval(expression,namespace)==expected
    namespace['core_instr']={'registered_ankle_priority':{'records':[],'omitted_records':1}}
    assert eval(expression,namespace)['registered_ankle_priority']['omitted_records']==1


@pytest.mark.parametrize('defect',['cached','old','future','birth','pts','missing-patch','not-current-pose','shadow','disabled','wrong-anatomy'])
def test_r131_registered_ankle_proof_requires_actual_current_original(defect):
    from dataclasses import replace
    p,t,c,d=_r131_ankle_original()
    if defect=='cached': d=replace(d,measurement_cached=True)
    elif defect=='old': d=replace(d,measurement_frame_id=61)
    elif defect=='future': d=replace(d,measurement_frame_id=63)
    elif defect=='birth': t['tracker_lifecycle_generation']=4
    elif defect=='pts': t['_xc3d_current_original']=replace(t['_xc3d_current_original'],pts_ns=1)
    elif defect=='missing-patch': d=replace(d,native_pose_support=())
    elif defect=='shadow': t['_xc3d_native_pose_measured']=False
    elif defect=='disabled': p.pipeline.config['xc3d_image_regions']['enabled']=False
    elif defect=='wrong-anatomy': d=replace(d,anchor_source='pose_torso_support')
    point=np.array([c.position.x,0.,c.position.z])
    proof=p._registered_ankle_original(track=t,selected=c,depth_result=d,
        depth_observation=hooks._DepthObservationResult(point,.9,'ok',raw_depth_m=12.,
            registered_depth_m=7.2,registration_status='ok',registration_id='reg'),
        person_anchor=p._resolve_person_depth_anchor(d),measurement=point,
        now_ts=t['media_pts_ns']/1e9,calib=SimpleNamespace(image_size=(1920,1080)),
        pose_is_current=defect!='not-current-pose')
    assert proof is None


@pytest.mark.parametrize('generic_pixel_motion',[False,True])
def test_r131_current_registered_ankle_cannot_be_reanchored_by_weak_floor_cadence(generic_pixel_motion):
    p,track,selected,depth = _r131_ankle_original()
    point = np.array([selected.position.x,0.,selected.position.z])
    proof = p._registered_ankle_original(track=track,selected=selected,depth_result=depth,
        depth_observation=hooks._DepthObservationResult(point,.9,'ok',raw_depth_m=12.,
            registered_depth_m=7.2,registration_status='ok',registration_id='reg'),
        person_anchor=p._resolve_person_depth_anchor(depth),measurement=point,
        now_ts=track['media_pts_ns']/1e9,calib=SimpleNamespace(image_size=(1920,1080)),pose_is_current=True)
    assert type(proof) is hooks.RegisteredAnkleOriginal
    state = hooks._WorldAnchorState()
    p._update_track_world_state({}, state,
        measurement=point, floor_y=0.,
        now_ts=proof.sample_ts, alpha=.3, beta=.15, quality='good',
        registered_ankle_original=proof,registered_ankle_binding=proof.binding[:-2],media_pts_ns=proof.media_pts_ns,
        contact_basis='depth:pose_ankle_support', verified_reacquire_support=True)
    assert state.measurement_accepted
    for frame in (63, 64, 65):
        p._update_track_world_state({}, state,
            measurement=np.array([11.3454876512, 0., 6.5970391003]), floor_y=0.,
            now_ts=frame/30., alpha=.3, beta=.15, quality='weak',
            registered_ankle_binding=proof.binding[:-2],
            image_motion_supported=generic_pixel_motion,
            contact_basis='pose_ankle_floor', verified_reacquire_support=True)
        assert not state.measurement_accepted and not state.reacquired
        assert state.accepted_ankle_original is proof
        assert state.accepted_ankle_original.sample_ts == proof.sample_ts


@pytest.mark.parametrize('unavailable', [False, True])
@pytest.mark.parametrize('generic_pixel_motion', [False, True])
def test_r136_floor_ray_geometry_keeps_true_depth_uv_basis_but_cannot_reanchor(unavailable, generic_pixel_motion):
    p, track, selected, depth = _r131_ankle_original(frame=64)
    point = np.array([selected.position.x, 0., selected.position.z])
    proof = p._registered_ankle_original(track=track, selected=selected, depth_result=depth,
        depth_observation=hooks._DepthObservationResult(point, .9, 'ok', raw_depth_m=12.,
            registered_depth_m=7.2, registration_status='ok', registration_id='reg'),
        person_anchor=p._resolve_person_depth_anchor(depth), measurement=point,
        now_ts=track['media_pts_ns']/1e9, calib=SimpleNamespace(image_size=(1920,1080)), pose_is_current=True)
    assert type(proof) is hooks.RegisteredAnkleOriginal
    state = hooks._WorldAnchorState()
    p._update_track_world_state({}, state, measurement=point, floor_y=0.,
        now_ts=proof.sample_ts, alpha=.3, beta=.15, quality='good',
        registered_ankle_original=proof, registered_ankle_binding=proof.binding[:-2],
        media_pts_ns=proof.media_pts_ns, contact_basis='depth:pose_ankle_support',
        selected_geometry_kind=selected.kind, verified_reacquire_support=True)
    if unavailable:
        hooks.mark_world_measurement_unavailable(state, reason='depth_measurement_not_current',
            now_ts=65/30., config=p._human_ground_cfg)
    floor = selected.model_copy(update={'candidate_id':'floor_ray', 'kind':'floor_ray',
        'position':Vector3(x=11.3454876512, y=0., z=6.5970391003)})
    original_ts = proof.sample_ts
    for frame in (66, 68, 70):
        row = {'world_resolver_selected_id':floor.candidate_id,
            'world_resolver_contact_basis':'depth:pose_ankle_support'}
        p._update_track_world_state(row, state,
            measurement=np.array([floor.position.x, 0., floor.position.z]), floor_y=0.,
            now_ts=frame/30., alpha=.3, beta=.15, quality='weak',
            registered_ankle_binding=proof.binding[:-2], selected_geometry_kind=floor.kind,
            contact_basis=row['world_resolver_contact_basis'],
            image_motion_supported=generic_pixel_motion, verified_reacquire_support=True)
        assert row['world_resolver_contact_basis'] == 'depth:pose_ankle_support'
        assert not state.measurement_accepted and not state.reacquired
        assert state.measurement_rejection_reason == 'weak_floor_reanchor_requires_independent_support'
        assert state.accepted_ankle_original is proof and proof.sample_ts == original_ts


def _upright_projection_calibration() -> SimpleNamespace:
    intrinsics = np.array(
        [
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera_world = np.array([0.0, 2.2, -6.0], dtype=np.float64)
    forward = np.array([0.0, -0.25, 1.0], dtype=np.float64)
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(right, forward)
    down /= np.linalg.norm(down)
    rotation_cw = np.column_stack([right, down, forward]).T
    extrinsics = np.eye(4, dtype=np.float64)
    extrinsics[:3, :3] = rotation_cw
    extrinsics[:3, 3] = -rotation_cw @ camera_world
    return SimpleNamespace(
        intrinsics=intrinsics,
        extrinsics_col_major=list(extrinsics.flatten(order="F")),
        floor_y=0.0,
        image_size=(1280, 720),
        unit_scale=1.0,
    )


def _occluded_upright_pose(
    calibration: SimpleNamespace,
) -> tuple[np.ndarray, list[float]]:
    def project(world_xyz: tuple[float, float, float]) -> tuple[float, float]:
        uv = project_world_to_image(
            world_xyz,
            calibration.intrinsics,
            calibration.extrinsics_col_major,
            calibration.image_size,
            unit_scale=1.0,
        )
        assert uv is not None
        return float(uv[0]), float(uv[1])

    pose = np.zeros((17, 3), dtype=np.float64)
    for name, world_xyz in {
        "nose": (0.0, 1.692, 6.0),
        "left_shoulder": (-0.18, 1.476, 6.0),
        "right_shoulder": (0.18, 1.476, 6.0),
        "left_hip": (-0.10, 0.99, 6.0),
        "right_hip": (0.10, 0.99, 6.0),
    }.items():
        u, v = project(world_xyz)
        pose[hooks._POSE_KPT_INDEX[name]] = [u, v, 0.98]

    foot_u, foot_v = project((0.0, 0.0, 6.0))
    _head_u, head_v = project((0.0, 1.8, 6.0))
    # The detector terminates on furniture, well above the actual floor.
    bbox = [foot_u - 42.0, head_v, 84.0, 0.68 * (foot_v - head_v)]
    return pose, [float(value) for value in bbox]


def _cohort(*, pcf_revision: str | None = None) -> WorldMeasurementCohort:
    return WorldMeasurementCohort(
        track_key="5:17:3",
        tracker_lifecycle_generation=3,
        camera_id="camera-0",
        source_id=5,
        tracker_id=17,
        frame_id=42,
        observed_at_us=123456,
        world_revision="world-revision",
        calibration_revision="calibration-revision",
        pcf_revision=pcf_revision,
    )


def _r123_contact_candidates(source: str, *, floor=True):
    """Literal R58 K29 ray/ROI; synthetic scalar covariance, no SDK receipt."""
    processor = _processor()
    processor.scene_priors = None
    processor._floor_candidate_covariance = lambda *_a, **_k: Matrix3(
        values=(.007293833248229415, 0., 0., 0., .03765852363999861, 0.,
                0., 0., .023378484592169384))
    anchor = processor._resolve_bbox_floor_anchor((657., 406.5, 85.5, 133.))
    if source != 'bbox_bottom':
        anchor = hooks._PoseAnchorCandidate(u=anchor.u, v=anchor.v, source=source,
            contact_basis='pose:ankle_pair', quality='good', height_lock_eligible=True)
    candidates = []
    processor._append_universal_world_candidates(candidates, sensor_id=1,
        track={'tracker_id':0,'world_floor_incidence_sin':.308536909}, camera_id='kitchen',
        calib=SimpleNamespace(), floor_candidate=np.array([17.0351691164,0.,6.4587978178]) if floor else None,
        floor_ray_admitted=True, anchor_candidate=anchor, pose_uv=(anchor.u,anchor.v),
        contact_basis=anchor.contact_basis, posture='unknown', occlusion_fraction=0.,
        image_motion_supported=True, depth_obs=None, depth_weight=0.,
        depth_observation=hooks._DepthObservationResult(None,0.,'depth_anchor_unavailable'),
        depth_result=None, person_anchor=None, reject_current_geometry=False, flip_u=False, flip_v=False)
    return processor, anchor, candidates


def test_r123_literal_counter_bbox_is_diagnostic_not_verified_floor():
    processor, anchor, candidates = _r123_contact_candidates('bbox_bottom')
    assert (anchor.u,anchor.v)==(699.75,539.5)  # Preserve image diagnostic ROI.
    assert not processor._anchor_is_verified_ground_contact(anchor)
    assert candidates == []
    assert processor._accepted_image_foot_for_recording(
        {'image_base':[701.,540.], '_world_current_image_foot_calib':[699.75,539.5]},
        world_source='person_anchor_floor_only', anchor=anchor)==(701.,540.)


@pytest.mark.parametrize('source',['pose_ankle_floor','pose_single_ankle_floor',
    'pose_ankle_support','pose_leg_floor'])
def test_r123_qualified_contact_paths_remain_current_floor_candidates(source):
    processor,anchor,candidates=_r123_contact_candidates(source)
    assert processor._anchor_is_verified_ground_contact(anchor)
    assert len(candidates)==1 and candidates[0]['kind']=='floor_ray'
    assert candidates[0]['anchor']==source


@pytest.mark.parametrize('basis,expected',[('depth:lower_body_band',True),
    ('depth:pose_ankle_support',True),('depth:torso_core',False),('depth:pose_torso_support',False)])
def test_r123_mask_contact_still_requires_actual_lower_body_band(basis,expected):
    anchor=hooks._PoseAnchorCandidate(100.,200.,'person_mask_floor',contact_basis=basis)
    assert hooks._AnalyticsTelemetryProcessor._anchor_is_verified_ground_contact(anchor) is expected


def test_r123_bbox_reference_does_not_suppress_exact_registered_body_column():
    from DS9.tests.test_registered_body_column import _r75_current_column
    from noesis_core.world.resolver import UniversalWorldMeasurementResolver,WorldMeasurementResolverConfig
    processor,track,column,depth=_r75_current_column()
    processor.scene_priors=None;processor.depth_registration=SimpleNamespace()
    processor._depth_candidate_covariance=lambda *_a,**_k:column.covariance
    processor._world_resolver_diag_max_candidates=4;processor._min_unposed_detection_confidence=.55
    processor._world_resolver=UniversalWorldMeasurementResolver(config=WorldMeasurementResolverConfig(registered_body_column=True))
    bbox=processor._resolve_bbox_floor_anchor(depth.bbox)
    torso=processor._resolve_person_depth_anchor(depth)
    candidates=[]
    processor._append_universal_world_candidates(candidates,sensor_id=2,track=track,camera_id='family-room',
        calib=SimpleNamespace(floor_y=0.),floor_candidate=np.array([20.,0.,20.]),
        floor_ray_admitted=True,anchor_candidate=bbox,pose_uv=(bbox.u,bbox.v),contact_basis=bbox.contact_basis,
        posture='unknown',observed_posture='unknown',occlusion_fraction=0.,image_motion_supported=False,
        depth_obs=np.array([column.position.x,.414,column.position.z]),depth_weight=.9,
        depth_observation=hooks._DepthObservationResult(np.array([column.position.x,.414,column.position.z]),
            .9,'ok',registered_depth_m=7.2,registration_status='ok',registration_id='reg'),
        depth_result=depth,person_anchor=torso,reject_current_geometry=False,flip_u=False,flip_v=False)
    assert all(c['kind']!='floor_ray' for c in candidates)
    typed=processor._build_world_measurement_set(cohort=column.cohort,candidates=candidates)
    typed=processor._body_column_fallback_set(typed,contact_basis={c['candidate_id']:c['contact_basis']for c in candidates},
        track=track,state=None,detector_confidence=.9)
    result=processor._world_resolver.resolve(typed)
    assert result.selected_candidate_id=='registered_body_column'
    assert result.support_state=='unknown' and result.posture=='unknown'


@pytest.mark.parametrize('resolver_enabled',[False,True])
def test_r123_actual_default_caller_cannot_seed_floor_from_bbox(resolver_enabled):
    processor,calibration=_r123_actual_contact_processor(resolver_enabled)
    for frame in range(1,5):
        track=_r123_actual_contact_row(processor,calibration,frame=frame,contact=False)
        assert track.get('world_valid') is not True
        assert track.get('world_measurement_accepted') is not True
        assert track.get('world_estimator_evaluated') is True
        assert track['bbox']==[657.,406.5,85.5,133.]


def _r123_actual_contact_processor(resolver_enabled=True, *, source=0, camera='cam0', binding=None):
    from noesis.calibration.world_fusion_policy import CameraWorldFusionProfile,WorldFusionPolicy
    calibration=_upright_projection_calibration()
    extrinsics=np.asarray(calibration.extrinsics_col_major).reshape((4,4),order='F').copy()
    extrinsics[:3,3]=-extrinsics[:3,:3]@np.array([0.,2.2,-3.])
    calibration.extrinsics_col_major=list(extrinsics.flatten(order='F'))
    calibration.camera_id=camera
    calibration.camera_calibration_sha256=binding.source_camera_calibration_sha256 if binding else 'b'*64
    calibration.world_frame_id='backend_world_m'
    calibration.world_frame_revision=binding.target_frame.revision if binding else 'r123'
    calibration.frame_transform_sha256=binding.target_from_source_sha256 if binding else 'a'*64
    provider=SimpleNamespace(snapshot=lambda *_:calibration,world_snapshot=lambda *_:calibration)
    policy=WorldFusionPolicy(policy_id='r123-synthetic',evidence={},cameras={camera:CameraWorldFusionProfile(
        camera_id=camera,calibration_fingerprint_sha256='a'*64,registration_id='reg-test',
        floor_weight_scale=1.,depth_weight_scale=1.,floor_only_allowed=True,floor_ray_max_range_m=20.)})
    processor=hooks._AnalyticsTelemetryProcessor(pipeline=SimpleNamespace(config={
        'models':{'pose':{'kpt_threshold':.35}},'canonical_world':{'measurement_resolver':{
            'enabled':resolver_enabled,'max_range_m':22.,'max_disagreement_m':1.25,
            'max_candidates':4}}}),tracking_pub=SimpleNamespace(),camera_labels={source:camera},
        sensor_id_map={source:source},bev_calibration=provider,world_fusion_policy=policy)
    return processor,calibration


def _r123_actual_contact_row(processor,calibration,*,frame,contact,source=0,camera='cam0',raw=0):
    pose=None;bbox=[657.,406.5,85.5,133.]
    if contact:
        pose,_=_occluded_upright_pose(calibration)
        for name,world in [('left_knee',(-.1,.5,6.)),('right_knee',(.1,.5,6.)),
                           ('left_ankle',(-.1,0.,6.)),('right_ankle',(.1,0.,6.))]:
            u,v=project_world_to_image(world,calibration.intrinsics,calibration.extrinsics_col_major,
                calibration.image_size,unit_scale=1.)
            pose[hooks._POSE_KPT_INDEX[name]]=[u,v,.99]
        visible=pose[pose[:,2]>.9]
        bbox=[float(visible[:,0].min()-15),float(visible[:,1].min()-10),
            float(np.ptp(visible[:,0])+30),float(np.ptp(visible[:,1])+20)]
    track=dict(source_id=source,camera_id=camera,tracker_id=raw,tracker_lifecycle_generation=1,
        lifecycle_started_frame=1,frame_id=frame,bbox=bbox,confidence=.95,tracker_confidence=.93,
        image_size=[1280,720],media_pts_ns=frame*100_000_000,observed_at_us=10_000_000+frame*100_000)
    processor._augment_track_with_world(source,camera,track,pose_kpts_abs=pose,
        pose_is_current=contact,world_now_ts=frame*.1)
    return track


def test_r136_actual_resolver_caller_forwards_selected_geometry_not_anatomy(monkeypatch):
    processor, calibration = _r123_actual_contact_processor()
    calls = []
    actual = processor._update_track_world_state
    def record(track, state, **kwargs):
        calls.append((kwargs.get('selected_geometry_kind'), kwargs.get('contact_basis')))
        return actual(track, state, **kwargs)
    monkeypatch.setattr(processor, '_update_track_world_state', record)
    rows = [_r123_actual_contact_row(processor, calibration, frame=f, contact=True) for f in range(1, 5)]
    assert rows[-1]['world_resolver_selected_id'] == 'floor_ray'
    assert rows[-1]['world_resolver_contact_basis'] == 'pose:ankle_pair'
    assert calls and all(kind == 'floor_ray' and basis == 'pose_ankle_floor' for kind, basis in calls)


@pytest.mark.parametrize('resolver_enabled',[False,True])
def test_r123_actual_qualified_ankles_upgrade_bbox_first_without_first_view_lock(resolver_enabled):
    processor,calibration=_r123_actual_contact_processor(resolver_enabled)
    for frame in range(1,4):
        assert _r123_actual_contact_row(processor,calibration,frame=frame,contact=False).get('world_valid') is not True
    rows=[_r123_actual_contact_row(processor,calibration,frame=f,contact=True) for f in range(4,8)]
    assert any(r.get('world_measurement_accepted') is True for r in rows)
    assert rows[-1]['world_valid'] is True and rows[-1]['world_source']=='pose_floor_only'
    if resolver_enabled:
        assert rows[-1]['world_resolver_contact_basis']=='pose:ankle_pair'


def _hypothesis(
    cohort: WorldMeasurementCohort,
    *,
    candidate_id: str,
    kind: str = "floor_ray",
    x: float = 1.0,
) -> WorldMeasurementHypothesis:
    return WorldMeasurementHypothesis(
        candidate_id=candidate_id,
        cohort=cohort,
        kind=kind,  # type: ignore[arg-type]
        position=Vector3(x=x, y=0.0, z=2.0),
        covariance=Matrix3(
            values=(
                0.04,
                0.0,
                0.0,
                0.0,
                0.04,
                0.0,
                0.0,
                0.0,
                0.04,
            )
        ),
        anchor="pose_ankle_support",
        support_state="floor",
        posture="standing",
        confidence=0.9,
        support_score=0.9,
        ray_incidence_sin=0.8,
        pixel_uncertainty_px=2.0,
    )


def test_hook_emits_snake_case_full_covariance_resolver_diagnostics() -> None:
    processor = _processor()
    processor.set_world_resolver_diagnostics_enabled(True)
    hooks.reset_core_path_instrumentation()
    cohort = _cohort()
    measurement_set = WorldMeasurementSet(
        cohort=cohort,
        hypotheses=(
            _hypothesis(cohort, candidate_id="floor_ray"),
            _hypothesis(
                cohort,
                candidate_id="registered_depth",
                kind="registered_depth",
                x=1.02,
            ),
        ),
    )
    track: dict[str, object] = {}

    point, result = processor._apply_resolved_world_measurement(track, measurement_set)

    assert point is not None
    assert result is not None
    diagnostic = track["world_resolver"]
    assert isinstance(diagnostic, dict)
    assert diagnostic["frame_id"] == 42
    assert diagnostic["world_frame_revision"] == "world-revision"
    assert diagnostic["selected_id"]
    assert diagnostic["contributor_ids"]
    assert diagnostic["resolved"]["position"] == {"x": point[0], "y": point[1], "z": point[2]}
    assert len(diagnostic["resolved"]["covariance"]["values"]) == 9
    assert len(diagnostic["candidates"]) == 2
    assert diagnostic["candidates"][0]["id"] == "floor_ray"
    assert len(diagnostic["candidates"][0]["covariance"]["values"]) == 9
    assert "world" not in diagnostic["candidates"][0]
    assert "covarianceXZ" not in diagnostic["candidates"][0]
    # Resolver covariance is private until PersonGroundState accepts the
    # current measurement and the emitted point is known.
    assert "world_covariance" not in track
    assert track["world_quantity"] == "ground_footprint"
    assert track["world_support_state"] == "floor"
    timing = hooks.get_core_path_instrumentation_snapshot()["stage_timings"][
        "world_resolver.resolve"
    ]
    assert timing["count"] == 1
    assert timing["total_ns"] >= 0


def test_request_gated_legacy_comparison_reconstructs_retired_room_policy_only() -> None:
    processor = _processor()
    processor.legacy_world_fusion_policy = SimpleNamespace(
        policy_id="historical-policy-id",
        profile=lambda _camera_id: SimpleNamespace(
            floor_weight_scale=1.0,
            depth_weight_scale=1.0,
            floor_only_allowed=False,
        ),
    )
    cohort = _cohort()
    measurement_set = WorldMeasurementSet(
        cohort=cohort,
        hypotheses=(
            _hypothesis(cohort, candidate_id="floor_ray", x=1.0),
            _hypothesis(
                cohort,
                candidate_id="registered_depth",
                kind="registered_depth",
                x=1.8,
            ),
        ),
    )

    compact_track: dict[str, object] = {}
    compact_point, _compact_result = processor._apply_resolved_world_measurement(
        compact_track,
        measurement_set,
    )
    assert compact_point is not None
    assert "world_resolver" not in compact_track

    processor.set_world_resolver_diagnostics_enabled(True)
    detailed_track: dict[str, object] = {}
    detailed_point, _detailed_result = processor._apply_resolved_world_measurement(
        detailed_track,
        measurement_set,
    )
    assert detailed_point is not None
    assert detailed_point.tolist() == pytest.approx(compact_point.tolist())
    diagnostics = detailed_track["world_resolver"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["legacy"] == {
        "id": "legacy_room_policy",
        "kind": "legacy_policy",
        "position": {"x": 1.8, "y": 0.0, "z": 2.0},
        "covariance": {
            "values": [
                0.04,
                0.0,
                0.0,
                0.0,
                0.04,
                0.0,
                0.0,
                0.0,
                0.04,
            ]
        },
        "source": "historical_depth_labeled_fused",
        "policy_id": "historical-policy-id",
    }


def test_ndarray_candidates_survive_hook_contract_boundary() -> None:
    processor = _processor()
    cohort = _cohort()
    candidate_set = processor._build_world_measurement_set(
        cohort=cohort,
        candidates=[
            {
                "candidate_id": "floor_ray",
                "kind": "floor_ray",
                "position": np.asarray([1.0, 0.0, 2.0], dtype=np.float64),
                "covariance": Matrix3(
                    values=(
                        0.04,
                        0.0,
                        0.0,
                        0.0,
                        0.04,
                        0.0,
                        0.0,
                        0.0,
                        0.04,
                    )
                ),
                "anchor": "pose_ankle_support",
                "support_state": "floor",
                "posture": "standing",
            }
        ],
    )

    assert candidate_set is not None
    assert len(candidate_set.hypotheses) == 1
    point, result = processor._apply_resolved_world_measurement({}, candidate_set)
    assert point is not None
    assert result is not None


def test_hook_cohort_uses_track_source_and_observed_time() -> None:
    processor = _processor()
    track = {
        "tracker_id": 17,
        "frame_id": 42,
        "source_id": 5,
        "observed_at_us": 123456,
        "tracker_lifecycle_generation": 3,
    }
    calib = SimpleNamespace(camera_calibration_sha256="calibration-revision")

    cohort = processor._world_measurement_cohort(
        99,
        "camera-0",
        track,
        calib=calib,
        world_frame_revision="world-revision",
        world_transform_sha256="a" * 64,
    )

    assert cohort is not None
    assert cohort.source_id == 5
    assert cohort.observed_at_us == 123456
    assert cohort.calibration_revision == "calibration-revision"
    assert cohort.world_revision == "world-revision"
    assert cohort.track_key == "5:17:3"
    assert cohort.tracker_lifecycle_generation == 3
    assert cohort.world_transform_sha256 == "a" * 64


def test_hook_cohort_rejects_missing_exact_lifecycle_or_transform_identity() -> None:
    processor = _processor()
    calib = SimpleNamespace(camera_calibration_sha256="calibration-revision")
    base = {
        "tracker_id": 17,
        "frame_id": 42,
        "source_id": 5,
        "observed_at_us": 123456,
        "tracker_lifecycle_generation": 3,
    }
    assert (
        processor._world_measurement_cohort(
            99,
            "camera-0",
            {**base, "tracker_lifecycle_generation": None},
            calib=calib,
            world_frame_revision="world-revision",
            world_transform_sha256="a" * 64,
        )
        is None
    )
    assert (
        processor._world_measurement_cohort(
            99,
            "camera-0",
            base,
            calib=calib,
            world_frame_revision="world-revision",
            world_transform_sha256=None,
        )
        is None
    )


def test_invalid_enabled_resolver_config_fails_closed() -> None:
    with pytest.raises(ValueError, match="measurement_resolver"):
        _processor(
            {
                "canonical_world": {
                    "measurement_resolver": {
                        "enabled": True,
                        "max_candidates": "not-an-int",
                    }
                }
            }
        )


@pytest.mark.parametrize(
    "patch",
    (
        {"enabled": "true"},
        {"legacy_comparator": False},
        {"max_range_m": 0.5},
        {"max_disagreement_m": float("nan")},
        {"max_candidates": 5},
        {"max_candidates": True},
    ),
)
def test_resolver_config_rejects_coercion_and_out_of_range_values(
    patch: dict[str, object],
) -> None:
    config: dict[str, object] = {
        "enabled": True,
        "max_range_m": 22.0,
        "max_disagreement_m": 1.25,
        "max_candidates": 4,
    }
    config.update(patch)
    with pytest.raises(ValueError, match="measurement_resolver"):
        _processor({"canonical_world": {"measurement_resolver": config}})


@pytest.mark.parametrize(
    "person_admission",
    (
        {},
        {"min_unposed_detection_confidence": "0.5"},
        {"min_unposed_detection_confidence": True},
        {"min_unposed_detection_confidence": -0.1},
        {"min_unposed_detection_confidence": 1.1},
        {"unexpected": 0.5},
    ),
)
def test_person_admission_config_rejects_missing_coercion_and_out_of_range_values(
    person_admission: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="person_admission"):
        _processor({"canonical_world": {"person_admission": person_admission}})


def test_person_admission_config_uses_explicit_default_when_omitted() -> None:
    assert _processor()._min_unposed_detection_confidence == pytest.approx(0.50)


def test_filtered_covariance_includes_resolver_to_emitted_displacement() -> None:
    processor = _processor()
    cohort = _cohort()
    measurement_set = WorldMeasurementSet(
        cohort=cohort,
        hypotheses=(_hypothesis(cohort, candidate_id="floor_ray"),),
    )
    track: dict[str, object] = {}
    point, result = processor._apply_resolved_world_measurement(track, measurement_set)
    assert point is not None and result is not None

    processor._publish_filtered_world_covariance(
        track,
        resolved=result,
        resolved_point=point,
        emitted_point=np.asarray([point[0] + 0.5, point[1], point[2]], dtype=np.float64),
    )

    covariance = track["world_covariance"]
    assert isinstance(covariance, list)
    assert len(covariance) == 9
    assert covariance[0] >= 0.29


def test_resolver_preserves_selected_anatomical_contact_basis() -> None:
    processor = _processor()
    processor.set_world_resolver_diagnostics_enabled(True)
    cohort = _cohort()
    measurement_set = WorldMeasurementSet(
        cohort=cohort,
        hypotheses=(_hypothesis(cohort, candidate_id="floor_ray"),),
    )
    track: dict[str, object] = {}

    point, result = processor._apply_resolved_world_measurement(
        track,
        measurement_set,
        contact_basis_by_candidate={"floor_ray": "pose:left_ankle"},
    )

    assert point is not None
    assert result is not None and result.status == "measured"
    assert track["world_resolver_contact_basis"] == "pose:left_ankle"
    candidate = track["world_resolver"]["candidates"][0]
    assert candidate["anchor"] == "pose_ankle_support"
    assert candidate["contact_basis"] == "pose:left_ankle"


def test_person_ground_consensus_collapses_only_observed_ankle_sides() -> None:
    processor = _processor()
    state = hooks._WorldAnchorState(
        ts=0.0,
        world_x=0.0,
        world_z=0.0,
        filtered_ts=0.0,
        last_good_world=(0.0, 0.0, 0.0),
        last_good_ts=0.0,
    )

    for index, (basis, expected_count) in enumerate(
        (
            ("pose:left_ankle", 1),
            ("pose:right_ankle", 2),
            ("depth:pose_torso_support", 1),
            ("depth:lower_body_band", 1),
        ),
        start=1,
    ):
        processor._update_track_world_state(
            {},
            state,
            measurement=np.asarray([5.0 + index * 0.1, 0.0, 0.0]),
            floor_y=0.0,
            now_ts=index * 0.1,
            alpha=0.45,
            beta=0.15,
            quality="good",
            contact_basis=processor._person_ground_consensus_contact_basis(basis),
            image_motion_supported=True,
        )
        assert state.measurement_accepted is False
        assert state.reacquire_count == expected_count

    assert processor._person_ground_consensus_contact_basis(
        "pose:ankle_pair"
    ) == "pose_ankle_floor"
    assert processor._person_ground_consensus_contact_basis(
        "pose:left_ankle"
    ) == "pose_single_ankle_floor"
    assert processor._person_ground_consensus_contact_basis(
        "pose:right_ankle"
    ) == "pose_single_ankle_floor"
    assert processor._person_ground_consensus_contact_basis(
        "depth:pose_torso_support"
    ) == "depth:pose_torso_support"


def test_resolver_apply_clears_compact_fields_on_rejection() -> None:
    processor = _processor()
    cohort = _cohort()
    measurement_set = WorldMeasurementSet(cohort=cohort, hypotheses=())
    track: dict[str, object] = {
        "world_quantity": "ground_footprint",
        "world_posture": "standing",
        "world_support_state": "floor",
        "world_resolver_confidence": 0.99,
        "world_resolver_selected_id": "stale",
        "world_resolver_fused": True,
        "world_resolver_disagreement_m": 2.0,
        "world_resolver_contact_basis": "stale",
        "world_resolver_source_continuity_match": True,
        "world_resolver": {"candidates": [{"stale": True}]},
        "world_covariance": [1.0] * 9,
    }

    point, result = processor._apply_resolved_world_measurement(track, measurement_set)

    assert point is None
    assert result is not None and result.status == "rejected"
    assert "world_resolver" not in track
    assert "world_covariance" not in track
    assert "world_resolver_selected_id" not in track
    assert "world_resolver_fused" not in track
    assert "world_resolver_disagreement_m" not in track
    assert "world_resolver_contact_basis" not in track
    assert track["world_resolver_confidence"] == 0.0


def test_torso_depth_anchor_never_enters_floor_ray_hypothesis() -> None:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    processor.scene_priors = None
    processor._floor_candidate_covariance = lambda *_args, **_kwargs: Matrix3(  # type: ignore[method-assign]
        values=(
            0.04,
            0.0,
            0.0,
            0.0,
            0.04,
            0.0,
            0.0,
            0.0,
            0.04,
        )
    )
    torso_anchor = hooks._PoseAnchorCandidate(
        u=100.0,
        v=200.0,
        source="person_mask_floor",
        contact_basis="depth:torso_core",
        quality="estimated",
    )
    candidates: list[dict[str, object]] = []
    processor._append_universal_world_candidates(
        candidates,
        sensor_id=0,
        track={"tracker_id": 7, "world_floor_incidence_sin": 0.8},
        camera_id="cam0",
        calib=SimpleNamespace(),
        floor_candidate=np.asarray([1.0, 0.0, 2.0], dtype=np.float64),
        floor_ray_admitted=True,
        anchor_candidate=torso_anchor,
        pose_uv=(100.0, 200.0),
        contact_basis="depth:torso_core",
        posture="standing",
        occlusion_fraction=0.0,
        image_motion_supported=True,
        depth_obs=None,
        depth_weight=0.0,
        depth_observation=hooks._DepthObservationResult(None, 0.0, "none"),
        depth_result=None,
        person_anchor=torso_anchor,
        reject_current_geometry=False,
        flip_u=False,
        flip_v=False,
    )

    assert candidates == []
    assert processor._anchor_is_verified_ground_contact(torso_anchor) is False


def test_torso_depth_remains_registered_depth_with_unknown_support() -> None:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    processor.scene_priors = None
    processor.depth_registration = None
    processor._depth_candidate_covariance = lambda *_args, **_kwargs: Matrix3(  # type: ignore[method-assign]
        values=(
            0.04,
            0.0,
            0.0,
            0.0,
            0.04,
            0.0,
            0.0,
            0.0,
            0.04,
        )
    )
    torso_anchor = hooks._PoseAnchorCandidate(
        u=100.0,
        v=200.0,
        source="person_mask_floor",
        contact_basis="depth:torso_core",
        quality="estimated",
    )
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=7,
        class_id=0,
        bbox=(0.0, 0.0, 10.0, 20.0),
        score=0.9,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=64,
        valid_fraction=0.9,
        anchor_source="torso_core",
        anchor_depth_m=5.0,
        anchor_sample_count=64,
        anchor_valid_fraction=0.9,
    )
    candidates: list[dict[str, object]] = []
    processor._append_universal_world_candidates(
        candidates,
        sensor_id=0,
        track={"tracker_id": 7, "world_floor_incidence_sin": 0.8},
        camera_id="cam0",
        calib=SimpleNamespace(floor_y=0.0),
        floor_candidate=None,
        floor_ray_admitted=False,
        anchor_candidate=torso_anchor,
        pose_uv=None,
        contact_basis="depth:torso_core",
        posture="standing",
        occlusion_fraction=0.75,
        image_motion_supported=False,
        depth_obs=np.asarray([1.0, 1.0, 2.0], dtype=np.float64),
        depth_weight=0.8,
        depth_observation=hooks._DepthObservationResult(
            np.asarray([1.0, 1.0, 2.0], dtype=np.float64),
            0.8,
            "ok",
            registered_depth_m=5.0,
            registration_status="raw_passthrough",
        ),
        depth_result=depth_result,
        person_anchor=torso_anchor,
        reject_current_geometry=False,
        flip_u=False,
        flip_v=False,
    )

    assert [item["kind"] for item in candidates] == ["registered_depth"]
    assert candidates[0]["support_state"] == "unknown"


@pytest.mark.parametrize(
    ("posture", "expected_support_state"),
    (("sitting", "seat"), ("lying", "couch")),
)
def test_current_body_depth_gets_typed_non_upright_support(
    posture: str,
    expected_support_state: str,
) -> None:
    processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
    processor.scene_priors = None
    processor._depth_candidate_covariance = lambda *_args, **_kwargs: Matrix3(  # type: ignore[method-assign]
        values=(
            0.04,
            0.0,
            0.0,
            0.0,
            0.04,
            0.0,
            0.0,
            0.0,
            0.04,
        )
    )
    body_anchor = hooks._PoseAnchorCandidate(
        u=100.0,
        v=200.0,
        source="person_body_projection",
        contact_basis="depth:torso_core",
        quality="estimated",
    )
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=7,
        class_id=0,
        bbox=(0.0, 0.0, 10.0, 20.0),
        score=0.9,
        sampling_mode="instance_mask",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=64,
        valid_fraction=0.9,
        anchor_source="torso_core",
        anchor_depth_m=5.0,
        anchor_sample_count=64,
        anchor_valid_fraction=0.9,
    )
    candidates: list[dict[str, object]] = []

    processor._append_universal_world_candidates(
        candidates,
        sensor_id=0,
        track={"tracker_id": 7, "world_floor_incidence_sin": 0.8},
        camera_id="cam0",
        calib=SimpleNamespace(floor_y=0.0),
        floor_candidate=None,
        floor_ray_admitted=False,
        anchor_candidate=None,
        pose_uv=None,
        contact_basis="depth:torso_core",
        posture=posture,
        occlusion_fraction=0.75,
        image_motion_supported=False,
        depth_obs=np.asarray([1.0, 1.0, 2.0], dtype=np.float64),
        depth_weight=0.8,
        depth_observation=hooks._DepthObservationResult(
            np.asarray([1.0, 1.0, 2.0], dtype=np.float64),
            0.8,
            "ok",
            registered_depth_m=5.0,
            registration_status="raw_passthrough",
        ),
        depth_result=depth_result,
        person_anchor=body_anchor,
        reject_current_geometry=False,
        flip_u=False,
        flip_v=False,
    )

    assert [item["kind"] for item in candidates] == ["registered_depth"]
    assert candidates[0]["support_state"] == expected_support_state
    assert candidates[0]["anchor"] == "person_body_projection"


def test_seated_pose_projection_requires_lifecycle_measured_height() -> None:
    processor = _processor()
    calibration = _upright_projection_calibration()
    state = hooks._WorldAnchorState()
    torso_world = np.asarray([0.35, 0.64, 5.2], dtype=np.float64)
    torso_uv = project_world_to_image(
        torso_world,
        calibration.intrinsics,
        calibration.extrinsics_col_major,
        calibration.image_size,
        unit_scale=1.0,
    )
    assert torso_uv is not None

    assert (
        processor._seated_pose_ground_projection(
            calibration,
            torso_uv=torso_uv,
            state=state,
            flip_u=False,
            flip_v=False,
        )
        is None
    )

    assert processor._record_seated_torso_height_reference(
        state,
        height_scene=0.64,
        uncertainty_scene=0.03,
        basis="registered_pose_torso_depth",
    )
    projection = processor._seated_pose_ground_projection(
        calibration,
        torso_uv=torso_uv,
        state=state,
        flip_u=False,
        flip_v=False,
    )

    assert projection is not None
    point, _covariance, height_m = projection
    assert point == pytest.approx([0.35, 0.0, 5.2], abs=1e-6)
    assert height_m == pytest.approx(0.64)


def test_registered_pose_torso_depth_measures_seated_projection_plane() -> None:
    processor = _processor()
    calibration = _upright_projection_calibration()
    state = hooks._WorldAnchorState()
    torso_world = np.asarray([-0.45, 0.71, 4.6], dtype=np.float64)
    torso_uv = project_world_to_image(
        torso_world,
        calibration.intrinsics,
        calibration.extrinsics_col_major,
        calibration.image_size,
        unit_scale=1.0,
    )
    assert torso_uv is not None
    extrinsics = np.asarray(
        calibration.extrinsics_col_major,
        dtype=np.float64,
    ).reshape((4, 4), order="F")
    camera_point = extrinsics[:3, :3] @ torso_world + extrinsics[:3, 3]
    registered_depth_m = float(camera_point[2])
    depth_result = ObjectDepthResult(
        source_id=0,
        frame_id=1,
        object_id=7,
        class_id=0,
        bbox=(0.0, 0.0, 10.0, 20.0),
        score=0.9,
        sampling_mode="pose_capsule_native",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=64,
        valid_fraction=0.9,
        anchor_uv=tuple(float(value) for value in torso_uv),
        anchor_source="pose_torso_support",
        anchor_depth_m=registered_depth_m,
        anchor_sample_count=64,
        anchor_valid_fraction=0.9,
        anchor_depth_spread_m=0.04,
    )
    depth_observation = hooks._DepthObservationResult(
        torso_world,
        0.8,
        "ok",
        registered_depth_m=registered_depth_m,
        registration_status="raw_passthrough",
    )

    measured = processor._seated_torso_height_from_registered_depth(
        calibration,
        torso_uv=torso_uv,
        depth_observation=depth_observation,
        depth_result=depth_result,
        flip_u=False,
        flip_v=False,
    )

    assert measured is not None
    assert measured[0] == pytest.approx(0.71, abs=1e-6)
    assert measured[1] > 0.0
    # A qualified sample at another anatomy pixel cannot teach this torso's
    # height. Otherwise shoulder fallback would create a circular pose plane.
    assert processor._seated_torso_height_from_registered_depth(
        calibration,
        torso_uv=(float(torso_uv[0]) + 3.0, float(torso_uv[1])),
        depth_observation=depth_observation,
        depth_result=depth_result,
        flip_u=False,
        flip_v=False,
    ) is None
    assert processor._record_seated_torso_height_reference(
        state,
        height_scene=measured[0],
        uncertainty_scene=measured[1],
        basis="registered_pose_torso_depth",
    )
    projection = processor._seated_pose_ground_projection(
        calibration,
        torso_uv=torso_uv,
        state=state,
        flip_u=False,
        flip_v=False,
    )
    assert projection is not None
    assert projection[0] == pytest.approx([-0.45, 0.0, 4.6], abs=1e-6)


def test_seated_height_rejects_non_depth_basis() -> None:
    processor = _processor()
    state = hooks._WorldAnchorState()
    assert not processor._record_seated_torso_height_reference(
        state,
        height_scene=0.88,
        uncertainty_scene=0.0,
        basis="published_ground_transition",
    )
    assert processor._record_seated_torso_height_reference(
        state,
        height_scene=0.63,
        uncertainty_scene=0.02,
        basis="registered_pose_torso_depth",
    )

    assert state.seated_torso_height_scene == pytest.approx(0.63)
    assert state.seated_torso_height_uncertainty_scene == pytest.approx(0.02)
    assert state.seated_torso_height_basis == "registered_pose_torso_depth"
    assert list(state.seated_torso_height_samples_scene) == [(0.63, 0.02)]


def test_moving_upright_lateral_truncation_excludes_interior_seating() -> None:
    processor = _processor()
    state = hooks._WorldAnchorState(
        height_ref_scene=1.7,
        last_full_body_ts=9.5,
        last_non_upright_ts=-1.0,
        upright_bbox_height_px=600.0,
        last_accepted_bbox_geometry=(1347.0, 524.25, 283.5, 396.0),
    )

    edge = processor._moving_upright_lateral_truncation(
        state=state,
        classified_posture="sitting",
        previous_posture="standing",
        previous_motion="walk",
        bbox=(1400.0, 200.0, 300.0, 520.0),
        image_size=(1920, 1080),
        now_ts=10.0,
        queue_metric_root_available=True,
    )
    interior = processor._moving_upright_lateral_truncation(
        state=state,
        classified_posture="sitting",
        previous_posture="standing",
        previous_motion="walk",
        bbox=(700.0, 400.0, 300.0, 300.0),
        image_size=(1920, 1080),
        now_ts=10.0,
        queue_metric_root_available=True,
    )
    state.last_accepted_bbox_geometry = (1400.0, 200.0, 300.0, 520.0)
    stationary_edge = processor._moving_upright_lateral_truncation(
        state=state,
        classified_posture="sitting",
        previous_posture="standing",
        previous_motion="walk",
        bbox=(1400.0, 200.0, 300.0, 520.0),
        image_size=(1920, 1080),
        now_ts=10.0,
        queue_metric_root_available=True,
    )
    hard_clip = processor._moving_upright_lateral_truncation(
        state=state,
        classified_posture="sitting",
        previous_posture="standing",
        previous_motion="walk",
        bbox=(1558.5, 436.5, 360.0, 415.5),
        image_size=(1920, 1080),
        now_ts=10.0,
        queue_metric_root_available=True,
    )

    assert edge[:3] == (True, True, False)
    assert edge[3] == pytest.approx(520.0 / 600.0)
    assert interior[:3] == (False, False, False)
    assert interior[3] == pytest.approx(0.5)
    assert stationary_edge[:3] == (False, True, False)
    assert hard_clip[:3] == (True, True, True)


def test_occluded_upright_body_planes_recover_floor_without_bbox_bottom() -> None:
    processor = _processor()
    calibration = _upright_projection_calibration()
    pose, furniture_truncated_bbox = _occluded_upright_pose(calibration)

    projection = processor._upright_pose_ground_projection(
        calibration,
        kpts_abs=pose,
        bbox=furniture_truncated_bbox,
        state=None,
        flip_u=False,
        flip_v=False,
    )

    assert projection is not None
    point, covariance, height_m, scatter_m, anchor_count, strong_proof = projection
    assert point == pytest.approx([0.0, 0.0, 6.0], abs=0.20)
    assert height_m == pytest.approx(1.8, abs=0.10)
    assert scatter_m < 0.30
    assert anchor_count == 5
    assert strong_proof is True
    assert covariance.values[0] >= 0.04
    assert covariance.values[8] >= 0.04


def test_torso_motion_anchor_tolerates_one_joint_dropout_only() -> None:
    processor = _processor()
    calibration = _upright_projection_calibration()
    pose, bbox = _occluded_upright_pose(calibration)

    pose[hooks._POSE_KPT_INDEX["right_shoulder"], 2] = 0.0
    assert processor._resolve_pose_torso_motion_anchor(pose, bbox=bbox) is not None

    pose[hooks._POSE_KPT_INDEX["right_hip"], 2] = 0.0
    assert processor._resolve_pose_torso_motion_anchor(pose, bbox=bbox) is None


def test_four_body_planes_remain_weak_without_independent_head_plane() -> None:
    processor = _processor()
    calibration = _upright_projection_calibration()
    pose, bbox = _occluded_upright_pose(calibration)
    pose[hooks._POSE_KPT_INDEX["nose"], 2] = 0.0

    projection = processor._upright_pose_ground_projection(
        calibration,
        kpts_abs=pose,
        bbox=bbox,
        state=None,
        flip_u=False,
        flip_v=False,
    )

    assert projection is not None
    _point, _covariance, _height_m, _scatter_m, anchor_count, strong_proof = (
        projection
    )
    assert anchor_count == 4
    assert strong_proof is False


def test_complete_side_profile_torso_does_not_require_apparent_pair_width() -> None:
    processor = _processor()
    calibration = _upright_projection_calibration()
    pose, bbox = _occluded_upright_pose(calibration)

    for left_name, right_name in (
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
    ):
        left_index = hooks._POSE_KPT_INDEX[left_name]
        right_index = hooks._POSE_KPT_INDEX[right_name]
        center_x = 0.5 * (pose[left_index, 0] + pose[right_index, 0])
        pose[left_index, 0] = center_x - 1.0
        pose[right_index, 0] = center_x + 1.0

    assert processor._resolve_pose_torso_motion_anchor(pose, bbox=bbox) is not None
    assert (
        processor._upright_pose_ground_projection(
            calibration,
            kpts_abs=pose,
            bbox=bbox,
            state=None,
            flip_u=False,
            flip_v=False,
        )
        is not None
    )


@pytest.mark.parametrize(
    "kind, anchor, support_state, posture, expected",
    (
        ("pose_scale", "seated_torso_plane", "seat", "sitting", True),
        ("pose_scale", "upright_body_plane", "unknown", "standing", True),
        (
            "registered_depth",
            "person_body_projection",
            "seat",
            "sitting",
            True,
        ),
        (
            "registered_depth",
            "person_body_projection",
            "couch",
            "lying",
            True,
        ),
        (
            "registered_depth",
            "person_body_projection",
            "unknown",
            "sitting",
            False,
        ),
        ("pose_scale", "seated_torso_plane", "seat", "standing", False),
    ),
)
def test_only_typed_body_projections_support_ground_footprint(
    kind: str,
    anchor: str,
    support_state: str,
    posture: str,
    expected: bool,
) -> None:
    candidate = SimpleNamespace(
        kind=kind,
        anchor=anchor,
        support_state=support_state,
        posture=posture,
    )

    assert (
        hooks._AnalyticsTelemetryProcessor
        ._resolver_candidate_is_ground_footprint_supported(candidate)
        is expected
    )


@pytest.mark.parametrize(
    "field, value",
    (
        ("source_id", 6),
        ("tracker_id", 18),
        ("frame_id", 43),
        ("media_pts_ns", 123457000),
    ),
)
def test_depth_measurement_requires_exact_current_track_cohort(field, value) -> None:
    result = ObjectDepthResult(
        source_id=5,
        frame_id=42,
        object_id=17,
        class_id=0,
        bbox=(0.0, 0.0, 10.0, 20.0),
        score=0.9,
        sampling_mode="pose_capsule_native",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=32,
        valid_fraction=0.9,
        measurement_frame_id=42,
        measurement_ts_us=123456,
        measurement_age_us=0,
        measurement_cached=False,
        depth_tensor_frame_id=42,
        depth_tensor_ts_us=123456,
        depth_tensor_age_frames=0,
        depth_tensor_age_us=0,
        ts_us=123456,
    )
    track = {
        "source_id": 5,
        "tracker_id": 17,
        "frame_id": 42,
        "media_pts_ns": 123456000,
    }
    assert hooks._AnalyticsTelemetryProcessor._depth_measurement_is_current(  # type: ignore[attr-defined]
        result,
        track=track,
    ) is True
    mismatched = dict(track)
    mismatched[field] = value
    assert hooks._AnalyticsTelemetryProcessor._depth_measurement_is_current(  # type: ignore[attr-defined]
        result,
        track=mismatched,
    ) is False


def test_depth_measurement_rejects_lagged_tensor_even_with_current_object_cohort() -> None:
    result = ObjectDepthResult(
        source_id=5,
        frame_id=42,
        object_id=17,
        class_id=0,
        bbox=(0.0, 0.0, 10.0, 20.0),
        score=0.9,
        sampling_mode="pose_capsule_native",
        status="ok",
        unit="m",
        is_metric=True,
        sample_count=32,
        valid_fraction=0.9,
        measurement_frame_id=42,
        measurement_ts_us=123456,
        measurement_age_us=0,
        measurement_cached=False,
        depth_tensor_frame_id=41,
        depth_tensor_ts_us=90123,
        depth_tensor_age_frames=1,
        depth_tensor_age_us=33333,
        ts_us=123456,
    )
    assert hooks._AnalyticsTelemetryProcessor._depth_measurement_is_current(  # type: ignore[attr-defined]
        result,
        track={
            "source_id": 5,
            "tracker_id": 17,
            "frame_id": 42,
            "media_pts_ns": 123456000,
        },
    ) is False


def test_weak_resolver_result_is_diagnostic_only() -> None:
    processor = _processor()
    processor.set_world_resolver_diagnostics_enabled(True)
    cohort = _cohort()
    measurement_set = WorldMeasurementSet(
        cohort=cohort,
        hypotheses=(
            WorldMeasurementHypothesis(
                candidate_id="weak",
                cohort=cohort,
                kind="gravity_reconstruction",
                position=Vector3(x=1.0, y=0.0, z=2.0),
                covariance=Matrix3(
                    values=(
                        10.0,
                        0.0,
                        0.0,
                        0.0,
                        10.0,
                        0.0,
                        0.0,
                        0.0,
                        10.0,
                    )
                ),
                anchor="learned_body_height",
                confidence=0.01,
                support_score=0.01,
                source_reliability=0.01,
            ),
        ),
    )
    track: dict[str, object] = {}

    point, result = processor._apply_resolved_world_measurement(track, measurement_set)

    assert point is not None
    assert result is not None
    assert result.quality == "weak"
    assert track["world_resolver"]["decision"] == "weak"  # type: ignore[index]


def test_weak_bbox_requires_motion_or_trusted_lifecycle_consensus() -> None:
    processor = _processor()
    cohort = _cohort(pcf_revision="revision")
    supported = WorldMeasurementHypothesis(
        candidate_id="floor",
        cohort=cohort,
        kind="floor_ray",
        position=Vector3(x=1.0, y=0.0, z=2.0),
        covariance=Matrix3(
            values=(0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04)
        ),
        anchor="bbox_bottom",
        confidence=0.75,
        support_score=0.72,
        support_state="floor",
        posture="standing",
        pcf=WorldPriorEvidence(
            prior_id="prior",
            revision_id="revision",
            status="fail",
            inside_extent=True,
            inside_authored_space=False,
            boundary_signed_distance_m=-2.0,
        ),
    )
    unsupported = WorldMeasurementHypothesis(
        candidate_id="gravity",
        cohort=cohort,
        kind="gravity_reconstruction",
        position=Vector3(x=1.0, y=0.0, z=2.0),
        covariance=Matrix3(
            values=(4.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 4.0)
        ),
        anchor="learned_body_height",
        confidence=0.75,
        support_score=0.72,
        source_reliability=0.01,
        support_state="floor",
        posture="standing",
    )
    floor_set = WorldMeasurementSet(cohort=cohort, hypotheses=(supported,))
    gravity_set = WorldMeasurementSet(cohort=cohort, hypotheses=(unsupported,))

    floor_result = processor._world_resolver.resolve(floor_set)
    gravity_result = processor._world_resolver.resolve(gravity_set)

    assert floor_result.quality == "weak"
    assert processor._weak_resolver_measurement_is_ground_supported(
        floor_result,
        floor_set,
    ) is False
    assert processor._weak_resolver_measurement_is_ground_supported(
        floor_result,
        floor_set,
        image_motion_supported=True,
    ) is True
    trusted_state = hooks._WorldAnchorState(
        last_good_world=(0.0, 0.0, 0.0),
        last_good_ts=1.0,
        last_output_world_x=0.0,
        last_output_world_z=0.0,
    )
    assert processor._weak_resolver_measurement_is_ground_supported(
        floor_result,
        floor_set,
        state=trusted_state,
    ) is True
    assert gravity_result.quality == "weak"
    assert processor._weak_resolver_measurement_is_ground_supported(
        gravity_result,
        gravity_set,
    ) is False


def test_weak_observed_ankle_pcf_conflict_requires_metric_lifecycle() -> None:
    processor = _processor()
    cohort = _cohort(pcf_revision="revision")
    ankle = WorldMeasurementHypothesis(
        candidate_id="ankle",
        cohort=cohort,
        kind="floor_ray",
        position=Vector3(x=1.0, y=0.0, z=2.0),
        covariance=Matrix3(
            values=(0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04)
        ),
        anchor="pose_ankle_floor",
        confidence=0.95,
        support_score=0.95,
        support_state="floor",
        posture="sitting",
        pcf=WorldPriorEvidence(
            prior_id="prior",
            revision_id="revision",
            status="fail",
            inside_extent=True,
            inside_authored_space=False,
            boundary_signed_distance_m=-2.0,
        ),
    )
    measurement_set = WorldMeasurementSet(cohort=cohort, hypotheses=(ankle,))
    result = processor._world_resolver.resolve(measurement_set)

    assert result.quality == "weak"
    assert processor._weak_resolver_measurement_is_ground_supported(
        result,
        measurement_set,
    ) is False
    assert processor._weak_resolver_measurement_is_ground_supported(
        result,
        measurement_set,
        verified_first_output_contact=True,
    ) is True
    trusted_state = hooks._WorldAnchorState(
        last_good_world=(0.0, 0.0, 0.0),
        last_good_ts=1.0,
        last_output_world_x=0.0,
        last_output_world_z=0.0,
    )
    assert processor._weak_resolver_measurement_is_ground_supported(
        result,
        measurement_set,
        state=trusted_state,
    ) is True

    tight_contact = {
        "pose_present": True,
        "world_floor_admitted": True,
        "world_floor_contact_plausible": True,
        "world_floor_contact_gap_ratio": 0.0,
        "world_floor_contact_range_delta_m": 0.0,
        "world_floor_incidence_sin": 0.30,
    }
    assert processor._first_output_ankle_contact_is_tight(tight_contact) is True
    reflected_contact = dict(
        tight_contact,
        world_floor_contact_gap_ratio=0.11,
        world_floor_contact_range_delta_m=1.54,
    )
    assert processor._first_output_ankle_contact_is_tight(reflected_contact) is False
    near_horizon_contact = dict(
        tight_contact,
        world_floor_incidence_sin=0.16,
    )
    assert (
        processor._first_output_ankle_contact_is_tight(near_horizon_contact)
        is False
    )
    assert (
        processor._cold_floor_ray_incidence_is_adequate(tight_contact) is True
    )
    assert (
        processor._cold_floor_ray_incidence_is_adequate(near_horizon_contact)
        is False
    )

def test_rich_resolver_diagnostics_are_off_by_default_but_summary_remains() -> None:
    processor = _processor()
    cohort = _cohort()
    measurement_set = WorldMeasurementSet(
        cohort=cohort,
        hypotheses=(_hypothesis(cohort, candidate_id="floor_ray"),),
    )
    track: dict[str, object] = {
        "world_resolver": {"stale": True},
    }

    point, result = processor._apply_resolved_world_measurement(track, measurement_set)

    assert point is not None
    assert result is not None
    assert "world_resolver" not in track
    assert track["world_resolver_confidence"] == pytest.approx(result.confidence)
    assert track["world_resolver_selected_id"] == "floor_ray"
    assert track["world_quantity"] == "ground_footprint"


def test_canonical_tracking_copy_excludes_rich_resolver_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _processor()
    published: list[object] = []
    monkeypatch.setattr(processor, "_publish_tracking_work", published.append)
    track: dict[str, object] = {
        "camera_id": "camera-0",
        "tracker_id": 17,
        "frame_id": 42,
        "class_id": 0,
        "world_resolver": {
            "contract": "noesis.world_resolver_diagnostics",
            "candidates": [{"id": "floor_ray", "covariance": {"values": [1.0] * 9}}],
        },
        "world_resolver_confidence": 0.9,
        "world_resolver_selected_id": "floor_ray",
        "_world_force_first_metric_publication": True,
    }
    continuity = processor._prepare_tracking_cohort(
        sensor_id=0,
        camera_id="camera-0",
        frame_id=42,
        observed_at_us=123456,
        tracks=[track],
        footpoints=[],
    )

    assert processor._enqueue_tracking_publication(
        sensor_id=0,
        camera_id="camera-0",
        frame_meta=SimpleNamespace(frame_number=42, buf_pts=123456000),
        tracks=[track],
        footpoints=[],
        now_ts=1.0,
        temporal_contract={"observed_at_us": 123456},
        continuity=continuity,
        force=True,
    )

    assert len(published) == 1
    queued_track = published[0].tracks[0]  # type: ignore[attr-defined]
    assert "world_resolver" not in queued_track
    assert "_world_force_first_metric_publication" not in queued_track
    assert queued_track["world_resolver_confidence"] == 0.9
    assert queued_track["world_resolver_selected_id"] == "floor_ray"
    assert "world_resolver" in track
    assert track["_world_force_first_metric_publication"] is True


def test_universal_resolver_source_continuity_is_evidence_not_a_veto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _processor()
    state = SimpleNamespace()
    track: dict[str, object] = {"world_resolver": {"decision": "good"}}
    monkeypatch.setattr(processor, "_admit_live_world_source", lambda *_args, **_kwargs: False)

    continuity_match = processor._observe_resolver_source_continuity(
        track,
        state,
        candidate_source="pose_floor_only",
        quality="good",
        depth_weight=0.0,
        posture="standing",
    )

    assert continuity_match is False
    assert track["world_resolver_source_continuity_match"] is False
    assert track["world_resolver"]["source_continuity_match"] is False  # type: ignore[index]
