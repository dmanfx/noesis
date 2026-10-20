"""Actual Python pose producer/consumer, mocked UID4 metadata; no SDK/GPU."""
import json
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def pose_path(monkeypatch):
    for name in ('pyservicemaker','pyds','noesis_pose_meta_ext','noesis_analytics_meta_ext',
        'noesis_reid_meta_ext','noesis_v3dt_meta_ext','noesis_depth_meta_ext','noesis_depth_tracking_tensor_ext'):
        monkeypatch.setitem(sys.modules,name,None)
    from noesis.pipelines import hooks
    calls=Counter()
    def extract(obj,gie,w,h,score,letterbox):
        calls[obj.object_id]+=1
        assert (gie,w,h,score,letterbox)==(4,640,640,.25,True)
        if obj.missing:return None
        joints=[[float(i+1),float(i+2),.9] for i in range(17)]
        return {'score':.9,'keypoints_roi':joints,
            'keypoints_abs':[[x+obj.rect_params.left,y+obj.rect_params.top,c] for x,y,c in joints]}
    def attach(batch,obj,text,replace):obj.payload=text;return True
    native=SimpleNamespace(extract_pose_keypoints=extract,attach_pose_features=attach,
        extract_pose_features=lambda obj:obj.payload)
    monkeypatch.setattr(hooks,'noesis_pose_meta_ext',native)
    monkeypatch.setenv('NOESIS_POSE_FEATURES_PER_FRAME_MAX','2')
    pipeline=SimpleNamespace(config={'models':{'pose':{'name':'yolo26_pose'}}},
        stable_id_mgr=SimpleNamespace(active_tracks={}),components={},frame_size=(1920,1080))
    producer=hooks.PoseFeatureProcessor(pipeline=pipeline,gie_id=4)
    pipeline.components['yolo26_pose']=SimpleNamespace(config={'_pose_feature_processor':producer})
    objects=[SimpleNamespace(object_id=i,class_id=0,missing=False,payload='',
        rect_params=SimpleNamespace(left=30.,top=40.,width=32.5 if i==4 else 100.,height=75. if i==4 else 200.))
        for i in range(5)]
    return SimpleNamespace(hooks=hooks,producer=producer,pipeline=pipeline,objects=objects,calls=calls)


def run_frame(n,frame,source=1):
    for obj in n.objects:obj.payload=''
    n.producer.handle_servicemaker_frame(object(),SimpleNamespace(source_id=source,
        frame_number=frame,buf_pts=frame*33_333_333,object_items=iter(n.objects)))


def task_frame(n, frame, source=1, objects=None):
    n.pipeline.config['xc3d_world_quality']={'contract':'noesis.xc3d.world_quality.v1','enabled':True}
    pts=frame*33_333_333+777
    receipt=dict(source_id=source,original_source_id=source,receipt_valid=True,
        sdk_source_id=source,original_sdk_source_id=source,slot=0,original_batch_id=0,
        original_buffer_scope=frame+1,original_ntp_ns=0,post_tracker_ntp_ns=0,
        original_frame=frame,original_pts_ns=pts,post_tracker_pts_ns=pts)
    n.producer.handle_servicemaker_frame(object(),SimpleNamespace(source_id=source,
        frame_number=frame+900,buf_pts=pts,object_items=_SinglePassBorrowedObjects(
            n.objects if objects is None else objects)), source_receipt=receipt)
    return (source, frame, pts)


def test_task_early_anchor_spends_existing_allowance_once_before_consumers(pose_path,monkeypatch):
    n=pose_path
    monkeypatch.setenv('NOESIS_POSE_FEATURES_PER_FRAME_MAX','1')
    monkeypatch.setenv('NOESIS_POSE_ANCHOR_NATIVE_EXTRACTS_PER_FRAME_MAX','1')
    source,frame,pts=task_frame(n,1)
    assert n.calls=={0:1,1:1}
    obj=n.objects[1];bbox=[30.,40.,100.,200.]
    assert obj.payload==''  # Raw anchor result is not new feature output.
    cohort=(source,1,frame,pts)
    assert n.producer._owned_pose_result(cohort,bbox,(4,(640,640),.25,True))[0]
    anchor=_fallback_anchor(n);status={}
    assert anchor._extract_pose_keypoints_for_anchor(obj,bbox,current_cohort=cohort,
        current_cohort_status=status).shape==(17,3)
    assert status['current'] and n.calls=={0:1,1:1}
    # Unattempted object cannot spend the relocated allowance downstream.
    assert anchor._extract_pose_keypoints_for_anchor(n.objects[4],[30.,40.,32.5,75.],
        current_cohort=(source,4,frame,pts),current_cohort_status={}) is None
    assert sum(n.calls.values())==2


@pytest.mark.parametrize('fault',['negative','bbox','profile','pts','expiry','eviction','reset','stop','wrap'])
def test_task_rendezvous_exact_miss_never_reextracts(pose_path,monkeypatch,fault):
    n=pose_path
    monkeypatch.setenv('NOESIS_POSE_FEATURES_PER_FRAME_MAX','0')
    monkeypatch.setenv('NOESIS_POSE_ANCHOR_NATIVE_EXTRACTS_PER_FRAME_MAX','1')
    if fault=='negative':n.objects[0].missing=True
    source,frame,pts=task_frame(n,10)
    bbox=[30.,40.,100.,200.];profile=(4,(640,640),.25,True)
    cohort=(source,0,frame,pts)
    if fault=='bbox':bbox[0]+=1
    elif fault=='profile':profile=(9,(640,640),.25,True)
    elif fault=='pts':cohort=(source,0,frame,pts+1)
    elif fault=='expiry':
        entry=n.producer._pose_inflight[(source,frame,pts)]
        n.producer._pose_inflight[(source,frame,pts)]=(entry[0]-3,entry[1],entry[2])
    elif fault=='eviction':
        for i in range(11,76):task_frame(n,i)
        assert len(n.producer._pose_inflight)<=64
    elif fault=='reset':n.producer._clear_pose_source(source)
    elif fault=='stop':n.pipeline._xc3d_applied_source_stop_mask=1<<source
    elif fault=='wrap':task_frame(n,1)
    found,result=n.producer._owned_pose_result(cohort,bbox,profile)
    assert result is None and found==(fault=='negative')
    before=sum(n.calls.values());anchor=_fallback_anchor(n)
    anchor._extract_pose_keypoints_for_anchor(n.objects[0],bbox,current_cohort=cohort,current_cohort_status={})
    assert sum(n.calls.values())==before


def test_task_exact_inflight_original_survives_latest_advance_not_run_or_epoch(pose_path,monkeypatch):
    from noesis.pipelines.source_acquisition import SourceAcquisitionState
    n=pose_path;n.pipeline.xc3d_source_acquisition=SourceAcquisitionState('run','graph')
    _,frame,pts=task_frame(n,1)
    task_frame(n,2)
    args=((1,0,frame,pts),[30.,40.,100.,200.],(4,(640,640),.25,True))
    assert n.producer._owned_pose_result(*args)[0]
    n.pipeline.xc3d_source_acquisition.reset(1,'test_reset')
    assert not n.producer._owned_pose_result(*args)[0]


def test_task_both_zero_and_missing_original_do_not_extract(pose_path,monkeypatch):
    n=pose_path
    monkeypatch.setenv('NOESIS_POSE_FEATURES_PER_FRAME_MAX','0')
    monkeypatch.setenv('NOESIS_POSE_ANCHOR_NATIVE_EXTRACTS_PER_FRAME_MAX','0')
    task_frame(n,1)
    assert not n.calls
    n.producer.handle_servicemaker_frame(object(),SimpleNamespace(source_id=1,
        frame_number=2,buf_pts=66_666_666,object_items=iter(n.objects)))
    assert not n.calls


def test_task_duplicate_and_expired_duplicate_never_spend_twice(pose_path,monkeypatch):
    n=pose_path
    monkeypatch.setenv('NOESIS_POSE_FEATURES_PER_FRAME_MAX','0')
    task_frame(n,1)
    count=sum(n.calls.values())
    n.producer._pose_inflight.clear()
    task_frame(n,1)
    assert sum(n.calls.values())==count


def test_task_native_marker_is_negative_and_no_unbound_late_fallback(pose_path,monkeypatch):
    n=pose_path
    monkeypatch.setenv('NOESIS_POSE_FEATURES_PER_FRAME_MAX','0')
    real=n.hooks.noesis_pose_meta_ext.extract_pose_keypoints
    monkeypatch.setattr(n.hooks.noesis_pose_meta_ext,'extract_pose_keypoints',
        lambda *args:{**real(*args),'pose_cache_reused':True})
    _,frame,pts=task_frame(n,1)
    assert n.producer._owned_pose_result((1,0,frame,pts),[30.,40.,100.,200.],
        (4,(640,640),.25,True))==(True,None)
    count=sum(n.calls.values())
    assert _fallback_anchor(n)._extract_pose_keypoints_for_anchor(n.objects[4],
        [30.,40.,32.5,75.]) is None
    assert sum(n.calls.values())==count


def test_fair_current_pose_late_small_child_is_not_starved(pose_path):
    n=pose_path
    for frame in range(1,9):
        before=sum(n.calls.values())
        run_frame(n,frame)
        assert sum(n.calls.values())-before<=2
    assert all(n.calls[i]>=2 for i in range(5)),n.calls


def test_current_tensor_wins_over_cached_pose_F(pose_path):
    n=pose_path
    n.producer._cache_pose_payload(1,0,[30.,40.,100.,200.],1,
        {'score':.9,'keypoints_roi':[[1.,1.,.9]]*17,'source_id':1,'object_id':0,'frame_id':1,'ts_us':33333})
    run_frame(n,2)
    assert n.calls[0]==1
    payload=json.loads(n.objects[0].payload)
    assert payload['frame_id']==2 and not payload.get('pose_cache_reused',False)


def test_failed_uid4_attempt_is_not_repeated_by_anchor(pose_path):
    n=pose_path
    n.objects[0].missing=True
    run_frame(n,1)
    anchor=object.__new__(n.hooks._AnalyticsTelemetryProcessor)
    anchor.pipeline=n.pipeline
    anchor._pose_anchor_native_remaining=1
    anchor._pose_anchor_gie_id=4;anchor._pose_anchor_model_size=(640,640)
    anchor._pose_anchor_score_threshold=.25;anchor._pose_anchor_letterbox=True
    status={}
    value=anchor._extract_pose_keypoints_for_anchor(n.objects[0],[30.,40.,100.,200.],
        current_cohort=(1,0,1,33_333_333),current_cohort_status=status)
    assert value is None and n.calls[0]==1 and status['current'] is False


def test_shared_current_pose_can_survive_optional_feature_attach_failure(pose_path,monkeypatch):
    n=pose_path
    monkeypatch.setattr(n.producer,'_attach_pose_payload',lambda *args:False)
    run_frame(n,1)
    anchor=object.__new__(n.hooks._AnalyticsTelemetryProcessor)
    anchor.pipeline=n.pipeline
    anchor._pose_anchor_native_remaining=1
    anchor._pose_anchor_gie_id=4;anchor._pose_anchor_model_size=(640,640)
    anchor._pose_anchor_score_threshold=.25;anchor._pose_anchor_letterbox=True
    status={}
    value=anchor._extract_pose_keypoints_for_anchor(n.objects[0],[30.,40.,100.,200.],
        current_cohort=(1,0,1,33_333_333),current_cohort_status=status)
    assert value.shape==(17,3) and n.calls[0]==1 and status['current'] is True


@pytest.mark.parametrize('fault',['source','raw','frame','pts','bbox','profile','next_frame','absence'])
def test_owned_pose_never_crosses_original_binding(pose_path,fault):
    n=pose_path;run_frame(n,1)
    cohort=(1,0,1,33_333_333);bbox=[30.,40.,100.,200.];profile=(4,(640,640),.25,True)
    assert n.producer._owned_pose_result(cohort,bbox,profile)[0]
    if fault=='source':cohort=(2,0,1,33_333_333)
    elif fault=='raw':cohort=(1,99,1,33_333_333)
    elif fault=='frame':cohort=(1,0,2,33_333_333)
    elif fault=='pts':cohort=(1,0,1,33_333_334)
    elif fault=='bbox':bbox[0]+=1
    elif fault=='profile':profile=(5,(640,640),.25,True)
    elif fault=='next_frame':run_frame(n,2)
    elif fault=='absence':
        n.objects=[];run_frame(n,2)
    assert n.producer._owned_pose_result(cohort,bbox,profile)==(False,None)


def test_fair_pose_order_and_sources_are_independent(pose_path):
    n=pose_path
    for frame in range(1,4):
        n.objects.reverse()
        run_frame(n,frame,source=1)
        run_frame(n,frame,source=2)
    assert all(n.calls[i]>=2 for i in range(5))
    assert len(n.producer._current_pose_results)==2


def test_pose_zero_budget_and_transient_population_bound(pose_path,monkeypatch):
    n=pose_path
    monkeypatch.setenv('NOESIS_POSE_FEATURES_PER_FRAME_MAX','0')
    run_frame(n,1)
    assert not n.calls and not n.producer._current_pose_results[1][2]
    monkeypatch.setenv('NOESIS_POSE_FEATURES_PER_FRAME_MAX','2')
    n.objects=n.objects*30
    run_frame(n,2)
    assert sum(n.calls.values())==2
    assert len(n.producer._current_pose_results[1][2])<=2


def test_current_anchor_does_not_accept_cached_F_retimestamp(pose_path):
    n=pose_path
    obj=n.objects[4]
    obj.payload=json.dumps({'source_id':1,'object_id':4,'frame_id':2,'ts_us':66666,
        'pose_cache_reused':True,'pose_cache_age_frames':1,'keypoints_abs':[[1.,1.,.9]]*17})
    anchor=object.__new__(n.hooks._AnalyticsTelemetryProcessor)
    anchor.pipeline=n.pipeline;anchor._pose_anchor_native_remaining=0
    status={}
    value=anchor._extract_pose_keypoints_for_anchor(obj,[30.,40.,32.5,75.],
        current_cohort=(1,4,2,66_666_666),current_cohort_status=status)
    assert value is None and status['current'] is False and not n.calls


def test_duplicate_object_does_not_duplicate_uid4_copy(pose_path):
    n=pose_path;n.objects=[n.objects[0],n.objects[0],n.objects[1]]
    run_frame(n,1)
    assert n.calls=={0:1,1:1}


class _BorrowedObject:
    """Model SDK iterator-owned current T; advancing invalidates old wrapper."""
    def __init__(self,data):
        object.__setattr__(self,'_data',data)
        object.__setattr__(self,'_valid',True)

    def __getattr__(self,name):
        if not self._valid:
            raise RuntimeError('SDK wrapper used after iterator advancement')
        return getattr(self._data,name)

    def __setattr__(self,name,value):
        if name.startswith('_'):
            object.__setattr__(self,name,value)
        else:
            if not self._valid:
                raise RuntimeError('SDK wrapper attachment after iterator advancement')
            setattr(self._data,name,value)


class _SinglePassBorrowedObjects:
    def __init__(self,objects):
        self.objects=objects
        self.index=0
        self.current=None

    def __iter__(self):return self

    def __next__(self):
        if self.current is not None:self.current._valid=False
        if self.index==len(self.objects):raise StopIteration
        self.current=_BorrowedObject(self.objects[self.index])
        self.index+=1
        return self.current


def test_single_pass_SDK_lifetime_fairness_and_current_results(pose_path):
    n=pose_path
    for frame in range(1,7):
        n.objects.reverse()
        for obj in n.objects:obj.payload=''
        borrowed=_SinglePassBorrowedObjects(n.objects)
        n.producer.handle_servicemaker_frame(object(),SimpleNamespace(source_id=1,
            frame_number=frame,buf_pts=frame*33_333_333,object_items=borrowed))
        assert borrowed.index==5
        assert len(n.producer._current_pose_results[1][2])<=2
    assert all(n.calls[i]>=2 for i in range(5)),n.calls


def test_single_pass_churn_never_starves_persistent_child_or_busy_frame(pose_path):
    n=pose_path
    child=n.objects[4]
    child.object_id=999
    for frame in range(1,17):
        # All four foreground IDs churn; the persistent child remains last.
        others=[SimpleNamespace(**vars(o)) for o in n.objects[:4]]
        for i,obj in enumerate(others):obj.object_id=frame*10+i
        objects=others+[child]
        before=sum(n.calls.values())
        n.producer.handle_servicemaker_frame(object(),SimpleNamespace(source_id=1,
            frame_number=frame,buf_pts=frame*33_333_333,
            object_items=_SinglePassBorrowedObjects(objects)))
        assert 1<=sum(n.calls.values())-before<=2
        assert len(n.producer._pose_candidates[1])==5
    assert n.calls[999]>=4,n.calls


def test_single_pass_moving_boxes_do_not_renew_scheduler_identity(pose_path):
    n=pose_path
    for frame in range(1,9):
        for obj in n.objects:obj.rect_params.left+=1.
        n.producer.handle_servicemaker_frame(object(),SimpleNamespace(source_id=1,
            frame_number=frame,buf_pts=frame*33_333_333,
            object_items=_SinglePassBorrowedObjects(n.objects)))
    assert all(n.calls[i]>=2 for i in range(5)),n.calls
    assert all(type(key[0]) is tuple and all(type(v) is int for v in key[0])
               and type(key[1]) is int for key in n.producer._pose_candidates[1])


def _fallback_anchor(n):
    anchor=object.__new__(n.hooks._AnalyticsTelemetryProcessor)
    anchor.pipeline=n.pipeline;anchor._pose_anchor_native_remaining=1
    anchor._pose_anchor_gie_id=4;anchor._pose_anchor_model_size=(640,640)
    anchor._pose_anchor_score_threshold=.25;anchor._pose_anchor_letterbox=True
    return anchor


def _actual_native_frame_context(n,frame=1):
    from noesis.pipelines.source_acquisition import (SourceAcquisitionState,ScopedFrameReceipts,
        PrivateAssociationWork,current_frame_key)
    acquisition=SourceAcquisitionState('synthetic-pose-run','synthetic-pose-graph')
    n.pipeline.xc3d_source_acquisition=acquisition
    pts=frame*33_333_333
    row={'source_id':1,'original_source_id':1,'original_frame':frame+100,
         'slot':0,'original_batch_id':0,'scope':31,'original_buffer_scope':41,
         'sdk_source_id':101,'original_sdk_source_id':101,
         'original_pts_ns':pts,'original_ntp_ns':10_000_000_000+pts,
         'post_tracker_ntp_ns':10_000_000_000+pts,
         'post_tracker_frame':frame,'post_tracker_pts_ns':pts,'receipt_valid':True}
    batch=object()
    live_frame=SimpleNamespace(batch_id=0,pad_index=1,source_id=101,
        frame_number=frame,buf_pts=pts,ntp_timestamp=row['original_ntp_ns'])
    scoped=ScopedFrameReceipts(batch,{'scope':31,'original_buffer_scope':41,'rows':[row]})
    matched=scoped.lookup(batch,live_frame)
    original=acquisition.observe_native(matched)
    work=PrivateAssociationWork(1,frame,pts,0,original,None)
    return work,matched,current_frame_key(live_frame)


def test_genuine_current_UID4_fallback_reaches_owned_body(pose_path):
    from noesis.pipelines.world_quality_adapter import capture_body_support
    n=pose_path;run_frame(n,1)
    obj=n.objects[4]  # Outside feature budget; SAME live current object tensor.
    assert n.calls[4]==0
    anchor=_fallback_anchor(n);status={}
    points=anchor._extract_pose_keypoints_for_anchor(obj,[30.,40.,32.5,75.],
        current_cohort=(1,4,1,33_333_333),current_cohort_status=status,
        native_frame_context=_actual_native_frame_context(n))
    assert points.shape==(17,3) and n.calls[4]==1 and status['current'] is True
    # Actual public owner row supplies lifecycle; no invented body receipt.
    from noesis_core.tracking_continuity import TrackingLifecycleRegistry
    owner=TrackingLifecycleRegistry()
    life=owner.identity_observation(1,4,frame_id=1,observed_at_us=33333,
        media_pts_ns=33_333_333,bbox=[30.,40.,32.5,75.])
    update=owner.update_frame(source_id=1,camera_id='family-room',frame_id=1,
        observed_at_us=33333,tracks=[{'tracker_id':4,'bbox':[30.,40.,32.5,75.]}],media_pts_ns=33_333_333)
    assert update.active_lifecycles[0].generation==life.generation
    body=capture_body_support({'bbox':[30.,40.,32.5,75.],'image_size':[1920,1080]},
        life,points,pose_is_current=status['current'])
    assert body is not None and len(body.keypoints)==17


@pytest.mark.parametrize('fault',['source','raw','frame','pts','zero_pts','bbox','profile',
                                  'missing','cached','nonfinite','no_processor'])
def test_native_fallback_cannot_invent_original_currentness(pose_path,monkeypatch,fault):
    n=pose_path;run_frame(n,1)
    obj=n.objects[4];anchor=_fallback_anchor(n);status={}
    cohort=(1,4,1,33_333_333);bbox=[30.,40.,32.5,75.]
    if fault=='source':cohort=(2,4,1,33_333_333)
    if fault=='raw':cohort=(1,99,1,33_333_333)
    if fault=='frame':cohort=(1,4,2,33_333_333)
    if fault=='pts':cohort=(1,4,1,33_333_334)
    if fault=='zero_pts':cohort=(1,4,1,0)
    if fault=='bbox':bbox[0]+=1.
    if fault=='profile':anchor._pose_anchor_gie_id=99
    if fault=='missing':obj.missing=True
    if fault=='cached':
        real=n.hooks.noesis_pose_meta_ext.extract_pose_keypoints
        monkeypatch.setattr(n.hooks.noesis_pose_meta_ext,'extract_pose_keypoints',
            lambda *a:{**real(*a),'pose_cache_reused':True})
    if fault=='nonfinite':
        monkeypatch.setattr(n.hooks.noesis_pose_meta_ext,'extract_pose_keypoints',
            lambda *a:{'score':.9,'keypoints_abs':[[float('nan'),1.,.9]]*17,
                       'keypoints_roi':[[1.,1.,.9]]*17})
    if fault=='no_processor':n.pipeline.components.clear()
    anchor._extract_pose_keypoints_for_anchor(obj,bbox,current_cohort=cohort,current_cohort_status=status,
        native_frame_context=_actual_native_frame_context(n))
    assert status['current'] is False


def test_original_F_native_fallback_survives_upstream_F_plus_one(pose_path):
    n=pose_path;run_frame(n,1)
    context=_actual_native_frame_context(n)
    data=SimpleNamespace(**vars(n.objects[4]))
    data.rect_params=SimpleNamespace(**vars(data.rect_params))
    iterator=_SinglePassBorrowedObjects([data])
    live_original=next(iterator)  # Original F remains live in its own iterator.
    run_frame(n,2)  # Upstream pose source latest mutable state is now F+1.
    assert n.producer._current_pose_results[1][:2]!=(1,33_333_333)
    anchor=_fallback_anchor(n);status={}
    points=anchor._extract_pose_keypoints_for_anchor(live_original,[30.,40.,32.5,75.],
        current_cohort=(1,4,1,33_333_333),current_cohort_status=status,native_frame_context=context)
    assert points.shape==(17,3) and status['current'] is True
    assert context[0].original.original_frame==101 and context[0].canonical_frame==1
    assert context[0].original.pts_ns==33_333_333  # Never relabeled F+1.
    assert live_original.payload==''  # No fresh JSON fabricated for this object.
    from noesis_core.tracking_continuity import TrackingLifecycleRegistry
    from noesis.pipelines.world_quality_adapter import capture_body_support
    owner=TrackingLifecycleRegistry()
    life=owner.identity_observation(1,4,frame_id=1,observed_at_us=33333,
        media_pts_ns=33_333_333,bbox=[30.,40.,32.5,75.])
    body=capture_body_support({'bbox':[30.,40.,32.5,75.],'image_size':[1920,1080]},
        life,points,pose_is_current=status['current'])
    assert body is not None and body.lifecycle.frame_id==1


@pytest.mark.parametrize('fault',['absent','buffer','slot','run','epoch','receipt','sdk_source','original_frame'])
def test_original_frame_context_denies_foreign_or_missing_binding(pose_path,fault):
    from dataclasses import replace
    n=pose_path;run_frame(n,1)
    work,row,key=_actual_native_frame_context(n)
    context=(work,row,key)
    if fault=='absent':context=None
    elif fault=='buffer':row['original_buffer_scope']+=1
    elif fault=='slot':key=(1,)+key[1:];context=(work,row,key)
    elif fault=='run':context=(replace(work,original=replace(work.original,run_id='foreign')),row,key)
    elif fault=='epoch':context=(replace(work,clock_epoch=1),row,key)
    elif fault=='receipt':row['receipt_valid']=False
    elif fault=='sdk_source':row['original_sdk_source_id']+=1
    elif fault=='original_frame':row['original_frame']+=1
    anchor=_fallback_anchor(n);status={}
    anchor._extract_pose_keypoints_for_anchor(n.objects[4],[30.,40.,32.5,75.],
        current_cohort=(1,4,1,33_333_333),current_cohort_status=status,native_frame_context=context)
    assert status['current'] is False


def test_actual_operator_passes_same_scoped_original_context_to_anchor(pose_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    from unittest.mock import Mock
    from noesis_core.runtime_publication import RuntimePublicationGate
    path=Path(__file__).with_name('test_xc3d_source_acquisition.py')
    spec=importlib.util.spec_from_file_location('pose_original_receipt_examples',path)
    examples=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=examples
    spec.loader.exec_module(examples)  # Helpers only; no old tests run.
    receipt=examples.row()
    hooks,operator,processor,_native=examples.operator_fixture(monkeypatch,[receipt])
    operator.handle_metadata(SimpleNamespace(frame_items=[examples.frame(receipt)]))
    kwargs=processor.handle_servicemaker_frame.call_args.kwargs
    work,row,key=kwargs['native_frame_context']
    assert work is kwargs['xc3d_private_evidence']
    assert key==(0,1,101,20001,33_333_333,1_033_333_333)
    n=pose_path;run_frame(n,2)
    n.pipeline.xc3d_source_acquisition=processor.pipeline.xc3d_source_acquisition
    anchor=_fallback_anchor(n);status={}
    points=anchor._extract_pose_keypoints_for_anchor(n.objects[4],[30.,40.,32.5,75.],
        current_cohort=(1,4,20001,33_333_333),current_cohort_status=status,
        native_frame_context=kwargs['native_frame_context'])
    assert points.shape==(17,3) and status['current'] is True
    admitted=object.__new__(hooks._AnalyticsTelemetryProcessor)
    admitted.publication_gate=RuntimePublicationGate()
    admitted._handle_servicemaker_frame_admitted=Mock()
    admitted.handle_servicemaker_frame(examples.frame(receipt),**kwargs)
    assert admitted._handle_servicemaker_frame_admitted.call_args.kwargs['native_frame_context'] is kwargs['native_frame_context']


def test_original_F_context_does_not_make_cached_JSON_a_current_tensor(pose_path):
    n=pose_path;run_frame(n,1)
    context=_actual_native_frame_context(n)
    old=SimpleNamespace(**vars(n.objects[4]))
    old.missing=True
    old.payload=json.dumps({'source_id':1,'object_id':4,'frame_id':1,'ts_us':33333,
        'pose_cache_reused':True,'pose_cache_age_frames':1,'keypoints_abs':[[31.,41.,.9]]*17})
    run_frame(n,2)
    anchor=_fallback_anchor(n);status={}
    points=anchor._extract_pose_keypoints_for_anchor(old,[30.,40.,32.5,75.],
        current_cohort=(1,4,1,33_333_333),current_cohort_status=status,native_frame_context=context)
    assert points is None and status['current'] is False
