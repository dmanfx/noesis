"""NEW actual producer/service/World path; ALL numerical evidence is SYNTHETIC."""
from dataclasses import replace
from types import SimpleNamespace
import time
from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest

from noesis.pipelines.projective_continuity import (
    ProjectiveQualityBridge,projective_continuity_enabled,runtime_projective_policy,sample_from_quality,
)
from noesis.pipelines.world_quality_adapter import capture_body_support,capture_quality_input,freeze_quality_work
from noesis.pipelines.source_acquisition import OriginalSwinExtraction,OriginalSwinCache
from noesis.identity_association_service import IdentityAssociationService
from noesis_core.contracts.base import Vector3
from noesis_core.contracts.world_measurement import WorldMeasurementHypothesis,ResolvedGroundMeasurement
from reid.mv3dt_person_association import ProjectiveCameraBinding,ProjectiveContinuityPolicy


@pytest.mark.parametrize('enabled',[False,True])
def test_target_group_support_is_filled_by_actual_final_membership_issuance(i138_image_native,monkeypatch,enabled):
    import json
    from noesis.pipelines.projective_continuity import _OriginalBodyDiagnostic
    n=i138_image_native;ledger=n.f.identity.ledger
    n.bridge._body_diagnostic=_OriginalBodyDiagnostic((217/30.,) if enabled else ())
    issued=[];actual=n.f.identity.spatial_memberships
    def issuance(**kwargs):
        pending=tuple(ledger._image_support_targets)
        result=actual(**kwargs)
        if pending:
            assert all(r['stage']=='actual_post_join_membership_check' for _,r,*_ in pending)
            issued.extend(r['reason'] for _,r,*_ in pending)
        return result
    monkeypatch.setattr(n.f.identity,'spatial_memberships',issuance)
    for frame in (216,217):
        n.now[0]=100.+(frame-218)/30.
        for source in (2,1):
            packet,tracks=n.packet(source,frame);publish_lineage(n,packet,tracks)
    body=n.bridge.diagnostics()['original_body_inspection_only']
    if enabled:
        rows=body['target_admission']['records']
        support=[r['appearance']['group_support'] for r in rows]
        assert issued and all(v is None for v in issued)
        records=[v for s in support for v in s['records']]
        assert records and all(r['cause']=='accepted' and len(r['members'])==2 for r in records)
        assert all(len(json.dumps(s).encode())<=8192 and len(s['records'])<=8 for s in support)
        records[0]['cause']='external mutation'
        assert 'external mutation' not in str(n.bridge.diagnostics())
    else:
        assert not issued and ledger._image_support_targets==()
        assert 'target_admission' not in body
    assert len(n.f.world.current_snapshot().entities)==1
    assert n.f.identity.stats().pending_reservations==0
    # A later ordinary bridge/native step cannot keep the prior target armed.
    n.now[0]+=.2
    packet,tracks=n.packet(2,224);publish_lineage(n,packet,tracks)
    assert ledger._image_support_targets==()


def test_target_group_support_exact_guard_and_bounds_without_classifier_reentry(i138_image_native,monkeypatch):
    import json
    n=i138_image_native;ledger=n.f.identity.ledger
    for source in (2,1):
        packet,tracks=n.packet(source,216);publish_lineage(n,packet,tracks)
    dto,=n.f.identity.spatial_memberships(at_interval=n.f.ingress._context)
    own=next(m.observation for m in dto.members if m.observation.member.source_id==2)
    peer=next(m.observation.member for m in dto.members if m.observation.member.source_id==1)
    # Diagnostic-only fault injection: missing peer lifecycle must STILL deny;
    # this is not a claimed public/native reproduction or authority fixture.
    ledger._members[peer]=replace(ledger._members[peer],spatial_lifecycle=None)
    expected=n.f.identity.spatial_memberships(at_interval=n.f.ingress._context)
    reason=next(e.reason for d in expected for e in d.eligibility if e.observation==own)
    assert reason=='image_region_current_support_required'
    calls=[];classification=ledger._image_region_support_error
    def counted(*args):calls.append(args[0]);return classification(*args)
    monkeypatch.setattr(ledger,'_image_region_support_error',counted)
    before=ledger.authority_revision
    sink=ledger.arm_image_region_support_diagnostic((own,)*10)
    assert not calls and ledger.authority_revision==before
    actual=n.f.identity.spatial_memberships(at_interval=n.f.ingress._context)
    assert actual==expected and calls.count(own)==1
    row=sink['records'][0]
    assert row['reason']==reason and row['cause']=='group_member_support' and row['blocking_source']==1
    assert next(m for m in row['members'] if m['stamp'][0]==1)['native'] is None
    assert len(sink['records'])==8 and sink['omitted_records']==2
    # Honest whole-record omission, never truncated/invented guard locations.
    sink['records'][-1]['padding']='x'*9000
    ledger._image_support_targets[0][1]['stage']='unobserved_stage'
    n.f.identity.spatial_memberships(at_interval=n.f.ingress._context)
    assert len(json.dumps(sink).encode())<=8192 and sink['omitted_records']>=3
    ledger.reset_source(replace(n.f.sources[0],source_epoch=1))
    assert ledger._image_support_targets==()


def test_target_group_support_observer_failure_does_not_change_actual_publication(i138_image_native,monkeypatch):
    from noesis.pipelines.projective_continuity import _OriginalBodyDiagnostic
    n=i138_image_native
    n.bridge._body_diagnostic=_OriginalBodyDiagnostic((217/30.,))
    monkeypatch.setattr(n.f.identity.ledger,'_image_support_members',
        lambda *_:(_ for _ in ()).throw(ValueError('optional diagnostic failed')))
    for frame in (216,217):
        n.now[0]=100.+(frame-218)/30.
        for source in (2,1):
            packet,tracks=n.packet(source,frame);publish_lineage(n,packet,tracks)
    support=[r['appearance']['group_support'] for r in
        n.bridge.diagnostics()['original_body_inspection_only']['target_admission']['records']]
    assert any(s['errors'] for s in support)
    assert len(n.f.world.current_snapshot().entities)==1
    assert n.f.identity.stats().pending_reservations==0


@pytest.mark.parametrize('enabled',[False,True])
def test_r171_actual_bridge_target_native_decisions_are_optional(i138_image_native,enabled):
    from noesis.pipelines.projective_continuity import _OriginalBodyDiagnostic
    n=i138_image_native
    n.bridge._body_diagnostic=_OriginalBodyDiagnostic((216/30.,217/30.) if enabled else ())
    for frame in (216,217):
        n.now[0]=100.+(frame-218)/30.
        for source in (2,1):
            packet,tracks=n.packet(source,frame,terminated=(4,) if source==2 and frame==217 else ())
            publish_lineage(n,packet,tracks)
    summary=n.bridge.evidence_summary() if hasattr(n.bridge,'evidence_summary') else n.bridge.diagnostics()
    body=summary['original_body_inspection_only']
    if not enabled:
        assert 'target_admission' not in body
    else:
        rows=body['target_admission']['records']
        assert rows and len(rows)<=8
        family=next(r for r in rows if r['source']==2 and r['frame']==217)
        own=next(m for m in family['native_members'] if m['raw']==1)
        assert own['active'] and own['retained'] and own['capsule_included'] and own['participant']
        assert own['birth'] is not None and not own['shadow']
        assert family['native_terminated']==[4] and not any(m['raw']==4 for m in family['native_members'])
        assert family['native_members_omitted']==family['native_active_omitted']==0
        assert family['native_shadow_omitted']==family['native_terminated_omitted']==0
        assert family['withdrawn_omitted']==family['retired_omitted']==0
        assert family['owner_frame']==217 and family['owner_epoch']==0
        assert not family['withdrawn'] and not family['retired']
        assert len(__import__('json').dumps(body['target_admission']).encode())<=55*1024
    assert len(n.f.world.current_snapshot().entities)==1


def test_r171_target_native_summary_detaches_and_enforces_final_bound():
    import json
    from noesis.pipelines.projective_continuity import _OriginalBodyDiagnostic
    d=_OriginalBodyDiagnostic((38.4,38.433333333))
    d.admission_records={(source,index):{'source':source,'frame':1152+index,'native_members':[
        {'raw':raw,'generation':raw+1,'marker':'x'*200} for raw in range(64)]}
        for source in range(3) for index in range(2)}
    result=d.admission_summary()
    assert len(result['records'])<=8 and result['omitted_records']>0
    assert len(json.dumps(result).encode())<=55*1024
    result['records'].clear()
    assert d.admission_records


def test_r171_optional_target_observer_failure_cannot_change_actual_join(i138_image_native,monkeypatch):
    n=i138_image_native
    monkeypatch.setattr(n.bridge._body_diagnostic,'observe_admission',
        lambda *_:(_ for _ in ()).throw(RuntimeError('optional diagnostic failed')))
    for frame in (216,217):
        n.now[0]=100.+(frame-218)/30.
        for source in (2,1):
            packet,tracks=n.packet(source,frame);publish_lineage(n,packet,tracks)
    assert len(n.f.world.current_snapshot().entities)==1
    assert n.f.identity.stats().pending_reservations==0
_path=Path(__file__).with_name('test_xc3d_world_quality_adapter.py')
_spec=importlib.util.spec_from_file_location('projective_original_quality_examples',_path)
_examples=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=_examples
_spec.loader.exec_module(_examples)  # Helpers only; no prior test is executed.
NativeQualityFixture=_examples.NativeQualityFixture
actual_lineage_callback=_examples.actual_lineage_callback
lineage_packet=_examples.lineage_packet
publish_lineage=_examples.publish_lineage
world_examples=_examples.world_examples


def test_i142_bridge_shadow_capsule_keeps_original_ingress_and_current_owner(tmp_path,monkeypatch):
    f,monitor,old,final,proof=_examples.i142_crowded_shadow_fixture(tmp_path,monkeypatch)
    try:
        shadow=next(c for c in proof.owner_availability if c.observation.member.source_id==2
            and c.observation.member.raw_id==9)
        assert shadow.observation not in proof.cohort
        assert shadow.observation.frame_id==1 and shadow.source_frame_id==2
        assert shadow.source_ingress_monotonic_s<shadow.owner_ingress_monotonic_s
        assert shadow.availability_checked_monotonic_s==shadow.owner_ingress_monotonic_s
        assert shadow.observation.member in shadow.shadow_members
        assert not any(o.stamp==shadow.observation for o in proof.originals)
        assert len(proof.owner_availability)<=64
        assert all(len(d.members)==1 for d in f.identity.spatial_memberships(at_interval=f.ingress._context))
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('barrier',['missing_competitor','foreign_original','mixed_profile','duplicate','stale'])
def test_i135_bridge_exact_current_feature_controls(tmp_path,monkeypatch,barrier):
    import json
    f,packet,monitor=_examples.joint_appearance_native_fixture(tmp_path,monkeypatch)
    try:
        f.publish([packet(2,1)])
        kitchen=packet(1,2);work,tracks=packet(2,2)
        values=work.swin_extractions
        if barrier=='missing_competitor':values=values[:1]
        elif barrier=='foreign_original':values=(replace(values[0],original=replace(values[0].original,original_frame=1)),)+values[1:]
        elif barrier=='mixed_profile':values=(replace(values[0],profile=replace(values[0].profile,fingerprint='foreign')),)+values[1:]
        elif barrier=='duplicate':values=values+(values[0],)
        elif barrier=='stale':values=(replace(values[0],ingress_monotonic_s=99.),)+values[1:]
        f.publish([kitchen,(replace(work,swin_extractions=values),tracks)])
        assert len(f.identity.spatial_memberships(at_interval=f.ingress._context))==3
        summary=f.ingress.projective_bridge._body_diagnostic.admission_summary()
        assert all(not r['appearance']['complete'] for r in summary['records'])
        assert len(json.dumps(summary).encode())<=55*1024
        changed=summary['records'][0]['appearance'];changed['participants'][0]['reason']='caller changed'
        assert 'caller changed' not in str(f.ingress.projective_bridge._body_diagnostic.admission_summary())
    finally:
        assert monitor.stop()['quiesced']
        f.world.close()


@pytest.fixture
def i138_image_native(tmp_path,monkeypatch):
    """Real caller/owner/World; SYNTHETIC source leases and occupied geometry."""
    from collections import deque
    from noesis.pipelines.projective_continuity import ImageRegionQualityBridge
    from noesis.pipelines.source_acquisition import SelectedSwinIngress,SourceAcquisitionState,OwnerSourceAvailability
    from noesis_core.runtime_publication import RuntimePublicationGate
    frame=[216];now=[99.8];base=time.time_ns();selected={}
    monkeypatch.setattr(time,'time_ns',lambda:base+frame[0]*33_333_333)
    monkeypatch.setattr(time,'monotonic',lambda:now[0])
    def factory(run,spatial):
        examples=world_examples();p=((100.,0.,0.,0.),(0.,100.,0.,0.),(0.,0.,1.,0.))
        policy=ProjectiveContinuityPolicy(tuple(ProjectiveCameraBinding(b,'synthetic-image-prior',
            'raster:'+b.camera_id,(1920,1080),'backend_world_m',
            examples.frame_binding(b.source_id).target_frame.revision,
            examples.frame_binding(b.source_id).source_camera_calibration_sha256,p,
            ((100.,0.,0.,-10. if b.source_id==1 else 10.),p[1],p[2])) for b in spatial.sources))
        selected['policy']=policy
        return IdentityAssociationService.for_estimated_image_regions(run,spatial,policy,
            live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    f=NativeQualityFixture(tmp_path,identity_factory=factory)
    pipeline=SimpleNamespace(xc3d_world_quality_ingress=f.ingress,
        xc3d_private_owner_state=SourceAcquisitionState(f.identity.run_id,'synthetic-quality-graph'),
        xc3d_private_owner_diagnostics=deque(maxlen=256))
    pipeline.xc3d_swin_ingress=SelectedSwinIngress(f.identity.run_id,'synthetic-quality-graph',
        f.profile,None,None,timestamp_origin='source_pts')
    f.ingress.origin_bindings=actual_lineage_callback(pipeline)
    bridge=ImageRegionQualityBridge(selected['policy'],lambda:pipeline.xc3d_swin_ingress.cache)
    f.ingress.projective_bridge=bridge
    processor=SimpleNamespace(pipeline=pipeline,tracking_pub=f.publisher,publication_gate=RuntimePublicationGate(),
        _stamp_footpoint_lifecycles=lambda *_:None,_commit_enqueued_world_output_watermarks=lambda *_a,**_k:None,
        _last_tracking_publish_ts_by_sensor={},_last_tracking_count_by_sensor={},bev_renderer=None)
    n=SimpleNamespace(f=f,pipeline=pipeline,processor=processor,bridge=bridge,frame=frame,now=now,proofs=[])
    join=f.identity.join_estimated_image_regions
    def captured(proof,**kwargs):
        n.proofs.append(proof)
        return join(proof,**kwargs)
    monkeypatch.setattr(f.identity,'join_estimated_image_regions',captured)
    def packet(source,number,*,terminated=(),raw=None):
        frame[0]=number;raw=raw if raw is not None else (6 if source==1 else 1)
        update=f.registries[source].update_frame
        with monkeypatch.context() as patch:
            patch.setattr(f.registries[source],'update_frame',lambda **kwargs:
                update(**{**kwargs,'terminated_tracker_ids':terminated}))
            work,tracks=lineage_packet(n,source,number,raws=(raw,),x=1.+source*.001-(raw-9)*3.,
                scale=4. if source==1 else 1.)
        inputs=[]
        for row,track in zip(work.rows,tracks):
            u=50. if source==1 else 40.;track['bbox']=[u-10.,20.,20.,40.]
            points=[(u,30.,.9)]*17
            points[5],points[6]=(u-2.,30.,.9),(u+2.,30.,.9)
            points[11],points[12]=(u-2.,42.,.9),(u+2.,42.,.9)
            inputs.append(replace(row.evidence,body=capture_body_support(track,row.evidence.lifecycle,
                points,pose_is_current=True)))
        receipt=work.private.owner_retention;private=work.private
        work=replace(freeze_quality_work(private,work.continuity,tuple(inputs)),ingress_monotonic_s=now[0])
        return replace(work,owner_availability=OwnerSourceAvailability(private.original,receipt,
            now[0],now[0],0.,20.,'running')),tracks
    n.packet=packet
    try:yield n
    finally:f.world.close()


def test_i138_unrelated_termination_preserves_current_capsule_owner_and_one_dot(i138_image_native):
    n=i138_image_native
    for frame in (216,217,218):
        n.now[0]=100.+(frame-218)/30.
        for source in (2,1):
            packet,tracks=n.packet(source,frame);snapshot=publish_lineage(n,packet,tracks)
    old,=snapshot.entities
    assert next(s for s in old.sources if s.accepted).camera_id=='family-room'
    n.now[0]=100.079387584
    packet,tracks=n.packet(2,220,terminated=(4,))
    # Existing original K218 is stale for a new visual proof, but its genuine
    # native/source lease remains intact. F220 itself stays within .35s.
    n.now[0]=100.415628009
    final=publish_lineage(n,packet,tracks)
    proof=n.proofs[-1]
    assert len(final.entities)==1, {'capsule_sources':
        {c.observation.member.source_id for c in proof.owner_availability},'dot_count':len(final.entities)}
    assert {c.observation.member.source_id for c in proof.owner_availability}=={1,2}
    family=next(c for c in proof.owner_availability if c.observation.member.source_id==2)
    assert family.observation.member.raw_id==1 and family.observation.frame_id==220
    assert not family.terminated_members and family.observation.member in family.active_members
    current=next(e for d in n.f.identity.spatial_memberships(at_interval=n.f.ingress._context)
        for e in d.eligibility if e.observation.member.source_id==2)
    assert current.eligible
    entity,=final.entities
    assert entity.entity_id==old.entity_id
    assert any(s.accepted and s.observed_at_us==packet.continuity.observed_at_us for s in entity.sources)
    actual=n.f.examples.actual_node_wire(n.f,n.f.wire,'i138-unrelated-native-termination')
    assert [len(s['people']) for s in actual['snapshots']]==[1]*7


@pytest.mark.parametrize('fault',['affected-id','same-id-rebirth','source-epoch','older-owner',
    'missing-owner','nonretained','shadow','expired-source'])
def test_i138_unrelated_fix_never_waives_affected_or_invalid_owner(i138_image_native,fault):
    from noesis_core.world_service import WorldPublicationRefused
    n=i138_image_native
    for frame in (216,217,218):
        n.now[0]=100.+(frame-218)/30.
        for source in (2,1):
            packet,tracks=n.packet(source,frame);publish_lineage(n,packet,tracks)
    n.now[0]=100.079387584
    if fault=='affected-id':
        before=n.f.world.current_snapshot();proof_count=len(n.proofs)
        with pytest.raises(ValueError,match='terminated tracker is also an active object'):
            n.packet(2,220,terminated=(1,))
        assert n.f.world.current_snapshot() is before and len(n.proofs)==proof_count
        return  # Actual native lifecycle rejects same-cohort resurrection first.
    packet,tracks=n.packet(2,220,terminated=(4,))
    frozen=packet.owner_availability
    if fault=='same-id-rebirth':
        # Raw-only termination must also win over a purported newer generation.
        owner=replace(frozen.owner,terminated=(1,),keys=((2,0,1,2),),
            active=((1,2,220,packet.continuity.observed_at_us,packet.continuity.media_pts_ns),))
        packet=replace(packet,owner_availability=replace(frozen,owner=owner))
    elif fault=='source-epoch':
        packet=replace(packet,owner_availability=replace(frozen,owner=replace(frozen.owner,source_epoch=1)))
    elif fault=='older-owner':
        packet=replace(packet,owner_availability=replace(frozen,owner=replace(frozen.owner,frame_id=219)))
    elif fault=='missing-owner':packet=replace(packet,owner_availability=None)
    elif fault=='nonretained':
        packet=replace(packet,owner_availability=replace(frozen,owner=replace(frozen.owner,keys=(),active=())))
    elif fault=='shadow':
        raw,generation,*_=frozen.owner.active[0]
        packet=replace(packet,owner_availability=replace(frozen,owner=replace(frozen.owner,shadow=((raw,generation),))))
    elif fault=='expired-source':
        packet=replace(packet,owner_availability=replace(frozen,last_progress_age_s=19.99))
    n.now[0]=100.415628009
    if fault in {'same-id-rebirth','source-epoch','older-owner','nonretained','shadow'}:
        before=n.f.world.current_snapshot();proof_count=len(n.proofs)
        with pytest.raises(WorldPublicationRefused,match='world_quality_owner_availability_binding'):
            publish_lineage(n,packet,tracks)
        assert n.f.world.current_snapshot() is before and len(n.proofs)==proof_count
        return  # Earlier exact owner binding checks remain mandatory.
    final=publish_lineage(n,packet,tracks)
    caps=n.proofs[-1].owner_availability
    if fault in {'same-id-rebirth','source-epoch','older-owner','missing-owner'}:
        assert all(c.observation.member.source_id!=2 for c in caps)
    current=next(e for d in n.f.identity.spatial_memberships(at_interval=n.f.ingress._context)
        for e in d.eligibility if e.observation.member.source_id==2)
    assert not current.eligible
    assert not any(s.accepted and s.observed_at_us==packet.continuity.observed_at_us
        for e in final.entities for s in e.sources)


def synthetic_policy(spatial):
    # Literal synthetic pinhole; calibrated-source SHA and source-frame revision
    # deliberately differ, as the actual product contracts require.
    projection=((1000.,0.,0.,0.),(0.,0.,0.,80.),(0.,0.,1.,0.))
    examples=world_examples()
    return ProjectiveContinuityPolicy(tuple(ProjectiveCameraBinding(b,'synthetic-review-prior',
        'synthetic-raster:'+b.camera_id,(1920,1080),'backend_world_m',
        examples.frame_binding(b.source_id).target_frame.revision,
        examples.frame_binding(b.source_id).source_camera_calibration_sha256,
        projection,projection) for b in spatial.sources if b.source_id in (1,2)))


@pytest.fixture
def projective_native(tmp_path,monkeypatch):
    from collections import deque
    from noesis.pipelines.source_acquisition import SelectedSwinIngress,SourceAcquisitionState
    from noesis_core.runtime_publication import RuntimePublicationGate
    frame=[1]
    ns=time.time_ns()
    real_monotonic=time.monotonic
    monkeypatch.setattr(time,'time_ns',lambda:ns+frame[0]*33_333_333)
    monkeypatch.setattr(time,'monotonic',lambda:100.+frame[0]/30.)
    def factory(run,spatial):
        return IdentityAssociationService.for_projective_continuity(run,spatial,synthetic_policy(spatial))
    f=NativeQualityFixture(tmp_path,identity_factory=factory)
    pipeline=SimpleNamespace(xc3d_world_quality_ingress=f.ingress,
        xc3d_private_owner_state=SourceAcquisitionState(f.identity.run_id,'synthetic-quality-graph'),
        xc3d_private_owner_diagnostics=deque(maxlen=256))
    pipeline.xc3d_swin_ingress=SelectedSwinIngress(f.identity.run_id,'synthetic-quality-graph',
        f.profile,None,None,timestamp_origin='source_pts')
    f.ingress.origin_bindings=actual_lineage_callback(pipeline)
    bridge=ProjectiveQualityBridge(f.identity._projective_policy,lambda:pipeline.xc3d_swin_ingress.cache,
        joint_threshold=.35)
    f.ingress.projective_bridge=bridge
    processor=SimpleNamespace(pipeline=pipeline,tracking_pub=f.publisher,publication_gate=RuntimePublicationGate(),
        _stamp_footpoint_lifecycles=lambda *_:None,_commit_enqueued_world_output_watermarks=lambda *_a,**_k:None,
        _last_tracking_publish_ts_by_sensor={},_last_tracking_count_by_sensor={},bev_renderer=None)
    native=SimpleNamespace(f=f,pipeline=pipeline,callback=f.ingress.origin_bindings,processor=processor,
        bridge=bridge,frame=frame,real_monotonic=real_monotonic)
    try:yield native
    finally:f.world.close()


def physical_packet(native,source,frame,*,pixel_offset=0.,missing_body=False,missing_radius=False,
                    stationary=False,stationary_from_start=False,quality=1.,empty=False):
    native.frame[0]=frame
    x=.04+.03*(3 if stationary_from_start or stationary and frame>3 else frame)
    packet,tracks=lineage_packet(native,source,frame,x=x+(.001 if source==2 else -.001),scale=quality,empty=empty)
    rows=[]
    for row,track in zip(packet.rows,tracks):
        u=500*x+pixel_offset
        track.update(bbox=[u-5.,20.,10.,40.],image_size=[1920,1080],confidence=.9,tracker_confidence=.9)
        joints=[[u,30.,.9] for _ in range(17)]
        joints[15]=joints[16]=[u,40.,.9]
        body=None if missing_body else capture_body_support(track,row.evidence.lifecycle,joints,pose_is_current=True)
        h=row.evidence.measurements.hypotheses[-1]
        pose=WorldMeasurementHypothesis.model_validate({**h.model_dump(),'candidate_id':'floor_ray',
            'kind':'floor_ray','position':Vector3(x=x,y=0.,z=2.),'posture':'standing','support_state':'floor',
            'anchor':'pose_ankle_floor','pixel_uncertainty_px':None if missing_radius else 1.})
        measurements=row.evidence.measurements.model_copy(update={'hypotheses':
            row.evidence.measurements.hypotheses[:-1]+(pose,)})
        rows.append(replace(row.evidence,body=body,measurements=measurements))
        cache=native.pipeline.xc3d_swin_ingress.cache
        assert cache.record(OriginalSwinExtraction(packet.private.original,track['tracker_id'],
            native.f.profile,(1.,)+(0.,)*255,time.monotonic()))
    result=freeze_quality_work(packet.private,packet.continuity,tuple(rows))
    return replace(result,ingress_monotonic_s=time.monotonic()),tracks


def publish_suppressed_swin(native,packet,tracks,history):
    from noesis.pipelines import hooks
    c=packet.continuity
    native.processor._tracking_lifecycle=native.f.registries[c.source_id]
    work=hooks._TrackingPublicationWork(source_id=c.source_id,camera_id=c.camera_id,
        frame_id=c.frame_id,observed_at_us=c.observed_at_us,now_ts=time.time(),
        timestamp_us=packet.private.media_pts_ns//1000,tracks=tracks,footpoints=[hooks.Footpoint(
            tracker_id=t['tracker_id'],tracker_lifecycle_generation=t['tracker_lifecycle_generation'],
            frame_id=c.frame_id,world_x=t['world'][0],world_y=t['world'][1],world_z=t['world'][2]) for t in tracks],
        temporal_contract={'media_pts_ns':c.media_pts_ns},continuity=c,source_epoch=c.source_epoch,
        xc3d_private_evidence=packet.private,world_quality_work=packet,historical_quality_work=history)
    hooks._AnalyticsTelemetryProcessor._publish_tracking_work(native.processor,work)


def test_suppressed_fresh_swin_F_to_successful_N_actual_FIFO(projective_native,monkeypatch):
    from noesis.pipelines.world_quality_adapter import OriginalQualityRetention,freeze_swin_work
    for name in ('pyservicemaker','pyds','noesis_pose_meta_ext','noesis_analytics_meta_ext',
        'noesis_reid_meta_ext','noesis_v3dt_meta_ext','noesis_depth_meta_ext','noesis_depth_tracking_tensor_ext'):
        monkeypatch.setitem(sys.modules,name,None)
    n=projective_native
    birth,tracks=physical_packet(n,1,1)
    publish_lineage(n,birth,tracks)
    fresh,tracks=physical_packet(n,1,2)
    retention=OriginalQualityRetention(include_barriers=True)
    fresh=freeze_swin_work(fresh,n.pipeline.xc3d_swin_ingress)
    retention.retain(fresh)  # Actual F is suppressed; canonical owner remains birth1.
    current,tracks=physical_packet(n,1,3)
    cache=n.pipeline.xc3d_swin_ingress.cache
    cache.pending[1].pop((current.private.original.buffer_scope,current.private.original.slot,9))
    publish_suppressed_swin(n,current,tracks,retention.take())
    original=next(iter(n.f.ingress.originals.values()))
    from reid.mv3dt_person_association import AcquisitionTime
    use=cache.use(original,AcquisitionTime(original.stamp.acquired_at_s,original.stamp.clock_domain,original.stamp.clock_basis))
    assert use is not None and use.appearance.stamp.frame_id==2
    assert use.appearance.acquired_at_s==fresh.private.original.pts_ns/1e9
    assert use.appearance_ingress_monotonic_s==fresh.swin_extractions[0].ingress_monotonic_s
    assert cache.producer_counts['source_1.bound_original_F']==1
    assert n.f.world.current_snapshot().entities[0].observed_at_us==current.continuity.observed_at_us


def test_latest_swin_pair_diagnostics_bound_and_do_not_change_authority(projective_native):
    n=projective_native
    for frame in (1,2):
        for source in (1,2):
            publish_lineage(n,*physical_packet(n,source,frame))
    ingress=n.f.ingress
    cache=n.pipeline.xc3d_swin_ingress.cache
    assert len(ingress.swin_pair_examples)==1
    sample,=ingress.swin_pair_examples.values()
    # Production diagnosis runs before sole post-publication owner binding;
    # preserve that actual older F instead of claiming the fresh N is bound.
    assert sample['current_frames']==(2,2) and sample['original_frames']==(2,1)
    ingress._diagnose_swin_pairs(cache)  # same pair after genuine owner binding
    sample,=ingress.swin_pair_examples.values()
    assert sample['current_frames']==(2,2) and sample['original_frames']==(2,2)
    assert sample['current_source_s']==sample['original_source_s']
    assert sample['current_prequeue_monotonic_s']==sample['original_prequeue_monotonic_s']
    assert sample['availability_ages_s']==(0.,0.) and sample['cosine']==1.
    frozen=dict(cache.counters)
    original=cache.members[1].pop((0,9,1))
    ingress._diagnose_swin_pairs(cache)
    missing,=ingress.swin_pair_examples.values()
    assert missing['original_appearance_present'] is False and 'cosine' not in missing
    cache.members[1][(0,9,1)]=original
    ingress._diagnose_swin_pairs(cache)
    assert dict(cache.counters)==frozen
    left=next(o for o in ingress.originals.values() if o.stamp.member.source_id==1)
    right=next(o for o in ingress.originals.values() if o.stamp.member.source_id==2)
    for frame in range(3,37):
        # Synthetic immutable keys test diagnostic eviction only, not owner
        # birth/retirement policy or publication of a fabricated observation.
        member=replace(left.stamp.member,raw_id=frame+100,lifecycle_generation=frame)
        observation=replace(left,stamp=replace(left.stamp,member=member))
        ingress.originals={member:observation,right.stamp.member:right}
        ingress._diagnose_swin_pairs(cache)
        assert len(ingress.swin_pair_examples)<=32
    samples=tuple(ingress.swin_pair_examples.values())
    assert len(samples)==32
    assert samples[-1]['members'][0][2]==136 and samples[-1]['current_frames']==(2,2)
    assert all(sample['members'][0][2]!=9 for sample in samples)
    assert not any('vector' in key for sample in samples for key in sample)


@pytest.mark.parametrize('fault',['inactive','termination','reorder','wrong_buffer','expired','gap','current_fresh'])
def test_suppressed_swin_extent_refusals_and_current_precedence(projective_native,fault,monkeypatch):
    from noesis.pipelines.world_quality_adapter import freeze_swin_work
    n=projective_native
    for name in ('pyservicemaker','pyds','noesis_pose_meta_ext','noesis_analytics_meta_ext',
        'noesis_reid_meta_ext','noesis_v3dt_meta_ext','noesis_depth_meta_ext','noesis_depth_tracking_tensor_ext'):
        monkeypatch.setitem(sys.modules,name,None)
    birth,tracks=physical_packet(n,1,1)
    publish_lineage(n,birth,tracks)
    fresh,tracks=physical_packet(n,1,2)
    fresh=freeze_swin_work(fresh,n.pipeline.xc3d_swin_ingress)
    history=[fresh]
    barrier,tracks=physical_packet(n,1,3)
    if fault=='inactive':
        receipt=barrier.private.owner_retention
        barrier=replace(barrier,private=replace(barrier.private,owner_retention=replace(receipt,shadow=((9,1),))))
    elif fault=='termination':
        receipt=barrier.private.owner_retention
        barrier=replace(barrier,private=replace(barrier.private,owner_retention=replace(receipt,terminated=(9,))))
    elif fault=='reorder':fresh=replace(fresh,ingress_monotonic_s=101.);history=[fresh]
    elif fault=='wrong_buffer':
        e=fresh.swin_extractions[0]
        fresh=replace(fresh,swin_extractions=(replace(e,original=replace(e.original,buffer_scope=999)),));history=[fresh]
    elif fault=='expired':
        e=fresh.swin_extractions[0]
        fresh=replace(fresh,swin_extractions=(replace(e,ingress_monotonic_s=90.),));history=[fresh]
    if fault!='gap':history.append(barrier)
    current,tracks=physical_packet(n,1,4)
    cache=n.pipeline.xc3d_swin_ingress.cache
    if fault!='current_fresh':cache.pending[1].pop((current.private.original.buffer_scope,current.private.original.slot,9))
    publish_suppressed_swin(n,current,tracks,tuple(history))
    stored=next(iter(cache.members[1].values()))
    assert stored[2]==(4 if fault=='current_fresh' else 1)
    assert n.f.world.current_snapshot().entities[0].observed_at_us==current.continuity.observed_at_us


def test_r15_five_hz_original_cache_bridge_World_Node(projective_native,monkeypatch):
    n=projective_native
    join=n.f.identity.join_projective_continuity
    joins=[]
    def diagnosed_join(proof,**kwargs):
        result=join(proof,**kwargs)
        joins.append({'result':(result.disposition,result.reason),
            'withdrawn':tuple((m.source_id,m.raw_id,m.lifecycle_generation) for m in result.withdrawn_members),
            'pairs':tuple(tuple(tuple((s.stamp.member.source_id,
            s.stamp.frame_id,s.stamp.lifecycle_started_frame,s.stamp.acquired_at_s,
            s.stamp.media_time_s,s.ingress_monotonic_s) for s in side)
            for side in (pair.left,pair.right)) for pair in proof.pairs),
            'evaluation_monotonic_s':kwargs['delivery_context'].evaluation_monotonic_s})
        return result
    monkeypatch.setattr(n.f.identity,'join_projective_continuity',diagnosed_join)
    for frame in (1,7,13,19):
        for source in (1,2):
            snapshot=publish_lineage(n,*physical_packet(n,source,frame))
    # Keep the real fourth-sample World assertion. The public owner guard must
    # agree with its accepted retrospective proof; do not bypass it in producer.
    current=tuple(o.stamp for o in n.f.ingress.originals.values())
    context=n.f.ingress._context
    support_errors=tuple((s.member.source_id,s.frame_id,
        n.f.identity.ledger.projective_continuity_error(s,context)) for s in current)
    assert snapshot.entities,{'joins':joins[-2:],'support_errors':support_errors,
        'counts':dict(n.bridge.counts)}
    head,=snapshot.entities
    assert n.f.identity.ledger.stats().active_persons==1
    assert n.bridge.diagnostics()['counts']['accepted']>0
    for history in n.bridge.history.values():
        assert [s.stamp.frame_id for s in history]==[1,7,13,19]
        assert history[-1].stamp.acquired_at_s-history[0].stamp.acquired_at_s==pytest.approx(.6)
        selected=n.bridge._trajectory(history[-1].stamp.member)
        assert selected is not None
        assert all(q.stamp.acquired_at_s-p.stamp.acquired_at_s<=.35 for p,q in zip(selected,selected[1:]))
    with monkeypatch.context() as node_patch:
        node_patch.setattr(time,'monotonic',n.real_monotonic)
        result=n.f.examples.actual_node_wire(n.f,n.f.wire,'r15-five-hz-originals')
    assert len(result['snapshots'][-1]['people'])==1


@pytest.mark.parametrize('fault',['gap','missing_original','old_appearance','absent_return','expired_peer'])
def test_r15_five_hz_actual_bridge_barriers(projective_native,fault):
    n=projective_native
    for frame in ((1,7,19) if fault=='gap' else (1,7,13)):
        if fault=='absent_return' and frame==13:
            publish_lineage(n,*physical_packet(n,2,10,empty=True))
        for source in (1,2):
            if fault=='expired_peer' and source==2 and frame!=1:
                continue
            packet,tracks=physical_packet(n,source,frame,missing_body=fault=='missing_original' and frame==7)
            if fault=='old_appearance' and frame!=1:
                cache=n.pipeline.xc3d_swin_ingress.cache
                cache.pending[source].pop((packet.private.original.buffer_scope,packet.private.original.slot,9))
            if fault=='absent_return' and source==2 and frame==13:
                from noesis_core.world_service import WorldPublicationRefused
                with pytest.raises(WorldPublicationRefused,match='lifecycle_frontier'):
                    publish_lineage(n,packet,tracks)
            else:
                publish_lineage(n,packet,tracks)
    assert not n.bridge.counts.get('accepted',0)
    stats=n.f.identity.ledger.stats()
    assert stats.closed_members==1 if fault=='absent_return' else stats.active_persons>=2


@pytest.mark.parametrize('fault',['source_gap','media_gap','arrival_gap','birth','clock','reorder','anatomy','expired_current','reset'])
def test_r15_retrospective_pruning_does_not_bridge_barriers(projective_native,fault):
    n=projective_native
    for frame in (1,7):
        publish_lineage(n,*physical_packet(n,1,frame))
    member,=n.bridge.history
    a,b=n.bridge.history[member]
    if fault=='source_gap':a=replace(a,stamp=replace(a.stamp,acquired_at_s=b.stamp.acquired_at_s-.36))
    elif fault=='media_gap':a=replace(a,stamp=replace(a.stamp,media_time_s=b.stamp.media_time_s-.36))
    elif fault=='arrival_gap':a=replace(a,ingress_monotonic_s=b.ingress_monotonic_s-.36)
    elif fault=='birth':a=replace(a,stamp=replace(a.stamp,lifecycle_started_frame=0))
    elif fault=='clock':a=replace(a,stamp=replace(a.stamp,clock_domain='foreign-clock'))
    elif fault=='reorder':a=replace(a,stamp=replace(a.stamp,frame_id=b.stamp.frame_id))
    elif fault=='anatomy':a=replace(a,support_basis='pose_torso_support')
    now=100.+13/30. if fault!='expired_current' else 101.
    current=dict(n.f.ingress.originals) if fault!='reset' else {}
    times={1:13*33_333_333/1e9}
    frozen=repr((a,b))
    kept=n.bridge._prune_originals({member:[a,b]},current,times,now,'test')
    rows=kept.get(member,[])
    assert len([s for s in rows if s.support_basis==b.support_basis])<2
    assert repr((a,b))==frozen


def test_r15_new_historical_arrival_is_not_given_retrospective_TTL(projective_native):
    n=projective_native
    old=None
    for frame in (1,7,13):
        packet,tracks=physical_packet(n,1,frame)
        if frame==1:old=packet
        publish_lineage(n,packet,tracks)
    n.bridge.history.clear()
    # Real OLD quality work arrives for the first time .4s later. The new
    # retention rule cannot admit it or retag cached depth as the endpoint.
    # Reuse the exact context already frozen by the actual ordered adapter.
    context=n.f.ingress._context
    assert n.bridge.ingest_historical_quality((old,),n.f.ingress.originals,context,time.monotonic()) is None
    assert n.bridge.history=={}


def test_projective_actual_original_bridge_to_service_World_publisher_best_view_and_handoff(projective_native,monkeypatch):
    n=projective_native
    for frame in (1,2,3):
        for source in (1,2):
            snapshot=publish_lineage(n,*physical_packet(n,source,frame,quality=4. if source==1 else 1.))
    head,=snapshot.entities
    assert len(head.sources)==2 and sum(s.accepted for s in head.sources)==1
    assert next(s.camera_id for s in head.sources if s.accepted)=='family-room'
    assert n.bridge.diagnostics()['counts']['accepted']>0
    # Stationary SAME native pair remains usable beyond the original .35s seed;
    # each fresh frame still has original body/appearance and reciprocal support.
    for frame in range(4,18):
        for source in (1,2):
            snapshot=publish_lineage(n,*physical_packet(n,source,frame,stationary=True,
                quality=1. if source==1 else 4.))
    changed,=snapshot.entities
    assert changed.entity_id==head.entity_id and len(changed.sources)==2
    assert next(s.camera_id for s in changed.sources if s.accepted)=='kitchen'
    assert changed.observed_at_us>head.observed_at_us
    # Real source termination/absence does not let a retired second head return.
    gone=publish_lineage(n,*physical_packet(n,2,18,empty=True))
    assert len(gone.entities)<=1
    assert n.bridge.diagnostics()['history_samples']<=128
    assert n.f.identity.ledger.stats().active_persons<=2
    # Exercise EVERY actual reducer input. Keep the existing128KiB output cap;
    # only omit unasserted heavy diagnostic fields from the test result JSON.
    before='const output = JSON.stringify({node:process.versions.node,diagnostics,snapshots});'
    after='const output = JSON.stringify({node:process.versions.node,diagnostics,snapshots:snapshots.map(s=>({...s,people:s.people.map(({sourceEvidence,worldCovariance,supportSurface,...p})=>p)}))});'
    assert before in n.f.examples.NODE_WIRE_PROGRAM
    monkeypatch.setattr(n.f.examples,'NODE_WIRE_PROGRAM',n.f.examples.NODE_WIRE_PROGRAM.replace(before,after))
    with monkeypatch.context() as node_patch:
        node_patch.setattr(time,'monotonic',n.real_monotonic)
        result=n.f.examples.actual_node_wire(n.f,n.f.wire,'projective-original-fifo-best-view')
    expected=[len(m['payload']['entities']) for m in n.f.wire if m['type']=='world_snapshot']
    assert [len(s['people']) for s in result['snapshots']]==expected


@pytest.mark.parametrize('fault',['missing_body','missing_radius','projective_offset','stationary_new'])
def test_projective_actual_bridge_no_independent_positive_cue_never_merges(projective_native,fault):
    n=projective_native
    for frame in (1,2,3,4):
        for source in (1,2):
            snapshot=publish_lineage(n,*physical_packet(n,source,frame,
                missing_body=fault=='missing_body',missing_radius=fault=='missing_radius',
                pixel_offset=30. if fault=='projective_offset' and source==2 else 0.,
                stationary_from_start=fault=='stationary_new'))
    assert len(snapshot.entities)==2 and n.f.identity.ledger.stats().active_persons==2
    assert not n.bridge.diagnostics()['counts'].get('accepted',0)


def test_projective_config_strict_explicit_disabled_by_default():
    assert projective_continuity_enabled({}) is False
    mode={'contract':'noesis.xc3d.projective_continuity.v1','enabled':True}
    with pytest.raises(ValueError):projective_continuity_enabled({'xc3d_projective_continuity':mode})
    assert projective_continuity_enabled({'xc3d_projective_continuity':mode,
        'xc3d_world_quality':{'contract':'noesis.xc3d.world_quality.v1','enabled':True}})
    with pytest.raises(ValueError):projective_continuity_enabled({'xc3d_projective_continuity':{**mode,'extra':1}})


@pytest.mark.parametrize('fault',['pose_scale_unknown','hidden_ankle_anchor','seated_floor','lying_floor',
    'unknown_support','unknown_posture','single_ankle_ambiguous','support_uv_unavailable','genuine_single_ankle'])
def test_projective_literal_floor_producer_semantics(projective_native,fault):
    n=projective_native
    for frame in (1,2,3,4):
        for source in (1,2):
            packet,tracks=physical_packet(n,source,frame)
            row,=packet.rows
            original=row.evidence.measurements.hypotheses[-1]
            updates={
                'pose_scale_unknown':{'kind':'pose_scale','support_state':'unknown','anchor':'upright_body_plane'},
                'hidden_ankle_anchor':{'anchor':'bbox_bottom'},
                'seated_floor':{'posture':'sitting'},
                'lying_floor':{'posture':'lying'},
                'unknown_support':{'support_state':'unknown'},
                'unknown_posture':{'posture':'unknown'},
                'single_ankle_ambiguous':{'anchor':'pose_single_ankle_floor'},
                'support_uv_unavailable':{'anchor':'pose_ankle_support'},
                'genuine_single_ankle':{'anchor':'pose_single_ankle_floor'},
            }[fault]
            hypothesis=WorldMeasurementHypothesis.model_validate({**original.model_dump(),**updates})
            measurements=row.evidence.measurements.model_copy(update={'hypotheses':
                row.evidence.measurements.hypotheses[:-1]+(hypothesis,)})
            body=row.evidence.body
            if fault=='genuine_single_ankle':
                joints=list(body.keypoints)
                joints[16]=(*joints[16][:2],0.)
                body=replace(body,keypoints=tuple(joints))
            packet=replace(packet,rows=(replace(row,evidence=replace(row.evidence,measurements=measurements,body=body)),))
            snapshot=publish_lineage(n,packet,tracks)
    if fault in {'unknown_posture','genuine_single_ankle'}:
        assert len(snapshot.entities)==1 and n.bridge.diagnostics()['counts']['accepted']>0
    else:
        assert len(snapshot.entities)==2
        assert not n.bridge.diagnostics()['counts'].get('accepted',0)
        reason=('sample_original_ankle_contact_unavailable' if fault in
            {'single_ankle_ambiguous','support_uv_unavailable'} else 'sample_original_ankle_floor_hypothesis_unavailable')
        assert n.bridge.diagnostics()['counts'][reason]>0


def test_projective_expected_unavailable_is_not_native_integrity_failure():
    from noesis_core.world.fusion import spatial_rejection_category
    assert spatial_rejection_category('projective_current_support_required')=='no_vote'
    assert spatial_rejection_category('current_visual_stale_member_delivery')=='no_vote'
    assert spatial_rejection_category('current_rejoin_proof_required')=='integrity'
    assert spatial_rejection_category('invented_projective_reason')=='integrity'


def test_projective_after_positive_missing_body_holds_honestly_without_killing_publisher(projective_native):
    n=projective_native
    for frame in (1,2,3):
        for source in (1,2):
            snapshot=publish_lineage(n,*physical_packet(n,source,frame))
    head,=snapshot.entities
    for frame in range(4,7):
        for source in (1,2):
            snapshot=publish_lineage(n,*physical_packet(n,source,frame,missing_body=True,stationary=True))
    assert n.f.wire[-1]['cohort']==n.f.wire[-2]['cohort']
    if snapshot.entities:
        assert snapshot.entities[0].observed_at_us==head.observed_at_us
    assert n.bridge.diagnostics()['counts']['unsupported_original_sample']>=6


@pytest.mark.parametrize('fault',['none','missing_body','projective_offset','stationary_new'])
def test_projective_numeric_inspection_is_independent_of_missing_appearance_and_never_joins(projective_native,monkeypatch,fault):
    n=projective_native
    # Genuine original body/measurement packets still reach the actual owner;
    # deliberately unavailable appearance MUST remain unavailable to joining.
    monkeypatch.setattr(n.pipeline.xc3d_swin_ingress.cache,'use',lambda *_:None)
    for frame in (1,2,3,4):
        for source in (1,2):
            snapshot=publish_lineage(n,*physical_packet(n,source,frame,
                missing_body=fault=='missing_body',
                pixel_offset=30. if fault=='projective_offset' and source==2 else 0.,
                stationary_from_start=fault=='stationary_new'))
    assert len(snapshot.entities)==2 and n.f.identity.ledger.stats().active_persons==2
    assert not n.bridge.counts.get('accepted',0)
    assert all(dto.association_basis=='source_local' for dto in n.f.identity.spatial_memberships(at_interval=n.f.ingress._context))
    assert n.bridge.numeric_counts['current_pair_slots']>0
    if fault=='none':
        assert n.bridge.numeric_counts['numeric_supported']>0
    elif fault=='missing_body':
        assert n.bridge.numeric_counts['current_body_endpoint_unavailable']>0
        assert not n.bridge.numeric_counts['complete_original_witnesses']
    else:
        assert n.bridge.numeric_counts['numeric_contradiction']>0
        assert n.bridge.numeric_counts['peer_support_misses' if fault=='projective_offset' else 'observed_degenerate_steps']>0
    assert len(n.bridge.numeric_examples)<=16 and n.f.identity.stats().pending_reservations==0


def test_projective_original_hypothesis_diagnostic_bypasses_only_representative_not_provenance_or_World_grants(projective_native,monkeypatch):
    n=projective_native
    monkeypatch.setattr(n.pipeline.xc3d_swin_ingress.cache,'use',lambda *_:None)
    old=None
    for frame in (1,2,3,4):
        for source in (1,2):
            packet,tracks=physical_packet(n,source,frame)
            if frame>1:
                evidence=packet.rows[0].evidence
                held=ResolvedGroundMeasurement.model_validate({**evidence.resolution.model_dump(),
                    'status':'hold','quality':'held','selected_candidate_id':None,'selected_kind':None,
                    'contributor_ids':(),'alternate_candidate_id':None,'fused':False})
                for track in tracks:
                    track.update(world=None,world_covariance=None,world_source=None,
                        world_frame=None,world_frame_revision=None,world_transform_sha256=None,
                        world_valid=False,world_quality='invalid',world_quality_reason='no_current_position',
                        world_measurement_accepted=False)
                frozen=capture_quality_input(tracks[0],evidence.lifecycle,evidence.measurements,
                    held,evidence.depth,evidence.body)
                packet=replace(freeze_quality_work(packet.private,packet.continuity,(frozen,)),
                    ingress_monotonic_s=packet.ingress_monotonic_s)
                assert packet.rows[0].denial is None and not packet.rows[0].current_measured
            snapshot=publish_lineage(n,packet,tracks)
        if frame==1:old={e.entity_id:e.observed_at_us for e in snapshot.entities}
        else:assert {e.entity_id:e.observed_at_us for e in snapshot.entities}==old
    assert not n.bridge.counts.get('accepted',0)
    # Original visible-floor evidence now reaches the SAME numeric gate even
    # when selected World is held/absent; missing appearance still cannot join.
    assert n.bridge.numeric_counts['numeric_supported']>0
    assert n.bridge.counts['original_floor_independent_of_selected_World']==6
    assert n.bridge.hypothesis_numeric_counts['numeric_supported']>0
    assert n.bridge.hypothesis_counts['kitchen:solely_representative_gate']==3
    assert n.bridge.hypothesis_counts['family-room:solely_representative_gate']==3
    assert sum(map(len,n.bridge.hypothesis_history.values()))<=128
    assert n.f.identity.stats().pending_reservations==0


def test_projective_retained_best_actual_FIFO_never_renews_old_point_or_grant(projective_native,monkeypatch):
    n=projective_native
    for frame in (1,2,3):
        for source in (1,2):
            snapshot=publish_lineage(n,*physical_packet(n,source,frame,quality=4. if source==1 else 1.))
    old,=snapshot.entities
    calls=[]
    prepare=n.f.identity.prepare_spatial_member
    def record(*a,**kw):
        calls.append(1)
        return prepare(*a,**kw)
    monkeypatch.setattr(n.f.identity,'prepare_spatial_member',record)
    snapshot=publish_lineage(n,*physical_packet(n,1,4,quality=4.,stationary=True))
    retained,=snapshot.entities
    assert retained.observed_at_us==old.observed_at_us and retained.position==old.position
    assert next(s.camera_id for s in retained.sources if s.accepted)=='family-room'
    assert calls==[]
    # SAME selected camera's next valid original advances even if noisier than
    # its own old covariance; it is not locked behind an older estimate.
    snapshot=publish_lineage(n,*physical_packet(n,2,4,quality=2.,stationary=True))
    updated,=snapshot.entities
    assert updated.observed_at_us>retained.observed_at_us
    # Fresh selection reserves once, then the existing commit preflight
    # revalidates that reservation against the latest genuine owner context.
    assert calls==[1,1]
    with monkeypatch.context() as node_patch:
        node_patch.setattr(time,'monotonic',n.real_monotonic)
        result=n.f.examples.actual_node_wire(n.f,n.f.wire,'projective-retained-grant-no-renewal')
    a,b=result['snapshots'][5:7]
    assert b['people'][0]['observedAtUs']==a['people'][0]['observedAtUs']
    assert b['people'][0]['trail']==a['people'][0]['trail']


def test_projective_frozen_peer_clock_actual_FIFO_detaches_estimate_not_live_person(projective_native):
    n=projective_native
    for frame in (1,2,3):
        for source in (1,2):snapshot=publish_lineage(n,*physical_packet(n,source,frame))
    old,=snapshot.entities
    # F produces NO native absence or new source clock. Only its genuine
    # prequeue delivery age expires while K continues stationary/current.
    for frame in range(4,18):
        snapshot=publish_lineage(n,*physical_packet(n,1,frame,stationary=True))
    head,=snapshot.entities
    assert head.entity_id==old.entity_id and head.observed_at_us>old.observed_at_us
    assert next(s.camera_id for s in head.sources if s.accepted)=='kitchen'
    assert n.f.ingress.times[2]==pytest.approx(.1,abs=1e-8)


def test_projective_continuous_moving_actual_FIFO_stays_one_person(projective_native):
    n=projective_native
    counts=[]
    for frame in range(1,8):
        for source in (1,2):
            snapshot=publish_lineage(n,*physical_packet(n,source,frame))
            if frame>=4:counts.append(len(snapshot.entities))
    assert counts==[1]*8


def test_projective_actual_prior_builder_retains_similarity_scale_and_directions():
    from noesis_core.contracts.scene_prior import world_to_scene_presentation_sha256,scene_revision_sha256
    from noesis.calibration.manager import CalibrationSnapshot
    from noesis.pipelines.world_quality_adapter import runtime_spatial_bindings
    ex=world_examples()
    frames={s:ex.frame_binding(s) for s in (1,2)}
    snapshots={}
    presentations={}
    for s,b in frames.items():
        # Distinct world revisions required for distinct P, as in actual K/F.
        if s==2:
            b=ex.frame_binding(s,target=ex.metric(.1))
            frames[s]=b
        snapshots[s]=CalibrationSnapshot(('living-room','kitchen','family-room')[s],np.diag([100.,100.,1.]),
            tuple(np.eye(4).flatten(order='F')),0.,(1920,1080),frame_contract=b.frame_transform(),
            camera_calibration_sha256=b.source_camera_calibration_sha256)
        P=np.eye(4);P[:3,:3]*=2. if s==1 else 3.;P[0,3]=7. if s==1 else 11.
        matrix=tuple(P.flatten(order='F'))
        presentations[b.target_frame.revision]={'contract':'noesis.scene_prior.world_to_scene_presentation',
            'contract_version':1,'world_frame':b.target_frame.model_dump(),'render_frame':'menon_scene',
            'world_to_scene_col_major':matrix,'world_to_scene_sha256':world_to_scene_presentation_sha256(b.target_frame,'menon_scene',matrix),
            'scene_revision_id':scene_revision_sha256(b.target_frame,'menon_scene',matrix),
            'source_transform_sha256s':(b.target_from_source_sha256,),
            'provenance':{'role':'synthetic','sha256':'a'*64,'version':'synthetic'},'accepted':True,
            'authored_scene_sha256':'b'*64,'registration_sha256':'c'*64}
    provider=SimpleNamespace(snapshot=lambda s,c:snapshots[s],
        snapshots=lambda labels:{s:snapshots[s] for s in labels},
        world_frame_presentations=lambda:presentations)
    scene=SimpleNamespace(frame_bindings=lambda:{('living-room','kitchen','family-room')[s]:b for s,b in frames.items()})
    bindings,_=runtime_spatial_bindings('synthetic', {1:'kitchen',2:'family-room'},provider,scene)
    policy=runtime_projective_policy(bindings,provider)
    K=np.asarray(snapshots[2].intrinsics)
    E=np.asarray(snapshots[2].world_extrinsics_col_major).reshape(4,4,order='F')
    P1=np.asarray(presentations[frames[1].target_frame.revision]['world_to_scene_col_major']).reshape(4,4,order='F')
    P2=np.asarray(presentations[frames[2].target_frame.revision]['world_to_scene_col_major']).reshape(4,4,order='F')
    expected=K@(E@np.linalg.inv(P2)@P1)[:3]
    assert np.allclose(policy.cameras[0].peer_projection,expected)
    assert not np.isclose(np.linalg.det((np.linalg.inv(P2)@P1)[:3,:3]),1.)
    assert all(c.measurement_calibration_sha256!=c.source.calibration_revision for c in policy.cameras)


@pytest.mark.parametrize('mode',['qualified','fresh_current','invalid_current','duplicate_current','stale_current','mixed_current','foreign_run','foreign_graph'])
def test_i139_real_fifo_binding_original_reuse_never_hides_bad_current(i138_image_native,mode):
    from noesis.pipelines.source_acquisition import OriginalSwinExtraction,OriginalSwinCache
    n=i138_image_native;cache=n.pipeline.xc3d_swin_ingress.cache
    for frame in (216,217,218):
        n.now[0]=100.+(frame-218)/30.
        for source in (2,1):
            work,tracks=n.packet(source,frame)
            if frame==218:
                value=OriginalSwinExtraction(work.private.original,1 if source==2 else 6,
                    n.f.profile,(1.,)+(0.,)*255,n.now[0])
                assert cache.record(value)
                work=replace(work,swin_extractions=(cache.freeze_exact(value.original,value.raw_id),))
            publish_lineage(n,work,tracks)
    original_ids={v[3].extraction_id for values in cache.members for v in values.values() if v[3] is not None}
    assert len(original_ids)==2
    n.now[0]=100.+1/30
    work,tracks=n.packet(2,219)
    if mode in ('foreign_run','foreign_graph'):
        n.bridge._cache=OriginalSwinCache(run_id='foreign' if mode=='foreign_run' else cache.run_id,
            graph_id='foreign' if mode=='foreign_graph' else cache.graph_id)
    elif mode!='qualified':
        value=OriginalSwinExtraction(work.private.original,1,n.f.profile,(1.,)+(0.,)*255,n.now[0])
        if mode=='invalid_current':value=replace(value,original=replace(value.original,original_frame=218))
        elif mode=='mixed_current':value=replace(value,profile=replace(value.profile,fingerprint='foreign'))
        elif mode=='stale_current':value=replace(value,ingress_monotonic_s=99.)
        elif mode=='fresh_current':value=replace(value,vector=(0.,1.)+(0.,)*254)
        work=replace(work,swin_extractions=(value,value) if mode=='duplicate_current' else (value,))
    publish_lineage(n,work,tracks)
    proof=n.proofs[-1];family=next(s for s in proof.cohort if s.member.source_id==2)
    values=[v for v in proof.appearances if v.appearance.stamp.member==family.member]
    assert bool(values) is (mode in ('qualified','fresh_current'))
    if values:
        value,=values
        assert family.frame_id==219
        if mode=='qualified':
            assert value.appearance.stamp.frame_id==218 and value.ingress_monotonic_s==100.
            assert value.appearance.extraction_id in original_ids
        else:
            assert value.appearance.stamp.frame_id==219 and value.ingress_monotonic_s==n.now[0]
            assert value.vector==(0.,1.)+(0.,)*254 and value.appearance.extraction_id not in original_ids
