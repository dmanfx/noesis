"""New T9 SDK-neutral ORIGINAL quality cases; synthetic vectors/time/geometry."""
from dataclasses import dataclass, replace
from types import SimpleNamespace
from pathlib import Path
import ast
import copy
import sys
import importlib.util
import json
import time

import pytest
from noesis.pipelines.world_quality_adapter import capture_quality_input, freeze_quality_work
from noesis.pipelines.source_acquisition import OriginalCohort, PrivateAssociationWork, freeze_owner_retention
from noesis.metadata.object_depth import ObjectDepthResult
from noesis_core.tracking_continuity import TrackingLifecycleRegistry
from noesis_core.contracts.base import Vector3, Matrix3
from noesis_core.contracts.world_measurement import (
    WorldMeasurementCohort, WorldMeasurementSet, WorldMeasurementHypothesis, ResolvedGroundMeasurement,
)


def world_examples():
    """Read FROZEN public World example helpers; do not rerun their tests."""
    path=Path(__file__).resolve().parents[2]/"tests/test_xc3d_world_physical_person.py"
    spec=importlib.util.spec_from_file_location("t9_actual_world_examples",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('fault',[None,'stop','revision','second-expiry','admission'])
def test_r173_actual_gate_normal_hold_expiry_reprepares_whole_cohort(monkeypatch,fault):
    from concurrent.futures import Future
    from noesis.server.websocket import GatedOutboundAdmission,OutboundAdmissionReceipt
    from noesis.pipelines.world_quality_adapter import OrderedQualityWorldIngress
    from noesis_core.world_service import WorldPreparedHoldExpired,WorldPublicationRefused
    examples=world_examples();monkeypatch.setitem(sys.modules,'tests.test_xc3d_world_physical_person',examples)
    path=Path(__file__).resolve().parents[2]/'tests/test_xc3d_split_original_transfer.py'
    spec=importlib.util.spec_from_file_location('r173_normal_hold_examples',path)
    transfer=importlib.util.module_from_spec(spec);spec.loader.exec_module(transfer)
    actor,harness,old,clock,current=transfer._r173_normal_committed_hold_with_public_clock()
    assert current.publication.snapshot.entities[0].position_state=='held'
    # Reuse the exact owned producer rows and public native originals that
    # created this real candidate, without consuming its pending reservation.
    original=next(o for o in actor.originals.values() if o.stamp.member.source_id==1)
    rows=[dict(actor.tracking_rows[original.stamp.observation_id])]
    actor.world.discard(current)
    ingress=OrderedQualityWorldIngress(actor.world,actor.identity,actor.sources,origin_bindings=lambda:{})
    # SAME detached plan, real owner/custody/public World and real outbound
    # gate. Native stage itself and physical-scene accuracy are not claimed.
    _track,_life,_ms,_res,_dep,private,continuity=original_quality(frame=4)
    pts=round(original.stamp.media_time_s*1e9)
    private=replace(private,media_pts_ns=pts)
    continuity=replace(continuity,observed_at_us=actor.us,media_pts_ns=pts)
    work=freeze_quality_work(private,continuity,())
    ingress._staged=([(work,actor.sources[0],(0,0,4,pts),frozenset(),tuple(rows),(original,))],(),())
    ingress._context=actor.context
    gates=[];batches=[];wire=[];preparations=[]
    prepare=ingress.prepare
    def captured(*args,**kwargs):
        result=prepare(*args,**kwargs);preparations.append(result);return result
    monkeypatch.setattr(ingress,'prepare',captured)
    class Outbound:
        def admit_broadcast_batch_sync(self,messages):
            if gates and fault=='admission':raise RuntimeError('r173-readmission-failed')
            frozen=json.loads(json.dumps(messages));batches.append(frozen)
            decision,submission=Future(),Future()
            decision.add_done_callback(lambda f:wire.extend(frozen) if f.result().release else None)
            gate=GatedOutboundAdmission(OutboundAdmissionReceipt(len(gates)+1,1,len(frozen),1,1,1024),decision,submission)
            gates.append(gate);actual=gate.commit_then_release
            def release(commit):
                if len(gates)==1:clock[0]=old.stale_after_us+1
                def checked():
                    try:return commit()
                    except WorldPreparedHoldExpired as error:
                        assert error.reason=='committed_spatial_hold_custody_changed'
                        if fault=='stop':actor.identity.stop_image_region_source(actor.sources[1],graph_id='r173-gate-stop')
                        elif fault=='revision':actor.identity.reset_source(replace(actor.sources[1],source_epoch=1))
                        raise
                if len(gates)==2 and fault=='second-expiry':
                    return actual(lambda:(_ for _ in ()).throw(WorldPreparedHoldExpired(
                        preparations[-1],actor.memberships,actor.context,reason='committed_spatial_hold_custody_changed')))
                return actual(checked)
            gate.commit_then_release=release
            return gate
    publisher=harness.publisher;publisher._ws=Outbound();publisher._world_service=ingress
    sequence=publisher._next_publication_sequence_by_source.get(1,0)
    before=(actor.world._revision,len(harness.journal))
    frontiers=actor.identity.stats().spatial_member_frontiers
    metadata=dict(camera_id='kitchen',frame_id=4,source_epoch=0,observed_at_us=continuity.observed_at_us,
        media_pts_ns=continuity.media_pts_ns)
    if fault:
        with pytest.raises((WorldPublicationRefused,RuntimeError)):publisher.publish(1,rows,frame_metadata=metadata)
        assert not wire and before==(actor.world._revision,len(harness.journal))
        assert publisher._next_publication_sequence_by_source.get(1,0)==sequence
        assert publisher._terminal_failure is not None
        # The explicit source reset clears its old frontier; no failed gate
        # may consume a current measurement frontier in either case.
        assert actor.identity.stats().spatial_member_frontiers==(0 if fault=='revision' else frontiers)
    else:
        receipt=publisher.publish(1,rows,frame_metadata=metadata)
        assert [g.state for g in gates]==['aborted','released']
        assert len(preparations)==2 and preparations[0].publication.snapshot.entities
        fresh,=preparations[1].publication.snapshot.entities
        assert fresh.observed_at_us>old.observed_at_us and fresh.stale_after_us>clock[0]
        assert fresh.observed_at_us==rows[0]['observed_at_us']==original.stamp.acquired_at_s*1e6
        assert [s.camera_id for s in fresh.sources if s.accepted]==['kitchen']
        assert next(s.observation_id for s in fresh.sources if s.accepted)==original.stamp.observation_id
        assert batches[0][0]['cohort']==batches[1][0]['cohort']
        assert receipt.tracking_publication_sequence==sequence
        assert publisher._next_publication_sequence_by_source[1]==sequence+1
        assert [m['type'] for m in wire][:2]==['tracking','world_snapshot']
        assert wire==batches[1] and actor.world._revision==before[0]+1
        assert actor.identity.stats().spatial_member_frontiers==frontiers+1
        result=examples.actual_node_wire(actor,harness.wire+wire,'r173-normal-hold-abort-reprepare')
        assert [len(s['people']) for s in result['snapshots']]==[1,1,1]
        assert all(d['accepted'] for d in result['diagnostics'])
    assert ingress._staged is None and actor.identity.stats().pending_reservations==0


@pytest.mark.parametrize('fault',[None,'stop','revision','admission','abort','second-expiry','foreign-expiry'])
def test_r168_actual_gate_same_cohort_expiry_reprepare(monkeypatch,fault):
    from concurrent.futures import Future
    from noesis.server.websocket import GatedOutboundAdmission,OutboundAdmissionReceipt
    from noesis.telemetry.publishers import TrackingTelemetryPublisher
    from noesis.pipelines.world_quality_adapter import OrderedQualityWorldIngress
    from noesis_core.world_service import WorldPreparedHoldExpired,WorldPublicationRefused
    examples=world_examples()
    monkeypatch.setitem(sys.modules,'tests.test_xc3d_world_physical_person',examples)
    path=Path(__file__).resolve().parents[2]/'tests/test_xc3d_split_original_transfer.py'
    spec=importlib.util.spec_from_file_location('r168_transfer_examples',path)
    transfer=importlib.util.module_from_spec(spec);spec.loader.exec_module(transfer)
    actual_node_wire=examples.actual_node_wire
    actor,harness,clock,owner_clock,old,held=transfer._r163_split_hold_with_public_clock(monkeypatch)
    actor.us=old.stale_after_us-1
    ingress=OrderedQualityWorldIngress(actor.world,actor.identity,actor.sources,origin_bindings=lambda:{})
    # A test-owned detached EMPTY plan contains no measurement or new native
    # grant. The real owner, original custody and World prepare/commit remain
    # production objects. Stage's native mutation is tested separately.
    track,life,ms,res,dep,private,continuity=original_quality(frame=4)
    continuity=replace(continuity,observed_at_us=actor.us)
    work=freeze_quality_work(private,continuity,())
    ingress._staged=([(work,actor.sources[0],(0,0,4,private.media_pts_ns),frozenset(),[],())],(),())
    ingress._context=actor.context
    wire=[];gates=[];batches=[];preparations=[]
    real_prepare=ingress.prepare
    def prepare(*args,**kwargs):
        prepared=real_prepare(*args,**kwargs);preparations.append(prepared);return prepared
    monkeypatch.setattr(ingress,'prepare',prepare)
    class Outbound:
        def admit_broadcast_batch_sync(self,messages):
            if len(gates)==1 and fault=='admission':raise RuntimeError('r168-admission-failed')
            frozen=json.loads(json.dumps(messages));batches.append(frozen)
            decision,submission=Future(),Future()
            decision.add_done_callback(lambda f:wire.extend(frozen) if f.result().release else None)
            gate=GatedOutboundAdmission(OutboundAdmissionReceipt(len(gates)+1,1,len(frozen),1,1,1024),decision,submission)
            gates.append(gate);actual=gate.commit_then_release
            def release(commit):
                if len(gates)==1:actor.us=old.stale_after_us+1
                def checked():
                    try:return commit()
                    except WorldPreparedHoldExpired as error:
                        if fault=='stop':actor.identity.stop_image_region_source(actor.sources[1],graph_id='r101-graph')
                        if fault=='revision':actor.identity.reset_source(replace(actor.sources[1],source_epoch=1))
                        if fault=='foreign-expiry':error.prepared=object()
                        raise
                if len(gates)==2 and fault=='second-expiry':
                    return actual(lambda:(_ for _ in ()).throw(WorldPreparedHoldExpired(
                        preparations[-1],actor.memberships,actor.context)))
                try:return actual(checked)
                finally:
                    if fault=='abort':gate._state='resolution_failed'
            gate.commit_then_release=release
            return gate
    publisher=harness.publisher
    publisher._ws=Outbound();publisher._world_service=ingress
    metadata_calls=[]
    publisher._metadata_getter=lambda *args:metadata_calls.append(args) or {}
    sequence=publisher._next_publication_sequence_by_source[1]
    before=(actor.world._revision,len(harness.journal))
    meta=dict(camera_id='kitchen',frame_id=4,source_epoch=0,observed_at_us=continuity.observed_at_us,
              media_pts_ns=continuity.media_pts_ns)
    if fault is not None:
        with pytest.raises((WorldPublicationRefused,RuntimeError)):publisher.publish(1,[],frame_metadata=meta)
        assert wire==[] and before==(actor.world._revision,len(harness.journal))
        assert publisher._terminal_failure is not None and ingress._staged is None
    else:
        receipt=publisher.publish(1,[],frame_metadata=meta)
        assert [g.state for g in gates]==['aborted','released']
        assert len(preparations)==2 and len(preparations[0].publication.snapshot.entities)==1
        assert not preparations[1].publication.snapshot.entities
        assert batches[0][0]['cohort']==batches[1][0]['cohort'] and receipt.tracking_publication_sequence==sequence
        assert len(wire)==len(batches[1]) and wire[0]['type']=='tracking' and wire[1]['type']=='world_snapshot'
        assert actor.world._revision==before[0]+1 and publisher._next_publication_sequence_by_source=={1:sequence+1}
        assert publisher._terminal_failure is None and ingress._staged is None
        result=actual_node_wire(actor,harness.wire+wire,'r168-abort-reprepare-readmit')
        assert [len(s['people']) for s in result['snapshots']]==[1,1,0]
        assert all(d['accepted'] for d in result['diagnostics'])
    assert actor.identity.stats().pending_reservations==0
    assert len(metadata_calls)==1


@dataclass(frozen=True)
class SyntheticCalibrationInput:
    camera_id: str
    camera_calibration_sha256: str
    frame_contract: object
    image_size: tuple=(1920,1080)


class NativeQualityFixture:
    """Changed adapter -> SINGLE real normal factory/owner/publisher.

    Every metric/appearance/timing/lineage field is SYNTHETIC. No SDK imported.
    """
    def __init__(self,tmp_path,*,source_ids=(1,2),identity_factory=None):
        from noesis_core.runtime_world import create_runtime_world_service
        from noesis.identity_association_service import IdentityAssociationService
        from noesis.pipelines.world_quality_adapter import runtime_spatial_bindings, OrderedQualityWorldIngress
        from reid.mv3dt_person_association import SpatialAssociationPolicy, AppearanceProfile
        from noesis.telemetry.publishers import TrackingTelemetryPublisher
        self.examples=world_examples()
        self.frames={s:self.examples.frame_binding(s) for s in source_ids}
        self.labels={s:("living-room","kitchen","family-room")[s] for s in source_ids}
        self.examples.CAMERAS.update(self.labels)  # SYNTHETIC consumer fixture only, no new runtime peer.
        self.provider=SimpleNamespace(snapshot=lambda s,c:SyntheticCalibrationInput(c,
            self.frames[s].source_camera_calibration_sha256,self.frames[s].frame_transform()))
        self.provider.snapshots=lambda labels:{s:self.provider.snapshot(s,c) for s,c in labels.items()}
        self.scene=SimpleNamespace(frame_bindings=lambda:{self.labels[s]:b for s,b in self.frames.items()})
        self.sources,active=runtime_spatial_bindings("synthetic-quality-run",self.labels,self.provider,self.scene)
        self.profile=AppearanceProfile("synthetic-selected-Swin","fc_pred",256,"synthetic-profile")
        factory=identity_factory or IdentityAssociationService.for_spatial
        self.identity=factory("synthetic-quality-run",
            SpatialAssociationPolicy(self.sources,self.profile,(("kitchen","family-room"),)))
        self.world=create_runtime_world_service(runtime="ds9",pipeline_config={},camera_labels=self.labels,
            calibration_provider=self.provider,repo_root=tmp_path,run_id="synthetic-quality-run",
            instance_id="synthetic-native-quality",software_revision="synthetic",
            journal_path=tmp_path/"world.sqlite3",association_service=self.identity,spatial_bindings=lambda:active)
        self.lineage={s:{} for s in source_ids}
        self.ingress=OrderedQualityWorldIngress(self.world,self.identity,self.sources,origin_bindings=lambda:self.lineage)
        self.registries={s:TrackingLifecycleRegistry(reappearance_grace_s=.43) for s in self.labels}
        self.wire=[]
        fixture=self
        class Outbound:
            def admit_broadcast_batch_sync(self,messages):
                copied=json.loads(json.dumps(messages,allow_nan=False))
                def commit_then_release(commit):
                    publication=commit()
                    assert fixture.world.current_snapshot() is publication.snapshot
                    fixture.wire.extend(copied)
                    return publication
                return SimpleNamespace(receipt=SimpleNamespace(submission_id=len(fixture.wire)+1,
                    message_count=len(copied)),commit_then_release=commit_then_release)
        self.publisher=TrackingTelemetryPublisher(Outbound(),world_service=self.ingress)

    def packet(self,source,frame,*,x=1.,scale=1.,cached=False,empty=False,shadow=False,raws=(9,),households=None):
        registry=self.registries[source]
        now=time.time_ns()//1000
        pts=frame*33_333_333
        original=replace(original_quality(frame,source)[5].original,run_id=self.world.producer.run_id)
        inputs,tracks=[],[]
        b=self.frames[source]
        for raw in (() if empty else raws):
            life=registry.identity_observation(source,raw,frame_id=frame,observed_at_us=now,media_pts_ns=pts)
            track,_life,measurements,resolution,depth,_private,_continuity=original_quality(frame,source)
            track.update(tracker_id=raw,tracker_lifecycle_generation=life.generation,observed_at_us=now,
                image_size=[1920,1080],confidence=.9,tracker_confidence=.9,world=[x+(raw-9)*3.,0.,2.],
                world_covariance=[v*scale for v in track["world_covariance"]],
                calibration_revision=b.source_frame.revision,world_frame_revision=b.target_frame.revision,
                world_calibration_sha256=b.source_camera_calibration_sha256,world_transform_sha256=b.target_from_source_sha256,
                source_epoch=life.source_epoch,identity_kind="unknown")
            track.update((households or {}).get(raw,{}))
            cohort=WorldMeasurementCohort.model_validate({**measurements.cohort.model_dump(),
                "track_key":f"{source}:{raw}:{life.generation}","tracker_id":raw,
                "tracker_lifecycle_generation":life.generation,"observed_at_us":now,
                "world_revision":b.target_frame.revision,"calibration_revision":b.source_camera_calibration_sha256,
                "world_transform_sha256":b.target_from_source_sha256})
            position=Vector3(x=track["world"][0],y=0.,z=2.)
            covariance=Matrix3(values=tuple(track["world_covariance"]))
            measurements=WorldMeasurementSet(cohort=cohort,hypotheses=tuple(
                WorldMeasurementHypothesis.model_validate({**h.model_dump(),"cohort":cohort,
                    "position":position,"covariance":covariance}) for h in measurements.hypotheses))
            resolution=ResolvedGroundMeasurement.model_validate({**resolution.model_dump(),"cohort":cohort,
                "position":position,"covariance":covariance})
            depth=replace(depth,object_id=raw,measurement_cached=cached)
            tracks.append(track)
            inputs.append(capture_quality_input(track,life,measurements,resolution,depth))
        continuity=registry.update_frame(source_id=source,camera_id=self.labels[source],frame_id=frame,
            observed_at_us=now,media_pts_ns=pts,tracks=tracks,shadow_tracker_ids=raws if shadow else ())
        retention=freeze_owner_retention(continuity,registry.identity_retention_keys(source,media_pts_ns=pts,
            max_gap_ns=430_000_000),430_000_000,original)
        private=PrivateAssociationWork(source,frame,pts,0,original,None,owner_retention=retention)
        return freeze_quality_work(private,continuity,tuple(inputs)),tracks

    def join_current(self):
        from reid.mv3dt_person_association import (
            AppearanceEvidence,appearance_extraction_identity,SpatialPersonCandidate,
            SpatialCurrentPersonProof,SpatialClockContext,
        )
        originals=tuple(self.ingress.originals.values())
        candidates=[]
        for left in (o for o in originals if o.stamp.member.source_id==1):
            for right in (o for o in originals if o.stamp.member.source_id==2):
                def appearance(o):
                    identity=appearance_extraction_identity(o.stamp,self.profile)
                    return AppearanceEvidence(o.stamp,identity,self.profile.model_id,o.stamp.acquired_at_s,
                        o.stamp.clock_domain,identity,self.profile)
                same=left.stamp.member.raw_id==right.stamp.member.raw_id
                candidates.append(SpatialPersonCandidate(left.stamp,right.stamp,appearance(left),appearance(right),
                    .9 if same else .2,"synthetic-copy:"+left.stamp.observation_id,"synthetic-copy:"+right.stamp.observation_id,
                    .03 if same else 3.,"synthetic-independent-geometry","synthetic-frame","synthetic-revision",
                    tuple(self.frames[s].source_frame.revision for s in (1,2)),
                    tuple(self.frames[s].target_from_source_sha256 for s in (1,2)),
                    ("synthetic-independent:"+left.stamp.observation_id,"synthetic-independent:"+right.stamp.observation_id)))
        context=SpatialClockContext(self.ingress._context.intervals,"synthetic-only-comparability","synthetic-common",
            tuple((s,0.,0.) for s in (1,2)),tuple((s,0.,100.) for s in (1,2)),"synthetic")
        proof=SpatialCurrentPersonProof("synthetic-current-person",tuple(o.stamp for o in originals),tuple(candidates),
            tuple((c.left.member,c.right.member) for c in candidates if c.similarity>=.6),"synthetic")
        result=self.identity.join_spatial(proof,at_interval=context)
        assert result.disposition=="accepted",result
        # Preserve the ACTUAL service's exact join barriers in this SAME actor.
        plans,withdrawals,retired=self.ingress._staged
        self.ingress._staged=(plans,tuple(sorted(set(withdrawals+result.withdrawn_members))),
            tuple(sorted(set(retired+result.retired_keys))))

    def publish(self,packets,*,join=False):
        from noesis.pipelines.source_acquisition import SwinOwnerBinding
        works=tuple(w for w,_ in packets)
        groups=tuple((w.continuity.source_id,t) for w,t in packets)
        self.ingress.stage(works,groups)
        if join:self.join_current()
        work,tracks=packets[0]
        self.publisher.publish(work.continuity.source_id,tracks,frame_metadata={
            "camera_id":work.continuity.camera_id,"frame_id":work.continuity.frame_id,
            "source_epoch":work.continuity.source_epoch,"observed_at_us":work.continuity.observed_at_us,
            "media_pts_ns":work.continuity.media_pts_ns})
        for w,_tracks in packets:
            source=w.continuity.source_id
            retained,_=__import__("noesis.pipelines.source_acquisition",fromlist=["owner_retention_members"]).owner_retention_members(
                w.private.owner_retention,w.private.original,w.continuity.source_epoch,w.continuity.frame_id)
            self.lineage[source]={k:o for k,o in self.lineage[source].items() if k in retained}
            for r in w.rows:
                life=r.evidence.lifecycle
                key=(life.source_epoch,life.tracker_id,life.generation)
                self.lineage[source].setdefault(key,SwinOwnerBinding(key,r.original,life.frame_id))
        return self.world.current_snapshot()


@pytest.fixture
def native_quality(tmp_path):
    fixture=NativeQualityFixture(tmp_path)
    try:yield fixture
    finally:fixture.world.close()


def test_r119_exact_current_newborn_precedes_only_unrelated_same_cohort_withdrawal(native_quality,monkeypatch):
    f=native_quality
    before=f.publish([f.packet(1,10,raws=(7,))])
    old=next(iter(f.ingress.originals.values()))
    work,tracks=f.packet(1,30,raws=(40,))
    row,=work.rows
    assert row.evidence.lifecycle.disposition=='new'
    assert row.evidence.lifecycle.previous_media_pts_ns is None
    assert row.evidence.lifecycle.frame_id==work.continuity.frame_id==30
    calls=[]
    observe=f.identity.observe_spatial;withdraw=f.identity.withdraw
    def observed(original,receipt):
        calls.append(('observe',original.stamp.member.raw_id));return observe(original,receipt)
    def withdrawn(removal):
        calls.append(('withdraw',removal.last_observation.member.raw_id));return withdraw(removal)
    monkeypatch.setattr(f.identity,'observe_spatial',observed)
    monkeypatch.setattr(f.identity,'withdraw',withdrawn)
    final=f.publish([(work,tracks)])
    new,=f.ingress.originals.values()
    assert new.stamp.member.raw_id==40 and new.stamp.lifecycle_started_frame==30
    assert new.stamp.frame_id==30
    assert old.stamp.member not in f.ingress.originals
    assert before.entities and len(final.entities)==1
    assert final.entities[0].entity_id!=before.entities[0].entity_id
    assert calls==[('observe',40),('withdraw',7)]
    note=f.identity.ledger.observation_refusal_diagnostic()
    assert note is None
    actual=f.examples.actual_node_wire(f,f.wire,'r119-exact-newborn')
    assert [len(s['people']) for s in actual['snapshots']]==[1,1]


def test_r119_existing_member_keeps_original_withdraw_then_update_order(native_quality,monkeypatch):
    f=native_quality;f.publish([f.packet(1,10,raws=(7,8))])
    work,tracks=f.packet(1,30,raws=(8,))
    calls=[];observe=f.identity.observe_spatial;withdraw=f.identity.withdraw
    def observed(original,receipt):
        calls.append(('observe',original.stamp.member.raw_id));return observe(original,receipt)
    def withdrawn(removal):
        calls.append(('withdraw',removal.last_observation.member.raw_id));return withdraw(removal)
    monkeypatch.setattr(f.identity,'observe_spatial',observed);monkeypatch.setattr(f.identity,'withdraw',withdrawn)
    final=f.publish([(work,tracks)])
    assert len(final.entities)==1 and calls==[('withdraw',7),('observe',8)]
    assert next(iter(f.ingress.originals.values())).stamp.lifecycle_started_frame==10


def test_r119_preexisting_owner_frontier_still_rejects_current_birth(native_quality):
    from reid.mv3dt_person_association import MemberRemoval
    from noesis_core.world_service import WorldPublicationRefused
    f=native_quality;before=f.publish([f.packet(1,10,raws=(7,))])
    old=next(iter(f.ingress.originals.values()))
    result=f.identity.withdraw(MemberRemoval('prior-real-barrier',old.stamp,'termination',
        30,30*33_333_333/1e9,old.stamp.clock_domain,30*33_333_333/1e9))
    assert result.disposition=='removed'
    work,tracks=f.packet(1,30,raws=(40,))
    with pytest.raises(WorldPublicationRefused,match='lifecycle_frontier'):
        f.ingress.stage((work,),((1,tracks),))
    assert f.world.current_snapshot() is before
    note=f.identity.ledger.observation_refusal_diagnostic()
    assert note['source_frontier']==30 and note['stamp']['lifecycle_started_frame']==30


def test_r119_reused_raw_new_generation_is_not_moved_ahead_of_old_removal(native_quality):
    from noesis_core.world_service import WorldPublicationRefused
    f=native_quality;before=f.publish([f.packet(1,10,raws=(40,))])
    old=next(iter(f.ingress.originals.values()))
    f.packet(1,20,empty=True)
    work,tracks=f.packet(1,30,raws=(40,))
    row,=work.rows
    assert row.evidence.lifecycle.disposition=='new'
    assert row.evidence.lifecycle.generation>old.stamp.member.lifecycle_generation
    with pytest.raises(WorldPublicationRefused,match='lifecycle_frontier'):
        f.ingress.stage((work,),((1,tracks),))
    assert f.world.current_snapshot() is before


def test_r119_closed_original_history_is_not_reopened(native_quality):
    from reid.mv3dt_person_association import MemberRemoval
    from noesis_core.world_service import WorldPublicationRefused
    f=native_quality;before=f.publish([f.packet(1,10,raws=(40,))])
    old=next(iter(f.ingress.originals.values()))
    result=f.identity.withdraw(MemberRemoval('actual-termination',old.stamp,'termination',
        20,20*33_333_333/1e9,old.stamp.clock_domain,20*33_333_333/1e9))
    assert result.disposition=='removed'
    work,tracks=f.packet(1,30,raws=(40,))
    assert work.rows[0].evidence.lifecycle.disposition=='continuous'
    with pytest.raises(WorldPublicationRefused,match='closed_lifecycle'):
        f.ingress.stage((work,),((1,tracks),))
    assert f.world.current_snapshot() is before


@pytest.mark.parametrize('case', ['frontier', 'control', 'observer_failure'])
def test_lifecycle_refusal_actual_adapter_unrelated_removal_late_native_birth(native_quality, caplog, monkeypatch, case):
    """Real native registry/retention -> adapter -> service; no fabricated ledger rows."""
    from noesis.pipelines.source_acquisition import SwinOwnerBinding, owner_retention_members
    from noesis_core.world_service import WorldPublicationRefused
    f = native_quality
    first = f.packet(1, 10, raws=(7,))
    if case != 'control':
        f.publish([first])
    # Native B exists before canonical first admission; native source continues
    # normally while those quality packets do not enter the canonical actor.
    birth, _ = f.packet(1, 15, raws=(7, 8))
    row = next(r for r in birth.rows if r.evidence.lifecycle.tracker_id == 8)
    life = row.evidence.lifecycle
    key = (life.source_epoch, life.tracker_id, life.generation)
    f.lineage[1][key] = SwinOwnerBinding(key, row.original, life.frame_id)
    f.packet(1, 20, raws=(8,))
    current, tracks = f.packet(1, 30, raws=(8,))
    retained, measured = owner_retention_members(current.private.owner_retention,
        current.private.original, current.continuity.source_epoch, current.continuity.frame_id)
    assert key in retained and key in measured and all(k[1] != 7 for k in retained)
    if case == 'control':
        f.publish([(current, tracks)])
        assert next(iter(f.ingress.originals.values())).stamp.lifecycle_started_frame == 15
        assert not any('World quality lifecycle refusal:' in r.message for r in caplog.records)
        return
    if case == 'observer_failure':
        monkeypatch.setattr(f.identity.ledger, 'observation_refusal_diagnostic',
                            lambda: (_ for _ in ()).throw(ValueError('optional observer failure')))
    before = f.world.current_snapshot()
    with pytest.raises(WorldPublicationRefused) as error:
        f.ingress.stage((current,), ((1, tracks),))
    assert error.value.reason == 'lifecycle_frontier'
    assert f.world.current_snapshot() is before  # No publication/measurement repair.
    logs = [r.message.split('World quality lifecycle refusal: ', 1)[1] for r in caplog.records
            if r.message.startswith('World quality lifecycle refusal: ')]
    if case == 'observer_failure':
        assert logs == []
        return
    assert len(logs) == 1 and len(logs[0].encode()) <= 8192
    note = json.loads(logs[0])
    assert note['kernel']['reason'] == 'lifecycle_frontier'
    assert note['kernel']['stamp']['member']['raw_id'] == 8
    assert note['kernel']['stamp']['lifecycle_started_frame'] == 15
    assert note['kernel']['source_frontier'] == 30
    assert note['kernel']['current_stamp'] is note['kernel']['closed_stamp'] is None
    assert note['original']['original_frame'] == 30
    assert note['native_birth']['original_frame'] == 15
    assert note['native_observation']['frame_id'] == 30
    assert note['native_lifecycle_started_frame'] == 15
    assert note['retained'] is note['measured'] is True
    assert note['active_row'][:2] == list(key[1:]) and not note['shadow'] and not note['terminated']
    # A repeated terminal refusal does not produce per-cohort diagnostic spam.
    with pytest.raises(WorldPublicationRefused, match='lifecycle_frontier'):
        f.ingress.stage((current,), ((1, tracks),))
    assert sum(r.message.startswith('World quality lifecycle refusal: ') for r in caplog.records) == 1


@pytest.mark.parametrize("order",[(1,2),(2,1)])
def test_t9_connected_native_adapter_REAL_factory_publisher_dynamic_quality_Node22(native_quality,order):
    f=native_quality
    first=f.publish([f.packet(s,1,x=1. if s==1 else 8.,scale=4. if s==1 else 1.) for s in order],join=True)
    head,=first.entities
    assert head.position.x==8. and sum(s.accepted for s in head.sources)==1
    assert next(s for s in head.sources if s.accepted).camera_id=="family-room"
    assert head.covariance.values[0]==.04 and len(head.sources)==2
    second=f.publish([f.packet(s,2,x=1. if s==1 else 8.,scale=1. if s==1 else 4.) for s in order])
    changed,=second.entities
    assert changed.entity_id==head.entity_id and changed.position.x==1. and changed.velocity_mps is None
    assert changed.continuity_segment_id>head.continuity_segment_id
    anchored=[]
    for message in f.wire:
        if message["type"]=="tracking":anchored.append(message)
        else:
            assert message["cohort"]==anchored[-1]["cohort"]
            if message["type"]=="world_snapshot":assert message["payload"]==anchored[-1]["world_snapshot"]
    result=f.examples.actual_node_wire(f,f.wire,"native-t9-quality-"+str(order[0]))
    assert [len(s["people"]) for s in result["snapshots"]]==[1,1]
    assert [s["people"][0]["worldPosition"][0] for s in result["snapshots"]]==[8.,1.]


def test_t9_connected_current_local_no_comparability_or_identity_perfect_gate(native_quality):
    f=native_quality
    snapshot=f.publish([f.packet(1,1)])
    head,=snapshot.entities
    assert len(head.sources)==1 and head.position.x==1.
    assert f.ingress._context.comparability_receipt_id is None
    assert f.identity.stats().pending_reservations==0
    original=next(iter(f.ingress.originals.values()))
    assert original.stamp.acquired_at_s==original.stamp.media_time_s==33_333_333/1e9
    assert original.stamp.clock_basis=="source-media" and original.stamp.lifecycle_started_frame==1


def test_t9_connected_cached_depth_original_hold_empty_expiry_and_reuse(native_quality):
    f=native_quality
    first=f.publish([f.packet(1,1)])
    head,=first.entities
    held=f.publish([f.packet(1,2,cached=True)])
    assert held.entities[0].observed_at_us==head.observed_at_us
    assert held.entities[0].position==head.position
    gone=f.publish([f.packet(1,20,empty=True)])
    assert gone.entities==() and any(e["payload"]["event_type"]=="lost" for e in f.wire if e["type"]=="world_event")
    f.registries[1].reset_source(1)
    returned=f.publish([f.packet(1,21)])
    assert returned.entities[0].entity_id!=head.entity_id
    assert next(iter(f.ingress.originals)).source_epoch==1


def test_t9_connected_known_INACTIVE_preserves_original_birth_and_never_new_measurement(native_quality):
    f=native_quality
    first=f.publish([f.packet(1,1)])
    source_original=next(iter(f.ingress.originals.values()))
    hold=f.publish([f.packet(1,2,empty=True,shadow=True)])
    assert next(iter(f.ingress.originals.values())) is source_original
    assert hold.entities[0].observed_at_us==first.entities[0].observed_at_us
    back=f.publish([f.packet(1,3)])
    assert back.entities[0].entity_id==first.entities[0].entity_id
    assert next(iter(f.ingress.originals.values())).stamp.lifecycle_started_frame==1


def test_t9_connected_two_people_current_bundle_and_A_only_preserves_B_Node22(native_quality):
    f=native_quality
    first=f.publish([f.packet(s,1,raws=(9,10)) for s in (1,2)],join=True)
    assert len(first.entities)==2
    b=next(e for e in first.entities if e.position.x==4.)
    second=f.publish([f.packet(1,2,raws=(9,))])
    prior=next(e for e in second.entities if e.entity_id==b.entity_id)
    assert prior.position==b.position and prior.observed_at_us==b.observed_at_us
    assert len(second.entities)==2
    result=f.examples.actual_node_wire(f,f.wire,"native-t9-two-people")
    assert [len(s["people"]) for s in result["snapshots"]]==[2,2]


@pytest.mark.parametrize("fault",["birth","changed_point","frame_binding","duplicate","capacity"])
def test_t9_connected_bad_private_input_refuses_before_owner_World_or_wire(native_quality,fault):
    from noesis_core.world_service import WorldPublicationRefused
    f=native_quality
    work,tracks=f.packet(1,1)
    if fault=="birth":work=replace(work,rows=(replace(work.rows[0],evidence=replace(work.rows[0].evidence,
        lifecycle=replace(work.rows[0].evidence.lifecycle,disposition="continuous"))),))
    elif fault=="changed_point":tracks[0]["world"][0]=99.
    elif fault=="frame_binding":tracks[0]["calibration_revision"]="other"
    elif fault=="duplicate":tracks=tracks*2
    else:work=replace(work,denial="world_quality_member_capacity")
    before=f.identity.stats()
    with pytest.raises(WorldPublicationRefused):f.ingress.stage(work,tracks)
    assert f.identity.stats()==before and f.world.current_snapshot() is None and f.wire==[]


def test_t9_final_actual_FIFO_hook_calls_quality_owner_before_REAL_publisher(native_quality,monkeypatch):
    for name in ("pyservicemaker","noesis_pose_meta_ext","noesis_analytics_meta_ext","noesis_v3dt_meta_ext",
        "noesis_reid_meta_ext","noesis_depth_meta_ext","noesis_depth_tracking_tensor_ext"):
        monkeypatch.setitem(sys.modules,name,None)
    from noesis.pipelines import hooks
    from noesis_core.runtime_publication import RuntimePublicationGate
    from noesis.telemetry.bev import Footpoint
    f=native_quality
    quality,tracks=f.packet(1,1)
    processor=SimpleNamespace(pipeline=SimpleNamespace(xc3d_world_quality_ingress=f.ingress),
        tracking_pub=f.publisher,publication_gate=RuntimePublicationGate(),_tracking_lifecycle=f.registries[1],
        _stamp_footpoint_lifecycles=lambda *_args:None,_commit_enqueued_world_output_watermarks=lambda *_a,**_k:None,
        _last_tracking_publish_ts_by_sensor={},_last_tracking_count_by_sensor={},bev_renderer=None)
    foot=Footpoint(tracker_id=9,tracker_lifecycle_generation=1,frame_id=1,world_x=1.,world_y=0.,world_z=2.)
    work=hooks._TrackingPublicationWork(1,"kitchen",1,quality.continuity.observed_at_us,time.time(),33333,
        tracks,[foot],{"media_pts_ns":quality.continuity.media_pts_ns},continuity=quality.continuity,
        source_epoch=0,world_quality_work=quality)
    hooks._AnalyticsTelemetryProcessor._publish_tracking_work(processor,work)
    head,=f.world.current_snapshot().entities
    assert head.position.x==1. and f.wire[0]["type"]=="tracking"
    assert len(f.wire[0]["world_snapshot"]["entities"])==1


@pytest.mark.parametrize("wrong",["missing","source_revision","transform","camera_hash"])
def test_t9_final_runtime_provider_refuses_mismatched_REAL_binding_without_PCFinvention(native_quality,wrong):
    from noesis.pipelines.world_quality_adapter import runtime_spatial_bindings
    f=native_quality
    if wrong=="missing":f.scene=SimpleNamespace(frame_bindings=lambda:{})
    else:
        before=f.provider.snapshot
        def mismatch(source,camera):
            value=before(source,camera)
            if wrong=="camera_hash":return replace(value,camera_calibration_sha256="e"*64)
            # VALID real revisioned transforms, wrong for this ORIGINAL source.
            contract=(f.examples.frame_binding(2).frame_transform() if wrong=="source_revision" else
                f.examples.frame_binding(source,target=f.examples.metric(offset=.2)).frame_transform())
            return replace(value,frame_contract=contract)
        f.provider.snapshot=mismatch
    with pytest.raises(ValueError,match="actual_frame_binding_unavailable"):
        runtime_spatial_bindings("synthetic-quality-run",f.labels,f.provider,f.scene)
    assert f.world.current_snapshot() is None


def test_runtime_binding_uses_one_atomic_batch_every_invocation(native_quality):
    from noesis.pipelines.world_quality_adapter import runtime_spatial_bindings
    f=native_quality
    values=f.provider.snapshots(f.labels)
    calls=[]
    def batch(labels):
        assert labels == f.labels;calls.append(dict(labels));return dict(values)
    def no_single(*args):raise AssertionError('legacy individual snapshot call')
    f.provider.snapshots=batch;f.provider.snapshot=no_single
    first=runtime_spatial_bindings('synthetic-quality-run',f.labels,f.provider,f.scene)
    assert len(calls)==1
    second=runtime_spatial_bindings('synthetic-quality-run',f.labels,f.provider,f.scene)
    assert len(calls)==2 and first==second
    values[2]=replace(values[2],camera_calibration_sha256='e'*64)
    with pytest.raises(ValueError,match='actual_frame_binding_unavailable'):
        runtime_spatial_bindings('synthetic-quality-run',f.labels,f.provider,f.scene)
    assert len(calls)==3  # A later commit call cannot reuse prepare's generation.


@pytest.mark.parametrize('bad',['missing_api','missing_source','extra_source'])
def test_runtime_binding_refuses_unavailable_or_wrong_batch(native_quality,bad):
    from noesis.pipelines.world_quality_adapter import runtime_spatial_bindings
    f=native_quality
    values=f.provider.snapshots(f.labels)
    if bad=='missing_api':del f.provider.snapshots
    else:
        if bad=='missing_source':values.pop(2)
        else:values[99]=values[1]
        f.provider.snapshots=lambda labels:values
    with pytest.raises(ValueError,match='atomic_calibration'):
        runtime_spatial_bindings('synthetic-quality-run',f.labels,f.provider,f.scene)


def test_t9_final_actual_public_API_check_no_legacy_constructor_bypass(monkeypatch):
    from noesis.pipelines.world_quality_adapter import require_world_quality_api
    import noesis_core.runtime_world as factory
    require_world_quality_api()
    monkeypatch.setattr(factory,"create_runtime_world_service",lambda **_kwargs:None)
    with pytest.raises(ValueError,match="public_api_unavailable"):require_world_quality_api()


def test_t9_final_original_household_labels_not_stripped_and_collision_quarantined(native_quality):
    f=native_quality
    descriptor={"identity_kind":"visitor","stable_id":7101,"visitor_generation":4}
    first=f.publish([f.packet(1,1,households={9:descriptor})])
    head,=first.entities
    assert head.subject.kind.value=="visitor" and head.subject.stable_id==7101 and head.subject.generation==4
    before=len(f.wire)
    work,tracks=f.packet(1,2,raws=(9,10),households={9:descriptor,10:descriptor})
    second=f.publish([(work,tracks)])
    assert len(f.wire)>before and len(second.entities)==2
    assert all(e.attribution_quarantine is not None for e in second.entities)
    assert all(t["stable_id"]==7101 and t["visitor_generation"]==4 for t in f.wire[before]["tracks"])
    assert f.identity.stats().pending_reservations==0


def test_t9_final_native_startup_and_operator_use_only_optin_normal_spatial_owner():
    root=Path(__file__).resolve().parents[1]/"noesis"
    tree=ast.parse((root/"ds9_runtime_core.py").read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_run_main")
    text=ast.unparse(main)
    assert "require_world_quality_api()" in text and "xc3d_world_quality_public_api_unavailable" in text
    assert "spatial_bindings" in text and "OrderedQualityWorldIngress" in text
    assert "except (KeyError, TypeError, ValueError, OSError, AcquisitionUnavailable)" in text
    hooks=(root/"pipelines/hooks.py").read_text()
    assert 'if candidate and not xc3d_world_quality_enabled(processor.pipeline.config):' in hooks
    tree=ast.parse(hooks)
    processor=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="_AnalyticsTelemetryProcessor")
    augment=next(n for n in processor.body if isinstance(n,ast.FunctionDef) and n.name=="_augment_track_with_world")
    assert 'calibration_frame_revision' in ast.unparse(augment)


@pytest.mark.parametrize("failure",[None,"attachment","source","Swin"])
def test_t9_attachment_startup_observer_failure_cannot_activate_empty_optin(failure):
    from noesis.pipelines.world_quality_adapter import require_world_quality_attachment
    pipeline=SimpleNamespace(xc3d_analytics_attachment_status={"status":"attached"},
        xc3d_swin_ingress=object(),xc3d_source_acquisition=object())
    if failure=="attachment":pipeline.xc3d_analytics_attachment_status={"status":"failed","reason":"actual-probe-error"}
    elif failure=="source":pipeline.xc3d_source_acquisition=None
    elif failure=="Swin":pipeline.xc3d_swin_ingress=None
    if failure:
        with pytest.raises(ValueError,match="analytics_attachment_unavailable"):require_world_quality_attachment(pipeline)
    else:require_world_quality_attachment(pipeline)
    source=(Path(__file__).resolve().parents[1]/"noesis/ds9_runtime_core.py").read_text()
    assert 'if xc3d_world_quality_enabled(pipeline.config):\n        from noesis.pipelines.world_quality_adapter import require_world_quality_attachment' in source


@pytest.mark.parametrize("future",[False,True])
def test_t9_original_lineage_bound_current_owner_and_future_birth(native_quality,future):
    from noesis_core.world_service import WorldPublicationRefused
    f=native_quality
    first=f.publish([f.packet(1,1)])
    work,tracks=f.packet(1,2)
    key=next(iter(f.lineage[1]))
    original=f.lineage[1][key]
    if future:
        f.lineage[1][key]=replace(original,original_birth=replace(original.original_birth,original_frame=100))
        before=f.identity.stats()
        with pytest.raises(WorldPublicationRefused,match="original_birth_binding"):f.ingress.stage(work,tracks)
        assert f.identity.stats()==before and f.world.current_snapshot() is first
    else:
        second=f.publish([(work,tracks)])
        assert second.entities[0].entity_id==first.entities[0].entity_id
        assert next(iter(f.ingress.originals.values())).stamp.lifecycle_started_frame==1


def test_t9_queued_reset_does_not_replace_older_original_FIFO_receipt(native_quality):
    f=native_quality
    older=f.packet(1,5)
    f.registries[1].reset_source(1)
    reset=f.packet(1,1)
    assert older[0].continuity.source_epoch==0 and reset[0].continuity.source_epoch==1
    first=f.publish([older])
    old_id=first.entities[0].entity_id
    assert next(iter(f.ingress.originals)).source_epoch==0  # Not latest media registry epoch1.
    second=f.publish([reset])
    assert len(second.entities)==1 and second.entities[0].entity_id!=old_id
    assert next(iter(f.ingress.originals)).source_epoch==1


def original_quality(frame=1,source=1,slot=0,scope=101):
    """REAL immutable contracts/registry, explicitly synthetic metric/clock values."""
    camera=("living-room","kitchen","family-room")[source]
    pts=frame*33_333_333
    original=OriginalCohort("synthetic-quality-run","synthetic-quality-graph",camera,source,0,frame,
        pts,10_000_000_000+pts,"source_sr",scope,slot,source+100)
    registry=TrackingLifecycleRegistry(reappearance_grace_s=.43)
    life=registry.identity_observation(source,9,frame_id=frame,observed_at_us=frame*1000,media_pts_ns=pts)
    track={"source_id":source,"camera_id":camera,"tracker_id":9,"tracker_lifecycle_generation":life.generation,
        "frame_id":frame,"media_pts_ns":pts,"observed_at_us":frame*1000,"bbox":[10.,10.,30.,60.],
        "world":[1.,0.,2.],"world_valid":True,"world_measurement_accepted":True,
        "world_covariance":[.04,0.,0.,0.,.04,0.,0.,0.,.09],"world_resolver_confidence":.7,
        "world_source":"registered_depth","world_frame":"backend_world_m",
        "world_frame_revision":"synthetic-revision","world_transform_sha256":"b"*64,
        "world_calibration_sha256":"a"*64,"world_quantity":"ground_footprint","world_posture":"standing",
        "world_support_state":"floor"}
    continuity=registry.update_frame(source_id=source,camera_id=camera,frame_id=frame,observed_at_us=frame*1000,
        media_pts_ns=pts,tracks=[track])
    retention=freeze_owner_retention(continuity,registry.identity_retention_keys(source,media_pts_ns=pts,
        max_gap_ns=430_000_000),430_000_000,original)
    private=PrivateAssociationWork(source,frame,pts,0,original,None,owner_retention=retention)
    cohort=WorldMeasurementCohort(track_key=f"{source}:9:{life.generation}",source_id=source,camera_id=camera,
        tracker_id=9,tracker_lifecycle_generation=life.generation,frame_id=frame,observed_at_us=frame*1000,
        world_revision="synthetic-revision",calibration_revision="a"*64,world_transform_sha256="b"*64)
    covariance=Matrix3(values=tuple(track["world_covariance"]))
    position=Vector3(x=1.,y=0.,z=2.)
    hypotheses=tuple(WorldMeasurementHypothesis(candidate_id=f"depth-{i}",cohort=cohort,kind="registered_depth",
        position=position,covariance=covariance,anchor="pose_torso_support",depth_support_fraction=.8,
        depth_registration_sigma_m=.4,posture="standing",support_state="floor") for i in range(4))
    measurements=WorldMeasurementSet(cohort=cohort,hypotheses=hypotheses)
    resolution=ResolvedGroundMeasurement(cohort=cohort,status="measured",position=position,covariance=covariance,
        selected_candidate_id="depth-0",selected_kind="registered_depth",contributor_ids=("depth-0",),
        confidence=.7,quality="good",reason="synthetic-current-depth")
    depth=ObjectDepthResult(source_id=source,frame_id=frame,object_id=9,class_id=0,bbox=(10.,10.,30.,60.),
        score=.9,sampling_mode="synthetic-person-patch",status="ok",unit="m",is_metric=True,
        sample_count=12,valid_fraction=.8,depth_median=2.,anchor_source="pose_torso_support",
        anchor_depth_m=2.,anchor_uv=(20.,40.),depth_spread_m=.3,measurement_frame_id=frame,
        measurement_ts_us=pts//1000,measurement_age_us=0,measurement_cached=False,
        depth_tensor_frame_id=frame,depth_tensor_ts_us=pts//1000,depth_tensor_age_frames=0,depth_tensor_age_us=0)
    return track,life,measurements,resolution,depth,private,continuity


def original_body_example():
    from noesis.pipelines.world_quality_adapter import capture_body_support
    track,life,*rest=original_quality()
    track.update(image_size=[1920,1080],confidence=.91,tracker_confidence=.82)
    points=[[20.,30.+i,.9] for i in range(17)]
    body=capture_body_support(track,life,points,pose_is_current=True)
    assert body is not None
    return track,life,rest,points,body


def test_t10_body_original_current_scalar_copy_survives_actual_finalizer():
    from noesis.pipelines import hooks
    from dataclasses import FrozenInstanceError
    track,life,rest,points,body=original_body_example()
    measurements,resolution,depth,private,continuity=rest
    # This is the actual callback's deferred finalizer after household overlay.
    result,=hooks._finalize_world_quality_inputs([(track,life,measurements,resolution,depth,body)])
    work=freeze_quality_work(private,continuity,(result,))
    assert work.denial is None and work.rows[0].evidence.body is body
    assert body.lifecycle is life and body.image_size==(1920,1080)
    assert body.bbox_xywh==(10.,10.,30.,60.) and len(body.keypoints)==17
    assert body.detector_confidence==.91 and body.tracker_confidence==.82
    points[0][0]=999.;track['bbox'][0]=999.;track['image_size'][0]=99
    assert body.keypoints[0][0]==20. and body.bbox_xywh[0]==10. and body.image_size[0]==1920
    with pytest.raises(FrozenInstanceError):body.keypoints=()
    assert 'body' not in track and 'keypoints' not in track  # Private handoff only.


@pytest.mark.parametrize('bad',[
    'cached','missing','oversized','short','nan','boolean','out_of_raster','bad_score',
    'empty_box','outside_box','bad_size','malformed',
])
def test_t10_body_unavailable_is_not_a_vote_or_publication_failure(bad):
    from noesis.pipelines.world_quality_adapter import capture_body_support
    track,life,rest,points,_=original_body_example()
    current=True
    if bad=='cached':current=False
    elif bad=='missing':points=None
    elif bad=='oversized':points=points*2
    elif bad=='short':points=points[:-1]
    elif bad=='nan':points[0][0]=float('nan')
    elif bad=='boolean':points[0][0]=True
    elif bad=='out_of_raster':points[5][0]=1921.
    elif bad=='bad_score':points[11][2]=1.1
    elif bad=='empty_box':track['bbox'][2]=0.
    elif bad=='outside_box':track['bbox'][0]=-1.
    elif bad=='bad_size':track['image_size']=[True,1080]
    else:points[0]=object()
    body=capture_body_support(track,life,points,pose_is_current=current)
    assert body is None
    measurements,resolution,depth,private,continuity=rest
    value=capture_quality_input(track,life,measurements,resolution,depth,body)
    work=freeze_quality_work(private,continuity,(value,))
    assert work.denial is None and work.rows[0].evidence.body is None


def test_t10_body_foreign_lifecycle_and_changed_raster_refused_before_owner():
    track,life,rest,points,body=original_body_example()
    measurements,resolution,depth,private,continuity=rest
    with pytest.raises(ValueError,match='body_lifecycle_mismatch'):
        capture_quality_input(track,life,measurements,resolution,depth,replace(body,lifecycle=replace(life,frame_id=9)))
    with pytest.raises(ValueError,match='body_raster_changed'):
        capture_quality_input({**track,'image_size':[1280,720]},life,measurements,resolution,depth,body)
    value=capture_quality_input(track,life,measurements,resolution,depth,body)
    forged=replace(value,body=replace(body,lifecycle=replace(life,frame_id=9)))
    assert freeze_quality_work(private,continuity,(forged,)).denial=='world_quality_body_lifecycle_mismatch'


def test_t10_body_actual_callback_uses_same_extraction_and_exact_current_flag():
    tree=ast.parse((Path(__file__).resolve().parents[1]/'noesis/pipelines/hooks.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
        and n.func.id=='capture_body_support']
    call,=calls
    assert [ast.unparse(a) for a in call.args]==['public_track','identity_observation','pose_kpts_abs']
    assert ast.unparse(call.keywords[0].value)=="pose_current_evidence.get('current') is True"
    # Nothing captures SDK objects, arrays or wall-clock time in this helper.
    from noesis.pipelines.world_quality_adapter import capture_body_support
    import inspect
    source=inspect.getsource(capture_body_support)
    assert all(name not in source for name in ('obj_meta','time.time','numpy','cuda','extract_pose'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_AnalyticsTelemetryProcessor')
    enqueue=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_enqueue_tracking_publication')
    writes=[n for n in ast.walk(enqueue) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
        and n.func.id=='replace' and any(k.arg=='ingress_monotonic_s' for k in n.keywords)]
    write,=writes
    keywords={k.arg:k.value for k in write.keywords}
    assert ast.unparse(write.args[0])=='quality_work'
    assert ast.unparse(keywords['ingress_monotonic_s'])=='original_ingress'
    ingress,=[n for n in ast.walk(enqueue) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='original_ingress' for t in n.targets)]
    assert ast.unparse(ingress.value)=='time.monotonic()'
    availability=keywords['owner_availability']
    assert isinstance(availability,ast.IfExp) and ast.unparse(availability.orelse)=='None'
    assert ast.unparse(availability.test)=="(getattr(self.pipeline, 'config', {}) or {}).get('xc3d_image_regions', {}).get('enabled') is True"
    snapshot=availability.body
    assert isinstance(snapshot,ast.Call) and ast.unparse(snapshot.func)=='freeze_owner_availability'
    assert [ast.unparse(a) for a in snapshot.args]==[
        'xc3d_private_evidence','continuity',"getattr(self.pipeline, 'source_progress_monitor', None)"]
    assert {k.arg:ast.unparse(k.value) for k in snapshot.keywords}=={'ingress_monotonic_s':'original_ingress'}
    assert ingress.lineno < write.lineno
    gate,=[n for n in ast.walk(enqueue) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and n.func.attr=='_publish_gate_due']
    assert write.lineno < gate.lineno
    assert write.lineno < next(n.lineno for n in ast.walk(enqueue) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id=='_TrackingPublicationWork')


def test_t10_body_numpy_scalar_input_becomes_owned_python_only():
    import numpy as np
    from noesis.pipelines.world_quality_adapter import capture_body_support
    track,life,_,points,_=original_body_example()
    array=np.asarray(points,dtype=np.float32)
    body=capture_body_support(track,life,array,pose_is_current=True)
    assert body is not None and all(type(v) is float for row in body.keypoints for v in row)
    array[:]=0
    assert body.keypoints[0][0]==20.


def test_t9_detach_reuses_ALL_original_contracts_once_no_covariance_or_time_rewrite():
    track,life,measurements,resolution,depth,private,continuity=original_quality()
    value=capture_quality_input(track,life,measurements,resolution,depth)
    work=freeze_quality_work(private,continuity,(value,))
    row,=work.rows
    assert work.denial is None and row.denial is None and row.current_measured and row.depth_current
    assert row.evidence.measurements is measurements and len(measurements.hypotheses)==4
    assert row.evidence.resolution is resolution and row.evidence.depth is depth
    assert row.original is private.original and row.evidence.lifecycle is life
    track["world"][0]=99.;track["world_covariance"][0]=999.
    assert dict(row.evidence.world)["world"]==(1.,0.,2.)
    assert dict(row.evidence.world)["world_covariance"][0]==.04
    assert not hasattr(work,"sdk_frame") and not hasattr(row,"membership_dto")


@pytest.mark.parametrize("changes",[
    {"measurement_cached":True},{"measurement_age_us":1},{"depth_tensor_age_frames":1},
    {"depth_tensor_age_us":1},{"depth_tensor_frame_id":2},{"measurement_frame_id":2},
    {"depth_tensor_ts_us":1},{"measurement_ts_us":1},{"source_id":2},{"object_id":10},
    {"is_metric":False},
])
def test_t9_cached_mismatched_or_unsupported_depth_remains_original_and_never_fresh(changes):
    track,life,measurements,resolution,depth,private,continuity=original_quality()
    changed=replace(depth,**changes)
    value=capture_quality_input(track,life,measurements,resolution,changed)
    row,=freeze_quality_work(private,continuity,(value,)).rows
    assert not row.depth_current and not row.current_measured
    assert row.denial=="world_quality_depth_not_exact_current" and row.evidence.depth is changed


@pytest.mark.parametrize("change",[{"world_measurement_accepted":False},{"world_source":"anchor_hold"},
    {"world_source":"cv_prediction"},{"world_covariance":None},{"world_covariance":[1.,2.]}])
def test_t9_prediction_hold_or_missing_covariance_never_mints_current_measurement(change):
    track,life,measurements,resolution,depth,private,continuity=original_quality()
    track.update(change)
    row,=freeze_quality_work(private,continuity,(capture_quality_input(track,life,measurements,resolution,depth),)).rows
    assert not row.current_measured and row.original.pts_ns==33_333_333


@pytest.mark.parametrize("fault",["borrowed","nan","foreign_frame","owner_missing","owner_future","duplicate","capacity"])
def test_t9_exact_cohort_denial_is_queued_not_silently_dropped_or_retimed(fault):
    track,life,measurements,resolution,depth,private,continuity=original_quality()
    if fault in {"borrowed","nan","foreign_frame"}:
        if fault=="borrowed": track["world"]=SimpleNamespace(sdk_surface=True)
        elif fault=="nan": track["world_covariance"][0]=float("nan")
        else: track["frame_id"]=2
        with pytest.raises(ValueError): capture_quality_input(track,life,measurements,resolution,depth)
        return
    value=capture_quality_input(track,life,measurements,resolution,depth)
    if fault=="owner_missing": private=replace(private,owner_retention=None)
    if fault=="owner_future": private=replace(private,owner_retention=replace(private.owner_retention,frame_id=2))
    inputs=(value,)* (2 if fault=="duplicate" else 129 if fault=="capacity" else 1)
    work=freeze_quality_work(private,continuity,inputs)
    assert work.denial and work.rows==() and work.continuity is continuity


def test_t9_partial_repeated_reordered_source_cohorts_keep_exact_slot_frame_scope_PTS():
    works=[]
    for frame,source,slot in ((11,1,1),(7,2,2),(10,1,0)):
        track,life,measurements,resolution,depth,private,continuity=original_quality(frame,source,slot,scope=500)
        works.append(freeze_quality_work(private,continuity,(capture_quality_input(track,life,measurements,resolution,depth),)))
    assert [(work.rows[0].original.source_id,work.rows[0].original.slot,work.rows[0].original.original_frame)
        for work in works]==[(1,1,11),(2,2,7),(1,0,10)]
    assert all(work.rows[0].original.buffer_scope==500 for work in works)
    assert works[1].rows[0].original.pts_ns==7*33_333_333  # A's later clock never renews B.


def test_t9_actual_hook_resolver_capture_is_private_before_public_scalar_copy_and_never_bypasses_owner_call_shape():
    path=Path(__file__).resolve().parents[1]/"noesis/pipelines/hooks.py"
    tree=ast.parse(path.read_text())
    processor=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=="_AnalyticsTelemetryProcessor")
    native=next(node for node in processor.body if isinstance(node,ast.FunctionDef) and node.name=="_handle_servicemaker_frame_admitted")
    calls=[node for node in ast.walk(native) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
    prepare=next(node for node in calls if node.func.attr=="_prepare_tracking_cohort")
    enqueue=next(node for node in calls if node.func.attr=="_enqueue_tracking_publication")
    assert not {"world_quality_inputs","world_quality_denial"}&{keyword.arg for keyword in prepare.keywords}
    assert {"world_quality_inputs","world_quality_denial"}<={keyword.arg for keyword in enqueue.keywords}
    source=path.read_text()
    assert 'quality_resolver = public_track.pop("_xc3d_quality_resolver"' in source
    assert '"_xc3d_quality_resolver",' in source
    assert "world_quality_work=" in source and "freeze_quality_work(" in source


@pytest.mark.parametrize("enabled,denial",[(False,None),(False,"synthetic-private-denial"),(True,None),(True,"synthetic-private-denial")])
def test_t9_actual_enqueue_detaches_to_existing_work_actor_and_feature_OFF_identical(monkeypatch,enabled,denial):
    for name in ("pyservicemaker","noesis_pose_meta_ext","noesis_analytics_meta_ext","noesis_v3dt_meta_ext",
        "noesis_reid_meta_ext","noesis_depth_meta_ext","noesis_depth_tracking_tensor_ext"):
        monkeypatch.setitem(sys.modules,name,None)
    from noesis.pipelines import hooks
    track,life,measurements,resolution,depth,private,continuity=original_quality()
    value=capture_quality_input(track,life,measurements,resolution,depth)
    cfg={"xc3d_world_quality":{"contract":"noesis.xc3d.world_quality.v1","enabled":True}} if enabled else {}
    outputs=[]
    processor=SimpleNamespace(pipeline=SimpleNamespace(config=cfg),
        _frame_timestamp_us=lambda _frame:33333,_tracking_interval_for_frame=lambda _count:0.,
        _publish_gate_due=lambda *_args,**_kwargs:True,_last_tracking_enqueue_ts_by_sensor={},
        _last_tracking_enqueue_count_by_sensor={},_canonical_publication_backlogged=lambda:False,
        _publish_tracking_work=outputs.append)
    assert hooks._AnalyticsTelemetryProcessor._enqueue_tracking_publication(processor,
        sensor_id=1,camera_id="kitchen",frame_meta=SimpleNamespace(frame_number=1),tracks=[track],footpoints=[],
        now_ts=.001,temporal_contract={"observed_at_us":1000,"media_pts_ns":private.media_pts_ns},
        continuity=continuity,force=True,xc3d_private_evidence=private,world_quality_inputs=(value,),world_quality_denial=denial)
    packet,=outputs
    assert packet.tracks==[track] and packet.tracks[0] is not track
    assert packet.continuity is continuity and packet.xc3d_private_evidence is private
    if enabled: assert packet.world_quality_work.raw_media_pts_ns == private.media_pts_ns
    if not enabled: assert packet.world_quality_work is None
    elif denial: assert packet.world_quality_work.denial==denial and packet.world_quality_work.rows==()
    else:
        assert packet.world_quality_work.rows[0].evidence is value
        assert packet.world_quality_work.rows[0].evidence.measurements is measurements
    track["world"][0]=999.
    assert packet.tracks[0]["world"][0]==1.  # actual public scalar copy remains original


@pytest.mark.parametrize("enabled",[False,True])
def test_t9_actual_resolver_hook_once_original_four_hypotheses_no_public_diagnostics_or_synthesized_quality(monkeypatch,enabled):
    for name in ("pyservicemaker","noesis_pose_meta_ext","noesis_analytics_meta_ext","noesis_v3dt_meta_ext",
        "noesis_reid_meta_ext","noesis_depth_meta_ext","noesis_depth_tracking_tensor_ext"):
        monkeypatch.setitem(sys.modules,name,None)
    from noesis.pipelines import hooks
    from unittest.mock import Mock
    track,_life,measurements,resolution,_depth,_private,_continuity=original_quality()
    cfg={"xc3d_world_quality":{"contract":"noesis.xc3d.world_quality.v1","enabled":True}} if enabled else {}
    resolve=Mock(return_value=resolution)
    processor=SimpleNamespace(pipeline=SimpleNamespace(config=cfg),_world_resolver=SimpleNamespace(resolve=resolve),
        _world_resolver_diagnostics_enabled=False,_world_resolver_diag_max_candidates=4)
    point,result=hooks._AnalyticsTelemetryProcessor._apply_resolved_world_measurement(processor,track,measurements)
    resolve.assert_called_once_with(measurements)
    assert result is resolution and tuple(point)==(1.,0.,2.) and "world_resolver" not in track
    assert track["world_resolver_confidence"]==.7
    if enabled: assert track["_xc3d_quality_resolver"]==(measurements,resolution)
    else: assert "_xc3d_quality_resolver" not in track


@pytest.mark.parametrize("field,value",[("world_frame","foreign-frame"),("world_frame_revision","foreign-revision"),
    ("world_transform_sha256","c"*64),("world_calibration_sha256","d"*64)])
def test_t9_frame_binding_conflict_preserves_both_originals_but_denies_current_measurement(field,value):
    track,life,measurements,resolution,depth,private,continuity=original_quality()
    track[field]=value
    input=capture_quality_input(track,life,measurements,resolution,depth)
    row,=freeze_quality_work(private,continuity,(input,)).rows
    assert row.denial=="world_quality_emitted_frame_binding_mismatch" and not row.current_measured
    assert dict(row.evidence.world)[field]==value and row.evidence.measurements is measurements


@pytest.mark.parametrize("field,value",[("world",[1.,2.]),("world_covariance",[False]+[0.]*8)])
def test_t9_malformed_emitted_point_or_covariance_cannot_get_private_current_flag(field,value):
    track,life,measurements,resolution,depth,private,continuity=original_quality()
    track[field]=value
    row,=freeze_quality_work(private,continuity,(capture_quality_input(track,life,measurements,resolution,depth),)).rows
    assert not row.current_measured and dict(row.evidence.world)[field]==tuple(value)


@pytest.fixture
def zero_native(tmp_path):
    f=NativeQualityFixture(tmp_path,source_ids=(0,1,2))
    try:yield f
    finally:f.world.close()


def zero_native_work(f,source=0):
    """Actual six-scalar/acquisition/lifecycle row SHAPE, SYNTHETIC values, no SDK."""
    from noesis.pipelines.source_acquisition import SourceAcquisitionState, ScopedFrameReceipts, current_frame_key
    path=Path(__file__).with_name("test_xc3d_source_acquisition.py")
    spec=importlib.util.spec_from_file_location("t9_zero_native_shapes",path)
    shapes=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shapes)  # helpers only; no old tests/setup/SDK invoked
    row=shapes.row(frame=0,source=source,scope=101)
    row.update(post_tracker_frame=0,original_ntp_ns=0,post_tracker_ntp_ns=0)
    frame=shapes.frame(row)
    key=current_frame_key(frame)
    assert key==(0,source,source+100,0,0,0)
    native_lifecycle=shapes.lifecycle(row,barrier=True)
    from noesis.pipelines import hooks
    shadows,terminated=hooks._tracker_lifecycle_ids_for_frame(native_lifecycle,
        source_id=source,frame_id=0,media_pts_ns=0,record_counters=False)
    batch=object()
    scope=ScopedFrameReceipts(batch,{"scope":101,"original_buffer_scope":101,"rows":[row]})
    original=SourceAcquisitionState(f.identity.run_id,"synthetic-quality-graph","source_pts").observe_native(scope.lookup_key(batch,key))
    temporal=hooks._frame_temporal_contract(frame,observed_at_s=time.time())
    continuity=f.registries[source].update_frame(source_id=source,camera_id=f.labels[source],frame_id=0,
        observed_at_us=temporal["observed_at_us"],media_pts_ns=temporal["media_pts_ns"],tracks=[],
        shadow_tracker_ids=shadows,terminated_tracker_ids=terminated)
    private=PrivateAssociationWork(source,0,0,0,original,None,terminated)
    work=freeze_quality_work(private,continuity,(),raw_media_pts_ns=temporal["media_pts_ns"])
    assert temporal["media_pts_ns"]==0 and continuity.media_pts_ns is None and private.owner_retention is None
    return work,frame,temporal


def test_t9_zero_actual_temporal_registry_enqueue_actor_publisher_and_Node22(zero_native,monkeypatch):
    import threading
    for name in ("pyservicemaker","noesis_pose_meta_ext","noesis_analytics_meta_ext","noesis_v3dt_meta_ext",
        "noesis_reid_meta_ext","noesis_depth_meta_ext","noesis_depth_tracking_tensor_ext"):
        monkeypatch.setitem(sys.modules,name,None)
    from noesis.pipelines import hooks
    from noesis_core.runtime_publication import RuntimePublicationGate
    f=zero_native
    work,frame,temporal=zero_native_work(f)
    pipeline=SimpleNamespace(config={"xc3d_world_quality":{"contract":"noesis.xc3d.world_quality.v1","enabled":True}},
        xc3d_world_quality_ingress=f.ingress)
    queued=[]
    processor=SimpleNamespace(pipeline=pipeline,tracking_pub=f.publisher,publication_gate=RuntimePublicationGate(),
        _tracking_lifecycle=f.registries[0],_source_frame_lock=threading.RLock(),_source_frame_inflight={},
        _last_source_frame_by_sensor={},_source_epoch_by_sensor={},_stamp_footpoint_lifecycles=lambda *_:None,
        _tracking_mode_is_mv3dt=lambda:False,_commit_enqueued_world_output_watermarks=lambda *_a,**_k:None,
        _frame_timestamp_us=lambda _:temporal["observed_at_us"],_tracking_interval_for_frame=lambda _:0.,
        _publish_gate_due=lambda *_a,**_k:True,_last_tracking_enqueue_ts_by_sensor={},
        _last_tracking_enqueue_count_by_sensor={},_canonical_publication_backlogged=lambda:False,
        _last_tracking_publish_ts_by_sensor={},_last_tracking_count_by_sensor={},bev_renderer=None)
    continuity,private=hooks._AnalyticsTelemetryProcessor._prepare_tracking_cohort(processor,sensor_id=0,
        camera_id="living-room",frame_id=0,observed_at_us=temporal["observed_at_us"],media_pts_ns=0,
        tracks=[],footpoints=[],xc3d_private_evidence=work.private)
    assert private.owner_retention is None and private.denial is None
    def publish(packet):
        queued.append(packet)
        hooks._AnalyticsTelemetryProcessor._publish_tracking_work(processor,packet)
    processor._publish_tracking_work=publish
    before=f.identity.stats()
    assert hooks._AnalyticsTelemetryProcessor._enqueue_tracking_publication(processor,sensor_id=0,camera_id="living-room",
        frame_meta=frame,tracks=[],footpoints=[],now_ts=time.time(),temporal_contract=temporal,continuity=continuity,
        force=True,xc3d_private_evidence=private)
    packet,=queued
    assert packet.world_quality_work.raw_media_pts_ns==0 and packet.temporal_contract["media_pts_ns"] is None
    assert not hasattr(packet,"frame_meta") and temporal["media_pts_ns"]==0
    assert replace(f.identity.stats(),clock_frontiers=before.clock_frontiers)==before
    assert f.identity.stats().pending_reservations==0 and f.identity.stats().clock_frontiers==1
    assert f.world.current_snapshot().entities==() and f.ingress.watermarks[0]==(0,0,0,0)
    assert f.ingress._context.comparability_receipt_id is None
    from dataclasses import fields
    interval=f.ingress._context.intervals[0]
    assert tuple(getattr(interval,field.name) for field in fields(interval))[1:]==(0.,0.,0.,0.)
    assert f.wire[0]["tracks"]==[] and f.wire[0]["media_pts_ns"] is None
    assert "tracker_lifecycle_evidence" not in f.wire[0]
    positive=f.publish([f.packet(0,1)])
    assert len(positive.entities)==1
    tracking=[row for row in f.wire if row["type"]=="tracking"]
    assert [row["frame_id"] for row in tracking]==[0,1]
    assert [row["tracking_publication_sequence"] for row in tracking]==[0,1]
    assert tracking[1]["media_pts_ns"]==33_333_333 and tracking[1]["track_count"]==1
    result=f.examples.actual_node_wire(f,f.wire,"native-t9-zero-then-positive")
    assert [len(row["people"]) for row in result["snapshots"]]==[0,1]


@pytest.mark.parametrize("fault",["occupied","active","shadow","terminated","tombstone","epoch","clock_epoch",
    "frame","original_frame","source","camera","raw","private_pts","original_pts","scope","slot","sdk",
    "graph","missing_original","private_denial","owner_receipt"])
def test_t9_zero_negative_receipts_refuse_before_identity_World_or_wire(zero_native,fault):
    from noesis_core.world_service import WorldPublicationRefused
    f=zero_native
    work,_frame,_temporal=zero_native_work(f)
    private,continuity=work.private,work.continuity
    inputs,tracks,raw=(),[],0
    if fault=="occupied":
        track,life,measurements,resolution,depth,*_=original_quality()
        inputs=(capture_quality_input(track,life,measurements,resolution,depth),)
        tracks=[track]
    elif fault in {"active","shadow","terminated","tombstone"}:
        field={"active":"active_lifecycles","shadow":"shadow_lifecycles","terminated":"terminated_tracker_ids","tombstone":"tombstones"}[fault]
        continuity=replace(continuity,**{field:(object(),)})
    elif fault=="epoch":continuity=replace(continuity,source_epoch=1)
    elif fault=="clock_epoch":private=replace(private,clock_epoch=1,original=replace(private.original,source_clock_epoch=1))
    elif fault=="frame":continuity=replace(continuity,frame_id=1)
    elif fault=="source":private=replace(private,source_id=1)
    elif fault=="camera":continuity=replace(continuity,camera_id="kitchen")
    elif fault=="raw":raw=None
    elif fault=="private_pts":private=replace(private,media_pts_ns=True)
    elif fault=="missing_original":private=replace(private,original=None)
    elif fault=="private_denial":private=replace(private,original=None,denial="exact_frame_private_receipt_unavailable")
    elif fault=="owner_receipt":private=replace(private,owner_retention=object())
    else:
        field,value={"original_frame":("original_frame",1),"original_pts":("pts_ns",1),"scope":("buffer_scope",0),
            "slot":("slot",9),"sdk":("sdk_source_id",True),"graph":("graph_id","")}[fault]
        private=replace(private,original=replace(private.original,**{field:value}))
    bad=freeze_quality_work(private,continuity,inputs,raw_media_pts_ns=raw)
    before=f.identity.stats()
    with pytest.raises(WorldPublicationRefused) as refused:f.ingress.stage(bad,tracks)
    if fault=="private_denial":assert refused.value.reason=="exact_frame_private_receipt_unavailable"
    assert f.identity.stats()==before and f.world.current_snapshot() is None and f.wire==[]
    assert f.ingress.times==f.ingress.watermarks==f.ingress.originals=={}


@pytest.mark.parametrize("state",["watermark","time","lineage","member","repeated"])
def test_t9_zero_prior_source_state_repeated_or_reordered_startup_cannot_renew(zero_native,state):
    from noesis_core.world_service import WorldPublicationRefused
    f=zero_native
    work,*_=zero_native_work(f)
    if state=="watermark":f.ingress.watermarks[0]=(0,0,1,33_333_333)
    elif state=="time":f.ingress.times[0]=.1
    elif state=="lineage":f.lineage[0][(0,9,1)]=object()
    elif state=="member":
        positive=f.packet(0,1)
        f.publish([positive])
        f.ingress.watermarks.pop(0)
        f.ingress.times.pop(0)
        f.lineage[0].clear()
    else:
        f.ingress.stage(work,[])
        f.publisher.publish(0,[],frame_metadata={"camera_id":"living-room","frame_id":0,"source_epoch":0,
            "observed_at_us":work.continuity.observed_at_us,"media_pts_ns":None})
    before=f.identity.stats(),f.world.current_snapshot(),len(f.wire),dict(f.ingress.times),dict(f.ingress.watermarks)
    with pytest.raises(WorldPublicationRefused):f.ingress.stage(work,[])
    assert (f.identity.stats(),f.world.current_snapshot(),len(f.wire),dict(f.ingress.times),dict(f.ingress.watermarks))==before


def test_t9_zero_empty_other_source_preserves_real_member_and_original_time(zero_native):
    f=zero_native
    first=f.publish([f.packet(2,1)])
    original=next(iter(f.ingress.originals.values()))
    work,*_=zero_native_work(f)
    f.ingress.stage(work,[])
    f.publisher.publish(0,[],frame_metadata={"camera_id":"living-room","frame_id":0,"source_epoch":0,
        "observed_at_us":work.continuity.observed_at_us,"media_pts_ns":None})
    retained=f.world.current_snapshot().entities[0]
    assert retained.entity_id==first.entities[0].entity_id and retained.position==first.entities[0].position
    assert retained.observed_at_us==first.entities[0].observed_at_us
    assert next(iter(f.ingress.originals.values())) is original and f.ingress.times[2]==33_333_333/1e9


@pytest.mark.parametrize("changed",["raw_zero","observed","camera","frame","epoch","occupied"])
def test_t9_zero_changed_public_metadata_never_consumes_spatial_preparation(zero_native,changed):
    from noesis_core.world_service import WorldPublicationRefused
    f=zero_native
    work,*_=zero_native_work(f)
    f.ingress.stage(work,[])
    metadata={"camera_id":"living-room","frame_id":0,"source_epoch":0,
        "observed_at_us":work.continuity.observed_at_us,"media_pts_ns":None}
    tracks=[]
    if changed=="raw_zero":metadata["media_pts_ns"]=0
    elif changed=="occupied":tracks=[{"tracker_id":9}]
    else:metadata[{"observed":"observed_at_us","camera":"camera_id","frame":"frame_id","epoch":"source_epoch"}[changed]]=1
    before=f.identity.stats()
    with pytest.raises(WorldPublicationRefused):f.ingress.prepare(0,tracks,metadata=metadata)
    assert f.identity.stats()==before and f.world.current_snapshot() is None and f.wire==[]


@pytest.mark.parametrize("raw",[None,0,True,33_333_334])
def test_t9_zero_next_positive_requires_exact_original_raw_and_owner_receipt(zero_native,raw):
    from noesis_core.world_service import WorldPublicationRefused
    f=zero_native
    good,tracks=f.packet(0,1)
    bad=freeze_quality_work(good.private,good.continuity,tuple(row.evidence for row in good.rows),raw_media_pts_ns=raw)
    before=f.identity.stats()
    with pytest.raises(WorldPublicationRefused):f.ingress.stage(bad,tracks)
    assert f.identity.stats()==before and f.world.current_snapshot() is None


def actual_lineage_callback(pipeline):
    """Compile ONLY the actual runtime lambda, without runtime/SDK imports."""
    from noesis.pipelines.world_quality_adapter import snapshot_swin_lineage
    path=Path(__file__).resolve().parents[1]/"noesis/ds9_runtime_core.py"
    tree=ast.parse(path.read_text())
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Name) and node.func.id=="OrderedQualityWorldIngress"]
    assert len(calls)==1
    callback=next(keyword.value for keyword in calls[0].keywords if keyword.arg=="origin_bindings")
    assert isinstance(callback,ast.Lambda)
    return eval(compile(ast.Expression(callback),str(path),"eval"),
        {"pipeline":pipeline,"snapshot_swin_lineage":snapshot_swin_lineage})


@pytest.fixture
def native_lineage(zero_native):
    """ACTUAL ingress/cache/FIFO producer; synthetic source scalars/geometry only."""
    from collections import deque
    from noesis.pipelines.source_acquisition import SelectedSwinIngress, SourceAcquisitionState
    from noesis_core.runtime_publication import RuntimePublicationGate
    f=zero_native
    pipeline=SimpleNamespace(xc3d_world_quality_ingress=f.ingress,
        xc3d_private_owner_state=SourceAcquisitionState(f.identity.run_id,"synthetic-quality-graph"),
        xc3d_private_owner_diagnostics=deque(maxlen=256))
    callback=actual_lineage_callback(pipeline)  # Created BEFORE native hook installs Swin ingress.
    from noesis.pipelines.source_acquisition import AcquisitionUnavailable
    with pytest.raises(AcquisitionUnavailable,match="selected_swin_ingress_unavailable"):callback()
    pipeline.xc3d_swin_ingress=SelectedSwinIngress(f.identity.run_id,"synthetic-quality-graph",
        f.profile,None,None,timestamp_origin="source_pts")
    f.ingress.origin_bindings=callback
    processor=SimpleNamespace(pipeline=pipeline,tracking_pub=f.publisher,publication_gate=RuntimePublicationGate(),
        _stamp_footpoint_lifecycles=lambda *_:None,_commit_enqueued_world_output_watermarks=lambda *_a,**_k:None,
        _last_tracking_publish_ts_by_sensor={},_last_tracking_count_by_sensor={},bev_renderer=None)
    return SimpleNamespace(f=f,pipeline=pipeline,callback=callback,processor=processor)


def lineage_packet(native,source,frame,**kwargs):
    """Real public retention and SourceAcquisitionState original, no seeded birth."""
    f=native.f
    packet,tracks=f.packet(source,frame,**kwargs)
    pts=frame*33_333_333
    row={"receipt_valid":True,"source_id":source,"original_source_id":source,
        "sdk_source_id":source+100,"original_sdk_source_id":source+100,
        "slot":0,"original_batch_id":0,"scope":frame+101,"original_buffer_scope":frame+101,
        "original_frame":frame,"post_tracker_frame":frame,
        "original_pts_ns":pts,"post_tracker_pts_ns":pts,
        "original_ntp_ns":10_000_000_000+pts,"post_tracker_ntp_ns":10_000_000_000+pts}
    original=native.pipeline.xc3d_swin_ingress.state.observe_native(row)
    registry=f.registries[source]
    retention=freeze_owner_retention(packet.continuity,registry.identity_retention_keys(source,
        media_pts_ns=pts,max_gap_ns=430_000_000),430_000_000,original)
    private=replace(packet.private,original=original,owner_retention=retention)
    return freeze_quality_work(private,packet.continuity,tuple(row.evidence for row in packet.rows)),tracks


def publish_lineage(native,packet,tracks):
    """Unchanged actual hook orders World publication BEFORE sole cache.bind_owner."""
    from noesis.pipelines import hooks
    source=packet.continuity.source_id
    native.processor._tracking_lifecycle=native.f.registries[source]
    footpoints=[hooks.Footpoint(tracker_id=track["tracker_id"],
        tracker_lifecycle_generation=track["tracker_lifecycle_generation"],frame_id=packet.continuity.frame_id,
        world_x=track["world"][0] if track.get("world") is not None else None,
        world_y=track["world"][1] if track.get("world") is not None else None,
        world_z=track["world"][2] if track.get("world") is not None else None) for track in tracks]
    work=hooks._TrackingPublicationWork(source_id=source,camera_id=packet.continuity.camera_id,
        frame_id=packet.continuity.frame_id,observed_at_us=packet.continuity.observed_at_us,
        now_ts=time.time(),timestamp_us=packet.private.media_pts_ns//1000,tracks=tracks,footpoints=footpoints,
        temporal_contract={"media_pts_ns":packet.continuity.media_pts_ns},continuity=packet.continuity,
        source_epoch=packet.continuity.source_epoch,xc3d_private_evidence=packet.private,world_quality_work=packet)
    hooks._AnalyticsTelemetryProcessor._publish_tracking_work(native.processor,work)
    return native.f.world.current_snapshot()


def test_t10_body_actual_postpublication_hook_binds_genuine_original_swin(native_lineage):
    from noesis.pipelines.source_acquisition import OriginalSwinExtraction
    from reid.mv3dt_person_association import AcquisitionTime
    native=native_lineage
    packet,tracks=lineage_packet(native,1,1)
    original=packet.private.original
    cache=native.pipeline.xc3d_swin_ingress.cache
    assert cache.record(OriginalSwinExtraction(original,9,native.f.profile,(1.,)+(0.,)*255,100.))
    assert native.f.ingress.originals=={} and not cache.members[1]
    publish_lineage(native,packet,tracks)  # Actual hook -> publisher -> owner -> appearance.
    observation,=native.f.ingress.originals.values()
    at=AcquisitionTime(observation.stamp.acquired_at_s,observation.stamp.clock_domain,observation.stamp.clock_basis)
    use=cache.use(observation,at)
    assert use is not None and use.appearance.stamp==observation.stamp and use.fresh_extraction
    assert use.appearance_ingress_monotonic_s==100.
    later,tracks=lineage_packet(native,1,2)
    publish_lineage(native,later,tracks)  # No new extraction: do not renew original frame/time.
    current,=native.f.ingress.originals.values()
    retained=cache.use(current,replace(at,at_s=current.stamp.acquired_at_s))
    assert retained is not None and not retained.fresh_extraction
    assert retained.appearance is use.appearance and retained.appearance_ingress_monotonic_s==100.
    assert retained.appearance.stamp.frame_id==1 and current.stamp.frame_id==2


def test_t10_body_failed_publisher_never_binds_original_appearance(native_lineage,monkeypatch):
    from noesis.pipelines.source_acquisition import OriginalSwinExtraction
    native=native_lineage
    packet,tracks=lineage_packet(native,1,1)
    cache=native.pipeline.xc3d_swin_ingress.cache
    cache.record(OriginalSwinExtraction(packet.private.original,9,native.f.profile,(1.,)+(0.,)*255,100.))
    def fail(*_a,**_k):raise RuntimeError('test_unchanged_failed_publication')
    monkeypatch.setattr(native.f.publisher,'publish',fail)
    with pytest.raises(RuntimeError,match='test_unchanged_failed_publication'):
        publish_lineage(native,packet,tracks)
    assert not cache.members[1] and not cache.owners[1]


def test_t9_lineage_actual_late_callback_empty_birth_FIFO_continuous_and_Node22(native_lineage,monkeypatch):
    from noesis.pipelines.source_acquisition import OriginalSwinCache
    for name in ("pyservicemaker","noesis_pose_meta_ext","noesis_analytics_meta_ext","noesis_v3dt_meta_ext",
        "noesis_reid_meta_ext","noesis_depth_meta_ext","noesis_depth_tracking_tensor_ext"):
        monkeypatch.setitem(sys.modules,name,None)
    native=native_lineage
    cache=native.pipeline.xc3d_swin_ingress.cache
    assert type(cache) is OriginalSwinCache and type(cache.owners) is list
    snapshot=native.callback()
    assert snapshot=={0:{},1:{},2:{}} and all(snapshot[s] is not cache.owners[s] for s in range(3))
    empty,*_=zero_native_work(native.f,source=2)
    assert publish_lineage(native,empty,[]).entities==() and native.callback()=={0:{},1:{},2:{}}
    first,tracks=lineage_packet(native,2,1)
    assert first.rows[0].evidence.lifecycle.disposition=="new" and not cache.owners[2]
    assert len(publish_lineage(native,first,tracks).entities)==1
    key,binding=next(iter(cache.owners[2].items()))
    assert binding.original_birth is first.private.original and binding.lifecycle_started_frame==1
    snapshot=native.callback()
    assert snapshot[2][key] is binding and snapshot[2][key].original_birth is first.private.original
    continued,tracks=lineage_packet(native,2,2)
    assert continued.rows[0].evidence.lifecycle.disposition!="new"
    assert len(publish_lineage(native,continued,tracks).entities)==1
    assert cache.owners[2][key] is binding
    assert native.pipeline.xc3d_private_owner_state.births[2][key] is first.private.original
    assert native.pipeline.xc3d_private_owner_diagnostics[-1][3]["status"]=="accepted_identity_dto_unavailable"
    assert next(iter(native.f.ingress.originals.values())).stamp.lifecycle_started_frame==1
    result=native.f.examples.actual_node_wire(native.f,native.f.wire,"native-t9-actual-lineage")
    assert [len(row["people"]) for row in result["snapshots"]]==[0,1,1]
    cache.owners[2].clear()  # Demonstrate OWNED snapshot; not a positive producer fixture.
    assert snapshot[2][key] is binding and snapshot[2][key].original_birth is first.private.original


def test_t9_lineage_snapshot_once_for_three_source_stage(native_lineage):
    native=native_lineage
    packets=[zero_native_work(native.f,source=s)[0] for s in range(3)]
    calls=[]
    def counted():
        calls.append(1)
        return native.callback()
    native.f.ingress.origin_bindings=counted
    native.f.ingress.stage(tuple(packets),tuple((s,[]) for s in range(3)))
    assert calls==[1] and native.pipeline.xc3d_swin_ingress.cache.owners==[{},{},{}]


@pytest.mark.parametrize("fault,reason",[
    ("missing_ingress","selected_swin_ingress_unavailable"),
    ("wrong_ingress","selected_swin_ingress_unavailable"),
    ("missing_cache","original_swin_cache_unavailable"),
    ("wrong_cache","original_swin_cache_unavailable"),
    ("arity2","lineage_source_arity"),("arity4","lineage_source_arity"),
    ("wrong_source_map","lineage_source_map_unavailable"),
    ("oversized_source_map","lineage_source_map_unavailable"),
    ("key_bool","lineage_member_key_invalid"),("key_shape","lineage_member_key_invalid"),
    ("value","lineage_original_binding_invalid"),("member","lineage_original_binding_invalid"),
    ("start_bool","lineage_original_binding_invalid"),("birth_type","lineage_original_binding_invalid"),
    ("birth_source","lineage_original_source_run_graph_mismatch"),
    ("birth_run","lineage_original_source_run_graph_mismatch"),
    ("birth_graph","lineage_original_source_run_graph_mismatch"),
    ("cache_run","lineage_run_graph_unavailable"),("cache_graph","lineage_run_graph_unavailable"),
])
def test_t9_lineage_actual_cache_malformed_provider_preserves_authority(native_lineage,fault,reason):
    from noesis.pipelines.source_acquisition import SwinOwnerBinding
    from noesis_core.world_service import WorldPublicationRefused
    native=native_lineage
    pipeline,cache=native.pipeline,native.pipeline.xc3d_swin_ingress.cache
    if fault=="missing_ingress":del pipeline.xc3d_swin_ingress
    elif fault=="wrong_ingress":pipeline.xc3d_swin_ingress=SimpleNamespace(cache=cache)
    elif fault=="missing_cache":del pipeline.xc3d_swin_ingress.cache
    elif fault=="wrong_cache":pipeline.xc3d_swin_ingress.cache=SimpleNamespace(owners=cache.owners)
    elif fault.startswith("arity"):cache.owners=[{} for _ in range(int(fault[-1]))]
    elif fault=="wrong_source_map":cache.owners[0]=None
    elif fault=="oversized_source_map":cache.owners[0]={i:None for i in range(257)}
    elif fault in {"cache_run","cache_graph"}:setattr(cache,fault[6:]+"_id","foreign")
    else:
        # Malformed negative input ONLY, never seed positive original lineage.
        key=(0,9,1)
        birth=original_quality(source=0)[5].original
        binding=SwinOwnerBinding(key,birth,1)
        if fault=="key_bool":key=(False,9,1)
        elif fault=="key_shape":key=(0,9)
        elif fault=="value":binding=object()
        elif fault=="member":binding=replace(binding,member=(0,10,1))
        elif fault=="start_bool":binding=replace(binding,lifecycle_started_frame=True)
        elif fault=="birth_type":binding=replace(binding,original_birth=object())
        elif fault.startswith("birth_"):
            field,value={"birth_source":("source_id",1),"birth_run":("run_id","foreign"),
                "birth_graph":("graph_id","foreign")}[fault]
            binding=replace(binding,original_birth=replace(birth,**{field:value}))
        cache.owners[0][key]=binding
    before=native.f.identity.stats(),native.f.world.current_snapshot(),len(native.f.wire)
    work,*_=zero_native_work(native.f,source=2)
    with pytest.raises(WorldPublicationRefused,match=reason) as refused:native.f.ingress.stage(work,[])
    assert refused.value.reason=="world_quality_"+reason
    assert (native.f.identity.stats(),native.f.world.current_snapshot(),len(native.f.wire))==before
    assert native.f.ingress._staged is None and native.f.ingress.watermarks=={}


@pytest.mark.parametrize("fault",["missing_selected_key","none_selected_map","wrong_top_level","provider_raised"])
def test_t9_lineage_no_missing_or_exception_to_empty_fallback(native_lineage,fault):
    from noesis_core.world_service import WorldPublicationRefused
    native=native_lineage
    def negative_provider():
        actual=native.callback()
        if fault=="provider_raised":raise RuntimeError("actual-lineage-provider-failure")
        if fault=="missing_selected_key":del actual[2]
        elif fault=="none_selected_map":actual[2]=None
        elif fault=="wrong_top_level":return native.pipeline.xc3d_swin_ingress.cache.owners
        return actual
    native.f.ingress.origin_bindings=negative_provider
    work,*_=zero_native_work(native.f,source=2)
    before=native.f.identity.stats()
    with pytest.raises(WorldPublicationRefused) as refused:native.f.ingress.stage(work,[])
    expected="actual-lineage-provider-failure" if fault=="provider_raised" else "world_quality_original_lineage_unavailable"
    assert refused.value.reason==expected and native.f.identity.stats()==before
    assert native.f.world.current_snapshot() is None and native.f.wire==[]


@pytest.mark.parametrize("fault",["future_start","future_original_frame","future_original_pts","clock_epoch"])
def test_t9_lineage_genuine_owner_current_future_birth_checks_unchanged(native_lineage,fault):
    from noesis_core.world_service import WorldPublicationRefused
    native=native_lineage
    first,tracks=lineage_packet(native,2,1)
    publish_lineage(native,first,tracks)
    cache=native.pipeline.xc3d_swin_ingress.cache
    key,binding=next(iter(cache.owners[2].items()))
    next_packet,tracks=lineage_packet(native,2,2)
    if fault=="future_start":bad=replace(binding,lifecycle_started_frame=3)
    else:
        field,value={"future_original_frame":("original_frame",3),"future_original_pts":("pts_ns",99_999_999),
            "clock_epoch":("source_clock_epoch",1)}[fault]
        bad=replace(binding,original_birth=replace(binding.original_birth,**{field:value}))
    cache.owners[2][key]=bad  # Negative after ACTUAL sole-writer original issuance.
    assert native.callback()[2][key] is bad
    before=native.f.identity.stats(),native.f.world.current_snapshot(),len(native.f.wire),dict(native.f.ingress.watermarks)
    with pytest.raises(WorldPublicationRefused,match="world_quality_original_birth_binding"):
        native.f.ingress.stage(next_packet,tracks)
    assert (native.f.identity.stats(),native.f.world.current_snapshot(),len(native.f.wire),dict(native.f.ingress.watermarks))==before


def test_t9_person_actual_legacy_provisional_producer_final_row_World_Node22(native_lineage):
    from reid.stable_id_manager import StableIDManager
    from noesis.pipelines import hooks
    from noesis.telemetry.world_contract_adapter import stable_identity_contract
    native=native_lineage
    packet,tracks=lineage_packet(native,1,1)
    manager=object.__new__(StableIDManager)  # Diagnostic producer only; no model/store initialization.
    manager.household_mode=True
    manager._resident_registry=manager._visitor_pool=None
    manager._pending_id_diag={}
    diag=manager._set_track_diag((1,9),event="provisional",reject_reason=None,
        embedding_present=False,pose_present=True,sid_candidate=123,stable_id=123,ts=1.,
        identity_kind="provisional",identity_state="provisional")
    tracks[0].update(stable_id=diag["stable_id"],**stable_identity_contract(manager,sensor_id=1,tracker_id=9,diagnostics=diag))
    evidence=packet.rows[0].evidence
    finalized=hooks._finalize_world_quality_inputs([(tracks[0],evidence.lifecycle,evidence.measurements,
        evidence.resolution,evidence.depth)])
    house=finalized[0].household
    assert (house.state,house.identity_key,house.stable_id,house.visitor_generation)==("provisional",None,None,None)
    assert finalized[0].depth is evidence.depth and finalized[0].measurements is evidence.measurements
    packet=freeze_quality_work(packet.private,packet.continuity,finalized)
    assert len(publish_lineage(native,packet,tracks).entities)==1
    assert tracks[0]["stable_id"]==123 and native.f.wire[0]["tracks"][0]["stable_id"]==123
    assert native.f.identity.stats().pending_reservations==0
    result=native.f.examples.actual_node_wire(native.f,native.f.wire,"native-t9-provisional-compatibility")
    assert [len(row["people"]) for row in result["snapshots"]]==[1]


@pytest.mark.parametrize("fields",[
    {"stable_id":1}, {"stable_id":1,"identity_kind":"unknown"},
    {"stable_id":True,"identity_kind":"provisional"},
    {"stable_id":1,"identity_kind":"provisional","identity_state":"resident"},
    {"stable_id":1,"identity_kind":"provisional","resident_uuid":"resident"},
    {"stable_id":1,"identity_kind":"provisional","visitor_generation":0},
    {"stable_id":1,"identity_kind":"provisional","identity_v2":{"subject_id":"resident:resolved"}},
    {"stable_id":1,"identity_kind":True},
])
def test_t9_person_conflicting_or_missing_household_truth_never_grants(native_lineage,fields):
    from noesis.pipelines import hooks
    native=native_lineage
    packet,tracks=lineage_packet(native,1,1)
    tracks[0].pop("identity_kind",None)
    tracks[0].update(fields)
    e=packet.rows[0].evidence
    before=native.f.identity.stats()
    with pytest.raises(ValueError,match="world_quality_original_household"):
        hooks._finalize_world_quality_inputs([(tracks[0],e.lifecycle,e.measurements,e.resolution,e.depth)])
    assert native.f.identity.stats()==before and native.f.wire==[] and native.f.world.current_snapshot() is None


@pytest.mark.parametrize("kind",["resident","visitor"])
def test_t9_person_resolved_descriptor_preserves_exact_uuid_sid_generation(kind):
    from noesis.pipelines.world_quality_adapter import household_from_final_track
    row={"identity_kind":kind,"identity_state":kind,"stable_id":17,
        "resident_uuid":"original-resident" if kind=="resident" else None,
        "visitor_generation":4 if kind=="visitor" else None}
    value=household_from_final_track(row)
    assert value.stable_id==17 and value.identity_key==("original-resident" if kind=="resident" else "visitor:17:g4")
    assert value.visitor_generation==(4 if kind=="visitor" else None)


def test_t9_person_actual_missing_embedding_overlay_precedes_callback_final_freeze(native_lineage):
    from noesis.identity_v2_service import IdentityV2Service, IdentityV2Mode
    from noesis.pipelines import hooks
    native=native_lineage
    packet,tracks=lineage_packet(native,1,1)
    track=tracks[0]
    track.update(stable_id=17,identity_kind="resident",identity_state="resident",resident_uuid="original-resident")
    e=packet.rows[0].evidence
    pending=[(track,e.lifecycle,e.measurements,e.resolution,e.depth)]
    service=object.__new__(IdentityV2Service)
    service.mode=IdentityV2Mode.AUTHORITATIVE
    service._apply_missing_embedding(track)  # Actual authoritative missing-embedding public overlay.
    result=hooks._finalize_world_quality_inputs(pending)
    assert result[0].household.state=="provisional" and result[0].household.stable_id is None
    assert track["stable_id"] is None and track["identity_v2"]["fresh_embedding"] is False
    tree=ast.parse((Path(__file__).resolve().parents[1]/"noesis/pipelines/hooks.py").read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="_AnalyticsTelemetryProcessor")
    callback=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_handle_servicemaker_frame_admitted")
    calls=[(n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id,n.lineno) for n in ast.walk(callback)
        if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))]
    line=lambda name:next(lineno for call,lineno in calls if call==name)
    assert line("_process_identity_v2_source_frame") < line("_finalize_world_quality_inputs") < line("_enqueue_tracking_publication")
    assert not any(call=="capture_quality_input" for call,_ in calls)


def test_t9_person_final_household_change_denies_before_owner_mutation(native_lineage):
    from noesis_core.world_service import WorldPublicationRefused
    native=native_lineage
    packet,tracks=lineage_packet(native,1,1)
    tracks[0].update(identity_kind="visitor",identity_state="visitor",stable_id=17,visitor_generation=1)
    before=native.f.identity.stats()
    with pytest.raises(WorldPublicationRefused,match="world_quality_final_household_changed"):
        native.f.ingress.stage(packet,tracks)
    assert native.f.identity.stats()==before and native.f.wire==[] and native.f.world.current_snapshot() is None


def runtime_close_functions():
    tree=ast.parse((Path(__file__).resolve().parents[1]/"noesis/ds9_runtime_core.py").read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)
        and n.name in {"_drain_runtime_publications","_finalize_canonical_world"}]
    namespace={}
    exec(compile(ast.Module(functions,type_ignores=[]),"actual-sdk-free-runtime-close","exec"),namespace)
    return namespace


def actual_worker_processor(native):
    from types import MethodType
    from noesis.pipelines import hooks
    processor=native.processor
    processor._tracking_lifecycle=native.f.registries[1]
    processor._tracking_publication_worker=processor._shadow_identity_worker=None
    processor._ensure_tracking_publication_worker=MethodType(hooks._AnalyticsTelemetryProcessor._ensure_tracking_publication_worker,processor)
    processor._publish_tracking_work=MethodType(hooks._AnalyticsTelemetryProcessor._publish_tracking_work,processor)
    processor.shutdown=MethodType(hooks._AnalyticsTelemetryProcessor.shutdown,processor)
    return processor


def person_worker_work(native,frame=1):
    from noesis.pipelines import hooks
    packet,tracks=lineage_packet(native,1,frame)
    foot=hooks.Footpoint(tracker_id=9,tracker_lifecycle_generation=tracks[0]["tracker_lifecycle_generation"],
        frame_id=frame,world_x=1.,world_y=0.,world_z=2.)
    return hooks._TrackingPublicationWork(1,"kitchen",frame,packet.continuity.observed_at_us,time.time(),
        packet.private.media_pts_ns//1000,tracks,[foot],{"media_pts_ns":packet.continuity.media_pts_ns},
        continuity=packet.continuity,xc3d_private_evidence=packet.private,world_quality_work=packet)


@pytest.mark.parametrize("mode",["drain","empty","terminal"])
def test_t9_person_actual_worker_REAL_journal_same_actor_close_once(native_lineage,monkeypatch,mode):
    import threading
    native=native_lineage
    processor=actual_worker_processor(native)
    world=native.f.world
    close_calls=[]
    original=type(world).close
    def close(target):
        close_calls.append((target,threading.get_ident()))
        return original(target)
    monkeypatch.setattr(type(world),"close",close)
    worker=processor._ensure_tracking_publication_worker()
    failure=RuntimeError("original-publication-error")
    reserved=[]
    if mode=="terminal":
        def fail(work):
            native.f.ingress.stage(work.world_quality_work,work.tracks)
            native.f.ingress.prepare(1,work.tracks,metadata={"camera_id":"kitchen","frame_id":work.frame_id,
                "source_epoch":0,"observed_at_us":work.observed_at_us,"media_pts_ns":work.continuity.media_pts_ns})
            reserved.append(native.f.identity.stats().pending_reservations)
            raise failure
        processor._publish_tracking_work=fail
    if mode!="empty":assert worker.enqueue(person_worker_work(native))
    if mode=="drain":assert worker.enqueue(person_worker_work(native,2))
    receipt=processor.shutdown(timeout_s=2.)
    assert receipt.completed and receipt.error is None and receipt.target is world
    assert close_calls==[(world,worker._thread.ident)] and receipt.thread_id==worker._thread.ident
    assert worker._thread.ident!=threading.get_ident() and world.closed
    assert native.f.identity.stats().pending_reservations==0
    assert not worker.enqueue(SimpleNamespace(source_id=1,frame_id=3))
    assert processor.shutdown(timeout_s=2.) is receipt and len(close_calls)==1
    assert runtime_close_functions()["_finalize_canonical_world"](world,native.f.ingress,receipt)
    assert len(close_calls)==1  # Actual main finalizer consumes receipt, never public close twice.
    if mode=="terminal":assert worker.terminal_failure is failure and reserved==[1]
    elif mode=="drain":assert len([row for row in native.f.wire if row["type"]=="tracking"])==2
    else:assert native.f.wire==[]


@pytest.mark.parametrize("mode",["error","timeout"])
def test_t9_person_worker_close_failure_is_distinct_and_receipt_not_fabricated(native_lineage,monkeypatch,mode):
    import threading
    native=native_lineage
    processor=actual_worker_processor(native)
    world=native.f.world
    entered,release=threading.Event(),threading.Event()
    original=type(world).close
    if mode=="error":
        journal_close=type(world._journal).close
        def failed_flush(journal):
            journal_close(journal)
            raise RuntimeError("original-journal-close-error")
        monkeypatch.setattr(type(world._journal),"close",failed_flush)
    def close(target):
        entered.set()
        if mode!="error":assert release.wait(2.)
        return original(target)
    monkeypatch.setattr(type(world),"close",close)
    # No-work shutdown must create the same EXISTING worker/finalizer.
    with pytest.raises(RuntimeError,match="did not quiesce"):
        processor.shutdown(timeout_s=.05)
    worker=processor._tracking_publication_worker
    assert entered.wait(1.) and worker.terminal_failure is None
    if mode=="error":
        receipt=processor.world_actor_close_receipt
        assert not receipt.completed and str(receipt.error)=="original-journal-close-error"
        assert world.closed  # Closed flag is NOT a successful durable-journal receipt.
    else:
        assert processor.world_actor_close_receipt is None and worker._thread.is_alive()
        release.set()
        receipt=processor.shutdown(timeout_s=2.)
        assert receipt.completed and world.closed
    monkeypatch.setattr(type(world),"close",original)


def test_t9_person_actual_runtime_gate_quiescence_FIFO_drain_then_receipt_consumption(native_lineage):
    import threading
    native=native_lineage
    processor=actual_worker_processor(native)
    gate=processor.publication_gate
    lease=gate.acquire()  # Actual admitted callback still finishing scalar enqueue.
    entered,release=threading.Event(),threading.Event()
    publish=processor._publish_tracking_work
    def held(work):
        entered.set()
        assert release.wait(2.)
        publish(work)
    processor._publish_tracking_work=held
    worker=processor._ensure_tracking_publication_worker()
    assert worker.enqueue(person_worker_work(native)) and entered.wait(1.)
    def finish_admitted():
        deadline=time.monotonic()+1.
        while not gate.snapshot().admission_closed and time.monotonic()<deadline:time.sleep(.001)
        assert gate.snapshot().admission_closed
        release.set()
        lease.release()
    callback=threading.Thread(target=finish_admitted)
    callback.start()
    methods=runtime_close_functions()
    publication,receipt=methods["_drain_runtime_publications"](gate,processor,timeout_s=2.)
    callback.join(1.)
    assert publication.quiesced and gate.acquire() is None
    assert worker.terminal_failure is None and receipt.completed and native.f.world.closed
    assert len(native.f.wire)>0 and methods["_finalize_canonical_world"](native.f.world,native.f.ingress,receipt)
    with pytest.raises(RuntimeError,match="acknowledgement unavailable"):
        methods["_finalize_canonical_world"](native.f.world,native.f.ingress,replace(receipt,target=object()))
    with pytest.raises(RuntimeError,match="acknowledgement unavailable"):
        methods["_finalize_canonical_world"](native.f.world,native.f.ingress,None)
    tree=ast.parse((Path(__file__).resolve().parents[1]/"noesis/ds9_runtime_core.py").read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_run_main")
    calls=[(n.func.id,n.lineno) for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
        and n.func.id in {"_drain_runtime_publications","_stop_websocket_server","_finalize_canonical_world"}]
    drain=next(line for name,line in calls if name=="_drain_runtime_publications")
    shutdown=min(line for name,line in calls if name=="_stop_websocket_server" and line>drain)
    finalize=next(line for name,line in calls if name=="_finalize_canonical_world")
    assert drain<shutdown<finalize


def invalidated_binding_packet(native,frame=2,*,overlay=False):
    """Actual resolver -> actual invalidator; SYNTHETIC metric/camera inputs."""
    from noesis.pipelines import hooks
    from noesis_core.world.resolver import UniversalWorldMeasurementResolver
    packet,tracks=lineage_packet(native,1,frame)
    track=tracks[0]
    # A locally accepted sample can still fail the final output gate. Preserve
    # that True on entry so the actual invalidator must withdraw the vote.
    assert track["world_measurement_accepted"] is True
    processor=SimpleNamespace(pipeline=SimpleNamespace(config={"xc3d_world_quality":{
        "contract":"noesis.xc3d.world_quality.v1","enabled":True}}),
        _world_resolver=UniversalWorldMeasurementResolver(),_world_resolver_diagnostics_enabled=False,
        _world_resolver_diag_max_candidates=4)
    point,resolution=hooks._AnalyticsTelemetryProcessor._apply_resolved_world_measurement(
        processor,track,packet.rows[0].evidence.measurements)
    assert point is not None and resolution.status=="measured"
    measurements,original_resolution=track.pop("_xc3d_quality_resolver")
    assert original_resolution is resolution
    hooks._AnalyticsTelemetryProcessor._invalidate_world_track(track,"synthetic-current-range-refusal")
    if overlay:
        from noesis.telemetry.person_ground_state import PersonGroundState
        state=PersonGroundState(measurement_accepted=True)
        before=copy.deepcopy(state)
        hooks._AnalyticsTelemetryProcessor._publish_ground_state_fields(track,state)
        assert state==before and state.measurement_accepted is True
        assert track["world_measurement_accepted"] is False
    inputs=hooks._finalize_world_quality_inputs([(track,packet.rows[0].evidence.lifecycle,
        measurements,resolution,packet.rows[0].evidence.depth)])
    return freeze_quality_work(packet.private,packet.continuity,inputs),tracks


@pytest.mark.parametrize("overlay",[False,True])
def test_t9_binding_actual_resolver_invalidator_World_absence_then_recovery(native_lineage,overlay):
    native=native_lineage
    first,tracks=lineage_packet(native,1,1)
    initial=publish_lineage(native,first,tracks)
    invalid,tracks=invalidated_binding_packet(native,overlay=overlay)
    row,=invalid.rows
    assert row.denial is None and not row.current_measured
    assert row.evidence.resolution.status=="measured" and row.evidence.measurements is not None
    assert "world" not in tracks[0] and "world_covariance" not in tracks[0]
    retained=publish_lineage(native,invalid,tracks)
    assert len(retained.entities)==1 and retained.entities[0].observed_at_us==initial.entities[0].observed_at_us
    assert retained.entities[0].position==initial.entities[0].position
    assert len(native.f.wire[3]["tracks"])==1  # Occupied row was not dropped.
    assert tracks[0].get("world_measurement_accepted") is not True and tracks[0].get("world") is None
    assert native.f.ingress._context.comparability_receipt_id is None
    assert native.pipeline.xc3d_swin_ingress.cache.members==[{},{},{}]  # No appearance materialized/granted.
    current,tracks=lineage_packet(native,1,3)
    recovered=publish_lineage(native,current,tracks)
    assert recovered.entities[0].observed_at_us==current.continuity.observed_at_us
    assert recovered.entities[0].observed_at_us!=initial.entities[0].observed_at_us
    assert recovered.entities[0].entity_id==initial.entities[0].entity_id


@pytest.mark.parametrize("world_valid",[False,True,None])
@pytest.mark.parametrize("locally_accepted",[False,True])
def test_r165_public_filter_overlay_does_not_restore_invalid_output_vote(world_valid,locally_accepted):
    from noesis.pipelines import hooks
    from noesis.telemetry.person_ground_state import PersonGroundState
    state=PersonGroundState(measurement_accepted=locally_accepted,
        measurement_rejection_reason="local-diagnostic",reacquire_count=2,trail_segment_id=7)
    before=copy.deepcopy(state)
    track={"world_valid":world_valid,"world_quality":"invalid" if world_valid is False else "held",
        "_xc3d_quality_resolver":object(),"_xc3d_current_original":object()}
    resolver,original=track["_xc3d_quality_resolver"],track["_xc3d_current_original"]
    hooks._AnalyticsTelemetryProcessor._publish_ground_state_fields(track,state)
    assert state==before
    assert track["world_measurement_accepted"] is (False if world_valid is False else locally_accepted)
    assert track["world_rejection_reason"]=="local-diagnostic" and track["world_reacquire_count"]==2
    assert track["_xc3d_quality_resolver"] is resolver and track["_xc3d_current_original"] is original


def test_r165_actual_producer_final_output_veto_keeps_local_accept_nonvoting(monkeypatch):
    from noesis.pipelines import hooks
    from DS9.tests.test_world_resolver_hook_integration import (
        _r123_actual_contact_processor, _r123_actual_contact_row,
    )
    processor,calibration=_r123_actual_contact_processor()
    actual=hooks.admit_human_ground_output
    local_accepts=[]
    def final_veto(state,**kwargs):
        point,continuous=actual(state,**kwargs)
        if state.measurement_accepted:
            local_accepts.append(state)
            return point,False
        return point,continuous
    monkeypatch.setattr(hooks,"admit_human_ground_output",final_veto)
    monkeypatch.setattr(processor,"_bounded_process_output_provenance",lambda *a,**k:None)
    rows=[_r123_actual_contact_row(processor,calibration,frame=f,contact=True) for f in range(1,5)]
    assert local_accepts and local_accepts[-1].measurement_accepted is True
    row=rows[-1]
    assert row["world_quality_reason"]=="canonical_continuity_provenance_unavailable"
    assert row["world_valid"] is False and row["world_measurement_accepted"] is False
    assert row.get("world") is None and row.get("world_covariance") is None
    assert "world_estimation_error" not in row


@pytest.mark.parametrize("fault",["point","covariance","valid","accepted","quality","reason","partial",
    "revision","transform","calibration","resolver_frame","resolver_camera","source_nonscalar","accepted_nonbool"])
def test_t9_binding_invalid_is_not_permission_for_mislabeled_or_current_point(native_lineage,fault,caplog):
    from noesis_core.world_service import WorldPublicationRefused
    native=native_lineage
    packet,tracks=invalidated_binding_packet(native,1)
    track=tracks[0]
    evidence=packet.rows[0].evidence
    if fault=="point":track["world"]=[1.,0.,2.]
    elif fault=="covariance":track["world_covariance"]=[.04,0.,0.,0.,.04,0.,0.,0.,.09]
    elif fault=="valid":track["world_valid"]=True
    elif fault=="accepted":track["world_measurement_accepted"]=True
    elif fault=="accepted_nonbool":track["world_measurement_accepted"]=1
    elif fault=="quality":track["world_quality"]="held"
    elif fault=="reason":track["world_quality_reason"]=""
    elif fault=="partial":track["world_frame"]="backend_world_m"
    elif fault=="revision":track["world_frame_revision"]="foreign-revision"
    elif fault=="transform":track["world_transform_sha256"]="f"*64
    elif fault=="calibration":track["world_calibration_sha256"]="f"*64
    elif fault=="source_nonscalar":track["world_source"]=[1.,2.,3.]
    else:
        field,value=("frame_id",99) if fault=="resolver_frame" else ("camera_id","foreign-camera")
        wrong=WorldMeasurementCohort.model_validate({**evidence.measurements.cohort.model_dump(),field:value})
        evidence=replace(evidence,measurements=WorldMeasurementSet(cohort=wrong,hypotheses=()),
            resolution=ResolvedGroundMeasurement.model_validate({**evidence.resolution.model_dump(),"cohort":wrong}))
    changed=capture_quality_input(track,evidence.lifecycle,evidence.measurements,evidence.resolution,evidence.depth)
    work=freeze_quality_work(packet.private,packet.continuity,(changed,))
    assert work.rows[0].denial in {"world_quality_emitted_frame_binding_mismatch","world_quality_original_resolver_cohort_mismatch"}
    before=native.f.identity.stats()
    with pytest.raises(WorldPublicationRefused):native.f.ingress.stage(work,tracks)
    assert native.f.identity.stats()==before and native.f.world.current_snapshot() is None and native.f.wire==[]
    logs=[record for record in caplog.records if record.message.startswith("World quality binding mismatch:")]
    assert len(logs)==1
    diagnostic=logs[0].args
    assert diagnostic["source_id"]==1 and diagnostic["frame_id"]==1
    assert len(repr(diagnostic))<4096 and set(diagnostic)=={"source_id","frame_id","reason","resolver_status",
        "resolver_binding","emitted","position_present","covariance_present"}
    assert "stable_id" not in repr(diagnostic) and "resident_uuid" not in repr(diagnostic)
    if fault=="source_nonscalar":assert diagnostic["emitted"]["world_source"]=="<non-scalar>"


@pytest.mark.parametrize("status,source",[("hold","anchor_hold"),("prediction","cv_prediction")])
def test_t9_binding_actual_noncurrent_DTO_preserves_original_time_and_position(native_lineage,status,source):
    native=native_lineage
    packet,tracks=lineage_packet(native,1,1)
    evidence=packet.rows[0].evidence
    resolution=ResolvedGroundMeasurement.model_validate({**evidence.resolution.model_dump(),"status":status,
        "quality":"held" if status=="hold" else "prediction",
        "selected_candidate_id":None,"selected_kind":None,"contributor_ids":(),"fused":False})
    tracks[0].update(world_source=source,world_covariance=None,world_measurement_accepted=False,
        world_observation_ts_us=123,world_measurement_frame_id=0)
    work=freeze_quality_work(packet.private,packet.continuity,(capture_quality_input(tracks[0],evidence.lifecycle,
        evidence.measurements,resolution,evidence.depth),))
    row,=work.rows
    assert row.denial is None and not row.current_measured and row.evidence.resolution is resolution
    assert dict(row.evidence.world)["world_observation_ts_us"]==123 and dict(row.evidence.world)["world_measurement_frame_id"]==0
    assert dict(row.evidence.world)["world"]==tuple(tracks[0]["world"])


@pytest.mark.parametrize("fault",[None,"missing_mode","positional_mode","wrong_default","missing_prepare","missing_commit"])
def test_r12_actual_startup_requires_quarantine_API_without_SDK(monkeypatch,fault):
    from noesis.pipelines.world_quality_adapter import require_world_quality_api
    from noesis_core.world_service import CanonicalWorldService
    from noesis.identity_association_service import IdentityAssociationService
    # All checks inspect actual public objects; negative signatures are never executed.
    if fault=="missing_mode":
        def obsolete(self,source,measurements,memberships,*,at_interval):raise AssertionError("not executed")
        monkeypatch.setattr(CanonicalWorldService,"prepare_spatial",obsolete)
    elif fault=="positional_mode":
        def positional(self,source,measurements,memberships,attribution_mode="strict",*,at_interval):raise AssertionError("not executed")
        monkeypatch.setattr(CanonicalWorldService,"prepare_spatial",positional)
    elif fault=="wrong_default":
        def permissive(self,source,measurements,memberships,*,at_interval,attribution_mode="physical_only_quarantine"):raise AssertionError("not executed")
        monkeypatch.setattr(CanonicalWorldService,"prepare_spatial",permissive)
    elif fault in {"missing_prepare","missing_commit"}:
        name="prepare_spatial_physical_member" if fault=="missing_prepare" else "commit_spatial_physical_member"
        monkeypatch.setattr(IdentityAssociationService,name,None)
    if fault is None:require_world_quality_api()
    else:
        with pytest.raises(ValueError,match="xc3d_world_quality_public_api_unavailable"):require_world_quality_api()


def r12_households(kind):
    return {9:{"identity_kind":"resident","resident_uuid":"synthetic-resident-a","stable_id":9},
        10:{"identity_kind":"resident","resident_uuid":"synthetic-resident-a" if kind=="subject" else "synthetic-resident-b",
            "stable_id":10 if kind=="subject" else 9},
        11:{"identity_kind":"resident","resident_uuid":"synthetic-unaffected","stable_id":11}}


@pytest.fixture
def r12_native_lineage(monkeypatch,request):
    # The actual scalar FIFO hook is the consumer, never an SDK/ELF import test.
    for name in ("pyservicemaker","pyds","noesis_pose_meta_ext","noesis_analytics_meta_ext",
        "noesis_v3dt_meta_ext","noesis_reid_meta_ext","noesis_depth_meta_ext","noesis_depth_tracking_tensor_ext"):
        monkeypatch.setitem(sys.modules,name,None)
    return request.getfixturevalue("native_lineage")


@pytest.mark.parametrize("kind",["subject","sid"])
@pytest.mark.parametrize("strict",[False,True])
def test_r12_actual_native_FIFO_conflict_bootstrap_cache_continuity_World_Node22(r12_native_lineage,monkeypatch,kind,strict):
    from noesis_core.world_service import WorldPublicationRefused
    native=r12_native_lineage
    f=native.f
    cache=native.pipeline.xc3d_swin_ingress.cache
    original_prepare=f.world.prepare_spatial
    issued=[]
    consumed=[]
    for name in ("commit_spatial_member","commit_spatial_physical_member"):
        actual=getattr(f.identity,name)
        def record_consumer(reservation,*,at_interval,consumer,_actual=actual):
            consumed.append((reservation.purpose,reservation.dto.association_revision))
            return _actual(reservation,at_interval=at_interval,consumer=consumer)
        monkeypatch.setattr(f.identity,name,record_consumer)
    def real_prepare(*args,**kwargs):
        issued.append(args[2])
        if strict:kwargs.pop("attribution_mode",None)  # Exercise actual unchanged strict default.
        return original_prepare(*args,**kwargs)
    monkeypatch.setattr(f.world,"prepare_spatial",real_prepare)
    packet,tracks=lineage_packet(native,1,1,raws=(9,10,11),households=r12_households(kind))
    before_tracks=copy.deepcopy(tracks)
    assert native.callback()=={0:{},1:{},2:{}}  # No pre-seeded owner or prospective birth.
    if strict:
        with pytest.raises(WorldPublicationRefused,match="settled_household_claim_conflict"):
            publish_lineage(native,packet,tracks)
        assert f.world.current_snapshot() is None and f.wire==[] and cache.owners==[{},{},{}]
        assert f.identity.stats().pending_reservations==0
        return
    first=publish_lineage(native,packet,tracks)
    owners=native.callback()[1]
    assert len(owners)==3 and all(o.original_birth is packet.private.original and o.lifecycle_started_frame==1 for o in owners.values())
    assert len(first.entities)==3 and len({e.entity_id for e in first.entities})==3
    quarantine=[e for e in first.entities if e.attribution_quarantine is not None]
    assert len(quarantine)==2 and all(e.subject.stable_id is None for e in quarantine)
    for entity in quarantine:
        assert len(entity.attribution_quarantine.owners)==2
        for claim in entity.attribution_quarantine.owners:
            expected=r12_households(kind)[claim.raw_id]
            assert (claim.source_id,claim.source_epoch,claim.lifecycle_started_frame,claim.frame_id)==(1,0,1,1)
            assert claim.household_identity_key==expected["resident_uuid"] and claim.household_stable_id==expected["stable_id"]
            assert claim.media_time_s==claim.acquired_at_s==packet.private.original.pts_ns/1e9
    assert next(e for e in first.entities if e.attribution_quarantine is None).subject.stable_id==11
    assert tracks==before_tracks and f.wire[0]["tracks"]==before_tracks
    for entity in first.entities:
        track=next(t for t in tracks if t["world"][0]==entity.position.x)
        assert tuple(track["world_covariance"])==entity.covariance.values
        assert entity.observed_at_us==packet.continuity.observed_at_us
    assert sorted(purpose for purpose,_ in consumed)==["physical_only_quarantine","physical_only_quarantine","strict"]
    next_packet,next_tracks=lineage_packet(native,1,2,x=1.1,raws=(9,10,11),households=r12_households(kind))
    assert all(r.evidence.lifecycle.disposition!="new" for r in next_packet.rows)
    second=publish_lineage(native,next_packet,next_tracks)
    assert all(native.callback()[1][k] is binding for k,binding in owners.items())
    assert {e.entity_id for e in second.entities}=={e.entity_id for e in first.entities}
    assert all(next(e for e in second.entities if e.entity_id==old.entity_id).position.x>old.position.x for old in first.entities)
    assert all(o.stamp.lifecycle_started_frame==1 and o.stamp.frame_id==2 for o in f.ingress.originals.values())
    for memberships in issued:
        assert len(memberships)==3 and len({m.association_revision for m in memberships})==1
    assert len(consumed)==6 and consumed[0][1]==consumed[1][1]==consumed[2][1]
    assert consumed[3][1]==consumed[4][1]==consumed[5][1]>consumed[0][1]
    assert f.identity.stats().pending_reservations==0
    result=f.examples.actual_node_wire(f,f.wire,"r12-native-cache-"+kind)
    assert all(d["accepted"] for d in result["diagnostics"])
    assert [len(s["people"]) for s in result["snapshots"]]==[3,3]
    for person in result["snapshots"][-1]["people"]:
        assert len(person["trail"])==2
        if person["attributionQuarantine"]:
            assert person["label"]=="Person (attribution quarantined)" and person["stableId"] is None


@pytest.mark.parametrize("fault",["reset","claim_change","wrong_purpose"])
def test_r12_actual_adapter_stale_or_wrong_purpose_aborts_before_wire_and_cache(r12_native_lineage,monkeypatch,fault):
    from noesis_core.world_service import WorldPublicationRefused
    native=r12_native_lineage
    f=native.f
    packet,tracks=lineage_packet(native,1,1,raws=(9,10),households=r12_households("sid"))
    if fault in {"reset","claim_change"}:
        original=f.world.commit
        def reset_before_commit(*args,**kwargs):
            if fault=="reset":
                binding=next(b for b in f.sources if b.source_id==1)
                assert f.identity.reset_source(replace(binding,source_epoch=1)).disposition=="reset"
            else:
                from reid.mv3dt_person_association import SourceObservation, SpatialLifecycleReceipt, HouseholdDescriptor
                previous=next(o for o in f.ingress.originals.values() if o.stamp.member.raw_id==9)
                stamp=replace(previous.stamp,frame_id=2,observation_id="synthetic-queued-household-change",
                    acquired_at_s=2*33_333_333/1e9,media_time_s=2*33_333_333/1e9)
                changed=SourceObservation(stamp,HouseholdDescriptor("resident","synthetic-changed-claim",20))
                result=f.identity.observe_spatial(changed,SpatialLifecycleReceipt("synthetic-queued-active",stamp,"ACTIVE"))
                assert result.disposition=="observed"
            return original(*args,**kwargs)
        monkeypatch.setattr(f.world,"commit",reset_before_commit)
    else:
        original=f.identity.prepare_spatial_physical_member
        def wrong_public_consumer(dto,reference,*,at_interval):
            result=original(dto,reference,at_interval=at_interval)
            assert result.disposition=="reserved"
            denied=f.identity.commit_spatial_member(result.reservation,at_interval=at_interval,
                consumer=lambda *_:pytest.fail("wrong purpose must not consume"))
            assert denied.reason=="spatial_reservation_purpose_mismatch"
            assert f.identity.discard_spatial_member(result.reservation).disposition=="discarded"
            return denied  # Genuine public owner denial, no invented positive capability.
        monkeypatch.setattr(f.identity,"prepare_spatial_physical_member",wrong_public_consumer)
    with pytest.raises(WorldPublicationRefused):publish_lineage(native,packet,tracks)
    assert f.world.current_snapshot() is None and f.wire==[] and native.callback()=={0:{},1:{},2:{}}
    assert f.identity.stats().pending_reservations==0


@pytest.mark.parametrize('living',[False,True])
def test_r29_image_owner_shadow_receipt_no_measurement_renewal_and_real_return(tmp_path,monkeypatch,living):
    """Real adapter/service/World/publisher; all scalar geometry is synthetic."""
    from noesis.identity_association_service import IdentityAssociationService
    from noesis.pipelines.projective_continuity import ImageRegionQualityBridge
    from noesis.pipelines.source_acquisition import OriginalSwinCache,owner_retention_members
    from noesis.pipelines.world_quality_adapter import capture_body_support
    from reid.mv3dt_person_association import ProjectiveContinuityPolicy,ProjectiveCameraBinding
    frame=[1];ns=time.time_ns()
    def wall_time():return (ns+frame[0]*33_333_333)/1e9
    # Source and publication clocks share the same synthetic microsecond epoch.
    # Derive both from one value to avoid float roundtrip's possible 1us skew.
    monkeypatch.setattr(time,'time',wall_time)
    monkeypatch.setattr(time,'time_ns',lambda:int(wall_time()*1_000_000)*1000)
    monkeypatch.setattr(time,'monotonic',lambda:100.+frame[0]/30)
    selected={}
    def factory(run,spatial):
        p=((100.,0.,0.,0.),(0.,100.,0.,0.),(0.,0.,1.,0.));examples=world_examples()
        policy=ProjectiveContinuityPolicy(tuple(ProjectiveCameraBinding(b,'synthetic-r29-prior','raster:'+b.camera_id,
            (1920,1080),'backend_world_m',examples.frame_binding(b.source_id).target_frame.revision,
            examples.frame_binding(b.source_id).source_camera_calibration_sha256,p,
            ((100.,0.,0.,-10. if b.source_id==1 else 10.),p[1],p[2])) for b in spatial.sources if b.source_id in (1,2)))
        selected['policy']=policy
        return IdentityAssociationService.for_estimated_image_regions(run,spatial,policy,live_track_continuity=True)
    f=NativeQualityFixture(tmp_path,source_ids=(0,1,2) if living else (1,2),identity_factory=factory)
    f.ingress.projective_bridge=ImageRegionQualityBridge(selected['policy'],OriginalSwinCache())
    def packet(source,at,**kwargs):
        frame[0]=at
        work,tracks=f.packet(source,at,**kwargs)
        inputs=[]
        for row,track in zip(work.rows,tracks):
            u=50. if source==1 else 40.
            track['bbox']=[u-10,20.,20.,40.]
            joints=[(u,30.,.9)]*17
            joints[5],joints[6]=(u-2,30.,.9),(u+2,30.,.9)
            joints[11],joints[12]=(u-2,42.,.9),(u+2,42.,.9)
            body=capture_body_support(track,row.evidence.lifecycle,joints,pose_is_current=True)
            inputs.append(replace(row.evidence,body=body))
        return replace(freeze_quality_work(work.private,work.continuity,tuple(inputs)),
            ingress_monotonic_s=time.monotonic()),tracks
    try:
        f.publish([packet(1,1,scale=.5)])
        first=f.publish([packet(2,1,scale=4.)])
        assert len(first.entities)==1
        original=next(o for o in f.ingress.originals.values() if o.stamp.member.source_id==2)
        active=f.publish([packet(1,2,scale=.5)])
        kept=active.entities[0]
        shadow,diagnostic_tracks=packet(2,2,scale=4.,shadow=True)
        # Actual lifecycle registry/frozen owner query supplies SHADOW truth;
        # separately retained emitted diagnostics do not override that receipt.
        owner_shadow,_=f.packet(2,2,empty=True,shadow=True)
        shadow=replace(shadow,private=replace(shadow.private,
            owner_retention=owner_shadow.private.owner_retention),continuity=owner_shadow.continuity)
        retained,measured=owner_retention_members(shadow.private.owner_retention,shadow.private.original,
            shadow.continuity.source_epoch,shadow.continuity.frame_id)
        assert retained and not measured and shadow.rows[0].current_measured
        assert shadow.rows[0].evidence.body is not None  # deceptive current-looking row is NOT native measured
        held=f.publish([(shadow,diagnostic_tracks)])
        assert f.ingress.projective_bridge.counts['retained_unmeasured_updates']==1
        assert next(o for o in f.ingress.originals.values() if o.stamp.member.source_id==2) is original
        assert original.stamp not in f.identity.image_region_active_cohort()
        assert len(held.entities)==1
        h=held.entities[0]
        assert (h.entity_id,h.position,h.observed_at_us,h.stale_after_us)==(
            kept.entity_id,kept.position,kept.observed_at_us,kept.stale_after_us)
        tracking=[r for r in f.wire if r['type']=='tracking'][-1]
        assert tracking['tracks'][0]['tracker_id']==9  # diagnostic row survives unchanged
        if living:
            local=f.publish([packet(0,2)])
            assert len(local.entities)==2
            assert all(s.member.source_id in (1,2) for s in f.identity.image_region_active_cohort())
            assert f.ingress.projective_bridge.counts['retained_unmeasured_updates']==1
        returned=f.publish([packet(2,3,scale=4.)])
        restored=next(o for o in f.ingress.originals.values() if o.stamp.member.source_id==2)
        assert restored.stamp.frame_id==3 and restored.stamp!=original.stamp
        assert restored.stamp in f.identity.image_region_active_cohort()
        assert len(returned.entities)==(2 if living else 1)
        assert f.ingress.projective_bridge.counts['result:current_visual_complete_cohort_required']==0
        wire=f.examples.actual_node_wire(f,f.wire,'r29-shadow-return-'+str(living))
        assert all(d['accepted'] for d in wire['diagnostics'])
    finally:f.world.close()


@pytest.mark.parametrize('living',[False,True])
def test_r120_r29_future_synthetic_epoch_has_coherent_source_and_publication_clock(tmp_path,monkeypatch,living):
    # No sleep may be needed for real wall time to catch a synthetic source.
    future=time.time_ns()+10_000_000_000
    monkeypatch.setattr(time,'time_ns',lambda:future)
    test_r29_image_owner_shadow_receipt_no_measurement_renewal_and_real_return(tmp_path,monkeypatch,living)


def joint_appearance_native_fixture(tmp_path,monkeypatch,*,features=True,feature_frame=2,occluded_competitor=False):
    """Synthetic fresh selected outputs through the real queued producer/owner."""
    from noesis.identity_association_service import IdentityAssociationService
    from noesis.pipelines.projective_continuity import ImageRegionQualityBridge,_OriginalBodyDiagnostic
    from noesis.pipelines.source_acquisition import OriginalSwinCache,OriginalSwinExtraction,freeze_owner_availability
    from noesis.pipelines.world_quality_adapter import capture_body_support,freeze_swin_work
    from noesis_core.source_progress import DecodedSourceProgressMonitor,SourceProgressPolicy
    from reid.mv3dt_person_association import ProjectiveContinuityPolicy,ProjectiveCameraBinding
    frame=[1];epoch=time.time_ns();selected={}
    monkeypatch.setattr(time,'time',lambda:(epoch+frame[0]*33_333_333)/1e9)
    monkeypatch.setattr(time,'time_ns',lambda:int(time.time()*1_000_000)*1000)
    monkeypatch.setattr(time,'monotonic',lambda:100.+frame[0]/30)
    def factory(run,spatial):
        p=((100.,0.,0.,0.),(0.,100.,0.,0.),(0.,0.,1.,0.));ex=world_examples()
        selected['policy']=ProjectiveContinuityPolicy(tuple(ProjectiveCameraBinding(b,'synthetic-i135',
            'raster:'+b.camera_id,(1920,1080),'backend_world_m',ex.frame_binding(b.source_id).target_frame.revision,
            ex.frame_binding(b.source_id).source_camera_calibration_sha256,p,
            ((100.,0.,0.,-10. if b.source_id==1 else 10.),p[1],p[2])) for b in spatial.sources))
        return IdentityAssociationService.for_estimated_image_regions(run,spatial,selected['policy'],
            live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    f=NativeQualityFixture(tmp_path,identity_factory=factory);cache=OriginalSwinCache()
    f.ingress.projective_bridge=ImageRegionQualityBridge(selected['policy'],cache)
    f.ingress.projective_bridge._body_diagnostic=_OriginalBodyDiagnostic((feature_frame/30,))
    monitor=DecodedSourceProgressMonitor(f.labels,policy=SourceProgressPolicy(poll_interval_seconds=60.),clock=time.monotonic)
    monitor.start(lambda failure:None)
    def packet(source,number,*,separate=False):
        frame[0]=number
        work,tracks=f.packet(source,number,raws=(9,) if source==1 else (9,10),scale=4. if source==1 else 1.)
        inputs=[];copied=[];mono=time.monotonic()
        for row,track in zip(work.rows,tracks):
            u=50. if source==1 else 40.
            dy=100. if separate else 0.
            track['bbox']=[u-10,20.+dy,20.,40.]
            joints=[(u,30.+dy,.9)]*17
            joints[5],joints[6]=(u-2,30.+dy,.9),(u+2,30.+dy,.9)
            joints[11],joints[12]=(u-2,42.+dy,.9),(u+2,42.+dy,.9)
            if occluded_competitor and number==feature_frame and track['tracker_id']==10:
                joints[11]=(u-2,42.+dy,.1)
            body=capture_body_support(track,row.evidence.lifecycle,joints,pose_is_current=True)
            inputs.append(replace(capture_quality_input(track,row.evidence.lifecycle,row.evidence.measurements,
                row.evidence.resolution,row.evidence.depth),body=body))
            vector=tuple(float(j==(1 if track['tracker_id']==10 else 0)) for j in range(256))
            value=OriginalSwinExtraction(work.private.original,track['tracker_id'],f.profile,vector,mono)
            if features and number==feature_frame:
                assert cache.record(value)
                copied.append(cache.freeze_exact(value.original,value.raw_id))
        monitor.record_progress(source,now=mono)
        availability=freeze_owner_availability(work.private,work.continuity,monitor,ingress_monotonic_s=mono)
        work=replace(freeze_quality_work(work.private,work.continuity,tuple(inputs)),
            ingress_monotonic_s=mono,owner_availability=availability,swin_extractions=tuple(copied))
        return work,tracks
    return f,packet,monitor


def i142_crowded_shadow_fixture(tmp_path,monkeypatch,*,fault=None):
    """Real retained-owner/public ingress; synthetic geometry, no alias proof."""
    from noesis.pipelines.source_acquisition import freeze_owner_availability
    from noesis.pipelines.world_quality_adapter import capture_body_support
    f,packet,monitor=joint_appearance_native_fixture(tmp_path,monkeypatch,features=True,feature_frame=1)
    first=f.publish([packet(1,1),packet(2,1)])
    old,=first.entities
    # Advance only this fixture's clock. The Kitchen original remains frame 1.
    cells=dict(zip(packet.__code__.co_freevars,packet.__closure__))
    cells['frame'].cell_contents[0]=2
    registry=f.registries[2];update=registry.update_frame
    with monkeypatch.context() as patch:
        patch.setattr(registry,'update_frame',lambda **kw:update(**{**kw,'shadow_tracker_ids':(9,)}))
        work,tracks=f.packet(2,2,raws=(10,11))
    inputs=[]
    for row,track in zip(work.rows,tracks):
        track['bbox']=[30.,20.,20.,40.]
        joints=[(40.,30.,.9)]*17
        joints[5],joints[6]=(38.,30.,.9),(42.,30.,.9)
        joints[11],joints[12]=(38.,42.,.9),(42.,42.,.9)
        inputs.append(replace(capture_quality_input(track,row.evidence.lifecycle,row.evidence.measurements,
            row.evidence.resolution,row.evidence.depth),body=capture_body_support(
                track,row.evidence.lifecycle,joints,pose_is_current=True)))
    mono=time.monotonic();monitor.record_progress(2,now=mono)
    availability=freeze_owner_availability(work.private,work.continuity,monitor,ingress_monotonic_s=mono)
    work=replace(freeze_quality_work(work.private,work.continuity,tuple(inputs)),
        ingress_monotonic_s=mono,owner_availability=availability)
    assert work.private.owner_retention.shadow==((9,1),)
    proofs=[];join=f.identity.join_estimated_image_regions
    def capture(proof,**kwargs):
        proofs.append(proof)
        return join(proof,**kwargs)
    monkeypatch.setattr(f.identity,'join_estimated_image_regions',capture)
    if fault=='missing-owner':work=replace(work,owner_availability=None)
    elif fault=='stall':work=replace(work,owner_availability=replace(availability,status='recovering'))
    elif fault=='stop':f.ingress.stop_image_region_source(2,graph_id=work.private.original.graph_id)
    final=f.publish([(work,tracks)])
    return f,monitor,old,final,proofs[-1]


def test_i142_crowded_shadow_keeps_only_finite_original_node(tmp_path,monkeypatch):
    f,monitor,old,final,proof=i142_crowded_shadow_fixture(tmp_path,monkeypatch)
    try:
        assert len(final.entities)==1
        held,=final.entities
        assert held.position_state=='held' and held.entity_id!=old.entity_id
        assert (held.position,held.covariance,held.observed_at_us,held.stale_after_us)==(
            old.position,old.covariance,old.observed_at_us,old.stale_after_us)
        assert [s.observation_id for s in held.sources if s.accepted]==[
            s.observation_id for s in old.sources if s.accepted]
        assert 1_966_666<=held.stale_after_us-final.published_at_us<=1_966_667
        assert not any(s.member.raw_id==9 and s.member.source_id==2 for s in proof.cohort)
        shadow=next(c for c in proof.owner_availability if c.observation.member.source_id==2
            and c.observation.member.raw_id==9)
        assert shadow.observation.frame_id==1 and shadow.source_frame_id==2
        assert shadow.observation.member in shadow.shadow_members
        dtos=f.identity.spatial_memberships(at_interval=f.ingress._context)
        assert all(len(d.members)==1 for d in dtos)
        assert not any(e.eligible for d in dtos for e in d.eligibility if e.observation.member.source_id==2)
        assert all(e.reason=='current_active_own_source' for d in dtos for e in d.eligibility
            if e.observation.member.source_id==1)  # Existing local permission, not a new measurement.
        record=f.identity.image_region_decision_diagnostics()['transitions'][-1]
        assert all(p['semantic_state'] is True and not p['supported'] for p in record['pairs'])
        actual=f.examples.actual_node_wire(f,f.wire,'i142-crowded-shadow-finite-original')
        assert [len(s['people']) for s in actual['snapshots']]==[1,1]
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('fault',['missing-owner','stall','stop'])
def test_i142_crowded_shadow_missing_authority_cannot_hold(tmp_path,monkeypatch,fault):
    f,monitor,old,final,proof=i142_crowded_shadow_fixture(tmp_path,monkeypatch,fault=fault)
    try:
        dtos=f.identity.spatial_memberships(at_interval=f.ingress._context)
        if fault!='stop':assert not final.entities
        assert not any(d.estimated_split_custody for d in dtos)
        assert not any(e.eligible for d in dtos for e in d.eligibility if e.observation.member.source_id==2)
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


def test_i144_forward_nonparticipant_gap_cannot_poison_or_resurrect(tmp_path,monkeypatch):
    from noesis.pipelines.source_acquisition import freeze_owner_availability
    f,monitor,old,final,proof=i142_crowded_shadow_fixture(tmp_path,monkeypatch)
    try:
        original_id=next(s.observation_id for s in old.sources if s.accepted)
        old_member=next(c.observation.member for c in proof.owner_availability
            if c.observation.member.source_id==2 and c.observation.member.raw_id==9)
        assert any(d.estimated_split_custody for d in f.identity.spatial_memberships(at_interval=f.ingress._context))
        frame=dict(zip(time.monotonic.__code__.co_freevars,time.monotonic.__closure__))['frame'].cell_contents
        now=[100.2];monkeypatch.setattr(time,'monotonic',lambda:now[0])
        # Unrelated Kitchen source keeps progressing while Family's queued
        # ownership receipt has a wall-delivery gap, not a media/birth rewrite.
        for number in (2,3):
            frame[0]=number
            work,tracks=f.packet(1,number,raws=(9,),cached=True,scale=4.)
            monitor.record_progress(1,now=now[0])
            work=replace(work,ingress_monotonic_s=now[0],owner_availability=freeze_owner_availability(
                work.private,work.continuity,monitor,ingress_monotonic_s=now[0]))
            f.publish([(work,tracks)]);now[0]+=.1
        results=[];join=f.identity.join_estimated_image_regions
        def capture(p,**kwargs):
            result=join(p,**kwargs);results.append((p,result));return result
        monkeypatch.setattr(f.identity,'join_estimated_image_regions',capture)
        for number in (3,4,5):
            now[0]=100.5+(number-3)/30.;frame[0]=number
            registry=f.registries[2];update=registry.update_frame
            with monkeypatch.context() as patch:
                patch.setattr(registry,'update_frame',lambda **kw:update(**{**kw,'shadow_tracker_ids':(9,)}))
                work,tracks=f.packet(2,number,raws=(12,13))
            monitor.record_progress(2,now=now[0])
            work=replace(work,ingress_monotonic_s=now[0],owner_availability=freeze_owner_availability(
                work.private,work.continuity,monitor,ingress_monotonic_s=now[0]))
            snapshot=f.publish([(work,tracks)])
            current,result=results[-1]
            assert result.reason!='image_region_owner_availability_reordered'
            assert result.disposition in ('accepted','observed')
            assert all(f.identity.ledger._members[s.member].image_region_owner_availability is not None
                for s in current.cohort if s.member.source_id==2)
            member=f.identity.ledger._members[old_member]
            assert not member.image_region_native_seed and not member.image_region_front_confirmed
            assert member.image_region_split_custody is None
            assert not any(d.estimated_split_custody for d in f.identity.spatial_memberships(at_interval=f.ingress._context))
            assert not next(e for d in f.identity.spatial_memberships(at_interval=f.ingress._context)
                for e in d.eligibility if e.observation.member==old_member).eligible
            assert not any(s.accepted and s.observation_id==original_id
                for e in snapshot.entities for s in e.sources)
            if number==3:assert member.image_region_owner_availability is None
            else:
                cap=member.image_region_owner_availability
                assert cap is not None and cap.observation==member.observation.stamp
                assert old_member in cap.shadow_members and cap.availability_status=='running'
                assert 0<=cap.source_media_time_s-cap.observation.media_time_s<=cap.owner_grace_ns/1e9
        assert old.stale_after_us==old.observed_at_us+2_000_000
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('features',[False,True])
@pytest.mark.parametrize('occluded_competitor',[False,True])
def test_i135_real_current_three_body_joint_appearance(tmp_path,monkeypatch,features,occluded_competitor):
    f,packet,monitor=joint_appearance_native_fixture(tmp_path,monkeypatch,features=features,feature_frame=3,
        occluded_competitor=occluded_competitor)
    try:
        first=f.publish([packet(2,1)])
        assert len(first.entities)==2
        # Literal current exclusions independently establish the third native
        # person before the later ambiguous three-body cohort, no private bits.
        middle=f.publish([packet(1,2,separate=True)])
        assert len(middle.entities)==3
        family=packet(2,3)
        final=f.publish([packet(1,3),family])
        dtos=f.identity.spatial_memberships(at_interval=f.ingress._context)
        assert sorted(len(v.members) for v in dtos)==([1,2] if features else [1,1,1])
        assert len(final.entities)==(2 if features else 3)
        if features:
            pair=next(v for v in dtos if len(v.members)==2)
            assert {m.observation.member.raw_id for m in pair.members}=={9}
        if occluded_competitor:
            record=f.identity.image_region_decision_diagnostics()['transitions'][-1]
            other=next(p for p in record['pairs'] if p['right']['raw_id']==10)
            assert other['semantic_state'] is None and not other['appearance_corroborated']
        actual=f.examples.actual_node_wire(f,f.wire,'i135-joint-appearance-'+str(features))
        assert [len(s['people']) for s in actual['snapshots']]==[2,3,2 if features else 3]
        accepted=next(s for e in final.entities for s in e.sources if s.accepted and s.camera_id=='family-room' and s.position.x==1.)
        point=next(e for e in final.entities if any(s is accepted for s in e.sources))
        assert accepted.observed_at_us==family[1][0]['observed_at_us']==point.observed_at_us
        assert point.stale_after_us==point.observed_at_us+2_000_000
        assert point.position.x==1. and point.position.z==2.
        assert point.covariance==accepted.covariance
        assert f.identity.stats().pending_reservations==0
        detail=f.ingress.projective_bridge._body_diagnostic.admission_summary()
        assert detail['records'] and all(r['appearance']['complete'] is features for r in detail['records'])
    finally:
        assert monitor.stop()['quiesced']
        f.world.close()


@pytest.mark.parametrize('features',[False,True])
def test_i139_fifo_qualified_original_features_current_body_node(tmp_path,monkeypatch,features):
    from noesis.pipelines.source_acquisition import SourceAcquisitionState
    from reid.mv3dt_person_association import AcquisitionTime
    f,packet,monitor=joint_appearance_native_fixture(tmp_path,monkeypatch,features=features,feature_frame=2)
    cache=f.ingress.projective_bridge.cache
    native=SourceAcquisitionState(f.identity.run_id,'synthetic-quality-graph')
    proofs=[];join=f.identity.join_estimated_image_regions
    def captured(proof,**kwargs):
        proofs.append(proof);return join(proof,**kwargs)
    monkeypatch.setattr(f.identity,'join_estimated_image_regions',captured)
    def publish(packets):
        snapshot=f.publish(packets)
        for work,tracks in packets:
            owner=native.consume_owner_work(work.private,work.continuity.source_epoch,tracks)
            cache.bind_owner(work.private.original,work.continuity.source_epoch,work.continuity.frame_id,
                owner['members'],owner_retention=work.private.owner_retention)
            f.ingress.bind_published_swin(cache,work.private.original)
        return snapshot
    try:
        assert len(publish([packet(2,1)]).entities)==2
        assert len(publish([packet(1,2,separate=True),packet(2,2)]).entities)==3
        bound=[]
        if features:
            for row in f.ingress.originals.values():
                use=cache.use(row,AcquisitionTime(row.stamp.acquired_at_s,row.stamp.clock_domain,row.stamp.clock_basis))
                assert use is not None;bound.append(use)
        # No new extraction at frame3: reuse must retain frame2's real stamp
        # and ingress, never manufacture a current extraction or new time.
        final=publish([packet(1,3),packet(2,3)])
        assert len(final.entities)==(2 if features else 3)
        if features:
            assert len(proofs[-1].appearances)==3
            assert all(v.appearance.stamp.frame_id==2 for v in proofs[-1].appearances)
            assert all(s.frame_id==3 for s in proofs[-1].cohort)
            assert {v.ingress_monotonic_s for v in proofs[-1].appearances}=={v.appearance_ingress_monotonic_s for v in bound}
            assert {v.appearance.extraction_id for v in proofs[-1].appearances}=={v.appearance.extraction_id for v in bound}
        actual=f.examples.actual_node_wire(f,f.wire,'i139-qualified-original-'+str(features))
        assert [len(s['people']) for s in actual['snapshots']]==[2,3,2 if features else 3]
        assert all(e.stale_after_us==e.observed_at_us+2_000_000 for e in final.entities)
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced']
        f.world.close()
