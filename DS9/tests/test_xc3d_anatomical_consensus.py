"""Actual compact producer/bridge/service path; every camera/depth value is SYNTHETIC."""
from dataclasses import replace
from collections import deque
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import time
import importlib.util
import sys

import pytest

from noesis.identity_association_service import IdentityAssociationService
from noesis.pipelines.projective_continuity import ProjectiveQualityBridge
from noesis.pipelines.source_acquisition import SelectedSwinIngress, SourceAcquisitionState
from noesis.pipelines.world_quality_adapter import freeze_quality_work
from noesis_core.runtime_publication import RuntimePublicationGate
from reid.mv3dt_person_association import AnatomicalConsensusPolicy, StaticAnatomyQualification

def _helpers(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

body_packet=_helpers(Path(__file__).with_name('test_xc3d_body_depth_handoff.py'),'anatomical_body_helpers').body_packet

examples = _helpers(Path(__file__).with_name('test_xc3d_projective_continuity.py'), 'anatomical_original_helpers')


@pytest.fixture
def anatomical_native(tmp_path, monkeypatch):
    frame=[1]
    ns=time.time_ns()
    monkeypatch.setattr(time,'time_ns',lambda:ns+frame[0]*33_333_333)
    monkeypatch.setattr(time,'monotonic',lambda:100.+frame[0]/30.)
    def factory(run, spatial):
        base=examples.synthetic_policy(spatial)
        projection=((1000.,0.,0.,0.),(0.,1000.,0.,0.),(0.,0.,1.,0.))
        base=replace(base,cameras=tuple(replace(c,own_projection=projection,peer_projection=projection) for c in base.cameras))
        qualification=StaticAnatomyQualification('SYNTHETIC-static-fixture',sha256(repr(base).encode()).hexdigest(),'b'*64)
        return IdentityAssociationService.for_anatomical_consensus(run,spatial,AnatomicalConsensusPolicy(base,qualification))
    f=examples.NativeQualityFixture(tmp_path,identity_factory=factory)
    pipeline=SimpleNamespace(xc3d_world_quality_ingress=f.ingress,
        xc3d_private_owner_state=SourceAcquisitionState(f.identity.run_id,'synthetic-quality-graph'),
        xc3d_private_owner_diagnostics=deque(maxlen=256))
    pipeline.xc3d_swin_ingress=SelectedSwinIngress(f.identity.run_id,'synthetic-quality-graph',f.profile,None,None,timestamp_origin='source_pts')
    f.ingress.origin_bindings=examples.actual_lineage_callback(pipeline)
    bridge=ProjectiveQualityBridge(f.identity._anatomical_policy.projective_policy,lambda:pipeline.xc3d_swin_ingress.cache,joint_threshold=.35)
    f.ingress.projective_bridge=bridge
    processor=SimpleNamespace(pipeline=pipeline,tracking_pub=f.publisher,publication_gate=RuntimePublicationGate(),
        _stamp_footpoint_lifecycles=lambda *_:None,_commit_enqueued_world_output_watermarks=lambda *_a,**_k:None,
        _last_tracking_publish_ts_by_sensor={},_last_tracking_count_by_sensor={},bev_renderer=None)
    n=SimpleNamespace(f=f,pipeline=pipeline,callback=f.ingress.origin_bindings,processor=processor,bridge=bridge,frame=frame)
    try:yield n
    finally:f.world.close()


def anatomy_packet(n, source, frame, monkeypatch, *, qualified_ranges=True, separated_ankles=False):
    packet,tracks,*_=body_packet(n,source,frame,monkeypatch,alternate=True,separated_ankles=separated_ankles)
    row,=packet.rows
    anatomy=row.evidence.anatomy_depths
    assert len(anatomy)==2
    if qualified_ranges:
        # Explicit SYNTHETIC identity registration/range support; production
        # initially leaves these fields None and never makes this substitution.
        anatomy=tuple(replace(a,receipt=replace(a.receipt,registration_status='ok',registration_id='SYNTHETIC-reg'),
            registered_depth_interval_m=(a.receipt.registered_depth_m-.05,a.receipt.registered_depth_m+.05),
            depth_convention='camera_z_m') for a in anatomy)
    inputs=(replace(row.evidence,anatomy_depths=anatomy),)
    frozen=freeze_quality_work(packet.private,packet.continuity,inputs,raw_media_pts_ns=packet.raw_media_pts_ns)
    return replace(frozen,ingress_monotonic_s=packet.ingress_monotonic_s),tracks


def test_anatomical_actual_producer_bridge_service_stationary_membership_and_real_current_world(anatomical_native,monkeypatch):
    n=anatomical_native
    for frame in (1,2):
        for source in (1,2):
            packet,tracks=anatomy_packet(n,source,frame,monkeypatch)
            snapshot=examples.publish_lineage(n,packet,tracks)
    assert n.f.identity.ledger.stats().active_persons==1, (dict(n.bridge.counts),n.bridge.anatomical_originals)
    assert n.bridge.counts['result:current_registered_anatomical_consensus']>=1
    dto,=n.f.identity.spatial_memberships(at_interval=n.f.ingress._context)
    assert all(e.eligible for e in dto.eligibility)
    assert len(snapshot.entities)==1 and len(snapshot.entities[0].sources)==2


@pytest.mark.parametrize('defect,reason',[
    ('qualification','anatomical_static_pair_qualification_unavailable'),
    ('ranges','anatomical_registered_interval_unavailable'),
    ('separated-ankles','anatomical_representative_outside_sampled_support'),
])
def test_anatomical_actual_missing_prerequisite_is_honest_no_join(anatomical_native,monkeypatch,defect,reason):
    n=anatomical_native
    if defect=='qualification':n.f.identity._anatomical_policy=replace(n.f.identity._anatomical_policy,qualification=None)
    for frame in (1,2):
        for source in (1,2):
            packet,tracks=anatomy_packet(n,source,frame,monkeypatch,qualified_ranges=defect!='ranges',separated_ankles=defect=='separated-ankles')
            examples.publish_lineage(n,packet,tracks)
    assert n.f.identity.ledger.stats().active_persons==2
    assert n.bridge.counts['result:'+reason]>0, (dict(n.bridge.counts),n.bridge.anatomical_originals)


def test_anatomical_historical_ingestion_preserves_F_without_current_or_ledger_side_effect(anatomical_native,monkeypatch):
    n=anatomical_native
    old=[]
    for source in (1,2):
        packet,tracks=anatomy_packet(n,source,1,monkeypatch)
        old.append(packet)
        examples.publish_lineage(n,packet,tracks)
    for source in (1,2):
        packet,tracks=anatomy_packet(n,source,2,monkeypatch)
        examples.publish_lineage(n,packet,tracks)
    n.bridge.anatomical_originals.clear()
    n.bridge.history.clear()
    before=repr(vars(n.f.identity)),repr(vars(n.f.identity.ledger)),repr(n.bridge.ingresses)
    assert n.bridge.ingest_historical_quality(tuple(old),n.f.ingress.originals,n.f.ingress._context,time.monotonic()) is None
    assert (repr(vars(n.f.identity)),repr(vars(n.f.identity.ledger)),repr(n.bridge.ingresses))==before
    assert {v[0].original.stamp.frame_id for v in n.bridge.anatomical_originals.values()}=={1}
    assert all(o.stamp.frame_id==2 for o in n.f.ingress.originals.values())
    assert all(v[0].original.ingress_monotonic_s==old[0].ingress_monotonic_s for v in n.bridge.anatomical_originals.values())
    n.bridge.anatomical_originals.clear()
    n.bridge.ingest_historical_quality(tuple(old),n.f.ingress.originals,n.f.ingress._context,time.monotonic()+.36)
    assert not n.bridge.anatomical_originals


def test_anatomical_real_source_absence_clears_prior_original_support_before_evaluation(anatomical_native,monkeypatch):
    n=anatomical_native
    for source in (1,2):
        packet,tracks=anatomy_packet(n,source,1,monkeypatch)
        examples.publish_lineage(n,packet,tracks)
    assert any(k.source_id==1 for k in n.bridge.anatomical_originals)
    packet,tracks=examples.physical_packet(n,1,2,empty=True)
    examples.publish_lineage(n,packet,tracks)
    assert not any(k.source_id==1 for k in n.bridge.anatomical_originals)
    assert not any(k.source_id==1 for k in n.bridge.history)
