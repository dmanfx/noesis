"""Synthetic qualified geometry/time ONLY; actual World/service/kernel consumer.

No SDK/native authenticity, current-camera admission, live activation or physical
placement is established by these scalar fixtures. No private Identity registry.
"""

from collections import deque
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import selectors
import subprocess
import time
from types import SimpleNamespace

import pytest

from noesis.identity_association_service import (
    IdentityAssociationService, MeasurementCohort, MeasurementReference,
    ProducerAdmission, policy_identity,
)
from noesis.telemetry.publishers import TrackingTelemetryPublisher
from noesis_core.contracts.base import ArtifactFingerprint, ProducerRef
from noesis_core.contracts.scene_prior import (
    ScenePriorFloorPlane, ScenePriorFrameBinding, ScenePriorFrameRef,
    ScenePriorMetricFrame, metric_frame_revision_sha256,
)
from noesis_core.coordinate_frames import RevisionedFrame, revisioned_frame_sha256, revisioned_transform_sha256
from noesis_core.world_service import CanonicalWorldService, WorldArtifacts, WorldPublicationRefused
from noesis_core.contracts.identity import SubjectRef
from noesis_core.contracts.world import CanonicalWorldIdentityError, WorldSnapshot, validate_world_identity_uniqueness
from noesis_core.world import GlobalWorldFusion
from noesis_core.world.fusion import select_source_local_observation
from reid.mv3dt_person_association import (
    AcquisitionTime, AppearanceEvidence, AppearanceProfile, AssociationPolicy,
    HouseholdDescriptor, LinkProof, MeasurementEvidence, MemberKey, MemberRemoval,
    NativeTransition, ObservationStamp, SourceBinding, SourceObservation, SourceAbsence,
    appearance_extraction_identity,
    PhysicalPersonKey,
    SpatialAssociationPolicy, SpatialClockInterval, SpatialClockContext,
    SpatialLifecycleReceipt, SpatialPersonCandidate, SpatialCurrentPersonProof,
)


CAMERAS = {1: "kitchen", 2: "family-room"}


@pytest.mark.parametrize('own_measurement', [True, False])
def test_r113_missing_current_peer_body_alone_does_not_block_own_current_point(
        tmp_path, monkeypatch, own_measurement):
    """Public GREEN control, not a reproduction of native F1171's denial.

    Both native lifecycles stay ACTIVE/current. Only K's current pose and
    position disappear; no ambiguity, new join, birth or private flag is forged.
    """
    import importlib.util
    import sys

    name = 'r113_group_support_native_examples'
    path = Path(__file__).parents[1] / 'DS9/tests/test_xc3d_continuous_peer_gap.py'
    spec = importlib.util.spec_from_file_location(name, path)
    examples = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, examples)
    spec.loader.exec_module(examples)
    peer = examples._PeerGap(tmp_path, monkeypatch)
    try:
        first = peer.publish(peer.packet(1, 2), peer.packet(2, 2))
        old, = first.entities
        old_source, = examples._accepted(old)
        assert old_source.camera_id == 'family-room'
        joined, = peer.memberships()
        assert joined.association_basis == 'estimated_semantic_joint_correspondence'
        original_key = joined.physical_key
        seed_members = {m.observation.member for m in joined.members}
        for frame in (3, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82):
            kitchen = peer.packet(1, frame, pose=False, valid=False)
            family = peer.packet(2, frame, valid=own_measurement)
            assert kitchen[0].rows[0].evidence.body is None
            body = family[0].rows[0].evidence.body
            assert body.lifecycle.frame_id == frame
            assert all(body.keypoints[i][2] == .9 for i in (0, 1, 2, 5, 6, 11, 12))
            snapshot = peer.publish(kitchen, family)
            dto, = peer.memberships()
            assert dto.physical_key == original_key
            assert {m.observation.member for m in dto.members} == seed_members
            own = next(e for e in dto.eligibility if e.observation.member.source_id == 2)
            assert own.observation.frame_id == frame
            assert own.eligible and own.reason == 'current_active_own_source'
            if own_measurement:
                current, = snapshot.entities
                accepted, = examples._accepted(current)
                assert current.entity_id == old.entity_id
                assert accepted.camera_id == 'family-room'
                assert accepted.observation_id == own.observation.observation_id
                assert current.observed_at_us == family[0].continuity.observed_at_us > old.observed_at_us
                assert current.observed_at_us <= snapshot.published_at_us
            else:
                for held in snapshot.entities:
                    assert (held.position, held.covariance, held.observed_at_us,
                            held.stale_after_us) == (old.position, old.covariance,
                            old.observed_at_us, old.stale_after_us)
                    assert tuple(s.observation_id for s in examples._accepted(held)) == (
                        old_source.observation_id,)
        assert snapshot.published_at_us > old.stale_after_us
        assert len(snapshot.entities) == int(own_measurement)
        actual = peer.examples.actual_node_wire(
            peer.f, peer.f.wire, 'r113-group-peer-body-' + str(own_measurement))
        assert all(row['accepted'] for row in actual['diagnostics'])
        assert max(len(row['people']) for row in actual['snapshots']) == 1
        assert len(actual['snapshots'][-1]['people']) == int(own_measurement)
        assert peer.f.identity.stats().pending_reservations == 0
    finally:
        peer.close()


@pytest.mark.parametrize('own_measurement', [True, False])
def test_r117_pending_peer_vote_does_not_poison_current_own_original(
        tmp_path, monkeypatch, own_measurement):
    """Public semantic quarantine, not manufactured flags or renewed holds.

    Metric points/local acceptance and anatomy are synthetic; typed native
    custody, association, World, publisher and Node are the actual consumers.
    """
    import importlib.util
    import sys
    from noesis.pipelines.source_acquisition import freeze_owner_availability, freeze_owner_retention
    from noesis.pipelines.world_quality_adapter import capture_body_support
    from noesis.pipelines.projective_continuity import _OriginalBodyDiagnostic

    name='r117_pending_peer_native_examples'
    path=Path(__file__).parents[1]/'DS9/tests/test_xc3d_continuous_peer_gap.py'
    spec=importlib.util.spec_from_file_location(name,path)
    examples=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,name,examples);spec.loader.exec_module(examples)
    peer=examples._PeerGap(tmp_path,monkeypatch)

    def packet(source,frame,*,other=False,crowded=False,pose=True,valid=True):
        peer.at=frame
        raws=(0,) if source==1 else ((3,4) if other else (3,))
        work,tracks=peer.f.packet(source,frame,raws=raws,x=1.-(raws[0]-9)*3.,
            scale=1. if source==2 else 4.,households={r:{'identity_kind':'visitor',
                'stable_id':1004 if source==1 else 1000 if r==3 else 1005,
                'visitor_generation':1} for r in raws})
        inputs=[]
        for row,track in zip(work.rows,tracks):
            u=50. if source==1 else 40.
            y=30. if track['tracker_id']!=4 else (50. if crowded else 80.)
            track['bbox']=[u-10.,y-12.,20.,50.]
            joints=[(u,y+8.,.9)]*17
            hip=32. if source==1 and crowded else 12.
            for i,dx,dy in ((0,0,-8),(1,-2,-9),(2,2,-9),(3,-4,-8),(4,4,-8),
                            (5,-3,0),(6,3,0),(11,-2,hip),(12,2,hip)):
                joints[i]=(u+dx,y+dy,.9)
            if not pose or not valid and track['tracker_id']!=4:
                track.update(world=None,world_valid=False,world_measurement_accepted=False,
                    world_quality='invalid',world_source=None)
            body=capture_body_support(track,row.evidence.lifecycle,joints,pose_is_current=True) if pose else None
            inputs.append(replace(peer.q.capture_quality_input(track,row.evidence.lifecycle,
                row.evidence.measurements,row.evidence.resolution,row.evidence.depth),body=body))
        registry=peer.f.registries[source]
        continuity=registry.update_frame(source_id=source,camera_id=peer.f.labels[source],frame_id=frame,
            observed_at_us=work.continuity.observed_at_us,media_pts_ns=work.continuity.media_pts_ns,tracks=tracks)
        retention=freeze_owner_retention(continuity,registry.identity_retention_keys(source,
            media_pts_ns=work.private.media_pts_ns,max_gap_ns=430_000_000),430_000_000,work.private.original)
        private=replace(work.private,owner_retention=retention);mono=time.monotonic()
        peer.monitor.record_progress(source,now=mono)
        availability=freeze_owner_availability(private,continuity,peer.monitor,ingress_monotonic_s=mono)
        assert availability is not None and availability.owner is retention
        return replace(peer.q.freeze_quality_work(private,continuity,tuple(inputs)),
            ingress_monotonic_s=mono,owner_availability=availability),tracks

    try:
        seed=peer.publish(packet(1,2),packet(2,2));original,=seed.entities
        paired,=peer.memberships();key=paired.physical_key
        own_seed=next(m.observation for m in paired.members if m.observation.member.source_id==2)
        # A genuinely separate F member is first established with a FALSE K
        # edge. The next CURRENT enlarged K torso supports both F torsos.
        independent=peer.publish(packet(1,3),packet(2,3,other=True))
        assert len(independent.entities)==2
        comparisons=peer.f.identity.image_region_decision_diagnostics()['transitions'][-1]['pairs']
        assert sorted(p['semantic_state'] for p in comparisons)==[False,True]
        quarantined=peer.publish(packet(1,4,crowded=True),packet(2,4,other=True,crowded=True))
        comparisons=peer.f.identity.image_region_decision_diagnostics()['transitions'][-1]['pairs']
        assert len(comparisons)==2 and all(p['semantic_state'] is True for p in comparisons)
        assert any(p['vote_quarantine'] for p in comparisons)
        pair=next(d for d in peer.memberships() if d.physical_key==key)
        own=next(e for e in pair.eligibility if e.observation.member.source_id==2)
        ambiguous=next(e for e in pair.eligibility if e.observation.member.source_id==1)
        assert own.observation.member==own_seed.member
        assert own.observation.lifecycle_started_frame==own_seed.lifecycle_started_frame
        assert own.eligible and not ambiguous.eligible and ambiguous.reason=='image_region_ambiguous_member'
        old=next(e for e in quarantined.entities if e.entity_id==original.entity_id)
        old_source,=examples._accepted(old)
        assert old_source.camera_id=='family-room'
        peer.f.ingress.projective_bridge._body_diagnostic=_OriginalBodyDiagnostic((75/30.,))
        # Below both unchanged source/ingress continuity limits; enough exact
        # cohorts to pass the old wire deadline within the Node output bound.
        for frame in (5,15,25,35,45,55,65,75):
            absent=packet(1,frame,pose=False,crowded=True)
            current=packet(2,frame,other=True,crowded=True,valid=own_measurement)
            assert absent[0].rows[0].evidence.body is None
            assert current[1][0]['world_measurement_accepted'] is own_measurement
            assert all(current[0].rows[0].evidence.body.keypoints[i][2]==.9
                for i in (0,1,2,5,6,11,12))
            if own_measurement:assert current[0].rows[0].evidence.resolution.status=='measured'
            snapshot=peer.publish(absent,current)
            for entity in snapshot.entities:
                assert all(s.camera_id!='kitchen' for s in examples._accepted(entity))
                if entity.entity_id==old.entity_id:
                    if not own_measurement:assert entity.observed_at_us==old.observed_at_us
                    if entity.observed_at_us==old.observed_at_us:
                        assert (entity.position,entity.covariance,entity.stale_after_us)==(
                            old.position,old.covariance,old.stale_after_us)
                    else:
                        accepted,=examples._accepted(entity)
                        dto=next(d for d in peer.memberships() if d.physical_key==key)
                        exact=next(e.observation for e in dto.eligibility if e.observation.member.source_id==2)
                        assert accepted.observation_id==exact.observation_id
                        assert entity.observed_at_us==current[0].continuity.observed_at_us
                        assert entity.observed_at_us<=snapshot.published_at_us
        assert snapshot.published_at_us>old.stale_after_us
        pair=next(d for d in peer.memberships() if d.physical_key==key)
        own=next(e for e in pair.eligibility if e.observation.member.source_id==2)
        ambiguous=next(e for e in pair.eligibility if e.observation.member.source_id==1)
        assert not ambiguous.eligible and ambiguous.reason=='image_region_ambiguous_member'
        summary=peer.f.ingress.projective_bridge.diagnostics()['original_body_inspection_only']
        support=next(r['appearance']['group_support'] for r in summary['target_admission']['records']
            if r['source']==2)
        own_support=next(r for r in support['records'] if r['query'][0]==2 and r['query'][1]==3)
        assert own_support['cause']=='accepted' and own_support['reason'] is None
        assert len(own_support['members'])==2 and all(m['dormant'] and m['native']=='ACTIVE'
            and m['caps']['stamp_matches'] and m['caps']['retained'] and m['caps']['active']
            for m in own_support['members'])
        pending=next(m for m in own_support['members'] if m['stamp'][0]==1)
        assert pending['vote_blocked'] and pending['vote_pending']
        actual=peer.examples.actual_node_wire(peer.f,peer.f.wire,'r117-pending-peer-own-point')
        assert all(r['accepted'] for r in actual['diagnostics'])
        assert max(len(s['people']) for s in actual['snapshots'])==2
        assert peer.f.identity.stats().pending_reservations==0
        # Desired physical authority is the literal F current original ONLY:
        # no K vote, new SAME assertion, old point transfer or deadline renewal.
        assert own.eligible, (own.reason,
            [(r['query'],r.get('cause'),r.get('reason')) for r in support['records']],
            [len(s['people']) for s in actual['snapshots']])
        assert len(snapshot.entities)==len(actual['snapshots'][-1]['people'])==1+int(own_measurement)
    finally:
        peer.close()


def _r40_empty_spatial(actor, *, frame, evaluation_s):
    """Real empty source cohort; media evaluation is NOT publication wall time."""
    preps,_=actor.normalize({1:[]},frame=frame,native_tokens=True,world_resolver_confidence=.8)
    actor.context=SpatialClockContext(tuple(SpatialClockInterval(b,
        10.2,10.2,evaluation_s,evaluation_s) for b in actor.sources))
    members=actor.identity.spatial_memberships(at_interval=actor.context)
    return actor.world.prepare_spatial(1,preps,members,at_interval=actor.context,
        attribution_mode='physical_only_quarantine')


@pytest.mark.parametrize('wall_delta',[2_000_000,2_100_000])
def test_r40_wire_expiry_intersects_slow_media_and_actual_node_recovery(wall_delta):
    actor=SpatialActor();harness=ActualPublicationHarness(actor)
    first=harness.publish(actor.prepare_rows({1:[(1,1.,1,1)]}),2)
    original=first.entities[0]
    actor.us=10_200_000+wall_delta
    candidate=_r40_empty_spatial(actor,frame=3,evaluation_s=10.3)
    # Source age .1s remains eligible, but the ORIGINAL wire deadline has elapsed.
    assert candidate.publication.snapshot.published_at_us>=original.stale_after_us
    assert not candidate.publication.snapshot.entities
    expired=harness.publish(candidate,3)
    assert [e.event_type for e in candidate.publication.events]==['lost']
    assert not expired.entities
    actor.us+=100_000
    recovered=harness.publish(actor.prepare_rows({1:[(1,1.05,1,1)]},frame=4),4)
    assert len(recovered.entities)==1 and recovered.entities[0].entity_id==original.entity_id
    assert recovered.entities[0].observed_at_us==actor.us
    assert recovered.entities[0].stale_after_us==actor.us+2_000_000
    actual=actual_node_wire(actor,harness.wire,'r40-wire-expiry-'+str(wall_delta))
    assert [len(s['people']) for s in actual['snapshots']]==[1,0,1]


def test_r113_observer_raw_quality_selector_discard_and_failure_parity(monkeypatch):
    actor=SpatialActor()
    candidate=actor.prepare_rows({1:[(1,1.,1,1)]},source_changes={1:{'world_resolver_confidence':None}})
    row,=actor.world._private_spatial_diagnostics()['records']
    assert row['transaction']=='prepared' and row['raw_reason']=='resolver_confidence_absent'
    assert row['owner_eligible'] is True and row['selector_reason']=='resolver_confidence_absent'
    actor.world.discard(candidate)
    assert actor.world._private_spatial_diagnostics()['records'][0]['transaction']=='discarded'
    def broken(row):raise ValueError('optional observer')
    monkeypatch.setattr(actor.world._spatial_observer,'record',broken)
    actual=actor.commit_world(actor.prepare_rows({1:[(1,1.,1,1)]},frame=3,
        source_changes={1:{'world_resolver_confidence':None}}))
    assert not actual.snapshot.entities
    assert actor.world._private_spatial_diagnostics()['observer_errors']>0


def test_r113_observer_queued_commit_failure_is_not_admission():
    actor=SpatialActor()
    candidate=actor.prepare_rows({1:[(1,1.,1,1)]},source_changes={1:{'world_resolver_confidence':None}})
    actor.identity.reset_source(replace(actor.sources[0],source_epoch=1))
    with pytest.raises(WorldPublicationRefused):actor.commit_world(candidate)
    row,=actor.world._private_spatial_diagnostics()['records']
    assert row['transaction']=='commit_failed' and actor.world.current_snapshot() is None


def test_r113a_unavailable_observer_cannot_change_actual_prepare_commit():
    actor=SpatialActor()
    actor.world._spatial_observer=None
    first=actor.commit_world(actor.prepare_rows({1:[(1,1.,1,1)]}))
    assert len(first.snapshot.entities)==1
    actor.us+=100_000
    held=actor.commit_world(actor.prepare_rows({1:[(1,1.1,1,1)]},frame=3,
        source_changes={1:{'world_resolver_confidence':None}}))
    assert len(held.snapshot.entities)==1
    assert held.snapshot.entities[0].observed_at_us==first.snapshot.entities[0].observed_at_us
    assert actor.world._revision==2


def test_r113_observer_pins_actual_held_loss_source_age_before_wire_and_close_once(caplog):
    import logging
    actor=SpatialActor();harness=ActualPublicationHarness(actor)
    first=harness.publish(actor.prepare_rows({1:[(1,1.,1,1)]}),2)
    actor.us+=100_000
    held=harness.publish(_r40_empty_spatial(actor,frame=3,evaluation_s=11.1),3)
    assert held.entities
    actor.us+=100_000
    candidate=_r40_empty_spatial(actor,frame=4,evaluation_s=12.3)
    assert not candidate.publication.snapshot.entities
    row=next(r for r in actor.world._private_spatial_diagnostics()['records'] if r['kind']=='unexpired_held_loss')
    assert row['transaction']=='prepared' and row['wire_remaining_us']>0
    assert row['omission_reason']=='source_age_expired' and row['source_age_us']>2_000_000
    harness.publish(candidate,4)
    row=next(r for r in actor.world._private_spatial_diagnostics()['records'] if r['kind']=='unexpired_held_loss')
    assert row['transaction']=='committed'
    caplog.set_level(logging.INFO,logger='noesis_core.world_service')
    actor.world.close();actor.world.close()
    assert caplog.text.count('Private World decision summary:')==1
    assert first.entities[0].stale_after_us>candidate.publication.snapshot.published_at_us


def test_r113_observer_owner_denial_is_separate_from_raw_quality():
    actor,harness,houses=_r62_actor()
    rows={1:[(1,1.,1,1)],2:[(101,1.03,1,1)]}
    harness.publish(actor.prepare_image(rows,households=houses),2)
    actor.us+=100_000;actor.mixed=True
    candidate=actor.prepare_image(rows,frame=3,households=houses)
    records=actor.world._private_spatial_diagnostics()['records']
    row=next(r for r in records if r['kind']=='current_original_nonvote')
    assert row['raw_reason'] is None and row['owner_eligible'] is False
    assert row['owner_reason']==row['selector_reason']=='image_region_contradictory_cue_member'
    harness.publish(candidate,3)
    assert all(r['transaction']=='committed' for r in actor.world._private_spatial_diagnostics()['records'])
    assert all(h.identity_key not in json.dumps(records) for h in houses.values())


def test_r113_observer_does_not_hide_earlier_normalization_integrity_refusal():
    actor=SpatialActor()
    actor.commit_world(actor.prepare_rows({1:[(1,1.,1,1)]}))
    actor.us+=100_000
    with pytest.raises(WorldPublicationRefused,match='accepted_world_position_unavailable'):
        actor.prepare_rows({1:[(1,50.,1,1)]},frame=3)
    assert actor.world._private_spatial_diagnostics()['records']==[]


def test_r113_observer_bound_detachment_and_pinned_committed_loss():
    from noesis_core.world_service import _SpatialDecisionObserver
    observer=_SpatialDecisionObserver()
    pinned={'kind':'unexpired_held_loss','transaction':'committed','candidate':1,'source':1,'method':None,
            'original_id':'exact-original','wire_remaining_us':1}
    observer.record(pinned)
    for index in range(30):
        observer.record({'kind':'current_original_nonvote','candidate':index+2,'transaction':'prepared',
            'source':index,'method':'registered_body_column','original_id':'\u2603'*240,'raw_reason':'y'*240,
            'owner_reason':'z'*240,'selector_reason':'q'*240})
    encoded=observer.encoded()
    assert len(encoded.encode())<=8192
    view=json.loads(encoded)
    assert len(view['records'])<=8 and view['omitted_records']>0
    assert any(r.get('original_id')=='exact-original' for r in view['records'])
    view['records'][0]['original_id']='changed'
    assert 'changed' not in observer.encoded()
    observer.record({'kind':'unexpired_held_loss','transaction':'committed','candidate':99,'source':2,'method':None})
    assert any(r.get('original_id')=='exact-original' for r in json.loads(observer.encoded())['records'])
    assert all(not isinstance(v,(dict,list,tuple)) for row in observer.records for v in row.values())


def test_r40_media_expiry_still_applies_before_wire_deadline():
    actor=SpatialActor();harness=ActualPublicationHarness(actor)
    first=harness.publish(actor.prepare_rows({1:[(1,1.,1,1)]}),2)
    actor.us+=100_000
    candidate=_r40_empty_spatial(actor,frame=3,evaluation_s=12.3)
    assert candidate.publication.snapshot.published_at_us<first.entities[0].stale_after_us
    assert not harness.publish(candidate,3).entities
    actual=actual_node_wire(actor,harness.wire,'r40-media-first-expiry')
    assert [len(s['people']) for s in actual['snapshots']]==[1,0]


def test_r40_expiry_preparation_keeps_custody_until_atomic_commit_and_reset_refuses():
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','r40-k',1001,1),
            (2,101):HouseholdDescriptor('visitor','r40-f',1000,1)}
    first=harness.publish(actor.prepare_image({1:[(1,1.,1,1)],2:[(101,1.03,1,1)]},
        households=houses),2)
    custody=actor.world._fusion.spatial_original_attribution_claims()
    actor.us+=2_100_000
    candidate=_r40_empty_spatial(actor,frame=3,evaluation_s=10.3)
    assert not candidate.publication.snapshot.entities
    assert actor.world.current_snapshot() is first
    assert actor.world._fusion.spatial_original_attribution_claims()==custody
    before=(len(harness.wire),len(harness.journal),actor.world._revision)
    harness.before_commit=lambda:actor.identity.reset_source(replace(actor.sources[1],source_epoch=1))
    with pytest.raises(WorldPublicationRefused):harness.publish(candidate,3)
    assert actor.world.current_snapshot() is first
    assert before==(len(harness.wire),len(harness.journal),actor.world._revision)


def _r40_image_seed(*,two_pairs=False):
    actor=ImageQuarantineActor(live_track_continuity=True);harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','r40-k',1001,1),
            (2,101):HouseholdDescriptor('visitor','r40-f',1000,1),
            (2,102):HouseholdDescriptor('visitor','r40-distinct',1002,1)}
    rows={1:[(1,1.,1,1)],2:[(101,1.03,1,1),(102,4.,1,1)]}
    if two_pairs:
        rows[1].append((2,4.03,1,1));houses[(1,2)]=HouseholdDescriptor('visitor','r40-k-distinct',1003,1)
    first=harness.publish(actor.prepare_image(rows,households=houses),2)
    assert len(first.entities)==2
    return actor,harness,houses,rows,first


def test_r40_ambiguous_member_actual_service_world_publisher_node_two_people_recovery():
    actor,harness,houses,rows,first=_r40_image_seed()
    key=next(e.entity_id for e in first.entities if e.attribution_quarantine is not None)
    other=next(e.entity_id for e in first.entities if e.entity_id!=key)
    actor.us+=100_000
    rows={1:[(1,8.,1,1)],2:[(101,1.05,1,1),(102,4.05,1,1)]}
    candidate=actor.prepare_image(rows,frame=3,households=houses,image_y={(2,102):30.},
        source_changes={1:{'world_covariance':tuple(v*.01 for v in COV)}})
    state=actor.world._prepared_candidates[candidate._candidate_id][1]
    assert len(state.spatial_reservations)==2
    assert all(r.reference.owner.source_id==2 for r in state.spatial_reservations)
    pair=next(d for d in actor.memberships if len(d.members)==2)
    blocked=next(e for e in pair.eligibility if e.observation.member.source_id==1)
    assert not blocked.eligible and blocked.reason=='image_region_ambiguous_member'
    reference=next(r.reference for r in state.spatial_reservations if r.reference.owner.raw_id==101)
    pending=actor.identity.stats().pending_reservations
    strict=actor.identity.prepare_spatial_member(pair,reference,at_interval=actor.context)
    assert strict.disposition=='denied' and strict.reason=='settled_household_claim_conflict'
    assert actor.identity.stats().pending_reservations==pending
    second=harness.publish(candidate,3)
    assert {e.entity_id for e in second.entities}=={key,other}
    own=next(e for e in second.entities if e.entity_id==key)
    assert own.position.x==1.05 and own.subject.stable_id is None
    assert own.attribution_quarantine.stable_ids==(1000,1001)
    assert [s.camera_id for s in own.sources if s.accepted]==['family-room']
    assert next(e for e in second.entities if e.entity_id==other).position.x==4.05
    actor.us+=100_000
    rows[1]=[(1,1.1,1,1)]
    third=harness.publish(actor.prepare_image(rows,frame=4,households=houses,
        source_changes={1:{'world_covariance':tuple(v*.01 for v in COV)}}),4)
    assert all(e.eligible for d in actor.memberships for e in d.eligibility)
    assert next(e for e in third.entities if e.entity_id==key).position.x==1.1
    actual=actual_node_wire(actor,harness.wire,'r40-ambiguous-two-person-recovery')
    assert [len(s['people']) for s in actual['snapshots']]==[2,2,2]


def test_r40_both_ambiguous_preserve_original_custody_without_votes_until_wire_expiry():
    actor,harness,houses,rows,first=_r40_image_seed(two_pairs=True)
    actor.us+=100_000
    candidate=actor.prepare_image(rows,frame=3,households=houses,
        image_y={(1,2):30.,(2,102):30.})
    assert all(not e.eligible and e.reason=='image_region_ambiguous_member'
               for d in actor.memberships for e in d.eligibility)
    assert not actor.world._prepared_candidates[candidate._candidate_id][1].spatial_reservations
    held=harness.publish(candidate,3)
    assert [(e.position,e.observed_at_us,e.stale_after_us) for e in held.entities]==[
        (e.position,e.observed_at_us,e.stale_after_us) for e in first.entities]
    actor.us+=2_100_000
    assert not harness.publish(_r40_empty_spatial(actor,frame=4,evaluation_s=10.4),4).entities
    actual=actual_node_wire(actor,harness.wire,'r40-both-ambiguous-expiry')
    assert [len(s['people']) for s in actual['snapshots']]==[2,2,0]


def test_r40_queued_old_peer_grant_refuses_after_actual_ambiguity_revision():
    from reid.mv3dt_person_association import (EstimatedImageRegionProof,
        CurrentVisualDeliveryContext,CurrentVisualIngress)
    actor,harness,houses,rows,first=_r40_image_seed()
    actor.us+=50_000
    first=harness.publish(actor.prepare_image(rows,frame=3,households=houses,
        image_y={(2,102):30.}),3)
    actor.us+=50_000
    queued=actor.prepare_image(rows,frame=4,households=houses,image_y={(2,102):30.})
    before=(len(harness.wire),len(harness.journal),actor.world._revision)
    def changed():
        actor.image_sequence+=1
        originals=tuple(actor.image_originals.values())
        mono=100.+(actor.us-10_200_000)/1e6
        result=actor.identity.join_estimated_image_regions(EstimatedImageRegionProof(
            'r40-real-later-classification',tuple(o.stamp for o in originals),originals),
            at_interval=actor.context,delivery_context=CurrentVisualDeliveryContext(
                'synthetic-image-world',actor.image_sequence,mono,
                tuple(CurrentVisualIngress(o.stamp,o.ingress_monotonic_s) for o in originals)))
        assert result.disposition in {'accepted','observed'}
    harness.before_commit=changed
    with pytest.raises(WorldPublicationRefused):harness.publish(queued,3)
    assert actor.world.current_snapshot() is first
    assert before==(len(harness.wire),len(harness.journal),actor.world._revision)


def test_r40_exact_ambiguous_reason_does_not_whitelist_unknown_or_legacy():
    from noesis_core.world.fusion import spatial_rejection_category
    assert spatial_rejection_category('image_region_ambiguous_member')=='no_vote'
    for reason in ('image_region_ambiguous_member_typo','current_rejoin_proof_required',
                   'resolver_covariance_invalid','spatial_original_member_context'):
        assert spatial_rejection_category(reason)=='integrity'


@pytest.mark.parametrize('defect',['covariance','frame','unknown-reason','legacy-rejoin'])
def test_r40_ambiguous_member_never_masks_actual_world_integrity(defect,monkeypatch):
    actor,harness,houses,rows,first=_r40_image_seed()
    actor.us+=100_000
    changes={}
    if defect=='covariance':changes={1:{'world_covariance':(.1,1.,0.,1.,.1,0.,0.,0.,.1)}}
    if defect=='frame':changes={1:{'world_transform_sha256':'f'*64}}
    if defect in {'unknown-reason','legacy-rejoin'}:
        actual=actor.identity.spatial_memberships
        def malformed(**kw):
            dtos=actual(**kw)
            reason='image_region_ambiguous_member_typo' if defect=='unknown-reason' else 'current_rejoin_proof_required'
            return tuple(replace(d,eligibility=tuple(replace(e,reason=reason)
                if e.reason=='image_region_ambiguous_member' else e for e in d.eligibility)) for d in dtos)
        monkeypatch.setattr(actor.identity,'spatial_memberships',malformed)
    before=(len(harness.wire),len(harness.journal),actor.world._revision)
    with pytest.raises((WorldPublicationRefused,ValueError)):
        actor.prepare_image(rows,frame=3,households=houses,image_y={(2,102):30.},source_changes=changes)
    assert actor.world.current_snapshot() is first
    assert before==(len(harness.wire),len(harness.journal),actor.world._revision)
    assert actor.identity.stats().pending_reservations==0

# Exact diagnostic-only RR input fields; source artifact SHA256
# 17a5d537824dccf0fa036bc45f7e831b35c27c533f876faf3404ad7c2c7b7b42.
# metric_frame/independent_geometry_receipt NULL; these are NOT synthetic positives.
CURRENT_DIAGNOSTIC_BINDINGS = json.loads(r'''{"family-room":{"contract":"noesis.scene_prior.frame_binding","contract_version":1,"source_camera_calibration_sha256":"9b34e3ea8a3808475584a55383afcff96373a9e0845d33dc026a6659c975be93","source_floor_plane":{"frame":{"frame_id":"backend_world_m","revision":"b73b6a742e2d0d5936876875d904aa74c98996088f06efd04d552dac3f96a214"},"normal":[-0.09059934064225462,0.9936095819714694,0.06731833397872138],"offset_m":0.17558377759227575},"source_frame":{"frame_id":"backend_world_m","revision":"b73b6a742e2d0d5936876875d904aa74c98996088f06efd04d552dac3f96a214"},"source_world_alignment_sha256":"88e05c3259bde42265c8bdc97e1c86762ca7540402041ebcceadb407d72df82b","target_floor_plane":{"frame":{"frame_id":"backend_world_m","revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4"},"normal":[0,1,0],"offset_m":0},"target_frame":{"frame_id":"backend_world_m","revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4"},"target_from_source_col_major":[0.9467470066425219,-0.09059934064225462,-0.30896903548513316,0,0.0649604489274947,0.9936095819714693,-0.09230459733740415,0,0.31535732984752884,0.06731833397872138,0.9465822713434718,0,-3.1508635068958624,0.17446172385443262,5.462715819965061,1],"target_from_source_sha256":"5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217","target_revision_id":"vt_family_room_stream_rgbmesh_20260623T215850_645134993","target_revision_metadata_sha256":"67dee858ad3c5b0564e7d364b9522e59842f14b7c7de1c07659216e1defedd9c"},"kitchen":{"contract":"noesis.scene_prior.frame_binding","contract_version":1,"source_camera_calibration_sha256":"9b34e3ea8a3808475584a55383afcff96373a9e0845d33dc026a6659c975be93","source_floor_plane":{"frame":{"frame_id":"backend_world_m","revision":"b73b6a742e2d0d5936876875d904aa74c98996088f06efd04d552dac3f96a214"},"normal":[0,1,0],"offset_m":0},"source_frame":{"frame_id":"backend_world_m","revision":"b73b6a742e2d0d5936876875d904aa74c98996088f06efd04d552dac3f96a214"},"source_world_alignment_sha256":"88e05c3259bde42265c8bdc97e1c86762ca7540402041ebcceadb407d72df82b","target_floor_plane":{"frame":{"frame_id":"backend_world_m","revision":"sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483"},"normal":[0,1,0],"offset_m":0},"target_frame":{"frame_id":"backend_world_m","revision":"sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483"},"target_from_source_col_major":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],"target_from_source_sha256":"af1290aa956661c7ece73215f4fe3ee0fc25a7fbef0c246fed2de66f12abd844","target_revision_id":"vt_kitchen_stream_rgbmesh_20260623T215835_609188061","target_revision_metadata_sha256":"47c38dc5358b1ca0a11be74c3eb56ca08289e2069ce53d43401f2ef60c249e44"}}''')
COV = (.1, 0., .025, 0., .01, 0., .025, 0., .2)


def test_spatial_transaction_old_strict_owner_cannot_enter_spatial_route():
    actor = OrderedActor()
    preps, _ = actor.normalize({1: [(1, 1., 1, 1)], 2: [(101, 8., 1, 1)]},
                               native_tokens=True, world_resolver_confidence=.8)
    originals = tuple(o for prep in preps for o in prep.publication.observations)
    rows = actor.world._spatial_measurements_locked(1, preps)
    assert all(row[0] is original for row, original in zip(rows, originals, strict=True))
    assert all(row[1:] == (.8, None) for row in rows)
    assert all(row[0].payload.world.covariance.values == COV for row in rows)
    journal = []
    actor.world._journal = SimpleNamespace(append_many=lambda *a, **k: journal.append(a))
    before = (actor.identity.stats(), dict(actor.world._sequence_by_source), actor.world._revision,
              actor.world._event_sequence)
    with pytest.raises(WorldPublicationRefused) as refusal:
        # No invented capability or interval type: the seam must NOT inspect
        # or accept proposed R8 fields until its actual owner API exists.
        context = SpatialClockContext(tuple(SpatialClockInterval(b, 10.2, 10.2, 10.2, 10.2) for b in actor.sources))
        actor.world.prepare_spatial(1, preps, (), at_interval=context)
    assert refusal.value.reason == "spatial_route_required"
    assert refusal.value.household_conflict is None
    assert actor.world.current_snapshot() is None and not journal
    assert before == (actor.identity.stats(), dict(actor.world._sequence_by_source), actor.world._revision,
                      actor.world._event_sequence)
    for prep in preps:
        actor.world.discard(prep)
    assert not actor.world._prepared_candidates


@pytest.mark.parametrize("changes,reason", [
    ({"world_covariance": None}, "resolver_covariance_absent"),
    ({"world_covariance": (.1, 1., 0., 1., .1, 0., 0., 0., .1)}, "resolver_covariance_invalid"),
    ({"world_resolver_confidence": None}, "resolver_confidence_absent"),
    ({"world_resolver_confidence": float("nan")}, "resolver_confidence_invalid"),
    ({"world_resolver_confidence": True}, "resolver_confidence_invalid"),
    ({"world_measurement_accepted": False}, "non_current_measurement"),
])
def test_spatial_quality_raw_presence_cannot_use_legacy_fallback(changes, reason):
    actor = OrderedActor()
    preps, _ = actor.normalize({1: [(1, 1., 1, 1)]}, native_tokens=True,
                               **({"world_resolver_confidence": .8} | changes))
    rows = actor.world._spatial_measurements_locked(1, preps)
    assert rows[0][2] == reason
    assert select_source_local_observation(rows)[0] is None
    actor.world.discard(preps[0])


def test_spatial_quality_original_receipt_bounds_identity_and_consumption():
    actor, foreign = OrderedActor(), OrderedActor()
    preps, _ = actor.normalize({1: [(1, 1., 1, 1)]}, native_tokens=True, world_resolver_confidence=.8)
    foreign_preps, _ = foreign.normalize({1: [(1, 1., 1, 1)]}, native_tokens=True, world_resolver_confidence=.8)
    with pytest.raises(RuntimeError, match="another service"):
        actor.world._spatial_measurements_locked(1, foreign_preps)
    with pytest.raises(RuntimeError, match="stale or consumed"):
        actor.world._spatial_measurements_locked(1, (replace(preps[0]),))
    with pytest.raises(ValueError, match="one exact source"):
        actor.world._spatial_measurements_locked(1, preps * 2)
    with pytest.raises(ValueError, match="one to three"):
        actor.world._spatial_measurements_locked(1, preps * 4)
    with pytest.raises(ValueError, match="trigger source"):
        actor.world._spatial_measurements_locked(2, preps)
    actor.world.discard(preps[0])
    with pytest.raises(RuntimeError, match="stale or consumed"):
        actor.world._spatial_measurements_locked(1, preps)
    foreign.world.discard(foreign_preps[0])


def test_spatial_quality_shared_rank_preserves_actual_strict_metric_ci():
    # NEW directly affected rank/normalization regression, real service/kernel
    # and old strict World path, SYNTHETIC admitted metric/time bindings only.
    actor = OrderedActor()
    candidate, _, _, _ = actor.two_people(native_tokens=True)
    publication = actor.world.commit(candidate, association_at=actor.at())
    assert len(publication.snapshot.entities) == 2
    for entity in publication.snapshot.entities:
        assert len(entity.sources) == 2 and all(source.accepted for source in entity.sources)
        assert entity.covariance is not None
        assert entity.covariance.values[2] != 0.  # Complete correlated matrix remains.
        assert min(s.position.x for s in entity.sources) <= entity.position.x <= max(s.position.x for s in entity.sources)


def metric(offset=0.):
    physical = (ArtifactFingerprint(role="camera_calibration_physical", sha256="a" * 64),
                ArtifactFingerprint(role="world_alignment_physical", sha256="b" * 64))
    revision = metric_frame_revision_sha256("backend_world_m", (0., 1., 0.), offset,
                                            tuple(p.sha256 for p in physical))
    frame = ScenePriorFrameRef(frame_id="backend_world_m", revision=revision)
    return ScenePriorMetricFrame(contract="noesis.scene_prior.metric_frame", contract_version=1,
                                 frame=frame, units="meters", accepted=True, physical_provenance=physical,
                                 floor_plane=ScenePriorFloorPlane(frame=frame, normal=(0., 1., 0.), offset_m=offset))


def frame_binding(source, target=None):
    target = target or metric()
    calibration, alignment = str(source) * 64, str(source + 2) * 64
    raw = ScenePriorFrameRef(frame_id="backend_world_m", revision=revisioned_frame_sha256(
        "backend_world_m", artifact_sha256s=(calibration, alignment)))
    # Legitimate DIFFERENT rigid source edges into exactly the SAME target.
    matrix = (1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
              0. if source == 1 else 5., -target.floor_plane.offset_m, 0., 1.)
    digest = revisioned_transform_sha256(RevisionedFrame(raw.frame_id, raw.revision),
                                         RevisionedFrame(target.frame.frame_id, target.frame.revision), matrix)
    return ScenePriorFrameBinding(
        contract="noesis.scene_prior.frame_binding", contract_version=2,
        source_frame=raw, target_frame=target.frame,
        source_camera_calibration_sha256=calibration, source_world_alignment_sha256=alignment,
        target_revision_id="synthetic-target", target_revision_metadata_sha256="e" * 64,
        target_from_source_col_major=matrix, target_from_source_sha256=digest,
        source_floor_plane=ScenePriorFloorPlane(frame=raw, normal=(0., 1., 0.), offset_m=0.),
        target_floor_plane=target.floor_plane, metric_frame=target,
        artifact_revision_id=f"synthetic-artifact-{source}",
        source_camera_calibration_physical_sha256="a" * 64,
        source_world_alignment_physical_sha256="b" * 64,
    )


class OrderedActor:
    """Owns the real service and World; queued media carries no service mutation."""

    def __init__(self, run_id="run-synthetic"):
        self.frames = {s: frame_binding(s) for s in CAMERAS}
        self.raw_frames = dict(self.frames)
        self.rasters = {s: (1920, 1080) for s in CAMERAS}
        self.sources = tuple(SourceBinding(CAMERAS[s], s, 0, self.frames[s].source_frame.revision,
                                          self.frames[s].target_from_source_sha256,
                                          "synthetic-capture", "ntp") for s in CAMERAS)
        selected = AppearanceProfile("synthetic-selected-swin", "synthetic-layer", 512, "synthetic-profile")
        self.policy = AssociationPolicy(self.sources, "backend_world_m", metric().frame.revision,
                                        selected.model_id, selected, (("kitchen", "family-room"),))
        self.admission = self.admitted(self.policy, run_id)
        self.identity = IdentityAssociationService(run_id, self.admission)
        self.us = 10_200_000
        self.world = CanonicalWorldService(
            producer=ProducerRef(runtime="test", instance_id="synthetic", run_id=run_id, software_revision="fixture"),
            artifacts=self.artifacts, clock_us=lambda: self.us,
            association_service=self.identity, physical_admission=lambda: self.admission,
            physical_bindings=lambda: self.frames, physical_rasters=lambda: self.rasters,
        )
        self.queue = deque()
        self.tracking_rows = {}

    @staticmethod
    def admitted(policy, run_id="run-synthetic"):
        return ProducerAdmission("admitted", "synthetic-source-receipt", "synthetic-only",
                                 run_id, policy, policy_identity(policy),
                                 "synthetic-metric-receipt", "synthetic-time-receipt",
                                 "synthetic-independent-receipt", "synthetic")

    def artifacts(self, source, metadata):
        b = self.raw_frames[source]
        return WorldArtifacts(
            calibration=ArtifactFingerprint(role="camera_calibration", sha256="9" * 64, version=b.source_frame.revision),
            model=ArtifactFingerprint(role="tracking_models", sha256="5" * 64),
            config=ArtifactFingerprint(role="runtime_config", sha256="6" * 64),
            camera_calibration_sha256=b.source_camera_calibration_sha256,
        )

    def at(self, seconds=None):
        return AcquisitionTime(self.us / 1_000_000 if seconds is None else seconds, "synthetic-capture", "ntp")

    def commit_world(self, receipt):
        return self.world.commit(receipt, association_at=self.at())

    def native_source_record(self, source, row, frame, household):
        epoch = next(s.source_epoch for s in self.sources if s.source_id == source)
        return SourceObservation(ObservationStamp(
            MemberKey(self.world.producer.run_id, CAMERAS[source], source, epoch,
                      row["tracker_id"], row["tracker_lifecycle_generation"]),
            frame, f"synthetic-native:{source}:{row['tracker_id']}:{frame}",
            self.us / 1_000_000, "synthetic-capture", frame / 10, "ntp", row["lifecycle_started_frame"],
        ), household)

    def normalize(self, rows, frame=2, native_tokens=False, households=None, **changes):
        """Actual public World producer normalization, original opaque receipts."""
        preps, native = [], {}
        for source, people in rows.items():
            b = self.frames[source]
            epoch = next(s.source_epoch for s in self.sources if s.source_id == source)
            tracks = []
            for raw, x, generation, started in people:
                row = {
                    "camera_id": CAMERAS[source], "tracker_id": raw, "frame_id": frame,
                    "source_epoch": epoch, "tracker_lifecycle_generation": generation,
                    "lifecycle_started_frame": started, "trail_segment_id": 1,
                    "observed_at_us": self.us, "captured_at_us": self.us,
                    "capture_time_status": "synced", "media_pts_ns": frame * 100_000_000,
                    "bbox": [10., 20., 30., 40.], "image_size": [1920, 1080],
                    "confidence": .9, "tracker_confidence": .9,
                    "world": [x, 0., 2.], "world_valid": True,
                    "world_frame": "backend_world_m", "world_frame_revision": b.target_frame.revision,
                    "world_transform_sha256": b.target_from_source_sha256,
                    "world_calibration_sha256": b.source_camera_calibration_sha256,
                    "calibration_revision": b.source_frame.revision, "world_covariance": COV,
                    "world_quality": "good", "world_source": "person_anchor_floor_only",
                    "world_measurement_accepted": True, "world_support_state": "floor",
                    "world_posture": "standing", "identity_kind": "unknown",
                    "world_support_surface": {"surface_id": "synthetic-floor", "revision": metric().frame.revision,
                                              "normal_world": [0., 1., 0.], "offset_m": 0.,
                                              "extent_xz": [[-10., -10.], [10., -10.], [10., 10.], [-10., 10.]]},
                }
                row.update(changes)
                household = (households or {}).get((source, raw))
                if household is not None:
                    row.update(identity_kind=household.state, resident_uuid=(household.identity_key if household.state == "resident" else None),
                               stable_id=household.stable_id, visitor_generation=household.visitor_generation)
                tracks.append(row)
            originals = tuple(self.native_source_record(source, row, frame,
                (households or {}).get((source, row["tracker_id"]), HouseholdDescriptor()))
                for row in tracks) if native_tokens else ()
            prep = self.world.prepare(source, tracks,
                                      metadata={"camera_id": CAMERAS[source], "source_epoch": epoch},
                                      physical_observations=originals)
            preps.append(prep)
            for envelope, row in zip(prep.publication.observations, tracks, strict=True):
                self.tracking_rows[envelope.observation_id] = dict(row)  # Original producer rows, including REAL lifecycle generation.
                member = MemberKey(self.world.producer.run_id, CAMERAS[source], source, epoch,
                                   row["tracker_id"], row["tracker_lifecycle_generation"])
                stamp = ObservationStamp(member, frame, envelope.observation_id, self.us / 1_000_000,
                                         "synthetic-capture", frame / 10, "ntp", row["lifecycle_started_frame"])
                native[(source, row["tracker_id"])] = next((o for o in originals if o.stamp.observation_id == envelope.observation_id), SourceObservation(stamp))
        return tuple(preps), native

    def proof(self, left, right, token, x):
        def measured(observation, dx):
            b = self.frames[observation.stamp.member.source_id]
            return MeasurementEvidence(observation.stamp, observation.stamp.member,
                                       "measurement:" + observation.stamp.observation_id,
                                       "backend_world_m", b.target_frame.revision, b.source_frame.revision,
                                       b.target_from_source_sha256, dx, 2.,
                                       independence_id="independent:" + observation.stamp.observation_id)
        def appearance(observation):
            profile = self.policy.appearance_profile
            return AppearanceEvidence(observation.stamp, "appearance:" + observation.stamp.observation_id,
                                      profile.model_id, observation.stamp.acquired_at_s, observation.stamp.clock_domain,
                                      appearance_extraction_identity(observation.stamp, profile), profile)
        return LinkProof(token, measured(left, x), measured(right, x + .03),
                         appearance(left), appearance(right), .9)

    @staticmethod
    def cohort(dto, local=None):
        references = dto.proof.measurements if dto.proof else (MeasurementReference.from_evidence(local),)
        return MeasurementCohort(dto.run_id, dto.association_revision, dto.policy_identity,
                                 dto.policy.target_frame, dto.policy.world_revision, references)

    def two_people(self, native_tokens=False):
        old = SourceObservation(ObservationStamp(MemberKey(self.world.producer.run_id, "kitchen", 1, 0, 2, 1),
                                                1, "synthetic-original-old-B", 10.1,
                                                "synthetic-capture", .1, "ntp", 1))
        assert self.identity.observe(old).disposition == "observed"
        preps, observations = self.normalize({1: [(1, 0., 1, 1), (22, 3., 1, 1)],
                                              2: [(101, .03, 1, 1), (102, 3.03, 1, 1)]}, native_tokens=native_tokens)
        a, ar, b, br = (observations[(1, 1)], observations[(2, 101)],
                         observations[(1, 22)], observations[(2, 102)])
        for observation in (a, ar, br):
            assert self.identity.observe(observation).disposition == "observed"
        pa, pb = self.proof(a, ar, "ordinary-A", 0.), self.proof(b, br, "adopted-B", 3.)
        first_a = self.identity.link(pa, self.at())
        event = NativeTransition("synthetic-adoption-B", "adoption", old.stamp, b.stamp, br.stamp, 10.2, "synthetic-capture")
        first_b = self.identity.transition(event, b, self.at(), pb)
        assert first_a.disposition == first_b.disposition == "accepted", (first_a.reason, first_b.reason)
        withdrawals = tuple(set(first_a.withdrawn_members + first_b.withdrawn_members))
        retired = tuple(set(first_a.retired_keys + first_b.retired_keys))
        revision = self.identity.stats().association_revision
        # REQUIRED R5: replay the original operation, NOT link(adoption proof).
        final_a = self.identity.link(pa, self.at()).dto
        final_b = self.identity.transition(event, b, self.at(), pb).dto
        assert final_a.association_revision == final_b.association_revision == revision
        assert self.identity.stats().association_revision == revision
        assert final_b.proof.receipt is first_b.dto.proof.receipt
        pairs = ((final_b, self.cohort(final_b)), (final_a, self.cohort(final_a)))
        prepared = self.world.prepare_physical(1, preps, pairs, at=self.at(),
                                               withdrawn_members=withdrawals, retired_keys=retired)
        return prepared, (final_a, final_b), (pa, pb, event, b), observations


class SpatialActor(OrderedActor):
    """REAL public spatial owner + World. ALL qualification is synthetic."""

    def __init__(self):
        super().__init__()
        self.sources = tuple(replace(b, clock_domain=f"synthetic-source-{b.source_id}") for b in self.sources)
        self.spatial_policy = SpatialAssociationPolicy(self.sources, self.policy.appearance_profile,
                                                       (("kitchen", "family-room"),))
        self.identity = IdentityAssociationService.for_spatial(self.world.producer.run_id, self.spatial_policy)
        self.world = CanonicalWorldService(producer=self.world.producer, artifacts=self.artifacts,
            clock_us=lambda: self.us, association_service=self.identity, spatial_bindings=lambda: self.frames)
        self.originals = {}
        self.context = None

    def native_source_record(self, source, row, frame, household):
        original = super().native_source_record(source, row, frame, household)
        binding = next(b for b in self.sources if b.source_id == source)
        return replace(original, stamp=replace(original.stamp, clock_domain=binding.clock_domain, clock_basis=binding.clock_basis))

    def clock_context(self, *, qualified=False):
        intervals = []
        for binding in self.sources:
            acquired = max((o.stamp.acquired_at_s for o in self.originals.values()
                            if o.stamp.member.source_id == binding.source_id and o.stamp.member.source_epoch == binding.source_epoch), default=self.us / 1_000_000)
            intervals.append(SpatialClockInterval(binding, acquired, acquired, self.us / 1_000_000, self.us / 1_000_000))
        return SpatialClockContext(tuple(intervals), "synthetic-only-comparability" if qualified else None,
            "synthetic-common" if qualified else None,
            tuple((b.source_id, 0., 0.) for b in self.sources) if qualified else (),
            tuple((b.source_id, 0., 1000.) for b in self.sources) if qualified else (), "synthetic" if qualified else None)

    def prepare_rows(self, rows, *, frame=2, join=False, source_changes=None, households=None,
                     withdrawn_members=(), retired_keys=(), attribution_mode="strict"):
        preps, current = [], {}
        for source, people in rows.items():
            part, native = self.normalize({source: people}, frame=frame, native_tokens=True,
                households=households, **({"world_resolver_confidence": .8} | (source_changes or {}).get(source, {})))
            preps.extend(part)
            current.update(native)
            for original in native.values():
                result = self.identity.observe_spatial(original, SpatialLifecycleReceipt(
                    "synthetic-native-active:" + original.stamp.observation_id, original.stamp, "ACTIVE"))
                assert result.disposition == "observed" or (result.disposition == "unresolved"
                    and result.reason == "current_cross_source_proof_required"), result
                self.originals[original.stamp.member] = original
        if join:
            lefts = [o for o in current.values() if o.stamp.member.source_id == 1]
            rights = [o for o in current.values() if o.stamp.member.source_id == 2]
            candidates = []
            for left in lefts:
                for right in rights:
                    pair = self.proof(left, right, "synthetic-appearance", 0.)
                    same = right.stamp.member.raw_id == left.stamp.member.raw_id + 100
                    candidates.append(SpatialPersonCandidate(left.stamp, right.stamp,
                        pair.left_appearance, pair.right_appearance, .9 if same else .2,
                        "synthetic-copy:" + left.stamp.observation_id, "synthetic-copy:" + right.stamp.observation_id,
                        .03 if same else 3., "synthetic-independent-geometry", "synthetic-qualified-frame",
                        "synthetic-qualified-revision", tuple(self.frames[s].source_frame.revision for s in (1, 2)),
                        tuple(self.frames[s].target_from_source_sha256 for s in (1, 2)),
                        ("synthetic-independent:" + left.stamp.observation_id, "synthetic-independent:" + right.stamp.observation_id)))
            proof = SpatialCurrentPersonProof("synthetic-current-person:" + str(frame),
                tuple(o.stamp for o in current.values()), tuple(candidates),
                tuple((c.left.member, c.right.member) for c in candidates if c.similarity >= .6), "synthetic")
            joined = self.identity.join_spatial(proof, at_interval=self.clock_context(qualified=True))
            assert joined.disposition == "accepted", joined
            withdrawn_members += joined.withdrawn_members
            retired_keys += joined.retired_keys
        self.context = self.clock_context()
        memberships = self.identity.spatial_memberships(at_interval=self.context)
        self.memberships = memberships
        return self.world.prepare_spatial(next(iter(rows)), tuple(preps), memberships, at_interval=self.context,
            withdrawn_members=withdrawn_members, retired_keys=retired_keys, attribution_mode=attribution_mode)

    def commit_world(self, receipt):
        return self.world.commit(receipt, association_at_interval=self.context)


@pytest.mark.parametrize("kind", ["subject", "sid"])
def test_attribution_r4_strict_refusal_explicit_physical_wire_and_moving_heads(kind):
    actor = SpatialActor()
    houses = {(1, 1): HouseholdDescriptor("resident", "synthetic-claim-a", 9),
              (1, 2): HouseholdDescriptor("resident", "synthetic-claim-a" if kind == "subject" else "synthetic-claim-b", 10 if kind == "subject" else 9),
              (1, 3): HouseholdDescriptor("resident", "synthetic-unaffected", 11)}
    rows = {1: [(1, 1., 1, 1), (2, 3., 1, 1), (3, 6., 1, 1)]}
    with pytest.raises(WorldPublicationRefused, match="settled_household_claim_conflict"):
        actor.prepare_rows(rows, households=houses)
    assert actor.world.current_snapshot() is None and actor.identity.stats().pending_reservations == 0
    harness = ActualPublicationHarness(actor)
    actor.us += 100_000
    candidate = actor.prepare_rows(rows, frame=3, households=houses, attribution_mode="physical_only_quarantine")
    claims = actor.identity.ledger.household_claim_owners()
    first = harness.publish(candidate, 3)
    quarantined = [e for e in first.entities if e.attribution_quarantine]
    assert len(quarantined) == 2 and len(first.entities) == 3
    assert len({e.entity_id for e in first.entities}) == 3
    for entity in quarantined:
        assert entity.subject.kind == "provisional" and entity.subject.stable_id is None
        assert len(entity.attribution_quarantine.owners) == 2
    assert next(e for e in first.entities if not e.attribution_quarantine).subject.stable_id == 11
    actor.us += 100_000
    second = harness.publish(actor.prepare_rows({1: [(1, 1.1, 1, 1), (2, 3.1, 1, 1), (3, 6.1, 1, 1)]},
        frame=4, households=houses, attribution_mode="physical_only_quarantine"), 4)
    assert {e.entity_id for e in second.entities} == {e.entity_id for e in first.entities}
    assert [o.household for _, o in claims] == [h for _, h in sorted((k, v) for k, v in houses.items())]
    assert [o.household for _, o in actor.identity.ledger.household_claim_owners()] == [o.household for _, o in claims]
    actual = actual_node_wire(actor, harness.wire, f"r4-quarantine-{kind}")
    assert all(d["accepted"] for d in actual["diagnostics"])
    assert [len(s["people"]) for s in actual["snapshots"]] == [3, 3]
    for person in actual["snapshots"][-1]["people"]:
        assert len(person["trail"]) == 2
        if person["attributionQuarantine"]:
            assert person["label"] == "Person (attribution quarantined)" and person["stableId"] is None


def test_attribution_r4_retained_original_component_reset_recovery_and_expiry():
    actor, harness = SpatialActor(), None
    harness = ActualPublicationHarness(actor)
    houses = {(1, 1): HouseholdDescriptor("resident", "synthetic-same", 9),
              (2, 2): HouseholdDescriptor("resident", "synthetic-same", 10)}
    first = harness.publish(actor.prepare_rows({1: [(1, 1., 1, 1)]}, households=houses), 2)
    a = first.entities[0]
    actor.us += 100_000
    conflict = harness.publish(actor.prepare_rows({2: [(2, 3., 1, 1)]}, frame=3, households=houses,
        attribution_mode="physical_only_quarantine"), 3, source_id=2)
    retained_a = next(e for e in conflict.entities if e.entity_id == a.entity_id)
    assert retained_a.attribution_quarantine and retained_a.position == a.position
    assert retained_a.observed_at_us == a.observed_at_us and retained_a.stale_after_us == a.stale_after_us
    assert retained_a.continuity_segment_id == a.continuity_segment_id
    actor.us += 100_000
    reset_binding = replace(actor.sources[1], source_epoch=1)
    barrier = actor.identity.reset_source(reset_binding)
    actor.sources = (actor.sources[0], reset_binding)
    recovered = harness.publish(actor.prepare_rows({1: [(1, 1.1, 1, 1)]}, frame=4, households=houses,
        withdrawn_members=barrier.withdrawn_members, retired_keys=barrier.retired_keys), 4)
    assert len(recovered.entities) == 1 and recovered.entities[0].entity_id == a.entity_id
    assert recovered.entities[0].attribution_quarantine is None and recovered.entities[0].subject.stable_id == 9
    actor.us += 2_100_000
    expired = harness.publish(actor.prepare_rows({1: []}, frame=30), 30)
    assert not expired.entities
    actual = actual_node_wire(actor, harness.wire, "r4-retained-reset-expiry")
    assert all(d["accepted"] for d in actual["diagnostics"])
    assert [len(s["people"]) for s in actual["snapshots"]] == [1, 2, 1, 0]


def test_attribution_r4_transitive_component_and_queued_revision_atomicity():
    actor = SpatialActor()
    harness = ActualPublicationHarness(actor)
    houses = {(1, 1): HouseholdDescriptor("resident", "synthetic-ab", 9),
              (1, 2): HouseholdDescriptor("resident", "synthetic-ab", 10),
              (1, 3): HouseholdDescriptor("resident", "synthetic-c", 10)}
    candidate = actor.prepare_rows({1: [(1, 1., 1, 1), (2, 3., 1, 1), (3, 6., 1, 1)]},
        households=houses, attribution_mode="physical_only_quarantine")
    assert all(len(e.attribution_quarantine.component_entity_ids) == 3 for e in candidate.publication.snapshot.entities)
    first = harness.publish(candidate, 2)
    actual = actual_node_wire(actor, harness.wire, "r4-transitive-component")
    assert all(d["accepted"] for d in actual["diagnostics"])
    assert len(actual["snapshots"][0]["people"]) == 3
    assert all(p["attributionQuarantine"] for p in actual["snapshots"][0]["people"])
    actor.us += 100_000
    candidate = actor.prepare_rows({1: [(1, 1.1, 1, 1), (2, 3.1, 1, 1), (3, 6.1, 1, 1)]},
        frame=3, households=houses, attribution_mode="physical_only_quarantine")
    before = actor.identity.stats().spatial_member_frontiers
    actor.identity.reset_source(replace(actor.sources[0], source_epoch=1))
    with pytest.raises(WorldPublicationRefused):
        actor.commit_world(candidate)
    assert actor.world.current_snapshot() is first and actor.world._revision == 1
    assert actor.identity.stats().spatial_member_frontiers <= before
    assert actor.identity.stats().pending_reservations == 0


@pytest.mark.parametrize("defect", ["missing-component", "changed-clock", "unknown-mode", "no-position"])
def test_attribution_r4_complete_component_no_partial_grants_and_no_vote(defect, monkeypatch):
    actor = SpatialActor()
    harness = ActualPublicationHarness(actor)
    houses = {(1, 1): HouseholdDescriptor("resident", "synthetic-same", 9),
              (2, 2): HouseholdDescriptor("resident", "synthetic-same", 10)}
    rows = {1: [(1, 1., 1, 1)], 2: [(2, 3., 1, 1)]}
    first = harness.publish(actor.prepare_rows(rows, households=houses,
        attribution_mode="physical_only_quarantine"), 2)
    actor.us += 100_000
    before = (actor.world._revision, actor.identity.stats().spatial_member_frontiers,
              len(harness.wire), len(harness.journal))
    if defect == "missing-component":
        prepare = actor.world.prepare_spatial
        monkeypatch.setattr(actor.world, "prepare_spatial", lambda source, preps, memberships, **kw:
            prepare(source, preps, memberships[:-1], **kw))
        with pytest.raises(WorldPublicationRefused, match="spatial_final_membership_cohort"):
            actor.prepare_rows(rows, frame=3, households=houses, attribution_mode="physical_only_quarantine")
    elif defect == "unknown-mode":
        with pytest.raises(WorldPublicationRefused, match="unsupported_attribution_mode"):
            actor.prepare_rows(rows, frame=3, households=houses, attribution_mode="unknown")
        for receipt, _ in tuple(actor.world._prepared_candidates.values()): actor.world.discard(receipt)
    elif defect == "changed-clock":
        candidate = actor.prepare_rows(rows, frame=3, households=houses, attribution_mode="physical_only_quarantine")
        actor.context = replace(actor.context, intervals=tuple(replace(i,
            evaluation_max_s=i.evaluation_max_s + 1.) if i.binding.source_id == 2 else i for i in actor.context.intervals))
        with pytest.raises(WorldPublicationRefused): actor.commit_world(candidate)
    else:
        held = harness.publish(actor.prepare_rows(rows, frame=3, households=houses,
            attribution_mode="physical_only_quarantine", source_changes={1: {"world_covariance": None},
                                                                       2: {"world_covariance": None}}), 3)
        assert [(e.position, e.observed_at_us, e.stale_after_us) for e in held.entities] == [
            (e.position, e.observed_at_us, e.stale_after_us) for e in first.entities]
        assert all(e.attribution_quarantine for e in held.entities)
        originals = {s.observation_id for e in first.entities for s in e.sources if s.accepted}
        recorded = {c.observation_id for e in held.entities for c in e.attribution_quarantine.retained_owners}
        assert originals <= recorded
        assert actor.identity.stats().spatial_member_frontiers == before[1]
        return
    assert actor.world.current_snapshot() is first
    assert before == (actor.world._revision, actor.identity.stats().spatial_member_frontiers,
                      len(harness.wire), len(harness.journal))
    assert actor.identity.stats().pending_reservations == 0


def test_attribution_r4_real_publisher_aborts_whole_quarantine_on_queued_owner_change():
    actor = SpatialActor()
    harness = ActualPublicationHarness(actor)
    houses = {(1, 1): HouseholdDescriptor("resident", "synthetic-same", 9),
              (2, 2): HouseholdDescriptor("resident", "synthetic-same", 10)}
    candidate = actor.prepare_rows({1: [(1, 1., 1, 1)], 2: [(2, 3., 1, 1)]},
        households=houses, attribution_mode="physical_only_quarantine")
    harness.before_commit = lambda: actor.identity.reset_source(replace(actor.sources[1], source_epoch=1))
    with pytest.raises(WorldPublicationRefused): harness.publish(candidate, 2)
    assert not harness.wire and not harness.journal and harness.failures
    assert actor.world.current_snapshot() is None and actor.world._revision == 0
    assert actor.identity.stats().pending_reservations == 0


def test_spatial_transaction_actual_wire_dynamic_quality_and_exact_age():
    actor, rows = SpatialActor(), {1: [(1, 1., 1, 1)], 2: [(101, 8., 1, 1)]}
    harness = ActualPublicationHarness(actor)
    worse_first = harness.publish(actor.prepare_rows({1: rows[1]},
        source_changes={1: {"world_covariance": tuple(v * 4 for v in COV)}}), 2)
    assert worse_first.entities[0].position.x == 1.
    actor.us += 100_000
    candidate = actor.prepare_rows(rows, frame=3, join=True, source_changes={1: {"world_covariance": tuple(v * 4 for v in COV)}})
    first = harness.publish(candidate, 3)
    head = first.entities[0]
    assert head.position.x == 8. and head.covariance.values == COV and sum(s.accepted for s in head.sources) == 1
    assert next(s for s in head.sources if s.accepted).camera_id == "family-room"
    assert len(head.sources) == 2 and all(s.observed_at_us == 10_300_000 for s in head.sources)
    assert head.entity_id == worse_first.entities[0].entity_id
    actor.us += 100_000
    rows[1] = [(1, 1.1, 1, 1)]
    rows[2] = [(101, 8.1, 1, 1)]
    candidate = actor.prepare_rows(rows, frame=4, source_changes={2: {"world_covariance": tuple(v * 4 for v in COV)}})
    second = harness.publish(candidate, 4)
    assert second.entities[0].entity_id == head.entity_id
    assert second.entities[0].position.x == 1.1 and second.entities[0].velocity_mps is None
    assert second.entities[0].continuity_segment_id > head.continuity_segment_id
    actor.us += 100_000
    candidate = actor.prepare_rows(rows, frame=5, source_changes={1: {"world_source": "anchor_hold", "world_measurement_accepted": False, "world_covariance": None}})
    third = harness.publish(candidate, 5)
    assert third.entities[0].position.x == 8.1 and third.entities[0].entity_id == head.entity_id
    assert next(s for s in third.entities[0].sources if s.camera_id == "kitchen").accepted is False
    actor.us += 800_000
    fourth = harness.publish(actor.prepare_rows({1: []}, frame=6), 6)
    assert fourth.entities[0].position_state == "held" and fourth.entities[0].observed_at_us == 10_500_000
    actor.us += 1_300_000
    fifth = harness.publish(actor.prepare_rows({1: []}, frame=7), 7)
    assert not fifth.entities
    assert all(message["world_snapshot"] == following["payload"] for message, following in
               zip(harness.wire, harness.wire[1:]) if message["type"] == "tracking")
    actual = actual_node_wire(actor, harness.wire, "t9-spatial-quality-reversal-hold-expiry")
    assert [len(s["people"]) for s in actual["snapshots"]] == [1, 1, 1, 1, 1, 0]
    assert [s["people"][0]["worldPosition"][0] for s in actual["snapshots"][:4]] == [1., 8., 1.1, 8.1]
    assert actual["snapshots"][1]["people"][0]["trailBreak"]


def test_spatial_transaction_actual_two_person_retained_promotion_and_removal():
    actor = SpatialActor()
    harness = ActualPublicationHarness(actor)
    rows = {1: [(1, 0., 1, 1), (2, 3., 1, 1)], 2: [(101, .03, 1, 1), (102, 3.03, 1, 1)]}
    first = harness.publish(actor.prepare_rows(rows, join=True), 2)
    assert len(first.entities) == 2
    a, b = sorted(first.entities, key=lambda e: e.position.x)
    actor.us += 100_000
    house = HouseholdDescriptor("visitor", "synthetic-visitor-a", 91, 4)
    next_rows = {1: [(1, .1, 1, 1)], 2: [(101, .13, 1, 1)]}
    promoted = harness.publish(actor.prepare_rows(next_rows, frame=3,
        households={(1, 1): house, (2, 101): house}), 3)
    current_a = next(e for e in promoted.entities if e.entity_id == a.entity_id)
    retained_b = next(e for e in promoted.entities if e.entity_id == b.entity_id)
    assert current_a.subject.stable_id == 91 and current_a.subject.generation == 4
    assert current_a.continuity_segment_id == a.continuity_segment_id
    assert retained_b.observed_at_us == b.observed_at_us and retained_b.sources == b.sources
    reset = actor.identity.reset_source(replace(actor.sources[0], source_epoch=1))
    actor.sources = (replace(actor.sources[0], source_epoch=1), actor.sources[1])
    actor.us += 100_000
    barrier = harness.publish(actor.prepare_rows({1: []}, frame=4,
        withdrawn_members=reset.withdrawn_members, retired_keys=reset.retired_keys), 4)
    assert all(s.camera_id != "kitchen" for e in barrier.entities for s in e.sources)
    assert {e.entity_id for e in barrier.entities} == {a.entity_id, b.entity_id}
    reset_family = actor.identity.reset_source(replace(actor.sources[1], source_epoch=1))
    actor.sources = (actor.sources[0], replace(actor.sources[1], source_epoch=1))
    actor.us += 100_000
    removed = harness.publish(actor.prepare_rows({1: []}, frame=5,
        withdrawn_members=reset_family.withdrawn_members, retired_keys=reset_family.retired_keys), 5)
    assert not removed.entities
    actor.us += 100_000
    reused = harness.publish(actor.prepare_rows({1: [(1, .2, 2, 6)]}, frame=6), 6)
    assert reused.entities[0].entity_id not in {a.entity_id, b.entity_id}
    assert reused.entities[0].subject.stable_id is None
    actual = actual_node_wire(actor, harness.wire, "t9-spatial-two-person-promotion-retained-barrier")
    assert [len(s["people"]) for s in actual["snapshots"]] == [2, 2, 2, 0, 1]
    promoted_person = next(p for p in actual["snapshots"][1]["people"] if p["entityId"] == a.entity_id)
    assert promoted_person["generation"] == 4 and len(promoted_person["trail"]) == 2


@pytest.mark.parametrize("defect", ["confidence", "covariance", "frame", "clock", "queued-reset", "stale-media", "source-clock", "rebind-at-commit"])
def test_spatial_transaction_atomic_quality_binding_and_queued_refusals(defect):
    actor = SpatialActor()
    first = actor.commit_world(actor.prepare_rows({1: [(1, 1., 1, 1)]}))
    actor.us += 100_000
    before, stats = actor.world.current_snapshot(), actor.identity.stats()
    if defect in {"confidence", "covariance"}:
        field = "world_resolver_confidence" if defect == "confidence" else "world_covariance"
        candidate = actor.prepare_rows({1: [(1, 1.1, 1, 1)]}, frame=3, source_changes={1: {field: None}})
        publication = actor.commit_world(candidate)
        assert publication.snapshot.entities == first.snapshot.entities
        assert actor.identity.stats().spatial_member_frontiers == stats.spatial_member_frontiers
        assert actor.world._revision == 2
        return
    if defect in {"confidence", "covariance", "frame", "stale-media"}:
        changes = {"confidence": {"world_resolver_confidence": None},
                   "covariance": {"world_covariance": None},
                   "frame": {"world_transform_sha256": "a" * 64},
                   "stale-media": {"media_pts_ns": 200_000_000}}[defect]
        with pytest.raises(WorldPublicationRefused):
            actor.prepare_rows({1: [(1, 1.1, 1, 1)]}, frame=3, source_changes={1: changes})
    else:
        candidate = actor.prepare_rows({1: [(1, 1.1, 1, 1)]}, frame=3)
        if defect == "clock":
            actor.context = replace(actor.context, intervals=tuple(replace(i, evaluation_min_s=i.evaluation_min_s + 1., evaluation_max_s=i.evaluation_max_s + 1.) for i in actor.context.intervals))
        elif defect == "queued-reset":
            actor.identity.reset_source(replace(actor.sources[0], source_epoch=1))
            stats = actor.identity.stats()  # The REAL external barrier clears its own frontier.
        elif defect == "source-clock":
            actor.context = replace(actor.context, intervals=tuple(replace(i, binding=replace(i.binding, clock_domain="unsupported")) for i in actor.context.intervals))
        else:
            actor.frames[1] = actor.frames[1].model_copy(update={"target_revision_metadata_sha256": "a" * 64})
        with pytest.raises(WorldPublicationRefused):
            actor.commit_world(candidate)
    assert actor.world.current_snapshot() is before is first.snapshot
    assert actor.world._revision == 1 and not actor.world._prepared_candidates
    assert actor.identity.stats().spatial_member_frontiers == stats.spatial_member_frontiers


@pytest.mark.parametrize("kind", ["owner-conflict", "retained-world"])
def test_spatial_transaction_actual_retained_conflict_atomic_and_exact_barrier_recovery(kind):
    actor = SpatialActor()
    harness = ActualPublicationHarness(actor)
    a_house, b_house = HouseholdDescriptor("resident", "synthetic-A", 11), HouseholdDescriptor("resident", "synthetic-B", 12)
    rows = {1: [(1, 0., 1, 1), (2, 3., 1, 1)], 2: [(101, .03, 1, 1), (102, 3.03, 1, 1)]}
    houses = {(source, raw): a_house if raw in {1, 101} else b_house for source, people in rows.items() for raw, *_ in people}
    prior = harness.publish(actor.prepare_rows(rows, join=True, households=houses), 2)
    old_b = next(e for e in prior.entities if e.subject.stable_id == 12)
    actor.us += 100_000
    withdrawals = []
    if kind == "retained-world":
        for old in tuple(actor.originals.values()):
            if old.stamp.member.raw_id in {2, 102}:
                result = actor.identity.withdraw(MemberRemoval("synthetic-terminate:" + old.stamp.observation_id,
                    old.stamp, "termination", 3, actor.us / 1_000_000, old.stamp.clock_domain, .3))
                assert result.disposition == "removed"
                withdrawals.extend(result.withdrawn_members)
        # REAL issued barrier deliberately withheld by this negative caller.
        actor.identity.spatial_memberships(at_interval=actor.clock_context())
    before_serial = actor.identity.stats().reservation_serial
    with pytest.raises(WorldPublicationRefused) as caught:
        actor.prepare_rows({1: [(1, .1, 1, 1)], 2: [(101, .13, 1, 1)]}, frame=3,
            households={(1, 1): b_house, (2, 101): b_house})
    if kind == "owner-conflict":
        conflict = next(dto.household_conflict for dto in actor.memberships if dto.household_conflict is not None)
        assert caught.value.household_conflict is conflict and caught.value.reason == conflict.reason
    else:
        assert caught.value.reason == "canonical_identity_duplicate" and caught.value.identity_conflicts
    assert actor.world.current_snapshot() is prior and old_b.observed_at_us == 10_200_000
    assert len(harness.journal) == 1 and harness.admissions == 1
    assert actor.identity.stats().reservation_serial == before_serial
    assert not actor.world._prepared_candidates and actor.identity.stats().pending_reservations == 0
    if kind == "retained-world":
        actor.us += 100_000
        recovered = harness.publish(actor.prepare_rows({1: [(1, .2, 1, 1)], 2: [(101, .23, 1, 1)]}, frame=4,
            households={(1, 1): b_house, (2, 101): b_house}, withdrawn_members=tuple(withdrawals)), 4)
        assert len(recovered.entities) == 1 and recovered.entities[0].subject.stable_id == 12
        actual = actual_node_wire(actor, harness.wire, "t9-spatial-retained-conflict-recovery")
        assert [len(s["people"]) for s in actual["snapshots"]] == [2, 1]


def test_spatial_transaction_full_existing_reservation_bound_and_discard():
    actor = SpatialActor()
    candidate = actor.prepare_rows({1: [(raw, 1., 1, 1) for raw in range(1, 65)]})
    assert actor.identity.stats().pending_reservations == 64
    publication = actor.commit_world(candidate)
    assert len(publication.snapshot.entities) == 64 and actor.identity.stats().pending_reservations == 0
    actor.us += 100_000
    candidate = actor.prepare_rows({1: [(1, 1.1, 1, 1)]}, frame=3)
    frontiers = actor.identity.stats().spatial_member_frontiers
    actor.world.discard(candidate)
    assert actor.identity.stats().pending_reservations == 0
    assert actor.identity.stats().spatial_member_frontiers == frontiers
    assert actor.world.current_snapshot() is publication.snapshot


def test_spatial_transaction_retained_person_requires_original_source_clock(monkeypatch):
    actor = SpatialActor()
    first = actor.commit_world(actor.prepare_rows({1: [(1, 0., 1, 1), (2, 3., 1, 1)],
        2: [(101, .03, 1, 1), (102, 3.03, 1, 1)]}, join=True))
    actor.us += 100_000
    context = actor.clock_context
    monkeypatch.setattr(actor, "clock_context", lambda **kw: replace(context(**kw),
        intervals=tuple(i for i in context(**kw).intervals if i.binding.source_id == 1)))
    with pytest.raises(WorldPublicationRefused, match="retained_spatial_clock_or_binding_unavailable"):
        actor.prepare_rows({1: [(1, .1, 1, 1)]}, frame=3)
    assert actor.world.current_snapshot() is first.snapshot
    assert actor.identity.stats().pending_reservations == 0


def test_spatial_unavailable_r3_actual_publication_retention_recovery_expiry():
    actor = SpatialActor()
    harness = ActualPublicationHarness(actor)
    no_point = {"world": None, "world_valid": False, "world_measurement_accepted": False,
                "world_covariance": None, "world_resolver_confidence": None}
    empty = harness.publish(actor.prepare_rows({1: [(1, 1., 1, 1)]}, source_changes={1: no_point}), 2)
    assert not empty.entities and actor.identity.stats().reservation_serial == 0
    actor.us += 100_000
    first = harness.publish(actor.prepare_rows({1: [(1, 1., 1, 1)]}, frame=3), 3)
    original = first.entities[0]
    serial = actor.identity.stats().reservation_serial
    for frame, changes in enumerate((no_point, {"world_support_state": "unknown"},
                                     {"world_covariance": None}, {"world_resolver_confidence": None}), 4):
        actor.us += 100_000
        publication = harness.publish(actor.prepare_rows({1: [(1, 2., 1, 1)]}, frame=frame,
            source_changes={1: changes}), frame)
        retained = publication.entities[0]
        assert retained.position == original.position and retained.covariance == original.covariance
        assert retained.observed_at_us == original.observed_at_us and retained.stale_after_us == original.stale_after_us
        assert actor.identity.stats().reservation_serial == serial
        assert actor.identity.stats().pending_reservations == 0
    actor.us += 100_000
    recovered = harness.publish(actor.prepare_rows({1: [(1, 1.1, 1, 1)]}, frame=8), 8)
    assert recovered.entities[0].position.x == 1.1 and recovered.entities[0].entity_id == original.entity_id
    actor.us += 800_000
    held = harness.publish(actor.prepare_rows({1: [(1, 1.1, 1, 1)]}, frame=9, source_changes={1: no_point}), 9)
    assert held.entities[0].position_state == "held" and held.entities[0].observed_at_us == recovered.entities[0].observed_at_us
    actor.us += 1_300_000
    expired = harness.publish(actor.prepare_rows({1: [(1, 1.1, 1, 1)]}, frame=10, source_changes={1: no_point}), 10)
    assert not expired.entities
    actual = actual_node_wire(actor, harness.wire, "r3-unavailable-retention-recovery-expiry")
    assert [len(s["people"]) for s in actual["snapshots"]] == [0, 1, 1, 1, 1, 1, 1, 1, 0]
    assert all(s["people"][0]["observedAtUs"] == original.observed_at_us for s in actual["snapshots"][1:6])
    assert all(len(s["people"][0]["trail"]) == 1 for s in actual["snapshots"][1:6])


def test_spatial_unavailable_r3_good_peer_and_other_person_progress_without_B_renewal():
    actor = SpatialActor()
    harness = ActualPublicationHarness(actor)
    rows = {1: [(1, 0., 1, 1), (2, 3., 1, 1)], 2: [(101, .03, 1, 1), (102, 3.03, 1, 1)]}
    initial = harness.publish(actor.prepare_rows(rows, join=True), 2)
    old_b = max(initial.entities, key=lambda e: e.position.x)
    actor.us += 100_000
    current = harness.publish(actor.prepare_rows({1: [(1, .1, 1, 1)], 2: [(101, .13, 1, 1)]}, frame=3,
        source_changes={1: {"world_covariance": None}}), 3)
    new_a = min(current.entities, key=lambda e: e.position.x)
    retained_b = next(e for e in current.entities if e.entity_id == old_b.entity_id)
    assert new_a.position.x == .13 and retained_b == old_b
    assert sum(s.accepted for s in new_a.sources) == 1
    assert next(s for s in new_a.sources if s.camera_id == "kitchen").rejection_reason == "resolver_covariance_absent"
    reset = actor.identity.reset_source(replace(actor.sources[1], source_epoch=1))
    actor.sources = (actor.sources[0], replace(actor.sources[1], source_epoch=1))
    actor.us += 100_000
    barrier = harness.publish(actor.prepare_rows({1: []}, frame=4, withdrawn_members=reset.withdrawn_members,
        retired_keys=reset.retired_keys), 4)
    assert not barrier.entities  # Ungranted Kitchen diagnostic cannot become a point.
    actual = actual_node_wire(actor, harness.wire, "r3-unavailable-peer-B-reset")
    assert [len(s["people"]) for s in actual["snapshots"]] == [2, 2, 0]


@pytest.mark.parametrize("changes,reason", [
    ({"world_resolver_confidence": float("nan")}, "resolver_confidence_invalid"),
    ({"world_resolver_confidence": 1.1}, "resolver_confidence_invalid"),
    ({"world_resolver_confidence": "0.8"}, "resolver_confidence_invalid"),
    ({"world_covariance": [1., 2.]}, "resolver_covariance_invalid"),
    ({"world_covariance": [float("nan")] * 9}, "resolver_covariance_invalid"),
    ({"world_covariance": (0., 0., 0., 0., .1, 0., 0., 0., .1)}, "resolver_covariance_invalid"),
    ({"world_covariance": (.1, 1., 0., 1., .1, 0., 0., 0., .1)}, "resolver_covariance_invalid"),
    ({"world": [float("nan"), 0., 2.]}, "world_position_invalid"),
    ({"world": [1.]}, "world_position_invalid"),
    ({"world_transform_sha256": None}, "world_registration_unavailable"),
    ({"world": None}, "accepted_world_position_unavailable"),
    pytest.param({"world_resolver_confidence": 2**1024}, "resolver_confidence_invalid", id="confidence-overflow"),
    pytest.param({"world_support_state": {"bad": "state"}}, "world_support_invalid", id="support-malformed"),
])
def test_spatial_unavailable_r3_integrity_fatal_even_with_valid_peer(changes, reason):
    actor = SpatialActor()
    harness = ActualPublicationHarness(actor)
    rows = {1: [(1, 0., 1, 1)], 2: [(101, .03, 1, 1)]}
    prior = harness.publish(actor.prepare_rows(rows, join=True), 2)
    actor.us += 100_000
    before = (actor.world._fusion, actor.world._revision, actor.world._event_sequence,
              actor.identity.stats().reservation_serial, len(harness.journal), len(harness.wire))
    with pytest.raises(WorldPublicationRefused) as error:
        actor.prepare_rows(rows, frame=3, source_changes={1: changes})
    assert error.value.reason == reason
    assert actor.world.current_snapshot() is prior
    assert before == (actor.world._fusion, actor.world._revision, actor.world._event_sequence,
              actor.identity.stats().reservation_serial, len(harness.journal), len(harness.wire))
    assert actor.identity.stats().pending_reservations == 0 and not actor.world._prepared_candidates


@pytest.mark.parametrize("defect,reason", [
    ("media", "spatial_original_media_binding"),
    ("birth", "spatial_original_member_context"),
    ("stale-owner", "spatial_membership_revision_changed"),
    ("unknown", "unexpected_spatial_rejection_reason"),
])
def test_spatial_unavailable_r3_original_authority_refusal_is_atomic(defect, reason, monkeypatch):
    actor = SpatialActor()
    harness = ActualPublicationHarness(actor)
    rows = {1: [(1, 0., 1, 1)], 2: [(101, .03, 1, 1)]}
    prior = harness.publish(actor.prepare_rows(rows, join=True), 2)
    previous_memberships = actor.memberships  # Actual old owner DTOs, not fabricated capabilities.
    actor.us += 100_000
    changes = {1: {"media_pts_ns": 200_000_000}} if defect == "media" else None
    if defect == "birth":
        # Contradictory raw birth versus the ORIGINAL native stamp. The owner
        # receives that real stamp, never a counterfeit member/grant.
        native = actor.native_source_record
        def original_birth(source, row, frame, household):
            original = native(source, row, frame, household)
            return replace(original, stamp=replace(original.stamp, lifecycle_started_frame=1))
        monkeypatch.setattr(actor, "native_source_record", original_birth)
        changes = {1: {"lifecycle_started_frame": 0}}
    elif defect == "stale-owner":
        prepare = actor.world.prepare_spatial
        monkeypatch.setattr(actor.world, "prepare_spatial", lambda source, preps, memberships, **kwargs:
            prepare(source, preps, previous_memberships, **kwargs))
    elif defect == "unknown":
        quality = actor.world._spatial_input_quality
        def unknown_quality(track, original):
            confidence, rejection, summary = quality(track, original)
            return confidence, "unrecognized-test-reason" if summary["source"] == 1 else rejection, summary
        monkeypatch.setattr(actor.world, "_spatial_input_quality", unknown_quality)
    before = (actor.world._fusion, actor.world._revision, actor.world._event_sequence,
              actor.identity.stats().reservation_serial, len(harness.journal), len(harness.wire))
    with pytest.raises(WorldPublicationRefused) as error:
        actor.prepare_rows(rows, frame=3, source_changes=changes)
    assert error.value.reason == reason and actor.world.current_snapshot() is prior
    assert before == (actor.world._fusion, actor.world._revision, actor.world._event_sequence,
              actor.identity.stats().reservation_serial, len(harness.journal), len(harness.wire))
    assert actor.identity.stats().pending_reservations == 0 and not actor.world._prepared_candidates


def test_spatial_unavailable_r3_diagnostics_bounded_exact_and_fatal_reserved(caplog):
    actor = SpatialActor()
    for frame in range(2, 14):
        actor.us += 100_000
        actor.commit_world(actor.prepare_rows({1: [(1, 1., 1, 1)]}, frame=frame,
            source_changes={1: {"world_covariance": None}}))
    records = [r for r in caplog.records if r.name == "noesis_core.world_service" and r.message.startswith("Spatial candidate classification ")]
    assert len(records) == 7
    assert actor.world._spatial_reason_counts["resolver_covariance_absent"] == 12
    actor.us += 100_000
    with pytest.raises(WorldPublicationRefused, match="resolver_covariance_invalid"):
        actor.prepare_rows({1: [(1, 1., 1, 1)]}, frame=14,
            source_changes={1: {"world_covariance": (0., 0., 0., 0., .1, 0., 0., 0., .1)}})
    records = [r for r in caplog.records if r.name == "noesis_core.world_service" and r.message.startswith("Spatial candidate classification ")]
    assert len(records) == 8 and records[-1].message.find("resolver_covariance_invalid") >= 0
    for record in records:
        assert len(record.message.encode()) <= 2048
        parsed = json.loads(record.message.removeprefix("Spatial candidate classification "))
        assert len(parsed["candidates"]) <= 2
        assert all(c["category"] in {"vote", "no_vote", "integrity"} for c in parsed["candidates"])
        assert not any(secret in record.message for secret in ("tracker_id", "stable_id", "resident", "gallery", "synthetic-native", "uuid", "bbox"))
    last = json.loads(records[-1].message.removeprefix("Spatial candidate classification "))["candidates"][0]
    assert last["covariance"]["check"] == "invalid" and last["covariance"]["values"][0] == 0.
    assert last["source"] == 1 and last["frame"] == 14 and last["confidence"]["check"] == "valid"


class ImageQuarantineActor(SpatialActor):
    """Actual image service/World; geometry, clocks and source leases SYNTHETIC."""

    def __init__(self, *, live_track_continuity=False, contradictory_cue_quarantine=False,
                 semantic_joint_max_mean_px=None, unresolved_birth_quarantine=False):
        super().__init__()
        from reid.mv3dt_person_association import ProjectiveContinuityPolicy, ProjectiveCameraBinding
        p=((100.,0.,0.,0.),(0.,100.,0.,0.),(0.,0.,1.,0.))
        self.image_policy=ProjectiveContinuityPolicy(tuple(ProjectiveCameraBinding(b,
            'synthetic-image-prior','synthetic-raster:'+b.camera_id,(1920,1080),'backend_world_m',
            self.frames[b.source_id].target_frame.revision,
            self.frames[b.source_id].source_camera_calibration_sha256,p,
            ((100.,0.,0.,-10. if b.source_id==1 else 10.),p[1],p[2])) for b in self.sources))
        self.identity=IdentityAssociationService.for_estimated_image_regions(self.world.producer.run_id,self.spatial_policy,self.image_policy,
            live_track_continuity=live_track_continuity,
            contradictory_cue_quarantine=contradictory_cue_quarantine,
            semantic_joint_max_mean_px=semantic_joint_max_mean_px,
            unresolved_birth_quarantine=unresolved_birth_quarantine)
        self.world=CanonicalWorldService(producer=self.world.producer,artifacts=self.artifacts,
            clock_us=lambda:self.us,association_service=self.identity,spatial_bindings=lambda:self.frames)
        self.image_originals={}
        self.image_sequence=0

    def prepare_image(self,rows,*,frame=2,households=None,mode='physical_only_quarantine',source_changes=None,
                      withdrawn_members=(),retired_keys=(),image_y=None,omit_image_current=False,
                      semantic_joints=None):
        from reid.mv3dt_person_association import (ImageRegionOriginal,ObservedImageRegion,EstimatedImageRegionProof,
            CurrentVisualDeliveryContext,CurrentVisualIngress)
        preps=[];mono=100.+(self.us-10_200_000)/1e6
        for source,people in rows.items():
            part,native=self.normalize({source:people},frame=frame,native_tokens=True,households=households,
                **({'world_resolver_confidence':.8}|(source_changes or {}).get(source,{})))
            preps.extend(part)
            for original in native.values():
                self.identity.observe_spatial(original,SpatialLifecycleReceipt('synthetic-native:'+original.stamp.observation_id,original.stamp,'ACTIVE'))
                self.originals[original.stamp.member]=original
                camera=next(c for c in self.image_policy.cameras if c.source.source_id==source)
                u=50. if source==1 else 40.
                y=30. if original.stamp.member.raw_id in {1,101} else 70.
                y=(image_y or {}).get((source,original.stamp.member.raw_id),y)
                patches=(ObservedImageRegion('pose_shoulder_support',(u,y),(u,y,u,y,2.,2.)),
                    ObservedImageRegion('pose_torso_support',(u,y+5.),(u,y,u,y+10.,2.,2.)))
                self.image_originals[original.stamp.member]=ImageRegionOriginal(original.stamp,mono,
                    'synthetic-body:'+original.stamp.observation_id,camera.raster_id,camera.prior_id,
                    camera.image_size,(u-5.,y-5.,10.,25.),patches,
                    semantic_joints=(semantic_joints or {}).get((source,original.stamp.member.raw_id),()))
        self.context=self.clock_context();self.image_sequence+=1
        originals=tuple(self.image_originals.values())
        proof=EstimatedImageRegionProof('synthetic-image:'+str(self.image_sequence),tuple(o.stamp for o in originals),
            tuple(o for o in originals if not omit_image_current or o.stamp.member.source_id not in rows))
        delivery=CurrentVisualDeliveryContext('synthetic-image-world',self.image_sequence,mono,
            tuple(CurrentVisualIngress(o.stamp,o.ingress_monotonic_s) for o in originals))
        joined=self.identity.join_estimated_image_regions(proof,at_interval=self.context,delivery_context=delivery)
        assert joined.disposition in {'accepted','observed'},joined
        self.memberships=self.identity.spatial_memberships(at_interval=self.context)
        return self.world.prepare_spatial(next(iter(rows)),tuple(preps),self.memberships,at_interval=self.context,
            withdrawn_members=withdrawn_members+joined.withdrawn_members,retired_keys=retired_keys+joined.retired_keys,
            attribution_mode=mode)


def _r101_actor(monkeypatch):
    import time as real_time
    from types import SimpleNamespace
    import noesis_core.world_service as world_module
    from reid.mv3dt_person_association import ImageRegionOwnerAvailability
    actor=ImageQuarantineActor(live_track_continuity=True,
        semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    clock=SimpleNamespace(**vars(real_time))
    clock.monotonic=lambda:100.+(actor.us-10_200_000)/1e6
    monkeypatch.setattr(world_module,'time',clock)
    original_join=actor.identity.join_estimated_image_regions
    def join(proof,**kwargs):
        caps=[]
        for o in proof.originals:
            members=tuple(p.stamp.member for p in proof.originals if p.stamp.member.source_id==o.stamp.member.source_id)
            caps.append(ImageRegionOwnerAvailability(o.stamp,'r101-graph',o.stamp.frame_id,
                o.stamp.media_time_s,1_000_000+o.stamp.frame_id,o.ingress_monotonic_s,
                350_000_000,members,members,(),(),o.ingress_monotonic_s,0.,20.,'running'))
        return original_join(replace(proof,owner_availability=tuple(caps)),**kwargs)
    actor.identity.join_estimated_image_regions=join
    actor.rows={1:[(1,1.,1,1)],2:[(101,1.03,1,1)]}
    def joints(u,y):return tuple((i,u+dx,y+dy,.9) for i,dx,dy in ((5,-2,0),(6,2,0),(11,-1,10),(12,1,10)))
    actor.joints={(1,1):joints(50.,30.),(2,101):joints(40.,30.)}
    return actor,ActualPublicationHarness(actor),clock


def _r101_split(actor,harness):
    first=harness.publish(actor.prepare_image(actor.rows,frame=2,semantic_joints=actor.joints),2)
    assert len(first.entities)==1
    actor.us+=100_000
    changed=dict(actor.joints)
    changed[(2,101)]=tuple((i,u,v+60.,c) for i,u,v,c in changed[(2,101)])
    candidate=actor.prepare_image(actor.rows,frame=3,semantic_joints=changed,image_y={(2,101):90.},
        source_changes={1:{'world_covariance':None},2:{'world_covariance':None}})
    assert len(actor.memberships)==2 and all(d.estimated_split_custody is not None for d in actor.memberships)
    return first,candidate


def test_r116_same_pair_rejoin_without_new_position_keeps_original_finite_hold(monkeypatch):
    actor,harness,clock=_r101_actor(monkeypatch)
    first,candidate=_r101_split(actor,harness)
    held=harness.publish(candidate,3)
    old=held.entities[0]
    episode=actor.memberships[0].estimated_split_custody
    actor.us+=100_000
    candidate=actor.prepare_image(actor.rows,frame=4,semantic_joints=actor.joints,
        source_changes={1:{'world_covariance':None},2:{'world_covariance':None}})
    assert len(actor.memberships)==1
    dto,=actor.memberships
    assert dto.physical_key==episode.origin_key and all(e.eligible for e in dto.eligibility)
    final=harness.publish(candidate,4)
    assert len(final.entities)==1
    current,=final.entities
    assert current.position_state=='held' and current.lifecycle=='held'
    # The split hold followed its native owner; certified SAME-pair reunion
    # restores the original key, not a second point or renewed measurement.
    assert current.entity_id==first.entities[0].entity_id
    assert (current.position,current.covariance,current.observed_at_us,
        current.stale_after_us,current.continuity_segment_id)==(
        old.position,old.covariance,old.observed_at_us,
        old.stale_after_us,old.continuity_segment_id)
    assert current.observed_at_us==first.entities[0].observed_at_us
    assert episode.owner_deadline_monotonic_s>clock.monotonic()
    assert current.stale_after_us>final.published_at_us
    assert actor.identity.stats().pending_reservations==0
    actual=actual_node_wire(actor,harness.wire,'r116-rejoin-held')
    assert [len(s['people']) for s in actual['snapshots']]==[1,1,1]
    actor.us=old.stale_after_us+1
    expired=harness.publish(actor.prepare_image(actor.rows,frame=5,semantic_joints=actor.joints,
        source_changes={1:{'world_covariance':None},2:{'world_covariance':None}}),5)
    assert not expired.entities


@pytest.mark.parametrize('boundary',['wire_expiry','owner_expiry','stop','reset','discard','superseded'])
def test_r116_rejoined_hold_never_crosses_commit_or_original_boundary(monkeypatch,boundary):
    actor,harness,clock=_r101_actor(monkeypatch)
    _,candidate=_r101_split(actor,harness)
    held=harness.publish(candidate,3);old=held.entities[0]
    actor.us+=100_000
    candidate=actor.prepare_image(actor.rows,frame=4,semantic_joints=actor.joints,
        source_changes={1:{'world_covariance':None},2:{'world_covariance':None}})
    if boundary=='discard':
        actor.world.discard(candidate)
        assert actor.world.current_snapshot() is held
        return
    if boundary=='wire_expiry':actor.us=old.stale_after_us+1
    elif boundary=='owner_expiry':clock.monotonic=lambda:121.
    elif boundary=='stop':actor.identity.stop_image_region_source(actor.sources[1],graph_id='r101-graph')
    elif boundary=='reset':actor.identity.reset_source(replace(actor.sources[1],source_epoch=1))
    elif boundary=='superseded':
        # The actual ordered actor forbids later work while this is pending.
        actor.world.discard(candidate)
        newer=actor.prepare_image(actor.rows,frame=5,semantic_joints=actor.joints)
        current=harness.publish(newer,5)
        assert current.entities and current.entities[0].observed_at_us>old.observed_at_us
    if boundary=='superseded':
        with pytest.raises(RuntimeError,match='unknown or consumed'):harness.publish(candidate,4)
    else:
        with pytest.raises(WorldPublicationRefused):harness.publish(candidate,4)
    assert actor.world.current_snapshot() is (current if boundary=='superseded' else held)


def test_r116_current_successor_consumes_hold_and_never_recaptures_replaced_original(monkeypatch):
    actor,harness,_=_r101_actor(monkeypatch)
    _,candidate=_r101_split(actor,harness)
    held=harness.publish(candidate,3);old=held.entities[0]
    actor.us+=100_000
    current=harness.publish(actor.prepare_image(actor.rows,frame=4,semantic_joints=actor.joints),4)
    assert len(current.entities)==1 and current.entities[0].observed_at_us>old.observed_at_us
    assert not actor.world._fusion.split_custody_originals()
    actor.us+=100_000
    final=harness.publish(actor.prepare_image(actor.rows,frame=5,semantic_joints=actor.joints,
        source_changes={1:{'world_covariance':None},2:{'world_covariance':None}}),5)
    assert all(e.observed_at_us!=old.observed_at_us for e in final.entities)
    assert not actor.world._fusion.split_custody_originals()


def test_r116_reserved_but_discarded_successor_does_not_consume_original(monkeypatch):
    actor,harness,_=_r101_actor(monkeypatch)
    _,candidate=_r101_split(actor,harness)
    held=harness.publish(candidate,3);old=held.entities[0]
    actor.us+=100_000
    candidate=actor.prepare_image(actor.rows,frame=4,semantic_joints=actor.joints)
    state=actor.world._prepared_candidates[candidate._candidate_id][1]
    assert state.spatial_reservations
    assert actor.world._fusion.split_custody_originals()
    assert not actor.world._fusion._consumed_split_episodes
    actor.world.discard(candidate)
    actor.us+=100_000
    final=harness.publish(actor.prepare_image(actor.rows,frame=5,semantic_joints=actor.joints,
        source_changes={1:{'world_covariance':None},2:{'world_covariance':None}}),5)
    assert len(final.entities)==1 and final.entities[0].position_state=='held'
    assert final.entities[0].observed_at_us==old.observed_at_us


def test_r116_reunion_hold_preserves_divergent_original_claims_and_distinct_person(monkeypatch):
    actor,harness,_=_r101_actor(monkeypatch)
    actor.rows[2].append((102,3.,1,1))
    actor.joints[(2,102)]=tuple((i,u,v+60.,c) for i,u,v,c in actor.joints[(2,101)])
    houses={(1,1):HouseholdDescriptor('visitor','r116-k',1001,1),
        (2,101):HouseholdDescriptor('visitor','r116-f',1000,1),
        (2,102):HouseholdDescriptor('visitor','r116-other',1002,1)}
    prepare=actor.prepare_image
    actor.prepare_image=lambda *a,**k:prepare(*a,households=houses,**k)
    first=harness.publish(actor.prepare_image(actor.rows,frame=2,semantic_joints=actor.joints),2)
    old=next(e for e in first.entities if e.attribution_quarantine is not None)
    actor.us+=100_000
    changed=dict(actor.joints)
    changed[(2,101)]=tuple((i,u,v+60.,c) for i,u,v,c in changed[(2,101)])
    split=harness.publish(actor.prepare_image(actor.rows,frame=3,semantic_joints=changed,
        image_y={(2,101):90.},source_changes={1:{'world_covariance':None},2:{'world_covariance':None}}),3)
    actor.us+=100_000
    final=harness.publish(actor.prepare_image(actor.rows,frame=4,semantic_joints=actor.joints,
        source_changes={1:{'world_covariance':None},2:{'world_covariance':None}}),4)
    assert len(first.entities)==len(split.entities)==len(final.entities)==2
    held=next(e for e in final.entities if e.entity_id==old.entity_id)
    assert held.position_state=='held' and held.observed_at_us==old.observed_at_us
    assert held.subject.kind=='provisional' and held.attribution_quarantine is not None
    assert set(held.attribution_quarantine.stable_ids)=={1000,1001}
    other=next(e for e in final.entities if e.entity_id!=old.entity_id)
    assert other.subject.stable_id==1002


@pytest.mark.parametrize('omitted',[False,True])
def test_r115_held_loss_captures_actual_split_view_lease_and_members(monkeypatch,omitted):
    actor,harness,clock=_r101_actor(monkeypatch)
    _,candidate=_r101_split(actor,harness)
    held=harness.publish(candidate,3)
    prior,=actor.world._fusion.split_custody_originals()
    view=actor.memberships[0].estimated_split_custody
    actor.us+=100_000
    # Omit via the real owner stop barrier, or retain an incoming owner view
    # whose fixed lease has expired. Neither is inferred from the fusion note.
    if omitted:
        actor.identity.stop_image_region_source(actor.sources[1],graph_id='r101-graph')
        candidate=_r40_empty_spatial(actor,frame=4,evaluation_s=10.4)
    else:
        clock.monotonic=lambda:view.owner_deadline_monotonic_s+.001
        changed=dict(actor.joints)
        changed[(2,101)]=tuple((i,u,v+60.,c) for i,u,v,c in changed[(2,101)])
        candidate=actor.prepare_image(actor.rows,frame=4,semantic_joints=changed,image_y={(2,101):90.},
            source_changes={1:{'world_covariance':None},2:{'world_covariance':None}})
    assert not candidate.publication.snapshot.entities
    row=next(r for r in actor.world._private_spatial_diagnostics()['records'] if r['kind']=='unexpired_held_loss')
    assert row['prior_episode']==prior.episode_id
    assert row['prior_origin_serial']==prior.origin_key.serial
    assert (row['prior_successor_a'],row['prior_successor_b'])==tuple(k.serial for k in prior.successor_keys)
    assert row['incoming_view_present'] is not omitted
    assert row['owner_lease_expired'] is (None if omitted else True)
    assert row['custody_installed'] is False and row['successor_selected'] is False
    assert row['original_member_eligible'] is not None
    assert row['peer_member_eligible'] is not None
    assert row['wire_remaining_us']>0 and row['transaction']=='prepared'
    assert len(row)<=32 and len(actor.world._spatial_observer.encoded().encode())<=8192
    actor.world.discard(candidate)
    assert actor.world.current_snapshot() is held
    assert next(r for r in actor.world._private_spatial_diagnostics()['records'] if r['kind']=='unexpired_held_loss')['transaction']=='discarded'


def test_r115_transient_custody_clear_is_not_reported_as_held_loss(monkeypatch):
    actor,harness,_=_r101_actor(monkeypatch)
    _,candidate=_r101_split(actor,harness)
    held=harness.publish(candidate,3)
    actor.us+=100_000
    changed=dict(actor.joints)
    changed[(2,101)]=tuple((i,u,v+60.,c) for i,u,v,c in changed[(2,101)])
    candidate=actor.prepare_image(actor.rows,frame=4,semantic_joints=changed,image_y={(2,101):90.},
        source_changes={1:{'world_covariance':None},2:{'world_covariance':None}})
    state=actor.world._prepared_candidates[candidate._candidate_id][1]
    assert state.fusion._omission_diagnostics[held.entities[0].entity_id][0]=='split_custody_omitted'
    assert len(state.fusion.split_custody_originals())==1
    final=harness.publish(candidate,4)
    assert final.entities[0].observed_at_us==held.entities[0].observed_at_us
    assert not any(r['kind']=='unexpired_held_loss' for r in actor.world._private_spatial_diagnostics()['records'])


def test_r101_split_original_actual_world_publisher_node_held_expiry(monkeypatch):
    actor,harness,_=_r101_actor(monkeypatch)
    first,candidate=_r101_split(actor,harness)
    final=harness.publish(candidate,3)
    old,=first.entities;held,=final.entities
    assert held.position_state=='held' and held.subject.kind=='provisional'
    assert (held.position,held.covariance,held.observed_at_us,held.stale_after_us,held.continuity_segment_id)==(
        old.position,old.covariance,old.observed_at_us,old.stale_after_us,old.continuity_segment_id)
    assert actor.identity.stats().pending_reservations==0
    actor.us=old.stale_after_us+1
    expired=harness.publish(actor.prepare_image(actor.rows,frame=4,semantic_joints=actor.joints,
        source_changes={1:{'world_covariance':None},2:{'world_covariance':None}}),4)
    assert not expired.entities
    actual=actual_node_wire(actor,harness.wire,'r101-split-custody')
    assert [len(s['people']) for s in actual['snapshots']]==[1,1,0]


@pytest.mark.parametrize('defect',['wall','revision','stop'])
def test_r101_split_queued_owner_barriers_atomic(monkeypatch,defect):
    actor,harness,clock=_r101_actor(monkeypatch)
    first,candidate=_r101_split(actor,harness)
    if defect=='wall':clock.monotonic=lambda:121.
    elif defect=='revision':actor.identity.reset_source(replace(actor.sources[1],source_epoch=1))
    else:actor.identity.stop_image_region_source(actor.sources[1],graph_id='r101-graph')
    with pytest.raises(WorldPublicationRefused):harness.publish(candidate,3)
    assert actor.world.current_snapshot() is first and len(harness.journal)==1


def test_r101_current_successor_consumes_original_without_double_head(monkeypatch):
    actor,harness,_=_r101_actor(monkeypatch)
    first,candidate=_r101_split(actor,harness)
    harness.publish(candidate,3)
    actor.us+=100_000
    final=harness.publish(actor.prepare_image(actor.rows,frame=4,semantic_joints=actor.joints),4)
    assert len(final.entities)==1 and final.entities[0].position_state=='measured'
    assert final.entities[0].observed_at_us>first.entities[0].observed_at_us
    assert actor.world._fusion.split_custody_originals()==()
    actual=actual_node_wire(actor,harness.wire,'r101-successor')
    assert [len(s['people']) for s in actual['snapshots']]==[1,1,1]


def test_r101_split_claim_custody_and_distinct_third_person(monkeypatch):
    actor,harness,_=_r101_actor(monkeypatch)
    actor.rows[2].append((102,5.,2,1))
    actor.joints[(2,102)]=tuple((i,u,v+70.,c) for i,u,v,c in actor.joints[(2,101)])
    houses={(1,1):HouseholdDescriptor('visitor','r101-k',1001,1),
        (2,101):HouseholdDescriptor('visitor','r101-f',1000,1),
        (2,102):HouseholdDescriptor('visitor','r101-third',1002,1)}
    actual_prepare=actor.prepare_image
    actor.prepare_image=lambda *a,**k:actual_prepare(*a,households=houses,**k)
    first=harness.publish(actor.prepare_image(actor.rows,frame=2,semantic_joints=actor.joints),2)
    old=next(e for e in first.entities if e.attribution_quarantine is not None)
    actor.us+=100_000
    changed=dict(actor.joints)
    changed[(2,101)]=tuple((i,u,v+60.,c) for i,u,v,c in changed[(2,101)])
    final=harness.publish(actor.prepare_image(actor.rows,frame=3,semantic_joints=changed,
        image_y={(2,101):90.},source_changes={1:{'world_covariance':None},2:{'world_covariance':None}}),3)
    assert len(final.entities)==2
    custody,=actor.world._fusion.split_custody_originals()
    view=next(d.estimated_split_custody for d in actor.memberships if d.estimated_split_custody is not None)
    destination=actor.identity.estimated_split_original_destination(view,custody.row.spatial_stamp,
        at_interval=actor.context)
    held=next(e for e in final.entities if e.entity_id==f'physical:{view.origin_key.run_id}:{destination.serial}')
    assert (held.position,held.covariance,held.observed_at_us,held.stale_after_us)==(
        old.position,old.covariance,old.observed_at_us,old.stale_after_us)
    assert held.position_state=='held' and held.subject.kind=='provisional'
    assert held.attribution_quarantine is not None
    assert held.observed_at_us==old.observed_at_us
    actual=actual_node_wire(actor,harness.wire,'r101-claims-third')
    assert [len(s['people']) for s in actual['snapshots']]==[2,2]


def _r107_still_split_candidate(actor,frame):
    changed=dict(actor.joints)
    changed[(2,101)]=tuple((i,u,v+60.,c) for i,u,v,c in changed[(2,101)])
    return actor.prepare_image(actor.rows,frame=frame,semantic_joints=changed,
        image_y={(2,101):90.},source_changes={2:{'world_covariance':None}})


def test_r107_still_split_current_successor_is_not_recaptured_as_old_original(monkeypatch):
    actor,harness,_=_r101_actor(monkeypatch)
    first,candidate=_r101_split(actor,harness)
    held=harness.publish(candidate,3)
    view=actor.memberships[0].estimated_split_custody
    actor.us+=100_000
    current=harness.publish(_r107_still_split_candidate(actor,4),4)
    assert len(actor.memberships)==2 and all(d.estimated_split_custody is view for d in actor.memberships)
    assert len(current.entities)==1 and current.entities[0].position_state=='measured'
    assert current.entities[0].observed_at_us>held.entities[0].observed_at_us
    actor.us+=100_000
    after=harness.publish(_r107_still_split_candidate(actor,5),5)
    assert len(after.entities)==1 and after.entities[0].observed_at_us>current.entities[0].observed_at_us
    assert actor.world._fusion.split_custody_originals()==()
    assert len(actor.world._fusion._consumed_split_episodes)==1
    actual=actual_node_wire(actor,harness.wire,'r107-still-split')
    assert [len(s['people']) for s in actual['snapshots']]==[1,1,1,1]
    actor.us+=100_000
    # A certified reunion now carries the SAME finite episode, never a fresh
    # lease. Keep its bounded consumed marker so the old original cannot return.
    harness.publish(actor.prepare_image(actor.rows,frame=6,semantic_joints=actor.joints),6)
    assert len(actor.world._fusion._consumed_split_episodes)==1
    reunion,=actor.world._fusion._consumed_split_episodes.values()
    assert reunion.rejoined_key==view.origin_key
    assert reunion.owner_deadline_monotonic_s==view.owner_deadline_monotonic_s
    assert not actor.world._fusion.split_custody_originals()


@pytest.mark.parametrize('defect',['revision','wall','stop'])
def test_r107_failed_successor_commit_cannot_consume_episode(monkeypatch,defect):
    actor,harness,clock=_r101_actor(monkeypatch)
    _,candidate=_r101_split(actor,harness)
    held=harness.publish(candidate,3)
    actor.us+=100_000
    candidate=_r107_still_split_candidate(actor,4)
    assert actor.world._fusion._consumed_split_episodes=={}
    if defect=='revision':actor.identity.reset_source(replace(actor.sources[1],source_epoch=1))
    elif defect=='wall':clock.monotonic=lambda:121.
    else:actor.identity.stop_image_region_source(actor.sources[1],graph_id='r101-graph')
    with pytest.raises(WorldPublicationRefused):harness.publish(candidate,4)
    assert actor.world.current_snapshot() is held
    assert actor.world._fusion._consumed_split_episodes=={}
    assert len(actor.world._fusion.split_custody_originals())==1


def test_r107_current_successor_expiry_never_resurrects_pre_split_hold(monkeypatch):
    actor,harness,_=_r101_actor(monkeypatch)
    _,candidate=_r101_split(actor,harness)
    harness.publish(candidate,3);actor.us+=100_000
    current=harness.publish(_r107_still_split_candidate(actor,4),4)
    actor.us=current.entities[0].stale_after_us+1
    no_position={1:{'world_covariance':None},2:{'world_covariance':None}}
    final=harness.publish(actor.prepare_image(actor.rows,frame=5,semantic_joints=actor.joints,
        source_changes=no_position),5)
    assert not final.entities and actor.world._fusion.split_custody_originals()==()
    assert actor.world._fusion._consumed_split_episodes=={}


def test_r107_forged_post_split_original_is_still_refused(monkeypatch):
    actor,harness,_=_r101_actor(monkeypatch)
    _,candidate=_r101_split(actor,harness)
    held=harness.publish(candidate,3);actor.us+=100_000
    original=GlobalWorldFusion.capture_split_custody
    def forged(fusion,*args,**kwargs):
        custody=original(fusion,*args,**kwargs)
        if custody is None:return None
        stamp=custody.row.spatial_stamp
        return replace(custody,row=replace(custody.row,spatial_stamp=replace(stamp,
            frame_id=99,acquired_at_s=stamp.acquired_at_s+9.,media_time_s=stamp.media_time_s+9.)))
    monkeypatch.setattr(GlobalWorldFusion,'capture_split_custody',forged)
    with pytest.raises(WorldPublicationRefused,match='split_custody_original_binding'):
        _r107_still_split_candidate(actor,4)
    assert actor.world.current_snapshot() is held and actor.world._fusion._consumed_split_episodes=={}


def _r48_clean_claim_transfer():
    """Actual public same-camera service sequence; synthetic positions/time only."""
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    rows={2:[(3,1.,3,2),(8,4.,5,158)]}
    old_houses={(2,3):HouseholdDescriptor('visitor','synthetic-original-1001',1001,1),
                (2,8):HouseholdDescriptor('provisional')}
    first=harness.publish(actor.prepare_image(rows,frame=186,households=old_houses),186,source_id=2)
    assert len(first.entities)==2 and all(e.attribution_quarantine is None for e in first.entities)
    old=next(e for e in first.entities if e.subject.stable_id==1001)
    current_houses={(2,3):HouseholdDescriptor('provisional'),(2,8):old_houses[(2,3)]}
    original_prepare=actor.world.prepare
    def current_inputs(source,tracks,**kwargs):
        # Genuine new original/household decision, but no fresh World vote for
        # raw3. The raw8 point is independently current and normally granted.
        tracks=[dict(t) for t in tracks]
        for t in tracks:
            if t['tracker_id']==3:
                t.update(world_valid=False,world_measurement_accepted=False,world_covariance=None)
        return original_prepare(source,tracks,**kwargs)
    actor.world.prepare=current_inputs
    actor.us+=100_000
    return actor,harness,rows,current_houses,first,old


def test_r48_clean_committed_claim_transfer_bootstraps_label_only_quarantine():
    actor,harness,rows,houses,first,old=_r48_clean_claim_transfer()
    before_custody=actor.world._fusion.spatial_original_attribution_claims()
    candidate=actor.prepare_image(rows,frame=188,households=houses)
    assert all(d.household_conflict is None for d in actor.memberships)
    final=harness.publish(candidate,188,source_id=2)
    assert len(final.entities)==2 and {e.entity_id for e in final.entities}=={e.entity_id for e in first.entities}
    kept=next(e for e in final.entities if e.entity_id==old.entity_id)
    assert (kept.position,kept.observed_at_us,kept.stale_after_us,kept.continuity_segment_id)==(
        old.position,old.observed_at_us,old.stale_after_us,old.continuity_segment_id)
    q=kept.attribution_quarantine
    assert q is not None and all(e.attribution_quarantine==q for e in final.entities)
    assert q.stable_ids==(1001,) and all(e.subject.kind=='provisional' and e.subject.stable_id is None for e in final.entities)
    assert set(before_custody[old.entity_id])<=set(q.owners+q.retained_owners)
    assert any(c.raw_id==3 and c.frame_id==186 and c.household_stable_id==1001 for c in q.owners)
    assert any(c.raw_id==3 and c.frame_id==188 and c.household_state=='provisional' for c in q.owners)
    assert any(c.raw_id==8 and c.frame_id==188 and c.household_stable_id==1001 for c in q.owners)
    assert actor.identity.stats().pending_reservations==0
    fresh=next(e for e in final.entities if e.entity_id!=old.entity_id)
    assert fresh.observed_at_us>next(e for e in first.entities if e.entity_id==fresh.entity_id).observed_at_us
    # Genuine new provisional measurement replaces the old settled sample;
    # only then may ordinary current owner attribution recover without a leak.
    del actor.world.prepare  # Remove the fixture's no-position producer wrapper.
    actor.us+=100_000
    recovered=harness.publish(actor.prepare_image(rows,frame=189,households=houses),189,source_id=2)
    assert len(recovered.entities)==2 and all(e.attribution_quarantine is None for e in recovered.entities)
    assert sorted(e.subject.stable_id for e in recovered.entities if e.subject.stable_id is not None)==[1001]
    assert next(e for e in recovered.entities if e.entity_id==old.entity_id).observed_at_us>old.observed_at_us
    actual=actual_node_wire(actor,harness.wire,'r48-clean-claim-transfer')
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert [len(s['people']) for s in actual['snapshots']]==[2,2,2]
    assert all(p['stableId'] is None for p in actual['snapshots'][1]['people'])


@pytest.mark.parametrize('defect',['strict-mode','missing-custody','foreign-claim',
    'changed-selected-original','unresolved-owner','grant-denied','queued-revision'])
def test_r48_claim_transfer_requires_current_owner_and_exact_committed_custody(defect,monkeypatch):
    from noesis.identity_association_service import SpatialMemberResult
    actor,harness,rows,houses,first,old=_r48_clean_claim_transfer()
    if defect=='missing-custody':
        actual=GlobalWorldFusion.spatial_original_attribution_claims
        monkeypatch.setattr(GlobalWorldFusion,'spatial_original_attribution_claims',
            lambda fusion:{} if fusion is actor.world._fusion else actual(fusion))
    elif defect=='foreign-claim':
        actual=GlobalWorldFusion.spatial_attribution_claims
        def corrupted(fusion):
            values=actual(fusion)
            if fusion is not actor.world._fusion:
                values[old.entity_id]=tuple(c.model_copy(update={'source_epoch':99}) for c in values[old.entity_id])
            return values
        monkeypatch.setattr(GlobalWorldFusion,'spatial_attribution_claims',corrupted)
    elif defect=='changed-selected-original':
        actual=GlobalWorldFusion.committed_spatial_original
        def changed(fusion,key,**kwargs):
            value=actual(fusion,key,**kwargs)
            if fusion is not actor.world._fusion and GlobalWorldFusion.physical_entity_id(key)==old.entity_id:
                return (value[0].model_copy(),*value[1:])
            return value
        monkeypatch.setattr(GlobalWorldFusion,'committed_spatial_original',changed)
    elif defect=='unresolved-owner':
        houses[(2,3)]=HouseholdDescriptor('provisional',constraint='unresolved')
    elif defect=='grant-denied':
        monkeypatch.setattr(actor.identity,'prepare_spatial_member',lambda *a,**k:
            SpatialMemberResult('denied','synthetic-current-authority-denied'))
    before=(len(harness.wire),len(harness.journal),actor.world._revision)
    if defect=='queued-revision':
        candidate=actor.prepare_image(rows,frame=188,households=houses)
        harness.before_commit=lambda:actor.identity.reset_source(replace(actor.sources[1],source_epoch=1))
        with pytest.raises(WorldPublicationRefused):harness.publish(candidate,188,source_id=2)
    else:
        with pytest.raises(WorldPublicationRefused):
            actor.prepare_image(rows,frame=188,households=houses,
                mode='strict' if defect=='strict-mode' else 'physical_only_quarantine')
    assert actor.world.current_snapshot() is first
    assert before==(len(harness.wire),len(harness.journal),actor.world._revision)
    assert actor.identity.stats().pending_reservations==0


def test_attribution_r19_actual_image_one_key_divergent_visitors_wire_and_recovery():
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','synthetic-k',1004,1),
        (2,101):HouseholdDescriptor('visitor','synthetic-f',1000,1)}
    rows={1:[(1,1.,1,1)],2:[(101,1.03,1,1)]}
    with pytest.raises(WorldPublicationRefused,match='settled_household_claim_conflict'):
        actor.prepare_image(rows,households=houses,mode='strict')
    assert actor.world.current_snapshot() is None and not harness.wire and not harness.journal
    assert actor.identity.stats().pending_reservations==0
    actor.us+=100_000
    first=harness.publish(actor.prepare_image(rows,frame=3,households=houses),3)
    assert len(first.entities)==1
    a=first.entities[0];q=a.attribution_quarantine
    assert q.component_entity_ids==(a.entity_id,) and len(q.owners)==2
    assert q.stable_ids==(1000,1004) and a.subject.kind=='provisional' and a.subject.stable_id is None
    assert {o.household.stable_id for _,o in actor.identity.ledger.household_claim_owners()}=={1000,1004}
    actor.us+=100_000
    held=harness.publish(actor.prepare_image(rows,frame=4,households=houses,
        source_changes={1:{'world_covariance':None},2:{'world_covariance':None}}),4)
    b=held.entities[0]
    assert (b.position,b.observed_at_us,b.stale_after_us,b.continuity_segment_id)==(a.position,a.observed_at_us,a.stale_after_us,a.continuity_segment_id)
    assert {c.observation_id for c in q.owners}<={c.observation_id for c in b.attribution_quarantine.retained_owners}
    actor.us+=100_000
    corrected={(1,1):houses[(1,1)],(2,101):houses[(1,1)]}
    final=harness.publish(actor.prepare_image(rows,frame=5,households=corrected,mode='strict'),5)
    assert final.entities[0].entity_id==a.entity_id and final.entities[0].attribution_quarantine is None
    assert final.entities[0].subject.stable_id==1004
    actual=actual_node_wire(actor,harness.wire,'r19-one-key-held-recovery')
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert [len(s['people']) for s in actual['snapshots']]==[1,1,1]
    assert actual['snapshots'][0]['people'][0]['stableId'] is None


def test_attribution_r19_cross_key_closure_and_queued_abort_preserve_all_originals():
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','synthetic-k',1004,1),
        (2,101):HouseholdDescriptor('visitor','synthetic-f',1000,1),
        (1,2):HouseholdDescriptor('visitor','synthetic-k',1004,1)}
    rows={1:[(1,1.,1,1),(2,4.,1,1)],2:[(101,1.03,1,1)]}
    candidate=actor.prepare_image(rows,households=houses)
    snapshot=candidate.publication.snapshot
    assert len(snapshot.entities)==2
    records=[e.attribution_quarantine for e in snapshot.entities]
    assert records[0]==records[1] and len(records[0].component_entity_ids)==2
    assert len(records[0].owners)==3 and records[0].stable_ids==(1000,1004)
    harness.before_commit=lambda:actor.identity.reset_source(replace(actor.sources[1],source_epoch=1))
    with pytest.raises(WorldPublicationRefused):harness.publish(candidate,2)
    assert not harness.wire and not harness.journal and actor.world.current_snapshot() is None
    assert actor.identity.stats().pending_reservations==0


def test_attribution_r19_retained_other_head_component_and_expiry_wire():
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','synthetic-k',1004,1),
        (2,101):HouseholdDescriptor('visitor','synthetic-f',1000,1),
        (1,2):HouseholdDescriptor('visitor','synthetic-k',1004,1)}
    initial=harness.publish(actor.prepare_image({1:[(2,4.,1,1)]},households=houses,mode='strict'),2)
    other=initial.entities[0]
    actor.us+=100_000
    second=harness.publish(actor.prepare_image({1:[(1,1.,1,1)],2:[(101,1.03,1,1)]},frame=3,households=houses),3)
    assert len(second.entities)==2 and all(e.attribution_quarantine for e in second.entities)
    kept=next(e for e in second.entities if e.entity_id==other.entity_id)
    assert (kept.position,kept.observed_at_us,kept.stale_after_us,kept.continuity_segment_id)==(other.position,other.observed_at_us,other.stale_after_us,other.continuity_segment_id)
    assert len(kept.attribution_quarantine.component_entity_ids)==2
    assert {s.observation_id for s in other.sources if s.accepted}<={c.observation_id for c in kept.attribution_quarantine.owners+kept.attribution_quarantine.retained_owners}
    actor.us+=2_100_000
    preps,_=actor.normalize({1:[]},frame=30,native_tokens=True,world_resolver_confidence=.8)
    actor.context=actor.clock_context()
    memberships=actor.identity.spatial_memberships(at_interval=actor.context)
    candidate=actor.world.prepare_spatial(1,tuple(preps),memberships,at_interval=actor.context,
        attribution_mode='physical_only_quarantine')
    final=harness.publish(candidate,30)
    assert not final.entities
    actual=actual_node_wire(actor,harness.wire,'r19-retained-component-expiry')
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert [len(s['people']) for s in actual['snapshots']]==[1,2,0]


def test_attribution_r21_image_split_keeps_original_quarantine_without_current_vote():
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','synthetic-k',1001,1),
        (2,101):HouseholdDescriptor('visitor','synthetic-f',1002,1)}
    first=harness.publish(actor.prepare_image({1:[(1,1.,1,1)],2:[(101,1.03,1,1)]},households=houses),2)
    previous=first.entities[0]
    actor.us+=100_000
    candidate=actor.prepare_image({1:[(1,1.1,1,1)]},frame=3,households=houses,
        image_y={(1,1):80.},source_changes={1:{'world_covariance':None}})
    second=harness.publish(candidate,3)
    assert len(second.entities)==1
    kept=second.entities[0]
    assert kept.attribution_quarantine==previous.attribution_quarantine
    assert (kept.position,kept.observed_at_us,kept.stale_after_us)==(previous.position,previous.observed_at_us,previous.stale_after_us)
    assert actor.identity.stats().pending_reservations==0
    actual=actual_node_wire(actor,harness.wire,'r21-split-held-only')
    assert all(d['accepted'] for d in actual['diagnostics'])


def test_attribution_r21_split_cannot_reuse_missing_quarantine_as_authority(caplog):
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','synthetic-k',1001,1),
        (2,101):HouseholdDescriptor('visitor','synthetic-f',1002,1)}
    first=harness.publish(actor.prepare_image({1:[(1,1.,1,1)],2:[(101,1.03,1,1)]},households=houses),2)
    actor.us+=100_000
    # Without a committed typed record there is no fallback authority,
    # even when all candidate positions happen to be unavailable.
    actor.world._fusion.set_attribution_quarantines({})
    before=(len(harness.wire),len(harness.journal),actor.world._revision)
    changes={1:{'world_covariance':None}}
    with pytest.raises(WorldPublicationRefused,match='retained_attribution_conflict_not_authorized'):
        actor.prepare_image({1:[(1,1.1,1,1)]},frame=3,households=houses,image_y={(1,1):80.},source_changes=changes)
    assert actor.world.current_snapshot() is first
    assert before==(len(harness.wire),len(harness.journal),actor.world._revision)
    assert actor.identity.stats().pending_reservations==0
    logs=[r.message for r in caplog.records if r.message.startswith('Attribution closure refusal ')]
    assert len(logs)==1 and '"retained":' in logs[0] and '"current":' in logs[0]


def _r25_public_held_claim_sequence():
    """R13 topology: held F105 carries K104 while decisions advance to K106/F107."""
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','synthetic-k',1001,1),
            (2,101):HouseholdDescriptor('visitor','synthetic-f',1002,1)}
    harness.publish(actor.prepare_image({1:[(1,1.,2,2)]},frame=104,households=houses,
        source_changes={1:{'world_covariance':tuple(v*4 for v in COV)}}),104)
    actor.us+=30_000
    first=harness.publish(actor.prepare_image({2:[(101,1.03,1,2)]},frame=105,households=houses,
        source_changes={2:{'world_covariance':tuple(v*.5 for v in COV)}}),105,source_id=2)
    old=first.entities[0]
    actor.us+=30_000
    harness.publish(actor.prepare_image({1:[(1,1.01,2,2)]},frame=106,households=houses,
        source_changes={1:{'world_covariance':tuple(v*4 for v in COV)}}),106)
    actor.us+=30_000
    previous=harness.publish(actor.prepare_image({2:[(101,1.03,1,2)]},frame=107,households=houses,
        source_changes={2:{'world_covariance':None}}),107,source_id=2)
    return actor,harness,houses,old,previous


def test_attribution_r25_exact_r13_overwritten_peer_claim_survives_real_rekey():
    actor,harness,houses,old,previous=_r25_public_held_claim_sequence()
    assert (previous.entities[0].position,previous.entities[0].observed_at_us,
        previous.entities[0].stale_after_us)==(old.position,old.observed_at_us,old.stale_after_us)
    custody=old.attribution_quarantine.owners
    assert {(c.source_id,c.frame_id) for c in custody}=={(1,104),(2,105)}
    assert {(c.source_id,c.frame_id) for c in actor.world._fusion.spatial_attribution_claims()[old.entity_id]}=={(1,106),(2,105)}
    all_live=actor.world._fusion.spatial_original_attribution_claims()[old.entity_id]
    assert {(c.source_id,c.frame_id) for c in all_live}=={(1,104),(1,106),(2,105)}
    detached=actor.world._fusion.spatial_original_attribution_claims();detached.clear()
    assert actor.world._fusion.spatial_original_attribution_claims()[old.entity_id]==all_live
    # Exact causal mismatch: the last typed record arbitrates K106/F107, while
    # F105's legitimate immutable sample still owns its K104 contributor claim.
    record=previous.entities[0].attribution_quarantine
    assert {(c.source_id,c.frame_id) for c in record.owners}=={(1,106),(2,107)}
    assert any(c.source_id==2 and c.frame_id==105 for c in record.retained_owners)
    actor.us+=3_000
    candidate=actor.prepare_image({1:[(1,1.02,2,2)]},frame=107,households=houses,
        image_y={(1,1):80.},source_changes={1:{'world_covariance':tuple(v*4 for v in COV)}})
    assert all(d.household_conflict is None for d in actor.memberships)
    prepared=actor.world._prepared_candidates[candidate._candidate_id][1]
    assert len(prepared.spatial_reservations)==1 and prepared.spatial_reservations[0].purpose=='strict'
    final=harness.publish(candidate,107)
    assert len(final.entities)==2
    held=next(e for e in final.entities if e.entity_id==old.entity_id)
    fresh=next(e for e in final.entities if e.entity_id!=old.entity_id)
    assert (held.position,held.observed_at_us,held.stale_after_us)==(old.position,old.observed_at_us,old.stale_after_us)
    q=held.attribution_quarantine
    assert q==fresh.attribution_quarantine and len(q.component_entity_ids)==2
    original_claims=set(q.owners+q.retained_owners)
    assert set(custody)<=original_claims
    assert all(e.subject.kind=='provisional' and e.subject.stable_id is None for e in final.entities)
    actual=actual_node_wire(actor,harness.wire,'r25-exact-r13-held-claim-rekey')
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert [len(s['people']) for s in actual['snapshots']]==[1,1,1,1,2]


@pytest.mark.parametrize('defect',['missing-anchor','hidden-fabrication','owner-denies','strict'])
def test_attribution_r25_full_original_custody_cannot_replace_authority(defect,monkeypatch):
    from noesis.identity_association_service import SpatialMemberResult
    actor,harness,houses,old,previous=_r25_public_held_claim_sequence()
    if defect=='missing-anchor':actor.world._fusion.set_attribution_quarantines({})
    if defect=='owner-denies':
        monkeypatch.setattr(actor.identity,'prepare_spatial_member',lambda *a,**kw:SpatialMemberResult('denied','actual-owner-denied'))
    if defect=='hidden-fabrication':
        actual=GlobalWorldFusion.spatial_original_attribution_claims
        def altered(fusion):
            rows=actual(fusion)
            if fusion is not actor.world._fusion:
                claim=rows[old.entity_id][0]
                rows[old.entity_id]+= (claim.model_copy(update={'raw_id':99,'observation_id':'unowned-hidden-claim'}),)
            return rows
        monkeypatch.setattr(GlobalWorldFusion,'spatial_original_attribution_claims',altered)
    before=(len(harness.wire),len(harness.journal),actor.world._revision)
    actor.us+=3_000
    with pytest.raises(WorldPublicationRefused):
        actor.prepare_image({1:[(1,1.02,2,2)]},frame=107,households=houses,image_y={(1,1):80.},
            source_changes={1:{'world_covariance':tuple(v*4 for v in COV)}},
            mode='strict' if defect=='strict' else 'physical_only_quarantine')
    assert actor.world.current_snapshot() is previous
    assert before==(len(harness.wire),len(harness.journal),actor.world._revision)
    assert actor.identity.stats().pending_reservations==0


@pytest.mark.parametrize('barrier',['expiry','termination','reset'])
def test_attribution_r25_original_sample_custody_releases_on_real_lifecycle(barrier):
    actor,harness,houses,old,previous=_r25_public_held_claim_sequence()
    actor.us+=3_000
    split=harness.publish(actor.prepare_image({1:[(1,1.02,2,2)]},frame=107,households=houses,
        image_y={(1,1):80.},source_changes={1:{'world_covariance':tuple(v*4 for v in COV)}}),107)
    fresh=next(e for e in split.entities if e.entity_id!=old.entity_id)
    withdrawn=retired=()
    if barrier=='expiry':actor.us+=2_100_000
    else:
        actor.us+=10_000
        if barrier=='reset':
            binding=replace(actor.sources[1],source_epoch=1)
            result=actor.identity.reset_source(binding);actor.sources=(actor.sources[0],binding)
        else:
            original=next(o for o in actor.originals.values() if o.stamp.member.source_id==2)
            result=actor.identity.withdraw(MemberRemoval('r25-termination',original.stamp,'termination',108,
                actor.us/1e6,original.stamp.clock_domain,10.8))
        withdrawn,retired=result.withdrawn_members,result.retired_keys
    preps,_=actor.normalize({1:[]},frame=130 if barrier=='expiry' else 108,native_tokens=True,world_resolver_confidence=.8)
    actor.context=actor.clock_context()
    candidate=actor.world.prepare_spatial(1,preps,actor.identity.spatial_memberships(at_interval=actor.context),
        at_interval=actor.context,withdrawn_members=withdrawn,retired_keys=retired,attribution_mode='physical_only_quarantine')
    final=harness.publish(candidate,130 if barrier=='expiry' else 108)
    if barrier=='expiry':
        assert not final.entities
        assert actor.world._fusion.spatial_original_attribution_claims()=={}
    else:
        assert len(final.entities)==1
        remaining=final.entities[0]
        assert remaining.entity_id==fresh.entity_id and remaining.attribution_quarantine is None
        assert remaining.subject.stable_id==1001
        assert (remaining.position,remaining.observed_at_us,remaining.stale_after_us)==(fresh.position,fresh.observed_at_us,fresh.stale_after_us)
        assert set(actor.world._fusion.spatial_original_attribution_claims())=={fresh.entity_id}
    actual=actual_node_wire(actor,harness.wire,'r25-original-custody-'+barrier)
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert len(actual['snapshots'][-1]['people'])==(0 if barrier=='expiry' else 1)


def test_bestview_r25_explicit_native_seed_basis_retains_exact_better_original():
    actor=ImageQuarantineActor(live_track_continuity=True);harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','synthetic-k',1001,1),
            (2,101):HouseholdDescriptor('visitor','synthetic-f',1002,1)}
    harness.publish(actor.prepare_image({2:[(101,1.,1,1)]},households=houses,
        source_changes={2:{'world_covariance':tuple(v*.5 for v in COV)}}),2,source_id=2)
    actor.us+=20_000
    harness.publish(actor.prepare_image({1:[(1,1.01,1,1)]},frame=3,households=houses,
        source_changes={1:{'world_covariance':tuple(v*4 for v in COV)}}),3)
    # Keep the same real native roster current while original image support ages.
    for frame,source in ((4,2),(5,1),(6,2)):
        actor.us+=190_000 if source==2 else 10_000
        raw=101 if source==2 else 1
        last=harness.publish(actor.prepare_image({source:[(raw,1.02,1,1)]},frame=frame,households=houses,
            source_changes={source:{'world_covariance':tuple(v*(.5 if source==2 else 4) for v in COV)}},
            omit_image_current=True),frame,source_id=source)
    assert actor.memberships[0].association_basis=='estimated_live_track_continuity'
    before=last.entities[0];frontiers=actor.identity.stats().spatial_member_frontiers
    actor.us+=10_000
    final=harness.publish(actor.prepare_image({1:[(1,1.03,1,1)]},frame=7,households=houses,
        source_changes={1:{'world_covariance':tuple(v*4 for v in COV)}},omit_image_current=True),7)
    assert actor.memberships[0].association_basis=='estimated_live_track_continuity'
    held=final.entities[0]
    assert (held.entity_id,held.position,held.observed_at_us,held.stale_after_us,held.continuity_segment_id)==(
        before.entity_id,before.position,before.observed_at_us,before.stale_after_us,before.continuity_segment_id)
    assert actor.identity.stats().spatial_member_frontiers==frontiers
    actor.us+=10_000
    reversed=harness.publish(actor.prepare_image({1:[(1,1.04,1,1)]},frame=8,households=houses,
        source_changes={1:{'world_covariance':tuple(v*.1 for v in COV)}},omit_image_current=True),8)
    better=reversed.entities[0]
    assert actor.memberships[0].association_basis=='estimated_live_track_continuity'
    assert better.entity_id==held.entity_id and better.observed_at_us>held.observed_at_us
    assert [s.camera_id for s in better.sources if s.accepted]==['kitchen']
    actual=actual_node_wire(actor,harness.wire,'r25-native-seed-bestview')
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert all(len(s['people'])==1 for s in actual['snapshots'])


def _r35_current_original_without_pair(actor, source, raw, frame, x, houses, *, region=False,
                                      withdrawn=(),retired=()):
    """Actual current producer APIs; never reuse the shadow's old pose as new."""
    from reid.mv3dt_person_association import (EstimatedImageRegionProof,ImageRegionOriginal,
        ObservedImageRegion,CurrentVisualDeliveryContext,CurrentVisualIngress)
    preps,native=actor.normalize({source:[(raw,x,1,1)]},frame=frame,native_tokens=True,
        households=houses,world_resolver_confidence=.8)
    current=next(iter(native.values()))
    observed=actor.identity.observe_spatial(current,SpatialLifecycleReceipt(
        'r35-current:'+current.stamp.observation_id,current.stamp,'ACTIVE'))
    assert observed.disposition=='observed' or (observed.disposition=='unresolved'
        and observed.reason=='current_cross_source_proof_required'),observed
    actor.originals[current.stamp.member]=current
    actor.context=actor.clock_context();actor.image_sequence+=1
    mono=100.+(actor.us-10_200_000)/1e6
    originals=()
    if region:
        camera=next(c for c in actor.image_policy.cameras if c.source.source_id==source)
        u=50. if source==1 else 40.
        originals=(ImageRegionOriginal(current.stamp,mono,'r35-fresh-body:'+current.stamp.observation_id,
            camera.raster_id,camera.prior_id,camera.image_size,(u-5.,25.,10.,25.),
            (ObservedImageRegion('pose_shoulder_support',(u,30.),(u,30.,u,30.,2.,2.)),
             ObservedImageRegion('pose_torso_support',(u,35.),(u,30.,u,40.,2.,2.)))),)
        actor.image_originals[current.stamp.member]=originals[0]
    else:actor.image_originals.pop(current.stamp.member,None)
    result=actor.identity.join_estimated_image_regions(EstimatedImageRegionProof('r35-no-pair:'+str(frame),
        actor.identity.image_region_active_cohort(),originals),at_interval=actor.context,
        delivery_context=CurrentVisualDeliveryContext('synthetic-image-world',actor.image_sequence,mono,
            (CurrentVisualIngress(current.stamp,mono),)))
    assert result.disposition in {'accepted','observed'},result
    memberships=actor.identity.spatial_memberships(at_interval=actor.context)
    candidate=actor.world.prepare_spatial(source,preps,memberships,at_interval=actor.context,
        withdrawn_members=withdrawn+observed.withdrawn_members+result.withdrawn_members,
        retired_keys=retired+observed.retired_keys+result.retired_keys,attribution_mode='physical_only_quarantine')
    return candidate,current,memberships


def _r35_seed_and_shadow(*,live=True,unrelated=False):
    from reid.mv3dt_person_association import SpatialLifecycleUpdate
    actor=ImageQuarantineActor(live_track_continuity=live);harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','r35-k',1001,1),
        (2,101):HouseholdDescriptor('visitor','r35-f',1000,1)}
    first=harness.publish(actor.prepare_image({1:[(1,1.,1,1)]+([(2,4.,1,1)]if unrelated else []),2:[(101,1.03,1,1)]},frame=110,
        households=houses,source_changes={1:{'world_covariance':tuple(v*.1 for v in COV)}}),110)
    old=next(o for o in actor.originals.values() if o.stamp.member.source_id==1)
    actor.us+=50_000
    shadow=actor.identity.update_spatial_lifecycle(SpatialLifecycleUpdate('r35-K111',old.stamp,111,
        actor.us/1e6,old.stamp.clock_domain,11.1))
    assert shadow.disposition=='observed'
    actor.image_originals.pop(old.stamp.member,None)
    return actor,harness,houses,first,old,shadow


def _r35_public_shadow_handoff(*,live=True):
    actor,harness,houses,first,old,shadow=_r35_seed_and_shadow(live=live)
    actor.us+=50_000
    candidate,peer,peer_members=_r35_current_original_without_pair(actor,2,101,112,1.04,houses,region=True)
    second=harness.publish(candidate,112,source_id=2)
    actor.us+=50_000
    candidate,current,members=_r35_current_original_without_pair(actor,1,1,113,1.05,houses)
    return actor,harness,houses,first,old,shadow,second,candidate,current,members,peer,peer_members


def test_shadow_r35_live_handoff_actual_world_publisher_node_one_key_no_label_leak():
    actor,harness,houses,first,old,shadow,second,candidate,current,members,peer,peer_members=_r35_public_shadow_handoff()
    state=actor.world._prepared_candidates[candidate._candidate_id][1]
    assert len(state.spatial_reservations)==1
    assert state.spatial_reservations[0].reference.stamp==current.stamp
    assert state.spatial_reservations[0].purpose=='physical_only_quarantine'
    third=harness.publish(candidate,113)
    actual=actual_node_wire(actor,harness.wire,'r35-live-shadow-handoff')
    assert [len(s['people']) for s in actual['snapshots']]==[1,1,1]
    key=first.entities[0].entity_id
    assert second.entities[0].entity_id==third.entities[0].entity_id==key
    assert not shadow.withdrawn_members and not shadow.retired_keys
    shadow_view=next(e for d in peer_members for e in d.eligibility if e.observation==old.stamp)
    assert not shadow_view.eligible
    assert shadow_view.lifecycle_receipt.native_state=='ACTIVE'
    original_view=next(m for d in peer_members for m in d.members if m.observation==old.stamp)
    assert original_view.household==houses[(1,1)]
    assert [s.camera_id for s in second.entities[0].sources if s.accepted]==['family-room']
    assert [s.camera_id for s in third.entities[0].sources if s.accepted]==['kitchen']
    for snapshot in (first,second,third):
        entity=snapshot.entities[0]
        assert entity.subject.kind=='provisional' and entity.subject.stable_id is None
        assert entity.attribution_quarantine.stable_ids==(1000,1001)
    assert third.entities[0].observed_at_us>second.entities[0].observed_at_us>first.entities[0].observed_at_us
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert all(p['stableId']is None for s in actual['snapshots'] for p in s['people'])


def test_shadow_r35_default_false_preserves_prior_split_and_requires_real_pair():
    actor,harness,_,first,old,shadow,second,candidate,*_=_r35_public_shadow_handoff(live=False)
    third=harness.publish(candidate,113)
    assert shadow.withdrawn_members==(old.stamp.member,)
    assert len(third.entities)==2
    actual=actual_node_wire(actor,harness.wire,'r35-default-shadow-split')
    assert [len(s['people']) for s in actual['snapshots']]==[1,1,2]
    assert all(d['accepted'] for d in actual['diagnostics'])


def test_shadow_r35_both_nonvoting_do_not_renew_original_point_or_expiry():
    from reid.mv3dt_person_association import SpatialLifecycleUpdate
    actor,harness,houses,first,old,shadow=_r35_seed_and_shadow()
    other=next(o for o in actor.originals.values()if o.stamp.member.source_id==2)
    result=actor.identity.update_spatial_lifecycle(SpatialLifecycleUpdate('r35-F111',other.stamp,111,
        actor.us/1e6,other.stamp.clock_domain,11.1))
    assert result.disposition=='observed'
    actor.us+=800_000
    def empty(frame):
        preps,_=actor.normalize({1:[]},frame=frame,native_tokens=True,world_resolver_confidence=.8)
        actor.context=actor.clock_context()
        members=actor.identity.spatial_memberships(at_interval=actor.context)
        assert all(not e.eligible for d in members for e in d.eligibility)
        candidate=actor.world.prepare_spatial(1,preps,members,at_interval=actor.context,
            attribution_mode='physical_only_quarantine')
        assert not actor.world._prepared_candidates[candidate._candidate_id][1].spatial_reservations
        return harness.publish(candidate,frame)
    held=empty(112)
    assert len(held.entities)==1 and held.entities[0].position_state=='held'
    a,b=first.entities[0],held.entities[0]
    assert (a.position,a.covariance,a.observed_at_us,a.stale_after_us)==(b.position,b.covariance,b.observed_at_us,b.stale_after_us)
    actor.us+=1_300_000;expired=empty(113)
    assert not expired.entities
    actual=actual_node_wire(actor,harness.wire,'r35-both-shadow-expiry')
    assert all(d['accepted']for d in actual['diagnostics'])
    assert [len(s['people'])for s in actual['snapshots']]==[1,1,0]
    assert len(actual['snapshots'][1]['people'][0]['trail'])==1


def test_shadow_r35_unrelated_distinct_person_is_not_absorbed():
    actor,harness,houses,first,old,shadow=_r35_seed_and_shadow(unrelated=True)
    actor.us+=50_000
    candidate,peer,members=_r35_current_original_without_pair(actor,2,101,112,1.04,houses,region=True)
    second=harness.publish(candidate,112,source_id=2)
    assert len(first.entities)==len(second.entities)==2
    assert {e.entity_id for e in first.entities}=={e.entity_id for e in second.entities}
    assert all(len(d.members)<=2 for d in members)
    actual=actual_node_wire(actor,harness.wire,'r35-shadow-distinct-peer')
    assert all(d['accepted']for d in actual['diagnostics'])
    assert [len(s['people'])for s in actual['snapshots']]==[2,2]


@pytest.mark.parametrize('defect',['strict-label','queued-reset'])
def test_shadow_r35_claims_and_queued_authority_remain_atomic(defect):
    actor,harness,_,first,old,shadow,second,candidate,current,members,*_=_r35_public_shadow_handoff()
    before=(len(harness.wire),len(harness.journal),actor.world._revision)
    if defect=='strict-label':
        reference=actor.world._prepared_candidates[candidate._candidate_id][1].spatial_reservations[0].reference
        dto=next(d for d in members if any(m.observation==current.stamp for m in d.members))
        strict=actor.identity.prepare_spatial_member(dto,reference,at_interval=actor.context)
        assert strict.disposition!='reserved' and strict.household_conflict is not None
        actor.world.discard(candidate)
    else:
        harness.before_commit=lambda:actor.identity.reset_source(replace(actor.sources[0],source_epoch=1))
        with pytest.raises(WorldPublicationRefused):harness.publish(candidate,113)
    assert actor.world.current_snapshot() is second
    assert (len(harness.wire),len(harness.journal),actor.world._revision)==before
    assert actor.identity.stats().pending_reservations==0


@pytest.mark.parametrize('barrier',['absence','termination','reset'])
def test_shadow_r35_real_native_barrier_cannot_leave_hidden_retained_member(barrier):
    actor,harness,houses,first,old,shadow=_r35_seed_and_shadow()
    actor.us+=50_000
    if barrier=='reset':
        binding=replace(actor.sources[0],source_epoch=1)
        event=actor.identity.reset_source(binding);actor.sources=(binding,actor.sources[1])
    else:
        event=actor.identity.withdraw(MemberRemoval('r35-real-'+barrier,old.stamp,barrier,112,
            actor.us/1e6,old.stamp.clock_domain,11.2))
    assert event.disposition in {'removed','reset'}
    actor.originals.pop(old.stamp.member,None)
    if barrier=='reset':
        from reid.mv3dt_person_association import EstimatedImageRegionProof,CurrentVisualDeliveryContext
        actor.context=actor.clock_context();actor.image_sequence+=1
        denied=actor.identity.join_estimated_image_regions(EstimatedImageRegionProof('r35-reset-no-old-prior',
            actor.identity.image_region_active_cohort(),()),at_interval=actor.context,
            delivery_context=CurrentVisualDeliveryContext('synthetic-image-world',actor.image_sequence,
                100.+(actor.us-10_200_000)/1e6,()))
        assert denied.disposition=='association_unresolved' and denied.reason=='image_region_source_binding'
        assert actor.world.current_snapshot() is first
        return  # Old epoch/prior cannot silently authorize the reset source.
    candidate,peer,members=_r35_current_original_without_pair(actor,2,101,113,1.04,houses,region=True,
        withdrawn=event.withdrawn_members,retired=event.retired_keys)
    assert all(m.observation.member!=old.stamp.member for d in members for m in d.members)
    final=harness.publish(candidate,113,source_id=2)
    assert len(final.entities)==1
    assert all(s.camera_id!='kitchen'for e in final.entities for s in e.sources)


@pytest.mark.parametrize('contradiction',['self-false','alternative-true'])
def test_shadow_r35_real_image_contradiction_revokes_retained_association(contradiction):
    actor,harness,houses,first,old,shadow,second,candidate,*_=_r35_public_shadow_handoff()
    harness.publish(candidate,113)
    actor.us+=50_000
    next_candidate=actor.prepare_image({1:[(1,1.06,1,1)]},frame=114,households=houses,
        image_y={(1,1):80.}if contradiction=='self-false'else None)
    final=harness.publish(next_candidate,114)
    if contradiction=='alternative-true':
        actor.us+=50_000
        final=harness.publish(actor.prepare_image({2:[(101,1.04,1,1),(202,4.,1,115)]},frame=115,
            households=houses,image_y={(2,202):30.}),115,source_id=2)
    same_pair=lambda d:{(m.observation.member.source_id,m.observation.member.raw_id)for m in d.members}>={(1,1),(2,101)}
    assert not any(same_pair(d)for d in actor.memberships)
    if contradiction=='alternative-true':assert len(final.entities)>=2
    else:
        # Rekeying invalidates the old peer grant; no current Family point
        # is fabricated just to display two keys after a real contradiction.
        assert all(not ({s.camera_id for s in e.sources if s.accepted}>={'kitchen','family-room'})
            for e in final.entities)
    actual=actual_node_wire(actor,harness.wire,'r35-revoke-'+contradiction)
    assert all(d['accepted']for d in actual['diagnostics'])


def test_shadow_r35_raw_id_new_birth_cannot_inherit_retained_physical_key():
    from reid.mv3dt_person_association import (EstimatedImageRegionProof,
        CurrentVisualDeliveryContext,CurrentVisualIngress)
    actor,harness,houses,first,old,shadow=_r35_seed_and_shadow()
    actor.us+=50_000
    ended=actor.identity.withdraw(MemberRemoval('r35-real-termination',old.stamp,'termination',112,
        actor.us/1e6,old.stamp.clock_domain,11.2))
    assert ended.disposition=='removed';actor.originals.pop(old.stamp.member)
    actor.us+=50_000
    preps,native=actor.normalize({1:[(1,1.05,2,113)]},frame=113,native_tokens=True,
        households=houses,world_resolver_confidence=.8)
    new=next(iter(native.values()))
    assert actor.identity.observe_spatial(new,SpatialLifecycleReceipt('r35-new-birth',new.stamp,'ACTIVE')).disposition=='observed'
    actor.originals[new.stamp.member]=new;actor.context=actor.clock_context()
    actor.image_sequence+=1;mono=100.+(actor.us-10_200_000)/1e6
    join=actor.identity.join_estimated_image_regions(EstimatedImageRegionProof('r35-no-new-birth-proof',
        actor.identity.image_region_active_cohort(),()),at_interval=actor.context,
        delivery_context=CurrentVisualDeliveryContext('synthetic-image-world',actor.image_sequence,mono,
            (CurrentVisualIngress(new.stamp,mono),)))
    assert join.disposition=='observed'
    memberships=actor.identity.spatial_memberships(at_interval=actor.context)
    owner=next(d for d in memberships if any(m.observation==new.stamp for m in d.members))
    assert GlobalWorldFusion.physical_entity_id(owner.physical_key)!=first.entities[0].entity_id
    candidate=actor.world.prepare_spatial(1,preps,memberships,at_interval=actor.context,
        withdrawn_members=ended.withdrawn_members+join.withdrawn_members,
        retired_keys=ended.retired_keys+join.retired_keys,attribution_mode='physical_only_quarantine')
    final=harness.publish(candidate,113)
    assert len(final.entities)==1 and final.entities[0].entity_id!=first.entities[0].entity_id
    actual=actual_node_wire(actor,harness.wire,'r35-new-birth-no-alias')
    assert all(d['accepted']for d in actual['diagnostics'])
    assert [len(s['people'])for s in actual['snapshots']]==[1,1]


def _r31_upright_changes():
    # Exact compact producer distinction from R17 F3/frame181; UNKNOWN is
    # intentional and is NOT an observed floor contact or measured depth.
    return dict(world_source='gravity_drop',world_quality='estimated',
        world_support_state='unknown',world_support_surface=None,world_posture='standing',
        world_resolver_selected_id='upright_pose_projection',
        world_resolver_contact_basis='pose:upright_body_planes')


def _r65_column_changes():
    return dict(world_source='registered_body_column',world_quality='estimated',
        world_support_state='unknown',world_support_surface=None,world_posture='unknown',
        world_resolver_selected_id='registered_body_column',
        world_resolver_contact_basis='depth:registered_body_column')


def test_r75_actual_torso_receipt_pgs_world_publisher_node_reanchor_and_expiry():
    """Synthetic source geometry; actual receipt producer/filter/owner/wire/Node."""
    from DS9.tests.test_registered_body_column import _r75_current_column,_r75_apply,_r75_occluded_state
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    state=_r75_occluded_state()
    counts=[]
    for i in range(3):
        p,t,c,d=_r75_current_column(179+i*6,6_000_000_000+i*200_000_000)
        point,_=_r75_apply(p,t,c,d,state)
        actor.us+=200_000
        change={**_r65_column_changes(),'world':list(point),'world_measurement_accepted':state.measurement_accepted,
            'world_valid':state.measurement_accepted,'world_measurement_rejection_reason':state.measurement_rejection_reason}
        snap=harness.publish(actor.prepare_image({2:[(4,float(point[0]),4,4)]},frame=t['frame_id'],
            source_changes={2:change}),t['frame_id'],source_id=2)
        counts.append(len(snap.entities))
    assert counts==[0,0,1]
    person=snap.entities[0]
    assert (person.position.x,person.position.z)==(float(point[0]),float(point[2]))
    assert person.support_state==person.posture=='unknown'
    assert person.position_state=='measured' and state.trail_break_required
    actor.us+=3_000_000
    expired=harness.publish(actor.prepare_image({2:[]},frame=200),200,source_id=2)
    assert not expired.entities
    actual=actual_node_wire(actor,harness.wire,'r75-real-column-reanchor')
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert [len(s['people']) for s in actual['snapshots']]==[0,0,1,0]


def test_r77_strong_current_posed_column_producer_pgs_world_publisher_node():
    from DS9.tests.test_registered_body_column import _r77_strong_fixture,_r77_resolved,_r75_apply,_r75_occluded_state
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor);state=_r75_occluded_state()
    counts=[]
    for i in range(3):
        p,t,c,d,joints=_r77_strong_fixture(242+i*6,8_074_733_333+i*200_000_000)
        assert p._person_admission_confidence(t,trusted_lifecycle=True)<.60
        selected=_r77_resolved(p,t,c,d,joints)
        point,_=_r75_apply(p,t,selected,d,state)
        actor.us+=200_000
        change={**_r65_column_changes(),'world':list(point),'world_measurement_accepted':state.measurement_accepted,
            'world_valid':state.measurement_accepted,'world_measurement_rejection_reason':state.measurement_rejection_reason}
        snap=harness.publish(actor.prepare_image({2:[(4,float(point[0]),4,4)]},frame=t['frame_id'],
            source_changes={2:change}),t['frame_id'],source_id=2)
        counts.append(len(snap.entities))
    assert counts==[0,0,1] and state.reacquired and state.trail_break_required
    person=snap.entities[0]
    assert (person.position.x,person.position.z)==(float(point[0]),float(point[2]))
    assert person.posture==person.support_state=='unknown'
    actor.us+=3_000_000
    assert not harness.publish(actor.prepare_image({2:[]},frame=270),270,source_id=2).entities
    actual=actual_node_wire(actor,harness.wire,'r77-posed-column-actual-consumers')
    assert all(d['accepted']for d in actual['diagnostics'])
    assert [len(s['people'])for s in actual['snapshots']]==[0,0,1,0]


def test_r66_public_semantic_mode_column_claims_bestview_actual_node():
    """Synthetic proper-front originals; ACTUAL public owner/World/wire/reducer."""
    actor=ImageQuarantineActor(live_track_continuity=True,semantic_joint_max_mean_px=33.724887199080946)
    harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','semantic-k',1004,1),
        (2,101):HouseholdDescriptor('visitor','semantic-f',1000,1)}
    def joints(u,y):return ((5,u-2.,y,.9),(6,u+2.,y,.9),(11,u-1.,y+10.,.9),(12,u+1.,y+10.,.9))
    observed={(1,1):joints(50.,30.),(2,101):joints(40.,30.),(2,102):joints(40.,70.)}
    rows={1:[(1,1.,1,1)],2:[(101,1.03,1,1),(102,5.,2,1)]}
    first=harness.publish(actor.prepare_image(rows,households=houses,semantic_joints=observed,
        source_changes={1:_r65_column_changes(),2:{**_r65_column_changes(),
            'world_covariance':tuple(v*.1 for v in COV)}}),2)
    dto=next(d for d in actor.memberships if len(d.members)==2)
    assert dto.association_basis=='estimated_semantic_joint_correspondence'
    assert len(first.entities)==2
    person=next(e for e in first.entities if e.attribution_quarantine is not None)
    assert person.subject.kind=='provisional' and person.subject.stable_id is None
    assert person.attribution_quarantine.stable_ids==(1000,1004)
    assert person.posture==person.support_state=='unknown'
    assert [s.camera_id for s in person.sources if s.accepted]==['family-room']
    actor.us+=100_000
    retained=harness.publish(actor.prepare_image({1:[(1,1.01,1,1)]},frame=3,households=houses,
        semantic_joints=observed,source_changes={1:_r65_column_changes()}),3)
    same=next(e for e in retained.entities if e.entity_id==person.entity_id)
    assert same.position==person.position and same.observed_at_us==person.observed_at_us
    # F's exact original is still its current eligible stamp, not an advanced
    # no-position cohort. Reuse must not mint a new timestamp or wire deadline.
    assert same.position_state=='measured' and same.stale_after_us==person.stale_after_us
    assert same.attribution_quarantine is not None
    wire=actual_node_wire(actor,harness.wire,'r66-public-semantic-column-bestview-claims')
    assert all(d['accepted'] for d in wire['diagnostics'])
    assert [len(s['people']) for s in wire['snapshots']]==[2,2]


@pytest.mark.parametrize('actual_posture',['unknown','standing','sitting','lying'])
def test_r65_column_actual_world_publisher_node_unknown_then_floor_recovery(actual_posture):
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    first=harness.publish(actor.prepare_image({1:[(1,1.,1,1)]},
        source_changes={1:{**_r65_column_changes(),'world_posture':actual_posture}}),2)
    assert len(first.entities)==1
    entity=first.entities[0]
    assert entity.support_state=='unknown' and entity.posture==actual_posture
    assert entity.position_state=='measured' and entity.covariance.values==COV
    actor.us+=100_000
    second=harness.publish(actor.prepare_image({2:[(101,1.03,1,1)]},frame=3,
        source_changes={2:{'world_covariance':tuple(v*.1 for v in COV)}}),3,source_id=2)
    assert len(second.entities)==1 and second.entities[0].entity_id==entity.entity_id
    assert second.entities[0].support_state=='floor'
    wire=actual_node_wire(actor,harness.wire,'r65-column-unknown-floor')
    assert all(d['accepted'] for d in wire['diagnostics'])
    assert [len(s['people']) for s in wire['snapshots']]==[1,1]


@pytest.mark.parametrize('bad',[
    {'world_resolver_selected_id':None},{'world_resolver_contact_basis':'depth:pose_torso_support'},
    {'world_source':'gravity_drop'},{'world_resolver_contact_basis':None},
])
def test_r65_column_no_untyped_unknown_support(bad):
    actor=ImageQuarantineActor();changes={**_r65_column_changes(),**bad}
    prepared=actor.prepare_image({1:[(1,1.,1,1)]},source_changes={1:changes})
    result=ActualPublicationHarness(actor).publish(prepared,2)
    assert len(result.entities)==0


@pytest.mark.parametrize('actual_posture',['unknown','standing','sitting','lying'])
def test_r65_column_original_claims_quarantine_distinct_person_and_expiry(actual_posture):
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','k-column',1004,1),
        (2,101):HouseholdDescriptor('visitor','f-column',1000,1)}
    rows={1:[(1,1.,1,1)],2:[(101,1.03,1,1),(102,5.,2,1)]}
    first=harness.publish(actor.prepare_image(rows,households=houses,
        source_changes={1:{**_r65_column_changes(),'world_posture':actual_posture},
                        2:{**_r65_column_changes(),'world_posture':actual_posture}}),2)
    assert len(first.entities)==2
    person=next(e for e in first.entities if e.attribution_quarantine is not None)
    assert person.subject.kind=='provisional' and person.subject.stable_id is None
    assert person.attribution_quarantine.stable_ids==(1000,1004)
    assert person.support_state=='unknown' and person.posture==actual_posture
    actor.us+=3_000_000
    expired=harness.publish(actor.prepare_image({1:[]},frame=3),3)
    assert len(expired.entities)==0
    wire=actual_node_wire(actor,harness.wire,'r65-column-claims-expiry')
    assert all(d['accepted'] for d in wire['diagnostics'])
    assert [len(s['people']) for s in wire['snapshots']]==[2,0]


def test_upright_r31_prepared_original_unknown_support_then_better_floor_wire():
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    first=harness.publish(actor.prepare_image({1:[(1,1.,1,1)]},
        source_changes={1:_r31_upright_changes()}),2)
    assert len(first.entities)==1
    original=first.entities[0]
    assert original.support_state=='unknown' and original.posture=='standing'
    assert original.position_state=='measured' and original.position.x==1.
    assert original.covariance.values==COV
    actor.us+=100_000
    second=harness.publish(actor.prepare_image({2:[(101,1.03,1,1)]},frame=3,
        source_changes={2:{'world_covariance':tuple(v*.1 for v in COV)}}),3,source_id=2)
    assert len(second.entities)==1 and second.entities[0].entity_id==original.entity_id
    assert second.entities[0].support_state=='floor'
    assert [s.camera_id for s in second.entities[0].sources if s.accepted]==['family-room']
    actual=actual_node_wire(actor,harness.wire,'r31-typed-upright-unknown-floor-replacement')
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert [len(s['people']) for s in actual['snapshots']]==[1,1]


@pytest.mark.parametrize('changes',[
    {'world_resolver_selected_id':None},
    {'world_resolver_selected_id':'gravity_reconstruction'},
    {'world_resolver_contact_basis':'learned_body_height'},
    {'world_posture':'unknown'},
    {'world_posture':'sitting'},
    {'world_measurement_accepted':False},
])
def test_upright_r31_no_untyped_gravity_or_noncurrent_admission(changes):
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    result=harness.publish(actor.prepare_image({1:[(1,1.,1,1)]},
        source_changes={1:_r31_upright_changes()|changes}),2)
    assert not result.entities and actor.identity.stats().pending_reservations==0
    assert all(d['accepted'] for d in actual_node_wire(actor,harness.wire,'r31-untyped')['diagnostics'])


@pytest.mark.parametrize('defect',['borrowed','copied-envelope','stale-stamp','wrong-source','stale-media'])
def test_upright_r31_support_cannot_leave_its_prepared_original(defect):
    actor=ImageQuarantineActor()
    preps,_=actor.normalize({1:[(1,1.,1,1),(2,4.,1,1)]},native_tokens=True,
        world_resolver_confidence=.8,**_r31_upright_changes())
    state=actor.world._prepared_candidates[preps[0]._candidate_id][1]
    originals=preps[0].publication.observations
    support=state.spatial_upright_support[originals[0].observation_id]
    if defect=='borrowed':support=state.spatial_upright_support[originals[1].observation_id]
    elif defect=='copied-envelope':support=replace(support,original=originals[0].model_copy())
    elif defect=='stale-stamp':support=replace(support,stamp=replace(support.stamp,frame_id=support.stamp.frame_id+1))
    elif defect=='wrong-source':support=replace(support,stamp=replace(support.stamp,
        member=replace(support.stamp.member,source_id=2)))
    else:support=replace(support,stamp=replace(support.stamp,media_time_s=support.stamp.media_time_s-.1))
    with pytest.raises(ValueError,match='exact prepared upright support'):
        select_source_local_observation(((originals[0],.8,None),),upright_support=(support,))
    # Without the exact receipt the same otherwise valid UNKNOWN remains no-vote.
    assert select_source_local_observation(((originals[0],.8,None),))[0] is None
    actor.world.discard(preps[0])


@pytest.mark.parametrize('defect',['covariance','stale-owner','owner-reset'])
def test_upright_r31_typed_support_never_overrides_integrity_or_commit_owner(defect):
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    if defect=='covariance':
        with pytest.raises(WorldPublicationRefused,match='resolver_covariance_invalid'):
            actor.prepare_image({1:[(1,1.,1,1)]},source_changes={1:_r31_upright_changes()|
                {'world_covariance':(.1,1.,0.,1.,.1,0.,0.,0.,.1)}})
    else:
        candidate=actor.prepare_image({1:[(1,1.,1,1)]},source_changes={1:_r31_upright_changes()})
        if defect=='owner-reset':
            harness.before_commit=lambda:actor.identity.reset_source(replace(actor.sources[0],source_epoch=1))
        else:
            original=next(iter(actor.originals.values()))
            advanced=replace(original,stamp=replace(original.stamp,frame_id=3,
                observation_id=original.stamp.observation_id+'-later',acquired_at_s=10.21,media_time_s=.3))
            harness.before_commit=lambda:actor.identity.observe_spatial(advanced,
                SpatialLifecycleReceipt('r31-new-current',advanced.stamp,'ACTIVE'))
        with pytest.raises(WorldPublicationRefused):harness.publish(candidate,2)
    assert actor.world.current_snapshot() is None and not harness.wire and not harness.journal
    assert actor.identity.stats().pending_reservations==0


def test_upright_r31_two_distinct_current_people_remain_two_actual_dots():
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    result=harness.publish(actor.prepare_image({1:[(1,1.,1,1),(2,4.,1,1)]},
        source_changes={1:_r31_upright_changes()}),2)
    assert len(result.entities)==2 and len({e.entity_id for e in result.entities})==2
    assert all(e.support_state=='unknown' for e in result.entities)
    actual=actual_node_wire(actor,harness.wire,'r31-two-distinct-upright')
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert len(actual['snapshots'][0]['people'])==2


@pytest.mark.parametrize('changes',[
    {'world_transform_sha256':'f'*64},
    {'calibration_revision':'different-calibration'},
    {'frame_id':3},
])
def test_upright_r31_wrong_source_frame_binding_cannot_gain_typed_authority(changes):
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    with pytest.raises((WorldPublicationRefused,ValueError)):
        candidate=actor.prepare_image({1:[(1,1.,1,1)]},
            source_changes={1:_r31_upright_changes()|changes})
        harness.publish(candidate,2)
    assert actor.world.current_snapshot() is None and not harness.wire and not harness.journal
    assert actor.identity.stats().pending_reservations==0


def _r29_actual_image_reactivation_wait(*,invalid_covariance=False):
    from reid.mv3dt_person_association import LifecycleRetention,SpatialLifecycleReceipt
    actor=ImageQuarantineActor(live_track_continuity=True);harness=ActualPublicationHarness(actor)
    first=harness.publish(actor.prepare_image({1:[(1,1.,1,1)],2:[(101,1.03,1,1)]},frame=112),112)
    old=next(o for o in actor.originals.values() if o.stamp.member.source_id==2)
    actor.us+=50_000
    # Genuine absence/native grace still needs cross-view rejoin proof.
    # R32 correctly lets a typed-shadow singleton recover source-local current
    # authority; that distinct case must no longer stand in for this wait.
    grace=LifecycleRetention('r29-real-native-grace',old.stamp,old.stamp.member,'inactive',.35)
    event=actor.identity.withdraw(MemberRemoval('r29-absence-F113',old.stamp,'absence',113,
        actor.us/1e6,old.stamp.clock_domain,11.3,retention=grace))
    assert event.disposition=='removed'
    actor.us+=50_000
    preps=[];fresh=[]
    for source,rows in ((1,[(1,1.1,1,1)]),(2,[(101,1.05,1,1)])):
        changes={'world_resolver_confidence':.8}
        if source==2:
            changes.update(world_measurement_accepted=False,world_covariance=(
                (.1,1.,0.,1.,.1,0.,0.,0.,.1) if invalid_covariance else None))
        part,native=actor.normalize({source:rows},frame=114,native_tokens=True,**changes)
        preps.extend(part)
        for original in native.values():
            result=actor.identity.observe_spatial(original,SpatialLifecycleReceipt(
                'r29-native:'+original.stamp.observation_id,original.stamp,'ACTIVE'))
            assert result.disposition=='observed' or (result.disposition=='unresolved'
                and result.reason=='current_cross_source_proof_required'),result
            actor.originals[original.stamp.member]=original;fresh.append(original)
    actor.context=actor.clock_context()
    # Actual current delivery is independent of missing image pair proof.
    # K has a genuine fresh synthetic region; reactivated F has NO fresh region.
    from reid.mv3dt_person_association import (EstimatedImageRegionProof,ImageRegionOriginal,
        ObservedImageRegion,CurrentVisualDeliveryContext,CurrentVisualIngress)
    mono=100.+(actor.us-10_200_000)/1e6
    k=next(o for o in fresh if o.stamp.member.source_id==1)
    camera=next(c for c in actor.image_policy.cameras if c.source.source_id==1)
    region=ImageRegionOriginal(k.stamp,mono,'r29-current-K114',camera.raster_id,camera.prior_id,
        camera.image_size,(45.,25.,10.,25.),
        (ObservedImageRegion('pose_torso_support',(50.,35.),(50.,30.,50.,40.,2.,2.)),))
    actor.image_sequence+=1
    result=actor.identity.join_estimated_image_regions(EstimatedImageRegionProof('r29-no-F-region',
        actor.identity.image_region_active_cohort(),(region,)),at_interval=actor.context,
        delivery_context=CurrentVisualDeliveryContext('synthetic-image-world',actor.image_sequence,mono,
            tuple(CurrentVisualIngress(o.stamp,mono) for o in fresh)))
    assert result.disposition=='observed',result
    members=actor.identity.spatial_memberships(at_interval=actor.context)
    return actor,harness,first,tuple(preps),tuple(fresh),members,event


def test_image_rejoin_r31_real_shadow_singleton_recovers_local_world_without_cross_seed():
    from reid.mv3dt_person_association import SpatialLifecycleUpdate
    actor=ImageQuarantineActor(live_track_continuity=True);harness=ActualPublicationHarness(actor)
    first=harness.publish(actor.prepare_image({2:[(101,1.,1,1)]},frame=110),110,source_id=2)
    old=next(iter(actor.originals.values()));actor.us+=50_000
    event=actor.identity.update_spatial_lifecycle(SpatialLifecycleUpdate('r31-real-shadow',old.stamp,111,
        actor.us/1e6,old.stamp.clock_domain,11.1))
    assert event.disposition=='observed';actor.us+=50_000
    final=harness.publish(actor.prepare_image({2:[(101,1.05,1,1)]},frame=114,
        withdrawn_members=event.withdrawn_members,retired_keys=event.retired_keys),114,source_id=2)
    assert len(final.entities)==1 and final.entities[0].entity_id==first.entities[0].entity_id
    assert final.entities[0].observed_at_us>first.entities[0].observed_at_us
    assert actor.memberships[0].association_basis=='source_local'
    assert all(e.eligible for e in actor.memberships[0].eligibility)
    actual=actual_node_wire(actor,harness.wire,'r31-real-shadow-source-local-recovery')
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert [len(s['people']) for s in actual['snapshots']]==[1,1]


def test_image_rejoin_r29_actual_F114_no_vote_other_person_publishes_then_real_proof_recovers():
    actor,harness,first,preps,fresh,members,event=_r29_actual_image_reactivation_wait()
    waiting=next(o for o in fresh if o.stamp.member.source_id==2)
    raw=actor.tracking_rows[waiting.stamp.observation_id]
    assert raw['world_valid'] is True and raw['world_measurement_accepted'] is False and raw['world_covariance'] is None
    eligibility=next(e for d in members for e in d.eligibility if e.observation==waiting.stamp)
    assert not eligibility.eligible and eligibility.reason=='image_region_rejoin_proof_required'
    candidate=actor.world.prepare_spatial(1,preps,members,at_interval=actor.context,
        withdrawn_members=event.withdrawn_members,retired_keys=event.retired_keys,
        attribution_mode='physical_only_quarantine')
    state=actor.world._prepared_candidates[candidate._candidate_id][1]
    assert len(state.spatial_reservations)==1
    assert state.spatial_reservations[0].reference.owner.source_id==1
    held=harness.publish(candidate,114)
    assert len(held.entities)==1 and held.entities[0].entity_id==first.entities[0].entity_id
    assert [s.camera_id for s in held.entities[0].sources if s.accepted]==['kitchen']
    actor.us+=50_000
    restored=harness.publish(actor.prepare_image({1:[(1,1.2,1,1)],2:[(101,1.23,1,1)]},frame=115,
        source_changes={2:{'world_covariance':tuple(v*.1 for v in COV)}}),115)
    assert all(e.eligible for d in actor.memberships for e in d.eligibility)
    assert len(restored.entities)==1 and restored.entities[0].entity_id==held.entities[0].entity_id
    assert [s.camera_id for s in restored.entities[0].sources if s.accepted]==['family-room']
    assert restored.entities[0].observed_at_us>held.entities[0].observed_at_us
    actual=actual_node_wire(actor,harness.wire,'r29-F114-image-wait-recovery')
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert all(len(s['people'])==1 for s in actual['snapshots'])


def test_image_rejoin_r29_previous_closed_classifier_reproduces_fatal(monkeypatch):
    from noesis_core.world import fusion
    actor,harness,first,preps,fresh,members,event=_r29_actual_image_reactivation_wait()
    # Restore ONLY the previous closed vocabulary; actual owner/World inputs
    # and public lifecycle/reactivation sequence remain exactly the same.
    monkeypatch.setattr(fusion,'SPATIAL_NO_VOTE_REASONS',
        fusion.SPATIAL_NO_VOTE_REASONS-{'image_region_rejoin_proof_required'})
    with pytest.raises(WorldPublicationRefused,match='unexpected_spatial_rejection_reason'):
        actor.world.prepare_spatial(1,preps,members,at_interval=actor.context,
            withdrawn_members=event.withdrawn_members,retired_keys=event.retired_keys,
            attribution_mode='physical_only_quarantine')
    assert actor.world.current_snapshot() is first and actor.identity.stats().pending_reservations==0


@pytest.mark.parametrize('defect',['invalid-covariance','forged-owner','stale-stamp','unknown-reason','legacy-rejoin'])
def test_image_rejoin_r29_integrity_never_hidden_by_expected_wait(defect,monkeypatch):
    actor,harness,first,preps,fresh,members,event=_r29_actual_image_reactivation_wait(
        invalid_covariance=defect=='invalid-covariance')
    if defect=='forged-owner':members=(replace(members[0],association_revision=members[0].association_revision+1),)+members[1:]
    if defect=='stale-stamp':
        from reid.mv3dt_person_association import SpatialLifecycleReceipt
        old=next(o for o in fresh if o.stamp.member.source_id==2)
        advanced=replace(old,stamp=replace(old.stamp,frame_id=115,
            observation_id=old.stamp.observation_id+'-next',acquired_at_s=old.stamp.acquired_at_s+.001,
            media_time_s=11.5))
        assert actor.identity.observe_spatial(advanced,SpatialLifecycleReceipt('real-next-owner',advanced.stamp,'ACTIVE')).disposition=='observed'
    if defect in {'unknown-reason','legacy-rejoin'}:
        # Fault only the actual owner's denial; no positive authority is faked.
        denial='current_rejoin_proof_required' if defect=='legacy-rejoin' else 'unrecognized_image_wait'
        changed=tuple(replace(d,eligibility=tuple(replace(e,reason=denial) if not e.eligible else e for e in d.eligibility)) for d in members)
        monkeypatch.setattr(actor.identity,'spatial_memberships',lambda **kw:changed);members=changed
    before=(len(harness.wire),len(harness.journal),actor.world._revision)
    with pytest.raises(WorldPublicationRefused):
        actor.world.prepare_spatial(1,preps,members,at_interval=actor.context,
            withdrawn_members=event.withdrawn_members,retired_keys=event.retired_keys,
            attribution_mode='physical_only_quarantine')
    assert actor.world.current_snapshot() is first
    assert before==(len(harness.wire),len(harness.journal),actor.world._revision)
    assert actor.identity.stats().pending_reservations==0


def test_attribution_r23_valid_split_attenuates_labels_with_real_current_grant_and_retained_custody():
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','synthetic-k',1004,1),
        (2,101):HouseholdDescriptor('visitor','synthetic-f',1000,1)}
    first=harness.publish(actor.prepare_image({1:[(1,1.,1,1)],2:[(101,1.03,1,1)]},households=houses),2)
    old=first.entities[0]
    actor.us+=100_000
    candidate=actor.prepare_image({1:[(1,1.1,1,1)]},frame=3,households=houses,image_y={(1,1):80.})
    assert all(d.household_conflict is None for d in actor.memberships)
    # Genuine current source-local physical capability, NOT forged household
    # conflict/reservation/purpose. World withholds its settled label only.
    state=actor.world._prepared_candidates[candidate._candidate_id][1]
    assert len(state.spatial_reservations)==1 and state.spatial_reservations[0].purpose=='strict'
    second=harness.publish(candidate,3)
    assert len(second.entities)==2
    kept=next(e for e in second.entities if e.entity_id==old.entity_id)
    fresh=next(e for e in second.entities if e.entity_id!=old.entity_id)
    assert (kept.position,kept.observed_at_us,kept.stale_after_us)==(old.position,old.observed_at_us,old.stale_after_us)
    assert fresh.observed_at_us>kept.observed_at_us and fresh.position.x==1.1
    q=kept.attribution_quarantine
    assert q==fresh.attribution_quarantine and len(q.component_entity_ids)==2
    assert len(q.owners)==3 and q.stable_ids==(1000,1004)
    assert all(e.subject.kind=='provisional' and e.subject.stable_id is None for e in second.entities)
    assert {(c.physical_serial,c.source_id,c.frame_id) for c in q.owners}=={
        (int(old.entity_id.rsplit(':',1)[-1]),1,2),(int(old.entity_id.rsplit(':',1)[-1]),2,2),
        (int(fresh.entity_id.rsplit(':',1)[-1]),1,3)}
    actor.us+=400_000
    third=harness.publish(actor.prepare_image({1:[(1,1.2,1,1)]},frame=4,households=houses,
        image_y={(1,1):80.}),4)
    still=next(e for e in third.entities if e.entity_id==old.entity_id)
    assert (still.position,still.observed_at_us,still.stale_after_us)==(old.position,old.observed_at_us,old.stale_after_us)
    assert still.attribution_quarantine is not None  # .35 vote expiry != 2s position custody
    actual=actual_node_wire(actor,harness.wire,'r23-valid-split-label-attenuation')
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert [len(s['people']) for s in actual['snapshots']]==[1,2,2]
    assert all(p['stableId'] is None for p in actual['snapshots'][-1]['people'])


@pytest.mark.parametrize('defect',['strict-mode','owner-denies','queued-reset','uncovered-claim'])
def test_attribution_r23_attenuation_cannot_mint_or_replace_physical_authority(defect,monkeypatch):
    from noesis.identity_association_service import SpatialMemberResult
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','synthetic-k',1004,1),
        (2,101):HouseholdDescriptor('visitor','synthetic-f',1000,1)}
    first=harness.publish(actor.prepare_image({1:[(1,1.,1,1)],2:[(101,1.03,1,1)]},households=houses),2)
    actor.us+=100_000
    if defect=='owner-denies':
        monkeypatch.setattr(actor.identity,'prepare_spatial_member',lambda *a,**k:SpatialMemberResult('denied','synthetic-current-authority-denied'))
    if defect=='uncovered-claim':
        actual=GlobalWorldFusion.spatial_attribution_claims
        def corrupted(fusion):
            rows=actual(fusion)
            if fusion is not actor.world._fusion:
                entity=first.entities[0].entity_id
                original=rows[entity][0]
                rows[entity]=rows[entity]+(original.model_copy(update={'raw_id':99,'observation_id':'fabricated-unowned-original'}),)
            return rows
        monkeypatch.setattr(GlobalWorldFusion,'spatial_attribution_claims',corrupted)
    before=(len(harness.wire),len(harness.journal),actor.world._revision)
    if defect=='queued-reset':
        candidate=actor.prepare_image({1:[(1,1.1,1,1)]},frame=3,households=houses,image_y={(1,1):80.})
        harness.before_commit=lambda:actor.identity.reset_source(replace(actor.sources[1],source_epoch=1))
        with pytest.raises(WorldPublicationRefused):harness.publish(candidate,3)
    else:
        with pytest.raises(WorldPublicationRefused):
            actor.prepare_image({1:[(1,1.1,1,1)]},frame=3,households=houses,image_y={(1,1):80.},
                mode='strict' if defect=='strict-mode' else 'physical_only_quarantine')
    assert actor.world.current_snapshot() is first
    assert before==(len(harness.wire),len(harness.journal),actor.world._revision)
    assert actor.identity.stats().pending_reservations==0


@pytest.mark.parametrize('barrier',['expiry','termination','reset'])
def test_attribution_r23_original_lifecycle_releases_derived_label_overlay(barrier):
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','synthetic-k',1004,1),
        (2,101):HouseholdDescriptor('visitor','synthetic-f',1000,1)}
    harness.publish(actor.prepare_image({1:[(1,1.,1,1)],2:[(101,1.03,1,1)]},households=houses),2)
    actor.us+=100_000
    second=harness.publish(actor.prepare_image({1:[(1,1.1,1,1)]},frame=3,
        households=houses,image_y={(1,1):80.}),3)
    fresh=next(e for e in second.entities if any(s.camera_id=='kitchen' and s.accepted for s in e.sources))
    withdrawn=();retired=()
    if barrier=='expiry':actor.us+=2_100_000
    else:
        actor.us+=100_000
        if barrier=='reset':
            binding=replace(actor.sources[1],source_epoch=1)
            result=actor.identity.reset_source(binding)
            actor.sources=(actor.sources[0],binding)
        else:
            old=next(o for o in actor.originals.values() if o.stamp.member.source_id==2)
            result=actor.identity.withdraw(MemberRemoval('synthetic-termination',old.stamp,'termination',4,
                actor.us/1e6,old.stamp.clock_domain,.4))
        withdrawn=result.withdrawn_members;retired=result.retired_keys
    preps,_=actor.normalize({1:[]},frame=30 if barrier=='expiry' else 4,native_tokens=True,world_resolver_confidence=.8)
    actor.context=actor.clock_context()
    candidate=actor.world.prepare_spatial(1,tuple(preps),actor.identity.spatial_memberships(at_interval=actor.context),
        at_interval=actor.context,withdrawn_members=withdrawn,retired_keys=retired,attribution_mode='physical_only_quarantine')
    final=harness.publish(candidate,30 if barrier=='expiry' else 4)
    if barrier=='expiry':assert not final.entities
    else:
        assert len(final.entities)==1 and final.entities[0].entity_id==fresh.entity_id
        assert final.entities[0].attribution_quarantine is None and final.entities[0].subject.stable_id==1004
        assert (final.entities[0].position,final.entities[0].observed_at_us,final.entities[0].stale_after_us)==(fresh.position,fresh.observed_at_us,fresh.stale_after_us)
    actual=actual_node_wire(actor,harness.wire,'r23-lifecycle-'+barrier)
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert [len(s['people']) for s in actual['snapshots']]==[1,2,0 if barrier=='expiry' else 1]


@pytest.mark.parametrize('continuation',[False,True])
def test_bestview_r22_image_actual_fifo_retains_better_original_and_reverses(continuation):
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','synthetic-k',1004,1),
        (2,101):HouseholdDescriptor('visitor','synthetic-f',1000,1)}
    first=harness.publish(actor.prepare_image({2:[(101,1.,1,1)]},households=houses,
        source_changes={2:{'world_covariance':tuple(v*.5 for v in COV)}}),2,source_id=2)
    old=first.entities[0];before=actor.identity.stats().spatial_member_frontiers
    actor.us+=100_000
    second=harness.publish(actor.prepare_image({1:[(1,1.1,1,1)]},frame=3,households=houses,
        source_changes={1:{'world_covariance':tuple(v*4 for v in COV)}}),3)
    kept=second.entities[0]
    assert kept.entity_id==old.entity_id
    assert (kept.position,kept.observed_at_us,kept.stale_after_us,kept.continuity_segment_id)==(old.position,old.observed_at_us,old.stale_after_us,old.continuity_segment_id)
    assert actor.identity.stats().spatial_member_frontiers==before
    assert [s.camera_id for s in kept.sources if s.accepted]==['family-room']
    assert kept.attribution_quarantine is not None and kept.subject.stable_id is None
    next_frame=4
    if continuation:
        actor.us+=50_000;next_frame=5
        continued=harness.publish(actor.prepare_image({1:[(1,1.15,1,1)]},frame=4,households=houses,
            source_changes={1:{'world_covariance':tuple(v*4 for v in COV)}},omit_image_current=True),4)
        assert actor.memberships[0].association_basis=='estimated_image_region_continuation'
        assert continued.entities[0].position==old.position
        assert actor.identity.stats().spatial_member_frontiers==before
    actor.us+=100_000
    reversed=harness.publish(actor.prepare_image({1:[(1,1.2,1,1)]},frame=next_frame,households=houses,
        source_changes={1:{'world_covariance':tuple(v*.1 for v in COV)}}),next_frame)
    better=reversed.entities[0]
    assert better.entity_id==old.entity_id and better.observed_at_us>old.observed_at_us
    assert [s.camera_id for s in better.sources if s.accepted]==['kitchen']
    actual=actual_node_wire(actor,harness.wire,'r22-current-image-quality')
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert all(len(s['people'])==1 for s in actual['snapshots'])


@pytest.mark.parametrize('defect',['changed-owner','active-frame','provisional'])
def test_bestview_r22_uncovered_old_original_cannot_win_on_covariance_alone(defect):
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','synthetic-k',1004,1),
        (2,101):HouseholdDescriptor('visitor','synthetic-f',1000,1)}
    if defect=='provisional':houses[(2,101)]=HouseholdDescriptor()
    first=harness.publish(actor.prepare_image({2:[(101,1.,1,1)]},households=houses,
        source_changes={2:{'world_covariance':tuple(v*.5 for v in COV)}}),2,source_id=2)
    if defect in {'advanced-stamp','changed-owner'}:
        actor.us+=50_000
        if defect=='changed-owner':houses[(2,101)]=HouseholdDescriptor('visitor','synthetic-new-f',1009,2)
        harness.publish(actor.prepare_image({2:[(101,1.01,1,1)]},frame=3,households=houses,
            source_changes={2:{'world_covariance':None}}),3,source_id=2)
    elif defect=='active-frame':
        actor.frames[2]=actor.frames[2].model_copy(update={'target_from_source_sha256':'f'*64})
    actor.us+=50_000
    second=harness.publish(actor.prepare_image({1:[(1,1.1,1,1)]},frame=4,households=houses,
        source_changes={1:{'world_covariance':tuple(v*4 for v in COV)}}),4)
    head=second.entities[0]
    assert [s.camera_id for s in head.sources if s.accepted]==['kitchen']
    assert head.observed_at_us>first.entities[0].observed_at_us


@pytest.mark.parametrize('advances',[1,3])
def test_bestview_r56_committed_hold_after_current_camera_has_no_position(advances):
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','r56-k',1004,1),
        (2,101):HouseholdDescriptor('visitor','r56-f',1000,1)}
    first=harness.publish(actor.prepare_image({2:[(101,1.,1,1)]},households=houses,
        source_changes={2:{'world_covariance':tuple(v*.5 for v in COV)}}),2,source_id=2)
    old=first.entities[0]
    for frame in range(3,3+advances):
        actor.us+=150_000
        harness.publish(actor.prepare_image({2:[(101,1.01,1,1)]},frame=frame,households=houses,
            source_changes={2:{'world_covariance':None}}),frame,source_id=2)
    actor.us+=50_000
    frame=3+advances
    candidate=actor.prepare_image({1:[(1,1.1,1,1)]},frame=frame,households=houses,
        source_changes={1:{'world_covariance':tuple(v*4 for v in COV)}})
    before=actor.identity.stats().spatial_member_frontiers
    held=harness.publish(candidate,frame).entities[0]
    assert held.position==old.position and held.position_state=='held'
    assert (held.observed_at_us,held.stale_after_us,held.continuity_segment_id,held.covariance)==(
        old.observed_at_us,old.stale_after_us,old.continuity_segment_id,old.covariance)
    assert held.attribution_quarantine and held.subject.stable_id is None
    assert actor.identity.stats().spatial_member_frontiers==before+1  # K ONLY, no old F replay.
    actor.us+=50_000
    fresh=harness.publish(actor.prepare_image({1:[(1,1.2,1,1)]},frame=frame+1,households=houses,
        source_changes={1:{'world_covariance':tuple(v*.1 for v in COV)}}),frame+1).entities[0]
    assert fresh.position_state=='measured' and fresh.observed_at_us>old.observed_at_us
    assert [s.camera_id for s in fresh.sources if s.accepted]==['kitchen']
    node=actual_node_wire(actor,harness.wire,'r56-committed-hold')
    assert all(d['accepted'] for d in node['diagnostics'])
    assert all(len(s['people'])==1 for s in node['snapshots'])


def _r62_actor(*,enabled=True):
    from reid.mv3dt_person_association import ImageRegionOwnerAvailability
    actor=ImageQuarantineActor(live_track_continuity=True,contradictory_cue_quarantine=enabled)
    actor.mixed=False;actor.all_false=False;actor.alternative=False
    original_join=actor.identity.join_estimated_image_regions
    def join(proof,**kwargs):
        changed=[];caps=[]
        for o in proof.originals:
            k=o.stamp.member
            patches=o.patches
            if actor.mixed and k.source_id==2 and k.raw_id==101:
                patches=tuple(replace(p,anchor_uv=(p.anchor_uv[0],80.),
                    capsule=(p.capsule[0],80.,p.capsule[2],80.,p.capsule[4],p.capsule[5]))
                    if actor.all_false or p.anatomy=='pose_shoulder_support' else p for p in patches)
            if actor.alternative and k.source_id==2 and k.raw_id==102:
                patches=tuple(replace(p,anchor_uv=(p.anchor_uv[0],30.),
                    capsule=(p.capsule[0],30.,p.capsule[2],40. if p.anatomy=='pose_torso_support' else 30.,
                             p.capsule[4],p.capsule[5])) for p in patches)
            changed.append(replace(o,bbox_xywh=(o.bbox_xywh[0],20.,o.bbox_xywh[2],100.),patches=patches))
            members=tuple(p.stamp.member for p in proof.originals if p.stamp.member.source_id==k.source_id)
            caps.append(ImageRegionOwnerAvailability(o.stamp,'r62-graph',o.stamp.frame_id,
                o.stamp.media_time_s,1_000_000+o.stamp.frame_id,o.ingress_monotonic_s,
                350_000_000,members,members,(),(),o.ingress_monotonic_s,0.,20.,'running'))
        return original_join(replace(proof,originals=tuple(changed),owner_availability=tuple(caps)),**kwargs)
    actor.identity.join_estimated_image_regions=join
    houses={(1,1):HouseholdDescriptor('visitor','r62-k',1001,1),
        (2,101):HouseholdDescriptor('visitor','r62-f',1000,1),
        (2,102):HouseholdDescriptor('visitor','r62-third',1002,1)}
    return actor,ActualPublicationHarness(actor),houses


def test_r62_contradictory_custody_actual_world_publisher_node_held_recovery():
    actor,harness,houses=_r62_actor()
    rows={1:[(1,1.,1,1)],2:[(101,1.03,1,1),(102,4.,1,1)]}
    first=harness.publish(actor.prepare_image(rows,households=houses),2)
    old=next(e for e in first.entities if e.attribution_quarantine is not None)
    other=next(e for e in first.entities if e.entity_id!=old.entity_id)
    frontiers=actor.identity.stats().spatial_member_frontiers
    actor.us+=100_000;actor.mixed=True
    # Wrong current coordinates/covariance cannot win a new physical grant.
    changed={1:[(1,8.,1,1)],2:[(101,9.,1,1),(102,4.05,1,1)]}
    candidate=actor.prepare_image(changed,frame=3,households=houses)
    pair=next(d for d in actor.memberships if len(d.members)==2)
    assert pair.association_basis=='estimated_image_region_conflict_custody'
    assert all(not e.eligible and e.reason=='image_region_contradictory_cue_member' for e in pair.eligibility)
    state=actor.world._prepared_candidates[candidate._candidate_id][1]
    assert len(state.spatial_reservations)==1 and state.spatial_reservations[0].reference.owner.raw_id==102
    second=harness.publish(candidate,3)
    kept=next(e for e in second.entities if e.entity_id==old.entity_id)
    assert kept.position_state=='held'
    assert (kept.position,kept.observed_at_us,kept.stale_after_us,kept.covariance,kept.continuity_segment_id)==(
        old.position,old.observed_at_us,old.stale_after_us,old.covariance,old.continuity_segment_id)
    assert kept.subject.stable_id is None and kept.attribution_quarantine is not None
    assert {c.household_stable_id for c in kept.attribution_quarantine.retained_owners}>={1000,1001}
    assert next(e for e in second.entities if e.entity_id==other.entity_id).position.x==4.05
    assert actor.identity.stats().spatial_member_frontiers==frontiers  # Existing third frontier advances; pair adds none.
    actor.us+=100_000;actor.mixed=False
    third=harness.publish(actor.prepare_image(rows,frame=4,households=houses),4)
    recovered=next(e for e in third.entities if e.entity_id==old.entity_id)
    assert recovered.position_state=='measured' and recovered.observed_at_us>old.observed_at_us
    node=actual_node_wire(actor,harness.wire,'r62-conflict-held-recovery')
    assert [len(s['people']) for s in node['snapshots']]==[2,2,2]
    assert all(d['accepted'] for d in node['diagnostics'])


@pytest.mark.parametrize('barrier',['expiry','queued-observe','queued-reset','queued-stop','binding'])
def test_r62_custody_commit_never_renews_or_ignores_owner_barrier(barrier):
    actor,harness,houses=_r62_actor();rows={1:[(1,1.,1,1)],2:[(101,1.03,1,1)]}
    first=harness.publish(actor.prepare_image(rows,households=houses),2)
    actor.us+=100_000;actor.mixed=True
    candidate=actor.prepare_image(rows,frame=3,households=houses)
    before=(len(harness.wire),len(harness.journal),actor.world._revision)
    if barrier=='expiry':harness.before_commit=lambda:setattr(actor,'us',actor.us+2_100_000)
    elif barrier=='queued-reset':harness.before_commit=lambda:actor.identity.reset_source(replace(actor.sources[0],source_epoch=1))
    elif barrier=='queued-stop':harness.before_commit=lambda:actor.identity.stop_image_region_source(actor.sources[0],graph_id='r62-graph')
    elif barrier=='binding':harness.before_commit=lambda:actor.frames.__setitem__(1,actor.frames[1].model_copy(update={'target_from_source_sha256':'f'*64}))
    else:
        old=next(iter(actor.originals.values()))
        later=replace(old,stamp=replace(old.stamp,observation_id='r62-later-current',frame_id=4,
            acquired_at_s=old.stamp.acquired_at_s+.01,media_time_s=.4))
        harness.before_commit=lambda:actor.identity.observe_spatial(later,SpatialLifecycleReceipt('r62-later',later.stamp,'ACTIVE'))
    with pytest.raises(WorldPublicationRefused):harness.publish(candidate,3)
    assert actor.world.current_snapshot() is first
    assert before==(len(harness.wire),len(harness.journal),actor.world._revision)
    assert actor.identity.stats().pending_reservations==0


def test_r62_repeated_mixed_custody_expires_original_then_unique_recovers():
    actor,harness,houses=_r62_actor();rows={1:[(1,1.,1,1)],2:[(101,1.03,1,1)]}
    first=harness.publish(actor.prepare_image(rows,households=houses),2)
    actor.mixed=True
    for frame in range(3,25):
        actor.us+=100_000
        snapshot=harness.publish(actor.prepare_image(rows,frame=frame,households=houses),frame)
        if actor.us<first.entities[0].stale_after_us:
            assert snapshot.entities[0].observed_at_us==first.entities[0].observed_at_us
            assert snapshot.entities[0].position_state=='held'
        else:assert not snapshot.entities
    actor.us+=100_000;actor.mixed=False
    recovered=harness.publish(actor.prepare_image(rows,frame=25,households=houses),25)
    assert len(recovered.entities)==1 and recovered.entities[0].observed_at_us==actor.us
    node=actual_node_wire(actor,harness.wire,'r62-original-expiry')
    assert not node['snapshots'][-2]['people'] and len(node['snapshots'][-1]['people'])==1


@pytest.mark.parametrize('case',['default','unseeded','all-false','alternative-true'])
def test_r62_conflict_custody_never_bootstraps_or_overrides_strict_revocation(case):
    actor,harness,houses=_r62_actor(enabled=case!='default')
    rows={1:[(1,1.,1,1)],2:[(101,1.03,1,1),(102,4.,1,1)]}
    if case!='unseeded':harness.publish(actor.prepare_image(rows,households=houses),2)
    actor.us+=100_000;actor.mixed=True
    actor.all_false=case=='all-false';actor.alternative=case=='alternative-true'
    candidate=actor.prepare_image(rows,frame=3,households=houses)
    assert all(d.association_basis!='estimated_image_region_conflict_custody' for d in actor.memberships)
    actor.world.discard(candidate)


def test_r62_only_exact_new_reason_is_no_vote():
    from noesis_core.world.fusion import spatial_rejection_category
    assert spatial_rejection_category('image_region_contradictory_cue_member')=='no_vote'
    for code in ('image_region_contradictory_cue_member_typo','current_rejoin_proof_required',
                 'resolver_covariance_invalid','spatial_original_member_context'):
        assert spatial_rejection_category(code)=='integrity'


def test_position_only_queued_pair_actual_world_node_preserves_original_expiry():
    actor,harness,houses=_r62_actor(enabled=False)
    rows={1:[(1,1.,1,1)],2:[(101,1.03,1,1)]}
    first=harness.publish(actor.prepare_image(rows,households=houses),2)
    old=first.entities[0]
    original_join=actor.identity.join_estimated_image_regions
    def delayed(proof,**kwargs):
        delivery=kwargs['delivery_context']
        kwargs['delivery_context']=replace(delivery,evaluation_monotonic_s=delivery.evaluation_monotonic_s+.5)
        actor.us+=500_000  # Actual simulated receipt delay; originals stay immutable.
        return original_join(proof,**kwargs)
    actor.identity.join_estimated_image_regions=delayed
    actor.us+=100_000
    second=harness.publish(actor.prepare_image(rows,frame=3,households=houses),3)
    current,=second.entities
    assert current.entity_id==old.entity_id and current.observed_at_us>old.observed_at_us
    assert current.observed_at_us==actor.us-500_000
    assert current.stale_after_us==current.observed_at_us+2_000_000
    assert current.position_state=='measured' and current.subject.stable_id is None
    assert current.attribution_quarantine is not None
    assert all(e.eligible for d in actor.memberships for e in d.eligibility)
    assert all(actor.identity.ledger.current_visual_liveness_error(m.observation)=='current_visual_stale_member_delivery'
        for d in actor.memberships for m in d.members)
    actor.us+=1_600_000
    expired=harness.publish(_r40_empty_spatial(actor,frame=4,evaluation_s=10.4),4)
    assert not expired.entities  # Old original was not retimed to publication.
    actual=actual_node_wire(actor,harness.wire,'position-only-queued-pair-expiry')
    assert [len(s['people']) for s in actual['snapshots']]==[1,1,0]
    assert all(d['accepted'] for d in actual['diagnostics'])


def test_r62_seed_without_committed_world_original_cannot_create_held_point():
    actor,harness,houses=_r62_actor();rows={1:[(1,1.,1,1)],2:[(101,1.03,1,1)]}
    uncommitted=actor.prepare_image(rows,households=houses)
    actor.world.discard(uncommitted)
    assert actor.world.current_snapshot() is None
    actor.us+=100_000;actor.mixed=True
    candidate=actor.prepare_image(rows,frame=3,households=houses)
    assert actor.memberships[0].association_basis=='estimated_image_region_conflict_custody'
    assert not actor.world._prepared_candidates[candidate._candidate_id][1].spatial_reservations
    assert not harness.publish(candidate,3).entities
    node=actual_node_wire(actor,harness.wire,'r62-no-committed-original')
    assert all(not s['people'] for s in node['snapshots'])


def test_r62_invalid_current_covariance_remains_atomic_integrity_not_custody():
    actor,harness,houses=_r62_actor();rows={1:[(1,1.,1,1)],2:[(101,1.03,1,1)]}
    first=harness.publish(actor.prepare_image(rows,households=houses),2)
    actor.us+=100_000;actor.mixed=True
    before=(len(harness.wire),len(harness.journal),actor.world._revision)
    with pytest.raises(WorldPublicationRefused,match='resolver_covariance_invalid'):
        actor.prepare_image(rows,frame=3,households=houses,
            source_changes={1:{'world_covariance':(.1,1.,0.,1.,.1,0.,0.,0.,.1)}})
    assert actor.world.current_snapshot() is first
    assert before==(len(harness.wire),len(harness.journal),actor.world._revision)


def test_r62_causal_previous_consumer_vocabulary_refuses_real_owner_custody(monkeypatch):
    import noesis_core.world.fusion as fusion
    import noesis_core.world_service as service
    actor,harness,houses=_r62_actor();rows={1:[(1,1.,1,1)],2:[(101,1.03,1,1)]}
    first=harness.publish(actor.prepare_image(rows,households=houses),2)
    actor.us+=100_000;actor.mixed=True
    previous=fusion.SPATIAL_NO_VOTE_REASONS-{'image_region_contradictory_cue_member'}
    monkeypatch.setattr(fusion,'SPATIAL_NO_VOTE_REASONS',previous)
    monkeypatch.setattr(service,'SPATIAL_NO_VOTE_REASONS',previous)
    with pytest.raises(WorldPublicationRefused,match='unexpected_spatial_rejection_reason'):
        actor.prepare_image(rows,frame=3,households=houses)
    assert actor.world.current_snapshot() is first and len(harness.journal)==1


@pytest.mark.parametrize('barrier',['reset','stop','correction','expiry','binding'])
def test_bestview_r56_queued_hold_revalidates_current_owner_and_original_deadline(barrier):
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','r56-k',1004,1),
        (2,101):HouseholdDescriptor('visitor','r56-f',1000,1)}
    first=harness.publish(actor.prepare_image({2:[(101,1.,1,1)]},households=houses,
        source_changes={2:{'world_covariance':tuple(v*.5 for v in COV)}}),2,source_id=2)
    actor.us+=50_000
    harness.publish(actor.prepare_image({2:[(101,1.01,1,1)]},frame=3,households=houses,
        source_changes={2:{'world_covariance':None}}),3,source_id=2)
    actor.us+=50_000
    candidate=actor.prepare_image({1:[(1,1.1,1,1)]},frame=4,households=houses,
        source_changes={1:{'world_covariance':tuple(v*4 for v in COV)}})
    before=(actor.world.current_snapshot(),len(harness.wire),len(harness.journal),actor.world._revision)
    def changed():
        if barrier=='reset':actor.identity.reset_source(replace(actor.sources[1],source_epoch=1))
        elif barrier=='stop':actor.identity.stop_image_region_source(actor.sources[1],graph_id='r56-real-stop')
        elif barrier=='expiry':actor.us=first.entities[0].stale_after_us+1
        elif barrier=='binding':actor.frames[2]=actor.frames[2].model_copy(update={'target_from_source_sha256':'f'*64})
        else:
            # A queued OWNER update, not an illegal later World preparation.
            old=next(o for o in actor.originals.values() if o.stamp.member.source_id==2)
            stamp=replace(old.stamp,observation_id='r56-corrected-original',frame_id=5,
                acquired_at_s=old.stamp.acquired_at_s+.01,media_time_s=old.stamp.media_time_s+.01)
            current=SourceObservation(stamp,HouseholdDescriptor('visitor','r56-corrected',1009,2))
            revision=actor.identity.stats().association_revision
            result=actor.identity.observe_spatial(current,SpatialLifecycleReceipt('r56-correction',stamp,'ACTIVE'))
            assert result.reason=='current_cross_source_proof_required'
            assert actor.identity.stats().association_revision>revision
    harness.before_commit=changed
    with pytest.raises((WorldPublicationRefused,ValueError)):
        harness.publish(candidate,4)
    assert before==(actor.world.current_snapshot(),len(harness.wire),len(harness.journal),actor.world._revision)


def test_bestview_r56_hold_preserves_distinct_person_and_same_source_fresh_recovery():
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','r56-k',1004,1),
        (2,101):HouseholdDescriptor('visitor','r56-f',1000,1),
        (2,102):HouseholdDescriptor('visitor','r56-other',1002,1)}
    first=harness.publish(actor.prepare_image({2:[(101,1.,1,1),(102,4.,1,1)]},households=houses,
        source_changes={2:{'world_covariance':tuple(v*.5 for v in COV)}}),2,source_id=2)
    old=next(e for e in first.entities if e.subject.stable_id==1000)
    other=next(e for e in first.entities if e.subject.stable_id==1002)
    actor.us+=50_000
    harness.publish(actor.prepare_image({2:[(101,1.01,1,1),(102,4.,1,1)]},frame=3,households=houses,
        source_changes={2:{'world_covariance':None}}),3,source_id=2)
    actor.us+=50_000
    held=harness.publish(actor.prepare_image({1:[(1,1.1,1,1)]},frame=4,households=houses,
        source_changes={1:{'world_covariance':tuple(v*4 for v in COV)}}),4)
    assert len(held.entities)==2
    kept=next(e for e in held.entities if e.entity_id==old.entity_id)
    untouched=next(e for e in held.entities if e.entity_id==other.entity_id)
    assert kept.position_state=='held' and kept.position==old.position
    assert (untouched.position,untouched.observed_at_us,untouched.stale_after_us)==(
        other.position,other.observed_at_us,other.stale_after_us)
    actor.us+=50_000
    fresh=harness.publish(actor.prepare_image({2:[(101,1.03,1,1),(102,4.01,1,1)]},frame=5,households=houses),5,source_id=2)
    renewed=next(e for e in fresh.entities if e.entity_id==old.entity_id)
    assert renewed.position_state=='measured' and renewed.observed_at_us>old.observed_at_us
    node=actual_node_wire(actor,harness.wire,'r56-distinct-person')
    assert all(len(s['people'])==2 for s in node['snapshots'])


def test_bestview_r22_anonymous_equal_original_descriptors_need_no_settled_identity():
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    first=harness.publish(actor.prepare_image({2:[(101,1.,1,1)]},
        source_changes={2:{'world_covariance':tuple(v*.5 for v in COV)}}),2,source_id=2)
    actor.us+=100_000
    second=harness.publish(actor.prepare_image({1:[(1,1.1,1,1)]},frame=3,
        source_changes={1:{'world_covariance':tuple(v*4 for v in COV)}}),3)
    assert second.entities[0].position==first.entities[0].position
    assert second.entities[0].observed_at_us==first.entities[0].observed_at_us
    assert second.entities[0].subject.stable_id is None
    assert second.entities[0].attribution_quarantine is None


def test_bestview_r22_camera_name_tie_is_not_better_measurement_quality():
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    harness.publish(actor.prepare_image({2:[(101,1.,1,1)]}),2,source_id=2)
    actor.us+=100_000
    second=harness.publish(actor.prepare_image({1:[(1,1.1,1,1)]},frame=3),3)
    assert [s.camera_id for s in second.entities[0].sources if s.accepted]==['kitchen']


def test_bestview_r22_expired_image_support_never_renews_old_position_or_grant():
    from reid.mv3dt_person_association import EstimatedImageRegionProof,CurrentVisualDeliveryContext,CurrentVisualIngress
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    first=harness.publish(actor.prepare_image({1:[(1,1.1,1,1)],2:[(101,1.,1,1)]},
        source_changes={1:{'world_covariance':tuple(v*4 for v in COV)},2:{'world_covariance':tuple(v*.5 for v in COV)}}),2)
    before=actor.identity.stats().spatial_member_frontiers
    actor.us+=400_000
    # Expired proof is not reissued/restamped. Current source advances through
    # public observation while the owner correctly denies that old image group.
    preps,native=actor.normalize({1:[(1,1.2,1,1)]},frame=3,native_tokens=True,
        world_resolver_confidence=.8,world_covariance=tuple(v*4 for v in COV))
    for o in native.values():
        actor.identity.observe_spatial(o,SpatialLifecycleReceipt('current:'+o.stamp.observation_id,o.stamp,'ACTIVE'))
        actor.originals[o.stamp.member]=o
    actor.context=actor.clock_context()
    originals=tuple(actor.originals.values())
    current=tuple(native.values())
    actor.identity.join_estimated_image_regions(EstimatedImageRegionProof('expired-support',
        tuple(o.stamp for o in originals),()),at_interval=actor.context,
        delivery_context=CurrentVisualDeliveryContext('synthetic-image-world',2,100.4,
            tuple(CurrentVisualIngress(o.stamp,100.4 if o in current else 100.) for o in originals)))
    memberships=actor.identity.spatial_memberships(at_interval=actor.context)
    candidate=actor.world.prepare_spatial(1,tuple(preps),memberships,at_interval=actor.context,
        attribution_mode='physical_only_quarantine')
    second=harness.publish(candidate,3)
    held=next(e for e in second.entities if e.entity_id==first.entities[0].entity_id)
    assert (held.position,held.observed_at_us,held.stale_after_us)==(first.entities[0].position,
        first.entities[0].observed_at_us,first.entities[0].stale_after_us)
    # The expired anonymous association is revoked, not kept alive to retain
    # its attractive covariance. A genuine current source-local point may enter
    # under a separate key; it cannot renew the old peer original.
    assert len(second.entities)==2 and len(memberships)==2
    fresh=next(e for e in second.entities if e.entity_id!=held.entity_id)
    assert [s.camera_id for s in fresh.sources if s.accepted]==['kitchen']
    assert fresh.observed_at_us>held.observed_at_us


class ActualPublicationHarness:
    """REAL publisher + actual original World receipts, synthetic outbound gate."""

    def __init__(self, actor):
        self.actor, self.wire, self.failures, self.journal = actor, [], [], []
        self.candidate = None
        self.prepare_failure = None
        self.before_commit = None
        self.admissions = 0
        harness = self

        class Journal:
            max_records = 256

            def append_many(self, payloads, *, recorded_at_us):
                harness.journal.append(json.loads(json.dumps(payloads)))
                return tuple(payloads)

        class Ingress:
            def prepare(self, source, tracks, *, metadata):
                if harness.prepare_failure is not None:
                    raise harness.prepare_failure
                assert harness.candidate is not None
                return harness.candidate

            def commit(self, receipt):
                if harness.before_commit is not None:
                    harness.before_commit()
                return actor.commit_world(receipt)

            def discard(self, receipt):
                actor.world.discard(receipt)

        class Outbound:
            def admit_broadcast_batch_sync(self, messages):
                harness.admissions += 1
                frozen = json.loads(json.dumps(messages, allow_nan=False))

                def commit_then_release(commit):
                    publication = commit()
                    assert actor.world.current_snapshot() is publication.snapshot
                    harness.wire.extend(frozen)
                    return publication

                return SimpleNamespace(receipt=SimpleNamespace(submission_id=harness.admissions,
                    message_count=len(frozen)), commit_then_release=commit_then_release)

        actor.world._journal = Journal()  # Test-owned synchronous journal boundary, no I/O.
        self.publisher = TrackingTelemetryPublisher(Outbound(), world_service=Ingress(), failure_callback=self.failures.append)

    def publish(self, candidate, frame, *, source_id=1):
        self.candidate = candidate
        observations = candidate.publication.observations if candidate is not None else ()
        rows = [dict(self.actor.tracking_rows[o.observation_id]) for o in observations]
        epoch = next(s.source_epoch for s in self.actor.sources if s.source_id == source_id)
        self.publisher.publish(source_id, rows, frame_metadata={"frame_id": frame, "observed_at_us": self.actor.us,
                                                       "source_epoch": epoch})
        return self.actor.world.current_snapshot()


@pytest.mark.parametrize('first_source',[1,2])
def test_r123_actual_contact_producer_world_publisher_node_quality_reversal(first_source):
    """Synthetic pose/calibration; actual hook→PGS→owner→World→publisher→Node.

    Camera arrival order must not turn an unobserved furniture bbox into feet.
    """
    from DS9.tests.test_world_resolver_hook_integration import (
        _r123_actual_contact_processor,_r123_actual_contact_row)
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    producers={s:_r123_actual_contact_processor(source=s,camera=CAMERAS[s],binding=actor.frames[s]) for s in (1,2)}
    def change(source,frame,contact):
        p,c=producers[source]
        row=_r123_actual_contact_row(p,c,source=source,camera=CAMERAS[source],
            raw=1 if source==1 else 101,frame=frame,contact=contact)
        assert 'world_estimation_error' not in row
        fields={k:v for k,v in row.items() if k.startswith('world_') or k=='world'}
        fields['world']=row.get('world')
        fields['world_covariance']=row.get('world_covariance')
        fields['world_support_surface']=None
        return fields
    raw={1:1,2:101};houses={(1,1):HouseholdDescriptor('visitor','r123-k',1004,1),
        (2,101):HouseholdDescriptor('visitor','r123-f',1000,1)}
    first=harness.publish(actor.prepare_image({first_source:[(raw[first_source],0.,1,1)]},frame=2,
        households=houses,source_changes={first_source:change(first_source,2,False)}),2,source_id=first_source)
    assert not first.entities  # Wrong bbox first never becomes a selected point.
    counts=[0]
    for frame in range(3,6):
        actor.us+=100_000
        snap=harness.publish(actor.prepare_image({1:[(1,0.,1,1)],2:[(101,0.,1,1)]},frame=frame,
            households=houses,source_changes={1:change(1,frame,False),2:change(2,frame,True)}),frame)
        counts.append(len(snap.entities))
    assert counts[-1]==1
    old=snap.entities[0];assert next(s.camera_id for s in old.sources if s.accepted)=='family-room'
    assert old.attribution_quarantine is not None and old.subject.stable_id is None
    assert old.support_state=='floor'
    for frame in range(6,9):
        actor.us+=100_000
        snap=harness.publish(actor.prepare_image({1:[(1,0.,1,1)],2:[(101,0.,1,1)]},frame=frame,
            households=houses,source_changes={1:change(1,frame,True),2:change(2,frame,False)}),frame)
        counts.append(len(snap.entities))
    current=snap.entities[0]
    assert current.entity_id==old.entity_id and current.observed_at_us>old.observed_at_us
    assert next(s.camera_id for s in current.sources if s.accepted)=='kitchen'
    assert current.attribution_quarantine is not None and current.subject.stable_id is None
    actor.us+=3_000_000
    expired=harness.publish(actor.prepare_image({1:[],2:[]},frame=40),40)
    assert not expired.entities
    actual=actual_node_wire(actor,harness.wire,'r123-contact-quality-'+str(first_source))
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert len(actual['snapshots'][0]['people'])==0 and len(actual['snapshots'][-1]['people'])==0
    assert max(len(s['people']) for s in actual['snapshots'])==1


def settled_cohort(actor, houses, *, frame=2, native_tokens=False):
    rows = {1: [(1, 0., 1, 1)], 2: [(101, .03, 1, 1)]}
    if len(houses) == 2:
        rows[1].append((22, 3., 1, 1))
        rows[2].append((102, 3.03, 1, 1))
    household_rows = {(source, row[0]): houses[0 if row[0] in {1, 101} else 1]
                      for source, people in rows.items() for row in people}
    preps, original = actor.normalize(rows, frame=frame, native_tokens=native_tokens, households=household_rows)
    original = {key: SourceObservation(o.stamp, houses[0 if key[1] in {1, 101} else 1])
                for key, o in original.items()}
    for o in original.values():
        actor.identity.observe(o)
    proofs = [actor.proof(original[(1, 1)], original[(2, 101)], f"original-household-A:{frame}", 0.)]
    if len(houses) == 2:
        proofs.append(actor.proof(original[(1, 22)], original[(2, 102)], f"original-household-B:{frame}", 3.))
    for proof in proofs:
        actor.identity.link(proof, actor.at())
    final = tuple(actor.identity.link(proof, actor.at()) for proof in proofs)  # ORIGINAL operation at final revision.
    refusal = next((r for r in final if r.dto is None), None)
    if refusal is not None:
        for prep in preps:
            actor.world.discard(prep)
        return WorldPublicationRefused.from_association_result(refusal), original, proofs, final
    pairs = tuple((r.dto, actor.cohort(r.dto)) for r in final)
    return actor.world.prepare_physical(1, preps, pairs, at=actor.at()), original, proofs, final


def synthetic_presentation(actor):
    # Explicit SYNTHETIC identity scene mapping, exact actual source/revision
    # hashes. The actual binding API validates this before any visible head.
    identity = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]
    target = actor.frames[1].target_frame.model_dump(mode="json")
    digest, registration = "c" * 64, "d" * 64
    binding = {CAMERAS[s]: {"contract": "noesis.calibration.frame_binding", "contract_version": 2,
        "calibration_frame": b.source_frame.model_dump(mode="json"), "world_frame": target,
        "target_from_calibration_col_major": list(b.target_from_source_col_major),
        "transform_sha256": b.target_from_source_sha256, "metric_frame": {"accepted": True, "frame": target}}
        for s, b in actor.frames.items()}
    presentation = {"contract": "noesis.scene_prior.world_to_scene_presentation", "contract_version": 1,
        "accepted": True, "render_frame": "menon_scene", "world_frame": target,
        "world_to_scene_col_major": identity, "world_to_scene_sha256": "e" * 64, "scene_revision_id": "f" * 64,
        "source_transform_sha256s": [b.target_from_source_sha256 for b in actor.frames.values()],
        "authored_scene_sha256": digest, "registration_sha256": registration, "provenance": {"sha256": registration}}
    return {"type": "calibration-bundle", "data": {"meta": {"authored_scene_sha256": digest,
        "registration_sha256": registration}, "cameras": {"frame_bindings": binding},
        "world_frame_presentations": {target["revision"]: presentation}}}


NODE_WIRE_PROGRAM = r'''
import fs from 'node:fs';
if (process.versions.node.split('.')[0] !== '22') throw Error('Node22 required');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const { createUnifiedBevBindings, createUnifiedBevState, reduceUnifiedBev } = await import(input.module);
createUnifiedBevBindings(input.bundle, input.bundle.data.meta.authored_scene_sha256);
let state = createUnifiedBevState({ authoredSceneSha256: input.bundle.data.meta.authored_scene_sha256 });
const diagnostics = [], snapshots = [];
for (const message of [input.bundle, ...input.wire]) {
  const receivedAtMs = message.type === 'calibration-bundle' ? 0 : 1000 +
    (message.cohort.observed_at_us - input.wire[0].cohort.observed_at_us) / 1000;
  state = reduceUnifiedBev(state, message, { receivedAtMs });
  diagnostics.push(state.diagnostic);
  if (message.type === 'world_snapshot') snapshots.push({cohort: message.cohort,
    people: state.people.map(p => ({entityId:p.entityId,subjectId:p.subjectId,stableId:p.stableId,
      generation:p.generation,continuitySegmentId:p.continuitySegmentId,observedAtUs:p.observedAtUs,
      staleAfterUs:p.staleAfterUs,worldPosition:p.worldPosition,worldCovariance:p.worldCovariance,
      supportSurface:p.supportSurface,sourceEvidence:p.sourceEvidence,
      attributionQuarantine:p.attributionQuarantine,label:p.label,identityKind:p.identityKind,
      trailBreak:p.trailBreak,trail:p.trail})), unavailable:state.unavailable});
}
const output = JSON.stringify({node:process.versions.node,diagnostics,snapshots});
if (Buffer.byteLength(output) > 128*1024) throw Error('bounded Node output exceeded');
process.stdout.write(output);
'''


def actual_node_wire(actor, wire, scenario):
    selected = os.environ.get("NOESIS_XC3D_NODE22")
    assert selected, "select Node22 explicitly via NOESIS_XC3D_NODE22; no fallback"
    payload = json.dumps({"module": (Path(__file__).resolve().parents[1] / "oai2-fe/src/lib/unifiedBevWorld.js").as_uri(),
                          "bundle": synthetic_presentation(actor), "wire": wire}, allow_nan=False).encode()
    assert len(payload) <= 512 * 1024
    process = subprocess.Popen([selected, "--input-type=module", "--eval", NODE_WIRE_PROGRAM],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    deadline, output = time.monotonic() + 10., bytearray()
    try:
        os.set_blocking(process.stdin.fileno(), False)
        os.set_blocking(process.stdout.fileno(), False)
        offset = 0
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            selector.register(process.stdin, selectors.EVENT_WRITE)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                assert remaining > 0, "Node subprocess exceeded10s"
                for key, _ in selector.select(remaining):
                    if key.fileobj is process.stdin:
                        try:
                            offset += os.write(key.fd, payload[offset:offset + 65536])
                        except BlockingIOError:
                            continue
                        if offset == len(payload):
                            selector.unregister(process.stdin)
                            process.stdin.close()
                        continue
                    try:
                        chunk = os.read(key.fd, min(65536, 128 * 1024 + 1 - len(output)))
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                    else:
                        output.extend(chunk)
                        assert len(output) <= 128 * 1024, "Node output exceeded128KiB"
        assert process.wait(timeout=max(.001, deadline - time.monotonic())) == 0, output.decode()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=1)
        if not process.stdin.closed:
            process.stdin.close()
        process.stdout.close()
    result = json.loads(output)
    assert result["node"].split(".")[0] == "22"
    assert all(d["accepted"] for d in result["diagnostics"]), result["diagnostics"]
    directory = os.environ.get("NOESIS_XC3D_REPORT_DIR")
    if directory:
        target = Path(directory)
        assert target.is_absolute() and target.is_relative_to(Path("/mnt"))
        target.mkdir(parents=True, exist_ok=True)
        # Generated CPU fixture evidence ONLY, never application source/authority.
        (target / f"{scenario}.json").write_text(json.dumps({"synthetic": True, "scenario": scenario,
            "bundle": synthetic_presentation(actor), "wire": wire, "actualNode": result}, allow_nan=False) + "\n")
    print("XC3D_NODE", scenario, json.dumps({"diagnostics": len(result["diagnostics"]),
        "visible": [len(s["people"]) for s in result["snapshots"]],
        "people": [[{"id": p["entityId"], "sid": p["stableId"], "generation": p["generation"],
                    "observed": p["observedAtUs"], "trail": len(p["trail"])} for p in s["people"]]
                   for s in result["snapshots"]]}))
    return result


def test_actual_two_people_one_ordered_world_commit_and_a_only_update_keeps_b_original():
    actor = OrderedActor()
    prepared, (a, b), _, _ = actor.two_people()
    assert actor.world.current_snapshot() is None
    assert len(prepared.publication.snapshot.entities) == 2
    publication = actor.world.commit(prepared, association_at=actor.at())
    assert actor.world.current_snapshot() is publication.snapshot
    assert len(publication.events) == 2 and all(e.event_type == "appeared" for e in publication.events)
    expected = {f"physical:run-synthetic:{dto.physical_key.serial}" for dto in (a, b)}
    assert {e.entity_id for e in publication.snapshot.entities} == expected
    assert all(e.subject.stable_id is None and e.subject.kind == "unknown" for e in publication.snapshot.entities)
    assert all(len(e.sources) == 2 and e.covariance.values[2] != 0 for e in publication.snapshot.entities)
    assert all(e.world_transform_sha256 is None for e in publication.snapshot.entities)  # two valid source edges
    original_b = next(e for e in publication.snapshot.entities if e.entity_id.endswith(f":{b.physical_key.serial}"))
    actor.us += 100_000
    preps, current = actor.normalize({1: [(1, .05, 1, 1)], 2: [(101, .08, 1, 1)]}, frame=3)
    for observation in current.values():
        actor.identity.observe(observation)
    left, right = current[(1, 1)], current[(2, 101)]
    result = actor.identity.link(actor.proof(left, right, "fresh-A", .05), actor.at())
    assert result.disposition == "accepted"
    assert actor.identity.prepare(b, actor.cohort(b), actor.at()).disposition == "denied"
    candidate = actor.world.prepare_physical(1, preps, ((result.dto, actor.cohort(result.dto)),), at=actor.at())
    second = actor.world.commit(candidate, association_at=actor.at())
    retained_b = next(e for e in second.snapshot.entities if e.entity_id == original_b.entity_id)
    assert retained_b.position == original_b.position
    assert retained_b.observed_at_us == original_b.observed_at_us
    assert retained_b.stale_after_us == original_b.stale_after_us
    assert retained_b.covariance == original_b.covariance
    assert retained_b.sources == original_b.sources
    assert not any(e.event_type in {"appeared", "lost"} for e in second.events)


def test_original_native_tokens_survive_actual_two_person_world_commit():
    actor = OrderedActor()
    prepared, _, _, observations = actor.two_people(native_tokens=True)
    expected = {record.stamp.observation_id for record in observations.values()}
    assert all(token.startswith("synthetic-native:") for token in expected)
    publication = actor.world.commit(prepared, association_at=actor.at())
    assert {source.observation_id for entity in publication.snapshot.entities
            for source in entity.sources} == expected
    assert {o.observation_id for o in publication.observations} == {
        observations[(1, 1)].stamp.observation_id, observations[(1, 22)].stamp.observation_id,
    }


def test_unmatched_original_native_record_cannot_be_silently_dropped():
    actor = OrderedActor()
    original = SourceObservation(ObservationStamp(MemberKey("run-synthetic", "kitchen", 1, 0, 9, 1),
                                                 2, "synthetic-native-extra", 10.2,
                                                 "synthetic-capture", .2, "ntp", 1))
    with pytest.raises(ValueError, match="lacks its exact normalized row"):
        actor.world.prepare(1, [], metadata={"camera_id": "kitchen", "source_epoch": 0},
                            physical_observations=(original,))
    assert actor.world.current_snapshot() is None


def test_actual_tracking_publisher_releases_one_two_head_snapshot_after_world_commit():
    actor = OrderedActor()
    prepared, _, _, _ = actor.two_people(native_tokens=True)
    released = []

    class FixtureWorldIngress:
        # The native ordered actor must implement this handoff in its own
        # scope. This fixture supplies the ORIGINAL actual World candidate,
        # not a replacement validator or fabricated publication.
        def prepare(self, source, tracks, *, metadata):
            assert source == 1 and len(tracks) == 2
            return prepared

        def commit(self, receipt):
            return actor.world.commit(receipt, association_at=actor.at())

        def discard(self, receipt):
            return actor.world.discard(receipt)

    class FixtureOutboundGate:
        def admit_broadcast_batch_sync(self, messages):
            assert actor.world.current_snapshot() is None

            def commit_then_release(commit):
                assert actor.world.current_snapshot() is None
                publication = commit()
                assert actor.world.current_snapshot() is publication.snapshot
                released.extend(messages)
                return publication

            return SimpleNamespace(receipt=SimpleNamespace(submission_id=1, message_count=len(messages)),
                                   commit_then_release=commit_then_release)

    publisher = TrackingTelemetryPublisher(FixtureOutboundGate(), world_service=FixtureWorldIngress())
    tracks = [{"tracker_id": raw, "tracker_lifecycle_generation": 1, "frame_id": 2,
               "world": [x, 0., 2.], "world_valid": True} for raw, x in ((1, 0.), (22, 3.))]
    receipt = publisher.publish(1, tracks, frame_metadata={"frame_id": 2, "observed_at_us": actor.us})
    assert receipt.canonical_world_admission_bound is True
    assert receipt.outbound_message_count == 4
    assert [message["type"] for message in released] == ["tracking", "world_snapshot", "world_event", "world_event"]
    assert len(released[0]["observations"]) == 2
    assert len(released[0]["world_snapshot"]["entities"]) == 2
    assert released[0]["world_snapshot"] == released[1]["payload"]
    assert all(message["cohort"] == released[0]["cohort"] for message in released)
    assert len(actor.world.current_snapshot().entities) == 2


def test_original_adoption_replay_and_expiry_are_not_generic_link_or_retimed_evidence():
    actor = OrderedActor()
    prepared, (_, b), (_, proof, event, current), _ = actor.two_people()
    assert actor.identity.link(proof, actor.at()).reason == "token_conflict"
    actor.world.discard(prepared)
    expired = actor.identity.transition(event, current, actor.at(10.56), proof)
    assert expired.disposition != "accepted"
    assert actor.world.current_snapshot() is None


@pytest.mark.parametrize("action", ["advance", "terminate", "reset"])
def test_revoked_between_prepare_and_commit_publishes_nothing(action):
    actor = OrderedActor()
    prepared, _, _, observations = actor.two_people()
    previous = observations[(1, 1)]
    if action == "advance":
        actor.identity.observe(SourceObservation(replace(previous.stamp, frame_id=3, observation_id="later-A",
                                                        acquired_at_s=10.3, media_time_s=.3)))
    elif action == "terminate":
        actor.identity.withdraw(MemberRemoval("synthetic-termination", previous.stamp, "termination", 3, 10.3,
                                              "synthetic-capture", .3))
    else:
        actor.identity.reset_source(replace(actor.sources[0], source_epoch=1))
    with pytest.raises(RuntimeError, match="denied"):
        actor.world.commit(prepared, association_at=actor.at(10.3))
    assert actor.world.current_snapshot() is None
    assert actor.identity.stats().pending_reservations == 0
    with pytest.raises(RuntimeError, match="consumed"):
        actor.world.commit(prepared, association_at=actor.at(10.3))


@pytest.mark.parametrize("house", [HouseholdDescriptor("provisional"),
    HouseholdDescriptor("resident", "synthetic-enrolled-resident", 17),
    HouseholdDescriptor("visitor", "synthetic-existing-visitor", 17, 4)])
def test_household_promotion_preserves_physical_entity_and_trail_segment(house):
    actor = OrderedActor()
    harness = ActualPublicationHarness(actor)
    # Actual empty publication0 warms the real publisher; the two promotion
    # publications then have the authoritative source1/publication1,2 cohorts.
    empty, _ = actor.normalize({1: []}, frame=1)
    harness.publish(actor.world.prepare_physical(1, empty, (), at=actor.at()), 1)
    prepared, (old, _), _, _ = actor.two_people()
    first = prepared.publication
    assert harness.publish(prepared, 2) is first.snapshot
    original = next(e for e in first.snapshot.entities if e.entity_id.endswith(f":{old.physical_key.serial}"))
    actor.us += 100_000
    preps, current = actor.normalize({1: [(1, .05, 1, 1)], 2: [(101, .08, 1, 1)]}, frame=3,
                                     households={(1, 1): house, (2, 101): house})
    left, right = (SourceObservation(o.stamp, house) for o in current.values())
    actor.identity.observe(left)
    actor.identity.observe(right)
    dto = actor.identity.link(actor.proof(left, right, "promotion", .05), actor.at()).dto
    candidate = actor.world.prepare_physical(1, preps, ((dto, actor.cohort(dto)),), at=actor.at())
    second = candidate.publication
    assert harness.publish(candidate, 3) is second.snapshot
    promoted = next(e for e in second.snapshot.entities if e.entity_id == original.entity_id)
    assert dto.physical_key == old.physical_key
    assert promoted.subject.kind == house.state
    assert promoted.subject.stable_id == house.stable_id
    if house.state == "visitor":
        assert promoted.subject.generation == 4
    assert promoted.continuity_segment_id == original.continuity_segment_id
    assert not any(e.event_type in {"appeared", "lost"} for e in second.events)
    if house.state == "visitor":
        result = actual_node_wire(actor, harness.wire, "t6-full-wire-promotion")
        before, after = result["snapshots"][-2:]
        assert before["cohort"]["tracking_publication_sequence"] == 1
        assert after["cohort"]["tracking_publication_sequence"] == 2
        previous = next(p for p in before["people"] if p["entityId"] == original.entity_id)
        current = next(p for p in after["people"] if p["entityId"] == original.entity_id)
        assert (previous["generation"], current["generation"], current["stableId"]) == (0, 4, 17)
        assert (previous["observedAtUs"], current["observedAtUs"]) == (10_200_000, 10_300_000)
        assert previous["continuitySegmentId"] == current["continuitySegmentId"]
        assert len(previous["trail"]) == 1 and len(current["trail"]) == 2
        assert current["trail"][0] == previous["trail"][0]
        other_before = next(p for p in before["people"] if p["entityId"] != original.entity_id)
        other_after = next(p for p in after["people"] if p["entityId"] != original.entity_id)
        for field in ("entityId", "subjectId", "stableId", "generation", "worldPosition", "observedAtUs",
                      "staleAfterUs", "worldCovariance", "supportSurface", "sourceEvidence", "trail"):
            assert other_before[field] == other_after[field]
        assert harness.publisher._terminal_failure is None


def test_settled_household_conflict_refuses_shared_world_grant():
    actor = OrderedActor()
    preps, records = actor.normalize({1: [(1, 0., 1, 1)], 2: [(101, .03, 1, 1)]})
    left = SourceObservation(records[(1, 1)].stamp, HouseholdDescriptor("resident", "synthetic-enrolled-a", 1))
    right = SourceObservation(records[(2, 101)].stamp, HouseholdDescriptor("visitor", "synthetic-visitor-b", 2, 1))
    actor.identity.observe(left)
    actor.identity.observe(right)
    result = actor.identity.link(actor.proof(left, right, "house-conflict", 0.), actor.at())
    assert result.reason == "identity_conflict" and result.dto is None
    with pytest.raises(ValueError, match="grants"):
        actor.world.prepare_physical(1, preps, (), at=actor.at())
    assert actor.world.current_snapshot() is None


@pytest.mark.parametrize("changes", [
    {"world_frame_revision": "foreign-revision"},
    {"world_transform_sha256": "f" * 64},
    {"calibration_revision": "foreign-source-calibration"},
    {"image_size": [1280, 720]},
    {"capture_time_status": "estimated"},
    {"captured_at_us": 10_199_999},
    {"source_epoch": 9},
    {"world_source": "mv3dt_peer_fused"},
    {"world_source": "anchor_hold", "world_quality": "held"},
    {"world_source": "cv_prediction", "world_quality": "held"},
    {"world_valid": False},
])
def test_original_geometry_time_epoch_raster_or_nonmeasurement_mismatch_cannot_bootstrap(changes):
    actor = OrderedActor()
    preps, records = actor.normalize({1: [(1, 0., 1, 1)], 2: [(101, .03, 1, 1)]}, **changes)
    for record in records.values():
        actor.identity.observe(record)
    left, right = records[(1, 1)], records[(2, 101)]
    result = actor.identity.link(actor.proof(left, right, "independent-pair", 0.), actor.at())
    assert result.disposition == "accepted"  # DTO coherence is not World admission.
    with pytest.raises((ValueError, RuntimeError)):
        actor.world.prepare_physical(1, preps, ((result.dto, actor.cohort(result.dto)),), at=actor.at())
    assert actor.world.current_snapshot() is None
    assert actor.identity.stats().pending_reservations == 0


@pytest.mark.parametrize("change", ["floor", "metric-null", "time-null", "independent-null", "raster"])
def test_active_binding_or_receipt_switch_before_commit_denies_all_heads(change):
    actor = OrderedActor()
    prepared, _, _, _ = actor.two_people()
    if change == "floor":
        actor.frames[2] = frame_binding(2, metric(offset=-.2))
    elif change == "metric-null":
        actor.frames = {}
    elif change == "raster":
        actor.rasters[2] = (1280, 720)
    else:
        field = "acquisition_receipt_id" if change == "time-null" else "independent_source_receipt_id"
        actor.admission = replace(actor.admission, **{field: None})
    with pytest.raises((ValueError, RuntimeError)):
        actor.world.commit(prepared, association_at=actor.at())
    assert actor.world.current_snapshot() is None
    assert actor.identity.stats().pending_reservations == 0


def test_actual_world_positions_cannot_be_replaced_by_different_claimed_proof_positions():
    actor = OrderedActor()
    preps, records = actor.normalize({1: [(1, 0., 1, 1)], 2: [(101, 4., 1, 1)]})
    for record in records.values():
        actor.identity.observe(record)
    # An upstream scalar claim alone is not a substitute for owned World data.
    bad_claim = actor.proof(records[(1, 1)], records[(2, 101)], "bad-position-claim", 0.)
    dto = actor.identity.link(bad_claim, actor.at()).dto
    assert dto is not None
    with pytest.raises(ValueError, match="original World measurements"):
        actor.world.prepare_physical(1, preps, ((dto, actor.cohort(dto)),), at=actor.at())
    assert actor.world.current_snapshot() is None


def test_held_retention_never_renews_clock_covariance_support_or_membership_and_expires_at_two_seconds():
    actor = OrderedActor()
    prepared, _, _, _ = actor.two_people()
    original = actor.world.commit(prepared, association_at=actor.at())
    assert all(s.covariance.values == COV for e in original.snapshot.entities for s in e.sources)
    actor.us = 11_000_000
    rows = {1: [(1, 0., 1, 1), (22, 3., 1, 1)], 2: [(101, .03, 1, 1), (102, 3.03, 1, 1)]}
    # Real original normalization of diagnostic-only rows: no new position or
    # Identity observe/link call. It neither drops the ordered frame nor renews B.
    preps, _ = actor.normalize(rows, frame=3, world_valid=False)
    held = actor.world.commit(actor.world.prepare_physical(1, preps, (), at=actor.at()), association_at=actor.at())
    assert all(e.lifecycle == "held" and e.position_state == "held" for e in held.snapshot.entities)
    for before, after in zip(original.snapshot.entities, held.snapshot.entities, strict=True):
        assert after.position == before.position and after.covariance == before.covariance
        assert after.support_surface == before.support_surface
        assert after.observed_at_us == before.observed_at_us and after.stale_after_us == before.stale_after_us
        assert after.sources == before.sources
    actor.us = 12_200_001
    preps, _ = actor.normalize(rows, frame=4, world_valid=False)
    expired = actor.world.commit(actor.world.prepare_physical(1, preps, (), at=actor.at()), association_at=actor.at())
    assert not expired.snapshot.entities
    assert len(expired.events) == 2 and all(e.event_type == "lost" for e in expired.events)


def test_queued_later_media_and_reset_wait_behind_older_work_then_remove_exact_old_epoch():
    actor = OrderedActor()
    prepared, _, _, records = actor.two_people()
    revision = actor.identity.stats().association_revision
    actor.queue.append(("later-media", ((1, .05, 1, 1),)))
    actor.queue.append(("source-reset", replace(actor.sources[0], source_epoch=1)))
    assert actor.identity.stats().association_revision == revision
    assert actor.identity.stats().pending_reservations == 2
    with pytest.raises(RuntimeError, match="queue"):
        actor.world.prepare(1, [], metadata={"camera_id": "kitchen"})
    original = actor.world.commit(prepared, association_at=actor.at())
    assert len(original.snapshot.entities) == 2
    kind, queued_rows = actor.queue.popleft()
    assert kind == "later-media"
    actor.us += 100_000
    preps, later = actor.normalize({1: queued_rows, 2: ((101, .08, 1, 1),)}, frame=3)
    for observation in later.values():
        actor.identity.observe(observation)
    fresh = actor.identity.link(actor.proof(later[(1, 1)], later[(2, 101)], "queued-fresh-A", .05), actor.at())
    assert fresh.disposition == "accepted"
    later_world = actor.world.commit(actor.world.prepare_physical(1, preps,
        ((fresh.dto, actor.cohort(fresh.dto)),), at=actor.at()), association_at=actor.at())
    assert len(later_world.snapshot.entities) == 2
    original_times = {e.entity_id: e.observed_at_us for e in later_world.snapshot.entities}
    assert sorted(original_times.values()) == [10_200_000, 10_300_000]
    _, new_binding = actor.queue.popleft()
    reset = actor.identity.reset_source(new_binding)
    assert len(reset.withdrawn_members) == 2
    actor.sources = (new_binding, actor.sources[1])
    actor.policy = replace(actor.policy, sources=actor.sources)
    actor.admission = actor.admitted(actor.policy)
    assert actor.identity.bind_admission(actor.admission).disposition == "bound"
    actor.us += 100_000
    preps, _ = actor.normalize({1: []}, frame=4)
    barrier = actor.world.prepare_physical(1, preps, (), at=actor.at(), withdrawn_members=reset.withdrawn_members)
    reset_world = actor.world.commit(barrier, association_at=actor.at())
    assert len(reset_world.snapshot.entities) == 2
    assert all(len(e.sources) == 1 and e.sources[0].camera_id == "family-room" for e in reset_world.snapshot.entities)
    assert {e.entity_id: e.observed_at_us for e in reset_world.snapshot.entities} == original_times
    assert actor.identity.observe(records[(1, 1)]).disposition not in {"observed", "accepted"}
    # SAME numeric ID but a new epoch/lifecycle is not the old physical person.
    actor.us += 100_000
    preps, current = actor.normalize({1: [(1, 8., 2, 5)]}, frame=5)
    record = current[(1, 1)]
    dto = actor.identity.observe(record).dto
    b = actor.frames[1]
    measured = MeasurementEvidence(record.stamp, record.stamp.member, "new-local-measurement",
                                   "backend_world_m", b.target_frame.revision, b.source_frame.revision,
                                   b.target_from_source_sha256, 8., 2., independence_id="new-independent-local")
    next_world = actor.world.commit(actor.world.prepare_physical(1, preps, ((dto, actor.cohort(dto, measured)),),
                                     at=actor.at(), withdrawn_members=reset.withdrawn_members), association_at=actor.at())
    assert len(next_world.snapshot.entities) == 3
    assert sum(s.camera_id == "kitchen" for e in next_world.snapshot.entities for s in e.sources) == 1


def test_correction_and_termination_remove_only_exact_contributor_before_new_local_person():
    actor = OrderedActor()
    prepared, (_, b), _, records = actor.two_people()
    actor.world.commit(prepared, association_at=actor.at())
    actor.us += 100_000
    preps, current = actor.normalize({1: [(22, 8., 1, 1)]}, frame=3)
    old, new, peer = records[(1, 22)], current[(1, 22)], records[(2, 102)]
    event = NativeTransition("synthetic-correction", "correction", old.stamp, new.stamp, peer.stamp,
                             actor.at().at_s, "synthetic-capture")
    result = actor.identity.transition(event, new, actor.at())
    assert result.disposition == "observed" and result.withdrawn_members == (old.stamp.member,)
    dto = result.dto
    edge = actor.frames[1]
    measured = MeasurementEvidence(new.stamp, new.stamp.member, "correction-measurement", "backend_world_m",
                                   edge.target_frame.revision, edge.source_frame.revision,
                                   edge.target_from_source_sha256, 8., 2., independence_id="independent-correction")
    corrected = actor.world.commit(actor.world.prepare_physical(1, preps, ((dto, actor.cohort(dto, measured)),),
                                    at=actor.at(), withdrawn_members=result.withdrawn_members), association_at=actor.at())
    old_entity = next(e for e in corrected.snapshot.entities if e.entity_id.endswith(f":{b.physical_key.serial}"))
    assert len(old_entity.sources) == 1 and old_entity.sources[0].camera_id == "family-room"
    assert dto.physical_key != b.physical_key
    actor.us += 100_000
    removal = actor.identity.withdraw(MemberRemoval("synthetic-exact-termination", new.stamp, "termination",
                                                     4, actor.at().at_s, "synthetic-capture", .4))
    preps, _ = actor.normalize({1: []}, frame=4)
    terminated = actor.world.commit(actor.world.prepare_physical(1, preps, (), at=actor.at(),
                                     withdrawn_members=removal.withdrawn_members), association_at=actor.at())
    assert all(not e.entity_id.endswith(f":{dto.physical_key.serial}") for e in terminated.snapshot.entities)
    assert any(e.entity_id == old_entity.entity_id for e in terminated.snapshot.entities)


def test_partial_grant_failure_consumes_intent_but_never_partially_publishes_world(monkeypatch):
    actor = OrderedActor()
    prepared, _, _, records = actor.two_people()
    commit = actor.identity.commit
    calls = []
    def interleave_after_first(reservation, at, consumer):
        result = commit(reservation, at, consumer)
        calls.append(result.disposition)
        assert actor.world.current_snapshot() is None
        if len(calls) == 1:
            old = records[(1, 1)]
            actor.identity.observe(SourceObservation(replace(old.stamp, frame_id=3, observation_id="queued-later-A",
                                                           acquired_at_s=10.3, media_time_s=.3)))
        return result
    monkeypatch.setattr(actor.identity, "commit", interleave_after_first)
    with pytest.raises(RuntimeError, match="denied"):
        actor.world.commit(prepared, association_at=actor.at())
    assert calls == ["committed", "denied"]
    assert actor.world.current_snapshot() is None
    assert actor.identity.stats().pending_reservations == 0
    with pytest.raises(RuntimeError, match="consumed"):
        actor.world.commit(prepared, association_at=actor.at())


def test_original_world_receipt_not_a_reconstructed_envelope_or_copied_receipt():
    actor = OrderedActor()
    preps, _ = actor.normalize({1: []})
    with pytest.raises(RuntimeError, match="stale or consumed"):
        actor.world.prepare_physical(1, (replace(preps[0]),), (), at=actor.at())
    actor.world.discard(preps[0])
    with pytest.raises(RuntimeError, match="stale or consumed"):
        actor.world.prepare_physical(1, preps, (), at=actor.at())
    assert actor.world.current_snapshot() is None


def test_discard_close_and_missing_commit_clock_release_all_unconsumed_reservations():
    for operation in ("discard", "close", "clock"):
        actor = OrderedActor()
        prepared, _, _, _ = actor.two_people()
        if operation == "discard":
            actor.world.discard(prepared)
        elif operation == "close":
            actor.world.close()
        else:
            with pytest.raises(ValueError, match="commit clock"):
                actor.world.commit(prepared)
        assert actor.world.current_snapshot() is None
        assert actor.identity.stats().pending_reservations == 0


def test_foreign_source_clock_and_stale_or_copied_dto_never_publish():
    actor = OrderedActor()
    prepared, (_, b), _, _ = actor.two_people()
    with pytest.raises(RuntimeError, match="denied"):
        actor.world.commit(prepared, association_at=AcquisitionTime(10.2, "foreign-clock", "ntp"))
    assert actor.world.current_snapshot() is None
    assert actor.identity.prepare(replace(b), actor.cohort(b), actor.at()).disposition == "denied"


def test_default_subject_key_behavior_and_limits_are_unchanged_outside_gate():
    actor = OrderedActor()
    actor.world = CanonicalWorldService(producer=actor.world.producer, artifacts=actor.artifacts, clock_us=lambda: actor.us)
    preps, _ = actor.normalize({1: [(1, 0., 1, 1)]}, identity_kind="resident", resident_uuid="synthetic-enrolled", stable_id=9)
    actor.world.commit(preps[0])
    preps, _ = actor.normalize({2: [(101, .03, 1, 1)]}, identity_kind="resident", resident_uuid="synthetic-enrolled", stable_id=9)
    publication = actor.world.commit(preps[0])
    assert len(publication.snapshot.entities) == 1
    assert publication.snapshot.entities[0].entity_id == "resident:synthetic-enrolled"
    assert publication.snapshot.entities[0].subject.stable_id == 9
    assert CanonicalWorldService.MAX_PREPARED_CANDIDATES == 16


def test_no_dto_and_no_position_when_real_admission_inputs_are_unavailable():
    unavailable = ProducerAdmission("unavailable", "XC3D-RR-WORLD-INPUTS-R1", "metric-time-independent-epoch-unadmitted")
    owner = IdentityAssociationService("run-synthetic", unavailable)
    actor = OrderedActor()
    preps, records = actor.normalize({1: [(1, 0., 1, 1)]})
    assert owner.observe(records[(1, 1)]).dto is None
    actor.world.discard(preps[0])
    actor.world = CanonicalWorldService(producer=actor.world.producer, artifacts=actor.artifacts,
                                       association_service=owner, clock_us=lambda: actor.us,
                                       physical_admission=lambda: unavailable,
                                       physical_bindings=lambda: None, physical_rasters=lambda: None)
    preps, _ = actor.normalize({1: [(1, 0., 1, 1)]})
    with pytest.raises(ValueError, match="current service grants"):
        actor.world.prepare_physical(1, preps, (), at=actor.at())
    assert actor.world.current_snapshot() is None


def test_exact_current_rr_v1_diagnostic_bindings_cannot_enable_cross_source_world():
    actor = OrderedActor()
    prepared, _, _, _ = actor.two_people()
    actor.frames = {s: ScenePriorFrameBinding.model_validate(CURRENT_DIAGNOSTIC_BINDINGS[camera])
                    for s, camera in CAMERAS.items()}
    assert all(binding.metric_frame is None for binding in actor.frames.values())
    assert actor.frames[1].target_frame != actor.frames[2].target_frame
    with pytest.raises(RuntimeError, match="physically admitted common metric/raster binding unavailable"):
        actor.world.commit(prepared, association_at=actor.at())
    assert actor.world.current_snapshot() is None
    assert actor.identity.stats().pending_reservations == 0


@pytest.mark.parametrize("count", [1, 2])
def test_t6_actual_settled_visible_two_or_one_linked_head(count):
    actor = OrderedActor()
    harness = ActualPublicationHarness(actor)
    houses = (HouseholdDescriptor("resident", "synthetic-distinct-A", 11),
              HouseholdDescriptor("visitor", "synthetic-distinct-B", 12, 3))[:count]
    candidate, _, _, results = settled_cohort(actor, houses)
    assert all(result.dto is not None for result in results)
    harness.publish(candidate, 2)
    assert len(harness.journal) == 1
    result = actual_node_wire(actor, harness.wire, f"t6-visible-{count}")
    people = result["snapshots"][-1]["people"]
    assert len(people) == count
    assert {p["stableId"] for p in people} == {house.stable_id for house in houses}
    assert all(len(p["sourceEvidence"]) == 2 for p in people)
    assert all(len(p["trail"]) == 1 and p["observedAtUs"] == 10_200_000 for p in people)


@pytest.mark.parametrize("clash", ["both", "subject", "sid", "visitor"])
def test_t6_original_household_refusal_is_atomic_and_real_owner_correction_recovers(clash):
    actor = OrderedActor()
    harness = ActualPublicationHarness(actor)
    good = (HouseholdDescriptor("resident", "synthetic-distinct-A", 11),
            HouseholdDescriptor("resident", "synthetic-distinct-B", 12))
    actor.us = 10_100_000
    first, _, _, _ = settled_cohort(actor, good, frame=1)
    before = harness.publish(first, 1)
    original_wire = json.loads(json.dumps(harness.wire))
    public_before = (actor.world._revision, actor.world._fusion, actor.world._sequence_by_source.copy(),
                     actor.world._previous_entities.copy(), actor.world._event_sequence,
                     actor.world._canonical_track_outputs.copy(), len(harness.journal), harness.admissions)
    actor.us = 10_200_000
    if clash == "visitor":
        bad = (HouseholdDescriptor("visitor", "synthetic-slot", 9, 1),
               HouseholdDescriptor("visitor", "synthetic-slot", 9, 2))
    else:
        bad = (HouseholdDescriptor("resident", "synthetic-duplicate-resident", 9),
               HouseholdDescriptor("resident", "synthetic-duplicate-resident" if clash != "sid" else "synthetic-other", 9 if clash != "subject" else 10))
    refusal, originals, proofs, issuance = settled_cohort(actor, bad)  # EXACT parent rows/frame2/x0-x3.
    assert isinstance(refusal, WorldPublicationRefused)
    assert refusal.reason == "settled_household_claim_conflict"
    assert refusal.household_conflict is issuance[0].household_conflict
    assert len({key for key, _ in refusal.household_conflict.owners}) == 2
    assert {o.stamp for _, o in refusal.household_conflict.owners} == {o.stamp for o in originals.values()}
    harness.prepare_failure = refusal
    with pytest.raises(WorldPublicationRefused) as caught:
        harness.publish(None, 2)
    assert caught.value is refusal and harness.failures[-1] is refusal
    assert actor.world.current_snapshot() is before and harness.wire == original_wire
    assert public_before == (actor.world._revision, actor.world._fusion, actor.world._sequence_by_source.copy(),
                            actor.world._previous_entities.copy(), actor.world._event_sequence,
                            actor.world._canonical_track_outputs.copy(), len(harness.journal), harness.admissions)
    assert actor.identity.stats().pending_reservations == 0
    prior_client = actual_node_wire(actor, harness.wire, f"t6-{clash}-refused-prior")
    assert len(prior_client["snapshots"][-1]["people"]) == 2  # No accepted empty replacement.
    harness.prepare_failure = None
    actor.us = 10_300_000
    # Genuine NEW producer records after refusal; old World counter IDs are
    # not reused/renamed/retimed to revive the rejected original proofs.
    corrected, _, _, final = settled_cohort(actor, good, frame=3, native_tokens=True)
    assert all(r.dto for r in final)
    harness.publish(corrected, 3)
    recovered = actual_node_wire(actor, harness.wire, f"t6-{clash}-corrected")
    assert [len(s["people"]) for s in recovered["snapshots"]] == [2, 2]
    assert {p["entityId"] for p in recovered["snapshots"][0]["people"]} == {
        p["entityId"] for p in recovered["snapshots"][1]["people"]}
    assert actor.identity.link(proofs[0], actor.at()).dto is None  # ORIGINAL stale proof not retimed.


@pytest.mark.parametrize("field", ["subject", "sid"])
def test_t6_public_fusion_entire_retained_ledger_guard_and_normal_expiry(field):
    # Independent canonical/fusion boundary. No invented owner DTO or Identity
    # grant forces a positive when its real eligibility correctly denies.
    actor = OrderedActor()
    actor.world = CanonicalWorldService(producer=actor.world.producer, artifacts=actor.artifacts, clock_us=lambda: actor.us)
    preps, _ = actor.normalize({1: [(1, 0., 1, 1)]})
    a = actor.world.commit(preps[0]).observations[0]
    preps, _ = actor.normalize({2: [(102, 3., 1, 1)]})
    b = actor.world.commit(preps[0]).observations[0]
    fusion = GlobalWorldFusion(actor.world.producer)
    subjects = (SubjectRef(subject_id="resident:A", kind="resident", resident_uuid="A", stable_id=11),
                SubjectRef(subject_id="resident:B", kind="resident", resident_uuid="B", stable_id=12))
    members = (MemberKey("run-synthetic", "kitchen", 1, 0, 1, 1),
               MemberKey("run-synthetic", "family-room", 2, 0, 102, 1))
    for serial, envelope, subject, member in zip((1, 2), (a, b), subjects, members, strict=True):
        fusion.ingest(envelope, subject, physical_key=PhysicalPersonKey("run-synthetic", serial),
                      physical_member=member, source_epoch=0, tracker_lifecycle_generation=1, trail_segment_id=1)
    first = fusion.snapshot(published_at_us=actor.us)
    original_b = first.entities[1]
    actor.us += 100_000
    fresh, _ = actor.normalize({1: [(1, .05, 1, 1)]}, frame=3)
    actor.world.commit(fresh[0])
    claim = SubjectRef(subject_id="resident:B" if field == "subject" else "resident:A",
                       kind="resident", resident_uuid="B" if field == "subject" else "A",
                       stable_id=11 if field == "subject" else 12)
    fusion.ingest(fresh[0].publication.observations[0], claim, physical_key=PhysicalPersonKey("run-synthetic", 1),
                  physical_member=members[0], source_epoch=0, tracker_lifecycle_generation=1, trail_segment_id=1)
    state = (fusion._snapshot_sequence, fusion._by_entity.copy(), fusion._entity_by_tracklet.copy(),
             fusion._last_fused.copy(), fusion._continuity_by_entity.copy())
    with pytest.raises(CanonicalWorldIdentityError) as caught:
        fusion.snapshot(published_at_us=actor.us)
    assert caught.value.reason == "canonical_identity_duplicate"
    assert caught.value.collisions[0].field == f"subject.{ 'subject_id' if field == 'subject' else 'stable_id' }"
    assert state == (fusion._snapshot_sequence, fusion._by_entity.copy(), fusion._entity_by_tracklet.copy(),
                     fusion._last_fused.copy(), fusion._continuity_by_entity.copy())
    assert original_b.observed_at_us == 10_200_000 and original_b.stale_after_us == 12_200_000
    # Real normal World age expiry, not an early pressure-driven release.
    actor.us = 12_201_000
    latest, _ = actor.normalize({1: [(1, .06, 1, 1)]}, frame=4)
    fusion.ingest(latest[0].publication.observations[0], claim, physical_key=PhysicalPersonKey("run-synthetic", 1),
                  physical_member=members[0], source_epoch=0, tracker_lifecycle_generation=1, trail_segment_id=1)
    recovered = fusion.snapshot(published_at_us=actor.us)
    assert len(recovered.entities) == 1 and recovered.entities[0].entity_id.endswith(":1")


@pytest.mark.parametrize("kind", ["unknown", "provisional", "resident", "visitor"])
@pytest.mark.parametrize("lifecycle", ["present", "held", "lost", "conflict"])
def test_t6_canonical_contract_all_subjects_all_lifecycles_and_sid_independently(kind, lifecycle):
    actor = OrderedActor()
    prepared, _, _, _ = actor.two_people()
    original = prepared.publication.snapshot
    first, second = original.entities
    data = first.subject.model_dump(mode="python")
    data.update(subject_id="duplicate-subject", kind=kind, stable_id=None,
                resident_uuid="synthetic-enrolled" if kind == "resident" else None)
    subject = SubjectRef.model_validate(data)
    entities = tuple(type(entity).model_validate({**entity.model_dump(mode="python"),
        "subject": subject, "lifecycle": "present" if lifecycle == "conflict" else lifecycle,
        "conflict": lifecycle == "conflict", "conflict_reason": "synthetic-spatial-conflict" if lifecycle == "conflict" else None})
        for entity in (first, second))
    with pytest.raises(CanonicalWorldIdentityError) as caught:
        validate_world_identity_uniqueness(entities)
    assert caught.value.collisions[0].field == "subject.subject_id"
    with pytest.raises(ValueError, match="canonical_identity_duplicate"):
        WorldSnapshot.model_validate({**original.model_dump(mode="python"), "entities": entities})
    sid_entities = (first, type(second).model_validate({**second.model_dump(mode="python"),
        "subject": SubjectRef(subject_id="different", kind="unknown", stable_id=9)}))
    sid_entities = (type(first).model_validate({**first.model_dump(mode="python"),
        "subject": SubjectRef(subject_id="first", kind="unknown", stable_id=9)}), sid_entities[1])
    with pytest.raises(CanonicalWorldIdentityError) as sid:
        validate_world_identity_uniqueness(sid_entities)
    assert sid.value.collisions[0].field == "subject.stable_id"
    actor.world.discard(prepared)


def test_t6_commit_typed_owner_conflict_aborts_admitted_gate_without_any_public_effect():
    actor = OrderedActor()
    harness = ActualPublicationHarness(actor)
    candidate, _, _, records = actor.two_people()
    before = (actor.world._revision, actor.world._fusion, actor.world._event_sequence)

    def truthful_later_owner_collision():
        for raw in (1, 22):
            old = records[(1, raw)]
            stamp = replace(old.stamp, frame_id=3, acquired_at_s=10.3, media_time_s=.3,
                            observation_id=f"later-original-household:{raw}")
            actor.identity.observe(SourceObservation(stamp, HouseholdDescriptor("resident", "same-real-owner-input", 9)))
        actor.us = 10_300_000

    harness.before_commit = truthful_later_owner_collision
    with pytest.raises(WorldPublicationRefused) as caught:
        harness.publish(candidate, 2)
    assert caught.value.reason == "settled_household_claim_conflict" and caught.value.household_conflict is not None
    assert harness.failures[-1] is caught.value and harness.admissions == 1
    assert harness.wire == [] and harness.journal == [] and actor.world.current_snapshot() is None
    assert before == (actor.world._revision, actor.world._fusion, actor.world._event_sequence)
    assert actor.identity.stats().pending_reservations == 0
    with pytest.raises(RuntimeError, match="poisoned"):
        harness.publish(candidate, 2)  # Existing post-admission publisher safety, no unsafe auto-retry.


def test_t6_capacity_none_conflict_is_original_typed_refusal_not_permission(monkeypatch):
    actor = OrderedActor()
    prepared, (a, _), _, records = actor.two_people()
    household = HouseholdDescriptor("resident", "capacity-original", 9)
    originals = tuple((PhysicalPersonKey("run-synthetic", 1 + i % 2),
        SourceObservation(ObservationStamp(MemberKey("run-synthetic", "kitchen", 1, 0, 1000 + i, 1),
            2, f"original-budget:{i}", 10.2, "synthetic-capture", .2, "ntp", 1), household)) for i in range(129))
    monkeypatch.setattr(actor.identity.ledger, "household_claim_owners", lambda: originals)  # Public view fault injection ONLY.
    result = actor.identity.observe(records[(1, 1)])
    assert result.disposition == "capacity" and result.reason == "household_claim_capacity"
    refusal = WorldPublicationRefused.from_association_result(result)
    assert refusal.reason == result.reason and refusal.household_conflict is result.household_conflict is None
    with pytest.raises(WorldPublicationRefused) as caught:
        actor.world.commit(prepared, association_at=actor.at())
    assert caught.value.reason == result.reason and caught.value.household_conflict is None
    assert actor.world.current_snapshot() is None and actor.identity.stats().pending_reservations == 0
    assert actor.identity.ledger.household_claim_owners() is originals
    assert actor.identity.ledger.policy.max_entities == 64 and actor.identity.ledger.policy.max_history == 128


def test_t6_exact_prepared_snapshot_guard_precedes_all_grants_and_cached_json(monkeypatch):
    actor = OrderedActor()
    harness = ActualPublicationHarness(actor)
    prepared, _, _, _ = actor.two_people()
    first, second = prepared.publication.snapshot.entities
    bad = type(second).model_validate({**second.model_dump(mode="python"), "subject": first.subject})
    # Adversarial frozen-object corruption ONLY for a denial check; no owner
    # DTO forgery/private Identity verifier or accepted malformed publication.
    object.__setattr__(prepared.publication.snapshot, "entities", (first, bad))
    calls = []
    public_commit = actor.identity.commit
    monkeypatch.setattr(actor.identity, "commit", lambda *args: (calls.append(args), public_commit(*args))[1])
    before = (actor.world._revision, actor.world._fusion, actor.world._event_sequence)
    with pytest.raises(WorldPublicationRefused) as caught:
        harness.publish(prepared, 2)
    assert caught.value.reason == "canonical_identity_duplicate"
    assert caught.value.household_conflict is None
    assert caught.value.identity_conflicts[0].entity_ids == tuple(sorted((first.entity_id, second.entity_id)))
    assert calls == [] and harness.wire == [] and harness.journal == []
    assert before == (actor.world._revision, actor.world._fusion, actor.world._event_sequence)
    assert actor.identity.stats().pending_reservations == 0 and actor.world.current_snapshot() is None


@pytest.mark.parametrize("field", ["subject", "sid"])
def test_t6_actual_world_retained_b_guard_before_grants_and_exact_barrier_recovers(field, monkeypatch):
    actor = OrderedActor()
    harness = ActualPublicationHarness(actor)
    houses = (HouseholdDescriptor("resident", "retained-A", 11), HouseholdDescriptor("resident", "retained-B", 12))
    prepared, records, _, _ = settled_cohort(actor, houses)
    before = harness.publish(prepared, 2)
    old_b = next(e for e in before.entities if e.subject.stable_id == 12)
    withdrawals = []
    # Exact producer-owned absence is pending a World removal barrier. Both
    # supporting Identity sources must release before fresh A can claim it.
    for source, raw in ((1, 22), (2, 102)):
        old = records[(source, raw)]
        removal = actor.identity.withdraw(MemberRemoval(f"owned-absence:{source}", old.stamp, "absence", 3,
                                                       10.3, "synthetic-capture", .3))
        assert removal.disposition == "removed"
        withdrawals.extend(removal.withdrawn_members)
    claim = HouseholdDescriptor("resident", "retained-B" if field == "subject" else "retained-A",
                                11 if field == "subject" else 12)
    actor.us += 100_000
    preps, original = actor.normalize({1: [(1, .05, 1, 1)], 2: [(101, .08, 1, 1)]}, frame=3, native_tokens=True)
    left, right = (SourceObservation(o.stamp, claim) for o in original.values())
    actor.identity.observe(left)
    actor.identity.observe(right)
    dto = actor.identity.link(actor.proof(left, right, "genuine-retained-A", .05), actor.at()).dto
    assert dto is not None  # REAL eligibility grant, no forged/private DTO patch.
    calls = []
    public_commit = actor.identity.commit
    monkeypatch.setattr(actor.identity, "commit", lambda *args: (calls.append(args), public_commit(*args))[1])
    with pytest.raises(WorldPublicationRefused) as caught:
        # Missing the queued exact removal barrier must not let new A override
        # B's legitimately retained ORIGINAL last accepted World evidence.
        actor.world.prepare_physical(1, preps, ((dto, actor.cohort(dto)),), at=actor.at())
    assert caught.value.reason == "canonical_identity_duplicate" and caught.value.household_conflict is None
    assert caught.value.identity_conflicts[0].field == f"subject.{ 'subject_id' if field == 'subject' else 'stable_id' }"
    assert calls == [] and actor.world.current_snapshot() is before
    assert old_b.observed_at_us == 10_200_000 and old_b.stale_after_us == 12_200_000
    assert actor.identity.stats().pending_reservations == 0 and len(harness.journal) == 1
    harness.prepare_failure = caught.value
    with pytest.raises(WorldPublicationRefused):
        harness.publish(None, 3)
    assert harness.admissions == 1 and len(harness.wire) == 4
    # Deliver ORIGINAL exact barrier before replacement at a genuinely later
    # source cohort. No forced expiry, identity clearing or pressure reset.
    harness.prepare_failure = None
    actor.us += 100_000
    preps, originals = actor.normalize({1: [(1, .06, 1, 1)], 2: [(101, .09, 1, 1)]}, frame=4, native_tokens=True)
    left, right = (SourceObservation(o.stamp, claim) for o in originals.values())
    actor.identity.observe(left)
    actor.identity.observe(right)
    fresh = actor.identity.link(actor.proof(left, right, "genuine-after-barrier", .06), actor.at()).dto
    candidate = actor.world.prepare_physical(1, preps, ((fresh, actor.cohort(fresh)),), at=actor.at(),
                                             withdrawn_members=tuple(withdrawals))
    harness.publish(candidate, 4)
    result = actual_node_wire(actor, harness.wire, f"t6-retained-{field}-barrier")
    assert [len(s["people"]) for s in result["snapshots"]] == [2, 1]
    assert result["snapshots"][0]["people"][1]["observedAtUs"] == 10_200_000
    assert result["snapshots"][1]["people"][0]["observedAtUs"] == 10_400_000


def test_t6_same_key_promotion_cannot_take_other_head_claim_or_release_remaining_source():
    actor = OrderedActor()
    harness = ActualPublicationHarness(actor)
    houses = (HouseholdDescriptor("unknown"), HouseholdDescriptor("resident", "occupied-household", 12))
    prepared, records, _, results = settled_cohort(actor, houses)
    before = harness.publish(prepared, 2)
    original_a = results[0].dto.physical_key
    old_b = records[(1, 22)]
    removal = actor.identity.withdraw(MemberRemoval("only-kitchen-leaves", old_b.stamp, "termination", 3,
                                                  10.3, "synthetic-capture", .3))
    assert removal.disposition == "removed"
    actor.us += 100_000
    preps, originals = actor.normalize({1: [(1, .05, 1, 1)], 2: [(101, .08, 1, 1)]}, frame=3, native_tokens=True)
    left, right = (SourceObservation(o.stamp, houses[1]) for o in originals.values())
    actor.identity.observe(left)
    actor.identity.observe(right)
    result = actor.identity.link(actor.proof(left, right, "same-key-occupied-promotion", .05), actor.at())
    assert result.dto is None and result.reason == "settled_household_claim_conflict"
    assert original_a in {key for key, _ in result.household_conflict.owners}
    assert records[(2, 102)] in {o for _, o in result.household_conflict.owners}
    refusal = WorldPublicationRefused.from_association_result(result)
    harness.prepare_failure = refusal
    with pytest.raises(WorldPublicationRefused):
        harness.publish(None, 3)
    for prep in preps:
        actor.world.discard(prep)
    assert actor.world.current_snapshot() is before and harness.admissions == 1
    actual = actual_node_wire(actor, harness.wire, "t6-occupied-promotion-remaining-source")
    assert len(actual["snapshots"][0]["people"]) == 2


def test_t6_original_adoption_expiry_is_typed_without_terminating_retained_b():
    actor = OrderedActor()
    harness = ActualPublicationHarness(actor)
    prepared, _, (_, proof, event, current), _ = actor.two_people(native_tokens=True)
    before = harness.publish(prepared, 2)
    actor.us += 400_000
    result = actor.identity.transition(event, current, actor.at(), proof)
    assert result.dto is None
    refusal = WorldPublicationRefused.from_association_result(result)
    assert refusal.reason == result.reason and refusal.household_conflict is None
    harness.prepare_failure = refusal
    with pytest.raises(WorldPublicationRefused):
        harness.publish(None, 3)
    assert actor.world.current_snapshot() is before and harness.admissions == 1
    actual = actual_node_wire(actor, harness.wire, "t6-original-adoption-expired")
    assert [p["observedAtUs"] for p in actual["snapshots"][0]["people"]] == [10_200_000, 10_200_000]


def test_t6_actual_epoch_reset_raw_reuse_expiry_and_new_run_wire_remain_safe():
    actor = OrderedActor()
    harness = ActualPublicationHarness(actor)
    houses = (HouseholdDescriptor("resident", "reset-A", 11), HouseholdDescriptor("visitor", "reset-B", 12, 3))
    prepared, _, _, _ = settled_cohort(actor, houses)
    first = harness.publish(prepared, 2)
    new_binding = replace(actor.sources[0], source_epoch=1)
    reset = actor.identity.reset_source(new_binding)
    assert len(reset.withdrawn_members) == 2
    actor.sources = (new_binding, actor.sources[1])
    actor.policy = replace(actor.policy, sources=actor.sources)
    actor.admission = actor.admitted(actor.policy)
    assert actor.identity.bind_admission(actor.admission).disposition == "bound"
    actor.us += 100_000
    empty, _ = actor.normalize({1: []}, frame=3)
    barrier = actor.world.prepare_physical(1, empty, (), at=actor.at(), withdrawn_members=reset.withdrawn_members)
    retained = harness.publish(barrier, 3)
    assert len(retained.entities) == 2
    assert all(len(e.sources) == 1 and e.sources[0].camera_id == "family-room" for e in retained.entities)
    assert all(e.observed_at_us == 10_200_000 and e.stale_after_us == 12_200_000 for e in retained.entities)
    # Raw1/new epoch+life is NEW physical ownership, not old A or visitor B.
    actor.us += 100_000
    preps, records = actor.normalize({1: [(1, 8., 2, 4)]}, frame=4, native_tokens=True)
    original = next(iter(records.values()))
    invalid = actor.identity.observe(SourceObservation(original.stamp, HouseholdDescriptor("visitor", "reset-B", 12, 4)))
    assert invalid.dto is None and invalid.household_conflict is not None
    assert any(o.stamp.member.camera_id == "family-room" for _, o in invalid.household_conflict.owners)
    harness.prepare_failure = WorldPublicationRefused.from_association_result(invalid)
    with pytest.raises(WorldPublicationRefused):
        harness.publish(None, 4)
    for prep in preps:
        actor.world.discard(prep)
    assert actor.world.current_snapshot() is retained and harness.admissions == 2
    actor.us += 100_000
    preps, records = actor.normalize({1: [(1, 8.05, 2, 4)]}, frame=5, native_tokens=True)
    record = SourceObservation(next(iter(records.values())).stamp, HouseholdDescriptor("visitor", "new-owner-slot", 13, 1))
    dto = actor.identity.observe(record).dto
    assert dto is not None
    edge = actor.frames[1]
    measurement = MeasurementEvidence(record.stamp, record.stamp.member, "original-new-life",
        "backend_world_m", edge.target_frame.revision, edge.source_frame.revision, edge.target_from_source_sha256,
        8.05, 2., independence_id="original-independent-new-life")
    candidate = actor.world.prepare_physical(1, preps, ((dto, actor.cohort(dto, measurement)),), at=actor.at())
    harness.prepare_failure = None
    third = harness.publish(candidate, 5)
    assert len(third.entities) == 3
    new = next(e for e in third.entities if e.subject.stable_id == 13)
    assert new.entity_id not in {e.entity_id for e in first.entities}
    # Exact delayed OLD epoch withdrawal cannot remove the new reused raw ID.
    actor.us = 12_600_000
    empty, _ = actor.normalize({1: []}, frame=6)
    expiry = actor.world.prepare_physical(1, empty, (), at=actor.at(), withdrawn_members=reset.withdrawn_members)
    assert harness.publish(expiry, 6).entities == ()
    # Real fresh service/World/publisher constructor for a different run; no
    # producer field relabel or forged old snapshot. Native operation is absent.
    following = OrderedActor("run-next")
    following.us = 13_000_000
    next_harness = ActualPublicationHarness(following)
    fresh, _, _, _ = settled_cohort(following, (HouseholdDescriptor("resident", "reset-A", 11),), frame=7,
                                  native_tokens=True)
    next_harness.publish(fresh, 7)
    result = actual_node_wire(actor, harness.wire + next_harness.wire, "t6-reset-reuse-expiry-next-run")
    assert [len(s["people"]) for s in result["snapshots"]] == [2, 2, 3, 0, 1]
    assert result["snapshots"][1]["people"][0]["observedAtUs"] == 10_200_000
    assert result["snapshots"][-1]["people"][0]["entityId"] == "physical:run-next:1"
    assert len(result["snapshots"][-1]["people"][0]["trail"]) == 1


def test_t6_exact_parent_four_original_resident_counterexample_refuses_atomically():
    actor = OrderedActor()
    harness = ActualPublicationHarness(actor)
    preps, original = actor.normalize({1: [(1, 0., 1, 1), (22, 3., 1, 1)],
                                      2: [(101, .03, 1, 1), (102, 3.03, 1, 1)]},
                                     identity_kind="resident", resident_uuid="synthetic-duplicate-resident", stable_id=9)
    household = HouseholdDescriptor("resident", "synthetic-duplicate-resident", 9)
    original = {key: SourceObservation(o.stamp, household) for key, o in original.items()}
    for record in original.values():
        actor.identity.observe(record)
    proofs = (actor.proof(original[(1, 1)], original[(2, 101)], "parent-original-A", 0.),
              actor.proof(original[(1, 22)], original[(2, 102)], "parent-original-B", 3.))
    for proof in proofs:
        actor.identity.link(proof, actor.at())
    final = tuple(actor.identity.link(proof, actor.at()) for proof in proofs)
    assert all(result.dto is None and result.reason == "settled_household_claim_conflict" for result in final)
    assert {o.stamp.observation_id for o in original.values()} == {
        "run-synthetic:1:0:1", "run-synthetic:1:1:22", "run-synthetic:2:0:101", "run-synthetic:2:1:102"}
    assert len({key for key, _ in final[0].household_conflict.owners}) == 2
    assert {o.stamp for _, o in final[0].household_conflict.owners} == {o.stamp for o in original.values()}
    refusal = WorldPublicationRefused.from_association_result(final[0])
    for prep in preps:
        actor.world.discard(prep)
    harness.prepare_failure = refusal
    with pytest.raises(WorldPublicationRefused) as caught:
        harness.publish(None, 2)
    assert caught.value.household_conflict is final[0].household_conflict
    assert harness.wire == [] and harness.journal == [] and harness.admissions == 0
    assert actor.world.current_snapshot() is None and actor.world._revision == 0
    directory = os.environ.get("NOESIS_XC3D_REPORT_DIR")
    if directory:
        target = Path(directory)
        assert target.is_absolute() and target.is_relative_to(Path("/mnt"))
        target.mkdir(parents=True, exist_ok=True)
        (target / "t6-exact-parent-refusal.json").write_text(json.dumps({"synthetic": True,
            "reason": refusal.reason, "ownerIssuedConflict": asdict(refusal.household_conflict),
            "originalObservations": [o.model_dump(mode="json") for prep in preps for o in prep.publication.observations],
            "releasedWire": harness.wire, "journal": harness.journal, "currentSnapshot": None}, allow_nan=False) + "\n")


def test_t6_canonical_entity_id_uniqueness_is_preserved_by_shared_rule():
    actor = OrderedActor()
    prepared, _, _, _ = actor.two_people()
    entity = prepared.publication.snapshot.entities[0]
    with pytest.raises(CanonicalWorldIdentityError) as caught:
        validate_world_identity_uniqueness((entity, entity))
    assert caught.value.collisions[0].field == "entity_id"
    assert caught.value.collisions[0].value == entity.entity_id
    with pytest.raises(ValueError, match="canonical_identity_duplicate"):
        WorldSnapshot.model_validate({**prepared.publication.snapshot.model_dump(mode="python"), "entities": (entity, entity)})
    actor.world.discard(prepared)
