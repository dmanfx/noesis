"""Explicitly synthetic admission/cohort consumer; no core World implementation."""

from dataclasses import FrozenInstanceError, replace

import pytest

from noesis.identity_association_service import (
    AssociationDTO, IdentityAssociationService, MeasurementCohort, MeasurementReference,
    ProducerAdmission, policy_identity, HouseholdClaimConflict,
    SpatialMeasurementReference, SpatialMembershipDTO, SpatialMemberReservation, SpatialMemberResult,
)
from reid.mv3dt_person_association import HouseholdDescriptor, SourceAbsence, SpatialLifecycleReceipt, CurrentVisualIngress
from tests.test_mv3dt_person_association import SOURCES, cached_proof, event, now, observation, policy, proof, removal
from tests.test_mv3dt_person_association import spatial_context, spatial_person_proof, spatial_policy
from tests.test_mv3dt_person_association import visual_proof, visual_delivery
from tests.test_mv3dt_person_association import projective_fixture
from tests.test_mv3dt_person_association import anatomical_fixture
from tests.test_mv3dt_person_association import image_region_fixture
from tests.test_mv3dt_person_association import two_cue_image_region_fixture
from tests.test_mv3dt_person_association import joint_appearance_fixture
from tests.test_mv3dt_person_association import live_image_step
from tests.test_mv3dt_person_association import spatial_shadow_update
from tests.test_mv3dt_person_association import image_vote_competition_fixture
from tests.test_mv3dt_person_association import dormant_image_fixture


def test_i135_public_joint_corroboration_and_queued_revision():
    policy,selected,p,rows,c,d=joint_appearance_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',policy,selected,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(replace(p,appearances=()),at_interval=c,delivery_context=d)
    assert len(owner.spatial_memberships(at_interval=c))==3
    result=owner.join_estimated_image_regions(p,at_interval=c,delivery_context=replace(d,sequence=d.sequence+1))
    assert result.disposition=='accepted'
    pair=next(v for v in owner.spatial_memberships(at_interval=c) if len(v.members)==2)
    assert {m.observation.member for m in pair.members}=={r.stamp.member for r in rows[:2]}
    prepared=owner.prepare_spatial_member(pair,spatial_reference(rows[0],c),at_interval=c)
    reservation=prepared.reservation
    assert reservation is not None,(prepared.reason,pair.eligibility)
    _,_,_,new,_,_=joint_appearance_fixture(2)
    owner.observe_spatial(new[0],SpatialLifecycleReceipt('next-native',new[0].stamp,'ACTIVE'))
    assert owner.commit_spatial_member(reservation,at_interval=c,
        consumer=lambda *_:pytest.fail('obsolete appearance classification')).disposition=='denied'


def test_image_regions_dormant_public_seq328_claims_current_peer_and_queued_revision():
    selected,*_=two_cue_image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    queued=[];key=[]
    def before(rows,context):
        dto=owner.spatial_memberships(at_interval=context)[0];key.append(dto.physical_key)
        queued.append(owner.prepare_spatial_physical_member(dto,spatial_reference(rows[0],context),at_interval=context).reservation)
    houses=(HouseholdDescriptor('visitor','k-claim',1001,1),HouseholdDescriptor('visitor','f-claim',1000,1))
    _,p,rows,c,d=dormant_image_fixture(owner,houses=houses,before_update=before)
    r=owner.join_estimated_image_regions(p,at_interval=c,delivery_context=d)
    dto,=owner.spatial_memberships(at_interval=c)
    assert dto.physical_key==key[0] and r.association_basis=='estimated_live_track_continuity'
    assert {e.observation.member.camera_id:e.eligible for e in dto.eligibility}=={'kitchen':True,'family-room':True}
    assert dto.household_conflict.stable_ids==(1000,1001)
    assert owner.prepare_spatial_member(dto,spatial_reference(rows[0],c),at_interval=c).reason=='settled_household_claim_conflict'
    intent=owner.prepare_spatial_physical_member(dto,spatial_reference(rows[0],c),at_interval=c).reservation
    assert intent is not None
    assert owner.commit_spatial_physical_member(intent,at_interval=c,consumer=lambda *_:'fresh-K').disposition=='committed'
    # Wall-stale body is NOT fresh association evidence. Its exact unconsumed
    # current source-local measurement can be consumed once, never retimed.
    f=owner.prepare_spatial_physical_member(dto,spatial_reference(rows[1],c),at_interval=c).reservation
    assert f is not None
    assert owner.commit_spatial_physical_member(f,at_interval=c,consumer=lambda *_:'original F').disposition=='committed'
    assert owner.prepare_spatial_physical_member(dto,spatial_reference(rows[1],c),at_interval=c).reason=='spatial_member_measurement_consumed'
    assert owner.commit_spatial_physical_member(queued[0],at_interval=c,consumer=lambda *_:pytest.fail('queued old')).disposition=='denied'
    owner.close_image_region_run()
    assert not any(e.eligible for x in owner.spatial_memberships(at_interval=c) for e in x.eligibility)


def test_r116_public_rejoin_dto_owns_original_episode_not_old_revision():
    from tests.test_mv3dt_person_association import semantic_joint_fixture,split_birth_fixture
    selected,*_=semantic_joint_fixture(competitor=None)
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    for frame in (1,2,3):
        selected,p,rows,c,d=split_birth_fixture(frame) if frame==2 else semantic_joint_fixture(frame,competitor=None)
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        owner.join_estimated_image_regions(p,at_interval=c,delivery_context=d)
        dtos=owner.spatial_memberships(at_interval=c)
        if frame==2:
            prior=dtos[0].estimated_split_custody
            old_dto=next(x for x in dtos if x.physical_key==prior.origin_key)
    dto,=dtos
    assert dto.physical_key==prior.origin_key and dto.estimated_split_custody.rejoined_key==prior.origin_key
    assert dto.estimated_split_custody.owner_deadline_monotonic_s==prior.owner_deadline_monotonic_s
    assert dto is not old_dto and dto.association_revision>old_dto.association_revision
    assert owner.prepare_spatial_member(old_dto,spatial_reference(rows[0],c),at_interval=c).disposition=='denied'
    assert owner.prepare_spatial_member(dto,spatial_reference(rows[0],c),at_interval=c).reservation is not None
    owner.close_image_region_run()
    assert not any(x.estimated_split_custody for x in owner.spatial_memberships(at_interval=c))


def test_image_regions_dormant_public_wall_delayed_originals_once_then_real_cap_expiry():
    selected,*_=two_cue_image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    _,p,rows,c,d=dormant_image_fixture(owner)
    owner.join_estimated_image_regions(p,at_interval=c,delivery_context=d)
    key=owner.spatial_memberships(at_interval=c)[0].physical_key
    d=replace(d,sequence=3,evaluation_monotonic_s=100.5)
    p=replace(p,proof_id='both-stale',originals=())
    owner.join_estimated_image_regions(p,at_interval=c,delivery_context=d)
    dto,=owner.spatial_memberships(at_interval=c)
    assert dto.physical_key==key and all(e.eligible for e in dto.eligibility)
    for row in rows:
        result=owner.prepare_spatial_member(dto,spatial_reference(row,c),at_interval=c)
        assert result.reservation is not None
        assert owner.commit_spatial_member(result.reservation,at_interval=c,consumer=lambda *_:'exact original').disposition=='committed'
        assert owner.prepare_spatial_member(dto,spatial_reference(row,c),at_interval=c).reason=='spatial_member_measurement_consumed'
    owner.join_estimated_image_regions(replace(p,proof_id='source-expired'),at_interval=c,
        delivery_context=replace(d,sequence=4,evaluation_monotonic_s=120.001))
    assert not any(e.eligible for x in owner.spatial_memberships(at_interval=c) for e in x.eligibility)


def test_image_regions_dormant_public_current_return_same_key_and_immutable_old_support():
    from tests.test_mv3dt_person_association import image_owner_capsule
    selected,*_=two_cue_image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    _,p,rows,c,d=dormant_image_fixture(owner)
    owner.join_estimated_image_regions(p,at_interval=c,delivery_context=d)
    key=owner.spatial_memberships(at_interval=c)[0].physical_key
    old=owner.ledger._members[rows[1].stamp.member].image_region_original
    _,p,updated,c,_=two_cue_image_region_fixture(frame=2)
    owner.observe_spatial(updated[1],SpatialLifecycleReceipt('actual-F-return',updated[1].stamp,'ACTIVE'))
    assert owner.ledger._members[rows[1].stamp.member].image_region_original==old
    d=replace(d,sequence=3,evaluation_monotonic_s=100.4,ingresses=(
        replace(d.ingresses[0],observation=updated[0].stamp,ingress_monotonic_s=100.1),
        replace(d.ingresses[1],observation=updated[1].stamp,ingress_monotonic_s=100.2)))
    p=replace(p,originals=tuple(replace(o,ingress_monotonic_s=t) for o,t in zip(p.originals,(100.1,100.2))),
        owner_availability=tuple(image_owner_capsule(row,t) for row,t in zip(updated,(100.1,100.2))))
    owner.join_estimated_image_regions(p,at_interval=c,delivery_context=d)
    dto,=owner.spatial_memberships(at_interval=c)
    assert dto.physical_key==key and all(e.eligible for e in dto.eligibility)


def vote_quarantine_step(owner,frame,*,competitor='supported',join=True):
    selected,proof,rows,context,delivery=two_cue_image_region_fixture(frame=frame,competitor=competitor)
    houses=(HouseholdDescriptor('visitor','visitor-kitchen',1001,1),
        HouseholdDescriptor('visitor','visitor-family',1000,1),HouseholdDescriptor('visitor','visitor-seated',1002,1))
    rows=tuple(replace(r,household=h) for r,h in zip(rows,houses))
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    result=owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery) if join else None
    return result,rows,context,proof,delivery


def test_image_regions_vote_quarantine_public_two_people_keep_two_keys_not_three():
    selected,_,_,_,_=two_cue_image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    _,old,context,_,_=vote_quarantine_step(owner,1,competitor='incompatible')
    seed=next(d for d in owner.spatial_memberships(at_interval=context) if len(d.members)==2)
    queued=owner.prepare_spatial_physical_member(seed,spatial_reference(old[0],context),at_interval=context).reservation
    result,rows,context,_,_=vote_quarantine_step(owner,2)
    dtos=owner.spatial_memberships(at_interval=context)
    assert len(dtos)==2,[(d.physical_key,len(d.members)) for d in dtos]
    pair=next(d for d in dtos if len(d.members)==2)
    assert pair.physical_key==seed.physical_key and pair.association_basis=='estimated_live_track_continuity'
    eligibility={e.observation.member:e for e in pair.eligibility}
    assert eligibility[rows[0].stamp.member].reason=='image_region_ambiguous_member'
    assert not eligibility[rows[0].stamp.member].eligible and eligibility[rows[1].stamp.member].eligible
    assert owner.prepare_spatial_physical_member(pair,spatial_reference(rows[0],context),at_interval=context).reason=='image_region_ambiguous_member'
    assert pair.household_conflict.stable_ids==(1000,1001)
    assert owner.prepare_spatial_member(pair,spatial_reference(rows[1],context),at_interval=context).reason=='settled_household_claim_conflict'
    intent=owner.prepare_spatial_physical_member(pair,spatial_reference(rows[1],context),at_interval=context).reservation
    assert intent is not None and owner.commit_spatial_physical_member(intent,at_interval=context,consumer=lambda *_:'unambiguous-peer').disposition=='committed'
    distinct=next(d for d in dtos if len(d.members)==1)
    assert distinct.members[0].observation==rows[2].stamp and distinct.eligibility[0].eligible
    assert owner.commit_spatial_physical_member(queued,at_interval=context,consumer=lambda *_:pytest.fail('old queued grant')).disposition=='denied'
    transition=owner.image_region_decision_diagnostics()['transitions'][-1]
    assert transition['authority_revision_after']>transition['authority_revision_before']
    assert next(m for m in transition['members'] if m['member']['raw_id']==7)['vote_block_after']['frame_id']==2


def test_image_regions_vote_quarantine_new_observation_stays_blocked_until_current_unique():
    selected,_,_,_,_=two_cue_image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    vote_quarantine_step(owner,1,competitor='incompatible')
    _,old,context,_,_=vote_quarantine_step(owner,2)
    pair=next(d for d in owner.spatial_memberships(at_interval=context) if len(d.members)==2);key=pair.physical_key
    block=owner.ledger._members[old[0].stamp.member].image_region_vote_blocked_stamp
    revision=owner.ledger.authority_revision
    _,rows,context,proof,delivery=vote_quarantine_step(owner,3,competitor='incompatible',join=False)
    before=next(d for d in owner.spatial_memberships(at_interval=context) if len(d.members)==2)
    pending=owner.ledger._members[rows[0].stamp.member]
    assert pending.image_region_vote_blocked_stamp==block and pending.image_region_vote_pending
    assert not next(e for e in before.eligibility if e.observation==rows[0].stamp).eligible
    assert owner.ledger.authority_revision>revision
    owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery)
    after=next(d for d in owner.spatial_memberships(at_interval=context) if len(d.members)==2)
    assert after.physical_key==key and all(e.eligible for e in after.eligibility)
    assert owner.ledger._members[rows[0].stamp.member].image_region_vote_blocked_stamp is None
    assert owner.commit_spatial_physical_member(owner.prepare_spatial_physical_member(after,spatial_reference(rows[0],context),at_interval=context).reservation,
        at_interval=context,consumer=lambda *_:'genuine-unique-current').disposition=='committed'
    # Invalid original binding may never masquerade as ordinary no-vote.
    _,rows,context,_,_=vote_quarantine_step(owner,4)
    pair=next(d for d in owner.spatial_memberships(at_interval=context) if len(d.members)==2)
    bad=replace(spatial_reference(rows[0],context),units='pixels')
    assert owner.prepare_spatial_physical_member(pair,bad,at_interval=context).reason=='spatial_original_measurement_binding'


@pytest.mark.parametrize('mode',['unknown','missing'])
def test_image_regions_vote_quarantine_does_not_clear_on_nonunique_or_cached_proof(mode):
    selected,_,_,_,_=two_cue_image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    vote_quarantine_step(owner,1,competitor='incompatible');vote_quarantine_step(owner,2)
    _,rows,context,proof,delivery=vote_quarantine_step(owner,3,competitor='missing',join=False)
    if mode=='missing':proof=replace(proof,originals=())
    owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery)
    pair=next(d for d in owner.spatial_memberships(at_interval=context) if len(d.members)==2)
    e=next(e for e in pair.eligibility if e.observation==rows[0].stamp)
    assert not e.eligible and e.reason=='image_region_ambiguous_member'


def test_image_regions_vote_quarantine_public_both_ambiguous_no_current_reservations():
    selected,_,_,_,_=image_vote_competition_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    keys=None
    for frame in (1,2):
        _,proof,rows,context,delivery=image_vote_competition_fixture(frame,crowded=frame==2)
        for i,r in enumerate(rows):
            row=replace(r,household=HouseholdDescriptor('visitor','visitor-'+str(i),1000+i,1))
            owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
        owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery)
        dtos=owner.spatial_memberships(at_interval=context)
        if keys is None:keys={d.physical_key for d in dtos};assert len(keys)==2
        else:
            assert {d.physical_key for d in dtos}==keys
            for dto in dtos:
                assert dto.household_conflict is not None
                assert all(not e.eligible and e.reason=='image_region_ambiguous_member' for e in dto.eligibility)
                for member in dto.members:
                    row=next(r for r in rows if r.stamp==member.observation)
                    assert owner.prepare_spatial_physical_member(dto,spatial_reference(row,context),at_interval=context).reason=='image_region_ambiguous_member'


def test_image_regions_vote_quarantine_ordered_classification_invalidates_queued_peer_grant():
    selected,_,_,_,_=two_cue_image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    vote_quarantine_step(owner,1,competitor='incompatible')
    _,rows,context,proof,delivery=vote_quarantine_step(owner,2)
    pair=next(d for d in owner.spatial_memberships(at_interval=context) if len(d.members)==2)
    intent=owner.prepare_spatial_physical_member(pair,spatial_reference(rows[1],context),at_interval=context).reservation
    assert intent is not None
    before=owner.ledger.authority_revision
    owner.join_estimated_image_regions(replace(proof,proof_id='later-current-classification'),at_interval=context,
        delivery_context=replace(delivery,sequence=3))
    assert owner.ledger.authority_revision>before
    assert owner.commit_spatial_physical_member(intent,at_interval=context,
        consumer=lambda *_:pytest.fail('queued stale classification invoked')).disposition=='denied'
    pair=next(d for d in owner.spatial_memberships(at_interval=context) if len(d.members)==2)
    assert next(e for e in pair.eligibility if e.observation==rows[0].stamp).reason=='image_region_ambiguous_member'
    assert next(e for e in pair.eligibility if e.observation==rows[1].stamp).eligible


def test_image_regions_public_present_partial_hypothesis_keeps_quarantine_current_only():
    selected,_,_,_,_=two_cue_image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    houses=(HouseholdDescriptor('visitor','visitor-kitchen',1001,1),HouseholdDescriptor('visitor','visitor-family',1000,1))
    key=None
    for frame in (1,2,3,4):
        _,proof,rows,context,delivery=two_cue_image_region_fixture(frame=frame,competitor=None if frame==1 else 'missing')
        rows=tuple(replace(r,household=houses[i]) if i<2 else r for i,r in enumerate(rows))
        if frame>=3:
            proof=replace(proof,originals=tuple(replace(o,patches=tuple(p for p in o.patches if p.anatomy=='pose_shoulder_support')) for o in proof.originals))
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('current:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        result=owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery)
        dto=next(d for d in owner.spatial_memberships(at_interval=context) if len(d.members)==2)
        if key is None:key=dto.physical_key
        assert dto.physical_key==key and all(e.eligible for e in dto.eligibility)
        assert dto.household_conflict.stable_ids==(1000,1001)
        if frame>=3:
            assert result.association_basis=='estimated_live_track_continuity'
            assert all(s.original_stamp.frame_id==1 for s in dto.image_region_support)
            ref=spatial_reference(rows[0],context)
            assert owner.prepare_spatial_member(dto,ref,at_interval=context).reason=='settled_household_claim_conflict'
            intent=owner.prepare_spatial_physical_member(dto,ref,at_interval=context).reservation
            assert intent is not None
            assert owner.commit_spatial_physical_member(intent,at_interval=context,consumer=lambda *_:'current-measurement-only').disposition=='committed'


def test_image_regions_exact_r19_stale_vote_has_no_native_owner_expiry_authority():
    selected,proof,old,context,delivery=two_cue_image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    for r in old:owner.observe_spatial(r,SpatialLifecycleReceipt('seed:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery)
    _,proof,rows,context,delivery=two_cue_image_region_fixture(frame=2)
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('current:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    ages=(.2591420286335051,.3584226416423917)
    delivery=replace(delivery,evaluation_monotonic_s=100.5,
        ingresses=tuple(replace(i,ingress_monotonic_s=100.5-age) for i,age in zip(delivery.ingresses,ages)))
    proof=replace(proof,originals=(replace(proof.originals[0],ingress_monotonic_s=delivery.ingresses[0].ingress_monotonic_s),))
    result=owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery)
    # A native ACTIVE stamp is not a current owner expiry/source-stop proof.
    # Until that real producer input exists, no age waiver is authorized.
    assert result.reason=='image_region_unavailable_or_ambiguous'
    assert result.withdrawn_members
    family=next(d for d in owner.spatial_memberships(at_interval=context) if d.members[0].observation==rows[1].stamp)
    assert not family.eligibility[0].eligible and family.eligibility[0].reason=='current_visual_stale_member_delivery'
    assert owner.prepare_spatial_member(family,spatial_reference(rows[1],context),at_interval=context).disposition=='denied'
    original=owner.ledger._visual_ingresses[rows[1].stamp.member]
    assert original.observation==rows[1].stamp and original.ingress_monotonic_s==100.5-ages[1]


@pytest.mark.parametrize('divergent',[False,True])
def test_image_regions_shadow_handoff_retains_key_no_shadow_vote_real_return_only(divergent):
    selected,proof,old,context,delivery=two_cue_image_region_fixture()
    if divergent:
        houses=(HouseholdDescriptor('visitor','visitor-kitchen',1001,1),HouseholdDescriptor('visitor','visitor-family',1000,1))
        old=tuple(replace(r,household=h) for r,h in zip(old,houses))
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    for r in old: owner.observe_spatial(r,SpatialLifecycleReceipt('seed:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery)
    seed,=owner.spatial_memberships(at_interval=context); key=seed.physical_key
    result=owner.update_spatial_lifecycle(spatial_shadow_update(old[0],frame=2,time=10.05,media=5.05))
    assert result.withdrawn_members==() and result.retired_keys==()
    assert result.association_basis=='estimated_live_track_continuity'
    _,proof,rows,context,delivery=two_cue_image_region_fixture(frame=3)
    if divergent: rows=tuple(replace(r,household=h) for r,h in zip(rows,houses))
    peer=rows[1]
    owner.observe_spatial(peer,SpatialLifecycleReceipt('peer',peer.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(replace(proof,cohort=owner.image_region_active_cohort(),originals=proof.originals[1:]),
        at_interval=context,delivery_context=visual_delivery((peer,),sequence=3,evaluation=100.1))
    dto,=owner.spatial_memberships(at_interval=context)
    assert dto.physical_key==key and dto.association_basis=='estimated_live_track_continuity'
    eligibility={e.observation.member:e.eligible for e in dto.eligibility}
    assert eligibility=={old[0].stamp.member:False,peer.stamp.member:True}
    assert owner.prepare_spatial_physical_member(dto,spatial_reference(old[0],context),at_interval=context).disposition=='denied'
    ref=spatial_reference(peer,context)
    if divergent:
        assert dto.household_conflict.stable_ids==(1000,1001)
        assert owner.prepare_spatial_member(dto,ref,at_interval=context).reason=='settled_household_claim_conflict'
    prepare=owner.prepare_spatial_physical_member if divergent else owner.prepare_spatial_member
    commit=owner.commit_spatial_physical_member if divergent else owner.commit_spatial_member
    intent=prepare(dto,ref,at_interval=context).reservation
    assert intent is not None
    assert commit(intent,at_interval=context,consumer=lambda *_:'fresh-peer').disposition=='committed'
    _,proof,rows,context,delivery=two_cue_image_region_fixture(frame=4)
    if divergent: rows=tuple(replace(r,household=h) for r,h in zip(rows,houses))
    for r in rows: owner.observe_spatial(r,SpatialLifecycleReceipt('return:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(replace(proof,originals=()),at_interval=context,delivery_context=delivery)
    dto,=owner.spatial_memberships(at_interval=context)
    assert dto.physical_key==key and all(e.eligible for e in dto.eligibility)
    assert all(s.original_stamp.frame_id==1 for s in dto.image_region_support)
    for r in rows:
        intent=prepare(dto,spatial_reference(r,context),at_interval=context).reservation
        assert intent is not None
        assert commit(intent,at_interval=context,consumer=lambda *_:'genuine-return').disposition=='committed'


def test_image_regions_both_shadows_never_vote_or_renew_originals():
    selected,proof,old,context,delivery=two_cue_image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    for r in old: owner.observe_spatial(r,SpatialLifecycleReceipt('seed:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery)
    seed,=owner.spatial_memberships(at_interval=context)
    for r in old: owner.update_spatial_lifecycle(spatial_shadow_update(r,frame=2,time=10.05,media=5.05,token='shadow:'+str(r.stamp.member.source_id)))
    _,_,current,context,_=two_cue_image_region_fixture(frame=2)
    dto,=owner.spatial_memberships(at_interval=context)
    assert dto.physical_key==seed.physical_key and not any(e.eligible for e in dto.eligibility)
    assert all(m.observation.stamp.frame_id==1 for m in owner.ledger._members.values())
    for r in old:
        assert owner.prepare_spatial_physical_member(dto,spatial_reference(r,context),at_interval=context).disposition=='denied'


def test_image_regions_live_service_original_pose_provenance_current_measurement_only():
    selected,_,_,_,_=image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    # Service uses the same public join without an invented pose/XYZ proof.
    _,seed,seed_context,_=live_image_step(owner,1,mode='fresh')
    for frame in range(2,7): result,rows,context,_=live_image_step(owner,frame)
    dto,=owner.spatial_memberships(at_interval=context)
    assert result.reason==dto.association_basis=='estimated_live_track_continuity'
    assert all(e.eligible for e in dto.eligibility)
    assert all(s.original_stamp.frame_id==1 and s.original_ingress_monotonic_s==100.
        and s.current_stamp.frame_id==6 and tuple(x.frame_id for x in s.native_seed_cohort)==(1,1)
        for s in dto.image_region_support)
    old=owner.prepare_spatial_member(dto,spatial_reference(seed[0],seed_context),at_interval=context)
    assert old.disposition=='denied'
    intent=owner.prepare_spatial_member(dto,spatial_reference(rows[0],context),at_interval=context).reservation
    assert intent is not None
    assert owner.commit_spatial_member(intent,at_interval=context,consumer=lambda *_:'current-only').disposition=='committed'


def test_image_regions_live_service_divergent_labels_still_strictly_quarantined():
    selected,_,_,_,_=image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    houses=(HouseholdDescriptor('visitor','visitor-kitchen',1004,1),HouseholdDescriptor('visitor','visitor-family',1000,1))
    live_image_step(owner,1,mode='fresh',houses=houses)
    for frame in range(2,7): _,rows,context,_=live_image_step(owner,frame,houses=houses)
    dto,=owner.spatial_memberships(at_interval=context)
    assert dto.household_claim is None and dto.household_conflict.stable_ids==(1000,1004)
    ref=spatial_reference(rows[0],context)
    assert owner.prepare_spatial_member(dto,ref,at_interval=context).reason=='settled_household_claim_conflict'
    intent=owner.prepare_spatial_physical_member(dto,ref,at_interval=context).reservation
    assert intent is not None
    assert owner.commit_spatial_physical_member(intent,at_interval=context,consumer=lambda *_:'quarantine').disposition=='committed'


@pytest.mark.parametrize('value',[1,0,'true',None])
def test_image_regions_live_service_requires_exact_boolean(value):
    selected,_,_,_,_=image_region_fixture()
    with pytest.raises(ValueError,match='image_region_live_track_boolean_required'):
        IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=value)


def test_image_regions_lifecycle_retained_competitor_causal_public_service_and_unknown_preservation():
    selected,proof,old,context,delivery=image_region_fixture(competitor='incompatible')
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    for r in old: owner.observe_spatial(r,SpatialLifecycleReceipt('n:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery)
    _,proof,rows,context,delivery=image_region_fixture(frame=9)
    for r in rows: owner.observe_spatial(r,SpatialLifecycleReceipt('n:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    assert owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery).reason=='current_visual_complete_cohort_required'
    # Stale ACTIVE is UNKNOWN rather than silently absent; literal accessor is read-only.
    revision=owner.ledger.authority_revision
    cohort=owner.image_region_active_cohort()
    assert old[2].stamp in cohort and owner.ledger.authority_revision==revision
    result=owner.join_estimated_image_regions(replace(proof,cohort=cohort,proof_id='full-cohort'),at_interval=context,
        delivery_context=replace(delivery,sequence=10))
    assert result.reason=='image_region_unavailable_or_ambiguous'
    assert owner.update_spatial_lifecycle(spatial_shadow_update(old[2])).reason=='retained_unmeasured'
    assert owner.image_region_active_cohort()==tuple(sorted((r.stamp for r in rows),key=lambda s:s.member))
    result=owner.join_estimated_image_regions(replace(proof,proof_id='classified-full-cohort'),at_interval=context,
        delivery_context=replace(delivery,sequence=11))
    assert result.disposition=='accepted' and result.reason=='estimated_current_image_regions'


def test_image_regions_lifecycle_joined_shadow_remaining_current_same_key_then_real_rejoin():
    selected,proof,old,context,delivery=image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=False)
    for r in old: owner.observe_spatial(r,SpatialLifecycleReceipt('n:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery)
    dto,=owner.spatial_memberships(at_interval=context); key=dto.physical_key
    stale_intent=owner.prepare_spatial_member(dto,spatial_reference(old[0],context),at_interval=context).reservation
    _,proof,rows,context,delivery=image_region_fixture(frame=9)
    owner.observe_spatial(rows[0],SpatialLifecycleReceipt('n:'+rows[0].stamp.observation_id,rows[0].stamp,'ACTIVE'))
    result=owner.update_spatial_lifecycle(spatial_shadow_update(old[1]))
    assert result.withdrawn_members==(old[1].stamp.member,)
    assert owner.commit_spatial_member(stale_intent,at_interval=context,consumer=lambda *_:pytest.fail('stale grant')).disposition=='denied'
    proof=replace(proof,cohort=owner.image_region_active_cohort(),originals=proof.originals[:1])
    owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=replace(delivery,ingresses=delivery.ingresses[:1]))
    dtos=owner.spatial_memberships(at_interval=context)
    live=next(d for d in dtos if d.physical_key==key); shadow=next(d for d in dtos if d.physical_key!=key)
    assert live.association_basis=='source_local' and all(e.eligible for e in live.eligibility)
    assert not any(e.eligible for e in shadow.eligibility)
    assert shadow.members[0].observation==old[1].stamp
    assert owner.ledger._members[old[1].stamp.member].spatial_lifecycle.native_state=='ACTIVE'
    assert all(not m.image_region_native_seed and not m.image_region_front_confirmed for m in owner.ledger._members.values())
    intent=owner.prepare_spatial_member(live,spatial_reference(rows[0],context),at_interval=context).reservation
    assert intent is not None
    assert owner.commit_spatial_member(intent,at_interval=context,consumer=lambda *_:'current source-local').disposition=='committed'
    assert owner.prepare_spatial_member(shadow,spatial_reference(old[1],context),at_interval=context).disposition=='denied'
    _,fresh,rows,context,delivery=image_region_fixture(frame=10)
    for r in rows: owner.observe_spatial(r,SpatialLifecycleReceipt('n:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    before=owner.spatial_memberships(at_interval=context)
    assert not next(e.eligible for d in before for e in d.eligibility if e.observation==rows[1].stamp)
    assert owner.join_estimated_image_regions(fresh,at_interval=context,delivery_context=delivery).disposition=='accepted'
    dto,=owner.spatial_memberships(at_interval=context)
    assert dto.physical_key==key and all(e.eligible for e in dto.eligibility)


def test_image_regions_lifecycle_apis_require_explicit_image_factory():
    selected,_,rows,_,_=image_region_fixture()
    owner=IdentityAssociationService.for_current_visual('run-a',spatial_policy())
    assert owner.update_spatial_lifecycle(spatial_shadow_update(rows[0])).reason=='estimated_image_regions_opt_in_required'
    with pytest.raises(ValueError,match='estimated_image_regions_opt_in_required'): owner.image_region_active_cohort()


def test_image_regions_lifecycle_new_current_pose_unknown_competitor_is_not_shadow():
    selected,proof,old,context,delivery=image_region_fixture(competitor='incompatible')
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    for r in old: owner.observe_spatial(r,SpatialLifecycleReceipt('n:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery)
    owner.update_spatial_lifecycle(spatial_shadow_update(old[2],frame=2,time=10.05,media=5.05))
    _,proof,rows,context,delivery=image_region_fixture(frame=3,competitor='missing')
    # This is a NEW genuine current native member, not an owner shadow receipt.
    newcomer=replace(rows[2],stamp=replace(rows[2].stamp,member=replace(rows[2].stamp.member,raw_id=11),observation_id='new-current-unknown'))
    rows=rows[:2]+(newcomer,)
    for r in rows: owner.observe_spatial(r,SpatialLifecycleReceipt('n:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    proof=replace(proof,cohort=owner.image_region_active_cohort())
    delivery=visual_delivery(rows,sequence=3,evaluation=100.1)
    assert newcomer.stamp in proof.cohort and old[2].stamp not in proof.cohort
    result=owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery)
    assert result.reason=='image_region_unavailable_or_ambiguous'
    assert not any(len(d.members)>1 and all(e.eligible for e in d.eligibility) for d in owner.spatial_memberships(at_interval=context))


def test_image_regions_lifecycle_joined_divergent_labels_preserved_through_shadow_rekey():
    selected,proof,rows,context,delivery=image_region_fixture()
    houses=(HouseholdDescriptor('visitor','visitor-kitchen',1004,1),HouseholdDescriptor('visitor','visitor-family',1000,1))
    rows=tuple(replace(r,household=h) for r,h in zip(rows,houses))
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=False)
    for r in rows: owner.observe_spatial(r,SpatialLifecycleReceipt('n:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery)
    before,=owner.spatial_memberships(at_interval=context)
    assert before.household_conflict.stable_ids==(1000,1004)
    owner.update_spatial_lifecycle(spatial_shadow_update(rows[1],frame=2,time=10.05,media=5.05))
    assert {o.stamp.member:o.household for _,o in owner.ledger.household_claim_owners()}=={r.stamp.member:r.household for r in rows}
    assert owner.ledger._members[rows[0].stamp.member].person_key==before.physical_key
    assert owner.ledger._members[rows[1].stamp.member].person_key!=before.physical_key
    assert not owner.ledger._members[rows[1].stamp.member].image_region_participant


def image_rejoin_waiting_owner():
    from reid.mv3dt_person_association import LifecycleRetention
    selected,_,_,_,_=image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    _,old,_,_=live_image_step(owner,1,mode='fresh')
    # Genuine absence/grace return still requires CROSS-VIEW rejoin proof,
    # unlike an explicitly shadowed source-local singleton's fresh observation.
    grace=LifecycleRetention('real-native-grace',old[1].stamp,old[1].stamp.member,'inactive',.35)
    owner.withdraw(removal(old[1],frame=2,time=10.05,media=5.05,retention=grace))
    _,rows,context,_=live_image_step(owner,3,mode='missing')
    dtos=owner.spatial_memberships(at_interval=context)
    waiting=next(d for d in dtos if d.members[0].observation==rows[1].stamp)
    return owner,waiting,rows,context


def test_image_regions_shadowed_single_source_real_current_measurement_is_local_not_cross_rejoin():
    selected,proof,rows,_,_=image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected)
    first=rows[1]
    owner.observe_spatial(first,SpatialLifecycleReceipt('first-real',first.stamp,'ACTIVE'))
    context=spatial_context((first,))
    owner.join_estimated_image_regions(replace(proof,cohort=(first.stamp,),originals=proof.originals[1:]),
        at_interval=context,delivery_context=visual_delivery((first,)))
    before,=owner.spatial_memberships(at_interval=context)
    assert before.eligibility[0].eligible and before.association_basis=='source_local'
    owner.update_spatial_lifecycle(spatial_shadow_update(first,frame=2,time=10.05,media=5.05))
    _,proof,rows,_,_=image_region_fixture(frame=3)
    current=rows[1]
    owner.observe_spatial(current,SpatialLifecycleReceipt('later-real',current.stamp,'ACTIVE'))
    context=spatial_context((current,))
    owner.join_estimated_image_regions(replace(proof,cohort=owner.image_region_active_cohort(),originals=proof.originals[1:]),
        at_interval=context,delivery_context=visual_delivery((current,),sequence=3,evaluation=100.1))
    dto,=owner.spatial_memberships(at_interval=context)
    assert dto.physical_key==before.physical_key and dto.association_basis=='source_local'
    assert dto.eligibility[0].eligible,dto.eligibility
    member=owner.ledger._members[current.stamp.member]
    assert not member.image_region_front_confirmed and member.image_region_native_seed==()
    intent=owner.prepare_spatial_member(dto,spatial_reference(current,context),at_interval=context).reservation
    assert intent is not None
    assert owner.commit_spatial_member(intent,at_interval=context,consumer=lambda *_:'actual-current-local').disposition=='committed'


def test_image_regions_former_cross_member_reactivates_separate_local_key_until_actual_pair_proof():
    selected,proof,old,context,delivery=image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=False)
    for row in old: owner.observe_spatial(row,SpatialLifecycleReceipt('seed:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery)
    seed,=owner.spatial_memberships(at_interval=context)
    owner.update_spatial_lifecycle(spatial_shadow_update(old[1],frame=2,time=10.05,media=5.05))
    _,fresh,rows,context,delivery=image_region_fixture(frame=3)
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('real:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(replace(fresh,originals=()),at_interval=context,delivery_context=delivery)
    dtos=owner.spatial_memberships(at_interval=context)
    assert len(dtos)==2 and all(d.association_basis=='source_local' and all(e.eligible for e in d.eligibility) for d in dtos)
    kitchen=next(d for d in dtos if d.members[0].observation==rows[0].stamp)
    family=next(d for d in dtos if d.members[0].observation==rows[1].stamp)
    assert kitchen.physical_key==seed.physical_key and family.physical_key!=seed.physical_key
    assert all(not m.image_region_front_confirmed and not m.image_region_native_seed for m in owner.ledger._members.values())
    intent=owner.prepare_spatial_member(family,spatial_reference(rows[1],context),at_interval=context).reservation
    assert intent is not None and owner.commit_spatial_member(intent,at_interval=context,consumer=lambda *_:'local-current').disposition=='committed'
    # Only the real NEW pair's current originals can establish shared membership.
    result=owner.join_estimated_image_regions(replace(fresh,proof_id='actual-new-pair'),at_interval=context,
        delivery_context=replace(delivery,sequence=4))
    assert result.disposition=='accepted'
    joined,=owner.spatial_memberships(at_interval=context)
    assert joined.physical_key==seed.physical_key and joined.association_basis=='estimated_current_image_regions'


def test_image_regions_existing_current_true_pair_keeps_estimated_key_with_unknown_third_person():
    selected,proof,old,context,delivery=two_cue_image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    for r in old: owner.observe_spatial(r,SpatialLifecycleReceipt('seed:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery)
    seed,=owner.spatial_memberships(at_interval=context)
    _,proof,rows,context,delivery=two_cue_image_region_fixture(frame=2,competitor='missing')
    for r in rows: owner.observe_spatial(r,SpatialLifecycleReceipt('real:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    result=owner.join_estimated_image_regions(proof,at_interval=context,delivery_context=delivery)
    assert result.reason=='estimated_live_track_continuity',result
    dtos=owner.spatial_memberships(at_interval=context)
    pair=next(d for d in dtos if len(d.members)==2); third=next(d for d in dtos if len(d.members)==1)
    assert pair.physical_key==seed.physical_key and third.physical_key!=seed.physical_key
    assert pair.association_basis=='estimated_live_track_continuity'
    assert all(e.eligible for d in dtos for e in d.eligibility)
    # UNKNOWN membership competition is not a fabricated pose. Each grant
    # still needs its independent genuine current source measurement reference.
    for dto,row in ((pair,rows[0]),(third,rows[2])):
        intent=owner.prepare_spatial_member(dto,spatial_reference(row,context),at_interval=context).reservation
        assert intent is not None
        assert owner.commit_spatial_member(intent,at_interval=context,consumer=lambda *_:'independent-current').disposition=='committed'
    diagnostic=owner.image_region_decision_diagnostics()['transitions'][-1]
    unknown=next(p for p in diagnostic['pairs'] if p['right']['raw_id']==9)
    assert unknown['state'] is None and unknown['possible']


def test_image_regions_rejoin_wait_is_explicit_false_eligibility_and_denied_vote():
    owner,dto,rows,context=image_rejoin_waiting_owner()
    assert not dto.eligibility[0].eligible and dto.eligibility[0].reason=='image_region_rejoin_proof_required'
    assert owner.ledger.spatial_member_error(rows[1].stamp,context)=='current_rejoin_proof_required'
    result=owner.prepare_spatial_member(dto,spatial_reference(rows[1],context),at_interval=context)
    assert result.disposition=='denied' and result.reason=='image_region_rejoin_proof_required' and result.reservation is None


@pytest.mark.parametrize('mutation',['owner','origin','units','calibration','transform','interval','evidence'])
def test_image_regions_rejoin_wait_never_masks_invalid_original_reference(mutation):
    owner,dto,rows,context=image_rejoin_waiting_owner()
    ref=spatial_reference(rows[1],context)
    if mutation=='owner': ref=replace(ref,owner=rows[0].stamp.member)
    elif mutation=='origin': ref=replace(ref,origin='fabricated')
    elif mutation=='units': ref=replace(ref,units='pixels')
    elif mutation=='calibration': ref=replace(ref,calibration_revision='invented')
    elif mutation=='transform': ref=replace(ref,transform_id='invented')
    elif mutation=='interval': ref=replace(ref,acquisition_interval=replace(ref.acquisition_interval,acquisition_min_s=10.))
    else: ref=replace(ref,evidence_id='')
    reason=owner.prepare_spatial_member(dto,ref,at_interval=context).reason
    assert reason=='spatial_reference_member' if mutation=='owner' else reason=='spatial_original_measurement_binding'


@pytest.mark.parametrize('threshold',[True,False,0.,-1.,float('nan'),float('inf'),'33'])
def test_semantic_joint_factory_threshold_is_explicit_finite_positive(threshold):
    from tests.test_mv3dt_person_association import semantic_joint_fixture
    selected,*_=semantic_joint_fixture()
    with pytest.raises(ValueError,match='semantic_joint_threshold_opt_in_required'):
        IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,
            live_track_continuity=True,semantic_joint_max_mean_px=threshold)


def test_semantic_joint_service_current_only_claims_queued_revision_and_missing_raw_custody():
    from tests.test_mv3dt_person_association import semantic_joint_fixture
    selected,*_=semantic_joint_fixture()
    with pytest.raises(ValueError,match='semantic_joint_threshold_opt_in_required'):
        IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,semantic_joint_max_mean_px=5.)
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,
        live_track_continuity=True,contradictory_cue_quarantine=True,semantic_joint_max_mean_px=5.)
    seed=None;queued=None
    for frame in (1,2,3):
        _,p,rows,c,d=semantic_joint_fixture(frame,mixed=True)
        rows=tuple(replace(r,household=HouseholdDescriptor('visitor','visitor-'+str(i),1000+i,1)) for i,r in enumerate(rows))
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        if frame==2:p=replace(p,originals=p.originals[1:])  # honest missing RAW K pose, not native absence
        result=owner.join_estimated_image_regions(p,at_interval=c,delivery_context=d)
        assert result.reason=='estimated_semantic_joint_correspondence'
        dto=next(x for x in owner.spatial_memberships(at_interval=c) if len(x.members)==2)
        assert dto.association_basis=='estimated_semantic_joint_correspondence'
        assert dto.household_conflict is not None and dto.household_claim is None
        assert {m.household.stable_id for m in dto.members}=={1000,1001}
        if frame==1:
            seed=dto.physical_key
            queued=owner.prepare_spatial_physical_member(dto,spatial_reference(rows[1],c),at_interval=c).reservation
            assert queued is not None
        else:
            assert dto.physical_key==seed
            if frame==2:
                assert all(s.original_stamp.frame_id==1 for s in dto.image_region_support)
                callbacks=[]
                assert owner.commit_spatial_physical_member(queued,at_interval=c,consumer=lambda *_:callbacks.append(1)).disposition=='denied'
                assert callbacks==[]
            current=owner.prepare_spatial_physical_member(dto,spatial_reference(rows[1],c),at_interval=c).reservation
            assert current is not None
            assert owner.commit_spatial_physical_member(current,at_interval=c,consumer=lambda *_:'genuine-current-only').disposition=='committed'
        assert owner.prepare_spatial_member(dto,spatial_reference(rows[0],c),at_interval=c).disposition=='denied'


def test_image_regions_contradictory_cue_public_votes_queue_and_unique_recovery():
    from tests.test_mv3dt_person_association import contradictory_cue_fixture
    selected,*_=contradictory_cue_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,
        live_track_continuity=True,contradictory_cue_quarantine=True)
    queued=None;seed=None;originals=None
    for frame in (1,2,3,4):
        selected,p,rows,c,d=contradictory_cue_fixture(frame,mixed=frame in (2,3))
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        if frame==3:
            old=next(x for x in owner.spatial_memberships(at_interval=c) if x.physical_key==seed)
            assert not any(e.eligible for e in old.eligibility)  # observe never clears block
        result=owner.join_estimated_image_regions(p,at_interval=c,delivery_context=d)
        dto=next(x for x in owner.spatial_memberships(at_interval=c) if len(x.members)==2)
        if frame==1:
            seed=dto.physical_key;originals=tuple(v.original_stamp for v in dto.image_region_support)
            queued=owner.prepare_spatial_member(dto,spatial_reference(rows[0],c),at_interval=c).reservation
            assert queued is not None
        elif frame in (2,3):
            assert dto.physical_key==seed and dto.association_basis=='estimated_image_region_conflict_custody'
            assert tuple(v.original_stamp for v in dto.image_region_support)==originals
            assert all(not e.eligible and e.reason=='image_region_contradictory_cue_member' for e in dto.eligibility)
            assert all(owner.prepare_spatial_member(dto,spatial_reference(r,c),at_interval=c).reason==
                'image_region_contradictory_cue_member' for r in rows[:2])
            if frame==2:
                called=[]
                assert owner.commit_spatial_member(queued,at_interval=c,consumer=lambda *_:called.append(1)).disposition=='denied'
                assert called==[]
            distinct=next(x for x in owner.spatial_memberships(at_interval=c) if len(x.members)==1)
            assert distinct.physical_key!=seed and distinct.eligibility[0].eligible
        else:
            assert dto.physical_key==seed and all(e.eligible for e in dto.eligibility)
            assert all(v.original_stamp.frame_id==4 for v in dto.image_region_support)
            reservation=owner.prepare_spatial_member(dto,spatial_reference(rows[0],c),at_interval=c).reservation
            assert owner.commit_spatial_member(reservation,at_interval=c,consumer=lambda *_:'fresh-only').disposition=='committed'


@pytest.mark.parametrize('value',[None,0,1,'true'])
def test_image_regions_contradictory_cue_factory_exact_boolean(value):
    from tests.test_mv3dt_person_association import contradictory_cue_fixture
    selected,*_=contradictory_cue_fixture()
    with pytest.raises(ValueError,match='image_region_contradictory_cue_opt_in_required'):
        IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,
            live_track_continuity=True,contradictory_cue_quarantine=value)


def test_image_regions_contradictory_cue_factory_requires_live_and_preserves_divergent_claims():
    from tests.test_mv3dt_person_association import contradictory_cue_fixture
    selected,*_=contradictory_cue_fixture()
    with pytest.raises(ValueError,match='image_region_contradictory_cue_opt_in_required'):
        IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,contradictory_cue_quarantine=True)
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,
        live_track_continuity=True,contradictory_cue_quarantine=True)
    for frame in (1,2):
        _,p,rows,c,d=contradictory_cue_fixture(frame,mixed=frame==2)
        rows=tuple(replace(r,household=HouseholdDescriptor('visitor','visitor-'+str(i),1000+i,1)) for i,r in enumerate(rows))
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        owner.join_estimated_image_regions(p,at_interval=c,delivery_context=d)
    dto=next(x for x in owner.spatial_memberships(at_interval=c) if len(x.members)==2)
    assert dto.household_conflict is not None and dto.household_claim is None
    assert {m.household.stable_id for m in dto.members}=={1000,1001}
    assert owner.prepare_spatial_member(dto,spatial_reference(rows[0],c),at_interval=c).disposition=='denied'
    assert owner.prepare_spatial_physical_member(dto,spatial_reference(rows[0],c),at_interval=c).reason=='image_region_contradictory_cue_member'
    invalid=replace(spatial_reference(rows[0],c),origin='predicted')
    assert owner.prepare_spatial_physical_member(dto,invalid,at_interval=c).reason=='spatial_original_measurement_binding'


def test_image_regions_real_service_actual_reserve_discard_commit_and_expiry():
    selected,candidate,rows,context,delivery=image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected)
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    assert owner.join_estimated_image_regions(candidate,at_interval=context,delivery_context=delivery).disposition=='accepted'
    dto,=owner.spatial_memberships(at_interval=context)
    assert dto.association_basis=='estimated_current_image_regions'
    assert all(e.eligible for e in dto.eligibility)
    ref=spatial_reference(rows[0],context)
    reserved=owner.prepare_spatial_member(dto,ref,at_interval=context).reservation
    assert reserved is not None
    assert owner.discard_spatial_member(reserved).disposition=='discarded'
    reserved=owner.prepare_spatial_member(dto,ref,at_interval=context).reservation
    assert owner.commit_spatial_member(reserved,at_interval=context,consumer=lambda *_:'real-consumer').disposition=='committed'
    expired=replace(context,intervals=tuple(replace(i,evaluation_min_s=i.evaluation_min_s+.36,evaluation_max_s=i.evaluation_max_s+.36) for i in context.intervals))
    assert not any(e.eligible for d in owner.spatial_memberships(at_interval=expired) for e in d.eligibility)


@pytest.mark.parametrize('mutation',['update','reset','removal','inactive'])
def test_image_regions_service_current_revision_and_lifecycle_barriers(mutation):
    selected,candidate,rows,context,delivery=image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected)
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(candidate,at_interval=context,delivery_context=delivery)
    dto,=owner.spatial_memberships(at_interval=context)
    reserved=owner.prepare_spatial_member(dto,spatial_reference(rows[0],context),at_interval=context).reservation
    if mutation in {'update','inactive'}:
        _,_,fresh,_,_=image_region_fixture(frame=2)
        owner.observe_spatial(fresh[0],SpatialLifecycleReceipt('new',fresh[0].stamp,'INACTIVE' if mutation=='inactive' else 'ACTIVE'))
    elif mutation=='reset': owner.reset_source(replace(SOURCES[1],source_epoch=1))
    else: owner.withdraw(removal(rows[0]))
    calls=[]
    assert owner.commit_spatial_member(reserved,at_interval=context,consumer=lambda *_:calls.append(1)).disposition=='denied'
    assert not calls


def test_image_regions_legacy_opt_in_and_household_quarantine_are_not_bypassed():
    selected,candidate,rows,context,delivery=image_region_fixture(competitor='incompatible')
    # Two physically distinct settled owners in one camera keep their original
    # contradictory claim; geometry may not merge them to make identity unique.
    house=HouseholdDescriptor('resident','duplicate-resident',9)
    rows=tuple(replace(r,household=house) for r in rows)
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected)
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    assert owner.join_current_visual(visual_proof(rows),at_interval=context,delivery_context=delivery).reason=='estimated_image_regions_route_required'
    assert owner.join_estimated_image_regions(candidate,at_interval=context,delivery_context=delivery).disposition=='accepted'
    dtos=owner.spatial_memberships(at_interval=context)
    assert len(dtos)==2 and all(d.household_conflict is not None for d in dtos)
    dto=next(d for d in dtos if any(m.observation==rows[0].stamp for m in d.members))
    ref=spatial_reference(rows[0],context)
    assert owner.prepare_spatial_member(dto,ref,at_interval=context).disposition=='denied'
    intent=owner.prepare_spatial_physical_member(dto,ref,at_interval=context).reservation
    assert intent is not None and intent.purpose=='physical_only_quarantine'
    assert owner.discard_spatial_member(intent).disposition=='discarded'
    intent=owner.prepare_spatial_physical_member(dto,ref,at_interval=context).reservation
    assert owner.commit_spatial_physical_member(intent,at_interval=context,consumer=lambda *_:'quarantined').disposition=='committed'


def test_image_regions_two_heads_same_cohort_and_single_person_update_survive():
    from reid.mv3dt_person_association import CurrentVisualDeliveryContext
    selected,candidate,rows,context,delivery=image_region_fixture(competitor='incompatible')
    extra=observation('kitchen',10)
    rows=rows+(extra,)
    c=next(c for c in selected.cameras if c.source.camera_id=='kitchen')
    original=replace(candidate.originals[2],stamp=extra.stamp,body_evidence_id='body:'+extra.stamp.observation_id,
        raster_id=c.raster_id,prior_id=c.prior_id,image_size=c.image_size)
    candidate=replace(candidate,cohort=tuple(r.stamp for r in rows),originals=candidate.originals+(original,))
    delivery=visual_delivery(rows)
    context=spatial_context(rows)
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected)
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    assert owner.join_estimated_image_regions(candidate,at_interval=context,delivery_context=delivery).disposition=='accepted'
    dtos=owner.spatial_memberships(at_interval=context)
    first_keys={d.physical_key for d in dtos}
    assert len(first_keys)==2 and all(len(d.members)==2 for d in dtos)
    fresh=observation('kitchen',7,frame=2,time=10.05,media=5.05)
    owner.observe_spatial(fresh,SpatialLifecycleReceipt('fresh-person-one',fresh.stamp,'ACTIVE'))
    rows=(fresh,)+rows[1:]
    first=replace(candidate.originals[0],stamp=fresh.stamp,ingress_monotonic_s=100.05,body_evidence_id='body:'+fresh.stamp.observation_id)
    candidate=replace(candidate,proof_id='one-person-update',cohort=tuple(r.stamp for r in rows),originals=(first,)+candidate.originals[1:])
    delivery=CurrentVisualDeliveryContext(delivery.actor_id,2,100.05,tuple(CurrentVisualIngress(r.stamp,100.05 if i==0 else 100.) for i,r in enumerate(rows)))
    context=spatial_context(rows)
    assert owner.join_estimated_image_regions(candidate,at_interval=context,delivery_context=delivery).disposition=='accepted'
    dtos=owner.spatial_memberships(at_interval=context)
    assert {d.physical_key for d in dtos}==first_keys
    assert all(any(e.eligible for e in d.eligibility) for d in dtos)


def test_image_regions_continuation_dto_keeps_old_support_but_grants_only_real_current_measurement():
    selected,candidate,rows,context,delivery=image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected)
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(candidate,at_interval=context,delivery_context=delivery)
    old_dto,=owner.spatial_memberships(at_interval=context)
    old_ref=spatial_reference(rows[0],context)
    stale_intent=owner.prepare_spatial_member(old_dto,old_ref,at_interval=context).reservation
    for frame in range(2,6):
        _,candidate,rows,context,delivery=image_region_fixture(frame=frame)
        for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
        result=owner.join_estimated_image_regions(replace(candidate,originals=()),at_interval=context,delivery_context=delivery)
        assert result.association_basis=='estimated_image_region_continuation'
    dto,=owner.spatial_memberships(at_interval=context)
    assert dto.physical_key==old_dto.physical_key and dto.association_revision>old_dto.association_revision
    assert dto.association_basis=='estimated_image_region_continuation'
    assert len(dto.image_region_support)==2
    assert all(v.current_stamp.frame_id==5 and v.original_stamp.frame_id==1 and v.original_ingress_monotonic_s==100. for v in dto.image_region_support)
    with pytest.raises(FrozenInstanceError): dto.image_region_support[0].original_ingress_monotonic_s=100.2
    assert owner.prepare_spatial_member(dto,old_ref,at_interval=context).disposition=='denied'
    assert owner.commit_spatial_member(stale_intent,at_interval=context,consumer=lambda *_:None).disposition=='denied'
    reference=spatial_reference(rows[0],context)
    prepared=owner.prepare_spatial_member(dto,reference,at_interval=context).reservation
    assert prepared is not None
    seen=[]
    def actual(dto,measurement):
        seen.append((dto.image_region_support[0].original_stamp.frame_id,measurement.stamp.frame_id))
    assert owner.commit_spatial_member(prepared,at_interval=context,consumer=actual).disposition=='committed'
    assert seen==[(1,5)]


def test_image_regions_continuation_expiry_uses_existing_ordinary_denial_and_reset_cannot_reseed():
    selected,candidate,rows,context,delivery=image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected)
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(candidate,at_interval=context,delivery_context=delivery)
    old=owner.spatial_memberships(at_interval=context)[0]
    for frame in range(2,4):
        _,candidate,rows,context,delivery=image_region_fixture(frame=frame)
        for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
        owner.join_estimated_image_regions(replace(candidate,originals=()),at_interval=context,delivery_context=delivery)
    dto,=owner.spatial_memberships(at_interval=context)
    advanced=replace(context,intervals=tuple(replace(i,evaluation_min_s=i.evaluation_min_s+.26,evaluation_max_s=i.evaluation_max_s+.26) for i in context.intervals))
    # Current rows are .26 old (<.35), but ORIGINAL confirmation is .36 old.
    result=owner.prepare_spatial_member(dto,spatial_reference(rows[0],context),at_interval=advanced)
    assert result.reason=='image_region_current_support_required'
    owner.reset_source(replace(SOURCES[1],source_epoch=1))
    assert owner.prepare_spatial_member(old,spatial_reference(rows[0],context),at_interval=context).disposition=='denied'


def test_image_regions_unconfirmed_or_unknown_pose_never_initially_joins():
    selected,candidate,rows,context,delivery=image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected)
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    result=owner.join_estimated_image_regions(replace(candidate,originals=()),at_interval=context,delivery_context=delivery)
    assert result.disposition=='observed' and len(owner.spatial_memberships(at_interval=context))==2


def test_image_regions_divergent_visitors_one_physical_person_requires_original_claim_quarantine():
    selected,candidate,rows,context,delivery=image_region_fixture()
    houses=(HouseholdDescriptor('visitor','visitor-kitchen',1004,1),HouseholdDescriptor('visitor','visitor-family',1000,1))
    rows=tuple(replace(r,household=h) for r,h in zip(rows,houses))
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected)
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    result=owner.join_estimated_image_regions(candidate,at_interval=context,delivery_context=delivery)
    assert result.disposition=='accepted',result
    dto,=owner.spatial_memberships(at_interval=context)
    conflict=dto.household_conflict
    assert type(conflict) is HouseholdClaimConflict and dto.household_claim is None
    assert conflict.reason=='settled_household_claim_conflict'
    assert conflict.stable_ids==(1000,1004)
    assert len({p for p,_ in conflict.owners})==1 and {o.stamp.member:o.household for _,o in conflict.owners}=={r.stamp.member:r.household for r in rows}
    ref=spatial_reference(rows[0],context)
    assert owner.prepare_spatial_member(dto,ref,at_interval=context).reason=='settled_household_claim_conflict'
    intent=owner.prepare_spatial_physical_member(dto,ref,at_interval=context).reservation
    assert intent is not None and intent.household_conflict is conflict
    assert owner.commit_spatial_physical_member(intent,at_interval=context,consumer=lambda d,r:(d.physical_key,r.stamp)).disposition=='committed'


@pytest.mark.parametrize('variant',['resident','resident_sid','visitor_generation'])
def test_image_regions_intra_claim_signature_variants_preserve_all_values(variant):
    selected,candidate,rows,context,delivery=image_region_fixture()
    if variant=='resident': houses=(HouseholdDescriptor('resident','A',9),HouseholdDescriptor('resident','B',9))
    elif variant=='resident_sid': houses=(HouseholdDescriptor('resident','A',9),HouseholdDescriptor('resident','A',10))
    else: houses=(HouseholdDescriptor('visitor','V',9,1),HouseholdDescriptor('visitor','V',9,2))
    rows=tuple(replace(r,household=h) for r,h in zip(rows,houses))
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected)
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    assert owner.join_estimated_image_regions(candidate,at_interval=context,delivery_context=delivery).disposition=='accepted'
    dto,=owner.spatial_memberships(at_interval=context)
    assert dto.household_claim is None and len(dto.household_conflict.owners)==2
    assert dto.household_conflict.stable_ids==tuple(sorted({h.stable_id for h in houses}))
    assert len(dto.household_conflict.subject_ids)==(1 if variant=='resident_sid' else 2)


def test_image_regions_intra_and_cross_duplicates_have_complete_owner_union_not_fake_people():
    selected,candidate,rows,context,delivery=image_region_fixture(competitor='incompatible')
    a,b=HouseholdDescriptor('visitor','A',1004,1),HouseholdDescriptor('visitor','B',1000,1)
    rows=tuple(replace(r,household=h) for r,h in zip(rows,(a,b,a)))
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected)
    for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    assert owner.join_estimated_image_regions(candidate,at_interval=context,delivery_context=delivery).disposition=='accepted'
    dtos=owner.spatial_memberships(at_interval=context)
    assert len(dtos)==2 and dtos[0].household_conflict==dtos[1].household_conflict
    conflict=dtos[0].household_conflict
    assert len(conflict.owners)==3 and len({key for key,_ in conflict.owners})==2
    assert conflict.stable_ids==(1000,1004) and len(conflict.subject_ids)==2
    assert {original for _,original in conflict.owners}==set(rows)


@pytest.mark.parametrize('constraint',['unresolved','rejected'])
def test_image_regions_intra_quarantine_never_bypasses_noneligible_or_legacy_gate(constraint):
    selected,candidate,rows,context,delivery=image_region_fixture()
    houses=(HouseholdDescriptor('visitor','A',1004,1),HouseholdDescriptor('visitor','B',1000,1))
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected)
    for i,row in enumerate(rows):
        row=replace(row,household=replace(houses[i],constraint=constraint if i==0 else 'eligible'))
        owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    assert owner.join_estimated_image_regions(candidate,at_interval=context,delivery_context=delivery).reason=='current_visual_household_conflict'
    assert len(owner.ledger._persons)==2
    legacy=IdentityAssociationService.for_current_visual('run-a',spatial_policy())
    for row,h in zip(rows,houses):
        row=replace(row,household=h)
        legacy.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
    assert legacy.join_current_visual(visual_proof(rows),at_interval=context,delivery_context=delivery).reason=='current_visual_household_conflict'


def test_image_regions_intra_quarantine_continues_then_recovers_only_from_real_owner_updates():
    selected,candidate,rows,context,delivery=image_region_fixture()
    houses=(HouseholdDescriptor('visitor','A',1004,1),HouseholdDescriptor('visitor','B',1000,1))
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected)
    old_intent=None
    for frame in (1,2,3):
        _,candidate,rows,context,delivery=image_region_fixture(frame=frame)
        rows=tuple(replace(r,household=houses[0] if frame==3 else houses[i]) for i,r in enumerate(rows))
        for row in rows: owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'))
        assert owner.join_estimated_image_regions(replace(candidate,originals=() if frame==2 else candidate.originals),at_interval=context,delivery_context=delivery).disposition=='accepted'
        dto,=owner.spatial_memberships(at_interval=context)
        reference=spatial_reference(rows[0],context)
        if frame<3:
            assert dto.household_conflict is not None and dto.household_claim is None
            assert owner.prepare_spatial_member(dto,reference,at_interval=context).reason=='settled_household_claim_conflict'
            if frame==2: assert dto.association_basis=='estimated_image_region_continuation'
            if frame==1: old_intent=owner.prepare_spatial_physical_member(dto,reference,at_interval=context).reservation
            else:
                assert owner.commit_spatial_physical_member(old_intent,at_interval=context,consumer=lambda *_:None).disposition=='denied'
                copied=replace(dto,household_conflict=replace(dto.household_conflict))
                assert owner.prepare_spatial_physical_member(copied,reference,at_interval=context).disposition=='denied'
        else:
            assert dto.household_conflict is None and dto.household_claim==houses[0]
            assert owner.prepare_spatial_member(dto,reference,at_interval=context).reservation is not None


def test_image_regions_one_member_historical_labels_cannot_manufacture_intra_conflict():
    selected,_,rows,_,_=image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected)
    current=replace(rows[0],stamp=replace(rows[0].stamp,frame_id=2),household=HouseholdDescriptor('resident','latest',9))
    owner.observe_spatial(current,SpatialLifecycleReceipt('native',current.stamp,'ACTIVE'))
    key=next(iter(owner.ledger._persons))
    historical=replace(rows[0],household=HouseholdDescriptor('resident','old',10))
    owner.ledger.household_claim_owners=lambda:((key,historical),(key,current))
    assert owner._household_check(key)==(None,None)


def test_image_regions_diagnostic_service_accessor_is_detached_and_never_mutates_authority():
    import json
    from tests.test_mv3dt_person_association import image_diagnostic_observe
    selected,_,_,_,_=image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected)
    image_diagnostic_observe(owner.ledger,1)
    image_diagnostic_observe(owner.ledger,2,mode='contradiction')
    revision=owner.ledger.authority_revision
    view=owner.image_region_decision_diagnostics()
    assert view['contract']=='noesis.identity.image_region_decisions.v1' and len(view['transitions'])==2
    json.dumps(view,allow_nan=False)
    view['transitions'].clear()
    assert len(owner.image_region_decision_diagnostics()['transitions'])==2
    assert owner.ledger.authority_revision==revision
    strict=IdentityAssociationService('empty',ProducerAdmission('unavailable','synthetic','unavailable'))
    assert strict.image_region_decision_diagnostics()['transitions']==[]


def admitted():
    selected = policy()
    return ProducerAdmission("admitted", "synthetic-source-receipt", "synthetic-test-only",
                             "run-a", selected, policy_identity(selected), "synthetic-metric",
                             "synthetic-acquisition", "synthetic-independent-source", "synthetic")


def service(**kwargs):
    return IdentityAssociationService("run-a", admitted(), **kwargs)


def accepted(owner=None, *, frame=1, household=HouseholdDescriptor()):
    owner = owner or service()
    time = 10 + frame / 30
    left = observation(frame=frame, time=time, media=frame / 30, household=household)
    right = observation("family-room", 8, frame=frame, time=time, media=frame / 30, household=household)
    owner.observe(left)
    owner.observe(right)
    pair = proof(left, right, f"service-pair-{frame}")
    result = owner.link(pair, now(time))
    assert result.disposition == "accepted", result
    return owner, left, right, pair, result.dto, now(time)


def cohort(dto):
    return MeasurementCohort(dto.run_id, dto.association_revision, dto.policy_identity,
                             dto.policy.target_frame, dto.policy.world_revision,
                             dto.proof.measurements)


class SyntheticExactConsumer:
    def __init__(self):
        self.calls = []

    def __call__(self, dto, exact):
        assert isinstance(dto, AssociationDTO)
        assert tuple(m.stamp for m in exact.measurements) == tuple(m.observation for m in dto.members)
        self.calls.append((dto.physical_key, dto.association_revision, exact))
        return dto.physical_key


def test_real_service_real_kernel_two_rows_one_key_exact_once_and_immutable() -> None:
    owner, left, right, pair, dto, at = accepted()
    assert len(dto.members) == 2 and dto.state == "accepted"
    assert dto.producer_admission.scope == "synthetic"
    assert dto.policy.sources == SOURCES
    assert not hasattr(dto, "position") and not hasattr(dto, "embedding")
    with pytest.raises(FrozenInstanceError):
        dto.association_revision = 0
    reservation = owner.prepare(dto, cohort(dto), at).reservation
    assert reservation is not None
    consumer = SyntheticExactConsumer()
    result = owner.commit(reservation, at, consumer)
    assert result.disposition == "committed" and result.consumer_value == dto.physical_key
    assert len(consumer.calls) == 1
    assert owner.prepare(dto, cohort(dto), at).reason == "cohort_already_committed"
    assert owner.commit(reservation, at, consumer).disposition == "denied"
    assert len(consumer.calls) == 1
    duplicate = owner.link(pair, at)
    assert duplicate.dto is dto
    assert duplicate.dto.proof.receipt.admitted_at == at


def test_honest_household_promotion_keeps_key_and_passes_descriptors_unchanged() -> None:
    owner, _, _, _, original, _ = accepted()
    resident = HouseholdDescriptor("resident", "enrolled-uuid", 5)
    _, _, _, _, promoted, _ = accepted(owner, frame=2, household=resident)
    assert promoted.physical_key == original.physical_key
    assert promoted.association_revision > original.association_revision
    assert all(m.household is resident for m in promoted.members)
    assert owner.prepare(original, cohort(original), now(10 + 2 / 30)).disposition == "denied"


@pytest.mark.parametrize("change", ["observation", "household", "proof", "epoch", "termination", "correction", "empty", "run"])
def test_any_authority_change_between_prepare_commit_denies_without_callback(change) -> None:
    owner, left, right, pair, dto, at = accepted()
    reserved = owner.prepare(dto, cohort(dto), at).reservation
    assert reserved is not None
    if change in {"observation", "household", "proof"}:
        changed_house = HouseholdDescriptor("provisional") if change == "household" else left.household
        fresh_left = observation(frame=2, time=10.1, media=.1, household=changed_house)
        owner.observe(fresh_left)
        if change == "proof":
            owner.link(proof(fresh_left, right, "different-current-proof"), now(10.1))
    elif change == "epoch":
        owner.reset_source(replace(SOURCES[1], source_epoch=1))
    elif change == "termination":
        owner.withdraw(removal(left, kind="termination", media=.1))
    elif change == "correction":
        owner.transition(event(left, left, right, kind="correction", time=at.at_s), left, at)
    elif change == "empty":
        owner.clear_source(SourceAbsence("empty", "kitchen", 1, 0, 2, 10.1, "capture-clock"))
    else:
        owner.reset_run("run-b")
    consumer = SyntheticExactConsumer()
    assert owner.commit(reserved, now(10.1), consumer).disposition == "denied"
    assert consumer.calls == []


def test_history_only_compaction_preserves_valid_reservation_and_authority() -> None:
    owner, _, _, _, dto, at = accepted()
    reservation = owner.prepare(dto, cohort(dto), at).reservation
    revision = owner.stats().association_revision
    authority = owner.ledger.authority_revision
    owner.compact_history()
    assert owner.stats().association_revision == revision
    assert owner.ledger.authority_revision == authority
    assert owner.commit(reservation, at, SyntheticExactConsumer()).disposition == "committed"


@pytest.mark.parametrize("foreign", ["dto-copy", "foreign-dto", "reservation-copy", "foreign-reservation", "discarded", "expired"])
def test_forged_foreign_discarded_and_expired_reservations_deny(foreign) -> None:
    owner, _, _, _, dto, at = accepted()
    reserved = owner.prepare(dto, cohort(dto), at).reservation
    other, _, _, _, other_dto, other_at = accepted()
    other_reserved = other.prepare(other_dto, cohort(other_dto), other_at).reservation
    consumer = SyntheticExactConsumer()
    if foreign == "dto-copy":
        assert owner.prepare(replace(dto), cohort(dto), at).disposition == "denied"
    elif foreign == "foreign-dto":
        assert owner.prepare(other_dto, cohort(other_dto), at).disposition == "denied"
    elif foreign == "reservation-copy":
        assert owner.commit(replace(reserved), at, consumer).disposition == "denied"
    elif foreign == "foreign-reservation":
        assert owner.commit(other_reserved, at, consumer).disposition == "denied"
    elif foreign == "discarded":
        assert owner.discard(reserved).disposition == "discarded"
        assert owner.commit(reserved, at, consumer).disposition == "denied"
    else:
        assert owner.commit(reserved, now(at.at_s + .36), consumer).disposition == "denied"
        assert owner.commit(reserved, at, consumer).disposition == "denied"
        assert owner.prepare(dto, cohort(dto), at).reason == "reordered_clock"
    assert consumer.calls == []


@pytest.mark.parametrize("field,value", [("run_id", "wrong-run"), ("association_revision", 999),
                                         ("policy_identity", "wrong-policy"), ("target_frame", "wrong-frame"),
                                         ("world_revision", "wrong-revision")])
def test_exact_measurement_cohort_binding_is_required(field, value) -> None:
    owner, _, _, _, dto, at = accepted()
    assert owner.prepare(dto, replace(cohort(dto), **{field: value}), at).disposition == "denied"


@pytest.mark.parametrize("field,value", [("transform_id", "wrong"), ("calibration_revision", "wrong"),
                                         ("origin", "predicted"), ("units", "cm"), ("evidence_id", "wrong")])
def test_exact_measurement_proof_original_provenance_required(field, value) -> None:
    owner, _, _, _, dto, at = accepted()
    exact = cohort(dto)
    altered = replace(exact.measurements[0], **{field: value})
    assert owner.prepare(dto, replace(exact, measurements=(altered, exact.measurements[1])), at).disposition == "denied"


def test_world_r2_denied_admission_creates_no_ledger_no_dto_no_callback() -> None:
    # Exact World R2 source receipt is DENIED; these are not admitted matrices.
    denied = ProducerAdmission("unavailable", "XC3D-WORLD-R2-007e8096", "common-metric-and-acquisition-denied")
    owner = IdentityAssociationService("run-a", denied)
    assert owner.ledger is None
    assert owner.observe(observation()).disposition == "unavailable"
    assert owner.link(None, now()).disposition == "unavailable"
    positive_owner, _, _, _, foreign_dto, at = accepted()
    assert owner.prepare(foreign_dto, cohort(foreign_dto), at).disposition == "denied"
    reservation = positive_owner.prepare(foreign_dto, cohort(foreign_dto), at).reservation
    consumer = SyntheticExactConsumer()
    assert owner.commit(reservation, at, consumer).disposition == "denied"
    assert consumer.calls == [] and owner.stats().current_dtos == 0


def test_missing_proof_distinct_people_household_conflict_and_missing_descriptors() -> None:
    owner = service()
    left = observation(household=HouseholdDescriptor("provisional"))
    right = observation("family-room", 8)
    owner.observe(left)
    owner.observe(right)
    assert owner.link(None, now()).disposition == "rejected"
    assert owner.link(proof(left, right, distance=2), now()).reason == "metric_separation"
    assert owner.stats().current_proofs == 0
    denied = observation(frame=2, time=10.1, media=5.1,
                         household=HouseholdDescriptor("unknown", constraint="rejected"))
    owner.observe(denied)
    assert owner.link(proof(denied, right, "reject"), now(10.1)).reason == "household_constraint"
    resident = HouseholdDescriptor("resident", "resident-a", 1)
    visitor = HouseholdDescriptor("visitor", "visitor-b", 11, 2)
    owner = service()
    left, right = observation(household=resident), observation("family-room", 8, household=visitor)
    owner.observe(left)
    owner.observe(right)
    assert owner.link(proof(left, right), now()).reason == "identity_conflict"


def test_correction_withdraws_only_offender_and_old_dto_cannot_commit() -> None:
    owner, left, right, _, dto, at = accepted()
    corrected = owner.transition(event(left, left, right, kind="correction", time=at.at_s), left, at)
    assert corrected.dto.physical_key != dto.physical_key
    assert corrected.dto.members[0].observation.member == left.stamp.member
    assert corrected.withdrawn_members == (left.stamp.member,)
    views = owner.ledger.snapshot(at)
    assert views[0].person_key == dto.physical_key
    assert views[0].members[0].observation == right.stamp
    assert owner.prepare(dto, cohort(dto), at).disposition == "denied"


def test_reservation_capacity_and_callback_failure_consume_intent_once() -> None:
    owner, _, _, _, dto, at = accepted(service(max_reservations=1))
    reserved = owner.prepare(dto, cohort(dto), at).reservation
    assert owner.prepare(dto, cohort(dto), at).disposition == "capacity"
    calls = []
    def failure(dto, exact):
        calls.append(dto.physical_key)
        assert owner.observe(observation(frame=2, time=10.1, media=.1)).reason == "reentrant_transaction"
        raise RuntimeError("synthetic consumer failure")
    with pytest.raises(RuntimeError, match="synthetic consumer failure"):
        owner.commit(reserved, at, failure)
    assert owner.commit(reserved, at, failure).disposition == "denied"
    assert len(calls) == 1


def test_service_10000_30hz_real_kernel_reserve_commit_and_post_compaction_dto_replay() -> None:
    owner = service()
    consumer = SyntheticExactConsumer()
    first_key = None
    old_dto = None
    old_pair = None
    for frame in range(1, 10001):
        _, left, right, pair, dto, at = accepted(owner, frame=frame)
        first_key = first_key or dto.physical_key
        assert dto.physical_key == first_key
        if old_dto is not None:
            assert owner.prepare(old_dto, cohort(old_dto), at).disposition == "denied"
        reservation = owner.prepare(dto, cohort(dto), at).reservation
        assert reservation is not None
        owner.compact_history()
        assert owner.commit(reservation, at, consumer).disposition == "committed"
        assert owner.commit(reservation, at, consumer).disposition == "denied"
        stats = owner.stats()
        kernel = owner.ledger.stats()
        assert stats.pending_reservations == 0
        assert stats.current_dtos <= 64 and stats.current_proofs <= 64 and stats.clock_frontiers <= 3
        assert stats.committed_cohorts <= 64
        assert kernel.receipts + kernel.retired_keys <= 128 and kernel.closed_members <= 128
        assert kernel.active_members <= 128 and kernel.lifecycle_frontiers <= 3
        if frame > 2:
            # Still within 0.35s, but exact old cohorts cannot regain admission.
            assert owner.link(old_pair, at).disposition != "accepted"
            assert owner.link(replace(old_pair, proof_id=f"renamed-{frame}"), at).disposition != "accepted"
        old_dto, old_pair = dto, pair
    assert len(consumer.calls) == 10000
    assert owner.ledger.stats().compacted_receipts > 128
    print(f"service_endurance: ticks=10000 callbacks={len(consumer.calls)} key={first_key} service={owner.stats()} kernel={owner.ledger.stats()}")


def test_malformed_forged_dto_cohort_and_reservation_deny_with_no_callback() -> None:
    owner, _, _, _, dto, at = accepted()
    assert owner.prepare(replace(dto, physical_key=[]), cohort(dto), at).disposition == "denied"
    malformed = replace(cohort(dto), measurements=(replace(cohort(dto).measurements[0], stamp=[]),))
    assert owner.prepare(dto, malformed, at).disposition == "denied"
    reservation = owner.prepare(dto, cohort(dto), at).reservation
    consumer = SyntheticExactConsumer()
    assert owner.commit(replace(reservation, serial=[]), at, consumer).disposition == "denied"
    assert owner.discard(replace(reservation, serial=[])).disposition == "denied"
    assert consumer.calls == []


def test_two_reservations_cannot_commit_the_same_measurement_cohort_twice() -> None:
    owner, _, _, _, dto, at = accepted()
    first = owner.prepare(dto, cohort(dto), at).reservation
    second = owner.prepare(dto, cohort(dto), at).reservation
    consumer = SyntheticExactConsumer()
    assert owner.commit(first, at, consumer).disposition == "committed"
    assert owner.commit(second, at, consumer).reason == "cohort_already_committed"
    assert len(consumer.calls) == 1


def test_cached_original_expiry_is_rechecked_at_prepare_and_commit_despite_fresh_geometry() -> None:
    owner = service()
    originals = (observation(), observation("family-room", 8))
    left = observation(frame=11, time=10.333, media=5.333)
    right = observation("family-room", 8, frame=11, time=10.333, media=5.333)
    owner.observe(left)
    owner.observe(right)
    result = owner.link(cached_proof(left, right, *originals), now(10.333))
    assert result.disposition == "accepted"
    dto = result.dto
    assert dto.proof.fresh_extractions == (False, False)
    assert dto.proof.receipt.expires_at_s == 10.35
    reserved = owner.prepare(dto, cohort(dto), now(10.333)).reservation
    consumer = SyntheticExactConsumer()
    assert owner.commit(reserved, now(10.351), consumer).disposition == "denied"
    assert owner.prepare(dto, cohort(dto), now(10.352)).disposition == "denied"
    assert consumer.calls == []


def test_selected_reinfer12_at30hz_honest_cache_and_fixed_gap_through_real_service() -> None:
    owner = service()
    consumer = SyntheticExactConsumer()
    first_key = None
    originals = None
    accepted_ticks = gaps = extraction_ticks = 0
    for frame in range(1, 10001):
        time = 10 + frame / 30
        left = observation(frame=frame, time=time, media=frame / 30)
        right = observation("family-room", 8, frame=frame, time=time, media=frame / 30)
        owner.observe(left)
        owner.observe(right)
        extracted = (frame - 1) % 12 == 0
        if extracted:
            originals = (left, right)
            extraction_ticks += 1
        pair = cached_proof(left, right, *originals, token=f"cadence-{frame}")
        result = owner.link(pair, now(time))
        if result.disposition == "accepted":
            dto = result.dto
            first_key = first_key or dto.physical_key
            assert dto.physical_key == first_key
            assert dto.proof.fresh_extractions == (extracted, extracted)
            assert tuple(a.stamp for a in dto.proof.appearances) == tuple(
                sorted((originals[0].stamp, originals[1].stamp), key=lambda s: s.member))
            assert dto.proof.receipt.expires_at_s == originals[0].stamp.acquired_at_s + .35
            reserved = owner.prepare(dto, cohort(dto), now(time)).reservation
            assert owner.commit(reserved, now(time), consumer).disposition == "committed"
            accepted_ticks += 1
        else:
            assert result.reason == "continuity_only", (frame, result)
            dto = result.dto
            assert dto.state == "continuity-only" and dto.physical_key == first_key
            assert dto.proof is None and dto.continuity.deadline_s == time + .35
            exact = MeasurementCohort(dto.run_id, dto.association_revision, dto.policy_identity,
                                     dto.policy.target_frame, dto.policy.world_revision,
                                     (MeasurementReference.from_evidence(pair.left), MeasurementReference.from_evidence(pair.right)))
            assert owner.prepare(dto, exact, now(time)).disposition == "denied"
            gaps += 1
        stats = owner.ledger.stats()
        assert stats.appearance_origins <= 128 and stats.continuity_gaps <= 64 and stats.previous_shared <= 64
        assert stats.receipts + stats.retired_keys <= 128 and stats.closed_members <= 128
        assert owner.stats().pending_reservations == 0
    assert accepted_ticks >= 9000 and len(consumer.calls) == accepted_ticks
    assert gaps > 0 and extraction_ticks < accepted_ticks
    print(f"cadence12: ticks=10000 accepted={accepted_ticks} continuity_only={gaps} actual_extractions_per_source={extraction_ticks} key={first_key} final={owner.ledger.stats()}")


def test_repeated_missing_proofs_never_move_first_gap_deadline_and_expiry_has_no_shared_authority() -> None:
    owner, _, _, _, dto, at = accepted()
    anchor = deadline = None
    for frame in range(2, 16):
        time = 10 + frame / 30
        left = observation(frame=frame, time=time, media=frame / 30)
        right = observation("family-room", 8, frame=frame, time=time, media=frame / 30)
        owner.observe(left)
        owner.observe(right)
        pair = replace(proof(left, right, f"missing-{frame}"), left_appearance=None, right_appearance=None)
        result = owner.link(pair, now(time))
        if result.dto is not None and result.dto.state == "continuity-only":
            gap = result.dto.continuity
            anchor = anchor or gap.first_missing_cohort
            deadline = deadline or gap.deadline_s
            assert gap.first_missing_cohort == anchor and gap.deadline_s == deadline
            assert owner.prepare(result.dto, None, now(time)).disposition == "denied"
        else:
            assert time >= deadline
            assert owner.ledger.stats().continuity_gaps == 0
            assert all(v.link is None and v.state != "continuity-only" for v in owner.ledger.snapshot(now(time)))
    assert deadline == anchor.at_s + .35


@pytest.mark.parametrize("boundary", ["absence", "termination", "correction", "epoch"])
def test_gap_boundary_immediately_invalidates_continuity_dto_and_reservation(boundary) -> None:
    owner, left, right, _, dto, at = accepted()
    reservation = owner.prepare(dto, cohort(dto), at).reservation
    fresh_left = observation(frame=2, time=10.1, media=.1)
    fresh_right = observation("family-room", 8, frame=2, time=10.1, media=.1)
    owner.observe(fresh_left)
    owner.observe(fresh_right)
    missing = replace(proof(fresh_left, fresh_right, "missing"), left_appearance=None)
    gap = owner.link(missing, now(10.1)).dto
    assert gap.state == "continuity-only"
    if boundary in {"absence", "termination"}:
        owner.withdraw(removal(fresh_left, kind=boundary, frame=3, time=10.2, media=.2))
    elif boundary == "correction":
        owner.transition(event(fresh_left, fresh_left, fresh_right, kind="correction", time=10.1), fresh_left, now(10.1))
    else:
        owner.reset_source(replace(SOURCES[1], source_epoch=1))
    assert owner.ledger.stats().continuity_gaps == 0
    consumer = SyntheticExactConsumer()
    assert owner.commit(reservation, now(10.2), consumer).disposition == "denied"
    assert owner.prepare(gap, None, now(10.2)).disposition == "denied"
    assert consumer.calls == []


def household_pairs(owner, first, second):
    """Direct scalar version of the parent x0/x3 four-row counterexample.

    World normalization/its original opaque stamps and the actual Node fixture
    stay with World. These original scalar observations are never retagged.
    """
    rows = (observation(raw=1, frame=2, time=10.2, media=.2, household=first),
            observation("family-room", 101, frame=2, time=10.2, media=.2, household=first),
            observation(raw=22, frame=2, time=10.2, media=.2, household=second),
            observation("family-room", 102, frame=2, time=10.2, media=.2, household=second))
    pairs = []
    for index, x in ((0, 0.), (2, 3.)):
        original = proof(rows[index], rows[index + 1], f"household-pair-{index}", distance=.03)
        pairs.append(replace(original, left=replace(original.left, x_m=x, y_m=2.),
                             right=replace(original.right, x_m=x + .03, y_m=2.)))
    for row in rows:
        owner.observe(row)
    for pair in pairs:
        owner.link(pair, now(10.2))
    return rows, tuple(pairs)


@pytest.mark.parametrize("other,subjects,sids", [
    (HouseholdDescriptor("resident", "synthetic-duplicate-resident", 9),
     ("resident:synthetic-duplicate-resident",), (9,)),
    (HouseholdDescriptor("resident", "synthetic-duplicate-resident", 10),
     ("resident:synthetic-duplicate-resident",), ()),
    (HouseholdDescriptor("resident", "different-resident", 9), (), (9,)),
])
def test_household_duplicate_resident_final_issuance_refuses_without_physical_merge(other, subjects, sids):
    owner = service()
    resident = HouseholdDescriptor("resident", "synthetic-duplicate-resident", 9)
    originals, pairs = household_pairs(owner, resident, other)
    views = owner.ledger.snapshot(now(10.2))
    assert len(views) == 2 and all(v.state == "accepted" for v in views)
    keys = tuple(v.person_key for v in views)
    for pair in pairs:  # Exact original operation replay at the final revision.
        result = owner.link(pair, now(10.2))
        assert result.disposition == "conflict" and result.dto is None
        conflict = result.household_conflict
        assert conflict.reason == "settled_household_claim_conflict"
        assert conflict.run_id == "run-a"
        assert conflict.association_revision == owner.stats().association_revision
        assert conflict.subject_ids == subjects and conflict.stable_ids == sids
        assert {k for k, _ in conflict.owners} == set(keys)
        by_stamp = {row.stamp: row for row in originals}
        assert {row for _, row in conflict.owners} == set(originals)
        assert all(row is by_stamp[row.stamp] for _, row in conflict.owners)
        assert conflict.owners == tuple(sorted(conflict.owners, key=lambda row: (row[0], row[1].stamp.member)))
        with pytest.raises(FrozenInstanceError):
            conflict.reason = "accepted"
    assert tuple(v.person_key for v in owner.ledger.snapshot(now(10.2))) == keys
    assert owner.ledger.stats().active_members == 4
    assert owner.ledger.policy.max_entities == 64 and owner.ledger.policy.max_history == 128
    assert owner.stats().current_dtos == 0


@pytest.mark.parametrize("second", [HouseholdDescriptor("visitor", "visitor-original", 9, 4),
                                   HouseholdDescriptor("visitor", "visitor-new", 9, 5),
                                   HouseholdDescriptor("handoff", "owner-held-original", 9)])
def test_household_visitor_generation_and_handoff_do_not_release_stable_id(second):
    owner = service()
    first = HouseholdDescriptor("visitor", "visitor-original", 9, 4)
    _, pairs = household_pairs(owner, first, second)
    result = owner.link(pairs[0], now(10.2))
    assert result.disposition == "conflict"
    assert result.household_conflict.stable_ids == (9,)
    expected = ("visitor:run-a:9:g4",) if second == first else ()
    assert result.household_conflict.subject_ids == expected
    assert {row.household for _, row in result.household_conflict.owners} == {first, second}


def test_household_valid_distinct_claims_final_revision_and_single_person_link_resolution():
    owner = service()
    first, second = HouseholdDescriptor("resident", "resident-a", 9), HouseholdDescriptor("visitor", "visitor-b", 10, 4)
    originals, pairs = household_pairs(owner, first, second)
    dtos = tuple(owner.link(pair, now(10.2)).dto for pair in pairs)
    assert len({d.physical_key for d in dtos}) == 2
    assert len({d.association_revision for d in dtos}) == 1
    reservations = tuple(owner.prepare(dto, cohort(dto), now(10.2)).reservation for dto in dtos)
    consumer = SyntheticExactConsumer()
    assert all(owner.commit(r, now(10.2), consumer).disposition == "committed" for r in reservations)
    assert len(consumer.calls) == 2
    single = service()
    left, right = originals[:2]
    key = single.observe(left).dto.physical_key
    pending = single.observe(right)
    assert pending.disposition == "conflict" and pending.dto is None
    assert single.ledger.stats().active_persons == 2
    result = single.link(pairs[0], now(10.2))
    assert result.disposition == "accepted" and result.dto.physical_key == key
    assert tuple(m.household for m in result.dto.members) == (first, first)
    assert single.prepare(result.dto, cohort(result.dto), now(10.2)).disposition == "reserved"


@pytest.mark.parametrize("initial", [HouseholdDescriptor(), HouseholdDescriptor("provisional")])
def test_household_promotion_conflict_keeps_key_and_truthful_later_correction_recovers(initial):
    owner = service()
    resident = HouseholdDescriptor("resident", "resident-a", 9)
    originals, pairs = household_pairs(owner, resident, initial)
    old = owner.link(pairs[1], now(10.2)).dto
    key = old.physical_key
    promoted = tuple(observation(row.stamp.member.camera_id, row.stamp.member.raw_id,
                                frame=3, time=10.3, media=.3, household=resident)
                     for row in originals[2:])
    for row in promoted:
        assert owner.observe(row).household_conflict is not None
    fresh = proof(*promoted, "promoted-original")
    assert owner.link(fresh, now(10.3)).disposition == "conflict"
    assert next(v for v in owner.ledger.snapshot(now(10.3)) if v.person_key == key).members[0].household is resident
    assert owner.prepare(old, cohort(old), now(10.3)).household_conflict is not None
    corrected_house = HouseholdDescriptor("visitor", "truthful-owner-correction", 10, 4)
    corrected = tuple(observation(row.stamp.member.camera_id, row.stamp.member.raw_id,
                                 frame=4, time=10.4, media=.4, household=corrected_house)
                      for row in promoted)
    for row in corrected:
        owner.observe(row)
    corrected_pair = proof(*corrected, "corrected-original")
    dto = owner.link(corrected_pair, now(10.4)).dto
    assert dto.physical_key == key and len(owner.ledger.snapshot(now(10.4))) == 2
    assert all(m.household is corrected_house for m in dto.members)
    assert owner.prepare(old, cohort(old), now(10.4)).disposition == "denied"
    assert owner.link(fresh, now(10.4)).disposition != "accepted"
    assert owner.link(replace(fresh, proof_id="renamed-old-promoted"), now(10.4)).disposition != "accepted"
    assert owner.prepare(dto, cohort(dto), now(10.4)).disposition == "reserved"


def test_household_collision_after_reservation_has_typed_denial_and_no_callback():
    resident = HouseholdDescriptor("resident", "resident-a", 9)
    owner, _, _, _, dto, at = accepted(household=resident)
    reservation = owner.prepare(dto, cohort(dto), at).reservation
    assert reservation is not None
    row = observation("living-room", 42, frame=2, time=10.1, media=.1, household=resident)
    conflict = owner.observe(row).household_conflict
    consumer = SyntheticExactConsumer()
    result = owner.commit(reservation, now(10.1), consumer)
    assert result.disposition == "denied" and result.household_conflict == conflict
    assert result.reason == "settled_household_claim_conflict" and consumer.calls == []
    assert owner.stats().pending_reservations == 0
    assert owner.ledger.stats().active_persons == 2


@pytest.mark.parametrize("release", ["expiry", "termination", "source", "epoch", "run"])
def test_household_owned_grace_and_other_source_support_release_only_at_real_boundary(release):
    from reid.mv3dt_person_association import LifecycleRetention
    resident = HouseholdDescriptor("resident", "resident-a", 9)
    owner, left, right, _, dto, _ = accepted(household=resident)
    retained = LifecycleRetention("original-owned-retention", left.stamp, left.stamp.member, "inactive", .35)
    owner.withdraw(removal(left, retention=retained, media=.1))
    other = observation("living-room", 42, frame=2, time=10.1, media=.1, household=resident)
    collision = owner.observe(other)
    assert collision.disposition == "conflict"
    assert {o.stamp for _, o in collision.household_conflict.owners} == {left.stamp, right.stamp, other.stamp}
    owner.expire_retention(1, .39)  # Original expiry; remaining Family source STILL owns claim.
    assert owner.observe(other).disposition == "conflict"
    assert {o.stamp for _, o in owner.observe(other).household_conflict.owners} == {right.stamp, other.stamp}
    if release == "expiry":
        retained = LifecycleRetention("right-retention", right.stamp, right.stamp.member, "inactive", .35)
        owner.withdraw(removal(right, token="right-absence", retention=retained, media=.1))
        assert owner.observe(other).disposition == "conflict"
        owner.expire_retention(2, .39)
    elif release == "termination":
        owner.withdraw(removal(right, kind="termination", token="right-terminated", media=.1))
    elif release == "source":
        owner.clear_source(SourceAbsence("right-source-absent", "family-room", 2, 0, 2, 10.1, "capture-clock"))
    elif release == "epoch":
        owner.reset_source(replace(SOURCES[2], source_epoch=1))
        assert owner.observe(other).disposition == "unavailable"
        changed = replace(admitted().policy, sources=owner.ledger.source_bindings())
        assert owner.bind_admission(replace(admitted(), policy=changed,
                                            policy_identity=policy_identity(changed))).disposition == "bound"
    else:
        assert owner.reset_run("run-b").disposition == "reset"
        assert owner.ledger.household_claim_owners() == ()
        assert owner.observe(other).disposition == "unavailable"
        return
    current = owner.observe(other)
    assert current.disposition == "observed" and current.household_conflict is None
    assert current.dto.physical_key != dto.physical_key
    # Numeric reuse is a NEW owned lifecycle, never a release of Living support.
    reused = observation(raw=left.stamp.member.raw_id, frame=3, started=3, generation=2,
                         time=10.2, media=.2, household=resident)
    result = owner.observe(reused)
    assert result.disposition == "conflict"
    assert len({k for k, _ in result.household_conflict.owners}) == 2


def test_household_conflict_aggregate_excludes_unrelated_original_owners():
    owner = service()
    resident = HouseholdDescriptor("resident", "resident-a", 9)
    originals, pairs = household_pairs(owner, resident, resident)
    unrelated = observation("living-room", 42, frame=2, time=10.2, media=.2,
                            household=HouseholdDescriptor("visitor", "visitor-unrelated", 10, 4))
    result = owner.observe(unrelated)
    assert result.dto is not None and result.household_conflict is None
    denied = owner.link(pairs[0], now(10.2))
    assert {row for _, row in denied.household_conflict.owners} == set(originals)
    assert unrelated not in {row for _, row in denied.household_conflict.owners}


def test_household_adoption_reissues_only_original_operation_at_final_revision():
    resident = HouseholdDescriptor("resident", "resident-a", 9)
    owner, left, right, pair, original, at = accepted(household=resident)
    current = observation(raw=17, frame=2, time=10.1, media=.1, household=resident)
    fresh = proof(current, right, "original-adoption-proof")
    native = event(left, current, right, token="original-adoption", time=10.1)
    adopted = owner.transition(native, current, now(10.1), fresh)
    assert adopted.dto.physical_key == original.physical_key
    outsider = observation("living-room", 42, frame=2, time=10.1, media=.1,
                           household=HouseholdDescriptor("resident", "resident-b", 10))
    owner.observe(outsider)
    assert owner.link(fresh, now(10.1)).dto is None
    final = owner.transition(native, current, now(10.1), fresh)
    assert final.dto.physical_key == original.physical_key
    assert final.dto.native_transition is native
    assert final.dto.association_revision == owner.stats().association_revision
    assert owner.prepare(adopted.dto, cohort(adopted.dto), now(10.1)).disposition == "denied"
    assert owner.link(pair, now(10.1)).dto is None
    assert owner.prepare(final.dto, cohort(final.dto), now(10.1)).disposition == "reserved"


def test_household_default_capacity_conflicts_do_not_reset_or_create_registry():
    owner = service()
    resident = HouseholdDescriptor("resident", "resident-a", 9)
    names = set(vars(owner))
    originals = tuple(observation(raw=raw, frame=2, time=10.2, media=.2, household=resident)
                      for raw in range(1, 65))
    for row in originals:
        result = owner.observe(row)
    assert result.disposition == "conflict"
    conflict = result.household_conflict
    assert len(conflict.owners) == 64 and len({k for k, _ in conflict.owners}) == 64
    assert {row for _, row in conflict.owners} == set(originals)
    assert set(vars(owner)) == names  # Derived view, no permanent claim registry.
    assert owner.ledger.stats().active_persons == 64
    assert owner.ledger.stats().allocated_serial == 64
    assert owner.ledger.stats().retired_keys == 0 and owner.ledger.stats().blocked_sources == 0
    overflow = owner.observe(observation(raw=65, frame=2, time=10.2, media=.2, household=resident))
    assert overflow.disposition == "capacity" and overflow.reason == "entity_capacity"
    assert owner.ledger.stats().active_persons == 64
    assert len(owner.ledger.household_claim_owners()) == 64


def test_household_multiple_original_grace_claims_stay_bounded_without_truncation_or_release():
    from reid.mv3dt_person_association import LifecycleRetention
    owner = service()
    resident = HouseholdDescriptor("resident", "resident-a", 9)
    left, right = observation(household=resident), observation("family-room", 8, household=resident)
    owner.observe(left)
    owner.observe(right)
    initial = owner.link(proof(left, right, "grace-initial"), now(10.)).dto
    other = observation("living-room", 42, household=resident)
    owner.observe(other)
    # All genuine original receipts fit their unchanged .35s media grace.
    # Fresh scalar source cohorts are explicit; none is cached/retimestamped.
    for index in range(3):
        time, media, frame = 10.001 + index / 1000, 5.001 + index / 1000, 2 + index * 2
        retained = LifecycleRetention(f"owned-{index}", left.stamp, left.stamp.member, "inactive", .35)
        assert owner.withdraw(removal(left, token=f"absence-{index}", frame=frame,
                                      time=time, media=media, retention=retained)).disposition == "removed"
        left = observation(raw=1000 + index, frame=frame + 1, started=frame + 1,
                           time=time, media=media, household=resident)
        owner.observe(left)
        result = owner.link(proof(left, right, f"reproved-{index}"), now(time))
    rows = owner.ledger.household_claim_owners()
    assert len(rows) == 6 and owner.ledger.stats().retained_members == 3
    assert result.disposition == "conflict" and result.reason == "settled_household_claim_conflict"
    assert result.household_conflict.owners == rows and result.dto is None
    views = owner.ledger.snapshot(now(time))
    assert len(views) == 2 and any(v.person_key == initial.physical_key and v.state == "accepted" for v in views)
    assert owner.ledger.policy.max_entities == 64 and owner.ledger.policy.max_history == 128
    assert owner.ledger.stats().blocked_sources == 0
    assert len(owner.ledger.household_claim_owners()) == 6


def test_household_conflict_output_budget_guard_fault_injection_denies_all_grants(monkeypatch):
    """Defensive output-budget check, NOT a reachable kernel-cohort claim.

    Inject an oversized read-only owner view into the real service. No kernel
    state/claim is fabricated or admitted by this check; only refusal is tested.
    """
    resident = HouseholdDescriptor("resident", "resident-a", 9)
    owner, left, right, pair, dto, at = accepted(household=resident)
    reservation = owner.prepare(dto, cohort(dto), at).reservation
    other = observation("living-room", 42, frame=2, time=10.1, media=.1, household=resident)
    owner.observe(other)
    originals = owner.ledger.household_claim_owners()
    additional = tuple((dto.physical_key, observation(raw=1000 + i, household=resident))
                       for i in range(126))
    oversized_view = tuple(sorted(originals + additional, key=lambda row: (row[0], row[1].stamp.member)))
    assert len(oversized_view) == 129
    monkeypatch.setattr(owner.ledger, "household_claim_owners", lambda: oversized_view)
    before = owner.ledger.stats()
    issued = owner.observe(other)  # Exact duplicate: no kernel mutation.
    prepared = owner.prepare(dto, cohort(dto), now(10.1))
    consumer = SyntheticExactConsumer()
    committed = owner.commit(reservation, now(10.1), consumer)
    assert issued.disposition == "capacity" and issued.dto is None
    assert prepared.disposition == committed.disposition == "denied"
    assert all(result.reason == "household_claim_capacity" and result.household_conflict is None
               for result in (issued, prepared, committed))
    assert prepared.reservation is None and committed.consumer_value is None and consumer.calls == []
    assert owner.ledger.stats() == before
    assert owner.ledger.household_claim_owners() is oversized_view
    assert owner.ledger.policy.max_entities == 64 and owner.ledger.policy.max_history == 128


def spatial_owner(rows=None, *, join=True):
    owner = IdentityAssociationService.for_spatial("run-a", spatial_policy())
    rows = rows or (observation(), observation("family-room", 8))
    for row in rows:
        owner.observe_spatial(row, SpatialLifecycleReceipt("native:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
    context = spatial_context(rows, qualified=join)
    if join:
        result = owner.join_spatial(spatial_person_proof(rows), at_interval=context)
        assert result.disposition == "accepted", result
    return owner, rows, context


def spatial_reference(row, context):
    interval = next(i for i in context.intervals if i.binding.source_id == row.stamp.member.source_id)
    return SpatialMeasurementReference(row.stamp, row.stamp.member, "original-measurement:" + row.stamp.observation_id,
                                       "original-source-frame", "original-source-revision", interval.binding.calibration_revision,
                                       interval.binding.transform_id, "measured", "m", "original-person-ground-stage", interval)


def test_spatial_api_local_example_reserve_discard_commit_without_common_metric():
    row = observation("living-room", 42)
    owner, rows, context = spatial_owner((row,), join=False)
    dto, = owner.spatial_memberships(at_interval=context)
    assert dto.eligibility[0].eligible and dto.household_claim is None
    assert isinstance(dto, SpatialMembershipDTO)
    assert not any(hasattr(dto, field) for field in ("position", "covariance", "confidence", "policy", "proof"))
    reference = spatial_reference(row, context)
    intent = owner.prepare_spatial_member(dto, reference, at_interval=context).reservation
    assert owner.discard_spatial_member(intent).disposition == "discarded"
    assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda *_: 0).disposition == "denied"
    intent = owner.prepare_spatial_member(dto, reference, at_interval=context).reservation
    calls = []
    result = owner.commit_spatial_member(intent, at_interval=context,
                                         consumer=lambda member, original: calls.append((member.physical_key, original)) or member.physical_key)
    assert result.disposition == "committed" and calls == [(dto.physical_key, reference)]
    assert owner.prepare_spatial_member(dto, reference, at_interval=context).reason == "spatial_member_measurement_consumed"
    assert owner.bind_admission(admitted()).disposition == "unavailable"
    assert owner.observe(row).disposition == "unavailable"
    print("local example:", result.disposition, dto.physical_key, reference.evidence_id)


def test_spatial_api_two_member_example_any_source_quality_reversal_and_original_replay():
    owner, rows, context = spatial_owner()
    dto, = owner.spatial_memberships(at_interval=context)
    physical = dto.physical_key
    assert len(dto.members) == 2 and all(e.eligible for e in dto.eligibility)
    worse = spatial_reference(rows[0], context)
    initial = owner.prepare_spatial_member(dto, worse, at_interval=context).reservation
    assert owner.commit_spatial_member(initial, at_interval=context, consumer=lambda d, r: r).consumer_value is worse
    # Better Family arrives on its NEXT ORIGINAL cohort, not a retagged cache.
    family = observation("family-room", 8, frame=2, time=10.1, media=5.1)
    owner.observe_spatial(family, SpatialLifecycleReceipt("family-next", family.stamp, "ACTIVE"))
    current_rows = rows[0], family
    context = spatial_context(current_rows, evaluation=10.1)
    dto, = owner.spatial_memberships(at_interval=context)
    assert dto.physical_key == physical
    better = spatial_reference(family, context)
    intent = owner.prepare_spatial_member(dto, better, at_interval=context).reservation
    assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda d, r: r).consumer_value is better
    assert owner.prepare_spatial_member(dto, spatial_reference(rows[0], context), at_interval=context).reason == "spatial_member_measurement_consumed"
    kitchen = observation(frame=3, time=10.2, media=5.2)
    owner.observe_spatial(kitchen, SpatialLifecycleReceipt("kitchen-next", kitchen.stamp, "ACTIVE"))
    context = spatial_context((kitchen, family), evaluation=10.2)
    dto, = owner.spatial_memberships(at_interval=context)
    reverse = spatial_reference(kitchen, context)
    intent = owner.prepare_spatial_member(dto, reverse, at_interval=context).reservation
    assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda d, r: r).consumer_value is reverse
    assert owner.prepare_spatial_member(dto, spatial_reference(family, context), at_interval=context).reason == "spatial_member_measurement_consumed"
    assert dto.physical_key == physical and owner.stats().spatial_member_frontiers == 2
    print("two-member example:", physical, "Kitchen -> Family -> Kitchen; originals once each")


@pytest.mark.parametrize("state", ["INACTIVE", "TENTATIVE", "UNKNOWN", "TERMINATED"])
def test_spatial_native_nonactive_cannot_mint_measurement_grant(state):
    owner = IdentityAssociationService.for_spatial("run-a", spatial_policy())
    row = observation()
    owner.observe_spatial(row, SpatialLifecycleReceipt("native-state", row.stamp, state))
    context = spatial_context((row,))
    dtos = owner.spatial_memberships(at_interval=context)
    if state == "TERMINATED":
        assert dtos == () and owner.ledger.stats().active_members == 0
    else:
        assert not dtos[0].eligibility[0].eligible
        assert owner.prepare_spatial_member(dtos[0], spatial_reference(row, context), at_interval=context).reason == "native_active_required"


@pytest.mark.parametrize("boundary", ["absence", "empty", "termination", "correction", "epoch", "run"])
def test_spatial_barrier_invalidates_queued_intent_preserves_other_source(boundary):
    owner, rows, context = spatial_owner()
    dto, = owner.spatial_memberships(at_interval=context)
    intent = owner.prepare_spatial_member(dto, spatial_reference(rows[0], context), at_interval=context).reservation
    if boundary in {"absence", "termination"}:
        changed = owner.withdraw(removal(rows[0], kind=boundary))
    elif boundary == "empty":
        changed = owner.clear_source(SourceAbsence("empty-K", "kitchen", 1, 0, 2, 10.1, "capture-clock"))
    elif boundary == "correction":
        changed = owner.transition(event(rows[0], rows[0], rows[1], kind="correction", time=10.05), rows[0], now())
    elif boundary == "epoch":
        changed = owner.reset_source(replace(SOURCES[1], source_epoch=1))
    else:
        changed = owner.reset_run("run-b")
    assert changed.withdrawn_members == ((rows[0].stamp.member,) if boundary != "run" else
                                         tuple(sorted(r.stamp.member for r in rows)))
    calls = []
    assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda *_: calls.append(1)).disposition == "denied"
    assert calls == []
    if boundary != "run":
        assert rows[1].stamp.member in owner.ledger._members
        assert owner.ledger._members[rows[1].stamp.member].person_key == dto.physical_key
        family_context = spatial_context((rows[1],), evaluation=10.1)
        current = owner.spatial_memberships(at_interval=family_context)
        family_dto = next(d for d in current if d.physical_key == dto.physical_key)
        assert owner.prepare_spatial_member(family_dto, spatial_reference(rows[1], family_context), at_interval=family_context).disposition == "reserved"


def test_spatial_absent_grace_return_requires_fresh_real_rejoin_and_numeric_reuse_is_new():
    from reid.mv3dt_person_association import LifecycleRetention
    owner, rows, context = spatial_owner()
    original, = owner.spatial_memberships(at_interval=context)
    retained = LifecycleRetention("genuine-return", rows[0].stamp, rows[0].stamp.member, "inactive", .35)
    owner.withdraw(removal(rows[0], retention=retained))
    returned = observation(frame=3, time=10.2, media=5.2)
    owner.observe_spatial(returned, SpatialLifecycleReceipt("returned-native", returned.stamp, "ACTIVE"))
    context = spatial_context((returned, rows[1]), evaluation=10.2, qualified=True)
    dto, = owner.spatial_memberships(at_interval=context)
    assert dto.physical_key == original.physical_key
    assert owner.prepare_spatial_member(dto, spatial_reference(returned, context), at_interval=context).reason == "current_rejoin_proof_required"
    assert owner.join_spatial(spatial_person_proof((returned, rows[1]), token="real-rejoin"), at_interval=context).disposition == "accepted"
    dto, = owner.spatial_memberships(at_interval=context)
    assert owner.prepare_spatial_member(dto, spatial_reference(returned, context), at_interval=context).disposition == "reserved"
    owner.withdraw(removal(returned, kind="termination", frame=4, time=10.3, media=5.3, token="terminated-original"))
    reused = observation(frame=5, started=5, generation=2, time=10.4, media=5.4)
    owner.observe_spatial(reused, SpatialLifecycleReceipt("new-native-life", reused.stamp, "ACTIVE"))
    context = spatial_context((reused, rows[1]), evaluation=10.4)
    dtos = owner.spatial_memberships(at_interval=context)
    assert len(dtos) == 2
    assert owner.ledger._members[reused.stamp.member].person_key != original.physical_key
    assert owner.prepare_spatial_member(original, spatial_reference(rows[0], spatial_context(rows)), at_interval=context).disposition == "denied"


def test_spatial_owned_membership_survives_shared_appearance_age_but_never_refreshes_it():
    owner, rows, context = spatial_owner()
    original_appearances = dict(owner.ledger._appearance_origins)
    physical = owner.spatial_memberships(at_interval=context)[0].physical_key
    next_row = observation(frame=20, time=10.7, media=5.7)
    owner.observe_spatial(next_row, SpatialLifecycleReceipt("ongoing-local-native", next_row.stamp, "ACTIVE"))
    context = spatial_context((next_row, rows[1]), evaluation=10.7)
    dto, = owner.spatial_memberships(at_interval=context)
    assert dto.physical_key == physical
    assert dict(owner.ledger._appearance_origins) == original_appearances
    assert owner.prepare_spatial_member(dto, spatial_reference(next_row, context), at_interval=context).disposition == "reserved"
    assert owner.prepare_spatial_member(dto, spatial_reference(rows[1], context), at_interval=context).reason == "stale_spatial_member"


@pytest.mark.parametrize("defect", ["copy-dto", "malformed-key", "foreign", "copy-intent", "discard", "stale", "epoch", "stage", "units", "birth", "retimestamp"])
def test_spatial_forged_stale_or_rebound_reference_denies_without_callback(defect):
    owner, rows, context = spatial_owner()
    dto, = owner.spatial_memberships(at_interval=context)
    reference = spatial_reference(rows[0], context)
    calls = []
    if defect == "copy-dto":
        assert owner.prepare_spatial_member(replace(dto), reference, at_interval=context).disposition == "denied"
    elif defect == "malformed-key":
        assert owner.prepare_spatial_member(replace(dto, physical_key=replace(dto.physical_key, serial=[])),
                                            reference, at_interval=context).disposition == "denied"
    elif defect == "foreign":
        other, _, _ = spatial_owner()
        assert other.prepare_spatial_member(dto, reference, at_interval=context).disposition == "denied"
    elif defect in {"copy-intent", "discard", "stale"}:
        intent = owner.prepare_spatial_member(dto, reference, at_interval=context).reservation
        if defect == "copy-intent":
            intent = replace(intent)
        elif defect == "discard":
            owner.discard_spatial_member(intent)
        else:
            context = spatial_context(rows, evaluation=10.36)
        assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda *_: calls.append(1)).disposition == "denied"
    else:
        if defect == "epoch":
            reference = replace(reference, stamp=replace(reference.stamp, member=replace(reference.owner, source_epoch=1)))
        elif defect == "stage":
            reference = replace(reference, origin="held")
        elif defect == "units":
            reference = replace(reference, units="cm")
        elif defect == "birth":
            reference = replace(reference, stamp=replace(reference.stamp, lifecycle_started_frame=0))
        else:
            reference = replace(reference, stamp=replace(reference.stamp, acquired_at_s=10.1))
        assert owner.prepare_spatial_member(dto, reference, at_interval=context).disposition == "denied"
    assert calls == []


def test_spatial_callback_failure_consumes_intent_and_member_frontier_without_reentrancy():
    owner, rows, context = spatial_owner()
    dto, = owner.spatial_memberships(at_interval=context)
    reference = spatial_reference(rows[0], context)
    intent = owner.prepare_spatial_member(dto, reference, at_interval=context).reservation
    def failure(dto, reference):
        assert owner.observe_spatial(rows[0], SpatialLifecycleReceipt("reentrant", rows[0].stamp, "ACTIVE")).reason == "reentrant_transaction"
        raise RuntimeError("synthetic bounded consumer failure")
    with pytest.raises(RuntimeError, match="synthetic bounded consumer failure"):
        owner.commit_spatial_member(intent, at_interval=context, consumer=failure)
    assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda *_: 0).disposition == "denied"
    assert owner.prepare_spatial_member(dto, reference, at_interval=context).reason == "spatial_member_measurement_consumed"


def test_spatial_household_conflict_separate_from_membership_and_unknown_peer_unchanged():
    resident = HouseholdDescriptor("resident", "resident-a", 9)
    rows = observation(household=resident), observation("family-room", 8)
    owner, rows, context = spatial_owner(rows)
    dto, = owner.spatial_memberships(at_interval=context)
    assert dto.household_claim is resident
    assert next(m for m in dto.members if m.observation.member.camera_id == "family-room").household.state == "unknown"
    intent = owner.prepare_spatial_member(dto, spatial_reference(rows[0], context), at_interval=context).reservation
    duplicate = observation("living-room", 42, household=resident)
    owner.observe_spatial(duplicate, SpatialLifecycleReceipt("duplicate-owner-original", duplicate.stamp, "ACTIVE"))
    calls = []
    denied = owner.commit_spatial_member(intent, at_interval=context, consumer=lambda *_: calls.append(1))
    assert denied.household_conflict is not None and denied.reason == "settled_household_claim_conflict"
    assert calls == [] and owner.ledger.stats().active_persons == 2
    context = spatial_context(rows + (duplicate,))
    dtos = owner.spatial_memberships(at_interval=context)
    assert all(d.household_claim is None and d.household_conflict is not None for d in dtos)


def test_spatial_source_media_clock_is_local_only_and_reset_rewind_denies_old_context():
    sources = tuple(replace(b, clock_basis="source-media", clock_domain=f"source-{b.source_id}-pts") for b in SOURCES)
    owner = IdentityAssociationService.for_spatial("run-a", spatial_policy(sources=sources))
    row = observation(time=5, media=5)
    row = replace(row, stamp=replace(row.stamp, clock_domain="source-1-pts", clock_basis="source-media"))
    owner.observe_spatial(row, SpatialLifecycleReceipt("native-local-PTS", row.stamp, "ACTIVE"))
    context = spatial_context((row,), sources=sources)
    dto, = owner.spatial_memberships(at_interval=context)
    assert owner.prepare_spatial_member(dto, spatial_reference(row, context), at_interval=context).disposition == "reserved"
    later = spatial_context((row,), sources=sources, evaluation=5.1)
    owner.spatial_memberships(at_interval=later)
    with pytest.raises(ValueError, match="spatial_clock_rewind"):
        owner.spatial_memberships(at_interval=context)
    owner.reset_source(replace(sources[1], source_epoch=1))
    with pytest.raises(ValueError, match="spatial_clock_binding"):
        owner.spatial_memberships(at_interval=later)


def test_spatial_final_revision_two_people_A_only_B_original_and_bounded_member_updates():
    rows = (observation(), observation("family-room", 8), observation(raw=17), observation("family-room", 18))
    owner, rows, context = spatial_owner(rows)
    dtos = owner.spatial_memberships(at_interval=context)
    assert len(dtos) == 2 and len({d.association_revision for d in dtos}) == 1
    original_B = next(d for d in dtos if any(m.observation.member.raw_id == 17 for m in d.members))
    original_A = next(d for d in dtos if d is not original_B)
    physical = original_A.physical_key
    old_reference = spatial_reference(rows[0], context)
    old_intent = owner.prepare_spatial_member(original_A, old_reference, at_interval=context).reservation
    for frame in range(2, 162):  # New bounded frontier regression, not a long soak.
        row = observation(frame=frame, time=10 + frame / 30, media=5 + frame / 30)
        owner.observe_spatial(row, SpatialLifecycleReceipt("native:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
        context = spatial_context((row, rows[1], rows[2], rows[3]), evaluation=row.stamp.acquired_at_s)
        current = owner.spatial_memberships(at_interval=context)
        current_A = next(d for d in current if d.physical_key == physical)
        current_B = next(d for d in current if d.physical_key == original_B.physical_key)
        assert current_B.members == original_B.members
        reference = spatial_reference(row, context)
        intent = owner.prepare_spatial_member(current_A, reference, at_interval=context).reservation
        assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda *_: physical).consumer_value == physical
        owner.compact_history()
        assert owner.prepare_spatial_member(current_A, old_reference, at_interval=context).disposition == "denied"
        assert owner.prepare_spatial_member(original_B, spatial_reference(rows[2], context), at_interval=context).disposition == "denied"
        assert owner.stats().spatial_member_frontiers <= 128 and owner.stats().current_dtos == 2
        assert owner.stats().pending_reservations == 0 and owner.stats().clock_frontiers <= 3
        assert owner.ledger.stats().receipts + owner.ledger.stats().retired_keys <= 128
    assert owner.commit_spatial_member(old_intent, at_interval=context, consumer=lambda *_: 0).disposition == "denied"


def test_spatial_default_strict_route_stays_denied_without_admission():
    owner = IdentityAssociationService("run-a", ProducerAdmission("unavailable", "real-unavailable", "no-qualified-common-time"))
    assert owner.ledger is None
    row = observation()
    assert owner.observe(row).disposition == "unavailable"
    assert owner.observe_spatial(row, SpatialLifecycleReceipt("native", row.stamp, "ACTIVE")).disposition == "unavailable"


def current_visual_owner(rows=None, **policy_changes):
    owner = IdentityAssociationService.for_current_visual("run-a", spatial_policy(**policy_changes))
    rows = rows or (observation(), observation("family-room", 8))
    for row in rows:
        owner.observe_spatial(row, SpatialLifecycleReceipt("native:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
    return owner, rows


def test_current_visual_explicit_factory_and_old_strict_routes_are_not_silently_enabled():
    owner, rows = current_visual_owner()
    candidate = visual_proof(rows)
    context, delivery = spatial_context(rows), visual_delivery(rows)
    strict = IdentityAssociationService.for_spatial("run-a", spatial_policy())
    assert strict.join_current_visual(candidate, at_interval=context, delivery_context=delivery).reason == "current_visual_opt_in_required"
    assert owner.join_spatial(spatial_person_proof(rows), at_interval=spatial_context(rows, qualified=True)).reason == "strict_spatial_factory_required"
    assert owner.join_current_visual(candidate, at_interval=context, delivery_context=delivery).association_basis == "current_visual"
    dto, = owner.spatial_memberships(at_interval=context)
    assert dto.association_basis == "current_visual"
    assert len(dto.members) == 2 and all(e.eligible for e in dto.eligibility)
    assert not any(hasattr(dto, f) for f in ("position", "covariance", "confidence", "common_domain", "overlap_permit"))


@pytest.mark.parametrize("first", ["kitchen", "family-room"])
def test_current_visual_same_person_any_source_first_second_reversal_without_group_change(first):
    owner, rows = current_visual_owner()
    context = spatial_context(rows)
    owner.join_current_visual(visual_proof(rows), at_interval=context, delivery_context=visual_delivery(rows))
    dto, = owner.spatial_memberships(at_interval=context)
    key = dto.physical_key
    selected = next(r for r in rows if r.stamp.member.camera_id == first)
    reference = spatial_reference(selected, context)
    intent = owner.prepare_spatial_member(dto, reference, at_interval=context).reservation
    assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda d, r: r).consumer_value is reference
    for sequence, camera in ((2, "family-room" if first == "kitchen" else "kitchen"), (3, first)):
        updated = tuple(observation(r.stamp.member.camera_id, r.stamp.member.raw_id, frame=sequence,
                                    time=10 + sequence / 30, media=5 + sequence / 30) for r in rows)
        for row in updated:
            owner.observe_spatial(row, SpatialLifecycleReceipt("native:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
        context = spatial_context(updated)
        owner.join_current_visual(visual_proof(updated), at_interval=context,
                                  delivery_context=visual_delivery(updated, sequence=sequence, evaluation=100 + sequence / 30))
        dto, = owner.spatial_memberships(at_interval=context)
        assert dto.physical_key == key
        reference = spatial_reference(next(r for r in updated if r.stamp.member.camera_id == camera), context)
        intent = owner.prepare_spatial_member(dto, reference, at_interval=context).reservation
        assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda d, r: r).consumer_value is reference


def test_current_visual_two_supported_pairs_final_revision_and_next_frame_B_original():
    rows = (observation(), observation("family-room", 8), observation(raw=17), observation("family-room", 18))
    owner, rows = current_visual_owner(rows)
    context = spatial_context(rows)
    grouped = owner.join_current_visual(visual_proof(rows), at_interval=context, delivery_context=visual_delivery(rows))
    assert grouped.disposition == "accepted" and len(grouped.retired_keys) == 2
    dtos = owner.spatial_memberships(at_interval=context)
    assert len(dtos) == 2 and len({d.association_revision for d in dtos}) == 1
    B = next(d for d in dtos if any(m.observation.member.raw_id == 17 for m in d.members))
    A = next(d for d in dtos if d is not B)
    old_intent = owner.prepare_spatial_member(A, spatial_reference(rows[0], context), at_interval=context).reservation
    current = observation(frame=2, time=10.1, media=5.1)
    owner.observe_spatial(current, SpatialLifecycleReceipt("native-A-next", current.stamp, "ACTIVE"))
    context = spatial_context((current,) + rows[1:], evaluation=10.1)
    current_dtos = owner.spatial_memberships(at_interval=context)
    assert next(d for d in current_dtos if d.physical_key == B.physical_key).members == B.members
    assert len({d.association_revision for d in current_dtos}) == 1
    assert owner.commit_spatial_member(old_intent, at_interval=context, consumer=lambda *_: 0).disposition == "denied"


@pytest.mark.parametrize("scores", [{(7, 8): .9, (17, 8): .9}, {(7, 8): .9, (7, 18): .9, (17, 8): .9, (17, 18): .9}])
def test_current_visual_similar_nearby_crossing_and_ties_do_not_pick_an_arbitrary_winner(scores):
    rows = (observation(), observation("family-room", 8), observation(raw=17), observation("family-room", 18))
    owner, rows = current_visual_owner(rows)
    result = owner.join_current_visual(visual_proof(rows, scores=scores), at_interval=spatial_context(rows), delivery_context=visual_delivery(rows))
    assert result.disposition == "observed" and owner.ledger.stats().active_persons == 4
    assert all(d.association_basis == "source_local" for d in owner.spatial_memberships(at_interval=spatial_context(rows)))


def test_current_visual_fresh_ambiguity_revokes_old_group_and_preserves_consumed_member_frontier():
    owner, rows = current_visual_owner()
    context = spatial_context(rows)
    owner.join_current_visual(visual_proof(rows), at_interval=context, delivery_context=visual_delivery(rows))
    old, = owner.spatial_memberships(at_interval=context)
    selected = spatial_reference(rows[1], context)
    intent = owner.prepare_spatial_member(old, selected, at_interval=context).reservation
    owner.commit_spatial_member(intent, at_interval=context, consumer=lambda *_: 1)
    pending = owner.prepare_spatial_member(old, spatial_reference(rows[0], context), at_interval=context).reservation
    competitor = observation(raw=17)
    owner.observe_spatial(competitor, SpatialLifecycleReceipt("competing-original-native", competitor.stamp, "ACTIVE"))
    all_rows = rows + (competitor,)
    result = owner.join_current_visual(visual_proof(all_rows, scores={(17, 8): .9}),
                                       at_interval=spatial_context(all_rows), delivery_context=visual_delivery(all_rows, sequence=2))
    assert rows[1].stamp.member in result.withdrawn_members
    assert owner.ledger.stats().active_persons == 3
    assert owner.ledger._closed == {} and owner.ledger._lifecycle_frontiers == {}
    dtos = owner.spatial_memberships(at_interval=spatial_context(all_rows))
    F = next(d for d in dtos if d.members[0].observation == rows[1].stamp)
    assert F.physical_key != old.physical_key and F.physical_key.serial > 2
    assert owner.prepare_spatial_member(F, selected, at_interval=spatial_context(all_rows)).reason == "spatial_member_measurement_consumed"
    assert owner.commit_spatial_member(pending, at_interval=context, consumer=lambda *_: 1).disposition == "denied"


def test_current_visual_unequal_source_pts_origins_are_not_relabelled_common_time():
    sources = tuple(replace(b, clock_basis="source-media", clock_domain=f"own-pts-{b.source_id}") for b in SOURCES)
    left, right = observation(time=5., media=5.), observation("family-room", 8, time=500., media=500.)
    rows = tuple(replace(r, stamp=replace(r.stamp, clock_basis="source-media", clock_domain=f"own-pts-{r.stamp.member.source_id}")) for r in (left, right))
    owner, rows = current_visual_owner(rows, sources=sources)
    context = spatial_context(rows, sources=sources)
    result = owner.join_current_visual(visual_proof(rows), at_interval=context, delivery_context=visual_delivery(rows))
    assert result.disposition == "accepted"
    dto, = owner.spatial_memberships(at_interval=context)
    assert {m.observation.acquired_at_s for m in dto.members} == {5., 500.}
    assert context.comparability_receipt_id is None and context.offsets == ()


@pytest.mark.parametrize("boundary", ["absence", "empty", "termination", "correction", "epoch", "run"])
def test_current_visual_real_boundaries_and_queued_intents_never_leak(boundary):
    owner, rows = current_visual_owner()
    context = spatial_context(rows)
    owner.join_current_visual(visual_proof(rows), at_interval=context, delivery_context=visual_delivery(rows))
    dto, = owner.spatial_memberships(at_interval=context)
    pending = owner.prepare_spatial_member(dto, spatial_reference(rows[0], context), at_interval=context).reservation
    if boundary in {"absence", "termination"}:
        owner.withdraw(removal(rows[0], kind=boundary))
    elif boundary == "empty":
        owner.clear_source(SourceAbsence("empty-current-K", "kitchen", 1, 0, 2, 10.1, "capture-clock"))
    elif boundary == "correction":
        owner.transition(event(rows[0], rows[0], rows[1], kind="correction", time=10.05), rows[0], now())
    elif boundary == "epoch":
        owner.reset_source(replace(SOURCES[1], source_epoch=1))
    else:
        owner.reset_run("run-b")
    calls = []
    assert owner.commit_spatial_member(pending, at_interval=context, consumer=lambda *_: calls.append(1)).disposition == "denied"
    assert calls == []
    if boundary != "run":
        assert owner.ledger._members[rows[1].stamp.member].person_key == dto.physical_key


def test_current_visual_reappearance_after_real_grace_needs_current_proof_and_raw_reuse_is_not_alias():
    from reid.mv3dt_person_association import LifecycleRetention
    owner, rows = current_visual_owner()
    context = spatial_context(rows)
    owner.join_current_visual(visual_proof(rows), at_interval=context, delivery_context=visual_delivery(rows))
    old, = owner.spatial_memberships(at_interval=context)
    grace = LifecycleRetention("owned-visual-grace", rows[0].stamp, rows[0].stamp.member, "inactive", .35)
    owner.withdraw(removal(rows[0], retention=grace))
    returned = observation(frame=3, time=10.2, media=5.2)
    owner.observe_spatial(returned, SpatialLifecycleReceipt("returned-original", returned.stamp, "ACTIVE"))
    context = spatial_context((returned, rows[1]), evaluation=10.2)
    dto, = owner.spatial_memberships(at_interval=context)
    assert owner.prepare_spatial_member(dto, spatial_reference(returned, context), at_interval=context).reason == "current_rejoin_proof_required"
    data = (returned, rows[1])
    # Family still uses its exact first ingress; returning K has a new ingress.
    delivery = replace(visual_delivery(data, sequence=2, evaluation=100.2),
                       ingresses=(CurrentVisualIngress(returned.stamp, 100.2, returned.stamp, 100.2),
                                  CurrentVisualIngress(rows[1].stamp, 100., rows[1].stamp, 100.)))
    assert owner.join_current_visual(visual_proof(data), at_interval=context, delivery_context=delivery).disposition == "accepted"
    assert owner.spatial_memberships(at_interval=context)[0].physical_key == old.physical_key
    owner.withdraw(removal(returned, kind="termination", frame=4, time=10.3, media=5.3, token="terminated-visual"))
    reused = observation(frame=5, started=5, generation=2, time=10.4, media=5.4)
    owner.observe_spatial(reused, SpatialLifecycleReceipt("new-numeric-life", reused.stamp, "ACTIVE"))
    assert owner.ledger._members[reused.stamp.member].person_key != old.physical_key


def test_current_visual_household_truth_unchanged_and_two_same_label_people_are_refused():
    house = HouseholdDescriptor("resident", "original-resident", 9)
    rows = (observation(household=house), observation("family-room", 8))
    owner, rows = current_visual_owner(rows)
    context = spatial_context(rows)
    owner.join_current_visual(visual_proof(rows), at_interval=context, delivery_context=visual_delivery(rows))
    dto, = owner.spatial_memberships(at_interval=context)
    assert dto.household_claim is house and sum(m.household.stable_id is None for m in dto.members) == 1
    rows = (observation(household=house), observation("family-room", 8, household=house),
            observation(raw=17, household=house), observation("family-room", 18, household=house))
    owner, rows = current_visual_owner(rows)
    owner.join_current_visual(visual_proof(rows), at_interval=spatial_context(rows), delivery_context=visual_delivery(rows))
    dtos = owner.spatial_memberships(at_interval=spatial_context(rows))
    assert len(dtos) == 2 and all(d.household_conflict is not None for d in dtos)
    for dto in dtos:
        row = next(r for r in rows if r.stamp == dto.members[0].observation)
        result = owner.prepare_spatial_member(dto, spatial_reference(row, spatial_context(rows)), at_interval=spatial_context(rows))
        assert result.reason == "settled_household_claim_conflict" and result.household_conflict is not None
    assert all(m.observation.household is house for m in owner.ledger._members.values())


def test_current_visual_distinct_lookalike_above_threshold_safety_counterexample():
    """Intentionally RED recognition regression: no hidden truth reaches estimator.

    Two independently labelled people can supply exactly the same observable
    isolated pair/score as a positive. R11 cannot discriminate this case.
    """
    independent_ground_truth = ("different-person-A", "different-person-B")
    owner, rows = current_visual_owner()
    candidate = visual_proof(rows, scores={(7, 8): .91}, token="isolated-distinct-lookalike")
    result = owner.join_current_visual(candidate, at_interval=spatial_context(rows), delivery_context=visual_delivery(rows))
    print("KNOWN FALSE MERGE:", independent_ground_truth, "cosine=.91", "K/raw7 + F/raw8",
          "original acquired=10.0; original/current ingress=100.0; eval=100.0;", result.disposition)
    assert owner.ledger.stats().active_persons == len(set(independent_ground_truth)), "R11 observable rule merged an isolated distinct look-alike pair"


def test_current_visual_prolonged_reinfer12_updates_keep_original_ages_and_all_state_bounded():
    owner = IdentityAssociationService.for_current_visual("run-a", spatial_policy())
    originals = None
    physical = None
    admitted = gaps = 0
    for frame in range(1, 1001):
        acquired, media, mono = 10 + frame / 30, 5 + frame / 30, 100 + frame / 30
        rows = (observation(frame=frame, time=acquired, media=media),
                observation("family-room", 8, frame=frame, time=acquired, media=media))
        for row in rows:
            owner.observe_spatial(row, SpatialLifecycleReceipt("native:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
        if (frame - 1) % 12 == 0:
            originals, original_mono = rows, mono
        result = owner.join_current_visual(visual_proof(rows, originals=originals, token=f"visual-{frame}"),
            at_interval=spatial_context(rows), delivery_context=visual_delivery(rows, sequence=frame, evaluation=mono,
                originals=originals, original_ingress=original_mono))
        admitted += result.disposition == "accepted"
        gaps += result.disposition == "association_unresolved"
        dto, = owner.spatial_memberships(at_interval=spatial_context(rows))
        physical = physical or dto.physical_key
        assert dto.physical_key == physical
        chosen = rows[frame % 2]
        reference = spatial_reference(chosen, spatial_context(rows))
        intent = owner.prepare_spatial_member(dto, reference, at_interval=spatial_context(rows)).reservation
        assert owner.commit_spatial_member(intent, at_interval=spatial_context(rows), consumer=lambda d, r: r).consumer_value is reference
        assert owner.prepare_spatial_member(dto, reference, at_interval=spatial_context(rows)).reason == "spatial_member_measurement_consumed"
        owner.compact_history()
        assert owner.ledger.stats().active_persons == 1 and owner.ledger.stats().allocated_serial == 2
        assert owner.ledger.stats().receipts + owner.ledger.stats().retired_keys <= 128
        assert len(owner.ledger._visual_ingresses) == 2 and len(owner.ledger._appearance_origins) == 2
        assert owner.stats().spatial_member_frontiers <= 128 and owner.stats().pending_reservations == 0
    assert admitted > 900 and gaps > 0
    print("visual endurance:", "ticks=1000", "supported=", admitted, "ordinary-reinfer-gaps=", gaps, "key=", physical)


def test_current_visual_stale_delivery_cannot_grant_an_unused_sample_with_a_frozen_source_clock():
    owner, rows = current_visual_owner()
    context = spatial_context(rows)
    owner.join_current_visual(visual_proof(rows), at_interval=context, delivery_context=visual_delivery(rows))
    dto, = owner.spatial_memberships(at_interval=context)
    unused = spatial_reference(rows[1], context)
    pending = owner.prepare_spatial_member(dto, unused, at_interval=context).reservation
    # Acquisition strings/PTS still look fresh locally. Actual process liveness
    # has advanced, and must NOT let an unconsumed dead-source sample through.
    owner.join_current_visual(visual_proof(rows), at_interval=context,
        delivery_context=visual_delivery(rows, sequence=2, evaluation=100.36, ingress=100., original_ingress=100.))
    current, = owner.spatial_memberships(at_interval=context)
    assert all(not e.eligible for e in current.eligibility)
    assert owner.prepare_spatial_member(current, unused, at_interval=context).reason == "current_visual_stale_member_delivery"
    calls = []
    assert owner.commit_spatial_member(pending, at_interval=context, consumer=lambda *_: calls.append(1)).disposition == "denied"
    assert calls == []


def test_current_visual_hard_settled_contradiction_never_uses_shared_labels_as_evidence():
    left = observation(household=HouseholdDescriptor("resident", "resident-A", 9))
    right = observation("family-room", 8, household=HouseholdDescriptor("visitor", "visitor-B", 10, 4))
    owner, rows = current_visual_owner((left, right))
    result = owner.join_current_visual(visual_proof(rows, scores={(7, 8): .99}),
                                       at_interval=spatial_context(rows), delivery_context=visual_delivery(rows))
    assert result.reason == "current_visual_household_conflict"
    assert owner.ledger.stats().active_persons == 2
    assert owner.ledger._members[left.stamp.member].observation.household is left.household
    assert owner.ledger._members[right.stamp.member].observation.household is right.household


def test_current_visual_split_capacity_refuses_without_dropping_members_or_raising_limits():
    owner, rows = current_visual_owner(max_entities=2, max_history=4)
    owner.join_current_visual(visual_proof(rows), at_interval=spatial_context(rows), delivery_context=visual_delivery(rows))
    competitor = observation("living-room", 42)
    owner.observe_spatial(competitor, SpatialLifecycleReceipt("native-L", competitor.stamp, "ACTIVE"))
    rows = (observation(frame=2, time=10.1, media=5.1),
            observation("family-room", 8, frame=2, time=10.1, media=5.1))
    for row in rows:
        owner.observe_spatial(row, SpatialLifecycleReceipt("fresh-contradiction:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
    all_rows = rows + (competitor,)
    context = spatial_context(all_rows, evaluation=10.1)
    # Living is LOCAL, but its real entity consumes the existing capacity.
    delivery = visual_delivery(rows, sequence=2, evaluation=100.1)
    delivery = replace(delivery, ingresses=delivery.ingresses + (CurrentVisualIngress(competitor.stamp, 100., competitor.stamp, 100.),))
    result = owner.join_current_visual(visual_proof(rows, scores={(7, 8): .1}), at_interval=context,
                                       delivery_context=delivery)
    assert result.reason == "current_visual_split_capacity" and result.disposition == "capacity"
    assert set(result.withdrawn_members) == {r.stamp.member for r in rows}
    assert owner.ledger.stats().active_persons == 2 and owner.ledger.stats().active_members == 3
    dtos = owner.spatial_memberships(at_interval=context)
    grouped = next(d for d in dtos if len(d.members) == 2)
    assert grouped.association_basis == "current_visual_unresolved" and all(not e.eligible for e in grouped.eligibility)
    assert owner.ledger.policy.max_entities == 2 and owner.ledger.policy.max_history == 4


def test_current_visual_repeated_truthful_split_rejoin_has_no_retired_key_resurrection_or_history_growth():
    owner = IdentityAssociationService.for_current_visual("run-a", spatial_policy())
    continuing = None
    previous_key = 0
    for frame in range(1, 301):
        rows = (observation(frame=frame, time=10 + frame / 30, media=5 + frame / 30),
                observation("family-room", 8, frame=frame, time=10 + frame / 30, media=5 + frame / 30))
        for row in rows:
            owner.observe_spatial(row, SpatialLifecycleReceipt("native:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
        low = frame % 2 == 0
        context = spatial_context(rows)
        result = owner.join_current_visual(visual_proof(rows, scores={(7, 8): .2 if low else .9}), at_interval=context,
            delivery_context=visual_delivery(rows, sequence=frame, evaluation=100 + frame / 30))
        dtos = owner.spatial_memberships(at_interval=context)
        continuing = continuing or dtos[0].physical_key
        assert continuing in {d.physical_key for d in dtos}
        assert len(dtos) == (2 if low else 1)
        if low:
            fresh = max(d.physical_key.serial for d in dtos)
            assert fresh > previous_key
            previous_key = fresh
        else:
            assert all(k.serial <= owner.ledger._retired_serial_frontier for k in result.retired_keys)
        assert owner.ledger.stats().receipts == owner.ledger.stats().retired_keys == 0
        assert len(owner.ledger._visual_ingresses) == 2
        assert len({d.association_revision for d in dtos}) == 1


@pytest.mark.parametrize("defect", ["copied-dto", "copied-intent", "discarded", "forged-basis", "callback-failure"])
def test_current_visual_provenance_never_bypasses_exact_intent_and_atomic_callback_rules(defect):
    owner, rows = current_visual_owner()
    context = spatial_context(rows)
    owner.join_current_visual(visual_proof(rows), at_interval=context, delivery_context=visual_delivery(rows))
    dto, = owner.spatial_memberships(at_interval=context)
    reference = spatial_reference(rows[0], context)
    calls = []
    if defect in {"copied-dto", "forged-basis"}:
        forged = replace(dto, association_basis="qualified_spatial") if defect == "forged-basis" else replace(dto)
        assert owner.prepare_spatial_member(forged, reference, at_interval=context).disposition == "denied"
    else:
        intent = owner.prepare_spatial_member(dto, reference, at_interval=context).reservation
        if defect == "copied-intent":
            intent = replace(intent)
        elif defect == "discarded":
            owner.discard_spatial_member(intent)
        else:
            def failure(dto, reference):
                calls.append(1)
                assert owner.observe_spatial(rows[0], SpatialLifecycleReceipt("reentrant", rows[0].stamp, "ACTIVE")).reason == "reentrant_transaction"
                raise RuntimeError("new visual consumer failed")
            with pytest.raises(RuntimeError, match="new visual consumer failed"):
                owner.commit_spatial_member(intent, at_interval=context, consumer=failure)
            assert owner.prepare_spatial_member(dto, reference, at_interval=context).reason == "spatial_member_measurement_consumed"
        assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda *_: calls.append(1)).disposition == "denied"
    assert calls == ([1] if defect == "callback-failure" else [])


def physical_quarantine_owner(kind="subject"):
    """Two synthetic distinct source-local people; no visual grouping or World."""
    left_house = HouseholdDescriptor("resident", "resident-a", 9)
    right_house = HouseholdDescriptor("resident", "resident-a" if kind == "subject" else "resident-b",
                                      10 if kind == "subject" else 9)
    rows = (observation(household=left_house), observation(raw=8, household=right_house),
            observation("living-room", 42, household=HouseholdDescriptor("resident", "unaffected-c", 11)))
    owner, _, context = spatial_owner(rows, join=False)
    return owner, rows, context, owner.spatial_memberships(at_interval=context)


@pytest.mark.parametrize("kind", ["subject", "sid"])
def test_physical_quarantine_original_conflict_purpose_two_distinct_keys_and_strict_unaffected(kind):
    owner, rows, context, dtos = physical_quarantine_owner(kind)
    assert callable(getattr(owner, "prepare_spatial_physical_member", None)), "R13 public physical prepare missing"
    assert callable(getattr(owner, "commit_spatial_physical_member", None)), "R13 public physical commit missing"
    assert len({d.association_revision for d in dtos}) == 1
    assert len({d.physical_key for d in dtos}) == 3
    before = owner.ledger.stats()
    owners = owner.ledger.household_claim_owners()
    calls = []
    for row in rows:
        dto = next(d for d in dtos if d.members[0].observation == row.stamp)
        reference = spatial_reference(row, context)
        if row is rows[2]:
            intent = owner.prepare_spatial_member(dto, reference, at_interval=context).reservation
            assert intent.purpose == "strict" and intent.household_conflict is None
            strict = owner.commit_spatial_member(intent, at_interval=context, consumer=lambda *_: "ordinary")
            assert strict.disposition == "committed" and strict.purpose == "strict"
            continue
        conflict = dto.household_conflict
        assert type(conflict) is HouseholdClaimConflict and dto.household_claim is None
        assert {original for _, original in conflict.owners} == set(rows[:2])
        assert owner.prepare_spatial_member(dto, reference, at_interval=context).reason == "settled_household_claim_conflict"
        prepared = owner.prepare_spatial_physical_member(dto, reference, at_interval=context)
        intent = prepared.reservation
        assert prepared.disposition == "reserved" and prepared.purpose == "physical_only_quarantine"
        assert prepared.household_conflict is intent.household_conflict is conflict
        assert intent.purpose == "physical_only_quarantine" and intent.dto is dto and intent.reference is reference
        with pytest.raises(FrozenInstanceError):
            intent.purpose = "strict"
        result = owner.commit_spatial_physical_member(intent, at_interval=context,
            consumer=lambda d, r: calls.append((d.physical_key, r)) or r)
        assert result.disposition == "committed" and result.consumer_value is reference
        assert result.purpose == "physical_only_quarantine" and result.household_conflict is conflict
        assert owner.prepare_spatial_physical_member(dto, reference, at_interval=context).reason == "spatial_member_measurement_consumed"
        assert owner.commit_spatial_physical_member(intent, at_interval=context, consumer=lambda *_: 0).disposition == "denied"
    assert len(calls) == 2 and calls[0][0] != calls[1][0]
    assert owner.ledger.household_claim_owners() == owners and owner.ledger.stats() == before
    assert all(owner.ledger._members[row.stamp.member].observation.household is row.household for row in rows)
    assert owner.ledger.policy.max_entities == 64 and owner.ledger.policy.max_history == 128
    assert owner.stats().pending_reservations == 0 and owner.stats().spatial_member_frontiers == 3


def test_physical_quarantine_requires_conflict_and_purpose_is_not_interchangeable():
    owner, rows, context, dtos = physical_quarantine_owner()
    a = next(d for d in dtos if d.members[0].observation == rows[0].stamp)
    c = next(d for d in dtos if d.members[0].observation == rows[2].stamp)
    reference, normal = spatial_reference(rows[0], context), spatial_reference(rows[2], context)
    assert owner.prepare_spatial_physical_member(c, normal, at_interval=context).reason == "physical_quarantine_conflict_required"
    physical = owner.prepare_spatial_physical_member(a, reference, at_interval=context).reservation
    strict = owner.prepare_spatial_member(c, normal, at_interval=context).reservation
    assert owner.commit_spatial_member(physical, at_interval=context, consumer=lambda *_: pytest.fail("wrong purpose")).reason == "spatial_reservation_purpose_mismatch"
    assert owner.commit_spatial_physical_member(strict, at_interval=context, consumer=lambda *_: pytest.fail("wrong purpose")).reason == "spatial_reservation_purpose_mismatch"
    assert owner.stats().pending_reservations == 2 and owner.stats().spatial_member_frontiers == 0
    discarded = owner.discard_spatial_member(physical)
    assert discarded.purpose == "physical_only_quarantine" and discarded.household_conflict is a.household_conflict
    assert owner.commit_spatial_physical_member(physical, at_interval=context, consumer=lambda *_: 0).disposition == "denied"
    replacement = owner.prepare_spatial_physical_member(a, reference, at_interval=context).reservation
    assert owner.commit_spatial_physical_member(replacement, at_interval=context, consumer=lambda d, r: r).consumer_value is reference
    assert owner.commit_spatial_member(strict, at_interval=context, consumer=lambda *_: "ordinary").consumer_value == "ordinary"
    assert SpatialMemberResult("denied", "old-positional-example").purpose == "strict"
    legacy = SpatialMemberReservation(owner._owner, 999, c, normal, context)
    assert legacy.purpose == "strict" and legacy.household_conflict is None


@pytest.mark.parametrize("defect", ["copy-dto", "copy-conflict", "strip-conflict", "foreign-dto", "copy-intent", "foreign-intent", "forged-purpose", "forged-conflict"])
def test_physical_quarantine_owner_object_and_purpose_provenance(defect):
    owner, rows, context, dtos = physical_quarantine_owner()
    dto = dtos[0]
    reference = spatial_reference(rows[0], context)
    calls = []
    if defect in {"copy-dto", "copy-conflict", "strip-conflict", "foreign-dto"}:
        other = owner
        changed = replace(dto)
        if defect == "copy-conflict":
            changed = replace(dto, household_conflict=replace(dto.household_conflict))
        elif defect == "strip-conflict":
            changed = replace(dto, household_conflict=None)
        elif defect == "foreign-dto":
            other, _, _, _ = physical_quarantine_owner()
            changed = dto
        assert other.prepare_spatial_physical_member(changed, reference, at_interval=context).disposition == "denied"
    else:
        intent = owner.prepare_spatial_physical_member(dto, reference, at_interval=context).reservation
        if defect == "foreign-intent":
            other, _, _, _ = physical_quarantine_owner()
        else:
            other = owner
            intent = replace(intent, **({"purpose": "strict"} if defect == "forged-purpose" else
                                        {"household_conflict": replace(intent.household_conflict)} if defect == "forged-conflict" else {}))
        assert other.commit_spatial_physical_member(intent, at_interval=context, consumer=lambda *_: calls.append(1)).disposition == "denied"
    assert calls == [] and owner.stats().spatial_member_frontiers == 0


@pytest.mark.parametrize("defect", ["owner", "epoch", "birth", "media", "acquisition", "origin", "units", "calibration", "transform", "stage", "source-interval"])
def test_physical_quarantine_original_measurement_checks_cannot_be_bypassed(defect):
    owner, rows, context, dtos = physical_quarantine_owner()
    reference = spatial_reference(rows[0], context)
    if defect == "owner":
        reference = replace(reference, owner=rows[1].stamp.member)
    elif defect == "epoch":
        reference = replace(reference, stamp=replace(reference.stamp, member=replace(reference.owner, source_epoch=1)))
    elif defect in {"birth", "media", "acquisition"}:
        name, value = {"birth": ("lifecycle_started_frame", 0), "media": ("media_time_s", 999.), "acquisition": ("acquired_at_s", 10.1)}[defect]
        reference = replace(reference, stamp=replace(reference.stamp, **{name: value}))
    elif defect == "source-interval":
        reference = replace(reference, acquisition_interval=replace(reference.acquisition_interval, acquisition_min_s=9.9))
    else:
        name, value = {"origin": ("origin", "held"), "units": ("units", "cm"), "calibration": ("calibration_revision", "other"),
                       "transform": ("transform_id", "other"), "stage": ("measured_stage_id", "")}[defect]
        reference = replace(reference, **{name: value})
    result = owner.prepare_spatial_physical_member(dtos[0], reference, at_interval=context)
    assert result.disposition == "denied" and result.reservation is None
    assert result.household_conflict is dtos[0].household_conflict
    assert owner.stats().spatial_member_frontiers == owner.stats().pending_reservations == 0


@pytest.mark.parametrize("state", ["INACTIVE", "TENTATIVE", "UNKNOWN"])
def test_physical_quarantine_nonactive_original_never_receives_a_grant(state):
    owner, rows, context, _ = physical_quarantine_owner()
    owner.observe_spatial(rows[0], SpatialLifecycleReceipt("native-nonactive", rows[0].stamp, state))
    dtos = owner.spatial_memberships(at_interval=context)
    result = owner.prepare_spatial_physical_member(dtos[0], spatial_reference(rows[0], context), at_interval=context)
    assert result.reason == "native_active_required" and result.reservation is None


@pytest.mark.parametrize("boundary", ["age", "update", "conflict-resolved", "conflict-grown", "termination", "empty", "epoch", "run"])
def test_physical_quarantine_queued_intent_revalidates_revision_conflict_and_lifecycle(boundary):
    owner, rows, context, dtos = physical_quarantine_owner()
    intent = owner.prepare_spatial_physical_member(dtos[0], spatial_reference(rows[0], context), at_interval=context).reservation
    if boundary == "age":
        context = spatial_context(rows, evaluation=10.36)
    elif boundary in {"update", "conflict-resolved", "conflict-grown"}:
        changed = tuple(observation(r.stamp.member.camera_id, r.stamp.member.raw_id, frame=2, time=10.1, media=5.1,
            household=HouseholdDescriptor("resident", "corrected-b", 20) if boundary == "conflict-resolved" and r is rows[1] else r.household) for r in rows)
        if boundary == "conflict-grown":
            changed += (observation(raw=9, frame=2, time=10.1, media=5.1, household=rows[0].household),)
        for r in changed:
            owner.observe_spatial(r, SpatialLifecycleReceipt("fresh:" + r.stamp.observation_id, r.stamp, "ACTIVE"))
        context = spatial_context(changed)
        current = owner.spatial_memberships(at_interval=context)
        assert len({d.association_revision for d in current}) == 1
        assert current[0].physical_key == dtos[0].physical_key
        if boundary == "conflict-resolved":
            assert current[0].household_conflict is None
            assert owner.prepare_spatial_physical_member(current[0], spatial_reference(changed[0], context), at_interval=context).reason == "physical_quarantine_conflict_required"
            assert owner.prepare_spatial_member(current[0], spatial_reference(changed[0], context), at_interval=context).disposition == "reserved"
    elif boundary == "termination":
        owner.withdraw(removal(rows[0], kind="termination"))
    elif boundary == "empty":
        owner.clear_source(SourceAbsence("real-empty-K", "kitchen", 1, 0, 2, 10.1, "capture-clock"))
    elif boundary == "epoch":
        owner.reset_source(replace(SOURCES[1], source_epoch=1))
    else:
        owner.reset_run("run-b")
    calls = []
    assert owner.commit_spatial_physical_member(intent, at_interval=context, consumer=lambda *_: calls.append(1)).disposition == "denied"
    assert calls == [] and owner.stats().spatial_member_frontiers == 0


@pytest.mark.parametrize("defect", ["incomplete", "changed", "overflow"])
def test_physical_quarantine_recomputes_complete_ownership_without_release_or_truncation(monkeypatch, defect):
    """Fault-injected derived owner view; overflow is not a reachable cohort claim."""
    owner, rows, context, dtos = physical_quarantine_owner()
    intent = owner.prepare_spatial_physical_member(dtos[0], spatial_reference(rows[0], context), at_interval=context).reservation
    originals = owner.ledger.household_claim_owners()
    if defect == "incomplete":
        injected = tuple(pair for pair in originals if pair[1] is not rows[1])
    elif defect == "changed":
        injected = tuple((key, replace(row, household=replace(row.household, stable_id=99))) if row is rows[1] else (key, row)
                         for key, row in originals)
    else:
        injected = originals + tuple((dtos[0].physical_key, observation(raw=100 + i, household=rows[0].household)) for i in range(128))
    before = owner.ledger.stats()
    monkeypatch.setattr(owner.ledger, "household_claim_owners", lambda: injected)
    reference = spatial_reference(rows[0], context)
    prepared = owner.prepare_spatial_physical_member(dtos[0], reference, at_interval=context)
    committed = owner.commit_spatial_physical_member(intent, at_interval=context, consumer=lambda *_: pytest.fail("incomplete owners"))
    reason = "household_claim_capacity" if defect == "overflow" else "physical_quarantine_conflict_changed"
    assert prepared.reason == committed.reason == reason
    assert prepared.household_conflict is committed.household_conflict is dtos[0].household_conflict
    assert owner.ledger.stats() == before and owner.ledger.household_claim_owners() is injected
    assert owner.stats().pending_reservations == owner.stats().spatial_member_frontiers == 0


@pytest.mark.parametrize("constraint", ["unresolved", "rejected"])
def test_physical_quarantine_does_not_override_noneligible_household_constraint(constraint):
    owner, rows, context, _ = physical_quarantine_owner()
    changed = observation(frame=2, time=10.1, media=5.1, household=replace(rows[0].household, constraint=constraint))
    owner.observe_spatial(changed, SpatialLifecycleReceipt("original-household-decision", changed.stamp, "ACTIVE"))
    context = spatial_context((changed,) + rows[1:])
    dtos = owner.spatial_memberships(at_interval=context)
    assert owner.prepare_spatial_physical_member(dtos[0], spatial_reference(changed, context), at_interval=context).reason == "spatial_household_conflict"


def test_physical_quarantine_does_not_override_within_group_household_contradiction():
    resident = HouseholdDescriptor("resident", "resident-a", 9)
    rows = observation(household=resident), observation("family-room", 8, household=resident)
    owner, _, _ = spatial_owner(rows)
    peer = observation("living-room", 42, household=resident)
    changed = observation("family-room", 8, frame=2, time=10.1, media=5.1,
                          household=HouseholdDescriptor("resident", "resident-b", 9))
    for r in (peer, changed):
        owner.observe_spatial(r, SpatialLifecycleReceipt("native:" + r.stamp.observation_id, r.stamp, "ACTIVE"))
    context = spatial_context((rows[0], peer, changed))
    dto = next(d for d in owner.spatial_memberships(at_interval=context) if len(d.members) == 2)
    assert dto.household_conflict is not None
    assert owner.prepare_spatial_physical_member(dto, spatial_reference(rows[0], context), at_interval=context).reason == "spatial_household_conflict"


def test_physical_quarantine_reservation_bound_discard_and_invalid_consumer_do_not_consume_sample():
    owner, rows, context, dtos = physical_quarantine_owner()
    dto, reference = dtos[0], spatial_reference(rows[0], context)
    intents = [owner.prepare_spatial_physical_member(dto, reference, at_interval=context).reservation for _ in range(64)]
    capacity = owner.prepare_spatial_physical_member(dto, reference, at_interval=context)
    assert capacity.reason == "reservation_capacity" and capacity.household_conflict is dto.household_conflict
    assert owner.stats().pending_reservations == 64 and owner._max_reservations == 64
    for intent in intents:
        assert owner.discard_spatial_member(intent).purpose == "physical_only_quarantine"
    intent = owner.prepare_spatial_physical_member(dto, reference, at_interval=context).reservation
    assert owner.commit_spatial_physical_member(intent, at_interval=context, consumer=None).reason == "invalid_consumer"
    assert owner.stats().pending_reservations == owner.stats().spatial_member_frontiers == 0
    assert owner.prepare_spatial_physical_member(dto, reference, at_interval=context).disposition == "reserved"


def test_physical_quarantine_callback_failure_reentrancy_and_replay_consume_exactly_once():
    owner, rows, context, dtos = physical_quarantine_owner()
    dto, reference = dtos[0], spatial_reference(rows[0], context)
    intent = owner.prepare_spatial_physical_member(dto, reference, at_interval=context).reservation
    calls = []
    def failing_consumer(exact, original):
        calls.append((exact, original))
        assert owner.prepare_spatial_physical_member(dto, reference, at_interval=context).reason == "reentrant_transaction"
        assert owner.commit_spatial_physical_member(intent, at_interval=context, consumer=lambda *_: pytest.fail("reentrant")).disposition == "denied"
        assert owner.discard_spatial_member(intent).disposition == "denied"
        assert owner.observe_spatial(rows[0], SpatialLifecycleReceipt("reentrant", rows[0].stamp, "ACTIVE")).reason == "reentrant_transaction"
        raise RuntimeError("physical-only consumer failed")
    with pytest.raises(RuntimeError, match="physical-only consumer failed"):
        owner.commit_spatial_physical_member(intent, at_interval=context, consumer=failing_consumer)
    assert calls == [(dto, reference)] and owner._committing is False
    assert owner.stats().pending_reservations == 0 and owner.stats().spatial_member_frontiers == 1
    assert owner.prepare_spatial_physical_member(dto, reference, at_interval=context).reason == "spatial_member_measurement_consumed"
    assert owner.commit_spatial_physical_member(intent, at_interval=context, consumer=lambda *_: pytest.fail("replay")).disposition == "denied"


def test_physical_quarantine_two_current_members_any_original_reserve_commit_discard_example():
    """Runnable SAME physical key, two original source measurements; synthetic proof."""
    house = HouseholdDescriptor("resident", "resident-a", 9)
    rows = observation(household=house), observation("family-room", 8)
    owner, _, _ = spatial_owner(rows)
    distinct = observation("living-room", 42, household=house)
    owner.observe_spatial(distinct, SpatialLifecycleReceipt("native-distinct", distinct.stamp, "ACTIVE"))
    context = spatial_context(rows + (distinct,))
    dtos = owner.spatial_memberships(at_interval=context)
    dto = next(d for d in dtos if len(d.members) == 2)
    conflict, key = dto.household_conflict, dto.physical_key
    assert next(m for m in dto.members if m.observation == rows[1].stamp).household.state == "unknown"
    calls = []
    for row in reversed(rows):
        original = spatial_reference(row, context)
        intent = owner.prepare_spatial_physical_member(dto, original, at_interval=context).reservation
        assert owner.discard_spatial_member(intent).household_conflict is conflict
        intent = owner.prepare_spatial_physical_member(dto, original, at_interval=context).reservation
        result = owner.commit_spatial_physical_member(intent, at_interval=context,
            consumer=lambda d, r: calls.append((d.physical_key, r)) or r)
        assert result.consumer_value is original and result.household_conflict is conflict
    assert [k for k, _ in calls] == [key, key]
    assert [r.stamp.member.camera_id for _, r in calls] == ["family-room", "kitchen"]
    assert owner.ledger.stats().active_persons == 2 and len({d.association_revision for d in dtos}) == 1
    assert owner.stats().spatial_member_frontiers == 2 and owner.stats().pending_reservations == 0


def test_physical_quarantine_real_original_grace_holds_claim_until_exact_expiry_then_strict_recovers():
    from reid.mv3dt_person_association import LifecycleRetention
    owner, rows, context, dtos = physical_quarantine_owner()
    key = dtos[0].physical_key
    retention = LifecycleRetention("genuine-original-grace", rows[1].stamp, rows[1].stamp.member, "inactive", .35)
    owner.withdraw(removal(rows[1], frame=2, time=10.1, media=5.1, retention=retention))
    context = spatial_context((rows[0], rows[2]), evaluation=10.1)
    dto = next(d for d in owner.spatial_memberships(at_interval=context) if d.physical_key == key)
    assert dto.household_conflict is not None
    assert {r.stamp for _, r in dto.household_conflict.owners} == {r.stamp for r in rows[:2]}
    original_claims = owner.ledger.household_claim_owners()
    reference = spatial_reference(rows[0], context)
    intent = owner.prepare_spatial_physical_member(dto, reference, at_interval=context).reservation
    assert intent is not None and owner.ledger.household_claim_owners() == original_claims
    owner.expire_retention(1, 5.46)
    assert owner.commit_spatial_physical_member(intent, at_interval=context, consumer=lambda *_: pytest.fail("expired claim")).disposition == "denied"
    assert owner.stats().spatial_member_frontiers == 0
    fresh = observation(frame=3, time=10.46, media=5.46, household=rows[0].household)
    owner.observe_spatial(fresh, SpatialLifecycleReceipt("fresh-after-real-expiry", fresh.stamp, "ACTIVE"))
    context = spatial_context((fresh, rows[2]), evaluation=10.46)
    dto = next(d for d in owner.spatial_memberships(at_interval=context) if d.physical_key == key)
    assert dto.household_conflict is None and dto.household_claim is rows[0].household
    reference = spatial_reference(fresh, context)
    assert owner.prepare_spatial_physical_member(dto, reference, at_interval=context).reason == "physical_quarantine_conflict_required"
    strict = owner.prepare_spatial_member(dto, reference, at_interval=context).reservation
    assert owner.commit_spatial_member(strict, at_interval=context, consumer=lambda d, r: r).consumer_value is reference


def projective_service_fixture():
    selected, candidate, rows, context, delivery = projective_fixture()
    owner = IdentityAssociationService.for_projective_continuity("run-a", spatial_policy(), selected)
    for row in rows:
        owner.observe_spatial(row, SpatialLifecycleReceipt("native:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
    result = owner.join_projective_continuity(candidate, at_interval=context, delivery_context=delivery)
    assert result.disposition == "accepted", result
    return owner, selected, candidate, rows, context, delivery


def anatomical_service_fixture(*, qualified=True, start=1):
    selected,candidate,rows,context,delivery=anatomical_fixture(qualified=qualified,start=start)
    owner=IdentityAssociationService.for_anatomical_consensus('run-a',spatial_policy(),selected)
    for row in rows:owner.observe_spatial(row,SpatialLifecycleReceipt('native',row.stamp,'ACTIVE'))
    return owner,selected,candidate,rows,context,delivery


def test_anatomical_service_current_grant_old_intent_and_explicit_route():
    owner,selected,candidate,rows,context,delivery=anatomical_service_fixture()
    assert owner.join_current_visual(candidate.visual,at_interval=context,delivery_context=delivery).reason=='anatomical_consensus_route_required'
    assert owner.join_anatomical_consensus(candidate,at_interval=context,delivery_context=delivery).disposition=='accepted'
    dto,=owner.spatial_memberships(at_interval=context)
    assert dto.association_basis=='current_registered_anatomical_consensus' and all(e.eligible for e in dto.eligibility)
    old=owner.prepare_spatial_member(dto,spatial_reference(rows[0],context),at_interval=context).reservation
    _,fresh,advanced,new_context,new_delivery=anatomical_fixture(start=2)
    for row in advanced:owner.observe_spatial(row,SpatialLifecycleReceipt('native-next',row.stamp,'ACTIVE'))
    # Historical geometry is NOT permission to use an old DTO or old measurement.
    assert owner.commit_spatial_member(old,at_interval=new_context,consumer=lambda *_:pytest.fail('old intent')).disposition=='denied'
    pair=candidate.pairs[0]
    historical=replace(fresh,pairs=(replace(pair,left=advanced[0].stamp,right=advanced[1].stamp),))
    assert owner.join_anatomical_consensus(historical,at_interval=new_context,delivery_context=new_delivery).disposition=='accepted'
    current,=owner.spatial_memberships(at_interval=new_context)
    assert current.physical_key==dto.physical_key and all(e.eligible for e in current.eligibility)
    assert owner.prepare_spatial_member(current,spatial_reference(rows[0],context),at_interval=new_context).disposition=='denied'
    reference=spatial_reference(advanced[1],new_context)
    intent=owner.prepare_spatial_member(current,reference,at_interval=new_context).reservation
    assert owner.commit_spatial_member(intent,at_interval=new_context,consumer=lambda d,r:r).consumer_value is reference


def test_anatomical_service_no_static_qualification_or_no_body_cannot_accept_old_point91_rule():
    owner,selected,candidate,rows,context,delivery=anatomical_service_fixture(qualified=False)
    assert owner.join_anatomical_consensus(candidate,at_interval=context,delivery_context=delivery).reason=='anatomical_static_pair_qualification_unavailable'
    owner,selected,candidate,rows,context,delivery=anatomical_service_fixture()
    candidate=replace(candidate,visual=replace(visual_proof(rows,scores={(7,8):.91}),selected_pairs=()),pairs=())
    assert owner.join_anatomical_consensus(candidate,at_interval=context,delivery_context=delivery).disposition=='observed'
    assert owner.ledger.stats().active_persons==2


def test_anatomical_stationary_30_rolling_original_updates_preserve_key_and_bounds():
    owner,selected,candidate,rows,context,delivery=anatomical_service_fixture()
    key=None
    for tick in range(30):
        _,candidate,rows,context,delivery=anatomical_fixture(start=1+4*tick)
        for row in rows:owner.observe_spatial(row,SpatialLifecycleReceipt('continuous-native',row.stamp,'ACTIVE'))
        assert owner.join_anatomical_consensus(candidate,at_interval=context,delivery_context=delivery).disposition=='accepted'
        dto,=owner.spatial_memberships(at_interval=context)
        key=key or dto.physical_key
        assert dto.physical_key==key and all(e.eligible for e in dto.eligibility)
        reference=spatial_reference(rows[tick%2],context)
        intent=owner.prepare_spatial_member(dto,reference,at_interval=context).reservation
        assert owner.commit_spatial_member(intent,at_interval=context,consumer=lambda d,r:r).consumer_value is reference
        assert sum(len(m.anatomical_originals) for m in owner.ledger._members.values())==4
        assert owner.ledger.stats().receipts==0


@pytest.mark.parametrize('barrier',['contradiction','inactive','epoch','missing'])
def test_anatomical_correction_or_native_barrier_invalidates_old_intent(barrier):
    owner,selected,candidate,rows,context,delivery=anatomical_service_fixture()
    assert owner.join_anatomical_consensus(candidate,at_interval=context,delivery_context=delivery).disposition=='accepted'
    dto,=owner.spatial_memberships(at_interval=context)
    old=owner.prepare_spatial_member(dto,spatial_reference(rows[0],context),at_interval=context).reservation
    if barrier=='epoch':
        owner.reset_source(replace(SOURCES[1],source_epoch=1))
    elif barrier=='inactive':
        owner.observe_spatial(rows[0],SpatialLifecycleReceipt('inactive-native',rows[0].stamp,'INACTIVE'))
        owner.observe_spatial(rows[0],SpatialLifecycleReceipt('active-native',rows[0].stamp,'ACTIVE'))
        denied=owner.join_anatomical_consensus(candidate,at_interval=context,delivery_context=replace(delivery,sequence=2))
        assert denied.reason=='anatomical_original_binding_or_age'
    else:
        _,fresh,advanced,context,delivery=anatomical_fixture(start=2,right_offset=100. if barrier=='contradiction' else 0.)
        for row in advanced:owner.observe_spatial(row,SpatialLifecycleReceipt('next',row.stamp,'ACTIVE'))
        if barrier=='missing':fresh=replace(fresh,pairs=())
        result=owner.join_anatomical_consensus(fresh,at_interval=context,delivery_context=delivery)
        if barrier=='contradiction':assert owner.ledger.stats().active_persons==2 and result.withdrawn_members
        else:
            current,=owner.spatial_memberships(at_interval=context)
            assert not any(e.eligible for e in current.eligibility)
    assert owner.commit_spatial_member(old,at_interval=context,consumer=lambda *_:pytest.fail('old-intent')).disposition=='denied'


@pytest.mark.parametrize("defect", ["valid", "numeric-false", "age", "binding", "missing-endpoint", "stationary-seed"])
def test_projective_inspection_diagnostics_do_not_mutate_or_grant_even_with_live_reservation(defect):
    owner, selected, candidate, rows, context, delivery = projective_service_fixture()
    dto, = owner.spatial_memberships(at_interval=context)
    reference = spatial_reference(rows[0], context)
    intent = owner.prepare_spatial_member(dto, reference, at_interval=context).reservation
    pair = candidate.pairs[0]
    if defect == "numeric-false":
        first = replace(pair.left[0], original_point=(100., 1., 1.))
        candidate = replace(candidate, pairs=(replace(pair, left=(first,) + pair.left[1:]),))
    elif defect == "age":
        first = replace(pair.left[0], ingress_monotonic_s=99.)
        candidate = replace(candidate, pairs=(replace(pair, left=(first,) + pair.left[1:]),))
    elif defect == "binding":
        first = replace(pair.left[0], raster_id="wrong-original-raster")
        candidate = replace(candidate, pairs=(replace(pair, left=(first,) + pair.left[1:]),))
    elif defect == "missing-endpoint":
        delivery = replace(delivery, ingresses=delivery.ingresses[:1])
    elif defect == "stationary-seed":
        def stopped(side):
            return tuple(replace(s, bbox_xywh=side[-1].bbox_xywh, original_point=side[-1].original_point,
                                 observed_anchor_uv=side[-1].observed_anchor_uv) for s in side)
        candidate = replace(candidate, pairs=(replace(pair, left=stopped(pair.left), right=stopped(pair.right)),))
    # Appearance selection is deliberately absent: inspection is numeric only.
    candidate = replace(candidate, visual=replace(candidate.visual, candidates=(), selected_pairs=()))
    before = repr(vars(owner)), repr(vars(owner.ledger)), owner.ledger.stats()
    report = owner.inspect_projective_continuity(candidate, at_interval=context, delivery_context=delivery)
    assert (repr(vars(owner)), repr(vars(owner.ledger)), owner.ledger.stats()) == before
    if defect in {"age", "binding", "missing-endpoint"}:
        assert report.error == ("projective_current_endpoint" if defect == "missing-endpoint"
                                else "projective_original_binding_or_age")
        assert report.pairs == ()
    else:
        assert report.error is None
        assert report.pairs == ((rows[0].stamp.member, rows[1].stamp.member, defect == "valid"),)
    with pytest.raises(FrozenInstanceError):
        report.error = "changed"
    assert not hasattr(report, "frontiers") and not hasattr(report, "owner")
    assert owner.commit_spatial_member(report, at_interval=context,
        consumer=lambda *_: pytest.fail("diagnostic must not commit")).disposition == "denied"
    # The actual previously reserved original remains unconsumed and usable.
    assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda d, r: r).consumer_value is reference


def test_projective_inspection_requires_explicit_factory_and_blocks_reentrancy_without_mutation():
    owner, selected, candidate, rows, context, delivery = projective_service_fixture()
    strict = IdentityAssociationService.for_spatial("run-a", spatial_policy())
    assert strict.inspect_projective_continuity(candidate, at_interval=context,
        delivery_context=delivery).error == "projective_continuity_opt_in_required"
    owner._committing = True
    before = repr(vars(owner)), repr(vars(owner.ledger))
    assert owner.inspect_projective_continuity(candidate, at_interval=context,
        delivery_context=delivery).error == "reentrant_transaction"
    assert (repr(vars(owner)), repr(vars(owner.ledger))) == before


def test_projective_service_explicit_route_any_member_and_strict_defaults():
    owner, selected, candidate, rows, context, delivery = projective_service_fixture()
    assert owner.join_current_visual(candidate.visual, at_interval=context, delivery_context=delivery).reason == "projective_continuity_route_required"
    strict = IdentityAssociationService.for_spatial("run-a", spatial_policy())
    assert strict.join_projective_continuity(candidate, at_interval=context, delivery_context=delivery).reason == "projective_continuity_opt_in_required"
    dto, = owner.spatial_memberships(at_interval=context)
    assert dto.association_basis == "prior_projective_continuity" and all(e.eligible for e in dto.eligibility)
    assert dto.household_claim is None and dto.household_conflict is None
    for row in reversed(rows):
        original = spatial_reference(row, context)
        intent = owner.prepare_spatial_member(dto, original, at_interval=context).reservation
        assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda d, r: r).consumer_value is original
    assert owner.stats().pending_reservations == 0 and owner.stats().spatial_member_frontiers == 2


@pytest.mark.parametrize("change", ["missing", "contradiction", "epoch", "run", "expiry", "copy"])
def test_projective_service_old_intent_missing_support_split_reset_and_expiry(change):
    owner, selected, candidate, rows, context, delivery = projective_service_fixture()
    dto, = owner.spatial_memberships(at_interval=context)
    intent = owner.prepare_spatial_member(dto, spatial_reference(rows[0], context), at_interval=context).reservation
    if change in {"missing", "contradiction"}:
        _, fresh, rows, context, delivery = projective_fixture(start=2)
        for row in rows:
            owner.observe_spatial(row, SpatialLifecycleReceipt("fresh:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
        if change == "missing":
            fresh = replace(fresh, pairs=())
        else:
            pair = fresh.pairs[0]
            side = pair.left[:-1] + (replace(pair.left[-1], original_point=(100., 1., 1.)),)
            fresh = replace(fresh, pairs=(replace(pair, left=side),))
        result = owner.join_projective_continuity(fresh, at_interval=context, delivery_context=delivery)
        assert result.disposition == "association_unresolved"
        current = owner.spatial_memberships(at_interval=context)
        if change == "missing":
            assert len(current) == 1 and current[0].physical_key == dto.physical_key
            assert not any(e.eligible for e in current[0].eligibility)
        else:
            assert len(current) == 2 and dto.physical_key in {d.physical_key for d in current}
            assert max(d.physical_key.serial for d in current) == 3  # Retired2 never resurrects.
            assert result.withdrawn_members and len({d.association_revision for d in current}) == 1
    elif change == "epoch":
        owner.reset_source(replace(SOURCES[1], source_epoch=1))
    elif change == "run":
        owner.reset_run("run-b")
    elif change == "expiry":
        context = spatial_context(rows, evaluation=10.36)  # Endpoint fresh, oldest support expired.
    else:
        intent = replace(intent)
    calls = []
    assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda *_: calls.append(1)).disposition == "denied"
    assert calls == [] and owner.stats().spatial_member_frontiers == 0


def test_projective_service_bounded_rolling_original_updates_and_retimestamp_refusal():
    owner, selected, candidate, rows, context, delivery = projective_service_fixture()
    first, = owner.spatial_memberships(at_interval=context)
    for start in range(2, 26):
        _, candidate, rows, context, delivery = projective_fixture(start=start)
        for row in rows:
            owner.observe_spatial(row, SpatialLifecycleReceipt("rolling:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
        result = owner.join_projective_continuity(candidate, at_interval=context, delivery_context=delivery)
        assert result.disposition == "accepted", result
        dto, = owner.spatial_memberships(at_interval=context)
        assert dto.physical_key == first.physical_key and all(e.eligible for e in dto.eligibility)
        original = spatial_reference(rows[start % 2], context)
        intent = owner.prepare_spatial_member(dto, original, at_interval=context).reservation
        assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda d, r: r).consumer_value is original
        assert len(owner.ledger._projective_frontiers) == 2
        assert owner.ledger.stats().receipts == owner.ledger.stats().retired_keys == 0
    pair = candidate.pairs[0]
    changed = replace(pair.left[-1], pixel_uncertainty_px=1.1)
    forged = replace(candidate, pairs=(replace(pair, left=pair.left[:-1] + (changed,)),))
    result = owner.join_projective_continuity(forged, at_interval=context, delivery_context=replace(delivery, sequence=26))
    assert result.reason == "projective_original_retimestamp"


def projective_stationary_inputs(start):
    """Same originals through seed frame3; actual new stopped originals thereafter."""
    selected, candidate, rows, context, delivery = projective_fixture(start=start)
    pair = candidate.pairs[0]
    def stopped(side):
        return tuple(replace(s, bbox_xywh=(55., 80., 10., 25.), original_point=(.6, 1., 1.),
                             observed_anchor_uv=(60., 100.)) if s.stamp.frame_id >= 3 else s for s in side)
    return selected, replace(candidate, pairs=(replace(pair, left=stopped(pair.left), right=stopped(pair.right)),)), rows, context, delivery


def test_projective_stationary_seed_renews_fresh_current_grants_for_more_than_350ms_but_new_stationary_pair_does_not():
    owner, selected, _, _, _, _ = projective_service_fixture()
    key = next(iter(owner.ledger._persons))
    for start in range(2, 23):
        _, candidate, rows, context, delivery = projective_stationary_inputs(start)
        for row in rows:
            owner.observe_spatial(row, SpatialLifecycleReceipt("stopped:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
        result = owner.join_projective_continuity(candidate, at_interval=context, delivery_context=delivery)
        assert result.disposition == "accepted", result
        dto, = owner.spatial_memberships(at_interval=context)
        assert dto.physical_key == key and all(e.eligible for e in dto.eligibility)
        original = spatial_reference(rows[start % 2], context)
        intent = owner.prepare_spatial_member(dto, original, at_interval=context).reservation
        assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda d, r: r).consumer_value is original
        assert len(owner.ledger._projective_frontiers) == 2 and owner.ledger.stats().receipts == 0
    assert rows[0].stamp.acquired_at_s - 10.1 > .35
    fresh = IdentityAssociationService.for_projective_continuity("run-a", spatial_policy(), selected)
    for row in rows:
        fresh.observe_spatial(row, SpatialLifecycleReceipt("new-stationary", row.stamp, "ACTIVE"))
    assert fresh.join_projective_continuity(candidate, at_interval=context, delivery_context=delivery).disposition == "association_unresolved"
    assert fresh.ledger.stats().active_persons == 2


@pytest.mark.parametrize("barrier", ["missing-body", "missing-radius", "missing-appearance", "nonactive", "absence", "skipped", "epoch", "run", "contradiction"])
def test_projective_stationary_renewal_privilege_does_not_survive_real_barriers_or_unavailable_cues(barrier):
    owner, selected, candidate, rows, context, delivery = projective_service_fixture()
    dto, = owner.spatial_memberships(at_interval=context)
    old = owner.prepare_spatial_member(dto, spatial_reference(rows[0], context), at_interval=context).reservation
    if barrier == "nonactive":
        owner.observe_spatial(rows[0], SpatialLifecycleReceipt("real-inactive", rows[0].stamp, "INACTIVE"))
    elif barrier == "absence":
        owner.withdraw(removal(rows[0], frame=4, time=10.15, media=5.15))
    elif barrier == "epoch":
        owner.reset_source(replace(SOURCES[1], source_epoch=1))
    elif barrier == "run":
        owner.reset_run("run-b")
    start = 3 if barrier == "skipped" else 2
    _, candidate, rows, context, delivery = projective_stationary_inputs(start)
    for row in rows:
        owner.observe_spatial(row, SpatialLifecycleReceipt("after-barrier:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
    if barrier == "missing-body":
        candidate = replace(candidate, pairs=())
    elif barrier == "missing-radius":
        pair = candidate.pairs[0]
        candidate = replace(candidate, pairs=(replace(pair, left=pair.left[:-1] + (replace(pair.left[-1], pixel_uncertainty_px=None),)),))
    elif barrier == "missing-appearance":
        candidate = replace(candidate, visual=replace(candidate.visual, candidates=()))
    elif barrier == "contradiction":
        pair = candidate.pairs[0]
        side = pair.left[:-1] + (replace(pair.left[-1], original_point=(100., 1., 1.)),)
        candidate = replace(candidate, pairs=(replace(pair, left=side),))
    result = owner.join_projective_continuity(candidate, at_interval=context, delivery_context=delivery)
    assert result.disposition != "accepted"
    assert owner.commit_spatial_member(old, at_interval=context, consumer=lambda *_: pytest.fail("old queued intent")).disposition == "denied"
    if barrier in {"missing-body", "missing-radius", "missing-appearance"}:
        current, = owner.spatial_memberships(at_interval=context)
        assert {e.reason for e in current.eligibility} == {"projective_current_support_required"}
        assert not any(m.spatial_join_required for m in owner.ledger._members.values())
        # Fresh stationary support alone cannot reacquire a revoked seed.
        _, later, rows, context, delivery = projective_stationary_inputs(3)
        for row in rows:
            owner.observe_spatial(row, SpatialLifecycleReceipt("fresh-after-missing", row.stamp, "ACTIVE"))
        assert owner.join_projective_continuity(later, at_interval=context, delivery_context=delivery).disposition != "accepted"
    if barrier == "contradiction":
        assert owner.ledger.stats().active_persons == 2 and max(k.serial for k in owner.ledger._persons) == 3


@pytest.mark.parametrize("removed_index", [0, 1])
@pytest.mark.parametrize("kind", ["absence", "termination"])
def test_projective_single_source_genuine_peer_withdrawal_preserves_key_and_current_local_grant(removed_index, kind):
    owner, selected, candidate, rows, context, delivery = projective_service_fixture()
    dto, = owner.spatial_memberships(at_interval=context)
    key = dto.physical_key
    old = owner.prepare_spatial_member(dto, spatial_reference(rows[1 - removed_index], context), at_interval=context).reservation
    removed = rows[removed_index]
    result = owner.withdraw(removal(removed, kind=kind, frame=4, time=10.15, media=5.15))
    assert result.withdrawn_members == (removed.stamp.member,)
    _, _, new_rows, _, _ = projective_fixture(start=2)
    survivor = new_rows[1 - removed_index]
    owner.observe_spatial(survivor, SpatialLifecycleReceipt("continuous-survivor", survivor.stamp, "ACTIVE"))
    context = spatial_context((survivor,))
    # No cross-camera body/motion proof, no fabricated absent peer input.
    local = replace(candidate, visual=visual_proof((survivor,), token="one-current-source"), pairs=())
    owner.join_projective_continuity(local, at_interval=context,
        delivery_context=visual_delivery((survivor,), sequence=2, evaluation=100.15))
    current, = owner.spatial_memberships(at_interval=context)
    assert current.physical_key == key and current.association_basis == "source_local"
    assert len(current.members) == 1 and current.members[0].observation == survivor.stamp
    assert current.eligibility[0].eligible and removed.stamp.member not in owner.ledger._members
    original = spatial_reference(survivor, context)
    intent = owner.prepare_spatial_member(current, original, at_interval=context).reservation
    assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda d, r: r).consumer_value is original
    assert owner.commit_spatial_member(old, at_interval=context, consumer=lambda *_: pytest.fail("old shared intent")).disposition == "denied"
    assert owner.prepare_spatial_member(current, replace(original, owner=removed.stamp.member), at_interval=context).disposition == "denied"


def projective_lag_inputs(left_start, right_start, sequence):
    selected, left_proof, left_rows, _, _ = projective_fixture(start=left_start)
    _, right_proof, right_rows, _, _ = projective_fixture(start=right_start)
    pair = replace(left_proof.pairs[0], right=right_proof.pairs[0].right)
    rows = left_rows[0], right_rows[1]
    candidate = replace(left_proof, visual=visual_proof(rows, token=f"lag-{sequence}"), pairs=(pair,))
    mono = max(pair.left[-1].ingress_monotonic_s, pair.right[-1].ingress_monotonic_s)
    delivery = replace(visual_delivery(rows, sequence=sequence, evaluation=mono), ingresses=tuple(
        CurrentVisualIngress(s.stamp, s.ingress_monotonic_s, s.stamp, s.ingress_monotonic_s)
        for s in (pair.left[-1], pair.right[-1])))
    return selected, candidate, rows, spatial_context(rows), delivery


@pytest.mark.parametrize("leading", [0, 1])
def test_projective_first_stop_current_endpoint_support_keeps_seed_but_cannot_create_new_pair(leading):
    owner, selected, _, _, _, _ = projective_service_fixture()
    key = next(iter(owner.ledger._persons))
    _, candidate, rows, context, delivery = projective_lag_inputs(2 if leading == 0 else 1, 2 if leading == 1 else 1, 2)
    pair = candidate.pairs[0]
    side = pair.left if leading == 0 else pair.right
    stopped = replace(side[-1], bbox_xywh=(55., 80., 10., 25.), original_point=(.6, 1., 1.),
                      observed_anchor_uv=(60., 100.))
    pair = replace(pair, **{"left" if leading == 0 else "right": side[:-1] + (stopped,)})
    candidate = replace(candidate, pairs=(pair,))
    row = rows[leading]
    owner.observe_spatial(row, SpatialLifecycleReceipt("first-real-stop", row.stamp, "ACTIVE"))
    result = owner.join_projective_continuity(candidate, at_interval=context, delivery_context=delivery)
    assert result.disposition == "accepted", result
    dto, = owner.spatial_memberships(at_interval=context)
    assert dto.physical_key == key and all(e.eligible for e in dto.eligibility)
    original = spatial_reference(row, context)
    intent = owner.prepare_spatial_member(dto, original, at_interval=context).reservation
    assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda d, r: r).consumer_value is original
    fresh = IdentityAssociationService.for_projective_continuity("run-a", spatial_policy(), selected)
    for current in rows:
        fresh.observe_spatial(current, SpatialLifecycleReceipt("unseeded-same-input", current.stamp, "ACTIVE"))
    assert fresh.join_projective_continuity(candidate, at_interval=context, delivery_context=delivery).disposition == "association_unresolved"
    assert fresh.ledger.stats().active_persons == 2


@pytest.mark.parametrize("first", [0, 1])
def test_projective_lag_seeded_alternating_original_callbacks_keep_one_key_and_current_grants(first):
    owner, _, _, _, _, _ = projective_service_fixture()
    key = next(iter(owner.ledger._persons))
    sequence = 1
    for start in range(2, 10):
        steps = ((start, start - 1, 0), (start, start, 1)) if first == 0 else ((start - 1, start, 1), (start, start, 0))
        for left, right, updated in steps:
            sequence += 1
            _, candidate, rows, context, delivery = projective_lag_inputs(left, right, sequence)
            row = rows[updated]
            owner.observe_spatial(row, SpatialLifecycleReceipt("async:" + row.stamp.observation_id, row.stamp, "ACTIVE"))
            result = owner.join_projective_continuity(candidate, at_interval=context, delivery_context=delivery)
            assert result.disposition == "accepted", result
            dto, = owner.spatial_memberships(at_interval=context)
            assert dto.physical_key == key and all(e.eligible for e in dto.eligibility)
            original = spatial_reference(row, context)
            intent = owner.prepare_spatial_member(dto, original, at_interval=context).reservation
            assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda d, r: r).consumer_value is original
    assert owner.ledger.stats().active_persons == 1 and len(owner.ledger._projective_frontiers) == 2
    assert owner.ledger.stats().receipts == owner.ledger.stats().retired_keys == 0


@pytest.mark.parametrize("defect", ["reverse", "diverge", "expired-lag", "lead-two", "reset", "unseeded"])
def test_projective_lag_cannot_authorize_new_pair_reverse_divergence_expiry_or_reset(defect):
    owner, selected, _, _, _, _ = projective_service_fixture()
    left_start = 3 if defect == "lead-two" else 2
    _, candidate, rows, context, delivery = projective_lag_inputs(left_start, 1, 2)
    if defect == "reset":
        owner.reset_source(replace(SOURCES[1], source_epoch=1))
    elif defect == "unseeded":
        owner = IdentityAssociationService.for_projective_continuity("run-a", spatial_policy(), selected)
        owner.observe_spatial(rows[1], SpatialLifecycleReceipt("never-seeded-F", rows[1].stamp, "ACTIVE"))
    owner.observe_spatial(rows[0], SpatialLifecycleReceipt("leading-current", rows[0].stamp, "ACTIVE"))
    if defect in {"reverse", "diverge", "expired-lag"}:
        pair = candidate.pairs[0]
        sample = pair.left[-1]
        if defect == "expired-lag":
            sample = replace(sample, ingress_monotonic_s=100.6)
            delivery = replace(delivery, evaluation_monotonic_s=100.6,
                ingresses=(replace(delivery.ingresses[0], ingress_monotonic_s=100.6,
                    appearance_ingress_monotonic_s=100.6), delivery.ingresses[1]))
        else:
            u, v = (55., 100.) if defect == "reverse" else (60., 110.)
            sample = replace(sample, bbox_xywh=(u - 5, v - 20, 10., 25.),
                             original_point=(u / 100, v / 100, 1.), observed_anchor_uv=(u, v))
        candidate = replace(candidate, pairs=(replace(pair, left=pair.left[:-1] + (sample,)),))
    result = owner.join_projective_continuity(candidate, at_interval=context, delivery_context=delivery)
    assert result.disposition != "accepted", result
    if defect == "unseeded":
        assert owner.ledger.stats().active_persons == 2


@pytest.mark.parametrize("boundary", ["inactive-peer", "stale-peer", "own-native-gap"])
def test_projective_single_source_does_not_fake_peer_absence_or_clear_own_native_barrier(boundary):
    owner, selected, candidate, rows, context, delivery = projective_service_fixture()
    if boundary == "inactive-peer":
        owner.observe_spatial(rows[1], SpatialLifecycleReceipt("actual-inactive-peer", rows[1].stamp, "INACTIVE"))
    elif boundary == "own-native-gap":
        owner.observe_spatial(rows[0], SpatialLifecycleReceipt("actual-own-inactive", rows[0].stamp, "INACTIVE"))
        owner.withdraw(removal(rows[1], kind="termination", frame=4, time=10.15, media=5.15))
    fresh = observation(frame=4, time=10.6 if boundary == "stale-peer" else 10.15,
                        media=5.6 if boundary == "stale-peer" else 5.15)
    owner.observe_spatial(fresh, SpatialLifecycleReceipt("fresh-current-source", fresh.stamp, "ACTIVE"))
    context = spatial_context((fresh,) if boundary == "own-native-gap" else (fresh, rows[1]), evaluation=fresh.stamp.acquired_at_s)
    visible = (fresh,) if boundary in {"own-native-gap", "stale-peer", "inactive-peer"} else (fresh, rows[1])
    owner.join_projective_continuity(replace(candidate, visual=visual_proof(visible), pairs=()), at_interval=context,
        delivery_context=visual_delivery(visible, sequence=2, evaluation=100.6))
    dtos = owner.spatial_memberships(at_interval=context)
    current = next(d for d in dtos if any(m.observation == fresh.stamp for m in d.members))
    assert len(current.members) == 1
    result = owner.prepare_spatial_member(current, spatial_reference(fresh, context), at_interval=context)
    assert result.disposition == ("reserved" if boundary == "stale-peer" else "denied")
    if boundary == "own-native-gap":
        assert result.reason == "current_rejoin_proof_required"
    else:
        peer = next(e for d in dtos for e in d.eligibility if e.observation.member == rows[1].stamp.member)
        assert not peer.eligible and rows[1].stamp.member in owner.ledger._members


def test_projective_single_source_stalled_delivery_detaches_association_without_fake_native_absence():
    owner, selected, candidate, rows, context, delivery = projective_service_fixture()
    original, = owner.spatial_memberships(at_interval=context)
    old = owner.prepare_spatial_member(original, spatial_reference(rows[0], context), at_interval=context).reservation
    peer = rows[1]
    fresh = observation(frame=4, time=10.6, media=5.6)
    owner.observe_spatial(fresh, SpatialLifecycleReceipt("genuinely-live-K", fresh.stamp, "ACTIVE"))
    # F acquisition/evaluation remains frozen: ONLY original real ingress exposes stalling.
    context = spatial_context((fresh, peer))
    local = replace(candidate, visual=visual_proof((fresh,)), pairs=())
    result = owner.join_projective_continuity(local, at_interval=context,
        delivery_context=visual_delivery((fresh,), sequence=2, evaluation=100.6))
    current = owner.spatial_memberships(at_interval=context)
    live = next(d for d in current if d.physical_key == original.physical_key)
    stale = next(d for d in current if d.physical_key != original.physical_key)
    assert live.association_basis == "source_local" and len(live.members) == 1 and live.eligibility[0].eligible
    assert stale.members[0].observation == peer.stamp and not stale.eligibility[0].eligible
    assert stale.eligibility[0].reason == "current_visual_stale_member_delivery"
    assert owner.ledger._members[peer.stamp.member].spatial_lifecycle.native_state == "ACTIVE"
    assert peer.stamp.member not in owner.ledger._closed and result.withdrawn_members == (peer.stamp.member,)
    assert max(d.physical_key.serial for d in current) == 3
    reference = spatial_reference(fresh, context)
    intent = owner.prepare_spatial_member(live, reference, at_interval=context).reservation
    assert owner.commit_spatial_member(intent, at_interval=context, consumer=lambda d, r: r).consumer_value is reference
    assert owner.commit_spatial_member(old, at_interval=context, consumer=lambda *_: pytest.fail("old shared intent")).disposition == "denied"
@pytest.mark.parametrize('bad',[None,0,1,'true',[],{}])
def test_unresolved_birth_service_requires_exact_bool(bad):
    from tests.test_mv3dt_person_association import semantic_joint_fixture
    selected,*_=semantic_joint_fixture()
    with pytest.raises(ValueError,match='birth_opt_in'):
        IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,
            live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=bad)


def test_unresolved_birth_service_pending_denies_vote_then_true_pair_recovers():
    from tests.test_mv3dt_person_association import semantic_joint_fixture
    selected,p,rows,c,d=semantic_joint_fixture(competitor=None)
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(replace(p,originals=p.originals[1:]),at_interval=c,delivery_context=d)
    for dto in owner.spatial_memberships(at_interval=c):
        r=next(r for r in rows if r.stamp==dto.members[0].observation)
        result=owner.prepare_spatial_member(dto,spatial_reference(r,c),at_interval=c)
        assert result.disposition=='denied' and result.reason=='image_region_ambiguous_member'
    selected,p,rows,c,d=semantic_joint_fixture(2,competitor=None)
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(p,at_interval=c,delivery_context=d)
    dto,=owner.spatial_memberships(at_interval=c)
    assert all(e.eligible for e in dto.eligibility)
    intent=owner.prepare_spatial_member(dto,spatial_reference(rows[0],c),at_interval=c).reservation
    assert intent is not None
    candidate=replace(rows[1],stamp=replace(rows[1].stamp,member=replace(rows[1].stamp.member,raw_id=99)))
    assert owner.observe_spatial(candidate,SpatialLifecycleReceipt('new-competitor',candidate.stamp,'ACTIVE')).disposition=='observed'
    assert owner.commit_spatial_member(intent,at_interval=c,consumer=lambda *_:pytest.fail('stale grant')).disposition=='denied'


def test_split_birth_public_false_split_only_original_key_votes_and_rejoin_recovers():
    from tests.test_mv3dt_person_association import semantic_joint_fixture,split_birth_fixture
    selected,*_=semantic_joint_fixture(competitor=None)
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    initial=None;queued=None
    for frame in (1,2,3,4):
        selected,p,rows,c,d=(semantic_joint_fixture(frame,competitor=None) if frame in (1,4) else split_birth_fixture(frame))
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        owner.join_estimated_image_regions(p,at_interval=c,delivery_context=d)
        dtos=owner.spatial_memberships(at_interval=c)
        if frame==1:
            initial=dtos[0].physical_key
            queued=owner.prepare_spatial_member(dtos[0],spatial_reference(rows[1],c),at_interval=c).reservation
            assert queued is not None
        elif frame in (2,3):
            assert len(dtos)==2
            k=next(x for x in dtos if x.physical_key==initial)
            f=next(x for x in dtos if x.physical_key!=initial)
            assert owner.prepare_spatial_member(f,spatial_reference(rows[1],c),at_interval=c).reason=='image_region_ambiguous_member'
            intent=owner.prepare_spatial_member(k,spatial_reference(rows[0],c),at_interval=c).reservation
            assert intent is not None
            assert owner.commit_spatial_member(intent,at_interval=c,consumer=lambda *_:'genuine K').disposition=='committed'
            assert owner.commit_spatial_member(queued,at_interval=c,consumer=lambda *_:pytest.fail('old F')).disposition=='denied'
        else:
            assert len(dtos)==1 and dtos[0].physical_key==initial and all(x.eligible for x in dtos[0].eligibility)


def estimated_split_custody_service_fixture():
    from tests.test_mv3dt_person_association import semantic_joint_fixture,split_birth_fixture
    selected,*_=semantic_joint_fixture(competitor=None)
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    for frame in (1,2):
        selected,p,rows,c,d=semantic_joint_fixture(frame,competitor=None) if frame==1 else split_birth_fixture(frame)
        for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('native:'+r.stamp.observation_id,r.stamp,'ACTIVE'))
        result=owner.join_estimated_image_regions(p,at_interval=c,delivery_context=d)
        if frame==1:
            dto,=owner.spatial_memberships(at_interval=c)
            queued=owner.prepare_spatial_member(dto,spatial_reference(rows[1],c),at_interval=c).reservation
    return owner,p,rows,c,d,result,queued


def test_estimated_split_custody_public_both_dtos_exact_views_no_old_measurement_grant():
    from noesis.identity_association_service import EstimatedSplitCustodyView
    owner,p,rows,c,d,result,queued=estimated_split_custody_service_fixture()
    dtos=owner.spatial_memberships(at_interval=c)
    assert len(dtos)==2 and all(len(dto.members)==1 for dto in dtos)
    view=dtos[0].estimated_split_custody
    assert type(view) is EstimatedSplitCustodyView and dtos[1].estimated_split_custody is view
    assert set(view.successor_keys)=={dto.physical_key for dto in dtos}
    assert rows[1].stamp.member in result.withdrawn_members  # Ordinary barrier intact.
    detached=next(dto for dto in dtos if dto.physical_key!=view.origin_key)
    assert not detached.eligibility[0].eligible
    assert owner.prepare_spatial_member(detached,spatial_reference(rows[1],c),at_interval=c).reason=='image_region_ambiguous_member'
    assert owner.commit_spatial_member(queued,at_interval=c,consumer=lambda *_:pytest.fail('queued old grant')).disposition=='denied'
    assert all(owner._spatial_issued[dto.physical_key] is dto for dto in dtos)
    # Copying an owner DTO/view cannot become a capability.
    forged=replace(dtos[0],estimated_split_custody=replace(view,episode_id='foreign'))
    assert owner.prepare_spatial_member(forged,spatial_reference(rows[0],c),at_interval=c).disposition=='denied'


@pytest.mark.parametrize('defect',['foreign_view','foreign_member','new_birth','future','clock','stop'])
def test_i121_public_native_destination_is_bound_readonly_not_point_authority(defect):
    owner,p,rows,c,d,result,queued=estimated_split_custody_service_fixture()
    dtos=owner.spatial_memberships(at_interval=c)
    view=dtos[0].estimated_split_custody
    original=view.prior_seed[1]
    destination=owner.estimated_split_original_destination(view,original,at_interval=c)
    assert destination==next(dto.physical_key for dto in dtos if any(m.observation.member==original.member for m in dto.members))
    assert all(owner._spatial_issued[dto.physical_key] is dto for dto in dtos)
    bad_view=view
    if defect=='foreign_view':bad_view=replace(view,episode_id='not-issued')
    elif defect=='foreign_member':original=replace(original,member=replace(original.member,raw_id=999))
    elif defect=='new_birth':original=replace(original,lifecycle_started_frame=original.lifecycle_started_frame+1)
    elif defect=='future':original=replace(original,frame_id=1000,acquired_at_s=1000.,media_time_s=1000.)
    elif defect=='clock':original=replace(original,clock_domain='foreign')
    else:owner.stop_image_region_source(next(b for b in owner._ledger.source_bindings()
        if b.source_id==original.member.source_id),graph_id=p.owner_availability[0].graph_id)
    with pytest.raises(ValueError):owner.estimated_split_original_destination(bad_view,original,at_interval=c)


def test_estimated_split_custody_public_native_shadow_does_not_become_current_vote():
    from tests.test_mv3dt_person_association import split_birth_fixture,image_owner_capsule,spatial_shadow_update
    owner,p,rows,c,d,result,_=estimated_split_custody_service_fixture()
    old=rows[0];view=owner.spatial_memberships(at_interval=c)[0].estimated_split_custody
    _,p,rows,c,d=split_birth_fixture(3)
    update=spatial_shadow_update(old,frame=3,time=rows[0].stamp.acquired_at_s,media=rows[0].stamp.media_time_s)
    owner.update_spatial_lifecycle(update)
    owner.observe_spatial(rows[1],SpatialLifecycleReceipt('fresh-peer',rows[1].stamp,'ACTIVE'))
    cap=replace(image_owner_capsule(old,100.05),source_frame_id=3,source_media_time_s=rows[0].stamp.media_time_s,
        source_observed_at_us=1000003,availability_checked_monotonic_s=d.evaluation_monotonic_s,
        owner_ingress_monotonic_s=d.evaluation_monotonic_s,shadow_members=(old.stamp.member,))
    p=replace(p,cohort=(rows[1].stamp,),originals=p.originals[1:],owner_availability=(cap,p.owner_availability[1]))
    owner.join_estimated_image_regions(p,at_interval=c,delivery_context=replace(d,ingresses=d.ingresses[1:]))
    dtos=owner.spatial_memberships(at_interval=c)
    assert all(dto.estimated_split_custody is view for dto in dtos)
    assert all(not e.eligible for dto in dtos for e in dto.eligibility)


def test_estimated_split_custody_public_selected_other_pair_revokes_queued_authority():
    from tests.test_mv3dt_person_association import semantic_joint_fixture,split_birth_fixture
    owner,p,rows,c,d,result,_=estimated_split_custody_service_fixture()
    dtos=owner.spatial_memberships(at_interval=c)
    view=dtos[0].estimated_split_custody
    origin=next(dto for dto in dtos if dto.physical_key==view.origin_key)
    intent=owner.prepare_spatial_member(origin,spatial_reference(rows[0],c),at_interval=c).reservation
    assert intent is not None
    selected,p,rows,c,d=semantic_joint_fixture(3,competitor='wrong')
    false=split_birth_fixture(3)[1].originals[1]
    true=replace(p.originals[2],patches=p.originals[1].patches,semantic_joints=p.originals[1].semantic_joints)
    p=replace(p,originals=(p.originals[0],false,true))
    for r in rows:owner.observe_spatial(r,SpatialLifecycleReceipt('qualified-alternative',r.stamp,'ACTIVE'))
    owner.join_estimated_image_regions(p,at_interval=c,delivery_context=d)
    current=owner.spatial_memberships(at_interval=c)
    assert all(dto.estimated_split_custody is None for dto in current)
    assert owner.commit_spatial_member(intent,at_interval=c,consumer=lambda *_:pytest.fail('old scope')).disposition=='denied'
    assert owner.ledger._members[rows[0].stamp.member].person_key==owner.ledger._members[rows[2].stamp.member].person_key


@pytest.mark.parametrize('paired',[False,True])
def test_position_only_queued_current_image_member_public_grant(paired):
    from tests.test_mv3dt_person_association import semantic_joint_fixture,queued_image_position_fixture
    selected,*_=semantic_joint_fixture(competitor=None)
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    _,p,rows,c,d=queued_image_position_fixture(owner,paired=paired)
    assert all(owner.ledger.current_visual_liveness_error(r.stamp)=='current_visual_stale_member_delivery' for r in rows)
    dto,=owner.spatial_memberships(at_interval=c)
    assert all(e.eligible for e in dto.eligibility)
    reference=spatial_reference(rows[0],c)
    result=owner.prepare_spatial_member(dto,reference,at_interval=c)
    assert result.reservation is not None
    assert owner.commit_spatial_member(result.reservation,at_interval=c,consumer=lambda *_:'original current position').disposition=='committed'
    assert owner.prepare_spatial_member(dto,reference,at_interval=c).reason=='spatial_member_measurement_consumed'


@pytest.mark.parametrize('barrier',['stop','reset','new_observation','malformed_ingress'])
def test_position_only_queued_reservation_rechecks_actual_owner_barriers(barrier):
    from tests.test_mv3dt_person_association import semantic_joint_fixture,queued_image_position_fixture
    selected,*_=semantic_joint_fixture(competitor=None)
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,
        live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    _,p,rows,c,d=queued_image_position_fixture(owner,paired=False)
    dto,=owner.spatial_memberships(at_interval=c)
    intent=owner.prepare_spatial_member(dto,spatial_reference(rows[0],c),at_interval=c).reservation
    assert intent is not None
    if barrier=='stop':owner.stop_image_region_source(SOURCES[2],graph_id='graph-a')
    elif barrier=='reset':owner.reset_source(replace(SOURCES[2],source_epoch=1))
    else:
        _,proof,new,context,delivery=semantic_joint_fixture(3,competitor=None)
        r=new[1];owner.observe_spatial(r,SpatialLifecycleReceipt('next-current',r.stamp,'ACTIVE'))
        if barrier=='malformed_ingress':
            proof=replace(proof,cohort=(r.stamp,),originals=proof.originals[1:],owner_availability=proof.owner_availability[1:])
            delivery=replace(delivery,ingresses=(replace(delivery.ingresses[1],ingress_monotonic_s=float('nan')),))
            assert owner.join_estimated_image_regions(proof,at_interval=spatial_context((r,)),delivery_context=delivery).disposition!='accepted'
            fresh,=owner.spatial_memberships(at_interval=spatial_context((r,)))
            assert not fresh.eligibility[0].eligible
    assert owner.commit_spatial_member(intent,at_interval=c,consumer=lambda *_:pytest.fail('stale queued authority')).disposition=='denied'
