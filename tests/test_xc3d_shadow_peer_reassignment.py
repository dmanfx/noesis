"""Causal owner-only control; synthetic geometry is not R97 private evidence."""
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pytest


def examples():
    spec=spec_from_file_location('shadow_reassignment_examples',
        Path(__file__).with_name('test_mv3dt_person_association.py'))
    result=module_from_spec(spec);spec.loader.exec_module(result)
    return result


def shadow_peer_reassignment_fixture(*, switch_fault=None, alternate_head=True, reverse=False, weak_overlap=False):
    m=examples();owner=m.PersonAssociationLedger('run-a',m.spatial_policy())
    def observe(rows):
        for row in rows:
            owner.observe_spatial(row,m.SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE'),
                retain_image_membership=True,image_source_local_reactivation=True,quarantine_unresolved_birth=True)
    def join(p,c,d):
        if reverse:
            p=replace(p,cohort=p.cohort[::-1],originals=p.originals[::-1],owner_availability=p.owner_availability[::-1])
            d=replace(d,ingresses=d.ingresses[::-1])
        return owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,
            delivery_context=d,live_track_continuity=True,semantic_joint_max_mean_px=5.,
            unresolved_birth_quarantine=True)
    selected,p,rows,c,d=m.local_occupancy_fixture(1,extra=False)
    observe(rows);assert join(p,c,d).disposition=='accepted'
    kitchen,old_family=rows;origin=owner._members[kitchen.stamp.member].person_key
    seed_original=p.originals[0]
    prior_birth=owner._members[old_family.stamp.member].image_region_birth_seed
    prior_birth_peers=owner._members[old_family.stamp.member].image_region_birth_peers

    # A genuine new Family observation has no current body output. B-native
    # custody preserves the pair but MUST NOT renew its visual confirmation.
    _,p,rows,c,d=m.local_occupancy_fixture(2,extra=False)
    family=rows[1];observe((family,))
    rows=(kitchen,family);c=m.spatial_context(rows)
    d=replace(d,ingresses=(m.CurrentVisualIngress(kitchen.stamp,100.),d.ingresses[1]))
    p=replace(p,cohort=tuple(row.stamp for row in rows),originals=(seed_original,),
        owner_availability=(m.image_owner_capsule(kitchen,100.),p.owner_availability[1]))
    assert join(p,c,d).disposition=='accepted'
    assert owner._members[family.stamp.member].visual_confirmed_frame==1
    assert family.stamp.frame_id==2 and owner._members[family.stamp.member].image_region_birth_established
    old_cap=p.owner_availability[1]

    # The advanced Family original becomes genuine native SHADOW, not an
    # ACTUAL participant. Kitchen advances too; both exceed confirmation1.
    _,p,rows,c,d=m.local_occupancy_fixture(3)
    event=m.spatial_shadow_update(family,frame=3,time=rows[0].stamp.acquired_at_s,
        media=rows[0].stamp.media_time_s)
    assert not owner.update_spatial_lifecycle(event,retain_image_membership=True).corrected_members
    kitchen,new_family=rows[0],rows[2]
    observe((kitchen,new_family))
    keys=(family.stamp.member,new_family.stamp.member)
    shadow=replace(old_cap,source_frame_id=3,source_media_time_s=event.at_media_time_s,
        source_observed_at_us=1000003,retained_members=keys,active_members=keys,
        shadow_members=(family.stamp.member,),availability_checked_monotonic_s=d.evaluation_monotonic_s,
        owner_ingress_monotonic_s=d.evaluation_monotonic_s)
    new_cap=replace(p.owner_availability[2],retained_members=keys,active_members=keys,
        shadow_members=(family.stamp.member,))
    p=replace(p,cohort=(kitchen.stamp,new_family.stamp),originals=(p.originals[0],p.originals[2]),
        owner_availability=(p.owner_availability[0],shadow,new_cap))
    d=replace(d,ingresses=(d.ingresses[0],d.ingresses[2]))
    c=m.spatial_context((kitchen,new_family))
    if switch_fault=='missing_old_owner':p=replace(p,owner_availability=(p.owner_availability[0],new_cap))
    elif switch_fault=='failed_old_owner':p=replace(p,owner_availability=(p.owner_availability[0],replace(shadow,availability_status='failed'),new_cap))
    elif switch_fault=='unretained_old_owner':p=replace(p,owner_availability=(p.owner_availability[0],replace(shadow,
        retained_members=(),active_members=(),shadow_members=()),new_cap))
    elif switch_fault=='new_pair_shadow':p=replace(p,owner_availability=(p.owner_availability[0],shadow,
        replace(new_cap,shadow_members=keys)))
    elif switch_fault=='stale_old_owner_chain':
        mono=100.5
        p=replace(p,originals=tuple(replace(o,ingress_monotonic_s=mono) for o in p.originals),
            owner_availability=tuple(replace(cap,availability_checked_monotonic_s=mono,owner_ingress_monotonic_s=mono,
                source_ingress_monotonic_s=cap.source_ingress_monotonic_s if cap.observation==family.stamp else mono)
                for cap in p.owner_availability))
        d=replace(d,evaluation_monotonic_s=mono,ingresses=tuple(replace(v,ingress_monotonic_s=mono) for v in d.ingresses))
    result=join(p,c,d)
    assert result.disposition=='accepted'
    assert family.stamp.member in result.corrected_members
    missing=owner._members[family.stamp.member]
    assert not missing.image_region_participant and missing.spatial_join_required
    assert missing.observation.stamp==family.stamp  # No new body, ingress or measurement.
    assert missing.person_key!=origin
    assert missing.image_region_split_birth is None  # Not a FALSE pending episode.
    assert (missing.image_region_rejected_birth_peer is None)==(switch_fault is None)
    assert (missing.image_region_rejected_birth_origin is None)==(switch_fault is None)
    assert missing.image_region_birth_established==(switch_fault is None)
    assert missing.image_region_birth_quarantined==(switch_fault is not None)
    if switch_fault is None:
        assert missing.image_region_birth_seed==prior_birth
        assert missing.image_region_birth_peers==prior_birth_peers
    # Classification is retained, but the genuine native SHADOW still cannot
    # supply a current measurement or vote through the public spatial guard.
    assert owner.spatial_member_error(family.stamp,c)=='current_rejoin_proof_required'
    assert owner._members[kitchen.stamp.member].person_key==origin
    assert owner._members[new_family.stamp.member].person_key==origin
    positive=owner.image_region_decision_diagnostics()['transitions'][-1]
    assert any(row['supported'] and row['unique'] and row['semantic_state'] is True
        and row['right']['raw_id']==new_family.stamp.member.raw_id for row in positive['pairs'])

    # Same-birth return, complete current geometry/native caps, and positively
    # separated local bodies. The existing new K/F pair is now LIVE custody.
    _,p,rows,c,d=m.local_occupancy_fixture(4)
    if not alternate_head:
        # The replacement's CURRENT head is unknown; no historical head is
        # substituted. The returning body's own strong anatomy is unchanged.
        p=replace(p,originals=tuple(replace(o,head_joints=()) if n==2 else o
            for n,o in enumerate(p.originals)))
    if weak_overlap:
        old,anchor=p.originals[1:]
        p=replace(p,originals=(p.originals[0],replace(old,head_joints=(),
            semantic_joints=anchor.semantic_joints,patches=anchor.patches),anchor))
    if switch_fault=='stale_old_owner_chain':
        # Continue forward from the actual delayed switching cohort; do not
        # rewind its monotonic ingress merely to satisfy a test.
        p=replace(p,originals=tuple(replace(o,ingress_monotonic_s=100.55) for o in p.originals),
            owner_availability=tuple(replace(cap,availability_checked_monotonic_s=100.55,
                source_ingress_monotonic_s=100.55) for cap in p.owner_availability))
        d=replace(d,evaluation_monotonic_s=100.55,ingresses=tuple(replace(v,ingress_monotonic_s=100.55) for v in d.ingresses))
    observe(rows)
    family_keys=tuple(row.stamp.member for row in rows if row.stamp.member.source_id==2)
    p=replace(p,owner_availability=tuple(replace(cap,retained_members=family_keys,active_members=family_keys)
        if cap.observation.member.source_id==2 else cap for cap in p.owner_availability))
    result=join(p,c,d)
    assert result.disposition=='accepted' and not result.corrected_members
    assert owner._members[kitchen.stamp.member].person_key==origin
    assert owner._members[new_family.stamp.member].person_key==origin
    assert m.PersonAssociationLedger._image_observed_supports_separate(p.originals[1],p.originals[2])==(alternate_head and not weak_overlap)
    assert owner.spatial_member_error(rows[1].stamp,c) is None
    assert owner._members[family.stamp.member].image_region_vote_blocked_stamp is None
    assert not owner._members[family.stamp.member].image_region_vote_pending
    assert len(owner._persons)==2
    return m,owner,origin,rows,c


def test_shadow_absent_during_positive_same_key_reassignment_does_not_latch_old_negative():
    _m,owner,_origin,rows,c=shadow_peer_reassignment_fixture()
    member=owner._members[rows[1].stamp.member]
    # Desired narrow behavior: an actual CURRENT positive reassignment made
    # the old negative fact obsolete. This does NOT merge either local body.
    assert owner.image_region_error(rows[1].stamp,c) is None
    assert member.image_region_rejected_birth_peer is None
    assert member.image_region_birth_established and not member.image_region_birth_quarantined


@pytest.mark.parametrize('reverse',[False,True])
def test_positive_shadow_reassignment_preserves_prior_birth_without_new_anchor_head(reverse):
    _m,owner,origin,rows,c=shadow_peer_reassignment_fixture(alternate_head=False,reverse=reverse)
    returning=owner._members[rows[1].stamp.member]
    assert returning.image_region_birth_established and not returning.image_region_birth_quarantined
    assert returning.image_region_rejected_birth_peer is None and returning.image_region_split_birth is None
    assert returning.person_key!=origin
    assert returning.observation==rows[1] and returning.spatial_lifecycle.stamp==rows[1].stamp
    assert returning.image_region_fragment_provisional is None
    assert owner.image_region_error(rows[1].stamp,c) is None


def test_preserved_birth_does_not_bypass_approved_weak_fragment_return_veto():
    _m,owner,_origin,rows,c=shadow_peer_reassignment_fixture(weak_overlap=True)
    returning=owner._members[rows[1].stamp.member]
    assert returning.image_region_birth_established and not returning.image_region_birth_quarantined
    assert returning.image_region_birth_seed.frame_id==1
    assert returning.image_region_fragment_provisional is not None
    assert owner.spatial_member_error(rows[1].stamp,c) is None
    assert owner.image_region_error(rows[1].stamp,c)=='image_region_ambiguous_member'


def test_without_positive_reassignment_a_genuine_false_birth_fact_stays_closed():
    m=examples();owner=m.PersonAssociationLedger('run-a',m.spatial_policy())
    for frame in (1,2,3):
        selected,p,rows,c,d=(m.semantic_joint_fixture(frame,competitor=None)
            if frame==1 else m.split_birth_fixture(frame))
        for row in rows:owner.observe_spatial(row,m.SpatialLifecycleReceipt('native:'+row.stamp.observation_id,
            row.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
        owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
            live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    member=owner._members[rows[1].stamp.member]
    assert member.image_region_rejected_birth_peer is not None
    assert member.image_region_split_birth is not None
    assert not member.image_region_birth_established
    assert owner.image_region_error(rows[1].stamp,c)=='image_region_ambiguous_member'


@pytest.mark.parametrize('fault',['missing_old_owner','failed_old_owner','unretained_old_owner',
    'new_pair_shadow','stale_old_owner_chain'])
def test_unqualified_switch_never_clears_negative_via_later_live_pair(fault):
    _m,owner,_origin,rows,c=shadow_peer_reassignment_fixture(switch_fault=fault)
    member=owner._members[rows[1].stamp.member]
    assert member.image_region_rejected_birth_peer is not None
    assert member.image_region_split_birth is None
    assert not member.image_region_birth_established and member.image_region_birth_quarantined
    assert owner.image_region_error(rows[1].stamp,c)=='image_region_ambiguous_member'


@pytest.mark.parametrize('barrier',['stopped','stale','default'])
def test_false_fact_remains_closed_without_qualified_event(barrier):
    m=examples();owner=m.PersonAssociationLedger('run-a',m.spatial_policy())
    for frame in (1,2,3):
        selected,p,rows,c,d=(m.semantic_joint_fixture(frame,competitor=None)
            if frame==1 else m.split_birth_fixture(frame))
        for row in rows:owner.observe_spatial(row,m.SpatialLifecycleReceipt('native:'+row.stamp.observation_id,
            row.stamp,'ACTIVE'),quarantine_unresolved_birth=True)
        if frame==3:
            previous=owner._members[rows[1].stamp.member].image_region_rejected_birth_peer
            if barrier=='stopped':owner.stop_image_region_source(m.SOURCES[1],graph_id='graph-a')
            if barrier=='stale':d=replace(d,evaluation_monotonic_s=d.evaluation_monotonic_s+.36)
        owner.join_estimated_image_regions(p,projective_policy=selected,at_interval=c,delivery_context=d,
            live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=barrier!='default' or frame<3)
    member=owner._members[rows[1].stamp.member]
    assert member.image_region_rejected_birth_peer==previous
    assert not member.image_region_birth_established
    assert owner.image_region_error(rows[1].stamp,c) is not None
