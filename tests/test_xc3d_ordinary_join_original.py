"""Ordinary current owner joins must not invent or renew old World originals.

Synthetic native/source geometry; actual public owner, World and publisher.
"""
from dataclasses import replace

import pytest

from tests.test_xc3d_world_physical_person import (
    _r101_actor, actual_node_wire, HouseholdDescriptor, WorldPublicationRefused, COV,
)


def _ordinary_join(monkeypatch, *, advance=False, replacement=False, households=None,
                   competing=False, retire_only=False, incoming_point=False, publication_clock=False):
    actor, harness, clock = _r101_actor(monkeypatch)
    if publication_clock:
        actor.publication_us = actor.us
        actor.world = type(actor.world)(producer=actor.world.producer, artifacts=actor.artifacts,
            clock_us=lambda: actor.publication_us, association_service=actor.identity,
            spatial_bindings=lambda: actor.frames)
    separated = {**actor.joints, (2, 101): tuple((i, u, v+60., c)
        for i, u, v, c in actor.joints[(2, 101)])}
    missing = {'world': None, 'world_valid': False,
               'world_measurement_accepted': False, 'world_covariance': None}
    first = harness.publish(actor.prepare_image(actor.rows, frame=2, households=households,
        semantic_joints=separated, image_y={(2, 101): 90.},
        source_changes={} if competing else {1: missing}), 2)
    old = next(e for e in first.entities
               if next(s.camera_id for s in e.sources if s.accepted) == 'family-room')
    before = actor.memberships
    assert len(before) == 2
    origin = next(d for d in before if d.physical_key.serial == int(old.entity_id.rsplit(':', 1)[1]))
    destination = next(d for d in before if d is not origin)
    assert destination.physical_key < origin.physical_key
    assert all(d.estimated_split_custody is None for d in before)
    assert next(s.camera_id for s in old.sources if s.accepted) == 'family-room'
    actor.us += 100_000
    if publication_clock:
        actor.publication_us = actor.us
    joined_joints = {**separated, (1, 1): tuple((i, u, v+60., c)
        for i, u, v, c in actor.joints[(1, 1)])}
    candidate = actor.prepare_image(actor.rows if advance else {1: actor.rows[1]}, frame=3,
        households=households, semantic_joints=separated if retire_only else joined_joints,
        image_y={(1, 1): 30. if retire_only else 90., (2, 101): 90.},
        retired_keys=(origin.physical_key,) if retire_only else (),
        source_changes={1: {'world_covariance': tuple(v*4 for v in COV)} if incoming_point else missing,
                        **({2: missing} if advance and not replacement else {})})
    if retire_only:
        return actor, harness, old, origin, None, candidate
    joined, = actor.memberships
    assert joined.physical_key == destination.physical_key
    assert origin.physical_key in joined.retired_keys
    assert joined.association_basis == 'estimated_semantic_joint_correspondence'
    assert joined.estimated_split_custody is None
    assert all(e.eligible and e.lifecycle_receipt.native_state == 'ACTIVE'
               for e in joined.eligibility)
    if not advance:
        assert next(m.observation for m in joined.members
                    if m.observation.member.source_id == 2) == origin.members[0].observation
    return actor, harness, old, origin, joined, candidate


def test_current_ordinary_join_preserves_exact_unexpired_original(monkeypatch):
    actor, harness, old, origin, joined, candidate = _ordinary_join(monkeypatch)
    final = harness.publish(candidate, 3)
    actual = actual_node_wire(actor, harness.wire, 'ordinary-join-original')
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert actor.identity.stats().pending_reservations == 0
    assert len(final.entities) == 1, 'ordinary join retired the only measured original'
    kept, = final.entities
    assert kept.entity_id == actor.world._fusion.physical_entity_id(joined.physical_key)
    assert kept.position == old.position
    assert kept.covariance == old.covariance
    assert kept.observed_at_us == old.observed_at_us
    assert kept.stale_after_us == old.stale_after_us
    assert [s.observation_id for s in kept.sources if s.accepted] == [
        s.observation_id for s in old.sources if s.accepted]
    assert kept.position_state == 'held'
    assert [len(s['people']) for s in actual['snapshots']] == [1, 1]


def test_reindexed_original_followup_and_original_expiry(monkeypatch):
    actor, harness, old, origin, joined, candidate = _ordinary_join(monkeypatch)
    first = harness.publish(candidate, 3)
    actor.us += 50_000
    follow = harness.publish(actor.prepare_image({1: []}, frame=4), 4)
    assert follow.entities[0].observed_at_us == old.observed_at_us
    assert follow.entities[0].stale_after_us == old.stale_after_us
    assert follow.entities[0].position == old.position
    assert follow.entities[0].covariance == old.covariance
    actor.us = old.stale_after_us + 1
    expired = harness.publish(actor.prepare_image({1: []}, frame=5), 5)
    assert not expired.entities
    assert actor.identity.stats().pending_reservations == 0


@pytest.mark.parametrize('replacement', [False, True])
def test_forward_stamp_cannot_reindex_historical_original(monkeypatch, replacement):
    actor, harness, old, origin, joined, candidate = _ordinary_join(monkeypatch,
        advance=True, replacement=replacement)
    final = harness.publish(candidate, 3)
    if replacement:
        entity, = final.entities
        assert entity.observed_at_us > old.observed_at_us
        assert next(s.observation_id for s in entity.sources if s.accepted) != next(
            s.observation_id for s in old.sources if s.accepted)
    else:
        assert not final.entities


def test_retirement_alone_is_not_an_observation_mapping(monkeypatch):
    actor, harness, old, origin, joined, candidate = _ordinary_join(monkeypatch, retire_only=True)
    assert not harness.publish(candidate, 3).entities


def test_competing_committed_original_is_not_arbitrarily_replaced(monkeypatch):
    actor, harness, old, origin, joined, candidate = _ordinary_join(monkeypatch, competing=True)
    final = harness.publish(candidate, 3)
    entity, = final.entities
    assert next(s.camera_id for s in entity.sources if s.accepted) == 'kitchen'
    assert entity.position != old.position


def test_worse_current_alternative_does_not_invalidate_reindexed_hold(monkeypatch):
    actor, harness, old, origin, joined, candidate = _ordinary_join(monkeypatch, incoming_point=True)
    entity, = harness.publish(candidate, 3).entities
    assert entity.position == old.position and entity.observed_at_us == old.observed_at_us
    assert entity.stale_after_us == old.stale_after_us and entity.position_state == 'held'
    assert actor.identity.stats().pending_reservations == 0


def test_expired_selected_destination_is_pruned_not_a_fatal_reindex(monkeypatch):
    actor, harness, clock = _r101_actor(monkeypatch)
    # Source currentness and the absolute wire deadline are independent bounds.
    # Use the service's public injected publication clock, not an owner-state bit.
    actor.us -= 100_000
    actor.publication_us = actor.us
    actor.world = type(actor.world)(producer=actor.world.producer, artifacts=actor.artifacts,
        clock_us=lambda: actor.publication_us, association_service=actor.identity,
        spatial_bindings=lambda: actor.frames)
    separated = {**actor.joints, (2, 101): tuple((i, u, v+60., c)
        for i, u, v, c in actor.joints[(2, 101)])}
    first = harness.publish(actor.prepare_image({1: actor.rows[1]}, frame=1,
        semantic_joints=actor.joints), 1)
    kitchen, = first.entities
    missing = {'world': None, 'world_valid': False,
               'world_measurement_accepted': False, 'world_covariance': None}
    actor.us += 100_000
    actor.publication_us = actor.us
    advanced = harness.publish(actor.prepare_image(actor.rows, frame=2,
        semantic_joints=separated, image_y={(2, 101): 90.}, source_changes={1: missing}), 2)
    family = next(e for e in advanced.entities if any(s.accepted and s.camera_id == 'family-room' for s in e.sources))
    assert len(advanced.entities) == 2
    actor.us += 100_000
    actor.publication_us = kitchen.stale_after_us + 1
    assert kitchen.stale_after_us <= actor.publication_us < family.stale_after_us
    joined_joints = {**separated, (1, 1): tuple((i, u, v+60., c)
        for i, u, v, c in actor.joints[(1, 1)])}
    candidate = actor.prepare_image({1: actor.rows[1]}, frame=3,
        semantic_joints=joined_joints, image_y={(1, 1): 90.}, source_changes={1: missing})
    dto, = actor.memberships
    assert all(e.eligible and e.lifecycle_receipt.native_state == 'ACTIVE' for e in dto.eligibility)
    # Positive expiry prunes only the old Kitchen slot; the still-current
    # immutable Family original is not a competing expired point.
    final = harness.publish(candidate, 3)
    kept, = final.entities
    assert kept.position == family.position and kept.covariance == family.covariance
    assert kept.observed_at_us == family.observed_at_us
    assert kept.stale_after_us == family.stale_after_us and kept.position_state == 'held'
    assert next(s.observation_id for s in kept.sources if s.accepted) == next(
        s.observation_id for s in family.sources if s.accepted)
    assert actor.identity.stats().pending_reservations == 0
    actual = actual_node_wire(actor, harness.wire, 'ordinary-join-expired-destination')
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert [len(s['people']) for s in actual['snapshots']] == [1, 2, 1]


@pytest.mark.parametrize('barrier', ['revision', 'stop', 'epoch', 'shadow', 'expiry'])
def test_observation_reindex_revalidates_actual_owner_and_original(monkeypatch, barrier):
    from reid.mv3dt_person_association import SpatialLifecycleReceipt, SpatialLifecycleUpdate
    actor, harness, old, origin, joined, candidate = _ordinary_join(monkeypatch)
    previous = (actor.world.current_snapshot(), actor.world._revision, len(harness.wire))
    stamp = origin.members[0].observation
    if barrier == 'stop':
        actor.identity.stop_image_region_source(actor.sources[1], graph_id='r101-graph')
    elif barrier == 'epoch':
        actor.identity.reset_source(replace(actor.sources[1], source_epoch=1))
    elif barrier == 'shadow':
        actor.identity.update_spatial_lifecycle(SpatialLifecycleUpdate('shadow-reindex', stamp,
            3, actor.us/1e6, stamp.clock_domain, .3))
    elif barrier == 'expiry':
        actor.us = old.stale_after_us+1
    else:
        original = actor.originals[stamp.member]
        later = replace(stamp, frame_id=4, observation_id='new-current-original',
            acquired_at_s=stamp.acquired_at_s+.1, media_time_s=stamp.media_time_s+.1)
        actor.identity.observe_spatial(replace(original, stamp=later),
            SpatialLifecycleReceipt('later-current', later, 'ACTIVE'))
    with pytest.raises(WorldPublicationRefused):
        harness.publish(candidate, 3)
    assert previous == (actor.world.current_snapshot(), actor.world._revision, len(harness.wire))
    assert actor.identity.stats().pending_reservations == 0


def test_exact_observation_reindex_preserves_settled_claim_history(monkeypatch):
    houses = {(1, 1): HouseholdDescriptor('visitor', 'kitchen-person', 1004, 1),
              (2, 101): HouseholdDescriptor('visitor', 'family-person', 1001, 1)}
    actor, harness, old, origin, joined, candidate = _ordinary_join(monkeypatch, households=houses)
    entity, = harness.publish(candidate, 3).entities
    assert entity.position == old.position and entity.stale_after_us == old.stale_after_us
    assert entity.attribution_quarantine is not None
    assert entity.subject.stable_id is None
    assert any(c.physical_serial == origin.physical_key.serial and c.source_id == 2
        for c in entity.attribution_quarantine.owners+entity.attribution_quarantine.retained_owners)
    actual = actual_node_wire(actor, harness.wire, 'ordinary-join-settled-history')
    assert all(d['accepted'] for d in actual['diagnostics'])


def test_settled_reindexed_original_followup_is_not_a_retired_key_hold(monkeypatch):
    houses = {(1, 1): HouseholdDescriptor('visitor', 'kitchen-person', 1004, 1),
              (2, 101): HouseholdDescriptor('visitor', 'family-person', 1001, 1)}
    actor, harness, old, origin, joined, candidate = _ordinary_join(monkeypatch, households=houses)
    first = harness.publish(candidate, 3)
    actor.us += 50_000
    second = harness.publish(actor.prepare_image({1: []}, frame=4, households=houses), 4)
    kept, = second.entities
    assert kept.position == first.entities[0].position == old.position
    assert kept.observed_at_us == old.observed_at_us and kept.stale_after_us == old.stale_after_us
    actual = actual_node_wire(actor, harness.wire, 'ordinary-join-settled-followup')
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert [len(s['people']) for s in actual['snapshots']] == [1, 1, 1]


def test_new_observation_reindex_wire_expiry_is_typed_not_terminal(monkeypatch):
    from noesis_core.world_service import WorldPreparedHoldExpired
    actor, harness, old, origin, joined, candidate = _ordinary_join(monkeypatch,
        publication_clock=True)
    before = (actor.world.current_snapshot(), actor.world._revision, len(harness.wire))
    actor.publication_us = old.stale_after_us+1
    with pytest.raises(WorldPreparedHoldExpired) as failure:
        actor.commit_world(candidate)
    assert failure.value.reason == 'ordinary_join_original_changed'
    assert before == (actor.world.current_snapshot(), actor.world._revision, len(harness.wire))
    assert actor.identity.stats().pending_reservations == 0
    final = harness.publish(actor.prepare_image({1: []}, frame=4), 4)
    assert not final.entities


def test_measured_label_custody_wire_expiry_after_reindex_is_typed_not_terminal(monkeypatch):
    from noesis_core.world_service import WorldPreparedHoldExpired
    houses = {(1, 1): HouseholdDescriptor('visitor', 'kitchen-person', 1004, 1),
              (2, 101): HouseholdDescriptor('visitor', 'family-person', 1001, 1)}
    actor, harness, old, origin, joined, candidate = _ordinary_join(monkeypatch,
        households=houses, publication_clock=True)
    harness.publish(candidate, 3)
    actor.us += 100_000
    actor.publication_us = actor.us
    joints = {key: tuple((i, u, v+60., c) for i, u, v, c in values)
              for key, values in actor.joints.items()}
    fresh = harness.publish(actor.prepare_image({2: actor.rows[2]}, frame=4,
        households=houses, semantic_joints=joints, image_y={(2, 101): 90.}), 4)
    current, = fresh.entities
    assert current.observed_at_us > old.observed_at_us
    assert current.position_state == 'measured' and current.attribution_quarantine is not None
    actor.us += 50_000
    actor.publication_us = current.stale_after_us-1
    attenuated = {**houses, (1, 1): HouseholdDescriptor('provisional'),
        (1, 2): HouseholdDescriptor('visitor', 'new-kitchen-person', 1004, 1)}
    missing = {'world': None, 'world_valid': False,
               'world_measurement_accepted': False, 'world_covariance': None}
    joints = {**joints, (1, 2): tuple((i,u,v+60.,c) for i,u,v,c in joints[(1,1)])}
    queued = actor.prepare_image({1: [*actor.rows[1], (2, 4., 2, 5)]}, frame=5, households=attenuated,
        semantic_joints=joints, image_y={(1, 1): 90., (1, 2): 150.},
        source_changes={1: missing})
    admitted, = queued.publication.snapshot.entities
    assert admitted.position_state == 'measured'
    assert admitted.observed_at_us == current.observed_at_us
    assert queued.publication.snapshot.published_at_us == current.stale_after_us-1
    state = actor.world._prepared_candidates[queued._candidate_id][1]
    assert state.spatial_holds and not state.spatial_reindexes
    before = (actor.world.current_snapshot(), actor.world._revision, len(harness.wire), len(harness.journal))
    actor.publication_us = current.stale_after_us+1
    with pytest.raises(WorldPreparedHoldExpired) as failure:
        actor.commit_world(queued)
    assert failure.value.reason == 'committed_spatial_hold_custody_changed'
    assert before == (actor.world.current_snapshot(), actor.world._revision, len(harness.wire), len(harness.journal))
    assert actor.identity.stats().pending_reservations == 0
    recovered = harness.publish(actor.prepare_image({1: []}, frame=6, households=attenuated), 6)
    assert not recovered.entities
    actual = actual_node_wire(actor, harness.wire, 'ordinary-join-measured-custody-expiry')
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert [len(s['people']) for s in actual['snapshots']] == [1, 1, 1, 0]


@pytest.mark.parametrize('changed', ['envelope', 'confidence', 'subject'])
def test_terminal_custody_diagnostic_distinguishes_actual_expiry_predicates(monkeypatch, caplog, changed):
    import json
    from noesis_core.world import GlobalWorldFusion
    from noesis_core.world_service import WorldPreparedHoldExpired
    from tests.test_xc3d_split_original_transfer import _r173_normal_committed_hold_with_public_clock
    actor, harness, old, clock, candidate = _r173_normal_committed_hold_with_public_clock()
    state=actor.world._prepared_candidates[candidate._candidate_id][1]
    original=GlobalWorldFusion.committed_spatial_original
    def malformed(fusion,key,**kwargs):
        value=original(fusion,key,**kwargs)
        if value is not None and fusion is state.fusion:
            # Negative proof inputs only; no forged grant is ever consumed.
            if changed=='envelope':return (value[0].model_copy(),*value[1:])
            if changed=='confidence':return (*value[:2],value[2]/2,value[3])
            return (*value[:3],value[3].model_copy(update={'generation':value[3].generation+1}))
        return value
    monkeypatch.setattr(GlobalWorldFusion,'committed_spatial_original',malformed)
    before=(actor.world.current_snapshot(),actor.world._revision,len(harness.wire),len(harness.journal))
    clock[0]=old.stale_after_us+1
    with pytest.raises(WorldPublicationRefused) as failure:actor.commit_world(candidate)
    assert not isinstance(failure.value,WorldPreparedHoldExpired)
    assert failure.value.reason=='committed_spatial_hold_custody_changed'
    assert before==(actor.world.current_snapshot(),actor.world._revision,len(harness.wire),len(harness.journal))
    assert actor.identity.stats().pending_reservations==0
    logs=[r for r in caplog.records if r.msg=='Committed spatial hold refusal: %s']
    assert len(logs)==1
    encoded=logs[0].args[0]; data=json.loads(encoded)
    assert len(encoded.encode())<=8192 and data['branch']=='custody_none'
    assert data['expected_present'] and data['historical_present'] and data['historical_stamp_equal']
    assert data['stamp_equal'] and data['selected_stamp_equal']
    assert data['envelope_identical'] is (changed!='envelope')
    assert data['confidence_equal'] is (changed!='confidence')
    assert data['subject_equal'] is (changed!='subject')
    assert data['selected_deadline_us']<data['commit_us']
    assert data['candidate_selected_equal'] and data['candidate_state']=='held'
    # The existing reserved fatal slot is one-shot, including direct repeats.
    actor.world._log_spatial_hold_refusal(candidate,state,*state.spatial_holds[0],state.fusion,None,
        commit_us=clock[0])
    assert len([r for r in caplog.records if r.msg=='Committed spatial hold refusal: %s'])==1


def test_terminal_custody_diagnostic_failure_cannot_replace_refusal(monkeypatch):
    import noesis_core.world_service as service
    from noesis_core.world import GlobalWorldFusion
    from tests.test_xc3d_split_original_transfer import _r173_normal_committed_hold_with_public_clock
    actor,harness,old,clock,candidate=_r173_normal_committed_hold_with_public_clock()
    state=actor.world._prepared_candidates[candidate._candidate_id][1]
    original=GlobalWorldFusion.committed_spatial_original
    def missing(fusion,key,**kwargs):
        return None if fusion is state.fusion else original(fusion,key,**kwargs)
    monkeypatch.setattr(GlobalWorldFusion,'committed_spatial_original',missing)
    def broken(*args,**kwargs):raise RuntimeError('unavailable diagnostic sink')
    monkeypatch.setattr(service._LOGGER,'error',broken)
    clock[0]=old.stale_after_us+1
    before=(actor.world.current_snapshot(),actor.world._revision,len(harness.wire),len(harness.journal))
    with pytest.raises(WorldPublicationRefused,match='committed_spatial_hold_custody_changed'):
        actor.commit_world(candidate)
    assert before==(actor.world.current_snapshot(),actor.world._revision,len(harness.wire),len(harness.journal))
    assert actor.identity.stats().pending_reservations==0


@pytest.mark.parametrize('defect', ['life', 'epoch', 'stamp', 'ambiguous',
                                   'missing-native', 'shadow', 'ineligible', 'unacknowledged'])
def test_reindex_predicate_rejects_nonliteral_or_uncertified_dto(monkeypatch, defect):
    actor, harness, old, origin, joined, candidate = _ordinary_join(monkeypatch)
    # Negative predicate inputs only; no altered DTO is issued or granted.
    stamp = origin.members[0].observation
    if defect in {'life', 'epoch', 'stamp'}:
        member = replace(stamp.member, lifecycle_generation=999) if defect == 'life' else (
            replace(stamp.member, source_epoch=1) if defect == 'epoch' else stamp.member)
        changed = replace(stamp, member=member, frame_id=stamp.frame_id+1)
        invalid = replace(joined, members=tuple(replace(m, observation=changed)
            if m.observation == stamp else m for m in joined.members))
    elif defect == 'ambiguous':
        invalid = replace(joined, association_basis='estimated_current_image_regions_unresolved')
    elif defect == 'unacknowledged':
        invalid = replace(joined, retired_keys=())
    else:
        invalid = replace(joined, eligibility=tuple(replace(e,
            eligible=False if defect == 'ineligible' else e.eligible,
            lifecycle_receipt=None if defect == 'missing-native' else replace(
                e.lifecycle_receipt, native_state='SHADOW') if defect == 'shadow' else e.lifecycle_receipt)
            if e.observation == stamp else e for e in joined.eligibility))
    assert not actor.world._ordinary_observation_reindexes((invalid,), {origin.physical_key}, actor.context)
    actor.world.discard(candidate)
    assert actor.identity.stats().pending_reservations == 0
