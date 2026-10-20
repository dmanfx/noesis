"""Synthetic native custody; real owner, World, publisher and production Node.

No native/GPU authenticity or physical calibration accuracy is claimed.
"""
from dataclasses import replace
import json

import pytest
from noesis_core.contracts.world import WorldAttributionClaim

from tests.test_xc3d_world_physical_person import (
    _r101_actor, actual_node_wire, HouseholdDescriptor, WorldPublicationRefused,
)


def _held_kitchen_with_current_provisional_replacement(monkeypatch):
    """R105-shaped public SHADOW correction; no fabricated claim/certificate."""
    from noesis.identity_association_service import IdentityAssociationService
    from reid.mv3dt_person_association import ImageRegionOwnerAvailability, SpatialLifecycleUpdate
    from tests.test_xc3d_world_physical_person import COV

    actor, harness, clock = _r101_actor(monkeypatch)
    houses = {(1, 1): HouseholdDescriptor('visitor', 'held-kitchen', 1004, 1),
        (2, 101): HouseholdDescriptor('visitor', 'historical-family', 1001, 1),
        (2, 102): HouseholdDescriptor('provisional')}
    first = harness.publish(actor.prepare_image(actor.rows, frame=2, households=houses,
        semantic_joints=actor.joints, source_changes={2: {'world_covariance': tuple(v*4 for v in COV)}}), 2)
    old, = first.entities
    assert old.attribution_quarantine is not None
    assert next(s.camera_id for s in old.sources if s.accepted) == 'kitchen'
    joined = []
    shadow = shadow_original = None

    def join(proof, **kwargs):
        mono = kwargs['delivery_context'].evaluation_monotonic_s
        caps = []
        for original in actor.image_originals.values():
            stamp = original.stamp
            members = tuple(o.stamp.member for o in actor.image_originals.values()
                if o.stamp.member.source_id == stamp.member.source_id)
            if shadow is not None and stamp.member.source_id == 2: members += (shadow.member,)
            ingress = original.ingress_monotonic_s
            caps.append(ImageRegionOwnerAvailability(stamp, 'r101-graph', stamp.frame_id,
                stamp.media_time_s, 1_000_000+stamp.frame_id, ingress, 350_000_000, members, members,
                (shadow.member,) if shadow is not None and stamp.member.source_id == 2 else (), (),
                ingress, 0., 20., 'running'))
        if shadow is not None:
            members = tuple(o.stamp.member for o in proof.originals if o.stamp.member.source_id == 2)+(shadow.member,)
            frame = max(o.stamp.frame_id for o in proof.originals if o.stamp.member.source_id == 2)
            caps.append(ImageRegionOwnerAvailability(shadow, 'r101-graph', frame, frame/10, 1_000_000+frame,
                shadow_original.ingress_monotonic_s, 350_000_000, members, members,
                (shadow.member,), (), mono, 0., 20., 'running', mono))
        result = IdentityAssociationService.join_estimated_image_regions(actor.identity,
            replace(proof, owner_availability=tuple(caps)), **kwargs)
        joined.append(result)
        return result
    monkeypatch.setattr(actor.identity, 'join_estimated_image_regions', join)
    actor.us += 100_000
    advanced = harness.publish(actor.prepare_image({2: actor.rows[2]}, frame=3, households=houses,
        semantic_joints=actor.joints, omit_image_current=True,
        source_changes={2: {'world_covariance': None}}), 3)
    _r175_assert_same_original(old, advanced.entities[0])
    shadow = next(o.stamp for m, o in actor.originals.items() if m.source_id == 2)
    actor.us += 100_000
    update = SpatialLifecycleUpdate('public-family-shadow', shadow, 4, actor.us/1e6,
        shadow.clock_domain, .4)
    result = actor.identity.update_spatial_lifecycle(update)
    assert result.disposition == 'observed' and not result.withdrawn_members
    shadow_original = actor.image_originals.pop(shadow.member)
    joints = {**actor.joints, (2, 102): actor.joints[(2, 101)]}

    def prepare(frame=4, *, mode='physical_only_quarantine'):
        if frame > 4:
            actor.identity.update_spatial_lifecycle(SpatialLifecycleUpdate(
                'public-family-shadow:'+str(frame), shadow, frame, actor.us/1e6,
                shadow.clock_domain, frame/10))
        return actor.prepare_image({1: actor.rows[1], 2: [(102, 4., 2, 4)]}, frame=frame,
            households=houses, semantic_joints=joints, image_y={(2, 102): 30.}, mode=mode,
            source_changes={1: {'world': None, 'world_valid': False,
                    'world_measurement_accepted': False, 'world_covariance': None},
                2: {'world_covariance': tuple(v*4 for v in COV)}})
    return actor, harness, clock, old, shadow, joined, prepare


def test_held_quarantined_original_survives_current_unselected_member_replacement(monkeypatch):
    actor, harness, clock, old, shadow, joined, prepare = _held_kitchen_with_current_provisional_replacement(monkeypatch)
    final = harness.publish(prepare(), 4)
    assert final.entities, (joined[-1].withdrawn_members, joined[-1].retired_keys)
    kept, = final.entities
    _r175_assert_same_original(old, kept)
    assert shadow.member in joined[-1].withdrawn_members
    assert all(m.source_id != 1 for m in joined[-1].withdrawn_members)
    current = next(d for d in actor.memberships if d.physical_key.serial == int(old.entity_id.rsplit(':', 1)[1]))
    assert {(m.observation.member.source_id, m.observation.member.raw_id) for m in current.members} == {(1, 1), (2, 102)}
    assert all(d.household_conflict is None and d.estimated_split_custody is None for d in actor.memberships)
    assert all(not e.eligible for d in actor.memberships for e in d.eligibility
        if e.observation.member == shadow.member)
    q = kept.attribution_quarantine
    assert q is not None and kept.subject.stable_id is None
    assert set(old.attribution_quarantine.owners+old.attribution_quarantine.retained_owners) <= set(q.owners+q.retained_owners)
    assert any(c.raw_id == shadow.member.raw_id and c.physical_serial == int(old.entity_id.rsplit(':',1)[1])
        for c in q.owners+q.retained_owners)
    assert actor.identity.stats().pending_reservations == 0
    actor.us += 50_000
    follow = harness.publish(prepare(5), 5)
    _r175_assert_same_original(old, follow.entities[0])
    history = set(old.attribution_quarantine.owners+old.attribution_quarantine.retained_owners)
    history.update(c for c in q.owners+q.retained_owners if c.raw_id == shadow.member.raw_id)
    assert history <= set(follow.entities[0].attribution_quarantine.owners+
        follow.entities[0].attribution_quarantine.retained_owners)
    actual = actual_node_wire(actor, harness.wire, 'held-quarantined-member-replacement')
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert [len(s['people']) for s in actual['snapshots']] == [1, 1, 1, 1]


@pytest.mark.parametrize('defect', ['missing-q', 'foreign-q', 'missing-original',
    'wrong-envelope', 'foreign-lifecycle', 'foreign-settled-claim', 'strict-mode'])
def test_unselected_member_replacement_never_launders_missing_or_foreign_custody(monkeypatch, defect):
    from noesis_core.world import GlobalWorldFusion
    actor, harness, clock, old, shadow, joined, prepare = _held_kitchen_with_current_provisional_replacement(monkeypatch)
    if defect in {'missing-q', 'foreign-q'}:
        actual = GlobalWorldFusion.attribution_quarantines
        def quarantines(fusion):
            values = actual(fusion)
            if fusion is actor.world._fusion:
                values = dict(values)
                record = values.pop(old.entity_id)
                if defect == 'foreign-q':
                    values[old.entity_id] = record.model_copy(update={'component_entity_ids': ('physical:run-synthetic:999',)})
            return values
        monkeypatch.setattr(GlobalWorldFusion, 'attribution_quarantines', quarantines)
    elif defect in {'missing-original', 'wrong-envelope'}:
        actual = GlobalWorldFusion.committed_spatial_original
        def original(fusion, key, **kwargs):
            value = actual(fusion, key, **kwargs)
            if value is not None and fusion is actor.world._fusion:
                return None if defect == 'missing-original' else (value[0].model_copy(), *value[1:])
            return value
        monkeypatch.setattr(GlobalWorldFusion, 'committed_spatial_original', original)
    elif defect in {'foreign-lifecycle', 'foreign-settled-claim'}:
        actual = GlobalWorldFusion.spatial_original_attribution_claims
        def claims(fusion):
            values = dict(actual(fusion))
            if old.entity_id in values:
                changes = {'lifecycle_generation': 999} if defect == 'foreign-lifecycle' else {
                    'household_identity_key': 'foreign-settled', 'household_stable_id': 9090}
                values[old.entity_id] = tuple(c.model_copy(update=changes) if c.raw_id == shadow.member.raw_id
                    else c for c in values[old.entity_id])
            return values
        monkeypatch.setattr(GlobalWorldFusion, 'spatial_original_attribution_claims', claims)
    before = (actor.world.current_snapshot(), actor.world._revision, len(harness.wire), len(harness.journal))
    with pytest.raises((WorldPublicationRefused, ValueError)):
        harness.publish(prepare(mode='strict' if defect == 'strict-mode' else 'physical_only_quarantine'), 4)
    assert before == (actor.world.current_snapshot(), actor.world._revision, len(harness.wire), len(harness.journal))
    assert actor.identity.stats().pending_reservations == 0


@pytest.mark.parametrize('barrier', ['queued-revision', 'source-stop', 'source-epoch', 'wire-expiry'])
def test_unselected_member_replacement_revalidates_actual_commit(monkeypatch, barrier):
    from reid.mv3dt_person_association import SpatialLifecycleReceipt
    actor, harness, clock, old, shadow, joined, prepare = _held_kitchen_with_current_provisional_replacement(monkeypatch)
    candidate = prepare()
    before = (actor.world.current_snapshot(), actor.world._revision, len(harness.wire), len(harness.journal))
    if barrier == 'source-stop':
        actor.identity.stop_image_region_source(actor.sources[1], graph_id='r101-graph')
    elif barrier == 'source-epoch':
        actor.identity.reset_source(replace(actor.sources[1], source_epoch=1))
    elif barrier == 'wire-expiry': actor.us = old.stale_after_us+1
    else:
        current = next(o for m, o in actor.originals.items() if m.source_id == 1)
        stamp = replace(current.stamp, frame_id=5, observation_id='next-native-kitchen',
            acquired_at_s=current.stamp.acquired_at_s+.01, media_time_s=current.stamp.media_time_s+.01)
        actor.identity.observe_spatial(replace(current, stamp=stamp), SpatialLifecycleReceipt('next-native', stamp, 'ACTIVE'))
    with pytest.raises(WorldPublicationRefused): harness.publish(candidate, 4)
    assert before == (actor.world.current_snapshot(), actor.world._revision, len(harness.wire), len(harness.journal))
    assert actor.identity.stats().pending_reservations == 0


def _single_historical_component_transfer(monkeypatch):
    """Public R100 shape; synthetic geometry, clocks and native receipts."""
    actor, harness, clock = _r101_actor(monkeypatch)
    houses = {(1, 1): HouseholdDescriptor('visitor', 'single-old-k', 1002, 1),
              (2, 101): HouseholdDescriptor('visitor', 'single-old-f', 1007, 1)}
    first = harness.publish(actor.prepare_image(actor.rows, frame=2,
        semantic_joints=actor.joints, households=houses,
        source_changes={1: {'world_covariance': None}}), 2)
    old, = first.entities
    assert old.attribution_quarantine is not None
    assert old.attribution_quarantine.component_entity_ids == (old.entity_id,)
    assert next(s.camera_id for s in old.sources if s.accepted) == 'family-room'
    actor.us += 100_000
    changed = {**actor.joints, (2, 101): tuple((i, u, v + 60., c)
        for i, u, v, c in actor.joints[(2, 101)])}
    houses = {**houses, (2, 101): HouseholdDescriptor('visitor', 'single-current-f', 1000, 1)}

    def prepare(frame=3):
        return actor.prepare_image(actor.rows, frame=frame, semantic_joints=changed,
            households=houses, image_y={(2, 101): 90.},
            source_changes={1: {'world_covariance': None}, 2: {'world_covariance': None}})
    return actor, harness, clock, old, prepare


def test_single_historical_component_certified_transfer_needs_no_visible_origin(monkeypatch):
    actor, harness, clock, old, prepare = _single_historical_component_transfer(monkeypatch)
    history = set(actor.world._fusion.spatial_original_attribution_claims()[old.entity_id])
    final = harness.publish(prepare(), 3)
    transferred, = final.entities
    assert transferred.entity_id != old.entity_id
    _r175_assert_same_original(old, transferred)
    assert transferred.subject.stable_id is None
    assert all(d.household_conflict is None for d in actor.memberships)
    retained = actor.world._fusion.spatial_attribution_claims()
    assert old.entity_id not in retained and transferred.entity_id in retained
    custody, = actor.world._fusion.split_custody_originals()
    assert custody.entity.attribution_quarantine is old.attribution_quarantine
    assert set(custody.row.attribution_claims) == history
    assert all(c.entity_id == old.entity_id for c in history)
    fixed_deadline = custody.owner_diagnostic_view.owner_deadline_monotonic_s
    actor.us += 50_000
    follow = harness.publish(prepare(4), 4)
    again, = follow.entities
    _r175_assert_same_original(old, again)
    custody, = actor.world._fusion.split_custody_originals()
    assert set(custody.row.attribution_claims) == history
    assert custody.owner_diagnostic_view.owner_deadline_monotonic_s == fixed_deadline
    actor.us = old.stale_after_us + 1
    expired = harness.publish(actor.prepare_image({1: [], 2: []}, frame=20), 20)
    assert not expired.entities
    actual = actual_node_wire(actor, harness.wire, 'single-historical-component-transfer')
    assert [len(s['people']) for s in actual['snapshots']] == [1, 1, 1, 0]
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert actor.identity.stats().pending_reservations == 0


@pytest.mark.parametrize('defect', ['missing-transfer', 'missing-q', 'foreign-q',
    'wrong-envelope', 'wrong-stamp', 'missing-history', 'foreign-history'])
def test_single_historical_component_requires_exact_transfer_history(monkeypatch, defect):
    from noesis_core.world import GlobalWorldFusion
    actor, harness, clock, old, prepare = _single_historical_component_transfer(monkeypatch)
    actual = GlobalWorldFusion.split_custody_originals

    def corrupted(fusion):
        records = actual(fusion)
        if fusion is actor.world._fusion or not records:
            return records
        if defect == 'missing-transfer':
            return ()
        transfer, = records
        if defect in {'missing-q', 'foreign-q'}:
            q = transfer.entity.attribution_quarantine
            q = None if defect == 'missing-q' else q.model_copy(update={
                'component_entity_ids': ('physical:run-synthetic:999',)})
            transfer = replace(transfer, entity=transfer.entity.model_copy(update={'attribution_quarantine': q}))
        elif defect == 'wrong-envelope':
            transfer = replace(transfer, row=replace(transfer.row, envelope=transfer.row.envelope.model_copy()))
        elif defect == 'wrong-stamp':
            stamp = transfer.row.spatial_stamp
            transfer = replace(transfer, row=replace(transfer.row, spatial_stamp=replace(stamp,
                member=replace(stamp.member, source_epoch=stamp.member.source_epoch + 1))))
        else:
            claims = transfer.row.attribution_claims
            claims = claims[:-1] if defect == 'missing-history' else claims + (
                claims[0].model_copy(update={'source_epoch': 99}),)
            transfer = replace(transfer, row=replace(transfer.row, attribution_claims=claims))
        return (transfer,)

    monkeypatch.setattr(GlobalWorldFusion, 'split_custody_originals', corrupted)
    before = (actor.world.current_snapshot(), actor.world._revision, len(harness.wire), len(harness.journal))
    with pytest.raises(WorldPublicationRefused, match='retained_attribution_conflict_not_authorized'):
        prepare()
    assert before == (actor.world.current_snapshot(), actor.world._revision, len(harness.wire), len(harness.journal))
    assert actor.identity.stats().pending_reservations == 0


@pytest.mark.parametrize('barrier', ['stop', 'source-epoch', 'queued-revision', 'wire-expiry', 'owner-expiry'])
def test_single_historical_component_keeps_queued_commit_barriers(monkeypatch, barrier):
    from reid.mv3dt_person_association import SpatialLifecycleReceipt
    actor, harness, clock, old, prepare = _single_historical_component_transfer(monkeypatch)
    candidate = prepare()
    before = (actor.world.current_snapshot(), actor.world._revision, len(harness.wire), len(harness.journal))
    if barrier == 'stop':
        actor.identity.stop_image_region_source(actor.sources[1], graph_id='r101-graph')
    elif barrier == 'source-epoch':
        actor.identity.reset_source(replace(actor.sources[1], source_epoch=1))
    elif barrier == 'queued-revision':
        original = next(o for m, o in actor.originals.items() if m.source_id == 2)
        stamp = replace(original.stamp, frame_id=4, observation_id='single-component-next-native',
            acquired_at_s=original.stamp.acquired_at_s + .01, media_time_s=original.stamp.media_time_s + .01)
        actor.identity.observe_spatial(replace(original, stamp=stamp),
            SpatialLifecycleReceipt('single-component-next-active', stamp, 'ACTIVE'))
    elif barrier == 'wire-expiry':
        actor.us = old.stale_after_us + 1
    else:
        clock.monotonic = lambda: 121.
    with pytest.raises(WorldPublicationRefused):
        harness.publish(candidate, 3)
    assert before == (actor.world.current_snapshot(), actor.world._revision, len(harness.wire), len(harness.journal))
    assert actor.identity.stats().pending_reservations == 0


def _clean_selected_provisional_with_unselected_settled_claim():
    """Public shape of live PID718460; metric/native data are synthetic."""
    from tests.test_xc3d_world_physical_person import ImageQuarantineActor, ActualPublicationHarness, COV
    actor = ImageQuarantineActor()
    harness = ActualPublicationHarness(actor)
    rows = {1: [(23, 1., 11, 1171)], 2: [(15, 1.03, 8, 705)]}
    houses = {(1, 23): HouseholdDescriptor('visitor', 'clean-old-k23', 1006, 2),
              (2, 15): HouseholdDescriptor('provisional')}
    image_y = {(1, 23): 30., (2, 15): 30., (1, 22): 110.}
    quality = {1: {'world_covariance': tuple(v * 4 for v in COV)},
               2: {'world_covariance': COV}}
    first = harness.publish(actor.prepare_image(rows, frame=1259, households=houses,
        image_y=image_y, source_changes=quality), 1259)
    old, = first.entities
    assert old.attribution_quarantine is None
    assert next(s.camera_id for s in old.sources if s.accepted) == 'family-room'
    historical = actor.world._fusion.spatial_original_attribution_claims()[old.entity_id]
    assert {(c.camera_id, c.household_state) for c in historical} == {
        ('family-room', 'provisional'), ('kitchen', 'visitor')}
    actor.us += 100_000
    houses = {**houses, (1, 23): HouseholdDescriptor('provisional'),
              (1, 22): HouseholdDescriptor('visitor', 'clean-new-k22', 1006, 2)}
    original_prepare = actor.world.prepare

    def no_new_alternative_point(source, tracks, **kwargs):
        return original_prepare(source, [dict(t, world=None, world_valid=False,
            world_measurement_accepted=False, world_covariance=None)
            if source == 1 and t['tracker_id'] == 23 else t for t in tracks], **kwargs)
    actor.world.prepare = no_new_alternative_point

    def prepare(*, mode='physical_only_quarantine', households=None, frame=1261):
        return actor.prepare_image({1: [(23, 1., 11, 1171), (22, 4., 12, 1261)]}, frame=frame,
            households=houses if households is None else households, image_y=image_y,
            source_changes=quality, mode=mode)
    return actor, harness, old, historical, prepare


def test_clean_selected_provisional_quarantines_advanced_unselected_label():
    actor, harness, old, historical, prepare = _clean_selected_provisional_with_unselected_settled_claim()
    candidate = prepare()
    assert all(dto.household_conflict is None for dto in actor.memberships)
    final = harness.publish(candidate, 1261)
    held = next(e for e in final.entities if e.entity_id == old.entity_id)
    _r175_assert_same_original(old, held)
    quarantine = held.attribution_quarantine
    assert quarantine is not None and held.subject.stable_id is None
    assert set(historical) <= set(quarantine.owners + quarantine.retained_owners)
    assert any(c.raw_id == 23 and c.frame_id == 1261 and c.household_state == 'provisional'
               for c in quarantine.owners)
    assert any(c.raw_id == 22 and c.household_stable_id == 1006 for c in quarantine.owners)
    newborn = next(e for e in final.entities if e.entity_id != old.entity_id)
    assert newborn.position.x == 4. and newborn.observed_at_us > old.observed_at_us
    assert newborn.stale_after_us == newborn.observed_at_us + 2_000_000
    actor.us += 50_000
    following = harness.publish(prepare(frame=1262), 1262)
    unchanged = next(e for e in following.entities if e.entity_id == old.entity_id)
    _r175_assert_same_original(old, unchanged)
    assert set(historical) <= set(unchanged.attribution_quarantine.owners +
                                  unchanged.attribution_quarantine.retained_owners)
    actor.us = old.stale_after_us + 1
    expired = harness.publish(actor.prepare_image({1: [], 2: []}, frame=1280), 1280)
    assert all(e.entity_id != old.entity_id for e in expired.entities)
    actual = actual_node_wire(actor, harness.wire, 'clean-unselected-label-attenuation')
    assert [len(s['people']) for s in actual['snapshots']] == [1, 2, 2, len(expired.entities)]
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert all(p['stableId'] is None for p in actual['snapshots'][1]['people'])
    assert actor.identity.stats().pending_reservations == 0


@pytest.mark.parametrize('field', ['physical_serial', 'source_epoch', 'raw_id', 'lifecycle_generation',
    'lifecycle_started_frame', 'clock_domain', 'clock_basis', 'frame_id', 'observation_id',
    'acquired_at_s', 'media_time_s', 'household_state', 'household_constraint'])
def test_clean_unselected_label_attenuation_rejects_foreign_or_nonmonotone_claim(monkeypatch, field):
    actor, harness, old, historical, prepare = _clean_selected_provisional_with_unselected_settled_claim()
    previous = next(c for c in historical if c.camera_id == 'kitchen')
    actual = actor.world._attribution_claim

    def corrupt(key, original):
        claim = actual(key, original)
        if claim.raw_id != 23 or claim.frame_id != 1261:
            return claim
        values = {'physical_serial': 999, 'source_epoch': 1, 'raw_id': 999,
            'lifecycle_generation': 99, 'lifecycle_started_frame': 1261,
            'clock_domain': 'foreign-clock', 'clock_basis': 'source-media',
            'frame_id': previous.frame_id, 'observation_id': previous.observation_id,
            'acquired_at_s': previous.acquired_at_s, 'media_time_s': previous.media_time_s,
            'household_state': 'visitor', 'household_constraint': 'ineligible'}
        return claim.model_copy(update={field: values[field]})
    monkeypatch.setattr(actor.world, '_attribution_claim', corrupt)
    before = (actor.world.current_snapshot(), actor.world._revision, len(harness.wire), len(harness.journal))
    with pytest.raises(WorldPublicationRefused):
        prepare()
    assert before == (actor.world.current_snapshot(), actor.world._revision, len(harness.wire), len(harness.journal))
    assert actor.identity.stats().pending_reservations == 0


@pytest.mark.parametrize('barrier', ['wrong-envelope', 'missing-custody', 'foreign-history',
    'stop-original', 'stop-alternative', 'source-epoch', 'queued-revision', 'queued-expiry'])
def test_clean_unselected_label_attenuation_keeps_real_custody_and_commit_barriers(monkeypatch, barrier):
    from noesis_core.world import GlobalWorldFusion
    from reid.mv3dt_person_association import SpatialLifecycleReceipt
    actor, harness, old, historical, prepare = _clean_selected_provisional_with_unselected_settled_claim()
    if barrier in {'wrong-envelope', 'missing-custody'}:
        actual = GlobalWorldFusion.committed_spatial_original
        def changed(fusion, key, **kwargs):
            value = actual(fusion, key, **kwargs)
            if value is not None and fusion is not actor.world._fusion and fusion.physical_entity_id(key) == old.entity_id:
                return None if barrier == 'missing-custody' else (value[0].model_copy(), *value[1:])
            return value
        monkeypatch.setattr(GlobalWorldFusion, 'committed_spatial_original', changed)
    elif barrier == 'foreign-history':
        actual = GlobalWorldFusion.spatial_original_attribution_claims
        def changed(fusion):
            value = actual(fusion)
            if fusion is not actor.world._fusion and old.entity_id in value:
                value[old.entity_id] += (historical[0].model_copy(update={'source_epoch': 99}),)
            return value
        monkeypatch.setattr(GlobalWorldFusion, 'spatial_original_attribution_claims', changed)
    before = (actor.world.current_snapshot(), actor.world._revision, len(harness.wire), len(harness.journal))
    with pytest.raises(WorldPublicationRefused):
        candidate = prepare()
        if barrier.startswith('stop-'):
            actor.identity.stop_image_region_source(actor.sources[1 if barrier == 'stop-original' else 0],
                                                   graph_id='r101-graph')
        elif barrier == 'source-epoch':
            actor.identity.reset_source(replace(actor.sources[1], source_epoch=1))
        elif barrier == 'queued-revision':
            previous = next(o for o in actor.originals.values() if o.stamp.member.source_id == 2)
            stamp = replace(previous.stamp, frame_id=1262, observation_id='clean-next-native-original',
                acquired_at_s=previous.stamp.acquired_at_s + .01, media_time_s=126.2)
            actor.identity.observe_spatial(replace(previous, stamp=stamp),
                SpatialLifecycleReceipt('clean-next-native', stamp, 'ACTIVE'))
        elif barrier == 'queued-expiry':
            actor.us = old.stale_after_us + 1
        harness.publish(candidate, 1261)
    assert before == (actor.world.current_snapshot(), actor.world._revision, len(harness.wire), len(harness.journal))
    assert actor.identity.stats().pending_reservations == 0


def test_clean_unselected_label_attenuation_does_not_relax_strict_mode():
    actor, harness, old, historical, prepare = _clean_selected_provisional_with_unselected_settled_claim()
    before = (actor.world.current_snapshot(), len(harness.wire), len(harness.journal))
    with pytest.raises(WorldPublicationRefused, match='retained_attribution_conflict_not_authorized'):
        prepare(mode='strict')
    assert before == (actor.world.current_snapshot(), len(harness.wire), len(harness.journal))
    assert actor.identity.stats().pending_reservations == 0


def _r201_forward_native_rekey(monkeypatch):
    actor, harness, clock, old, dto, _, _ = _r198_clean_original_with_exact_rekey(monkeypatch)
    actor.us += 20_000
    houses = {(1, 1): HouseholdDescriptor('provisional'),
        (2, 101): HouseholdDescriptor('visitor', 'r198-f', 1000, 1)}
    joints = {**actor.joints, (2, 101): tuple((i, u, v + 60., c)
        for i, u, v, c in actor.joints[(2, 101)])}
    def prepare(*, mode='physical_only_quarantine', households=None):
        return actor.prepare_image({2: actor.rows[2]}, frame=8,
            semantic_joints=joints, households=houses if households is None else households, image_y={(2, 101): 90.},
            source_changes={2: {'world_covariance': None}}, mode=mode)
    return actor, harness, clock, old, dto, prepare


def test_r201_forward_same_native_rekey_only_attenuates_labels(monkeypatch):
    actor, harness, clock, old, dto, prepare = _r201_forward_native_rekey(monkeypatch)
    historical = actor.world._fusion.spatial_original_attribution_claims()[old.entity_id]
    candidate = prepare()
    family = next(d for d in actor.memberships if d.physical_key == dto.physical_key)
    member, = family.members
    assert member.observation.frame_id == 8
    assert any(e.eligible and e.observation == member.observation for e in family.eligibility)
    assert family.eligibility[0].lifecycle_receipt.native_state == 'ACTIVE'
    final = harness.publish(candidate, 8)
    current, = final.entities
    _r175_assert_same_original(old, current)
    assert current.attribution_quarantine is not None and current.subject.stable_id is None
    assert set(historical) <= set(current.attribution_quarantine.owners + current.attribution_quarantine.retained_owners)
    actor.us += 50_000
    houses = {(1, 1): HouseholdDescriptor('provisional'),
        (2, 101): HouseholdDescriptor('visitor', 'r198-f', 1000, 1)}
    refreshed = harness.publish(actor.prepare_image({1: [(1, 1.06, 1, 1)]}, frame=9,
        households=houses, semantic_joints=actor.joints), 9)
    advanced, = refreshed.entities
    assert advanced.position.x == 1.06 and advanced.observed_at_us > old.observed_at_us
    assert any(s.accepted and s.observation_id == 'synthetic-native:1:1:9' for s in advanced.sources)
    assert advanced.subject.stable_id is None
    assert {c for c in historical if c.camera_id == 'family-room'} <= set(
        advanced.attribution_quarantine.owners + advanced.attribution_quarantine.retained_owners)
    assert actor.identity.stats().pending_reservations == 0
    node = actual_node_wire(actor, harness.wire, 'r201-forward-native-label-only')
    assert all(d['accepted'] for d in node['diagnostics'])
    assert all(len(s['people']) == 1 for s in node['snapshots'])


@pytest.mark.parametrize('field', ['run_id', 'camera_id', 'source_id', 'source_epoch', 'raw_id',
    'lifecycle_generation', 'lifecycle_started_frame', 'clock_domain', 'clock_basis',
    'household_state', 'household_identity_key',
    'household_visitor_generation', 'household_constraint',
    'frame_id', 'observation_id', 'acquired_at_s', 'media_time_s'])
def test_r201_forward_label_provenance_requires_exact_lifecycle_household_and_monotone_stamp(monkeypatch, field):
    actor, harness, clock, old, dto, prepare = _r201_forward_native_rekey(monkeypatch)
    actual = actor.world._attribution_claim
    historical = next(c for c in actor.world._fusion.spatial_original_attribution_claims()[old.entity_id]
        if c.camera_id == 'family-room')
    def wrong(key, original):
        claim = actual(key, original)
        if key == dto.physical_key:
            value = getattr(claim, field)
            if field in {'physical_serial', 'frame_id', 'observation_id', 'acquired_at_s', 'media_time_s'}:
                value = getattr(historical, field)
            else:
                value = value + ':foreign' if isinstance(value, str) else 1 if value is None else value + 1
            return claim.model_copy(update={field: value})
        return claim
    monkeypatch.setattr(actor.world, '_attribution_claim', wrong)
    before = (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    with pytest.raises((WorldPublicationRefused, ValueError)):
        prepare()
    assert before == (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    assert actor.identity.stats().pending_reservations == 0


@pytest.mark.parametrize('field', list(WorldAttributionClaim.model_fields))
def test_r201_forward_comparison_cannot_substitute_any_immutable_or_monotone_field(monkeypatch, field):
    from reid.mv3dt_person_association import SourceObservation
    actor, harness, clock, old, dto, prepare = _r201_forward_native_rekey(monkeypatch)
    candidate = prepare()
    dto = next(d for d in actor.memberships if d.physical_key == dto.physical_key)
    member, = dto.members
    new = actor.world._attribution_claim(dto.physical_key, SourceObservation(member.observation, member.household))
    previous = next(c for c in actor.world._fusion.spatial_original_attribution_claims()[old.entity_id]
        if c.camera_id == 'family-room')
    assert actor.world._same_forward_rekeyed_native_claim(previous, new)
    if field in {'physical_serial', 'frame_id', 'observation_id', 'acquired_at_s', 'media_time_s'}:
        value = getattr(previous, field)
    else:
        value = getattr(new, field)
        value = value + ':foreign' if isinstance(value, str) else 1 if value is None else value + 1
    assert not actor.world._same_forward_rekeyed_native_claim(previous, new.model_copy(update={field: value}))
    actor.world.discard(candidate)
    assert actor.identity.stats().pending_reservations == 0


@pytest.mark.parametrize('barrier', ['stop-original', 'stop-rekey', 'epoch-original', 'epoch-rekey',
    'revision', 'native-reuse', 'wire-expiry', 'discard'])
def test_r201_forward_label_projection_requires_final_original_and_owner_custody(monkeypatch, barrier):
    from reid.mv3dt_person_association import SpatialLifecycleReceipt, MemberRemoval
    actor, harness, clock, old, dto, prepare = _r201_forward_native_rekey(monkeypatch)
    candidate = prepare()
    before = (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    if barrier == 'discard':
        actor.world.discard(candidate)
    else:
        if barrier.startswith('stop-'):
            actor.identity.stop_image_region_source(actor.sources[0 if barrier == 'stop-original' else 1], graph_id='r101-graph')
        elif barrier.startswith('epoch-'):
            actor.identity.reset_source(replace(actor.sources[0 if barrier == 'epoch-original' else 1], source_epoch=1))
        elif barrier == 'wire-expiry':
            actor.us = old.stale_after_us + 1
        else:
            original = next(o for m, o in actor.originals.items() if m.source_id == 2)
            stamp = replace(original.stamp, frame_id=9, observation_id='r201-current-next-native',
                acquired_at_s=actor.us / 1e6 + .01, media_time_s=.9)
            if barrier == 'native-reuse':
                removed = actor.identity.withdraw(MemberRemoval('r201-actual-termination', original.stamp,
                    'termination', 9, actor.us / 1e6, original.stamp.clock_domain, .9))
                assert removed.disposition == 'removed'
                stamp = replace(stamp, member=replace(stamp.member, lifecycle_generation=2),
                    frame_id=10, lifecycle_started_frame=10, media_time_s=1.)
            actor.identity.observe_spatial(replace(original, stamp=stamp),
                SpatialLifecycleReceipt('r201-next-native', stamp, 'ACTIVE'))
        with pytest.raises(WorldPublicationRefused):
            harness.publish(candidate, 8)
    assert before == (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    assert actor.identity.stats().pending_reservations == 0


def test_r201_forward_rekey_does_not_relax_strict_attribution(monkeypatch):
    actor, harness, clock, old, dto, prepare = _r201_forward_native_rekey(monkeypatch)
    before = (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    with pytest.raises(WorldPublicationRefused, match='retained_attribution_conflict_not_authorized'):
        prepare(mode='strict')
    assert before == (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    assert actor.identity.stats().pending_reservations == 0


@pytest.mark.parametrize('household', [HouseholdDescriptor('visitor', 'different-current-person', 1000, 1),
    HouseholdDescriptor('visitor', 'r198-f', 1000, 2),
    HouseholdDescriptor('resident', 'different-current-person', 1000)])
def test_r201_forward_current_household_change_is_not_label_rekey_authority(monkeypatch, household):
    actor, harness, clock, old, dto, prepare = _r201_forward_native_rekey(monkeypatch)
    before = (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    with pytest.raises(WorldPublicationRefused, match='retained_attribution_conflict_not_authorized'):
        prepare(households={(1, 1): HouseholdDescriptor('provisional'), (2, 101): household})
    assert before == (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    assert actor.identity.stats().pending_reservations == 0


def test_r201_forward_current_inactive_receipt_cannot_anchor_labels(monkeypatch):
    from reid.mv3dt_person_association import SpatialLifecycleReceipt
    actor, harness, clock, old, dto, prepare = _r201_forward_native_rekey(monkeypatch)
    candidate = prepare()
    actor.world.discard(candidate)
    original = next(o for m, o in actor.originals.items() if m.source_id == 2)
    result = actor.identity.observe_spatial(original,
        SpatialLifecycleReceipt('r201-actual-inactive-owner', original.stamp, 'INACTIVE'))
    assert result.disposition in {'observed', 'unresolved'}
    memberships = actor.identity.spatial_memberships(at_interval=actor.context)
    current = next(d for d in memberships if d.physical_key == dto.physical_key)
    assert current.eligibility[0].lifecycle_receipt.native_state == 'INACTIVE'
    preps, _ = actor.normalize({2: []}, frame=9, native_tokens=True)
    before = (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    with pytest.raises(WorldPublicationRefused, match='retained_attribution_conflict_not_authorized'):
        actor.world.prepare_spatial(2, preps, memberships, at_interval=actor.context,
            attribution_mode='physical_only_quarantine')
    assert before == (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    assert actor.identity.stats().pending_reservations == 0


def _r198_clean_original_with_exact_rekey(monkeypatch):
    """A real public native-gap split; no invented claims or termination."""
    from reid.mv3dt_person_association import SpatialLifecycleUpdate, SpatialClockContext, SpatialClockInterval
    actor, harness, clock = _r101_actor(monkeypatch)
    houses = {(1, 1): HouseholdDescriptor('provisional'),
        (2, 101): HouseholdDescriptor('visitor', 'r198-f', 1000, 1)}
    first = harness.publish(actor.prepare_image(actor.rows, frame=2, semantic_joints=actor.joints,
        households=houses, source_changes={2: {'world_covariance': None}}), 2)
    old, = first.entities
    assert old.attribution_quarantine is None
    assert next(s.camera_id for s in old.sources if s.accepted) == 'kitchen'
    actor.us += 300_000
    fresh = harness.publish(actor.prepare_image({1: [(1, 1.02, 1, 1)]}, frame=3,
        semantic_joints=actor.joints, households=houses), 3)
    old, = fresh.entities
    family = next(o.stamp for m, o in actor.originals.items() if m.source_id == 2)
    actor.us += 150_000
    # A genuine gap beyond continuity is allowed to split membership, never
    # fabricate a native end or revoke the still-live Kitchen original.
    result = actor.identity.update_spatial_lifecycle(SpatialLifecycleUpdate(
        'r198-real-retained-gap', family, 4, actor.us / 1e6, family.clock_domain,
        family.media_time_s + .45))
    assert result.disposition == 'observed' and result.withdrawn_members == (family.member,)
    actor.context = SpatialClockContext(tuple(SpatialClockInterval(b,
        10.2, 10.2, actor.us / 1e6, actor.us / 1e6) for b in actor.sources))
    actor.memberships = actor.identity.spatial_memberships(at_interval=actor.context)
    family_dto = next(d for d in actor.memberships if any(m.observation == family for m in d.members))
    assert not old.entity_id.endswith(':' + str(family_dto.physical_key.serial))
    assert all(d.household_conflict is None for d in actor.memberships)
    def prepare(frame=5, *, mode='physical_only_quarantine'):
        preps, _ = actor.normalize({2: []}, frame=frame, native_tokens=True)
        return actor.world.prepare_spatial(2, preps, actor.memberships, at_interval=actor.context,
            withdrawn_members=result.withdrawn_members, attribution_mode=mode)
    def refresh():
        changed = {**actor.joints, (2, 101): tuple((i, u, v + 60., c)
            for i, u, v, c in actor.joints[(2, 101)])}
        return actor.prepare_image({1: [(1, 1.06, 1, 1)], 2: actor.rows[2]}, frame=8,
            semantic_joints=changed, households=houses, image_y={(2, 101): 90.},
            source_changes={2: {'world_covariance': None}})
    return actor, harness, clock, old, family_dto, prepare, refresh


def test_r198_exact_current_original_rekey_is_label_provenance_only(monkeypatch):
    actor, harness, clock, old, family_dto, prepare, refresh = _r198_clean_original_with_exact_rekey(monkeypatch)
    before = actor.world._fusion.spatial_original_attribution_claims()[old.entity_id]
    final = harness.publish(prepare(), 5)
    current, = final.entities
    _r175_assert_same_original(old, current)
    assert current.attribution_quarantine is not None and current.subject.stable_id is None
    assert set(before) <= set(current.attribution_quarantine.owners + current.attribution_quarantine.retained_owners)
    actor.us += 20_000
    unchanged = harness.publish(prepare(frame=6), 6)
    _r175_assert_same_original(old, unchanged.entities[0])
    actor.us += 50_000
    fresh = harness.publish(refresh(), 8)
    advanced, = fresh.entities
    assert advanced.observed_at_us > old.observed_at_us and advanced.position.x == 1.06
    assert any(s.accepted and s.observation_id == 'synthetic-native:1:1:8' for s in advanced.sources)
    assert advanced.subject.stable_id is None
    assert actor.identity.stats().pending_reservations == 0
    node = actual_node_wire(actor, harness.wire, 'r198-exact-original-rekey')
    assert all(d['accepted'] for d in node['diagnostics'])
    assert all(len(s['people']) == 1 for s in node['snapshots'])


@pytest.mark.parametrize('field', [name for name in WorldAttributionClaim.model_fields if name != 'physical_serial'])
def test_r198_rekey_comparison_requires_every_original_claim_field(monkeypatch, field):
    from reid.mv3dt_person_association import SourceObservation
    actor, harness, clock, old, dto, prepare, refresh = _r198_clean_original_with_exact_rekey(monkeypatch)
    member, = dto.members
    current = actor.world._attribution_claim(dto.physical_key, SourceObservation(member.observation, member.household))
    historical = next(c for c in actor.world._fusion.spatial_original_attribution_claims()[old.entity_id]
        if c.raw_id == member.observation.member.raw_id)
    assert actor.world._same_rekeyed_original_claim(historical, current)
    value = getattr(current, field)
    changed = value + ':different' if isinstance(value, str) else 1 if value is None else value + 1
    assert not actor.world._same_rekeyed_original_claim(historical, current.model_copy(update={field: changed}))


@pytest.mark.parametrize('field', ['observation_id', 'raw_id', 'lifecycle_generation', 'source_epoch',
    'frame_id', 'acquired_at_s', 'clock_domain', 'household_identity_key'])
def test_r198_wrong_current_rekey_cannot_anchor_world_labels(monkeypatch, field):
    actor, harness, clock, old, dto, prepare, refresh = _r198_clean_original_with_exact_rekey(monkeypatch)
    actual = actor.world._attribution_claim
    def wrong(key, original):
        claim = actual(key, original)
        if key == dto.physical_key:
            value = getattr(claim, field)
            claim = claim.model_copy(update={field: value + ':foreign' if isinstance(value, str) else value + 1})
        return claim
    monkeypatch.setattr(actor.world, '_attribution_claim', wrong)
    before = (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    with pytest.raises(WorldPublicationRefused, match='retained_attribution_conflict_not_authorized'):
        prepare()
    assert before == (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    assert actor.identity.stats().pending_reservations == 0


@pytest.mark.parametrize('barrier', ['stop', 'rekey-stop', 'epoch', 'revision', 'rekey-revision',
    'native-reuse', 'wire-expiry', 'discard'])
def test_r198_label_rekey_never_replaces_final_native_custody(monkeypatch, barrier):
    from reid.mv3dt_person_association import SpatialLifecycleReceipt, MemberRemoval
    actor, harness, clock, old, dto, prepare, refresh = _r198_clean_original_with_exact_rekey(monkeypatch)
    candidate = prepare()
    before = (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    if barrier == 'discard':
        actor.world.discard(candidate)
    else:
        if barrier == 'stop':
            actor.identity.stop_image_region_source(actor.sources[0], graph_id='r101-graph')
        elif barrier == 'rekey-stop':
            actor.identity.stop_image_region_source(actor.sources[1], graph_id='r101-graph')
        elif barrier == 'epoch':
            actor.identity.reset_source(replace(actor.sources[0], source_epoch=1))
        elif barrier == 'wire-expiry':
            actor.us = old.stale_after_us + 1
        else:
            original = next(o for m, o in actor.originals.items()
                if m.source_id == (2 if barrier in {'rekey-revision', 'native-reuse'} else 1))
            stamp = replace(original.stamp, frame_id=8, observation_id='r198-actual-next-native',
                acquired_at_s=actor.us / 1e6 + .01, media_time_s=.8)
            if barrier == 'native-reuse':
                removed = actor.identity.withdraw(MemberRemoval('r198-real-termination', original.stamp,
                    'termination', 7, actor.us / 1e6, original.stamp.clock_domain, .7))
                assert removed.disposition == 'removed'
                stamp = replace(stamp, member=replace(stamp.member, lifecycle_generation=2),
                    lifecycle_started_frame=8)
            actor.identity.observe_spatial(replace(original, stamp=stamp),
                SpatialLifecycleReceipt('r198-next-native', stamp, 'ACTIVE'))
        with pytest.raises(WorldPublicationRefused):
            harness.publish(candidate, 5)
    assert before == (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    assert actor.identity.stats().pending_reservations == 0


def test_r198_exact_rekey_does_not_relax_strict_attribution_mode(monkeypatch):
    actor, harness, clock, old, dto, prepare, refresh = _r198_clean_original_with_exact_rekey(monkeypatch)
    before = (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    with pytest.raises(WorldPublicationRefused, match='retained_attribution_conflict_not_authorized'):
        prepare(mode='strict')
    assert before == (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    assert actor.identity.stats().pending_reservations == 0


def _r196_transferred_historical_component(monkeypatch, *, before_split=None):
    """Public R88-shaped ownership; synthetic geometry/native receipts only."""
    actor, harness, clock = _r101_actor(monkeypatch)
    rows = {1: actor.rows[1], 2: actor.rows[2] + [(102, 4., 2, 1)]}
    joints = {**actor.joints, (2, 102): tuple((i, u, v + 70., c)
        for i, u, v, c in actor.joints[(2, 101)])}
    houses = {(1, 1): HouseholdDescriptor('visitor', 'r196-k', 1004, 1),
        (2, 101): HouseholdDescriptor('visitor', 'r196-f', 1000, 1),
        (2, 102): HouseholdDescriptor('visitor', 'r196-f', 1000, 1)}
    first = harness.publish(actor.prepare_image(rows, frame=2, semantic_joints=joints,
        households=houses, image_y={(2, 102): 100.},
        source_changes={1: {'world_covariance': None}}), 2)
    original_id = next(o.stamp.observation_id for m, o in actor.originals.items() if m.raw_id == 101)
    old = next(e for e in first.entities if any(s.accepted and s.observation_id == original_id
        for s in e.sources))
    assert old.attribution_quarantine is not None
    assert actor.world._fusion._last_fused[old.entity_id].entity.attribution_quarantine is None
    if before_split is not None:
        before_split(actor, old)
    actor.us += 100_000
    changed = {**joints, (2, 101): tuple((i, u, v + 60., c)
        for i, u, v, c in joints[(2, 101)])}
    houses[(2, 101)] = HouseholdDescriptor('provisional')
    split = harness.publish(actor.prepare_image(rows, frame=3, semantic_joints=changed,
        households=houses, image_y={(2, 101): 90., (2, 102): 100.},
        source_changes={1: {'world_covariance': None}, 2: {'world_covariance': None}}), 3)
    destination = next(d.physical_key for d in actor.memberships
        if any(m.observation.member.raw_id == 101 for m in d.members))
    transferred = next(e for e in split.entities if e.entity_id.endswith(':' + str(destination.serial)))
    assert transferred.entity_id != old.entity_id and transferred.position_state == 'held'
    _r175_assert_same_original(old, transferred)
    custody, = actor.world._fusion.split_custody_originals()
    assert custody.entity.attribution_quarantine is old.attribution_quarantine
    assert transferred.attribution_quarantine is None
    assert next(m.household.state for d in actor.memberships for m in d.members
        if m.observation.member.raw_id == 101) == 'provisional'
    assert all(d.household_conflict is None for d in actor.memberships)
    actor.us += 50_000
    actual_prepare = actor.world.prepare
    def unavailable_original(source, tracks, **kwargs):
        # Public current input: this native member has no new coordinate.
        tracks = [{**t, 'world_covariance': None} if source == 2 and t['tracker_id'] == 101
            else t for t in tracks]
        return actual_prepare(source, tracks, **kwargs)
    monkeypatch.setattr(actor.world, 'prepare', unavailable_original)
    def advance(frame=4):
        return actor.prepare_image({1: rows[1], 2: [rows[2][0], (102, 4.02, 2, 1)]}, frame=frame,
            semantic_joints=changed, households=houses,
            image_y={(2, 101): 90., (2, 102): 100.}, source_changes={1: {'world_covariance': None}})
    return actor, harness, clock, old, transferred, destination, advance


def test_r196_transferred_historical_claims_anchor_current_component(monkeypatch):
    actor, harness, clock, old, transferred, destination, advance = _r196_transferred_historical_component(monkeypatch)
    originals = actor.world._fusion.spatial_original_attribution_claims()[transferred.entity_id]
    final = harness.publish(advance(), 4)
    kept = next(e for e in final.entities if e.entity_id == transferred.entity_id)
    _r175_assert_same_original(old, kept)
    assert kept.subject.stable_id is None
    other = next(e for e in final.entities if e.entity_id != kept.entity_id)
    assert other.observed_at_us > old.observed_at_us
    assert any(s.accepted and s.observation_id == 'synthetic-native:2:102:4' for s in other.sources)
    assert actor.world._fusion.spatial_original_attribution_claims()[kept.entity_id] == originals
    assert other.attribution_quarantine is not None and other.subject.stable_id is None
    historical = set(old.attribution_quarantine.owners + old.attribution_quarantine.retained_owners)
    assert historical <= set(other.attribution_quarantine.owners + other.attribution_quarantine.retained_owners)
    assert kept.entity_id not in other.attribution_quarantine.component_entity_ids
    actor.us += 50_000
    follow = harness.publish(advance(frame=5), 5)
    _r175_assert_same_original(old, next(e for e in follow.entities if e.entity_id == kept.entity_id))
    assert actor.identity.stats().pending_reservations == 0
    assert all(d['accepted'] for d in actual_node_wire(actor, harness.wire, 'r196-transferred-historical')['diagnostics'])


@pytest.mark.parametrize('defect', ['missing-custody', 'missing-q', 'foreign-q',
    'wrong-original', 'wrong-birth', 'missing-claims'])
def test_r196_historical_anchor_requires_exact_committed_transfer(monkeypatch, defect):
    from noesis_core.world import GlobalWorldFusion
    actor, harness, clock, old, transferred, destination, advance = _r196_transferred_historical_component(monkeypatch)
    actual = GlobalWorldFusion.split_custody_originals
    def corrupted(fusion):
        records = actual(fusion)
        if fusion is actor.world._fusion or not records:
            return records
        if defect == 'missing-custody':
            return ()
        record, = records
        if defect in {'missing-q', 'foreign-q'}:
            q = record.entity.attribution_quarantine
            q = None if defect == 'missing-q' else q.model_copy(update={
                'component_entity_ids': ('physical:run-synthetic:999',)})
            record = replace(record, entity=record.entity.model_copy(update={'attribution_quarantine': q}))
        elif defect == 'wrong-original':
            record = replace(record, row=replace(record.row, envelope=record.row.envelope.model_copy(
                update={'observed_at_us': record.row.envelope.observed_at_us + 1})))
        elif defect == 'wrong-birth':
            stamp = record.row.spatial_stamp
            record = replace(record, row=replace(record.row, spatial_stamp=replace(stamp,
                member=replace(stamp.member, lifecycle_generation=stamp.member.lifecycle_generation + 1))))
        else:
            record = replace(record, row=replace(record.row,
                attribution_claims=record.row.attribution_claims[:-1]))
        return (record,)
    monkeypatch.setattr(GlobalWorldFusion, 'split_custody_originals', corrupted)
    before = (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    with pytest.raises(WorldPublicationRefused, match='retained_attribution_conflict_not_authorized'):
        advance()
    assert before == (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    assert actor.identity.stats().pending_reservations == 0


@pytest.mark.parametrize('defect', ['missing', 'foreign'])
def test_r196_corrupt_committed_label_evidence_refuses_before_transfer(monkeypatch, defect):
    from pydantic import ValidationError
    saved = {}
    def corrupt(actor, old):
        saved['actor'] = actor
        saved['snapshot'] = actor.world.current_snapshot()
        if defect == 'missing':
            actor.world._fusion.set_attribution_quarantines({})
        else:
            records = actor.world._fusion.attribution_quarantines()
            records[old.entity_id] = old.attribution_quarantine.model_copy(
                update={'association_revision': old.attribution_quarantine.association_revision + 1})
            actor.world._fusion.set_attribution_quarantines(records)
    error = WorldPublicationRefused if defect == 'missing' else ValidationError
    reason = 'canonical_identity_duplicate' if defect == 'missing' else 'attribution conflict digest mismatch'
    with pytest.raises(error, match=reason):
        _r196_transferred_historical_component(monkeypatch, before_split=corrupt)
    assert saved['actor'].world.current_snapshot() is saved['snapshot']
    assert saved['actor'].identity.stats().pending_reservations == 0


@pytest.mark.parametrize('barrier', ['stop', 'source-epoch', 'owner-revision', 'wire-expiry', 'owner-expiry', 'discard'])
def test_r196_historical_labels_never_replace_commit_authority(monkeypatch, barrier):
    from reid.mv3dt_person_association import SpatialLifecycleReceipt
    actor, harness, clock, old, transferred, destination, advance = _r196_transferred_historical_component(monkeypatch)
    candidate = advance()
    before = (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    if barrier == 'discard':
        actor.world.discard(candidate)
    else:
        if barrier == 'stop':
            actor.identity.stop_image_region_source(actor.sources[1], graph_id='r101-graph')
        elif barrier == 'source-epoch':
            actor.identity.reset_source(replace(actor.sources[1], source_epoch=1))
        elif barrier == 'owner-revision':
            original = next(o for m, o in actor.originals.items() if m.raw_id == 102)
            stamp = replace(original.stamp, frame_id=5, observation_id='r196-real-next-original',
                acquired_at_s=original.stamp.acquired_at_s + .01, media_time_s=original.stamp.media_time_s + .01)
            actor.identity.observe_spatial(replace(original, stamp=stamp),
                SpatialLifecycleReceipt('r196-real-active', stamp, 'ACTIVE'))
        elif barrier == 'wire-expiry':
            actor.us = old.stale_after_us + 1
        else:
            clock.monotonic = lambda: 121.
        with pytest.raises(WorldPublicationRefused):
            harness.publish(candidate, 4)
    assert before == (actor.world.current_snapshot(), len(harness.wire), len(harness.journal), actor.world._revision)
    assert actor.identity.stats().pending_reservations == 0


def _r175_retained_position_with_advanced_current_member(*,disjoint_conflict=False):
    actor,harness,first,old,prepare=_r156_retained_family_claim_transition(disjoint_conflict=disjoint_conflict)
    actor.us+=20_000
    advanced=harness.publish(actor.prepare_image({2:[(15,1.03,8,705)]},frame=1260,
        households={(2,15):HouseholdDescriptor('visitor','r156-f15',1006,1)},
        image_y={(2,15):30.},source_changes={2:{'world_covariance':None}}),1260,source_id=2)
    prior=next(e for e in advanced.entities if e.entity_id==old.entity_id)
    assert prior.observed_at_us==old.observed_at_us and prior.attribution_quarantine is not None
    current=next(o for o in actor.originals.values() if o.stamp.member.source_id==2)
    assert current.stamp.frame_id==1260
    actor.us+=20_000
    return actor,harness,old,prepare


def _r175_assert_same_original(old,kept):
    assert (kept.position,kept.covariance,kept.observed_at_us,kept.stale_after_us,
        kept.continuity_segment_id)==(old.position,old.covariance,old.observed_at_us,
        old.stale_after_us,old.continuity_segment_id)
    prior=next(s for s in old.sources if s.accepted)
    selected=next(s for s in kept.sources if s.accepted)
    fields=('observation_id','camera_id','observed_at_us','position','covariance',
        'world_frame','world_frame_revision','world_transform_sha256','calibration_revision')
    assert tuple(getattr(selected,f) for f in fields)==tuple(getattr(prior,f) for f in fields)


def test_r174_current_reserved_alternative_is_not_an_advancing_selected_original(monkeypatch):
    """Public R81-shaped Q transition with a newer unmeasured Family member."""
    actor,harness,old,prepare=_r175_retained_position_with_advanced_current_member()
    actual=actor.world._prepare_attribution_quarantines
    def observed(working,memberships,mode,**kwargs):
        # Observe actual prepared inputs without changing any authority/decision.
        key,stamp=next((key,stamp) for key,stamp in kwargs['committed_holds']
            if working.physical_entity_id(key)==old.entity_id)
        selected=working.committed_spatial_original(key,published_at_us=actor.us)
        reference=next(ref for dto,ref in kwargs['physical_selections'] if dto.physical_key==key)
        assert selected is not None and selected[1]==stamp
        assert stamp.member.source_id==2 and stamp.frame_id==1259
        assert reference.stamp.member.source_id==1 and reference.stamp.frame_id==1261
        assert all(dto.household_conflict is None for dto in memberships)
        return actual(working,memberships,mode,**kwargs)
    monkeypatch.setattr(actor.world,'_prepare_attribution_quarantines',observed)
    candidate=prepare(frame=1261)
    final=harness.publish(candidate,1261)
    kept=next(e for e in final.entities if e.entity_id==old.entity_id)
    _r175_assert_same_original(old,kept)
    assert kept.attribution_quarantine is not None and kept.subject.stable_id is None
    assert next(s.observation_id for s in kept.sources if s.accepted)==next(s.observation_id for s in old.sources if s.accepted)
    assert all(d['accepted'] for d in actual_node_wire(actor,harness.wire,'r174-reserved-not-selected')['diagnostics'])


@pytest.mark.parametrize('new_person_position',[False,True])
def test_r175_new_current_claim_is_not_a_grant_or_loss_of_retained_geometry(monkeypatch,new_person_position):
    from tests.test_xc3d_world_physical_person import COV
    actor,harness,old,_=_r175_retained_position_with_advanced_current_member()
    previous_claims=set(old.attribution_quarantine.owners+old.attribution_quarantine.retained_owners)
    if not new_person_position:
        actual=actor.world.prepare
        def unavailable(source,tracks,**kwargs):
            tracks=[{**t,'world':None,'world_valid':False,'world_measurement_accepted':False,
                'world_covariance':None} if t['tracker_id']==22 else t for t in tracks]
            return actual(source,tracks,**kwargs)
        monkeypatch.setattr(actor.world,'prepare',unavailable)
    candidate=actor.prepare_image({1:[(23,1.,11,1171),(21,4.,9,1116),(22,7.,12,1261)]},frame=1261,
        households={(1,23):HouseholdDescriptor('provisional'),
            (1,21):HouseholdDescriptor('visitor','r156-k21-old',1002,1),
            (1,22):HouseholdDescriptor('visitor','r175-new-person',1001,1)},
        image_y={(1,23):30.,(1,21):70.,(1,22):110.},
        source_changes={1:{'world_covariance':tuple(v*4 for v in COV)}})
    assert all(d.household_conflict is None for d in actor.memberships)
    final=harness.publish(candidate,1261)
    kept=next(e for e in final.entities if e.entity_id==old.entity_id)
    _r175_assert_same_original(old,kept)
    q=kept.attribution_quarantine
    assert q is not None and kept.subject.stable_id is None
    assert previous_claims<=set(q.owners+q.retained_owners)
    assert any(c.raw_id==22 and c.household_stable_id==1001 for c in q.owners)
    new=next(d.physical_key for d in actor.memberships if any(m.observation.member.raw_id==22 for m in d.members))
    assert any(e.entity_id.endswith(':'+str(new.serial)) for e in final.entities)==new_person_position
    assert actor.identity.stats().pending_reservations==0
    assert all(d['accepted'] for d in actual_node_wire(actor,harness.wire,'r175-new-current-claim-'+str(new_person_position))['diagnostics'])


@pytest.mark.parametrize('barrier',['missing-hold','foreign-original','missing-prior','forged-prior','disjoint-prior',
    'changed-selected-original','wrong-current-stamp','wrong-current-descriptor','wrong-current-member',
    'denied-current','stop','source-epoch','queued-revision','queued-expiry'])
def test_r175_retained_label_projection_preserves_all_authority_barriers(monkeypatch,barrier):
    from noesis_core.world import GlobalWorldFusion
    from noesis_core.contracts.world import WorldAttributionQuarantine
    from noesis.identity_association_service import SpatialMemberResult
    actor,harness,old,prepare=_r175_retained_position_with_advanced_current_member(
        disjoint_conflict=barrier=='disjoint-prior')
    if barrier=='missing-hold':
        actual=actor.world._prepare_attribution_quarantines
        def missing(working,memberships,mode,**kwargs):
            kwargs['committed_holds'][:]=[(k,s) for k,s in kwargs['committed_holds']
                if working.physical_entity_id(k)!=old.entity_id]
            return actual(working,memberships,mode,**kwargs)
        monkeypatch.setattr(actor.world,'_prepare_attribution_quarantines',missing)
    elif barrier=='foreign-original':
        actual=GlobalWorldFusion.spatial_original_attribution_claims
        def foreign(fusion):
            result=actual(fusion)
            if fusion is actor.world._fusion:
                result[old.entity_id]+=(result[old.entity_id][0].model_copy(update={'source_epoch':99}),)
            return result
        monkeypatch.setattr(GlobalWorldFusion,'spatial_original_attribution_claims',foreign)
    elif barrier=='missing-prior':actor.world._fusion.set_attribution_quarantines({})
    elif barrier=='forged-prior':
        payload=old.attribution_quarantine.model_dump(mode='json',exclude={'conflict_id'})
        payload['association_revision']+=99
        forged=WorldAttributionQuarantine(conflict_id=WorldAttributionQuarantine.digest(payload),**payload)
        actor.world._fusion.set_attribution_quarantines({old.entity_id:forged})
    elif barrier=='disjoint-prior':
        records=actor.world._fusion.attribution_quarantines()
        unrelated=next(q for q in records.values() if old.entity_id not in q.component_entity_ids)
        records[old.entity_id]=unrelated
        actual=GlobalWorldFusion.attribution_quarantines
        monkeypatch.setattr(GlobalWorldFusion,'attribution_quarantines',
            lambda fusion:records if fusion is actor.world._fusion else actual(fusion))
    elif barrier=='changed-selected-original':
        actual=GlobalWorldFusion.committed_spatial_original
        def changed(fusion,key,**kwargs):
            value=actual(fusion,key,**kwargs)
            if value is not None and fusion is not actor.world._fusion and fusion.physical_entity_id(key)==old.entity_id:
                return (value[0].model_copy(),*value[1:])
            return value
        monkeypatch.setattr(GlobalWorldFusion,'committed_spatial_original',changed)
    elif barrier.startswith('wrong-current-'):
        actual=GlobalWorldFusion.spatial_attribution_claims
        def corrupt(fusion):
            claims=actual(fusion)
            if fusion is not actor.world._fusion:
                update={'frame_id':1262} if barrier=='wrong-current-stamp' else (
                    {'household_identity_key':'r175-foreign','household_stable_id':1099}
                    if barrier=='wrong-current-descriptor' else {'lifecycle_generation':99})
                claims[old.entity_id]=tuple(c.model_copy(update=update)
                    if c.raw_id==23 and c.frame_id==1261 else c for c in claims[old.entity_id])
            return claims
        monkeypatch.setattr(GlobalWorldFusion,'spatial_attribution_claims',corrupt)
    elif barrier=='denied-current':
        monkeypatch.setattr(actor.identity,'prepare_spatial_member',
            lambda *a,**k:SpatialMemberResult('denied','r175-current-grant-denied'))
    before=(actor.world.current_snapshot(),actor.world._revision,len(harness.wire),len(harness.journal))
    with pytest.raises(WorldPublicationRefused):
        candidate=prepare(frame=1261)
        if barrier=='stop':actor.identity.stop_image_region_source(actor.sources[1],graph_id='r175-stop')
        elif barrier=='source-epoch':actor.identity.reset_source(replace(actor.sources[1],source_epoch=1))
        elif barrier=='queued-revision':
            from reid.mv3dt_person_association import SpatialLifecycleReceipt
            previous=next(o for o in actor.originals.values() if o.stamp.member.source_id==2)
            stamp=replace(previous.stamp,frame_id=1262,observation_id='r175-real-later',
                acquired_at_s=previous.stamp.acquired_at_s+.01,media_time_s=126.2)
            actor.identity.observe_spatial(replace(previous,stamp=stamp),SpatialLifecycleReceipt('r175-native',stamp,'ACTIVE'))
        elif barrier=='queued-expiry':actor.us=old.stale_after_us+1
        harness.publish(candidate,1261)
    assert before==(actor.world.current_snapshot(),actor.world._revision,len(harness.wire),len(harness.journal))
    assert actor.identity.stats().pending_reservations==0


def _r173_normal_committed_hold_with_public_clock():
    """Public best-view HELD transaction crosses only its original wire limit.

    Synthetic microseconds/context; real owner, World and publication gate.
    No native owner changes, source advancement or private grant fabrication.
    """
    from tests.test_xc3d_world_physical_person import ImageQuarantineActor,ActualPublicationHarness,COV,CanonicalWorldService
    actor=ImageQuarantineActor();publication_clock=[None]
    actor.world=CanonicalWorldService(producer=actor.world.producer,artifacts=actor.artifacts,
        clock_us=lambda:actor.us if publication_clock[0] is None else publication_clock[0],
        association_service=actor.identity,spatial_bindings=lambda:actor.frames)
    harness=ActualPublicationHarness(actor)
    houses={(1,1):HouseholdDescriptor('visitor','r172-k',1004,1),
            (2,101):HouseholdDescriptor('visitor','r172-f',1000,1)}
    first=harness.publish(actor.prepare_image({2:[(101,1.,1,1)]},households=houses,
        source_changes={2:{'world_covariance':tuple(v*.5 for v in COV)}}),2,source_id=2)
    old,=first.entities
    actor.us+=50_000
    harness.publish(actor.prepare_image({2:[(101,1.01,1,1)]},frame=3,households=houses,
        source_changes={2:{'world_covariance':None}}),3,source_id=2)
    actor.us+=50_000
    publication_clock[0]=old.stale_after_us-1
    candidate=actor.prepare_image({1:[(1,1.1,1,1)]},frame=4,households=houses,
        source_changes={1:{'world_covariance':tuple(v*4 for v in COV)}})
    return actor,harness,old,publication_clock,candidate


def test_r172_normal_committed_hold_wire_expiry_has_typed_exact_reason():
    from noesis_core.world_service import WorldPreparedHoldExpired
    actor,harness,old,publication_clock,candidate=_r173_normal_committed_hold_with_public_clock()
    kept,=candidate.publication.snapshot.entities
    assert kept.position_state=='held' and kept.observed_at_us==old.observed_at_us
    assert kept.stale_after_us-candidate.publication.snapshot.published_at_us==1
    state=actor.world._prepared_candidates[candidate._candidate_id][1]
    assert state.spatial_holds and not state.spatial_split_custody
    before=(actor.world.current_snapshot(),actor.world._revision,len(harness.wire),len(harness.journal))
    publication_clock[0]=old.stale_after_us+1
    with pytest.raises(WorldPreparedHoldExpired) as failure:
        harness.publish(candidate,4)
    assert failure.value.reason=='committed_spatial_hold_custody_changed'
    assert before==(actor.world.current_snapshot(),actor.world._revision,len(harness.wire),len(harness.journal))
    assert actor.identity.stats().pending_reservations==0


@pytest.mark.parametrize('barrier',['stop','source-epoch','owner-revision','native-reuse',
    'foreign-envelope','wrong-stamp','wrong-wire','missing-prepared-custody'])
def test_r173_only_unchanged_original_wire_expiry_is_recoverable(monkeypatch,barrier):
    from noesis_core.world_service import WorldPreparedHoldExpired
    from noesis_core.world import GlobalWorldFusion
    actor,harness,old,clock,candidate=_r173_normal_committed_hold_with_public_clock()
    state=actor.world._prepared_candidates[candidate._candidate_id][1]
    if barrier=='stop':actor.identity.stop_image_region_source(actor.sources[1],graph_id='r173-real-stop')
    elif barrier=='source-epoch':actor.identity.reset_source(replace(actor.sources[1],source_epoch=1))
    elif barrier=='owner-revision':
        from reid.mv3dt_person_association import SpatialLifecycleReceipt
        previous=next(o for o in actor.originals.values() if o.stamp.member.source_id==2)
        stamp=replace(previous.stamp,frame_id=5,observation_id='r173-real-owner-update',
            acquired_at_s=previous.stamp.acquired_at_s+.01,media_time_s=previous.stamp.media_time_s+.01)
        actor.identity.observe_spatial(replace(previous,stamp=stamp),SpatialLifecycleReceipt('r173-current-active',stamp,'ACTIVE'))
    elif barrier=='native-reuse':
        from reid.mv3dt_person_association import SpatialLifecycleReceipt,MemberRemoval
        previous=next(o for o in actor.originals.values() if o.stamp.member.source_id==2)
        removed=actor.identity.withdraw(MemberRemoval('r173-real-termination',previous.stamp,'termination',
            5,previous.stamp.acquired_at_s+.01,previous.stamp.clock_domain,previous.stamp.media_time_s+.01))
        assert removed.disposition=='removed'
        stamp=replace(previous.stamp,member=replace(previous.stamp.member,lifecycle_generation=2),
            frame_id=6,lifecycle_started_frame=6,observation_id='r173-new-native-birth',
            acquired_at_s=previous.stamp.acquired_at_s+.02,media_time_s=previous.stamp.media_time_s+.02)
        observed=actor.identity.observe_spatial(replace(previous,stamp=stamp),SpatialLifecycleReceipt('r173-new-active',stamp,'ACTIVE'))
        assert observed.disposition=='observed'
    else:
        actual=GlobalWorldFusion.committed_spatial_original
        def changed(fusion,key,**kwargs):
            value=actual(fusion,key,**kwargs)
            if value is not None and fusion is state.fusion:
                if barrier=='missing-prepared-custody':return None
                if barrier=='wrong-stamp':return (value[0],replace(value[1],frame_id=value[1].frame_id+1),*value[2:])
                return (value[0].model_copy(update={'observed_at_us':value[0].observed_at_us+1}
                    if barrier=='wrong-wire' else {}),*value[1:])
            return value
        monkeypatch.setattr(GlobalWorldFusion,'committed_spatial_original',changed)
    before=(actor.world.current_snapshot(),actor.world._revision,len(harness.wire),len(harness.journal))
    clock[0]=old.stale_after_us+1
    with pytest.raises(WorldPublicationRefused) as failure:harness.publish(candidate,4)
    assert not isinstance(failure.value,WorldPreparedHoldExpired)
    assert before==(actor.world.current_snapshot(),actor.world._revision,len(harness.wire),len(harness.journal))
    assert actor.identity.stats().pending_reservations==0


def test_r171_target_measured_original_loss_records_exact_barrier_without_changing_decision(monkeypatch):
    from tests.test_xc3d_world_physical_person import SpatialActor,ActualPublicationHarness
    outputs=[]
    for enabled in (False,True):
        monkeypatch.setenv('NOESIS_XC3D_BODY_DIAGNOSTIC_TARGETS_S','.2,.3' if enabled else '')
        actor=SpatialActor();harness=ActualPublicationHarness(actor)
        first=harness.publish(actor.prepare_rows({1:[(1,1.,1,1)]}),2)
        old,=first.entities
        barrier=actor.identity.reset_source(replace(actor.sources[0],source_epoch=1))
        actor.sources=(replace(actor.sources[0],source_epoch=1),actor.sources[1]);actor.us+=100_000
        final=harness.publish(actor.prepare_rows({1:[(2,4.,1,3)]},frame=3,
            withdrawn_members=barrier.withdrawn_members,retired_keys=barrier.retired_keys),3)
        outputs.append((first.model_dump(mode='json'),final.model_dump(mode='json')))
        summary=actor.world._private_spatial_diagnostics()
        targets=[r for r in summary['records'] if r['kind']=='target_cohort_admission']
        assert bool(targets)==enabled
        if enabled:
            members=[v for r in targets for v in json.loads(r['target_members'])['items']]
            assert any(v['raw']==1 and v['eligible'] is True and v['key']==1 for v in members)
            losses=[v for r in targets for v in json.loads(r['target_losses'])['items']]
            loss=next(v for v in losses if v['original_id']==next(s.observation_id for s in old.sources if s.accepted))
            assert loss['prior_state']=='measured' and loss['raw']==1 and loss['birth']==1
            assert loss['deadline_us']==old.stale_after_us and loss['observed_us']==old.observed_at_us
            assert loss['remaining_us']>0 and (loss['withdrawn'] or loss['retired'])
            assert loss['last_executed_omission_branch'] in {'withdrawn_physical_member','retired_physical_key'}
            assert all(r['transaction']=='committed' for r in targets)
            assert len(actor.world._spatial_observer.encoded().encode())<=8192
        assert actor.identity.stats().pending_reservations==0
    assert outputs[0]==outputs[1]


def test_r171_target_observer_nearest_capture_and_final_bytes(monkeypatch):
    from noesis_core.world_service import _SpatialDecisionObserver
    monkeypatch.setenv('NOESIS_XC3D_BODY_DIAGNOSTIC_TARGETS_S','38.4,38.433333333')
    observer=_SpatialDecisionObserver()
    def row(candidate,distance):
        return dict(kind='target_cohort_admission',candidate=candidate,transaction='prepared',source=2,
            method=None,target_index=1,distance_s=distance,frame=1153,
            target_members='x'*4096,target_losses='y'*2048,target_barriers='z'*2000)
    # Oversized legal rows may be omitted, never truncated into invented facts.
    observer.record(row(1,.03));observer.record(row(2,0.));observer.record(row(3,.02))
    assert observer.records[0]['candidate']==2
    view=json.loads(observer.encoded())
    assert len(observer.encoded().encode())<=8192 and view['omitted_records']>=1
    assert len(view['records'])<=8


def _r156_retained_family_claim_transition(*, disjoint_conflict=False):
    """Public owner/World sequence shaped like R71; geometry/time are synthetic."""
    from tests.test_xc3d_world_physical_person import ImageQuarantineActor, ActualPublicationHarness, COV
    actor = ImageQuarantineActor()
    harness = ActualPublicationHarness(actor)
    rows = {1: [(23, 1., 11, 1171), (21, 4., 9, 1116)],
            2: [(15, 1.03, 8, 705)]}
    image_y = {(1, 23): 30., (2, 15): 30., (1, 21): 70.}
    houses = {(1, 23): HouseholdDescriptor('visitor', 'r156-old-k23', 1001, 1),
              (2, 15): HouseholdDescriptor('visitor', 'r156-f15', 1006, 1),
              (1, 21): HouseholdDescriptor('visitor', 'r156-k21-old', 1002, 1)}
    if disjoint_conflict:
        rows[1].extend(((22, 7., 10, 1135), (24, 10., 12, 1240)))
        image_y.update({(1, 22): 110., (1, 24): 150.})
        houses.update({(1, raw): HouseholdDescriptor('visitor', 'r156-disjoint', 1005, 1)
                       for raw in (22, 24)})
    quality = {1: {'world_covariance': tuple(v * 4 for v in COV)},
               2: {'world_covariance': COV}}
    first = harness.publish(actor.prepare_image(rows, frame=1259, households=houses,
        image_y=image_y, source_changes=quality), 1259)
    old = next(e for e in first.entities if e.attribution_quarantine is not None)
    assert old.attribution_quarantine.component_entity_ids == (old.entity_id,)
    assert next(s for s in old.sources if s.accepted).camera_id == 'family-room'
    actor.us += 100_000
    houses = {**houses, (1, 23): HouseholdDescriptor('provisional'),
              (1, 21): HouseholdDescriptor('visitor', 'r156-k21-new', 1001, 1)}
    # Family's better selected ORIGINAL remains fixed. Only the unselected
    # Kitchen alternative and the other genuine person's current claim advance.
    def prepare(*, mode='physical_only_quarantine', advance_family=False, frame=1261):
        return actor.prepare_image(rows if advance_family else {1: rows[1]}, frame=frame, households=houses,
            image_y=image_y, source_changes=quality, mode=mode)
    return actor, harness, first, old, prepare


def test_r156_current_provisional_alternative_preserves_exact_quarantined_original():
    actor, harness, first, old, prepare = _r156_retained_family_claim_transition()
    original_claims = set(old.attribution_quarantine.owners + old.attribution_quarantine.retained_owners)
    candidate = prepare()
    assert all(d.household_conflict is None for d in actor.memberships)
    assert actor.identity.stats().pending_reservations == 1  # Other current person, not the old F point.
    final = harness.publish(candidate, 1261)
    kept = next(e for e in final.entities if e.entity_id == old.entity_id)
    assert (kept.position, kept.covariance, kept.observed_at_us, kept.stale_after_us,
            kept.continuity_segment_id) == (old.position, old.covariance, old.observed_at_us,
                old.stale_after_us, old.continuity_segment_id)
    accepted = next(s for s in kept.sources if s.accepted)
    assert accepted.observation_id == next(s for s in old.sources if s.accepted).observation_id
    q = kept.attribution_quarantine
    assert q is not None and set(q.component_entity_ids) == {e.entity_id for e in final.entities}
    assert all(e.attribution_quarantine == q and e.subject.stable_id is None for e in final.entities)
    assert original_claims <= set(q.owners + q.retained_owners)
    assert any(c.raw_id == 23 and c.frame_id == 1261 and c.household_state == 'provisional' for c in q.owners)
    assert any(c.raw_id == 21 and c.frame_id == 1261 and c.household_stable_id == 1001 for c in q.owners)
    assert actor.identity.stats().pending_reservations == 0
    actual = actual_node_wire(actor, harness.wire, 'r156-retained-quarantine-current-alternative')
    assert [len(s['people']) for s in actual['snapshots']] == [2, 2]
    assert all(d['accepted'] for d in actual['diagnostics'])
    assert all(p['stableId'] is None for p in actual['snapshots'][-1]['people'])


def test_r156_disjoint_current_conflict_never_authorizes_or_enters_retained_component():
    actor, harness, first, old, prepare = _r156_retained_family_claim_transition(disjoint_conflict=True)
    final = harness.publish(prepare(), 1261)
    assert len(first.entities) == len(final.entities) == 4
    kept = next(e for e in final.entities if e.entity_id == old.entity_id)
    q = kept.attribution_quarantine
    assert q.stable_ids == (1001,) and len(q.component_entity_ids) == 2
    unrelated = [e for e in final.entities if e.entity_id not in q.component_entity_ids]
    assert len(unrelated) == 2
    assert all(e.attribution_quarantine.stable_ids == (1005,) for e in unrelated)
    assert not set(q.owners + q.retained_owners) & set(unrelated[0].attribution_quarantine.owners)
    actual = actual_node_wire(actor, harness.wire, 'r156-disjoint-component')
    assert [len(s['people']) for s in actual['snapshots']] == [4, 4]


@pytest.mark.parametrize('defect', ['missing-custody', 'missing-quarantine', 'uncovered-original-claim',
    'foreign-current-claim', 'changed-selected-original', 'grant-denied', 'strict-mode'])
def test_r156_attenuation_requires_exact_selected_custody_and_current_claims(monkeypatch, defect):
    from noesis_core.world import GlobalWorldFusion
    actor, harness, first, old, prepare = _r156_retained_family_claim_transition()
    if defect == 'missing-custody':
        actual = GlobalWorldFusion.spatial_original_attribution_claims
        monkeypatch.setattr(GlobalWorldFusion, 'spatial_original_attribution_claims',
            lambda f: {} if f is actor.world._fusion else actual(f))
    elif defect == 'missing-quarantine':
        actor.world._fusion.set_attribution_quarantines({})
    elif defect == 'uncovered-original-claim':
        actual = GlobalWorldFusion.spatial_original_attribution_claims
        def uncovered(fusion):
            claims = actual(fusion)
            if fusion is actor.world._fusion:
                claims[old.entity_id] += (claims[old.entity_id][0].model_copy(update={'source_epoch': 99}),)
            return claims
        monkeypatch.setattr(GlobalWorldFusion, 'spatial_original_attribution_claims', uncovered)
    elif defect == 'foreign-current-claim':
        actual = GlobalWorldFusion.spatial_attribution_claims
        def corrupt(fusion):
            claims = actual(fusion)
            if fusion is not actor.world._fusion:
                claims[old.entity_id] = tuple(c.model_copy(update={'source_epoch': 99})
                    if c.raw_id == 23 and c.frame_id == 1261 else c for c in claims[old.entity_id])
            return claims
        monkeypatch.setattr(GlobalWorldFusion, 'spatial_attribution_claims', corrupt)
    elif defect == 'changed-selected-original':
        actual = GlobalWorldFusion.committed_spatial_original
        def changed(fusion, key, **kwargs):
            value = actual(fusion, key, **kwargs)
            if fusion is not actor.world._fusion and fusion.physical_entity_id(key) == old.entity_id:
                return (value[0].model_copy(), *value[1:])
            return value
        monkeypatch.setattr(GlobalWorldFusion, 'committed_spatial_original', changed)
    elif defect == 'grant-denied':
        from noesis.identity_association_service import SpatialMemberResult
        monkeypatch.setattr(actor.identity, 'prepare_spatial_member',
            lambda *a, **k: SpatialMemberResult('denied', 'r156-current-authority-denied'))
    before = (len(harness.wire), len(harness.journal), actor.world._revision)
    with pytest.raises(WorldPublicationRefused):
        prepare(mode='strict' if defect == 'strict-mode' else 'physical_only_quarantine')
    assert actor.world.current_snapshot() is first
    assert before == (len(harness.wire), len(harness.journal), actor.world._revision)
    assert actor.identity.stats().pending_reservations == 0


@pytest.mark.parametrize('barrier', ['queued-revision', 'source-stop'])
def test_r156_label_only_projection_revalidates_owner_at_actual_commit(barrier):
    actor, harness, first, old, prepare = _r156_retained_family_claim_transition()
    candidate = prepare()
    before = (len(harness.wire), len(harness.journal), actor.world._revision)
    if barrier == 'queued-revision':
        harness.before_commit = lambda: actor.identity.reset_source(replace(actor.sources[0], source_epoch=1))
    else:
        harness.before_commit = lambda: actor.identity.stop_image_region_source(actor.sources[1], graph_id='r156-stop')
    with pytest.raises(WorldPublicationRefused):
        harness.publish(candidate, 1261)
    assert actor.world.current_snapshot() is first
    assert before == (len(harness.wire), len(harness.journal), actor.world._revision)
    assert actor.identity.stats().pending_reservations == 0


def test_r156_attenuated_original_still_expires_without_any_clock_renewal():
    from tests.test_xc3d_world_physical_person import _r40_empty_spatial
    actor, harness, first, old, prepare = _r156_retained_family_claim_transition()
    final = harness.publish(prepare(), 1261)
    kept = next(e for e in final.entities if e.entity_id == old.entity_id)
    assert kept.stale_after_us == old.stale_after_us
    actor.us = max(e.stale_after_us for e in final.entities) + 1
    expired = harness.publish(_r40_empty_spatial(actor, frame=1262,
        evaluation_s=actor.us / 1_000_000), 1262)
    assert not expired.entities
    actual = actual_node_wire(actor, harness.wire, 'r156-original-expiry')
    assert [len(s['people']) for s in actual['snapshots']] == [2, 2, 0]


def test_r156_genuine_fresh_replacement_ends_original_label_only_quarantine():
    actor, harness, first, old, prepare = _r156_retained_family_claim_transition()
    held = harness.publish(prepare(), 1261)
    actor.us += 100_000
    fresh = harness.publish(prepare(advance_family=True, frame=1262), 1262)
    assert len(fresh.entities) == 2 and all(e.attribution_quarantine is None for e in fresh.entities)
    assert next(e for e in fresh.entities if e.entity_id == old.entity_id).observed_at_us > old.observed_at_us
    assert sorted(e.subject.stable_id for e in fresh.entities) == [1001, 1006]
    actual = actual_node_wire(actor, harness.wire, 'r156-genuine-fresh-replacement')
    assert [len(s['people']) for s in actual['snapshots']] == [2, 2, 2]
    assert all(p['stableId'] is None for p in actual['snapshots'][1]['people'])


def _r160_clean_retained_label_transition(age_us):
    from tests.test_xc3d_world_physical_person import ImageQuarantineActor, ActualPublicationHarness
    actor = ImageQuarantineActor()
    harness = ActualPublicationHarness(actor)
    first = harness.publish(actor.prepare_image({1: [(30, 1., 29, 10953)]}, frame=11054,
        households={(1, 30): HouseholdDescriptor('visitor', 'r160-old', 1001, 1)}), 11054)
    old, = first.entities
    actor.us += age_us
    original = actor.world.prepare
    def unavailable(source, tracks, **kwargs):
        return original(source, [dict(t, world_valid=False, world_measurement_accepted=False,
            world_covariance=None) for t in tracks], **kwargs)
    actor.world.prepare = unavailable
    def prepare():
        return actor.prepare_image({1: [(30, 1., 29, 10953), (31, 4., 30, 11067)]}, frame=11067,
            households={(1, 30): HouseholdDescriptor('provisional'),
                        (1, 31): HouseholdDescriptor('visitor', 'r160-current', 1001, 1)})
    return actor, harness, first, old, prepare


@pytest.mark.parametrize('age_us', [100_000, 467_522])
def test_r160_attribution_uses_committed_original_ttl_not_current_vote_cutoff(age_us):
    actor, harness, first, old, prepare = _r160_clean_retained_label_transition(age_us)
    candidate = prepare()
    assert all(d.household_conflict is None for d in actor.memberships)
    assert actor.identity.stats().pending_reservations == 0
    final = harness.publish(candidate, 11067)
    kept, = final.entities
    assert (kept.entity_id, kept.position, kept.covariance, kept.observed_at_us,
        kept.stale_after_us, kept.sources, kept.continuity_segment_id) == (old.entity_id,
            old.position, old.covariance, old.observed_at_us, old.stale_after_us,
            old.sources, old.continuity_segment_id)
    assert kept.attribution_quarantine is not None and kept.subject.stable_id is None
    assert any(c.raw_id == 30 and c.frame_id == 11054 and c.household_stable_id == 1001
        for c in kept.attribution_quarantine.owners)
    assert any(c.raw_id == 30 and c.frame_id == 11067 and c.household_state == 'provisional'
        for c in kept.attribution_quarantine.owners)
    actual = actual_node_wire(actor, harness.wire, 'r160-original-age-' + str(age_us))
    assert [len(s['people']) for s in actual['snapshots']] == [1, 1]
    assert all(d['accepted'] for d in actual['diagnostics'])


@pytest.mark.parametrize('barrier', ['queued-expiry', 'stop', 'source-epoch', 'reuse',
    'wrong-envelope', 'missing-custody', 'foreign-claims'])
def test_r160_finite_attribution_hold_cannot_bypass_original_or_owner_guards(monkeypatch, barrier):
    from noesis_core.world import GlobalWorldFusion
    actor, harness, first, old, prepare = _r160_clean_retained_label_transition(467_522)
    before = (len(harness.wire), len(harness.journal), actor.world._revision)
    if barrier in {'wrong-envelope', 'missing-custody'}:
        actual = GlobalWorldFusion.committed_spatial_original
        def corrupted(fusion, key, **kwargs):
            value = actual(fusion, key, **kwargs)
            if fusion is not actor.world._fusion and value is not None:
                return None if barrier == 'missing-custody' else (value[0].model_copy(), *value[1:])
            return value
        monkeypatch.setattr(GlobalWorldFusion, 'committed_spatial_original', corrupted)
    elif barrier == 'foreign-claims':
        actual = GlobalWorldFusion.spatial_attribution_claims
        def foreign(fusion):
            claims = actual(fusion)
            if fusion is not actor.world._fusion:
                claims[old.entity_id] = tuple(c.model_copy(update={'source_epoch': 99}) for c in claims[old.entity_id])
            return claims
        monkeypatch.setattr(GlobalWorldFusion, 'spatial_attribution_claims', foreign)
    if barrier in {'wrong-envelope', 'missing-custody', 'foreign-claims'}:
        with pytest.raises(WorldPublicationRefused): prepare()
    else:
        candidate = prepare()
        if barrier == 'queued-expiry':
            actor.us = old.stale_after_us + 1
        elif barrier == 'stop':
            harness.before_commit = lambda: actor.identity.stop_image_region_source(actor.sources[0], graph_id='r160-stop')
        elif barrier == 'reuse':
            from tests.test_xc3d_world_physical_person import MemberRemoval, SpatialLifecycleReceipt
            def reuse():
                original = next(o for o in actor.originals.values() if o.stamp.member.raw_id == 30)
                stamp = original.stamp
                actor.identity.withdraw(MemberRemoval('r160-native-end', stamp, 'termination',
                    stamp.frame_id + 1, stamp.acquired_at_s + .1, stamp.clock_domain, stamp.media_time_s + .1))
                replacement = replace(original, stamp=replace(stamp,
                    member=replace(stamp.member, lifecycle_generation=31),
                    lifecycle_started_frame=stamp.frame_id + 2, frame_id=stamp.frame_id + 2,
                    observation_id='r160-reused-original', acquired_at_s=stamp.acquired_at_s + .2,
                    media_time_s=stamp.media_time_s + .2))
                result = actor.identity.observe_spatial(replacement,
                    SpatialLifecycleReceipt('r160-reused-active', replacement.stamp, 'ACTIVE'))
                assert result.disposition in {'observed', 'unresolved'}
            harness.before_commit = reuse
        else:
            harness.before_commit = lambda: actor.identity.reset_source(replace(actor.sources[0], source_epoch=1))
        with pytest.raises(WorldPublicationRefused): harness.publish(candidate, 11067)
    assert actor.world.current_snapshot() is first
    assert before == (len(harness.wire), len(harness.journal), actor.world._revision)
    assert actor.identity.stats().pending_reservations == 0


def test_r160_already_expired_original_is_not_a_label_projection_anchor():
    actor, harness, first, old, prepare = _r160_clean_retained_label_transition(2_100_001)
    final = harness.publish(prepare(), 11067)
    assert not final.entities and actor.identity.stats().pending_reservations == 0
    actual = actual_node_wire(actor, harness.wire, 'r160-original-already-expired')
    assert [len(s['people']) for s in actual['snapshots']] == [1, 0]


def _r162_advancing_geometry_with_retained_claims():
    from tests.test_xc3d_world_physical_person import ImageQuarantineActor, ActualPublicationHarness, COV
    actor = ImageQuarantineActor()
    harness = ActualPublicationHarness(actor)
    rows = {1: [(20, 1., 9, 1116), (22, 4., 11, 1171)], 2: [(3, 1.03, 2, 2)]}
    image_y = {(1, 20): 30., (2, 3): 30., (1, 22): 70.}
    houses = {(1, 20): HouseholdDescriptor('visitor', 'r162-K', 1003, 1),
              (2, 3): HouseholdDescriptor('visitor', 'r162-F', 1004, 1),
              (1, 22): HouseholdDescriptor('visitor', 'r162-other-old', 1002, 1)}
    quality = {1: {'world_covariance': COV}, 2: {'world_covariance': tuple(v * 4 for v in COV)}}
    first = harness.publish(actor.prepare_image(rows, frame=1210, households=houses,
        image_y=image_y, source_changes=quality), 1210)
    old = next(e for e in first.entities if e.attribution_quarantine is not None)
    assert next(s for s in old.sources if s.accepted).camera_id == 'kitchen'
    actor.us += 100_000
    houses = {**houses, (1, 20): HouseholdDescriptor('provisional'),
              (1, 22): HouseholdDescriptor('visitor', 'r162-other-current', 1003, 1)}
    def prepare(*, mode='physical_only_quarantine'):
        return actor.prepare_image({1: [(20, 1.25, 9, 1116), (22, 4.1, 11, 1171)]},
            frame=1211, households=houses, image_y=image_y, source_changes=quality, mode=mode)
    return actor, harness, first, old, prepare


def test_r162_prior_quarantine_proves_claims_not_a_requirement_to_hold_position():
    actor, harness, first, old, prepare = _r162_advancing_geometry_with_retained_claims()
    original_claims = set(actor.world._fusion.spatial_original_attribution_claims()[old.entity_id])
    candidate = prepare()
    assert all(d.household_conflict is None for d in actor.memberships)
    assert actor.identity.stats().pending_reservations == 2  # BOTH genuine current positions require grants.
    final = harness.publish(candidate, 1211)
    advanced = next(e for e in final.entities if e.entity_id == old.entity_id)
    assert advanced.position != old.position and advanced.observed_at_us == actor.us > old.observed_at_us
    assert advanced.stale_after_us > old.stale_after_us
    accepted = next(s for s in advanced.sources if s.accepted)
    assert accepted.observation_id != next(s for s in old.sources if s.accepted).observation_id
    assert accepted.observation_id == 'synthetic-native:1:20:1211'
    assert advanced.position_state == 'measured'
    q = advanced.attribution_quarantine
    assert q is not None and original_claims <= set(q.owners + q.retained_owners)
    assert all(e.attribution_quarantine == q and e.subject.stable_id is None for e in final.entities)
    assert any(c.raw_id == 20 and c.frame_id == 1210 and c.household_stable_id == 1003
        for c in q.owners + q.retained_owners)
    assert any(c.raw_id == 20 and c.frame_id == 1211 and c.household_state == 'provisional'
        for c in q.owners + q.retained_owners)
    assert actor.identity.stats().pending_reservations == 0
    actual = actual_node_wire(actor, harness.wire, 'r162-advancing-physical-quarantined-attribution')
    assert [len(s['people']) for s in actual['snapshots']] == [2, 2]
    assert all(d['accepted'] for d in actual['diagnostics'])


@pytest.mark.parametrize('defect', ['missing-prior', 'forged-prior', 'foreign-original-claim',
    'denied-current-grant', 'mismatched-frame', 'strict-mode'])
def test_r162_old_quarantine_never_substitutes_for_current_authority(monkeypatch, defect):
    from noesis_core.world import GlobalWorldFusion
    from noesis_core.contracts.world import WorldAttributionQuarantine
    from noesis.identity_association_service import SpatialMemberResult
    actor, harness, first, old, prepare = _r162_advancing_geometry_with_retained_claims()
    if defect == 'missing-prior':
        actor.world._fusion.set_attribution_quarantines({})
    elif defect == 'forged-prior':
        payload = old.attribution_quarantine.model_dump(mode='json', exclude={'conflict_id'})
        payload['association_revision'] += 1
        forged = WorldAttributionQuarantine(conflict_id=WorldAttributionQuarantine.digest(payload), **payload)
        actor.world._fusion.set_attribution_quarantines({old.entity_id: forged})
    elif defect == 'foreign-original-claim':
        actual = GlobalWorldFusion.spatial_original_attribution_claims
        def foreign(fusion):
            values = actual(fusion)
            if fusion is actor.world._fusion:
                values[old.entity_id] += (values[old.entity_id][0].model_copy(update={'source_epoch': 99}),)
            return values
        monkeypatch.setattr(GlobalWorldFusion, 'spatial_original_attribution_claims', foreign)
    elif defect == 'denied-current-grant':
        monkeypatch.setattr(actor.identity, 'prepare_spatial_member',
            lambda *a, **k: SpatialMemberResult('denied', 'r162-current-grant-denied'))
    elif defect == 'mismatched-frame':
        actual = actor.world.prepare
        def mismatched(source, tracks, **kwargs):
            kwargs['physical_observations'] = tuple(replace(o, stamp=replace(o.stamp,
                frame_id=o.stamp.frame_id + 1)) for o in kwargs['physical_observations'])
            return actual(source, tracks, **kwargs)
        monkeypatch.setattr(actor.world, 'prepare', mismatched)
    before = (len(harness.wire), len(harness.journal), actor.world._revision)
    expected = ValueError if defect == 'mismatched-frame' else WorldPublicationRefused
    with pytest.raises(expected, match='original native observation lacks its exact normalized row'
                       if defect == 'mismatched-frame' else None):
        prepare(mode='strict' if defect == 'strict-mode' else 'physical_only_quarantine')
    assert actor.world.current_snapshot() is first
    assert before == (len(harness.wire), len(harness.journal), actor.world._revision)
    assert actor.identity.stats().pending_reservations == 0


@pytest.mark.parametrize('barrier', ['stop', 'source-epoch', 'queued-expiry'])
def test_r162_fresh_reference_is_revalidated_after_label_projection(barrier):
    actor, harness, first, old, prepare = _r162_advancing_geometry_with_retained_claims()
    candidate = prepare()
    before = (len(harness.wire), len(harness.journal), actor.world._revision)
    if barrier == 'stop':
        harness.before_commit = lambda: actor.identity.stop_image_region_source(actor.sources[0], graph_id='r162-stop')
    elif barrier == 'source-epoch':
        harness.before_commit = lambda: actor.identity.reset_source(replace(actor.sources[0], source_epoch=1))
    else:
        actor.us = max(e.stale_after_us for e in candidate.publication.snapshot.entities) + 1
        actor.context = actor.clock_context()  # Truthful delayed source evaluation, not the frozen prepare time.
    with pytest.raises(WorldPublicationRefused):
        harness.publish(candidate, 1211)
    assert actor.world.current_snapshot() is first
    assert before == (len(harness.wire), len(harness.journal), actor.world._revision)
    assert actor.identity.stats().pending_reservations == 0


def _r166_clean_original_with_new_current_label():
    from tests.test_xc3d_world_physical_person import ImageQuarantineActor, ActualPublicationHarness, COV
    from reid.mv3dt_person_association import MemberRemoval
    actor=ImageQuarantineActor();harness=ActualPublicationHarness(actor)
    rows={1:[(0,1.,1,2)],2:[(3,1.03,3,2)]}
    image_y={(1,0):30.,(2,3):30.}
    houses={(1,0):HouseholdDescriptor('visitor','r166-old-K',1002,1),
            (2,3):HouseholdDescriptor('provisional')}
    quality={1:{'world_covariance':tuple(v*4 for v in COV)},2:{'world_covariance':COV}}
    first=harness.publish(actor.prepare_image(rows,frame=300,households=houses,
        image_y=image_y,source_changes=quality),300)
    old,=first.entities
    assert old.attribution_quarantine is None
    assert [s.camera_id for s in old.sources if s.accepted]==['family-room']
    claims=actor.world._fusion.spatial_original_attribution_claims()[old.entity_id]
    peer=next(o for o in actor.originals.values() if o.stamp.member.source_id==1)
    actor.us+=100_000
    removed=actor.identity.withdraw(MemberRemoval('r166-actual-peer-termination',peer.stamp,
        'termination',301,actor.us/1e6,peer.stamp.clock_domain,30.1))
    assert removed.disposition=='removed'
    actor.image_originals.pop(peer.stamp.member);actor.originals.pop(peer.stamp.member)
    houses[(2,3)]=HouseholdDescriptor('visitor','r166-new-F',1001,1)
    def prepare(mode='physical_only_quarantine'):
        return actor.prepare_image({2:rows[2]},frame=307,households=houses,image_y=image_y,
            source_changes={2:{'world_covariance':None}},withdrawn_members=removed.withdrawn_members,mode=mode)
    return actor,harness,first,old,claims,prepare


def test_r166_clean_committed_original_can_quarantine_new_label_without_a_new_point():
    actor,harness,first,old,claims,prepare=_r166_clean_original_with_new_current_label()
    candidate=prepare()
    assert all(d.household_conflict is None for d in actor.memberships)
    assert actor.identity.stats().pending_reservations==0
    final=harness.publish(candidate,307)
    kept,=final.entities
    assert (kept.position,kept.covariance,kept.observed_at_us,kept.stale_after_us,
            kept.continuity_segment_id)==(old.position,old.covariance,old.observed_at_us,
                old.stale_after_us,old.continuity_segment_id)
    assert kept.subject.stable_id is None and kept.attribution_quarantine is not None
    q=kept.attribution_quarantine
    assert set(claims)<=set(q.owners+q.retained_owners) and q.stable_ids==(1001,1002)
    assert any(c.source_id==2 and c.frame_id==307 and c.household_stable_id==1001 for c in q.owners)
    assert next(s for s in kept.sources if s.accepted).observation_id==next(s for s in old.sources if s.accepted).observation_id
    actual=actual_node_wire(actor,harness.wire,'r166-clean-original-new-current-label')
    assert [len(s['people']) for s in actual['snapshots']]==[1,1]
    assert all(d['accepted'] for d in actual['diagnostics'])
    actor.us=old.stale_after_us+1
    from tests.test_xc3d_world_physical_person import _r40_empty_spatial
    expired=harness.publish(_r40_empty_spatial(actor,frame=308,evaluation_s=12.3),308)
    assert not expired.entities


@pytest.mark.parametrize('barrier',['missing-custody','wrong-envelope','foreign-claims','strict-mode',
    'queued-expiry','stop','source-epoch','queued-revision'])
def test_r166_clean_original_projection_preserves_custody_and_current_owner_guards(monkeypatch,barrier):
    from noesis_core.world import GlobalWorldFusion
    actor,harness,first,old,claims,prepare=_r166_clean_original_with_new_current_label()
    before=(len(harness.wire),len(harness.journal),actor.world._revision)
    if barrier in {'missing-custody','wrong-envelope'}:
        actual=GlobalWorldFusion.committed_spatial_original
        def changed(fusion,key,**kwargs):
            value=actual(fusion,key,**kwargs)
            if value is not None and fusion is not actor.world._fusion:
                return None if barrier=='missing-custody' else (value[0].model_copy(),*value[1:])
            return value
        monkeypatch.setattr(GlobalWorldFusion,'committed_spatial_original',changed)
    elif barrier=='foreign-claims':
        actual=GlobalWorldFusion.spatial_attribution_claims
        def changed(fusion):
            values=actual(fusion)
            if fusion is not actor.world._fusion:
                values[old.entity_id]=tuple(c.model_copy(update={'source_epoch':99}) for c in values[old.entity_id])
            return values
        monkeypatch.setattr(GlobalWorldFusion,'spatial_attribution_claims',changed)
    if barrier in {'missing-custody','wrong-envelope','foreign-claims','strict-mode'}:
        with pytest.raises(WorldPublicationRefused):prepare('strict' if barrier=='strict-mode' else 'physical_only_quarantine')
    else:
        candidate=prepare()
        if barrier=='queued-expiry':actor.us=old.stale_after_us+1
        elif barrier=='stop':
            harness.before_commit=lambda:actor.identity.stop_image_region_source(actor.sources[1],graph_id='r166-stop')
        elif barrier=='source-epoch':
            harness.before_commit=lambda:actor.identity.reset_source(replace(actor.sources[1],source_epoch=1))
        else:
            from reid.mv3dt_person_association import SpatialLifecycleReceipt
            def later():
                original=next(o for o in actor.originals.values() if o.stamp.member.source_id==2)
                stamp=replace(original.stamp,frame_id=308,observation_id='r166-genuine-later',
                    acquired_at_s=original.stamp.acquired_at_s+.01,media_time_s=30.8)
                result=actor.identity.observe_spatial(replace(original,stamp=stamp),
                    SpatialLifecycleReceipt('r166-later-active',stamp,'ACTIVE'))
                assert result.disposition=='observed'
            harness.before_commit=later
        with pytest.raises(WorldPublicationRefused):harness.publish(candidate,307)
    assert actor.world.current_snapshot() is first
    assert before==(len(harness.wire),len(harness.journal),actor.world._revision)
    assert actor.identity.stats().pending_reservations==0


def _r163_split_hold_with_public_clock(monkeypatch):
    from tests.test_xc3d_world_physical_person import CanonicalWorldService, ActualPublicationHarness
    actor, _, owner_clock = _r101_actor(monkeypatch)
    class PublicationClock:
        before = after = None
        calls = 0
        def __call__(self):
            if self.before is None:
                return actor.us
            self.calls += 1
            # Normalization, custody capture, attribution, then snapshot. The
            # public clock crosses expiry between capture and snapshot only.
            return self.before if self.calls <= 3 else max(actor.us, self.after)
    clock = PublicationClock()
    actor.world = CanonicalWorldService(producer=actor.world.producer, artifacts=actor.artifacts,
        clock_us=clock, association_service=actor.identity, spatial_bindings=lambda: actor.frames)
    harness = ActualPublicationHarness(actor)
    first = harness.publish(actor.prepare_image(actor.rows, frame=2, semantic_joints=actor.joints,
        source_changes={1: {'world_covariance': None}}), 2)
    old, = first.entities
    actor.us += 100_000
    changed = dict(actor.joints)
    changed[(2, 101)] = tuple((i, u, v + 60., c) for i, u, v, c in changed[(2, 101)])
    held = harness.publish(actor.prepare_image(actor.rows, frame=3, semantic_joints=changed,
        image_y={(2, 101): 90.}, source_changes={1: {'world_covariance': None},
            2: {'world_covariance': None}}), 3)
    assert len(held.entities) == 1 and held.entities[0].observed_at_us == old.observed_at_us
    return actor, harness, clock, owner_clock, old, held


def test_r163_expired_hold_omitted_by_snapshot_is_not_a_commit_dependency(monkeypatch):
    from tests.test_xc3d_world_physical_person import _r40_empty_spatial
    actor, harness, clock, _, old, held = _r163_split_hold_with_public_clock(monkeypatch)
    actor.us = clock.before = old.stale_after_us - 1
    clock.after = old.stale_after_us + 1
    candidate = _r40_empty_spatial(actor, frame=4, evaluation_s=10.4)
    assert clock.calls == 4 and not candidate.publication.snapshot.entities
    assert candidate.publication.snapshot.published_at_us > old.stale_after_us
    assert actor.world.current_snapshot() is held  # Prepare is still private.
    final = harness.publish(candidate, 4)
    assert not final.entities and actor.identity.stats().pending_reservations == 0
    assert not actor.world._fusion.split_custody_originals()
    actual = actual_node_wire(actor, harness.wire, 'r163-expired-before-outbound-admission')
    assert [len(s['people']) for s in actual['snapshots']] == [1, 1, 0]
    assert all(d['accepted'] for d in actual['diagnostics'])


def test_r163_hold_in_admitted_snapshot_must_still_refuse_later_expiry(monkeypatch):
    from tests.test_xc3d_world_physical_person import _r40_empty_spatial
    actor, harness, clock, _, old, held = _r163_split_hold_with_public_clock(monkeypatch)
    actor.us = old.stale_after_us - 1
    candidate = _r40_empty_spatial(actor, frame=4, evaluation_s=10.4)
    assert len(candidate.publication.snapshot.entities) == 1
    before = (len(harness.wire), len(harness.journal), actor.world._revision)
    actor.us = old.stale_after_us + 1
    from noesis_core.world_service import WorldPreparedHoldExpired
    with pytest.raises(WorldPreparedHoldExpired, match='split_custody_original_changed') as expired:
        harness.publish(candidate, 4)
    assert expired.value.prepared is candidate
    assert actor.world.current_snapshot() is held
    assert before == (len(harness.wire), len(harness.journal), actor.world._revision)


@pytest.mark.parametrize('barrier', ['stop', 'source-epoch', 'owner-expiry'])
def test_r163_omitted_old_hold_does_not_bypass_current_owner_validation(monkeypatch, barrier):
    from tests.test_xc3d_world_physical_person import _r40_empty_spatial
    actor, harness, clock, owner_clock, old, held = _r163_split_hold_with_public_clock(monkeypatch)
    actor.us = clock.before = old.stale_after_us - 1
    clock.after = old.stale_after_us + 1
    candidate = _r40_empty_spatial(actor, frame=4, evaluation_s=10.4)
    assert not candidate.publication.snapshot.entities
    if barrier == 'stop':
        harness.before_commit = lambda: actor.identity.stop_image_region_source(actor.sources[1], graph_id='r101-graph')
    elif barrier == 'source-epoch':
        harness.before_commit = lambda: actor.identity.reset_source(replace(actor.sources[1], source_epoch=1))
    else:
        owner_clock.monotonic = lambda: 121.
    before = (len(harness.wire), len(harness.journal), actor.world._revision)
    with pytest.raises(WorldPublicationRefused): harness.publish(candidate, 4)
    assert actor.world.current_snapshot() is held
    assert before == (len(harness.wire), len(harness.journal), actor.world._revision)


def test_r168_expiry_does_not_classify_changed_original_as_retryable(monkeypatch):
    from tests.test_xc3d_world_physical_person import _r40_empty_spatial
    from noesis_core.world import GlobalWorldFusion
    from noesis_core.world_service import WorldPreparedHoldExpired
    actor,harness,clock,owner_clock,old,held=_r163_split_hold_with_public_clock(monkeypatch)
    actor.us=old.stale_after_us-1
    candidate=_r40_empty_spatial(actor,frame=4,evaluation_s=10.4)
    original=GlobalWorldFusion.capture_split_custody
    def changed(fusion,*args,**kwargs):
        value=original(fusion,*args,**kwargs)
        if value is not None:
            return replace(value,row=replace(value.row,envelope=value.row.envelope.model_copy(
                update={'observation_id':'r168-foreign-original'})))
        return value
    monkeypatch.setattr(GlobalWorldFusion,'capture_split_custody',changed)
    actor.us=old.stale_after_us+1
    with pytest.raises(WorldPublicationRefused) as rejected:harness.publish(candidate,4)
    assert not isinstance(rejected.value,WorldPreparedHoldExpired)
    assert actor.world.current_snapshot() is held and actor.identity.stats().pending_reservations==0


def _i136_nonvote(frame=162, **changes):
    return dict(dict(kind='current_original_nonvote', candidate=frame, transaction='prepared',
        source=2, method='registered_depth', frame=frame, raw=8, generation=5,
        birth=158, epoch=0, original_id='current:'+str(frame),
        owner_eligible=False), **changes)


def test_i136_first_nonvote_survives_later_same_lineage_rejection():
    from noesis_core.world_service import _SpatialDecisionObserver
    observer = _SpatialDecisionObserver()
    first = _i136_nonvote()
    observer.record(first)
    observer.outcome(162, 'committed')
    observer.record(_i136_nonvote(186))
    observer.outcome(186, 'committed')
    row, = json.loads(observer.encoded())['records']
    assert row['frame'] == 162 and row['transaction'] == 'committed'
    first['frame'] = 999
    row['frame'] = 998
    assert observer.records[0]['frame'] == 162


@pytest.mark.parametrize('failure', [dict(raw_entry_eligible=False,owner_eligible=True),
    dict(owner_eligible=False),dict(owner_eligible=True,selector_reason='resolver_covariance_absent'),
    dict(owner_eligible=True,reservation_outcome='reserved',final_source_accepted=False)])
def test_i136_healthy_alternative_cannot_hide_first_genuine_denial(failure):
    from noesis_core.world_service import _SpatialDecisionObserver
    observer = _SpatialDecisionObserver()
    observer.record(_i136_nonvote(150, owner_eligible=True,
        raw_entry_eligible=True, selector_reason='source_local_alternative'))
    observer.outcome(150,'committed')
    observer.record(_i136_nonvote(162, **failure))
    observer.outcome(162,'committed')
    observer.record(_i136_nonvote(186, owner_eligible=False))
    row, = observer.records
    assert row['frame']==162 and row['transaction']=='committed'


@pytest.mark.parametrize('field,value', [('source',1),('method','floor_ray'),('raw',9),
    ('generation',6),('birth',159),('epoch',1)])
def test_i136_new_lineage_or_method_does_not_inherit_old_denial(field,value):
    from noesis_core.world_service import _SpatialDecisionObserver
    observer = _SpatialDecisionObserver()
    observer.record(_i136_nonvote())
    observer.outcome(162, 'committed')
    observer.record(_i136_nonvote(186, **{field:value}))
    assert any(row['frame']==186 for row in observer.records)


def test_i136_discarded_attempt_does_not_hide_first_actual_rejection():
    from noesis_core.world_service import _SpatialDecisionObserver
    observer = _SpatialDecisionObserver()
    observer.record(_i136_nonvote())
    observer.outcome(162, 'discarded')
    observer.record(_i136_nonvote(186))
    observer.outcome(186, 'committed')
    row, = observer.records
    assert row['frame']==186 and row['transaction']=='committed'


@pytest.mark.parametrize('accepted,valid', [(False,False),(False,True),(None,None)])
def test_i136_raw_entry_predicate_observed_without_grant(accepted,valid):
    from tests.test_xc3d_world_physical_person import SpatialActor
    actor = SpatialActor()
    if accepted is None and valid is None:
        with pytest.raises(WorldPublicationRefused, match='measurement_state_unavailable'):
            actor.prepare_rows({1:[(1,1.,1,1)]},source_changes={1:{
                'world_measurement_accepted':accepted,'world_valid':valid,
                'world_resolver_selected_id':'registered_depth'}})
        assert actor.world.current_snapshot() is None
        assert not actor.world._private_spatial_diagnostics()['records']
        return  # Earlier integrity refusal stays earlier; never a diagnostic grant.
    candidate = actor.prepare_rows({1:[(1,1.,1,1)]},source_changes={1:{
        'world_measurement_accepted':accepted,'world_valid':valid,
        'world_resolver_selected_id':'registered_depth'}})
    assert not candidate.publication.snapshot.entities
    row, = actor.world._private_spatial_diagnostics()['records']
    assert row['raw_accepted'] is accepted and row['raw_world_valid'] is valid
    assert row['raw_entry_eligible'] is False
    assert row['selector_evaluated'] is True
    assert row['transaction']=='prepared' and row['final_source_accepted'] is None
    actor.world.discard(candidate)
    assert actor.world._private_spatial_diagnostics()['records'][0]['transaction']=='discarded'


def test_i136_observer_bounds_malformed_detachment_and_first_loss_pin():
    from noesis_core.world_service import _SpatialDecisionObserver
    observer = _SpatialDecisionObserver()
    observer.record(dict(kind='unexpired_held_loss', candidate=1, transaction='committed',
        source='family-room', method=None, prior_owner_guard='x'*2048))
    for source in range(30):
        observer.record(_i136_nonvote(source=source, raw=source,
            final_reason='r'*240, owner_reason='o'*240))
    assert len(observer.records)==8
    view=json.loads(observer.encoded())
    assert len(view['records'])<=8 and len(observer.encoded().encode())<=8192
    assert any(r['kind']=='unexpired_held_loss' for r in view['records'])
    omitted=observer.omitted
    for bad in ({'kind':'bad','nonfinite':float('nan')}, {'kind':'bad','sdk':object()},
                {'kind':'bad','nested':[]}, {'kind':'bad','large':'x'*241}):
        observer.record(bad)
    assert observer.omitted==omitted+4
    view['records'].clear()
    assert len(observer.records)==8


def test_i136_same_candidate_two_losses_pin_first_but_discard_is_replaceable():
    from noesis_core.world_service import _SpatialDecisionObserver
    observer = _SpatialDecisionObserver()
    first = dict(kind='unexpired_held_loss', candidate=127, transaction='prepared',
        source='family-room', method=None, original_id='cap-F94')
    observer.record(first)
    observer.record({**first, 'original_id':'child-F94'})
    assert observer.records[0]['original_id']=='cap-F94'
    observer.outcome(127,'discarded')
    observer.record({**first, 'candidate':128, 'original_id':'next-actual-loss'})
    observer.outcome(128,'committed')
    observer.record({**first, 'candidate':129, 'original_id':'later-loss'})
    assert observer.records[0]['original_id']=='next-actual-loss'


def test_i136_measured_first_loss_is_observed_without_authority_change():
    from tests.test_xc3d_world_physical_person import SpatialActor, ActualPublicationHarness
    outputs = []
    for disabled in (False, True):
        actor = SpatialActor()
        harness = ActualPublicationHarness(actor)
        first = harness.publish(actor.prepare_rows({1:[(1,1.,1,1)]}), 2)
        old, = first.entities
        assert old.position_state == 'measured'
        assert actor.world._previous_entities[old.entity_id] == old
        if disabled:
            actor.world._spatial_observer = None
        barrier = actor.identity.reset_source(replace(actor.sources[0], source_epoch=1))
        actor.sources = (replace(actor.sources[0], source_epoch=1), actor.sources[1])
        actor.us += 100_000
        final = harness.publish(actor.prepare_rows({1:[]}, frame=3,
            withdrawn_members=barrier.withdrawn_members, retired_keys=barrier.retired_keys), 3)
        assert not final.entities
        outputs.append((first.model_dump(mode='json'), final.model_dump(mode='json')))
        if not disabled:
            records = actor.world._private_spatial_diagnostics()
            losses = [r for r in records['records'] if r['kind'] == 'unexpired_held_loss']
            assert len(losses) == 1, records
            row, = losses
            assert row['prior_position_state'] == 'measured'
            assert row['original_observed_at_us'] == old.observed_at_us
            assert row['original_wire_deadline_us'] == old.stale_after_us
            assert row['wire_remaining_us'] > 0 and row['transaction'] == 'committed'
    assert outputs[0] == outputs[1]


def _transfer_split(monkeypatch, *, households=None, before_split=None):
    actor,harness,clock=_r101_actor(monkeypatch)
    first=harness.publish(actor.prepare_image(actor.rows,frame=2,semantic_joints=actor.joints,households=households,
        source_changes={1:{'world_covariance':None}}),2)
    old,=first.entities
    assert old.sources[0].camera_id=='family-room'
    if before_split is not None:
        before_split(actor, harness, clock, old)
    actor.us+=100_000
    changed=dict(actor.joints)
    changed[(2,101)]=tuple((i,u,v+60.,c) for i,u,v,c in changed[(2,101)])
    candidate=actor.prepare_image(actor.rows,frame=3,semantic_joints=changed,
        image_y={(2,101):90.},households=households,source_changes={1:{'world_covariance':None},2:{'world_covariance':None}})
    destination=next(d for d in actor.memberships if d.members[0].observation.member.source_id==2)
    assert len(actor.memberships)==2 and destination.physical_key.serial!=int(old.entity_id.rsplit(':',1)[1])
    assert all(d.estimated_split_custody is not None for d in actor.memberships)
    return actor,harness,clock,old,candidate,destination


def _r158_clean_households():
    return {(1, 1): HouseholdDescriptor('provisional'),
            (2, 101): HouseholdDescriptor('visitor', 'r158-clean-F', 1006, 1)}


def test_r158_clean_certified_transfer_projects_labels_without_changing_original(monkeypatch):
    from tests.test_xc3d_world_physical_person import _r40_empty_spatial
    actor, harness, clock, old, candidate, dto = _transfer_split(monkeypatch,
        households=_r158_clean_households())
    assert old.attribution_quarantine is None
    originals = set(actor.world._fusion.spatial_original_attribution_claims()[old.entity_id])
    view = dto.estimated_split_custody
    fixed_deadline = view.owner_deadline_monotonic_s
    assert all(d.household_conflict is None for d in actor.memberships)
    assert actor.identity.stats().pending_reservations == 0
    held = harness.publish(candidate, 3)
    current, = held.entities
    assert current.entity_id != old.entity_id and current.entity_id.endswith(':' + str(dto.physical_key.serial))
    assert current.position_state == current.lifecycle == 'held'
    assert current.subject.kind == 'provisional' and current.subject.stable_id is None
    assert (current.position, current.covariance, current.observed_at_us, current.stale_after_us,
        current.sources, current.continuity_segment_id) == (old.position, old.covariance,
            old.observed_at_us, old.stale_after_us, old.sources, old.continuity_segment_id)
    q = current.attribution_quarantine
    assert q is not None and originals <= set(q.owners + q.retained_owners)
    assert any(c.physical_serial == int(old.entity_id.rsplit(':', 1)[-1])
        and c.household_stable_id == 1006 for c in q.owners + q.retained_owners)
    assert any(c.physical_serial == dto.physical_key.serial and c.household_stable_id == 1006
        for c in q.owners)
    for frame in (4, 5):
        actor.us += 50_000
        follow = harness.publish(_r40_empty_spatial(actor, frame=frame, evaluation_s=10.4), frame)
        entity, = follow.entities
        assert (entity.position, entity.observed_at_us, entity.stale_after_us, entity.sources) == (
            old.position, old.observed_at_us, old.stale_after_us, old.sources)
        assert entity.attribution_quarantine == q
        assert actor.world._fusion.split_custody_originals()[0].owner_diagnostic_view.owner_deadline_monotonic_s == fixed_deadline
        assert actor.identity.stats().pending_reservations == 0
    actor.us = old.stale_after_us + 1
    expired = harness.publish(_r40_empty_spatial(actor, frame=6, evaluation_s=10.4), 6)
    assert not expired.entities
    actual = actual_node_wire(actor, harness.wire, 'r158-clean-certified-transfer')
    assert [len(s['people']) for s in actual['snapshots']] == [1, 1, 1, 1, 0]
    assert all(d['accepted'] for d in actual['diagnostics'])


@pytest.mark.parametrize('defect', ['missing-certificate', 'forged-certificate', 'foreign-owner',
    'missing-original-claims', 'foreign-retained-claims'])
def test_r158_clean_transfer_requires_issued_certificate_and_exact_original_claims(monkeypatch, defect):
    from noesis_core.world import GlobalWorldFusion
    captured = {}
    def corrupt(actor, harness, clock, old):
        captured.update(actor=actor, harness=harness, snapshot=actor.world.current_snapshot(),
            effects=(len(harness.wire), len(harness.journal), actor.world._revision))
        if defect == 'missing-original-claims':
            original = GlobalWorldFusion.spatial_original_attribution_claims
            monkeypatch.setattr(GlobalWorldFusion, 'spatial_original_attribution_claims',
                lambda f: {} if f is actor.world._fusion else original(f))
        elif defect == 'foreign-retained-claims':
            original = GlobalWorldFusion.spatial_attribution_claims
            def foreign(fusion):
                values = original(fusion)
                if fusion is not actor.world._fusion:
                    values = {k: tuple(c.model_copy(update={'source_epoch': 99}) for c in cs)
                              for k, cs in values.items()}
                return values
            monkeypatch.setattr(GlobalWorldFusion, 'spatial_attribution_claims', foreign)
        else:
            original = actor.world.prepare_spatial
            def unissued(source, preps, memberships, **kwargs):
                if defect == 'missing-certificate':
                    memberships = tuple(replace(d, estimated_split_custody=None) for d in memberships)
                elif defect == 'forged-certificate':
                    view = memberships[0].estimated_split_custody
                    forged = replace(view, native_destinations=tuple((s, view.successor_keys[0])
                        for s, k in view.native_destinations))
                    memberships = tuple(replace(d, estimated_split_custody=forged) for d in memberships)
                else:
                    memberships = tuple(replace(d, owner_instance=object()) for d in memberships)
                return original(source, preps, memberships, **kwargs)
            monkeypatch.setattr(actor.world, 'prepare_spatial', unissued)
    with pytest.raises(WorldPublicationRefused):
        _transfer_split(monkeypatch, households=_r158_clean_households(), before_split=corrupt)
    actor, harness = captured['actor'], captured['harness']
    assert actor.world.current_snapshot() is captured['snapshot']
    assert captured['effects'] == (len(harness.wire), len(harness.journal), actor.world._revision)
    assert actor.identity.stats().pending_reservations == 0


def test_r158_disputed_original_cannot_become_clean_by_losing_private_quarantine(monkeypatch):
    captured = {}
    def omit(actor, harness, clock, old):
        assert old.attribution_quarantine is not None
        captured.update(actor=actor, snapshot=actor.world.current_snapshot())
        actor.world._fusion.set_attribution_quarantines({})
    houses = {(1, 1): HouseholdDescriptor('visitor', 'r158-disputed-K', 1001, 1),
              (2, 101): HouseholdDescriptor('visitor', 'r158-disputed-F', 1006, 1)}
    with pytest.raises(WorldPublicationRefused, match='retained_attribution_conflict_not_authorized'):
        _transfer_split(monkeypatch, households=houses, before_split=omit)
    assert captured['actor'].world.current_snapshot() is captured['snapshot']
    assert captured['actor'].identity.stats().pending_reservations == 0


@pytest.mark.parametrize('barrier', ['stop', 'source-epoch', 'owner-expiry', 'wire-expiry', 'discard'])
def test_r158_clean_transferred_projection_rechecks_original_and_owner_at_commit(monkeypatch, barrier):
    actor, harness, clock, old, candidate, dto = _transfer_split(monkeypatch,
        households=_r158_clean_households())
    first = actor.world.current_snapshot()
    before = (len(harness.wire), len(harness.journal), actor.world._revision)
    if barrier == 'discard':
        actor.world.discard(candidate)
    else:
        if barrier == 'stop':
            harness.before_commit = lambda: actor.identity.stop_image_region_source(actor.sources[1], graph_id='r101-graph')
        elif barrier == 'source-epoch':
            harness.before_commit = lambda: actor.identity.reset_source(replace(actor.sources[1], source_epoch=1))
        elif barrier == 'owner-expiry':
            clock.monotonic = lambda: 121.
        else:
            actor.us = old.stale_after_us + 1
        with pytest.raises(WorldPublicationRefused):
            harness.publish(candidate, 3)
    assert actor.world.current_snapshot() is first
    assert before == (len(harness.wire), len(harness.journal), actor.world._revision)
    assert actor.identity.stats().pending_reservations == 0


def test_i128_transferred_original_quarantine_survives_unchanged_cohorts_until_exact_expiry(monkeypatch):
    from tests.test_xc3d_world_physical_person import _r40_empty_spatial
    houses={(1,1):HouseholdDescriptor('visitor','i128-k',1001,1),
        (2,101):HouseholdDescriptor('visitor','i128-f',1000,1)}
    actor,harness,clock,old,candidate,dto=_transfer_split(monkeypatch,households=houses)
    held=harness.publish(candidate,3)
    entity,=held.entities
    quarantine=entity.attribution_quarantine
    assert quarantine is not None
    assert actor.world._fusion.attribution_quarantines()[entity.entity_id]==quarantine
    assert entity.entity_id!=old.entity_id
    fixed=actor.memberships[0].estimated_split_custody.owner_deadline_monotonic_s
    original_claims=set(old.attribution_quarantine.owners+old.attribution_quarantine.retained_owners)
    for frame in (4,5,6):
        actor.us+=50_000
        final=harness.publish(_r40_empty_spatial(actor,frame=frame,evaluation_s=10.4),frame)
        current,=final.entities
        assert current.entity_id==entity.entity_id and current.attribution_quarantine==quarantine
        assert (current.position,current.covariance,current.observed_at_us,current.stale_after_us)==(
            old.position,old.covariance,old.observed_at_us,old.stale_after_us)
        assert original_claims<=set(current.attribution_quarantine.owners+current.attribution_quarantine.retained_owners)
        assert all(c.physical_serial==int(old.entity_id.rsplit(':',1)[1]) for c in original_claims)
        assert actor.world._fusion.split_custody_originals()[0].owner_diagnostic_view.owner_deadline_monotonic_s==fixed
        assert not actor.world._prepared_candidates and actor.identity.stats().pending_reservations==0
    actor.us=old.stale_after_us+1
    expired=harness.publish(_r40_empty_spatial(actor,frame=7,evaluation_s=10.4),7)
    assert not expired.entities and not actor.world._fusion.attribution_quarantines()
    actual=actual_node_wire(actor,harness.wire,'i128-transferred-attribution-followup')
    assert [len(s['people']) for s in actual['snapshots']]==[1,1,1,1,1,0]
    assert all(d['accepted'] for d in actual['diagnostics'])


@pytest.mark.parametrize('defect',['missing','altered'])
def test_i128_transferred_claims_without_committed_quarantine_never_bootstrap(defect,monkeypatch):
    from tests.test_xc3d_world_physical_person import _r40_empty_spatial
    houses={(1,1):HouseholdDescriptor('visitor','i128-k',1001,1),
        (2,101):HouseholdDescriptor('visitor','i128-f',1000,1)}
    actor,harness,clock,old,candidate,dto=_transfer_split(monkeypatch,households=houses)
    held=harness.publish(candidate,3)
    entity,=held.entities
    if defect=='missing':actor.world._fusion.set_attribution_quarantines({})
    else:
        corrupt=entity.attribution_quarantine.model_copy(update={'owners':(),'retained_owners':()})
        actor.world._fusion.set_attribution_quarantines({entity.entity_id:corrupt})
    actor.us+=50_000
    with pytest.raises(WorldPublicationRefused,match='retained_attribution_conflict_not_authorized'):
        _r40_empty_spatial(actor,frame=4,evaluation_s=10.4)
    assert actor.world.current_snapshot() is held


@pytest.mark.parametrize('barrier',['stop','reset','owner_expiry','wire_expiry','original','claims'])
def test_i128_already_transferred_hold_rechecks_boundaries_at_actual_commit(monkeypatch,barrier):
    from tests.test_xc3d_world_physical_person import _r40_empty_spatial
    houses={(1,1):HouseholdDescriptor('visitor','i128-k',1001,1),
        (2,101):HouseholdDescriptor('visitor','i128-f',1000,1)}
    actor,harness,clock,old,candidate,dto=_transfer_split(monkeypatch,households=houses)
    held=harness.publish(candidate,3)
    actor.us+=50_000
    queued=_r40_empty_spatial(actor,frame=4,evaluation_s=10.4)
    if barrier=='stop':actor.identity.stop_image_region_source(actor.sources[1],graph_id='r101-graph')
    elif barrier=='reset':actor.identity.reset_source(replace(actor.sources[1],source_epoch=1))
    elif barrier=='owner_expiry':clock.monotonic=lambda:121.
    elif barrier=='wire_expiry':actor.us=old.stale_after_us+1
    else:
        original,=actor.world._fusion.split_custody_originals()
        row=(replace(original.row,envelope=original.row.envelope.model_copy(
            update={'observed_at_us':original.row.envelope.observed_at_us+1})) if barrier=='original'
            else replace(original.row,attribution_claims=original.row.attribution_claims[:-1]))
        actor.world._fusion._split_custody[original.episode_id]=replace(original,row=row)
    with pytest.raises(WorldPublicationRefused):harness.publish(queued,4)
    assert actor.world.current_snapshot() is held


@pytest.mark.parametrize('transition',['successor','rejoin'])
def test_i128_transferred_quarantine_follows_real_successor_or_reunion_only(monkeypatch,transition):
    from tests.test_xc3d_world_physical_person import _r40_empty_spatial
    houses={(1,1):HouseholdDescriptor('visitor','i128-k',1001,1),
        (2,101):HouseholdDescriptor('visitor','i128-f',1000,1)}
    actor,harness,clock,old,candidate,dto=_transfer_split(monkeypatch,households=houses)
    held=harness.publish(candidate,3)
    actor.us+=50_000
    changed=dict(actor.joints)
    if transition=='successor':
        changed[(2,101)]=tuple((i,u,v+60.,c) for i,u,v,c in changed[(2,101)])
    candidate=actor.prepare_image(actor.rows,frame=4,semantic_joints=changed,
        image_y={(2,101):90.} if transition=='successor' else {},households=houses,
        source_changes={} if transition=='successor' else {1:{'world_covariance':None},2:{'world_covariance':None}})
    final=harness.publish(candidate,4)
    current,=final.entities
    if transition=='successor':
        assert current.observed_at_us>old.observed_at_us
        assert not actor.world._fusion.split_custody_originals()
        assert current.subject.kind=='visitor' and current.subject.stable_id==1001
        assert current.attribution_quarantine is None
        assert all(s.observation_id!=old.sources[0].observation_id for s in current.sources if s.accepted)
    else:
        assert current.entity_id==old.entity_id and current.observed_at_us==old.observed_at_us
        assert current.stale_after_us==old.stale_after_us
        assert current.subject.kind=='provisional' and current.attribution_quarantine is not None
    actor.us+=50_000
    following=harness.publish(_r40_empty_spatial(actor,frame=5,evaluation_s=10.4),5)
    assert len(following.entities)==1
    assert following.entities[0].entity_id==current.entity_id


def test_i124_first_loss_preserves_prior_fixed_certificate_not_new_creation(monkeypatch):
    from tests.test_xc3d_world_physical_person import _r40_empty_spatial
    actor,harness,clock,old,candidate,dto=_transfer_split(monkeypatch)
    harness.publish(candidate,3)
    prior,=actor.world._fusion.split_custody_originals()
    view=prior.owner_diagnostic_view
    assert view is dto.estimated_split_custody
    assert replace(prior,owner_diagnostic_view=None)==prior
    actor.us+=100_000
    actor.identity.stop_image_region_source(actor.sources[1],graph_id='r101-graph')
    candidate=_r40_empty_spatial(actor,frame=4,evaluation_s=10.4)
    harness.publish(candidate,4)
    record=next(r for r in actor.world._private_spatial_diagnostics()['records'] if r['kind']=='unexpired_held_loss')
    guard=json.loads(record['prior_owner_guard'])
    assert not record['incoming_view_present']
    assert guard['fixed_deadline_mono_s']==view.owner_deadline_monotonic_s
    assert guard['checked_mono_s']==clock.monotonic() and not guard['expired_now']
    assert guard['current_query_reason']=='split_custody_native_binding'
    assert any(row[5] for row in guard['member_guards'])
    assert len(json.dumps(actor.world._private_spatial_diagnostics()).encode())<=8192


@pytest.mark.parametrize('observer',['absent','raising','missing_metadata','owner_expiry'])
def test_i124_optional_guard_never_changes_actual_loss_or_deadline(monkeypatch,observer):
    from tests.test_xc3d_world_physical_person import _r40_empty_spatial
    actor,harness,clock,old,candidate,dto=_transfer_split(monkeypatch)
    held=harness.publish(candidate,3)
    view=dto.estimated_split_custody
    if observer=='absent':actor.world._spatial_observer=None
    elif observer=='raising':
        def fail(*args,**kwargs):raise RuntimeError('optional observer')
        monkeypatch.setattr(actor.identity,'estimated_split_custody_diagnostic',fail)
    elif observer=='missing_metadata':
        actor.world._fusion._split_custody={k:replace(v,owner_diagnostic_view=None)
            for k,v in actor.world._fusion._split_custody.items()}
    actor.us+=100_000
    if observer=='owner_expiry':clock.monotonic=lambda:view.owner_deadline_monotonic_s+.001
    else:actor.identity.stop_image_region_source(actor.sources[1],graph_id='r101-graph')
    candidate=_r40_empty_spatial(actor,frame=4,evaluation_s=10.4)
    final=harness.publish(candidate,4)
    assert not final.entities and held.entities[0].stale_after_us==old.stale_after_us
    if observer=='owner_expiry':
        record=next(r for r in actor.world._private_spatial_diagnostics()['records'] if r['kind']=='unexpired_held_loss')
        guard=json.loads(record['prior_owner_guard'])
        assert guard['expired_now'] and guard['fixed_deadline_mono_s']==view.owner_deadline_monotonic_s


def test_i121_selected_original_follows_only_its_native_destination(monkeypatch):
    actor,harness,clock,old,candidate,dto=_transfer_split(monkeypatch)
    final=harness.publish(candidate,3)
    held,=final.entities
    assert held.entity_id==f'physical:{dto.run_id}:{dto.physical_key.serial}'
    assert (held.position,held.covariance,held.observed_at_us,held.stale_after_us)==(
        old.position,old.covariance,old.observed_at_us,old.stale_after_us)
    assert held.position_state=='held' and held.subject.kind=='provisional'
    assert actor.identity.stats().pending_reservations==0
    actual=actual_node_wire(actor,harness.wire,'i121-native-destination')
    assert [len(s['people']) for s in actual['snapshots']]==[1,1]
    assert actual['snapshots'][-1]['people'][0]['entityId']==held.entity_id


@pytest.mark.parametrize('barrier',['stop','reset','owner_expiry','wire_expiry'])
def test_i121_transfer_commit_revalidates_real_owner_and_original_boundaries(monkeypatch,barrier):
    actor,harness,clock,old,candidate,dto=_transfer_split(monkeypatch)
    first=actor.world.current_snapshot()
    if barrier=='stop':actor.identity.stop_image_region_source(actor.sources[1],graph_id='r101-graph')
    elif barrier=='reset':actor.identity.reset_source(replace(actor.sources[1],source_epoch=1))
    elif barrier=='owner_expiry':clock.monotonic=lambda:121.
    else:actor.us=old.stale_after_us+1
    with pytest.raises(WorldPublicationRefused):harness.publish(candidate,3)
    assert actor.world.current_snapshot() is first


def test_i121_discarded_current_successor_does_not_consume_transfer(monkeypatch):
    actor,harness,clock,old,candidate,dto=_transfer_split(monkeypatch)
    held=harness.publish(candidate,3)
    actor.us+=100_000
    changed=dict(actor.joints)
    changed[(2,101)]=tuple((i,u,v+60.,c) for i,u,v,c in changed[(2,101)])
    candidate=actor.prepare_image(actor.rows,frame=4,semantic_joints=changed,image_y={(2,101):90.})
    assert actor.world._prepared_candidates[candidate._candidate_id][1].spatial_reservations
    actor.world.discard(candidate)
    assert actor.world.current_snapshot() is held
    assert not actor.world._fusion._consumed_split_episodes
    actor.us+=100_000
    final=harness.publish(actor.prepare_image(actor.rows,frame=5,semantic_joints=changed,
        image_y={(2,101):90.},source_changes={1:{'world_covariance':None},2:{'world_covariance':None}}),5)
    assert len(final.entities)==1 and final.entities[0].entity_id==held.entities[0].entity_id
    assert final.entities[0].observed_at_us==old.observed_at_us


def test_i121_current_successor_consumes_entire_scope_without_doublehead(monkeypatch):
    actor,harness,clock,old,candidate,dto=_transfer_split(monkeypatch)
    held=harness.publish(candidate,3)
    actor.us+=100_000
    changed=dict(actor.joints)
    changed[(2,101)]=tuple((i,u,v+60.,c) for i,u,v,c in changed[(2,101)])
    final=harness.publish(actor.prepare_image(actor.rows,frame=4,semantic_joints=changed,
        image_y={(2,101):90.}),4)
    assert len(final.entities)==1 and final.entities[0].position_state=='measured'
    assert final.entities[0].observed_at_us>old.observed_at_us
    assert final.entities[0].entity_id!=held.entities[0].entity_id
    assert not actor.world._fusion.split_custody_originals()
    actual=actual_node_wire(actor,harness.wire,'i121-consumed')
    assert [len(s['people']) for s in actual['snapshots']]==[1,1,1]


def test_i121_same_pair_reunion_and_expiry_never_renew_original(monkeypatch):
    actor,harness,clock,old,candidate,dto=_transfer_split(monkeypatch)
    held=harness.publish(candidate,3)
    actor.us+=100_000
    final=harness.publish(actor.prepare_image(actor.rows,frame=4,semantic_joints=actor.joints,
        source_changes={1:{'world_covariance':None},2:{'world_covariance':None}}),4)
    assert len(final.entities)==1 and final.entities[0].entity_id==old.entity_id
    assert final.entities[0].observed_at_us==old.observed_at_us
    assert final.entities[0].stale_after_us==old.stale_after_us
    actor.us=old.stale_after_us+1
    expired=harness.publish(actor.prepare_image(actor.rows,frame=5,semantic_joints=actor.joints,
        source_changes={1:{'world_covariance':None},2:{'world_covariance':None}}),5)
    assert not expired.entities


def test_i121_transfer_preserves_complete_claims_and_independent_third_person(monkeypatch):
    actor,harness,clock=_r101_actor(monkeypatch)
    actor.rows[2].append((102,3.,1,1))
    actor.joints[(2,102)]=tuple((i,u,v+60.,c) for i,u,v,c in actor.joints[(2,101)])
    houses={(1,1):HouseholdDescriptor('visitor','i121-k',1001,1),
        (2,101):HouseholdDescriptor('visitor','i121-f',1000,1),
        (2,102):HouseholdDescriptor('visitor','i121-third',1002,1)}
    first=harness.publish(actor.prepare_image(actor.rows,frame=2,semantic_joints=actor.joints,
        households=houses,source_changes={1:{'world_covariance':None}}),2)
    old=next(e for e in first.entities if e.attribution_quarantine is not None)
    actor.us+=100_000
    changed=dict(actor.joints)
    changed[(2,101)]=tuple((i,u,v+60.,c) for i,u,v,c in changed[(2,101)])
    final=harness.publish(actor.prepare_image(actor.rows,frame=3,semantic_joints=changed,
        image_y={(2,101):90.},households=houses,
        source_changes={1:{'world_covariance':None},2:{'world_covariance':None}}),3)
    assert len(final.entities)==2
    held=next(e for e in final.entities if e.position_state=='held')
    assert held.entity_id!=old.entity_id and held.subject.kind=='provisional'
    assert held.attribution_quarantine is not None
    assert set(held.attribution_quarantine.stable_ids)=={1000,1001}
    assert next(e for e in final.entities if e is not held).subject.stable_id==1002
    actual=actual_node_wire(actor,harness.wire,'i121-claims-third')
    assert [len(s['people']) for s in actual['snapshots']]==[2,2]
