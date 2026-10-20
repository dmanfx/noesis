"""Approved returning-fragment recall tradeoff, genuine public owner/World path."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from dataclasses import replace
import pytest


def fixture(tmp_path,monkeypatch,**kwargs):
    path=Path(__file__).with_name('test_xc3d_provisional_extra_birth.py')
    spec=spec_from_file_location('fragment_return_examples',path)
    module=module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.public_fixture(tmp_path,monkeypatch,**kwargs)


@pytest.mark.parametrize('new_anchor',[False,True])
def test_genuine_shadow_return_with_current_competition_withholds_fresh_point(tmp_path,monkeypatch,new_anchor):
    f,packet,monitor=fixture(tmp_path,monkeypatch)
    try:
        first=f.publish([packet(1,(12,) if new_anchor else (10,12),candidate='head')])
        owner=f.identity.ledger
        key=next(k for k in owner._members if k.raw_id==12)
        assert owner._members[key].image_region_birth_established
        old=next(e for e in first.entities if e.entity_id.endswith(':'+str(owner._members[key].person_key.serial)))
        shadow=packet(2,(10,),shadow_raws=(12,))
        assert shadow[0].continuity.shadow_lifecycles==((12,key.lifecycle_generation),)
        f.publish([shadow])
        assert not owner._members[key].image_region_participant
        returned=f.publish([packet(3,candidate='weak')])
        member=owner._members[key]
        assert member.image_region_birth_established and member.image_region_participant
        assert owner.image_region_error(member.observation.stamp,f.ingress._context)=='image_region_ambiguous_member'
        kept=next(e for e in returned.entities if e.entity_id==old.entity_id)
        assert (kept.position,kept.covariance,kept.observed_at_us,kept.stale_after_us)==(
            old.position,old.covariance,old.observed_at_us,old.stale_after_us)
        assert kept.position_state==old.position_state  # Existing bounded retention, not a new World state.
        fact=member.image_region_fragment_provisional
        assert fact is not None and fact.original==member.observation.stamp
        # Genuine ordered receipts remain within the existing .35s native and
        # delivery fences; compact whole cohorts fit the Node fixture budget.
        for frame in (4,14,24,34,44,54,64,65):
            snapshot=f.publish([packet(frame,candidate='weak')])
            m=owner._members[key]
            assert m.image_region_fragment_provisional==fact and m.image_region_birth_established
            old_rows=[e for e in snapshot.entities if e.entity_id==old.entity_id]
            if old_rows:
                assert old_rows[0].observed_at_us==old.observed_at_us
                assert old_rows[0].stale_after_us==old.stale_after_us
        assert len(snapshot.entities)==1 and not old_rows
        actual=f.examples.actual_node_wire(f,f.wire,'fragment-return-finite-old-point')
        counts=[len(s['people']) for s in actual['snapshots']]
        assert counts[:3]==([1,2,2] if new_anchor else [2,2,2]) and counts[-1]==1 and max(counts)==2
        assert all(d['accepted'] for d in actual['diagnostics'])
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('release',['head','different_torso'])
def test_return_veto_requires_independent_anatomy_to_release(tmp_path,monkeypatch,reverse,release):
    f,packet,monitor=fixture(tmp_path,monkeypatch,reverse=reverse)
    try:
        f.publish([packet(1,candidate='head')]);owner=f.identity.ledger
        key=next(k for k in owner._members if k.raw_id==12)
        f.publish([packet(2,(10,),shadow_raws=(12,))]);f.publish([packet(3)])
        fact=owner._members[key].image_region_fragment_provisional;assert fact is not None
        for frame,kind in ((4,'duplicate_head'),(5,'head_unknown')):
            f.publish([packet(frame,candidate=kind)])
            assert owner._members[key].image_region_fragment_provisional==fact
        current_work,current_tracks=packet(6,candidate=release)
        released=f.publish([(current_work,current_tracks)])
        m=owner._members[key]
        assert m.image_region_fragment_provisional is None
        assert m.image_region_birth_established
        assert owner.image_region_error(m.observation.stamp,f.ingress._context) is None
        own=next(e for e in released.entities if e.entity_id.endswith(':'+str(m.person_key.serial)))
        assert own.observed_at_us==next(t['observed_at_us'] for t in current_tracks if t['tracker_id']==12)
        assert len(released.entities)==2 and f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('case',['no_shadow','reliable_head','different_torso','no_anchor_head','default'])
def test_return_creation_is_not_a_normal_established_track_policy(tmp_path,monkeypatch,case):
    f,packet,monitor=fixture(tmp_path,monkeypatch,enabled=case!='default')
    try:
        f.publish([packet(1,candidate='head')]);owner=f.identity.ledger
        key=next(k for k in owner._members if k.raw_id==12)
        if case=='no_shadow':f.publish([packet(2,candidate='head')])
        else:f.publish([packet(2,(10,),shadow_raws=(12,))])
        result=f.publish([packet(3,candidate='head' if case=='reliable_head' else
            'different_torso' if case=='different_torso' else 'weak',anchor='weak' if case=='no_anchor_head' else 'head')])
        m=owner._members[key]
        assert m.image_region_fragment_provisional is None
        assert m.image_region_birth_established==(case!='default')
        assert len(result.entities)==2
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('uncertain',['body_none','partial','anchor_absent','caps_missing','caps_failed'])
def test_observed_return_negative_is_not_released_by_unknown(tmp_path,monkeypatch,uncertain):
    from noesis.pipelines.world_quality_adapter import capture_body_support,freeze_quality_work
    f,packet,monitor=fixture(tmp_path,monkeypatch)
    try:
        f.publish([packet(1,candidate='head')]);owner=f.identity.ledger
        key=next(k for k in owner._members if k.raw_id==12)
        f.publish([packet(2,(10,),shadow_raws=(12,))]);f.publish([packet(3)])
        fact=owner._members[key].image_region_fragment_provisional;assert fact is not None
        for frame in (4,5):
            work,tracks=packet(frame,(12,) if uncertain=='anchor_absent' else (10,12),
                candidate='weak' if uncertain=='partial' else 'head',
                shadow_raws=(10,) if uncertain=='anchor_absent' else ())
            inputs=[]
            for row,track in zip(work.rows,tracks):
                body=row.evidence.body
                if track['tracker_id']==12:
                    if uncertain=='body_none':body=None
                    if uncertain=='partial':
                        points=list(body.keypoints);x,y,_=points[11];points[11]=(x,y,.1)
                        body=capture_body_support(track,row.evidence.lifecycle,points,pose_is_current=True)
                inputs.append(replace(row.evidence,body=body))
            availability=None if uncertain=='caps_missing' else replace(work.owner_availability,status='failed') if uncertain=='caps_failed' else work.owner_availability
            work=replace(freeze_quality_work(work.private,work.continuity,tuple(inputs)),
                ingress_monotonic_s=work.ingress_monotonic_s,owner_availability=availability)
            f.publish([(work,tracks)])
            assert owner._members[key].image_region_fragment_provisional==fact
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('bad_caps',['missing','failed'])
def test_return_cannot_create_competition_without_current_caps(tmp_path,monkeypatch,bad_caps):
    f,packet,monitor=fixture(tmp_path,monkeypatch)
    try:
        f.publish([packet(1,candidate='head')]);owner=f.identity.ledger
        key=next(k for k in owner._members if k.raw_id==12)
        f.publish([packet(2,(10,),shadow_raws=(12,))])
        work,tracks=packet(3)
        caps=None if bad_caps=='missing' else replace(work.owner_availability,status='failed')
        f.publish([(replace(work,owner_availability=caps),tracks)])
        assert owner._members[key].image_region_fragment_provisional is None
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('event',['stop','epoch','run','termination'])
def test_fragment_negative_has_native_lifecycle_scope(tmp_path,monkeypatch,event):
    from reid.mv3dt_person_association import MemberRemoval
    f,packet,monitor=fixture(tmp_path,monkeypatch)
    try:
        f.publish([packet(1,candidate='head')]);owner=f.identity.ledger
        key=next(k for k in owner._members if k.raw_id==12)
        f.publish([packet(2,(10,),shadow_raws=(12,))]);f.publish([packet(3)])
        member=owner._members[key];assert member.image_region_fragment_provisional is not None
        binding=next(b for b in owner.source_bindings() if b.source_id==2)
        if event=='stop':
            assert owner.stop_image_region_source(binding,graph_id=owner._image_graph_id).disposition=='observed'
            assert owner.image_region_error(member.observation.stamp,f.ingress._context)=='image_region_current_support_required'
        elif event=='epoch':
            assert owner.reset_source(replace(binding,source_epoch=binding.source_epoch+1)).disposition=='reset'
            assert not any(k.source_id==2 for k in owner._members)
        elif event=='run':
            assert owner.reset_run('next-native-run').disposition=='reset' and not owner._members
        else:
            s=member.observation.stamp
            assert owner.withdraw(MemberRemoval('actual-native-end',s,'termination',4,
                s.acquired_at_s+1/30,s.clock_domain,s.media_time_s+1/30)).disposition=='removed'
            assert key not in owner._members
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()
