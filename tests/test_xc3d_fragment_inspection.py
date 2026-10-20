"""Private inspection of genuine owner decisions; never an authority input."""
from contextlib import contextmanager
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json


@contextmanager
def public(tmp_path,monkeypatch,targets=None):
    if targets is None:monkeypatch.delenv('NOESIS_XC3D_BODY_DIAGNOSTIC_TARGETS_S',raising=False)
    else:monkeypatch.setenv('NOESIS_XC3D_BODY_DIAGNOSTIC_TARGETS_S',targets)
    spec=spec_from_file_location('fragment_inspection_examples',Path(__file__).with_name('test_xc3d_ambiguous_fragment_return.py'))
    m=module_from_spec(spec);spec.loader.exec_module(m)
    f,packet,monitor=m.fixture(tmp_path,monkeypatch)
    try:yield f,packet
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


def returned(f,packet):
    first=f.publish([packet(1,candidate='head')])
    original=next(e for e in first.entities if e.entity_id.endswith(':'+str(
        next(m.person_key.serial for k,m in f.identity.ledger._members.items() if k.raw_id==12))))
    f.publish([packet(2,(10,),shadow_raws=(12,))])
    result=f.publish([packet(3,candidate='weak')])
    retained=next(e for e in result.entities if e.entity_id==original.entity_id)
    assert len(result.entities)==2
    assert (retained.position,retained.covariance,retained.observed_at_us,retained.stale_after_us)==(
        original.position,original.covariance,original.observed_at_us,original.stale_after_us)
    assert retained.stale_after_us==retained.observed_at_us+2_000_000
    return next(m for k,m in f.identity.ledger._members.items() if k.raw_id==12)


def test_fragment_snapshot_is_exact_detached_and_read_only(tmp_path,monkeypatch):
    with public(tmp_path,monkeypatch) as (f,packet):
        m=returned(f,packet);s=m.observation.stamp
        before=(f.identity.ledger._authority_revision,f.identity.stats())
        value=f.identity._private_image_region_ambiguity(s)
        assert value['stamp_match'] and value['available']
        assert value['cause']=='fragment_provisional' and value['birth_established']
        assert not value['birth_quarantined'] and not value['vote_blocked']
        assert value['fragment_return_current']
        assert value['fragment_original']==[2,0,12,s.member.lifecycle_generation,s.lifecycle_started_frame,3]
        assert value['fragment_anchor_total']==1 and value['fragment_anchor_omitted']==0
        assert value['fragment_anchors'][0][2]==10
        value['fragment_original'][5]=999;value['fragment_anchors'][0][2]=999
        again=f.identity._private_image_region_ambiguity(s)
        assert again['fragment_original'][5]==3 and again['fragment_anchors'][0][2]==10
        assert before==(f.identity.ledger._authority_revision,f.identity.stats())
        stale=f.identity._private_image_region_ambiguity(replace(s,frame_id=s.frame_id+1))
        assert not stale['available'] and not stale['stamp_match'] and stale['cause']=='stale_stamp'
        assert stale['birth_established'] is None and stale['fragment_original'] is None
        assert f.identity._private_image_region_ambiguity(None)['cause']=='unavailable'
        assert f.identity._private_image_region_ambiguity(replace(s,member=replace(s.member,raw_id=9999)))['cause']=='unavailable'


def test_ordinary_birth_is_not_reported_as_fragment(tmp_path,monkeypatch):
    with public(tmp_path,monkeypatch) as (f,packet):
        f.publish([packet(1,(10,))]);f.publish([packet(2)])
        m=next(m for k,m in f.identity.ledger._members.items() if k.raw_id==12)
        value=f.identity._private_image_region_ambiguity(m.observation.stamp)
        assert value['cause']=='birth_quarantined' and value['provisional_extra']
        assert not value['birth_established'] and value['fragment_original'] is None


def test_actual_vote_block_is_not_reported_as_birth_or_fragment():
    from noesis.identity_association_service import IdentityAssociationService
    from tests.test_identity_association_service import vote_quarantine_step
    from tests.test_mv3dt_person_association import spatial_policy,two_cue_image_region_fixture
    selected,*_=two_cue_image_region_fixture()
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',spatial_policy(),selected,live_track_continuity=True)
    vote_quarantine_step(owner,1,competitor='incompatible')
    _,rows,*_=vote_quarantine_step(owner,2)
    value=owner._private_image_region_ambiguity(rows[0].stamp)
    assert value['cause']=='vote_blocked' and value['vote_blocked']
    assert not value['birth_quarantined'] and value['fragment_original'] is None


def test_diagnostic_anchor_projection_is_bounded_without_touching_owner(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from noesis.identity_association_service import IdentityAssociationService
    with public(tmp_path,monkeypatch) as (f,packet):
        m=returned(f,packet);s=m.observation.stamp;fact=m.image_region_fragment_provisional
        # Pure output-bound control, NOT an inserted authority/member grant.
        inspect_only=SimpleNamespace(_ledger=SimpleNamespace(_members={s.member:replace(m,
            image_region_fragment_provisional=replace(fact,anchors=fact.anchors*10))}))
        value=IdentityAssociationService._private_image_region_ambiguity(inspect_only,s)
        assert value['fragment_anchor_total']==10 and len(value['fragment_anchors'])==8
        assert value['fragment_anchor_omitted']==2
        assert f.identity.ledger._members[s.member] is m


def test_world_target_off_never_queries_private_cause(tmp_path,monkeypatch):
    from noesis.identity_association_service import IdentityAssociationService
    calls=[]
    def forbidden(*_):
        calls.append(True);raise AssertionError('default OFF must never query')
    with public(tmp_path,monkeypatch) as (f,packet):
        monkeypatch.setattr(IdentityAssociationService,'_private_image_region_ambiguity',forbidden)
        returned(f,packet)
        assert not calls
        assert not any(r['kind']=='target_cohort_admission' for r in f.world._private_spatial_diagnostics()['records'])


def test_world_target_adds_only_bounded_detached_cause(tmp_path,monkeypatch):
    with public(tmp_path,monkeypatch,'0.1') as (f,packet):
        returned(f,packet)
        value=f.world._private_spatial_diagnostics()
        target=next(r for r in value['records'] if r['kind']=='target_cohort_admission' and r['source']==2)
        members=json.loads(target['target_members'])['items']
        row=next(m for m in members if m['raw']==12)
        assert row['reason']=='image_region_ambiguous_member'
        assert row['ambiguity']['cause']=='fragment_provisional' and row['ambiguity']['stamp_match']
        assert 'ambiguity' not in next(m for m in members if m['raw']==10)
        assert len(value['records'])<=8 and len(target['target_members'].encode())<=4096
        assert len(f.world._spatial_observer.encoded().encode())<=8192
        text=target['target_members']
        row['ambiguity']['fragment_anchors'].clear()
        again=f.world._private_spatial_diagnostics()
        assert next(r for r in again['records'] if r['kind']=='target_cohort_admission'
            and r['source']==2)['target_members']==text


def test_unavailable_private_diagnostic_never_changes_world_admission(tmp_path,monkeypatch):
    from noesis.identity_association_service import IdentityAssociationService
    with public(tmp_path,monkeypatch,'0.1') as (f,packet):
        def unavailable(*_):raise RuntimeError('inspection unavailable')
        monkeypatch.setattr(IdentityAssociationService,'_private_image_region_ambiguity',unavailable)
        m=returned(f,packet)
        assert m.image_region_fragment_provisional is not None
        assert f.identity.ledger.image_region_error(m.observation.stamp,f.ingress._context)=='image_region_ambiguous_member'
        value=f.world._private_spatial_diagnostics()
        row=next(r for r in value['records'] if r['kind']=='target_cohort_admission' and r['source']==2)
        member=next(m for m in json.loads(row['target_members'])['items'] if m['raw']==12)
        assert member['ambiguity']==dict(available=False,stamp_match=None,cause='diagnostic_unavailable')
        assert len(f.world._spatial_observer.encoded().encode())<=8192
        assert f.identity.stats().pending_reservations==0
