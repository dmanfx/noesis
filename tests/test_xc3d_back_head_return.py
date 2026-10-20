"""R107/900 literal anatomy in synthetic CURRENT public owner/World cohorts.

Raw10/12 are fixture-native anchor/return keys, not aliases for R107 raw23/19.
Only the recorded geometry is reused; no recorded receipt is retimed.
"""
from contextlib import contextmanager
from dataclasses import replace
from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path
import time
import pytest

CHILD_BOX=(456.497711,428.68866,46.8082199,125.258202)
CHILD=((467.210052,447.096497,.141479492),(467.598175,445.162903,.181030273),(469.305939,444.505463,.0296478271),
 (471.557068,445.975006,.853515625),(480.251129,443.96405,.512207031),
 (466.356171,459.664856,.997070312),(491.817352,456.416412,.992675781),(463.173523,480.470306,.980957031),
 (500.045654,470.260925,.906738281),(461.698639,491.762512,.852539062),(493.602753,464.073425,.538574219),
 (475.98172,493.386719,.992675781),(493.292236,491.839844,.984863281),(480.095886,517.827332,.984863281),
 (493.136993,515.429688,.960449219),(479.863007,545.980469,.98046875),(494.922363,540.411743,.963378906))
ANCHOR_BOX=(452.25,353.,59.25,203.)
ANCHOR=((472.616211,374.075775,.960449219),(475.537842,369.646088,.972167969),(468.170258,371.860931,.894042969),
 (484.429718,369.076569,.893554688),(464.867554,374.518738,.314941406),
 (500.054047,388.503906,.983398438),(460.167572,392.8703,.990234375),(504.118927,422.865631,.823730469),
 (459.913513,422.106262,.961425781),(498.275665,454.,.543945312),(467.916229,432.737488,.935546875),
 (497.767578,442.609375,.891601562),(473.124329,443.875,.927246094),(497.767578,482.476562,.518554688),
 (474.267578,481.84375,.612792969),(498.529724,516.521851,.176391602),(477.697296,516.142212,.220214844))
CHILD_BOX_942=(460.779602,430.21405,43.5581131,122.117188)
CHILD_942=((468.646942,444.301575,.106689453),(469.184204,442.578918,.116943359),(471.64035,442.349213,.0150680542),
 (473.098694,445.0672,.914550781),(484.074554,443.574219,.551269531),
 (468.263153,460.456238,.999023438),(493.131592,457.700012,.994140625),(465.423248,482.046875,.995605469),
 (501.651306,474.543762,.709472656),(465.730255,494.679688,.957519531),(496.125,482.659363,.291748047),
 (477.703949,494.450012,.994628906),(494.589905,493.53125,.982910156),(480.850891,518.873413,.993652344),
 (494.436401,516.270325,.975097656),(481.618408,544.521851,.984863281),(496.662292,541.382812,.962890625))
ANCHOR_BOX_942=(452.,353.,62.,199.)
ANCHOR_942=((475.662445,374.2453,.986816406),(479.020294,370.146088,.994628906),(471.931458,371.326172,.942382812),
 (488.720825,368.966003,.969726562),(469.692902,372.195709,.167236328),
 (504.639587,390.269531,.979492188),(465.588837,391.698059,.991210938),(509.987305,422.193756,.730957031),
 (461.85788,422.566406,.895996094),(500.659912,449.273438,.510253906),(465.464478,444.925781,.707519531),
 (498.670044,447.782806,.87890625),(474.294403,447.28595,.923339844),(500.286804,482.564056,.219482422),
 (472.677673,480.576569,.349365234),(501.530457,504.923431,.0693969727),(475.662445,502.811707,.123657227))


@contextmanager
def fixture(tmp_path,monkeypatch,reverse=False):
    spec=spec_from_file_location('back_head_examples',Path(__file__).with_name('test_xc3d_ambiguous_fragment_return.py'))
    m=module_from_spec(spec);spec.loader.exec_module(m)
    f,packet,monitor=m.fixture(tmp_path,monkeypatch,reverse=reverse)
    try:yield f,packet
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


def literal(f,packet,frame,*,fault=None,record=900):
    from noesis.pipelines.world_quality_adapter import capture_body_support,freeze_quality_work
    work,tracks=packet(frame,(12,) if fault=='anchor_absent' else (10,12),
        shadow_raws=(10,) if fault=='anchor_absent' else ())
    inputs=[]
    for row,track in zip(work.rows,tracks):
        child=track['tracker_id']==12
        boxes=(CHILD_BOX,ANCHOR_BOX) if record==900 else (CHILD_BOX_942,ANCHOR_BOX_942)
        joints=(CHILD,ANCHOR) if record==900 else (CHILD_942,ANCHOR_942)
        box=list(boxes[0 if child else 1]);j=list(joints[0 if child else 1])
        if child:
            if fault in ('one_ear','low_ear'):j[4]=(*j[4][:2],.1 if fault=='one_ear' else .349)
            elif fault=='ear_edge':j[3]=(box[0],j[3][1],j[3][2])
            elif fault=='torso_edge':j[5]=(box[0],j[5][1],j[5][2])
            elif fault=='inverted':j[3]=(j[3][0],470.,j[3][2]);j[4]=(j[4][0],470.,j[4][2])
            elif fault=='overspan':j[3]=(box[0]+1,j[3][1],j[3][2]);j[4]=(box[0]+box[2]-1,j[4][1],j[4][2])
            elif fault=='outside_shoulders':j[3]=(box[0]+1,j[3][1],j[3][2]);j[4]=(box[0]+3,j[4][1],j[4][2])
        elif fault=='duplicate_head' or fault=='second_anchor_overlaps' and track['tracker_id']==14:
            box[1]+=73.4;j=[(x,y+73.4,c) for x,y,c in j]
        elif fault=='anchor_head_unknown':j=[(x,y,.1 if i<5 else c) for i,(x,y,c) in enumerate(j)]
        track['bbox']=box
        body=capture_body_support(track,row.evidence.lifecycle,j,pose_is_current=True)
        inputs.append(replace(row.evidence,body=body))
    work=replace(freeze_quality_work(work.private,work.continuity,tuple(inputs)),
        ingress_monotonic_s=work.ingress_monotonic_s,owner_availability=work.owner_availability)
    if fault=='caps_missing':work=replace(work,owner_availability=None)
    elif fault=='caps_failed':work=replace(work,owner_availability=replace(work.owner_availability,status='failed'))
    return work,tracks


def seeded_return(f,packet):
    first=f.publish([packet(1,candidate='head')])
    key=next(k for k in f.identity.ledger._members if k.raw_id==12)
    original=next(e for e in first.entities if e.entity_id.endswith(':'+str(f.identity.ledger._members[key].person_key.serial)))
    f.publish([packet(2,(10,),shadow_raws=(12,))]);f.publish([packet(3,candidate='weak')])
    m=f.identity.ledger._members[key]
    assert m.image_region_birth_established and m.image_region_fragment_provisional is not None
    return key,original,m.image_region_fragment_provisional


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('record',[900,942])
def test_literal_r107_back_head_releases_only_established_return(tmp_path,monkeypatch,reverse,record):
    with fixture(tmp_path,monkeypatch,reverse) as (f,packet):
        key,old,fact=seeded_return(f,packet)
        work,tracks=literal(f,packet,4,record=record);snapshot=f.publish([(work,tracks)])
        m=f.identity.ledger._members[key]
        assert m.image_region_birth_established and not m.image_region_birth_quarantined
        assert m.image_region_fragment_provisional is None
        assert f.identity.ledger.image_region_error(m.observation.stamp,f.ingress._context) is None
        own=next(e for e in snapshot.entities if e.entity_id==old.entity_id)
        assert own.observed_at_us==next(t['observed_at_us'] for t in tracks if t['tracker_id']==12)
        assert own.observed_at_us>old.observed_at_us and own.stale_after_us==own.observed_at_us+2_000_000
        assert own.stale_after_us>old.stale_after_us
        actual=f.examples.actual_node_wire(f,f.wire,'back-head-current-'+str(record)+'-'+str(reverse))
        assert [len(s['people']) for s in actual['snapshots']]==[2,2,2,2]
        assert all(d['accepted'] for d in actual['diagnostics']) and f.identity.stats().pending_reservations==0


@pytest.mark.parametrize('fault',['one_ear','low_ear','ear_edge','torso_edge','inverted','overspan',
    'outside_shoulders','duplicate_head','anchor_head_unknown','anchor_absent','caps_missing','caps_failed','stale'])
def test_unknown_or_inconsistent_back_head_never_releases(tmp_path,monkeypatch,fault):
    with fixture(tmp_path,monkeypatch) as (f,packet):
        key,old,fact=seeded_return(f,packet)
        item=literal(f,packet,4,fault=fault)
        # Freeze genuine current receipts first, then delay the actor; never
        # forge an old ingress under a new availability receipt.
        with monkeypatch.context() as queued:
            if fault=='stale':
                now=time.monotonic();queued.setattr(time,'monotonic',lambda:now+.36)
            snapshot=f.publish([item])
        m=f.identity.ledger._members[key]
        assert m.image_region_fragment_provisional==fact
        assert f.identity.ledger.image_region_error(m.observation.stamp,f.ingress._context) is not None
        retained=next(e for e in snapshot.entities if e.entity_id==old.entity_id)
        assert (retained.position,retained.covariance,retained.observed_at_us,retained.stale_after_us)==(
            old.position,old.covariance,old.observed_at_us,old.stale_after_us)


def test_two_ear_new_birth_remains_provisional(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (f,packet):
        f.publish([packet(1,(10,))]);snapshot=f.publish([literal(f,packet,2)])
        m=next(m for k,m in f.identity.ledger._members.items() if k.raw_id==12)
        assert not m.image_region_birth_established and m.image_region_birth_quarantined
        assert m.image_region_provisional_extra is not None and m.image_region_fragment_provisional is None
        assert len(snapshot.entities)==1


@pytest.mark.parametrize('fault',['frame','source','epoch','life','raster','dimensions','integrity','not_valid'])
def test_return_release_rejects_foreign_current_original(tmp_path,monkeypatch,fault):
    """Real admitted owner inputs, then deliberately invalid read-only proof.

    Changed originals are supplied only to the exact predicate under test,
    never passed to World or represented as fresh owner measurements.
    """
    with fixture(tmp_path,monkeypatch) as (f,packet):
        key,old,fact=seeded_return(f,packet)
        f.publish([literal(f,packet,4,fault='one_ear')])
        owner=f.identity.ledger;originals=dict(f.ingress.projective_bridge.image_originals)
        o=originals[key];s=o.stamp
        # Use the literal reliable ears only in this withheld proof, preserving
        # all genuine admitted native/availability receipts from the cohort.
        head=tuple((i,x,y,c) for i,(x,y,c) in enumerate(CHILD[:5]) if c>=.35)
        o=replace(o,head_joints=head)
        if fault=='frame':o=replace(o,stamp=replace(s,frame_id=s.frame_id-1))
        elif fault=='source':o=replace(o,stamp=replace(s,member=replace(s.member,source_id=1)))
        elif fault=='epoch':o=replace(o,stamp=replace(s,member=replace(s.member,source_epoch=s.member.source_epoch+1)))
        elif fault=='life':o=replace(o,stamp=replace(s,lifecycle_started_frame=s.lifecycle_started_frame+1))
        elif fault=='raster':o=replace(o,raster_id='foreign-raster')
        elif fault=='dimensions':o=replace(o,image_size=(1280,720))
        originals[key]=o
        valid={k:True for k in originals};integrity=valid.copy()
        if fault=='integrity':integrity[key]=False
        elif fault=='not_valid':valid[key]=False
        denied=owner._image_provisional_extras(originals,valid,integrity,f.ingress._context,fact.policy)
        assert key in denied and owner._members[key].image_region_fragment_provisional==fact
        assert owner.image_region_error(owner._members[key].observation.stamp,f.ingress._context) is not None
        assert f.identity.stats().pending_reservations==0


def test_every_original_anchor_must_have_current_separated_head(tmp_path,monkeypatch):
    from noesis.pipelines.world_quality_adapter import capture_body_support,freeze_quality_work
    with fixture(tmp_path,monkeypatch) as (f,base):
        def packet(frame,raws=(10,12),**kwargs):
            work,tracks=base(frame,(*raws,14),**kwargs)
            anchor=next(row.evidence.body for row,t in zip(work.rows,tracks) if t['tracker_id']==10)
            inputs=[]
            for row,t in zip(work.rows,tracks):
                body=row.evidence.body
                if t['tracker_id']==14:
                    t['bbox']=[anchor.bbox_xywh[0]+10,*anchor.bbox_xywh[1:]]
                    js=tuple((x+10,y,c) for x,y,c in anchor.keypoints)
                    body=capture_body_support(t,row.evidence.lifecycle,js,pose_is_current=True)
                inputs.append(replace(row.evidence,body=body))
            return replace(freeze_quality_work(work.private,work.continuity,tuple(inputs)),
                ingress_monotonic_s=work.ingress_monotonic_s,owner_availability=work.owner_availability),tracks
        key,old,fact=seeded_return(f,packet)
        assert {s.member.raw_id for s in fact.anchors}=={10,14}
        # Anchor10 is positively separated, but ORIGINAL anchor14 now has a
        # current overlapping head. A first-anchor shortcut must not release.
        snapshot=f.publish([literal(f,packet,4,fault='second_anchor_overlaps')])
        m=f.identity.ledger._members[key]
        assert m.image_region_fragment_provisional==fact
        assert f.identity.ledger.image_region_error(m.observation.stamp,f.ingress._context) is not None
        own=next(e for e in snapshot.entities if e.entity_id==old.entity_id)
        assert (own.observed_at_us,own.stale_after_us)==(old.observed_at_us,old.stale_after_us)
