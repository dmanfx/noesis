"""Approved birth-only recall tradeoff; synthetic anatomy, real public owner chain.

F206 confidence/bbox/torso-midpoint values are from R96. The quartet coordinates
are synthetic supports around those midpoints, not recovered R96 keypoints.
"""
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import time
import pytest


def examples():
    path=Path(__file__).resolve().parents[1]/'DS9/tests/test_xc3d_world_quality_adapter.py'
    spec=spec_from_file_location('provisional_extra_examples',path)
    result=module_from_spec(spec);sys.modules[spec.name]=result;spec.loader.exec_module(result)
    return result


def public_fixture(tmp_path,monkeypatch,*,reverse=False,enabled=True):
    from noesis.pipelines.projective_continuity import ImageRegionQualityBridge
    from noesis.pipelines.world_quality_adapter import capture_body_support,freeze_quality_work
    from noesis.pipelines.source_acquisition import freeze_owner_availability,OriginalSwinCache
    from noesis_core.source_progress import DecodedSourceProgressMonitor,SourceProgressPolicy
    from noesis.identity_association_service import IdentityAssociationService
    from reid.mv3dt_person_association import ProjectiveContinuityPolicy,ProjectiveCameraBinding
    m=examples();clock=[1];epoch=time.time_ns();selected={}
    monkeypatch.setattr(time,'monotonic',lambda:100.+clock[0]/30)
    monkeypatch.setattr(time,'time',lambda:(epoch+clock[0]*33_333_333)/1e9)
    monkeypatch.setattr(time,'time_ns',lambda:int(time.time()*1_000_000)*1000)
    def factory(run,spatial):
        p=((100.,0.,0.,0.),(0.,100.,0.,0.),(0.,0.,1.,0.));ex=m.world_examples()
        selected['policy']=ProjectiveContinuityPolicy(tuple(ProjectiveCameraBinding(b,'extra-birth',
            'raster:'+b.camera_id,(1920,1080),'backend_world_m',ex.frame_binding(b.source_id).target_frame.revision,
            ex.frame_binding(b.source_id).source_camera_calibration_sha256,p,
            ((100.,0.,0.,-10. if b.source_id==1 else 10.),p[1],p[2])) for b in spatial.sources))
        return IdentityAssociationService.for_estimated_image_regions(run,spatial,selected['policy'],
            live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=enabled)
    f=m.NativeQualityFixture(tmp_path,identity_factory=factory)
    f.ingress.projective_bridge=ImageRegionQualityBridge(selected['policy'],OriginalSwinCache())
    monitor=DecodedSourceProgressMonitor(f.labels,policy=SourceProgressPolicy(poll_interval_seconds=60.),clock=time.monotonic)
    monitor.start(lambda failure:None)
    def packet(frame,raws=(10,12),*,candidate='weak',anchor='head',source=2,shadow_raws=()):
        clock[0]=frame
        # Feed genuine public native-presence inputs to the registry BEFORE
        # it freezes retention/caps; never patch owner state after admission.
        update=f.registries[source].update_frame
        with monkeypatch.context() as native:
            if shadow_raws:
                native.setattr(f.registries[source],'update_frame',lambda **kw:update(**dict(kw,shadow_tracker_ids=shadow_raws)))
            work,tracks=f.packet(source,frame,raws=tuple(reversed(raws)) if reverse else raws)
        inputs=[]
        for row,track in zip(work.rows,tracks):
            extra=track['tracker_id']!=10
            u=496.4189709472656 if extra else 494.64683166503903
            y=420.2125 if extra else 420.26502807617186
            track['bbox']=([484.39508056640625,358.26251220703125,30.553571701049805,186.6374969482422]
                if extra else [442.5,356.5,72.,186.])
            width=6. if extra else 24.
            js=[(u,y,.1)]*17
            for i,v in {5:(u-width/2,y-30.,.5224609375 if extra else .99658203125),
                6:(u+width/2,y-30.,.734375 if extra else .9345703125),
                11:(u-width/2,y+30.,.74658203125 if extra else .9775390625),
                12:(u+width/2,y+30.,.7744140625 if extra else .92333984375)}.items():js[i]=v
            if (candidate in {'head','head_unknown','duplicate_head'} if extra else anchor=='head'):
                # The positive head-recovery control is truly separated from
                # the anchor support; confidence alone cannot prove that.
                head_u=(510. if candidate=='head' else 494.64683166503903 if candidate=='duplicate_head' else u) if extra else u
                for i,v in {0:(head_u,375.,.89),1:(head_u-1,373.,.89),3:(head_u-3,373.,.89)}.items():js[i]=v
            if extra and candidate=='different_torso':
                track['bbox'][0]+=200.;js=[(x+200.,yy,c) for x,yy,c in js]
            inputs.append(replace(row.evidence,body=capture_body_support(track,row.evidence.lifecycle,js,pose_is_current=True)))
        mono=time.monotonic();monitor.record_progress(source,now=mono)
        availability=freeze_owner_availability(work.private,work.continuity,monitor,ingress_monotonic_s=mono)
        return replace(freeze_quality_work(work.private,work.continuity,tuple(inputs)),
            ingress_monotonic_s=mono,owner_availability=availability),tracks
    return f,packet,monitor


# Exact retained R108 F216 torso/head coordinates, synthetic fixture-native
# raw12/10 and clocks. These do NOT alias the recording's raw IDs or receipts.
R108_216=((498.1862487792969,381.19842529296875,.02178955078125),(496.5830078125,378.58282470703125,.0166015625),
 (498.69635009765625,377.7835998535156,.0137939453125),(495.562744140625,377.7109375,.0352783203125),
 (498.84210205078125,376.5484313964844,.03460693359375),(501.6842041015625,392.75079345703125,.42236328125),
 (504.96356201171875,391.66094970703125,.470947265625),(517.6437377929688,428.8609313964844,.435546875),
 (518.2996215820312,429.44219970703125,.4482421875),(519.90283203125,459.66717529296875,.155029296875),
 (518.3724975585938,457.48748779296875,.1744384765625),(505.32794189453125,455.7437438964844,.53955078125),
 (506.9311828613281,454.87188720703125,.493896484375),(516.1862182617188,490.6187438964844,.51123046875),
 (518.1538696289062,486.2593688964844,.495361328125),(523.1093139648438,529.8531494140625,.5400390625),
 (521.6517944335938,529.7078247070312,.53759765625))


def native_box_packet(packet,frame,*,clear=False,anchor_pose=False,body_missing=False):
    from noesis.pipelines.world_quality_adapter import capture_body_support,freeze_quality_work
    work,tracks=packet(frame)
    inputs=[]
    for row,t in zip(work.rows,tracks):
        if t['tracker_id']==10:
            t['bbox']=[439.75,357.,97.75,186.]
            body=None
            if anchor_pose:
                js=list(row.evidence.body.keypoints)
                for i,p in {5:(479.152252,393.611725,.982421875),6:(488.931335,392.680481,.970214844),
                    11:(491.725372,448.671875,.978515625),12:(499.874634,447.97345,.981445312)}.items():js[i]=p
                body=capture_body_support(t,row.evidence.lifecycle,js,pose_is_current=True)
        else:
            t['bbox']=[496.5830078125+(200 if clear else 0),377.12969970703125,39.206478118896484,164.0578155517578]
            js=tuple((x+(200 if clear else 0),y,c) for x,y,c in R108_216)
            body=None if body_missing else capture_body_support(t,row.evidence.lifecycle,js,pose_is_current=True)
        inputs.append(replace(row.evidence,body=body))
    return replace(freeze_quality_work(work.private,work.continuity,tuple(inputs)),
        ingress_monotonic_s=work.ingress_monotonic_s,owner_availability=work.owner_availability),tracks


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('anchor_pose',[False,True])
def test_native_box_literal216_provisional_until_positive_clear(tmp_path,monkeypatch,reverse,anchor_pose):
    f,packet,monitor=public_fixture(tmp_path,monkeypatch,reverse=reverse)
    try:
        first=f.publish([packet(1,(10,))]);old=first.entities[0]
        second=f.publish([native_box_packet(packet,2,anchor_pose=anchor_pose)])
        k=next(k for k in f.identity.ledger._members if k.raw_id==12)
        m=f.identity.ledger._members[k]
        assert not m.image_region_birth_established and m.image_region_birth_quarantined
        assert m.image_region_provisional_extra.basis=='current_native_box'
        assert len(second.entities)==1
        fact=m.image_region_provisional_extra
        third=f.publish([native_box_packet(packet,3,body_missing=True)])
        assert f.identity.ledger._members[k].image_region_provisional_extra==fact and len(third.entities)==1
        current= native_box_packet(packet,4,clear=True)
        fourth=f.publish([current]);m=f.identity.ledger._members[k]
        assert m.image_region_provisional_extra is None and m.image_region_birth_established
        own=next(e for e in fourth.entities if e.entity_id.endswith(':'+str(m.person_key.serial)))
        assert own.observed_at_us==next(t['observed_at_us'] for t in current[1] if t['tracker_id']==12)
        assert own.stale_after_us==own.observed_at_us+2_000_000 and old.stale_after_us==old.observed_at_us+2_000_000
        actual=f.examples.actual_node_wire(f,f.wire,'native-box-literal216-'+str(reverse))
        assert [len(s['people']) for s in actual['snapshots']]==[1,1,1,2]
        assert all(d['accepted'] for d in actual['diagnostics']) and f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('fault',['missing','overflow','duplicate','frame','source','epoch','birth','raster','prior',
    'size','ingress','nan','zero_box','shadow','caps_missing','caps_failed','stale'])
def test_native_box_invalid_optional_proof_cannot_create_veto(tmp_path,monkeypatch,fault):
    f,packet,monitor=public_fixture(tmp_path,monkeypatch)
    try:
        f.publish([packet(1,(10,))]);join=f.identity.join_estimated_image_regions
        def changed(proof,**kwargs):
            rows=proof.native_boxes
            anchor=next((b for b in rows if b.stamp.member.raw_id==10),None)
            if anchor is not None:
                s=anchor.stamp
                if fault=='frame':anchor=replace(anchor,stamp=replace(s,frame_id=s.frame_id-1))
                elif fault=='source':anchor=replace(anchor,stamp=replace(s,member=replace(s.member,source_id=1)))
                elif fault=='epoch':anchor=replace(anchor,stamp=replace(s,member=replace(s.member,source_epoch=1)))
                elif fault=='birth':anchor=replace(anchor,stamp=replace(s,lifecycle_started_frame=s.lifecycle_started_frame+1))
                elif fault=='raster':anchor=replace(anchor,raster_id='foreign')
                elif fault=='prior':anchor=replace(anchor,prior_id='foreign')
                elif fault=='size':anchor=replace(anchor,image_size=(1280,720))
                elif fault=='ingress':anchor=replace(anchor,ingress_monotonic_s=anchor.ingress_monotonic_s-.01)
                elif fault=='nan':anchor=replace(anchor,bbox_xywh=(float('nan'),0.,1.,1.))
                elif fault=='zero_box':anchor=replace(anchor,bbox_xywh=(0.,0.,0.,1.))
                rows=tuple(anchor if b.stamp.member.raw_id==10 else b for b in rows)
                if fault=='duplicate':rows=(*rows,anchor)
                elif fault=='overflow':rows=(anchor,)*65
            if fault=='missing':rows=()
            return join(replace(proof,native_boxes=rows),**kwargs)
        monkeypatch.setattr(f.identity,'join_estimated_image_regions',changed)
        if fault=='shadow':
            # Genuine native unmeasured lifecycle, not a privately edited flag.
            work,tracks=packet(2,(12,),shadow_raws=(10,))
        else:work,tracks=native_box_packet(packet,2)
        if fault.startswith('caps_'):work=replace(work,owner_availability=None if fault=='caps_missing'
            else replace(work.owner_availability,status='failed'))
        with monkeypatch.context() as queued:
            if fault=='stale':
                now=time.monotonic();queued.setattr(time,'monotonic',lambda:now+.36)
            f.publish([(work,tracks)])
        member=next(m for k,m in f.identity.ledger._members.items() if k.raw_id==12)
        assert member.image_region_provisional_extra is None
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


def test_native_box_independent_back_head_newcomer_survives(tmp_path,monkeypatch):
    from noesis.pipelines.world_quality_adapter import capture_body_support,freeze_quality_work
    path=Path(__file__).with_name('test_xc3d_back_head_return.py')
    spec=spec_from_file_location('native_box_back_head_literal',path);literal=module_from_spec(spec);spec.loader.exec_module(literal)
    f,packet,monitor=public_fixture(tmp_path,monkeypatch)
    try:
        f.publish([packet(1,(10,))]);work,tracks=native_box_packet(packet,2,anchor_pose=True)
        inputs=[]
        for row,t in zip(work.rows,tracks):
            if t['tracker_id']==12:
                t['bbox']=list(literal.CHILD_BOX);js=literal.CHILD
            else:
                js=list(row.evidence.body.keypoints)
                for i,p in {5:(478.,400.,.9),6:(486.,400.,.9),11:(478.,420.,.9),12:(486.,420.,.9)}.items():js[i]=p
            inputs.append(replace(row.evidence,body=capture_body_support(t,row.evidence.lifecycle,js,pose_is_current=True)))
        work=replace(freeze_quality_work(work.private,work.continuity,tuple(inputs)),
            ingress_monotonic_s=work.ingress_monotonic_s,owner_availability=work.owner_availability)
        result=f.publish([(work,tracks)])
        member=next(m for k,m in f.identity.ledger._members.items() if k.raw_id==12)
        assert member.image_region_birth_established and member.image_region_provisional_extra is None
        assert len(result.entities)==2
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('fault',['missing_anchor','caps_missing','caps_failed'])
def test_native_box_fact_unknown_does_not_release(tmp_path,monkeypatch,fault):
    f,packet,monitor=public_fixture(tmp_path,monkeypatch)
    try:
        f.publish([packet(1,(10,))]);f.publish([native_box_packet(packet,2)])
        k=next(k for k in f.identity.ledger._members if k.raw_id==12)
        fact=f.identity.ledger._members[k].image_region_provisional_extra
        assert fact.basis=='current_native_box'
        work,tracks=(packet(3,(12,),candidate='head',shadow_raws=(10,)) if fault=='missing_anchor'
            else native_box_packet(packet,3,clear=True))
        if fault.startswith('caps_'):work=replace(work,owner_availability=None if fault=='caps_missing'
            else replace(work.owner_availability,status='failed'))
        result=f.publish([(work,tracks)])
        assert f.identity.ledger._members[k].image_region_provisional_extra==fact
        assert len(result.entities)==1 and f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('mode',['unestablished_anchor','default','already_established'])
def test_native_box_does_not_change_other_birth_or_established_policies(tmp_path,monkeypatch,mode):
    f,packet,monitor=public_fixture(tmp_path,monkeypatch,enabled=mode!='default')
    try:
        if mode=='already_established':f.publish([packet(1,candidate='head')])
        elif mode!='unestablished_anchor':f.publish([packet(1,(10,))])
        result=f.publish([native_box_packet(packet,2)])
        assert len(result.entities)==2
        assert all(m.image_region_provisional_extra is None for m in f.identity.ledger._members.values())
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('event',['stop','epoch','termination'])
def test_native_box_fact_has_no_authority_after_native_barrier(tmp_path,monkeypatch,event):
    from reid.mv3dt_person_association import MemberRemoval
    f,packet,monitor=public_fixture(tmp_path,monkeypatch)
    try:
        f.publish([packet(1,(10,))]);f.publish([native_box_packet(packet,2)])
        owner=f.identity.ledger;k=next(k for k in owner._members if k.raw_id==12)
        m=owner._members[k];assert m.image_region_provisional_extra.basis=='current_native_box'
        s=m.observation.stamp;binding=next(b for b in owner.source_bindings() if b.source_id==2)
        if event=='stop':
            assert owner.stop_image_region_source(binding,graph_id=owner._image_graph_id).disposition=='observed'
            assert owner.image_region_error(s,f.ingress._context) is not None
        elif event=='epoch':
            assert owner.reset_source(replace(binding,source_epoch=1)).disposition=='reset'
            assert k not in owner._members
        else:
            assert owner.withdraw(MemberRemoval('native-box-real-end',s,'termination',3,
                s.acquired_at_s+1/30,s.clock_domain,s.media_time_s+1/30)).disposition=='removed'
            assert k not in owner._members
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


def qualified_overlap_packet(packet,frame,*,candidate='overlap',anchor='current',other=False):
    from noesis.pipelines.world_quality_adapter import capture_body_support,freeze_quality_work
    # Exact R113 native1057 child22 / proposal32 CURRENT unclipped head+torso.
    # Native IDs/clocks here are fixture-owned; other joints remain unavailable.
    joints={10:((0,478.302155,450.78125,.831054688),(1,480.381287,448.221863,.915039062),
        (3,489.309357,448.831238,.944824219),(5,489.67627,461.262512,.990722656),
        (6,501.417267,456.875,.921386719),(11,498.604309,491.243744,.963867188),
        (12,506.920868,489.049988,.901855469)),
        12:((0,476.710846,448.984375,.907714844),(1,479.13855,446.757812,.958984375),
        (3,488.771088,447.421875,.905761719),(5,490.963867,460.078125,.975585938),
        (6,486.656616,457.421875,.812988281),(11,489.162659,493.515625,.868652344),
        (12,486.578308,492.03125,.606445312))}
    raws=(12,) if anchor=='shadow' else (10,12,14) if other else (10,12)
    work,tracks=packet(frame,raws,candidate='head',shadow_raws=(10,) if anchor=='shadow' else ())
    inputs=[]
    for row,t in zip(work.rows,tracks):
        raw=t['tracker_id'];key=12 if raw==12 else 10
        t['bbox']=([468.683746,429.335938,31.051205,117.304688] if raw==12
            else [440.,400.,220.,200.] if raw==14 else [470.719421,428.721863,48.798561,122.240623])
        js=[(t['bbox'][0]+1,t['bbox'][1]+1,.1)]*17
        for i,x,y,c in joints[key]:
            js[i]=(x+(100. if raw==14 and i<5 else 0),y,
                .1 if raw==10 and anchor=='head_unknown' and i<5 else c)
        if raw==12 and candidate=='separated_head':
            t['bbox'][2]+=100.;js=[(x+(40. if i<5 else 0),y,c) for i,(x,y,c) in enumerate(js)]
        if raw==12 and candidate=='different_torso':
            t['bbox'][0]+=200.;js=[(x+200.,y,c) for x,y,c in js]
        body=None if raw==10 and anchor=='pose_missing' else capture_body_support(
            t,row.evidence.lifecycle,js,pose_is_current=True)
        inputs.append(replace(row.evidence,body=body))
    return replace(freeze_quality_work(work.private,work.continuity,tuple(inputs)),
        ingress_monotonic_s=work.ingress_monotonic_s,owner_availability=work.owner_availability),tracks


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('other',[False,True])
def test_r113_qualified_but_overlapping_head_is_not_independent(tmp_path,monkeypatch,reverse,other):
    f,packet,monitor=public_fixture(tmp_path,monkeypatch,reverse=reverse)
    try:
        n=2 if other else 1
        first=f.publish([packet(1,(10,14) if other else (10,),candidate='head')])
        second=f.publish([qualified_overlap_packet(packet,2,other=other)])
        k=next(k for k in f.identity.ledger._members if k.raw_id==12)
        m=f.identity.ledger._members[k]
        assert not m.image_region_birth_established and m.image_region_birth_quarantined
        fact=m.image_region_provisional_extra
        assert fact.basis=='current_torso' and {s.member.raw_id for s in fact.anchors}=={10}
        assert len(second.entities)==n
        f.publish([qualified_overlap_packet(packet,3,anchor='pose_missing',other=other)])
        assert f.identity.ledger._members[k].image_region_provisional_extra==fact
        final_packet=qualified_overlap_packet(packet,4,candidate='separated_head',other=other)
        final=f.publish([final_packet])
        m=f.identity.ledger._members[k]
        assert m.image_region_birth_established and m.image_region_provisional_extra is None
        own=next(e for e in final.entities if e.entity_id.endswith(':'+str(m.person_key.serial)))
        assert own.observed_at_us==next(t['observed_at_us'] for t in final_packet[1] if t['tracker_id']==12)
        assert own.stale_after_us==own.observed_at_us+2_000_000
        assert all(e.stale_after_us==e.observed_at_us+2_000_000 for e in first.entities)
        actual=f.examples.actual_node_wire(f,f.wire,'r113-qualified-'+str(reverse)+str(other))
        assert [len(s['people']) for s in actual['snapshots']]==[n,n,n,n+1]
        assert all(d['accepted'] for d in actual['diagnostics']) and f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('mode',['separated_head','different_torso','head_unknown','pose_missing',
    'shadow','not_established','default','stale_delivery'])
def test_qualified_head_creation_preserves_current_independent_or_unknown_cases(tmp_path,monkeypatch,mode):
    f,packet,monitor=public_fixture(tmp_path,monkeypatch,enabled=mode!='default')
    try:
        if mode!='not_established':f.publish([packet(1,(10,))])
        work,tracks=qualified_overlap_packet(packet,2,
            candidate=mode if mode in {'separated_head','different_torso'} else 'overlap',
            anchor=mode if mode in {'head_unknown','pose_missing','shadow'} else 'current')
        if mode=='stale_delivery':monkeypatch.setattr(time,'monotonic',lambda:work.ingress_monotonic_s+.36)
        f.publish([(work,tracks)])
        m=next(v for k,v in f.identity.ledger._members.items() if k.raw_id==12)
        assert m.image_region_provisional_extra is None
        if mode not in {'stale_delivery','default'}:assert m.image_region_birth_established
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('unknown',[False,True])
def test_r111_torso_fact_cannot_release_into_current_native_box(tmp_path,monkeypatch,reverse,unknown):
    from noesis.pipelines.world_quality_adapter import capture_body_support,freeze_quality_work
    # Literal R111 CURRENT 216/219 head+torso; fixture native IDs and clocks
    # are synthetic. Unused joints are unavailable, not invented leg support.
    poses={
        (2,10):[(0,461.845703,375.479675,.925292969),(1,464.391693,371.837494,.926269531),
            (3,473.881317,372.993744,.91015625),(5,494.133545,391.320312,.989257812),
            (6,468.789307,396.7547,.982910156),(11,500.961426,448.265625,.985839844),
            (12,486.264099,450.,.986816406)],
        (2,12):[(5,501.684204,392.823425,.419921875),(6,504.890686,391.733582,.470458984),
            (11,505.327942,455.743744,.536621094),(12,506.858307,454.871887,.491943359)],
        (3,10):[(0,459.531158,374.439056,.711914062),(1,461.845703,370.392181,.607421875),
            (3,470.17804,372.935944,.862792969),(5,470.525208,397.332825,.992675781),
            (6,497.605347,391.898438,.974609375),(11,486.379822,450.115631,.982421875),
            (12,506.979218,446.53125,.981933594)],
        (3,12):[(4,566.748047,359.1297,.653320312),(5,547.403076,385.904694,.853515625),
            (6,556.775208,389.31134,.816894531),(11,552.810059,465.512512,.833496094),
            (12,562.302307,468.381256,.765625)]}
    f,packet,monitor=public_fixture(tmp_path,monkeypatch,reverse=reverse)
    try:
        first=f.publish([packet(1,(10,14),candidate='head')])
        def current(frame,*,clear=False,missing=False):
            work,tracks=packet(frame,(10,12) if missing else (10,12,14),shadow_raws=(14,) if missing else ());inputs=[]
            for row,t in zip(work.rows,tracks):
                raw=t['tracker_id'];body=None
                t['bbox']=([439.75,357.,97.75,186.] if raw==10 else [528.5,339.5,88.,232.5]
                    if raw==14 else [496.583008,377.057037,39.2064781,164.130463]
                    if frame==2 else [527.937988,338.68985,47.220932,192.803909])
                if raw!=14:
                    js=[(t['bbox'][0]+1,t['bbox'][1]+1,.1)]*17
                    for i,x,y,c in poses[(min(frame,3),raw)]:js[i]=(x,y,c)
                    if clear and raw==12:
                        t['bbox'][0]+=200.;js=[(x+200.,y,c) for x,y,c in js]
                    body=capture_body_support(t,row.evidence.lifecycle,js,pose_is_current=True)
                inputs.append(replace(row.evidence,body=body))
            return replace(freeze_quality_work(work.private,work.continuity,tuple(inputs)),
                ingress_monotonic_s=work.ingress_monotonic_s,owner_availability=work.owner_availability),tracks
        f.publish([current(2)])
        k=next(k for k in f.identity.ledger._members if k.raw_id==12)
        old=f.identity.ledger._members[k].image_region_provisional_extra
        assert old is not None and old.basis=='current_torso'
        migrated=f.publish([current(3)])
        m=f.identity.ledger._members[k]
        assert not m.image_region_birth_established and m.image_region_birth_quarantined
        assert m.image_region_provisional_extra.original==old.original
        assert m.image_region_provisional_extra.basis==old.basis
        assert m.image_region_provisional_extra.anchors==old.anchors
        assert {s.member.raw_id for s in m.image_region_provisional_extra.box_anchors}=={10,14}
        assert len(migrated.entities)==2
        if unknown:
            f.publish([current(4,clear=True,missing=True)])
            assert not f.identity.ledger._members[k].image_region_birth_established
            actual=f.examples.actual_node_wire(f,f.wire,'r111-mixed-unknown-'+str(reverse))
            assert [len(s['people']) for s in actual['snapshots']]==[2,2,2,2]
            assert all(d['accepted'] for d in actual['diagnostics']) and f.identity.stats().pending_reservations==0
            return
        final_packet=current(5 if unknown else 4,clear=True)
        final=f.publish([final_packet])
        assert f.identity.ledger._members[k].image_region_birth_established and len(final.entities)==3
        own=next(e for e in final.entities if e.entity_id.endswith(':'+str(m.person_key.serial)))
        assert own.observed_at_us==next(t['observed_at_us'] for t in final_packet[1] if t['tracker_id']==12)
        assert own.stale_after_us==own.observed_at_us+2_000_000
        assert all(e.stale_after_us==e.observed_at_us+2_000_000 for e in first.entities)
        actual=f.examples.actual_node_wire(f,f.wire,'r111-mixed-'+str(reverse)+str(unknown))
        assert [len(s['people']) for s in actual['snapshots']]==([2,2,2,2,3] if unknown else [2,2,2,3])
        assert all(d['accepted'] for d in actual['diagnostics']) and f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


def test_native_box_migration_cannot_escape_into_another_current_anchor(tmp_path,monkeypatch):
    from noesis.pipelines.world_quality_adapter import capture_body_support,freeze_quality_work
    f,packet,monitor=public_fixture(tmp_path,monkeypatch)
    try:
        f.publish([packet(1,(10,14),candidate='head')])
        def current(frame,shift):
            work,tracks=packet(frame,(10,12,14));inputs=[]
            for row,t in zip(work.rows,tracks):
                raw=t['tracker_id'];body=None
                if raw==12:
                    t['bbox']=[496.5830078125+shift,377.12969970703125,39.206478118896484,164.0578155517578]
                    body=capture_body_support(t,row.evidence.lifecycle,tuple((x+shift,y,c) for x,y,c in R108_216),pose_is_current=True)
                else:t['bbox']=[439.75 if raw==10 else 650.,357.,97.75,186.]
                inputs.append(replace(row.evidence,body=body))
            return replace(freeze_quality_work(work.private,work.continuity,tuple(inputs)),
                ingress_monotonic_s=work.ingress_monotonic_s,owner_availability=work.owner_availability),tracks
        f.publish([current(2,0)])
        k=next(k for k in f.identity.ledger._members if k.raw_id==12)
        original=f.identity.ledger._members[k].image_region_provisional_extra.original
        assert {s.member.raw_id for s in f.identity.ledger._members[k].image_region_provisional_extra.anchors}=={10}
        migrated=f.publish([current(3,200)])
        m=f.identity.ledger._members[k]
        assert not m.image_region_birth_established and m.image_region_provisional_extra.original==original
        assert {s.member.raw_id for s in m.image_region_provisional_extra.anchors}=={10,14}
        assert len(migrated.entities)==2
        clear=f.publish([current(4,400)])
        assert f.identity.ledger._members[k].image_region_provisional_extra is None
        assert f.identity.ledger._members[k].image_region_birth_established and len(clear.entities)==3
        actual=f.examples.actual_node_wire(f,f.wire,'native-box-migration')
        assert [len(s['people']) for s in actual['snapshots']]==[2,2,2,3]
        assert all(d['accepted'] for d in actual['diagnostics']) and f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('reverse',[False,True])
def test_extra_birth_public_current_overlap_is_provisional_then_releases(tmp_path,monkeypatch,reverse):
    f,packet,monitor=public_fixture(tmp_path,monkeypatch,reverse=reverse)
    try:
        first=f.publish([packet(1,(10,))]);old=first.entities[0]
        suppressed=f.publish([packet(2)])
        member=next(v for v in f.identity.ledger._members.values() if v.observation.stamp.member.raw_id==12)
        assert not member.image_region_birth_established and member.image_region_birth_quarantined
        assert f.identity.ledger.image_region_error(member.observation.stamp,f.ingress._context)=='image_region_ambiguous_member'
        assert len(suppressed.entities)==1
        # No measurement/lease renewal is hidden in provisional status.
        assert old.stale_after_us==old.observed_at_us+2_000_000
        released_packet=packet(3,candidate='head')
        released=f.publish([released_packet])
        current=next(e for e in released.entities if e.entity_id.endswith(':'+str(member.person_key.serial)))
        assert current.observed_at_us==next(t['observed_at_us'] for t in released_packet[1] if t['tracker_id']==12)
        assert current.stale_after_us==current.observed_at_us+2_000_000
        actual=f.examples.actual_node_wire(f,f.wire,'extra-provisional-'+str(reverse))
        assert [len(s['people']) for s in actual['snapshots']]==[1,1,2]
        assert all(d['accepted'] for d in actual['diagnostics'])
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('mode',['head','head_unknown','duplicate_head','different_torso','established','two_weak','new_anchor','default'])
@pytest.mark.parametrize('reverse',[False,True])
def test_extra_birth_preserves_recall_and_order_controls(tmp_path,monkeypatch,mode,reverse):
    f,packet,monitor=public_fixture(tmp_path,monkeypatch,reverse=reverse,enabled=mode!='default')
    try:
        if mode=='established':f.publish([packet(1,candidate='head')])
        elif mode not in {'two_weak','new_anchor'}:f.publish([packet(1,(10,))])
        candidate=mode if mode in {'head','head_unknown','duplicate_head','different_torso'} else 'weak'
        final=f.publish([packet(2,candidate=candidate,anchor='weak' if mode=='two_weak' else 'head')])
        # These historical fixture names both supply fully qualified CURRENT
        # triplets positively overlapping the established anchor's head.
        # Qualification alone is no longer independent-anatomy evidence.
        overlapping_head=mode in {'head_unknown','duplicate_head'}
        assert len(final.entities)==(1 if mode=='new_anchor' or overlapping_head else 2)
        if overlapping_head:
            m=next(v for k,v in f.identity.ledger._members.items() if k.raw_id==12)
            assert not m.image_region_birth_established and m.image_region_birth_quarantined
            assert m.image_region_provisional_extra.basis=='current_torso'
            assert {s.member.raw_id for s in m.image_region_provisional_extra.anchors}=={10}
        if mode=='established':
            assert all(v.image_region_birth_established for v in f.identity.ledger._members.values())
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('fault',['frame','media','acquisition','epoch','raster','dimensions',
    'invalid','not_valid','partial_torso','clipped_torso','collinear_far','caps_missing','caps_failed','stop'])
def test_extra_birth_requires_exact_positive_current_competition(tmp_path,monkeypatch,fault):
    """Direct predicate controls use real previously validated public inputs.

    Deliberately changed inputs are never submitted as grants/World originals.
    Missing/bad current proof must NOT become evidence to suppress a newcomer.
    """
    f,packet,monitor=public_fixture(tmp_path,monkeypatch)
    try:
        f.publish([packet(1,(10,))])
        work,tracks=packet(2)
        if fault in {'caps_missing','caps_failed'}:
            work=replace(work,owner_availability=None if fault=='caps_missing'
                else replace(work.owner_availability,status='failed'))
        f.publish([(work,tracks)])
        owner=f.identity.ledger
        originals=dict(f.ingress.projective_bridge.image_originals)
        valid={k:True for k in originals};integrity=valid.copy()
        k=next(k for k in originals if k.raw_id==12);o=originals[k]
        if fault in {'frame','media','acquisition','epoch'}:
            s=o.stamp
            if fault=='frame':s=replace(s,frame_id=s.frame_id-1)
            elif fault=='media':s=replace(s,media_time_s=s.media_time_s-.01)
            elif fault=='acquisition':s=replace(s,acquired_at_s=s.acquired_at_s-.01)
            else:s=replace(s,member=replace(s.member,source_epoch=s.member.source_epoch+1))
            originals[k]=replace(o,stamp=s)
        elif fault=='raster':originals[k]=replace(o,raster_id='other-raster')
        elif fault=='dimensions':originals[k]=replace(o,image_size=(1280,720))
        elif fault=='invalid':integrity[k]=False
        elif fault=='not_valid':valid[k]=False
        elif fault=='partial_torso':originals[k]=replace(o,semantic_joints=o.semantic_joints[:2])
        elif fault=='clipped_torso':originals[k]=replace(o,semantic_joints=tuple(
            (i,o.bbox_xywh[0],y,c) for i,x,y,c in o.semantic_joints))
        elif fault=='collinear_far':
            # Collinear disjoint segments do NOT intersect merely because
            # orientation products are zero; no UNKNOWN-as-overlap shortcut.
            a=next(v for key,v in originals.items() if key.raw_id==10)
            originals[k]=replace(o,bbox_xywh=(a.bbox_xywh[0],600.,a.bbox_xywh[2],200.),
                semantic_joints=tuple((i,x,y+300.,c) for i,x,y,c in a.semantic_joints))
        elif fault=='stop':
            binding=next(b for b in owner.policy.sources if b.source_id==2)
            assert owner.stop_image_region_source(binding,graph_id=owner._image_graph_id).disposition=='observed'
        fact=owner._members[k].image_region_provisional_extra
        result=owner._image_provisional_extras(originals,valid,integrity,f.ingress._context,
            fact.policy if fact is not None else 'no-qualified-original',
            tuple(f.ingress.projective_bridge.native_boxes.values()))
        # Bad/missing inputs cannot CREATE competition, nor erase an already
        # observed negative fact. They are not independent anatomy.
        released=fault=='collinear_far'  # Genuine positive full-torso separation.
        assert result==(set() if fact is None or released else {k})
        assert owner._members[k].image_region_provisional_extra==(None if released else fact)
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


def test_extra_birth_body_uncertainty_is_not_an_independent_join(tmp_path,monkeypatch):
    f,packet,monitor=public_fixture(tmp_path,monkeypatch)
    try:
        f.publish([packet(1,(10,))])
        f.publish([packet(2),packet(2,(12,),source=1)])
        owner=f.identity.ledger
        extra=next(v for v in owner._members.values()
            if v.observation.stamp.member.source_id==2 and v.observation.stamp.member.raw_id==12)
        assert not extra.image_region_birth_established
        assert len(owner._persons[extra.person_key])==1
        diagnostic=owner.image_region_decision_diagnostics()['transitions'][-1]
        comparisons=[p for p in diagnostic['pairs'] if p['right']['raw_id']==12]
        assert comparisons and all(not p['supported'] for p in comparisons)
        assert any(p['possible'] for p in comparisons)  # Not invented FALSE/excluded competitor.
        assert owner.image_region_error(extra.observation.stamp,f.ingress._context) is not None
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('uncertain',['candidate_missing','candidate_partial','anchor_missing',
    'anchor_partial','caps_missing','caps_failed','touching'])
@pytest.mark.parametrize('release',['head','different_torso'])
def test_extra_birth_provisional_until_positive_independent_anatomy(tmp_path,monkeypatch,uncertain,release):
    from noesis.pipelines.world_quality_adapter import capture_body_support,freeze_quality_work
    f,packet,monitor=public_fixture(tmp_path,monkeypatch)
    def extra():return next(v for v in f.identity.ledger._members.values()
        if v.observation.stamp.member.raw_id==12)
    try:
        f.publish([packet(1,(10,))]);f.publish([packet(2)])
        fact=extra().image_region_provisional_extra
        assert fact is not None and fact.original.frame_id==2
        assert tuple(s.member.raw_id for s in fact.anchors)==(10,)
        for frame in (3,4):
            work,tracks=packet(frame);inputs=[]
            for row,track in zip(work.rows,tracks):
                raw=track['tracker_id'];body=row.evidence.body
                if (uncertain=='candidate_missing' and raw==12) or (uncertain=='anchor_missing' and raw==10):body=None
                if (uncertain=='candidate_partial' and raw==12) or (uncertain=='anchor_partial' and raw==10):
                    js=list(body.keypoints);x,y,_=js[11];js[11]=(x,y,.1)
                    body=capture_body_support(track,row.evidence.lifecycle,js,pose_is_current=True)
                if uncertain=='touching' and raw==12:
                    # Exact positive separation requires >, not >=. Synthetic
                    # torso centers/half-widths: 100/12 and115/3 are tangent.
                    js=list(body.keypoints)
                    for i,x,y in ((5,112.,390.),(6,118.,390.),(11,112.,450.),(12,118.,450.)):js[i]=(x,y,.9)
                    track['bbox']=[95.,350.,40.,200.]
                    body=capture_body_support(track,row.evidence.lifecycle,js,pose_is_current=True)
                elif uncertain=='touching' and raw==10:
                    js=list(body.keypoints)
                    for i,x,y in ((5,88.,390.),(6,112.,390.),(11,88.,450.),(12,112.,450.)):js[i]=(x,y,.9)
                    track['bbox']=[70.,350.,60.,200.]
                    body=capture_body_support(track,row.evidence.lifecycle,js,pose_is_current=True)
                inputs.append(replace(row.evidence,body=body))
            changed=freeze_quality_work(work.private,work.continuity,tuple(inputs))
            availability=(None if uncertain=='caps_missing' else replace(work.owner_availability,status='failed')
                if uncertain=='caps_failed' else work.owner_availability)
            work=replace(changed,ingress_monotonic_s=work.ingress_monotonic_s,owner_availability=availability)
            f.publish([(work,tracks)])
            assert extra().image_region_provisional_extra==fact  # No renewal/replacement.
            assert not extra().image_region_birth_established and extra().image_region_birth_quarantined
            assert f.identity.ledger.image_region_error(extra().observation.stamp,f.ingress._context) is not None
        released=f.publish([packet(5,candidate=release)])
        assert extra().image_region_provisional_extra is None
        assert extra().image_region_birth_established and not extra().image_region_birth_quarantined
        assert len(released.entities)==2
        actual=f.examples.actual_node_wire(f,f.wire,'latched-'+uncertain+'-'+release)
        assert [len(s['people']) for s in actual['snapshots']]==[1,1,1,1,2]
        assert all(d['accepted'] for d in actual['diagnostics'])
        assert all(e.stale_after_us==e.observed_at_us+2_000_000 for e in released.entities)
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('reset',['termination_new_birth','source_epoch','run','policy'])
def test_extra_birth_negative_fact_does_not_cross_lifecycle_or_policy(tmp_path,monkeypatch,reset):
    from reid.mv3dt_person_association import MemberRemoval,SpatialLifecycleReceipt
    f,packet,monitor=public_fixture(tmp_path,monkeypatch)
    try:
        f.publish([packet(1,(10,))]);f.publish([packet(2)])
        owner=f.identity.ledger
        m=next(v for v in owner._members.values() if v.observation.stamp.member.raw_id==12)
        s=m.observation.stamp;fact=m.image_region_provisional_extra
        assert fact is not None
        if reset=='termination_new_birth':
            result=owner.withdraw(MemberRemoval('native-termination',s,'termination',3,
                s.acquired_at_s+1/30,s.clock_domain,s.media_time_s+1/30))
            assert result.disposition=='removed' and s.member not in owner._members
            new=replace(s,member=replace(s.member,lifecycle_generation=s.member.lifecycle_generation+1),
                frame_id=4,lifecycle_started_frame=4,acquired_at_s=s.acquired_at_s+2/30,
                media_time_s=s.media_time_s+2/30,observation_id='genuine-new-native-birth')
            result=owner.observe_spatial(replace(m.observation,stamp=new),SpatialLifecycleReceipt('native-new',new,'ACTIVE'),
                quarantine_unresolved_birth=True,retain_image_membership=True,image_source_local_reactivation=True)
            assert result.disposition=='observed'
            assert owner._members[new.member].image_region_provisional_extra is None
            assert not owner._members[new.member].image_region_birth_established
        elif reset=='source_epoch':
            b=next(b for b in owner.source_bindings() if b.source_id==2)
            assert owner.reset_source(replace(b,source_epoch=b.source_epoch+1)).disposition=='reset'
            assert not any(k.source_id==2 for k in owner._members)
        elif reset=='run':
            assert owner.reset_run('genuinely-new-run').disposition=='reset'
            assert not owner._members
        else:
            originals=dict(f.ingress.projective_bridge.image_originals)
            unavailable={k:False for k in originals}
            assert owner._image_provisional_extras(originals,unavailable,unavailable,f.ingress._context,
                'different-accepted-policy')==set()
            now=owner._members[s.member]
            assert now.image_region_provisional_extra is None and not now.image_region_birth_established
            assert now.image_region_birth_quarantined  # Invalidating a negative fact grants nothing.
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


def test_extra_birth_unavailable_native_anchor_is_not_independence(tmp_path,monkeypatch):
    f,packet,monitor=public_fixture(tmp_path,monkeypatch)
    try:
        f.publish([packet(1,(10,))]);f.publish([packet(2)])
        owner=f.identity.ledger
        key=next(k for k in owner._members if k.raw_id==12)
        fact=owner._members[key].image_region_provisional_extra
        # Native owner honestly classifies missing10 as retained SHADOW; the
        # candidate even moves away, but absent anchor cannot prove separation.
        missing=packet(3,(12,),candidate='different_torso',shadow_raws=(10,))
        assert missing[0].continuity.shadow_lifecycles==((10,fact.anchors[0].member.lifecycle_generation),)
        f.publish([missing])
        assert owner._members[key].image_region_provisional_extra==fact
        assert not owner._members[key].image_region_birth_established
        assert owner.image_region_error(owner._members[key].observation.stamp,f.ingress._context) is not None
        assert len(f.world.current_snapshot().entities)==1
        f.publish([packet(4,(12,),candidate='head',shadow_raws=(10,))])
        assert owner._members[key].image_region_provisional_extra==fact
        assert not owner._members[key].image_region_birth_established
        assert len(f.world.current_snapshot().entities)==1
        # Only a genuine CURRENT same-birth anchor return can support positive
        # independent head separation; its earlier shadow is not a substitute.
        f.publish([packet(5,(10,12),candidate='head')])
        assert owner._members[key].image_region_provisional_extra is None
        assert owner._members[key].image_region_birth_established
        assert len(f.world.current_snapshot().entities)==2
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('reverse',[False,True])
def test_extra_birth_reliable_but_nonindependent_head_cannot_release(tmp_path,monkeypatch,reverse):
    f,packet,monitor=public_fixture(tmp_path,monkeypatch,reverse=reverse)
    try:
        f.publish([packet(1,(10,))]);f.publish([packet(2)])
        owner=f.identity.ledger;key=next(k for k in owner._members if k.raw_id==12)
        fact=owner._members[key].image_region_provisional_extra
        for frame,head in ((3,'head_unknown'),(4,'duplicate_head')):
            f.publish([packet(frame,candidate=head)])
            assert owner._members[key].image_region_provisional_extra==fact
            assert not owner._members[key].image_region_birth_established
            assert owner.image_region_error(owner._members[key].observation.stamp,f.ingress._context) is not None
        f.publish([packet(5,candidate='head')])
        assert owner._members[key].image_region_provisional_extra is None
        actual=f.examples.actual_node_wire(f,f.wire,'positive-head-separation-'+str(reverse))
        assert [len(s['people']) for s in actual['snapshots']]==[1,1,1,1,2]
        assert all(d['accepted'] for d in actual['diagnostics'])
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


def test_extra_birth_missing_anchor_head_is_unknown_but_torso_release_unchanged(tmp_path,monkeypatch):
    f,packet,monitor=public_fixture(tmp_path,monkeypatch)
    try:
        f.publish([packet(1,(10,))]);f.publish([packet(2)])
        owner=f.identity.ledger;key=next(k for k in owner._members if k.raw_id==12)
        fact=owner._members[key].image_region_provisional_extra
        f.publish([packet(3,candidate='head',anchor='weak')])
        assert owner._members[key].image_region_provisional_extra==fact
        assert not owner._members[key].image_region_birth_established
        f.publish([packet(4,candidate='different_torso',anchor='weak')])
        assert owner._members[key].image_region_provisional_extra is None
        assert owner._members[key].image_region_birth_established
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()
