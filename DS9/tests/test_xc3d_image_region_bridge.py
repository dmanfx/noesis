"""NEW image adapter checks. Integration positives use SYNTHETIC geometry only."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import importlib.util
import sys,time,json
import pytest
from noesis.pipelines.projective_continuity import (
    ImageRegionQualityBridge,full_observed_image_regions,image_regions_enabled,runtime_image_region_policy)
from noesis.pipelines.world_quality_adapter import capture_body_support,freeze_quality_work
from noesis.identity_association_service import IdentityAssociationService
from reid.mv3dt_person_association import ProjectiveContinuityPolicy,ProjectiveCameraBinding

def helpers():
    path=Path(__file__).with_name('test_xc3d_world_quality_adapter.py')
    spec=importlib.util.spec_from_file_location('new_image_bridge_helpers',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


def test_native_box_proof_is_current_detached_and_not_anatomy(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[2]/'tests/test_xc3d_provisional_extra_birth.py'
    spec=importlib.util.spec_from_file_location('native_box_public_examples',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    f,packet,monitor=m.public_fixture(tmp_path,monkeypatch)
    try:
        f.publish([packet(1,(10,))]);seen=[];join=f.identity.join_estimated_image_regions
        def capture(proof,**kwargs):seen.append(proof);return join(proof,**kwargs)
        monkeypatch.setattr(f.identity,'join_estimated_image_regions',capture)
        work,tracks=m.native_box_packet(packet,2)
        result=f.publish([(work,tracks)]);proof=seen[-1]
        anchor=next(b for b in proof.native_boxes if b.stamp.member.raw_id==10)
        assert anchor.stamp.frame_id==2 and anchor.ingress_monotonic_s==work.ingress_monotonic_s
        assert anchor.bbox_xywh==(439.75,357.,97.75,186.) and anchor.image_size==(1920,1080)
        assert anchor.stamp not in tuple(o.stamp for o in proof.originals)  # No fabricated pose.
        next(t for t in tracks if t['tracker_id']==10)['bbox'][0]=1.
        assert anchor.bbox_xywh==(439.75,357.,97.75,186.) and len(result.entities)==1
        f.publish([m.native_box_packet(packet,3)])
        assert anchor.stamp.frame_id==2 and all(b.stamp.frame_id==3 for b in seen[-1].native_boxes)
        f.ingress.projective_bridge.stop_source(2)
        assert not f.ingress.projective_bridge.native_boxes
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()

def test_default_off_and_exclusive_configuration():
    assert image_regions_enabled({}) is False
    with pytest.raises(ValueError):image_regions_enabled({'xc3d_image_regions':{'enabled':True}})
    from noesis_core.world.fusion import spatial_rejection_category
    assert spatial_rejection_category('image_region_current_support_required')=='no_vote'
    assert spatial_rejection_category('image_region_noncurrent_member')=='integrity'


@pytest.mark.parametrize('excluded',[False,True])
def test_i159_real_partial_original_negative_birth_node(tmp_path,monkeypatch,excluded):
    m=helpers()
    f,packet,monitor=m.joint_appearance_native_fixture(tmp_path,monkeypatch,features=False)
    try:
        first=f.publish([packet(2,1)])
        assert len(first.entities)==2
        work,tracks=packet(1,2,separate=excluded)
        inputs=[]
        for row in work.rows:
            body=row.evidence.body
            joints=tuple((x,y,.1 if i in (11,12) else confidence)
                for i,(x,y,confidence) in enumerate(body.keypoints))
            inputs.append(replace(row.evidence,body=replace(body,keypoints=joints)))
        work=replace(work,rows=tuple(replace(row,evidence=e) for row,e in zip(work.rows,inputs)))
        final=f.publish([(work,tracks)])
        actual=f.examples.actual_node_wire(f,f.wire,'i159-partial-negative-'+str(excluded))
        assert [len(s['people']) for s in actual['snapshots']]==[2,3 if excluded else 2]
        assert all(d['accepted'] for d in actual['diagnostics'])
        assert len(final.entities)==(3 if excluded else 2)
        original=next(v for v in f.ingress.projective_bridge.image_originals.values()
            if v.stamp.member.source_id==1)
        assert tuple(j[0] for j in original.semantic_joints)==(5,6)
        assert original.stamp.frame_id==2 and original.stamp.media_time_s==work.private.media_pts_ns/1e9
        assert all(len(v.members)==1 for v in f.identity.spatial_memberships(at_interval=f.ingress._context))
        if excluded:
            current=next(e for e in final.entities if any(s.accepted and s.camera_id=='kitchen' for s in e.sources))
            assert current.observed_at_us==tracks[0]['observed_at_us']
            assert current.stale_after_us==current.observed_at_us+2_000_000
            assert (current.position.x,current.position.z)==(1.,2.)
            assert current.position==next(s.position for s in current.sources if s.accepted)
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()


@pytest.mark.parametrize('mode',['separate','weak_head','head_edge','duplicate_pending'])
@pytest.mark.parametrize('reverse',[False,True])
def test_local_occupancy_public_native_original_world_node(tmp_path,monkeypatch,mode,reverse):
    from noesis.pipelines.source_acquisition import freeze_owner_availability,OriginalSwinCache
    from noesis_core.source_progress import DecodedSourceProgressMonitor,SourceProgressPolicy
    m=helpers();clock=[1];epoch=time.time_ns()
    monkeypatch.setattr(time,'monotonic',lambda:100.+clock[0]/30)
    monkeypatch.setattr(time,'time',lambda:(epoch+clock[0]*33_333_333)/1e9)
    monkeypatch.setattr(time,'time_ns',lambda:int(time.time()*1_000_000)*1000)
    selected={}
    def factory(run,spatial):
        p=((100.,0.,0.,0.),(0.,100.,0.,0.),(0.,0.,1.,0.));ex=m.world_examples()
        selected['policy']=ProjectiveContinuityPolicy(tuple(ProjectiveCameraBinding(b,'local-occupancy',
            'raster:'+b.camera_id,(1920,1080),'backend_world_m',ex.frame_binding(b.source_id).target_frame.revision,
            ex.frame_binding(b.source_id).source_camera_calibration_sha256,p,
            ((100.,0.,0.,-10. if b.source_id==1 else 10.),p[1],p[2])) for b in spatial.sources))
        return IdentityAssociationService.for_estimated_image_regions(run,spatial,selected['policy'],
            live_track_continuity=True,semantic_joint_max_mean_px=5.,unresolved_birth_quarantine=True)
    f=m.NativeQualityFixture(tmp_path,identity_factory=factory)
    f.ingress.projective_bridge=ImageRegionQualityBridge(selected['policy'],OriginalSwinCache())
    monitor=DecodedSourceProgressMonitor(f.labels,policy=SourceProgressPolicy(poll_interval_seconds=60.),clock=time.monotonic)
    monitor.start(lambda failure:None)
    def packet(source,frame,raws,*,exclude_old=False):
        clock[0]=frame;work,tracks=f.packet(source,frame,raws=tuple(reversed(raws)) if reverse else raws,
            scale=4. if source==1 else 1.)
        inputs=[]
        for row,track in zip(work.rows,tracks):
            raw=track['tracker_id'];u=50. if source==1 else 40. if raw==9 else 10.
            track['bbox']=[u-8,15.,16.,40.]
            js=[(u,30.,.1)]*17
            for i,v in {0:(u,23.,.9),1:(u-1,21.,.9),3:(u-3,21.,.9),
                5:(u-2,30.,.9),6:(u+2,30.,.9),11:(u-2,42.,.9),12:(u+2,42.,.9)}.items():js[i]=v
            if exclude_old and source==2 and raw==9:
                track['bbox'][1]+=20.
                js=[(x,y+20.,confidence) for x,y,confidence in js]
            if mode=='weak_head' and source==2 and raw==10:
                for i in (0,1,3):js[i]=(js[i][0],js[i][1],.1)
            if mode=='head_edge' and source==2 and raw==10:js[0]=(1920.,js[0][1],.9)
            inputs.append(replace(row.evidence,body=capture_body_support(track,row.evidence.lifecycle,js,pose_is_current=True)))
        mono=time.monotonic();monitor.record_progress(source,now=mono)
        availability=freeze_owner_availability(work.private,work.continuity,monitor,ingress_monotonic_s=mono)
        return replace(freeze_quality_work(work.private,work.continuity,tuple(inputs)),
            ingress_monotonic_s=mono,owner_availability=availability),tracks
    try:
        assert len(f.publish([packet(2,1,(9,))]).entities)==1
        assert len(f.publish([packet(1,2,(9,))]).entities)==1
        raws=(9,10,11) if mode=='duplicate_pending' else (9,10)
        family=packet(2,3,raws);kitchen=packet(1,3,(9,))
        final=f.publish([family,kitchen] if reverse else [kitchen,family])
        actual=f.examples.actual_node_wire(f,f.wire,'local-occupancy-'+mode+str(reverse))
        assert [len(s['people']) for s in actual['snapshots']]==[1,1,2 if mode=='separate' else 1]
        assert all(d['accepted'] for d in actual['diagnostics'])
        if mode=='separate':
            member=next(v for v in f.identity.ledger._members.values() if v.observation.stamp.member.source_id==2 and v.observation.stamp.member.raw_id==10)
            memberships=f.identity.spatial_memberships(at_interval=f.ingress._context)
            eligibility=next(e for dto in memberships for e in dto.eligibility if e.observation.member==member.observation.stamp.member)
            assert eligibility.eligible and eligibility.reason=='current_active_own_source'
            assert f.identity.ledger.image_region_error(member.observation.stamp,f.ingress._context) is None
            assert member.image_region_vote_blocked_stamp is None and not member.image_region_vote_pending
            kitchen_eligibility=next(e for dto in memberships for e in dto.eligibility if e.observation.member.source_id==1)
            assert not kitchen_eligibility.eligible and kitchen_eligibility.reason=='image_region_ambiguous_member'
            entity=next(e for e in final.entities if e.entity_id.endswith(':'+str(member.person_key.serial)))
            track=next(t for t in family[1] if t['tracker_id']==10)
            assert entity.observed_at_us==track['observed_at_us']
            assert entity.stale_after_us==entity.observed_at_us+2_000_000
            assert (entity.position.x,entity.position.z)==(4.,2.)
            assert entity.position==next(s.position for s in entity.sources if s.accepted)
            old_deadline=entity.stale_after_us
            later=f.publish([packet(1,4,(9,)),packet(2,4,(9,10),exclude_old=True)])
            assert len(later.entities)==2
            assert entity.stale_after_us==old_deadline  # Prior committed original is immutable.
            assert all(e.stale_after_us==e.observed_at_us+2_000_000 for e in later.entities)
            groups=f.identity.spatial_memberships(at_interval=f.ingress._context)
            joined=[g for g in groups if len(g.members)==2]
            assert len(joined)==1 and {s.observation.member.raw_id for s in joined[0].members}=={9,10}
            final_wire=f.examples.actual_node_wire(f,f.wire,'local-occupancy-rematch-'+str(reverse))
            assert [len(s['people']) for s in final_wire['snapshots']]==[1,1,2,2]
            assert all(d['accepted'] for d in final_wire['diagnostics'])
        elif mode=='head_edge':
            original=next(o for k,o in f.ingress.projective_bridge.image_originals.items()
                if k.source_id==2 and k.raw_id==10)
            assert tuple(j[0] for j in original.semantic_joints)==(5,6,11,12)
            assert tuple(j[0] for j in original.head_joints)==(1,3)
            assert f.identity.ledger.image_region_error(original.stamp,f.ingress._context)=='image_region_ambiguous_member'
        assert f.identity.stats().pending_reservations==0
    finally:
        assert monitor.stop()['quiesced'];f.world.close()

@pytest.mark.parametrize('setting',['absent',False,True])
def test_r25_actual_startup_call_forwards_only_explicit_live_continuity(setting):
    import ast
    cfg,bindings,provider,root=_startup_fixture()
    cfg['xc3d_image_regions'].pop('contradictory_cue_quarantine',None)
    cfg['xc3d_image_regions'].pop('semantic_joint_max_mean_px',None)
    cfg['xc3d_image_regions'].pop('unresolved_birth_quarantine',None)
    cfg['xc3d_image_regions'].pop('live_track_continuity',None)
    if setting!='absent':cfg['xc3d_image_regions']['live_track_continuity']=setting
    assert image_regions_enabled(cfg)
    tree=ast.parse((root/'DS9/noesis/ds9_runtime_core.py').read_text())
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
        and n.value.func.attr=='for_estimated_image_regions')
    seen=[]
    def factory(*args,live_track_continuity=False,contradictory_cue_quarantine=False,semantic_joint_max_mean_px=None,unresolved_birth_quarantine=False):
        seen.append((args,live_track_continuity,contradictory_cue_quarantine));return 'single-actual-owner'
    env={'IdentityAssociationService':NS(for_estimated_image_regions=factory),
         'native_run_id':'run','quality_policy':object(),'quality_projective_policy':object(),
         'pipeline':NS(config=cfg)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),'actual-startup','exec'),env)
    assert seen==[(('run',env['quality_policy'],env['quality_projective_policy']),setting is True,False)]
    assert env['quality_identity']=='single-actual-owner'

@pytest.mark.parametrize('bad',[None,0,1,'true',[],{}])
def test_r25_live_continuity_config_requires_literal_boolean(bad):
    cfg,_,_,_=_startup_fixture();cfg['xc3d_image_regions']['live_track_continuity']=bad
    with pytest.raises(ValueError,match='task configuration'):image_regions_enabled(cfg)

@pytest.mark.parametrize('setting',['absent',False,True])
@pytest.mark.parametrize('live',[False,True])
def test_r32_conflict_config_actual_factory_requires_explicit_live_boolean(setting,live):
    import ast
    cfg,bindings,provider,root=_startup_fixture()
    value=cfg['xc3d_image_regions'];value.pop('contradictory_cue_quarantine',None)
    value.pop('semantic_joint_max_mean_px',None)
    value.pop('unresolved_birth_quarantine',None)
    value['live_track_continuity']=live
    if setting!='absent':value['contradictory_cue_quarantine']=setting
    if setting is True and not live:
        with pytest.raises(ValueError,match='task configuration'):image_regions_enabled(cfg)
        return
    assert image_regions_enabled(cfg)
    tree=ast.parse((root/'DS9/noesis/ds9_runtime_core.py').read_text())
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
        and n.value.func.attr=='for_estimated_image_regions')
    seen=[]
    def factory(*args,live_track_continuity=False,contradictory_cue_quarantine=False,semantic_joint_max_mean_px=None,unresolved_birth_quarantine=False):
        seen.append((live_track_continuity,contradictory_cue_quarantine));return 'actual-owner'
    env={'IdentityAssociationService':NS(for_estimated_image_regions=factory),
        'native_run_id':'run','quality_policy':object(),'quality_projective_policy':object(),
        'pipeline':NS(config=cfg)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),'actual-startup','exec'),env)
    assert seen==[(live,setting is True)]

@pytest.mark.parametrize('bad',[None,0,1,'true',[],{}])
def test_r32_conflict_config_rejects_nonbool_and_unknown_keys(bad):
    cfg,_,_,_=_startup_fixture();cfg['xc3d_image_regions']['contradictory_cue_quarantine']=bad
    with pytest.raises(ValueError,match='task configuration'):image_regions_enabled(cfg)
    cfg,_,_,_=_startup_fixture();cfg['xc3d_image_regions']['unexpected_policy']=False
    with pytest.raises(ValueError,match='task configuration'):image_regions_enabled(cfg)

@pytest.mark.parametrize('threshold',[True,False,0.,-1.,float('nan'),float('inf'),'33'])
def test_semantic_joint_config_threshold_validation(threshold):
    cfg,*_=_startup_fixture();cfg['xc3d_image_regions']['semantic_joint_max_mean_px']=threshold
    with pytest.raises(ValueError,match='task configuration'):image_regions_enabled(cfg)


def test_semantic_joint_config_factory_actual_keyword_none_or_fixed():
    import ast
    cfg,_,_,root=_startup_fixture()
    cfg['xc3d_image_regions']['unresolved_birth_quarantine']=False
    for threshold in (None,33.724887199080946):
        cfg['xc3d_image_regions']['live_track_continuity']=True
        cfg['xc3d_image_regions']['semantic_joint_max_mean_px']=threshold
        assert image_regions_enabled(cfg)
        tree=ast.parse((root/'DS9/noesis/ds9_runtime_core.py').read_text())
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
            and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='for_estimated_image_regions')
        seen=[]
        def factory(*args,**kwargs):seen.append(kwargs)
        env={'IdentityAssociationService':NS(for_estimated_image_regions=factory),'native_run_id':'run',
            'quality_policy':object(),'quality_projective_policy':object(),'pipeline':NS(config=cfg)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),'actual-startup','exec'),env)
        assert seen[0]['semantic_joint_max_mean_px'] is threshold


@pytest.mark.parametrize('bad',[None,0,1,'true',[],{}])
def test_unresolved_birth_config_requires_exact_boolean(bad):
    cfg,*_=_startup_fixture();cfg['xc3d_image_regions']['unresolved_birth_quarantine']=bad
    with pytest.raises(ValueError,match='task configuration'):image_regions_enabled(cfg)


@pytest.mark.parametrize('missing',['live_track_continuity','semantic_joint_max_mean_px'])
def test_unresolved_birth_config_requires_semantic_live_and_factory_forwards_raw(missing):
    import ast
    cfg,_,_,root=_startup_fixture()
    tree=ast.parse((root/'DS9/noesis/ds9_runtime_core.py').read_text())
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='for_estimated_image_regions')
    seen=[]
    env={'IdentityAssociationService':NS(for_estimated_image_regions=lambda *a,**k:seen.append(k)),
        'native_run_id':'run','quality_policy':object(),'quality_projective_policy':object(),'pipeline':NS(config=cfg)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),'actual-startup','exec'),env)
    assert seen[0]['unresolved_birth_quarantine'] is True
    cfg['xc3d_image_regions'][missing]=False if missing=='live_track_continuity' else None
    with pytest.raises(ValueError,match='task configuration'):image_regions_enabled(cfg)


def test_full_regions_small_pose_no_sampler_minimum_or_invented_joint():
    points=[(50.,30.,.9)]*17
    points[5],points[6]=(49.,30.,.9),(51.,30.,.9)
    points[11],points[12]=(48.5,42.,.9),(51.5,42.,.9)
    body=NS(keypoints=tuple(points))
    torso,shoulder=full_observed_image_regions(body)
    assert torso.capsule==(50.,30.,50.,42.,1.5,1.5)
    assert shoulder.capsule==(50.,30.,50.,30.,1.,1.)
    points[11]=(48.5,42.,.1)
    assert len(full_observed_image_regions(NS(keypoints=tuple(points))))==1
    points[5]=(49.,30.,.1)
    assert full_observed_image_regions(NS(keypoints=tuple(points)))==()

def test_task_artifact_frozen_and_startup_factory_names_actual_api():
    import yaml,hashlib,ast
    root=Path(__file__).resolve().parents[2]
    cfg=yaml.safe_load((root/'DS9/config/infer_xc3d_image_regions_task_r1.yaml').read_text())
    assert image_regions_enabled(cfg)
    artifact=(root/cfg['xc3d_image_regions']['prior_path']).read_bytes()
    assert hashlib.sha256(artifact).hexdigest()==cfg['xc3d_image_regions']['prior_sha256']
    payload=json.loads(artifact);assert payload['estimated_only'] is True
    assert {c['camera_id'] for c in payload['cameras']}=={'kitchen','family-room'}
    tree=ast.parse((root/'DS9/noesis/ds9_runtime_core.py').read_text())
    assert any(isinstance(n,ast.Attribute) and n.attr=='for_estimated_image_regions' for n in ast.walk(tree))

def _startup_fixture(fault=None):
    import yaml
    from noesis_core.contracts.scene_prior import scene_revision_sha256
    from reid.mv3dt_person_association import SourceBinding
    root=Path(__file__).resolve().parents[2]
    cfg=yaml.safe_load((root/'DS9/config/infer_xc3d_image_regions_task_r1.yaml').read_text())
    artifact=json.loads((root/cfg['xc3d_image_regions']['prior_path']).read_text())
    snapshots={};presentations={};bindings=[]
    for row in artifact['cameras']:
        world=NS(frame_id='backend_world_m',revision=row['world_revision'])
        contract=NS(source_frame=NS(revision=row['source_calibration_revision']),target_frame=world,transform_sha256=row['transform_id'])
        snapshot=NS(image_size=tuple(row['image_size']),intrinsics=[list(a) for a in row['intrinsics']],
            world_extrinsics_col_major=list(row['world_extrinsics_col_major']),frame_contract=contract,
            camera_calibration_sha256=row['measurement_calibration_sha256'])
        snapshots[row['source_id']]=snapshot
        from noesis_core.contracts.scene_prior import ScenePriorFrameRef
        frame=ScenePriorFrameRef(frame_id='backend_world_m',revision=row['world_revision'])
        presentations[row['world_revision']]={'contract':'noesis.scene_prior.world_to_scene_presentation','contract_version':1,
            'world_frame':frame.model_dump(),'render_frame':'menon_scene','world_to_scene_col_major':row['presentation_col_major'],
            'world_to_scene_sha256':row['presentation_sha256'],'scene_revision_id':scene_revision_sha256(frame,'menon_scene',tuple(row['presentation_col_major'])),
            'source_transform_sha256s':[row['transform_id']],'accepted':True,'provenance':{'role':'fixture','sha256':'a'*64,'version':'synthetic'},
            'authored_scene_sha256':row['authored_scene_sha256'],'registration_sha256':'c'*64}
        bindings.append(SourceBinding(row['camera_id'],row['source_id'],0,row['source_calibration_revision'],row['transform_id'],'synthetic-clock','source-media'))
    if fault=='raster':snapshots[1].image_size=(1280,720)
    if fault=='K':snapshots[1].intrinsics[0][0]+=1
    if fault=='E':snapshots[1].world_extrinsics_col_major[12]+=.01
    if fault=='hash':cfg['xc3d_image_regions']['prior_sha256']='0'*64
    provider=NS(snapshot=lambda s,c:snapshots[s],world_frame_presentations=lambda:presentations)
    return cfg,tuple(bindings),provider,root

def _diagnostic_bridge():
    cfg,bindings,provider,root=_startup_fixture()
    return ImageRegionQualityBridge(runtime_image_region_policy(cfg,bindings,provider,root),None)

@pytest.mark.parametrize('invalid',[None,[],(object(),)])
def test_r29_owner_participant_cohort_corruption_is_not_optional_diagnostic(invalid):
    bridge=_diagnostic_bridge()
    owner=NS(image_region_active_cohort=lambda:invalid,
        join_estimated_image_regions=lambda *a,**kw:pytest.fail('invalid authority must not join'))
    with pytest.raises(ValueError,match='owner_cohort_binding'):
        bridge.step([],{},NS(intervals=()),owner)

def _real_diagnostic_identity(bridge):
    from noesis.identity_association_service import IdentityAssociationService
    from reid.mv3dt_person_association import SpatialAssociationPolicy,AppearanceProfile
    return IdentityAssociationService.for_estimated_image_regions('snapshot-test',SpatialAssociationPolicy(
        tuple(c.source for c in bridge.policy.cameras),AppearanceProfile('swin','layer',256,'profile'),
        (('kitchen','family-room'),)),bridge.policy)


def test_encoded_owner_snapshot_canonical_step_has_zero_decode_copy_and_no_late_refetch(monkeypatch):
    import noesis.pipelines.projective_continuity as module
    import reid.mv3dt_person_association as kernel
    from reid.mv3dt_person_association import ImageRegionDecisionSnapshot
    bridge=_diagnostic_bridge();identity=_real_diagnostic_identity(bridge);calls=[]
    def join(*a,**k):
        calls.append('join')
        identity.ledger._image_decision_last_completed={'sequence':1,'members':[],'pairs':[],
            'pre_components':[],'post_components':[],'outcome':{'reason':'exact_completed_join'}}
        return NS(reason='unchanged_join')
    monkeypatch.setattr(identity,'join_estimated_image_regions',join)
    with monkeypatch.context() as m:
        m.setattr(kernel.json,'loads',lambda *a,**k:pytest.fail('optional decode in step'))
        m.setattr(module,'_copy_image_region_decisions',lambda *a:pytest.fail('recursive optional copy in step'))
        m.setattr(module,'deepcopy',lambda *a:pytest.fail('deepcopy in step'))
        assert bridge.step([],{},NS(intervals=()),identity).reason=='unchanged_join'
    assert calls==['join'] and type(bridge._image_region_decisions) is ImageRegionDecisionSnapshot
    expected=identity.image_region_decision_diagnostics()
    identity.ledger.reset_run('later-run')
    monkeypatch.setattr(identity,'image_region_decision_snapshot',lambda:pytest.fail('late re-fetch'))
    got=bridge.diagnostics()['image_region_decisions'];assert got==expected
    got['transitions'].clear()
    assert bridge.diagnostics()['image_region_decisions']==expected


@pytest.mark.parametrize('fault',['missing','raises','non_json','records','nonfinite'])
def test_encoded_real_service_preserves_injected_legacy_accessor_controls(monkeypatch,fault):
    bridge=_diagnostic_bridge();identity=_real_diagnostic_identity(bridge)
    result=NS(reason='unchanged_join')
    monkeypatch.setattr(identity,'join_estimated_image_regions',lambda *a,**k:result)
    value={'contract':'noesis.identity.image_region_decisions.v1','limit':8,'evicted':0,'transitions':[]}
    if fault=='non_json':value['bad']=object()
    if fault=='records':value['transitions']=[{}]*9
    if fault=='nonfinite':value['bad']=float('nan')
    def accessor():
        if fault=='raises':raise ValueError('optional failure')
        return value
    monkeypatch.setattr(identity,'image_region_decision_diagnostics',None if fault=='missing' else accessor)
    assert bridge.step([],{},NS(intervals=()),identity) is result
    assert bridge.diagnostics()['image_region_decisions']['available'] is False


def test_encoded_snapshot_malformed_private_override_is_optional(monkeypatch):
    from reid.mv3dt_person_association import ImageRegionDecisionSnapshot
    bridge=_diagnostic_bridge();identity=_real_diagnostic_identity(bridge)
    result=NS(reason='unchanged_join')
    monkeypatch.setattr(identity,'join_estimated_image_regions',lambda *a,**k:result)
    monkeypatch.setattr(identity,'image_region_decision_snapshot',lambda:object.__new__(ImageRegionDecisionSnapshot))
    assert bridge.step([],{},NS(intervals=()),identity) is result
    assert bridge.diagnostics()['image_region_decisions']['available'] is False


def test_owner_decision_snapshot_after_same_join_is_detached_and_optional():
    bridge=_diagnostic_bridge();calls=[]
    payload={'contract':'noesis.identity.image_region_decisions.v1','limit':8,'evicted':0,
             'transitions':[{'members':[],'pairs':[],'pre_components':[],'post_components':[],
                             'outcome':{'reason':'actual_owner_reason'}}]}
    result=NS(reason='unchanged_join')
    def join(*args,**kwargs):calls.append('join');return result
    def diagnostics():calls.append('diagnostic');return payload
    identity=NS(join_estimated_image_regions=join,image_region_decision_diagnostics=diagnostics,
                image_region_active_cohort=lambda:())
    assert bridge.diagnostics()['image_region_decisions']=={}
    assert bridge.step([],{},NS(intervals=()),identity) is result
    assert calls==['join','diagnostic']
    frozen=bridge.diagnostics()['image_region_decisions']
    assert frozen==payload
    payload['transitions'].clear();frozen['transitions'][0]['members'].append('caller mutation')
    assert bridge.diagnostics()['image_region_decisions']['transitions'][0]['members']==[]
    assert calls==['join','diagnostic']

@pytest.mark.parametrize('fault',['missing','raises','non_json','records','members','pairs','nonfinite'])
def test_optional_owner_diagnostic_failure_does_not_change_join(fault):
    bridge=_diagnostic_bridge();result=NS(reason='unchanged_join')
    payload={'contract':'noesis.identity.image_region_decisions.v1','limit':8,'evicted':0,'transitions':[]}
    record={'members':[],'pairs':[],'pre_components':[],'post_components':[]}
    if fault=='non_json':payload['unsupported']=object()
    if fault=='records':payload['transitions']=[record]*9
    if fault in ('members','pairs'):
        record[fault]=[{}]*(65 if fault=='members' else 129);payload['transitions']=[record]
    if fault=='nonfinite':payload['bad']=float('nan')
    def accessor():
        if fault=='raises':raise RuntimeError('optional diagnostic')
        return payload
    identity=NS(join_estimated_image_regions=lambda *a,**kw:result,image_region_active_cohort=lambda:())
    if fault!='missing':identity.image_region_decision_diagnostics=accessor
    assert bridge.step([],{},NS(intervals=()),identity) is result
    assert bridge.diagnostics()['image_region_decisions']['available'] is False
    assert bridge.counts['result:unchanged_join']==1

def test_actual_join_failure_is_not_hidden_by_optional_diagnostics():
    bridge=_diagnostic_bridge();calls=[]
    def join(*args,**kwargs):raise ValueError('actual authority refusal')
    identity=NS(join_estimated_image_regions=join,
                image_region_decision_diagnostics=lambda:calls.append('unexpected'),image_region_active_cohort=lambda:())
    with pytest.raises(ValueError,match='actual authority refusal'):
        bridge.step([],{},NS(intervals=()),identity)
    assert calls==[] and bridge.diagnostics()['image_region_decisions']=={}

@pytest.mark.parametrize('fault',[None,'raster','K','E','hash'])
def test_actual_startup_loader_exact_saved_geometry_and_mismatch_refusal(fault):
    cfg,bindings,provider,root=_startup_fixture(fault)
    if fault:
        with pytest.raises(ValueError):runtime_image_region_policy(cfg,tuple(bindings),provider,root)
    else:
        policy=runtime_image_region_policy(cfg,tuple(bindings),provider,root)
        artifact=json.loads((root/'DS9/config/xc3d_image_regions_prior_r1.json').read_text())
        assert all(c.prior_id.startswith('estimated-task:') for c in policy.cameras)
        assert [list(map(list,c.peer_projection)) for c in policy.cameras]==[r['candidate_peer_projection'] for r in artifact['cameras']]

@pytest.mark.parametrize('absolute',[False,True])
def test_render_two_roots_keeps_only_task_prior_in_code_root(tmp_path,absolute):
    import copy,yaml
    cfg,bindings,provider,root=_startup_fixture()
    spec=importlib.util.spec_from_file_location('image_replay_renderer',root/'DS9/scripts/render_recorded_replay_config.py')
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    code=tmp_path/'code';original=tmp_path/'original'
    code.mkdir();original.mkdir()
    baseline=code/'DS9/config/infer_xc3d_image_regions_task_r1.yaml'
    baseline.parent.mkdir(parents=True)
    prior_relative=Path('DS9/config/xc3d_image_regions_prior_r1.json')
    prior=code/prior_relative
    prior.write_bytes((root/prior_relative).read_bytes())
    baseline.write_text(yaml.safe_dump(cfg))
    if absolute:cfg['xc3d_image_regions']['prior_path']=str(prior)
    cfg['calibration_path']='config/camera_calibration.json'
    cfg['other_prior']='DS9/config/xc3d_image_regions_prior_r1.json'
    cfg['absolute_asset']=str(code/'DS9/config/existing.yaml')
    cfg['model']='DS9/models/existing.engine'
    before=copy.deepcopy(cfg)
    rendered=renderer.render_replay_config(cfg,{c:tmp_path/(c+'.mp4') for c in renderer.CANONICAL_CAMERA_ORDER},baseline_path=baseline,original_root=original)
    assert cfg==before
    assert not (original/prior_relative).exists()
    assert rendered['xc3d_image_regions']['prior_path']==str(prior)
    assert rendered['other_prior']==str(original/prior_relative)
    assert rendered['calibration_path']==str(original/'config/camera_calibration.json')
    assert rendered['absolute_asset']==str(original/'DS9/config/existing.yaml')
    assert rendered['model']=='DS9/models/existing.engine'
    policy=runtime_image_region_policy(rendered,bindings,provider,code)
    assert len(policy.cameras)==2
    rendered['xc3d_image_regions']['prior_sha256']='0'*64
    with pytest.raises(ValueError):runtime_image_region_policy(rendered,bindings,provider,code)

@pytest.mark.parametrize('bad',['DS9/config/other.json','../original/DS9/config/xc3d_image_regions_prior_r1.json'])
def test_render_task_prior_rejects_non_task_code_scope(tmp_path,bad):
    cfg,_,_,root=_startup_fixture()
    spec=importlib.util.spec_from_file_location('image_replay_renderer_scope',root/'DS9/scripts/render_recorded_replay_config.py')
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    cfg['xc3d_image_regions']['prior_path']=bad
    original=tmp_path/'original';original.mkdir()
    with pytest.raises(renderer.ReplayConfigError,match='task prior'):
        renderer.render_replay_config(cfg,{c:tmp_path/(c+'.mp4') for c in renderer.CANONICAL_CAMERA_ORDER},baseline_path=root/'DS9/config/infer_xc3d_image_regions_task_r1.yaml',original_root=original)

@pytest.mark.parametrize('birth_policy',[False,True])
def test_actual_image_bridge_service_world_publisher_and_node(tmp_path,monkeypatch,birth_policy):
    for name in ('pyservicemaker','pyds','noesis_pose_meta_ext','noesis_analytics_meta_ext',
        'noesis_reid_meta_ext','noesis_v3dt_meta_ext','noesis_depth_meta_ext','noesis_depth_tracking_tensor_ext'):
        monkeypatch.setitem(sys.modules,name,None)
    m=helpers();frame=[1];ns=time.time_ns()
    monkeypatch.setattr(time,'time_ns',lambda:ns+frame[0]*33_333_333)
    monkeypatch.setattr(time,'monotonic',lambda:100.+frame[0]/30)
    selected={}
    def factory(run,spatial):
        examples=m.world_examples();P=((100.,0.,0.,0.),(0.,100.,0.,0.),(0.,0.,1.,0.))
        def shifted(x):return ((100.,0.,0.,x),(0.,100.,0.,0.),(0.,0.,1.,0.))
        policy=ProjectiveContinuityPolicy(tuple(ProjectiveCameraBinding(b,'synthetic-image-prior','raster:'+b.camera_id,
            (1920,1080),'backend_world_m',examples.frame_binding(b.source_id).target_frame.revision,
            examples.frame_binding(b.source_id).source_camera_calibration_sha256,P,
            shifted(-10. if b.source_id==1 else 10.)) for b in spatial.sources))
        selected['policy']=policy
        return IdentityAssociationService.for_estimated_image_regions(run,spatial,policy,
            live_track_continuity=birth_policy,semantic_joint_max_mean_px=5. if birth_policy else None,
            unresolved_birth_quarantine=birth_policy)
    f=m.NativeQualityFixture(tmp_path,identity_factory=factory)
    from collections import deque
    from noesis.pipelines.source_acquisition import SelectedSwinIngress,SourceAcquisitionState
    from noesis_core.runtime_publication import RuntimePublicationGate
    pipeline=NS(xc3d_world_quality_ingress=f.ingress,xc3d_private_owner_state=SourceAcquisitionState(f.identity.run_id,'synthetic-quality-graph'),xc3d_private_owner_diagnostics=deque(maxlen=256))
    pipeline.xc3d_swin_ingress=SelectedSwinIngress(f.identity.run_id,'synthetic-quality-graph',f.profile,None,None,timestamp_origin='source_pts')
    f.ingress.origin_bindings=m.actual_lineage_callback(pipeline)
    bridge=ImageRegionQualityBridge(selected['policy'],lambda:pipeline.xc3d_swin_ingress.cache)
    f.ingress.projective_bridge=bridge
    processor=NS(pipeline=pipeline,tracking_pub=f.publisher,publication_gate=RuntimePublicationGate(),_stamp_footpoint_lifecycles=lambda *_:None,
        _commit_enqueued_world_output_watermarks=lambda *_a,**_k:None,_last_tracking_publish_ts_by_sensor={},_last_tracking_count_by_sensor={},bev_renderer=None)
    n=NS(f=f,pipeline=pipeline,processor=processor)
    try:
        for i in (1,2,3):
            frame[0]=i
            for source in ((2,1) if birth_policy else (1,2)):
                packet,tracks=m.lineage_packet(n,source,i,x=1.+source*.001)
                inputs=[]
                for row,track in zip(packet.rows,tracks):
                    u=50. if source==1 else 40.
                    track['bbox']=[u-10,20.,20.,40.]
                    points=[(u,30.,.9)]*17
                    points[5],points[6]=(u-2,30.,.9),(u+2,30.,.9)
                    points[11],points[12]=(u-2,42.,.9),(u+2,42.,.9)
                    body=None if birth_policy and i==1 and source==1 else capture_body_support(track,row.evidence.lifecycle,points,pose_is_current=True)
                    inputs.append(replace(row.evidence,body=body))
                packet=replace(freeze_quality_work(packet.private,packet.continuity,tuple(inputs)),ingress_monotonic_s=time.monotonic())
                snapshot=m.publish_lineage(n,packet,tracks)
                if birth_policy:
                    assert len(snapshot.entities)==1  # unresolved new local position is NOT a second head
                    if i==1 and source==1:
                        pending=next(x for x in f.identity.ledger._members.values() if x.observation.stamp.member.source_id==1)
                        assert pending.image_region_birth_quarantined
                        assert f.identity.ledger.stats().active_persons==2  # no fabricated merge
        assert f.identity.ledger.stats().active_persons==1,dict(bridge.counts)
        assert all(tuple(j[0] for j in m.image_region_original.semantic_joints)==(5,6,11,12)
            and all(j[3]==.9 for j in m.image_region_original.semantic_joints)
            for m in f.identity.ledger._members.values())
        assert len(snapshot.entities)==1 and len(snapshot.entities[0].sources)==2
        assert bridge.counts['current_region_originals']==(5 if birth_policy else 6)
        decisions=bridge.diagnostics()['image_region_decisions']
        assert decisions==f.identity.image_region_decision_diagnostics()
        assert decisions['transitions']  # actual owner captured the physical merge
        decisions['transitions'].clear()
        assert bridge.diagnostics()['image_region_decisions']['transitions']
        wire=f.examples.actual_node_wire(f,f.wire,'new-image-bridge')
        assert wire
    finally:f.world.close()


def test_r27_real_bridge_keeps_unknown_active_until_explicit_owner_classification(monkeypatch):
    """Owner cohort is complete; no native state is fabricated for retained F9."""
    path=Path(__file__).resolve().parents[2]/'tests/test_mv3dt_person_association.py'
    spec=importlib.util.spec_from_file_location('r27_public_image_examples',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    from reid.mv3dt_person_association import SpatialLifecycleReceipt
    selected,_,rows,context,delivery=m.image_region_fixture(competitor='missing')
    owner=IdentityAssociationService.for_estimated_image_regions('run-a',m.spatial_policy(),selected)
    bridge=ImageRegionQualityBridge(selected,None)
    now=[delivery.evaluation_monotonic_s]
    monkeypatch.setattr(time,'monotonic',lambda:now[0])
    def plans(originals):
        result=[]
        for original in originals:
            stamp=original.stamp;member=stamp.member
            camera=next(c for c in selected.cameras if c.source.source_id==member.source_id)
            life=NS(tracker_id=member.raw_id,generation=member.lifecycle_generation)
            points=[(0.,0.,0.)]*17
            points[5],points[6]=(60.,100.,.9),(64.,100.,.9)
            points[11],points[12]=(60.,110.,.9),(64.,110.,.9)
            body=None if member.raw_id==9 else NS(lifecycle=life,image_size=camera.image_size,
                bbox_xywh=(40.,50.,60.,140.),keypoints=tuple(points))
            row=NS(denial=None,evidence=NS(lifecycle=life,body=body),
                original=NS(original_frame=stamp.frame_id,pts_ns=round(stamp.media_time_s*1e9)))
            work=NS(continuity=NS(source_id=member.source_id),rows=(row,),ingress_monotonic_s=now[0],
                swin_extractions=(),private=NS(original=None))
            result.append((work,None,None,None,None,(original,)))
        return result
    current={r.stamp.member:r for r in rows}
    for row in rows:
        assert owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE')).disposition=='observed'
    initial=bridge.step(plans(rows),current,context,owner)
    assert initial.reason!='current_visual_complete_cohort_required'
    _,_,fresh,fresh_context,fresh_delivery=m.image_region_fixture(frame=11)
    now[0]=fresh_delivery.evaluation_monotonic_s
    for row in fresh:
        assert owner.observe_spatial(row,SpatialLifecycleReceipt('native:'+row.stamp.observation_id,row.stamp,'ACTIVE')).disposition=='observed'
        current[row.stamp.member]=row
    retained=rows[-1]
    result=bridge.step(plans(fresh),current,fresh_context,owner)
    assert result.reason!='current_visual_complete_cohort_required'
    assert result.disposition=='observed' and owner.ledger.stats().active_persons==3
    assert len(current)==3 and retained.stamp.member in current
    assert retained.stamp not in tuple(o.stamp for o in bridge.image_originals.values())
    assert len(owner.ledger.household_claim_owners())==3
    # Complete cohort keeps UNKNOWN F9 until the separate actual owner event.
    from reid.mv3dt_person_association import SpatialLifecycleUpdate
    update=SpatialLifecycleUpdate('actual-retained-owner-event',retained.stamp,11,
        fresh[-1].stamp.acquired_at_s,retained.stamp.clock_domain,fresh[-1].stamp.media_time_s)
    updated=owner.update_spatial_lifecycle(update)
    assert updated.disposition=='observed',updated
    assert retained.stamp not in owner.image_region_active_cohort()
    joined=bridge.step([],current,fresh_context,owner)
    assert joined.disposition=='accepted',joined
    assert owner.ledger.stats().active_persons==2  # joined K7/F8; F9 custody preserved
    assert len(owner.ledger.household_claim_owners())==3
