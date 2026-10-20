"""Actual producer/transport/gate/World code, with SYNTHETIC depth/video scalars."""
from dataclasses import replace
from types import SimpleNamespace
import json
import time
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from noesis.pipelines import hooks
from noesis.metadata.object_depth import ObjectDepthResult
from noesis.pipelines.world_quality_adapter import capture_body_support, freeze_quality_work
from noesis.pipelines.projective_continuity import sample_from_quality
from reid.mv3dt_person_association import PersonAssociationLedger
def _helpers(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


_projective=_helpers(Path(__file__).with_name('test_xc3d_projective_continuity.py'),'body_depth_projective_helpers')
projective_native=_projective.projective_native
physical_packet=_projective.physical_packet
publish_lineage=_projective.publish_lineage


class SyntheticDepthDevice:
    """No SDK/GPU claim; supplies bounded stats to the real native-sampler adapter."""
    def __init__(self):
        self.calls = []

    def sample_pose_capsule_stats(self, capsules, limit):
        self.calls.append((capsules, limit))
        depth = 3. if len(self.calls) == 1 and len(capsules) == 2 else 2.
        return dict(sample_count=80, valid_fraction=.9, depth_median=depth,
                    depth_p10=depth-.02, depth_p90=depth+.02, sampled_capsule_area_px=100)


def body_packet(n, source, frame, monkeypatch, *, quality=1., alternate=False, body_offset_px=0., separated_ankles=False):
    packet, tracks = physical_packet(n, source, frame, quality=quality)
    row, = packet.rows
    track, = tracks
    u = row.evidence.body.keypoints[5][0] + body_offset_px
    track['bbox'] = [u-20., 10., 40., 60.]
    joints = np.zeros((17, 3))
    for index, x, y in ((5,u-8.,20.),(6,u+8.,20.),(11,u-8.,40.),(12,u+8.,40.)):
        joints[index] = (x,y,.9)
    separation, ankle_radius = (10.,1.) if separated_ankles else (1.,2.)
    if alternate:
        joints[15],joints[16]=(u-separation,60.,.9),(u+separation,60.,.9)
    processor = object.__new__(hooks._ObjectDepthFusionProcessor)
    processor.depth_unit, processor.depth_is_metric, processor.depth_model_name = 'm', True, 'synthetic'
    processor._result_cache = {}
    capsule, uv = processor._pose_torso_capsule_geometry(joints, bbox=track['bbox'])
    device = SyntheticDepthDevice()
    obj = SimpleNamespace(object_id=track['tracker_id'],class_id=0,confidence=.9)
    depth = processor._sample_pose_capsule_result_native(
        SimpleNamespace(source_id=source,frame_number=frame,buf_pts=track['media_pts_ns']),obj,
        bbox=track['bbox'],depth_frame=SimpleNamespace(depth_device_frame=device,frame_w=1920,frame_h=1080),
        crop_rect=(0,0,1920,1080),body_mask=np.ones((1,1),bool),contact_mask=np.ones((1,1),bool),
        contact_uv=(u,60.) if alternate else None,visible_ankles=2 if alternate else 0,
        contact_capsules=((u-separation,55.,u-separation,60.,ankle_radius,ankle_radius),
                          (u+separation,55.,u+separation,60.,ankle_radius,ankle_radius)) if alternate else (),
        torso_capsule=capsule,torso_uv=uv)
    depth = replace(depth,depth_tensor_frame_id=frame,depth_tensor_ts_us=track['media_pts_ns']//1000,
                    depth_tensor_age_frames=0,depth_tensor_age_us=0)
    # Exercise actual attach/parse entrypoints; only the opaque native string
    # storage is a fake. Native C++ roundtrip is separately exercised by replay.
    storage = {}
    extension = SimpleNamespace(attach_object_depth=lambda _b,_o,p,_r: storage.update(payload=p) or True,
                              extract_object_depth=lambda _o:storage['payload'])
    monkeypatch.setattr(hooks,'noesis_depth_meta_ext',extension)
    assert processor._attach_object_depth_payload(None,obj,depth.to_native_dict())
    decoded = hooks._extract_object_depth_result_from_meta(obj)
    assert decoded == depth
    assert 'native_pose_support' not in decoded.to_dict()
    assert 'native_pose_support' not in decoded.to_json()
    processor._cache_result(source_id=source,pts_us=track['media_pts_ns']//1000,obj_meta=obj,result=decoded)
    cache = processor._cached_payload(source_id=source,frame_id=frame+1,
        pts_us=track['media_pts_ns']//1000+33333,obj_meta=obj,bbox=track['bbox'])
    assert ObjectDepthResult.from_dict(cache).native_pose_support == depth.native_pose_support
    assert cache['measurement_cached'] is True
    telemetry = object.__new__(hooks._AnalyticsTelemetryProcessor)
    telemetry.depth_registration = None
    calib = SimpleNamespace(camera_id=row.evidence.measurements.cohort.camera_id,image_size=(1920,1080),
        intrinsics=np.diag([1000.,1000.,1.]),extrinsics_col_major=np.eye(4).reshape(-1,order='F').tolist(),
        floor_y=0.,unit_scale=1.)
    selected_anchor = telemetry._resolve_person_depth_anchor(decoded)
    selected = telemetry._depth_observation_from_anchor(calib=calib,anchor=selected_anchor,
        depth_result=decoded,flip_u=False,flip_v=False,track=track)
    receipt = telemetry._original_body_depth_receipt(cohort=row.evidence.measurements.cohort,
        depth_result=decoded,selected_observation=selected,selected_anchor=selected_anchor,calib=calib,
        track=track,pose_is_current=True,flip_u=False,flip_v=False)
    assert receipt is not None and receipt.original_point[1] != 0.
    assert receipt.patch.capsules == (tuple(capsule),)
    assert len(device.calls) == (2 if alternate else 1)
    if alternate:
        assert decoded.anchor_source == 'pose_ankle_support' and selected.raw_depth_m == 3.
        assert receipt.raw_depth_m == 2. and receipt.patch.anchor_uv != decoded.anchor_uv
    body = capture_body_support(track,row.evidence.lifecycle,joints,pose_is_current=True,
        image_patches=hooks._current_body_image_patches(joints,track['bbox'],pose_is_current=True))
    anatomies = telemetry._original_anatomy_depth_receipts(body_receipt=receipt,
        cohort=row.evidence.measurements.cohort, depth_result=decoded, selected_observation=selected,
        selected_anchor=selected_anchor, calib=calib, track=track, pose_is_current=True,
        flip_u=False, flip_v=False)
    inputs = hooks._finalize_world_quality_inputs([(track,row.evidence.lifecycle,row.evidence.measurements,
        row.evidence.resolution,decoded,body,receipt,anatomies)])
    result = replace(freeze_quality_work(packet.private,packet.continuity,inputs),
                     ingress_monotonic_s=packet.ingress_monotonic_s)
    return result,tracks,telemetry,calib,selected,selected_anchor,processor,device


def body_projection(n):
    # SYNTHETIC K/E; unlike the old floor fixture this preserves Y explicitly.
    projection=((1000.,0.,0.,0.),(0.,1000.,0.,0.),(0.,0.,1.,0.))
    policy=replace(n.bridge.policy,cameras=tuple(replace(c,own_projection=projection,
        peer_projection=projection) for c in n.bridge.policy.cameras))
    n.bridge.policy=policy
    n.bridge.cameras={c.source.source_id:c for c in policy.cameras}
    n.f.identity._projective_policy=policy


def test_actual_body_sampler_native_transport_original_point_gate_world_bev_best_view(projective_native,monkeypatch):
    n=projective_native
    body_projection(n)
    for frame in (1,2,3,4,5):
        for source in (1,2):
            packet,tracks,*_=body_packet(n,source,frame,monkeypatch,
                quality=(4. if source==1 else 1.) if frame<4 else (1. if source==1 else 4.),
                alternate=source==1)
            snapshot=publish_lineage(n,packet,tracks)
    entity,=snapshot.entities
    assert len(entity.sources)==2
    assert next(s.camera_id for s in entity.sources if s.accepted)=='kitchen'
    assert n.bridge.counts['body_depth:valid:pose_torso_support'] == 10
    assert n.bridge.counts['accepted']>0
    # Real public tracking/World/BEV cohorts use the ground filter, not the
    # private body height. Actual Node22 reducer consumes every wire cohort.
    assert entity.position.y == 0.
    assert 'native_pose_support' not in json.dumps(n.f.wire)
    assert '_xc3d_body_depth' not in json.dumps(n.f.wire)
    with monkeypatch.context() as scoped:
        scoped.setattr(time,'monotonic',n.real_monotonic)
        result=n.f.examples.actual_node_wire(n.f,n.f.wire,'body-depth-original-support')
    assert all(d['accepted'] for d in result['diagnostics'])
    assert result['snapshots'][-1]['people']


def test_two_distinct_body_trajectories_with_identical_appearance_do_not_join(projective_native,monkeypatch):
    n=projective_native
    body_projection(n)
    for frame in (1,2,3,4):
        for source in (1,2):
            packet,tracks,*_=body_packet(n,source,frame,monkeypatch,body_offset_px=120. if source==2 else 0.)
            snapshot=publish_lineage(n,packet,tracks)
    assert len(snapshot.entities)==2
    assert n.bridge.counts['body_depth:valid:pose_torso_support']==8
    assert n.bridge.counts['accepted']==0
    assert n.bridge.numeric_counts['peer_support_misses']>0


def test_genuine_body_point_independent_of_held_world_without_refreshing_it(projective_native,monkeypatch):
    n=projective_native
    body_projection(n)
    packet,tracks,*_=body_packet(n,1,1,monkeypatch)
    row,=packet.rows
    fields=dict(row.evidence.world)
    fields.update(world_source='anchor_hold',world_measurement_accepted=False)
    held=replace(row,current_measured=False,evidence=replace(row.evidence,world=tuple(fields.items())))
    before=held.evidence.world
    # Use the actual lifecycle observation which the owner constructs on ingest.
    publish_lineage(n,packet,tracks)
    stamp=next(o.stamp for o in n.f.ingress.originals.values())
    sample=sample_from_quality(held,stamp,n.bridge.cameras[1],packet.ingress_monotonic_s,joint_threshold=.35)
    assert sample is not None and sample.pixel_uncertainty_px is None
    assert sample.observed_support_radius_px == row.evidence.body_depth.patch.enclosing_radius_px()
    assert held.evidence.world == before and held.current_measured is False
    assert sample.original_point[1] != 0.


@pytest.mark.parametrize('fault',['missing','ambiguous','stale_patch','cached','tensor','raster','calibration',
                                    'registration','pose','wrong_cohort','flipped'])
def test_body_handoff_refuses_bad_original_support(projective_native,monkeypatch,fault):
    n=projective_native
    body_projection(n)
    packet,tracks,*_=body_packet(n,1,1,monkeypatch)
    publish_lineage(n,packet,tracks)
    row,=packet.rows
    ev=row.evidence
    receipt=ev.body_depth
    patch=receipt.patch
    if fault=='missing':ev=replace(ev,body_depth=None)
    elif fault=='ambiguous':ev=replace(ev,depth=replace(ev.depth,pose_depth_samples=ev.depth.pose_depth_samples*2))
    elif fault=='stale_patch':ev=replace(ev,body_depth=replace(receipt,patch=replace(patch,measurement_frame_id=0)))
    elif fault=='cached':ev=replace(ev,depth=replace(ev.depth,measurement_cached=True))
    elif fault=='tensor':ev=replace(ev,depth=replace(ev.depth,depth_tensor_frame_id=0))
    elif fault=='raster':ev=replace(ev,body_depth=replace(receipt,patch=replace(patch,image_size=(1280,720))))
    elif fault=='calibration':ev=replace(ev,body_depth=replace(receipt,
        cohort=receipt.cohort.model_copy(update={'calibration_revision':'wrong'})))
    elif fault=='registration':ev=replace(ev,body_depth=replace(receipt,registration_status='unavailable'))
    elif fault=='pose':ev=replace(ev,body=replace(ev.body,keypoints=tuple((u,v,0.) for u,v,c in ev.body.keypoints)))
    elif fault=='wrong_cohort':ev=replace(ev,body_depth=replace(receipt,
        cohort=receipt.cohort.model_copy(update={'frame_id':0})))
    elif fault=='flipped':ev=replace(ev,body_depth=replace(receipt,image_flips=(True,False)))
    frozen=freeze_quality_work(packet.private,packet.continuity,(ev,))
    checked,=frozen.rows
    stamp=next(o.stamp for o in n.f.ingress.originals.values())
    # No floor contact exists here, so unavailable body may not silently use
    # inferred feet, public held output or the detector rectangle as a cue.
    assert sample_from_quality(checked,stamp,n.bridge.cameras[1],packet.ingress_monotonic_s,joint_threshold=.35) is None


def test_body_projection_preserves_y_applies_B_once_and_reuses_selected_computation(projective_native,monkeypatch):
    packet,tracks,telemetry,calib,selected,anchor,*_=body_packet(projective_native,1,1,monkeypatch)
    row,=packet.rows
    original=row.evidence.body_depth
    # SAME selected computation must not be reprojected/re-registered.
    with monkeypatch.context() as scoped:
        scoped.setattr(telemetry,'_depth_observation_from_anchor',lambda **_:pytest.fail('duplicate projection'))
        repeated=telemetry._original_body_depth_receipt(cohort=original.cohort,depth_result=row.evidence.depth,
            selected_observation=selected,selected_anchor=anchor,calib=calib,track=tracks[0],
            pose_is_current=True,flip_u=False,flip_v=False)
    assert repeated==original
    B=np.eye(4)
    B[:3,3]=(1.,2.,3.)
    calib.target_from_calibration_col_major=B.reshape(-1,order='F').tolist()
    point=telemetry._project_pixel_to_world_observation(calib,*original.patch.anchor_uv,depth_m=2.,flip_u=False,flip_v=False)
    assert point==pytest.approx(np.array(original.original_point)+(1.,2.,3.))
    assert point[1] != 0. and point[1] != original.original_point[1]+4.
    radius=original.patch.enclosing_radius_px()
    assert radius > max(original.patch.capsules[0][4:])


def test_numeric_gate_refuses_mixed_anatomy_and_does_not_reinterpret_support_as_sigma():
    kernel=_helpers(Path(__file__).resolve().parents[2]/'tests/test_mv3dt_person_association.py','body_depth_kernel_helpers')
    owner,policy,proof,rows,context,delivery=kernel.projective_kernel_fixture()
    pair=proof.pairs[0]
    def body(side,anatomy):
        return tuple(replace(s,pixel_uncertainty_px=None,support_basis=anatomy,observed_support_radius_px=1.) for s in side)
    mixed=replace(proof,pairs=(replace(pair,left=body(pair.left,'pose_torso_support'),
                                       right=body(pair.right,'pose_shoulder_support')),))
    result=owner.inspect_projective_continuity(mixed,projective_policy=policy,at_interval=context,delivery_context=delivery)
    assert result.error=='projective_same_observed_anatomy_required'
    invalid=replace(pair.left[0],support_basis='pose_torso_support',observed_support_radius_px=1.)
    assert PersonAssociationLedger._support_radius(invalid) is None


def test_two_anatomy_real_sampler_transport_preserves_stats_and_sample_budget(projective_native,monkeypatch):
    from dataclasses import FrozenInstanceError
    packet,tracks,telemetry,calib,selected,anchor,processor,device=body_packet(
        projective_native,1,1,monkeypatch,alternate=True,separated_ankles=True)
    row,=packet.rows
    ankle,torso=row.evidence.anatomy_depths
    assert ankle.sample.anchor_source=='pose_ankle_support' and torso.receipt is row.evidence.body_depth
    assert ankle.receipt.patch.capsules==tuple(tuple(c) for c in device.calls[0][0])
    assert len(ankle.receipt.patch.capsules)==2 and len(device.calls)==2
    assert ankle.sample.depth_p10==2.98 and ankle.sample.depth_p90==3.02
    assert ankle.sample.sample_count==80 and ankle.sample.valid_fraction==.9
    assert ankle.receipt.original_point[1]!=0. and torso.receipt.original_point[1]!=0.
    assert ankle.registered_depth_interval_m is None and ankle.depth_convention is None
    assert ankle.receipt.cohort==torso.receipt.cohort==row.evidence.measurements.cohort
    assert row.evidence.depth.anchor_source=='pose_ankle_support' and selected.raw_depth_m==3.
    with pytest.raises(FrozenInstanceError):ankle.sample=None
    assert 'native_pose_support' not in row.evidence.depth.to_json()
    assert not any(k.startswith('_xc3d_') for k in tracks[0])


def test_separated_ankle_union_mean_is_not_relocated_into_a_capsule(projective_native,monkeypatch):
    packet,tracks,telemetry,calib,selected,anchor,processor,device=body_packet(
        projective_native,1,1,monkeypatch,alternate=True,separated_ankles=True)
    row,=packet.rows
    ankle,torso=row.evidence.anatomy_depths
    u,v=ankle.sample.anchor_uv
    # Real sampler adapter gets TWO disjoint original supports. Its median is
    # a union-patch statistic projected at the actual contact mean, not a
    # measured material pixel at that mean. Never relocate/inflate it.
    capsules=tuple(tuple(c) for c in device.calls[0][0])
    original=ankle.receipt.patch
    assert len(capsules)==2 and original.capsules==capsules
    assert all(abs(u-c[0])>c[5] for c in capsules)
    changed=replace(ankle,receipt=replace(ankle.receipt,patch=original))
    assert changed.sample.anchor_uv==(u,v) and changed.receipt.patch.capsules==capsules
    assert changed.sample.depth_median==ankle.sample.depth_median
    assert changed.registered_depth_interval_m is None


@pytest.mark.parametrize('fault',['duplicate','sample','patch','cohort','raster','frame','body','cached'])
def test_anatomy_adapter_does_not_admit_foreign_or_refreshed_originals(projective_native,monkeypatch,fault):
    from noesis.pipelines.world_quality_adapter import capture_quality_input
    packet,tracks,*_=body_packet(projective_native,1,1,monkeypatch,alternate=True)
    row,=packet.rows
    ev=row.evidence
    ankle,torso=ev.anatomy_depths
    values=ev.anatomy_depths
    if fault=='duplicate':values=(ankle,ankle)
    elif fault=='sample':values=(replace(ankle,sample=replace(ankle.sample,sample_count=79)),torso)
    elif fault=='patch':values=(replace(ankle,receipt=replace(ankle.receipt,
        patch=replace(ankle.receipt.patch,capsules=((1.,1.,2.,2.,1.,1.),)))),torso)
    elif fault=='cohort':values=(replace(ankle,receipt=replace(ankle.receipt,
        cohort=ankle.receipt.cohort.model_copy(update={'calibration_revision':'foreign'}))),torso)
    elif fault=='raster':values=(replace(ankle,receipt=replace(ankle.receipt,
        patch=replace(ankle.receipt.patch,image_size=(1280,720)))),torso)
    elif fault=='frame':values=(replace(ankle,receipt=replace(ankle.receipt,
        patch=replace(ankle.receipt.patch,measurement_frame_id=0))),torso)
    elif fault=='body':ev=replace(ev,body=None)
    else:ev=replace(ev,depth=replace(ev.depth,measurement_cached=True))
    if fault=='cached':
        # Private truth may be retained, but cannot become CURRENT on freeze.
        value=capture_quality_input(tracks[0],ev.lifecycle,ev.measurements,ev.resolution,ev.depth,
            ev.body,ev.body_depth,values)
        frozen=freeze_quality_work(packet.private,packet.continuity,(value,))
        assert not frozen.rows[0].depth_current
    else:
        with pytest.raises(ValueError,match='world_quality_original_anatomy'):
            capture_quality_input(tracks[0],ev.lifecycle,ev.measurements,ev.resolution,ev.depth,
                ev.body,ev.body_depth,values)


def late_depth_fixture(n,monkeypatch):
    """Real producer plan/seal/sample/project methods; all camera/SDK data synthetic."""
    body_projection(n)
    work,tracks,telemetry,calib,*_=body_packet(n,1,1,monkeypatch,alternate=True)
    row,=work.rows
    original_depth=row.evidence.depth
    missing=replace(row,evidence=replace(row.evidence,depth=None,body_depth=None,anatomy_depths=()),
        depth_current=False,current_measured=False)
    work=replace(work,rows=(missing,))
    store=hooks._AlignedDepthFrameStore(max_entries=24)
    fusion=hooks._ObjectDepthFusionProcessor(store,(1920,1080),'synthetic','m',True,2)
    fusion.original_depth_rendezvous_enabled=True
    monkeypatch.setattr(fusion,'_extract_instance_mask_payload',lambda _o:None)
    monkeypatch.setattr(fusion,'_attached_pose_keypoints',lambda _o,_b:np.asarray(row.evidence.body.keypoints))
    obj=SimpleNamespace(object_id=row.evidence.lifecycle.tracker_id,class_id=0,confidence=.9)
    frame=SimpleNamespace(source_id=1,frame_number=1,frame_width=1920,frame_height=1080,
        buf_pts=work.private.original.pts_ns)
    plan=fusion._prepare_original_pose_plan(frame,obj,frame_id=1,pts_us=frame.buf_pts//1000,
        source_id=1,bbox=tuple(tracks[0]['bbox']))
    assert plan is not None
    fusion._retain_original_pose_plan(plan)
    fusion.seal_original_depth_work(work)
    assert next(iter(fusion._original_plans.values()))[1] is not None
    device=SyntheticDepthDevice()
    device.ready=False
    device.is_ready=lambda:device.ready
    depth=hooks._AlignedDepthFrame((1,1,plan.pts_us),1,1,plan.pts_us,None,None,
        1920,1080,518,294,'m',True,'synthetic',device)
    store.put(depth)
    calib.camera_calibration_sha256=row.evidence.measurements.cohort.calibration_revision
    calib.frame_contract=n.f.frames[1].frame_transform()
    telemetry.bev_calibration=SimpleNamespace(snapshot=lambda *_:calib)
    # Original body/World snapshot is emitted without the late result. This
    # does not retrofit that publication or assert a native successful join.
    publish_lineage(n,work,tracks)
    return fusion,device,telemetry,work,plan,original_depth


def cadence_packet(n, frame, *, empty=False):
    """Synthetic source input through real acquisition/lifecycle owner methods.

    Unlike the older clock-zero helper, this also represents an explicit
    acquisition reset. Host monotonic time is independently controlled by n.frame.
    """
    from noesis.pipelines.source_acquisition import freeze_owner_retention
    work, tracks = n.f.packet(1, frame, empty=empty)
    pts = frame * 33_333_333
    original = n.pipeline.xc3d_swin_ingress.state.observe_native({
        'receipt_valid': True, 'source_id': 1, 'original_source_id': 1,
        'sdk_source_id': 101, 'original_sdk_source_id': 101,
        'slot': 0, 'original_batch_id': 0, 'scope': frame + 101,
        'original_buffer_scope': frame + 101, 'original_frame': frame,
        'post_tracker_frame': frame, 'original_pts_ns': pts,
        'post_tracker_pts_ns': pts, 'original_ntp_ns': 10_000_000_000 + pts,
        'post_tracker_ntp_ns': 10_000_000_000 + pts})
    retention = freeze_owner_retention(work.continuity,
        n.f.registries[1].identity_retention_keys(1, media_pts_ns=pts,
            max_gap_ns=430_000_000), 430_000_000, original)
    private = replace(work.private, original=original, owner_retention=retention,
        clock_epoch=original.source_clock_epoch)
    return replace(freeze_quality_work(private, work.continuity,
        tuple(row.evidence for row in work.rows)), ingress_monotonic_s=time.monotonic()), tracks


def cadence_fusion(n, monkeypatch):
    fusion = hooks._ObjectDepthFusionProcessor(hooks._AlignedDepthFrameStore(),
        (1920, 1080), 'synthetic', 'm', True, 2)
    fusion.original_depth_rendezvous_enabled = True
    monkeypatch.setattr(fusion, '_max_hz_per_track', lambda: 5.)
    # Only sampling-resource reconciliation is under test here; late sampling
    # and real producer -> World behavior have separate actual-method tests.
    monkeypatch.setattr(fusion, 'drain_original_depth_work', lambda *_: ())
    n.pipeline.object_depth_fusion_processor = fusion
    return fusion, SimpleNamespace(object_id=9)


def test_cadence_actual_FIFO_reset_clears_old_marker_not_plain_backward_PTS(projective_native, monkeypatch):
    n = projective_native
    fusion, obj = cadence_fusion(n, monkeypatch)
    n.frame[0] = 300
    old, tracks = cadence_packet(n, 300)
    publish_lineage(n, old, tracks)
    fusion._original_plan_attempts[(1, 9)] = 10_000_000
    assert not fusion._sample_due(source_id=1, obj_meta=obj, pts_us=0)
    assert not fusion._sample_due(source_id=1, obj_meta=obj, pts_us=200_000)
    assert fusion._sample_due(source_id=1, obj_meta=obj, pts_us=10_200_000)
    fusion._result_cache[(1, 9)] = {'_sample_ts_us': 10_000_000}
    fusion._depth_retry_tracks.add((1, 9))
    before = n.f.world.current_snapshot()
    # The real canonical registry and acquisition state explicitly advance
    # their epochs; a low PTS alone never does this.
    n.f.registries[1].reset_source(1)
    n.pipeline.xc3d_swin_ingress.state.reset(1, 'explicit_test_reset')
    n.frame[0] = 301  # Host monotonic time does NOT rewind with source media.
    reset, tracks = cadence_packet(n, 1)
    assert reset.denial is None and reset.private.clock_epoch == 1
    publish_lineage(n, reset, tracks)
    assert n.f.world.current_snapshot() is not before
    assert (1, 9) not in fusion._original_plan_attempts
    assert (1, 9) not in fusion._result_cache and (1, 9) not in fusion._depth_retry_tracks
    assert fusion._sample_due(source_id=1, obj_meta=obj, pts_us=0)
    # Re-arming after the accepted boundary does not weaken ordinary 5 Hz.
    assert fusion._claim_actual_depth_attempt((1, 9), 0, 1)
    assert not fusion._sample_due(source_id=1, obj_meta=obj, pts_us=199_999)
    assert fusion._sample_due(source_id=1, obj_meta=obj, pts_us=200_000)
    fusion._original_plan_attempts[(1, 9)] = 10_000
    n.frame[0] = 302
    current, tracks = cadence_packet(n, 2)
    publish_lineage(n, current, tracks)
    assert fusion._original_plan_attempts[(1, 9)] == 10_000
    assert not fusion._sample_due(source_id=1, obj_meta=obj, pts_us=0)
    assert not fusion.accept_original_depth_work(old, n.f.ingress.originals)
    assert not fusion.accept_original_depth_work(current, n.f.ingress.originals)
    assert not fusion.accept_original_depth_work(replace(current, denial='rejected'), n.f.ingress.originals)
    assert fusion._original_plan_attempts[(1, 9)] == 10_000


def test_cadence_actual_absence_clears_resources_but_refused_reuse_does_not_rearm(projective_native, monkeypatch):
    from noesis_core.world_service import WorldPublicationRefused
    n = projective_native
    fusion, obj = cadence_fusion(n, monkeypatch)
    work, tracks = cadence_packet(n, 1)
    publish_lineage(n, work, tracks)
    fusion._original_plan_attempts[(1, 9)] = 33_333
    fusion._result_cache[(1, 9)] = {'_sample_ts_us': 33_333}
    n.frame[0] = 2
    absent, tracks = cadence_packet(n, 2, empty=True)
    publish_lineage(n, absent, tracks)
    assert (1, 9) not in fusion._original_plan_attempts and (1, 9) not in fusion._result_cache
    n.frame[0] = 20  # Beyond the unchanged .43s native reappearance retention.
    reappeared, tracks = cadence_packet(n, 20)
    assert reappeared.rows[0].evidence.lifecycle.generation > work.rows[0].evidence.lifecycle.generation
    # A retained numeric raw-ID marker cannot claim a current native owner.
    fusion._original_plan_attempts[(1, 9)] = 10_000_000
    # Existing ordered ownership refuses this fixture's retained-raw reuse at
    # lifecycle_frontier. That is NOT an accepted reset and must not re-arm it.
    # Do not weaken the owner to turn this into a sampling positive case.
    with pytest.raises(WorldPublicationRefused, match='lifecycle_frontier'):
        publish_lineage(n, reappeared, tracks)
    assert fusion._original_plan_attempts[(1, 9)] == 10_000_000
    assert not fusion._sample_due(source_id=1, obj_meta=obj, pts_us=100_000)


def test_cadence_older_FIFO_barrier_preserves_future_sealed_budget(projective_native, monkeypatch):
    n = projective_native
    fusion, device, telemetry, first, plan, _ = late_depth_fixture(n, monkeypatch)
    monkeypatch.setattr(fusion, '_max_hz_per_track', lambda: 5.)
    monkeypatch.setattr(fusion, 'drain_original_depth_work', lambda *_: ())
    n.pipeline.object_depth_fusion_processor = fusion
    assert fusion.accept_original_depth_work(first, n.f.ingress.originals)
    n.frame[0] = 2
    absent, absent_tracks = cadence_packet(n, 2, empty=True)
    n.f.registries[1].reset_source(1)
    n.pipeline.xc3d_swin_ingress.state.reset(1, 'explicit_test_reset')
    n.frame[0] = 3
    later, later_tracks = cadence_packet(n, 1)
    later_life = later.rows[0].evidence.lifecycle
    assert later_life.disposition in {'new', 'reappeared'}
    later_plan = replace(plan, frame_id=1, pts_us=later.private.original.pts_ns // 1000)
    # This future callback already made its SAME budgeted attempt. The actor
    # is still processing an older absence; neither plan nor cache is evidence
    # for that absence, but its budget must not be erased/resampled.
    fusion._retain_original_pose_plan(later_plan)
    key = (1, 9, later_plan.frame_id, later_plan.pts_us)
    fusion._original_plans[key] = (later_plan, replace(later, rows=(later.rows[0],)))
    hint = (later.private.original.run_id, later.private.original.graph_id,
        later_life.source_epoch, later.private.clock_epoch, later_life.generation)
    fusion._result_cache[(1, 9)] = {'_sample_ts_us': later_plan.pts_us,
        '_original_depth_binding': hint, '_original_attempt_frame': 1}
    publish_lineage(n, absent, absent_tracks)
    assert key in fusion._original_plans and fusion._original_plan_attempts[(1, 9)] == later_plan.pts_us
    assert fusion._result_cache[(1, 9)]['_original_attempt_frame'] == 1
    publish_lineage(n, later, later_tracks)
    assert key in fusion._original_plans and fusion._original_plan_attempts[(1, 9)] == later_plan.pts_us
    obj = SimpleNamespace(object_id=9)
    assert not fusion._sample_due(source_id=1, obj_meta=obj, pts_us=later_plan.pts_us + 199_999)
    assert fusion._sample_due(source_id=1, obj_meta=obj, pts_us=later_plan.pts_us + 200_000)


def test_original_depth_pose_rendezvous_late_F_is_sampled_once_not_endpoint_N(projective_native,monkeypatch):
    n=projective_native
    fusion,device,telemetry,work,plan,_=late_depth_fixture(n,monkeypatch)
    assert fusion.drain_original_depth_work(n.f.ingress.originals,n.f.ingress._context,
        plan.ingress_monotonic_s+.01,telemetry)==()
    assert device.calls==[]  # query-only: D(F) is not ready; no wait or sample.
    current,tracks,*_=body_packet(n,1,2,monkeypatch,alternate=True)
    fusion.seal_original_depth_work(current)
    publish_lineage(n,current,tracks)
    before=(n.f.world.current_snapshot(),tuple(n.f.wire),dict(n.f.ingress.originals))
    device.ready=True
    late,=fusion.drain_original_depth_work(n.f.ingress.originals,n.f.ingress._context,
        plan.ingress_monotonic_s+.05,telemetry)
    result,=late.rows
    assert len(device.calls)==2  # SAME one planned ankle+torso attempt, not twice.
    assert result.original is work.rows[0].original and result.evidence.lifecycle.frame_id==1
    assert result.evidence.depth.measurement_frame_id==1 and result.evidence.depth.depth_tensor_frame_id==1
    assert late.ingress_monotonic_s==plan.ingress_monotonic_s
    assert result.evidence.world==work.rows[0].evidence.world and not result.current_measured
    assert len(result.evidence.anatomy_depths)==2 and result.depth_current
    assert before==(n.f.world.current_snapshot(),tuple(n.f.wire),dict(n.f.ingress.originals))
    n.bridge.ingest_historical_quality((late,),n.f.ingress.originals,n.f.ingress._context,
        plan.ingress_monotonic_s+.05)
    assert before==(n.f.world.current_snapshot(),tuple(n.f.wire),dict(n.f.ingress.originals))
    assert fusion.drain_original_depth_work(n.f.ingress.originals,n.f.ingress._context,
        plan.ingress_monotonic_s+.06,telemetry)==() and len(device.calls)==2


@pytest.mark.parametrize('fault',['expired','no_exact_depth','reappeared','absent','reset','wrong_calibration'])
def test_late_original_depth_never_crosses_lifecycle_age_or_exact_binding(projective_native,monkeypatch,fault):
    n=projective_native
    fusion,device,telemetry,work,plan,_=late_depth_fixture(n,monkeypatch)
    current,tracks,*_=body_packet(n,1,2,monkeypatch,alternate=True)
    if fault in {'reappeared','reset'}:
        r=current.rows[0]
        life=replace(r.evidence.lifecycle,**({'disposition':'reappeared'} if fault=='reappeared' else {'source_epoch':1}))
        current=replace(current,rows=(replace(r,evidence=replace(r.evidence,lifecycle=life)),))
    elif fault=='absent':current=replace(current,rows=())
    fusion.seal_original_depth_work(current)
    device.ready=True
    if fault=='no_exact_depth':fusion.depth_store=hooks._AlignedDepthFrameStore()
    if fault=='wrong_calibration':telemetry.bev_calibration=SimpleNamespace(snapshot=lambda *_:None)
    now=plan.ingress_monotonic_s+(.36 if fault=='expired' else .05)
    result=fusion.drain_original_depth_work(n.f.ingress.originals,n.f.ingress._context,now,telemetry)
    assert result==() and device.calls==[]


def test_suppressed_original_retention_is_bounded_private_and_preserves_F(projective_native,monkeypatch):
    from noesis.pipelines.world_quality_adapter import OriginalQualityRetention
    work,tracks,*_=body_packet(projective_native,1,1,monkeypatch,alternate=True)
    queue=OriginalQualityRetention()
    queue.retain(work)
    queue.retain(replace(work,ingress_monotonic_s=work.ingress_monotonic_s+.1))
    one,=queue.take()
    assert one.ingress_monotonic_s==work.ingress_monotonic_s
    for frame in range(130):
        original=replace(work.private.original,original_frame=frame)
        queue.retain(replace(work,private=replace(work.private,original=original)))
    values=queue.take()
    assert len(values)==128 and queue.dropped_rows==2
    assert all(v.ingress_monotonic_s==work.ingress_monotonic_s for v in values)
    assert all(v.rows[0].evidence is work.rows[0].evidence for v in values)
    assert queue.take()==()


def test_actual_enqueue_throttle_keeps_private_F_but_publishes_only_current_N(projective_native,monkeypatch):
    processor=object.__new__(hooks._AnalyticsTelemetryProcessor)
    processor.pipeline=SimpleNamespace(config={'xc3d_world_quality':{
        'contract':'noesis.xc3d.world_quality.v1','enabled':True}},ds_pipeline=None)
    processor._last_tracking_enqueue_ts_by_sensor={}
    processor._last_tracking_enqueue_count_by_sensor={}
    monkeypatch.setattr(processor,'_frame_timestamp_us',lambda frame:frame.buf_pts//1000)
    monkeypatch.setattr(processor,'_tracking_interval_for_frame',lambda _:1.)
    gates=iter((False,True))
    monkeypatch.setattr(processor,'_publish_gate_due',lambda *_a,**_k:next(gates))
    monkeypatch.setattr(processor,'_canonical_publication_backlogged',lambda:False)
    published=[]
    monkeypatch.setattr(processor,'_publish_tracking_work',published.append)
    original_mono=None
    for frame in (1,2):
        work,tracks,*_=body_packet(projective_native,1,frame,monkeypatch,alternate=True)
        meta=SimpleNamespace(frame_number=frame,buf_pts=work.private.original.pts_ns)
        accepted=processor._enqueue_tracking_publication(sensor_id=1,camera_id='kitchen',frame_meta=meta,
            tracks=tracks,footpoints=(),now_ts=float(frame),
            temporal_contract={'media_pts_ns':work.private.media_pts_ns,
                'observed_at_us':work.continuity.observed_at_us},continuity=work.continuity,force=False,
            xc3d_private_evidence=work.private,world_quality_inputs=(work.rows[0].evidence,))
        assert accepted==(frame==2)
        if frame==1:original_mono=time.monotonic()
    publication,=published
    historical,=publication.historical_quality_work
    assert publication.frame_id==publication.world_quality_work.continuity.frame_id==2
    assert historical.continuity.frame_id==historical.rows[0].evidence.lifecycle.frame_id==1
    assert historical.ingress_monotonic_s==original_mono
    assert historical.rows[0].evidence.depth.measurement_frame_id==1
    assert publication.tracks[0]['frame_id']==2
    assert not any(key.startswith('_xc3d') for key in publication.tracks[0])


@pytest.mark.parametrize('fault',[None,'plateau','domain','registration','scale','raw'])
def test_registered_quantile_interval_requires_entire_same_observable_mapping(projective_native,monkeypatch,fault):
    from noesis.calibration.depth_registration import DepthRegistrationEntry, DepthRegistrationBundle, DepthRegistrationManager
    packet,tracks,telemetry,calib,*_=body_packet(projective_native,1,1,monkeypatch,alternate=True)
    ankle=packet.rows[0].evidence.anatomy_depths[0]
    entry=DepthRegistrationEntry(calib.camera_id,1,'piecewise_linear_1d','dav2_anchor_range_m_raw',
        'mapanything_room_range_m','people_tracking_depth_registration',(1.,5.),(1.,2.99,3.01,5.),
        (1.,2.99,3.01,5.),{},{},{},{},{},'synthetic',{})
    if fault=='plateau':entry=replace(entry,knots_registered_m=(1.,3.,3.,5.))
    elif fault=='domain':entry=replace(entry,raw_range_domain_m=(3.,5.))
    telemetry.depth_registration=DepthRegistrationManager(DepthRegistrationBundle({calib.camera_id:entry}))
    receipt=replace(ankle.receipt,registration_status='ok',registration_id=entry.registration_id)
    if fault=='registration':receipt=replace(receipt,registration_id='foreign')
    elif fault=='scale':calib.unit_scale=2.
    elif fault=='raw':receipt=replace(receipt,registration_status='raw_passthrough')
    result=telemetry._original_registered_anatomy_interval(ankle.sample,receipt,calib)
    assert result==((2.98,3.02) if fault is None else None)
