"""Private owned-original scalar inspection; no SDK/native execution."""
from dataclasses import dataclass, replace
from types import SimpleNamespace as NS

import pytest

from noesis.pipelines.projective_continuity import _OriginalBodyDiagnostic
from noesis.pipelines.source_acquisition import OriginalCohort


@dataclass(frozen=True)
class Life:
    tracker_id: int = 7
    generation: int = 2


@dataclass(frozen=True)
class Body:
    keypoints: tuple = ((10.,20.,.9),)*17
    bbox_xywh: tuple = (1.,2.,32.5,75.)
    image_patches: tuple = ()
    lifecycle: Life = Life()
    image_size: tuple = (1920,1080)


def sample(media=2., *, raw=7, body=True):
    original=OriginalCohort('run','graph','kitchen',1,3,60,int(media*1e9),0,'source_pts')
    row=NS(original=original,evidence=NS(lifecycle=Life(raw),body=Body() if body else None,
        measurements=None,body_depth=None,anatomy_depths=()),depth_current=False,
        current_measured=False,denial=None)
    return row,Life(raw),NS(raster_id='literal-raster',source=Life(1),image_size=(1920,1080))


def test_default_disabled_does_not_touch_row(monkeypatch):
    monkeypatch.delenv('NOESIS_XC3D_BODY_DIAGNOSTIC_TARGETS_S',raising=False)
    d=_OriginalBodyDiagnostic.from_environment()
    d.observe(None,None,None,None)
    assert d.summary()=={'enabled':False,'targets_s':(),'limit':64,'errors':0,'records':()}


@pytest.mark.parametrize('value', ['1,2,3,4,5','nan','-1','1,1','x','9'*129])
def test_invalid_configuration_refused_before_callback(monkeypatch,value):
    monkeypatch.setenv('NOESIS_XC3D_BODY_DIAGNOSTIC_TARGETS_S',value)
    with pytest.raises(ValueError):
        _OriginalBodyDiagnostic.from_environment()


def test_nearest_original_immutable_and_absent_support():
    d=_OriginalBodyDiagnostic((2.,))
    row,stamp,camera=sample(1.91)
    d.observe(row,stamp,camera,100.)
    closer=replace(row.original,original_frame=62,pts_ns=2010000000)
    row.original=closer
    d.observe(row,stamp,camera,100.1)
    summary=d.summary()
    record=summary['records'][0]
    assert record['original']['original_frame']==62
    assert record['original']['pts_ns']==2010000000
    assert len(record['body']['keypoints'])==17
    assert record['pose_profile'] is None and record['body_depth'] is None
    assert record['hypotheses'] is None and record['anatomy_depths']==()
    assert record['depth_current'] is False
    record['body']['keypoints']=()
    assert len(d.summary()['records'][0]['body']['keypoints'])==17
    assert row.evidence.body==Body()
    absent,stamp,camera=sample(2.,raw=8,body=False)
    d.observe(absent,stamp,camera,100.2)
    assert d.summary()['records'][1]['body'] is None


def test_population_cap_and_outside_or_unavailable_media():
    d=_OriginalBodyDiagnostic((2.,6.,10.,15.))
    for raw in range(80):
        for target in d.targets:
            row,stamp,camera=sample(target,raw=raw)
            d.observe(row,stamp,camera,100.)
    assert len(d.records)==64
    for media in (0.,2.2,200.):
        row,stamp,camera=sample(media,raw=999)
        d.observe(row,stamp,camera,100.)
    assert len(d.records)==64


def test_nonowned_value_only_fails_optional_diagnostic():
    d=_OriginalBodyDiagnostic((2.,))
    row,stamp,camera=sample()
    row.evidence.body=object()
    d.observe(row,stamp,camera,100.)
    assert d.errors==1 and not d.records


def test_actual_owned_quality_row_and_publication(monkeypatch,tmp_path):
    import importlib.util
    import sys
    from pathlib import Path
    for name in ('pyservicemaker','pyds','noesis_pose_meta_ext','noesis_analytics_meta_ext',
                 'noesis_reid_meta_ext','noesis_v3dt_meta_ext','noesis_depth_meta_ext',
                 'noesis_depth_tracking_tensor_ext'):
        monkeypatch.setitem(sys.modules,name,None)
    path=Path(__file__).with_name('test_xc3d_projective_continuity.py')
    spec=importlib.util.spec_from_file_location('body_diagnostic_producer_examples',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)  # Existing helpers only, no old tests run.
    monkeypatch.setenv('NOESIS_XC3D_BODY_DIAGNOSTIC_TARGETS_S','0.033333333')
    generator=module.projective_native.__wrapped__(tmp_path,monkeypatch)
    native=next(generator)
    try:
        packet,tracks=module.physical_packet(native,1,1)
        before=packet.rows
        module.publish_lineage(native,packet,tracks)
        summary=native.bridge.diagnostics()['original_body_inspection_only']
        assert summary['errors']==0 and len(summary['records'])==1
        record=summary['records'][0]
        assert len(record['body']['keypoints'])==17
        assert record['measurement_cohort']['calibration_revision']
        assert record['hypotheses'] and record['camera_binding']['transform_id']
        assert record['original']['pts_ns']==packet.rows[0].original.pts_ns
        assert packet.rows==before
    finally:
        generator.close()


def test_actual_registered_receipt_and_quantiles_are_not_invented():
    from noesis.metadata.object_depth import NativePosePatchSupport,PoseDepthSample
    from noesis.pipelines.world_quality_adapter import OriginalBodyDepth,OriginalAnatomyDepth
    from noesis_core.contracts.world_measurement import WorldMeasurementCohort
    cohort=WorldMeasurementCohort(track_key='k:7:g2',tracker_lifecycle_generation=2,
        camera_id='kitchen',source_id=1,tracker_id=7,frame_id=60,observed_at_us=2000000,
        world_revision='world',calibration_revision='calibration')
    patch=NativePosePatchSupport('pose_torso_support',(10.,20.),((10.,20.,10.,25.,1.,1.),),
        (1920,1080),60,2000000)
    receipt=OriginalBodyDepth(cohort,patch,(1.,2.,3.),2.,3.,'ok','registration',(False,False))
    sample_depth=PoseDepthSample('pose_torso_support',(10.,20.),8,.8,10,2.,1.8,2.2,'good')
    anatomy=OriginalAnatomyDepth(receipt,sample_depth,(2.7,3.3),'camera_z_m')
    row,stamp,camera=sample()
    row.evidence.body_depth=receipt
    row.evidence.anatomy_depths=(anatomy,)
    d=_OriginalBodyDiagnostic((2.,))
    d.observe(row,stamp,camera,100.)
    record=d.summary()['records'][0]
    assert d.errors==0 and record['body_depth']['registered_depth_m']==3.
    assert record['anatomy_depths'][0]['sample']['depth_p10']==1.8
    assert record['anatomy_depths'][0]['registered_depth_interval_m']==(2.7,3.3)


def registered_original(media=1.95,frame=59):
    from noesis.metadata.object_depth import NativePosePatchSupport
    from noesis.pipelines.world_quality_adapter import OriginalBodyDepth
    from noesis_core.contracts.world_measurement import WorldMeasurementCohort
    from reid.mv3dt_person_association import ObservationStamp,MemberKey
    row,_,camera=sample(media)
    stamp=ObservationStamp(member=MemberKey('run','kitchen',1,3,7,2),frame_id=frame,
        acquired_at_s=media,clock_domain='source-media:run',observation_id='original-'+str(frame),
        clock_basis='source-media',media_time_s=media,lifecycle_started_frame=1)
    row.original=replace(row.original,run_id=stamp.member.run_id,source_clock_epoch=stamp.member.source_epoch,original_frame=frame)
    cohort=WorldMeasurementCohort(track_key='k:7:g2',tracker_lifecycle_generation=2,
        camera_id='kitchen',source_id=1,tracker_id=7,frame_id=frame,observed_at_us=2000000,
        world_revision='world',calibration_revision='calibration')
    patch=NativePosePatchSupport('pose_torso_support',(10.,20.),((10.,20.,10.,25.,1.,1.),),
        (1920,1080),frame,row.original.pts_ns//1000)
    row.evidence.body_depth=OriginalBodyDepth(cohort,patch,(1.,2.,3.),2.,3.,'ok','registration',(False,False))
    row.depth_current=True
    return row,stamp,camera


def test_current_registered_original_beats_nearer_cached_without_restamping_or_authority():
    d=_OriginalBodyDiagnostic((2.,));cached,stamp,camera=sample(2.)
    d.observe(cached,stamp,camera,100.)
    row,stamp,camera=registered_original();original=row.original;receipt=row.evidence.body_depth
    d.observe(row,stamp,camera,100.1)
    record=d.summary()['records'][0]
    assert record['original']['original_frame']==59 and record['distance_s']==pytest.approx(.05)
    assert record['selection_current_registered_original'] is True
    assert record['selection_status']=='current_registered_original_within_window'
    assert record['stamp']['frame_id']==59 and record['body_depth']['patch']['measurement_frame_id']==59
    d.observe(cached,sample(2.)[1],camera,100.2)
    assert d.summary()['records'][0]['original']['original_frame']==59
    assert row.original is original and row.evidence.body_depth is receipt


@pytest.mark.parametrize('fault',['cached','bad_registration','old_frame','bad_point','outside'])
def test_invalid_cached_or_outside_original_never_wins_preference(fault):
    d=_OriginalBodyDiagnostic((2.,));base,stamp,camera=sample(1.99)
    d.observe(base,stamp,camera,100.)
    row,stamp,camera=registered_original(1.8 if fault=='outside' else 1.95)
    if fault=='cached':row.depth_current=False
    elif fault=='bad_registration':row.evidence.body_depth=replace(row.evidence.body_depth,registration_id=None)
    elif fault=='old_frame':row.evidence.body_depth=replace(row.evidence.body_depth,patch=replace(row.evidence.body_depth.patch,measurement_frame_id=1))
    elif fault=='bad_point':row.evidence.body_depth=replace(row.evidence.body_depth,original_point=(float('nan'),2.,3.))
    d.observe(row,stamp,camera,100.1)
    record=d.summary()['records'][0]
    assert record['original']['pts_ns']==base.original.pts_ns
    assert record['selection_current_registered_original'] is False
    assert record['selection_status']=='closest_available_original_within_window'
