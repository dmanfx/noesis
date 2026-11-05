from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import numpy as np
from noesis_core.object_spatial import ObjectSpatialSampler
spec=importlib.util.spec_from_file_location('object_spatial_fixture',Path(__file__).with_name('test_view_coverage.py'))
fmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(fmod)

def test_original_depth_declared_fisheye_surface_and_mismatched_frame(tmp_path):
    f=fmod.fixture(tmp_path)
    cal=replace(f.snapshot,projection_metadata_status='declared',projection_model='opencv_fisheye',distortion_model='fisheye',distortion_coefficients=(0.,0.,0.,0.))
    w,h=cal.image_size
    depth=SimpleNamespace(source_id=0,frame_id=2,pts_us=2000,unit='m',is_metric=True,frame_w=w,frame_h=h,depth_device_frame=None,depth_map=np.full((h,w),2.),valid_mask=np.ones((h,w),bool),model_name='depth-fixture')
    calls=[]
    def resolve(**kw):calls.append(kw);return depth,0,0.
    sampler=ObjectSpatialSampler(SimpleNamespace(resolve=resolve),lambda _:cal)
    capture=SimpleNamespace(camera_id='camera-a',source_id=0,frame_id=2,temporal=(1000000,1000000,'estimated',2000000),width=w,height=h)
    measurement,reason=sampler.sample(capture,(w/2-1,h/2-1,2,2))
    assert reason=='measured_surface' and measurement.quantity=='image_center_surface'
    assert measurement.frame_id==2 and measurement.depth_m==2 and measurement.confidence=='estimated'
    assert measurement.world_frame_revision==cal.frame_contract.target_frame.revision
    assert calls[0]['max_age_frames']==0 and calls[0]['wait_ms']==0
    depth.frame_id=1
    assert sampler.sample(capture,(1,1,2,2))==(None,'original_depth_unavailable')
    depth.frame_id=2
    cal=replace(cal,projection_metadata_status='undeclared')
    assert sampler.sample(capture,(1,1,2,2))==(None,'projection_undeclared_or_unsupported')
