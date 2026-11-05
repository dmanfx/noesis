from types import SimpleNamespace
import time
import pytest
from noesis_core.object_tracks import ObjectTrackService, ObjectTrackConflict
from noesis_core.object_observations import ObjectObservationService

@pytest.fixture
def track_fixture(tmp_path):
    mono=[10.];stamp=[1_000_000]
    tracks=ObjectTrackService('run-1',tmp_path/'objects.sqlite',monotonic=lambda:mono[0],clock_us=lambda:stamp[0])
    observations=ObjectObservationService('run-1',track_service=tracks,monotonic=lambda:mono[0],clock_us=lambda:stamp[0])
    def frame(n,epoch=1,objects=True,confidence=.9):
        cap=observations.begin_frame(SimpleNamespace(pipeline_width=640,pipeline_height=480,source_width=1280,source_height=960),camera_id='camera-a',source_id=0,source_epoch=epoch,frame_id=n,
            temporal={'observed_at_us':stamp[0],'captured_at_us':stamp[0],'capture_time_status':'estimated','media_pts_ns':n*1000000})
        if objects:cap.capture(SimpleNamespace(class_id=56,label='chair',object_id=9007199254740995,confidence=confidence,tracker_confidence=.7,rect_params=SimpleNamespace(left=1.,top=2.,width=30.,height=40.)),0)
        cap.submit()
        deadline=time.monotonic()+2
        while observations._queue.unfinished_tasks and time.monotonic()<deadline:time.sleep(.001)
        assert observations._queue.unfinished_tasks==0
    yield SimpleNamespace(tracks=tracks,observations=observations,frame=frame,mono=mono,stamp=stamp,path=tmp_path/'objects.sqlite')
    observations.close();tracks.close()

def command(observation,revision=None,op='11111111-1111-4111-8111-111111111111'):
    return {'actor_id':'owner','operation_id':op,'expected_revision':revision,'selection':{'id':'22222222-2222-4222-8222-222222222222','label':'Desk chair','retired':False,'observation_id':observation}}

def test_native_continuity_selection_automatic_last_seen_and_original_receipt(track_fixture):
    f=track_fixture;f.frame(1);first=f.tracks.snapshot().tracks[0]
    assert first.key.native_id=='9007199254740995'
    c=command(first.last_seen.observation_id);review=f.tracks.prepare(c)
    saved=f.tracks.confirm({'review_token':review.review_token,'command':c})
    assert saved.record.selections[0].state=='observed'
    f.stamp[0]+=100000;f.mono[0]+=.1;f.frame(2,confidence=-.1)
    current=f.tracks.read('owner').record
    assert current.revision==saved.record.revision
    assert current.selections[0].last_seen.frame_id==2
    assert current.selections[0].last_seen.evidence=='tracker_only'
    assert f.tracks.lookup(c).record.selections[0].last_seen.frame_id==1
    f.frame(3,objects=False);assert f.tracks.read('owner').record.selections[0].state=='not_observed'
    f.mono[0]+=6;assert f.tracks.read('owner').record.selections[0].state=='source_lost'
    f.frame(4,epoch=2);assert f.tracks.read('owner').record.selections[0].state=='source_replaced'
    assert f.tracks.read('owner').record.selections[0].spatial is None

def test_gaps_fence_review_and_restart_preserves_last_seen(track_fixture):
    f=track_fixture;f.frame(1);c=command(f.tracks.snapshot().tracks[0].last_seen.observation_id)
    review=f.tracks.prepare(c);f.observations._drop()
    with pytest.raises(ObjectTrackConflict):f.tracks.confirm({'review_token':review.review_token,'command':c})
    f.frame(2);c=command(f.tracks.snapshot().tracks[-1].last_seen.observation_id,op='33333333-3333-4333-8333-333333333333');review=f.tracks.prepare(c)
    f.tracks.confirm({'review_token':review.review_token,'command':c});f.observations.close();f.tracks.close()
    restarted=ObjectTrackService('run-2',f.path,monotonic=lambda:f.mono[0],clock_us=lambda:f.stamp[0])
    try:
        row=restarted.read('owner').record.selections[0]
        assert row.last_seen.frame_id==2 and row.state=='run_replaced'
        assert restarted.read('different-owner').record.selections==()
    finally:restarted.close()

def test_native_termination_and_reused_id_never_resurrect_saved_object(track_fixture):
    f=track_fixture;f.frame(1);first=f.tracks.snapshot().tracks[0]
    c=command(first.last_seen.observation_id);review=f.tracks.prepare(c)
    f.tracks.confirm({'review_token':review.review_token,'command':c})
    cap=f.observations.begin_frame(SimpleNamespace(pipeline_width=640,pipeline_height=480,source_width=1280,source_height=960),camera_id='camera-a',source_id=0,source_epoch=1,frame_id=2,
        temporal={'observed_at_us':f.stamp[0],'captured_at_us':f.stamp[0],'capture_time_status':'estimated','media_pts_ns':2000000})
    cap.capture_lifecycle({'source_id':0,'frame_id':2,'media_pts_ns':2000000,'terminated_available':True,'termination_boundary':'complete_buffer_source_barrier','object_terminated':[{'tracker_id':9007199254740995,'class_id':56}]})
    cap.submit();f.observations._queue.join()
    assert f.tracks.read('owner').record.selections[0].state=='terminated'
    f.frame(3)
    assert f.tracks.read('owner').record.selections[0].state=='terminated'
    assert f.tracks.snapshot().tracks[-1].key!=first.key
