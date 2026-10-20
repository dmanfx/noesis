from __future__ import annotations

import pytest
from pathlib import Path
import subprocess

from noesis.pipelines.hooks import _tracker_lifecycle_ids_for_frame


def _evidence() -> dict[str, object]:
    return {
        "source_id": 0, "frame_id": 42, "media_pts_ns": 1_400_000_000,
        "shadow_available": True, "terminated_available": True,
        "shadow": [{"tracker_id": 7, "class_id": 0, "tracker_state": 2, "tracker_frame_id": 42}],
        "terminated": [],
    }


def _parse(evidence: object) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return _tracker_lifecycle_ids_for_frame(evidence, source_id=0, frame_id=42, media_pts_ns=1_400_000_000)


def test_only_current_inactive_person_is_positive_native_presence() -> None:
    evidence = _evidence()
    evidence["shadow"] += [
        {"tracker_id": 8, "class_id": 0, "tracker_state": 3, "tracker_frame_id": 42},
    ]
    assert _parse(evidence) == ((7,), ())
    evidence["shadow"].append({"tracker_id": 9, "class_id": 0, "tracker_state": 2, "tracker_frame_id": 41})
    with pytest.raises(ValueError,match='no exact current-frame'):
        _parse(evidence)
    evidence["shadow"].pop()
    evidence["terminated"] = [{"tracker_id": 7, "class_id": 0}]
    assert _parse(evidence) == ((), (7,))


@pytest.mark.parametrize("field,value", [("source_id", 1), ("frame_id", 41), ("media_pts_ns", 1)])
def test_native_presence_requires_exact_batch_frame_identity(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="cohort mismatch"):
        _parse({**_evidence(), field: value})


def test_native_presence_requires_explicit_available_metadata_and_bounded_ids() -> None:
    with pytest.raises(ValueError, match="no metadata receipt"):
        _parse({**_evidence(), "shadow_available": False})
    with pytest.raises(ValueError, match="ID bounds"):
        _parse({**_evidence(), "terminated": [{}] * 257})
    with pytest.raises(ValueError, match="duplicate"):
        _parse({**_evidence(), "shadow": _evidence()["shadow"] * 2})
    assert _parse(None) == ((), ())


@pytest.fixture(scope='module')
def native_misc_harness(tmp_path_factory):
    """Compile VERBATIM production grouping function, not a parallel algorithm.

    Scalar ABI-shaped shims only; no SDK library/module/GPU execution.
    """
    root=Path(__file__).resolve().parents[1]
    source=(root/'native/noesis_analytics_meta_ext.cpp').read_text()
    production=source[source.index('struct LifecycleFrame {'):source.index('\n}  // namespace',source.index('struct LifecycleFrame {'))]
    prelude=r'''
#include <array>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <string>
#include <unordered_map>
constexpr unsigned int kMaxLifecycleStreams=64,kMaxLifecycleTargetsPerStream=256;
struct Sample { uint32_t frameNum,age; int trackerState; };
struct Target { Sample* list; uint32_t numObj,numAllocated; uint64_t uniqueId; uint16_t classId; };
struct Stream { Target* list; uint32_t streamID; uint64_t surfaceStreamID; uint32_t numAllocated,numFilled; };
struct NvDsTargetMiscDataBatch { Stream* list; uint32_t numAllocated,numFilled; };
namespace deepstream {
struct UserMetadata { void* value; void* userData() const {return value;} };
struct BatchMetadata { void* value; template<class F> void iterate(F f,int) const {f(UserMetadata{value});} };
}
'''
    main=r'''
int main(int argc,char**argv) {
 std::string mode=argc>1?argv[1]:"shadow";
 std::array<LifecycleFrame,kMaxLifecycleStreams> frames;
 for(unsigned i=0;i<2;++i){frames[i].pad_index=1;frames[i].slot=i;frames[i].sdk_source_id=101;
 frames[i].frame_id=10+i;frames[i].media_pts_ns=1000+100*i;frames[i].source_frame_count=2;}
 Sample samples[2]={{10,5,2},{11,6,2}};
 Target targets[2]={{samples,1,0,7,0},{samples+1,1,0,8,0}};
 Stream streams[3]={{targets,1,1ULL<<32,1,1},{targets+1,1,1ULL<<32,1,1},{nullptr,1,1ULL<<32,0,0}};
 NvDsTargetMiscDataBatch misc{streams,2,2};
 if(mode=="reordered"){std::swap(streams[0],streams[1]);std::swap(frames[0],frames[1]);}
 if(mode=="duplicate"){samples[1].frameNum=10;targets[1].uniqueId=7;}
 if(mode=="stale")samples[1].frameNum=9;
 if(mode=="bounds")streams[1].numFilled=2;
 if(mode=="ambiguous"){frames[1].frame_id=10;samples[1].frameNum=10;}
 if(mode=="empty")streams[1].numFilled=0;
 if(mode=="empty_first")streams[0].numFilled=0;
 if(mode=="all_empty"){streams[0].numFilled=0;streams[1].numFilled=0;}
 if(mode=="empty_reordered"){streams[1].numFilled=0;std::swap(streams[0],streams[1]);std::swap(frames[0],frames[1]);}
 if(mode=="incomplete")misc.numFilled=1;
 if(mode=="mixed_group"){streams[0].numAllocated=2;streams[0].numFilled=2;streams[1].numFilled=0;}
 if(mode=="surface")streams[1].surfaceStreamID++;
 if(mode=="group_limit"){misc.numAllocated=3;misc.numFilled=3;}
 if(mode=="term_duplicate")targets[1].uniqueId=7;
 try {
  copy_tracker_misc(deepstream::BatchMetadata{&misc},1,frames,2,mode!="termination"&&mode!="term_duplicate");
  for(unsigned i=0;i<2;++i){auto&f=frames[i];
   std::cout<<f.frame_id<<" "<<f.media_pts_ns<<" "<<f.shadow_available<<" "<<f.terminated_available<<" "<<f.termination_barrier;
   std::cout<<" "<<f.shadow.size();for(auto&t:f.shadow)std::cout<<" "<<t.tracker_id<<" "<<t.frame_id<<" "<<t.state;
   std::cout<<" "<<f.terminated.size();for(auto id:f.terminated)std::cout<<" "<<id;std::cout<<"\n";}
 }catch(const std::exception&e){std::cerr<<e.what();return 2;}
}
'''
    folder=tmp_path_factory.mktemp('native-lifecycle-scalar')
    path=folder/'fixture.cpp';path.write_text(prelude+production+main)
    result=subprocess.run(['c++','-std=c++17','-O0','-Wall','-Wextra','-Werror',str(path),'-o',str(folder/'fixture')],
        capture_output=True,text=True,timeout=25)
    assert result.returncode==0,result.stderr
    return folder/'fixture'


def _native_rows(binary,mode):
    result=subprocess.run([str(binary),mode],capture_output=True,text=True,timeout=3)
    assert result.returncode==0,result.stderr
    rows=[]
    for line in result.stdout.splitlines():
        values=iter(map(int,line.split()));frame,pts,shadow,term,barrier=[next(values) for _ in range(5)]
        shadows=[{'tracker_id':next(values),'tracker_frame_id':next(values),'tracker_state':next(values),'class_id':0}
                 for _ in range(next(values))]
        terms=[{'tracker_id':next(values),'class_id':0} for _ in range(next(values))]
        rows.append(dict(source_id=1,frame_id=frame,media_pts_ns=pts,shadow_available=bool(shadow),
            terminated_available=bool(term),termination_boundary='complete_buffer_source_barrier' if barrier else 'not_at_buffer_barrier',
            shadow=shadows,terminated=terms))
    return rows


@pytest.mark.parametrize('mode',['shadow','reordered'])
def test_native_repeated_source_distinct_frames_reaches_exact_consumer(native_misc_harness,mode):
    rows=sorted(_native_rows(native_misc_harness,mode),key=lambda r:r['frame_id'])
    assert [_tracker_lifecycle_ids_for_frame(r,source_id=1,frame_id=r['frame_id'],media_pts_ns=r['media_pts_ns']) for r in rows]==[((7,),()),((8,),())]


def test_native_repeated_termination_uses_only_complete_source_barrier(native_misc_harness):
    rows=_native_rows(native_misc_harness,'termination')
    assert rows[0]['terminated']==[] and rows[0]['terminated_available'] is False
    assert _tracker_lifecycle_ids_for_frame(rows[1],source_id=1,frame_id=11,media_pts_ns=1100)==((),(7,8))


@pytest.mark.parametrize('mode',['duplicate','stale','bounds','ambiguous','term_duplicate','surface','group_limit','mixed_group'])
def test_native_repeated_groups_preserve_integrity_refusals(native_misc_harness,mode):
    result=subprocess.run([str(native_misc_harness),mode],capture_output=True,text=True,timeout=3)
    assert result.returncode==2 and result.stderr


@pytest.mark.parametrize('mode',['empty','empty_first','all_empty'])
def test_native_complete_empty_coverage_reaches_actual_consumer(native_misc_harness,mode):
    rows=sorted(_native_rows(native_misc_harness,mode),key=lambda r:r['frame_id'])
    assert all(row['shadow_available'] is True for row in rows)
    assert rows[0 if mode=='empty_first' else 1]['shadow']==[]
    assert [_tracker_lifecycle_ids_for_frame(r,source_id=1,frame_id=r['frame_id'],media_pts_ns=r['media_pts_ns']) for r in rows] == (
        [((),()),((),())] if mode=='all_empty' else
        [((),()),((8,),())] if mode=='empty_first' else [((7,),()),((),())])


def test_native_conflicting_group_order_does_not_infer_unstamped_empty(native_misc_harness):
    rows=sorted(_native_rows(native_misc_harness,'empty_reordered'),key=lambda r:r['frame_id'])
    assert rows[0]['shadow_available'] is True and rows[1]['shadow_available'] is False
    # The production serializer supplies this marker for unresolved repeated
    # frames; shadow_available=False alone also permits termination-only rows.
    rows[1]['lifecycle_unavailable']='repeated_source_shadow_snapshot_not_exact'
    with pytest.raises(ValueError,match='repeated_source_shadow_snapshot_not_exact'):
        _tracker_lifecycle_ids_for_frame(rows[1],source_id=1,frame_id=11,media_pts_ns=1100)


def test_native_incomplete_repeated_group_never_fabricates_current_shadow_receipt(native_misc_harness):
    rows=_native_rows(native_misc_harness,'incomplete')
    assert rows[0]['shadow_available'] is True and rows[1]['shadow_available'] is False
    assert rows[1]['shadow']==[]
    # Actual native serializer retains this precise fail-closed marker.
    source=(Path(__file__).resolve().parents[1]/'native/noesis_analytics_meta_ext.cpp').read_text()
    assert 'current.source_frame_count > 1 && !current.shadow_available' in source
    rows[1]['lifecycle_unavailable']='repeated_source_shadow_snapshot_not_exact'
    with pytest.raises(ValueError,match='repeated_source_shadow_snapshot_not_exact'):
        _tracker_lifecycle_ids_for_frame(rows[1],source_id=1,frame_id=11,media_pts_ns=1100)


def test_repeated_source_epoch_or_pts_order_conflict_is_still_denied():
    from noesis.pipelines.source_acquisition import ordered_current_frames
    from types import SimpleNamespace
    frames=[SimpleNamespace(batch_id=i,pad_index=1,source_id=101,frame_number=10+i,
        buf_pts=1100-100*i,ntp_timestamp=10000+i) for i in range(2)]
    with pytest.raises(ValueError,match='intrabuffer_epoch_or_clock_order'):
        ordered_current_frames(frames)
