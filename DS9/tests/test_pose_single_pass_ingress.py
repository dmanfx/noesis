"""Direct scalar production fixture + inert graph/profile wiring; no SDK/GPU."""
import ast
import configparser
from pathlib import Path
import subprocess
import json

ROOT = Path(__file__).resolve().parents[2]


def test_literal_full_body_parent_correspondence_and_complete_roster(tmp_path):
    source = ROOT / 'DS9/tests/native/test_pose_single_pass_ingress.cpp'
    exe = tmp_path / 'pose-full-body-parent'
    subprocess.run(['c++', '-std=c++17', '-O0', '-Wall', '-Wextra', str(source), '-o', str(exe)],
                   check=True, capture_output=True, text=True, timeout=30)
    result = subprocess.run([str(exe), '--full-body-parent'], check=True,
                            capture_output=True, text=True, timeout=5)
    assert 'literal383 full-body / 378+384 / unique roster / anatomy / R79 / full scan PASS' in result.stdout


def test_full_body_route_owns_complete_current_parent_roster_without_tensor_reads():
    text = (ROOT / 'DS9/native/noesis_pose_meta_ext.cpp').read_text()
    ingress = text.split('  void ingress(GstBuffer* buffer)', 1)[1].split('  void tracked(', 1)[0]
    roster = ingress.split('std::vector<noesis::pose::Box> current_parents;', 1)[1].split(
        'guint64 generation;', 1)[0]
    assert 'current_parents.reserve(kContextObjects)' in roster
    assert 'current_parents.size()>=kContextObjects' in roster
    assert 'o->unique_component_id==proposal_uid)continue;' in roster
    assert 'receipt->crop.original_left' in roster and ':box(*o)' in roster
    assert all(binding in roster for binding in ('receipt->source!=frame->source_id',
        'receipt->frame!=frame->frame_num', 'receipt->pts!=frame->buf_pts', 'receipt->restored'))
    assert ingress.index('bool parent_roster_complete=true;') < ingress.index('unsigned crop_id=0;')
    assert 'parent_roster_complete?&current_parents:nullptr' in ingress
    assert ingress.count('copy_float32_tensor_to_host(') == 1
    selector = text.split('inline int unique_context_row(', 1)[1].split('// Scalar-only decisions', 1)[0]
    assert 'current_parents=nullptr' in selector
    assert 'iou_count==0 && contained_count==0 && full_body_count==1' in selector
    assert selector.count('read(row,col)') == 1
    helper = text.split('inline bool full_body_context_owner(', 1)[1].split('template <typename Reader>', 1)[0]
    assert 'read(' not in helper and 'current_parents->size()>128' in helper
    assert 'owners==1 && own_parent' in helper
    assert 't<parent[1] || b>parent[1]+parent[3]' in helper


def test_literal_parent_crop_ambiguity_recovery_preserves_unique_and_contained(tmp_path):
    source = ROOT / 'DS9/tests/native/test_pose_single_pass_ingress.cpp'
    exe = tmp_path / 'pose-parent-ambiguity'
    subprocess.run(['c++', '-std=c++17', '-O0', '-Wall', '-Wextra', str(source), '-o', str(exe)],
                   check=True, capture_output=True, text=True, timeout=30)
    result = subprocess.run([str(exe), '--parent-ambiguity'], check=True,
                            capture_output=True, text=True, timeout=5)
    assert 'parent crop complete705 / ambiguous708 / legacy single / contained / finite PASS' in result.stdout


def test_extra_crop_support_is_promotion_only_after_parent_exclusion():
    text = (ROOT / 'DS9/native/noesis_pose_meta_ext.cpp').read_text()
    candidates = text.split('inline std::vector<CurrentCandidate> current_candidates', 1)[1].split(
        'inline bool duplicate_box', 1)[0]
    assert 'extra_crop_supported' not in candidates
    promotion = text.split('for(const auto& output:outputs){const auto& c=output.c;', 1)[1].split(
        'auto* o=nvds_acquire_obj_meta_from_pool', 1)[0]
    assert promotion.index('if(output.parent)continue;') < promotion.index(
        'extra_crop_supported(c.box,output.crop)') < promotion.index('duplicate_box')
    assert 'inspection_.disposition(output.inspection_token,6)' in promotion


def test_two_complete_parent_rows_need_mutual_same_body_and_no_new_tensor_reads():
    text = (ROOT / 'DS9/native/noesis_pose_meta_ext.cpp').read_text()
    selector = text.split('inline int unique_context_row(', 1)[1].split('// Scalar-only decisions', 1)[0]
    assert 'std::array<std::array<float,57>,2> complete_rows{};' in selector
    assert 'if(complete_count==2 && same_complete_context_body(' in selector
    assert 'accepted=first_complete;' in selector
    assert selector.count('read(row,col)') == 1
    body = text.split('inline bool same_complete_context_body(', 2)[2].split('template<typename References>', 1)[0]
    assert 'same_visible_head(x,crop,y,crop) && same_visible_head(y,crop,x,crop)' in body
    assert 'read(' not in body and 'score>' not in body and 'score<' not in body
    promotion = text.split('for(const auto& output:outputs){const auto& c=output.c;', 1)[1]
    assert 'same_complete_context_body' not in promotion


def test_literal_internal_crop_cuts_and_camera_edges(tmp_path):
    source = ROOT / 'DS9/tests/native/test_pose_single_pass_ingress.cpp'
    exe = tmp_path / 'pose-crop-support'
    subprocess.run(['c++', '-std=c++17', '-O0', '-Wall', '-Wextra', str(source), '-o', str(exe)],
                   check=True, capture_output=True, text=True, timeout=30)
    result = subprocess.run([str(exe), '--crop-support'], check=True,
                            capture_output=True, text=True, timeout=5)
    assert 'literal crop cuts / interior / camera edges / parent exclusion PASS' in result.stdout


def test_same_frame_anatomy_literal_fragment_and_distinct_controls(tmp_path):
    source = ROOT / 'DS9/tests/native/test_pose_single_pass_ingress.cpp'
    exe = tmp_path / 'pose-anatomy'
    subprocess.run(['c++', '-std=c++17', '-O0', '-Wall', '-Wextra', str(source), '-o', str(exe)],
                   check=True, capture_output=True, text=True, timeout=30)
    result = subprocess.run([str(exe), '--anatomy'], check=True, capture_output=True, text=True, timeout=5)
    assert 'literal R78 fragment / nested child / wrong torso / nearby ambiguity' in result.stdout


def test_anatomy_reuses_current_owned_rows_only():
    text = (ROOT / 'DS9/native/noesis_pose_meta_ext.cpp').read_text()
    assert 'std::vector<const Output*> anatomy;' in text
    assert 'if(output.parent)anatomy.push_back(&output)' in text
    assert 'unique_visible_head(c,output.crop,anatomy)' in text
    assert 'if(output.parent)continue;' in text
    assert text.count('rows=noesis::pose::current_candidates(c,') == 1
    # References are callback-local owned rows, not SDK pointers or old frames.
    assert 'std::vector<const Output*> anatomy;' not in text.split('class SinglePassPoseState')[1].split(' public:')[0]


def test_owned_ingress_inspection_bounds_and_noninterference(tmp_path):
    source = ROOT / 'DS9/tests/native/test_pose_single_pass_ingress.cpp'
    exe = tmp_path / 'pose-inspection'
    subprocess.run(['c++', '-std=c++17', '-O0', '-Wall', '-Wextra', str(source), '-o', str(exe)],
                   check=True, capture_output=True, text=True, timeout=30)
    def inspect(flag):
        result = subprocess.run([str(exe), flag], check=True, capture_output=True, text=True, timeout=5)
        assert len(result.stdout.encode()) <= 65536
        return json.loads(result.stdout)
    d = inspect('--inspection')
    assert d['frames'] == [705, 1056, 1153] and d['seen'] == [0, 1, 0]
    assert d['frame_candidate_counts'] == [0, 2, 0]
    assert d['missing_frames'] == 2 and d['binding_mismatch'] == 5
    mother, child = d['records']
    assert (mother['raw_id'], child['raw_id']) == (28, 25)
    assert (mother['origin_uid'], child['origin_uid']) == (1, 4)
    assert child['frame'] == 1056 and child['pts'] == 35200000000 and child['slot'] == 2
    assert mother['row'] == mother['parent_row'] == child['parent_row'] == 1
    assert child['row'] == 0 and len(child['unclipped_joints_xyc']) == 51
    assert mother['unclipped_joints_xyc'][0] == 452 and mother['attached_joints_xyc'][0] == 460
    capped = inspect('--inspection-caps')
    assert capped['overflow'] == 38 and len(capped['records']) <= 64
    assert capped['byte_omitted'] > 0


def test_inspection_wiring_is_read_once_owned_and_not_a_decision():
    text = (ROOT / 'DS9/native/noesis_pose_meta_ext.cpp').read_text()
    assert text.count('std::getenv("NOESIS_POSE_INGRESS_INSPECT")') == 1
    assert 'inspection_lock.try_lock()' in text
    assert 'std::try_to_lock' in text
    assert 'copy.json()' in text and 'd["inspection"]' in text
    assert 'inspection_token' in text and 'o->object_id,box(*o),p->value' in text
    # Compact rows are reused AFTER the existing once-only tensor extraction.
    assert text.count('rows=noesis::pose::current_candidates(c,') == 1


def test_actual_native_scalar_ingress(tmp_path):
    source = ROOT / 'DS9/tests/native/test_pose_single_pass_ingress.cpp'
    exe = tmp_path / 'pose-ingress'
    subprocess.run(['c++', '-std=c++17', '-O0', '-Wall', '-Wextra', str(source), '-o', str(exe)],
                   check=True, capture_output=True, text=True, timeout=30)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=5)
    assert 'mother+child/dedup/distinct' in result.stdout


def test_one_sgie_and_exclusion_before_same_tracker():
    text = (ROOT / 'DS9/noesis/pipelines/deepstream_pipeline.py').read_text()
    ast.parse(text)
    assert 'primary_branch = pose.name if pose_single_pass else tracker_ingress' in text
    assert '_link(pose.name, tracker_ingress)' in text
    assert text.count('if pose is not None and not pose_single_pass:') == 2
    assert text.index('_link(pose.name, tracker_ingress)') < text.index('_link(exclude_component.name, tracker.name)')


def test_original_profile_and_current_consumer_boundary():
    from noesis.reid_swin_profile import validate_reid_swin_nvinfer_properties, REID_SWIN_OBJECT_ORIGINS
    parser = configparser.ConfigParser()
    parser.read(ROOT / 'DS9/pipelines/config_infer_secondary_reid_swin.ini')
    validate_reid_swin_nvinfer_properties(parser['property'])
    assert REID_SWIN_OBJECT_ORIGINS == (1, 4)
    parser['property']['operate-on-gie-id'] = '1'
    import pytest
    with pytest.raises(ValueError):
        validate_reid_swin_nvinfer_properties(parser['property'])
    pose = configparser.ConfigParser()
    pose.read(ROOT / 'DS9/pipelines/config_infer_secondary_yolo26_pose_n_test.ini')
    props = pose['property']
    assert tuple(props[k] for k in ('gie-unique-id','network-type','batch-size','infer-dims',
                 'input-object-min-width','input-object-min-height','operate-on-gie-id')) == (
                 '4','100','3','3;640;640','16','16','-1')


def test_no_proposal_or_borrowed_parent_identity_can_reach_tracker():
    text = (ROOT / 'DS9/native/noesis_pose_meta_ext.cpp').read_text()
    assert 'for(auto* o:remove)nvds_remove_obj_meta_from_frame(frame,o);' in text
    assert 'o->unique_component_id=4;o->object_id=UNTRACKED_OBJECT_ID;' in text
    assert 'proposal/untracked object escaped NvDCF' in text
    assert 'pose tracker original receipt conflict' in text
    assert 'unsupported person origin before Swin' in text
    assert 'std::shared_ptr<SinglePassPoseState> self' in text
    hooks = (ROOT / 'DS9/noesis/pipelines/hooks.py').read_text()
    assert "'world_observation_stage' if small_context else component.name" in hooks
    assert "target.get('minIouDiff4NewTarget')" in hooks


def test_actual_public_anchor_attachment_uses_selected_policy(monkeypatch):
    import sys
    from types import SimpleNamespace
    for name in ('pyservicemaker', 'pyds', 'noesis_pose_meta_ext', 'noesis_analytics_meta_ext',
                 'noesis_reid_meta_ext', 'noesis_v3dt_meta_ext', 'noesis_depth_meta_ext',
                 'noesis_depth_tracking_tensor_ext'):
        monkeypatch.setitem(sys.modules, name, None)
    from noesis.pipelines import hooks
    monkeypatch.setattr(hooks, 'noesis_analytics_meta_ext', SimpleNamespace(
        register_sender_report_anchor_factory=lambda: True))
    calls = []
    class DS:
        def attach(self, *args, **kwargs): calls.append((args, kwargs))
        def __getitem__(self, name):
            return SimpleNamespace(find=lambda anchor: (name, anchor))
    pose = SimpleNamespace(name='yolo26_pose')
    tracker = SimpleNamespace(name='tracker', config={'ll-config-file': str(ROOT / 'DS9/config/nvtracker.yaml')})
    pipeline = SimpleNamespace(ds_pipeline=DS(), components={'tracker': tracker})
    result = hooks._attach_small_pose_native_context(pipeline, pose, lambda *args: args)
    assert result == (('yolo26_pose', 'pose_roi_context_anchor'), 'yolo26_pose',
                      ('tracker', 'pose_tracker_ingress_anchor'), 'tracker', .55, 32, 90)
    assert [c[0][0] for c in calls] == ['yolo26_pose', 'tracker']
    assert all(c[1] == {'tips': 'src'} for c in calls)
    tracker.config = {}
    import pytest
    with pytest.raises(RuntimeError, match='policy path unavailable'):
        hooks._attach_small_pose_native_context(pipeline, pose, lambda *args: args)


def test_pgie_precursor_copy_exception_is_callsite_and_type_bounded():
    text = (ROOT / 'DS9/native/noesis_pose_meta_ext.cpp').read_text()
    assert 'private_pose_buffer(GstBuffer* source,bool after_pgie=false)' in text
    assert 'private_pose_buffer(b,true)' in text
    assert 'private_pose_buffer(buffer)' in text  # existing strict posttracker caller
    assert 'user.base_meta.meta_type!=NVDS_PREPROCESS_BATCH_META' in text
    assert 'p.target_unique_ids.size()!=1 || p.target_unique_ids[0]!=1' in text
    assert 'user.base_meta.batch_meta!=&batch' in text
    assert 'pose cannot omit a still-required preprocess tensor' in text
    assert 'pose batch userdata type loss/substitution' in text
    for guard in ('next==old', 'next->max_frames_in_batch!=old->max_frames_in_batch',
                  'gst_buffer_n_memory(copy)!=gst_buffer_n_memory(source)',
                  'gst_buffer_peek_memory(copy,n)!=gst_buffer_peek_memory(source,n)'):
        assert guard in text
