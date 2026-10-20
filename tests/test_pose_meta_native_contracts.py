from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List
import ast
from pathlib import Path
import pytest

from noesis.pipelines import hooks


@dataclass
class _Rect:
    left: float = 12.0
    top: float = 18.0
    width: float = 140.0
    height: float = 220.0


@dataclass
class _ObjMeta:
    object_id: int = 2
    class_id: int = 0
    confidence: float = 0.9
    rect_params: _Rect = field(default_factory=_Rect)
    _pose_payload: str = ""


@dataclass
class _FrameMeta:
    object_items: List[Any]
    source_id: int = 0
    frame_number: int = 1
    buf_pts: int = 0


class _PoseNativeStub:
    def __init__(self) -> None:
        self.attach_calls = 0

    def extract_pose_keypoints(
        self,
        obj_meta: Any,
        gie_id: int,
        model_w: int,
        model_h: int,
        score_threshold: float,
        letterbox: bool,
    ) -> Dict[str, Any]:
        left = float(getattr(obj_meta.rect_params, "left", 0.0))
        top = float(getattr(obj_meta.rect_params, "top", 0.0))
        keypoints_roi = []
        keypoints_abs = []
        for idx in range(17):
            x = 8.0 + float(idx)
            y = 6.0 + float(idx)
            c = 0.92
            keypoints_roi.append((x, y, c))
            keypoints_abs.append((x + left, y + top, c))
        return {
            "score": 0.9,
            "keypoints_roi": keypoints_roi,
            "keypoints_abs": keypoints_abs,
        }

    def attach_pose_features(
        self,
        _batch_meta: Any,
        obj_meta: Any,
        payload_json: str,
        replace_existing: bool = True,
    ) -> bool:
        self.attach_calls += 1
        obj_meta._pose_payload = str(payload_json)
        return True

    def extract_pose_features(self, obj_meta: Any) -> str:
        return str(getattr(obj_meta, "_pose_payload", ""))


def _pipeline_stub() -> Any:
    return SimpleNamespace(stable_id_mgr=SimpleNamespace(active_tracks={}), frame_size=(1920, 1080), config={})


def test_i141_native_pose_diagnostic_is_optional_same_pass_source_contract():
    source = (Path(__file__).resolve().parents[1] / "DS9/native/noesis_pose_meta_ext.cpp").read_text()
    wrapper = source.split("py::object extract_pose_keypoints(", 1)[1].split("#ifndef NOESIS_POSE_CONTEXT_NATIVE_TEST", 1)[0]
    assert "py::object diagnostics=py::none()" in wrapper
    assert "PyDict_CheckExact(diagnostics.ptr())" in wrapper
    assert wrapper.count("decode_pose_keypoints_payload(") == 1
    assert "copy_float32_tensor_to_host" not in wrapper
    assert 'py::arg("diagnostics") = py::none()' in source
    for field in ("tensor_count", "context_present", "context_valid", "shape_valid",
                  "rows", "cols", "qualifying_rows_seen", "result", "finite_rows_seen",
                  "padding_rows_seen", "score_pass_rows", "class_pass_rows", "bbox_pass_rows",
                  "center_pass_rows", "iou_pass_rows", "max_score", "max_iou"):
        assert f'out["{field}"]' in wrapper
    assert 'diagnostics.is_none()?nullptr:&diagnostic' in wrapper


def test_i156_pre_iou_diagnostic_is_fixed_bounded_unclipped_and_not_a_match_input():
    source = (Path(__file__).resolve().parents[1] / "DS9/native/noesis_pose_meta_ext.cpp").read_text()
    wrapper = source.split("py::object extract_pose_keypoints(", 1)[1].split("#ifndef NOESIS_POSE_CONTEXT_NATIVE_TEST", 1)[0]
    for name in ("pre_iou_parent_bbox_xywh", "pre_iou_candidates_seen", "pre_iou_scan_complete",
                 "pre_iou_invalid_row_index", "pre_iou_candidates"):
        assert f'out["{name}"]' in wrapper
    observer = source.split("inline void observe_context_candidate(", 1)[1].split("inline bool context_row_matches", 1)[0]
    assert "std::array<std::optional<ContextCandidateDiagnostic>,2>" in source
    assert "pre_iou_candidates_seen++" in observer and "std::clamp" not in observer
    assert "std::isfinite(v)" in observer and "{5,6,11,12}" in observer
    assert "copy_float32_tensor_to_host" not in observer
    assert "if(!(iou>=.5)) {" in source
    assert wrapper.count("decode_pose_keypoints_payload(") == 1
    assert "if(!observer) return;" in wrapper


def test_i158_strict_contained_route_is_context_only_complete_and_not_diagnostic_driven():
    source = (Path(__file__).resolve().parents[1] / "DS9/native/noesis_pose_meta_ext.cpp").read_text()
    helper = source.split("inline bool strict_context_torso_contained(", 1)[1].split("inline bool context_row_matches", 1)[0]
    assert "std::clamp" not in helper and "confidence<.35f" in helper
    assert "x<l || x>a || y<t || y>b" in helper
    selector = source.split("inline int unique_context_row(", 1)[1].split("}  // namespace pose", 1)[0]
    assert "if(iou_count==0 && contained_count==1) accepted=contained;" in selector
    assert selector.count("read(row,col)") == 1
    assert selector.index('diagnostic->result="context_nonfinite_row"') < selector.index("contained_count==1")
    assert "if(accepted>=0)" in selector and 'diagnostic->result="context_ambiguous_rows"' in selector
    extraction = source.split("bool extract_best_pose_row(", 1)[1].split("float clipf", 1)[0]
    assert extraction.count("unique_context_row(") == 1
    assert extraction.index("if(context) {") < extraction.index("unique_context_row(")
    assert "score_threshold!=.25f" in extraction and "accessor.cols!=57" in extraction
    assert "context->restored" in extraction


def test_pose_payload_limit_blocks_oversized_attach(monkeypatch) -> None:
    hooks.reset_core_path_instrumentation()
    native = _PoseNativeStub()
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", native)
    monkeypatch.setenv("NOESIS_POSE_META_MAX_JSON_BYTES", "128")

    proc = hooks.PoseFeatureProcessor(pipeline=_pipeline_stub(), gie_id=4)
    obj = _ObjMeta()
    frame = _FrameMeta(object_items=[obj])

    proc.handle_servicemaker_frame(object(), frame)

    assert native.attach_calls == 0
    assert obj._pose_payload == ""


def test_pose_payload_roundtrip_allows_valid_payload(monkeypatch) -> None:
    hooks.reset_core_path_instrumentation()
    native = _PoseNativeStub()
    monkeypatch.setattr(hooks, "noesis_pose_meta_ext", native)
    monkeypatch.setenv("NOESIS_POSE_META_MAX_JSON_BYTES", "65536")

    pipeline = _pipeline_stub()
    proc = hooks.PoseFeatureProcessor(pipeline=pipeline, gie_id=4)
    overlay_proc = hooks.PoseKeypointOverlayProcessor(pipeline=pipeline, gie_id=4)

    obj = _ObjMeta()
    frame = _FrameMeta(object_items=[obj])

    proc.handle_servicemaker_frame(object(), frame)

    assert native.attach_calls == 1
    assert obj._pose_payload

    payload = overlay_proc._extract_pose_payload(obj)
    assert isinstance(payload, dict)
    assert "keypoints_roi" in payload
    assert "keypoints_abs" in payload

    snap = hooks.get_core_path_instrumentation_snapshot()
    counters = snap.get("counters", {})
    assert int(counters.get("tensor_boundary_copy_bytes_total.pose_meta", 0)) > 0


@pytest.mark.parametrize('value', [None, 0, 1, 'true'])
def test_small_roi_context_requires_literal_boolean(value):
    pipeline=_pipeline_stub();pipeline.config={'models':{'pose':{'xc3d_small_roi_context':value}}}
    with pytest.raises(ValueError,match='literal boolean'):
        hooks.attach_pose_feature_hook(pipeline)


def test_small_roi_context_requires_actual_task_and_enabled_profile(monkeypatch):
    pipeline=_pipeline_stub();pipeline.config={'models':{'pose':{'xc3d_small_roi_context':True}}}
    monkeypatch.setattr(hooks,'xc3d_world_quality_enabled',lambda _:False)
    with pytest.raises(ValueError,match='task-only'):
        hooks.attach_pose_feature_hook(pipeline)
    monkeypatch.setattr(hooks,'xc3d_world_quality_enabled',lambda _:True)
    with pytest.raises(RuntimeError,match='enabled pose'):
        hooks.attach_pose_feature_hook(pipeline)


def test_small_roi_context_restore_attachment_precedes_pose_features_source():
    source=Path(hooks.__file__).read_text()
    tree=ast.parse(source)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='attach_pose_feature_hook')
    segment=ast.get_source_segment(source,function)
    assert segment.index('_attach_small_pose_native_context(') < segment.index('Probe("pose_features"')
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_attach_small_pose_native_context')
    helper_source=ast.get_source_segment(source,helper)
    assert '.find(name)' in helper_source
    assert "'noesis_xc3d_sr_anchor_v1', name, tips='src'" in helper_source
    assert 'small pose ROI context requires actual Service Maker/native APIs' in segment


@pytest.mark.parametrize("width", [30.0,64.0,90.0,127.99])
def test_small_roi_context_missing_native_result_does_not_attach_old_small_pose(monkeypatch,width):
    native=_PoseNativeStub();native.extract_pose_keypoints=lambda *args:None
    monkeypatch.setattr(hooks,'noesis_pose_meta_ext',native)
    proc=hooks.PoseFeatureProcessor(pipeline=_pipeline_stub(),gie_id=4,small_roi_context=True)
    cache_calls=[]
    proc._cached_pose_payload=lambda *args: cache_calls.append(args)
    obj=_ObjMeta(rect_params=_Rect(width=width,height=260))
    proc.handle_servicemaker_frame(object(),_FrameMeta(object_items=[obj]))
    assert native.attach_calls==0
    assert cache_calls==[]


@pytest.mark.parametrize("context,width", [(True,128.0),(True,129.0),(False,30.0),(False,90.0)])
def test_small_roi_context_off_or_large_preserves_cache_lookup(monkeypatch,context,width):
    native=_PoseNativeStub();native.extract_pose_keypoints=lambda *args:None
    monkeypatch.setattr(hooks,'noesis_pose_meta_ext',native)
    proc=hooks.PoseFeatureProcessor(pipeline=_pipeline_stub(),gie_id=4,small_roi_context=context)
    cache_calls=[]
    proc._cached_pose_payload=lambda *args: cache_calls.append(args)
    obj=_ObjMeta(rect_params=_Rect(width=width,height=160))
    proc.handle_servicemaker_frame(object(),_FrameMeta(object_items=[obj]))
    assert len(cache_calls)==1
    assert native.attach_calls==0


def test_small_roi_context_default_off_preserves_processor_profile():
    proc=hooks.PoseFeatureProcessor(pipeline=_pipeline_stub(),gie_id=4)
    assert proc.small_roi_context is False
    assert (proc.gie_id,proc.model_size,proc.score_threshold,proc.letterbox)==(4,(640,640),.25,True)


@pytest.mark.parametrize('width,height,contextual',[
    (94.5,125.,False),  # R46 Family44 bent original.
    (103.5,123.,False), # R46 Family50 bent original.
    (46.5,98.5,True),   # R46 Kitchen287 original.
    (65.,100.,True),(.6501*100,100.,False),(.6499*100,100.,True),
    (16.,100.,True),(15.99,100.,False),(127.99,260.,True),(128.,260.,False),
])
def test_pose_context_narrow_geometry_uses_same_native_cache_policy(monkeypatch,width,height,contextual):
    native=_PoseNativeStub();native.extract_pose_keypoints=lambda *args:None
    monkeypatch.setattr(hooks,'noesis_pose_meta_ext',native)
    proc=hooks.PoseFeatureProcessor(pipeline=_pipeline_stub(),gie_id=4,small_roi_context=True)
    calls=[];proc._cached_pose_payload=lambda *args:calls.append(args)
    obj=_ObjMeta(rect_params=_Rect(width=width,height=height))
    proc.handle_servicemaker_frame(object(),_FrameMeta(object_items=[obj]))
    assert len(calls)==(0 if contextual else 1)
    assert native.attach_calls==0


def _actual_rendered_normalized_pose(tmp_path, monkeypatch):
    """Actual renderer + actual runtime/graph path functions, no SDK graph."""
    import importlib.util
    import os
    import yaml
    root=Path(__file__).resolve().parents[1]
    renderer_path=root/'DS9/scripts/render_recorded_replay_config.py'
    spec=importlib.util.spec_from_file_location('pose_context_actual_renderer',renderer_path)
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    task=root/'DS9/config/infer_xc3d_image_regions_task_r1.yaml'
    monkeypatch.setenv('NOESIS_XC3D_ORIGINAL_ROOT',str(root))
    engine_root=tmp_path/'selected-models'/'engines'
    monkeypatch.setenv('NOESIS_ENGINE_DIR',str(engine_root))
    rendered=renderer.render_replay_config(yaml.safe_load(task.read_text()),
        {camera:tmp_path/(camera+'.mp4') for camera in renderer.CANONICAL_CAMERA_ORDER},
        baseline_path=task,original_root=root)
    core=root/'DS9/noesis/ds9_runtime_core.py'
    nodes=ast.parse(core.read_text()).body
    names={'_artifact_dir','_model_dir','_engine_dir','_onnx_dir','_resolve_pipeline_cfg_path'}
    ns={'Path':Path,'Any':Any,'os':os,'DS9_ROOT':root/'DS9','REPO_ROOT':root}
    exec(compile(ast.Module(body=[n for n in nodes if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(core),'exec'),ns)
    engine=str(ns['_resolve_pipeline_cfg_path'](task,rendered['models']['pose']['engine']))
    loader=root/'DS9/noesis/pipelines/deepstream_pipeline.py'
    build=next(n for n in ast.parse(loader.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='build_pipeline')
    normalize=next(n for n in build.body if isinstance(n,ast.FunctionDef) and n.name=='_abs_or_same')
    ns.update({'base_dir':tmp_path})
    exec(compile(ast.Module(body=[normalize],type_ignores=[]),str(loader),'exec'),ns)
    rendered['models']['pose']['engine']=ns['_abs_or_same'](engine)
    return rendered['models']['pose'],engine_root


def test_small_roi_context_actual_renderer_normalization_engine_identity(tmp_path,monkeypatch):
    pose,selected=_actual_rendered_normalized_pose(tmp_path,monkeypatch)
    assert Path(pose['engine']).is_absolute()
    hooks._require_small_pose_engine(pose['engine'],selected)
    hooks._require_small_pose_engine('DS9/models/engines/yolo26n-pose_b3_fp16.engine',selected)


@pytest.mark.parametrize('wrong',['foreign_absolute','wrong_logical'])
def test_small_roi_context_rejects_foreign_engine_identity(tmp_path,monkeypatch,wrong):
    pose,selected=_actual_rendered_normalized_pose(tmp_path,monkeypatch)
    pose['engine']=(str(tmp_path/'foreign'/'yolo26n-pose_b3_fp16.engine') if wrong=='foreign_absolute'
                    else 'DS9/models/engines/yolo26s-pose_b3_fp16.engine')
    with pytest.raises(RuntimeError,match='models.pose.engine'):
        hooks._require_small_pose_engine(pose['engine'],selected)


@pytest.mark.parametrize('foreign',[False,True])
def test_small_roi_context_actual_normalized_hook_startup(tmp_path,monkeypatch,foreign):
    pose,selected=_actual_rendered_normalized_pose(tmp_path,monkeypatch)
    if foreign:pose['engine']=str(tmp_path/'foreign'/'yolo26n-pose_b3_fp16.engine')
    events=[]
    class DS:
        def attach(self,name,probe,*args,**kwargs):
            if args:
                assert probe=='noesis_xc3d_sr_anchor_v1' and args in (
                    ('pose_roi_context_anchor',), ('pose_tracker_ingress_anchor',))
                assert kwargs=={'tips':'src'}
                events.append(args[0])
            else:events.append(probe)
        def __getitem__(self,name):return SimpleNamespace(find=lambda _:object())
    pipeline=_pipeline_stub();pipeline.config={'models':{'pose':pose}}
    pipeline.components={pose['name']:SimpleNamespace(name=pose['name'],config={}),
        'tracker':SimpleNamespace(name='tracker',config={'ll-config-file':str(
            Path(__file__).resolve().parents[1]/'DS9/config/nvtracker.yaml')})}
    pipeline.ds_pipeline=DS()
    monkeypatch.setattr(hooks,'DS9_ENGINE_ROOT',selected)
    monkeypatch.setattr(hooks,'xc3d_world_quality_enabled',lambda _:True)
    monkeypatch.setattr(hooks,'BatchMetadataOperator',object)
    monkeypatch.setattr(hooks,'BufferOperator',object)
    monkeypatch.setattr(hooks,'Probe',lambda name,operator:name)
    monkeypatch.setattr(hooks,'_PoseFeatureOperator',lambda _:object())
    def register():events.append('register');return True
    monkeypatch.setattr(hooks,'noesis_analytics_meta_ext',SimpleNamespace(register_sender_report_anchor_factory=register))
    def controller(*args):
        assert args[1]==pose['name'] and args[3]=='tracker' and args[4:]==(.55,32,90)
        events.append('controller');return SimpleNamespace(close=lambda:None)
    monkeypatch.setattr(hooks,'noesis_pose_meta_ext',SimpleNamespace(SinglePassPoseController=controller))
    if foreign:
        with pytest.raises(RuntimeError,match='models.pose.engine'):
            hooks.attach_pose_feature_hook(pipeline)
        assert events==[]
    else:
        hooks.attach_pose_feature_hook(pipeline)
        assert events==['register','pose_roi_context_anchor','pose_tracker_ingress_anchor','controller','pose_features']


@pytest.mark.parametrize('result',[False,None,0,1,'true'])
def test_small_roi_context_native_factory_failure_is_closed(monkeypatch,result):
    events=[]
    monkeypatch.setattr(hooks,'noesis_analytics_meta_ext',
        SimpleNamespace(register_sender_report_anchor_factory=lambda:result))
    pipeline=SimpleNamespace(ds_pipeline=SimpleNamespace(attach=lambda *a,**k:events.append('attach')))
    with pytest.raises(RuntimeError,match='native anchor factory registration failed'):
        hooks._attach_small_pose_native_context(pipeline,SimpleNamespace(name='yolo26_pose'),lambda *a:events.append('controller'))
    assert events==[]


def test_small_roi_context_missing_native_factory_is_closed(monkeypatch):
    monkeypatch.setattr(hooks,'noesis_analytics_meta_ext',None)
    with pytest.raises(RuntimeError,match='native anchor factory registration failed'):
        hooks._attach_small_pose_native_context(object(),object(),object())
