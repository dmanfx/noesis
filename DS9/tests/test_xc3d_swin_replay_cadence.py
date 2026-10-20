"""Actual recorded renderer -> native startup capability guard, no SDK."""
import copy
import importlib.util
from pathlib import Path
import pytest
import yaml
from noesis.v3dt_assets import validate_xc3d_capabilities, V3DTAssetError

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('selection', ['artifact', 'model', 'directories'])
def test_replay_ini_resolves_same_selected_external_artifacts(replay, tmp_path, monkeypatch, selection):
    from noesis.reid_swin_profile import (render_reid_swin_replay_ini,
        validate_reid_swin_source_config, REID_SWIN_CONFIG_NAME,
        REID_SWIN_ENGINE_NAME, REID_SWIN_ONNX_NAME)
    for name in ('NOESIS_DS9_ARTIFACT_ROOT','NOESIS_MODEL_DIR','NOESIS_ENGINE_DIR','NOESIS_ONNX_DIR'):
        monkeypatch.delenv(name, raising=False)
    models = tmp_path/'selected/models'
    onnx, engines = models/'onnx', models/'engines'
    if selection == 'artifact': monkeypatch.setenv('NOESIS_DS9_ARTIFACT_ROOT', str(models.parent))
    elif selection == 'model': monkeypatch.setenv('NOESIS_MODEL_DIR', str(models))
    else:
        monkeypatch.setenv('NOESIS_DS9_ARTIFACT_ROOT', str(tmp_path/'not-selected'))
        monkeypatch.setenv('NOESIS_ONNX_DIR', str(onnx))
        monkeypatch.setenv('NOESIS_ENGINE_DIR', str(engines))
    config, _ = replay
    config['xc3d_swin_replay_every_frame'] = True
    source = tmp_path/'task'/REID_SWIN_CONFIG_NAME
    source.parent.mkdir()
    config['models']['reid']['config-file-path'] = str(source)
    source.write_text(render_reid_swin_replay_ini(config))
    props = validate_reid_swin_source_config(source, pipeline_config=config)
    assert props['onnx-file'] == str(onnx/REID_SWIN_ONNX_NAME)
    assert props['model-engine-file'] == str(engines/REID_SWIN_ENGINE_NAME)
    source.write_text(source.read_text().replace(str(engines/REID_SWIN_ENGINE_NAME),
                                               str(tmp_path/REID_SWIN_ENGINE_NAME)))
    with pytest.raises(ValueError, match='artifact binding changed'):
        validate_reid_swin_source_config(source, pipeline_config=config)


def i150_task_swin(replay, tmp_path, monkeypatch):
    from noesis.reid_swin_profile import render_reid_swin_replay_ini, REID_SWIN_CONFIG_NAME
    from noesis import reid_swin_profile as profile
    rendered, baseline = replay
    # Real checked-out source properties, inert path-bound engine fixture:
    # no SDK import, deserialization or model execution is part of this test.
    engine = tmp_path / profile.REID_SWIN_ENGINE_NAME
    engine.write_bytes(b'inert engine fixture, never deserialized')
    canonical = tmp_path / 'catalog' / REID_SWIN_CONFIG_NAME
    canonical.parent.mkdir()
    canonical.write_text((ROOT/'DS9/pipelines'/REID_SWIN_CONFIG_NAME).read_text().replace(
        '../models/engines/'+profile.REID_SWIN_ENGINE_NAME, str(engine)).replace(
        '../models/onnx/'+profile.REID_SWIN_ONNX_NAME, str(tmp_path/profile.REID_SWIN_ONNX_NAME)))
    monkeypatch.setattr(profile, '_canonical_swin_ini', lambda:canonical)
    for config in (rendered, baseline): config['models']['reid']['engine'] = str(engine)
    rendered['xc3d_swin_replay_every_frame'] = True
    source = tmp_path / 'swin-every-frame' / REID_SWIN_CONFIG_NAME
    source.parent.mkdir()
    source.write_text(render_reid_swin_replay_ini(rendered))
    rendered['models']['reid']['config-file-path'] = str(source)
    return rendered, baseline, source


def test_i150_actual_replay_source_preflight_and_effective_profile(replay, tmp_path, monkeypatch):
    from noesis.reid_swin_profile import (validate_reid_swin_source_config,
        validate_reid_swin_nvinfer_properties, load_reid_swin_nvinfer_properties)
    from noesis.pipelines.source_acquisition import selected_swin_profile
    from noesis_core.inference_runtime_contract import render_nvinfer_engine_only_config
    import ast
    import logging
    from typing import Any, Dict
    from noesis.reid_swin_profile import validate_reid_swin_model_config
    rendered, baseline, source = i150_task_swin(replay, tmp_path, monkeypatch)
    canonical = ROOT / 'DS9/pipelines/config_infer_secondary_reid_swin.ini'
    before = canonical.read_bytes()
    validate_xc3d_capabilities(rendered, baseline)
    props = validate_reid_swin_source_config(source, pipeline_config=rendered)
    assert props['secondary-reinfer-interval'] == '0'
    with pytest.raises(ValueError, match='secondary-reinfer-interval=12'):
        validate_reid_swin_nvinfer_properties(props)
    # Compile ONLY the actual pure preflight function; importing the whole
    # runtime would attest/load native modules outside this CPU test's scope.
    tree = ast.parse((ROOT/'DS9/noesis/ds9_runtime_core.py').read_text())
    preflight = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == '_preflight_reid_profile')
    env = {'Path':Path, 'Dict':Dict, 'Any':Any, 'logging':logging,
        '_validate_reid_swin_model_config':validate_reid_swin_model_config,
        '_validate_reid_swin_source_config':validate_reid_swin_source_config,
        '_resolve_pipeline_cfg_path':lambda yaml_path, raw: Path(raw) if Path(raw).is_absolute() else ROOT/raw}
    exec(compile(ast.Module(body=[preflight], type_ignores=[]), '<actual-swin-preflight>', 'exec'), env)
    env['_preflight_reid_profile'](rendered, tmp_path/'replay.yaml', logging.getLogger('i150'))
    engine = Path(rendered['models']['reid']['engine'])
    effective = tmp_path/'effective'/source.name
    effective.parent.mkdir()
    effective.write_text(render_nvinfer_engine_only_config(source_config=source,
        engine_path=engine, repo_root=ROOT))
    model = dict(rendered['models']['reid'], **{'config-file-path':str(effective)})
    component = {'config-file-path':str(effective), 'model-engine-file':str(engine),
                 'batch-size':16, 'unique-id':3, 'output-tensor-meta':True}
    selected = selected_swin_profile(effective, source_ini=source, engine_path=engine,
        model_config=model, runtime_properties=component, pipeline_config=rendered)
    assert selected != selected_swin_profile(canonical)  # actual0 fingerprint, not12 relabel
    assert selected == selected_swin_profile(source, pipeline_config=rendered)
    assert canonical.read_bytes() == before
    assert load_reid_swin_nvinfer_properties(canonical)['secondary-reinfer-interval'] == '12'
    # Effective output is still the exact normal engine-only renderer result.
    effective.write_text(effective.read_text().replace('secondary-reinfer-interval=0',
                                                     'secondary-reinfer-interval=1'))
    with pytest.raises(ValueError, match='selected_swin_effective_derivation'):
        selected_swin_profile(effective, source_ini=source, engine_path=engine,
            model_config=model, runtime_properties=component, pipeline_config=rendered)


@pytest.mark.parametrize('fault', ['absent','false','bool-int','nan','infinite','live','loop',
    'replay','world','camera','mux','candidate','async','scale','interval','extra','foreign-engine'])
def test_i150_replay_cadence_is_not_a_profile_or_live_bypass(replay, tmp_path, monkeypatch, fault):
    from noesis.reid_swin_profile import validate_reid_swin_source_config
    rendered, baseline, source = i150_task_swin(replay, tmp_path, monkeypatch)
    if fault == 'absent': rendered.pop('xc3d_swin_replay_every_frame')
    elif fault == 'false': rendered['xc3d_swin_replay_every_frame'] = False
    elif fault == 'bool-int': rendered['xc3d_swin_replay_every_frame'] = 1
    elif fault == 'nan': rendered['xc3d_swin_replay_every_frame'] = float('nan')
    elif fault == 'infinite': rendered['xc3d_swin_replay_every_frame'] = float('inf')
    elif fault == 'live': rendered['sources'][1]['uri'] = 'rtsp://invalid/no-network'
    elif fault == 'loop': rendered['sources'][1]['file-loop'] = True
    elif fault == 'replay': rendered['recorded_replay']['realtime'] = 1
    elif fault == 'world': rendered['xc3d_world_quality']['enabled'] = False
    elif fault == 'camera': rendered['sources'][1]['camera_id'] = 'family-room'
    elif fault == 'mux': rendered['streammux']['live-source'] = False
    elif fault == 'candidate': rendered['xc3d_candidate'] = {}
    elif fault == 'foreign-engine':
        source.write_text(source.read_text().replace('model-engine-file = ', 'model-engine-file = /foreign/'))
    else:
        old,new = {'async':('classifier-async-mode = 0','classifier-async-mode = 1'),
            'scale':('net-scale-factor = 0.01735207','net-scale-factor = 1'),
            'interval':('secondary-reinfer-interval = 0','secondary-reinfer-interval = 1'),
            'extra':('[property]','[property]\ninterval = 0')}[fault]
        assert old in source.read_text()
        source.write_text(source.read_text().replace(old,new))
    with pytest.raises(ValueError): validate_reid_swin_source_config(source, pipeline_config=rendered)
    with pytest.raises(V3DTAssetError): validate_xc3d_capabilities(rendered, baseline)


def test_i150_all_selected_profile_callers_forward_actual_context_and_default_stays12():
    import ast
    from noesis.reid_swin_profile import reid_swin_replay_every_frame, validate_reid_swin_source_config
    assert reid_swin_replay_every_frame({}) is False
    assert reid_swin_replay_every_frame({'xc3d_swin_replay_every_frame':False}) is False
    canonical = ROOT/'DS9/pipelines/config_infer_secondary_reid_swin.ini'
    assert validate_reid_swin_source_config(canonical)['secondary-reinfer-interval'] == '12'
    for relative in ('DS9/noesis/ds9_runtime_core.py','DS9/noesis/pipelines/hooks.py'):
        tree = ast.parse((ROOT/relative).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'selected_swin_profile']
        assert calls and all('pipeline_config' in {k.arg for k in call.keywords} for call in calls)
    for relative in ('DS9/noesis/ds9_runtime_core.py','DS9/scripts/ds9_preflight.py'):
        tree = ast.parse((ROOT/relative).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id.lstrip('_') == 'validate_reid_swin_source_config']
        assert calls and all('pipeline_config' in {k.arg for k in call.keywords} for call in calls)


@pytest.fixture
def replay(tmp_path, monkeypatch):
    script = ROOT/'DS9/scripts/render_recorded_replay_config.py'
    spec = importlib.util.spec_from_file_location('actual_replay_camera_guard', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    task = ROOT/'DS9/config/infer_xc3d_image_regions_task_r1.yaml'
    baseline = yaml.safe_load((ROOT/'DS9/config/infer_yolo26_960_test.yaml').read_text())
    videos = {camera:tmp_path/(camera+'.mp4') for camera in module.CANONICAL_CAMERA_ORDER}
    monkeypatch.setenv('NOESIS_XC3D_ORIGINAL_ROOT', str(ROOT))
    rendered = module.render_replay_config(yaml.safe_load(task.read_text()), videos,
        baseline_path=task, original_root=ROOT)
    return rendered, baseline
