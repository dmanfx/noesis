from __future__ import annotations

import os
import ast
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_detector_selector_does_not_reach_native_nvinfer_properties() -> None:
    # Exercise the actual nested property adapter without creating a GPU graph.
    source = (REPO_ROOT / "DS9/noesis/pipelines/deepstream_pipeline.py").read_text()
    node = next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.FunctionDef) and node.name == "_nvinfer_props")
    namespace = {"Dict": dict, "Any": object, "_abs_or_same": lambda value: value,
                 "EngineOnlyRuntimeError": RuntimeError}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "nvinfer-property-adapter", "exec"), namespace)
    raw = {"engine": "test.engine", "input_size": [960, 960], "batch_size": 3,
           "gie_id": 1, "attach_tensor_meta": False}
    props = namespace["_nvinfer_props"](raw)
    assert props == {"model-engine-file": "test.engine", "batch-size": 3,
                     "unique-id": 1, "output-tensor-meta": False}
    assert raw["input_size"] == [960, 960]


def test_resolution_selection_preserves_camera_and_secondary_coordinates(tmp_path: Path) -> None:
    code = r'''
import configparser
import logging
from pathlib import Path
from unittest import mock
import yaml

with (
    mock.patch("noesis.native_artifact_provenance.attest_ds9_native_artifacts"),
    mock.patch("noesis.runtime_paths.require_ds9_native_extension_origins"),
):
    from noesis import ds9_runtime_core as runtime

logger = logging.getLogger("resolution-test")
assert runtime._resolve_yolo_detect_assets("yolo26", "m")["engine"].name == "yolo26m_b3_fp16.engine"
assert runtime._resolve_yolo_detect_assets("yolo26-watch", "m")["engine"].name == "watch-26m-v1-e44-b3-fp16.engine"
for profile, size, stem in (("yolo26", "m", "yolo26m"), ("yolo26", "l", "yolo26l"), ("yolo26-watch", "m", "watch-26m-v1-e44")):
    assets = runtime._materialize_yolo_detect_pgie_ini(profile, size, logger, resolution=960)
    assert assets["engine"].name == stem + "_960_b3_fp16.engine"
    parser = configparser.ConfigParser()
    parser.read(assets["preprocess_config"])
    props = parser["property"]
    assert props["network-input-shape"] == "3;3;960;960"
    assert props["processing-width"] == props["processing-height"] == "960"
    assert props["tensor-name"] == "images"

for shape in ([960, 640], [640, 960], [True, True], [960.0, 960.0], [1280, 1280], None):
    try:
        runtime._yolo_detect_input_resolution({"input_size": shape})
    except SystemExit:
        pass
    else:
        raise AssertionError(f"accepted unsupported detector dimensions: {shape}")

base = Path("DS9/config/infer_yolo26_960_test.yaml")
original = runtime._canonicalize_effective_pipeline_paths(yaml.safe_load(base.read_text()), base)
with mock.patch.object(runtime, "_require_nonempty_runtime_engine"):
    effective_path = runtime._materialize_effective_pipeline_yaml(base, "yolo26", logger, pgie_size="m")
effective = yaml.safe_load(effective_path.read_text())
assert effective["models"]["pgie"]["input_size"] == [960, 960]
assert effective["models"]["pose"] == original["models"]["pose"]
assert effective["models"]["reid"] == original["models"]["reid"]
assert effective["streammux"] == original["streammux"]
assert effective["tracker"] == original["tracker"]
assert effective["models"]["pose"]["model_size"] == [640, 640]
assert Path(effective["models"]["pose"]["engine"]).name == "yolo26n-pose_b3_fp16.engine"

# Preflight must reject stale 640 preprocessing before native inference starts.
preproc = Path(effective["preprocess"]["config-file"])
preproc.write_text(preproc.read_text().replace("network-input-shape=3;3;960;960", "network-input-shape=3;3;640;640"))
with mock.patch.object(runtime, "_require_nonempty_runtime_engine"):
    try:
        runtime._preflight_pgie_profile("yolo26", effective, effective_path, logger)
    except SystemExit as exc:
        assert "preprocess dimensions" in str(exc)
    else:
        raise AssertionError("accepted mismatched preprocessing")
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT / "DS9"), str(REPO_ROOT)))
    env["NOESIS_BUILD_DIR"] = str(tmp_path)
    env["NOESIS_DEEPSTREAM_MAJOR"] = "9"
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
