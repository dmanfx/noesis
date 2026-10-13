from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_watch_profile_preserves_person_consumers_and_fails_closed(tmp_path: Path) -> None:
    code = r'''
import configparser
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import yaml

with (
    mock.patch("noesis.native_artifact_provenance.attest_ds9_native_artifacts"),
    mock.patch("noesis.runtime_paths.require_ds9_native_extension_origins"),
):
    from noesis import ds9_runtime_core as runtime

args = SimpleNamespace(pgie_profile="yolo26-watch", _pgie_profile_explicit=True, size=None)
assert runtime._resolve_pgie_selection(args, "baseline") == ("yolo26-watch", "m")
for mode, size in (("baseline", "l"), ("v3dt", None), ("mv3dt", None)):
    args.size = size
    try:
        runtime._resolve_pgie_selection(args, mode)
    except SystemExit:
        pass
    else:
        raise AssertionError("incompatible watch selection accepted")

logger = logging.getLogger("watch-profile-test")
assets = runtime._materialize_yolo_detect_pgie_ini("yolo26-watch", "m", logger)
p = configparser.ConfigParser()
p.read(assets["pgie_config"])
props = p["property"]
assert Path(props["model-engine-file"]).name == "watch-26m-v1-e44-b3-fp16.engine"
assert props["num-detected-classes"] == "1"
assert props["cluster-mode"] == "4"
assert props["gie-unique-id"] == "1"
assert props["batch-size"] == "3"
assert props["interval"] == "1"
assert props["parse-bbox-func-name"] == "NvDsInferParseYolo"
assert "onnx-file" not in props
assert Path(props["labelfile-path"]).read_text().splitlines() == ["person"]
assert p["class-attrs-0"]["pre-cluster-threshold"] == "0.25"
preproc = configparser.ConfigParser()
preproc.read(assets["preprocess_config"])
assert preproc["property"]["tensor-name"] == "images"
assert preproc["property"]["network-input-shape"] == "3;3;640;640"
base = Path("DS9/config/infer.yaml")
out = runtime._materialize_effective_pipeline_yaml(base, "yolo26-watch", logger, pgie_size="m")
effective = yaml.safe_load(out.read_text())
original = runtime._canonicalize_effective_pipeline_paths(yaml.safe_load(base.read_text()), base)
assert effective["models"]["pose"] == original["models"]["pose"]
assert effective["models"]["reid"] == original["models"]["reid"]
assert Path(effective["models"]["pgie"]["engine"]).name == "watch-26m-v1-e44-b3-fp16.engine"
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT / "DS9"), str(REPO_ROOT)))
    env["NOESIS_BUILD_DIR"] = str(tmp_path)
    env["NOESIS_DEEPSTREAM_MAJOR"] = "9"
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
