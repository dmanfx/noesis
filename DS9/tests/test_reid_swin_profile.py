from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import onnx
import yaml

from noesis.reid_swin_profile import (
    REID_SWIN_EMBEDDING_DIM,
    REID_SWIN_ENGINE_NAME,
    REID_SWIN_ONNX_NAME,
    REID_SWIN_ONNX_SHA256,
    REID_SWIN_OUTPUT_LAYER,
    load_reid_swin_nvinfer_properties,
    sha256_file,
    validate_reid_swin_model_config,
    validate_reid_swin_nvinfer_properties,
    validate_reid_swin_onnx_source,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = REPO_ROOT / "DS9"
SOURCE_ONNX_ROOT = (REPO_ROOT / "models" / "onnx").resolve()
PROVENANCE_ONNX_ROOT = DS9_ROOT / "models" / "onnx"


def _normalized_properties(path: Path) -> dict[str, str]:
    props = dict(load_reid_swin_nvinfer_properties(path))
    for key in ("onnx-file", "model-engine-file"):
        props[key] = Path(props[key]).name
    return props


def _characterize_hook(adapter_root: Path, module_name: str = "hooks") -> dict[str, object]:
    script = r'''
import json
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

adapter = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
sys.path = [str(adapter), str(repo)] + [
    value for value in sys.path
    if value and Path(value).resolve() not in {adapter, repo}
]
from importlib import import_module
hooks = import_module(f"noesis.pipelines.{sys.argv[3]}")

cfg = {
    "models": {
        "reid": {
            "gie_id": 3,
            "layer": "fc_pred",
            "embedding_dim": 256,
        }
    }
}
processor_kwargs = dict(
    pipeline=SimpleNamespace(config=cfg),
    tracking_pub=SimpleNamespace(),
    camera_labels={},
    sensor_id_map={},
)
if "publication_gate" in inspect.signature(
    hooks._AnalyticsTelemetryProcessor
).parameters:
    from noesis_core.runtime_publication import RuntimePublicationGate

    processor_kwargs["publication_gate"] = RuntimePublicationGate()
processor = hooks._AnalyticsTelemetryProcessor(**processor_kwargs)
calls = []
class Native:
    @staticmethod
    def extract_reid_embedding(obj, gie_id, layer, dimension, normalize):
        calls.append([gie_id, layer, dimension, normalize])
        return [1.0] + [0.0] * (dimension - 1)
hooks.noesis_reid_meta_ext = Native()
embedding = processor._extract_reid_embedding_native(object())
print(json.dumps({
    "call": calls[0],
    "dimension": int(embedding.size),
    "layer_field": processor._reid_layer_name,
    "dimension_field": processor._reid_embedding_dim,
}, sort_keys=True))
'''
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(adapter_root),
            str(REPO_ROOT),
            module_name,
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


class ReidSwinProfileTests(unittest.TestCase):
    def test_ds8_and_ds9_declare_one_exact_profile(self) -> None:
        ds8_yaml = yaml.safe_load((REPO_ROOT / "config" / "infer.yaml").read_text(encoding="utf-8"))
        ds9_yaml = yaml.safe_load((DS9_ROOT / "config" / "infer.yaml").read_text(encoding="utf-8"))
        validate_reid_swin_model_config(ds8_yaml["models"]["reid"])
        validate_reid_swin_model_config(ds9_yaml["models"]["reid"])

        ds8_ini = REPO_ROOT / "pipelines" / "config_infer_secondary_reid_swin.ini"
        ds9_ini = DS9_ROOT / "pipelines" / "config_infer_secondary_reid_swin.ini"
        validate_reid_swin_nvinfer_properties(load_reid_swin_nvinfer_properties(ds8_ini))
        validate_reid_swin_nvinfer_properties(load_reid_swin_nvinfer_properties(ds9_ini))
        self.assertEqual(_normalized_properties(ds8_ini), _normalized_properties(ds9_ini))

    def test_staged_onnx_is_exact_tao_deployable_contract(self) -> None:
        path = SOURCE_ONNX_ROOT / REID_SWIN_ONNX_NAME
        validate_reid_swin_onnx_source(path)
        self.assertEqual(sha256_file(path), REID_SWIN_ONNX_SHA256)

        model = onnx.load(str(path), load_external_data=False)
        self.assertEqual(model.ir_version, 9)
        self.assertEqual([(row.domain, row.version) for row in model.opset_import], [("", 14)])
        self.assertEqual(model.graph.input[0].name, "input")
        self.assertEqual(model.graph.output[0].name, REID_SWIN_OUTPUT_LAYER)
        input_dims = model.graph.input[0].type.tensor_type.shape.dim
        self.assertEqual([row.dim_value for row in input_dims[1:]], [3, 256, 128])
        final_weight = next(row for row in model.graph.initializer if row.name == "onnx::MatMul_13204")
        self.assertEqual(list(final_weight.dims), [768, REID_SWIN_EMBEDDING_DIM])
        self.assertFalse(
            any(row.data_location == onnx.TensorProto.EXTERNAL for row in model.graph.initializer)
        )

        provenance = json.loads(
            (
                PROVENANCE_ONNX_ROOT
                / f"{path.name}.provenance.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(provenance["output_sha256"], REID_SWIN_ONNX_SHA256)
        self.assertEqual(provenance["output_tensors"], {"fc_pred": ["batch", 256]})
        self.assertEqual(
            provenance["staging_aggregate_sha256"],
            "ddcac1498c68ad4d27065ffe0ecd3e1ba034fb9aa747bc3119d42cbc0c5cde99",
        )

    def test_ds8_ds9_hooks_request_fc_pred_256_from_native_bridge(self) -> None:
        expected = {
            "call": [3, "fc_pred", 256, True],
            "dimension": 256,
            "layer_field": "fc_pred",
            "dimension_field": 256,
        }
        self.assertEqual(_characterize_hook(REPO_ROOT), expected)
        self.assertEqual(_characterize_hook(REPO_ROOT, "hooks_v3dt_reimpl"), expected)
        self.assertEqual(_characterize_hook(DS9_ROOT), expected)

    def test_runtime_preflight_fails_closed_on_missing_ds9_engine(self) -> None:
        script = r'''
import logging
import sys
from pathlib import Path
import yaml

ds9 = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
sys.path = [str(ds9), str(repo)] + [
    value for value in sys.path
    if value and Path(value).resolve() not in {ds9, repo}
]
from noesis import ds9_runtime_core as runtime

config_path = ds9 / "config" / "infer.yaml"
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
try:
    runtime._preflight_reid_profile(cfg, config_path, logging.getLogger("test"))
except SystemExit as exc:
    if "Canonical DS9 TAO Swin-Tiny ReID engine missing" not in str(exc):
        raise
    print(str(exc))
else:
    raise AssertionError("missing DS9 ReID engine was accepted")
'''
        result = subprocess.run(
            [sys.executable, "-c", script, str(DS9_ROOT), str(REPO_ROOT)],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("--only reid_swin", result.stdout)

    def test_active_ds9_surfaces_have_no_osnet_substitute(self) -> None:
        active = (
            DS9_ROOT / "config" / "infer.yaml",
            DS9_ROOT / "pipelines" / "config_infer_secondary_reid_swin.ini",
            DS9_ROOT / "scripts" / "rebuild_engines.py",
            DS9_ROOT / "scripts" / "stage_canonical_sources.py",
            DS9_ROOT / "scripts" / "run_canonical_engine_maintenance.sh",
            DS9_ROOT / "asset_manifest.yaml",
        )
        for path in active:
            self.assertNotIn("osnet", path.read_text(encoding="utf-8").lower(), path)
        self.assertFalse(
            (DS9_ROOT / "pipelines" / "config_infer_secondary_reid_osnet.ini").exists()
        )

        maintenance = active[4].read_text(encoding="utf-8")
        for token in (
            f'[reid_swin]="{REID_SWIN_ENGINE_NAME}"',
            "NOESIS_REID_SWIN_TRT_TIMEOUT_SECONDS:-2400",
            "NOESIS_REID_SWIN_GPU_GUARD_MB:-11000",
        ):
            self.assertIn(token, maintenance)


if __name__ == "__main__":
    unittest.main()
