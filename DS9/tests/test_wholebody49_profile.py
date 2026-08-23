from __future__ import annotations

import configparser
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import onnx

from DS9.noesis import runtime_config
from noesis.deimv2_wholebody49_assets import (
    materialize_wholebody49_configs,
    resolve_wholebody49_assets,
    validate_wholebody49_pgie_properties,
    validate_wholebody49_preprocess_properties,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = REPO_ROOT / "DS9"
SOURCE_MODEL_ROOT = (REPO_ROOT / "models").resolve()
SOURCE_ONNX_ROOT = SOURCE_MODEL_ROOT / "onnx"
PROVENANCE_ONNX_ROOT = DS9_ROOT / "models" / "onnx"


def _properties(path: Path) -> configparser.SectionProxy:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path, encoding="utf-8")
    return parser["property"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_stage_module():
    path = DS9_ROOT / "scripts" / "stage_canonical_sources.py"
    spec = importlib.util.spec_from_file_location("ds9_stage_canonical_sources", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Wholebody49ProfileTests(unittest.TestCase):
    def test_shared_materializer_resolves_ds9_owned_artifact_roots(self) -> None:
        with tempfile.TemporaryDirectory() as raw_build:
            env = {
                "NOESIS_MODEL_DIR": str(DS9_ROOT / "models"),
                "NOESIS_ONNX_DIR": str(DS9_ROOT / "models" / "onnx"),
                "NOESIS_ENGINE_DIR": str(DS9_ROOT / "models" / "engines"),
                "NOESIS_PIPELINE_DIR": str(DS9_ROOT / "pipelines"),
                "NOESIS_BUILD_DIR": raw_build,
            }
            with mock.patch.dict(os.environ, env, clear=False):
                small = materialize_wholebody49_configs(
                    size="s", batch_size=3, src_ids=(0, 1, 2)
                )
                large = materialize_wholebody49_configs(
                    size="x", batch_size=3, src_ids=(0, 1, 2)
                )

            for assets in (small, large):
                self.assertTrue(str(assets["onnx"]).startswith(str(DS9_ROOT / "models")))
                self.assertTrue(str(assets["engine"]).startswith(str(DS9_ROOT / "models")))
                self.assertTrue(str(assets["parser"]).startswith(str(DS9_ROOT / "pipelines")))
                self.assertFalse(str(assets["engine"]).startswith(str(REPO_ROOT / "models" / "engines")))

            small_props = _properties(Path(small["pgie_config"]))
            large_props = _properties(Path(large["pgie_config"]))
            self.assertEqual(validate_wholebody49_pgie_properties(small_props), "masks")
            self.assertEqual(validate_wholebody49_pgie_properties(large_props), "boxes")
            validate_wholebody49_preprocess_properties(
                _properties(Path(small["preprocess_config"]))
            )

            mask_osd = runtime_config.derive_osd_policy_from_ini(small_props)
            box_osd = runtime_config.derive_osd_policy_from_ini(large_props)
            self.assertEqual(mask_osd, {"process-mode": 1, "display-mask": 1, "display-bbox": 1, "display-text": 1})
            self.assertEqual(box_osd, {"process-mode": 1, "display-mask": 0, "display-bbox": 1, "display-text": 1})

    def test_contract_validator_rejects_tensor_or_mask_drift(self) -> None:
        mask_props = dict(_properties(DS9_ROOT / "pipelines" / "config_infer_primary_deimv2_wholebody49_masks.template.ini"))
        mask_props.update(
            {
                "onnx-file": "model.onnx",
                "model-engine-file": "model.engine",
                "labelfile-path": "classes.txt",
                "custom-lib-path": "parser.so",
                "batch-size": "3",
            }
        )
        self.assertEqual(validate_wholebody49_pgie_properties(mask_props), "masks")
        mask_props["output-blob-names"] = "label_xyxy_score"
        with self.assertRaisesRegex(ValueError, "output-blob-names"):
            validate_wholebody49_pgie_properties(mask_props)

        preproc = dict(_properties(DS9_ROOT / "pipelines" / "config_preproc.ini"))
        preproc.update(
            {
                "network-input-shape": "3;3;640;640",
                "processing-width": "640",
                "processing-height": "640",
                "tensor-name": "wrong",
                "maintain-aspect-ratio": "0",
                "symmetric-padding": "0",
            }
        )
        with self.assertRaisesRegex(ValueError, "tensor-name=images"):
            validate_wholebody49_preprocess_properties(preproc)

    def test_ds9_preflight_fails_closed_on_missing_runtime_engine(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            env = {
                "NOESIS_MODEL_DIR": str(DS9_ROOT / "models"),
                "NOESIS_ONNX_DIR": str(DS9_ROOT / "models" / "onnx"),
                "NOESIS_ENGINE_DIR": str(root / "missing-engines"),
                "NOESIS_PIPELINE_DIR": str(DS9_ROOT / "pipelines"),
                "NOESIS_BUILD_DIR": str(root / "build"),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                assets = materialize_wholebody49_configs(
                    size="s", batch_size=3, src_ids=(0, 1, 2)
                )
            yaml_path = root / "effective.yaml"
            yaml_path.write_text("{}\n", encoding="utf-8")
            cfg = {
                "preprocess": {"config-file": str(assets["preprocess_config"])},
                "models": {
                    "pgie": {
                        "config-file-path": str(assets["pgie_config"]),
                        "engine": str(assets["engine"]),
                    }
                },
            }
            script = r'''
import json
import logging
import sys
from pathlib import Path

ds9 = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
sys.path = [str(ds9), str(repo)] + [
    value for value in sys.path
    if value and Path(value).resolve() not in {ds9, repo}
]
from noesis import ds9_runtime_core as runtime

cfg = json.loads(sys.argv[3])
try:
    runtime._preflight_pgie_profile(
        "wholebody49", cfg, Path(sys.argv[4]), logging.getLogger("test")
    )
except SystemExit as exc:
    message = str(exc)
    if "Wholebody49 DS9 PGIE engine missing" not in message:
        raise
    print(message)
else:
    raise AssertionError("missing DS9 engine was accepted")
'''
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(DS9_ROOT),
                    str(REPO_ROOT),
                    json.dumps(cfg),
                    str(yaml_path),
                ],
                cwd=str(REPO_ROOT),
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("Wholebody49 DS9 PGIE engine missing", result.stdout)

    def test_onnx_sources_have_the_promoted_input_output_contracts(self) -> None:
        expected = {
            "s": {
                "label_xyxy_score": ["N", 1240, 6],
                "masks": ["N", 1240, 80, 80],
            },
            "x": {"label_xyxy_score": ["N", 1240, 6]},
        }
        env = {
            "NOESIS_MODEL_DIR": str(SOURCE_MODEL_ROOT),
            "NOESIS_ONNX_DIR": str(SOURCE_ONNX_ROOT),
            "NOESIS_ENGINE_DIR": str(DS9_ROOT / "models" / "engines"),
            "NOESIS_PIPELINE_DIR": str(DS9_ROOT / "pipelines"),
            "NOESIS_BUILD_DIR": str(DS9_ROOT / "build"),
        }
        with mock.patch.dict(os.environ, env, clear=False):
            for size, expected_outputs in expected.items():
                model = onnx.load(str(resolve_wholebody49_assets(size)["onnx"]), load_external_data=False)
                self.assertEqual(model.opset_import[0].version, 17)
                self.assertEqual(model.graph.input[0].name, "images")
                self.assertEqual(model.graph.input[0].type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
                input_dims = model.graph.input[0].type.tensor_type.shape.dim
                self.assertEqual([dim.dim_value for dim in input_dims[1:]], [3, 640, 640])
                self.assertFalse(
                    any(
                        tensor.data_location == onnx.TensorProto.EXTERNAL
                        for tensor in model.graph.initializer
                    )
                )
                actual_outputs = {}
                for value in model.graph.output:
                    self.assertEqual(value.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
                    dims = value.type.tensor_type.shape.dim
                    actual_outputs[value.name] = [
                        (dim.dim_param or "N") if index == 0 else dim.dim_value
                        for index, dim in enumerate(dims)
                    ]
                self.assertEqual(actual_outputs, expected_outputs)

    def test_ds9_parser_source_matches_ds8_and_exports_both_entrypoints(self) -> None:
        root_source = REPO_ROOT / "pipelines" / "nvdsinfer_deimv2_wholebody49" / "nvdsinfer_deimv2_wholebody49.cpp"
        ds9_source = DS9_ROOT / "pipelines" / "nvdsinfer_deimv2_wholebody49" / "nvdsinfer_deimv2_wholebody49.cpp"
        self.assertEqual(ds9_source.read_bytes(), root_source.read_bytes())
        text = ds9_source.read_text(encoding="utf-8")
        self.assertIn("resolve_exact_layers", text)
        self.assertIn('kLabelLayerName[] = "label_xyxy_score"', text)
        self.assertIn('kMaskLayerName[] = "masks"', text)
        self.assertIn("kQueryCount = 1240", text)
        self.assertIn("set_normalized_bbox", text)
        self.assertNotIn("normalized_boxes", text)
        self.assertNotIn("outputLayersInfo[0]", text)
        self.assertIn("NvDsInferParseDeimv2Wholebody49Boxes", text)

    def test_runtime_and_staging_surfaces_declare_the_profile(self) -> None:
        runtime = (DS9_ROOT / "noesis" / "ds9_runtime_core.py").read_text(encoding="utf-8")
        self.assertIn('"wholebody49"', runtime)
        self.assertIn("_shared_materialize_wholebody49_configs", runtime)
        self.assertIn("_validate_wholebody49_pgie_properties", runtime)
        self.assertIn("_resolve_pgie_selection", runtime)
        self.assertIn('profile == "wholebody49"', runtime)

        stage = _load_stage_module()
        destinations = {destination.as_posix() for _source, destination in stage.CANONICAL_SOURCES}
        self.assertIn(
            "models/onnx/deimv2_wholebody49_dinov3_s_masks_640_ds8norm.onnx",
            destinations,
        )
        self.assertIn(
            "models/onnx/deimv2_wholebody49_dinov3_x_boxes_640_ds8norm.onnx",
            destinations,
        )

        maintenance = (DS9_ROOT / "scripts" / "run_canonical_engine_maintenance.sh").read_text(
            encoding="utf-8"
        )
        for token in (
            '[wholebody49_s_masks]="deimv2_wholebody49_dinov3_s_masks_640_b3_fp16.engine"',
            '[wholebody49_x_boxes]="deimv2_wholebody49_dinov3_x_boxes_640_b3_fp16.engine"',
            "NOESIS_WHOLEBODY49_S_TRT_TIMEOUT_SECONDS:-2400",
            "NOESIS_WHOLEBODY49_S_GPU_GUARD_MB:-10000",
            "NOESIS_WHOLEBODY49_X_TRT_TIMEOUT_SECONDS:-2400",
            "NOESIS_WHOLEBODY49_X_GPU_GUARD_MB:-11000",
        ):
            self.assertIn(token, maintenance)
        self.assertEqual(maintenance.count("NOESIS_WHOLEBODY49_S_GPU_GUARD_MB:-10000"), 1)
        self.assertEqual(maintenance.count("NOESIS_WHOLEBODY49_X_GPU_GUARD_MB:-11000"), 1)

    def test_staged_source_provenance_matches_local_bytes(self) -> None:
        for name in (
            "deimv2_wholebody49_dinov3_s_masks_640_ds8norm.onnx",
            "deimv2_wholebody49_dinov3_x_boxes_640_ds8norm.onnx",
        ):
            model_path = SOURCE_ONNX_ROOT / name
            provenance = json.loads(
                (
                    PROVENANCE_ONNX_ROOT
                    / f"{name}.provenance.json"
                ).read_text(encoding="utf-8")
            )
            digest = _sha256(model_path)
            self.assertEqual(provenance["output_sha256"], digest)
            self.assertEqual(provenance["output_bytes"], model_path.stat().st_size)
            self.assertEqual(provenance["external_tensor_count"], 0)


if __name__ == "__main__":
    unittest.main()
