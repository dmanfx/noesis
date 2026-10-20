"""Baseline pose eligibility/config rendering only; no SDK or engine execution."""
from __future__ import annotations

import configparser
from pathlib import Path

import pytest
import yaml

from noesis_core.inference_runtime_contract import materialize_nvinfer_engine_only_config


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "DS9/config/infer_yolo26_960_test.yaml"


@pytest.fixture
def selected_pose(tmp_path):
    pose = yaml.safe_load(BASELINE.read_text())["models"]["pose"]
    source = ROOT / pose["config-file-path"]
    assert source.name == "config_infer_secondary_yolo26_pose_n_test.ini"
    engine = tmp_path / "inert-selected-engine.engine"
    engine.write_bytes(b"Materializer fixture only; never deserialized or executed.")
    derived = materialize_nvinfer_engine_only_config(
        source_config=source, engine_path=engine, output_root=tmp_path / "runtime",
        component_name=pose["name"], repo_root=ROOT,
    )
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(derived)
    return pose, parser["property"], engine


@pytest.mark.parametrize("width,height,eligible", [
    (32.5, 75.0, True), (34.5, 77.0, True), (16.0, 16.0, True),
    (15.999, 77.0, False), (40.0, 15.999, False), (0.0, 75.0, False),
    (0.0, 0.0, False),
])
def test_selected_pose_original_roi_eligibility(selected_pose, width, height, eligible):
    _, properties, _ = selected_pose
    minimum = (int(properties["input-object-min-width"]), int(properties["input-object-min-height"]))
    assert minimum == (16, 16)
    original_rect = (width, height)
    # Installed gstnvinfer should_infer_object checks the ORIGINAL rectangle
    # before the normal even-pixel crop/letterbox; this is not confidence proof.
    admitted = width >= minimum[0] and height >= minimum[1]
    assert admitted is eligible
    assert (width, height) == original_rect


def test_selected_pose_engine_only_renderer_preserves_model_and_preprocessing(selected_pose):
    pose, properties, engine = selected_pose
    assert Path(pose["engine"]).name == "yolo26n-pose_b3_fp16.engine"
    assert (pose["batch_size"], pose["gie_id"], pose["model_size"]) == (3, 4, [640, 640])
    assert (pose["score_threshold"], pose["kpt_threshold"], pose["letterbox"]) == (.25, .35, True)
    expected = {
        "batch-size": "3", "network-mode": "2", "process-mode": "2", "network-type": "100",
        "gie-unique-id": "4", "classifier-async-mode": "0", "infer-dims": "3;640;640",
        "net-scale-factor": "0.003921568627", "model-color-format": "0",
        "maintain-aspect-ratio": "1", "symmetric-padding": "1", "scaling-filter": "1",
        "output-blob-names": "output0", "output-tensor-meta": "1", "disable-output-host-copy": "0",
        "operate-on-gie-id": "-1", "operate-on-class-ids": "0", "secondary-reinfer-interval": "8",
    }
    assert {key: properties[key] for key in expected} == expected
    assert properties["model-engine-file"] == str(engine)
    assert "onnx-file" not in properties and "engine-create-func-name" not in properties
