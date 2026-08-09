from pathlib import Path
import sys

import numpy as np
import onnx
from PIL import Image


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import model_setup
from snapshot import SemanticSnapshotEmitter
import sources


def test_all_exported_onnx_models_are_uint8_class_maps():
    for size in model_setup.SUPPORTED_SIZES:
        path = model_setup.MODEL_ROOT / "onnx" / f"yolo26{size}-sem-ade20k_b3.onnx"
        model_setup.validate_onnx_contract(path)
        model = onnx.load(str(path), load_external_data=False)
        assert model.graph.input[0].name == "images"
        assert model.graph.output[0].name == "output0"


def test_three_room_sources_resolve_only_in_memory(tmp_path: Path):
    config = tmp_path / "infer.yaml"
    config.write_text(
        """
streammux:
  width: 1920
  height: 1080
sources:
  - uri_secret: living-room
  - uri_secret: kitchen
  - uri_secret: family-room
""".strip()
        + "\n",
        encoding="utf-8",
    )
    registry = {
        "living-room": "rtsp://camera.invalid/living",
        "kitchen": "rtsp://camera.invalid/kitchen",
        "family-room": "rtsp://camera.invalid/family",
    }
    result = sources.load_three_room_sources(config, camera_registry=registry)
    assert result["sensor_names"] == ["Living Room", "Kitchen", "Family Room"]
    assert result["uris"] == list(registry.values())
    assert "rtsp://" not in config.read_text(encoding="utf-8")


def test_nvinfer_template_requires_semantic_parser():
    text = model_setup.CONFIG_TEMPLATE.read_text(encoding="utf-8")
    assert "network-type=2" in text
    assert "num-detected-classes=150" in text
    assert "output-blob-names=output0" in text
    assert "parse-segmentation-func-name=NvDsInferParseYolo26SemanticADE20K" in text


def test_hover_viewer_uses_portable_repo_relative_assets():
    viewer_root = PIPELINE_ROOT / "viewer"
    script = (viewer_root / "viewer.js").read_text(encoding="utf-8")
    page = (viewer_root / "index.html").read_text(encoding="utf-8")
    assert "living-room_raw.jpg" in script
    assert "living_room_well_lit_class_map.png" in script
    assert "tooltipLabel" in script
    assert "modelSelect" in page
    assert "/home/" not in script
    assert "file://" not in script


def test_snapshot_persists_aligned_raw_frame_and_class_map(tmp_path: Path):
    emitter = SemanticSnapshotEmitter(
        output_dir=tmp_path,
        model_size="s",
        labels=[f"class-{index}" for index in range(150)],
        sensor_names=["Living Room"],
    )
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[:, :, 0] = 180
    class_map = np.array([[0, 1, 1], [2, 2, 2]], dtype=np.uint8)

    emitter._capture(0, rgb, class_map)
    summary = emitter.finalize()
    capture = summary["captures"][0]

    raw_path = Path(capture["raw_path"])
    class_map_path = Path(capture["class_map_path"])
    assert raw_path.exists()
    assert class_map_path.exists()
    assert Image.open(raw_path).size == Image.open(class_map_path).size == (3, 2)
