from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_sources_module() -> object:
    module_name = "test_yolo26_seg_depth_sources_module"
    module_path = Path(__file__).resolve().parents[1] / "testpipelines" / "yolo26-seg-depth" / "sources.py"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _payload() -> dict[str, object]:
    return {
        "uris": [
            "rtsp://example/living",
            "rtsp://example/kitchen",
            "rtsp://example/family",
        ],
        "sensor_ids": ["0", "1", "2"],
        "sensor_names": [
            "Living Room Camera",
            "Kitchen Camera",
            "Family Room Camera",
        ],
    }


def test_select_source_uses_adjacent_file_uri_from_infer_yaml(tmp_path: Path) -> None:
    module = _load_sources_module()
    infer_yaml = tmp_path / "infer.yaml"
    infer_yaml.write_text(
        """
sources:
  - element: nvurisrcbin
    uri: rtsp://example/living
    #uri: file:///tmp/living.mp4
  - element: nvurisrcbin
    uri: rtsp://example/kitchen
    #uri: file:///tmp/kitchen.mp4
  - element: nvurisrcbin
    #uri: rtsp://example/family
    uri: file:///tmp/family.mp4
""".strip()
        + "\n",
        encoding="utf-8",
    )

    selected = module.select_source(
        _payload(),
        camera="Family Room Camera",
        frame_size=(1920, 1080),
        infer_config_path=infer_yaml,
        source_mode="file",
    )

    assert selected["uri"] == "file:///tmp/family.mp4"


def test_select_source_uses_rtsp_uri_in_stream_mode_even_if_active_is_file(tmp_path: Path) -> None:
    module = _load_sources_module()
    infer_yaml = tmp_path / "infer.yaml"
    infer_yaml.write_text(
        """
sources:
  - element: nvurisrcbin
    uri: rtsp://example/living
    #uri: file:///tmp/living.mp4
  - element: nvurisrcbin
    uri: rtsp://example/kitchen
    #uri: file:///tmp/kitchen.mp4
  - element: nvurisrcbin
    #uri: rtsp://example/family
    uri: file:///tmp/family.mp4
""".strip()
        + "\n",
        encoding="utf-8",
    )

    selected = module.select_source(
        _payload(),
        camera="Family Room Camera",
        frame_size=(1920, 1080),
        infer_config_path=infer_yaml,
        source_mode="stream",
    )

    assert selected["uri"] == "rtsp://example/family"


def test_select_source_uses_active_uri_mode_from_infer_yaml(tmp_path: Path) -> None:
    module = _load_sources_module()
    infer_yaml = tmp_path / "infer.yaml"
    infer_yaml.write_text(
        """
sources:
  - element: nvurisrcbin
    uri: rtsp://example/living
    #uri: file:///tmp/living.mp4
  - element: nvurisrcbin
    uri: rtsp://example/kitchen
    #uri: file:///tmp/kitchen.mp4
  - element: nvurisrcbin
    #uri: rtsp://example/family
    uri: file:///tmp/family.mp4
""".strip()
        + "\n",
        encoding="utf-8",
    )

    selected = module.select_source(
        _payload(),
        camera="Family Room Camera",
        frame_size=(1920, 1080),
        infer_config_path=infer_yaml,
        source_mode="active",
    )

    assert selected["uri"] == "file:///tmp/family.mp4"
