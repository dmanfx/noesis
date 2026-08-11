from __future__ import annotations

import configparser
from pathlib import Path

from noesis.deimv2_wholebody49_assets import (
    WHOLEBODY49_SIZES,
    materialize_wholebody49_configs,
    resolve_wholebody49_assets,
)


def _read_props(path: Path) -> configparser.SectionProxy:
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    return parser["property"]


def test_wholebody49_promoted_sizes_resolve_to_canonical_assets() -> None:
    assert WHOLEBODY49_SIZES == ("s", "x")

    small = resolve_wholebody49_assets("s")
    assert small["mode"] == "masks"
    assert Path(small["onnx"]).name == "deimv2_wholebody49_dinov3_s_masks_640_ds8norm.onnx"
    assert Path(small["engine"]).name == "deimv2_wholebody49_dinov3_s_masks_640_b3_fp16.engine"

    large = resolve_wholebody49_assets("x")
    assert large["mode"] == "boxes"
    assert Path(large["onnx"]).name == "deimv2_wholebody49_dinov3_x_boxes_640_ds8norm.onnx"
    assert Path(large["engine"]).name == "deimv2_wholebody49_dinov3_x_boxes_640_b3_fp16.engine"


def test_wholebody49_materializes_mask_and_box_configs() -> None:
    small = materialize_wholebody49_configs(
        size="s",
        batch_size=3,
        src_ids=(0, 1, 2),
        logger=None,
    )
    small_props = _read_props(Path(small["pgie_config"]))
    assert small_props["network-type"] == "3"
    assert small_props["parse-bbox-instance-mask-func-name"] == "NvDsInferParseDeimv2Wholebody49"
    assert small_props["output-blob-names"] == "label_xyxy_score;masks"
    assert Path(small_props["onnx-file"]).exists()
    assert Path(small_props["model-engine-file"]).exists()
    assert Path(small_props["custom-lib-path"]).exists()

    large = materialize_wholebody49_configs(
        size="x",
        batch_size=3,
        src_ids=(0, 1, 2),
        logger=None,
    )
    large_props = _read_props(Path(large["pgie_config"]))
    assert large_props["network-type"] == "0"
    assert large_props["parse-bbox-func-name"] == "NvDsInferParseDeimv2Wholebody49Boxes"
    assert large_props["output-blob-names"] == "label_xyxy_score"
    assert Path(large_props["onnx-file"]).exists()
    assert Path(large_props["model-engine-file"]).exists()
    assert Path(large_props["custom-lib-path"]).exists()

    preproc_props = _read_props(Path(small["preprocess_config"]))
    assert preproc_props["tensor-name"] == "images"
    assert preproc_props["network-input-shape"] == "3;3;640;640"
    assert preproc_props["maintain-aspect-ratio"] == "0"
    assert preproc_props["symmetric-padding"] == "0"
