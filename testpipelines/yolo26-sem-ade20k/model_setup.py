"""Resolve validated YOLO26-sem-ADE20K TensorRT assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnx


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = REPO_ROOT / "models" / "yolo26_sem_ade20k"
LABELS_PATH = MODEL_ROOT / "labels" / "ade20k_labels.txt"
CONFIG_TEMPLATE = PIPELINE_ROOT / "config_infer_yolo26_sem_ade20k.template.ini"
BUILD_DIR = PIPELINE_ROOT / "build"
SUPPORTED_SIZES = ("s", "l")


@dataclass(frozen=True)
class ModelAssets:
    size: str
    weights: Path
    onnx: Path
    engine: Path
    labels: Path


def resolve_assets(size: str) -> ModelAssets:
    normalized = str(size).strip().lower()
    if normalized not in SUPPORTED_SIZES:
        raise ValueError(f"Model size must be one of {SUPPORTED_SIZES}, got {size!r}")
    assets = ModelAssets(
        size=normalized,
        weights=MODEL_ROOT / "weights" / f"yolo26{normalized}-sem-ade20k.pt",
        onnx=MODEL_ROOT / "onnx" / f"yolo26{normalized}-sem-ade20k_b3.onnx",
        engine=MODEL_ROOT / "engines" / f"yolo26{normalized}-sem-ade20k_b3_fp16.engine",
        labels=LABELS_PATH,
    )
    for label, path in (
        ("official checkpoint", assets.weights),
        ("ONNX export", assets.onnx),
        ("TensorRT engine", assets.engine),
        ("ADE20K labels", assets.labels),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"YOLO26{normalized} {label} missing or empty: {path}")
    validate_onnx_contract(assets.onnx)
    return assets


def validate_onnx_contract(path: Path) -> None:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or model.graph.input[0].name != "images":
        raise RuntimeError(f"Expected one ONNX input named images: {path}")
    if len(model.graph.output) != 1 or model.graph.output[0].name != "output0":
        raise RuntimeError(f"Expected one ONNX output named output0: {path}")
    input_type = model.graph.input[0].type.tensor_type
    input_shape = [int(dim.dim_value) for dim in input_type.shape.dim]
    if input_type.elem_type != onnx.TensorProto.FLOAT or input_shape != [3, 3, 640, 640]:
        raise RuntimeError(f"Expected FLOAT images input [3,3,640,640], got {input_shape}: {path}")
    output_type = model.graph.output[0].type.tensor_type
    if output_type.elem_type != onnx.TensorProto.UINT8:
        raise RuntimeError(f"Expected UINT8 semantic class-map output: {path}")
    output_shape = [int(dim.dim_value) for dim in output_type.shape.dim]
    if output_shape != [3, 640, 640]:
        raise RuntimeError(f"Expected semantic output shape [3,640,640], got {output_shape}: {path}")


def materialize_nvinfer_config(size: str, parser_library: Path) -> tuple[Path, ModelAssets]:
    assets = resolve_assets(size)
    if not parser_library.is_file():
        raise FileNotFoundError(f"Semantic parser missing: {parser_library}")
    text = CONFIG_TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "@ENGINE_PATH@": str(assets.engine.resolve()),
        "@LABELS_PATH@": str(assets.labels.resolve()),
        "@CUSTOM_LIB@": str(parser_library.resolve()),
    }
    for token, value in replacements.items():
        if token not in text:
            raise RuntimeError(f"Config template is missing token {token}")
        text = text.replace(token, value)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    output = BUILD_DIR / f"config_infer_yolo26{assets.size}_sem_ade20k_b3.ini"
    output.write_text(text, encoding="utf-8")
    return output, assets
