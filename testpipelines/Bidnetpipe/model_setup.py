"""Model download + export utilities for BiSeNetV2 ADE20K segmentation."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import onnx
import onnx.shape_inference
import requests
import torch

MODEL_ROOT = Path("models/bisenetv2_ade20k")
WEIGHTS_DIR = MODEL_ROOT / "weights"
ONNX_DIR = MODEL_ROOT / "onnx"
ENGINE_DIR = MODEL_ROOT / "engines"
LABELS_DIR = MODEL_ROOT / "labels"

WEIGHTS_URL = (
    "https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/"
    "model_final_v2_ade20k.pth"
)
LABELS_URL = "https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/objectInfo150.csv"
BISENET_REPO = "https://github.com/CoinCheung/BiSeNet"
BISENET_CONFIG_REL = "configs/bisenetv2_ade20k.py"
BISENET_ADE20K_DATA_REL = "lib/data/ade20k.py"

WEIGHTS_PATH = WEIGHTS_DIR / "model_final_v2_ade20k.pth"
ONNX_PATH = ONNX_DIR / "bisenetv2_ade20k.onnx"
MODEL_INFO_PATH = ONNX_DIR / "model_info.json"
LABELS_CSV_PATH = LABELS_DIR / "objectInfo150.csv"
LABELS_JSON_PATH = LABELS_DIR / "labels.json"

NVINFER_CONFIG_PATH = Path("config/nvinfer_bisenetv2_seg.txt")
ENGINE_PATH = ENGINE_DIR / "bisenetv2_ade20k_b3_fp16.engine"
CUSTOM_PARSER_LIB_PATH = Path(__file__).resolve().parent / "custom" / "libbidnet_segparser.so"


@dataclass
class ModelArtifacts:
    weights_path: Path
    onnx_path: Path
    engine_path: Path
    labels_csv_path: Path
    labels_json_path: Path
    model_info_path: Path
    nvinfer_config_path: Path


class BiSeNetWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, mean: List[float], std: List[float]):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().div(255.0)
        x = (x - self.mean) / self.std
        logits = self.model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        return torch.softmax(logits, dim=1)


def _ensure_dirs() -> None:
    for path in (WEIGHTS_DIR, ONNX_DIR, ENGINE_DIR, LABELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)


def _clone_bisenet_repo(tmp_dir: Path) -> Path:
    repo_dir = tmp_dir / "BiSeNet"
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    subprocess.run(
        ["git", "clone", "--depth", "1", BISENET_REPO, str(repo_dir)],
        check=True,
        env=env,
    )
    return repo_dir


def _load_cfg(repo_dir: Path):
    sys.path.insert(0, str(repo_dir))
    try:
        from configs import set_cfg_from_file  # type: ignore

        cfg = set_cfg_from_file(str(repo_dir / BISENET_CONFIG_REL))
    finally:
        if str(repo_dir) in sys.path:
            sys.path.remove(str(repo_dir))
    return cfg


def _extract_mean_std(repo_dir: Path) -> Tuple[List[float], List[float]]:
    data_path = repo_dir / BISENET_ADE20K_DATA_REL
    text = data_path.read_text()
    mean_match = re.search(r"mean=\(([^)]*)\)", text)
    std_match = re.search(r"std=\(([^)]*)\)", text)
    if not mean_match or not std_match:
        raise RuntimeError("Failed to parse mean/std from ade20k dataset config")
    mean = [float(x.strip()) for x in mean_match.group(1).split(",") if x.strip()]
    std = [float(x.strip()) for x in std_match.group(1).split(",") if x.strip()]
    if len(mean) != 3 or len(std) != 3:
        raise RuntimeError("Unexpected mean/std length parsed from ade20k config")
    return mean, std


def _load_model(repo_dir: Path, cfg, weights_path: Path) -> torch.nn.Module:
    sys.path.insert(0, str(repo_dir))
    try:
        from lib.models import model_factory  # type: ignore

        model = model_factory[cfg.model_type](cfg.n_cats, aux_mode="eval")
    finally:
        if str(repo_dir) in sys.path:
            sys.path.remove(str(repo_dir))
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _export_onnx(repo_dir: Path, cfg, weights_path: Path, onnx_path: Path) -> None:
    mean, std = _extract_mean_std(repo_dir)
    model = _load_model(repo_dir, cfg, weights_path)
    wrapped = BiSeNetWrapper(model, mean=mean, std=std)
    dummy_input = torch.randn(3, 3, cfg.cropsize[0], cfg.cropsize[1], dtype=torch.float32)
    torch.onnx.export(
        wrapped,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["segmentation"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        external_data=False,
    )


def _inspect_onnx(onnx_path: Path, n_classes: int) -> Dict[str, int]:
    model = onnx.load(str(onnx_path))
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    output = model.graph.output[0]
    dims = []
    for dim in output.type.tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        else:
            dims.append(None)
    if len(dims) != 4:
        raise RuntimeError(f"Unexpected output rank for ONNX: {dims}")

    order = None
    height = width = None
    if dims[1] == n_classes:
        order = "NCHW"
        height = dims[2]
        width = dims[3]
        seg_order = 0
    elif dims[3] == n_classes:
        order = "NHWC"
        height = dims[1]
        width = dims[2]
        seg_order = 1
    else:
        raise RuntimeError(f"Unable to determine output layout from dims: {dims}")

    if height is None or width is None:
        raise RuntimeError(f"ONNX output shape missing height/width: {dims}")
    return {
        "order": order,
        "height": int(height),
        "width": int(width),
        "segmentation_output_order": int(seg_order),
    }


def _parse_labels(labels_csv_path: Path) -> Dict[str, object]:
    with labels_csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        id_to_name: Dict[int, str] = {}
        for row in reader:
            idx_raw = row.get("Idx")
            name_raw = row.get("Name")
            if not idx_raw or not name_raw:
                continue
            idx = int(idx_raw) - 1
            id_to_name[idx] = name_raw
    if not id_to_name:
        raise RuntimeError("Parsed zero labels from ADE20K CSV")

    floor_id: Optional[int] = None
    for idx, name in id_to_name.items():
        tokens = [token.strip().lower() for token in name.split(";")]
        if "floor" in tokens or "flooring" in tokens:
            floor_id = idx
            break
    if floor_id is None:
        raise RuntimeError("Failed to locate floor class in ADE20K labels")

    id_list = [id_to_name[i] for i in sorted(id_to_name.keys())]
    return {
        "id_to_name": id_list,
        "floor_class_id": floor_id,
        "free_space_ids": [floor_id],
    }


def _write_labels_json(labels_json_path: Path, labels_info: Dict[str, object]) -> None:
    labels_json_path.write_text(json.dumps(labels_info, indent=2))


def _write_model_info(
    model_info_path: Path,
    cfg,
    output_info: Dict[str, int],
    labels_info: Dict[str, object],
) -> None:
    payload = {
        "input": {
            "batch": 3,
            "channels": 3,
            "height": int(cfg.cropsize[0]),
            "width": int(cfg.cropsize[1]),
        },
        "output": {
            "classes": int(cfg.n_cats),
            "height": int(output_info["height"]),
            "width": int(output_info["width"]),
            "order": output_info["order"],
            "segmentation_output_order": int(output_info["segmentation_output_order"]),
        },
        "labels": {
            "floor_class_id": labels_info["floor_class_id"],
            "free_space_ids": labels_info["free_space_ids"],
        },
    }
    model_info_path.write_text(json.dumps(payload, indent=2))


def _write_nvinfer_config(
    nvinfer_config_path: Path,
    onnx_path: Path,
    engine_path: Path,
    output_order: int,
    num_classes: int,
) -> None:
    nvinfer_config_path.parent.mkdir(parents=True, exist_ok=True)
    contents = "\n".join(
        [
            "[property]",
            f"onnx-file={onnx_path}",
            f"model-engine-file={engine_path}",
            "batch-size=3",
            "network-type=2",
            f"num-detected-classes={int(num_classes)}",
            "network-mode=2",
            "gpu-id=0",
            "model-color-format=0",
            "net-scale-factor=1.0",
            f"segmentation-output-order={output_order}",
            "segmentation-threshold=0.0",
            "gie-unique-id=1",
            "output-blob-names=segmentation",
            f"custom-lib-path={CUSTOM_PARSER_LIB_PATH.resolve()}",
            "parse-segmentation-func-name=NvDsInferParseCustomBiSeNetFloor",
        ]
    )
    nvinfer_config_path.write_text(contents + "\n")


def _safe_write(fn, path: Path, *args, **kwargs) -> bool:
    try:
        fn(path, *args, **kwargs)
        return True
    except PermissionError:
        return False


def ensure_model_artifacts(
    allow_download: bool = True,
    allow_export: bool = True,
    allow_engine_build: bool = True,
) -> ModelArtifacts:
    _ensure_dirs()

    if not WEIGHTS_PATH.exists():
        if not allow_download:
            raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
        _download(WEIGHTS_URL, WEIGHTS_PATH)

    if not LABELS_CSV_PATH.exists():
        if not allow_download:
            raise FileNotFoundError(f"Labels CSV not found: {LABELS_CSV_PATH}")
        _download(LABELS_URL, LABELS_CSV_PATH)

    labels_info = _parse_labels(LABELS_CSV_PATH)
    if not LABELS_JSON_PATH.exists():
        _safe_write(_write_labels_json, LABELS_JSON_PATH, labels_info)

    def _valid_onnx(path: Path) -> bool:
        try:
            model = onnx.load(str(path))
            onnx.checker.check_model(model)
        except Exception:
            return False
        return True

    needs_export = not ONNX_PATH.exists() or not _valid_onnx(ONNX_PATH)
    needs_info = not MODEL_INFO_PATH.exists()

    if needs_export and not allow_export:
        raise FileNotFoundError(f"ONNX not found or invalid: {ONNX_PATH}")

    if needs_export or needs_info:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = _clone_bisenet_repo(Path(tmp))
            cfg = _load_cfg(repo_dir)
            if needs_export:
                _export_onnx(repo_dir, cfg, WEIGHTS_PATH, ONNX_PATH)
            output_info = _inspect_onnx(ONNX_PATH, n_classes=cfg.n_cats)
            if needs_info:
                _safe_write(_write_model_info, MODEL_INFO_PATH, cfg, output_info, labels_info)
            if not NVINFER_CONFIG_PATH.exists():
                _safe_write(
                    _write_nvinfer_config,
                    NVINFER_CONFIG_PATH,
                    ONNX_PATH.resolve(),
                    ENGINE_PATH.resolve(),
                    output_info["segmentation_output_order"],
                    cfg.n_cats,
                )
    if not MODEL_INFO_PATH.exists():
        raise FileNotFoundError(f"Model info not found: {MODEL_INFO_PATH}")
    model_info = json.loads(MODEL_INFO_PATH.read_text())
    if not NVINFER_CONFIG_PATH.exists():
        seg_order = model_info["output"]["segmentation_output_order"]
        _safe_write(
            _write_nvinfer_config,
            NVINFER_CONFIG_PATH,
            ONNX_PATH.resolve(),
            ENGINE_PATH.resolve(),
            seg_order,
            model_info["output"]["classes"],
        )

    if not ENGINE_PATH.exists():
        if not allow_engine_build:
            raise FileNotFoundError(f"Engine not found: {ENGINE_PATH}")
        trtexec = shutil.which("trtexec")
        if not trtexec:
            raise FileNotFoundError("trtexec not found on PATH; cannot build engine")
        subprocess.run(
            [
                trtexec,
                f"--onnx={ONNX_PATH}",
                f"--saveEngine={ENGINE_PATH}",
                "--fp16",
            ],
            check=True,
        )

    return ModelArtifacts(
        weights_path=WEIGHTS_PATH,
        onnx_path=ONNX_PATH,
        engine_path=ENGINE_PATH,
        labels_csv_path=LABELS_CSV_PATH,
        labels_json_path=LABELS_JSON_PATH,
        model_info_path=MODEL_INFO_PATH,
        nvinfer_config_path=NVINFER_CONFIG_PATH,
    )


def load_model_info(path: Path = MODEL_INFO_PATH) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Model info not found: {path}")
    return json.loads(path.read_text())


def load_labels(path: Path = LABELS_JSON_PATH) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Labels JSON not found: {path}")
    return json.loads(path.read_text())


if __name__ == "__main__":
    ensure_model_artifacts()
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"ONNX: {ONNX_PATH}")
    print(f"Engine: {ENGINE_PATH}")
    print(f"Labels: {LABELS_JSON_PATH}")
    print(f"Model info: {MODEL_INFO_PATH}")
