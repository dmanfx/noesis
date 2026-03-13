"""Download, export, and materialize room-layout segmentation artifacts."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import onnx
import requests
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = REPO_ROOT / "models" / "room_layout_segformer_b5_ade20k"
SNAPSHOT_DIR = MODEL_ROOT / "hf_snapshot"
ONNX_DIR = MODEL_ROOT / "onnx"
ENGINE_DIR = MODEL_ROOT / "engines"
LABELS_DIR = MODEL_ROOT / "labels"
BUILD_DIR = PIPELINE_ROOT / "build"

MODEL_ID = "nvidia/segformer-b5-finetuned-ade-640-640"
MODEL_CARD_URL = "https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640"
SEGFORMER_PAPER_URL = "https://arxiv.org/abs/2105.15203"
MMSEG_METAFILE_URL = (
    "https://raw.githubusercontent.com/open-mmlab/mmsegmentation/main/configs/segformer/metafile.yaml"
)
MMDEPLOY_MMSEG_SUPPORT_URL = (
    "https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmseg.html"
)
ADE20K_LABELS_URL = (
    "https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/objectInfo150.csv"
)

INPUT_WIDTH = 1024
INPUT_HEIGHT = 576
MIN_BATCH = 1
OPT_BATCH = 1
MAX_BATCH = 3
GIE_UNIQUE_ID = 1

ONNX_PATH = ONNX_DIR / "segformer_b5_room_layout_groups_1024x576.onnx"
ENGINE_PATH = ENGINE_DIR / "segformer_b5_room_layout_groups_1024x576_b1_b3_fp16.engine"
MODEL_INFO_PATH = MODEL_ROOT / "model_info.json"
ADE20K_LABELS_CSV_PATH = LABELS_DIR / "objectInfo150.csv"
OUTPUT_LABELS_JSON_PATH = LABELS_DIR / "output_labels.json"
OUTPUT_LABELS_TXT_PATH = LABELS_DIR / "output_labels.txt"
NVINFER_CONFIG_TEMPLATE_PATH = PIPELINE_ROOT / "config_infer_room_layout_seg.template.ini"

LAYOUT_GROUPS = [
    {
        "output_id": 1,
        "name": "wall",
        "source_class_ids": [0],
        "palette": [232, 77, 85],
    },
    {
        "output_id": 2,
        "name": "floor",
        "source_class_ids": [3],
        "palette": [94, 178, 85],
    },
    {
        "output_id": 3,
        "name": "ceiling",
        "source_class_ids": [5],
        "palette": [90, 165, 240],
    },
    {
        "output_id": 4,
        "name": "window",
        "source_class_ids": [8],
        "palette": [79, 224, 220],
    },
    {
        "output_id": 5,
        "name": "door",
        "source_class_ids": [14, 58],
        "palette": [237, 193, 74],
    },
    {
        "output_id": 6,
        "name": "stairs",
        "source_class_ids": [53, 59, 121],
        "palette": [214, 131, 45],
    },
]

OUTPUT_LABELS = [
    {
        "output_id": 0,
        "name": "other",
        "source_class_ids": [],
        "palette": [0, 0, 0],
    },
    *LAYOUT_GROUPS,
]


@dataclass
class ModelArtifacts:
    snapshot_dir: Path
    onnx_path: Path
    engine_path: Path
    labels_csv_path: Path
    labels_json_path: Path
    labels_txt_path: Path
    model_info_path: Path


class SegformerRoomLayoutWrapper(torch.nn.Module):
    """Collapse full ADE20K logits into a compact room-layout output space."""

    def __init__(self, model: torch.nn.Module, group_ids: List[List[int]]) -> None:
        super().__init__()
        self.model = model
        self._group_ids = [list(ids) for ids in group_ids]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=x).logits
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)
        grouped: List[torch.Tensor] = []
        grouped_total = torch.zeros_like(probs[:, :1, :, :])
        for ids in self._group_ids:
            part = probs[:, ids, :, :].sum(dim=1, keepdim=True)
            grouped.append(part)
            grouped_total = grouped_total + part
        other = (1.0 - grouped_total).clamp(min=0.0, max=1.0)
        return torch.cat([other] + grouped, dim=1)


def _ensure_dirs() -> None:
    for path in (SNAPSHOT_DIR, ONNX_DIR, ENGINE_DIR, LABELS_DIR, BUILD_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)


def _ensure_hf_snapshot(allow_download: bool = True) -> Path:
    config_path = SNAPSHOT_DIR / "config.json"
    has_weights = any(SNAPSHOT_DIR.glob("*.safetensors")) or any(SNAPSHOT_DIR.glob("*.bin"))
    if config_path.exists() and has_weights:
        return SNAPSHOT_DIR
    if not allow_download:
        raise FileNotFoundError(
            f"Local Hugging Face snapshot missing under {SNAPSHOT_DIR}; downloads disabled"
        )
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model.save_pretrained(SNAPSHOT_DIR)
    return SNAPSHOT_DIR


def _ensure_labels_csv(allow_download: bool = True) -> Path:
    if ADE20K_LABELS_CSV_PATH.exists():
        return ADE20K_LABELS_CSV_PATH
    if not allow_download:
        raise FileNotFoundError(f"ADE20K labels CSV missing: {ADE20K_LABELS_CSV_PATH}")
    _download_file(ADE20K_LABELS_URL, ADE20K_LABELS_CSV_PATH)
    return ADE20K_LABELS_CSV_PATH


def _load_ade20k_labels(csv_path: Path) -> Dict[int, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        labels: Dict[int, str] = {}
        for row in reader:
            idx_raw = row.get("Idx")
            name_raw = row.get("Name")
            if not idx_raw or not name_raw:
                continue
            labels[int(idx_raw) - 1] = str(name_raw)
    if not labels:
        raise RuntimeError(f"Parsed zero ADE20K labels from {csv_path}")
    return labels


def _materialize_output_labels(labels_by_id: Dict[int, str]) -> List[Dict[str, object]]:
    output_labels: List[Dict[str, object]] = []
    for entry in OUTPUT_LABELS:
        source_names = [labels_by_id[idx] for idx in entry["source_class_ids"] if idx in labels_by_id]
        output_labels.append(
            {
                "output_id": int(entry["output_id"]),
                "name": str(entry["name"]),
                "source_class_ids": list(entry["source_class_ids"]),
                "source_class_names": source_names,
                "palette": list(entry["palette"]),
            }
        )
    return output_labels


def _write_output_labels(output_labels: List[Dict[str, object]]) -> None:
    OUTPUT_LABELS_JSON_PATH.write_text(json.dumps(output_labels, indent=2), encoding="utf-8")
    label_lines = [str(entry["name"]) for entry in output_labels]
    OUTPUT_LABELS_TXT_PATH.write_text("\n".join(label_lines) + "\n", encoding="utf-8")


def _build_model_info(output_labels: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "model": {
            "name": "SegFormer-B5 room-layout groups",
            "base_model_id": MODEL_ID,
            "task": "semantic segmentation",
            "dataset": "ADE20K",
            "references": {
                "model_card": MODEL_CARD_URL,
                "paper": SEGFORMER_PAPER_URL,
                "mmseg_model_zoo": MMSEG_METAFILE_URL,
                "mmdeploy_mmseg_support": MMDEPLOY_MMSEG_SUPPORT_URL,
                "labels_csv": ADE20K_LABELS_URL,
            },
            "benchmark": {
                "ade20k_miou_single_scale": 49.62,
                "ade20k_miou_ms_flip": 50.36,
            },
            "deployment_notes": {
                "why_this_model": (
                    "SegFormer-B5 is the strongest readily exportable ADE20K checkpoint in this "
                    "workspace using the current Transformers->ONNX->TensorRT path without "
                    "bringing in unsupported Mask2Former/mmdeploy custom-ops requirements."
                ),
                "output_contract": (
                    "The exported model collapses 150 ADE20K classes into compact room-layout "
                    "groups plus an 'other' background channel."
                ),
            },
        },
        "input": {
            "min_batch": MIN_BATCH,
            "opt_batch": OPT_BATCH,
            "max_batch": MAX_BATCH,
            "channels": 3,
            "height": INPUT_HEIGHT,
            "width": INPUT_WIDTH,
        },
        "output": {
            "classes": len(output_labels),
            "height": INPUT_HEIGHT,
            "width": INPUT_WIDTH,
            "order": "NCHW",
            "segmentation_output_order": 0,
        },
        "inference": {
            "gie_unique_id": GIE_UNIQUE_ID,
        },
        "labels": output_labels,
    }


def _write_model_info(model_info: Dict[str, object]) -> None:
    MODEL_INFO_PATH.write_text(json.dumps(model_info, indent=2), encoding="utf-8")


def _valid_onnx(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
    except Exception:
        return False
    return True


def _export_onnx(snapshot_dir: Path) -> None:
    model = SegformerForSemanticSegmentation.from_pretrained(snapshot_dir).eval()
    wrapped = SegformerRoomLayoutWrapper(
        model=model,
        group_ids=[list(entry["source_class_ids"]) for entry in LAYOUT_GROUPS],
    ).eval()
    dummy = torch.randn(MIN_BATCH, 3, INPUT_HEIGHT, INPUT_WIDTH, dtype=torch.float32)
    torch.onnx.export(
        wrapped,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=["segmentation"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        external_data=False,
        dynamic_axes={"input": {0: "batch"}, "segmentation": {0: "batch"}},
    )


def _engine_is_current() -> bool:
    if not ENGINE_PATH.exists() or not ONNX_PATH.exists():
        return False
    return ENGINE_PATH.stat().st_mtime >= ONNX_PATH.stat().st_mtime


def _build_engine() -> None:
    trtexec = shutil.which("trtexec")
    if not trtexec:
        raise FileNotFoundError("trtexec not found on PATH; cannot build TensorRT engine")
    shape = f"input:{MIN_BATCH}x3x{INPUT_HEIGHT}x{INPUT_WIDTH}"
    max_shape = f"input:{MAX_BATCH}x3x{INPUT_HEIGHT}x{INPUT_WIDTH}"
    subprocess.run(
        [
            trtexec,
            f"--onnx={ONNX_PATH}",
            f"--minShapes={shape}",
            f"--optShapes={shape}",
            f"--maxShapes={max_shape}",
            f"--saveEngine={ENGINE_PATH}",
            "--fp16",
            "--skipInference",
        ],
        check=True,
    )


def ensure_model_artifacts(
    *,
    allow_download: bool = True,
    allow_export: bool = True,
    allow_engine_build: bool = True,
) -> ModelArtifacts:
    _ensure_dirs()
    snapshot_dir = _ensure_hf_snapshot(allow_download=allow_download)
    labels_csv_path = _ensure_labels_csv(allow_download=allow_download)
    labels_by_id = _load_ade20k_labels(labels_csv_path)
    output_labels = _materialize_output_labels(labels_by_id)

    if not OUTPUT_LABELS_JSON_PATH.exists() or not OUTPUT_LABELS_TXT_PATH.exists():
        _write_output_labels(output_labels)
    if not MODEL_INFO_PATH.exists():
        _write_model_info(_build_model_info(output_labels))

    if not _valid_onnx(ONNX_PATH):
        if not allow_export:
            raise FileNotFoundError(f"ONNX missing or invalid and export disabled: {ONNX_PATH}")
        _export_onnx(snapshot_dir)
    if not _engine_is_current():
        if not allow_engine_build:
            raise FileNotFoundError(f"TensorRT engine missing or stale: {ENGINE_PATH}")
        _build_engine()

    return ModelArtifacts(
        snapshot_dir=snapshot_dir,
        onnx_path=ONNX_PATH,
        engine_path=ENGINE_PATH,
        labels_csv_path=labels_csv_path,
        labels_json_path=OUTPUT_LABELS_JSON_PATH,
        labels_txt_path=OUTPUT_LABELS_TXT_PATH,
        model_info_path=MODEL_INFO_PATH,
    )


def load_model_info(path: Path = MODEL_INFO_PATH) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Model info not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_output_labels(path: Path = OUTPUT_LABELS_JSON_PATH) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Output labels not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected output labels list in {path}")
    return payload


def materialize_nvinfer_config(batch_size: int) -> Path:
    if batch_size < MIN_BATCH or batch_size > MAX_BATCH:
        raise ValueError(f"batch_size must be between {MIN_BATCH} and {MAX_BATCH}, got {batch_size}")
    template = NVINFER_CONFIG_TEMPLATE_PATH.read_text(encoding="utf-8")
    contents = template
    replacements = {
        "@ONNX_PATH@": str(ONNX_PATH.resolve()),
        "@ENGINE_PATH@": str(ENGINE_PATH.resolve()),
        "@LABELS_PATH@": str(OUTPUT_LABELS_TXT_PATH.resolve()),
        "@BATCH_SIZE@": str(int(batch_size)),
        "@NUM_CLASSES@": str(len(OUTPUT_LABELS)),
        "@GIE_UNIQUE_ID@": str(GIE_UNIQUE_ID),
    }
    for key, value in replacements.items():
        contents = contents.replace(key, value)
    output_path = BUILD_DIR / f"config_infer_room_layout_seg_b{batch_size}.ini"
    output_path.write_text(contents, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    artifacts = ensure_model_artifacts()
    print(f"Snapshot: {artifacts.snapshot_dir}")
    print(f"ONNX: {artifacts.onnx_path}")
    print(f"Engine: {artifacts.engine_path}")
    print(f"Labels JSON: {artifacts.labels_json_path}")
    print(f"Model info: {artifacts.model_info_path}")

