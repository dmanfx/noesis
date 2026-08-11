from __future__ import annotations

import re
from pathlib import Path
import os
from typing import Dict, Iterable
import logging

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_dir() -> Path:
    return Path(os.environ.get("NOESIS_BUILD_DIR", REPO_ROOT / "build")).expanduser().resolve()


def resolve_yolo26_assets(size: str) -> Dict[str, Path]:
    size_norm = str(size or "").strip().lower()
    if size_norm not in ("n", "s", "m"):
        raise ValueError(f"YOLO26 size must be one of n/s/m (got: {size})")
    return {
        "template": (REPO_ROOT / "pipelines" / "config_infer_primary_yolo26_seg.template.ini").resolve(),
        "preproc_template": (REPO_ROOT / "pipelines" / "config_preproc.ini").resolve(),
        "onnx": (REPO_ROOT / "models" / f"yolo26{size_norm}-seg_fused.onnx").resolve(),
        "engine": (REPO_ROOT / "models" / "engines" / f"yolo26{size_norm}-seg_fused_b3_fp16.engine").resolve(),
        "labels": (REPO_ROOT / "models" / "coco_labels.txt").resolve(),
        "parser": (REPO_ROOT / "pipelines" / "nvdsinfer_yolo26_seg" / "libnvdsinfer_yolo26_seg.so").resolve(),
        "output": (_build_dir() / f"config_infer_primary_yolo26_seg_{size_norm}.ini").resolve(),
        "default_output": (_build_dir() / f"config_infer_primary_yolo26_seg_{size_norm}.ini").resolve(),
    }


def materialize_yolo26_pgie_ini(
    *,
    size: str,
    output_path: Path,
    batch_size: int = 3,
    onnx_path: Path | None = None,
    engine_path: Path | None = None,
    include_model_source: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    assets = resolve_yolo26_assets(size)
    template_path = assets["template"]
    if not template_path.exists():
        raise FileNotFoundError(f"YOLO26 PGIE template missing: {template_path}")

    text = template_path.read_text(encoding="utf-8")
    if include_model_source:
        text = text.replace("@ONNX_PATH@", str((onnx_path or assets["onnx"]).resolve()))
    else:
        text = re.sub(r"(?m)^\s*onnx-file\s*=.*(?:\n|$)", "", text)
    text = text.replace("@ENGINE_PATH@", str((engine_path or assets["engine"]).resolve()))
    text = text.replace("@LABELS_PATH@", str(assets["labels"]))
    text = text.replace("@CUSTOM_LIB@", str(assets["parser"]))
    text = re.sub(r"(?m)^batch-size=\d+\s*$", f"batch-size={int(batch_size)}", text, count=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    if logger is not None:
        logger.info("YOLO26 PGIE config materialized: %s", output_path)
    return output_path


def materialize_yolo26_preproc_config(
    *,
    output_path: Path,
    batch_size: int,
    src_ids: Iterable[int | str] = (0,),
    logger: logging.Logger | None = None,
) -> Path:
    template_path = resolve_yolo26_assets("s")["preproc_template"]
    if not template_path.exists():
        raise FileNotFoundError(f"YOLO26 preprocess template missing: {template_path}")

    src_ids_text = ";".join(str(x) for x in src_ids)
    text = template_path.read_text(encoding="utf-8")
    text = re.sub(
        r"(?m)^network-input-shape=\d+;3;640;640\s*$",
        f"network-input-shape={int(batch_size)};3;640;640",
        text,
        count=1,
    )
    text, tensor_name_count = re.subn(
        r"(?m)^tensor-name=.*$",
        "tensor-name=images",
        text,
        count=1,
    )
    if tensor_name_count != 1:
        raise ValueError(f"YOLO26 preprocess template is missing tensor-name: {template_path}")
    text = re.sub(r"(?m)^src-ids=.*$", f"src-ids={src_ids_text}", text, count=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    if logger is not None:
        logger.info("YOLO26 preprocess config materialized: %s", output_path)
    return output_path
