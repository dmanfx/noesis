from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable

from noesis.yolo26_assets import (
    materialize_yolo26_pgie_ini as _materialize_yolo26_pgie_ini,
    materialize_yolo26_preproc_config as _materialize_yolo26_preproc_config,
    resolve_yolo26_assets as _resolve_yolo26_assets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_yolo26_seg_assets(size: str) -> Dict[str, Path]:
    return dict(_resolve_yolo26_assets(size))


def _default_preproc_output_path(size: str, batch_size: int) -> Path:
    size_norm = str(size or "").strip().lower()
    build_dir = Path(os.environ.get("NOESIS_BUILD_DIR", REPO_ROOT / "build")).expanduser().resolve()
    return (build_dir / f"config_preproc_yolo26_seg_{size_norm}_b{int(batch_size)}.ini").resolve()


def materialize_yolo26_seg_configs(
    *,
    size: str,
    batch_size: int,
    src_ids: Iterable[int | str],
    logger: logging.Logger | None = None,
    onnx_path: Path | None = None,
    engine_path: Path | None = None,
    pgie_output_path: Path | None = None,
    preprocess_output_path: Path | None = None,
) -> Dict[str, Path]:
    if int(batch_size) <= 0:
        raise ValueError(f"YOLO26 batch_size must be positive (got: {batch_size})")

    src_ids_tuple = tuple(src_ids)
    if not src_ids_tuple:
        raise ValueError("YOLO26 src_ids must not be empty")

    assets = resolve_yolo26_seg_assets(size)
    resolved_onnx = (onnx_path or assets["onnx"]).resolve()
    resolved_engine = (engine_path or assets["engine"]).resolve()
    resolved_pgie_output = (pgie_output_path or assets["default_output"]).resolve()
    resolved_preproc_output = (preprocess_output_path or _default_preproc_output_path(size, batch_size)).resolve()

    pgie_config_path = _materialize_yolo26_pgie_ini(
        size=size,
        output_path=resolved_pgie_output,
        batch_size=int(batch_size),
        onnx_path=resolved_onnx,
        engine_path=resolved_engine,
        logger=logger,
    ).resolve()
    preprocess_config_path = _materialize_yolo26_preproc_config(
        output_path=resolved_preproc_output,
        batch_size=int(batch_size),
        src_ids=src_ids_tuple,
        logger=logger,
    ).resolve()

    return {
        **assets,
        "onnx": resolved_onnx,
        "engine": resolved_engine,
        "pgie_config": pgie_config_path,
        "preprocess_config": preprocess_config_path,
    }
