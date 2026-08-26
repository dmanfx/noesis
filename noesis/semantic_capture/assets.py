"""Resolve immutable TensorRT assets and materialize an engine-only nvinfer config."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_SIZES = ("s", "l")
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_TEMPLATE = PACKAGE_ROOT / "config_infer_yolo26_sem_ade20k.template.ini"


@dataclass(frozen=True)
class RuntimeAssets:
    size: str
    engine: Path
    labels: Path
    parser: Path


def _asset_root() -> Path:
    configured = str(os.environ.get("NOESIS_SEMANTIC_ASSET_ROOT", "") or "").strip()
    return (
        Path(configured).expanduser().resolve()
        if configured
        else (REPO_ROOT / "DS9" / "artifacts" / "yolo26_sem_ade20k").resolve()
    )


def _build_root() -> Path:
    configured = str(os.environ.get("NOESIS_BUILD_DIR", "") or "").strip()
    return (
        Path(configured).expanduser().resolve()
        if configured
        else (REPO_ROOT / "build").resolve()
    )


def resolve_assets(size: str) -> RuntimeAssets:
    normalized = str(size or "").strip().lower()
    if normalized not in SUPPORTED_SIZES:
        raise ValueError(f"Model size must be one of {SUPPORTED_SIZES}, got {size!r}")
    root = _asset_root()
    assets = RuntimeAssets(
        size=normalized,
        engine=root / "engines" / f"yolo26{normalized}-sem-ade20k_b3_fp16.engine",
        labels=root / "labels" / "ade20k_labels.txt",
        parser=root / "custom" / "libnvdsinfer_yolo26_sem_ade20k.so",
    )
    for label, path in (
        ("TensorRT engine", assets.engine),
        ("ADE20K labels", assets.labels),
        ("DeepStream semantic parser", assets.parser),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"YOLO26{normalized} {label} missing or empty: {path}")
    return assets


def materialize_nvinfer_config(size: str) -> tuple[Path, RuntimeAssets]:
    assets = resolve_assets(size)
    text = CONFIG_TEMPLATE.read_text(encoding="utf-8")
    for token, value in {
        "@ENGINE_PATH@": str(assets.engine),
        "@LABELS_PATH@": str(assets.labels),
        "@CUSTOM_LIB@": str(assets.parser),
    }.items():
        if token not in text:
            raise RuntimeError(f"Semantic nvinfer template is missing token {token}")
        text = text.replace(token, value)
    config_dir = _build_root() / "semantic-seg" / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    output = config_dir / f"config_infer_yolo26{assets.size}_sem_ade20k_b3.ini"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)
    return output, assets
