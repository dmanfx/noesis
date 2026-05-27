from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build"
ONNX_DIR = REPO_ROOT / "models" / "onnx"
ENGINE_DIR = REPO_ROOT / "models" / "engines"
DA2_REPO_DIR = REPO_ROOT / "models" / "depth_anything_v2" / "Depth-Anything-V2" / "metric_depth"
DA2_CHECKPOINT_PATH = REPO_ROOT / "models" / "depth_anything_v2" / "checkpoints" / "depth_anything_v2_metric_hypersim_vits.pth"

DEFAULT_GIE_ID = 5
DEFAULT_BATCH_SIZE = 3
DEFAULT_INTERVAL = 1
DEFAULT_INPUT_SIZE = (518, 294)  # width, height
DEFAULT_MODEL_NAME = "depth-anything-v2-metric-hypersim-vits"

_B1_ONNX_TEMPLATE = ONNX_DIR / "depth_anything_v2_metric_hypersim_vits_294x518_b1.onnx"
_B3_ONNX_TEMPLATE = ONNX_DIR / "depth_anything_v2_metric_hypersim_vits_294x518_b3.onnx"
_B3_ENGINE_TEMPLATE = ENGINE_DIR / "depth_anything_v2_metric_hypersim_vits_294x518_b3_fp16.engine"
_CONFIG_TEMPLATE = REPO_ROOT / "pipelines" / "config_infer_secondary_depth_tracking_da2.template.ini"


@dataclass(frozen=True)
class DepthTrackingAssets:
    config_path: Path
    onnx_path: Path
    engine_path: Path
    model_name: str
    unit: str
    is_metric: bool
    gie_id: int
    batch_size: int
    interval: int
    input_size: Tuple[int, int]


def _trtexec() -> str:
    binary = shutil.which("trtexec")
    if not binary:
        raise FileNotFoundError("trtexec not found on PATH")
    return binary


def _run(cmd: list[str], logger: logging.Logger | None = None) -> None:
    if logger is not None:
        logger.info("Running: %s", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _ensure_nonempty(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"{label} is empty: {path}")


def _replace_token(text: str, token: str, value: object) -> str:
    return text.replace(token, str(value))


def _export_batch_onnx(
    *,
    output_onnx: Path,
    batch_size: int,
    width: int,
    height: int,
    force_rebuild: bool = False,
) -> Path:
    if batch_size == 1 and _B1_ONNX_TEMPLATE.exists() and _B1_ONNX_TEMPLATE.stat().st_size > 0:
        return _B1_ONNX_TEMPLATE
    if not force_rebuild and output_onnx.exists() and output_onnx.stat().st_size > 0:
        return output_onnx

    _ensure_nonempty(DA2_CHECKPOINT_PATH, label="Depth tracking checkpoint")
    if not DA2_REPO_DIR.exists():
        raise FileNotFoundError(f"Depth tracking repo missing: {DA2_REPO_DIR}")

    import sys

    if str(DA2_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(DA2_REPO_DIR))
    try:
        import torch
        import torch.nn as nn
        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
        from depth_anything_v2.dinov2_layers import attention as da2_attention  # type: ignore
        from depth_anything_v2.dinov2_layers import block as da2_block  # type: ignore
    finally:
        if str(DA2_REPO_DIR) in sys.path:
            sys.path.remove(str(DA2_REPO_DIR))

    da2_attention.XFORMERS_AVAILABLE = False
    da2_block.XFORMERS_AVAILABLE = False

    class ExportWrapper(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
            x = x.float().div(255.0)
            x = (x - self.mean) / self.std
            return self.inner(x)

    model = DepthAnythingV2(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        max_depth=20.0,
    )
    state = torch.load(DA2_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    wrapper = ExportWrapper(model.eval()).eval()
    dummy = torch.randint(0, 256, (int(batch_size), 3, int(height), int(width)), dtype=torch.uint8).float()

    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            output_onnx,
            input_names=["input"],
            output_names=["depth"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
            external_data=False,
        )
    _ensure_nonempty(output_onnx, label="Depth tracking batch ONNX")
    return output_onnx


def _build_engine(
    *,
    onnx_path: Path,
    engine_path: Path,
    logger: logging.Logger | None = None,
) -> Path:
    if engine_path.exists() and engine_path.stat().st_size > 0:
        return engine_path
    if engine_path.exists():
        engine_path.unlink()
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _trtexec(),
            f"--onnx={onnx_path}",
            "--fp16",
            f"--saveEngine={engine_path}",
        ],
        logger=logger,
    )
    _ensure_nonempty(engine_path, label="Depth tracking TensorRT engine")
    return engine_path


def _config_output_path(*, width: int, height: int, batch_size: int, interval: int) -> Path:
    return (BUILD_DIR / f"config_infer_depth_tracking_da2_vits_{height}x{width}_b{batch_size}_i{interval}.ini").resolve()


def _batch_onnx_path(*, width: int, height: int, batch_size: int) -> Path:
    return (ONNX_DIR / f"depth_anything_v2_metric_hypersim_vits_{height}x{width}_b{batch_size}.onnx").resolve()


def _engine_path(*, width: int, height: int, batch_size: int) -> Path:
    return (ENGINE_DIR / f"depth_anything_v2_metric_hypersim_vits_{height}x{width}_b{batch_size}_fp16.engine").resolve()


def materialize_depth_tracking_assets(
    *,
    logger: logging.Logger | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    interval: int = DEFAULT_INTERVAL,
    input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
    gie_id: int = DEFAULT_GIE_ID,
) -> DepthTrackingAssets:
    width = int(input_size[0])
    height = int(input_size[1])
    batch_size = int(batch_size)
    interval = int(interval)
    gie_id = int(gie_id)

    if batch_size <= 0:
        raise ValueError(f"Depth tracking batch_size must be positive (got {batch_size})")
    if interval < 0:
        raise ValueError(f"Depth tracking interval must be non-negative (got {interval})")
    if width <= 0 or height <= 0:
        raise ValueError(f"Depth tracking input_size must be positive (got {input_size})")
    if width % 14 or height % 14:
        raise ValueError(
            f"Depth tracking input_size must be multiples of 14 for DAv2 (got {width}x{height})"
        )

    if (width, height) != DEFAULT_INPUT_SIZE:
        raise ValueError(
            "Only the approved DS8 depth-tracking operating point is currently supported: "
            f"{DEFAULT_INPUT_SIZE[0]}x{DEFAULT_INPUT_SIZE[1]}"
        )

    batch_onnx = _B3_ONNX_TEMPLATE if batch_size == 3 else _batch_onnx_path(width=width, height=height, batch_size=batch_size)
    engine_path = _B3_ENGINE_TEMPLATE if batch_size == 3 else _engine_path(width=width, height=height, batch_size=batch_size)
    config_path = _config_output_path(width=width, height=height, batch_size=batch_size, interval=interval)
    engine_ready = engine_path.exists() and engine_path.stat().st_size > 0

    batch_onnx = _export_batch_onnx(
        output_onnx=batch_onnx,
        batch_size=batch_size,
        width=width,
        height=height,
        force_rebuild=not engine_ready,
    )
    engine_path = _build_engine(onnx_path=batch_onnx, engine_path=engine_path, logger=logger)

    _ensure_nonempty(_CONFIG_TEMPLATE, label="Depth tracking config template")
    text = _CONFIG_TEMPLATE.read_text(encoding="utf-8")
    text = _replace_token(text, "@ONNX_PATH@", batch_onnx.resolve())
    text = _replace_token(text, "@ENGINE_PATH@", engine_path.resolve())
    text = _replace_token(text, "@BATCH_SIZE@", batch_size)
    text = _replace_token(text, "@INTERVAL@", interval)
    text = _replace_token(text, "@GIE_ID@", gie_id)
    text = _replace_token(text, "@HEIGHT@", height)
    text = _replace_token(text, "@WIDTH@", width)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists() or config_path.read_text(encoding="utf-8") != text:
        config_path.write_text(text, encoding="utf-8")

    return DepthTrackingAssets(
        config_path=config_path,
        onnx_path=batch_onnx,
        engine_path=engine_path,
        model_name=DEFAULT_MODEL_NAME,
        unit="m",
        is_metric=True,
        gie_id=gie_id,
        batch_size=batch_size,
        interval=interval,
        input_size=(width, height),
    )


def ensure_native_object_depth_extension(logger: logging.Logger | None = None) -> None:
    import sysconfig

    ext_suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
    ext_path = REPO_ROOT / f"noesis_depth_meta_ext{ext_suffix}"
    source_path = REPO_ROOT / "native" / "noesis_depth_meta_ext.cpp"
    if ext_path.exists() and ext_path.stat().st_size > 0 and ext_path.stat().st_mtime >= source_path.stat().st_mtime:
        return
    _run([str(REPO_ROOT / "scripts" / "build_noesis_depth_meta_ext.sh")], logger=logger)
    _ensure_nonempty(ext_path, label="Object depth extension")


def ensure_native_depth_tracking_tensor_extension(logger: logging.Logger | None = None) -> None:
    import sysconfig

    ext_suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
    ext_path = REPO_ROOT / f"noesis_depth_tracking_tensor_ext{ext_suffix}"
    source_path = REPO_ROOT / "native" / "noesis_depth_tracking_tensor_ext.cpp"
    if ext_path.exists() and ext_path.stat().st_size > 0 and ext_path.stat().st_mtime >= source_path.stat().st_mtime:
        return
    _run([str(REPO_ROOT / "scripts" / "build_noesis_depth_tracking_tensor_ext.sh")], logger=logger)
    _ensure_nonempty(ext_path, label="Depth tracking tensor extension")


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_GIE_ID",
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_INTERVAL",
    "DEFAULT_MODEL_NAME",
    "DepthTrackingAssets",
    "ensure_native_depth_tracking_tensor_extension",
    "ensure_native_object_depth_extension",
    "materialize_depth_tracking_assets",
]
