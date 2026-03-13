"""Asset provisioning helpers for the seg+depth DS8 prototype."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import onnx
import requests

from noesis.yolo26_seg_materialization import materialize_yolo26_seg_configs

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger(__name__)

BUILD_DIR = Path(__file__).resolve().parent / "build"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

ONNX_DIR = REPO_ROOT / "models" / "onnx"
ENGINE_DIR = REPO_ROOT / "models" / "engines"

YOLO_FUSED_ONNX = REPO_ROOT / "models" / "yolo26s-seg_fused.onnx"
YOLO_BATCH1_ONNX = ONNX_DIR / "yolo26s-seg_fused_b1.onnx"
YOLO_ENGINE_PATH = ENGINE_DIR / "yolo26s-seg_fused_b1_fp16.engine"
YOLO_LABELS_PATH = REPO_ROOT / "models" / "coco_labels.txt"
YOLO_PARSER_LIB = REPO_ROOT / "pipelines" / "nvdsinfer_yolo26_seg" / "libnvdsinfer_yolo26_seg.so"
YOLO_PARSER_DIR = REPO_ROOT / "pipelines" / "nvdsinfer_yolo26_seg"

DA2_TEMPLATE_PATH = Path(__file__).resolve().parent / "config_infer_depth_anything_v2.template.ini"
DA2_REPO_DIR = REPO_ROOT / "models" / "depth_anything_v2" / "Depth-Anything-V2" / "metric_depth"
DA2_CHECKPOINT_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/"
    "resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true"
)
DA2_CHECKPOINT_PATH = (
    REPO_ROOT
    / "models"
    / "depth_anything_v2"
    / "checkpoints"
    / "depth_anything_v2_metric_hypersim_vits.pth"
)
DEFAULT_DEPTH_INPUT_SIZE = (924, 518)  # width, height
DA2_ONNX_PATH = ONNX_DIR / "depth_anything_v2_metric_hypersim_vits_518x924_b1.onnx"
DA2_ENGINE_PATH = ENGINE_DIR / "depth_anything_v2_metric_hypersim_vits_518x924_b1_fp16.engine"
DA2_MODEL_NAME = "depth-anything-v2-metric-hypersim-vits"

DA3_REPO_DIR = REPO_ROOT / "models" / "depth_anything_v3" / "Depth-Anything-3"


@dataclass(frozen=True)
class SegAssets:
    onnx: Path
    engine: Path
    parser: Path
    labels: Path
    config_path: Path
    preprocess_config_path: Path


@dataclass(frozen=True)
class DepthAssets:
    onnx: Path
    engine: Path
    config_path: Path
    model_name: str
    unit: str
    is_metric: bool
    fallback_reason: Optional[str] = None


@dataclass(frozen=True)
class PipelineAssets:
    seg: SegAssets
    depth: Optional[DepthAssets]


def _ensure_nonempty(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.is_file() and path.stat().st_size <= 0:
        raise RuntimeError(f"{label} is empty: {path}")


def _run(cmd: list[str]) -> None:
    LOGGER.info("Running: %s", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _replace_path(text: str, marker: str, value: Path) -> str:
    return text.replace(marker, str(value.resolve()))


def _write_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _trtexec() -> str:
    binary = shutil.which("trtexec")
    if not binary:
        raise FileNotFoundError("trtexec not found on PATH")
    return binary


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)


def _ensure_yolo_parser_lib() -> None:
    if YOLO_PARSER_LIB.exists() and YOLO_PARSER_LIB.stat().st_size > 0:
        return
    _run(["make", "-C", str(YOLO_PARSER_DIR)])
    _ensure_nonempty(YOLO_PARSER_LIB, label="YOLO26 fused parser library")


def _materialize_yolo_batch1_onnx() -> None:
    if YOLO_BATCH1_ONNX.exists() and YOLO_BATCH1_ONNX.stat().st_size > 0:
        return
    _ensure_nonempty(YOLO_FUSED_ONNX, label="YOLO26 fused ONNX")
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(YOLO_FUSED_ONNX))
    for value in list(model.graph.input) + list(model.graph.output):
        dims = value.type.tensor_type.shape.dim
        if not dims:
            continue
        dims[0].dim_value = 1
        dims[0].dim_param = ""
    onnx.save(model, str(YOLO_BATCH1_ONNX))
    _ensure_nonempty(YOLO_BATCH1_ONNX, label="YOLO26 batch-1 fused ONNX")


def _build_yolo_engine() -> None:
    if YOLO_ENGINE_PATH.exists() and YOLO_ENGINE_PATH.stat().st_size > 0:
        return
    if YOLO_ENGINE_PATH.exists():
        YOLO_ENGINE_PATH.unlink()
    _materialize_yolo_batch1_onnx()
    YOLO_ENGINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _trtexec(),
            f"--onnx={YOLO_BATCH1_ONNX}",
            "--fp16",
            f"--saveEngine={YOLO_ENGINE_PATH}",
        ]
    )
    _ensure_nonempty(YOLO_ENGINE_PATH, label="YOLO26 fused batch-1 engine")


def _materialize_yolo_configs() -> Dict[str, Path]:
    _ensure_nonempty(YOLO_LABELS_PATH, label="YOLO labels")
    _ensure_yolo_parser_lib()
    _build_yolo_engine()
    return materialize_yolo26_seg_configs(
        size="s",
        batch_size=1,
        src_ids=(0,),
        logger=LOGGER,
        onnx_path=YOLO_BATCH1_ONNX,
        engine_path=YOLO_ENGINE_PATH,
        pgie_output_path=BUILD_DIR / "config_infer_yolo26_seg_s_b1.ini",
        preprocess_output_path=BUILD_DIR / "config_preproc_yolo26_b1.ini",
    )


def _depth_size_suffix(size: Tuple[int, int]) -> str:
    width, height = int(size[0]), int(size[1])
    return f"{height}x{width}"


def _da2_onnx_path(size: Tuple[int, int]) -> Path:
    if tuple(size) == DEFAULT_DEPTH_INPUT_SIZE:
        return DA2_ONNX_PATH
    return ONNX_DIR / f"depth_anything_v2_metric_hypersim_vits_{_depth_size_suffix(size)}_b1.onnx"


def _da2_engine_path(size: Tuple[int, int]) -> Path:
    if tuple(size) == DEFAULT_DEPTH_INPUT_SIZE:
        return DA2_ENGINE_PATH
    return ENGINE_DIR / f"depth_anything_v2_metric_hypersim_vits_{_depth_size_suffix(size)}_b1_fp16.engine"


def _da2_config_path(size: Tuple[int, int], *, depth_every_n_frames: int) -> Path:
    interval = max(0, int(depth_every_n_frames) - 1)
    if tuple(size) == DEFAULT_DEPTH_INPUT_SIZE and interval == 0:
        return BUILD_DIR / "config_infer_depth_anything_v2_metric_vits_b1.ini"
    return BUILD_DIR / f"config_infer_depth_anything_v2_metric_vits_{_depth_size_suffix(size)}_b1_i{interval}.ini"


def _clone_da3_repo() -> None:
    if DA3_REPO_DIR.exists():
        return
    DA3_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/ByteDance-Seed/Depth-Anything-3",
            str(DA3_REPO_DIR),
        ]
    )


def _da3_missing_python_deps() -> list[str]:
    deps = {
        "addict": "addict",
        "e3nn": "e3nn",
        "einops": "einops",
        "evo": "evo",
        "fastapi": "fastapi",
        "huggingface_hub": "huggingface_hub",
        "moviepy": "moviepy",
        "omegaconf": "omegaconf",
        "open3d": "open3d",
        "pillow_heif": "pillow_heif",
        "plyfile": "plyfile",
        "pycolmap": "pycolmap",
        "requests": "requests",
        "safetensors": "safetensors",
        "torchvision": "torchvision",
        "trimesh": "trimesh",
        "typer": "typer",
        "uvicorn": "uvicorn",
        "xformers": "xformers",
    }
    missing: list[str] = []
    for label, module_name in deps.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(label)
    return sorted(missing)


def _try_da3_candidate() -> tuple[bool, Optional[str]]:
    try:
        _clone_da3_repo()
    except Exception as exc:
        return False, f"DA3 clone failed: {exc}"

    missing = _da3_missing_python_deps()
    if missing:
        joined = ", ".join(missing)
        return False, f"DA3 environment incomplete: missing python deps [{joined}]"

    src_dir = DA3_REPO_DIR / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        importlib.import_module("depth_anything_3.api")
    except Exception as exc:
        return False, f"DA3 import failed: {exc}"
    finally:
        if str(src_dir) in sys.path:
            sys.path.remove(str(src_dir))
    return False, "DA3 import succeeded but automated ONNX export is not wired on this host; falling back to DAv2"


def _download_da2_checkpoint() -> None:
    if DA2_CHECKPOINT_PATH.exists() and DA2_CHECKPOINT_PATH.stat().st_size > 0:
        return
    _download(DA2_CHECKPOINT_URL, DA2_CHECKPOINT_PATH)
    _ensure_nonempty(DA2_CHECKPOINT_PATH, label="DAv2 metric checkpoint")


def _export_da2_onnx(depth_input_size: Tuple[int, int]) -> Path:
    width, height = int(depth_input_size[0]), int(depth_input_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid depth input size: {depth_input_size}")
    onnx_path = _da2_onnx_path((width, height))
    if onnx_path.exists() and onnx_path.stat().st_size > 0:
        return onnx_path
    _download_da2_checkpoint()
    if not DA2_REPO_DIR.exists():
        raise FileNotFoundError(f"Depth Anything metric repo missing: {DA2_REPO_DIR}")

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

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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
    model.eval()
    wrapper = ExportWrapper(model).eval()
    dummy = torch.randint(0, 256, (1, 3, height, width), dtype=torch.uint8).float()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["depth"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
            external_data=False,
        )
    _ensure_nonempty(onnx_path, label="DAv2 ONNX")
    return onnx_path


def _build_da2_engine(depth_input_size: Tuple[int, int]) -> Path:
    engine_path = _da2_engine_path(depth_input_size)
    if engine_path.exists() and engine_path.stat().st_size > 0:
        return engine_path
    if engine_path.exists():
        engine_path.unlink()
    onnx_path = _export_da2_onnx(depth_input_size)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _trtexec(),
            f"--onnx={onnx_path}",
            "--fp16",
            f"--saveEngine={engine_path}",
        ]
    )
    _ensure_nonempty(engine_path, label="DAv2 engine")
    return engine_path


def _materialize_da2_config(
    depth_input_size: Tuple[int, int],
    *,
    depth_every_n_frames: int,
) -> Tuple[Path, Path, Path]:
    _ensure_nonempty(DA2_TEMPLATE_PATH, label="DAv2 template")
    width, height = int(depth_input_size[0]), int(depth_input_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid depth input size: {depth_input_size}")
    onnx_path = _da2_onnx_path((width, height))
    engine_path = _build_da2_engine((width, height))
    output_path = _da2_config_path((width, height), depth_every_n_frames=max(1, int(depth_every_n_frames)))
    text = DA2_TEMPLATE_PATH.read_text(encoding="utf-8")
    text = _replace_path(text, "@ONNX_PATH@", onnx_path)
    text = _replace_path(text, "@ENGINE_PATH@", engine_path)
    text = text.replace("interval=0", f"interval={max(0, int(depth_every_n_frames) - 1)}", 1)
    text = text.replace("infer-dims=3;518;924", f"infer-dims=3;{height};{width}", 1)
    _write_if_changed(output_path, text)
    return output_path, onnx_path, engine_path


def _prepare_depth_assets(
    depth_preference: str,
    *,
    depth_input_size: Tuple[int, int],
    depth_every_n_frames: int,
) -> DepthAssets:
    pref = str(depth_preference or "auto").strip().lower()
    if pref not in {"auto", "da2", "da3"}:
        raise ValueError(f"Unsupported depth preference: {depth_preference}")

    fallback_reason = None
    if pref in {"auto", "da3"}:
        ok, reason = _try_da3_candidate()
        if ok:
            raise RuntimeError("DA3 candidate unexpectedly returned success without assets")
        fallback_reason = reason
        if pref == "da3":
            raise RuntimeError(reason or "DA3 preparation failed")

    config_path, onnx_path, engine_path = _materialize_da2_config(
        depth_input_size,
        depth_every_n_frames=depth_every_n_frames,
    )
    return DepthAssets(
        onnx=onnx_path,
        engine=engine_path,
        config_path=config_path,
        model_name=DA2_MODEL_NAME,
        unit="m",
        is_metric=True,
        fallback_reason=fallback_reason,
    )


def _sysconfig_extension_suffix() -> str:
    import sysconfig

    return str(sysconfig.get_config_var("EXT_SUFFIX") or ".so")


def ensure_native_object_depth_extension() -> None:
    ext_path = REPO_ROOT / f"noesis_depth_meta_ext{_sysconfig_extension_suffix()}"
    source_path = REPO_ROOT / "native" / "noesis_depth_meta_ext.cpp"
    if ext_path.exists() and ext_path.stat().st_size > 0 and ext_path.stat().st_mtime >= source_path.stat().st_mtime:
        return
    _run([str(REPO_ROOT / "scripts" / "build_noesis_depth_meta_ext.sh")])
    _ensure_nonempty(ext_path, label="Object depth extension")


def ensure_pipeline_assets(
    depth_preference: str = "auto",
    *,
    enable_depth: bool = True,
    depth_input_size: Tuple[int, int] = DEFAULT_DEPTH_INPUT_SIZE,
    depth_every_n_frames: int = 1,
) -> PipelineAssets:
    ensure_native_object_depth_extension()
    seg_configs = _materialize_yolo_configs()
    seg_config = seg_configs["pgie_config"]
    seg_preproc = seg_configs["preprocess_config"]
    depth_assets: Optional[DepthAssets] = None
    if enable_depth:
        depth_assets = _prepare_depth_assets(
            depth_preference,
            depth_input_size=(int(depth_input_size[0]), int(depth_input_size[1])),
            depth_every_n_frames=max(1, int(depth_every_n_frames)),
        )
        LOGGER.info(
            "Prepared pipeline assets: seg_engine=%s depth_engine=%s depth_model=%s metric=%s fallback=%s depth_input=%sx%s depth_every_n_frames=%d",
            YOLO_ENGINE_PATH,
            depth_assets.engine,
            depth_assets.model_name,
            depth_assets.is_metric,
            depth_assets.fallback_reason,
            int(depth_input_size[0]),
            int(depth_input_size[1]),
            max(1, int(depth_every_n_frames)),
        )
    else:
        LOGGER.info("Prepared seg-only pipeline assets: seg_engine=%s", YOLO_ENGINE_PATH)
    return PipelineAssets(
        seg=SegAssets(
            onnx=YOLO_BATCH1_ONNX,
            engine=YOLO_ENGINE_PATH,
            parser=YOLO_PARSER_LIB,
            labels=YOLO_LABELS_PATH,
            config_path=seg_config,
            preprocess_config_path=seg_preproc,
        ),
        depth=depth_assets,
    )
