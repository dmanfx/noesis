#!/usr/bin/env python3
"""Standalone MapAnything monocular depth ONNX exporter."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import logging
import subprocess
import shutil
import sys
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download
from onnx import numpy_helper
from packaging.version import InvalidVersion, Version

try:  # Optional dependency used for graph simplification
    import onnxsim  # type: ignore
except Exception:  # pragma: no cover - best effort dependency
    onnxsim = None  # type: ignore

LOGGER = logging.getLogger("export_ma_onnx")
MIN_UNICEPTION_VERSION = Version("0.1.7")


class ModelCompatibilityError(RuntimeError):
    """Raised before weights load when model source and checkpoint APIs differ."""


def _safe_output_name(value: str) -> str:
    raw = str(value or "").strip()
    candidate = Path(raw)
    if (
        not raw
        or candidate.name != raw
        or candidate.suffix.lower() != ".onnx"
        or candidate.is_absolute()
        or ".." in candidate.parts
    ):
        raise ValueError(
            "--output-name must be one relative .onnx filename without directories"
        )
    return raw


def _safe_report_name(value: str) -> str:
    raw = str(value or "").strip()
    candidate = Path(raw)
    if (
        not raw
        or candidate.name != raw
        or candidate.suffix.lower() != ".txt"
        or candidate.is_absolute()
        or ".." in candidate.parts
    ):
        raise ValueError(
            "--report-name must be one relative .txt filename without directories"
        )
    return raw


class MapAnythingDepthWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        normalization_type: str,
        mean_tensor: torch.Tensor,
        std_tensor: torch.Tensor,
        use_fused_input: bool,
        *,
        output_height: int,
        output_width: int,
        return_conf_mask: bool = False,
        include_intrinsics: bool = False,
        fx_default: float = 1000.0,
        fy_default: float = 1000.0,
    ) -> None:
        super().__init__()
        self.model = base_model
        self.norm_type = normalization_type
        self.use_fused_input = use_fused_input
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        self.return_conf_mask = bool(return_conf_mask)
        self.include_intrinsics = bool(include_intrinsics)
        self.fx_default = float(fx_default)
        self.fy_default = float(fy_default)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)

    def _build_intrinsics(self, images: torch.Tensor) -> torch.Tensor:
        _, _, height, width = images.shape
        device = images.device
        dtype = images.dtype
        intrinsics = torch.tensor(
            [
                self.fx_default,
                0.0,
                float(width) / 2.0,
                0.0,
                self.fy_default,
                float(height) / 2.0,
                0.0,
                0.0,
                1.0,
            ],
            device=device,
            dtype=dtype,
        ).view(1, 3, 3)
        return intrinsics.expand(images.size(0), -1, -1)

    def forward(self, fused_or_images: torch.Tensor):  # type: ignore[override]
        """ONNX-exportable forward wrapper around `MapAnything.forward()`.

        IMPORTANT: Do not call `MapAnything.infer()` here. The `infer()` path performs
        postprocessing with numpy conversions (e.g. edge masks) which breaks ONNX export
        and can yield incorrect/traced graphs. We call `forward()` and derive:
        - `depth`: `pts3d_cam[..., 2]` (Z-depth in camera frame)
        - `conf`: model confidence (if available)
        - `mask`: non-ambiguous mask (if available)
        using torch ops only.
        """
        if self.use_fused_input:
            fused = fused_or_images
            if fused.dim() != 4:
                raise ValueError(f"Expected fused tensor with shape (N,12,H,W), got {tuple(fused.shape)}")
            if fused.size(1) != 12:
                raise ValueError(f"Expected fused tensor to have 12 channels, got {fused.size(1)}")
            images = fused[:, :3, :, :]
            intrinsics_mat = None
            if self.include_intrinsics:
                intr_map = fused[:, 3:, :, :]
                # Average spatial dimensions to recover the flattened 3x3 intrinsics.
                intr_flat = intr_map.mean(dim=(-2, -1))
                intrinsics_mat = intr_flat.view(-1, 3, 3)
        else:
            images = fused_or_images
            if images.dim() != 4:
                raise ValueError(f"Expected images with shape (N,3,H,W), got {tuple(images.shape)}")
            if images.size(1) != 3:
                raise ValueError(f"Expected images to have 3 channels, got {images.size(1)}")
            intrinsics_mat = self._build_intrinsics(images) if self.include_intrinsics else None

        normalized = (images - self.mean) / self.std

        view = {
            "img": normalized,
            "data_norm_type": [self.norm_type],
        }

        # `MapAnything.forward()` accepts ray directions in camera frame, not raw intrinsics.
        if intrinsics_mat is not None:
            from mapanything.utils.geometry import get_rays_in_camera_frame  # type: ignore

            _, ray_dirs = get_rays_in_camera_frame(
                intrinsics=intrinsics_mat,
                height=int(images.shape[-2]),
                width=int(images.shape[-1]),
                normalize_to_unit_sphere=True,
            )
            view["ray_directions_cam"] = ray_dirs

        preds = self.model.forward([view], memory_efficient_inference=False)
        if not isinstance(preds, (list, tuple)) or not preds:
            raise RuntimeError("MapAnything.forward returned no outputs")
        out = preds[0]
        if not isinstance(out, dict):
            raise TypeError(f"Expected MapAnything.forward output dict, got {type(out)!r}")

        if "pts3d_cam" in out:
            depth_z = out["pts3d_cam"][..., 2]  # (N, H, W)
        elif "depth_along_ray" in out:
            depth_z = out["depth_along_ray"].squeeze(-1)
        else:
            raise KeyError("MapAnything.forward output missing pts3d_cam/depth_along_ray")

        output_shape = (
            images.shape[0],
            1,
            self.output_height,
            self.output_width,
        )
        depth_out = depth_z.to(torch.float32).reshape(output_shape)
        if not self.return_conf_mask:
            return depth_out

        conf = out.get("conf")
        if conf is None:
            conf_out = torch.zeros_like(depth_out)
        else:
            conf_out = conf.to(torch.float32).reshape(output_shape)

        non_ambiguous_mask = out.get("non_ambiguous_mask")
        if non_ambiguous_mask is None:
            mask_out = torch.ones_like(depth_out)
        else:
            mask_out = non_ambiguous_mask.to(torch.float32).reshape(output_shape)

        return depth_out, conf_out, mask_out

@dataclass
class ExportConfig:
    """Runtime parameters for the export routine."""

    repo_path: Path
    outdir: Path
    output_name: Optional[str]
    report_name: Optional[str]
    fail_if_output_exists: bool
    height: int
    width: int
    opset: int
    checkpoint_path: Optional[Path]
    repo_url: Optional[str]
    repo_branch: Optional[str]
    fused_input: bool
    hf_model_id: Optional[str]
    hf_revision: Optional[str]
    include_intrinsics: bool
    return_conf_mask: bool
    export_device: str
    skip_eager_smoke: bool
    skip_ort: bool
    skip_simplify: bool
    skip_shape_inference: bool


def parse_args() -> ExportConfig:
    parser = argparse.ArgumentParser(description="Export MapAnything monocular depth model to ONNX")
    parser.add_argument("--repo", required=True, help="Path to the MapAnything repository")
    parser.add_argument("--outdir", default="ma_onnx_out_clean", help="Directory where ONNX files will be written")
    parser.add_argument(
        "--output-name",
        default="",
        help=(
            "Optional exact ONNX filename inside --outdir. The default remains "
            "model.onnx (or model_fused.onnx)."
        ),
    )
    parser.add_argument(
        "--report-name",
        default="",
        help=(
            "Optional export-report filename inside --outdir. With an explicit "
            "--output-name, the default is <output-stem>.export_report.txt."
        ),
    )
    parser.add_argument(
        "--fail-if-output-exists",
        action="store_true",
        help="Refuse to replace any ONNX, sidecar, derived graph, or report output.",
    )
    # For 1920x1080 sources, MapAnything preprocess uses max side=518 and rounds to patch-size (14):
    # 518x294 (W×H). Keep these as the default export dims for DS9.1 full-frame inference.
    parser.add_argument("--h", type=int, default=294, help="Input image height (default: 294 for 16:9 sources)")
    parser.add_argument("--w", type=int, default=518, help="Input image width (default: 518 for 16:9 sources)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--ckpt", default="", help="Optional checkpoint path for weight loading")
    parser.add_argument(
        "--hf-model-id",
        default="facebook/map-anything-apache",
        help="HuggingFace model id to load via MapAnything.from_pretrained (recommended; pass '' to disable).",
    )
    parser.add_argument(
        "--hf-revision",
        default="",
        help=(
            "Optional immutable Hugging Face model revision. Candidate/release "
            "work should always pass a full commit SHA."
        ),
    )
    parser.add_argument("--repo-url", default=None, help="Repository URL for reporting")
    parser.add_argument("--repo-branch", default=None, help="Repository branch for reporting")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help=(
            "Exact device used to load and export the model. The exporter fails "
            "if the requested device is unavailable; it never falls back."
        ),
    )
    parser.add_argument(
        "--skip-eager-smoke",
        action="store_true",
        help=(
            "Skip the redundant eager wrapper forward before ONNX tracing. "
            "The ONNX export and downstream engine functional test still run."
        ),
    )
    parser.add_argument("--skip-ort", action="store_true", help="Skip ONNX Runtime smoke test (saves RAM/time).")
    parser.add_argument("--skip-simplify", action="store_true", help="Skip onnxsim simplification (saves RAM/time).")
    parser.add_argument("--skip-shape-inference", action="store_true", help="Skip ONNX shape inference step.")
    fused_help = "Export fused 12-channel input variant (mapanything_fused). Disable to export images-only model."
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--fused-input",
            action=argparse.BooleanOptionalAction,  # type: ignore[attr-defined]
            default=False,
            help=fused_help,
        )
    else:  # pragma: no cover - fallback for older Python
        parser.add_argument("--fused-input", dest="fused_input", action="store_true", default=False, help=fused_help)
        parser.add_argument("--no-fused-input", dest="fused_input", action="store_false")
    intrinsics_help = "Whether to compute ray_directions_cam from intrinsics (images-only inference works without calibration inputs)."
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--include-intrinsics",
            action=argparse.BooleanOptionalAction,  # type: ignore[attr-defined]
            default=False,
            help=intrinsics_help,
        )
        parser.add_argument(
            "--return-conf-mask",
            action=argparse.BooleanOptionalAction,  # type: ignore[attr-defined]
            default=True,
            help="Return depth + conf + mask outputs (for DS9.1 tensor meta postprocess).",
        )
    else:  # pragma: no cover
        parser.add_argument("--include-intrinsics", dest="include_intrinsics", action="store_true", default=False, help=intrinsics_help)
        parser.add_argument("--no-include-intrinsics", dest="include_intrinsics", action="store_false")
        parser.add_argument("--return-conf-mask", dest="return_conf_mask", action="store_true", default=True, help="Return depth + conf + mask outputs.")
        parser.add_argument("--no-return-conf-mask", dest="return_conf_mask", action="store_false")
    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    checkpoint_path: Optional[Path] = None
    if args.ckpt:
        checkpoint_path = Path(args.ckpt).expanduser().resolve()

    return ExportConfig(
        repo_path=repo_path,
        outdir=outdir,
        output_name=_safe_output_name(args.output_name) if args.output_name else None,
        report_name=_safe_report_name(args.report_name) if args.report_name else None,
        fail_if_output_exists=bool(args.fail_if_output_exists),
        height=args.h,
        width=args.w,
        opset=args.opset,
        checkpoint_path=checkpoint_path,
        repo_url=args.repo_url,
        repo_branch=args.repo_branch,
        fused_input=bool(args.fused_input),
        hf_model_id=str(args.hf_model_id).strip() or None,
        hf_revision=str(args.hf_revision).strip() or None,
        include_intrinsics=bool(getattr(args, "include_intrinsics", False)),
        return_conf_mask=bool(getattr(args, "return_conf_mask", True)),
        export_device=str(args.device),
        skip_eager_smoke=bool(getattr(args, "skip_eager_smoke", False)),
        skip_ort=bool(getattr(args, "skip_ort", False)),
        skip_simplify=bool(getattr(args, "skip_simplify", False)),
        skip_shape_inference=bool(getattr(args, "skip_shape_inference", False)),
    )


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _model_filename(cfg: ExportConfig) -> str:
    return cfg.output_name or ("model_fused.onnx" if cfg.fused_input else "model.onnx")


def _report_filename(cfg: ExportConfig) -> str:
    if cfg.report_name:
        return cfg.report_name
    if cfg.output_name:
        return f"{Path(cfg.output_name).stem}.export_report.txt"
    return "export_report.txt"


def _guard_export_outputs(cfg: ExportConfig) -> None:
    if not cfg.fail_if_output_exists:
        return
    model_path = cfg.outdir / _model_filename(cfg)
    candidates = (
        model_path,
        model_path.with_suffix(model_path.suffix + ".data"),
        model_path.with_name(f"{model_path.stem}-inferred.onnx"),
        model_path.with_name(f"{model_path.stem}-sim.onnx"),
        cfg.outdir / _report_filename(cfg),
    )
    existing = [path for path in candidates if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            "refusing to replace existing export output(s): "
            + ", ".join(str(path) for path in existing)
        )


def get_git_commit(repo_path: Path) -> str:
    try:
        return (
            subprocess.check_output([
                "git",
                "-C",
                str(repo_path),
                "rev-parse",
                "HEAD",
            ], text=True)
            .strip()
        )
    except Exception as exc:  # pragma: no cover - git may be unavailable
        LOGGER.warning("Could not determine git commit: %s", exc)
        return "unknown"


def _explicit_keyword_parameters(callable_obj: Any) -> set[str]:
    return {
        parameter.name
        for parameter in inspect.signature(callable_obj).parameters.values()
        if parameter.name != "self"
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def _validate_hf_config_compatibility(
    *,
    model_class: type,
    encoder_class: type,
    model_config: Dict[str, Any],
    uniception_version: str,
) -> None:
    """Require the source APIs that give every checkpoint option its semantics."""

    try:
        installed_uniception = Version(uniception_version)
    except InvalidVersion as exc:
        raise ModelCompatibilityError(
            f"unable to interpret UniCeption version {uniception_version!r}"
        ) from exc
    if installed_uniception < MIN_UNICEPTION_VERSION:
        raise ModelCompatibilityError(
            "Hugging Face MapAnything v1.1 checkpoints require UniCeption "
            f">={MIN_UNICEPTION_VERSION}; found {installed_uniception}. "
            "Use a task-isolated environment with the matching official "
            "MapAnything dependencies; do not strip checkpoint options."
        )

    model_parameters = _explicit_keyword_parameters(model_class.__init__)
    unsupported_model_options = sorted(set(model_config) - model_parameters)
    if unsupported_model_options:
        raise ModelCompatibilityError(
            "MapAnything source does not implement checkpoint option(s): "
            + ", ".join(unsupported_model_options)
        )

    encoder_config = model_config.get("encoder_config")
    if not isinstance(encoder_config, dict):
        raise ModelCompatibilityError(
            "Hugging Face MapAnything config is missing encoder_config"
        )
    encoder_parameters = _explicit_keyword_parameters(encoder_class.__init__)
    factory_only_options = {"encoder_str", "uses_torch_hub"}
    unsupported_encoder_options = sorted(
        set(encoder_config) - encoder_parameters - factory_only_options
    )
    if unsupported_encoder_options:
        raise ModelCompatibilityError(
            "UniCeption DINOv2 source does not implement checkpoint option(s): "
            + ", ".join(unsupported_encoder_options)
        )


def _load_and_validate_hf_config(
    cfg: ExportConfig,
    *,
    model_class: type,
    encoder_class: type,
) -> Dict[str, Any]:
    if not cfg.hf_model_id:
        raise ModelCompatibilityError("Hugging Face model id is required")
    config_path = hf_hub_download(
        repo_id=cfg.hf_model_id,
        filename="config.json",
        revision=cfg.hf_revision,
    )
    with Path(config_path).open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    if not isinstance(model_config, dict):
        raise ModelCompatibilityError(
            "Hugging Face MapAnything config must be a JSON object"
        )
    try:
        uniception_version = importlib.metadata.version("uniception")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ModelCompatibilityError(
            "UniCeption is not installed in the exporter environment"
        ) from exc
    _validate_hf_config_compatibility(
        model_class=model_class,
        encoder_class=encoder_class,
        model_config=model_config,
        uniception_version=uniception_version,
    )
    return model_config


def _load_hf_pretrained_model(cfg: ExportConfig, model_class: type) -> Any:
    if not cfg.hf_model_id:
        raise ModelCompatibilityError("Hugging Face model id is required")
    return model_class.from_pretrained(
        cfg.hf_model_id,
        revision=cfg.hf_revision,
        strict=True,
    )


def _resolve_export_device(requested: str) -> torch.device:
    """Resolve one explicit export device without an implicit fallback."""

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA export was requested, but torch.cuda.is_available() is false"
        )
    return device


def load_model(cfg: ExportConfig) -> Dict[str, Any]:
    """Load the MapAnything model using Hydra configuration."""
    sys.path.insert(0, str(cfg.repo_path))

    from hydra import compose, initialize_config_dir  # type: ignore
    from mapanything.models import init_model  # type: ignore
    from mapanything.models import MapAnything  # type: ignore
    from uniception.models.encoders.dinov2 import DINOv2Encoder  # type: ignore
    from uniception.models.encoders.image_normalizations import (  # type: ignore
        IMAGE_NORMALIZATION_DICT,
    )

    configs_dir = cfg.repo_path / "configs"
    if not configs_dir.exists():
        raise FileNotFoundError(f"Could not locate configs directory at {configs_dir}")

    checkpoint_status = "no checkpoint provided"
    checkpoint_error: Optional[str] = None

    if cfg.hf_model_id:
        _load_and_validate_hf_config(
            cfg,
            model_class=MapAnything,
            encoder_class=DINOv2Encoder,
        )
        revision_label = cfg.hf_revision or "main"
        LOGGER.info(
            "Loading MapAnything via HuggingFace: %s@%s",
            cfg.hf_model_id,
            revision_label,
        )
        model = _load_hf_pretrained_model(cfg, MapAnything)
        model.eval()
        checkpoint_status = f"hf:{cfg.hf_model_id}@{revision_label} (strict)"
    else:
        LOGGER.info("Composing Hydra config for MapAnything model")
        with initialize_config_dir(config_dir=str(configs_dir), job_name="ma_export", version_base=None):
            hydra_cfg = compose(config_name="model/mapanything.yaml")

            model_section = hydra_cfg.model
        model_str = model_section.model_str
        model_config = model_section.model_config
        torch_hub_force_reload = bool(model_section.get("torch_hub_force_reload", False))

        @contextlib.contextmanager
        def disable_dinov2_pretrained_download():
            import torch.hub as torch_hub  # Local import to avoid global mutation if torch is unavailable
            original_load = torch_hub.load

            def patched_load(repo_or_dir, model, *args, **kwargs):
                if repo_or_dir == "facebookresearch/dinov2":
                    kwargs.setdefault("pretrained", False)
                    kwargs.setdefault("weights", None)
                    LOGGER.info("Patching torch.hub.load for %s to avoid pretrained weights", model)
                return original_load(repo_or_dir, model, *args, **kwargs)

            torch_hub.load = patched_load  # type: ignore[assignment]
            try:
                yield
            finally:
                torch_hub.load = original_load  # type: ignore[assignment]

        LOGGER.info("Instantiating model '%s'", model_str)
        with disable_dinov2_pretrained_download():
            model = init_model(
                model_str=model_str,
                model_config=model_config,
                torch_hub_force_reload=torch_hub_force_reload,
            )
        model.eval()

    if cfg.checkpoint_path:
        LOGGER.info("Loading checkpoint from %s", cfg.checkpoint_path)
        try:
            state_dict = torch.load(cfg.checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            checkpoint_status = "loaded"
            if missing or unexpected:
                checkpoint_status += f" (missing={len(missing)}, unexpected={len(unexpected)})"
                if missing:
                    LOGGER.warning("Missing keys while loading checkpoint: %s", missing)
                if unexpected:
                    LOGGER.warning("Unexpected keys while loading checkpoint: %s", unexpected)
        except Exception as exc:  # pragma: no cover - best effort
            checkpoint_status = "failed"
            checkpoint_error = str(exc)
            LOGGER.warning("Failed to load checkpoint: %s", exc)

    norm_type = getattr(model.encoder, "data_norm_type", "dinov2")
    img_norm = IMAGE_NORMALIZATION_DICT.get(norm_type)
    if img_norm is None:
        LOGGER.warning("Unknown normalization type '%s', defaulting to identity", norm_type)
        mean = torch.zeros(1, 3, 1, 1)
        std = torch.ones(1, 3, 1, 1)
    else:
        mean = torch.tensor(img_norm.mean, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(img_norm.std, dtype=torch.float32).view(1, -1, 1, 1)

    wrapper = MapAnythingDepthWrapper(
        model,
        norm_type,
        mean,
        std,
        use_fused_input=cfg.fused_input,
        output_height=cfg.height,
        output_width=cfg.width,
        include_intrinsics=cfg.include_intrinsics,
        return_conf_mask=cfg.return_conf_mask,
    )
    target_device = _resolve_export_device(cfg.export_device)
    wrapper.to(target_device)
    LOGGER.info("Moved export wrapper to explicit device: %s", target_device)

    model_class_name = model.__class__.__name__
    model_module_path = inspect.getfile(model.__class__)

    return {
        "model": model,
        "wrapper": wrapper,
        "model_class_name": model_class_name,
        "model_module_path": model_module_path,
        "norm_type": norm_type,
        "checkpoint_status": checkpoint_status,
        "checkpoint_error": checkpoint_error,
        "img_norm": img_norm,
        "export_device": str(target_device),
    }


def export_onnx(wrapper: nn.Module, cfg: ExportConfig, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Export the wrapped model to ONNX (static and dynamic versions)."""
    import os
    os.makedirs(cfg.outdir, exist_ok=True)

    # Create dummy inputs
    height = cfg.height
    width = cfg.width
    if height % 14 != 0 or width % 14 != 0:
        orig_height, orig_width = height, width
        height = ((height // 14) + 1) * 14 if height % 14 != 0 else height
        width = ((width // 14) + 1) * 14 if width % 14 != 0 else width
        print(f"[WARNING] Adjusting input size from {orig_height}x{orig_width} to {height}x{width} to satisfy encoder patch size 14")

    try:
        export_device = next(wrapper.parameters()).device
    except StopIteration:
        export_device = getattr(getattr(wrapper, "mean", None), "device", torch.device("cpu"))

    dummy_images = torch.rand(1, 3, height, width, dtype=torch.float32, device=export_device)
    dummy_images_cpu = dummy_images.detach().cpu()

    output_names = ["depth"]
    if cfg.return_conf_mask:
        output_names = ["depth", "conf", "mask"]

    if cfg.fused_input:
        intr_template = torch.tensor(
            [
                1000.0,
                0.0,
                width / 2.0,
                0.0,
                1000.0,
                height / 2.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=torch.float32,
            device=export_device,
        ).view(1, 9, 1, 1)
        dummy_intr_map = intr_template.expand(-1, -1, height, width)
        dummy_fused = torch.cat([dummy_images, dummy_intr_map], dim=1)
        dummy_fused_cpu = dummy_fused.detach().cpu()
        export_inputs = (dummy_fused,)
        input_names = ["mapanything_fused"]
        dynamic_axes_inputs = {"mapanything_fused": {0: "batch"}}
        model_path = cfg.outdir / _model_filename(cfg)
    else:
        export_inputs = (dummy_images,)
        input_names = ["images"]
        dynamic_axes_inputs = {"images": {0: "batch"}}
        model_path = cfg.outdir / _model_filename(cfg)

    print(f"[INFO] Exporting static ONNX model to {model_path}")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            export_inputs,
            model_path.as_posix(),
            export_params=True,
            dynamo=False,
            opset_version=cfg.opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                **dynamic_axes_inputs,
                **{name: {0: "batch"} for name in output_names},
            },
        )

    static_external = False
    # Avoid loading massive external tensor blobs into memory unless needed.
    onnx_model = onnx.load(model_path.as_posix(), load_external_data=False)
    external_data_path = model_path.with_suffix(model_path.suffix + ".data")
    try:
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in onnx_model.graph.initializer):
            static_external = True
    except Exception:
        pass
    if external_data_path.exists():
        static_external = True

    def sanitize_infinite_constants(model: onnx.ModelProto) -> bool:
        modified = False
        for node in model.graph.node:
            if node.op_type != "Constant":
                continue
            for attr in node.attribute:
                if attr.name != "value" or attr.type != onnx.AttributeProto.TENSOR:
                    continue
                try:
                    arr = numpy_helper.to_array(attr.t)
                except Exception:
                    continue
                if not np.issubdtype(arr.dtype, np.floating):
                    continue
                if not np.isinf(arr).any():
                    continue
                dtype = arr.dtype
                finfo = np.finfo(dtype)
                arr = np.where(np.isposinf(arr), finfo.max, arr)
                arr = np.where(np.isneginf(arr), finfo.min, arr)
                attr.t.CopyFrom(numpy_helper.from_array(arr, name=attr.t.name or node.output[0]))
                modified = True
        return modified

    if sanitize_infinite_constants(onnx_model):
        LOGGER.info("Replaced +/-inf constants with finite float32 limits for TensorRT compatibility")
        onnx.save_model(
            onnx_model,
            model_path.as_posix(),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path.name,
        )
        static_external = True

    LOGGER.info("Running ONNX checker")
    onnx.checker.check_model(model_path.as_posix())

    if cfg.skip_shape_inference:
        LOGGER.info("Skipping shape inference (--skip-shape-inference)")
    else:
        LOGGER.info("Running shape inference")
        inferred_path = model_path.with_name(f"{model_path.stem}-inferred.onnx")
        onnx.shape_inference.infer_shapes_path(model_path.as_posix(), inferred_path.as_posix())

    simplified_model_path: Optional[Path] = None
    if cfg.skip_simplify:
        simplifier_status = "skipped (--skip-simplify)"
    elif onnxsim is None:
        simplifier_status = "onnxsim not installed"
    else:
        simplifier_target = model_path.with_name(f"{model_path.stem}-sim.onnx")
        try:
            LOGGER.info("Running onnxsim simplification")
            input_shapes = {"mapanything_fused": [1, 12, height, width]} if cfg.fused_input else {
                "images": [1, 3, height, width],
            }
            simplified_model, check_ok = onnxsim.simplify(
                model_path.as_posix(),
                input_shapes=input_shapes,
                overwrite_input_shapes=True,
            )
            if not check_ok:
                simplifier_status = "onnxsim reported validation failure"
            else:
                onnx.save(simplified_model, simplifier_target.as_posix())
                simplifier_status = f"ok ({simplifier_target.name})"
                simplified_model_path = simplifier_target
        except Exception as exc:  # pragma: no cover - best effort
            simplifier_status = f"failed: {exc}"

    unique_ops = sorted({node.op_type for node in onnx_model.graph.node})

    if cfg.skip_ort:
        LOGGER.info("Skipping ONNX Runtime smoke test (--skip-ort)")
        ort_result = {"skipped": True}
    else:
        LOGGER.info("Running ONNX Runtime smoke test")
        ort_session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
        if cfg.fused_input:
            ort_inputs = {"mapanything_fused": dummy_fused_cpu.numpy()}
        else:
            ort_inputs = {"images": dummy_images_cpu.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        ort_result = {
            "num_outputs": len(ort_outputs),
            "shapes": [list(out.shape) for out in ort_outputs],
            "dtypes": [str(out.dtype) for out in ort_outputs],
        }

    dynamic_status = "not attempted"
    dynamic_error: Optional[str] = None
    LOGGER.info("Skipping dynamic H/W export to preserve static weights")
    dynamic_status = "skipped"
    dynamic_external = False
    dynamic_error = None

    if simplified_model_path is None:
        fallback_target = model_path.with_name(f"{model_path.stem}-sim.onnx")
        if not fallback_target.exists():
            shutil.copyfile(model_path, fallback_target)
        simplifier_status = f"{simplifier_status}; fallback copy created"
        simplified_model_path = fallback_target

    return {
        "model_path": model_path,
        "unique_ops": unique_ops,
        "simplifier_status": simplifier_status,
        "simplified_model_path": simplified_model_path,
        "ort_result": ort_result,
        "dynamic_status": dynamic_status,
        "dynamic_error": dynamic_error,
        "height": height,
        "width": width,
        "external_data_path": external_data_path if static_external else None,
        "patch_size": getattr(getattr(artifacts["model"], "encoder", None), "patch_size", None),
        "static_external_data": static_external,
        "dynamic_external_data": dynamic_external,
    }


def write_report(
    cfg: ExportConfig,
    artifacts: Dict[str, Any],
    export_info: Dict[str, Any],
    git_sha: str,
) -> None:
    report_path = cfg.outdir / _report_filename(cfg)
    ort_info = export_info["ort_result"]
    unique_ops = export_info["unique_ops"]

    lines = [
        "MapAnything Monocular Depth ONNX Export",
        "=======================================",
        f"Repository URL: {cfg.repo_url or 'unknown'}",
        f"Repository Branch: {cfg.repo_branch or 'unknown'}",
        f"Commit SHA: {git_sha}",
        "",
        f"Model class: {artifacts['model_class_name']}",
        f"Model definition: {artifacts['model_module_path']}",
        f"Wrapper module: {__name__}.MapAnythingDepthWrapper",
        f"Normalization type: {artifacts['norm_type']}",
        f"Fused input enabled: {cfg.fused_input}",
        f"Include intrinsics: {cfg.include_intrinsics}",
        f"Return conf/mask: {cfg.return_conf_mask}",
        f"Export device: {artifacts['export_device']}",
        f"HuggingFace model id: {cfg.hf_model_id or 'none'}",
        f"HuggingFace revision: {cfg.hf_revision or 'main'}",
        f"ONNX filename: {Path(export_info['model_path']).name}",
        "",
        "Input specification:",
    ]
    if cfg.fused_input:
        lines.extend(
            [
                f"  mapanything_fused: (N, 12, H, W) float32 (H={export_info['height']}, W={export_info['width']})",
                "    ▹ channels 0-2 = RGB in 0..1 range, channels 3-11 = tiled 3x3 intrinsics",
            ]
        )
    else:
        lines.extend(
            [
                f"  images:  (N, 3, H, W) float32, normalized to 0..1 (H={export_info['height']}, W={export_info['width']})",
                "    ▹ ray_directions_cam computed from pinhole intrinsics" if cfg.include_intrinsics else "    ▹ images-only (no calibration inputs)",
            ]
        )
    lines.extend(
        [
            "Output specification:",
            "  depth:   (N, 1, H, W) float32",
            "  conf:    (N, 1, H, W) float32" if cfg.return_conf_mask else "",
            "  mask:    (N, 1, H, W) float32" if cfg.return_conf_mask else "",
            "",
            f"Opset: {cfg.opset}",
            f"Unique ONNX ops ({len(unique_ops)}): {', '.join(unique_ops)}",
            "",
            "ONNX Runtime smoke test:",
            "  skipped: true" if ort_info.get("skipped") else f"  outputs: {ort_info.get('num_outputs')}",
            "" if ort_info.get("skipped") else f"  shapes: {ort_info.get('shapes')}",
            "" if ort_info.get("skipped") else f"  dtypes: {ort_info.get('dtypes')}",
            "",
            f"Checkpoint status: {artifacts['checkpoint_status']}",
        ]
    )
    if artifacts["checkpoint_error"]:
        lines.append(f"Checkpoint error: {artifacts['checkpoint_error']}")

    lines.extend(
        [
            "",
            f"Simplifier status: {export_info['simplifier_status']}",
            f"Dynamic H/W export: {export_info['dynamic_status']}",
        ]
    )
    if export_info.get("dynamic_error"):
        lines.append(f"Dynamic export error: {export_info['dynamic_error']}")
    if export_info.get("patch_size"):
        lines.append(
            f"Note: Input resolution aligned to encoder patch size {export_info['patch_size']}"
        )
    if export_info.get("static_external_data"):
        lines.append(
            "Static export saved tensors to external data files "
            f"({Path(export_info['external_data_path']).name})"
        )
    if export_info.get("simplified_model_path"):
        lines.append(f"Simplified graph saved to {export_info['simplified_model_path'].name}")

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    LOGGER.info("Wrote export report to %s", report_path)


def main() -> None:
    setup_logging()
    cfg = parse_args()
    _guard_export_outputs(cfg)
    print(f"[INFO] Starting export with configuration: {cfg}")

    # Load model and artifacts
    artifacts = load_model(cfg)

    # Adjust height to satisfy encoder patch size
    height = cfg.height
    width = cfg.width
    if height % 14 != 0 or width % 14 != 0:
        orig_height, orig_width = height, width
        height = ((height // 14) + 1) * 14 if height % 14 != 0 else height
        width = ((width // 14) + 1) * 14 if width % 14 != 0 else width
        print(f"[WARNING] Adjusting input size from {orig_height}x{orig_width} to {height}x{width} to satisfy encoder patch size 14")

    wrapper: MapAnythingDepthWrapper = artifacts["wrapper"]

    wrapper.eval()
    if cfg.skip_eager_smoke:
        LOGGER.info("Skipping redundant eager wrapper smoke (--skip-eager-smoke)")
    else:
        try:
            export_device = next(wrapper.parameters()).device
        except StopIteration:
            export_device = getattr(
                getattr(wrapper, "mean", None),
                "device",
                torch.device("cpu"),
            )
        dummy_images = torch.rand(
            1,
            3,
            height,
            width,
            dtype=torch.float32,
            device=export_device,
        )
        with torch.inference_mode():
            if cfg.fused_input:
                intr_template = torch.tensor(
                    [
                        1000.0,
                        0.0,
                        width / 2.0,
                        0.0,
                        1000.0,
                        height / 2.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    dtype=torch.float32,
                    device=export_device,
                ).view(1, 9, 1, 1)
                dummy_intr_map = intr_template.expand(-1, -1, height, width)
                dummy_fused = torch.cat([dummy_images, dummy_intr_map], dim=1)
                test_outputs = wrapper(dummy_fused)
            else:
                test_outputs = wrapper(dummy_images)
        if isinstance(test_outputs, (tuple, list)):
            print(
                "Test output shapes:",
                [tuple(getattr(x, "shape", ())) for x in test_outputs],
            )
        else:
            print("Test output shape:", tuple(getattr(test_outputs, "shape", ())))
        del test_outputs
        del dummy_images
        if export_device.type == "cuda":
            torch.cuda.empty_cache()

    # Export ONNX models
    export_info = export_onnx(wrapper, cfg, artifacts)

    git_sha = get_git_commit(cfg.repo_path)
    write_report(cfg, artifacts, export_info, git_sha)

    print("[INFO] Export pipeline completed successfully!")
    print(f"[INFO] Output directory: {cfg.outdir}")
    return export_info


if __name__ == "__main__":
    main()
