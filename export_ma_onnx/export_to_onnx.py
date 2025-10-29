#!/usr/bin/env python3
"""Standalone MapAnything monocular depth ONNX exporter."""

from __future__ import annotations

import argparse
import logging
import subprocess
import shutil
import sys
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import numpy as np
from onnx import numpy_helper

try:  # Optional dependency used for graph simplification
    import onnxsim  # type: ignore
except Exception:  # pragma: no cover - best effort dependency
    onnxsim = None  # type: ignore

from mapanything.utils.inference import preprocess_input_views_for_inference


LOGGER = logging.getLogger("export_ma_onnx")

class MapAnythingDepthWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        normalization_type: str,
        mean_tensor: torch.Tensor,
        std_tensor: torch.Tensor,
        use_fused_input: bool,
    ) -> None:
        super().__init__()
        self.model = base_model
        self.norm_type = normalization_type
        self.use_fused_input = use_fused_input
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)

    def forward(
        self,
        fused_or_images: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_fused_input:
            if intrinsics is not None:
                raise ValueError("Fused-input wrapper expects a single tensor argument")
            fused = fused_or_images
            if fused.dim() != 4:
                raise ValueError(f"Expected fused tensor with shape (N,12,H,W), got {tuple(fused.shape)}")
            if fused.size(1) != 12:
                raise ValueError(f"Expected fused tensor to have 12 channels, got {fused.size(1)}")
            images = fused[:, :3, :, :]
            intr_map = fused[:, 3:, :, :]
            # Average spatial dimensions to recover the flattened 3x3 intrinsics.
            intr_flat = intr_map.mean(dim=(-2, -1))
            intrinsics_mat = intr_flat.view(-1, 3, 3)
        else:
            images = fused_or_images
            if intrinsics is None:
                raise ValueError("Two-input wrapper requires intrinsics tensor")
            intrinsics_mat = intrinsics
            if images.dim() != 4:
                raise ValueError(f"Expected images with shape (N,3,H,W), got {tuple(images.shape)}")
            if intrinsics_mat.dim() != 3:
                raise ValueError(f"Expected intrinsics with shape (N,3,3), got {tuple(intrinsics_mat.shape)}")
            if images.size(0) != intrinsics_mat.size(0):
                raise ValueError(f"Batch size mismatch: images {images.size(0)} vs intrinsics {intrinsics_mat.size(0)}")

        normalized = (images - self.mean) / self.std

        view = {
            "img": normalized[0].unsqueeze(0),
            "intrinsics": intrinsics_mat[0].unsqueeze(0),
            "data_norm_type": [self.norm_type],
        }
        output = self.model.infer([view])
        output_dict = output[0]
        depth_tensor = output_dict["depth_along_ray"] if "depth_along_ray" in output_dict else output_dict["depth_z"]
        depth = depth_tensor[0, :, :, 0]
        return depth.unsqueeze(0).unsqueeze(0)

@dataclass
class ExportConfig:
    """Runtime parameters for the export routine."""

    repo_path: Path
    outdir: Path
    height: int
    width: int
    opset: int
    checkpoint_path: Optional[Path]
    repo_url: Optional[str]
    repo_branch: Optional[str]
    fused_input: bool


def parse_args() -> ExportConfig:
    parser = argparse.ArgumentParser(description="Export MapAnything monocular depth model to ONNX")
    parser.add_argument("--repo", required=True, help="Path to the MapAnything repository")
    parser.add_argument("--outdir", default="ma_onnx_out_fused", help="Directory where ONNX files will be written")
    parser.add_argument("--h", type=int, default=512, help="Input image height")
    parser.add_argument("--w", type=int, default=512, help="Input image width")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--ckpt", default="", help="Optional checkpoint path for weight loading")
    parser.add_argument("--repo-url", default=None, help="Repository URL for reporting")
    parser.add_argument("--repo-branch", default=None, help="Repository branch for reporting")
    fused_help = "Export fused single-input variant (mapanything_fused). Disable for legacy two-input export."
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--fused-input",
            action=argparse.BooleanOptionalAction,  # type: ignore[attr-defined]
            default=True,
            help=fused_help,
        )
    else:  # pragma: no cover - fallback for older Python
        parser.add_argument("--fused-input", dest="fused_input", action="store_true", default=True, help=fused_help)
        parser.add_argument("--no-fused-input", dest="fused_input", action="store_false")
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
        height=args.h,
        width=args.w,
        opset=args.opset,
        checkpoint_path=checkpoint_path,
        repo_url=args.repo_url,
        repo_branch=args.repo_branch,
        fused_input=bool(args.fused_input),
    )


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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


def load_model(cfg: ExportConfig) -> Dict[str, Any]:
    """Load the MapAnything model using Hydra configuration."""
    sys.path.insert(0, str(cfg.repo_path))

    from hydra import compose, initialize_config_dir  # type: ignore
    from mapanything.models import init_model  # type: ignore
    from uniception.models.encoders.image_normalizations import (  # type: ignore
        IMAGE_NORMALIZATION_DICT,
    )

    configs_dir = cfg.repo_path / "configs"
    if not configs_dir.exists():
        raise FileNotFoundError(f"Could not locate configs directory at {configs_dir}")

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

    checkpoint_status = "no checkpoint provided"
    checkpoint_error: Optional[str] = None
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

    wrapper = MapAnythingDepthWrapper(model, norm_type, mean, std, use_fused_input=cfg.fused_input)

    import inspect

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

    dummy_images = torch.rand(1, 3, height, width, dtype=torch.float32)
    dummy_intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0)

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
        ).view(1, 9, 1, 1)
        dummy_intr_map = intr_template.expand(-1, -1, height, width)
        dummy_fused = torch.cat([dummy_images, dummy_intr_map], dim=1)
        export_inputs = (dummy_fused,)
        input_names = ["mapanything_fused"]
        dynamic_axes_inputs = {"mapanything_fused": {0: "batch", 2: "height", 3: "width"}}
        model_path = cfg.outdir / "model_fused.onnx"
    else:
        export_inputs = (dummy_images, dummy_intrinsics)
        input_names = ["images", "intrinsics"]
        dynamic_axes_inputs = {
            "images": {0: "batch"},
            "intrinsics": {0: "batch"},
        }
        model_path = cfg.outdir / "model.onnx"

    print(f"[INFO] Exporting static ONNX model to {model_path}")
    torch.onnx.export(
        wrapper,
        export_inputs,
        model_path.as_posix(),
        export_params=True,
        opset_version=cfg.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["depth"],
        dynamic_axes={
            **dynamic_axes_inputs,
            "depth": {0: "batch", 2: "height", 3: "width"},
        },
    )

    static_external = False
    onnx_model = onnx.load(model_path.as_posix(), load_external_data=True)

    def sanitize_infinite_constants(model: onnx.ModelProto) -> bool:
        modified = False
        for node in model.graph.node:
            if node.op_type != "Constant":
                continue
            for attr in node.attribute:
                if attr.name != "value" or attr.type != onnx.AttributeProto.TENSOR:
                    continue
                arr = numpy_helper.to_array(attr.t)
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
            all_tensors_to_one_file=False,
        )
        onnx_model = onnx.load(model_path.as_posix(), load_external_data=True)

    LOGGER.info("Running ONNX checker")
    onnx.checker.check_model(model_path.as_posix())

    LOGGER.info("Running shape inference")
    inferred_path = cfg.outdir / ("model_fused-inferred.onnx" if cfg.fused_input else "model-inferred.onnx")
    onnx.shape_inference.infer_shapes_path(model_path.as_posix(), inferred_path.as_posix())

    simplified_model_path: Optional[Path] = None
    if onnxsim is None:
        simplifier_status = "onnxsim not installed"
    else:
        simplifier_target = cfg.outdir / ("model_fused_sim.onnx" if cfg.fused_input else "model_sim.onnx")
        try:
            LOGGER.info("Running onnxsim simplification")
            input_shapes = {"mapanything_fused": [1, 12, height, width]} if cfg.fused_input else {
                "images": [1, 3, height, width],
                "intrinsics": [1, 3, 3],
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

    LOGGER.info("Running ONNX Runtime smoke test")
    ort_session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    if cfg.fused_input:
        ort_inputs = {"mapanything_fused": dummy_fused.numpy()}
    else:
        ort_inputs = {
            "images": dummy_images.numpy(),
            "intrinsics": dummy_intrinsics.numpy(),
        }
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_result = {
        "num_outputs": len(ort_outputs),
        "shapes": [list(out.shape) for out in ort_outputs],
        "dtypes": [str(out.dtype) for out in ort_outputs],
    }

    dynamic_status = "not attempted"
    dynamic_error: Optional[str] = None
    dynamic_path = cfg.outdir / "model-dyn.onnx"
    LOGGER.info("Skipping dynamic H/W export to preserve static weights")
    dynamic_status = "skipped"
    dynamic_external = False
    dynamic_error = None

    if simplified_model_path is None:
        fallback_target = cfg.outdir / ("model_fused_sim.onnx" if cfg.fused_input else "model_sim.onnx")
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
    report_path = cfg.outdir / "export_report.txt"
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
        "",
        "Input specification:",
    ]
    if cfg.fused_input:
        lines.extend(
            [
                f"  mapanything_fused: (N, 12, H, W) float32 (H={export_info['height']}, W={export_info['width']})",
                "    ▹ channels 0-2 = RGB (normalized), channels 3-11 = tiled 3x3 intrinsics",
            ]
        )
    else:
        lines.extend(
            [
                f"  images:  (N, 3, H, W) float32, normalized to 0..1 (H={export_info['height']}, W={export_info['width']})",
                "  intrinsics: (N, 3, 3) float32",
            ]
        )
    lines.extend(
        [
            "Output specification:",
            "  depth:   (N, 1, H, W) float32",
            "",
            f"Opset: {cfg.opset}",
            f"Unique ONNX ops ({len(unique_ops)}): {', '.join(unique_ops)}",
            "",
            "ONNX Runtime smoke test:",
            f"  outputs: {ort_info['num_outputs']}",
            f"  shapes: {ort_info['shapes']}",
            f"  dtypes: {ort_info['dtypes']}",
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
        lines.append("Static export saved tensors to external data files (model.onnx.data)")
    if export_info.get("simplified_model_path"):
        lines.append(f"Simplified graph saved to {export_info['simplified_model_path'].name}")

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    LOGGER.info("Wrote export report to %s", report_path)


def main() -> None:
    setup_logging()
    cfg = parse_args()
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

    # Test wrapper
    dummy_images = torch.rand(1, 3, height, width, dtype=torch.float32)
    dummy_intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0)
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
        ).view(1, 9, 1, 1)
        dummy_intr_map = intr_template.expand(-1, -1, height, width)
        dummy_fused = torch.cat([dummy_images, dummy_intr_map], dim=1)
        test_depth = wrapper(dummy_fused)
    else:
        test_depth = wrapper(dummy_images, dummy_intrinsics)
    print("Test depth shape:", test_depth.shape)

    wrapper.eval()

    # Export ONNX models
    export_info = export_onnx(wrapper, cfg, artifacts)

    git_sha = get_git_commit(cfg.repo_path)
    write_report(cfg, artifacts, export_info, git_sha)

    print(f"[INFO] Export pipeline completed successfully!")
    print(f"[INFO] Output directory: {cfg.outdir}")
    return export_info


if __name__ == "__main__":
    main()
