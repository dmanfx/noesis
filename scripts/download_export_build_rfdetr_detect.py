#!/usr/bin/env python3
"""Download/export RF-DETR detector ONNX files and build TensorRT engines."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import urllib.request
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from rfdetr.models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch

REPO_ROOT = Path(__file__).resolve().parents[1]


VARIANTS = {
    "n": {"model": "rf-detr-nano.pth", "resolution": 384, "max_detections": 30},
    "s": {"model": "rf-detr-small.pth", "resolution": 512, "max_detections": 50},
    "m": {"model": "rf-detr-medium.pth", "resolution": 576, "max_detections": 80},
}


def _run(cmd: list[str], *, clean_library_path: bool = False) -> None:
    print(f"[RUN] {' '.join(str(part) for part in cmd)}")
    env = os.environ.copy()
    if clean_library_path:
        env.pop("LD_LIBRARY_PATH", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
    subprocess.run([str(part) for part in cmd], check=True, cwd=str(REPO_ROOT), env=env)


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    print(f"[DL ] {url} -> {dst}")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dst)


def _patch_rfdetr_for_tensorrt_export() -> None:
    import rfdetr.models.backbone.projector as projector
    import rfdetr.models.ops.modules.ms_deform_attn as msda_module

    def layer_norm_forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (int(x.size(3)),), self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

    projector.LayerNorm.forward.__code__ = layer_norm_forward.__code__

    def ms_deform_attn_forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        class MultiscaleDeformableAttnPlugin(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
                value = value.permute(0, 2, 3, 1)
                n, lq, heads, levels, points, coords = sampling_locations.shape
                del n, lq, heads, levels, points, coords
                attention_weights = attention_weights.view(
                    attention_weights.shape[0],
                    attention_weights.shape[1],
                    attention_weights.shape[2],
                    attention_weights.shape[3] * attention_weights.shape[4],
                )
                return ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights)

            @staticmethod
            def symbolic(g, value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
                return g.op(
                    "TRT::MultiscaleDeformableAttnPlugin_TRT",
                    value,
                    spatial_shapes,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                )

        n, len_q, _ = query.shape
        del len_q
        n, len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        sampling_offsets = self.sampling_offsets(query).view(
            n, -1, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            n, -1, self.n_heads, self.n_levels * self.n_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}")

        attention_weights = F.softmax(attention_weights, -1)
        value = value.transpose(1, 2).contiguous().view(n, self.n_heads, self.d_model // self.n_heads, len_in)
        value = value.permute(0, 3, 1, 2)
        levels, points = sampling_locations.shape[3:5]
        attention_weights = attention_weights.view(n, -1, self.n_heads, levels, points)

        output = MultiscaleDeformableAttnPlugin.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights
        )
        output = output.view(n, -1, self.d_model)
        return self.output_proj(output)

    msda_module.MSDeformAttn.forward.__code__ = ms_deform_attn_forward.__code__


class DeepStreamRFDETRDetectOutput(nn.Module):
    def __init__(self, max_detections: int, person_class_idx: int = 1) -> None:
        super().__init__()
        self.max_detections = int(max_detections)
        self.person_class_idx = int(person_class_idx)

    def forward(self, outputs):
        if isinstance(outputs, dict):
            boxes = outputs["pred_boxes"]
            logits = outputs["pred_logits"]
        else:
            boxes, logits = outputs
        max_det = self.max_detections
        if max_det > 0 and boxes.shape[1] > max_det:
            person_idx = self.person_class_idx if logits.shape[2] > self.person_class_idx else 0
            scores = torch.sigmoid(logits[:, :, person_idx])
            _, topk_idx = torch.topk(scores, k=max_det, dim=1, largest=True, sorted=False)
            boxes = torch.gather(boxes, 1, topk_idx.unsqueeze(-1).expand(-1, -1, boxes.shape[2]))
            logits = torch.gather(logits, 1, topk_idx.unsqueeze(-1).expand(-1, -1, logits.shape[2]))
        return boxes, logits


def _model_class(size: str):
    from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall

    return {"n": RFDETRNano, "s": RFDETRSmall, "m": RFDETRMedium}[size]


def _export_onnx(size: str, *, batch: int, opset: int, simplify: bool) -> Path:
    from rfdetr.main import HOSTED_MODELS

    _patch_rfdetr_for_tensorrt_export()
    spec = VARIANTS[size]
    weight_key = str(spec["model"])
    weights = REPO_ROOT / "models" / weight_key
    if not weights.exists():
        url = HOSTED_MODELS.get(weight_key)
        if not url:
            raise KeyError(f"RF-DETR hosted URL missing for {weight_key}")
        _download(url, weights)

    resolution = int(spec["resolution"])
    max_detections = int(spec["max_detections"])
    model = _model_class(size)(
        pretrain_weights=str(weights),
        resolution=resolution,
        num_classes=90,
        device="cpu",
    )
    inner = deepcopy(model.model.model).eval()
    if hasattr(inner, "export"):
        inner.export()
    wrapped = nn.Sequential(inner, DeepStreamRFDETRDetectOutput(max_detections=max_detections)).eval()

    onnx_path = (REPO_ROOT / "models" / "onnx" / f"rfdetr_{size}_{resolution}.onnx").resolve()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(int(batch), 3, resolution, resolution)
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["dets", "labels"],
            opset_version=int(opset),
            do_constant_folding=True,
            dynamo=False,
            external_data=False,
        )

    if simplify:
        import onnx
        import onnxslim

        model_onnx = onnx.load(onnx_path)
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, onnx_path)
    return onnx_path


def _build_engine(onnx_path: Path, engine_path: Path) -> Path:
    if engine_path.exists() and engine_path.stat().st_size > 0:
        return engine_path
    trt_plugin_so = (
        REPO_ROOT
        / "external"
        / "DeepStream-Yolo-Seg"
        / "nvdsinfer_custom_impl_Yolo_seg"
        / "libnvdsinfer_custom_impl_Yolo_seg.so"
    ).resolve()
    if not trt_plugin_so.exists():
        raise FileNotFoundError(f"Missing TensorRT plugin library: {trt_plugin_so}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    if engine_path.exists():
        engine_path.unlink()
    _run(
        [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--dynamicPlugins={trt_plugin_so}",
            "--fp16",
            "--memPoolSize=workspace:4096",
            "--skipInference",
        ],
        clean_library_path=True,
    )
    if not engine_path.exists() or engine_path.stat().st_size <= 0:
        raise RuntimeError(f"TensorRT engine was not created: {engine_path}")
    return engine_path


def _sizes(raw: str) -> list[str]:
    sizes = [item.strip().lower() for item in str(raw or "").split(",") if item.strip()]
    invalid = [item for item in sizes if item not in VARIANTS]
    if invalid:
        raise SystemExit(f"Unsupported size(s): {invalid}. Expected subset of n,s,m")
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="n,s,m", help="Comma-separated RF-DETR sizes to build. Default: n,s,m")
    parser.add_argument("--batch", type=int, default=3, help="Static export batch size. Default: 3")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset. Default: 17")
    parser.add_argument("--simplify", action="store_true", help="Enable ONNX simplification after export.")
    parser.add_argument("--skip-export", action="store_true", help="Reuse existing ONNX files.")
    parser.add_argument("--skip-engine", action="store_true", help="Do not build TensorRT engines.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs: list[tuple[str, Path, Path]] = []
    for size in _sizes(args.sizes):
        resolution = int(VARIANTS[size]["resolution"])
        onnx_path = (REPO_ROOT / "models" / "onnx" / f"rfdetr_{size}_{resolution}.onnx").resolve()
        if not args.skip_export:
            onnx_path = _export_onnx(size, batch=int(args.batch), opset=int(args.opset), simplify=bool(args.simplify))
        elif not onnx_path.exists():
            raise FileNotFoundError(f"Missing ONNX for --skip-export: {onnx_path}")

        engine_path = (
            REPO_ROOT / "models" / "engines" / f"rfdetr_{size}_{resolution}_b{int(args.batch)}_fp16.engine"
        ).resolve()
        if not args.skip_engine:
            engine_path = _build_engine(onnx_path, engine_path)
        elif not engine_path.exists():
            raise FileNotFoundError(f"Missing engine for --skip-engine: {engine_path}")
        outputs.append((size, onnx_path, engine_path))

    print("\nBuilt RF-DETR detector assets:")
    for size, onnx_path, engine_path in outputs:
        print(f"  size={size} onnx={onnx_path} engine={engine_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
