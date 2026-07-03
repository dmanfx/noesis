#!/usr/bin/env python3
"""Download/export YOLO11 detect/seg ONNX files and build TensorRT engines."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, clean_library_path: bool = False) -> None:
    print(f"[RUN] {' '.join(str(part) for part in cmd)}")
    env = os.environ.copy()
    if clean_library_path:
        env.pop("LD_LIBRARY_PATH", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
    subprocess.run([str(part) for part in cmd], check=True, cwd=str(REPO_ROOT), env=env)


def _trtexec() -> str:
    binary = shutil.which("trtexec")
    if not binary:
        raise FileNotFoundError("trtexec not found on PATH")
    return binary


def _sizes(raw: str) -> list[str]:
    sizes = [item.strip().lower() for item in str(raw or "").split(",") if item.strip()]
    invalid = [item for item in sizes if item not in {"s", "m", "l"}]
    if invalid:
        raise SystemExit(f"Unsupported YOLO11 size(s): {invalid}. Expected subset of s,m,l")
    return sizes


def _profiles(raw: str) -> set[str]:
    items = {item.strip().lower() for item in str(raw or "").split(",") if item.strip()}
    if not items:
        return {"detect", "seg"}
    if "both" in items:
        items.remove("both")
        items.update({"detect", "seg"})
    invalid = sorted(items - {"detect", "seg"})
    if invalid:
        raise SystemExit(f"Unsupported profile(s): {invalid}. Expected detect,seg,both")
    return items


def _ensure_weights(size: str, *, seg: bool) -> Path:
    suffix = "-seg" if seg else ""
    name = f"yolo11{size}{suffix}.pt"
    models_dir = REPO_ROOT / "models"
    target = models_dir / name
    if target.exists() and target.stat().st_size > 0:
        return target

    root_download = REPO_ROOT / name
    if root_download.exists() and root_download.stat().st_size > 0:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root_download), str(target))
        return target

    from ultralytics import YOLO

    model = YOLO(name)
    if getattr(model, "ckpt_path", None):
        ckpt_path = Path(str(model.ckpt_path))
        if ckpt_path.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ckpt_path, target)
            return target
    if root_download.exists() and root_download.stat().st_size > 0:
        shutil.move(str(root_download), str(target))
        return target
    raise FileNotFoundError(f"Unable to download YOLO11 weights: {name}")


def _export_detect_onnx(size: str, *, opset: int, rebuild: bool) -> Path:
    target = REPO_ROOT / "models" / f"yolo11{size}.onnx"
    if target.exists() and target.stat().st_size > 0 and not rebuild:
        return target

    import torch
    import torch.nn as nn
    from ultralytics import YOLO
    from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder

    class DeepStreamDetectRows(nn.Module):
        def __init__(self, model: nn.Module):
            super().__init__()
            self.model = model

        def forward(self, x):  # type: ignore[no-untyped-def]
            pred = self.model(x)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred = pred.transpose(1, 2)
            boxes = pred[:, :, :4]
            scores = pred[:, :, 4:]
            conf, cls = scores.max(dim=-1, keepdim=True)
            return torch.cat([boxes, conf, cls.float()], dim=-1)

    weights = _ensure_weights(size, seg=False)
    yolo = YOLO(str(weights))
    model = deepcopy(yolo.model).to("cpu")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    for module in model.modules():
        if isinstance(module, (Detect, RTDETRDecoder)):
            module.dynamic = False
            module.export = True
            module.format = "onnx"
        elif isinstance(module, C2f):
            module.forward = module.forward_split

    wrapped = DeepStreamDetectRows(model).eval()
    sample = torch.zeros(3, 3, 640, 640)
    tmp = target.with_suffix(".onnx.tmp")
    if tmp.exists():
        tmp.unlink()
    torch.onnx.export(
        wrapped,
        sample,
        tmp,
        verbose=False,
        opset_version=int(opset),
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
    )
    tmp.replace(target)
    return target


def _export_seg_onnx(size: str, *, opset: int, rebuild: bool) -> Path:
    if size == "s":
        target = REPO_ROOT / "models" / "yolo11s-seg_cust_fused.onnx"
    else:
        target = REPO_ROOT / "models" / f"yolo11{size}-seg_cust.onnx"
    if target.exists() and target.stat().st_size > 0 and not rebuild:
        return target

    weights = _ensure_weights(size, seg=True)
    import torch
    import torch.nn as nn
    from ultralytics import YOLO
    from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder
    import ultralytics.models.yolo
    import ultralytics.utils
    import ultralytics.utils.tal as tal

    sys.modules["ultralytics.yolo"] = ultralytics.models.yolo
    sys.modules["ultralytics.yolo.utils"] = ultralytics.utils

    def _dist2bbox(distance, anchor_points, xywh=False, dim=-1):  # type: ignore[no-untyped-def]
        left_top, right_bottom = distance.chunk(2, dim)
        return torch.cat([anchor_points - left_top, anchor_points + right_bottom], dim)

    tal.dist2bbox.__code__ = _dist2bbox.__code__

    class RoiAlign(torch.autograd.Function):
        @staticmethod
        def forward(  # type: ignore[no-untyped-def]
            ctx,
            features,
            rois,
            batch_indices,
            coordinate_transformation_mode,
            mode,
            output_height,
            output_width,
            sampling_ratio,
            spatial_scale,
        ):
            del ctx, batch_indices, coordinate_transformation_mode, mode, sampling_ratio, spatial_scale
            return torch.randn(
                [rois.shape[0], features.shape[1], output_height, output_width],
                device=rois.device,
                dtype=rois.dtype,
            )

        @staticmethod
        def symbolic(  # type: ignore[no-untyped-def]
            graph,
            features,
            rois,
            batch_indices,
            coordinate_transformation_mode,
            mode,
            output_height,
            output_width,
            sampling_ratio,
            spatial_scale,
        ):
            return graph.op(
                "TRT::ROIAlignX_TRT",
                features,
                rois,
                batch_indices,
                coordinate_transformation_mode_i=coordinate_transformation_mode,
                mode_i=mode,
                output_height_i=output_height,
                output_width_i=output_width,
                sampling_ratio_i=sampling_ratio,
                spatial_scale_f=spatial_scale,
            )

    class NMS(torch.autograd.Function):
        @staticmethod
        def forward(ctx, boxes, scores, score_threshold, iou_threshold, max_output_boxes):  # type: ignore[no-untyped-def]
            del ctx, score_threshold, iou_threshold
            batch_size = scores.shape[0]
            num_classes = scores.shape[-1]
            num_detections = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
            detection_boxes = torch.randn(batch_size, max_output_boxes, 4)
            detection_scores = torch.randn(batch_size, max_output_boxes)
            detection_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
            detection_indices = torch.randint(0, max_output_boxes, (batch_size, max_output_boxes), dtype=torch.int32)
            return num_detections, detection_boxes, detection_scores, detection_classes, detection_indices

        @staticmethod
        def symbolic(graph, boxes, scores, score_threshold, iou_threshold, max_output_boxes):  # type: ignore[no-untyped-def]
            return graph.op(
                "TRT::EfficientNMSX_TRT",
                boxes,
                scores,
                score_threshold_f=score_threshold,
                iou_threshold_f=iou_threshold,
                max_output_boxes_i=max_output_boxes,
                background_class_i=-1,
                score_activation_i=0,
                class_agnostic_i=0,
                box_coding_i=0,
                outputs=5,
            )

    class DeepStreamSegRows(nn.Module):
        def __init__(self, nc: int, conf_threshold: float, iou_threshold: float, max_detections: int):
            super().__init__()
            self.nc = int(nc)
            self.conf_threshold = float(conf_threshold)
            self.iou_threshold = float(iou_threshold)
            self.max_detections = int(max_detections)

        def forward(self, x):  # type: ignore[no-untyped-def]
            preds = x[0].transpose(1, 2)
            boxes = preds[:, :, :4]
            scores = preds[:, :, 4 : self.nc + 4]
            masks = preds[:, :, self.nc + 4 :]
            protos = x[1]

            _, detection_boxes, detection_scores, detection_classes, detection_indices = NMS.apply(
                boxes,
                scores,
                self.conf_threshold,
                self.iou_threshold,
                self.max_detections,
            )

            batch_size, num_protos, proto_h, proto_w = protos.shape
            total_detections = batch_size * self.max_detections
            batch_index = torch.ones_like(detection_indices) * torch.arange(
                batch_size, device=boxes.device, dtype=torch.int32
            ).unsqueeze(1)
            batch_index = batch_index.view(total_detections).to(torch.int32)
            box_index = detection_indices.view(total_detections).to(torch.int32)

            selected_boxes = boxes[batch_index, box_index]
            selected_masks = masks[batch_index, box_index]
            pooled_proto = RoiAlign.apply(
                protos,
                selected_boxes,
                batch_index,
                1,
                1,
                int(proto_h),
                int(proto_w),
                0,
                0.25,
            )
            mask_rows = torch.matmul(
                selected_masks.unsqueeze(1),
                pooled_proto.view(total_detections, num_protos, proto_h * proto_w),
            )
            mask_rows = mask_rows.sigmoid().view(batch_size, self.max_detections, proto_h * proto_w)
            return (
                torch.cat(
                    [detection_boxes, detection_scores.unsqueeze(-1), detection_classes.unsqueeze(-1), mask_rows],
                    dim=-1,
                ),
                protos,
            )

    yolo = YOLO(str(weights))
    model = deepcopy(yolo.model).to("cpu")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    for module in model.modules():
        if isinstance(module, (Detect, RTDETRDecoder)):
            module.dynamic = False
            module.export = True
            module.format = "onnx"
        elif isinstance(module, C2f):
            module.forward = module.forward_split

    wrapped = nn.Sequential(model, DeepStreamSegRows(len(model.names), 0.25, 0.45, 30)).eval()
    sample = torch.zeros(1, 3, 640, 640)
    tmp = target.with_suffix(".onnx.tmp")
    if tmp.exists():
        tmp.unlink()
    torch.onnx.export(
        wrapped,
        sample,
        tmp,
        verbose=False,
        opset_version=int(opset),
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output0", "output1"],
        dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}, "output1": {0: "batch"}},
        training=torch.onnx.TrainingMode.EVAL,
        keep_initializers_as_inputs=False,
        dynamo=False,
    )
    tmp.replace(target)
    return target


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    input_name: str,
    batch: int,
    rebuild: bool,
    plugin_lib: Path | None = None,
) -> Path:
    if engine_path.exists() and engine_path.stat().st_size > 0 and not rebuild:
        return engine_path
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = engine_path.with_suffix(engine_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    cmd = [
        _trtexec(),
        f"--onnx={onnx_path}",
        "--fp16",
        "--memPoolSize=workspace:4096",
        f"--minShapes={input_name}:1x3x640x640",
        f"--optShapes={input_name}:{int(batch)}x3x640x640",
        f"--maxShapes={input_name}:{int(batch)}x3x640x640",
        f"--saveEngine={tmp}",
        "--skipInference",
    ]
    if plugin_lib is not None:
        if not plugin_lib.exists():
            raise FileNotFoundError(f"TensorRT plugin library missing: {plugin_lib}")
        cmd.append(f"--staticPlugins={plugin_lib}")
    _run(cmd, clean_library_path=True)
    if not tmp.exists() or tmp.stat().st_size <= 0:
        raise RuntimeError(f"TensorRT engine was not created: {tmp}")
    if plugin_lib is not None:
        _run(
            [
                _trtexec(),
                f"--loadEngine={tmp}",
                f"--dynamicPlugins={plugin_lib}",
                "--duration=1",
                "--warmUp=100",
                "--iterations=5",
            ],
            clean_library_path=True,
        )
    tmp.replace(engine_path)
    return engine_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default="both", help="Comma-separated profiles: detect,seg,both. Default: both")
    parser.add_argument("--sizes", default="s,m,l", help="Comma-separated YOLO11 sizes to build: s,m,l. Default: s,m,l")
    parser.add_argument("--batch", type=int, default=3, help="Export/build max batch size. Default: 3")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset for new exports. Default: 18")
    parser.add_argument("--skip-export", action="store_true", help="Reuse existing ONNX files.")
    parser.add_argument("--skip-engine", action="store_true", help="Do not build TensorRT engines.")
    parser.add_argument("--rebuild", action="store_true", help="Overwrite existing ONNX/engine files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles = _profiles(args.profiles)
    sizes = _sizes(args.sizes)
    outputs: list[tuple[str, str, Path, Path]] = []

    for size in sizes:
        if "detect" in profiles:
            onnx_path = REPO_ROOT / "models" / f"yolo11{size}.onnx"
            if not args.skip_export:
                onnx_path = _export_detect_onnx(size, opset=int(args.opset), rebuild=bool(args.rebuild))
            elif not onnx_path.exists():
                raise FileNotFoundError(f"Missing detect ONNX for --skip-export: {onnx_path}")
            engine_path = REPO_ROOT / "models" / "engines" / f"yolo11{size}_b3_fp16.engine"
            if not args.skip_engine:
                engine_path = _build_engine(
                    onnx_path,
                    engine_path,
                    input_name="input",
                    batch=int(args.batch),
                    rebuild=bool(args.rebuild),
                )
            elif not engine_path.exists():
                raise FileNotFoundError(f"Missing detect engine for --skip-engine: {engine_path}")
            outputs.append(("detect", size, onnx_path, engine_path))

        if "seg" in profiles:
            if size == "s":
                onnx_path = REPO_ROOT / "models" / "yolo11s-seg_cust_fused.onnx"
                engine_path = REPO_ROOT / "models" / "engines" / "yolo11s-seg_cust_fused.engine"
            else:
                onnx_path = REPO_ROOT / "models" / f"yolo11{size}-seg_cust.onnx"
                engine_path = REPO_ROOT / "models" / "engines" / f"yolo11{size}-seg_cust.engine"
            if not args.skip_export:
                onnx_path = _export_seg_onnx(size, opset=int(args.opset), rebuild=bool(args.rebuild))
            elif not onnx_path.exists():
                raise FileNotFoundError(f"Missing seg ONNX for --skip-export: {onnx_path}")
            if not args.skip_engine:
                plugin_lib = (
                    REPO_ROOT
                    / "external"
                    / "DeepStream-Yolo-Seg"
                    / "nvdsinfer_custom_impl_Yolo_seg"
                    / "libnvdsinfer_custom_impl_Yolo_seg.so"
                )
                engine_path = _build_engine(
                    onnx_path,
                    engine_path,
                    input_name="images",
                    batch=int(args.batch),
                    rebuild=bool(args.rebuild),
                    plugin_lib=plugin_lib,
                )
            elif not engine_path.exists():
                raise FileNotFoundError(f"Missing seg engine for --skip-engine: {engine_path}")
            outputs.append(("seg", size, onnx_path, engine_path))

    print("\nBuilt YOLO11 profile assets:")
    for profile, size, onnx_path, engine_path in outputs:
        print(f"  profile={profile} size={size} onnx={onnx_path} engine={engine_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
