#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DS9_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DS9_ROOT.parent
DS9_ONNX = DS9_ROOT / "models" / "onnx"
DS9_ENGINES = DS9_ROOT / "models" / "engines"
RFDETR_TRT_PLUGIN = Path(os.environ.get("NOESIS_RFDETR_TRT_PLUGIN_LIB", str(DS9_ROOT / "plugins" / "libnvdsinfer_custom_impl_Yolo_seg.so")))


@dataclass(frozen=True)
class EngineSpec:
    name: str
    source_onnx: Path
    staged_onnx: Path
    engine: Path
    trtexec_args: tuple[str, ...]
    precision_arg: str = "--fp16"


def _run(cmd: Sequence[str], *, dry_run: bool = False) -> None:
    print("[RUN]", " ".join(str(part) for part in cmd))
    if not dry_run:
        subprocess.run([str(part) for part in cmd], cwd=str(REPO_ROOT), check=True)


def _stage_onnx(src: Path, dst: Path, *, dry_run: bool = False) -> None:
    if not src.exists() or src.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing ONNX source: {src}")
    if src.resolve() == dst.resolve():
        print(f"[STAGE] {src} already staged")
        return
    print(f"[STAGE] {src} -> {dst}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _build(spec: EngineSpec, *, trtexec: str, dry_run: bool = False) -> None:
    _stage_onnx(spec.source_onnx, spec.staged_onnx, dry_run=dry_run)
    if not dry_run:
        spec.engine.parent.mkdir(parents=True, exist_ok=True)
        if spec.engine.exists():
            spec.engine.unlink()
    plugin_args: list[str] = []
    if spec.name.startswith("rfdetr"):
        if not RFDETR_TRT_PLUGIN.exists() or RFDETR_TRT_PLUGIN.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing RF-DETR TensorRT plugin library: {RFDETR_TRT_PLUGIN}")
        plugin_args.append(f"--dynamicPlugins={RFDETR_TRT_PLUGIN}")
    _run(
        [
            trtexec,
            f"--onnx={spec.staged_onnx}",
            spec.precision_arg,
            *plugin_args,
            *spec.trtexec_args,
            f"--saveEngine={spec.engine}",
            "--skipInference",
        ],
        dry_run=dry_run,
    )
    if not dry_run and (not spec.engine.exists() or spec.engine.stat().st_size <= 0):
        raise RuntimeError(f"TensorRT engine was not created: {spec.engine}")


def _specs(include_mapanything: bool) -> list[EngineSpec]:
    specs = [
        EngineSpec(
            "yolo11_seg",
            DS9_ONNX / "yolo11s-seg_cust_fused.onnx",
            DS9_ONNX / "yolo11s-seg_cust_fused.onnx",
            DS9_ENGINES / "yolo11s-seg_cust_fused.engine",
            (
                "--minShapes=images:3x3x640x640",
                "--optShapes=images:3x3x640x640",
                "--maxShapes=images:3x3x640x640",
            ),
        ),
        EngineSpec(
            "yolo11",
            DS9_ONNX / "yolo11m.onnx",
            DS9_ONNX / "yolo11m.onnx",
            DS9_ENGINES / "yolo11m_b3_fp16.engine",
            (
                "--minShapes=input:3x3x640x640",
                "--optShapes=input:3x3x640x640",
                "--maxShapes=input:3x3x640x640",
            ),
        ),
        EngineSpec(
            "reid_osnet",
            REPO_ROOT / "models" / "engines" / "reid_osnet_ibn_msmt17_dyn_b16.onnx",
            DS9_ONNX / "reid_osnet_ibn_msmt17_dyn_b16.onnx",
            DS9_ENGINES / "reid_osnet_ibn_msmt17_dyn_b16_fp16.engine",
            (
                "--minShapes=input:1x3x256x128",
                "--optShapes=input:16x3x256x128",
                "--maxShapes=input:16x3x256x128",
            ),
        ),
        EngineSpec(
            "yolo26_pose_n",
            DS9_ONNX / "yolo26n-pose_b3.onnx",
            DS9_ONNX / "yolo26n-pose_b3.onnx",
            DS9_ENGINES / "yolo26n-pose_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "depth_anything_v2_tracking",
            REPO_ROOT / "models" / "onnx" / "depth_anything_v2_metric_hypersim_vits_294x518_b3.onnx",
            DS9_ONNX / "depth_anything_v2_metric_hypersim_vits_294x518_b3.onnx",
            DS9_ENGINES / "depth_anything_v2_metric_hypersim_vits_294x518_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "yolo26_seg_n",
            REPO_ROOT / "models" / "yolo26n-seg_fused.onnx",
            DS9_ONNX / "yolo26n-seg_fused.onnx",
            DS9_ENGINES / "yolo26n-seg_fused_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "yolo26_seg_s",
            REPO_ROOT / "models" / "yolo26s-seg_fused.onnx",
            DS9_ONNX / "yolo26s-seg_fused.onnx",
            DS9_ENGINES / "yolo26s-seg_fused_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "yolo26_seg_m",
            REPO_ROOT / "models" / "yolo26m-seg_fused.onnx",
            DS9_ONNX / "yolo26m-seg_fused.onnx",
            DS9_ENGINES / "yolo26m-seg_fused_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "yolo26_n",
            DS9_ONNX / "yolo26n.onnx",
            DS9_ONNX / "yolo26n.onnx",
            DS9_ENGINES / "yolo26n_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "yolo26_s",
            DS9_ONNX / "yolo26s.onnx",
            DS9_ONNX / "yolo26s.onnx",
            DS9_ENGINES / "yolo26s_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "yolo26_m",
            DS9_ONNX / "yolo26m.onnx",
            DS9_ONNX / "yolo26m.onnx",
            DS9_ENGINES / "yolo26m_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "yolo26_l",
            DS9_ONNX / "yolo26l.onnx",
            DS9_ONNX / "yolo26l.onnx",
            DS9_ENGINES / "yolo26l_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "yolo26_x",
            DS9_ONNX / "yolo26x.onnx",
            DS9_ONNX / "yolo26x.onnx",
            DS9_ENGINES / "yolo26x_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "rfdetr_n",
            DS9_ONNX / "rfdetr_n_384.onnx",
            DS9_ONNX / "rfdetr_n_384.onnx",
            DS9_ENGINES / "rfdetr_n_384_b3_fp16.engine",
            ("--memPoolSize=workspace:4096",),
        ),
        EngineSpec(
            "rfdetr_s",
            DS9_ONNX / "rfdetr_s_512.onnx",
            DS9_ONNX / "rfdetr_s_512.onnx",
            DS9_ENGINES / "rfdetr_s_512_b3_fp16.engine",
            ("--memPoolSize=workspace:4096",),
        ),
        EngineSpec(
            "rfdetr_m",
            DS9_ONNX / "rfdetr_m_576.onnx",
            DS9_ONNX / "rfdetr_m_576.onnx",
            DS9_ENGINES / "rfdetr_m_576_b3_fp16.engine",
            ("--memPoolSize=workspace:4096",),
        ),
        EngineSpec(
            "rfdetr_seg_n",
            REPO_ROOT / "models" / "onnx" / "rfdetr_seg_n_312.onnx",
            DS9_ONNX / "rfdetr_seg_n_312.onnx",
            DS9_ENGINES / "rfdetr_seg_n_312_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "rfdetr_seg_s",
            REPO_ROOT / "models" / "onnx" / "rfdetr_seg_s_384.onnx",
            DS9_ONNX / "rfdetr_seg_s_384.onnx",
            DS9_ENGINES / "rfdetr_seg_s_384_b3_fp16.engine",
            (),
        ),
        EngineSpec(
            "rfdetr_seg_m",
            REPO_ROOT / "models" / "onnx" / "rfdetr_seg_m_432.onnx",
            DS9_ONNX / "rfdetr_seg_m_432.onnx",
            DS9_ENGINES / "rfdetr_seg_m_432_b3_fp16.engine",
            (),
        ),
    ]
    if include_mapanything:
        specs.append(
            EngineSpec(
                "mapanything",
                DS9_ONNX / "mapanything_images_294x518_b3.onnx",
                DS9_ONNX / "mapanything_images_294x518_b3.onnx",
                DS9_ENGINES / "mapanything_images_294x518_b3_fp16.plan",
                (
                    "--minShapes=images:3x3x294x518",
                    "--optShapes=images:3x3x294x518",
                    "--maxShapes=images:3x3x294x518",
                    "--builderOptimizationLevel=0",
                    "--maxAuxStreams=0",
                    "--memPoolSize=workspace:1024",
                ),
                "--bf16",
            )
        )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild Noesis TensorRT engines for DeepStream 9 / TensorRT 10.14")
    parser.add_argument("--only", default="", help="Comma-separated engine names to build.")
    parser.add_argument("--include-mapanything", action="store_true", help="Also build MapAnything from DS9/models/onnx/mapanything_images_294x518_b3.onnx.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    trtexec = shutil.which("trtexec")
    if not trtexec:
        raise SystemExit("trtexec not found on PATH")

    selected = {item.strip() for item in str(args.only or "").split(",") if item.strip()}
    specs = _specs(include_mapanything=bool(args.include_mapanything))
    for spec in specs:
        if selected and spec.name not in selected:
            continue
        _build(spec, trtexec=trtexec, dry_run=bool(args.dry_run))

    print("[OK] DS9 engine rebuild complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
