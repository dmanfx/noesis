#!/usr/bin/env python3
"""Download RF-DETR-Seg n/s/m weights, export ONNX, and build TensorRT engines."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    "n": {
        "model": "rfdetr-seg-nano",
        "weight_key": "rf-detr-seg-nano.pt",
        "resolution": 312,
        "max_detections": 10,
    },
    "s": {
        "model": "rfdetr-seg-small",
        "weight_key": "rf-detr-seg-small.pt",
        "resolution": 384,
        "max_detections": 20,
    },
    "m": {
        "model": "rfdetr-seg-medium",
        "weight_key": "rf-detr-seg-medium.pt",
        "resolution": 432,
        "max_detections": 30,
    },
}


def _run(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    print(f"[DL ] {url} -> {dst}")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dst)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sizes",
        default="n,s,m",
        help="Comma-separated RF-DETR sizes to build (n,s,m). Default: n,s,m",
    )
    p.add_argument("--batch", type=int, default=3, help="Static export batch size. Default: 3")
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip weight download and reuse local files.",
    )
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX export and reuse existing ONNX files.",
    )
    p.add_argument(
        "--skip-engine",
        action="store_true",
        help="Skip TensorRT engine build.",
    )
    p.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable ONNX simplification step in exporter.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if shutil.which("trtexec") is None and not args.skip_engine:
        raise SystemExit("trtexec not found in PATH")

    try:
        from rfdetr.main import HOSTED_MODELS
    except Exception as exc:
        raise SystemExit(f"Unable to import rfdetr HOSTED_MODELS: {exc}") from exc

    export_script = REPO_ROOT / "external" / "DeepStream-Yolo-Seg" / "utils" / "export_rfdetr_seg.py"
    if not export_script.exists():
        raise SystemExit(f"Missing exporter: {export_script}")
    trt_plugin_so = (
        REPO_ROOT
        / "external"
        / "DeepStream-Yolo-Seg"
        / "nvdsinfer_custom_impl_Yolo_seg"
        / "libnvdsinfer_custom_impl_Yolo_seg.so"
    )
    if not args.skip_engine and not trt_plugin_so.exists():
        raise SystemExit(f"Missing TensorRT plugin library required for ROIAlignX_TRT: {trt_plugin_so}")

    sizes = [s.strip().lower() for s in str(args.sizes or "").split(",") if s.strip()]
    if not sizes:
        raise SystemExit("No sizes selected")
    invalid = [s for s in sizes if s not in VARIANTS]
    if invalid:
        raise SystemExit(f"Unsupported size(s): {invalid}. Expected subset of n,s,m")

    weights_dir = REPO_ROOT / "models"
    onnx_dir = REPO_ROOT / "models" / "onnx"
    engines_dir = REPO_ROOT / "models" / "engines"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    engines_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[tuple[str, Path, Path, Path]] = []
    for size in sizes:
        spec = VARIANTS[size]
        weight_key = str(spec["weight_key"])
        resolution = int(spec["resolution"])
        max_detections = int(spec["max_detections"])
        model_name = str(spec["model"])

        url = HOSTED_MODELS.get(weight_key)
        if not url:
            raise SystemExit(f"RF-DETR hosted URL missing for key: {weight_key}")

        weight_path = weights_dir / f"rf-detr-seg-{size}.pt"
        onnx_path = onnx_dir / f"rfdetr_seg_{size}_{resolution}.onnx"
        engine_path = engines_dir / f"rfdetr_seg_{size}_{resolution}_b3_fp16.engine"

        if not args.skip_download:
            _download(url, weight_path)
        elif not weight_path.exists():
            raise SystemExit(f"Missing weights for --skip-download: {weight_path}")

        if not args.skip_export:
            export_cmd = [
                "python3",
                str(export_script),
                "-m",
                model_name,
                "-w",
                str(weight_path),
                "-o",
                str(onnx_path),
                "-n",
                "90",
                "-s",
                str(resolution),
                "--batch",
                str(int(args.batch)),
                "--max-detections",
                str(max_detections),
                "--opset",
                "17",
            ]
            if not args.no_simplify:
                export_cmd.append("--simplify")
            _run(
                export_cmd
            )
        elif not onnx_path.exists():
            raise SystemExit(f"Missing ONNX for --skip-export: {onnx_path}")

        if not args.skip_engine:
            _run(
                [
                    "trtexec",
                    f"--onnx={onnx_path}",
                    f"--saveEngine={engine_path}",
                    f"--dynamicPlugins={trt_plugin_so}",
                    "--fp16",
                    "--skipInference",
                ]
            )
        elif not engine_path.exists():
            raise SystemExit(f"Missing engine for --skip-engine: {engine_path}")

        outputs.append((size, weight_path, onnx_path, engine_path))

    print("\nBuilt RF-DETR assets:")
    for size, w, o, e in outputs:
        print(f"  size={size} weights={w} onnx={o} engine={e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
