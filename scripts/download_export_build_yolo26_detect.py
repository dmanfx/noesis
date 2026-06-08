#!/usr/bin/env python3
"""Download/export YOLO26 detector ONNX files and build TensorRT engines."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
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
    invalid = [item for item in sizes if item not in {"n", "s", "m", "l", "x"}]
    if invalid:
        raise SystemExit(f"Unsupported size(s): {invalid}. Expected subset of n,s,m,l,x")
    return sizes


def _export_onnx(size: str, *, batch: int, opset: int, simplify: bool, device: str) -> Path:
    from ultralytics import YOLO

    weights = REPO_ROOT / "models" / f"yolo26{size}.pt"
    if not weights.exists():
        YOLO(f"yolo26{size}.pt")
        downloaded = REPO_ROOT / f"yolo26{size}.pt"
        if downloaded.exists():
            downloaded.replace(weights)
    if not weights.exists():
        raise FileNotFoundError(f"YOLO26 detector weights missing: {weights}")

    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=640,
        batch=int(batch),
        dynamic=False,
        simplify=bool(simplify),
        opset=int(opset),
        nms=False,
        device=device,
    )
    exported_path = Path(exported).resolve()
    target = (REPO_ROOT / "models" / f"yolo26{size}.onnx").resolve()
    if exported_path != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        exported_path.replace(target)
    return target


def _build_engine(onnx_path: Path, engine_path: Path) -> Path:
    if engine_path.exists() and engine_path.stat().st_size > 0:
        return engine_path
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    if engine_path.exists():
        engine_path.unlink()
    _run(
        [
            _trtexec(),
            f"--onnx={onnx_path}",
            "--fp16",
            "--memPoolSize=workspace:4096",
            f"--saveEngine={engine_path}",
            "--skipInference",
        ],
        clean_library_path=True,
    )
    if not engine_path.exists() or engine_path.stat().st_size <= 0:
        raise RuntimeError(f"TensorRT engine was not created: {engine_path}")
    return engine_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="n,s,m", help="Comma-separated YOLO26 sizes to build. Supported: n,s,m,l,x. Default: n,s,m")
    parser.add_argument("--batch", type=int, default=3, help="Static export batch size. Default: 3")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset. Default: 18")
    parser.add_argument("--device", default="cpu", help="Ultralytics export device. Default: cpu")
    parser.add_argument("--simplify", action="store_true", help="Enable ONNX simplification during export.")
    parser.add_argument("--skip-export", action="store_true", help="Reuse existing ONNX files.")
    parser.add_argument("--skip-engine", action="store_true", help="Do not build TensorRT engines.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sizes = _sizes(args.sizes)
    outputs: list[tuple[str, Path, Path]] = []
    for size in sizes:
        onnx_path = (REPO_ROOT / "models" / f"yolo26{size}.onnx").resolve()
        if not args.skip_export:
            onnx_path = _export_onnx(
                size,
                batch=int(args.batch),
                opset=int(args.opset),
                simplify=bool(args.simplify),
                device=str(args.device),
            )
        elif not onnx_path.exists():
            raise FileNotFoundError(f"Missing ONNX for --skip-export: {onnx_path}")

        engine_path = (REPO_ROOT / "models" / "engines" / f"yolo26{size}_b{int(args.batch)}_fp16.engine").resolve()
        if not args.skip_engine:
            engine_path = _build_engine(onnx_path, engine_path)
        elif not engine_path.exists():
            raise FileNotFoundError(f"Missing engine for --skip-engine: {engine_path}")
        outputs.append((size, onnx_path, engine_path))

    print("\nBuilt YOLO26 detector assets:")
    for size, onnx_path, engine_path in outputs:
        print(f"  size={size} onnx={onnx_path} engine={engine_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
