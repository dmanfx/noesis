#!/usr/bin/env python3
"""Utility script to fetch, export, and optimize Depth Anything V2 for DeepStream.

This automates the step-by-step plan described in the project documentation:

1. Clone Depth-Anything-V2 and install dependencies (optional toggle).
2. Download the metric ViT-L checkpoint.
3. Export the PyTorch model to ONNX with dynamic batch support.
4. Build a TensorRT engine targeting DeepStream 7.1 with FP16 precision.

The resulting engine is written to ``models/engines/depth_anything_v2_metric_vitl_fp16.engine``
which matches ``pipelines/config_infer_secondary_depth_anything_v2.ini``.

Usage example::

    python scripts/prepare_depth_anything_v2_engine.py --install-deps \
        --workspace models/depth_anything_v2

The script performs idempotent checks so re-running is safe; existing artifacts are reused
unless ``--force`` is provided.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import urllib.request

DEPTH_REPO_URL = "https://github.com/LiheYoung/Depth-Anything-V2.git"
WEIGHT_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Large/resolve/main/"
    "depth_anything_v2_metric_vitl.pth"
)
DEFAULT_ONNX_NAME = "depth_anything_v2_metric_vitl.onnx"
DEFAULT_ENGINE_NAME = "depth_anything_v2_metric_vitl_fp16.engine"


def _run(cmd: Iterable[str], cwd: Optional[Path] = None, check: bool = True) -> None:
    cmd_list = list(cmd)
    print(f"🚀 Running: {' '.join(cmd_list)}")
    subprocess.run(cmd_list, cwd=str(cwd) if cwd else None, check=check)


def _download(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"✅ Download already present: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Downloading {url} → {dest}")
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out_f:
        shutil.copyfileobj(response, out_f)
    print(f"✅ Downloaded {dest} ({dest.stat().st_size / (1024 * 1024):.1f} MB)")


def _maybe_clone_repo(repo_dir: Path, force: bool = False) -> None:
    if repo_dir.exists() and any(repo_dir.iterdir()) and not force:
        print(f"✅ Depth-Anything-V2 repo already present: {repo_dir}")
        return
    if repo_dir.exists() and force:
        print(f"♻️  Removing existing repo at {repo_dir}")
        shutil.rmtree(repo_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", DEPTH_REPO_URL, str(repo_dir)])


def _maybe_install_requirements(repo_dir: Path) -> None:
    req_file = repo_dir / "requirements.txt"
    if not req_file.exists():
        print(f"⚠️  requirements.txt not found in {repo_dir}; skipping pip install")
        return
    print(f"📦 Installing Depth-Anything-V2 requirements from {req_file}")
    _run([sys.executable, "-m", "pip", "install", "-r", str(req_file)])


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fd:
        for chunk in iter(lambda: fd.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_to_onnx(repo_dir: Path, weights_path: Path, onnx_path: Path, force: bool = False) -> None:
    if onnx_path.exists() and not force:
        print(f"✅ ONNX already exists: {onnx_path}")
        return

    sys.path.insert(0, str(repo_dir))
    print("🧠 Loading Depth Anything V2 model for ONNX export…")
    import torch

    from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

    checkpoint = torch.load(weights_path, map_location="cpu")
    model = DepthAnythingV2(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 768, 768],
        metric=True,
    )
    model.load_state_dict(checkpoint)
    model.eval()

    dummy = torch.randn(1, 3, 518, 518)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["pred"],
        dynamic_axes={"input": {0: "batch"}, "pred": {0: "batch"}},
    )
    print(f"✅ Exported ONNX to {onnx_path}")


def build_tensorrt_engine(onnx_path: Path, engine_path: Path, fp16: bool = True, force: bool = False) -> None:
    if engine_path.exists() and not force:
        print(f"✅ TensorRT engine already exists: {engine_path}")
        return

    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError("trtexec not found in PATH. Ensure TensorRT / DeepStream 7.1 is installed.")

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--workspace=4096",
        "--minShapes=input:1x3x224x224",
        "--optShapes=input:1x3x518x518",
        "--maxShapes=input:1x3x1024x1024",
        "--verbose",
    ]
    if fp16:
        cmd.append("--fp16")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    _run(cmd)
    print(f"✅ TensorRT engine created at {engine_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Depth Anything V2 TensorRT engine for DeepStream")
    parser.add_argument("--workspace", type=Path, default=Path("models/depth_anything_v2"), help="Workspace directory")
    parser.add_argument("--install-deps", action="store_true", help="Install Depth-Anything-V2 pip requirements")
    parser.add_argument("--force", action="store_true", help="Force regeneration of all artifacts")
    parser.add_argument("--skip-trtexec", action="store_true", help="Skip TensorRT engine build step")
    parser.add_argument("--engine-path", type=Path, default=Path("models/engines") / DEFAULT_ENGINE_NAME,
                        help="Output path for TensorRT engine")
    parser.add_argument("--onnx-path", type=Path, default=None, help="Optional override for ONNX output path")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    repo_dir = workspace / "Depth-Anything-V2"
    checkpoints_dir = workspace / "checkpoints"

    _maybe_clone_repo(repo_dir, force=args.force)
    if args.install_deps:
        _maybe_install_requirements(repo_dir)

    weights_path = checkpoints_dir / "depth_anything_v2_metric_vitl.pth"
    _download(WEIGHT_URL, weights_path, force=args.force)
    print(f"📦 Checkpoint SHA256: {_hash_file(weights_path)}")

    onnx_path = args.onnx_path.resolve() if args.onnx_path else workspace / DEFAULT_ONNX_NAME
    export_to_onnx(repo_dir, weights_path, onnx_path, force=args.force)

    if args.skip_trtexec:
        print("⏭️  Skipping TensorRT engine build as requested")
        return

    engine_path = args.engine_path.resolve()
    build_tensorrt_engine(onnx_path, engine_path, fp16=True, force=args.force)
    print("🎉 Depth Anything V2 TensorRT preparation complete")


if __name__ == "__main__":
    main()
