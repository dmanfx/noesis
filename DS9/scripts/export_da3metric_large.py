#!/usr/bin/env python3
"""Export the official DA3Metric-Large checkpoint for Noesis DS9.

The exported network keeps the official DA3 metric-depth head and sky head,
adds the official ImageNet normalization inside the graph, and exposes the
existing Noesis manual-depth tensor surface: ``depth``, ``conf``, and ``mask``.
DA3Metric-Large does not predict confidence.  ``conf`` and ``mask`` therefore
carry the official non-sky decision (sky score < 0.3); metric scaling remains a
camera-calibration postprocess in the runtime as specified by the DA3 authors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Sequence


OFFICIAL_SOURCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
OFFICIAL_MODEL_REVISION = "4010e39f3634a45bc60553321fb49fb760bd594e"
OFFICIAL_MODEL_ID = "depth-anything/DA3METRIC-LARGE"
OFFICIAL_WEIGHT_SHA256 = "bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776"
OFFICIAL_CONFIG_SHA256 = "a336f3e76fe375aaae17a9aed9130c9f2aa061535d317ec57dcb2f1f02e1dd53"
INPUT_HEIGHT = 294
INPUT_WIDTH = 518
BATCH_SIZE = 3
PATCH_SIZE = 14
METRIC_FOCAL_DENOMINATOR = 300.0
SKY_THRESHOLD = 0.3
OUTPUT_NAMES = ("depth", "conf", "mask")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_official_source(source_dir: Path) -> None:
    if not (source_dir / ".git").is_dir():
        raise RuntimeError(f"DA3 source is not a Git checkout: {source_dir}")
    revision = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != OFFICIAL_SOURCE_REVISION:
        raise RuntimeError(
            "official DA3 source revision mismatch: "
            f"expected={OFFICIAL_SOURCE_REVISION} actual={revision}"
        )


def _require_official_checkpoint(checkpoint_dir: Path) -> None:
    expected = {
        "model.safetensors": OFFICIAL_WEIGHT_SHA256,
        "config.json": OFFICIAL_CONFIG_SHA256,
    }
    for name, digest in expected.items():
        path = checkpoint_dir / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"official DA3 checkpoint file is missing: {path}")
        observed = _sha256_file(path)
        if observed != digest:
            raise RuntimeError(
                f"official DA3 checkpoint digest mismatch for {name}: "
                f"expected={digest} actual={observed}"
            )


def _install_source_import(source_dir: Path) -> None:
    source_python = str((source_dir / "src").resolve())
    while source_python in sys.path:
        sys.path.remove(source_python)
    sys.path.insert(0, source_python)


def _load_export_wrapper(source_dir: Path, checkpoint_dir: Path):
    _install_source_import(source_dir)

    import torch
    from torch import nn
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.model.dinov2.layers.attention import Attention

    api = DepthAnything3.from_pretrained(
        str(checkpoint_dir),
        local_files_only=True,
    )
    network = api.model.eval()

    # The official convenience forward replaces sky depth with a sampled p99.
    # Keep raw head outputs in the engine and export the official deterministic
    # sky threshold instead; this avoids random/quantile graph operators.
    def _preserve_raw_outputs(_self, output):
        return output

    network._process_mono_sky_estimation = types.MethodType(
        _preserve_raw_outputs,
        network,
    )

    # Express attention as explicit matmul/softmax so ONNX and TensorRT own one
    # deterministic, portable graph rather than an exporter-specific SDPA op.
    attention_modules = 0
    for module in network.modules():
        if isinstance(module, Attention):
            module.fused_attn = False
            attention_modules += 1
    if attention_modules != 24:
        raise RuntimeError(
            "DA3Metric-Large attention topology drifted: "
            f"expected=24 actual={attention_modules}"
        )

    class DA3MetricLargeNoesisWrapper(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model
            self.register_buffer(
                "image_mean",
                torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
                    1, 3, 1, 1
                ),
            )
            self.register_buffer(
                "image_std",
                torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
                    1, 3, 1, 1
                ),
            )

        def forward(self, images):
            normalized = (images / 255.0 - self.image_mean) / self.image_std
            output = self.model(
                normalized.unsqueeze(1),
                export_feat_layers=[],
                infer_gs=False,
                use_ray_pose=False,
            )
            depth = output.depth
            non_sky = (output.sky < SKY_THRESHOLD).to(depth.dtype)
            return depth, non_sky, non_sky

    return DA3MetricLargeNoesisWrapper(network).eval()


def _prepare_fixture(image_paths: Sequence[Path]):
    import cv2
    import numpy as np

    if len(image_paths) not in {1, BATCH_SIZE}:
        raise ValueError(
            f"fixture requires one image or exactly {BATCH_SIZE} images"
        )
    tensors: list[Any] = []
    sources: list[dict[str, Any]] = []
    for path in image_paths:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"fixture image is not a regular file: {path}")
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"unable to decode RGB fixture image: {path}")
        source_height, source_width = bgr.shape[:2]
        scale = min(INPUT_WIDTH / source_width, INPUT_HEIGHT / source_height)
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        resized = cv2.resize(
            bgr,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LANCZOS4,
        )
        canvas = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
        left = (INPUT_WIDTH - resized_width) // 2
        top = (INPUT_HEIGHT - resized_height) // 2
        canvas[top : top + resized_height, left : left + resized_width] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensors.append(
            np.ascontiguousarray(
                np.transpose(rgb.astype(np.float32), (2, 0, 1)),
                dtype=np.dtype("<f4"),
            )
        )
        sources.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "source_size": [int(source_height), int(source_width)],
                "resized_size": [int(resized_height), int(resized_width)],
                "pad_top": int(top),
                "pad_left": int(left),
            }
        )
    if len(tensors) == 1:
        tensors *= BATCH_SIZE
    batch = np.ascontiguousarray(np.stack(tensors), dtype=np.dtype("<f4"))
    expected = (BATCH_SIZE, 3, INPUT_HEIGHT, INPUT_WIDTH)
    if batch.shape != expected:
        raise RuntimeError(
            f"fixture shape mismatch: expected={expected} actual={batch.shape}"
        )
    return batch, sources


def _tensor_summary(value) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(value, dtype=np.float32)
    finite = np.isfinite(array)
    finite_values = array[finite]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_count": int(np.count_nonzero(finite)),
        "positive_count": int(np.count_nonzero(finite & (array > 0.0))),
        "min": float(finite_values.min()) if finite_values.size else None,
        "mean": float(finite_values.mean()) if finite_values.size else None,
        "median": float(np.median(finite_values)) if finite_values.size else None,
        "p99": float(np.quantile(finite_values, 0.99)) if finite_values.size else None,
        "max": float(finite_values.max()) if finite_values.size else None,
    }


def export(args: argparse.Namespace) -> None:
    import numpy as np
    import onnx
    import torch

    source_dir = args.source_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    onnx_path = args.onnx.expanduser().resolve()
    fixture_path = args.fixture.expanduser().resolve()
    reference_path = args.reference.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()

    _require_official_source(source_dir)
    _require_official_checkpoint(checkpoint_dir)
    fixture, fixture_sources = _prepare_fixture(args.fixture_image)
    wrapper = _load_export_wrapper(source_dir, checkpoint_dir)

    device = torch.device(args.device)
    wrapper = wrapper.to(device)
    sample = torch.from_numpy(fixture[:1]).to(device)
    with torch.inference_mode():
        reference_tensors = tuple(
            value.detach().float().cpu().numpy()
            for value in wrapper(torch.from_numpy(fixture).to(device))
        )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (sample,),
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=list(OUTPUT_NAMES),
        dynamic_axes={
            "images": {0: "batch"},
            "depth": {0: "batch"},
            "conf": {0: "batch"},
            "mask": {0: "batch"},
        },
        external_data=True,
        dynamo=False,
    )

    model = onnx.load(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(model)
    graph_inputs = [item.name for item in model.graph.input]
    graph_outputs = [item.name for item in model.graph.output]
    if graph_inputs != ["images"] or graph_outputs != list(OUTPUT_NAMES):
        raise RuntimeError(
            "exported DA3 ONNX binding mismatch: "
            f"inputs={graph_inputs} outputs={graph_outputs}"
        )

    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_bytes(fixture.tobytes(order="C"))
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        reference_path,
        **dict(zip(OUTPUT_NAMES, reference_tensors, strict=True)),
    )

    receipt = {
        "contract": "noesis.da3metric_large.export.v1",
        "official_source": {
            "repository": "https://github.com/ByteDance-Seed/Depth-Anything-3.git",
            "revision": OFFICIAL_SOURCE_REVISION,
        },
        "official_checkpoint": {
            "model_id": OFFICIAL_MODEL_ID,
            "revision": OFFICIAL_MODEL_REVISION,
            "weight_sha256": OFFICIAL_WEIGHT_SHA256,
            "config_sha256": OFFICIAL_CONFIG_SHA256,
        },
        "tensor_contract": {
            "input": {
                "name": "images",
                "dtype": "float32",
                "shape": ["batch", 3, INPUT_HEIGHT, INPUT_WIDTH],
                "range": [0.0, 255.0],
                "color": "RGB",
                "normalization": "official_imagenet_in_graph",
            },
            "outputs": [
                {
                    "name": name,
                    "dtype": "float32",
                    "shape": ["batch", 1, INPUT_HEIGHT, INPUT_WIDTH],
                }
                for name in OUTPUT_NAMES
            ],
            "confidence_semantics": "official_non_sky_binary_validity_not_model_confidence",
            "sky_threshold": SKY_THRESHOLD,
            "metric_scale": "depth_m = raw_depth * model_input_focal_px / 300",
            "metric_focal_denominator": METRIC_FOCAL_DENOMINATOR,
        },
        "onnx": {
            "path": str(onnx_path),
            "size_bytes": int(onnx_path.stat().st_size),
            "sha256": _sha256_file(onnx_path),
            "opset": 17,
            "node_count": len(model.graph.node),
            "initializer_count": len(model.graph.initializer),
        },
        "fixture": {
            "path": str(fixture_path),
            "size_bytes": int(fixture_path.stat().st_size),
            "sha256": _sha256_file(fixture_path),
            "shape": list(fixture.shape),
            "sources": fixture_sources,
        },
        "pytorch_reference": {
            "path": str(reference_path),
            "sha256": _sha256_file(reference_path),
            "outputs": {
                name: _tensor_summary(value)
                for name, value in zip(OUTPUT_NAMES, reference_tensors, strict=True)
            },
        },
    }
    _write_json(receipt_path, receipt)
    os.chmod(onnx_path, 0o600)
    os.chmod(fixture_path, 0o600)
    os.chmod(reference_path, 0o600)
    os.chmod(receipt_path, 0o600)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export official DA3Metric-Large for the Noesis manual-depth lane"
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--fixture-image",
        type=Path,
        action="append",
        required=True,
        help="One image repeated across batch three, or exactly three images.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    export(_parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
