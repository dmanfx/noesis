#!/usr/bin/env python3
"""Export pinned DA3-Small for an isolated, matched depth comparison.

The TensorRT surface matches the DA3Metric-Large manual-depth experiment:
FP32 RGB 0..255 input, ImageNet normalization in graph, dynamic batch, and
``depth/conf/mask`` outputs at 294x518.  Unlike DA3Metric, depth is relative,
``conf`` is the model's uncalibrated depth confidence, and ``mask`` is all ones.
This exporter does not install or select the engine for the live runtime.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from export_da3metric_large import _sha256_file, _tensor_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--fixture-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    import numpy as np
    import onnx
    import torch
    from torch import nn

    source = args.source_dir.resolve()
    checkpoint = args.checkpoint_dir.resolve()
    output = args.output_dir.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    sys.path.insert(0, str(source / "src"))
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.model.dinov2.layers.attention import Attention

    model = DepthAnything3.from_pretrained(
        str(checkpoint), local_files_only=True
    ).model.eval()
    for module in model.modules():
        if isinstance(module, Attention):
            module.fused_attn = False

    class Wrapper(nn.Module):
        def __init__(self, network: nn.Module) -> None:
            super().__init__()
            self.network = network
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, images):
            normalized = (images / 255.0 - self.mean) / self.std
            result = self.network(
                normalized.unsqueeze(1), export_feat_layers=[], infer_gs=False,
                use_ray_pose=False,
            )
            depth = result.depth
            return depth, result.depth_conf, torch.ones_like(depth)

    with np.load(args.fixture_npz, allow_pickle=False) as fixture:
        rgb = np.asarray(fixture["model_rgb"], dtype=np.uint8)
    if rgb.shape != (294, 518, 3):
        raise ValueError(f"unexpected fixture RGB shape: {rgb.shape}")
    sample_np = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
    wrapper = Wrapper(model).to(args.device).eval()
    sample = torch.from_numpy(sample_np).to(args.device)
    with torch.inference_mode():
        reference = tuple(
            value.float().cpu().numpy() for value in wrapper(sample)
        )

    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / "da3_small_294x518_b3.onnx"
    # The official RoPE position getter uses cartesian_prod for a fixed pixel
    # grid.  Its equivalent meshgrid form is supported by ONNX opset 17.
    original_cartesian_prod = torch.cartesian_prod

    def exportable_cartesian_prod(y, x):
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((yy, xx), dim=-1).reshape(-1, 2)

    try:
        torch.cartesian_prod = exportable_cartesian_prod
        torch.onnx.export(
            wrapper, (sample,), str(onnx_path), export_params=True,
            opset_version=17, do_constant_folding=True,
            input_names=["images"], output_names=["depth", "conf", "mask"],
            dynamic_axes={name: {0: "batch"} for name in ("images", "depth", "conf", "mask")},
            external_data=True, dynamo=False,
        )
    finally:
        torch.cartesian_prod = original_cartesian_prod
    onnx.checker.check_model(onnx.load(str(onnx_path), load_external_data=False))
    np.savez_compressed(output / "pytorch_reference.npz", depth=reference[0],
                        conf=reference[1], mask=reference[2])
    (output / "fixture_b1.raw").write_bytes(sample_np.tobytes())
    (output / "fixture_b3.raw").write_bytes(
        np.ascontiguousarray(np.repeat(sample_np, 3, axis=0)).tobytes()
    )
    receipt = {
        "model": "depth-anything/DA3-SMALL",
        "checkpoint_sha256": _sha256_file(checkpoint / "model.safetensors"),
        "config_sha256": _sha256_file(checkpoint / "config.json"),
        "source_revision": revision,
        "fixture": str(args.fixture_npz.resolve()),
        "onnx_sha256": _sha256_file(onnx_path),
        "input": "images FP32 RGB 0..255 Nx3x294x518; ImageNet normalization in graph",
        "outputs": "depth relative, conf uncalibrated model confidence, mask all ones",
        "reference": {name: _tensor_summary(value) for name, value in
                      zip(("depth", "conf", "mask"), reference, strict=True)},
    }
    (output / "export_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
