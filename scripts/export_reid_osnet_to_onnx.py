#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch


class ReidExportWrapper(torch.nn.Module):
    """OSNet wrapper that includes ImageNet normalization + L2-normalized output."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        *,
        mean_rgb: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std_rgb: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(mean_rgb, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std_rgb, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # Expected input: RGB float tensor in [0,255] (DeepStream nvinfer default net-scale-factor=1, offsets=0).
        x = x / 255.0
        x = (x - self.mean) / self.std
        feats = self.backbone(x)
        feats = torch.nn.functional.normalize(feats, p=2.0, dim=1)
        return feats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export OSNet ReID model to ONNX (dynamic batch)")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/osnet_ibn_x1_0_msmt17.pt"),
        help="Path to a torchreid-compatible OSNet state_dict file.",
    )
    parser.add_argument(
        "--model-name",
        default="osnet_ibn_x1_0",
        help="torchreid model name (e.g., osnet_x1_0, osnet_ibn_x1_0).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Input height (default: 256).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Input width (default: 128).",
    )
    parser.add_argument(
        "--batch-max",
        type=int,
        default=16,
        help="Max batch size for dynamic export (default: 16).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/engines/reid_osnet_ibn_msmt17_dyn_b16.onnx"),
        help="Output ONNX path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    weights_path = Path(args.weights).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    try:
        from torchreid import models  # type: ignore
    except Exception as exc:
        raise RuntimeError("torchreid is required for OSNet export") from exc

    backbone = models.build_model(
        str(args.model_name),
        num_classes=1000,
        loss="softmax",
        pretrained=False,
    )
    state = torch.load(weights_path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"weights file must be a state_dict dict; got {type(state)!r}")
    state = {k.replace("module.", ""): v for k, v in state.items() if not str(k).startswith("classifier.")}
    missing = backbone.load_state_dict(state, strict=False)
    if getattr(missing, "unexpected_keys", None):
        raise RuntimeError(f"unexpected keys while loading weights: {missing.unexpected_keys}")

    model = ReidExportWrapper(backbone).eval()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h = int(args.height)
    w = int(args.width)
    if h <= 0 or w <= 0:
        raise ValueError("height/width must be positive")

    dummy = torch.zeros((1, 3, h, w), dtype=torch.float32)

    dynamic_axes = {
        "input": {0: "batch"},
        "features": {0: "batch"},
    }

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["features"],
        opset_version=int(args.opset),
        dynamo=False,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    try:
        import onnx

        m = onnx.load(str(out_path))
        onnx.checker.check_model(m)
    except Exception:
        # Keep export usable even when onnx isn't installed.
        pass

    print(f"✅ Exported ONNX: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
