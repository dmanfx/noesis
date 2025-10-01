#!/usr/bin/env python3
"""
Download and cache OSNet ReID weights via torchreid, then save to ./models/.

- Uses config defaults (REID_MODEL_NAME, REID_IMAGE_SIZE) if no args provided.
- Requires torchreid to be installed in the active Python environment.

Usage examples:
  python3 scripts/download_reid_weights.py
  python3 scripts/download_reid_weights.py --model osnet_ibn_x1_0 --height 256 --width 128 --out models/osnet_ibn_x1_0_msmt17.pth
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="torchreid model name, e.g. osnet_ibn_x1_0")
    parser.add_argument("--height", type=int, default=None, help="input height (default from config or 256)")
    parser.add_argument("--width", type=int, default=None, help="input width (default from config or 128)")
    parser.add_argument("--device", default="cuda:0", help="device for model init (cuda:0 or cpu)")
    parser.add_argument("--out", default=None, help="output .pth path in ./models/")
    args = parser.parse_args()

    # Resolve defaults from config if available
    model_name = args.model
    height = args.height
    width = args.width
    try:
        from config import AppConfig
        cfg = AppConfig.ModelsSettings()
        if model_name is None:
            model_name = getattr(cfg, 'REID_MODEL_NAME', 'osnet_ibn_x1_0')
        if height is None or width is None:
            h, w = getattr(cfg, 'REID_IMAGE_SIZE', [256, 128])
            height = height or int(h)
            width = width or int(w)
    except Exception:
        model_name = model_name or 'osnet_ibn_x1_0'
        height = height or 256
        width = width or 128

    try:
        import torch
        from torchreid.utils import FeatureExtractor  # type: ignore
    except Exception as e:
        print("ERROR: torchreid is not available in this environment:", e, file=sys.stderr)
        sys.exit(2)

    print(f"Initializing FeatureExtractor: model={model_name}, size=({height},{width}), device={args.device}")
    extractor = FeatureExtractor(
        model_name=model_name,
        device=args.device,
        image_size=(height, width),
        pixel_norm=True,
    )
    # Grab state dict and save to models/
    os.makedirs('models', exist_ok=True)
    out_path = args.out or os.path.join('models', f"{model_name}_pretrained.pth")
    try:
        torch.save(extractor.model.state_dict(), out_path)
        print(f"Saved weights to {out_path}")
    except Exception as e:
        print("ERROR: Failed to save weights:", e, file=sys.stderr)
        sys.exit(3)

    print("Done.")


if __name__ == "__main__":
    main()

