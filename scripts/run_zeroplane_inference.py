#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _prepare_zeroplane_paths(zeroplane_repo: Path) -> None:
    repo = Path(zeroplane_repo).resolve()
    # ZeroPlane's DUSt3R dependency imports a top-level `models` namespace from
    # croco. The Noesis repo also has models.py, so remove repo-root entries
    # before adding ZeroPlane third-party paths.
    sys.path[:] = [p for p in sys.path if p not in ("", str(REPO_ROOT))]
    sys.path[:0] = [
        str(repo / "third_party" / "dust3r" / "croco"),
        str(repo / "third_party" / "dust3r"),
        str(repo),
        str(repo / "ZeroPlane" / "modeling" / "pixel_decoder" / "ops"),
        str(repo / "demo"),
    ]


def _get_coordinate_map(k: np.ndarray, *, device: torch.device, h: int, w: int, original_h: int, original_w: int) -> torch.Tensor:
    k_inv = torch.as_tensor(np.linalg.inv(k), dtype=torch.float32, device=device)
    x = torch.arange(w, dtype=torch.float32, device=device).view(1, w) / float(w) * float(original_w)
    y = torch.arange(h, dtype=torch.float32, device=device).view(h, 1) / float(h) * float(original_h)
    xx = x.repeat(h, 1)
    yy = y.repeat(1, w)
    xy1 = torch.stack((xx, yy, torch.ones((h, w), dtype=torch.float32, device=device))).view(3, -1)
    return torch.matmul(k_inv, xy1)


def _setup_cfg(args: argparse.Namespace):
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from ZeroPlane import add_ZeroPlane_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ZeroPlane_config(cfg)
    cfg.merge_from_file(str(args.config_file))
    opts = [
        "MODEL.WEIGHTS",
        str(args.checkpoint),
        "MODEL.DEVICE",
        str(args.device),
    ]
    opts.extend(args.opts or [])
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


def run(args: argparse.Namespace) -> None:
    zeroplane_repo = Path(args.zeroplane_repo).resolve()
    _prepare_zeroplane_paths(zeroplane_repo)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("ZeroPlane requested CUDA but torch.cuda.is_available() is false")
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"ZeroPlane checkpoint is missing: {args.checkpoint}")

    from detectron2.data.detection_utils import read_image
    from predictor import VisualizationDemo

    cfg = _setup_cfg(args)
    demo = VisualizationDemo(cfg)
    normal_anchor = zeroplane_repo / "cluster_anchor" / "new_mixed_normal_anchors_7.npy"
    offset_anchor = zeroplane_repo / "cluster_anchor" / "new_mixed_offset_anchors_20.npy"
    if not normal_anchor.exists() or not offset_anchor.exists():
        raise SystemExit(f"ZeroPlane anchor files are missing under {zeroplane_repo / 'cluster_anchor'}")

    device = torch.device(cfg.MODEL.DEVICE)
    anchors = {
        "anchor_normals": torch.tensor(np.load(normal_anchor), dtype=torch.float32, device=device),
        "anchor_offsets": torch.tensor(np.load(offset_anchor), dtype=torch.float32, device=device),
    }
    k = np.asarray([[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    image = read_image(str(args.image), format="RGB")
    resized = cv2.resize(image, (int(args.resize_w), int(args.resize_h)), interpolation=cv2.INTER_LINEAR)
    k_inv_dot_xy1 = _get_coordinate_map(
        k,
        device=device,
        h=int(args.resize_h),
        w=int(args.resize_w),
        original_h=int(args.original_h),
        original_w=int(args.original_w),
    )
    predictions = demo.run_on_image(resized, anchors, k_inv_dot_xy1)
    sem_seg = predictions["sem_seg"].detach().float().cpu()
    segmentation = sem_seg.argmax(dim=0).numpy().astype(np.int32)
    planar_depth = predictions["planes_depth"].detach().float().cpu().numpy().astype(np.float32)
    params = predictions.get("valid_params")
    params_np = params.detach().float().cpu().numpy().astype(np.float32) if params is not None else np.zeros((0, 3), dtype=np.float32)
    valid_count = int(params_np.shape[0])
    masks = np.zeros((valid_count, segmentation.shape[0], segmentation.shape[1]), dtype=bool)
    confidence = np.zeros((valid_count,), dtype=np.float32)
    sem_np = sem_seg.numpy()
    for idx in range(valid_count):
        mask = (segmentation == idx) & np.isfinite(planar_depth) & (planar_depth > 0.0)
        masks[idx] = mask
        confidence[idx] = float(np.mean(sem_np[idx][mask])) if np.any(mask) else 0.0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frame_id=np.asarray(str(args.frame_id)),
        masks=masks,
        segmentation=segmentation,
        planar_depth=planar_depth,
        params=params_np,
        confidence=confidence,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ZeroPlane on one frame and emit Noesis planar inference output")
    parser.add_argument("--zeroplane-repo", type=Path, default=REPO_ROOT / "external" / "ZeroPlane")
    parser.add_argument("--config-file", type=Path, default=REPO_ROOT / "external" / "ZeroPlane" / "configs" / "ZeroPlaneNYUV2" / "dust3r_large_dpt_bs16_50ep.yaml")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "external" / "ZeroPlane" / "checkpoints" / "dust3r_encoder_released.pth")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--original-w", type=int, required=True)
    parser.add_argument("--original-h", type=int, required=True)
    parser.add_argument("--resize-w", type=int, default=256)
    parser.add_argument("--resize-h", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
