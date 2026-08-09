"""Run one saved image through an exact batch-3 YOLO26 semantic engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tensorrt as trt
import torch


PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_ROOT.parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import model_setup
from snapshot import _PALETTE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO26 ADE20K TensorRT still-image inference")
    parser.add_argument("--size", choices=model_setup.SUPPORTED_SIZES, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.55)
    return parser.parse_args()


def _load_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 150:
        raise RuntimeError(f"Expected 150 ADE20K labels in {path}, got {len(labels)}")
    return labels


def _preprocess(image: Image.Image) -> torch.Tensor:
    resized = image.convert("RGB").resize((640, 640), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    chw = np.ascontiguousarray(np.transpose(array, (2, 0, 1)))
    batch = np.repeat(chw[np.newaxis, ...], 3, axis=0)
    return torch.from_numpy(batch).to(device="cuda", dtype=torch.float32).contiguous()


def _infer(engine_path: Path, input_tensor: torch.Tensor) -> np.ndarray:
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"TensorRT could not deserialize {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"TensorRT could not create an execution context for {engine_path}")
    output_tensor = torch.empty((3, 640, 640), dtype=torch.uint8, device="cuda")
    context.set_tensor_address("images", int(input_tensor.data_ptr()))
    context.set_tensor_address("output0", int(output_tensor.data_ptr()))
    stream = torch.cuda.current_stream()
    if not context.execute_async_v3(stream_handle=int(stream.cuda_stream)):
        raise RuntimeError(f"TensorRT execution failed for {engine_path}")
    stream.synchronize()
    output = output_tensor.cpu().numpy()
    if not np.array_equal(output[0], output[1]) or not np.array_equal(output[0], output[2]):
        raise RuntimeError("Duplicated still-image batch produced inconsistent semantic maps")
    return output[0]


def _render(
    image: Image.Image,
    class_map: np.ndarray,
    *,
    labels: list[str],
    model_size: str,
    alpha: float,
) -> tuple[Image.Image, list[dict[str, object]]]:
    if class_map.shape != (640, 640):
        raise RuntimeError(f"Expected a 640x640 class map, got {class_map.shape}")
    if class_map.size == 0 or int(class_map.max()) >= len(labels):
        raise RuntimeError("Semantic class map is empty or outside ADE20K label bounds")
    source = image.convert("RGB")
    expanded_map = np.asarray(
        Image.fromarray(class_map, mode="L").resize(source.size, Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    source_array = np.asarray(source, dtype=np.uint8)
    overlay = _PALETTE[expanded_map % len(_PALETTE)]
    blended = np.clip(
        source_array.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    ids, counts = np.unique(class_map, return_counts=True)
    order = np.argsort(-counts)
    total = float(class_map.size)
    top_classes = [
        {
            "id": int(ids[index]),
            "name": labels[int(ids[index])],
            "fraction": round(float(counts[index]) / total, 4),
        }
        for index in order[:8]
    ]
    header_height = 72
    canvas = Image.new("RGB", (source.width, source.height + header_height), (18, 18, 18))
    canvas.paste(Image.fromarray(blended, mode="RGB"), (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 10),
        f"YOLO26{model_size}-sem-ADE20K | Living Room | saved well-lit static anchor",
        fill=(255, 255, 255),
    )
    summary = " | ".join(
        f"{entry['name']} {100.0 * float(entry['fraction']):.0f}%" for entry in top_classes[:6]
    )
    draw.text((12, 39), summary, fill=(220, 220, 220))
    return canvas, top_classes


def main() -> None:
    args = _parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Saved static-camera image not found: {args.image}")
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between zero and one")
    assets = model_setup.resolve_assets(args.size)
    labels = _load_labels(assets.labels)
    image = Image.open(args.image).convert("RGB")
    input_tensor = _preprocess(image)
    class_map = _infer(assets.engine, input_tensor)
    masked, top_classes = _render(
        image,
        class_map,
        labels=labels,
        model_size=args.size,
        alpha=float(args.alpha),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    masked_path = args.output_dir / f"yolo26{args.size}_living_room_well_lit_masked.jpg"
    class_map_path = args.output_dir / f"yolo26{args.size}_living_room_well_lit_class_map.png"
    summary_path = args.output_dir / f"yolo26{args.size}_living_room_well_lit_summary.json"
    masked.save(masked_path, quality=94)
    Image.fromarray(class_map, mode="L").save(class_map_path)
    summary = {
        "model": f"yolo26{args.size}-sem-ade20k",
        "source_image": str(args.image),
        "source_size": [image.width, image.height],
        "engine": str(assets.engine),
        "batch_contract": [3, 3, 640, 640],
        "batch_fill": "same verified Living Room frame repeated three times",
        "masked_path": str(masked_path),
        "class_map_path": str(class_map_path),
        "top_classes": top_classes,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    print("YOLO26_SEM_STILL_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
