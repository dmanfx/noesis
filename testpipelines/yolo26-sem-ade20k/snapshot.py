"""Capture blended per-room semantic frames from DS8 buffers and metadata."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw
from pyservicemaker import BufferOperator

try:
    import torch
    import torch.utils.dlpack as torch_dlpack
except Exception:  # pragma: no cover - runtime dependency check
    torch = None  # type: ignore
    torch_dlpack = None  # type: ignore


_PALETTE_HEX = (
    "042AFF", "0BDBEB", "F3F3F3", "00DFB7", "111F68",
    "FF6FDD", "FF444F", "CCED00", "00F344", "BD00FF",
    "00B4FF", "DD00BA", "00FFFF", "26C000", "01FFB3",
    "7D24FF", "7B0068", "FF1B6C", "FC6D2F", "A2FF0B",
)
_PALETTE = np.asarray(
    [tuple(int(value[index:index + 2], 16) for index in (0, 2, 4)) for value in _PALETTE_HEX],
    dtype=np.uint8,
)


def _meta_value(meta: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(meta, Mapping) and name in meta:
            return meta[name]
        if hasattr(meta, name):
            return getattr(meta, name)
    return default


def _tensor_to_numpy(tensor: Any) -> np.ndarray | None:
    if isinstance(tensor, np.ndarray):
        return np.asarray(tensor)
    if torch is None or torch_dlpack is None or not callable(getattr(tensor, "__dlpack__", None)):
        return None
    try:
        stream = int(torch.cuda.current_stream().cuda_stream) if torch.cuda.is_available() else 0
        capsule = tensor.__dlpack__(stream)
        return torch_dlpack.from_dlpack(capsule).detach().cpu().numpy()
    except Exception:
        return None


def _frame_to_rgb(frame: np.ndarray) -> np.ndarray | None:
    array = np.asarray(frame)
    if array.ndim != 3:
        return None
    if array.shape[-1] >= 3:
        rgb = array[:, :, :3]
    elif array.shape[0] >= 3:
        rgb = np.transpose(array[:3, :, :], (1, 2, 0))
    else:
        return None
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _segmentation_item(frame_meta: Any) -> Any | None:
    for item in list(getattr(frame_meta, "segmentation_items", []) or []):
        candidate = item.as_segmentation() if hasattr(item, "as_segmentation") else item
        if hasattr(candidate, "class_map") and hasattr(candidate, "width"):
            return candidate
    return None


def _safe_name(value: str) -> str:
    text = "_".join(part for part in "".join(ch if ch.isalnum() else " " for ch in value.lower()).split())
    return text or "room"


class SemanticSnapshotEmitter(BufferOperator):
    def __init__(
        self,
        *,
        output_dir: Path,
        model_size: str,
        labels: list[str],
        sensor_names: list[str],
        warmup_frames: int = 8,
        alpha: float = 0.55,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_size = str(model_size)
        self.labels = list(labels)
        self.sensor_names = list(sensor_names)
        self.warmup_frames = max(1, int(warmup_frames))
        self.alpha = min(1.0, max(0.0, float(alpha)))
        self.complete_event = threading.Event()
        self._seen: dict[int, int] = {}
        self._captures: dict[int, dict[str, object]] = {}
        self.last_error: str | None = None

    def handle_buffer(self, buffer: Any) -> int:  # type: ignore[override]
        batch_meta = getattr(buffer, "batch_meta", None)
        frame_items = getattr(batch_meta, "frame_items", None) if batch_meta is not None else None
        if frame_items is None:
            return 1
        for batch_id, frame_meta in enumerate(frame_items):
            source_id = int(_meta_value(frame_meta, "source_id", "pad_index", default=batch_id) or 0)
            if source_id in self._captures:
                continue
            self._seen[source_id] = self._seen.get(source_id, 0) + 1
            if self._seen[source_id] < self.warmup_frames:
                continue
            try:
                segmentation = _segmentation_item(frame_meta)
                if segmentation is None:
                    continue
                tensor = buffer.extract(batch_id)
                frame = _tensor_to_numpy(tensor)
                rgb = _frame_to_rgb(frame) if frame is not None else None
                if rgb is None:
                    raise RuntimeError("DS8 RGB frame extraction returned an unsupported tensor")
                height = int(segmentation.height)
                width = int(segmentation.width)
                class_map = np.asarray(segmentation.class_map, dtype=np.int32).reshape(height, width)
                if class_map.size == 0 or int(class_map.min()) < 0 or int(class_map.max()) >= len(self.labels):
                    raise RuntimeError("Semantic class map is empty or outside ADE20K label bounds")
                self._capture(source_id, rgb, class_map)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                print(f"YOLO26_SEM_CAPTURE_ERROR source={source_id} error={self.last_error}", flush=True)
        if len(self._captures) == len(self.sensor_names):
            self.complete_event.set()
        return 1

    def _capture(self, source_id: int, rgb: np.ndarray, class_map: np.ndarray) -> None:
        room_name = self.sensor_names[source_id] if source_id < len(self.sensor_names) else f"Room {source_id}"
        height, width = class_map.shape
        raw = np.asarray(
            Image.fromarray(rgb, mode="RGB").resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        overlay = _PALETTE[class_map % len(_PALETTE)]
        blended = np.clip(
            raw.astype(np.float32) * (1.0 - self.alpha) + overlay.astype(np.float32) * self.alpha,
            0,
            255,
        ).astype(np.uint8)
        ids, counts = np.unique(class_map, return_counts=True)
        order = np.argsort(-counts)
        total = float(class_map.size)
        top_classes = [
            {
                "id": int(ids[index]),
                "name": self.labels[int(ids[index])],
                "fraction": round(float(counts[index]) / total, 4),
            }
            for index in order[:5]
        ]
        canvas = Image.new("RGB", (width, height + 58), (18, 18, 18))
        canvas.paste(Image.fromarray(blended, mode="RGB"), (0, 58))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), f"YOLO26{self.model_size}-sem-ADE20K | {room_name}", fill=(255, 255, 255))
        summary = " | ".join(f"{entry['name']} {100.0 * float(entry['fraction']):.0f}%" for entry in top_classes[:4])
        draw.text((10, 31), summary, fill=(220, 220, 220))
        stem = f"source_{source_id}_{_safe_name(room_name)}"
        raw_path = self.output_dir / f"{stem}_raw.jpg"
        masked_path = self.output_dir / f"{stem}_masked.jpg"
        map_path = self.output_dir / f"{stem}_class_map.png"
        Image.fromarray(raw, mode="RGB").save(raw_path, quality=94)
        canvas.save(masked_path, quality=94)
        Image.fromarray(class_map.astype(np.uint8), mode="L").save(map_path)
        self._captures[source_id] = {
            "source_id": source_id,
            "room": room_name,
            "raw_path": str(raw_path),
            "masked_path": str(masked_path),
            "class_map_path": str(map_path),
            "top_classes": top_classes,
        }
        print(f"YOLO26_SEM_CAPTURED size={self.model_size} source={source_id} room={room_name!r}", flush=True)

    def finalize(self) -> dict[str, object]:
        expected = set(range(len(self.sensor_names)))
        missing = sorted(expected - set(self._captures))
        if missing:
            detail = f"; last_error={self.last_error}" if self.last_error else ""
            raise RuntimeError(f"Missing semantic snapshots for source IDs {missing}{detail}")
        ordered = [self._captures[index] for index in sorted(self._captures)]
        images = [Image.open(str(item["masked_path"])).convert("RGB") for item in ordered]
        mosaic = Image.new("RGB", (sum(image.width for image in images), max(image.height for image in images)))
        x_offset = 0
        for image in images:
            mosaic.paste(image, (x_offset, 0))
            x_offset += image.width
        mosaic_path = self.output_dir / f"yolo26{self.model_size}_sem_ade20k_three_rooms_masked.jpg"
        mosaic.save(mosaic_path, quality=94)
        summary: dict[str, object] = {
            "model": f"yolo26{self.model_size}-sem-ade20k",
            "alpha": self.alpha,
            "mosaic_path": str(mosaic_path),
            "captures": ordered,
        }
        summary_path = self.output_dir / "capture_summary.json"
        summary["summary_path"] = str(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
