"""Tensor-meta pose overlay for YOLO26 pose outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import logging
import math
import os
import time

import numpy as np
import torch
import torch.utils.dlpack as torch_dlpack
from pyservicemaker import BatchMetadataOperator, osd

logger = logging.getLogger(__name__)


def _to_numpy(tensor: Any) -> np.ndarray:
    dlpack_fn = getattr(tensor, "__dlpack__", None)
    if not callable(dlpack_fn):
        raise TypeError("Tensor output does not expose __dlpack__")
    stream = 0
    if torch.cuda.is_available():
        stream = int(torch.cuda.current_stream().cuda_stream)
    capsule = dlpack_fn(stream)
    torch_tensor = torch_dlpack.from_dlpack(capsule)
    return torch_tensor.detach().cpu().numpy()


class PoseOverlay(BatchMetadataOperator):
    def __init__(
        self,
        *,
        gie_id: int = 1,
        model_size: Tuple[int, int] = (640, 640),
        score_threshold: float = 0.25,
        kpt_threshold: float = 0.35,
        letterbox: bool = True,
    ) -> None:
        super().__init__()
        self.gie_id = int(gie_id)
        self.model_width = int(model_size[0])
        self.model_height = int(model_size[1])
        self.score_threshold = float(score_threshold)
        self.kpt_threshold = float(kpt_threshold)
        self.letterbox = bool(letterbox)
        self._logged_layers = False
        self._debug = os.environ.get("YOLO26_POSE_DEBUG", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._debug_last_log = 0.0
        self._debug_frames = 0
        self._debug_with_tensor = 0
        self._debug_drawn = 0
        self._debug_missing = 0
        self._logged_shape = False
        self._skeleton = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
        ]

    def _letterbox_params(self, frame_w: int, frame_h: int) -> Tuple[float, float, float]:
        if frame_w <= 0 or frame_h <= 0:
            return 1.0, 0.0, 0.0
        gain = min(self.model_width / frame_w, self.model_height / frame_h)
        new_w = frame_w * gain
        new_h = frame_h * gain
        pad_x = (self.model_width - new_w) / 2.0
        pad_y = (self.model_height - new_h) / 2.0
        return gain, pad_x, pad_y

    def _map_point(
        self,
        x: float,
        y: float,
        *,
        frame_w: int,
        frame_h: int,
        gain: float,
        pad_x: float,
        pad_y: float,
    ) -> Tuple[float, float]:
        if self.letterbox:
            if gain > 0.0:
                x = (x - pad_x) / gain
                y = (y - pad_y) / gain
        else:
            x = x * (float(frame_w) / float(self.model_width))
            y = y * (float(frame_h) / float(self.model_height))
        return x, y

    def _frame_size(self, frame_meta: Any) -> Tuple[int, int]:
        for w_name, h_name in (
            ("pipeline_width", "pipeline_height"),
            ("source_width", "source_height"),
            ("frame_width", "frame_height"),
            ("width", "height"),
        ):
            width = getattr(frame_meta, w_name, None)
            height = getattr(frame_meta, h_name, None)
            if width and height:
                return int(width), int(height)
        raise RuntimeError(
            "Frame metadata missing width/height attributes (source_width/source_height or pipeline_width/pipeline_height)"
        )

    def _select_batch(self, output: np.ndarray, frame_meta: Any) -> np.ndarray:
        if output.ndim != 3:
            return output
        batch_idx = None
        for attr in ("batch_id", "pad_index", "source_id"):
            value = getattr(frame_meta, attr, None)
            if value is not None:
                batch_idx = int(value)
                break
        if batch_idx is None:
            batch_idx = 0
        batch_idx = max(0, min(int(batch_idx), output.shape[0] - 1))
        return output[batch_idx]

    def _extract_from_tensor_output(self, tensor_meta: Any) -> np.ndarray:
        unique_id = int(getattr(tensor_meta, "unique_id", -1))
        if unique_id != self.gie_id:
            raise RuntimeError(
                f"Tensor meta unique_id={unique_id} does not match gie_id={self.gie_id}"
            )
        layers = getattr(tensor_meta, "get_layers", None)
        if not callable(layers):
            raise RuntimeError("Tensor meta does not expose get_layers()")
        layer_map = layers()
        if not layer_map:
            raise RuntimeError("Tensor meta contains no layers")
        if not self._logged_layers:
            self._logged_layers = True
            logger.info("YOLO26 pose tensor layers: %s", list(layer_map.keys()))
        if "output0" not in layer_map:
            raise RuntimeError(f"Missing output0 in tensor layers: {list(layer_map.keys())}")
        tensor = layer_map["output0"]
        output = _to_numpy(tensor)
        return output

    def _extract_output(self, frame_meta: Any) -> np.ndarray:
        tensor_items = getattr(frame_meta, "tensor_items", None)
        if tensor_items is None:
            raise RuntimeError("Frame metadata missing tensor_items")
        for item in tensor_items:
            tensor_meta = item.as_tensor_output() if hasattr(item, "as_tensor_output") else item
            if int(getattr(tensor_meta, "unique_id", -1)) != self.gie_id:
                continue
            return self._extract_from_tensor_output(tensor_meta)
        raise RuntimeError("No tensor output found for frame")

    def handle_metadata(self, batch_meta: Any) -> None:
        frame_items = getattr(batch_meta, "frame_items", None)
        if frame_items is None:
            raise RuntimeError("Batch metadata missing frame_items")

        now = time.time()
        drawn_this_batch = False
        tensor_found = False
        for frame_meta in frame_items:
            output = self._extract_output(frame_meta)
            tensor_found = True
            output = self._select_batch(output, frame_meta)
            if output.ndim != 2 or output.shape[1] < 6:
                raise RuntimeError(f"Unexpected pose output shape: {output.shape}")
            if self._debug and not self._logged_shape:
                self._logged_shape = True
                logger.info("YOLO26 pose output shape: %s", output.shape)

            frame_w, frame_h = self._frame_size(frame_meta)
            gain, pad_x, pad_y = self._letterbox_params(frame_w, frame_h)
            normalized = float(np.max(output[:, :4])) <= 2.0
            display_meta = None
            any_drawn = False
            for row in output:
                score = float(row[4])
                if score < self.score_threshold:
                    continue

                if display_meta is None:
                    display_meta = batch_meta.acquire_display_meta()
                    if not display_meta:
                        raise RuntimeError("Failed to acquire display meta")

                x1, y1, x2, y2 = map(float, row[:4])
                if normalized:
                    x1 *= self.model_width
                    x2 *= self.model_width
                    y1 *= self.model_height
                    y2 *= self.model_height
                x1 = max(0.0, min(x1, float(self.model_width)))
                y1 = max(0.0, min(y1, float(self.model_height)))
                x2 = max(0.0, min(x2, float(self.model_width)))
                y2 = max(0.0, min(y2, float(self.model_height)))

                x1, y1 = self._map_point(
                    x1,
                    y1,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    gain=gain,
                    pad_x=pad_x,
                    pad_y=pad_y,
                )
                x2, y2 = self._map_point(
                    x2,
                    y2,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    gain=gain,
                    pad_x=pad_x,
                    pad_y=pad_y,
                )

                x1 = max(0.0, min(x1, float(frame_w)))
                y1 = max(0.0, min(y1, float(frame_h)))
                x2 = max(0.0, min(x2, float(frame_w)))
                y2 = max(0.0, min(y2, float(frame_h)))

                rect = osd.Rect()
                rect.left = int(x1)
                rect.top = int(y1)
                rect.width = int(max(0.0, x2 - x1))
                rect.height = int(max(0.0, y2 - y1))
                rect.border_width = 2
                rect.border_color = osd.Color(0.0, 1.0, 0.0, 1.0)
                display_meta.add_rect(rect)
                any_drawn = True

                kpts = row[6:]
                if kpts.size < 51:
                    continue
                kpts = kpts.reshape(-1, 3)

                pts: List[Tuple[float, float, float]] = []
                for xk, yk, ck in kpts:
                    xk = float(xk)
                    yk = float(yk)
                    if normalized:
                        xk *= self.model_width
                        yk *= self.model_height
                    xk, yk = self._map_point(
                        xk,
                        yk,
                        frame_w=frame_w,
                        frame_h=frame_h,
                        gain=gain,
                        pad_x=pad_x,
                        pad_y=pad_y,
                    )
                    if frame_w > 0:
                        xk = max(0.0, min(xk, float(frame_w - 1)))
                    if frame_h > 0:
                        yk = max(0.0, min(yk, float(frame_h - 1)))
                    pts.append((xk, yk, float(ck)))

                for i, j in self._skeleton:
                    if i >= len(pts) or j >= len(pts):
                        continue
                    x1k, y1k, c1 = pts[i]
                    x2k, y2k, c2 = pts[j]
                    if c1 < self.kpt_threshold or c2 < self.kpt_threshold:
                        continue
                    line = osd.Line()
                    line.x1 = int(x1k)
                    line.y1 = int(y1k)
                    line.x2 = int(x2k)
                    line.y2 = int(y2k)
                    line.width = 2
                    line.color = osd.Color(0.0, 0.8, 1.0, 1.0)
                    display_meta.add_line(line)
                    any_drawn = True

                for xk, yk, ck in pts:
                    if ck < self.kpt_threshold:
                        continue
                    circ = osd.Circle()
                    circ.xc = int(xk)
                    circ.yc = int(yk)
                    circ.radius = 3
                    circ.width = 2
                    circ.color = osd.Color(1.0, 0.0, 0.0, 1.0)
                    display_meta.add_circle(circ)
                    any_drawn = True

            if any_drawn and display_meta is not None:
                frame_meta.append(display_meta)
                drawn_this_batch = True

        if self._debug:
            self._debug_frames += 1
            if tensor_found:
                self._debug_with_tensor += 1
            else:
                self._debug_missing += 1
            if drawn_this_batch:
                self._debug_drawn += 1
            if (now - float(self._debug_last_log)) >= 1.0:
                logger.info(
                    "YOLO26 pose debug: frames=%d tensor_batches=%d missing=%d drawn=%d",
                    self._debug_frames,
                    self._debug_with_tensor,
                    self._debug_missing,
                    self._debug_drawn,
                )
                self._debug_frames = 0
                self._debug_with_tensor = 0
                self._debug_missing = 0
                self._debug_drawn = 0
                self._debug_last_log = float(now)
