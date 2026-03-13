"""Compare a prototype person anchor against a saved MapAnything depth snapshot."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a NOESIS.OBJECT_DEPTH payload against saved MapAnything depth.")
    parser.add_argument("--object-depth-json", type=Path, required=True, help="Path to a JSON file containing one object_depth payload.")
    parser.add_argument("--ma-depth-json", type=Path, required=True, help="Path to a saved ma_depth_response JSON file.")
    parser.add_argument("--frame-width", type=int, default=1920, help="Canonical DS8 frame width used by the object-depth payload.")
    parser.add_argument("--frame-height", type=int, default=1080, help="Canonical DS8 frame height used by the object-depth payload.")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _decode_depth(payload: Mapping[str, Any]) -> Tuple[np.ndarray, int, int]:
    shape = payload.get("shape")
    if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
        height = int(payload.get("height", 0) or 0)
        width = int(payload.get("width", 0) or 0)
        shape = (height, width)
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("ma_depth_response payload is missing a valid shape")
    depth_b64 = payload.get("depth_b64") or payload.get("depth_z_b64")
    if not isinstance(depth_b64, str) or not depth_b64:
        raise ValueError("ma_depth_response payload is missing depth_b64/depth_z_b64")
    raw = base64.b64decode(depth_b64)
    arr = np.frombuffer(raw, dtype=np.float32)
    if arr.size != height * width:
        raise ValueError(f"Decoded depth has {arr.size} elements, expected {height * width}")
    return arr.reshape((height, width)), width, height


def _sample_anchor(depth_map: np.ndarray, *, anchor_uv: Tuple[float, float], frame_size: Tuple[int, int]) -> Tuple[float | None, float | None]:
    frame_w, frame_h = frame_size
    depth_h, depth_w = depth_map.shape[:2]
    scale_x = float(depth_w) / float(max(1, frame_w))
    scale_y = float(depth_h) / float(max(1, frame_h))
    u = max(0, min(depth_w - 1, int(round(float(anchor_uv[0]) * scale_x))))
    v = max(0, min(depth_h - 1, int(round(float(anchor_uv[1]) * scale_y))))
    point_value = float(depth_map[v, u]) if np.isfinite(depth_map[v, u]) else None
    y0 = max(0, v - 2)
    y1 = min(depth_h, v + 3)
    x0 = max(0, u - 2)
    x1 = min(depth_w, u + 3)
    patch = np.asarray(depth_map[y0:y1, x0:x1], dtype=np.float32)
    patch_values = patch[np.isfinite(patch)]
    patch_median = float(np.median(patch_values)) if patch_values.size else None
    return point_value, patch_median


def main() -> None:
    args = _parse_args()
    object_payload = _load_json(args.object_depth_json)
    if not isinstance(object_payload, Mapping):
        raise SystemExit("object depth JSON must decode to a dict")
    ma_response = _load_json(args.ma_depth_json)
    payload = ma_response.get("payload") if isinstance(ma_response, Mapping) and "payload" in ma_response else ma_response
    if not isinstance(payload, Mapping):
        raise SystemExit("MapAnything JSON must decode to a dict payload")

    anchor_uv = object_payload.get("anchor_uv")
    if not (isinstance(anchor_uv, (list, tuple)) and len(anchor_uv) == 2):
        raise SystemExit("object depth payload is missing anchor_uv")

    depth_map, depth_w, depth_h = _decode_depth(payload)
    point_depth, patch_median = _sample_anchor(
        depth_map,
        anchor_uv=(float(anchor_uv[0]), float(anchor_uv[1])),
        frame_size=(int(args.frame_width), int(args.frame_height)),
    )
    raw_anchor_depth = object_payload.get("anchor_depth_m")
    raw_anchor_depth = float(raw_anchor_depth) if raw_anchor_depth is not None else None
    point_delta = None if raw_anchor_depth is None or point_depth is None else float(point_depth - raw_anchor_depth)
    patch_delta = None if raw_anchor_depth is None or patch_median is None else float(patch_median - raw_anchor_depth)
    summary = {
        "camera": payload.get("camera"),
        "object_source_id": object_payload.get("source_id"),
        "object_frame_id": object_payload.get("frame_id"),
        "anchor_uv": [float(anchor_uv[0]), float(anchor_uv[1])],
        "anchor_depth_m": raw_anchor_depth,
        "mapanything_shape": [depth_h, depth_w],
        "mapanything_anchor_point_m": point_depth,
        "mapanything_anchor_patch_median_m": patch_median,
        "delta_point_m": point_delta,
        "delta_patch_median_m": patch_delta,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
