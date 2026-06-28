#!/usr/bin/env python3
"""Build a visual/numeric BEV path alignment report from saved telemetry."""

from __future__ import annotations

import argparse
import base64
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _percentile(values: Sequence[float], pct: float) -> float | None:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return None
    idx = int(round((len(vals) - 1) * max(0.0, min(1.0, pct))))
    return float(vals[idx])


def _stats(values: Sequence[float]) -> dict[str, Any]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return {
        "count": len(vals),
        "p05": _percentile(vals, 0.05),
        "p50": _percentile(vals, 0.50),
        "p95": _percentile(vals, 0.95),
        "mean": float(np.mean(vals)) if vals else None,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


def _read_messages(path: Path) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                messages.append(payload)
    return messages


def _floorplan_ready_messages(messages: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    ready_cameras: set[str] = set()
    ready: list[Mapping[str, Any]] = []
    for msg in messages:
        msg_type = msg.get("type")
        if msg_type == "floorplan_response":
            cam = str(msg.get("camera_id") or msg.get("camera") or "")
            if cam and not msg.get("error"):
                ready_cameras.add(cam)
                ready.append(msg)
            continue
        if msg_type == "tracking":
            cam = str(msg.get("camera_id") or "")
            if cam in ready_cameras:
                ready.append(msg)
            continue
        if msg_type == "bev-frame":
            cam = str(msg.get("cameraId") or "")
            if cam in ready_cameras:
                ready.append(msg)
    return ready


def _decode_grid(layer: Mapping[str, Any]) -> np.ndarray:
    shape = layer.get("grid_shape")
    raw = layer.get("grid_b64")
    if not isinstance(shape, Sequence) or len(shape) != 2 or not isinstance(raw, str):
        raise ValueError("floorplan grid layer must include grid_shape=[rows, cols] and grid_b64")
    rows, cols = int(shape[0]), int(shape[1])
    arr = np.frombuffer(base64.b64decode(raw), dtype="<f4")
    if arr.size != rows * cols:
        raise ValueError(f"grid size mismatch: decoded {arr.size}, expected {rows * cols}")
    return arr.reshape((rows, cols))


def _first_floorplan(messages: Sequence[Mapping[str, Any]], camera: str) -> Mapping[str, Any]:
    for msg in messages:
        if msg.get("type") != "floorplan_response":
            continue
        cam = str(msg.get("camera_id") or msg.get("camera") or "")
        if cam == camera and not msg.get("error"):
            return msg
    raise RuntimeError(f"No successful floorplan_response found for {camera}")


def _magmaish(norm: np.ndarray) -> np.ndarray:
    stops = np.array(
        [
            [0, 0, 0],
            [22, 3, 52],
            [87, 16, 110],
            [188, 55, 84],
            [249, 142, 50],
            [252, 253, 191],
        ],
        dtype=np.float32,
    )
    x = np.clip(norm, 0.0, 1.0) * float(len(stops) - 1)
    lo = np.floor(x).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    frac = (x - lo)[..., None]
    rgb = stops[lo] * (1.0 - frac) + stops[hi] * frac
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _render_floorplan(floorplan: Mapping[str, Any], *, scale: int) -> Image.Image:
    layer = floorplan.get("height_agl") or floorplan.get("height") or floorplan.get("density")
    if not isinstance(layer, Mapping):
        raise RuntimeError("floorplan has no renderable height/density layer")
    grid = _decode_grid(layer)
    finite = np.isfinite(grid)
    positive = finite & (grid > 0)
    if np.any(positive):
        lo = float(np.nanpercentile(grid[positive], 2))
        hi = float(np.nanpercentile(grid[positive], 98))
    elif np.any(finite):
        lo = float(np.nanmin(grid[finite]))
        hi = float(np.nanmax(grid[finite]))
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    norm = (np.nan_to_num(grid, nan=0.0) - lo) / (hi - lo)
    rgb = _magmaish(norm)
    rgb[~finite] = 0
    img = Image.fromarray(rgb, mode="RGB")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
    return img


def _line_mask(image: Image.Image, color: str) -> np.ndarray:
    arr = np.asarray(image.convert("RGBA"))
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    a = arr[..., 3].astype(np.int16)
    if color == "blue":
        return (a > 0) & (b > 135) & (r < 120) & (g < 160) & ((b - r) > 80)
    if color == "red":
        # Keep this strict so the annotation line is not confused with the
        # orange/yellow heatmap pixels in the BEV floorplan itself.
        return (a > 0) & (r > 210) & (g < 85) & (b < 85) & ((r - g) > 130) & ((r - b) > 130)
    raise ValueError(f"unsupported line color: {color}")


def _fit_colored_line(image_path: Path, color: str) -> dict[str, Any]:
    image = Image.open(image_path)
    mask = _line_mask(image, color)
    ys, xs = np.nonzero(mask)
    if xs.size < 12:
        raise RuntimeError(f"Unable to find enough {color} pixels in {image_path}")
    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    center = coords.mean(axis=0)
    centered = coords - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    if axis[0] < 0:
        axis = -axis
    projection = centered @ axis
    p1 = center + axis * projection.min()
    p2 = center + axis * projection.max()
    w, h = image.size
    return {
        "color": color,
        "image": str(image_path),
        "image_size": [int(w), int(h)],
        "pixel_count": int(xs.size),
        "p1_px": [float(p1[0]), float(p1[1])],
        "p2_px": [float(p2[0]), float(p2[1])],
        "p1_norm": [float(p1[0] / max(1, w - 1)), float(p1[1] / max(1, h - 1))],
        "p2_norm": [float(p2[0] / max(1, w - 1)), float(p2[1] / max(1, h - 1))],
    }


def _point_segment_distance(
    point: tuple[float, float],
    line: Mapping[str, Any],
) -> float:
    ax, ay = [float(v) for v in line["p1_norm"]]
    bx, by = [float(v) for v in line["p2_norm"]]
    px, py = point
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom <= 0.0:
        return float(math.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / denom))
    qx, qy = ax + t * vx, ay + t * vy
    return float(math.hypot(px - qx, py - qy))


def _line_y_at_x(x: float, line: Mapping[str, Any], *, margin: float) -> float | None:
    ax, ay = [float(v) for v in line["p1_norm"]]
    bx, by = [float(v) for v in line["p2_norm"]]
    lo_x, hi_x = sorted((ax, bx))
    if x < lo_x - margin or x > hi_x + margin:
        return None
    if abs(bx - ax) < 1e-9:
        return None
    t = (x - ax) / (bx - ax)
    return float(ay + t * (by - ay))


def _iter_bev_points(
    messages: Sequence[Mapping[str, Any]],
    *,
    camera: str,
    display_source: str | None,
) -> Iterable[dict[str, Any]]:
    for msg in messages:
        if msg.get("type") != "bev-frame" or str(msg.get("cameraId") or "") != camera:
            continue
        ts = int(msg.get("ts") or 0)
        for fp in msg.get("footpoints") or []:
            if not isinstance(fp, Mapping):
                continue
            if display_source and str(fp.get("displaySource") or "") != display_source:
                continue
            nx = _finite_float(fp.get("normX"))
            ny = _finite_float(fp.get("normY"))
            if nx is None or ny is None:
                continue
            yield {
                "ts": ts,
                "normX": float(nx),
                "normY": float(ny),
                "trackerId": fp.get("trackerId"),
                "stableId": fp.get("stableId"),
                "displaySource": fp.get("displaySource"),
                "floorplanInside": fp.get("floorplanInside"),
                "anchorSource": fp.get("anchorSource"),
            }


def _analyze_points(points: Sequence[Mapping[str, Any]], blue: Mapping[str, Any], red: Mapping[str, Any]) -> dict[str, Any]:
    blue_dist: list[float] = []
    red_dist: list[float] = []
    blue_y_error: list[float] = []
    red_y_error: list[float] = []
    for pt in points:
        nx = float(pt["normX"])
        ny = float(pt["normY"])
        blue_dist.append(_point_segment_distance((nx, ny), blue))
        red_dist.append(_point_segment_distance((nx, ny), red))
        by = _line_y_at_x(nx, blue, margin=0.08)
        if by is not None:
            blue_y_error.append(ny - by)
        ry = _line_y_at_x(nx, red, margin=0.08)
        if ry is not None:
            red_y_error.append(ny - ry)
    closer_to_blue = sum(1 for b, r in zip(blue_dist, red_dist) if b < r)
    return {
        "point_count": len(points),
        "closer_to_blue_count": int(closer_to_blue),
        "closer_to_blue_rate": (float(closer_to_blue) / len(points)) if points else None,
        "distance_to_blue_norm": _stats(blue_dist),
        "distance_to_red_norm": _stats(red_dist),
        "signed_y_error_to_blue_norm": _stats(blue_y_error),
        "signed_y_error_to_red_norm": _stats(red_y_error),
        "abs_y_error_to_blue_norm": _stats([abs(v) for v in blue_y_error]),
        "abs_y_error_to_red_norm": _stats([abs(v) for v in red_y_error]),
    }


def _draw_line_norm(draw: ImageDraw.ImageDraw, line: Mapping[str, Any], size: tuple[int, int], fill: tuple[int, int, int], width: int) -> None:
    w, h = size
    p1 = line["p1_norm"]
    p2 = line["p2_norm"]
    draw.line(
        [
            (float(p1[0]) * (w - 1), float(p1[1]) * (h - 1)),
            (float(p2[0]) * (w - 1), float(p2[1]) * (h - 1)),
        ],
        fill=fill,
        width=width,
    )


def _point_pixel(pt: Mapping[str, Any], size: tuple[int, int]) -> tuple[float, float]:
    w, h = size
    return float(pt["normX"]) * (w - 1), float(pt["normY"]) * (h - 1)


def _draw_overlay(
    base: Image.Image,
    points: Sequence[Mapping[str, Any]],
    blue: Mapping[str, Any],
    red: Mapping[str, Any],
    metrics: Mapping[str, Any],
    output_path: Path,
) -> None:
    img = base.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width = max(3, int(round(min(img.size) * 0.012)))
    _draw_line_norm(draw, red, img.size, (255, 46, 46, 230), width)
    _draw_line_norm(draw, blue, img.size, (55, 105, 255, 230), width)

    by_track: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pt in points:
        key = str(pt.get("stableId") or pt.get("trackerId") or "unknown")
        by_track[key].append(pt)
    for track_points in by_track.values():
        ordered = sorted(track_points, key=lambda item: int(item.get("ts") or 0))
        if len(ordered) >= 2:
            draw.line([_point_pixel(pt, img.size) for pt in ordered], fill=(255, 245, 87, 150), width=max(2, width // 2))
    radius = max(3, width)
    for pt in points:
        x, y = _point_pixel(pt, img.size)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 245, 87, 220), outline=(0, 0, 0, 220), width=1)

    img = Image.alpha_composite(img, overlay).convert("RGB")
    draw_rgb = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    blue_p50 = ((metrics.get("distance_to_blue_norm") or {}).get("p50"))
    red_p50 = ((metrics.get("distance_to_red_norm") or {}).get("p50"))
    closer = metrics.get("closer_to_blue_rate")
    label = f"current points vs target: blue p50={blue_p50:.3f} red p50={red_p50:.3f} closer_blue={closer:.2%}" if isinstance(blue_p50, float) and isinstance(red_p50, float) and isinstance(closer, float) else "current points vs target"
    pad = 8
    box = draw_rgb.textbbox((0, 0), label, font=font)
    draw_rgb.rectangle((pad - 4, pad - 4, pad + (box[2] - box[0]) + 4, pad + (box[3] - box[1]) + 6), fill=(0, 0, 0))
    draw_rgb.text((pad, pad), label, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def run_report(
    *,
    messages_path: Path,
    screenshot_path: Path,
    camera: str,
    output_dir: Path,
    scale: int,
    display_source: str | None,
) -> tuple[Path, Path, dict[str, Any]]:
    messages = _read_messages(messages_path)
    ready = _floorplan_ready_messages(messages)
    floorplan = _first_floorplan(ready, camera)
    base = _render_floorplan(floorplan, scale=max(1, int(scale)))
    blue = _fit_colored_line(screenshot_path, "blue")
    red = _fit_colored_line(screenshot_path, "red")
    points = list(_iter_bev_points(ready, camera=camera, display_source=display_source))
    metrics = _analyze_points(points, blue, red)
    report = {
        "camera": camera,
        "messages": str(messages_path),
        "screenshot": str(screenshot_path),
        "display_source_filter": display_source,
        "floorplan_grid_shape": floorplan.get("height_agl", floorplan.get("height", {})).get("grid_shape"),
        "target_lines": {"blue_expected": blue, "red_old_observed": red},
        "metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{camera}_path_alignment_report.json"
    overlay_path = output_dir / f"{camera}_path_alignment_overlay.png"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _draw_overlay(base, points, blue, red, metrics, overlay_path)
    return json_path, overlay_path, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", required=True, help="Saved diagnostics messages.ndjson")
    parser.add_argument("--screenshot", required=True, help="User red-vs-blue BEV screenshot")
    parser.add_argument("--camera", default="living-room")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument("--display-source", default="registered_depth_anchor", help="Filter BEV footpoints by displaySource; use empty string for all")
    args = parser.parse_args()

    display_source = str(args.display_source).strip() or None
    json_path, overlay_path, report = run_report(
        messages_path=Path(args.messages),
        screenshot_path=Path(args.screenshot),
        camera=str(args.camera),
        output_dir=Path(args.output_dir),
        scale=int(args.scale),
        display_source=display_source,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "report": str(json_path),
                "overlay": str(overlay_path),
                "metrics": report["metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
