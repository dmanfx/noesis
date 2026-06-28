#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import scipy.ndimage as ndi  # type: ignore
except Exception:  # pragma: no cover - optional analysis dependency
    ndi = None  # type: ignore

# Ensure the repo root is importable when running as `python3 scripts/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
except Exception as exc:  # pragma: no cover - optional visualization dependency
    print(f"[FAIL] matplotlib required to render images: {exc}")
    sys.exit(1)

try:
    import websockets  # type: ignore
except Exception as exc:  # pragma: no cover - import guard
    print(f"[FAIL] websockets package required: {exc}")
    sys.exit(1)

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover - import guard
    print(f"[FAIL] pyyaml package required: {exc}")
    sys.exit(1)

try:
    from geometry.depth_source import DepthStorageManager  # type: ignore
except Exception:
    DepthStorageManager = None  # type: ignore

HEIGHT_CONTRAST_PCT_LO = 5.0
HEIGHT_CONTRAST_PCT_HI = 95.0
HEIGHT_CONTRAST_GAMMA = 1.0
HEIGHT_CONTRAST_DENSITY_THRESH = 1e-6

HEIGHT_AGL_FIXED_VMIN = 0.0
HEIGHT_AGL_FIXED_VMAX = 1.2


@dataclass(frozen=True)
class RenderSpec:
    name: str
    cmap: str
    percentile_lo: float = 2.0
    percentile_hi: float = 98.0
    fixed_vmin: Optional[float] = None
    fixed_vmax: Optional[float] = None
    gamma: float = 1.0


def _read_cameras(cameras_path: Path) -> List[str]:
    try:
        with cameras_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cams = data.get("cameras", {})
        out: List[str] = []
        if isinstance(cams, dict):
            for _, cfg in cams.items():
                name = (cfg or {}).get("name")
                if isinstance(name, str) and name.strip():
                    out.append(name.strip())
        return out
    except Exception:
        return []


def _pick_camera(cameras_path: Path, prefer_substr: str = "kitchen") -> str:
    cams = _read_cameras(cameras_path)
    if not cams:
        return "kitchen"
    prefer = prefer_substr.strip().lower()
    if prefer:
        for cam in cams:
            if prefer in cam.lower():
                return cam
    return cams[0]


def _resolve_scaled_intrinsics_from_cameras_yaml(
    cameras_path: Path,
    camera_id: str,
    *,
    depth_shape: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """Best-effort resolve fx,fy,cx,cy for the depth payload resolution using config/cameras.yaml."""
    with cameras_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cameras = data.get("cameras", {})
    intr_models = data.get("intrinsics_models", {})
    if not isinstance(cameras, dict) or not isinstance(intr_models, dict):
        raise ValueError("invalid cameras.yaml structure")

    cam_entry = None
    # Allow camera_id to be "1" or "kitchen".
    try:
        cam_idx = int(str(camera_id).strip())
        cam_entry = cameras.get(cam_idx) or cameras.get(str(cam_idx))
    except Exception:
        cam_entry = None
    if cam_entry is None:
        for _, entry in cameras.items():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("name", "")).strip() == str(camera_id).strip():
                cam_entry = entry
                break
    if not isinstance(cam_entry, dict):
        raise ValueError(f"camera '{camera_id}' not found in {cameras_path}")

    model = str(cam_entry.get("model", "")).strip()
    if not model:
        raise ValueError(f"camera '{camera_id}' missing model in {cameras_path}")
    model_entry = intr_models.get(model)
    if not isinstance(model_entry, dict):
        raise ValueError(f"intrinsics model '{model}' not found in {cameras_path}")
    intr = model_entry.get("intrinsics")
    if not isinstance(intr, dict):
        raise ValueError(f"intrinsics model '{model}' missing intrinsics block in {cameras_path}")

    fx = float(intr.get("fx"))
    fy = float(intr.get("fy"))
    cx = float(intr.get("cx"))
    cy = float(intr.get("cy"))
    h, w = int(depth_shape[0]), int(depth_shape[1])
    # Assume principal point is near the center; infer reference resolution.
    ref_w = max(1.0, float(cx) * 2.0)
    ref_h = max(1.0, float(cy) * 2.0)
    sx = float(w) / ref_w
    sy = float(h) / ref_h
    return fx * sx, fy * sy, cx * sx, cy * sy


def _load_extrinsics_E(calibration_json: Path, camera_id: str) -> List[float]:
    data = json.loads(calibration_json.read_text(encoding="utf-8"))
    cams = data.get("cameras")
    if not isinstance(cams, dict):
        raise ValueError("calibration JSON missing cameras dict")
    entry = cams.get(camera_id)
    if entry is None:
        # Best-effort: allow numeric camera id mapping via name equality.
        for name, cfg in cams.items():
            if str(name).strip() == str(camera_id).strip():
                entry = cfg
                break
    if not isinstance(entry, dict):
        raise ValueError(f"camera '{camera_id}' missing in {calibration_json}")
    E = entry.get("E")
    if not (isinstance(E, list) and len(E) == 16):
        raise ValueError(f"camera '{camera_id}' missing E[16] in {calibration_json}")
    return [float(x) for x in E]


def _decode_b64_array(b64: str, dtype: np.dtype, shape: Tuple[int, ...]) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=dtype)
    needed = int(np.prod(shape))
    if arr.size < needed:
        raise ValueError(f"Decoded array too small: got {arr.size}, need {needed}")
    arr = arr[:needed]
    return arr.reshape(shape)


def _decode_grid(layer: Dict[str, Any]) -> np.ndarray:
    b64 = layer.get("grid_b64")
    shape = layer.get("grid_shape") or layer.get("shape")
    if not isinstance(b64, str) or not b64:
        raise ValueError("Missing grid_b64")
    if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
        raise ValueError("Missing/invalid grid_shape")
    h, w = int(shape[0]), int(shape[1])
    return _decode_b64_array(b64, np.float32, (h, w))


def _apply_image_flip(grid: np.ndarray, image_flip: Dict[str, Any]) -> np.ndarray:
    out = grid
    flip_u = bool(image_flip.get("u", False))
    flip_v = bool(image_flip.get("v", False))
    if flip_v:
        out = out[::-1, :]
    if flip_u:
        out = out[:, ::-1]
    return out


def _percentile_range(values: np.ndarray, lo: float, hi: float) -> Tuple[float, float]:
    if values.size == 0:
        return (0.0, 1.0)
    q_lo, q_hi = np.percentile(values, [lo, hi])
    if not np.isfinite(q_lo) or not np.isfinite(q_hi) or float(q_hi) <= float(q_lo):
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return (0.0, 1.0)
        if vmax == vmin:
            vmax = vmin + 1e-6
        return (vmin, vmax)
    return (float(q_lo), float(q_hi))


def _render_grid_to_png(
    grid: np.ndarray,
    out_path: Path,
    spec: RenderSpec,
    invalid_mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    arr = np.array(grid, dtype=np.float32)
    finite = np.isfinite(arr)
    if invalid_mask is not None:
        finite &= invalid_mask.astype(bool)
    vals = arr[finite]
    if spec.fixed_vmin is not None and spec.fixed_vmax is not None:
        vmin, vmax = float(spec.fixed_vmin), float(spec.fixed_vmax)
    else:
        vmin, vmax = _percentile_range(vals, spec.percentile_lo, spec.percentile_hi)
    denom = vmax - vmin
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    norm = (arr - vmin) / denom
    norm = np.clip(norm, 0.0, 1.0)
    if spec.gamma and float(spec.gamma) != 1.0:
        # Boost low values for visualization (gamma < 1) or compress (gamma > 1).
        norm = np.power(norm, float(spec.gamma)).astype(np.float32, copy=False)
    rgba = cm.get_cmap(spec.cmap)(norm)  # type: ignore[arg-type]
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    if invalid_mask is not None:
        rgb[~invalid_mask.astype(bool)] = 0
    Image.fromarray(rgb).save(out_path)
    return (vmin, vmax)


def _render_bw_to_png(grid: np.ndarray, out_path: Path, threshold: float = 0.5) -> None:
    arr = np.array(grid, dtype=np.float32)
    bw = (arr > threshold).astype(np.uint8) * 255
    Image.fromarray(bw, mode="L").save(out_path)


def _render_rgb_to_png(rgb: np.ndarray, out_path: Path) -> None:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected RGB array (H,W,3), got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    Image.fromarray(arr, mode="RGB").save(out_path)


def _render_composite_to_png(
    walkable: np.ndarray,
    obstacle_height: np.ndarray,
    out_path: Path,
    *,
    obstacle_max: float = 1.8,
    obstacle_eps: float = 0.05,
    floor_rgb: Tuple[int, int, int] = (230, 228, 222),
) -> None:
    w = np.asarray(walkable, dtype=np.float32)
    oh = np.asarray(obstacle_height, dtype=np.float32)
    if w.shape != oh.shape:
        raise ValueError(f"shape mismatch for composite: walkable={w.shape} obstacle_height={oh.shape}")
    h_px, w_px = w.shape
    rgb = np.zeros((h_px, w_px, 3), dtype=np.uint8)
    floor_mask = np.isfinite(w) & (w > 0.5)
    if np.any(floor_mask):
        rgb[floor_mask, 0] = np.uint8(floor_rgb[0])
        rgb[floor_mask, 1] = np.uint8(floor_rgb[1])
        rgb[floor_mask, 2] = np.uint8(floor_rgb[2])
    obs_mask = np.isfinite(oh) & (oh > float(obstacle_eps))
    if np.any(obs_mask):
        denom = max(1e-6, float(obstacle_max))
        t = np.clip(oh / denom, 0.0, 1.0)
        rgba = cm.get_cmap("inferno")(t)  # type: ignore[arg-type]
        rgb_obs = (rgba[:, :, :3] * 255.0).astype(np.uint8)
        rgb[obs_mask] = rgb_obs[obs_mask]
    Image.fromarray(rgb).save(out_path)


def _make_montage(
    tiles: List[Tuple[str, Path]],
    out_path: Path,
    ncols: int = 2,
    title: str = "",
) -> None:
    if not tiles:
        return
    n = len(tiles)
    ncols = max(1, min(int(ncols), n))
    nrows = int(np.ceil(n / ncols))

    fig_w = 8 * ncols
    fig_h = 4.5 * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=150)
    if nrows == 1 and ncols == 1:
        axes_list = [axes]
    elif nrows == 1:
        axes_list = list(axes)
    elif ncols == 1:
        axes_list = list(axes)
    else:
        axes_list = [ax for row in axes for ax in row]

    for idx, ax in enumerate(axes_list):
        if idx >= n:
            ax.axis("off")
            continue
        label, path = tiles[idx]
        try:
            img = Image.open(path)
            ax.imshow(img)
            ax.set_title(label)
            ax.axis("off")
        except Exception:
            ax.set_title(f"{label} (failed to load)")
            ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


async def _recv_json(ws, timeout_s: float = 30.0) -> Dict[str, Any]:
    msg = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
    payload = json.loads(msg)
    if not isinstance(payload, dict):
        raise ValueError("Non-dict WS payload")
    return payload


async def _request_floorplan(
    ws,
    camera: str,
    request_id: str,
    max_age_sec: float,
    grid_res_m: float,
    max_extent_m: float,
    cache_only: bool,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    req = {
        "type": "get_floorplan",
        "camera": camera,
        "request_id": request_id,
        "max_age_sec": float(max_age_sec),
        "grid_res_m": float(grid_res_m),
        "max_extent_m": float(max_extent_m),
        "cache_only": bool(cache_only),
    }
    await ws.send(json.dumps(req))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        payload = await _recv_json(ws, timeout_s=min(5.0, max(0.1, deadline - time.time())))
        if payload.get("type") == "floorplan_response" and payload.get("request_id") == request_id:
            return payload
    raise TimeoutError("Timed out waiting for floorplan_response")


async def _request_ma_depth(
    ws,
    camera: str,
    request_id: str,
    ts_max_us: Optional[int] = None,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    req: Dict[str, Any] = {"type": "get_ma_depth", "camera": camera, "request_id": request_id}
    if ts_max_us is not None:
        req["ts_max_us"] = int(ts_max_us)
    await ws.send(json.dumps(req))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        payload = await _recv_json(ws, timeout_s=min(5.0, max(0.1, deadline - time.time())))
        if payload.get("type") == "ma_depth_response" and payload.get("request_id") == request_id:
            return payload
    raise TimeoutError("Timed out waiting for ma_depth_response")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _summarize_walkable(arr: np.ndarray) -> Dict[str, Any]:
    vals = np.array(arr, dtype=np.float32)
    finite = np.isfinite(vals)
    if not finite.any():
        return {"floor_cells": 0, "obstacle_cells": 0, "floor_frac": 0.0}
    floor = np.logical_and(finite, vals > 0.5)
    obstacle = np.logical_and(finite, vals <= 0.5)
    floor_cells = int(floor.sum())
    obstacle_cells = int(obstacle.sum())
    denom = max(1, floor_cells + obstacle_cells)
    return {"floor_cells": floor_cells, "obstacle_cells": obstacle_cells, "floor_frac": floor_cells / denom}


def _strip_big_grids(resp: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(resp)
    for key in (
        "density",
        "height",
        "height_agl",
        "distance",
        "gradient",
        "obstacle_height",
        "walkable",
    ):
        val = out.get(key)
        if isinstance(val, dict):
            small = dict(val)
            for k in list(small.keys()):
                if k.endswith("_b64"):
                    small[k] = f"<{k} omitted>"
            out[key] = small
    return out


def _parse_grid_res_list(text: str) -> List[float]:
    out: List[float] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out or [0.5]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump DS8 floorplan layers + MapAnything camera heatmap to PNGs for rapid iteration."
    )
    parser.add_argument("--ws", default="ws://127.0.0.1:6008", help="WebSocket URL")
    parser.add_argument("--from-disk", action="store_true", help="Load latest depth snapshot from disk (no WS required)")
    parser.add_argument(
        "--ma-depth-json",
        type=Path,
        default=None,
        help="Load a saved ma_depth_response JSON (as written by this script) instead of using WS or disk.",
    )
    parser.add_argument("--depth-base", type=Path, default=Path("data/depth"), help="Depth snapshot base path for --from-disk")
    parser.add_argument("--cameras-config", type=Path, default=Path("config/cameras.yaml"))
    parser.add_argument("--calibration-json", type=Path, default=Path("config/camera_calibration.json"))
    parser.add_argument("--camera", default="", help="Camera id (default: first kitchen-like in cameras.yaml)")
    parser.add_argument("--out-dir", type=Path, default=Path("output/floorplan_debug"), help="Output directory root")
    parser.add_argument(
        "--grid-res-m",
        default="0.5",
        help="Comma-separated grid resolutions to request (meters per cell). Example: 0.5,0.15",
    )
    parser.add_argument("--max-extent-m", type=float, default=20.0, help="Max extent (meters)")
    parser.add_argument("--max-age-sec", type=float, default=0.0, help="Max age of cached floorplan snapshot (sec)")
    parser.add_argument("--cache-only", action="store_true", help="Only serve cached floorplans")
    parser.add_argument("--offline-floorplan", action="store_true", help="Compute floorplan locally from ma_depth_response + config instead of calling get_floorplan over WS")
    parser.add_argument("--floorplan-timeout-s", type=float, default=180.0, help="Timeout per floorplan RPC (seconds)")
    parser.add_argument("--depth-timeout-s", type=float, default=60.0, help="Timeout for MapAnything depth RPC (seconds)")
    parser.add_argument(
        "--apply-image-flip",
        action="store_true",
        help="Legacy debug option: reapply the diagnostic image_flip hint onto floorplan grids.",
    )
    parser.add_argument("--no-flip", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--depth-ts-max-us",
        type=int,
        default=-1,
        help="Optional ts_max_us for get_ma_depth (-1 = omit; 0 = allow any; >0 = explicit cutoff).",
    )
    parser.add_argument("--no-montage", action="store_true", help="Skip writing montage.png")
    parser.add_argument(
        "--keep-floorplan-grids",
        action="store_true",
        help="Write ws_floorplan_response JSON including grid_b64 (large). Default strips base64 grids.",
    )
    args = parser.parse_args()

    camera = str(args.camera or "").strip() or _pick_camera(args.cameras_config)
    grid_res_list = _parse_grid_res_list(args.grid_res_m)

    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir) / f"{camera}_{ts_tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    depth_ts_max_us: Optional[int]
    if args.depth_ts_max_us is None or int(args.depth_ts_max_us) < 0:
        depth_ts_max_us = None
    else:
        depth_ts_max_us = int(args.depth_ts_max_us)

    async def _inner(ws) -> None:
        depth_ok = False
        depth_error: Optional[str] = None
        payload: Dict[str, Any] = {}
        h = 0
        w = 0
        depth: Optional[np.ndarray] = None
        conf: Optional[np.ndarray] = None
        mask: Optional[np.ndarray] = None
        disk_mgr = None

        if args.ma_depth_json:
            try:
                raw = json.loads(Path(args.ma_depth_json).read_text(encoding="utf-8"))
                (out_root / "ma_depth_input.json").write_text(
                    json.dumps(_strip_big_grids(raw), indent=2, sort_keys=True), encoding="utf-8"
                )
                payload = raw.get("payload") if isinstance(raw, dict) else None
                if not isinstance(payload, dict):
                    payload = raw if isinstance(raw, dict) else {}
                shape = payload.get("shape")
                if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
                    raise ValueError("ma_depth_json missing payload.shape")
                h, w = int(shape[0]), int(shape[1])
                depth_b64 = payload.get("depth_b64") or payload.get("depth_z_b64")
                if not isinstance(depth_b64, str) or not depth_b64:
                    raise ValueError("ma_depth_json missing payload.depth_b64")
                depth = _decode_b64_array(depth_b64, np.float32, (h, w))
                conf_b64 = payload.get("conf_b64")
                if isinstance(conf_b64, str) and conf_b64:
                    conf = _decode_b64_array(conf_b64, np.float32, (h, w))
                mask_b64 = payload.get("mask_b64")
                if isinstance(mask_b64, str) and mask_b64:
                    mask = _decode_b64_array(mask_b64, np.uint8, (h, w))
                depth_ok = True
            except Exception as exc:
                depth_ok = False
                depth_error = str(exc)
        elif args.from_disk:
            if DepthStorageManager is None:
                depth_error = "from-disk unavailable (geometry.depth_source import failed)"
            else:
                try:
                    disk_mgr = DepthStorageManager(
                        base_path=Path(args.depth_base),
                        # IMPORTANT: This debug script should never prune user data on disk.
                        # Disable retention/count enforcement by setting them to 0.
                        max_snapshots_per_camera=0,
                        retention_minutes=0.0,
                        max_total_bytes=None,
                        enable_async=False,
                        enforce_async=False,
                        zarr_clevel=0,
                        zarr_chunk_px=0,
                        min_conf=0.0,
                    )
                    path_entry = disk_mgr.latest_entry(camera, depth_ts_max_us)
                    if not path_entry:
                        depth_error = f"no depth snapshot found on disk for camera '{camera}' under {args.depth_base}"
                    else:
                        datasets = disk_mgr.load_datasets(path_entry)
                        if not datasets:
                            depth_error = f"failed to load datasets from: {path_entry}"
                        else:
                            depth = np.asarray(datasets.get("depth"), dtype=np.float32)
                            conf = np.asarray(datasets.get("conf"), dtype=np.float32)
                            mask = np.asarray(datasets.get("mask"), dtype=np.uint8)
                            if depth.ndim != 2:
                                raise ValueError(f"unexpected depth shape: {depth.shape}")
                            h, w = int(depth.shape[0]), int(depth.shape[1])
                            payload = {"ts": int(path_entry.stem)}
                            (out_root / "disk_depth_snapshot.json").write_text(
                                json.dumps(
                                    {
                                        "camera": camera,
                                        "depth_base": str(Path(args.depth_base)),
                                        "snapshot_path": str(path_entry),
                                        "snapshot_ts_us": int(path_entry.stem) if path_entry.stem.isdigit() else None,
                                        "shape": [h, w],
                                    },
                                    indent=2,
                                    sort_keys=True,
                                ),
                                encoding="utf-8",
                            )
                            depth_ok = True
                except Exception as exc:
                    depth_error = str(exc)
        else:
            if ws is None:
                depth_error = "missing websocket handle"
            else:
                depth_req_id = f"dump-ma-depth-{int(time.time()*1000)}"
                try:
                    depth_resp = await _request_ma_depth(
                        ws, camera, depth_req_id, ts_max_us=depth_ts_max_us, timeout_s=float(args.depth_timeout_s)
                    )
                    (out_root / "ws_ma_depth_response.json").write_text(
                        json.dumps(_strip_big_grids(depth_resp), indent=2, sort_keys=True), encoding="utf-8"
                    )

                    depth_ok = bool(depth_resp.get("ok", False)) and "payload" in depth_resp
                    if not depth_ok:
                        depth_error = str(depth_resp.get("error", "unknown"))
                    else:
                        payload = depth_resp.get("payload") or {}
                        shape = payload.get("shape")
                        if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
                            raise ValueError("ma_depth_response.payload.shape missing/invalid")
                        h, w = int(shape[0]), int(shape[1])
                        depth_b64 = payload.get("depth_b64") or payload.get("depth_z_b64")
                        if not isinstance(depth_b64, str) or not depth_b64:
                            raise ValueError("ma_depth_response.payload.depth_b64 missing")
                        depth = _decode_b64_array(depth_b64, np.float32, (h, w))
                        conf = None
                        conf_b64 = payload.get("conf_b64")
                        if isinstance(conf_b64, str) and conf_b64:
                            conf = _decode_b64_array(conf_b64, np.float32, (h, w))
                        mask = None
                        if isinstance(payload.get("mask_b64"), str) and payload.get("mask_b64"):
                            mask = _decode_b64_array(payload["mask_b64"], np.uint8, (h, w))
                except Exception as exc:
                    depth_ok = False
                    depth_error = str(exc)

        if depth_ok and depth is not None:
            valid = np.isfinite(depth) & (depth > 0.1)
            if mask is not None:
                valid &= np.asarray(mask).astype(bool)
            depth_png = out_root / "camera_depth_turbo.png"
            vmin, vmax = _render_grid_to_png(
                depth,
                depth_png,
                RenderSpec("camera_depth", cmap="turbo", percentile_lo=2.0, percentile_hi=98.0),
                invalid_mask=valid,
            )
            (out_root / "camera_depth_range.json").write_text(
                json.dumps({"vmin": vmin, "vmax": vmax}, indent=2, sort_keys=True), encoding="utf-8"
            )
            if mask is not None:
                Image.fromarray((np.asarray(mask).astype(np.uint8) * 255), mode="L").save(out_root / "camera_mask.png")
        else:
            (out_root / "camera_depth_error.txt").write_text(depth_error or "unknown", encoding="utf-8")

        # Precompute a calibration bundle for any local floorplan generation paths.
        calib_bundle: Optional[Dict[str, Any]] = None
        if depth_ok and h > 0 and w > 0:
            try:
                fx, fy, cx, cy = _resolve_scaled_intrinsics_from_cameras_yaml(
                    args.cameras_config,
                    camera,
                    depth_shape=(h, w),
                )
                E = _load_extrinsics_E(args.calibration_json, camera)
                calib_bundle = {"cameras": {"K": {camera: [fx, fy, cx, cy]}, "E": {camera: E}}}
                if disk_mgr is not None:
                    setattr(disk_mgr, "calibration_bundle", calib_bundle)
            except Exception as exc:
                (out_root / "calibration_error.txt").write_text(str(exc), encoding="utf-8")

        # Request/generate floorplans for each requested resolution.
        montage_tiles: List[Tuple[str, Path]] = []
        summaries: List[str] = []
        for grid_res_m in grid_res_list:
            req_id = f"dump-floorplan-{grid_res_m}-{int(time.time()*1000)}"
            clean_topdown_png: Optional[Path] = None
            clean_topdown_rgb_png: Optional[Path] = None
            try:
                if args.from_disk:
                    if disk_mgr is None:
                        raise RuntimeError("from-disk requested but DepthStorageManager could not be initialized")
                    resp = disk_mgr.generate_topdown_floorplan(
                        camera,
                        max_age_sec=float(args.max_age_sec),
                        grid_res_m=float(grid_res_m),
                        max_extent_m=float(args.max_extent_m),
                        cache_only=bool(args.cache_only),
                    )
                    resp.setdefault("served_from_cache", False)
                    if hasattr(disk_mgr, "generate_clean_topdown_floorplan"):
                        try:
                            rgb = disk_mgr.generate_clean_topdown_floorplan(
                                camera,
                                max_age_sec=float(args.max_age_sec),
                                grid_res_m=float(grid_res_m),
                                max_extent_m=float(args.max_extent_m),
                            )
                            if rgb is not None:
                                clean_topdown_png = out_root / f"clean_topdown_floorplan__grid{grid_res_m}.png"
                                _render_rgb_to_png(rgb, clean_topdown_png)
                        except Exception as exc:
                            (out_root / f"clean_topdown_floorplan_error__grid{grid_res_m}.txt").write_text(
                                    str(exc), encoding="utf-8"
                                )
                    if hasattr(disk_mgr, "generate_clean_topdown_rgb"):
                        try:
                            rgb2 = disk_mgr.generate_clean_topdown_rgb(
                                camera,
                                max_age_sec=float(args.max_age_sec),
                            )
                            if rgb2 is not None:
                                clean_topdown_rgb_png = out_root / f"clean_topdown_rgb__grid{grid_res_m}.png"
                                _render_rgb_to_png(rgb2, clean_topdown_rgb_png)
                        except Exception as exc:
                            (out_root / f"clean_topdown_rgb_error__grid{grid_res_m}.txt").write_text(
                                str(exc), encoding="utf-8"
                            )
                elif args.offline_floorplan:
                    if not depth_ok:
                        raise RuntimeError("offline-floorplan requires a valid ma_depth_response payload")
                    if DepthStorageManager is None:
                        raise RuntimeError("offline-floorplan unavailable (geometry.depth_source import failed)")
                    if depth is None:
                        raise RuntimeError("offline-floorplan missing decoded depth array")
                    if conf is None:
                        # DepthStorageManager expects a conf array; fall back to 1.0 everywhere.
                        conf = np.ones_like(depth, dtype=np.float32)
                    if mask is None:
                        mask = np.ones(depth.shape, dtype=np.uint8)
                    if not calib_bundle:
                        raise RuntimeError("offline-floorplan missing calibration bundle (see calibration_error.txt)")
                    with tempfile.TemporaryDirectory(prefix="noesis_floorplan_offline_") as tmp:
                        mgr = DepthStorageManager(
                            base_path=Path(tmp),
                            # Debug: never prune the single snapshot we just stored.
                            max_snapshots_per_camera=0,
                            retention_minutes=0.0,
                            max_total_bytes=None,
                            enable_async=False,
                            enforce_async=False,
                            zarr_clevel=0,
                            zarr_chunk_px=0,
                            min_conf=0.0,
                        )
                        setattr(mgr, "calibration_bundle", calib_bundle)
                        now_us = int(time.time() * 1_000_000)
                        ts_us = int(payload.get("ts", 0) or 0)
                        if ts_us <= 0 or ts_us > (now_us + 10_000_000):
                            ts_us = now_us
                        mgr.store(camera, ts_us, depth, conf, np.asarray(mask).astype(np.uint8, copy=False))  # type: ignore[arg-type]
                        resp = mgr.generate_topdown_floorplan(
                            camera,
                            max_age_sec=float(args.max_age_sec),
                            grid_res_m=float(grid_res_m),
                            max_extent_m=float(args.max_extent_m),
                            cache_only=bool(args.cache_only),
                        )
                        resp.setdefault("served_from_cache", False)
                        if hasattr(mgr, "generate_clean_topdown_floorplan"):
                            try:
                                rgb = mgr.generate_clean_topdown_floorplan(
                                    camera,
                                    max_age_sec=float(args.max_age_sec),
                                    grid_res_m=float(grid_res_m),
                                    max_extent_m=float(args.max_extent_m),
                                )
                                if rgb is not None:
                                    clean_topdown_png = out_root / f"clean_topdown_floorplan__grid{grid_res_m}.png"
                                    _render_rgb_to_png(rgb, clean_topdown_png)
                            except Exception as exc:
                                (out_root / f"clean_topdown_floorplan_error__grid{grid_res_m}.txt").write_text(
                                    str(exc), encoding="utf-8"
                                )
                        if hasattr(mgr, "generate_clean_topdown_rgb"):
                            try:
                                rgb2 = mgr.generate_clean_topdown_rgb(
                                    camera,
                                    max_age_sec=float(args.max_age_sec),
                                )
                                if rgb2 is not None:
                                    clean_topdown_rgb_png = out_root / f"clean_topdown_rgb__grid{grid_res_m}.png"
                                    _render_rgb_to_png(rgb2, clean_topdown_rgb_png)
                            except Exception as exc:
                                (out_root / f"clean_topdown_rgb_error__grid{grid_res_m}.txt").write_text(
                                    str(exc), encoding="utf-8"
                                )
                else:
                    if ws is None:
                        raise RuntimeError("missing websocket handle for get_floorplan")
                    resp = await _request_floorplan(
                        ws,
                        camera,
                        req_id,
                        max_age_sec=args.max_age_sec,
                        grid_res_m=grid_res_m,
                        max_extent_m=args.max_extent_m,
                        cache_only=args.cache_only,
                        timeout_s=float(args.floorplan_timeout_s),
                    )

                fp_json = resp if bool(args.keep_floorplan_grids) else _strip_big_grids(resp)
                (out_root / f"ws_floorplan_response__grid{grid_res_m}.json").write_text(
                    json.dumps(fp_json, indent=2, sort_keys=True), encoding="utf-8"
                )
                if resp.get("error"):
                    (out_root / f"floorplan_error__grid{grid_res_m}.txt").write_text(
                        str(resp.get("error")), encoding="utf-8"
                    )
                    continue
            except Exception as exc:
                (out_root / f"floorplan_error__grid{grid_res_m}.txt").write_text(str(exc), encoding="utf-8")
                continue

            image_flip = resp.get("image_flip") if isinstance(resp.get("image_flip"), dict) else {}
            do_flip = bool(args.apply_image_flip) and not bool(args.no_flip)

            def _get_layer(name: str) -> Optional[np.ndarray]:
                layer = resp.get(name)
                if not isinstance(layer, dict):
                    return None
                grid = _decode_grid(layer)
                return _apply_image_flip(grid, image_flip) if (do_flip and image_flip) else grid

            density = _get_layer("density")
            height = _get_layer("height")
            height_agl = _get_layer("height_agl")
            distance = _get_layer("distance")
            gradient = _get_layer("gradient")
            obstacle_height = _get_layer("obstacle_height")
            walkable = _get_layer("walkable")

            meta_out: Dict[str, Any] = {
                "camera_id": resp.get("camera_id") or resp.get("cameraId") or camera,
                "grid_res_m": float(resp.get("grid_res_m", grid_res_m)),
                "max_extent_m": float(resp.get("max_extent_m", args.max_extent_m)),
                "served_from_cache": bool(resp.get("served_from_cache", False)),
                "snapshot_ts": resp.get("snapshot_ts"),
                "frame": resp.get("frame"),
                "bounds": resp.get("bounds"),
                "scale_m_per_px": _safe_float(resp.get("scale_m_per_px")),
                "image_flip": image_flip,
                "clean_floorplan_meta": resp.get("clean_floorplan_meta"),
                "height_agl_meta": resp.get("height_agl_meta"),
            }
            if walkable is not None:
                meta_out["walkable_summary"] = _summarize_walkable(walkable)
                ws_summary = meta_out["walkable_summary"]
                if isinstance(ws_summary, dict):
                    floor_frac = _safe_float(ws_summary.get("floor_frac"))
                    summaries.append(
                        "grid_res_m="
                        f"{grid_res_m} floor_frac={floor_frac:.3f} "
                        f"floor_cells={ws_summary.get('floor_cells')} obstacle_cells={ws_summary.get('obstacle_cells')} "
                        f"served_from_cache={meta_out.get('served_from_cache')}"
                    )
            (out_root / f"floorplan_meta__grid{grid_res_m}.json").write_text(
                json.dumps(meta_out, indent=2, sort_keys=True), encoding="utf-8"
            )

            if density is not None:
                out = out_root / f"floorplan_density_gray__grid{grid_res_m}.png"
                _render_grid_to_png(density, out, RenderSpec("density", cmap="gray"), invalid_mask=np.isfinite(density))
                montage_tiles.append((f"density (grid={grid_res_m})", out))
                if height is not None:
                    out = out_root / f"floorplan_height_inferno__grid{grid_res_m}.png"
                    _render_grid_to_png(height, out, RenderSpec("height", cmap="inferno"), invalid_mask=np.isfinite(height))
                    montage_tiles.append((f"height (grid={grid_res_m})", out))
                    # High-contrast visualization: auto-range on observed cells and hide unobserved cells
                    # (density == 0) so occlusions show up as black instead of being filled as "floor".
                    contrast_mask = np.isfinite(height)
                    if density is not None:
                        contrast_mask &= np.isfinite(density) & (density > float(HEIGHT_CONTRAST_DENSITY_THRESH))
                    contrast_vals = height[contrast_mask]
                    if contrast_vals.size > 0:
                        vmin = float(np.percentile(contrast_vals, float(HEIGHT_CONTRAST_PCT_LO)))
                        vmax = float(np.percentile(contrast_vals, float(HEIGHT_CONTRAST_PCT_HI)))
                        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                            vmin, vmax = float(np.nanmin(contrast_vals)), float(np.nanmax(contrast_vals))
                    else:
                        vmin, vmax = 0.0, 1.0
                    out2 = out_root / f"floorplan_height_contrast_turbo__grid{grid_res_m}.png"
                    _render_grid_to_png(
                        height,
                        out2,
                        RenderSpec(
                            "height_contrast",
                            cmap="turbo",
                            fixed_vmin=vmin,
                            fixed_vmax=vmax,
                            gamma=float(HEIGHT_CONTRAST_GAMMA),
                        ),
                        invalid_mask=contrast_mask,
                    )
                    (out_root / f"floorplan_height_contrast_range__grid{grid_res_m}.json").write_text(
                        json.dumps(
                            {
                                "vmin": vmin,
                                "vmax": vmax,
                                "pct_lo": float(HEIGHT_CONTRAST_PCT_LO),
                                "pct_hi": float(HEIGHT_CONTRAST_PCT_HI),
                                "gamma": float(HEIGHT_CONTRAST_GAMMA),
                                "masked_cells": int(contrast_vals.size),
                            },
                            indent=2,
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                    montage_tiles.append((f"height contrast (grid={grid_res_m})", out2))
                if height_agl is not None:
                    agl_mask = np.isfinite(height_agl)
                    if density is not None:
                        agl_mask &= np.isfinite(density) & (density > float(HEIGHT_CONTRAST_DENSITY_THRESH))
                    out_agl = out_root / f"floorplan_height_agl_turbo__grid{grid_res_m}.png"
                    _render_grid_to_png(
                        height_agl,
                        out_agl,
                        RenderSpec(
                            "height_agl",
                            cmap="turbo",
                            fixed_vmin=float(HEIGHT_AGL_FIXED_VMIN),
                            fixed_vmax=float(HEIGHT_AGL_FIXED_VMAX),
                        ),
                        invalid_mask=agl_mask,
                    )
                    montage_tiles.append((f"height_agl (grid={grid_res_m})", out_agl))
            if distance is not None:
                out = out_root / f"floorplan_distance_viridis__grid{grid_res_m}.png"
                _render_grid_to_png(
                    distance, out, RenderSpec("distance", cmap="viridis"), invalid_mask=np.isfinite(distance)
                )
                montage_tiles.append((f"distance (grid={grid_res_m})", out))
            if gradient is not None:
                out = out_root / f"floorplan_gradient_viridis__grid{grid_res_m}.png"
                _render_grid_to_png(
                    gradient,
                    out,
                    RenderSpec("gradient", cmap="viridis", fixed_vmin=0.0, fixed_vmax=1.0),
                    invalid_mask=np.isfinite(gradient),
                )
                montage_tiles.append((f"gradient (grid={grid_res_m})", out))
            if obstacle_height is not None:
                out = out_root / f"floorplan_obstacle_height_clean_inferno__grid{grid_res_m}.png"
                _render_grid_to_png(
                    obstacle_height,
                    out,
                    RenderSpec("obstacle_height", cmap="inferno", percentile_lo=1.0, percentile_hi=99.0),
                    invalid_mask=np.isfinite(obstacle_height),
                )
                montage_tiles.append((f"obstacle_height (grid={grid_res_m})", out))
            if clean_topdown_png is not None and clean_topdown_png.exists():
                montage_tiles.append((f"clean topdown (grid={grid_res_m})", clean_topdown_png))
            if clean_topdown_rgb_png is not None and clean_topdown_rgb_png.exists():
                montage_tiles.append((f"clean topdown rgb (grid={grid_res_m})", clean_topdown_rgb_png))
            if walkable is not None:
                out = out_root / f"floorplan_walkable_bw__grid{grid_res_m}.png"
                _render_bw_to_png(walkable, out)
                montage_tiles.append((f"walkable bw (grid={grid_res_m})", out))
            if walkable is not None and obstacle_height is not None:
                out = out_root / f"floorplan_composite__grid{grid_res_m}.png"
                _render_composite_to_png(
                    walkable,
                    obstacle_height,
                    out,
                    obstacle_max=float(_safe_float((resp.get("obstacle_height") or {}).get("value_max"), 1.8)),
                )
                montage_tiles.append((f"composite (grid={grid_res_m})", out))
            # Dump raw grids + lightweight diagnostics for quick iteration.
            grids_to_save: Dict[str, np.ndarray] = {}
            for name, arr in (
                ("density", density),
                ("height", height),
                ("height_agl", height_agl),
                ("distance", distance),
                ("gradient", gradient),
                ("obstacle_height", obstacle_height),
                ("walkable", walkable),
            ):
                if arr is None:
                    continue
                grids_to_save[name] = np.asarray(arr)
            if grids_to_save:
                np.savez_compressed(out_root / f"grids__grid{grid_res_m}.npz", **grids_to_save)

            analysis: Dict[str, Any] = {"grid_res_m": float(grid_res_m)}
            observed_mask = None
            if density is not None:
                observed_mask = np.isfinite(density) & (density > float(HEIGHT_CONTRAST_DENSITY_THRESH))

            def _corr(a: Optional[np.ndarray], b: Optional[np.ndarray], mask: Optional[np.ndarray]) -> Optional[float]:
                if a is None or b is None:
                    return None
                aa = np.asarray(a, dtype=np.float32)
                bb = np.asarray(b, dtype=np.float32)
                if aa.shape != bb.shape:
                    return None
                m = np.isfinite(aa) & np.isfinite(bb)
                if mask is not None:
                    m &= mask
                if int(np.count_nonzero(m)) < 128:
                    return None
                xa = aa[m].ravel()
                xb = bb[m].ravel()
                try:
                    return float(np.corrcoef(xa, xb)[0, 1])
                except Exception:
                    return None

            analysis["corr"] = {
                "height_vs_distance": _corr(height, distance, observed_mask),
                "height_agl_vs_distance": _corr(height_agl, distance, observed_mask),
            }

            if obstacle_height is not None and ndi is not None:
                oh = np.asarray(obstacle_height, dtype=np.float32)
                m = np.isfinite(oh) & (oh > 0.05)
                labeled, num = ndi.label(m)
                counts = np.bincount(labeled.ravel())
                if counts.size:
                    counts[0] = 0
                top = np.argsort(counts)[::-1][:12]
                comps = [{"label": int(i), "cells": int(counts[i])} for i in top if int(counts[i]) > 0]
                analysis["obstacle_components"] = {"count": int(num), "top": comps}

            (out_root / f"analysis__grid{grid_res_m}.json").write_text(
                json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8"
            )

        if not args.no_montage:
            montage_out = out_root / "montage.png"
            tiles: List[Tuple[str, Path]] = []
            if (out_root / "camera_depth_turbo.png").exists():
                tiles.append(("camera depth (turbo)", out_root / "camera_depth_turbo.png"))
            tiles.extend(montage_tiles)
            _make_montage(tiles, montage_out, ncols=2, title=f"{camera} {ts_tag}")

        (out_root / "summary.txt").write_text("\n".join(summaries) + ("\n" if summaries else ""), encoding="utf-8")

    async def _run() -> None:
        if args.from_disk or args.ma_depth_json:
            await _inner(None)
            return
        # Depth + floorplan payloads contain large base64 grids; disable the default 1MB frame limit.
        async with websockets.connect(args.ws, max_size=None) as ws:
            await _inner(ws)

    asyncio.run(_run())

    print(f"[OK] wrote debug images to: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
