#!/usr/bin/env python3
import argparse
import configparser
import ctypes
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import pyds

from noesis_core.runtime_secrets import load_camera_uri_registry


DEFAULT_MULTIURI_CONFIG = Path("pipelines/noesis_multiurisrcbin.ini")
DEFAULT_NVINFER_CONFIG = Path("pipelines/config_infer_secondary_mapanything.ini")


def _read_multiuri_list(path: Path) -> List[str]:
    parser = configparser.ConfigParser()
    parser.read(path)
    if not parser.has_section("source-list"):
        raise RuntimeError(f"Missing [source-list] in {path}")
    if not parser.has_option("source-list", "list"):
        raise RuntimeError(f"Missing source-list.list in {path}")
    raw = parser.get("source-list", "list")
    items = [item.strip() for item in raw.split(";")]
    refs = [item.removeprefix("camera-secret:") for item in items if item]
    if not refs or any(item == ref for item, ref in zip([item for item in items if item], refs)):
        raise RuntimeError(
            f"source-list.list in {path} must contain only camera-secret references"
        )
    registry = load_camera_uri_registry()
    missing = [ref for ref in refs if ref not in registry]
    if missing:
        raise RuntimeError(f"source-list.list references missing camera secret: {missing[0]}")
    return [registry[ref] for ref in refs]


def _read_multiuri_settings(path: Path) -> Dict[str, str]:
    parser = configparser.ConfigParser()
    parser.read(path)
    settings: Dict[str, str] = {}
    for section in ("source-list", "source-attr-all", "streammux"):
        if not parser.has_section(section):
            continue
        for key, value in parser.items(section):
            settings[f"{section}.{key}"] = value
    return settings


def _flatten_layer_dims(layer) -> tuple:
    dims = getattr(layer, "dims", None) or getattr(layer, "inferDims", None)
    if dims is None:
        raise AttributeError("Layer is missing dims information")
    if hasattr(dims, "d"):
        values = [dims.d[i] for i in range(getattr(dims, "numDims", 0))]
    else:
        values = list(dims)
    return tuple(int(v) if int(v) > 0 else 1 for v in values)


def _layer_dtype(layer) -> np.dtype:
    data_type = getattr(layer, "dataType", getattr(layer, "data_type", None))
    if isinstance(data_type, str):
        key = data_type.lower()
        if key in ("float", "float32", "fp32"):
            return np.float32
        if key in ("half", "float16", "fp16"):
            return np.float16
        if key in ("int32", "sint32"):
            return np.int32
        if key in ("uint8", "uchar", "uchar8"):
            return np.uint8
    try:
        from pyds import NvDsInferDataType  # type: ignore

        mapping = {
            NvDsInferDataType.NVDSINFER_TENSOR_FLOAT32: np.float32,
            NvDsInferDataType.NVDSINFER_TENSOR_FLOAT16: np.float16,
            NvDsInferDataType.NVDSINFER_TENSOR_INT32: np.int32,
            NvDsInferDataType.NVDSINFER_TENSOR_INT8: np.int8,
            NvDsInferDataType.NVDSINFER_TENSOR_UINT8: np.uint8,
        }
        return mapping.get(data_type, np.float32)
    except Exception:
        return np.float32


def _numpy_from_layer(layer) -> np.ndarray:
    shape = _flatten_layer_dims(layer)
    numel = int(np.prod(shape)) if shape else 0
    if numel <= 0:
        return np.empty(0, dtype=np.float32)

    buffer_obj = getattr(layer, "buffer", None)
    if buffer_obj is None:
        raise AttributeError("Layer has no buffer pointer")

    ptr_val = pyds.get_ptr(buffer_obj)
    dtype = _layer_dtype(layer)
    ctype_map = {
        np.float32: ctypes.c_float,
        np.float16: ctypes.c_uint16,
        np.int32: ctypes.c_int32,
        np.int8: ctypes.c_int8,
        np.uint8: ctypes.c_uint8,
    }
    ctype = ctype_map.get(dtype, ctypes.c_float)
    ptr = ctypes.cast(ptr_val, ctypes.POINTER(ctype))
    flat = np.ctypeslib.as_array(ptr, shape=(numel,))
    if dtype == np.float16:
        arr = flat.view(np.float16).copy().astype(np.float32)
        return arr.reshape(shape)
    return np.array(flat, dtype=dtype, copy=True).reshape(shape)


def _classify_layer_name(name: str) -> str:
    lowered = name.lower()
    if "depth" in lowered or "disp" in lowered:
        return "depth"
    if "conf" in lowered:
        return "confidence"
    if "mask" in lowered or "valid" in lowered:
        return "mask"
    return name


def _extract_tensors(tensor_meta) -> Dict[str, np.ndarray]:
    tensors: Dict[str, np.ndarray] = {}
    count = (
        getattr(tensor_meta, "num_layers", None)
        or getattr(tensor_meta, "num_output_layers", None)
        or getattr(tensor_meta, "num_out_layers", None)
        or 0
    )
    layers_seq: List[object] = []
    if hasattr(tensor_meta, "output_layers_info"):
        try:
            layers_seq = list(tensor_meta.output_layers_info)
        except Exception:
            layers_seq = []
    if count and not layers_seq:
        try:
            layers_seq = [pyds.get_nvds_LayerInfo(tensor_meta, i) for i in range(count)]
        except Exception:
            layers_seq = []
    if not count and layers_seq:
        count = len(layers_seq)
    if not layers_seq:
        return tensors
    for idx, layer in enumerate(layers_seq):
        array = _numpy_from_layer(layer)
        raw_name = getattr(layer, "layerName", None) or getattr(layer, "name", None) or f"layer_{idx}"
        tensors[_classify_layer_name(str(raw_name))] = array
    return tensors


def _show_heatmap(
    depth: np.ndarray,
    mask: Optional[np.ndarray],
    scale: float,
    *,
    ignore_mask: bool,
    min_mask_coverage: float,
    colormap: str,
    min_percentile: float,
    max_percentile: float,
    show_text: bool,
) -> Optional[np.ndarray]:
    valid = np.isfinite(depth)
    if mask is not None and not ignore_mask:
        coverage = float(np.mean(mask > 0)) if mask.size else 0.0
        if coverage >= min_mask_coverage:
            valid &= mask > 0
    if not np.any(valid):
        return None
    vals = depth[valid]
    lo = float(np.percentile(vals, min_percentile))
    hi = float(np.percentile(vals, max_percentile))
    if hi <= lo:
        hi = lo + 1e-3
    norm = (depth - lo) / (hi - lo)
    norm = np.clip(norm, 0, 1)
    gray = (norm * 255).astype(np.uint8)
    if colormap == "gray":
        heat = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        cmap = {
            "jet": cv2.COLORMAP_JET,
            "turbo": cv2.COLORMAP_TURBO,
            "magma": cv2.COLORMAP_MAGMA,
            "inferno": cv2.COLORMAP_INFERNO,
            "plasma": cv2.COLORMAP_PLASMA,
            "viridis": cv2.COLORMAP_VIRIDIS,
        }.get(colormap, cv2.COLORMAP_TURBO)
        heat = cv2.applyColorMap(gray, cmap)
    heat[~valid] = 0
    if scale != 1.0:
        heat = cv2.resize(heat, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if show_text:
        text = f"min={lo:.2f} max={hi:.2f}"
        cv2.putText(heat, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return heat


class MosaicDisplay:
    def __init__(self, width: int, height: int, fps: float, sink_name: str, *, debug: bool = False) -> None:
        self.pipeline = Gst.Pipeline.new("heatmap_mosaic_display")
        self.appsrc = Gst.ElementFactory.make("appsrc", "src")
        queue = Gst.ElementFactory.make("queue", "queue")
        videoconvert = Gst.ElementFactory.make("videoconvert", "conv")
        capsfilter = Gst.ElementFactory.make("capsfilter", "caps")
        sink = Gst.ElementFactory.make(sink_name, "sink")
        if sink is None:
            sink = Gst.ElementFactory.make("autovideosink", "sink")
        if not self.pipeline or not self.appsrc or not queue or not videoconvert or not capsfilter or not sink:
            raise RuntimeError("Failed to build mosaic display pipeline")

        self.appsrc.set_property("is-live", True)
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("do-timestamp", True)
        self.appsrc.set_property("block", False)
        fps_num = max(1, int(round(fps)))
        self.appsrc.set_property(
            "caps",
            Gst.Caps.from_string(f"video/x-raw,format=BGRx,width={width},height={height},framerate={fps_num}/1"),
        )
        capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=I420"))
        if sink.props and "sync" in sink.props:
            sink.set_property("sync", False)
        if sink_name == "xvimagesink":
            if sink.props and "autopaint-colorkey" in sink.props:
                sink.set_property("autopaint-colorkey", False)
            if sink.props and "colorkey" in sink.props:
                sink.set_property("colorkey", 0)

        self.pipeline.add(self.appsrc)
        self.pipeline.add(queue)
        self.pipeline.add(videoconvert)
        self.pipeline.add(capsfilter)
        self.pipeline.add(sink)
        if not self.appsrc.link(queue):
            raise RuntimeError("Failed to link appsrc -> queue")
        if not queue.link(videoconvert):
            raise RuntimeError("Failed to link queue -> videoconvert")
        if not videoconvert.link(capsfilter):
            raise RuntimeError("Failed to link videoconvert -> capsfilter")
        if not capsfilter.link(sink):
            raise RuntimeError("Failed to link capsfilter -> sink")

        self.pipeline.set_state(Gst.State.PLAYING)
        if debug:
            state = self.pipeline.get_state(2 * Gst.SECOND)
            print(f"[heatmap] mosaic display state={state.state.value_nick}", flush=True)

    def push(self, frame: np.ndarray, debug: bool = False) -> None:
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        buf = Gst.Buffer.new_allocate(None, frame.nbytes, None)
        buf.fill(0, frame.tobytes())
        res = self.appsrc.emit("push-buffer", buf)
        if debug and res != Gst.FlowReturn.OK:
            print(f"[heatmap] push-buffer mosaic -> {res}", flush=True)


def _infer_probe(pad, info, state):
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    l_frame = batch_meta.frame_meta_list
    now = time.time()

    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        source_id = getattr(frame_meta, "source_id", None)
        if source_id is None:
            source_id = getattr(frame_meta, "pad_index", 0)
        source_id = int(source_id)

        last = state["last_show"].get(source_id, 0.0)
        if now - last < state["min_interval"]:
            l_frame = l_frame.next
            continue

        l_user = frame_meta.frame_user_meta_list
        if l_user is None:
            if state["debug"] and now - state["last_meta_warn"] > 2.0:
                state["last_meta_warn"] = now
                print("[heatmap] no frame_user_meta_list seen yet", flush=True)
            l_frame = l_frame.next
            continue
        while l_user:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            if user_meta.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                tensors = _extract_tensors(tensor_meta)
                if state["debug"] and not tensors and now - state["last_tensor_warn"] > 2.0:
                    state["last_tensor_warn"] = now
                    count = (
                        getattr(tensor_meta, "num_layers", None)
                        or getattr(tensor_meta, "num_output_layers", None)
                        or getattr(tensor_meta, "num_out_layers", None)
                        or 0
                    )
                    print(f"[heatmap] tensor meta present but no layers decoded (reported count={count})", flush=True)
                if tensors and source_id not in state["layer_logged"]:
                    state["layer_logged"].add(source_id)
                    print(f"[heatmap] stream {source_id} tensor layers: {sorted(tensors.keys())}", flush=True)
                depth = tensors.get("depth")
                if depth is None and tensors:
                    for name, arr in tensors.items():
                        if arr.ndim >= 3 and arr.shape[-3] == 1:
                            depth = arr
                            break
                    if depth is None:
                        depth = next(iter(tensors.values()))
                if depth is not None:
                    batch_id = int(getattr(frame_meta, "batch_id", 0))
                    if depth.ndim == 4:
                        depth = depth[batch_id]
                    if depth.ndim == 3 and depth.shape[0] == 1:
                        depth = depth[0]
                    mask = tensors.get("mask")
                    if mask is not None:
                        if mask.ndim == 4:
                            mask = mask[batch_id]
                        if mask.ndim == 3 and mask.shape[0] == 1:
                            mask = mask[0]
                    heat = _show_heatmap(
                        depth,
                        mask,
                        state["scale"],
                        ignore_mask=state["ignore_mask"],
                        min_mask_coverage=state["min_mask_coverage"],
                        colormap=state["colormap"],
                        min_percentile=state["min_percentile"],
                        max_percentile=state["max_percentile"],
                        show_text=state["show_text"],
                    )
                    if heat is not None:
                        with state["lock"]:
                            state["frames"][source_id] = heat
                        state["last_show"][source_id] = now
                        dump_dir = state.get("dump_dir")
                        dump_every = float(state.get("dump_every", 0.0) or 0.0)
                        if dump_dir and dump_every > 0.0:
                            last_dump = state["last_dump"].get(source_id, 0.0)
                            if now - last_dump >= dump_every:
                                state["last_dump"][source_id] = now
                                out_path = Path(dump_dir) / f"heatmap_stream{source_id}.jpg"
                                try:
                                    if cv2.imwrite(str(out_path), heat) and state["debug"]:
                                        print(f"[heatmap] wrote {out_path}", flush=True)
                                except Exception:
                                    pass
                        if state["debug"] and now - state["last_stats"].get(source_id, 0.0) > 2.0:
                            state["last_stats"][source_id] = now
                            try:
                                dmin = float(np.nanmin(depth))
                                dmax = float(np.nanmax(depth))
                                print(f"[heatmap] stream {source_id} depth range: {dmin:.4f}..{dmax:.4f}", flush=True)
                            except Exception:
                                pass
                    break
            l_user = l_user.next

        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK


def main() -> None:
    parser = argparse.ArgumentParser(description="Display low-FPS MapAnything heatmaps from nvmultiurisrcbin config.")
    parser.add_argument(
        "--multiuri-config",
        default=str(DEFAULT_MULTIURI_CONFIG),
        help="Path to nvmultiurisrcbin INI (source-list.list provides camera-secret references).",
    )
    parser.add_argument(
        "--nvinfer-config",
        default=str(DEFAULT_NVINFER_CONFIG),
        help="Path to MapAnything nvinfer config.",
    )
    parser.add_argument("--heatmap-fps", type=float, default=5.0, help="Target heatmap display FPS per stream.")
    parser.add_argument("--scale", type=float, default=0.7, help="Display scale factor for heatmap windows.")
    parser.add_argument("--width", type=int, default=1920, help="Output width for nvmultiurisrcbin.")
    parser.add_argument("--height", type=int, default=1080, help="Output height for nvmultiurisrcbin.")
    parser.add_argument(
        "--infer-interval",
        type=int,
        default=-1,
        help="nvinfer interval (frames to skip). -1 auto from heatmap-fps assuming 30 fps.",
    )
    parser.add_argument(
        "--sink",
        default="glimagesink",
        help="Video sink for heatmap windows (e.g. glimagesink, xvimagesink, ximagesink).",
    )
    parser.add_argument("--ignore-mask", action="store_true", help="Ignore mask output when rendering heatmap.")
    parser.add_argument(
        "--min-mask-coverage",
        type=float,
        default=0.01,
        help="Minimum mask coverage required to apply it (0..1).",
    )
    parser.add_argument("--dump-dir", default="", help="Optional directory to write JPEG heatmaps.")
    parser.add_argument("--dump-every", type=float, default=0.0, help="Seconds between heatmap JPEG writes.")
    parser.add_argument(
        "--colormap",
        default="turbo",
        help="Colormap: turbo, jet, magma, inferno, plasma, viridis, gray.",
    )
    parser.add_argument("--min-percentile", type=float, default=2.0, help="Low percentile for normalization.")
    parser.add_argument("--max-percentile", type=float, default=98.0, help="High percentile for normalization.")
    parser.add_argument("--show-text", action="store_true", help="Overlay min/max depth on frames.")
    parser.add_argument("--tile-cols", type=int, default=2, help="Tiled output columns.")
    parser.add_argument("--tile-rows", type=int, default=2, help="Tiled output rows.")
    parser.add_argument(
        "--tile-width",
        type=int,
        default=0,
        help="Tile width (0 = use heatmap frame width).",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=0,
        help="Tile height (0 = use heatmap frame height).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug prints for tensor/meta flow.")
    args = parser.parse_args()

    multiuri_path = Path(args.multiuri_config)
    if not multiuri_path.exists():
        raise SystemExit(f"Missing multiuri config: {multiuri_path}")

    uris = _read_multiuri_list(multiuri_path)
    if len(uris) != 3:
        raise SystemExit(f"Expected 3 URIs in {multiuri_path}, found {len(uris)}")
    ini_settings = _read_multiuri_settings(multiuri_path)

    Gst.init(None)
    pipeline = Gst.Pipeline.new("ma_heatmap_multiuri")

    srcbin = Gst.ElementFactory.make("nvmultiurisrcbin", "multi_src")
    if srcbin is None:
        raise SystemExit("Failed to create nvmultiurisrcbin")
    srcbin.set_property("uri-list", ",".join(uris))
    srcbin.set_property("sensor-id-list", ",".join(str(i) for i in range(len(uris))))
    srcbin.set_property("width", int(args.width))
    srcbin.set_property("height", int(args.height))

    max_batch = ini_settings.get("source-list.max-batch-size")
    if max_batch and max_batch.isdigit():
        srcbin.set_property("max-batch-size", int(max_batch))
    else:
        srcbin.set_property("max-batch-size", len(uris))

    for key in ("batched-push-timeout", "enable-padding", "nvbuf-memory-type", "sync-inputs", "drop-pipeline-eos",
                "cache-buffer", "sort-batch", "align-first-buffer"):
        val = ini_settings.get(f"streammux.{key}")
        if val is not None and val.strip() != "":
            try:
                srcbin.set_property(key, int(val))
            except Exception:
                pass

    for key in ("gpu-id", "cudadec-memtype", "latency", "select-rtp-protocol"):
        val = ini_settings.get(f"source-attr-all.{key}")
        if val is not None and val.strip() != "":
            try:
                srcbin.set_property(key, int(val))
            except Exception:
                pass

    reconnect = ini_settings.get("source-attr-all.rtsp-reconnect-interval-sec")
    if reconnect and reconnect.isdigit():
        srcbin.set_property("rtsp-reconnect-interval", int(reconnect))
    init_reconnect = ini_settings.get("source-attr-all.init-rtsp-reconnect-interval-sec")
    if init_reconnect and init_reconnect.isdigit():
        srcbin.set_property("init-rtsp-reconnect-interval", int(init_reconnect))
    attempts = ini_settings.get("source-attr-all.rtsp-reconnect-attempts")
    if attempts and attempts.isdigit():
        srcbin.set_property("rtsp-reconnect-attempts", int(attempts))

    srcbin.set_property("port", "0")

    infer = Gst.ElementFactory.make("nvinfer", "ma_infer")
    if infer is None:
        raise SystemExit("Failed to create nvinfer")
    infer.set_property("config-file-path", str(Path(args.nvinfer_config)))
    if args.infer_interval < 0:
        auto_interval = max(int(round(30.0 / max(args.heatmap_fps, 0.1))) - 1, 0)
        infer.set_property("interval", int(auto_interval))
        print(f"[heatmap] auto interval set to {auto_interval}", flush=True)
    else:
        infer.set_property("interval", int(args.infer_interval))

    sink = Gst.ElementFactory.make("fakesink", "sink")
    sink.set_property("sync", False)

    pipeline.add(srcbin)
    pipeline.add(infer)
    pipeline.add(sink)

    if not srcbin.link(infer):
        raise SystemExit("Failed to link nvmultiurisrcbin -> nvinfer")
    if not infer.link(sink):
        raise SystemExit("Failed to link nvinfer -> fakesink")

    dump_dir = args.dump_dir.strip()
    if dump_dir:
        try:
            Path(dump_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            dump_dir = ""

    tile_cols = max(1, int(args.tile_cols))
    tile_rows = max(1, int(args.tile_rows))
    state = {
        "last_show": {},
        "min_interval": 1.0 / max(args.heatmap_fps, 0.1),
        "scale": float(args.scale),
        "frames": {},
        "latest_frames": {},
        "mosaic_display": None,
        "tile_w": 0,
        "tile_h": 0,
        "tile_cols": tile_cols,
        "tile_rows": tile_rows,
        "last_mosaic_push": 0.0,
        "stream_index_map": {},
        "next_stream_index": 0,
        "layer_logged": set(),
        "last_meta_warn": 0.0,
        "last_tensor_warn": 0.0,
        "last_stats": {},
        "debug": bool(args.debug),
        "ignore_mask": bool(args.ignore_mask),
        "min_mask_coverage": float(args.min_mask_coverage),
        "dump_dir": dump_dir,
        "dump_every": float(args.dump_every or 0.0),
        "last_dump": {},
        "colormap": str(args.colormap or "turbo").lower(),
        "min_percentile": float(args.min_percentile),
        "max_percentile": float(args.max_percentile),
        "show_text": bool(args.show_text),
        "lock": threading.Lock(),
    }
    srcpad = infer.get_static_pad("src")
    srcpad.add_probe(Gst.PadProbeType.BUFFER, _infer_probe, state)

    pipeline.set_state(Gst.State.PLAYING)
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def _on_bus_message(_bus, message):
        mtype = message.type
        if mtype == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print(f"[heatmap] GstError: {err} debug={dbg}", flush=True)
        elif mtype == Gst.MessageType.WARNING:
            err, dbg = message.parse_warning()
            print(f"[heatmap] GstWarning: {err} debug={dbg}", flush=True)
        return True

    bus.connect("message", _on_bus_message)
    loop = GLib.MainLoop()

    def _display_tick() -> bool:
        now = time.time()
        pending: Dict[int, np.ndarray] = {}
        with state["lock"]:
            if state["frames"]:
                pending = dict(state["frames"])
                state["frames"].clear()
                state["latest_frames"].update(pending)
            latest = dict(state["latest_frames"])
        if not latest:
            return True
        if not pending and now - state["last_mosaic_push"] < state["min_interval"]:
            return True
        display = state["mosaic_display"]
        if display is None:
            sample = next(iter(latest.values()))
            h, w = sample.shape[:2]
            tile_w = int(args.tile_width) if args.tile_width > 0 else w
            tile_h = int(args.tile_height) if args.tile_height > 0 else h
            cols = state["tile_cols"]
            rows = state["tile_rows"]
            display = MosaicDisplay(
                width=tile_w * cols,
                height=tile_h * rows,
                fps=args.heatmap_fps,
                sink_name=args.sink,
                debug=state["debug"],
            )
            state["mosaic_display"] = display
            state["tile_w"] = tile_w
            state["tile_h"] = tile_h
        tile_w = state["tile_w"]
        tile_h = state["tile_h"]
        cols = state["tile_cols"]
        rows = state["tile_rows"]
        max_tiles = cols * rows
        canvas = np.zeros((tile_h * rows, tile_w * cols, 3), dtype=np.uint8)
        idx_map = state["stream_index_map"]
        for source_id, frame in latest.items():
            if source_id not in idx_map:
                idx = state["next_stream_index"]
                idx_map[source_id] = idx
                state["next_stream_index"] = idx + 1
                if state["debug"]:
                    print(f"[heatmap] mapped source_id {source_id} -> tile {idx}", flush=True)
            tile_index = idx_map[source_id]
            if tile_index >= max_tiles:
                if state["debug"]:
                    print(f"[heatmap] skipping source_id {source_id} (tile {tile_index} out of grid)", flush=True)
                continue
            row = tile_index // cols
            col = tile_index % cols
            if frame.shape[0] != tile_h or frame.shape[1] != tile_w:
                tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            else:
                tile = frame
            y0 = row * tile_h
            x0 = col * tile_w
            canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
        display.push(canvas, debug=state["debug"] and bool(pending))
        state["last_mosaic_push"] = now
        return True

    GLib.timeout_add(50, _display_tick)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
