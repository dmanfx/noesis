"""
DS8Adapter

Builds a DeepStream 8 GStreamer pipeline using GI (no pyservicemaker) to expose
an appsink-driven API compatible with the DS7 surface used in main.py:

Public surface:
- start() -> bool
- stop() -> None
- get_stats() -> dict
- source_info: Dict[int, Dict[str, Any]]
- read_encoded_jpeg(sensor_id: int, timeout: float=0.1) -> (bool, bytes|None)
- read_mosaic_jpeg(timeout: float=0.1) -> (bool, bytes|None)

Architecture (high level):
  nvmultiurisrcbin -> pre_tee
    pre_tee -> nvstreamdemux -> per-stream: queue(leaky) -> nvvideoconvert -> caps(I420, NVMM) -> nvjpegenc -> appsink_mde_i
    pre_tee -> [optional nvdspreprocess] -> nvinfer -> nvtracker -> [optional nvdsanalytics] -> nvmultistreamtiler -> nvvideoconvert -> caps(I420, NVMM) -> nvjpegenc -> appsink_mosaic

Notes:
- End-to-end NVMM; only touch CPU for encoded JPEG byte copy in appsink callbacks.
- All upstream queues are leaky to avoid backpressure.
- nvinfer is configured from the existing DS7 INI file; we also enable tensor meta.
"""

from __future__ import annotations

import os
import threading
import time
import queue
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import gi  # type: ignore
    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import Gst, GLib, GstApp  # type: ignore
except Exception:  # pragma: no cover - allow import without GI for tests
    gi = None
    Gst = None
    GLib = None
    GstApp = None


class DS8Adapter:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.logger = self._get_logger()

        camera_cfg = getattr(self.config, "cameras", None)
        default_width = int(getattr(camera_cfg, "CAMERA_WIDTH", 1920)) if camera_cfg else 1920
        default_height = int(getattr(camera_cfg, "CAMERA_HEIGHT", 1080)) if camera_cfg else 1080

        self.sources: List[Dict[str, Any]] = []

        def _append_source(entry: Dict[str, Any], fallback_name: str, source_type: str) -> None:
            url = str(entry.get("url", "") or "").strip()
            if not url:
                self.logger.warning("Skipping %s source without URL", source_type)
                return
            name = entry.get("name") or fallback_name
            self.sources.append(
                {
                    "name": name,
                    "clean_name": self._clean_name(name),
                    "url": url,
                    "width": int(entry.get("width", default_width)),
                    "height": int(entry.get("height", default_height)),
                    "type": source_type,
                }
            )

        rtsp_streams = getattr(camera_cfg, "RTSP_STREAMS", []) if camera_cfg else []
        for stream in rtsp_streams:
            if isinstance(stream, dict) and stream.get("enabled", True):
                _append_source(stream, stream.get("name") or f"Camera {len(self.sources)+1}", "rtsp")

        if camera_cfg and getattr(camera_cfg, "USE_WEBCAM", False):
            webcam_url = str(getattr(camera_cfg, "WEBCAM_URL", "v4l2:///dev/video0"))
            webcam_entry = {
                "name": "Webcam",
                "url": webcam_url,
                "width": default_width,
                "height": default_height,
            }
            _append_source(webcam_entry, "Webcam", "webcam")

        video_files = getattr(camera_cfg, "VIDEO_FILES", []) if camera_cfg else []
        for idx, video_path in enumerate(video_files or []):
            if not isinstance(video_path, str) or not video_path:
                continue
            abs_path = os.path.abspath(video_path)
            if not os.path.exists(abs_path) and not video_path.startswith("file://"):
                self.logger.warning("Skipping file source (not found): %s", abs_path)
                continue
            url = video_path if video_path.startswith("file://") else f"file://{abs_path}"
            file_entry = {
                "name": f"Video File {idx}",
                "url": url,
                "width": default_width,
                "height": default_height,
            }
            _append_source(file_entry, file_entry["name"], "file")

        # Canonical per-source info (sensor ids 0..N-1)
        self.source_info: Dict[int, Dict[str, Any]] = {}
        for idx, src in enumerate(self.sources):
            name = src.get("name") or f"Camera {idx+1}"
            clean = self._clean_name(name)
            self.source_info[idx] = {
                "name": name,
                "clean_name": clean,
                "url": src.get("url", ""),
                "width": int(src.get("width", default_width)),
                "height": int(src.get("height", default_height)),
                "type": src.get("type", "rtsp"),
            }

        # Queues for encoded JPEG payloads
        self.jpeg_queues: Dict[int, queue.Queue[bytes]] = {i: queue.Queue(maxsize=12) for i in self.source_info}
        self.mosaic_queue: queue.Queue[bytes] = queue.Queue(maxsize=2)

        # Internal state
        self.pipeline = None
        self.mainloop: Optional[GLib.MainLoop] = None
        self.mainloop_thread: Optional[threading.Thread] = None
        self.running = False
        self._prepared = False
        self._activated = False
        self._errors: List[str] = []
        self._start_time = 0.0
        self._frame_count_total = 0
        self._frame_counters: Dict[int, int] = {i: 0 for i in self.source_info}
        self._mosaic_counter = 0
        self._frame_count_lock = threading.Lock()
        self._mde_first_log: Dict[int, bool] = {}
        self._mosaic_first_logged = False
        self._read_empty_last: Dict[int, float] = {}

        # Named elements for clarity
        self._elements: Dict[str, Any] = {}

        # Allow attaching external publisher(s) for parity with DS7 (no-op here)
        self.occupancy_publisher = None

    # -------------------- Public API --------------------
    def start(self) -> bool:
        if self.running:
            return True

        # GI / GStreamer init
        if not Gst:  # GI not available
            self._errors.append("GI/GStreamer not available; DS8Adapter running in dry mode")
            self.logger.warning("GI/GStreamer not available; DS8Adapter dry-run (no GPU)")
            self.running = True
            self._prepared = False
            self._activated = False
            return True

        try:
            Gst.init(None)
        except Exception as e:
            self._errors.append(f"Gst.init failed: {e}")
            self.logger.error("GStreamer initialization failed: %s", e)
            return False

        try:
            self._build_pipeline()
            self._prepared = True
        except Exception as e:
            self._errors.append(f"Pipeline build failed: {e}")
            self.logger.exception("Failed to build DS8 pipeline: %s", e)
            return False

        if not self.pipeline:
            self._errors.append("Pipeline not created")
            return False

        # Bus watch
        bus = self.pipeline.get_bus()
        if bus is not None:
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)

        # Mainloop
        self.mainloop = GLib.MainLoop() if GLib else None
        if self.mainloop:
            def _run_loop():
                try:
                    self.mainloop.run()
                except Exception as e:
                    self.logger.debug("GLib mainloop exited: %s", e)
            self.mainloop_thread = threading.Thread(target=_run_loop, name="DS8-GLib", daemon=True)
            self.mainloop_thread.start()

        # Set PLAYING
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._errors.append("Pipeline failed to enter PLAYING")
            self.logger.error("DS8 pipeline failed to start (PLAYING)")
            return False
        time.sleep(2.0)
        try:
            change_return, state, pending = self.pipeline.get_state(0)
            self.logger.info(f"Pipeline state after PLAYING: {state.value_name}, pending: {pending.value_name}")
            if state != Gst.State.PLAYING:
                self.logger.error("Pipeline failed to reach PLAYING")
        except Exception as e:
            self.logger.error("Pipeline state query failed: %s", e)

        self._start_time = time.time()
        self.running = True
        self._activated = True
        self.logger.info("DS8Adapter pipeline PLAYING with %d sources", len(self.source_info))
        return True

    def stop(self) -> None:
        self.running = False
        if not Gst or not self.pipeline:
            return
        try:
            self.pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass
        # Stop mainloop
        try:
            if self.mainloop:
                self.mainloop.quit()
        except Exception:
            pass
        # Best-effort join
        try:
            if self.mainloop_thread and self.mainloop_thread.is_alive():
                self.mainloop_thread.join(timeout=1.0)
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        elapsed = max(1e-3, time.time() - (self._start_time or time.time()))
        with self._frame_count_lock:
            fc = int(self._frame_count_total)
        fps = fc / elapsed if elapsed > 0 else 0.0
        buffers_flowing = fc > 0
        return {
            "prepared": self._prepared,
            "activated": self._activated,
            "errors": list(self._errors),
            "fps": fps,
            "sources": len(self.source_info),
            "buffers_flowing": buffers_flowing,
        }

    def set_trail_visualization(self, enabled: bool) -> None:
        # Compatibility no-op for UI toggle
        return

    # DS7 parity no-ops for UI-driven controls
    def update_confidence_threshold(self, confidence_threshold: float) -> bool:  # noqa: ARG002
        return False

    def update_iou_threshold(self, iou_threshold: float) -> bool:  # noqa: ARG002
        return False

    def set_detection_enabled(self, enabled: bool) -> bool:  # noqa: ARG002
        return False

    def update_target_classes(self, classes: List[int]) -> bool:  # noqa: ARG002
        return False

    def read_encoded_jpeg(self, sensor_id: int, timeout: float = 0.1) -> Tuple[bool, Optional[bytes]]:
        q = self.jpeg_queues.get(sensor_id)
        if not q:
            return False, None
        try:
            data = q.get(timeout=timeout)
            return (True, data) if data else (False, None)
        except queue.Empty:
            try:
                import time as _t
                now = _t.time()
                last = float(self._read_empty_last.get(sensor_id, 0.0))
                if now - last >= 5.0:
                    self.logger.debug("read_encoded_jpeg: queue empty for sensor_id=%s", sensor_id)
                    self._read_empty_last[sensor_id] = now
            except Exception:
                pass
            return False, None

    def read_mosaic_jpeg(self, timeout: float = 0.1) -> Tuple[bool, Optional[bytes]]:
        try:
            data = self.mosaic_queue.get(timeout=timeout)
            return (True, data) if data else (False, None)
        except queue.Empty:
            return False, None

    # -------------------- Internals --------------------
    def _get_logger(self):
        import logging
        return logging.getLogger("DS8Adapter")

    def _clean_name(self, name: str) -> str:
        s = (name or "").strip()
        if not s:
            return "camera"
        if "Living Room" in s:
            return "living-room"
        if "Kitchen" in s:
            return "kitchen"
        if "Family Room" in s:
            return "family-room"
        return s.lower().replace(" ", "-").replace("_", "-")

    def _make(self, factory: str, name: str):
        el = Gst.ElementFactory.make(factory, name)
        if not el:
            raise RuntimeError(f"Failed to create element {factory} ({name})")
        self._elements[name] = el
        return el

    def _set_q_leaky(self, q) -> None:
        try:
            q.set_property("max-size-buffers", 12)
            q.set_property("max-size-bytes", 0)
            q.set_property("leaky", 2)  # upstream
        except Exception:
            pass

    def _build_pipeline(self) -> None:
        self.pipeline = Gst.Pipeline.new("ds8_pipeline")
        if not self.pipeline:
            raise RuntimeError("Gst.Pipeline.new failed")

        # Elements: source and tee
        src = self._make("nvmultiurisrcbin", "src")
        tee = self._make("tee", "pre_tee")
        demux = self._make("nvstreamdemux", "pre_demux")
        # Add decoupling queues immediately after tee for robust scheduling
        q_to_demux = self._make("queue", "pre_to_demux_q")
        q_to_mosaic = self._make("queue", "pre_to_mosaic_q")
        self._set_q_leaky(q_to_demux)
        self._set_q_leaky(q_to_mosaic)

        # Optional preprocess (forced enabled when config present)
        preproc_cfg = str(getattr(self.config.processing, "DEEPSTREAM_PREPROCESS_CONFIG", "") or "")
        use_preprocess_cfg = bool(preproc_cfg)
        disable_preproc_env = False
        preproc = None
        preproc_caps = None
        if use_preprocess_cfg and preproc_cfg and os.path.exists(preproc_cfg):
            preproc = self._make("nvdspreprocess", "preprocess")
            try:
                preproc.set_property("config-file", os.path.abspath(preproc_cfg))
                self.logger.info(f"nvdspreprocess enabled with config {preproc_cfg}")
                preproc_caps = self._make("capsfilter", "preproc_caps")
                preproc_caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, width=640, height=640"))
            except Exception as e:
                self.logger.error(f"nvdspreprocess setup failed: {e}; falling back to nvinfer scaling")
                preproc = None
                preproc_caps = None
        if not preproc:
            self.logger.warning("Using nvinfer internal scaling (may drop frames)")

        # PGIE / tracker / analytics / tiler / encode branch
        pgie = self._make("nvinfer", "pgie")
        tracker = self._make("nvtracker", "tracker")
        analytics_cfg = self._select_analytics_cfg()
        try:
            analytics = self._make("nvdsanalytics", "analytics")
            analytics.set_property("config-file", analytics_cfg)
            self.logger.info("nvdsanalytics enabled with config %s", analytics_cfg)
        except Exception as e:
            raise RuntimeError(f"nvdsanalytics setup failed: {e}")

        mosaic_q = self._make("queue", "mosaic_q")
        tiler = self._make("nvmultistreamtiler", "mosaic_tiler")
        mosaic_osd = self._make("nvdsosd", "mosaic_osd")
        mosaic_conv_pre = self._make("nvvideoconvert", "mosaic_conv_pre")
        mosaic_caps_rgba = self._make("capsfilter", "mosaic_caps_rgba")
        mosaic_conv_post = self._make("nvvideoconvert", "mosaic_conv_post")
        mosaic_caps = self._make("capsfilter", "mosaic_caps")
        mosaic_enc = self._make("nvjpegenc", "mosaic_enc")
        mosaic_sink = self._make("appsink", "mosaic_sink")

        # Configure source properties
        uris = [str(self.source_info[i]["url"]) for i in sorted(self.source_info.keys())]
        uri_list = " ".join(uris)
        sensor_ids = ",".join(str(i) for i in sorted(self.source_info.keys()))
        try:
            src.set_property("uri-list", uri_list)
        except Exception:
            pass
        try:
            src.set_property("sensor-id-list", sensor_ids)
        except Exception:
            pass
        try:
            src.set_property("max-batch-size", len(self.source_info))
        except Exception:
            pass
        try:
            w = max(int(v["width"]) for v in self.source_info.values())
            h = max(int(v["height"]) for v in self.source_info.values())
        except Exception:
            w, h = 1920, 1080
        for k, v in ("width", w), ("height", h), ("live-source", 1), ("drop-pipeline-eos", 1):
            try:
                src.set_property(k, v)
            except Exception:
                pass
        for k, v in {
            "latency": 0,
            
        }.items():
            try:
                src.set_property(k, v)
            except Exception as e:
                self.logger.warning(f"RTSP prop {k} failed: {e}")
        try:
            src.set_property("batched-push-timeout", 50000)
            self.logger.info("Set batched-push-timeout to 50000")
        except Exception as e:
            self.logger.warning("Failed to set batched-push-timeout: %s", e)
        for k, v in {"cudadec-memtype": 0, "drop-frame-interval": 0}.items():
            try:
                src.set_property(k, v)
            except Exception as e:
                self.logger.warning(f"nvmultiurisrcbin prop {k} failed: {e}")

        try:
            self.pipeline.add(src)
            src.set_state(Gst.State.PAUSED)
            time.sleep(0.1)
            src.set_state(Gst.State.READY)
            self.logger.info("nvmultiurisrcbin pad negotiation complete")
        except Exception as e:
            self.logger.error(f"Source negotiation failed: {e}")

        # PGIE configuration with explicit parser validation
        nvinfer_ini = os.path.abspath(getattr(self.config.processing, "DEEPSTREAM_INFER_CONFIG", "pipelines/config_infer_primary_yolo11.ini"))
        custom_lib = "/opt/nvidia/deepstream/deepstream/lib/libnvdsparsebbox_yolo11.so"
        try:
            pgie.set_property("config-file-path", nvinfer_ini)
            self.logger.info(f"nvinfer config loaded: {nvinfer_ini}")
            self.logger.info(f"Custom YOLOv11 parser at: {custom_lib} (DS8 compatible)")
        except Exception as e:
            self.logger.error(f"nvinfer setup failed: {e}")
        try:
            pgie.set_property("max-batch-size", len(self.source_info))
        except Exception:
            pass
        tensor_meta_enabled = False
        try:
            pgie.set_property("input-tensor-meta", True)
            tensor_meta_enabled = True
        except Exception:
            try:
                pgie.set_property("output-tensor-meta", True)
                tensor_meta_enabled = True
            except Exception:
                pass
        if tensor_meta_enabled:
            self.logger.info("nvinfer ready with tensor-meta=1")

        # Tracker configuration (NvDCF) with forced absolute paths
        ll_cfg_raw = getattr(self.config.processing, "DEEPSTREAM_TRACKER_CONFIG", "")
        ll_lib_raw = getattr(self.config.processing, "DEEPSTREAM_TRACKER_LIB", "")
        ll_cfg = os.path.abspath(ll_cfg_raw) if ll_cfg_raw else ""
        ll_lib = os.path.abspath(ll_lib_raw) if ll_lib_raw else ""
        try:
            if ll_cfg:
                tracker.set_property("ll-config-file", ll_cfg)
                self.logger.info(f"nvtracker ll-config-file set to {ll_cfg}")
            if ll_lib:
                tracker.set_property("ll-lib-file", ll_lib)
                self.logger.info(f"nvtracker ll-lib-file set to {ll_lib}")
            tracker.set_property("enable-batch-process", 1)
            tracker.set_property("enable-past-frame", 1)
        except Exception as e:
            self.logger.error(f"nvtracker config load failed: {e}; using defaults (degraded mode)")

        # Tiler and encoder caps/props
        try:
            tiler.set_property("rows", 2)
            tiler.set_property("columns", 2)
        except Exception:
            pass
        try:
            tiler.set_property("width", w)
            tiler.set_property("height", h)
        except Exception:
            pass
        try:
            tiler.set_property("enable-padding", 1)
        except Exception:
            pass
        try:
            mosaic_osd.set_property("process-mode", 0)  # GPU mode
            mosaic_osd.set_property("display-text", 1)
        except Exception:
            pass
        try:
            mosaic_caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
        except Exception:
            pass
        try:
            mosaic_caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
        except Exception:
            pass
        try:
            jpeg_q = int(getattr(self.config.visualization, "JPEG_QUALITY", 85) or 85)
        except Exception:
            jpeg_q = 85
        try:
            mosaic_enc.set_property("quality", jpeg_q)
            mosaic_enc.set_property("preset-level", 1)
        except Exception:
            pass
        try:
            # Queue leaky upstream to avoid backpressure
            self._set_q_leaky(mosaic_q)
        except Exception:
            pass
        try:
            # Appsink common properties
            mosaic_sink.set_property("emit-signals", True)
            mosaic_sink.set_property("sync", False)
            mosaic_sink.set_property("drop", True)
            mosaic_sink.set_property("max-buffers", 1)
        except Exception:
            pass

        # Add base elements
        base_elements = [tee, q_to_demux, q_to_mosaic, demux]
        if preproc:
            base_elements.append(preproc)
        if preproc_caps:
            base_elements.append(preproc_caps)
        base_elements.extend([pgie, tracker, analytics, mosaic_q, tiler, mosaic_conv_pre, mosaic_caps_rgba, mosaic_osd, mosaic_conv_post, mosaic_caps, mosaic_enc, mosaic_sink])
        for el in base_elements:
            if el is not None:
                self.pipeline.add(el)

        # Link nvmultiurisrcbin -> pre_tee with dynamic pad-added handler and static attempt
        def _on_multiurisrc_pad_added(_bin, pad):
            try:
                sink = tee.get_static_pad("sink")
                if sink and not sink.is_linked():
                    result = pad.link(sink)
                    if result == Gst.PadLinkReturn.OK:
                        self.logger.info(f"Pad linked: src pad {pad.get_name()} to tee.sink")
                    else:
                        self.logger.error(f"Pad link failed: {Gst.PadLinkReturn.get_name(result)}")
            except Exception as e:
                self.logger.error("pad-added handler error: %s", e)
        try:
            src.connect("pad-added", _on_multiurisrc_pad_added)
        except Exception:
            self.logger.debug("Could not connect pad-added on nvmultiurisrcbin")
        # Try static link as well (harmless if dynamic-only)
        try:
            spad = src.get_static_pad("src")
            sink = tee.get_static_pad("sink")
            if spad and sink and not sink.is_linked():
                if spad.link(sink) == Gst.PadLinkReturn.OK:
                    self.logger.info("linked nvmultiurisrcbin → pre_tee (static)")
                else:
                    self.logger.warning("Static link nvmultiurisrcbin → pre_tee failed; waiting for pad-added")
        except Exception:
            self.logger.debug("Static pad link attempt failed; will rely on pad-added")

        # Tee branch 1: pre_tee → q_to_demux → pre_demux
        tee2demux = tee.get_request_pad("src_%u")
        demux_sink = demux.get_static_pad("sink")
        q1_sink = q_to_demux.get_static_pad("sink")
        q1_src = q_to_demux.get_static_pad("src")
        if tee2demux is None or demux_sink is None or q1_sink is None or q1_src is None:
            raise RuntimeError("Failed to get pads for tee->demux")
        if tee2demux.link(q1_sink) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Link failed for pre_tee(src) -> pre_to_demux_q(sink)")
        if q1_src.link(demux_sink) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Link failed for pre_to_demux_q(src) -> pre_demux(sink)")
        self.logger.info("linked pre_tee → pre_to_demux_q → pre_demux")

        # Tee branch 2: pre_tee → q_to_mosaic → [preproc] → nvinfer → nvtracker → nvdsanalytics_post → mosaic_q → tiler → osd → conv → caps → enc → sink
        tee2mosaic = tee.get_request_pad("src_%u")
        if tee2mosaic is None:
            raise RuntimeError("Failed to get tee src pad for mosaic")
        first_el = preproc if preproc else pgie
        first_sink_pad = first_el.get_static_pad("sink")
        q2_sink = q_to_mosaic.get_static_pad("sink")
        q2_src = q_to_mosaic.get_static_pad("src")
        if first_sink_pad is None or q2_sink is None or q2_src is None:
            raise RuntimeError("Failed to get sink pad for first mosaic element")
        if tee2mosaic.link(q2_sink) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Link failed for pre_tee(src) -> pre_to_mosaic_q(sink)")
        if q2_src.link(first_sink_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Link failed for pre_to_mosaic_q(src) -> {first_el.name}(sink)")
        else:
            if preproc:
                self.logger.info("linked pre_tee → pre_to_mosaic_q → nvdspreprocess")
            else:
                self.logger.info("linked pre_tee → pre_to_mosaic_q → nvinfer")

        # Link mosaic chain after analytics
        if preproc and preproc_caps:
            chain = [preproc, preproc_caps, pgie, tracker, analytics, mosaic_q, tiler, mosaic_conv_pre, mosaic_caps_rgba, mosaic_osd, mosaic_conv_post, mosaic_caps, mosaic_enc, mosaic_sink]
        else:
            chain = [pgie, tracker, analytics, mosaic_q, tiler, mosaic_conv_pre, mosaic_caps_rgba, mosaic_osd, mosaic_conv_post, mosaic_caps, mosaic_enc, mosaic_sink]
        for a, b in zip(chain, chain[1:]):
            if a is None or b is None:
                continue
            if not a.link(b):
                raise RuntimeError(f"Failed to link {a.name} -> {b.name}")
            else:
                if a is analytics and b is mosaic_q:
                    self.logger.info("linked nvdsanalytics_post → mosaic_q")
                if a is preproc_caps and b is pgie:
                    self.logger.info("Linked preproc → caps → nvinfer")

        # Per-stream MDE branches from demux using requested pads src_i
        for idx in sorted(self.source_info.keys()):
            q = self._make("queue", f"mde_{idx}queue")
            self._set_q_leaky(q)
            conv = self._make("nvvideoconvert", f"mde{idx}conv")
            caps = self._make("capsfilter", f"mde{idx}caps")
            enc = self._make("nvjpegenc", f"mde{idx}enc")
            sink = self._make("appsink", f"mde_{idx}_sink")

            try:
                caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
            except Exception:
                pass
            try:
                enc.set_property("quality", jpeg_q)
                enc.set_property("preset-level", 1)
            except Exception:
                pass
            try:
                sink.set_property("emit-signals", True)
                sink.set_property("sync", False)
                sink.set_property("drop", True)
                sink.set_property("max-buffers", 1)
            except Exception:
                pass

            # Add and link
            for el in [q, conv, caps, enc, sink]:
                self.pipeline.add(el)
            for a, b in zip([q, conv, caps, enc], [conv, caps, enc, sink]):
                if not a.link(b):
                    raise RuntimeError(f"Failed to link {a.name} -> {b.name}")

            # Link demux requested pad -> q.sink
            demux_src = demux.get_request_pad(f"src_{idx}")
            if demux_src is None:
                raise RuntimeError(f"Failed to request demux src pad src_{idx}")
            sinkpad = q.get_static_pad("sink")
            if sinkpad is None:
                raise RuntimeError(f"Failed to get sink pad for {q.name}")
            if demux_src.link(sinkpad) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Link failed for demux(src_{idx}) -> {q.name}(sink)")
            else:
                self.logger.info("linked pre_demux.src_%d → %s.sink", idx, q.name)

            # Connect appsink callback for this sensor index
            try:
                sink.connect("new-sample", self._on_new_jpeg_sample, idx)
            except Exception:
                raise RuntimeError(f"Failed to connect appsink new-sample for sensor {idx}")

        # Appsink callback for mosaic
        mosaic_sink.connect("new-sample", self._on_new_mosaic_sample)

    # -- Callbacks --
    def _on_new_jpeg_sample(self, appsink: GstApp.AppSink, sensor_id: int):
        sample = None
        buf = None
        mapinfo = None
        try:
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.ERROR
            buf = sample.get_buffer()
            if not buf:
                return Gst.FlowReturn.ERROR
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                return Gst.FlowReturn.ERROR
            data = bytes(mapinfo.data) if mapinfo and mapinfo.data else None
            if data:
                q = self.jpeg_queues.get(int(sensor_id))
                if q:
                    try:
                        q.put_nowait(data)
                        with self._frame_count_lock:
                            self._frame_count_total += 1
                        self._frame_counters[int(sensor_id)] = self._frame_counters.get(int(sensor_id), 0) + 1
                        frame_count = self._frame_counters[int(sensor_id)]
                        if frame_count % 300 == 0:
                            self.logger.info(f"MDE sensor {sensor_id}: {frame_count} frames")
                        if not self._mde_first_log.get(int(sensor_id), False):
                            self.logger.info("MDE sink %s first frame: %d bytes", int(sensor_id), len(data))
                            self._mde_first_log[int(sensor_id)] = True
                    except queue.Full:
                        pass
        except Exception as e:
            self.logger.debug("jpeg appsink error: %s", e)
            return Gst.FlowReturn.ERROR
        finally:
            try:
                if buf is not None and mapinfo is not None:
                    buf.unmap(mapinfo)
            except Exception:
                pass
            try:
                if sample is not None:
                    sample.unref()
            except Exception:
                pass
        return Gst.FlowReturn.OK

    def _on_new_mosaic_sample(self, appsink: GstApp.AppSink):
        sample = None
        buf = None
        mapinfo = None
        try:
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.ERROR
            buf = sample.get_buffer()
            if not buf:
                return Gst.FlowReturn.ERROR
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                return Gst.FlowReturn.ERROR
            data = bytes(mapinfo.data) if mapinfo and mapinfo.data else None
            if data:
                try:
                    self.mosaic_queue.put_nowait(data)
                    self._mosaic_counter += 1
                    if self._mosaic_counter % 300 == 0:
                        self.logger.info(f"Mosaic: {self._mosaic_counter} frames")
                    if not self._mosaic_first_logged:
                        self.logger.info("Mosaic first frame: %d bytes", len(data))
                        self._mosaic_first_logged = True
                except queue.Full:
                    # Drop if back-pressured
                    pass
        except Exception as e:
            self.logger.debug("mosaic appsink error: %s", e)
            return Gst.FlowReturn.ERROR
        finally:
            try:
                if buf is not None and mapinfo is not None:
                    buf.unmap(mapinfo)
            except Exception:
                pass
            try:
                if sample is not None:
                    sample.unref()
            except Exception:
                pass
        return Gst.FlowReturn.OK

    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            msg = f"GstError: {err} debug={dbg or ''}"
            self._errors.append(msg)
            self.logger.error(msg)
        elif t == Gst.MessageType.WARNING:
            warn, dbg = message.parse_warning()
            self.logger.warning(f"GstWarning: {warn} debug={dbg or ''}")
        elif t == Gst.MessageType.INFO:
            info, dbg = message.parse_info()
            self.logger.info(f"GstInfo: {info} debug={dbg or ''}")
        elif t == Gst.MessageType.EOS:
            self.logger.info("Pipeline received EOS")
        return True

    # Helpers
    def _select_analytics_cfg(self) -> str:
        """Select an nvdsanalytics plugin config based on stage placement.

        Rules requested by user:
        - If analytics is directly after PGIE, use the "_exclude" variant.
        - If analytics is after tracker, use the "_post" variant.
        We place analytics after tracker in this pipeline, so prefer
        pipelines/config_nvdsanalytics_post.ini.
        Fallbacks: pipelines/config_nvdsanalytics_exclude.ini, pipelines/config_nvdsanalytics.ini.
        Raises if no suitable file exists.
        """
        # Determine stage: our graph is pgie -> tracker -> analytics
        stage = "post_tracker"
        search: List[str] = []
        if stage == "post_tracker":
            search = [
                "pipelines/config_nvdsanalytics_post.ini",
                "pipelines/config_nvdsanalytics.ini",
            ]
        else:  # direct after PGIE
            search = [
                "pipelines/config_nvdsanalytics_exclude.ini",
                "pipelines/config_nvdsanalytics.ini",
            ]
        for rel in search:
            p = os.path.abspath(rel)
            if os.path.exists(p):
                return p
        raise RuntimeError(
            "nvdsanalytics config not found; expected one of: " + ", ".join(search)
        )
