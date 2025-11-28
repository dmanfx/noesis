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
  nvmultiurisrcbin -> [optional nvdspreprocess] -> nvinfer -> nvtracker -> [optional nvdsanalytics]
      -> nvmultistreamtiler -> nvvideoconvert -> caps(I420, NVMM) -> nvjpegenc -> appsink_mosaic
      -> (optional) nveglglessink display branch

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
import configparser


import gi  # type: ignore
gi.require_version("GstApp", "1.0")
gi.require_version("Gst", "1.0")
gi.require_version("GstRtsp", "1.0")
from gi.repository import Gst, GLib, GstApp, GstRtsp  # type: ignore

import yaml



class DS8Adapter:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.logger = self._get_logger()

        # Require deepstream-test5 style config for nvmultiurisrcbin
        self.ds_multiurisrc_cfg = str(getattr(self.config.processing, "DS_MULTIURISRC_CONFIG", "") or "").strip()
        if not self.ds_multiurisrc_cfg or not os.path.exists(self.ds_multiurisrc_cfg):
            raise RuntimeError("DS_MULTIURISRC_CONFIG missing or not found; provide a deepstream-test5 style config file")

        # Minimal multi-URI config: load config file and parse URIs from [source-list] list=
        self._multiuri_conf = self._load_multiuri_config(self.ds_multiurisrc_cfg)
        self._uris = self._parse_rtsp_list(self._multiuri_conf, self.ds_multiurisrc_cfg)
        sl = self._multiuri_conf.get('source-list', {})
        smx = self._multiuri_conf.get('streammux', {})
        self._multiuri_ip = sl.get('http-ip', 'localhost')
        self._multiuri_port = int(sl.get('http-port', '9010')) if 'http-port' in sl else 9010
        self._multiuri_max_batch = int(sl.get('max-batch-size', str(len(self._uris) or 1)))
        self._rtp_proto = int(self._multiuri_conf.get('source-attr-all', {}).get('select-rtp-protocol', '4'))
        self._multiuri_width = int(smx.get('width', '1920')) if 'width' in smx else 1920
        self._multiuri_height = int(smx.get('height', '1080')) if 'height' in smx else 1080
        self._multiuri_bpt = int(smx.get('batched-push-timeout', '40000')) if 'batched-push-timeout' in smx else 40000

        # Canonical per-source info (sensor ids 0..N-1), built from config file
        uris = self._uris
        width = self._multiuri_width
        height = self._multiuri_height
        self.source_info: Dict[int, Dict[str, Any]] = {}
        for idx, uri in enumerate(uris):
            name = f"Camera {idx+1}"
            self.source_info[idx] = {
                "name": name,
                "clean_name": self._clean_name(name),
                "url": uri,
                "width": width,
                "height": height,
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

        # Lightweight probe counters for flow tracing
        self._probe_counts: Dict[str, int] = {}
        self._probe_last_log: Dict[str, float] = {}


        # Allow attaching external publisher(s) for parity with DS7 (no-op here)
        self.occupancy_publisher = None
        # REST seeding guard
        self._rest_seeded = False

        # Cache of elements we care to watch for state-changes
        self._state_watch_names = {
            "src",
            "preprocess",
            "pgie",
            "tracker",
            "analytics_post",
            "post_analytics_tee",
            "mosaic_q",
            "mosaic_tiler",
            "mosaic_conv_pre",
            "mosaic_caps_rgba",
            "mosaic_osd",
            "mosaic_conv_post",
            "mosaic_caps",
            "mosaic_enc",
            "mosaic_sink",
            "egl_q",
            "egl_tiler",
            "egl_conv",
            "egl_caps_rgba",
            "egl_sink",
        }

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

        # Initialize GStreamer – fail fast if unavailable
        Gst.init(None)

        # Log required plugin availability upfront for quicker diagnostics
        # Log plugin availability (non-fatal)
        self._log_required_plugins()

        # Build pipeline – let exceptions surface
        self._build_pipeline()
        self._prepared = True

        if not self.pipeline:
            self._errors.append("Pipeline not created")
            return False

        # Bus watch
        bus = self.pipeline.get_bus()
        if bus is None:
            raise RuntimeError("Failed to acquire GstBus from pipeline")
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        # Mainloop
        self.mainloop = GLib.MainLoop() if GLib else None
        if self.mainloop:
            def _run_loop():
                self.mainloop.run()
            self.mainloop_thread = threading.Thread(target=_run_loop, name="DS8-GLib", daemon=True)
            self.mainloop_thread.start()

        # Set PLAYING
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("DS8 pipeline failed to enter PLAYING state")
        # Give the pipeline a moment to transition and then query with a real timeout
        time.sleep(1.5)
        # Wait up to 5 seconds for the state change to complete
        change_return, state, pending = self.pipeline.get_state(5 * Gst.SECOND)
        self.logger.info(f"Pipeline state after PLAYING: {state.value_name}, pending: {pending.value_name}")
        if state != Gst.State.PLAYING:
            # Many live/RTSP pipelines report ASYNC/NO_PREROLL and take longer to settle.
            # Treat this as a transitional condition instead of an error if we're headed to PLAYING.
            try:
                pending_name = pending.value_name  # may raise if pending is None
            except Exception:
                pending_name = str(pending)
            if pending == Gst.State.PLAYING or change_return in (
                Gst.StateChangeReturn.ASYNC,
                Gst.StateChangeReturn.NO_PREROLL,
            ):
                self.logger.warning(
                    "Pipeline still transitioning to PLAYING (state=%s, pending=%s)",
                    state.value_name,
                    pending_name,
                )
            else:
                raise RuntimeError("Pipeline did not reach PLAYING")

        # Static uri-list already applied on the element; nothing else to do before reporting state

        self._start_time = time.time()
        self.running = True
        self._activated = True
        self.logger.info("DS8Adapter pipeline PLAYING with %d sources", len(self.source_info))
        return True

    def stop(self) -> None:
        self.running = False
        if not Gst or not self.pipeline:
            return
        self.pipeline.set_state(Gst.State.NULL)
        # Stop mainloop
        if self.mainloop:
            self.mainloop.quit()
        # Best-effort join
        if self.mainloop_thread and self.mainloop_thread.is_alive():
            self.mainloop_thread.join(timeout=1.0)

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
            import time as _t
            now = _t.time()
            last = float(self._read_empty_last.get(sensor_id, 0.0))
            if now - last >= 5.0:
                self.logger.debug(
                    "read_encoded_jpeg: queue empty for sensor_id=%s", sensor_id
                )
                self._read_empty_last[sensor_id] = now
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

    def _attach_flow_probe(self, pad: Optional[Gst.Pad], label: str, period_sec: float = 5.0) -> None:
        """Attach a safe buffer probe that logs INFO on first buffer and every period.

        - Never maps or copies buffer data
        - Throttles to avoid log spam
        """
        if pad is None:
            self.logger.warning(f"Probe attach skipped: pad is None for {label}")
            return
        def _cb(_pad, _info, self_ref=self, lbl=label, per=period_sec):
            now = time.time()
            cnt = self_ref._probe_counts.get(lbl, 0) + 1
            self_ref._probe_counts[lbl] = cnt
            last = float(self_ref._probe_last_log.get(lbl, 0.0))
            if cnt == 1 or (now - last) >= per:
                self_ref.logger.info("flow %s: %d", lbl, cnt)
                self_ref._probe_last_log[lbl] = now
            return Gst.PadProbeReturn.OK
        pad.add_probe(Gst.PadProbeType.BUFFER | Gst.PadProbeType.BUFFER_LIST, _cb)
        # Also trace CAPS/EVENTS to verify negotiation (avoid value_nick which can segfault)
        def _ev(_pad, info, self_ref=self, lbl=label):
            try:
                ev = info.get_event()
            except Exception:
                return Gst.PadProbeReturn.OK
            if not ev:
                return Gst.PadProbeReturn.OK
            return Gst.PadProbeReturn.OK
        pad.add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, _ev)

    

    def _load_multiuri_config(self, path: str) -> Dict[str, Dict[str, str]]:
        cp = configparser.ConfigParser(strict=False)
        cp.read(path)
        data: Dict[str, Dict[str, str]] = {}
        for sec in cp.sections():
            data[sec] = {k: v for k, v in cp.items(sec)}
        return data

    def _parse_rtsp_list(self, conf: Dict[str, Dict[str, str]], path: str) -> List[str]:
        if 'source-list' not in conf or 'list' not in conf['source-list']:
            raise RuntimeError(f"Missing [source-list].list in {path}")
        raw = conf['source-list']['list'].strip()
        raw = raw.replace(',', ';').replace('\n', ';')
        uris = [u.strip() for u in raw.split(';') if u.strip()]
        if not uris:
            raise RuntimeError(f"No URIs found in [source-list] list= of {path}")
        return uris

    def _attach_osd_text_probe(self, osd_el) -> None:
        """Append confidence to OSD text safely on the OSD sink pad.

        Updates NvDsObjectMeta.text_params.display_text to include
        "<label> <conf> ID:<id>" for each object. Probe must not raise.
        """
        if not osd_el:
            return
        pad = None
        try:
            pad = osd_el.get_static_pad("sink")
        except Exception:
            pad = None
        if pad is None:
            return

        def _cb(_pad, info, self_ref=self):
            try:
                buf = info.get_buffer()
                if not buf:
                    return Gst.PadProbeReturn.OK
                try:
                    import pyds  # type: ignore
                except Exception:
                    return Gst.PadProbeReturn.OK
                try:
                    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
                except Exception:
                    batch_meta = None
                if not batch_meta:
                    return Gst.PadProbeReturn.OK
                l_frame = getattr(batch_meta, 'frame_meta_list', None)
                while l_frame:
                    try:
                        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                    except Exception:
                        l_frame = l_frame.next
                        continue
                    l_obj = frame_meta.obj_meta_list
                    while l_obj:
                        try:
                            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                            # Build "label conf ID" text
                            label = getattr(obj_meta, 'obj_label', '') or ''
                            conf = float(getattr(obj_meta, 'confidence', 0.0) or 0.0)
                            oid = int(getattr(obj_meta, 'object_id', -1) or -1)
                            txt = f"{label} {conf:.2f} ID:{oid}".strip()
                            # Assign display text directly (prefer plain string in DS8 Python bindings)
                            tp = obj_meta.text_params
                            tp.display_text = txt
                            # Increase font size and choose a readable font
                            try:
                                tp.font_params.font_name = "Sans"
                            except Exception:
                                pass
                            try:
                                tp.font_params.font_size = 22
                            except Exception:
                                pass
                            # Keep background disabled to avoid occlusion
                            try:
                                tp.set_bg_clr = 0
                            except Exception:
                                pass
                            obj_meta.text_params = tp
                        except Exception:
                            pass
                        l_obj = l_obj.next
                    l_frame = l_frame.next
            except Exception:
                # Never propagate from probe
                pass
            return Gst.PadProbeReturn.OK

        try:
            pad.add_probe(Gst.PadProbeType.BUFFER, _cb)
        except Exception:
            pass

    def _build_pipeline(self) -> None:
        self.pipeline = Gst.Pipeline.new("ds8_pipeline")
        if not self.pipeline:
            raise RuntimeError("Gst.Pipeline.new failed")

        # Elements: source
        src = self._make("nvmultiurisrcbin", "src")

        # Optional preprocess (forced enabled when config present)
        preproc_cfg = str(getattr(self.config.processing, "DEEPSTREAM_PREPROCESS_CONFIG", "") or "")
        preproc = None
        preproc_caps = None
        if preproc_cfg and os.path.exists(preproc_cfg):
            # Honor enable=0 inside the preproc config
            cp = configparser.ConfigParser(interpolation=None, delimiters=("="))
            cp.read(preproc_cfg)
            enabled = 1
            if cp.has_section("property") and cp.has_option("property", "enable"):
                try:
                    enabled = int(cp.get("property", "enable") or "1")
                except Exception:
                    enabled = 1
            if enabled:
                preproc = self._make("nvdspreprocess", "preprocess")
                preproc.set_property("config-file", os.path.abspath(preproc_cfg))
                self.logger.info(f"nvdspreprocess enabled with config {preproc_cfg}")
            else:
                self.logger.info(f"nvdspreprocess disabled via config {preproc_cfg} (enable=0)")
        if not preproc:
            self.logger.info("Using nvinfer internal scaling (preprocess disabled)")

        # PGIE / tracker / analytics / tiler / encode branch
        pgie = self._make("nvinfer", "pgie")
        # Configure nvinfer with primary config
        infer_cfg = str(getattr(self.config.processing, "DEEPSTREAM_INFER_CONFIG", "") or "").strip()
        if infer_cfg:
            infer_cfg_abs = os.path.abspath(infer_cfg)
            if not os.path.exists(infer_cfg_abs):
                raise RuntimeError(f"nvinfer config not found: {infer_cfg_abs}")
            try:
                pgie.set_property("config-file-path", infer_cfg_abs)
            except Exception:
                pgie.set_property("config-file", infer_cfg_abs)
            self.logger.info("nvinfer configured with %s", infer_cfg_abs)
        else:
            raise RuntimeError("DEEPSTREAM_INFER_CONFIG is empty; set processing.DEEPSTREAM_INFER_CONFIG")
        tracker = self._make("nvtracker", "tracker")
        # Configure nvtracker (library + YAML config) if provided
        try:
            trk_lib = str(getattr(self.config.processing, "DEEPSTREAM_TRACKER_LIB", "") or "").strip()
            if trk_lib:
                tracker.set_property("ll-lib-file", trk_lib)
                self.logger.info("nvtracker ll-lib-file set: %s", trk_lib)
        except Exception:
            pass
        try:
            trk_cfg = str(getattr(self.config.processing, "DEEPSTREAM_TRACKER_CONFIG", "") or "").strip()
            if trk_cfg:
                trk_cfg_abs = os.path.abspath(trk_cfg)
                if os.path.exists(trk_cfg_abs):
                    tracker.set_property("ll-config-file", trk_cfg_abs)
                    self.logger.info("nvtracker ll-config-file set: %s", trk_cfg_abs)
                else:
                    self.logger.warning("nvtracker config not found: %s", trk_cfg_abs)
        except Exception:
            pass
        analytics_cfg = self._select_analytics_cfg()
        try:
            analytics = self._make("nvdsanalytics", "analytics")
            analytics.set_property("config-file", analytics_cfg)
            self.logger.info("nvdsanalytics enabled with config %s", analytics_cfg)
        except Exception as e:
            raise RuntimeError(f"nvdsanalytics setup failed: {e}")

        mosaic_q = self._make("queue", "mosaic_q")
        tiler = self._make("nvmultistreamtiler", "mosaic_tiler")
        # Minimal mode: skip mosaic/JPEG branch while tracing flow
        mosaic_osd = self._make("nvdsosd", "mosaic_osd")
        mosaic_conv_pre = self._make("nvvideoconvert", "mosaic_conv_pre")
        mosaic_caps_rgba = self._make("capsfilter", "mosaic_caps_rgba")
        mosaic_conv_post = self._make("nvvideoconvert", "mosaic_conv_post")
        mosaic_caps = self._make("capsfilter", "mosaic_caps")
        mosaic_enc = self._make("nvjpegenc", "mosaic_enc")
        mosaic_sink = self._make("appsink", "mosaic_sink")
        post_analytics_tee = self._make("tee", "post_analytics_tee")
        pre_video_tee = None

        disable_egl = str(os.environ.get("NOESIS_DISABLE_EGL", "0")).strip().lower() in {"1", "true", "yes", "y"}
        if disable_egl:
            self.logger.info("EGL display branch disabled via NOESIS_DISABLE_EGL")
        egl_q = egl_tiler = egl_conv = egl_caps_rgba = egl_sink = None
        if not disable_egl:
            try:
                egl_q = self._make("queue", "egl_q")
                egl_sink = self._make("nveglglessink", "egl_sink")
                self.logger.info("EGL display branch enabled (nveglglessink created)")
            except Exception as exc:
                self.logger.warning("Disabling EGL display branch: %s", exc)
                disable_egl = True
                egl_q = egl_tiler = egl_conv = egl_caps_rgba = egl_sink = None


        # Configure source properties
        # Apply nvmultiurisrcbin configuration from file (template-style)
        uris = list(self._uris)
        batch_size = max(self._multiuri_max_batch, len(uris)) if uris else self._multiuri_max_batch
        src.set_property("max-batch-size", int(batch_size))
        w, h = int(self._multiuri_width), int(self._multiuri_height)
        for key, value in (("width", w), ("height", h), ("live-source", 1)):
            src.set_property(key, value)
        src.set_property("batched-push-timeout", int(self._multiuri_bpt))
        # REST control and RTSP transport hints
        src.set_property("ip-address", str(self._multiuri_ip))
        src.set_property("port", int(self._multiuri_port))
        src.set_property("select-rtp-protocol", int(self._rtp_proto or 4))

        # Apply static URI list and sensor IDs like the template service
        uri_list = ",".join(uris)
        sid_list = ",".join(str(i) for i in range(len(uris)))
        src.set_property("uri-list", uri_list)
        src.set_property("sensor-id-list", sid_list)
        self.logger.info("Configured nvmultiurisrcbin with %d URIs (uri-list set)", len(uris))
        self.logger.info("URIs: %s", "; ".join(uris))

        self.pipeline.add(src)

        # Minimal pre-inference path: mirror template (no nvinfer/tracker/analytics in this link path)
        minimal_pre_infer = True

        # Tiler and encoder caps/props
        tiler.set_property("rows", 1)
        tiler.set_property("columns", 3)
        tiler.set_property("width", w)
        tiler.set_property("height", h)
        mosaic_osd.set_property("process-mode", 0)  # GPU mode
        mosaic_osd.set_property("display-text", 1)
        mosaic_osd.set_property("display-bbox", 1)
        mosaic_caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
        mosaic_caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
        try:
            jpeg_q = int(getattr(self.config.visualization, "JPEG_QUALITY", 85) or 85)
        except Exception:
            jpeg_q = 85
        mosaic_enc.set_property("quality", jpeg_q)
        # Queue leaky upstream to avoid backpressure
        self._set_q_leaky(mosaic_q)
        # Appsink common properties
        mosaic_sink.set_property("emit-signals", True)
        mosaic_sink.set_property("sync", False)
        mosaic_sink.set_property("drop", True)
        mosaic_sink.set_property("max-buffers", 1)
        # Wire appsink to push encoded mosaic JPEGs into the queue for WebSocket broadcast
        try:
            mosaic_sink.connect("new-sample", self._on_new_mosaic_sample)
            self.logger.info("mosaic appsink new-sample connected")
        except Exception:
            # Non-fatal: leave mosaic broadcast disabled if connect fails
            self.logger.warning("Unable to connect mosaic appsink new-sample")

        if not disable_egl and egl_q is not None:
            self._set_q_leaky(egl_q)
            try:
                egl_sink.set_property("sync", False)
                egl_sink.set_property("qos", False)
            except Exception:
                pass

        # Add base elements
        base_elements = []
        # Always include mosaic chain
        base_elements.extend([mosaic_q, tiler, mosaic_conv_pre, mosaic_caps_rgba, mosaic_osd, mosaic_conv_post, mosaic_caps, mosaic_enc, mosaic_sink])
        # Only add infer/analytics chain when not in minimal mode
        if not minimal_pre_infer:
            if preproc:
                base_elements.append(preproc)
            base_elements.extend([pgie, tracker, analytics, post_analytics_tee])
        if not disable_egl and egl_q and egl_sink:
            base_elements.extend([egl_q, egl_sink])
        for el in base_elements:
            if el is not None:
                self.pipeline.add(el)
        # Direct link in minimal path: nvmultiurisrcbin → nvvideoconvert(pre_video) → nvinfer → nvtracker → nvdsanalytics → [tee|mosaic_q]
        if minimal_pre_infer:
            pre_video = self._make("nvvideoconvert", "pre_video")
            # Ensure pgie is added in minimal path
            self.pipeline.add(pre_video)
            self.pipeline.add(pgie)
            self.pipeline.add(tracker)
            self.pipeline.add(analytics)
            if not src.link(pre_video):
                raise RuntimeError("Failed to link nvmultiurisrcbin -> pre_video")
            pre_video_sink_pad = pre_video.get_static_pad("sink")
            # Insert nvinfer between pre_video and downstream branches
            if not pre_video.link(pgie):
                raise RuntimeError("Failed to link pre_video -> nvinfer")
            if not pgie.link(tracker):
                raise RuntimeError("Failed to link nvinfer -> nvtracker")
            if not tracker.link(analytics):
                raise RuntimeError("Failed to link nvtracker -> nvdsanalytics")
            # Build shared display chain: analytics → tiler → conv_pre → caps_rgba → osd → tee
            # Link analytics to shared tiler
            if not analytics.link(tiler):
                raise RuntimeError("Failed to link nvdsanalytics -> mosaic_tiler (shared)")
            shared_chain = [tiler, mosaic_conv_pre, mosaic_caps_rgba, mosaic_osd]
            for a, b in zip(shared_chain, shared_chain[1:]):
                if not a.link(b):
                    raise RuntimeError(f"Failed to link {a.name} -> {b.name}")
            # Attach OSD text probe to add confidence to display text
            self._attach_osd_text_probe(mosaic_osd)
            try:
                self.pipeline.add(post_analytics_tee)
            except Exception:
                pass
            if not mosaic_osd.link(post_analytics_tee):
                raise RuntimeError("Failed to link mosaic_osd -> post_analytics_tee")
            # Branch to sinks: JPEG and EGL
            tee2mosaic = post_analytics_tee.get_request_pad("src_%u")
            mosaic_sink_pad = mosaic_q.get_static_pad("sink")
            if tee2mosaic is None or mosaic_sink_pad is None or tee2mosaic.link(mosaic_sink_pad) != Gst.PadLinkReturn.OK:
                raise RuntimeError("Failed to link post_analytics_tee to mosaic_q")
            self._attach_flow_probe(tee2mosaic, "post_analytics_tee → mosaic_q")
            if not disable_egl and egl_q and egl_sink:
                tee2egl = post_analytics_tee.get_request_pad("src_%u")
                egl_sink_pad = egl_q.get_static_pad("sink")
                if tee2egl is None or egl_sink_pad is None or tee2egl.link(egl_sink_pad) != Gst.PadLinkReturn.OK:
                    raise RuntimeError("Failed to link post_analytics_tee to egl_q")
                self._attach_flow_probe(tee2egl, "post_analytics_tee → egl_q")
            else:
                # No EGL; still build shared chain but link directly to mosaic branch
                if not analytics.link(tiler):
                    raise RuntimeError("Failed to link nvdsanalytics -> mosaic_tiler (shared)")
                for a, b in zip([tiler, mosaic_conv_pre, mosaic_caps_rgba, mosaic_osd], [mosaic_conv_pre, mosaic_caps_rgba, mosaic_osd, None]):
                    if b is None:
                        break
                    if not a.link(b):
                        raise RuntimeError(f"Failed to link {a.name} -> {b.name}")
                # Attach OSD text probe (no tee in this branch)
                self._attach_osd_text_probe(mosaic_osd)
                if not mosaic_osd.link(mosaic_q):
                    raise RuntimeError("Failed to link mosaic_osd -> mosaic_q")
                self.logger.info("linked analytics → shared_tiler/osd → mosaic_q (no EGL)")
            if not (not disable_egl and egl_q and egl_sink):
                pass
            else:
                self.logger.info("linked analytics → shared_tiler/osd → tee → [mosaic|egl]")
            self._attach_flow_probe(src.get_static_pad("src"), "nvmultiurisrcbin.src")
            if pre_video_sink_pad:
                self._attach_flow_probe(pre_video_sink_pad, "pre_video.sink")
            # Add probes on nvinfer pads for visibility
            try:
                self._attach_flow_probe(pgie.get_static_pad("sink"), "nvinfer.sink")
            except Exception:
                pass
            try:
                self._attach_flow_probe(pgie.get_static_pad("src"), "nvinfer.src")
            except Exception:
                pass
            try:
                self._attach_flow_probe(tracker.get_static_pad("sink"), "nvtracker.sink")
            except Exception:
                pass
            try:
                self._attach_flow_probe(tracker.get_static_pad("src"), "nvtracker.src")
            except Exception:
                pass
            try:
                self._attach_flow_probe(analytics.get_static_pad("sink"), "nvdsanalytics.sink")
            except Exception:
                pass
            try:
                self._attach_flow_probe(analytics.get_static_pad("src"), "nvdsanalytics.src")
            except Exception:
                pass
            
        else:
            # Direct link: nvmultiurisrcbin → [preprocess]|nvinfer
            first_el = preproc if preproc else pgie
            if not src.link(first_el):
                raise RuntimeError(f"Failed to link nvmultiurisrcbin -> {first_el.name}")
            self.logger.info(f"linked nvmultiurisrcbin → {first_el.name}")
            self._attach_flow_probe(src.get_static_pad("src"), "nvmultiurisrcbin.src")
            self._attach_flow_probe(first_el.get_static_pad("sink"), f"{first_el.name}.sink")
            # Also trace PGIE sink to confirm infer branch is receiving frames
            pgie_sink_pad = pgie.get_static_pad("sink")
            self._attach_flow_probe(pgie_sink_pad, "nvinfer.sink")
        # Add deeper probes along mosaic chain to pinpoint stalls
        if not minimal_pre_infer:
            self._attach_flow_probe(pgie.get_static_pad("src"), "nvinfer.src")
            self._attach_flow_probe(tracker.get_static_pad("sink"), "nvtracker.sink")
            self._attach_flow_probe(tracker.get_static_pad("src"), "nvtracker.src")
            self._attach_flow_probe(analytics.get_static_pad("sink"), "nvdsanalytics.sink")
            self._attach_flow_probe(analytics.get_static_pad("src"), "nvdsanalytics.src")
        self._attach_flow_probe(mosaic_q.get_static_pad("src"), "mosaic_q.src")
        self._attach_flow_probe(tiler.get_static_pad("sink"), "mosaic_tiler.sink")
        self._attach_flow_probe(tiler.get_static_pad("src"), "mosaic_tiler.src")
        self._attach_flow_probe(mosaic_conv_pre.get_static_pad("sink"), "mosaic_conv_pre.sink")
        self._attach_flow_probe(mosaic_caps_rgba.get_static_pad("src"), "mosaic_caps_rgba.src")
        self._attach_flow_probe(mosaic_osd.get_static_pad("src"), "mosaic_osd.src")
        self._attach_flow_probe(mosaic_conv_post.get_static_pad("sink"), "mosaic_conv_post.sink")
        self._attach_flow_probe(mosaic_caps.get_static_pad("src"), "mosaic_caps.src")
        self._attach_flow_probe(mosaic_enc.get_static_pad("sink"), "mosaic_enc.sink")
        self._attach_flow_probe(mosaic_sink.get_static_pad("sink"), "mosaic_appsink.sink")

        # Link core chain only when not minimal
        if not minimal_pre_infer:
            if preproc:
                core_chain = [preproc, pgie, tracker, analytics]
            else:
                core_chain = [pgie, tracker, analytics]
            for a, b in zip(core_chain, core_chain[1:]):
                if not a.link(b):
                    raise RuntimeError(f"Failed to link {a.name} -> {b.name}")
                elif preproc and a is preproc and b is pgie:
                    self.logger.info("Linked nvdspreprocess → nvinfer (tensor meta)")
            if not analytics.link(post_analytics_tee):
                raise RuntimeError("Failed to link nvdsanalytics_post to post_analytics_tee")

        # Branch 1: mosaic appsink/WebSocket path
        if not minimal_pre_infer:
            tee2mosaic_post = post_analytics_tee.get_request_pad("src_%u")
            mosaic_sink_pad = mosaic_q.get_static_pad("sink")
            if tee2mosaic_post is None or mosaic_sink_pad is None:
                raise RuntimeError("Failed to prepare post-analytics mosaic branch pads")
            if tee2mosaic_post.link(mosaic_sink_pad) != Gst.PadLinkReturn.OK:
                raise RuntimeError("Failed to link post_analytics_tee to mosaic_q")
            self._attach_flow_probe(tee2mosaic_post, "post_analytics_tee → mosaic_q")
        # Link mosaic branch chain from its per-branch queue to encoder/appsink
        if minimal_pre_infer:
            mosaic_branch_chain = [mosaic_q, mosaic_conv_post, mosaic_caps, mosaic_enc, mosaic_sink]
            for a, b in zip(mosaic_branch_chain, mosaic_branch_chain[1:]):
                if not a.link(b):
                    raise RuntimeError(f"Failed to link {a.name} -> {b.name}")
            self.logger.info("linked tee → mosaic_q → jpegenc → appsink")
        else:
            mosaic_chain = [mosaic_q, tiler, mosaic_conv_pre, mosaic_caps_rgba, mosaic_osd, mosaic_conv_post, mosaic_caps, mosaic_enc, mosaic_sink]
            for a, b in zip(mosaic_chain, mosaic_chain[1:]):
                if not a.link(b):
                    raise RuntimeError(f"Failed to link {a.name} -> {b.name}")
            self.logger.info("linked nvdsanalytics_post → mosaic_q → mosaic_sink")

        # Branch 2: optional EGL display
        if not disable_egl and egl_q and egl_sink:
            if not minimal_pre_infer:
                tee2egl = post_analytics_tee.get_request_pad("src_%u")
                egl_sink_pad = egl_q.get_static_pad("sink")
                if tee2egl is None or egl_sink_pad is None:
                    raise RuntimeError("Failed to prepare EGL branch pads")
                if tee2egl.link(egl_sink_pad) != Gst.PadLinkReturn.OK:
                    raise RuntimeError("Failed to link post_analytics_tee to egl_q")
                self._attach_flow_probe(tee2egl, "post_analytics_tee → egl_q")
            # Link final EGL queue to sink (shared chain already produced RGBA)
            if not egl_q.link(egl_sink):
                raise RuntimeError("Failed to link egl_q -> nveglglessink")
            self.logger.info("linked tee → egl_q → nveglglessink")

        # CPU display fallback removed (temporary feature)

        # Per-stream MDE branches removed during simplification (pre-inference tee disabled)

        # Appsink disabled in minimal mode

    # -- Callbacks --
    def _on_new_jpeg_sample(self, appsink: GstApp.AppSink, sensor_id: int):
        sample = None
        buf = None
        mapinfo = None
        sample = None
        buf = None
        mapinfo = None
        ret = Gst.FlowReturn.OK
        try:
            sample = appsink.emit("pull-sample")
            if not sample:
                self.logger.warning("appsink pull-sample returned None (sensor_id=%s)", sensor_id)
                return Gst.FlowReturn.ERROR
            buf = sample.get_buffer()
            if not buf:
                self.logger.warning("appsink sample missing buffer (sensor_id=%s)", sensor_id)
                return Gst.FlowReturn.ERROR
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                self.logger.warning("appsink buffer map failed (sensor_id=%s)", sensor_id)
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
                        self.logger.warning("jpeg queue full for sensor_id=%s; dropping", sensor_id)
        finally:
            if buf is not None and mapinfo is not None:
                try:
                    buf.unmap(mapinfo)
                except Exception:
                    pass
            if sample is not None:
                try:
                    sample.unref()
                except Exception:
                    pass
        return ret

    def _on_new_mosaic_sample(self, appsink: GstApp.AppSink):
        sample = None
        buf = None
        mapinfo = None
        sample = None
        buf = None
        mapinfo = None
        ret = Gst.FlowReturn.OK
        try:
            sample = appsink.emit("pull-sample")
            if not sample:
                self.logger.warning("mosaic appsink pull-sample returned None")
                return Gst.FlowReturn.ERROR
            buf = sample.get_buffer()
            if not buf:
                self.logger.warning("mosaic appsink sample missing buffer")
                return Gst.FlowReturn.ERROR
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                self.logger.warning("mosaic appsink buffer map failed")
                return Gst.FlowReturn.ERROR
            data = bytes(mapinfo.data) if mapinfo and mapinfo.data else None
            if data:
                try:
                    # Latest-wins: drain any pending items to keep latency low
                    while True:
                        try:
                            _ = self.mosaic_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.mosaic_queue.put_nowait(data)
                    self._mosaic_counter += 1
                    if self._mosaic_counter % 300 == 0:
                        self.logger.info(f"Mosaic: {self._mosaic_counter} frames")
                    if not self._mosaic_first_logged:
                        self.logger.info("Mosaic first frame: %d bytes", len(data))
                        self._mosaic_first_logged = True
                except queue.Full:
                    self.logger.warning("mosaic queue full; dropping")
        finally:
            if buf is not None and mapinfo is not None:
                try:
                    buf.unmap(mapinfo)
                except Exception:
                    pass
            if sample is not None:
                try:
                    sample.unref()
                except Exception:
                    pass
        return ret

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
        elif t == Gst.MessageType.STATE_CHANGED:
            try:
                old, new, pending = message.parse_state_changed()
                src = message.src
                name = src.get_name() if hasattr(src, 'get_name') else str(src)
                if name in getattr(self, '_state_watch_names', set()):
                    self.logger.debug(
                        "state-changed: %s %s -> %s (pending=%s)",
                        name, getattr(old, 'value_name', old), getattr(new, 'value_name', new), getattr(pending, 'value_name', pending)
                    )
            except Exception:
                pass
        elif t == Gst.MessageType.EOS:
            self.logger.info("Pipeline received EOS")
        return True

    def _log_required_plugins(self) -> None:
        """Log presence/absence of required GStreamer/DeepStream plugins for diagnostics."""
        try:
            reg = Gst.Registry.get()
        except Exception:
            self.logger.debug("Gst.Registry not available for plugin check")
            return
        required = [
            ("nvmultiurisrcbin", "DeepStream multi-URI source"),
            ("nvstreamdemux", "DeepStream stream demux"),
            ("nvdspreprocess", "DeepStream preprocess"),
            ("nvinfer", "DeepStream inference"),
            ("nvtracker", "DeepStream tracker"),
            ("nvdsosd", "On-screen display"),
            ("nvmultistreamtiler", "Multistream tiler"),
            ("nvvideoconvert", "Video convert (NVMM)"),
            ("nvjpegenc", "NV JPEG encoder"),
            ("nveglglessink", "EGL GL sink"),
        ]
        for name, desc in required:
            present = bool(reg.find_feature(name, Gst.ElementFactory))
            lvl = self.logger.info if present else self.logger.warning
            lvl("plugin %s: %s", name, "available" if present else f"MISSING ({desc})")

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
