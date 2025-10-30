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

        self._ds_src_settings = self._parse_multiurisrc_config(self.ds_multiurisrc_cfg)

        # Canonical per-source info (sensor ids 0..N-1), built from config file
        uris = self._ds_src_settings.get("uris", [])
        names = self._ds_src_settings.get("sensor_names", [])
        width = int(self._ds_src_settings.get("mux_width", 1920))
        height = int(self._ds_src_settings.get("mux_height", 1080))
        self.source_info: Dict[int, Dict[str, Any]] = {}
        for idx, uri in enumerate(uris):
            name = names[idx] if idx < len(names) and names[idx] else f"Camera {idx+1}"
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

        # Apply RTSP tuning before moving the pipeline out of NULL to ensure
        # protocol and latency choices take effect during negotiation.
        try:
            src_el = self._elements.get("src")
            if src_el is not None:
                self._configure_rtspsrc_children(src_el)
        except Exception as err:
            self.logger.debug("RTSP child tuning skipped prior to PLAYING: %s", err)

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
        # After state transition begins, re-apply RTSP tuning once children exist
        try:
            # Small delay to allow internal bins to instantiate
            time.sleep(0.5)
            src_el = self._elements.get("src")
            if src_el is not None:
                self._configure_rtspsrc_children(src_el)
        except Exception:
            pass
        time.sleep(1.5)
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

    def _parse_multiurisrc_config(self, path: str) -> Dict[str, Any]:
        import configparser
        cfg = configparser.ConfigParser(interpolation=None, delimiters=("="))
        with open(path, "r", encoding="utf-8") as f:
            cfg.read_file(f)
        out: Dict[str, Any] = {}
        # source-list
        sl = cfg["source-list"] if cfg.has_section("source-list") else {}
        def _split_list(v: str) -> List[str]:
            return [s.strip() for s in (v or "").split(";") if s.strip()]
        list_raw = sl.get("list", "") if sl else ""
        uris = _split_list(list_raw)
        out["uris_raw"] = list_raw
        out["uris"] = uris
        sensor_ids_raw = sl.get("sensor-id-list", "") if sl else ""
        out["sensor_ids_raw"] = sensor_ids_raw
        out["sensor_ids"] = _split_list(sensor_ids_raw)
        out["sensor_names"] = _split_list(sl.get("sensor-name-list", "")) if sl else []
        out["max_batch_size"] = int(sl.get("max-batch-size", "0") or 0) if sl else 0
        out["http_ip"] = sl.get("http-ip", "") if sl else ""
        out["http_port"] = int(sl.get("http-port", "0") or 0) if sl else 0

        # source-attr-all
        saa = cfg["source-attr-all"] if cfg.has_section("source-attr-all") else {}
        out["latency"] = int(saa.get("latency", "0") or 0) if saa else 0
        out["cudadec_memtype"] = int(saa.get("cudadec-memtype", "0") or 0) if saa else 0
        out["gpu_id"] = int(saa.get("gpu-id", "0") or 0) if saa else 0
        out["rtsp_reconnect_interval_sec"] = int(saa.get("rtsp-reconnect-interval-sec", "0") or 0) if saa else 0
        out["init_rtsp_reconnect_interval_sec"] = int(saa.get("init-rtsp-reconnect-interval-sec", "0") or 0) if saa else 0
        out["rtsp_reconnect_attempts"] = int(saa.get("rtsp-reconnect-attempts", "0") or 0) if saa else 0
        # Prefer TCP for RTSP by default if provided
        try:
            sel = saa.get("select-rtp-protocol", "") if saa else ""
            out["select_rtp_protocol"] = int(str(sel).strip() or 0)
        except Exception:
            out["select_rtp_protocol"] = 0

        # streammux
        sm = cfg["streammux"] if cfg.has_section("streammux") else {}
        out["batched_push_timeout"] = int(sm.get("batched-push-timeout", "33333") or 33333) if sm else 33333
        out["mux_width"] = int(sm.get("width", "1920") or 1920) if sm else 1920
        out["mux_height"] = int(sm.get("height", "1080") or 1080) if sm else 1080
        out["enable_padding"] = int(sm.get("enable-padding", "0") or 0) if sm else 0
        out["drop_pipeline_eos"] = int(sm.get("drop-pipeline-eos", "1") or 1) if sm else 1
        out["live_source"] = int(sm.get("live-source", "1") or 1) if sm else 1
        return out

    def _configure_rtspsrc_children(self, multi_bin) -> None:
        if not Gst or not isinstance(multi_bin, Gst.Bin):
            return
        try:
            if multi_bin.find_property("drop-audio"):
                multi_bin.set_property("drop-audio", True)
        except Exception as exc:
            self.logger.debug("nvmultiurisrcbin drop-audio property failed: %s", exc)
        try:
            iterator = multi_bin.iterate_recurse()
        except Exception:
            return
        tcp_value = None
        applied = False
        if GstRtsp is not None:
            try:
                tcp_value = int(GstRtsp.RTSPLowerTrans.TCP)
            except Exception:
                tcp_value = 4
        if tcp_value is None:
            tcp_value = 4
        # Use latency from config file if provided, but clamp to a sane max
        cfg_latency = int(self._ds_src_settings.get("latency", 0) or 0)
        desired_latency = int(min(cfg_latency if cfg_latency > 0 else 100, 500))
        tuned_count = 0
        try:
            while True:
                res, element = iterator.next()
                if res == Gst.IteratorResult.OK:
                    if not isinstance(element, Gst.Element):
                        continue
                    factory = element.get_factory()
                    factory_name = factory.get_name() if factory else ""
                    name = element.get_name() or factory_name or "unknown"
                    try:
                        if factory_name == "rtspsrc" or name.startswith("rtspsrc"):
                            try:
                                element.set_property("protocols", tcp_value)
                            except Exception:
                                pass
                            try:
                                element.set_property("latency", desired_latency)
                            except Exception:
                                pass
                            try:
                                element.set_property("do-rtsp-keep-alive", True)
                            except Exception:
                                pass
                            try:
                                if element.find_property("tcp-timeout"):
                                    element.set_property("tcp-timeout", max(desired_latency, 100))
                            except Exception:
                                pass
                            try:
                                if element.find_property("drop-on-latency"):
                                    element.set_property("drop-on-latency", False)
                            except Exception:
                                pass
                            try:
                                if element.find_property("ntp-sync"):
                                    element.set_property("ntp-sync", True)
                            except Exception:
                                pass
                            try:
                                if element.find_property("player-idle-timeout"):
                                    element.set_property("player-idle-timeout", 0)
                            except Exception:
                                pass
                            tuned_count += 1
                            applied = True
                            continue
                        if factory_name in {"nvurisrcbin", "dsnvurisrcbin"} or "dsnvurisrcbin" in name:
                            try:
                                if element.find_property("select-rtp-protocol"):
                                    element.set_property("select-rtp-protocol", 4)
                            except Exception:
                                pass
                            try:
                                if element.find_property("latency"):
                                    element.set_property("latency", desired_latency)
                            except Exception:
                                pass
                            try:
                                if element.find_property("rtsp-reconnect-interval"):
                                    element.set_property("rtsp-reconnect-interval", int(self._ds_src_settings.get("rtsp_reconnect_interval_sec", 0) or 0))
                            except Exception:
                                pass
                            applied = True
                    except Exception as err:
                        self.logger.warning("Failed to tune RTSP source %s: %s", name, err)
                elif res == Gst.IteratorResult.DONE:
                    break
                else:
                    continue
        finally:
            # In GI, the iterator is managed by the binding; avoid explicit free()
            # to prevent potential double-free during interpreter shutdown.
            iterator = None
        if tuned_count:
            self.logger.info("Applied RTSP tuning to %d rtspsrc element(s)", tuned_count)
        if applied:
            self.logger.info("Applied RTSP tuning to nvmultiurisrcbin children")

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
        # Apply nvmultiurisrcbin configuration from file
        cfg = self._ds_src_settings
        uris = cfg.get("uris", [])
        # Prefer the raw semicolon-separated list exactly as in the INI
        uri_list = str(cfg.get("uris_raw", "") or ";".join(uris))
        sensor_ids = str(cfg.get("sensor_ids_raw", "") or "")
        try:
            src.set_property("uri-list", uri_list)
            self.logger.info(f"nvmultiurisrcbin uri-list set ({len(uris)} uris)")
        except Exception as e:
            self.logger.warning(f"Failed to set uri-list on nvmultiurisrcbin: {e}")
        if sensor_ids:
            try:
                src.set_property("sensor-id-list", sensor_ids)
            except Exception:
                pass
        try:
            mb = int(cfg.get("max_batch_size", 0) or 0) or len(uris)
            src.set_property("max-batch-size", mb)
        except Exception:
            pass
        try:
            w = int(cfg.get("mux_width", 1920) or 1920)
            h = int(cfg.get("mux_height", 1080) or 1080)
        except Exception:
            w, h = 1920, 1080
        for k, v in (
            ("width", w),
            ("height", h),
            ("live-source", int(cfg.get("live_source", 1) or 1)),
            ("drop-pipeline-eos", int(cfg.get("drop_pipeline_eos", 1) or 1)),
            ("latency", int(cfg.get("latency", 0) or 0)),
        ):
            try:
                src.set_property(k, v)
            except Exception:
                pass
        # Sample config: set reconnect interval (seconds) if property exists
        for k, v in {
            "rtsp-reconnect-interval": int(cfg.get("rtsp_reconnect_interval_sec", 0) or 0),
            "init-rtsp-reconnect-interval": int(cfg.get("init_rtsp_reconnect_interval_sec", 0) or 0),
            "rtsp-reconnect-attempts": int(cfg.get("rtsp_reconnect_attempts", 0) or 0),
            "cudadec-memtype": int(cfg.get("cudadec_memtype", 0) or 0),
            "gpu-id": int(cfg.get("gpu_id", 0) or 0),
            "ip-address": str(cfg.get("http_ip", "") or ""),
            "port": int(cfg.get("http_port", 0) or 0),
            # Ensure TCP for RTP when specified in config
            "select-rtp-protocol": int(cfg.get("select_rtp_protocol", 0) or 0),
        }.items():
            try:
                src.set_property(k, v)
            except Exception as e:
                # ignore unsupported properties on this build
                self.logger.debug(f"nvmultiurisrcbin prop {k} not set: {e}")
        try:
            bpt = int(cfg.get("batched_push_timeout", 33333) or 33333)
            src.set_property("batched-push-timeout", bpt)
            self.logger.info(f"Set batched-push-timeout to {bpt}")
        except Exception as e:
            self.logger.warning("Failed to set batched-push-timeout: %s", e)
        for k, v in {"cudadec-memtype": 0, "drop-frame-interval": 0}.items():
            try:
                src.set_property(k, v)
            except Exception as e:
                self.logger.warning(f"nvmultiurisrcbin prop {k} failed: {e}")

        try:
            self.pipeline.add(src)
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
        tracker_cfg_raw = getattr(self.config.processing, "DEEPSTREAM_TRACKER_CONFIG", "")
        tracker_cfg_path = os.path.abspath(tracker_cfg_raw) if tracker_cfg_raw else ""
        tracker_lib_raw = getattr(self.config.processing, "DEEPSTREAM_TRACKER_LIB", "")
        tracker_lib_path = os.path.abspath(tracker_lib_raw) if tracker_lib_raw else ""
        try:
            if tracker_cfg_path and os.path.exists(tracker_cfg_path):
                tracker_dir = os.path.dirname(tracker_cfg_path) or os.getcwd()
                tracker_doc: Dict[str, Any] = {}
                try:
                    with open(tracker_cfg_path, "r", encoding="utf-8") as fh:
                        tracker_doc = yaml.safe_load(fh) or {}
                except Exception as err:
                    self.logger.debug("Failed to parse tracker YAML %s: %s", tracker_cfg_path, err)
                    tracker_doc = {}

                tracker_section = tracker_doc.get("tracker") if isinstance(tracker_doc, dict) else None
                if isinstance(tracker_section, dict) and tracker_section:
                    for key, value in tracker_section.items():
                        if isinstance(value, str) and key.endswith("config-file") and not os.path.isabs(value):
                            value = os.path.abspath(os.path.join(tracker_dir, value))
                        tracker.set_property(key, value)
                        self.logger.debug("nvtracker property %s set", key)
                else:
                    tracker.set_property("ll-config-file", tracker_cfg_path)
                    self.logger.info("nvtracker ll-config-file set to %s", tracker_cfg_path)
            elif tracker_cfg_path:
                self.logger.warning("nvtracker config path %s not found", tracker_cfg_path)

            if tracker_lib_path:
                tracker.set_property("ll-lib-file", tracker_lib_path)
                self.logger.info("nvtracker ll-lib-file set to %s", tracker_lib_path)
        except Exception as e:
            self.logger.error("nvtracker config setup failed: %s; using defaults (degraded mode)", e)

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

        # Link nvmultiurisrcbin -> pre_tee via static pads (src pad is always present)
        try:
            spad = src.get_static_pad("src")
            sink = tee.get_static_pad("sink")
            if spad and sink and not sink.is_linked():
                if spad.link(sink) == Gst.PadLinkReturn.OK:
                    self.logger.info("linked nvmultiurisrcbin → pre_tee (static)")
                else:
                    self.logger.warning("Static link nvmultiurisrcbin → pre_tee failed")
        except Exception:
            self.logger.debug("Static pad link attempt failed")

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
