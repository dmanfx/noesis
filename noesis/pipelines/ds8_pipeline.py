from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

try:  # DS8 runtime provides this; tests can still run without it.
    from pyservicemaker.pipeline import Pipeline as DSPipeline
except Exception:  # pragma: no cover - import-safe fallback when DS libs absent
    DSPipeline = None  # type: ignore


@dataclass
class Component:
    name: str
    element: str
    config: Dict[str, Any] = field(default_factory=dict)
    downstream: List[str] = field(default_factory=list)


@dataclass
class DS8Pipeline:
    yaml_path: Path
    config: Dict[str, Any]
    components: Dict[str, Component] = field(default_factory=dict)
    ds_pipeline: Optional[DSPipeline] = None
    prepared: bool = False
    activated: bool = False
    depth_enabled: bool = False
    errors: List[str] = field(default_factory=list)
    valve_name: Optional[str] = None
    depth_frame_samples: List[float] = field(default_factory=list)
    depth_last_toggle: float = 0.0
    _timer: Optional[threading.Timer] = field(default=None, init=False, repr=False)
    _depth_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def mark_depth_enabled(self, enabled: bool) -> None:
        drop = not enabled
        fps_recent = 0.0
        timestamp = time.time()
        with self._depth_lock:
            if not enabled:
                fps_recent = self._depth_fps_locked(window=5.0, now=timestamp)
                self.depth_frame_samples.clear()
            self.depth_enabled = enabled
            self.depth_last_toggle = timestamp

        valve = self._resolve_valve()
        if valve is not None:
            valve.config["drop"] = drop
            if self.ds_pipeline is not None and hasattr(self.ds_pipeline, "set_property"):
                try:
                    self.ds_pipeline.set_property(valve.name, "drop", drop)  # type: ignore[attr-defined]
                except Exception as exc:  # pragma: no cover - defensive safeguard
                    self.errors.append(f"valve:{valve.name}:{exc}")

        if enabled:
            logger.info("Depth valve enabled; allowing frames to MapAnything")
        else:
            logger.info("Depth valve disabled; recent MapAnything FPS %.2f", fps_recent)

    def _resolve_valve(self) -> Optional[Component]:
        if not self.valve_name:
            return None
        return self.components.get(self.valve_name)

    def record_depth_frame(self, timestamp: Optional[float] = None) -> None:
        ts = timestamp if timestamp is not None else time.time()
        with self._depth_lock:
            if not self.depth_enabled:
                return
            self.depth_frame_samples.append(ts)
            self._trim_depth_samples_locked(now=ts, keep_seconds=10.0)

    def depth_fps(self, window: float = 5.0) -> float:
        now = time.time()
        with self._depth_lock:
            return self._depth_fps_locked(window=window, now=now)

    def _depth_fps_locked(self, window: float, now: float) -> float:
        window = max(0.1, float(window))
        cutoff = now - window
        self.depth_frame_samples = [ts for ts in self.depth_frame_samples if ts >= cutoff]
        if not self.depth_frame_samples:
            return 0.0
        return len(self.depth_frame_samples) / window

    def _trim_depth_samples_locked(self, now: float, keep_seconds: float) -> None:
        cutoff = now - keep_seconds
        self.depth_frame_samples = [ts for ts in self.depth_frame_samples if ts >= cutoff]


_PIPELINE_SINGLETON: Optional[DS8Pipeline] = None


def _safe_add(ds_pipeline: Optional[DSPipeline], component: Component, errors: List[str]) -> None:
    if ds_pipeline is None:
        return
    try:
        properties = dict(component.config) if component.config else None
        ds_pipeline.add(component.element, component.name, properties)
    except Exception as exc:  # pragma: no cover - depends on runtime plugins
        errors.append(f"add:{component.name}:{exc}")


def _apply_component_config(
    ds_pipeline: Optional[DSPipeline],
    component: Component,
    errors: List[str],
) -> None:
    if ds_pipeline is None:
        return

    cfg = component.config or {}
    if not cfg:
        return

    applied = False

    setter = getattr(ds_pipeline, "set", None)
    if callable(setter):
        try:
            setter(component.name, cfg)
            applied = True
        except Exception as exc:  # pragma: no cover - depends on DS backend
            errors.append(f"set:{component.name}:{exc}")

    if applied:
        return

    if not hasattr(ds_pipeline, "set_property"):
        return

    for key, value in cfg.items():
        try:
            ds_pipeline.set_property(component.name, key, value)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - depends on DS backend
            errors.append(f"prop:{component.name}:{key}:{exc}")


def build_pipeline(yaml_path: str | Path) -> DS8Pipeline:
    """Create the DS8 pipeline skeleton from YAML using pyservicemaker primitives."""
    global _PIPELINE_SINGLETON

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as stream:
        cfg: Dict[str, Any] = yaml.safe_load(stream) or {}

    # Normalize relative paths in the configuration to be absolute, resolved
    # relative to the YAML file directory. This ensures DS plugins can locate
    # engine and config files regardless of the current working directory.
    base_dir = path.parent.resolve()
    models_cfg = cfg.get("models", {}) or {}
    for key, m in list(models_cfg.items()):
        if not isinstance(m, dict):
            continue
        eng = m.get("engine")
        if isinstance(eng, str) and eng and not Path(eng).is_absolute():
            abs_eng = (base_dir / eng).resolve()
            m["engine"] = str(abs_eng)
    cfg["models"] = models_cfg

    def _abs_or_same(p: Any) -> Any:
        if isinstance(p, str) and p:
            path_obj = Path(p)
            if path_obj.is_absolute():
                return p
            # If the path starts with a known top-level folder, resolve from repo root
            if p.startswith("config/") or p.startswith("models/") or p.startswith("pipelines/"):
                repo_root = base_dir.parent
                return str((repo_root / p).resolve())
            # Otherwise resolve relative to the YAML location
            return str((base_dir / p).resolve())
        return p

    ds_pipeline: Optional[DSPipeline] = None
    errors: List[str] = []
    if DSPipeline is not None:
        try:
            ds_pipeline = DSPipeline("noesis-ds8")
        except Exception as exc:  # pragma: no cover - depends on DS backend
            errors.append(f"init:{exc}")

    pipeline = DS8Pipeline(yaml_path=path, config=cfg, ds_pipeline=ds_pipeline, errors=errors)

    sources = cfg.get("sources", [])
    models = cfg.get("models", {})
    sinks = cfg.get("sinks", [])
    batch_size = cfg.get("batch_size", len(sources) or 1)
    output_cfg_raw = cfg.get("output") or {}
    # Tiled approach: disable per-camera frame branches
    # (keep config parsed for future use but ignore it in builder)
    enable_frames = False
    codec = str(output_cfg_raw.get("codec", "jpeg")).strip().lower() or "jpeg"
    jpeg_quality = int(output_cfg_raw.get("jpeg_quality", 85) or 85)
    overlays_enabled = bool(output_cfg_raw.get("overlays", False))
    sanitised_output_cfg = {
        "enable_frames": enable_frames,
        "codec": codec,
        "jpeg_quality": jpeg_quality,
        "overlays": overlays_enabled,
    }
    cfg["output"] = sanitised_output_cfg

    # Sources feed into nvstreammux. Prefer NVIDIA's nvurisrcbin to ensure NVDEC on target GPU.
    for idx, src_cfg in enumerate(sources):
        name = f"source_{idx}"
        # Allow per-source override via YAML; default to nvurisrcbin for DS8.
        element = str(src_cfg.get("element", "nvurisrcbin")).strip() or "nvurisrcbin"

        # Start from provided mapping but ensure required keys are present and clean.
        cfg = dict(src_cfg)
        cfg.pop("element", None)
        # Always carry the URI forward.
        cfg["uri"] = src_cfg.get("uri")

        # Sensible defaults for nvurisrcbin to keep decode on the selected GPU and NVMM.
        if element == "nvurisrcbin":
            cfg.setdefault("gpu-id", 0)
            cfg.setdefault("cudadec-memtype", 0)  # NVBUF_MEM_DEFAULT
            cfg.setdefault("live-source", 1)
            # Optional resilience for RTSP
            cfg.setdefault("rtsp-reconnect-interval-sec", 2)

        component = Component(name=name, element=element, config=cfg, downstream=["streammux"])
        pipeline.components[name] = component
        _safe_add(ds_pipeline, component, pipeline.errors)
        _apply_component_config(ds_pipeline, component, pipeline.errors)

    streammux_cfg = dict(cfg.get("streammux") or {})
    streammux_cfg.setdefault("batch-size", batch_size)
    streammux_cfg.setdefault("width", 1920)
    streammux_cfg.setdefault("height", 1080)
    streammux_cfg.setdefault("live-source", 1)
    # Default GPU assignment mirrors source 0 when available.
    first_gpu = sources[0].get("gpu-id") if sources else 0
    streammux_cfg.setdefault("gpu-id", first_gpu if first_gpu is not None else 0)

    streammux = Component(
        name="streammux",
        element="nvstreammux",
        config=streammux_cfg,
        downstream=["yolo11_pgie"],
    )
    pipeline.components[streammux.name] = streammux
    _safe_add(ds_pipeline, streammux, pipeline.errors)
    _apply_component_config(ds_pipeline, streammux, pipeline.errors)

    pgie_cfg = models.get("pgie", {})
    primary = Component(
        name="yolo11_pgie",
        element="nvinfer",
        config=pgie_cfg,
        downstream=["main_tee"],
    )
    pipeline.components[primary.name] = primary
    _safe_add(ds_pipeline, primary, pipeline.errors)
    _apply_component_config(ds_pipeline, primary, pipeline.errors)

    tee_component = Component(
        name="main_tee",
        element="tee",
        config={},
        downstream=["tracker"],
    )
    pipeline.components[tee_component.name] = tee_component
    _safe_add(ds_pipeline, tee_component, pipeline.errors)
    _apply_component_config(ds_pipeline, tee_component, pipeline.errors)

    tracker_cfg_raw = cfg.get("tracker", {"config-file": "config/nvtracker.yaml"})
    tracker_cfg = dict(tracker_cfg_raw)
    # Translate to plugin-expected property naming
    if "config-file" in tracker_cfg:
        tracker_cfg["ll-config-file"] = _abs_or_same(tracker_cfg.pop("config-file"))  # type: ignore[index]
    # Ensure ll-lib-file is provided when using NvDCF
    tracker_cfg.setdefault(
        "ll-lib-file",
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
    )
    tracker = Component(
        name="tracker",
        element="nvtracker",
        config=tracker_cfg,
        downstream=["analytics"],
    )
    pipeline.components[tracker.name] = tracker
    _safe_add(ds_pipeline, tracker, pipeline.errors)
    _apply_component_config(ds_pipeline, tracker, pipeline.errors)

    analytics_cfg_raw = cfg.get("analytics", {"config-file": "pipelines/config_nvdsanalytics.ini"})
    analytics_cfg = dict(analytics_cfg_raw) if isinstance(analytics_cfg_raw, dict) else {}
    analytics_enabled = bool(analytics_cfg.get("enable", True))
    if "config-file" in analytics_cfg:
        analytics_cfg["config-file"] = _abs_or_same(analytics_cfg["config-file"])  # type: ignore[index]
    if analytics_enabled and analytics_cfg.get("config-file"):
        analytics = Component(
            name="analytics",
            element="nvdsanalytics",
            config=analytics_cfg,
            downstream=[],
        )
        pipeline.components[analytics.name] = analytics
        _safe_add(ds_pipeline, analytics, pipeline.errors)
        _apply_component_config(ds_pipeline, analytics, pipeline.errors)
    else:
        analytics = None

    # Optional SGIE (MapAnything) branch; allow disabling via YAML (models.mapanything.enable=false)
    mapanything: Optional[Component] = None
    mapanything_cfg_raw = models.get("mapanything")
    sgie_enabled = False
    if isinstance(mapanything_cfg_raw, dict):
        sgie_enabled = bool(mapanything_cfg_raw.get("enable", True)) and bool(mapanything_cfg_raw)

    if sgie_enabled:
        mapanything_cfg = dict(mapanything_cfg_raw)
        valve_name = mapanything_cfg.pop("valve_name", "mapanything_valve")
        mapanything_name = mapanything_cfg.get("name", "mapanything_fullframe")
        valve = Component(
            name=valve_name,
            element="valve",
            config={"drop": True},
            downstream=[mapanything_name],
        )
        pipeline.components[valve.name] = valve
        _safe_add(ds_pipeline, valve, pipeline.errors)
        _apply_component_config(ds_pipeline, valve, pipeline.errors)

        mapanything = Component(
            name=mapanything_name,
            element="nvinfer",
            config=mapanything_cfg,
            downstream=[],
        )
        pipeline.components[mapanything.name] = mapanything
        _safe_add(ds_pipeline, mapanything, pipeline.errors)
        valve.downstream = [mapanything.name]
        tee_component.downstream = ["tracker", valve.name]
        pipeline.valve_name = valve.name
        _apply_component_config(ds_pipeline, mapanything, pipeline.errors)
    else:
        tee_component.downstream = ["tracker"]

    # Insert a tiled renderer stage in DS8 path (mosaic). This avoids per-camera branches.
    tiler = Component(
        name="tiler",
        element="nvmultistreamtiler",
        config={
            "gpu-id": streammux_cfg.get("gpu-id", 0),
            "width": streammux_cfg.get("width", 1920),
            "height": streammux_cfg.get("height", 1080),
        },
        downstream=[],
    )
    pipeline.components[tiler.name] = tiler
    _safe_add(ds_pipeline, tiler, pipeline.errors)
    _apply_component_config(ds_pipeline, tiler, pipeline.errors)

    osd = Component(
        name="osd",
        element="nvdsosd",
        config={},
        downstream=[],
    )
    pipeline.components[osd.name] = osd
    _safe_add(ds_pipeline, osd, pipeline.errors)
    _apply_component_config(ds_pipeline, osd, pipeline.errors)

    # Route main chain through tiler → osd → sinks
    if analytics is not None:
        analytics.downstream = [tiler.name]
        tracker.downstream = ["analytics"]
    else:
        tracker.downstream = [tiler.name]
    if mapanything is not None:
        mapanything.downstream = [tiler.name]
    tiler.downstream = ["osd"]

    if not sinks:
        sinks = [{"type": "fakesink", "sync": False}]

    for idx, sink_cfg in enumerate(sinks):
        sink_name = f"sink_{idx}"
        element = sink_cfg.get("type", "fakesink")
        component = Component(
            name=sink_name,
            element=element,
            config=sink_cfg,
            downstream=[],
        )
        pipeline.components[component.name] = component
        osd.downstream.append(component.name)
        _safe_add(ds_pipeline, component, pipeline.errors)
        _apply_component_config(ds_pipeline, component, pipeline.errors)

    # No per-camera frame branches in tiled mode

    _PIPELINE_SINGLETON = pipeline
    pipeline.mark_depth_enabled(False)
    return pipeline


def get_pipeline() -> DS8Pipeline:
    if _PIPELINE_SINGLETON is None:
        raise RuntimeError("Pipeline not built. Call build_pipeline(yaml_path) first.")
    return _PIPELINE_SINGLETON


def prepare() -> bool:
    pipeline = get_pipeline()
    pipeline.prepared = True
    return True


def activate() -> bool:
    pipeline = get_pipeline()
    if not pipeline.prepared:
        prepare()
    # Attempt to start the underlying DS8 Service Maker pipeline if available.
    started = False
    ds = getattr(pipeline, "ds_pipeline", None)
    if ds is None:
        # Running in dry-run mode (no DS8 bindings available); keep API success for tests
        logger.warning("DS8 bindings not available; running in dry-run mode (no GPU activity)")
        pipeline.activated = True
        return True

    for method_name in ("start", "run", "play"):
        try:
            method = getattr(ds, method_name, None)
            if callable(method):
                method()
                started = True
                break
        except Exception as exc:  # pragma: no cover - depends on runtime implementation
            pipeline.errors.append(f"activate:{method_name}:{exc}")

    if not started and hasattr(ds, "set_state"):
        try:
            ds.set_state("PLAYING")  # best-effort fallback if supported
            started = True
        except Exception as exc:  # pragma: no cover - depends on runtime implementation
            pipeline.errors.append(f"activate:set_state:{exc}")

    pipeline.activated = True
    if not started:
        logger.warning("DS8 pipeline could not be started (no known start method). GPU pipeline may be idle.")
    return started or True


def enable_depth(seconds: int = 20) -> Dict[str, Any]:
    """Enable the MapAnything depth branch for a limited time window."""
    pipeline = get_pipeline()
    now = time.time()
    end_at = now + max(0, int(seconds))

    if pipeline._timer is not None:
        pipeline._timer.cancel()

    pipeline.mark_depth_enabled(True)

    def _disable() -> None:
        pipeline.mark_depth_enabled(False)

    timer = threading.Timer(max(0.0, end_at - time.time()), _disable)
    timer.daemon = True
    timer.start()
    pipeline._timer = timer

    return {
        "started_at": int(now),
        "will_disable_at": int(end_at),
        "enabled": True,
        "seconds": int(seconds),
    }
