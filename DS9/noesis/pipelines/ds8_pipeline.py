from __future__ import annotations

import logging
import math
import os
import json
import hashlib
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from noesis_core.inference_runtime_contract import (
    EngineOnlyRuntimeError,
    materialize_nvinfer_engine_only_config,
    materialize_nvtracker_engine_only_config,
)
from noesis_core.runtime_secrets import load_pipeline_config, redact_runtime_secrets
from noesis_core.servicemaker_shutdown import (
    SyntheticStubEosMessage,
    synthetic_stub_lifecycle_evidence,
)
from noesis_core.source_progress import (
    DecodedSourceProgressMonitor,
    SourceProgressPolicy,
)
from noesis.telemetry.latency_metrics import LatencyCollector

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
DS9_ROOT = REPO_ROOT / "DS9"


def validate_exclusion_stream_coverage(
    sources: Sequence[Mapping[str, Any]],
    analytics_config: Mapping[str, Any],
) -> None:
    """Require one explicit exclusion-stream policy for every canonical source."""
    stages = analytics_config.get("stages")
    exclude = stages.get("exclude") if isinstance(stages, Mapping) else None
    streams = exclude.get("streams") if isinstance(exclude, Mapping) else None
    if not isinstance(streams, Mapping):
        raise RuntimeError(
            "enabled analytics exclusion requires analytics.stages.exclude.streams"
        )
    expected: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise RuntimeError(f"pipeline source {index} must be a mapping")
        raw_source_id = source.get("source-id", index)
        if isinstance(raw_source_id, bool):
            raise RuntimeError(f"pipeline source {index} has an invalid boolean source-id")
        try:
            source_id = int(raw_source_id)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"pipeline source {index} has a non-integer source-id: {raw_source_id!r}"
            ) from exc
        if source_id != index:
            raise RuntimeError(
                "pipeline source-id ordering must be canonical and contiguous: "
                f"source {index} declares {source_id}"
            )
        expected.add(str(source_id))

    normalized: list[str] = []
    for raw_stream_id in streams:
        stream_id = str(raw_stream_id)
        if (
            not stream_id.isascii()
            or not stream_id.isdigit()
            or (len(stream_id) > 1 and stream_id.startswith("0"))
        ):
            raise RuntimeError(
                f"analytics exclusion has a non-canonical stream id: {raw_stream_id!r}"
            )
        normalized.append(stream_id)
    if len(normalized) != len(set(normalized)):
        raise RuntimeError("analytics exclusion contains ambiguous duplicate stream ids")

    actual = set(normalized)
    if actual != expected:
        raise RuntimeError(
            "analytics exclusion stream coverage does not match canonical sources "
            f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
        )


def require_native_exclusion_element(config: Mapping[str, Any]) -> str:
    """Require the sole canonical pre-tracker exclusion implementation."""
    element = str(config.get("element") or "").strip()
    if element != "nvdsroiexclude":
        raise RuntimeError(
            "enabled analytics exclusion requires element='nvdsroiexclude'; "
            f"got {element or '<missing>'!r}"
        )
    return element

try:  # DS9 runtime provides this; tests can still run without it.
    from pyservicemaker import Pipeline as DSPipeline
except Exception:  # pragma: no cover - import-safe fallback when DS libs absent
    DSPipeline = None  # type: ignore

try:  # Flow BufferOperator may not be available on all DS9 builds
    # Service Maker docs: from pyservicemaker import BufferOperator, Probe
    from pyservicemaker import BufferOperator, Probe  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional dependency or older DS8 build
    BufferOperator = None  # type: ignore
    Probe = None  # type: ignore

_BufferOperatorBase = BufferOperator if BufferOperator is not None else object


class _NoopPipelineNode:
    """Property holder with explicit synthetic control-node semantics."""

    def __init__(
        self,
        name: str,
        element: str = "",
        properties: Optional[Dict[str, Any]] = None,
        *,
        owner: Optional["_NoopDSPipeline"] = None,
    ) -> None:
        self.name = name
        self.element = element
        self.properties: Dict[str, Any] = dict(properties or {})
        self._owner = owner
        self._lock = threading.RLock()
        self._request_pending = False
        if self.element in {"noesiseos", "noesisforceidr"}:
            self.properties.setdefault("request-sequence", 0)
            self.properties.setdefault("accepted-sequence", 0)
            self.properties.setdefault("last-request-ok", False)
        if self.element == "nvdsroiexclude":
            self._seed_synthetic_analytics_receipt()

    @staticmethod
    def _sha256_file(path_value: Any) -> str:
        path = Path(str(path_value or ""))
        if not path.is_file():
            return ""
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _seed_synthetic_analytics_receipt(self) -> None:
        active_sha256 = self._sha256_file(self.properties.get("config-file"))
        self.properties.setdefault("reload-request-sequence", 0)
        self.properties.setdefault("reload-accepted-sequence", 0)
        self.properties.setdefault("reload-failed-sequence", 0)
        self.properties.setdefault("last-reload-ok", True)
        self.properties.setdefault("expected-config-sha256", active_sha256)
        self.properties.setdefault("active-config-sha256", active_sha256)
        self.properties.setdefault("reload-error-count", 0)
        self.properties.setdefault("objects-removed-count", 0)
        self.properties.setdefault("last-reload-error", "")

    def get(self, key: str) -> Any:
        with self._lock:
            if key not in self.properties:
                raise KeyError(key)
            return self.properties[key]

    def set(self, props: Dict[str, Any]) -> None:
        if not isinstance(props, dict):
            return
        if "request-sequence" in props and self.element in {
            "noesiseos",
            "noesisforceidr",
        }:
            self._request_monotonic_control(int(props["request-sequence"]))
            props = {key: value for key, value in props.items() if key != "request-sequence"}
        with self._lock:
            previous_reload_sequence = int(
                self.properties.get("reload-request-sequence", 0)
            )
            self.properties.update(props)
            if self.element == "nvdsroiexclude":
                self._apply_synthetic_analytics_reload(
                    props,
                    previous_reload_sequence=previous_reload_sequence,
                )

    def _request_monotonic_control(self, requested_sequence: int) -> None:
        if requested_sequence < 0 or requested_sequence > 0xFFFFFFFF:
            raise ValueError("synthetic control request sequence is outside uint32")
        with self._lock:
            current = int(self.properties.get("request-sequence", 0))
            if requested_sequence <= current or self._request_pending:
                return
            self.properties["request-sequence"] = requested_sequence
            self.properties["last-request-ok"] = False
            self._request_pending = True

        def _accept() -> None:
            accepted = False
            try:
                if self.element == "noesiseos":
                    if self._owner is None:
                        raise RuntimeError("synthetic EOS node has no pipeline owner")
                    self._owner._accept_orderly_eos(requested_sequence)
                accepted = True
            except Exception:
                logger.exception(
                    "Synthetic %s request %d failed",
                    self.element,
                    requested_sequence,
                )
            finally:
                with self._lock:
                    self._request_pending = False
                    self.properties["last-request-ok"] = accepted
                    if accepted:
                        self.properties["accepted-sequence"] = requested_sequence

        worker = threading.Thread(
            target=_accept,
            name=f"Synthetic-{self.element}-{requested_sequence}",
            daemon=True,
        )
        worker.start()

    def _apply_synthetic_analytics_reload(
        self,
        props: Mapping[str, Any],
        *,
        previous_reload_sequence: int,
    ) -> None:
        if "config-file" in props and "reload-request-sequence" not in props:
            active_sha256 = self._sha256_file(self.properties.get("config-file"))
            self.properties["active-config-sha256"] = active_sha256
            self.properties["expected-config-sha256"] = active_sha256
        if "reload-request-sequence" not in props:
            return
        sequence = int(props["reload-request-sequence"])
        if sequence <= previous_reload_sequence:
            return
        expected_sha256 = str(self.properties.get("expected-config-sha256") or "")
        actual_sha256 = self._sha256_file(self.properties.get("config-file"))
        self.properties["reload-request-sequence"] = sequence
        if expected_sha256 and actual_sha256 == expected_sha256:
            self.properties["reload-accepted-sequence"] = sequence
            self.properties["active-config-sha256"] = actual_sha256
            self.properties["last-reload-ok"] = True
            self.properties["last-reload-error"] = ""
            return
        self.properties["reload-failed-sequence"] = sequence
        self.properties["reload-error-count"] = int(
            self.properties.get("reload-error-count", 0)
        ) + 1
        self.properties["last-reload-ok"] = False
        self.properties["last-reload-error"] = "synthetic_config_sha256_mismatch"


class _NoopDSPipeline:
    """Explicit, non-promotable pure-Python lifecycle test backend."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.lifecycle_evidence = synthetic_stub_lifecycle_evidence()
        self.nodes: Dict[str, _NoopPipelineNode] = {}
        self.links: List[Tuple[Any, ...]] = []
        self.attachments: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
        self.prepared = False
        self.activated = False
        self._on_message: Optional[Any] = None
        self._stopped = threading.Event()

    def add(self, element: str, name: str, properties: Optional[Dict[str, Any]] = None) -> None:
        self.nodes[name] = _NoopPipelineNode(
            name=name,
            element=element,
            properties=properties,
            owner=self,
        )

    def set(self, name: str, cfg: Dict[str, Any]) -> None:
        self[name].set(cfg)

    def set_property(self, name: str, key: str, value: Any) -> None:
        self[name].set({key: value})

    def link(self, *names: Any) -> None:
        self.links.append(tuple(names))

    def attach(self, *args: Any, **kwargs: Any) -> None:
        self.attachments.append((tuple(args), dict(kwargs)))

    def prepare(self, *args: Any, **kwargs: Any) -> int:
        on_message = args[0] if args else kwargs.get("on_message")
        if on_message is not None and not callable(on_message):
            raise TypeError("synthetic pipeline on_message callback must be callable")
        self._on_message = on_message
        self.prepared = True
        return 1

    def activate(self) -> None:
        if not self.prepared:
            raise RuntimeError("synthetic pipeline must be prepared before activation")
        self._stopped.clear()
        self.activated = True

    def wait(self) -> None:
        self._stopped.wait()

    def _accept_orderly_eos(self, request_sequence: int) -> None:
        callback = self._on_message
        if not callable(callback):
            raise RuntimeError("synthetic pipeline EOS callback is unavailable")
        callback(SyntheticStubEosMessage(request_sequence=request_sequence))
        self.activated = False
        self._stopped.set()

    def __getitem__(self, name: str) -> _NoopPipelineNode:
        if name not in self.nodes:
            self.nodes[name] = _NoopPipelineNode(name=name, owner=self)
        return self.nodes[name]


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
    ds_pipeline: Optional[Any] = None
    prepared: bool = False
    activated: bool = False
    depth_enabled: bool = False
    depth_gate_supported: bool = False
    errors: List[str] = field(default_factory=list)
    valve_name: Optional[str] = None
    depth_gate_attach: Optional[str] = None
    shutdown_eos_component_name: str = "orderly_eos_control"
    lifecycle_evidence: Optional[Dict[str, object]] = None
    depth_frame_samples: List[float] = field(default_factory=list)
    depth_last_toggle: float = 0.0
    frame_size: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    analytics_reload_count: int = 0
    latency_collector: Optional[LatencyCollector] = None
    source_progress_monitor: Optional[DecodedSourceProgressMonitor] = None
    source_progress_targets: Dict[int, str] = field(default_factory=dict)
    _timer: Optional[threading.Timer] = field(default=None, init=False, repr=False)
    _prime_timer: Optional[threading.Timer] = field(default=None, init=False, repr=False)
    _depth_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _set_valve_drop(self, drop: bool) -> bool:
        valve = self._resolve_valve()
        if valve is None:
            return False
        drop_bool = bool(drop)
        valve.config["drop"] = drop_bool
        ds = getattr(self, "ds_pipeline", None)
        if ds is not None:
            try:
                node = ds[valve.name]  # type: ignore[index]
                node.set({"drop": drop_bool})
            except Exception:
                logger.exception("Failed to set valve.drop=%s on %s", drop_bool, valve.name)
        return True

    def mark_depth_enabled(self, enabled: bool) -> None:
        timestamp = time.time()
        fps_recent = 0.0
        with self._depth_lock:
            self.depth_enabled = enabled
            self.depth_last_toggle = timestamp
            if not enabled:
                fps_recent = self._depth_fps_locked(window=5.0, now=timestamp)
                self.depth_frame_samples.clear()
        # Drive upstream valve (if present) so MapAnything inference is physically gated.
        drop_state = not enabled
        if self._set_valve_drop(drop_state):
            logger.info("Depth gate toggled: enabled=%s valve.drop=%s", enabled, drop_state)
        if enabled:
            logger.info("Depth branch enabled; allowing frames to MapAnything")
        else:
            logger.info("Depth branch disabled; recent MapAnything FPS %.2f", fps_recent)

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

    def cancel_control_timers(self, *, join_timeout: float = 1.0) -> int:
        """Cancel and join depth-control timers before native graph teardown."""

        with self._depth_lock:
            timers = [timer for timer in (self._timer, self._prime_timer) if timer is not None]
            self._timer = None
            self._prime_timer = None
        for timer in timers:
            timer.cancel()
        current = threading.current_thread()
        for timer in timers:
            if timer is current or not timer.is_alive():
                continue
            timer.join(timeout=max(0.0, float(join_timeout)))
            if timer.is_alive():
                raise RuntimeError(f"pipeline control timer did not terminate: {timer.name}")
        return len(timers)


class DepthGateOperator(_BufferOperatorBase):  # pragma: no cover - runtime only
    """Drop buffers when depth is disabled to gate MapAnything processing."""

    def __init__(self, pipeline: DS8Pipeline) -> None:
        if BufferOperator is None:
            raise RuntimeError("pyservicemaker.BufferOperator is unavailable")
        super().__init__()
        self.pipeline = pipeline
        self._last_drop_logged = False

    def handle_buffer(self, buffer) -> bool:  # type: ignore[override]
        enabled = getattr(self.pipeline, "depth_enabled", False)
        if not enabled:
            if not self._last_drop_logged:
                logger.debug("DepthGateOperator dropping buffer (depth disabled)")
                self._last_drop_logged = True
            return False
        self._last_drop_logged = False
        return True


class LatencyProbeOperator(_BufferOperatorBase):  # pragma: no cover - runtime only
    """End-of-pipeline latency sampler using NVDS built-in latency measurement."""

    def __init__(self, pipeline: DS8Pipeline) -> None:
        if BufferOperator is None:
            raise RuntimeError("pyservicemaker.BufferOperator is unavailable")
        super().__init__()
        self.pipeline = pipeline
        self._warned = False
        self._warned_unsupported = False

    def handle_buffer(self, buffer) -> bool:  # type: ignore[override]
        collector = getattr(self.pipeline, "latency_collector", None)
        if collector is None or not getattr(collector, "enabled", False):
            return True

        mod = getattr(type(buffer), "__module__", "")
        if not mod.startswith("pyservicemaker"):
            if not self._warned_unsupported:
                self._warned_unsupported = True
                logger.warning(
                    "Latency probe: buffer type %s is unsupported; disabling latency stats",
                    type(buffer).__name__,
                )
            collector.disable("unsupported_buffer_type")
            return True

        try:
            collector.record_from_sm_buffer(buffer)
        except Exception as exc:
            if not self._warned:
                self._warned = True
                logger.warning("Latency probe: SM buffer path failed (%s)", exc)
        return True


class DecodedProgressProbeOperator(_BufferOperatorBase):  # pragma: no cover - runtime only
    """Count buffers at the canonical post-decode/dewarp liveness boundary."""

    def __init__(
        self, monitor: DecodedSourceProgressMonitor, source_id: int
    ) -> None:
        if BufferOperator is None:
            raise RuntimeError("pyservicemaker.BufferOperator is unavailable")
        super().__init__()
        self.monitor = monitor
        self.source_id = int(source_id)

    def handle_buffer(self, buffer) -> bool:  # type: ignore[override]
        self.monitor.record_progress(self.source_id)
        return True


_PIPELINE_SINGLETON: Optional[DS8Pipeline] = None


def _safe_add(ds_pipeline: Optional[DSPipeline], component: Component, errors: List[str]) -> None:
    if ds_pipeline is None:
        return
    try:
        properties = dict(component.config) if component.config else None
        ds_pipeline.add(component.element, component.name, properties)
    except Exception as exc:  # pragma: no cover - depends on runtime plugins
        errors.append(f"add:{component.name}:{redact_runtime_secrets(exc)}")


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
            errors.append(f"set:{component.name}:{redact_runtime_secrets(exc)}")

    if applied:
        return

    if not hasattr(ds_pipeline, "set_property"):
        return

    for key, value in cfg.items():
        try:
            ds_pipeline.set_property(component.name, key, value)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - depends on DS backend
            errors.append(f"prop:{component.name}:{key}:{redact_runtime_secrets(exc)}")


def _safe_link(ds_pipeline: Optional[DSPipeline], errors: List[str], *names: str) -> None:
    """Link pipeline components by name, recording failures as actionable errors."""
    if ds_pipeline is None or len(names) < 2:
        return
    path = "->".join(names)
    try:
        ds_pipeline.link(*names)
    except Exception as exc:  # pragma: no cover - depends on runtime plugins
        errors.append(f"link:{path}:{exc}")


def _safe_link_with_hints(
    ds_pipeline: Optional[DSPipeline],
    errors: List[str],
    source: str,
    sink: str,
    source_hint: str,
    sink_hint: str,
) -> None:
    """Link pipeline components with explicit pad hints (e.g., streammux sink pads)."""
    if ds_pipeline is None:
        return
    try:
        ds_pipeline.link((source, sink), (source_hint, sink_hint))
    except Exception as exc:  # pragma: no cover - depends on runtime plugins
        errors.append(f"link:{source}->{sink}({source_hint}->{sink_hint}):{exc}")


def _read_dewarper_output_size(config_path: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse nvdewarper config to extract output-width/output-height."""
    if not config_path:
        return None
    path = Path(config_path)
    if not path.exists():
        return None
    width = height = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("output-width="):
                try:
                    width = int(raw.split("=", 1)[1].strip())
                except Exception:
                    continue
            elif raw.startswith("output-height="):
                try:
                    height = int(raw.split("=", 1)[1].strip())
                except Exception:
                    continue
    except Exception:
        return None
    if width and height:
        return width, height
    return None


def _attach_depth_gate(pipeline: DS8Pipeline) -> None:
    """Attach a Probe-wrapped BufferOperator gate to the configured depth branch."""
    # If an upstream valve was configured, treat gating as supported without BufferOperator.
    valve = pipeline._resolve_valve()
    if valve is not None:
        pipeline.depth_gate_supported = True
        # Ensure initial drop state respects current depth_enabled flag.
        try:
            if pipeline.ds_pipeline is not None:
                node = pipeline.ds_pipeline[valve.name]  # type: ignore[index]
                node.set({"drop": not pipeline.depth_enabled})
        except Exception:
            pipeline.errors.append(f"depth_valve_set:{valve.name}")
        return
    if BufferOperator is None or Probe is None:
        pipeline.depth_gate_supported = False
        return
    if pipeline.ds_pipeline is None:
        pipeline.depth_gate_supported = False
        return
    target = pipeline.depth_gate_attach
    if not target or target not in pipeline.components:
        pipeline.depth_gate_supported = False
        return
    try:
        gate = DepthGateOperator(pipeline)
        probe = Probe("depth_gate", gate)
        pipeline.ds_pipeline.attach(target, probe)
        pipeline.depth_gate_supported = True
    except Exception as exc:  # pragma: no cover - runtime dependent
        pipeline.depth_gate_supported = False
        pipeline.errors.append(f"depth_gate_attach:{target}:{exc}")


def _attach_fps_probes(pipeline: DS8Pipeline) -> None:
    """Optionally attach lightweight FPS probes to key nodes for debugging.

    Enabled only when NOESIS_DS8_FPS_PROBE is truthy (\"1\", \"true\", \"yes\", \"on\").
    Probes log cumulative frame counts and approximate FPS at INFO level.
    """
    if BufferOperator is None or Probe is None:
        return
    if pipeline.ds_pipeline is None:
        return

    flag = os.environ.get("NOESIS_DS8_FPS_PROBE", "")
    if str(flag).strip().lower() not in ("1", "true", "yes", "on"):
        return

    ds = pipeline.ds_pipeline

    class _FPSProbe(BufferOperator):  # pragma: no cover - runtime diagnostics
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name
            self._count = 0
            self._t0 = time.time()

        def handle_buffer(self, buffer) -> bool:  # type: ignore[override]
            self._count += 1
            if self._count in (1, 10, 100) or (self._count % 100) == 0:
                now = time.time()
                dt = max(0.001, now - self._t0)
                fps = self._count / dt
                logger.info("FPSProbe[%s]: frames=%d, fps=%.2f", self.name, self._count, fps)
            return True

    debug_nodes = [
        # Core graph stages (sources do not support probes reliably here)
        "streammux",
        "yolo11_pgie",
        "mapanything_queue",
        "mapanything_valve",
        "mapanything_fullframe",
        "tracker",
        "analytics",
        "tiler",
        "osd",
    ]

    for node_name in debug_nodes:
        if node_name not in pipeline.components:
            continue
        try:
            probe = Probe(f"fps_{node_name}", _FPSProbe(node_name))
            ds.attach(node_name, probe)
            logger.info("FPS probe attached to '%s' (NOESIS_DS8_FPS_PROBE=1)", node_name)
        except Exception as exc:  # pragma: no cover - runtime dependent
            pipeline.errors.append(f"fps_probe_attach:{node_name}:{exc}")


def _attach_latency_probe(pipeline: DS8Pipeline) -> None:
    """Attach an end-of-pipeline latency probe and initialize the collector."""
    # Always construct the collector so stats can surface enabled/disabled reasons.
    if pipeline.latency_collector is None:
        try:
            window_sec = float(os.environ.get("NOESIS_DS8_LATENCY_WINDOW_SEC", "10") or "10")
        except Exception:
            window_sec = 10.0
        pipeline.latency_collector = LatencyCollector(window_sec=window_sec)

    flag = os.environ.get("NOESIS_DS8_LATENCY_PROBE")
    if flag is not None and str(flag).strip().lower() in ("0", "false", "no", "n", "off"):
        return
    if BufferOperator is None or Probe is None:
        return
    if pipeline.ds_pipeline is None:
        return
    # Service Maker buffer probes attach to output pads; the OSD is the last stable
    # element before the sink tee and optional RTSP branches.
    if "osd" not in pipeline.components:
        return
    try:
        probe = Probe("latency_probe", LatencyProbeOperator(pipeline))
        pipeline.ds_pipeline.attach("osd", probe, tips="src")
        logger.info(
            "Latency probe attached to osd:src (window_sec=%.1f, env NVDS_ENABLE_LATENCY_MEASUREMENT=%s)",
            float(pipeline.latency_collector.window_sec),
            os.environ.get("NVDS_ENABLE_LATENCY_MEASUREMENT"),
        )
    except Exception as exc:  # pragma: no cover - runtime dependent
        # Non-fatal: latency telemetry should never prevent the pipeline from starting.
        logger.warning("Latency probe attach failed (non-fatal): %s", exc)


def _attach_source_progress_probes(pipeline: DS8Pipeline) -> None:
    """Attach mandatory per-source decoded/dewarped progress probes."""

    monitor = pipeline.source_progress_monitor
    targets = pipeline.source_progress_targets
    if monitor is None and not targets:
        return
    if monitor is None or not targets:
        pipeline.errors.append("source_progress:missing_monitor_or_targets")
        return
    if isinstance(pipeline.ds_pipeline, _NoopDSPipeline):
        return
    if BufferOperator is None or Probe is None or pipeline.ds_pipeline is None:
        pipeline.errors.append("source_progress:servicemaker_probe_unavailable")
        return
    for source_id, target in sorted(targets.items()):
        if target not in pipeline.components:
            pipeline.errors.append(f"source_progress:missing_target:{target}")
            continue
        try:
            probe = Probe(
                f"source_progress_{source_id}",
                DecodedProgressProbeOperator(monitor, source_id),
            )
            pipeline.ds_pipeline.attach(target, probe, tips="src")
            logger.info(
                "Decoded progress probe attached to source=%d target=%s:src",
                source_id,
                target,
            )
        except Exception as exc:  # pragma: no cover - runtime dependent
            pipeline.errors.append(
                f"source_progress_attach:{source_id}:{target}:"
                f"{redact_runtime_secrets(exc)}"
            )


def build_pipeline(yaml_path: str | Path) -> DS8Pipeline:
    """Create the DS8 pipeline skeleton from YAML using pyservicemaker primitives."""
    global _PIPELINE_SINGLETON

    def _env_truthy(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return bool(default)
        val = str(raw).strip().lower()
        if val in ("1", "true", "yes", "y", "on"):
            return True
        if val in ("0", "false", "no", "n", "off"):
            return False
        return bool(default)

    def _is_local_mp4_uri(uri: str) -> bool:
        value = str(uri or "").strip()
        if not value:
            return False
        if not value.lower().startswith("file:"):
            return False
        return value.lower().endswith((".mp4", ".mkv"))

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(path)

    cfg: Dict[str, Any] = load_pipeline_config(path, materialize_secrets=True)

    errors: List[str] = []

    # Normalize relative paths in the configuration to be absolute, resolved
    # relative to the YAML file directory. This ensures DS plugins can locate
    # engine and config files regardless of the current working directory.
    base_dir = path.parent.resolve()
    runtime_output_root = Path(
        os.environ.get("NOESIS_BUILD_DIR", DS9_ROOT / "build")
    ).expanduser().resolve(strict=False)

    def _abs_or_same(p: Any) -> Any:
        if isinstance(p, str) and p:
            path_obj = Path(p)
            if path_obj.is_absolute():
                return p
            # If the path starts with a known top-level folder, resolve from repo root
            if p.startswith("DS9/"):
                return str((REPO_ROOT / p).resolve())
            if p.startswith(("build/", "config/", "models/", "pipelines/")):
                return str((DS9_ROOT / p).resolve())
            # Otherwise resolve relative to the YAML location
            return str((base_dir / p).resolve())
        return p

    def _nvinfer_props(raw: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(raw) if isinstance(raw, dict) else {}
        engine = cfg.pop("engine", None)
        if isinstance(engine, str) and engine:
            cfg["model-engine-file"] = _abs_or_same(engine)
        bs = cfg.pop("batch_size", cfg.pop("batch-size", None))
        if bs is not None:
            try:
                cfg["batch-size"] = int(bs)
            except Exception:
                cfg["batch-size"] = bs
        gie_id = cfg.pop("gie_id", cfg.pop("gie-id", None))
        if gie_id is not None:
            try:
                cfg["unique-id"] = int(gie_id)
            except Exception:
                cfg["unique-id"] = gie_id
        cfg.pop("network_mode", None)
        attach = cfg.pop("attach_tensor_meta", None)
        if attach is not None:
            cfg["output-tensor-meta"] = bool(attach)
        if "force_engine_rebuild" in cfg:
            raise EngineOnlyRuntimeError(
                "force_engine_rebuild is forbidden in a production runtime config; "
                "build engines with an explicit offline maintenance command"
            )
        # Tensor-meta product fields are consumed by telemetry/identity hooks.
        cfg.pop("layer", None)
        cfg.pop("embedding_dim", None)
        cfg.pop("model_size", None)
        cfg.pop("score_threshold", None)
        cfg.pop("kpt_threshold", None)
        cfg.pop("letterbox", None)
        # Runtime-only selector consumed by the MapAnything post-process hook.
        # It is not an nvinfer property.
        cfg.pop("profile", None)
        cfg.pop("backend", None)
        cfg.pop("metric_focal_denominator", None)
        cfg.pop("enable", None)
        cfg.pop("name", None)
        return cfg

    models_cfg = cfg.get("models", {}) or {}
    if not isinstance(models_cfg, dict):
        raise EngineOnlyRuntimeError("models must be a mapping with an active pgie entry")
    if not isinstance(models_cfg.get("pgie"), dict) or not models_cfg["pgie"]:
        raise EngineOnlyRuntimeError(
            "production runtime requires an active models.pgie with config-file-path and engine"
        )
    for key, m in list(models_cfg.items()):
        if not isinstance(m, dict):
            continue
        if "force_engine_rebuild" in m:
            raise EngineOnlyRuntimeError(
                f"models.{key}.force_engine_rebuild is forbidden in production runtime config"
            )
        eng = m.get("engine")
        if isinstance(eng, str) and eng and not Path(eng).is_absolute():
            m["engine"] = _abs_or_same(eng)
        cfg_file = m.get("config-file-path") or m.get("config-file")
        if isinstance(cfg_file, str) and cfg_file:
            m_key = "config-file-path" if "config-file-path" in m else "config-file"
            m[m_key] = _abs_or_same(cfg_file)
        enabled = key == "pgie" or bool(m.get("enable", True))
        if not enabled:
            continue
        cfg_file = m.get("config-file-path") or m.get("config-file")
        eng = m.get("engine")
        if not cfg_file or not eng:
            raise EngineOnlyRuntimeError(
                f"active models.{key} must define both config-file-path and engine"
            )
        component_name = str(m.get("name") or key).strip() or str(key)
        runtime_config = materialize_nvinfer_engine_only_config(
            source_config=Path(str(cfg_file)),
            engine_path=Path(str(eng)),
            output_root=runtime_output_root,
            component_name=component_name,
            repo_root=REPO_ROOT,
        )
        m.pop("config-file", None)
        m["config-file-path"] = str(runtime_config)
    cfg["models"] = models_cfg

    ds_pipeline: Optional[Any] = None
    # Native Service Maker linking can segfault in unit-test environments.
    # Use a pure-Python stub automatically under pytest unless explicitly disabled.
    under_pytest = "PYTEST_CURRENT_TEST" in os.environ
    use_stub = _env_truthy("NOESIS_DS9_STUB_PIPELINE", default=False) or (
        under_pytest and not _env_truthy("NOESIS_DS9_FORCE_NATIVE_TEST_PIPELINE", default=False)
    )
    if use_stub:
        ds_pipeline = _NoopDSPipeline("noesis-ds9-stub")
        logger.warning(
            '{"event":"pipeline_backend_selected","backend":"synthetic_stub",'
            '"native_runtime":false,"promotable":false}'
        )
    else:
        if DSPipeline is None:
            errors.append("pyservicemaker:unavailable")
        if DSPipeline is not None:
            try:
                ds_pipeline = DSPipeline("noesis-ds8")
            except Exception as exc:  # pragma: no cover - depends on DS backend
                errors.append(f"init:{exc}")

    pipeline = DS8Pipeline(
        yaml_path=path,
        config=cfg,
        ds_pipeline=ds_pipeline,
        errors=errors,
        lifecycle_evidence=(
            synthetic_stub_lifecycle_evidence() if use_stub else None
        ),
    )

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

    # Parse mosaic_output config for RTSP/WebRTC toggle support
    mosaic_output_raw = cfg.get("mosaic_output") or {}
    deprecated_h264_keys = sorted(
        key
        for key in (
            "rtsp_iframeinterval",
            "rtsp_idrinterval",
            "rtsp_profile",
            "rtsp_preset_id",
        )
        if key in mosaic_output_raw
    )
    if deprecated_h264_keys:
        raise ValueError(
            "mosaic_output uses retired RTSP-prefixed H.264 keys: "
            + ", ".join(deprecated_h264_keys)
            + "; rename them to h264_*"
        )
    rtsp_enabled = bool(mosaic_output_raw.get("rtsp_enabled", False))
    rtsp_port = int(mosaic_output_raw.get("rtsp_port", 8554) or 8554)
    rtsp_path = str(mosaic_output_raw.get("rtsp_path", "mosaic")).strip() or "mosaic"
    mosaic_webrtc_enabled = bool(mosaic_output_raw.get("mosaic_webrtc_enabled", False))
    video_bitrate_kbps = int(mosaic_output_raw.get("video_bitrate_kbps", 12000))
    if not 1000 <= video_bitrate_kbps <= 50000:
        raise ValueError("mosaic_output.video_bitrate_kbps must be between 1000 and 50000")
    encoder_cfg = str(
        mosaic_output_raw.get("encoder", "nvv4l2h264enc")
    ).strip()
    if encoder_cfg != "nvv4l2h264enc":
        raise ValueError(
            "mosaic_output.encoder must be nvv4l2h264enc; the canonical GPU path has no fallback"
        )
    # GOP tuning for the explicit NVENC node. Default IDR every 10 frames so
    # browser recovery after any AU loss is short (~0.3s at 30 fps).
    h264_iframeinterval = int(mosaic_output_raw.get("h264_iframeinterval", 10))
    h264_idrinterval = int(mosaic_output_raw.get("h264_idrinterval", 10))
    if h264_iframeinterval < 1 or h264_idrinterval < 1:
        raise ValueError("mosaic H.264 iframeinterval and idrinterval must be positive")
    h264_profile = int(mosaic_output_raw.get("h264_profile", 0))
    h264_preset_id = int(mosaic_output_raw.get("h264_preset_id", 4))
    if not 1 <= h264_preset_id <= 7:
        raise ValueError("mosaic_output.h264_preset_id must be between 1 and 7")
    mosaic_h264_shm_socket = str(
        os.environ.get("NOESIS_MOSAIC_H264_SHM", "")
        or mosaic_output_raw.get("mosaic_h264_shm_socket", "")
        or "/tmp/noesis-mosaic-h264"
    ).strip() or "/tmp/noesis-mosaic-h264"
    mosaic_h264_shm_size_bytes = int(
        mosaic_output_raw.get("mosaic_h264_shm_size_bytes", 64 * 1024 * 1024)
    )
    if not 4 * 1024 * 1024 <= mosaic_h264_shm_size_bytes <= 512 * 1024 * 1024:
        raise ValueError(
            "mosaic_output.mosaic_h264_shm_size_bytes must be between 4 MiB and 512 MiB"
        )

    # Environment overrides so toggles actually affect the built graph
    env_rtsp = os.environ.get("NOESIS_MOSAIC_RTSP_ENABLED")
    if env_rtsp is not None:
        rtsp_enabled = str(env_rtsp).strip().lower() in ("1", "true", "yes", "on")
    env_webrtc = os.environ.get("NOESIS_MOSAIC_WEBRTC_ENABLED")
    if env_webrtc is not None:
        mosaic_webrtc_enabled = str(env_webrtc).strip().lower() in ("1", "true", "yes", "on")
    # WebRTC no longer depends on RTSP; it consumes encoded AUs via SHM.
    mosaic_encode_enabled = bool(rtsp_enabled or mosaic_webrtc_enabled)

    encoder_name = encoder_cfg

    cfg["mosaic_output"] = {
        "rtsp_enabled": rtsp_enabled,
        "rtsp_port": rtsp_port,
        "rtsp_path": rtsp_path,
        "mosaic_webrtc_enabled": mosaic_webrtc_enabled,
        "mosaic_encode_enabled": mosaic_encode_enabled,
        "video_bitrate_kbps": video_bitrate_kbps,
        "encoder": encoder_name,
        "h264_iframeinterval": h264_iframeinterval,
        "h264_idrinterval": h264_idrinterval,
        "h264_profile": h264_profile,
        "h264_preset_id": h264_preset_id,
        "mosaic_h264_shm_socket": mosaic_h264_shm_socket,
        "mosaic_h264_shm_size_bytes": mosaic_h264_shm_size_bytes,
    }

    streammux_cfg = dict(cfg.get("streammux") or {})
    streammux_cfg.setdefault("batch-size", batch_size)
    streammux_cfg.setdefault("width", 1920)
    streammux_cfg.setdefault("height", 1080)
    streammux_cfg.setdefault("live-source", 1)
    # Preserve input aspect by padding when streammux scales to the configured size.
    streammux_cfg.setdefault("enable-padding", 1)
    first_gpu = sources[0].get("gpu-id") if sources else 0
    streammux_cfg.setdefault("gpu-id", first_gpu if first_gpu is not None else 0)
    try:
        zero_copy_gpu_id = int(streammux_cfg.get("gpu-id", 0) or 0)
    except Exception:
        zero_copy_gpu_id = 0
    streammux_cfg["gpu-id"] = zero_copy_gpu_id
    try:
        zero_copy_nvbuf_memory_type = int(streammux_cfg.get("nvbuf-memory-type", 0) or 0)
    except Exception:
        zero_copy_nvbuf_memory_type = 0
    streammux_cfg["nvbuf-memory-type"] = zero_copy_nvbuf_memory_type

    def _dewarper_enabled(source_cfg: Dict[str, Any]) -> bool:
        raw = source_cfg.get("dewarper")
        return isinstance(raw, dict) and bool(raw.get("enable", False))

    use_dewarper = any(_dewarper_enabled(s) for s in sources)
    streammux_element = str(streammux_cfg.pop("element", "") or "").strip()
    sources_have_element = any(
        isinstance(s, dict) and str(s.get("element") or "").strip() for s in sources
    )
    if not streammux_element:
        # If sources declare explicit elements (e.g., nvurisrcbin), prefer per-source ingest
        # through nvstreammux even when no dewarper is enabled.
        if sources_have_element:
            streammux_element = "nvstreammux"
        else:
            streammux_element = "nvstreammux" if use_dewarper else "nvmultiurisrcbin"

    per_source_pipeline = use_dewarper or streammux_element.lower() == "nvstreammux"

    # Track source output nodes when linking into nvstreammux.
    source_nodes: List[Tuple[str, int]] = []
    source_progress_labels: Dict[int, str] = {}
    source_progress_targets: Dict[int, str] = {}

    # Precompute URIs for tiler layout / diagnostics.
    uris = [str(s.get("uri") or "").strip() for s in sources if str(s.get("uri") or "").strip()]

    if per_source_pipeline:
        # Per-source pipeline: nvurisrcbin → decoded-frame isolation queue →
        # (optional dewarper) → nvstreammux.
        streammux = Component(
            name="streammux",
            element=streammux_element,
            config=streammux_cfg,
            downstream=["yolo11_pgie"],
        )
        pipeline.components[streammux.name] = streammux
        _safe_add(ds_pipeline, streammux, pipeline.errors)
        _apply_component_config(ds_pipeline, streammux, pipeline.errors)

        loop_local_mp4 = _env_truthy("NOESIS_DS8_LOOP_LOCAL_MP4", default=True)
        for idx, source_cfg in enumerate(sources):
            camera_id = str(
                source_cfg.get("uri_secret") or f"source-{idx}"
            ).strip()
            source_progress_labels[idx] = camera_id or f"source-{idx}"
            props = dict(source_cfg)
            props.pop("uri_secret", None)
            dewarp_cfg_raw = props.pop("dewarper", None)
            element = props.pop("element", "nvurisrcbin")
            uri = str(props.get("uri") or "").strip()
            if uri:
                props["uri"] = uri
            props.setdefault("source-id", idx)
            props.setdefault("gpu-id", zero_copy_gpu_id)
            if uri.lower().startswith("rtsp"):
                # Keep conservative RTSP reconnect defaults for live feeds.
                props.setdefault("rtsp-reconnect-interval", 10)
                props.setdefault("init-rtsp-reconnect-interval", 5)
                props.setdefault("rtsp-reconnect-attempts", 4)
            if loop_local_mp4 and _is_local_mp4_uri(uri):
                props.setdefault("file-loop", True)

            source = Component(
                name=f"source_{idx}",
                element=element,
                config=props,
                downstream=[],
            )
            pipeline.components[source.name] = source
            _safe_add(ds_pipeline, source, pipeline.errors)
            _apply_component_config(ds_pipeline, source, pipeline.errors)

            # Isolate each decoded source before dewarping/muxing so one
            # reconnecting camera cannot back-pressure the other sources. The
            # progress probe is attached after this queue, or after dewarping
            # when dewarping is enabled, so packet arrival alone is never
            # counted as decoded-frame progress.
            decode_queue = Component(
                name=f"source_decode_queue_{idx}",
                element="queue",
                config={
                    "leaky": 2,
                    "max-size-buffers": 4,
                    "max-size-bytes": 0,
                    "max-size-time": 0,
                },
                downstream=[],
            )
            pipeline.components[decode_queue.name] = decode_queue
            _safe_add(ds_pipeline, decode_queue, pipeline.errors)
            _apply_component_config(ds_pipeline, decode_queue, pipeline.errors)
            _safe_link(ds_pipeline, pipeline.errors, source.name, decode_queue.name)
            source.downstream = [decode_queue.name]

            dewarp_cfg = dict(dewarp_cfg_raw) if isinstance(dewarp_cfg_raw, dict) else {}
            dewarp_enabled = bool(dewarp_cfg.get("enable", False))
            if dewarp_enabled:
                dewarp_cfg.pop("enable", None)
                if "config-file" in dewarp_cfg:
                    dewarp_cfg["config-file"] = _abs_or_same(dewarp_cfg["config-file"])  # type: ignore[index]
                dewarp_out = _read_dewarper_output_size(dewarp_cfg.get("config-file"))
                dewarp_cfg.setdefault("source-id", idx)
                dewarp_cfg.setdefault("gpu-id", zero_copy_gpu_id)
                dewarp_cfg.setdefault("nvbuf-memory-type", zero_copy_nvbuf_memory_type)

                conv = Component(
                    name=f"dewarper_conv_{idx}",
                    element="nvvideoconvert",
                    config={
                        "gpu-id": zero_copy_gpu_id,
                        "nvbuf-memory-type": zero_copy_nvbuf_memory_type,
                    },
                    downstream=[],
                )
                caps_in = Component(
                    name=f"dewarper_caps_{idx}",
                    element="capsfilter",
                    config={"caps": "video/x-raw(memory:NVMM),format=RGBA"},
                    downstream=[],
                )
                dewarper = Component(
                    name=f"dewarper_{idx}",
                    element="nvdewarper",
                    config=dewarp_cfg,
                    downstream=[],
                )
                caps_out_caps = "video/x-raw(memory:NVMM),format=RGBA"
                if dewarp_out:
                    caps_out_caps = f"{caps_out_caps},width={dewarp_out[0]},height={dewarp_out[1]}"
                caps_out = Component(
                    name=f"dewarper_caps_out_{idx}",
                    element="capsfilter",
                    config={"caps": caps_out_caps},
                    downstream=[],
                )
                for comp in (conv, caps_in, dewarper, caps_out):
                    pipeline.components[comp.name] = comp
                    _safe_add(ds_pipeline, comp, pipeline.errors)
                    _apply_component_config(ds_pipeline, comp, pipeline.errors)

                _safe_link(ds_pipeline, pipeline.errors, decode_queue.name, conv.name)
                _safe_link(ds_pipeline, pipeline.errors, conv.name, caps_in.name)
                _safe_link(ds_pipeline, pipeline.errors, caps_in.name, dewarper.name)
                _safe_link(ds_pipeline, pipeline.errors, dewarper.name, caps_out.name)
                decode_queue.downstream = [conv.name]
                conv.downstream = [caps_in.name]
                caps_in.downstream = [dewarper.name]
                dewarper.downstream = [caps_out.name]
                source_nodes.append((caps_out.name, idx))
                source_progress_targets[idx] = caps_out.name
            else:
                source_nodes.append((decode_queue.name, idx))
                source_progress_targets[idx] = decode_queue.name
    else:
        # Multi-URI source path: nvmultiurisrcbin performs source ingest + mux.
        uri_list = ",".join(uris)
        sensor_id_list = ",".join(str(i) for i in range(len(uris))) if uris else ""

        multi_cfg: Dict[str, Any] = {
            "uri-list": uri_list,
            "sensor-id-list": sensor_id_list,
            "max-batch-size": batch_size,
            "width": streammux_cfg.get("width", 1920),
            "height": streammux_cfg.get("height", 1080),
            "batched-push-timeout": streammux_cfg.get("batched-push-timeout", 40000),
            "live-source": streammux_cfg.get("live-source", 1),
            "enable-padding": streammux_cfg.get("enable-padding", 1),
            "nvbuf-memory-type": zero_copy_nvbuf_memory_type,
            "sync-inputs": streammux_cfg.get("sync-inputs", 0),
            "gpu-id": zero_copy_gpu_id,
            # Keep behavioral knobs consistent with current deployment INI:
            # - Avoid propagating EOS downstream when all sources hit EOS.
            # - Disable REST control API (port=0) since DS8 drives URIs from YAML.
            "drop-pipeline-eos": 1,
            "cache-buffer": streammux_cfg.get("cache-buffer", 0),
            "sort-batch": streammux_cfg.get("sort-batch", 0),
            "align-first-buffer": streammux_cfg.get("align-first-buffer", 0),
            "port": "0",
        }
        loop_local_mp4 = _env_truthy("NOESIS_DS8_LOOP_LOCAL_MP4", default=True)
        local_mp4_indices = [idx for idx, uri in enumerate(uris) if _is_local_mp4_uri(uri)]
        if loop_local_mp4 and local_mp4_indices:
            # Verified via `gst-inspect-1.0 nvmultiurisrcbin`:
            # file-loop: Loop file sources after EOS. Src type must be source-type-uri
            # and uri starting with 'file:/'.
            multi_cfg["file-loop"] = True
            logger.info("DS8: enabled nvmultiurisrcbin file-loop for local mp4 sources: %s", local_mp4_indices)
        # Optional RTSP tuning derived from per-source config
        if sources:
            sample = dict(sources[0])
            if "latency" in sample:
                multi_cfg["latency"] = sample.get("latency", 100)
            if "select-rtp-protocol" in sample:
                multi_cfg["select-rtp-protocol"] = sample.get("select-rtp-protocol", 0)
            if "cudadec-memtype" in sample:
                multi_cfg["cudadec-memtype"] = sample.get("cudadec-memtype", 0)
        # Reasonable RTSP reconnect defaults for multi-URI ingest.
        multi_cfg.setdefault("rtsp-reconnect-interval", 10)
        multi_cfg.setdefault("init-rtsp-reconnect-interval", 5)
        multi_cfg.setdefault("rtsp-reconnect-attempts", 4)

        streammux = Component(
            name="streammux",
            element="nvmultiurisrcbin",
            config=multi_cfg,
            downstream=["yolo11_pgie"],
        )

        pipeline.components[streammux.name] = streammux
        _safe_add(ds_pipeline, streammux, pipeline.errors)
        _apply_component_config(ds_pipeline, streammux, pipeline.errors)

    if source_progress_targets:
        pipeline.source_progress_targets = dict(source_progress_targets)
        pipeline.source_progress_monitor = DecodedSourceProgressMonitor(
            source_progress_labels,
            policy=SourceProgressPolicy.from_environment(),
        )

    try:
        pipeline.frame_size = (
            int(streammux_cfg.get("width", 0) or 0),
            int(streammux_cfg.get("height", 0) or 0),
        )
    except Exception:
        pipeline.frame_size = (0, 0)

    preprocess_cfg = dict(cfg.get("preprocess") or {})
    preprocess_enabled = bool(preprocess_cfg.get("enable", True) and preprocess_cfg)
    preprocess_component: Optional[Component] = None
    if preprocess_enabled:
        preprocess_cfg.pop("enable", None)
        if "config-file" in preprocess_cfg:
            preprocess_cfg["config-file"] = _abs_or_same(preprocess_cfg["config-file"])  # type: ignore[index]
        preprocess_component = Component(
            name="preprocess",
            element=preprocess_cfg.pop("element", "nvdspreprocess"),
            config=preprocess_cfg,
            downstream=["yolo11_pgie"],
        )
        pipeline.components[preprocess_component.name] = preprocess_component
        _safe_add(ds_pipeline, preprocess_component, pipeline.errors)
        _apply_component_config(ds_pipeline, preprocess_component, pipeline.errors)
        streammux.downstream = [preprocess_component.name]

    pgie_cfg = _nvinfer_props(models.get("pgie", {}))
    primary = Component(
        name="yolo11_pgie",
        element="nvinfer",
        config=pgie_cfg,
        downstream=["main_tee"],
    )
    pipeline.components[primary.name] = primary
    _safe_add(ds_pipeline, primary, pipeline.errors)
    _apply_component_config(ds_pipeline, primary, pipeline.errors)
    if preprocess_component is None:
        streammux.downstream = [primary.name]
    else:
        preprocess_component.downstream = [primary.name]

    shutdown_eos = Component(
        name=pipeline.shutdown_eos_component_name,
        element="noesiseos",
        config={},
        downstream=[preprocess_component.name if preprocess_component else primary.name],
    )
    pipeline.components[shutdown_eos.name] = shutdown_eos
    _safe_add(ds_pipeline, shutdown_eos, pipeline.errors)
    _apply_component_config(ds_pipeline, shutdown_eos, pipeline.errors)
    streammux.downstream = [shutdown_eos.name]

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
    tracker_source_raw = tracker_cfg.pop("config-file", None)
    direct_tracker_source = tracker_cfg.pop("ll-config-file", None)
    if tracker_source_raw is not None and direct_tracker_source is not None:
        raise EngineOnlyRuntimeError(
            "tracker must not define both config-file and ll-config-file"
        )
    tracker_source_raw = (
        tracker_source_raw if tracker_source_raw is not None else direct_tracker_source
    )
    if tracker_source_raw is not None:
        tracker_source = Path(str(_abs_or_same(tracker_source_raw)))
        tracker_cfg["ll-config-file"] = str(
            materialize_nvtracker_engine_only_config(
                source_config=tracker_source,
                output_root=runtime_output_root,
                component_name="tracker",
                repo_root=REPO_ROOT,
            )
        )
    # Ensure ll-lib-file is provided when using NvDCF.
    ds_home = str(os.environ.get("NOESIS_DEEPSTREAM_HOME", "/opt/nvidia/deepstream/deepstream-9.0") or "").rstrip("/")
    tracker_cfg.setdefault(
        "ll-lib-file",
        f"{ds_home}/lib/libnvds_nvmultiobjecttracker.so",
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

    # Optional per-object ReID SGIE for stable identity assignment.
    reid: Optional[Component] = None
    reid_cfg_raw = models.get("reid")
    reid_enabled = False
    if isinstance(reid_cfg_raw, dict):
        reid_enabled = bool(reid_cfg_raw.get("enable", True)) and bool(reid_cfg_raw)
    if reid_enabled:
        raw = dict(reid_cfg_raw) if isinstance(reid_cfg_raw, dict) else {}
        reid_name = str(raw.get("name") or "reid_sgie").strip() or "reid_sgie"
        reid_cfg = _nvinfer_props(raw)
        reid = Component(
            name=reid_name,
            element="nvinfer",
            config=reid_cfg,
            downstream=["analytics"],
        )
        pipeline.components[reid.name] = reid
        _safe_add(ds_pipeline, reid, pipeline.errors)
        _apply_component_config(ds_pipeline, reid, pipeline.errors)
        tracker.downstream = [reid.name]

    # Optional per-object Pose SGIE (YOLO26 pose) for feature extraction.
    pose: Optional[Component] = None
    pose_cfg_raw = models.get("pose")
    pose_enabled = False
    if isinstance(pose_cfg_raw, dict):
        pose_enabled = bool(pose_cfg_raw.get("enable", True)) and bool(pose_cfg_raw)
    if pose_enabled:
        raw = dict(pose_cfg_raw) if isinstance(pose_cfg_raw, dict) else {}
        pose_name = str(raw.get("name") or "yolo26_pose").strip() or "yolo26_pose"
        pose_cfg = _nvinfer_props(raw)
        pose = Component(
            name=pose_name,
            element="nvinfer",
            config=pose_cfg,
            downstream=["analytics"],
        )
        pipeline.components[pose.name] = pose
        _safe_add(ds_pipeline, pose, pipeline.errors)
        _apply_component_config(ds_pipeline, pose, pipeline.errors)

    analytics_cfg_raw = cfg.get("analytics", {"config-file": "pipelines/config_nvdsanalytics_post.ini"})
    analytics_cfg = dict(analytics_cfg_raw) if isinstance(analytics_cfg_raw, dict) else {}
    analytics_enabled = bool(analytics_cfg.get("enable", True))
    env_analytics_path = os.environ.get("NOESIS_ANALYTICS_CONFIG")
    if env_analytics_path:
        analytics_cfg["stages_config"] = env_analytics_path
    stages_config_path = analytics_cfg.get("stages_config")
    if stages_config_path:
        try:
            stages_path = Path(_abs_or_same(stages_config_path))  # type: ignore[arg-type]
            with stages_path.open("r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            loaded_analytics = loaded.get("analytics")
            loaded_stages = loaded_analytics.get("stages") if isinstance(loaded_analytics, dict) else None
            if not isinstance(loaded_stages, dict) or not isinstance(loaded_stages.get("exclude"), dict):
                raise ValueError("analytics stages config must contain an exclude mapping")
            if env_analytics_path:
                analytics_cfg["stages"] = loaded_stages
            else:
                analytics_cfg.setdefault("stages", loaded_stages)
            analytics_cfg["stages_config"] = str(stages_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load analytics stages config: {exc}") from exc
    if "config-file" in analytics_cfg:
        analytics_cfg["config-file"] = _abs_or_same(analytics_cfg["config-file"])  # type: ignore[index]
    exclude_cfg = dict(analytics_cfg.get("exclude") or {})
    exclude_enabled = bool(exclude_cfg.get("enable", False))
    exclude_component: Optional[Component] = None
    if exclude_enabled:
        validate_exclusion_stream_coverage(sources, analytics_cfg)
        exclude_element = require_native_exclusion_element(exclude_cfg)
        exclude_cfg.pop("enable", None)
        exclude_cfg.pop("element", None)
        env_exclude_path = os.environ.get("NOESIS_ANALYTICS_EXCLUDE_CONFIG")
        if env_exclude_path:
            exclude_cfg["config-file"] = env_exclude_path
        if "config-file" in exclude_cfg:
            exclude_cfg["config-file"] = _abs_or_same(exclude_cfg["config-file"])  # type: ignore[index]
        exclude_component = Component(
            name="analytics_exclude",
            element=exclude_element,
            config=exclude_cfg,
            downstream=["tracker"],
        )
        pipeline.components[exclude_component.name] = exclude_component
        _safe_add(ds_pipeline, exclude_component, pipeline.errors)
        _apply_component_config(ds_pipeline, exclude_component, pipeline.errors)
    analytics_cfg.pop("exclude", None)
    cfg["analytics"] = analytics_cfg

    analytics = None
    if analytics_enabled and analytics_cfg.get("config-file"):
        analytics_component_cfg = dict(analytics_cfg)
        analytics_component_cfg.pop("stages", None)
        analytics_component_cfg.pop("stages_config", None)
        analytics = Component(
            name="analytics",
            element="nvdsanalytics",
            config=analytics_component_cfg,
            downstream=[],
        )
        pipeline.components[analytics.name] = analytics
        _safe_add(ds_pipeline, analytics, pipeline.errors)
        _apply_component_config(ds_pipeline, analytics, pipeline.errors)

    # Required baseline depth-tracking branch (DAv2 full-frame) plus optional MapAnything branch.
    depth_tracking: Optional[Component] = None
    depth_tracking_sink: Optional[Component] = None
    depth_tracking_queue: Optional[Component] = None
    depth_tracking_cfg_raw = models.get("depth_tracking")
    depth_tracking_enabled = False
    if isinstance(depth_tracking_cfg_raw, dict):
        depth_tracking_enabled = bool(depth_tracking_cfg_raw.get("enable", False)) and bool(depth_tracking_cfg_raw)

    primary_branch = exclude_component.name if exclude_component is not None else tracker.name
    tee_downstreams: List[str] = [primary_branch]

    if depth_tracking_enabled:
        depth_tracking_cfg = _nvinfer_props(dict(depth_tracking_cfg_raw))
        depth_tracking_name = str(depth_tracking_cfg.get("name", "depth_tracking_fullframe"))
        depth_tracking = Component(
            name=depth_tracking_name,
            element="nvinfer",
            config=depth_tracking_cfg,
            downstream=[],
        )
        depth_tracking_queue = Component(
            name="depth_tracking_queue",
            element="queue",
            config={"max-size-buffers": 4, "max-size-bytes": 0, "max-size-time": 0},
            downstream=[depth_tracking.name],
        )
        depth_tracking_sink = Component(
            name=f"{depth_tracking.name}_sink",
            element="fakesink",
            config={"sync": False},
            downstream=[],
        )
        pipeline.components[depth_tracking_queue.name] = depth_tracking_queue
        _safe_add(ds_pipeline, depth_tracking_queue, pipeline.errors)
        _apply_component_config(ds_pipeline, depth_tracking_queue, pipeline.errors)
        pipeline.components[depth_tracking.name] = depth_tracking
        _safe_add(ds_pipeline, depth_tracking, pipeline.errors)
        _apply_component_config(ds_pipeline, depth_tracking, pipeline.errors)
        pipeline.components[depth_tracking_sink.name] = depth_tracking_sink
        _safe_add(ds_pipeline, depth_tracking_sink, pipeline.errors)
        _apply_component_config(ds_pipeline, depth_tracking_sink, pipeline.errors)
        depth_tracking.downstream = [depth_tracking_sink.name]
        tee_downstreams.append(depth_tracking_queue.name)

    # Optional full-frame MapAnything branch; allow disabling via YAML (models.mapanything.enable=false)
    mapanything: Optional[Component] = None
    mapanything_sink: Optional[Component] = None
    mapanything_rgb_convert: Optional[Component] = None
    mapanything_rgb_caps: Optional[Component] = None
    mapanything_cfg_raw = models.get("mapanything")
    sgie_enabled = False
    if isinstance(mapanything_cfg_raw, dict):
        sgie_enabled = bool(mapanything_cfg_raw.get("enable", True)) and bool(mapanything_cfg_raw)

    if sgie_enabled:
        mapanything_cfg = dict(mapanything_cfg_raw)
        mapanything_cfg.setdefault("attach_tensor_meta", True)
        mapanything_cfg.setdefault("gie_id", 2)
        mapanything_cfg = _nvinfer_props(mapanything_cfg)
        mapanything_name = str(mapanything_cfg.get("name", "mapanything_fullframe"))
        mapanything = Component(
            name=mapanything_name,
            element="nvinfer",
            config=mapanything_cfg,
            downstream=[],
        )
        # Insert an upstream valve so MapAnything can be gated without consuming GPU.
        mapanything_queue = Component(
            name="mapanything_queue",
            element="queue",
            config={"max-size-buffers": 1, "leaky": 2},
            downstream=[],
        )
        pipeline.components[mapanything_queue.name] = mapanything_queue
        _safe_add(ds_pipeline, mapanything_queue, pipeline.errors)
        _apply_component_config(ds_pipeline, mapanything_queue, pipeline.errors)

        mapanything_valve = Component(
            name="mapanything_valve",
            element="valve",
            # Default to dropping so MapAnything inference stays OFF unless explicitly enabled.
            # activate() briefly opens the valve to allow preroll/caps negotiation.
            config={"drop": True, "drop-mode": 1},  # forward-sticky-events
            downstream=[],
        )
        pipeline.components[mapanything_valve.name] = mapanything_valve
        _safe_add(ds_pipeline, mapanything_valve, pipeline.errors)
        _apply_component_config(ds_pipeline, mapanything_valve, pipeline.errors)

        mapanything_rgb_convert = Component(
            name="mapanything_rgb_convert",
            element="nvvideoconvert",
            config={
                "gpu-id": zero_copy_gpu_id,
                "nvbuf-memory-type": zero_copy_nvbuf_memory_type,
            },
            downstream=[],
        )
        pipeline.components[mapanything_rgb_convert.name] = mapanything_rgb_convert
        _safe_add(ds_pipeline, mapanything_rgb_convert, pipeline.errors)
        _apply_component_config(
            ds_pipeline,
            mapanything_rgb_convert,
            pipeline.errors,
        )

        mapanything_rgb_caps = Component(
            name="mapanything_rgb_caps",
            element="capsfilter",
            config={
                "caps": (
                    "video/x-raw(memory:NVMM),format=RGB,"
                    f"width={int(streammux_cfg.get('width', 1920) or 1920)},"
                    f"height={int(streammux_cfg.get('height', 1080) or 1080)}"
                )
            },
            downstream=[],
        )
        pipeline.components[mapanything_rgb_caps.name] = mapanything_rgb_caps
        _safe_add(ds_pipeline, mapanything_rgb_caps, pipeline.errors)
        _apply_component_config(
            ds_pipeline,
            mapanything_rgb_caps,
            pipeline.errors,
        )

        pipeline.components[mapanything.name] = mapanything
        _safe_add(ds_pipeline, mapanything, pipeline.errors)
        tee_downstreams.append(mapanything_queue.name)
        _apply_component_config(ds_pipeline, mapanything, pipeline.errors)
        pipeline.depth_gate_attach = mapanything.name
        pipeline.valve_name = mapanything_valve.name
        pipeline.depth_gate_supported = True
        # Drop MapAnything branch output after tensor/RGB capture; tensors and
        # the exact source surface are consumed together by the downstream
        # buffer probe.
        mapanything_sink = Component(
            name=f"{mapanything.name}_sink",
            element="fakesink",
            config={"sync": False},
            downstream=[],
        )
        pipeline.components[mapanything_sink.name] = mapanything_sink
        _safe_add(ds_pipeline, mapanything_sink, pipeline.errors)
        _apply_component_config(ds_pipeline, mapanything_sink, pipeline.errors)
        # nvinfer accepts NV12/RGBA, while Buffer.extract() supports RGB only.
        # Convert after nvinfer and attach the capture probe after the RGB caps.
        # The normally closed upstream valve keeps both inference and conversion
        # off the idle/hot lane.
        mapanything_queue.downstream = [mapanything_valve.name]
        mapanything_valve.downstream = [mapanything.name]
        mapanything.downstream = [mapanything_rgb_convert.name]
        mapanything_rgb_convert.downstream = [mapanything_rgb_caps.name]
        mapanything_rgb_caps.downstream = [mapanything_sink.name]

    tee_component.downstream = tee_downstreams

    world_observation_stage = Component(
        name="world_observation_stage",
        element="queue",
        config={"max-size-buffers": 4, "max-size-bytes": 0, "max-size-time": 0},
        downstream=["tracking_telemetry_stage"],
    )
    tracking_telemetry_stage = Component(
        name="tracking_telemetry_stage",
        element="queue",
        config={"max-size-buffers": 4, "max-size-bytes": 0, "max-size-time": 0},
        downstream=["tiler"],
    )
    for stage_component in (world_observation_stage, tracking_telemetry_stage):
        pipeline.components[stage_component.name] = stage_component
        _safe_add(ds_pipeline, stage_component, pipeline.errors)
        _apply_component_config(ds_pipeline, stage_component, pipeline.errors)

    # Insert a tiled renderer stage in DS8 path (mosaic). This avoids per-camera branches.
    source_count = len(uris) or 1
    tiler_square_seq_grid_env = str(os.environ.get("NOESIS_MOSAIC_TILER_SQUARE_SEQ_GRID", "")).strip().lower()
    tiler_square_seq_grid = tiler_square_seq_grid_env in ("1", "true", "yes", "on")
    tiler_columns_env = str(os.environ.get("NOESIS_MOSAIC_TILER_COLUMNS", "")).strip()
    tiler_rows_env = str(os.environ.get("NOESIS_MOSAIC_TILER_ROWS", "")).strip()
    tiler_columns_override = tiler_columns_env.isdigit()
    tiler_rows_override = tiler_rows_env.isdigit()
    tiler_columns: Optional[int] = int(tiler_columns_env) if tiler_columns_override else None
    tiler_rows: Optional[int] = int(tiler_rows_env) if tiler_rows_override else None
    if tiler_columns_override or tiler_rows_override:
        # Explicit layout override; disable square auto-tiling.
        tiler_square_seq_grid = False

    tiler_cfg: Dict[str, Any] = {
        "gpu-id": zero_copy_gpu_id,
        "width": streammux_cfg.get("width", 1920),
        "height": streammux_cfg.get("height", 1080),
    }
    if tiler_square_seq_grid:
        # Use the plugin's square layout mode to keep tile aspect aligned to the output aspect.
        # This avoids the common 1xN layout that makes tiles tall/skinny when output is 1920x1080.
        tiler_cfg["square-seq-grid"] = True
    else:
        base_columns = tiler_columns if tiler_columns is not None else 3
        base_rows = tiler_rows if tiler_rows is not None else 1
        columns = max(1, base_columns)
        rows = max(1, base_rows)
        if not tiler_rows_override and source_count > columns * rows:
            rows = max(rows, math.ceil(source_count / columns))
        tiler_cfg["columns"] = columns
        tiler_cfg["rows"] = rows
        if columns > 1 or rows > 1:
            # Preserve per-tile aspect ratio by sizing the *overall* mosaic to a multiple of a
            # 16:9-ish tile size, rather than forcing tiles into streammux-sized output cells.
            mux_w = int(streammux_cfg.get("width", 1920) or 1920)
            mux_h = int(streammux_cfg.get("height", 1080) or 1080)
            mux_w = max(1, mux_w)
            mux_h = max(1, mux_h)
            tile_h_env = str(os.environ.get("NOESIS_MOSAIC_TILER_TILE_HEIGHT", "")).strip()
            tile_h = int(tile_h_env) if tile_h_env.isdigit() else min(mux_h, 720)
            tile_h = max(2, int(tile_h))
            if tile_h % 2:
                tile_h += 1
            mux_aspect = mux_w / mux_h if mux_h else (16.0 / 9.0)
            tile_w = int(round(tile_h * mux_aspect))
            tile_w = max(2, tile_w)
            if tile_w % 2:
                tile_w += 1
            tiler_cfg["width"] = int(tile_w * columns)
            tiler_cfg["height"] = int(tile_h * rows)

    logger.info(
        json.dumps(
            {
                "event": "ds8_mosaic_tiler_config",
                "source_count": source_count,
                "width": int(tiler_cfg.get("width", 0) or 0),
                "height": int(tiler_cfg.get("height", 0) or 0),
                "square_seq_grid": bool(tiler_cfg.get("square-seq-grid", False)),
                "columns": tiler_cfg.get("columns"),
                "rows": tiler_cfg.get("rows"),
                "streammux_enable_padding": int(streammux_cfg.get("enable-padding", 0) or 0),
            },
            separators=(",", ":"),
        )
    )

    tiler = Component(
        name="tiler",
        element="nvmultistreamtiler",
        config=tiler_cfg,
        downstream=[],
    )
    pipeline.components[tiler.name] = tiler
    _safe_add(ds_pipeline, tiler, pipeline.errors)
    _apply_component_config(ds_pipeline, tiler, pipeline.errors)

    osd_process_mode = 0
    osd = Component(
        name="osd",
        element="nvdsosd",
        config={
            # Keep OSD on GPU so the tiler -> OSD -> encoder path stays zero-copy/NVMM.
            "process-mode": osd_process_mode,
            # Show instance segmentation masks from NvDsInferInstanceMaskInfo
            "display-mask": 1,
            # Hide bbox rectangles to emphasize masks; set to 1 if you want both
            "display-bbox": 0,
            # Keep labels visible (class name + confidence)
            "display-text": 1,
        },
        downstream=[],
    )
    pipeline.components[osd.name] = osd
    _safe_add(ds_pipeline, osd, pipeline.errors)
    _apply_component_config(ds_pipeline, osd, pipeline.errors)

    # Tee branches should not be allowed to stall the whole graph if a branch is
    # missing/unlinked at runtime (e.g. optional RTSP/WebRTC output paths).
    sink_tee = Component(name="sink_tee", element="tee", config={"allow-not-linked": True}, downstream=[])
    pipeline.components[sink_tee.name] = sink_tee
    _safe_add(ds_pipeline, sink_tee, pipeline.errors)
    _apply_component_config(ds_pipeline, sink_tee, pipeline.errors)

    # Route main chain through tiler → osd → sink tee → sinks
    if exclude_component is not None:
        exclude_component.downstream = [tracker.name]

    if analytics is not None:
        tracker.downstream = [analytics.name]
        chain_start = analytics
    else:
        chain_start = tracker

    chain: List[Component] = []
    if reid is not None:
        chain.append(reid)
    if pose is not None:
        chain.append(pose)

    if chain:
        chain_start.downstream = [chain[0].name]
        for idx in range(len(chain) - 1):
            chain[idx].downstream = [chain[idx + 1].name]
        chain[-1].downstream = [world_observation_stage.name]
    else:
        chain_start.downstream = [world_observation_stage.name]

    if mapanything is not None:
        if mapanything_rgb_convert is None:
            pipeline.errors.append("mapanything_exact_rgb_chain_missing")
        else:
            mapanything.downstream = [mapanything_rgb_convert.name]
    tiler.downstream = ["osd"]

    if not sinks:
        # Default sinks reserved for Flow retrievers (mosaic/BEV branches).
        sinks = [{"name": "mosaic_sink", "type": "fakesink", "sync": False}]

    sink_names: List[str] = []
    for idx, sink_cfg in enumerate(sinks):
        sink_props = dict(sink_cfg)
        sink_name = sink_props.pop("name", f"sink_{idx}")
        if str(sink_name) == "mosaic_sink":
            # Keep mosaic sink as a semantic placeholder; encoded-AU SHM to
            # WebRTC is the canonical mosaic output path.
            continue
        element = sink_props.pop("type", "fakesink")
        sink_names.append(sink_name)
        component = Component(
            name=sink_name,
            element=element,
            config=sink_props,
            downstream=[],
        )
        pipeline.components[component.name] = component
        sink_tee.downstream.append(component.name)
        _safe_add(ds_pipeline, component, pipeline.errors)
        _apply_component_config(ds_pipeline, component, pipeline.errors)

    # Mosaic encode branch: raw surfaces from sink_tee → NVENC → H.264 AUs.
    # WebRTC consumes AUs through shmsink (no UDP loopback, no RTSP hop).
    # Optional RTSP remains available via nvrtspoutsinkbin for tooling only.
    mosaic_branch: Dict[str, Component] = {}
    mosaic_vconv_nvbuf_memory_type = zero_copy_nvbuf_memory_type
    if mosaic_encode_enabled:
        # Pre-encode leaky queue: protects analytics when WebRTC/RTSP consumers
        # stall. Drops raw frames only (never mid-GOP AUs).
        mosaic_queue = Component(
            name="mosaic_encode_queue",
            element="queue",
            config={
                "leaky": 2,
                "max-size-buffers": 4,
                "max-size-bytes": 0,
                "max-size-time": 0,
            },
            downstream=[],
        )
        mosaic_vconv = Component(
            name="mosaic_encode_vconv",
            element="nvvideoconvert",
            config={
                "gpu-id": zero_copy_gpu_id,
                "nvbuf-memory-type": mosaic_vconv_nvbuf_memory_type,
            },
            downstream=[],
        )
        mosaic_caps = Component(
            name="mosaic_encoder_caps",
            element="capsfilter",
            config={"caps": "video/x-raw(memory:NVMM),format=NV12"},
            downstream=[],
        )
        mosaic_force_idr = Component(
            name="mosaic_force_idr",
            element="noesisforceidr",
            config={"stream-id": "mosaic"},
            downstream=[],
        )
        h264_profile_value = int(h264_profile)
        h264_encoder_profiles = {0: 0, 1: 2, 2: 4}
        if h264_profile_value not in h264_encoder_profiles:
            raise ValueError(
                "mosaic_output.h264_profile must be 0 (baseline), 1 (main), or 2 (high)"
            )
        mosaic_encoder = Component(
            name="mosaic_h264_encoder",
            element="nvv4l2h264enc",
            config={
                "gpu-id": zero_copy_gpu_id,
                "bitrate": video_bitrate_kbps * 1000,
                "control-rate": 1,
                "preset-id": int(h264_preset_id),
                "tuning-info-id": 2,
                "iframeinterval": max(1, int(h264_iframeinterval)),
                "idrinterval": max(1, int(h264_idrinterval)),
                "profile": h264_encoder_profiles[h264_profile_value],
                "insert-sps-pps": True,
            },
            downstream=[],
        )
        mosaic_parse = Component(
            name="mosaic_h264_parse",
            element="h264parse",
            config={"config-interval": -1},
            downstream=[],
        )
        mosaic_h264_caps = Component(
            name="mosaic_h264_au_caps",
            element="capsfilter",
            config={
                "caps": "video/x-h264,stream-format=byte-stream,alignment=au",
            },
            downstream=[],
        )

        mosaic_branch = {
            "queue": mosaic_queue,
            "vconv": mosaic_vconv,
            "caps": mosaic_caps,
            "force_idr": mosaic_force_idr,
            "encoder": mosaic_encoder,
            "parse": mosaic_parse,
            "h264_caps": mosaic_h264_caps,
        }

        # Clean stale SHM path before bind, but fail if another runtime owns it.
        from noesis.mosaic_h264_bridge import ensure_parent_dir

        ensure_parent_dir(mosaic_h264_shm_socket)

        need_tee = bool(mosaic_webrtc_enabled and rtsp_enabled)
        if need_tee:
            mosaic_tee = Component(
                name="mosaic_h264_tee",
                element="tee",
                config={"allow-not-linked": True},
                downstream=[],
            )
            mosaic_branch["tee"] = mosaic_tee

        if mosaic_webrtc_enabled:
            # Non-leaky post-encode queue: if SHM consumers stall, backpressure
            # reaches the pre-encode leaky queue (raw frame drops only).
            webrtc_au_queue = Component(
                name="mosaic_webrtc_au_queue",
                element="queue",
                config={
                    "leaky": 0,
                    "max-size-buffers": 2,
                    "max-size-bytes": 0,
                    "max-size-time": 0,
                },
                downstream=[],
            )
            webrtc_shmsink = Component(
                name="mosaic_h264_shmsink",
                element="shmsink",
                config={
                    "socket-path": mosaic_h264_shm_socket,
                    "shm-size": int(mosaic_h264_shm_size_bytes),
                    "wait-for-connection": False,
                    "sync": False,
                    "async": False,
                },
                downstream=[],
            )
            mosaic_branch["webrtc_queue"] = webrtc_au_queue
            mosaic_branch["shmsink"] = webrtc_shmsink

        if rtsp_enabled:
            rtsp_mount = f"/{rtsp_path}" if not rtsp_path.startswith("/") else rtsp_path
            rtsp_out_queue = Component(
                name="mosaic_rtsp_out_queue",
                element="queue",
                config={
                    "leaky": 0,
                    "max-size-buffers": 4,
                    "max-size-bytes": 0,
                    "max-size-time": 0,
                },
                downstream=[],
            )
            rtsp_out = Component(
                name="rtsp_out",
                element="nvrtspoutsinkbin",
                config={
                    "rtsp-port": rtsp_port,
                    "rtsp-mount-point": rtsp_mount,
                    "bypass-codecs": True,
                    "sync": False,
                },
                downstream=[],
            )
            mosaic_branch["rtsp_queue"] = rtsp_out_queue
            mosaic_branch["rtsp_out"] = rtsp_out
        for comp in mosaic_branch.values():
            pipeline.components[comp.name] = comp
            _safe_add(ds_pipeline, comp, pipeline.errors)
            _apply_component_config(ds_pipeline, comp, pipeline.errors)

        sink_tee.downstream.append(mosaic_queue.name)
        mosaic_queue.downstream = [mosaic_vconv.name]
        mosaic_vconv.downstream = [mosaic_caps.name]
        mosaic_caps.downstream = [mosaic_force_idr.name]
        mosaic_force_idr.downstream = [mosaic_encoder.name]
        mosaic_encoder.downstream = [mosaic_parse.name]
        mosaic_parse.downstream = [mosaic_h264_caps.name]

        if need_tee:
            mosaic_h264_caps.downstream = [mosaic_branch["tee"].name]
            tee_targets: List[str] = []
            if mosaic_webrtc_enabled:
                mosaic_branch["tee"].downstream.append(mosaic_branch["webrtc_queue"].name)
                mosaic_branch["webrtc_queue"].downstream = [mosaic_branch["shmsink"].name]
                tee_targets.append(mosaic_branch["webrtc_queue"].name)
            if rtsp_enabled:
                mosaic_branch["tee"].downstream.append(mosaic_branch["rtsp_queue"].name)
                mosaic_branch["rtsp_queue"].downstream = [mosaic_branch["rtsp_out"].name]
                tee_targets.append(mosaic_branch["rtsp_queue"].name)
            mosaic_branch["tee"].downstream = tee_targets
        elif mosaic_webrtc_enabled:
            mosaic_h264_caps.downstream = [mosaic_branch["webrtc_queue"].name]
            mosaic_branch["webrtc_queue"].downstream = [mosaic_branch["shmsink"].name]
        else:
            mosaic_h264_caps.downstream = [mosaic_branch["rtsp_queue"].name]
            mosaic_branch["rtsp_queue"].downstream = [mosaic_branch["rtsp_out"].name]

        pipeline.mosaic_h264_shm_socket = mosaic_h264_shm_socket  # type: ignore[attr-defined]
        pipeline.mosaic_encode_enabled = True  # type: ignore[attr-defined]
        pipeline.mosaic_webrtc_shm_enabled = bool(mosaic_webrtc_enabled)  # type: ignore[attr-defined]

    source_decode_memtypes: List[Optional[int]] = []
    for source_cfg in sources:
        if not isinstance(source_cfg, dict):
            source_decode_memtypes.append(None)
            continue
        decode_mem = source_cfg.get("cudadec-memtype", source_cfg.get("cudadec_memtype"))
        try:
            source_decode_memtypes.append(int(decode_mem) if decode_mem is not None else None)
        except Exception:
            source_decode_memtypes.append(None)
    logger.info(
        json.dumps(
            {
                "event": "ds8_zero_copy_memory_config",
                "streammux_element": streammux.element,
                "per_source_pipeline": per_source_pipeline,
                "source_count": len(sources),
                "streammux_gpu_id": zero_copy_gpu_id,
                "streammux_nvbuf_memory_type": zero_copy_nvbuf_memory_type,
                "source_cudadec_memtype": source_decode_memtypes,
                "tiler_gpu_id": int(tiler_cfg.get("gpu-id", zero_copy_gpu_id) or zero_copy_gpu_id),
                "osd_process_mode": osd_process_mode,
                "rtsp_enabled": rtsp_enabled,
                "mosaic_vconv_nvbuf_memory_type": (
                    mosaic_vconv_nvbuf_memory_type if mosaic_encode_enabled else None
                ),
            },
            separators=(",", ":"),
        )
    )

    # No per-camera frame branches in tiled mode

    # Link graph now that all nodes exist.
    def _link(src: str, dst: str) -> None:
        if src not in pipeline.components or dst not in pipeline.components:
            return
        _safe_link(ds_pipeline, pipeline.errors, src, dst)

    if source_nodes:
        for source_name, source_idx in source_nodes:
            if source_name not in pipeline.components or "streammux" not in pipeline.components:
                continue
            _safe_link_with_hints(
                ds_pipeline,
                pipeline.errors,
                source_name,
                "streammux",
                "",
                "sink_%u",
            )

    _link("streammux", shutdown_eos.name)
    if preprocess_component is not None:
        _link(shutdown_eos.name, preprocess_component.name)
        _link(preprocess_component.name, primary.name)
    else:
        _link(shutdown_eos.name, primary.name)

    _link(primary.name, tee_component.name)
    _link(tee_component.name, primary_branch)
    if exclude_component is not None:
        _link(exclude_component.name, tracker.name)
    if mapanything is not None:
        mapanything_queue = pipeline.components.get("mapanything_queue")
        mapanything_valve = pipeline.components.get("mapanything_valve")
        if (
            mapanything_queue is None
            or mapanything_valve is None
            or mapanything_rgb_convert is None
            or mapanything_rgb_caps is None
            or mapanything_sink is None
        ):
            pipeline.errors.append("mapanything_exact_rgb_chain_missing")
        else:
            _link(tee_component.name, mapanything_queue.name)
            _link(mapanything_queue.name, mapanything_valve.name)
            _link(mapanything_valve.name, mapanything.name)
            _link(mapanything.name, mapanything_rgb_convert.name)
            _link(mapanything_rgb_convert.name, mapanything_rgb_caps.name)
            _link(mapanything_rgb_caps.name, mapanything_sink.name)
    if depth_tracking is not None:
        if depth_tracking_queue is not None:
            _link(tee_component.name, depth_tracking_queue.name)
            _link(depth_tracking_queue.name, depth_tracking.name)
        else:
            _link(tee_component.name, depth_tracking.name)
        if depth_tracking_sink is not None:
            _link(depth_tracking.name, depth_tracking_sink.name)

    if analytics is not None:
        _link(tracker.name, analytics.name)
        chain_start = analytics
    else:
        chain_start = tracker

    if reid is not None:
        _link(chain_start.name, reid.name)
        chain_start = reid
    if pose is not None:
        _link(chain_start.name, pose.name)
        chain_start = pose
    _link(chain_start.name, world_observation_stage.name)
    _link(world_observation_stage.name, tracking_telemetry_stage.name)
    _link(tracking_telemetry_stage.name, tiler.name)

    _link(tiler.name, osd.name)
    _link(osd.name, sink_tee.name)

    # Link mosaic encode branch: pre-encode leaky queue → NVENC → AU caps →
    # SHM (WebRTC) and/or optional RTSP bypass sink.
    if mosaic_branch:
        _link(sink_tee.name, mosaic_branch["queue"].name)
        _link(mosaic_branch["queue"].name, mosaic_branch["vconv"].name)
        _link(mosaic_branch["vconv"].name, mosaic_branch["caps"].name)
        _link(mosaic_branch["caps"].name, mosaic_branch["force_idr"].name)
        _link(mosaic_branch["force_idr"].name, mosaic_branch["encoder"].name)
        _link(mosaic_branch["encoder"].name, mosaic_branch["parse"].name)
        _link(mosaic_branch["parse"].name, mosaic_branch["h264_caps"].name)
        if "tee" in mosaic_branch:
            _link(mosaic_branch["h264_caps"].name, mosaic_branch["tee"].name)
            if "webrtc_queue" in mosaic_branch:
                _link(mosaic_branch["tee"].name, mosaic_branch["webrtc_queue"].name)
                _link(mosaic_branch["webrtc_queue"].name, mosaic_branch["shmsink"].name)
            if "rtsp_queue" in mosaic_branch:
                _link(mosaic_branch["tee"].name, mosaic_branch["rtsp_queue"].name)
                _link(mosaic_branch["rtsp_queue"].name, mosaic_branch["rtsp_out"].name)
        elif "webrtc_queue" in mosaic_branch:
            _link(mosaic_branch["h264_caps"].name, mosaic_branch["webrtc_queue"].name)
            _link(mosaic_branch["webrtc_queue"].name, mosaic_branch["shmsink"].name)
        elif "rtsp_queue" in mosaic_branch:
            _link(mosaic_branch["h264_caps"].name, mosaic_branch["rtsp_queue"].name)
            _link(mosaic_branch["rtsp_queue"].name, mosaic_branch["rtsp_out"].name)

    for sink_name in sink_names:
        _link(sink_tee.name, sink_name)

    # Every gated branch downstream of orderly_eos_control must forward sticky
    # events.  A mode-0 valve can accept and then discard EOS, leaving the
    # Service Maker wait loop alive during process teardown.
    for component in pipeline.components.values():
        if component.element != "valve":
            continue
        try:
            drop_mode = int(component.config.get("drop-mode", 0))
        except Exception:
            drop_mode = 0
        if drop_mode not in (1, 2):
            pipeline.errors.append(
                f"shutdown_eos_valve:{component.name}:drop-mode={drop_mode}"
            )

    _attach_depth_gate(pipeline)
    _attach_fps_probes(pipeline)
    _attach_latency_probe(pipeline)
    _attach_source_progress_probes(pipeline)
    _PIPELINE_SINGLETON = pipeline
    # Depth is disabled by default; the MapAnything gate is closed shortly after activation
    # (see activate()) to avoid a "stuck-at-PAUSED" preroll issue when the valve is closed
    # before negotiation completes.
    return pipeline


def get_pipeline() -> DS8Pipeline:
    if _PIPELINE_SINGLETON is None:
        raise RuntimeError("Pipeline not built. Call build_pipeline(yaml_path) first.")
    return _PIPELINE_SINGLETON


def prepare(on_message=None) -> bool:
    pipeline = get_pipeline()
    if pipeline.errors:
        logger.error("Cannot prepare DS8 pipeline; errors present: %s", pipeline.errors)
        return False
    if pipeline.ds_pipeline is None:
        pipeline.errors.append("pyservicemaker:unavailable")
        logger.error("Cannot prepare DS8 pipeline; pyservicemaker unavailable")
        return False
    try:
        if on_message is not None:
            result = pipeline.ds_pipeline.prepare(on_message)  # type: ignore[union-attr]
        else:
            result = pipeline.ds_pipeline.prepare()  # type: ignore[union-attr]
        # pyservicemaker.Pipeline.prepare() returns 1 on success, -1 on failure.
        if isinstance(result, int) and result != 1:
            pipeline.errors.append(f"prepare:return:{result}")
            logger.error("DS8 pipeline prepare returned non-success code: %s", result)
            pipeline.prepared = False
            return False
        pipeline.prepared = True
        return True
    except Exception as exc:  # pragma: no cover - depends on runtime implementation
        pipeline.errors.append(f"prepare:{redact_runtime_secrets(exc)}")
        logger.error("DS8 pipeline prepare failed: %s", exc)
        pipeline.prepared = False
        return False


def activate() -> bool:
    pipeline = get_pipeline()
    if not pipeline.prepared:
        if not prepare():
            return False
    # Attempt to start the underlying DS8 Service Maker pipeline if available.
    ds = getattr(pipeline, "ds_pipeline", None)
    if ds is None:
        logger.error("DS8 bindings not available; activation aborted")
        pipeline.activated = False
        return False
    try:
        # Prime MapAnything branch negotiation: leave the valve open briefly so SGIE can preroll,
        # then close it if depth remains disabled (default). Without this, closing the valve
        # before the first preroll buffer can stall the whole pipeline.
        if pipeline._prime_timer is not None:
            try:
                pipeline._prime_timer.cancel()
            except Exception:
                pass
            pipeline._prime_timer = None
        if pipeline.valve_name:
            pipeline._set_valve_drop(False)
        ds.activate()  # type: ignore[union-attr]
        pipeline.activated = True
        if pipeline.valve_name and not pipeline.depth_enabled:
            prime_env = os.environ.get("NOESIS_MAPANYTHING_GATE_PRIME_SECONDS", "1.0")
            try:
                prime_seconds = float(str(prime_env).strip())
            except Exception:
                prime_seconds = 1.0
            prime_seconds = max(0.1, min(10.0, prime_seconds))

            def _close_gate() -> None:
                with pipeline._depth_lock:
                    if pipeline.depth_enabled:
                        return
                pipeline._set_valve_drop(True)
                logger.info("MapAnything gate primed; closed valve after %.2fs", prime_seconds)

            timer = threading.Timer(prime_seconds, _close_gate)
            timer.daemon = True
            timer.start()
            pipeline._prime_timer = timer
        return True
    except Exception as exc:  # pragma: no cover - depends on runtime implementation
        pipeline.errors.append(f"activate:{redact_runtime_secrets(exc)}")
        logger.error("DS8 pipeline activation failed: %s", exc)
        pipeline.activated = False
        return False


def enable_depth(seconds: int = 20) -> Dict[str, Any]:
    """Enable the MapAnything depth branch for a limited time window."""
    pipeline = get_pipeline()
    if not pipeline.depth_gate_supported:
        warning = "Depth gating not configured; depth branch will not be dropped upstream (logical gating only)."
        if warning not in pipeline.errors:
            pipeline.errors.append(warning)
        logger.warning(warning)
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
