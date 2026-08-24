from typing import Dict, Optional, Any, Callable, List, NamedTuple, Sequence, Set
import asyncio
import concurrent.futures
import hmac
from http import HTTPStatus
from websockets.legacy.server import serve
import websockets
import inspect
import json
import logging
import math
import os
from collections import deque
import threading
import time
import uuid

from models import convert_numpy_types
from noesis.server.internal_auth import InternalAuthConfig, validate_internal_auth_listener


def _depth_rpc_timeout_seconds() -> float:
    raw = str(
        os.environ.get("NOESIS_DEPTH_RPC_TIMEOUT_SECONDS", "110.0")
    ).strip()
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "NOESIS_DEPTH_RPC_TIMEOUT_SECONDS must be a finite number"
        ) from exc
    if not math.isfinite(value) or not 1.0 <= value <= 120.0:
        raise ValueError(
            "NOESIS_DEPTH_RPC_TIMEOUT_SECONDS must be in [1, 120]"
        )
    return value


def _serialize_boundary_json(
    payload: Dict[str, Any],
    submitted_ns: Optional[int] = None,
) -> tuple[str, int, float, float, float]:
    """Serialize one public payload and return all local boundary timings.

    ``submitted_ns`` is supplied for executor-backed serialization so queue and
    worker-dispatch latency remains observable and part of the authoritative
    end-to-end boundary budget.
    """

    worker_start_ns = time.perf_counter_ns()
    dispatch_wait_ms = (
        max(0.0, (worker_start_ns - submitted_ns) / 1_000_000.0)
        if submitted_ns is not None
        else 0.0
    )
    encode_start_ns = time.perf_counter_ns()
    conversion_ns = 0

    def _numpy_default(value: Any) -> Any:
        nonlocal conversion_ns
        started_ns = time.perf_counter_ns()
        try:
            converted = convert_numpy_types(value)
        except Exception as exc:
            raise _BoundaryConversionError(type(exc).__name__) from exc
        finally:
            conversion_ns += time.perf_counter_ns() - started_ns
        if converted is value:
            raise TypeError(
                f"Object of type {type(value).__name__} is not JSON serializable"
            )
        return converted

    # stdlib's encoder walks the tree once and invokes the default only for
    # NumPy leaves. This preserves tuple/list wire behavior without a recursive
    # pre-copy. ensure_ascii=True means character length equals UTF-8 byte length.
    message_text = json.dumps(
        payload,
        separators=(",", ":"),
        default=_numpy_default,
        ensure_ascii=True,
        allow_nan=False,
    )
    wall_encode_ns = time.perf_counter_ns() - encode_start_ns
    payload_bytes = len(message_text)
    convert_ms = conversion_ns / 1_000_000.0
    encode_ms = max(0, wall_encode_ns - conversion_ns) / 1_000_000.0
    return message_text, payload_bytes, dispatch_wait_ms, convert_ms, encode_ms


def _prewarm_boundary_serializer() -> None:
    """Start the dedicated serializer worker before it enters a gated path."""


class _BoundaryConversionError(TypeError):
    """Internal marker separating NumPy conversion failures from JSON errors."""

    def __init__(self, original_error_type: str) -> None:
        self.original_error_type = str(original_error_type)
        super().__init__(self.original_error_type)


class _BoundaryDispatchError(RuntimeError):
    """Internal marker for executor submission failures already recorded."""


class BoundaryResponseModelContractError(RuntimeError):
    """Raised when a JSON boundary receives no response-assembly timing."""


class BoundaryResponseModelTiming(NamedTuple):
    duration_ms: float


class BoundaryTimedPayload(dict):
    """Wire-compatible dict carrying private response-assembly timing."""

    boundary_response_model_timing: BoundaryResponseModelTiming

    def __init__(
        self,
        payload: Dict[str, Any],
        timing: BoundaryResponseModelTiming,
    ) -> None:
        super().__init__(payload)
        self.boundary_response_model_timing = timing


class FrozenOutboundJSON(NamedTuple):
    """Immutable, pre-encoded JSON admitted from a producer thread."""

    encoded: str
    payload_bytes: int
    message_type: str
    convert_ms: float
    encode_ms: float
    route: str
    coalesce_key: Optional[str]
    telemetry_payload: Optional[Dict[str, Any]]
    owner_token: object


class ProviderQuiescenceReceipt(NamedTuple):
    admission_closed: bool
    admitted_calls: int
    completed_calls: int
    active_calls: int
    pending_futures: int
    executor_joined: bool

    @property
    def quiesced(self) -> bool:
        return bool(
            self.admission_closed
            and self.active_calls == 0
            and self.pending_futures == 0
            and self.executor_joined
        )


class ProviderAdmissionClosed(RuntimeError):
    """Raised when shutdown has closed blocking WebSocket RPC admission."""


class ProviderCapacityExceeded(RuntimeError):
    """Raised before submit when the owned provider queue is saturated."""


class ProviderQuiescenceTimeout(RuntimeError):
    """Raised when an admitted blocking provider outlives its shutdown bound."""

    def __init__(self, receipt: ProviderQuiescenceReceipt) -> None:
        self.receipt = receipt
        super().__init__(
            "blocking WebSocket providers did not quiesce "
            f"(active_calls={receipt.active_calls}, "
            f"admitted_calls={receipt.admitted_calls}, "
            f"completed_calls={receipt.completed_calls}, "
            f"pending_futures={receipt.pending_futures})"
        )


class OutboundQuiescenceReceipt(NamedTuple):
    admission_closed: bool
    admitted_submissions: int
    completed_submissions: int
    failed_submissions: int
    aborted_submissions: int
    pending_futures: int
    admitted_bytes: int
    completed_bytes: int
    aborted_bytes: int
    inflight_bytes: int
    peak_inflight_bytes: int
    max_inflight_bytes: int

    @property
    def quiesced(self) -> bool:
        return bool(
            self.admission_closed
            and self.pending_futures == 0
            and self.inflight_bytes == 0
        )


class OutboundQuiescenceTimeout(RuntimeError):
    """Raised when an admitted cross-thread send outlives its stop barrier."""

    def __init__(self, receipt: OutboundQuiescenceReceipt) -> None:
        self.receipt = receipt
        super().__init__(
            "cross-thread WebSocket submissions did not quiesce "
            f"(admitted_submissions={receipt.admitted_submissions}, "
            f"completed_submissions={receipt.completed_submissions}, "
            f"pending_futures={receipt.pending_futures}, "
            f"inflight_bytes={receipt.inflight_bytes})"
        )


class OutboundAdmissionReceipt(NamedTuple):
    """Proof that one bounded submission entered the owned event-loop queue."""

    submission_id: int
    admitted_at_ns: int
    message_count: int
    payload_bytes: int
    inflight_bytes: int
    max_inflight_bytes: int


class _OutboundGateDecision(NamedTuple):
    release: bool
    resolved_at_ns: int


_OUTBOUND_ABORTED = object()


class OutboundGateResolutionError(RuntimeError):
    """Raised when an admitted gated submission can no longer be resolved."""


class GatedOutboundAdmission:
    """One-shot authority gate for an already bounded outbound submission.

    Admission freezes and reserves the exact bytes, but the event-loop task
    cannot begin client delivery until ``release`` succeeds.  ``abort`` lets
    the task retire without delivering any member of the batch.  The
    ``commit_then_release`` helper makes the intended authority ordering
    explicit and resolves the gate on both callback outcomes.
    """

    def __init__(
        self,
        receipt: OutboundAdmissionReceipt,
        decision: concurrent.futures.Future[_OutboundGateDecision],
        submission: concurrent.futures.Future[Any],
    ) -> None:
        self.receipt = receipt
        self._decision = decision
        self._submission = submission
        self._lock = threading.Lock()
        self._state = "pending"

    @property
    def resolved(self) -> bool:
        with self._lock:
            return self._state in {"released", "aborted", "resolution_failed"}

    @property
    def state(self) -> str:
        with self._lock:
            return str(self._state)

    def _claim(self, state: str) -> None:
        with self._lock:
            if self._state != "pending":
                raise OutboundGateResolutionError(
                    f"outbound authority gate is already {self._state}"
                )
            self._state = state

    def _resolve_claimed(
        self,
        *,
        claimed_state: str,
        release: bool,
    ) -> OutboundAdmissionReceipt:
        try:
            if self._submission.done() and not self._decision.done():
                raise OutboundGateResolutionError(
                    "outbound submission ended before authority resolution"
                )
            self._decision.set_result(
                _OutboundGateDecision(
                    release=bool(release),
                    resolved_at_ns=time.perf_counter_ns(),
                )
            )
        except BaseException as exc:
            with self._lock:
                self._state = "resolution_failed"
            raise OutboundGateResolutionError(
                "outbound authority gate is no longer resolvable"
            ) from exc
        with self._lock:
            if self._state != claimed_state:
                self._state = "resolution_failed"
                raise OutboundGateResolutionError(
                    "outbound authority gate state changed during resolution"
                )
            self._state = "released" if release else "aborted"
        return self.receipt

    def commit_then_release(
        self,
        commit: Callable[[], Any],
    ) -> Any:
        """Run one synchronous authority commit, then release or abort."""

        if not callable(commit):
            raise TypeError("outbound authority commit must be callable")
        self._claim("committing")
        try:
            result = commit()
        except BaseException as commit_error:
            try:
                self._resolve_claimed(
                    claimed_state="committing",
                    release=False,
                )
            except BaseException:
                raise OutboundGateResolutionError(
                    "authority commit failed and outbound abort could not be acknowledged"
                ) from commit_error
            raise
        self._resolve_claimed(
            claimed_state="committing",
            release=True,
        )
        return result


class OutboundAdmissionError(RuntimeError):
    """Base class for failures before an outbound submission is admitted."""


class OutboundAdmissionClosed(OutboundAdmissionError):
    """Raised when lifecycle state cannot accept another outbound submission."""


class OutboundCapacityExceeded(OutboundAdmissionError):
    """Raised instead of growing the owned outbound queue without a bound."""


class CanonicalOutboundRouteRequired(OutboundAdmissionError):
    """Raised when a canonical type attempts to bypass its dedicated route."""


class FrozenOutboundOwnershipError(OutboundAdmissionError):
    """Raised when pre-encoded JSON was not frozen by this server instance."""


class StatsQuiescenceReceipt(NamedTuple):
    admission_closed: bool
    admitted_snapshots: int
    completed_snapshots: int
    failed_snapshots: int
    pending_futures: int
    executor_joined: bool

    @property
    def quiesced(self) -> bool:
        return bool(
            self.admission_closed
            and self.pending_futures == 0
            and self.executor_joined
        )


class StatsAdmissionClosed(RuntimeError):
    """Raised when shutdown has closed stats source collection."""


class StatsQuiescenceTimeout(RuntimeError):
    def __init__(self, receipt: StatsQuiescenceReceipt) -> None:
        self.receipt = receipt
        super().__init__(
            "WebSocket stats collector did not quiesce "
            f"(pending_futures={receipt.pending_futures}, "
            f"admitted_snapshots={receipt.admitted_snapshots}, "
            f"completed_snapshots={receipt.completed_snapshots})"
        )


class WebSocketStartupReceipt(NamedTuple):
    startup_signaled: bool
    server_bound: bool
    start_task_done: bool
    thread_stopped: bool
    event_loop_closed: bool

    @property
    def quiesced(self) -> bool:
        return bool(self.thread_stopped and self.event_loop_closed)


class WebSocketStartupError(RuntimeError):
    """Raised only after bounded WebSocket startup cleanup is attempted."""

    def __init__(
        self,
        message: str,
        receipt: WebSocketStartupReceipt,
    ) -> None:
        self.receipt = receipt
        super().__init__(
            f"{message} (startup_signaled={receipt.startup_signaled}, "
            f"server_bound={receipt.server_bound}, "
            f"start_task_done={receipt.start_task_done}, "
            f"thread_stopped={receipt.thread_stopped}, "
            f"event_loop_closed={receipt.event_loop_closed})"
        )


class WebSocketServer:
    """Manages WebSocket server for broadcasting data to clients"""

    HEALTH_PATH = "/healthz"
    AUTHORITY_GATED_MESSAGE_TYPES = frozenset(
        {"tracking", "world_snapshot", "world_event"}
    )
    # BEV frames and their fail-closed status/error records share one ordered
    # sender route.  A status is not a substitute for a frame, but it must
    # occupy the same bounded outbound path so a late error cannot overtake a
    # newer admitted frame without carrying an older cohort that the client
    # can reject.
    DEDICATED_BEV_MESSAGE_TYPES = frozenset({"bev-frame", "bev-status"})
    CANONICAL_MESSAGE_TYPES = (
        AUTHORITY_GATED_MESSAGE_TYPES | DEDICATED_BEV_MESSAGE_TYPES
    )
    DEFAULT_MAX_TELEMETRY_CLIENTS = 8
    HARD_MAX_TELEMETRY_CLIENTS = 16
    TELEMETRY_CAPACITY_CLOSE_CODE = 1013
    TELEMETRY_CAPACITY_CLOSE_REASON = "telemetry_capacity_reached"
    DEPTH_REQUEST_ID_MAX_BYTES = 128
    DEPTH_REQUEST_ID_POLICY_CLOSE_CODE = 1008
    DEPTH_REQUEST_ID_POLICY_CLOSE_REASON = "invalid_depth_request_id"
    HEALTH_PAYLOAD = '{"type":"health","contract":"noesis.ws.health","contract_version":1}'
    CLOSE_HANDSHAKE_TIMEOUT_S = 1.0
    # websockets' legacy server can consume four times close_timeout while it
    # completes the closing handshake.  The listener and explicit client close
    # paths begin together and share one authoritative six-second deadline.
    LEGACY_CLOSE_TIMEOUT_MULTIPLIER = 4.0
    SERVER_CLOSE_GRACE_S = (
        CLOSE_HANDSHAKE_TIMEOUT_S * LEGACY_CLOSE_TIMEOUT_MULTIPLIER + 0.25
    )
    CLIENT_CLOSE_GRACE_S = SERVER_CLOSE_GRACE_S
    CLIENT_ABORT_DRAIN_S = 1.0
    LISTENER_SHUTDOWN_TIMEOUT_S = 6.0
    OUTBOUND_SUBMISSION_DRAIN_TIMEOUT_S = 2.0
    OUTBOUND_MAX_INFLIGHT = 256
    OUTBOUND_MAX_INFLIGHT_BYTES = 256 * 1024 * 1024
    OUTBOUND_BATCH_MAX_MESSAGES = 256
    OUTBOUND_MESSAGE_MAX_BYTES = 16 * 1024 * 1024
    OUTBOUND_BATCH_MAX_BYTES = 32 * 1024 * 1024
    STATS_COLLECTOR_DRAIN_TIMEOUT_S = 2.0
    WEBRTC_GATEWAY_DRAIN_TIMEOUT_S = 9.0
    STARTUP_TIMEOUT_S = 5.0
    PROVIDER_WORKERS = 3
    PROVIDER_MAX_INFLIGHT = 6
    ASYNC_SHUTDOWN_TIMEOUT_S = (
        OUTBOUND_SUBMISSION_DRAIN_TIMEOUT_S
        + STATS_COLLECTOR_DRAIN_TIMEOUT_S
        + LISTENER_SHUTDOWN_TIMEOUT_S
        + 0.5
    )
    EVENT_LOOP_THREAD_JOIN_TIMEOUT_S = 2.0
    RUNTIME_SHUTDOWN_TIMEOUT_S = (
        ASYNC_SHUTDOWN_TIMEOUT_S + EVENT_LOOP_THREAD_JOIN_TIMEOUT_S
    )
    BOUNDARY_WINDOWS_S = (10.0, 60.0)
    BOUNDARY_MAX_WINDOW_S = max(BOUNDARY_WINDOWS_S)
    BOUNDARY_MAX_DETAIL_BUCKETS = 512
    BOUNDARY_AGGREGATE_SAMPLE_LIMIT = 8192
    BOUNDARY_DETAIL_SAMPLE_LIMIT = 1024
    BOUNDARY_ROUTE_OVERFLOW_KEY = "ws|__overflow__|__overflow__|__overflow__"
    BOUNDARY_STAGE_OVERFLOW_KEY = (
        "ws|__overflow__|__overflow__|__overflow__|__overflow__"
    )
    BOUNDARY_ERROR_OVERFLOW_KEY = (
        "ws|__overflow__|__overflow__|__overflow__|__overflow__|__overflow__"
    )
    
    def __init__(
        self, 
        host: str = "127.0.0.1",
        port: int = 6008,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
        stats_callback: Optional[Callable[[], Dict[str, Any]]] = None,
        toggle_callback: Optional[Callable[[str, bool], None]] = None,
        initial_trail_state: bool = True,
        internal_auth_config: Optional[InternalAuthConfig] = None,
        health_payload_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        """Initialize the WebSocket server
        
        Args:
            host: Server host address
            port: Server port
            event_loop: Optional asyncio event loop
            stats_callback: Optional callback to get statistics
            toggle_callback: Optional callback to handle toggle updates
            initial_trail_state: The initial state of the trail visualization
        """
        self.host = host
        self.port = port
        self.event_loop = event_loop
        self.stats_callback = stats_callback
        self.toggle_callback = toggle_callback
        self.initial_trail_state = initial_trail_state
        self.health_payload_getter = health_payload_getter
        self.boundary_failure_callback: Optional[Callable[[BaseException], None]] = None
        self.lifecycle_failure_callback: Optional[Callable[[BaseException], None]] = None
        self._shutdown_quiesced = False
        auth_config = internal_auth_config or InternalAuthConfig.from_env()
        validate_internal_auth_listener(auth_config.mode, host)
        self._internal_auth_mode = auth_config.mode
        self._internal_auth_token = auth_config.token
        self.connected_clients = set()
        self.server = None
        self.server_task = None
        self.running = True
        self.logger = logging.getLogger("WebSocketServer")
        self._stats_task = None # Added reference for the periodic stats task
        self._last_stats_info_log: float = 0.0
        # Binary frame coalescer state: keep only latest per camera.
        # Currently unused for BEV (JPEG binary retired — meta-only mode). Retained for future binary depth or similar.
        self._latest_binary_by_cam: Dict[str, bytes] = {}
        self._binary_flush_task: Optional[asyncio.Task] = None
        self._binary_sending: bool = False
        # Lightweight telemetry for Menon calibration/coordinate RPCs
        self._telemetry: Dict[str, Dict[str, Any]] = {"rx": {}, "tx": {}}
        self._telemetry_task: Optional[asyncio.Task] = None
        # Optional BEV control callbacks
        self.bev_config_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self.bev_overlay_callback: Optional[Callable[[str, bool], None]] = None
        # Optional getter for trail settings sync payloads
        self.trail_settings_getter: Optional[Callable[[], Dict[str, Any]]] = None
        # RPC guardrails
        self._depth_rpc_tracker: Dict[str, float] = {}
        self._floorplan_rpc_tracker: Dict[str, float] = {}
        self._depth_rate_limit_window = 0.5  # seconds per camera/client
        self._floorplan_rate_limit_window = 2.0
        self._depth_rpc_timeout = _depth_rpc_timeout_seconds()
        # A quality-first manual capture includes the full inference burst,
        # coherent fusion, dense backprojection, and grid encoding.  Dense
        # 1080p products can legitimately take longer than 30 seconds on the
        # production GPU, so keep the RPC alive long enough to publish the
        # completed exact-snapshot result instead of timing out while the
        # provider continues successfully in the background.
        self._floorplan_rpc_timeout = 90.0
        self._calibration_rpc_timeout = 30.0
        self._tracker_prune_window = 30.0
        # Optional calibration + RPC callbacks
        self.calibration_getter: Optional[Callable[[], Dict[str, Any]]] = None
        self.pixel_to_world_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
        self.set_extrinsics_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
        self.solve_pnp_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
        self.set_align_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
        self.ma_depth_provider: Optional[Callable[..., Optional[Dict[str, Any]]]] = None
        self.floorplan_provider: Optional[Callable[[Optional[list], float, float, float, bool], Optional[Dict[str, Any]]]] = None
        # Optional auto-calibration handler (cameraId -> result)
        self.auto_calibrate_handler: Optional[Callable[[Optional[str]], Dict[str, Any]]] = None
        # WebRTC gateway pool for the canonical H.264-SHM delivery path.
        self.webrtc_gateway: Optional[Any] = None
        self.webrtc_gateways: List[Any] = []
        try:
            self._webrtc_max_clients = max(1, int(os.environ.get("NOESIS_MOSAIC_WEBRTC_MAX_CLIENTS", "5")))
        except Exception:
            self._webrtc_max_clients = 5
        try:
            self._webrtc_initial_clients = max(0, int(os.environ.get("NOESIS_MOSAIC_WEBRTC_INITIAL_CLIENTS", "1")))
        except Exception:
            self._webrtc_initial_clients = 1
        self._webrtc_gateway_factory: Optional[Callable[[], Any]] = None
        self.webrtc_activity_callback: Optional[Callable[[bool], None]] = None
        # WebRTC signaling ownership: only the connection that most recently sent a
        # webrtc_offer should receive webrtc_answer / server ICE candidates.
        self._webrtc_owner: Optional[Any] = None
        self._webrtc_owner_ip: Optional[str] = None
        # Multi-gateway ownership maps (one websocket owner per gateway instance).
        self._webrtc_gateway_owner: Dict[Any, Any] = {}
        self._webrtc_gateway_owner_ip: Dict[Any, str] = {}
        self._webrtc_client_gateway: Dict[Any, Any] = {}
        self._webrtc_gateway_resetting: Set[Any] = set()
        self._webrtc_shutdown = False
        self._webrtc_lifecycle_condition = threading.Condition()
        self._webrtc_lifecycle_executor: Optional[
            concurrent.futures.ThreadPoolExecutor
        ] = None
        self._webrtc_lifecycle_futures: Set[
            concurrent.futures.Future[Any]
        ] = set()
        self._webrtc_retired_gateways: Set[Any] = set()
        self._webrtc_lifecycle_owned_gateways: Set[Any] = set()
        self._webrtc_lifecycle_failures: List[str] = []
        try:
            configured_client_limit = int(
                os.environ.get(
                    "NOESIS_WS_MAX_TELEMETRY_CLIENTS",
                    str(self.DEFAULT_MAX_TELEMETRY_CLIENTS),
                )
            )
        except (TypeError, ValueError):
            configured_client_limit = self.DEFAULT_MAX_TELEMETRY_CLIENTS
        self._max_telemetry_clients = max(
            1,
            min(
                int(self.HARD_MAX_TELEMETRY_CLIENTS),
                int(configured_client_limit),
            ),
        )
        self._telemetry_client_rejections = 0
        self._telemetry_client_peak = 0
        # Every cross-thread publisher/signaling submission acquires a lease
        # before run_coroutine_threadsafe. Stop closes admission atomically and
        # awaits all admitted futures on the owning event loop before listener
        # teardown, so no producer can enqueue work behind the stop barrier.
        self._outbound_condition = threading.Condition()
        self._canonical_outbound_token = object()
        self._frozen_outbound_token = object()
        self._outbound_admission_open = True
        self._outbound_futures: Set[concurrent.futures.Future[Any]] = set()
        self._outbound_bytes_by_future: Dict[
            concurrent.futures.Future[Any], int
        ] = {}
        self._outbound_admitted_submissions = 0
        self._outbound_completed_submissions = 0
        self._outbound_failed_submissions = 0
        self._outbound_aborted_submissions = 0
        self._outbound_admitted_bytes = 0
        self._outbound_completed_bytes = 0
        self._outbound_aborted_bytes = 0
        self._outbound_inflight_bytes = 0
        self._outbound_peak_inflight_bytes = 0
        self._outbound_last_failure_type: Optional[str] = None
        self._outbound_max_inflight = int(self.OUTBOUND_MAX_INFLIGHT)
        self._outbound_max_inflight_bytes = int(
            self.OUTBOUND_MAX_INFLIGHT_BYTES
        )
        self._outbound_shutdown_receipt: Optional[OutboundQuiescenceReceipt] = None
        self._stats_condition = threading.Condition()
        self._stats_admission_open = True
        self._stats_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._stats_futures: Set[concurrent.futures.Future[Any]] = set()
        self._stats_inflight: Optional[concurrent.futures.Future[Any]] = None
        self._stats_inflight_kind: Optional[str] = None
        self._stats_admitted_snapshots = 0
        self._stats_completed_snapshots = 0
        self._stats_failed_snapshots = 0
        self._stats_last_failure_type: Optional[str] = None
        self._stats_generation = 0
        self._stats_future_generation: Dict[
            concurrent.futures.Future[Any], int
        ] = {}
        self._stats_cache: Any = None
        self._stats_cache_at_s = 0.0
        self._stats_cache_ttl_s = 0.2
        self._stats_shutdown_receipt: Optional[StatsQuiescenceReceipt] = None
        # Boundary serialization metrics (JSON conversion at WS boundary).
        self._boundary_lock = threading.Lock()
        self._boundary_samples_ms = deque()
        self._boundary_count = 0
        self._boundary_total_ms = 0.0
        self._boundary_max_ms = 0.0
        self._boundary_last_ms = 0.0
        self._boundary_total_bytes = 0
        self._boundary_last_bytes = 0
        self._boundary_budget_ms = 3.0
        self._boundary_violations = 0
        self._boundary_route_metrics: Dict[str, Dict[str, Any]] = {}
        self._boundary_stage_metrics: Dict[str, Dict[str, Any]] = {}
        self._boundary_budget_path_samples: Set[str] = set()
        self._boundary_aggregate_truncation = self._new_boundary_truncation()
        self._boundary_error_counts: Dict[str, int] = {}
        self._boundary_errors_total = 0
        self._boundary_clock: Callable[[], float] = time.monotonic
        # Large depth/floorplan payloads must not compete with provider work on
        # asyncio's process-wide default executor.  One bounded worker preserves
        # serialization ordering and exposes any real queueing in the 3 ms gate.
        self._serializer_executor_lock = threading.Lock()
        self._serializer_executor: Optional[
            concurrent.futures.ThreadPoolExecutor
        ] = None
        # All blocking RPC callbacks share one owned, bounded executor and an
        # admission barrier.  Cancelling an asyncio waiter cannot cancel a
        # running thread, so native/store teardown must drain these leases.
        self._provider_condition = threading.Condition()
        self._provider_admission_open = True
        self._provider_executor: Optional[
            concurrent.futures.ThreadPoolExecutor
        ] = None
        self._provider_futures: Set[concurrent.futures.Future[Any]] = set()
        self._provider_admitted_calls = 0
        self._provider_completed_calls = 0
        self._provider_active_calls = 0
        self._latest_json_by_key: Dict[str, Any] = {}
        self._latest_json_enqueue_ms_by_key: Dict[str, float] = {}
        self._latest_json_response_model_ms_by_key: Dict[str, float] = {}
        self._json_flush_task: Optional[asyncio.Task] = None
        self._json_sending: bool = False
        self._json_last_sent: Dict[str, float] = {}
        self._json_coalesce_interval_by_type: Dict[str, float] = self._load_json_coalesce_intervals()

    def _load_json_coalesce_intervals(self) -> Dict[str, float]:
        # Canonical tracking, global-world, and BEV publishers own their rate
        # gates before sequence assignment.  Replacing those messages here
        # would create externally visible sequence gaps and split exact frame
        # cohorts, so the generic latest-only coalescer owns no canonical type.
        return {}

    def _new_boundary_bucket(self) -> Dict[str, Any]:
        return {
            "samples": deque(),
            "truncation": self._new_boundary_truncation(),
            "count": 0,
            "total_ms": 0.0,
            "max_ms": 0.0,
            "last_ms": 0.0,
            "total_bytes": 0,
            "last_bytes": 0,
        }

    def _new_boundary_truncation(self) -> Dict[str, Dict[str, Any]]:
        return {
            f"{int(window_s)}s": {
                "count": 0,
                "max_ms": None,
                "until_s": 0.0,
            }
            for window_s in self.BOUNDARY_WINDOWS_S
        }

    def _append_bounded_boundary_sample(
        self,
        samples: deque,
        sample: tuple[float, float, int],
        *,
        limit: int,
        truncation: Dict[str, Dict[str, Any]],
    ) -> None:
        now_s = float(sample[0])
        for evidence in truncation.values():
            if now_s > float(evidence.get("until_s", 0.0)):
                evidence.update(count=0, max_ms=None, until_s=0.0)
        if len(samples) >= int(limit):
            dropped = samples.popleft()
            for window_s in self.BOUNDARY_WINDOWS_S:
                key = f"{int(window_s)}s"
                evidence = truncation[key]
                evidence["count"] = int(evidence.get("count", 0)) + 1
                prior_max = evidence.get("max_ms")
                evidence["max_ms"] = max(
                    float(dropped[1]),
                    float(prior_max) if prior_max is not None else float(dropped[1]),
                )
                evidence["until_s"] = max(
                    float(evidence.get("until_s", 0.0)),
                    float(dropped[0]) + float(window_s),
                )
        samples.append(sample)

    def _update_boundary_bucket(
        self,
        bucket: Dict[str, Any],
        duration_ms: float,
        payload_bytes: int,
        observed_at_s: float,
    ) -> None:
        self._append_bounded_boundary_sample(
            bucket["samples"],
            (float(observed_at_s), float(duration_ms), int(payload_bytes)),
            limit=self.BOUNDARY_DETAIL_SAMPLE_LIMIT,
            truncation=bucket["truncation"],
        )
        bucket["count"] = int(bucket.get("count", 0)) + 1
        bucket["total_ms"] = float(bucket.get("total_ms", 0.0)) + float(duration_ms)
        bucket["max_ms"] = max(float(bucket.get("max_ms", 0.0)), float(duration_ms))
        bucket["last_ms"] = float(duration_ms)
        bucket["total_bytes"] = int(bucket.get("total_bytes", 0)) + int(payload_bytes)
        bucket["last_bytes"] = int(payload_bytes)

    def _prune_boundary_samples(self, samples: deque, now_s: float) -> None:
        cutoff_s = float(now_s) - float(self.BOUNDARY_MAX_WINDOW_S)
        while samples and float(samples[0][0]) < cutoff_s:
            samples.popleft()

    def _bounded_boundary_key(
        self,
        metrics: Dict[str, Any],
        desired_key: str,
        overflow_key: str,
    ) -> str:
        if desired_key in metrics:
            return desired_key
        if len(metrics) < (self.BOUNDARY_MAX_DETAIL_BUCKETS - 1):
            return desired_key
        return overflow_key

    def _record_boundary_serialization_stage(
        self,
        duration_ms: float,
        payload_bytes: int,
        *,
        channel: str = "ws",
        route: str = "broadcast",
        message_type: str = "unknown",
        stage: str = "total",
        outcome: str = "ok",
        include_budget: bool = True,
    ) -> None:
        d = max(0.0, float(duration_ms))
        b = max(0, int(payload_bytes))
        observed_at_s = float(self._boundary_clock())
        with self._boundary_lock:
            self._prune_boundary_samples(
                self._boundary_samples_ms,
                observed_at_s,
            )
            if include_budget and stage == "total":
                self._append_bounded_boundary_sample(
                    self._boundary_samples_ms,
                    (observed_at_s, d, b),
                    limit=self.BOUNDARY_AGGREGATE_SAMPLE_LIMIT,
                    truncation=self._boundary_aggregate_truncation,
                )
                self._boundary_count += 1
                self._boundary_total_ms += d
                self._boundary_max_ms = max(self._boundary_max_ms, d)
                self._boundary_last_ms = d
                self._boundary_total_bytes += b
                self._boundary_last_bytes = b
                if d > float(self._boundary_budget_ms):
                    self._boundary_violations += 1

            # Route summaries are totals only; adding component stages here
            # would multiply counts and dilute route-level p99.
            if stage == "total":
                route_key = self._bounded_boundary_key(
                    self._boundary_route_metrics,
                    f"{channel}|{route}|{message_type}|{outcome}",
                    self.BOUNDARY_ROUTE_OVERFLOW_KEY,
                )
                route_bucket = self._boundary_route_metrics.get(route_key)
                if route_bucket is None:
                    route_bucket = self._new_boundary_bucket()
                    self._boundary_route_metrics[route_key] = route_bucket
                self._prune_boundary_samples(route_bucket["samples"], observed_at_s)
                self._update_boundary_bucket(route_bucket, d, b, observed_at_s)

            stage_key = self._bounded_boundary_key(
                self._boundary_stage_metrics,
                f"{channel}|{route}|{message_type}|{stage}|{outcome}",
                self.BOUNDARY_STAGE_OVERFLOW_KEY,
            )
            stage_bucket = self._boundary_stage_metrics.get(stage_key)
            if stage_bucket is None:
                stage_bucket = self._new_boundary_bucket()
                self._boundary_stage_metrics[stage_key] = stage_bucket
            self._prune_boundary_samples(stage_bucket["samples"], observed_at_s)
            self._update_boundary_bucket(stage_bucket, d, b, observed_at_s)
            if include_budget and stage == "total":
                self._boundary_budget_path_samples.add(stage_key)

    def _record_boundary_serialization_error(
        self,
        *,
        channel: str,
        route: str,
        message_type: str,
        stage: str,
        error: BaseException,
    ) -> None:
        error_type = str(
            getattr(error, "original_error_type", type(error).__name__)
        )
        desired_key = f"{channel}|{route}|{message_type}|{stage}|{error_type}"
        with self._boundary_lock:
            key = self._bounded_boundary_key(
                self._boundary_error_counts,
                desired_key,
                self.BOUNDARY_ERROR_OVERFLOW_KEY,
            )
            self._boundary_error_counts[key] = int(
                self._boundary_error_counts.get(key, 0)
            ) + 1
            self._boundary_errors_total += 1
        callback = self.boundary_failure_callback
        if callable(callback):
            try:
                callback(error)
            except Exception:
                self.logger.critical(
                    "WebSocket boundary failure callback raised",
                    exc_info=True,
                )

    def record_response_model_error(
        self,
        *,
        route: str,
        message_type: str,
        error: BaseException,
    ) -> None:
        self._record_boundary_serialization_error(
            channel="ws",
            route=route,
            message_type=message_type,
            stage="response_model",
            error=error,
        )

    def _record_boundary_serialization(self, duration_ms: float, payload_bytes: int) -> None:
        self._record_boundary_serialization_stage(
            duration_ms,
            payload_bytes,
            channel="ws",
            route="legacy",
            message_type="legacy",
            stage="total",
            outcome="ok",
            include_budget=True,
        )

    @staticmethod
    def _percentile_from_ordered(
        ordered: List[float],
        q: float,
    ) -> Optional[float]:
        if not ordered:
            return None
        q = max(0.0, min(1.0, float(q)))
        if len(ordered) == 1:
            return float(ordered[0])
        idx = int(round(q * (len(ordered) - 1)))
        idx = max(0, min(len(ordered) - 1, idx))
        return float(ordered[idx])

    def _boundary_window_summaries(
        self,
        samples: List[tuple[float, float, int]],
        *,
        now_s: float,
        truncation: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        # One duration sort feeds both timestamp-filtered windows.
        ordered = sorted(samples, key=lambda item: item[1])
        summaries: Dict[str, Dict[str, Any]] = {}
        for window_s in self.BOUNDARY_WINDOWS_S:
            cutoff_s = float(now_s) - float(window_s)
            window_ordered = [item for item in ordered if item[0] >= cutoff_s]
            durations = [float(item[1]) for item in window_ordered]
            chronological = [item for item in samples if item[0] >= cutoff_s]
            count = len(durations)
            total_ms = sum(durations)
            key = f"{int(window_s)}s"
            evidence = (truncation or {}).get(key, {})
            evidence_active = bool(
                int(evidence.get("count", 0)) > 0
                and now_s <= float(evidence.get("until_s", 0.0))
            )
            truncated_max = (
                float(evidence["max_ms"])
                if evidence_active and evidence.get("max_ms") is not None
                else None
            )
            observed_p99 = self._percentile_from_ordered(durations, 0.99)
            summaries[key] = {
                "window_sec": float(window_s),
                "count": count,
                "retained_count": count,
                "avg_ms": (total_ms / float(count)) if count > 0 else None,
                "p50_ms": self._percentile_from_ordered(durations, 0.50),
                "p95_ms": self._percentile_from_ordered(durations, 0.95),
                "p99_ms": self._worst_optional(observed_p99, truncated_max),
                "max_ms": self._worst_optional(
                    durations[-1] if durations else None,
                    truncated_max,
                ),
                "last_ms": float(chronological[-1][1]) if chronological else None,
                "total_bytes": sum(int(item[2]) for item in chronological),
                "last_payload_bytes": (
                    int(chronological[-1][2]) if chronological else 0
                ),
                "violations": sum(
                    1
                    for value in durations
                    if value > float(self._boundary_budget_ms)
                ),
                "truncated": evidence_active,
                "truncated_count": (
                    int(evidence.get("count", 0)) if evidence_active else 0
                ),
                "truncated_max_ms": truncated_max,
            }
        return summaries

    @staticmethod
    def _worst_optional(*values: Any) -> Optional[float]:
        present = [float(value) for value in values if value is not None]
        return max(present) if present else None

    def get_boundary_serialization_metrics(
        self,
        *,
        include_details: bool = True,
    ) -> Dict[str, Any]:
        now_s = float(self._boundary_clock())
        with self._boundary_lock:
            self._prune_boundary_samples(self._boundary_samples_ms, now_s)
            if include_details:
                for bucket in self._boundary_route_metrics.values():
                    self._prune_boundary_samples(bucket["samples"], now_s)
                for bucket in self._boundary_stage_metrics.values():
                    self._prune_boundary_samples(bucket["samples"], now_s)
            else:
                for key in self._boundary_budget_path_samples:
                    bucket = self._boundary_stage_metrics.get(key)
                    if bucket is not None:
                        self._prune_boundary_samples(bucket["samples"], now_s)
            samples = list(self._boundary_samples_ms)
            aggregate_truncation = {
                key: dict(value)
                for key, value in self._boundary_aggregate_truncation.items()
            }
            lifetime_count = int(self._boundary_count)
            lifetime_total_ms = float(self._boundary_total_ms)
            lifetime_max_ms = float(self._boundary_max_ms)
            lifetime_last_ms = float(self._boundary_last_ms)
            lifetime_total_bytes = int(self._boundary_total_bytes)
            lifetime_last_bytes = int(self._boundary_last_bytes)
            lifetime_violations = int(self._boundary_violations)
            route_snapshot = {
                str(k): {
                    "samples": list(v.get("samples", [])),
                    "truncation": {
                        key: dict(value)
                        for key, value in v.get("truncation", {}).items()
                    },
                    "lifetime_count": int(v.get("count", 0)),
                    "lifetime_total_ms": float(v.get("total_ms", 0.0)),
                    "lifetime_max_ms": float(v.get("max_ms", 0.0)),
                    "lifetime_last_ms": float(v.get("last_ms", 0.0)),
                    "lifetime_total_bytes": int(v.get("total_bytes", 0)),
                    "lifetime_last_bytes": int(v.get("last_bytes", 0)),
                }
                for k, v in self._boundary_route_metrics.items()
                if include_details
            }
            stage_snapshot = {
                str(k): {
                    "samples": list(v.get("samples", [])),
                    "truncation": {
                        key: dict(value)
                        for key, value in v.get("truncation", {}).items()
                    },
                    "lifetime_count": int(v.get("count", 0)),
                    "lifetime_total_ms": float(v.get("total_ms", 0.0)),
                    "lifetime_max_ms": float(v.get("max_ms", 0.0)),
                    "lifetime_last_ms": float(v.get("last_ms", 0.0)),
                    "lifetime_total_bytes": int(v.get("total_bytes", 0)),
                    "lifetime_last_bytes": int(v.get("last_bytes", 0)),
                }
                for k, v in self._boundary_stage_metrics.items()
                if include_details
            }
            budget_path_snapshot = {
                key: {
                    "samples": list(
                        self._boundary_stage_metrics[key].get("samples", [])
                    ),
                    "truncation": {
                        window: dict(value)
                        for window, value in self._boundary_stage_metrics[key]
                        .get("truncation", {})
                        .items()
                    },
                }
                for key in self._boundary_budget_path_samples
                if key in self._boundary_stage_metrics
            }
            error_counts = dict(self._boundary_error_counts)
            errors_total = int(self._boundary_errors_total)

        with self._stats_condition:
            stats_collection_failures = int(self._stats_failed_snapshots)
            stats_last_failure_type = self._stats_last_failure_type
        with self._outbound_condition:
            outbound_admission = {
                "admission_open": bool(self._outbound_admission_open),
                "pending_submissions": len(self._outbound_futures),
                "max_pending_submissions": int(self._outbound_max_inflight),
                "admitted_submissions": int(
                    self._outbound_admitted_submissions
                ),
                "completed_submissions": int(
                    self._outbound_completed_submissions
                ),
                "failed_submissions": int(
                    self._outbound_failed_submissions
                ),
                "aborted_submissions": int(
                    self._outbound_aborted_submissions
                ),
                "admitted_bytes": int(self._outbound_admitted_bytes),
                "completed_bytes": int(self._outbound_completed_bytes),
                "aborted_bytes": int(self._outbound_aborted_bytes),
                "inflight_bytes": int(self._outbound_inflight_bytes),
                "peak_inflight_bytes": int(
                    self._outbound_peak_inflight_bytes
                ),
                "max_inflight_bytes": int(
                    self._outbound_max_inflight_bytes
                ),
                "last_failure_type": self._outbound_last_failure_type,
            }
        telemetry_clients = {
            "connected": len(self.connected_clients),
            "peak": int(self._telemetry_client_peak),
            "max": int(self._max_telemetry_clients),
            "rejections": int(self._telemetry_client_rejections),
        }

        def _summarize(
            snapshot: Dict[str, Dict[str, Any]],
            *,
            mark_budgeted: bool,
        ) -> Dict[str, Dict[str, Any]]:
            out: Dict[str, Dict[str, Any]] = {}
            for key, item in snapshot.items():
                windows = self._boundary_window_summaries(
                    item["samples"],
                    now_s=now_s,
                    truncation=item.get("truncation"),
                )
                summary_10s = windows["10s"]
                summary_60s = windows["60s"]
                out[key] = {
                    **summary_60s,
                    "p99_ms": self._worst_optional(
                        summary_10s["p99_ms"],
                        summary_60s["p99_ms"],
                    ),
                    "p99_10s_ms": summary_10s["p99_ms"],
                    "p99_60s_ms": summary_60s["p99_ms"],
                    "lifetime_count": int(item["lifetime_count"]),
                    "lifetime_total_ms": float(item["lifetime_total_ms"]),
                    "lifetime_max_ms": float(item["lifetime_max_ms"]),
                    "lifetime_last_ms": float(item["lifetime_last_ms"]),
                    "lifetime_total_bytes": int(item["lifetime_total_bytes"]),
                    "lifetime_last_payload_bytes": int(
                        item["lifetime_last_bytes"]
                    ),
                }
                if mark_budgeted:
                    out[key]["budgeted"] = key in budget_path_snapshot
            return out

        aggregate_windows = self._boundary_window_summaries(
            samples,
            now_s=now_s,
            truncation=aggregate_truncation,
        )
        route_summary = _summarize(route_snapshot, mark_budgeted=False)
        stage_summary = _summarize(stage_snapshot, mark_budgeted=True)
        budget_windows = {
            key: self._boundary_window_summaries(
                path["samples"],
                now_s=now_s,
                truncation=path.get("truncation"),
            )
            for key, path in budget_path_snapshot.items()
        }

        def _path_value(key: str, window_key: str) -> Any:
            summary = budget_windows[key][window_key]
            # Once cardinality overflows, fail safely on the worst observation
            # rather than letting unrelated paths dilute a sparse offender.
            if key == self.BOUNDARY_STAGE_OVERFLOW_KEY:
                return summary.get("max_ms")
            return summary.get("p99_ms")

        max_path_10s = self._worst_optional(
            *(_path_value(key, "10s") for key in budget_windows)
        )
        max_path_60s = self._worst_optional(
            *(_path_value(key, "60s") for key in budget_windows)
        )
        budget_path_summaries = [
            {
                "key": key,
                "p99_10s_ms": windows["10s"].get("p99_ms"),
                "p99_60s_ms": windows["60s"].get("p99_ms"),
                "max_ms": self._worst_optional(
                    windows["10s"].get("max_ms"),
                    windows["60s"].get("max_ms"),
                ),
                "truncated": bool(
                    windows["10s"].get("truncated")
                    or windows["60s"].get("truncated")
                ),
            }
            for key, windows in budget_windows.items()
        ]
        budget_path_summaries.sort(
            key=lambda item: self._worst_optional(
                item.get("p99_10s_ms"),
                item.get("p99_60s_ms"),
            )
            or 0.0,
            reverse=True,
        )
        aggregate_windows["10s"]["max_path_p99_ms"] = max_path_10s
        aggregate_windows["60s"]["max_path_p99_ms"] = max_path_60s
        summary_10s = aggregate_windows["10s"]
        summary_60s = aggregate_windows["60s"]
        return {
            **summary_60s,
            "p99_ms": self._worst_optional(
                summary_10s["p99_ms"],
                summary_60s["p99_ms"],
            ),
            "p99_10s_ms": summary_10s["p99_ms"],
            "p99_60s_ms": summary_60s["p99_ms"],
            "max_path_p99_ms": self._worst_optional(
                max_path_10s,
                max_path_60s,
            ),
            "max_path_p99_10s_ms": max_path_10s,
            "max_path_p99_60s_ms": max_path_60s,
            "windows": aggregate_windows,
            "budget_ms": float(self._boundary_budget_ms),
            "violations": lifetime_violations,
            "lifetime_count": lifetime_count,
            "lifetime_avg_ms": (
                lifetime_total_ms / float(lifetime_count)
                if lifetime_count > 0
                else None
            ),
            "lifetime_max_ms": (
                lifetime_max_ms if lifetime_count > 0 else None
            ),
            "lifetime_last_ms": (
                lifetime_last_ms if lifetime_count > 0 else None
            ),
            "lifetime_total_bytes": lifetime_total_bytes,
            "lifetime_last_payload_bytes": lifetime_last_bytes,
            "errors_total": errors_total,
            "errors": error_counts,
            "boundary_serialization_errors_total": errors_total,
            "boundary_serialization_errors": error_counts,
            "stats_collection_failures_total": stats_collection_failures,
            "stats_collection_last_failure_type": stats_last_failure_type,
            "outbound_admission": outbound_admission,
            "telemetry_clients": telemetry_clients,
            "detail_truncated": bool(
                summary_10s.get("truncated")
                or summary_60s.get("truncated")
                or self.BOUNDARY_ROUTE_OVERFLOW_KEY in route_summary
                or self.BOUNDARY_STAGE_OVERFLOW_KEY in stage_summary
                or self.BOUNDARY_ERROR_OVERFLOW_KEY in error_counts
                or any(item["truncated"] for item in budget_path_summaries)
            ),
            "detail_bucket_limit": int(self.BOUNDARY_MAX_DETAIL_BUCKETS),
            "sample_limits": {
                "aggregate": int(self.BOUNDARY_AGGREGATE_SAMPLE_LIMIT),
                "per_detail_bucket": int(self.BOUNDARY_DETAIL_SAMPLE_LIMIT),
            },
            "details_included": bool(include_details),
            "budget_path_count": len(budget_path_summaries),
            "top_budget_paths": budget_path_summaries[:8],
            "routes": route_summary,
            "stages": stage_summary,
        }

    def get_boundary_serialization_metrics_compact(self) -> Dict[str, Any]:
        """Return the gate surface without sorting/building detail maps."""

        return self.get_boundary_serialization_metrics(include_details=False)

    def reset_boundary_serialization_metrics(self) -> None:
        with self._boundary_lock:
            self._boundary_samples_ms.clear()
            self._boundary_count = 0
            self._boundary_total_ms = 0.0
            self._boundary_max_ms = 0.0
            self._boundary_last_ms = 0.0
            self._boundary_total_bytes = 0
            self._boundary_last_bytes = 0
            self._boundary_violations = 0
            self._boundary_route_metrics.clear()
            self._boundary_stage_metrics.clear()
            self._boundary_budget_path_samples.clear()
            self._boundary_aggregate_truncation = self._new_boundary_truncation()
            self._boundary_error_counts.clear()
            self._boundary_errors_total = 0

    @staticmethod
    def response_model_timing_since(started_ns: int) -> BoundaryResponseModelTiming:
        return BoundaryResponseModelTiming(
            duration_ms=max(
                0.0,
                (time.perf_counter_ns() - int(started_ns)) / 1_000_000.0,
            )
        )

    @staticmethod
    def timed_payload_since(
        payload: Dict[str, Any],
        started_ns: int,
    ) -> BoundaryTimedPayload:
        timed = BoundaryTimedPayload(payload, BoundaryResponseModelTiming(0.0))
        timed.boundary_response_model_timing = (
            WebSocketServer.response_model_timing_since(started_ns)
        )
        return timed

    @staticmethod
    def _response_model_ms(
        timing: Optional[BoundaryResponseModelTiming],
    ) -> float:
        return max(0.0, float(timing.duration_ms)) if timing is not None else 0.0

    def _require_response_model_timing(
        self,
        payload: Dict[str, Any],
        timing: Optional[BoundaryResponseModelTiming],
        *,
        route: str,
        message_type: str,
    ) -> BoundaryResponseModelTiming:
        resolved = timing
        if resolved is None:
            candidate = getattr(payload, "boundary_response_model_timing", None)
            if isinstance(candidate, BoundaryResponseModelTiming):
                resolved = candidate
        if not isinstance(resolved, BoundaryResponseModelTiming):
            error = BoundaryResponseModelContractError(
                "JSON boundary payload is missing response-model timing"
            )
            self._record_boundary_serialization_error(
                channel="ws",
                route=route,
                message_type=message_type,
                stage="response_model",
                error=error,
            )
            raise error
        return resolved

    def _ensure_serializer_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        with self._serializer_executor_lock:
            executor = self._serializer_executor
            if executor is None:
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="NoesisWSSerializer",
                )
                self._serializer_executor = executor
            return executor

    def _shutdown_serializer_executor(self) -> None:
        with self._serializer_executor_lock:
            executor = self._serializer_executor
            self._serializer_executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    async def _prewarm_serializer_executor(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._ensure_serializer_executor(),
            _prewarm_boundary_serializer,
        )

    def _ensure_provider_executor_locked(
        self,
    ) -> concurrent.futures.ThreadPoolExecutor:
        executor = self._provider_executor
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.PROVIDER_WORKERS,
                thread_name_prefix="NoesisWSProvider",
            )
            self._provider_executor = executor
        return executor

    def _ensure_stats_executor_locked(
        self,
    ) -> concurrent.futures.ThreadPoolExecutor:
        executor = self._stats_executor
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="NoesisWSStats",
            )
            self._stats_executor = executor
        return executor

    def _stats_snapshot_completed(
        self,
        future: concurrent.futures.Future[Any],
    ) -> None:
        result: Any = None
        succeeded = False
        failure_type: Optional[str] = None
        try:
            result = future.result()
            succeeded = True
        except BaseException as exc:
            failure_type = type(exc).__name__
            succeeded = False
        with self._stats_condition:
            generation = self._stats_future_generation.pop(future, -1)
            is_authoritative = bool(
                generation == self._stats_generation
                and self._stats_inflight is future
            )
            if succeeded and is_authoritative:
                self._stats_cache = result
                self._stats_cache_at_s = time.monotonic()
            if not succeeded:
                self._stats_failed_snapshots += 1
                self._stats_last_failure_type = failure_type
            self._stats_futures.discard(future)
            if self._stats_inflight is future:
                self._stats_inflight = None
                self._stats_inflight_kind = None
            if succeeded:
                self._stats_completed_snapshots += 1
            self._stats_condition.notify_all()

    async def _get_stats_snapshot(self, *, force: bool = False) -> Any:
        callback = self.stats_callback
        if not callable(callback):
            return None
        now_s = time.monotonic()
        with self._stats_condition:
            if not self._stats_admission_open:
                raise StatsAdmissionClosed("stats collector admission is closed")
            future = self._stats_inflight
            if future is None and (
                not force
                and self._stats_cache is not None
                and now_s - self._stats_cache_at_s <= self._stats_cache_ttl_s
            ):
                return self._stats_cache
            if future is None:
                executor = self._ensure_stats_executor_locked()
                future = executor.submit(callback)
                self._stats_inflight = future
                self._stats_inflight_kind = "collect"
                self._stats_futures.add(future)
                self._stats_future_generation[future] = self._stats_generation
                self._stats_admitted_snapshots += 1
                future.add_done_callback(self._stats_snapshot_completed)
        return await asyncio.wrap_future(future)

    async def _clear_and_refresh_stats(self) -> Any:
        callback = self.stats_callback
        clear_stats = getattr(callback, "clear_stats", None)
        if not callable(callback) or not callable(clear_stats):
            raise RuntimeError("stats callback does not expose clear_stats")
        while True:
            wait_then_retry = False
            with self._stats_condition:
                if not self._stats_admission_open:
                    raise StatsAdmissionClosed("stats collector admission is closed")
                future = self._stats_inflight
                if future is not None:
                    # Concurrent clear requests share one clear+collect. A clear
                    # arriving behind an ordinary collection waits without
                    # submitting more executor work, then owns the next slot.
                    wait_then_retry = self._stats_inflight_kind != "clear"
                else:
                    executor = self._ensure_stats_executor_locked()
                    self._stats_generation += 1
                    generation = self._stats_generation
                    self._stats_cache = None
                    self._stats_cache_at_s = 0.0

                    def _clear_then_collect() -> Any:
                        clear_stats()
                        return callback()

                    future = executor.submit(_clear_then_collect)
                    self._stats_futures.add(future)
                    self._stats_inflight = future
                    self._stats_inflight_kind = "clear"
                    self._stats_future_generation[future] = generation
                    self._stats_admitted_snapshots += 1
                    future.add_done_callback(self._stats_snapshot_completed)
            result = await asyncio.wrap_future(future)
            if not wait_then_retry:
                return result

    def quiesce_stats_collector(
        self,
        *,
        timeout_s: float,
    ) -> StatsQuiescenceReceipt:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._stats_condition:
            self._stats_admission_open = False
            while self._stats_futures:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    receipt = StatsQuiescenceReceipt(
                        admission_closed=True,
                        admitted_snapshots=self._stats_admitted_snapshots,
                        completed_snapshots=self._stats_completed_snapshots,
                        failed_snapshots=self._stats_failed_snapshots,
                        pending_futures=len(self._stats_futures),
                        executor_joined=False,
                    )
                    self._stats_shutdown_receipt = receipt
                    raise StatsQuiescenceTimeout(receipt)
                self._stats_condition.wait(timeout=remaining)
            executor = self._stats_executor
            self._stats_executor = None

        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        with self._stats_condition:
            receipt = StatsQuiescenceReceipt(
                admission_closed=not self._stats_admission_open,
                admitted_snapshots=self._stats_admitted_snapshots,
                completed_snapshots=self._stats_completed_snapshots,
                failed_snapshots=self._stats_failed_snapshots,
                pending_futures=len(self._stats_futures),
                executor_joined=True,
            )
            self._stats_shutdown_receipt = receipt
        return receipt

    def _provider_call_completed(
        self,
        future: concurrent.futures.Future[Any],
    ) -> None:
        with self._provider_condition:
            self._provider_futures.discard(future)
            self._provider_active_calls -= 1
            self._provider_completed_calls += 1
            self._provider_condition.notify_all()

    async def _run_blocking_provider(
        self,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with self._provider_condition:
            if not self._provider_admission_open:
                raise ProviderAdmissionClosed(
                    "blocking WebSocket provider admission is closed"
                )
            if self._provider_active_calls >= self.PROVIDER_MAX_INFLIGHT:
                raise ProviderCapacityExceeded(
                    "blocking WebSocket provider capacity is exhausted"
                )
            executor = self._ensure_provider_executor_locked()
            self._provider_admitted_calls += 1
            self._provider_active_calls += 1
            try:
                future = executor.submit(callback, *args, **kwargs)
            except Exception:
                self._provider_admitted_calls -= 1
                self._provider_active_calls -= 1
                self._provider_condition.notify_all()
                raise
            self._provider_futures.add(future)
            future.add_done_callback(self._provider_call_completed)
        return await asyncio.wrap_future(future)

    def quiesce_blocking_providers(
        self,
        *,
        timeout_s: float = 5.0,
    ) -> ProviderQuiescenceReceipt:
        """Close provider admission, drain every lease, and join the pool."""

        timeout_s = max(0.0, float(timeout_s))
        deadline = time.monotonic() + timeout_s
        with self._provider_condition:
            self._provider_admission_open = False
            while self._provider_active_calls > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    receipt = ProviderQuiescenceReceipt(
                        admission_closed=True,
                        admitted_calls=self._provider_admitted_calls,
                        completed_calls=self._provider_completed_calls,
                        active_calls=self._provider_active_calls,
                        pending_futures=len(self._provider_futures),
                        executor_joined=False,
                    )
                    raise ProviderQuiescenceTimeout(receipt)
                self._provider_condition.wait(timeout=remaining)
            executor = self._provider_executor
            self._provider_executor = None

        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

        with self._provider_condition:
            receipt = ProviderQuiescenceReceipt(
                admission_closed=not self._provider_admission_open,
                admitted_calls=self._provider_admitted_calls,
                completed_calls=self._provider_completed_calls,
                active_calls=self._provider_active_calls,
                pending_futures=len(self._provider_futures),
                executor_joined=True,
            )
        if not receipt.quiesced:
            raise RuntimeError(f"invalid provider quiescence receipt: {receipt!r}")
        return receipt

    async def _send_json_with_boundary_metrics(
        self,
        websocket: Any,
        payload: Dict[str, Any],
        *,
        route: str,
        message_type: str = "unknown",
        use_to_thread_json: bool = False,
        response_model_timing: Optional[BoundaryResponseModelTiming] = None,
    ) -> None:
        dispatch_wait_ms = 0.0
        response_model_timing = self._require_response_model_timing(
            payload,
            response_model_timing,
            route=route,
            message_type=message_type,
        )
        response_model_ms = self._response_model_ms(response_model_timing)
        try:
            if use_to_thread_json:
                submitted_ns = time.perf_counter_ns()
                loop = asyncio.get_running_loop()
                try:
                    serialization_future = loop.run_in_executor(
                        self._ensure_serializer_executor(),
                        _serialize_boundary_json,
                        payload,
                        submitted_ns,
                    )
                except Exception as exc:
                    self._record_boundary_serialization_error(
                        channel="ws",
                        route=route,
                        message_type=message_type,
                        stage="worker_dispatch_wait",
                        error=exc,
                    )
                    raise _BoundaryDispatchError(type(exc).__name__) from exc
                message_text, payload_bytes, dispatch_wait_ms, convert_ms, encode_ms = (
                    await serialization_future
                )
                self._record_boundary_serialization_stage(
                    dispatch_wait_ms,
                    payload_bytes,
                    channel="ws",
                    route=route,
                    message_type=message_type,
                    stage="worker_dispatch_wait",
                    outcome="ok",
                    include_budget=False,
                )
            else:
                message_text, payload_bytes, _, convert_ms, encode_ms = (
                    _serialize_boundary_json(payload)
                )
        except Exception as exc:
            if not isinstance(exc, _BoundaryDispatchError):
                self._record_boundary_serialization_error(
                    channel="ws",
                    route=route,
                    message_type=message_type,
                    stage=(
                        "numpy_convert"
                        if isinstance(exc, _BoundaryConversionError)
                        else "json_encode"
                    ),
                    error=exc,
                )
            raise
        self._record_boundary_serialization_stage(
            response_model_ms,
            payload_bytes,
            channel="ws",
            route=route,
            message_type=message_type,
            stage="response_model",
            outcome="ok",
            include_budget=False,
        )
        self._record_boundary_serialization_stage(
            convert_ms,
            payload_bytes,
            channel="ws",
            route=route,
            message_type=message_type,
            stage="numpy_convert",
            outcome="ok",
            include_budget=False,
        )

        self._record_boundary_serialization_stage(
            encode_ms,
            payload_bytes,
            channel="ws",
            route=route,
            message_type=message_type,
            stage="json_encode",
            outcome="ok",
            include_budget=False,
        )

        # Measure local coroutine/task construction only. Awaiting the task is
        # still lifecycle-owned, but transport flow-control is outside the CPU
        # serialization budget.
        send_start_ns = time.perf_counter_ns()
        try:
            delivery_task = asyncio.create_task(websocket.send(message_text))
        except Exception as exc:
            self._record_boundary_serialization_error(
                channel="ws",
                route=route,
                message_type=message_type,
                stage="send_dispatch",
                error=exc,
            )
            raise
        send_ms = (time.perf_counter_ns() - send_start_ns) / 1_000_000.0
        self._record_boundary_serialization_stage(
            send_ms,
            payload_bytes,
            channel="ws",
            route=route,
            message_type=message_type,
            stage="send_dispatch",
            outcome="ok",
            include_budget=False,
        )

        # The acceptance total is the complete local boundary: executor dispatch
        # (when used), conversion, JSON encoding, and websocket send dispatch.
        total_ms = float(
            response_model_ms
            + dispatch_wait_ms
            + convert_ms
            + encode_ms
            + send_ms
        )
        self._record_boundary_serialization_stage(
            total_ms,
            payload_bytes,
            channel="ws",
            route=route,
            message_type=message_type,
            stage="total",
            outcome="ok",
            include_budget=True,
        )
        await delivery_task

    def _get_webrtc_owner(self) -> Optional[Any]:
        owner = self._webrtc_owner
        if owner is None:
            return None
        if owner in self.connected_clients:
            return owner
        # Owner disconnected; clear to avoid sending signaling to stale sockets.
        self._webrtc_owner = None
        self._webrtc_owner_ip = None
        return None

    def _get_gateway_owner(self, gateway: Any) -> Optional[Any]:
        owner = self._webrtc_gateway_owner.get(gateway)
        if owner is None:
            return None
        if owner in self.connected_clients:
            return owner
        self._webrtc_gateway_owner.pop(gateway, None)
        self._webrtc_gateway_owner_ip.pop(gateway, None)
        for client, assigned_gateway in list(self._webrtc_client_gateway.items()):
            if assigned_gateway is gateway:
                self._webrtc_client_gateway.pop(client, None)
        if not self._webrtc_shutdown:
            self._reset_revoked_gateway(gateway, reason="stale_owner")
        return None

    def _set_gateway_owner(self, gateway: Any, websocket: Any, client_ip: str) -> None:
        if self._webrtc_shutdown:
            return
        self._webrtc_gateway_owner[gateway] = websocket
        self._webrtc_gateway_owner_ip[gateway] = client_ip
        self._webrtc_client_gateway[websocket] = gateway
        self._notify_webrtc_activity()

    def _clear_gateway_owner_for_client(self, websocket: Any) -> None:
        with self._webrtc_lifecycle_condition:
            gateway = self._webrtc_client_gateway.pop(websocket, None)
            released_gateways = set()
            if gateway is not None and self._webrtc_gateway_owner.get(gateway) is websocket:
                self._webrtc_gateway_owner.pop(gateway, None)
                self._webrtc_gateway_owner_ip.pop(gateway, None)
                released_gateways.add(gateway)
            for gw, owner in list(self._webrtc_gateway_owner.items()):
                if owner is websocket:
                    self._webrtc_gateway_owner.pop(gw, None)
                    self._webrtc_gateway_owner_ip.pop(gw, None)
                    released_gateways.add(gw)
            if gateway is not None:
                released_gateways.add(gateway)
            shutdown = bool(self._webrtc_shutdown)
        if shutdown:
            return
        for released_gateway in released_gateways:
            if not self._retire_extra_idle_gateway(released_gateway):
                self._reset_revoked_gateway(released_gateway, reason="owner_disconnected")
        self._notify_webrtc_activity()

    def _schedule_gateway_stop(
        self,
        gateway: Any,
        *,
        reason: str,
        ownership_registered: bool = False,
    ) -> None:
        stop = getattr(gateway, "stop", None)
        if not callable(stop):
            with self._webrtc_lifecycle_condition:
                self._webrtc_lifecycle_failures.append(
                    f"{reason}: gateway has no stop method"
                )
            return

        with self._webrtc_lifecycle_condition:
            if not ownership_registered:
                self._webrtc_retired_gateways.add(gateway)
            if self._webrtc_shutdown:
                return
            executor = self._webrtc_lifecycle_executor
            if executor is None:
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="NoesisWebRTCLifecycle",
                )
                self._webrtc_lifecycle_executor = executor
            future = executor.submit(stop)
            self._webrtc_lifecycle_futures.add(future)
            self._webrtc_lifecycle_owned_gateways.add(gateway)

        def _complete(done: concurrent.futures.Future[Any]) -> None:
            failure: Optional[str] = None
            try:
                done.result()
            except Exception as exc:
                failure = f"{reason}: {type(exc).__name__}: {exc}"
            with self._webrtc_lifecycle_condition:
                self._webrtc_lifecycle_futures.discard(done)
                self._webrtc_lifecycle_owned_gateways.discard(gateway)
                self._webrtc_retired_gateways.discard(gateway)
                if failure is not None:
                    self._webrtc_lifecycle_failures.append(failure)
                self._webrtc_lifecycle_condition.notify_all()

        future.add_done_callback(_complete)

    def begin_webrtc_shutdown(self, *, timeout_s: float = 5.0) -> List[Any]:
        """Close gateway admission and drain every owned lifecycle worker."""

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._webrtc_lifecycle_condition:
            self._webrtc_shutdown = True
            lifecycle_owned = set(self._webrtc_lifecycle_owned_gateways)
            gateways = list(
                dict.fromkeys(
                    list(self.webrtc_gateways)
                    + [
                        gateway
                        for gateway in self._webrtc_retired_gateways
                        if gateway not in lifecycle_owned
                    ]
                )
            )
            self._webrtc_gateway_factory = None
            self._webrtc_gateway_owner.clear()
            self._webrtc_gateway_owner_ip.clear()
            self._webrtc_client_gateway.clear()
            self._webrtc_gateway_resetting.clear()
            self.webrtc_gateways.clear()
            self.webrtc_gateway = None
            self._webrtc_owner = None
            self._webrtc_owner_ip = None
            self.webrtc_activity_callback = None
            while self._webrtc_lifecycle_futures:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(
                        "WebRTC lifecycle workers did not quiesce before timeout"
                    )
                self._webrtc_lifecycle_condition.wait(timeout=remaining)
            executor = self._webrtc_lifecycle_executor
            self._webrtc_lifecycle_executor = None
            failures = list(self._webrtc_lifecycle_failures)
            for gateway in gateways:
                self._webrtc_retired_gateways.discard(gateway)

        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        if failures:
            raise RuntimeError(
                "WebRTC lifecycle worker failure: " + "; ".join(failures)
            )
        return gateways

    def _remove_failed_gateway(self, gateway: Any) -> None:
        with self._webrtc_lifecycle_condition:
            if self._webrtc_shutdown:
                return
            self._webrtc_gateway_resetting.discard(gateway)
            if gateway not in self.webrtc_gateways:
                return
            self.webrtc_gateways.remove(gateway)
            self._webrtc_retired_gateways.add(gateway)
            self._webrtc_gateway_owner.pop(gateway, None)
            self._webrtc_gateway_owner_ip.pop(gateway, None)
            for client, assigned_gateway in list(self._webrtc_client_gateway.items()):
                if assigned_gateway is gateway:
                    self._webrtc_client_gateway.pop(client, None)
            if self.webrtc_gateway is gateway:
                self.webrtc_gateway = self.webrtc_gateways[0] if self.webrtc_gateways else None

        self._schedule_gateway_stop(
            gateway,
            reason="reset_failure",
            ownership_registered=True,
        )

    def report_webrtc_gateway_failure(self, gateway: Any, reason: str) -> None:
        """Retire one failed media slot without affecting healthy peers."""
        self.logger.error(
            "WebRTC gateway slot failed and will be retired: %s",
            str(reason or "gateway_pipeline_failure"),
        )
        self._remove_failed_gateway(gateway)

    def _reset_revoked_gateway(self, gateway: Any, *, reason: str) -> None:
        with self._webrtc_lifecycle_condition:
            if (
                self._webrtc_shutdown
                or gateway not in self.webrtc_gateways
                or gateway in self._webrtc_gateway_resetting
            ):
                return
            reset_peer = getattr(gateway, "reset_peer", None)
            if callable(reset_peer):
                self._webrtc_gateway_resetting.add(gateway)
        if not callable(reset_peer):
            self.logger.error("WebRTC gateway cannot revoke media; removing unsafe slot")
            self._remove_failed_gateway(gateway)
            return

        def _complete(ok: bool) -> None:
            with self._webrtc_lifecycle_condition:
                self._webrtc_gateway_resetting.discard(gateway)
                shutdown = bool(self._webrtc_shutdown)
            if not ok and not shutdown:
                self.logger.error("WebRTC peer reset failed; removing gateway slot")
                self._remove_failed_gateway(gateway)

        try:
            reset_peer(reason=reason, on_complete=_complete)
        except Exception:
            self.logger.exception("WebRTC gateway peer reset raised")
            _complete(False)

    def _has_active_webrtc_owner(self) -> bool:
        for gateway in list(self.webrtc_gateways):
            if self._get_gateway_owner(gateway) is not None:
                return True
        return self._get_webrtc_owner() is not None

    def _notify_webrtc_activity(self) -> None:
        if self._webrtc_shutdown:
            return
        callback = self.webrtc_activity_callback
        if not callable(callback):
            return
        try:
            callback(bool(self._has_active_webrtc_owner()))
        except Exception:
            self.logger.debug("WebRTC activity callback failed", exc_info=True)

    def _retire_extra_idle_gateway(self, gateway: Any) -> bool:
        # Removal from the live registry and transfer into retired ownership are
        # one transaction. Shutdown therefore sees the gateway in exactly one
        # registry even when it races this retirement.
        with self._webrtc_lifecycle_condition:
            if self._webrtc_shutdown:
                return False
            if gateway not in self.webrtc_gateways:
                return False
            owner = self._webrtc_gateway_owner.get(gateway)
            if owner is not None and owner in self.connected_clients:
                return False
            if gateway in self._webrtc_gateway_resetting:
                return False
            if len(self.webrtc_gateways) <= int(self._webrtc_initial_clients):
                return False
            self.webrtc_gateways.remove(gateway)
            self._webrtc_retired_gateways.add(gateway)
            self._webrtc_gateway_resetting.discard(gateway)
            if self.webrtc_gateway is gateway:
                self.webrtc_gateway = self.webrtc_gateways[0] if self.webrtc_gateways else None

        self._schedule_gateway_stop(
            gateway,
            reason="idle_retirement",
            ownership_registered=True,
        )
        return True

    def _create_webrtc_gateway(self) -> Optional[Any]:
        if self._webrtc_shutdown:
            return None
        if len(self.webrtc_gateways) >= int(self._webrtc_max_clients):
            return None
        factory = self._webrtc_gateway_factory
        if not callable(factory):
            return None
        gateway: Optional[Any] = None
        try:
            gateway = factory()
            if gateway is not None and gateway not in self.webrtc_gateways:
                self.register_webrtc_gateway(gateway)
            return gateway
        except Exception:
            self.logger.exception("WebRTC gateway factory failed")
            if gateway is not None:
                stop = getattr(gateway, "stop", None)
                if callable(stop):
                    try:
                        stop()
                    except Exception:
                        self.logger.exception(
                            "Unregistered WebRTC gateway did not stop after factory failure"
                        )
            return None

    def _select_gateway_for_client(self, websocket: Any) -> Optional[Any]:
        if self._webrtc_shutdown:
            return None
        assigned_gateway = self._webrtc_client_gateway.get(websocket)
        if assigned_gateway in self.webrtc_gateways and assigned_gateway not in self._webrtc_gateway_resetting:
            owner = self._get_gateway_owner(assigned_gateway)
            if owner is None or owner is websocket:
                return assigned_gateway
            self._webrtc_client_gateway.pop(websocket, None)

        for gateway in self.webrtc_gateways:
            if gateway in self._webrtc_gateway_resetting:
                continue
            owner = self._get_gateway_owner(gateway)
            if gateway in self._webrtc_gateway_resetting:
                continue
            if owner is None:
                return gateway
        return self._create_webrtc_gateway()

    @classmethod
    def _canonical_message_type(cls, message: Any) -> Optional[str]:
        if isinstance(message, FrozenOutboundJSON):
            message_type = cls._raw_json_object_type(message.encoded)
        elif isinstance(message, dict):
            message_type = str(message.get("type", ""))
        elif isinstance(message, (str, bytes)):
            message_type = cls._raw_json_object_type(message)
        else:
            return None
        return (
            message_type
            if message_type in cls.CANONICAL_MESSAGE_TYPES
            else None
        )

    @classmethod
    def _raw_json_object_type(cls, message: str | bytes) -> str:
        if isinstance(message, bytes):
            if len(message) > int(cls.OUTBOUND_MESSAGE_MAX_BYTES):
                return ""
            candidate = message.lstrip()
            if candidate.startswith(b"\xef\xbb\xbf"):
                candidate = candidate[3:].lstrip()
            if not candidate.startswith(b"{"):
                return ""
            try:
                text = candidate.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                return ""
        else:
            if len(message) > int(cls.OUTBOUND_MESSAGE_MAX_BYTES):
                return ""
            text = message.lstrip()
            if text.startswith("\ufeff"):
                text = text[1:].lstrip()
            if not text.startswith("{"):
                return ""
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return ""
        if not isinstance(payload, dict):
            return ""
        value = payload.get("type")
        return str(value) if isinstance(value, str) else ""

    def _validate_frozen_outbound(self, message: Any) -> None:
        if not isinstance(message, FrozenOutboundJSON):
            return
        if message.owner_token is not self._frozen_outbound_token:
            raise FrozenOutboundOwnershipError(
                "pre-encoded outbound JSON does not belong to this server"
            )
        encoded_type = self._raw_json_object_type(message.encoded)
        if not encoded_type or encoded_type != str(message.message_type):
            raise FrozenOutboundOwnershipError(
                "pre-encoded outbound JSON type does not match its frozen envelope"
            )
        if (
            not message.encoded.isascii()
            or len(message.encoded) != int(message.payload_bytes)
        ):
            raise FrozenOutboundOwnershipError(
                "pre-encoded outbound JSON byte count is not exact"
            )

    @classmethod
    def _reject_generic_canonical_route(
        cls,
        message: Any,
        *,
        route: str,
    ) -> None:
        message_type = cls._canonical_message_type(message)
        if message_type is None:
            return
        required = (
            "admit_broadcast_batch_sync"
            if message_type in cls.AUTHORITY_GATED_MESSAGE_TYPES
            else "broadcast_bev_sync"
        )
        raise CanonicalOutboundRouteRequired(
            f"canonical {message_type} cannot use {route}; "
            f"use {required}"
        )

    async def _send_frozen_json_to_client(
        self,
        websocket: Any,
        message: FrozenOutboundJSON,
        *,
        publisher_enqueue_ms: float,
        response_model_timing: Optional[BoundaryResponseModelTiming],
    ) -> None:
        if not isinstance(response_model_timing, BoundaryResponseModelTiming):
            raise BoundaryResponseModelContractError(
                "pre-encoded WebSocket JSON is missing response timing"
            )
        response_model_ms = self._response_model_ms(response_model_timing)
        payload_bytes = int(message.payload_bytes)
        route = str(message.route)
        self._record_boundary_serialization_stage(
            response_model_ms,
            payload_bytes,
            channel="ws",
            route=route,
            message_type=message.message_type,
            stage="response_model",
            outcome="ok",
            include_budget=False,
        )
        self._record_boundary_serialization_stage(
            publisher_enqueue_ms,
            payload_bytes,
            channel="ws",
            route=route,
            message_type=message.message_type,
            stage="publisher_enqueue",
            outcome="ok",
            include_budget=False,
        )
        send_started_ns = time.perf_counter_ns()
        try:
            delivery_task = asyncio.create_task(websocket.send(message.encoded))
        except Exception as exc:
            self._record_boundary_serialization_error(
                channel="ws",
                route=route,
                message_type=message.message_type,
                stage="send_dispatch",
                error=exc,
            )
            raise
        send_ms = (
            time.perf_counter_ns() - send_started_ns
        ) / 1_000_000.0
        self._record_boundary_serialization_stage(
            send_ms,
            payload_bytes,
            channel="ws",
            route=route,
            message_type=message.message_type,
            stage="send_dispatch",
            outcome="ok",
            include_budget=False,
        )
        total_ms = float(
            response_model_ms
            + publisher_enqueue_ms
            + message.convert_ms
            + message.encode_ms
            + send_ms
        )
        self._record_boundary_serialization_stage(
            total_ms,
            payload_bytes,
            channel="ws",
            route=route,
            message_type=message.message_type,
            stage="total",
            outcome="ok",
            include_budget=True,
        )
        await delivery_task

    async def _send_to_client(
        self,
        websocket: Any,
        message: Any,
        *,
        publisher_enqueue_ms: float = 0.0,
        response_model_timing: Optional[BoundaryResponseModelTiming] = None,
    ) -> None:
        """Send a message to one client, mirroring broadcast() semantics."""
        self._validate_frozen_outbound(message)
        self._reject_generic_canonical_route(
            message,
            route="send_to_client",
        )
        if not websocket or websocket not in self.connected_clients:
            return

        try:
            if isinstance(message, FrozenOutboundJSON):
                if message.telemetry_payload is not None:
                    try:
                        self._record_tx(message.telemetry_payload)
                    except Exception:
                        pass
                await self._send_frozen_json_to_client(
                    websocket,
                    message,
                    publisher_enqueue_ms=publisher_enqueue_ms,
                    response_model_timing=response_model_timing,
                )
            elif isinstance(message, dict):
                message_type = str(message.get("type", "unknown"))
                response_model_timing = self._require_response_model_timing(
                    message,
                    response_model_timing,
                    route="send_to_client",
                    message_type=message_type,
                )
                try:
                    self._record_tx(message)
                except Exception:
                    pass
                await self._send_json_with_boundary_metrics(
                    websocket,
                    message,
                    route="send_to_client",
                    message_type=message_type,
                    response_model_timing=response_model_timing,
                )
            elif isinstance(message, str):
                await websocket.send(message)
            elif isinstance(message, bytes):
                await websocket.send(message)
            else:
                self.logger.warning("Unknown message type for send_to_client: %s", type(message))
        except BoundaryResponseModelContractError:
            # A local producer-contract failure is not evidence that the remote
            # client disconnected. Keep the client registered and fail loudly.
            raise
        except Exception as exc:
            client_ip = websocket.remote_address if hasattr(websocket, 'remote_address') else "Unknown"
            self.logger.debug("Failed to send message to %s: %s", client_ip, exc)
            try:
                self.connected_clients.discard(websocket)
            except Exception:
                pass

    async def _send_to_client_from_sync_submission(
        self,
        websocket: Any,
        message: Any,
        submitted_ns: int,
        *,
        response_model_timing: Optional[BoundaryResponseModelTiming],
    ) -> None:
        await self._send_to_client(
            websocket,
            message,
            publisher_enqueue_ms=max(
                0.0,
                (time.perf_counter_ns() - submitted_ns) / 1_000_000.0,
            ),
            response_model_timing=response_model_timing,
        )

    def _outbound_submission_completed(
        self,
        future: concurrent.futures.Future[Any],
    ) -> None:
        failure: Optional[BaseException] = None
        aborted = False
        try:
            if future.cancelled():
                failure = RuntimeError("outbound submission was cancelled")
            else:
                result = future.result()
                aborted = result is _OUTBOUND_ABORTED
        except BaseException as exc:  # future inspection must never escape
            failure = exc
        with self._outbound_condition:
            self._outbound_futures.discard(future)
            reserved_bytes = int(
                self._outbound_bytes_by_future.pop(future, 0)
            )
            if reserved_bytes > self._outbound_inflight_bytes:
                invariant_failure = RuntimeError(
                    "outbound byte reservation accounting underflow"
                )
                if failure is None:
                    failure = invariant_failure
                self._outbound_inflight_bytes = 0
            else:
                self._outbound_inflight_bytes -= reserved_bytes
            self._outbound_completed_bytes += reserved_bytes
            self._outbound_completed_submissions += 1
            if aborted:
                self._outbound_aborted_submissions += 1
                self._outbound_aborted_bytes += reserved_bytes
            if failure is not None:
                self._outbound_failed_submissions += 1
                self._outbound_last_failure_type = type(failure).__name__
            self._outbound_condition.notify_all()
        if failure is not None:
            self.logger.error(
                "Admitted WebSocket submission failed asynchronously: %s: %s",
                type(failure).__name__,
                failure,
            )
            callback = self.boundary_failure_callback
            if callable(callback):
                try:
                    callback(failure)
                except Exception:
                    self.logger.exception(
                        "WebSocket boundary failure callback raised"
                    )

    def _admit_outbound_coroutine(
        self,
        coroutine_factory: Callable[[], Any],
        *,
        message_count: int = 1,
        payload_bytes: int = 0,
    ) -> tuple[
        OutboundAdmissionReceipt,
        concurrent.futures.Future[Any],
    ]:
        """Atomically admit, schedule, and register one cross-thread send."""

        count = int(message_count)
        if count <= 0 or count > int(self.OUTBOUND_BATCH_MAX_MESSAGES):
            raise ValueError(
                "outbound message_count must be within the bounded batch limit"
            )
        byte_count = int(payload_bytes)
        if byte_count < 0:
            raise ValueError("outbound payload_bytes must be nonnegative")
        with self._outbound_condition:
            if not self._outbound_admission_open:
                raise OutboundAdmissionClosed("outbound admission is closed")
            if not self.running:
                raise OutboundAdmissionClosed("WebSocket server is stopped")
            loop = self.event_loop
            if loop is None:
                raise OutboundAdmissionClosed(
                    "WebSocket event loop is unavailable"
                )
            try:
                loop_closed = bool(loop.is_closed())
            except Exception:
                loop_closed = False
            if loop_closed:
                raise OutboundAdmissionClosed(
                    "WebSocket event loop is closed"
                )
            if len(self._outbound_futures) >= int(self._outbound_max_inflight):
                raise OutboundCapacityExceeded(
                    "WebSocket outbound submission capacity is exhausted"
                )
            if (
                self._outbound_inflight_bytes + byte_count
                > int(self._outbound_max_inflight_bytes)
            ):
                raise OutboundCapacityExceeded(
                    "WebSocket outbound byte capacity is exhausted"
                )

            coroutine = coroutine_factory()
            try:
                future = asyncio.run_coroutine_threadsafe(coroutine, loop)
            except Exception:
                close = getattr(coroutine, "close", None)
                if callable(close):
                    close()
                raise
            self._outbound_admitted_submissions += 1
            submission_id = int(self._outbound_admitted_submissions)
            admitted_at_ns = time.perf_counter_ns()
            self._outbound_futures.add(future)
            self._outbound_bytes_by_future[future] = byte_count
            self._outbound_admitted_bytes += byte_count
            self._outbound_inflight_bytes += byte_count
            self._outbound_peak_inflight_bytes = max(
                self._outbound_peak_inflight_bytes,
                self._outbound_inflight_bytes,
            )
            receipt = OutboundAdmissionReceipt(
                submission_id=submission_id,
                admitted_at_ns=admitted_at_ns,
                message_count=count,
                payload_bytes=byte_count,
                inflight_bytes=int(self._outbound_inflight_bytes),
                max_inflight_bytes=int(self._outbound_max_inflight_bytes),
            )
        future.add_done_callback(self._outbound_submission_completed)
        return receipt, future

    def _submit_outbound_coroutine(
        self,
        coroutine_factory: Callable[[], Any],
        *,
        message_count: int = 1,
        payload_bytes: int = 0,
    ) -> OutboundAdmissionReceipt:
        receipt, _future = self._admit_outbound_coroutine(
            coroutine_factory,
            message_count=message_count,
            payload_bytes=payload_bytes,
        )
        return receipt

    async def quiesce_outbound_submissions(
        self,
        *,
        timeout_s: float,
    ) -> OutboundQuiescenceReceipt:
        """Close sync-publisher admission and await every admitted coroutine."""

        with self._outbound_condition:
            self._outbound_admission_open = False
            futures = list(self._outbound_futures)
            admitted = int(self._outbound_admitted_submissions)

        if futures:
            wrapped = [asyncio.wrap_future(future) for future in futures]
            _, pending = await asyncio.wait(
                wrapped,
                timeout=max(0.0, float(timeout_s)),
            )
            if not pending:
                await asyncio.gather(*wrapped, return_exceptions=True)

        with self._outbound_condition:
            receipt = OutboundQuiescenceReceipt(
                admission_closed=not self._outbound_admission_open,
                admitted_submissions=admitted,
                completed_submissions=int(self._outbound_completed_submissions),
                failed_submissions=int(self._outbound_failed_submissions),
                aborted_submissions=int(self._outbound_aborted_submissions),
                pending_futures=len(self._outbound_futures),
                admitted_bytes=int(self._outbound_admitted_bytes),
                completed_bytes=int(self._outbound_completed_bytes),
                aborted_bytes=int(self._outbound_aborted_bytes),
                inflight_bytes=int(self._outbound_inflight_bytes),
                peak_inflight_bytes=int(self._outbound_peak_inflight_bytes),
                max_inflight_bytes=int(self._outbound_max_inflight_bytes),
            )
            self._outbound_shutdown_receipt = receipt
        if not receipt.quiesced:
            raise OutboundQuiescenceTimeout(receipt)
        return receipt

    def send_to_client_sync(
        self,
        websocket: Any,
        message: Any,
        *,
        response_model_timing: Optional[BoundaryResponseModelTiming] = None,
    ) -> OutboundAdmissionReceipt:
        """Thread-safe one-client send for use from other threads."""
        self._validate_frozen_outbound(message)
        self._reject_generic_canonical_route(
            message,
            route="send_to_client_sync",
        )
        if isinstance(message, dict):
            response_model_timing = self._require_response_model_timing(
                message,
                response_model_timing,
                route="send_to_client",
                message_type=str(message.get("type", "unknown")),
            )
        frozen_message, payload_bytes = self._freeze_outbound_message(
            message,
            route="send_to_client",
        )
        submitted_ns = time.perf_counter_ns()
        return self._submit_outbound_coroutine(
            lambda: self._send_to_client_from_sync_submission(
                websocket,
                frozen_message,
                submitted_ns,
                response_model_timing=response_model_timing,
            ),
            message_count=1,
            payload_bytes=payload_bytes,
        )

    # ---------------- Menon telemetry helpers ----------------
    def _telemetry_now(self) -> float:
        try:
            return time.time()
        except Exception:
            return 0.0

    def _short_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Produce a compact view of calibration/coordinate payloads for logging."""
        try:
            t = d.get('type') if isinstance(d, dict) else None
            out: Dict[str, Any] = {'type': t}
            if t == 'set_align':
                al = d.get('align', {}) if isinstance(d, dict) else {}
                mx = al.get('matrix') if isinstance(al, dict) else None
                out.update({
                    'floor_y': al.get('floor_y'),
                    's': (al.get('units') or {}).get('s_obj_to_m') if isinstance(al.get('units'), dict) else None,
                    'matrix': f"len={len(mx)}" if isinstance(mx, list) else None
                })
            elif t == 'set_extrinsics':
                out.update({
                    'cameraId': d.get('cameraId'),
                    'E': f"len={len(d.get('E'))}" if isinstance(d.get('E'), list) else None,
                    'Twc': f"len={len(d.get('Twc'))}" if isinstance(d.get('Twc'), list) else None,
                })
            elif t == 'solve_pnp':
                pts2 = d.get('points2D') if isinstance(d.get('points2D'), list) else None
                pts3 = d.get('points3D') if isinstance(d.get('points3D'), list) else None
                out.update({'cameraId': d.get('cameraId'), 'points2D': len(pts2) if pts2 else 0, 'points3D': len(pts3) if pts3 else 0})
            elif t == 'pixel_to_world':
                out.update({'camId': d.get('camId') or d.get('cameraId'), 'u': d.get('u'), 'v': d.get('v'), 'request_id': d.get('request_id') or d.get('reqId')})
            elif t == 'pixel_to_world_response':
                out.update({'ok': d.get('ok'), 'world': d.get('world'), 'request_id': d.get('request_id') or d.get('reqId')})
            elif t == 'set_extrinsics_result' or t == 'set_align_result' or t == 'solve_pnp_result':
                out.update({'ok': d.get('ok'), 'error': d.get('error')})
            elif t == 'calibration-bundle':
                data = d.get('data') if isinstance(d.get('data'), dict) else {}
                cams = data.get('cameras') if isinstance(data.get('cameras'), dict) else {}
                al = data.get('align') if isinstance(data.get('align'), dict) else {}
                out.update({'cameras': len(cams), 'floor_y': al.get('floor_y'), 's': (al.get('units') or {}).get('s_obj_to_m') if isinstance(al.get('units'), dict) else None})
            else:
                # Default: include a shallow subset
                for k in ('cameraId', 'camId', 'u', 'v', 'ok'):
                    if k in d:
                        out[k] = d.get(k)
            return out
        except Exception:
            return {'type': d.get('type') if isinstance(d, dict) else None}

    def _record_rx(self, msg: Dict[str, Any]) -> None:
        try:
            t = msg.get('type') if isinstance(msg, dict) else None
            if t in {'set_align', 'set_extrinsics', 'solve_pnp', 'pixel_to_world'}:
                self._telemetry['rx'][t] = {'t': self._telemetry_now(), 'data': self._short_dict(msg)}
        except Exception:
            pass

    def _record_tx(self, msg: Dict[str, Any]) -> None:
        try:
            t = msg.get('type') if isinstance(msg, dict) else None
            if t in {'calibration-bundle', 'set_align_result', 'set_extrinsics_result', 'solve_pnp_result', 'pixel_to_world_response'}:
                self._telemetry['tx'][t] = {'t': self._telemetry_now(), 'data': self._short_dict(msg)}
        except Exception:
            pass

    def _resolve_trail_settings(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        if callable(self.trail_settings_getter):
            try:
                raw = self.trail_settings_getter() or {}
                if isinstance(raw, dict):
                    cfg = dict(raw)
            except Exception as exc:
                self.logger.debug("Trail settings getter failed: %s", exc)
        cfg["enabled"] = bool(self.initial_trail_state)
        return cfg

    # ---------------- WebRTC gateway signaling ----------------

    def register_webrtc_gateway(self, gateway: Any) -> None:
        """Register the MosaicWebRTCGateway for signaling."""
        with self._webrtc_lifecycle_condition:
            if self._webrtc_shutdown:
                raise RuntimeError("WebRTC gateway admission is closed")
            if gateway not in self.webrtc_gateways:
                if len(self.webrtc_gateways) >= self._webrtc_max_clients:
                    self.logger.warning(
                        "Ignoring extra WebRTC gateway registration (max=%d)",
                        self._webrtc_max_clients,
                    )
                    return
                self.webrtc_gateways.append(gateway)
                self.logger.info(
                    "WebRTC gateway registered with WebSocketServer (%d/%d)",
                    len(self.webrtc_gateways),
                    self._webrtc_max_clients,
                )
            self.webrtc_gateway = self.webrtc_gateways[0] if self.webrtc_gateways else gateway

    def register_webrtc_gateway_factory(
        self,
        factory: Callable[[], Any],
        *,
        max_clients: Optional[int] = None,
        initial_clients: Optional[int] = None,
    ) -> None:
        """Register a demand-driven gateway factory for extra WebRTC slots."""
        with self._webrtc_lifecycle_condition:
            if self._webrtc_shutdown:
                raise RuntimeError("WebRTC gateway factory admission is closed")
            self._webrtc_gateway_factory = factory
            if max_clients is not None:
                self._webrtc_max_clients = max(1, int(max_clients))
            if initial_clients is not None:
                self._webrtc_initial_clients = max(0, int(initial_clients))

    def send_webrtc_answer(self, sdp: str, gateway: Optional[Any] = None) -> None:
        """Send WebRTC answer SDP to the owning client (fallback: broadcast)."""
        if self._webrtc_shutdown:
            return
        response_model_started_ns = time.perf_counter_ns()
        msg = {"type": "webrtc_answer", "sdp": sdp}
        response_model_timing = self.response_model_timing_since(
            response_model_started_ns
        )
        owner = self._get_gateway_owner(gateway) if gateway is not None else self._get_webrtc_owner()
        owner_ip = self._webrtc_gateway_owner_ip.get(gateway) if gateway is not None else self._webrtc_owner_ip
        if owner is not None:
            self.logger.info("<<< Sending webrtc_answer to WebRTC owner %s", owner_ip or "unknown")
        else:
            if gateway is not None:
                self.logger.warning("No gateway owner for webrtc_answer; dropping answer")
            else:
                self.logger.info("<<< Sending webrtc_answer to %d connected clients", len(self.connected_clients))
        if owner is not None:
            self.send_to_client_sync(
                owner,
                msg,
                response_model_timing=response_model_timing,
            )
        elif gateway is None:
            self.broadcast_sync(
                msg,
                response_model_timing=response_model_timing,
            )

    def send_webrtc_ice(self, mline_index: int, candidate: str, gateway: Optional[Any] = None) -> None:
        """Send WebRTC ICE candidate to the owning client (fallback: broadcast)."""
        if self._webrtc_shutdown:
            return
        response_model_started_ns = time.perf_counter_ns()
        msg = {
            "type": "webrtc_ice_candidate",
            "candidate": candidate,
            "sdpMLineIndex": mline_index,
        }
        response_model_timing = self.response_model_timing_since(
            response_model_started_ns
        )
        owner = self._get_gateway_owner(gateway) if gateway is not None else self._get_webrtc_owner()
        owner_ip = self._webrtc_gateway_owner_ip.get(gateway) if gateway is not None else self._webrtc_owner_ip
        if owner is not None:
            self.logger.info(
                "<<< Sending webrtc_ice_candidate (mline=%d) to WebRTC owner %s",
                mline_index,
                owner_ip or "unknown",
            )
        else:
            if gateway is not None:
                self.logger.warning("No gateway owner for webrtc_ice_candidate; dropping candidate")
            else:
                self.logger.info("<<< Sending webrtc_ice_candidate (mline=%d) to %d clients", mline_index, len(self.connected_clients))
        if owner is not None:
            self.send_to_client_sync(
                owner,
                msg,
                response_model_timing=response_model_timing,
            )
        elif gateway is None:
            self.broadcast_sync(
                msg,
                response_model_timing=response_model_timing,
            )

    def send_webrtc_error(self, error: str, gateway: Optional[Any] = None) -> None:
        """Send WebRTC error to the owning client (fallback: broadcast)."""
        if self._webrtc_shutdown:
            return
        response_model_started_ns = time.perf_counter_ns()
        msg = {"type": "webrtc_error", "error": error}
        response_model_timing = self.response_model_timing_since(
            response_model_started_ns
        )
        owner = self._get_gateway_owner(gateway) if gateway is not None else self._get_webrtc_owner()
        owner_ip = self._webrtc_gateway_owner_ip.get(gateway) if gateway is not None else self._webrtc_owner_ip
        if owner is not None:
            self.send_to_client_sync(
                owner,
                msg,
                response_model_timing=response_model_timing,
            )
            self.logger.warning("<<< WebRTC error (owner=%s): %s", owner_ip or "unknown", error)
        else:
            if gateway is not None:
                self.logger.warning("<<< WebRTC error dropped (no owner): %s", error)
            else:
                self.broadcast_sync(
                    msg,
                    response_model_timing=response_model_timing,
                )
                self.logger.warning("<<< Broadcast WebRTC error: %s", error)

    async def _periodic_menon_telemetry_log(self, interval_seconds: float = 1.0) -> None:
        """Emit a concise 1 Hz INFO log with latest Menon calibration/coordinate RX/TX."""
        keys_rx = ['set_align', 'set_extrinsics', 'solve_pnp', 'pixel_to_world']
        keys_tx = ['calibration-bundle', 'set_align_result', 'set_extrinsics_result', 'solve_pnp_result', 'pixel_to_world_response']
        while self.running:
            try:
                parts = []
                # RX summary
                rx_items = []
                for k in keys_rx:
                    entry = self._telemetry['rx'].get(k)
                    if entry:
                        rx_items.append(f"{k}:{entry['data']}")
                parts.append(f"rx=[{'; '.join(rx_items) if rx_items else '-'}]")
                # TX summary
                tx_items = []
                for k in keys_tx:
                    entry = self._telemetry['tx'].get(k)
                    if entry:
                        tx_items.append(f"{k}:{entry['data']}")
                parts.append(f"tx=[{'; '.join(tx_items) if tx_items else '-'}]")
                env_flag = str(os.environ.get("NOESIS_MENON_IO_DEBUG", "")).strip().lower()
                if env_flag in ("1", "true", "yes", "on"):
                    self.logger.debug(f"MENON I/O | {' | '.join(parts)}")
            except asyncio.CancelledError:
                # Properly handle cancellation
                break
            except Exception:
                # Never fail loop due to logging issues
                pass
            finally:
                # Always yield to avoid event loop starvation
                try:
                    await asyncio.sleep(min(interval_seconds, 1.0))
                except asyncio.CancelledError:
                    break
    
    async def _cleanup_stale_connections(self):
        """Periodically clean up any stale or closed connections"""
        try:
            # Since the broadcast method already handles cleanup when connections fail,
            # we'll make this cleanup much simpler and safer
            stale_clients = set()

            for client in self.connected_clients:
                try:
                    # Check various attributes that might indicate a closed connection
                    is_closed = False

                    # Try the most common attributes first
                    if hasattr(client, 'closed') and getattr(client, 'closed', False):
                        is_closed = True
                    elif hasattr(client, 'close_code') and getattr(client, 'close_code', None) is not None:
                        is_closed = True
                    elif hasattr(client, 'state') and getattr(client, 'state', None) == 'CLOSED':
                        is_closed = True

                    if is_closed:
                        stale_clients.add(client)

                except Exception:
                    # If we can't safely inspect the client, assume it's stale
                    stale_clients.add(client)

            if stale_clients:
                for client in stale_clients:
                    if client in self.connected_clients:
                        self.connected_clients.remove(client)
                        client_ip = getattr(client, 'remote_address', 'Unknown') if hasattr(client, 'remote_address') else "Unknown"
                        self.logger.info(f"Cleaned up stale connection for client {client_ip}")
                self.logger.debug(f"Cleaned up {len(stale_clients)} stale connections")

        except Exception as e:
            self.logger.error(f"Error during connection cleanup: {e}")

    async def _periodic_stats_broadcast(self, interval_seconds: float = 1.0):
        """Periodically fetches and broadcasts stats."""
        self.logger.info(f"Starting periodic stats broadcast every {interval_seconds} seconds.")
        cleanup_counter = 0

        while self.running:
            try:
                # Periodic cleanup of stale connections (every 60 seconds, reduced frequency)
                cleanup_counter += 1
                if cleanup_counter >= 60:
                    await self._cleanup_stale_connections()
                    cleanup_counter = 0

                if self.stats_callback and self.connected_clients: # Only send if callback exists and clients are connected
                    self.logger.debug(f"📊 Broadcasting stats to {len(self.connected_clients)} clients")
                    stats_payload = await self._get_stats_snapshot()
                    if stats_payload: # Ensure callback returned something
                        envelope_started_ns = time.perf_counter_ns()
                        stats_message = {
                            'type': 'stats',
                            'payload': stats_payload
                        }
                        source_timing = getattr(
                            stats_payload,
                            "boundary_response_model_timing",
                            None,
                        )
                        response_model_timing = BoundaryResponseModelTiming(
                            self._response_model_ms(source_timing)
                            + self.response_model_timing_since(
                                envelope_started_ns
                            ).duration_ms
                        )
                        await self.broadcast(
                            stats_message,
                            response_model_timing=response_model_timing,
                        )
                        self.logger.debug("✅ Stats broadcast completed")

                        # Periodic INFO log (every ~5s) to surface telemetry presence
                        now = time.time()
                        if now - self._last_stats_info_log >= 5.0:
                            if self.logger.isEnabledFor(logging.DEBUG):
                                cam_count = len((stats_payload or {}).get('cameras', {}))
                                uptime = (stats_payload or {}).get('uptime', 0)
                                self.logger.debug(
                                    "Stats sent: clients=%d cameras=%d uptime_s=%.1f",
                                    len(self.connected_clients),
                                    cam_count,
                                    uptime,
                                )
                            self._last_stats_info_log = now
                    else:
                        self.logger.debug("Stats callback returned empty payload, skipping broadcast.")
                elif not self.connected_clients:
                    #self.logger.debug("No clients connected, skipping stats broadcast.")
                    pass

                # Wait for the next interval - use shorter intervals to allow for cancellation
                sleep_time = 5.0 if not self.connected_clients else interval_seconds
                await asyncio.sleep(min(sleep_time, 1.0))  # Cap at 1 second for responsiveness

            except asyncio.CancelledError:
                self.logger.info("Periodic stats broadcast task cancelled.")
                break # Exit loop if task is cancelled
            except StatsAdmissionClosed:
                self.logger.info("Periodic stats collection admission closed")
                break
            except Exception as e:
                self.logger.error(f"Error during periodic stats broadcast: {e}")
                # Avoid tight loop on persistent error
                await asyncio.sleep(interval_seconds) 

    async def start(self):
        """Start the WebSocket server and the periodic stats broadcast."""
        try:
            # Capture the running loop for cross-thread callbacks
            try:
                self.event_loop = asyncio.get_running_loop()
            except RuntimeError:
                self.event_loop = None

            # Remove one-time thread creation from the measured first large
            # response and isolate serialization from shared provider workers.
            await self._prewarm_serializer_executor()

            # Start the server (legacy API for compatibility)
            self.server = await serve(
                self.handle_client,
                self.host,
                self.port,
                process_request=self._process_request,
                max_size=None,       # allow large binary frames
                max_queue=1,         # minimize buffering to reduce latency
                ping_interval=60,    # keep-alive pings every 60 seconds (was 20)
                ping_timeout=30,     # wait 30 seconds for pong response (was 20)
                close_timeout=self.CLOSE_HANDSHAKE_TIMEOUT_S,
            )

            # If bound to port 0 (ephemeral), capture the actual port
            try:
                if self.port in (0, None) and self.server and getattr(self.server, 'sockets', None):
                    sock = self.server.sockets[0]
                    bound = sock.getsockname()
                    if isinstance(bound, tuple) and len(bound) >= 2:
                        self.port = int(bound[1])
            except Exception:
                pass

            # Start periodic stats broadcast task if callback is provided
            if self.stats_callback:
                self._stats_task = asyncio.create_task(
                    self._periodic_stats_broadcast(),
                    name="PeriodicStatsBroadcast"
                )

            # Always start Menon I/O telemetry logger (1 Hz)
            self._telemetry_task = asyncio.create_task(
                self._periodic_menon_telemetry_log(),
                name="MenonTelemetryLogger"
            )

            self.logger.info(f"WebSocket server running on {self.host}:{self.port}")
            print(f"🚀 WebSocket server is LIVE on {self.host}:{self.port}")
            print(f"📡 Server listening for connections on ws://{self.host}:{self.port}")
            print(f"🔄 Periodic stats broadcast: {'ENABLED' if self.stats_callback else 'DISABLED'}")
            print("💓 Keep-alive: ping every 60s, timeout 30s")
            self.logger.info("WebSocket server startup completed successfully")

        except OSError as e:
            self._shutdown_serializer_executor()
            self.logger.error(f"Failed to start server (Port {self.port} likely in use): {e}")
            raise
        except Exception as e:
            self._shutdown_serializer_executor()
            self.logger.error(f"Server failed: {e}")
            raise

    async def _process_request(self, _path: str, request_headers: Any) -> Any:
        if self._internal_auth_mode == "disabled":
            return None
        authorization = ""
        try:
            authorization = str(request_headers.get("Authorization", ""))
        except Exception:
            authorization = ""
        prefix = "Bearer "
        supplied = authorization[len(prefix) :].strip() if authorization.startswith(prefix) else ""
        expected = str(self._internal_auth_token or "")
        if supplied and expected and hmac.compare_digest(supplied, expected):
            return None
        body = b'{"error":"internal_auth_required","message":"Authenticated appliance gateway required."}'
        return (
            HTTPStatus.UNAUTHORIZED,
            [
                ("Content-Type", "application/json"),
                ("Cache-Control", "no-store"),
                ("Content-Length", str(len(body))),
            ],
            body,
        )

    async def stop(self):
        """Gracefully stop the WebSocket server and the periodic stats broadcast."""
        self._shutdown_quiesced = False
        self.logger.info("Stopping WebSocket server...")
        self.running = False
        outbound_receipt = await self.quiesce_outbound_submissions(
            timeout_s=self.OUTBOUND_SUBMISSION_DRAIN_TIMEOUT_S
        )
        self.logger.info(
            "Cross-thread WebSocket submissions quiesced: admitted=%d completed=%d failed=%d aborted=%d bytes=%d",
            outbound_receipt.admitted_submissions,
            outbound_receipt.completed_submissions,
            outbound_receipt.failed_submissions,
            outbound_receipt.aborted_submissions,
            outbound_receipt.completed_bytes,
        )
        stats_receipt = self.quiesce_stats_collector(
            timeout_s=self.STATS_COLLECTOR_DRAIN_TIMEOUT_S
        )
        self.logger.info(
            "WebSocket stats collector quiesced: admitted=%d completed=%d",
            stats_receipt.admitted_snapshots,
            stats_receipt.completed_snapshots,
        )
        detached_gateways = self.begin_webrtc_shutdown(timeout_s=5.0)
        for gateway in detached_gateways:
            stop_gateway = getattr(gateway, "stop", None)
            if not callable(stop_gateway):
                raise RuntimeError("registered WebRTC gateway has no stop method")
            stop_gateway()
        # Runtime shutdown drains providers before it reaches the listener.  A
        # direct stop must still fail closed rather than cancel asyncio handlers
        # while their executor threads continue touching native/store state.
        provider_receipt = self.quiesce_blocking_providers(timeout_s=0.0)
        self.logger.info(
            "Blocking WebSocket providers quiesced: admitted=%d completed=%d",
            provider_receipt.admitted_calls,
            provider_receipt.completed_calls,
        )

        # Immediately cancel all background tasks to prevent them from continuing
        tasks_to_cancel = []

        # Cancel any pending binary flush task
        if self._binary_flush_task and not self._binary_flush_task.done():
            self._binary_flush_task.cancel()
            tasks_to_cancel.append(self._binary_flush_task)

        if self._json_flush_task and not self._json_flush_task.done():
            self._json_flush_task.cancel()
            tasks_to_cancel.append(self._json_flush_task)

        # Cancel the periodic stats task
        if self._stats_task and not self._stats_task.done():
            self.logger.info("Cancelling periodic stats broadcast task...")
            self._stats_task.cancel()
            tasks_to_cancel.append(self._stats_task)

        # Cancel Menon telemetry task
        if self._telemetry_task and not self._telemetry_task.done():
            self.logger.info("Cancelling Menon telemetry task...")
            self._telemetry_task.cancel()
            tasks_to_cancel.append(self._telemetry_task)

        # Cancel server task if it exists
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()
            tasks_to_cancel.append(self.server_task)

        # Wait for all tasks to cancel
        if tasks_to_cancel:
            try:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                self.logger.info(f"Successfully cancelled {len(tasks_to_cancel)} background tasks")
            except Exception as e:
                self.logger.warning(f"Error waiting for task cancellation: {e}")

        # Close admission first, then run server-owned handler drain and the
        # explicit client handshakes concurrently under one total deadline. A
        # handler admitted at the close/snapshot boundary may never appear in
        # connected_clients, so wait_closed() is the authoritative proof.
        listener_deadline = time.monotonic() + self.LISTENER_SHUTDOWN_TIMEOUT_S
        listener_wait_task: Optional[asyncio.Task[Any]] = None
        try:
            if self.server:
                self.server.close()
                listener_wait_task = asyncio.create_task(
                    self.server.wait_closed(),
                    name="WebSocketServerHandlerDrain",
                )

            clients = list(self.connected_clients)
            if clients:
                close_tasks = [
                    asyncio.create_task(client.close(), name="WebSocketClientClose")
                    for client in clients
                ]
                first_wait_s = min(
                    self.CLIENT_CLOSE_GRACE_S,
                    max(0.0, listener_deadline - time.monotonic()),
                )
                _, pending = await asyncio.wait(close_tasks, timeout=first_wait_s)
                if pending:
                    for client, close_task in zip(clients, close_tasks):
                        if close_task not in pending:
                            continue
                        transport = getattr(client, "transport", None)
                        abort = getattr(transport, "abort", None)
                        fail_connection = getattr(client, "fail_connection", None)
                        if callable(abort):
                            abort()
                        elif callable(fail_connection):
                            fail_connection()
                    abort_wait_s = min(
                        self.CLIENT_ABORT_DRAIN_S,
                        max(0.0, listener_deadline - time.monotonic()),
                    )
                    _, pending = await asyncio.wait(pending, timeout=abort_wait_s)
                if pending:
                    for close_task in pending:
                        close_task.cancel()
                    await asyncio.gather(*close_tasks, return_exceptions=True)
                    raise RuntimeError(
                        "WebSocket client close handlers remained active after transport abort"
                    )
                await asyncio.gather(*close_tasks, return_exceptions=True)
                self.connected_clients.clear()
                self.logger.info("Closed %d client connections", len(close_tasks))

            if listener_wait_task is not None:
                remaining = listener_deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(
                        "WebSocket listener exceeded the total shutdown deadline"
                    )
                await asyncio.wait_for(
                    asyncio.shield(listener_wait_task),
                    timeout=remaining,
                )
                self.server = None
                self.logger.info("WebSocket server stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping WebSocket server: {e}")
            raise
        finally:
            if listener_wait_task is not None and not listener_wait_task.done():
                listener_wait_task.cancel()
                await asyncio.gather(listener_wait_task, return_exceptions=True)
            self._shutdown_serializer_executor()
        self._shutdown_quiesced = True

    def report_lifecycle_failure(self, error: BaseException) -> None:
        callback = self.lifecycle_failure_callback
        if callable(callback):
            callback(error)

    async def handle_client(self, websocket, path=None):
        """Handle incoming WebSocket connections and messages."""
        request_path = path if isinstance(path, str) else getattr(websocket, "path", None)
        if request_path == self.HEALTH_PATH:
            # Appliance liveness proves the authenticated WebSocket handler can
            # exchange a frame without registering as a telemetry/WebRTC client
            # or materializing expensive initial UI snapshots.
            try:
                response_model_started_ns = time.perf_counter_ns()
                payload = (
                    self.health_payload_getter()
                    if self.health_payload_getter is not None
                    else None
                )
            except Exception as exc:
                self.logger.warning("Appliance WebSocket health is not ready: %s", exc)
                await websocket.close(code=1013, reason="health_not_ready")
                return
            if payload is None:
                await websocket.send(self.HEALTH_PAYLOAD)
            elif isinstance(payload, dict):
                await self._send_json_with_boundary_metrics(
                    websocket,
                    payload,
                    route="health",
                    message_type="health",
                    response_model_timing=self.response_model_timing_since(
                        response_model_started_ns
                    ),
                )
            else:
                raise RuntimeError("dynamic WebSocket health payload must be a dict")
            await websocket.close(code=1000, reason="health_complete")
            return
        try:
            client_ip = websocket.remote_address[0] if hasattr(websocket, 'remote_address') else "Unknown"
        except Exception as exc:
            self.logger.error(f"handle_client: unable to read remote address: {exc}")
            client_ip = "Unknown"
        if len(self.connected_clients) >= int(self._max_telemetry_clients):
            self._telemetry_client_rejections += 1
            self.logger.warning(
                "Rejecting telemetry client %s at bounded capacity %d",
                client_ip,
                self._max_telemetry_clients,
            )
            try:
                await websocket.close(
                    code=self.TELEMETRY_CAPACITY_CLOSE_CODE,
                    reason=self.TELEMETRY_CAPACITY_CLOSE_REASON,
                )
            except Exception as exc:
                self.logger.debug(
                    "Telemetry capacity close failed for %s: %s",
                    client_ip,
                    exc,
                )
            return
        self.connected_clients.add(websocket)
        self._telemetry_client_peak = max(
            self._telemetry_client_peak,
            len(self.connected_clients),
        )
        self.logger.info(f"Client {client_ip} connected. Total clients: {len(self.connected_clients)}")
        print(f"✅ Client {client_ip} connected! Total clients: {len(self.connected_clients)}")

        try:
            # Send initial trail visualization state to new client
            try:
                response_model_started_ns = time.perf_counter_ns()
                initial_trail_message = {
                    'type': 'trail_visualization_enabled_update',
                    'enabled': self.initial_trail_state
                }
                await self._send_json_with_boundary_metrics(
                    websocket,
                    initial_trail_message,
                    route="initial_trail_state",
                    message_type="trail_visualization_enabled_update",
                    response_model_timing=self.response_model_timing_since(
                        response_model_started_ns
                    ),
                )
                self.logger.info(f"Sent initial trail visualization state ({self.initial_trail_state}) to {client_ip}")
            except Exception as e:
                self.logger.warning(f"Could not send initial trail visualization state to {client_ip}: {e}")

            # Send initial trail settings so UI trail tuning can mirror backend config.
            try:
                trail_settings = self._resolve_trail_settings()
                response_model_started_ns = time.perf_counter_ns()
                initial_trail_settings = {
                    'type': 'trail_settings_update',
                    'config': trail_settings,
                }
                await self._send_json_with_boundary_metrics(
                    websocket,
                    initial_trail_settings,
                    route="initial_trail_settings",
                    message_type="trail_settings_update",
                    response_model_timing=self.response_model_timing_since(
                        response_model_started_ns
                    ),
                )
                self.logger.info(f"Sent initial trail settings to {client_ip}")
            except Exception as e:
                self.logger.warning(f"Could not send initial trail settings to {client_ip}: {e}")

            # Send initial calibration bundle if available
            try:
                if callable(self.calibration_getter):
                    bundle = self.calibration_getter() or {}
                    if bundle:
                        response_model_started_ns = time.perf_counter_ns()
                        msg = {'type': 'calibration-bundle', 'data': bundle}
                        await self._send_json_with_boundary_metrics(
                            websocket,
                            msg,
                            route="initial_calibration",
                            message_type="calibration-bundle",
                            response_model_timing=self.response_model_timing_since(
                                response_model_started_ns
                            ),
                        )
                        try:
                            self._record_tx(msg)
                        except Exception:
                            pass
                        self.logger.info(f"Sent calibration-bundle to {client_ip}")
            except Exception as e:
                self.logger.warning(f"Could not send calibration-bundle to {client_ip}: {e}")

            # Send an immediate stats snapshot on connect for quicker UI readiness
            try:
                if self.stats_callback:
                    snap = (await self._get_stats_snapshot()) or {}
                    if snap:
                        envelope_started_ns = time.perf_counter_ns()
                        stats_message = {'type': 'stats', 'payload': snap}
                        source_timing = getattr(
                            snap,
                            "boundary_response_model_timing",
                            None,
                        )
                        response_model_timing = BoundaryResponseModelTiming(
                            self._response_model_ms(source_timing)
                            + self.response_model_timing_since(
                                envelope_started_ns
                            ).duration_ms
                        )
                        await self._send_json_with_boundary_metrics(
                            websocket,
                            stats_message,
                            route="initial_stats",
                            message_type="stats",
                            response_model_timing=response_model_timing,
                        )
                        self.logger.info(f"Sent initial stats snapshot to {client_ip}")
            except Exception as e:
                self.logger.debug(f"Could not send initial stats snapshot to {client_ip}: {e}")

            # Process messages from client
            async for message in websocket:
                try:
                    data = json.loads(message)
                    # Record RX telemetry for Menon RPCs
                    try:
                        self._record_rx(data)
                    except Exception:
                        pass

                    # Handle clear_stats command
                    if data.get('type') == 'clear_stats':
                        self.logger.info(f"Received clear_stats request from {client_ip}")
                        # Call clear stats function if registered
                        if self.stats_callback and hasattr(self.stats_callback, 'clear_stats'):
                            self.logger.info("Stats cleared. Broadcasting updated stats...")
                            # Broadcast updated stats
                            if self.stats_callback:
                                stats_payload = await self._clear_and_refresh_stats()
                                envelope_started_ns = time.perf_counter_ns()
                                stats_message = {
                                    'type': 'stats',
                                    'payload': stats_payload
                                }
                                source_timing = getattr(
                                    stats_payload,
                                    "boundary_response_model_timing",
                                    None,
                                )
                                await self.broadcast(
                                    stats_message,
                                    response_model_timing=BoundaryResponseModelTiming(
                                        self._response_model_ms(source_timing)
                                        + self.response_model_timing_since(
                                            envelope_started_ns
                                        ).duration_ms
                                    ),
                                )
                        else:
                            self.logger.warning("Stats callback not available to clear stats.")

                    # Handle visualization toggle command
                    elif data.get('type') == 'set_vis_toggle':
                        toggle_name = data.get('toggle_name')
                        enabled = data.get('enabled')

                        if toggle_name is not None and isinstance(enabled, bool):
                            self.logger.info(f"Received set_vis_toggle from {client_ip}: {toggle_name} = {enabled}")

                            # Call toggle callback if available
                            if self.toggle_callback:
                                try:
                                    self.toggle_callback(toggle_name, enabled)
                                except Exception as e:
                                    self.logger.error(f"Error in toggle callback: {e}")

                            # Broadcast to all clients including the sender
                            response_model_started_ns = time.perf_counter_ns()
                            broadcast_message = {
                                'type': 'toggle_update',
                                'toggle_name': toggle_name,
                                'enabled': enabled
                            }
                            await self.broadcast(
                                broadcast_message,
                                response_model_timing=self.response_model_timing_since(
                                    response_model_started_ns
                                ),
                            )
                            if toggle_name == 'trail_visualization_enabled':
                                try:
                                    response_model_started_ns = time.perf_counter_ns()
                                    trail_settings_message = {
                                        'type': 'trail_settings_update',
                                        'config': self._resolve_trail_settings(),
                                    }
                                    await self.broadcast(
                                        trail_settings_message,
                                        response_model_timing=self.response_model_timing_since(
                                            response_model_started_ns
                                        ),
                                    )
                                except Exception as e:
                                    self.logger.debug("Failed to broadcast trail settings update: %s", e)
                        else:
                            self.logger.warning(f"Invalid set_vis_toggle message from {client_ip}: {data}")

                    # Handle client heartbeat ping messages
                    elif data.get('type') == 'ping':
                        timestamp = data.get('timestamp')
                        self.logger.debug(f"Received ping from {client_ip} (timestamp: {timestamp})")
                        # Send pong response
                        try:
                            response_model_started_ns = time.perf_counter_ns()
                            pong_message = {
                                'type': 'pong',
                                'timestamp': timestamp
                            }
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                pong_message,
                                route="pong",
                                message_type="pong",
                                response_model_timing=self.response_model_timing_since(
                                    response_model_started_ns
                                ),
                            )
                        except Exception as e:
                            self.logger.warning(f"Failed to send pong response to {client_ip}: {e}")

                    # Handle BEV config overrides
                    elif data.get('type') == 'bev-config':
                        cam_id = data.get('cameraId') or data.get('camId')
                        cfg = data.get('config') or {}
                        if isinstance(cam_id, str) and isinstance(cfg, dict):
                            self.logger.info("Received BEV config for %s: %s", cam_id, cfg)
                            if callable(self.bev_config_callback):
                                try:
                                    self.bev_config_callback(cam_id, cfg)
                                except Exception as exc:
                                    self.logger.error("BEV config callback failed: %s", exc)
                            response_model_started_ns = time.perf_counter_ns()
                            ack = {
                                'type': 'bev-config-ack',
                                'cameraId': cam_id,
                                'config': cfg,
                            }
                            await self.broadcast(
                                ack,
                                response_model_timing=self.response_model_timing_since(
                                    response_model_started_ns
                                ),
                            )
                        else:
                            self.logger.warning("Invalid BEV config payload from %s: %s", client_ip, data)

                    elif data.get('type') == 'bev-overlay':
                        cam_id = data.get('cameraId') or data.get('camId')
                        enabled = data.get('enabled')
                        if isinstance(cam_id, str) and isinstance(enabled, bool):
                            self.logger.info("Received BEV overlay toggle for %s: %s", cam_id, enabled)
                            if callable(self.bev_overlay_callback):
                                try:
                                    self.bev_overlay_callback(cam_id, enabled)
                                except Exception as exc:
                                    self.logger.error("BEV overlay callback failed: %s", exc)
                            response_model_started_ns = time.perf_counter_ns()
                            overlay_message = {
                                'type': 'bev-overlay-update',
                                'cameraId': cam_id,
                                'enabled': enabled,
                            }
                            await self.broadcast(
                                overlay_message,
                                response_model_timing=self.response_model_timing_since(
                                    response_model_started_ns
                                ),
                            )
                        else:
                            self.logger.warning("Invalid BEV overlay message from %s: %s", client_ip, data)

                    elif data.get('type') == 'auto_calibrate_pose':
                        cam_id = data.get('camera') or data.get('cameraId') or data.get('camId')
                        response_fields: Dict[str, Any] = {}
                        if callable(self.auto_calibrate_handler):
                            try:
                                out = await asyncio.wait_for(
                                    self._run_blocking_provider(
                                        self.auto_calibrate_handler,
                                        cam_id if isinstance(cam_id, str) else None,
                                    ),
                                    timeout=30.0
                                )
                                if isinstance(out, dict):
                                    response_fields = out
                            except asyncio.TimeoutError:
                                response_fields = {'ok': False, 'error': 'timeout'}
                            except ProviderAdmissionClosed:
                                response_fields = {'ok': False, 'error': 'shutting_down'}
                            except ProviderCapacityExceeded:
                                response_fields = {
                                    'ok': False,
                                    'error': 'provider_capacity_exceeded',
                                }
                            except Exception as exc:
                                response_fields = {'ok': False, 'error': str(exc)}
                        else:
                            response_fields = {'ok': False, 'error': 'no_handler'}
                        response_model_started_ns = time.perf_counter_ns()
                        result = {
                            'type': 'auto_calibrate_result',
                            **response_fields,
                        }
                        try:
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="auto_calibrate_pose",
                                message_type="auto_calibrate_result",
                                response_model_timing=self.response_model_timing_since(
                                    response_model_started_ns
                                ),
                            )
                        except Exception:
                            pass

                    # ---- Spatial & calibration RPCs ----
                    # legacy 'get_transformation' removed; calibration-bundle is source of truth

                    elif data.get('type') == 'pixel_to_world':
                        req = data
                        req_id = req.get('request_id') or req.get('reqId')
                        try:
                            if callable(self.pixel_to_world_handler):
                                response_fields = await asyncio.wait_for(
                                    self._run_blocking_provider(
                                        self.pixel_to_world_handler,
                                        req,
                                    ),
                                    timeout=self._calibration_rpc_timeout,
                                ) or {}
                            else:
                                response_fields = {'ok': False, 'error': 'no_handler'}
                        except asyncio.TimeoutError:
                            response_fields = {'ok': False, 'error': 'timeout'}
                        except ProviderAdmissionClosed:
                            response_fields = {'ok': False, 'error': 'shutting_down'}
                        except ProviderCapacityExceeded:
                            response_fields = {
                                'ok': False,
                                'error': 'provider_capacity_exceeded',
                            }
                        except Exception as e:
                            response_fields = {'ok': False, 'error': str(e)}

                        response_model_started_ns = time.perf_counter_ns()
                        result = {'type': 'pixel_to_world_response'}
                        if req_id is not None:
                            result['request_id'] = req_id
                            result.setdefault('reqId', req_id)
                        result.update(response_fields)
                        world_val = result.get('world')
                        if isinstance(world_val, (list, tuple)) and len(world_val) >= 3:
                            try:
                                result['world'] = {
                                    'x': float(world_val[0]),
                                    'y': float(world_val[1]),
                                    'z': float(world_val[2]),
                                }
                            except Exception:
                                pass

                        try:
                            # Record TX telemetry and send
                            self._record_tx(result)
                        except Exception:
                            pass
                        try:
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="pixel_to_world",
                                message_type="pixel_to_world_response",
                                response_model_timing=self.response_model_timing_since(
                                    response_model_started_ns
                                ),
                            )
                        except Exception:
                            pass

                    elif data.get('type') == 'set_extrinsics':
                        req = data
                        try:
                            if callable(self.set_extrinsics_handler):
                                response_fields = await asyncio.wait_for(
                                    self._run_blocking_provider(
                                        self.set_extrinsics_handler,
                                        req,
                                    ),
                                    timeout=self._calibration_rpc_timeout,
                                ) or {}
                            else:
                                response_fields = {'ok': False, 'error': 'no_handler'}
                        except asyncio.TimeoutError:
                            response_fields = {'ok': False, 'error': 'timeout'}
                        except ProviderAdmissionClosed:
                            response_fields = {'ok': False, 'error': 'shutting_down'}
                        except ProviderCapacityExceeded:
                            response_fields = {
                                'ok': False,
                                'error': 'provider_capacity_exceeded',
                            }
                        except Exception as e:
                            response_fields = {'ok': False, 'error': str(e)}
                        response_model_started_ns = time.perf_counter_ns()
                        result = {'type': 'set_extrinsics_result', **response_fields}
                        try:
                            try:
                                self._record_tx(result)
                            except Exception:
                                pass
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="set_extrinsics",
                                message_type="set_extrinsics_result",
                                response_model_timing=self.response_model_timing_since(
                                    response_model_started_ns
                                ),
                            )
                        except Exception:
                            pass

                    elif data.get('type') == 'solve_pnp':
                        req = data
                        try:
                            if callable(self.solve_pnp_handler):
                                response_fields = await asyncio.wait_for(
                                    self._run_blocking_provider(
                                        self.solve_pnp_handler,
                                        req,
                                    ),
                                    timeout=self._calibration_rpc_timeout,
                                ) or {}
                            else:
                                response_fields = {'ok': False, 'error': 'no_handler'}
                        except asyncio.TimeoutError:
                            response_fields = {'ok': False, 'error': 'timeout'}
                        except ProviderAdmissionClosed:
                            response_fields = {'ok': False, 'error': 'shutting_down'}
                        except ProviderCapacityExceeded:
                            response_fields = {
                                'ok': False,
                                'error': 'provider_capacity_exceeded',
                            }
                        except Exception as e:
                            response_fields = {'ok': False, 'error': str(e)}
                        response_model_started_ns = time.perf_counter_ns()
                        result = {'type': 'solve_pnp_result', **response_fields}
                        try:
                            try:
                                self._record_tx(result)
                            except Exception:
                                pass
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="solve_pnp",
                                message_type="solve_pnp_result",
                                response_model_timing=self.response_model_timing_since(
                                    response_model_started_ns
                                ),
                            )
                        except Exception:
                            pass

                    elif data.get('type') == 'set_align':
                        req = data
                        try:
                            if callable(self.set_align_handler):
                                response_fields = await asyncio.wait_for(
                                    self._run_blocking_provider(
                                        self.set_align_handler,
                                        req,
                                    ),
                                    timeout=self._calibration_rpc_timeout,
                                ) or {}
                            else:
                                response_fields = {'ok': False, 'error': 'no_handler'}
                        except asyncio.TimeoutError:
                            response_fields = {'ok': False, 'error': 'timeout'}
                        except ProviderAdmissionClosed:
                            response_fields = {'ok': False, 'error': 'shutting_down'}
                        except ProviderCapacityExceeded:
                            response_fields = {
                                'ok': False,
                                'error': 'provider_capacity_exceeded',
                            }
                        except Exception as e:
                            response_fields = {'ok': False, 'error': str(e)}
                        response_model_started_ns = time.perf_counter_ns()
                        result = {'type': 'set_align_result', **response_fields}
                        try:
                            try:
                                self._record_tx(result)
                            except Exception:
                                pass
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="set_align",
                                message_type="set_align_result",
                                response_model_timing=self.response_model_timing_since(
                                    response_model_started_ns
                                ),
                            )
                        except Exception:
                            pass

                    elif data.get('type') in ('get_ma_depth', 'get_ma_depth_cache'):
                        request_id = data.get('request_id')
                        try:
                            request_id_bytes = (
                                request_id.encode('utf-8')
                                if isinstance(request_id, str)
                                else b''
                            )
                        except UnicodeEncodeError:
                            request_id_bytes = b''
                        if not (
                            isinstance(request_id, str)
                            and bool(request_id.strip())
                            and 0
                            < len(request_id_bytes)
                            <= self.DEPTH_REQUEST_ID_MAX_BYTES
                        ):
                            self.logger.warning(
                                "Closing depth RPC client %s for an invalid request_id",
                                client_ip,
                            )
                            try:
                                await websocket.close(
                                    code=self.DEPTH_REQUEST_ID_POLICY_CLOSE_CODE,
                                    reason=self.DEPTH_REQUEST_ID_POLICY_CLOSE_REASON,
                                )
                            except Exception as exc:
                                self.logger.debug(
                                    "Depth RPC policy close failed for %s: %s",
                                    client_ip,
                                    exc,
                                )
                            break
                        camera = (
                            data.get('camera')
                            or data.get('cameraId')
                            or data.get('camera_id')
                            or data.get('camId')
                        )
                        ts_max_us = (
                            data.get('ts_max_us')
                            or data.get('tsMaxUs')
                            or data.get('ts_maxUS')
                            or data.get('tsMax_us')
                        )
                        if ts_max_us is None:
                            ts_max_us = data.get('ts_max') or data.get('tsMax')
                        cache_only = bool(data.get('cache_only', data.get('cacheOnly', False)))
                        if data.get('type') == 'get_ma_depth_cache':
                            cache_only = True

                        provider = self.ma_depth_provider if callable(self.ma_depth_provider) else None
                        if not camera or provider is None:
                            response_model_started_ns = time.perf_counter_ns()
                            result = {
                                'type': 'ma_depth_response',
                                'camera': camera,
                                'request_id': request_id,
                                'cache_only': cache_only,
                                'served_from_cache': False,
                                'ts_us': 0,
                                'error': 'no_provider',
                                'ok': False,
                            }
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_ma_depth",
                                message_type="ma_depth_response",
                                response_model_timing=self.response_model_timing_since(
                                    response_model_started_ns
                                ),
                            )
                            continue

                        mode_key = "cache" if cache_only else "fresh"
                        rate_key = f"{client_ip}:{camera}:{mode_key}"
                        now = time.time()
                        last = self._depth_rpc_tracker.get(rate_key, 0.0)
                        if now - last < self._depth_rate_limit_window:
                            self.logger.debug(f"Depth RPC throttled for {rate_key}")
                            response_model_started_ns = time.perf_counter_ns()
                            result = {
                                'type': 'ma_depth_response',
                                'camera': camera,
                                'request_id': request_id,
                                'cache_only': cache_only,
                                'served_from_cache': False,
                                'ts_us': 0,
                                'error': 'rate_limited',
                                'ok': False,
                            }
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_ma_depth",
                                message_type="ma_depth_response",
                                response_model_timing=self.response_model_timing_since(
                                    response_model_started_ns
                                ),
                            )
                            continue
                        self._depth_rpc_tracker[rate_key] = now
                        if len(self._depth_rpc_tracker) > 256:
                            self._depth_rpc_tracker = {
                                k: v for k, v in self._depth_rpc_tracker.items() if now - v <= self._tracker_prune_window
                            }

                        try:
                            provider_kwargs: Dict[str, Any] = {}
                            try:
                                signature = inspect.signature(provider)
                                if (
                                    'cache_only' in signature.parameters
                                    or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
                                ):
                                    provider_kwargs['cache_only'] = cache_only
                            except Exception:
                                provider_kwargs = {}
                            provider_start_ns = time.perf_counter_ns()
                            payload = await asyncio.wait_for(
                                self._run_blocking_provider(
                                    provider,
                                    camera,
                                    ts_max_us,
                                    request_id,
                                    **provider_kwargs,
                                ),
                                timeout=self._depth_rpc_timeout,
                            )
                            provider_ms = (time.perf_counter_ns() - provider_start_ns) / 1_000_000.0
                            self._record_boundary_serialization_stage(
                                provider_ms,
                                0,
                                channel="ws",
                                route="get_ma_depth",
                                message_type="ma_depth_response",
                                stage="provider_wait",
                                outcome="ok",
                                include_budget=False,
                            )
                            response_model_started_ns = time.perf_counter_ns()
                            if payload:
                                if 'type' not in payload:
                                    payload['type'] = 'ma_depth_response'
                                payload.setdefault('camera', camera)
                                payload['request_id'] = request_id
                                payload.setdefault('cache_only', cache_only)
                                payload.setdefault('served_from_cache', False)
                                payload.setdefault('ts_us', 0)
                                payload.setdefault('ok', 'error' not in payload)
                                await self._send_json_with_boundary_metrics(
                                    websocket,
                                    payload,
                                    route="get_ma_depth",
                                    message_type="ma_depth_response",
                                    use_to_thread_json=True,
                                    response_model_timing=self.response_model_timing_since(
                                        response_model_started_ns
                                    ),
                                )
                            else:
                                result = {
                                    'type': 'ma_depth_response',
                                    'camera': camera,
                                    'request_id': request_id,
                                    'cache_only': cache_only,
                                    'served_from_cache': False,
                                    'ts_us': 0,
                                    'error': 'not_available',
                                    'ok': False,
                                }
                                await self._send_json_with_boundary_metrics(
                                    websocket,
                                    result,
                                    route="get_ma_depth",
                                    message_type="ma_depth_response",
                                    response_model_timing=self.response_model_timing_since(
                                        response_model_started_ns
                                    ),
                                )
                        except ProviderAdmissionClosed:
                            response_model_started_ns = time.perf_counter_ns()
                            result = {
                                'type': 'ma_depth_response',
                                'camera': camera,
                                'request_id': request_id,
                                'cache_only': cache_only,
                                'served_from_cache': False,
                                'ts_us': 0,
                                'error': 'shutting_down',
                                'ok': False,
                            }
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_ma_depth",
                                message_type="ma_depth_response",
                                response_model_timing=self.response_model_timing_since(
                                    response_model_started_ns
                                ),
                            )
                        except ProviderCapacityExceeded:
                            response_model_started_ns = time.perf_counter_ns()
                            result = {
                                'type': 'ma_depth_response',
                                'camera': camera,
                                'request_id': request_id,
                                'cache_only': cache_only,
                                'served_from_cache': False,
                                'ts_us': 0,
                                'error': 'provider_capacity_exceeded',
                                'ok': False,
                            }
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_ma_depth",
                                message_type="ma_depth_response",
                                response_model_timing=self.response_model_timing_since(
                                    response_model_started_ns
                                ),
                            )
                        except asyncio.TimeoutError:
                            self.logger.warning(f"Depth RPC timed out for {camera} from {client_ip}")
                            response_model_started_ns = time.perf_counter_ns()
                            result = {
                                'type': 'ma_depth_response',
                                'camera': camera,
                                'request_id': request_id,
                                'cache_only': cache_only,
                                'served_from_cache': False,
                                'ts_us': 0,
                                'error': 'timeout',
                                'ok': False,
                            }
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_ma_depth",
                                message_type="ma_depth_response",
                                response_model_timing=self.response_model_timing_since(
                                    response_model_started_ns
                                ),
                            )
                        except Exception:
                            response_model_started_ns = time.perf_counter_ns()
                            result = {
                                'type': 'ma_depth_response',
                                'camera': camera,
                                'request_id': request_id,
                                'cache_only': cache_only,
                                'served_from_cache': False,
                                'ts_us': 0,
                                'error': 'provider_failed',
                                'ok': False,
                            }
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_ma_depth",
                                message_type="ma_depth_response",
                                response_model_timing=self.response_model_timing_since(
                                    response_model_started_ns
                                ),
                            )

                    elif data.get('type') == 'get_floorplan':
                        request_id = data.get('request_id') or data.get('requestId') or str(uuid.uuid4())
                        camera = data.get('camera')
                        if not camera and isinstance(data.get('cameras'), list) and data['cameras']:
                            camera = str(data['cameras'][0])
                        camera = str(camera) if camera else ''
                        max_age_sec = float(data.get('max_age_sec', data.get('maxAgeSec', 60.0)))
                        grid_res_m = float(data.get('grid_res_m', data.get('gridResM', 0.5)))
                        max_extent_m = float(data.get('max_extent_m', data.get('maxExtentM', 20.0)))
                        cache_only = bool(data.get('cache_only', data.get('cacheOnly', False)))
                        scene_prior_only = bool(
                            data.get('scene_prior_only', data.get('scenePriorOnly', False))
                        )
                        snapshot_ref = data.get('snapshot_ref', data.get('snapshotRef'))
                        snapshot_id = data.get('snapshot_id', data.get('snapshotId'))
                        snapshot_content_sha256 = data.get(
                            'snapshot_content_sha256',
                            data.get('snapshotContentSha256'),
                        )
                        exact_snapshot_requested = any(
                            value is not None
                            for value in (
                                snapshot_ref,
                                snapshot_id,
                                snapshot_content_sha256,
                            )
                        )

                        provider = getattr(self, 'floorplan_provider', None)
                        base_response_started_ns = time.perf_counter_ns()
                        result = {
                            'type': 'floorplan_response',
                            'request_id': request_id,
                            'camera_id': camera,
                            'cache_only': cache_only,
                            'scene_prior_only': scene_prior_only,
                        }
                        base_response_model_ms = self.response_model_timing_since(
                            base_response_started_ns
                        ).duration_ms
                        if not callable(provider):
                            completion_started_ns = time.perf_counter_ns()
                            result['error'] = 'no_provider'
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_floorplan",
                                message_type="floorplan_response",
                                response_model_timing=BoundaryResponseModelTiming(
                                    base_response_model_ms
                                    + self.response_model_timing_since(
                                        completion_started_ns
                                    ).duration_ms
                                ),
                            )
                            continue

                        mode_key = (
                            "scene_prior"
                            if scene_prior_only
                            else "cache"
                            if cache_only
                            else "exact"
                            if exact_snapshot_requested
                            else "fresh"
                        )
                        rate_key = f"{client_ip}:{camera or 'unknown'}:{mode_key}"
                        now = time.time()
                        last = self._floorplan_rpc_tracker.get(rate_key, 0.0)
                        if now - last < self._floorplan_rate_limit_window:
                            self.logger.debug(f"Floorplan RPC throttled for {rate_key}")
                            completion_started_ns = time.perf_counter_ns()
                            result.update({
                                'served_from_cache': False,
                                'error': 'rate_limited',
                                'ok': False,
                            })
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_floorplan",
                                message_type="floorplan_response",
                                response_model_timing=BoundaryResponseModelTiming(
                                    base_response_model_ms
                                    + self.response_model_timing_since(
                                        completion_started_ns
                                    ).duration_ms
                                ),
                            )
                            continue
                        self._floorplan_rpc_tracker[rate_key] = now
                        if len(self._floorplan_rpc_tracker) > 256:
                            self._floorplan_rpc_tracker = {
                                k: v for k, v in self._floorplan_rpc_tracker.items() if now - v <= self._tracker_prune_window
                            }

                        try:
                            provider_start_ns = time.perf_counter_ns()
                            provider_kwargs = {'cache_only': cache_only}
                            if scene_prior_only:
                                provider_kwargs['scene_prior_only'] = True
                            if exact_snapshot_requested:
                                provider_kwargs.update({
                                    'snapshot_ref': snapshot_ref,
                                    'snapshot_id': snapshot_id,
                                    'snapshot_content_sha256': snapshot_content_sha256,
                                })
                            payload = await asyncio.wait_for(
                                self._run_blocking_provider(
                                    provider,
                                    camera,
                                    max_age_sec,
                                    grid_res_m,
                                    max_extent_m,
                                    **provider_kwargs,
                                ),
                                timeout=self._floorplan_rpc_timeout
                            )
                            provider_ms = (time.perf_counter_ns() - provider_start_ns) / 1_000_000.0
                            self._record_boundary_serialization_stage(
                                provider_ms,
                                0,
                                channel="ws",
                                route="get_floorplan",
                                message_type="floorplan_response",
                                stage="provider_wait",
                                outcome="ok",
                                include_budget=False,
                            )
                            completion_started_ns = time.perf_counter_ns()
                            if payload:
                                result.update(payload)
                                await self._send_json_with_boundary_metrics(
                                    websocket,
                                    result,
                                    route="get_floorplan",
                                    message_type="floorplan_response",
                                    use_to_thread_json=True,
                                    response_model_timing=BoundaryResponseModelTiming(
                                        base_response_model_ms
                                        + self.response_model_timing_since(
                                            completion_started_ns
                                        ).duration_ms
                                    ),
                                )
                            else:
                                completion_started_ns = time.perf_counter_ns()
                                result['error'] = 'no_payload'
                                await self._send_json_with_boundary_metrics(
                                    websocket,
                                    result,
                                    route="get_floorplan",
                                    message_type="floorplan_response",
                                    response_model_timing=BoundaryResponseModelTiming(
                                        base_response_model_ms
                                        + self.response_model_timing_since(
                                            completion_started_ns
                                        ).duration_ms
                                    ),
                                )
                        except ProviderAdmissionClosed:
                            completion_started_ns = time.perf_counter_ns()
                            result.update({
                                'served_from_cache': False,
                                'error': 'shutting_down',
                                'ok': False,
                            })
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_floorplan",
                                message_type="floorplan_response",
                                response_model_timing=BoundaryResponseModelTiming(
                                    base_response_model_ms
                                    + self.response_model_timing_since(
                                        completion_started_ns
                                    ).duration_ms
                                ),
                            )
                        except ProviderCapacityExceeded:
                            completion_started_ns = time.perf_counter_ns()
                            result.update({
                                'served_from_cache': False,
                                'error': 'provider_capacity_exceeded',
                                'ok': False,
                            })
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_floorplan",
                                message_type="floorplan_response",
                                response_model_timing=BoundaryResponseModelTiming(
                                    base_response_model_ms
                                    + self.response_model_timing_since(
                                        completion_started_ns
                                    ).duration_ms
                                ),
                            )
                        except asyncio.TimeoutError:
                            self.logger.warning(f"Floorplan RPC timed out for {camera} from {client_ip}")
                            completion_started_ns = time.perf_counter_ns()
                            result['error'] = 'timeout'
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_floorplan",
                                message_type="floorplan_response",
                                response_model_timing=BoundaryResponseModelTiming(
                                    base_response_model_ms
                                    + self.response_model_timing_since(
                                        completion_started_ns
                                    ).duration_ms
                                ),
                            )
                        except Exception as exc:
                            completion_started_ns = time.perf_counter_ns()
                            result['error'] = str(exc)
                            self.logger.error(f"Floorplan generation error: {exc}")
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                result,
                                route="get_floorplan",
                                message_type="floorplan_response",
                                response_model_timing=BoundaryResponseModelTiming(
                                    base_response_model_ms
                                    + self.response_model_timing_since(
                                        completion_started_ns
                                    ).duration_ms
                                ),
                            )

                    # Handle WebRTC signaling: offer from browser
                    elif data.get('type') == 'webrtc_offer':
                        self.logger.info(">>> Received webrtc_offer from client %s", client_ip)
                        # The registered gateway pool is the only mosaic WebRTC path;
                        # fail closed when it is unavailable.
                        if self.webrtc_gateways or self._webrtc_gateway_factory is not None:
                            sdp = data.get('sdp', '')
                            sdp_lines = len(sdp.split('\n')) if sdp else 0
                            gateway = self._select_gateway_for_client(websocket)
                            if gateway is None:
                                error_started_ns = time.perf_counter_ns()
                                error_message = {
                                    'type': 'webrtc_error',
                                    'error': 'webrtc_capacity_reached',
                                }
                                await self._send_json_with_boundary_metrics(
                                    websocket,
                                    error_message,
                                    route="webrtc_offer",
                                    message_type="webrtc_error",
                                    response_model_timing=self.response_model_timing_since(
                                        error_started_ns
                                    ),
                                )
                                self.logger.warning(
                                    "    No free WebRTC gateway slot for %s (capacity=%d)",
                                    client_ip,
                                    len(self.webrtc_gateways),
                                )
                                continue
                            self._set_gateway_owner(gateway, websocket, client_ip)
                            slot = self.webrtc_gateways.index(gateway)
                            self.logger.info(
                                "    Forwarding offer to gateway slot %d (%d SDP lines)",
                                slot,
                                sdp_lines,
                            )
                            gateway.accept_offer(sdp)
                        else:
                            error_started_ns = time.perf_counter_ns()
                            error_message = {
                                'type': 'webrtc_error',
                                'error': 'no_webrtc_gateway',
                            }
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                error_message,
                                route="webrtc_offer",
                                message_type="webrtc_error",
                                response_model_timing=self.response_model_timing_since(
                                    error_started_ns
                                ),
                            )
                            self.logger.warning("    No mosaic WebRTC gateway available")

                    # Handle WebRTC signaling: ICE candidate from browser
                    elif data.get('type') == 'webrtc_ice_candidate':
                        self.logger.info(">>> Received webrtc_ice_candidate from client %s", client_ip)
                        if self.webrtc_gateways or self._webrtc_gateway_factory is not None:
                            gateway = self._webrtc_client_gateway.get(websocket)
                            if gateway is None or gateway not in self.webrtc_gateways:
                                try:
                                    response_model_started_ns = time.perf_counter_ns()
                                    error_message = {
                                        'type': 'webrtc_error',
                                        'error': 'webrtc_not_owner',
                                    }
                                    await self._send_json_with_boundary_metrics(
                                        websocket,
                                        error_message,
                                        route="webrtc_ice_candidate",
                                        message_type="webrtc_error",
                                        response_model_timing=self.response_model_timing_since(
                                            response_model_started_ns
                                        ),
                                    )
                                except Exception:
                                    pass
                                continue
                            owner = self._get_gateway_owner(gateway)
                            if owner is not websocket:
                                try:
                                    response_model_started_ns = time.perf_counter_ns()
                                    error_message = {
                                        'type': 'webrtc_error',
                                        'error': 'webrtc_not_owner',
                                    }
                                    await self._send_json_with_boundary_metrics(
                                        websocket,
                                        error_message,
                                        route="webrtc_ice_candidate",
                                        message_type="webrtc_error",
                                        response_model_timing=self.response_model_timing_since(
                                            response_model_started_ns
                                        ),
                                    )
                                except Exception:
                                    pass
                                continue
                            candidate = data.get('candidate', '')
                            mline_index = data.get('sdpMLineIndex', 0)
                            self.logger.info(
                                "    Adding ICE candidate to gateway slot %d (mline=%d bytes=%d)",
                                self.webrtc_gateways.index(gateway),
                                mline_index,
                                len(candidate) if candidate else 0,
                            )
                            gateway.accept_ice(candidate, mline_index)
                        else:
                            response_model_started_ns = time.perf_counter_ns()
                            error_message = {
                                'type': 'webrtc_error',
                                'error': 'no_webrtc_gateway',
                            }
                            await self._send_json_with_boundary_metrics(
                                websocket,
                                error_message,
                                route="webrtc_ice_candidate",
                                message_type="webrtc_error",
                                response_model_timing=self.response_model_timing_since(
                                    response_model_started_ns
                                ),
                            )

                except json.JSONDecodeError:
                    self.logger.warning(f"Received non-JSON message from {client_ip}. Ignoring.")
                except websockets.exceptions.ConnectionClosed:
                    # Let the outer lifecycle handler classify and remove the
                    # closed client exactly once.  Treating it as a request
                    # failure keeps iterating a dead socket and emits false
                    # runtime ERRORs during normal gate disconnects/shutdown.
                    raise
                except Exception as e:
                    self.logger.error(f"Error processing message from {client_ip}: {e}")

        except websockets.exceptions.ConnectionClosedOK:
            self.logger.info(f"Client {client_ip} disconnected normally (code: 1000).")
            print(f"❌ Client {client_ip} disconnected normally.")
        except websockets.exceptions.ConnectionClosedError as e:
            # Log ping timeout errors as info instead of warning to reduce noise
            if "keepalive ping timeout" in str(e).lower():
                self.logger.info(f"Client {client_ip} disconnected due to ping timeout - connection cleaned up.")
            else:
                self.logger.warning(f"Client {client_ip} disconnected with error: {e}")
                print(f"❌ Client {client_ip} disconnected with error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error with client {client_ip}: {e}")
        finally:
            # Ensure client is removed from set - use discard to avoid KeyError if already removed
            try:
                self.connected_clients.discard(websocket)
                self._clear_gateway_owner_for_client(websocket)
                if self._webrtc_owner is websocket:
                    self._webrtc_owner = None
                    self._webrtc_owner_ip = None
                self.logger.info(f"Client {client_ip} removed. Total clients: {len(self.connected_clients)}")
                print(f"👋 Client {client_ip} removed. Total clients: {len(self.connected_clients)}")
            except Exception as e:
                self.logger.warning(f"Error removing client {client_ip} from connected clients: {e}")
    
    async def broadcast(
        self,
        message: Any,
        *,
        publisher_enqueue_ms: float = 0.0,
        response_model_timing: Optional[BoundaryResponseModelTiming] = None,
        _canonical_outbound_token: object | None = None,
    ) -> None:
        """Broadcast a message to all connected WebSocket clients

        Args:
            message: Message to broadcast (dict, bytes, or string)
        """
        self._validate_frozen_outbound(message)
        canonical_type = self._canonical_message_type(message)
        if (
            canonical_type is not None
            and _canonical_outbound_token is not self._canonical_outbound_token
        ):
            self._reject_generic_canonical_route(
                message,
                route="broadcast",
            )
        if not self.connected_clients:
            return

        message_str = ""
        disconnected_clients = []
        # Use a list to preserve order for correct result-to-client mapping
        active_clients = list(self.connected_clients)
        is_json_message = isinstance(message, (dict, FrozenOutboundJSON))
        message_type = "unknown"
        payload_bytes = 0
        response_model_ms = 0.0

        try:
            # Prepare message based on type
            if isinstance(message, FrozenOutboundJSON):
                message_type = message.message_type
                if not isinstance(
                    response_model_timing,
                    BoundaryResponseModelTiming,
                ):
                    raise BoundaryResponseModelContractError(
                        "pre-encoded WebSocket JSON is missing response timing"
                    )
                response_model_ms = self._response_model_ms(
                    response_model_timing
                )
                message_str = message.encoded
                payload_bytes = int(message.payload_bytes)
                convert_ms = float(message.convert_ms)
                encode_ms = float(message.encode_ms)
                self._record_boundary_serialization_stage(
                    response_model_ms,
                    payload_bytes,
                    channel="ws",
                    route="broadcast",
                    message_type=message_type,
                    stage="response_model",
                    outcome="ok",
                    include_budget=False,
                )
                if message.telemetry_payload is not None:
                    try:
                        self._record_tx(message.telemetry_payload)
                    except Exception:
                        pass
                self._record_boundary_serialization_stage(
                    publisher_enqueue_ms,
                    payload_bytes,
                    channel="ws",
                    route="broadcast",
                    message_type=message_type,
                    stage="publisher_enqueue",
                    outcome="ok",
                    include_budget=False,
                )
            elif isinstance(message, dict):
                # Convert to JSON string
                message_type = str(message.get("type", "unknown"))
                response_model_timing = self._require_response_model_timing(
                    message,
                    response_model_timing,
                    route="broadcast",
                    message_type=message_type,
                )
                response_model_ms = self._response_model_ms(
                    response_model_timing
                )
                try:
                    message_str, payload_bytes, _, convert_ms, encode_ms = (
                        _serialize_boundary_json(message)
                    )
                except Exception as exc:
                    self._record_boundary_serialization_error(
                        channel="ws",
                        route="broadcast",
                        message_type=message_type,
                        stage=(
                            "numpy_convert"
                            if isinstance(exc, _BoundaryConversionError)
                            else "json_encode"
                        ),
                        error=exc,
                    )
                    raise
                self._record_boundary_serialization_stage(
                    response_model_ms,
                    payload_bytes,
                    channel="ws",
                    route="broadcast",
                    message_type=message_type,
                    stage="response_model",
                    outcome="ok",
                    include_budget=False,
                )
                self._record_boundary_serialization_stage(
                    convert_ms,
                    payload_bytes,
                    channel="ws",
                    route="broadcast",
                    message_type=message_type,
                    stage="numpy_convert",
                    outcome="ok",
                    include_budget=False,
                )
                # Record TX telemetry for tracked messages (e.g., calibration-bundle)
                try:
                    self._record_tx(message)
                except Exception:
                    pass
                self._record_boundary_serialization_stage(
                    encode_ms,
                    payload_bytes,
                    channel="ws",
                    route="broadcast",
                    message_type=message_type,
                    stage="json_encode",
                    outcome="ok",
                    include_budget=False,
                )
                self._record_boundary_serialization_stage(
                    publisher_enqueue_ms,
                    payload_bytes,
                    channel="ws",
                    route="broadcast",
                    message_type=message_type,
                    stage="publisher_enqueue",
                    outcome="ok",
                    include_budget=False,
                )
            elif isinstance(message, bytes):
                # Route binary frames into coalescer; actual sending is handled elsewhere
                await self._coalesce_binary_and_maybe_flush(message)
                return
            elif isinstance(message, str):
                # String message
                message_str = message
            else:
                self.logger.warning(f"Unknown message type: {type(message)}")
                return

            # Construct and schedule delivery tasks locally, then stop the CPU
            # boundary clock before awaiting transport flow-control.
            delivery_tasks: List[asyncio.Task[Any]] = []
            send_start_ns = time.perf_counter_ns()
            try:
                for client in active_clients:
                    delivery_tasks.append(
                        asyncio.create_task(client.send(message_str))
                    )
            except Exception as exc:
                for task in delivery_tasks:
                    task.cancel()
                if delivery_tasks:
                    await asyncio.gather(*delivery_tasks, return_exceptions=True)
                if is_json_message:
                    self._record_boundary_serialization_error(
                        channel="ws",
                        route="broadcast",
                        message_type=message_type,
                        stage="send_dispatch",
                        error=exc,
                    )
                raise
            send_ms = (time.perf_counter_ns() - send_start_ns) / 1_000_000.0
            if is_json_message:
                self._record_boundary_serialization_stage(
                    send_ms,
                    payload_bytes,
                    channel="ws",
                    route="broadcast",
                    message_type=message_type,
                    stage="send_dispatch",
                    outcome="ok",
                    include_budget=False,
                )
                # The assembled broadcast total begins when a producer submits
                # cross-thread work and ends after local fanout send dispatch.
                total_ms = float(
                    response_model_ms
                    + publisher_enqueue_ms
                    + convert_ms
                    + encode_ms
                    + send_ms
                )
                self._record_boundary_serialization_stage(
                    total_ms,
                    payload_bytes,
                    channel="ws",
                    route="broadcast",
                    message_type=message_type,
                    stage="total",
                    outcome="ok",
                    include_budget=True,
                )

            results = await asyncio.gather(
                *delivery_tasks,
                return_exceptions=True,
            )

            # Check for errors and mark disconnected clients
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    client = active_clients[i] if i < len(active_clients) else None
                    if client:
                        disconnected_clients.append(client)
                    client_ip = client.remote_address if client and hasattr(client, 'remote_address') else "Unknown"
                    # Only log ping timeout errors as debug to reduce noise
                    if isinstance(result, websockets.exceptions.ConnectionClosedOK):
                        self.logger.debug(
                            "Client %s closed before broadcast delivery: %s",
                            client_ip,
                            result,
                        )
                    elif isinstance(result, websockets.exceptions.ConnectionClosedError):
                        self.logger.warning(
                            "Client %s disconnected before broadcast delivery: %s",
                            client_ip,
                            result,
                        )
                    elif "keepalive ping timeout" in str(result):
                        self.logger.debug(f"Client {client_ip} ping timeout - will be cleaned up")
                    else:
                        self.logger.error(f"Failed to send message to {client_ip}: {result}")

        except Exception as e:
            self.logger.error(f"Broadcast error: {e}")
            raise
        finally:
            # Immediately clean up disconnected clients from the set
            if disconnected_clients:
                for client in disconnected_clients:
                    if client in self.connected_clients:
                        self.connected_clients.remove(client)
                        client_ip = client.remote_address if hasattr(client, 'remote_address') else "Unknown"
                        self.logger.info(f"Removed disconnected client {client_ip} from connected clients")
                self.logger.debug(f"Cleaned up {len(disconnected_clients)} disconnected clients during broadcast")
    
    async def _broadcast_from_sync_submission(
        self,
        message: Any,
        submitted_ns: int,
        response_model_timing: Optional[BoundaryResponseModelTiming] = None,
        *,
        canonical_outbound_token: object | None = None,
    ) -> None:
        publisher_enqueue_ms = max(
            0.0,
            (time.perf_counter_ns() - submitted_ns) / 1_000_000.0,
        )
        if self._should_coalesce_json_message(message):
            await self._coalesce_json_and_maybe_flush(
                message,
                publisher_enqueue_ms=publisher_enqueue_ms,
                response_model_timing=response_model_timing,
            )
            return
        await self.broadcast(
            message,
            publisher_enqueue_ms=publisher_enqueue_ms,
            response_model_timing=response_model_timing,
            _canonical_outbound_token=canonical_outbound_token,
        )

    async def _broadcast_batch_from_sync_submission(
        self,
        messages: Sequence[Any],
        submitted_ns: int,
        response_model_timings: Sequence[
            Optional[BoundaryResponseModelTiming]
        ],
        *,
        canonical_outbound_token: object | None = None,
    ) -> None:
        """Deliver one admitted canonical batch in exact list order.

        The batch intentionally bypasses latest-only coalescing.  Admission is
        atomic, while actual client delivery remains the normal WebSocket
        delivery contract and can fail asynchronously per connection.
        """

        for message, response_model_timing in zip(
            messages,
            response_model_timings,
            strict=True,
        ):
            publisher_enqueue_ms = max(
                0.0,
                (time.perf_counter_ns() - submitted_ns) / 1_000_000.0,
            )
            await self.broadcast(
                message,
                publisher_enqueue_ms=publisher_enqueue_ms,
                response_model_timing=response_model_timing,
                _canonical_outbound_token=canonical_outbound_token,
            )

    async def _broadcast_gated_batch_from_sync_submission(
        self,
        messages: Sequence[Any],
        response_model_timings: Sequence[
            Optional[BoundaryResponseModelTiming]
        ],
        decision: concurrent.futures.Future[_OutboundGateDecision],
    ) -> Any:
        """Wait for authority resolution before beginning any batch delivery."""

        resolved = await asyncio.wrap_future(decision)
        if not resolved.release:
            return _OUTBOUND_ABORTED
        await self._broadcast_batch_from_sync_submission(
            messages,
            int(resolved.resolved_at_ns),
            response_model_timings,
            canonical_outbound_token=self._canonical_outbound_token,
        )
        return None

    def _freeze_outbound_message(
        self,
        message: Any,
        *,
        route: str = "broadcast",
    ) -> tuple[Any, int]:
        """Detach one admitted value from caller-owned mutable objects."""

        if isinstance(message, dict):
            message_type = str(message.get("type", "unknown"))
            coalesce_key = (
                self._json_coalesce_key(message)
                if route == "broadcast"
                else None
            )
            try:
                (
                    encoded,
                    payload_bytes,
                    _dispatch_ms,
                    convert_ms,
                    encode_ms,
                ) = _serialize_boundary_json(message)
            except Exception as exc:
                self._record_boundary_serialization_error(
                    channel="ws",
                    route=route,
                    message_type=message_type,
                    stage=(
                        "admission_freeze_numpy_convert"
                        if isinstance(exc, _BoundaryConversionError)
                        else "admission_freeze_json_encode"
                    ),
                    error=exc,
                )
                raise
            if payload_bytes > int(self.OUTBOUND_MESSAGE_MAX_BYTES):
                raise ValueError("outbound JSON message exceeds the byte limit")
            telemetry_payload = None
            if message_type in {
                "calibration-bundle",
                "set_align_result",
                "set_extrinsics_result",
                "solve_pnp_result",
                "pixel_to_world_response",
            }:
                decode_started_ns = time.perf_counter_ns()
                telemetry_payload = json.loads(encoded)
                encode_ms += (
                    time.perf_counter_ns() - decode_started_ns
                ) / 1_000_000.0
            self._record_boundary_serialization_stage(
                convert_ms,
                payload_bytes,
                channel="ws",
                route=route,
                message_type=message_type,
                stage="admission_freeze_numpy_convert",
                outcome="ok",
                include_budget=False,
            )
            self._record_boundary_serialization_stage(
                encode_ms,
                payload_bytes,
                channel="ws",
                route=route,
                message_type=message_type,
                stage="admission_freeze_json_encode",
                outcome="ok",
                include_budget=False,
            )
            return (
                FrozenOutboundJSON(
                    encoded=encoded,
                    payload_bytes=int(payload_bytes),
                    message_type=message_type,
                    convert_ms=float(convert_ms),
                    encode_ms=float(encode_ms),
                    route=str(route),
                    coalesce_key=coalesce_key,
                    telemetry_payload=telemetry_payload,
                    owner_token=self._frozen_outbound_token,
                ),
                int(payload_bytes),
            )
        if isinstance(message, str):
            payload_bytes = len(message.encode("utf-8"))
            if payload_bytes > int(self.OUTBOUND_MESSAGE_MAX_BYTES):
                raise ValueError("outbound text message exceeds the byte limit")
            return str(message), payload_bytes
        if isinstance(message, bytes):
            payload_bytes = len(message)
            if payload_bytes > int(self.OUTBOUND_MESSAGE_MAX_BYTES):
                raise ValueError("outbound binary message exceeds the byte limit")
            return bytes(message), payload_bytes
        raise TypeError("outbound message must be dict, str, or bytes")

    def broadcast_sync(
        self,
        message: Any,
        *,
        response_model_timing: Optional[BoundaryResponseModelTiming] = None,
    ) -> OutboundAdmissionReceipt:
        """Synchronous version of broadcast for use from other threads
        
        Args:
            message: Message to broadcast
        """
        self._reject_generic_canonical_route(
            message,
            route="broadcast_sync",
        )
        if isinstance(message, dict):
            response_model_timing = self._require_response_model_timing(
                message,
                response_model_timing,
                route="broadcast",
                message_type=str(message.get("type", "unknown")),
            )
        frozen_message, payload_bytes = self._freeze_outbound_message(message)
        # Capture submission before crossing into the event-loop thread.  The
        # receiving coroutine records only this dispatch delay; any intentional
        # latest-only coalescing dwell begins after receipt and is excluded.
        submitted_ns = time.perf_counter_ns()
        return self._submit_outbound_coroutine(
            lambda: self._broadcast_from_sync_submission(
                frozen_message,
                submitted_ns,
                response_model_timing=response_model_timing,
            ),
            message_count=1,
            payload_bytes=payload_bytes,
        )

    def broadcast_batch_sync(
        self,
        messages: Sequence[Any],
        *,
        response_model_timing: Optional[BoundaryResponseModelTiming] = None,
    ) -> OutboundAdmissionReceipt:
        """Atomically admit one ordered, non-coalesced outbound batch.

        The returned receipt proves bounded sender admission only.  It is not
        an acknowledgement from any connected WebSocket client.
        """

        if isinstance(messages, (str, bytes, bytearray)):
            raise TypeError("outbound batch must be a sequence of messages")
        batch = tuple(messages)
        for message in batch:
            self._reject_generic_canonical_route(
                message,
                route="broadcast_batch_sync",
            )
        frozen_messages, validated_timings, total_bytes = (
            self._freeze_outbound_batch(
                batch,
                response_model_timing=response_model_timing,
            )
        )
        submitted_ns = time.perf_counter_ns()
        return self._submit_outbound_coroutine(
            lambda: self._broadcast_batch_from_sync_submission(
                frozen_messages,
                submitted_ns,
                validated_timings,
            ),
            message_count=len(frozen_messages),
            payload_bytes=total_bytes,
        )

    def _freeze_outbound_batch(
        self,
        messages: Sequence[Any],
        *,
        response_model_timing: Optional[BoundaryResponseModelTiming],
    ) -> tuple[
        tuple[Any, ...],
        tuple[Optional[BoundaryResponseModelTiming], ...],
        int,
    ]:
        """Validate and freeze one all-or-none ordered outbound batch."""

        if isinstance(messages, (str, bytes, bytearray)):
            raise TypeError("outbound batch must be a sequence of messages")
        batch = tuple(messages)
        if not batch:
            raise ValueError("outbound batch must contain at least one message")
        if len(batch) > int(self.OUTBOUND_BATCH_MAX_MESSAGES):
            raise ValueError("outbound batch exceeds the message limit")
        validated_timings: list[Optional[BoundaryResponseModelTiming]] = []
        frozen_messages: list[Any] = []
        total_bytes = 0
        for message in batch:
            if not isinstance(message, (dict, str, bytes)):
                raise TypeError(
                    "outbound batch messages must be dict, str, or bytes"
                )
            timing = response_model_timing
            if isinstance(message, dict):
                timing = self._require_response_model_timing(
                    message,
                    timing,
                    route="broadcast",
                    message_type=str(message.get("type", "unknown")),
                )
            validated_timings.append(timing)
            frozen, payload_bytes = self._freeze_outbound_message(message)
            frozen_messages.append(frozen)
            total_bytes += int(payload_bytes)
            if total_bytes > int(self.OUTBOUND_BATCH_MAX_BYTES):
                raise ValueError("outbound batch exceeds the byte limit")
        return (
            tuple(frozen_messages),
            tuple(validated_timings),
            int(total_bytes),
        )

    @staticmethod
    def _validate_authority_gated_batch(
        messages: Sequence[Any],
    ) -> tuple[Any, ...]:
        """Require the exact tracking-first canonical cohort shape."""

        batch = tuple(messages)
        if not batch or not all(isinstance(item, dict) for item in batch):
            raise ValueError(
                "authority-gated batch must contain JSON objects"
            )
        tracking = batch[0]
        if str(tracking.get("type", "")) != "tracking":
            raise ValueError(
                "authority-gated batch must begin with tracking"
            )
        cohort = tracking.get("cohort")
        if not isinstance(cohort, dict):
            raise ValueError(
                "authority-gated tracking message requires a cohort"
            )
        cohort_fields = (
            "source_id",
            "frame_id",
            "observed_at_us",
            "tracking_publication_sequence",
        )
        if set(cohort) != set(cohort_fields):
            raise ValueError(
                "authority-gated tracking cohort shape is not exact"
            )
        source_id = cohort["source_id"]
        sequence = cohort["tracking_publication_sequence"]
        if (
            isinstance(source_id, bool)
            or not isinstance(source_id, int)
            or source_id < 0
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
        ):
            raise ValueError(
                "authority-gated tracking cohort identity is invalid"
            )
        for optional_field in ("frame_id", "observed_at_us"):
            value = cohort[optional_field]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or (optional_field == "observed_at_us" and value == 0)
            ):
                raise ValueError(
                    "authority-gated tracking cohort time/frame is invalid"
                )
        for field in cohort_fields:
            if tracking.get(field) != cohort[field]:
                raise ValueError(
                    "authority-gated tracking cohort fields do not match"
                )

        snapshot = tracking.get("world_snapshot")
        events = tracking.get("world_events")
        if snapshot is None:
            if events not in (None, []):
                raise ValueError(
                    "authority-gated tracking events require a world snapshot"
                )
            if len(batch) != 1:
                raise ValueError(
                    "tracking-only authority batch cannot contain extra messages"
                )
            return batch
        if not isinstance(snapshot, dict) or not isinstance(events, list):
            raise ValueError(
                "authority-gated world snapshot/events shape is invalid"
            )
        if len(batch) != 2 + len(events):
            raise ValueError(
                "authority-gated batch message count does not match its cohort"
            )
        snapshot_message = batch[1]
        if (
            str(snapshot_message.get("type", "")) != "world_snapshot"
            or snapshot_message.get("payload") != snapshot
        ):
            raise ValueError(
                "authority-gated world snapshot message is not exact"
            )
        for field in cohort_fields:
            if snapshot_message.get(field) != cohort[field]:
                raise ValueError(
                    "authority-gated world snapshot cohort fields do not match"
                )
        if snapshot_message.get("cohort") != cohort:
            raise ValueError(
                "authority-gated world snapshot cohort is not exact"
            )
        for message, event in zip(batch[2:], events, strict=True):
            if (
                str(message.get("type", "")) != "world_event"
                or message.get("payload") != event
                or message.get("cohort") != cohort
            ):
                raise ValueError(
                    "authority-gated world event message is not exact"
                )
            for field in cohort_fields:
                if message.get(field) != cohort[field]:
                    raise ValueError(
                        "authority-gated world event cohort fields do not match"
                    )
        return batch

    def admit_broadcast_batch_sync(
        self,
        messages: Sequence[Any],
        *,
        response_model_timing: Optional[BoundaryResponseModelTiming] = None,
    ) -> GatedOutboundAdmission:
        """Bound one ordered batch while withholding delivery for authority.

        The returned handle owns a one-shot release/abort decision.  Until that
        decision is resolved, the exact frozen bytes remain part of outbound
        count/byte capacity and shutdown quiescence.
        """

        canonical_batch = self._validate_authority_gated_batch(messages)
        frozen_messages, validated_timings, total_bytes = (
            self._freeze_outbound_batch(
                canonical_batch,
                response_model_timing=response_model_timing,
            )
        )
        decision: concurrent.futures.Future[_OutboundGateDecision] = (
            concurrent.futures.Future()
        )
        receipt, submission_future = self._admit_outbound_coroutine(
            lambda: self._broadcast_gated_batch_from_sync_submission(
                frozen_messages,
                validated_timings,
                decision,
            ),
            message_count=len(frozen_messages),
            payload_bytes=total_bytes,
        )

        def _cancel_unresolved_gate_on_submission_failure(
            future: concurrent.futures.Future[Any],
        ) -> None:
            failed = future.cancelled()
            if not failed:
                try:
                    failed = future.exception() is not None
                except BaseException:
                    failed = True
            if failed and not decision.done():
                decision.cancel()

        submission_future.add_done_callback(
            _cancel_unresolved_gate_on_submission_failure
        )
        return GatedOutboundAdmission(
            receipt,
            decision,
            submission_future,
        )

    def broadcast_bev_sync(
        self,
        message: Any,
        *,
        response_model_timing: Optional[BoundaryResponseModelTiming] = None,
    ) -> OutboundAdmissionReceipt:
        """Admit one canonical BEV frame/status through its typed route."""

        if not isinstance(message, dict):
            raise TypeError(
                "broadcast_bev_sync requires a typed BEV message"
            )
        message_type = str(message.get("type", ""))
        if message_type not in self.DEDICATED_BEV_MESSAGE_TYPES:
            raise TypeError(
                "broadcast_bev_sync requires type=bev-frame or bev-status"
            )
        response_model_timing = self._require_response_model_timing(
            message,
            response_model_timing,
            route="broadcast",
            message_type=message_type,
        )
        frozen_message, payload_bytes = self._freeze_outbound_message(message)
        submitted_ns = time.perf_counter_ns()
        return self._submit_outbound_coroutine(
            lambda: self._broadcast_from_sync_submission(
                frozen_message,
                submitted_ns,
                response_model_timing=response_model_timing,
                canonical_outbound_token=self._canonical_outbound_token,
            ),
            message_count=1,
            payload_bytes=payload_bytes,
        )

    def broadcast_frame(self, frame_data: Dict[str, Any]) -> None:
        """
        Synchronous helper so DeepStream pipeline can push frames without asyncio context.
        Converts payload to JSON and re-uses existing broadcast_sync().
        """
        from models import convert_numpy_types
        response_model_started_ns = time.perf_counter_ns()
        payload = {
            "type": "frame",
            "payload": convert_numpy_types(frame_data),
        }
        self.broadcast_sync(
            payload,
            response_model_timing=self.response_model_timing_since(
                response_model_started_ns
            ),
        )

    # ---------------- Binary coalescer helpers ----------------
    def _extract_cam_id_from_binary(self, data: bytes) -> Optional[str]:
        try:
            if not data or len(data) < 1:
                return None
            id_len = data[0]
            if len(data) < 1 + id_len:
                return None
            cam_id_bytes = data[1:1 + id_len]
            return cam_id_bytes.decode('utf-8', errors='ignore')
        except Exception:
            return None

    async def _coalesce_binary_and_maybe_flush(self, data: bytes) -> None:
        """Keep only the latest binary frame per camera and schedule a flush if idle."""
        try:
            cam_id = self._extract_cam_id_from_binary(data)
            if cam_id is None:
                # If we cannot parse, fallback to immediate broadcast
                await self._broadcast_binary_immediate(data)
                return
            # Coalesce latest per camera
            self._latest_binary_by_cam[cam_id] = data

            # Schedule a flush if not currently sending
            if not self._binary_sending:
                # Avoid creating multiple concurrent flushers
                if self._binary_flush_task is None or self._binary_flush_task.done():
                    self._binary_flush_task = asyncio.create_task(self._flush_binary_queue(), name="BinaryFlush")
        except Exception as e:
            self.logger.debug(f"Coalescer error: {e}")

    async def _broadcast_binary_immediate(self, data: bytes) -> None:
        """Fallback path to send a single binary payload to all clients."""
        if not self.connected_clients:
            return
        active_clients = list(self.connected_clients)
        results = await asyncio.gather(
            *[client.send(data) for client in active_clients],
            return_exceptions=True
        )
        disconnected_clients = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                client = active_clients[i] if i < len(active_clients) else None
                if client:
                    disconnected_clients.append(client)
                client_ip = client.remote_address if client and hasattr(client, 'remote_address') else "Unknown"
                if isinstance(result, websockets.exceptions.ConnectionClosedOK):
                    self.logger.debug(
                        "Client %s closed before binary delivery: %s",
                        client_ip,
                        result,
                    )
                elif isinstance(result, websockets.exceptions.ConnectionClosedError):
                    self.logger.warning(
                        "Client %s disconnected before binary delivery: %s",
                        client_ip,
                        result,
                    )
                elif "keepalive ping timeout" in str(result).lower():
                    self.logger.debug(f"Client {client_ip} ping timeout - will be cleaned up")
                else:
                    self.logger.error(f"Failed to send binary message to {client_ip}: {result}")
        # Cleanup
        if disconnected_clients:
            for client in disconnected_clients:
                if client in self.connected_clients:
                    self.connected_clients.remove(client)
                    client_ip = client.remote_address if hasattr(client, 'remote_address') else "Unknown"
                    self.logger.info(f"Removed disconnected client {client_ip} from connected clients")

    async def _flush_binary_queue(self) -> None:
        """Flush latest binary frames per camera to all clients, dropping superseded frames."""
        if self._binary_sending:
            return
        self._binary_sending = True
        try:
            # Loop while there is data and server running
            while self.running and self._latest_binary_by_cam:
                if not self.connected_clients:
                    # No clients; keep latest for later but don't spin
                    await asyncio.sleep(0.05)
                    continue
                # Snapshot current latest and clear for coalescing of new arrivals
                batch = list(self._latest_binary_by_cam.items())
                self._latest_binary_by_cam.clear()

                # Send each camera's latest frame once
                for cam_id, payload in batch:
                    await self._broadcast_binary_immediate(payload)

                # Yield control to allow new coalescing
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Binary flush error: {e}")
        finally:
            self._binary_sending = False

    def _json_coalesce_key(self, message: Dict[str, Any]) -> Optional[str]:
        message_type = str(message.get("type", "") or "")
        if message_type in {
            "tracking",
            "world_snapshot",
            "world_event",
            "bev-frame",
            "bev-status",
        }:
            return None
        if message_type not in self._json_coalesce_interval_by_type:
            return None
        return f"{message_type}:global"

    def _should_coalesce_json_message(self, message: Any) -> bool:
        if isinstance(message, FrozenOutboundJSON):
            key = message.coalesce_key
            message_type = message.message_type
        elif isinstance(message, dict):
            key = self._json_coalesce_key(message)
            message_type = str(message.get("type", ""))
        else:
            return False
        if key is None:
            return False
        interval = float(
            self._json_coalesce_interval_by_type.get(message_type, 0.0)
            or 0.0
        )
        return interval > 0.0

    async def _coalesce_json_and_maybe_flush(
        self,
        message: Any,
        *,
        publisher_enqueue_ms: float = 0.0,
        response_model_timing: Optional[BoundaryResponseModelTiming] = None,
    ) -> None:
        try:
            self._validate_frozen_outbound(message)
            if isinstance(message, FrozenOutboundJSON):
                message_type = message.message_type
                if not isinstance(
                    response_model_timing,
                    BoundaryResponseModelTiming,
                ):
                    raise BoundaryResponseModelContractError(
                        "pre-encoded coalesced JSON is missing response timing"
                    )
                key = message.coalesce_key
            elif isinstance(message, dict):
                message_type = str(message.get("type", "unknown"))
                response_model_timing = self._require_response_model_timing(
                    message,
                    response_model_timing,
                    route="broadcast",
                    message_type=message_type,
                )
                key = self._json_coalesce_key(message)
            else:
                raise TypeError("JSON coalescer requires a JSON message")
            if key is None:
                await self.broadcast(
                    message,
                    publisher_enqueue_ms=publisher_enqueue_ms,
                    response_model_timing=response_model_timing,
                )
                return
            self._latest_json_by_key[key] = message
            self._latest_json_enqueue_ms_by_key[key] = max(
                0.0,
                float(publisher_enqueue_ms),
            )
            self._latest_json_response_model_ms_by_key[key] = (
                self._response_model_ms(response_model_timing)
            )
            if self._json_flush_task is None or self._json_flush_task.done():
                self._json_flush_task = asyncio.create_task(self._flush_json_queue(), name="JsonBroadcastFlush")
        except (
            BoundaryResponseModelContractError,
            CanonicalOutboundRouteRequired,
        ):
            raise
        except Exception as exc:
            self.logger.debug("JSON coalescer error: %s", exc)

    async def _flush_json_queue(self) -> None:
        if self._json_sending:
            return
        self._json_sending = True
        try:
            while self.running and self._latest_json_by_key:
                if not self.connected_clients:
                    self._latest_json_by_key.clear()
                    self._latest_json_enqueue_ms_by_key.clear()
                    self._latest_json_response_model_ms_by_key.clear()
                    return
                now = time.time()
                ready: List[tuple[Any, float, float]] = []
                next_due: Optional[float] = None
                for key, payload in list(self._latest_json_by_key.items()):
                    message_type = (
                        payload.message_type
                        if isinstance(payload, FrozenOutboundJSON)
                        else str(payload.get("type", "") or "")
                    )
                    interval = float(self._json_coalesce_interval_by_type.get(message_type, 0.0) or 0.0)
                    last_sent = float(self._json_last_sent.get(key, 0.0) or 0.0)
                    due = last_sent + interval
                    if now >= due:
                        enqueue_ms = self._latest_json_enqueue_ms_by_key.pop(
                            key,
                            0.0,
                        )
                        response_model_ms = (
                            self._latest_json_response_model_ms_by_key.pop(
                                key,
                                0.0,
                            )
                        )
                        ready.append((payload, enqueue_ms, response_model_ms))
                        self._latest_json_by_key.pop(key, None)
                        self._json_last_sent[key] = now
                    else:
                        next_due = due if next_due is None else min(next_due, due)

                for payload, enqueue_ms, response_model_ms in ready:
                    await self.broadcast(
                        payload,
                        publisher_enqueue_ms=enqueue_ms,
                        response_model_timing=BoundaryResponseModelTiming(
                            response_model_ms
                        ),
                    )

                if not ready:
                    sleep_s = 0.02
                    if next_due is not None:
                        sleep_s = max(0.005, min(0.05, float(next_due) - time.time()))
                    await asyncio.sleep(sleep_s)
                else:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.logger.error("JSON broadcast flush error: %s", exc)
        finally:
            self._json_sending = False


class WebSocketClient:
    """WebSocket client for testing the server"""
    
    def __init__(self, uri: str = "ws://localhost:6008"):
        """Initialize the WebSocket client
        
        Args:
            uri: WebSocket server URI
        """
        self.uri = uri
        self.websocket = None
        self.running = False
        self.logger = logging.getLogger("WebSocketClient")
    
    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.running = True
            self.logger.info(f"Connected to {self.uri}")
            return True
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.running = False
            self.logger.info("Disconnected")
    
    async def send(self, message):
        """Send a message to the server
        
        Args:
            message: Message to send (dict or string)
        """
        if not self.websocket:
            self.logger.error("Not connected")
            return
            
        try:
            # Convert dict to JSON string
            if isinstance(message, dict):
                message = json.dumps(message)
                
            await self.websocket.send(message)
        except Exception as e:
            self.logger.error(f"Send error: {e}")
    
    async def receive(self):
        """Receive a message from the server
        
        Returns:
            The received message
        """
        if not self.websocket:
            self.logger.error("Not connected")
            return None
            
        try:
            message = await self.websocket.recv()
            
            # Try to parse as JSON
            try:
                return json.loads(message)
            except json.JSONDecodeError:
                return message
        except Exception as e:
            self.logger.error(f"Receive error: {e}")
            return None
    
    async def listen(self, callback):
        """Listen for messages from the server
        
        Args:
            callback: Function to call with received messages
        """
        if not self.websocket:
            self.logger.error("Not connected")
            return
            
        self.running = True
        
        try:
            while self.running:
                message = await self.receive()
                if message:
                    callback(message)
        except Exception as e:
            self.logger.error(f"Listen error: {e}")
        finally:
            self.running = False 
    
