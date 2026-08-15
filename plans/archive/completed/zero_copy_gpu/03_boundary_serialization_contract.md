# Boundary Serialization Contract (WS/REST/Depth Retrieval)

## Scope and Intent
- Owner: `DOC-BOUNDARY`
- Scope: CPU-side serialization boundaries only (WebSocket, REST, depth retrieval response assembly).
- Non-goal: changing any existing WS/REST payload schema, field names, or binary framing.
- SLO: every budgeted boundary path must satisfy p99 <= 3.0 ms in both the
  monotonic rolling 10-second and 60-second windows.

This contract defines where serialization is allowed, how it is measured, and what must stay wire-compatible.

## Contract-Preserved Interfaces

### WebSocket (must stay wire-compatible)
- `stats` (`type: "stats"`, `payload` unchanged)
- `depth_result` (`type: "depth_result"`, `payload` from `DepthResult.to_dict()`)
- `tracking` (`type: "tracking"`, `source_id`, `tracks`)
- `bev-frame` JSON payload + optional binary frame format `[len(header)][header="bev:<camera>"][jpeg bytes]`
- RPC responses:
  - `ma_depth_response` (including `camera`, `request_id`, `served_from_cache`, `ts_us`, `ok`, optional `error`, optional `payload`)
  - `floorplan_response` (including `request_id`, `camera_id`, `cache_only`, optional floorplan fields, optional `error`)
  - Existing calibration/control responses (`pixel_to_world_response`, `set_extrinsics_result`, `solve_pnp_result`, `set_align_result`, etc.)

### REST (must stay wire-compatible)
- `POST /api/v1/depth/refresh` response model `DepthRefreshResponse`
- `GET /api/v1/analytics/rois` response model `ROIListResponse`
- `POST /api/v1/analytics/rois` response model `ROIUpdateResponse`
- `ReID` endpoints under `/api/v1/reid/...` (all current response models and keys)

### Depth Retrieval Semantics (must stay wire-compatible)
- `get_ma_depth`: request key aliases accepted today (`camera/cameraId/camId`, `ts_max_us/tsMaxUs/...`, `request_id/requestId/...`) and current response shape.
- `get_floorplan`: current request aliases and response shape, including error semantics.

## Current Boundary Map (Code Anchors)
- WS outbound serialization:
  - `websocket_server.py:WebSocketServer.broadcast`
  - `websocket_server.py:WebSocketServer._send_to_client`
- WS depth/floorplan retrieval response assembly:
  - `websocket_server.py:WebSocketServer.handle_client` branches `get_ma_depth` and `get_floorplan`
  - Provider execution in `noesis/ds8_runtime.py` (`_ds8_ma_depth_provider`, `_ds8_floorplan_provider`) is retrieval compute, not serialization budget.
- REST serialization:
  - `noesis/server/depth_api.py:refresh_depth`
  - `noesis/server/analytics_api.py:list_rois`, `update_rois`
  - `noesis/server/reid_api.py` endpoint handlers
- Telemetry publisher ingress to WS boundary:
  - `noesis/telemetry/publishers.py:DepthTelemetryPublisher.publish`
  - `noesis/telemetry/publishers.py:TrackingTelemetryPublisher.publish`
  - `noesis/telemetry/bev.py:BevRenderer._publish` (JSON status + optional JPEG binary frame enqueue)

## Serialization Budget Definition (p99 <= 3.0 ms)

### What counts toward the 3.0 ms budget
- `convert_numpy_types(...)`
- JSON encoding (`json.dumps` or equivalent serializer)
- boundary response object materialization (`Pydantic model -> serializable`)
- executor queue/worker dispatch introduced by offloaded serialization
- producer-to-event-loop enqueue dispatch plus send dispatch at the boundary
  (up to websocket send scheduling / response object creation)
- REST endpoint response-model assembly, sync-worker handoff, FastAPI response-
  model validation/filtering, the one actual wire render, and response creation.

### What does not count toward the 3.0 ms budget
- Provider compute and blocking waits:
  - `ma_depth_provider` work (`load_latest_depth`, base64 payload build, normals attachment, gating wait)
  - `floorplan_provider` generation work
- GPU pipeline stages
- Network transit and remote client processing
- intentional latest-only JSON coalescing/rate-limit dwell after the event loop
  has accepted a producer submission

### Budget split (target)
- WS telemetry broadcast path:
  - numpy conversion: <= 0.6 ms p99
  - JSON encode: <= 1.4 ms p99
  - send dispatch (event-loop local): <= 1.0 ms p99
- WS depth/floorplan response serialization path:
  - payload normalization/conversion: <= 0.7 ms p99
  - JSON encode: <= 1.6 ms p99
  - send dispatch: <= 0.7 ms p99
- REST response serialization path:
  - model/payload assembly: <= 1.0 ms p99
  - JSON encode + response build: <= 2.0 ms p99

## Instrumentation Contract

### Timing source and sampling
- Use `time.perf_counter_ns()` for stage timings.
- Report milliseconds as `float` with 3 decimal precision.
- Keep per-metric rolling window in-process (reuse `LatencyWindow` pattern from `noesis/telemetry/latency_metrics.py`).
- Required percentiles: `p50`, `p95`, `p99`, `max`, `count`.

### Required metric names
- `boundary_serialization_total_ms`
  - histogram/window metric
  - tags: `channel`, `route`, `message_type`, `outcome`
- `boundary_serialization_stage_ms`
  - histogram/window metric
  - tags: `channel`, `route`, `message_type`, `stage`, `outcome`
  - `stage` in: `numpy_convert`, `json_encode`, `response_model`, `send_dispatch`, `publisher_enqueue`
  - additive diagnostic stage: `worker_dispatch_wait` for executor-backed JSON
- `boundary_serialization_payload_bytes`
  - histogram/window metric
  - tags: `channel`, `route`, `message_type`
- `boundary_serialization_errors_total`
  - counter
  - tags: `channel`, `route`, `message_type`, `stage`, `error_type`
  - REST records only response-model assembly, FastAPI validation/render, and
    local response-object failures. Provider/domain and network delivery errors
    remain outside this serialization counter.
- `boundary_serialization_samples_total`
  - counter
  - tags: `channel`, `route`, `message_type`, `outcome`
- `boundary_retrieval_provider_ms`
  - separate (excluded from 3ms budget)
  - tags: `provider` (`ma_depth`|`floorplan`), `outcome`
- `boundary_cpu_serialization_p99_ms`
  - derived gauge (max p99 across in-scope boundary paths in current window)
  - this is the gate metric for acceptance `A2`.
  - expose `p99_10s_ms`, `p99_60s_ms`, `max_path_p99_10s_ms`, and
    `max_path_p99_60s_ms`; legacy `p99_ms` and `max_path_p99_ms` are the
    conservative worst of the two required windows.
  - compute max-path values only from budgeted `stage=total` buckets. A pooled
    aggregate cannot admit a sparse slow path.

### Required instrumentation points
- `websocket_server.py`
  - `broadcast`:
    - time producer submission through event-loop coroutine start
      (`stage=publisher_enqueue`), excluding intentional coalescing dwell
    - time `convert_numpy_types` (`stage=numpy_convert`)
    - time `json.dumps` (`stage=json_encode`)
    - time `asyncio.gather(client.send(...))` dispatch (`stage=send_dispatch`)
    - emit `payload_bytes` from encoded message length
  - `_send_to_client`:
    - same staged timings for single-client path
  - `handle_client`:
    - `get_ma_depth` response assembly:
      - provider wait (`boundary_retrieval_provider_ms`, excluded from gate)
      - `convert_numpy_types` + `json.dumps` + `websocket.send` as in-scope serialization stages
    - `get_floorplan` response assembly:
      - provider wait metric (excluded) + in-scope encode/send stages
- `noesis/server/depth_api.py`
  - `refresh_depth`: mark response-model construction on `Request.state`; the
    boundary route measures the completed FastAPI response body after its one
    real render.
- `noesis/server/analytics_api.py`
  - `list_rois`, `update_rois`: mark response assembly, then measure the exact
    returned response bytes without serializing a surrogate.
- `noesis/server/reid_api.py`
  - all handlers: mark model/dict assembly; preserve their declared response
    models and let FastAPI perform aliasing, exclusion, validation, and encoding
    exactly once.
- `noesis/server/boundary_metrics.py`
  - `BoundaryMetricsRoute` wraps FastAPI's normal route handler and observes its
    final bounded byte body. `mark_rest_response` stores timing metadata only;
    it must never call `model_dump`, `json.dumps`, or another renderer.
  - timestamp samples with a monotonic clock, prune them beyond 60 seconds, and
    derive both required windows from one duration sort per bucket. Detailed
    route/stage cardinality is capped at 512 buckets per map.
  - route summaries contain only `stage=total` samples; component timings live
    only in stage summaries and must not multiply or dilute route count/p99.
  - cap every aggregate, route, stage, and budget-path sample buffer at 4096.
    On saturation, retain bounded drop count/time/max evidence and force the
    affected path/window above budget until the last saturation event expires;
    never silently evict a still-live offender.
  - expose `get_boundary_serialization_metrics_compact()` for runtime stats. It
    computes aggregate and budget-path authority without materializing full
    route/stage summaries, and returns at most eight ranked `top_budget_paths`
    plus path count/truncation metadata for actionable diagnostics.
- `noesis/telemetry/publishers.py`
  - `DepthTelemetryPublisher.publish`, `TrackingTelemetryPublisher.publish`:
    - `stage=publisher_enqueue` around `broadcast_sync` call.
    - optional `stage=response_model` for `DepthResult.to_dict()` in depth publisher.
- `noesis/telemetry/bev.py`
  - `BevRenderer._publish`:
    - JSON status encode/send dispatch timing in WS boundary metrics
    - JPEG encode timing as `stage=jpeg_encode` (tracked separately, excluded from JSON serialization SLO unless included explicitly in future policy)

## Optimization Directives (Schema-Safe)
- Do not rename/remove/add required wire fields in existing WS/REST contracts.
- Keep single-encode behavior for WS broadcast fanout (encode once, reuse string for all clients).
- Keep large payload JSON encoding off the main event-loop hot path for depth/floorplan (`asyncio.to_thread(json.dumps, ...)` or equivalent).
- Use an owned bounded serializer executor when offloading large payloads so
  provider work cannot create avoidable queue contention; prewarm it and join
  it during WebSocket shutdown. Any dispatch wait that still occurs remains in
  the 3 ms total.
- Run `convert_numpy_types` exactly once per outbound dict payload.
- For REST, retain FastAPI's native `response_model` and direct JSON fast path;
  never return a manually encoded substitute response solely for measurement.
- Prefer compact JSON separators (`separators=(",", ":")`) for WS/REST responses where behavior is unchanged.
- Avoid deep-copying payload dicts on boundary path unless mutation protection is required.
- Keep binary coalescing behavior (`_coalesce_binary_and_maybe_flush`) for BEV binary frames; do not add per-client per-frame copies.
- Preserve rate limiting semantics in depth/floorplan RPC trackers; throttling behavior is contract-observable.

## Stats Surface Contract (Additive Only)
- Existing `stats` payload remains backward compatible.
- Add optional node under `stats.payload.pipeline`:
  - `boundary_serialization_ms`:
    - `window_sec`
    - `samples`
    - `p50`
    - `p95`
    - `p99`
    - `max`
    - `violations`
    - `budget_ms` (=3.0)
    - `p99_10s_ms`, `p99_60s_ms`
    - `max_path_p99_10s_ms`, `max_path_p99_60s_ms`
- Add optional per-camera/path stats only as additive fields; existing keys stay untouched.

## Acceptance Gates
- Gate 1: aggregate and max-path p99 are each <= 3.0 ms over both true rolling
  10-second and 60-second windows. Max-path, not pooled aggregate, is the
  admission authority when they differ.
- Gate 2: `boundary_serialization_errors_total` has no sustained growth under nominal load.
- Gate 3: no WS/REST contract diffs for existing message types/endpoints (schema snapshot tests pass).
- Gate 4: depth retrieval correctness unchanged (`served_from_cache`, `request_id`, `error/ok` semantics preserved).
- Gate 5: adversarial REST tests prove exactly one FastAPI serialization, exact
  measured/returned byte-length equality, preserved status/headers/aliases/
  `exclude_none`/OpenAPI, deterministic sample expiry, and sparse-path detection.
- Gate 6: cardinality overflow uses overflow max (not pooled p99), high-rate
  sample saturation is bounded and fail-closed until expiry, compact output is
  bounded/actionable, and model/validation/render errors increment only the
  tagged REST boundary counters.

## Rollout Notes
- Phase 1: instrument-only (no behavior changes), collect baseline.
- Phase 2: apply safe optimizations above.
- Phase 3: enforce alert on `boundary_cpu_serialization_p99_ms > 3.0` and track violations in runtime reports.
