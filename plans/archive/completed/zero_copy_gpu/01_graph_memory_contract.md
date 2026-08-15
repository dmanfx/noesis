# DS8 Graph Memory Contract (Zero-Copy Core Path)

## 1. Scope and Decision
This contract is authoritative for DS8 core-path memory behavior in `noesis/pipelines/ds8_pipeline.py` and attached DS8 hooks/runtime integration.

Decision:
- Core graph is GPU/NVMM-only. CPU frame materialization is forbidden in core path.
- Boundary serialization paths (WS/REST/depth payload assembly) may use CPU memory, but must be explicitly instrumented and budgeted.
- Core-path zero-copy violations are fatal in production mode.

Core path covered by this contract:
- Source ingest/decode -> mux -> preprocess -> infer -> tracking/analytics -> tiler -> OSD -> sink tee -> RTSP branch.

Boundary path (allowed controlled host conversion):
- Telemetry and API payload assembly after metadata extraction.

## 2. Current DS8 Graph (As Built)
Graph builder: `noesis/pipelines/ds8_pipeline.py::build_pipeline`

### 2.1 Ingest variants
Variant A (per-source path, when `nvstreammux` flow is selected):
- `source_{i}` (`nvurisrcbin`) -> optional dewarper subgraph:
- `dewarper_conv_{i}` (`nvvideoconvert`) -> `dewarper_caps_{i}` (`capsfilter`, `video/x-raw(memory:NVMM),format=RGBA`) -> `dewarper_{i}` (`nvdewarper`) -> `dewarper_caps_out_{i}` (`capsfilter`, NVMM RGBA) -> `dewarper_post_conv_{i}` (`nvvideoconvert`) -> `dewarper_post_caps_{i}` (`capsfilter`, `video/x-raw(memory:NVMM),format=NV12`) -> `streammux` (`nvstreammux`)

Variant B (multi-URI ingest path):
- `streammux` (`nvmultiurisrcbin`) performs ingest + batching directly.

### 2.2 Common downstream graph
- `streammux` -> optional `preprocess` (`nvdspreprocess`) -> `yolo11_pgie` (`nvinfer`) -> `main_tee`

Main branch:
- `main_tee` -> optional `analytics_exclude` (`nvdsroiexclude`) -> `tracker` (`nvtracker`) -> optional `analytics` (`nvdsanalytics`) -> optional `reid_osnet` (`nvinfer`) -> optional `yolo26_pose` (`nvinfer`) -> `tiler` (`nvmultistreamtiler`) -> `osd` (`nvdsosd`) -> `sink_tee`

Depth/MapAnything branch:
- `main_tee` -> `mapanything_queue` (`queue`) -> `mapanything_valve` (`valve`) -> `mapanything_fullframe` (`nvinfer`) -> `mapanything_fullframe_sink` (`fakesink`)

Output branches from `sink_tee`:
- Optional user sinks (except `mosaic_sink` placeholder)
- Optional RTSP branch: `rtsp_queue` (`queue`) -> `rtsp_vconv` (`nvvideoconvert`) -> `rtsp_out` (`nvrtspoutsinkbin`)

## 3. Required Memory Invariants

### I1. Core buffers remain NVMM
Every video frame buffer on the core path MUST remain `video/x-raw(memory:NVMM)` from decode output through `rtsp_out` input.

### I2. Canonical nvbuf memory type
A single canonical memory type MUST be used across allocators/converters in the core path, sourced from `streammux.nvbuf-memory-type`.
- No hardcoded per-branch override that diverges from canonical value.

### I3. OSD must stay on GPU path
`nvdsosd` must run in GPU mode for zero-copy core path.
- Any OSD mode that materializes CPU surfaces is a core-path violation.

### I4. No frame-to-CPU API in core
Core-path code MUST NOT call frame extraction/conversion APIs that create host frame copies (directly or indirectly).
- Tensor metadata decoding for boundary payloads is allowed only after explicit boundary classification.

### I5. Tee isolation without copy fallback
All tee branches in core path MUST be non-blocking (`queue`/`leaky` where needed) and MUST NOT introduce CPU memory fallback due to backpressure/caps renegotiation.

### I6. Violations are explicit and counted
Each zero-copy violation MUST increment a runtime counter and record structured context (`component`, `reason`, `action`).

### I7. Boundary serialization budget
Boundary serialization p99 MUST stay <= 3.0 ms per assembled message path.

## 4. Per-Element Memory Expectations

| Element(s) | Expected memory behavior | Required config/condition | Violation condition |
|---|---|---|---|
| `source_{i}` (`nvurisrcbin`) | Decode outputs NVMM-capable surfaces for downstream | `gpu-id` set; RTSP reconnect knobs may vary | Downstream caps negotiate to non-NVMM |
| `streammux` (`nvstreammux` or `nvmultiurisrcbin`) | Batch surfaces remain NVMM | Explicit `nvbuf-memory-type`; consistent `gpu-id`; width/height fixed | Missing/implicit memory type or mismatch with downstream allocators |
| `dewarper_conv_{i}`, `dewarper_post_conv_{i}` (`nvvideoconvert`) | Convert in-GPU without host copy | `nvbuf-memory-type` == canonical | Converter allocates non-canonical memory |
| `dewarper_caps_{i}`, `dewarper_caps_out_{i}`, `dewarper_post_caps_{i}` (`capsfilter`) | Hard enforce NVMM caps | Caps include `memory:NVMM` | Caps omit NVMM feature |
| `preprocess` (`nvdspreprocess`) | GPU preprocess on NVMM input | DS preprocess config compatible with NVMM input | Fallback to host preprocessing |
| `yolo11_pgie`, `mapanything_fullframe`, `reid_osnet`, `yolo26_pose` (`nvinfer`) | Consume NVMM, produce metadata/tensors without frame copy | `output-tensor-meta` only where needed; unique IDs set | Host frame extraction in inference path |
| `tracker` (`nvtracker`) | Operates on metadata/surfaces in GPU pipeline | Tracker lib/config valid | Tracker path forces host copy due to unsupported format |
| `analytics_exclude`/`analytics` (`nvdsroiexclude`/`nvdsanalytics`) | Metadata operations only, no frame copy | Config present and parseable | CPU frame conversion introduced by hook/plugin fallback |
| `tiler` (`nvmultistreamtiler`) | Compose mosaic in NVMM | Width/height explicit | Output caps renegotiate to system memory |
| `osd` (`nvdsosd`) | Render overlay without host frame conversion | GPU mode required | CPU OSD mode in core path |
| `sink_tee`, `main_tee`, `mapanything_queue`, `rtsp_queue` | Routing only; no memory conversion | Queue/leaky config prevents stall | Backpressure causes renegotiation/copy path |
| `mapanything_valve` (`valve`) | Drop/forward only; no conversion | `drop-mode` forward sticky events | Valve state handling causes renegotiation to host memory |
| `rtsp_vconv` (`nvvideoconvert`) | Keep NVMM before RTSP sink bin | `nvbuf-memory-type` == canonical; optional NVMM capsfilter after vconv | Hardcoded memory type mismatch |
| `rtsp_out` (`nvrtspoutsinkbin`) | Accept NVMM input and handle encode/payloader internally | H264 settings + `sync=false` | Requires host buffers at input pad |
| `mapanything_fullframe_sink`/other fakesinks | Terminal drop only | `sync=false` for non-blocking | Sink blocks and triggers upstream renegotiation |

## 5. Failure Modes and Required Handling

| Failure mode | Where seen | Detection | Required action |
|---|---|---|---|
| OSD in CPU mode | `osd` config in graph build | Build-time contract check on `osd.process-mode` | Mark fatal `zero_copy_violation` and fail prepare/activate in production |
| Memory-type drift | `nvvideoconvert`/mux/rtsp branch | Build-time compare of `nvbuf-memory-type` across components | Normalize to canonical value; fail if unresolved |
| Caps lose NVMM feature | dewarper caps or RTSP branch | Build-time static checks + runtime caps probe where available | Fail startup (hard) for core edges |
| Hidden host conversion in hooks | tensor/frame conversion helpers | Runtime violation recorder around conversion functions | Count, log structured event, fail-fast in production |
| Tee branch backpressure leading renegotiation | `main_tee`/`sink_tee` branches | Runtime watchdog + error logs + queue occupancy heuristics | Keep branch leaky/non-blocking; treat renegotiation to host as violation |
| Boundary serialization regressions | WS/REST/depth providers | Per-path duration histograms | Keep p99 <= 3.0 ms; emit warning/error when exceeded |

## 6. Enforcement Checks

### 6.1 Build-time checks (must run on every graph build)
- Validate core component set and ordering for both ingest variants.
- Validate canonical memory type presence and consistency.
- Validate NVMM caps on all explicit capsfilters in core path.
- Validate OSD GPU mode.
- Validate RTSP branch converter memory type consistency.

### 6.2 Runtime checks (must run continuously)
- Violation counter and event log for any host conversion reached from core context.
- Stats payload includes:
  - `zero_copy_violations`
  - `zero_copy_last_violation`
  - boundary serialization p50/p95/p99 by path
- Production fail-fast when `zero_copy_violations > 0` on core path.

### 6.3 CI/Smoke checks
- Unit tests enforce graph invariants and guard against accidental CPU-mode regressions.
- Smoke test verifies zero violations during live run and boundary p99 budget.

## 7. Required Code Changes (Implementation-Ready)

### P0 (must land first)
1. `noesis/pipelines/ds8_pipeline.py::build_pipeline`
- Set OSD to GPU mode for core zero-copy path.
- Remove hardcoded `rtsp_vconv` memory override; use canonical `streammux.nvbuf-memory-type`.
- Add explicit NVMM capsfilter after `rtsp_vconv` (before `rtsp_out`) to prevent implicit host-memory negotiation.

2. `noesis/pipelines/ds8_pipeline.py::DS8Pipeline` (dataclass)
- Add zero-copy contract runtime fields:
  - `zero_copy_violations: int`
  - `zero_copy_last_violation: Optional[Dict[str, Any]]`
  - `boundary_serialize_stats: Dict[str, Any]`

3. `noesis/pipelines/ds8_pipeline.py` (new function) `:: _validate_zero_copy_contract`
- Implement graph/static checks for invariants I1-I5.
- Record violations as `pipeline.errors` entries prefixed with `zero_copy:`.
- Call from `build_pipeline` before returning.

4. `noesis/pipelines/ds8_pipeline.py::prepare` and `noesis/pipelines/ds8_pipeline.py::activate`
- Enforce fail-fast: if fatal zero-copy violations exist, return `False` and keep pipeline deactivated.

### P1 (core observability and guardrails)
5. `noesis/pipelines/hooks.py` (new helpers)
- Add `record_zero_copy_violation(pipeline, location, reason, fatal=True)`.
- Add boundary scope markers so tensor->numpy conversions are explicitly classified as boundary-only.

6. `noesis/pipelines/hooks.py::_numpy_from_layer`
- Guard usage with boundary classification; if reached from core context, call violation recorder.

7. `noesis/pipelines/hooks.py::_dlpack_tensor_to_numpy`
- Same guard: boundary-only conversion; core-path invocation is a violation.

8. `noesis/pipelines/hooks.py::_AnalyticsTelemetryProcessor._tensor_to_embedding`
- Keep conversion allowed only for telemetry boundary; add timing and violation hooks.

9. `noesis/ds8_runtime.py::_build_stats_callback`
- Expose zero-copy metrics and boundary serialization percentiles in `/stats` payload.

10. `noesis/ds8_runtime.py::main`
- Enforce production fail-fast policy on startup when contract validation reports core violations.

### P2 (budget enforcement + tests)
11. `noesis/telemetry/publishers.py::DepthTelemetryPublisher.publish`
- Instrument serialization/dispatch duration into boundary stats.

12. `noesis/telemetry/publishers.py::TrackingTelemetryPublisher.publish`
- Instrument serialization/dispatch duration into boundary stats.

13. `websocket_server.py::_send_to_client`
- Time JSON serialization + send scheduling for boundary budget accounting.

14. `tests/test_zero_copy_invariants.py` (new)
- Add unit coverage for graph invariants (NVMM caps, OSD mode, memory-type consistency, fail-fast behavior).

15. `scripts/zero_copy_smoke_test.py` (new)
- Runtime smoke that asserts:
  - `zero_copy_violations == 0`
  - boundary serialization p99 <= 3.0 ms

## 8. Acceptance Criteria
- A1: `core_zero_copy_violations == 0` during runtime and smoke.
- A2: `boundary_cpu_serialization_p99_ms <= 3.0`.
- A3: Unit tests include zero-copy invariant coverage and pass.
- A4: Smoke script passes on target DS8 runtime.

This document is decision-complete for DS8 core graph memory behavior under the zero-copy cutover.
