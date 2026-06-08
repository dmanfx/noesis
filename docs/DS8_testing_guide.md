# DS8 Testing & Validation Guide
_Status: current as of 2026-03-16._

This guide describes how to validate DS8 changes using the current runtime and contracts.

For calibration, BEV, tracking-world, virtual-twin, or Menon-facing spatial
changes, also consult `plans/noesis_menon_validation/`. That workstream defines
the layered validation toolbox, report schema, cross-space acceptance gates, and
agent instructions for proving the full chain from camera pixels to Menon
reprojection.

Validation tiers:

- Tier 1: unit, schema, and math checks with no GPU or runtime.
- Tier 2: offline fixture and regression reports with visual artifacts.
- Tier 3: DS8 runtime or saved WebSocket telemetry from `noesis/ds8_runtime.py`.
- Tier 4: DS8 plus Menon cross-space traces or browser evidence.

See `plans/noesis_menon_validation/validation_tiers.md` and
`plans/noesis_menon_validation/validation_asset_inventory.md` for command
selection and reusable existing smoke tests.

The first reusable fixture runner is available for GPU-free checks:

```bash
python3 scripts/noesis_validation_runner.py \
  --fixture-registry plans/noesis_menon_validation/fixture_registry.json \
  --fixture-id minimal_validation_fixture
python3 -m pytest tests/test_validation_toolbox.py -q
```

The runner writes JSON, Markdown, and visual artifacts under
`diagnostics/validation/<run_id>/`.
The fixture report now includes generated-scene coordinate-system checks and
room geometry-constraint checks when the fixture provides those fields.

For virtual-twin plane reconstruction changes that touch MapAnything normals,
run the focused unit/bundle checks before live Menon validation:

```bash
python3 -m pytest \
  tests/test_mapanything_normals_fusion.py \
  tests/test_virtual_twin_geometry.py \
  tests/test_virtual_twin_builder.py \
  -q
```

These tests prove that dense MapAnything depth-derived normals are persisted in
revision evidence, surfaced in `planes.json`, accepted when they agree with the
fitted stream plane, and rejected when they clearly contradict the plane.

Saved or live WebSocket telemetry can also be validated through the same report
schema:

```bash
python3 scripts/noesis_validation_telemetry_report.py \
  --input plans/noesis_menon_validation/minimal_telemetry.ndjson \
  --run-id minimal_telemetry

# Against a running DS8 WebSocket server:
python3 scripts/noesis_validation_telemetry_report.py \
  --ws ws://127.0.0.1:6008 \
  --duration 20 \
  --run-id live_tracking_bev_window
```

This emits tracking/BEV contract checks, motion checks, BEV/track agreement, a
copied telemetry NDJSON, and `tracking/track_audit.json` under the run directory.
The audit includes identity, current room, named world position, projection,
temporal, and ReID confidence, doorway/occlusion history, impossible-motion
events, speed summary, and warnings. The telemetry report also validates
occlusion bridges when `occluded` samples include uncertainty evidence.

Captured Menon placement traces can be validated against the same report schema:

```bash
python3 scripts/noesis_validation_menon_trace_report.py \
  --trace plans/noesis_menon_validation/minimal_menon_trace.json \
  --run-id minimal_menon_trace

# When a live Menon checkout is required for acceptance evidence:
MENON_ROOT=../Menon python3 scripts/noesis_validation_menon_trace_report.py \
  --trace plans/noesis_menon_validation/minimal_menon_trace.json \
  --require-menon-root \
  --run-id minimal_menon_trace_with_checkout
```

This emits world-to-Menon transform, placement, floor-contact, trail-agreement,
BEV-to-Menon trail agreement, avatar scale/collision/orientation, timestamp, and
transform-audit checks. If Menon is required but no valid checkout is provided,
the report records blocked Menon evidence instead of treating a Noesis-only run
as complete.

When Menon is running in a browser, capture its live debug state and validate it
through the same Menon trace contract:

```bash
python3 scripts/noesis_validation_capture_menon_trace.py \
  --url http://127.0.0.1:5173 \
  --output diagnostics/validation/menon_browser_trace/trace.json \
  --screenshot \
  --validate
```

This writes the raw browser snapshot beside the converted trace, optionally
captures a screenshot, and runs the Menon trace report. Missing Playwright,
unreachable Menon, missing browser debug globals, missing declared
`backend_world_m` evidence, or a missing world-to-scene transform is a failed or
blocked validation, not a substitute fixture pass.

To run the registered fixtures as a small regression suite:

```bash
python3 scripts/noesis_validation_regression_runner.py \
  --fixture-registry plans/noesis_menon_validation/fixture_registry.json \
  --run-id minimal_regression
```

This writes `regression_summary.json` plus per-case reports under
`diagnostics/validation/<run_id>/`. The registry can declare expected status,
minimum check counts, and maximum failure/warning/blocked counts; expectation
misses are reported as regression failures.

## 1. Quick Sanity Checks

### DS8 runtime smoke test

Once `noesis/ds8_runtime.py` and `config/infer.yaml` are wired:

```bash
python3 noesis/ds8_runtime.py \
  --pipeline-config config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --enable-rest
```

YOLO26 segmentation profile (size defaults to `m` when omitted):

```bash
python3 noesis/ds8_runtime.py \
  --pipeline-config config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --pgie-profile yolo26_seg \
  --size m
```

Notes:

- The YOLO26 DS8 profile uses fused assets with a single output tensor (`output0`) and `disable-output-host-copy=1` to keep mask work on-GPU:
  `models/yolo26{n,s,m}-seg_fused.onnx` and `models/engines/yolo26{n,s,m}-seg_fused_b3_fp16.engine`.
- The fused YOLO26 engines compose masks for the top 30 detections (matching YOLO11) to keep ROIAlign cost bounded.
- Performance note: ROIAlign cost scales with the number of detections composed. With the unfused/top-300 path, `roi_align_proto` dominated YOLO26n (~18.6 ms) and drove very high GPU utilization; after capping to 30 detections, `roi_align_proto` dropped to ~3.8 ms and YOLO26n GPU usage in the DS8 pipeline dropped dramatically (reported ~10% on the nano model; exact numbers depend on stream FPS and host/GPU).

Check logs for:

- Successful DS8 pipeline build (no missing elements or config errors).
- WebSocket server startup on the configured host/port.
- REST server startup (if enabled).
- Mosaic WebRTC toggles are consumed **at build time**: set `NOESIS_MOSAIC_WEBRTC_ENABLED` before starting the runtime so the RTSP output and WebRTC gateway are enabled (WebRTC will auto-enable RTSP if requested).
- If MapAnything SGIE gating is enabled, expect a startup log like `MapAnything gate primed; closed valve after 1.00s` (tunable via `NOESIS_MAPANYTHING_GATE_PRIME_SECONDS`) to confirm the SGIE branch prerolls and then closes when depth is disabled.
- Baseline non-`v3dt` startup now also requires a valid depth-registration artifact (`depth_registration.path`, default `config/depth_registration.json`). DS8 should fail fast before activation if the artifact is missing or stale for any enabled camera.

**Important:** All DS8 tests should also use the real RTSP streams defined in configuration files:

- Prefer `config/infer.yaml` `sources` entries for DS8.

Do **not** rely on environment variables to specify URIs when adding or testing DS8 features; instead, update the appropriate config file so behavior is fully driven by configuration.

### Build or refresh the depth-registration artifact

Baseline pose+depth world tracking depends on a prebuilt DAv2->MapAnything
registration artifact. The canonical operational flow is:

```bash
bash services/mapanything_svc/run.sh
env CUDA_VISIBLE_DEVICES='' python3 scripts/build_depth_registration.py \
  --output config/depth_registration.json
```

Notes:

- The builder uses the live RTSP sources from `config/infer.yaml` by default.
- Empty-room captures are preferred, but the builder now filters to temporally
  stable pixels so minor/static occupancy does not automatically invalidate the
  fit.
- Rebuild `config/depth_registration.json` when the registration inputs change:
  camera intrinsics or image size. Pose-only extrinsics edits, floor-height
  edits, and scene-unit updates do not invalidate the registration mapping
  because the fit is image-space DAv2 depth vs MapAnything reference depth.
- The runtime does not generate this artifact automatically.

### Registration-backed runtime smoke

After the artifact is built:

```bash
timeout 25s python3 noesis/ds8_runtime.py --pgie-profile yolo26_seg --size s --disable-rest
```

Expected:

- DS8 starts cleanly without a depth-registration startup error.
- `mapanything_fullframe` and `depth_tracking_fullframe` both load.
- The runtime enters the main loop and reaches timeout without registration
  validation failures.

### StableID pose integration (unit)

```bash
python3 -m pytest tests/test_stable_id_manager_pose.py
```

Notes:
- Rebuild `noesis_pose_meta_ext` before runtime tests that read pose meta.

## 2. REST Endpoint Validation

### Depth API

With DS8 runtime running:

```bash
curl "http://127.0.0.1:8080/api/v1/depth/refresh?seconds=20"
```

If the standalone virtual-twin artifact API is already using `8080`, run DS8
with `--rest-port 8082` and use
`http://127.0.0.1:8082/api/v1/depth/refresh?seconds=20` for this check.

Expect a JSON payload matching `DepthRefreshResponse` and see depth bursts (depth telemetry and stored maps) during the enabled window.

### Analytics ROI API

Use the endpoints defined in `noesis/server/analytics_api.py` (paths may vary; adjust as needed):

- GET ROI config for a stage.
- POST ROI updates for one or more streams.

Verify that:

- Analytics config file is updated.
- Exclusion polygons in DS8 reflect changes (e.g., objects inside new ROIs are pruned).

## 3. WebSocket Validation

With DS8 runtime active, connect a WS client (your UI or a small script) to the configured WS host/port.

Check for:

- `stats` messages at regular intervals with fields `prepared`, `activated`, `depth_enabled`, `depth_fps`, `errors`.
- Mosaic video rendered in the UI via WebRTC (not WebSocket JPEG frames).
- BEV frames and overlays updating in response to BEV config/overlay messages.
- Depth (`depth_result`) and tracking (`tracking`) messages present and well-formed.

### Depth drawer RPC gates (no log grepping)

These scripts exercise the WS RPC contracts used by `DepthDrawer` in `oai2-fe/`:

```bash
python3 scripts/ma_depth_rpc_smoke_test.py
python3 scripts/floorplan_rpc_smoke_test.py
```

Expected:

- `ma_depth_rpc_smoke_test.py` prints `[PASS]` and proves both cache-first and fresh depth retrieval (`served_from_cache=false` on the fresh call, and `ts_us` increases).
- `floorplan_rpc_smoke_test.py` prints `[PASS]` and returns a non-error `floorplan_response` with required fields (density/height/distance layers).
- These RPCs validate only the gated MapAnything lane. Baseline DAv2 tracking
  depth is observed through `tracking`, `stats.payload.cameras[*].tracking.active_tracks[]`,
  and the on-screen `z=` label.

### SV3DT/MV3DT 3D meta (tracking payload)

This checks that `bbox3d` appears in tracking telemetry when SV3DT is enabled:

```bash
python3 scripts/sv3dt_meta_smoke_test.py
# Live RTSP validation:
# python3 scripts/sv3dt_meta_smoke_test.py --pipeline-config config/infer_v3dt_baseline.yaml
# then run DS8 with --tracking-mode v3dt and live RTSP sources in that config
```

Expected:

- `[PASS]` after 3D tracking is active (BodyPose3DNet assets + SV3DT tracker config required).
  - Tracking mode must be `v3dt` (the smoke test defaults `NOESIS_TRACKING_MODE=v3dt`).
  - The script defaults to `config/infer_v3dt_sample.yaml` (offline Retail02 clip) when present to avoid depending on live camera occupancy.
  - When testing live RTSP (`config/infer_v3dt_baseline.yaml` with real sources), ensure a person is visible; otherwise the script may report “no tracking messages observed”.

### V3DT Forensics Toolkit (Snapshot + Telemetry + Panel)

Use this when you need concrete calibration math + per-frame tracking metrics in one place:

```bash
# 1) Snapshot calibration + camInfo math
python3 scripts/v3dt_forensics.py snapshot --pipeline-config build/effective_pipeline_yolo11_seg.yaml

# 2) Enable per-frame tracking log (NDJSON) while DS8 runs
export NOESIS_V3DT_DIAG_LOG=1
export NOESIS_V3DT_DIAG_DIR=diagnostics
python3 noesis/ds8_runtime.py --tracking-mode v3dt

# 3) Analyze the log (optionally pass the snapshot)
python3 scripts/v3dt_forensics.py analyze --log diagnostics/v3dt_frames_<session>.ndjson \\
  --snapshot diagnostics/v3dt_snapshot_<timestamp>.json

# 4) Generate the HTML panel
python3 scripts/v3dt_forensics.py panel --snapshot diagnostics/v3dt_snapshot_<timestamp>.json \\
  --report diagnostics/v3dt_report_<timestamp>.json
```

See `docs/DS8_v3dt_forensics.md` for details.

### Auto-calibration RPC (wiring + contract)

```bash
python3 scripts/auto_calibrate_rpc_smoke_test.py
```

Expected:

- `[PASS]` output with a valid `auto_calibrate_result` message (contract only; calibration may still fail if depth is unavailable).
- `error` is not `no_handler` (regression guard).
- By default the script requests a non-existent camera id to avoid persisting to `config/camera_calibration.json`; to exercise the real calibrate-all path use `--calibrate-all` (may persist extrinsics).

Config:

- `NOESIS_AUTOCALIB_ENABLE_SECONDS` controls the depth burst when auto-calibration needs to open the depth gate (default: 10, clamped to 1–15).

### WebRTC mosaic (RTSP → WebRTC gateway)

When `NOESIS_MOSAIC_RTSP_ENABLED=1` and `NOESIS_MOSAIC_WEBRTC_ENABLED=1`, DS8 starts an RTSP mosaic output and a separate RTSP→WebRTC gateway (`noesis/mosaic_webrtc_gateway.py`) that negotiates WebRTC over the existing WebSocket server.

ICE configuration (server-side `webrtcbin`):

- `NOESIS_MOSAIC_WEBRTC_STUN_SERVER` (default: `stun://stun.l.google.com:19302`)
- `NOESIS_MOSAIC_WEBRTC_TURN_SERVER` (optional; can include credentials, so avoid logging it)

Headless gate (proves SDP + RTP + decode):

```bash
NOESIS_MOSAIC_RTSP_ENABLED=1 NOESIS_MOSAIC_WEBRTC_ENABLED=1 python3 noesis/ds8_runtime.py --log-level INFO
python3 scripts/webrtc_gateway_smoke_test.py --ws ws://127.0.0.1:6008 --duration 5 --pt 103
```

Browser-offer replay gate (SDP-level; no ICE/DTLS required):

```bash
python3 scripts/webrtc_gateway_browser_offer_replay_test.py --ws ws://127.0.0.1:6008 --offer .cursor/webrtc_offers/offer_*.sdp
```

Expected:

- Smoke test JSON includes `answer_video_direction="sendonly"`, `rtp_packets>0`, `decoded_frames>0`.
- Replay test asserts the returned answer is `sendonly` (guards against `a=inactive` answers for browser offers).

If the dashboard shows ICE “connected” and `bytesReceived>0` but `framesDecoded=0` / `videoWidth=0`, the peer likely hasn’t received a decodable IDR yet (or the IDR was corrupted by packet loss). Mitigations:

- Ensure the gateway is allowed to wait for the first RTSP frame before answering (it refuses to answer if RTSP has not produced any frames yet).
- Ensure the RTSP output produces frequent IDR frames (see `config/infer.yaml` `mosaic_output.rtsp_iframeinterval` / `mosaic_output.rtsp_idrinterval`).
- If this persists, check `.cursor/debug.log` for the gateway’s `gateway first keyframe` / `keyframes` counters; receiving RTP bytes with no keyframes strongly suggests the receiver joined mid-GOP or keyframes are being lost.

### Mosaic aspect sanity (avoid stretched tiles)

The DS8 pipeline logs a structured tiler configuration event during build:

- `{"event":"ds8_mosaic_tiler_config", ... "square_seq_grid": true, ...}` in logs (and `.cursor/debug.log` when JSON logging is enabled).

Default behavior enables `nvmultistreamtiler square-seq-grid=true` to preserve per-tile aspect within a fixed 1920×1080 mosaic output. Override only if you explicitly want a non-square layout:

- Disable square tiling: `NOESIS_MOSAIC_TILER_SQUARE_SEQ_GRID=0`
- Explicit layout: `NOESIS_MOSAIC_TILER_COLUMNS=<N>` and/or `NOESIS_MOSAIC_TILER_ROWS=<N>`

## 4. Cross-Run Consistency Checks

For a curated set of test streams, run repeated DS8 sessions and capture telemetry for offline comparison.

### Suggested approach

1. Run DS8 session A for N seconds on a fixed set of streams and record:
   - WebSocket telemetry (e.g., via a client that logs `stats`, `tracking`, `depth_result`, `bev-frame`).
   - Any key logs about analytics and occupancy.
2. Run DS8 session B on the same streams and record the same data.
3. Compare:
   - Object counts and classes per frame (tolerate small differences if model configs differ, but investigate large discrepancies).
   - Zone occupancy over time.
   - Depth min/max distributions and burst timing.
   - BEV overlays/footpoints (visual and numeric.

## 5. Focused Component Tests

When changing a specific DS8 component, perform focused tests:

- **Pipeline config (`ds8_pipeline` / `infer.yaml`):**
  - Verify DS8 pipeline can start and process frames with no errors.
  - Check logs for any configuration warnings (paths, batch size, unique IDs).

- **Hooks (`noesis/pipelines/hooks.py`):**
  - Add temporary debug logging to confirm calibration-bundle intrinsics/extrinsics, MapAnything tensors, analytics user meta, and exclusion pruning are functioning.

- **Telemetry (`noesis/telemetry/*`):**
  - Confirm WS messages have the expected `type` and field structure.

- **REST APIs (`noesis/server/*`):**
  - Exercise endpoints manually with `curl` or HTTP clients and validate error handling.

### MapAnything engine rebuild (ONNX → TensorRT)

If MapAnything depth outputs become sparse/mostly-zero (e.g., black heatmap), rebuild the ONNX export and TensorRT engine using the DS8-compatible `forward()` export path.

Export ONNX (torch-only; outputs `depth/conf/mask`):

```bash
python3 export_ma_onnx/export_to_onnx.py --repo external/map-anything --outdir /tmp/ma_onnx_out_forward --h 294 --w 518
```

Build TensorRT engine (this repo’s DS8 config uses `batch_size=3`, so the engine is built fixed-batch=3):

```bash
trtexec --onnx=/tmp/ma_onnx_out_forward/model.onnx --bf16 \
  --minShapes=images:3x3x294x518 --optShapes=images:3x3x294x518 --maxShapes=images:3x3x294x518 \
  --saveEngine=models/mapanything_depth/1/model.plan
```

Notes:

- If you change `config/infer.yaml` `batch_size` or the number of sources, rebuild the engine with matching shapes.
- TensorRT 10.13 has hit a Myelin internal error when attempting dynamic batch 1–3 for this model; fixed batch=3 is the validated path here.

## 6. Automation & CI

If you introduce automated tests (e.g., pytest), follow these principles:

- Keep tests narrowly focused on DS8 logic (metadata parsing, config normalization, simple dry-run pipeline constructs) that can run without GPUs.
- Avoid introducing heavyweight integration tests that require full DeepStream runtime unless the CI environment explicitly supports it.
