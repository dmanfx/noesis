# DS8 Migration Knowledge Base (for Planner/Debugger Agents)
_Status: current as of 2026-03-16._

This is the compressed, current-state summary for the active DS8 runtime. Use
it as an orientation document, not as a substitute for `plans/DS8/*.md` or the
current code.

## 1. Canonical Scope

- Legacy stack:
  - `deepstream_video_pipeline.py`
  - retained only for history or explicit maintenance
- DS8 canonical stack:
  - `noesis/ds8_runtime.py`
  - `noesis/pipelines/ds8_pipeline.py`
  - `noesis/pipelines/hooks.py`
  - `noesis/telemetry/*`
  - `noesis/server/*`
  - `noesis/metadata/*`

Non-negotiables:

- Do not route DS8 failures through legacy runtime paths.
- Do not invent hidden fallback paths.
- Keep decode, preprocess, inference, tracking, analytics, OSD, and primary
  transport on the GPU wherever DeepStream supports it.

## 2. Current DS8 Runtime Topology

Baseline non-`v3dt` graph shape is now:

- `sources -> streammux -> preprocess -> PGIE -> main_tee`
- main branch:
  - `analytics_exclude -> tracker -> analytics -> reid -> pose`
  - `world_observation_stage -> tracking_telemetry_stage -> tiler -> osd -> sink_tee`
- always-on baseline depth branch:
  - `depth_tracking_queue -> depth_tracking_fullframe -> depth_tracking_fullframe_sink`
- gated MapAnything branch:
  - `mapanything_queue -> mapanything_valve -> mapanything_fullframe -> mapanything_fullframe_sink`
- sink branch:
  - RTSP mosaic for the WebRTC gateway
  - BEV is produced from telemetry, not a second video branch

The important ownership split is:

- baseline world tracking uses the always-on DAv2 lane
- MapAnything remains the gated full-frame RPC/depth/floorplan lane

## 3. Depth Architecture

There are three distinct depth contracts now.

### 3.1 MapAnything full-frame depth

Owner:

- `MapAnythingProcessor`
- `DepthStorageManager`
- `DepthTelemetryPublisher`

Purpose:

- `depth_result`
- `ma_depth_response`
- floorplan generation
- offline reference source for registration

It is gate-controlled and does not directly own `track.world`.

### 3.2 DAv2 object depth

Owner:

- baseline DAv2 tracking lane
- `NOESIS.OBJECT_DEPTH`

Purpose:

- attach raw per-object depth stats and anchor-band support
- provide the concurrent range observation used by the baseline world estimator

Important rule:

- raw `NOESIS.OBJECT_DEPTH` stays raw
- runtime registration does not overwrite `anchor_depth_m`

### 3.3 DAv2 -> MapAnything registration artifact

Owner:

- offline builder `scripts/build_depth_registration.py`
- runtime loader `noesis/calibration/depth_registration.py`

Purpose:

- map raw DAv2 anchor range into MapAnything-aligned room range before
  projection on the pose ray

Important rules:

- artifact is read-only at runtime
- baseline non-`v3dt` startup fails fast without valid entries for all enabled
  cameras
- MapAnything is an offline reference source here, not a runtime fallback

## 4. Baseline World Tracking

Current baseline world tracking is a single fused estimator.

Current-frame anchor authority:

- pose-derived image anchor when available (posture-aware: ankles standing, hip/body sitting/lying)
- otherwise the person mask/depth image anchor from `NOESIS.OBJECT_DEPTH.anchor_uv`
- bent-leg `pose_leg_floor` extrapolation is rejected; sticky source hysteresis reduces thrash

Concurrent observations:

- floor-plane observation from that current anchor
- DAv2 depth observation from that same current-anchor ray

Per-track world update states:

- `pose_depth_fused`
- `pose_floor_only`
- `person_anchor_depth_fused`
- `person_anchor_floor_only`
- `gravity_drop` (upright height-lock / lower-body occlusion only; not sit/lie)
- `anchor_hold`

Human pathing (2026-07-08):

- Shared module: `noesis/telemetry/person_ground_state.py`
- Stationary lock (`motion_mode=idle|sit|lie`) freezes world and sets
  `trail_append_allowed=false` so BEV + OSD trails stop scribbling in place
- Human CV filter owns track-position filtering once (adaptive Q/R, ~4 m/s gate,
  idle deadzone); default world max speed is human-scale, not the old 120 m/s open gate
- Path history: min-step + RDP on committed samples only

Current downstream ownership:

- backend owns canonical `track.world`
- BEV world mode consumes canonical `track.world`
- BEV does not run a second world-space smoother on world-mode head points
- world-mode BEV skips `anchor_hold` heads/trails to avoid stale drift
- BEV + OSD trails honor `trail_append_allowed` from the shared ground state

## 5. Offline Registration Build

Canonical workflow:

```bash
bash services/mapanything_svc/run.sh
env CUDA_VISIBLE_DEVICES='' python3 scripts/build_depth_registration.py \
  --output config/depth_registration.json
timeout 25s python3 noesis/ds8_runtime.py --pgie-profile yolo26_seg --size s --disable-rest
```

Operational notes:

- Builder uses RTSP sources from `config/infer.yaml` by default.
- Empty-room captures are preferred.
- The builder now filters to temporally stable pixels, which lets it tolerate
  minor/static occupancy instead of blindly fitting moving foreground pixels.
- The MapAnything service is still active tooling for offline registration work.
  The deprecated part is the old live-runtime microservice depth path.

## 6. Observability Surfaces

Current depth/world observability is intentionally split:

- `depth_result`
  - MapAnything full-frame only
- `tracking.tracks[]`
  - canonical world output + DAv2 registration diagnostics
- `stats.payload.cameras[*].tracking.active_tracks[]`
  - same per-track depth fields for the current camera
- OSD `z=` label
  - uses the same `depth_used_m` value that the estimator actually projected

Key per-track depth fields:

- `depth_anchor_m`
- `depth_used_m`
- `depth_registered_m`
- `depth_registration_status`
- `depth_registration_id`
- `depth_anchor_sample_count`
- `depth_anchor_valid_fraction`

## 7. What Is Still Separate

- `v3dt` remains its own tracking mode.
- MapAnything RPC/floorplan behavior remains separate from baseline tracking.
- `NOESIS.OBJECT_DEPTH` diagnostic world fields remain non-authoritative in the
  main runtime once the fused estimator is active.

## 8. How To Use This File

Before changing DS8 runtime behavior:

1. Read `docs/DS8_README_FOR_AGENTS.md`.
2. Read `plans/DS8/ds8_master_work_orders.md`.
3. Read the relevant checklist under `plans/DS8/`.
4. Use this file only as the short current-state map.

If this file disagrees with the code or current checklists, update it.
