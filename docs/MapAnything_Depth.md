# MapAnything Depth (DS8)
_Status: validated against the shared DS8/V3DT/DS9 capture contracts on 2026-07-11._

MapAnything is still a first-class DS8 depth component, but it is no longer the
only depth-related path in the runtime.

## Roles

DS8 currently uses two distinct depth lanes:

- `models.mapanything`
  - Full-frame SGIE branch in the live DS8 pipeline.
  - On-demand and gate-controlled.
  - Owns the `depth_result` WebSocket payload and `get_ma_depth` RPC contract.
  - Persists dense full-frame snapshots under `data/depth/...`.
- `models.depth_tracking`
  - Full-frame Depth Anything V2 metric lane used by baseline non-`v3dt`
    world tracking.
  - Always on in baseline mode.
  - Does not publish a second full-frame depth WebSocket stream.
  - Contributes through `NOESIS.OBJECT_DEPTH` and the fused backend world
    estimator in `noesis/pipelines/hooks.py`.

MapAnything also acts as the offline reference source for the DAv2 room
registration artifact used by baseline tracking.

## Runtime Topology

MapAnything stays on its own DS8 branch:

- `main_tee -> mapanything_queue -> mapanything_valve -> mapanything_fullframe -> mapanything_fullframe_sink`

Config source:

- `config/infer.yaml`
- `models.mapanything.*`
- default `gie_id=2`

Gate control:

- REST: `POST /api/v1/depth/refresh?seconds=N`
- WebSocket RPC: a non-cache `get_ma_depth` or stale/missing
  `get_floorplan` request can open one bounded refresh window
- Runtime: `DS8Pipeline.mark_depth_enabled()` controls `mapanything_valve.drop`

The gate applies only to MapAnything. It does not control the always-on baseline
DAv2 tracking lane. The graph has one process-wide MapAnything valve, so the
depth, floorplan, and calibration capture paths share one
`CaptureEventController`. The controller canonicalizes camera aliases, takes a
non-blocking per-camera admission lock, and then takes a process-wide gate lock.
That second lock prevents two different cameras from claiming the same valve at
once; contention returns the stable `capture_event_busy` error instead of
interleaving capture cohorts.

## Postprocess, Storage, and RPC

Live MapAnything processing is owned by `MapAnythingProcessor` in
`noesis/pipelines/hooks.py`.

Responsibilities:

- decode MapAnything tensor outputs
- align depth/conf/mask to camera frame geometry
- store dense snapshots through `geometry.depth_source.DepthStorageManager`
  with optional `rgb` image layers for fresh dashboard-triggered captures
- wait for the exact transactional write receipt before recording the frame or
  constructing/publishing its `DepthResult`
- fuse raw snapshots from one refresh burst into a single `capture_event_fused`
  Zarr before floorplan, normals, and room reconstruction consume the result
- publish `DepthResult`
- serve `get_ma_depth` via the runtime provider path

The full-frame `depth_result` and `ma_depth_response` contracts remain
MapAnything-specific. They are not reused for the baseline DAv2 tracking lane.

Storage is mandatory for a publishable MapAnything result. The bounded commit
wait is configured by `NOESIS_DEPTH_STORE_COMMIT_TIMEOUT_S` (30 seconds by
default, finite input required, clamped to 0.1–60 seconds). Timeout or write
failure poisons the non-daemon worker, invokes the runtime failure callback,
and emits neither depth telemetry nor a depth-FPS receipt. The retired
`NOESIS_DEPTH_STORE_ENABLED` switch does not enable a memory-only mode, and
there is no synchronous/direct-write fallback under queue or disk failure.

MapAnything postprocessing is an owned non-daemon worker in baseline DS8 and
the protected V3DT reimplementation. It accepts work atomically against
shutdown, preserves jobs captured while the gate was open, drains FIFO work,
surfaces final-job failure, and sentinel-stops/joins before depth storage is
released. Runtime shutdown first rejects and joins blocking REST/WebSocket
providers, stops WebSocket publication, closes the depth gate, and proves
pipeline EOS/wait so no probe can enter; only then does it join this worker.
An unresolved provider, probe, worker, or listener lease is watchdog-fatal and
does not release storage.

MapAnything normals are derived from dense depth and calibration. Runtime
`ma_depth_response` payloads can attach them on demand, and the virtual-twin
builder now computes the same camera-space normal evidence from persisted depth
snapshots before writing revision artifacts. Those normals are used to validate
and score generated planes; they do not replace MapAnything depth, ZeroPlane
masks, or camera-derived plane equations.

Room reconstruction now uses two explicit fusion levels:

- Intra-capture fusion combines the valid, agreeing pixels from the raw Zarrs
  emitted during one MapAnything refresh burst. This creates one capture event
  and stores any captured RGB beside the fused depth.
- Inter-capture fusion combines the latest capture events, defaulting to four,
  into the reconstruction mesh/point artifact that Menon displays.

`noesis_core/capture_event_fusion.py` is the runtime-neutral admission contract
for new intra-capture work. It accepts only typed raw snapshots newer than the
request baseline, rejects duplicate/mixed/overwide cohorts and recursive
derived inputs, seals deterministic source/result evidence, and rejects
`cache_only` before a store or optional frame provider can be touched. RGB is a
typed timestamped optional input from an already-active pipeline; when no such
provider exists, depth-only fusion must be explicitly requested or the result
is `rgb_frame_unavailable`. Opening a second camera reader is not an accepted
RGB path.

DS8, protected V3DT, and DS9 use the same runtime capture sequence:

1. Close the global MapAnything valve, prove the MapAnything worker idle, flush
   the transactional depth-store frontier, and record the camera's last raw
   timestamp.
2. Open one bounded burst, close the valve, and repeat both the worker-idle and
   durable-storage barriers.
3. Admit exactly one raw-only cohort newer than the recorded baseline and write
   one `capture_event_fused` / `intra_capture` snapshot.
4. Revalidate that committed result, then reload it by its portable storage
   reference, write ID, content and manifest digests, sequence, and exact
   timestamp. The response never substitutes the latest snapshot.

The current graph deliberately requests depth-only fusion. The coordinator's
sealed evidence records `rgb.status=not_requested` and the compact response
evidence declares `capture_mode=depth_only`. If a future caller requires RGB
without a reviewed pipeline-owned frame provider, it receives
`rgb_frame_unavailable`; no reader, appsink, or decoded-frame fallback is
opened.

Cache-only requests are separated before capture admission. They may return an
already-valid depth or floorplan cache entry, but they do not open the valve,
wait for a burst, select a cohort, write a fused depth snapshot, generate a new
floorplan, or mutate the active-floorplan registry. A miss is an explicit
`no_cached_depth` or `no_cached_floorplan` response.

## Top-Down Floorplan Artifact

DS8 also creates a per-camera top-down floorplan from the same persisted
MapAnything snapshots. This is the floor-shaped raster used by the depth drawer,
inline BEV floorplan mode, floorplan debug dumps, and downstream spatial
validation/visualization paths.

The floorplan is not a hand-authored room map and it is not the live person-depth
source for baseline tracking. It is a derived diagnostic/reference artifact built
from dense full-frame MapAnything depth, confidence, mask, and the current
calibration bundle.

Creation path:

- WebSocket clients request it with `get_floorplan`; DS8 returns
  `floorplan_response`.
- A `cache_only=true` request asks
  `DepthStorageManager.generate_topdown_floorplan(...)` only for an already
  valid cache entry. A normal request is an explicit fresh capture and never
  performs this generic latest/current lookup first.
- Cache validity is bound to camera, grid resolution, extent, floorplan
  contract version, units, calibration fingerprint, and snapshot identity. An
  exact capture-event cache key includes the fused write ID; a cache record for
  another snapshot cannot satisfy it.
- For every non-cache request, the shared capture controller opens the bounded
  MapAnything burst described above. The provider then generates the floorplan
  from that exact fused snapshot by portable `snapshot_ref`, `snapshot_id`, and
  `snapshot_content_sha256`; a pre-existing raw/latest floorplan cannot satisfy
  a fresh request.
- The result must return the same reference, write ID, content digest, and
  timestamp. An identity or integrity mismatch is an error; the provider does
  not return the earlier stale payload or silently regenerate from "latest."
- Reusable latest-snapshot JSON caches are persisted under
  `<depth_base>/floorplans/<camera>/grid<grid_res>__ext<max_extent>.json`;
  the default `<depth_base>` is `data/depth` from `config/mapanything.ini`, unless
  runtime storage is overridden with `--storage-base`. Exact capture-event
  results publish independent copies under both the bounded write-ID key and
  the explicit cache-only `latest` memory alias, then enter the active registry;
  their fused depth source is independently durable.
- Only a successful, contract-valid response becomes the camera's active
  floorplan. That bounded registry supplies metric bounds, grid shape and
  resolution, calibration fingerprint, exact snapshot identity, and optional
  ray-alignment evidence to the camera-local BEV renderer.

Calibration changes invalidate geometry, not just pixels. Updating one
camera's extrinsics clears that camera's active-floorplan record and floorplan
cache; changing shared alignment clears every active record and in-memory
cache. Persisted cache entries carry a calibration fingerprint and are rejected
after a mismatch. Until a new floorplan is accepted, BEV cannot keep publishing
the superseded active-floorplan geometry.

Generation details:

- Valid depth pixels are back-projected with the camera intrinsics into the
  canonical camera-local frame: `X` is image-right, `Y` is image-down for raw
  camera points, and `Z` is forward depth.
- The floorplan raster is built in a gravity-aligned camera ground frame, not
  Menon scene axes and not BEV screen pixels. `X` is calibrated camera-right
  projected onto the ground plane; `Z` is calibrated camera-forward projected
  onto the ground plane.
- Grid columns increase from `min_x` to `max_x`; row 0 corresponds to `max_z`,
  and rows advance toward smaller `Z`. This is the same convention encoded by
  `frame=camera_local_ground_m` and
  `orientation=camera_ground_right_forward`.
- `image_flip` in `floorplan_response` is diagnostic only. The serialized grids
  already have the correct `X/Z` orientation, so renderers must not mirror or
  rotate the raster again using that hint.
- The stored mask admits every finite positive depth prediction. Model
  ambiguity and pixels outside the calibrated dewarper footprint remain
  explicit uncertainty signals and reduce confidence rather than erasing
  geometry. The stricter combined mask remains separate calibration evidence.

Payload layers:

- `density`: normalized support count per cell.
- `distance`: mean forward depth per cell.
- `height`: weighted height grid, normalized for visualization.
- `height_agl`: height above the estimated floor, with floor bias correction.
- `height_agl_meta.floor_offset_m`: the finite, bounded low-percentile floor
  correction actually subtracted before rasterization. Both floorplan
  generators apply this correction exactly once; `0.0` means no correction was
  warranted.
- `gradient`: normalized height-edge magnitude.
- `obstacle_height` and `walkable`: optional clean layers that classify floor
  versus furniture/obstacles on the same raster surface. These are secondary
  derived layers; the detailed height/AGL raster remains the base diagnostic map.
- `ray_to_floorplan_alignment`: robust fit from rays intersecting the authored
  calibration floor into the depth-derived floorplan frame. Correspondences are
  limited to depth points within 0.06 m of the AGL-corrected observed floor
  (`height_agl_meta.floor_y + floor_offset_m`); arbitrary points that merely
  share a walkable raster cell are not floor-contact evidence. Too few valid
  contacts produces an explicit unavailable fit rather than a substitute.

Debug/validation helpers:

- `python3 scripts/floorplan_rpc_smoke_test.py`
- `python3 scripts/dump_floorplan_views.py --cache-only`

## Offline DAv2 -> MapAnything Registration

Baseline non-`v3dt` world tracking now requires a prebuilt registration artifact
that maps raw DAv2 anchor range into MapAnything-aligned room range.

Canonical pieces:

- builder: `scripts/build_depth_registration.py`
- artifact: `config/depth_registration.json`
- schema/loader: `noesis/calibration/depth_registration.py`
- fitter: `noesis/calibration/depth_registration_builder.py`

Operational rules:

- The artifact is generated offline and loaded read-only by DS8 at startup.
- DS8 does not auto-generate, auto-refresh, or auto-download this artifact.
- Missing or stale entries are a fatal startup error in baseline mode.
- Legacy registration artifacts bind MapAnything by normalized model/profile
  fields, including engine path, name, batch, GIE, tensor-meta behavior, and
  scope. They do not contain an engine-byte hash, and their original fingerprint
  also included the nvinfer config path. Runtime validation therefore ignores
  only the content-addressed engine-only config path and cadence/source
  plumbing while continuing to reject model/engine/input/semantic changes.
  New artifact builders should bind engine and runtime-config content hashes in
  a versioned contract when backward compatibility permits.
- Empty-room RTSP captures are preferred, but the builder now filters samples to
  temporally stable pixels so minor/static occupancy does not automatically
  poison the fit.

Typical workflow:

```bash
bash services/mapanything_svc/run.sh
env CUDA_VISIBLE_DEVICES='' python3 scripts/build_depth_registration.py \
  --output config/depth_registration.json
timeout 25s python3 noesis/ds8_runtime.py --pgie-profile yolo26_seg --size s --disable-rest
```

The builder uses live RTSP sources from `config/infer.yaml` by default.

## MapAnything Service Notes

The local MapAnything service is still an active tool for offline registration
work. The deprecated part is the old live-runtime microservice/adapter depth
path; the canonical runtime depth path is the DS8 SGIE branch.

Current service ownership:

- startup script: `services/mapanything_svc/run.sh`
- app: `services/mapanything_svc/server.py`
- weight pin: `docs/ma-integration/weights.sha`

The service has no built-in RPC key. It reads an authority-grade key from the
owner-only file declared by `config/mapanything.ini` (or
`NOESIS_MAPANYTHING_API_KEY_FILE`) and rejects missing or invalid credentials.
Clients use the same file through `load_service_config`; no key is stored in
tracked config, tests, logs, or command arguments. See `Runtime_Secrets.md` for
provisioning and coordinated rotation.

The service is primarily used to produce reference dense depth for registration
builds, not to replace the DS8 in-pipeline MapAnything branch.

## What MapAnything Does Not Own

MapAnything is not the canonical owner of baseline person world tracking.

It does not:

- own `track.world` in baseline mode
- emit `NOESIS.OBJECT_DEPTH`
- replace the pose-first anchor chain
- act as a runtime fallback for missing DAv2 registration

Baseline room-relative tracking remains:

- pose-first anchor authority
- DAv2 object depth on the same pose ray
- optional offline DAv2 -> MapAnything registration correction
- one fused backend world estimator

## Quick Validation

- `python3 scripts/ma_depth_rpc_smoke_test.py --no-spawn`
  - validates live `ma_depth_response`
- `python3 scripts/floorplan_rpc_smoke_test.py --no-spawn`
  - validates live `floorplan_response`
- `python3 scripts/build_depth_registration.py --help`
  - validates builder surface
- `timeout 25s python3 noesis/ds8_runtime.py --pgie-profile yolo26_seg --size s --disable-rest`
  - proves DS8 can start with the current registration artifact
