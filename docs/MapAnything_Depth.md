# MapAnything Depth (DS8)
_Status: validated against code on 2026-03-16._

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

- REST: `GET /api/v1/depth/refresh?seconds=N`
- WebSocket RPC: `get_ma_depth` can open a short refresh window
- Runtime: `DS8Pipeline.mark_depth_enabled()` controls `mapanything_valve.drop`

The gate applies only to MapAnything. It does not control the always-on baseline
DAv2 tracking lane.

## Postprocess, Storage, and RPC

Live MapAnything processing is owned by `MapAnythingProcessor` in
`noesis/pipelines/hooks.py`.

Responsibilities:

- decode MapAnything tensor outputs
- align depth/conf/mask to camera frame geometry
- store dense snapshots through `geometry.depth_source.DepthStorageManager`
  with optional `rgb` image layers for fresh dashboard-triggered captures
- fuse raw snapshots from one refresh burst into a single `capture_event_fused`
  Zarr before floorplan, normals, and room reconstruction consume the result
- publish `DepthResult`
- serve `get_ma_depth` via the runtime provider path

The full-frame `depth_result` and `ma_depth_response` contracts remain
MapAnything-specific. They are not reused for the baseline DAv2 tracking lane.

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
- `noesis/ds8_runtime.py` handles the request through its floorplan provider and
  calls `DepthStorageManager.generate_topdown_floorplan(...)`.
- The storage manager first checks its in-memory and on-disk floorplan cache.
  Cache validity is keyed by camera, grid resolution, extent, floorplan contract
  version, units, and calibration fingerprint.
- On a cache miss, generation reads the latest valid MapAnything Zarr snapshot
  from `<depth_base>/<camera>/<YYYYMMDD>/<HH>/<timestamp_us>.zarr`.
- If the snapshot is missing or stale and the request is not `cache_only`, the
  DS8 provider may open a short MapAnything depth burst
  (`NOESIS_FLOORPLAN_DEPTH_ENABLE_SECONDS`, falling back to
  `NOESIS_DEPTH_RPC_ENABLE_SECONDS`) and then regenerate from the fresh snapshot.
- The resulting JSON cache is persisted under
  `<depth_base>/floorplans/<camera>/grid<grid_res>__ext<max_extent>.json`;
  the default `<depth_base>` is `data/depth` from `config/mapanything.ini`, unless
  runtime storage is overridden with `--storage-base`.

Generation details:

- Valid depth pixels are back-projected with the camera intrinsics into the
  canonical camera-local frame: `X` is image-right, `Y` is image-down for raw
  camera points, and `Z` is forward depth.
- The floorplan raster is built in camera-local ground `X/Z`, not Menon scene
  axes and not BEV screen pixels.
- Grid columns increase from `min_x` to `max_x`; row 0 corresponds to `max_z`,
  and rows advance toward smaller `Z`. This is the same convention encoded by
  `frame=camera_local_ground_m` and `orientation=camera_xz_forward`.
- `image_flip` in `floorplan_response` is diagnostic only. The serialized grids
  already have the correct `X/Z` orientation, so renderers must not mirror or
  rotate the raster again using that hint.
- Confidence and mask are used as soft weights. They do not hard-drop every
  masked or low-confidence point, which keeps the raster useful in sparse or
  partially occluded rooms.

Payload layers:

- `density`: normalized support count per cell.
- `distance`: mean forward depth per cell.
- `height`: weighted height grid, normalized for visualization.
- `height_agl`: height above the estimated floor, with floor bias correction.
- `gradient`: normalized height-edge magnitude.
- `obstacle_height` and `walkable`: optional clean layers that classify floor
  versus furniture/obstacles on the same raster surface. These are secondary
  derived layers; the detailed height/AGL raster remains the base diagnostic map.
- `ray_to_floorplan_alignment`: diagnostic fit between calibrated floor-contact
  rays and the depth-derived floorplan frame.

Debug/validation helpers:

- `python3 scripts/floorplan_rpc_smoke_test.py`
- `python3 scripts/dump_floorplan_views.py --cache-only`

## Offline DAv2 -> MapAnything Registration

Baseline non-`v3dt` world tracking now requires a prebuilt registration artifact
that maps raw DAv2 anchor range into MapAnything-aligned room range.

Canonical pieces:

- builder: [build_depth_registration.py](/home/mayor/Noesis_Devel/scripts/build_depth_registration.py)
- artifact: [depth_registration.json](/home/mayor/Noesis_Devel/config/depth_registration.json)
- schema/loader: [depth_registration.py](/home/mayor/Noesis_Devel/noesis/calibration/depth_registration.py)
- fitter: [depth_registration_builder.py](/home/mayor/Noesis_Devel/noesis/calibration/depth_registration_builder.py)

Operational rules:

- The artifact is generated offline and loaded read-only by DS8 at startup.
- DS8 does not auto-generate, auto-refresh, or auto-download this artifact.
- Missing or stale entries are a fatal startup error in baseline mode.
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
