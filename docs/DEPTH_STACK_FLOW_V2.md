# Depth stack flow
_Status: canonical native DS9.1 depth topology, updated 2026-08-15._

DS9.1 has two GPU-first full-frame depth branches: always-on DAv2 supplies
tracking observations, while gated MapAnything supplies dense snapshots, RPCs,
and floorplans. Earlier MapAnything-only flowcharts are archived under
`docs/history/`.

## High-Level Flow

```mermaid
flowchart LR
  SRC[nvurisrcbin (per source)] --> MUX[nvstreammux]
  MUX --> PRE[nvdspreprocess]
  PRE --> PGIE[nvinfer (PGIE)]
  PGIE --> TEE[main_tee]
  TEE --> TRK[nvtracker] --> ANA[nvdsanalytics] --> REID[ReID / pose SGIEs] --> WORLD[world_observation_stage]
  WORLD --> TRACK[tracking_telemetry_stage]
  TEE --> DTQ[depth_tracking_queue] --> DAV2[nvinfer (DAv2 full-frame)] --> DTFS[fakesink]
  TEE --> MAQ[mapanything_queue] --> MAV[mapanything_valve] --> MASGIE[nvinfer (MapAnything SGIE)] --> MAFS[fakesink]

  subgraph Hooks
    DAV2 -. device tensor meta .-> CAPTURE[DAv2 GPU align + exact-frame store]
    WORLD -. object meta .-> FUSION[bounded exact-frame rendezvous + person ROI sampling]
    CAPTURE --> FUSION
    FUSION --> ODM[NOESIS.OBJECT_DEPTH]
    ODM --> PERSON[PersonGroundState + strict observation]
    MASGIE -. tensor meta .-> MAP[MapAnythingProcessor → DepthResult + Zarr]
  end

  PERSON --> TRACKING[tracking + world telemetry]
  MAP --> ZARR[(data/depth/<cam>/<date>/<ts>.zarr)]
  MAP --> DEPTH_RES[depth_result telemetry]
  MAP --> WS_RPC[get_ma_depth RPC]
  ZARR --> FLOORPLAN[get_floorplan RPC -> floorplan_response]
  FLOORPLAN --> FPCACHE[(data/depth/floorplans/<cam>/grid...json)]
```

## Runtime Touchpoints
- **DAv2 cadence:** the active baseline runs the
  `models.depth_tracking` branch continuously. V3DT disables it because the 3D
  tracker owns that observation.
- **Exact-frame rendezvous:** the capture and fusion operators key on exact
  source/frame/PTS. Fusion waits for
  `NOESIS_OBJECT_DEPTH_EXACT_FRAME_WAIT_MS` (20 ms default, 250 ms hard cap),
  then may consume only an admissible same-source prior frame within cadence.
- **Object depth:** the native bridge attaches `NOESIS.OBJECT_DEPTH` only when
  attachment succeeds. A non-`ok` object-local sample may still be attached for
  diagnostics; native rejection/unavailability means no meta and a failure
  counter.
- **Canonical observation:** `depth_present` is true only for `status="ok"`
  with finite positive usable depth and no registration rejection. A raw anchor
  is not sufficient when registration status is `ok`.
- **MapAnything gating:** `mapanything_valve.drop` toggles through
  `DS8Pipeline.mark_depth_enabled()`; REST `/api/v1/depth/refresh` and
  `get_ma_depth` open the gate for a bounded window.
- **MapAnything postprocess:** `MapAnythingProcessor` aligns depth/conf/mask,
  records `DepthResult`, and stores snapshots through `DepthStorageManager`.
- **Telemetry:** only MapAnything emits full-frame `type:"depth_result"`.
  DAv2 contributes object metadata, track diagnostics, strict observations, and
  world state instead of a duplicate dense stream.
- **RPCs:** `get_ma_depth` returns the latest MapAnything snapshot (optionally
  with normals); `get_floorplan` derives top-down grids from stored depth and
  the calibration bundle.
- **Floorplan:** `get_floorplan` is served by `DepthStorageManager.generate_topdown_floorplan(...)`. It creates a per-camera `camera_local_ground_m` X/Z raster from the latest MapAnything snapshot, caches the result under `<depth_base>/floorplans/<camera>/`, and returns density, distance, height, `height_agl`, gradient, and optional clean `obstacle_height`/`walkable` layers.

## Operational Counters

`stats.payload.pipeline.zero_copy_core.counters` includes:

- exact-frame flow: `depth_bridge_put_total`,
  `depth_bridge_exact_resolve_total`, `depth_bridge_wait_total`,
  `depth_bridge_wait_timeout_total`
- bounded fallback: `depth_bridge_lagged_resolve_total`,
  `depth_bridge_lagged_age_frames_total`,
  `depth_bridge_lagged_age_us_total`, `depth_bridge_miss_total`
- device/object work: `depth_tracking_device_frames_total`,
  `object_depth_gpu_roi_copies_total`
- metadata outcome: `object_depth_attach_total`,
  `object_depth_status_total.<status>`, `object_depth_attach_failure_total`,
  `object_depth_attach_failure_total.<reason>`

Successful status counters advance only after native user-meta attachment
succeeds. Missing extensions/functions, native exceptions, and native rejection
are visible failures, with rate-limited warnings.

## Storage & Calibration
- Snapshots: `data/depth/<camera>/<YYYYMMDD>/<HH>/<timestamp_us>.zarr` (Blosc/Zarr, managed by `DepthStorageManager`).
- Floorplans: `data/depth/floorplans/<camera>/grid<grid_res>__ext<max_extent>.json` by default. The cache is invalidated by floorplan contract, units, and calibration fingerprint.
- Calibration: `calibration-bundle` (intrinsics/extrinsics/align) is served on connect and cached on the storage manager for world projections and floorplans.
- Registration: baseline startup loads the read-only DAv2→MapAnything
  registration artifact and validates every active camera before activation.
- Floorplan grid convention: columns increase in camera-local `X`; row 0 is farthest forward `Z`; rows advance toward the camera. The `image_flip` field is diagnostic-only and should not be reapplied by renderers.

## Related Docs
- `MapAnything_Depth.md` – pipeline branch, gating, normals, env toggles
- `metadata_contracts.md` – `DepthResult`, registration, and
  `NOESIS.OBJECT_DEPTH` schemas
- `api_contracts_ws.md` – `ma_depth_response` / `floorplan_response` contracts
