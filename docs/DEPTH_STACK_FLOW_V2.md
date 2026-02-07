# Depth Stack Flow (DS8)
_Status: validated against code on 2026-02-02._

The DS8 depth path is entirely on-GPU and driven by the MapAnything full-frame SGIE branch. Earlier flowcharts are archived under `docs/history/`.

## High-Level Flow

```mermaid
flowchart LR
  SRC[nvurisrcbin (per source)] --> MUX[nvstreammux]
  MUX --> PRE[nvdspreprocess]
  PRE --> PGIE[nvinfer (PGIE)]
  PGIE --> TEE[main_tee]
  TEE --> TRK[nvtracker] --> ANA[nvdsanalytics]
  TEE --> MAQ[mapanything_queue] --> MAV[mapanything_valve] --> MASGIE[nvinfer (MapAnything SGIE)] --> MAFS[fakesink]

  subgraph Hooks
    MASGIE -. tensor meta .-> MAP[MapAnythingProcessor → DepthResult + Zarr]
    ANA -. metadata .-> TELE[Analytics/Tracking Telemetry]
  end

  MAP --> ZARR[(data/depth/<cam>/<date>/<ts>.zarr)]
  MAP --> DEPTH_RES[depth_result telemetry]
  MAP --> WS_RPC[get_ma_depth RPC]
  ZARR --> FLOORPLAN[get_floorplan RPC]
```

## Runtime Touchpoints
- **Gating:** `mapanything_valve.drop` toggled via `DS8Pipeline.mark_depth_enabled()`; REST `/api/v1/depth/refresh` and `get_ma_depth` open the gate for a bounded window.
- **Postprocess:** `MapAnythingProcessor` aligns depth/conf/mask to frame size, records `DepthResult`, and stores snapshots through `DepthStorageManager` (async by default).
- **Telemetry:** `DepthResult` is broadcast on WebSocket (`type: depth_result`).
- **RPCs:** `get_ma_depth` returns the latest snapshot (optionally with normals); `get_floorplan` derives top-down grids from stored depth + calibration bundle.

## Storage & Calibration
- Snapshots: `data/depth/<camera>/<YYYYMMDD>/<HH>/<timestamp_us>.zarr` (Blosc/Zarr, managed by `DepthStorageManager`).
- Calibration: `calibration-bundle` (intrinsics/extrinsics/align) is served on connect and cached on the storage manager for world projections and floorplans.

## Related Docs
- `MapAnything_Depth.md` – pipeline branch, gating, normals, env toggles
- `DS8_metadata_contracts.md` – `DepthResult` schema
- `DS8_api_contracts_ws.md` – `ma_depth_response` / `floorplan_response` contracts
