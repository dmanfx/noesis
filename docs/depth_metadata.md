# Depth Metadata (DS8)
_Status: validated against code on 2026-02-02._

DS8 does **not** attach per-object depth user meta. Depth is produced by the MapAnything full-frame SGIE branch and surfaced through two channels:

1) **DepthResult telemetry** (`type: depth_result`)
- Producer: `MapAnythingProcessor` → `DepthTelemetryPublisher`.
- Shape: see `docs/DS8_metadata_contracts.md` (§Depth Result) and `noesis/metadata/depth_result.py`.
- `depth_map_ref` points to the stored Zarr snapshot on disk.

2) **Depth snapshots for RPCs** (`ma_depth_response`)
- Producer: `noesis/ds8_runtime._ds8_ma_depth_provider` backed by `DepthStorageManager`.
- Payload is the cached snapshot (`depth_b64`, `conf_b64`, `mask_b64`, `shape`) with optional normals when `NOESIS_MAPANYTHING_NORMALS_ENABLE=1`.
- Storage layout: `data/depth/<camera>/<YYYYMMDD>/<HH>/<timestamp_us>.zarr/` (managed by `geometry/depth_source.DepthStorageManager`).

Calibration is distributed via the **calibration-bundle** WebSocket message and cached on the storage manager so depth→world projections stay consistent.

The older per-object `NOESIS.MDE` user-meta format is archived under `docs/history/` for reference.
