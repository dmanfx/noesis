# MapAnything Depth (DS8)
_Status: validated against code on 2026-02-02._

DS8 runs MapAnything as a **full-frame SGIE** inside the Service Maker pipeline. The legacy microservice/adapter flow has been archived to `docs/history/ds7/MapAnything_Depth_legacy.md`.

## Tooling
- **Heatmap viewer (3‑stream):** `docs/MapAnything_Heatmap_Viewer.md` (script: `scripts/ma_heatmap_multiuri.py`)

## Pipeline Topology
- Branch: `main_tee → mapanything_queue → mapanything_valve → mapanything_fullframe (nvinfer) → fakesink`.
- Config source: `config/infer.yaml` (`models.mapanything.*`). Default `gie_id=2`, `attach_tensor_meta=true`, `batch_size=3`.
- Valve gating: `mapanything_valve.drop` is toggled by `DS8Pipeline.mark_depth_enabled()`.
  - REST: `GET /api/v1/depth/refresh?seconds=N` (see `noesis/server/depth_api.py`).
  - WebSocket: `get_ma_depth` RPC triggers a short gate-open burst (`NOESIS_DEPTH_RPC_ENABLE_SECONDS`, default 2s).
  - Startup: optional prime via `NOESIS_DEPTH_ENABLE_SECONDS` CLI/env (defaults to 0 = closed).

## Postprocess & Storage
- Processor: `MapAnythingProcessor` (`noesis/pipelines/hooks.py`), attached only when `NOESIS_MAPANYTHING_POSTPROCESS_ENABLED` is truthy (default `1`).
- Tensor decode: expects DS8 `TensorOutputUserMetadata`; converts layers to numpy (DLPack via torch).
- Alignment: depth/conf/mask are letterboxed to the source frame size.
- Storage: `geometry/depth_source.DepthStorageManager` (configured via `mapanything_config.load_service_config()`), defaults:
  - Base path: `data/depth`
  - Max snapshots per camera: `service.storage.max_snapshots_per_camera`
  - Retention: `service.storage.snapshot_retention_minutes`
  - Async writes enabled unless `storage.async_enabled` is false.
- Telemetry: publishes `DepthResult` (see `docs/DS8_metadata_contracts.md`).

## Depth Retrieval (WebSocket RPC)
- Handler: `noesis/ds8_runtime._ds8_ma_depth_provider` → registered as `WebSocketServer.ma_depth_provider` (`get_ma_depth`).
- Response: `ma_depth_response` with fields `camera`, `request_id`, `served_from_cache`, `ts_us`, `ok`, optional `error`, and `payload` containing `depth_b64`, `conf_b64`, `mask_b64`, `shape`.
- Cache behavior:
  - If `ts_max_us` is provided, the freshest snapshot **at or before** that timestamp is returned.
  - Otherwise, the latest snapshot is returned; if MapAnything is gated off, a refresh window is opened (`enable_depth`) and the RPC waits up to ~3s for a newer snapshot.
- Normals: optional attachment when `NOESIS_MAPANYTHING_NORMALS_ENABLE=1` (default). Space and dtype can be set via `NOESIS_MAPANYTHING_NORMALS_SPACE` (`camera`|`world`, default `camera`) and `NOESIS_MAPANYTHING_NORMALS_DTYPE` (`float16` default).

## Depth Telemetry (always-on)
- `DepthResult` telemetry is emitted for every MapAnything inference when the gate is open. Payload fields are `source_id`, `frame_id`, `ts` (epoch seconds), `width`, `height`, `depth_map_ref`, `minmax`, `unit="m"`.

## Gating Notes
- If DS8 bindings lack `BufferOperator`, `depth_gate_supported` is false; the logical flag still tracks enable/disable, but SGIE work may continue. A warning is pushed to `pipeline.errors` in this case.
- Valve priming on activation: when the valve exists and depth is initially disabled, DS8 keeps it open briefly (`NOESIS_MAPANYTHING_GATE_PRIME_SECONDS`, default 1.0) to avoid preroll stalls.

## Quick Validation
- Ensure `mapanything_valve` exists in the built pipeline (`ds8_pipeline.build_pipeline` should set `pipeline.valve_name`).
- Run `python3 scripts/ma_depth_rpc_smoke_test.py --no-spawn` with DS8 runtime active; expect `ma_depth_response.ok=true` and `served_from_cache` to flip between true/false depending on recent activity.
- Confirm depth snapshots written under `data/depth/<camera>/` and `DepthResult` telemetry flowing on the WebSocket stream.
