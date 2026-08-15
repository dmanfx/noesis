# DS8 Zero-Copy Hotspot Inventory

## Core hotspots identified
- `noesis/pipelines/ds8_pipeline.py`
  - `osd.process-mode` enforced to GPU mode (`0`) for zero-copy core path.
- `noesis/pipelines/hooks.py`
  - DLPack/tensor conversions to CPU numpy in MapAnything boundary path.
  - Pose/ReID paths run on native DS8 metadata extraction in production.
  - OpenCV resize and numpy operations in depth tensor handling.
- `websocket_server.py`
  - repeated JSON serialization and numpy conversion in broadcast loops.

## Boundary hotspots (allowed with budget)
- `noesis/pipelines/hooks.py` -> depth storage payload preparation.
- `websocket_server.py` -> JSON serialization for WS messages.
- `noesis/server/*` -> REST response serialization.

## Immediate implementation priorities
1. Extended target-hardware soak/perf repetition for release confidence.

## 2026-02-08 progress update
- ReID native extractor shipped:
  - `native/noesis_reid_meta_ext.cpp`
  - `scripts/build_noesis_reid_meta_ext.sh`
- ReID hook now uses native extraction as the only production path.
- Runtime no longer exposes temporary migration path-mode stats.
- StableID GPU extension implemented:
  - backend selection added (`compute_backend=auto|cpu|gpu`) with strict GPU enforcement in hard-cutover.
  - GPU vector similarity path integrated in `StableIDManager._gallery_best`.
  - identity lifecycle/alias/hysteresis logic remains CPU-owned.
  - runtime metrics exposed via `get_sid_metrics`: `stableid_backend_mode`, `stableid_gpu_match_p50_ms`, `stableid_gpu_match_p95_ms`, `stableid_gallery_size`.
- MapAnything hardening implemented:
  - removed fallback-to-first-tensor behavior on GIE mismatch;
  - enforced async-only processing in production mode;
  - added `tensor_gie_mismatch_drops_total.mapanything` and `tensor_host_copies_total.mapanything` counters.
- WS boundary serialization hardening completed:
  - route/stage-tagged boundary metrics in `websocket_server.py` with `budget_ms` + `violations`;
  - explicit `provider_wait` stage for `get_ma_depth` / `get_floorplan` (excluded from budget);
  - compact JSON encoding + single-encode fanout preserved for broadcast path.
- REST boundary serialization hardening completed:
  - shared REST boundary metrics recorder in `noesis/server/boundary_metrics.py`;
  - route/stage instrumentation added for depth/analytics/reid API responses;
  - DS8 stats now publish combined boundary p99 plus WS/REST split p99 fields.
- CPU fallback cleanup completed:
  - removed CPU fallback stable-id allocator path in analytics telemetry;
  - removed deprecated CPU migration env flags/guards and dead conversion helpers;
  - enforced StableID GPU backend requirement in hard-cutover mode.
- Finalization closure:
  - added missing validation/gate assets (`zero_copy_stats_smoke_test.py`, `zero_copy_runtime_gate.py`, `zero_copy_perf_gate.py`, and corresponding test modules);
  - added missing tensor-path counters (`tensor_host_copies_total.reid|pose`, `tensor_boundary_copy_bytes_total.pose_meta|depth_store`);
  - fixed WS exception-path robustness (`import websockets` for disconnect handler in `websocket_server.py`).
