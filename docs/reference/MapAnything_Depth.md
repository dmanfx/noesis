# MapAnything Depth Integration

This reference documents the end-to-end depth estimation stack that augments the DeepStream pipeline with Meta's MapAnything model. It covers the GPU microservice, adapters, core integration, storage contracts, diagnostics front-ends, and operational procedures.

## Overview

```
[DeepStream Pipeline] → [MapAnything Service (FastAPI/GPU)] → [Depth Storage (Zarr)]
                              ↓                                        ↓
[Noesis Core (geometry/)] → [Floor Plane Fit] → [Calibration Updates]
                              ↓                                        ↓
[WebSocket Server] ←→  Menon (Three.js) + oai2-fe (Operations Dashboard)
                              ↓
                    [InfluxDB Metrics] & [MQTT Depth Summaries]
```

Key goals:
- Provide monocular depth estimates for each camera at ~2 Hz with <100 ms latency.
- Persist depth maps in Zarr for replay and tooling.
- Broadcast depth quality telemetry to both Menon (visual overlays) and oai2-fe (diagnostics drawer).
- Maintain calibration scale and pose confidence using multi-view inference.
- Enforce licensing compliance for the Apache MapAnything model weights.

## Components

### FastAPI Microservice (`services/mapanything_svc/`)
- `server.py`: FastAPI app exposing `/infer_mono` and `/infer_multi` endpoints. Loads the MapAnything model lazily with BF16 AMP, Apache model guard, and checksum verification (`docs/ma-integration/weights.sha`).
- `run.sh`: Launch helper that executes `python3 -m uvicorn` with environment controlled by `config/mapanything.ini` (host, port, model device, AMP settings).
- Runtime features:
  - Per-scene asyncio queue for multi-view requests (drops stale work while accepting latest scene bundle).
  - Exponential backoff retry wrapper around inference calls.
  - Dynamic memory-efficient toggle when VRAM usage exceeds 80%.
  - `/health` and `/metrics` endpoints for operations dashboards.

### Adapter Layer (`adapters/mapanything_adapter.py`)
- Converts DeepStream BGR frames to RGB, resizes to configured max resolution, scales intrinsics, and encodes payloads in base64.
- Caches intrinsics and scale factors per camera; supports both mono and batched multi-view requests.

### Geometry Integration (`geometry/`)
- `depth_source.py`: HTTP client for the microservice with retry/backoff, summary computation, Zarr persistence, and latest-depth RPC provider.
  - Batches mono callers through `/infer_multi` when `[performance] multi_batch_size > 1`, flushing within ~5–20 ms so multiple cameras share preprocessing/inference work while still falling back to `/infer_mono` on errors.
- `depth_publisher.py`: MQTT + Influx publisher for depth summaries (`noesis/geometry/<room>/<cam>/depth_summary` and `mde.depth.summary`, `mde.scale`, `mde.pose.error`).
- `floor.py`: Backprojection and RANSAC utility to fit floor planes and map camera planes into world space.
- `transform.py`: Extended pixel-to-world helper that prefers MapAnything depth when confidence exceeds configured threshold.

### Core Wiring (`main.py` / `websocket_server.py`)
- Spawns the microservice if not running, verifies health, and schedules mono depth inference via a thread pool.
- Motion gating: skips redundant inference when confidence ≥ 0.9 and no active tracks are present.
- Broadcasts `ma_diagnostics` messages and augments pixel-to-world responses with `method`, `conf`, and `depth` metadata.
- Serves `get_ma_depth` RPC retrieving the most recent Zarr snapshot for a camera.

### Frontend Touchpoints
- **Menon**
  - `src/services/WebSocketClient.js`: Handles `ma_diagnostics` and `ma_depth_response`; exposes `requestMADepth` helper.
  - `src/ui/components/SettingsPanel.js`: Adds MapAnything toggles (heatmap, confidence tint, HUD status) with persistence.
  - `src/features/occupancy/OccupancyVisualizer.js`: Renders viridis heatmaps atop room planes and listens for depth summaries.
  - `src/three/Overlays.ts`: Displays MDE HUD status indicator driven by summary confidence.
- **oai2-fe**
  - `src/hooks/useWebSocketClient.ts`: Subscribes to `ma_diagnostics` and `ma_depth_response` events.
  - `src/components/DepthDrawer.tsx`: Slide-out diagnostics drawer with heatmap canvas, stats, histogram, and metrics.
  - `src/styles/depth-drawer.css`: Styling for the diagnostics drawer.

## REST & RPC Interfaces

### Microservice API
- `POST /infer_mono`
  - Payload: `{ "view": { "cam_id": str, "img_b64": str, "shape": [H,W,3], "intrinsics"?: [[...]] } }`
  - Response: `{ "cam_id": str, "depth_z": [[...]], "conf": [[...]], "mask": [[bool]], "shape": [H,W], "ts_us": int }`
- `POST /infer_multi`
  - Payload: `{ "scene_id": str, "views": [ ... same as mono ... ] }`
  - Response: depth/conf/mask per camera plus flattened pose/intrinsics dictionaries and `scale`.
- `GET /health`: `{ "status": "ok", "model_loaded": bool, "device": str, "started_at": float }`
- `GET /metrics`: latency counters, invocation counts, VRAM usage, queue depths, last error.

### WebSocket Messages
Refer to `docs/reference/WebSocket_API.md` for full schemas. New/updated messages include:
- `ma_diagnostics`: Broadcast depth summary (median, p10/p90, confidence, valid ratio, method).
- `ma_depth_response`: Base64-encoded depth/conf/mask arrays returned from `get_ma_depth` requests.
- `pixel_to_world_response`: Now includes `method`, `conf`, and `depth` when MapAnything depth is used.
- Client RPC `get_ma_depth`: Fetches most recent Zarr snapshot below a given timestamp.

## Configuration

### `config/mapanything.ini`
```
[service]
host = 127.0.0.1
port = 8001
api_key = noesis_secret

[inference]
model_id = facebook/map-anything-apache
device = cuda:0
amp_dtype = bf16
memory_efficient_mono = false
memory_efficient_multi = true
apply_mask = true
mask_edges = true
confidence_percentile = 10

[performance]
max_res = 960
mono_freq = 2
multi_batch_size = 6
multi_interval = 60
min_conf = 0.5

[storage]
depth_base = data/depth
calib_base = data/calib
max_snapshots_per_camera = 600
snapshot_retention_minutes = 10
```
- Parsed through `mapanything_config.load_service_config()` and shared across service and client components.
- To override host/port at runtime, set `MA_SERVICE_HOST` / `MA_SERVICE_PORT` environment variables.

### Application Config (`config.py`)
- Integrations:
  - `AppConfig.integrations`: adds depth summaries to the existing MQTT + Influx publisher (see below).
- Calibration bundle now includes `metric_scale` and per-camera `pose_confidence` fields.
- `main.py` respects `visualization.JPEG_QUALITY` for both occupancy feeds and MapAnything overlays.

## Storage & Metrics

### Zarr Layout
```
data/depth/<camera_id>/<YYYYMMDD>/<HH>/<timestamp_us>.zarr/
  depth_z (float32)
  conf (float32)
  mask (uint8)
  attrs: camera_id, timestamp_us, stored_at, shape
```
- Managed by `DepthStorageManager` with Blosc (zstd) compression and 128×128 chunks. The manager enforces
  per-camera retention using a ring buffer (`max_snapshots_per_camera`) and time-based expiry (`snapshot_retention_minutes`).

### MQTT
- Topic: `noesis/geometry/<room>/<cam>/depth_summary`
- Payload: `{ "ts_us": int, "median": float, "p10": float, "p90": float, "conf_mean": float, "valid_ratio": float, "sample_count": int }`

### InfluxDB (bucket `noesis_raw`)
- Measurements:
  - `mde.depth.summary` (tags: camera, room; fields: median, p10, p90, conf_mean, valid_ratio, sample_count)
  - `mde.scale` (tags: camera, room; fields: scale, pose_error?)
  - `mde.pose.error` (optional depending on calibration updates)
- Write precision: microseconds for depth summaries, nanoseconds for scale/pose metrics.

## Frontend Interactions

### Menon
- Settings toggles broadcast `ma:toggle` events that OccupancyVisualizer listens for to enable/disable rendering.
- Heatmaps are rendered as `THREE.CanvasTexture` planes; confidence modulates alpha when enabled.
- HUD indicator reflects MapAnything confidence (green ≥ 0.5, amber otherwise).
- `requestMADepth(cameraId, tsMax)` helper triggers `get_ma_depth` RPC via WebSocket.

### oai2-fe
- DepthDrawer exposes four tabs: Heatmap, Stats, Histogram, Metrics.
- Uses base64-decoded Float32 arrays to populate canvas elements (no heavyweight chart libraries required).
- Drawer auto-refreshes when diagnostics arrive or when the user switches cameras.
- WebSocket client exposes `requestMapAnythingDepth(camId)` for manual refresh actions.

## Operations & Tooling

- **Service launch**: `python3 main.py` automatically spawns the microservice; to run manually use `services/mapanything_svc/run.sh`.
- **Checksum verification**: `scripts/verify_ma_weights.py` hashes the Apache weights and compares against `docs/ma-integration/weights.sha`.
- **Performance smoke test**: `scripts/perf_test.py --iterations 20` measures average mono latency using the local service.
- **Batch tuning**: `[performance] multi_batch_size` controls how many mono requests are grouped per `/infer_multi` call; reduce to `1` to disable batching or adjust alongside `multi_interval` when large camera fleets require stronger coalescing.
- **Manual diagnostics**: `curl -H "X-API-Key: noesis_secret" -X POST http://127.0.0.1:8001/infer_mono -d '{...}'` to exercise the endpoint with encoded frames.
- **Log awareness**: microservice logs memory-efficiency toggles, checksum status, and inference errors; Noesis logs indicate when motion gating skips frames.

## Troubleshooting

| Symptom | Likely Cause | Remedy |
|---------|--------------|--------|
| `uvicorn: not found` when launching service | `uvicorn` not on PATH | `run.sh` now invokes `python3 -m uvicorn`; ensure Python env has FastAPI/uvicorn installed |
| 500 from `/infer_mono` | Model load failure or checksum mismatch | Check logs for checksum error, confirm `docs/ma-integration/weights.sha` entry matches installed weights |
| Missing heatmaps in Menon | MapAnything toggles disabled or WS depth cache empty | Enable toggles in Settings, ensure `ma_diagnostics` messages are flowing, check `window.maDepthCache`|
| oai2-fe depth drawer empty | `ma_diagnostics` not received | Verify WebSocket connection status chip, confirm server broadcasting (watch backend logs) |
| High VRAM usage | Large frames or multi-view backlog | Service auto-switches to memory efficient mode above 80% usage; consider lowering `[performance] max_res` |

## Related Documents
- `docs/ma-integration/PRD.md`: Product requirements.
- `docs/ma-integration/Blueprint.md`: Architecture blueprint for the integration.
- `docs/reference/WebSocket_API.md`: Message schemas (updated with MapAnything topics).
- `docs/reference/Configuration_Map.md`: Config key mapping, including MapAnything entries.
- `docs/reference/Integrations_Playbook.md`: MQTT/Influx integration details expanded to cover depth summaries.

Maintain this document as the authoritative reference whenever the MapAnything service, adapters, metrics, or front-end diagnostics evolve.
