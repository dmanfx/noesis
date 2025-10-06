# Configuration Map (config.py → Pipeline)

This document maps `config.py` sections and keys to their use in `deepstream_video_pipeline.py` and `websocket_server.py`.

## cameras (AppConfig.CameraSettings)

- `RTSP_STREAMS`: Used to build `sources` passed into `DeepStreamVideoPipeline`.
  - Fields per stream: `name`, `url`, `width`, `height`, `enabled`
- Derived inside pipeline:
  - `batch_size = len(enabled sources)`
  - `max_width = max(width)`, `max_height = max(height)`

## processing (AppConfig.ProcessingSettings)

- `DEEPSTREAM_PREPROCESS_CONFIG`: set on `nvdspreprocess.config-file`
- Other DeepStream-related knobs exist but the pipeline derives batch/size from sources via `nvmultiurisrcbin`.

## models (AppConfig.ModelsSettings)

- `MODEL_PATH`: used in `_check_for_engine_file()` to infer candidate engine paths under `models/engines/`
- `DETECTION_ENGINE_PATH`: optional override placed at highest priority in engine search
- Runtime detection parameters are applied via `nvinfer` properties/methods

## visualization (AppConfig.VisualizationSettings)

- Trail visualization parameters consumed by `_per_branch_osd_probe()`:
  - `TRAIL_LENGTH`, `TRAIL_TIMEOUT_S`, `TRAIL_DRAW_STRIDE`, `TRAIL_SHOW_LABELS`, `TRAIL_DRAW_SEGMENTS`
- `USE_NATIVE_DEEPSTREAM_OSD`: application-level flag in `main.py` that bypasses Python annotation; DeepStream OSD runs in per-branch paths regardless.

## websocket (AppConfig.WebSocketSettings)

- `HOST`, `PORT`: passed to `WebSocketServer`
- `MAX_FPS`, `JPEG_QUALITY`: front-end throttling defaults; note that `visualization.JPEG_QUALITY` now configures the DeepStream `nvjpegenc` quality used for binary frames

## tracking (AppConfig.TrackingSettings)

- `USE_NATIVE_DEEPSTREAM_TRACKER`: currently informational; pipeline always uses `nvtracker`

## integrations (AppConfig.IntegrationsSettings)

- `ENABLE_OCCUPANCY_PUBLISH`: toggles MQTT + Influx occupancy publishing
- `HEARTBEAT_SEC`: heartbeat period for retained MQTT refresh (no Influx write)
- MQTT: `BASE_TOPIC`, `STATUS_TOPIC`, `MQTT_HOST`, `MQTT_PORT`, `MQTT_USERNAME`, `MQTT_PASSWORD`, `MQTT_QOS`, `MQTT_RETAIN`
  - Consumed to initialize `OccupancyPublisher` MQTT client and retained topics
- Influx: `INFLUX_URL`, `INFLUX_ORG`, `INFLUX_TOKEN`, `INFLUX_BUCKET_RAW`
  - Consumed to initialize `InfluxDBClient` and async write API
  - MapAnything depth summaries piggyback on the same publisher (`DepthDiagnosticsPublisher`) using these credentials

## output (AppConfig.OutputSettings)

- `OUTPUT_DIR`, `SAVE_FRAMES`: respected at higher layers when saving frames; DeepStream provides JPEG bytes

## MapAnything configuration (`config/mapanything.ini`)

Parsed by `mapanything_config.load_service_config()` and shared between `services/mapanything_svc` and `geometry/depth_source.py`.

- `[service]`
  - `host`, `port`: FastAPI bind address consumed by `run.sh` and `main.py` health checks
  - `api_key`: Required header for `/infer_mono` and `/infer_multi`
- `[inference]`
  - `model_id`: HuggingFace repo id (must end with `-apache`)
  - `device`: torch device string (e.g., `cuda:0`)
  - `amp_dtype`: `bf16` or `fp16`
  - `memory_efficient_mono`, `memory_efficient_multi`: Baseline flags toggled dynamically when VRAM usage exceeds 80%
  - `apply_mask`, `mask_edges`, `confidence_percentile`: forwarded to MapAnything inference kwargs
- `[performance]`
  - `max_res`: Downsample cap applied in `mapanything_adapter`
  - `mono_freq`, `multi_batch_size`, `multi_interval`: Scheduling hints used by `depth_source`
  - `min_conf`: Confidence threshold for preferring MapAnything over floor-plane fallback
- `[storage]`
  - `depth_base`, `calib_base`: Root directories for Zarr depth snapshots and calibration archive

## Runtime configuration via WebSocket

- Detection config updates → `DeepStreamVideoPipeline.update_detection_config()`
  - Sets `confidence-threshold`, `iou-threshold`, `enable`, `custom-lib-props` for target classes
- Detection toggles → `update_detection_toggle()` for `detect_people`, `detect_vehicles`, `detect_furniture`
- Visualization toggle → `set_trail_visualization()` via `toggle_callback`
- MapAnything: `get_ma_depth` RPC and `ma_diagnostics` broadcast supply depth overlays to Menon and oai2-fe
