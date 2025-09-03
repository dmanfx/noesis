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

## output (AppConfig.OutputSettings)

- `OUTPUT_DIR`, `SAVE_FRAMES`: respected at higher layers when saving frames; DeepStream provides JPEG bytes

## Runtime configuration via WebSocket

- Detection config updates → `DeepStreamVideoPipeline.update_detection_config()`
  - Sets `confidence-threshold`, `iou-threshold`, `enable`, `custom-lib-props` for target classes
- Detection toggles → `update_detection_toggle()` for `detect_people`, `detect_vehicles`, `detect_furniture`
- Visualization toggle → `set_trail_visualization()` via `toggle_callback`
