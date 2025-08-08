# WebSocket API Reference

This document describes the WebSocket server behavior and message schemas used by `websocket_server.py` and `deepstream_video_pipeline.py`.

## Server

- Host: `config.websocket.HOST` (default `0.0.0.0`)
- Port: `config.websocket.PORT` (default `6008`)
- Periodic stats broadcast every ~1s when clients are connected and `stats_callback` is set.

## Messages (Server → Client)

- Stats
  - Type: `stats`
  - Schema:
    ```json
    {"type":"stats","payload":{}}
    ```
    - Payload fields from `DeepStreamVideoPipeline.get_stats()`:
      - `pipeline_type`: "deepstream"
      - `running`: bool
      - `frames_processed`: int
      - `fps`: float
      - `runtime_seconds`: float
      - `batch_size`: int
      - `sources`: int
      - `queue_sizes`: { "source_<id>": int }
      - `tracking`: per-source tracking state

- Telemetry frame
  - Type: `frame`
  - Emitted via `WebSocketServer.broadcast_frame()` from `_analytics_probe()`
  - Schema:
    ```json
    {"type":"frame","payload":{
      "source_id": int,
      "timestamp": float,
      "tracking": {
        "active_tracks": [
          {"track_id": int, "camera_id": "camera_<id>", "confidence": float,
           "bbox": [left, top, width, height], "class_id": int, "center": [x, y],
           "tracker_confidence": float}
        ],
        "occupancy": {"<roi>": int},
        "transitions": [{"track_id": int, "camera_id": string, "line_name": string, "timestamp": float}]
      }
    }}
    ```

- Encoded JPEG frame (binary message)
  - Transported as raw bytes with a fixed prefix, not JSON:
    - 1 byte: camera_id length (0..255)
    - N bytes: camera_id UTF-8 (e.g., "living-room", "kitchen")
    - Remaining bytes: JPEG payload
  - Emitted from the JPEG processing loop in `main.py` using `broadcast_sync`
  - Frontend decodes by reading first byte for id length, then slicing id and JPEG

- Detection config sync (on connect)
  - Type: `detection_config_sync`
  - Schema:
    ```json
    {"type":"detection_config_sync","config":{
      "confidence_threshold": float,
      "iou_threshold": float,
      "detection_enabled": bool,
      "target_classes": [int]
    }}
    ```

- Trail visualization initial state (on connect)
  - Type: `trail_visualization_enabled_update`
  - Schema: `{ "type": "trail_visualization_enabled_update", "enabled": bool }`

- Broadcasts for updates
  - Detection config update: `{ "type":"detection_config_update", "config": { ... } }`
  - Detection toggle update: `{ "type":"detection_toggle_update", "toggle_name": string, "enabled": bool }`
  - Visualization toggle update: `{ "type":"toggle_update", "toggle_name": string, "enabled": bool }`

## Messages (Client → Server)

- Clear stats
  - `{ "type": "clear_stats" }`

- Set visualization toggle
  - `{ "type": "set_vis_toggle", "toggle_name": string, "enabled": bool }`

- Update detection config
  - `{ "type": "update_detection_config", "config": { "confidence_threshold": float?, "iou_threshold": float?, "detection_enabled": bool?, "target_classes": [int]? } }`

- Set detection toggle (class groups)
  - `{ "type": "set_detection_toggle", "toggle_name": "detect_people"|"detect_vehicles"|"detect_furniture", "enabled": bool }`

## Runtime Notes

- JPEG frames are throttled by per-source `appsink` and queue size; oldest frames may be dropped under load.
- Telemetry frames are emitted only when there is per-source tracking data.
- Errors in per-client send are logged and do not disrupt other clients.
