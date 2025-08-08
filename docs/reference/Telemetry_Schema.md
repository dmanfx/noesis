# Telemetry Schema (DeepStream Analytics)

This document defines the telemetry payloads produced from the DeepStream pipeline via `_analytics_probe()` and related helpers in `deepstream_video_pipeline.py`.

## Object-level data (per detection)

Produced by `_parse_obj_meta()` and `_extract_analytics_obj_meta()`:

```json
{
  "class_id": int,
  "confidence": float,
  "bbox": [left, top, width, height],
  "object_id": int,
  "secondary_inference": {
    "classification": string,
    "confidence": float,
    "component_id": int
  },
  "analytics": {
    "dirStatus": {"<roi>": 0|1}? ,
    "lcStatus": {"<line>": 0|1}? ,
    "ocStatus": {"<roi>": 0|1}? ,
    "roiStatus": {"<roi>": 0|1}? ,
    "direction_status": "(legacy alias)",
    "line_crossing_status": "(legacy alias)",
    "overcrowding_status": "(legacy alias)",
    "roi_status": "(legacy alias)"
  }
}
```

Notes:
- Keys match DeepStream NVDS Analytics casing for downstream compatibility.
- Secondary inference is optional and present only if configured (SGIE currently not linked).

## Frame-level analytics

When per-object ROI is not available, `_extract_analytics_frame_meta()` may provide summary counts:

```json
{
  "objects_in_roi": {"<roi>": int},
  "line_crossing_cumulative": {"<line>": int},
  "line_crossing_current": {"<line>": int},
  "overcrowding_status": {"<roi>": 0|1}
}
```

## Tracking state (per source)

The `_analytics_probe()` updates `self.live_tracking_state[source_id]` with:

```json
{
  "active_tracks": [
    {
      "track_id": int,
      "camera_id": "camera_<id>",
      "confidence": float,
      "bbox": [left, top, width, height],
      "class_id": int,
      "center": [x, y],
      "tracker_confidence": float
    }
  ],
  "occupancy": {"<roi>": int},
  "transitions": [
    {"track_id": int, "camera_id": string, "line_name": string, "timestamp": float}
  ]
}
```

## Telemetry frame envelope

Broadcast via `WebSocketServer.broadcast_frame()`:

```json
{
  "type": "frame",
  "payload": {
    "source_id": int,
    "timestamp": float,
    "tracking": { /* tracking state per above */ }
  }
}
```

## Stats payload

Returned by `DeepStreamVideoPipeline.get_stats()` and sent as `{type:"stats"}`:

```json
{
  "pipeline_type": "deepstream",
  "running": bool,
  "frames_processed": int,
  "fps": float,
  "runtime_seconds": float,
  "batch_size": int,
  "sources": int,
  "queue_sizes": {"source_<id>": int},
  "tracking": { /* per-source tracking states */ }
}
```
