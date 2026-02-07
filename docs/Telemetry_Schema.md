# Telemetry Schema (DS8)
_Status: validated against code on 2026-02-02._

Canonical WebSocket payloads live in `docs/DS8_api_contracts_ws.md`. This page summarizes **where** telemetry is produced in the DS8 stack and the exact field sets emitted today. Older telemetry descriptions are archived under `docs/history/`.

## Producers and Message Types

- **Stats** – built in `noesis/ds8_runtime.py:_build_stats_callback` and broadcast once per second when clients are connected.
- **Tracking** – `TrackingTelemetryPublisher.publish` in `noesis/telemetry/publishers.py`, fed by `_AnalyticsTelemetryProcessor` (`noesis/pipelines/hooks.py`).
- **Depth** – `DepthTelemetryPublisher.publish` in `noesis/telemetry/publishers.py`, fed by `MapAnythingProcessor`.
- **BEV** – `BevRenderer._publish` in `noesis/telemetry/bev.py` (JSON always; optional JPEG when enabled).

## Stats Payload (type: `stats`)

```json
{
  "type": "stats",
  "payload": {
    "timestamp": <float>,
    "uptime": <float>,
    "stack": "ds8",
    "application": {
      "running": <bool>,
      "cameras_active": <int>,
      "processors_active": <int>
    },
    "pipeline": {
      "stack": "ds8",
      "prepared": <bool>,
      "activated": <bool>,
      "depth_enabled": <bool>,
      "depth_fps": <float>,
      "analytics_reload_count": <int>,
      "mosaic_layout": { /* may be null */ },
      "latency_ms": { /* present only when NVDS latency is enabled */ },
      "errors": [<string>, ...]
    },
    "cameras": {
      "<camera_id>": {
        "fps": <float>,
        "frames_processed": <int>,
        "status": "running"|"unknown",
        "tracking": {
          "occupancy": {"<zone>": <int>, ...},
          "active_tracks": [ /* internal diagnostic only */ ],
          "transitions": [ /* line/zone transitions */ ]
        },
        "latency_ms": { /* present when NVDS latency enabled */ }
      }
    }
  }
}
```

Notes:
- `latency_ms` comes from `LatencyCollector` and is populated only when `NVDS_ENABLE_LATENCY_MEASUREMENT` is truthy and the DeepStream latency meta library is available.
- `mosaic_layout` mirrors the current tiler layout: `mosaic_w/h`, optional `rows/cols`, `square_seq_grid`, `frame_w/h`, `source_count`, and `sources` array of `{source_id, camera_id}`.

## Tracking Payload (type: `tracking`)

Emitted once per frame per source. Only **people** tracks (class_id=0) are published; raw tracker IDs remain internal.

```json
{
  "type": "tracking",
  "source_id": <int>,
  "tracks": [
    {
      "stable_id": <int>,
      "camera_id": "<camera>",
      "bbox": [<float left>, <float top>, <float width>, <float height>],
      "center": [<float cx>, <float cy>],
      "class_id": 0,
      "confidence": <float|null>,
      "tracker_confidence": <float|null>,
      "analytics": { /* NvDsAnalytics obj meta, camel + snake case keys */ },
      "zone": "<string|null>",
      "frame_id": <int>,
      "dwell_time": <float|null>,
      "bbox3d": { /* present in V3DT/SV3DT mode */ },
      "velocity3d": [<float x>, <float y>, <float z>],
      "visibility": <float|null>,
      "image_foot": [<float u>, <float v>],
      "image_base": [<float u>, <float v>],
      "world": [<float x>, <float y>, <float z>],
      "world_valid": <bool>,
      "world_frame": "camera_local"|"world"|null,
      "world_source": "sv3dt"|"ray"|null
    }
  ]
}
```

- `stable_id` is always present for people tracks; `track_id` is never exposed.
- `world`/`bbox3d` fields appear only when V3DT/SV3DT metadata is available (tracker config under `config/v3dt/`).

## Depth Payload (type: `depth_result`)

Published per MapAnything inference result via `DepthResult.to_dict()`:

```json
{
  "type": "depth_result",
  "payload": {
    "source_id": <int>,
    "frame_id": <int>,
    "ts": <int epoch_seconds>,
    "width": <int>,
    "height": <int>,
    "depth_map_ref": "<path or opaque ref>",
    "minmax": [<float min>, <float max>],
    "unit": "m"
  }
}
```

## BEV Payload (type: `bev-frame`)

```json
{
  "type": "bev-frame",
  "cameraId": "<camera>",
  "ts": <int microseconds>,
  "w": <int>,
  "h": <int>,
  "mpp": <float meters_per_px>,
  "xMin": <float>, "xMax": <float>,
  "zMin": <float>, "zMax": <float>,
  "overlay": <bool>,
  "footpoints": [ {"x": <float>, "y": <float>, "method": "bbox"|"sv3dt", "stableId": <int|null>} ],
  "H": [<9 floats>],
  "sampleXZ": [<float x>, <float z>] | null
}
```

- Optional JPEG binary follows the framing `[len(header)] [header="bev:<camera>"] [JPEG bytes]` when `bev.jpeg_enabled=true` or `NOESIS_BEV_JPEG_ENABLED=1`.
- Footpoints use `stable_id` when available; tracker IDs are never exposed.

## Related Docs

- Contracts and RPCs: `docs/DS8_api_contracts_ws.md`
- Metadata shapes: `docs/DS8_metadata_contracts.md`
- BEV rendering and smoothing: `docs/DS8_Baselines.md` (BEV section)
