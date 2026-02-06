# WebSocket API (DS8)
_Status: validated against code on 2026-02-02._

The canonical DS8 WebSocket contracts are maintained in `docs/DS8_api_contracts_ws.md`. This page provides the quick operator view and pointers to the code paths that implement each message. The DS7-era reference has been archived to `docs/history/ds7/WebSocket_API_ds7.md`.

## Server Basics
- Host/port: configurable via `NOESIS_WS_HOST` / `NOESIS_WS_PORT` (defaults `0.0.0.0:6008`).
- Implementation: `websocket_server.WebSocketServer`.
- Broadcast cadence: stats every ~1s when clients are connected; tracking/depth/BEV on arrival.
- Mosaic delivery: **WebRTC-only** (RTSP→WebRTC gateway). No mosaic JPEG binaries are emitted in DS8.

## Message Types (summary)
- `stats` – pipeline and per-camera stats (`ds8_runtime._build_stats_callback`).
- `tracking` – people tracks with `stable_id` (from `_AnalyticsTelemetryProcessor`).
- `depth_result` – MapAnything depth burst metadata (`DepthTelemetryPublisher`).
- `bev-frame` – BEV metadata; optional JPEG binary when enabled (`BevRenderer`).
- WebRTC signaling: `webrtc_offer`, `webrtc_answer`, `webrtc_ice_candidate`, `webrtc_error`.
- RPC responses: `ma_depth_response`, `floorplan_response`, `pixel_to_world_response`, `set_extrinsics_result`, `set_align_result`, `solve_pnp_result`, `auto_calibrate_result`.

For full field-level schemas, see `DS8_api_contracts_ws.md` and `DS8_metadata_contracts.md` (tracking/depth/BEV) and `DS8_pose_stable_id_integration.md` (pose-assisted IDs).

## Implementation References
- Server: `websocket_server.py`
- Runtime wiring: `noesis/ds8_runtime.py` (registers stats/depth/floorplan providers and WebRTC gateway)
- Tracking/depth publishers: `noesis/telemetry/publishers.py`
- BEV rendering: `noesis/telemetry/bev.py`

