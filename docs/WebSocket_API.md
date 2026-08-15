# WebSocket API quick reference

Status: canonical DS9.1 transport, 2026-08-15.

The field-level contract is `api_contracts_ws.md`. Noesis listens on loopback
port 6008 with required internal bearer authentication. Browsers connect
through Menon using a short-lived one-use ticket; they never see the bearer.

## Main message families

- `stats`: source progress, FPS, capability state, zero-copy and boundary
  metrics.
- `tracking`: StableID tracks, occupancy, transitions, world/depth diagnostics.
- `world_snapshot` and `world_event`: committed canonical world cohort.
- `bev-frame`: metadata-only BEV/floorplan presentation paired with committed
  tracking.
- `depth_result` and `ma_depth_response`: MapAnything capture metadata and
  bounded component descriptors.
- `floorplan_response`: fresh/cache/PCF floorplan contract.
- `calibration-bundle` and calibration/control responses.
- WebRTC offer/answer/ICE messages for the H.264 mosaic.

Canonical tracking, world, and BEV messages use typed, ordered publication and
are never generic latest-only traffic. Mosaic media is WebRTC-only; RTSP and
BEV JPEG output are disabled.

Implementation anchors:

- server: `websocket_server.py`;
- runtime wiring: `DS9/noesis/ds9_runtime_core.py`;
- publishers: `DS9/noesis/telemetry/` and shared `noesis/telemetry/`;
- browser consumer: `oai2-fe/` through Menon.
