# OAI² frontend

The Noesis dashboard presents the native DS9.1 application's WebRTC mosaic,
per-camera views, tracking, world/BEV, depth/PCF, occupancy, transitions,
calibration, and household identity controls.

Production delivery and browser authentication are owned by Menon. The browser
does not connect directly to loopback Noesis REST/WebSocket ports or receive
the internal bearer token.

## Development

```bash
npm install
npm run dev
npm run build
```

Run focused component/library tests for the view changed, then build once. A
docs-only edit does not require the frontend build.

## Current data paths

- One H.264 WebRTC mosaic is cropped client-side using
  `stats.payload.pipeline.mosaic_layout`.
- `tracking`, `world_snapshot`, and `bev-frame` supply live people/world/BEV.
- PCF/Scene Prior is the canonical floorplan/depth-panel presentation source
  for admitted camera bindings.
- PCF provides static geometry only. Dots/trails come from the paired committed
  tracking cohort; empty occupancy legitimately renders none.
- `ma_depth_response` and `floorplan_response` provide manual capture and
  diagnostics through authenticated Menon proxy routes.
- StableID resident/visitor operations use the authenticated ReID API.

Full wire details are in `../docs/api_contracts_ws.md` and
`../docs/api_contracts_rest.md`.
