# OAI² Frontend

A reimagined, minimal‑friction console for your spatial perception system. It preserves all existing functionality of `electron-frontend` while providing a fresh layout and interactions.

- WebSocket: `ws://localhost:6008`
- Features: live WebRTC mosaic, per-camera fullscreen crops with paired BEV, FPS, zone occupancy, active tracks, transitions, object trails, detection controls, telemetry panel, clear stats, **People** drawer for household resident enroll/rename/remove + identity health

## Scripts

- `npm install`
- `npm run dev` (development)
- `npm run build` (production bundle)
- `npm run preview` (serve build)

## Notes

- Trails toggle broadcasts with `{ type: 'set_vis_toggle', toggle_name: 'trail_visualization_enabled', enabled }`.
- Clear stats sends `{ type: 'clear_stats' }`.
- Detection controls emit `{ type: 'update_detection_config', config }` and `{ type: 'set_detection_toggle', toggle_name, enabled }`.
- Stats payload is consumed as `{ type: 'stats', payload }` with `payload.cameras[camId].tracking.{occupancy,active_tracks,transitions}`.
- Per-camera fullscreen uses `payload.pipeline.mosaic_layout` from the stats feed to crop individual camera tiles out of the single WebRTC mosaic locally.
- People drawer (household identity):
  - Topbar **People** opens a drawer that calls `/api/v1/reid/residents*`, `/api/v1/reid/identity_health`, and suggest-only `/api/v1/reid/aliases/suggest`.
  - Enroll flow: select a live **visitor** (IDs 1000–1031) → enter display name → `POST .../residents/enroll`.
  - Rename / remove use `PATCH` / `DELETE` on `/api/v1/reid/residents/{uuid}`.
  - Requires household mode on the runtime (`NOESIS_HOUSEHOLD_IDENTITY=1`, default on).
- Depth Drawer (MapAnything):
  - The Depth drawer is fed by `ma_depth_response` payloads.
  - The **Normals** tab (when present) visualizes per-pixel normals `(nx, ny, nz)` by mapping each component from `[-1,+1]` → `[0,255]` as RGB (R=X, G=Y, B=Z).
  - Invalid/near-zero normals are rendered transparent (typically because depth was invalid/masked there).

No changes are made to `./electron-frontend/`. This app lives in `./oai2-fe/`.
