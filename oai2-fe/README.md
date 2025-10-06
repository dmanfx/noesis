# OAI² Frontend

A reimagined, minimal‑friction console for your spatial perception system. It preserves all existing functionality of `electron-frontend` while providing a fresh layout and interactions.

- WebSocket: `ws://localhost:6008`
- Features: 3 live streams, FPS, zone occupancy, active tracks, transitions, object trails, detection controls, telemetry panel, fullscreen per stream, clear stats

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

No changes are made to `./electron-frontend/`. This app lives in `./oai2-fe/`.

