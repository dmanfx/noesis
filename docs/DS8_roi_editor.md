# DS8 ROI Editor (Exclusion Zones)
_Status: current as of 2026-07-10._

The ROI editor is a dashboard drawer for the writable `exclude` analytics
stage. It previews one camera from the live WebRTC mosaic, edits polygons in the
stage's `config_width`/`config_height` pixel space, and calls the exact
`GET/POST /api/v1/analytics/rois` contract.

The durable source is the analytics YAML. The API derives the native INI and
commits it to the repo-owned pre-tracker `nvdsroiexclude` element. There is no
Python pruning fallback. A save is successful only when REST returns
`reloaded=true` with a complete native hash/sequence receipt; the server does
not infer success from a file write.

## Access boundary

Appliance browsers reach Noesis through the authenticated same-origin gateway.
The gateway keeps the owner-only internal bearer and sends it to Noesis; the
browser must never receive or persist that token. Direct browser-to-Noesis REST
with required authentication is not an appliance path.

Explicit loopback development may use `NOESIS_INTERNAL_AUTH_MODE=disabled`, but
both REST and WebSocket must bind to `localhost` or a literal loopback address.
If cross-origin loopback development is required, list exact origins in
`NOESIS_REST_CORS_ORIGINS`. Wildcard/regex CORS and
`NOESIS_REST_CORS_ALLOW_ALL` are forbidden.

The runtime must have REST and the RTSP-to-WebRTC mosaic enabled. The dashboard
also needs `stats.payload.pipeline.mosaic_layout`; its ordered source rows map
analytics stream IDs to mosaic tiles. Do not guess tile placement from camera
names when layout metadata is absent.

## Editing behavior

- Points are stored in analytics-stage pixel coordinates, independent of the
  drawer size or displayed tile crop.
- Only completed polygons with at least three points are submitted. The server
  additionally enforces finite, in-bounds, non-degenerate geometry and strict
  ROI IDs.
- The editor updates one selected stream per apply; all other durable stream
  policies remain unchanged.
- If the last completed ROI is deleted, apply sends `enable=false` with an
  empty ROI list. An enabled stream with no ROI is rejected with HTTP 422.
- A successful response replaces the drawer state with server readback and
  clears that stream's undo/redo history. A failed or ambiguous native commit
  remains an error and must not be presented as applied.

API shapes, size limits, receipt fields, rollback, poisoning, and shutdown
quiescence are normative in `docs/DS8_api_contracts_rest.md`.

## Validation

Run the UI-independent contract gates first:

```bash
python3 -m pytest -q \
  tests/test_analytics_api.py \
  tests/test_pipeline_build.py \
  tests/test_roi_reload_smoke_test.py \
  DS9/tests/test_nvdsroiexclude_plugin.py
```

Then capture current behavior only with a real occupied camera and the
authenticated restore gate:

```bash
python3 scripts/roi_reload_smoke_test.py \
  --hot-restore \
  --camera <occupied-camera> \
  --auth-token-file "$HOME/.local/state/noesis/gateway-token" \
  --evidence "$HOME/.local/state/noesis/diagnostics/roi-hot-restore.json"
```

The gate must observe advancing real tracker frames, temporary full-frame
exclusion, an increased native removal count, unconditional exact restore, and
the person's return. An unoccupied scene is blocked, and historical
counter-only smoke evidence is not acceptance for the hardened path.

## Troubleshooting

- **Layout unavailable:** require current `mosaic_layout` rows/columns and
  source entries; do not hardcode a tile order.
- **GET/POST returns 401/403:** use the appliance gateway, or explicit
  loopback-only development auth disablement. Do not put the internal bearer in
  frontend code, a URL, or local storage.
- **Apply returns 422:** check final-ROI enable state, ROI ID, coordinate bounds,
  point count, and polygon area.
- **Apply returns 503:** inspect the native reload receipt/error and runtime
  poison state. Resolve the defect and restart cleanly if poisoned; do not retry
  through another pruning path.
- **UI says applied but behavior is unchanged:** verify the response carried
  matching request/accepted sequences and the expected active INI SHA-256, then
  use the occupied restore gate. A reload counter alone is insufficient.
