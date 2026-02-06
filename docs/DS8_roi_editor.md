# DS8 ROI Editor (Exclusion Zones)
_Status: current as of 2026-02-02._

This document describes how the DS8 ROI editor works and how to run it.

## Overview

The ROI editor is a web UI drawer that edits **exclusion polygons** for the
DS8 `exclude` analytics stage. It updates `config/nvdsanalytics.yaml` through
`/api/v1/analytics/rois` and hot-reloads the running pipeline via the existing
analytics reload bridge.

The editor uses a **solo-tile zoom** view derived from the live WebRTC mosaic.
ROI points are stored in the same coordinate system as the exclude stage
(`config_width` / `config_height`).

## Requirements

- DS8 runtime running with REST enabled:
  - `python3 noesis/ds8_runtime.py --enable-rest ...`
- REST reachable from the UI host (CORS enabled when cross-origin).
- Mosaic layout must be present in `stats.payload.pipeline.mosaic_layout`. The
  default square-seq-grid layout from `ds8_pipeline` is fine; explicit rows/cols
  are only required if the UI needs a non-square layout.

## Environment Variables

- `NOESIS_REST_CORS_ORIGINS`: Comma-separated list of allowed origins.
  - Example: `NOESIS_REST_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173`
- `NOESIS_REST_CORS_ALLOW_ALL=1`: Allow all origins (dev only).
- `NOESIS_MOSAIC_TILER_COLUMNS` / `NOESIS_MOSAIC_TILER_ROWS`: Explicit tiler layout (optional; default square-seq-grid is supported).

## API Touchpoints

- REST:
  - `GET /api/v1/analytics/rois?stage=exclude`
  - `POST /api/v1/analytics/rois`
- WebSocket stats payload (additive):
  - `payload.pipeline.mosaic_layout`

## Troubleshooting

- If the editor says "layout unavailable":
  - Confirm `stats.payload.pipeline.mosaic_layout` is present (DS8 runtime builds this from the tiler config).
  - If you overrode tiler envs, set `NOESIS_MOSAIC_TILER_COLUMNS/ROWS` or leave defaults for square-seq-grid.
- If the editor cannot save:
  - Ensure `--enable-rest` is set.
  - Confirm CORS settings allow the UI origin.
- If exclusion does not change:
  - Check DS8 logs for "Applied exclusion runtime update".
  - Verify `config/config_nvdsanalytics_exclude.ini` was regenerated.
