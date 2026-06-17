# Static ROI Exclusion (nvdsroiexclude)

This document describes the DS8 static ROI exclusion path and how it is configured in the Service Maker pipeline.

- Element: `nvdsroiexclude` (inserted as `analytics_exclude` when `analytics.exclude.enable: true` in `config/infer.yaml`)
- Type: `GstBaseTransform` (in‑place, metadata‑only)
- Placement: tee → `analytics_exclude` → `nvtracker` (see `noesis/pipelines/ds8_pipeline.py`)
- Scope: removes per‑frame `NvDsObjectMeta` whose bbox is fully inside configured static ROIs. Optionally draws ROI outlines for diagnostics.
- Fallback/logging: `_ExcludePruneProcessor` in `noesis/pipelines/hooks.py` logs would‑be removals for DS8, but cannot delete objects itself because DS8 `object_items` are read‑only. The element above performs the actual pruning.

## Why move exclusion into a plugin?

- Removes objects before tracking/ReID to cut downstream load and avoid spurious tracks.
- Eliminates Python pad‑probe hot paths and DS8 metadata safety pitfalls.
- Keeps ROI semantics consistent across sources and tiled layouts.

## High‑level Flow

1. `nvinfer` attaches object metas (rect_params).
2. `nvdsroiexclude` loads static ROIs per stream, transforms ROI points to the streammux (pipeline) coordinates, and removes objects fully inside any ROI polygon.
3. `nvdsroiexclude` optionally attaches `NvDsDisplayMeta` line segments to visualize ROIs.
4. `nvtracker` runs on the pruned object list.
5. `nvdsosd` renders overlays (including the ROI outlines) later in the pipeline.

## Configuration

The plugin accepts a DeepStream INI modeled after `nvdsanalytics` ROI filtering:

```
[property]
# Reference design resolution of the ROI coordinates
config-width=1920
config-height=1080

# Draw ROI outlines for diagnostics (non‑zero enables)
osd-mode=1

# Stream 0 (pad-index or source-id depending on id-mode)
[roi-filtering-stream-0]
# Optional group toggle (default true)
enable=1
# ROI polygons in config resolution: x1;y1; x2;y2; ...
roi-RF1=100;100;200;200;200;300;100;300
# Optional per‑ROI toggle
enable-RF1=1

# Stream 1
[roi-filtering-stream-1]
enable=1
roi-RF1=1434;102; 1777;183; 1762;412; 1419;363
```

Notes
- Keys starting with `roi-` define polygons (≥ 3 points) in `[property].config-width/height` space.
- `enable` under a stream disables/enables all ROIs in that group.
- `enable-<LABEL>` can toggle an individual ROI (e.g., `enable-RF2=0`).

### Stream Keying (id-mode)

`nvdsroiexclude` can associate ROI groups with streams via either:
- `source-id` (default): `NvDsFrameMeta.source_id`, stable per camera/source.
- `pad-index`: `NvDsFrameMeta.pad_index`, the streammux/tiler pad (tile) index.

Use `pad-index` when your ROI config is authored by tile position (top‑left, top‑right, etc.). Use `source-id` when authored per camera identity.

Set via element property or env:
- Pipeline default: `source-id`
- Env override: `NOESIS_DS_EXCLUDE_ID_MODE=pad-index|source-id`

## Coordinate Mapping (tiler/letterbox aware)

To ensure overlays and exclusion match the composed video:
- The plugin maps ROI points from config resolution to the streammux ("pipeline") resolution for each frame: `frame_meta.pipeline_width/height`.
- Uniform scale + padding offset (letterboxing) are applied:
  - `s = min(pipeline_w/config_w, pipeline_h/config_h)`
  - `dx = (pipeline_w − config_w*s)/2`
  - `dy = (pipeline_h − config_h*s)/2`
  - `(x, y) → (x*s + dx, y*s + dy)`

This keeps ROI outlines aligned with tiles regardless of aspect ratio differences and `enable-padding` in streammux.

## Overlay (OSD)

- If `[property].osd-mode != 0`, the plugin acquires `NvDsDisplayMeta` and pushes line segments for each ROI polygon.
- Rendering is performed by `nvdsosd` downstream; the plugin itself does not modify pixels.
- Default style: green lines, width=3.

## Removal Semantics

- An object is removed if all 4 bbox corners are inside any enabled ROI polygon.
- Boundary is considered inside (ray‑casting with horizontal boundary inclusion).
- Removal uses `nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)` and holds `batch_meta->meta_mutex` while iterating/modifying.

## DS9 Pipeline Integration

- Configured via `DS9/config/infer.yaml` `analytics.exclude` block (default element `nvdsroiexclude`, config-file `DS9/config/config_nvdsanalytics_exclude.ini`).
- Built in `DS9/noesis/pipelines/ds8_pipeline.py` as `analytics_exclude` and linked before `nvtracker`.
- Runtime reload: `DS9/noesis/server/analytics_api.py` regenerates the exclusion INI (see `_persist_exclude_ini`) and triggers reload via `attach_analytics_reload_bridge`.
- Env overrides:
  - `NOESIS_ANALYTICS_EXCLUDE_CONFIG` to point the pipeline at a different INI.
  - `NOESIS_DS_EXCLUDE_ID_MODE=pad-index|source-id` passed through the element config.

## Build & Install

Source: `DS9/csrc/nvdsroiexclude/`
- Build
  - `DS9/scripts/build_gst_plugins.sh`
- Install
  - DS9 runtime uses the local plugin path:
    - `export GST_PLUGIN_PATH=$PWD/DS9/gst-plugins:${GST_PLUGIN_PATH:-}`
  - Do not install this plugin over a DS8 runtime tree.
- Verify
  - `gst-inspect-1.0 nvdsroiexclude` (check `config-file`, `id-mode` properties and pad templates)

## Runtime Verification

- Set `[property].osd-mode=1` and confirm green ROI outlines on the composed output.
- Objects entirely inside an ROI disappear before the tracker (track counts drop; ReID load can drop slightly).
- Enable plugin logs: `GST_DEBUG="nvdsroiexclude:6,*:2"` to see one‑time config summary and per‑stream draw notes.

## Troubleshooting

- ROI outlines offset or on wrong tile:
  - Ensure `id-mode` matches how your ROIs are authored: use `pad-index` for tile position, `source-id` for camera identity.
  - Confirm `[property].config-width/height` match the design space used for ROI coordinates (commonly the mux output size, e.g., 1920x1080).
  - Streammux letterboxing is handled automatically via uniform scale + `dx/dy`, so you don’t need to pre‑adjust.
- No overlays but removals occur (or vice versa): ensure `osd-mode` is non‑zero for overlays; both overlay and exclusion share the same transformed polygons.
- Plugin not found: verify `GST_PLUGIN_PATH=$PWD/DS9/gst-plugins:${GST_PLUGIN_PATH:-}` and `LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream-9.0/lib:${LD_LIBRARY_PATH:-}`.
- Python segfaults in metadata probes: this plugin removes the hot path; if needed, temporarily set `NOESIS_DISABLE_ANALYTICS_PROBE=1` during dev.

## Developer Notes

- Element class: `GstNvDsROIExclude` (GObject). We manage C++ members explicitly (heap‑allocated) to avoid GObject/C++ lifetime issues.
- Metadata access: obtain `NvDsBatchMeta*` via `gst_buffer_get_nvds_batch_meta()`; guard traversal with `meta_mutex`.
- Geometry: classic ray‑casting with boundary inclusion; corners‑inside test for bbox containment.
- Tuning: extend `osd-mode` to support fill/labels if desired. Consider class filters or time‑of‑day gating as future enhancements.
- Unit smoke tests:
  - `videotestsrc ! nvvideoconvert ! video/x-raw(memory:NVMM) ! nvdsroiexclude config-file=... ! fakesink`
  - Use `osd-mode=1` for visualization even in synthetic tests (attach `nvdsosd` downstream if you want to see pixels).

## Quick Reference

- Element properties
  - `config-file` (string): path to exclusion INI.
  - `id-mode` (string): `source-id` | `pad-index`.
- Env (pipeline)
  - `NOESIS_DS_EXCLUDE_ID_MODE=pad-index|source-id`
  - `NOESIS_DISABLE_ANALYTICS_PROBE=1` (dev)
  - `NOESIS_DISABLE_MOSAIC_OSD_PROBE=1` (dev)
- Default config path: `pipelines/config_nvdsanalytics_exclude.ini`

## When Adding or Shuffling Streams

Follow this checklist to keep ROI mapping correct when you add/rename/reorder sources:

- Source order: Update `pipelines/noesis_multiurisrcbin.ini` `[source-list].list` so the order matches how you want streams indexed (0..N‑1). The 2x2 tiler tile order is `0=TL, 1=TR, 2=BL, 3=BR`.
- ROI groups per index: In `pipelines/config_nvdsanalytics_exclude.ini`, make sure `[roi-filtering-stream-N]` sections align with the intended stream index. For example, stream 1’s ROIs apply to tile `TR` in a 2x2 grid.
- ID mode: Default is `source-id` (stable per camera). Only switch to `pad-index` if your ROI INI is authored by tile position and you ensure stable pad ordering.
- Stability if using pad-index: Consider setting `sensorID-padID-mapping=1` (in `[source-attr-all]`) and `sort-batch=1` (in `[streammux]`) in `pipelines/noesis_multiurisrcbin.ini` to make pad indices deterministic.
- Visual validation: Temporarily set `[property].osd-mode=1` in `config_nvdsanalytics_exclude.ini` and run with `GST_DEBUG="nvdsroiexclude:6,*:2"` to see a one‑time config summary and confirm ROI outlines appear on the intended tiles.
