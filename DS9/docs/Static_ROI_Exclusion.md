# Native static ROI exclusion

`nvdsroiexclude` is an in-place metadata-only `GstBaseTransform` in the
canonical DS9.1 graph:

```text
YOLO26 PGIE → nvdsroiexclude → NvDCF → nvdsanalytics
```

It removes detections fully contained by configured static polygons before
tracking, ReID, pose, world, and telemetry work. The Python hooks do not emulate
removal and there is no legacy fallback.

## Authority

- Source: `DS9/csrc/nvdsroiexclude/`.
- Runtime plugin: native DS9.1 artifact selected by the supervisor.
- Pipeline wiring: `DS9/noesis/pipelines/ds8_pipeline.py` (legacy filename,
  DS9-owned copy).
- Public config: `DS9/config/infer.yaml` `analytics.exclude`.
- Runtime state/config: path selected by `NOESIS_ANALYTICS_EXCLUDE_CONFIG`.
- REST persistence/reload: `DS9/noesis/server/analytics_api.py`.

## Polygon and removal semantics

- INI sections are `roi-filtering-stream-N`; `roi-<label>` contains at least
  three polygon points.
- `source-id` is the default stream key. `pad-index` is allowed only when the
  config is intentionally authored by stable tile order.
- Config-space points are uniformly scaled and letterbox-offset into the
  pipeline frame.
- A bounding box is removed when all four corners lie in an enabled polygon;
  the boundary counts as inside.
- Optional OSD mode attaches display metadata only; it does not alter pixels.

## Reload contract

The ROI editor updates only the pre-tracker exclusion stage. A reload is
accepted only after the plugin reports the exact expected config hash and
sequence. Failed reload does not claim success or mutate the post-tracker room
taxonomy.

## Focused validation

After source/plugin changes:

1. Rebuild only `nvdsroiexclude` against the native DS9.1 stack.
2. Run `gst-inspect-1.0 nvdsroiexclude` with the native plugin path/registry.
3. Run the focused parser/reload tests.
4. Use one short ROI reload smoke and confirm detections inside the edited
   polygon disappear before tracker counts.

Do not install or load the archived root DS8 plugin.
