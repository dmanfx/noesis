# MapAnything Heatmap Viewer (3‑Stream)

This repo includes a small proof‑of‑concept script that runs the MapAnything TensorRT engine on the three camera-secret references listed in `pipelines/noesis_multiurisrcbin.ini` and renders low-FPS depth heatmaps.

## Script

`scripts/ma_heatmap_multiuri.py`

### What it does
- Resolves `camera-secret:*` entries from `pipelines/noesis_multiurisrcbin.ini`
  through the owner-only registry documented in `Runtime_Secrets.md`.
- Builds a lightweight GStreamer pipeline: `nvmultiurisrcbin → nvinfer (MapAnything) → fakesink`.
- Extracts tensor meta (depth/conf/mask), converts to a heatmap, and displays it in windows.
- Optional JPEG dumps for offline verification.

## Quick Start

```bash
python3 scripts/ma_heatmap_multiuri.py \
  --width 1280 --height 720 \
  --infer-interval 15 \
  --ignore-mask \
  --colormap turbo \
  --show-text
```

## Key Options

- `--width / --height`: output size for `nvmultiurisrcbin` (controls scale + GPU load).
  - Good next step down from 1280×720: `960×540` (or `960×544` for 16‑pixel alignment).
- `--infer-interval`: frame skipping on the **inference** side (largest GPU saver).
  - Example: `--infer-interval 15` ≈ 1/16 frames.
- `--sink`: display sink (default `glimagesink`).
  - Try `xvimagesink` if GL output fails.
- `--ignore-mask`: render depth without applying the mask (useful if mask is sparse).
- `--dump-dir / --dump-every`: write JPEGs periodically for debug.

## Debug / Validation

Write JPEG heatmaps to `/tmp`:

```bash
python3 scripts/ma_heatmap_multiuri.py \
  --width 1280 --height 720 \
  --infer-interval 15 \
  --ignore-mask \
  --dump-dir /tmp --dump-every 2 \
  --debug
```

Check:
- `/tmp/heatmap_stream0.jpg`
- `/tmp/heatmap_stream1.jpg`
- `/tmp/heatmap_stream2.jpg`

If JPEGs look correct but windows are blank/green, switch sinks:

```bash
--sink glimagesink
# or
--sink xvimagesink
```

## Performance Tips

Biggest wins:
- Increase `--infer-interval`
- Reduce `--width/--height`
- Run fewer streams

Tiling the **display windows** does **not** materially reduce GPU load; inference dominates.

## Dependencies

- DeepStream 8 runtime + `pyds`
- TensorRT engine: `models/mapanything_depth/1/model.plan`
- MapAnything nvinfer config: `pipelines/config_infer_secondary_mapanything.ini`

## Status
- Proof-of-concept helper for the legacy GI pipeline. The canonical DS8 stack
  uses the full-frame MapAnything SGIE branch documented in
  `MapAnything_Depth.md` and `DS8_Baselines.md`. Keep this script/doc for
  one-off debugging; it is not part of the supported DS8 runtime.
