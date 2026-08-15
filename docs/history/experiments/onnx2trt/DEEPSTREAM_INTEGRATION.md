# DeepStream Fused SGIE Integration Notes

This document captures how the fused-input MapAnything TensorRT plan is consumed by DeepStream using standard `nvinfer`, how the custom preprocess library prepares tensors, and how depth metadata is attached to tracked objects.

## Element Placement

```
nvmultiurisrcbin → nvdspreprocess (primary) → nvinfer (YOLO PGIE)
  → queue (q_after_pgie) → nvdsanalytics_exclude
  → queue (q_after_exclude) → nvdspreprocess (mapanything_preprocess)
  → nvinfer (mapanything_sgie_fused UID=22)
  → queue (q_before_tracker) → nvtracker → …
```

Key points:

- `pipelines/config_preprocess_mapanything_fused.ini` configures the custom preprocess library `libmapanything_preprocess_fused.so`. It crops each PGIE ROI to 518×518 FP16 tensors, normalizes RGB, and appends nine intrinsics channels (broadcast per pixel) by reading `MA_INTRINSICS_TABLE`.
- `pipelines/config_infer_secondary_mapanything_fused.ini` points `nvinfer` at the fused TensorRT plan (`models/engines/ma_model_fp16_b3_fused.plan`). The config enables `output-tensor-meta=1` so tensor metadata travels downstream, and assigns `gie-unique-id=22`.
- The pipeline constructor resolves both config paths from `processing.DEEPSTREAM_MAPANYTHING_PREPROCESS_CONFIG` and `processing.DEEPSTREAM_MAPANYTHING_SGIE_CONFIG`, falling back to the repo defaults. It also ensures `MA_INTRINSICS_TABLE` is set before the pipeline runs.

## Pad-Probe Post Processing

- Pad probe `_mapanything_depth_probe` is registered on `mapanything_sgie_fused`’s src pad in `deepstream_video_pipeline.py`.
- Helpers in `pipelines/mapanything_depth_postprocess.py`:
  - Extract depth / confidence / mask layers from `NvDsInferTensorMeta`
  - Compute median, p10, p90, confidence mean, and valid ratio per ROI
  - Sample depth around the detection anchor and attach rich `mde` JSON as `NVDS_USER_OBJ_META`
  - Fall back to floor-plane intersection when tensor output is invalid

Metadata Example:

```json
{
  "mde": {
    "depth_m": 1.24,
    "conf": 0.78,
    "median": 1.22,
    "p10": 1.05,
    "p90": 1.38,
    "valid_ratio": 0.62,
    "samples": 12,
    "summary_sample_count": 276,
    "scale": 0.97,
    "pose": [0.01, -0.12, 0.34, ...],
    "world": [0.45, 1.82, 0.15]
  }
}
```

## Validation Checklist

- `deepstream_video_pipeline.py` logs show `NvDsInferContext[UID 22]` loading `ma_model_fp16_b3_fused.plan`.
- Custom preprocess library emits no warnings about missing intrinsics; confirm `MA_INTRINSICS_TABLE` points at the expected table.
- Pad-probe logs (rate-limited) include `MapAnything depth cam=…` entries for detected tracks.
- Downstream telemetry (tracker / websocket) continues to include `depth["mde"]`.
