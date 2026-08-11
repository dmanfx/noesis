# MapAnything Fused Preprocess Library

This directory contains the custom `nvdspreprocess` implementation that prepares the fused
`(B, 12, H, W)` tensor consumed by the MapAnything SGIE. The library performs the following
steps for every ROI in the batch:

- Copies the scaled RGB crop produced by `nvdspreprocess`.
- Normalizes pixel values by `pixel-normalization-factor` (defaults to `1/255`).
- Appends nine additional channels which broadcast the calibrated camera intrinsics so the SGIE
  receives a single fused tensor.

## Build Instructions

The library depends on the DeepStream SDK headers and CUDA runtime. Adjust `DEEPSTREAM_SDK_ROOT_DIR`
or `DS_SDK_ROOT` if DeepStream is installed in a non-default location.

```bash
cd pipelines/mapanything_preprocess_fused
make
```

The resulting shared object is written to:

```
pipelines/mapanything_preprocess_fused/libmapanything_preprocess_fused.so
```

You can also use CMake directly:

```bash
cmake -S pipelines/mapanything_preprocess_fused -B pipelines/mapanything_preprocess_fused/build
cmake --build pipelines/mapanything_preprocess_fused/build
```

## Configuration

Use the provided DeepStream preprocess configuration `pipelines/config_preprocess_mapanything_fused.ini`.
Key properties:

- `tensor-name=mapanything_fused` – matches the SGIE input layer name.
- `tensor-data-type=5` – requests FP16 tensors to align with the TensorRT engine.
- `network-input-shape=3;12;518;518` – DeepStream 7.1 expects batch;channel;height;width order for NCHW; match this to the SGIE batch size.
- `scaling-filter` and `scaling-pool-memory-type` must be provided for the 7.1 parser; `1` and `2` request bilinear scaling with CUDA device buffers.
- `intrinsics-table` – path to the whitespace-separated calibration table. Each row adopts the
  format `source_id fx 0 cx 0 fy cy 0 0 1`. Missing entries fall back to the identity matrix.

Place the intrinsics table at `models/mapanything_depth/intrinsics_table.txt` (sample file included)
or override the path via the `[user-configs]` section.
