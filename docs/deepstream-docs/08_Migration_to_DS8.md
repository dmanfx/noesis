# 08. Migration to DeepStream 8

This page distills what changes when moving from DeepStream 6.x/7.x to 8 and provides a practical checklist to get applications running quickly.

## What Changed in DS8

- TensorRT 10.x
  - x86/dGPU uses TensorRT 10.9; Jetson uses TensorRT 10.13.
  - Engines built with TensorRT 8.x are not compatible; rebuild engines and, for INT8, regenerate calibration caches.
- New/updated plugins
  - `nvmultiurisrcbin` simplifies multi-URI ingest + internal muxing.
  - `nvstreammux` (new) adds richer muxing and tuning; legacy mux remains available.
  - REST support for select components (e.g., analytics and tracker) enables runtime control.
- Containers and platforms
  - Official Docker images for x86/ARM and Jetson. Development inside x86 containers is supported.
  - New platforms supported (e.g., Blackwell dGPU, Jetson Thor). Jetson packages are based on newer JetPack BSPs.
- OpenCV defaults
  - OpenCV support is disabled by default for sample plugins. Enable in Makefiles if needed.

## Migration Checklist

1. Rebuild model engines
   - Export to ONNX and rebuild TensorRT engines using TensorRT 10.
   - Re-run INT8 calibration with a representative dataset if using INT8.
2. Update pipeline muxing
   - If using `nvstreammux`, review whether to keep legacy or adopt the new mux. See 12_NvStreamMux_New_Guide.md for differences and tuning.
3. Validate tracker behavior
   - Confirm tracker library choice (IOU/NvDCF) and configs; REST control is available in DS8 for some use cases.
4. Confirm plugin availability
   - Some GStreamer codecs/utilities may need installation inside containers (e.g., `gstreamer1.0-libav`, `-good`).
5. Measure `nvinfer` vs `nvinferserver`
   - Triton integration is convenient but may have different performance characteristics for some models; validate throughput and latency.
6. Refresh sample config files
   - DS8 sample configs and schema names may have evolved. Start from DS8 samples and transpose custom values.

## Quick Steps to Rebuild Engines

- Using `trtexec` (example):
  - FP16: `trtexec --onnx=model.onnx --saveEngine=model_fp16.engine --fp16`
  - INT8: `trtexec --onnx=model.onnx --saveEngine=model_int8.engine --int8 --calib=calib.cache --calibInput=model_input --optShapes=input:1x3xH xW`

Ensure the `batch-size` in `nvinfer` config matches the upstream mux’s batch size. See 04_Custom_Models.md for more details.

## Testing Tips

- Clear GStreamer registry when switching mux variants or plugin builds: `rm -f ~/.cache/gstreamer-1.0/registry.*.bin`.
- In containers, always start with `--gpus all` and bind host display if needed for visualization.
- Use `GST_DEBUG=3` for link issues and `GST_DEBUG_DUMP_DOT_DIR` to inspect pipeline graphs.
