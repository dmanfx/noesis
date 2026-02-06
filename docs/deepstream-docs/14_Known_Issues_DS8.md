# 14. Known Issues and Notes (DS8)

This page summarizes notable caveats observed in DS8 environments. Always consult NVIDIA’s release notes for authoritative, up-to-date details.

## Platform and versions

- DS8 supports newer dGPU architectures and Jetson variants. Match your NVIDIA driver, CUDA, TensorRT, and DS package versions.
- On Jetson, DS images are tied to specific JetPack BSP versions; confirm compatibility before upgrading.

## Triton specifics

- Some models show lower throughput or higher latency via `nvinferserver` compared to native `nvinfer`. Measure both options and choose per model.
- Backend availability varies by platform and container tag. TensorFlow backend may be unavailable in some DS8 Triton builds; check the container’s release notes.

## GStreamer plugins

- If you see missing demux/parse warnings in containers, install `gstreamer1.0-libav` and `gstreamer1.0-plugins-good` inside the container.
- When switching between legacy and new `nvstreammux`, clear the GStreamer registry cache to ensure `gst-inspect` and runtime use the intended variant.

## nvmultiurisrcbin behaviors

- Prefer `live-source=1` in the embedded mux to avoid stalling when active sources < max batch.
- Keep `drop-pipeline-eos=1` to allow new sources after all previous sources reached EOS.
- The new mux may not be supported for all audio+video mixes in this bin; validate your case.

## Miscellaneous

- OpenCV features are disabled by default for certain DS8 sample plugins; enable at build time if required.
- For display issues on headless systems, pass correct X11/Wayland bindings to containers or use fakesink for headless runs.
