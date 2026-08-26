# NVIDIA DeepStream SDK 8 – Project Docs

> **Historical reference:** DeepStream 8 is retired in Noesis. This captured
> reference is for migration provenance only and does not describe the native
> DeepStream 9.1 application.

This folder provides a self-contained, distilled reference for NVIDIA DeepStream SDK 8. Content is paraphrased and tailored for fast, offline lookup by engineers and agents working in this repo. It focuses on x86/dGPU and adds Jetson notes where behavior diverges.

Start with 01_Core_Concepts.md for a quick mental model. Use the plugin and cheat sheets for implementation details and the migration and Docker pages when moving between environments or versions.

## DeepStream 8 Highlights

- Supports latest NVIDIA platforms including Blackwell dGPUs and Jetson Thor.
- Ships Docker images for x86/ARM and Jetson; development inside x86 containers is supported.
- Integrates Triton Inference Server; use `nvinferserver` or NvDsTritonExt for model serving.
- New/updated plugins and flows: `nvmultiurisrcbin`, `nvstreammux` (new), REST control for select plugins.
- TensorRT 10.x required; engines from older DS may need rebuild and recalibration.

## Table of Contents

1.  Core Concepts & Architecture – `01_Core_Concepts.md`
2.  GStreamer Plugins (updated for DS8) – `02_GStreamer_Plugins.md`
3.  Sample Applications – `03_Sample_Applications.md`
4.  Using Custom Models – `04_Custom_Models.md`
5.  Performance & Tuning – `05_Performance.md`
6.  Troubleshooting & FAQ – `06_Troubleshooting.md`
7.  3D DeepStream Overview – `07_3D_DeepStream.md`
8.  Migration to DeepStream 8 – `08_Migration_to_DS8.md`
9.  Docker & Installation – `09_Docker_and_Installation.md`
10. Triton Integration – `10_Triton_Integration.md`
11. nvmultiurisrcbin Cheat Sheet – `11_NvMultiUriSrcBin_CheatSheet.md`
12. nvstreammux (New) Guide – `12_NvStreamMux_New_Guide.md`
13. REST API Server – `13_REST_API_Server.md`
14. Known Issues & Limitations – `14_Known_Issues_DS8.md`

These docs summarize NVIDIA’s official DeepStream 8 developer guide and release notes. For full details, consult NVIDIA’s site.
