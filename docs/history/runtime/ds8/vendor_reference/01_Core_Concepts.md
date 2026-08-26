# 01. DeepStream Core Concepts & Architecture (DS8)

This guide summarizes the DeepStream 8 architecture and core concepts used across our pipelines.

## What is DeepStream?

DeepStream is a GStreamer-based SDK for high-performance, real-time video/audio analytics on NVIDIA GPUs. It provides GPU-accelerated plugins for decode, batching, inference, tracking, analytics, and visualization. DeepStream 8 targets modern platforms (x86/dGPU, ARM SBSA, Jetson) and integrates tightly with TensorRT 10 and Triton.

## High-Level Architecture

A DeepStream application is a GStreamer pipeline: data flows from sources through processing elements to sinks. Typical video analytics stages:

1. Input and Decode
   - Sources from files, RTSP/HTTP, cameras; decode via NVDEC.
   - In DS8, `nvmultiurisrcbin` simplifies multi-URI ingestion and batching.
2. Preprocess and Batching
   - `nvstreammux` (legacy) or `nvstreammux` (new) batches frames for GPU efficiency.
   - Optional `nvdspreprocess` or `nvinfer` internal preprocessor for resize/color/normalize.
3. Inference
   - `nvinfer` for TensorRT engines; `nvinferserver` for Triton-served models.
   - Outputs attach inference metadata to frames/objects.
4. Tracking
   - `nvtracker` assigns persistent IDs for detections across frames.
5. Secondary Inference
   - Additional `nvinfer` on object crops for classification/attributes.
6. Analytics and Visualization
   - `nvdsanalytics` for rules/ROI metrics; `nvdsosd` for OSD.
7. Output
   - Display sinks, file sinks, RTSP out, or metadata egress via `nvmsgconv`/`nvmsgbroker`.

## DS8 Key Building Blocks

- Sources
  - `nvmultiurisrcbin` for multi-URI ingest + built-in decode + mux.
  - `nvurisrcbin` for single URI ingest.
- Batching
  - `nvstreammux` (legacy) and `nvstreammux` (new). The new mux adds richer config and tuning; see 12_NvStreamMux_New_Guide.md.
- Inference
  - `nvinfer` for local TensorRT engines. `nvinferserver` to call Triton (HTTP/gRPC) with NvDsTritonExt integration.
- Metadata
  - Frame/meta structures carry detections, classifications, tracks, analytics, and user meta across the pipeline.

## Metadata Overview

As buffers traverse the pipeline, plugins add or modify metadata:
- `NvDsBatchMeta` per batched buffer, `NvDsFrameMeta` per frame.
- `NvDsObjectMeta` for detections with bbox, label, confidence, and track `object_id`.
- `NvDsClassifierMeta` for class/attributes.
- `NvDsUserMeta` for custom payloads; avoid copying surfaces to host unless necessary.

## Application Development Patterns

Two common approaches:
- Programmatic pipelines (Python/C/C++): create elements, set properties, link pads, and attach pad probes to read/write metadata.
- Config-driven pipelines (deepstream-app): reference app that builds pipelines from config files.

General workflow:
1. Design the pipeline (sources → mux → infer → track → analytics → sinks).
2. Configure element properties (batch size, dimensions, models, trackers).
3. Access metadata in pad probes to implement app logic and egress.
4. Tune performance and memory; iterate with profiling.

See also:
- 02_GStreamer_Plugins.md for element details.
- 12_NvStreamMux_New_Guide.md for DS8 mux tuning.
- 10_Triton_Integration.md if serving models via Triton.
