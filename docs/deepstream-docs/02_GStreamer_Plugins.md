# 02. DeepStream GStreamer Plugins (DS8)

Quick reference for essential DeepStream 8 plugins used in our pipelines. Grouped by purpose with key properties and usage tips.

---

Sources

- `nvmultiurisrcbin` (multi-URI source + mux)
  - Function: Ingests multiple URIs and internally creates `nvurisrcbin` instances and a `nvstreammux` to batch them.
  - Highlights: Dynamic add/remove; reconnection; REST control; properties to configure internal `nvstreammux` and per-source behavior.
  - See 11_NvMultiUriSrcBin_CheatSheet.md for examples and property notes.
- `nvurisrcbin` (single URI)
  - Function: Ingests one URI with decode, depay, parse, and caps handling.
  - Use when you need fine-grained control per source without the multi-bin wrapper.

---

Batching and Composition

- `nvstreammux` (legacy)
  - Function: Batches decoded frames from multiple sources into a single `NvBufSurface` with `batch-size` frames.
  - Key properties: `width`, `height`, `batch-size`, `batched-push-timeout`, `live-source`.
- `nvstreammux` (new)
  - Function: Updated mux with richer configuration for mixed source types, audio/video sync, cascaded muxing, and metadata handling.
  - See 12_NvStreamMux_New_Guide.md for tuning patterns and differences from legacy mux.
- `nvmultistreamtiler`
  - Function: Creates a visible mosaic for multiple streams; use for dashboards and debugging.

---

Inference

- `nvinfer`
  - Function: TensorRT-based inference on batched surfaces; primary and secondary modes supported.
  - Key props: `config-file-path`, `model-engine-file`, `batch-size` (must match mux), `process-mode`, `unique-id`.
  - Output: Attaches `NvDsInferTensorMeta` and object/classifier metadata to frames/objects.
- `nvinferserver`
  - Function: Uses Triton Inference Server (HTTP/gRPC) for inference. Suitable for multi-model, multi-framework serving.
  - Notes: See 10_Triton_Integration.md for supported backends and config examples.
- `nvdspreprocess`
  - Function: Flexible pre-processing (resize, crop, color, normalization) for complex multi-model flows.

---

Tracking and Analytics

- `nvtracker`
  - Function: Assigns persistent IDs across frames using IOU, NvDCF, etc.
  - Key props: `ll-lib-file`, `ll-config-file`, `gpu-id`.
- `nvdsanalytics`
  - Function: ROI/line rules and derived events (counts, direction, occupancy).
  - Key prop: `config-file-path` for rules.

---

Visualization and Utilities

- `nvdsosd`
  - Function: GPU OSD for bboxes/text; keep `process-mode=0` for GPU.
- `nvvideoconvert`
  - Function: Efficient GPU color/format/resize conversion. DS8 adds format coverage on x86.
- `nvdewarper`
  - Function: De-warp fisheye/360 sources; GPU-accelerated.

---

IoT Egress

- `nvmsgconv` and `nvmsgbroker`
  - Function: Convert `NvDsBatchMeta` into JSON/protobuf and send to Kafka/MQTT/HTTP via adapters.
  - Key props: `config-file-path` (schema), `proto-lib`, `conn-str`, `topic`.

---

Notes and DS8 specifics

- REST API: DS8 introduces REST configuration/control for select elements (e.g., analytics and tracker); see 13_REST_API_Server.md.
- OpenCV in plugins: OpenCV is disabled by default in DS8; enable at build time for sample plugins if needed.
- Triton vs TensorRT: Some models perform better with local `nvinfer`; measure and choose accordingly.
