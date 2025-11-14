# Reference Documentation

This folder contains the core reference documentation for the DeepStream video processing pipeline implementation.

## Documents

### `pipeline_flow.md`
Purpose: Comprehensive documentation of the DeepStream video processing pipeline flow and architecture.

### `DEEPSTREAM_PIPELINE_MAP.md`
Purpose: Visual mapping and detailed specification from RTSP input to WebSocket output.

### `WebSocket_API.md`
Purpose: Message types and schemas for the WebSocket server, including runtime config updates and telemetry frames.

### `Telemetry_Schema.md`
Purpose: Detailed schema for analytics/telemetry payloads and stats produced by the pipeline.

### `DeepStream_Elements_and_Probes.md`
Purpose: Inventory of DeepStream elements, properties, probes, and dynamic branch behavior.

### `Configuration_Map.md`
Purpose: Mapping from `config.py` to pipeline components and runtime controls.

### `Occupancy_Publishing.md`
Purpose: Reference for the MQTT + InfluxDB occupancy publisher and HomeSeer integration.

### `Integrations_Playbook.md`
Purpose: Step-by-step integration guide for InfluxDB + HomeSeer and a reusable pattern to add future integrations and stats.

### `Metadata_Ops.md`
Purpose: How and when to use the shared metadata operator helpers (`pipelines/meta_ops.py`) to traverse, read, and modify DeepStream metadata (DS8 operator with DS7 fallbacks).

### `Static_ROI_Exclusion.md`
Purpose: Detailed reference for the `nvdsroiexclude` plugin (static ROI pruning pre‑tracker), config syntax, stream keying (`id-mode`), coordinate mapping with tiler, OSD overlays, build/install, and troubleshooting.

### `Dynamic_Sensors.md`
Purpose: How to add/remove sources at runtime using `nvmultiurisrcbin` REST API.

### `MapAnything_Depth.md`
Purpose: End-to-end reference for the MapAnything depth microservice, adapters, storage, diagnostics, and operations workflows.

## Usage

Use these documents as the authoritative reference for architecture, configuration, implementation, performance, and runtime APIs.

## Maintenance

Update these docs when pipeline elements, configs, or runtime interfaces change. 
