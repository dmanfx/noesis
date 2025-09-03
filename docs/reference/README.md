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

### `Dynamic_Sensors.md`
Purpose: How to add/remove sources at runtime using `nvmultiurisrcbin` REST API.

## Usage

Use these documents as the authoritative reference for architecture, configuration, implementation, performance, and runtime APIs.

## Maintenance

Update these docs when pipeline elements, configs, or runtime interfaces change. 
