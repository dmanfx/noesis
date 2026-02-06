# DepthResult Telemetry Contract

## Overview
The Noesis DS8 pipeline emits compact `DepthResult` payloads for every depth burst that
Menon and other downstream consumers must ingest. This document codifies the payload
shape, field semantics, and serialization rules so that the contract remains stable
during the DeepStream 8 migration.

## Schema
All payloads are JSON objects that mirror the `DepthResult` dataclass in
`noesis/metadata/depth_result.py`. Fields, types, and semantics:

- `source_id` (`int`): Logical camera/source identifier supplied by DeepStream.
- `frame_id` (`int`): Incrementing frame counter within the source stream when the depth
  snapshot was produced.
- `ts` (`int`): Unix epoch seconds representing the depth snapshot timestamp. Naive
  datetimes are interpreted as UTC before serialization.
- `width` (`int`): Width of the depth raster in pixels. Must be greater than zero.
- `height` (`int`): Height of the depth raster in pixels. Must be greater than zero.
- `depth_map_ref` (`string`): Stable locator where the depth map is stored. Examples:
  `zarr://depth/snapshots/cam01/1710787902`, `s3://noesis-depth/cam03/1710787902.zarr`.
- `minmax` (`[float, float]` or `{min:float, max:float}`): Minimum and maximum metric
  depth values (meters) observed in the raster. The JSON encoding always publishes this
  field as a two-element array `[min, max]`.
- `unit` (`string`): Measurement unit for the depth values. Current value is always `"m"`
  (meters) but the field is retained for forward compatibility.

Optional producers may attach additional metadata alongside the core contract. Unknown
keys must be ignored by consumers to preserve backwards compatibility.

## Serialization Rules
1. The canonical implementation lives in `DepthResult.to_dict()` / `to_json()`. Other
   languages must maintain identical key names and types.
2. `minmax` accepts either a two-element sequence or a mapping with `min`/`max` keys when
   constructing `DepthResult`. Consumers should expect the serialized JSON to contain an
   array because the contract normalizes it.
3. Producers must provide non-empty `depth_map_ref` strings and positive image
   dimensions; the serializer raises a `ValueError` if these constraints are violated.
4. Timestamps must remain in **seconds**. Millisecond or microsecond precision should be
   sent through auxiliary fields if required.

## Sample Payload
```json
{
  "source_id": 7,
  "frame_id": 19342,
  "ts": 1715116805,
  "width": 1920,
  "height": 1080,
  "depth_map_ref": "zarr://depth/snapshots/cam07/1715116805",
  "minmax": [0.42, 18.75],
  "unit": "m"
}
```

## Frontend Integration Notes (oai2-fe)
- The oai2-fe React dashboard consumes `DepthResult` messages over the websocket telemetry channel and maps them to its local model.
- The `depth_map_ref` key is opaque; oai2-fe resolves it via the configured depth store (e.g., Zarr over HTTP). Do not infer semantics from the URI.
- Use `unit` when presenting UI overlays (currently meters).
- Apply `minmax` for color‑map scaling prior to loading Zarr chunks to prevent out‑of‑range gradients.

Compatibility
- Existing Menon consumers remain supported on the same websocket topics. Payload format and semantics are identical.

## Versioning
The current contract is tagged as **DepthResult v1**. Any additive changes require
updating this document and bumping the contract version so that Menon can deploy the
corresponding consumer update in lockstep.
