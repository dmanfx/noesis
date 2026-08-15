# DepthResult telemetry contract

`DepthResult` is the compact metadata envelope emitted for a successful
MapAnything full-frame result. It is not the always-on DAv2 object-depth
payload and it does not expose a filesystem path.

## Required fields

| Field | Meaning |
| --- | --- |
| `source_id` | Logical source index |
| `frame_id` | Source frame identifier |
| `ts` | Capture timestamp under the current contract |
| `width`, `height` | Positive raster dimensions |
| `depth_map_ref` | Opaque `noesis-depth://artifact/<sha256>` reference |
| `minmax` | Two finite depth range values |
| `unit` | `m` |

`DepthResult.to_public_dict()` is the public serialization authority. Private
storage paths are hashed into the opaque reference before publication. Clients
must not parse the reference or use it as a URL.

The manual `ma_depth_response` contract returns bounded, digest-bound component
descriptors and same-origin URLs through Menon. It does not inline unbounded
depth tensors. See `api_contracts_ws.md` for that full RPC shape and
`depth_metadata.md` for the separate DAv2 object-depth path.

Menon/oai2-fe may use `minmax` for presentation, but calibrated geometry comes
from the bound depth, camera, registration, and floorplan contracts—not from
the display range alone.
