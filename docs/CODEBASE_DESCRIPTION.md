# Codebase description

Status: native DeepStream 9.1 architecture, 2026-08-15.

## Runtime ownership

Noesis has one supported perception runtime. The launcher
`DS9/noesis/ds9_runtime.py` establishes DS9.1 import and artifact authority,
then invokes `DS9/noesis/ds9_runtime_core.py`. The core assembles the pipeline,
shared product services, WebSocket/REST boundaries, identity, world state,
depth storage, and shutdown lifecycle.

`noesis/` and `noesis_core/` are shared application libraries. They are not a
second DeepStream runtime. A few mirrored files retain DS8-era filenames, most
notably `DS9/noesis/pipelines/ds8_pipeline.py`; the containing DS9 tree and
launcher establish their active authority.

## Component map

```mermaid
flowchart TB
    subgraph HOST[Native Ubuntu host]
        SYSTEMD[noesis-appliance.service]
        SUP[DS9 native supervisor]
        ENTRY[DS9 runtime entrypoint]
        GRAPH[Service Maker / DeepStream graph]
        SHARED[Shared noesis + noesis_core services]
        STATE[(World · identity · scene · depth state)]
        WS[WebSocket + WebRTC signaling :6008]
        REST[REST :8080]
        MEDIA[H.264 SHM media edge]
    end

    CAMS[Living · Kitchen · Family cameras] --> GRAPH
    SYSTEMD --> SUP --> ENTRY --> GRAPH
    ENTRY --> SHARED
    GRAPH --> SHARED
    SHARED <--> STATE
    SHARED --> WS
    SHARED --> REST
    GRAPH --> MEDIA

    subgraph MENON[Menon browser boundary]
        GATEWAY[Authenticated gateway]
        UI[oai2-fe dashboard]
    end

    WS --> GATEWAY
    REST --> GATEWAY
    MEDIA --> GATEWAY
    GATEWAY --> UI
```

## Source ownership

| Area | Owner paths | Notes |
| --- | --- | --- |
| Runtime startup | `DS9/noesis/ds9_runtime.py`, `DS9/scripts/run_canonical_runtime_host.py` | Native-only platform, venv, secret, artifact, and port checks |
| Pipeline graph | `DS9/noesis/pipelines/`, `DS9/config/infer.yaml`, `DS9/pipelines/` | Service Maker graph and inference configs |
| Native metadata | `DS9/native/`, `DS9/gst-plugins/`, `DS9/csrc/` | DS9.1/CUDA 13.2 build authority |
| Runtime APIs | `noesis/server/`, `DS9/noesis/server/` | Shared contracts with DS9-specific adapters where required |
| Telemetry/world | `noesis/telemetry/`, `noesis_core/`, `DS9/noesis/telemetry/` | Transactional tracking/world and BEV publication |
| Identity | `reid/`, `noesis/identity_v2_service.py`, `noesis/server/reid_api.py` | Stable IDs and resident/visitor policy |
| Depth/reconstruction | `geometry/`, `DS9/noesis/calibration/`, `tools/mapanything_phone_scan/` | DAv2, MapAnything, registration, Scene Prior, offline reconstruction |
| Dashboard | `oai2-fe/` | Noesis diagnostic UI; Menon owns browser-facing serving/auth |
| Contracts | `contracts/`, `docs/api_contracts_*.md`, `docs/metadata_contracts.md` | JSON schema, TypeScript, and narrative contracts |

## Authority flow

```mermaid
sequenceDiagram
    participant C as Camera
    participant P as DS9.1 pipeline
    participant W as Canonical world service
    participant T as WS/REST/media boundaries
    participant M as Menon gateway
    participant U as Dashboard

    C->>P: decoded NVMM frames
    P->>P: detect, track, ReID, pose, depth
    P->>W: timestamped observations
    W->>W: commit tracking/world transaction
    W->>T: release tracking, world, then BEV cohort
    P->>T: encoded H.264 mosaic
    T->>M: authenticated loopback transports
    M->>U: browser session, telemetry, media, controls
```

Publication lifecycle is deliberate: the runtime closes and drains the shared
publication gate before stopping WebSocket egress so no admitted callback races
the transport shutdown.

## Legacy boundary

DS8, DS9.0, and container deployment code or documents remain temporarily only
for the pending destructive cleanup in
`plans/ds91_native_host_only_migration.md`. They own no runtime, model, service,
or validation authority. Current work must never depend on them.
