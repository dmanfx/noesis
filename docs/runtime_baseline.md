# Native DeepStream 9.1 runtime baseline

Status: accepted application baseline, updated 2026-08-25.

## Platform

| Component | Canonical value |
| --- | --- |
| DeepStream | 9.1.0 |
| SDK root | `/opt/nvidia/deepstream/deepstream-9.1` |
| CUDA | 13.2 (`/usr/local/cuda-13.2`) |
| TensorRT | 10.16.0.72 (`TensorRT v101600`) |
| GStreamer | 1.24.2 |
| Python | 3.12 native virtual environment |
| Driver floor | 595.58.03 |
| Validated host driver | 595.71.05 |
| Runtime backend | `native_host` |

The canonical service invokes `DS9/scripts/run_canonical_runtime_host.py`; it
does not use Docker, a deployment selector, or a container filesystem.

## Selected application lane

- Config: `DS9/config/infer.yaml`.
- Cameras: Living Room, Kitchen, Family Room via `nvurisrcbin`.
- Batch: 3 at 1920×1080, live input, 40 ms mux timeout.
- Detector: YOLO26-m FP16, GIE 1.
- Tracker: NvDCF baseline.
- ReID: Swin Tiny, 256-dimensional embedding, GIE 3.
- Pose: YOLO26-n pose, GIE 4.
- Tracking depth: DepthAnythingV2 metric, always on, GIE 5.
- Person world localization: universal typed floor/depth hypothesis resolver
  with full covariance, conservative PCF evidence, then the shared
  `PersonGroundState`; no room-specific localization policy.
- Manual/full-frame depth: MapAnything FP32, request-gated, GIE 2.
- Analytics: native ROI exclusion followed by `nvdsanalytics`.
- Output: one 3840×720 H.264 mosaic through SHM/WebRTC.
- RTSP: disabled.
- BEV: JSON/metadata only in `camera_local_ground_m`; PCF is the canonical
  floorplan presentation source where a camera has an admitted Scene Prior.
  The off-by-default dashboard `Localization details` overlay compares the
  exact current hypotheses, uncertainty, and retired room-policy measurement
  without changing the canonical dot. The old policy is diagnostic-only and
  cannot become runtime localization authority.

MV3DT, V3DT activation, and AMC are not part of this baseline. Assets/configs
for future experiments do not constitute an enabled capability.

## Model realization

The accepted DS9.1 realization digest is:

```text
9c3815bbf86eb41a94efb504fad79a9208c543e05f98c68d017cb1a8dcfd2b26
```

The canonical host gate requires the five selected engines: YOLO26 detection,
Swin ReID, YOLO26 pose, DAv2 tracking depth, and MapAnything. It rejects DS8,
DS9.0, Docker-overlay, or mismatched native origins.

## Network and browser boundary

| Interface | Binding | Owner |
| --- | --- | --- |
| WebSocket telemetry and WebRTC signaling | `127.0.0.1:6008` | Noesis |
| REST | `127.0.0.1:8080` | Noesis |
| Mosaic media | private SHM socket | Noesis → Menon |
| Browser TLS/session/dashboard | LAN-facing gateway | Menon |

Noesis requires private camera, MapAnything, and internal bearer-token files.
Browsers never receive the internal token or connect directly to the Noesis
ports.

## Performance reference

The accepted occupied-scene baseline preserves all selected models, input
resolution, inference cadence, tracking, depth, pose, identity, world/BEV, and
WebRTC output. It does not obtain throughput by skipping inference or reducing
quality.

The 2026-08-23 recovery removed a full 3840x720 RGBA device-to-host and
host-to-device round trip from every mosaic frame, made secondary DAv2 work
latest-frame-only and readiness-query-only, pooled native CUDA/pinned resources,
moved evidence/gallery persistence off callbacks, prewarmed StableID CUDA work,
and explicitly sized streammux/tiler pools.

A full three-room replay using the 82.7-second Family Room motion clip sustained
30.10 encoded access units per second over 90.1 seconds. Encoded-AU p99 was
80 ms, the maximum gap was 144 ms, no gap exceeded 150 or 250 ms, all sources
remained healthy at approximately 30 FPS, the H.264 feeder dropped zero frames,
and WebRTC decoded 908 frames in 30 seconds.

The finalized live baseline then measured 29.8-30.0 FPS for every camera with
zero source recovery attempts. A direct 30-second H.264 sample delivered 912
access units at 30.36 FPS; p99 was 57.5 ms, the maximum gap was 65.9 ms, and no
gap exceeded 100 ms. A direct WebRTC client decoded 305 frames in 10 seconds.

These are practical regression references on the accepted host, not universal
hardware guarantees. Compare source progress, encoded cadence/drops, and
WebRTC decode separately according to
[`performance_invariants.md`](performance_invariants.md); the dashboard's
aggregate receiver FPS is not a substitute for those measurements.

## Accepted visual behavior

- The oai2-fe dashboard receives live mosaic, tracking, world, depth, and BEV
  data through Menon.
- Canonical PCF BEV frames admit the associated committed tracking cohort, so
  people appear as dots/trails on PCF-backed floorplans.
- A revision-matched candidate well beyond the authored boundary or measured
  prior extent is retained for diagnostics but quarantined as weak; agreement
  between two equally contradictory sources cannot turn it into an accepted
  fused position.
- Empty occupancy is valid and does not require synthetic tracks.
