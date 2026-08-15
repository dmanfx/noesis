# Native DeepStream 9.1 runtime baseline

Status: accepted application baseline, updated 2026-08-15.

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
- Manual/full-frame depth: MapAnything FP32, request-gated, GIE 2.
- Analytics: native ROI exclusion followed by `nvdsanalytics`.
- Output: one 3840×720 H.264 mosaic through SHM/WebRTC.
- RTSP: disabled.
- BEV: JSON/metadata only in `camera_local_ground_m`; PCF is the canonical
  floorplan presentation source where a camera has an admitted Scene Prior.

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

After restoring dependency/environment parity and removing identity-retention
work from the per-frame hot path, the live three-camera app returned to roughly
29.9 FPS per camera.

A bounded 90.67-second recorded pressure run using the non-July sample MP4s
measured 25.45 FPS per camera (76.36 aggregate), average CPU 128.2%, average GPU
53.1%, and VRAM p95 6380 MiB, with no pipeline errors or zero-copy violations.
This is a practical regression reference, not a hardware benchmark guarantee.

## Accepted visual behavior

- The oai2-fe dashboard receives live mosaic, tracking, world, depth, and BEV
  data through Menon.
- Canonical PCF BEV frames admit the associated committed tracking cohort, so
  people appear as dots/trails on PCF-backed floorplans.
- Empty occupancy is valid and does not require synthetic tracks.
