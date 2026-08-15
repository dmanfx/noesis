## WebRTC Output Rollout Plan (DeepStream/GStreamer-first)

This plan replaces JPEG-over-WebSocket with a WebRTC video path using DeepStream/GStreamer built-ins. It preserves the existing analytics, telemetry, and RPCs without CPU fallbacks.

---

### 1) Goals and Hard Constraints
- Reduce egress bandwidth to ≤5 Mbps per 1080p camera while keeping analytics intact.
- Zero CPU video path: decode, OSD, colorspace, and encode remain on GPU; no software encode/decodes.
- Maintain existing websocket channel for stats/RPCs and use it strictly for WebRTC signaling (per‑peer), not media.
- Support 3 cameras and ≥10 concurrent viewers with single-encode-per-camera fan-out (no per-viewer encodes).
- Keep JPEG broadcaster behind a feature flag for quick rollback; do not remove MDE frame supply.

**COMMENTS**
- Zero-CPU goal conflicts with later “CPU appsink tap for MDE.” Either commit to GPU-only MDE supply or explicitly mark a temporary exception with measured CPU budget, acceptance limits, and a removal plan.
- Add measurable SLAs now (latency, time-to-first-frame, stall/reconnect rates, encoder utilization, browser coverage) and explicit rollback triggers tied to those metrics, per user preferences in docs/reference/memories.md.

---

### 2) Current Pipeline Snapshot (Relevant Bits)
- RTSP ingest → DeepStream decode/infer/track/OSD on GPU.
- Post-OSD frames are JPEG-encoded and broadcast via websocket.
- MDE mono depth currently decodes these JPEGs for inference.

Impact: Removing JPEG requires a new GPU-accessible frame tap for MDE to avoid CPU copies.

**COMMENTS**
- Reflect that the current codebase uses DeepStream-native sources (`nvmultiurisrcbin`/`nvurisrcbin`) rather than raw `rtspsrc`; keep the snapshot consistent with present architecture to avoid regressions.
- Call out where the GPU-accessible tap will attach and how it replaces the existing JPEG dependency in the current MDE path.

---

### 3) Target Architecture (DS/GStreamer-first)
- Media Flow per camera (GPU only):
  src (rtspsrc) → depay → parse → `nvv4l2decoder` → `nvstreammux` → `nvinfer` (PGIE) → `nvtracker` → `nvdsanalytics` → `nvvideoconvert` (GPU) → `nvdsosd` (GPU) → `nvvideoconvert` (NV12) → `nvv4l2h264enc` (CBR/VBR-LowDelay) → `h264parse` → `rtph264pay` → `webrtcbin`
- Signaling: reuse existing `WebSocketServer` for SDP/ICE with per-connection messages; media never traverses WS.
- Fan-out: single encode per camera; `webrtcbin` publishes one RTP stream; the browser SFU layer handles per-peer duplication internally. No additional SFU service is introduced at this stage.
- Internal frame tap: before H.264 encode, expose a GPU NV12 frame path to the MDE mono component via a GPU-only tee. No CPU fallbacks.

**COMMENTS**
- “Browser SFU layer” is incorrect. Browsers do not fan-out upstream media. Use one encoder per camera feeding multiple per-peer `webrtcbin` instances server-side, or introduce an SFU. Document egress scaling and NIC headroom for N viewers.
- Clarify GLib main loop vs asyncio: `webrtcbin` runs on a GLib loop. Define threading and cross-posting (e.g., `GLib.idle_add`) rather than assuming a shared asyncio loop.
- The current WebSocket server mixes JSON control and binary JPEG. Either dedicate a signaling endpoint/path or strictly namespace signaling to avoid collisions with telemetry/binary traffic during A/B.
- Prefer DeepStream-native sources (e.g., `nvmultiurisrcbin`) in the media flow to match the existing DS pipeline.

---

### 4) Built-ins Mapping (DeepStream plugins)
- `nvstreammux`: batch and synchronize multi-cam frames; `live-source=true`, `batch-size=NUM_CAMERAS`.
- `nvinfer`: primary detector (TensorRT); `unique-id=1`, `batch-size` matches mux.
- `nvtracker`: NvDCF or configured tracker; maintains object IDs.
- `nvdsanalytics`: high-level analytics zones; configs unchanged.
- `nvdsosd`: GPU OSD; `process-mode=0`, boxes/text as today.
- `nvvideoconvert`: GPU colorspace to NV12 for encoder.
- `nvv4l2h264enc`: NVENC; configure low-latency, profile, bitrate, IDR interval.
- `h264parse` + `rtph264pay`: packetize for RTP.
- `webrtcbin`: DTLS/SRTP, ICE, RTP/RTCP; integrate with WS signaling.

Reference: docs/deepstream-docs/02_GStreamer_Plugins.md and 03_Sample_Applications.md (see `deepstream-test1/3/4` patterns and `rtsp-in-rtsp-out`).

**COMMENTS**
- Add explicit preference for DS-native source bins (`nvmultiurisrcbin`/`nvurisrcbin`) and queue configurations compatible with NVMM (GPU memory).
- Include recommended queue leaky settings on GPU paths to bound latency (e.g., `queue max-size-buffers=1 leaky=2` before encoder/payloader).

---

### 5) Codec and Transport Parameters (H.264)
- Profile: Constrained Baseline or Main as negotiated; disable B-frames for low-latency.
- Rate control: Start CBR 4–5 Mbps @1080p30 per camera; VBV tuned for ~150 ms; `iframeinterval=30` (IDR every 1s).
- Packetization: `rtph264pay config-interval=1` (periodic SPS/PPS), MTU ≤1200 bytes, `webrtcbin` `bundle-policy=max-bundle`.
- Key caps: `video/x-h264,stream-format=byte-stream,alignment=au,profile=constrained-baseline`.

Browser notes:
- Safari: H.264 only; ensure `packetization-mode=1` and no B‑frames.
- Chrome/Firefox/Edge: H.264 works; VP9/AV1 omitted for now to keep NVENC only.

**COMMENTS**
- Specify `packetization-mode=1` and `profile-level-id` constraints in SDP/caps explicitly; ensure periodic SPS/PPS in-band and alignment with payloader `pt`.
- Define PLI/FIR → force-IDR mechanics (map RTCP events to `nvv4l2h264enc` keyframe requests) and confirm no B-frames with zerolatency.
- No ABR/simulcast/SVC is planned; document congestion handling (per-client policing, drop oldest frames) and caps to prevent latency growth.

---

### 6) Signaling Design (within current `WebSocketServer`)
- Per-connection state with a `peer_id`; route messages by `peer_id` not broadcast.
- Message types (JSON): `webrtc-offer`, `webrtc-answer`, `ice-candidate`, `webrtc-close`.
- Server maintains `peer_id -> webrtcbin` mapping; one receive-only transceiver per subscribed camera.
- The WS server and `webrtcbin` share the same asyncio loop; DS thread interactions use thread-safe dispatch.

Signaling servers:
- STUN: not required for now (LAN/local testing only)
- TURN: not configured for now; add later when WAN access is needed

**COMMENTS**
- Do not defer STUN/TURN. Provide concrete STUN servers now and define TURN rollout criteria, metrics (ICE time, relay ratio), and credential rotation. Remove placeholders per “No placeholder code” preference.
- Define auth tokens for signaling, per-camera authorization, rate limits, and session IDs to support reconnect/ICE restarts. Update docs/reference/WebSocket_API.md with the signaling schema.
- Avoid mixing signaling with binary frame traffic; separate endpoint/path or strict namespacing is needed.
- Correct loop model: manage a GLib main loop for `webrtcbin`; interact from asyncio via thread-safe dispatch.

---

### 7) Runnable Pipelines (per camera; conceptual gst-launch)

Single camera to WebRTC (abbrev; signaling wired in code):
```
gst-launch-1.0 -e \
  rtspsrc location=rtsp://CAM ! rtph264depay ! h264parse ! nvv4l2decoder ! \
  nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! \
  nvstreammux name=mux batch-size=1 live-source=true width=1920 height=1080 ! \
  nvinfer config-file-path=pipelines/dstest1_pgie_config.txt unique-id=1 ! \
  nvtracker ll-lib-file=... ll-config-file=... ! \
  nvdsanalytics config-file=config_nvdsanalytics.ini ! \
  nvdsosd process-mode=0 ! nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! \
  nvv4l2h264enc iframeinterval=30 insert-sps-pps=true preset-level=1 zerolatency=true maxperf-enable=1 \
                 control-rate=1 bitrate=5000000 ! h264parse ! rtph264pay config-interval=1 pt=96 ! \
  webrtcbin name=wb bundle-policy=max-bundle
```

Note: In code we build a multi-source graph with one encode branch per camera feeding a dedicated `webrtcbin` pad; muxing for inference still uses `nvstreammux` upstream.

**COMMENTS**
- Treat this gst-launch as illustrative only; align implementation with DS-native sources and multi-source graph used in code.
- Include caps enforcing `packetization-mode=1` and set explicit MTU to avoid fragmentation issues.
- Add explicit leaky queue settings ahead of encoder/payloader to bound latency under backpressure.

---

### 8) MDE Frame Supply (no JPEG dependency)
- Add a tee before `nvv4l2h264enc` to expose frames:
  - Primary: GPU NV12 path (reserved; pending future MDE GPU ingest support)
  - Practical now: low-rate CPU appsink tap (≤5 fps) for MDE only.
- Remove MDE’s dependency on JPEG queues; wire it to the new tap.
- CPU cost estimate (per camera @1080p, 5 fps): ~5–12% of one core for NV12→BGR + resize + base64.
- With 3 cameras: ~15–36% of a core total, distributed across cores.
- Integration touchpoint: MDE remains CPU RGB via `/infer_mono` and `/infer_multi`.

**COMMENTS**
- This contradicts the “Zero CPU” goal. Prefer a GPU nvjpegenc branch (still GPU encode) or define a CUDA/NVMM ingest for the adapter; quantify any unavoidable CPU copy cost and cap it. Remove the placeholder by specifying the concrete MDE GPU entrypoint and owner.
- Plan removal of base64 for future GPU-native ingestion to reduce CPU overhead and latency.

---

### 9) Frontend Wiring
- New hook `useWebRTCStreams` (oai2-fe): per camera, create `RTCPeerConnection`, add transceiver `recvonly`, perform WS signaling, attach to `<video>`.
- Autoplay/muted handling; display per-peer stats (RTT, loss, bitrate) from `getStats()`.
- Feature flag: `?transport=webrtc` or config toggle; on failure, operator can flip back to JPEG without restarting the server.

**COMMENTS**
- Define the exact hook interface and lifecycle: autoplay with `muted`, attach `MediaStream` via `srcObject`, cleanup on camera switch/unmount, and handle background tab throttling.
- Replace the ad‑hoc JPEG id framing (currently used by the WebSocket client) with track labels/metadata; define mapping from camera IDs to tracks.
- Specify A/B within `StreamPanel` and automatic fallback to JPEG when negotiation fails.
- Extend WebSocket_API with signaling message types to keep client/server contracts in sync.

---

### 10) Multi-Viewer Strategy
- Single encode per camera; `webrtcbin` duplicates packets to peers.
- Guardrails: `MAX_WEBRTC_VIEWERS=10` soft cap; emit WARN when exceeded; reject new viewers beyond cap.
- Optional future: add an external SFU only if single-encode fan-out proves insufficient (out of scope here).

**COMMENTS**
- Clarify that duplication occurs server-side via multiple per-peer `RTCPeerConnection`s fed by a single encoder, not in the browser. Provide egress budgeting and NIC/OS tuning targets; document the rejection behavior when caps are hit.
- Add criteria to introduce an SFU if viewer counts or WAN usage outgrow single-host fan-out.

---

### 11) Bench and Acceptance
Targets:
- Per camera: ≥30 fps end-to-end, added encoder latency ≤250 ms; bitrate ≤5 Mbps @1080p.
- 3 cameras, 5–10 viewers: stable for >1 hour; no dropped frames attributable to encoder starvation.

Quick bench (record during runs):
- Report: fps, end-to-end latency_ms, gpu.util; note device, res, batch.
- Verdict: PASS/FAIL vs target.

**COMMENTS**
- Add WebRTC-specific metrics: ICE gather/connect times, TURN relay ratio, RTT, inbound/outbound bitrate, frames dropped, encoder QP, PLI/FIR counts, and time-to-first-frame.
- Include egress throughput and relay ratio thresholds; wire alerts to rollout rollback criteria.

---

### 12) Rollout Phases (incremental, minimal changes)
1) Enable DS WebRTC for one camera, keep JPEG for others (feature flag). Wire MDE to new frame tap. Prove parity.
2) Extend to all cameras; cap viewers; add telemetry and reconnection.
3) Hardening: browser matrix (Chrome/Firefox/Edge/Safari), soak tests, runbook. TURN/STUN to be added later if needed.
4) Flip default to WebRTC; retain JPEG as fallback toggle.

**COMMENTS**
- Add a kill switch and explicit rollback triggers (latency, ICE failure rate, relay ratio, error budgets). Introduce STUN immediately and TURN in Phase 2/3 with acceptance thresholds.
- Define per-browser/device compatibility targets and fallback behavior to JPEG/HLS where needed.

---

### 13) Operational Notes / Runbook
- Restart sequence: close peer connections → set `webrtcbin` to NULL → recreate pads → resume.
- Reconnection: support ICE restart on WS signal `webrtc-restart`.
- Alarms: NVENC session errors, bitrate collapse, TURN-only connectivity.

**COMMENTS**
- Document GLib/asyncio ownership and restart sequencing explicitly; include how to force keyframes (PLI/FIR) and drain queues safely.
- Add OS/network tuning (e.g., UDP buffer sizes, NIC offloads) and telemetry dashboards referenced by alarms.

---

### 14) Open Items / Placeholders (answer then search/replace)
- [MDE_GPU_ENTRYPOINT] = exact function/class to ingest GPU NV12 for MDE

**COMMENTS**
- Replace this with a concrete function/class name and owner/date. Per preferences, avoid placeholders by committing to either a GPU nvjpegenc branch or a CUDA/NVMM ingest path.

---

### 15) Checklist (DeepStream docs-first compliance)
- Built-ins mapped (by name), DS-first justification complete.
- Runnable pipeline provided; knobs listed and justified.
- Bench contract included; acceptance criteria explicit.
- No CPU fallback; JPEG path retained only as feature-flag fallback.

**COMMENTS**
- Add checks for “Signaling API updated,” “Security/auth documented (WSS, tokens, origin checks),” and “Egress budget/NIC tuning documented.”
