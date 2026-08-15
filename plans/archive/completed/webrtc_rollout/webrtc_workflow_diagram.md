## WebRTC Workflow (DeepStream → GStreamer → Browser)

The diagram traces one camera from RTSP ingest to the browser, including signaling and the internal MDE tap. Replicate the media branch per camera.

```
Legend:
  [GPU]  = GPU-resident buffer (NVMM)
  [CPU]  = CPU object/message
  (WS)   = WebSocket signaling only (no media)

RTSP Camera                                Server (DeepStream/GStreamer)                                   Browser
------------                                --------------------------------                                   -----------------
rtsp://cam ───────────────┐                 (Threaded DS app, asyncio WS loop)                               (Main thread)
                           │
                           ▼
                     rtspsrc (udp, latency=X) [CPU control, RTP packets]
                           │
                           ▼
                   rtph264depay → h264parse [CPU control]
                           │
                           ▼
                    nvv4l2decoder (NVDEC) ─────────────────────────────────────────────────────────────────────────▶ [GPU]
                           │
                           ▼
   (per-source pad) ──────▶ nvstreammux (batch-size=N, live-source=true) [GPU NVMM]
                           │
                           ▼
                         nvinfer (PGIE) [GPU] ──► nvtracker [GPU] ──► nvdsanalytics [GPU]
                           │
                           ▼
                        nvdsosd process-mode=0 [GPU]
                           │
                           ▼
                 nvvideoconvert → NV12 caps [GPU]
                           │
                           ├─────────┐
                           │         │
                           │         ▼
                           │   (Tee for MDE)
                           │       │
                           │       ├── GPU tap (reserved) [GPU]
                           │       └── appsink ≤5 fps → NV12→BGR→MDE (CPU) [CPU]
                           │
                           ▼
                   nvv4l2h264enc (NVENC; CBR ~5Mbps, iframeinterval=30, no B-frames) [GPU]
                           │
                           ▼
                        h264parse → rtph264pay config-interval=1 pt=96 [CPU control, GPU-backed payload]
                           │
                           ▼
                        webrtcbin (DTLS/SRTP, ICE, RTP/RTCP) [CPU control + GPU-backed RTP path]
                           │                     ▲
                           │                     │ (WS)
                           │                     │
                           └───────────── WebSocketServer (per-peer signaling, SDP/ICE) ───────────────┐
                                                                                                        │
                                                                                                        ▼
                                                                                              RTCPeerConnection
                                                                                                 (recvonly)
                                                                                                        │
                                                                                                        ▼
                                    <video>
```

**COMMENTS**
- Media duplication must occur server-side (multiple per-peer `RTCPeerConnection`s) from a single camera encoder; browsers do not provide SFU fan-out. Document egress scaling and NIC/OS tuning for multi-viewer scenarios.
- Clarify loop ownership: `webrtcbin` runs on a GLib main loop; integrate with the asyncio-based WebSocket signaling via thread-safe cross-posts (`GLib.idle_add`) rather than assuming a shared loop.
- Add explicit leaky queue positions (before encoder/payloader) to bound latency under backpressure.

Signaling states (per peer):
- Client → WS: `webrtc-offer` (SDP, [PLACEHOLDER_STUN_LIST], [PLACEHOLDER_TURN])
- Server → `webrtcbin`: set-remote/prepare-answer; WS → Client: `webrtc-answer`
- Bidirectional ICE candidates via WS: `ice-candidate`
- On reconnect: WS `webrtc-restart` → ICE restart on `webrtcbin`

**COMMENTS**
- Replace placeholders with concrete STUN servers now; define TURN credentials (ephemeral), rotation schedule, and rollout criteria. Update the WebSocket API reference with the signaling schema.
- Add per-connection auth (short‑lived tokens), per-camera authorization, connection IDs, rate limits/backoff, and session resumption behavior.

Backpressure and health:
- `webrtcbin` RTCP receiver reports NACK/PLI; trigger keyframe request on encoder if needed.
- Drop policy: upstream queues bounded; prefer dropping oldest at tee before encoder rather than accumulating latency.

**COMMENTS**
- Specify element properties to enforce the policy (e.g., `queue max-size-buffers=1 leaky=2` ahead of encoder/payloader) and map PLI/FIR to force-IDR on the encoder.
- Capture RTCP stats (PLI/FIR counts, jitter, packet loss) and expose them in telemetry for alerting.

Limits:
- Max concurrent viewers per camera: 10
- Strictly single NVENC session per camera; no per-viewer encodes.

**COMMENTS**
- Document enforcement mechanics (reject policy with clear error) and rationale (encoder/NIC limits). Provide an egress budget and monitoring thresholds.

Security:
- DTLS certificate auto-generated or provisioned; WSS required in non-localhost. TURN credentials must be configured.

**COMMENTS**
- Require WSS with origin checks, short‑lived signaling tokens, and redaction of SDP/candidate details in logs. Define TURN credential rotation and storage.

```
