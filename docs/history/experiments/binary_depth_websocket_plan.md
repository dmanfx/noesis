<!-- 2fe42dbd-6f7e-4f44-b7b8-7a2137b6336a b7eb0e10-4beb-4457-a940-3fad8db533d7 -->
# Binary Depth (1 Hz) over WebSocket

## Decisions

- Binary only (no JSON/base64 fallback, no format toggles)
- Transport over existing WebSocket at 1 Hz (separate binary frame, not embedded in stats JSON)
- No gzip; reduce size via float16 (depth, conf) and uint8 (mask)
- Simple header; derive buffer lengths from H×W×dtype; include magic and version

## Wire Format

- Media: WebSocket binary frame
- Header (little-endian):
- magic: 1 byte 0xFD (depth-binary marker)
- version: uint8 (=1)
- cam_id_len: uint8
- cam_id: bytes[cam_id_len]
- ts_ms: float64
- height: uint32
- width: uint32
- depth_dtype: uint8 (0=float16, 1=float32)
- conf_dtype: uint8 (0=float16, 1=float32)
- mask_dtype: uint8 (0=uint8)
- Payload: depth bytes, then conf bytes, then mask bytes
- depth bytes size = H×W×(2 or 4)
- conf bytes size = H×W×(2 or 4)
- mask bytes size = H×W×1

## Backend Changes

- `websocket_server.py`
- Add a periodic 1 Hz task (co-scheduled with stats) that broadcasts latest depth per camera as a binary frame using the above header.
- Reuse `broadcast_sync` to push the binary bytes.
- `main.py`
- Source the latest depth per camera from `ApplicationManager._latest_depth_results`.
- Convert depth/conf to contiguous arrays; downcast to float16 for wire; keep mask as uint8.
- Compose header with struct.pack (e.g., `"<BBB{cam_len}s d I I B B B"`).
- Align the broadcast tick with stats to approximate "alongside telemetry".

## Frontend Changes

- `oai2-fe/src/hooks/useWebSocketClient.ts`
- In `onmessage`, if `ev.data` is `ArrayBuffer`, read first byte; if 0xFD, parse as depth-binary per header; otherwise handle existing JPEG path.
- Add a new callback `onDepthBinary` (plumbed through to App) or directly update depth store.
- `oai2-fe/src/App.tsx`
- Handle parsed depth frames: store `{ ts, shape, depth: Float32Array, conf: Float32Array, mask: Uint8Array }` in `maDepthData[camId]`.
- `oai2-fe/src/components/DepthDrawer.tsx`
- Update to accept typed arrays directly; skip base64 decode when typed arrays are present.
- `oai2-fe/src/lib/float16.ts`
- Implement a tiny `halfToFloatArray(u16: Uint16Array): Float32Array` helper for expanding float16 to float32 (JS lacks native Float16Array).

## Notes

- Only broadcast cameras with fresh depth within last ~5s to avoid stale spam.
- Maintain backward compatibility for JPEG frames (no framing change).
- No JSON base64 path; depth drawer's manual refresh can be kept as a fallback UI action but will use binary data once available.

## Validation

- Unit-test float16 decode helper.
- Log one-line metrics per tick: cams sent, bytes per cam, encode time.
- Verify drawer renders new frames at ~1 Hz with lowered CPU.

### To-dos

- [ ] Add 1 Hz depth binary broadcast in `websocket_server.py`
- [ ] Compose header+payload with float16 depth/conf in `main.py`
- [ ] Parse depth binary frames in `oai2-fe/src/hooks/useWebSocketClient.ts`
- [ ] Add `halfToFloatArray` in `oai2-fe/src/lib/float16.ts`
- [ ] Update `DepthDrawer.tsx` to consume typed arrays directly
- [ ] Align depth tick with stats tick; gate stale cameras (<5s)
