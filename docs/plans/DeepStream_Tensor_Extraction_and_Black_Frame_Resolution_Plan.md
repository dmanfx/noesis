# DeepStream Tensor Extraction and Black Frame Resolution Plan

## Overview
This plan addresses the black frames issue in the front-end and the failure to extract tensors (`tensor=❌`) from the DeepStream pipeline. The root causes include missing tensor metadata from `nvdspreprocess` and improper access to GPU (NVMM) surfaces without CPU copies. All solutions maintain strict GPU-only processing as per requirements.

Key constraints:
- No CPU involvement in frame/tensor handling.
- Prioritize DeepStream native features.
- Focus on zero-copy GPU tensor access in Python for flexibility.

## Phase 1: Verify Tensor Metadata Presence
**Rationale**: Black frames and `tensor=❌` often stem from `nvdspreprocess` not attaching tensor metadata. We'll add probes and enhanced logging to confirm if meta is generated.

**Steps**:
1. Add a pad probe on the `nvdspreprocess` sink pad to log buffer metadata lists (e.g., count of user metas per buffer).
2. In `_on_new_sample`, implement enhanced metadata search:
   - Scan both batch-level (`batch_user_meta_list`) and frame-level (`frame_user_meta_list`) for the tensor meta type (`NVIDIA.NVDSPREPROCESS.TENSOR`).
   - Log details: meta count, types, and if tensor meta is found.
3. Run the pipeline for ~30 seconds and review logs for "Found tensor meta" messages.

**Expected Outcomes**:
- If meta is missing: Proceed to Phase 2 (config fixes).
- If meta is present but extraction fails: Jump to Phase 3 (zero-copy access).

## Phase 2: Fix and Align `nvdspreprocess` Configuration
**Rationale**: Mismatched config params between `nvdspreprocess` and `nvinfer` can suppress tensor meta generation.

**Steps**:
1. Validate key params in `config_preproc.txt` against `config_infer_primary_yolo11.txt`:
   - `network-input-shape` (e.g., 1;3;640;640).
   - `processing-width`/`height` (match model input).
   - `network-input-order` (0=NCHW, 1=NHWC).
   - `tensor-name` (match model's input tensor name, e.g., "input_1:0").
2. Confirm `tensor-data-type`:
   - Mapping: 0=FP32, 1=UINT8, 2=INT8, 3=UINT32, 4=INT32, 5=FP16.
   - Research: Check YOLOv11 model's ONNX input dtype (likely FP32=0; you've set to 0, but confirm if FP16=5 is needed for optimization).
   - If mismatch, update to match model (e.g., FP32=0).
3. Ensure `output-tensor-meta=1` and `tensor-buf-pool-size` >= batch size.
4. Restart pipeline and re-run Phase 1 verification.

**Expected Outcomes**:
- Tensor meta should now appear in logs. If not, deeper issue (e.g., plugin version mismatch) – investigate logs for errors.

## Phase 3A: Zero-Copy GPU Tensor Access in Python
**Rationale**: To access full NVMM surfaces without CPU copies, use DeepStream Python bindings for direct CUDA pointer mapping to PyTorch tensors.

**Steps**:
1. In `_on_new_sample` (after meta verification):
   - Use `pyds.get_nvds_buf_surface_gpu(hash(gst_buffer), batch_id)` to get dtype, shape, strides, CUDA ptr, and size.
   - Create a CuPy array from the CUDA ptr (zero-copy) via `cp.ndarray` with UnownedMemory/MemoryPointer.
   - Convert to PyTorch tensor via `torch.from_dlpack(cupy_array.toDlpack())` (zero-copy GPU transfer).
2. Handle strides/padding: If tensor has padding (common in DeepStream), use `tensor.as_strided` or contiguous() if needed (still GPU-only).
3. Update `frame_dict` to include this GPU tensor.
4. For front-end: Send tensor metadata or use NVENC to stream RTSP previews (GPU-only).

**Expected Outcomes**:
- Logs show full tensor shapes (e.g., [3, 1080, 1920]) and no more `tensor=❌`.
- Front-end receives drawable frames (via metadata or RTSP).

## Phase 4: Integrate with WebSocket/Front-End (GPU JPEG Version)
**Rationale**: With tensors accessible, ensure metadata flows to the UI. Use GPU-accelerated JPEG encoding to generate images without CPU copies, maintaining the current base64 WebSocket flow for compatibility.

**Steps**:
1. In `_on_new_sample`, after getting zero-copy GPU tensor (from Phase 3A), add a GStreamer branch using `nvjpegenc` (GPU-only JPEG encoder) to compress the tensor to JPEG bytes.
2. Base64-encode the JPEG bytes (on CPU – minimal overhead) and include in JSON with detections/analytics.
3. Broadcast via WebSocket (keep existing format).
4. Confirm front-end renders non-black frames.

**Expected Outcomes**:
- GPU-only image generation; no black frames; compatible with current Electron app.

## Phase 5: Refactor to RTSP with Client-Side Drawing
**Rationale**: Optimize for efficiency by switching to RTSP streaming with metadata overlays drawn on the client-side.

**Steps**:
1. In the pipeline, add a tee after `nvdsosd` (or custom viz/tracking toggle point) to an NVENC encoder → RTP payloader → RTSP sink (e.g., via `gst-rtsp-server` or `rtspclientsink`).
2. Start an RTSP server to serve the stream (e.g., `rtsp://localhost:8554/stream`).
3. Update WebSocket to send only lightweight metadata (JSON with bboxes, timestamps, analytics).
4. Front-End Updates:
   - Replace Canvas with `<video>` element to play RTSP (use `video.js` or Electron's media support for RTSP).
   - On WebSocket message, draw bboxes/overlays on a transparent Canvas overlaid on the video.
   - Handle sync: Match metadata timestamps to video `currentTime` (buffer 1-2 packets for jitter).
5. Integrate toggle: Custom viz/tracking branch feeds the tee when enabled.
6. Test end-to-end: Confirm real-time video with aligned overlays.

**Expected Outcomes**:
- Lower bandwidth/latency; fully GPU-only; scalable to multiple cameras.

## Additional Notes
- **Testing**: After each phase, run for 30s and share logs. Revert changes if issues arise.
- **Research on Dtype**: Confirm YOLOv11 engine dtype (e.g., via TensorRT tools) – if FP16, set to 5; else 0.
- **Fallback**: If zero-copy fails, fall back to metadata-only (draw synthetic frames in UI based on bboxes).
- **Tools Needed**: Ensure PyCUDA and CuPy are installed for Phase 3A. 