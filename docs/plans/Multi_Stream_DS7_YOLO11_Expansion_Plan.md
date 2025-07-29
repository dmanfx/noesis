# Multi-Stream Expansion Plan for DeepStream 7 + YOLO11

## Goal
Enable processing of three or more concurrent video streams with the existing DeepStream 7 / YOLO11 pipeline. Each input stream should undergo identical preprocessing, inference and post-processing. The output must remain a set of independent streams delivered over WebSocket to the React/Electron frontend where each stream will map to its own player element.

## Current State Summary
- `DeepStreamVideoPipeline` is built around a **single** RTSP source. `nvstreammux` is configured with `batch-size=1` and only one source bin (`sink_0`).
- `ApplicationManager` launches one `DeepStreamProcessorWrapper` per camera. The frontend currently assumes exactly two cameras and statically renders `<img id="kitchen-stream">` and `<img id="living-room-stream">`.
- `config_preproc.txt` and `config_infer_primary_yolo11.txt` are tuned for batch size 1.

The system technically supports multiple pipelines running in parallel, but there is no unified multi-input pipeline and the frontend is hard‑coded for two streams.

## Required Changes
### 1. Configuration Updates
- **`AppConfig.cameras.RTSP_STREAMS`** already accepts a list of dictionaries. Populate this list with all desired streams (`enabled: true` for active ones).
- **Batch sizes**:
  - `config.processing.DEEPSTREAM_MUX_BATCH_SIZE` should equal the number of enabled streams if a single pipeline handles all of them.
  - Update `batch-size` in `config_infer_primary_yolo11.txt` to match the mux batch size.
- **Preprocess configuration**:
  - In `config_preproc.txt`, set `src-ids` in `[group-0]` to include all stream IDs (e.g. `src-ids=0;1;2;3`).
  - Add additional `[group-x]` sections if different preprocessing per camera is needed (currently one group is sufficient).

### 2. DeepStream Pipeline Refactor
Modify `DeepStreamVideoPipeline` so a single instance can manage multiple sources:
1. Accept a `sources: List[Dict]` argument rather than a single `rtsp_url`.
2. During `_create_pipeline`:
   - Set `streammux` `batch-size` to `len(self.sources)`.
   - Iterate over `self.sources`, create a source bin for each with `_create_source_bin(index, config)` and link each to `streammux` pad `sink_i`.
3. After inference/tracking/analytics, split the batched stream back into per-source streams using **`nvstreamdemux`**. For each demux pad:
   - Attach a branch consisting of `nvvideoconvert → nvdsosd → nvvideoconvert → capsfilter → nvjpegenc → queue → appsink`.
   - Tag the encoded frame with the corresponding `source_id` so `read_encoded_jpeg()` can push `(camera_id, jpeg_bytes)` to the queue.
4. Update `read_encoded_jpeg()` and WebSocket broadcasting so each encoded JPEG is prefixed with the camera identifier (current binary format already supports this).
5. Ensure metadata probes (for detections/analytics) use the `source_id` from `NvDsFrameMeta` to associate detections with the correct stream.

### 3. Application Layer
- Adapt `create_deepstream_video_processor()` and `DeepStreamProcessorWrapper` to pass a **list of sources** to `DeepStreamVideoPipeline` instead of creating one pipeline per camera.
- Adjust `ApplicationManager._start_frame_processors()` logic so it launches a single multi-source processor when using the unified pipeline (or keep the per-camera model if desired but ensure config batch sizes match). The processor should expose per-camera stats.

### 4. Frontend (React/Electron)
- Replace the two hard-coded `<img>` tags with a dynamic list driven by the cameras reported over WebSocket.
- When the client receives a binary frame, use the camera ID prefix to create or update the corresponding `<img>` element. Unknown camera IDs should result in creating new DOM nodes (video containers with fullscreen buttons and FPS counters).
- Update FPS tracking logic to store stats per camera ID instead of `living-room` / `kitchen` keys.

### 5. Testing and Validation
1. Incrementally add streams in `config.cameras.RTSP_STREAMS` and verify the pipeline starts with appropriate `batch-size`.
2. Confirm each stream's detections appear independently in the frontend and that FPS per stream meets expectations (>30 FPS target for 4 streams).
3. Monitor GPU memory usage; adjust TensorRT batch size and memory pools if needed.
4. Validate that disabling/enabling individual streams via config does not require code changes.

## References
- NVIDIA DeepStream documentation: [Multi-source pipelines and nvstreamdemux](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreamdemux.html)
- Example implementations from [marcoslucianops/DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) which demonstrate multi-source configuration and per-stream output handling.

Implementing these changes will allow the application to ingest an arbitrary number of RTSP/video inputs, process them with the same YOLO11 inference pipeline, and stream each result to its own player in the frontend without merging them into a mosaic.
