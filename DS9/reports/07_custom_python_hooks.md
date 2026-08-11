# 07 Custom Python Hooks, MapAnything, BEV, Overlay, Telemetry

Original audit: 2026-05-10

Reconciled with the current exact-capture path: 2026-07-25

Scope: DS9 migration audit for custom Python hooks while keeping live DS8
untouched. This report audits only the DS9 tree; later reconciliation records
the implemented DS9 path and its isolated validation evidence.

Audited local sources:

- [`../noesis/pipelines/hooks.py`](../noesis/pipelines/hooks.py)
- [`../noesis/pipelines/ds8_pipeline.py`](../noesis/pipelines/ds8_pipeline.py)
- [`../../noesis/telemetry/bev.py`](../../noesis/telemetry/bev.py)
- [`../pipelines/config_infer_secondary_mapanything.ini`](../pipelines/config_infer_secondary_mapanything.ini)
- [`../docs/MapAnything_Depth.md`](../docs/MapAnything_Depth.md)
- [`../docs/history/ds8/DS8_testing_guide.md`](../docs/history/ds8/DS8_testing_guide.md)
- [`../docs/history/ds8/ds8_design_decisions.md`](../docs/history/ds8/ds8_design_decisions.md)

Additional DS9-local evidence cited for runtime wiring/defaults:

- [`../noesis/ds9_runtime_core.py`](../noesis/ds9_runtime_core.py)
- [`../config/infer.yaml`](../config/infer.yaml)

Official NVIDIA DS9 evidence used for DS9 claims:

- [Service Maker Python Pipeline API](https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_service_maker_python_into_to_pipeline_api.html)
- [Service Maker Python Advanced Features: metadata](https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_service_maker_python_advanced_features.html#leveraging-metadata)
- [TensorOutputUserMetadata SDK API](https://docs.nvidia.com/metropolis/deepstream/9.0/sdk-api/classdeepstream_1_1TensorOutputUserMetadata.html)
- [Gst-nvinfer tensor metadata](https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_plugin_gst-nvinfer.html#tensor-metadata)
- [Gst-nvinfer configuration keys](https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_plugin_gst-nvinfer.html#gst-nvinfer-file-configuration-specifications)
- [PyDS NvDsInferTensorMeta](https://docs.nvidia.com/metropolis/deepstream/9.0/python-api/PYTHON_API/NvDsInfer/NvDsInferTensorMeta.html)
- [PyDS methods](https://docs.nvidia.com/metropolis/deepstream/9.0/python-api/PYTHON_API/Methods/methodsdoc.html)

## Executive Call

The canonical MapAnything path is now proven without an appsink or a separate
CPU video branch: keep full-frame `nvinfer` on GPU/NVMM with
`output-tensor-meta`, convert its output to explicit NVMM RGB, attach a Service
Maker `BufferOperator` probe to `mapanything_rgb_caps`, and terminate the
branch at `fakesink`. The probe reads the exact RGB batch surface and preserved
typed tensor metadata from the same post-inference buffer. The Service Maker
`frame_meta.tensor_items` wrapper is not an authority here: live evidence
exposed only sibling DAv2 UID 5, while the DS9-native typed reader found the
required MapAnything UID 2.

An isolated canonical-container run on 2026-07-25 passed this path for
living-room, kitchen, and family-room with same-buffer `1920x1080` `rgb8`
evidence and zero mutation during the cache-only follow-up. This validates the
capture and manual fresh/cache-only contracts, not production promotion or
absolute depth accuracy.

`BatchMetadataOperator`, Service Maker display metadata, and the trail overlay
path remain canonical. The DS9 code already puts overlays into display metadata
upstream of `nvdsosd`; it does not need an appsink branch to draw trails or
labels.

The custom Python audited here is mostly edge metadata, postprocess, telemetry,
and serialization. It does not introduce a CPU video branch in the canonical
pipeline. Raw `pyds` tensor, object-list, analytics-user-meta, and OSD-label
compatibility helpers remain quarantined from the canonical MapAnything and
Service Maker paths.

## MapAnything Tensor Path

### Local evidence

The DS9 builder creates a dedicated MapAnything branch behind a queue and valve,
then terminates it at a sink for metadata consumption rather than frame output.
It defaults the MapAnything SGIE to tensor metadata output through the builder
property map:

- `attach_tensor_meta` maps to `output-tensor-meta` in
  [`ds8_pipeline.py`](../noesis/pipelines/ds8_pipeline.py).
- The MapAnything branch sets `attach_tensor_meta=True`, `gie_id=2`, uses a
  leaky queue and normally closed gate valve, then links
  `nvinfer -> nvvideoconvert -> NVMM RGB caps -> fakesink` in
  [`ds8_pipeline.py`](../noesis/pipelines/ds8_pipeline.py).
- The MapAnything INI is full-frame SGIE input, GPU selected, tensor-from-meta
  disabled, and `output-tensor-meta=1` in
  [`config_infer_secondary_mapanything.ini`](../pipelines/config_infer_secondary_mapanything.ini).
- The design history explicitly records this as an SGIE branch that terminates
  at `fakesink`, not a tiler/appsink path:
  [`ds8_design_decisions.md`](../docs/history/ds8/ds8_design_decisions.md).
- The runtime only attaches the MapAnything postprocess hook if MapAnything is
  enabled and the component exists in
  `ds9_runtime_core.py`.

The hook side is buffer-and-metadata exact:

- `attach_mapanything_postprocess_hook` attaches a Service Maker `Probe` with
  `_MapAnythingBufferOperator` to `mapanything_rgb_caps` and treats
  native/attach failure as a startup failure.
- `_MapAnythingBufferOperator.handle_buffer` iterates
  `buffer.batch_meta.frame_items` and invokes only
  `MapAnythingProcessor.handle_native_buffer_frame_ds9`; wrapper tensor items
  do not select MapAnything.
- The DS9-owned native bridge traverses typed frame user metadata, requires one
  exact UID 2 record and exact `depth/conf/mask` per-frame shapes, and never
  reapplies `batch_id` because nvinfer already frame-offsets the pointers.
- `Buffer.extract(batch_id)` reads the exact `1920x1080` RGB surface from that
  same post-conversion buffer; source, batch, frame, media PTS, and RGB digest
  remain one capture-event cohort.
- The bridge copies exactly three `294x518` float32 maps (`1,827,504` bytes)
  while nvinfer's metadata owns the source pointers. It releases the Python GIL
  during those synchronous copies and records size and duration.
- Only the owned arrays cross to the bounded worker for alignment, validity
  masking, storage, and RPC publication. Runtime shutdown closes admission,
  drains accepted jobs, and joins the non-daemon worker; final-job poison is
  fatal even if no subsequent frame arrives.

### DS9 call

Keep the Service Maker buffer-probe shape on the post-inference RGB caps
component. It is the DS9 MapAnything target and does not need appsink or a
separate CPU video-frame source.

The native exact reader is the canonical DS9 adapter, not a fallback. The
wrapper, generic native reader, and raw-PyDS alternatives are absent from the
production selection path. The probe-local host copy is an edge
postprocess/RPC cost, not a CPU video branch. It cannot safely move to the
worker without first introducing an owned device-buffer clone contract because
the current source pointers are nvinfer-owned only for the metadata lifetime.
If performance targets require removing it from the callback, the next
canonical move is an owned C++/CUDA tensor clone/postprocessor or custom
metadata producer, not asynchronous dereference of borrowed pointers and not
appsink.

## Overlay, Display Meta, And Trail APIs

### Official DS9 evidence

NVIDIA DS9 Service Maker Python documents pipeline probes/operators and metadata
access through Service Maker. The Advanced Features metadata page documents
`BatchMetadataOperator`, batch/frame/object metadata iteration, display metadata
acquisition, and appending display metadata to frames. The SDK API documents
tensor output user metadata with layer access. Gst-nvinfer documents tensor
metadata production through `output-tensor-meta`.

### Local evidence

The DS9 hook module imports Service Maker metadata and OSD primitives directly:

- `BatchMetadataOperator`, `Probe`, and `pyservicemaker.osd` are imported in
  [`hooks.py`](../noesis/pipelines/hooks.py).
- Trail overlay hook attachment is documented as upstream of `nvdsosd` because
  display metadata must be present before OSD in
  [`hooks.py`](../noesis/pipelines/hooks.py).
- `TrailOverlayProcessor.handle_batch_ds8` requires Service Maker OSD bindings,
  iterates `batch_meta.frame_items`, acquires display metadata, appends it to the
  frame, and uses `ds_osd.Line` / `ds_osd.Text` in
  [`hooks.py`](../noesis/pipelines/hooks.py).
- Pose keypoint overlay follows the same display metadata path with
  `ds_osd.Line` and `ds_osd.Circle` in
  [`hooks.py`](../noesis/pipelines/hooks.py).
- The main pipeline configures `nvdsosd` with GPU processing mode to keep the
  tiler to OSD to encoder path on GPU/NVMM in
  [`ds8_pipeline.py`](../noesis/pipelines/ds8_pipeline.py).
- The design history records trail overlay as Service Maker display metadata
  using `BatchMetadata.acquire_display_meta()`, `FrameMetadata.append()`, and
  `pyservicemaker.osd.Line/Text`, with no CPU appsink branch:
  [`ds8_design_decisions.md`](../docs/history/ds8/ds8_design_decisions.md).

### DS9 call

Keep. `BatchMetadataOperator` plus Service Maker display metadata is still the
canonical overlay route. Rename DS8-specific function/class wording later if
desired, but do not replace this path with pyds pad probes or appsink drawing.

## BEV, Telemetry, And Serialization

### Local evidence

BEV is already an edge telemetry renderer, not a pipeline video branch:

- Runtime config wires BEV publishing and creates `BevRenderer` with configurable
  JPEG behavior in `ds9_runtime_core.py` and
  `ds9_runtime_core.py`.
- The DS9 config has BEV JPEG disabled by default and output overlays disabled:
  [`config/infer.yaml`](../config/infer.yaml).
- `BevRenderer.render_and_publish` only allocates the BGR image when JPEG is
  enabled, otherwise it builds JSON footpoint/trail telemetry in
  [`bev.py`](../../noesis/telemetry/bev.py).
- `_publish` always broadcasts JSON metadata and only encodes JPEG bytes when
  JPEG is enabled in [`bev.py`](../../noesis/telemetry/bev.py).
- The design history records meta-only BEV as the default and frequent BEV JPEG
  as disabled by default:
  [`ds8_design_decisions.md`](../docs/history/ds8/ds8_design_decisions.md).

Telemetry hooks are metadata-first:

- Analytics telemetry attaches a `BatchMetadataOperator` in
  [`hooks.py`](../noesis/pipelines/hooks.py).
- The production Service Maker path iterates `frame_meta.object_items` in
  [`hooks.py`](../noesis/pipelines/hooks.py).
- ReID, pose, depth, and V3DT data extraction are edge metadata/identity inputs,
  not video frame copies. They do account host-copy counters where they cross to
  CPU; see [`hooks.py`](../noesis/pipelines/hooks.py).

### DS9 call

Keep BEV JSON and telemetry serialization as edge metadata. Keep BEV JPEG
disabled by default. If BEV JPEG is enabled, treat it as an explicit diagnostic
or UI output path because it uses CPU image composition and JPEG encoding.

## PyDS Path List In `hooks.py`

These paths should be removed from the DS9 canonical path or quarantined behind
explicit compatibility/debug gates. They are not the desired DS9 production
surface when Service Maker metadata exposes the needed data.

| Area | Current path | Evidence | Production call status | DS9 call |
| --- | --- | --- | --- | --- |
| Optional import | `import pyds` with `PYDS_AVAILABLE` | [`hooks.py`](../noesis/pipelines/hooks.py) | Module-wide optional compatibility | Keep only if a documented DS9 gap remains; otherwise quarantine behind a compatibility module. |
| Raw frame tensor iterator | `_iter_frame_tensor_meta` walks `frame_user_meta_list`, casts `NvDsUserMeta`, checks `NVDSINFER_TENSOR_OUTPUT_META`, casts `NvDsInferTensorMeta` | [`hooks.py`](../noesis/pipelines/hooks.py) | Not used by the exact MapAnything buffer-operator path | Keep quarantined; it is not selected by MapAnything. |
| Raw tensor dtype mapping | `_layer_dtype` imports `NvDsInferDataType` | [`hooks.py`](../noesis/pipelines/hooks.py) | Used by quarantined pyds compatibility code | Keep out of the canonical MapAnything path. |
| Raw tensor buffer copy | `_numpy_from_layer` can call `pyds.get_ptr(layer.buffer)` and copy layer bytes into NumPy | [`hooks.py`](../noesis/pipelines/hooks.py) | Used by quarantined pyds compatibility code | Keep out of production MapAnything selection. |
| Raw layer extraction | `_extract_tensor_layers` can use `pyds.get_nvds_LayerInfo` | [`hooks.py`](../noesis/pipelines/hooks.py) | Used by quarantined pyds compatibility code | Keep isolated from MapAnything. |
| MapAnything raw pyds handler | Removed from `MapAnythingProcessor` | 2026-07-11 source audit | No parallel MapAnything path remains | Keep removed. |
| Generic pyds helpers | `_resolve_pyds_cast`, `_resolve_pyds_attr`, `_iter_meta_entries` | [`hooks.py`](../noesis/pipelines/hooks.py) | Support quarantined compatibility traversal | Keep behind the explicit compatibility gate where still required. |
| Analytics raw user meta | `_extract_analytics_obj_meta` walks object user meta and casts `NvDsAnalyticsObjInfo` | [`hooks.py`](../noesis/pipelines/hooks.py) | Compatibility path; the production Service Maker analytics path uses `nvdsanalytics_obj_items` | Keep quarantined from production selection. |
| Analytics raw object builder | `_build_track_dict` uses non-Service-Maker object fields and pyds analytics extraction | [`hooks.py`](../noesis/pipelines/hooks.py) | Production path uses `_build_track_dict_ds8` | Keep quarantined from production selection. |
| OSD label raw fallback | `_OsdLabelProcessor.handle_frame` uses pyds-style object iteration | [`hooks.py`](../noesis/pipelines/hooks.py) | Production operator calls `handle_frame_ds8` | Keep quarantined; retain the Service Maker object-items path. |

Resolved native ownership:

| Area | Current path | Evidence | DS9 call |
| --- | --- | --- | --- |
| MapAnything native tensor and exact-RGB ownership | `_MapAnythingBufferOperator` uses only `handle_native_buffer_frame_ds9`; the DS9-built bridge requires one raw UID 2 record and exact per-frame `depth/conf/mask` shape. The same post-inference, post-conversion buffer supplies preserved tensor metadata and the explicit NVMM RGB surface through `Buffer.extract(batch_id)`. Python validates batch/source/frame/PTS identity without another offset, bounds and times the tensor and armed-camera RGB copies, then hands owned arrays to one joined bounded worker. | DS9 nvinfer `attach_tensor_output_meta`, installed nvinfer/nvvideoconvert caps and metadata-copy contracts, focused exact-capture/lifecycle tests, and the 2026-07-25 isolated three-camera exact-RGB live gate | Canonical. The normally closed valve remains before inference and conversion. Appsink, source-reader, generic native, raw-PyDS, manual batch-slice, and separately reconciled RGB alternatives are absent; ambiguity, attach failure, queue/worker failure, poison, and unresolved teardown are fatal. |

## GPU-Only And Zero-Copy Assessment

No audited DS9 custom Python path adds an appsink or CPU video branch to the
canonical pipeline. The main video path remains DeepStream GPU/NVMM through
streammux, inference, tracking, tiler, GPU `nvdsosd`, and RTSP/WebRTC output.

The CPU crossings found here are edge metadata or serialization:

- The exact frame-local MapAnything tensors are copied once to owned NumPy
  arrays by the DS9-native selector: exactly `1,827,504` bytes per captured
  frame, with duration and byte counters. The borrowed pointers are
  dereferenced under their metadata-lifetime lease while the GIL is released;
  alignment, masking, storage, and publication then run in a bounded,
  runtime-joined worker. This is an instrumented CPU-edge tensor copy for depth
  snapshots/RPC, not a CPU video-frame branch.
- Trail and pose overlays create display metadata. They do not draw on CPU frame
  pixels.
- Analytics, pose, ReID, V3DT, object depth, websocket, and REST payloads
  serialize metadata or compact feature data at the edge.
- BEV JSON is edge telemetry. BEV JPEG is CPU composition/encoding but disabled
  by default and should remain opt-in.

Risk areas if left unquarantined:

- Raw pyds tensor extraction can become a hidden host-copy fallback for
  MapAnything and conceal Service Maker tensor metadata failures.
- A second MapAnything reader would conceal failure of the exact typed
  same-buffer contract; wrapper, generic-native, and raw-PyDS alternatives
  therefore remain absent from its selection path.

## Final Keep / Change / Remove Calls

| Component | Call | Rationale |
| --- | --- | --- |
| Service Maker `BatchMetadataOperator` hooks | Keep | Canonical DS9 hook surface for metadata probes and custom processing. |
| MapAnything `nvinfer -> NVMM RGB -> fakesink` branch with `output-tensor-meta=1` | Keep | Canonical same-buffer tensor/RGB branch; avoids appsink or a separately reconciled video source. |
| MapAnything exact native tensor metadata handling | Keep | Canonical DS9 UID/layer/per-frame ownership adapter; startup rejects missing, stale, generic, or ambiguous bridge contracts. |
| MapAnything async CPU postprocess | Keep, bounded and joined | Edge dense-depth/RPC postprocess over owned arrays. Keep gated/async with explicit runtime teardown; move the probe-local copy only after an owned device-clone contract exists. |
| MapAnything pyds raw tensor path | Removed from canonical path | A duplicate reader could hide failure of the exact DS9 contract. |
| MapAnything generic/wrapper tensor alternatives | Removed from canonical path | Live wrapper evidence carried only sibling UID 5; exact typed UID 2 capture is the sole authority. |
| Trail overlay display metadata | Keep | Canonical Service Maker display meta path upstream of `nvdsosd`; no CPU drawing branch. |
| Pose keypoint overlay display metadata | Keep | Same canonical OSD/display meta path. |
| Analytics telemetry Service Maker path | Keep | Uses `frame_meta.object_items` and edge serialization. |
| Analytics pyds user-meta fallback | Remove or quarantine | Keep only if DS9 Python lacks documented analytics object item access. |
| BEV JSON telemetry | Keep | Edge metadata serialization. |
| BEV JPEG | Keep disabled by default | CPU composition/encoding is acceptable only as explicit diagnostic/UI output. |
| Native pose/depth/ReID metadata bridges | Keep for first DS9 port | They cover metadata surfaces not proven available in Service Maker Python. Continue to rebuild/smoke-test when touched. |

Overall recommendation: port DS9 with Service Maker metadata hooks as the
canonical path, keep custom Python at the metadata/serialization edge, and
quarantine or remove raw pyds fallbacks before declaring the DS9 path production
ready.
