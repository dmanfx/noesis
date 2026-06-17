# 07 Custom Python Hooks, MapAnything, BEV, Overlay, Telemetry

Date: 2026-05-10

Scope: DS9 migration audit for custom Python hooks while keeping live DS8 untouched.
This report only audits the DS9 tree. Code and config were not edited.

Audited local sources:

- [`../noesis/pipelines/hooks.py`](../noesis/pipelines/hooks.py)
- [`../noesis/pipelines/ds8_pipeline.py`](../noesis/pipelines/ds8_pipeline.py)
- [`../noesis/telemetry/bev.py`](../noesis/telemetry/bev.py)
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

DS9 Service Maker does provide the canonical shape this repo wants for
MapAnything tensor handling without an appsink or CPU video branch: keep the
full-frame `nvinfer` branch GPU/NVMM, enable `output-tensor-meta`, terminate the
branch at `fakesink`, and consume tensor metadata from a Service Maker metadata
probe. That is the canonical migration direction. It still needs an installed
DS9 smoke test proving the Python wrapper exposes the same `tensor_items` /
`as_tensor_output()` shape used here, but the design target is correct.

`BatchMetadataOperator`, Service Maker display metadata, and the trail overlay
path remain canonical. The DS9 code already puts overlays into display metadata
upstream of `nvdsosd`; it does not need an appsink branch to draw trails or
labels.

The custom Python audited here is mostly edge metadata, postprocess, telemetry,
and serialization. It does not introduce a CPU video branch in the canonical
pipeline. The places to quarantine for DS9 prep are the raw `pyds` fallbacks in
`hooks.py`: raw tensor extraction, raw object-list traversal, raw analytics user
meta traversal, OSD label fallback, and Python object pruning.

## MapAnything Tensor Path

### Local evidence

The DS9 builder creates a dedicated MapAnything branch behind a queue and valve,
then terminates it at a sink for metadata consumption rather than frame output.
It defaults the MapAnything SGIE to tensor metadata output through the builder
property map:

- `attach_tensor_meta` maps to `output-tensor-meta` in
  [`ds8_pipeline.py`](../noesis/pipelines/ds8_pipeline.py#L512-L532).
- The MapAnything branch sets `attach_tensor_meta=True`, `gie_id=2`, uses a
  leaky queue and gate valve, then links to a branch sink in
  [`ds8_pipeline.py`](../noesis/pipelines/ds8_pipeline.py#L1049-L1113).
- The MapAnything INI is full-frame SGIE input, GPU selected, tensor-from-meta
  disabled, and `output-tensor-meta=1` in
  [`config_infer_secondary_mapanything.ini`](../pipelines/config_infer_secondary_mapanything.ini#L1-L22).
- The design history explicitly records this as an SGIE branch that terminates
  at `fakesink` and consumes outputs through `BatchMetadataOperator` tensor meta,
  not a tiler/appsink path:
  [`ds8_design_decisions.md`](../docs/history/ds8/ds8_design_decisions.md#L386-L388).
- The runtime only attaches the MapAnything postprocess hook if MapAnything is
  enabled and the component exists in
  `ds9_runtime_core.py`.

The hook side is already metadata-first:

- `attach_mapanything_postprocess_hook` attaches a Service Maker `Probe` with
  `_MapAnythingOperator` in
  [`hooks.py`](../noesis/pipelines/hooks.py#L533-L570).
- `_MapAnythingOperator.handle_metadata` iterates `batch_meta.frame_items`,
  reads `frame_meta.tensor_items`, calls `as_tensor_output()`, matches the
  configured GIE id, and sends the tensor metadata to
  `MapAnythingProcessor.handle_nvds_tensor_ds8` in
  [`hooks.py`](../noesis/pipelines/hooks.py#L7405-L7507).
- `MapAnythingProcessor.handle_nvds_tensor_ds8` consumes Service Maker tensor
  metadata with `get_layers()`, clones tensors, and puts bounded async jobs on a
  worker queue in [`hooks.py`](../noesis/pipelines/hooks.py#L1431-L1502).
- The async worker performs tensor-to-NumPy conversion away from the probe
  thread and emits dense depth snapshots/RPC data in
  [`hooks.py`](../noesis/pipelines/hooks.py#L1504-L1777).

### DS9 call

Keep the Service Maker metadata-probe shape. It is the DS9 migration target for
MapAnything and does not need appsink or CPU video frames.

Change only the fallback discipline: the native tensor fallback
`handle_native_frame_ds8` is not a `pyds` fallback, but it is still a fallback.
For DS9 prep it should become an explicit DS9 adapter with loud logging and a
smoke-test gate, or be removed once DS9 Service Maker tensor metadata is proven.
It should not remain a silent canonical alternate.

The CPU tensor copy inside MapAnything postprocess is an edge postprocess/RPC
cost, not a CPU video branch. Keep it gated, bounded, and async. If DS9
performance targets require eliminating that dense host copy, the next canonical
move is a C++/CUDA postprocess or custom metadata producer, not appsink.

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
  [`hooks.py`](../noesis/pipelines/hooks.py#L38-L43).
- Trail overlay hook attachment is documented as upstream of `nvdsosd` because
  display metadata must be present before OSD in
  [`hooks.py`](../noesis/pipelines/hooks.py#L773-L817).
- `TrailOverlayProcessor.handle_batch_ds8` requires Service Maker OSD bindings,
  iterates `batch_meta.frame_items`, acquires display metadata, appends it to the
  frame, and uses `ds_osd.Line` / `ds_osd.Text` in
  [`hooks.py`](../noesis/pipelines/hooks.py#L2625-L3062).
- Pose keypoint overlay follows the same display metadata path with
  `ds_osd.Line` and `ds_osd.Circle` in
  [`hooks.py`](../noesis/pipelines/hooks.py#L3649-L3802).
- The main pipeline configures `nvdsosd` with GPU processing mode to keep the
  tiler to OSD to encoder path on GPU/NVMM in
  [`ds8_pipeline.py`](../noesis/pipelines/ds8_pipeline.py#L1211-L1224).
- The design history records trail overlay as Service Maker display metadata
  using `BatchMetadata.acquire_display_meta()`, `FrameMetadata.append()`, and
  `pyservicemaker.osd.Line/Text`, with no CPU appsink branch:
  [`ds8_design_decisions.md`](../docs/history/ds8/ds8_design_decisions.md#L545-L551).

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
  [`config/infer.yaml`](../config/infer.yaml#L135-L153).
- `BevRenderer.render_and_publish` only allocates the BGR image when JPEG is
  enabled, otherwise it builds JSON footpoint/trail telemetry in
  [`bev.py`](../noesis/telemetry/bev.py#L580-L1002).
- `_publish` always broadcasts JSON metadata and only encodes JPEG bytes when
  JPEG is enabled in [`bev.py`](../noesis/telemetry/bev.py#L1014-L1104).
- The design history records meta-only BEV as the default and frequent BEV JPEG
  as disabled by default:
  [`ds8_design_decisions.md`](../docs/history/ds8/ds8_design_decisions.md#L560-L561) and
  [`ds8_design_decisions.md`](../docs/history/ds8/ds8_design_decisions.md#L623-L625).

Telemetry hooks are metadata-first:

- Analytics telemetry attaches a `BatchMetadataOperator` in
  [`hooks.py`](../noesis/pipelines/hooks.py#L714-L770).
- The production Service Maker path iterates `frame_meta.object_items` in
  [`hooks.py`](../noesis/pipelines/hooks.py#L4688-L5019).
- ReID, pose, depth, and V3DT data extraction are edge metadata/identity inputs,
  not video frame copies. They do account host-copy counters where they cross to
  CPU, for example pose metadata in
  [`hooks.py`](../noesis/pipelines/hooks.py#L3296-L3352), pose payload
  serialization in [`hooks.py`](../noesis/pipelines/hooks.py#L3455-L3478), and
  ReID embeddings in [`hooks.py`](../noesis/pipelines/hooks.py#L4645-L4686).

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
| Optional import | `import pyds` with `PYDS_AVAILABLE` | [`hooks.py`](../noesis/pipelines/hooks.py#L45-L48) | Module-wide optional compatibility | Keep only if a documented DS9 gap remains; otherwise quarantine behind a compatibility module. |
| Raw frame tensor iterator | `_iter_frame_tensor_meta` walks `frame_user_meta_list`, casts `NvDsUserMeta`, checks `NVDSINFER_TENSOR_OUTPUT_META`, casts `NvDsInferTensorMeta` | [`hooks.py`](../noesis/pipelines/hooks.py#L275-L308) | Not used by the Service Maker MapAnything operator path | Remove or quarantine after DS9 tensor-items smoke test passes. |
| Raw tensor dtype mapping | `_layer_dtype` imports `NvDsInferDataType` | [`hooks.py`](../noesis/pipelines/hooks.py#L1039-L1065) | Used by raw pyds tensor fallback | Remove from canonical MapAnything path. |
| Raw tensor buffer copy | `_numpy_from_layer` calls `pyds.get_ptr(layer.buffer)` and copies layer bytes into NumPy | [`hooks.py`](../noesis/pipelines/hooks.py#L1068-L1104) | Used by raw pyds tensor fallback | Remove from production; this is a host tensor fallback, not canonical DS9 Service Maker handling. |
| Raw layer extraction | `_extract_tensor_layers` falls back to `pyds.get_nvds_LayerInfo` | [`hooks.py`](../noesis/pipelines/hooks.py#L1118-L1160) | Used by raw pyds tensor fallback | Remove or isolate in a debug-only adapter. |
| MapAnything raw pyds handler | `MapAnythingProcessor.handle_nvds_tensor` calls `_extract_tensor_layers` | [`hooks.py`](../noesis/pipelines/hooks.py#L1570-L1580) | Parallel legacy path beside `handle_nvds_tensor_ds8` | Remove from DS9 canonical path once Service Maker tensor metadata is validated. |
| Generic pyds helpers | `_resolve_pyds_cast`, `_resolve_pyds_attr`, `_iter_meta_entries` | [`hooks.py`](../noesis/pipelines/hooks.py#L7597-L7613) | Support legacy/pyds traversal helpers | Quarantine in compatibility code if still needed for tests. |
| Analytics raw user meta | `_extract_analytics_obj_meta` walks object user meta and casts `NvDsAnalyticsObjInfo` | [`hooks.py`](../noesis/pipelines/hooks.py#L6573-L6618) | The production Service Maker analytics path uses `nvdsanalytics_obj_items`; this is a fallback | Keep only if DS9 Python lacks the Service Maker analytics item surface; otherwise remove. |
| Analytics raw object builder | `_build_track_dict` uses non-Service-Maker object fields and pyds analytics extraction | [`hooks.py`](../noesis/pipelines/hooks.py#L6529-L6572) | Production path uses `_build_track_dict_ds8` | Quarantine or remove from DS9 production. |
| OSD label raw fallback | `_OsdLabelProcessor.handle_frame` uses pyds-style object iteration | [`hooks.py`](../noesis/pipelines/hooks.py#L7000-L7006) | Production operator calls `handle_frame_ds8` | Remove or quarantine; keep Service Maker object-items path. |
| Python object pruning | `_ExcludePruneProcessor.handle_frame` and `_remove_obj` use raw frame object lists and `nvds_remove_obj_meta_from_frame` | [`hooks.py`](../noesis/pipelines/hooks.py#L7251-L7335) | Production operator calls `handle_frame_ds8`, which is read-only and logs would-prune | Remove from production. Use `nvdsroiexclude` or a native/plugin path for real pruning. |

Not pyds, but still a fallback to track:

| Area | Current path | Evidence | DS9 call |
| --- | --- | --- | --- |
| MapAnything native tensor fallback | `_MapAnythingOperator.handle_metadata` falls back to `handle_native_frame_ds8` when Service Maker tensor items do not match; `handle_native_frame_ds8` uses `noesis_depth_tracking_tensor_ext.capture_tensor_layers` | [`hooks.py`](../noesis/pipelines/hooks.py#L7458-L7497), [`hooks.py`](../noesis/pipelines/hooks.py#L1598-L1634) | Quarantine as an explicit DS9 adapter or remove after DS9 Service Maker tensor metadata is validated. Do not let it silently mask a broken canonical tensor path. |

## GPU-Only And Zero-Copy Assessment

No audited DS9 custom Python path adds an appsink or CPU video branch to the
canonical pipeline. The main video path remains DeepStream GPU/NVMM through
streammux, inference, tracking, tiler, GPU `nvdsosd`, and RTSP/WebRTC output.

The CPU crossings found here are edge metadata or serialization:

- MapAnything dense tensors are cloned from metadata and converted to NumPy in a
  bounded async worker for depth snapshots and depth RPC. This is a real host
  tensor copy, but it is not a CPU video-frame branch.
- Trail and pose overlays create display metadata. They do not draw on CPU frame
  pixels.
- Analytics, pose, ReID, V3DT, object depth, websocket, and REST payloads
  serialize metadata or compact feature data at the edge.
- BEV JSON is edge telemetry. BEV JPEG is CPU composition/encoding but disabled
  by default and should remain opt-in.

Risk areas if left unquarantined:

- Raw pyds tensor extraction can become a hidden host-copy fallback for
  MapAnything and conceal Service Maker tensor metadata failures.
- Python object pruning through pyds can become a hidden replacement for
  `nvdsroiexclude` or a native/plugin pruning path.
- The native MapAnything tensor fallback can conceal a broken Service Maker
  tensor-items path unless it is logged and gated as an explicit compatibility
  adapter.

## Final Keep / Change / Remove Calls

| Component | Call | Rationale |
| --- | --- | --- |
| Service Maker `BatchMetadataOperator` hooks | Keep | Canonical DS9 hook surface for metadata probes and custom processing. |
| MapAnything `nvinfer` branch with `output-tensor-meta=1` and `fakesink` | Keep | Canonical tensor metadata branch; avoids appsink/CPU video frames. |
| MapAnything Service Maker tensor metadata handling | Keep | Correct canonical target; validate installed DS9 Python wrapper with a focused smoke test. |
| MapAnything async CPU postprocess | Keep, bounded | Edge dense-depth/RPC postprocess. Keep gated/async; move to C++/CUDA only if DS9 performance demands it. |
| MapAnything pyds raw tensor path | Remove or quarantine | Duplicate fallback that copies raw tensors to CPU and can hide canonical tensor metadata failures. |
| MapAnything native tensor fallback | Change/quarantine | Not pyds, but still a fallback. Make explicit and loud, or remove after DS9 tensor metadata validation. |
| Trail overlay display metadata | Keep | Canonical Service Maker display meta path upstream of `nvdsosd`; no CPU drawing branch. |
| Pose keypoint overlay display metadata | Keep | Same canonical OSD/display meta path. |
| Analytics telemetry Service Maker path | Keep | Uses `frame_meta.object_items` and edge serialization. |
| Analytics pyds user-meta fallback | Remove or quarantine | Keep only if DS9 Python lacks documented analytics object item access. |
| BEV JSON telemetry | Keep | Edge metadata serialization. |
| BEV JPEG | Keep disabled by default | CPU composition/encoding is acceptable only as explicit diagnostic/UI output. |
| Python pyds object pruning | Remove from production | Real pruning should stay in `nvdsroiexclude` or native/plugin code, not Python list mutation. |
| Native pose/depth/ReID metadata bridges | Keep for first DS9 port | They cover metadata surfaces not proven available in Service Maker Python. Continue to rebuild/smoke-test when touched. |

Overall recommendation: port DS9 with Service Maker metadata hooks as the
canonical path, keep custom Python at the metadata/serialization edge, and
quarantine or remove raw pyds fallbacks before declaring the DS9 path production
ready.
