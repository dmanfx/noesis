# DS9 Service Maker, PyDS, Intrinsics, and Latency Audit

Date: 2026-05-10

Scope: DS9 prep only. Live DS8 remains read-only context.

Files audited:

- `DS9/README.md`
- `DS9/noesis/pipelines/ds8_pipeline.py`
- `DS9/noesis/pipelines/hooks.py`
- `DS9/noesis/metadata/intrinsics.py`
- `DS9/noesis/telemetry/latency_metrics.py`
- `DS9/native/noesis_latency_ext.cpp`
- `docs/history/runtime/ds8/DS8_README_FOR_AGENTS.md`
- `docs/history/runtime/ds8/DS8_metadata_contracts.md`
- `DS9/docs/MapAnything_Depth.md`
- `DS9/docs/Static_ROI_Exclusion.md`
- `docs/history/runtime/ds8/DS8_pose_stable_id_integration.md`

Official DeepStream 9 sources used for DS9 claims:

- Service Maker Python overview: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_python.html
- Service Maker Pipeline APIs: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_python_into_to_pipeline_api.html
- Service Maker Python advanced features: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_python_advanced_features.html
- Service Maker plugin and latency probe example: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_plugin.html
- DeepStream OpenTelemetry support: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_OpenTelemetry.html
- PyDS methods: https://docs.nvidia.com/metropolis/deepstream/dev-guide/python-api/PYTHON_API/Methods/methodsdoc.html
- PyDS `NvDsObjectMeta`: https://docs.nvidia.com/metropolis/deepstream/dev-guide/python-api/PYTHON_API/NvDsMeta/NvDsObjectMeta.html
- PyDS `NvDsUserMeta`: https://docs.nvidia.com/metropolis/deepstream/dev-guide/python-api/PYTHON_API/NvDsMeta/NvDsUserMeta.html
- PyDS `NvDsInferTensorMeta`: https://docs.nvidia.com/metropolis/deepstream/dev-guide/python-api/PYTHON_API/NvDsInfer/NvDsInferTensorMeta.html
- PyDS `NvDsAnalyticsObjInfo`: https://docs.nvidia.com/metropolis/deepstream/dev-guide/python-api/PYTHON_API/NvDsAnalyticsMeta/NvDsAnalyticsObjInfo.html
- PyDS `NvDsMetaType`: https://docs.nvidia.com/metropolis/deepstream/9.0/python-api/PYTHON_API/NvDsMeta/NvDsMetaType.html
- DS9 C latency API header reference: https://docs.nvidia.com/metropolis/deepstream/9.0/sdk-api/9_80_2sources_2includes_2nvds__latency__meta_8h.html

## Executive Decision

DS9 Service Maker Python exposes the core APIs this prep tree needs for graph construction, runtime state transitions, probes, batch metadata iteration, and display metadata. Keep Service Maker as the canonical graph and metadata-hook layer.

PyDS should not replace Service Maker for ordinary frame/object traversal, display metadata, or pipeline control. Keep PyDS quarantined to raw `NvDs*` access where Service Maker still lacks a documented DS9 Python surface or where a native plugin/bridge is explicitly chosen.

DS9 OpenTelemetry can replace custom latency exposure for production observability if Noesis is willing to consume/export metrics via the DS9 `nvmultiurisrcbin` + `nvdslogger` path. It does not, by itself, replace the current in-process rolling per-source latency samples returned by `noesis_latency_ext.measure_buffer_latency(...)`.

Per-frame intrinsics `NvDsUserMeta` should not be a required DS9 runtime contract. DS9 prep can rely on calibration bundles and the runtime calibration provider for canonical camera intrinsics/extrinsics. Keep the loader/parser, but quarantine the PyDS frame-user-meta attachment as an explicit adapter only.

## Service Maker Exposure

Verdict: keep.

Official DS9 Service Maker Python docs expose:

- `Pipeline` graph APIs: `add`, `link`, `attach`, `start`, and `wait`.
- Runtime state APIs: `prepare`, `activate`, `wait`, and `stop`.
- `Probe` plus `BatchMetadataOperator.handle_metadata(batch_meta)`.
- `batch_meta.frame_items`, `frame_meta.object_items`, `batch_meta.acquire_display_meta()`, Service Maker `osd` drawing primitives, and `frame_meta.append(display_meta)`.
- `BufferOperator` plus `Probe` for in-place buffer inspection/gating.

That covers the DS9 copies of:

- graph construction and activation in `DS9/noesis/pipelines/ds8_pipeline.py`
- intrinsics, MapAnything, pose, depth-fusion, analytics, label, trail, keypoint, and exclusion hook attachment in `DS9/noesis/pipelines/hooks.py`
- Service Maker display-meta overlays for labels, trails, and pose keypoints
- the current `BufferOperator` depth gate/FPS/latency-probe shape

Gaps: the official Service Maker Python docs do not prove a full replacement for arbitrary object-level custom user-meta attach/extract, raw `NvDsInferTensorMeta` fallback traversal, or metadata removal. Those paths should stay native/plugin-backed or explicitly quarantined, not silently recast as ordinary Service Maker Python.

## PyDS Exposure

Verdict: quarantine, with narrow kept uses.

Official DS9 PyDS exposes canonical raw metadata APIs for `NvDsObjectMeta`, `NvDsUserMeta`, `NvDsInferTensorMeta`, `NvDsAnalyticsObjInfo`, user meta add/remove, display meta add/remove, object removal, metadata locks, `gst_buffer_get_nvds_batch_meta`, and `nvds_measure_buffer_latency`.

That does not make PyDS the preferred DS9 path for this prep tree. In this code, Service Maker wrappers already cover the normal hook path. PyDS is only more canonical than ad hoc ctypes/string/object poking when the code is deliberately working with raw `NvDs*` metadata.

Recommended policy:

- Prefer Service Maker metadata wrappers for production `BatchMetadataOperator` hooks.
- Prefer native bridges/plugins for object-level user meta attach/extract and pre-tracker object pruning when Service Maker Python cannot safely expose the mutation.
- Use PyDS only behind an explicit DS9 adapter boundary, with a removal target once Service Maker/native coverage is confirmed.

## Latency Exposure

Verdict: change default, quarantine custom shim.

Current DS9 copy has two latency routes:

- `LatencyCollector.record_from_gst_buffer_ptr(...)` loads DeepStream latency symbols with ctypes, gets batch meta through PyDS, locks metadata, and calls the C latency function with a scratch `NvDsFrameLatencyInfo` array.
- `LatencyCollector.record_from_sm_buffer(...)` calls `noesis_latency_ext.measure_buffer_latency(buffer)`, whose native implementation wraps Service Maker C++ `Buffer.measureLatency()`.

Official DS9 alternatives:

- DeepStream OpenTelemetry supports export through `nvmultiurisrcbin`; `Gst-nvdslogger` collects FPS, latency, and frame numbers and exports through OTLP/HTTP.
- Official PyDS exposes `pyds.nvds_measure_buffer_latency(gst_buffer: int)`, but the Python doc describes a source-count return, not per-source latency sample structs.
- Official Service Maker plugin docs show a C++ latency probe using `buffer.measureLatency()`.

Decision:

- Use DS9 OpenTelemetry as the default production latency exposure path if external Prometheus/Grafana/OTLP metrics satisfy the product surface.
- Do not keep `noesis_latency_ext` as the default DS9 path just to duplicate DS9 OTel.
- If Noesis must keep an in-process REST/WebSocket latency snapshot, keep a quarantined latency adapter: either the current native shim or a DS9 Service Maker C++ plugin that exports sample values into Noesis. PyDS alone is not enough evidence for the existing rolling-sample API.
- Remove the duplicate `noesis_latency_ext` import block in a later code-edit pass.

## Intrinsics

Verdict: calibration bundle/provider is canonical; per-frame user meta is optional/quarantined.

Local DS9 docs copied from DS8 already identify the calibration bundle/runtime provider as the primary intrinsics path and frame user meta as optional. The audited code confirms that `DS9/noesis/metadata/intrinsics.py` still supports `NvDsUserMeta` attachment through PyDS, while `DS9/noesis/pipelines/hooks.py` attaches an intrinsics `BatchMetadataOperator` to `streammux`.

Decision:

- Keep camera config parsing and runtime calibration bundle/provider ownership.
- Do not require `NOESIS.INTRINSICS` per-frame user meta in DS9 prep.
- Quarantine `attach_intrinsics(...)->_attach_to_pyds_frame(...)` as an explicit adapter for a future DS plugin or bridge that truly requires frame-local K values.
- Remove mandatory intrinsics probe wiring from the canonical DS9 path after verifying every depth/BEV/world consumer reads from the runtime calibration provider.

## Production PyDS Paths

| File | PyDS path | Current purpose | DS9 disposition |
| --- | --- | --- | --- |
| `DS9/noesis/pipelines/ds8_pipeline.py` | none | Service Maker graph, probes, prepare/activate | Keep Service Maker. No PyDS action. |
| `DS9/noesis/pipelines/hooks.py` | `import pyds`; `_resolve_pyds_cast`; `_resolve_pyds_attr`; `_iter_meta_entries` | Shared raw-`NvDs*` helper surface | Quarantine. Keep only while raw PyDS adapters remain. |
| `DS9/noesis/pipelines/hooks.py` | `_iter_frame_tensor_meta`: `frame_user_meta_list`, `NvDsUserMeta.cast`, `NvDsInferTensorMeta.cast`, `NVDSINFER_TENSOR_OUTPUT_META` | Raw tensor-meta fallback by `gie_id` | Change/quarantine. Prefer Service Maker `frame_meta.tensor_items` + `as_tensor_output()` + tensor layers. |
| `DS9/noesis/pipelines/hooks.py` | `_layer_dtype`: `NvDsInferDataType` | Decode raw tensor layer dtype | Quarantine with raw tensor fallback only. |
| `DS9/noesis/pipelines/hooks.py` | `_numpy_from_layer`: `pyds.get_ptr(layer.buffer)` | CPU-copy raw tensor layer buffer | Quarantine/remove from default. It is a host-copy path and should not be canonical GPU-first DS9. |
| `DS9/noesis/pipelines/hooks.py` | `_extract_tensor_layers`: `pyds.get_nvds_LayerInfo(...)` | Raw tensor layer traversal | Change. If PyDS fallback remains, use documented `NvDsInferTensorMeta.output_layers_info(j)` or native bridge coverage; do not depend on undocumented helper names. |
| `DS9/noesis/pipelines/hooks.py` | `MapAnythingProcessor.handle_nvds_tensor(...)` | Raw `NvDsInferTensorMeta` MapAnything processing | Quarantine. Primary DS9 path should use Service Maker tensor metadata or native tensor bridge if Service Maker coverage is incomplete. |
| `DS9/noesis/pipelines/hooks.py` | `_iter_object_meta`, `_OsdLabelProcessor.handle_frame`: `NvDsObjectMeta.cast`, `obj_meta_list` | Raw object traversal for non-Service-Maker metadata | Quarantine. Service Maker `object_items` path is canonical in these hooks. |
| `DS9/noesis/pipelines/hooks.py` | `_extract_analytics_obj_meta`: `obj_user_meta_list`, `NvDsUserMeta.cast`, `NvDsAnalyticsObjInfo.cast`, `nvds_get_user_meta_type` | Raw analytics object meta extraction | Quarantine. Keep only if DS9 Service Maker lacks `nvdsanalytics_obj_items`; otherwise use Service Maker wrapper. |
| `DS9/noesis/pipelines/hooks.py` | `_ExcludePruneProcessor.handle_frame`: `NvDsObjectMeta.cast`, `nvds_remove_obj_meta_from_frame` | Python-side object pruning | Remove from production. Keep `nvdsroiexclude`/native plugin behavior; fail loudly if the plugin path is unavailable. |
| `DS9/noesis/metadata/intrinsics.py` | `nvds_acquire_user_meta_from_pool`, `nvds_add_user_meta_to_frame`, `NvDsUserMeta.cast`, `nvds_get_user_meta_type` | Attach/copy/release `NOESIS.INTRINSICS` frame user meta | Change/quarantine. Do not require in canonical DS9; rely on calibration bundle/provider. |
| `DS9/noesis/telemetry/latency_metrics.py` | `pyds.gst_buffer_get_nvds_batch_meta`, `nvds_acquire_meta_lock`, `nvds_release_meta_lock` | Batch sizing and locking for ctypes latency call | Change. Prefer DS9 OTel for default metrics; keep only for explicit in-process latency adapter. |
| `DS9/noesis/telemetry/latency_metrics.py` | ctypes load of `nvds_measure_buffer_latency` plus `NvDsFrameLatencyInfo` mirror | Retrieve per-source latency samples from C API | Change/quarantine. Official PyDS documents `nvds_measure_buffer_latency`, but not the current per-source sample return shape. |
| `DS9/native/noesis_latency_ext.cpp` | no PyDS; `deepstream::Buffer.measureLatency()` wrapper | Service Maker Buffer latency samples for Python | Change/quarantine. Replace default production exposure with DS9 OTel, or keep only if Noesis still needs in-process sample values. |

## Final Keep / Change / Remove Table

| Item | Decision | Rationale |
| --- | --- | --- |
| Service Maker `Pipeline` graph construction/linking/attach/start/prepare/activate | Keep | Official DS9 Service Maker Python exposes these APIs and they match the staged graph code. |
| Service Maker `BatchMetadataOperator` + `Probe` hooks | Keep | Official DS9 docs show batch/frame/object metadata traversal through this exact model. |
| Service Maker display metadata (`batch_meta.acquire_display_meta`, `osd.Text/Line/Circle`, `frame_meta.append`) | Keep | Official DS9 docs expose the display-meta pattern needed by labels, trails, and keypoints. |
| PyDS for ordinary display/meta traversal already covered by Service Maker | Remove from canonical path | It is lower-level and not more canonical for Service Maker hooks. |
| PyDS raw tensor fallback | Change/quarantine | Keep only behind an explicit adapter until Service Maker tensor metadata is proven sufficient for MapAnything/depth/pose. |
| PyDS analytics object meta fallback | Change/quarantine | Prefer Service Maker analytics wrappers; retain PyDS only if DS9 wrapper coverage is missing. |
| Python-side PyDS object pruning | Remove from production | `nvdsroiexclude`/native plugin is the safe pre-tracker path; hidden Python pruning fallback should not mask plugin failure. |
| Per-frame intrinsics `NvDsUserMeta` | Remove from canonical DS9 contract; quarantine adapter | Calibration bundle/runtime provider is the canonical prep path. |
| Intrinsics config loader and calibration bundle/provider usage | Keep | Required for camera geometry, BEV/world/depth consumers, and avoids per-frame metadata dependency. |
| `noesis_latency_ext` default path | Change/quarantine | DS9 OTel should own production metrics; shim is only justified for in-process per-source samples. |
| `LatencyCollector` rolling snapshot model | Change | Keep only if product APIs require in-process latency; otherwise consume/export DS9 OTel metrics. |
| ctypes latency C API path | Change/quarantine | Official PyDS exposes the latency call but not the current per-source sample struct API; avoid private ctypes default. |
