# DS9 ROI Exclusion Audit

Scope: `nvdsroiexclude`, pre-tracker ROI pruning, and whether DS9 provides an
official replacement for the existing object-meta removal semantics. Live DS8
runtime code/config remains untouched.

## Executive Answer

DS9 provides the official metadata removal API
`nvds_remove_obj_meta_from_frame`, but I did not find an official DS9
GStreamer plugin that is equivalent to the current `nvdsroiexclude`
pre-tracker polygon pruning behavior.

`nvdsanalytics` ROI filtering is not an object-removal replacement. Official
DS9 docs describe it as an analytics element that returns the original batched
buffer and attaches frame/object analytics metadata. ROI filtering detects
whether objects are in an ROI and attaches/counts that result; it does not
remove `NvDsObjectMeta` before `nvtracker`.

The current `nvdsroiexclude` source is not available in this checkout. Only
build products and CMake dependency records are present under
`DS9/artifacts/nvdsroiexclude_build/` and `build/nvdsroiexclude/`.

## Evidence

### Official DS9 docs

- [DS9 metadata API](https://docs.nvidia.com/metropolis/deepstream/9.0/sdk-api/group__metadata__structures.html#_CPPv431nvds_remove_obj_meta_from_frameP13NvDsFrameMetaP14NvDsObjectMeta):
  `nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)` is an official API
  and removes object metadata from the frame metadata.
- [DS9 Python API](https://docs.nvidia.com/metropolis/deepstream/9.0/python-api/PYTHON_API/Methods/methodsdoc.html#nvds-remove-obj-meta-from-frame):
  `pyds.nvds_remove_obj_meta_from_frame(...)` is also listed in the official
  Python bindings.
- [DS9 Gst-nvdsanalytics](https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_plugin_gst-nvdsanalytics.html):
  the plugin performs ROI filtering, overcrowding, direction, and line crossing
  on metadata from `nvinfer`/`nvtracker`, and its outputs are the original
  batched buffer plus `NvDsAnalyticsFrameMeta` and `NvDsAnalyticsObjInfo`.
- [DS9 nvdsanalytics ROI config](https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_plugin_gst-nvdsanalytics.html#configuration-file-parameters):
  `roi-<label>` and `inverse-roi` configure labels/counting behavior. The docs
  describe attaching/counting ROI status, not deleting object metadata.
- [DS9 Gst-nvinfer](https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_plugin_gst-nvinfer.html):
  `nvinfer` has detector output filtering by size/class and top/bottom RoI
  bands, but that is not arbitrary per-stream polygon exclusion equivalent to
  `nvdsroiexclude`.
- [DS9 plugin overview](https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_plugin_Intro.html):
  the public plugin guide lists `nvdsanalytics` and other DS9 plugins, but no
  official `nvdsroiexclude` plugin.

### Local DS9 snapshot

- [Static_ROI_Exclusion.md](../../../Static_ROI_Exclusion.md) records the
  current behavior: `nvdsroiexclude` is an in-place metadata-only
  `GstBaseTransform` placed before `nvtracker`, removing object metas fully
  inside configured ROIs.
- [config/infer.yaml](../../../../DS9/config/infer.yaml) enables an `analytics.exclude`
  stage with `element: nvdsroiexclude` and
  `config-file: config/config_nvdsanalytics_exclude.ini`.
- [config/config_nvdsanalytics_exclude.ini](../../../../DS9/config/config_nvdsanalytics_exclude.ini)
  carries the per-stream polygon exclusion ROIs currently intended for the
  plugin.
- [config/nvdsanalytics.yaml](../../../../DS9/config/nvdsanalytics.yaml) keeps an `exclude`
  stage shaped like `nvdsanalytics` ROI config, but local pipeline config
  overrides the element to `nvdsroiexclude` for actual pre-tracker removal.
- [noesis/pipelines/deepstream_pipeline.py](../../../../DS9/noesis/pipelines/deepstream_pipeline.py)
  builds `analytics_exclude` from the `analytics.exclude` block and links it
  before `tracker`.
- [noesis/pipelines/hooks.py](../../../../DS9/noesis/pipelines/hooks.py) contains
  `_ExcludePruneProcessor`, but the DS8 Service Maker path only logs would-be
  removals because `object_items` are read-only. The pyds removal path exists
  in the generic handler, but it is not the current proven pre-tracker DS9
  replacement.
- [noesis/server/analytics_api.py](../../../../DS9/noesis/server/analytics_api.py) renders
  an `nvdsroiexclude`-style INI from the analytics ROI model, so the REST/API
  surface is already tied to the custom plugin format.

## Source Availability Finding

Search result: no source tree for `csrc/nvdsroiexclude/` or
`DS9/csrc/nvdsroiexclude/` exists in this checkout.

Found artifacts:

- `DS9/artifacts/nvdsroiexclude_build/libgstnvdsroiexclude.so`
- `DS9/artifacts/nvdsroiexclude_build/CMakeCache.txt`
- `DS9/artifacts/nvdsroiexclude_build/CMakeFiles/gstnvdsroiexclude.dir/gstnvdsroiexclude.cpp.o.d`

The CMake cache points at
`/home/mayor/Noesis_Devel/csrc/nvdsroiexclude` as the original source
directory, and the dependency file names `gstnvdsroiexclude.cpp` and
`gstnvdsroiexclude.h`. Those source files are absent. Binary strings in
`libgstnvdsroiexclude.so` still expose the element name, class name,
`config-file`, `id-mode`, `source-id`, `pad-index`, `osd-mode`, and
`nvds_remove_obj_meta_from_frame`, but strings are not a rebuildable source.

Conclusion: this checkout currently has a build artifact only, not maintainable
plugin source.

## Answers

1. Does DS9 offer an official equivalent?

Yes for the low-level metadata removal API. No for an official pre-tracker
polygon ROI pruning plugin equivalent to the current `nvdsroiexclude` behavior.
The DS9-safe implementation still needs a custom element/probe that uses the
official API at the correct point in the graph.

2. Can `nvdsanalytics` ROI exclusion remove objects before tracker?

No. It can be run on primary detector metadata and can annotate/count ROI
membership, but official DS9 docs describe its output as original buffer plus
analytics metadata. It is not a metadata deletion element and should not be used
as the DS9 replacement for pre-tracker pruning.

3. Is current plugin source available?

No source was found in this checkout. Only binary/build artifacts are present.

4. What must happen before DS9 migration?

Before cutover, recover or reconstruct `nvdsroiexclude` source under the DS9
sandbox, rebuild it against DS9 headers/libraries, and validate it as a
pre-tracker element. Acceptance should include `gst-inspect-1.0
nvdsroiexclude`, successful pipeline load, ROI config reload behavior, and a
smoke test showing objects inside exclusion polygons are absent from tracker
input and downstream ReID/pose, not merely hidden or marked after tracking.

## Keep / Replace / Reconstruct Calls

| Component | Call | Rationale |
| --- | --- | --- |
| Pre-tracker ROI pruning semantics | Keep | Required to avoid creating tracks/ReID/pose work for excluded detections. `nvdsanalytics` cannot supply this behavior. |
| `nvdsanalytics` as replacement | Replace: no | It annotates/counts ROI events; it does not remove object metadata. Keep it for post-tracker analytics, not exclusion pruning. |
| DS9 official metadata API | Keep/use | `nvds_remove_obj_meta_from_frame` is the official DS9 mechanism to delete object metadata. |
| Current `libgstnvdsroiexclude.so` artifact | Do not rely on it for migration | It is a binary artifact with missing source and must not be the only DS9 migration path. |
| `nvdsroiexclude` source | Reconstruct or recover | Required before DS9 migration can be considered complete and auditable. |
| Python `_ExcludePruneProcessor` removal path | Do not promote as first replacement | It is not the proven current pre-tracker plugin path and risks moving pruning into a Python/hook hot path unless explicitly redesigned and validated. |

Final recommendation: keep the behavior, do not replace it with
`nvdsanalytics`, and reconstruct/recover a source-controlled DS9
`nvdsroiexclude` plugin before migration.
