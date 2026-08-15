# DS9 Prep Decision Ledger

Date: 2026-05-10

Historical note: this is the first-port decision ledger. For current launch
status and commands, start with `README.md`, `docs/migration_state.md`, and
`docs/validation_runbook.md`.

Scope: first-port DS9 preparation only. Live DS8 runtime paths remain untouched.

This ledger consolidates the bridge audit reports into definitive first-port
calls. The rule remains fail-fast: if DS9 does not prove an equivalent
canonical API or plugin path, keep or rebuild the existing Noesis bridge/parser
for the first DS9 bring-up instead of hiding the gap behind a fallback.

## First-Port Calls

| Area | Call | Evidence reports |
| --- | --- | --- |
| Pose metadata bridge | Keep/rebuild `noesis_pose_meta_ext`. DS9 proves native object tensor/user-meta surfaces, but not a Service Maker Python replacement for object-level SGIE tensor access, custom user-meta lifecycle, keypoint decode, and pose feature payload attach/read. | `02_pose_bridge.md`, `07_custom_python_hooks.md` |
| YOLO26 pose SGIE config | Keep parser-free tensor mode: `network-type=100` and `output-tensor-meta=1`. | `02_pose_bridge.md`, `06_custom_parsers.md` |
| YOLO26 pose no-op parser | Remove from the active DS9 build/runtime target. Archive the source as historical evidence because the DS9 pose config does not reference a custom parser and DS9 documents tensor-only inference. | `02_pose_bridge.md`, `06_custom_parsers.md` |
| Depth object metadata bridge | Keep/rebuild `noesis_depth_meta_ext`. DS9 Python Service Maker replacement is not proven for arbitrary object user-meta lifecycle or Service Maker object mask extraction. | `03_depth_bridges.md` |
| Depth tracking tensor bridge | Keep/rebuild `noesis_depth_tracking_tensor_ext`. Validate DS9 DAv2 device tensor behavior, `out_buf_ptrs_dev`, and `disable-output-host-copy=1` before claiming zero-copy parity. | `03_depth_bridges.md` |
| ReID metadata bridge | Keep/rebuild `noesis_reid_meta_ext`. DS9 does not prove Python object-level SGIE tensor access at the current StableID hook point. Tracker ReID is a future experiment, not the first-port replacement. | `04_reid_v3dt_bridges.md` |
| V3DT metadata bridge | Keep/rebuild `noesis_v3dt_meta_ext`. DS9 C++ exposes the V3DT metadata wrappers, but Python exposure and world-foot parity are not proven. | `04_reid_v3dt_bridges.md` |
| YOLO26-Seg parser | Keep/rebuild against DS9 headers. DS9 has no built-in parser for the fused YOLO26 mask layout or its GPU-first fused-output contract. | `06_custom_parsers.md` |
| YOLO11-Seg parser | Keep/rebuild against DS9 headers. Harden or constrain the output-copy/dtype contract during implementation. | `06_custom_parsers.md` |
| RF-DETR-Seg parser | Keep/rebuild against DS9 headers. Person-logit scoring, class remap, and bbox-relative masks remain model-specific. | `06_custom_parsers.md` |
| ROI exclusion | Keep pre-tracker pruning semantics and reconstruct or recover `nvdsroiexclude` source under DS9. Do not treat `nvdsanalytics` or Python object pruning as an equivalent replacement. | `05_roi_exclude.md` |
| PyDS raw metadata paths | Quarantine behind explicit DS9 adapter/debug boundaries and remove from the canonical path when Service Maker/native coverage is proven. Do not let raw tensor extraction, analytics fallback, OSD fallback, or Python pruning silently mask failures. | `01_servicemaker_pyds_intrinsics_latency.md`, `07_custom_python_hooks.md` |
| Latency | Use DS9 OpenTelemetry as the default production latency exposure path. Keep `noesis_latency_ext` or ctypes latency only as an explicit in-process adapter if product APIs still require rolling per-source samples. | `01_servicemaker_pyds_intrinsics_latency.md` |
| Intrinsics | Use the calibration bundle/runtime calibration provider as canonical. Do not make per-frame `NOESIS.INTRINSICS` user meta a required DS9 runtime contract. | `01_servicemaker_pyds_intrinsics_latency.md` |
| Service Maker hooks | Keep `BatchMetadataOperator`, metadata probes, display metadata overlays, MapAnything tensor metadata branch, analytics telemetry, and BEV JSON as the canonical DS9 shape. | `01_servicemaker_pyds_intrinsics_latency.md`, `07_custom_python_hooks.md` |

## Non-Goals For First Port

- Do not replace Noesis native metadata bridges merely because DS9 exposes the
  underlying C++ metadata classes.
- Do not replace dedicated OSNet ReID SGIE with tracker-internal ReID until a
  separate DS9 experiment proves feature availability, cadence, identity
  stability, and StableID parity.
- Do not use CPU video branches, appsink drawing, Python object pruning, or raw
  PyDS tensor extraction as hidden compatibility paths.
- Do not rely on copied binary artifacts for `nvdsroiexclude`; source must be
  recovered or reconstructed before the DS9 migration can be audited.

## Prep Completed After Audit

- Reconstructed `nvdsroiexclude` source under `csrc/nvdsroiexclude/` so the DS9
  path no longer depends only on copied binary/build artifacts.
- Added DS9-only build wrappers under `scripts/` that refuse the active DS8
  symlink and require `DS9_DEEPSTREAM_HOME` or
  `/opt/nvidia/deepstream/deepstream-9.0`.
- Updated active parser Makefiles to default to
  `/opt/nvidia/deepstream/deepstream-9.0` and fail before build if the
  DeepStream root does not identify as DS9. The launch build wrapper emits
  parser outputs under their active `DS9/pipelines/nvdsinfer_*` directories.
- Added `DS9_REBUILD_AND_SMOKE_GATES.md` for the post-install rebuild and smoke
  sequence.

## Acceptance Gates

- Rebuild every kept bridge/parser against installed DS9 headers and treat
  compiler/API errors as migration blockers.
- Run focused smoke tests for pose metadata, object depth metadata, depth tensor
  residency, ReID extraction, V3DT extraction, parser outputs, ROI pruning, and
  Service Maker tensor metadata before declaring DS9 production-ready.
- Keep fallbacks loud and opt-in during DS9 prep. A fallback that hides a broken
  canonical path is a bug.
