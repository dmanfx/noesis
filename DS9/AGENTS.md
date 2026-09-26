# AGENTS.md — DeepStream 9.1 native runtime

`DS9/` owns the only executable DeepStream application in this repository.

## Policy precedence

[Root policy](../AGENTS.md) owns branch, runtime, artifact, task-scope, and
completion rules. This file adds DS9-specific implementation guidance.

## Runtime authority

- Native supervisor: [run_canonical_runtime_host.py](scripts/run_canonical_runtime_host.py).
- Entrypoint: [ds9_runtime.py](noesis/ds9_runtime.py); orchestration:
  [ds9_runtime_core.py](noesis/ds9_runtime_core.py).
- Selected graph/models/outputs: [infer.yaml](config/infer.yaml) and
  [runtime baseline](../docs/runtime_baseline.md). Verify observed runtime
  claims against the selected process and artifacts.
- Source ownership, including mirrored interfaces:
  [runtime_ownership.yaml](docs/runtime_ownership.yaml).

## Required workflow

1. Read the relevant NVIDIA skill and routed references using
   [skill routing](docs/deepstream_9_1_agent_skills.md) for SDK-facing changes.
2. Confirm APIs/config keys against the installed DS9.1 stack.
3. Change the smallest owned implementation surface.
4. Rebuild only affected native/parser/plugin/engine artifacts.
5. Run focused tests for affected contracts and, when practical, one bounded
   live or recorded smoke through the changed producer and direct consumer.

Documentation-only changes use the root documentation checks. Select other
checks from the [testing guide](../docs/testing_guide.md).

## Hot-path invariants

Read and preserve [performance invariants](../docs/performance_invariants.md) for any graph, callback,
native bridge, telemetry, persistence, or media change.

- The canonical core remains GPU/NVMM through GPU `nvdsosd` and NVENC. On the
  installed DS9.1 stack `nvdsosd process-mode=1` is GPU mode; do not copy the
  archived DS8 `process-mode=0` guidance.
- Keep the DAv2 secondary queue bounded and latest-frame-only, and keep its
  CUDA readiness query-only. Optional work may lose freshness; it may not
  back-pressure tracking, OSD, or encode.
- `disable-output-host-copy=1` is not sufficient evidence for a zero-copy
  claim. Inspect the native consumer and runtime counters for downstream copies
  and synchronization.
- Performance work includes a motion/occupancy-heavy recorded sample and,
  when practical, a bounded live check. Performance or zero-copy claims need
  separate source-progress, encoded-AU gap/drop, and WebRTC-decode evidence.

## Artifact and capability constraints

- Engine declarations: [asset_manifest.yaml](asset_manifest.yaml).
  Realized artifacts and `asset_realization.json` live under
  `NOESIS_DS9_ARTIFACT_ROOT`, resolved by the native supervisor.
- Native sources: [native/](native/) and [csrc/](csrc/); GStreamer plugins:
  [gst-plugins/](gst-plugins/). Runtime binary locations come from the supervisor.
- Parser/inference configs: [pipelines/](pipelines/); application config:
  [config/](config/); pipeline implementation: [noesis/pipelines/](noesis/pipelines/).
