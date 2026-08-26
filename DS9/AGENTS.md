# AGENTS.md — DeepStream 9.1 native runtime

`DS9/` owns the only executable DeepStream application in this repository.

## Policy precedence

- Root `AGENTS.md` applies first; this file narrows DS9.1 work.
- Repository history under `../docs/history/` is non-normative migration and
  experiment evidence.
- Work in this subtree belongs on the repository `DS9` branch. Treat
  `feature_DS8` as read-only history, verify the current branch before a
  commit, and do not switch the shared checkout without explicit user
  direction.

## Runtime authority

- DeepStream 9.1.0 at `/opt/nvidia/deepstream/deepstream-9.1`.
- CUDA 13.2, TensorRT 10.16.0.72, GStreamer 1.24.2, Python 3.12.
- Native-host execution through `scripts/run_canonical_runtime_host.py`.
- Entrypoint `noesis/ds9_runtime.py`; core wiring
  `noesis/ds9_runtime_core.py`; config `config/infer.yaml`.
- Canonical lane: YOLO26-m, NvDCF baseline tracking, Swin ReID, YOLO26 pose,
  always-on DAv2 tracking depth, and gated MapAnything manual depth.
- WebRTC/SHM mosaic enabled; RTSP disabled.

Do not use Docker, a container supervisor, DS8/DS9.0 binaries, or
`noesis/ds8_runtime.py`. Those retired surfaces are not valid implementation
or validation paths.

## Required workflow

1. Read the relevant NVIDIA skill and routed references using
   `docs/deepstream_9_1_agent_skills.md`.
2. Confirm APIs/config keys against the installed DS9.1 stack.
3. Change the smallest owned implementation surface.
4. Rebuild only affected native/parser/plugin/engine artifacts.
5. Run focused tests and one bounded direct runtime or recorded smoke for the
   changed capability.

Do not create release candidates, selectors, state clones, bundles, promotion
gates, or broad validation runs for ordinary DS9.1 work.

## Hot-path invariants

Read and preserve `../docs/performance_invariants.md` for any graph, callback,
native bridge, telemetry, persistence, or media change.

- The canonical core remains GPU/NVMM through GPU `nvdsosd` and NVENC. On the
  installed DS9.1 stack `nvdsosd process-mode=1` is GPU mode; do not copy the
  archived DS8 `process-mode=0` guidance.
- Do not add full-frame host staging, `cudaDeviceSynchronize`, steady-state
  per-frame CUDA allocation, or blocking I/O to an always-on operator/callback.
- Keep the DAv2 secondary queue bounded and latest-frame-only, and keep its
  CUDA readiness query-only. Optional work may lose freshness; it may not
  back-pressure tracking, OSD, or encode.
- Do not apply latest-only behavior to canonical tracking/world/BEV cohorts.
  They retain exact order and revision identity through bounded admission.
- Reuse device/pinned-host pools and private nonblocking streams/events. Share
  one extraction/conversion among multiple consumers and bound work that grows
  with detections, tracks, cameras, clients, or retained evidence.
- `disable-output-host-copy=1` is not sufficient evidence for a zero-copy
  claim. Inspect the native consumer and runtime counters for downstream copies
  and synchronization.
- Do not reduce model size, resolution, inference interval, tracker quality, or
  enabled outputs as a performance fix unless the user explicitly accepts that
  tradeoff after a matched comparison.
- For hot-path changes, validate a motion/occupancy-heavy recorded input and a
  bounded live run. Measure source progress, encoded-AU gaps/drops, and WebRTC
  decode separately; do not use the dashboard's aggregate FPS as sole proof.

## Artifact and capability constraints

- Engines: external root selected by `NOESIS_DS9_ARTIFACT_ROOT`; declarations
  and provenance in `asset_manifest.yaml` and `asset_realization.json`.
- Native extension sources: `native/`; runtime binaries are installed into the
  native root/realized location selected by the supervisor.
- Parser/config sources: `pipelines/`; GStreamer plugins: `gst-plugins/`.
- Fail closed on missing or incompatible artifacts. Never fall back to archived
  engines or native libraries.
- AMC remains deferred. MV3DT is an explicit opt-in limited to the accepted
  Kitchen/Family Room peer edge; baseline tracking remains canonical.

Record current DS9.1 decisions in `../docs/architecture_decisions.md` and dated
milestones in `../docs/upgrade_history.md`; put migration diaries under the
repository `docs/history/` or `plans/archive/` tree according to content type.
