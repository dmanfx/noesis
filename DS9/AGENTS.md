# AGENTS.md — DeepStream 9.1 native runtime

`DS9/` owns the only executable DeepStream application in this repository.

## Policy precedence

- Root `AGENTS.md` applies first; this file narrows DS9.1 work.
- `DS9/docs/history/` is non-normative migration and experiment evidence.

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
`noesis/ds8_runtime.py`. Container scripts retained pending archival/removal are
not valid implementation or validation paths.

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

## Artifact and capability constraints

- Engines: external root selected by `NOESIS_DS9_ARTIFACT_ROOT`; declarations
  and provenance in `asset_manifest.yaml` and `asset_realization.json`.
- Native extension sources: `native/`; runtime binaries are installed into the
  native root/realized location selected by the supervisor.
- Parser/config sources: `pipelines/`; GStreamer plugins: `gst-plugins/`.
- Fail closed on missing or incompatible artifacts. Never fall back to archived
  engines or native libraries.
- AMC and MV3DT remain disabled under the geometry gate in root policy.

Record current DS9.1 decisions in `../docs/architecture_decisions.md` and dated
milestones in `../docs/upgrade_history.md`; put migration diaries under
`docs/history/`.
