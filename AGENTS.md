# AGENTS.md — Noesis repository policy

Noesis runs one canonical perception stack: **DeepStream 9.1 directly on the
host**. `DS9/` owns the executable adapter, pipeline, configs, native bridges,
parsers, and model realization. Shared product contracts and services remain in
`noesis/` and `noesis_core/` and are imported by the DS9.1 adapter.

DeepStream 8 and DeepStream 9.0 are historical only. Their documentation lives
under `docs/history/` and `plans/archive/`; it is non-normative. Do not start,
repair, extend, or route failures through a DS8/9.0 runtime. Container-era DS9.1
documents are historical as well: the application and engine-maintenance path
are native-host only and must not require Docker.

## Policy precedence

- This file is the repository-wide baseline.
- A nearer `AGENTS.md` may add narrower rules for its subtree.
- Historical documents never override current guidance.
- Current architecture and operating facts are indexed by `docs/README.md`.

## Required DeepStream route

Before SDK-facing work, read the matching skill under `.agents/skills/` and the
references it routes to. Use `DS9/docs/deepstream_9_1_agent_skills.md` to choose:

- `deepstream-dev` for Service Maker, GStreamer, runtime, and SDK work;
- `deepstream-generate-pipeline` for a new graph or generated config;
- `deepstream-import-vision-model` for model and TensorRT work;
- `deepstream-profile-pipeline` for FPS, latency, or utilization work;
- `deepstream-run-mv3dt` only for an explicitly approved MV3DT task.

Repository pins override older vendor examples: DeepStream 9.1, CUDA 13.2,
TensorRT 10.16.0.72, GStreamer 1.24.2, Python 3.12, and driver 595.58.03 or
newer. Verify API names and config keys in the installed modules or official
documentation; never guess.

AMC remains deferred. MV3DT is an explicit runtime opt-in with one accepted
Kitchen/Family Room peer edge. Living Room participates only as a local stream;
Living Room/Family Room do not overlap, and Kitchen/Living Room are adjacency
only. Do not add another MV3DT peer edge without accepted geometry and occupied
overlap evidence for that exact pair.

## Canonical implementation boundary

- Runtime entrypoint: `DS9/noesis/ds9_runtime.py`.
- Runtime orchestration and product wiring: `DS9/noesis/ds9_runtime_core.py`.
- Pipeline implementation: `DS9/noesis/pipelines/`.
- Canonical pipeline config: `DS9/config/infer.yaml`.
- Native host supervisor: `DS9/scripts/run_canonical_runtime_host.py`.
- Native build and maintenance scripts: `DS9/scripts/*_host*` and
  `DS9/scripts/build_all_native_ds9.sh`.
- Shared contracts/services: `noesis/` and `noesis_core/`.

Some shared or mirrored modules retain `ds8` in a filename for source-history
reasons. A filename is not runtime authority: the canonical copy is the one
loaded by the DS9.1 entrypoint and documented in `DS9/docs/runtime_ownership.yaml`.
Do not rename these interfaces casually or use the old runtime to test them.

Do not introduce DeepStream 8/9.0 compatibility paths, container fallbacks,
alternate engines, substitute algorithms, or degraded modes without explicit
user approval. A canonical-path failure must be surfaced and fixed or reported.

## GPU and native-artifact rules

- Keep decode, preprocessing, inference, tracking, analytics, tiling, and OSD
  in NVMM/GPU memory wherever DeepStream supports it.
- CPU work is allowed only at explicit product boundaries such as JSON/REST
  serialization, BEV composition, and approved persistence or display work.
- Do not add analytics appsinks or raw-frame CPU branches.
- DS9.1 engines must be built with TensorRT 10.16.0.72. Native extensions,
  parsers, and GStreamer plugins must be built against the installed DS9.1/CUDA
  13.2 host stack.
- Never load binaries from a DS8, DS9.0, Docker overlay, or archived artifact
  root into the canonical process.
- Changes to native metadata plumbing require rebuilding only the affected
  binary and exercising its direct metadata consumer.

## Hot-path and performance invariants

The accepted runtime behavior is defined in
`docs/performance_invariants.md`. Preserve it for pipeline, native bridge,
tracking, identity, telemetry, persistence, dashboard, and media changes.

- The always-on media path must not wait for optional depth, evidence,
  persistence, reconstruction, external integration, or display work. Such
  work uses finite queues/workers and degrades its own freshness when full.
- Canonical `tracking`, `world_snapshot`, `world_event`, and `bev-frame`
  publications are the exception to latest-only handling: they remain one
  exact ordered cohort and may not be silently dropped, coalesced, or joined
  by last-seen state.
- Do not perform filesystem/database/network writes, process-wide CUDA
  synchronization, cold model/backend initialization, or unbounded waits in an
  always-on frame callback. A documented SDK-lifetime copy on an isolated,
  request-gated capture branch is allowed only when it is bounded and measured.
- Reuse bounded pools for device buffers, CUDA streams/events, pinned host
  memory, and scratch state. Extract a native tensor or surface once per frame
  and share the compact result instead of repeating conversion or copies.
- Put explicit limits on work amplified by frames, detections, tracks, cameras,
  clients, or retained evidence. Validate occupied scenes, not only empty-room
  throughput.
- Preserve selected models, input resolution, inference cadence, tracking
  quality, and enabled outputs before considering any quality/throughput
  tradeoff. Reducing capability requires explicit user approval and matched
  quality evidence.
- A zero-copy or performance claim requires end-to-end measurement of the
  native consumer and direct media path. Config flags or dashboard FPS alone
  are not proof; measure per-source progress, encoded access-unit cadence and
  drops, and WebRTC decoded frames separately.

## Development and validation fast path

Ordinary work uses direct application validation:

1. Reproduce the actual failure or incomplete behavior.
2. Change the smallest relevant implementation surface.
3. Run syntax/static checks for those files.
4. Run focused unit/component tests for the affected contracts.
5. Exercise the changed producer and direct consumer in one bounded live or
   recorded application smoke when practical.
6. Inspect concrete output (detections, tracks, identity, depth, world/BEV,
   media, latency, GPU/CPU use) only where the change can affect it.

Stop escalating when the behavior is proven and no evidence indicates a wider
regression. Do not rerun unchanged suites, rebuild unchanged artifacts, or turn
a typo/doc/comment change into application validation.

Normal development must not create or invoke immutable appliance releases,
state clones, deployment selectors, bundle inventories, candidate publication,
promotion ceremonies, rollback rehearsals, evidence sealing, or independent
reviewer lanes. Existing historical tooling does not make it the default.
Production promotion is allowed only when the user explicitly requests it or a
real external lifecycle boundary cannot be exercised directly, and even then
checks remain capability-scoped.

Do not recursively scan, hash, copy, or inventory large model, artifact,
recording, cache, build, staging, or runtime trees when an explicit path,
manifest, changed-file list, or existing result answers the question.

The task is complete when requested behavior works, directly affected tests
pass, the relevant integration path has been exercised where practical, and
known limitations are stated. Full-suite, long-soak, packaging, and formal
release proof are exceptional, not implicit completion criteria.

## Application contracts and product rules

- WebSocket contract: `docs/api_contracts_ws.md`.
- REST contract: `docs/api_contracts_rest.md`.
- Metadata contract: `docs/metadata_contracts.md`.
- Runtime and pipeline baseline: `docs/runtime_baseline.md` and
  `DS9/PIPELINE_GRAPH.md`.
- Testing guidance: `docs/testing_guide.md`.
- Architecture decisions: `docs/architecture_decisions.md`.

`stable_id` is the user-visible person identity. Tracker IDs are process-local
diagnostic values. World and BEV outputs must preserve explicit coordinate
frames, timestamps, source identity, and authority; do not infer or silently
mix coordinate spaces.

AMC and MV3DT are not enabled merely because code/config/assets exist. Static
Scene Prior/PCF evidence is presentation and reconstruction authority only where
its explicit camera/revision binding says so; it must not silently replace live
tracking/world authority.

## Files, docs, and commits

- Preserve unrelated dirty work. Stage by explicit path or hunk and keep
  commits coherent.
- Use portable repo-relative paths or environment variables; do not add
  machine-specific `/home/...` paths to code or docs.
- Record non-trivial current decisions in `docs/architecture_decisions.md`.
- Record completed upgrades and behavioral milestones in
  `docs/upgrade_history.md`.
- Put superseded implementation material under `docs/history/` or
  `plans/archive/` with a clear historical status; do not keep stale guidance in
  the active index.
- After changing agent guidance or documentation, run
  `./scripts/check_agents_docs_consistency.py` and `git diff --check`. Docs-only
  work does not trigger application, packaging, or release validation.
