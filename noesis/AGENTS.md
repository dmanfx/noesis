# AGENTS.md – `noesis/` (DS8 Stack)

This directory tree is the canonical DeepStream 8 / Service Maker / Flow implementation for the application.

## Policy precedence

- This file extends root `AGENTS.md` for the `noesis/` subtree.
- If a deeper subtree defines its own `AGENTS.md`, the nearest file overrides parent scope for that subtree.
- Historical files under `docs/history/` are archival and non-normative for active implementation work.

## Scope

- Use this tree for **DS8-only** work:
  - `noesis/ds8_runtime.py` – DS8 runtime harness.
  - `noesis/pipelines/*` – DS8 pipeline graph and metadata hooks.
  - `noesis/server/*` – DS8 REST APIs (depth, analytics, ReID aliases).
  - `noesis/telemetry/*` – BEV, depth, tracking publishers.
  - `noesis/metadata/*` – intrinsics and depth schemas.
  - `noesis/diagnostics/*` – DS8 forensics and telemetry diagnostics.
  - Related adapter modules live at repo root under `adapters/*` (not under `noesis/`).

## DS8-Specific Rules

1. **Do not use GI/GStreamer directly here**
   - Do not use GI/GStreamer to construct or introspect the DS8 Service Maker pipeline graph under `noesis/pipelines/*` or to implement metadata extraction hooks.
   - Exception: the encoded-mosaic delivery edge is intentionally outside the Service Maker analytics graph. `noesis/mosaic_h264_bridge.py` owns the single GI `shmsrc → h264parse → appsink` AU reader, and `noesis/mosaic_webrtc_gateway.py` owns each GI `appsrc → rtph264pay → webrtcbin` peer pipeline. `noesis/ds8_runtime.py` may wire those edge pipelines and the repo-owned mosaic force-IDR control. Do not extend this exception into analytics or raw-frame CPU branches.
   - All DS8 pipeline construction should use DeepStream 8 Service Maker APIs (`pyservicemaker`) and DeepStream Python bindings (`pyds`) where needed.

2. **No fallbacks without explicit user approval**
   - Do not add or activate fallback paths, degraded modes, substitute algorithms, alternate metadata sources, or legacy runtime routes unless the user explicitly asks for that fallback or agrees after you discuss the blocker.
   - If the canonical DS8 path fails, surface the failure and stop instead of masking it with a fallback.

3. **Prefer Service Maker / Flow APIs**
   - For pipeline construction, use the documented `Pipeline` API.
   - For data retrieval and gating, use documented Service Maker/Flow primitives such as `BufferRetriever`, `BufferOperator`, and batch metadata (`Buffer.batch_meta`).

4. **No deprecated-stack pad probes or analytics appsinks**
   - Do not add new pad probes or appsinks in DS8 code.
   - Metadata extraction should be done via DS8 batch metadata operators and Service Maker operators, not via GStreamer pad probes. The encoded-AU edge exception above is transport-only and must not inspect raw frames or analytics metadata.

5. **Doc-backed API usage only**
   - Before using or adding any Service Maker / DeepStream calls, verify them against:
     - The official DS8 documentation, or
     - The installed Python module documentation (e.g., `help(pyservicemaker.Pipeline)`, `help(pyds)`).
   - Do not introduce calls to undocumented attributes or methods.

6. **Keep behavior in sync with plans**
   - When modifying code under `noesis/`, always check the relevant DS8 planning documents in `plans/DS8/` and respect the master work orders.
   - If you need to deviate, update the appropriate plan/checklist and log the decision in `plans/DS8/ds8_design_decisions.md`.

7. **Logging and robustness**
   - Favor clear, actionable log messages when DS8 operations fail (e.g., pipeline build errors, hook attachment errors).
   - Handle missing DS8 libs gracefully in tests, but in production paths prefer failing fast over silently degrading into deprecated runtime behavior.

8. **Native metadata extension validation**
   - If changes affect V3DT or pose metadata extraction paths, rebuild and validate native bridges (`noesis_v3dt_meta_ext`, `noesis_pose_meta_ext`) before considering work complete.
   - Use focused smoke checks (for example `scripts/sv3dt_meta_smoke_test.py`) after extension-related changes.

9. **Portable paths only**
   - Do not introduce new hardcoded absolute machine-local paths in DS8 runtime code.
   - Use repo-relative resolution (`REPO_ROOT`) or explicit config/env inputs.

10. **Direct application validation**
   - Validate ordinary DS8 changes with the affected focused tests, direct
     contract consumers, and one practical runtime or recorded smoke.
   - Do not invoke DS9/Menon appliance staging, immutable candidate selectors,
     state cloning, bundle publication, or promotion ceremonies for DS8 work.
     Those mechanics belong only to an explicitly requested external release
     or service-lifecycle task.
