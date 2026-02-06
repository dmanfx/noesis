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
  - `noesis/server/*` – DS8 REST APIs (depth, analytics).
  - `noesis/telemetry/*` – BEV, depth, tracking publishers.
  - `noesis/metadata/*` – intrinsics and depth schemas.
  - `noesis/config/*` – DS8 config adapters.

## DS8-Specific Rules

1. **Do not use GI/GStreamer directly here**
   - Do not use GI/GStreamer to construct or introspect the DS8 Service Maker pipeline graph under `noesis/pipelines/*` or to implement metadata extraction hooks.
   - Exception: the RTSP→WebRTC mosaic gateway (`noesis/mosaic_webrtc_gateway.py`) is an intentional, separate GStreamer pipeline (GI) used only for mosaic delivery; `noesis/ds8_runtime.py` may also use GI for that gateway and RTSP keyframe requests.
   - All DS8 pipeline construction should use DeepStream 8 Service Maker APIs (`pyservicemaker`) and DeepStream Python bindings (`pyds`) where needed.

2. **Prefer Service Maker / Flow APIs**
   - For pipeline construction, use the documented `Pipeline` API.
   - For data retrieval and gating, use documented Service Maker/Flow primitives such as `BufferRetriever`, `BufferOperator`, and batch metadata (`Buffer.batch_meta`).

3. **No deprecated-stack pad probes or appsinks**
   - Do not add new pad probes or appsinks in DS8 code.
   - Metadata extraction should be done via DS8 batch metadata operators and Service Maker operators, not via GStreamer pad probes.

4. **Doc-backed API usage only**
   - Before using or adding any Service Maker / DeepStream calls, verify them against:
     - The official DS8 documentation, or
     - The installed Python module documentation (e.g., `help(pyservicemaker.Pipeline)`, `help(pyds)`).
   - Do not introduce calls to undocumented attributes or methods.

5. **Keep behavior in sync with plans**
   - When modifying code under `noesis/`, always check the relevant DS8 planning documents in `plans/DS8/` and respect the master work orders.
   - If you need to deviate, update the appropriate plan/checklist and log the decision in `plans/DS8/ds8_design_decisions.md`.

6. **Logging and robustness**
   - Favor clear, actionable log messages when DS8 operations fail (e.g., pipeline build errors, hook attachment errors).
   - Handle missing DS8 libs gracefully in tests, but in production paths prefer failing fast over silently degrading into deprecated runtime behavior.
