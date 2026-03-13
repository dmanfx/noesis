# Reference Documentation

This folder is the live reference for the Noesis video stack. Historical plans
and worklogs live in `docs/history/`.

## Quick Links
- **Start here (DS8):** `DS8_README_FOR_AGENTS.md`
- **Current baselines & defaults:** `DS8_Baselines.md`
- **Pose-assisted StableID:** `DS8_pose_stable_id_integration.md`
- **APIs & contracts:** `DS8_api_contracts_ws.md`, `DS8_api_contracts_rest.md`,
  `DS8_metadata_contracts.md`, `Telemetry_Schema.md`
- **WebSocket overview:** `WebSocket_API.md`
- **Testing:** `DS8_testing_guide.md`
- **Codebase overview:** `CODEBASE_DESCRIPTION.md`
- **V3DT forensics & calibration:** `DS8_v3dt_forensics.md`
- **Depth / MapAnything:** `MapAnything_Depth.md`, `MapAnything_Heatmap_Viewer.md`, `DEPTH_STACK_FLOW_V2.md`, `depth_metadata.md`
- **Archived references:** see `history/` (archived)

## Navigation by Topic
- **Pipeline & flow:** `flow_diagram_high_level.md`,
  `flow_diagram_low_level.md`, `DS8_MIGRATION_KNOWLEDGE_BASE.md`
- **Config maps:** `Configuration_Map.md`, `CONVENTIONS.md`
- **Telemetry & integrations:** `WebSocket_API.md`, `Telemetry_Schema.md`,
  `Occupancy_Publishing.md`, `Integrations_Playbook.md`
- **ROI & analytics:** `Static_ROI_Exclusion.md`, `DS8_roi_editor.md`
- **Depth & geometry:** `DEPTH_STACK_FLOW_V2.md`, `depth_metadata.md`,
  `MapAnything_Heatmap_Viewer.md`, `Dynamic_Sensors.md`

## History Archive
- Full DS8 migration work orders, checklists, and iteration logs are mirrored
  under `docs/history/ds8/` (copy of `plans/DS8/`). Use those for provenance;
  the docs here are the canonical current guidance.
- Additional historical material also lives under `docs/history/`.

## Maintenance
- Keep live/reference guidance in `docs/`. Move dated logs or experiment notes to
  `docs/history/`.
- Add a “Deprecated / Stale” section to any doc that no longer matches the
  active stack.
- When you edit docs, append a one-line entry under “Doc changes” (date + what changed) for any **substantive** update: new features/processes/workflows, contract or baseline changes, behavior-affecting diagrams. Skip trivial typo/style fixes or purely cosmetic diagram tweaks.
- Status headers (“Status: validated as of …”) should only be updated when the doc has been re-validated against code/configs.

## Doc changes (2026-02-04)
- Added ReID alias REST contracts, guardrails, and limitations (`DS8_api_contracts_rest.md`).
- Logged the soft-merge alias design decision and updated DS8 hooks/runtime checklists (`plans/DS8/ds8_design_decisions.md`, `plans/DS8/ds8_migration_checklist_hooks.md`, `plans/DS8/ds8_migration_checklist_ds8_runtime.md`).

## Doc changes (2026-03-06)
- Documented the canonical `NOESIS.OBJECT_DEPTH` DS8 object user-meta contract and logged the prototype depth-fusion metadata decision/progress (`DS8_metadata_contracts.md`, `plans/DS8/ds8_design_decisions.md`, `plans/DS8/ds8_migration_checklist_hooks.md`).

## Doc changes (2026-03-07)
- Updated the DS8 seg+depth rebuild notes to document the deterministic linear depth->seg path, canonical frame-space fusion, internal aligned-depth cache, and the no-bbox-fallback object-depth contract (`DS8_metadata_contracts.md`, `plans/DS8/ds8_design_decisions.md`, `plans/DS8/ds8_migration_checklist_hooks.md`, `docs/README.md`).
- Standardized agent-facing no-fallback policy language across repo, DS8, planning, and docs quickstart guidance so agents must surface blockers instead of masking them with alternates (`AGENTS.md`, `noesis/AGENTS.md`, `plans/AGENTS.md`, `plans/menon_world_unification/AGENTS.md`, `plans/DS8/v3dt/AGENTS.md`, `docs/AGENTS.md`, `docs/DS8_README_FOR_AGENTS.md`).

## Doc changes (2026-02-22)
- Updated BEV/floorplan tracking + trails docs to match world-first `menon_scene` (scene units) behavior and frontend-owned smoothing/persistence (`DS8_Baselines.md`, `DS8_README_FOR_AGENTS.md`, `CODEBASE_DESCRIPTION.md`).
- Updated WS contracts for tracking world fields and `floorplan_response` payload shape (`DS8_api_contracts_ws.md`).
- Removed the operator-facing coordinate-toggle plan text to reflect the current single supported mode (`plans/bev_world_space_migration_plan.md`).

## Doc changes (2026-02-07)
- Removed remaining references to the deprecated mosaic JPEG/WebSocket path from active docs.

## Doc changes (2026-02-06)
- Standardized AGENTS policy precedence across repo/docs/plans scopes and clarified archive docs as non-normative (`AGENTS.md`, `docs/AGENTS.md`, `plans/AGENTS.md`, `noesis/AGENTS.md`, `docs/history/ds8/v3dt/AGENTS.md`).
- Reduced V3DT AGENTS duplication by keeping only subtree-specific constraints and locking recovery commands to baseline calibration defaults (`plans/DS8/v3dt/AGENTS.md`, `plans/DS8/v3dt/README.md`).
- Removed active-doc references to deprecated stack naming and deprecated runtime entrypoint terms; active docs are now DS8-only while historical material remains under `docs/history/`.

## Doc changes (2026-02-02)
- Refreshed WebSocket contracts (`DS8_api_contracts_ws.md`) and telemetry summary (`Telemetry_Schema.md`) to match current DS8 runtime output.
- Archived older references for WebSocket/telemetry/depth/occupancy and replaced with DS8 summaries (`WebSocket_API.md`, `depth_metadata.md`, `Occupancy_Publishing.md`, `MapAnything_Depth.md`).
- Updated baselines and overview docs for RTSP→WebRTC mosaic delivery, BEV frame defaults, and ReID/pose/MapAnything env toggles (`DS8_Baselines.md`, `DS8_README_FOR_AGENTS.md`, `CODEBASE_DESCRIPTION.md`).
- Clarified exclusion/ROI handling and MapAnything engine details (`Static_ROI_Exclusion.md`, `DS8_roi_editor.md`, `DS8_MIGRATION_KNOWLEDGE_BASE.md`).
