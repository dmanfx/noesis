# AGENTS.md (Root) – Guidance for Codex Agents

This repository is centered on the DeepStream 8 stack:

- **DS8 (canonical):** `noesis/` tree (`ds8_runtime.py`, `pipelines/`, `server/`, `telemetry/`, `metadata/`, `config/`) built on DeepStream 8 Service Maker and Flow APIs.
- **Pre-DS8 artifacts (deprecated):** retained only for historical context or explicit one-off maintenance.

## Policy precedence

- This file is the canonical repo-wide policy baseline.
- A nearer `AGENTS.md` in a child directory may add narrower rules and overrides this file for that subtree.
- Files under `docs/history/` are archival and non-normative for active implementation work.

When you (the agent) work in this repo:

1. **Respect the deprecated-stack vs DS8 split**
   - Do not silently route DS8 failures through removed pre-DS8 runtime paths. If DS8 code cannot be made to work, fail loudly in logs/docs and stop.
   - Only modify deprecated pre-DS8 artifacts if the user explicitly asks for those changes.
   - Prefer implementing new functionality in the DS8 stack under `noesis/`.
   - Launch and validate runtime behavior through `noesis/ds8_runtime.py` for active work.

2. **No fallbacks without explicit user approval**
   - Do not introduce, enable, or rely on fallback paths, degraded modes, substitute workflows, legacy routes, or "temporary" alternates unless the user explicitly asks for that fallback or agrees after you surface the blocker.
   - If the canonical path fails, report the failure clearly, explain what is blocked, and stop instead of masking the problem with a fallback.
   - Treat hidden fallbacks as bugs because they can conceal the real defect or misconfiguration.

3. **Follow the DS8 plan and checklists**
   - Before changing DS8-related code, read:
     - `plans/DS8/ds8_master_work_orders.md`
     - The relevant checklist(s) under `plans/DS8/` (e.g., `ds8_migration_checklist_ds8_pipeline.md`, `ds8_migration_checklist_hooks.md`).
   - Treat these documents as the authoritative description of what to do and in what order. If code and docs disagree, surface the discrepancy in a doc update rather than guessing.

4. **No guessing of DeepStream/Service Maker APIs**
   - For any call into DeepStream 8 or Service Maker (e.g., `pyservicemaker`, `pyds`, `nvinfer`, `nvdsanalytics`):
     - Use only methods, properties, and config keys that you can verify in the official docs or the installed Python modules.
     - Do **not** invent new method names or config keys.
   - The main DS8 docs to consult (non-exhaustive):
     - `docs/DS8_README_FOR_AGENTS.md` in this repo (local orientation + links).
     - Service Maker for Python docs (Pipeline APIs, Flow APIs, Advanced Features).
     - Inference Builder docs.
     - DS8 analytics / nvdsanalytics YAML docs.

5. **GPU-first policy for DS8**
   - Keep decode, pre-processing, inference, tracking, analytics, tiling, and OSD on the GPU (NVMM) wherever DeepStream supports it.
   - Convert to CPU only at the very edge when required (e.g., final JPEG encoding if not handled by `nvjpegenc`, BEV composition, or explicit ReID crops).
   - Do not add new appsink/CPU-based branches to the DS8 path; those patterns belong only to deprecated pre-DS8 code.

6. **Testing and validation**
   - When you make DS8 changes, use `docs/DS8_testing_guide.md` as your reference for how to validate behavior (unit tests, functional tests, and DS8 regression checks).
   - Prefer small, focused validations (e.g., “does analytics ROI reload work?”) rather than only end-to-end tests.
   - For interface/contract changes, also consult:
     - `docs/DS8_api_contracts_ws.md` (WebSocket message contracts).
     - `docs/DS8_api_contracts_rest.md` (REST contracts, including ReID alias endpoints from `noesis/server/reid_api.py`).
     - `docs/DS8_metadata_contracts.md` (metadata/user-meta schemas).

7. **Progress documentation**
   - When completing work defined in `plans/DS8/*.md`:
     - Update the relevant checklist item from `[ ]` to `[x]`.
     - Add a short, dated note under that item describing what you changed and how you validated it.
   - Do this **often**: after each logically separate task or small cluster of related edits, so the plans reflect near real-time progress.
   - Do **not** add progress notes into code comments unless the user explicitly asks; keep them in the planning docs.

8. **Design decisions**
   - When you make a non-trivial design choice for DS8 (API shape, data schema, etc.), capture it in `plans/DS8/ds8_design_decisions.md` with a short rationale and any doc references you relied on.

9. **Docs and instruction hygiene**
   - When editing docs, AGENTS files, plans, or prompts, verify any referenced local file paths and commands still exist in this workspace before finalizing.
   - After changing `AGENTS.md` or files under `docs/`, run `./scripts/check_agents_docs_consistency.py` and fix failures before closing the task.

10. **Native extension discipline**
   - If you change V3DT/pose metadata plumbing or contracts (`noesis/pipelines/hooks.py`, `noesis/metadata/*`, `native/*`), rebuild the relevant native extension(s) and run focused smoke tests (for example `scripts/sv3dt_meta_smoke_test.py`) to confirm metadata extraction still works.

11. **Portable path requirement**
   - Do not add new hardcoded machine-local absolute paths (for example `/home/...`) in runtime code, docs, or scripts.
   - Prefer `REPO_ROOT`, config paths, or environment variables for path resolution.

These rules are meant to make your work predictable and auditable for future Codex runs.
