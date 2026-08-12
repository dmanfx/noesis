# Quickstart for Agents

Audience: Codex/ChatGPT agents and humans looking for the fastest path to the
right doc. Read this before touching code.

## Policy precedence
- This file is a docs quickstart index, not the canonical global policy source.
- Global policy lives in `AGENTS.md` at repo root.
- For any subtree, the nearest `AGENTS.md` takes precedence over parent scope.
- Files under `docs/history/` are archival and non-normative for active implementation work.

## Ground Rules
- **DS8 is canonical.** Only change deprecated pre-DS8 artifacts if a user
  explicitly asks. Keep GPU-first; no new CPU appsink branches in DS8.
- **No fallbacks unless the user explicitly approves them.** Do not use fallback
  paths, degraded modes, substitute workflows, or "temporary" alternates to
  hide a blocker. Surface the failure and discuss it instead.
- **Follow the DS8 plans/checklists.** If code and docs disagree, surface the
  discrepancy and update docs rather than guessing.
- **User-facing explanations should be semantic first.** Prefer describing behavior,
  variables, and outcomes in plain language; only include file/line citations when
  the user explicitly asks for code references.
- **Stable IDs:** `stable_id` is the only user-visible person ID. `track_id` is
  internal. BEV uses `stable_id` as the user-visible identity; BEV may also
  carry `trackerId` as a debug/fallback identity key (not stable across restarts).

## Navigation (start here)
- **DS8 orientation:** `DS8_README_FOR_AGENTS.md`
- **API contracts:** `DS8_api_contracts_ws.md`, `DS8_api_contracts_rest.md`,
  `DS8_metadata_contracts.md`
- **Baselines & current decisions:** `DS8_Baselines.md` (latest defaults, trail &
  ID rules, detector profiles, MapAnything gating, V3DT baseline)
- **Testing:** `DS8_testing_guide.md`
- **Codebase overview:** `CODEBASE_DESCRIPTION.md`
- **Design decisions archive:** `DS8_MIGRATION_KNOWLEDGE_BASE.md` and
  `plans/DS8/ds8_design_decisions.md` (also mirrored in `history/ds8/`)
- **V3DT forensics & tracker tuning:** `DS8_v3dt_forensics.md`,
  `DS8_Baselines.md` → V3DT section, `history/ds8/v3dt/` for iteration logs.

## Historical Docs
- Full migration work orders, checklists, and dated iteration notes are mirrored
  under `docs/history/ds8/` (copied from `plans/DS8/`). Use these for provenance
  and “why was this choice made?” trails.
- Older stack material is archived under `docs/history/`.

## When Writing New Docs
- Keep normative guidance in `docs/` (live), push long-form logs to
  `docs/history/`.
- Add a “Deprecated / Stale” section to any doc that no longer reflects the
  active stack.
- Whenever you change docs, add a dated one-liner under the “Doc changes” block in
  `docs/README.md` for substantive updates (new features/processes/workflows,
  contract/baseline changes, behavior-affecting diagrams). Skip trivial typo/style
  fixes and purely cosmetic diagram tweaks.
- Only update per-doc “Status: validated as of …” headers after re-validating the
  doc against live code/configs.
- Verify that command snippets and referenced local file paths still exist in this
  workspace before finalizing edits.
- After AGENTS/docs changes, run `./scripts/check_agents_docs_consistency.py` and
  resolve any failures before closing the task.
- Documentation and guidance changes use the docs-consistency check and
  `git diff --check` only; do not trigger application, packaging, appliance
  staging, or release validation unless the documentation change also changes
  an executable contract and that contract is explicitly being tested.

## If You’re Stuck
- Confirm you are working on the DS8 canonical stack (not deprecated paths).
- Check env flags in `DS8_Baselines.md` and `DS8_README_FOR_AGENTS.md`.
- Re-read `DS8_MIGRATION_KNOWLEDGE_BASE.md` for constraints (mosaic via WebRTC,
  MapAnything full-frame, no deprecated-stack fallbacks).
