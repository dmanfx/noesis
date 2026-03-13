# AGENTS.md – `plans/` (Planning & Progress Docs)

This directory contains planning documents, work orders, and checklists. It is the primary place for Codex agents to record DS8 migration progress.

## Policy precedence

- This file extends root `AGENTS.md` for the `plans/` subtree.
- For nested planning scopes (for example `plans/DS8/v3dt/AGENTS.md`), the nearest `AGENTS.md` overrides this file.
- Archived planning folders are non-normative unless the user explicitly asks for historical maintenance.

## Usage Rules

1. **Authoritative task definitions**
   - Treat documents under `plans/DS8/` as the authoritative definition of DS8 work items and execution order.
   - Before changing DS8 code, read:
     - `plans/DS8/ds8_master_work_orders.md`
     - Any relevant checklist files (e.g., `ds8_migration_checklist_ds8_pipeline.md`, `ds8_migration_checklist_hooks.md`).

2. **Recording progress**
   - When you complete an item in a checklist or work order:
     - Change the checkbox from `[ ]` to `[x]`.
     - Immediately beneath the item, add a one-line note with date and a short description, for example:
       - `_2025-03-10 (Codex): Implemented DS8 MapAnything SGIE branch and validated tensor meta decoding on test stream X._`
   - Update checklists **frequently** (ideally after each small, coherent unit of work), rather than batching changes at the end of a session.
   - Keep these notes concise and focused on what changed and how it was validated.

3. **No fallback workarounds without explicit user approval**
   - Do not treat a fallback path, degraded mode, alternate workflow, or "temporary" substitute as an acceptable way to complete planned work unless the user explicitly asks for it or agrees after you discuss the blocker.
   - If the intended path is blocked, record the blocker clearly in the plan docs instead of silently substituting a fallback.

4. **Do not edit historical archives**
   - Files under `plans/archive/` and `plans/DS8/archive/` are historical and should not be edited unless explicitly requested.
   - New planning and progress should go into non-archive files.

5. **No code in planning docs**
   - Avoid embedding large code blocks in planning documents.
   - If you need to reference code, prefer short snippets or file/line pointers.

6. **Design decisions**
   - For non-trivial design choices (new APIs, schema changes, behavioral changes), add an entry to `plans/DS8/ds8_design_decisions.md` instead of burying the rationale in a comment.
