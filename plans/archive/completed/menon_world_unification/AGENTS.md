# AGENTS.md - plans/menon_world_unification

This file defines the local working contract for Menon world unification execution work.

## Scope and precedence

- Scope: everything under `plans/menon_world_unification/`.
- This file extends parent policy in:
  - `Noesis_Devel/AGENTS.md`
  - `Noesis_Devel/plans/AGENTS.md`

## Required read order for each session

1. `AGENTS.md`
2. `README.md`
3. `work_order.md`
4. `contracts.md`
5. `decisions.md`
6. `validation_matrix.md`
7. `handoff.md`

## Session protocol

### Start of session

- Add one timestamped line in `timeline.md` with session intent.
- Add one timestamped line in `session_notes.md` with immediate execution plan.
- Confirm active checklist item in `work_order.md`.

### During session

- Update `work_order.md` checkboxes immediately when an item is completed.
- Add a one-line dated validation note below each completed item.
- If interface behavior changes, update `contracts.md` in the same session.
- If a non-trivial technical decision is made, record it in `decisions.md`.
- Do not switch to fallback paths, degraded modes, or substitute workflows unless the user explicitly asks for that fallback or agrees after you surface the blocker; record the blocker instead of masking it.

### End of session

- Add a timestamped completion line in `timeline.md`.
- Update `handoff.md` with:
  - current status
  - validated evidence
  - open risks
  - next three concrete actions

## Evidence requirements

- Every completion note must include:
  - command or verification method
  - relevant file paths
  - pass/fail outcome
- Keep evidence concise and reproducible.
