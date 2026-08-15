# AGENTS.md — Noesis/Menon validation toolbox

This directory owns stable fixtures and concise guidance for direct validation
between the native DS9.1 producer and Menon consumers.

## Policy precedence

- Root `AGENTS.md` and `plans/AGENTS.md` apply.
- The detailed DS8-era design diary is archived under
  `plans/archive/completed/noesis_menon_validation_ds8_docs/`.
- Fixture files in this directory remain at their current paths because tests
  and validation scripts consume them directly.

## Rules

1. Pick the smallest tier in `validation_tiers.md` that proves the changed
   producer, contract, and consumer.
2. Reuse fixture and saved-telemetry results when inputs did not change.
3. Use the native DS9.1 service for live checks; never launch
   `noesis/ds8_runtime.py` or a container fallback.
4. A health endpoint proves liveness, not world/tracking correctness.
5. Preserve explicit coordinate frames, camera IDs, run/sequence IDs,
   timestamps, transforms, and StableID semantics.
6. Do not stage releases, clone state, publish selectors, create candidate
   bundles, or run broad suites for normal validation work.
7. Treat unavailable unrelated evidence as out of scope, not a reason to widen
   the task.
8. Update generated schemas only when their source contract changes.

After documentation changes, run
`./scripts/check_agents_docs_consistency.py` and `git diff --check`.
