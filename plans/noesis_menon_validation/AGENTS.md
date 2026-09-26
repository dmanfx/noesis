# AGENTS.md — Noesis/Menon validation toolbox

This directory owns stable fixtures and concise guidance for direct validation
between the native DS9.1 producer and Menon consumers.

## Policy precedence

[Root policy](../../AGENTS.md) and [plan policy](../AGENTS.md) apply.

- Fixture files in this directory remain at their current paths because tests
  and validation scripts consume them directly.

## Rules

1. Pick the smallest tier in [validation_tiers.md](validation_tiers.md) that proves the changed
   producer, contract, and consumer.
2. Reuse fixture and saved-telemetry results when inputs did not change.
3. Attach live checks to the selected native service under the
   [runtime boundary](../../DS9/docs/runtime_host_boundary.md).
4. A health endpoint proves liveness, not world/tracking correctness.
5. Preserve explicit coordinate frames, camera IDs, run/sequence IDs,
   timestamps, transforms, and StableID semantics.
6. Treat unavailable unrelated evidence as out of scope, not a reason to widen
   the task.
7. Update generated schemas only when their source contract changes.

Documentation changes use the [root documentation checks](../../AGENTS.md#files-docs-and-commits).
