# AGENTS.md — Plans and work records

## Policy precedence

- Root `AGENTS.md` applies.
- A nearer workstream `AGENTS.md` may add product-specific rules.
- `plans/archive/` is historical and non-normative.
- Active planning work belongs on the repository `DS9` branch. Do not switch
  to or commit on the historical `feature_DS8` branch without explicit user
  direction.

## Rules

1. Keep active plans limited to unfinished work against the canonical native
   DS9.1 application.
2. Move completed or superseded DS8, DS9.0, container, prototype, and migration
   plans to `plans/archive/`; do not use them as an execution checklist.
3. Update an active checkbox when its implementation and focused validation are
   complete, with one concise dated note.
4. Do not turn a plan into release machinery. Default to direct component and
   application validation described in `docs/testing_guide.md`.
5. Record durable architecture choices in `docs/architecture_decisions.md`, not
   only in an implementation worklog.
6. Do not edit archived plans except for an explicit historical correction.
