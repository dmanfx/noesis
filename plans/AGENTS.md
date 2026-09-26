# AGENTS.md — Plans and work records

## Policy precedence

[Root policy](../AGENTS.md) owns runtime, branch, archive, and validation rules.
Workstream instructions add only their product-specific planning requirements.

## Rules

1. Keep active plans limited to unfinished work against the canonical native
   DS9.1 application.
2. Move completed or superseded DS8, DS9.0, container, prototype, and migration
   plans to [archive/](archive/); do not use them as an execution checklist.
3. Update an active checkbox when its implementation and focused validation are
   complete, with one concise dated note.
4. Do not turn a plan into release machinery. Default to direct component and
   application validation described in [testing guide](../docs/testing_guide.md).
5. Record durable architecture choices in [architecture decisions](../docs/architecture_decisions.md), not
   only in an implementation worklog.
6. Do not edit archived plans except for an explicit historical correction.
