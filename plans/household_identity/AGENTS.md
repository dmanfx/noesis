# AGENTS.md — Household identity

This directory tracks the remaining StableID/household-identity acceptance work
for the canonical native DeepStream 9.1 application.

## Policy precedence

[Root policy](../../AGENTS.md) and [plan policy](../AGENTS.md) apply.
[work_order.md](work_order.md) tracks remaining acceptance work;
[contracts.md](contracts.md) and [decisions.md](decisions.md) define identity
behavior. Implementation constraints are routed from [reid/AGENTS.md](../../reid/AGENTS.md).

## Read by task

| Task | Read before changing or claiming the behavior |
| --- | --- |
| Orientation or locating a work item | [README.md](README.md) |
| Identity assignment, resident/visitor policy, or persistence | [contracts.md](contracts.md) and [decisions.md](decisions.md) |
| Overlap or exclusivity | [camera_topology.md](camera_topology.md) |
| Embedding or per-frame budgets | [performance.md](performance.md) |
| Enrollment, scorer, or authority acceptance | [work_order.md](work_order.md), [calibration_and_enrollment.md](calibration_and_enrollment.md), and [validation.md](validation.md) |
| Public wire fields | The affected [WebSocket](../../docs/api_contracts_ws.md), [REST](../../docs/api_contracts_rest.md), or [metadata](../../docs/metadata_contracts.md) contract |
| Selecting verification | [testing guide](../../docs/testing_guide.md) and the affected contract's tests |

## Work and validation

- Use the implementation paths and product constraints in the identity guidance;
  acceptance checklists do not authorize enabling identity authority.
- Update only the completed work-order item, after its actual implementation and
  focused validation. Keep outstanding evidence requirements explicit.
- Record an identity-specific decision in [decisions.md](decisions.md), and link
  that decision from [architecture_decisions.md](../../docs/architecture_decisions.md)
  when it affects repository architecture; avoid two competing copies.
- Use the root documentation checks for documentation-only changes. Runtime
  behavior changes use focused identity tests and one bounded recorded or live
  consumer smoke when practical; preserve the separate authority-acceptance gates.
