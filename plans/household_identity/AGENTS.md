# AGENTS.md — Household identity

This directory tracks the remaining StableID/household-identity acceptance work
for the canonical native DeepStream 9.1 application.

## Policy precedence

- Root `AGENTS.md` and `plans/AGENTS.md` apply.
- Historical DS8 implementation notes were moved to
  `plans/archive/completed/household_identity_ds8_implementation/` and are not
  runtime instructions.
- `work_order.md` is the active checklist; `decisions.md` and `contracts.md`
  define the current product behavior.

## Required reading

1. `README.md`
2. `work_order.md`
3. `decisions.md` and `contracts.md`
4. `camera_topology.md` for overlap or exclusivity work
5. `performance.md` before changing embedding or per-frame budgets
6. `validation.md` before claiming identity authority or acceptance
7. `docs/api_contracts_ws.md`, `docs/api_contracts_rest.md`, and
   `docs/testing_guide.md` for current application boundaries

## Product rules

- Residents use a bounded, enrolled identity space; visitors are ephemeral.
- Provisional evidence does not mint a permanent public identity.
- A stable identity may be co-visible only across an accepted overlap edge and
  only when fresh geometry and appearance evidence support the same person.
- No pressure-driven auto-merge or hidden identity fallback is allowed.
- Use Swin ReID tensor metadata from the canonical DS9.1 SGIE/native bridge.
  Do not add CPU crop-to-TorchReID extraction or raise embedding budgets without
  measuring the live three-camera path.
- Preserve `stable_id` as the public identity; tracker IDs remain process-local.

## Canonical implementation boundary

- Identity engine: `reid/stable_id_manager.py`
- Runtime construction: `DS9/noesis/ds9_runtime_core.py`
- Per-frame integration: `DS9/noesis/pipelines/hooks.py`
- REST service: `noesis/server/reid_api.py`
- ReID config: `DS9/pipelines/config_infer_secondary_reid_swin.ini`
- Pipeline selection: `DS9/config/infer.yaml`
- Camera topology: `config/camera_topology.yaml`

MV3DT is not the baseline identity path. The accepted Kitchen/Family Room lane
is a separate explicit runtime opt-in; do not use an MV3DT remap as an identity
solution or enable cross-camera overlap outside the geometry boundary recorded
in `camera_topology.md`.

## Work and validation

- Change the smallest producer/contract/consumer surface.
- Run focused identity tests and one bounded recorded or live identity smoke
  when runtime behavior changes.
- Do not stage releases, build candidates, or run broad suites for ordinary
  identity work.
- Update `work_order.md` only when an item is actually complete. Record durable
  design changes in both `decisions.md` and `docs/architecture_decisions.md`.
- After documentation changes, run
  `./scripts/check_agents_docs_consistency.py` and `git diff --check`.
