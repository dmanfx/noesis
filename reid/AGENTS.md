# AGENTS.md — Stable identity implementation

## Policy precedence

[Root policy](../AGENTS.md) applies. This file routes identity work in `reid/`
and identity integration in the adapter or shared services.

## Product rules and reading

- For assignment, resident/visitor policy, or persistence, read the
  [identity contracts](../plans/household_identity/contracts.md) and relevant
  [decisions](../plans/household_identity/decisions.md). Residents are enrolled;
  visitor identities are bounded and generational; provisional evidence does
  not mint permanent public identity. `stable_id` remains the public identity.
- Cross-camera activity requires the exact accepted overlap permit with fresh
  geometry and appearance evidence. Use the
  [topology contract](../plans/household_identity/camera_topology.md);
  identity topology does not enable AMC or MV3DT. No pressure-driven auto-merge
  or hidden identity fallback is allowed.
- Embeddings come from the selected Swin SGIE/native metadata bridge. Preserve
  exact frame/tracker joins. Do not add CPU crop-to-TorchReID extraction or raise
  embedding budgets without measuring the live three-camera path; read
  [performance guidance](../plans/household_identity/performance.md) for that work.
- For enrollment, scorer changes, or authority acceptance, read the
  [enrollment gates](../plans/household_identity/calibration_and_enrollment.md),
  [validation gates](../plans/household_identity/validation.md), and the affected
  [work-order item](../plans/household_identity/work_order.md). Fixtures alone
  do not authorize an authority cutover.

## Implementation and validation

- Manager: [stable_id_manager.py](stable_id_manager.py); construction:
  [DS9 runtime core](../DS9/noesis/ds9_runtime_core.py); per-frame integration:
  [DS9 hooks](../DS9/noesis/pipelines/hooks.py).
- Service boundary: [reid_api.py](../noesis/server/reid_api.py); selected SGIE:
  [Swin config](../DS9/pipelines/config_infer_secondary_reid_swin.ini).
- For wire changes, read the affected [REST](../docs/api_contracts_rest.md),
  [WebSocket](../docs/api_contracts_ws.md), or [metadata](../docs/metadata_contracts.md)
  contract. Keep identity state private and evidence exports free of embedding
  vectors under the identity persistence contract.
- Run focused identity tests and, when practical, one bounded recorded or live
  producer/consumer smoke for runtime behavior changes. Separate those checks
  from the authority-acceptance gates; docs-only edits use root docs checks.
