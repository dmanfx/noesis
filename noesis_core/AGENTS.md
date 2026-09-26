# AGENTS.md — Core world and product contracts

## Policy precedence

[Root policy](../AGENTS.md) applies. This subtree owns SDK-neutral contracts and
world logic; the native adapter and shared services remain separate consumers.

## Read by change

- World hypotheses, resolution, or temporal state:
  [universal world localization](../docs/universal_world_localization.md).
- Public fields, frames, revisions, or provenance:
  [metadata contracts](../docs/metadata_contracts.md) and the affected
  [WebSocket contract](../docs/api_contracts_ws.md).
- Scene Prior admission or camera binding:
  [Scene Prior contract](../docs/scene_prior_v1.md).
- Cross-view validation:
  [direct world/BEV checks](../docs/testing_guide.md#direct-physical-worldbev-check).

## Contract rules

1. Preserve explicit coordinate frame, units, source/camera identity, observation
   time, lifecycle generation, calibration/world revision, and transform identity.
   Fail closed when the required evidence or binding does not match.
2. Keep one canonical backend person-ground authority. Views apply their explicit
   revision-bound transforms; they do not invent coordinates or repair placement.
3. Preserve the distinction between current measurements, prediction, hold,
   diagnostics, and absent state. Prediction and hold are not fresh fusion evidence;
   lifecycle removal and trail breaks must reach the direct consumer.
4. Keep measurement resolution separate from the temporal PersonGroundState
   filter. Preserve covariance and contributor provenance through accepted fusion
   and transforms; correlated estimates are not independent corroboration.
5. Scene Prior and structural evidence may affect only their declared authority.
   They do not silently replace calibration or relocate live people.

## Validation

Use focused tests of the changed contract and its actual adapter/service consumer.
Cover relevant mismatches and stale/lifecycle inputs as well as the accepted case.
Follow the [testing guide](../docs/testing_guide.md) for a practical direct smoke;
physical placement claims need independent placement evidence. Documentation-only
changes use the root documentation checks.
