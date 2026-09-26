# Stable identity and ReID

The canonical DS9.1 lane produces 256-dimensional Swin Tiny embeddings through
the ReID SGIE and native metadata bridge. The
[DS9 hooks](../DS9/noesis/pipelines/hooks.py) retain a legacy
[StableID manager](stable_id_manager.py) call. The
[identity-v2 source-frame service](../noesis/identity_v2_service.py) replaces that
public assignment when authoritative mode is selected and admitted. The
configured process mode and authority artifacts determine live ownership of
`stable_id` output. The current hook does not pass the
manager's optional pose-matching inputs; see
[pose metadata and StableID routing](../docs/pose_stable_id_integration.md).

## Current rules

- `stable_id` is public identity; tracker IDs are transient diagnostics.
- Enrolled residents occupy a bounded closed-world roster.
- Unknown people remain visitors and use recyclable generation-safe slots.
- One stable identity cannot be active for incompatible people/cameras.
- Kitchen/Family Room is configured for conditional dual-camera identity
  permits, requiring current geometry, time, and appearance evidence under the
  [topology contract](../plans/household_identity/camera_topology.md) and
  [configured bounds](../config/camera_topology.yaml). Configuration does not
  prove live authority acceptance or enable AMC/MV3DT.
- Weak/provisional evidence does not mint a permanent resident identity.
- Runtime hot-path retention work is bounded and not performed per frame.

## Runtime path

- Engine/config: `DS9/pipelines/config_infer_secondary_reid_swin.ini` and the
  DS9.1 artifact realization.
- Native extraction: DS9 ReID metadata extension.
- Legacy manager: `reid/stable_id_manager.py`; authoritative source-frame
  resolver: `noesis/identity_v2_service.py`.
- Service/API: `noesis/identity_v2_service.py` and `noesis/server/reid_api.py`.
- Wiring: `DS9/noesis/ds9_runtime_core.py` and DS9 pipeline hooks.

The old OSNet/torchreid CPU-crop route is historical and is not a fallback.
Current contracts and remaining evidence gates are documented under
`plans/household_identity/`.
