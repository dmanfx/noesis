# Stable identity and ReID

The canonical DS9.1 lane produces 256-dimensional Swin Tiny embeddings through
the ReID SGIE and native metadata bridge. `reid/stable_id_manager.py` combines
them with track continuity, pose/quality evidence, resident/visitor policy,
camera topology, and persisted identity state to produce user-visible
`stable_id` values.

## Current rules

- `stable_id` is public identity; tracker IDs are transient diagnostics.
- Enrolled residents occupy a bounded closed-world roster.
- Unknown people remain visitors and use recyclable generation-safe slots.
- One stable identity cannot be active for incompatible people/cameras.
- Any future dual-camera permit requires accepted overlap geometry; MV3DT is
  currently disabled.
- Weak/provisional evidence does not mint a permanent resident identity.
- Runtime hot-path retention work is bounded and not performed per frame.

## Runtime path

- Engine/config: `DS9/pipelines/config_infer_secondary_reid_swin.ini` and the
  DS9.1 artifact realization.
- Native extraction: DS9 ReID metadata extension.
- Product manager: `reid/stable_id_manager.py`.
- Service/API: `noesis/identity_v2_service.py` and `noesis/server/reid_api.py`.
- Wiring: `DS9/noesis/ds9_runtime_core.py` and DS9 pipeline hooks.

The old OSNet/torchreid CPU-crop route is historical and is not a fallback.
Current contracts and remaining evidence gates are documented under
`plans/household_identity/`.
