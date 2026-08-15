# Household identity — current decisions

Historical decision detail is preserved in
`plans/archive/completed/household_identity_ds8_implementation/decisions_pre_native_20260815.md`.
The decisions below govern the native DeepStream 9.1 application.

## D1 — Closed residents, bounded visitors

Enrolled residents use durable UUID-backed identities. Unknown people use
ephemeral visitor slots with generation and TTL semantics. Provisional tracks
do not acquire a permanent public identity.

## D2 — Exclusivity is the default

One identity may be active on more than one camera only for a configured
overlap pair and only with fresh geometry, time, and appearance evidence. No
appearance-only permit is allowed in household mode.

## D3 — Swin is the selected ReID model

`DS9/pipelines/config_infer_secondary_reid_swin.ini` is selected by
`DS9/config/infer.yaml`. SOLIDER is research-only unless measured Swin failures
justify a separate compare-first change. Retired OSNet/DS8 profiles are not
fallbacks.

## D4 — One process-owned identity service

The native DS9.1 runtime constructs one identity-v2 service per process and
resolves each complete source-frame cohort once. Shared product code lives in
`noesis/` and `reid/`; executable integration lives under `DS9/noesis/`.

## D5 — Public authority requires real evidence

Calibration authorizes scorer policy only. Public identity-v2 authority also
requires an exact native DS9.1 coordinator replay and occupied-scene report,
bound by `noesis.identity.authority_cutover` v1. Fixture evidence cannot clear
this gate.

## D6 — Names bind to resident UUIDs

Human names attach to durable resident records, not raw tracker IDs or visitor
slot numbers. `stable_id` remains the compatibility wire identity.

## D7 — Performance is a product contract

Identity uses SGIE tensor metadata and native extraction. Do not add a CPU
crop/TorchReID path, an appsink, unbounded per-frame embedding work, or a hidden
model fallback. A change must stay within the targets in `performance.md`.

## D8 — Geometry cannot create identity evidence

World/topology evidence may block or permit an otherwise admissible assignment;
it cannot manufacture appearance evidence or override a hard open-set reject.

## D9 — Human-reviewed enrollment and merge

Enrollment and alias merges are explicit owner actions. Pressure-driven or
background auto-merge remains disabled.

Record any new durable choice here and summarize cross-system consequences in
`docs/architecture_decisions.md`.
