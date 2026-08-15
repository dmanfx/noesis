# Household identity — current contracts

Exact public field definitions live in `docs/api_contracts_ws.md`,
`docs/api_contracts_rest.md`, and `docs/metadata_contracts.md`. This file records
identity-specific invariants.

## Public identities

| State | Meaning | Public behavior |
| --- | --- | --- |
| `provisional` | Evidence is incomplete | No permanent resident/visitor identity |
| `visitor` | Confirmed unknown person | Bounded TTL visitor identity |
| `resident` | Enrolled household member | Durable UUID/name plus compatibility `stable_id` |
| `handoff` | Cross-camera continuity is resolving | Preserve the accepted identity only while constraints hold |

`stable_id` is public identity. `tracker_id` is process-local and must not be
used as a household identity.

## Evidence and assignment

- A source-frame cohort is resolved once after tracker, ReID, pose, and world
  primitives are available.
- Authoritative output is one-to-one unless an exact overlap permit is present.
- Missing evidence is provisional; contradictory admissible evidence resolves
  to unknown. Neither case may silently reuse legacy StableID output.
- OSD labels join only an exact camera/frame/tracker decision downstream of
  resolution. A miss remains neutral.

## Topology

`config/camera_topology.yaml` maps the source order in `DS9/config/infer.yaml`.
The configured Kitchen ↔ Family Room edge is an identity constraint only; it
does not enable MV3DT or AMC. Missing/invalid topology or world evidence denies
dual-camera activity.

## Persistence and privacy

Identity state is owner-private under the configured Noesis state root.
Evidence exports contain scores and provenance, never query or gallery vectors.
Residents are durable; visitors are generational; old open-world gallery state
is archive input only and is never silently loaded as current authority.

## Authority artifacts

- `noesis.identity.open_set_calibration` v2 binds the exact model semantic
  profile, benchmark dataset, household dataset, policy, and metrics.
- `noesis.identity.authority_cutover` v1 separately binds scorer bytes, the
  native DS9.1 executable profile, topology, cameras, coordinator replay, and
  occupied-scene report.
- The schema retains `ds8` as a historical enum value. New authority artifacts
  for this application must use `runtime: ds9`; DS8 evidence is ineligible.
