# Household Identity — Contracts

Additive contract changes for WS/REST/metadata. Keep backward compatible where
possible; document removals explicitly.

## Identity states

| `identity_state` | Meaning | Public ID behavior |
|------------------|---------|-------------------|
| `provisional` | Confirming; weak/no emb | May omit permanent ID or use ephemeral display only |
| `visitor` | Unknown person, TTL slot | Visitor numeric ID |
| `resident` | Enrolled household member | Resident numeric ID |
| `handoff` | Cross-cam continuity in progress | Keep prior ID |

## WebSocket `tracking` track object (additive)

Existing fields remain (`stable_id`, `tracker_id`, `id_event`, `id_reject_reason`,
`embedding_present`, `pose_present`, `sid_candidate`).

Add:

| Field | Type | Notes |
|-------|------|-------|
| `identity_state` | string | See table above |
| `identity_kind` | string | `resident` \| `visitor` \| `provisional` |
| `reid_confidence` | float \| null | Best match score used for assignment |
| `reid_required` | float \| null | Threshold applied |
| `overlap_permit` | bool | True if dual-cam activity allowed via topology |
| `resident_uuid` | string \| null | Enrollment UUID when resident |
| `display_name` | string \| null | Human alias if set |

Update `docs/DS8_api_contracts_ws.md` when implemented.

## BEV `footpoints`

Keep `stableId` as the public numeric ID. Optionally mirror `identity_kind` and
`display_name` if cheap; do not require for Phase 0.

## REST (`noesis/server/reid_api.py`)

### Keep
- list/suggest/set/unset alias endpoints (suggest-only by default in household mode)

### Add (Phase 3; stub OK in Phase 1)
- `GET /reid/residents` — enrolled residents
- `POST /reid/residents/enroll` — bind current track/SID → resident
- `PATCH /reid/residents/{uuid}` — set `display_name`, gallery refresh
- `DELETE /reid/residents/{uuid}` — soft-delete / archive
- `GET /reid/identity_health` — mint rate, false-share counters, gallery sizes

Update `docs/DS8_api_contracts_rest.md` when implemented.

## Camera topology config

New file (portable path): `config/camera_topology.yaml`

```yaml
version: 1
cameras:
  kitchen:
    source_id: 0          # must match runtime source indexing
  family-room:
    source_id: 1
  living-room:
    source_id: 2
overlaps:
  - cameras: [kitchen, family-room]
    max_world_dist_m: 1.25
    max_time_delta_s: 0.35
    require_appearance_sim: 0.60   # soft; geometry is primary
    enabled: true
```

Exact `source_id` values must be verified against the active infer/sources config
during Phase 0 implementation — do not hardcode machine-local URIs.

## Persistence files (household mode)

| File | Role |
|------|------|
| `~/.noesis/household/residents.json` | Enrollment records + names + UUIDs |
| `~/.noesis/household/resident_gallery.npz` | Quality-gated resident embeddings |
| `~/.noesis/household/visitor_gallery.npz` | Ephemeral visitor embeddings |
| `~/.noesis/household/sid_pool.json` | Free visitor slots |
| `~/.noesis/household/backups/` | Archived pre-cutover state |

Legacy `~/.noesis/reid_gallery.npz` / `reid_aliases.json` are archived on cutover
(see D8), not used as primary store in household mode.

## Metrics (`get_sid_metrics` extensions)

Add counters:
- `false_share_blocked_count`
- `overlap_permit_grant_count`
- `overlap_permit_deny_count`
- `provisional_count`
- `resident_count` / `visitor_count`
- `mint_visitor_count` / `promote_resident_count`
- `assignment_conflict_count`

## Identity-v2 calibration authority

The generated product schemas are authoritative for exact fields:

- `noesis.identity.evidence_labels` v2 declares `benchmark` or `household`,
  immutable source/labeling provenance, a private `truth_person_key`, and one
  physical `encounter_id` for every score event. Version 1 labels reject.
- `noesis.identity.calibration_dataset` v2 embeds the exact chained score
  evidence plus deterministic evidence units keyed by session, run, source,
  camera, tracker, and five-second bucket. Units are capped at 300 observations;
  train fitting records one center representative for at most eight temporally
  spread units per encounter.
- `noesis.identity.open_set_calibration` v2 binds one benchmark dataset and one
  household dataset for the same engine/layer/dimension and runtime-derived
  semantic profile. It carries scorer-only authority scope, a conservative
  gallery envelope, the benchmark and household-tightened policies,
  person/encounter-balanced fit, four metric blocks, and acceptance gates.
  Version 1 artifacts reject rather than being interpreted under v2 semantics.

Each block reports raw observation/unit/encounter/person counts and
encounter-worst-case outcomes. Benchmark confidence is truth-person-worst-case
over subject-disjoint, challenge-covered resident and unknown people; repeated
encounters never increase its exact one-sided 95% denominator. Household gates
remain physical-encounter checks and do not claim the generic 1% result.
Household evidence can only retain or raise rejection gates; resident
prevalence remains bounded post-gate ranking utility and cannot rescue a
rejection.

`noesis.identity.authority_cutover` v1 is a separate public-runtime contract.
It binds one exact scorer artifact to the active DS8/DS9 model semantic profile,
executable authority-runtime profile, camera topology, and camera set. Its
`coordinator_replay` and `occupied_scene` members are distinct literal-pass
evidence records with owner-private report path, byte size, SHA-256, revision,
and completion time. The cutover artifact itself requires an independent exact-
byte environment pin, and runtime startup re-reads both report files. A scorer
artifact never satisfies this contract and cannot by itself enable public
authoritative identity.
