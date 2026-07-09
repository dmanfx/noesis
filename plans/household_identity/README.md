# Household Identity Rework

Status: planning complete 2026-07-08; implementation in progress.

Transform Noesis StableID from an open-world ReID minting engine into a
**closed-world household identity system** with geometry-aware multi-camera
association, quality-gated galleries, and a stronger ReID backbone — while
keeping the DS8 path GPU-first and low-latency.

## Problem statement

Current StableID does a decent job of cross-camera ReID, but:

1. The same SID can appear on two cameras for **different** people (false share).
2. Kitchen ↔ family-room FoV overlap means the **same** person can legitimately
   appear on both cameras at once — exclusivity must allow that case.
3. SID values climb into the thousands because the system mints freely, then
   auto-merges aliases as cleanup. Live state showed gallery SID ~4132 and
   504 alias mappings (222 collapsing into SID 358).
4. Pose fusion exists in the manager but is not wired from hooks.
5. Soft `max_total_ids` does not create a true closed resident set.

User goal: consistently ID ~4 household residents day-to-day, with ≤10–15 total
people over short windows, then alias residents to human names.

## Doctrine

```text
tracklet (sensor, tracker_id)
  + ReID embedding (SGIE, zero-copy tensor meta)
  + pose features (optional boost)
  + world/BEV footpoint (geometry)
        │
        ▼
  Identity Resolver
        │
        ├── resident gallery (enrolled, stable, named)
        ├── visitor gallery (ephemeral, TTL)
        └── overlap permit (camera topology + world distance)
        │
        ▼
  public identity: resident_id | visitor_id | provisional
```

## Workstream files

| File | Role |
|------|------|
| `AGENTS.md` | Agent operating rules |
| `README.md` | This overview |
| `work_order.md` | Ordered checklist + gates |
| `decisions.md` | Locked design decisions |
| `contracts.md` | WS/REST/schema contracts |
| `performance.md` | Zero-copy / latency budgets |
| `validation.md` | Acceptance tests and metrics |
| `camera_topology.md` | Overlap graph for kitchen/family-room |
| `phase0_stop_the_bleeding.md` | Exclusivity, pose wire, reset, telemetry |
| `phase1_closed_world.md` | Resident/visitor spaces + mutex |
| `phase2_sota_matching.md` | Backbone, quality gallery, assignment |
| `phase3_enrollment_product.md` | Names, enrollment UX, health reports |

## Phase summary

### Phase 0 — Stop the bleeding
Archive poisoned gallery/aliases; disable pressure auto-merge; enforce
geometry-aware exclusivity (overlap exception); wire pose into `update()`;
raise embedding budget carefully; emit confidence/reject telemetry.

### Phase 1 — Closed-world identity
Split resident vs visitor ID spaces; provisional public suppression; hard
global mutex with overlap permits; retire reliance on `_public_stable_id`
band-aids.

### Phase 2 — SOTA matching
Default/harden Swin-Tiny ReID (already in `config/infer.yaml`); optional
SOLIDER upgrade path; quality-gated gallery updates; multi-exemplar outfit
clusters; Hungarian/MNN assignment; spatiotemporal handoff using topology+BEV.

### Phase 3 — Product layer
Enrollment API + UI; human name aliases bound to resident UUID; suggest-only
merges; nightly identity health report.

## Related code (current)

- `reid/stable_id_manager.py` — identity engine
- `noesis/pipelines/hooks.py` / `hooks_v3dt_reimpl.py` — embedding extract + assign
- `noesis/ds8_runtime.py` — manager construction / env knobs
- `noesis/server/reid_api.py` — alias REST
- `pipelines/config_infer_secondary_reid_swin.ini` — default ReID SGIE
- `pipelines/config_infer_secondary_reid_osnet.ini` — legacy OSNet SGIE
- `native/noesis_reid_meta_ext.cpp` — zero-copy tensor unwrap

## Success metric (14-day home validation)

- Resident SID space ≤ enrolled count
- Visitor SIDs ≤ ~20 with recycle
- Zero false same-SID shares across cameras for different people
- True overlap co-visibility keeps the same SID
- Cross-camera handoff recall for residents >90% within 30s
- False merge rate ≈ 0 under manual audit
