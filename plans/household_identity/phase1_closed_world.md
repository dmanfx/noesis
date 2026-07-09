# Phase 1 — Closed-World Identity

**Goal:** Make public identity a small resident set + ephemeral visitors, with
provisional suppression and hard global mutex (overlap permits from Phase 0).

**Depends on:** Phase 0 gate G0

## Scope

### In
- `identity_kind`: resident | visitor | provisional
- Separate persistence for resident vs visitor galleries
- Visitor TTL recycle (target ≤ ~16–32 slots)
- Confirmation gates before visitor mint / resident bind
- Reduce/retire V3DT `_public_stable_id` remapping once real IDs are small
- Metrics for mint/promote/recycle

### Out
- Full enrollment UI (Phase 3) — CLI/REST stub OK
- Outfit clustering / Hungarian (Phase 2)

## Technical design

### ID spaces

Option A (preferred): typed public IDs
- Residents: `1..R_max` (R_max = enrolled count, sticky)
- Visitors: `1000..1000+V_max-1` recycled (or `v1..vN` if wire allows ints only — keep ints)

Option B: single int space with kind metadata (UI uses kind). Prefer Option A
for user-visible “IDs stay small” validation.

### Lifecycle

```text
new track
  → provisional (no public permanent ID / or display-only)
  → strong emb + pass gates
       → match resident? → resident
       → match visitor? → visitor
       → else mint visitor slot
```

Resident creation in Phase 1: via REST stub `enroll` or env bootstrap list;
full UX in Phase 3.

### Confirmation gates (defaults)

- min crop height / blur (existing)
- ≥ N embeddings within window (e.g. 3) before leaving provisional
- gallery sim ≥ resident_threshold (stricter) or visitor_threshold
- exclusivity / overlap checks

### V3DT public remap

Once household mode emits small IDs natively:
- Set `public_max_id=0` (passthrough) in household configs
- Keep code path for non-household experiments only

### Soft cap replacement

Remove reliance on `max_total_ids` soft mint+merge. Visitor cap is hard recycle;
resident cap is enrollment count.

## Files to touch

- `reid/stable_id_manager.py` — kind/state machine, dual galleries
- `noesis/server/reid_api.py` — stub enroll/list
- `noesis/ds8_runtime.py` / v3dt runtime — config
- `config/infer_v3dt_reimpl_*.yaml` — disable public remap under household
- `tests/test_stable_id_resident_visitor.py`
- `docs/DS8_api_contracts_rest.md`

## Action checklist

- [ ] Implement identity_kind + provisional suppression
- [ ] Resident/visitor gallery split + visitor TTL recycle
- [ ] REST stub for list/enroll residents
- [ ] Disable V3DT public remap when household mode on
- [ ] Unit tests for mint/recycle/promote
- [ ] Update contracts docs

## Risks

- Breaking existing UI that assumes monotonic SIDs — mitigate with additive
  `identity_kind` and document ranges.
- Enrolling wrong person early — require explicit enroll; no auto-promote to
  resident in Phase 1.
