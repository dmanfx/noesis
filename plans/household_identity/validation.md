# Household Identity — Validation

## Gates

| Gate | When | Pass criteria |
|------|------|---------------|
| G0 | After Phase 0 | Unit tests green; exclusivity+overlap unit cases; pose wired; poisoned state archived on household enable; confidence fields present in tracking payload |
| G1 | After Phase 1 | Resident/visitor split live; provisional not permanent; dual-cam false share blocked in synthetic tests; overlap co-vis allowed |
| G2 | After Phase 2 | Swin path confirmed default; quality gallery gates; assignment conflicts measured; no torchreid on DS8 |
| G3 | After Phase 3 | Enroll+name round-trip via REST; health endpoint; docs contracts updated |

## Required automated tests

Add under `tests/` (create missing StableID tests — prior docs referenced files
that are absent):

- `tests/test_stable_id_exclusivity.py` — same SID two cams denied without overlap
- `tests/test_stable_id_overlap_permit.py` — granted/denied by distance/time/topology
- `tests/test_stable_id_provisional.py` — no permanent mint without emb confirmation
- `tests/test_stable_id_resident_visitor.py` — ID space split + visitor recycle
- `tests/test_stable_id_pose_wire.py` — pose features affect score when quality OK
- `tests/test_household_state_archive.py` — cutover archives legacy files

Run focused:

```bash
python3 -m pytest tests/test_stable_id_*.py tests/test_household_*.py -q
```

## Live smoke (manual / scripted)

```bash
# Prefer existing smoke patterns
python3 scripts/reid_stable_id_smoke_test.py
```

Additional checks to add in Phase 0/1:
- Log/metric: `overlap_permit_grant_count` increments in overlap scene
- Log/metric: `false_share_blocked_count` increments in two-person similar-clothes scene
- Confirm public IDs stay in resident/visitor ranges after 30+ min run

## 14-day home acceptance (user)

See README success metric. Record results under this folder as
`evidence/YYYY-MM-DD_home_acceptance.md` when collected.

## Perf check

Before/after detection-wake or live_health snapshot at 3 cams / 2 people.
Fail Phase 0 if identity changes blow the +10% CPU budget without justification
in `decisions.md`.
