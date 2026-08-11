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

## Identity-v2 calibration authority gate

Run:

```bash
python3 -m pytest -q \
  tests/test_identity_v2_calibration.py \
  tests/test_identity_v2_scoring.py \
  tests/test_identity_v2_service.py \
  tests/test_noesis_core_contracts.py
python3 scripts/export_noesis_core_schemas.py --check
```

Required automated evidence includes rejection of v1 labels/datasets/artifacts,
session/run/person/cross-stratum leakage, repeated-person count inflation,
household unknown-person split reuse, over-cap units, non-deterministic thinning,
blocked or zero-exemplar challenges, candidate-kind flips, active semantic and
artifact-pin drift, conservative gallery limits, pre-mutation gallery overflow,
a claimed-pass artifact with a failing exact bound, and any household policy
that lowers a benchmark rejection gate. They must also prove that rebuilding
the loaded ReID extraction component changes model semantics and that scorer
calibration cannot start public authority without exact, distinct, private
coordinator-replay and occupied-scene reports bound through
`noesis.identity.authority_cutover` v1. Adversarial fixtures must prove that one
bad frame fails its encounter and repeated encounters for one person do not
inflate benchmark confidence.

Real authority is a separate blocked gate. The active model profile needs:

- a provenance-locked, subject-disjoint benchmark with at least 50 resident/50
  unknown challenge-covered train people and 300 resident/300 unknown
  challenge-covered holdout people, all from a licensed real-human corpus; and
- an independently labeled home-domain dataset with at least 10 known/10
  unknown train encounters and 20 known/20 unknown holdout encounters.

Benchmark holdout one-sided 95% FAR and misidentification upper bounds must be
at most 1%. Household train and holdout must each contain zero false-accept and
zero misidentification encounters, with FRR at most 35%. No frame, five-second
unit, second tracklet, or additional encounter for the same truth person is
another benchmark confidence trial. Fixture evidence never clears the real
gate. The artifact authorizes the scorer policy only; coordinator replay and
occupied-scene DS8/DS9 validation remain separate cutover gates enforced at
startup. Each cutover report and the cutover artifact require exact byte hashes;
their runtime, executable code profile, topology, cameras, model semantics, and
scorer artifact must match the process being promoted.

## Live smoke (manual / scripted)

```bash
# Prefer existing smoke patterns
python3 scripts/reid_stable_id_smoke_test.py
```

For canonical DS9 acceptance, use
`DS9/scripts/ds9_identity_shadow_live_gate.py` through
`DS9/scripts/ds9_live_validation_runner.py`. The default proves fresh
embedding-backed shadow wiring and tracker-local subject continuity while
keeping public authority blocked. `--identity-require-cross-camera` and
`--identity-require-open-set` make naturally occurring mechanism observations
mandatory, but the report still records semantic accuracy as not evaluated
without licensed truth labels.

_2026-07-10 (Codex): Replaced DS9's repeated-numeric-ID acceptance check with
the evidence-scoped shadow gate above; focused tests cover subject flips,
coherent observation keys, opt-in absent events, and blocked authority (37
focused tests and 278 complete DS9 tests passed)._

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
