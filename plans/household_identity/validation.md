# Household identity — validation

Use the narrowest gate that covers the change. Do not run release staging or a
complete DS9.1 suite for ordinary identity work.

## Focused component gate

Choose the relevant tests from the existing identity files, for example:

```bash
python3 -m pytest -q \
  tests/test_stable_id_assignment.py \
  tests/test_stable_id_exclusivity.py \
  tests/test_household_state_archive.py
```

For identity-v2 calibration or service changes, add only the directly affected
calibration/scoring/service/schema tests and run
`python3 scripts/export_noesis_core_schemas.py --check` when a generated
contract changes.

## Direct native application smoke

When runtime identity behavior changes:

1. use the installed native DS9.1 service, not a second runtime;
2. observe authenticated tracking/identity telemetry on port 6008 for a short,
   occupied interval;
3. require fresh Swin-backed evidence and advancing source frames;
4. confirm tracker continuity and that public/OSD identity fields agree;
5. inspect only the relevant identity counters and one resource snapshot.

Use `DS9/scripts/ds9_identity_shadow_live_gate.py` directly when its required
session/run identifiers are available. It is the focused native identity gate;
there is no broad release-validation wrapper in the active repository.

Cross-camera and open-set flags are meaningful only when the observed scene
actually contains those events. A liveness run without truth labels cannot
claim semantic identity accuracy.

## Public authority gate

Identity-v2 authority remains blocked until all of these exist for the exact
native DS9.1 model profile:

- a licensed, subject-disjoint benchmark with at least 50 resident and 50
  unknown train people, plus 300 resident and 300 unknown holdout people;
- independent household evidence with at least 10 known/10 unknown train and
  20 known/20 unknown holdout encounters;
- benchmark holdout one-sided 95% FAR and misidentification bounds at or below
  1%; household train/holdout with zero false accepts and misidentifications and
  FRR at or below 35%;
- one passing coordinator replay and one passing occupied-scene DS9.1 report;
- an owner-reviewed cutover artifact whose scorer, model, code, topology,
  cameras, and report hashes all match.

Fixture evidence does not satisfy this gate.

## Home acceptance

Record a 14-day owner review only when collected. Targets are bounded resident
and visitor spaces, no false identity sharing on non-overlap cameras, correct
Kitchen/Family co-visibility when present, greater than 90% resident handoff
recall within 30 seconds, and no material performance regression.
