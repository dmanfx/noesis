# Household identity — current work order

Status: the household StableID implementation is integrated into the native
DeepStream 9.1 application. Public identity-v2 authority remains in shadow until
real benchmark and household evidence satisfy the gates below.

The original DS8 implementation checklist and dated validation diary are
preserved under
`plans/archive/completed/household_identity_ds8_implementation/`.

## Completed application work

- [x] Default-on household StableID construction in
  `DS9/noesis/ds9_runtime_core.py`.
- [x] Swin SGIE tensor metadata joined in
  `DS9/noesis/pipelines/hooks.py` without a CPU frame-extraction branch.
- [x] Resident, visitor, provisional, gallery-quality, assignment, continuity,
  topology, enrollment, health, and public-wire contracts implemented.
- [x] Identity v2 shadow service and exact downstream OSD join implemented.
- [x] No pressure-driven auto-merge in household mode.
- [x] Native DS9.1 migration preserved identity code, model/config authority,
  and the accepted three-camera performance baseline.

## Remaining authority work

- [ ] Collect a licensed, provenance-bound open-set benchmark satisfying the
  minimum independent-person counts in `calibration_and_enrollment.md`.
- [ ] Collect an independently labeled household train/holdout dataset from the
  current cameras and exact native DS9.1 model profile.
- [ ] Generate and owner-review the v2 scorer artifact; do not enable
  authoritative mode from fixture evidence.
- [ ] Run one coordinator replay and one occupied-scene native DS9.1 identity
  smoke against the exact scorer/model/topology binding.
- [ ] Generate and owner-review the `noesis.identity.authority_cutover` v1
  artifact with `runtime: ds9`, then enable authority only if all pins match.
- [ ] Complete the 14-day household acceptance record in `validation.md`.

## Geometry boundary

- `config/camera_topology.yaml` contains the Kitchen ↔ Family Room identity
  overlap pair. A permit still requires fresh time, world-distance, and
  appearance evidence and fails closed otherwise.
- Living Room ↔ Family Room do not overlap.
- Kitchen ↔ Living Room are adjacency only.
- This identity topology does not enable MV3DT or AMC. Both remain deferred.

## Optional work

- [ ] Evaluate SOLIDER only if accepted Swin evidence shows a consequential
  appearance-confusion problem. It is not a default or fallback.

Use focused tests and one direct live/recorded consumer check for each change.
Do not create release candidates or run broad validation for this workstream.
