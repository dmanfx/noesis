# Identity v2 calibration and enrollment

Status: the correlation-aware v2 contracts and tools are implemented. No real
benchmark/household artifact currently authorizes public identity-v2 output;
shadow remains the safe default.

The full 2026-07-10 design and command record is preserved at
`plans/archive/completed/household_identity_ds8_implementation/calibration_and_enrollment_pre_native_20260815.md`.

## Evidence model

- Benchmark evidence is provenance-bound, licensed real-human, subject
  disjoint, and replayed against the exact active Swin model semantics.
- Household evidence is independently labeled from the installed cameras. It
  checks domain shift and may only tighten benchmark policy.
- Score evidence contains no embeddings. It is private, hash chained, bounded,
  owner-owned, and tied to a stable person/encounter/session/run structure.
- Frames or multiple encounters from one person do not create independent
  confidence trials.

Set `NOESIS_IDENTITY_V2_EVIDENCE_RUNTIME=ds9` for new runtime evidence. The
schema's `ds8` value remains only for reading historical records.

## Workflow

1. Keep `NOESIS_IDENTITY_V2_MODE=shadow`.
2. Capture benchmark and household score evidence with explicit independent
   session/run IDs.
3. Validate and label the evidence with
   `scripts/identity_v2_calibrate.py`.
4. Build disjoint benchmark and household datasets.
5. Calibrate one `noesis.identity.open_set_calibration` v2 artifact and review
   its exact provenance and acceptance metrics.
6. Run one direct native DS9.1 coordinator replay and one occupied-scene check.
7. Build and owner-review one `noesis.identity.authority_cutover` v1 artifact
   with `runtime: ds9`.
8. Enable authoritative mode only when the artifact, model semantic profile,
   executable profile, topology, cameras, and both report bytes all match.

Exact minimum populations and acceptance thresholds are in `validation.md` and
enforced by the generated schemas and calibration code. Do not weaken those
gates, substitute fixture data, or carry a DS8 authority artifact forward.

## Enrollment

Menon exposes enrollment only to the authenticated owner. A proposal consumes a
fresh server observation key, shows the intended action, and requires explicit
confirmation. The browser never receives an embedding. Stale, consumed, or
conflicting evidence fails and requires a fresh proposal.

Names bind to resident UUIDs. Visitor slots and tracker IDs are not durable name
authorities. Legacy identity state may be reviewed through the dry-run migration
tool, but there is no automatic apply or merge path.
