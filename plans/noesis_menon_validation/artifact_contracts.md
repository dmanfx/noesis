# Validation artifact contracts

Every validation report must state:

- the producer/runtime and exact input identity;
- camera and coordinate-frame scope;
- whether evidence is fixture, saved telemetry, live native DS9.1, or
  Noesis-to-Menon;
- pass, fail, warning, or blocked per evaluated check;
- bounded metrics and paths needed to reproduce the result.

Do not claim semantic accuracy from unlabeled liveness data. Do not claim a
Menon placement pass from Noesis-only output. Missing unrelated inputs are
outside scope; missing required inputs make the affected check blocked.

Machine-readable shapes remain defined by the validation scripts and
`fixture_registry.json`. Historical exhaustive field descriptions are archived
under `plans/archive/completed/noesis_menon_validation_ds8_docs/`.
