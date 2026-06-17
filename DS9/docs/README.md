# DS9 Docs Index

This directory is the launch handoff for the DeepStream 9 Noesis app under
`DS9/`.

## Read First

1. `../README.md` - current DS9 launch status, runtime ownership boundary, build
   order, run commands, and latest validation summary.
2. `migration_state.md` - detailed migration state, validated gates, and
   remaining work.
3. `known_blockers.md` - active caveats and resolved blockers.
4. `validation_runbook.md` - commands for the next validation pass.
5. `../asset_manifest.yaml` - expected DS9-owned assets and generated outputs.

## Current Position

DS9 runtime execution is owned by:

- `DS9/noesis/ds9_runtime.py`
- `DS9/noesis/ds9_runtime_core.py`

DS9 executable code must not import or spawn `noesis/ds8_runtime.py`. The static
prep check enforces that entrypoint boundary.

DS9 is not yet a hermetic standalone repository. It still consumes parent-repo
shared application context such as `config/cameras.yaml`, common calibration and
geometry helpers, WebSocket server code, and a few root validation clients. That
is an explicit packaging boundary, not a DS8 runtime fallback.

## Active Docs

- `known_blockers.md` - current blocker/caveat list.
- `migration_state.md` - detailed state and historical validation evidence.
- `validation_runbook.md` - operator validation commands.
- `MapAnything_Depth.md` - MapAnything depth behavior notes.
- `Static_ROI_Exclusion.md` - ROI exclusion behavior notes.

## Historical Reference

`history/ds8/` contains copied DS8-era contract/reference documents and the
DS8 design-decision ledger. Keep them only as migration evidence until a later
docs pass rewrites the remaining contracts into DS9-native names.

Do not use historical DS8 docs as permission to route DS9 execution through DS8
runtime paths, DS8 engines, or DS8 native extension binaries.
