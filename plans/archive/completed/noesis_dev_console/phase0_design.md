# Noesis Dev Console — Phase 0 Design Gate

Date: 2026-06-07

## Scope

Local web utility (`noesis/dev_console`) for DS8 pipeline launch, preset management, preflight validation, and runtime hot controls. Separate from `oai2-fe`.

## API / Schema Decisions

- **LaunchSpec**: dataclass → argv + env + optional overlay YAML path
- **ValidationResult**: severity `block|warn|info`, code, message, fix_hint
- **Preset**: typed `canonical|smoke|experimental|archival` with required artifact declarations
- **Materialization**: committed artifacts under `build/dev_console/<launch_id>/` via `NOESIS_DEV_CONSOLE_LAUNCH_DIR`

## Security

- Console binds `127.0.0.1:9090`
- Child runtime defaults `--ws-host 127.0.0.1 --rest-host 127.0.0.1`
- Runtime proxy v1 allowlist: stats, depth refresh, trail toggle, BEV config/overlay, WebRTC signaling only

## Docs / Tests Touched

- `plans/noesis_dev_console/plan.md`
- `plans/DS8/ds8_design_decisions.md` (entry added)
- `docs/DS8_testing_guide.md` (Phase 7)
- `tests/test_dev_console.py`