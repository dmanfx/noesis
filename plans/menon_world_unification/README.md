# Menon World Unification

This folder is the execution workspace for strict pose-first world unification between Menon and Noesis.

## Mission

Implement and validate a single pose-driven world coordinate path where:
- Menon exports camera poses (`position + yaw/pitch/roll`).
- Noesis derives `E` (world->camera) from pose.
- Baseline DS8 tracking and BEV outputs stay in `menon_scene`.
- No fallback calibration paths are used in strict mode.

## Current phase

- Phase 1: baseline integration
- Phase 1.5: consistency and validation hardening
- Phase 1.6: OBJ-unit baseline lock (no fallback, no client-side rescaling)

## Startup checklist for new agents

1. Read `AGENTS.md`.
2. Read `work_order.md`.
3. Read `contracts.md`.
4. Continue from the top unchecked item in `work_order.md`.
5. Record progress in `timeline.md` and `session_notes.md`.

## Artifact map

- `work_order.md`: execution checklist authority.
- `contracts.md`: interface and behavior authority.
- `decisions.md`: ADR-style decision log.
- `validation_matrix.md`: requirement-to-verification mapping.
- `timeline.md`: append-only chronology.
- `session_notes.md`: active execution notes.
- `handoff.md`: snapshot for next agent handoff.
