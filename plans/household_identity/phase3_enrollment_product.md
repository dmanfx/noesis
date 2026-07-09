# Phase 3 — Enrollment & Product Layer

**Goal:** Let the user name household members and bind them to stable resident
identities; expose identity health; keep merges suggest-only.

**Depends on:** Phase 2 gate G2 (Phase 1 stub may exist earlier)

## Scope

### In
- Resident enrollment REST (complete)
- `display_name` / `resident_uuid` on tracking + optional BEV
- Frontend affordance to name/enroll (minimal, match existing oai2-fe patterns)
- Identity health endpoint + simple report script
- Docs: WS/REST contracts, user-facing README notes
- Deprecate reliance on soft-merge as primary identity tool

### Out
- Full multi-user auth / cloud identity
- Automatic clothing-change wardrobe UI

## Technical design

### Enrollment flow

1. User selects a track on mosaic/BEV with strong `reid_confidence`
2. `POST /reid/residents/enroll` with `{ stable_id or track key, display_name }`
3. Server copies quality-gated embeddings into resident gallery, assigns UUID,
   allocates sticky resident public ID
4. Future matches retrieve that resident

### Suggest merges

Keep suggest API; never auto-apply in household mode. UI shows candidates with
block reasons (copresence, both active, low sim).

### Health report

`GET /reid/identity_health` + `scripts/household_identity_health_report.py`:
- resident/visitor counts
- mint rate / hour
- false_share_blocked / overlap grants
- gallery ages / exemplar counts
- provisional ratio

### Frontend

Minimal changes in `oai2-fe`:
- Show `display_name` when present else `stable_id`
- Color key remains stable_id/resident id
- Optional enroll control only if REST available — match existing panel style;
  do not redesign the whole app

## Files to touch

- `noesis/server/reid_api.py`
- `oai2-fe/src/*` (LegendPanel / StreamPanel / App as needed)
- `docs/DS8_api_contracts_rest.md`, `docs/DS8_api_contracts_ws.md`
- `scripts/household_identity_health_report.py`
- `reid/stable_id_manager.py` — enroll helpers

## Action checklist

- [ ] Complete enroll/list/patch/delete REST
- [ ] Wire display_name/uuid into tracking payloads
- [ ] Minimal FE name display + enroll hook
- [ ] Health endpoint + report script
- [ ] Update docs; mark soft-merge plan superseded for household mode
- [ ] G3 validation

## Risks

- Enrolling during dual-cam overlap with wrong track selection — require
  high confidence + single selected track confirmation.
