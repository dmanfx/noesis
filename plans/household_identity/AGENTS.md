# AGENTS.md – `plans/household_identity/`

Operating rules for the Household Identity / StableID–ReID rework.

## Policy precedence

- Extends root `AGENTS.md` and `plans/AGENTS.md`.
- DS8 rules still apply: canonical path under `noesis/`, no hidden fallbacks, GPU-first / zero-copy where DeepStream supports it, no guessed DeepStream APIs.
- This directory is the authoritative task definition for identity work. If code and these plans disagree, update the plan or record a design decision — do not silently invent behavior.

## Required read order

1. `AGENTS.md` (this file)
2. `README.md`
3. `decisions.md`
4. `contracts.md`
5. `work_order.md`
6. The phase doc for the active task (`phase0_*.md` … `phase3_*.md`)
7. `performance.md` before any change that touches embeddings, gallery matching, or per-frame budgets
8. `validation.md` before claiming a phase complete
9. Relevant runtime docs: `docs/DS8_api_contracts_ws.md`, `docs/DS8_api_contracts_rest.md`, `docs/DS8_testing_guide.md`

## Non-negotiable product rules

1. **Closed-world residents.** Enrolled household members occupy a small stable resident ID space. Day-to-day matching retrieves against that gallery; it does not mint unbounded SIDs for known people.
2. **Visitors are ephemeral.** Unknown people get visitor slots with TTL recycle. Visitor minting must not pollute resident numbering.
3. **Global exclusivity with FoV-overlap exception.** A SID may be active on two cameras only when geometry (and optionally appearance) supports the same physical person in a configured overlap region. Different people must never share a SID.
4. **Provisional ≠ public.** Weak / no-embedding / confirming tracks must not mint permanent public IDs.
5. **No auto-merge under pressure.** Alias merges are enrollment/user-driven (or high-confidence suggest-only). Do not use auto-merge as a substitute for correct matching.
6. **Zero-copy / low-latency.** Prefer SGIE tensor meta → native extract → numpy view / GPU matmul. Do not add CPU crop→torchreid paths on the DS8 runtime. Do not raise per-frame embedding budgets without measuring detection-wake cost.
7. **No poisoned-state carryover.** Identity state files that encode historical fragmentation must be archived/reset as part of cutover, not silently reused.

## Implementation boundaries

- Primary code: `reid/stable_id_manager.py`, `noesis/pipelines/hooks.py`, `noesis/pipelines/hooks_v3dt_reimpl.py`, `noesis/ds8_runtime.py`, `noesis/server/reid_api.py`, ReID SGIE configs under `pipelines/`, models under `models/`.
- Do not route identity through deprecated pre-DS8 runtimes.
- Do not treat V3DT `_public_stable_id` remapping as the real identity fix; it may remain as a temporary UI shim only until closed-world IDs land, then retire or reduce to pass-through.
- Portable paths only (no new `/home/...` hardcodes).

## Progress updates

- Update `work_order.md` checkboxes after each coherent unit of work.
- Add a dated one-line note under completed items.
- Non-trivial design changes → `decisions.md` and a short entry in `plans/DS8/ds8_design_decisions.md`.
- After docs/AGENTS edits: run `./scripts/check_agents_docs_consistency.py`.

## Parallelism guidance for subagents

- Prefer one phase owner at a time for `reid/stable_id_manager.py` (high conflict risk).
- Safe parallel tracks: backbone/export scripts, REST/enrollment API, frontend alias UX, unit tests, docs/contracts, camera-topology config.
- Always re-read `contracts.md` before changing WS/REST fields.
