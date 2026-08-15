# Validation tiers

Choose the first tier that proves the change and stop when it passes.

| Tier | Use | Evidence |
| --- | --- | --- |
| 1 — static/unit | Local implementation or schema change | syntax/static check plus directly affected unit tests |
| 2 — deterministic fixture | Coordinate, report, or consumer contract | registered fixture or saved trace through its exact validator |
| 3 — native runtime | Live tracking, depth, world, identity, media, or lifecycle | short attach-only DS9.1 check of the affected output |
| 4 — Noesis + Menon | Browser delivery or cross-space consumption changed | one authenticated end-to-end producer/consumer smoke |

Do not automatically combine tiers. A frontend style change does not need a
perception run; a model parser change does not need every Menon surface; a docs
change needs no application run.

Runtime checks attach to the installed native service. No tier requires an
appliance release, selector, state clone, candidate, container, or broad suite.
