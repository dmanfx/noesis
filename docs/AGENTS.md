# AGENTS.md — Documentation

This directory contains current Noesis/DeepStream 9.1 documentation. Repository
policy lives in the root `AGENTS.md`; this file narrows documentation practice.

## Policy precedence

- Root `AGENTS.md` is authoritative.
- `docs/history/` is an archive and is never normative.
- `docs/README.md` is the current documentation index.
- Current documentation work belongs on the repository `DS9` branch. Do not
  switch to or commit on the historical `feature_DS8` branch without explicit
  user direction.

## Documentation rules

1. Describe the canonical native-host DS9.1 application, not DS8, DS9.0, or the
   retired DS9.1 container deployment.
2. Verify behavior against current source, `DS9/config/infer.yaml`, the native
   host supervisor, and live state when a claim is environment-dependent.
3. Preserve useful superseded material under `docs/history/` with a dated
   status notice. Do not merely rename an old claim and present it as current.
4. Keep current contracts in generically named files such as
   `api_contracts_ws.md`; runtime-family history belongs in the archive.
5. Update diagrams whenever topology, ownership, transport, or data authority
   changes.
6. Add substantive milestones to `upgrade_history.md` and non-trivial current
   choices to `architecture_decisions.md`.
7. Use repo-relative paths and environment-variable placeholders. Never put
   credentials or machine-specific absolute paths in documentation.
8. Prefer direct, focused validation commands. Do not prescribe staging,
   selectors, candidate ceremony, broad suites, or repeated unchanged checks.

After documentation changes, run `./scripts/check_agents_docs_consistency.py`
and `git diff --check`. Do not run application tests for docs-only changes.
