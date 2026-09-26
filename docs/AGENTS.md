# AGENTS.md — Documentation

This directory contains current Noesis/DeepStream 9.1 documentation. Repository
policy lives in the root `AGENTS.md`; this file narrows documentation practice.

## Policy precedence

[Root policy](../AGENTS.md) owns runtime, branch, archive, and validation rules.
The [documentation index](README.md) routes current specifications.

## Documentation rules

1. Verify operational claims against current source, the
   [pipeline config](../DS9/config/infer.yaml), and the selected native process
   when behavior is environment-dependent.
2. Keep public contracts in generically named files; place superseded material
   in [history/](history/) with a dated status and update incoming index links.
3. Update diagrams when topology, ownership, transport, or data authority changes.
4. Record non-trivial decisions in [architecture_decisions.md](architecture_decisions.md)
   and completed milestones in [upgrade_history.md](upgrade_history.md).
5. Use portable relative Markdown links for important local references so the
   documentation checker can verify their targets. Use environment placeholders
   for external runtime roots and never include credentials.

Documentation-only edits use the checks in the root
[files, docs, and commits rules](../AGENTS.md#files-docs-and-commits).
