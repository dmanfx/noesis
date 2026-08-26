# Native DeepStream 9.1 migration — cleanup status

Status: application migration, performance acceptance, and repository
consolidation are complete. Machine-level retirement of the isolated Noesis
Docker service and its storage remains a separately controlled cleanup phase.

The detailed migration diary, machine inventory, commands, and intermediate
failures are preserved at
`plans/archive/migrations/ds91_native_host_only_migration_worklog_20260813.md`.
It is historical evidence, not an execution script.

## Accepted native state

- [x] DeepStream 9.1, CUDA 13.2, TensorRT 10.16.0.72, GStreamer 1.24.2,
  Python 3.12, and the accepted driver run directly on the host.
- [x] `noesis-appliance.service` launches
  `DS9/scripts/run_canonical_runtime_host.py`; the Noesis process has no
  container ancestor or Docker socket dependency.
- [x] Native extensions, parsers, plugins, and the selected TensorRT engines
  resolve from the native DS9.1 environment/artifact realization.
- [x] Live three-camera behavior returned to roughly 29.9 FPS/camera after
  restoring dependency, environment, and pipeline parity.
- [x] A 90.67-second non-July recorded pressure run sustained 25.45 FPS/camera
  with no pipeline or zero-copy errors.
- [x] Menon/oai2-fe media, tracking, depth/world output, and PCF BEV tracking
  were exercised directly.
- [x] Current documentation describes only the native DS9.1 application;
  superseded implementation records are archived.

## Repository retirement completed

The 2026-08-26 consolidation completed these repository-scoped steps without
changing the installed service or external dependency worktrees:

- [x] Preserve and independently verify supplemental history, orphaned
  workspace material, and root evidence archives. Their checksums and
  extraction checks are recorded under `archive/manifests/`.
- [x] Remove executable DS8/DS9.0 entrypoints, duplicate configs, build
  wrappers, compatibility launch paths, and root native artifacts from the
  active repository.
- [x] Remove repository Dockerfiles, image builders, container supervisors,
  registry/promotion machinery, and container-only validation paths.
- [x] Consolidate executable ownership under `DS9/`, shared application code
  under `noesis/` and `noesis_core/`, and external dependency provenance under
  `third_party/`.
- [x] Confirm the canonical DeepStream installation path resolves to 9.1 and
  the active Noesis process uses the native-host supervisor.

## Remaining machine-level removal

These steps are intentionally destructive and were not implied by the
repository-only consolidation. Resolve exact targets and open handles again
immediately before acting:

- [ ] Stop and disable `noesis-ds9-secondary-docker.service`, which was still
  active at the repository cleanup checkpoint.
- [ ] Remove its isolated Docker images, networks, daemon configuration, and
  `/mnt/noesis_storage/noesis-ds9-docker` data root after confirming ownership.
- [ ] Inventory and remove any superseded DS9.0/container artifact roots while
  preserving the accepted DS9.1 realization and external archives.
- [ ] Remove stale Noesis Docker/image variables from machine-level environment
  and service files, if any remain after the daemon is retired.

Do not create DS8/DS9.0 stubs, aliases, or fallbacks. Archive material is for
reading only and must not remain executable.

## Ordinary Docker boundary

The system Docker daemon historically hosted an unrelated Z-Wave JS workload.
Do not stop or uninstall ordinary Docker until that workload has been moved or
the user explicitly accepts its outage. Noesis can and must remain fully
Docker-free regardless of that separate service.

## Direct acceptance after each cleanup phase

Run only the checks invalidated by removal:

1. `run_canonical_runtime_host.py check` passes and reports only native 9.1
   origins.
2. Restart the existing Noesis user service once.
3. Confirm all three sources advance; tracking, identity, depth/world, WebRTC,
   and the dashboard work.
4. Take one short FPS/CPU/GPU snapshot and compare it with
   `docs/runtime_baseline.md`.
5. Confirm no Noesis process, service, environment, open file, or socket points
   to DS8, DS9.0, a Docker daemon, or a removed path.

No candidate publication, selector, state clone, release bundle, broad suite,
or rollback rehearsal is part of this cleanup.
