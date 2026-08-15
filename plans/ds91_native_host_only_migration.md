# Native DeepStream 9.1 migration — remaining cleanup

Status: application migration and performance acceptance are complete. Removal
of the inert DS8/DS9.0 and Noesis-Docker surfaces remains pending.

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

## Remaining legacy removal

These steps are intentionally destructive. Resolve exact targets immediately
before acting and preserve the existing legacy archive/checksums first.

- [ ] Verify the existing DS8/DS9.0 archive can still list and extract
  representative source/config records.
- [ ] Remove executable DS8 and DS9.0 runtime entrypoints, active configs,
  services, build wrappers, and compatibility launch links outside the archive.
- [ ] Remove installed DeepStream 8.0/9.0 SDK/package directories after proving
  the native service and build tools resolve only 9.1 libraries.
- [ ] Move or remove old DS9.0/container artifact roots without changing the
  accepted DS9.1 realization.
- [ ] Remove Noesis Dockerfiles, image-build/runtime scripts, isolated-daemon
  configuration, images, networks, and data root after resolving their exact
  ownership and confirming no open handles.
- [ ] Stop and disable the isolated Noesis Docker daemon/socket.
- [ ] Remove stale Noesis Docker/image variables from active environment and
  service files.

Do not create DS8/DS9.0 stubs, aliases, or fallbacks. Archive material is for
reading only and must not remain executable.

## Ordinary Docker boundary

The system Docker daemon historically hosted an unrelated Z-Wave JS workload.
Do not stop or uninstall ordinary Docker until that workload has been moved or
the user explicitly accepts its outage. Noesis can and must remain fully
Docker-free regardless of that separate service.

## Direct acceptance after cleanup

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
