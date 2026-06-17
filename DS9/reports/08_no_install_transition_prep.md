# No-Install DS9 Transition Prep

Date: 2026-05-11

Scope: work completed without downloading, installing, or changing the live DS8
runtime path.

## Local Header Check

`/opt/nvidia/deepstream/deepstream-9.0` is not present. The active
`/opt/nvidia/deepstream/deepstream` symlink still points at `deepstream-8.0`.
CUDA currently resolves to 13.0 on this machine; the DS9 prep scripts require a
CUDA 13.1 root unless `DS9_ALLOW_CUDA_MISMATCH=1` is explicitly set for
investigation.

Decision: do not rebuild anything against the active symlink. A DS9 rebuild
must fail loudly until a real DS9 root is available through
`DS9_DEEPSTREAM_HOME` or `/opt/nvidia/deepstream/deepstream-9.0`, and a DS9
CUDA root is available through `DS9_CUDA_HOME` or `/usr/local/cuda-13.1`.

## Prep Work Completed

- Added shared DS9 build guardrails in `scripts/ds9_build_env.sh`.
- Added `scripts/check_ds9_prereqs.sh` to verify DS9 headers, Service Maker
  headers, parser headers, CUDA headers, pybind11, and GStreamer build metadata.
- Reworked native bridge build scripts to route through
  `scripts/build_native_ext_ds9.sh` and emit outputs under `artifacts/native/`.
- Added `scripts/build_all_native_ds9.sh`.
- Updated active custom parser Makefiles to default to DeepStream 9 and emit
  parser outputs under `artifacts/parsers/`.
  - Current launch cleanup retargets active parser outputs to their
    `DS9/pipelines/nvdsinfer_*` directories; `artifacts/parsers/` is historical
    prep output.
- Added `scripts/build_all_parsers_ds9.sh`.
- Reconstructed the source-controlled `nvdsroiexclude` plugin under
  `csrc/nvdsroiexclude/`.
- Added `scripts/build_nvdsroiexclude_ds9.sh`.
- Added `scripts/run_static_prep_checks.sh`.
- Added `DS9_REBUILD_AND_SMOKE_GATES.md`.

## Current Blocker

Actual compile/rebuild work is blocked by the absence of DS9 headers/libraries
and CUDA 13.1 on this machine. This is intentional: compiling against DS8
headers or the current CUDA 13.0 stack would produce misleading artifacts and
could hide DS9 API drift.

## Validation Run During Prep

Completed:

```bash
./DS9/scripts/run_static_prep_checks.sh
```

Result: passed.

Expected blocker:

```bash
./DS9/scripts/check_ds9_prereqs.sh
```

Result: fails because `/opt/nvidia/deepstream/deepstream-9.0` is not present.

## Next Action After DS9 Is Available

Run:

```bash
DS9_DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-9.0 \
DS9_CUDA_HOME=/usr/local/cuda-13.1 \
  ./DS9/scripts/check_ds9_prereqs.sh
```

If it passes, continue with:

```bash
./DS9/scripts/build_all_native_ds9.sh
./DS9/scripts/build_all_parsers_ds9.sh
./DS9/scripts/build_nvdsroiexclude_ds9.sh
```
