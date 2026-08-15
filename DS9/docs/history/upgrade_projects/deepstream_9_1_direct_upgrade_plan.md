# DeepStream 9.1 Direct Upgrade Plan

Status: complete; optional telemetry-boundary optimization deferred, 2026-08-12.

This is a direct upgrade of the existing Noesis DeepStream 9.0 application to
DeepStream 9.1. The first 9.1 runtime must minimally preserve the current
baseline graph, configuration process, public contracts, and operator flow.
This is not a redesign, a parallel runtime, or a formal release ceremony.

## Exact target

| Authority | Required value |
| --- | --- |
| Official dGPU base | `nvcr.io/nvidia/deepstream:9.1-triton-multiarch@sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994` |
| DeepStream SDK | `9.1` |
| SDK root | `/opt/nvidia/deepstream/deepstream-9.1` |
| CUDA | `13.2.0.046` |
| TensorRT | `10.16.0.72` |
| Minimum NVIDIA driver | `595.58.03` |
| Current host driver | `595.71.05` (passes the version floor) |

The bundled Service Maker wheel must be installed explicitly in the derived
image. Repository pins override older 9.0 examples retained in dated reports or
upstream skill text.

## Current state

- The derived images are `noesis-ds9-dev:9.1-20260812`
  (`sha256:88d80ad35f12ec3a574cf2555a8242d33ac4110abdcc5f88a6cbdee40dfcf872`)
  and `noesis-ds9-runtime:9.1-20260812`
  (`sha256:b97a32b082e74265c15e767bcaafa4dc1d8947e53feb36adb9baafdf69ba762e`).
  Their parent and layer lineage were inspected against the exact base digest.
- The 9.1 build produced and load-checked 7 native extensions, 7 nvinfer
  parsers, 1 TensorRT plugin, and 3 GStreamer plugins. The base manifest is
  `9218943c6c595be38576483188c27c5d0dbd86c563a6d12efa8d8e8fc507bbee`.
- The isolated 9.1 artifact realization contains 10 finalized, deserialized
  engines and has SHA-256
  `9c3815bbf86eb41a94efb504fad79a9208c543e05f98c68d017cb1a8dcfd2b26`.
  The selected baseline engines and future-disabled SV3DT assets were rebuilt;
  MV3DT remained disabled. The 9.0 realization remains intact for rollback.
- Commit `5a6c93c` rebound hardened DAv2 and MapAnything registrations to the
  rebuilt engine/config identities without changing calibration knots or world
  weights. Commits `4c87067` and `39c9eab` restored the shared DS9 runtime
  publication gate and its direct fixtures; 17 affected and 9 shared focused
  checks passed.
- The initial managed activation was
  `deploy-20260812-ds91-direct-r9`. Its 17.2 FPS sample exposed a CPU regression
  rather than a DeepStream or TensorRT throughput limit. Commit `4c19515`
  restored the one-second identity-retention cadence instead of performing
  durable retention on every source frame, and capped DS9 CPU math pools before
  NumPy-backed imports. Forty-two directly affected tests passed.
- The corrected three-camera live run sustained 29.8-30 FPS per source;
  authenticated WebRTC decoded 239 frames in eight seconds. A resource snapshot
  showed about 86% container CPU, 53% GPU utilization, 6,308 MiB VRAM, 1.643 GiB
  container RAM, no pipeline/boundary errors, and zero zero-copy violations.
- The strict 3 ms telemetry boundary remains an optional optimization target.
  Corrected live samples observed roughly 6-11 ms WebSocket p99 without
  suppressing source, tracking/world, or media throughput. The earlier 97.7 ms
  p99 was CPU-bound boundary preparation during the regression, not inference,
  camera, network, or dashboard-rendering latency.
- Earlier r7/r8 candidates failed closed and restored the predecessor as a
  whole. The broader static-prep wrapper is still not a pass because of the
  unrelated parity-marker classifications and missing `apply_source_hysteresis`
  markers already recorded before this upgrade.

## Parity boundary

The direct upgrade preserves these current application capabilities before any
new 9.1 feature is enabled:

- three secret-backed RTSP sources, per-source dewarping, decoded-source
  isolation, and bounded progress monitoring;
- preprocess, PGIE selection, ROI exclusion, NvDCF/SV3DT tracking, ReID, pose,
  DAv2 object depth, manual-only MapAnything, analytics, tiling, OSD, and the
  one-encode H.264 SHM/WebRTC path;
- REST and WebSocket contracts, canonical observations/world publication,
  stable identity behavior, scene-prior/scene-fusion diagnostics, floorplan and
  depth workflows, and bounded orderly shutdown;
- existing runtime secret, external artifact-root, and writable-state
  boundaries.

The selected 9.0 baseline is the behavioral comparison point. Its binary
artifacts are not migration inputs.

## Execution order

### 1. Pin the 9.1 toolchain

- [x] Replace active 9.0 image, SDK, CUDA, TensorRT, tracker-library, and build
  defaults with the exact target values above.
- [x] Keep the image reference digest-pinned and install the bundled
  `pyservicemaker` wheel explicitly.
- [x] Update version assertions and focused tests without weakening fail-closed
  behavior.
- [x] Confirm the host driver passes `595.58.03` without changing the host CUDA
  or TensorRT installations.

### 2. Pull and inspect the official base

- [x] Pull the exact `linux/amd64` digest through the existing secondary-Docker
  boundary.
- [x] Confirm DeepStream 9.1, CUDA `13.2.0.046`, TensorRT `10.16.0.72`, SDK
  root, Python ABI, Service Maker wheel, headers, tracker library, and required
  GStreamer factories.
- [x] Build the normal derived development image and thin runtime layer once.
  Their immutable IDs are recorded in Current state.

### 3. Rebuild every SDK-coupled binary

- [x] Rebuild all DS9 native Python extensions from `DS9/native/`, including
  analytics, pose, depth, ReID, V3DT, latency, and orderly-EOS bridges selected
  by the manifest.
- [x] Rebuild every active nvinfer parser and TensorRT/GStreamer plugin against
  the 9.1 headers and libraries.
- [x] Verify import origin, ABI filename, linker dependencies, SDK runpaths,
  parser symbols, and `gst-inspect-1.0` for the affected plugin factories.
- [x] Update artifact provenance only from the bytes actually produced by this
  rebuild. Do not copy or relabel a 9.0 `.so`.

### 4. Rebuild selected TensorRT engines

- [x] Rebuild the engines selected by the canonical baseline and its direct
  downstream consumers with TensorRT `10.16.0.72`.
- [x] Rebuild the DS9-owned SV3DT BodyPose3DNet and internal tracker-ReID
  engines, but keep the separate MV3DT capability disabled.
- [x] Deserialize each installed engine once and update its realization from
  the actual 9.1 result. Do not rebuild unused variants merely to broaden the
  upgrade.
- [x] In one process, verify both `pyservicemaker` import and the application's
  required Torch CUDA import before accepting the Python dependency lock.

### 5. Run focused parity checks

Use the narrowest useful checks and do not repeatedly rerun unchanged suites:

Focused source/authority, native, manifest, appliance, and directly affected
runtime checks passed without repeating the full suite. The broader static-prep
script remains open for the unrelated findings listed in Current state.

- [x] Run static/version assertions for the changed Docker, path, build, and
  manifest contracts.
- [x] Import-smoke rebuilt native extensions; symbol/load-smoke rebuilt parsers
  and plugins; deserialize selected engines.
- [x] Run focused runtime/config, tracking/world, analytics, depth, identity,
  media, and shutdown tests affected by the upgrade.
- [x] Run one bounded live three-camera baseline smoke covering authenticated
  readiness, advancing sources and tracking/world capability health, decoded
  WebRTC media, and one resource snapshot. A separate recorded replay was not
  repeated because the live path directly exercised the required graph. The
  telemetry-p99 observation in Current state remains a focused follow-up.

No full-suite repetition, long soak, sealed replay, or publication ceremony is
required for this direct development upgrade unless focused evidence exposes a
cross-system failure.

### 6. Activate directly

- [x] Use the existing bounded runtime selector/service path with the new 9.1
  image and artifact realization.
- [x] Preserve the current private state, runtime secrets, and external
  scene-fusion catalog; do not package generated scene evidence into Git.
- [x] Confirm service readiness, advancing source/tracking/world health, media,
  and the normal predecessor-stop/candidate-start transaction. No extra restart
  ceremony was added.
- [x] Confirm failed 9.1 candidates reselect the last known-good 9.0
  release as a whole. Never mix 9.0 engines or native libraries into 9.1.

## DS9/9.1 opportunities beyond the current DS8 application

These are post-parity opportunities, not upgrade requirements, and none is
enabled by this plan. NVIDIA groups most of them under the DeepStream 9.0
previous-release feature list in the 9.1 release notes. Some also appeared in
DS8 developer-preview documentation; “beyond DS8” here means beyond the
capabilities accepted in this application's DS8 baseline, not a claim that no
DS8 preview artifact ever existed.

| Capability | Potential value to Noesis | Decision after parity |
| --- | --- | --- |
| MV3DT plus pose-aware SV3DT/MV3DT | Cross-camera identity consistency and fused world positions in true overlap | High future value, but disabled under the topology/geometry gate below |
| AutoMagicCalib | Produce and export a consistent calibration set for MV3DT | Deferred until Kitchen geometry and Kitchen/Family Room evidence are ready |
| MaskTracker | Longer-lived segmentation masks and better occlusion handling | Evaluate later on representative scenes; its SAM2 compute/memory cost must beat the current segmentation/tracking lane |
| Dynamic stream handling in the demuxer | Cleaner camera recovery and source lifecycle handling | Candidate after baseline parity; preserve the existing source-health contracts |
| `nvtracker` and `nvdsanalytics` REST control | Bounded tracker/ROI tuning without rebuilding the graph | Candidate only behind current authenticated configuration and persistence contracts |
| OpenTelemetry and Prometheus exporters | SDK-native FPS, latency, source-health, GPU, and memory observability | Compare with existing telemetry before adding another exporter path |
| Sparse4D multi-camera BEV 3D detection/tracking | Research path for richer multi-camera 3D perception | Research lane only; different outputs and compute budget, not a PGIE replacement |
| Inference Builder | Repeatable custom-model import and TensorRT realization | Useful for the next model onboarding; retain repository manifests and direct runtime gates |
| Cosmos Reason 2 through the VLM plugin | Optional incident and scene summaries downstream | Optional semantic consumer only; never geometry, tracking, or world-state authority |
| Open-source `nvdsanalytics`, `nvdsmetainsert`, and `nvdsmetaextract` | Easier debugging and extension of metadata contracts | Review when a concrete metadata gap appears; do not fork components speculatively |
| DeepStream monorepo and 13 agentic skills | Skill-first, version-aligned API, graph, model, profiling, MV3DT, and AMC workflows | Already adopted as the first work route; repository pins remain authoritative |

The DS9.1 Jetson, RTX Pro 4500, and newer Triton platform support does not
materially benefit this RTX 3060 dGPU deployment and is not an implementation
target.

Official sources:

- [DeepStream 9.1 release notes](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_Release_notes.html)
- [NVIDIA DeepStream 9.1 skills and MV3DT/AMC workflow](https://developer.nvidia.com/blog/build-a-multi-camera-3d-tracking-application-with-nvidia-deepstream-9-1-skills/)

## MV3DT and AMC boundary

MV3DT remains a distinct, disabled capability. It must not alias the existing
SV3DT `v3dt` mode or silently fall back to it.

The only prospective vision-neighbor graph is:

```text
kitchen     <-> family-room   (future, disabled)
living-room     no MV3DT edge
```

Living Room and Family Room do not overlap. Kitchen and Living Room are close
adjacencies, not an MV3DT overlap pair. Kitchen/Family Room activation is
blocked until Kitchen geometry is corrected and one synchronized occupied
session proves overlap, timestamp alignment, peer association, and fused
positions.

AMC is deferred with that geometry work. Do not start the AMC stack, generate
an AMC dataset, modify calibration through AMC, or claim an auto-calibrated
multi-view result during this upgrade. Installing or documenting the AMC skills
does not enable AMC.

## Completion definition

The upgrade is complete when the 9.1 image and all selected SDK-coupled
artifacts are built on the exact target stack, focused checks pass, the normal
three-camera baseline runs through its existing application workflow, and no
material functional or performance regression is evident. MV3DT/AMC evidence
is explicitly outside this completion boundary.
