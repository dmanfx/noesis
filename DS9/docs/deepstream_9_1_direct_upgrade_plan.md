# DeepStream 9.1 Direct Upgrade Plan

Status: active execution plan, 2026-08-12.

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

- The source/toolchain pin update is complete. Its focused source suite passed
  29 tests with 1 skipped, and all 4 direct 9.1 pin assertions passed.
- The exact 9.1 base image is pulled and inspected as Linux/amd64 image
  `sha256:c41fa01c8657a7476a4b252261d9277f5117c33083c399a49b2a993ef9f6ac70`;
  the SDK/toolchain inspection passed.
- No derived 9.1 development or runtime image has been built.
- No native bridge, parser, GStreamer/TensorRT plugin, or TensorRT engine has
  been rebuilt for 9.1.
- No 9.1 recorded-media or live-camera runtime validation has run.
- Existing 9.0 images, artifact realizations, engine hashes, canary reports,
  and performance results remain historical comparison evidence only.

Nothing compiled for DS8 or DS9.0 may be admitted into the 9.1 runtime. A
historical pass does not satisfy a 9.1 build, load, or runtime gate.

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
- [ ] Build the normal derived development image and thin runtime layer once.
  Record their resulting immutable IDs before a runtime launch.

### 3. Rebuild every SDK-coupled binary

- [ ] Rebuild all DS9 native Python extensions from `DS9/native/`, including
  analytics, pose, depth, ReID, V3DT, latency, and orderly-EOS bridges selected
  by the manifest.
- [ ] Rebuild every active nvinfer parser and TensorRT/GStreamer plugin against
  the 9.1 headers and libraries.
- [ ] Verify import origin, ABI filename, linker dependencies, SDK runpaths,
  parser symbols, and `gst-inspect-1.0` for the affected plugin factories.
- [ ] Update artifact provenance only from the bytes actually produced by this
  rebuild. Do not copy or relabel a 9.0 `.so`.

### 4. Rebuild selected TensorRT engines

- [ ] Rebuild the engines selected by the canonical baseline and its direct
  downstream consumers with TensorRT `10.16.0.72`.
- [ ] Rebuild the DS9-owned SV3DT BodyPose3DNet and internal tracker-ReID
  engines, but keep the separate MV3DT capability disabled.
- [ ] Deserialize each installed engine once and update its realization from
  the actual 9.1 result. Do not rebuild unused variants merely to broaden the
  upgrade.
- [ ] In one process, verify both `pyservicemaker` import and the application's
  required Torch CUDA import before accepting the Python dependency lock.

### 5. Run focused parity checks

Use the narrowest useful checks and do not repeatedly rerun unchanged suites:

The focused source suite currently passes 29 tests with 1 skipped, and all 4
direct 9.1 pin assertions pass. The broader static-prep script remains open
because it still reports pre-existing parity-marker classifications and missing
`apply_source_hysteresis` markers in both hook copies; it must not be reported
as passed.

- [ ] Run static/version assertions for the changed Docker, path, build, and
  manifest contracts.
- [ ] Import-smoke rebuilt native extensions; symbol/load-smoke rebuilt parsers
  and plugins; deserialize selected engines.
- [ ] Run focused runtime/config, tracking/world, analytics, depth, identity,
  media, and shutdown tests affected by the upgrade.
- [ ] Run one short recorded-input pipeline smoke and inspect detections,
  tracks, identity/depth/world payloads, mosaic output, and clean shutdown.
- [ ] Run one bounded live three-camera baseline smoke when private sources and
  artifacts are present. Compare obvious FPS/latency/GPU-memory behavior with
  the retained 9.0 baseline; investigate only a material regression.

No full-suite repetition, long soak, sealed replay, or publication ceremony is
required for this direct development upgrade unless focused evidence exposes a
cross-system failure.

### 6. Activate directly

- [ ] Use the existing bounded runtime selector/service path with the new 9.1
  image and artifact realization.
- [ ] Preserve the current private state, runtime secrets, and external
  scene-fusion catalog; do not package generated scene evidence into Git.
- [ ] Confirm service readiness, advancing source/tracking/world health, media,
  depth/floorplan access, and a normal stop/start cycle.
- [ ] If the 9.1 runtime fails, stop it and reselect the last known-good 9.0
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
