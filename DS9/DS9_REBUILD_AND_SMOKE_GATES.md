# DS9.1 Rebuild And Smoke Gates

This file lists the focused rebuild/smoke work to run only after DeepStream 9.1
is installed or its root is provided through `DS9_DEEPSTREAM_HOME`.

The exact target is SDK root `/opt/nvidia/deepstream/deepstream-9.1`, CUDA
`13.2.0.046`, TensorRT `10.16.0.72`, and driver `595.58.03` or newer. The exact
official base image is pulled and inspected; no application-owned 9.1 binary or
engine has been accepted yet. Hashes and results below that identify TensorRT
10.14 or DeepStream 9.0 are retained historical evidence and must be regenerated
before they can satisfy these gates.

Do not run these against the live DS8 symlink.

## Governance Gate

Before changing the SDK stack or building an artifact, validate the declared
ownership and manifest structure:

```bash
python3 DS9/scripts/validate_runtime_ownership.py
python3 DS9/scripts/validate_asset_manifest.py
python3 -m unittest discover -s DS9/tests -v
```

After rebuilding the canonical graph, require files and provenance:

```bash
python3 DS9/scripts/validate_asset_manifest.py \
  --check-files --profile canonical --require-provenance
```

Before cutover, `validate_runtime_ownership.py --require-parity` must also pass.
Do not waive a declared capability gap by copying a DS8 binary or importing a
DS8 runtime/preflight helper.

## Prerequisite Gate

```bash
DS9_DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-9.1 \
DS9_CUDA_HOME=/usr/local/cuda-13.2 \
  ./DS9/scripts/check_ds9_prereqs.sh
```

The script must report a DeepStream 9 root. If it reports missing DS9 headers,
stop; do not point it at `/opt/nvidia/deepstream/deepstream`.

## Native Bridge Rebuilds

```bash
DS9_DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-9.1 \
DS9_CUDA_HOME=/usr/local/cuda-13.2 \
  ./DS9/scripts/build_all_native_ds9.sh
```

Expected outputs go under `DS9/native_extensions/` when using the launch
wrapper `DS9/scripts/build_native_extensions.sh` or the per-module
`DS9/scripts/build_native_ext_ds9.sh` helpers. Both paths build from
`DS9/native/`; do not point DS9 native rebuilds at root `native/`.

Required bridges:

- `noesis_pose_meta_ext`
- `noesis_depth_meta_ext`
- `noesis_depth_tracking_tensor_ext`
- `noesis_reid_meta_ext`
- `noesis_v3dt_meta_ext`
- `noesis_analytics_meta_ext`
- `noesis_latency_ext` only if in-process latency remains required

`noesis_depth_tracking_tensor_ext` also builds the sibling CUDA kernel source
`DS9/native/noesis_depth_tracking_tensor_kernels.cu` and must expose these
`AlignedDepthFrameDevice` methods after rebuild:

- `sample_roi_stats`
- `sample_masked_roi_stats`
- `sample_masked_person_roi_stats`

Focused import check:

```bash
python3 - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path("DS9/native_extensions").resolve()))
import noesis_depth_tracking_tensor_ext as ext
for name in ("sample_roi_stats", "sample_masked_roi_stats", "sample_masked_person_roi_stats"):
    assert hasattr(ext.AlignedDepthFrameDevice, name), name
print("DS9 depth tensor CUDA sampler bindings present")
PY
```

The YOLO26 pose SGIE now targets the DS9-staged batch-3 asset pair
`DS9/models/onnx/yolo26n-pose_b3.onnx` and
`DS9/models/engines/yolo26n-pose_b3_fp16.engine`. The CPU-exported ONNX source
is staged and hash-verified. Rebuild/runtime must fail fast until the DS9.1
TensorRT 10.16 engine exists; do not copy or reuse the root DS8 engine.

## Parser Rebuilds

```bash
DS9_DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-9.1 \
DS9_CUDA_HOME=/usr/local/cuda-13.2 \
  ./DS9/scripts/build_all_parsers_ds9.sh
```

Expected outputs go under each active parser directory in `DS9/pipelines/`.

Active DS9 parsers:

- `libnvdsinfer_yolo26_seg.so`
- `libnvdsinfer_yolo11_seg.so`
- `libnvdsparsebbox_yolo.so`
- `libnvdsinfer_rfdetr.so`
- `libnvdsinfer_rfdetr_seg.so`
- `libnvdsinfer_deimv2_wholebody49.so`

The Wholebody49 parser source currently has SHA-256
`0b162028c4a4b1c78df2d046ea67d3e12642ede2e7323103edc7872b27b956c6` and
pins the exact batch-stripped NvDsInfer callback contract. The root DS8 parser
and DS9 parser were rebuilt independently from those byte-identical sources;
their distinct binaries have SHA-256
`b70c0413d7dc92e38356ae1dcf4b4eef56f9a4ebb6cd0acdfe6d0f9dc78538c4`
and `b6480bbc437db9e8863c819b87e3f9102bea4c122ab3969dcd1280baad747e3c`,
respectively. Never copy the DS8 binary into DS9. The native adversarial harness
and manifest file/provenance validation remain mandatory before a Wholebody49
runtime lane may start.

The YOLO26 pose no-op parser is archived and must not be rebuilt for the first
DS9 target unless a DS9 runtime test proves tensor-only SGIE config is rejected.

## ROI Exclusion Plugin Rebuild

```bash
DS9_DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-9.1 \
  ./DS9/scripts/build_nvdsroiexclude_ds9.sh
```

Expected output:

- `DS9/build/nvdsroiexclude/libgstnvdsroiexclude.so`

Prep validation after build:

```bash
GST_PLUGIN_PATH="$PWD/DS9/build/nvdsroiexclude:${GST_PLUGIN_PATH:-}" \
  gst-inspect-1.0 nvdsroiexclude
```

Do not install the plugin over the live DS8 runtime during prep.

## MapAnything FP32 Functional Admission

As of 2026-07-12, MapAnything is correctness-first FP32. The prior DS9 FP16
batch-three plan deserialized but emitted all-NaN depth, an all-zero mask, and
invalid confidence sentinels. A separately isolated FP32 diagnostic produced
finite positive depth with dense masks; this selects the build precision but
does not yet prove DS8 parity or runtime readiness.

The write-free maintenance plan is:

```bash
NOESIS_DS9_DOCKER_ROOT=<absolute-secondary-docker-root> \
NOESIS_DS9_ARTIFACT_ROOT=<absolute-artifact-root> \
  DS9/scripts/run_canonical_engine_maintenance.sh \
    --only mapanything --plan
```

The real command is identical without `--plan` and requires an announced
exclusive-GPU window. It builds
`models/engines/mapanything_images_294x518_b3_fp32.plan` without a precision
flag. The argument vector must not contain `--fp16`, `--bf16`, or an empty
placeholder.

Before `install_candidate`, the container must execute one real inference
against the pinned owner-private batch-three fixture below
`models/engine_validation/mapanything/`. The exact command requires
`--dumpOutput` plus `--exportOutput`; a `--skipInference` load is only a
deserialization check and cannot satisfy functional admission. The receipt
binds candidate and fixture hashes, exact command, build-image/runtime
provenance, float32 tensor contracts, finite/positive counts, confidence
sentinels, mask coverage, distribution envelopes, batch consistency, and the
sealed output JSON.

The builder refuses installation on any receipt failure. The host finalizer
and independently callable realization reconciler re-evaluate the receipt
against the installed bytes before writing external realization. A failed
candidate leaves the prior canonical engine in place and publishes no new
realization. Focused CPU coverage is:

```bash
python3 -m pytest -q \
  DS9/tests/test_mapanything_engine_quality_gate.py \
  DS9/tests/test_engine_build_specs.py \
  DS9/tests/test_engine_maintenance_safety.py \
  DS9/tests/test_engine_provenance_reconcile.py
```

After a successful build, the separate fresh depth/floorplan live gate remains
mandatory for every configured camera. Functional candidate admission is not
runtime promotion evidence.

The guarded 2026-07-12 build completed as transaction
`20260712T052425739603Z`. It installed the 3,883,865,652-byte FP32 engine at
SHA-256 `eabc1169c7d725ed7cff171ed54c402c4282fdcf23b87a58f4e41a95fe23ecc8`.
The pinned batch-three inference produced 100% finite positive depth and
confidence plus 99.9954% mask coverage; candidate and installed-path
deserialization both passed. The sealed 25 ms NVML trace contains 8,038
samples, a 29.100129 ms maximum gap, and a 4,655 MiB maximum observed under
the 9,000 MiB guard. Final realization
`6fab7d456c031490f640ee2c3ce5a38922a96ed86a965020ca3051820306dce4`
passes canonical, V3DT, and Wholebody49 file/provenance validation. This closes
engine construction only; it does not close the fresh per-camera runtime gate.

## Wholebody49 Guarded Engine Maintenance

Stage and verify the promoted ONNX sources without GPU access:

```bash
NOESIS_DS9_ARTIFACT_ROOT=<absolute-artifact-root> \
  python3 DS9/scripts/stage_canonical_sources.py --verify-only
```

Preview the exact TensorRT commands through the isolated `runc` daemon. Plan
mode exposes no GPU devices and creates no engine:

```bash
NOESIS_DS9_DOCKER_ROOT=<absolute-secondary-docker-root> \
NOESIS_DS9_ARTIFACT_ROOT=<absolute-artifact-root> \
  DS9/scripts/run_canonical_engine_maintenance.sh \
    --only wholebody49_s_masks --plan

NOESIS_DS9_DOCKER_ROOT=<absolute-secondary-docker-root> \
NOESIS_DS9_ARTIFACT_ROOT=<absolute-artifact-root> \
  DS9/scripts/run_canonical_engine_maintenance.sh \
    --only wholebody49_x_boxes --plan
```

An actual build is allowed only in an announced exclusive-GPU window; the
wrapper refuses to start while any compute owner exists. The explicit planning
budgets are:

- `wholebody49_s_masks`: 42,539,309-byte ONNX; current DS8 reference engine
  33,429,356 bytes; 2,400-second minimum host watchdog; 10,000 MiB GPU-memory
  guard.
- `wholebody49_x_boxes`: 202,881,311-byte ONNX; current DS8 reference engine
  113,269,708 bytes; 2,400-second minimum host watchdog; 11,000 MiB GPU-memory
  guard.

The original `trtexec --memPoolSize=workspace:4096` contract was insufficient:
the first S-mask attempt reached 9,441 MiB, and a second attempt with the exact
4 GiB workspace flag still reached 9,436 MiB. Transactions
`20260711T062159242747Z` and `20260711T071427217911Z` both stopped at the
unchanged 9,000 MiB guard, removed their authorized zero-byte candidates through
the host finalizer, and restored the prior realization.

The first dedicated-builder attempt then set both WORKSPACE and TACTIC_DRAM to
4 GiB. Transaction `20260711T075013061941Z` still reached 9,441 MiB across 78
host samples and stopped at the same guard before candidate publication. The
host finalizer proved the candidate absent and the realization already at its
prior value. This is evidence that the 4 GiB tactic-DRAM limit was still too
high, not permission to weaken the outer guard.

The following single-variable 3 GiB TACTIC_DRAM trial never reached optimizer
allocation. Transaction `20260711T080353252694Z` failed after two samples at a
51 MiB GPU peak, before candidate creation, and the host transaction proved the
candidate absent and the prior realization retained. TensorRT reported exactly:

```text
Error Code 3: API Usage Error (Parameter check failed, condition: (pool == MemoryPoolType::kDLA_MANAGED_SRAM && poolSize == 0) || (static_cast<int64_t>(poolSize) >= kDLA_MIN_MEMORY_POOL_SIZE && hasSingleBit(poolSize)).  In validatePoolSize at /_src/optimizer/api/builderConfig.cpp:340)
```

Thus 3 GiB is not a legal TensorRT 10.14 memory-pool limit. It is an API
contract failure, not evidence about optimizer memory demand.

The next legal 2 GiB TACTIC_DRAM trial did reach the optimizer, but it did not
lower the observed high-water mark. Transaction `20260711T081443440868Z`
reached 9,441 MiB across 78 host samples and stopped at the unchanged 9,000 MiB
S guard. No candidate was published, the host transaction completed clean
rollback with the engine already absent, and realization
`c183f91aaecaca3764081b87c962a87233b2b5c21b2d028e2c12f113a9cb16dd`
was preserved. The identical 4 GiB and 2 GiB TACTIC_DRAM peaks show that this
pool was not the governing bound at those values; they do not justify weakening
the outer guard.

The first S build with the resulting 2 GiB WORKSPACE ceiling also failed to
lower peak pressure. Transaction `20260711T083837156365Z` reached 9,436 MiB
across 78 samples and stopped at the same outer guard. The finalizer again
proved no candidate, an already-absent engine, complete cleanup, and the prior
realization
`cda791ecc5aacd6e043f8714c6d19c8bd72879f5492dbea56fe223e34a59dc99`
unchanged. The 2 GiB S workspace was retained for the following
optimization-level experiment because workspace limiting alone had been
falsified as a way to bring the default optimizer below 9,000 MiB.

Optimization level 0 then changed the failure from host-memory pressure to an
exact tactic-admission constraint. Transaction `20260711T130350917674Z` peaked
at only 506 MiB across 19 samples, but the final mask-producing Myelin node
ending at `/postprocessor/Reshape_4` had no tactic inside the 2 GiB WORKSPACE
budget. Its two half-format attempts each requested exactly 4,571,136,000 bytes
against budgets of 1,952,111,042 and 1,954,461,634 bytes. The float-format
attempt requested 9,142,272,000 bytes against a 1,856,672,132-byte budget.
TensorRT skipped all three, advised increasing WORKSPACE, and failed closed with
Error Code 10 and no implementation for the node. No candidate was published;
the host transaction completed clean rollback with the engine already absent
and preserved realization
`3a896c8dc8dc50d424e9f87e69e9396f9d10e8251e44312d2859312596defc8e`.
The reviewed successor therefore raises only S WORKSPACE to 8,589,934,592
bytes: enough for the evidenced 4,571,136,000-byte half tactic while remaining
below the 9,142,272,000-byte float tactic. X WORKSPACE, both optimization
levels, shared TACTIC_DRAM, FP16, tactic sources, auxiliary streams, profiles,
outputs, and the 9,000/11,000 MiB outer guards remain unchanged.

The first 8 GiB run proved that admission envelope sufficient for engine
generation. Transaction `20260711T131135058479Z` completed engine generation
in 55.7403 seconds. The old wrapper recorded 4,880 MiB across 19 coarse host
samples, while TensorRT reported a 4,603 MiB peak for its own allocator only;
neither value proves the process stayed below the 9,000 MiB outer guard. The
candidate was still withheld before publication because the builder's bounded
logger treated an overlong VERBOSE diagnostic like a captured diagnostic and
made its truncation
flag sticky. The transcript contained 55,191 VERBOSE, 25 INFO, three WARNING,
zero ERROR, and zero INTERNAL_ERROR messages. The warnings were the two FP16
layer-normalization notices and the deliberately excluded 9,142,272,000-byte
float tactic. The builder therefore failed closed after serialization, created
no candidate, rolled back with the engine already absent, and preserved
realization
`493d1cdb168a80751a366b0d12382e0e647e1a2a91b35cc14ee3c09a48a95bb0`.
This is proof that 8 GiB solved tactic admission, not permission to weaken
diagnostic handling.

The reviewed logger contract now ignores only `Severity::kVERBOSE`, before
bounded copy or truncation bookkeeping. INFO, WARNING, ERROR, and
INTERNAL_ERROR remain bounded and emitted. Truncation of any captured message
remains fatal, and ERROR/INTERNAL_ERROR still set sticky fatal state. Both
Wholebody source contracts pin minimum severity `info`, verbose policy
`ignored_before_copy`, captured-message truncation `fatal`, and error state
`sticky_fatal`; producer and independent validators require matching exact
transcript markers.

After that logger correction, transaction `20260711T132034937854Z` generated a
25,348,956-byte S engine in 58.7399 seconds and passed independent candidate and
installed-path deserialization. Host finalization then correctly rolled back
because the Wholebody proof had not received its caller-known maintenance
manifest path; the no-prior candidate was removed and realization
`e87a620dbcc1f231cc8e2e753e20fe49babb00322ce1184f71fba09a697a501b`
remained authoritative. The old wrapper's 4,846 MiB value came from only 20
coarse samples and conflicts with a contemporaneous roughly 9,472 MiB operator
observation. It is not accepted peak evidence, and the 9,000 MiB S ceiling is
not cleared.

Maintenance now creates the container without starting it, launches an
identity-bound NVML-v2 sampler, requires its first sample, and only then starts
the container. The sampler reads raw `used` and `reserved` bytes every 25 ms,
binds the exact GPU UUID, container ID, transaction digest, artifact-root ID,
and wrapper process identity, and fails closed on any raw-byte breach, NVML
error, sampler death, malformed evidence, or gap above 250 ms. Its owner-private
JSONL lives at
`models/engine_finalize/<transaction>-<engine>/gpu-memory.jsonl`, inside the
retained committed transaction cohort. The host independently reconstructs the
maximum observed and evidence digest; it never labels a sampled maximum as the
true peak. Finalization, reconciliation, and later realization validation all
reopen the same raw evidence and require the exact stored summary. Direct
reconciliation cannot omit this proof. Only the eight exact pre-guard realized
artifact/output/maintenance tuples frozen at the 2026-07-11T13:18:00Z cutoff
remain grandfathered; Wholebody49 has no exemption. This is fail-closed
same-user operational evidence, not a cryptographic attestation. A future
hardening pass may cross-check the later committed transaction manifest after
commit ordering permits it.

The first sealed run of the then-current 8 GiB S policy, transaction
`20260711T150649395390Z`, captured 2,368 samples with a 25.457673 ms maximum gap
and stopped cleanly at 9,481 MiB against the 9,000 MiB guard. Reducing S
WORKSPACE to 6 GiB retained enough tactic-admission capacity, but transaction
`20260711T151452845569Z` still reached 9,476 MiB and rolled back at the same
guard. These two high-frequency traces agree with the earlier operator
observation and falsify WORKSPACE reduction as a control for the full-device
peak. The installed TensorRT 10.14 headers define WORKSPACE as intermediate
operation memory and optimization level 0 as selecting the first tactic that
succeeds; the successful transcript separately reports only 4,603 MiB for the
TensorRT allocator. The reviewed policy therefore uses the smallest tested
WORKSPACE that admits the evidenced half tactic, 6 GiB, and raises only the
outer S guard to 10,000 MiB. That keeps more than 2 GiB free on the 12 GiB
target GPU and remains stricter than X's existing 11,000 MiB guard.

The source-policy transition was append-only: transaction
`20260711T151317697215Z` rebased the exact builder/source-contract pair and
produced realization
`f1bad48cc1638399da2a08f614801c7120409a8d8a8fc5c730f846146f231224`.
The first 10,000 MiB build generated and loaded an S candidate but transaction
`20260711T151938705545Z` rejected a disconnected source-rebase graph and rolled
the candidate and realization back. Validation now requires each rebase graph
to be internally exact while allowing a current realization to monotonically
succeed its terminal: realized artifact IDs cannot be dropped and timestamps
must strictly advance.

TensorRT 10.14 `trtexec` does not expose `TACTIC_DRAM` in its memory-pool CLI,
so both Wholebody variants now use the DS9-owned
`DS9/csrc/wholebody49_engine_builder/wholebody49_engine_builder.cpp`. The
reviewed contract creates the TensorRT 10 explicit-batch network, parses an
exact private snapshot of the pinned zero-external-data ONNX, selects FP16,
adds one fixed `images=3x3x640x640` profile, and selects the memory contract from
the already validated `--variant`: S masks uses exactly 6,442,450,944 bytes of
WORKSPACE under its 10,000 MiB guard, while X boxes retains exactly
4,294,967,296 bytes under its 11,000 MiB guard. Both use exactly
2,147,483,648 bytes of TACTIC_DRAM. TACTIC_DRAM must be a positive power of two;
WORKSPACE must be positive and MiB-aligned. C++ compile-time assertions, the
Python maintenance producer, and the independent manifest validator enforce
those pool-specific SDK contracts before GPU work. A
registered error recorder and both memory-pool getters must confirm the selected
variant value and shared tactic limit were accepted exactly. The transcript
contains exactly one matching workspace marker. Builder optimization level is
also selected and getter-verified by variant: S uses level 0, which the installed
TensorRT 10.14.1.48 headers define as disabling dynamic kernel generation and
selecting the first tactic that executes successfully, while X explicitly
retains level 3. Exactly one matching optimization-level marker is required.
Tactic sources and auxiliary-stream policy remain unchanged. The current
memory and optimization policies retain the tested 6 GiB S admission envelope;
the larger X model retains its reviewed memory and optimization policy.

The maintenance path also snapshots and hashes the C++ source before compiling
it in the pinned DS9 image, records the exact compile command/log and executable
digest, binds the emitted transcript to the selected S-mask or X-box output
contract, and creates the candidate with no-follow/exclusive mode 0600.
`trtexec` remains the independent candidate and installed-engine deserializer.
The current host guards are 10,000 MiB for S and 11,000 MiB for X; both retain
2,400-second watchdogs. A CPU-only plan passing this contract is not build
evidence. Rebuild either realized engine only in an announced exclusive-GPU
window and through the same guarded transaction path.

The current committed artifacts are:

- S transaction `20260711T152338477896Z`: 25,327,772 bytes, SHA-256
  `1fb95225e8258af13ac96de5136b85dadb60419a8c49058e70536eef06dd6bdf`,
  2,738 samples, 27.742316 ms maximum gap, and 9,481 MiB maximum observed under
  the 10,000 MiB guard.
- X transaction `20260711T152552934749Z`: 108,579,580 bytes, SHA-256
  `a5f4322d7e123461a1bbc64388b6f0e30c9359091e76ac7a95144648841138c3`,
  12,750 samples, 30.858598 ms maximum gap, and 1,017 MiB maximum observed under
  the 11,000 MiB guard.

Both mode-0600 candidates and installed paths deserialized independently. The
final external realization is
`6fab7d456c031490f640ee2c3ce5a38922a96ed86a965020ca3051820306dce4`;
canonical, V3DT, and Wholebody49 file/provenance profiles validate with no
errors or blockers. This proves artifact realization, not live mask/box quality.

Reference engine sizes estimate disk only; DS9 output bytes may differ. The
wrapper reserves its conservative candidate budget, two independent prior-
engine snapshots, and 10 GiB of residual capacity. After
build, require manifest provenance, separate deserialization, the exact ONNX
tensor names/shapes, parser/mask/bbox semantics, mask-derived OSD policy, an
occupied-scene quality replay, throughput, and observed GPU-memory evidence.
Run live acceptance through the enumerated `wholebody49-s` and
`wholebody49-x` lanes in `run_canonical_runtime_container.py`, then attach
`ds9_live_validation_runner.py` with the matching `--session-id`, lane, and
supervisor `--ownership-evidence-dir`. The occupied-scene gate requires
instance-mask sampling for S and bbox-only sampling for X, and writes the exact
`wholebody49-occupied-scene.json` report into that session's launcher evidence.
Each lane must run the supervisor with `--resource-soak` for at least 300
seconds. The attached runner also seals `wholebody49-media-decode.json`, proving
both direct RTSP H264 decode and decoded WebRTC frames, and the supervisor seals
`runtime-resource-soak-report.json` plus its raw samples. Promotion requires all
three typed reports from the same session, runtime identity, container,
checkout, realization, primary engine, and config cohort. The occupied v2
transcript requires every configured source, at least a 30-second window,
strictly advancing frame IDs, a provisional floor of 2 frame IDs/second, and a
maximum 2.5-second observation gap. Resource ceilings are provisional policy
bounds rather than measured throughput claims: cgroup memory below 20 GiB,
post-warmup growth at most 512 MiB and slope at most 1 MiB/s, PIDs below 4096,
S GPU process memory below 10,000 MiB, and X below 11,000 MiB. Do not
auto-relax a failed bound. An empty scene remains an honest failed occupied
gate; never synthesize a person to clear it. CPU `avdec_h264` is confined to
this edge validation client and is not part of the runtime graph.

## V3DT Guarded Engine Maintenance

The DS9-owned V3DT pipeline, camera inventory, camInfo, tracker config, source
provenance, native bridge, NvMOT helper, and runtime materializer are staged.
The large inputs live under `NOESIS_DS9_ARTIFACT_ROOT`; do not restore ignored
copies under the root checkout.

Preview both exact build paths without a GPU:

```bash
NOESIS_DS9_DOCKER_ROOT=<absolute-secondary-docker-root> \
NOESIS_DS9_ARTIFACT_ROOT=<absolute-artifact-root> \
  DS9/scripts/run_canonical_engine_maintenance.sh \
    --only bodypose3dnet --plan

NOESIS_DS9_DOCKER_ROOT=<absolute-secondary-docker-root> \
NOESIS_DS9_ARTIFACT_ROOT=<absolute-artifact-root> \
  DS9/scripts/run_canonical_engine_maintenance.sh \
    --only v3dt_tracker_reid --plan
```

In an announced exclusive-GPU window, rerun those commands without `--plan`,
in the same order. BodyPose3DNet must exist before NvMOT initializes the tracker
graph and serializes its internal ReID engine. The wrapper rejects any active
GPU compute owner and any host driver below the DeepStream 9.1 minimum of
`595.58.03`, preserves 10 GiB of residual artifact-root capacity, writes engines
atomically, and separately deserializes each new engine. A prelaunch host
transaction binds the builder to the exact prior engine; host commit then
publishes the engine and external realization together or restores both.
Afterward, record manifest provenance and run:

```bash
NOESIS_DS9_ARTIFACT_ROOT=<absolute-artifact-root> \
  python3 DS9/scripts/ds9_preflight.py \
    --config DS9/config/infer_v3dt.yaml \
    --cameras-config DS9/config/cameras_v3dt.yaml

NOESIS_DS9_ARTIFACT_ROOT=<absolute-artifact-root> \
  python3 DS9/scripts/sv3dt_meta_smoke_test.py
```

The smoke requires the owner-only internal bearer and must be followed by the
authenticated `DS9/scripts/v3dt_world_contract_smoke_test.py`, bbox3d coverage,
global-world-v2 replay, identity continuity, throughput/GPU-memory, and
clean-shutdown gates. The v2 gate binds the exact same supervisor session and
effective config hashes, requires all-camera continuity, converts the locked
`xzy` tuple to Y-up `backend_world_m`, and rejects v1/camera-local/raw-tuple
evidence. MV3DT remains disabled and separate: only Kitchen/Family Room is a
prospective future edge, Living Room has no edge, and corrected Kitchen
geometry plus synchronized occupied evidence is required before activation.
AMC execution is deferred and is not part of this rebuild.

## Runtime Smoke Gates After DS9.1 Cutover

- Pose: tensor-only YOLO26 pose SGIE emits object-level tensor metadata and
  `noesis_pose_meta_ext` extracts keypoints and attaches `NOESIS.POSE_FEATURES`.
  Passed in the core DS9 parity run through the DS9-native Service Maker/native
  extension path.
- Object depth: `noesis_depth_meta_ext` attaches/extracts `NOESIS.OBJECT_DEPTH`
  and mask extraction still matches `NvOSD_MaskParams`. Passed in the focused
  DS9 bridge smoke with object-depth GPU ROI copies, attaches, and `ok`
  payloads observed.
- DAv2 depth: `noesis_depth_tracking_tensor_ext` sees `out_buf_ptrs_dev` with
  `disable-output-host-copy=1`, aligns full-frame depth on GPU, and copies only
  object ROIs. Passed in the focused DS9 bridge smoke with native device-frame
  captures and zero core-path CPU-copy violations.
- ReID: `noesis_reid_meta_ext` is now bound to the canonical TAO Swin-Tiny
  `fc_pred/256` SGIE contract at the StableID hook point. Historical OSNet
  stable-ID/native extraction evidence remains useful only for bridge behavior;
  the selected DS9 Swin engine is realized, but live identity-quality evidence
  remains required.
- V3DT: `noesis_v3dt_meta_ext` extracts visibility, image foot, and 3D bbox
  metadata with current world-foot derivation. DS9-native source/config/build
  staging and selected engine realization are complete; fresh runtime evidence
  remains required.
- ROI exclusion: `nvdsroiexclude` removes objects before tracker; excluded
  objects do not produce tracks, ReID embeddings, or pose work. Focused host MP4
  ROI pruning and hot restore passed.
- Alternate profiles: YOLO11 detect-only, YOLO26 detect-only `n/s/m/l/x`,
  YOLO26 segmentation `n/s/m`, RF-DETR segmentation `n/s/m`, and RF-DETR
  detect-only `n/s/m` materialization pass. Wholebody49 `s` mask and `x` box
  source/config/parser materialization and DS9-native engine realization pass;
  runtime quality remains blocked on fresh mask/box and occupied-scene
  validation. Focused startup smokes have covered representative
  detector/segmentation variants.
- PyDS quarantine: the default DS9 run does not require
  `NOESIS_DS9_ALLOW_PYDS_COMPAT=1`. MapAnything no longer has a native
  compatibility switch: the exact DS9-owned native UID/layer/batch selector is
  its sole path, and an older bridge binary blocks startup. Acceptance also
  requires the exact bounded copy counters plus a drained/joined worker with no
  queue, poison, or shutdown failure counters.

Do not treat artifact presence or preflight import success as behavior parity.
Object-depth, depth tensor, ReID, ROI prune/restore, YOLO11 detect-only,
YOLO26 detect-only, YOLO26-seg, RF-DETR-seg, and RF-DETR detect-only now have
focused DS9 evidence. All selected V3DT engines and the static shared-world
conversion are complete. V3DT still needs one fresh occupied, same-session v2
behavior/resource/shutdown bundle, followed separately by MV3DT overlap and
fusion evidence, before full option-surface parity is claimed.
