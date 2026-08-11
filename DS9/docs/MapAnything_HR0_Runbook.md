# MapAnything HR-0 Runbook

This runbook prepares the isolated FP32 batch-three `378x672` MapAnything
candidate defined by `DS9/config/mapanything_profiles.json`. HR-0 is an
offline export, source inspection, candidate build, functional inference, and
benchmark exercise. It does not select the candidate in
`DS9/config/infer.yaml`, replace the canonical engine, update the asset
manifest, alter depth-registration authority, or start/restart DS9.

## Why this profile

| Contract | Canonical | HR-0 |
| --- | ---: | ---: |
| Input per batch | `3x3x294x518` | `3x3x378x672` |
| Native pixels per frame | 152,292 | 254,016 |
| Relative native pixels | 1.000 | 1.668 |
| Three-layer D2H per frame | 1,827,504 bytes | 3,048,192 bytes |
| Three-layer D2H per batch | 5,482,512 bytes | 9,144,576 bytes |
| Precision | FP32 | FP32 |
| Runtime interval for a controlled A/B | 89 | 89 |

Both HR-0 dimensions are divisible by the encoder patch size of 14. `672x378`
is exactly 16:9, so a 16:9 full-resolution input does not spend native output
pixels on letterbox padding.

The candidate engine path is below
`DS9/models/engines/candidates/`. Its ONNX bundle has a dedicated directory
below `DS9/models/onnx/candidates/`, so exporter-created external initializer
files cannot collide with the canonical ONNX bundle. The current engine
remains `mapanything_images_294x518_b3_fp32.plan`.

## Pin the model source and export environment

HR-0 is valid only with this complete source identity:

| Component | Required identity |
| --- | --- |
| MapAnything source | official `facebookresearch/map-anything` v1.1.3, commit `9d1db2dd728bd8a10e74b15d2eb646e1bf933791` |
| Hugging Face model | `facebook/map-anything-apache`, revision `4cf3561e403dcec91b41629f0ce7793e3f04d15c` |
| UniCeption | `>=0.1.7` |
| Checkpoint loading | strict |
| TensorRT builder | exact DS9 banner `TensorRT v101401` |

Do not export this checkpoint through the repository's legacy
`src/mapanything` checkout. Older source can accept a newer configuration
while omitting architecture fields, which makes a successful-looking export
an invalid comparison. Use a clean, detached official checkout and a
task-isolated Python environment. For example:

```bash
export MAPANYTHING_WORK_ROOT="${EXTERNAL_WORK_ROOT}/mapanything-hr0"
export MAPANYTHING_V11_REPO="${MAPANYTHING_WORK_ROOT}/map-anything-v1.1.3"
export MAPANYTHING_V11_VENV="${MAPANYTHING_WORK_ROOT}/python-env"
export MAPANYTHING_V11_PYTHON="${MAPANYTHING_V11_VENV}/bin/python"
export MAPANYTHING_EXPORT_DEVICE="cuda"

git clone https://github.com/facebookresearch/map-anything.git \
  "${MAPANYTHING_V11_REPO}"
git -C "${MAPANYTHING_V11_REPO}" checkout --detach \
  9d1db2dd728bd8a10e74b15d2eb646e1bf933791
test "$(git -C "${MAPANYTHING_V11_REPO}" rev-parse HEAD)" = \
  "9d1db2dd728bd8a10e74b15d2eb646e1bf933791"
test -z "$(git -C "${MAPANYTHING_V11_REPO}" status --porcelain)"

python3 -m venv --system-site-packages "${MAPANYTHING_V11_VENV}"
"${MAPANYTHING_V11_PYTHON}" -m pip install --upgrade "uniception>=0.1.7"
```

The example selects CUDA for the coordinated GPU-first export path.
`MAPANYTHING_EXPORT_DEVICE` may instead be `cpu` when an operator has measured
enough host-memory and time headroom, but the choice must be explicit. The
exporter never changes devices on failure. A failed CUDA export therefore
stops the transaction; an operator must deliberately start a new transaction
with `--export-device cpu` to use CPU.

The guarded preflight verifies the exact MapAnything commit, source package
origin, task virtual environment, UniCeption version, pinned Hugging Face
revision, strict checkpoint-load policy, and exact DS9 TensorRT banner before
export. It records that identity in the guarded-run receipt. A local
measurement build made with another TensorRT ABI is not a promotion artifact.

## Generate the exact command set

From the repository root:

```bash
"${MAPANYTHING_V11_PYTHON}" DS9/scripts/mapanything_profile_tool.py \
  plan \
  --profile hr0_378x672_b3_fp32 \
  --mapanything-repo "${MAPANYTHING_V11_REPO}" \
  --python "${MAPANYTHING_V11_PYTHON}" \
  --export-device "${MAPANYTHING_EXPORT_DEVICE}" \
  --shell
```

The command set includes:

- private candidate/evidence directory creation;
- ONNX export from the pinned clean official checkout;
- one-image repeated-batch and three-image scene-fixture preparation;
- ONNX plus external-initializer inspection and hashing;
- a fixed-batch-three FP32 TensorRT build;
- deserialization and one real functional inference;
- strict numerical output summarization;
- compute-only and transfer-inclusive timing runs; and
- TensorRT engine I/O inspection.

If large DS9 artifacts live outside the checkout, provide the same physical
root used by DS9 maintenance:

```bash
"${MAPANYTHING_V11_PYTHON}" DS9/scripts/mapanything_profile_tool.py \
  plan \
  --profile hr0_378x672_b3_fp32 \
  --artifact-root "${NOESIS_DS9_ARTIFACT_ROOT}" \
  --mapanything-repo "${MAPANYTHING_V11_REPO}" \
  --python "${MAPANYTHING_V11_PYTHON}" \
  --export-device "${MAPANYTHING_EXPORT_DEVICE}" \
  --shell
```

The artifact root must contain `models/`; logical artifact identities remain
under `DS9/models/`. Supplying `--mapanything-repo`, `--python`, and
`--export-device` is mandatory for an operator run even though the CLI exposes
defaults for inspection and test use.

## Use the guarded transaction

The `plan --shell` output is an inspection surface, not the preferred executor.
Use `guarded-run` for the actual candidate transaction. It is read-only by
default and requires an external artifact root:

```bash
export NOESIS_DS9_ARTIFACT_ROOT="${EXTERNAL_STORAGE_ROOT}/noesis-ds9-artifacts"

"${MAPANYTHING_V11_PYTHON}" DS9/scripts/mapanything_profile_tool.py guarded-run \
  --mapanything-repo "${MAPANYTHING_V11_REPO}" \
  --python "${MAPANYTHING_V11_PYTHON}" \
  --export-device "${MAPANYTHING_EXPORT_DEVICE}" \
  --functional-image "${LOCKED_CORPUS_ROOT}/repeated-frame.png" \
  --scene-image "${LOCKED_CORPUS_ROOT}/camera-0.png" \
  --scene-image "${LOCKED_CORPUS_ROOT}/camera-1.png" \
  --scene-image "${LOCKED_CORPUS_ROOT}/camera-2.png" \
  --dry-run
```

Replace the illustrative paths with real operator-controlled paths. The
artifact root must already exist, be caller-owned and writable, and reside on
a different filesystem from the checkout. When artifact, cache, and temporary
roots share one filesystem, the guard reserves:

- 16 GiB for the ONNX bundle, candidate engine, fixtures, and evidence;
- 8 GiB for model caches;
- 24 GiB for exporter and TensorRT temporary work; and
- 8 GiB of residual headroom.

That is a fail-closed 56 GiB free-space requirement, not an estimate that
permits consuming the final 56 GiB. Optional `--cache-root` and `--temp-root`
arguments can place those two workloads on other high-capacity filesystems.
Otherwise they use task-specific directories below the artifact root. The
wrapper sets `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TORCH_HOME`,
`XDG_CACHE_HOME`, and `TMPDIR` for its subprocesses; it does not repurpose a
home or Codex directory.

The dry run fails unless:

- the profile is an isolated candidate;
- all full-resolution fixture images decode and meet the native dimensions;
- the candidate ONNX directory, engine, fixture directory, evidence directory,
  and temporary directory are absent;
- the shared DS9 artifact-transaction lock and every existing top-level DS9
  maintenance lock in that artifact root are available;
- no GPU compute owner, `trtexec`, MapAnything exporter, or other guarded HR
  run is active;
- the required executables resolve and `trtexec --help` contains the exact DS9
  `TensorRT v101401` banner; and
- the deployment selector, canonical MapAnything INI, asset manifest, and
  engine-source authority can be fingerprinted.

The ONNX exporter runs in a private staging directory, the engine and
functional fixture use partial names, and candidate outputs are installed
without replacement only after export, source inspection, deserialization,
functional inference, both benchmarks, and engine inspection succeed.
Canonical selectors and authorities are hashed again before installation and
at completion.

Execution requires both explicit flags:

```bash
"${MAPANYTHING_V11_PYTHON}" DS9/scripts/mapanything_profile_tool.py guarded-run \
  --mapanything-repo "${MAPANYTHING_V11_REPO}" \
  --python "${MAPANYTHING_V11_PYTHON}" \
  --export-device "${MAPANYTHING_EXPORT_DEVICE}" \
  --functional-image "${LOCKED_CORPUS_ROOT}/repeated-frame.png" \
  --scene-image "${LOCKED_CORPUS_ROOT}/camera-0.png" \
  --scene-image "${LOCKED_CORPUS_ROOT}/camera-1.png" \
  --scene-image "${LOCKED_CORPUS_ROOT}/camera-2.png" \
  --execute \
  --confirm-exclusive-gpu-window
```

Do not use those flags until the coordinated GPU window begins. The wrapper
continues polling GPU owners and conflicting processes during every phase,
terminates its subprocess group on contention, timeout, or oversized logs,
and leaves a private fail-closed receipt. It never edits runtime selectors or
promotion manifests.

## Prepare quality fixtures

Use decoded RGB frames at least `672x378`. Do not feed HR-0 an image previously
downsampled to `518x294`.

Prepare two fixtures:

1. One fixed full-resolution frame repeated across all three batch members.
   This isolates numerical validity and batch divergence.
2. Three full-resolution frames from the locked camera corpus. This measures
   representative timing and scene output.

The generated plan prints both commands with image placeholders. The fixture
tool performs aspect-preserving downsampling, symmetric zero padding, BGR-to-RGB
conversion, `0..1` scaling, NCHW packing, and exclusive owner-private writes.
It refuses to upscale a source or overwrite an existing fixture. Each fixture
gets a receipt containing its exact shape, size, SHA-256, source dimensions,
resize, and padding.

The CPU fixture transform is a reproducible functional/benchmark input. It is
not claimed to be pixel-identical to DeepStream's GPU preprocessing.

## Export and inspect before building

Run the guarded executor only in a coordinated GPU window. The exporter now
accepts `--output-name`, so the ONNX and its external-data files retain the
candidate identity instead of being emitted as a generic `model.onnx`. The
entire bundle is staged in a dedicated candidate directory.

The generated exporter command pins the Hugging Face revision and includes
`--skip-eager-smoke`. This skips only the redundant eager wrapper forward that
would otherwise run immediately before export and can unnecessarily double
peak memory pressure. ONNX tracing still executes the model graph. Subsequent
source inspection, TensorRT deserialization, real engine inference, and strict
numerical output summarization remain mandatory, so this option does not waive
functional validation.

Run the printed `inspect-source` command before TensorRT. It fails unless the
candidate has:

- input `images`, float32, dynamic or fixed batch three, `3x378x672`;
- exactly `depth`, `conf`, and `mask` float32 outputs;
- one channel per output;
- ONNX opset 17; and
- at least one present external initializer.

The resulting source receipt binds the main ONNX and every referenced sidecar
by size and SHA-256. It is candidate evidence, not a silent edit to
`DS9/config/engine_source_contracts.json`.

## Build and measure

Do not build while the live DS9 graph owns the GPU. In the coordinated window,
the guarded executor runs these phases in order:

1. `build_candidate`
2. `deserialize_candidate`
3. `functional_inference`
4. `summarize_functional_output`
5. `benchmark_compute_only`
6. `benchmark_with_transfers`
7. `inspect_engine`

No FP16 or BF16 flag is allowed. The build uses fixed
`min/opt/max=images:3x3x378x672`, builder optimization level 0, no auxiliary
streams, and a 1024 MiB workspace. The functional summary rejects malformed
JSON, shape/name drift, NaN, and Inf, then records positive fractions,
distribution percentiles, mask coverage, and maximum identical-batch delta.

Capture peak VRAM with the repository's existing NVML sampler during the build
and both timing runs. Preserve the full `trtexec` transcripts alongside the
generated JSON evidence.

## Compare quality

Use a three-arm comparison on the same full-resolution source frames:

| Arm | Source/checkpoint contract | Native input | What it isolates |
| --- | --- | ---: | --- |
| A | existing legacy canonical | `294x518` | deployed baseline |
| B | official v1.1.3 plus pinned checkpoint | `294x518` | source/checkpoint architecture gain relative to A |
| C | official v1.1.3 plus pinned checkpoint | `378x672` | resolution gain relative to B |

Interpret `B - A` as the model-source/checkpoint effect, `C - B` as the native
resolution effect, and `C - A` as the total candidate effect. A two-arm
canonical-versus-HR-0 comparison cannot attribute a gain or regression to
resolution because it changes source architecture and resolution together.

Keep batch size, FP32 precision, full-resolution inputs, preprocessing,
runtime interval, alignment, fusion configuration, and scoring code identical
across all three arms. Keep B in its own candidate ONNX, engine, fixture, and
evidence paths; it must not overwrite either A's canonical artifacts or C's
HR-0 artifacts. Compare native tensors before 1080p alignment as well as the
downstream aligned/fused products. At minimum record:

- valid and mask coverage;
- depth/confidence distributions;
- identical-batch maximum delta;
- edge transition width and RGB/depth edge alignment;
- floor-plane and fixed-anchor residuals;
- thin-structure continuity and hole/component counts;
- TensorRT latency percentiles;
- D2H bytes and duration;
- peak VRAM; and
- tracking-FPS impact during the same manual burst.

Promotion thresholds live in
`DS9/docs/MapAnything_Depth_Panel_Quality_Plan.md`. A visually sharper image
without metric-depth, coverage, stability, and runtime headroom is not a
winner.

## Promotion boundary

The Python exact-capture bridge is ready for either profile: expected height,
width, and payload bytes are instance-level values resolved from the selected
profile. The `profile` selector is removed before nvinfer properties are
materialized.

Promotion remains a separate reviewed change. It must update and validate the
engine source authority, asset manifest/realization, runtime model binding,
depth-registration fingerprints, and live quality evidence together. Until
then, omit `profile` from the active model entry; the default remains
`canonical_294x518_b3_fp32`.
