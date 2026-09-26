# Engine build and optional measurement — Steps 4–5

Build the engine required by the selected consumer. Capacity exploration is a
separate, explicitly requested benchmark task; it is not an import prerequisite.
Existing canonical-engine maintenance follows the native host maintenance route
in the repository's DeepStream guidance.

## Resolve the build contract

Use the task, model metadata, checked-in consumer config, and existing build
manifest to establish:

- Exact ONNX path, every input name, dtype, layout, and fixed/dynamic dimension.
  Inspect with `scripts/model/inspect-onnx.py` relative to this skill; read the
  full input list, not only its convenience height/width fields.
- Consumer batch behavior and required min/opt/max shapes. Set the optimum to
  the normal requested workload and the maximum to its supported upper bound.
  Include batch 1 only when required and supported. For fixed-shape ONNX, keep
  its fixed batch and omit dynamic shape profiles; do not relabel it dynamic.
- Selected precision, calibration/plugins, spatial resolution, and exact output
  path. Preserve the model and consumer quality contract.
- Native DS9.1/CUDA 13.2/TensorRT 10.16.0.72 environment, builder executable and
  library origins, available GPU resources, and build workspace/time budget.

Reuse established evidence. Ask only for missing choices that prevent a valid
build; never choose batch 64, a 32 GiB workspace, or the first wildcard engine by
habit. Do not infer stream capacity from the maximum batch dimension.

## Step 4 — Build once for that contract

Prefer the selected native builder when one exists. For a new standalone import,
verify flags using the installed `trtexec --help`. The following **Bash template**
applies only to a verified single-input NCHW model with dynamic batch and fixed
C/H/W. Resolve every variable first; it is not a model-independent command.
`BUILD_FLAGS` is a Bash array containing the verified precision/plugin options
for this model, including no precision override when its typed graph requires
that. `BUILD_TIMEOUT_S` and `WORKSPACE_MIB` come from the bounded build plan.

```bash
set -euo pipefail
: "${TRTEXEC:?}" "${ONNX_PATH:?}" "${ENGINE:?}" "${BUILD_LOG:?}"
: "${INPUT_NAME:?}" "${C:?}" "${H:?}" "${W:?}"
: "${MIN_BS:?}" "${OPT_BS:?}" "${MAX_BS:?}"
: "${BUILD_TIMEOUT_S:?}" "${WORKSPACE_MIB:?}"
declare -p BUILD_FLAGS >/dev/null
# ENGINE and BUILD_LOG must be fresh paths in an existing task output directory.
test ! -e "$ENGINE"
test ! -e "$BUILD_LOG"
timeout --signal=TERM --kill-after=30s "$BUILD_TIMEOUT_S" "$TRTEXEC" \
  --onnx="$ONNX_PATH" \
  --minShapes="${INPUT_NAME}:${MIN_BS}x${C}x${H}x${W}" \
  --optShapes="${INPUT_NAME}:${OPT_BS}x${C}x${H}x${W}" \
  --maxShapes="${INPUT_NAME}:${MAX_BS}x${C}x${H}x${W}" \
  "${BUILD_FLAGS[@]}" \
  --memPoolSize="workspace:${WORKSPACE_MIB}M" \
  --skipInference --saveEngine="$ENGINE" 2>&1 | tee "$BUILD_LOG"
test -s "$ENGINE"
```

For multiple inputs or dynamic spatial dimensions, construct profiles from all
verified input contracts instead of adapting only batch. Validate positive
integer dimensions and `min <= opt <= max` before launch. The workspace limit
is a tactic-workspace budget, not a total GPU-memory cap; monitor the build and
stop if the task's resource bound is exceeded.

Preserve the builder's exit status (`pipefail` above). A stale or partial engine
file does not prove success. On failure/timeout, retain the log and mark any
partial output unusable. Diagnose the actual failure before a bounded retry; do
not switch toolchains, export shapes, algorithms, or precision speculatively.

Record the exact artifact, profile, input/precision contract, builder version,
and result for the next phase. Build completion proves serialization only;
exercise deserialization/inference and the direct DeepStream consumer when those
phases are in scope. An engine-only task can stop with downstream validation
explicitly unperformed.

## Step 5 — Optional bounded benchmark

Run this phase only when performance evidence is requested or a specific build
concern requires a bounded inference check. It is not required to generate a
parser/config or to validate detections.

Before a capacity experiment, record the target FPS, allowed batch list, maximum
batch, maximum rebuild count, per-run duration and total time budget, GPU-memory
budget, and stop conditions. Default additional rebuilds to zero for an import.
Select these bounds from the requested workload and available resources; never
run a doubling loop until an extrapolated stream count fits a profile.

Benchmark only shapes supported by the built engine. A batch-1 latency result
exists only if batch 1 is supported. Keep output precision, model resolution,
inference cadence, and quality settings fixed. An isolated inference experiment
may use `--noDataTransfers`; label its result **inference-only**. It does not
prove zero-copy or end-to-end media performance.

For the single-input dynamic example above, a selected supported `MEASURE_BS`
and fresh `BENCH_LOG` can be measured with the same exit-status protection:

```bash
set -euo pipefail
: "${TRTEXEC:?}" "${ENGINE:?}" "${BENCH_LOG:?}" "${BENCH_TIMEOUT_S:?}"
: "${INPUT_NAME:?}" "${MEASURE_BS:?}" "${C:?}" "${H:?}" "${W:?}"
: "${DURATION_S:?}" "${WARMUP_MS:?}"
test ! -e "$BENCH_LOG"
timeout --signal=TERM --kill-after=30s "$BENCH_TIMEOUT_S" "$TRTEXEC" \
  --loadEngine="$ENGINE" \
  --shapes="${INPUT_NAME}:${MEASURE_BS}x${C}x${H}x${W}" \
  --noDataTransfers --duration="$DURATION_S" --warmUp="$WARMUP_MS" \
  2>&1 | tee "$BENCH_LOG"
```

Check completion and metric units before reporting throughput. For batched
requests, images/s is queries/s multiplied by the measured batch size. Dividing
that by target FPS yields an inference-only sizing estimate, not a measured
DeepStream stream ceiling. Source count, mux batch, and engine batch are separate
contracts. Validate every source, decode/media progress and direct consumers in
an authorized application benchmark before making a capacity claim. Use
[the profiling skill](../../deepstream-profile-pipeline/SKILL.md) for that work.

## Failure diagnosis and delivery

- Confirm the installed builder's memory-pool unit syntax and the effective
  workspace shown in its log. Do not silently substitute a fixed large budget.
- A `ForeignNode`/tactic failure needs the exact node, input profile, plugins,
  toolchain and resource evidence. Export changes or static-batch simplification
  alter the model contract and need their own equivalence checks; they are not
  automatic fallback instructions.
- Throughput may vary across supported shapes. Explore it only inside the
  benchmark's batch, rebuild, resource and time bounds.
- Report created artifacts, checks performed, failed/skipped checks, and any
  requested phase still incomplete. Do not emit a capacity number when no
  benchmark ran or an integration-ready claim when only serialization passed.

For a requested legacy benchmark report, retain the report reader's expected
log names in that task's output directory only for measurements actually run.
Do not launch missing measurements just to populate a template or fabricate
zero-valued results. An incompatible report schema is a reporting limitation.
