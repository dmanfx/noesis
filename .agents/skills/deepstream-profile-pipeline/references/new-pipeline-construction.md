# New-pipeline construction with performance requirements

Use only when construction is requested. Existing-runtime diagnosis follows the
[skill entrypoint](../SKILL.md#existing-pipeline-diagnosis). Read the stages needed
by the requested result and preserve their dependencies; a new graph does not
require every possible sweep, rebuild or report.

## 1. Resolve requirements and construct the baseline

Resolve source types/counts/resolution/FPS, model input and precision, inference
cadence, tracking, output branches and latency/throughput targets from the task
and consumer contracts. Use the generator for a requested graph and the import
skill only for a requested new detector. Prefer existing compatible native
engines. Missing precision, quality or output authority is not permission to
choose a faster lower-quality preset.

Consider these settings only where the new graph leaves them open; validate
properties using the installed SDK and measure their effect:

| Setting | Evidence to use |
| --- | --- |
| Engine min/opt/max shapes | Actual supported consumer batches and model input dimensions |
| Mux batch and timeout | Source count, batching behavior, source cadence and latency target |
| Mux dimensions | Product geometry/output contract and source/model requirements |
| Decoder surfaces and memory type | NVMM path, measured pressure and bounded memory budget |
| Tracker config/dimensions | Selected tracking quality and occupied-scene evidence |
| Queue sizes | Bounded worker/service time and publication ordering contract |
| Output branches | Requested visible, encoded, telemetry and persistence outputs |

The [vendor derivation table](config-derivation-rules.md) offers hypotheses, not
permission to overwrite these contracts. Calibration-file presence alone does
not select INT8, and a performance request does not select a reduced tracker
resolution or remove OSD/media output. An isolated fakesink variant can answer a
component question; preserve the requested graph for end-to-end validation.

## 2. Establish coverage and hardware context

Read [NVTX coverage](nvtx-coverage.md) for the installed toolchain and inspect
actual capture ranges. Classify missing instrumentation as unmeasured. Do not
auto-inject instrumentation unless that implementation work is in scope.

Collect relevant GPU identity, free memory and decoder/encoder/utilization
signals with installed `nvidia-smi` queries. Use
[hardware formulas](hw-ceiling-formulas.md) only for labeled theoretical context;
measured source/model performance determines whether the target is met.

## 3. Run a bounded component experiment when needed

Set a finite batch list, run duration, total experiment budget, memory limit and
maximum rebuild count before execution. Keep every tested shape inside the
engine's supported profile. An import defaults to no additional capacity
rebuilds. Stop at the budget or answered question; do not expand the sweep until
a desired result appears.

For inference isolation, hold model, precision, input resolution and cadence
constant and distinguish queries/s, frames/s, batches/s and per-source FPS.
Compare supported neighboring batches only if that addresses the requested
consumer behavior. A plateau in this variant does not alone choose mux settings
or establish the application's maximum stream count.

## 4. Apply supported configuration changes

Use the baseline contract and experiment to select only settings that remain
within requested quality, output and latency bounds. Read before editing and
preserve unrelated config. Do not apply a whole formula table when evidence
supports one change. Record a missing performance target as a limitation when
meeting it would require an unauthorized capability reduction.

## 5. Verify the requested full graph

Use [Nsight CLI recipes](nsys-cli-recipes.md) for a bounded capture and check which
flags/report names the installed version supports. A typical capture is 30 s;
choose a duration that covers the observed behavior and stays within the task's
budget. Extract only the reports relevant to the question, such as CUDA kernel,
memcpy, NVTX and supported utilization summaries.

Check occupied inputs, each source's progress, expected detections/tracks, and
the requested direct consumers. Measure encoded cadence/drops and WebRTC decoded
frames separately when those outputs are present. NVTX ranges may overlap;
do not sum them as exclusive wall-time or call CPU range duration GPU compute.
An absent range is a measurement gap, not a broken plugin by itself.

## Delivery checklist

- Requested behavior and whether its measured target was met.
- Exact graph/config/engine and representative input provenance.
- Relevant component versus end-to-end measurements with units and duration.
- Changed settings and the evidence supporting them.
- Failed, skipped or uninstrumented checks and remaining limitations.

Link detailed capture/CSV artifacts when useful. Omit unrun stages and empty
report tables; do not present theoretical ceilings as achieved capacity. No
additional tuning loop follows once the requested result is established.
