# Configuration candidates from measured evidence

Use with [new-pipeline construction](new-pipeline-construction.md), after resolving
the selected consumer's quality, output and latency requirements. These are
questions to test within that contract, not closed-form settings to apply to
every pipeline. Existing-runtime diagnosis changes only a demonstrated culprit.

## Inputs

Record source count/cadence/resolution, model input/profile/precision, tracker
configuration, required outputs, the target latency/FPS, and matched component
and application observations. Separate theoretical hardware limits from measured
results. Set finite batch, run, rebuild, time and resource bounds before a sweep.

## R1 — Streammux

Derive mux batch and timeout from actual source batching and latency requirements.
A measured inference plateau is evidence for a candidate batch, not proof that
all sources can meet their deadline. Keep mux dimensions and padding consistent
with the product geometry/output contract. Read memory-type semantics in the
installed plugin; a numeric flag alone does not prove zero-copy behavior.

## R2 — Inference

Use the exact selected engine and its supported shapes. Verify the relation
between mux and inference batches for this graph rather than forcing a universal
equality. Keep model precision, input resolution, preprocessing and inference
cadence fixed. Calibration-file presence alone does not authorize INT8, and a
performance request does not authorize changing FP32/FP16 or skipping inference.
Verify model-specific dynamic-input configuration with the installed SDK.

A missing or incompatible engine is a build prerequisite. Route it to the
native maintenance/import workflow; do not trigger a cold engine build in a
profile run or guess a `fakesrc` graph to create it.

## R3 — Decoder

Keep the native device-memory path and source resolution. Size additional
surfaces from measured pressure, bounded memory and latency. Verify each property
on the element that actually owns it; source-bin and decoder properties are not
interchangeable. File looping is useful only for an authorized isolated benchmark
and must not disguise a missing live-source or temporal validation check.

## R4 — Tracker

Preserve the selected tracker algorithm, dimensions, library and configuration.
Profile representative occupied scenes, including motion and occlusion, before
attributing costs. A smaller tracker resolution or performance preset is a
quality tradeoff requiring explicit approval and matched tracking evidence.

### R4a — Inference interval with tracking

Tracker-ID retention alone does not establish detection recall, StableID
correctness, motion accuracy, or safe gaps in inference. Keep the selected
interval. If a tradeoff experiment is explicitly authorized, define its fixture,
quality metrics, limits and acceptance thresholds first; assess all affected
consumers. Do not step the interval upward until an arbitrary retention
percentage holds or use tracker-local IDs as product-identity proof.

## R5 — Queues

Assign each queue a finite capacity from work rate, service time, memory and
freshness needs. Optional display/persistence/network work must not stall the
always-on media path. Verify bounded worker behavior at the actual boundary;
one generic queue-size multiplier does not establish this.

Canonical tracking/world/BEV publications remain one exact ordered cohort and
cannot be made leaky or coalesced. Optional consumers may degrade their own
freshness according to the documented product boundary. Do not apply a generic
Kafka or file-sink leaky/nonleaky policy to all outputs.

## R6 — OSD, tiler and visible sinks

Preserve required outputs, even when the performance request does not restate
them. An isolated fakesink variant can measure one component, but its throughput
cannot establish full-graph performance. Exercise the requested media output
and direct consumer in the final bounded check.

## R7 — Capacity and bottleneck reporting

An inference-only frames/s divided by target FPS is a sizing estimate. Hardware
decode/bandwidth formulas are theoretical estimates. Their minimum does not
establish the application's maximum stream count.

Claim measured capacity only for the source count, input occupancy, model,
configuration and outputs actually exercised. Each source must meet the target;
check encoded cadence/drops and WebRTC decoded frames separately when relevant.
If instrumentation cannot attribute the bottleneck, state the uncertainty and
choose the next bounded measurement. Do not recommend an algorithm, resolution,
precision or output reduction merely because a theoretical ceiling is low.

## Before the application measurement

Verify the selected artifacts and supported shapes, required output branches,
queue ordering/bounds, and native plugin/library origins. Separate startup/build
time from steady-state evidence without hiding startup failures. Stop on a
failed prerequisite and report it; do not rewrite the graph to bypass it.
