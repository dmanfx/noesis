---
name: "deepstream-profile-pipeline"
description: "Profile a DeepStream pipeline with Nsight Systems and derive its configs from the measurement. Use when the user asks for an efficient, performant, or profiled pipeline — or to benchmark, tune, or measure FPS."
metadata:
  author: "NVIDIA CORPORATION"
  tags:
    - deepstream
    - profiling
    - nsight-systems
    - nvtx
    - nvidia-smi
    - benchmarking
  languages:
    - bash
    - python
    - yaml
  domain: video-analytics
  team: deepstream-sdk
owner: "NVIDIA CORPORATION"
service: "deepstream"
version: "0.1.0"
reviewed: "2026-04-24"
license: CC-BY-4.0 AND Apache-2.0
compatibility: >
  DeepStream SDK 9.1.0 on the native Ubuntu 24.04 Noesis host. Use the
  installed SDK and the repository's exact CUDA/TensorRT/native artifacts;
  Docker and older DeepStream SDKs are not valid Noesis profiling targets.
  Requires `nsys` (Nsight Systems 2024+) and `nvidia-smi` on PATH. No GUI
  dependency; the skill runs headlessly with `nsys profile` and `nsys stats`.
data_classification: "internal"
---

# DeepStream profiling

Select the mode before reading detailed procedures. Use the actual pipeline,
selected native environment and matched input evidence; a measurement request
does not authorize reduced capability or an engine-capacity sweep.

## Task routes

| Task | Route |
| --- | --- |
| Diagnose or repair FPS/latency in an existing pipeline | [Existing-pipeline diagnosis](#existing-pipeline-diagnosis) |
| Construct a new pipeline with explicit performance requirements | [New-pipeline construction](references/new-pipeline-construction.md) |
| Generate a graph without a performance question | [Pipeline generator](../deepstream-generate-pipeline/SKILL.md) |
| Import a new object detector | [Model import](../deepstream-import-vision-model/SKILL.md); return here only for requested profiling |

## Invariants and evidence

Read [performance invariants](../../../docs/performance_invariants.md) and the
[DS9.1 skill route](../../../DS9/docs/deepstream_9_1_agent_skills.md).
Preserve selected model, precision, resolution, inference cadence, tracking
quality and enabled outputs. Quality/throughput tradeoffs require explicit
approval and matched quality evidence. Exact ordered canonical publication
cohorts cannot use generic leaky/latest-only handling.

Keep decode through OSD in NVMM/GPU memory where supported. Optional work must
remain bounded and must not stall the media path. Do not restart services,
rebuild engines, or modify configuration merely to obtain a profile when the
question can be answered from existing evidence or an isolated bounded run.

Use terminal `nsys profile`/`nsys stats`; no GUI dependency. Verify installed
flags, plugin properties, and report names before execution. Repository pins
and the actual native environment override vendor formulas and examples.

## Existing-pipeline diagnosis

1. Read the launch/configuration and reproduce the affected behavior with a
   bounded live or recorded input. Distinguish checked-in selection from the
   loaded process/config and the observed result.
2. State the question and smallest useful measurement: existing telemetry,
   targeted isolated benchmark, or bounded Nsight capture. Set run duration,
   resource/output limits and stopping conditions before launching work.
3. Check occupied scenes and the affected direct consumer. Per-source progress,
   encoded access-unit cadence/drops and WebRTC decoded frames are separate
   signals; dashboard FPS alone cannot prove end-to-end throughput.
4. Attribute the bottleneck using measured evidence. Missing NVTX coverage means
   that element is not directly measured, not that its cost is zero. NVTX range
   duration is not automatically GPU execution time or a share of wall time;
   concurrent ranges overlap. Use the relevant CUDA/CPU reports.
5. If a repair is requested, change the smallest demonstrated culprit and verify
   the changed producer/direct consumer once. Stop when the behavior is proven
   and no evidence indicates a broader regression.

Report the result, changed files if any, matched inputs, measurements, and
material gaps. Separate isolated inference results, theoretical estimates, and
full application observations. Omit unused report sections; never invent
coverage, GPU time, capacity or success to fill a template.

## Measurement references

Read only the reference needed for the selected question. Verify examples
against the installed native stack and preserve the invariants above.

| Reference | Use when |
| --- | --- |
| [NVTX coverage](references/nvtx-coverage.md) | Determine which elements can be measured directly |
| [Nsight CLI recipes](references/nsys-cli-recipes.md) | Capture and extract supported reports |
| [Hardware ceiling formulas](references/hw-ceiling-formulas.md) | Compare observations with explicitly labeled theoretical estimates |
| [Configuration derivation candidates](references/config-derivation-rules.md) | Evaluate new-graph candidates where requirements leave a setting open; never automatic overrides |
| [New-pipeline construction](references/new-pipeline-construction.md) | Bounded construction measurements and delivery checklist |

<!-- Signing refresh marker. -->
