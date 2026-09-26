---
name: deepstream-import-vision-model
description: >
  Import a new object-detection model from HuggingFace or NVIDIA NGC into
  DeepStream. Use for requested detector acquisition, ONNX export, parser
  integration, and native TensorRT realization. Do not use for existing-engine
  maintenance, non-detector models, or general runtime debugging.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA CORPORATION
  version: 1.2.2
---

# DeepStream Import Vision Model

When this skill is active, **read the relevant reference document before starting each phase**. Do not rely on memory — reference documents contain exact script paths, bash variable conventions, log filename contracts, and critical parsing rules.

**Current scope:** Object detection models only. Fail fast on classification, segmentation, or other architectures detected in `config.json`.

## Noesis scope and native-host boundary

Use this route only for a new detector import. For maintenance of selected
engines or non-detector models, use `deepstream-dev`,
`DS9/docs/runtime_host_boundary.md`, and
`DS9/scripts/run_canonical_engine_maintenance_host.sh`; inspect the selected
artifact contract before running maintenance. Do not substitute architectures.

Repository pins govern every reference and helper: native DeepStream 9.1,
CUDA 13.2, TensorRT 10.16.0.72. Use canonical DS9 artifact/config paths from the
checkout and its manifests. Generic output layouts below are examples for new
standalone imports, not authority to relocate existing artifacts.

Run only phases required by the authorized import. Direct producer/consumer
validation is required where practical; capacity sweeps, charts, and HTML/PDF
reports are conditional on the requested deliverable. Install only dependencies
needed by those phases. Never execute a helper's software-encoder fallback on
Noesis; inspect its branch before use and report a missing canonical encoder.
A fallback in a vendor reference or script does not authorize it.

## Pipeline Overview

| Step | Phase | Reference | What it does |
|------|-------|-----------|--------------|
| 1–3 | Model Acquire | [references/model-acquire.md](references/model-acquire.md) | Browse HF/NGC, detect format, download ONNX or export SafeTensors |
| 4–5 | Engine Build  | [references/engine-build.md](references/engine-build.md) | Build the consumer-required profile; measure only requested supported batches |
| 6–7 | DS Pipeline   | [references/pipeline-run.md](references/pipeline-run.md) | Parser/config and expected-fixture validation; optional bounded application benchmark |
| 8   | Report        | [references/report-generation.md](references/report-generation.md) | 5 charts, HTML, PDF benchmark report |

Complete the authorized phases autonomously without per-step reconfirmation.

## Pre-flight Checks

Select checks and dependencies by phase:

| Phase | Preflight |
| --- | --- |
| Acquire/export | Exact model architecture/format and the selected utility environment's required download/export packages |
| Engine build | Exact model/consumer profile, native builder/library versions, available GPU resources and bounded build plan |
| Integration | Selected compatible engine/parser, installed plugins/application, expected positive/negative fixture and supported batch |
| Benchmark/report | Requested measurement bounds and only the required measurement/chart/report tools |

Reuse the configured utility environment (the shared `build/.venv_optimum` for
standalone exports) before creating one. Install packages only when the selected
phase needs them; downloading an ONNX file does not require an exporter, SDK
wheel, plotting library or PDF renderer. A runtime import failure first requires
checking the selected interpreter and module origin. Do not install packages
merely because the full-report example uses them.

For native execution verify the pinned builder/runtime and installed plugins.
Use `trtexec --help` for version/flag inspection; no engine build is needed for
that check. Validate the actual fixture path. When the task supplies no source,
check the documented vendor sample and establish its target content; a missing
sample is not permission to silently pick a different recording.

## Example output structure for a new standalone import

Use this layout only when no canonical artifact contract already specifies paths.

```
models/{model_name}/
  model/           <- ONNX file(s)
  parser/          <- .cpp, Makefile, .so
  config/          <- nvinfer config, ds-app config, labels.txt
  scripts/         <- run helper scripts
  benchmarks/
    engines/       <- exact selected engine, optional timing cache, build logs
    b1/            <- BS=1 log only if requested and supported
    b{MAX_BS}/     <- requested supported-batch log
    ds/            <- DS benchmark logs
  reports/         <- benchmark_report.md, .html, .pdf, benchmark_data.json
    charts/        <- chart_*.png (5 charts)
  samples/         <- output .mp4 from the canonical encoder, test frames
    kitti_output/  <- KITTI detection .txt files
```

Create only the directories needed by the selected phases; this layout does
not require a benchmark or report.

## Critical Rules

1. **Exact engine contract** — record the selected artifact path and supported input/profile shapes. Preserve the consumer manifest naming; use a dynamic-batch name only for a dynamic-batch engine. Never select an engine by first wildcard match.
2. **Consumer batch contract** — derive stream count, mux batch, and inference batch separately from the selected graph and supported engine profile. A helper that assumes equality is usable only when that contract holds.
3. **Benchmark logs** — legacy report readers expect `trtexec_b1.log`, `trtexec_b${MAX_BS}.log`, `ds_s${N}_run1.log`, and `ds_s${N}_run2.log`. Preserve those names only for requested, actually performed measurements in a fresh task output directory; report unmeasured fields explicitly.
4. **Parser zero-init** — always `NvDsInferObjectDetectionInfo obj = {};`. Required for DeepStream OBB support; bare `obj;` leaves `rotation_angle` uninitialized, causing tilted bounding boxes.
5. **Detection validation** — verify execution/frame coverage, output schema/geometry, and known positive/negative fixture expectations independently. Detection-bearing-frame fraction has no universal pass threshold. See [detection validation](references/detection-validation.md); missing positive evidence leaves integration incomplete.
6. **Reuse the selected utility environment** — standalone exports normally share `build/.venv_optimum`; do not create per-model environments without a task-specific dependency/isolation need. SDK dependencies are conditional on SDK execution.
7. **Inference-only measurements** — `--noDataTransfers` isolates device inference. It does not prove the consumer's memory path, zero-copy behavior, or end-to-end throughput.
8. **Report HTML+PDF (when requested)** — use `scripts/report/md-to-html-pdf.py` relative to this skill. Never write a custom HTML generator or call `wkhtmltopdf` directly.
9. **Object detection only** — reject non-detection architectures from `config.json` before building anything.
10. **Canonical encoder only** — do not use `theoraenc`, `x264enc`, or `openh264enc` as a Noesis fallback. If NVENC is unavailable, report the canonical-path failure; do not silently substitute an encoder or claim video validation passed.
11. **Video source (MANDATORY)** — default is always `sample_720p.mp4` (1280×720). Never autonomously substitute `sample_1080p_h264.mp4` or any other file. Only use a different video when the user explicitly provides a path (via `DS_VIDEO` env var or script argument).

## Pipeline Timing

Wrap every step:

```bash
STEP_START=$(date +%s.%N)
# ... step commands ...
STEP_END=$(date +%s.%N)
STEP_DURATION=$(echo "$STEP_END - $STEP_START" | bc)
echo "[Step N] completed in ${STEP_DURATION}s"
```

Track `PIPELINE_START` (before Step 1) and `PIPELINE_END` (after Step 8). Report all durations in the benchmark report.

## Report output (only when the full report is requested)

1. `benchmark_report.md` — markdown source (12 mandatory sections)
2. `benchmark_report.html` — styled HTML (charts base64-inlined, no local file access)
3. `benchmark_report_{model_name}.pdf` — via `md-to-html-pdf.py`; verify charts are embedded by counting `data:image/png` occurrences in the HTML output: `grep -o 'data:image/png' benchmark_report.html | wc -l` should equal 5

Run charts and report scripts with the shared venv active: `source build/.venv_optimum/bin/activate`.

## Reference Documents

**IMPORTANT**: Read the relevant reference before starting each phase. Do NOT generate code from memory.

| Document | Use When |
|----------|----------|
| [references/model-acquire.md](references/model-acquire.md) | Steps 1–3: HF/NGC URL parsing, format detection, ONNX download, SafeTensors export, label extraction |
| [references/engine-build.md](references/engine-build.md) | Steps 4–5: consumer-bound engine profiles and optional bounded benchmarks |
| [references/pipeline-run.md](references/pipeline-run.md) | Steps 6–7: parser/config, supported-batch fixture validation, and optional application benchmark |
| [references/report-generation.md](references/report-generation.md) | Step 8: benchmark_data.json, 5 charts, 12-section markdown report, HTML + PDF |

## Scripts

Located in `scripts/`.

| Script | Phase | Purpose |
|--------|-------|---------|
| `model/hf-list-files.sh` | 1–3 | List HuggingFace repo files |
| `model/hf-download-config.sh` | 1–3 | Download config.json from HF |
| `model/ngc-list-files.sh` | 1–3 | List NGC model files |
| `model/ngc-download.sh` | 1–3 | Download NGC model archive |
| `model/safetensors-to-onnx.sh` | 1–3 | Export SafeTensors → ONNX via optimum-cli |
| `model/inspect-onnx.py` | 1–5 | Inspect ONNX input/output shapes |
| `model/make-static-batch-onnx.py` | 4–5 | Bake batch dim into ONNX |
| `model/cleanup.sh` | Any | Remove staging dirs, preserve shared venv |
| `engine/benchmark-trtexec.sh` | 4–5 | Run trtexec with standard flags |
| `deepstream/ds-single-stream.sh` | 6–7 | Vendor validation helper; inspect first and exclude its fallback branches on Noesis |
| `deepstream/ds-sweep.sh` | 6–7 | 2-phase batch size sweep |
| `deepstream/benchmark-ds.sh` | 6–7 | Fixed-stream DS benchmark |
| `deepstream/ds-kitti-dump.sh` | 6–7 | KITTI detection dump via deepstream-app |
| `deepstream/ds-perf-run.sh` | 7 | Step 7c two-run benchmark — wraps `deepstream-app` with `enable-perf-measurement=1`, writes fixed-name log for the report parser |
| `deepstream/extract-frame.sh` | 6–7 | Extract sample frames from output video from the canonical `.mp4` path |
| `report/generate-benchmark-charts.py` | 8 | Generate 5 benchmark PNG charts |
| `report/md-to-html-pdf.py` | 8 | Markdown → styled HTML → PDF (canonical benchmark report path) |
| `report/md-to-pdf.sh` | Any | Markdown → PDF via pandoc/pdflatex — for design docs and references only, NOT for benchmark reports (use md-to-html-pdf.py for those) |
| `report/report-style.css` | 8 | CSS for HTML report |
| `report/render-mermaid-for-pdf.py` | 8 | Mermaid diagram → PNG |
| `report/mermaid-puppeteer.json` | 8 | Vetted Puppeteer config for Mermaid (sandboxed; non-root) |
| `report/mermaid-puppeteer-root.json` | 8 | Vetted Puppeteer config for Mermaid (used when running as root) |

## Quick Error Reference

| Error | Fix |
|-------|-----|
| Tilted/diagonal bounding boxes | Parser struct not zero-initialized — use `NvDsInferObjectDetectionInfo obj = {};` |
| Zero KITTI files | Check source progress, writer semantics and `[application] gie-kitti-output-dir`; files alone do not identify a parser failure |
| Engine rebuilds every DS run | `model-engine-file` path wrong — check relative path from `config/` dir |
| `setDimensions` negative dims | Add `infer-dims=3;H;W` to nvinfer config for dynamic ONNX models |
| `--memPoolSize` workspace 0.03 MiB | Use `M` suffix not `MiB` — e.g. `--memPoolSize=workspace:32768M` |
| ForeignNode build failure (DETR) | Use dynamo export path or run `onnxsim` — see references/engine-build.md |
| Zero detections | Compare known-positive fixture expectations, then trace preprocessing, tensors, parser and writer; negative frames may validly be empty |
| `No module named 'pyservicemaker'` | Verify selected runtime interpreter/import origin; repair its matching native SDK dependency only if it executes SDK code |

<!-- Signing refresh marker. -->
