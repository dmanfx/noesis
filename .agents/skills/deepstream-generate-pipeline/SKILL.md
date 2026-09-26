---
name: deepstream-generate-pipeline
description: Generate a new DeepStream GStreamer graph or explicitly requested pipeline configuration. Resolve requirements from the task, checkout, and installed environment. Use deepstream-dev for existing-runtime repairs and deepstream-profile-pipeline for existing-pipeline performance diagnosis.
owner: NVIDIA CORPORATION
service: deepstream
version: 1.0.0
reviewed: 2026-04-27
license: CC-BY-4.0 AND Apache-2.0
---

# DeepStream Pipeline Builder

Generate and validate `gst-launch-1.0` pipelines for NVIDIA DeepStream SDK by resolving pipeline requirements from the task and existing evidence, then assembling the pipeline using a standalone BM25 retrieval backend with structural metadata boosting (similarity search over 270+ verified pipelines, zero external dependencies).

## Prerequisites

- **Python:** 3.8+ (stdlib only — no pip packages required)
- **DeepStream SDK:** Installed at `/opt/nvidia/deepstream/deepstream/` (for `gst-inspect-1.0` validation and element verification)
- **GStreamer:** `gst-launch-1.0` and `gst-inspect-1.0` on `PATH` (installed with DeepStream)
- **Platform:** x86 dGPU (T4, A100, L40, RTX, etc.) or aarch64 — Jetson (Orin, Xavier, Nano) / SBSA (Grace, GH200)

## Usage Examples

```text
# Fully specified — skips most questions
detect and track on 4 rtsp streams and display on jetson

# Partially specified — asks remaining questions
give me a pipeline to infer on an image

# Minimal — resolves repository/environment facts, then asks consequential unknowns
build a pipeline
```

## Supported Configurations

| Parameter | Options |
| --- | --- |
| **Input** | Local video (.mp4/.h264/.h265), local image (.jpg/.png), RTSP stream, USB camera, test pattern |
| **Inference** | None, primary (nvinfer), primary+secondary, with preprocessor, Triton (nvinferserver) |
| **Tracker** | None, NvDCF, IOU, NvSORT, DeepSORT |
| **Sink** | Display (dGPU/Jetson), save (JPG/PNG/MP4/H264), RTSP out, fakesink |
| **Platform** | x86 dGPU (T4, A100, L40, RTX, etc.) or aarch64 — Jetson (Orin, Xavier, Nano) / SBSA (Grace, GH200) |
| **Extras** | Resize, rotate/flip, crop, color format conversion |

## Scripts

| Script | Purpose |
| --- | --- |
| `scripts/generate_pipeline.py` | BM25 retrieval engine — scores and ranks pipelines from `data/data.csv`. Supports `--format {json,compact,summary}` (default `json`) |
| `scripts/validate_pipeline.py` | 4-stage validator: syntax, elements, properties, live parse. Supports `--format {json,summary}` (default `json`) |
| `scripts/lint_data.py` | Data quality linter for the pipeline CSV (`--fix` to auto-repair) |

## Workflow

### Step 1 — Collect Pipeline Requirements

> **You MUST `Read references/requirement-extraction.md` before doing this step.**
> It contains the query-inference table, compound-extraction examples, the full
> `AskUserQuestion` question bank (with the default-first ordering contract), the
> automatic-OSD and extras/flip-method rules, and the dynamic question-reduction
> examples that this step depends on. Apply them under repository policy.

**Order of operations:**

1. **Resolve known parameters** from the user's request, applicable repository policy, existing configs, and installed environment. Record evidence for source count, inference, tracker, sink, platform, and extras; never re-ask facts already established.
2. **Ask only consequential unknowns** using the available clarification tool, within its question limits. The reference question bank is a vocabulary, not a mandatory questionnaire or a requirement for a tool named `AskUserQuestion`. Continue independent work while awaiting required answers.
3. **Use stated assumptions only for optional choices.** Missing required source/model evidence, dismissed questions, and elapsed time do not authorize guessed inputs or capability reductions. Briefly state resolved requirements and material assumptions.

Follow the inference table, question bank, and OSD/extras rules in
`references/requirement-extraction.md` to decide which questions to ask and how to
place transform elements, then proceed to Step 2.

### Step 2 — Build the Natural Language Query

From the user's answers, construct a single descriptive query string. Follow this pattern:

```text
Please provide a GStreamer pipeline that [operation] on [num_sources] [input_type] [input_detail] [tracker_detail] and [output_action] [platform_detail]
```

**Examples of constructed queries:**

| User Selections | Constructed Query |
| --- | --- |
| Local video, 1 source, Primary detector, No tracker, Display, dGPU | "Please provide a GStreamer pipeline that performs primary inference on a single mp4 video and displays the output" |
| RTSP, 4 sources, Primary+Secondary, NvDCF, Save MP4, dGPU | "Please provide a GStreamer pipeline that performs primary and secondary inference with NvDCF tracker on 4 RTSP streams and saves output to MP4 file" |
| Local video, 2 sources, Primary with preprocessor, IOU, Display, Jetson | "Please provide a GStreamer pipeline that performs preprocessing before primary inference with IOU tracker on 2 mp4 streams and displays the output on Jetson" |
| Local image, 1 source, None, No tracker, Save file, dGPU, Rotate 90° cw | "Please provide a GStreamer pipeline that rotates a single jpg image 90° clockwise before processing and saves it to a file" |
| Local video, 3 sources, Primary detector, NvDCF, Save MP4, dGPU, Rotate 180° | "Please provide a GStreamer pipeline that rotates 3 mp4 videos 180° before primary inference with NvDCF tracker and saves output to MP4 file" |

### Step 3 — Run the Pipeline Generator Script

Execute the backend script with the constructed query and user parameters:

```bash
python3 <skill-path>/scripts/generate_pipeline.py \
  --query "<constructed_query>" \
  --source-type "<Local video file|Local image file|RTSP stream|USB camera|Test pattern>" \
  --num-sources <N> \
  --inference "<None|primary|primary+secondary|primary+preprocess|primary+secondary+preprocess|primary-triton|primary+secondary-triton>" \
  --tracker "<none|NvDCF|IOU|NvSORT|DeepSORT>" \
  --sink "<display|display-jetson|save-jpg|save-png|save-mp4|save-h264|rtsp-out|fakesink>" \
  --platform "<dGPU|Jetson|SBSA>" \
  --extras "<none|resize|rotate|crop|color-convert|osd>" \
  --format compact
```

> **Always pass `--format compact`.** The `compact` mode returns only confidence + the top retrieved pipeline (~25 lines), instead of dumping all 10 retrievals as ~150 lines of JSON in the chat. The `json` mode (default for backward compat) is only useful when debugging the retriever directly. A `summary` mode (single human-readable line) also exists for non-Claude callers.

The script will (zero external dependencies — pure Python stdlib):

1. Load the pipeline dataset (270+ verified DeepStream pipelines)
2. Extract structural metadata from each pipeline (platform, source type, sink type, inference mode, tracker, stream count)
3. Score with BM25 (document-length-normalized) + domain-specific synonym expansion on both queries and documents
4. Apply structural boosting — results matching the user's platform/source/sink/inference get boosted, mismatches get penalized
5. Return the top-K results as JSON with a `confidence` field (`high`/`medium`/`low`) based on the top score
6. Claude uses these retrieved examples + the assembly rules below to construct the final pipeline

When `confidence` is `low`, rely more heavily on the assembly rules below rather than the retrieved examples.

### Step 4 — Validate the Pipeline

Before presenting, run the validation script to catch syntax errors, unknown elements, and linking issues:

```bash
python3 <skill-path>/scripts/validate_pipeline.py "<assembled_pipeline>" --format summary
```

> **Always pass `--format summary`.** Summary prints a single status line (e.g. `valid · 11 elements · 0 warnings · live-parse skipped (multi-stream)`), with errors/warnings indented underneath only if present. The default `json` mode emits ~40 lines of structured output and is only useful for programmatic callers.

The validator performs 4 checks:

1. **Syntax check** — unbalanced quotes, empty pipe segments, missing source/sink
2. **Element check** — verifies each element exists via `gst-inspect-1.0`
3. **Property check** — validates known properties for DeepStream elements
4. **Live parse check** — uses `gst-launch-1.0` itself to construct the pipeline graph (with fakesrc/fakesink substituted), catching linking errors and pad mismatches. **Automatically skipped for multi-stream pipelines** (those with named pad refs like `m.sink_0`) since fakesrc cannot negotiate caps through named pads.

If validation fails, make at most two evidence-based fixes and revalidate each
changed graph. If errors remain, stop with **failed validation** and the actual
remaining errors. Do not infer that other checks passed. Warnings and skipped
checks retain their individual status; retrieval confidence is not validation.

### Step 5 — Deliver the Result and Its Evidence

Check the actual local input, model/config, tracker/library, and output-parent
paths referenced by the assembled graph. Do not preflight unrelated default
samples or substitute a sample for a missing requested source. Report unverified
remote sources explicitly. A graph parse does not prove source access, inference,
tracking, or end-to-end output.

Lead with the result: runnable artifact, partially verified draft, or failed
validation. Link a requested artifact or show a readable shell command. Use
portable repository paths or documented environment variables for saved files;
quote paths and allow line continuations. Preserve the selected configuration.

State checks actually completed, relevant warnings/skips, and what remains
unproven. Include a stage table only when it helps explain a complex graph; omit
canned suggestion lists and mandatory follow-up questions. For examples, read
[references/output-format.md](references/output-format.md).

Before delivery, verify that no success badge or ready-to-run claim contradicts
a failed check, missing input, unresolved placeholder, or skipped runtime smoke.
A reviewable failed draft may still be delivered, clearly labeled with the
blocking error and the next necessary action.

### Step 6 — Apply Requested Refinements or Save the Artifact

Create or update the script/config when the task requests or implies a file
artifact. Follow the repository's path and editing conventions; report its
location and relevant checks. A request for a command alone can be satisfied in
chat without creating a file. For refinements, apply clear changes directly and
resolve only new consequential unknowns; do not repeat the requirements interview.

---

## Pipeline Assembly Rules

When the script is not available or fails, assemble the pipeline using the rules in [references/assembly-rules.md](references/assembly-rules.md). These rules cover source elements, multi-stream patterns, inference chains, tracker configs, sink elements, and extra operations. They also serve as validation for script output.

---

## Error Handling

| Failure | Cause | Recovery |
| --- | --- | --- |
| `generate_pipeline.py` returns `confidence: low` | Query doesn't match any pipeline in the dataset closely | Rely on the assembly rules in this skill instead of retrieved examples |
| `validate_pipeline.py` reports unknown element | GStreamer/DeepStream not installed or not on `PATH` | Verify the selected native environment; repair only the confirmed missing dependency within task scope |
| Validation fails after 2 retries | Unusual element combination or linking issue | Deliver a failed draft with the remaining errors and actual per-check status; do not claim readiness |
| Script not found at `<skill-path>/scripts/` | Skill not installed correctly or path misconfigured | Verify the skill directory is symlinked into `.claude/skills/` or `.cursor/skills/` |

## Testing

For changes to retrieval or validation code, run the affected tests. Documentation-only changes use repository documentation checks. The full suite is available when retrieval quality or validator correctness is affected:

```bash
python3 -m unittest discover -s <skill-path>/tests -v
```

The suite includes:

- **Unit tests** for the BM25 retriever (tokenizer, synonym expansion, metadata extraction, scoring)
- **Unit tests** for the validator (syntax, structure, property, named-pad checks)
- **Golden regression tests** — 20+ query→expected-result pairs ensuring retrieval quality doesn't regress
- **Data quality linter** — checks the CSV for duplicates, syntax issues, and structural bugs:

```bash
python3 <skill-path>/scripts/lint_data.py          # report issues
python3 <skill-path>/scripts/lint_data.py --fix     # auto-fix and overwrite
```

---

## Security, Limitations & Notes

Security posture, known limitations, and operational notes are documented in `references/security-and-limitations.md`. Read that file when you need details on subprocess safety, input validation, platform/SDK requirements, the multi-stream dry-run caveat, or sample-path/config-file reminders.

