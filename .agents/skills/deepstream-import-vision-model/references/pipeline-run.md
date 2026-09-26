# DeepStream integration and optional benchmark — Steps 6–7

Use the selected engine to validate the requested detector integration. Engine
capacity exploration and full reports are independent optional phases.

## Preflight: explicit artifacts and supported shapes

Read the build result and consumer configuration. Resolve `MODEL_NAME`,
`MODEL_FILENAME`, the exact `ONNX_FILE` and `ENGINE`, input dimensions `H`/`W`,
labels/`NUM_LABELS`, parser names, and model preprocessing. Do not pick the first
wildcard engine or infer a shape contract solely from its filename. No trtexec
capacity log or `PEAK_GPU_STREAMS` value is required for Step 6.

Use the native host build environment and CUDA 13.2; verify the installed stack
with the repository's host scripts. Do not auto-select the newest CUDA directory
or fall back to another release. Reuse the selected Python environment for the
inspection's actual dependencies.

Select `VIDEO` from the task and its expected detection behavior. When no source
is supplied, the vendor sample is
`/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4`; verify its
presence and relevant target content before treating it as a positive fixture.
Keep negative frames as valid input. Choose a bounded duration/frame range and
fresh output directory so older files cannot satisfy a check.

Set `SMOKE_BS` from the verified engine and consumer contract. The single-stream
vendor helpers require batch 1; if the engine does not support it, use a bounded
consumer at a supported batch without changing the engine to suit a helper.

The parser/config examples below are starting points. Verify model-specific
activation, decoding, normalization, output shape and class semantics against
its exporter and real fixture tensors; model-family names or random inputs alone
do not establish correctness.

## Step 6: DeepStream Integration

```bash
STEP6_START=$(date +%s.%N)
```

### 6a: Inspect Model Output Format

Verify output tensor shapes and value ranges before writing the parser:
```bash
python3 -c "
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession('$ONNX_FILE')
inp = sess.get_inputs()[0]
out = sess.get_outputs()
print(f'Input: {inp.name} shape={inp.shape}')
for o in out: print(f'Output: {o.name} shape={o.shape}')
dummy = np.random.randn(*[d if isinstance(d,int) else 1 for d in inp.shape]).astype(np.float32)
result = sess.run(None, {inp.name: dummy})
for i,r in enumerate(result): print(f'Output[{i}] range: [{r.min():.4f}, {r.max():.4f}]')
"
```

**CRITICAL**: Determine the correct `net-scale-factor` from the output ranges and model family:

| Model expects | net-scale-factor | Notes |
|---------------|-----------------|-------|
| 0–255 input (OpenCV Zoo) | `1.0` | No normalization |
| 0–1 normalized | `0.00392156862745098` (1/255) | Standard |
| ImageNet normalized | `0.01752` + offsets | Rare in DS |

Wrong scale factor = zero detections. Always verify with KITTI dump (Step 6g) before benchmarks.

### 6b: Write Custom Bounding Box Parser

Create `models/$MODEL_NAME/parser/nvdsinfer_custombboxparser_${MODEL_NAME_SAFE}.cpp`:
```cpp
extern "C"
bool NvDsInferParseCustom${PARSER_FUNC_SUFFIX}(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList);

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustom${PARSER_FUNC_SUFFIX});
```

Parser implementation rules:
- Include `nvdsinfer_custom_impl.h` and use `NvDsInferObjectDetectionInfo` (classId, left, top, width, height, detectionConfidence)
- Decode model-specific output format into pixel-space bounding boxes:
  - YOLOX-style `[N, num_anchors, 5+C]`: decode grid offsets, exp(w/h), objectness×class_score
  - SSD-style `[N, num_dets, 6]`: extract class, confidence, normalized → pixel coords
  - YOLO with BatchedNMS: parse keepCount, bboxes, scores, classes from 4 output layers
- **Clip all coordinates** to `[0, networkInfo.width-1]` and `[0, networkInfo.height-1]`
- Use `detectionParams.perClassPreclusterThreshold` for confidence filtering
- **NMS**: Dense heads → `cluster-mode=2` (DeepStream NMS). Fused TRT NMS → `cluster-mode=4`
- **Sanity check for undecoded output**: if bbox values land in [0, 3], the parser is reading grid-space offsets. Most models need `(raw + grid_offset) * stride` for cx/cy and `exp(raw) * stride` for w/h. Verify raw output ranges with Python/ONNX Runtime before writing the parser.
- Reference: `/opt/nvidia/deepstream/deepstream/sources/libs/nvdsinfer_customparser/nvdsinfer_custombboxparser.cpp`; Header: `sources/includes/nvdsinfer_custom_impl.h`

#### Model-family parser patterns

- **DETR / Conditional DETR**: outputs `logits [B, num_queries, num_classes+1]` and `pred_boxes [B, num_queries, 4]`. Boxes are `(cx, cy, w, h)` normalized to `[0,1]` — convert to `(left, top, width, height)` in pixels. Use **softmax** (not sigmoid) on logits. **Background class is the LAST index** (e.g., index 91 for a 92-class DETR, despite `config.json` showing `"0": "N/A"`). Skip the background class when iterating. DETR uses Hungarian matching — NMS is not needed; set `cluster-mode=4` (not `nms-iou-threshold=0.0`, which is a legacy key).
- **OWL-ViT / CLIP-based zero-shot detectors**: outputs `logits [B, num_patches, num_classes]` and `pred_boxes [B, num_patches, 4]`. **Sigmoid** activation (per-class independent scoring, not softmax). Boxes are `(cx, cy, w, h)` normalized `[0,1]`. Use `cluster-mode=2` (NMS with IoU threshold). CLIP preprocessing: `net-scale-factor=0.01459`, `offsets=122.77;116.75;104.09`. Confidence threshold 0.10 works well for general detection; lower to 0.05 for recall-focused tasks.
- **HF RT-DETR preprocessing quirk**: `RTDetrImageProcessor` may have `do_normalize=false` even though `image_mean`/`image_std` fields exist. When `do_normalize=false`, the model expects `[0,1]` scaled input — set `net-scale-factor=1/255` with no offsets. The ONNX export does NOT bake normalization into the first Conv layer. Verify with ONNX Runtime on a real frame before debugging nvinfer.

#### NGC TAO models — use the built-in parser library

NVIDIA NGC TAO models (trafficcamnet, peoplenet, TrafficCamNet Transformer Lite, etc.) ship with TAO-specific parsers pre-compiled into a system library:
- **Library path**: `/opt/nvidia/deepstream/deepstream/lib/libnvds_infercustomparser.so` — NOT `libnvds_infercustomparser_tao.so` (even if the NGC YAML config suggests it).
- Custom parse function names: `NvDsInferParseCustomDDETRTAO`, `NvDsInferParseCustomRTDETRTAO`, etc.
- **No custom parser compilation needed** — point `custom-lib-path` at the system library and `parse-bbox-func-name` at the TAO function.
- KITTI dump from `deepstream-app` may emit zero-valued bbox coordinates for DETR/RT-DETR parsers even when detections are correct. Verify visually with JPEG frame extraction instead.

### `network-type` vs `model-type` — use `network-type=0`

- `model-type` is a legacy/unknown key — nvinfer ignores it with a warning.
- `network-type=0` (Detector) is required to invoke `parse-bbox-func-name`.
- `network-type=100` (Other) does NOT invoke the custom bbox parser — it requires `output-tensor-meta=1` for external post-processing.
- **Symptom of the wrong key**: custom parse function is never called (zero detections, no parser debug output) — check that `network-type=0` is set.

### 6c: Create Makefile

Write `models/$MODEL_NAME/parser/Makefile` using Python to guarantee literal TAB characters in recipe lines (heredoc in bash can produce spaces, which break make):
```bash
python3 - << EOF
model = '$MODEL_NAME'
model_safe = '$MODEL_NAME_SAFE'
content = (
    "DEEPSTREAM_DIR ?= /opt/nvidia/deepstream/deepstream\n"
    "CUDA_VER ?= 13.2\n"
    "CC := g++\n"
    "CFLAGS := -Wall -std=c++11 -shared -fPIC\n"
    "CFLAGS += -I\$(DEEPSTREAM_DIR)/sources/includes -I/usr/local/cuda-\$(CUDA_VER)/include\n"
    "LIBS := -lnvinfer\n"
    "LFLAGS := -Wl,--start-group \$(LIBS) -Wl,--end-group\n"
    f"SRCFILES := nvdsinfer_custombboxparser_{model_safe}.cpp\n"
    f"TARGET_LIB := libnvdsinfer_{model_safe}_parser.so\n"
    "\n"
    "all: \$(TARGET_LIB)\n"
    "\$(TARGET_LIB): \$(SRCFILES)\n"
    "\t\$(CC) -o \$@ \$^ \$(CFLAGS) \$(LFLAGS)\n"   # TAB required by make
    "clean:\n"
    "\trm -rf \$(TARGET_LIB)\n"                       # TAB required by make
)
with open(f'models/{model}/parser/Makefile', 'w') as f:
    f.write(content)
print(f"Makefile written: models/{model}/parser/Makefile")
EOF
```

### 6d: Build Parser Library

```bash
make -C models/$MODEL_NAME/parser \
  DEEPSTREAM_DIR=/opt/nvidia/deepstream/deepstream \
  CUDA_VER=$CUDA_VER

# Verify the symbol is exported
nm -D models/$MODEL_NAME/parser/libnvdsinfer_${MODEL_NAME_SAFE}_parser.so | grep NvDsInferParseCustom
```

### 6e: Create nvinfer Config File

```bash
cat > models/$MODEL_NAME/config/config_infer_primary_${MODEL_NAME}.txt << EOF
[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file=../model/${MODEL_FILENAME}.onnx
model-engine-file=${ENGINE_FROM_CONFIG}
labelfile-path=labels.txt
batch-size=${SMOKE_BS}
network-mode=${NETWORK_MODE}
num-detected-classes=${NUM_LABELS}
process-mode=1
interval=0
gie-unique-id=1
network-type=0
custom-lib-path=../parser/libnvdsinfer_${MODEL_NAME_SAFE}_parser.so
parse-bbox-func-name=NvDsInferParseCustom${PARSER_FUNC_SUFFIX}
# 2=DeepStream NMS (dense heads: YOLO, SSD). Use 4 if engine has fused NMS output
cluster-mode=2
infer-dims=3;${H};${W}
maintain-aspect-ratio=1

[class-attrs-all]
topk=200
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
EOF
```

> **Path note**: All paths are relative to the `config/` directory where this file lives.
> `ENGINE_FROM_CONFIG` is the exact selected engine path relative to that directory;
> `SMOKE_BS` is supported by both engine and smoke consumer. `NETWORK_MODE` matches
> the selected engine precision. Resolve these before writing the config.
> `net-scale-factor` defaults to `1/255` — update to `1.0` if the model expects 0–255 input (verify via Step 6a).

Verify label count matches:
```bash
echo "labels.txt: $NUM_LABELS classes -> num-detected-classes=$NUM_LABELS"
```

### 6f: Bounded visual and metadata validation

Run the configured producer and its direct metadata/media consumer with the
selected fixture. Inspect the vendor helper before using it: only its canonical
NVENC path is allowed on Noesis. No `theoraenc`, `x264enc`, or `openh264enc`
fallback is authorized. Missing NVENC is a canonical-path failure; report the
affected check as failed/incomplete and retain independently valid metadata
checks. Do not silently substitute a different validation pipeline after a crash.

Confirm source/frame progress and the run's exit status. Extract a bounded
sample of the **exact output from this run**, including known positive and
negative frames. Check labels, finite confidence/coordinates, frame geometry,
and class/box placement through the direct consumer.

### 6g: Detection validity and fixture expectations

Follow [detection-validation.md](detection-validation.md). A detection-bearing
frame fraction is descriptive; there is no universal 90% occupancy gate.
Positive fixture behavior must still be demonstrated before declaring the new
detector integration validated.

`gie-kitti-output-dir` belongs in the `deepstream-app` `[application]` section,
not an `nvinfer` config. The skill's `scripts/deepstream/ds-kitti-dump.sh` helper
creates that application config. Inspect its run/timeout behavior and verify
frame progress independently: output-file count is not a substitute for a
successful run or processed-frame coverage. Use a fresh output directory and a
source range whose expected target presence is established.

Missing files require inspection of application progress, writer configuration,
and writer semantics. Empty files can represent valid negative frames. A
known-positive mismatch requires investigation of source/preprocessing, tensor
semantics, thresholds, parser and downstream writer evidence; it does not by
itself identify a broken parser. Keep the integration incomplete until the
expected behavior is demonstrated, without guessing a repair.

## Step 7: Optional application benchmark

Only run when requested. Route to
[the profiling skill](../../deepstream-profile-pipeline/SKILL.md) and the bounds in
[engine-build.md](engine-build.md#step-5--optional-bounded-benchmark). Start from
the actual consumer/source configuration and preserve enabled outputs and
quality settings. Record finite run counts, duration, stream/batch ceilings,
rebuild and resource budgets before any sweep. Do not equate stream count with
engine maximum batch or infer capacity from one aggregate FPS value.

Legacy benchmark helpers may strip output stages or assume batch equals source
count. Inspect those assumptions before use; an isolated run measures only that
variant. Report end-to-end per-source progress, encoded cadence/drops, and WebRTC
decoded frames separately where those paths are in scope. Every claimed source
must meet the requested FPS with the expected detections and outputs.

## Delivery

Report the exact engine/parser/config, fixture and frame range, checks completed,
remaining failures and unmeasured phases. Benchmark/report generation is not a
prerequisite for successful scoped integration. Conversely, throughput or an
output file alone cannot establish detector correctness. Continue to the report
reference only if a report was requested; populate it with measured facts.
