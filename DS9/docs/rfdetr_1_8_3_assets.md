# RF-DETR 1.8.3 DS9 assets

Status date: 2026-07-25

## Release and capability decision

The reviewed open-source export authority is RF-DETR `1.8.3`, tag `1.8.3`,
commit `3bd6bffbcb13cac3a5b1c37da5a0fd5453b50c86`. This is the newest stable
RF-DETR release at the status date.

The current DS9 YOLO model set provides object detection, instance
segmentation, and person pose/keypoints. RF-DETR has the following matching
families:

- detection: Nano, Small, Medium, Large, XLarge, and 2XLarge;
- instance segmentation: Nano, Small, Medium, Large, XLarge, and 2XLarge;
- keypoints: one preview checkpoint for COCO 17-keypoint person pose.

RF-DETR does not provide current model families matching YOLO classification,
oriented boxes, open-vocabulary detection, or tracking. Those capabilities are
not represented by substitute or legacy RF-DETR assets.

Detection XLarge and 2XLarge are distributed through `rfdetr-plus==1.0.2`
under PML-1.0. They require the user to accept the Platform Model License and
use a Roboflow account. They remain recorded in the model matrix but disabled;
Noesis does not accept that license on the user's behalf.

Deprecated or superseded checkpoints are excluded:

- RF-DETR Base, Base v2, and Base O365;
- the old RF-DETR Large checkpoint, superseded by `rf-detr-large-2026.pth`;
- RF-DETR Segmentation Preview.

The complete reviewed matrix, including official URLs, expected byte counts,
MD5 values, classes, resolutions, licenses, and tensor contracts, is
`DS9/config/rfdetr_1_8_3_models.json`.

## Export contract

All exports preserve the official RF-DETR 1.8.3 raw query tensors. No
continuation of the old DS9 hand-selected top-K caps is inferred.

- ONNX opset: 17
- input tensor: `input`
- batch: static 3, matching the current DS9 mux/inference asset contract
- shape: each model's native published square resolution
- detection outputs: `dets`, `labels`
- segmentation outputs: `dets`, `labels`, `masks`
- keypoint outputs: `dets`, `labels`, `keypoints`

The keypoint preview checkpoint uses the legacy background-first schema
`[0, 17]`. Its exact static-batch output contract is:

```text
dets      [3, 100, 4]
labels    [3, 100, 2]
keypoints [3, 100, 34, 8]
```

RF-DETR 1.8.3 reports four obsolete `keypoint_head.keypoint_proj` checkpoint
keys as unused. The current model consumes its current `keypoint_embed` and
transformer keypoint weights; the four reported keys do not have corresponding
parameters in the 1.8.3 model. This upstream legacy warning remains part of the
validation record and the keypoint model must receive a media-quality test
before runtime promotion.

## Artifact layout

The exporter requires an explicit `NOESIS_DS9_ARTIFACT_ROOT`; it does not
default to the repository or root filesystem. The paths below are versioned
and unpromoted:

```text
DS9/models/checkpoints/rfdetr/1.8.3/
DS9/models/onnx/rfdetr/1.8.3/
DS9/models/engines/rfdetr/1.8.3/
DS9/models/provenance/rfdetr/1.8.3/
```

On an appliance, these virtual DS9 paths are provided by the configured
artifact root.

The CPU-only download/export entry point is:

```bash
export NOESIS_DS9_ARTIFACT_ROOT=/path/to/ds9-artifacts
export RFDETR_EXPORT_VENV=/path/to/pinned-rfdetr-1.8.3-venv

"${RFDETR_EXPORT_VENV}/bin/python" \
  DS9/scripts/export_rfdetr_1_8_3.py --phase all
```

The script:

1. rejects unreviewed model IDs, paths, URLs, package versions, and source
   commits;
2. validates every checkpoint's official byte count and MD5;
3. exports one model per subprocess on CPU;
4. runs `onnx.checker` and checks every input/output shape;
5. atomically publishes the ONNX file and immutable SHA-256 provenance;
6. refuses to re-bless an existing ONNX file unless its receipt, checkpoint,
   release, environment, hash, and tensor contract all match.

Explicit requests for license-gated models fail unless `--include-pml` is
combined with `RFDETR_PML_ACCEPTED=1` after the user has personally accepted
PML-1.0.

## Current realization

All 11 Apache-2.0 checkpoints and immutable normalized-input ONNX files have
been downloaded/exported and validated:

| Family | Variants | Resolution |
| --- | --- | --- |
| detection | Nano, Small, Medium, Large | 384, 512, 576, 704 |
| segmentation | Nano, Small, Medium, Large, XLarge, 2XLarge | 312, 384, 432, 504, 624, 768 |
| keypoint | Preview | 576 |

Each model also has a revision-bound runtime ONNX adapter. Its public input is
static-B3 RGB NCHW float32 in `[0,1]`; two float32 graph nodes then apply
`(input - mean) / std` with ImageNet constants. Adapter revision
`sub_div_float32_v1` deliberately uses `Sub` followed by `Div`, matching the
official torchvision normalization operation. The earlier reciprocal-`Mul`
prototype was rejected because its extra rounding was enough to reorder
queries in the most sensitive segmentation and keypoint models.

TensorRT engines must be produced by the immutable DS9 build image
`noesis-ds9-dev:9.0-20260710` at image ID
`sha256:7476b1021376cd67793c95d949cdc7d46eef7704ab98a5a76feed461e4f907a4`
with TensorRT `10.14.1.48`. The host TensorRT installation is not an authority
for these engines.

The versioned, unpromoted builder is:

```bash
export NOESIS_DS9_ARTIFACT_ROOT=/path/to/ds9-artifacts
export NOESIS_DS9_DOCKER_ROOT=/path/to/secondary-docker

python3 DS9/scripts/build_rfdetr_1_8_3_engines.py \
  --plan \
  --runtime-engine-profile fp16_tf32 \
  --model detect_nano,detect_small,detect_medium,detect_large \
  --model seg_nano,seg_small,seg_medium,seg_large,seg_xlarge,seg_2xlarge \
  --model keypoint_preview
```

Remove `--plan` only inside the approved exclusive GPU window. The builder
uses a networkless, read-only, capability-dropped container as the caller UID,
builds one static-B3 engine at a time, validates candidate and installed
deserialization, installs atomically, and writes SHA-256 receipts and private
logs.

The selected deployable profile is FP16 with TF32 enabled
(`fp16_tf32`). An exclusive GPU window produced all 11 runtime-input engines.
Every engine passed candidate deserialization before installation and
installed-file deserialization afterward. The model matrix binds every
enabled row to its exact runtime ONNX, ONNX receipt, engine, and engine receipt
with four SHA-256 anchors. Runtime materialization additionally attests the
selected parser source bundle and binary against `DS9/asset_manifest.yaml`.
The rendered PGIE configs are engine-only and require `network-mode=2`.

Before labeled local evaluation was available, four representative precision
canaries were assessed with strict ONNX-to-TensorRT fidelity gates:

| TensorRT profile | Representative result |
| --- | --- |
| FP32, TF32 disabled | Detection Medium, Segmentation Small/Medium, and Keypoint Preview all passed |
| FP32, TF32 enabled | Detection Medium and Segmentation Small passed; Segmentation Medium and Keypoint Preview failed |
| FP16, TF32 disabled | Detection Medium passed; both segmentation models and Keypoint Preview failed |
| FP16 with selected FP32 heads, TF32 disabled | Detection Medium passed; both segmentation models and Keypoint Preview failed |

The selective-head profile did not recover the strict fidelity failures. The
full-FP32 result established that those threshold crossings were TensorRT
precision effects rather than checkpoint or ONNX-export failures. A separate
official PyTorch-export-path versus immutable ONNX semantic check also passed
for Detection Medium, Segmentation Small, Segmentation Medium, and Keypoint
Preview. Those historical fidelity results and superseded reports remain
immutable diagnostic evidence; they are not ground-truth accuracy
measurements. No training or fine-tuning was performed.

## Standalone media-quality validation contract

Media quality is evaluated outside the active DeepStream graph by
`DS9/scripts/validate_rfdetr_1_8_3_media.py`. The corpus is 24 private frames
arranged as eight static-B3 batches. Each batch uses the fixed camera order
`family-room`, `kitchen`, `living-room` at one of these exact zero-based source
frame indices:

```text
180, 360, 630, 810, 1080, 1260, 1530, 1710
```

The validator accepts only the following reviewed 68-second local clips:

| Camera | Validator filename | Provenance identity | SHA-256 |
| --- | --- | --- | --- |
| family-room | `family-room-calibration-occupied-v3-68s.mp4` | `calibration-inputs/02-family-room.mp4` | `3906092f2f650a8a8dd50f73d8aba94b5d3c14f573ae7ccea21813d2b9e9b53c` |
| kitchen | `kitchen-calibration-occupied-v3-68s.mp4` | `calibration-inputs/01-kitchen.mp4` | `747726df51577072ccb8666f025c8881f04a31499007805e37fcc1c1b9236194` |
| living-room | `living-room-calibration-occupied-v3-68s.mp4` | `calibration-inputs/00-living-room.mp4` | `57c3a96fb4d17072f71c54ca35330abb4518c2ab121196897820df895992e564` |

The clips contain private home imagery and identifiable people. Extracted
frames, normalized tensors, raw outputs, overlays, logs, and the final report
must remain below the validator's caller-owned mode-`0700` run directory;
evidence files are mode `0600`. Media is never committed or uploaded.

Every frame is resized directly to the selected model's published square
resolution without letterboxing and converted to RGB float32 in `[0, 1]`.
The normalized-input baseline lane applies ImageNet normalization before ONNX
Runtime. The deployable runtime lane instead passes RGB01 tensors to the
revision-bound runtime ONNX, whose `Sub` and `Div` nodes apply the same mean
`[0.485, 0.456, 0.406]` and standard deviation
`[0.229, 0.224, 0.225]`. CPU ONNX Runtime in the pinned RF-DETR 1.8.3 export
environment supplies the reference raw tensors. TensorRT runs through the
dedicated `DS9/native/rfdetr_1_8_3_media_runner.cpp`, compiled and executed
with the immutable DS9 image recorded above and TensorRT `10.14.1.48`; it does
not use the host TensorRT installation.

The gate first requires exact tensor names and shapes, finite values, and
repeatable TensorRT output. It then applies the same post-processing to the
ONNX Runtime and TensorRT tensors and records two conversion scopes. A model's
overall automated result passes only when every applicable scope passes;
person-task success never overrides an all-class failure.

- all-class conversion, for detection and segmentation: strong reference
  queries at score `0.7` or greater are paired one-to-one with TensorRT
  candidate queries by a lexicographic Hungarian assignment. An edge must have
  box IoU of at least `0.5` before class or retention priority applies. Across
  each image, the assignment maximizes retained correct-class spatial matches,
  then correct-class spatial matches, then spatial matches, and finally total
  IoU. Matched pairs require 100% spatial class agreement and retention,
  box-IoU median at least `0.98` and fifth percentile at least `0.90`, and
  score absolute-error 95th percentile at most `0.03`;
- person-task conversion, for every family: RF-DETR COCO class index `1` is
  read directly rather than using each query's all-class argmax. ONNX and
  TensorRT prediction sets are independently thresholded at `0.5`, then
  matched symmetrically by spatial match and IoU. Reference retention and
  candidate precision must both be `1.0`, so this scope catches TensorRT-only
  people as well as dropped ONNX people. Box and ordinary detection-score gates
  use the same `0.98`, `0.90`, and `0.03` limits above;
- segmentation: matched pairs in both applicable scopes have mask logits
  bilinearly resized to source resolution and thresholded at zero. Mask-IoU
  median is at least `0.95` and fifth percentile at least `0.85`, while
  mask-area relative-error 95th percentile is at most `0.10`, excluding
  reference masks smaller than 64 source pixels;
- keypoints: the model has only the person-task conversion scope. The base
  person sigmoid score has absolute-error 95th percentile at most `0.03`.
  Because the official uncertainty-fused score is not bounded to `[0,1]`, its
  separate relative-error 95th percentile is at most `0.03`; it is no longer
  judged by the ordinary absolute-score gate. OKS median is at least `0.97`
  and minimum at least `0.90`; coordinate-error 95th percentile is at most
  `0.01` of the reference box diagonal; findable-keypoint-set Jaccard is at
  least `0.90`, with findability defined as keypoint confidence at least
  `0.5`.

The report records both assignment audits, including every matched and
unmatched person query, score, spatial flag, and IoU. It also records
same-query raw-tensor absolute errors for diagnosis, but they are not
postprocessed quality gates: FP16 DETR proposal ordering may change even when
the decoded predictions remain equivalent.

Validation is intentionally four-phase: `prepare` extracts the corpus and
records CPU ONNX Runtime references, `trt` runs the prepared inputs in the
exclusive GPU window, `compare` applies the automated gates and creates a
24-frame private contact sheet for every model, and `review` seals an explicit
human decision with reviewer identity and notes. A human pass cannot override
failed automated gates. Review covers box alignment, mask silhouettes,
keypoint anatomy and left/right topology, duplicates and portrait
false-positive stressors, and all 20 occupied plus four negative frames.

## 2026-07-24 baseline FP16 media-quality result

Evidence run `rfdetr-media-20260724t151506z` completed all 88 CPU ONNX Runtime
reference cases and all 88 TensorRT cases. Each TensorRT case ran twice with
bitwise-stable, finite output. The final semantic report is
`report-semantic-v3.json`; its explicit visual-review seal is
`visual-review-report-semantic-v3.json`. Both live under the private
`models/validation/rfdetr/1.8.3/rfdetr-media-20260724t151506z/` artifact
directory.

The automated and overall review status is **failed**. Detection Small and
Medium passed every gate. The remaining results are:

| Models | Failed gate |
| --- | --- |
| Detection Nano, Large | retention `0.987952`, `0.992701` |
| Segmentation Nano, Small | retention `0.980583`, `0.981481` |
| Segmentation Medium | retention `0.964912`; score-error p95 `0.040689` |
| Segmentation Large | retention `0.969388`; score-error p95 `0.047059` |
| Segmentation XLarge | retention `0.976562`; score-error p95 `0.036375` |
| Segmentation 2XLarge | score-error p95 `0.040201` |
| Keypoint Preview | score-error p95 `0.054773` |

All 1,184 strong references received a spatial, correct-class match. Every
box-IoU gate passed, all six segmentation variants passed every mask-IoU and
mask-area gate, and Keypoint Preview passed OKS, coordinate-error, and
visible-keypoint-set gates. The failed retention rows represent 16 reference
predictions whose TensorRT score fell below the strict `0.7` floor.

Visual inspection found no systematic scale, translation, box, mask, or
skeleton-topology defect. All four living-room negative frames had zero person
predictions at score `0.5` or greater for every model. Kitchen wall portraits
repeatedly produced person and pose false positives, and some occupied
family-room frames produced low-confidence person misses. Those are checkpoint
semantic findings, not conversion geometry defects.

The initial same-query report and the box-only Hungarian successor remain
immutable diagnostic evidence. The final report records their successor
lineage and uses the auditable semantic assignment described above. No failed
or superseded report authorizes promotion.

These checks preserve the strict fidelity failure as diagnostic evidence. They
do not measure ground-truth AP, invalidate the checkpoint family, or supersede
the later full-corpus local accuracy comparison.

## 2026-07-25 runtime-input FP32 media-quality result

Evidence run `rfdetr-media-20260725t014411439175z` exercised the actual
deployable contract: raw RGB01 input, `sub_div_float32_v1` runtime ONNX, and
the selected `fp32_no_tf32` engines. All 88 CPU ONNX Runtime cases and all 88
TensorRT cases completed. TensorRT repeatability and finite-output checks
passed for every case.

The automated result is **passed for all 11 models**. The final report is
`report-runtime-fp32_no_tf32.json` under the private
`models/validation/rfdetr/1.8.3/runtime/fp32_no_tf32/`
`rfdetr-media-20260725t014411439175z/` directory; its SHA-256 is
`d3d45dcb6ca21ef574f8af9981d982a94908d42a90cfaa14d4e244a514829fd6`.

| Family | Passed variants |
| --- | --- |
| detection | Nano, Small, Medium, Large |
| segmentation | Nano, Small, Medium, Large, XLarge, 2XLarge |
| keypoint | Preview |

Every all-class and person-task scope passed. Reference retention and
candidate precision were both `1.0`; all box-IoU, score-error, mask-IoU,
mask-area, OKS, keypoint-coordinate, and findable-keypoint-set gates passed.
The contact sheets showed no systematic scale or translation error, mask
misregistration, duplicate conversion artifact, or skeleton topology defect.
Kitchen wall portraits still generate some person/keypoint false positives,
but they are present in the reference checkpoint output as well as TensorRT;
they are model-semantic errors and require dataset/model work to remove
reliably. This agent visual inspection is not represented as the separate
human review seal.

The report was generated before source-level runtime selection and therefore
correctly records its evidence phase as unpromoted. At that stage, the model
matrix was updated to select the exact attested FP32/no-TF32 artifacts. The
later local labeled comparison superseded that precision selection while
preserving this report as historical fidelity evidence. No deployment selector
or running appliance model was changed by the media gate.

The broad legacy `validate_asset_manifest.py --profile rfdetr` gate is not a
gate for this 1.8.3 selection and remains red on obsolete FP16 engine rows in
`DS9/asset_manifest.yaml`. Those rows cover only the earlier detection and
segmentation N/S/M artifacts; they are not selected by the baseline appliance
lane or by the matrix-driven 1.8.3 materializer. The current materializer
instead attests all 11 runtime ONNX files, receipts, selected FP16/TF32 engines,
and engine receipts directly, while the manifest still attests each selected
parser source and binary. The obsolete rows were not removed merely to make
the broad profile appear green: a truthful migration also requires a reviewed
RF-DETR supervisor lane and an asset-realization transaction for the complete
11-model set.

## 2026-07-25 local labeled FP16 versus FP32 accuracy

The precision decision uses a head-to-head evaluation executed locally on this
system, not published checkpoint accuracy. Both RF-DETR precision suites used
the same 5,000-image COCO 2017 validation corpus, person-only annotations,
post-processing, top-K policy, and local evaluator. The complete evidence is
relative to `NOESIS_DS9_ARTIFACT_ROOT` at:

```text
models/benchmarks/head-to-head/coco-person-full-20260725T215428672011Z/
```

The table reports the measured primary AP change in points, calculated as
FP16/TF32 minus FP32/no-TF32. Detection uses bounding-box AP, segmentation uses
mask AP, and keypoint uses OKS AP.

| Family | Variant | Primary AP delta (points) |
| --- | --- | ---: |
| detection | Nano | +0.0341 |
| detection | Small | +0.0532 |
| detection | Medium | +0.0246 |
| detection | Large | -0.0194 |
| segmentation | Nano | +0.0061 |
| segmentation | Small | +0.0165 |
| segmentation | Medium | +0.0435 |
| segmentation | Large | -0.0324 |
| segmentation | XLarge | -0.0045 |
| segmentation | 2XLarge | -0.0078 |
| keypoint | Preview | -0.0873 |

Every measured change is within one tenth of an AP point. This local,
ground-truth result shows no material FP16 accuracy loss across the selected
RF-DETR matrix, so `fp16_tf32` is selected for the runtime and for the
apples-to-apples comparison with the current YOLO FP16 engines. The older
24-frame ONNX-to-TensorRT fidelity reports remain useful for diagnosing
score-threshold movement, but they do not override this labeled accuracy
result.

## 2026-07-26 RF-DETR Large promotion-candidate gate

The reviewed RF-DETR detection cadence is `interval=1`. The versioned
materializer rejects any other value so the detector runs every other frame
and preserves headroom for tracking, ReID, pose, depth, MapAnything, mosaic,
and telemetry work.

A controlled live gate compared the active YOLO26-M FP16 baseline with
RF-DETR Large FP16/TF32 at that cadence. Both arms used the same three camera
inputs, DS9 runtime image, baseline tracking mode, downstream models, cloned
world/identity/scene/analytics state, 60-second collection duration, and
storage-backed artifact/runtime roots. This was a quality-upgrade candidate
versus the incumbent, not a size- or performance-matched architecture
comparison.

| Metric | YOLO26-M baseline | RF-DETR Large, `interval=1` |
| --- | ---: | ---: |
| Mean / maximum GPU utilization | 43.05% / 51% | 61.95% / 89% |
| Mean / maximum GPU memory | 6,174 / 6,182 MiB | 6,149 / 6,151 MiB |
| Mean / maximum board power | 97.57 / 99.30 W | 125.05 / 141.21 W |
| Mean / maximum GPU temperature | 63.44 / 64 C | 66.93 / 68 C |
| Tracking messages | 350 | 351 |
| Raw depth counter maximum | 4,417 | 2,672 |
| Tracks / embedding tracks | 0 / 0 | 0 / 0 |
| Pipeline errors / CPU-copy violations | 0 / 0 | 0 / 0 |
| Maximum boundary p99 | 5.859 ms | 4.549 ms |
| RTSP H.264 hardware-decode gate | pass after retry | pass |

RF-DETR Large raised mean GPU utilization by 18.90 percentage points and mean
power by 27.48 W. Its WebSocket boundary latency was not worse, and the full
media and runtime-continuity gates passed. The strict zero-copy helper marked
both arms red because one statistics sample omitted a boundary p99 field;
neither arm recorded a boundary error counter or CPU-copy violation.

The depth-counter values above are cumulative maxima, not first-to-last
deltas. The earlier interpretation of them as a 39.5% throughput reduction is
withdrawn. The candidate also used the active release runtime with the
versioned RF-DETR engine config but the release's older detection parser,
rather than the current attested parser. The resource profile remains useful,
but this run is not exact current-parser semantic evidence.

The scene was unoccupied throughout both arms. Zero detected tracks therefore
supports only an empty-scene false-positive observation; it does not measure
occupied-person recall, track continuity, ReID, pose, or object-depth quality.
This gate does not promote RF-DETR Large over YOLO26-M. The selector-driven
YOLO26-M baseline was restored after the experiment. An occupied,
camera-balanced gate is still required before any PGIE promotion decision.

## 2026-07-27 matched YOLO26-M versus RF-DETR Medium gate

The corrected peer comparison used YOLO26-M and RF-DETR Medium because their
locally measured person AP and isolated batch-3 latency are the closest pair
for the current production tier. Both arms used FP16/TF32, batch size 3,
`interval=1`, the same three cameras, generation-221 release topology,
downstream models, state, and collection windows. The RF-DETR arm used the
current attested detection parser and exact versioned Medium engine.

| Metric | YOLO26-M | RF-DETR Medium |
| --- | ---: | ---: |
| Local person bbox AP / AP50 / AP75 | 63.350 / 85.295 / 69.039 | 62.989 / 87.012 / 67.466 |
| Isolated batch-3 p95 latency | 9.375 ms | 10.463 ms |
| Live mean / maximum GPU utilization | 44.23% / 59% | 46.57% / 60% |
| Live mean / maximum GPU memory | 3,500 / 3,508 MiB | 3,419 / 3,419 MiB |
| Live mean / maximum board power | 98.25 / 99.75 W | 102.39 / 104.16 W |
| Live mean / maximum GPU temperature | 63.66 / 65 C | 64.00 / 65 C |
| Depth device-frame delta rate | 43.63/s | 44.80/s |
| Tracking messages in 60 seconds | 352 | 353 |
| Tracks / embedding tracks | 0 / 0 | 0 / 0 |
| Pipeline errors / CPU-copy violations | 0 / 0 | 0 / 0 |
| Maximum boundary p99 | 3.369 ms | 1.861 ms |
| Strict zero-copy helper | fail: one missing p99 sample | pass |
| RTSP H.264 hardware-decode gate | pass | pass |

RF-DETR Medium added 2.34 GPU-utilization percentage points and 4.14 W mean
power while using about 81 MiB less GPU memory. Its measured depth-device
throughput was 2.7% higher in the independent first-to-last counter windows,
which is effectively preserved for this short live gate. YOLO26-M retains a
0.36-point AP advantage, 1.09 ms lower isolated p95 latency, and the lower
live GPU/power cost; RF-DETR Medium has a 1.72-point AP50 advantage and better
boundary latency in this run.

Both scenes were unoccupied. This comparison establishes resource and
continuity parity but cannot decide occupied-person recall, tracking, ReID,
pose, or object-depth quality. YOLO26-M remains the selected production PGIE
pending a camera-balanced occupied gate.

## 2026-07-27 guided occupied YOLO26-M versus RF-DETR Medium gate

One operator repeated the same path through the three camera views for each
two-minute arm. The first RF-DETR attempt is explicitly invalid for comparison:
the operator remained on a phone call in the living-room and kitchen views and
never entered the family-room view. Its apparent 3.59-times person-row advantage
described a different route and dwell time, not a model advantage.

The replacement RF-DETR arm started from a fresh clone of the same pre-gate
state and followed the YOLO route. The runs were still sequential rather than a
replay of one immutable clip, so person-row counts are comparative live evidence
rather than ground-truth recall. Both valid arms retained FP16/TF32, batch size
3, `interval=1`, the same camera configuration, downstream engines, and
storage-backed evidence.

| Metric | YOLO26-M | RF-DETR Medium |
| --- | ---: | ---: |
| Mean / maximum GPU utilization | 48.20% / 74% | 49.48% / 84% |
| Mean / maximum GPU memory | 3,502 / 3,513 MiB | 3,460 / 3,468 MiB |
| Mean / maximum board power | 100.27 / 112.32 W | 104.55 / 122.92 W |
| Mean / maximum GPU temperature | 64.05 / 66 C | 64.53 / 68 C |
| Tracking messages | 924 | 874 |
| Person track rows | 282 | 232 |
| Depth-ok person rows | 280 (99.29%) | 222 (95.69%) |
| Fresh embedding track rows | 26 | 19 |
| Identity observation journal rows | 66 | 54 |
| Journal rows: family / kitchen / living | 20 / 27 / 19 | 24 / 28 / 2 |
| Unique camera/tracker pairs | 9 | 8 |
| Visitor subjects / cross-camera subjects | 2 / 2 | 2 / 1 |
| Pipeline errors / CPU-copy violations | 0 / 0 | 0 / 0 |
| RTSP H.264 hardware-decode gate | pass | pass |

On the matched path RF-DETR produced 17.73% fewer person rows, 26.92% fewer
fresh embedding rows, and 18.18% fewer identity-journal rows. It covered the
family-room and kitchen portions, but recorded only two living-room journal
observations versus YOLO's 19. Its depth-ok share was 3.60 percentage points
lower. No pipeline error or CPU-copy violation explains the difference.
Identity remains uncalibrated shadow evidence, so subject counts are continuity
diagnostics rather than semantic accuracy scores.

The matched RF-DETR arm cost 1.28 more mean GPU-utilization percentage points,
4.28 W more mean power, and 0.48 C more mean temperature while using about
42 MiB less GPU memory. Its maximum utilization was ten percentage points
higher. The invalid phone-interrupted run remains useful only as evidence that
RF-DETR load rises materially with sustained occupied detections; it is not
used in the selection decision.

YOLO26-M therefore remains the selected PGIE. RF-DETR Medium is not rejected as
an architecture, but it is not promoted from this matched-path gate. A future
RF-DETR candidate needs to recover living-room persistence and the downstream
depth/embedding yield without spending more occupied GPU and power headroom.

## 2026-07-26 runtime-input FP16 media-fidelity result

Evidence run `rfdetr-media-20260726t012828290418z` repeated the full private
media workflow after the model matrix selected `fp16_tf32`. It prepared all 88
CPU ONNX Runtime references from the pinned clips, executed all 88 cases
through the exact selected FP16/TF32 engines, and generated
`report-runtime-fp16_tf32.json` under the private
`models/validation/rfdetr/1.8.3/runtime/fp16_tf32/` run directory. The report
SHA-256 is
`4d90e1fa668c7fdefb915f670d61d1de4e641e023d10f45977e6d2cf063e0213`.
No visual-review seal was recorded for this run.

The strict automated fidelity result is **failed**: Detection Small and
Detection Medium passed, while the other nine variants crossed one or more
zero-tolerance retention, candidate-precision, or score-error gates. Every
box-geometry gate passed, all segmentation mask gates passed, and all
keypoint geometry/topology gates passed. This reproduces the earlier FP16
threshold sensitivity against the actual runtime-input engines; it is not a
ground-truth accuracy measurement and does not itself authorize selection.
The preceding 5,000-image labeled comparison is the evidence used for the
precision decision.

## Keypoint runtime integration

DS9 now has an explicit `rfdetr_keypoint` PGIE profile for the sole reviewed
RF-DETR 1.8.3 Keypoint Preview variant. It does not select or retain the YOLO
pose SGIE as a fallback. The profile requires the runtime-derived ONNX/engine
contract recorded in the model matrix. The selected runtime artifact is the
FP16, TF32-enabled engine under the `fp16_tf32` profile; the rendered PGIE
config requires `network-mode=2`. Materialization fails rather than using an
engine outside the matrix-selected profile.

For this profile, `nvinfer` owns full-frame preprocessing so its output tensor
metadata is attached directly to each frame. The input is resized directly to
`576x576`, converted to RGB, and scaled by `1/255`; the runtime-derived graph
then applies the reviewed ImageNet mean and standard deviation. Letterboxing,
symmetric padding, and external input-tensor metadata are disabled.

`NvDsInferParseRFDETRKeypoint` validates the exact `dets`, `labels`, and
`keypoints` output names and shapes, decodes normalized `cxcywh` boxes, and
uses raw class-logit slot `1` for the base person sigmoid. It then reproduces
the official `trace_alpha=0.2` precision-Cholesky uncertainty fusion before
thresholding and publishing detection confidence. Direct checkpoint
inspection confirmed `_kp_active_mask == [0, 17]`, so the active COCO joints
are exactly keypoint slots `17:34`; slot `0` is the legacy background class.
Parsed person objects are exposed as DS class `0`.

The DS9 `noesis_pose_meta_ext` native extension reads the frame-owned PGIE
tensor metadata through the public Service Maker metadata API. At
`world_observation_stage`, it associates raw person queries with tracked person
objects using reciprocal unique best-IoU matches, a minimum IoU of `0.70`, and
a runner-up ambiguity margin of `0.05`. Ambiguous or unmatched objects are
skipped and counted; the profile permits no stale pose cache. Accepted rows
feed the existing `NOESIS.POSE_FEATURES` contract and overlay path. Startup
fails if the exact parser/config, frame tensor metadata, engine, native bridge
symbol, or explicit YOLO-pose disablement is absent.

Focused validation completed:

- the keypoint parser and native pose extension built against the installed
  DeepStream 9 SDK;
- the parser is included in the DS9 all-parser build and artifact inventory;
- the `rfdetr_keypoint` artifact profile passed staged file and provenance
  verification;
- all 30 focused keypoint runtime tests passed, including preprocessing,
  output schema, no-fallback selection, threshold/top-k locking,
  fail-closed metadata attachment, native/Python object binding, full matched
  object coverage, and strict reciprocal association checks.

This is an implemented, build-validated runtime path, not a live-runtime
deployment claim. The reviewed input adapter and matrix-selected
`fp16_tf32` engine are materialized and provenance-attested successfully.
A live three-camera profile smoke could not run from the current checkout
because `DS9/noesis/ds9_runtime_core.py` imports the absent
`noesis_core.runtime_publication` module before profile preflight. No deployed
snapshot or substitute import path was used as a fallback. The exact existing
appliance deployment is restored independently through its selector after the
offline validation window.
