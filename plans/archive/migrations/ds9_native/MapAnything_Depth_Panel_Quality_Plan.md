# MapAnything Depth Panel Quality Plan

_Current implementation and validation status as of 2026-07-27._

## Outcome

The highest-quality validated MapAnything lane is the historical 24-encoder /
24-information-sharing-block export rebuilt for the installed DS9 toolchain as
a TensorRT 10.14.1.48 BF16 engine. It reproduces the recognizable Plan 14 room
geometry while running substantially faster than the strict-FP32 build of the
same source. It is the selected successor engine.

The successor is being packaged and has **not** been activated. The production
selector has not yet been switched, so this document does not claim that the
selected engine or the final floorplan changes are live.

## Implemented quality corrections

- **Primary floorplan:** requests and cache probes use a `0.04 m` grid. The
  primary product is the observed metric `height` grid, not a semantic
  walkable/obstacle composite or an AGL reconstruction. Finite observed cells
  are scaled from p3 to p97 with gamma `0.9`, preserving the relative relief
  needed to recognize counters, couches, tables, chairs, walls, and room
  boundaries.
- **Unknown-space handling:** the renderer repairs only an isolated
  one-cell mask hole. A masked cell is display-repaired with the median of its
  eight neighbors only when at least five neighbors are finite and valid.
  Larger holes remain unknown. The same rule is used on screen and in export;
  no broad inpainting, smoothing, inferred floor, or semantic decoration is
  applied to the primary image.
- **Native temporal inference:** manual Refresh now stays on the native
  streammux / `nvinfer` preprocessing path and feeds the real six-frame cohort
  into robust scale/confidence fusion. The defective request-local path that
  duplicated one RGB image across all three batch slots and bypassed temporal
  fusion was removed.
- **Dark-frame rejection:** a capture is rejected before manual or temporal
  writes when luminance p90 is below `25` and `p99 - p50` is below `20`.
  This prevents a nearly black camera frame from replacing a useful cached
  result.
- **Snapshot authority:** one explicit Refresh still produces one fresh dense
  snapshot and derives the floorplan from that exact identity. Passive load,
  reconnect, selection, 3D, and export remain cache-only.

Focused validation currently passes:

- `45/45` depth-quality frontend tests, including the primary render/export
  behavior.
- `55/55` focused capture-adapter, controller, and runtime tests, plus Python
  syntax validation.

## Engine defect and source selection

The deformed current-canonical output is not primarily a floorplan renderer,
TensorRT-version, precision, or postprocessing problem. The same exact B3 input
reproduces the bad spatial geometry in the current canonical TensorRT engine
and in ONNX Runtime CUDA. Scalar, affine, power, piecewise, and alignment
recovery attempts did not restore the room geometry.

The decisive difference is the exported model architecture:

- Selected historical source:
  `utils/onnx2trt/export_ma_onnx/ma_onnx_out_forward/model.onnx`
  (`a2ffbd4b6b2d376bce725bc309e65b32bb2cb298174ac3e80acd915d2fb78a4e`),
  with 24 encoder blocks, 24 information-sharing blocks, encoder width 1024,
  information-sharing width 768, and approximately 558 million parameters.
- Current canonical export: 24 encoder blocks, 16 information-sharing blocks,
  widths 1536/1536, and approximately 920 million parameters.

The architecture/source change, rather than a benign re-export of the same
network, explains why rebuilding the current canonical ONNX in BF16, strict
FP32, and ONNX Runtime did not recover the Plan 14 geometry.

## Same-input engine evidence

All values below use the same three-camera B3 input. Plan 14 is the previously
recognizable output used as the local reference.

### Selected 24+24 TensorRT 10.14 BF16 engine

- Engine:
  `/mnt/noesis_storage/noesis-validation/mapanything-legacy24-trt101401-20260727/mapanything_legacy24x768_b3_bf16_trt101401.plan`
- SHA-256:
  `dd59e2a00d2f62f6c47890b549c9df8b389400626c645023e8fd8ade69321efa`
- Size: `1,312,500,364` bytes.
- Exact B3 GPU latency: `275.635 ms`.
- Mask agreement with Plan 14: `1.0` for every camera.

| Camera | Depth correlation | MAE | p95 absolute error |
| --- | ---: | ---: | ---: |
| Kitchen | 0.999998616 | 3.580 mm | 9.160 mm |
| Living room | 0.999999580 | 2.159 mm | 5.180 mm |
| Family room | 0.999998611 | 2.044 mm | 5.600 mm |

The frontend-equivalent `0.04 m` floorplans retain the Plan 14 bounds and
observed geometry. Observed-mask IoU is `0.9680`, `0.9878`, and `0.9531` for
kitchen, living room, and family room respectively.

Comparison mosaic:

`/mnt/noesis_storage/noesis-validation/mapanything-legacy24-trt101401-20260727/plan14-vs-legacy24-trt101401-vs-v11-control-0p04-pure-height.png`

Mosaic SHA-256:
`41ea00c28ea03284b9b5264fdb0256376e8e706c3edf00d10e9f33baba2aaf0f`.

Detailed metrics and receipt:

- `/mnt/noesis_storage/noesis-validation/mapanything-legacy24-trt101401-20260727/comparison-metrics.json`
- `/mnt/noesis_storage/noesis-validation/mapanything-legacy24-trt101401-20260727/validation-receipt.json`
  (SHA-256
  `1ffb537982997788ff545c0dffaf391e505709a8d5fa463e040d48e5f91da969`).

### Strict-FP32 build of the same 24+24 source

The strict-FP32 build is coherent but is not the winner:

- Engine:
  `/mnt/noesis_storage/noesis-validation/mapanything-legacy24-trt101401-20260727/fp32-strict/mapanything_legacy24x768_b3_fp32_strict_trt101401.plan`
- SHA-256:
  `fe528132c732a240d8045c2f42233bf95e83822b7b6fb0d29a21cba0f096afe7`
- Size: `2,434,366,164` bytes.
- Exact B3 GPU latency: `714.551 ms`.
- Correlation with Plan 14, kitchen/living/family:
  `0.999943283 / 0.999985811 / 0.999971409`.
- MAE, kitchen/living/family:
  `11.519 / 9.217 / 7.457 mm`.
- p95 absolute error, kitchen/living/family:
  `21.800 / 23.920 / 21.010 mm`.
- Mask agreement: `1.0` for every camera.

BF16 is selected because it is closer to Plan 14, uses approximately half the
engine storage, and reduces exact-B3 latency by about 61% relative to strict
FP32.

### MapAnything v1.1.3 candidates

- The v1.1.3 `294x518` control is spatially coherent and is a valid contingency
  candidate, but it does not match the selected 24+24 engine. Against Plan 14,
  depth MAE is `0.363 / 0.326 / 0.302 m` and correlation is
  `0.976 / 0.988 / 0.979` for kitchen/living/family. Exact-B3 latency is about
  `1.137 s`.
- The v1.1.3 `378x672` HR arm adds some local texture, but worsens room scale
  and unknown-space voids in kitchen and living room while increasing exact-B3
  latency to about `1.963 s`.

Neither v1.1.3 candidate is selected. No different model family has been
integrated or deployed.

## Remaining activation work

- Finish the narrow successor release with the selected BF16 engine, updated
  engine/source contracts, asset realization, and exact engine fingerprints.
- Refresh the depth-registration fingerprint for the selected engine without
  claiming surveyed metric accuracy.
- Run the focused profile/contract checks and frontend production build.
- Activate the successor selector once, restart the canonical DS9 service, and
  trigger one manual six-frame Refresh per camera.
- Inspect and publish the actual live three-camera primary-floorplan mosaic.
  Dark captures must fail closed; they must not be substituted or written as a
  successful refresh.

Surveyed anchors remain the correct future gate for absolute scale and world
alignment. They are not required to validate this relative visual-quality
improvement, and no anchor placement is assumed in this pass.
