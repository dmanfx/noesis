# MapAnything Engine Parity Plan

This plan defines how to build the DS9 MapAnything TensorRT engine so its
runtime depth behavior matches the current DS8 floorplan-quality baseline.

The goal is not a bit-identical TensorRT plan. The goal is output parity:
DS9 depth snapshots must generate the same readable BEV floorplans with clear
room bounds, stable floor regions, and large object footprints.

## Reference Baseline

The current good DS8 runtime uses:

```text
engine: models/mapanything_depth/1/model.plan
sha256: 83db4111e936f66f3c68acb5a1b94c7737a3d60dbb9250809aded481745aeb49
size: 1328123748 bytes
```

Observed engine contract:

```text
TensorRT engine type: explicit batch
input:  images float32 (3, 3, 294, 518)
output: depth  float32 (3, 1, 294, 518)
output: conf   float32 (3, 1, 294, 518)
output: mask   float32 (3, 1, 294, 518)
profile: fixed batch 3, fixed 294x518
```

DS9 must reproduce this tensor IO surface and the resulting depth statistics.

## Source ONNX Contract

Use the same MapAnything export surface currently staged under the shared model
store:

```text
onnx: models/onnx/mapanything_images_294x518_b3.onnx
sha256: 0a27dca24238d8103b83df67ce8f71d11c6a79574d83d6e9bab237f3d9c222c7
input: images, NCHW, normalized 0..1
shape: batch x 3 x 294 x 518
outputs: depth, conf, mask
model id: facebook/map-anything-apache
```

The ONNX file uses external tensor sidecar files. Staging for DS9 must copy the
ONNX plus every referenced external tensor file into the DS9 artifact area. A
tiny ONNX file without sidecars is not a valid build source.

Target staging location:

```text
DS9/models/onnx/mapanything_images_294x518_b3.onnx
```

That directory is a generated DS9 artifact location and may need to be created
by the staging step.

## Build Requirements

Build only inside a real DS9 environment:

```text
DeepStream: 9.0
TensorRT: 10.14.1.48
output: DS9/models/engines/mapanything_images_294x518_b3_fp16.plan
```

The DS9 MapAnything build should be functionally equivalent to:

```bash
trtexec \
  --onnx=DS9/models/onnx/mapanything_images_294x518_b3.onnx \
  --fp16 \
  --minShapes=images:3x3x294x518 \
  --optShapes=images:3x3x294x518 \
  --maxShapes=images:3x3x294x518 \
  --builderOptimizationLevel=0 \
  --maxAuxStreams=0 \
  --memPoolSize=workspace:1024 \
  --saveEngine=DS9/models/engines/mapanything_images_294x518_b3_fp16.plan \
  --skipInference
```

Critical issue to correct before building:

```text
DS9/scripts/rebuild_engines.py currently names the MapAnything output
*_fp16.plan but uses --bf16 for the MapAnything EngineSpec.
```

For parity with the current good DS8 behavior, the first DS9 parity build must
use real `--fp16`, not BF16.

## Runtime Config Contract

`DS9/config/infer.yaml` and `DS9/pipelines/config_infer_secondary_mapanything.ini`
must preserve the DS8-good runtime semantics with DS9-local paths:

```ini
process-mode=1
network-type=100
model-engine-file=DS9/models/engines/mapanything_images_294x518_b3_fp16.plan
batch-size=3
gie-unique-id=2
network-mode=2
model-color-format=0
net-scale-factor=0.003921568627
input-tensor-from-meta=0
infer-dims=3;294;518
output-blob-names=depth;conf;mask
maintain-aspect-ratio=1
symmetric-padding=1
output-tensor-meta=1
```

`gie-unique-id=2` is required. If DS9 depth logs show only DAv2 tensor IDs, for
example `available_ids=[5, 5]`, then MapAnything tensors are not being consumed
and the build/runtime path fails parity.

## Do Not Use As Parity Targets

These known alternate or failed variants are not the target:

```text
models/mapanything_depth/1/model_bf16_nonorm.plan
models/mapanything_depth/1/model_fp16_nan.plan
models/mapanything_depth/1/model_518x518_prev.plan
models/engines/mapanything_images_294x518_b3_bf16_backup_20260617.plan
```

The old fused `518x518` DS8 build logs are also not the target. The good live
plan is images-only, `294x518`, batch 3, with three outputs.

## Validation Gates

### Static Engine Inspection

After building, inspect the DS9 engine:

```bash
polygraphy inspect model DS9/models/engines/mapanything_images_294x518_b3_fp16.plan --model-type=engine
```

Expected contract:

```text
images float32 (3, 3, 294, 518)
depth  float32 (3, 1, 294, 518)
conf   float32 (3, 1, 294, 518)
mask   float32 (3, 1, 294, 518)
```

### Runtime Snapshot Statistics

DS9 MapAnything snapshots must resemble the DS8-good data:

```text
stored aligned shape: 1080x1920
finite positive depth pixels: about 2073600
mask true fraction: about 1.0
```

Reject a DS9 build if mask coverage returns to the prior poor range of about
`0.77` to `0.79`.

Depth distributions should stay close to the DS8-good baseline:

```text
kitchen:     median about 0.54 m, p95 about 3.48 m, max below about 9 m
living-room: median about 1.88 m, p95 about 5.27 m, max below about 9 m
family-room: median about 2.96 m, p95 about 6.46 m, max below about 9 m
```

### Floorplan Producer Contract

The DS9 depth snapshots must feed the known-good BEV producer path:

```text
floorplan contract: 7
gridResM: 0.15
method: weighted height with residual clean floorplan
exclude: AGL-grid and top-surface experiment paths
```

Expected floorplan cache shape should remain compact, roughly similar to the
current DS8 `grid0p15__ext20` outputs, not the very large noisy `0.05 m` grids.

### Visual Acceptance

For each camera, the DS9 `grid0p15__ext20` floorplan must show:

- readable room bounds
- coherent walkable floor regions
- large interior object footprints such as tables or islands
- no broad smear of noisy color blotches
- no mostly black or low-information BEV canvas

If the visual output fails this gate, do not tune the renderer first. Inspect
the MapAnything engine, tensor UID path, depth snapshot statistics, and BEV
producer contract first.

## Implementation Sequence

1. Stage the DS8-good MapAnything ONNX and every external tensor sidecar into
   the DS9 generated model area.
2. Update the DS9 MapAnything engine spec to use `--fp16` instead of `--bf16`.
3. Build fixed batch-3 `294x518` only.
4. Inspect the engine IO with Polygraphy.
5. Run DS9 depth and capture fresh Zarr snapshot statistics for all active
   cameras.
6. Compare depth/mask statistics against the DS8-good baseline above.
7. Generate `grid0p15__ext20` floorplans from DS9 snapshots.
8. Accept only if the visual floorplans match the DS8-good readability.
9. Regenerate or repin `DS9/config/depth_registration.json` only after the final
   DS9 engine/config pair is selected, so profile fingerprints match the actual
   runtime assets.

## Notes

Renderer changes, dewarper-validity masks, and floorplan postprocessing should
not be used to compensate for a bad DS9 MapAnything engine. The first parity
requirement is matching the dense `depth/conf/mask` behavior of the DS8-good
engine.
