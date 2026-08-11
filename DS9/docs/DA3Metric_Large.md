# DA3Metric-Large Manual Depth Backend

_Status: deployed and selector-bound on 2026-08-05 EDT (2026-08-06 UTC).
The RTX 3060 FP16 engine, canonical three-camera canary, appliance readiness,
and one fresh manual capture have passed._

## Scope

DA3Metric-Large is a restart-scoped alternative for the gated DS9 manual-depth
branch historically named `mapanything`. It preserves that branch's component
name, GIE UID `2`, valve, Refresh/cache-only lifecycle, native tensor capture,
storage layout, REST/WebSocket contracts, and `depth`/`conf`/`mask` tensor
names. It does not replace the always-on Depth Anything V2 tracking lane.

There is no automatic fallback between DA3Metric-Large and MapAnything. An
unknown selection or missing engine/config fails startup.

## Official Inputs

- Source: [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3),
  exact commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
- Checkpoint: [depth-anything/DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE),
  exact revision `4010e39f3634a45bc60553321fb49fb760bd594e`.
- `model.safetensors` SHA-256:
  `bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776`.
- `config.json` SHA-256:
  `a336f3e76fe375aaae17a9aed9130c9f2aa061535d317ec57dcb2f1f02e1dd53`.
- License: Apache-2.0.

The checkpoint is pinned by revision and digest. Runtime startup never
downloads or substitutes model weights.

## Export and Tensor Contract

`DS9/scripts/export_da3metric_large.py` exports the official metric head at a
fixed spatial profile and dynamic batch:

- input `images`: FP32 RGB `0..255`, `N x 3 x 294 x 518`;
- official ImageNet normalization is embedded in the graph;
- outputs `depth`, `conf`, and `mask`: FP32,
  `N x 1 x 294 x 518`;
- TensorRT production profile: fixed batch `3` for the three canonical cameras;
- ONNX opset: `17`.

The official model has a depth head and a sky head, not a learned confidence
head. The export converts `sky < 0.3` into binary non-sky validity and exposes
that same validity map as both `conf` and `mask` for the existing Noesis
contract. Consumers must not interpret `conf` as calibrated model uncertainty.

The official metric rule is applied per camera after inference:

```text
depth_m = raw_depth * model_input_focal_px / 300
```

`model_input_focal_px` is derived from each calibrated camera intrinsics matrix
after the exact resize/letterbox transform into `294 x 518`. Dewarper validity
is then intersected as before.

The official convenience path replaces sky depth using a sampled percentile.
The export deliberately keeps the deterministic raw metric head and applies
the official sky decision as validity, avoiding stochastic percentile/random
operators in the TensorRT graph.

## Installed Artifacts

Paths below are relative to the canonical DS9 artifact root:

- ONNX: `models/onnx/da3metric_large_294x518_b3.onnx`
  - SHA-256: `9d0193542845d601f733d82379197def618dec7c0182cc73adbcce9ded7cbb67`
- TensorRT engine: `models/engines/da3metric_large_294x518_b3_fp16.engine`
  - SHA-256: `a426275c0c76798d7eeccd2ef1f57edc02258fe4d059d1b4afb1c36dc31cea47`
  - size: `681,032,644` bytes
- Export receipt:
  `models/engine_validation/da3metric-large/export-receipt.json`
- Validation receipt:
  `models/engine_validation/da3metric-large/validation-receipt.json`

The engine is specific to the installed RTX 3060 (compute capability 8.6),
TensorRT 10.14.1.48, the reviewed DS9 build image, fixed batch `3`, and FP32
I/O with FP16 internal compute. Rebuild it after changing the GPU, TensorRT,
CUDA/driver compatibility boundary, ONNX, profile, or build image.

## Selection and Toggle

Accepted values are exactly `mapanything` and `da3metric-large`. MapAnything is
the code default when no selector is supplied.

For direct DS9 launches, select at process start with either:

```bash
python3 DS9/noesis/ds9_runtime.py --manual-depth-model da3metric-large
```

or:

```bash
NOESIS_MANUAL_DEPTH_MODEL=da3metric-large \
  python3 DS9/scripts/run_canonical_runtime_container.py plan --lane baseline
```

The canonical supervisor validates the value and injects it into the runtime
container. It is not a hot in-process switch.

For the appliance, the companion Menon bundle renderer accepts the same closed
choice:

```bash
node scripts/appliance-systemd.mjs --render \
  --output-dir "$BUNDLE" \
  --menon-repo "$RELEASE/menon" \
  --noesis-repo "$RELEASE/noesis" \
  --manual-depth-model da3metric-large \
  --homeseer-base-url http://127.0.0.1
```

Render and validate a new bundle, then activate it through the normal selector
transaction. To return to MapAnything, render with
`--manual-depth-model mapanything` and activate that bundle. Do not hand-edit
the installed `noesis.env`; it is part of the bundle integrity contract.

## Registration Boundary

Existing DAv2-to-reference depth-registration fingerprints remain tied to the
effective `depth_registration.mapanything_reference`. Selecting DA3Metric-Large
does not rewrite, reinterpret, or silently refresh those MapAnything-derived
calibrations. Registration-evidence capture rejects DA3 as the reference
backend until a separately reviewed DA3 calibration contract exists.

## Validation Record

The FP16 engine passed:

- exact `images`, `depth`, `conf`, and `mask` bindings at batch `3`;
- all `456,876` reference depth values finite and positive;
- PyTorch-versus-TensorRT depth MAE `0.0125001 m`, RMSE `0.0174303 m`, maximum
  absolute error `0.146664 m`, mean relative error `0.00221468`, and
  correlation `0.999976983`;
- binary validity agreement `0.999862107` (63 threshold-edge pixels differed);
- mean GPU compute `57.53396 ms` per three-camera batch and p95 `57.806545 ms`;
- 213 of 220 compute layers reporting FP16 output, with FP32 I/O;
- a canonical real-camera canary with all three sources, clean shutdown, and
  the DA3 engine/config deserialized under UID `2`;
- selector-bound Noesis and Menon production readiness;
- one five-second manual Refresh producing one committed `1080 x 1920` Zarr
  snapshot for family room, kitchen, and living room, with all `6,220,800`
  depth pixels finite and positive; the valve closed again after the window.

Focused source validation is:

```bash
python3 -m pytest -q DS9/tests/test_da3metric_large_runtime.py
```

`DS9/scripts/validate_da3metric_large_engine.py` regenerates the numerical and
provenance receipt from the exact ONNX, fixture, PyTorch reference, TensorRT
outputs, timing export, and layer-information export.
