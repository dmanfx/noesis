# DEIMv2 Wholebody49 Prototype

DS8 Service Maker prototype for PINTO0309 `488_DEIMv2-Wholebody49`.

It runs the three RTSP sources from the DS8 config, uses the mask-capable
DEIMv2 DINOv3-S ONNX, builds a TensorRT FP16 engine, emits DeepStream object
and body instance-mask metadata through a native parser, and writes lightweight
JSON summaries for the model's whole-body classes, attributes, keypoint boxes,
skeleton links, and mask-capable detections. Non-body wholebody outputs stay
available through tensor metadata, JSON summaries, and the prototype overlay.

```bash
python3 testpipelines/deimv2-wholebody49/main.py
```

The default sink is a visible 1920x1080 three-column mosaic. Each tile renders
body instance masks, whole-body/object/attribute/bone boxes, keypoint dots,
skeleton links, source labels, and per-group counts. Use `--headless` only for
CI or command-line smoke tests.

For a lower-GPU trial that prunes the mask head from ONNX/TensorRT and keeps the
same RTSP/mosaic/overlay setup otherwise:

```bash
python3 testpipelines/deimv2-wholebody49/main.py --model-variant boxes
```

The `boxes` alias maps to `dinov3_s_boxes` and keeps `label_xyxy_score` only.
It still draws bboxes, keypoint dots, and skeleton links from tensor metadata,
but it does not render body instance masks.

For the larger PINTO demo-quality backbone without masks:

```bash
python3 testpipelines/deimv2-wholebody49/main.py --model-variant x_boxes
```

For an experimental INT8 build of that same DINOv3-X label-only model, first
capture representative calibration frames from the same three RTSP sources:

```bash
python3 testpipelines/deimv2-wholebody49/capture_calibration_images.py \
  --frames-per-source 48
```

Then build the separate INT8 engine from that calibration-image directory:

```bash
python3 testpipelines/deimv2-wholebody49/model_setup.py \
  --model-variant x_boxes_int8 \
  --int8-calibration-images data/deimv2_wholebody49/int8_calibration/<run_dir>
```

Once the INT8 engine exists, run it explicitly:

```bash
python3 testpipelines/deimv2-wholebody49/main.py --model-variant x_boxes_int8
```

`x_boxes_int8` never falls back to the FP16 X engine. If the INT8 engine is
missing or stale, `model_setup.py` requires either `--int8-calibration-images`
or `--int8-calibration-cache`.

To reduce X GPU pressure while keeping the mosaic visually stable, skip one
inference frame and reuse the last smoothed overlay on skipped tensor frames:

```bash
python3 testpipelines/deimv2-wholebody49/main.py \
  --model-variant x_boxes \
  --infer-interval 1
```

`--infer-interval 1` writes `interval=1` into the DeepStream `nvinfer` config.
Skipped tensor frames are marked in logs and JSON with `cached=1` /
`reused_tensor_cache: true`. Use `--disable-cached-overlay` for raw interval
debugging.

If CPU-side overlay construction becomes the bottleneck in crowded scenes, cap
the number of detections drawn while keeping the decoded JSON summaries intact:

```bash
python3 testpipelines/deimv2-wholebody49/main.py \
  --model-variant x_boxes \
  --infer-interval 1 \
  --max-draw 160
```

Supported variant names and aliases:

- `masks` / `dinov3_s_masks`: DINOv3-S with `label_xyxy_score` and `masks`.
- `boxes` / `dinov3_s_boxes`: DINOv3-S label-only, no mask output.
- `x_masks` / `dinov3_x_masks`: DINOv3-X with `label_xyxy_score` and `masks`.
- `x_boxes` / `dinov3_x_boxes`: DINOv3-X label-only, no mask output.
- `x_boxes_int8` / `dinov3_x_boxes_int8`: DINOv3-X label-only INT8 engine,
  no mask output, requires representative calibration before first build.

The official `488_DEIMv2-Wholebody49` resources archive used by this prototype
does not currently include a DINOv3-L ONNX. Requests such as
`--model-variant dinov3_l_boxes` fail explicitly instead of falling back to S or
X.

The default thresholds now mirror PINTO's demo posture: `--score-threshold 0.50`
for objects/body masks, `--keypoint-score-threshold` defaults to the same value,
and `--attribute-score-threshold 0.75` keeps helper labels from flickering. For a
more exploratory overlay, lower them explicitly:

```bash
python3 testpipelines/deimv2-wholebody49/main.py \
  --score-threshold 0.35 \
  --keypoint-score-threshold 0.35 \
  --attribute-score-threshold 0.35
```

The overlay also applies class-aware filtering and per-source temporal smoothing
by default. Use `--disable-class-aware-filtering` or `--disable-smoothing` only
for raw-tensor debugging.

Useful setup and validation commands:

```bash
python3 testpipelines/deimv2-wholebody49/model_setup.py
python3 testpipelines/deimv2-wholebody49/model_setup.py --model-variant boxes
python3 testpipelines/deimv2-wholebody49/model_setup.py --model-variant x_boxes
python3 testpipelines/deimv2-wholebody49/capture_calibration_images.py --frames-per-source 48
python3 testpipelines/deimv2-wholebody49/model_setup.py --model-variant x_boxes_int8 --int8-calibration-images data/deimv2_wholebody49/int8_calibration/<run_dir>
make -C testpipelines/deimv2-wholebody49/parser
python3 -m pytest -q testpipelines/deimv2-wholebody49/tests
```

Outputs are written under `data/deimv2_wholebody49/` by default.
