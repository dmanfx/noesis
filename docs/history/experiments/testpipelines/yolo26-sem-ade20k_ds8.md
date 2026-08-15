# YOLO26 ADE20K Semantic Testpipeline

Runs the retained official Ultralytics `yolo26{s,l}-sem-ade20k` checkpoints as
batch-3 FP16 TensorRT engines on the three canonical Noesis room streams.

The current Ultralytics semantic export emits one `UINT8` class-id map per
frame. The custom DeepStream parser preserves that exact 150-class map as
`NvDsInferSegmentationMeta`. A final snapshot probe copies only the selected
RGB evidence frames to the CPU edge and writes:

- one aligned raw RGB JPEG per room;
- one blended masked JPEG per room;
- one raw class-map PNG per room;
- one three-room masked mosaic;
- one JSON summary with the dominant ADE20K classes.

Camera locators are resolved in memory from `config/infer.yaml` and the
owner-only Noesis camera registry. They are never written into this directory.

Example:

```bash
python3 testpipelines/yolo26-sem-ade20k/main.py \
  --size s \
  --headless \
  --duration 30
```

Run one saved still image through an exact TensorRT engine. The still is
repeated across the engine's fixed three-frame batch and the runner verifies
that all three output class maps are identical:

```bash
python3 testpipelines/yolo26-sem-ade20k/still_image.py \
  --size l \
  --image path/to/living-room.jpg \
  --output-dir build/yolo26_sem_ade20k/still_comparison/yolo26l
```

Inspect the saved well-lit class maps interactively. The no-build viewer lets
you select Small or Large, shows the exact ADE20K label under
the mouse cursor, and lists every class present with its color and pixel share.
Click a legend entry to isolate that class; click it again or use **Clear
filter** to restore the complete image:

```bash
python3 testpipelines/yolo26-sem-ade20k/viewer/serve_viewer.py
```

Open `http://127.0.0.1:8766/testpipelines/yolo26-sem-ade20k/viewer/index.html`.

The same three-camera saved evidence is available in the `oai2-fe` Depth
drawer under the **Sem-seg** tab. Its camera selector switches among Living
Room, Kitchen, and Family Room, while the model selector switches among the
two retained TensorRT flavors. It is a display-only view and never launches inference
or a depth refresh.

Required artifacts live under `models/yolo26_sem_ade20k/`:

- `weights/yolo26{size}-sem-ade20k.pt`
- `onnx/yolo26{size}-sem-ade20k_b3.onnx`
- `engines/yolo26{size}-sem-ade20k_b3_fp16.engine`
- `labels/ade20k_labels.txt`

The ADE20K release provides `.pt` checkpoints, not prebuilt TensorRT engines.
Export with a current isolated Ultralytics environment and build static
`images=3x3x640x640` engines using the installed DS8 TensorRT `trtexec`. The
static batch matches this testpipeline's exact three-room contract and avoids
shipping an unused dynamic batch profile.
