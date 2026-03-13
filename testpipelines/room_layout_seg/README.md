# Room Layout Segmentation Utility

DS8 Service Maker utility pipeline that segments room-structure classes and emits reusable mask bundles for downstream Noesis consumers.

## Model choice

This pipeline uses `nvidia/segformer-b5-finetuned-ade-640-640` as the base model, then exports a custom ONNX/TensorRT wrapper that collapses the 150 ADE20K classes into compact layout groups:

- `other`
- `wall`
- `floor`
- `ceiling`
- `window`
- `door`
- `stairs`

The wrapper preserves the original model’s semantic probabilities but avoids emitting a full 150-channel tensor at `1024x576`, which would be unnecessarily heavy for a periodic utility.

## Outputs

For each source, the probe writes a stable bundle under `data/room_layout_masks/<sensor_id>_<sensor_name>/`:

- `layout_manifest.json`
- `layout_class_map.png`
- `layout_preview.png`
- `layout_probabilities_fp16.npz`
- `mask_wall.png`
- `mask_floor.png`
- `mask_ceiling.png`
- `mask_window.png`
- `mask_door.png`
- `mask_stairs.png`

The manifest records the output label mapping back to the original ADE20K class IDs so downstream Noesis code does not need to guess.

## Usage

Generate or reuse `sources.yaml`, then run the pipeline:

```bash
python3 testpipelines/room_layout_seg/main.py --camera "Family Room Camera" --duration 12
```

If `testpipelines/room_layout_seg/sources.yaml` is missing, `main.py` generates it from `config/infer.yaml`.

Useful flags:

- `--headless` to skip the EGL preview
- `--max-sources 2` to tile multiple cameras
- `--frames-per-package 24` to control how much evidence is accumulated before emitting masks
- `--emit-every-frames 24` to refresh the bundle periodically during longer runs

## Artifacts

Materialized artifacts are stored under `models/room_layout_segformer_b5_ade20k/`:

- Hugging Face snapshot
- exported ONNX
- TensorRT engine
- compact output label metadata
- `model_info.json`
