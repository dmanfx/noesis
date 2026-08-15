# YOLO11-Seg Model Generation Script

## Overview

`generate_seg_model.py` is a convenience script that automates the complete workflow of exporting a YOLO11-segmentation PyTorch model (`.pt`) to ONNX format and then building a TensorRT FP16 engine for DeepStream integration.

## Prerequisites

1. **Export venv**: A Python virtual environment at `~/venvs/torch_export` with:
   - `torch==2.2.2` (CUDA-enabled or CPU)
   - `onnx==1.15.0`
   - `onnxsim==0.4.36`
   - `onnxslim==0.1.47`
   - `ultralytics==8.3.25`
   - Other dependencies (numpy<2, protobuf, opencv-python, etc.)

2. **TensorRT**: `trtexec` must be available in PATH

3. **Export script**: `utils/export_yolo11_seg.py` must exist

## Quick Start

### Basic Usage (using script defaults)

```bash
python utils/generate_seg_model.py
```

This will:
- Export `models/yolo11m-seg.pt` → `models/yolo11m-seg_cust.onnx`
- Build TensorRT engine → `models/yolo11m-seg_cust.engine`

### Custom Paths

```bash
python utils/generate_seg_model.py \
  --weights models/yolo11s-seg.pt \
  --onnx models/yolo11s-seg_cust.onnx \
  --engine models/engines/yolo11s-seg_cust.engine
```

### Partial Workflows

```bash
# Only export ONNX (skip engine build)
python utils/generate_seg_model.py --skip-engine

# Only build engine (skip ONNX export)
python utils/generate_seg_model.py --skip-export
```

## Configuration

Edit the variables at the top of `generate_seg_model.py` to change defaults:

```python
# Source and destination paths
SOURCE_WEIGHTS = "models/yolo11m-seg.pt"
OUTPUT_ONNX = "models/yolo11m-seg_cust.onnx"
OUTPUT_ENGINE = "models/yolo11m-seg_cust.engine"

# Export parameters
IMG_SIZE = [640, 640]
MAX_DETECTIONS = 30
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
OPSET_VERSION = 18
USE_DYNAMIC_BATCH = True
SIMPLIFY_ONNX = True

# TensorRT engine parameters
FP16_PRECISION = True
MIN_BATCH_SIZE = 1
OPT_BATCH_SIZE = 3
MAX_BATCH_SIZE = 3
```

## Export Parameters

The script uses the following export settings (matching DeepStream integration requirements):

- **Image size**: 640×640
- **Max detections**: 30
- **Confidence threshold**: 0.25
- **IoU threshold**: 0.5
- **ONNX opset**: 18
- **Dynamic batch**: Enabled (supports batch 1-3)
- **ONNX simplification**: Enabled (using onnxslim)

## TensorRT Engine

The generated engine:

- **Precision**: FP16
- **Batch profiles**:
  - Min: 1×640×640
  - Opt: 3×640×640
  - Max: 3×640×640
- **Outputs**: `boxes`, `scores`, `classes`, `coeffs`, `proto`, `pre_boxes`

## Output Files

### ONNX Model (`*.onnx`)

The ONNX file contains:
- Dynamic batch dimension
- NMS integrated via TensorRT custom op
- Separate outputs for boxes, scores, classes, mask coefficients, and prototypes
- Optimized and simplified graph

### TensorRT Engine (`*.engine`)

The engine file:
- FP16 precision for optimal performance
- Optimized for batch sizes 1-3
- Ready for DeepStream inference configuration

## Integration with DeepStream

After generating the engine, configure DeepStream to use it:

1. Update `config_infer_primary_yolo11_seg.ini`:
   ```ini
   model-engine-file=models/yolo11m-seg_cust.engine
   output-blob-names=boxes;scores;classes;coeffs;proto;pre_boxes
   ```

2. Ensure the custom parser (`libnvdsinfer_yolo11_seg.so`) is built and configured

3. The parser handles mask composition from `coeffs` and `proto` tensors

## Troubleshooting

### Export Fails

- Verify venv exists and has correct packages: `source ~/venvs/torch_export/bin/activate && python -c "import torch, onnx, ultralytics; print('OK')"`
- Check that source weights file exists
- Ensure `export_yolo11_seg.py` is present in `utils/`

### Engine Build Fails

- Verify `trtexec` is in PATH: `which trtexec`
- Check ONNX file was created successfully
- Ensure GPU is available and TensorRT is properly installed
- For static batch models, remove `--minShapes/--optShapes/--maxShapes` flags (edit script)

### Performance Issues

- Verify FP16 is enabled
- Check batch size matches your use case
- Profile with `trtexec --loadEngine=... --shapes=images:3x3x640x640 --avgRuns=200`

## Related Files

- `utils/export_yolo11_seg.py` - ONNX export script
- `utils/export_yolo11_seg.py.bak` - Backup export script (fused output version)
- `pipelines/nvdsinfer_yolo11_seg/nvdsinfer_yolo11_seg.cpp` - Custom parser for DeepStream
- `pipelines/config_infer_primary_yolo11_seg.ini` - DeepStream inference config

## Notes

- The script uses the **coeffs+proto** export format (not fused masks)
- Mask composition happens in the DeepStream parser, not in the engine
- Dynamic batch allows runtime flexibility but requires proper shape profiles
- Engine size is typically ~48-49 MB for medium models
