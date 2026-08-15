# YOLO11m-Seg Custom Export

Steps to reproduce the lighter-fused mask engine:

```bash
# 1) Activate the python environment that has torch/onnx installed
# 2) Export ONNX with the new fused 128×128 mask grid (keep 640×640 input)
python DeepStream-Yolo-Seg/utils/export_yolo11_seg.py \
  --weights models/yolo11m-seg.pt \
  --img 640 640 \
  --max-detections 30 \
  --conf 0.25 \
  --iou 0.5 \
  --opset 17 \
  --simplify \
  --out models/yolo11m-seg_cust.onnx

# 3) (Optional) Inspect output names/shapes
python - <<'PY'
import onnx
m = onnx.load("models/yolo11m-seg_cust.onnx")
print([o.name for o in m.graph.output])
for o in m.graph.output:
    print(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
PY

# Expected: fused detection tensors whose mask vector length is 16384 (128×128)

# 4) Build TensorRT FP16 engine with explicit batch=3 profiles for 640
trtexec \
  --onnx=models/yolo11m-seg_cust.onnx \
  --saveEngine=models/engines/yolo11m-seg_cust.engine \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:3x3x640x640 \
  --maxShapes=images:3x3x640x640 \
  --verbose \
  --profilingVerbosity=detailed \
  --exportLayerInfo=layers.json \
  --exportProfile=profiles.json \
  --separateProfileRun

# 5) Quick perf check
trtexec --loadEngine=models/engines/yolo11m-seg_cust.engine --shapes=images:3x3x640x640 --avgRuns=200
```
