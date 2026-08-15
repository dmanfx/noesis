# ONNX → TensorRT Export Pipeline

This document captures the steps (and required tweaks) to convert MapAnything’s monocular depth model into a TensorRT engine that DeepStream can ingest as a secondary GIE.

## 1. Prepare the Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch onnx onnxruntime onnxsim hydra-core
```

Ensure the MapAnything source repo (and any private adapters) are on the `PYTHONPATH`.

## 2. Export the Model to ONNX (Fused Input)

Example invocation (run from repo root):

```bash
python3 export_ma_onnx/export_to_onnx.py \
  --repo external/map-anything \
  --outdir ma_onnx_out_fused \
  --h 518 \
  --w 518 \
  --opset 17
```

What this does:

- Wraps the MapAnything depth model with the new fused-input helper so `forward(mapanything_fused)` accepts a `(B, 12, H, W)` tensor.
- Splits the fused tensor into RGB (channels 0-2) and tiled intrinsics (channels 3-11 → averaged back to a 3×3 matrix).
- Emits `ma_onnx_out_fused/model_fused.onnx`, a shape-inferred copy (`model_fused-inferred.onnx`), and a best-effort simplified graph (`model_fused_sim.onnx`). If `onnxsim` cannot rewrite the graph because of the large external data tensors, the exporter copies the original model to the `_sim` path and records the failure reason in `export_report.txt`.
- Runs `onnx.checker` and an `onnxruntime` smoke test with the fused binding name `mapanything_fused`.

CLI flags:

- `--no-fused-input` revives the legacy two-input flow for emergency rollbacks.
- `--ckpt` can be pointed at an Apache MapAnything checkpoint if you need to swap weights.

Logs and reports land in `logs/export_ma_fused.log` and `ma_onnx_out_fused/export_report.txt`.

## 3. Validate the ONNX Graph

Smoke tests are already part of the exporter, but you can rerun them manually:

```bash
python3 - <<'PY'
import onnx, onnxruntime as ort
path = "ma_onnx_out_fused/model_fused_sim.onnx"
onnx.checker.check_model(path)
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
import numpy as np
fused = np.random.randn(1, 12, 518, 518).astype("float32")
out = sess.run(None, {"mapanything_fused": fused})
print("ORT OK:", [o.shape for o in out])
PY
```

## 4. Build the TensorRT Engine (Requires CUDA)

Run on a TensorRT 10.x GPU host:

```bash
trtexec \
  --onnx=ma_onnx_out_fused/model_fused_sim.onnx \
  --saveEngine=models/engines/ma_model_fp16_b3_fused.plan \
  --minShapes=mapanything_fused:1x12x518x518 \
  --optShapes=mapanything_fused:3x12x518x518 \
  --maxShapes=mapanything_fused:3x12x518x518 \
  --fp16 \
  --memPoolSize=workspace:4096 \
  --verbose \
  --dumpLayerInfo | tee logs/trtexec_fused_verbose.log
tail -n 50 logs/trtexec_fused_verbose.log > logs/trtexec_fused_verbose_tail.log
```

If you run this inside a sandbox without GPU access you'll see `Cuda failure ... OS call failed`. Capture the log (as above) and re-run on a CUDA-enabled machine to actually build the engine.

Expected artifacts after a successful build:

- `models/engines/ma_model_fp16_b3_fused.plan`
- `logs/trtexec_fused_verbose.log`
- `logs/trtexec_fused_verbose_tail.log`

## 5. DeepStream Integration Check

Once the fused engine exists, confirm DeepStream loads it without rebuilding:

```bash
GST_DEBUG=2 python3 deepstream_video_pipeline.py --config pipelines/config_infer_primary_yolo11.ini
```

Watch for:

- `nvinfer` logging the fused binding `mapanything_fused` with batch size 3.
- `MapAnything depth cam=... depth=...` pad-probe logs once detections are processed.

Configs that must match the fused contract:

- `pipelines/config_preprocess_mapanything_fused.ini`
- `pipelines/config_infer_secondary_mapanything_fused.ini`
- `models/mapanything_depth/config.pbtxt` (single input named `mapanything_fused`)

## 6. Bundle Evidence for Upstream Contribution

Collect:

- CLI invocations (`export_ma_fused.log`, `trtexec` command + tail).
- Updated configs / README snippets referencing the fused tensor.
- Pad-probe diff demonstrating metadata attachment (unchanged consumers, single input provider).
- Environment snapshot (`ENVIRONMENT_TEMPLATE.md` + `collect_env.py` output).
