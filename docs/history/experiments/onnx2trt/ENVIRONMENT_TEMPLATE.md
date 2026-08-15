# MapAnything Export Environment Summary

Populate the table below with the exact versions used during ONNX export, TensorRT engine build, and DeepStream validation. This is the information the MapAnything maintainers will use to reproduce or audit the integration.

| Category        | Value / Notes |
| --------------- | ------------- |
| **GPU**         | _(e.g. NVIDIA RTX 4090)_ |
| **GPU Driver**  | _(e.g. 550.54.14)_ |
| **CUDA Toolkit**| _(e.g. 12.4)_ |
| **TensorRT**    | _(e.g. 10.1.0.13)_ |
| **DeepStream**  | _(e.g. 7.1.0)_ |
| **cuDNN**       | _(e.g. 9.0)_ |
| **Python**      | _(e.g. 3.10.12)_ |
| **PyTorch**     | _(e.g. 2.2.2+cu121)_ |
| **ONNX Runtime**| _(if used for validation)_ |
| **ONNX Opset**  | _(e.g. 17)_ |
| **Export Script** | Path + git commit hash |
| **Hydra/HuggingFace Config** | Relevant config snapshot or commit |
| **Operating System** | _(e.g. Ubuntu 22.04.3 LTS)_ |

### Recommended Commands

```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__)"
python3 -c "import onnx; import onnxruntime; print(onnx.__version__, onnxruntime.__version__)"
deepstream-app --version-all
dpkg -l | grep TensorRT
```

Attach the output from `ma_onnxTRT/collect_env.py` as `environment_snapshot.json` when submitting upstream.
