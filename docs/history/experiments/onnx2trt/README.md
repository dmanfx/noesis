# MapAnything ONNX → TensorRT Contribution Bundle

This folder packages the artifacts and documentation needed to upstream the DeepStream SGIE work to the MapAnything project. Everything here is safe to share publicly (no weights or proprietary config blobs) and mirrors the evidence Meta’s team typically requests when validating external contributions.

## Contents

| File | Purpose |
| ---- | ------- |
| `ENVIRONMENT_TEMPLATE.md` | Table to capture the exact runtime stack used for export/build/testing. |
| `EXPORT_PIPELINE.md` | Step-by-step notes covering ONNX export, TensorRT conversion, and validation. |
| `DEEPSTREAM_INTEGRATION.md` | Summary of how the fused-input SGIE, preprocess, and probe wiring land inside DeepStream. |
| `CONTRIBUTION_CHECKLIST.md` | Final punch list to make sure nothing is missing when submitting. |
| `collect_env.py` | Script that prints a JSON snapshot of GPU / CUDA / TensorRT / DeepStream versions. |

Existing logs and artifacts you can reference alongside this pack:

- `logs/export_ma_fused.log` – fused-input ONNX exporter run (CLI, stdout, validation summary)
- `logs/trtexec_fused_verbose.log` – TensorRT build attempt for the fused FP16 batch-3 engine (rerun on GPU host to regenerate the plan)
- `logs/trtexec_fused_verbose_tail.log` – tail snippet capturing build status/perf summary
- `ma_onnx_out_fused/model_fused.onnx` & `ma_onnx_out_fused/model_fused_sim.onnx` – fused-input ONNX graphs staged for TensorRT
- `models/engines/ma_model_fp16_b3_fused.plan` – target path for the fused-engine (generate on CUDA-enabled node before packaging)
- `pipelines/mapanything_preprocess_fused/libmapanything_preprocess_fused.so` – custom `nvdspreprocess` library that appends intrinsics channels
- `pipelines/config_preprocess_mapanything_fused.ini` / `pipelines/config_infer_secondary_mapanything_fused.ini` – DeepStream configs that drive fused preprocess + SGIE `nvinfer`

## How to Use This Bundle

1. **Capture the environment**
   ```bash
   python3 ma_onnxTRT/collect_env.py | tee ma_onnxTRT/environment_snapshot.json
   ```
   Copy key fields into `ENVIRONMENT_TEMPLATE.md`.

2. **Document the pipeline**
   - Fill in ONNX export details, CLI flags, and validation results in `EXPORT_PIPELINE.md`.
   - Note GPU throughput / latency figures from `logs/trtexec_fused_verbose_tail.log` once the build is re-run on a GPU node.

3. **Sync DeepStream notes**
   - Ensure the config snippets and SGIE wiring described in `DEEPSTREAM_INTEGRATION.md` exactly match the repo state you intend to contribute.
   - Build the custom preprocess library (`cd pipelines/mapanything_preprocess_fused && make`) so the `.so` is bundled alongside the configs.

4. **Run the checklist**
   - `CONTRIBUTION_CHECKLIST.md` serves as the final gate before sharing a PR or patch set.

5. **Package for sharing**
   - Include this entire `ma_onnxTRT/` directory, the relevant logs, and references to code diffs when creating the upstream contribution.

> Tip: Meta’s reviewers appreciate seeing both the raw command invocations (e.g. `trtexec` call) and the resulting performance summaries. Keep those snippets nearby when you draft the contribution notes or PR description.
