# Contribution Checklist for MapAnything Fused SGIE Support

Use this list before submitting patches or documentation upstream.

## Environment & Reproducibility
- [ ] `environment_snapshot.json` created via `python3 ma_onnxTRT/collect_env.py`
- [ ] `ENVIRONMENT_TEMPLATE.md` filled with GPU, CUDA, TensorRT, DeepStream, PyTorch versions
- [ ] GPU driver / CUDA compatibility verified with `nvidia-smi`

## Model Export & Engine Build
- [ ] ONNX export script and config captured (path + commit hash)
- [ ] `onnxsim` / `onnx.checker` run on final ONNX file
- [ ] `trtexec` command + performance summary (latency, throughput) documented
- [ ] FP16 batch-3 plan stored at `models/engines/ma_model_fp16_b3_fused.plan` (not shared publicly)

## DeepStream Integration
- [ ] `pipelines/config_preprocess_mapanything_fused.ini` and `pipelines/config_infer_secondary_mapanything_fused.ini` committed with correct tensor names and batch size
- [ ] Custom preprocess library (`pipelines/mapanything_preprocess_fused/libmapanything_preprocess_fused.so`) builds cleanly
- [ ] Pad-probe logic (`_mapanything_depth_probe`) merged and unit-tested
- [ ] `pipelines/mapanything_depth_postprocess.py` tests passing (`pytest tests_mapanything/...`)
- [ ] `docs/reference/MapAnything_Depth.md` updated with metadata contract and validation notes

## Validation Artifacts
- [ ] DeepStream run logs show fused SGIE loading & depth metadata attachment (`UID 22`)
- [ ] Optional: screenshots or telemetry dumps demonstrating depth fields downstream
- [ ] Any non-default configs (e.g. analytics exclusion, intrinsics table generation) noted for reviewers

## Submission Packaging
- [ ] Include this `ma_onnxTRT/` directory and referenced log files
- [ ] Summarize changes using the template in `README.md`
- [ ] Double-check no proprietary weights/configs are bundled
