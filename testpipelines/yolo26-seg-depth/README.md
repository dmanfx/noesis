# YOLO26 Seg + Depth Prototype

DS8 single-stream prototype for the `Family Room Camera` that combines:

- `YOLO26-Seg` small instance segmentation
- `Depth Anything V2` metric depth (indoor small / Hypersim) by default, with optional explicit DA3 probing only when requested
- Canonical `NOESIS.OBJECT_DEPTH` user meta attached to each `NvDsObjectMeta`
- Person-only spatial projection fields (`anchor_uv`, `anchor_depth_m`, `world_point*`) derived from the existing DS8 calibration bundle

This prototype stays on the canonical DS8 path:

- `pyservicemaker` Pipeline API only
- `BatchMetadataOperator` for post-infer fusion
- native object-meta attachment via `noesis_depth_meta_ext`

## Usage

```bash
python3 testpipelines/yolo26-seg-depth/main.py --camera "Family Room Camera"
python3 testpipelines/yolo26-seg-depth/main.py --headless --duration 15
python3 testpipelines/yolo26-seg-depth/main.py --camera "Family Room Camera" --depth-preference da2
python3 testpipelines/yolo26-seg-depth/main.py --camera "Family Room Camera" --frame-width 1920 --frame-height 1080
python3 testpipelines/yolo26-seg-depth/main.py --camera "Family Room Camera" --source-mode file
python3 testpipelines/yolo26-seg-depth/main.py --camera "Family Room Camera" --source-mode active
python3 testpipelines/yolo26-seg-depth/main.py --camera "Family Room Camera" --disable-depth --headless --duration 30
python3 testpipelines/yolo26-seg-depth/main.py --camera "Family Room Camera" --depth-every-n-frames 2 --headless --duration 30
python3 testpipelines/yolo26-seg-depth/main.py --camera "Family Room Camera" --depth-input-shape 924x518 --show-bbox
```

## Notes

- The detector path uses a batch-1 variant of the fused YOLO26 small ONNX, but the config/preprocess materialization is kept in parity with the main DS8 runtime path.
- This prototype is single-stream only. It no longer builds `nvmultistreamtiler`; the graph is `source -> [optional depth] -> seg -> overlay -> nvdsosd -> sink`.
- The depth path auto-provisions `Depth Anything V2 Metric Hypersim Small` into:
  - `models/depth_anything_v2/checkpoints/depth_anything_v2_metric_hypersim_vits.pth`
  - `models/onnx/depth_anything_v2_metric_hypersim_vits_518x924_b1.onnx`
  - `models/engines/depth_anything_v2_metric_hypersim_vits_518x924_b1_fp16.engine`
- The launcher now defaults to `--depth-preference da2`, so DA3 is not probed unless you explicitly request `auto` or `da3`.
- Benchmark / tuning controls:
  - `--disable-depth` keeps the same source/display path but removes `depth_infer` entirely.
  - `--depth-every-n-frames N` sets the DAv2 `nvinfer interval` to `N-1` and reuses the last aligned depth frame for skipped frames.
  - `--depth-input-shape WIDTHxHEIGHT` rebuilds the DAv2 ONNX/TRT/config at that shape.
  - `--show-bbox` re-enables bbox drawing; default overlay is masks + text only.
- Source selection keeps using camera names from `sources.yaml`, but URI resolution can now come from the main `config/infer.yaml`:
  - `--source-mode stream` picks the RTSP URI for that camera.
  - `--source-mode file` picks the adjacent commented `file://...` URI for that camera when present.
  - `--source-mode active` uses whichever URI is currently active in `config/infer.yaml`.
- Fusion now uses post-mux DS8 frame coordinates as the only legal sampling space. `source_frame_width` is diagnostic only.
- Object depth is sampled strictly from instance masks. There is no bbox fallback in the current prototype path.
- For `class_id == 0` (person), the fusion stage also derives a mask-based foot anchor plus a calibration-aware world projection. It uses lower-body depth when it agrees with the floor-plane projection and otherwise keeps the floor-projected point as the guarded world location without introducing per-camera tuning constants.
- The prototype calibration resolver binds the single-stream runtime source back to the selected camera’s canonical DS8 calibration entry and mirrors the current main-runtime baseline by treating the selected `camera_calibration.json` extrinsics as meter-scale (`unit_scale=1.0`) for live person-world projection.
- Once per second, the fusion probe logs frame-level finite-depth coverage plus per-object depth status/statistics, and the launcher prints a final `seg_depth_summary` line with averaged FPS, detection count, depth cadence, reuse counts, and mean tensor/alignment/fusion timings.
- File mode is uncapped by default, so it is useful for overhead A/B checks but not for live-FPS comparison against RTSP.
- Offline MapAnything sanity checks are available with:
  - `python3 testpipelines/yolo26-seg-depth/compare_mapanything_anchor.py --object-depth-json /path/to/object_depth.json --ma-depth-json /path/to/ws_ma_depth_response.json`
