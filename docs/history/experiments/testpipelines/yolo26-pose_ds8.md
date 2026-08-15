# YOLO26 Pose Test Pipeline

Minimal DS8 Service Maker pipeline to visualize YOLO26 pose keypoints.

## Usage

```bash
python testpipelines/yolo26-pose/main.py --camera "Kitchen Camera" -msize s
```

Flags:
- `--camera <name>` selects a named camera from `sources.yaml` (the selected stream is duplicated to fill batch=3).
- `-msize {n,s,m}` selects the model size (engine/onnx under `models/`).
- `--headless` uses `fakesink` instead of an on-screen display.
