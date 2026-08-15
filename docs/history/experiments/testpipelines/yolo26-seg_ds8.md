# YOLO26 Segmentation Test Pipeline

Minimal DS8 Service Maker pipeline to visualize YOLO26 instance segmentation output.

## Usage

```bash
python testpipelines/yolo26-seg/main.py --camera "Kitchen Camera" -msize s
```

Flags:
- `--camera <name>` selects a named camera from `sources.yaml` (the selected stream is duplicated to fill batch=3).
- `-msize {n,s,m}` selects the model size (engine/onnx under `models/`).
- `--headless` uses `fakesink` instead of an on-screen display.

## Parser build

The custom instance-mask parser is built on demand by `parser_setup.py`. To build manually:

```bash
make -C testpipelines/yolo26-seg/parser
```
