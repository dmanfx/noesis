# Noesis: A GPU-Accelerated Video Analytics Pipeline

Noesis is a high-performance, real-time video analytics application designed for pure GPU processing. It leverages the power of NVIDIA DeepStream to create an end-to-end pipeline that handles everything from video decoding to AI inference and streaming, all on the GPU. This approach minimizes CPU bottlenecks and provides a robust, scalable foundation for demanding video analysis tasks.

See diagrams in `docs/reference/pipeline_flow.md`.

## Key Features

- **End-to-End GPU Processing**: The entire pipeline, from RTSP stream decoding to AI inference and visualization, runs on the GPU, ensuring maximum performance and minimal latency.
- **DeepStream-Native**: Built on the NVIDIA DeepStream SDK, Noesis uses optimized GStreamer plugins for all core video processing tasks.
- **YOLOv11 Integration**: The pipeline uses a custom-parsed YOLOv11 model for primary object detection, with support for other models via configuration.
- **Advanced Analytics**: Integrated with `nvdsanalytics` for high-level event detection, including:
    - **ROI (Region of Interest) Filtering**: Monitor specific areas of the video feed.
    - **Line Crossing Detection**: Trigger events when objects cross a virtual line.
    - **Direction Detection**: Analyze the direction of object movement.
    - **Overcrowding Detection**: Monitor the number of objects in a defined area.
- **Motion-Trail Visualization**: Draw persistent, fading trails behind tracked objects (GPU-rendered via `nvdsosd`) with configurable length, opacity, stride, and optional labels. Features intelligent line budget allocation and frame-rate optimization.
- **Real-time Streaming**: Processed video and metadata are streamed in real-time to a web-based frontend via WebSockets, allowing for remote monitoring and control.
- **Configurable Architecture**: Noesis is highly configurable, with the ability to toggle between native DeepStream components and custom Python-based logic for tasks like object tracking and visualization.
- **Robust and Scalable**: Designed for production environments, with features like automatic pipeline recovery, health monitoring, and support for multiple camera streams.

## Architecture Overview

The Noesis pipeline is divided into two main layers:

1.  **DeepStream Pipeline Layer**: This is the core of the application, where all heavy lifting is done. It's a GStreamer pipeline that uses a series of optimized plugins to:
    - Decode and batch multiple streams (`nvmultiurisrcbin`).
    - Preprocess the frames for inference (`nvdspreprocess`).
    - Run a YOLOv11 object detection model (`nvinfer`).
    - Track objects across frames (`nvtracker`).
    - Perform high-level analytics (`nvdsanalytics`).
    - Overlay visualizations on the video per stream (`nvdsosd` in per-branch paths).

2.  **DS8 Runtime Layer**: This layer coordinates startup, configuration, telemetry, and APIs around the DS8 pipeline:
    - `noesis/ds8_runtime.py` is the canonical runtime entrypoint.
    - `noesis/pipelines/ds8_pipeline.py` builds the Service Maker graph and core DeepStream components.
    - `noesis/pipelines/hooks.py` handles metadata extraction, overlays, analytics wiring, and runtime callbacks.
    - `websocket_server.py` streams telemetry, tracking, BEV, and WebRTC signaling to clients.
    - `noesis/server/*.py` provides the FastAPI endpoints used by the DS8 runtime REST service.

For a more detailed breakdown of the pipeline, see the `docs/reference/DEEPSTREAM_PIPELINE_MAP.md`.

## Getting Started

### Prerequisites

- **Hardware**: An NVIDIA GPU with CUDA support (Turing architecture or later recommended).
- **Software**:
    - Ubuntu 20.04 or later.
    - NVIDIA DeepStream 6.0 or later.
    - Python 3.8 or later.
    - GStreamer and its development libraries.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/noesis.git
    cd noesis
    ```

2.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Activate DeepStream environment (once per shell)**:
    ```bash
    source ./activate_deepstream.sh
    # Optional sanity check
    gst-inspect-1.0 nvmultiurisrcbin | head -n 5
    ```

4.  **Configure the pipeline**:
    - Set stream + pipeline topology in `config/infer.yaml` (or another runtime YAML passed via `--pipeline-config`).
    - Set camera metadata/intrinsics in `config/cameras.yaml` (or another YAML passed via `--cameras-config`).
    - Review DeepStream config files under `pipelines/` for model, tracker, preprocess, and analytics tuning.

### Running the Application

Use the DS8 runtime harness as the canonical entrypoint:

```bash
python3 noesis/ds8_runtime.py \
  --pipeline-config config/infer.yaml \
  --cameras-config config/cameras.yaml
```

Example with V3DT tracking mode and REST disabled:

```bash
python3 noesis/ds8_runtime.py \
  --pipeline-config config/infer_v3dt_baseline.yaml \
  --cameras-config config/cameras.yaml \
  --tracking-mode v3dt \
  --disable-rest
```

To start the dashboard UI in development mode run the following inside `electron-frontend`:

```bash
npm install
npm run dev
```

For a production build run `npm run build` and then launch Electron with `npm start`. This will open the bundled `dist/index.html` automatically.

## Documentation

- `docs/reference/DEEPSTREAM_PIPELINE_MAP.md`: Detailed map of the entire pipeline, from input to output.
- `docs/reference/pipeline_flow.md`: High-level architecture with Mermaid diagrams.
- `docs/reference/Static_ROI_Exclusion.md`: ROI exclusion plugin usage, stream ID vs pad index mapping, and stream shuffle checklist.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License.
