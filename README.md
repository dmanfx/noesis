# Noesis: A GPU-Accelerated Video Analytics Pipeline

Noesis is a high-performance, real-time video analytics application designed for pure GPU processing. It leverages the power of NVIDIA DeepStream to create an end-to-end pipeline that handles everything from video decoding to AI inference and streaming, all on the GPU. This approach minimizes CPU bottlenecks and provides a robust, scalable foundation for demanding video analysis tasks.

![Pipeline Flow](docs/reference/pipeline_flow.png)

## Key Features

- **End-to-End GPU Processing**: The entire pipeline, from RTSP stream decoding to AI inference and visualization, runs on the GPU, ensuring maximum performance and minimal latency.
- **DeepStream-Native**: Built on the NVIDIA DeepStream SDK, Noesis uses optimized GStreamer plugins for all core video processing tasks.
- **YOLOv11 Integration**: The pipeline uses a custom-parsed YOLOv11 model for primary object detection, with support for other models via configuration.
- **Advanced Analytics**: Integrated with `nvdsanalytics` for high-level event detection, including:
    - **ROI (Region of Interest) Filtering**: Monitor specific areas of the video feed.
    - **Line Crossing Detection**: Trigger events when objects cross a virtual line.
    - **Direction Detection**: Analyze the direction of object movement.
    - **Overcrowding Detection**: Monitor the number of objects in a defined area.
- **Real-time Streaming**: Processed video and metadata are streamed in real-time to a web-based frontend via WebSockets, allowing for remote monitoring and control.
- **Configurable Architecture**: Noesis is highly configurable, with the ability to toggle between native DeepStream components and custom Python-based logic for tasks like object tracking and visualization.
- **Robust and Scalable**: Designed for production environments, with features like automatic pipeline recovery, health monitoring, and support for multiple camera streams.

## Architecture Overview

The Noesis pipeline is divided into two main layers:

1.  **DeepStream Pipeline Layer**: This is the core of the application, where all heavy lifting is done. It's a GStreamer pipeline that uses a series of optimized plugins to:
    - Decode multiple RTSP streams (`nvurisrcbin`).
    - Batch them for efficient processing (`nvstreammux`).
    - Preprocess the frames for inference (`nvdspreprocess`).
    - Run a YOLOv11 object detection model (`nvinfer`).
    - Track objects across frames (`nvtracker`).
    - Perform high-level analytics (`nvdsanalytics`).
    - Overlay visualizations on the video (`nvdsosd`).

2.  **Python Application Layer**: This layer acts as a high-level coordinator. It starts and stops the DeepStream pipeline, extracts metadata and processed frames, and handles application-level logic:
    - The `DeepStreamProcessorWrapper` class encapsulates the DeepStream pipeline, providing a clean interface for the main application.
    - The `ApplicationManager` coordinates all components, including the pipeline, WebSocket server, and result processing.
    - A `WebSocketServer` streams video and analytics data to a web frontend and allows for real-time configuration changes.

For a more detailed breakdown of the pipeline, see the [DeepStream Pipeline Map](docs/reference/DEEPSTREAM_PIPELINE_MAP.md).

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

3.  **Configure the pipeline**:
    - Edit `config.py` to set up your camera streams, model paths, and other pipeline settings.
    - Review the DeepStream configuration files (`config_infer_primary_yolo11.txt`, `config_preproc.txt`, etc.) to customize the inference and preprocessing steps.

### Running the Application

To run the application with a single RTSP stream:

```bash
python main.py --rtsp "your_rtsp_stream_url"
```

You can also run with a local video file or a webcam:

```bash
# From a video file
python main.py --video /path/to/your/video.mp4

# From a webcam
python main.py --webcam
```

Once the application is running, you can view the output by opening the `index.html` file in the `electron-frontend` directory in a web browser.

## Documentation

- **[DeepStream Pipeline Map](docs/reference/DEEPSTREAM_PIPELINE_MAP.md)**: A detailed, step-by-step map of the entire pipeline, from input to output.
- **[Pipeline Flow](docs/pipeline_flow.md)**: A high-level overview of the pipeline's architecture with Mermaid diagrams.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details. 