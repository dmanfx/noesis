# YOLO Video Tracking with GPU RTSP Pipeline

A high-performance video analysis system with pure GPU RTSP processing, object detection, tracking, and real-time visualization.

## Features

- **Pure GPU RTSP Processing**: Zero CPU copies using PyNvCodec and NVIDIA DALI
- **Hardware-Accelerated Decoding**: NVDEC support for H.264/HEVC streams
- **TensorRT Inference**: Optimized GPU inference for maximum performance
- **Real-time Tracking**: Multi-object tracking with ByteTrack algorithm
- **Live Visualization**: WebSocket-based dashboard with real-time video streams
- **Multi-Stream Support**: Process multiple RTSP streams simultaneously

## Installation

### System Requirements

- NVIDIA GPU with NVDEC support (GTX 1050+ or RTX series)
- CUDA Toolkit 11.8+
- Driver version 470+
- Python 3.8+

### Quick Install

1. **Install PyNvCodec dependencies for GPU RTSP processing:**
   ```bash
   pip install nvidia-pyindex nvidia-vpf
   ```

2. **Install project dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO models:**
   ```bash
   # Place your YOLO models in the models/ directory
   # Default paths in config.py:
   # - models/yolo12m.pt (detection)
   # - models/yolo11m-pose.pt (pose estimation)
   ```

### Advanced Installation

For development or custom configurations:

```bash
# Clone repository
git clone <repository-url>
cd <project-directory>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyNvCodec for GPU RTSP (optional but recommended)
pip install nvidia-pyindex nvidia-vpf
```

## Quick Start

### Basic Usage

```python
from main import main
from config import AppConfig

# Configure RTSP streams
config = AppConfig()
config.cameras.RTSP_STREAMS = [
    {
        "name": "Camera 1",
        "url": "rtsp://192.168.1.100:554/stream",
        "width": 1920,
        "height": 1080
    }
]

# Run the system
main()
```

### GPU RTSP Pipeline

The system automatically uses PyNvCodec for pure GPU RTSP processing:

```python
# Automatic GPU pipeline selection (default)
config.processing.USE_LEGACY_NVDEC_READER = False

# Force legacy NVDEC if needed
config.processing.USE_LEGACY_NVDEC_READER = True
```

### Configuration

Edit `config.py` or `config.json` to customize:

- **Video Sources**: RTSP streams, video files, webcams
- **Model Settings**: Detection thresholds, TensorRT optimization
- **Processing**: GPU acceleration, batch sizes, threading
- **Visualization**: WebSocket server, display options

## GPU RTSP Pipeline

### Overview

The GPU RTSP pipeline provides zero-copy video processing:

```
RTSP Stream → PyNvCodec Decoder → GPU Surface → DALI Pipeline → TensorRT Inference
```

### Benefits

- **60%+ CPU Reduction**: Eliminates GPU↔CPU memory transfers
- **Lower Latency**: Direct GPU surface processing
- **Higher Throughput**: Hardware-accelerated decode + preprocessing
- **Memory Efficiency**: Reduced GPU memory fragmentation

### Troubleshooting

If PyNvCodec is not available, the system automatically falls back to legacy NVDEC:

```bash
# Check PyNvCodec installation
python -c "import nvidia.vpf; print('PyNvCodec available')"

# Run validation tests
python -m pytest tests/test_rtsp_gpu_path.py -v
```

## Usage Examples

### Multi-Stream Processing

```python
# Configure multiple RTSP streams
config.cameras.RTSP_STREAMS = [
    {"name": "Camera 1", "url": "rtsp://192.168.1.100:554/stream"},
    {"name": "Camera 2", "url": "rtsp://192.168.1.101:554/stream"},
    {"name": "Camera 3", "url": "rtsp://192.168.1.102:554/stream"}
]

# Enable unified GPU pipeline
config.processing.USE_UNIFIED_GPU_PIPELINE = True
```

### Performance Optimization

```python
# Enable TensorRT optimization
config.models.ENABLE_TENSORRT = True
config.models.TENSORRT_FP16 = True

# GPU memory optimization
config.processing.ENABLE_MEMORY_POOLING = True
config.processing.GPU_MEMORY_POOL_SIZE_MB = 512

# DALI preprocessing
config.processing.ENABLE_DALI = True
config.processing.DALI_TARGET_WIDTH = 640
config.processing.DALI_TARGET_HEIGHT = 640
```

### Web Dashboard

Access the live dashboard at `http://localhost:6008` after starting the system:

```python
# Enable WebSocket server
config.websocket.ENABLE_SERVER = True
config.websocket.PORT = 6008
config.websocket.MAX_FPS = 20
```

## Architecture

### Components

- **Video Input**: RTSP streams, video files, webcams
- **GPU Pipeline**: PyNvCodec decoder + DALI preprocessing
- **AI Models**: YOLO detection + pose estimation (TensorRT optimized)
- **Tracking**: ByteTrack multi-object tracking
- **Output**: WebSocket streaming + file export

### Performance

- **RTSP Decode**: 60+ FPS (1080p)
- **Detection**: 100+ FPS (640x640)
- **End-to-End**: 30+ FPS (multi-stream)
- **Memory**: <2GB GPU memory (4 streams)

## Configuration Reference

### Key Settings

```python
# GPU RTSP Pipeline
config.processing.USE_LEGACY_NVDEC_READER = False  # Use PyNvCodec
config.processing.ENABLE_DALI = True               # Enable DALI preprocessing

# TensorRT Optimization
config.models.ENABLE_TENSORRT = True               # Enable TensorRT
config.models.TENSORRT_FP16 = True                 # Use FP16 precision
config.models.FORCE_GPU_ONLY = True                # No CPU fallbacks

# Performance
config.processing.ENABLE_MEMORY_POOLING = True     # GPU memory pooling
config.processing.TARGET_FPS = 30                  # Target processing FPS
```

### RTSP Stream Configuration

```python
config.cameras.RTSP_STREAMS = [
    {
        "name": "Living Room",
        "url": "rtsp://192.168.1.100:554/stream",
        "width": 1920,
        "height": 1080,
        "username": "admin",      # Optional
        "password": "password"    # Optional
    }
]
```

## Troubleshooting

### Common Issues

1. **PyNvCodec Not Available**
   ```bash
   pip install nvidia-pyindex nvidia-vpf
   ```

2. **CUDA Out of Memory**
   - Reduce number of streams
   - Lower resolution
   - Enable memory pooling

3. **RTSP Connection Failures**
   - Verify stream URL
   - Check network connectivity
   - Try different transport (TCP/UDP)

4. **Low Performance**
   - Enable TensorRT optimization
   - Use GPU preprocessing
   - Adjust batch sizes

### Debug Mode

Enable detailed logging:

```python
config.app.DEBUG = True
config.app.LOG_LEVEL = 10  # DEBUG
config.processing.ENABLE_PROFILING = True
```

### Validation

Run comprehensive tests:

```bash
# Test GPU RTSP pipeline
python -m pytest tests/test_rtsp_gpu_path.py -v

# Test TensorRT engines
python -m pytest tests/test_tensorrt.py -v

# Performance benchmarks
python performance_benchmark.py --duration 60
```

## Documentation

- [GPU RTSP Pipeline Guide](docs/GPU_RTSP_Pipeline.md)
- [Configuration Reference](docs/Configuration.md)
- [Performance Tuning](docs/Performance.md)
- [API Documentation](docs/API.md)

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Run tests: `python -m pytest tests/ -v`
4. Submit pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/ -v --cov=.
```

## License

[Your License Here]

## Support

For issues and questions:

1. Check the [troubleshooting guide](docs/GPU_RTSP_Pipeline.md#troubleshooting)
2. Run validation tests
3. Enable debug logging
4. Create an issue with logs and system info

## Changelog

### Latest Version

#### Added
- Native GPU RTSP ingest via PyNvCodec (no CPU hop)
- DALIExternalSourcePipeline for zero-copy processing
- Automatic pipeline selection logic
- Comprehensive error handling and reconnection
- Performance monitoring and statistics

#### Changed
- create_optimal_dali_pipeline auto-selects GPU reader for rtsp://
- Enhanced resource cleanup and memory management
- Improved fallback behavior for legacy compatibility

See [CHANGELOG.md](CHANGELOG.md) for complete version history. 