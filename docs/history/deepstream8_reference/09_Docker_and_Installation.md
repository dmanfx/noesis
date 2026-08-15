# 09. Docker and Installation (DS8)

This page summarizes how to get DeepStream 8 up and running on x86/dGPU and Jetson using NVIDIA’s official containers. Use containers for consistent dev/test environments.

## x86/dGPU Containers

- Images are hosted on `nvcr.io` under `nvidia/deepstream`.
- Common tags:
  - `deepstream:8.0-gc-triton-devel` – full SDK + graph composer + Triton backends for development.
  - `deepstream:8.0-triton-multiarch` – Triton + dev environment.
  - `deepstream:8.0-samples-multiarch` – runtime + plugins + reference apps and sample assets.

Login and pull:

```
docker login nvcr.io
docker pull nvcr.io/nvidia/deepstream:8.0-gc-triton-devel
```

Run with GPU access and X11 for display:

```
docker run --rm -it --gpus all \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $PWD:/workspace \
  nvcr.io/nvidia/deepstream:8.0-gc-triton-devel /bin/bash
```

Inside the container, sample apps live under `/opt/nvidia/deepstream/deepstream/sources/apps/`.

## Jetson Containers

- Images are hosted on `nvcr.io` under `nvidia/deepstream` (Jetson). Typical tags:
  - `deepstream:8.0-samples-multiarch`
  - `deepstream:8.0-triton-multiarch`
- Jetson containers include runtime and samples; development is typically done natively on device, then containerized.
- Ensure JetPack BSP (per DS8 release notes) and NVIDIA Container Runtime are installed on the device.

Basic run command on Jetson:

```
sudo docker run --rm -it --runtime nvidia \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  nvcr.io/nvidia/deepstream:8.0-samples-multiarch /bin/bash
```

## Installing Non-Container (x86)

Container is recommended. If installing natively, align NVIDIA driver, CUDA, TensorRT, and DeepStream versions per DS8 release notes. Verify with `nvidia-smi`, `gst-inspect-1.0 | rg nvinfer`, and run sample apps.

## Useful Host Packages in Containers

- Some pipelines need these inside the container:
  - `gstreamer1.0-libav`, `gstreamer1.0-plugins-good` (for parsers/demuxers)
  - Protocol libs for `nvmsgbroker` (Kafka/MQTT adapters)

Install example:

```
apt-get update && apt-get install -y \
  gstreamer1.0-libav gstreamer1.0-plugins-good \
  libssl-dev librdkafka-dev
```

## Verifications

- Test GPU access: `nvidia-smi` (x86) or `tegrastats` (Jetson).
- List DS plugins: `gst-inspect-1.0 | rg -i "nv(stream|infer|tracker|ds|msg)"`.
- Run a sample pipeline and confirm decode + inference + OSD.
