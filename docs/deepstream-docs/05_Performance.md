# 05. Performance Tuning & Optimization (dGPU)

This document covers key strategies and configurations for optimizing the performance of DeepStream applications on a dGPU setup.

## The Goal of Optimization

The primary goal is to maximize **throughput** (the number of frames or streams processed per second) while maintaining an acceptable level of accuracy and keeping resource utilization (GPU, CPU, memory) within reasonable limits.

## Key Areas for Performance Tuning

### 1. Batch Size (`nvstreammux` and `nvinfer`)

*   **Impact:** This is the single most important parameter for performance.
*   **Concept:** The dGPU achieves maximum efficiency when it can process many frames in parallel. Increasing the `batch-size` allows the `nvinfer` plugin to run inference on a larger batch of images at once, which better saturates the GPU's computational cores.
*   **How to Tune:**
    *   In the `nvstreammux` configuration, set the `batch-size`.
    *   Ensure the `batch-size` in the `nvinfer` config file matches the `nvstreammux` setting.
    *   Increase the batch size incrementally and measure the throughput until you either hit a performance plateau or exceed the GPU's memory capacity.
*   **Trade-off:** Larger batch sizes introduce higher latency, as the system must wait for the batch to be filled before processing. For real-time applications, you must find a balance between throughput and latency.

### 2. Model Precision (FP16/INT8)

*   **Impact:** High.
*   **Concept:** By default, models run at full 32-bit precision (FP32). Using lower precision can dramatically increase performance.
    *   **FP16 (Half Precision):** Offers a significant speedup (often ~2x) with a negligible drop in accuracy for most models. It also cuts the model's memory footprint in half. This is often the "sweet spot" for performance vs. accuracy.
    *   **INT8 (8-bit Integer):** Offers the highest performance (another ~2x over FP16) but requires a **calibration** step. During calibration, TensorRT analyzes the distribution of weights and activations with a sample dataset to determine how to quantize them to 8-bit integers without a significant loss of accuracy.
*   **How to Tune:**
    *   When building the TensorRT engine (e.g., with `trtexec`), specify the `--fp16` or `--int8` flag.
    *   For INT8, you must also provide a calibration cache.

### 3. Tracker Selection and Configuration (`nvtracker`)

*   **Impact:** Medium to High.
*   **Concept:** Object tracking can be computationally expensive. The choice of tracker and its configuration can significantly affect performance.
*   **How to Tune:**
    *   The **IOU (Intersection over Union)** tracker is generally the fastest and most lightweight. It's less accurate than other methods but is very efficient. It's a good baseline.
    *   The **NvDCF** tracker is more robust and accurate but also more computationally intensive.
    *   Experiment with the `max-shadow-tracking-age` and `min-hits` parameters in the tracker's configuration file. Lowering these values can reduce the computational load but may result in more fragmented tracks.

### 4. Zero-Copy and Memory Management

*   **Impact:** High.
*   **Concept:** The goal is to keep data on the GPU as much as possible to avoid expensive memory copies between the CPU (system RAM) and GPU (VRAM).
*   **How to Tune:**
    *   **Use hardware decoders (`nvv4l2decoder`):** This ensures decoded frames land directly in GPU memory.
    *   **Keep processing on the GPU:** Use GPU-accelerated plugins wherever possible (e.g., `process-mode=0` in `nvdsosd`).
    *   If you need to access frame data on the CPU (e.g., in a probe function), do so sparingly. Accessing the metadata is cheap; accessing the pixel data itself can cause a performance-killing `cudaMemcpy` operation. The `pyds.get_nvds_buf_surface()` function is an example of an operation that can trigger this if not used carefully.

### 5. Input Resolution and Framerate

*   **Impact:** High.
*   **Concept:** The size and rate of the input data directly impact the workload.
*   **How to Tune:**
    *   In the `nvstreammux` config, set the `width` and `height`. Is it necessary to process the video at its full 4K resolution, or would 1080p suffice for the model to make accurate detections? Downscaling the processing resolution significantly reduces the load on all downstream elements.
    *   If the source camera is sending frames at 30 FPS, but the events you care about happen much slower, you can configure `nvinfer` to skip frames (`interval > 0`). This is a very effective way to reduce the processing load. 