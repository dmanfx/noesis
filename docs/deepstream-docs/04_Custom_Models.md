# 04. Using Custom Models with DeepStream

This document outlines the process and best practices for integrating custom-trained AI models into a DeepStream pipeline. The core plugin for this is `nvinfer`.

## The Role of TensorRT

DeepStream does not directly use models in their native format (like TensorFlow `.pb` or PyTorch `.pt`). For high-performance inference, DeepStream requires models to be converted into a **TensorRT engine**.

**TensorRT** is an NVIDIA SDK for high-performance deep learning inference. It takes a trained network and applies numerous optimizations, such as:
*   **Layer and Tensor Fusion:** Merging multiple layers into a single operation.
*   **Precision Calibration:** Optimizing the model to run at lower precision (FP16, INT8) with minimal accuracy loss, which significantly increases throughput.
*   **Kernel Auto-Tuning:** Selecting the most efficient CUDA kernels for the target dGPU.

The result of this process is a `.engine` file, which is a self-contained, optimized version of your model ready for deployment.

## The Workflow for Custom Models

1.  **Train Your Model:** Train your detection or classification model using your framework of choice (PyTorch, TensorFlow, etc.).

2.  **Convert to an Intermediate Format (Optional but Recommended):**
    *   It is often easiest to first export your model to the **ONNX (Open Neural Network Exchange)** format. Most modern frameworks have built-in support for this.
    *   ONNX provides a standardized format that TensorRT can reliably parse.

3.  **Build the TensorRT Engine:**
    *   Use the `trtexec` command-line tool (provided with TensorRT) or a custom script (like `tensorrt_builder.py` in this project) to convert the ONNX model into a TensorRT `.engine` file.
    *   During this step, you will specify the desired precision (e.g., FP32, FP16). For FP16, the conversion is usually straightforward. For INT8, it requires a calibration step with a representative dataset.

4.  **Configure the `nvinfer` Plugin:**
    *   This is the most critical step. You must create a configuration file for `nvinfer` that tells it how to use your custom model.
    *   Key parameters in this file include:
        *   `model-engine-file`: Path to your generated `.engine` file.
        *   `network-type`: `0` for a detector, `1` for a classifier.
        *   `num-detected-classes`: The number of classes your object detector can identify.
        *   `cluster-mode`: How to cluster bounding boxes (e.g., `2` for DBSCAN).
        *   `parse-bbox-func-name`: **Crucially**, the name of the function used to parse the model's raw output tensor into bounding boxes.

5.  **Implement a Custom Output Parser (If Necessary):**
    *   The `nvinfer` plugin has built-in parsers for many standard model architectures (like SSD, Faster R-CNN).
    *   However, for custom models, especially custom-trained YOLO variants, the exact format of the output tensor (the "head" of the network) may be different.
    *   In this case, you must provide a **custom output parsing function**. This is typically a C/C++ function compiled into a shared library (`.so` file) that `nvinfer` can load.
    *   This function receives the raw tensor output from the model and is responsible for decoding it into a list of `NvDsInferObjectDetectedInfo` structs (which contain `left`, `top`, `width`, `height`, `detectionConfidence`, and `classId`).
    *   The `libnvdsparsebbox_yolo11.cpp` in this project is an example of such a custom parser.

6.  **Integrate and Test:**
    *   Point the `nvinfer` element in your Python script to the new configuration file.
    *   Run the pipeline and verify that the detections are correct. The `nvdsosd` plugin is invaluable here for visualizing the bounding boxes produced by your custom parser.

## Key Takeaways

*   DeepStream requires a **TensorRT engine**, not the raw model file.
*   The **`nvinfer` configuration file** is the bridge between your DeepStream application and your custom model.
*   For non-standard model outputs, you **must provide a custom output parsing function** compiled as a shared library. This is the most common point of failure and success when integrating custom detectors. 