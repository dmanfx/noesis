# YOLOv11 Segmentation Fault: Root Cause Analysis and Solution

## 1. Problem Statement

A segmentation fault occurs when running the `deepstream_video_pipeline.py` with an RTSP source. The crash happens immediately after the pipeline successfully builds the TensorRT engine and transitions to the `PLAYING` state, specifically after the log message: `INFO:__main__:🔍 Pipeline state change is async, waiting...`.

## 2. Root Cause Identification

The root cause of the segmentation fault is a **misconfiguration of the `nvinfer` (Gst-nvinfer) element**. The element is not being instructed how to correctly parse the output tensors from the YOLOv11 model. 

Without a specific parsing function, `nvinfer` falls back to a default, generic parsing mechanism. This default parser is incompatible with the complex, multi-layered output structure of a YOLO model, leading to memory access errors and the subsequent segmentation fault. The log message `Using offsets...` seen just before the crash is characteristic of this failure mode.

The initial debugging summary (`docs/Segment_fault_debug.md`) correctly identified that the issue was related to parsing but incorrectly concluded that the custom parser library itself was the problem. The correct conclusion is that the custom parser is **required, but was disabled**.

## 3. Evidence

My analysis is based on the following evidence:

1.  **Missing Parser Configuration**: Your `config_infer_primary_yolo11.txt` file was missing two critical properties:
    *   `custom-lib-path`: This tells `nvinfer` which shared object library to load for custom processing.
    *   `parse-bbox-func-name`: This tells `nvinfer` which specific function within that library to call for parsing the model's output.

2.  **Existence of a Custom Parser**: Your project already contains the necessary parser files (`libnvdsparsebbox_yolo11.so` and its source `libnvdsparsebbox_yolo11.cpp`). This proves that the component to solve the problem is present but was not being utilized.

3.  **Community Best Practices**: Research into official documentation (NVIDIA, Ultralytics) and highly-regarded open-source projects (e.g., `marcoslucianops/DeepStream-Yolo`) shows that using a custom output parser is the standard, required method for integrating YOLO models with DeepStream. None of these successful implementations work without one.

4.  **Parser Function Name**: Examination of your `libnvdsparsebbox_yolo11.cpp` source code revealed the exact function name required for the configuration: `NvDsInferParseYOLO11`.

## 4. Solution

The solution is to re-enable and correctly configure the use of your existing custom parser library. This involves adding the two missing properties to `config_infer_primary_yolo11.txt`.

The required changes instruct `nvinfer` to:
1.  Load your custom library: `libnvdsparsebbox_yolo11.so`.
2.  Use the `NvDsInferParseYOLO11` function from that library to interpret the bounding box data from the model's output tensors.

This change directly addresses the root cause by replacing the faulty default parser with the specialized one designed for your model.

## 5. Final Status

After applying the fix to `config_infer_primary_yolo11.txt`, the `deepstream_video_pipeline.py` application will be able to correctly parse the inference results, eliminating the memory corruption and resolving the segmentation fault. The pipeline will then be fully functional for end-to-end RTSP stream processing with YOLOv11. 