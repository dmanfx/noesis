# 03. DeepStream Sample Applications Guide

This document provides a guide to the most relevant DeepStream sample applications, focusing on the Python examples. Each entry includes a summary of the concepts it demonstrates, helping you quickly identify which sample to consult for a specific task.

The Python samples are typically located in:
`/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/`

---

### `deepstream-test1`

*   **Location:** `deepstream-test1/`
*   **Demonstrates:**
    *   The most basic DeepStream pipeline.
    *   **Pipeline:** File Source -> Decoder -> StreamMuxer -> Primary Inference -> Tiler -> OSD -> Sink.
    *   How to build a pipeline using Python bindings.
    *   Basic `nvinfer` configuration for a single detector.
    *   Attaching a "probe" to a Gst Pad to access metadata.
*   **When to use:** As the "Hello World" of DeepStream. Excellent for understanding the fundamental structure of a Python-based DS application.

---

### `deepstream-test2`

*   **Location:** `deepstream-test2/`
*   **Demonstrates:**
    *   Adding a secondary inference engine to the pipeline.
    *   **Pipeline:** Adds a `nvtracker` and a secondary `nvinfer` to the `test1` pipeline.
    *   How to run a classifier model on the objects detected by the primary model.
    *   Configuration of `nvinfer` to operate in secondary mode.
*   **When to use:** When you need to understand how to chain inference models (e.g., a vehicle detector followed by a make/model classifier).

---

### `deepstream-test3`

*   **Location:** `deepstream-test3/`
*   **Demonstrates:**
    *   Handling multiple video stream inputs.
    *   **Pipeline:** Creates multiple source bins (source + decoder) and links them to a single `nvstreammux`.
    *   How to dynamically add and remove sources from a running pipeline.
*   **When to use:** A critical reference for any application that needs to process more than one camera feed simultaneously.

---

### `deepstream-test4` (Analytics)

*   **Location:** `deepstream-test4/`
*   **Demonstrates:**
    *   Using the `nvdsanalytics` plugin.
    *   Defining lines and polygons (Regions of Interest - ROIs) in the analytics configuration file.
    *   Accessing analytics metadata from a probe function to count line crossings or objects within an ROI.
*   **When to use:** When you need to implement business logic based on object location or movement, such as counting cars entering a parking lot.

---

### `deepstream-test5` (IoT and Messaging)

*   **Location:** `deepstream-test5/`
*   **Demonstrates:**
    *   A complete end-to-end IoT application.
    *   Sending DeepStream metadata to a message broker (e.g., Kafka, MQTT) or REST endpoint.
    *   Using the `nvmsgconv` and `nvmsgbroker` plugins.
    *   Includes an example message consumer to show how to receive and parse the metadata.
*   **When to use:** Essential reference for any task involving sending pipeline results to a server or dashboard. This is the foundation for building headless analytics applications.

---

### `deepstream-image-meta-test`

*   **Location:** `deepstream-image-meta-test/`
*   **Demonstrates:**
    *   A more advanced use case of metadata handling.
    *   Attaching custom user metadata (`NvDsUserMeta`) to the buffer.
    *   Saving cropped images of detected objects.
    *   Saving the full frame image.
*   **When to use:** When you need to save images of detections for later review or for training new models. Also a good example for how to extend the built-in metadata.

---

### `deepstream-rtsp-in-rtsp-out`

*   **Location:** `deepstream-rtsp-in-rtsp-out/`
*   **Demonstrates:**
    *   A full pipeline that takes an RTSP stream as input and exposes the processed output as another RTSP stream.
    *   Proper configuration of RTSP source and RTSP sink elements.
*   **When to use:** The canonical example for building applications that provide a processed video feed to other systems or for remote viewing. 