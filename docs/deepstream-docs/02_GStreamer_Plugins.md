# 02. DeepStream GStreamer Plugins (dGPU)

This document provides a detailed reference for the essential GStreamer plugins provided by the DeepStream SDK, focusing on those relevant for a dGPU on Ubuntu deployment.

---

### `nvstreammux` (Stream Muxer)

The stream muxer is one of the most critical components in a multi-stream DeepStream pipeline.

*   **Function:** Combines multiple input streams (from decoders) into a single batched buffer. This is essential for maximizing dGPU utilization, as inference is performed on a batch of frames at once.
*   **Key Properties:**
    *   `width`, `height`: The resolution of the output buffer. Input streams will be scaled to this size.
    *   `batch-size`: The number of frames from a single source to batch together.
    *   `batched-push-timeout`: The maximum time (in microseconds) the muxer waits to form a batch. If a new buffer doesn't arrive in this time, it pushes the incomplete batch downstream. Setting this to `-1` can help in live streams to push frames as soon as they come.
    *   `live-source`: Set to `true` for live streams (like RTSP) to manage synchronization.
*   **Output:** A single video stream containing batched `NvBufSurface` buffers. Each frame in the batch is identifiable by its `batch_id` in the metadata.

---

### `nvv4l2decoder` (Video Decoder)

*   **Function:** Hardware-accelerated decoding of compressed video streams (H.264, H.265, etc.). It offloads the CPU-intensive decoding process to the dGPU's dedicated hardware decoder (NVDEC).
*   **Key Properties:**
    *   It has very few user-configurable properties. It is largely a "plug-and-play" element that is placed right after the source.
*   **Output:** Decoded raw video frames in GPU memory (`NV12` format typically).

---

### `nvinfer` (Inference Engine)

This is the core plugin for running AI models.

*   **Function:** Performs inference using a TensorRT engine. It can function as a primary detector or a secondary classifier/regressor.
*   **Key Properties:**
    *   `config-file-path`: The path to a detailed configuration file that specifies the model, input/output layers, preprocessing, post-processing, etc.
    *   `model-engine-file`: Path to the pre-built TensorRT engine file.
    *   `batch-size`: The number of frames to infer on simultaneously. This must match the `batch-size` of the upstream `nvstreammux`.
    *   `process-mode`: Set to `1` for primary inference or `2` for secondary inference.
    *   `unique-id`: A unique identifier for the `nvinfer` instance, especially when you have multiple in the pipeline.
*   **Output:** Attaches `NvDsInferTensorMeta` and `NvDsObjectMeta` (for detectors) or `NvDsClassifierMeta` (for classifiers) to the frame's metadata.

---

### `nvtracker` (Object Tracker)

*   **Function:** Assigns and maintains a unique ID for each detected object across consecutive frames. This transforms a simple list of detections into meaningful object tracks.
*   **Key Properties:**
    *   `ll-lib-file`: Path to the specific tracker library to use (e.g., NvDCF, IOU, KLT).
    *   `ll-config-file`: Path to the configuration file for the chosen tracker library.
    *   `gpu-id`: The ID of the GPU to use for tracking.
*   **Output:** Adds a unique `object_id` to the `NvDsObjectMeta` for each object it tracks. This ID persists for as long as the tracker can follow the object.

---

### `nvdsosd` (On-Screen Display)

*   **Function:** Renders visual information onto the video frames. It's primarily used for debugging and creating visual outputs.
*   **Key Properties:**
    *   `show-bbox`: Enable/disable rendering of bounding boxes.
    *   `show-text`: Enable/disable rendering of text labels.
    *   `process-mode`: `0` for GPU, `1` for CPU. Should be kept on GPU for performance.
*   **Output:** A video stream with the specified visuals (boxes, text, masks) burned into the frames.

---

### `nvdsanalytics` (Video Analytics)

*   **Function:** Performs higher-level scene analytics based on object tracking data.
*   **Key Properties:**
    *   `config-file-path`: Path to a configuration file where you can define rules, like lines or polygons of interest.
*   **Output:** Attaches `NvDsAnalyticsFrameMeta` metadata, which can contain information about events like line crossings, objects in a region, and directional movement.

---

### `nvdspreprocess` (Custom Preprocessor)

*   **Function:** A flexible plugin for custom preprocessing when `nvinfer`'s internal capabilities are not sufficient. It allows for complex custom transformations.
*   **Key Properties:**
    *   `config-file-path`: Path to a configuration file that defines the custom processing chain.
*   **Output:** A preprocessed buffer ready for a downstream `nvinfer` element.

---

### `nvmsgconv` & `nvmsgbroker` (IoT Messaging)

These two plugins work together to send DeepStream metadata to the cloud or an external server.

*   **`nvmsgconv` (Message Converter):**
    *   **Function:** Converts the DeepStream metadata (`NvDsBatchMeta`) into a specific message schema (e.g., JSON).
    *   **Key Properties:** `config-file-path` specifies the conversion schema.
*   **`nvmsgbroker` (Message Broker):**
    *   **Function:** Takes the formatted message from `nvmsgconv` and sends it to a message broker (like Kafka, MQTT, or a simple REST endpoint) using a specified protocol adapter.
    *   **Key Properties:** `proto-lib` (path to the protocol adapter), `conn-str` (connection string for the broker), `topic`.
*   **Output:** These are "sink" plugins. They do not output a video stream but send data out of the pipeline. 