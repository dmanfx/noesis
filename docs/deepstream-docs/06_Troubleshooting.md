# 06. Troubleshooting & FAQ (dGPU)

This document provides a list of common issues, questions, and troubleshooting steps for developing DeepStream applications on a dGPU with Ubuntu.

---

### Q: My pipeline fails to build or hangs on startup. What should I check first?

**A:** This is often caused by a mismatch in capabilities ("Caps") between two linked GStreamer elements.

1.  **Check the logs:** Run your Python script with the `GST_DEBUG` environment variable set. This provides verbose logging about the GStreamer pipeline's negotiation process. A good starting point is:
    ```bash
    GST_DEBUG=3 python3 your_app.py
    ```
2.  **Look for "not-linked" errors:** The logs will often point to the exact elements that failed to link and why (e.g., "reason: could not agree on Caps").
3.  **Mismatched Resolution/Format:** A common cause is the `nvstreammux` outputting at a resolution that a downstream element (like `nvinfer`) is not configured to accept. Ensure the `width` and `height` are consistent.
4.  **Mismatched Batch Size:** Ensure the `batch-size` property is identical between your `nvstreammux` and `nvinfer` elements.

---

### Q: I'm seeing "Error: Could not get cuda device count" or similar CUDA errors.

**A:** This usually indicates an issue with the NVIDIA driver or CUDA toolkit installation.

1.  **Run `nvidia-smi`:** Open a terminal and run this command. It should successfully print a table with your GPU information and the driver version. If it fails, your driver is not installed correctly.
2.  **Check Driver/CUDA Version:** Ensure the installed NVIDIA driver version meets the minimum requirement for your version of the DeepStream SDK. This information is in the official NVIDIA documentation.
3.  **Environment:** If running in a container, ensure it was started with the `--gpus all` flag to properly expose the GPU devices to the container.

---

### Q: My custom model is not detecting anything, or the bounding boxes are nonsensical.

**A:** This is almost always an issue with the `nvinfer` configuration or the custom output parser.

1.  **Validate the Engine:** Ensure your TensorRT `.engine` file was built correctly for your target GPU and with the correct precision (FP16/FP32). Use `trtexec` to benchmark the engine standalone to confirm it can run.
2.  **Check the `nvinfer` Config File:**
    *   Is the `num-detected-classes` correct?
    *   Is the `network-type` set correctly (0 for detector)?
    *   Most importantly, is the output parsing configured correctly?
3.  **The Custom Parser (`libnvdsparsebbox_...`):**
    *   This is the most likely culprit. The logic inside your C/C++ parsing function must perfectly match the output tensor layout of your specific model.
    *   Add `printf` statements inside your parser function to print the raw tensor values and the dimensions of the output layers. Recompile, run the pipeline, and check the console. This will tell you if the function is being invoked and what data it's seeing.
    *   Verify that you are correctly calculating the bounding box coordinates (`left`, `top`, `width`, `height`) and that they are scaled to the dimensions of the `nvstreammux` buffer, not the model's input dimensions.

---

### Q: The video output is choppy or the pipeline cannot keep up with a real-time stream.

**A:** This is a performance issue. Refer to `05_Performance.md` for a full guide. The quick checklist is:

1.  **Increase Batch Size:** This is the most effective way to improve throughput.
2.  **Use FP16 Precision:** Convert your model to FP16. The performance gain is usually significant for a minimal accuracy trade-off.
3.  **Check for Memory Copies:** Are you accessing pixel data on the CPU in a probe function? This is very slow. Try to limit probes to only accessing metadata.
4.  **Lower Input Resolution:** In the `nvstreammux` config, reduce the `width` and `height` to the minimum required for your model to function.
5.  **Use a Lighter Tracker:** The IOU tracker is faster than NvDCF.

---

### Q: I'm getting an "out of memory" error.

**A:** This means your pipeline is trying to allocate more VRAM than your dGPU has available.

1.  **Reduce Batch Size:** This is the primary consumer of VRAM.
2.  **Check `nvstreammux` properties:** If `num-surfaces-per-frame` is set, ensure it's not excessively high.
3.  **Check Number of Streams:** Processing many high-resolution streams simultaneously requires a large amount of memory for the decoded buffers in the `nvstreammux`.
4.  **Use `nvidia-smi`:** Monitor the "Memory-Usage" column of the `nvidia-smi` output while your application is running to see exactly how much VRAM is being used. 