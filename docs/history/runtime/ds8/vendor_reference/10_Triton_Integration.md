# 10. Triton Integration in DeepStream 8

DeepStream integrates Triton Inference Server via the `nvinferserver` plugin and the NvDsTritonExt extension, enabling multi-model serving and framework flexibility.

## When to Use Triton

- Serve multiple models or ensembles in one process.
- Use non-TensorRT backends (e.g., PyTorch, ONNXRuntime) supported by the shipped Triton distribution.
- Deploy the same model server to multiple apps.

Note: For some models, direct `nvinfer` may yield higher throughput/latency than Triton; measure and choose based on your workload.

## Basic Concepts

- `nvinferserver` connects to Triton over HTTP or gRPC and performs inference on DeepStream batched surfaces.
- Models are organized in a Triton model repository; input/output names and shapes must match what the plugin expects.

## Minimal `nvinferserver` Pipeline Sketch

```
... ! nvstreammux name=mux batch-size=8 width=1280 height=720 ! \
  nvinferserver config-file-path=triton_pgie_infer.txt ! ...
```

Example `triton_pgie_infer.txt` (high level):

```
infer_config {
  unique_id: 1
  backend {
    triton {
      model_name: "yolov8_trt"
      version: -1
      url: "localhost:8001"  # gRPC
      use_ssl: false
    }
  }
  preprocess {
    network_input_format: IMAGE_RGB
    tensor_order: TENSOR_ORDER_LINEAR
    maintain_aspect_ratio: 0
  }
  # postprocess / labels config as per your model
}
input_control { process_mode: PRIMARY }
```

Adjust to your model’s repository layout, I/O names, and pre/post-processing requirements.

## NvDsTritonExt

- Extension for DeepStream to work with Triton backends packaged for DS8.
- Some backends may not be available on all platforms; consult DS8 release notes for supported backend versions.

## Operational Tips

- Co-locate Triton in the same container or host it externally; ensure network reachability.
- Prefer gRPC for lower overhead; HTTP is also supported.
- Batch sizing: align `nvstreammux` batch with Triton backend expectations; verify dynamic batching settings in Triton configs if enabled.
- Monitor performance: compare `nvinfer` vs `nvinferserver` and choose per model.
