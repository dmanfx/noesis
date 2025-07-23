#!/usr/bin/env python3
"""Minimal ONNX → TensorRT engine builder.
Usage: python3 build_engine.py [onnx_path] [engine_out]
Defaults: models/yolo11m.onnx → models/engines/yolo11m_fp16.engine
"""
import sys, os, tensorrt as trt

onnx_path   = sys.argv[1] if len(sys.argv) > 1 else "models/yolo11m.onnx"
engine_path = sys.argv[2] if len(sys.argv) > 2 else "models/engines/yolo11m_fp16.engine"

logger  = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser  = trt.OnnxParser(network, logger)

# Precision flag (fp32 default). Pass 'fp16' as 3rd CLI arg to enable FP16.
precision = (sys.argv[3].lower() if len(sys.argv) > 3 else "fp32")

with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        sys.exit(1)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB
if precision == "fp16":
    config.set_flag(trt.BuilderFlag.FP16)

profile = builder.create_optimization_profile()
input_tensor = network.get_input(0)  # assumes single input
profile.set_shape(input_tensor.name, (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))
config.add_optimization_profile(profile)

engine = builder.build_serialized_network(network, config)
if engine is None:
    print("Engine build failed")
    sys.exit(1)

os.makedirs(os.path.dirname(engine_path), exist_ok=True)
with open(engine_path, "wb") as f:
    f.write(engine)
print("✅ Saved engine:", engine_path) 