#!/usr/bin/env python3
"""Minimal ONNX → TensorRT engine builder for MapAnything depth model.
Usage: python3 build_engine.py [onnx_path] [engine_out] [precision]
Defaults: ma_onnx_out/model_embedded.onnx → models/engines/mapanything_depth_fp16.engine fp16
"""
import sys, os
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

onnx_path = sys.argv[1] if len(sys.argv) > 1 else "ma_onnx_out/model-dyn.onnx"
engine_path = sys.argv[2] if len(sys.argv) > 2 else "models/engines/mapanything_depth_fp16.engine"
precision = (sys.argv[3].lower() if len(sys.argv) > 3 else "fp16")

with open(onnx_path, "rb") as model:
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            sys.exit(1)

        # Debug prints
        print(f"Parsed network: {network.num_inputs} inputs, {network.num_outputs} outputs")
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            print(f"Input {i}: {inp.name} shape {inp.shape}")
        for i in range(network.num_outputs):
            out = network.get_output(i)
            print(f"Output {i}: {out.name} shape {out.shape}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 32 * 1024 * 1024 * 1024)  # 32 GB

        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)

        # Add dynamic profile for batch 1-3
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)  # Assume first input is the main (images+intrinsics)
        profile.set_shape(input_tensor.name, min=(1, 12, 518, 518), opt=(2, 12, 518, 518), max=(3, 12, 518, 518))
        config.add_optimization_profile(profile)

        # For static model, no profile needed; explicit batch handles it
        # If dynamic, add profile here

        serialized_engine = builder.build_serialized_network(network, config)
        if not serialized_engine:
            print("ERROR: Failed to build engine")
            sys.exit(1)

os.makedirs(os.path.dirname(engine_path), exist_ok=True)
with open(engine_path, "wb") as f:
    f.write(serialized_engine)
print(f"✅ Saved engine: {engine_path}") 