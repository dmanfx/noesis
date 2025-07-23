#!/usr/bin/env python3
"""
convert_pt_to_engine.py
-----------------------
Convert a PyTorch `.pt` model checkpoint to a TensorRT `.engine` file.

Usage:
    python convert_pt_to_engine.py /path/to/model.pt /path/to/output/engine.engine \
                                   --input-shape 1 3 640 640 --fp16

Positional arguments:
    input_pt         Path to the input .pt file (PyTorch checkpoint / scripted model).
    output_engine    Destination path for the generated .engine file.

Optional arguments:
    --input-shape    Four integers representing N C H W (default: 1 3 640 640).
    --fp16           Enable FP16 precision (default: enabled). Add `--no-fp16` to disable.

The script performs the following steps:
1. Loads the .pt model.
2. Exports it to a temporary ONNX file.
3. Invokes NVIDIA's `trtexec` to build the TensorRT engine.
4. Cleans up the temporary ONNX file.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .pt model to TensorRT .engine")
    parser.add_argument("input_pt", type=str, help="Path to the input .pt model file")
    parser.add_argument("output_engine", type=str, help="Destination path for the output .engine file")
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=4,
        metavar=("N", "C", "H", "W"),
        default=(1, 3, 640, 640),
        help="Input tensor shape as N C H W (default: 1 3 640 640)",
    )
    fp16_group = parser.add_mutually_exclusive_group()
    fp16_group.add_argument("--fp16", dest="fp16", action="store_true", help="Enable FP16 precision (default)")
    fp16_group.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16 precision")
    parser.set_defaults(fp16=True)
    return parser.parse_args()


def find_trtexec() -> str:
    """Locate the `trtexec` executable in PATH."""
    trtexec_path = shutil.which("trtexec")
    if trtexec_path is None:
        sys.stderr.write("ERROR: `trtexec` not found in PATH. Please install TensorRT and ensure trtexec is accessible.\n")
        sys.exit(1)
    return trtexec_path


def load_model(pt_path: str):
    """Load a PyTorch model from a .pt checkpoint or scripted module."""
    try:
        # Try loading as Ultralytics YOLO model first
        from ultralytics import YOLO
        yolo_model = YOLO(pt_path)
        model = yolo_model.model.eval()
        print("Loaded model using Ultralytics YOLO API")
        return model
    except ImportError:
        print("Ultralytics not available, trying direct torch.load...")
    except Exception as e:
        print(f"Ultralytics load failed: {e}, trying direct torch.load...")

    # Fallback to direct torch.load
    try:
        model = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        # Try with weights_only=False for older PyTorch checkpoints
        try:
            model = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as e2:
            sys.stderr.write(f"ERROR: Failed to load model: {e2}\n")
            sys.exit(1)

    # Handle checkpoints that store the model under a key, e.g., YOLO format
    if isinstance(model, dict):
        if "model" in model:
            model = model["model"]
        elif "state_dict" in model:
            # User must reconstruct the original architecture to load state_dict; skipping.
            sys.stderr.write("ERROR: Provided .pt appears to be a state_dict only. Unable to reconstruct model.\n")
            sys.exit(1)

    if not isinstance(model, torch.nn.Module):
        sys.stderr.write("ERROR: Unable to load a Torch nn.Module from the provided .pt file.\n")
        sys.exit(1)

    return model.eval()


def export_to_onnx(model: torch.nn.Module, input_shape: tuple) -> str:
    """Export the model to a temporary ONNX file and return its path."""
    dummy_input = torch.randn(*input_shape)
    tmp_onnx = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp_onnx.close()  # Close so torch.onnx can write to it on Windows/Linux

    torch.onnx.export(
        model,
        (dummy_input,),
        tmp_onnx.name,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    return tmp_onnx.name


def build_engine(onnx_path: str, engine_path: str, fp16: bool):
    """Convert ONNX to TensorRT engine using PyTorch TensorRT backend."""
    try:
        import tensorrt as trt
        import onnx
        from onnx import shape_inference
    except ImportError as e:
        sys.stderr.write(f"ERROR: Required TensorRT/ONNX modules not found: {e}\n")
        sys.stderr.write("Please install: pip install tensorrt onnx\n")
        sys.exit(1)

    # Load and optimize ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx_model = shape_inference.infer_shapes(onnx_model)
    
    # Create TensorRT logger and builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # Set precision
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 precision enabled")
    else:
        print("FP16 not available, using FP32")
    
    # Set workspace size (4GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)
    
    # Parse ONNX
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    if not parser.parse(onnx_model.SerializeToString()):
        for error in range(parser.num_errors):
            print(f"ONNX parse error: {parser.get_error(error)}")
        sys.exit(1)
    
    # Configure dynamic batch dimensions for multi-stream support
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape
    print(f"Input tensor shape: {input_shape}")
    
    # Set up dynamic batch dimensions for multi-stream support
    # Batch dimension (first dimension) will be dynamic: -1 to 4
    # This allows processing 1-4 streams simultaneously
    profile = builder.create_optimization_profile()
    
    # Define dynamic batch range: min=1, optimal=2, max=4
    # This supports 1-4 simultaneous streams
    profile.set_shape("input", 
                     min=(1, 3, 640, 640),    # Minimum batch size 1
                     opt=(2, 3, 640, 640),    # Optimal batch size 2  
                     max=(4, 3, 640, 640))    # Maximum batch size 4
    
    config.add_optimization_profile(profile)
    print("Configured dynamic batch dimensions: batch size 1-4 for multi-stream support")
    
    # Build engine
    print("Building TensorRT engine...")
    engine = builder.build_serialized_network(network, config)
    
    if engine is None:
        sys.stderr.write("ERROR: Failed to build TensorRT engine\n")
        sys.exit(1)
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine)
    
    print(f"Engine saved to {engine_path}")


def main():
    args = parse_args()

    # Resolve absolute paths
    input_pt = os.path.abspath(args.input_pt)
    output_engine = os.path.abspath(args.output_engine)

    # Ensure output directory exists
    Path(output_engine).parent.mkdir(parents=True, exist_ok=True)

    # Load model & export to ONNX
    print("Loading model…")
    model = load_model(input_pt)
    print("Exporting to ONNX…")
    onnx_path = export_to_onnx(model, tuple(args.input_shape))

    try:
        print("Building TensorRT engine…")
        build_engine(onnx_path, output_engine, args.fp16)
        print(f"Success! Engine saved to {output_engine}")
    finally:
        # Clean up temp ONNX
        if os.path.exists(onnx_path):
            os.remove(onnx_path)


if __name__ == "__main__":
    main() 