#!/usr/bin/env python3
"""
Convert YOLO11 models to TensorRT format for DeepStream integration
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Optional

def convert_yolo11_to_tensorrt(model_path, output_path, precision='fp16', input_size=(640, 640)):
    """
    Convert YOLO11 PyTorch model to TensorRT engine
    
    Args:
        model_path: Path to YOLO11 .pt file
        output_path: Path to save TensorRT engine
        precision: 'fp16' or 'fp32'
        input_size: (width, height) for input tensor
    """
    
    print(f"Converting {model_path} to TensorRT...")
    
    # Load YOLO11 model with weights_only=False for PyTorch 2.6+ compatibility
    yolo_obj = None  # Keep reference to ultralytics loader if used
    try:
        model = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Failed to load model with weights_only=False: {e}")
        # Try with ultralytics YOLO loader
        try:
            from ultralytics import YOLO
            yolo_obj = YOLO(model_path)
            model = yolo_obj.model  # Get the underlying PyTorch model
        except Exception as e2:
            print(f"Failed to load model with ultralytics: {e2}")
            raise RuntimeError(f"Could not load YOLO11 model: {e}")

    # Ensure we now have a torch.nn.Module instance
    if not isinstance(model, torch.nn.Module):
        # Some loaders might return wrapper objects with a `model` attribute
        sub_model = getattr(model, "model", None)
        if isinstance(sub_model, torch.nn.Module):
            model = sub_model  # type: ignore[assignment]
        else:
            # As a last resort, if we have a `yolo_obj`, use its export method
            if yolo_obj is not None:
                print("Using Ultralytics export() fallback to create ONNX model ...")
                onnx_path = output_path.replace('.engine', '.onnx')
                try:
                    yolo_obj.export(format='onnx', imgsz=list(input_size), dynamic=True, half=(precision=='fp16'), opset=11, simplify=True, device='cpu')
                    # Find the exported ONNX file (Ultralytics places it under runs/export/.../model.onnx)
                    # If a specific output path is requested, move/rename the file
                    default_onnx = Path("runs/export").glob("**/*.onnx")
                    exported_onnx = next(default_onnx, None)
                    if exported_onnx is None:
                        raise RuntimeError("Ultralytics export did not produce an ONNX file.")
                    os.replace(exported_onnx, onnx_path)
                    print(f"ONNX model exported to {onnx_path}")
                    # Skip the torch.onnx export path
                    pt_model = None
                except Exception as ex:
                    raise RuntimeError(f"Ultralytics export() failed: {ex}")
            else:
                raise RuntimeError("Loaded YOLO11 model is not a torch.nn.Module")
    pt_model: Optional[torch.nn.Module] = model if isinstance(model, torch.nn.Module) else None

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[1], input_size[0])

    # Export to ONNX if we have a valid torch.nn.Module
    onnx_path = output_path.replace('.engine', '.onnx')

    if pt_model is not None:
        print(f"Exporting to ONNX via torch.onnx.export: {onnx_path}")
        torch.onnx.export(
            pt_model,
            (dummy_input,),  # args must be a tuple for torch.onnx.export
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    
    # Convert ONNX to TensorRT
    print(f"Converting ONNX to TensorRT: {output_path}")
    
    # Use trtexec to convert ONNX to TensorRT
    cmd = f"trtexec --onnx={onnx_path} --saveEngine={output_path}"
    
    if precision == 'fp16':
        cmd += " --fp16"
    
    cmd += f" --minShapes=input:1x3x{input_size[1]}x{input_size[0]}"
    cmd += f" --optShapes=input:1x3x{input_size[1]}x{input_size[0]}"
    cmd += f" --maxShapes=input:1x3x{input_size[1]}x{input_size[0]}"
    
    print(f"Running: {cmd}")
    os.system(cmd)
    
    print(f"Conversion complete: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO11 models to TensorRT')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO11 .pt file')
    parser.add_argument('--output', type=str, help='Output path for TensorRT engine')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp16', 'fp32'], help='Precision')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640], help='Input size (width height)')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        model_name = Path(args.model).stem
        args.output = f"models/engines/{model_name}_{args.precision}.engine"
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Convert model
    convert_yolo11_to_tensorrt(
        args.model,
        args.output,
        args.precision,
        tuple(args.input_size)
    )

if __name__ == "__main__":
    main() 