#!/usr/bin/env python3
"""
Generate YOLO11-seg ONNX and TensorRT engine from a .pt weights file.

This script exports a YOLO11-seg model to ONNX and then builds a TensorRT FP16 engine
with the same parameters we've been using for DeepStream integration.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# ============================================================================
# CONFIGURATION - Edit these variables as needed
# ============================================================================

# Path to the PyTorch weights file (.pt)
SOURCE_WEIGHTS = "models/yolo11m-seg.pt"

# Output paths
OUTPUT_ONNX = "models/yolo11m-seg_cust.onnx"
OUTPUT_ENGINE = "models/yolo11m-seg_cust.engine"

# Export script path (relative to project root)
EXPORT_SCRIPT = "utils/export_yolo11_seg.py"

# Python venv path for export (with torch 2.2.2, onnx 1.15.0, etc.)
VENV_PATH = os.path.expanduser("~/venvs/torch_export")

# Export parameters
IMG_SIZE = [640, 640]
MAX_DETECTIONS = 30
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
OPSET_VERSION = 18
USE_DYNAMIC_BATCH = True
SIMPLIFY_ONNX = True

# TensorRT engine build parameters
FP16_PRECISION = True
MIN_BATCH_SIZE = 1
OPT_BATCH_SIZE = 3
MAX_BATCH_SIZE = 3

# ============================================================================
# FUNCTIONS
# ============================================================================

def check_file_exists(filepath, description):
    """Check if a file exists and raise an error if not."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"{description} not found: {filepath}")
    return True

def check_venv(venv_path):
    """Check if venv exists and has required packages."""
    venv_python = os.path.join(venv_path, "bin", "python")
    if not os.path.isfile(venv_python):
        raise FileNotFoundError(f"Venv Python not found: {venv_python}")
    return venv_python

def run_export(venv_python, export_script, weights, onnx_output, project_root):
    """Run the ONNX export using the venv Python."""
    print(f"\n{'='*70}")
    print(f"STEP 1: Exporting {weights} to ONNX")
    print(f"{'='*70}\n")
    
    cmd = [
        venv_python,
        export_script,
        "--weights", weights,
        "--img", str(IMG_SIZE[0]), str(IMG_SIZE[1]),
        "--max-detections", str(MAX_DETECTIONS),
        "--conf-threshold", str(CONF_THRESHOLD),
        "--iou-threshold", str(IOU_THRESHOLD),
        "--opset", str(OPSET_VERSION),
        "--out", onnx_output,
    ]
    
    if USE_DYNAMIC_BATCH:
        cmd.append("--dynamic")
    
    if SIMPLIFY_ONNX:
        cmd.append("--simplify")
    
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=project_root, check=True)
    
    if not os.path.isfile(os.path.join(project_root, onnx_output)):
        raise RuntimeError(f"ONNX export failed - output file not created: {onnx_output}")
    
    print(f"\n✓ ONNX export successful: {onnx_output}\n")
    return True

def run_trtexec(onnx_input, engine_output, project_root):
    """Build TensorRT engine using trtexec."""
    print(f"\n{'='*70}")
    print(f"STEP 2: Building TensorRT engine")
    print(f"{'='*70}\n")
    
    onnx_path = os.path.join(project_root, onnx_input)
    engine_path = os.path.join(project_root, engine_output)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    
    cmd = [
        "trtexec",
        "--onnx", onnx_path,
        "--saveEngine", engine_path,
    ]
    
    if FP16_PRECISION:
        cmd.append("--fp16")
    
    if USE_DYNAMIC_BATCH:
        cmd.extend([
            "--minShapes", f"images:{MIN_BATCH_SIZE}x3x{IMG_SIZE[0]}x{IMG_SIZE[1]}",
            "--optShapes", f"images:{OPT_BATCH_SIZE}x3x{IMG_SIZE[0]}x{IMG_SIZE[1]}",
            "--maxShapes", f"images:{MAX_BATCH_SIZE}x3x{IMG_SIZE[0]}x{IMG_SIZE[1]}",
        ])
    
    cmd.append("--verbose")
    
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=project_root, check=True)
    
    if not os.path.isfile(engine_path):
        raise RuntimeError(f"TensorRT engine build failed - output file not created: {engine_output}")
    
    # Get file size
    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"\n✓ TensorRT engine build successful: {engine_output} ({size_mb:.2f} MB)\n")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate YOLO11-seg ONNX and TensorRT engine from .pt weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths from script variables
  python utils/generate_seg_model.py
  
  # Override source and outputs
  python utils/generate_seg_model.py --weights models/yolo11s-seg.pt \\
                                      --onnx models/yolo11s-seg_cust.onnx \\
                                      --engine models/engines/yolo11s-seg_cust.engine
        """
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        default=SOURCE_WEIGHTS,
        help=f"Path to source .pt weights file (default: {SOURCE_WEIGHTS})"
    )
    
    parser.add_argument(
        "--onnx",
        type=str,
        default=OUTPUT_ONNX,
        help=f"Path to output ONNX file (default: {OUTPUT_ONNX})"
    )
    
    parser.add_argument(
        "--engine",
        type=str,
        default=OUTPUT_ENGINE,
        help=f"Path to output TensorRT engine file (default: {OUTPUT_ENGINE})"
    )
    
    parser.add_argument(
        "--venv",
        type=str,
        default=VENV_PATH,
        help=f"Path to Python venv for export (default: {VENV_PATH})"
    )
    
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX export (only build engine from existing ONNX)"
    )
    
    parser.add_argument(
        "--skip-engine",
        action="store_true",
        help="Skip engine build (only export ONNX)"
    )
    
    args = parser.parse_args()
    
    # Get project root (assume script is in utils/, so go up one level)
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    
    # Resolve paths relative to project root
    weights_path = os.path.join(project_root, args.weights)
    onnx_path = args.onnx if os.path.isabs(args.onnx) else os.path.join(project_root, args.onnx)
    engine_path = args.engine if os.path.isabs(args.engine) else os.path.join(project_root, args.engine)
    export_script_path = os.path.join(project_root, EXPORT_SCRIPT)
    
    print(f"\n{'='*70}")
    print(f"YOLO11-Seg Model Generation")
    print(f"{'='*70}")
    print(f"Project root: {project_root}")
    print(f"Source weights: {weights_path}")
    print(f"Output ONNX: {onnx_path}")
    print(f"Output engine: {engine_path}")
    print(f"{'='*70}\n")
    
    # Validate inputs
    if not args.skip_export:
        check_file_exists(weights_path, "Source weights file")
        check_file_exists(export_script_path, "Export script")
        venv_python = check_venv(args.venv)
    
    # Step 1: Export to ONNX
    if not args.skip_export:
        run_export(
            venv_python,
            export_script_path,
            weights_path,
            onnx_path,
            project_root
        )
    else:
        print("Skipping ONNX export (--skip-export)\n")
    
    # Step 2: Build TensorRT engine
    if not args.skip_engine:
        check_file_exists(onnx_path, "ONNX file")
        run_trtexec(onnx_path, engine_path, project_root)
    else:
        print("Skipping engine build (--skip-engine)\n")
    
    print(f"\n{'='*70}")
    print(f"✓ All steps completed successfully!")
    print(f"{'='*70}\n")
    print(f"ONNX: {onnx_path}")
    print(f"Engine: {engine_path}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        sys.exit(1)



