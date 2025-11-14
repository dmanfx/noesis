#!/usr/bin/env python3
"""
Download YOLO11s-seg model and export to ONNX with batch size 3.

This script:
1. Downloads yolo11s-seg.pt using ultralytics
2. Saves it to models/ folder
3. Exports it to ONNX format with batch size 3
4. Saves the ONNX model to models/ folder
"""
import os
import sys
from pathlib import Path

# Add the DeepStream-Yolo-Seg utils to path for the export function
sys.path.insert(0, str(Path(__file__).parent.parent / "DeepStream-Yolo-Seg" / "utils"))

from ultralytics import YOLO


def download_yolo11s_seg(models_dir: Path) -> Path:
    """Download YOLO11s-seg model and save to models directory."""
    print("Downloading YOLO11s-seg model...")
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "yolo11s-seg.pt"
    
    # Check if model already exists
    if model_path.exists():
        print(f"✅ Model already exists at: {model_path}")
        return model_path
    
    # Download the model using ultralytics
    # This will automatically download from the official repository
    print("Loading YOLO11s-seg model (this will download if not cached)...")
    model = YOLO("yolo11s-seg.pt")
    
    # Try to find the cached model file
    import shutil
    from ultralytics.utils import SETTINGS
    
    # Check ultralytics cache directory
    cache_dir = Path(SETTINGS.get('weights_dir', Path.home() / '.ultralytics' / 'weights'))
    cached_model = cache_dir / "yolo11s-seg.pt"
    
    if cached_model.exists():
        shutil.copy2(cached_model, model_path)
        print(f"✅ Model downloaded and saved to: {model_path}")
    else:
        # If not in expected cache, try to get from model.ckpt_path
        if hasattr(model, 'ckpt_path') and model.ckpt_path:
            ckpt_path = Path(model.ckpt_path)
            if ckpt_path.exists():
                shutil.copy2(ckpt_path, model_path)
                print(f"✅ Model downloaded and saved to: {model_path}")
            else:
                # Fallback: save the full model
                import torch
                torch.save(model.model, model_path)
                print(f"✅ Model saved to: {model_path}")
        else:
            # Last resort: save the full model
            import torch
            torch.save(model.model, model_path)
            print(f"✅ Model saved to: {model_path}")
    
    return model_path


def export_to_onnx(pt_path: Path, models_dir: Path, batch_size: int = 3):
    """Export YOLO11s-seg model to ONNX with specified batch size."""
    print(f"\nExporting model to ONNX with batch size {batch_size}...")
    
    onnx_output = models_dir / "yolo11s-seg.onnx"
    
    # First, try using ultralytics built-in export as a fallback
    # This gives us a basic ONNX model
    try:
        print("Attempting export using DeepStream custom export script...")
        # Import the export function from the DeepStream-Yolo-Seg utils
        from export_yolo11_seg import main as export_main
        
        # Create a mock args object with the required parameters
        class Args:
            def __init__(self):
                self.weights = str(pt_path)
                self.size = [640]  # Default size
                self.opset = 18
                self.simplify = False
                self.dynamic = False  # Static batch size
                self.batch = batch_size
                self.conf_threshold = 0.25
                self.iou_threshold = 0.45
                self.max_detections = 100
        
        args = Args()
        export_main(args)
        
        # The export script saves to the same directory as the .pt file with .onnx extension
        expected_onnx = pt_path.with_suffix('.onnx')
        if expected_onnx.exists():
            if expected_onnx != onnx_output:
                import shutil
                shutil.move(str(expected_onnx), str(onnx_output))
            print(f"✅ ONNX model saved to: {onnx_output}")
            return
            
    except Exception as e1:
        print(f"⚠️  DeepStream export failed: {e1}")
        print("Trying ultralytics built-in export as fallback...")
        
        # Fallback to ultralytics built-in export
        try:
            model = YOLO(str(pt_path))
            # Export with batch size
            model.export(
                format='onnx',
                imgsz=640,
                batch=batch_size,
                dynamic=False,  # Static batch size
                simplify=False,
                opset=18
            )
            
            # The export saves to the same directory as the .pt file
            expected_onnx = pt_path.with_suffix('.onnx')
            if expected_onnx.exists():
                if expected_onnx != onnx_output:
                    import shutil
                    shutil.move(str(expected_onnx), str(onnx_output))
                print(f"✅ ONNX model exported (basic) to: {onnx_output}")
                print("⚠️  Note: This is a basic ONNX export. For DeepStream, you may need to use the custom export script.")
            else:
                raise Exception("ONNX file not found after export")
                
        except Exception as e2:
            print(f"❌ Both export methods failed:", file=sys.stderr)
            print(f"   DeepStream export: {e1}", file=sys.stderr)
            print(f"   Ultralytics export: {e2}", file=sys.stderr)
            raise Exception(f"Failed to export model: {e2}")


def main():
    """Main function to download and export YOLO11s-seg model."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    print("=" * 60)
    print("YOLO11s-seg Download and Export")
    print("=" * 60)
    
    try:
        # Step 1: Download the model
        pt_path = download_yolo11s_seg(models_dir)
        
        # Step 2: Export to ONNX with batch size 3
        export_to_onnx(pt_path, models_dir, batch_size=3)
        
        print("\n" + "=" * 60)
        print("✅ Success! Model files:")
        print(f"   - PyTorch: {pt_path}")
        print(f"   - ONNX:    {models_dir / 'yolo11s-seg.onnx'}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

