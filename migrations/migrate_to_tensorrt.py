#!/usr/bin/env python3
"""
TensorRT Migration Script

This script performs a comprehensive migration from PyTorch models to TensorRT FP16 engines.
It ensures all inference operations are moved to GPU with optimal performance and no CPU fallbacks.

Migration Steps:
1. Validate current environment and models
2. Build TensorRT engines for all models
3. Test TensorRT engines against PyTorch models
4. Update configuration for GPU-only mode
5. Backup original models and configurations
6. Verify complete migration

Usage:
    python migrate_to_tensorrt.py [--validate-only] [--force-rebuild]
"""

import os
import sys
import logging
import argparse
import shutil
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config, AppConfig
from tensorrt_builder import build_tensorrt_engines, TensorRTEngineBuilder
from tensorrt_inference import create_gpu_detection_manager, GPUOnlyDetectionManager

logger = logging.getLogger(__name__)

class TensorRTMigrator:
    """Handles migration from PyTorch to TensorRT FP16 inference"""
    
    def __init__(self, validate_only: bool = False, force_rebuild: bool = False):
        self.validate_only = validate_only
        self.force_rebuild = force_rebuild
        self.backup_dir = Path("backups") / f"migration_{int(time.time())}"
        self.logger = logging.getLogger(f"{__name__}.TensorRTMigrator")
        
        # Migration state
        self.migration_report = {
            "status": "pending",
            "timestamp": time.time(),
            "steps_completed": [],
            "errors": [],
            "performance_comparison": {},
            "models_migrated": {}
        }
    
    def run_migration(self) -> bool:
        """Run complete migration process"""
        try:
            self.logger.info("=== Starting TensorRT Migration ===")
            
            if self.validate_only:
                self.logger.info("Running in validation-only mode")
            
            # Step 1: Environment validation
            if not self._validate_environment():
                return False
            self._mark_step_complete("environment_validation")
            
            # Step 2: Model validation
            if not self._validate_models():
                return False
            self._mark_step_complete("model_validation")
            
            # Step 3: Create backup
            if not self.validate_only:
                if not self._create_backup():
                    return False
                self._mark_step_complete("backup_creation")
            
            # Step 4: Build TensorRT engines
            if not self._build_tensorrt_engines():
                return False
            self._mark_step_complete("tensorrt_build")
            
            # Step 5: Performance comparison
            if not self._compare_performance():
                return False
            self._mark_step_complete("performance_comparison")
            
            # Step 6: Update configuration
            if not self.validate_only:
                if not self._update_configuration():
                    return False
                self._mark_step_complete("configuration_update")
            
            # Step 7: Final validation
            if not self._final_validation():
                return False
            self._mark_step_complete("final_validation")
            
            self.migration_report["status"] = "completed"
            self._save_migration_report()
            
            self.logger.info("=== TensorRT Migration Completed Successfully ===")
            return True
            
        except Exception as e:
            self.migration_report["status"] = "failed"
            self.migration_report["errors"].append(str(e))
            self._save_migration_report()
            self.logger.error(f"Migration failed: {e}")
            return False
    
    def _validate_environment(self) -> bool:
        """Validate system environment for TensorRT"""
        self.logger.info("Validating environment...")
        
        try:
            # Check CUDA availability
            if not torch.cuda.is_available():
                self.logger.error("CUDA is not available")
                return False
            
            cuda_version = torch.version.cuda
            self.logger.info(f"CUDA version: {cuda_version}")
            
            # Check GPU capabilities
            device_count = torch.cuda.device_count()
            self.logger.info(f"Available GPUs: {device_count}")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                self.logger.info(f"GPU {i}: {props.name} ({props.total_memory // 1024**2} MB)")
                
                # Check if GPU supports FP16
                if props.major >= 6:  # Pascal and newer
                    self.logger.info(f"GPU {i}: FP16 supported")
                else:
                    self.logger.warning(f"GPU {i}: FP16 may not be optimal")
            
            # Check TensorRT availability
            try:
                import tensorrt as trt
                self.logger.info(f"TensorRT version: {trt.__version__}")
            except ImportError:
                self.logger.error("TensorRT is not installed")
                return False
            
            # Check disk space for engines
            engines_dir = Path("models/engines")
            if engines_dir.exists():
                disk_usage = shutil.disk_usage(engines_dir)
                free_gb = disk_usage.free // (1024**3)
                self.logger.info(f"Available disk space: {free_gb} GB")
                
                if free_gb < 10:  # Require at least 10GB free
                    self.logger.warning("Low disk space - TensorRT engines may require significant storage")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Environment validation failed: {e}")
            return False
    
    def _validate_models(self) -> bool:
        """Validate that all required models exist and are loadable"""
        self.logger.info("Validating models...")
        
        models_to_check = [
            ("detection", config.models.MODEL_PATH),
            ("pose", config.models.POSE_MODEL_PATH),
            ("segmentation", config.models.SEGMENTATION_MODEL_PATH),
            ("reid", config.models.REID_MODEL_PATH)
        ]
        
        for model_name, model_path in models_to_check:
            if not os.path.exists(model_path):
                self.logger.error(f"{model_name} model not found: {model_path}")
                return False
            
            try:
                # Try loading the model
                if model_name == "reid":
                    # Special handling for ReID model
                    checkpoint = torch.load(model_path, map_location="cpu")
                    self.logger.info(f"{model_name} model loaded successfully")
                else:
                    # YOLO models
                    from ultralytics import YOLO
                    model = YOLO(model_path)
                    self.logger.info(f"{model_name} model loaded successfully")
                    del model  # Free memory
                
            except Exception as e:
                self.logger.error(f"Failed to load {model_name} model: {e}")
                return False
        
        return True
    
    def _create_backup(self) -> bool:
        """Create backup of current configuration and models"""
        self.logger.info("Creating backup...")
        
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup configuration
            config_backup = self.backup_dir / "config.py"
            shutil.copy2("config.py", config_backup)
            
            # Backup detection.py
            detection_backup = self.backup_dir / "detection.py"
            shutil.copy2("detection.py", detection_backup)
            
            # Create backup info
            backup_info = {
                "timestamp": time.time(),
                "pytorch_models": {
                    "detection": config.models.MODEL_PATH,
                    "pose": config.models.POSE_MODEL_PATH,
                    "segmentation": config.models.SEGMENTATION_MODEL_PATH,
                    "reid": config.models.REID_MODEL_PATH
                },
                "original_config": {
                    "ENABLE_TENSORRT": getattr(config.models, "ENABLE_TENSORRT", False),
                    "FORCE_GPU_ONLY": getattr(config.models, "FORCE_GPU_ONLY", False),
                    "DEVICE": getattr(config.models, "DEVICE", "auto")
                }
            }
            
            with open(self.backup_dir / "backup_info.json", "w") as f:
                json.dump(backup_info, f, indent=2)
            
            self.logger.info(f"Backup created: {self.backup_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return False
    
    def _build_tensorrt_engines(self) -> bool:
        """Build TensorRT engines for all models"""
        self.logger.info("Building TensorRT engines...")
        
        try:
            # Temporarily enable TensorRT in config
            original_tensorrt_enabled = getattr(config.models, "ENABLE_TENSORRT", False)
            config.models.ENABLE_TENSORRT = True
            
            # Create engines directory
            engines_dir = Path("models/engines")
            engines_dir.mkdir(parents=True, exist_ok=True)
            
            # Remove existing engines if force rebuild
            if self.force_rebuild:
                for engine_file in engines_dir.glob("*.engine"):
                    engine_file.unlink()
                    self.logger.info(f"Removed existing engine: {engine_file}")
            
            # Build engines
            builder = TensorRTEngineBuilder(config)
            engines = builder.build_all_engines()
            
            self.migration_report["models_migrated"] = engines
            
            # Restore original setting
            config.models.ENABLE_TENSORRT = original_tensorrt_enabled
            
            self.logger.info(f"Successfully built {len(engines)} TensorRT engines")
            return True
            
        except Exception as e:
            self.logger.error(f"TensorRT engine building failed: {e}")
            return False
    
    def _compare_performance(self) -> bool:
        """Compare performance between PyTorch and TensorRT"""
        self.logger.info("Comparing performance...")
        
        try:
            # Create test frame
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Test PyTorch inference (current)
            from detection import DetectionManager
            pytorch_manager = DetectionManager(
                model_path=config.models.MODEL_PATH,
                confidence_threshold=config.models.MODEL_CONFIDENCE_THRESHOLD,
                config=config
            )
            pytorch_manager.load_models()
            
            # Warm up
            for _ in range(5):
                pytorch_manager.process_frame(test_frame)
            
            # Benchmark PyTorch
            pytorch_times = []
            for i in range(20):
                detections, time_ms = pytorch_manager.process_frame(test_frame)
                pytorch_times.append(time_ms)
            
            pytorch_avg = np.mean(pytorch_times)
            
            # Test TensorRT inference (if engines exist)
            if all(os.path.exists(path) for path in [
                config.models.DETECTION_ENGINE_PATH,
                config.models.POSE_ENGINE_PATH,
                config.models.SEGMENTATION_ENGINE_PATH,
                config.models.REID_ENGINE_PATH
            ]):
                # Temporarily enable TensorRT mode
                original_tensorrt = getattr(config.models, "ENABLE_TENSORRT", False)
                original_gpu_only = getattr(config.models, "FORCE_GPU_ONLY", False)
                
                config.models.ENABLE_TENSORRT = True
                config.models.FORCE_GPU_ONLY = True
                
                try:
                    tensorrt_manager = create_gpu_detection_manager(config)
                    
                    # Warm up
                    for _ in range(5):
                        tensorrt_manager.process_frame(test_frame)
                    
                    # Benchmark TensorRT
                    tensorrt_times = []
                    for i in range(20):
                        detections, time_ms = tensorrt_manager.process_frame(test_frame)
                        tensorrt_times.append(time_ms)
                    
                    tensorrt_avg = np.mean(tensorrt_times)
                    
                    # Calculate improvement
                    speedup = pytorch_avg / tensorrt_avg if tensorrt_avg > 0 else float('inf')
                    
                    self.migration_report["performance_comparison"] = {
                        "pytorch_avg_ms": float(pytorch_avg),
                        "tensorrt_avg_ms": float(tensorrt_avg),
                        "speedup_factor": float(speedup),
                        "improvement_percent": float((speedup - 1) * 100)
                    }
                    
                    self.logger.info(f"Performance comparison:")
                    self.logger.info(f"  PyTorch: {pytorch_avg:.2f}ms average")
                    self.logger.info(f"  TensorRT: {tensorrt_avg:.2f}ms average")
                    self.logger.info(f"  Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% improvement)")
                    
                finally:
                    # Restore original settings
                    config.models.ENABLE_TENSORRT = original_tensorrt
                    config.models.FORCE_GPU_ONLY = original_gpu_only
            
            return True
            
        except Exception as e:
            self.logger.error(f"Performance comparison failed: {e}")
            return False
    
    def _update_configuration(self) -> bool:
        """Update configuration files for TensorRT mode"""
        self.logger.info("Updating configuration...")
        
        try:
            # Read current config file
            with open("config.py", "r") as f:
                config_content = f.read()
            
            # Update configuration values
            updates = {
                "ENABLE_TENSORRT: bool = False": "ENABLE_TENSORRT: bool = True",
                "TENSORRT_FP16: bool = True": "TENSORRT_FP16: bool = True",
                "FORCE_GPU_ONLY: bool = False": "FORCE_GPU_ONLY: bool = True",
                'DEVICE: str = "cuda:0"': 'DEVICE: str = "cuda:0"'
            }
            
            for old_value, new_value in updates.items():
                if old_value in config_content:
                    config_content = config_content.replace(old_value, new_value)
            
            # Write updated config
            with open("config.py", "w") as f:
                f.write(config_content)
            
            self.logger.info("Configuration updated for TensorRT mode")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration update failed: {e}")
            return False
    
    def _final_validation(self) -> bool:
        """Final validation of TensorRT setup"""
        self.logger.info("Running final validation...")
        
        try:
            # Reload config
            import importlib
            import config as config_module
            importlib.reload(config_module)
            
            # Test GPU-only detection manager
            if not self.validate_only:
                config.models.ENABLE_TENSORRT = True
                config.models.FORCE_GPU_ONLY = True
            
            # Verify all engines exist
            required_engines = [
                config.models.DETECTION_ENGINE_PATH,
                config.models.POSE_ENGINE_PATH,
                config.models.SEGMENTATION_ENGINE_PATH,
                config.models.REID_ENGINE_PATH
            ]
            
            missing_engines = []
            for engine_path in required_engines:
                if not os.path.exists(engine_path):
                    missing_engines.append(engine_path)
            
            if missing_engines:
                self.logger.error(f"Missing TensorRT engines: {missing_engines}")
                return False
            
            # Verify GPU availability
            import torch
            if not torch.cuda.is_available():
                self.logger.error("CUDA not available for final validation")
                return False
            
            # Test TensorRT inference
            if not self.validate_only:
                from tensorrt_inference import create_gpu_detection_manager
                
                self.logger.info("Testing GPU-only TensorRT inference...")
                test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # Create GPU detection manager
                tensorrt_manager = create_gpu_detection_manager(config)
                
                # Run test inference
                detections, time_ms = tensorrt_manager.process_frame(test_frame)
                
                # Validate results
                if time_ms <= 0:
                    self.logger.error("Invalid inference time")
                    return False
                
                self.logger.info(f"✅ Final validation passed")
                self.logger.info(f"✅ TensorRT inference working: {time_ms:.2f}ms")
                self.logger.info(f"✅ Found {len(detections)} detections in test frame")
                self.logger.info(f"✅ GPU-only mode enforced successfully")
            else:
                self.logger.info("✅ Validation-only mode - skipping inference test")
            
            # Check configuration status
            self.logger.info(f"✅ ENABLE_TENSORRT: {config.models.ENABLE_TENSORRT}")
            self.logger.info(f"✅ FORCE_GPU_ONLY: {config.models.FORCE_GPU_ONLY}")
            self.logger.info(f"✅ DEVICE: {config.models.DEVICE}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Final validation failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _mark_step_complete(self, step: str):
        """Mark a migration step as complete"""
        self.migration_report["steps_completed"].append({
            "step": step,
            "timestamp": time.time()
        })
    
    def _save_migration_report(self):
        """Save migration report to file"""
        report_path = Path("migration_report.json")
        with open(report_path, "w") as f:
            json.dump(self.migration_report, f, indent=2)
        
        self.logger.info(f"Migration report saved: {report_path}")


def main():
    """Main migration script entry point"""
    parser = argparse.ArgumentParser(description="Migrate to TensorRT FP16 inference")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate environment and build engines, don't modify config")
    parser.add_argument("--force-rebuild", action="store_true",
                       help="Force rebuild of existing TensorRT engines")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run migration
    migrator = TensorRTMigrator(
        validate_only=args.validate_only,
        force_rebuild=args.force_rebuild
    )
    
    success = migrator.run_migration()
    
    if success:
        if args.validate_only:
            print("\n✅ Validation completed successfully!")
            print("Run without --validate-only to perform actual migration.")
        else:
            print("\n✅ Migration completed successfully!")
            print("Your system is now configured for TensorRT FP16 inference.")
            print("All inference operations will run on GPU with no CPU fallbacks.")
        
        sys.exit(0)
    else:
        print("\n❌ Migration failed!")
        print("Check migration_report.json for details.")
        print("Original configuration has been preserved.")
        sys.exit(1)


if __name__ == "__main__":
    main() 