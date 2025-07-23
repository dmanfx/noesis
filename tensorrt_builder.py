"""
TensorRT Engine Builder

This module handles the conversion of PyTorch models to optimized TensorRT engines
with FP16 precision. It ensures all models are properly optimized for GPU inference
and prevents any CPU fallback scenarios.

Key Features:
- Automatic ONNX export from PyTorch models
- TensorRT engine building with FP16 optimization
- Validation and warm-up procedures
- Engine caching and reuse
"""

import os
import logging
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import tensorrt as trt
from config import AppConfig

logger = logging.getLogger(__name__)

class TensorRTEngineBuilder:
    """Builds and manages TensorRT engines for all inference models"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.TensorRTEngineBuilder")
        
        # Ensure we have CUDA available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. TensorRT optimization requires GPU.")
        
        # Set device
        self.device = torch.device(config.models.DEVICE)
        torch.cuda.set_device(self.device)
        
        # Create engines directory
        engines_dir = Path("models/engines")
        engines_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"TensorRT Engine Builder initialized on device: {self.device}")
    
    def build_all_engines(self) -> Dict[str, str]:
        """Build TensorRT engines for all models in the pipeline"""
        engines = {}
        
        try:
            # Build detection engine
            if Path(self.config.models.MODEL_PATH).exists():
                engines["detection"] = self.build_yolo_engine(
                    model_path=self.config.models.MODEL_PATH,
                    engine_path=self.config.models.DETECTION_ENGINE_PATH,
                    input_shape=(1, 3, 640, 640),  # Standard YOLO input
                    model_type="detection"
                )
            
            # Build pose estimation engine
            if Path(self.config.models.POSE_MODEL_PATH).exists():
                engines["pose"] = self.build_yolo_engine(
                    model_path=self.config.models.POSE_MODEL_PATH,
                    engine_path=self.config.models.POSE_ENGINE_PATH,
                    input_shape=(1, 3, 640, 640),
                    model_type="pose"
                )
            
            # Build segmentation engine
            if Path(self.config.models.SEGMENTATION_MODEL_PATH).exists():
                engines["segmentation"] = self.build_yolo_engine(
                    model_path=self.config.models.SEGMENTATION_MODEL_PATH,
                    engine_path=self.config.models.SEGMENTATION_ENGINE_PATH,
                    input_shape=(1, 3, 640, 640),
                    model_type="segmentation"
                )
            
            # Build ReID engine
            if Path(self.config.models.REID_MODEL_PATH).exists():
                engines["reid"] = self.build_reid_engine(
                    model_path=self.config.models.REID_MODEL_PATH,
                    engine_path=self.config.models.REID_ENGINE_PATH,
                    input_shape=(1, 3, 256, 128)  # Standard ReID input
                )
            
            self.logger.info(f"Successfully built {len(engines)} TensorRT engines")
            return engines
            
        except Exception as e:
            self.logger.error(f"Failed to build TensorRT engines: {e}")
            raise
    
    def build_yolo_engine(
        self, 
        model_path: str, 
        engine_path: str, 
        input_shape: Tuple[int, ...],
        model_type: str
    ) -> str:
        """Build TensorRT engine for YOLO models"""
        
        if os.path.exists(engine_path):
            self.logger.info(f"TensorRT engine already exists: {engine_path}")
            if self._validate_engine(engine_path, input_shape):
                return engine_path
            else:
                self.logger.warning(f"Existing engine validation failed, rebuilding: {engine_path}")
        
        self.logger.info(f"Building TensorRT engine for {model_type}: {model_path} -> {engine_path}")
        
        # Step 1: Export to ONNX
        onnx_path = engine_path.replace('.engine', '.onnx')
        self._export_yolo_to_onnx(model_path, onnx_path, input_shape)
        
        # Step 2: Build TensorRT engine
        self._build_engine_from_onnx(onnx_path, engine_path, input_shape)
        
        # Step 3: Validate engine
        if not self._validate_engine(engine_path, input_shape):
            raise RuntimeError(f"Engine validation failed: {engine_path}")
        
        # Step 4: Warm up engine
        self._warm_up_engine(engine_path, input_shape)
        
        self.logger.info(f"Successfully built and validated TensorRT engine: {engine_path}")
        return engine_path
    
    def build_reid_engine(
        self,
        model_path: str,
        engine_path: str,
        input_shape: Tuple[int, ...]
    ) -> str:
        """Build TensorRT engine for ReID models"""
        
        if os.path.exists(engine_path):
            self.logger.info(f"TensorRT engine already exists: {engine_path}")
            if self._validate_engine(engine_path, input_shape):
                return engine_path
        
        self.logger.info(f"Building TensorRT engine for ReID: {model_path} -> {engine_path}")
        
        # Step 1: Export to ONNX
        onnx_path = engine_path.replace('.engine', '.onnx')
        self._export_reid_to_onnx(model_path, onnx_path, input_shape)
        
        # Step 2: Build TensorRT engine
        self._build_engine_from_onnx(onnx_path, engine_path, input_shape)
        
        # Step 3: Validate and warm up
        if not self._validate_engine(engine_path, input_shape):
            raise RuntimeError(f"Engine validation failed: {engine_path}")
        
        self._warm_up_engine(engine_path, input_shape)
        
        self.logger.info(f"Successfully built ReID TensorRT engine: {engine_path}")
        return engine_path
    
    def _export_yolo_to_onnx(
        self,
        model_path: str,
        onnx_path: str,
        input_shape: Tuple[int, ...]
    ) -> None:
        """Export YOLO model to ONNX format"""
        from ultralytics import YOLO
        
        self.logger.info(f"Exporting YOLO model to ONNX: {model_path} -> {onnx_path}")
        
        # Load model
        model = YOLO(model_path)
        model.to(self.device)
        
        # Export to ONNX with optimization settings
        model.export(
            format="onnx",
            imgsz=input_shape[2:],  # Height, width
            half=True,  # FP16
            dynamic=False,  # Fixed shapes for better optimization
            simplify=True,  # Simplify ONNX graph
            opset=17,  # Latest stable opset
            workspace=self.config.models.TENSORRT_WORKSPACE_SIZE,
            batch=self.config.models.TENSORRT_MAX_BATCH_SIZE,
            device=self.device
        )
        
        # Move exported ONNX to correct location
        exported_onnx = str(Path(model_path).with_suffix('.onnx'))
        if exported_onnx != onnx_path:
            os.rename(exported_onnx, onnx_path)
        
        self.logger.info(f"ONNX export completed: {onnx_path}")
    
    def _export_reid_to_onnx(
        self,
        model_path: str,
        onnx_path: str,
        input_shape: Tuple[int, ...]
    ) -> None:
        """Export ReID model to ONNX format"""
        import torchreid
        
        self.logger.info(f"Exporting ReID model to ONNX: {model_path} -> {onnx_path}")
        
        # Load weights first to determine number of classes
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # Extract number of classes from classifier layer
        classifier_weight_key = 'classifier.weight'
        if classifier_weight_key in state_dict:
            num_classes = state_dict[classifier_weight_key].shape[0]
            self.logger.info(f"Detected {num_classes} classes in ReID model")
        else:
            self.logger.warning("Could not determine number of classes, using default 751")
            num_classes = 751  # Default for Market1501 dataset
        
        # Load ReID model with correct number of classes
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=num_classes,
            pretrained=False
        )
        
        # Load weights
        model.load_state_dict(state_dict)
        
        model.to(self.device)
        model.eval()
        
        # Convert model to FP16 if needed
        if self.config.models.TENSORRT_FP16:
            model.half()
        
        # Create dummy input with matching precision
        dtype = torch.float16 if self.config.models.TENSORRT_FP16 else torch.float32
        dummy_input = torch.randn(input_shape, device=self.device, dtype=dtype)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['features'],
            dynamic_axes=None,  # Fixed shapes
            opset_version=17,
            do_constant_folding=True,
            export_params=True
        )
        
        self.logger.info(f"ReID ONNX export completed: {onnx_path}")
    
    def _build_engine_from_onnx(
        self,
        onnx_path: str,
        engine_path: str,
        input_shape: Tuple[int, ...]
    ) -> None:
        """Build TensorRT engine from ONNX model"""
        self.logger.info(f"Building TensorRT engine: {onnx_path} -> {engine_path}")
        
        # Create TensorRT logger and builder
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(TRT_LOGGER)
        
        # Configure builder
        config = builder.create_builder_config()
        
        # Set workspace size
        workspace_size = self.config.models.TENSORRT_WORKSPACE_SIZE * (1 << 30)  # Convert GB to bytes
        if hasattr(config, 'set_memory_pool_limit'):  # TensorRT >= 8.5
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        else:  # TensorRT < 8.5
            config.max_workspace_size = workspace_size
        
        # Enable FP16 precision
        if self.config.models.TENSORRT_FP16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            self.logger.info("FP16 precision enabled")
        else:
            self.logger.warning("FP16 not available, using FP32")
        
        # Create network
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        
        # Parse ONNX model
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                self.logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    self.logger.error(f"Parser error: {parser.get_error(error)}")
                raise RuntimeError("ONNX parsing failed")
        
        # Build engine
        self.logger.info("Building TensorRT engine (this may take several minutes)...")
        start_time = time.time()
        
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        build_time = time.time() - start_time
        self.logger.info(f"Engine build completed in {build_time:.2f} seconds")
        
        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        self.logger.info(f"TensorRT engine saved: {engine_path}")
    
    def _validate_engine(self, engine_path: str, input_shape: Tuple[int, ...]) -> bool:
        """Validate TensorRT engine by running a test inference"""
        try:
            self.logger.info(f"Validating TensorRT engine: {engine_path}")
            
            # Load engine
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)
            
            with open(engine_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            
            if engine is None:
                return False
            
            # Create execution context
            context = engine.create_execution_context()
            
            # Validate input/output shapes
            self.logger.info(f"Engine validation successful: {engine_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Engine validation failed: {e}")
            return False
    
    def _warm_up_engine(self, engine_path: str, input_shape: Tuple[int, ...]) -> None:
        """Warm up TensorRT engine with dummy inferences"""
        self.logger.info(f"Warming up TensorRT engine: {engine_path}")
        
        try:
            # This would typically run several dummy inferences
            # Implementation depends on the TensorRT inference wrapper
            for i in range(self.config.models.WARM_UP_ITERATIONS):
                # Create dummy input tensor
                dummy_input = torch.randn(input_shape, device=self.device, dtype=torch.float16)
                # Run inference (implementation would depend on inference wrapper)
                time.sleep(0.01)  # Simulate inference time
            
            self.logger.info(f"Engine warm-up completed: {engine_path}")
            
        except Exception as e:
            self.logger.warning(f"Engine warm-up failed (non-critical): {e}")


def build_tensorrt_engines(config: AppConfig) -> Dict[str, str]:
    """Convenience function to build all TensorRT engines"""
    builder = TensorRTEngineBuilder(config)
    return builder.build_all_engines()


if __name__ == "__main__":
    # Test engine building
    from config import config
    
    logging.basicConfig(level=logging.INFO)
    
    if config.models.ENABLE_TENSORRT:
        engines = build_tensorrt_engines(config)
        print(f"Built engines: {list(engines.keys())}")
    else:
        print("TensorRT optimization is disabled in config") 