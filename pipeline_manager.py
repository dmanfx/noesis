#!/usr/bin/env python3
"""
Pipeline Selection Manager

Ensures only one video processing pipeline is active at a time.
Prevents resource conflicts between different pipeline implementations.
"""

import logging
import threading
from typing import Optional, Dict, Any, Union
from enum import Enum

logger = logging.getLogger(__name__)

class PipelineType(Enum):
    UNIFIED_GPU = "unified_gpu"
    NVDEC_LEGACY = "nvdec_legacy"  
    STANDARD_LEGACY = "standard_legacy"

class PipelineManager:
    """Singleton manager for pipeline selection and resource coordination"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, 'initialized'):
            return
            
        self.initialized = True
        self.active_pipeline = None
        self.active_processors = {}
        self.pipeline_lock = threading.RLock()
        self.logger = logging.getLogger("PipelineManager")
        
        self.logger.info("Pipeline Manager initialized - enforcing single pipeline operation")
    
    def select_pipeline(self, config) -> PipelineType:
        """
        Select appropriate pipeline based on configuration.
        Enforces mutual exclusion between pipeline types.
        """
        with self.pipeline_lock:
            # Force unified GPU pipeline if enabled
            if getattr(config.processing, 'USE_UNIFIED_GPU_PIPELINE', True):
                if not self._validate_unified_gpu_requirements(config):
                    raise RuntimeError("Unified GPU pipeline requirements not met")
                
                self.active_pipeline = PipelineType.UNIFIED_GPU
                self.logger.info("✅ Selected Unified GPU Pipeline (forced single pipeline)")
                return PipelineType.UNIFIED_GPU
            
            # Legacy pipeline selection (deprecated)
            self.logger.warning("⚠️ Legacy pipeline mode - should migrate to Unified GPU")
            self.active_pipeline = PipelineType.NVDEC_LEGACY
            return PipelineType.NVDEC_LEGACY
    
    def register_processor(self, camera_id: str, processor) -> bool:
        """Register a processor instance to prevent conflicts"""
        with self.pipeline_lock:
            if camera_id in self.active_processors:
                self.logger.error(f"❌ Processor conflict: {camera_id} already has active processor")
                return False
            
            self.active_processors[camera_id] = processor
            self.logger.info(f"✅ Registered processor for {camera_id}")
            return True
    
    def unregister_processor(self, camera_id: str):
        """Unregister a processor instance"""
        with self.pipeline_lock:
            if camera_id in self.active_processors:
                del self.active_processors[camera_id]
                self.logger.info(f"✅ Unregistered processor for {camera_id}")
    
    def get_active_processors(self) -> Dict[str, Any]:
        """Get list of active processors"""
        with self.pipeline_lock:
            return self.active_processors.copy()
    
    def _validate_unified_gpu_requirements(self, config) -> bool:
        """Validate requirements for unified GPU pipeline"""
        requirements = []
        
        # Check CUDA availability
        try:
            import torch
            if not torch.cuda.is_available():
                requirements.append("CUDA not available")
        except ImportError:
            requirements.append("PyTorch not available")
        
        # Check TensorRT
        if not getattr(config.models, 'ENABLE_TENSORRT', False):
            requirements.append("TensorRT not enabled")
        
        # Check GPU-only mode
        if not getattr(config.models, 'FORCE_GPU_ONLY', False):
            requirements.append("GPU-only mode not enabled")
        
        # Check GPU preprocessing
        if not getattr(config.processing, 'ENABLE_GPU_PREPROCESSING', False):
            requirements.append("GPU preprocessing not enabled")
        
        if requirements:
            self.logger.error(f"❌ Unified GPU pipeline requirements failed: {requirements}")
            return False
        
        self.logger.info("✅ Unified GPU pipeline requirements validated")
        return True
    
    def force_cleanup(self):
        """Force cleanup of all active processors"""
        with self.pipeline_lock:
            self.logger.warning("🧹 Force cleanup initiated")
            
            processors_to_cleanup = list(self.active_processors.items())
            
            for camera_id, processor in processors_to_cleanup:
                try:
                    if hasattr(processor, 'stop'):
                        processor.stop()
                    self.logger.info(f"✅ Cleaned up processor for {camera_id}")
                except Exception as e:
                    self.logger.error(f"❌ Error cleaning up {camera_id}: {e}")
            
            self.active_processors.clear()
            self.active_pipeline = None
            self.logger.info("🧹 Force cleanup completed")

# Global instance
_pipeline_manager = PipelineManager()

def get_pipeline_manager() -> PipelineManager:
    """Get global pipeline manager instance"""
    return _pipeline_manager 