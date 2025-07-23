"""
TensorRT Inference Manager

This module provides GPU-only inference using optimized TensorRT engines.
It completely eliminates CPU fallbacks and ensures all operations run on GPU
with FP16 precision for maximum performance.

Key Features:
- GPU-only inference with no CPU fallbacks
- FP16 precision throughout the pipeline
- Efficient memory management
- Batch processing support
- Comprehensive error handling with strict GPU enforcement
"""

import os
os.environ['no_proxy'] = '*'
import time
import logging
import threading
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
from cuda_context_manager import CUDAContextScope, initialize_cuda
from typing import Dict, List, Tuple, Optional
from config import AppConfig
from gpu_memory_pool import get_global_memory_pool
from detection import Detection

logger = logging.getLogger(__name__)

# Initialize CUDA context manager once
_cuda_manager = None

# Global shutdown flag to suppress TensorRT error logging during shutdown
_shutting_down = False
_shutdown_lock = threading.Lock()

def set_tensorrt_shutdown_mode(shutting_down: bool = True):
    """Set shutdown mode to suppress TensorRT error logging with proper synchronization"""
    global _shutting_down
    with _shutdown_lock:
        _shutting_down = shutting_down
        if shutting_down:
            logger.info("TensorRT shutdown mode activated")

def is_shutting_down() -> bool:
    """Thread-safe check for shutdown state"""
    global _shutting_down
    with _shutdown_lock:
        return _shutting_down

def _ensure_cuda_initialized():
    """Ensure CUDA context manager is initialized"""
    global _cuda_manager
    if _cuda_manager is None:
        device_str = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        device_id = int(device_str.split(',')[0]) if device_str else 0
        _cuda_manager = initialize_cuda(device_id)
    return _cuda_manager

class TensorRTInferenceEngine:
    """TensorRT inference engine wrapper with unified memory pool integration"""
    
    def __init__(self, engine_path: str, device: str = "cuda:0"):
        """
        Initialize TensorRT inference engine.
        
        Args:
            engine_path: Path to TensorRT engine file
            device: CUDA device to use
        """
        self.engine_path = engine_path
        self.device = torch.device(device)
        self.logger = logging.getLogger(f"{__name__}.TensorRTEngine")
        self.model_manager = None  # Will be set by model manager
        
        # Ensure CUDA is initialized
        self.cuda_manager = _ensure_cuda_initialized()
        
        # Use unified GPU memory pool
        self.memory_pool = get_global_memory_pool(device=device)
        
        # --- HOT-FIX: global re-entrancy guard -------------
        self._infer_lock = threading.Lock()
        # ---------------------------------------------------
        
        # Load TensorRT engine
        self._load_engine()
        
        # Pre-allocate buffers using memory pool
        self._allocate_buffers()
        
        self.logger.info(f"TensorRT engine loaded from {engine_path}")
    
    def _load_engine(self):
        """Load TensorRT engine from file"""
        # Use CUDA context scope
        with CUDAContextScope(self.cuda_manager):
            # Create logger for TensorRT
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Load engine
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                raise RuntimeError(f"Failed to load TensorRT engine from {self.engine_path}")
            
            # Create execution context
            self.context = self.engine.create_execution_context()
            
            # Create CUDA stream
            self.stream = cuda.Stream()
    
    def _allocate_buffers(self):
        """Allocate input/output buffers using unified memory pool"""
        # Use CUDA context scope
        with CUDAContextScope(self.cuda_manager):
            self.inputs = []
            self.outputs = []
            self.bindings = []
            
            # Use modern TensorRT API if available
            if hasattr(self.engine, 'num_io_tensors'):
                # TensorRT 10.x API
                for i in range(self.engine.num_io_tensors):
                    tensor_name = self.engine.get_tensor_name(i)
                    tensor_shape = self.engine.get_tensor_shape(tensor_name)
                    tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
                    is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
                    
                    # Convert TensorRT dtype to numpy dtype
                    if tensor_dtype == trt.float32:
                        np_dtype = np.float32
                        torch_dtype = torch.float32
                    elif tensor_dtype == trt.float16:
                        np_dtype = np.float16
                        torch_dtype = torch.float16
                    else:
                        np_dtype = np.float32
                        torch_dtype = torch.float32
                    
                    # Calculate buffer size
                    size = trt.volume(tensor_shape)
                    
                    # Allocate host buffer
                    host_buffer = cuda.pagelocked_empty(size, np_dtype)
                    
                    # Allocate device buffer using memory pool
                    # For dynamic batch dimensions, use max batch size for allocation
                    device_tensor_shape = tuple(abs(dim) if dim != -1 else 4 for dim in tensor_shape)
                    if self.memory_pool:
                        device_tensor, alloc_id = self.memory_pool.get_tensor(device_tensor_shape, torch_dtype)
                        # ENSURE tensor is not returned to pool while in use
                        device_tensor.requires_grad_(False)  # Prevent gradient tracking
                        device_buffer = device_tensor.data_ptr()
                    else:
                        # Fallback allocation
                        device_tensor = torch.zeros(device_tensor_shape, dtype=torch_dtype, device=self.device)
                        alloc_id = None
                        device_buffer = device_tensor.data_ptr()
                    
                    # Store buffer info
                    buffer_info = {
                        'name': tensor_name,
                        'host': host_buffer,
                        'device': device_buffer,
                        'device_tensor': device_tensor,
                        'alloc_id': alloc_id,
                        'size': size * np_dtype().itemsize,  # Store actual byte size
                        'shape': device_tensor_shape,
                        'dtype': torch_dtype
                    }
                    
                    if is_input:
                        self.inputs.append(buffer_info)
                    else:
                        self.outputs.append(buffer_info)
                    
                    self.bindings.append(int(device_buffer))
            else:
                # Legacy API for older TensorRT versions
                for binding in self.engine:
                    binding_idx = self.engine.get_binding_index(binding)
                    shape = self.engine.get_binding_shape(binding_idx)
                    dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                    size = trt.volume(shape)
                    
                    # Convert to torch dtype
                    torch_dtype = torch.float16 if dtype == np.float16 else torch.float32
                    
                    # Allocate device buffer using memory pool
                    # For dynamic batch dimensions, use max batch size for allocation
                    device_tensor_shape = tuple(abs(dim) if dim != -1 else 4 for dim in shape)
                    if self.memory_pool:
                        device_tensor, alloc_id = self.memory_pool.get_tensor(device_tensor_shape, torch_dtype)
                        # ENSURE tensor is not returned to pool while in use
                        device_tensor.requires_grad_(False)  # Prevent gradient tracking
                        device_buffer = device_tensor.data_ptr()
                    else:
                        # Fallback allocation
                        device_tensor = torch.zeros(device_tensor_shape, dtype=torch_dtype, device=self.device)
                        alloc_id = None
                        device_buffer = device_tensor.data_ptr()
                    
                    buffer_info = {
                        'name': binding,
                        'host': host_buffer,
                        'device': device_buffer,
                        'device_tensor': device_tensor,
                        'alloc_id': alloc_id,
                        'size': size * dtype().itemsize,  # Store actual byte size
                        'shape': device_tensor_shape,
                        'dtype': torch_dtype
                    }
                    
                    if self.engine.binding_is_input(binding_idx):
                        self.inputs.append(buffer_info)
                    else:
                        self.outputs.append(buffer_info)
                    
                    self.bindings.append(int(device_buffer))
    
    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run inference on input tensor
        
        Args:
            input_tensor: Input tensor on GPU (FP16)
            
        Returns:
            Output tensor on GPU (FP16)
        """
        with self._infer_lock:
            # Use CUDA context scope for inference
            with CUDAContextScope(self.cuda_manager):
                # Mark tensors as actively used
                if self.memory_pool:
                    for inp in self.inputs:
                        if 'alloc_id' in inp and inp['alloc_id'] is not None:
                            self.memory_pool.touch_tensor(inp['alloc_id'])
                    for out in self.outputs:
                        if 'alloc_id' in out and out['alloc_id'] is not None:
                            self.memory_pool.touch_tensor(out['alloc_id'])
                
                # Ensure input is on correct device and FP16
                if input_tensor.device != self.device:
                    input_tensor = input_tensor.to(self.device)
                
                if input_tensor.dtype != torch.float16:
                    input_tensor = input_tensor.half()
                
                # Validate input tensor properties
                if input_tensor.numel() == 0:
                    raise RuntimeError("Empty tensor cannot be copied")
                if not input_tensor.is_contiguous():
                    input_tensor = input_tensor.contiguous()

                # Validate buffer sizes match
                expected_size = input_tensor.numel() * input_tensor.element_size()
                allocated_size = self.inputs[0]['size'] if 'size' in self.inputs[0] else expected_size
                if expected_size != allocated_size:
                    raise RuntimeError(f"Buffer size mismatch: expected {expected_size}, allocated {allocated_size}")

                # Validate memory addresses are valid
                if self.inputs[0]['device'] == 0:
                    raise RuntimeError("Invalid device buffer address")
                
                # Ensure previous operations complete
                self.stream.synchronize()
                
                # Enhanced error logging
                self.logger.debug(f"TensorRT inference: input_tensor shape={input_tensor.shape}, dtype={input_tensor.dtype}, device={input_tensor.device}")
                self.logger.debug(f"Buffer info: input_buffer_size={self.inputs[0].get('size', 'unknown')}, output_buffer_size={self.outputs[0].get('size', 'unknown')}")
                
                # Direct GPU-to-GPU memory copy with error recovery
                try:
                    cuda.memcpy_dtod_async(
                        self.inputs[0]['device'], 
                        input_tensor.data_ptr(), 
                        expected_size,
                        self.stream
                    )
                except Exception as e:
                    self.logger.error(f"CUDA memory copy failed: {e}")
                    self.logger.error(f"Input tensor: shape={input_tensor.shape}, device={input_tensor.device}, dtype={input_tensor.dtype}")
                    self.logger.error(f"Buffer info: device_ptr={self.inputs[0]['device']}, size={self.inputs[0].get('size', 'unknown')}")
                    raise RuntimeError(f"cuMemcpyDtoDAsync failed: {e}")
                
                # Set tensor addresses for modern API
                if hasattr(self.context, 'set_tensor_address'):
                    # TensorRT 10.x API - set tensor addresses
                    for inp in self.inputs:
                        self.context.set_tensor_address(inp['name'], int(inp['device']))
                    for out in self.outputs:
                        self.context.set_tensor_address(out['name'], int(out['device']))
                
                # Run inference using modern API
                if hasattr(self.context, 'execute_async_v3'):
                    # TensorRT 10.x API
                    self.context.execute_async_v3(stream_handle=self.stream.handle)
                else:
                    # Fallback for older versions
                    self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
                
                # Transfer output data back to host
                cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
                
                # Synchronize stream
                self.stream.synchronize()
                
                # Convert output back to tensor
                output_shape = self.outputs[0]['shape']  # Use pre-stored shape
                output_np = self.outputs[0]['host'].reshape(output_shape)
                output_tensor = torch.from_numpy(output_np).to(self.device, dtype=torch.float16)
                
                return output_tensor
    
    def __del__(self):
        """Cleanup: return memory to pool"""
        self.cleanup()
    
    def cleanup(self):
        """Explicit cleanup method to return memory to pool immediately"""
        # Check if CUDA is still available before cleanup
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            if not is_shutting_down():
                self.logger.warning("CUDA context already destroyed, skipping TensorRT cleanup")
            return
        
        try:
            if hasattr(self, 'memory_pool') and hasattr(self, 'inputs') and hasattr(self, 'outputs'):
                # Return input buffers to pool
                for inp in getattr(self, 'inputs', []):
                    if 'device_tensor' in inp and 'alloc_id' in inp:
                        try:
                            self.memory_pool.return_tensor(inp['device_tensor'], inp['alloc_id'])
                            if not is_shutting_down():
                                self.logger.debug(f"Returned input tensor to pool: alloc_id={inp['alloc_id']}")
                        except Exception as e:
                            if not is_shutting_down():
                                self.logger.warning(f"Failed to return input tensor to pool: {e}")
                
                # Return output buffers to pool
                for out in getattr(self, 'outputs', []):
                    if 'device_tensor' in out and 'alloc_id' in out:
                        try:
                            self.memory_pool.return_tensor(out['device_tensor'], out['alloc_id'])
                            if not is_shutting_down():
                                self.logger.debug(f"Returned output tensor to pool: alloc_id={out['alloc_id']}")
                        except Exception as e:
                            if not is_shutting_down():
                                self.logger.warning(f"Failed to return output tensor to pool: {e}")
                
                # Clear references
                self.inputs = []
                self.outputs = []
                
                # Clean up TensorRT objects
                try:
                    if hasattr(self, 'context') and self.context is not None:
                        del self.context
                        self.context = None
                except Exception as e:
                    if not is_shutting_down():
                        self.logger.warning(f"Error cleaning up TensorRT context: {e}")
                
                try:
                    if hasattr(self, 'engine') and self.engine is not None:
                        del self.engine
                        self.engine = None
                except Exception as e:
                    if not is_shutting_down():
                        self.logger.warning(f"Error cleaning up TensorRT engine: {e}")
                
                try:
                    if hasattr(self, 'stream') and self.stream is not None:
                        self.stream.synchronize()
                        del self.stream
                        self.stream = None
                except Exception as e:
                    if not is_shutting_down():
                        self.logger.warning(f"Error cleaning up CUDA stream: {e}")
                
                if not is_shutting_down():
                    self.logger.info("TensorRT engine memory returned to pool")
                    
        except Exception as e:
            if not is_shutting_down():
                self.logger.error(f"Error during TensorRT engine cleanup: {e}")

    def create_execution_context(self) -> "TensorRTExecutionContext":
        """Create a new execution context for this engine"""
        return TensorRTExecutionContext(self)


def _allocate_ctx_buffers(context, memory_pool, device):
    """
    Helper function to allocate buffers for a TensorRT execution context.
    Extracted from TensorRTInferenceEngine._allocate_buffers() to avoid code duplication.
    """
    inputs = []
    outputs = []
    bindings = []
    
    engine = context.engine
    
    # Use modern TensorRT API if available
    if hasattr(engine, 'num_io_tensors'):
        # TensorRT 10.x API
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            tensor_shape = engine.get_tensor_shape(tensor_name)
            tensor_dtype = engine.get_tensor_dtype(tensor_name)
            is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            
            # Convert TensorRT dtype to numpy dtype
            if tensor_dtype == trt.float32:
                np_dtype = np.float32
                torch_dtype = torch.float32
            elif tensor_dtype == trt.float16:
                np_dtype = np.float16
                torch_dtype = torch.float16
            else:
                np_dtype = np.float32
                torch_dtype = torch.float32
            
            # Calculate buffer size
            size = trt.volume(tensor_shape)
            
            # Allocate host buffer
            host_buffer = cuda.pagelocked_empty(size, np_dtype)
            
            # Allocate device buffer using memory pool
            # For dynamic batch dimensions, use max batch size for allocation
            device_tensor_shape = tuple(abs(dim) if dim != -1 else 4 for dim in tensor_shape)
            if memory_pool:
                device_tensor, alloc_id = memory_pool.get_tensor(device_tensor_shape, torch_dtype)
                device_tensor.requires_grad_(False)
                device_buffer = device_tensor.data_ptr()
            else:
                device_tensor = torch.zeros(device_tensor_shape, dtype=torch_dtype, device=device)
                alloc_id = None
                device_buffer = device_tensor.data_ptr()
            
            # Store buffer info
            buffer_info = {
                'name': tensor_name,
                'host': host_buffer,
                'device': device_buffer,
                'device_tensor': device_tensor,
                'alloc_id': alloc_id,
                'size': size * np_dtype().itemsize,
                'shape': device_tensor_shape,
                'dtype': torch_dtype
            }
            
            if is_input:
                inputs.append(buffer_info)
            else:
                outputs.append(buffer_info)
            
            bindings.append(int(device_buffer))
    else:
        # Legacy API for older TensorRT versions
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            shape = engine.get_binding_shape(binding_idx)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            size = trt.volume(shape)
            
            # Convert to torch dtype
            torch_dtype = torch.float16 if dtype == np.float16 else torch.float32
            
            # Allocate host buffer
            host_buffer = cuda.pagelocked_empty(size, dtype)
            
            # Allocate device buffer using memory pool
            # For dynamic batch dimensions, use max batch size for allocation
            device_tensor_shape = tuple(abs(dim) if dim != -1 else 4 for dim in shape)
            if memory_pool:
                device_tensor, alloc_id = memory_pool.get_tensor(device_tensor_shape, torch_dtype)
                device_tensor.requires_grad_(False)
                device_buffer = device_tensor.data_ptr()
            else:
                device_tensor = torch.zeros(device_tensor_shape, dtype=torch_dtype, device=device)
                alloc_id = None
                device_buffer = device_tensor.data_ptr()
            
            buffer_info = {
                'name': binding,
                'host': host_buffer,
                'device': device_buffer,
                'device_tensor': device_tensor,
                'alloc_id': alloc_id,
                'size': size * np.dtype(dtype).itemsize,
                'shape': device_tensor_shape,
                'dtype': torch_dtype
            }
            
            if engine.binding_is_input(binding_idx):
                inputs.append(buffer_info)
            else:
                outputs.append(buffer_info)
            
            bindings.append(int(device_buffer))
    
    return inputs, outputs, bindings


def _infer_with_ctx(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Helper function to run inference with a TensorRT execution context.
    Extracted from TensorRTInferenceEngine.infer() to avoid code duplication.
    """
    # Use CUDA context scope for inference
    with CUDAContextScope(ctx.base.cuda_manager):
        # Mark tensors as actively used
        if ctx.memory_pool:
            for inp in ctx.inputs:
                if 'alloc_id' in inp and inp['alloc_id'] is not None:
                    ctx.memory_pool.touch_tensor(inp['alloc_id'])
            for out in ctx.outputs:
                if 'alloc_id' in out and out['alloc_id'] is not None:
                    ctx.memory_pool.touch_tensor(out['alloc_id'])
        
        # Ensure input is on correct device and FP16
        if input_tensor.device != ctx.device:
            input_tensor = input_tensor.to(ctx.device)
        
        if input_tensor.dtype != torch.float16:
            input_tensor = input_tensor.half()
        
        # Validate input tensor properties
        if input_tensor.numel() == 0:
            raise RuntimeError("Empty tensor cannot be copied")
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        # Validate buffer sizes match
        expected_size = input_tensor.numel() * input_tensor.element_size()
        allocated_size = ctx.inputs[0]['size'] if 'size' in ctx.inputs[0] else expected_size
        if expected_size != allocated_size:
            raise RuntimeError(f"Buffer size mismatch: expected {expected_size}, allocated {allocated_size}")

        # Validate memory addresses are valid
        if ctx.inputs[0]['device'] == 0:
            raise RuntimeError("Invalid device buffer address")
        
        # Ensure previous operations complete
        # ctx.stream.synchronize()  # Removed unnecessary pre-sync
        
        # Enhanced error logging
        ctx.logger.debug(f"TensorRT inference: input_tensor shape={input_tensor.shape}, dtype={input_tensor.dtype}, device={input_tensor.device}")
        ctx.logger.debug(f"Buffer info: input_buffer_size={ctx.inputs[0].get('size', 'unknown')}, output_buffer_size={ctx.outputs[0].get('size', 'unknown')}")
        
        # Direct GPU-to-GPU memory copy with error recovery
        try:
            cuda.memcpy_dtod_async(
                ctx.inputs[0]['device'], 
                input_tensor.data_ptr(), 
                expected_size,
                ctx.stream
            )
        except Exception as e:
            ctx.logger.error(f"CUDA memory copy failed: {e}")
            ctx.logger.error(f"Input tensor: shape={input_tensor.shape}, device={input_tensor.device}, dtype={input_tensor.dtype}")
            ctx.logger.error(f"Buffer info: device_ptr={ctx.inputs[0]['device']}, size={ctx.inputs[0].get('size', 'unknown')}")
            raise RuntimeError(f"cuMemcpyDtoDAsync failed: {e}")
        
        # Set tensor addresses for modern API
        if hasattr(ctx.context, 'set_tensor_address'):
            # TensorRT 10.x API - set tensor addresses
            for inp in ctx.inputs:
                ctx.context.set_tensor_address(inp['name'], int(inp['device']))
            for out in ctx.outputs:
                ctx.context.set_tensor_address(out['name'], int(out['device']))
        
        # Run inference using modern API
        if hasattr(ctx.context, 'execute_async_v3'):
            # TensorRT 10.x API
            ctx.context.execute_async_v3(stream_handle=ctx.stream.handle)
        else:
            # Fallback for older versions
            ctx.context.execute_async_v2(bindings=ctx.bindings, stream_handle=ctx.stream.handle)
        
        # Transfer output data back to host
        cuda.memcpy_dtoh_async(ctx.outputs[0]['host'], ctx.outputs[0]['device'], ctx.stream)
        
        # Synchronize stream
        ctx.stream.synchronize()
        
        # Convert output back to tensor
        output_shape = ctx.outputs[0]['shape']
        output_np = ctx.outputs[0]['host'].reshape(output_shape)
        output_tensor = torch.from_numpy(output_np).to(ctx.device, dtype=torch.float16)
        
        return output_tensor


class TensorRTExecutionContext:
    """
    Per-pipeline wrapper: owns its own execution context, stream and buffers,
    while sharing the deserialised ICudaEngine from TensorRTInferenceEngine.
    """
    def __init__(self, base_engine: "TensorRTInferenceEngine"):
        self.base = base_engine
        self.device = base_engine.device
        self.memory_pool = base_engine.memory_pool
        self.logger = logging.getLogger(f"{__name__}.TensorRTExecCtx")
        self._infer_lock = threading.Lock()  # context-local
        self.base_model_manager = base_engine.model_manager
        
        # Cleanup state tracking
        self._cleanup_lock = threading.Lock()
        self._is_cleaned_up = False
        self._context_id = id(self)  # Unique identifier for this context

        # create context & stream
        self.context = base_engine.engine.create_execution_context()
        self.stream = cuda.Stream()

        # allocate new buffers – copy logic from base_engine._allocate_buffers()
        self.inputs, self.outputs, self.bindings = \
            _allocate_ctx_buffers(self.context, self.memory_pool, self.device)
        
        # Log context creation for debugging
        self.logger.debug(f"Created execution context {self._context_id}")

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        with self._infer_lock:
            return _infer_with_ctx(self, input_tensor)

    def cleanup(self):
        """Cleanup method for this execution context only"""
        # Prevent double cleanup with atomic check-and-set
        with self._cleanup_lock:
            if self._is_cleaned_up:
                if not is_shutting_down():
                    self.logger.debug(f"Context {self._context_id} already cleaned up, skipping")
                return
            self._is_cleaned_up = True
        
        if not is_shutting_down():
            self.logger.debug(f"Starting cleanup for context {self._context_id}")
        
        try:
            # Check if CUDA is still available before cleanup
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                if not is_shutting_down():
                    self.logger.warning("CUDA context already destroyed, skipping TensorRT cleanup")
                return
            
            if hasattr(self, 'memory_pool') and hasattr(self, 'inputs') and hasattr(self, 'outputs'):
                # Return input buffers to pool
                for inp in getattr(self, 'inputs', []):
                    if 'device_tensor' in inp and 'alloc_id' in inp:
                        try:
                            self.memory_pool.return_tensor(inp['device_tensor'], inp['alloc_id'])
                            if not is_shutting_down():
                                self.logger.debug(f"Returned input tensor to pool: alloc_id={inp['alloc_id']}")
                        except Exception as e:
                            if not is_shutting_down():
                                self.logger.warning(f"Failed to return input tensor to pool: {e}")
                
                # Return output buffers to pool
                for out in getattr(self, 'outputs', []):
                    if 'device_tensor' in out and 'alloc_id' in out:
                        try:
                            self.memory_pool.return_tensor(out['device_tensor'], out['alloc_id'])
                            if not is_shutting_down():
                                self.logger.debug(f"Returned output tensor to pool: alloc_id={out['alloc_id']}")
                        except Exception as e:
                            if not is_shutting_down():
                                self.logger.warning(f"Failed to return output tensor to pool: {e}")
                
                # Clear references
                self.inputs = []
                self.outputs = []
                
                # Clean up TensorRT objects
                try:
                    if hasattr(self, 'context') and self.context is not None:
                        del self.context
                        self.context = None
                except Exception as e:
                    if not is_shutting_down():
                        self.logger.warning(f"Error cleaning up TensorRT context: {e}")
                
                try:
                    if hasattr(self, 'stream') and self.stream is not None:
                        self.stream.synchronize()
                        del self.stream
                        self.stream = None
                except Exception as e:
                    if not is_shutting_down():
                        self.logger.warning(f"Error cleaning up CUDA stream: {e}")
                
                if not is_shutting_down():
                    self.logger.info("TensorRT execution context memory returned to pool")
                    
        except Exception as e:
            if not is_shutting_down():
                self.logger.error(f"Error during TensorRT execution context cleanup: {e}")
        finally:
            # Decrement context counter after all cleanup is complete
            self._decrement_context_counter()
        
        if not is_shutting_down():
            self.logger.debug(f"Cleanup completed for context {self._context_id}")

    def _decrement_context_counter(self):
        """Safely decrement the context counter with error handling"""
        try:
            if hasattr(self, 'base_model_manager') and self.base_model_manager:
                with self.base_model_manager._ctx_lock:
                    if self.base_model_manager._active_contexts > 0:
                        self.base_model_manager._active_contexts -= 1
                        if not is_shutting_down():
                            self.logger.debug(f"Decremented context counter to {self.base_model_manager._active_contexts}")
                    else:
                        if not is_shutting_down():
                            self.logger.warning(f"Context counter already at 0, cannot decrement for context {self._context_id}")
                    self.base_model_manager._ctx_cv.notify_all()
        except Exception as e:
            if not is_shutting_down():
                self.logger.error(f"Error decrementing context counter for context {self._context_id}: {e}")


class TensorRTModelManager:
    """Manages all TensorRT inference engines with strict GPU enforcement"""
    
    _instance_lock = threading.Lock()
    _shared_instance = None
    
    @classmethod
    def get_shared(cls, config: AppConfig) -> "TensorRTModelManager":
        with cls._instance_lock:
            if cls._shared_instance is None:
                cls._shared_instance = cls(config)
            return cls._shared_instance
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.device = torch.device(config.models.DEVICE)
        self.logger = logging.getLogger(f"{__name__}.TensorRTModelManager")
        
        # Strict GPU enforcement
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GPU-only inference requires CUDA.")
        
        if not config.models.FORCE_GPU_ONLY:
            raise RuntimeError("FORCE_GPU_ONLY must be enabled for TensorRT inference")
        
        # Set device
        torch.cuda.set_device(self.device)
        
        # Use unified GPU memory pool
        self.memory_pool = get_global_memory_pool(device=str(self.device))
        
        # Initialize engines
        self.engines = {}
        # Context lifetime management
        self._ctx_lock = threading.Lock()
        self._active_contexts = 0
        self._ctx_cv = threading.Condition(self._ctx_lock)
        
        self._load_all_engines()
        
        self.logger.info(f"TensorRT Model Manager initialized with {len(self.engines)} engines")
        self.logger.info(f"GPU-only mode enforced on device: {self.device}")
        self.logger.info(f"Using unified memory pool with {self.memory_pool.get_stats()['total_allocated_mb']:.2f} MB pre-allocated")
    
    def create_pipeline_contexts(self) -> Dict[str, TensorRTExecutionContext]:
        """Create execution contexts for all engines - one per pipeline"""
        contexts = {}
        created_count = 0
        
        for name, eng in self.engines.items():
            context = None
            try:
                context = eng.create_execution_context()
            finally:
                # increment only if creation succeeded
                if context is not None:
                    with self._ctx_lock:
                        self._active_contexts += 1
                        created_count += 1
            contexts[name] = context
            
            # Validate that the context was created properly
            if not hasattr(context, '_context_id'):
                self.logger.error(f"Context {name} missing required _context_id attribute")
            else:
                self.logger.debug(f"Created context {name} with ID {context._context_id}")
        
        self.logger.info(f"Created {created_count} execution contexts, total active: {self._active_contexts}")
        
        # Validate singleton behavior
        if hasattr(self.__class__, '_shared_instance') and self.__class__._shared_instance is not self:
            self.logger.error("Singleton violation detected: multiple TensorRTModelManager instances exist!")
        else:
            self.logger.debug("Singleton validation passed: using shared instance")
        
        return contexts
    
    def _load_all_engines(self):
        """Load all available TensorRT engines"""
        engine_configs = [
            ("detection", self.config.models.DETECTION_ENGINE_PATH),
            ("pose", self.config.models.POSE_ENGINE_PATH),
            ("segmentation", self.config.models.SEGMENTATION_ENGINE_PATH),
            ("reid", self.config.models.REID_ENGINE_PATH)
        ]
        
        for engine_name, engine_path in engine_configs:
            if os.path.exists(engine_path):
                try:
                    engine = TensorRTInferenceEngine(engine_path, str(self.device))
                    engine.model_manager = self  # Set reference to model manager
                    self.engines[engine_name] = engine
                    self.logger.info(f"Loaded {engine_name} TensorRT engine")
                except Exception as e:
                    self.logger.error(f"Failed to load {engine_name} engine: {e}")
                    if self.config.models.FORCE_GPU_ONLY:
                        raise RuntimeError(f"Critical: Failed to load required {engine_name} engine")
            else:
                self.logger.warning(f"TensorRT engine not found: {engine_path}")
                if self.config.models.FORCE_GPU_ONLY:
                    raise FileNotFoundError(f"Required TensorRT engine missing: {engine_path}")
    
    def detect_objects(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run object detection inference
        
        Args:
            image_tensor: Input image tensor (1, 3, 640, 640) on GPU, FP16
            
        Returns:
            Detection results tensor on GPU, FP16
        """
        if "detection" not in self.engines:
            raise RuntimeError("Detection engine not available")
        
        return self.engines["detection"].infer(image_tensor)
    
    def estimate_pose(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run pose estimation inference
        
        Args:
            image_tensor: Input image tensor (1, 3, 640, 640) on GPU, FP16
            
        Returns:
            Pose keypoints tensor on GPU, FP16
        """
        if "pose" not in self.engines:
            raise RuntimeError("Pose engine not available")
        
        return self.engines["pose"].infer(image_tensor)
    
    def segment_objects(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run segmentation inference
        
        Args:
            image_tensor: Input image tensor (1, 3, 640, 640) on GPU, FP16
            
        Returns:
            Segmentation masks tensor on GPU, FP16
        """
        if "segmentation" not in self.engines:
            raise RuntimeError("Segmentation engine not available")
        
        return self.engines["segmentation"].infer(image_tensor)
    
    def extract_features(self, person_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract ReID features
        
        Args:
            person_tensor: Person crop tensor (1, 3, 256, 128) on GPU, FP16
            
        Returns:
            Feature vector tensor on GPU, FP16
        """
        if "reid" not in self.engines:
            raise RuntimeError("ReID engine not available")
        
        return self.engines["reid"].infer(person_tensor)

    def cleanup(self):
        """Explicit cleanup method to return all engine memory immediately"""
        # Check if CUDA is still available before cleanup
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            if not is_shutting_down():
                self.logger.warning("CUDA context already destroyed, skipping TensorRT Model Manager cleanup")
            return
        
        if not is_shutting_down():
            self.logger.info("Cleaning up TensorRT Model Manager...")
        
        # Wait for all active contexts to finish
        with self._ctx_cv:
            while self._active_contexts > 0:
                if not is_shutting_down():
                    self.logger.info(f"Waiting for {self._active_contexts} contexts to finish...")
                self._ctx_cv.wait(timeout=1.0)
        
        for engine_name, engine in self.engines.items():
            try:
                if hasattr(engine, 'cleanup'):
                    engine.cleanup()
                    if not is_shutting_down():
                        self.logger.info(f"Cleaned up {engine_name} engine")
            except Exception as e:
                if not is_shutting_down():
                    self.logger.warning(f"Failed to cleanup {engine_name} engine: {e}")
        
        self.engines.clear()
        
        if not is_shutting_down():
            self.logger.info("TensorRT Model Manager cleanup complete")


class GPUOnlyDetectionManager:
    """
    GPU-only detection manager that replaces the existing DetectionManager
    with TensorRT optimization and strict GPU enforcement
    """
    
    def __init__(self, config: AppConfig, model_manager: TensorRTModelManager = None):
        self.config = config
        self.device = torch.device(config.models.DEVICE)
        self.logger = logging.getLogger(f"{__name__}.GPUOnlyDetectionManager")
        
        # Ensure CUDA is initialized
        self.cuda_manager = _ensure_cuda_initialized()
        
        # Strict GPU enforcement
        if not config.models.FORCE_GPU_ONLY:
            raise RuntimeError("GPU-only detection requires FORCE_GPU_ONLY=True")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        # Initialize TensorRT model manager (singleton)
        self.model_manager = model_manager or TensorRTModelManager.get_shared(config)
        # one context per model for THIS pipeline
        self.ctx = self.model_manager.create_pipeline_contexts()
        
        # Initialize preprocessing (GPU-only)
        self.input_size = (640, 640)  # Standard YOLO input size
        
        self.logger.info("GPU-only Detection Manager initialized")
        self.logger.info(f"All inference will run on: {self.device}")
        self.logger.info("CPU fallback is DISABLED")
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame on GPU with FP16 precision
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            Preprocessed tensor on GPU, FP16
        """
        # Convert to tensor and move to GPU
        frame_tensor = torch.from_numpy(frame).to(self.device, dtype=torch.float16)
        
        # Ensure correct channel order (BGR -> RGB)
        if frame_tensor.shape[2] == 3:
            frame_tensor = frame_tensor[:, :, [2, 1, 0]]  # BGR to RGB
        
        # Resize to target size
        frame_tensor = torch.nn.functional.interpolate(
            frame_tensor.permute(2, 0, 1).unsqueeze(0),  # (H, W, C) -> (1, C, H, W)
            size=self.input_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize to [0, 1]
        frame_tensor = frame_tensor / 255.0
        
        return frame_tensor
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], float]:
        """
        Process frame with GPU-only TensorRT inference
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (detections, processing_time_ms)
        """
        # Phase 2 – DeepStream now supplies detections; skip TensorRT path entirely.
        return [], 0.0
        
        start_time = time.time()
        
        try:
            # Preprocess frame on GPU
            input_tensor = self.preprocess_frame(frame)
            
            # Run detection inference
            detection_output = self.ctx["detection"].infer(input_tensor)
            
            # Run pose inference if available
            pose_output = None
            if "pose" in self.ctx:
                pose_output = self.ctx["pose"].infer(input_tensor)
            
            # Run segmentation inference if available
            segmentation_output = None
            if "segmentation" in self.ctx:
                segmentation_output = self.ctx["segmentation"].infer(input_tensor)
            
            # Post-process results (on GPU)
            detections = self._postprocess_detections(
                detection_output, 
                pose_output, 
                segmentation_output,
                frame.shape
            )
            
            # Extract ReID features for person detections
            detections = self._extract_reid_features(detections, frame, input_tensor)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return detections, processing_time
            
        except Exception as e:
            self.logger.error(f"GPU inference failed: {e}")
            if self.config.models.FORCE_GPU_ONLY:
                raise RuntimeError(f"GPU-only inference failed: {e}")
            return [], 0.0
    
    def process_tensor(self, tensor: torch.Tensor) -> Tuple[List[Detection], float]:
        """
        Process tensor directly with pure GPU pipeline - ENHANCED for DALI integration.
        
        Args:
            tensor: Preprocessed tensor on GPU (C, H, W) or (1, C, H, W), FP16
            
        Returns:
            Tuple of (detections, processing_time_ms)
        """
        # Phase 2 – DeepStream now supplies detections; skip TensorRT path entirely.
        return [], 0.0
        
        start_time = time.time()
        
        try:
            # Validate tensor format
            if tensor.device.type != 'cuda':
                raise RuntimeError("GPU-only mode: Input tensor must be on GPU")
            
            # Handle both (C, H, W) and (1, C, H, W) formats from DALI
            if tensor.dim() == 3:
                # Add batch dimension: (C, H, W) -> (1, C, H, W)
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() == 4:
                if tensor.shape[0] != 1:
                    raise RuntimeError(f"Expected batch size 1, got {tensor.shape[0]}")
            else:
                raise RuntimeError(f"Expected tensor shape (C, H, W) or (1, C, H, W), got {tensor.shape}")
            
            # Ensure FP16 precision for optimal TensorRT performance
            if tensor.dtype != torch.float16:
                tensor = tensor.half()
            
            # Run detection inference directly with preprocessed tensor
            detection_output = self.ctx["detection"].infer(tensor)
            
            # Run pose inference if available
            pose_output = None
            if "pose" in self.ctx:
                pose_output = self.ctx["pose"].infer(tensor)
            
            # Run segmentation inference if available
            segmentation_output = None
            if "segmentation" in self.ctx:
                segmentation_output = self.ctx["segmentation"].infer(tensor)
            
            # Calculate original shape from tensor
            # DALI provides preprocessed tensors at target resolution
            tensor_shape = tensor.shape  # (1, C, H, W)
            original_shape = (tensor_shape[2], tensor_shape[3], tensor_shape[1])  # (H, W, C)
            
            # Post-process results (on GPU) - PURE GPU PROCESSING
            detections = self._postprocess_detections_gpu_only(
                detection_output, 
                pose_output, 
                segmentation_output,
                original_shape
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            return detections, processing_time_ms
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"GPU tensor inference failed: {e}")
            if self.config.models.FORCE_GPU_ONLY:
                raise RuntimeError(f"GPU-only tensor inference failed: {e}")
            return [], processing_time_ms
    
    def _postprocess_detections(
        self, 
        detection_output: torch.Tensor,
        pose_output: Optional[torch.Tensor],
        segmentation_output: Optional[torch.Tensor],
        original_shape: Tuple[int, int, int]
    ) -> List[Dict]:
        """
        Post-process detection results on GPU
        
        Args:
            detection_output: Raw detection tensor from TensorRT
            pose_output: Raw pose tensor from TensorRT (optional)
            segmentation_output: Raw segmentation tensor from TensorRT (optional)
            original_shape: Original frame shape for coordinate scaling
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        try:
            # Strict GPU enforcement
            if self.config.models.FORCE_GPU_ONLY:
                if detection_output.device.type != 'cuda':
                    raise RuntimeError("GPU enforcement violation: detection output not on GPU")
                if detection_output.dtype != torch.float16:
                    raise RuntimeError("GPU enforcement violation: detection output not FP16")
            
            # YOLO output format: [batch, num_detections, 85] where 85 = [x, y, w, h, conf, class_probs...]
            # Apply confidence threshold filtering on GPU
            conf_mask = detection_output[0, :, 4] > self.config.models.MODEL_CONFIDENCE_THRESHOLD
            filtered_detections = detection_output[0, conf_mask]
            
            if len(filtered_detections) > 0:
                # Convert xywh to xyxy format on GPU
                boxes = filtered_detections[:, :4].clone()
                boxes[:, 0] -= boxes[:, 2] / 2  # x_center - width/2 = x1
                boxes[:, 1] -= boxes[:, 3] / 2  # y_center - height/2 = y1
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x1 + width = x2
                boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y1 + height = y2
                
                # Scale coordinates to original frame size
                scale_x = original_shape[1] / self.input_size[1]  # width scale
                scale_y = original_shape[0] / self.input_size[0]  # height scale
                
                boxes[:, [0, 2]] *= scale_x  # Scale x coordinates
                boxes[:, [1, 3]] *= scale_y  # Scale y coordinates
                
                # Get class predictions and confidences
                confidences = filtered_detections[:, 4]
                class_probs = filtered_detections[:, 5:]
                class_ids = torch.argmax(class_probs, dim=1)
                
                # Apply NMS on GPU using torchvision
                import torchvision
                keep_indices = torchvision.ops.nms(
                    boxes, 
                    confidences, 
                    iou_threshold=self.config.models.MODEL_IOU_THRESHOLD
                )
                
                # Convert final results to list of dictionaries
                final_boxes = boxes[keep_indices]
                final_confidences = confidences[keep_indices]
                final_class_ids = class_ids[keep_indices]
                
                # STRICT GPU-only: Keep tensors on GPU until absolutely required
                if not self.config.models.FORCE_GPU_ONLY:
                    raise RuntimeError("GPU-only mode enforced: FORCE_GPU_ONLY must be True")
                
                # Minimal CPU conversion only for final output format
                boxes_cpu = final_boxes.detach().cpu().numpy()
                confs_cpu = final_confidences.detach().cpu().numpy()
                classes_cpu = final_class_ids.detach().cpu().numpy()
                
                # Build detection dictionaries
                for i in range(len(boxes_cpu)):
                    class_id = int(classes_cpu[i])
                    
                    # Filter by target classes if specified
                    if (self.config.models.TARGET_CLASSES and 
                        len(self.config.models.TARGET_CLASSES) > 0 and 
                        class_id not in self.config.models.TARGET_CLASSES):
                        continue
                    
                    detection = {
                        'bbox': boxes_cpu[i].tolist(),
                        'confidence': float(confs_cpu[i]),
                        'class_id': class_id,
                        'class_name': self.config.models.CLASS_NAMES.get(class_id, f"class_{class_id}"),
                        'keypoints': None,
                        'mask': None,
                        'features': None
                    }
                    detections.append(detection)
                
                self.logger.debug(f"Post-processed {len(detections)} detections from {len(filtered_detections)} candidates")
            
        except Exception as e:
            self.logger.error(f"Error in post-processing detections: {e}")
            if self.config.models.FORCE_GPU_ONLY:
                raise RuntimeError(f"GPU-only post-processing failed: {e}")
        
        return detections
    
    def _postprocess_detections_gpu_only(
        self, 
        detection_output: torch.Tensor,
        pose_output: Optional[torch.Tensor],
        segmentation_output: Optional[torch.Tensor],
        original_shape: Tuple[int, int, int]
    ) -> List[Detection]:
        """
        Post-process detection results with PURE GPU operations - NO CPU conversion.
        
        This method is optimized for DALI integration and eliminates all tensor→numpy
        conversions until the final output format is required.
        
        Args:
            detection_output: Raw detection tensor from TensorRT
            pose_output: Raw pose tensor from TensorRT (optional)
            segmentation_output: Raw segmentation tensor from TensorRT (optional)
            original_shape: Original frame shape for coordinate scaling (H, W, C)
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        try:
            # Enhanced detection processing logging
            self.logger.debug(f"Processing detection output: shape={detection_output.shape}, dtype={detection_output.dtype}, device={detection_output.device}")
            self.logger.debug(f"Original shape: {original_shape}, confidence threshold: {self.config.models.MODEL_CONFIDENCE_THRESHOLD}")
            
            # Strict GPU enforcement
            if self.config.models.FORCE_GPU_ONLY:
                if detection_output.device.type != 'cuda':
                    raise RuntimeError("GPU enforcement violation: detection output not on GPU")
                if detection_output.dtype != torch.float16:
                    raise RuntimeError("GPU enforcement violation: detection output not FP16")
            
            # YOLO output format: [batch, num_detections, 85] where 85 = [x, y, w, h, conf, class_probs...]
            # Apply confidence threshold filtering on GPU
            conf_mask = detection_output[0, :, 4] > self.config.models.MODEL_CONFIDENCE_THRESHOLD
            filtered_detections = detection_output[0, conf_mask]
            
            if len(filtered_detections) > 0:
                # Convert xywh to xyxy format on GPU
                boxes = filtered_detections[:, :4].clone()
                boxes[:, 0] -= boxes[:, 2] / 2  # x_center - width/2 = x1
                boxes[:, 1] -= boxes[:, 3] / 2  # y_center - height/2 = y1
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x1 + width = x2
                boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y1 + height = y2
                
                # Scale coordinates to original frame size
                # DALI provides frames at target resolution, so we scale from model input to original
                scale_x = original_shape[1] / self.input_size[1]  # width scale
                scale_y = original_shape[0] / self.input_size[0]  # height scale
                
                boxes[:, [0, 2]] *= scale_x  # Scale x coordinates
                boxes[:, [1, 3]] *= scale_y  # Scale y coordinates
                
                # Get class predictions and confidences
                confidences = filtered_detections[:, 4]
                class_probs = filtered_detections[:, 5:]
                class_ids = torch.argmax(class_probs, dim=1)
                
                # Apply NMS on GPU using torchvision
                import torchvision
                keep_indices = torchvision.ops.nms(
                    boxes, 
                    confidences, 
                    iou_threshold=self.config.models.MODEL_IOU_THRESHOLD
                )
                
                # Final GPU tensors after NMS
                final_boxes = boxes[keep_indices]
                final_confidences = confidences[keep_indices]
                final_class_ids = class_ids[keep_indices]
                
                # CRITICAL: Minimal CPU conversion ONLY for final output format
                # This is the ONLY place where GPU→CPU transfer happens
                if not self.config.models.FORCE_GPU_ONLY:
                    raise RuntimeError("GPU-only mode enforced: FORCE_GPU_ONLY must be True")
                
                # Convert to CPU only at the very end for output format compatibility
                boxes_cpu = final_boxes.detach().cpu().numpy()
                confs_cpu = final_confidences.detach().cpu().numpy()
                classes_cpu = final_class_ids.detach().cpu().numpy()
                
                # Build Detection objects
                for i in range(len(boxes_cpu)):
                    class_id = int(classes_cpu[i])
                    
                    # Filter by target classes if specified
                    if (self.config.models.TARGET_CLASSES and 
                        len(self.config.models.TARGET_CLASSES) > 0 and 
                        class_id not in self.config.models.TARGET_CLASSES):
                        continue
                    
                    # CREATE Detection object instead of dictionary
                    detection = Detection(
                        bbox=tuple(boxes_cpu[i].tolist()),
                        confidence=float(confs_cpu[i]),
                        class_id=class_id,
                        class_name=self.config.models.CLASS_NAMES.get(class_id, f"class_{class_id}")
                    )
                    detections.append(detection)
                
                self.logger.debug(f"GPU-only post-processed {len(detections)} detections from {len(filtered_detections)} candidates")
            
        except Exception as e:
            self.logger.error(f"Error in GPU-only post-processing: {e}")
            if self.config.models.FORCE_GPU_ONLY:
                raise RuntimeError(f"GPU-only post-processing failed: {e}")
        
        return detections
    
    def _extract_reid_features(
        self, 
        detections: List[Dict], 
        frame: np.ndarray,
        frame_tensor: torch.Tensor
    ) -> List[Dict]:
        """
        Extract ReID features for person detections on GPU
        
        Args:
            detections: List of detection dictionaries
            frame: Original frame (for fallback)
            frame_tensor: Preprocessed frame tensor on GPU
            
        Returns:
            Detections with added feature vectors
        """
        if "reid" not in self.ctx:
            return detections
        
        for detection in detections:
            if detection.get('class_id') == 0:  # Person class
                try:
                    # Strict GPU enforcement
                    if self.config.models.FORCE_GPU_ONLY:
                        if frame_tensor.device.type != 'cuda':
                            raise RuntimeError("GPU enforcement violation: frame tensor not on GPU")
                    
                    # Get bbox coordinates
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    # Convert bbox coordinates to tensor space
                    # Original frame: frame.shape, Tensor: frame_tensor.shape
                    orig_h, orig_w = frame.shape[:2]
                    tensor_h, tensor_w = frame_tensor.shape[2], frame_tensor.shape[3]
                    
                    # Scale bbox to tensor coordinates
                    scale_x = tensor_w / orig_w
                    scale_y = tensor_h / orig_h
                    
                    tx1 = int(x1 * scale_x)
                    ty1 = int(y1 * scale_y)
                    tx2 = int(x2 * scale_x)
                    ty2 = int(y2 * scale_y)
                    
                    # Ensure coordinates are within bounds
                    tx1 = max(0, min(tx1, tensor_w - 1))
                    ty1 = max(0, min(ty1, tensor_h - 1))
                    tx2 = max(tx1 + 1, min(tx2, tensor_w))
                    ty2 = max(ty1 + 1, min(ty2, tensor_h))
                    
                    # Crop person region from GPU tensor
                    person_crop = frame_tensor[0, :, ty1:ty2, tx1:tx2]  # Shape: [C, H, W]
                    
                    if person_crop.shape[1] > 0 and person_crop.shape[2] > 0:
                        # Resize to ReID input size (256, 128) on GPU
                        person_resized = torch.nn.functional.interpolate(
                            person_crop.unsqueeze(0),  # Add batch dim: [1, C, H, W]
                            size=(256, 128),
                            mode='bilinear',
                            align_corners=False
                        )
                        
                        # Ensure FP16 precision
                        if self.config.models.FORCE_GPU_ONLY and person_resized.dtype != torch.float16:
                            person_resized = person_resized.half()
                        
                        # Extract features using TensorRT ReID model
                        features = self.ctx["reid"].infer(person_resized)
                        
                        # Strict GPU-only: Minimal CPU conversion for final output
                        if features is not None:
                            if not self.config.models.FORCE_GPU_ONLY:
                                raise RuntimeError("GPU-only mode enforced: FORCE_GPU_ONLY must be True")
                            
                            if features.device.type != 'cuda':
                                raise RuntimeError("GPU enforcement violation: ReID features not on GPU")
                            
                            # Final CPU conversion for output format only
                            features_np = features.detach().cpu().numpy().flatten()
                            
                            # Normalize features
                            norm = np.linalg.norm(features_np)
                            if norm > 0:
                                features_np = features_np / norm
                            
                            detection['features'] = features_np
                        else:
                            detection['features'] = None
                    else:
                        self.logger.warning(f"Invalid person crop size: {person_crop.shape}")
                        detection['features'] = None
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract ReID features: {e}")
                    if self.config.models.FORCE_GPU_ONLY:
                        raise RuntimeError(f"GPU-only ReID feature extraction failed: {e}")
                    detection['features'] = None
        
        return detections
    
    def cleanup(self):
        """Cleanup GPU detection manager resources"""
        # Check if CUDA is still available before cleanup
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            if not is_shutting_down():
                self.logger.warning("CUDA context already destroyed, skipping GPU detection manager cleanup")
            return
        
        try:
            # Clean up execution contexts
            if hasattr(self, 'ctx'):
                for ctx_name, ctx in self.ctx.items():
                    try:
                        if hasattr(ctx, 'cleanup'):
                            ctx.cleanup()
                            if not is_shutting_down():
                                self.logger.debug(f"Cleaned up execution context: {ctx_name}")
                    except Exception as e:
                        if not is_shutting_down():
                            self.logger.warning(f"Failed to cleanup execution context {ctx_name}: {e}")
                self.ctx.clear()
            
            # Clean up model manager
            if hasattr(self, 'model_manager') and self.model_manager:
                self.model_manager.cleanup()
                if not is_shutting_down():
                    self.logger.info("GPU detection manager cleanup complete")
        except Exception as e:
            if not is_shutting_down():
                self.logger.error(f"Error during GPU detection manager cleanup: {e}")


# Factory function to create GPU-only detection manager
def create_gpu_detection_manager(config: AppConfig) -> GPUOnlyDetectionManager:
    """Create and initialize GPU-only detection manager"""
    
    # Verify TensorRT engines exist
    from tensorrt_builder import build_tensorrt_engines
    
    if config.models.ENABLE_TENSORRT:
        # Build engines if they don't exist
        try:
            engines = build_tensorrt_engines(config)
            logger.info(f"TensorRT engines ready: {list(engines.keys())}")
        except Exception as e:
            logger.error(f"Failed to build TensorRT engines: {e}")
            raise
    
    return GPUOnlyDetectionManager(config)


if __name__ == "__main__":
    # Test TensorRT inference
    from config import config
    
    logging.basicConfig(level=logging.INFO)
    
    if config.models.ENABLE_TENSORRT and config.models.FORCE_GPU_ONLY:
        detection_manager = create_gpu_detection_manager(config)
        
        # Create dummy frame for testing
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run inference
        detections, time_ms = detection_manager.process_frame(test_frame)
        print(f"Processed frame in {time_ms:.2f}ms, found {len(detections)} detections")
    else:
        print("TensorRT inference is not enabled or GPU-only mode is not forced") 