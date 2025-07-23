#!/usr/bin/env python3
"""
Centralized CUDA Context Manager

This module provides centralized CUDA context management to replace PyCUDA's
autoinit and optimize GPU resource usage across multiple components.

Key features:
- Shared CUDA context across all components using PyTorch's primary context
- Eliminates dual-context bugs by attaching to existing PyTorch context
- Context push/pop methods are now lightweight no-ops (backward compatibility)
- Minimal context switches between pipeline stages
- Context validation and error handling

IMPORTANT: This implementation uses cuda.Context.attach() to share PyTorch's
primary CUDA context, eliminating the "invalid resource handle" and 
"cuTensor permute execute failed" errors caused by multiple CUDA contexts.
"""

import pycuda.driver as cuda
import torch
import threading
import logging
from typing import Optional, Dict, Any
import atexit


class CUDAContextManager:
    """
    Centralized CUDA context manager for optimized GPU resource usage.
    
    This singleton class manages a shared CUDA context across all pipeline
    components, replacing the need for pycuda.autoinit in each module.
    
    SHARED CONTEXT DESIGN:
    - Uses cuda.Context.attach() to share PyTorch's primary CUDA context
    - Eliminates dual-context bugs that cause "invalid resource handle" errors
    - Context push/pop methods are now no-ops for backward compatibility
    - All CUDA operations share the same context, preventing resource conflicts
    """
    
    _instance: Optional['CUDAContextManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls, device_id: int = 0):
        """Ensure singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, device_id: int = 0):
        """Initialize CUDA context manager (only runs once)"""
        # Skip if already initialized
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.device_id = device_id
        self.logger = logging.getLogger("CUDAContextManager")
        
        # Thread-local storage for context stack
        self._thread_local = threading.local()
        
        # Initialize CUDA
        self._init_cuda()
        
        # Register cleanup
        atexit.register(self.cleanup)
        
        self.logger.info(f"CUDA context manager initialized for device {device_id}")
    
    def _init_cuda(self):
        """Initialize CUDA driver and attach to PyTorch's primary context"""
        try:
            # Initialize CUDA driver
            cuda.init()
            
            # Check device availability
            device_count = cuda.Device.count()
            if device_count == 0:
                raise RuntimeError("No CUDA devices available")
            
            if self.device_id >= device_count:
                raise RuntimeError(f"Invalid device ID {self.device_id}, only {device_count} devices available")
            
            # Get device
            self.device = cuda.Device(self.device_id)
            
            # Attach to PyTorch's primary context instead of creating a new one
            self.context = cuda.Context.attach()
            
            # Get device properties
            self.device_name = self.device.name()
            self.compute_capability = self.device.compute_capability()
            self.total_memory = self.device.total_memory()
            
            self.logger.info(f"CUDA device initialized: {self.device_name}")
            self.logger.info(f"Compute capability: {self.compute_capability}")
            self.logger.info(f"Total memory: {self.total_memory / (1024**3):.2f} GB")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CUDA: {e}")
            raise RuntimeError(f"CUDA initialization failed: {e}")
    
    def push_context(self) -> bool:
        """
        Push CUDA context onto the current thread's stack.
        
        NOTE: This is now a no-op since we use PyTorch's shared primary context.
        The context is always current and doesn't need explicit push/pop operations.
        
        Returns:
            bool: Always returns False (context already current)
        """
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Context push requested on thread {threading.current_thread().name} (no-op)")
        
        return False
    
    def pop_context(self) -> bool:
        """
        Pop CUDA context from the current thread's stack.
        
        NOTE: This is now a no-op since we use PyTorch's shared primary context.
        The context is always current and doesn't need explicit push/pop operations.
        
        Returns:
            bool: Always returns False (no context to pop)
        """
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Context pop requested on thread {threading.current_thread().name} (no-op)")
        
        return False
    
    def synchronize(self):
        """Synchronize the CUDA context"""
        try:
            # Push context if needed
            pushed = self.push_context()
            
            # Synchronize
            cuda.Context.synchronize()
            
            # Pop if we pushed
            if pushed:
                self.pop_context()
                
        except Exception as e:
            self.logger.error(f"Failed to synchronize context: {e}")
            raise RuntimeError(f"Failed to synchronize CUDA context: {e}")
    
    def get_memory_info(self) -> Dict[str, int]:
        """
        Get current GPU memory usage information.
        
        Returns:
            Dict containing free and total memory in bytes
        """
        try:
            # Push context if needed
            pushed = self.push_context()
            
            # Get memory info
            free, total = cuda.mem_get_info()
            
            # Pop if we pushed
            if pushed:
                self.pop_context()
            
            return {
                'free': free,
                'total': total,
                'used': total - free,
                'free_mb': free / (1024 * 1024),
                'total_mb': total / (1024 * 1024),
                'used_mb': (total - free) / (1024 * 1024)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory info: {e}")
            return {'error': str(e)}
    
    def validate_context(self) -> bool:
        """
        Validate that the CUDA context is properly initialized and accessible.
        
        Returns:
            bool: True if context is valid
        """
        try:
            # Push context
            pushed = self.push_context()
            
            # Try to allocate a small amount of memory as a test
            test_mem = cuda.mem_alloc(1024)  # 1KB
            test_mem.free()
            
            # Pop if we pushed
            if pushed:
                self.pop_context()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Context validation failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up CUDA context (called automatically at exit)"""
        try:
            if hasattr(self, 'context') and self.context:
                # Since we're using PyTorch's shared primary context, we don't need
                # to explicitly detach - PyTorch will handle context cleanup
                self.logger.info("CUDA context cleaned up")
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    @classmethod
    def get_instance(cls, device_id: int = 0) -> 'CUDAContextManager':
        """Get or create the singleton instance"""
        if cls._instance is None:
            cls._instance = cls(device_id)
        return cls._instance


class CUDAContextScope:
    """
    Context manager for CUDA context push/pop operations.
    
    Usage:
        manager = CUDAContextManager.get_instance()
        with CUDAContextScope(manager):
            # CUDA operations here
            pass
    """
    
    def __init__(self, context_manager: CUDAContextManager):
        self.context_manager = context_manager
        self.pushed = False
    
    def __enter__(self):
        """Push context on enter"""
        self.pushed = self.context_manager.push_context()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Pop context on exit"""
        if self.pushed:
            self.context_manager.pop_context()


# Global function for easy access
def get_cuda_context_manager(device_id: int = 0) -> CUDAContextManager:
    """Get the global CUDA context manager instance"""
    return CUDAContextManager.get_instance(device_id)


# Replacement for pycuda.autoinit
def initialize_cuda(device_id: int = 0):
    """
    Initialize CUDA context manager as a replacement for pycuda.autoinit.
    
    This should be called once at the start of the application instead of
    importing pycuda.autoinit in each module.
    """
    manager = get_cuda_context_manager(device_id)
    
    # Validate context
    if not manager.validate_context():
        raise RuntimeError("Failed to initialize CUDA context")
    
    # Set PyTorch to use the same device
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        torch.cuda.init()
    
    return manager


# Test function
def test_cuda_context_manager():
    """Test the CUDA context manager"""
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing CUDA Context Manager...")
    
    # Initialize
    manager = initialize_cuda(device_id=0)
    
    # Test context operations
    print("\nTesting context push/pop...")
    with CUDAContextScope(manager):
        print("Context pushed")
        
        # Get memory info
        mem_info = manager.get_memory_info()
        print(f"GPU Memory: {mem_info['used_mb']:.2f}/{mem_info['total_mb']:.2f} MB used")
    
    print("Context popped")
    
    # Test from multiple threads
    import concurrent.futures
    
    def thread_test(thread_id):
        with CUDAContextScope(manager):
            mem_info = manager.get_memory_info()
            print(f"Thread {thread_id}: Memory used = {mem_info['used_mb']:.2f} MB")
            
            # Allocate some memory
            mem = cuda.mem_alloc(10 * 1024 * 1024)  # 10MB
            mem.free()
            
            return True
    
    print("\nTesting multi-threaded access...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(thread_test, i) for i in range(3)]
        results = [f.result() for f in futures]
    
    print(f"All threads completed: {all(results)}")
    
    # Validate context
    print(f"\nContext valid: {manager.validate_context()}")
    
    print("\nCUDA Context Manager test complete!")


if __name__ == "__main__":
    test_cuda_context_manager() 