# Clean Shutdown Implementation

## Overview

This document describes the comprehensive clean shutdown implementation that ensures Ctrl-C (SIGINT) properly terminates the application without hanging or leaving zombie processes.

## Problem Statement

The original application had several shutdown issues:
1. `main.py::_signal_handler` would exit early with `sys.exit(0)`, leaving threads/processes running
2. Thread joins with 5s timeouts each caused >30s total shutdown time
3. Second Ctrl-C while `.join()` was blocking would raise `KeyboardInterrupt` in the join caller
4. AsyncIO event-loop close sometimes blocked forever
5. Multiple signal handlers in dashboard servers created conflicts

## Solution Architecture

### A. Central Interrupt Controller (main.py)

**Global Variables:**
```python
INTERRUPT_COUNT = 0
MAX_SHUTDOWN_TIME = 15  # Maximum time to wait for graceful shutdown
```

**Enhanced Signal Handler:**
- Increments `INTERRUPT_COUNT` on each signal
- First interrupt (INTERRUPT_COUNT = 1): Starts graceful shutdown in background thread
- Second interrupt (INTERRUPT_COUNT ≥ 2): Forces immediate exit with `os._exit(1)`
- No more `sys.exit(0)` that leaves threads running

### B. Graceful Shutdown Thread

**Implementation:**
```python
def _graceful_exit(self):
    """Perform graceful shutdown with timeout"""
    try:
        self.logger.info("Starting graceful shutdown process...")
        start_time = time.time()
        
        # Call stop method
        self.stop()
        
        # Wait for shutdown to complete or timeout
        while self.running and (time.time() - start_time) < MAX_SHUTDOWN_TIME:
            time.sleep(0.1)
        
        if self.running:
            self.logger.error(f"Graceful shutdown timed out after {MAX_SHUTDOWN_TIME}s, forcing exit")
            os._exit(1)
        else:
            self.logger.info("Graceful shutdown completed successfully")
            os._exit(0)
            
    except Exception as e:
        self.logger.error(f"Error during graceful shutdown: {e}")
        os._exit(1)
```

### C. Safe Thread Joining Utility

**File:** `utils/interrupt.py`

**Key Features:**
- Reduces all join timeouts to 2 seconds
- Logs warnings for threads that don't stop gracefully
- Cannot set `daemon=True` on running threads (Python limitation)
- Threads are abandoned if they don't stop, allowing interpreter exit

**Usage:**
```python
from utils.interrupt import safe_join, safe_process_join

# For threads
safe_join(thread, timeout=2.0, name="thread_name")

# For processes
safe_process_join(process, timeout=2.0, name="process_name")
```

### D. AsyncIO Shutdown Fix

**Implementation:**
- Wraps WebSocket server stop in `asyncio.wait_for` with 3s timeout
- Uses `asyncio.run_coroutine_threadsafe` for thread-safe operation
- Always calls `asyncio.set_event_loop(None)` after loop close
- Prevents infinite blocking on event loop operations

### E. Dashboard Server Integration

**Changes Made:**
- Removed duplicate signal handlers from `dashboard_server.py` and `simple_dashboard_server.py`
- Added global instances for external cleanup
- Exposed `stop_dashboard()` functions for main application to call
- Integrated cleanup into main application's stop sequence

### F. Updated Components

**Files Modified:**
1. `main.py` - Central interrupt controller and graceful shutdown
2. `utils/interrupt.py` - Safe joining utilities (NEW)
3. `gpu_pipeline.py` - Updated to use safe_join
4. `dali_video_processor.py` - Updated to use safe_join  
5. `nvdec_reader.py` - Updated to use safe_join
6. `simple_dashboard_server.py` - Removed signal handlers, added stop function
7. `dashboard_server.py` - Removed signal handlers, added stop function
8. `tensorrt_inference.py` - TensorRT cleanup spam elimination (NEW)

## Behavior Changes

### Before
- One Ctrl-C: `sys.exit(0)` leaving threads running
- Shutdown time: >30 seconds due to sequential 5s timeouts
- Second Ctrl-C: Raised exception in join caller
- Dashboard servers: Conflicting signal handlers

### After  
- One Ctrl-C: Graceful shutdown with 15s total timeout
- Shutdown time: ≤15 seconds maximum
- Second Ctrl-C: Immediate exit with `os._exit(1)`
- Dashboard servers: Coordinated cleanup via main application

## Testing

**Acceptance Criteria Met:**
✅ One Ctrl-C → logs "graceful shutdown started", exits in ≤15s  
✅ Second Ctrl-C anytime → immediate exit  
✅ No "Exception ignored in:" tracebacks  
✅ GPU memory pool cleanup still executed on graceful path  
✅ No functional behavior changes during normal operation  

## Implementation Details

### Thread Join Timeouts Reduced
- All join timeouts reduced from 5s to 2s
- Total shutdown time reduced from >30s to ≤15s
- Threads that don't stop are logged and abandoned

### Process Management
- Analysis processes are terminated then killed if needed
- Multiprocessing manager shutdown is handled gracefully
- Process joins use timeout with kill fallback

### Memory Management
- GPU memory cleanup still occurs during graceful shutdown
- Memory pools are cleared before exit
- Torch CUDA cache is emptied

### Error Handling
- All cleanup operations are wrapped in try-catch blocks
- Errors during shutdown are logged but don't prevent exit
- Timeout mechanisms prevent infinite hanging

## Usage Notes

### For Developers
- Use `safe_join()` for all new thread join operations
- Set `daemon=True` on threads that don't need graceful shutdown
- Avoid blocking operations in cleanup code
- Test shutdown behavior with both single and double Ctrl-C

### For Users
- Single Ctrl-C: Wait up to 15 seconds for graceful shutdown
- Double Ctrl-C: Immediate exit if first shutdown is taking too long
- Look for "graceful shutdown started" message to confirm proper handling

## Future Improvements

1. **Thread Pool Management**: Consider using ThreadPoolExecutor for better thread lifecycle management
2. **Async Context Managers**: Use async context managers for resource cleanup
3. **Signal Handling**: Add SIGTERM handling for systemd compatibility
4. **Monitoring**: Add metrics for shutdown time and success rate
5. **Testing**: Automated tests for shutdown scenarios

### I. TensorRT CUDA Context Cleanup Fixes

**Problem Solved:** TensorRT was spamming ~50+ error messages about "invalid device context" during shutdown when engines were destroyed after CUDA context invalidation.

**Solution Implementation:**
- Added global shutdown flag `_shutting_down` to suppress TensorRT error logging during shutdown
- Added `set_tensorrt_shutdown_mode()` function to control error suppression
- Enhanced all TensorRT cleanup methods with CUDA context validation
- Implemented early GPU cleanup in main.py before stopping processors
- Reordered shutdown sequence to clean TensorRT engines while CUDA context is still valid

**Key Changes:**
```python
# Check CUDA availability before cleanup
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    if not _shutting_down:
        self.logger.warning("CUDA context already destroyed, skipping cleanup")
    return

# Suppress logging during shutdown
if not _shutting_down:
    self.logger.info("Normal cleanup message")
```

**Cleanup Order:**
1. Set TensorRT shutdown mode
2. Early GPU cleanup (TensorRT engines, memory pools)
3. Stop processors/pipelines
4. Stop threads/processes
5. Clean remaining resources

## Related Files

- `main.py` - Main application and signal handling
- `utils/interrupt.py` - Safe joining utilities
- `gpu_pipeline.py` - GPU pipeline shutdown
- `tensorrt_inference.py` - TensorRT cleanup spam elimination
- `*dashboard_server.py` - Dashboard server integration
- `*video_processor.py` - Video processor shutdown

This implementation ensures reliable, fast, and clean application shutdown while maintaining all existing functionality during normal operation. The TensorRT cleanup fixes eliminate error spam and ensure proper GPU resource cleanup order. 