# LibProxy Segmentation Fault Fix

## Problem
When running the application in Cursor's integrated terminal (or other IDEs), the application may crash with "aborted (core dumped)" error during DeepStream pipeline initialization. This is caused by libproxy trying to resolve proxy settings and causing a segmentation fault.

## Root Cause
The libproxy library attempts to detect proxy settings during GStreamer/DeepStream initialization, which can cause segmentation faults in certain terminal environments.

## Solution
Add the following environment variable fix to the top of key Python modules:

```python
import os
os.environ['no_proxy'] = '*'
```

## Modules That Need This Fix

### ✅ Already Fixed
- `main.py` - Line 18
- `config.py` - Line 18  
- `gpu_pipeline.py` - Line 12
- `deepstream_video_pipeline.py` - Line 20
- `tensorrt_inference.py` - Line 16
- `utils.py` - Line 21

### 🔧 Additional Modules That May Need This Fix
If you encounter the same issue when running other scripts, add the fix to:

- Any script that imports GStreamer/DeepStream components
- Any script that uses TensorRT inference
- Any script that creates CUDA contexts
- Test scripts and standalone modules

## Alternative Approaches
Some modules use different approaches that also work:

### Method 1: no_proxy (Recommended)
```python
import os
os.environ['no_proxy'] = '*'
```

### Method 2: LIBPROXY_IGNORE_PLUGINS
```python
import os
os.environ['LIBPROXY_IGNORE_PLUGINS'] = 'gsettings,networkmanager'
```

## When to Apply This Fix
- When running scripts in Cursor's integrated terminal
- When running scripts in other IDE terminals
- When experiencing "aborted (core dumped)" errors during pipeline initialization
- When the error occurs after "Pipeline state change is async, waiting..." message

## Testing
To verify the fix works:
1. Run the script in Cursor's integrated terminal
2. The script should start without segmentation faults
3. The pipeline should transition to PLAYING state successfully

## Notes
- This fix is environment-specific and doesn't affect functionality
- It only prevents libproxy from attempting proxy resolution
- The fix is safe to apply to all modules that use GStreamer/DeepStream
- No performance impact or side effects 