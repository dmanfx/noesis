% GStreamer and DeepStream Cache Clearing  
_Status: current as of 2026-02-02._

%% Overview
This document provides commands for clearing GStreamer and DeepStream caches when pipelines fail to detect plugins or after system updates.

%% When to Clear Cache
Clear the cache when experiencing:
- Pipeline fails to start with plugin detection errors
- GStreamer cannot find plugins that should be available
- After installing or updating GStreamer/DeepStream plugins
- After moving or updating plugin libraries
- Pipeline worked previously but suddenly stops working
- "Could not find plugin" or similar errors

%% GStreamer Plugin Registry Cache

%%% Standard Cache Clear
The most common cache to clear is the GStreamer plugin registry:

```bash
rm -f ${HOME}/.cache/gstreamer-1.0/registry.x86_64.bin
```

This removes the cached plugin registry. GStreamer will automatically rebuild it on the next pipeline run.

%%% Complete User Cache Clear
For a more thorough cleanup of all GStreamer user cache:

```bash
rm -rf ${HOME}/.cache/gstreamer-1.0/*
```

%%% System-Wide Cache Clear
If you have root access and need to clear system-wide cache:

```bash
sudo rm -f /root/.cache/gstreamer-1.0/registry.x86_64.bin
```

%% DeepStream and TensorRT Caches

%%% TensorRT Engine Cache
TensorRT engine files are typically cached in model directories. Check your DeepStream config files for `engine-file` paths. To force regeneration:

1. Locate engine files in your model directories (usually `.engine` or `.trt` files)
2. Delete the engine files to force TensorRT to rebuild them
3. The engine files will be regenerated on the next inference run

Example:
```bash
# Find and remove TensorRT engine files (adjust paths as needed)
find . -name "*.engine" -type f -delete
find . -name "*.trt" -type f -delete
```

%%% DeepStream Model Cache
DeepStream may cache model configurations. Check your config files for cache-related settings.

%% Verification

After clearing the cache, verify GStreamer can detect your plugins:

```bash
# List available GStreamer plugins
gst-inspect-1.0 | grep -i nv

# Check specific DeepStream plugins
gst-inspect-1.0 nvurisrcbin
gst-inspect-1.0 nvinfer
gst-inspect-1.0 nvtracker
```

%% Notes

- The registry cache is automatically regenerated on the next pipeline run
- No restart is required after clearing the cache
- Clearing cache does not affect installed plugins, only the discovery cache
- If issues persist after clearing cache, check plugin installation paths and environment variables
