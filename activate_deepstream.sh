#!/bin/bash

# DeepStream 7.1 Environment Setup
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

# DeepStream environment variables
export DEEPSTREAM_DIR=/opt/nvidia/deepstream/deepstream-7.1
export PATH=$DEEPSTREAM_DIR/bin:$PATH
export LD_LIBRARY_PATH=$DEEPSTREAM_DIR/lib:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=$DEEPSTREAM_DIR/lib/gst-plugins
export GST_PLUGIN_SCANNER=$DEEPSTREAM_DIR/lib/gst-plugins

# Additional GStreamer paths
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:/usr/lib/x86_64-linux-gnu/gstreamer-1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

# Python path for DeepStream bindings
export PYTHONPATH=$DEEPSTREAM_DIR/lib:$PYTHONPATH

echo "DeepStream environment activated!"
echo "CUDA: $(nvcc --version | head -1)"
echo "DeepStream: $(deepstream-app --version 2>/dev/null || echo 'Not found')"
echo "Python: $(python3 --version)"
echo "GStreamer: $(gst-launch-1.0 --version | head -1)" 