#!/usr/bin/env python3
"""
Minimal test to build TensorRT engine without preprocess step
"""
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp
import os
import sys

# Initialize GStreamer
Gst.init(None)

def test_engine_build():
    """Test TensorRT engine building without preprocess"""
    print("🔍 Testing TensorRT engine build without preprocess...")
    
    # Create minimal pipeline: source -> streammux -> nvinfer -> fakesink
    pipeline = Gst.Pipeline()
    
    # Create elements
    source = Gst.ElementFactory.make("nvurisrcbin", "source")
    streammux = Gst.ElementFactory.make("nvstreammux", "mux")
    nvinfer = Gst.ElementFactory.make("nvinfer", "infer")
    fakesink = Gst.ElementFactory.make("fakesink", "sink")
    
    if not all([source, streammux, nvinfer, fakesink]):
        print("❌ Failed to create pipeline elements")
        return False
    
    # Configure source
    source.set_property("uri", "rtsp://192.168.3.214:7447/jdr9oLlBkjyl3gDm?")
    
    # Configure streammux
    streammux.set_property("batch-size", 1)
    streammux.set_property("width", 1280)
    streammux.set_property("height", 720)
    streammux.set_property("batched-push-timeout", 4000000)
    streammux.set_property("live-source", 1)
    
    # Configure nvinfer
            nvinfer.set_property("config-file-path", "pipelines/config_infer_primary_yolo11.txt")
    nvinfer.set_property("input-tensor-meta", True)
    
    # Add elements to pipeline
    pipeline.add(source)
    pipeline.add(streammux)
    pipeline.add(nvinfer)
    pipeline.add(fakesink)
    
    # Link elements
    source.link(streammux)
    streammux.link(nvinfer)
    nvinfer.link(fakesink)
    
    print("✅ Pipeline created, attempting to build engine...")
    
    # Set pipeline to READY state (this should trigger engine building)
    ret = pipeline.set_state(Gst.State.READY)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("❌ Failed to set pipeline to READY state")
        return False
    
    print("✅ Pipeline set to READY state")
    
    # Set to PLAYING state
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("❌ Failed to set pipeline to PLAYING state")
        return False
    
    print("✅ Pipeline set to PLAYING state - engine should be building...")
    
    # Wait a bit for engine building
    import time
    time.sleep(10)
    
    # Check if engine was created
    engine_path = "models/engines/yolo11m.onnx_b1_gpu0_fp16.engine"
    if os.path.exists(engine_path):
        print(f"✅ Engine built successfully: {engine_path}")
        print(f"   Size: {os.path.getsize(engine_path) / (1024*1024):.1f} MB")
        return True
    else:
        print(f"❌ Engine not found at: {engine_path}")
        return False

if __name__ == "__main__":
    success = test_engine_build()
    sys.exit(0 if success else 1) 