#!/usr/bin/env python3
"""
DeepStream Inference Bridge

This module connects DeepStream's preprocessing output to TensorRT inference
and feeds results back through the pipeline with optional tracking.

Pipeline flow:
    appsink (tensor) -> TensorRT (YOLO/Seg/Pose) -> metadata -> appsrc -> tracker -> ...
"""

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GLib, GstApp, GObject

import sys
import time
import logging
import threading
import queue
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch

# Initialize GStreamer
Gst.init(None)

# DeepStream Python bindings
sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import pyds

from config import AppConfig
from tensorrt_inference import GPUOnlyDetectionManager, TensorRTModelManager
from detection import Detection
from tracking import TrackingSystem
from models import AnalysisFrame, convert_detection_to_detection_result


class DeepStreamInferenceBridge:
    """
    Bridge between DeepStream preprocessing and TensorRT inference.
    
    This class:
    1. Receives preprocessed tensors from appsink
    2. Runs TensorRT inference (detection, segmentation, pose)
    3. Converts results to DeepStream metadata
    4. Optionally applies tracking via DeepStream tracker
    """
    
    def __init__(
        self,
        config: AppConfig,
        output_queue: Optional[queue.Queue] = None
    ):
        """
        Initialize inference bridge.
        
        Args:
            config: Application configuration
            output_queue: Queue for processed analysis frames
        """
        self.config = config
        self.output_queue = output_queue
        
        self.logger = logging.getLogger("DeepStreamInferenceBridge")
        
        # Device configuration
        self.device = torch.device(config.models.DEVICE)
        self.device_id = int(config.models.DEVICE.split(':')[-1]) if ':' in config.models.DEVICE else 0
        
        # Initialize TensorRT models
        self.model_manager = TensorRTModelManager.get_shared(config)
        self.gpu_detector = GPUOnlyDetectionManager(
            config=config,
            model_manager=self.model_manager
        )
        
        # Initialize tracking (Python-based for now)
        if config.tracking.ENABLE_TRACKING:
            self.tracking_system = TrackingSystem(
                tracker_config=config.tracking.get_tracker_config(),
                inactive_threshold_seconds=config.tracking.INACTIVE_THRESHOLD_SECONDS,
                trace_persistence_seconds=config.tracking.TRACE_PERSISTENCE_SECONDS
            )
        else:
            self.tracking_system = None
        
        # Pipeline components
        self.pipeline = None
        self.appsrc = None
        self.tracker = None
        self.sink = None
        
        # Performance tracking
        self.frame_count = 0
        self.inference_time_total = 0
        
        self.logger.info("DeepStream Inference Bridge initialized")
    
    def create_inference_pipeline(self, width: int, height: int, fps: int = 30) -> Gst.Element:
        """
        Create the inference part of the pipeline.
        
        Args:
            width: Frame width
            height: Frame height
            fps: Frame rate
            
        Returns:
            Pipeline element that can be linked after preprocessing
        """
        # Create bin for inference components
        inference_bin = Gst.Bin.new("inference-bin")
        
        # Create appsrc for injecting detection results
        self.appsrc = Gst.ElementFactory.make("appsrc", "detection_src")
        if not self.appsrc:
            raise RuntimeError("Failed to create appsrc")
        
        # Configure appsrc
        caps = Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM),format=NV12,width={width},height={height},framerate={fps}/1"
        )
        self.appsrc.set_property("caps", caps)
        self.appsrc.set_property("is-live", True)
        self.appsrc.set_property("block", True)
        self.appsrc.set_property("format", Gst.Format.TIME)
        
        inference_bin.add(self.appsrc)
        
        # Create tracker if enabled
        if self.config.processing.DEEPSTREAM_TRACKER_LIB:
            self.tracker = Gst.ElementFactory.make("nvtracker", "tracker")
            if not self.tracker:
                self.logger.warning("Failed to create nvtracker")
            else:
                # Set tracker properties
                self.tracker.set_property("gpu-id", self.device_id)
                self.tracker.set_property("ll-lib-file", self.config.processing.DEEPSTREAM_TRACKER_LIB)
                
                # Set config file if provided
                if self.config.processing.DEEPSTREAM_TRACKER_CONFIG:
                    import os
                    config_path = os.path.join(
                        os.path.dirname(__file__), 
                        self.config.processing.DEEPSTREAM_TRACKER_CONFIG
                    )
                    self.tracker.set_property("ll-config-file", config_path)
                
                inference_bin.add(self.tracker)
                
                # Link appsrc to tracker
                if not self.appsrc.link(self.tracker):
                    raise RuntimeError("Failed to link appsrc to tracker")
        
        # Create OSD if enabled
        last_element = self.tracker if self.tracker else self.appsrc
        
        if self.config.processing.DEEPSTREAM_ENABLE_OSD:
            osd = Gst.ElementFactory.make("nvdsosd", "osd")
            if osd:
                osd.set_property("gpu-id", self.device_id)
                osd.set_property("process-mode", 0)  # CPU mode
                osd.set_property("display-text", 1)
                osd.set_property("display-bbox", 1)
                osd.set_property("display-mask", 0)
                
                inference_bin.add(osd)
                
                if not last_element.link(osd):
                    raise RuntimeError("Failed to link to OSD")
                    
                last_element = osd
        
        # Create ghost pads
        # Input pad (from preprocessing) – expose the bin's sink as the sink pad of appsrc
        sink_internal = self.appsrc.get_static_pad("sink")
        if not sink_internal:
            raise RuntimeError("appsrc has no sink pad – cannot create ghost sink pad")
        sink_pad = Gst.GhostPad.new("sink", sink_internal)
        inference_bin.add_pad(sink_pad)
        
        # Output pad (to encoder/sink) - connects to the last element's src pad
        src_pad = Gst.GhostPad.new("src", last_element.get_static_pad("src"))
        inference_bin.add_pad(src_pad)
        
        self.logger.info("Created inference pipeline components")
        return inference_bin
    
    def process_tensor(self, tensor: torch.Tensor, metadata: Dict[str, Any]) -> Gst.Buffer:
        """
        Process tensor through TensorRT and create output buffer.
        
        Args:
            tensor: Preprocessed GPU tensor
            metadata: Frame metadata (source_id, frame_num, etc.)
            
        Returns:
            GStreamer buffer with detection metadata
        """
        start_time = time.time()
        
        # Run TensorRT inference
        detections, inference_ms = self.gpu_detector.process_tensor(tensor)
        
        # Create Detection objects if not already
        detection_objects = []
        for det in detections:
            if isinstance(det, Detection):
                detection_objects.append(det)
            else:
                # Convert dict to Detection object
                detection_objects.append(Detection(
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    class_id=det['class_id'],
                    class_name=det.get('class_name', f"class_{det['class_id']}")
                ))
        
        # Apply Python tracking if enabled and no DeepStream tracker
        tracks = []
        if self.tracking_system and not self.tracker:
            # Need frame for tracking - convert tensor to numpy
            frame_np = self._tensor_to_numpy(tensor)
            tracks = self.tracking_system.update(
                detection_objects, 
                frame_np,
                f"camera_{metadata.get('source_id', 0)}"
            )
        
        # Create GStreamer buffer
        buffer = Gst.Buffer.new()
        
        # Set timestamp
        buffer.pts = metadata.get('timestamp', 0) * Gst.SECOND
        buffer.dts = buffer.pts
        
        # Add DeepStream metadata
        if pyds:
            self._add_deepstream_metadata(buffer, detection_objects, metadata)
        
        # Track performance
        self.frame_count += 1
        self.inference_time_total += (time.time() - start_time)
        
        # Send to output queue if provided
        if self.output_queue:
            # Convert tensor to numpy for visualization
            frame_np = self._tensor_to_numpy(tensor)
            
            # Convert detections to DetectionResult
            detection_results = []
            for i, det in enumerate(detection_objects):
                detection_results.append(
                    convert_detection_to_detection_result(det, detection_id=i)
                )
            
            # Create analysis frame
            analysis_frame = AnalysisFrame(
                frame_id=metadata.get('frame_num', self.frame_count),
                camera_id=f"camera_{metadata.get('source_id', 0)}",
                timestamp=metadata.get('timestamp', time.time()),
                frame=frame_np,
                detections=detection_results,
                tracks=tracks,
                frame_width=tensor.shape[-1],  # Width
                frame_height=tensor.shape[-2],  # Height
                processing_time=inference_ms
            )
            
            try:
                self.output_queue.put(analysis_frame, block=False)
            except queue.Full:
                self.logger.debug("Output queue full")
        
        return buffer
    
    def _add_deepstream_metadata(
        self, 
        buffer: Gst.Buffer, 
        detections: List[Detection],
        metadata: Dict[str, Any]
    ):
        """Add DeepStream metadata to buffer."""
        # Get batch metadata
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if not batch_meta:
            # Create new batch metadata
            batch_meta = pyds.nvds_create_batch_meta(1)
            pyds.gst_buffer_add_nvds_meta(hash(buffer), batch_meta, None)
        
        # Add frame metadata
        frame_meta = pyds.nvds_acquire_frame_meta_from_pool(batch_meta)
        frame_meta.frame_num = metadata.get('frame_num', self.frame_count)
        frame_meta.source_id = metadata.get('source_id', 0)
        frame_meta.batch_id = 0
        
        # Add object metadata for each detection
        for det in detections:
            obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
            
            # Set bounding box
            obj_meta.rect_params.left = float(det.bbox[0])
            obj_meta.rect_params.top = float(det.bbox[1])
            obj_meta.rect_params.width = float(det.bbox[2] - det.bbox[0])
            obj_meta.rect_params.height = float(det.bbox[3] - det.bbox[1])
            
            # Set class info
            obj_meta.class_id = det.class_id
            obj_meta.confidence = det.confidence
            obj_meta.obj_label = det.class_name
            
            # Set unique object ID (for tracking)
            obj_meta.object_id = pyds.UNTRACKED_OBJECT_ID
            
            # Add to frame
            pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)
        
        # Add frame to batch
        pyds.nvds_add_frame_meta_to_batch(batch_meta, frame_meta)
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert GPU tensor to numpy array."""
        # Handle different tensor formats
        if tensor.dim() == 4:
            # Batch dimension - take first
            tensor = tensor[0]
        
        # Ensure CHW format
        if tensor.shape[0] != 3:
            self.logger.warning(f"Unexpected tensor shape: {tensor.shape}")
        
        # Convert to CPU and numpy
        tensor_cpu = tensor.cpu()
        
        # Convert from CHW to HWC
        if tensor_cpu.shape[0] == 3:
            tensor_cpu = tensor_cpu.permute(1, 2, 0)
        
        # Convert to uint8
        if tensor_cpu.dtype in [torch.float16, torch.float32]:
            tensor_cpu = (tensor_cpu * 255).clamp(0, 255).to(torch.uint8)
        
        # Convert to numpy
        frame_np = tensor_cpu.numpy()
        
        # Convert RGB to BGR for OpenCV
        if frame_np.shape[2] == 3:
            frame_np = frame_np[:, :, [2, 1, 0]]
        
        return frame_np
    
    def push_buffer(self, buffer: Gst.Buffer) -> bool:
        """Push buffer to appsrc."""
        if not self.appsrc:
            return False
        
        # Push buffer
        ret = self.appsrc.emit("push-buffer", buffer)
        return ret == Gst.FlowReturn.OK
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        avg_time = self.inference_time_total / self.frame_count if self.frame_count > 0 else 0
        
        return {
            'frames_processed': self.frame_count,
            'average_inference_ms': avg_time * 1000,
            'model_manager': 'TensorRT',
            'tracking_enabled': self.tracking_system is not None,
            'deepstream_tracker': self.tracker is not None
        }


def connect_inference_to_pipeline(
    pipeline: Gst.Pipeline,
    preprocess_element: Gst.Element,
    config: AppConfig,
    output_queue: Optional[queue.Queue] = None
) -> DeepStreamInferenceBridge:
    """
    Connect inference bridge to existing pipeline.
    
    Args:
        pipeline: GStreamer pipeline
        preprocess_element: Preprocessing element to connect after
        config: Application configuration
        output_queue: Optional output queue
        
    Returns:
        Configured inference bridge
    """
    # Create inference bridge
    bridge = DeepStreamInferenceBridge(config, output_queue)
    
    # Get dimensions from preprocess element
    # This is a simplified version - in practice you'd query caps
    width = 640
    height = 640
    fps = 30
    
    # Create inference pipeline
    inference_bin = bridge.create_inference_pipeline(width, height, fps)
    
    # Add to main pipeline
    pipeline.add(inference_bin)
    
    # Link preprocessing to inference
    if not preprocess_element.link(inference_bin):
        raise RuntimeError("Failed to link preprocessing to inference")
    
    return bridge 