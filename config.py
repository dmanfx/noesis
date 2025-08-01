"""
Configuration Management Module

This module defines the application's configuration structure using Python dataclasses.
It provides a centralized location for all configurable parameters and constants
used throughout the application, replacing global variables and hardcoded values.

The configuration is organized into logical groups (AppSettings, CameraSettings, etc.)
which helps with organization and makes it easier to add new configuration options.

Features:
- Type-hinted configuration parameters for better IDE support and validation
- Default values for all parameters to ensure the application can run with minimal setup
- Functions to load/save configuration from/to JSON files
- Path resolution to ensure file paths are properly handled
"""

import os
os.environ['no_proxy'] = '*'
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class AppConfig:
    """Application configuration root class that contains all configuration sections"""
    
    @dataclass
    class AppSettings:
        """General application settings for logging, versioning, and basic behavior"""
        APP_NAME: str = "YOLO Video Tracking"
        VERSION: str = "1.0.0"
        START_TIME: float = field(default_factory=time.time)
        LOG_LEVEL: int = 30  # WARNING level for production performance
        LOG_FILE: Optional[str] = "logs/app.log"
        DEBUG: bool = False  # Disable debug mode for performance
    
    @dataclass
    class CameraSettings:
        """Camera and video source settings that define input sources and their properties"""
        USE_WEBCAM: bool = False
        VIDEO_FILES: List[str] = field(default_factory=list)
        RTSP_STREAMS: List[Dict[str, Any]] = field(default_factory=lambda: [
            {
                "name": "Living Room Camera",
                "url": "rtsp://192.168.3.214:7447/jdr9oLlBkjyl3gDm?",
                "width": 1920,
                "height": 1080,
                "enabled": False
            },
            {
                "name": "Kitchen Camera", 
                "url": "rtsp://192.168.3.214:7447/qt3VqVdZpgG1B4Vk?",
                "width": 1920,
                "height": 1080,
                "enabled": True
            },
            {
                "name": "Family Room Camera",
                "url": "rtsp://192.168.3.214:7447/4qWTBhW6b4nLeUFE?",
                "width": 1280,
                "height": 720,
                "enabled": False
            }
        ])
        CAMERA_WIDTH: int = 1920
        CAMERA_HEIGHT: int = 1080
    
    @dataclass
    class ProcessingSettings:
        """Frame processing settings that control threading, performance, and pipeline behavior"""
        ENABLE_PROCESSING: bool = True
        MAX_QUEUE_SIZE: int = 20  # OPTIMIZED: Reduced from 30 for lower latency
        ANALYSIS_FRAME_INTERVAL: int = 1  # Process every frame (removed artificial limitation)
        TARGET_FPS: int = 30  # Realistic target for RTX 3060 (increased from 10)
        ENABLE_PROFILING: bool = False  # Enable profiling to identify CPU usage sources
        
       
        
        # GPU Frame Preprocessing Settings
        ENABLE_GPU_PREPROCESSING: bool = True  # FORCE GPU-accelerated frame preprocessing
        GPU_BATCH_SIZE: int = 2  # OPTIMIZED: Reduced from 4 to minimize GPU memory spikes
        GPU_PREPROCESSING_DEVICE: str = "cuda:0"  # GPU device for preprocessing
        
                      
        # Performance and Profiling Settings
        PROFILING_SAMPLING_RATE: int = 100  # OPTIMIZED: Profile every 100th frame (reduced overhead)
        
        # GPU Memory Optimization Settings (NEW for Phase 3.1.2)
        GPU_MEMORY_POOL_SIZE_MB: int = 500  # Pre-allocated GPU memory pool size
        ENABLE_MEMORY_POOLING: bool = True  # Use memory pooling for GPU operations
        
        # Thread Optimization Settings (NEW for Phase 3.1.2)
        USE_THREAD_AFFINITY: bool = True  # Pin threads to specific CPU cores
        
                
        # DeepStream Pipeline Configuration (RECOMMENDED)
        ENABLE_DEEPSTREAM: bool = True  # Enable DeepStream pipeline for video processing
        DEEPSTREAM_SOURCE_LATENCY: int = 50  # Reduced latency for real-time processing
        DEEPSTREAM_MUX_BATCH_SIZE: int = 1  # Single frame processing for lower latency
        DEEPSTREAM_MUX_SCALE_MODE: int = 2  # 0=stretch, 1=crop, 2=letter-box
        DEEPSTREAM_PREPROCESS_CONFIG: str = "pipelines/config_preproc.ini"  # Path to preprocessing config file
        DEEPSTREAM_TRACKER_CONFIG: str = "pipelines/tracker_nvdcf.yml"  # Path to tracker config file
        DEEPSTREAM_TRACKER_LIB: str = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"  # DeepStream tracker library
        DEEPSTREAM_ENABLE_OSD: bool = True  # Enable on-screen display for visualization
        
        # DEPRECATED: Unified GPU Pipeline Configuration (removed - DeepStream-only now)
        # USE_UNIFIED_GPU_PIPELINE: bool = True  # Enable unified GPU pipeline for optimal performance
        # UNIFIED_PIPELINE_THREADS: Optional[int] = None  # Auto-calculated if None
        
        def __post_init__(self):
            """Post-initialization to handle deprecated settings and warnings."""
            import warnings
            import logging
            
            logger = logging.getLogger("config.ProcessingSettings")
            
            # Validate DeepStream configuration
            if self.ENABLE_DEEPSTREAM:
                if not self.DEEPSTREAM_PREPROCESS_CONFIG:
                    logger.warning("⚠️  DEEPSTREAM_PREPROCESS_CONFIG is empty. Using default preprocessing.")
                
                if self.DEEPSTREAM_MUX_BATCH_SIZE <= 0:
                    logger.warning("⚠️  DEEPSTREAM_MUX_BATCH_SIZE <= 0. Will be auto-calculated from enabled streams.")
                
                logger.info("✅ DeepStream pipeline enabled - recommended for optimal performance.")
            else:
                logger.warning("⚠️  DeepStream pipeline disabled. Consider enabling for better performance.")
            
            # Memory optimization warnings
            if self.GPU_MEMORY_POOL_SIZE_MB > 1000:
                logger.warning(f"⚠️  Large GPU memory pool size ({self.GPU_MEMORY_POOL_SIZE_MB}MB) may exceed available GPU memory.")
            
            if self.GPU_BATCH_SIZE > 4:
                logger.warning(f"⚠️  Large GPU batch size ({self.GPU_BATCH_SIZE}) may cause GPU memory issues.")
            
            # Thread optimization info
            if self.USE_THREAD_AFFINITY:
                logger.info("✅ Thread affinity enabled for better performance.")
            
            if self.ENABLE_MEMORY_POOLING:
                logger.info("✅ GPU memory pooling enabled for better memory management.")
    
    @dataclass
    class ModelsSettings:
        """AI model settings for detection, tracking, and feature extraction"""
        MODEL_PATH: str = "models/yolo11m.onnx"  # Path to detection model
        POSE_MODEL_PATH: str = "models/yolo11m-pose.pt"  # Path to pose estimation model
        SEGMENTATION_MODEL_PATH: str = "models/yolo11m-seg.pt"  # Path to segmentation model
        REID_MODEL_PATH: str = "models/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"  # Path to ReID model
        MODEL_CONFIDENCE_THRESHOLD: float = 0.25  # Minimum confidence score for a detection
        MODEL_IOU_THRESHOLD: float = 0.45  # IoU threshold for NMS
        TARGET_CLASSES: List[int] = field(default_factory=lambda: [0,1])  # Empty list = all classes
        CLASS_NAMES: Dict[int, str] = field(default_factory=lambda: {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
            6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
            16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
            27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
            46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
            51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
            56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
            61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
            67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
            72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
            77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        })
        TRACE_PERSISTENCE_SECONDS: float = 3.0  # How long to keep inactive traces
        ENABLE_SEGMENTATION: bool = True  # Enable segmentation processing
        
        # TensorRT Optimization Settings
        ENABLE_TENSORRT: bool = True  # Enable TensorRT optimization for all models
        TENSORRT_FP16: bool = True  # Use FP16 precision
        TENSORRT_WORKSPACE_SIZE: int = 2          # OPTIMIZED: Reduced from 4GB to 2GB for memory efficiency
        TENSORRT_MAX_BATCH_SIZE: int = 4  # Maximum batch size for multi-stream support
        FORCE_GPU_ONLY: bool = True  # Never fallback to CPU inference
        DEVICE: str = "cuda:0"  # Fixed GPU device, no auto-detection
        
        # TensorRT Engine Paths (will be auto-generated)
        DETECTION_ENGINE_PATH: str = "models/engines/yolo11m_dynamic.engine"
        POSE_ENGINE_PATH: str = "models/engines/pose_fp16.engine"
        SEGMENTATION_ENGINE_PATH: str = "models/engines/segmentation_fp16.engine"
        REID_ENGINE_PATH: str = "models/engines/reid_fp16.engine"
        
        # Performance Settings
        WARM_UP_ITERATIONS: int = 10  # Number of warm-up iterations for TensorRT
        ENABLE_DYNAMIC_SHAPES: bool = False  # Use fixed input shapes for better performance
        OPTIMIZE_FOR_INFERENCE: bool = True  # Apply inference-specific optimizations
    
    @dataclass
    class VisualizationSettings:
        """Settings for visualization."""
        
        # Display settings
        SHOW_DETECTIONS: bool = True
        SHOW_TRACKING: bool = True
        SHOW_TRAILS: bool = True
        SHOW_FPS: bool = True
        SHOW_POSE: bool = True
        SHOW_SEGMENTATION: bool = False  # Disabled by default for performance
        
        # Missing attributes that are referenced in the code
        SHOW_DETECTION_BOXES: bool = True  # Enable detection box visualization
        SHOW_TRACKING_BOXES: bool = True   # Enable tracking box visualization
        SHOW_TRACES: bool = True           # Enable trace visualization
        SHOW_KEYPOINTS: bool = True        # Enable keypoint visualization
        SHOW_MASKS: bool = True            # Enable mask visualization
        
        # Visual style settings
        BOX_THICKNESS: int = 2
        TEXT_SCALE: float = 0.5
        TEXT_THICKNESS: int = 1
        TRACE_LENGTH: int = 30
        TRAIL_LENGTH: int = 200
        TRAIL_DRAW_SEGMENTS: int = 100
        TRAIL_VISUALIZATION_ENABLED: bool = True
        TRAIL_TIMEOUT_S: float = 10.0  # seconds to keep a disappeared track's trail
        TRAIL_DRAW_STRIDE: int = 2  # draw every Nth frame (≥1)
        TRAIL_SHOW_LABELS: bool = False
        
        # Additional visual style settings referenced in logs
        KEYPOINT_RADIUS: int = 3          # Radius for keypoint visualization
        TRACE_THICKNESS: int = 2          # Thickness for trace lines
        MASK_ALPHA: float = 0.5           # Alpha transparency for mask overlay
        
        # Annotation settings
        SHOW_LABELS: bool = True
        SHOW_CONFIDENCE: bool = True
        SHOW_TRACK_IDS: bool = True
        
        # GPU Visualization settings
        USE_GPU_VISUALIZATION: bool = False  # Enable GPU-accelerated visualization
        USE_NVENC: bool = False  # Enable NVENC hardware video encoding
        NVENC_CODEC: str = 'h264_nvenc'  # h264_nvenc or hevc_nvenc
        NVENC_PRESET: str = 'll'  # low-latency preset
        NVENC_BITRATE: int = 4000000  # 4 Mbps
        JPEG_QUALITY: int = 85  # JPEG encoding quality
        USE_NATIVE_DEEPSTREAM_OSD: bool = True  # If True, use DeepStream's native OSD, skip Python annotation
        
        def __post_init__(self):
            """Validate visualization settings."""
            # Validate visual parameters
            self.BOX_THICKNESS = max(1, self.BOX_THICKNESS)
            self.TEXT_SCALE = max(0.1, self.TEXT_SCALE)
            self.TEXT_THICKNESS = max(1, self.TEXT_THICKNESS)
            self.TRACE_LENGTH = max(1, self.TRACE_LENGTH)
            self.TRAIL_LENGTH = max(1, self.TRAIL_LENGTH)
            self.KEYPOINT_RADIUS = max(1, self.KEYPOINT_RADIUS)
            self.TRACE_THICKNESS = max(1, self.TRACE_THICKNESS)
            
            # Validate trail parameters
            self.TRAIL_TIMEOUT_S = max(0.1, self.TRAIL_TIMEOUT_S)
            self.TRAIL_DRAW_STRIDE = max(1, self.TRAIL_DRAW_STRIDE)
            
            # Validate encoding parameters
            self.NVENC_BITRATE = max(1000000, self.NVENC_BITRATE)  # Min 1 Mbps
            self.JPEG_QUALITY = max(1, min(100, self.JPEG_QUALITY))
    
    @dataclass
    class TrackingSettings:
        """Object tracking configuration that integrates with existing TrackingSystem"""
        
        # Enable/disable tracking
        ENABLE_TRACKING: bool = True
        
        # ByteTrack configuration parameters (matches tracking.py defaults)
        TRACK_THRESH: float = 0.25          # Low threshold to create tracks
        TRACK_BUFFER: int = 30              # Frames to keep track without detection  
        MATCH_THRESH: float = 0.75          # IoU threshold for matching
        # TrackingSystem configuration (matches tracking.py)
        INACTIVE_THRESHOLD_SECONDS: float = 1.0   # Time threshold to mark tracks as inactive
        TRACE_PERSISTENCE_SECONDS: float = 5.0    # Time to keep inactive traces
        USE_NATIVE_DEEPSTREAM_TRACKER: bool = True  # If True, use DeepStream's native tracker IDs, skip Python tracking
        
        def get_tracker_config(self) -> Dict[str, Any]:
            """Get configuration dict for TrackingSystem initialization"""
            return {
                "track_thresh": self.TRACK_THRESH,
                "track_buffer": self.TRACK_BUFFER,
                "match_thresh": self.MATCH_THRESH,
                "frame_rate": 30, # Default frame rate for tracking
            }
    
    @dataclass
    class OutputSettings:
        """Output settings for saving results to disk (frames, videos, detection data)#save #frame"""
        OUTPUT_DIR: str = "output"
        SAVE_FRAMES: bool = False
        SAVE_DETECTIONS: bool = False
        FRAME_SAVE_INTERVAL: int = 1  # Save every Nth frame
    
    @dataclass
    class WebSocketSettings:
        """WebSocket server settings for broadcasting results to clients"""
        HOST: str = "0.0.0.0"  # Listen on all network interfaces
        PORT: int = 6008
        MAX_CLIENTS: int = 10
        JPEG_QUALITY: int = 70  # JPEG quality for frame compression (0-100)
        MAX_FPS: int = 20  # Maximum FPS for WebSocket streaming
    
    # Initialize all configuration sections with default values
    app: AppSettings = field(default_factory=AppSettings)
    cameras: CameraSettings = field(default_factory=CameraSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    models: ModelsSettings = field(default_factory=ModelsSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    websocket: WebSocketSettings = field(default_factory=WebSocketSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    
    def get_camera_count(self) -> int:
        """Calculate total number of camera sources configured."""
        enabled_rtsp = 0
        if self.cameras.RTSP_STREAMS:
            enabled_rtsp = sum(1 for stream in self.cameras.RTSP_STREAMS if stream.get("enabled", True))
        
        return (
            enabled_rtsp + 
            (1 if self.cameras.USE_WEBCAM else 0) + 
            len(self.cameras.VIDEO_FILES)
        )
    
    # DEPRECATED: get_unified_pipeline_threads method removed
    # The system now uses DeepStream-only architecture
    # Threading is handled internally by DeepStream


# Create default configuration instance
config = AppConfig()


# Apply environment-based configuration overrides
if os.getenv("DEBUG", "0") == "1":
    config.app.DEBUG = True
    config.app.LOG_LEVEL = 10  # DEBUG

# Resolve paths relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure model paths are absolute for reliable loading
if not os.path.isabs(config.models.MODEL_PATH):
    config.models.MODEL_PATH = os.path.join(script_dir, config.models.MODEL_PATH)

if not os.path.isabs(config.models.POSE_MODEL_PATH):
    config.models.POSE_MODEL_PATH = os.path.join(script_dir, config.models.POSE_MODEL_PATH)

if not os.path.isabs(config.models.SEGMENTATION_MODEL_PATH):
    config.models.SEGMENTATION_MODEL_PATH = os.path.join(script_dir, config.models.SEGMENTATION_MODEL_PATH)

# Ensure output path is absolute
if not os.path.isabs(config.output.OUTPUT_DIR):
    config.output.OUTPUT_DIR = os.path.join(script_dir, config.output.OUTPUT_DIR)

# Ensure log path is absolute and directory exists
if config.app.LOG_FILE and not os.path.isabs(config.app.LOG_FILE):
    log_dir = os.path.join(script_dir, os.path.dirname(config.app.LOG_FILE))
    os.makedirs(log_dir, exist_ok=True)
    config.app.LOG_FILE = os.path.join(script_dir, config.app.LOG_FILE)


def load_config_from_file(config_file: str) -> AppConfig:
    """Load configuration from JSON file and merge with default configuration
    
    This function allows loading custom configuration from a JSON file, overriding
    the default values while preserving any settings not specified in the file.
    
    Args:
        config_file: Path to configuration JSON file
        
    Returns:
        AppConfig: Updated configuration with values from the file
    """
    import json
    import logging
    
    logger = logging.getLogger("config")
    
    if not os.path.exists(config_file):
        logger.warning(f"Configuration file not found: {config_file}")
        return config
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Update configuration sections and properties
        for section_name, section_data in config_data.items():
            if hasattr(config, section_name):
                section = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        # Special handling for RTSP_STREAMS legacy format conversion
                        if key == "RTSP_STREAMS":
                            if isinstance(value, dict):
                                # Convert legacy dictionary format to new list format
                                rtsp_list = []
                                for stream_name, stream_url in value.items():
                                    rtsp_list.append({
                                        "name": stream_name,
                                        "url": stream_url,
                                        "width": 1920,  # Default resolution
                                        "height": 1080
                                    })
                                setattr(section, key, rtsp_list)
                                logger.info(f"Converted legacy RTSP_STREAMS dict to new format: {len(rtsp_list)} streams")
                            elif isinstance(value, list):
                                # Check if it's old string list format
                                if value and isinstance(value[0], str):
                                    # Convert legacy string list to new dict format
                                    rtsp_list = []
                                    for i, stream_url in enumerate(value):
                                        rtsp_list.append({
                                            "name": f"Camera {i+1}",
                                            "url": stream_url,
                                            "width": 1920,  # Default resolution
                                            "height": 1080
                                        })
                                    setattr(section, key, rtsp_list)
                                    logger.info(f"Converted legacy RTSP_STREAMS string list to new format: {len(rtsp_list)} streams")
                                else:
                                    # Already in new format
                                    setattr(section, key, value)
                            else:
                                setattr(section, key, value)
                        else:
                            setattr(section, key, value)
                    else:
                        logger.warning(f"Unknown configuration key: {section_name}.{key}")
            else:
                logger.warning(f"Unknown configuration section: {section_name}")
        
        logger.info(f"Loaded configuration from {config_file}")
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration from {config_file}: {e}")
        return config


def save_config_to_file(config: AppConfig, config_file: str) -> bool:
    """Save current configuration to a JSON file
    
    This function serializes the current configuration to a JSON file,
    which can later be loaded using load_config_from_file.
    
    Args:
        config: Configuration to save
        config_file: Path to output configuration file
        
    Returns:
        bool: True if successful, False otherwise
    """
    import json
    import logging
    
    logger = logging.getLogger("config")
    
    try:
        # Create configuration data structure for serialization
        config_data = {}
        
        for section_name in ["app", "cameras", "processing", "models", "visualization", "output", "websocket", "tracking"]:
            section = getattr(config, section_name)
            section_data = {}
            
            for key in section.__annotations__:
                value = getattr(section, key)
                
                # Convert non-serializable types (like tuples) to serializable ones (like lists)
                if isinstance(value, tuple):
                    value = list(value)
                
                section_data[key] = value
            
            config_data[section_name] = section_data
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        # Save configuration to file
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        logger.info(f"Saved configuration to {config_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving configuration to {config_file}: {e}")
        return False


# DEPRECATED: Unified GPU pipeline validation functions removed
# The system now uses DeepStream-only architecture
# These functions were part of a transitional unified pipeline concept
# that has been superseded by the DeepStream implementation
