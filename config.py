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
        PERFORMANCE_MODE: bool = True  # Enable performance optimizations
        ENABLE_DEBUG_LOGGING: bool = False  # Explicit debug control
    
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
        FRAME_RATE: int = 30
    
    @dataclass
    class ProcessingSettings:
        """Frame processing settings that control threading, performance, and pipeline behavior"""
        ENABLE_PROCESSING: bool = True
        ENABLE_THREADING: bool = True
        ENABLE_MULTIPROCESSING: bool = False  # Keep disabled to avoid process explosion
        MAX_QUEUE_SIZE: int = 20  # OPTIMIZED: Reduced from 30 for lower latency
        ANALYSIS_FRAME_INTERVAL: int = 1  # Process every frame (removed artificial limitation)
        FRAME_SKIP: int = 0  # No artificial frame skipping (removed limitation)
        AUTO_FRAME_SKIP: bool = True  # Automatically adjust frame skip based on processing performance
        TARGET_FPS: int = 30  # Realistic target for RTX 3060 (increased from 10)
        ENABLE_PROFILING: bool = False  # Enable profiling to identify CPU usage sources
        
        # Hardware Video Decoding Settings (DEPRECATED - Use DeepStream instead)
        ENABLE_NVDEC: bool = False  # DEPRECATED: Use ENABLE_DEEPSTREAM instead
        NVDEC_FALLBACK_TO_CPU: bool = False  # DEPRECATED: DeepStream handles GPU-only decoding
        NVDEC_BUFFER_SIZE: int = 5  # DEPRECATED: DeepStream manages buffer sizes automatically
        
        # GPU Frame Preprocessing Settings
        ENABLE_GPU_PREPROCESSING: bool = True  # FORCE GPU-accelerated frame preprocessing
        GPU_BATCH_SIZE: int = 2  # OPTIMIZED: Reduced from 4 to minimize GPU memory spikes
        GPU_PREPROCESSING_DEVICE: str = "cuda:0"  # GPU device for preprocessing
        
        # CPU Preprocessing Settings (DEPRECATED - GPU-only operation)
        ENABLE_OPTIMIZED_PREPROCESSING: bool = False  # DEPRECATED: Use GPU preprocessing only
        PREPROCESSING_THREADS: int = 2  # DEPRECATED: Unused in GPU-only mode
        PREPROCESSING_ALGORITHM: str = "INTER_LINEAR"  # DEPRECATED: Unused in GPU-only mode
        
        # Unified GPU Pipeline Settings (DEPRECATED - Use DeepStream directly)
        USE_UNIFIED_GPU_PIPELINE: bool = False  # DEPRECATED: Set to False to use DeepStream directly
        UNIFIED_PIPELINE_THREADS: Optional[int] = None  # Auto-calculated based on camera count (None = auto)
        PIPELINE_QUEUE_TIMEOUT_MS: float = 5.0  # OPTIMIZED: Reduced from 10.0 for quicker response
        ENABLE_PIPELINE_VALIDATION: bool = True  # Validate pipeline configuration on startup
        
        # Performance and Profiling Settings
        PROFILING_SAMPLING_RATE: int = 100  # OPTIMIZED: Profile every 100th frame (reduced overhead)
        MEMORY_MONITORING_ENABLED: bool = True  # Enable GPU memory monitoring
        PERFORMANCE_ALERTS_ENABLED: bool = True  # Enable performance degradation alerts
        
        # GPU Memory Optimization Settings (NEW for Phase 3.1.2)
        GPU_MEMORY_POOL_SIZE_MB: int = 500  # Pre-allocated GPU memory pool size
        GPU_MEMORY_DEFRAG_INTERVAL: int = 1000  # Defragment memory every N frames
        ENABLE_MEMORY_POOLING: bool = True  # Use memory pooling for GPU operations
        
        # Thread Optimization Settings (NEW for Phase 3.1.2)
        USE_THREAD_AFFINITY: bool = True  # Pin threads to specific CPU cores
        THREAD_PRIORITY: str = "HIGH"  # Thread priority: "NORMAL", "HIGH", "REALTIME"
        DECODER_THREAD_PRIORITY: str = "REALTIME"  # NVDEC decoder thread priority
        
        # Legacy Settings (DEPRECATED - Remove in future versions)
        USE_LEGACY_NVDEC_READER: bool = False  # DEPRECATED: Use DeepStream pipeline instead
        
        # PyNvVideoCodec GPU Reader Settings (DEPRECATED)
        GPU_READER_QUEUE_SIZE: int = 30  # DEPRECATED: Use DeepStream queue management
        GPU_READER_MAX_CONSECUTIVE_FAILURES: int = 10  # DEPRECATED: Use DeepStream error handling
        
        # DeepStream Pipeline Configuration (RECOMMENDED)
        ENABLE_DEEPSTREAM: bool = True  # Enable DeepStream pipeline for video processing
        DEEPSTREAM_SOURCE_LATENCY: int = 50  # Reduced latency for real-time processing
        DEEPSTREAM_MUX_BATCH_SIZE: int = 1  # Single frame processing for lower latency
        DEEPSTREAM_MUX_SCALE_MODE: int = 2  # 0=stretch, 1=crop, 2=letter-box
        DEEPSTREAM_PREPROCESS_CONFIG: str = "config_preproc.txt"  # Path to preprocessing config file
        DEEPSTREAM_TRACKER_CONFIG: str = "config_tracker_nvdcf_batch.yml"  # Path to batch tracker config file
        DEEPSTREAM_TRACKER_LIB: str = ""  # Empty string uses NvDCF, or path to custom tracker lib like "libbytetrack_ds.so"
        DEEPSTREAM_ENABLE_OSD: bool = True  # Enable on-screen display for visualization
        
        def __post_init__(self):
            """Post-initialization to handle deprecated settings and warnings."""
            import warnings
            import logging
            
            logger = logging.getLogger("config.ProcessingSettings")
            
            # Check for deprecated NVDEC settings
            if self.ENABLE_NVDEC:
                warnings.warn(
                    "ENABLE_NVDEC is deprecated. Use ENABLE_DEEPSTREAM=True instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
                logger.warning("⚠️  ENABLE_NVDEC is deprecated. Use ENABLE_DEEPSTREAM=True instead.")
            
            if self.NVDEC_FALLBACK_TO_CPU:
                warnings.warn(
                    "NVDEC_FALLBACK_TO_CPU is deprecated. DeepStream handles GPU-only decoding automatically.",
                    DeprecationWarning,
                    stacklevel=2
                )
                logger.warning("⚠️  NVDEC_FALLBACK_TO_CPU is deprecated. DeepStream handles GPU-only decoding automatically.")
            
            if self.USE_LEGACY_NVDEC_READER:
                warnings.warn(
                    "USE_LEGACY_NVDEC_READER is deprecated. Use DeepStream pipeline instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
                logger.warning("⚠️  USE_LEGACY_NVDEC_READER is deprecated. Use DeepStream pipeline instead.")
            
            if self.ENABLE_OPTIMIZED_PREPROCESSING:
                warnings.warn(
                    "ENABLE_OPTIMIZED_PREPROCESSING is deprecated. GPU preprocessing is now the default.",
                    DeprecationWarning,
                    stacklevel=2
                )
                logger.warning("⚠️  ENABLE_OPTIMIZED_PREPROCESSING is deprecated. GPU preprocessing is now the default.")
            
            # Validate DeepStream configuration
            if self.ENABLE_DEEPSTREAM:
                if not self.DEEPSTREAM_PREPROCESS_CONFIG:
                    logger.warning("⚠️  DEEPSTREAM_PREPROCESS_CONFIG is empty. Using default preprocessing.")
                
                if self.DEEPSTREAM_MUX_BATCH_SIZE <= 0:
                    logger.warning("⚠️  DEEPSTREAM_MUX_BATCH_SIZE <= 0. Will be auto-calculated from enabled streams.")
                
                logger.info("✅ DeepStream pipeline enabled - recommended for optimal performance.")
            else:
                logger.warning("⚠️  DeepStream pipeline disabled. Consider enabling for better performance.")
            
            # Validate legacy settings conflicts
            if self.ENABLE_NVDEC and self.ENABLE_DEEPSTREAM:
                logger.warning("⚠️  Both ENABLE_NVDEC and ENABLE_DEEPSTREAM are enabled. DeepStream will take precedence.")
            
            if not self.ENABLE_NVDEC and not self.ENABLE_DEEPSTREAM:
                logger.error("❌ Neither ENABLE_NVDEC nor ENABLE_DEEPSTREAM is enabled. At least one must be enabled for video processing.")
            
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
        TRAIL_LENGTH: int = 50
        
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
        FRAME_RATE: int = 30                # Assumed frame rate for tracking
        
        # TrackingSystem configuration (matches tracking.py)
        INACTIVE_THRESHOLD_SECONDS: float = 1.0   # Time threshold to mark tracks as inactive
        TRACE_PERSISTENCE_SECONDS: float = 5.0    # Time to keep inactive traces
        USE_NATIVE_DEEPSTREAM_TRACKER: bool = False  # If True, use DeepStream's native tracker IDs, skip Python tracking
        
        def get_tracker_config(self) -> Dict[str, Any]:
            """Get configuration dict for TrackingSystem initialization"""
            return {
                "track_thresh": self.TRACK_THRESH,
                "track_buffer": self.TRACK_BUFFER,
                "match_thresh": self.MATCH_THRESH,
                "frame_rate": self.FRAME_RATE,
            }
    
    @dataclass
    class OutputSettings:
        """Output settings for saving results to disk (frames, videos, detection data)#save #frame"""
        OUTPUT_DIR: str = "output"
        SAVE_FRAMES: bool = False
        SAVE_VIDEO: bool = False
        SAVE_DETECTIONS: bool = False
        FRAME_SAVE_INTERVAL: int = 1  # Save every Nth frame
        VIDEO_FPS: int = 15
        VIDEO_CODEC: str = "mp4v"
        JPEG_QUALITY: int = 90
    
    @dataclass
    class WebSocketSettings:
        """WebSocket server settings for broadcasting results to clients"""
        ENABLE_SERVER: bool = True
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
    
    def get_unified_pipeline_threads(self) -> int:
        """
        Get the number of threads for unified pipeline processing.
        
        Returns:
            Number of threads needed (auto-calculated if not explicitly set)
        """
        if self.processing.UNIFIED_PIPELINE_THREADS is not None:
            return self.processing.UNIFIED_PIPELINE_THREADS
        
        # Auto-calculate: one thread per camera source
        camera_count = self.get_camera_count()
        return max(1, camera_count)  # At least 1 thread even if no cameras configured


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


def validate_unified_pipeline_config(config: AppConfig) -> Dict[str, List[str]]:
    """
    Validate configuration for unified GPU pipeline compatibility.
    Enhanced with Phase 3.1.1 strict GPU-only enforcement.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Dict with 'errors', 'warnings', and 'info' lists
    """
    import logging
    
    logger = logging.getLogger("config_validation")
    
    validation_results = {
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # Critical requirements for unified GPU pipeline
    if config.processing.USE_UNIFIED_GPU_PIPELINE:
        
        # GPU-only requirements - STRICT ENFORCEMENT
        if not config.models.FORCE_GPU_ONLY:
            validation_results['errors'].append(
                "Unified GPU pipeline requires FORCE_GPU_ONLY=True"
            )
        
        if not config.models.ENABLE_TENSORRT:
            validation_results['errors'].append(
                "Unified GPU pipeline requires ENABLE_TENSORRT=True"
            )
        
        # Check for video decoding capability (DeepStream or NVDEC)
        if not config.processing.ENABLE_DEEPSTREAM and not config.processing.ENABLE_NVDEC:
            validation_results['errors'].append(
                "Unified GPU pipeline requires either ENABLE_DEEPSTREAM=True or ENABLE_NVDEC=True for video decoding"
            )
        
        if config.processing.NVDEC_FALLBACK_TO_CPU:
            validation_results['errors'].append(
                "NVDEC_FALLBACK_TO_CPU must be False for strict GPU-only operation"
            )
        
        if not config.processing.ENABLE_GPU_PREPROCESSING:
            validation_results['errors'].append(
                "Unified GPU pipeline requires ENABLE_GPU_PREPROCESSING=True"
            )
        
        # Incompatible settings that must be disabled
        
        if config.processing.ENABLE_MULTIPROCESSING:
            validation_results['errors'].append(
                "ENABLE_MULTIPROCESSING must be False when using unified pipeline"
            )
        
        if config.processing.ENABLE_OPTIMIZED_PREPROCESSING:
            validation_results['warnings'].append(
                "ENABLE_OPTIMIZED_PREPROCESSING is ignored in GPU-only mode"
            )
        
        # Memory and performance settings validation
        if config.processing.MAX_QUEUE_SIZE > 30:
            validation_results['warnings'].append(
                f"Large queue size ({config.processing.MAX_QUEUE_SIZE}) may increase latency and memory usage"
            )
        
        if config.processing.GPU_BATCH_SIZE > 4:
            validation_results['warnings'].append(
                f"Large GPU batch size ({config.processing.GPU_BATCH_SIZE}) may exceed GPU memory"
            )
        
        if config.processing.NVDEC_BUFFER_SIZE > 10:
            validation_results['warnings'].append(
                f"Large NVDEC buffer ({config.processing.NVDEC_BUFFER_SIZE}) may increase memory usage"
            )
        
        # TensorRT configuration validation
        if not config.models.TENSORRT_FP16:
            validation_results['warnings'].append(
                "TENSORRT_FP16=True recommended for optimal performance and memory efficiency"
            )
        
        if config.models.TENSORRT_WORKSPACE_SIZE > 4:
            validation_results['warnings'].append(
                f"Large TensorRT workspace ({config.models.TENSORRT_WORKSPACE_SIZE}GB) may reduce available GPU memory"
            )
        
        # Thread configuration validation
        num_cameras = config.get_camera_count()
        required_threads = config.get_unified_pipeline_threads()
        
        # Only validate if threads are explicitly set (not auto-calculated)
        if config.processing.UNIFIED_PIPELINE_THREADS is not None:
            if config.processing.UNIFIED_PIPELINE_THREADS < num_cameras:
                validation_results['errors'].append(
                    f"UNIFIED_PIPELINE_THREADS ({config.processing.UNIFIED_PIPELINE_THREADS}) "
                    f"must be >= number of cameras ({num_cameras})"
                )
        else:
            # Auto-calculation is being used
            validation_results['info'].append(
                f"✅ Auto-calculated pipeline threads: {required_threads} (for {num_cameras} cameras)"
            )
        
        # Performance optimization validation
        if config.processing.ENABLE_PROFILING and config.processing.PROFILING_SAMPLING_RATE < 50:
            validation_results['warnings'].append(
                f"Low profiling sampling rate ({config.processing.PROFILING_SAMPLING_RATE}) may impact performance"
            )
        
        # Tracking configuration validation
        if config.tracking.ENABLE_TRACKING:
            if config.tracking.TRACK_THRESH <= 0 or config.tracking.TRACK_THRESH > 1:
                validation_results['errors'].append(
                    f"TRACK_THRESH ({config.tracking.TRACK_THRESH}) must be between 0 and 1"
                )
            
            if config.tracking.TRACK_BUFFER <= 0:
                validation_results['errors'].append(
                    f"TRACK_BUFFER ({config.tracking.TRACK_BUFFER}) must be positive"
                )
            
            if config.tracking.MATCH_THRESH <= 0 or config.tracking.MATCH_THRESH > 1:
                validation_results['errors'].append(
                    f"MATCH_THRESH ({config.tracking.MATCH_THRESH}) must be between 0 and 1"
                )
            
            if config.tracking.INACTIVE_THRESHOLD_SECONDS <= 0:
                validation_results['errors'].append(
                    f"INACTIVE_THRESHOLD_SECONDS ({config.tracking.INACTIVE_THRESHOLD_SECONDS}) must be positive"
                )
        
        # Information messages
        validation_results['info'].append(
            f"✅ Unified GPU pipeline configured for {num_cameras} cameras with {required_threads} threads"
        )
        validation_results['info'].append(
            f"✅ TensorRT FP16: {config.models.TENSORRT_FP16}"
        )
        validation_results['info'].append(
            f"✅ GPU device: {config.models.DEVICE}"
        )
        validation_results['info'].append(
            f"✅ Memory pooling: {'Enabled' if config.processing.ENABLE_MEMORY_POOLING else 'Disabled'}"
        )
        validation_results['info'].append(
            f"✅ Tracking: {'Enabled' if config.tracking.ENABLE_TRACKING else 'Disabled'}"
        )
        
        # Check for deprecated/obsolete settings
        obsolete_settings = []
        if hasattr(config.processing, 'PREPROCESSING_THREADS'):
            obsolete_settings.append("PREPROCESSING_THREADS (unused in GPU-only mode)")
        if hasattr(config.processing, 'PREPROCESSING_ALGORITHM'):
            obsolete_settings.append("PREPROCESSING_ALGORITHM (unused in GPU-only mode)")
        
        if obsolete_settings:
            validation_results['info'].append(
                f"ℹ️  Obsolete settings can be removed: {', '.join(obsolete_settings)}"
            )
        
    else:
        # Legacy pipeline validation
        validation_results['warnings'].append("⚠️  Using legacy pipeline - consider migrating to unified GPU pipeline")
        
        if config.models.FORCE_GPU_ONLY:
            validation_results['errors'].append(
                "FORCE_GPU_ONLY=True is incompatible with legacy pipeline"
            )
        
        if config.processing.ENABLE_MEMORY_POOLING:
            validation_results['warnings'].append(
                "Memory pooling is only effective with unified GPU pipeline"
            )
    
    # Log validation results
    for error in validation_results['errors']:
        logger.error(f"❌ Configuration Error: {error}")
    
    for warning in validation_results['warnings']:
        logger.warning(f"⚠️  Configuration Warning: {warning}")
    
    for info in validation_results['info']:
        logger.info(f"ℹ️  Configuration Info: {info}")
    
    return validation_results


def get_pipeline_migration_guide() -> str:
    """
    Get migration guide for transitioning to unified GPU pipeline.
    
    Returns:
        String with migration instructions
    """
    guide = """
    📋 Unified GPU Pipeline Migration Guide
    
    Required Configuration Changes:
    
    1. Processing Settings:
       - USE_UNIFIED_GPU_PIPELINE = True
       - FORCE_GPU_ONLY = True
       - ENABLE_TENSORRT = True
       - ENABLE_GPU_PREPROCESSING = True
       - ENABLE_MULTIPROCESSING = False (recommended)
    
    2. Hardware Requirements:
       - NVIDIA GPU with CUDA support
       - NVDEC support (recommended)
       - Sufficient GPU memory for all models
    
    3. Performance Settings:
       - TENSORRT_FP16 = True (recommended)
       - NVDEC_FALLBACK_TO_CPU = False
       - Adjust UNIFIED_PIPELINE_THREADS based on camera count
    
    4. Validation:
       - Run validate_unified_pipeline_config() before deployment
       - Test with all cameras to ensure GPU memory sufficiency
       - Monitor performance metrics during initial deployment
    
    Benefits:
       - Reduced CPU usage (target: 5-10% vs 60-70%)
       - Eliminated multiprocessing overhead
       - Improved memory efficiency
       - Better GPU utilization
    
    Rollback:
       - Set USE_UNIFIED_GPU_PIPELINE = False
       - Revert FORCE_GPU_ONLY to False if needed
       - Re-enable ENABLE_MULTIPROCESSING if desired
    """
    
    return guide


# Validate configuration on import if unified pipeline is enabled
if config.processing.USE_UNIFIED_GPU_PIPELINE:
    validation_results = validate_unified_pipeline_config(config)
    
    if validation_results['errors']:
        import logging
        logger = logging.getLogger("config")
        logger.error("❌ Configuration validation failed for unified GPU pipeline")
        for error in validation_results['errors']:
            logger.error(f"   - {error}")
        logger.info("🔧 Run get_pipeline_migration_guide() for help")
    else:
        print("✅ Unified GPU pipeline configuration validated successfully")
