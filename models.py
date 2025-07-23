"""
Data Models Module

This module defines the core data structures used throughout the application
for representing detection, tracking, and analysis results. These models provide
a structured way to handle and process data as it flows through the system.

Each model includes methods for converting to dictionary and JSON formats to
facilitate communication with frontend systems and data persistence. The models
also provide helper properties for common calculations.

Key components:
- Point: Simple named tuple for 2D coordinates
- DetectionResult: Container for object detection data
- TrackingResult: Container for object tracking data with history
- AnalysisFrame: Container for a processed frame with all analysis results
- Utility functions for data type conversion and serialization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
import numpy as np
import time
from collections import deque
import json
import cv2

class Point(NamedTuple):
    """Simple 2D point representation for coordinates
    
    Used for representing detection centers, keypoints, and other spatial data.
    Being a NamedTuple provides immutability and unpacking capability.
    """
    x: float
    y: float


@dataclass
class DetectionResult:
    """Container for object detection results from a single detection
    
    Stores all relevant information about a detected object in a single frame,
    including bounding box coordinates, class information, confidence score,
    and optional keypoints (for pose estimation) and feature vectors (for re-identification).
    
    The properties (center, width, height, area) provide convenient access to
    derived measurements without having to recalculate them each time.
    """
    id: int  # Local identifier within a frame
    class_id: int  # Class identifier (e.g., 0 for person)
    confidence: float  # Confidence score
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    keypoints: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (keypoints_xy, keypoints_conf)
    feature_vector: Optional[np.ndarray] = None  # Appearance feature vector
    mask: Optional[np.ndarray] = None  # Segmentation mask (H x W binary array)
    
    @property
    def center(self) -> Point:
        """Calculate center point of the detection bounding box
        
        Returns:
            Point with x,y coordinates of the center
        """
        x1, y1, x2, y2 = self.bbox
        return Point((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def width(self) -> float:
        """Calculate width of the detection bounding box
        
        Returns:
            Width in pixels
        """
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Calculate height of the detection bounding box
        
        Returns:
            Height in pixels
        """
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        """Calculate area of the detection bounding box
        
        Returns:
            Area in square pixels
        """
        return self.width * self.height
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization
        
        Handles conversion of numpy arrays to lists and ensures all values
        are of serializable types. This method is used for WebSocket communication
        and data logging.
        
        Returns:
            Dictionary representation with serializable values
        """
        result = {
            "id": int(self.id),
            "class_id": int(self.class_id),
            "confidence": float(self.confidence),
            "bbox": [float(x) for x in self.bbox],
            "center": {"x": float(self.center.x), "y": float(self.center.y)}
        }
        
        # Add keypoints if available
        if self.keypoints:
            keypoints_xy, keypoints_conf = self.keypoints
            result["keypoints"] = {
                "positions": keypoints_xy.tolist() if isinstance(keypoints_xy, np.ndarray) else keypoints_xy,
                "confidences": keypoints_conf.tolist() if isinstance(keypoints_conf, np.ndarray) else keypoints_conf
            }
        
        # Add feature vector if available
        if self.feature_vector is not None:
            result["feature_vector"] = self.feature_vector.tolist() if isinstance(self.feature_vector, np.ndarray) else self.feature_vector
        
        # Add mask if available
        if self.mask is not None:
            result["mask"] = self.mask.tolist() if isinstance(self.mask, np.ndarray) else self.mask
            
        return result


@dataclass
class TrackingResult:
    """Container for tracking results that persists across multiple frames
    
    Represents a tracked object with its identification, current state, and history.
    TrackingResult extends DetectionResult by adding temporal information (history)
    and maintaining state across frames. This allows for motion analysis, trajectory
    visualization, and zone-based analytics.
    """
    track_id: int  # Unique tracking identifier
    camera_id: str  # Camera identifier
    detection: DetectionResult  # Current detection
    state: str = "Unknown"  # Current state (e.g., "Walking", "Standing", "Sitting")
    last_seen: float = field(default_factory=time.time)  # Timestamp
    positions: deque = field(default_factory=lambda: deque(maxlen=50))  # Position history
    keypoints_history: deque = field(default_factory=lambda: deque(maxlen=50))  # Keypoint history
    feature_history: deque = field(default_factory=lambda: deque(maxlen=50))  # Feature history
    velocity: Tuple[float, float] = (0.0, 0.0)  # Current velocity (vx, vy)
    zone: Optional[str] = None  # Current zone name
    dwell_time: float = 0.0  # Time spent in current zone
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization
        
        Includes all tracking information including history data for
        visualization of tracks and analysis of motion patterns.
        
        Returns:
            Dictionary representation with serializable values
        """
        result = {
            "track_id": int(self.track_id),
            "camera_id": self.camera_id,
            "detection": self.detection.to_dict(),
            "state": self.state,
            "last_seen": float(self.last_seen),
            "velocity": [float(self.velocity[0]), float(self.velocity[1])],
            "speed": float(np.linalg.norm(self.velocity)),
        }
        
        if self.zone:
            result["zone"] = self.zone
            result["dwell_time"] = float(self.dwell_time)
            
        # Add recent positions as trail
        if self.positions:
            result["trail"] = [{"x": float(p[0]), "y": float(p[1])} for p in list(self.positions)]
            
        return result


@dataclass
class AnalysisFrame:
    """Container for synchronized frame data and analysis results
    
    This is the highest-level data structure that contains everything related to
    a processed frame: the frame itself, all detections, all tracking results,
    and performance metrics. It represents the complete state of analysis for
    a given point in time for a specific camera.
    
    AnalysisFrame objects are the main data structure passed between the processing
    pipeline components and used for visualization and data communication.
    """
    frame_id: int  # Unique identifier for this frame
    camera_id: str  # Camera identifier
    timestamp: float  # Capture timestamp
    frame: Optional[np.ndarray] = None  # Raw OpenCV frame (might be None in data streams)
    detections: List[DetectionResult] = field(default_factory=list)  # Detection results
    tracks: List[TrackingResult] = field(default_factory=list)  # Tracking results
    frame_width: int = 0  # Frame width
    frame_height: int = 0  # Frame height
    processing_time: float = 0.0  # Total processing time in milliseconds
    detection_time: float = 0.0  # Time spent on detection in milliseconds
    tracking_time: float = 0.0  # Time spent on tracking in milliseconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization
        
        Creates a complete representation of the frame analysis including
        all detections, tracks, and performance metrics.
        
        Returns:
            Dictionary representation with serializable values
        """
        result = {
            "frame_id": int(self.frame_id),
            "camera_id": self.camera_id,
            "timestamp": float(self.timestamp),
            "frame_dimensions": {
                "width": int(self.frame_width),
                "height": int(self.frame_height),
            },
            "timing": {
                "processing_time": float(self.processing_time),
                "detection_time": float(self.detection_time),
                "tracking_time": float(self.tracking_time),
            },
            "detections": [d.to_dict() for d in self.detections],
            "tracks": [t.to_dict() for t in self.tracks],
        }
        return result
    
    def encode_frame(self, quality: int = 90) -> Optional[bytes]:
        """Encode the frame as JPEG bytes for transmission
        
        Compresses the frame for efficient network transmission or storage.
        Used primarily for sending frames via WebSockets.
        
        Args:
            quality: JPEG compression quality (0-100)
            
        Returns:
            Compressed JPEG bytes or None if frame is not available
        """
        if self.frame is None:
            return None
        _, encoded_img = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return encoded_img.tobytes()


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization
    
    Recursively processes dictionaries, lists, and numpy arrays to ensure
    all values are JSON serializable. This is essential for WebSocket
    communication and data storage.
    
    Args:
        obj: Object to convert (can be dict, list, numpy array, etc.)
        
    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def convert_detection_to_detection_result(detection, detection_id: int = 0) -> DetectionResult:
    """Convert Detection object to DetectionResult object
    
    This function bridges the gap between the Detection objects returned by
    DetectionManager and the DetectionResult objects expected by AnalysisFrame.
    It preserves all data including masks, keypoints, and feature vectors.
    
    Args:
        detection: Detection object from detection.py
        detection_id: Unique ID for this detection within the frame
        
    Returns:
        DetectionResult object with all data preserved
    """
    # Convert keypoints if present
    keypoints = None
    if hasattr(detection, 'keypoints') and detection.keypoints is not None:
        keypoints = detection.keypoints
    
    # Convert feature vector if present  
    feature_vector = None
    if hasattr(detection, 'features') and detection.features is not None:
        feature_vector = detection.features
    elif hasattr(detection, 'feature_vector') and detection.feature_vector is not None:
        feature_vector = detection.feature_vector
    
    # Convert mask if present
    mask = None
    if hasattr(detection, 'mask') and detection.mask is not None:
        mask = detection.mask
    
    return DetectionResult(
        id=detection_id,
        class_id=detection.class_id,
        confidence=detection.confidence,
        bbox=detection.bbox,
        keypoints=keypoints,
        feature_vector=feature_vector,
        mask=mask
    )


def to_json(obj: Any) -> str:
    """Convert an object to JSON string, handling numpy types
    
    Convenience wrapper combining convert_numpy_types and json.dumps
    to directly convert complex objects to JSON strings.
    
    Args:
        obj: Object to convert to JSON
        
    Returns:
        JSON string representation
    """
    return json.dumps(convert_numpy_types(obj)) 