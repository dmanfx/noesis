import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque

from ultralytics import YOLO
import supervision as sv
from pathlib import Path
import os
import torch
from dataclasses import dataclass, field
from models import DetectionResult, convert_numpy_types
import torchreid
from torchvision import transforms as T
from PIL import Image

# Will be dynamically imported if available
ultralytics = None
StrongSORT = None

# Import deep-person-reid
try:
    import torchreid
except ImportError:
    torchreid = None
    logging.warning("torchreid not available. Feature extraction will be disabled.")

# Import supervision for detection handling
try:
    import supervision as sv
except ImportError:
    sv = None
    logging.warning("supervision not available. Some detection features may be limited.")

# Global variables for optional dependencies
ultralytics = None
StrongSORT = None

# GPU-ONLY ENFORCEMENT MODE
class GPUOnlyEnforcer:
    """Enforces GPU-only operations by intercepting CPU usage attempts"""
    
    @staticmethod
    def enforce_tensor_on_gpu(tensor, operation_name: str):
        """Ensure tensor is on GPU, raise exception if not"""
        if hasattr(tensor, 'device') and tensor.device.type != 'cuda':
            raise RuntimeError(f"GPU ENFORCEMENT VIOLATION: {operation_name} tensor is on {tensor.device}, expected CUDA")
    
    @staticmethod
    def enforce_no_cpu_numpy(data, operation_name: str):
        """Prevent CPU numpy operations in GPU-only mode"""
        if isinstance(data, np.ndarray):
            raise RuntimeError(f"GPU ENFORCEMENT VIOLATION: {operation_name} using CPU numpy array, should use GPU tensors")
    
    @staticmethod
    def enforce_gpu_model(model, operation_name: str):
        """Ensure model is on GPU"""
        if hasattr(model, 'device'):
            if model.device.type != 'cuda':
                raise RuntimeError(f"GPU ENFORCEMENT VIOLATION: {operation_name} model is on {model.device}, expected CUDA")
        # Check model parameters
        for name, param in model.named_parameters():
            if param.device.type != 'cuda':
                raise RuntimeError(f"GPU ENFORCEMENT VIOLATION: {operation_name} model parameter {name} is on {param.device}, expected CUDA")


class Detection:
    """Represents a single object detection."""
    
    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        confidence: float,
        class_id: int,
        class_name: Optional[str] = None,
        features: Optional[np.ndarray] = None
    ):
        """
        Initialize a detection.
        
        Args:
            bbox: Bounding box coordinates as (x1, y1, x2, y2)
            confidence: Detection confidence score
            class_id: Class ID of the detected object
            class_name: Human-readable name of the detected class
            features: Optional feature vector for re-identification
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name or f"class_{class_id}"
        self.features = features
        # Add keypoints attribute for storing pose keypoints
        self.keypoints = None
        # Add mask attribute for storing segmentation masks
        self.mask = None
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        """Get the area of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class Track:
    """Represents a tracked object across multiple frames."""
    
    def __init__(
        self,
        detection: Detection = None,
        track_id: int = None,
        class_id: int = None,
        class_name: str = None,
        bbox: Tuple[float, float, float, float] = None,
        confidence: float = None,
        max_age: int = 30
    ):
        """
        Initialize a track from a detection or explicit parameters.
        
        Args:
            detection: Initial detection that started this track
            track_id: Unique identifier for this track
            class_id: Class ID if no detection provided
            class_name: Class name if no detection provided
            bbox: Bounding box if no detection provided
            confidence: Confidence score if no detection provided
            max_age: Maximum number of frames a track can exist without matching detections
        """
        if detection is not None:
            self.bbox = detection.bbox
            self.confidence = detection.confidence
            self.class_id = detection.class_id
            self.class_name = detection.class_name
            self.features = detection.features
            self.keypoints = detection.keypoints if hasattr(detection, 'keypoints') else None
            self.mask = detection.mask if hasattr(detection, 'mask') else None
            self.history = [detection.bbox]
        else:
            self.bbox = bbox
            self.confidence = confidence
            self.class_id = class_id
            self.class_name = class_name or f"class_{class_id}" if class_id is not None else "unknown"
            self.features = None
            self.keypoints = None
            self.mask = None
            self.history = [bbox] if bbox else []
        
        self.track_id = track_id
        self.max_age = max_age
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.is_confirmed = False
        self.active = True
        self.timestamp = time.time()
    
    def update(self, detection: Detection) -> None:
        """
        Update this track with a new matching detection.
        
        Args:
            detection: New detection matching this track
        """
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.features = detection.features
        self.mask = detection.mask if hasattr(detection, 'mask') else None
        self.history.append(detection.bbox)
        self.hits += 1
        self.time_since_update = 0
        
        # Confirm track if it has enough hits
        if not self.is_confirmed and self.hits >= 3:
            self.is_confirmed = True
    
    def predict(self) -> None:
        """
        Predict new location based on previous motion.
        This is a simple implementation that could be enhanced with Kalman filter.
        """
        # Increment age and time since last update
        self.age += 1
        self.time_since_update += 1
        
        # For now, we'll just use the last position
        # This could be extended with a motion model
        pass
    
    def mark_missed(self) -> None:
        """Mark this track as having missed a detection match."""
        self.time_since_update += 1
    
    def is_deleted(self) -> bool:
        """Check if this track should be deleted based on age."""
        return self.time_since_update > self.max_age


class DetectionManager:
    """Manages YOLO detection and tracking using ultralytics and StrongSORT."""
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
        target_classes: Optional[List[int]] = None,
        device: Optional[str] = None,
        config: Optional[Any] = None
    ):
        """Initialize the detection manager.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            target_classes: List of target class IDs (None for all classes)
            device: Device to run model on ('cpu', 'cuda', etc.)
            config: Application configuration object
        """
        self.logger = logging.getLogger("detection")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes
        self.config = config
        
        # GPU enforcement based on configuration
        if config and hasattr(config.models, 'FORCE_GPU_ONLY') and config.models.FORCE_GPU_ONLY:
            if not torch.cuda.is_available():
                raise RuntimeError("FORCE_GPU_ONLY is enabled but CUDA is not available")
            self.device = torch.device(config.models.DEVICE)
            self.force_gpu_only = True
            self.logger.info(f"GPU-only mode enforced on device: {self.device}")
        else:
            self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
            self.force_gpu_only = False
            if device is None:
                self.logger.warning("Device not specified, auto-detected: " + str(self.device))
        
        self.models_loaded = False
        self.model = None
        self.pose_model = None
        self.tracker = None
        self.tracks = {}  # track_id -> Track
        self.last_detections = {}  # frame_id -> List[Detection]
        self.stats = {
            "frames_processed": 0,
            "detections": 0,
            "total_processing_time": 0,
        }
        
        # Initialize classes with defaults
        self.class_names = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            # ... add more default classes as needed
        }
        
        # Try importing ultralytics
        self._import_dependencies()
    
    def _import_dependencies(self):
        """Import necessary dependencies for detection and tracking."""
        global ultralytics, StrongSORT
        
        try:
            import ultralytics
            self.logger.info("Successfully imported ultralytics")
        except ImportError:
            self.logger.error("Failed to import ultralytics. Please install with: pip install ultralytics")
            return False
        
        try:
            from strongsort.strong_sort import StrongSORT
            self.logger.info("Successfully imported StrongSORT")
        except ImportError:
            self.logger.warning("StrongSORT not available. Tracking will use built-in YOLO tracker.")
            StrongSORT = None
        
        return True
    
    def load_models(self):
        """Load YOLO detection and pose models."""
        if not ultralytics:
            self.logger.error("Cannot load models: ultralytics not imported")
            return False
        
        try:
            # Load detection model
            self.logger.info(f"Loading detection model from {self.model_path}")
            self.model = ultralytics.YOLO(self.model_path)
            
            # CRITICAL FIX: Move model to GPU after loading
            if self.force_gpu_only:
                self.logger.info(f"Moving detection model to GPU device: {self.device}")
                self.model.to(self.device)
            
            # Load pose model if keypoints are enabled and config is available
            if self.config and hasattr(self.config, 'visualization') and self.config.visualization.SHOW_KEYPOINTS:
                pose_model_path = self.config.models.POSE_MODEL_PATH
                self.logger.info(f"Loading pose model from {pose_model_path}")
                self.pose_model = ultralytics.YOLO(pose_model_path)
                
                # CRITICAL FIX: Move pose model to GPU after loading
                if self.force_gpu_only:
                    self.logger.info(f"Moving pose model to GPU device: {self.device}")
                    self.pose_model.to(self.device)
            
            # Load segmentation model if enabled
            if self.config and hasattr(self.config.models, 'ENABLE_SEGMENTATION') and self.config.models.ENABLE_SEGMENTATION:
                seg_model_path = self.config.models.SEGMENTATION_MODEL_PATH
                self.logger.info(f"Loading segmentation model from {seg_model_path}")
                self.seg_model = ultralytics.YOLO(seg_model_path)
                
                # CRITICAL FIX: Move segmentation model to GPU after loading
                if self.force_gpu_only:
                    self.logger.info(f"Moving segmentation model to GPU device: {self.device}")
                    self.seg_model.to(self.device)
            
            self.models_loaded = True
            self.logger.info("Models loaded successfully")
            
            # Update class names from model
            if hasattr(self.model, 'names'):
                for idx, name in self.model.names.items():
                    self.class_names[idx] = name
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def init_tracker(self):
        """Initialize the object tracker."""
        if StrongSORT:
            try:
                # Initialize StrongSORT tracker
                self.logger.info("Initializing StrongSORT tracker")
                self.tracker = StrongSORT(
                    model_weights='osnet_x0_25_msmt17.pt',
                    device='cuda' if ultralytics.CUDA else 'cpu',
                    max_dist=0.2,
                    max_iou_distance=0.7,
                    max_age=30,
                    n_init=3,
                    nn_budget=100,
                    mc_lambda=0.995
                )
                self.logger.info("StrongSORT tracker initialized")
            except Exception as e:
                self.logger.error(f"Error initializing StrongSORT tracker: {e}")
                self.tracker = None
        else:
            self.logger.info("Using built-in YOLO ByteTrack")
    
    def process_frame(self, frame: np.ndarray, frame_id: str = "default") -> Tuple[List[Detection], float]:
        """Process a frame with object detection and tracking.
        
        Args:
            frame: Input frame as numpy array
            frame_id: Identifier for the frame source
            
        Returns:
            Tuple containing lists of detections and processing time in ms
        """
        if not self.models_loaded:
            if not self.load_models():
                return [], 0.0
        
        start_time = time.time()
        
        try:
            # GPU-ONLY ENFORCEMENT: Check model is on GPU before inference
            if self.force_gpu_only:
                GPUOnlyEnforcer.enforce_gpu_model(self.model.model, "YOLO Detection")
                self.logger.debug(f"🔒 GPU ENFORCEMENT: YOLO model verified on GPU")
            
            # Run YOLO detection
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.target_classes if self.target_classes else None,
                device=self.device,  # CRITICAL FIX: Force GPU inference
                verbose=False,
                stream=False,
                # tracker="bytetrack.yaml"  # REMOVED: Tracking will be handled externally
            )
            
            # GPU-ONLY ENFORCEMENT: Verify results are on GPU
            if self.force_gpu_only and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    if hasattr(result.boxes, 'data'):
                        GPUOnlyEnforcer.enforce_tensor_on_gpu(result.boxes.data, "YOLO Detection Results")
                        self.logger.debug(f"🔒 GPU ENFORCEMENT: YOLO results verified on GPU")
                    else:
                        self.logger.warning("⚠️  YOLO results missing data tensor for GPU verification")
            
            # Process detection results
            detections = []
            # tracks = [] # REMOVED
            
            if len(results) > 0:
                # Get boxes, classes, scores from the first result
                result = results[0]
                
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    # Process detections
                    for box in boxes:
                        try:
                            class_id = int(box.cls.item())
                            confidence = float(box.conf.item())
                            
                            # Convert to pixel coordinates and ensure they are integers
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            bbox = (x1, y1, x2, y2)
                            
                            # Get class name
                            class_name = self.class_names.get(class_id, f"class_{class_id}")
                            
                            # Create Detection object
                            detection = Detection(
                                class_id=class_id,
                                class_name=class_name,
                                confidence=confidence,
                                bbox=bbox
                            )
                            
                            detections.append(detection)
                        except Exception as e:
                            self.logger.error(f"Error processing detection box: {e}")
            
            # Run segmentation if enabled
            if (hasattr(self, 'seg_model') and 
                self.config and hasattr(self.config.models, 'ENABLE_SEGMENTATION') and 
                self.config.models.ENABLE_SEGMENTATION and
                len(detections) > 0):
                
                try:
                    # GPU-ONLY ENFORCEMENT: Check segmentation model is on GPU
                    if self.force_gpu_only:
                        GPUOnlyEnforcer.enforce_gpu_model(self.seg_model.model, "Segmentation")
                        self.logger.debug(f"🔒 GPU ENFORCEMENT: Segmentation model verified on GPU")
                    
                    seg_results = self.seg_model.predict(
                        source=frame,
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        classes=self.target_classes if self.target_classes else None,
                        device=self.device,  # CRITICAL FIX: Force GPU inference for segmentation
                        verbose=False
                    )
                    
                    if len(seg_results) > 0 and hasattr(seg_results[0], 'masks') and seg_results[0].masks is not None:
                        # GPU enforcement for segmentation masks
                        if self.config and hasattr(self.config.models, 'FORCE_GPU_ONLY') and self.config.models.FORCE_GPU_ONLY:
                            masks_data = seg_results[0].masks.data
                            if masks_data.device.type != 'cuda':
                                raise RuntimeError("GPU enforcement violation: segmentation masks not on GPU")
                            if masks_data.dtype != torch.float16:
                                masks_data = masks_data.half()
                        else:
                            masks_data = seg_results[0].masks.data.cpu().numpy()  # Shape: (N, H, W)
                        frame_height, frame_width = frame.shape[:2]
                        mask_height, mask_width = masks_data.shape[1], masks_data.shape[2]
                        
                        self.logger.info(f"DetectionManager: Found {len(masks_data)} segmentation masks from model")
                        self.logger.debug(f"DetectionManager: Frame size: {frame_width}x{frame_height}, Mask size: {mask_width}x{mask_height}")
                        
                        # Match masks to detections by IoU
                        masks_assigned = 0
                        for i, detection in enumerate(detections):
                            best_mask_idx = -1
                            best_iou = 0.0
                            
                            for j, mask in enumerate(masks_data):
                                # Create bbox from mask - handle GPU tensors properly
                                if self.config and hasattr(self.config.models, 'FORCE_GPU_ONLY') and self.config.models.FORCE_GPU_ONLY:
                                    # For GPU tensors, use torch operations
                                    mask_binary = mask > 0.5
                                    if mask_binary.any():
                                        coords = torch.nonzero(mask_binary, as_tuple=False)
                                        if len(coords) > 0:
                                            mask_bbox = [
                                                float(coords[:, 1].min()),  # x1
                                                float(coords[:, 0].min()),  # y1
                                                float(coords[:, 1].max()),  # x2
                                                float(coords[:, 0].max())   # y2
                                            ]
                                        else:
                                            continue
                                    else:
                                        continue
                                else:
                                    # For CPU numpy arrays
                                    mask_coords = np.where(mask > 0.5)
                                    if len(mask_coords[0]) > 0:
                                        mask_bbox = [
                                            float(np.min(mask_coords[1])),  # x1
                                            float(np.min(mask_coords[0])),  # y1
                                            float(np.max(mask_coords[1])),  # x2
                                            float(np.max(mask_coords[0]))   # y2
                                        ]
                                    else:
                                        continue
                                    
                                    # Scale mask bbox to frame coordinates if needed
                                    if mask_width != frame_width or mask_height != frame_height:
                                        scale_x = frame_width / mask_width
                                        scale_y = frame_height / mask_height
                                        mask_bbox = [
                                            mask_bbox[0] * scale_x,  # x1
                                            mask_bbox[1] * scale_y,  # y1
                                            mask_bbox[2] * scale_x,  # x2
                                            mask_bbox[3] * scale_y   # y2
                                        ]
                                    
                                    # Calculate IoU with detection bbox
                                    iou = self._calculate_bbox_iou(detection.bbox, mask_bbox)
                                    if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                                        best_iou = iou
                                        best_mask_idx = j
                            
                            # Assign best matching mask (store original resolution, will be resized in visualization)
                            if best_mask_idx >= 0:
                                detections[i].mask = masks_data[best_mask_idx]
                                masks_assigned += 1
                                self.logger.debug(f"DetectionManager: Assigned mask {best_mask_idx} to detection {i} (IoU: {best_iou:.3f})")
                        
                        self.logger.info(f"DetectionManager: Assigned {masks_assigned} masks to {len(detections)} detections")
                    else:
                        self.logger.warning("DetectionManager: No masks found in segmentation results")
                                
                except Exception as e:
                    self.logger.error(f"Error in segmentation processing: {e}")
                
                # REMOVED: Internal track processing logic
                # if hasattr(result, 'tracker_results') and result.tracker_results is not None:
                #    ...

            # REMOVED: Pose estimation logic (should be handled externally)
            # try:
            #    if (...):
            #        ...
            # except Exception as e:
            #    ...
            
            # Update statistics
            self.stats["frames_processed"] += 1
            self.stats["detections"] += len(detections)
            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time
            
            # REMOVED: Storing last detections and track cleanup
            # self.last_detections[frame_id] = detections
            # self._cleanup_tracks()
            
            return detections, processing_time * 1000  # Return time in milliseconds
            
        except Exception as e:
            self.logger.error(f"Error in process_frame: {e}")
            import traceback
            traceback.print_exc()
            return [], 0.0  # Return empty list and zero processing time on error
    
    def _cleanup_tracks(self):
        """Remove inactive tracks older than the trace persistence time."""
        current_time = time.time()
        
        # Set default trace persistence time if config is not available
        if self.config and hasattr(self.config, 'models') and hasattr(self.config.models, 'TRACE_PERSISTENCE_SECONDS'):
            trace_persistence = self.config.models.TRACE_PERSISTENCE_SECONDS
        else:
            trace_persistence = 30.0  # default value
            
        inactive_threshold = current_time - trace_persistence
        
        # Remove old inactive tracks
        to_remove = []
        for track_id, track in self.tracks.items():
            # Check if track has timestamp attribute and active attribute
            has_timestamp = hasattr(track, 'timestamp') and track.timestamp is not None
            is_inactive = hasattr(track, 'active') and not track.active
            
            if is_inactive and has_timestamp and track.timestamp < inactive_threshold:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def _calculate_bbox_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_active_tracks(self) -> List[Track]:
        """Get all currently active tracks.
        
        Returns:
            List of active Track objects
        """
        return [track for track in self.tracks.values() if track.active]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.stats.copy()
        
        # Calculate average processing time
        if stats["frames_processed"] > 0:
            stats["avg_processing_time"] = stats["total_processing_time"] / stats["frames_processed"]
            stats["avg_fps"] = stats["frames_processed"] / max(1, stats["total_processing_time"])
        else:
            stats["avg_processing_time"] = 0
            stats["avg_fps"] = 0
        
        stats["active_tracks"] = len(self.get_active_tracks())
        stats["total_tracks"] = len(self.tracks)
        
        return stats


def draw_detections(frame: np.ndarray, detections: List[Detection], config) -> np.ndarray:
    """Draw detection boxes and information on the frame.
    
    Args:
        frame: Input frame
        detections: List of Detection objects
        config: Application configuration
        
    Returns:
        Frame with visualizations
    """
    result = frame.copy()
    
    for detection in detections:
        if config.visualization.SHOW_DETECTION_BOXES:
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            
            # Draw bounding box
            cv2.rectangle(
                result, 
                (x1, y1), 
                (x2, y2), 
                config.visualization.COLOR_DETECTION, 
                config.visualization.BOX_THICKNESS
            )
            
            # Draw label
            label = f"{detection.class_name} {detection.confidence:.2f}"
            text_size, _ = cv2.getTextSize(
                label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                config.visualization.TEXT_THICKNESS
            )
            
            cv2.rectangle(
                result, 
                (x1, y1 - text_size[1] - 5), 
                (x1 + text_size[0], y1), 
                config.visualization.COLOR_DETECTION, 
                -1
            )
            
            cv2.putText(
                result, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                config.visualization.TEXT_THICKNESS
            )
        
        # Draw keypoints if available
        if config.visualization.SHOW_KEYPOINTS and detection.keypoints is not None:
            for kp in detection.keypoints:
                x, y, conf = kp
                if conf > 0.5:  # Only draw keypoints with sufficient confidence
                    cv2.circle(
                        result, 
                        (int(x), int(y)), 
                        config.visualization.KEYPOINT_RADIUS, 
                        (0, 255, 255), 
                        -1
                    )
    
    return result


def draw_tracks(frame: np.ndarray, tracks: List[Track], config) -> np.ndarray:
    """Draw tracking boxes, traces, and information on the frame.
    
    Args:
        frame: Input frame
        tracks: List of Track objects
        config: Application configuration
        
    Returns:
        Frame with visualizations
    """
    result = frame.copy()
    
    for track in tracks:
        x1, y1, x2, y2 = [int(coord) for coord in track.bbox]
        
        if config.visualization.SHOW_TRACKING_BOXES:
            # Draw bounding box
            cv2.rectangle(
                result, 
                (x1, y1), 
                (x2, y2), 
                config.visualization.COLOR_TRACKING, 
                config.visualization.BOX_THICKNESS
            )
            
            # Draw label with track ID
            label = f"{track.class_name} #{track.track_id} {track.confidence:.2f}"
            text_size, _ = cv2.getTextSize(
                label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                config.visualization.TEXT_THICKNESS
            )
            
            cv2.rectangle(
                result, 
                (x1, y1 - text_size[1] - 5), 
                (x1 + text_size[0], y1), 
                config.visualization.COLOR_TRACKING, 
                -1
            )
            
            cv2.putText(
                result, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                config.visualization.TEXT_THICKNESS
            )
        
        # Draw motion trace
        if config.visualization.SHOW_TRACES and len(track.history) > 1:
            points = [(int(x), int(y)) for _, x, y in track.history]
            for i in range(1, len(points)):
                cv2.line(
                    result, 
                    points[i-1], 
                    points[i], 
                    config.visualization.COLOR_TRACKING, 
                    config.visualization.TRACE_THICKNESS
                )
        
        # Draw keypoints if available
        if config.visualization.SHOW_KEYPOINTS and track.keypoints is not None:
            for kp in track.keypoints:
                x, y, conf = kp
                if conf > 0.5:  # Only draw keypoints with sufficient confidence
                    cv2.circle(
                        result, 
                        (int(x), int(y)), 
                        config.visualization.KEYPOINT_RADIUS, 
                        (0, 255, 255), 
                        -1
                    )
    
    return result


class PoseEstimator:
    """Handles loading the YOLO Pose model and performing pose estimation."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.3, device: Optional[str] = None):
        """
        Initializes the PoseEstimator.

        Args:
            model_path (str): Path to the YOLOv8-Pose model weights file.
            confidence_threshold (float): Minimum confidence for detected poses.
            device (Optional[str]): Device to run the model on ('cpu', 'cuda', etc.). Defaults to auto-detect.
        """
        self.logger = logging.getLogger(f"PoseEstimator")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device # Can be 'cpu', 'cuda:0', etc.

        if not os.path.exists(self.model_path):
            self.logger.error(f"Pose model file not found at: {self.model_path}")
            raise FileNotFoundError(f"Pose model not found: {self.model_path}")

        try:
            self.logger.info(f"Loading YOLO Pose model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            # Move model to specified device if provided
            if self.device:
                self.model.to(self.device)
            self.logger.info(f"YOLO Pose model loaded successfully onto device: {self.device or 'auto'}")
            # Perform a dummy inference? (Optional, might pre-compile)
            # _ = self.model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        except Exception as e:
            self.logger.exception(f"Failed to load YOLO Pose model: {e}")
            raise RuntimeError(f"Could not load pose model: {e}") from e

    def estimate(self, frame: np.ndarray, detections: Optional[Union[sv.Detections, List[Any]]] = None) -> Tuple[List[Optional[Tuple[np.ndarray, np.ndarray]]], float]:
        """
        Performs pose estimation on a frame.

        Args:
            frame (np.ndarray): The input video frame.
            detections (Optional[Union[sv.Detections, List]]): Object detections (either a sv.Detections object
                                                  or a list of DetectionResult objects).
                                                  Used for associating poses with objects. If None,
                                                  poses for the whole frame are returned without association.

        Returns:
            Tuple[List[Optional[Tuple[np.ndarray, np.ndarray]]], float]:
                - A list where each element corresponds to an input detection (if provided).
                  Each element is either None (no pose found/associated) or a tuple
                  containing (keypoints_xy, keypoints_conf).
                - Pose estimation time in milliseconds.
        """
        start_time = time.time()
        keypoints_list = []

        try:
            # GPU-ONLY ENFORCEMENT: Check pose model is on GPU
            if hasattr(self, 'config') and self.config and hasattr(self.config.models, 'FORCE_GPU_ONLY') and self.config.models.FORCE_GPU_ONLY:
                GPUOnlyEnforcer.enforce_gpu_model(self.model.model, "Pose Estimation")
                self.logger.debug(f"🔒 GPU ENFORCEMENT: Pose model verified on GPU")
            
            # Run pose estimation on the full frame
            # Conf threshold can be applied here or during association
            pose_results = self.model(frame, verbose=False, conf=self.confidence_threshold, device=self.device)[0]

            # Check if keypoints exist in the results
            if pose_results.keypoints is None:
                 # No keypoints detected in the frame at all
                 if detections is not None:
                     keypoints_list = [None] * (len(detections) if hasattr(detections, '__len__') else 0)
                 else:
                     keypoints_list = [] # Return empty list if no input detections
                 return keypoints_list, (time.time() - start_time) * 1000


            # Extract keypoints data (assuming Ultralytics v8 results structure)
            # GPU enforcement for pose keypoints
            if hasattr(self, 'config') and self.config and hasattr(self.config.models, 'FORCE_GPU_ONLY') and self.config.models.FORCE_GPU_ONLY:
                raw_keypoints_xy = pose_results.keypoints.xy if pose_results.keypoints.xy is not None else None
                raw_keypoints_conf = pose_results.keypoints.conf if pose_results.keypoints.conf is not None else None
                if raw_keypoints_xy is not None and raw_keypoints_xy.device.type != 'cuda':
                    raise RuntimeError("GPU enforcement violation: pose keypoints not on GPU")
                if raw_keypoints_conf is not None and raw_keypoints_conf.device.type != 'cuda':
                    raise RuntimeError("GPU enforcement violation: pose confidence not on GPU")
            else:
                # Convert to NumPy arrays on CPU
                raw_keypoints_xy = pose_results.keypoints.xy.cpu().numpy() if pose_results.keypoints.xy is not None else None
                raw_keypoints_conf = pose_results.keypoints.conf.cpu().numpy() if pose_results.keypoints.conf is not None else None
            num_pose_detections = len(raw_keypoints_xy) if raw_keypoints_xy is not None else 0

            if num_pose_detections == 0:
                if detections is not None:
                    keypoints_list = [None] * (len(detections) if hasattr(detections, '__len__') else 0)
                else:
                    keypoints_list = []
                return keypoints_list, (time.time() - start_time) * 1000

            # If input detections are provided, associate poses with them
            if detections is not None and hasattr(detections, '__len__') and len(detections) > 0:
                keypoints_list = [None] * len(detections) # Initialize with Nones
                # GPU enforcement for pose boxes
                if hasattr(self, 'config') and self.config and hasattr(self.config.models, 'FORCE_GPU_ONLY') and self.config.models.FORCE_GPU_ONLY:
                    pose_boxes_xyxy = pose_results.boxes.xyxy if pose_results.boxes is not None else None
                    if pose_boxes_xyxy is not None and pose_boxes_xyxy.device.type != 'cuda':
                        raise RuntimeError("GPU enforcement violation: pose boxes not on GPU")
                else:
                    pose_boxes_xyxy = pose_results.boxes.xyxy.cpu().numpy() if pose_results.boxes is not None else None

                if pose_boxes_xyxy is None:
                    self.logger.warning("Pose results missing bounding boxes for association.")
                    return keypoints_list, (time.time() - start_time) * 1000 # Return Nones

                # Basic Center Point Association
                # Check if detections is a sv.Detections object or a list of DetectionResult
                is_sv_detections = hasattr(detections, 'xyxy')
                
                for i in range(len(detections)):
                    if is_sv_detections:
                        # Get bounding box from sv.Detections
                        det_xyxy = detections.xyxy[i]
                    else:
                        # Get bounding box from DetectionResult
                        det_xyxy = detections[i].bbox
                        
                    det_center_x = (det_xyxy[0] + det_xyxy[2]) / 2
                    det_center_y = (det_xyxy[1] + det_xyxy[3]) / 2
                    best_match_idx = -1
                    min_dist_sq = float('inf')
                    
                    # Iterate through pose boxes to find the closest one
                    for j in range(min(num_pose_detections, len(pose_boxes_xyxy))):
                        pose_xyxy = pose_boxes_xyxy[j]
                        pose_center_x = (pose_xyxy[0] + pose_xyxy[2]) / 2
                        pose_center_y = (pose_xyxy[1] + pose_xyxy[3]) / 2
                        dist_sq = (det_center_x - pose_center_x)**2 + (det_center_y - pose_center_y)**2

                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            best_match_idx = j

                    # Assign keypoints if a plausible match is found
                    # Threshold could be based on detection size, e.g., 75% of width squared
                    max_allowed_dist_sq = ((det_xyxy[2] - det_xyxy[0]) * 0.75)**2
                    if best_match_idx != -1 and min_dist_sq < max_allowed_dist_sq:
                        matched_kps_xy = raw_keypoints_xy[best_match_idx]
                        # Handle case where confidence might be missing
                        matched_kps_conf = raw_keypoints_conf[best_match_idx] if raw_keypoints_conf is not None and best_match_idx < len(raw_keypoints_conf) else np.ones(matched_kps_xy.shape[0]) # Default conf to 1 if missing
                        keypoints_list[i] = (matched_kps_xy, matched_kps_conf)

            else:
                # If no input detections, return all detected poses
                keypoints_list = []
                for j in range(num_pose_detections):
                     matched_kps_xy = raw_keypoints_xy[j]
                     matched_kps_conf = raw_keypoints_conf[j] if raw_keypoints_conf is not None and j < len(raw_keypoints_conf) else np.ones(matched_kps_xy.shape[0])
                     keypoints_list.append((matched_kps_xy, matched_kps_conf))


        except Exception as e:
            self.logger.exception(f"Error during pose estimation: {e}")
            # Return list of Nones matching input detections on error, or empty list
            if detections is not None and hasattr(detections, '__len__'):
                keypoints_list = [None] * len(detections)
            else:
                keypoints_list = []

        estimation_time_ms = (time.time() - start_time) * 1000
        return keypoints_list, estimation_time_ms


class FeatureExtractor:
    """Handles loading the OSNet model and extracting appearance features for Re-ID."""

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initializes the FeatureExtractor.

        Args:
            model_path (str): Path to the OSNet model weights file.
            device (Optional[str]): Device to run the model on ('cpu', 'cuda', etc.). Defaults to auto-detect.
        """
        self.logger = logging.getLogger(f"FeatureExtractor")
        self.model_path = model_path
        self.device = device # Can be 'cpu', 'cuda:0', etc.

        if not os.path.exists(self.model_path):
            self.logger.error(f"OSNet model file not found at: {self.model_path}")
            raise FileNotFoundError(f"OSNet model not found: {self.model_path}")

        try:
            self.logger.info(f"Loading OSNet model from: {self.model_path}")
            # Initialize OSNet model
            self.model = torchreid.models.build_model(
                name='osnet_x1_0',
                num_classes=1,  # Dummy number of classes
                loss='softmax',
                pretrained=False
            )
            
            # Load pretrained weights
            torchreid.utils.load_pretrained_weights(self.model, self.model_path)
            
            # Move model to specified device if provided
            if self.device:
                self.logger.info(f"Moving ReID model to device: {self.device}")
                self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            self.logger.info(f"OSNet model loaded successfully onto device: {self.device or 'auto'}")
            
            # Define transforms for input images
            self.transforms = T.Compose([
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            self.logger.exception(f"Failed to load OSNet model: {e}")
            raise RuntimeError(f"Could not load OSNet model: {e}") from e

    def extract(self, frame: np.ndarray, detections) -> Tuple[List[Optional[np.ndarray]], float]:
        """
        Extracts appearance features for each detection in the frame.

        Args:
            frame (np.ndarray): The input video frame.
            detections: Either sv.Detections object or a list of Detection objects.

        Returns:
            Tuple[List[Optional[np.ndarray]], float]: List of feature vectors and extraction time in ms.
        """
        start_time = time.time()
        features_list = []
        
        try:
            # Handle either sv.Detections or list of Detection objects
            if hasattr(detections, 'xyxy'):  # sv.Detections object
                bboxes = detections.xyxy
            else:  # List of Detection objects
                bboxes = [det.bbox for det in detections]
            
            for i, bbox in enumerate(bboxes):
                # Crop person from frame using bounding box
                x1, y1, x2, y2 = map(int, bbox)
                # Ensure coordinates are within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    self.logger.warning(f"Invalid bbox coordinates: {bbox}")
                    features_list.append(None)
                    continue
                
                person_img = frame[y1:y2, x1:x2]
                
                if person_img.size == 0:
                    self.logger.warning(f"Empty person image for bbox: {bbox}")
                    features_list.append(None)
                    continue
                
                # Convert to PIL Image and apply transforms
                person_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
                person_tensor = self.transforms(person_pil).unsqueeze(0)
                
                # Move to device and ensure FP16 if GPU-only mode
                if hasattr(self, 'config') and self.config and hasattr(self.config.models, 'FORCE_GPU_ONLY') and self.config.models.FORCE_GPU_ONLY:
                    person_tensor = person_tensor.to(self.device, dtype=torch.float16)
                elif self.device:
                    person_tensor = person_tensor.to(self.device)
                
                # GPU-ONLY ENFORCEMENT: Verify tensor is on GPU before inference
                if hasattr(self, 'config') and self.config and hasattr(self.config.models, 'FORCE_GPU_ONLY') and self.config.models.FORCE_GPU_ONLY:
                    GPUOnlyEnforcer.enforce_tensor_on_gpu(person_tensor, "ReID Feature Extraction Input")
                    GPUOnlyEnforcer.enforce_gpu_model(self.model, "ReID Feature Extraction")
                    self.logger.debug(f"🔒 GPU ENFORCEMENT: ReID input tensor and model verified on GPU")
                
                # Extract features
                with torch.no_grad():
                    features = self.model(person_tensor)
                
                # GPU-ONLY ENFORCEMENT: Verify output features are on GPU
                if hasattr(self, 'config') and self.config and hasattr(self.config.models, 'FORCE_GPU_ONLY') and self.config.models.FORCE_GPU_ONLY:
                    GPUOnlyEnforcer.enforce_tensor_on_gpu(features, "ReID Feature Extraction Output")
                    self.logger.debug(f"🔒 GPU ENFORCEMENT: ReID output features verified on GPU")
                
                # Handle GPU-only mode - keep tensors on GPU until final conversion
                if hasattr(self, 'config') and self.config and hasattr(self.config.models, 'FORCE_GPU_ONLY') and self.config.models.FORCE_GPU_ONLY:
                    if features.device.type != 'cuda':
                        raise RuntimeError("GPU enforcement violation: ReID features not on GPU")
                    # Keep features on GPU, convert to numpy only when absolutely necessary
                    features_np = features.detach().cpu().numpy().flatten()
                else:
                    features_np = features.cpu().numpy().flatten()  # Flatten to ensure 1D array
                if np.linalg.norm(features_np) > 0:
                    features_np = features_np / np.linalg.norm(features_np)
                
                features_list.append(features_np)
                
        except Exception as e:
            self.logger.exception(f"Error during feature extraction: {e}")
            # Return list of Nones matching input detections on error
            features_list = [None] * (len(detections) if hasattr(detections, '__len__') else 0)
        
        extraction_time_ms = (time.time() - start_time) * 1000
        return features_list, extraction_time_ms 