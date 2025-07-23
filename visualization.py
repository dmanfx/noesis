from typing import List, Tuple, Dict, Optional, Any
import cv2
import numpy as np
import random
import supervision as sv  # Add Supervision import
import torch
import logging

from detection import Detection, Track
from models import DetectionResult, TrackingResult

# Import GPU visualization if available
try:
    from gpu_visualization import GPUVisualizer
    GPU_VISUALIZATION_AVAILABLE = True
except ImportError:
    GPU_VISUALIZATION_AVAILABLE = False
    logging.getLogger(__name__).info("GPU visualization not available - using CPU rendering")

# Import NVENC encoder if available
try:
    from nvenc_encoder import NVENCEncoder, NVJPEGEncoder, EncoderConfig
    NVENC_AVAILABLE = True
except ImportError:
    NVENC_AVAILABLE = False
    logging.getLogger(__name__).info("NVENC encoder not available - using CPU encoding")

class Visualizer:
    """Class for visualizing detection and tracking results."""
    
    def __init__(
        self,
        class_names: List[str],
        colors: Optional[List[Tuple[int, int, int]]] = None,
        thickness: int = 2,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        trail_length: int = 30
    ):
        """
        Initialize visualizer.
        
        Args:
            class_names: List of class names
            colors: List of colors for each class (BGR format)
            thickness: Line thickness for bounding boxes
            text_scale: Scale factor for text
            text_thickness: Thickness of text
            text_color: Color of text (BGR format)
            trail_length: Maximum length of track trails
        """
        self.class_names = class_names
        self.thickness = thickness
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.text_color = text_color
        self.trail_length = trail_length
        
        # Generate colors if not provided
        if colors is None:
            self.colors = self._generate_colors(len(class_names))
        else:
            self.colors = colors
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """
        Generate random colors for each class.
        
        Args:
            num_classes: Number of classes
            
        Returns:
            List of BGR colors
        """
        colors = []
        for _ in range(num_classes):
            # Generate vibrant colors (avoiding dark colors)
            color = tuple(random.randint(100, 255) for _ in range(3))
            colors.append(color)
        return colors
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        draw_labels: bool = True,
        draw_confidence: bool = False
    ) -> np.ndarray:
        """
        Draw detection bounding boxes on the frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            draw_labels: Whether to draw class labels
            draw_confidence: Whether to include confidence scores in labels
            
        Returns:
            Frame with drawn detections
        """
        result = frame.copy()
        
        for detection in detections:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, detection.bbox)
            
            # Get class color
            color = self.colors[detection.class_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, self.thickness)
            
            # Draw label
            if draw_labels:
                label_text = self.class_names[detection.class_id]
                
                if draw_confidence:
                    label_text += f" {detection.confidence:.2f}"
                
                # Draw label background
                text_size = cv2.getTextSize(
                    label_text, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    self.text_scale, 
                    self.text_thickness
                )[0]
                
                cv2.rectangle(
                    result,
                    (x1, y1 - text_size[1] - 5),
                    (x1 + text_size[0], y1),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    result,
                    label_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_scale,
                    self.text_color,
                    self.text_thickness
                )
        
        return result
    
    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[TrackingResult],
        draw_labels: bool = True,
        draw_track_ids: bool = True,
        draw_trails: bool = True
    ) -> np.ndarray:
        """
        Draw tracks on the frame.
        
        Args:
            frame: Input frame
            tracks: List of TrackingResult objects
            draw_labels: Whether to draw class labels
            draw_track_ids: Whether to include track IDs in labels
            draw_trails: Whether to draw track trails
            
        Returns:
            Frame with drawn tracks
        """
        result = frame.copy()
        
        for track in tracks:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, track.bbox)
            
            # Get class color
            color = self.colors[track.class_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, self.thickness)
            
            # Draw label
            if draw_labels:
                label_text = self.class_names[track.class_id]
                
                if draw_track_ids:
                    label_text += f" #{track.track_id}"
                
                # Draw label background
                text_size = cv2.getTextSize(
                    label_text, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    self.text_scale, 
                    self.text_thickness
                )[0]
                
                cv2.rectangle(
                    result,
                    (x1, y1 - text_size[1] - 5),
                    (x1 + text_size[0], y1),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    result,
                    label_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_scale,
                    self.text_color,
                    self.text_thickness
                )
            
            # Draw trail
            if draw_trails and len(track.history) > 1:
                # Limit trail length
                trail = track.history[-self.trail_length:]
                
                # Draw trail lines
                for i in range(len(trail) - 1):
                    pt1 = (int((trail[i][0] + trail[i][2]) / 2), int((trail[i][1] + trail[i][3]) / 2))
                    pt2 = (int((trail[i+1][0] + trail[i+1][2]) / 2), int((trail[i+1][1] + trail[i+1][3]) / 2))
                    cv2.line(result, pt1, pt2, color, self.thickness)
        
        return result
    
    def add_info_text(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int] = (10, 30),
        color: Tuple[int, int, int] = (255, 255, 255),
        scale: float = None,
        thickness: int = None
    ) -> np.ndarray:
        """
        Add information text to the frame.
        
        Args:
            frame: Input frame
            text: Text to add
            position: Position of the text (x, y)
            color: Color of the text (BGR format)
            scale: Text scale factor (uses default if None)
            thickness: Text thickness (uses default if None)
            
        Returns:
            Frame with added text
        """
        result = frame.copy()
        
        cv2.putText(
            result,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale or self.text_scale,
            color,
            thickness or self.text_thickness
        )
        
        return result
    
    def add_fps_counter(
        self,
        frame: np.ndarray,
        fps: float,
        position: Tuple[int, int] = (10, 30),
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Add FPS counter to the frame.
        
        Args:
            frame: Input frame
            fps: FPS value
            position: Position of the counter (x, y)
            color: Color of the counter (BGR format)
            
        Returns:
            Frame with FPS counter
        """
        return self.add_info_text(
            frame=frame,
            text=f"FPS: {fps:.1f}",
            position=position,
            color=color,
            scale=self.text_scale * 1.2,
            thickness=self.text_thickness + 1
        ) 

class VisualizationManager:
    """Manages visualization of detection and tracking results."""
    
    def __init__(self, use_gpu: bool = False, device: str = 'cuda:0'):
        """Initialize visualization manager.
        
        Args:
            use_gpu: Whether to use GPU-accelerated visualization
            device: GPU device to use
        """
        self.visualizer = None
        self.fps_history = []
        self.fps_window_size = 30  # Calculate FPS over last 30 frames
        self.use_gpu = use_gpu and GPU_VISUALIZATION_AVAILABLE
        self.device = device
        self.logger = logging.getLogger("VisualizationManager")
        
        # Default class names if not initialized
        self.default_class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
        
        # GPU visualizer
        self.gpu_visualizer = None
        if self.use_gpu:
            try:
                self.gpu_visualizer = GPUVisualizer(device=device)
                self.logger.info(f"GPU visualization enabled on {device}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize GPU visualizer: {e}")
                self.use_gpu = False
        
        # NVENC encoder for hardware video encoding
        self.nvenc_encoder = None
        self.nvjpeg_encoder = None
        
        # Add direct Supervision annotators like in YOLOrun_dub.py
        # BoxAnnotator only accepts thickness parameter
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.0)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)
        
    def initialize(self, config):
        """Initialize visualizer with configuration.
        
        Args:
            config: Application configuration
        """
        # Check if GPU visualization is enabled in config
        if hasattr(config.visualization, 'USE_GPU_VISUALIZATION'):
            self.use_gpu = config.visualization.USE_GPU_VISUALIZATION and GPU_VISUALIZATION_AVAILABLE
        
        self.visualizer = Visualizer(
            class_names=config.models.CLASS_NAMES,
            thickness=config.visualization.BOX_THICKNESS,
            text_scale=config.visualization.TEXT_SCALE,
            text_thickness=config.visualization.TEXT_THICKNESS,
            trail_length=config.visualization.TRACE_LENGTH
        )
        
        # Initialize hardware encoders if enabled
        if hasattr(config.visualization, 'USE_NVENC') and config.visualization.USE_NVENC and NVENC_AVAILABLE:
            try:
                # NVENC for video encoding
                encoder_config = EncoderConfig(
                    width=config.cameras.CAMERA_WIDTH,
                    height=config.cameras.CAMERA_HEIGHT,
                    fps=30,
                    codec='h264_nvenc',
                    bitrate=getattr(config.visualization, 'NVENC_BITRATE', 4000000),
                    gpu_id=int(self.device.split(':')[-1]) if ':' in self.device else 0
                )
                self.nvenc_encoder = NVENCEncoder(encoder_config)
                self.nvenc_encoder.start()
                self.logger.info("NVENC hardware video encoding enabled")
                
                # NVJPEG for image encoding
                self.nvjpeg_encoder = NVJPEGEncoder(
                    quality=getattr(config.visualization, 'JPEG_QUALITY', 85),
                    device=self.device
                )
                self.logger.info("NVJPEG hardware image encoding enabled")
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize hardware encoders: {e}")
        
        # Update Supervision annotators with config values
        # BoxAnnotator only accepts thickness parameter
        self.box_annotator = sv.BoxAnnotator(
            thickness=config.visualization.BOX_THICKNESS
        )
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=config.visualization.TEXT_THICKNESS,
            text_scale=config.visualization.TEXT_SCALE
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=config.visualization.BOX_THICKNESS,
            trace_length=config.visualization.TRACE_LENGTH
        )
    
    def get_class_names(self):
        """Get class names, using default if visualizer is not initialized."""
        if self.visualizer and hasattr(self.visualizer, 'class_names'):
            return self.visualizer.class_names
        return self.default_class_names

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        tracks: List[TrackingResult],
        show_traces: bool = True,
        show_detection_boxes: bool = True,
        show_tracking_boxes: bool = True,
        show_keypoints: bool = True,
        show_masks: bool = True,
        mask_alpha: float = 0.5,
        fps: Optional[float] = None,
        frame_tensor: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """Annotate frame with detection and tracking visualizations.
        
        Args:
            frame: Input frame
            detections: List of detections
            tracks: List of TrackingResult objects
            show_traces: Whether to show track traces
            show_detection_boxes: Whether to show detection boxes
            show_tracking_boxes: Whether to show tracking boxes
            show_keypoints: Whether to show pose keypoints
            show_masks: Whether to show segmentation masks
            mask_alpha: Transparency for mask overlays (0.0-1.0)
            fps: Current FPS to display
            frame_tensor: Optional GPU tensor if using GPU visualization
            
        Returns:
            Annotated frame
        """
        # Use GPU visualization if enabled and tensor provided
        if self.use_gpu and self.gpu_visualizer and frame_tensor is not None:
            return self._annotate_frame_gpu(
                frame_tensor=frame_tensor,
                detections=detections,
                tracks=tracks,
                show_traces=show_traces,
                show_detection_boxes=show_detection_boxes,
                show_tracking_boxes=show_tracking_boxes,
                show_keypoints=show_keypoints,
                show_masks=show_masks,
                mask_alpha=mask_alpha,
                fps=fps
            )
        
        # Otherwise use CPU visualization
        return self._annotate_frame_cpu(
            frame=frame,
            detections=detections,
            tracks=tracks,
            show_traces=show_traces,
            show_detection_boxes=show_detection_boxes,
            show_tracking_boxes=show_tracking_boxes,
            show_keypoints=show_keypoints,
            show_masks=show_masks,
            mask_alpha=mask_alpha,
            fps=fps
        )
    
    def _annotate_frame_gpu(
        self,
        frame_tensor: torch.Tensor,
        detections: List[DetectionResult],
        tracks: List[TrackingResult],
        show_traces: bool = True,
        show_detection_boxes: bool = True,
        show_tracking_boxes: bool = True,
        show_keypoints: bool = True,
        show_masks: bool = True,
        mask_alpha: float = 0.5,
        fps: Optional[float] = None
    ) -> np.ndarray:
        """GPU-accelerated frame annotation.
        
        Args:
            frame_tensor: GPU tensor (C, H, W) in RGB format [0,1]
            detections: List of detections
            tracks: List of TrackingResult objects
            Other args same as annotate_frame
            
        Returns:
            Annotated frame as numpy array
        """
        try:
            # Draw detections and tracks
            annotated_tensor = self.gpu_visualizer.draw_detections(
                frame_tensor=frame_tensor,
                detections=detections,
                tracks=tracks
            )
            
            # Add FPS overlay
            if fps is not None:
                self.fps_history.append(fps)
                if len(self.fps_history) > self.fps_window_size:
                    self.fps_history.pop(0)
                
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                overlay_info = {'fps': avg_fps, 'detection_count': len(detections)}
                annotated_tensor = self.gpu_visualizer.create_overlay_tensor(
                    annotated_tensor, overlay_info
                )
            
            # Encode to JPEG using hardware acceleration
            encoded_jpeg = self.gpu_visualizer.encode_frame_gpu(annotated_tensor)
            
            # Decode back to numpy for compatibility
            # TODO: Return tensor directly when full GPU pipeline is ready
            import cv2
            frame_np = cv2.imdecode(
                np.frombuffer(encoded_jpeg, np.uint8),
                cv2.IMREAD_COLOR
            )
            
            # Convert BGR to RGB
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            
            return frame_np
            
        except Exception as e:
            self.logger.error(f"GPU annotation failed: {e}")
            # Fallback to CPU visualization
            frame_np = frame_tensor.cpu().numpy()
            if frame_np.shape[0] == 3:  # CHW format
                frame_np = frame_np.transpose(1, 2, 0)
            frame_np = (frame_np * 255).astype(np.uint8)
            
            return self._annotate_frame_cpu(
                frame=frame_np,
                detections=detections,
                tracks=tracks,
                show_traces=show_traces,
                show_detection_boxes=show_detection_boxes,
                show_tracking_boxes=show_tracking_boxes,
                show_keypoints=show_keypoints,
                show_masks=show_masks,
                mask_alpha=mask_alpha,
                fps=fps
            )
    
    def _annotate_frame_cpu(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        tracks: List[TrackingResult],
        show_traces: bool = True,
        show_detection_boxes: bool = True,
        show_tracking_boxes: bool = True,
        show_keypoints: bool = True,
        show_masks: bool = True,
        mask_alpha: float = 0.5,
        fps: Optional[float] = None
    ) -> np.ndarray:
        """Original CPU-based frame annotation (moved from annotate_frame)."""
        if not frame.data or frame.size == 0:
            print("Warning: Empty frame provided to annotate_frame")
            return frame
            
        result = frame.copy()
        
        try:
            # Get class names, using default if visualizer is not initialized
            class_names = self.get_class_names()
            
            # Debug print to check tracks data
            # if tracks:
            #     print(f"Annotating frame with {len(tracks)} tracks")
            #     # More robust debug print that checks attribute existence
            #     first_track = tracks[0]
            #     track_id = first_track.track_id if hasattr(first_track, 'track_id') else 'unknown'
            #     # Get class_id from detection if available
            #     class_id = first_track.detection.class_id if hasattr(first_track, 'detection') and hasattr(first_track.detection, 'class_id') else 'unknown'
            #     # Get bbox from either track directly or its detection
            #     bbox = None
            #     if hasattr(first_track, 'bbox'):
            #         bbox = first_track.bbox
            #     elif hasattr(first_track, 'detection') and hasattr(first_track.detection, 'bbox'):
            #         bbox = first_track.detection.bbox
            #     print(f"First track sample: ID={track_id}, Class={class_id}, BBox={bbox}")
            
            # DEBUG: Log tracking data for troubleshooting
            if tracks:
                self.logger.info(f"VisualizationManager: Processing {len(tracks)} tracks")
                for i, track in enumerate(tracks[:3]):  # Log first 3 tracks
                    has_detection = hasattr(track, 'detection') and track.detection is not None
                    has_bbox = has_detection and hasattr(track.detection, 'bbox') and track.detection.bbox is not None
                    self.logger.info(f"Track {i}: ID={track.track_id}, has_detection={has_detection}, has_bbox={has_bbox}")

            # Convert our TrackingResult objects to Supervision Detections for annotation
            if show_tracking_boxes and tracks and len(tracks) > 0:
                # Extract bounding boxes and necessary data
                boxes = []
                confidence_values = []
                class_ids = []
                tracker_ids = []
                
                for track in tracks:
                    # Robust bbox extraction with validation
                    bbox = None
                    
                    # Method 1: Direct bbox (shouldn't exist for TrackingResult)
                    if hasattr(track, 'bbox') and track.bbox is not None:
                        bbox = track.bbox
                    
                    # Method 2: From detection (primary method for TrackingResult)
                    elif hasattr(track, 'detection') and track.detection is not None:
                        if hasattr(track.detection, 'bbox') and track.detection.bbox is not None:
                            bbox = track.detection.bbox
                    
                    # Skip track if no valid bbox found
                    if bbox is None:
                        self.logger.info(f"Skipping track {track.track_id}: no valid bbox found")
                        continue
                    
                    # Validate bbox format [x1, y1, x2, y2]
                    if not (isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) == 4):
                        self.logger.info(f"Skipping track {track.track_id}: invalid bbox format {bbox}")
                        continue
                    
                    # Validate bbox values are numeric
                    try:
                        x1, y1, x2, y2 = [float(coord) for coord in bbox]
                        if not all(coord >= 0 for coord in [x1, y1, x2, y2]) or x2 <= x1 or y2 <= y1:
                            self.logger.info(f"Skipping track {track.track_id}: invalid bbox coordinates {bbox}")
                            continue
                    except (ValueError, TypeError):
                        self.logger.info(f"Skipping track {track.track_id}: non-numeric bbox {bbox}")
                        continue
                    
                    boxes.append(bbox)
                    
                    # Get confidence from detection or default to 1.0
                    if hasattr(track, 'detection') and track.detection and hasattr(track.detection, 'confidence'):
                        confidence_values.append(track.detection.confidence)
                    else:
                        confidence_values.append(1.0)
                    
                    # Get class_id from detection or default to 0 (person)
                    if hasattr(track, 'detection') and track.detection and hasattr(track.detection, 'class_id'):
                        class_ids.append(track.detection.class_id)
                    else:
                        class_ids.append(0)
                    
                    # Get track_id
                    tracker_ids.append(track.track_id)
                
                # Only proceed if we have valid boxes
                if boxes:
                    # Create Supervision Detections object
                    sv_tracked_detections = sv.Detections(
                        xyxy=np.array(boxes),
                        confidence=np.array(confidence_values),
                        class_id=np.array(class_ids),
                        tracker_id=np.array(tracker_ids)
                    )
                    
                    # Create labels only for tracks with valid bboxes (those in boxes list)
                    labels = []
                    valid_track_count = 0
                    for track in tracks:
                        # Check if this track has a valid bbox (was added to boxes list)
                        track_has_valid_bbox = False
                        
                        # Method 1: Direct bbox (shouldn't exist for TrackingResult)
                        if hasattr(track, 'bbox') and track.bbox is not None:
                            track_has_valid_bbox = True
                        # Method 2: From detection (primary method for TrackingResult)
                        elif hasattr(track, 'detection') and track.detection is not None:
                            if hasattr(track.detection, 'bbox') and track.detection.bbox is not None:
                                bbox = track.detection.bbox
                                # Validate bbox format and values
                                if (isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) == 4):
                                    try:
                                        x1, y1, x2, y2 = [float(coord) for coord in bbox]
                                        if all(coord >= 0 for coord in [x1, y1, x2, y2]) and x2 > x1 and y2 > y1:
                                            track_has_valid_bbox = True
                                    except (ValueError, TypeError):
                                        pass
                        
                        if track_has_valid_bbox and hasattr(track, 'track_id'):
                            # Get class from detection
                            class_id = 0  # Default to person (0)
                            if hasattr(track, 'detection') and track.detection and hasattr(track.detection, 'class_id'):
                                class_id = track.detection.class_id
                            
                            # Use class_id as index only if it's a valid index in class_names
                            if 0 <= class_id < len(class_names):
                                class_name = class_names[class_id]
                            else:
                                class_name = "Unknown"
                                
                            confidence = track.detection.confidence if hasattr(track, 'detection') and track.detection and hasattr(track.detection, 'confidence') else 1.0
                            label_text = f"{class_name} #{track.track_id} {confidence:.2f}"
                            labels.append(label_text)
                            valid_track_count += 1
                    
                    # Apply annotations in sequence like YOLOrun_dub.py
                    result = self.box_annotator.annotate(scene=result, detections=sv_tracked_detections)
                    result = self.label_annotator.annotate(scene=result, detections=sv_tracked_detections, labels=labels)
                    
                    # Draw custom traces using stored bottom-centre points instead of centroid
                    if show_traces:
                        result = self._draw_custom_traces(result, tracks)
            
            # Draw detection boxes if enabled
            if show_detection_boxes and detections:
                class_names = self.get_class_names()
                for detection in detections:
                    x1, y1, x2, y2 = map(int, detection.bbox)
                    # Use list indexing with bounds checking
                    if 0 <= detection.class_id < len(class_names):
                        class_name = class_names[detection.class_id]
                    else:
                        class_name = f"class_{detection.class_id}"
                    
                    # Draw bounding box
                    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{class_name} {detection.confidence:.2f}"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(result, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw keypoints if enabled
            if show_keypoints:
                for track in tracks:
                    if hasattr(track, 'keypoints') and track.keypoints is not None:
                        result = self._draw_keypoints(result, track.keypoints)
                    elif hasattr(track, 'detection') and track.detection and hasattr(track.detection, 'keypoints') and track.detection.keypoints is not None:
                        result = self._draw_keypoints(result, track.detection.keypoints)
                
                # Also draw keypoints from detections if available
                for detection in detections:
                    if detection.keypoints is not None:
                        result = self._draw_keypoints(result, detection.keypoints)
            
            # Draw masks if enabled
            if show_masks and detections:
                print(f"VisualizationManager: Processing {len(detections)} detections for masks")
                result = self._draw_masks(result, detections, alpha=mask_alpha)
            
            # Update and draw FPS
            if fps is not None:
                self.fps_history.append(fps)
                if len(self.fps_history) > self.fps_window_size:
                    self.fps_history.pop(0)
                
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                # Only use visualizer.add_fps_counter if visualizer is not None
                if self.visualizer:
                    result = self.visualizer.add_fps_counter(result, avg_fps)
                else:
                    # Simple fallback for FPS display
                    cv2.putText(
                        result,
                        f"FPS: {avg_fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )
                
            return result
            
        except Exception as e:
            import traceback
            print(f"Error in annotate_frame: {e}")
            traceback.print_exc()
            return frame.copy()  # Return original frame on error
    
    def _draw_keypoints(
        self,
        frame: np.ndarray,
        keypoints,
        confidence_threshold: float = 0.5,
        radius: int = 4,
        color: Tuple[int, int, int] = (0, 255, 255)
    ) -> np.ndarray:
        """Draw pose keypoints on the frame.
        
        Args:
            frame: Input frame
            keypoints: Array or tuple of keypoints. Could be:
                       - np.ndarray with shape (N, 3) where each row is x, y, confidence
                       - Tuple of (xy_array, conf_array) in the format from YOLOv8-pose
            confidence_threshold: Minimum confidence to draw keypoint
            radius: Radius of keypoint circles
            color: Color of keypoints (BGR format)
            
        Returns:
            Frame with drawn keypoints
        """
        if frame is None or not frame.data:
            return frame
            
        result = frame.copy()
        
        try:
            # Handle the case where keypoints come as a tuple of (xy_array, conf_array)
            if isinstance(keypoints, tuple) and len(keypoints) == 2:
                xy_array, conf_array = keypoints
                
                # Check if both arrays exist and have valid shapes
                if xy_array is not None and conf_array is not None:
                    # Draw each keypoint with sufficient confidence
                    for i in range(len(xy_array)):
                        if i < len(conf_array) and conf_array[i] > confidence_threshold:
                            x, y = xy_array[i]
                            cv2.circle(
                                result,
                                (int(x), int(y)),
                                radius,
                                color,
                                -1
                            )
                    
            # Handle the case where keypoints is a numpy array with shape (N, 3)
            elif hasattr(keypoints, 'shape') and len(keypoints.shape) == 2 and keypoints.shape[1] == 3:
                for kp in keypoints:
                    x, y, conf = kp
                    if conf > confidence_threshold:
                        cv2.circle(
                            result,
                            (int(x), int(y)),
                            radius,
                            color,
                            -1
                        )
        except Exception as e:
            print(f"Error drawing keypoints: {e}")
            import traceback
            traceback.print_exc()
        
        return result 

    def _draw_custom_traces(
        self,
        frame: np.ndarray,
        tracks: List[TrackingResult],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw trace lines using bottom-centre points stored in Track.positions.

        Args:
            frame: Image to draw on.
            tracks: List of TrackingResult objects.
            color: Default colour if per-track colour cannot be determined.
            thickness: Line thickness.

        Returns:
            Annotated image.
        """
        if frame is None or frame.size == 0:
            return frame

        out = frame.copy()

        # Determine colour palette (falls back to green)
        palette = None
        if hasattr(self, "visualizer") and self.visualizer and hasattr(self.visualizer, "colors"):
            palette = self.visualizer.colors

        for track in tracks:
            if not hasattr(track, "positions") or len(track.positions) < 2:
                continue

            # Choose colour based on class_id if palette available
            if palette and hasattr(track, "detection") and track.detection is not None:
                try:
                    col = palette[track.detection.class_id % len(palette)]
                except Exception:
                    col = color
            else:
                col = color

            pts = list(track.positions)
            for p, q in zip(pts[:-1], pts[1:]):
                cv2.line(out, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), col, thickness)

        return out 
    
    def _draw_masks(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        alpha: Optional[float] = None
    ) -> np.ndarray:
        """Draw segmentation masks on the frame with class labels
        
        Args:
            frame: Input frame
            detections: List of DetectionResult objects with masks
            alpha: Mask transparency (0.0-1.0)
            
        Returns:
            Frame with mask overlays and class labels
        """
        if alpha is None:
            alpha = 0.5  # Default transparency
        
        result = frame.copy()
        
        # Generate colors for different detections
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        masks_drawn = 0
        frame_height, frame_width = frame.shape[:2]
        class_names = self.get_class_names()
        
        for i, detection in enumerate(detections):
            if detection.mask is not None:
                mask = detection.mask
                color = colors[i % len(colors)]
                
                # Debug logging
                print(f"Drawing mask {i}: shape={mask.shape}, dtype={mask.dtype}, min={np.min(mask)}, max={np.max(mask)}")
                print(f"Frame shape: {frame.shape}, mask shape: {mask.shape}")
                
                # Resize mask to match frame dimensions if needed
                if mask.shape != (frame_height, frame_width):
                    print(f"Resizing mask from {mask.shape} to ({frame_height}, {frame_width})")
                    mask_resized = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_resized = mask
                
                # Create colored overlay
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                for c in range(3):
                    colored_mask[:, :, c] = mask_binary * color[c]
                
                # Blend with original frame
                mask_area = mask_binary > 0
                if np.any(mask_area):
                    result[mask_area] = cv2.addWeighted(
                        result[mask_area], 
                        1 - alpha, 
                        colored_mask[mask_area], 
                        alpha, 
                        0
                    )
                    masks_drawn += 1
                    
                    # Add class label to the mask
                    self._add_mask_label(result, detection, mask_binary, color, class_names)
        
        if masks_drawn > 0:
            print(f"VisualizationManager: Successfully drew {masks_drawn} masks with labels")
        
        return result
    
    def _add_mask_label(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
        mask_binary: np.ndarray,
        mask_color: Tuple[int, int, int],
        class_names: List[str]
    ) -> None:
        """Add class label to a segmentation mask
        
        Args:
            frame: Frame to draw on (modified in-place)
            detection: Detection result with class information
            mask_binary: Binary mask to find centroid for label placement
            mask_color: Color of the mask for label background
            class_names: List of class names
        """
        try:
            # Get class name
            if 0 <= detection.class_id < len(class_names):
                class_name = class_names[detection.class_id]
            else:
                class_name = f"class_{detection.class_id}"
            
            # Get confidence
            confidence = detection.confidence if hasattr(detection, 'confidence') else 1.0
            
            # Create label text
            label_text = f"{class_name} {confidence:.2f}"
            
            # Find centroid of mask for label placement
            mask_coords = np.where(mask_binary > 0)
            if len(mask_coords[0]) > 0:
                centroid_y = int(np.mean(mask_coords[0]))
                centroid_x = int(np.mean(mask_coords[1]))
                
                # Text properties
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, font, font_scale, font_thickness
                )
                
                # Position label near centroid but ensure it's visible
                label_x = max(5, min(centroid_x - text_width // 2, frame.shape[1] - text_width - 5))
                label_y = max(text_height + 5, min(centroid_y, frame.shape[0] - baseline - 5))
                
                # Draw background rectangle with mask color (semi-transparent)
                bg_color = tuple(int(c * 0.8) for c in mask_color)  # Darker version of mask color
                cv2.rectangle(
                    frame,
                    (label_x - 3, label_y - text_height - 3),
                    (label_x + text_width + 3, label_y + baseline + 3),
                    bg_color,
                    -1
                )
                
                # Draw text with contrasting color
                text_color = (255, 255, 255) if sum(bg_color) < 384 else (0, 0, 0)
                cv2.putText(
                    frame,
                    label_text,
                    (label_x, label_y),
                    font,
                    font_scale,
                    text_color,
                    font_thickness
                )
                
                print(f"Added mask label: {label_text} at ({label_x}, {label_y})")
                
        except Exception as e:
            print(f"Error adding mask label: {e}")
            import traceback
            traceback.print_exc()
    
    def encode_frame_hardware(self, frame: np.ndarray) -> Optional[bytes]:
        """Encode frame using hardware acceleration if available.
        
        Args:
            frame: Frame to encode (numpy array)
            
        Returns:
            Encoded JPEG bytes or None if encoding failed
        """
        if self.nvjpeg_encoder:
            try:
                # Convert to tensor if needed
                if isinstance(frame, np.ndarray):
                    # Convert HWC to CHW
                    frame_chw = frame.transpose(2, 0, 1)
                    # Convert to tensor
                    frame_tensor = torch.from_numpy(frame_chw).to(self.device, dtype=torch.float32) / 255.0
                else:
                    frame_tensor = frame
                
                # Encode using hardware
                return self.nvjpeg_encoder.encode_tensor(frame_tensor)
                
            except Exception as e:
                self.logger.debug(f"Hardware encoding failed: {e}")
        
        # Fallback to software encoding
        return None
    
    def encode_video_frame_hardware(self, frame: np.ndarray) -> bool:
        """Encode video frame using NVENC hardware acceleration.
        
        Args:
            frame: Frame to encode
            
        Returns:
            True if frame was successfully queued for encoding
        """
        if self.nvenc_encoder:
            try:
                # Convert to tensor if needed
                if isinstance(frame, np.ndarray):
                    # Convert HWC to CHW
                    frame_chw = frame.transpose(2, 0, 1)
                    # Convert to tensor
                    frame_tensor = torch.from_numpy(frame_chw).to(self.device, dtype=torch.float32) / 255.0
                else:
                    frame_tensor = frame
                
                # Queue frame for encoding
                return self.nvenc_encoder.encode_tensor(frame_tensor)
                
            except Exception as e:
                self.logger.debug(f"Video encoding failed: {e}")
        
        return False
    
    def get_encoded_video_data(self, timeout: float = 0.1) -> Optional[bytes]:
        """Get encoded video data from NVENC encoder.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Encoded video data or None
        """
        if self.nvenc_encoder:
            return self.nvenc_encoder.get_encoded_frame(timeout)
        return None
    
    def cleanup(self):
        """Clean up GPU resources and stop encoders."""
        if self.nvenc_encoder:
            try:
                self.nvenc_encoder.stop()
                self.logger.info("Stopped NVENC encoder")
            except Exception as e:
                self.logger.error(f"Error stopping NVENC encoder: {e}")
        
        if self.gpu_visualizer:
            # GPU visualizer cleanup if needed
            pass
    
    def get_visualization_stats(self) -> Dict[str, Any]:
        """Get visualization performance statistics.
        
        Returns:
            Dictionary of performance stats
        """
        stats = {}
        
        if self.gpu_visualizer:
            stats['gpu_visualizer'] = self.gpu_visualizer.get_stats()
        
        if self.nvjpeg_encoder:
            stats['nvjpeg_encoder'] = self.nvjpeg_encoder.get_stats()
        
        if self.nvenc_encoder:
            stats['nvenc_encoder'] = self.nvenc_encoder.get_stats()
        
        stats['use_gpu'] = self.use_gpu
        stats['fps_avg'] = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        return stats