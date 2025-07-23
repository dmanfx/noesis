from typing import List, Dict, Tuple, Optional
import numpy as np
from detection import Detection, Track

class Tracker:
    """Simple IoU-based object tracker."""
    
    def __init__(
        self,
        max_age: int = 30,
        min_iou: float = 0.3,
        min_confidence: float = 0.5
    ):
        """
        Initialize tracker.
        
        Args:
            max_age: Maximum number of frames a track can exist without matching detections
            min_iou: Minimum IoU for matching detections to tracks
            min_confidence: Minimum confidence threshold for detections
        """
        self.max_age = max_age
        self.min_iou = min_iou
        self.min_confidence = min_confidence
        self.tracks = []
        self.next_id = 1
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of new detections
        
        Returns:
            List of active tracks
        """
        # Filter detections based on confidence
        filtered_detections = [d for d in detections if d.confidence >= self.min_confidence]
        
        # Predict new locations of existing tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to existing tracks
        if len(self.tracks) > 0 and len(filtered_detections) > 0:
            matches, unmatched_tracks, unmatched_detections = self._match_detections_to_tracks(
                filtered_detections
            )
            
            # Update matched tracks
            for track_idx, detection_idx in matches:
                self.tracks[track_idx].update(filtered_detections[detection_idx])
            
            # Mark unmatched tracks as missed
            for track_idx in unmatched_tracks:
                self.tracks[track_idx].mark_missed()
            
            # Create new tracks for unmatched detections
            for detection_idx in unmatched_detections:
                self._initiate_track(filtered_detections[detection_idx])
        else:
            # If no tracks exist, create new tracks for all detections
            for detection in filtered_detections:
                self._initiate_track(detection)
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Return active tracks
        return [t for t in self.tracks if t.is_confirmed]
    
    def _initiate_track(self, detection: Detection) -> None:
        """
        Create a new track from a detection.
        
        Args:
            detection: Detection to create a track from
        """
        track = Track(detection, self.next_id, self.max_age)
        self.next_id += 1
        self.tracks.append(track)
    
    def _match_detections_to_tracks(
        self, 
        detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks based on IoU.
        
        Args:
            detections: List of new detections
        
        Returns:
            Tuple of (matches, unmatched_tracks, unmatched_detections)
        """
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(self.tracks))), list(range(len(detections)))
        
        # Calculate IoU between each detection and track
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for t, track in enumerate(self.tracks):
            for d, detection in enumerate(detections):
                iou_matrix[t, d] = self._calculate_iou(track.bbox, detection.bbox)
        
        # Use a greedy matching approach
        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))
        
        # Find matches greedily
        while True:
            # Find highest IoU
            if len(unmatched_tracks) == 0 or len(unmatched_detections) == 0:
                break
            
            # Get subset of the IoU matrix for unmatched tracks and detections
            sub_iou = iou_matrix[unmatched_tracks][:, unmatched_detections]
            
            # Find the best match
            if np.max(sub_iou) < self.min_iou:
                break
                
            # Get indices of max value
            t_idx, d_idx = np.unravel_index(np.argmax(sub_iou), sub_iou.shape)
            
            # Convert to original indices
            track_idx = unmatched_tracks[t_idx]
            det_idx = unmatched_detections[d_idx]
            
            # Add match and remove from unmatched lists
            matches.append((track_idx, det_idx))
            unmatched_tracks.remove(track_idx)
            unmatched_detections.remove(det_idx)
        
        return matches, unmatched_tracks, unmatched_detections
    
    @staticmethod
    def _calculate_iou(bbox1: Tuple[float, float, float, float], 
                        bbox2: Tuple[float, float, float, float]) -> float:
        """
        Calculate IoU between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
        
        Returns:
            IoU score between 0 and 1
        """
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou 