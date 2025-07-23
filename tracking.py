from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np
import time
import cv2
import supervision as sv
from collections import deque
import threading
import os

from models import DetectionResult, TrackingResult, Point, convert_numpy_types


class TrackingSystem:
    """Manages object tracking using ByteTrack"""
    
    def __init__(
        self, 
        tracker_config: Optional[Dict[str, Any]] = None,
        inactive_threshold_seconds: float = 1.0,
        trace_persistence_seconds: float = 5.0
    ):
        """Initialize the tracking system
        
        Args:
            tracker_config: Configuration for the ByteTrack tracker
            inactive_threshold_seconds: Time threshold to mark tracks as inactive
            trace_persistence_seconds: Time to keep inactive traces
        """
        self.tracker_config = tracker_config or {}
        self.inactive_threshold_seconds = inactive_threshold_seconds
        self.trace_persistence_seconds = trace_persistence_seconds
        self.tracker = None
        self.active_tracks: Dict[Tuple[str, int], TrackingResult] = {}  # (camera_id, track_id) -> TrackingResult
        self.inactive_tracks: Dict[Tuple[str, int], Dict[str, Any]] = {}  # (camera_id, track_id) -> inactive track data
        self.lock = threading.Lock()
    
    def initialize_tracker(self):
        """Initialize the ByteTrack tracker"""
        if self.tracker is not None:
            return self.tracker
            
        try:
            # Default configuration
            default_config = {
                "track_thresh": 0.25,  # Low threshold to create tracks
                "track_buffer": 30,    # Frames to keep track without detection
                "match_thresh": 0.75,   # IoU threshold for matching
                "frame_rate": 30,      # Assuming 30 FPS
            }
            
            # Override with user config
            config = {**default_config, **self.tracker_config}
            
            # Create the tracker
            self.tracker = sv.ByteTrack(
                track_activation_threshold=config.get("track_thresh", 0.25),
                lost_track_buffer=config.get("track_buffer", 90),
                minimum_matching_threshold=config.get("match_thresh", 0.75),
                frame_rate=config.get("frame_rate", 30)
            )
            print("[TrackingSystem] ByteTrack initialized successfully.")
            return self.tracker
            
        except Exception as e:
            print(f"[TrackingSystem Error] Failed to initialize ByteTrack: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def update(
        self, 
        detections: List[DetectionResult], 
        frame: np.ndarray,
        camera_id: str,
        keypoints_list: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        feature_vectors: Optional[List[np.ndarray]] = None
    ) -> List[TrackingResult]:
        """Update tracking with new detections
        
        Args:
            detections: List of detection results
            frame: The current frame
            camera_id: Camera identifier
            keypoints_list: Optional list of keypoints for each detection
            feature_vectors: Optional list of feature vectors for each detection
            
        Returns:
            List[TrackingResult]: Updated tracking results
        """
        if not self.initialize_tracker():
            return []
            
        try:
            # Skip if no detections
            if not detections:
                # --- ADDED LOG --- 
                # print(f"[TrackingSystem-{camera_id}] No detections received.")
                self._update_inactive_tracks(time.time()) # Still need to update inactive tracks even if no new detections
                return []
                
            # --- ADDED LOG --- __removed log for now as tracking is super stable.
            #print(f"[TrackingSystem-{camera_id}] Received {len(detections)} detections.")
            
            # Convert to Supervision Detections format
            sv_detections = sv.Detections(
                xyxy=np.array([d.bbox for d in detections]),
                confidence=np.array([d.confidence for d in detections]),
                class_id=np.array([d.class_id for d in detections])
            )
            
            # Store original indices for later mapping
            # Creating a custom data dictionary if it doesn't exist
            if not hasattr(sv_detections, 'data') or sv_detections.data is None:
                sv_detections.data = {}
            sv_detections.data['original_index'] = np.arange(len(detections))
            
            # Update tracker
            tracked_detections = self.tracker.update_with_detections(detections=sv_detections)
            
            # --- ADDED LOG --- ___removed for now as tracking is super stable.
            #print(f"[TrackingSystem-{camera_id}] ByteTrack returned {len(tracked_detections)} tracked detections.")
            #if len(tracked_detections) > 0:
            #     print(f"[TrackingSystem-{camera_id}] Track IDs: {tracked_detections.tracker_id}")
            # --- END ADDED LOG ---
            
            # Convert to our TrackingResult format
            tracking_results = []
            frame_height, frame_width = frame.shape[:2]
            current_time = time.time()
            
            # --- REVISED LOGIC: Map tracked IDs back to original detections using IoU --- 
            active_track_ids_in_frame = set()
            matched_original_indices = set()
            
            # Build a map of tracker_id to its data for quick lookup
            track_id_map = {}
            if len(tracked_detections) > 0:
                for i, tracker_id in enumerate(tracked_detections.tracker_id):
                     track_id_map[tracker_id] = {
                         'xyxy': tracked_detections.xyxy[i],
                         'confidence': tracked_detections.confidence[i],
                         'class_id': tracked_detections.class_id[i]
                     }
            
            # Iterate through original detections and try to match them to tracked results
            for original_idx, original_detection in enumerate(detections):
                best_match_iou = -1.0
                matched_tracker_id = None
                matched_track_data = None

                # Find the best matching track for this original detection
                for tracker_id, track_data in track_id_map.items():
                    iou = self._calculate_iou(original_detection.bbox, track_data['xyxy'])
                    # Use a reasonable IoU threshold for matching (e.g., 0.5)
                    if iou > 0.5 and iou > best_match_iou:
                        best_match_iou = iou
                        matched_tracker_id = tracker_id
                        matched_track_data = track_data

                # If a good match was found
                if matched_tracker_id is not None:
                    # Mark original index as matched to avoid double-matching
                    if original_idx in matched_original_indices:
                         continue # Already matched this original detection
                    matched_original_indices.add(original_idx)
                    
                    # Get keypoints and features for this original detection
                    keypoints = keypoints_list[original_idx] if keypoints_list and original_idx < len(keypoints_list) else None
                    feature_vector = feature_vectors[original_idx] if feature_vectors and original_idx < len(feature_vectors) else None
                    
                    track_key = (camera_id, int(matched_tracker_id))
                    active_track_ids_in_frame.add(track_key)
                    
                    with self.lock:
                        if track_key in self.active_tracks:
                            # Update existing active track
                            track = self.active_tracks[track_key]
                            track.detection = original_detection # Update with latest detection data
                            track.detection.keypoints = keypoints
                            track.detection.feature_vector = feature_vector
                            track.last_seen = current_time
                            # Use bottom-centre of the bounding box instead of geometric centre
                            x1, y1, x2, y2 = original_detection.bbox
                            center_x = (x1 + x2) / 2.0
                            center_y = y2  # bottom edge (closer to the floor)
                            track.positions.append((center_x, center_y))
                            if keypoints: track.keypoints_history.append(keypoints)
                            if feature_vector is not None: track.feature_history.append(feature_vector)
                            # Update velocity (simple example)
                            if len(track.positions) >= 2: 
                                prev_pos = track.positions[-2]; curr_pos = track.positions[-1]
                                vx = (curr_pos[0] - prev_pos[0]) * 30 # Assuming 30 FPS
                                vy = (curr_pos[1] - prev_pos[1]) * 30
                                track.velocity = (vx, vy)
                            tracking_results.append(track)
                        
                        elif track_key in self.inactive_tracks:
                            # Reinstate inactive track
                            inactive_data = self.inactive_tracks.pop(track_key)
                            track = TrackingResult(
                                track_id=int(matched_tracker_id),
                                camera_id=camera_id,
                                detection=original_detection, # Use current detection data
                                state=inactive_data.get("state", "Active"),
                                last_seen=current_time,
                                positions=deque(inactive_data.get("positions", []), maxlen=50),
                                keypoints_history=deque(inactive_data.get("keypoints_history", []), maxlen=50),
                                feature_history=deque(inactive_data.get("feature_history", []), maxlen=50),
                                velocity=inactive_data.get("velocity", (0.0, 0.0)),
                                zone=inactive_data.get("zone", None),
                                dwell_time=inactive_data.get("dwell_time", 0.0)
                            )
                            # Use bottom-centre of the bounding box instead of geometric centre
                            x1, y1, x2, y2 = original_detection.bbox
                            center_x = (x1 + x2) / 2.0
                            center_y = y2  # bottom edge (closer to the floor)
                            track.positions.append((center_x, center_y))
                            if keypoints: track.keypoints_history.append(keypoints)
                            if feature_vector is not None: track.feature_history.append(feature_vector)
                            self.active_tracks[track_key] = track
                            tracking_results.append(track)
                            
                        else:
                            # Create new track
                            track = TrackingResult(
                                track_id=int(matched_tracker_id),
                                camera_id=camera_id,
                                detection=original_detection
                            )
                            
                            # Validate TrackingResult creation
                            if track.detection is None:
                                print(f"WARNING: Track {track.track_id} created with None detection")
                            elif not hasattr(track.detection, 'bbox') or track.detection.bbox is None:
                                print(f"WARNING: Track {track.track_id} has detection but no bbox")
                            
                            # Initialize zone based on camera ID
                            if "living" in camera_id.lower() or "rtsp_0" == camera_id:
                                track.zone = "Living Room"
                            elif "kitchen" in camera_id.lower() or "rtsp_1" == camera_id:
                                track.zone = "Kitchen"
                            
                            # Use bottom-centre of the bounding box instead of geometric centre
                            x1, y1, x2, y2 = original_detection.bbox
                            center_x = (x1 + x2) / 2.0
                            center_y = y2  # bottom edge (closer to the floor)
                            
                            track.positions.append((center_x, center_y))
                            if keypoints: track.keypoints_history.append(keypoints)
                            if feature_vector is not None: track.feature_history.append(feature_vector)
                            self.active_tracks[track_key] = track
                            tracking_results.append(track)
            # --- END REVISED LOGIC ---

            # Update inactive tracks (handle tracks that were not seen in this frame)
            with self.lock:
                inactive_candidate_keys = set(self.active_tracks.keys()) - active_track_ids_in_frame
                for track_key in inactive_candidate_keys:
                    if track_key[0] == camera_id: # Only consider tracks for the current camera
                         track = self.active_tracks.pop(track_key)
                         # Check if should be moved to inactive or deleted
                         if current_time - track.last_seen < self.inactive_threshold_seconds:
                             # Keep active for a bit longer (grace period)
                             self.active_tracks[track_key] = track 
                         else:
                             # Move to inactive
                             self.inactive_tracks[track_key] = {
                                "last_seen": track.last_seen,
                                "positions": list(track.positions),
                                "keypoints_history": list(track.keypoints_history),
                                "feature_history": list(track.feature_history),
                                "velocity": track.velocity,
                                "zone": track.zone,
                                "dwell_time": track.dwell_time,
                                "state": "Inactive"
                            }
                            #print(f"[TrackingSystem] Track {track_key} moved to inactive.")
            
            # Prune very old inactive tracks
            self._update_inactive_tracks(current_time)
            
            # --- ADDED LOG --- ___removed for now as tracking is super stable.
            #print(f"[TrackingSystem-{camera_id}] Update finished. Active tracks: {len(self.active_tracks)}. Returning {len(tracking_results)} results.")
            # --- END ADDED LOG ---
            
            return tracking_results

        except Exception as e:
            print(f"[TrackingSystem Error] Error during update: {e}")
            import traceback
            traceback.print_exc()
            return []
            
    def _update_inactive_tracks(self, current_time: float):
        """Remove inactive tracks that exceed the persistence time."""
        with self.lock:
            to_remove = []
            for track_key, inactive_data in self.inactive_tracks.items():
                if current_time - inactive_data['last_seen'] > self.trace_persistence_seconds:
                    to_remove.append(track_key)
            
            if to_remove:
                for key in to_remove:
                    del self.inactive_tracks[key]
                    print(f"[TrackingSystem] Pruned inactive track {key}")

    def get_active_tracks(self) -> List[TrackingResult]:
        """Get currently active tracks
        
        Returns:
            List[TrackingResult]: Active tracking results
        """
        with self.lock:
            # Filter out tracks that haven't been seen recently (over 3 seconds)
            current_time = time.time()
            active_tracks = []
            for track_key, track in list(self.active_tracks.items()):
                # If the track hasn't been seen in the last 3 seconds (adjust as needed),
                # move it to inactive
                if current_time - track.last_seen > 3.0:
                    # Move to inactive
                    self.inactive_tracks[track_key] = {
                        "last_seen": track.last_seen,
                        "positions": list(track.positions),
                        "keypoints_history": list(track.keypoints_history) if hasattr(track, 'keypoints_history') else [],
                        "feature_history": list(track.feature_history) if hasattr(track, 'feature_history') else [],
                        "velocity": track.velocity if hasattr(track, 'velocity') else (0.0, 0.0),
                        "zone": track.zone if hasattr(track, 'zone') else None,
                        "dwell_time": track.dwell_time if hasattr(track, 'dwell_time') else 0.0,
                        "state": "Inactive"
                    }
                    # Remove from active tracks
                    del self.active_tracks[track_key]
                    print(f"[TrackingSystem] Track {track_key} moved to inactive (get_active_tracks).")
                else:
                    active_tracks.append(track)
            
            return active_tracks
    
    def get_active_tracks_data(self) -> List[Dict[str, Any]]:
        """Get serializable data for currently active tracks.
        
        Returns:
            List[Dict[str, Any]]: List of track data dictionaries
        """
        active_tracks = self.get_active_tracks()  # This now ensures tracks are truly active
        serializable_tracks = []
        
        for track in active_tracks:
            # Convert TrackingResult object to a serializable dictionary
            track_data = {
                "track_id": track.track_id,
                "camera_id": track.camera_id,
                "state": track.state,
                "last_seen": track.last_seen,
                # Convert positions deque to list for serialization
                "positions": list(track.positions) if hasattr(track, 'positions') and track.positions else [],
                # Get bbox if available (from detection)
                "bbox": track.detection.bbox if hasattr(track, 'detection') and track.detection else None,
                # Get center point if available (from detection)
                "center": track.detection.center if hasattr(track, 'detection') and track.detection else None,
                # Add other relevant fields
                "velocity": getattr(track, 'velocity', (0.0, 0.0)),
                "zone": getattr(track, 'zone', None),
                "dwell_time": getattr(track, 'dwell_time', 0.0),
            }
            
            # Convert any numpy types to Python native types
            serializable_tracks.append(convert_numpy_types(track_data))
            
        return serializable_tracks
    
    def get_occupancy_data(self) -> Dict[str, Any]:
        """Get occupancy data (counts of tracks by zone).
        
        Returns:
            Dict[str, Any]: Dictionary with zone names as keys and counts as values
        """
        # Initialize with zero counts for main zones to ensure they're always present
        occupancy = {
            'Living Room': 0,
            'Kitchen': 0
        }
        
        active_tracks = self.get_active_tracks()
        
        # Count tracks by zone
        for track in active_tracks:
            zone = getattr(track, 'zone', None)
            camera_id = track.camera_id
            
            # Use camera_id for null/unknown zones (either "Living Room" or "Kitchen")
            if not zone or zone == "Outside":
                if "living" in camera_id.lower() or "rtsp_0" == camera_id:
                    occupancy['Living Room'] += 1
                elif "kitchen" in camera_id.lower() or "rtsp_1" == camera_id:
                    occupancy['Kitchen'] += 1
            else:
                # If zone is set, increment that zone's count
                occupancy[zone] = occupancy.get(zone, 0) + 1
            
        return occupancy
    
    def get_inactive_tracks(self) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """Get recently inactive tracks
        
        Returns:
            Dict[Tuple[str, int], Dict[str, Any]]: Inactive tracks data
        """
        with self.lock:
            return self.inactive_tracks.copy()
    
    def get_all_track_ids(self) -> Set[Tuple[str, int]]:
        """Get all active and inactive track IDs
        
        Returns:
            Set[Tuple[str, int]]: Set of (camera_id, track_id) tuples
        """
        with self.lock:
            return set(self.active_tracks.keys()) | set(self.inactive_tracks.keys())
    
    def clear(self):
        """Clear all tracking data"""
        with self.lock:
            self.active_tracks.clear()
            self.inactive_tracks.clear()
            if self.tracker:
                try:
                    # Reset ByteTrack (if method exists)
                    if hasattr(self.tracker, 'reset') and callable(self.tracker.reset):
                        self.tracker.reset()
                except Exception as e:
                    print(f"[TrackingSystem Warning] Failed to reset tracker: {e}")
                    # Recreate tracker as fallback
                    self.tracker = None
                    self.initialize_tracker() 

    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes
        
        Args:
            box1: First bounding box (x1, y1, x2, y2)
            box2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            float: IoU score
        """
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate area of intersection rectangle
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        intersection = width * height
        
        # Calculate area of both bounding boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate union area
        union = box1_area + box2_area - intersection
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        
        return iou 