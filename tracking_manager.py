"""
tracking_manager.py
Main tracking management system that coordinates zones, tracks, and transitions.
"""
import time
import itertools
import json
from collections import deque
import numpy as np
import supervision as sv
import cv2

from database import DB_FILENAME, DatabaseManager
from geometry import Position, Zone, ZoneTransition
from track import Track, STATE_UNKNOWN
from metrics import calculate_reid_similarity
from utils import convert_np_for_json, filter_detections_by_exclusion_zones, COORD_SYS_WIDTH, COORD_SYS_HEIGHT


# Constants for Re-ID
INACTIVE_THRESHOLD_SECONDS = 30.0  # Base seconds without update to mark track potentially inactive (increased from 2.0s)
MIN_CONFIDENCE_THRESHOLD = 0.3    # Minimum confidence score for a detection
REID_MAX_TIME_GAP = 20.0          # Seconds before an inactive track is not considered for Re-ID (increased from 5.0)
REID_NEGATIVE_CONSTRAINT_THRESHOLD = 0.70  # Min score to suppress Re-ID if matches an active track (lowered to be less strict)
REID_THRESHOLD_INCREASE_FACTOR = 0.05  # Factor to increase threshold based on time gap (halved to be more lenient with time)
REID_CONFIDENCE_FACTOR = 0.5  # Influence of detection confidence on score
REID_SIMILARITY_THRESHOLD = 0.35  # Threshold to consider a Re-ID match (lowered to be more lenient)
REID_MAX_SPATIAL_DISTANCE = COORD_SYS_WIDTH / 3.0  # Max distance for spatial score normalization (increased)


class TrackingManager:
    """Main class for managing tracking and zones."""
    
    def __init__(self, db_path=DB_FILENAME):
        """Initialize the TrackingManager.

        Args:
            db_path (str, optional): Path to the SQLite database file. Defaults to DB_FILENAME.
        """
        self.db_path = db_path
        self.db_manager = DatabaseManager(db_path)
        
        # Dictionary to hold active tracks, keyed by (camera, track_id) tuple
        self.tracks = {}  # e.g., self.tracks[('Living Room', 123)] = Track object

        # Dictionary holding Zone objects, keyed by CAMERA_ID, then ZONE_NAME
        self.zones = {
            'Living Room': {
                # Full Field of View for Living Room
                'Living Room Full': Zone('Living Room Full', 0, 0, COORD_SYS_WIDTH, COORD_SYS_HEIGHT),
            },
            'Kitchen': {
                # Full Field of View for Kitchen
                'Kitchen Full': Zone('Kitchen Full', 0, 0, COORD_SYS_WIDTH, COORD_SYS_HEIGHT),
            }
        }
        
        # Store camera dimensions for heatmap normalization
        self.camera_dimensions = {}  # Stores {'camera_id': {'width': w, 'height': h}}

        # Exclusion zones
        self.exclusion_zones = {}  # Will be keyed by camera ID, then zone name

        # Deque to store recent zone transitions (limited history in memory)
        self.zone_transitions = deque(maxlen=100)  # Keep last 100 transitions in memory
        
        # Re-ID related members
        self.recently_inactive_tracks = {}  # Key: (camera, track_id), Value: Track object
        self.reid_check_interval = 5  # Seconds: How often to clear very old inactive tracks
        self.last_reid_cleanup = time.time()
        
        # Configuration for Re-ID scores
        self.reid_config = {
            'weight_oks': 0.35,
            'weight_spatial': 0.25,
            'weight_appearance': 0.30,
            'weight_state': 0.10,
            'max_spatial_distance': REID_MAX_SPATIAL_DISTANCE,
            'appearance_threshold': 0.5,
            'similarity_threshold': REID_SIMILARITY_THRESHOLD,
            'max_time_gap': REID_MAX_TIME_GAP,
            'negative_constraint_threshold': REID_NEGATIVE_CONSTRAINT_THRESHOLD,
            'threshold_increase_factor': REID_THRESHOLD_INCREASE_FACTOR,
            'confidence_factor': REID_CONFIDENCE_FACTOR
        }

        # Validate weights sum approximately to 1
        total_weight = (self.reid_config['weight_oks'] + 
                        self.reid_config['weight_spatial'] + 
                        self.reid_config['weight_appearance'] +
                        self.reid_config['weight_state'])
        if not np.isclose(total_weight, 1.0):
            print("[Re-ID Warning] Score weights do not sum to 1.0! Check configuration.")
    
    def set_exclusion_zones(self, zones_dict, camera_id=None):
        """Set zones to exclude from tracking (e.g., TV screens).
        
        Args:
            zones_dict: Dictionary of zone_name -> (x1, y1, x2, y2) coordinates
            camera_id: Optional camera ID to which these exclusion zones apply.
                      If None, the zones apply to all cameras.
        """
        if camera_id:
            # Set exclusion zones for a specific camera
            if camera_id not in self.exclusion_zones:
                self.exclusion_zones[camera_id] = {}
            
            # Add or update each zone
            for zone_name, coords in zones_dict.items():
                self.exclusion_zones[camera_id][zone_name] = coords
            
            print(f"Set {len(zones_dict)} exclusion zones for camera '{camera_id}': {list(zones_dict.keys())}")
        else:
            # Set global exclusion zones (apply to all cameras)
            for cam_id in self.zones.keys():
                if cam_id not in self.exclusion_zones:
                    self.exclusion_zones[cam_id] = {}
                
                # Add or update each zone
                for zone_name, coords in zones_dict.items():
                    self.exclusion_zones[cam_id][zone_name] = coords
            
            print(f"Set {len(zones_dict)} global exclusion zones: {list(zones_dict.keys())}")
    
    def get_speed(self):
        """Return a default speed value when no specific track speed is available."""
        return 0.0  # Default speed when no track is specified

    def update_track_history(self, tracks_data, camera, frame_width, frame_height, keypoints_list, features_list, confidence_list):
        """Updates the state of tracks based on new detection data from a camera.
        
        Args:
            tracks_data (list): List of dicts [{track_id:.., box:.., ...}] detections
            camera (str): Camera identifier (e.g., 'Living Room')
            frame_width (int): Width of the video frame
            frame_height (int): Height of the video frame
            keypoints_list (list): List of keypoints data matching track indices
            features_list (list): List of feature vectors matching track indices
            confidence_list (list): List of confidence scores matching track indices
        """
        try:
            # Update camera dimensions if needed
            if camera not in self.camera_dimensions or \
               self.camera_dimensions[camera]['width'] != frame_width or \
               self.camera_dimensions[camera]['height'] != frame_height:
                print(f"[TrackingManager] Updating dimensions for {camera} to {frame_width}x{frame_height}")
                self.camera_dimensions[camera] = {'width': frame_width, 'height': frame_height}
            
            # Get current time once for this batch
            current_time = time.time()
            
            # Track which IDs are updated in this cycle
            updated_track_ids = set()
            reidentified_ids = set()  # Tracks matched via re-identification
            
            # Process Active Tracks and Re-Identification
            new_detections = []  # Detections that don't match existing active tracks
            matched_to_active = set()  # Keep track of detection indices matched to active tracks

            # Iterate through current detections
            for i, det_data in enumerate(tracks_data):
                track_id = det_data['track_id']
                bbox = det_data['box']  # Box is [x1, y1, x2, y2]
                confidence = det_data.get('confidence', 1.0)
                class_id = det_data.get('class_id')
                
                # Get corresponding aux data (handle index errors)
                keypoints = keypoints_list[i] if i < len(keypoints_list) else None
                feature = features_list[i] if i < len(features_list) else None
                original_confidence = confidence_list[i] if i < len(confidence_list) else confidence

                track_key = (camera, track_id)  # Unique key for the global tracks dict
                updated_track_ids.add(track_key)
                matched_to_active.add(i)  # Mark this detection index as matched

                # Calculate center x, y
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                scaled_x = center_x
                scaled_y = center_y

                if track_key in self.tracks:
                    # Update Existing Active Track
                    track = self.tracks[track_key]
                    # Update position, keypoints, features, etc.
                    pos = track.update_position(scaled_x, scaled_y, current_time)
                    track.last_bbox = bbox  # Update last bbox
                    track.add_keypoints(keypoints, current_time)
                    track.add_feature(feature, current_time)
                    track.add_confidence(original_confidence, current_time)
                    track.update_velocity()  # Update velocity based on new position
                else:
                    # Handle Potentially New Track (or Re-Identified Inactive Track)
                    # Add it to the list to be checked against inactive tracks later
                    new_detections.append({
                        'det_index': i,  # Store original index for later reference
                        'track_id': track_id,  # Original tracker ID (might change if re-ID)
                        'bbox': bbox,
                        'confidence': confidence,
                        'class_id': class_id,
                        'center': (scaled_x, scaled_y),
                        'keypoints': keypoints,
                        'feature': feature,
                        'original_confidence': original_confidence
                    })

            # Try to Re-Identify Unmatched Detections with Inactive Tracks
            unmatched_new_detections = []  # Detections that are truly new
            if new_detections and self.recently_inactive_tracks:
                # First try cross-camera re-identification for each new detection
                for new_det_idx, new_det in enumerate(new_detections):
                    # Check if this detection is in a camera transition area
                    cross_camera_match, matched_key, match_score = self._check_cross_camera_reid(
                        new_det, camera, current_time
                    )
                    
                    if cross_camera_match and matched_key in self.recently_inactive_tracks:
                        # We have a cross-camera match!
                        cross_camera_track = self.recently_inactive_tracks.pop(matched_key)
                        original_id = new_det['track_id']
                        cross_camera_id = cross_camera_track.track_id
                        
                        # Print useful info about the cross-camera match
                        print(f"[Cross-Camera ReID] Match found! Camera {camera} det {original_id} -> "
                              f"Camera {cross_camera_track.camera} track {cross_camera_id} (score: {match_score:.2f})")
                        
                        # Create a new track ID for the cross-camera track
                        # The camera will change but we keep the original track ID for continuity
                        new_key = (camera, cross_camera_id)
                        
                        # Check if new_key already exists in active tracks (shouldn't normally happen)
                        if new_key in self.tracks:
                            print(f"[Cross-Camera Warning] Track ID {cross_camera_id} already exists in {camera}. "
                                  f"Treating detection as new instead.")
                            unmatched_new_detections.append(new_det)
                            continue
                            
                        # Update the track with the new camera and position
                        cross_camera_track.camera = camera  # Update camera to current one
                        
                        # Update the track with the new detection data
                        pos = cross_camera_track.update_position(
                            new_det['center'][0], new_det['center'][1], current_time
                        )
                        cross_camera_track.last_bbox = new_det['bbox']
                        cross_camera_track.add_keypoints(new_det['keypoints'], current_time)
                        cross_camera_track.add_feature(new_det['feature'], current_time)
                        cross_camera_track.add_confidence(new_det['original_confidence'], current_time)
                        cross_camera_track.update_velocity()
                        
                        # Add the track to active tracks with the new camera key
                        self.tracks[new_key] = cross_camera_track
                        updated_track_ids.add(new_key)
                        reidentified_ids.add(new_key)
                        
                        # Add transition event for analytics
                        self._handle_zone_transition(
                            cross_camera_track, 
                            f"{matched_key[0]} Exit", 
                            f"{camera} Entry"
                        )
                    else:
                        # No cross-camera match, add to list for regular ReID processing
                        unmatched_new_detections.append(new_det)
                        
                # For detections that didn't match across cameras, try regular same-camera ReID
                final_unmatched = []
                
                # Build lists for efficient similarity calculation
                inactive_track_keys = list(self.recently_inactive_tracks.keys())
                inactive_tracks = list(self.recently_inactive_tracks.values())

                # Pre-calculate data for inactive tracks if needed
                inactive_features = [t.get_latest_feature() for t in inactive_tracks]
                inactive_kps = [t.get_latest_smoothed_keypoints() for t in inactive_tracks]
                inactive_bboxes = [t.last_bbox for t in inactive_tracks]
                inactive_states = [t.get_latest_state() for t in inactive_tracks]
                inactive_times = [t.last_seen for t in inactive_tracks]

                # Iterate through detections that weren't matched to active tracks
                for new_det in unmatched_new_detections:
                    best_match_score = -1.0
                    best_match_inactive_idx = -1
                    best_match_threshold = -1.0  # Store threshold for best match

                    # Get data for the new detection
                    det_feature = new_det['feature']
                    det_kps = new_det['keypoints']
                    det_bbox = new_det['bbox']
                    det_state = detect_state(det_kps[0], det_kps[1], 0) if det_kps else STATE_UNKNOWN
                    det_conf = new_det['original_confidence']

                    # Compare against all potentially suitable inactive tracks
                    for inactive_idx, inactive_track in enumerate(inactive_tracks):
                        # Basic Check: Must be from the same camera
                        if inactive_track.camera != camera:
                            continue
                        
                        # Calculate time gap
                        time_gap = current_time - inactive_times[inactive_idx]
                        if time_gap > self.reid_config['max_time_gap']:
                            continue  # Too much time has passed

                        # Calculate Similarity Score
                        score, scores_debug = calculate_reid_similarity(
                            inactive_kps[inactive_idx],
                            det_kps,
                            inactive_bboxes[inactive_idx],
                            det_bbox,
                            inactive_features[inactive_idx],
                            det_feature,
                            inactive_states[inactive_idx],
                            det_state,
                            time_gap,
                            det_conf,
                            self.reid_config
                        )

                        # Negative Constraint Check
                        # Check if this detection strongly matches any active track
                        should_suppress = False
                        for active_key, active_track in self.tracks.items():
                            if active_track.camera != camera:
                                continue  # Only compare within the same camera
                                
                            active_score, _ = calculate_reid_similarity(
                                active_track.get_latest_smoothed_keypoints(),
                                det_kps,
                                active_track.last_bbox,
                                det_bbox,
                                active_track.get_latest_feature(),
                                det_feature,
                                active_track.get_latest_state(),
                                det_state,
                                0,  # Time gap is 0 for active tracks
                                det_conf,
                                self.reid_config
                            )
                            if active_score > self.reid_config['negative_constraint_threshold']:
                                should_suppress = True
                                break  # Stop checking other active tracks
                        
                        if should_suppress:
                            continue  # Skip this inactive track if suppression is active

                        # Calculate dynamic threshold based on time gap
                        dynamic_threshold = self.reid_config['similarity_threshold'] + \
                                          (self.reid_config['threshold_increase_factor'] * 
                                           (time_gap / self.reid_config['max_time_gap']))
                        dynamic_threshold = min(dynamic_threshold, 0.95)  # Cap threshold 

                        # Update best match if current score is higher and exceeds threshold
                        if score > best_match_score and score > dynamic_threshold:
                            best_match_score = score
                            best_match_inactive_idx = inactive_idx
                            best_match_threshold = dynamic_threshold
                    
                    # Apply Match if Score Exceeded Dynamic Threshold
                    if best_match_inactive_idx != -1:
                        # Match found!
                        matched_inactive_key = inactive_track_keys[best_match_inactive_idx]

                        # Check if key still exists before popping
                        if matched_inactive_key in self.recently_inactive_tracks:
                            matched_inactive_track = self.recently_inactive_tracks.pop(matched_inactive_key)
                            original_tracker_id = new_det['track_id']
                            reid_id = matched_inactive_track.track_id

                            # Update the matched track with the new detection data
                            pos = matched_inactive_track.update_position(
                                new_det['center'][0], new_det['center'][1], current_time
                            )
                            matched_inactive_track.last_bbox = new_det['bbox']
                            matched_inactive_track.add_keypoints(new_det['keypoints'], current_time)
                            matched_inactive_track.add_feature(new_det['feature'], current_time)
                            matched_inactive_track.add_confidence(new_det['original_confidence'], current_time)
                            matched_inactive_track.update_velocity()

                            # Add the re-identified track back to the main active tracks dictionary
                            self.tracks[matched_inactive_key] = matched_inactive_track
                            updated_track_ids.add(matched_inactive_key)
                            reidentified_ids.add(matched_inactive_key)
                        else:
                            # The track was already popped (likely by another match in the same batch)
                            print(f"[Re-ID Warning] Inactive track {matched_inactive_key} already popped. "
                                  f"Treating Det {new_det['track_id']} as new.")
                            final_unmatched.append(new_det)
                    else:
                        # No suitable inactive match found, this is likely a truly new track
                        final_unmatched.append(new_det)
                
                # Use the final unmatched list
                unmatched_new_detections = final_unmatched
            else:
                # No new detections or no inactive tracks to check against
                unmatched_new_detections.extend(new_detections)

            # Create New Tracks for remaining unmatched detections
            for new_det in unmatched_new_detections:
                # Use the tracker's assigned ID for the new track
                new_track_id = new_det['track_id']
                new_track_key = (camera, new_track_id)
                
                # Avoid creating duplicate track if ID somehow reappeared
                if new_track_key in self.tracks:
                    track = self.tracks[new_track_key]
                else:
                    track = Track(new_track_id, camera)
                    self.tracks[new_track_key] = track
                
                # Initialize the new track with current detection data
                pos = track.update_position(new_det['center'][0], new_det['center'][1], current_time)
                track.last_bbox = new_det['bbox']
                track.add_keypoints(new_det['keypoints'], current_time)
                track.add_feature(new_det['feature'], current_time)
                track.add_confidence(new_det['original_confidence'], current_time)
                track.update_velocity()
                updated_track_ids.add(new_track_key)

            # Manage Inactive Tracks
            # Identify tracks that weren't updated in this cycle
            current_active_keys = set(self.tracks.keys())
            all_keys_this_cycle = updated_track_ids
            lost_track_keys = current_active_keys - all_keys_this_cycle

            for track_key in lost_track_keys:
                if track_key in self.tracks:  # Check if it wasn't just re-identified and removed
                    track = self.tracks.pop(track_key)  # Remove from active tracks
                    # Add to recently inactive dictionary for potential Re-ID later
                    self.recently_inactive_tracks[track_key] = track
            
            # Update Zones and Store Data for all updated tracks
            for track_key in updated_track_ids:
                if track_key in self.tracks:  # Ensure track exists
                    track = self.tracks[track_key]
                    position = track.get_latest_position()
                    if position:
                        # Determine current zone based on latest position
                        zone_obj = self.get_zone(position, track.camera)
                        new_zone_name = zone_obj.name if zone_obj else "Outside"
                        previous_zone_name = track.current_zone
                        
                        # Update track's current zone and handle transitions
                        if track.update_zone(new_zone_name):
                            self._handle_zone_transition(track, previous_zone_name, new_zone_name)
                    
                    # Store the updated track data
                    self._store_track(track)
            
            # Periodic Cleanup of OLD Inactive Tracks
            if current_time - self.last_reid_cleanup > self.reid_check_interval:
                cutoff_time = current_time - self.reid_config['max_time_gap'] * 1.5
                keys_to_remove = [
                    key for key, track in self.recently_inactive_tracks.items() 
                    if track.last_seen < cutoff_time
                ]
                if keys_to_remove:
                    print(f"[Re-ID Cleanup] Removing {len(keys_to_remove)} very old inactive tracks.")
                    for key in keys_to_remove:
                        del self.recently_inactive_tracks[key]
                self.last_reid_cleanup = current_time

            # Final Occupancy Update
            self._update_occupancy()

        except Exception as e:
            print(f"[Error] Unexpected error in update_track_history: {e}")
            import traceback; traceback.print_exc()

    def _handle_zone_transition(self, track, previous_zone_name, new_zone_name):
        """Handles the logic when a track moves between zones.

        Args:
            track (Track): The track object that moved.
            previous_zone_name (str | None): The name of the zone the track came from.
            new_zone_name (str): The name of the new zone entered.
        """
        # Record entry time if entering a defined zone (not 'Outside')
        if new_zone_name != "Outside":
            track.zone_entry_times[new_zone_name] = track.last_seen

        # Use 'Outside' if previous_zone_name was None for the transition log
        from_zone_log = previous_zone_name if previous_zone_name is not None else 'Outside'

        # Create transition object using zone names
        transition = ZoneTransition(track.track_id, from_zone_log, new_zone_name, 
                                   track.camera, track.last_seen)

        # Add to recent transitions deque (in-memory)
        self.zone_transitions.append(transition)
        
        # Store transition persistently in the database
        self.db_manager.store_transition(transition)
    
    def _store_track(self, track):
        """Store or update track data in the database."""
        try:
            # Convert dict to be JSON serializable
            track_data_dict = track.to_dict()
            track_data_serializable = convert_np_for_json(track_data_dict)
            track_data_json = json.dumps(track_data_serializable)
            
            # Store in database
            self.db_manager.store_track(track, track_data_json)
        except Exception as e:
            print(f"[Error] Unexpected error in _store_track: {e}")
    
    def _update_occupancy(self):
        """Update occupancy counts for all camera-specific zones."""
        # Reset all zone occupancy counts to 0 first
        for camera_zones in self.zones.values():
            for zone in camera_zones.values():
                zone.reset_occupancy()  # Use the new method

        # Count active tracks in each zone (with proper camera-zone tracking)
        for track_key_tuple, track in self.tracks.items():
            # Validate key format before unpacking
            if not isinstance(track_key_tuple, tuple) or len(track_key_tuple) != 2:
                continue
            
            camera, track_id = track_key_tuple

            if camera not in self.zones:  # Skip if zones aren't defined for this camera
                continue
            
            camera_zone_definitions = self.zones[camera]

            # Check if track is active and currently assigned to a specific zone
            if track.is_active() and track.current_zone and track.current_zone != "Outside":
                # Increment the occupancy count for that specific Zone object
                if track.current_zone in camera_zone_definitions:
                    camera_zone_definitions[track.current_zone].increment_occupancy()
                else:
                    print(f"[Occupancy Warning] Track {camera}_{track_id} has zone '{track.current_zone}' which is not defined for this camera")
    
    def _update_heatmap(self):
        """Update heatmap data in the database."""
        points_to_update = []  # List of (normalized_x, normalized_y) tuples
        try:
            # Collect all current positions from active tracks
            current_time = time.time()
            for track_key_tuple, track in self.tracks.items():
                # Validate key format before unpacking
                if not isinstance(track_key_tuple, tuple) or len(track_key_tuple) != 2:
                    continue
                camera, track_id = track_key_tuple

                # Consider a track active if recently updated
                if track.is_active(threshold=INACTIVE_THRESHOLD_SECONDS):
                    pos = track.get_latest_position()
                    if pos:
                        # Quantize positions to a grid (e.g., 20x20 pixels)
                        grid_size = 20
                        grid_x = max(0, min(COORD_SYS_WIDTH - grid_size, 
                                         int(pos.x / grid_size) * grid_size))
                        grid_y = max(0, min(COORD_SYS_HEIGHT - grid_size, 
                                         int(pos.y / grid_size) * grid_size))
                        points_to_update.append((grid_x, grid_y))

            if points_to_update:
                self.db_manager.update_heatmap(points_to_update)
        except Exception as e:
            print(f"[Error] Unexpected error in _update_heatmap: {e}")
    
    def get_zone(self, position, camera_id):
        """Determine which zone a position belongs to for a specific camera."""
        # Look up zones specific to the given camera
        if camera_id in self.zones:
            camera_zones = self.zones[camera_id]
            for zone in camera_zones.values():
                if zone.contains(position):
                    return zone  # Return the Zone object
        return None  # No containing zone found for this camera
    
    def get_track_stats(self):
        """Get statistics for all currently active tracked objects."""
        stats = {}
        current_time = time.time()
        
        for track_key_tuple, track in self.tracks.items():
            # Validate key format before unpacking
            if not isinstance(track_key_tuple, tuple) or len(track_key_tuple) != 2:
                continue
            
            camera, track_id = track_key_tuple

            # Only include tracks considered currently active
            if not track.is_active(threshold=INACTIVE_THRESHOLD_SECONDS):
                continue
            
            # Get the latest known position
            position = track.get_latest_position()
            if not position:
                continue
            
            # Construct the unique key combining camera and track ID
            track_key_str = f"{camera}_{track_id}"
            
            # Calculate current dwell time in the current zone
            current_dwell = track.get_dwell_time(track.current_zone) if track.current_zone else 0
            
            # Build the dictionary for this track
            stats[track_key_str] = {
                "track_id": track.track_id,
                "camera": camera,
                "direction": track.format_direction(),
                "speed": round(track.get_speed(), 1),
                "avg_speed": round(track.get_average_speed(), 1),
                "zone": track.current_zone,
                "last_seen": track.last_seen,
                "position": [round(position.x), round(position.y)],
                "dwell_time": round(current_dwell, 1),
                "state": track.get_latest_state(),
                "confidence": track.get_latest_confidence(),
            }
        
        return stats
    
    def get_recent_transitions(self, limit=50):
        """Get recent zone transitions."""
        # Return transitions from the in-memory deque
        return [t.to_dict() for t in reversed(list(self.zone_transitions))][:limit]

    def get_occupancy(self):
        """Get current occupancy across all camera-specific zones by directly counting active tracks.
        This simplified approach ensures consistency with the active tracks display.
        Always includes entries for main zones even if they're empty.
        """
        # Initialize with zero counts for main zones to ensure they're always present
        flat_occupancy = {
            'Living Room Full': 0,
            'Kitchen Full': 0
        }
        
        # For each active track, increment its zone counter based on its camera
        for track_key_tuple, track in self.tracks.items():
            # Skip tracks that aren't active
            if not track.is_active():
                continue
            
            # Extract camera from track key
            if not isinstance(track_key_tuple, tuple) or len(track_key_tuple) != 2:
                continue
            
            camera, track_id = track_key_tuple
            
            # Increment zone counter based on camera, regardless of track's current zone
            # This ensures tracks are counted in their respective camera's zone
            if 'Living Room' in camera:
                flat_occupancy['Living Room Full'] += 1
            elif 'Kitchen' in camera:
                flat_occupancy['Kitchen Full'] += 1
        
        return flat_occupancy
    
    def get_heatmap_data(self, date_filter=None):
        """Get aggregated heatmap data for visualization."""
        return self.db_manager.get_heatmap_data(date_filter)

    def get_trails(self, limit_per_track=15):
        """Get recent trail data for active tracks for visualization."""
        trails = {}
        for track_key_tuple, track in self.tracks.items():
            # Validate key format before unpacking
            if not isinstance(track_key_tuple, tuple) or len(track_key_tuple) != 2:
                continue
            camera, track_id = track_key_tuple

            # Only include active tracks with enough points
            if not track.is_active(threshold=INACTIVE_THRESHOLD_SECONDS) or len(track.positions) < 2:
                continue
            
            # Get the last N positions from the track's history deque
            recent_positions = list(track.positions)[-limit_per_track:]
            
            # Format positions into simple {x, y} dictionaries
            formatted_trail = [{'x': round(p.x), 'y': round(p.y)} for p in recent_positions]
            
            # Use the combined key for the frontend
            track_key_str = f"{camera}_{track_id}"
            trails[track_key_str] = formatted_trail
        
        return trails

    def get_historical_data(self, start_time=None, end_time=None, limit=100):
        """Get historical tracking data from database."""
        return self.db_manager.get_historical_data(start_time, end_time, limit)
    
    def cleanup_old_data(self, days=7):
        """Remove track history and heatmap data older than specified days."""
        return self.db_manager.cleanup_old_data(days)

    def clear_stats(self):
        """Clears in-memory statistics like occupancy and recent transitions."""
        # Clear occupancy counts in zone objects
        for camera_zones in self.zones.values():
            for zone in camera_zones.values():
                zone.occupancy = 0
        
        # Clear the recent transitions deque
        self.zone_transitions.clear()

    def filter_detections_by_exclusion_zones(self, detections: sv.Detections, camera_id: str) -> sv.Detections:
        """Filters detections based on exclusion zones."""
        return filter_detections_by_exclusion_zones(detections, self.exclusion_zones, camera_id)

    def _check_cross_camera_reid(self, new_detection, camera_id, current_time):
        """Special handling for cross-camera ReID between Living Room and Kitchen
        
        This method implements special logic for tracking people as they move between
        the two camera views, taking into account the physical relationship between
        the spaces (e.g., Living Room left side connects to Kitchen right side).
        
        Args:
            new_detection (dict): Detection data including position, keypoints, features
            camera_id (str): Current camera ID
            current_time (float): Current timestamp
            
        Returns:
            tuple: (bool, track_key, score) - Whether match found, matched track key, and match score
        """
        # Only handle specific cameras we know about
        if camera_id not in ['Living Room', 'Kitchen']:
            return False, None, 0.0
            
        # Determine which camera would be linked for cross-camera tracking
        adjacent_camera = 'Kitchen' if camera_id == 'Living Room' else 'Living Room'
        
        # Get detection position (center)
        det_x, det_y = new_detection['center']
        
        # Check if detection is in the edge area where we'd expect cross-camera movement
        # For Living Room, check if detection is on left edge (transitions to Kitchen right edge)
        # For Kitchen, check if detection is on right edge (transitions to Living Room left edge)
        edge_threshold = 0.25 * COORD_SYS_WIDTH  # Use 25% of width as transition zone
        
        in_transition_zone = False
        if camera_id == 'Living Room' and det_x < edge_threshold:
            # Living Room left edge
            in_transition_zone = True
        elif camera_id == 'Kitchen' and det_x > (COORD_SYS_WIDTH - edge_threshold):
            # Kitchen right edge
            in_transition_zone = True
            
        if not in_transition_zone:
            # Not in a transition zone between cameras
            return False, None, 0.0
            
        # Look for potential matches in adjacent camera's inactive tracks
        best_match_key = None
        best_match_score = 0.0
        
        # Define threshold here, outside the loop
        cross_camera_threshold = self.reid_config['similarity_threshold'] * 0.9
        
        # Filter inactive tracks by adjacent camera and recency
        for track_key, track in self.recently_inactive_tracks.items():
            track_camera, track_id = track_key
            
            # Only consider tracks from the adjacent camera
            if track_camera != adjacent_camera:
                continue
                
            # Check recency - only consider tracks that disappeared recently
            time_gap = current_time - track.last_seen
            if time_gap > self.reid_config['max_time_gap']:
                continue
                
            # Check if track was in the corresponding edge zone
            last_pos = track.get_latest_position()
            if not last_pos:
                continue
                
            # Fix: Access Position object's attributes directly instead of unpacking
            track_x, track_y = last_pos.x, last_pos.y
            
            # Logic for corresponding edge zone
            in_corresponding_zone = False
            if adjacent_camera == 'Living Room' and track_x < edge_threshold:
                # Living Room left edge connects to Kitchen
                in_corresponding_zone = True
            elif adjacent_camera == 'Kitchen' and track_x > (COORD_SYS_WIDTH - edge_threshold):
                # Kitchen right edge connects to Living Room
                in_corresponding_zone = True
                
            if not in_corresponding_zone:
                continue
                
            # Prioritize ReID calculation for cross-camera matches
            reid_score = self._calculate_reid_score(
                new_detection,
                track,
                time_gap,
                increased_threshold=False,  # Don't increase threshold for cross-camera transitions
                edge_case=True  # Signal this is an edge case for special scoring
            )
            
            if reid_score > cross_camera_threshold and reid_score > best_match_score:
                best_match_key = track_key
                best_match_score = reid_score
        
        return best_match_score > cross_camera_threshold, best_match_key, best_match_score
    
    def _calculate_reid_score(self, detection, track, time_gap, increased_threshold=True, edge_case=False):
        """Calculate the Re-ID score between a detection and a track.
        
        Args:
            detection (dict): Detection data (with keypoints, features, etc.)
            track (Track): Track object to compare against
            time_gap (float): Time difference between detection and track
            increased_threshold (bool): Whether to increase threshold with time
            edge_case (bool): Whether this is a special case (e.g., cross-camera)
            
        Returns:
            float: Combined Re-ID score
        """
        # Calculate different similarity scores
        
        # Get keypoints for both
        det_keypoints = detection.get('keypoints')
        track_keypoints = track.get_latest_smoothed_keypoints()
        
        # Get feature vectors
        det_feature = detection.get('feature')
        track_feature = track.get_latest_feature()
        
        # Get state if available
        det_state = detection.get('state', STATE_UNKNOWN)
        track_state = track.state
        
        # Base scores - zero is no match, closer to 1 is better match
        oks_score = 0.0  # Object Keypoint Similarity
        spatial_score = 0.0  # Spatial proximity
        feature_score = 0.0  # Appearance similarity
        state_score = 0.0  # State similarity
        
        # Calculate keypoint similarity if available
        if det_keypoints is not None and track_keypoints is not None:
            oks_score = calculate_reid_similarity(det_keypoints, track_keypoints)
        
        # Calculate spatial proximity (higher for closer predictions)
        det_pos = detection['center']
        track_pos = track.get_latest_position()
        track_vel = track.velocity
        
        if det_pos and track_pos:
            # For cross-camera edge cases, we handle distance differently
            if edge_case:
                # For cross-camera, we care less about distance and more about appearance
                spatial_score = 0.7  # Default good score for cross-camera transitions
            else:
                # Normal spatial scoring
                distance = np.sqrt((det_pos[0] - track_pos[0])**2 + (det_pos[1] - track_pos[1])**2)
                
                # Calculate expected position using velocity
                if time_gap < 10.0 and track_vel:  # Limit how far we project
                    # Predict position based on velocity
                    pred_x = track_pos[0] + track_vel[0] * time_gap
                    pred_y = track_pos[1] + track_vel[1] * time_gap
                    
                    # Use predicted position if it's better
                    pred_distance = np.sqrt((det_pos[0] - pred_x)**2 + (det_pos[1] - pred_y)**2)
                    distance = min(distance, pred_distance)
                
                # Normalize distance to a similarity score
                max_distance = self.reid_config['max_spatial_distance']
                spatial_score = max(0.0, 1.0 - (distance / max_distance))
        
        # Calculate appearance similarity if features available
        if det_feature is not None and track_feature is not None:
            # Cosine similarity
            norm_det = np.linalg.norm(det_feature)
            norm_track = np.linalg.norm(track_feature)
            
            if norm_det > 0 and norm_track > 0:
                feature_score = np.dot(det_feature, track_feature) / (norm_det * norm_track)
                # Normalize from [-1,1] to [0,1]
                feature_score = (feature_score + 1) / 2
        
        # Calculate state similarity
        if det_state != STATE_UNKNOWN and track_state != STATE_UNKNOWN:
            state_score = 1.0 if det_state == track_state else 0.2
        else:
            # Default state score if unknown
            state_score = 0.5
        
        # Combined score with weighted components
        # Weights should sum to 1.0
        combined_score = (
            self.reid_config['weight_oks'] * oks_score +
            self.reid_config['weight_spatial'] * spatial_score +
            self.reid_config['weight_appearance'] * feature_score +
            self.reid_config['weight_state'] * state_score
        )
        
        # For edge cases (like cross-camera), we can apply different scoring
        if edge_case:
            # For cross-camera, prioritize appearance over spatial proximity
            combined_score = (
                (self.reid_config['weight_oks'] * 1.2) * oks_score +  # Boost keypoint importance
                (self.reid_config['weight_spatial'] * 0.5) * spatial_score +  # Reduce spatial importance
                (self.reid_config['weight_appearance'] * 1.5) * feature_score +  # Boost appearance importance
                self.reid_config['weight_state'] * state_score
            )
        
        # Adjust threshold based on time gap if needed
        if increased_threshold and time_gap > 1.0:
            # Increase threshold with time gap (harder to match the longer it's been)
            time_penalty = min(0.3, time_gap * self.reid_config['threshold_increase_factor'])
            # More time means higher threshold needed
            combined_score -= time_penalty
        
        # Confidence modifier - more confident detections have higher scores
        confidence = detection.get('original_confidence', 0.5)
        confidence_boost = (confidence - 0.5) * self.reid_config['confidence_factor']
        combined_score += confidence_boost
        
        return max(0.0, min(1.0, combined_score))  # Clamp to [0,1]


# Singleton instance for backward compatibility
_manager_instance = None

def initialize_tracking_manager(db_path=DB_FILENAME):
    """Initializes or reinitializes the global tracking manager instance."""
    global _manager_instance
    _manager_instance = TrackingManager(db_path=db_path)
    return _manager_instance


# Import this for backward compatibility
from track import detect_state 