"""
track.py
Contains the Track class and related functionality for tracking objects.
"""
import numpy as np
import time
from collections import deque
from geometry import Position

# Constants
HISTORY_LENGTH = 50  # Max number of positions/keypoints/features to store per track
EMA_ALPHA = 0.3  # Smoothing factor for EMA on keypoints (lower value = more smoothing)
VELOCITY_SMOOTHING_WINDOW = 5  # Window for velocity averaging
VISIBILITY_THRESHOLD = 0.5  # Minimum confidence for a keypoint to be considered valid in OKS

# State definitions
STATE_UNKNOWN = "Unknown"
STATE_STANDING = "Standing"
STATE_SITTING = "Sitting"
STATE_WALKING = "Walking"
STATE_MIN_WALKING_SPEED = 50  # Pixels per second threshold to consider walking
STATE_SITTING_THRESHOLD_RATIO = 1.2  # Ratio shoulder_y / hip_y to consider sitting


def detect_state(keypoints_xy, keypoints_conf, current_speed):
    """
    Infers basic state (Standing, Sitting, Walking, Unknown) based on keypoints and speed.
    
    Args:
        keypoints_xy: Nx2 numpy array of smoothed keypoint coordinates [[x1, y1], ...].
        keypoints_conf: N numpy array of keypoint confidences [c1, ...].
        current_speed: Current estimated speed of the track (pixels/sec).
    
    Returns:
        str: One of STATE_STANDING, STATE_SITTING, STATE_WALKING, STATE_UNKNOWN.
    """
    if keypoints_xy is None or keypoints_conf is None or len(keypoints_xy) < 17:
        return STATE_UNKNOWN  # Need full set of keypoints

    # Keypoint indices (assuming standard COCO 17 keypoints)
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
    LEFT_HIP, RIGHT_HIP = 11, 12
    LEFT_KNEE, RIGHT_KNEE = 13, 14
    LEFT_ANKLE, RIGHT_ANKLE = 15, 16

    # Check visibility of crucial keypoints
    shoulder_vis = (keypoints_conf[LEFT_SHOULDER] > VISIBILITY_THRESHOLD or
                    keypoints_conf[RIGHT_SHOULDER] > VISIBILITY_THRESHOLD)
    hip_vis = (keypoints_conf[LEFT_HIP] > VISIBILITY_THRESHOLD or
               keypoints_conf[RIGHT_HIP] > VISIBILITY_THRESHOLD)

    if not (shoulder_vis and hip_vis):
        return STATE_UNKNOWN  # Cannot determine state without shoulders and hips

    # Get average shoulder and hip Y coordinates
    shoulder_y = np.mean([keypoints_xy[kp, 1] for kp in [LEFT_SHOULDER, RIGHT_SHOULDER] if keypoints_conf[kp] > VISIBILITY_THRESHOLD])
    hip_y = np.mean([keypoints_xy[kp, 1] for kp in [LEFT_HIP, RIGHT_HIP] if keypoints_conf[kp] > VISIBILITY_THRESHOLD])

    # Basic Walking Detection (based on speed)
    if current_speed > STATE_MIN_WALKING_SPEED:
        return STATE_WALKING

    # Basic Sitting vs Standing Detection
    if hip_y > 0:  # Avoid division by zero
        ratio = shoulder_y / hip_y
        if ratio > STATE_SITTING_THRESHOLD_RATIO:
            # Check if knees are also bent significantly
            knee_vis = (keypoints_conf[LEFT_KNEE] > VISIBILITY_THRESHOLD or keypoints_conf[RIGHT_KNEE] > VISIBILITY_THRESHOLD)
            ankle_vis = (keypoints_conf[LEFT_ANKLE] > VISIBILITY_THRESHOLD or keypoints_conf[RIGHT_ANKLE] > VISIBILITY_THRESHOLD)
            if knee_vis and ankle_vis:
                knee_y = np.mean([keypoints_xy[kp, 1] for kp in [LEFT_KNEE, RIGHT_KNEE] if keypoints_conf[kp] > VISIBILITY_THRESHOLD])
                ankle_y = np.mean([keypoints_xy[kp, 1] for kp in [LEFT_ANKLE, RIGHT_ANKLE] if keypoints_conf[kp] > VISIBILITY_THRESHOLD])
                # If knees are significantly higher than ankles, likely sitting
                if knee_y > ankle_y * 1.1:
                    return STATE_SITTING

    # Default to Standing if not Walking or Sitting (and keypoints were valid)
    return STATE_STANDING


class Track:
    """Class representing a tracked object."""
    def __init__(self, track_id, camera):
        self.track_id = track_id
        self.camera = camera
        self.positions = deque(maxlen=HISTORY_LENGTH)  # Store last N positions
        self.last_bbox = None  # Store the last known bounding box [x1, y1, x2, y2]
        self.keypoints_history = deque(maxlen=HISTORY_LENGTH)  # Store raw keypoints
        self.smoothed_keypoints_history = deque(maxlen=HISTORY_LENGTH)  # Store EMA smoothed keypoints
        self.feature_history = deque(maxlen=HISTORY_LENGTH)  # Store appearance features
        self.velocity_history = deque(maxlen=HISTORY_LENGTH)  # Store velocities
        self.ema_alpha = EMA_ALPHA  # EMA factor for smoothing
        self.velocity_smoothing_window = VELOCITY_SMOOTHING_WINDOW  # Window for velocity averaging
        self.confidence_history = deque(maxlen=HISTORY_LENGTH)  # Store detection confidence
        self.state_history = deque(maxlen=HISTORY_LENGTH)  # Store detected states
        self.state = STATE_UNKNOWN  # Current detected state
        self.current_zone = None
        self.last_seen = time.time()
        self.zone_entry_times = {}  # Zone name -> entry timestamp
    
    def update_position(self, x, y, timestamp=None):
        """Update position of the tracked object.
           NOTE: This only updates position. Keypoints/Features should be added separately.
           Call update_velocity() *after* this if tracking velocity.
        """
        timestamp = timestamp or time.time()
        position = Position(x, y, timestamp)
        self.positions.append(position)
        self.last_seen = timestamp  # Update last_seen with position timestamp
        return position
    
    def add_keypoints(self, keypoints_data, timestamp=None):
        """Adds keypoint data for the current position, with EMA smoothing.
           keypoints_data: Tuple of (keypoints_xy, keypoints_conf) where:
             - keypoints_xy: Array of shape (N, 2) containing (x, y) coords
             - keypoints_conf: Array of shape (N,) containing confidence values
        """
        timestamp = timestamp or self.last_seen
        
        # Convert NumPy arrays to lists for serialization
        if keypoints_data is not None and isinstance(keypoints_data, tuple) and len(keypoints_data) == 2:
            kpts_xy, kpts_conf = keypoints_data  # Unpack tuple
            
            # Ensure both components are converted to standard Python types
            if kpts_xy is not None:
                if isinstance(kpts_xy, np.ndarray):
                    kpts_xy = kpts_xy.tolist()  # Convert to regular Python list
                # Further ensure each point is also a list not a tuple
                if isinstance(kpts_xy, (list, tuple)) and kpts_xy and isinstance(kpts_xy[0], (list, tuple, np.ndarray)):
                    kpts_xy = [list(map(float, pt)) for pt in kpts_xy]
                    
            if kpts_conf is not None:
                if isinstance(kpts_conf, np.ndarray):
                    kpts_conf = kpts_conf.tolist()  # Convert to regular Python list
                # Ensure each confidence value is a float
                if isinstance(kpts_conf, (list, tuple)):
                    kpts_conf = list(map(float, kpts_conf))
            
            # Restructure as a clean tuple of Python lists
            keypoints_data = (kpts_xy, kpts_conf)
        
        # Store raw keypoint data
        self.keypoints_history.append({"timestamp": timestamp, "data": keypoints_data})
        
        # Apply EMA smoothing if previous keypoints exist
        if keypoints_data is None:
            # Just store None if no keypoints provided
            self.smoothed_keypoints_history.append({"timestamp": timestamp, "data": None})
            
            # Set state to UNKNOWN if no keypoints
            if self.state != STATE_UNKNOWN:
                self.state = STATE_UNKNOWN
                self.state_history.append({"timestamp": timestamp, "state": STATE_UNKNOWN})
                
            return
            
        # Unpack the current keypoints
        kpts_xy, kpts_conf = keypoints_data
        
        # If no previous smoothed keypoints, initialize with current
        if not self.smoothed_keypoints_history:
            self.smoothed_keypoints_history.append({"timestamp": timestamp, "data": keypoints_data})
            
            # Detect initial state
            new_state = detect_state(np.array(kpts_xy), np.array(kpts_conf), self.get_speed())
            self.state = new_state
            self.state_history.append({"timestamp": timestamp, "state": new_state})
            
            return
            
        # Get most recent smoothed keypoints
        prev_smoothed = self.smoothed_keypoints_history[-1]["data"]
        if prev_smoothed is None:
            # If previous was None, just use current directly
            self.smoothed_keypoints_history.append({"timestamp": timestamp, "data": keypoints_data})
            
            # Detect state after missing keypoints
            new_state = detect_state(np.array(kpts_xy), np.array(kpts_conf), self.get_speed())
            self.state = new_state
            self.state_history.append({"timestamp": timestamp, "state": new_state})
            
            return
            
        # Unpack previous smoothed keypoints
        prev_kpts_xy, prev_kpts_conf = prev_smoothed
        
        # Check dimensions match
        if kpts_xy is None or prev_kpts_xy is None or len(kpts_xy) != len(prev_kpts_xy):
            # Dimensions don't match, can't smooth - use current directly
            self.smoothed_keypoints_history.append({"timestamp": timestamp, "data": keypoints_data})
            
            # Detect state with unsmoothed data
            new_state = detect_state(np.array(kpts_xy), np.array(kpts_conf), self.get_speed())
            if new_state != self.state:
                self.state = new_state
                self.state_history.append({"timestamp": timestamp, "state": new_state})
                
            return
        
        # Apply EMA smoothing
        # Convert to numpy arrays for calculation
        kpts_xy_np = np.array(kpts_xy)
        prev_kpts_xy_np = np.array(prev_kpts_xy)
        kpts_conf_np = np.array(kpts_conf)
        
        # Apply EMA
        smoothed_xy = prev_kpts_xy_np + self.ema_alpha * (kpts_xy_np - prev_kpts_xy_np)
        
        # Handle confidence values - typically we don't smooth these but take the current ones
        smoothed_conf = kpts_conf if kpts_conf is not None else prev_kpts_conf
        
        # Convert back to lists before storing
        smoothed_xy_list = smoothed_xy.tolist() if isinstance(smoothed_xy, np.ndarray) else smoothed_xy
        smoothed_conf_list = smoothed_conf.tolist() if isinstance(smoothed_conf, np.ndarray) else smoothed_conf
            
        # Store smoothed data
        self.smoothed_keypoints_history.append({"timestamp": timestamp, "data": (smoothed_xy_list, smoothed_conf_list)})
                                             
        # Detect state using smoothed keypoints
        new_state = detect_state(smoothed_xy, kpts_conf_np, self.get_speed())
        if new_state != self.state:
            self.state = new_state
            self.state_history.append({"timestamp": timestamp, "state": new_state})

    def update_velocity(self):
        """Calculates velocity based on the last two positions and updates history."""
        if len(self.positions) < 2:
            return  # Not enough data to calculate velocity

        pos2 = self.positions[-1]  # Most recent
        pos1 = self.positions[-2]  # Second most recent

        time_diff = pos2.timestamp - pos1.timestamp
        if time_diff <= 0:
            vx, vy = 0.0, 0.0  # Avoid division by zero or negative time
        else:
            dx = pos2.x - pos1.x
            dy = pos2.y - pos1.y
            vx = dx / time_diff  # Pixels per second
            vy = dy / time_diff  # Pixels per second
        
        self.velocity_history.append((vx, vy))

    def get_latest_velocity(self):
        """Calculates the average velocity over the smoothing window."""
        if not self.velocity_history:
            return (0.0, 0.0)  # No velocity data

        # Get velocities within the window
        window_size = min(len(self.velocity_history), self.velocity_smoothing_window)
        recent_velocities = list(self.velocity_history)[-window_size:]
        
        if not recent_velocities:
            return (0.0, 0.0)
             
        # Calculate average velocity
        avg_vx = sum(v[0] for v in recent_velocities) / len(recent_velocities)
        avg_vy = sum(v[1] for v in recent_velocities) / len(recent_velocities)

        return (avg_vx, avg_vy)

    def predict_position(self, time_delta):
        """Predicts future position based on latest position and velocity."""
        if not self.positions:
            return None  # Cannot predict without a starting position

        latest_pos = self.positions[-1]
        latest_vx, latest_vy = self.get_latest_velocity()

        # Predict new coordinates
        predicted_x = latest_pos.x + latest_vx * time_delta
        predicted_y = latest_pos.y + latest_vy * time_delta

        # Create a new Position object for the prediction
        # Timestamp is estimated based on the last known position's timestamp
        predicted_timestamp = latest_pos.timestamp + time_delta 
        return Position(predicted_x, predicted_y, predicted_timestamp)
    
    def add_feature(self, feature_data, timestamp=None):
        """Adds appearance feature data for the most recent timestamp."""
        timestamp = timestamp or self.last_seen
        # More thorough conversion of NumPy arrays to lists
        if feature_data is not None:
            if isinstance(feature_data, np.ndarray):
                feature_data = feature_data.tolist()  # Basic conversion
            elif isinstance(feature_data, (list, tuple)):
                # Deep conversion if it's a list/tuple containing NumPy elements
                feature_data = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in feature_data]
        
        self.feature_history.append({"timestamp": timestamp, "data": feature_data})

    def add_confidence(self, confidence_score, timestamp=None):
        """Adds detection confidence score for the most recent timestamp."""
        timestamp = timestamp or self.last_seen
        self.confidence_history.append({"timestamp": timestamp, "confidence": confidence_score})

    def get_latest_smoothed_keypoints(self):
        """Returns the most recent smoothed keypoint data, or None if unavailable."""
        if not self.smoothed_keypoints_history:
            return None
        latest_entry = self.smoothed_keypoints_history[-1]
        return latest_entry["data"] if latest_entry else None
    
    def get_latest_feature(self):
        """Returns the most recent appearance feature vector, or None if unavailable."""
        if not self.feature_history:
            return None
        latest_entry = self.feature_history[-1]
        # Return as numpy array for calculations if stored as list
        feature_data = latest_entry["data"] if latest_entry else None
        return np.array(feature_data) if feature_data is not None else None
    
    def get_latest_confidence(self):
        """Returns the most recent confidence score, or None if history is empty."""
        if not self.confidence_history:
            return None
        return self.confidence_history[-1]["confidence"]  # Get confidence value

    def get_latest_state(self):
        """Returns the most recent detected state, or STATE_UNKNOWN if history is empty."""
        if not self.state_history:
            return STATE_UNKNOWN
        return self.state_history[-1]["state"]  # Get state value

    def get_direction(self):
        """Calculate direction of movement."""
        if len(self.positions) < 2:
            return None
            
        pos1 = self.positions[-2]
        pos2 = self.positions[-1]
        dx = pos2.x - pos1.x
        dy = pos2.y - pos1.y
        
        angle = np.arctan2(dy, dx)
        return angle
    
    def get_speed(self):
        """Calculate current speed in pixels per second."""
        if len(self.positions) < 2:
            return 0
            
        pos1 = self.positions[-2]
        pos2 = self.positions[-1]
        
        distance = np.sqrt((pos2.x - pos1.x)**2 + (pos2.y - pos1.y)**2)
        time_diff = pos2.timestamp - pos1.timestamp
        
        return distance / time_diff if time_diff > 0 else 0
    
    def get_average_speed(self, window=5):
        """Calculate average speed over the last 'window' positions."""
        if len(self.positions) < window + 1:
            return self.get_speed()
        
        speeds = []
        positions = list(self.positions)
        for i in range(-window, 0):
            if i+1 < 0:  # Make sure we have a pair of positions
                pos1 = positions[i]
                pos2 = positions[i+1]
                distance = pos1.distance_to(pos2)
                time_diff = pos2.timestamp - pos1.timestamp
                if time_diff > 0:
                    speeds.append(distance / time_diff)
        
        return sum(speeds) / len(speeds) if speeds else 0
    
    def get_dwell_time(self, zone_name):
        """Calculate time spent in a specific zone."""
        if zone_name not in self.zone_entry_times or self.current_zone != zone_name:
            return 0
        
        return time.time() - self.zone_entry_times[zone_name]
    
    def format_direction(self):
        """Convert direction from radians to human-readable format."""
        direction = self.get_direction()
        if direction is None:
            return "unknown"
            
        angles = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        index = int(((direction + np.pi) / (2 * np.pi / 8)) % 8)
        return angles[index]
    
    def get_latest_position(self):
        """Get the latest position."""
        if not self.positions:
            return None
        return self.positions[-1]
    
    def is_active(self, threshold=30):
        """Check if the track has been seen within the threshold (seconds)."""
        time_since_last_seen = time.time() - self.last_seen
        is_active = time_since_last_seen < threshold
        
        # Just return the result without logging
        return is_active
    
    def update_zone(self, new_zone):
        """Update the current zone and record entry time if changing zones."""
        zone_changed = (new_zone != self.current_zone)
        self.current_zone = new_zone  # Update the current zone name
        if zone_changed and new_zone is not None and new_zone != 'Outside':
            # If entering a new *defined* zone, record entry time
            self.zone_entry_times[new_zone] = time.time()
        # Return whether the zone actually changed
        return zone_changed
    
    def to_dict(self):
        """Convert Track object to a dictionary for serialization."""
        # Convert deques to lists for JSON compatibility
        positions_list = [p.to_dict() for p in self.positions]
        keypoints_history_list = list(self.keypoints_history)
        smoothed_keypoints_history_list = list(self.smoothed_keypoints_history)
        feature_history_list = list(self.feature_history)
        velocity_history_list = list(self.velocity_history)
        confidence_history_list = list(self.confidence_history)
        state_history_list = list(self.state_history)

        return {
            "track_id": self.track_id, 
            "camera": self.camera,
            "positions": positions_list,
            "keypoints_history": keypoints_history_list,
            "smoothed_keypoints_history": smoothed_keypoints_history_list,
            "feature_history": feature_history_list,
            "velocity_history": velocity_history_list,
            "confidence_history": confidence_history_list,
            "state_history": state_history_list,
            "ema_alpha": self.ema_alpha,
            "velocity_smoothing_window": self.velocity_smoothing_window,
            "current_zone": self.current_zone,
            "last_seen": self.last_seen,
            "zone_entry_times": self.zone_entry_times,
            "last_bbox": self.last_bbox
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create a Track object from a dictionary (e.g., loaded from DB/JSON)."""
        track = cls(data["track_id"], data["camera"])
        # Restore positions deque
        track.positions = deque(
            [Position.from_dict(p_data) for p_data in data.get("positions", [])],
            maxlen=HISTORY_LENGTH
        )
        # Restore keypoints history deque
        track.keypoints_history = deque(
            data.get("keypoints_history", []),
            maxlen=HISTORY_LENGTH
        )
        # Restore smoothed keypoints history deque
        track.smoothed_keypoints_history = deque(
            data.get("smoothed_keypoints_history", []),
            maxlen=HISTORY_LENGTH
        )
        # Restore feature history deque
        track.feature_history = deque(
            data.get("feature_history", []),
            maxlen=HISTORY_LENGTH
        )
        # Restore velocity history deque
        track.velocity_history = deque(
            data.get("velocity_history", []),
            maxlen=HISTORY_LENGTH
        )
        # Restore confidence history deque
        track.confidence_history = deque(
            data.get("confidence_history", []),
            maxlen=HISTORY_LENGTH
        )
        # Restore state history deque
        track.state_history = deque(
            data.get("state_history", []),
            maxlen=HISTORY_LENGTH
        )
        # Restore other attributes
        track.ema_alpha = data.get("ema_alpha", EMA_ALPHA)
        track.current_zone = data.get("current_zone")
        track.last_seen = data.get("last_seen", time.time())  # Default to now if missing
        track.zone_entry_times = data.get("zone_entry_times", {})
        track.last_bbox = data.get("last_bbox")

        # Ensure last_seen is updated if positions exist
        if track.positions:
            track.last_seen = max(track.last_seen, track.positions[-1].timestamp)

        return track 