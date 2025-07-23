"""
geometry.py
Contains geometric primitives used for tracking and zone management.
"""
import numpy as np
import time

class Position:
    """Class representing a position with timestamp."""
    def __init__(self, x, y, timestamp=None):
        self.x = x
        self.y = y
        self.timestamp = timestamp or time.time()
    
    @property
    def coords(self):
        """Return (x, y) tuple."""
        return (self.x, self.y)
    
    def distance_to(self, other):
        """Calculate distance to another position."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_dict(self):
        """Convert to dict for JSON serialization."""
        return {
            "x": float(self.x),
            "y": float(self.y),
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dict."""
        return cls(data["x"], data["y"], data["timestamp"])


class Zone:
    """Class representing a zone in the room."""
    def __init__(self, name, x1, y1, x2, y2):
        # Ensure coordinates are ordered correctly
        self.name = name
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.occupancy = 0 # Occupancy specific to this zone instance
    
    def contains(self, position):
        """Check if a position is inside this zone."""
        return (self.x1 <= position.x <= self.x2 and 
                self.y1 <= position.y <= self.y2)
    
    def reset_occupancy(self):
        """Reset the occupancy count to zero."""
        self.occupancy = 0
        
    def increment_occupancy(self):
        """Increment the occupancy count by one."""
        self.occupancy += 1
    
    def to_dict(self):
        """Convert to dict for JSON serialization."""
        return {
            "name": self.name,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "occupancy": self.occupancy
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dict."""
        zone = cls(data["name"], data["x1"], data["y1"], data["x2"], data["y2"])
        zone.occupancy = data.get("occupancy", 0)
        return zone


class ZoneTransition:
    """Class representing a transition between zones."""
    def __init__(self, track_id, from_zone, to_zone, camera, timestamp=None):
        self.track_id = track_id
        self.from_zone = from_zone
        self.to_zone = to_zone
        self.camera = camera
        self.timestamp = timestamp or time.time()
    
    def to_dict(self):
        """Convert to dict for JSON serialization."""
        return {
            "track_id": self.track_id,
            "from_zone": self.from_zone,
            "to_zone": self.to_zone,
            "camera": self.camera,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dict."""
        return cls(
            data["track_id"], 
            data["from_zone"], 
            data["to_zone"], 
            data["camera"], 
            data["timestamp"]
        ) 