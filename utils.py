"""
Utility Functions Module

This module provides various utility functions used throughout the application.
These functions handle common tasks like file operations, performance metrics,
image processing, and logging configuration.

Having these utilities in a separate module helps to:
1. Avoid code duplication across the application
2. Provide a single place to update and maintain common functionality
3. Keep other modules focused on their primary responsibilities
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import time
import cv2
import functools
import json
import logging
import os
os.environ['no_proxy'] = '*'
from datetime import datetime
import threading
from pathlib import Path
import supervision as sv
from geometry import Position, Zone, ZoneTransition
from track import Track
from collections import defaultdict

# Constants
COORD_SYS_WIDTH = 1024
COORD_SYS_HEIGHT = 576


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization

    Args:
        obj: Object to convert

    Returns:
        Converted object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        return convert_numpy_types(obj.to_dict())
    return obj


def encode_frame(frame: np.ndarray, quality: int = 90) -> Optional[bytes]:
    """Encode a frame as JPEG bytes

    Args:
        frame: Frame to encode
        quality: JPEG quality (0-100)

    Returns:
        JPEG bytes or None if encoding failed
    """
    if frame is None:
        return None

    try:
        # Encode frame as JPEG
        _, encoded_img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return encoded_img.tobytes()
    except Exception as e:
        print(f"[Frame Encoding Error] {e}")
        return None


def calculate_fps(frame_count: int, elapsed_time: float, previous_fps: float = 0.0, alpha: float = 0.9) -> float:
    """Calculate FPS with smoothing

    Args:
        frame_count: Number of frames processed
        elapsed_time: Time elapsed in seconds
        previous_fps: Previous FPS value for smoothing
        alpha: Smoothing factor (0-1)

    Returns:
        Smoothed FPS value
    """
    if elapsed_time <= 0:
        return previous_fps

    # Calculate raw FPS
    raw_fps = frame_count / elapsed_time

    # Apply smoothing
    if previous_fps > 0:
        smoothed_fps = alpha * previous_fps + (1 - alpha) * raw_fps
    else:
        smoothed_fps = raw_fps

    return smoothed_fps


def setup_logging(log_level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Set up application-wide logging configuration
    
    Configures the root logger with consistent formatting for both console
    and optional file output. This ensures all log messages across the
    application have a uniform appearance and can be easily filtered.
    
    Args:
        log_level: Logging level (e.g., logging.DEBUG, logging.INFO)
        log_file: Optional path to log file for persistent logging
    
    Returns:
        Configured root logger instance
    """
    # Create formatters
    console_formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicate logs
    root_logger.handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
    
    return root_logger


def timer_decorator(name: str = None) -> Callable:
    """Decorator to measure and log function execution time
    
    Wraps a function to measure its execution time, which is useful for
    performance profiling and optimization.
    
    Args:
        name: Optional name for the timer (defaults to function name)
    
    Returns:
        Decorated function that logs execution time
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or func.__name__
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
            logger = logging.getLogger('timer')
            logger.debug(f"{timer_name} took {elapsed_time:.2f}ms")
            return result
        return wrapper
    return decorator


def ensure_dir(directory: str) -> bool:
    """Ensure a directory exists, creating it if necessary

    Args:
        directory: Directory path

    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return True
    except Exception as e:
        print(f"[Directory Error] Failed to create directory {directory}: {e}")
        return False


def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """Get current timestamp as a formatted string
    
    Used for creating timestamped file names, logs, and displays.
    
    Args:
        format_str: Format string for the timestamp (strftime format)
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format_str)


def create_background_thread(target: Callable, args: Tuple = (), daemon: bool = True) -> threading.Thread:
    """Create and start a background thread

    Args:
        target: Thread target function
        args: Arguments for the target function
        daemon: Whether the thread should be a daemon

    Returns:
        Started thread
    """
    thread = threading.Thread(target=target, args=args, daemon=daemon)
    thread.start()
    return thread


def safe_file_write(file_path: str, content: str, mode: str = 'w') -> bool:
    """Safely write content to a file

    Args:
        file_path: Path to file
        content: Content to write
        mode: File open mode

    Returns:
        True if write succeeded, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Write file with temporary name first
        temp_path = f"{file_path}.temp"
        with open(temp_path, mode) as f:
            f.write(content)

        # Rename to final path (atomic operation)
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rename(temp_path, file_path)
        return True
    except Exception as e:
        print(f"[File Write Error] Failed to write to {file_path}: {e}")
        return False


def read_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Read a JSON file

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data or None if reading failed
    """
    try:
        if not os.path.exists(file_path):
            return None

        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[JSON Read Error] Failed to read {file_path}: {e}")
        return None


def write_json_file(file_path: str, data: Dict[str, Any]) -> bool:
    """Write data to a JSON file

    Args:
        file_path: Path to JSON file
        data: Data to write

    Returns:
        True if write succeeded, False otherwise
    """
    try:
        # Convert numpy types
        data = convert_numpy_types(data)

        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Write JSON to temporary file
        temp_path = f"{file_path}.temp"
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Rename to final path (atomic operation)
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rename(temp_path, file_path)
        return True
    except Exception as e:
        print(f"[JSON Write Error] Failed to write to {file_path}: {e}")
        return False


def resize_frame(frame: np.ndarray, target_width: Optional[int] = None, 
                target_height: Optional[int] = None) -> np.ndarray:
    """Resize a frame while maintaining aspect ratio
    
    Resizes an image to fit within target dimensions while preserving
    the original aspect ratio to prevent distortion.
    
    Args:
        frame: OpenCV image to resize
        target_width: Desired maximum width (None to scale based only on height)
        target_height: Desired maximum height (None to scale based only on width)
    
    Returns:
        Resized image
    """
    if frame is None:
        return None
    
    height, width = frame.shape[:2]
    
    # Return original if no resize needed
    if (target_width is None or width <= target_width) and \
       (target_height is None or height <= target_height):
        return frame
    
    # Calculate new dimensions
    if target_width is not None and target_height is not None:
        width_ratio = target_width / width
        height_ratio = target_height / height
        # Use the smaller ratio to ensure both dimensions fit
        ratio = min(width_ratio, height_ratio)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
    elif target_width is not None:
        ratio = target_width / width
        new_width = target_width
        new_height = int(height * ratio)
    else:  # target_height is not None
        ratio = target_height / height
        new_height = target_height
        new_width = int(width * ratio)
    
    # Resize the image
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def draw_text(frame: np.ndarray, text: str, position: Tuple[int, int], 
             font_scale: float = 0.5, thickness: int = 1,
             text_color: Tuple[int, int, int] = (255, 255, 255),
             bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0)) -> np.ndarray:
    """Draw text on an image with optional background for better readability
    
    Creates text overlays for visualizations with optional background
    rectangle to ensure visibility on any image content.
    
    Args:
        frame: OpenCV image to draw on
        text: Text to display
        position: (x, y) coordinates for the text
        font_scale: Scale factor for the font
        thickness: Thickness of the text lines
        text_color: (B, G, R) color tuple for the text
        bg_color: Optional (B, G, R) color tuple for the background
        
    Returns:
        Image with text drawn on it
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Make a copy to avoid modifying the original
    result = frame.copy()
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Draw background rectangle if bg_color is provided
    if bg_color is not None:
        cv2.rectangle(
            result,
            (position[0], position[1] - text_height - baseline),
            (position[0] + text_width, position[1] + baseline),
            bg_color,
            -1  # Filled rectangle
        )
    
    # Draw text
    cv2.putText(
        result,
        text,
        position,
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )
    
    return result


def get_unique_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate a list of visually distinct colors
    
    Creates a set of colors with good visual separation for
    visualizing different tracks or categories.
    
    Args:
        n: Number of colors to generate
        
    Returns:
        List of (B, G, R) color tuples
    """
    # Pre-defined set of distinct colors for small n
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 0),    # Dark blue
        (0, 128, 0),    # Dark green
        (0, 0, 128),    # Dark red
        (128, 128, 0),  # Dark cyan
        (128, 0, 128),  # Dark magenta
        (0, 128, 128),  # Dark yellow
    ]
    
    # Return pre-defined colors if we have enough
    if n <= len(colors):
        return colors[:n]
    
    # Generate additional colors using HSV color space for better separation
    result = colors.copy()
    
    # Generate remaining colors with evenly distributed hues
    remaining = n - len(colors)
    for i in range(remaining):
        # Distribute hue evenly in the range [0, 180)
        hue = int(180 * i / remaining)
        # Use high saturation and value for visibility
        hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        # Convert to BGR
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        result.append((int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])))
    
    return result


class PerformanceMonitor:
    """Utility class for tracking performance metrics
    
    Tracks execution times, counts, and moving averages for
    performance monitoring and optimization.
    """
    
    def __init__(self, window_size: int = 100):
        """Initialize a performance monitor
        
        Args:
            window_size: Number of samples to keep for rolling averages
        """
        self.window_size = window_size
        self.times = {}
        self.counters = {}
        self.averages = {}
        self.lock = threading.Lock()
    
    def start_timer(self, name: str) -> None:
        """Start a named timer
        
        Args:
            name: Timer identifier
        """
        with self.lock:
            self.times[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed time
        
        Args:
            name: Timer identifier
            
        Returns:
            Elapsed time in milliseconds
        """
        with self.lock:
            if name not in self.times:
                return 0.0
            
            elapsed = (time.time() - self.times[name]) * 1000  # ms
            
            # Update rolling average
            if name not in self.averages:
                self.averages[name] = [elapsed]
            else:
                self.averages[name].append(elapsed)
                # Keep only the last window_size samples
                if len(self.averages[name]) > self.window_size:
                    self.averages[name].pop(0)
            
            return elapsed
    
    def increment_counter(self, name: str, value: int = 1) -> int:
        """Increment a named counter
        
        Args:
            name: Counter identifier
            value: Value to add to the counter
            
        Returns:
            New counter value
        """
        with self.lock:
            if name not in self.counters:
                self.counters[name] = value
            else:
                self.counters[name] += value
            return self.counters[name]
    
    def get_counter(self, name: str) -> int:
        """Get current value of a counter
        
        Args:
            name: Counter identifier
            
        Returns:
            Current counter value or 0 if not found
        """
        with self.lock:
            return self.counters.get(name, 0)
    
    def get_average_time(self, name: str) -> float:
        """Get average time for a named timer
        
        Args:
            name: Timer identifier
            
        Returns:
            Average time in milliseconds or 0.0 if not found
        """
        with self.lock:
            times = self.averages.get(name, [])
            if not times:
                return 0.0
            return sum(times) / len(times)
    
    def reset(self) -> None:
        """Reset all timers and counters"""
        with self.lock:
            self.times = {}
            self.counters = {}
            self.averages = {}


def convert_np_for_json(obj):
    """Recursively converts NumPy types in an object to standard Python types for JSON."""
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                      np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_np_for_json(item) for item in obj]  # Convert arrays to lists recursively
    elif isinstance(obj, list):
        return [convert_np_for_json(item) for item in obj]  # Recursively convert lists
    elif isinstance(obj, dict):
        return {k: convert_np_for_json(v) for k, v in obj.items()}  # Recursively convert dict values
    elif isinstance(obj, (Position, Zone, Track)):  # Handle custom objects if needed
        return convert_np_for_json(obj.to_dict())
    return obj


def filter_detections_by_exclusion_zones(detections: sv.Detections, exclusion_zones: dict, camera_id: str) -> sv.Detections:
    """Filters detections based on whether their bottom-center point is inside any defined exclusion zones.

    Args:
        detections (sv.Detections): The detections to filter.
        exclusion_zones (dict): Dictionary of camera_id -> {zone_name -> (x1, y1, x2, y2)}.
        camera_id (str): The ID of the camera where detections occurred.

    Returns:
        sv.Detections: The filtered detections.
    """
    # Check if exclusion zones are defined and if the specific camera has any
    if not exclusion_zones or camera_id not in exclusion_zones:
        return detections

    cam_zones = exclusion_zones[camera_id]
    if not cam_zones:  # Check if the dictionary for the camera is empty
        return detections

    # Get bottom-center coordinates of detections
    # Ensure xyxy is available and not empty before calculating anchor points
    if detections.xyxy is None or len(detections.xyxy) == 0:
        return detections  # Return empty/original if no boxes

    anchor_points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    include_mask = np.ones(len(detections), dtype=bool)

    for zone_name, coords in cam_zones.items():
        # Assuming coords are (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, coords)  # Ensure integer coords
        # Create polygon for zone checking (closed loop)
        zone_polygon = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
            [x1, y1]  # Close the polygon
        ], dtype=np.int32)

        # Check which points are inside this zone
        try:
            # Use cv2.pointPolygonTest for each point
            is_inside = np.array([cv2.pointPolygonTest(zone_polygon, tuple(map(int, point)), False) >= 0 
                                for point in anchor_points])

            # Update mask: set to False if inside this exclusion zone
            include_mask[is_inside] = False
        except Exception as e_poly_check:
            print(f"[Error] Checking exclusion zone '{zone_name}' failed: {e_poly_check}")
            continue

    # Return only detections where the mask is True
    return detections[include_mask] 


class RateLimitedLogger:
    """A logger wrapper that rate-limits log messages to prevent spam."""
    
    def __init__(self, logger: logging.Logger, rate_limit_seconds: float = 5.0):
        """Initialize rate-limited logger.
        
        Args:
            logger: The underlying logger instance
            rate_limit_seconds: Minimum seconds between identical messages
        """
        self.logger = logger
        self.rate_limit_seconds = rate_limit_seconds
        self.last_log_times: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def _get_message_key(self, level: int, message: str) -> str:
        """Generate a unique key for rate limiting based on level and message."""
        return f"{level}:{message}"
    
    def _should_log(self, message_key: str) -> bool:
        """Check if enough time has passed to log this message again."""
        current_time = time.time()
        with self.lock:
            last_time = self.last_log_times.get(message_key, 0)
            if current_time - last_time >= self.rate_limit_seconds:
                self.last_log_times[message_key] = current_time
                return True
            return False
    
    def debug(self, message: str, *args, **kwargs):
        """Rate-limited debug logging."""
        message_key = self._get_message_key(logging.DEBUG, message)
        if self._should_log(message_key):
            self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Rate-limited info logging."""
        message_key = self._get_message_key(logging.INFO, message)
        if self._should_log(message_key):
            self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Rate-limited warning logging."""
        message_key = self._get_message_key(logging.WARNING, message)
        if self._should_log(message_key):
            self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Rate-limited error logging."""
        message_key = self._get_message_key(logging.ERROR, message)
        if self._should_log(message_key):
            self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Rate-limited critical logging."""
        message_key = self._get_message_key(logging.CRITICAL, message)
        if self._should_log(message_key):
            self.logger.critical(message, *args, **kwargs)
    
    def log_detection_count(self, camera_id: str, detection_count: int, frame_id: int):
        """Special method for logging detection counts with rate limiting.
        
        Only logs if detection_count > 0 and rate limit has passed.
        """
        if detection_count > 0:
            message = f"Camera {camera_id} - Frame {frame_id}: {detection_count} detections"
            message_key = self._get_message_key(logging.INFO, f"detection_count:{camera_id}")
            if self._should_log(message_key):
                self.logger.info(message) 