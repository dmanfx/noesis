import logging
import threading
import time
from typing import Dict


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