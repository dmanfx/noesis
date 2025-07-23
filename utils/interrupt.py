#!/usr/bin/env python3
"""
Interrupt and Thread Management Utilities

This module provides utilities for safe thread management during application shutdown,
including timeout-based joins with daemon fallback for graceful cleanup.
"""

import logging
import threading
import multiprocessing
from typing import Optional


def safe_join(thread: threading.Thread, timeout: float = 2.0, name: str = "") -> bool:
    """
    Safely join a thread with timeout and daemon fallback.
    
    Args:
        thread: Thread to join
        timeout: Timeout in seconds for join operation
        name: Optional name for logging (uses thread.name if not provided)
        
    Returns:
        True if thread joined successfully, False if it timed out
    """
    if not thread or not thread.is_alive():
        return True
        
    logger = logging.getLogger(name or thread.name or "safe_join")
    
    try:
        thread.join(timeout)
        if thread.is_alive():
            logger.warning(f"Thread still alive after {timeout}s timeout, will be abandoned")
            # Note: Cannot set daemon=True on running thread, but it will be abandoned
            # The Python interpreter will exit when main thread exits regardless
            return False
        else:
            logger.debug(f"Thread joined successfully within {timeout}s")
            return True
    except Exception as e:
        logger.error(f"Error during thread join: {e}")
        return False


def safe_process_join(process: multiprocessing.Process, timeout: float = 2.0, name: str = "") -> bool:
    """
    Safely join a multiprocessing Process with timeout and kill fallback.
    
    Args:
        process: Process to join
        timeout: Timeout in seconds for join operation  
        name: Optional name for logging
        
    Returns:
        True if process joined successfully, False if it was killed
    """
    if not process or not process.is_alive():
        return True
        
    logger = logging.getLogger(name or f"process_{process.pid}" or "safe_process_join")
    
    try:
        process.join(timeout)
        if process.is_alive():
            logger.warning(f"Process still alive after {timeout}s timeout, killing")
            process.kill()
            process.join(1.0)  # Give it 1 second to die
            if process.is_alive():
                logger.error("Process still alive after kill, may be zombie")
                return False
            else:
                logger.debug("Process killed successfully")
                return False
        else:
            logger.debug(f"Process joined successfully within {timeout}s")
            return True
    except Exception as e:
        logger.error(f"Error during process join: {e}")
        try:
            process.kill()
            process.join(1.0)
        except:
            pass  # Ignore cleanup errors
        return False 