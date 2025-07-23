"""
Exponential Backoff Utility
===========================

Provides intelligent backoff strategies for network reconnection attempts,
reducing CPU usage by avoiding aggressive retry loops.
"""

import random
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging


@dataclass
class BackoffState:
    """Tracks backoff state for a connection"""
    attempts: int = 0
    last_attempt: float = 0.0
    current_delay: float = 1.0
    consecutive_successes: int = 0


class ExponentialBackoff:
    """
    Exponential backoff with jitter for network reconnection.
    
    Features:
    - Exponential delay increase on failures
    - Random jitter to avoid thundering herd
    - Maximum delay cap
    - Success-based reset
    """
    
    def __init__(self,
                 initial_delay: float = 1.0,
                 max_delay: float = 60.0,
                 factor: float = 2.0,
                 jitter: float = 0.1):
        """
        Initialize exponential backoff.
        
        Args:
            initial_delay: Initial retry delay in seconds
            max_delay: Maximum retry delay in seconds
            factor: Exponential growth factor
            jitter: Random jitter factor (0.0 to 1.0)
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.factor = factor
        self.jitter = jitter
        self.logger = logging.getLogger("ExponentialBackoff")
        
    def get_delay(self, attempts: int) -> float:
        """
        Calculate next retry delay.
        
        Args:
            attempts: Number of failed attempts
            
        Returns:
            Delay in seconds before next retry
        """
        # Calculate base delay
        delay = min(
            self.initial_delay * (self.factor ** (attempts - 1)),
            self.max_delay
        )
        
        # Add jitter
        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay += random.uniform(-jitter_range, jitter_range)
            
        # Ensure positive delay
        return max(0.1, delay)
        
    def reset(self):
        """Reset backoff to initial state"""
        # This is handled by the connection tracking
        pass


class AdaptiveBackoff(ExponentialBackoff):
    """
    Adaptive backoff that adjusts based on network conditions.
    
    Features:
    - Tracks success/failure patterns
    - Adjusts aggressiveness based on stability
    - Network condition awareness
    """
    
    def __init__(self,
                 initial_delay: float = 1.0,
                 max_delay: float = 60.0,
                 factor: float = 2.0,
                 jitter: float = 0.1,
                 stability_threshold: int = 5):
        """
        Initialize adaptive backoff.
        
        Args:
            initial_delay: Initial retry delay
            max_delay: Maximum retry delay
            factor: Exponential growth factor
            jitter: Random jitter factor
            stability_threshold: Successful connections before reducing delays
        """
        super().__init__(initial_delay, max_delay, factor, jitter)
        self.stability_threshold = stability_threshold
        self.state = BackoffState()
        
    def record_attempt(self, success: bool):
        """
        Record connection attempt result.
        
        Args:
            success: Whether the connection succeeded
        """
        if success:
            self.state.consecutive_successes += 1
            if self.state.consecutive_successes >= self.stability_threshold:
                # Network is stable, reduce delays
                self.factor = max(1.5, self.factor * 0.9)
                self.logger.info(f"Network stable, reducing backoff factor to {self.factor:.2f}")
            self.state.attempts = 0
        else:
            self.state.consecutive_successes = 0
            self.state.attempts += 1
            if self.state.attempts > 10:
                # Many failures, increase delays
                self.factor = min(3.0, self.factor * 1.1)
                self.logger.info(f"Network unstable, increasing backoff factor to {self.factor:.2f}")
                
    def get_delay(self, attempts: Optional[int] = None) -> float:
        """
        Get adaptive delay based on tracked state.
        
        Args:
            attempts: Override attempt count (uses internal state if None)
            
        Returns:
            Delay in seconds
        """
        if attempts is None:
            attempts = self.state.attempts
            
        return super().get_delay(attempts)


class CircuitBreakerBackoff:
    """
    Circuit breaker pattern for connection management.
    
    States:
    - CLOSED: Normal operation, connections allowed
    - OPEN: Connections blocked due to failures
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 half_open_requests: int = 3):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Time before trying half-open state
            half_open_requests: Test requests in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_attempts = 0
        
        self.logger = logging.getLogger("CircuitBreaker")
        
    def can_attempt(self) -> bool:
        """Check if connection attempt is allowed"""
        current_time = time.time()
        
        if self.state == "CLOSED":
            return True
            
        elif self.state == "OPEN":
            # Check if we should transition to half-open
            if current_time - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.half_open_attempts = 0
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
            
        elif self.state == "HALF_OPEN":
            # Allow limited attempts in half-open state
            return self.half_open_attempts < self.half_open_requests
            
    def record_success(self):
        """Record successful connection"""
        if self.state == "HALF_OPEN":
            self.half_open_attempts += 1
            if self.half_open_attempts >= self.half_open_requests:
                # Enough successes, close circuit
                self.state = "CLOSED"
                self.failure_count = 0
                self.logger.info("Circuit breaker CLOSED - service recovered")
        elif self.state == "CLOSED":
            self.failure_count = 0
            
    def record_failure(self):
        """Record failed connection"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == "CLOSED":
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
                
        elif self.state == "HALF_OPEN":
            # Failure in half-open state, reopen circuit
            self.state = "OPEN"
            self.logger.warning("Circuit breaker reopened due to failure in HALF_OPEN state")
            
    def get_state_info(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'can_attempt': self.can_attempt(),
            'time_until_recovery': max(
                0,
                self.recovery_timeout - (time.time() - self.last_failure_time)
            ) if self.state == "OPEN" else 0
        }


class SmartBackoffManager:
    """
    Manages multiple backoff strategies for different connection types.
    """
    
    def __init__(self):
        """Initialize smart backoff manager"""
        self.strategies = {}
        self.logger = logging.getLogger("SmartBackoffManager")
        
    def add_connection(self,
                      connection_id: str,
                      strategy: str = "adaptive",
                      **kwargs):
        """
        Add a connection with specified backoff strategy.
        
        Args:
            connection_id: Unique connection identifier
            strategy: Backoff strategy ("exponential", "adaptive", "circuit_breaker")
            **kwargs: Strategy-specific parameters
        """
        if strategy == "exponential":
            self.strategies[connection_id] = ExponentialBackoff(**kwargs)
        elif strategy == "adaptive":
            self.strategies[connection_id] = AdaptiveBackoff(**kwargs)
        elif strategy == "circuit_breaker":
            self.strategies[connection_id] = CircuitBreakerBackoff(**kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        self.logger.info(f"Added {strategy} backoff for {connection_id}")
        
    def can_attempt(self, connection_id: str) -> bool:
        """Check if connection attempt is allowed"""
        strategy = self.strategies.get(connection_id)
        if not strategy:
            return True
            
        if isinstance(strategy, CircuitBreakerBackoff):
            return strategy.can_attempt()
        return True
        
    def get_delay(self, connection_id: str, attempts: Optional[int] = None) -> float:
        """Get retry delay for connection"""
        strategy = self.strategies.get(connection_id)
        if not strategy:
            return 1.0
            
        if isinstance(strategy, CircuitBreakerBackoff):
            # Circuit breaker doesn't use delays
            return 0.1 if strategy.can_attempt() else float('inf')
        elif isinstance(strategy, AdaptiveBackoff):
            return strategy.get_delay(attempts)
        else:
            return strategy.get_delay(attempts or 1)
            
    def record_result(self, connection_id: str, success: bool):
        """Record connection attempt result"""
        strategy = self.strategies.get(connection_id)
        if not strategy:
            return
            
        if isinstance(strategy, CircuitBreakerBackoff):
            if success:
                strategy.record_success()
            else:
                strategy.record_failure()
        elif isinstance(strategy, AdaptiveBackoff):
            strategy.record_attempt(success)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get backoff statistics for all connections"""
        stats = {}
        for conn_id, strategy in self.strategies.items():
            if isinstance(strategy, CircuitBreakerBackoff):
                stats[conn_id] = strategy.get_state_info()
            elif isinstance(strategy, AdaptiveBackoff):
                stats[conn_id] = {
                    'attempts': strategy.state.attempts,
                    'consecutive_successes': strategy.state.consecutive_successes,
                    'current_factor': strategy.factor
                }
            else:
                stats[conn_id] = {'type': 'exponential'}
                
        return stats


# Example usage
if __name__ == "__main__":
    # Test exponential backoff
    backoff = ExponentialBackoff(initial_delay=1.0, max_delay=30.0)
    print("Exponential backoff delays:")
    for i in range(1, 8):
        print(f"  Attempt {i}: {backoff.get_delay(i):.2f}s")
        
    # Test adaptive backoff
    adaptive = AdaptiveBackoff(initial_delay=1.0)
    print("\nAdaptive backoff with failures:")
    for i in range(5):
        adaptive.record_attempt(False)
        print(f"  After failure {i+1}: {adaptive.get_delay():.2f}s")
        
    # Test circuit breaker
    breaker = CircuitBreakerBackoff(failure_threshold=3)
    print("\nCircuit breaker test:")
    for i in range(5):
        if breaker.can_attempt():
            print(f"  Attempt {i+1}: Allowed")
            breaker.record_failure()
        else:
            print(f"  Attempt {i+1}: Blocked (circuit open)")
    print(f"  State: {breaker.get_state_info()}") 