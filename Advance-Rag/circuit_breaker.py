import time
import logging

logger = logging.getLogger("AdvancedRAG")

class CircuitBreaker:
    """Circuit breaker pattern for failing APIs."""
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open

    def record_success(self):
        if self.state == "half-open":
            self.state = "closed"
            self.failures = 0

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.error(f"Circuit Breaker tripped to OPEN. Failure count: {self.failures}")

    def check_call_allowed(self):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                logger.info("Circuit Breaker transitioned to HALF-OPEN. Retrying service...")
            else:
                raise Exception("Circuit breaker is OPEN - calls temporarily disabled.")
