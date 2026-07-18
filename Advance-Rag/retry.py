import time
import random
import asyncio
import logging
from functools import wraps
from typing import Callable

logger = logging.getLogger("AdvancedRAG")

def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exceptions: tuple = (Exception,),
):
    """Retry decorator with exponential backoff for synchronous calls."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2**attempt), max_delay)
                        delay = delay * (0.5 + random.random()) # Add Jitter
                        logger.warning(f"Sync Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


def with_retry_async(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exceptions: tuple = (Exception,),
):
    """Retry decorator with exponential backoff for asynchronous generators/calls."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2**attempt), max_delay)
                        delay = delay * (0.5 + random.random()) # Add Jitter
                        logger.warning(f"Async Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                        await asyncio.sleep(delay)
            raise last_exception
        return wrapper
    return decorator
