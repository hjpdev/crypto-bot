"""
Rate limiter for exchange API calls.

This module provides rate limiting functionality for exchange API calls to prevent
rate limit errors by tracking call frequency and implementing exponential backoff.
"""

import time
import logging
import functools
import random
from typing import Callable, Dict, Any, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimiter:
    """
    Implements rate limiting and exponential backoff for API calls.

    This class tracks API call frequency and implements exponential backoff
    strategy when rate limit errors are encountered.
    """

    def __init__(
        self,
        calls_per_second: float = 1.0,
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
    ):
        """
        Initialize the rate limiter.

        Args:
            calls_per_second: Maximum number of calls allowed per second
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            backoff_factor: Multiplier for exponential backoff
            jitter: Whether to add random jitter to backoff times
        """
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_factor = backoff_factor
        self.jitter = jitter

        self.last_call_time: Dict[str, float] = {}
        self.retry_counts: Dict[str, int] = {}

    def _calculate_backoff(self, retry_count: int) -> float:
        """Calculate backoff time (seconds) using exponential backoff strategy."""
        backoff = self.initial_backoff * (self.backoff_factor**retry_count)

        if self.jitter:
            # Add random jitter between 0 and 0.5 * backoff
            backoff = backoff + (random.random() * 0.5 * backoff)

        return backoff

    def _enforce_rate_limit(self, method_name: str) -> None:
        """Enforce rate limits by waiting if needed."""
        current_time = time.time()

        if method_name in self.last_call_time:
            elapsed = current_time - self.last_call_time[method_name]
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.4f}s")
                time.sleep(sleep_time)

        self.last_call_time[method_name] = time.time()

    def reset_retry_count(self, method_name: str) -> None:
        """Reset the retry counter for a specific method."""
        if method_name in self.retry_counts:
            del self.retry_counts[method_name]

    def rate_limited(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator for rate-limited methods.

        Args:
            func: The function to apply rate limiting to

        Returns:
            Rate-limited wrapped function
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get function name or generate a unique identifier if __name__ is not available
            method_name = getattr(func, "__name__", f"func_{id(func)}")
            self._enforce_rate_limit(method_name)
            return func(*args, **kwargs)

        return wrapper

    def with_backoff(
        self, func: Callable[..., T], rate_limit_errors: Optional[tuple] = None
    ) -> Callable[..., T]:
        """
        Decorator that implements exponential backoff for rate limit errors.

        Args:
            func: The function to apply backoff to
            rate_limit_errors: Tuple of exception types that indicate rate limiting

        Returns:
            Wrapped function with backoff logic
        """
        if rate_limit_errors is None:
            # Default to these common error types for CCXT
            from ccxt.base.errors import RateLimitExceeded, ExchangeNotAvailable, NetworkError

            rate_limit_errors = (RateLimitExceeded, ExchangeNotAvailable, NetworkError)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get function name or generate a unique identifier if __name__ is not available
            method_name = getattr(func, "__name__", f"func_{id(func)}")

            # Apply rate limiting
            self._enforce_rate_limit(method_name)

            # Initialize retry count if not present
            if method_name not in self.retry_counts:
                self.retry_counts[method_name] = 0

            try:
                result = func(*args, **kwargs)
                # Reset retry count on success
                self.reset_retry_count(method_name)
                return result

            except rate_limit_errors as e:
                # Handle rate limit errors with backoff
                if self.retry_counts[method_name] >= self.max_retries:
                    logger.error(f"Maximum retries ({self.max_retries}) exceeded for {method_name}")
                    self.reset_retry_count(method_name)
                    raise

                # Calculate backoff time
                retry_count = self.retry_counts[method_name]
                backoff_time = self._calculate_backoff(retry_count)

                logger.warning(
                    f"Rate limit error: {method_name}, retry {retry_count + 1}/{self.max_retries} "
                    f"after {backoff_time:.2f}s backoff: {str(e)}"
                )

                # Increment retry counter and wait
                self.retry_counts[method_name] += 1
                time.sleep(backoff_time)

                # Recursive call to retry
                return wrapper(*args, **kwargs)

        return wrapper
