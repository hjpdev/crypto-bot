"""
Base process functionality for background processes in the crypto trading bot.

This module provides a base class for implementing concurrent processes
with proper health checking, error handling, and graceful shutdown capabilities.
"""

import abc
import logging
import time
import threading
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class ProcessState(str, Enum):
    """Enum representing the possible states of a process."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    TERMINATED = "terminated"


class BaseProcess(abc.ABC):
    """
    Base class for implementing background processes.

    Provides common functionality for all processes including:
    - Lifecycle management (start, stop, pause, resume)
    - Health monitoring
    - Error handling and recovery
    - Interval management
    - Graceful shutdown support

    Subclasses should implement the `_run_iteration` method.
    """

    def __init__(
        self,
        name: str,
        interval_seconds: float = 60.0,
        max_errors: int = 5,
        error_cooldown_seconds: float = 60.0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the base process.

        Args:
            name: Process name used for identification and logging
            interval_seconds: Time in seconds between process iterations
            max_errors: Maximum consecutive errors before process enters error state
            error_cooldown_seconds: Cooldown period after multiple errors
            logger: Logger to use (if None, a new logger will be created)
        """
        self.name = name
        self.interval_seconds = interval_seconds
        self.max_errors = max_errors
        self.error_cooldown_seconds = error_cooldown_seconds
        self.logger = logger or logging.getLogger(f"process.{name}")

        # State management
        self._state = ProcessState.INITIALIZING
        self._last_iteration_time: Optional[datetime] = None
        self._last_successful_iteration_time: Optional[datetime] = None
        self._error_count = 0
        self._total_iterations = 0
        self._successful_iterations = 0
        self._total_errors = 0
        self._start_time: Optional[datetime] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    @property
    def state(self) -> ProcessState:
        """Get the current process state."""
        with self._lock:
            return self._state

    @property
    def uptime_seconds(self) -> float:
        """Get the process uptime in seconds."""
        if self._start_time is None:
            return 0.0
        return (datetime.utcnow() - self._start_time).total_seconds()

    @property
    def is_healthy(self) -> bool:
        """Check if the process is currently healthy."""
        # Process is healthy if:
        # 1. It's in the RUNNING state
        # 2. It's not experiencing too many errors
        # 3. It had a successful iteration recently
        with self._lock:
            if self._state != ProcessState.RUNNING:
                return False
            if self._error_count >= self.max_errors:
                return False
            if self._last_successful_iteration_time is None:
                return False

            # Check if we've had a successful iteration within 2x the expected interval
            max_interval = self.interval_seconds * 2
            time_since_last_success = (
                datetime.utcnow() - self._last_successful_iteration_time
            ).total_seconds()
            return time_since_last_success <= max_interval

    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information about the process."""
        with self._lock:
            status = {
                "name": self.name,
                "state": self._state.value,
                "uptime_seconds": self.uptime_seconds,
                "total_iterations": self._total_iterations,
                "successful_iterations": self._successful_iterations,
                "total_errors": self._total_errors,
                "current_error_count": self._error_count,
                "last_iteration": (
                    self._last_iteration_time.isoformat() if self._last_iteration_time else None
                ),
                "last_successful_iteration": (
                    self._last_successful_iteration_time.isoformat()
                    if self._last_successful_iteration_time
                    else None
                ),
                "interval_seconds": self.interval_seconds,
                "is_healthy": self.is_healthy,
            }
            return status

    def start(self) -> None:
        """Start the process in a new thread."""
        with self._lock:
            if self._state in (ProcessState.RUNNING, ProcessState.INITIALIZING):
                self.logger.warning(f"Process '{self.name}' is already running")
                return

            self.logger.info(f"Starting process '{self.name}'")
            self._stop_event.clear()
            self._pause_event.clear()
            self._start_time = datetime.utcnow()
            self._state = ProcessState.RUNNING
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 10.0) -> bool:
        """
        Stop the process gracefully.

        Args:
            timeout: Maximum time to wait for the process to stop

        Returns:
            bool: True if process stopped gracefully, False if it timed out
        """
        with self._lock:
            if self._state in (ProcessState.STOPPED, ProcessState.TERMINATED):
                self.logger.warning(f"Process '{self.name}' is already stopped")
                return True

            self.logger.info(f"Stopping process '{self.name}'")
            self._state = ProcessState.STOPPING
            self._stop_event.set()

        # Wait for thread to finish outside the lock
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                # Thread didn't finish in time
                self.logger.warning(f"Process '{self.name}' did not stop within {timeout} seconds")
                with self._lock:
                    self._state = ProcessState.ERROR
                return False

        with self._lock:
            self._state = ProcessState.STOPPED
            self._thread = None
        return True

    def pause(self) -> None:
        """Pause the process temporarily."""
        with self._lock:
            if self._state != ProcessState.RUNNING:
                self.logger.warning(
                    f"Cannot pause process '{self.name}' in state {self._state.value}"
                )
                return

            self.logger.info(f"Pausing process '{self.name}'")
            self._state = ProcessState.PAUSED
            self._pause_event.set()

    def resume(self) -> None:
        """Resume a paused process."""
        with self._lock:
            if self._state != ProcessState.PAUSED:
                self.logger.warning(
                    f"Cannot resume process '{self.name}' in state {self._state.value}"
                )
                return

            self.logger.info(f"Resuming process '{self.name}'")
            self._state = ProcessState.RUNNING
            self._pause_event.clear()

    def set_interval(self, interval_seconds: float) -> None:
        """Update the process interval."""
        with self._lock:
            self.logger.info(f"Updating process '{self.name}' interval to {interval_seconds}s")
            self.interval_seconds = interval_seconds

    def _run(self) -> None:
        """Main process loop."""
        self.logger.info(f"Process '{self.name}' started")

        try:
            # Call initialization hook
            self._on_start()

            while not self._stop_event.is_set():
                # Handle pause state
                if self._pause_event.is_set():
                    time.sleep(0.1)  # Short sleep while paused
                    continue

                iteration_start_time = datetime.utcnow()
                try:
                    with self._lock:
                        self._last_iteration_time = iteration_start_time
                        self._total_iterations += 1

                    # Run the actual process logic
                    self._run_iteration()

                    with self._lock:
                        self._last_successful_iteration_time = datetime.utcnow()
                        self._successful_iterations += 1
                        self._error_count = 0  # Reset error count after success

                except Exception as e:
                    # Handle errors
                    self._handle_error(e)

                # Calculate sleep time
                elapsed = (datetime.utcnow() - iteration_start_time).total_seconds()
                sleep_time = max(0.0, self.interval_seconds - elapsed)

                # Sleep until next interval or until stopped
                if sleep_time > 0 and not self._stop_event.is_set():
                    # Use smaller sleep increments to allow for faster process termination
                    increment = min(0.1, sleep_time)
                    end_time = time.time() + sleep_time
                    while time.time() < end_time and not self._stop_event.is_set():
                        time.sleep(increment)

        except Exception as e:
            self.logger.exception(f"Unexpected error in process '{self.name}': {e}")
            with self._lock:
                self._state = ProcessState.ERROR
                self._total_errors += 1

        finally:
            # Call cleanup hook
            try:
                self._on_stop()
            except Exception as e:
                self.logger.exception(f"Error during process '{self.name}' cleanup: {e}")

            with self._lock:
                if self._state != ProcessState.ERROR:
                    self._state = ProcessState.STOPPED
            self.logger.info(f"Process '{self.name}' stopped")

    def _handle_error(self, error: Exception) -> None:
        """Handle process errors with appropriate logging and state management."""
        self.logger.error(f"Error in process '{self.name}': {error}")
        self.logger.exception(error)

        with self._lock:
            self._error_count += 1
            self._total_errors += 1

            if self._error_count >= self.max_errors:
                self.logger.warning(
                    f"Process '{self.name}' reached max consecutive errors ({self.max_errors}), "
                    f"cooling down for {self.error_cooldown_seconds}s"
                )

                # Apply cooldown period after too many errors
                time.sleep(self.error_cooldown_seconds)
                self._error_count = 0  # Reset counter after cooldown

    def _on_start(self) -> None:
        """Hook called when the process starts."""
        pass

    def _on_stop(self) -> None:
        """Hook called when the process stops."""
        pass

    @abc.abstractmethod
    def _run_iteration(self) -> None:
        """
        Run a single iteration of the process.

        This method must be implemented by subclasses.
        It contains the main logic of the process.
        """
        raise NotImplementedError("Subclasses must implement _run_iteration()")
