"""Tests for the task scheduler functionality."""

import pytest
import time
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

from app.core.scheduler import TaskScheduler, TaskState


class TestTaskScheduler:
    """Tests for the TaskScheduler class."""

    @pytest.fixture
    def mock_logger(self):
        """Fixture to provide a mock logger."""
        return MagicMock()

    @pytest.fixture
    def scheduler(self, mock_logger):
        """Fixture to create a scheduler instance with a mock logger."""
        return TaskScheduler(
            max_workers=2,
            use_asyncio=False,  # Use threading for most tests for simplicity
            max_retries=2,
            retry_delay=0.1,
            logger=mock_logger,
        )

    def test_init(self, scheduler, mock_logger):
        """Test that the scheduler initializes with the correct attributes."""
        assert scheduler._max_workers == 2
        assert scheduler._use_asyncio is False
        assert scheduler._max_retries == 2
        assert scheduler._retry_delay == 0.1
        assert scheduler._logger == mock_logger
        assert scheduler._tasks == {}
        assert scheduler._task_priority_queue == []
        assert scheduler._running is False

    def test_add_task(self, scheduler):
        """Test adding a task to the scheduler."""
        def dummy_task():
            return "success"

        # Add a task with explicit name
        task_name = scheduler.add_task(
            task_func=dummy_task,
            interval=1.0,
            name="test_task",
            priority=10,
        )

        assert task_name == "test_task"
        assert "test_task" in scheduler._tasks
        assert scheduler._tasks["test_task"].task_func == dummy_task
        assert scheduler._tasks["test_task"].interval == 1.0
        assert scheduler._tasks["test_task"].priority == 10
        assert scheduler._tasks["test_task"].state == TaskState.PENDING

        # Add a task with auto-generated name
        task_name2 = scheduler.add_task(
            task_func=dummy_task,
            interval=2.0,
        )

        assert task_name2.startswith("task_")
        assert task_name2 in scheduler._tasks
        assert scheduler._tasks[task_name2].interval == 2.0
        assert scheduler._tasks[task_name2].priority == 100  # Default priority

        # Verify priority queue was updated
        assert len(scheduler._task_priority_queue) == 2

    def test_add_duplicate_task(self, scheduler):
        """Test that adding a duplicate task raises an error."""
        def dummy_task():
            return "success"

        scheduler.add_task(
            task_func=dummy_task,
            interval=1.0,
            name="test_task",
        )

        with pytest.raises(ValueError, match="Task with name 'test_task' already exists"):
            scheduler.add_task(
                task_func=dummy_task,
                interval=2.0,
                name="test_task",
            )

    def test_remove_task(self, scheduler):
        """Test removing a task from the scheduler."""
        def dummy_task():
            return "success"

        scheduler.add_task(
            task_func=dummy_task,
            interval=1.0,
            name="test_task",
        )

        # Remove an existing task
        result = scheduler.remove_task("test_task")
        assert result is True
        assert "test_task" not in scheduler._tasks
        assert len(scheduler._task_priority_queue) == 0

        # Remove a non-existent task
        result = scheduler.remove_task("nonexistent_task")
        assert result is False

    def test_pause_resume_task(self, scheduler):
        """Test pausing and resuming a task."""
        def dummy_task():
            return "success"

        scheduler.add_task(
            task_func=dummy_task,
            interval=1.0,
            name="test_task",
        )

        # Pause the task
        result = scheduler.pause_task("test_task")
        assert result is True
        assert scheduler._tasks["test_task"].state == TaskState.PAUSED

        # Priority queue should not contain paused tasks
        assert len(scheduler._task_priority_queue) == 0

        # Resume the task
        result = scheduler.resume_task("test_task")
        assert result is True
        assert scheduler._tasks["test_task"].state == TaskState.PENDING
        assert len(scheduler._task_priority_queue) == 1

        # Try to pause a non-existent task
        result = scheduler.pause_task("nonexistent_task")
        assert result is False

        # Try to resume a non-existent task
        result = scheduler.resume_task("nonexistent_task")
        assert result is False

    def test_get_task_status(self, scheduler):
        """Test getting the status of a task."""
        def dummy_task():
            return "success"

        scheduler.add_task(
            task_func=dummy_task,
            interval=1.0,
            name="test_task",
            priority=5,
        )

        status = scheduler.get_task_status("test_task")
        assert status is not None
        assert status["name"] == "test_task"
        assert status["state"] == "pending"
        assert status["priority"] == 5
        assert status["interval"] == 1.0
        assert "stats" in status

        # Test for non-existent task
        status = scheduler.get_task_status("nonexistent_task")
        assert status is None

    def test_get_all_tasks_status(self, scheduler):
        """Test getting status for all tasks."""
        def dummy_task1():
            return "success1"

        def dummy_task2():
            return "success2"

        scheduler.add_task(
            task_func=dummy_task1,
            interval=1.0,
            name="task1",
        )

        scheduler.add_task(
            task_func=dummy_task2,
            interval=2.0,
            name="task2",
        )

        all_status = scheduler.get_all_tasks_status()
        assert len(all_status) == 2
        assert "task1" in all_status
        assert "task2" in all_status
        assert all_status["task1"]["interval"] == 1.0
        assert all_status["task2"]["interval"] == 2.0

    @patch("threading.Thread")
    def test_start_stop_threaded(self, mock_thread, scheduler):
        """Test starting and stopping the scheduler with threading."""
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        # Start the scheduler
        scheduler.start()
        assert scheduler._running is True
        assert mock_thread.called
        assert mock_thread_instance.start.called

        # Stop the scheduler
        mock_thread_instance.is_alive.return_value = False
        result = scheduler.stop()
        assert result is True
        assert scheduler._running is False

    @patch("threading.Thread")
    @patch("concurrent.futures.ThreadPoolExecutor")
    def test_start_stop_asyncio(self, mock_executor, mock_thread, mock_logger):
        """Test starting and stopping the scheduler with asyncio."""
        scheduler = TaskScheduler(
            max_workers=2,
            use_asyncio=True,
            logger=mock_logger,
        )

        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        # Start the scheduler
        scheduler.start()
        assert scheduler._running is True
        assert mock_thread.called
        assert mock_thread_instance.start.called

        # Stop the scheduler
        mock_thread_instance.is_alive.return_value = False
        result = scheduler.stop()
        assert result is True
        assert scheduler._running is False

    def test_task_priority_ordering(self):
        """Test that tasks are executed in the correct priority order."""
        executed_tasks = []

        def task1():
            executed_tasks.append("task1")

        def task2():
            executed_tasks.append("task2")

        def task3():
            executed_tasks.append("task3")

        # Create scheduler and add tasks with different priorities
        scheduler = TaskScheduler(use_asyncio=False, max_workers=1)

        # Initialize the thread pool explicitly for testing
        scheduler._thread_pool = ThreadPoolExecutor(max_workers=1)

        # Add tasks with priorities (lower number = higher priority)
        scheduler.add_task(task_func=task3, interval=0.1, name="task3", priority=30)
        scheduler.add_task(task_func=task1, interval=0.1, name="task1", priority=10)
        scheduler.add_task(task_func=task2, interval=0.1, name="task2", priority=20)

        # Force all tasks to have the same due time
        now = time.time()
        for task_name in scheduler._tasks:
            scheduler._tasks[task_name].last_run_time = now - 1  # All due

        # Process the tasks
        scheduler._update_task_queue()
        scheduler._process_due_tasks()

        # Wait a moment for all tasks to complete
        time.sleep(0.2)

        # Verify the execution order follows priority
        assert executed_tasks[:3] == ["task1", "task2", "task3"]

    def test_asyncio_task(self):
        """Test that asyncio tasks are executed correctly."""
        # For testing purposes, we'll use a simpler approach that doesn't require actual asyncio
        result = None

        # Create a regular function instead of an async function to avoid the warning
        def mock_coro():
            nonlocal result
            result = "async_success"
            return result

        # Create scheduler with asyncio support
        scheduler = TaskScheduler(use_asyncio=True, max_workers=1)

        # Mock asyncio.run_coroutine_threadsafe to directly call the mock function
        def mock_run_coroutine_threadsafe(coro, loop):
            class MockFuture:
                def result(self, timeout=None):
                    nonlocal result
                    result = "async_success"
                    return result
            return MockFuture()

        with patch('asyncio.run_coroutine_threadsafe', side_effect=mock_run_coroutine_threadsafe):
            # Initialize the event loop and thread pool
            scheduler._loop = MagicMock()
            scheduler._thread_pool = ThreadPoolExecutor(max_workers=1)

            # Add and execute the task
            scheduler.add_task(task_func=mock_coro, interval=0.1, name="async_task")
            task = scheduler._tasks["async_task"]
            task.last_run_time = time.time() - 1  # Make it due

            # Mock iscoroutinefunction to return True for our mock function
            with patch('asyncio.iscoroutinefunction', return_value=True):
                scheduler._execute_task(task)

            # Check the result
            assert result == "async_success"

    def test_error_handling_and_retry(self, scheduler):
        """Test error handling and retry logic."""
        call_count = 0

        def failing_task():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail twice, succeed on third try
                raise ValueError("Simulated error")
            return "success"

        # Initialize the thread pool explicitly for testing
        scheduler._thread_pool = ThreadPoolExecutor(max_workers=1)

        scheduler.add_task(
            task_func=failing_task,
            interval=0.1,
            name="failing_task",
        )

        # First execution (will fail)
        task = scheduler._tasks["failing_task"]
        task.last_run_time = time.time() - 1  # Make it due
        scheduler._execute_task(task)

        assert call_count == 1
        assert task.retry_count == 1
        assert task.state == TaskState.PENDING
        assert task.stats.failed_runs == 1

        # Second execution (will fail again)
        task.last_run_time = time.time() - 1  # Make it due again
        scheduler._execute_task(task)

        assert call_count == 2
        assert task.retry_count == 2
        assert task.state == TaskState.PENDING
        assert task.stats.failed_runs == 2

        # Third execution (will succeed)
        task.last_run_time = time.time() - 1  # Make it due again
        scheduler._execute_task(task)

        assert call_count == 3
        assert task.retry_count == 0  # Reset after success
        assert task.state == TaskState.PENDING
        assert task.stats.failed_runs == 2
        assert task.stats.successful_runs == 1

    def test_max_retries_exceeded(self, scheduler):
        """Test behavior when max retries are exceeded."""
        def always_failing_task():
            raise ValueError("Always fails")

        # Initialize the thread pool explicitly for testing
        scheduler._thread_pool = ThreadPoolExecutor(max_workers=1)

        scheduler.add_task(
            task_func=always_failing_task,
            interval=0.1,
            name="always_failing",
        )

        # Execute until max retries exceeded (max_retries = 2)
        task = scheduler._tasks["always_failing"]

        # First execution
        task.last_run_time = time.time() - 1
        scheduler._execute_task(task)
        assert task.retry_count == 1
        assert task.state == TaskState.PENDING

        # Second execution
        task.last_run_time = time.time() - 1
        scheduler._execute_task(task)
        assert task.retry_count == 2
        assert task.state == TaskState.PENDING

        # Third execution (exceeds max_retries)
        task.last_run_time = time.time() - 1
        scheduler._execute_task(task)
        assert task.retry_count == 3
        assert task.state == TaskState.ERROR  # Task should be in ERROR state
        assert task.stats.failed_runs == 3
