from app.core.database import db, get_db, get_engine, create_all_tables, Base, Database
from app.core.process import BaseProcess, ProcessState
from app.core.scheduler import TaskScheduler, TaskState, ScheduledTask, TaskStatistics

from app.core.exceptions import (
    CryptoBotError,
    ConfigError,
    APIError,
    ExchangeError,
    ExchangeConnectionError,
    DatabaseError,
    DatabaseConnectionError,
    StrategyError,
    ValidationError,
    AuthenticationError,
    InsufficientFundsError,
    OrderError,
)

__all__ = [
    # Database exports
    "db",
    "get_db",
    "get_engine",
    "create_all_tables",
    "Base",
    "Database",
    # Process exports
    "BaseProcess",
    "ProcessState",
    # Scheduler exports
    "TaskScheduler",
    "TaskState",
    "ScheduledTask",
    "TaskStatistics",
    # Exception exports
    "CryptoBotError",
    "ConfigError",
    "APIError",
    "ExchangeError",
    "ExchangeConnectionError",
    "DatabaseError",
    "DatabaseConnectionError",
    "StrategyError",
    "ValidationError",
    "AuthenticationError",
    "InsufficientFundsError",
    "OrderError",
]
