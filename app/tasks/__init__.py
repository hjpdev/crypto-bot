"""
Task modules for the crypto trading bot.

This package contains task classes that are designed to run on a schedule
via the TaskScheduler system.
"""

from app.tasks.market_data_collector import MarketDataCollector
from app.tasks.performance_calculator import PerformanceCalculator
from app.tasks.data_integrity_checker import DataIntegrityChecker

__all__ = [
    "MarketDataCollector",
    "PerformanceCalculator",
    "DataIntegrityChecker",
]
