"""
Processes module for the crypto trading bot.

This module provides background processes that run continuously
to perform various tasks such as scanning markets, managing positions,
and monitoring system health.
"""

from app.processes.scanner_process import ScannerProcess

__all__ = [
    "ScannerProcess",
]
