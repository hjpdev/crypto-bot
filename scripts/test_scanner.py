#!/usr/bin/env python
"""
Test script for the opportunity scanner process.

This script initializes the required services and runs the opportunity scanner
to verify it's functioning correctly.
"""

import asyncio
import logging
import signal
import sys
import time
from typing import Dict, Any, List, Optional

from app.config.config import Config
from app.core.database import get_db
from app.services.exchange_service import ExchangeService
from app.services.market_analyzer import MarketAnalyzer
from app.services.market_filter import MarketFilter
from app.services.indicator_service import IndicatorService
from app.strategies import MomentumStrategy
from app.processes.scanner_process import ScannerProcess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("test_scanner")


def setup_services(config: Config) -> Dict[str, Any]:
    """
    Initialize services required for the scanner.

    Args:
        config: Application configuration

    Returns:
        Dictionary containing all services
    """
    logger.info("Initializing services...")

    # Get exchange configuration
    exchange_config = config.get_nested("exchange", {})
    exchange_id = exchange_config.get("id", "binance")
    api_key = exchange_config.get("api_key")
    api_secret = exchange_config.get("secret")

    # Initialize exchange service
    logger.info(f"Setting up exchange service for {exchange_id}")
    exchange_service = ExchangeService(
        exchange_id=exchange_id,
        api_key=api_key,
        secret=api_secret,
        enableRateLimit=True,
    )

    # Initialize indicator service
    indicator_service = IndicatorService()

    # Initialize market analyzer
    market_analyzer = MarketAnalyzer(indicator_service)

    # Initialize market filter
    market_filter = MarketFilter(exchange_service)

    # Get database connection
    database = get_db()

    # Return all services in a dictionary
    return {
        "exchange_service": exchange_service,
        "indicator_service": indicator_service,
        "market_analyzer": market_analyzer,
        "market_filter": market_filter,
        "database": database,
    }


def setup_strategies(config: Config) -> List[Any]:
    """
    Initialize and configure trading strategies.

    Args:
        config: Application configuration

    Returns:
        List of configured strategy instances
    """
    logger.info("Setting up strategies...")

    strategies = []

    # Default momentum strategy configuration if not in config
    default_momentum_config = {
        "risk_per_trade": 2.0,              # Required by BaseStrategy
        "max_open_positions": 5,            # Required by BaseStrategy
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "volume_change_threshold": 1.5,
        "trend_ema_period": 50,
        "min_confidence_threshold": 0.6,
        "atr_multiplier": 2.0,
        "risk_reward_targets": [1.5, 2.5, 3.5],
        "timeframes": ["1h", "4h", "1d"]
    }

    # Get momentum strategy configuration or use defaults
    momentum_config = config.get_nested("strategies.momentum", {})

    # Merge with defaults
    for key, value in default_momentum_config.items():
        if key not in momentum_config:
            momentum_config[key] = value

    # Initialize momentum strategy
    momentum_strategy = MomentumStrategy(momentum_config)
    strategies.append(momentum_strategy)

    logger.info(f"Created {len(strategies)} strategies")
    return strategies


def initialize_scanner_process(config: Config, services: Dict[str, Any]) -> ScannerProcess:
    """
    Initialize the scanner process with all services.

    Args:
        config: Application configuration
        services: Dictionary of services

    Returns:
        Configured scanner process instance
    """
    logger.info("Initializing scanner process...")

    scanner_process = ScannerProcess(
        config=config,
        exchange_service=services["exchange_service"],
        market_analyzer=services["market_analyzer"],
        market_filter=services["market_filter"],
        database=services.get("database"),
        interval_seconds=60,  # Use shorter interval for testing
        logger=logging.getLogger("scanner"),
    )

    # Add strategies to the scanner
    strategies = setup_strategies(config)
    for strategy in strategies:
        scanner_process.scanner.add_strategy(strategy)

    logger.info("Scanner process initialized")
    return scanner_process


def setup_signal_handlers(scanner_process: ScannerProcess) -> None:
    """
    Set up signal handlers for graceful shutdown.

    Args:
        scanner_process: Scanner process to stop on signals
    """
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, stopping scanner...")
        scanner_process.stop()
        logger.info("Scanner stopped, exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def get_test_symbols() -> List[str]:
    """
    Get a list of test symbols to scan.

    Returns:
        List of symbol pairs to scan
    """
    # Common market pairs for testing
    return [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "BNB/USDT",
        "ADA/USDT"
    ]


def run_manual_scan(scanner_process: ScannerProcess) -> None:
    """
    Run a manual scan with predefined test symbols.

    Args:
        scanner_process: The scanner process to use
    """
    test_symbols = get_test_symbols()
    logger.info(f"Running manual scan with test symbols: {test_symbols}")

    try:
        # Use scanner directly without relying on the process's automatic scanning
        opportunities = scanner_process.scanner.scan_markets(
            symbols=test_symbols,
            timeframes=scanner_process._configured_timeframes
        )

        logger.info(f"Manual scan complete. Found {len(opportunities)} opportunities")

        # Log each opportunity
        for i, opp in enumerate(opportunities):
            logger.info(f"Opportunity {i+1}: {opp.symbol} - {opp.signal_type.value} - "
                      f"Confidence: {opp.confidence:.2f} - Priority: {opp.priority}")

    except Exception as e:
        logger.exception(f"Error during manual scan: {e}")


def main() -> None:
    """Main function to run the scanner process test."""
    logger.info("Starting scanner test")

    # Load configuration
    config = Config.get_instance()

    # Set up services
    services = setup_services(config)

    # Create and configure scanner process
    scanner_process = initialize_scanner_process(config, services)

    # Set up signal handlers
    setup_signal_handlers(scanner_process)

    # Option 1: Run a manual scan (faster for testing)
    run_manual_scan(scanner_process)

    # Option 2: Start the scanner process (runs continuously)
    use_continuous_mode = True  # Set to True to run the continuous scanner

    if use_continuous_mode:
        # Start the scanner process
        logger.info("Starting scanner process...")
        scanner_process.start()

        try:
            # Monitor the scanner
            logger.info("Scanner running. Press Ctrl+C to stop.")

            # Run for a limited time for testing
            run_time = 300  # 5 minutes for testing

            start_time = time.time()
            while time.time() - start_time < run_time:
                # Print status every 30 seconds
                status = scanner_process.get_detailed_status()
                logger.info(f"Scanner status: {status['state']} - Found {status['last_scan_opportunities']} opportunities in last scan")

                # Sleep between status checks
                time.sleep(30)

            # Stop the scanner after the test period
            logger.info(f"Test complete after {run_time} seconds, stopping scanner...")
            scanner_process.stop()

        except KeyboardInterrupt:
            # This should be caught by the signal handler, but just in case
            logger.info("Keyboard interrupt received, stopping scanner...")
            scanner_process.stop()

    logger.info("Scanner test completed")


if __name__ == "__main__":
    main()