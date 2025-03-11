"""
Scanner process implementation for the crypto trading bot.

This module implements a process that periodically scans markets for trading
opportunities using the OpportunityScanner service.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from app.config.config import Config
from app.core.database import Database
from app.core.exceptions import ExchangeError
from app.core.process import BaseProcess
from app.services.exchange_service import ExchangeService
from app.services.market_analyzer import MarketAnalyzer
from app.services.market_filter import MarketFilter
from app.services.scanner import OpportunityScanner


class ScannerProcess(BaseProcess):
    """
    Process that periodically scans markets for trading opportunities.

    This process runs the market scanner at configured intervals,
    handles errors gracefully, and integrates with other components
    of the application.
    """

    def __init__(
        self,
        config: Config,
        exchange_service: ExchangeService,
        market_analyzer: MarketAnalyzer,
        market_filter: MarketFilter,
        database: Optional[Database] = None,
        interval_seconds: float = 300,  # Default to 5 minutes
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the scanner process.

        Args:
            config: Application configuration
            exchange_service: Service for interacting with exchanges
            market_analyzer: Service for analyzing market data
            market_filter: Service for filtering markets
            database: Database instance for storing opportunities
            interval_seconds: Time between scans in seconds
            logger: Logger instance
        """
        super().__init__(
            name="market_scanner",
            interval_seconds=interval_seconds,
            logger=logger or logging.getLogger("process.scanner"),
        )

        self.config = config
        self.exchange_service = exchange_service
        self.market_analyzer = market_analyzer
        self.market_filter = market_filter
        self.database = database

        # Initialize scanner service
        self.scanner = OpportunityScanner(
            exchange_service=exchange_service,
            market_analyzer=market_analyzer,
            database=database,
            logger=self.logger,
        )

        # Set configured symbols or default to None (will be fetched dynamically)
        self._configured_symbols = self._get_configured_symbols()

        # Set configured timeframes or default to scanner defaults
        self._configured_timeframes = self._get_configured_timeframes()

        # Cache for eligible symbols with expiry
        self._symbol_cache: Dict[str, Any] = {}
        self._symbol_cache_expiry: Optional[datetime] = None
        self._symbol_cache_ttl = 3600  # Cache TTL in seconds (1 hour)

        # Process statistics
        self._total_opportunities_found = 0
        self._last_scan_opportunities = 0
        self._last_scan_duration = 0.0

        self.logger.info(f"Scanner process initialized with interval of {interval_seconds}s")

    def _get_configured_symbols(self) -> Optional[List[str]]:
        """Get configured symbols from config or None to use dynamic filtering."""
        try:
            symbols = self.config.get_nested("scanner.symbols", None)
            if symbols and isinstance(symbols, list):
                self.logger.info(f"Using {len(symbols)} configured symbols from config")
                return symbols
            return None
        except Exception as e:
            self.logger.warning(f"Error reading symbol configuration: {e}")
            return None

    def _get_configured_timeframes(self) -> List[str]:
        """Get configured timeframes from config or use defaults."""
        try:
            timeframes = self.config.get_nested("scanner.timeframes", None)
            if timeframes and isinstance(timeframes, list):
                self.logger.info(f"Using configured timeframes: {timeframes}")
                return timeframes
            return self.scanner.default_timeframes
        except Exception as e:
            self.logger.warning(f"Error reading timeframe configuration: {e}")
            return self.scanner.default_timeframes

    def _load_strategies(self) -> None:
        """Load and configure strategies from configuration."""
        try:
            # This implementation is a placeholder - the actual implementation
            # would depend on your strategy loading mechanism
            strategy_configs = self.config.get_nested("strategies", {})
            strategy_names = list(strategy_configs.keys())

            if not strategy_names:
                self.logger.warning("No strategies configured")
                return

            self.logger.info(f"Loading {len(strategy_names)} strategies: {strategy_names}")

            # Example of loading strategies - this would be replaced with actual implementation
            # that instantiates the strategy classes based on configuration
            """
            for strategy_name, strategy_config in strategy_configs.items():
                strategy_class = self._get_strategy_class(strategy_name)
                if strategy_class:
                    strategy = strategy_class(strategy_config)
                    self.scanner.add_strategy(strategy)
            """

            # For now, we'll assume strategy loading happens elsewhere
            # and they're added to the scanner externally
            if not self.scanner.strategies:
                self.logger.warning("No strategies loaded in scanner")

        except Exception as e:
            self.logger.exception(f"Error loading strategies: {e}")

    def _get_eligible_symbols(self) -> List[str]:
        """
        Get the list of eligible symbols to scan.

        Uses configured symbols if available, otherwise applies
        market filtering to get eligible symbols.

        Returns:
            List of symbols to scan
        """
        # Use configured symbols if available
        if self._configured_symbols:
            return self._configured_symbols

        # Check if we have a valid cached result
        current_time = datetime.utcnow()
        if (
            self._symbol_cache
            and self._symbol_cache_expiry
            and current_time < self._symbol_cache_expiry
        ):
            self.logger.debug(f"Using cached symbols ({len(self._symbol_cache)} markets)")
            return list(self._symbol_cache)

        try:
            # Apply market filtering to get eligible symbols
            self.logger.info("Fetching and filtering eligible markets")

            # Get all available symbols
            all_symbols = self.exchange_service.exchange.symbols

            if not all_symbols:
                self.logger.warning("No symbols returned from exchange")
                return []

            # Apply market filter
            self.logger.info(f"Filtering {len(all_symbols)} markets")

            # Configuration for filtering
            market_filter_config = self.config.get_nested("market_filter", {})

            # Apply filtering
            filtered_symbols = self.market_filter.filter_symbols(
                all_symbols,
                min_volume_usd=market_filter_config.get("min_volume_usd", 1000000),
                min_price_usd=market_filter_config.get("min_price_usd", 0.1),
                excluded_symbols=market_filter_config.get("excluded_symbols", []),
                quote_currencies=market_filter_config.get(
                    "quote_currencies", ["USDT", "USD", "BUSD"]
                ),
                max_symbols=market_filter_config.get("max_symbols", 20),
            )

            # Update cache
            self._symbol_cache = filtered_symbols
            self._symbol_cache_expiry = current_time + timedelta(seconds=self._symbol_cache_ttl)

            self.logger.info(f"Filtered to {len(filtered_symbols)} eligible markets")

            return filtered_symbols

        except ExchangeError as e:
            self.logger.error(f"Exchange error getting symbols: {e}")
            # Return cached result if available, otherwise empty list
            return list(self._symbol_cache) if self._symbol_cache else []
        except Exception as e:
            self.logger.exception(f"Unexpected error getting symbols: {e}")
            return list(self._symbol_cache) if self._symbol_cache else []

    def _run_iteration(self) -> None:
        """Run a single iteration of the scanner process."""
        start_time = datetime.utcnow()
        self.logger.info("Starting market scan iteration")

        try:
            # Get eligible symbols to scan
            symbols = self._get_eligible_symbols()

            if not symbols:
                self.logger.warning("No eligible symbols to scan")
                return

            self.logger.info(f"Scanning {len(symbols)} markets")

            # Scan markets for opportunities
            opportunities = self.scanner.scan_markets(
                symbols=symbols, timeframes=self._configured_timeframes
            )

            # Update stats
            self._last_scan_opportunities = len(opportunities)
            self._total_opportunities_found += len(opportunities)

            # Process the opportunities (e.g., notify other components)
            # This would depend on your application's architecture

        except Exception as e:
            self.logger.exception(f"Error in scanner process iteration: {e}")

        finally:
            # Calculate and log duration
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            self._last_scan_duration = duration

            self.logger.info(
                f"Scan iteration completed in {duration:.2f}s with "
                f"{self._last_scan_opportunities} opportunities found"
            )

    def _on_start(self) -> None:
        """Initialize on process start."""
        self.logger.info("Scanner process starting")

        # Load strategies when process starts
        self._load_strategies()

        # Set caching parameters
        self._symbol_cache_ttl = self.config.get_nested(
            "scanner.symbol_cache_ttl", self._symbol_cache_ttl
        )

        # Other initialization if needed

    def _on_stop(self) -> None:
        """Clean up on process stop."""
        self.logger.info(
            f"Scanner process stopping. Total opportunities found: {self._total_opportunities_found}"
        )

        # Clean up resources if needed

    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status information about the scanner process."""
        base_status = self.get_status()

        # Add scanner-specific status information
        scanner_status = {
            **base_status,
            "total_opportunities_found": self._total_opportunities_found,
            "last_scan_opportunities": self._last_scan_opportunities,
            "last_scan_duration": self._last_scan_duration,
            "symbols_count": len(self._symbol_cache) if self._symbol_cache else 0,
            "timeframes": self._configured_timeframes,
            "strategies_count": len(self.scanner.strategies),
            "symbol_cache_ttl": self._symbol_cache_ttl,
            "symbol_cache_expiry": (
                self._symbol_cache_expiry.isoformat() if self._symbol_cache_expiry else None
            ),
        }

        return scanner_status
