"""
Market data collection task for the crypto trading bot.

This module provides functionality for collecting market data on a schedule,
handling error recovery, and storing the data for use by the trading system.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from app.core.exceptions import APIError, ExchangeConnectionError
from app.core.scheduler import TaskScheduler


class MarketDataCollector:
    """
    Collects and stores market data (OHLCV) for configured trading pairs.

    Features:
    - Scheduled data collection for multiple symbols
    - Error recovery and retry mechanisms
    - Handling of missed intervals
    - Efficient storage of collected data
    """

    def __init__(
        self,
        exchange_service,
        storage_service,
        config_service,
        interval_minutes: int = 5,
        max_retries: int = 3,
        backfill_missing: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the MarketDataCollector.

        Args:
            exchange_service: Service for interacting with exchanges
            storage_service: Service for storing market data
            config_service: Service for accessing system configuration
            interval_minutes: How often to collect data in minutes
            max_retries: Maximum number of retry attempts when collection fails
            backfill_missing: Whether to backfill missing data points
            logger: Logger instance for logging events
        """
        self._exchange_service = exchange_service
        self._storage_service = storage_service
        self._config_service = config_service
        self._interval_minutes = interval_minutes
        self._max_retries = max_retries
        self._backfill_missing = backfill_missing
        self._logger = logger or logging.getLogger(__name__)

        # State tracking
        self._last_collection_time: Dict[str, datetime] = {}
        self._active_symbols: Set[str] = set()
        self._collection_stats: Dict[str, Dict[str, Any]] = {}
        self._error_counts: Dict[str, int] = {}

        self._logger.info(
            f"MarketDataCollector initialized with interval of {interval_minutes} minutes"
        )

    def register_with_scheduler(self, scheduler: TaskScheduler) -> str:
        """
        Register collection tasks with the scheduler.

        Args:
            scheduler: TaskScheduler instance

        Returns:
            Task name that can be used to reference the scheduled task
        """
        task_name = scheduler.add_task(
            task_func=self.run,
            interval=self._interval_minutes * 60,  # Convert to seconds
            name="market_data_collection",
            priority=10,  # High priority task
        )

        # Also register metadata update task (lower frequency)
        metadata_task_name = scheduler.add_task(
            task_func=self.update_cryptocurrency_metadata,
            interval=24 * 60 * 60,  # Once per day
            name="cryptocurrency_metadata_update",
            priority=50,  # Medium priority
        )

        self._logger.info(
            f"Registered market data collection task: {task_name} and "
            f"metadata update task: {metadata_task_name}"
        )

        return task_name

    def add_symbols(self, symbols: List[str]) -> None:
        """
        Add symbols to the collection list.

        Args:
            symbols: List of trading pair symbols (e.g., 'BTC/USDT')
        """
        # Validate symbols first
        valid_symbols = []
        for symbol in symbols:
            try:
                # Check if the exchange supports this symbol
                if self._exchange_service.is_valid_symbol(symbol):
                    valid_symbols.append(symbol)
                else:
                    self._logger.warning(f"Symbol {symbol} is not valid for the exchange")
            except Exception as e:
                self._logger.error(f"Error validating symbol {symbol}: {e}")

        # Add valid symbols to active set
        self._active_symbols.update(valid_symbols)

        # Initialize stats for new symbols
        for symbol in valid_symbols:
            if symbol not in self._collection_stats:
                self._collection_stats[symbol] = {
                    "total_collections": 0,
                    "successful_collections": 0,
                    "failed_collections": 0,
                    "last_successful_time": None,
                    "last_error": None,
                    "last_error_time": None,
                }

        self._logger.info(f"Added {len(valid_symbols)} symbols to collection list")

    def remove_symbols(self, symbols: List[str]) -> None:
        """
        Remove symbols from the collection list.

        Args:
            symbols: List of trading pair symbols to remove
        """
        for symbol in symbols:
            self._active_symbols.discard(symbol)

        self._logger.info(f"Removed {len(symbols)} symbols from collection list")

    def get_active_symbols(self) -> List[str]:
        """
        Get the list of symbols being actively collected.

        Returns:
            List of active symbols
        """
        return list(self._active_symbols)

    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for data collection by symbol.

        Returns:
            Dictionary of collection statistics by symbol
        """
        return self._collection_stats

    def run(self) -> bool:
        """
        Main entry point for scheduled execution.

        Collects market data for all active symbols. This method is called
        by the scheduler at regular intervals.

        Returns:
            True if all collections were successful, False otherwise
        """
        if not self._active_symbols:
            self._logger.info("No active symbols configured for collection")
            return True

        self._logger.info(
            f"Starting market data collection for {len(self._active_symbols)} symbols"
        )
        start_time = time.time()

        # Split symbols into batches if there are too many to prevent timeouts
        batch_size = 10
        symbol_batches = [
            list(self._active_symbols)[i : i + batch_size]
            for i in range(0, len(self._active_symbols), batch_size)
        ]

        all_successful = True
        for batch in symbol_batches:
            batch_success = self.collect_data(batch)
            all_successful = all_successful and batch_success

        duration = time.time() - start_time
        self._logger.info(
            f"Completed market data collection in {duration:.2f}s. "
            f"Status: {'Success' if all_successful else 'Partial failure'}"
        )

        return all_successful

    def collect_data(self, symbols: List[str]) -> bool:
        """
        Collect and store market data for the specified symbols.

        Args:
            symbols: List of symbols to collect data for

        Returns:
            True if all collections were successful, False otherwise
        """
        if not symbols:
            return True

        now = datetime.now()
        all_successful = True

        for symbol in symbols:
            symbol_start_time = time.time()

            try:
                # Determine time range to collect
                last_time = self._last_collection_time.get(symbol)

                # If backfilling is enabled and we have a last collection time
                if self._backfill_missing and last_time:
                    # Calculate expected data points since last collection
                    expected_points = (now - last_time).total_seconds() / (
                        self._interval_minutes * 60
                    )

                    # If we missed more than one interval, backfill the missing data
                    if expected_points > 1.5:  # Allow some tolerance
                        self._logger.info(
                            f"Backfilling {int(expected_points)} missing intervals for {symbol}"
                        )
                        self._backfill_data(symbol, last_time, now)

                # Collect current OHLCV data
                timeframe = self._get_exchange_timeframe()
                ohlcv_data = self._exchange_service.fetch_ohlcv(
                    symbol, timeframe=timeframe, limit=1
                )

                if ohlcv_data and len(ohlcv_data) > 0:
                    # Process and store the data
                    self._process_ohlcv_data(symbol, ohlcv_data)

                    # Update successful collection time
                    self._last_collection_time[symbol] = now
                    self._collection_stats[symbol]["last_successful_time"] = now
                    self._collection_stats[symbol]["successful_collections"] += 1
                    self._error_counts[symbol] = 0  # Reset error count
                else:
                    self._logger.warning(f"No OHLCV data returned for {symbol}")
                    self._update_collection_error(symbol, "No data returned from exchange")
                    all_successful = False

            except (APIError, ExchangeConnectionError) as e:
                # Handle potentially recoverable errors
                self._logger.error(f"API error collecting data for {symbol}: {e}")
                self._update_collection_error(symbol, str(e))
                all_successful = False

            except Exception as e:
                # Handle unexpected errors
                self._logger.exception(f"Unexpected error collecting data for {symbol}: {e}")
                self._update_collection_error(symbol, str(e))
                all_successful = False

            finally:
                # Update total collection count
                self._collection_stats[symbol]["total_collections"] += 1
                duration = time.time() - symbol_start_time
                self._logger.debug(f"Collection for {symbol} completed in {duration:.2f}s")

        return all_successful

    def update_cryptocurrency_metadata(self) -> bool:
        """
        Update metadata for cryptocurrencies.

        This includes information like market cap, volume, and other
        reference data that doesn't change as frequently as price data.

        Returns:
            True if update was successful, False otherwise
        """
        try:
            self._logger.info("Updating cryptocurrency metadata")

            # Get list of all currencies used in active symbols
            currencies = set()
            for symbol in self._active_symbols:
                base, quote = symbol.split("/")
                currencies.add(base)
                currencies.add(quote)

            # Fetch metadata for each currency
            for currency in currencies:
                try:
                    metadata = self._exchange_service.fetch_currency_info(currency)
                    if metadata:
                        # Store the metadata
                        self._storage_service.update_cryptocurrency_metadata(currency, metadata)
                except Exception as e:
                    self._logger.error(f"Error updating metadata for {currency}: {e}")

            self._logger.info(f"Updated metadata for {len(currencies)} currencies")
            return True

        except Exception as e:
            self._logger.exception(f"Failed to update cryptocurrency metadata: {e}")
            return False

    def _backfill_data(self, symbol: str, start_time: datetime, end_time: datetime) -> None:
        """
        Backfill missing data points between start_time and end_time.

        Args:
            symbol: Symbol to backfill data for
            start_time: Start time for backfill
            end_time: End time for backfill
        """
        try:
            # Calculate timeframe in milliseconds for the exchange API
            timeframe = self._get_exchange_timeframe()

            # Fetch historical data
            ohlcv_data = self._exchange_service.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=int(start_time.timestamp() * 1000),  # Convert to milliseconds
                limit=100,  # Reasonable limit to prevent very large requests
            )

            if ohlcv_data and len(ohlcv_data) > 0:
                # Process and store the data
                self._process_ohlcv_data(symbol, ohlcv_data)
                self._logger.info(f"Backfilled {len(ohlcv_data)} data points for {symbol}")
            else:
                self._logger.warning(f"No data available for backfill for {symbol}")

        except Exception as e:
            self._logger.error(f"Error during backfill for {symbol}: {e}")

    def _process_ohlcv_data(self, symbol: str, ohlcv_data: List[List[float]]) -> None:
        """
        Process and store OHLCV data.

        Args:
            symbol: Symbol the data is for
            ohlcv_data: List of OHLCV candles from the exchange
        """
        processed_data = []

        for candle in ohlcv_data:
            # Standard OHLCV format is [timestamp, open, high, low, close, volume]
            if len(candle) >= 6:
                timestamp_ms = candle[0]
                candle_time = datetime.fromtimestamp(timestamp_ms / 1000.0)

                processed_data.append(
                    {
                        "symbol": symbol,
                        "timestamp": candle_time,
                        "open": candle[1],
                        "high": candle[2],
                        "low": candle[3],
                        "close": candle[4],
                        "volume": candle[5],
                    }
                )

        if processed_data:
            # Store the processed data
            self._storage_service.store_ohlcv_data(processed_data)

    def _update_collection_error(self, symbol: str, error_message: str) -> None:
        """
        Update error statistics for a symbol.

        Args:
            symbol: Symbol that encountered an error
            error_message: Description of the error
        """
        now = datetime.now()

        # Update collection stats
        self._collection_stats[symbol]["failed_collections"] += 1
        self._collection_stats[symbol]["last_error"] = error_message
        self._collection_stats[symbol]["last_error_time"] = now

        # Increment error count
        self._error_counts[symbol] = self._error_counts.get(symbol, 0) + 1

        # Log if we're approaching max retries
        if self._error_counts[symbol] >= self._max_retries:
            self._logger.warning(
                f"Symbol {symbol} has failed {self._error_counts[symbol]} times consecutively"
            )

    def _get_exchange_timeframe(self) -> str:
        """
        Convert internal interval to exchange timeframe format.

        Returns:
            String timeframe in exchange format (e.g., '5m', '1h')
        """
        # Common exchange timeframe format mapping
        if self._interval_minutes < 1:
            return f"{int(self._interval_minutes * 60)}s"
        elif self._interval_minutes < 60:
            return f"{self._interval_minutes}m"
        elif self._interval_minutes < 1440:  # Less than 1 day
            return f"{self._interval_minutes // 60}h"
        else:
            return f"{self._interval_minutes // 1440}d"
