"""
Comprehensive data collection service for cryptocurrency market data.

This module provides a centralized service for collecting various types of market data,
coordinating collection activities, and ensuring data integrity and availability.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import asyncio
import time
import pandas as pd

from app.core.exceptions import (
    APIError,
    ExchangeConnectionError,
    ValidationError,
)


class DataCollector:
    """
    Coordinates various data collection activities, manages data storage and organization,
    and handles both historical and real-time data.

    The DataCollector serves as a high-level abstraction for all data collection operations,
    providing a unified interface for collecting different types of market data.

    Features:
    - Collects OHLCV data for specified symbols and timeframes
    - Collects order book snapshots
    - Creates comprehensive market snapshots
    - Backfills missing data
    - Validates collected data
    - Supports both synchronous and asynchronous collection
    """

    def __init__(
        self,
        exchange_service,
        data_storage,
        session_provider,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_concurrent_requests: int = 5,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the DataCollector with necessary services and configuration.

        Args:
            exchange_service: Service for interacting with exchanges
            data_storage: Service for storing collected data
            session_provider: Callable that provides a database session
            max_retries: Maximum number of retry attempts for failed operations
            retry_delay: Initial delay between retries in seconds
            backoff_factor: Factor by which the retry delay increases after each attempt
            max_concurrent_requests: Maximum number of concurrent requests for async operations
            logger: Logger instance for logging events
        """
        self._exchange_service = exchange_service
        self._data_storage = data_storage
        self._session_provider = session_provider
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._backoff_factor = backoff_factor
        self._max_concurrent_requests = max_concurrent_requests
        self._logger = logger or logging.getLogger(__name__)

        # Collection statistics tracking
        self._collection_stats: Dict[str, Dict[str, Any]] = {}

        self._logger.info("DataCollector initialized")

    def collect_ohlcv(
        self,
        symbols: List[str],
        timeframes: List[str] = ["1h"],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        validate: bool = True,
        store: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Collect OHLCV data for specified symbols and timeframes.

        Args:
            symbols: List of trading symbols (e.g., "BTC/USD")
            timeframes: List of timeframes to collect (e.g., ["1m", "1h", "1d"])
            start_time: Start time for data collection (None for recent data)
            end_time: End time for data collection (None for up to current time)
            validate: Whether to validate data before storing
            store: Whether to store the collected data

        Returns:
            A dictionary mapping symbols to their collection results

        Raises:
            ExchangeConnectionError: If there is a connection issue with the exchange
            ValidationError: If data validation fails
            DataError: If there is an issue with data processing
        """
        results = {}

        for symbol in symbols:
            symbol_results = {}
            for timeframe in timeframes:
                try:
                    # Convert to milliseconds timestamp if needed
                    since = int(start_time.timestamp() * 1000) if start_time else None
                    until = int(end_time.timestamp() * 1000) if end_time else None

                    self._logger.info(
                        f"Collecting OHLCV data for {symbol} on {timeframe} timeframe"
                    )

                    if since and until:
                        ohlcv_data = self._exchange_service.fetch_historical_ohlcv(
                            symbol, timeframe, since, until
                        )
                    else:
                        ohlcv_data = self._exchange_service.fetch_ohlcv(symbol, timeframe, since)

                    if not ohlcv_data:
                        self._logger.warning(f"No OHLCV data available for {symbol} on {timeframe}")
                        symbol_results[timeframe] = {"status": "no_data", "count": 0}
                        continue

                    if validate:
                        self.validate_collected_data(ohlcv_data)

                    if store:
                        stored_count = self._data_storage.store_ohlcv(symbol, timeframe, ohlcv_data)
                        self._logger.info(
                            f"Stored {stored_count} OHLCV records for {symbol} on {timeframe}"
                        )
                        symbol_results[timeframe] = {"status": "success", "count": stored_count}
                    else:
                        symbol_results[timeframe] = {
                            "status": "success",
                            "count": len(ohlcv_data),
                            "data": ohlcv_data,
                        }

                    # Update collection stats
                    self._update_collection_stats(symbol, "ohlcv", timeframe, True)

                except (ExchangeConnectionError, APIError) as e:
                    self._logger.error(
                        f"Exchange error while collecting {symbol} on {timeframe}: {str(e)}"
                    )
                    symbol_results[timeframe] = {"status": "exchange_error", "error": str(e)}
                    self._update_collection_stats(symbol, "ohlcv", timeframe, False, str(e))

                except ValidationError as e:
                    self._logger.error(f"Validation error for {symbol} on {timeframe}: {str(e)}")
                    symbol_results[timeframe] = {"status": "validation_error", "error": str(e)}
                    self._update_collection_stats(symbol, "ohlcv", timeframe, False, str(e))

                except Exception as e:
                    self._logger.exception(
                        f"Unexpected error collecting {symbol} on {timeframe}: {str(e)}"
                    )
                    symbol_results[timeframe] = {"status": "error", "error": str(e)}
                    self._update_collection_stats(symbol, "ohlcv", timeframe, False, str(e))

            results[symbol] = symbol_results

        return results

    async def collect_ohlcv_async(
        self,
        symbols: List[str],
        timeframes: List[str] = ["1h"],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Asynchronously collect OHLCV data for multiple symbols and timeframes.

        Args:
            symbols: List of trading symbols (e.g., "BTC/USD")
            timeframes: List of timeframes to collect (e.g., ["1m", "1h", "1d"])
            start_time: Start time for data collection (None for recent data)
            end_time: End time for data collection (None for up to current time)

        Returns:
            A dictionary mapping symbols to their collection results
        """
        # Create a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self._max_concurrent_requests)

        async def collect_single(symbol: str, timeframe: str) -> Tuple[str, str, Dict[str, Any]]:
            async with semaphore:
                try:
                    result = self.collect_ohlcv([symbol], [timeframe], start_time, end_time)
                    return symbol, timeframe, result.get(symbol, {}).get(timeframe, {})
                except Exception as e:
                    self._logger.exception(
                        f"Async collection error for {symbol} on {timeframe}: {str(e)}"
                    )
                    return symbol, timeframe, {"status": "error", "error": str(e)}

        # Create collection tasks
        tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                tasks.append(collect_single(symbol, timeframe))

        # Execute tasks concurrently
        results = {}
        for symbol, timeframe, result in await asyncio.gather(*tasks):
            if symbol not in results:
                results[symbol] = {}
            results[symbol][timeframe] = result

        return results

    def collect_order_book(
        self,
        symbols: List[str],
        depth: Optional[int] = None,
        store: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Collect order book snapshots for specified symbols.

        Args:
            symbols: List of trading symbols (e.g., "BTC/USD")
            depth: Depth of the order book to collect (None for exchange default)
            store: Whether to store the collected snapshots

        Returns:
            A dictionary mapping symbols to their order book snapshots

        Raises:
            ExchangeConnectionError: If there is a connection issue with the exchange
            ValidationError: If data validation fails
        """
        results = {}

        for symbol in symbols:
            try:
                self._logger.info(f"Collecting order book for {symbol} with depth {depth}")

                # Implement with retry logic
                retry_count = 0
                current_delay = self._retry_delay

                while retry_count <= self._max_retries:
                    try:
                        order_book = self._exchange_service.fetch_order_book_snapshot(
                            symbol, depth=depth
                        )
                        break
                    except (ExchangeConnectionError, APIError) as e:
                        retry_count += 1
                        if retry_count > self._max_retries:
                            raise

                        self._logger.warning(
                            f"Retry {retry_count}/{self._max_retries} for {symbol} order book: {str(e)}"
                        )
                        time.sleep(current_delay)
                        current_delay *= self._backoff_factor

                if store:
                    timestamp = datetime.utcnow()
                    self._data_storage.store_order_book(symbol, order_book, timestamp)
                    results[symbol] = {"status": "success", "timestamp": timestamp}
                else:
                    results[symbol] = {"status": "success", "data": order_book}

                # Update collection stats
                self._update_collection_stats(symbol, "order_book", None, True)

            except (ExchangeConnectionError, APIError) as e:
                self._logger.error(f"Exchange error collecting order book for {symbol}: {str(e)}")
                results[symbol] = {"status": "exchange_error", "error": str(e)}
                self._update_collection_stats(symbol, "order_book", None, False, str(e))

            except Exception as e:
                self._logger.exception(
                    f"Unexpected error collecting order book for {symbol}: {str(e)}"
                )
                results[symbol] = {"status": "error", "error": str(e)}
                self._update_collection_stats(symbol, "order_book", None, False, str(e))

        return results

    def collect_market_snapshots(
        self,
        symbols: List[str],
        include_trades: bool = True,
        include_order_book: bool = True,
        include_ticker: bool = True,
        store: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create comprehensive market snapshots for specified symbols.

        A market snapshot includes various market data points like ticker,
        order book, recent trades, and other relevant information captured
        at a specific point in time.

        Args:
            symbols: List of trading symbols (e.g., "BTC/USD")
            include_trades: Whether to include recent trades in the snapshot
            include_order_book: Whether to include order book in the snapshot
            include_ticker: Whether to include ticker data in the snapshot
            store: Whether to store the collected snapshots

        Returns:
            A dictionary mapping symbols to their market snapshots
        """
        results = {}
        timestamp = datetime.utcnow()

        for symbol in symbols:
            try:
                self._logger.info(f"Creating market snapshot for {symbol}")
                snapshot_data = {"timestamp": timestamp}

                # Collect ticker data if requested
                if include_ticker:
                    try:
                        ticker = self._exchange_service.get_ticker(symbol)
                        snapshot_data["ticker"] = ticker
                    except Exception as e:
                        self._logger.warning(f"Failed to get ticker for {symbol}: {str(e)}")
                        snapshot_data["ticker"] = None

                # Collect order book if requested
                if include_order_book:
                    try:
                        order_book = self._exchange_service.fetch_order_book_snapshot(symbol)
                        snapshot_data["order_book"] = order_book
                    except Exception as e:
                        self._logger.warning(f"Failed to get order book for {symbol}: {str(e)}")
                        snapshot_data["order_book"] = None

                # Collect recent trades if requested
                if include_trades:
                    try:
                        trades = self._exchange_service.fetch_recent_trades(symbol)
                        snapshot_data["trades"] = trades
                    except Exception as e:
                        self._logger.warning(f"Failed to get recent trades for {symbol}: {str(e)}")
                        snapshot_data["trades"] = None

                if store:
                    self._data_storage.store_market_snapshot(symbol, snapshot_data)
                    results[symbol] = {"status": "success", "timestamp": timestamp}
                else:
                    results[symbol] = {"status": "success", "data": snapshot_data}

                # Update collection stats
                self._update_collection_stats(symbol, "market_snapshot", None, True)

            except Exception as e:
                self._logger.exception(f"Error creating market snapshot for {symbol}: {str(e)}")
                results[symbol] = {"status": "error", "error": str(e)}
                self._update_collection_stats(symbol, "market_snapshot", None, False, str(e))

        return results

    def backfill_missing_data(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        start_time: datetime = None,
        end_time: datetime = None,
        check_only: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Identify and backfill gaps in OHLCV data.

        Args:
            symbols: List of trading symbols (e.g., "BTC/USD")
            timeframe: Timeframe to check for gaps (e.g., "1h")
            start_time: Start time for backfill check
            end_time: End time for backfill check (defaults to current time)
            check_only: If True, only check for gaps but don't fill them

        Returns:
            A dictionary with backfill results for each symbol
        """
        results = {}
        end_time = end_time or datetime.utcnow()

        for symbol in symbols:
            try:
                self._logger.info(f"Checking data continuity for {symbol} on {timeframe} timeframe")

                # Get gaps in the data
                gaps = self._data_storage.check_data_continuity(
                    symbol, timeframe, start_time, end_time
                )

                if not gaps:
                    self._logger.info(f"No gaps found for {symbol} on {timeframe}")
                    results[symbol] = {"status": "no_gaps", "gaps_found": 0}
                    continue

                results[symbol] = {
                    "status": "gaps_found" if check_only else "backfilled",
                    "gaps_found": len(gaps),
                    "gap_periods": [
                        {"start": gap[0].isoformat(), "end": gap[1].isoformat()} for gap in gaps
                    ],
                }

                if check_only:
                    continue

                # Backfill each gap
                for gap_start, gap_end in gaps:
                    self._logger.info(
                        f"Backfilling {symbol} on {timeframe} from {gap_start} to {gap_end}"
                    )

                    # Convert to milliseconds for the exchange API
                    since = int(gap_start.timestamp() * 1000)
                    until = int(gap_end.timestamp() * 1000)

                    # Fetch and store the missing data
                    ohlcv_data = self._exchange_service.fetch_historical_ohlcv(
                        symbol, timeframe, since, until
                    )

                    if ohlcv_data:
                        stored_count = self._data_storage.store_ohlcv(symbol, timeframe, ohlcv_data)
                        self._logger.info(
                            f"Backfilled {stored_count} records for {symbol} on {timeframe}"
                        )
                    else:
                        self._logger.warning(
                            f"No data available to backfill for {symbol} on {timeframe} "
                            f"from {gap_start} to {gap_end}"
                        )

            except Exception as e:
                self._logger.exception(
                    f"Error during backfill for {symbol} on {timeframe}: {str(e)}"
                )
                results[symbol] = {"status": "error", "error": str(e)}

        return results

    def validate_collected_data(self, data: Any) -> bool:
        """
        Validate the integrity and quality of collected data.

        This performs various checks to ensure the data is valid and meets
        quality standards before storage.

        Args:
            data: The data to validate (type depends on data being validated)

        Returns:
            True if data passes validation, False otherwise

        Raises:
            ValidationError: If validation fails with details about the failure
        """
        # The implementation will vary based on the type of data
        if isinstance(data, list) and all(isinstance(item, list) for item in data):
            # Looks like OHLCV data
            return self._validate_ohlcv_data(data)
        elif isinstance(data, dict) and "bids" in data and "asks" in data:
            # Looks like order book data
            return self._validate_order_book_data(data)
        elif isinstance(data, dict) and "timestamp" in data:
            # Looks like a market snapshot
            return self._validate_market_snapshot(data)
        else:
            raise ValidationError(f"Unknown data format for validation: {type(data)}")

    def _validate_ohlcv_data(self, ohlcv_data: List[List[Union[int, float]]]) -> bool:
        """
        Validate OHLCV data for integrity and correctness.

        Args:
            ohlcv_data: List of OHLCV candles

        Returns:
            True if data passes validation

        Raises:
            ValidationError: If validation fails
        """
        if not ohlcv_data:
            raise ValidationError("OHLCV data is empty")

        # Check data structure
        for i, candle in enumerate(ohlcv_data):
            if len(candle) < 6:
                raise ValidationError(f"OHLCV candle at index {i} has incomplete data: {candle}")

            # Unpack values for validation
            timestamp, open_price, high_price, low_price, close_price, volume = candle[:6]

            # Type checking
            if not isinstance(timestamp, (int, float)):
                raise ValidationError(f"Timestamp at index {i} is not a number: {timestamp}")

            for value, name in [
                (open_price, "open price"),
                (high_price, "high price"),
                (low_price, "low price"),
                (close_price, "close price"),
                (volume, "volume"),
            ]:
                if not isinstance(value, (int, float)):
                    raise ValidationError(
                        f"{name.capitalize()} at index {i} is not a number: {value}"
                    )

            # Price consistency checks
            if low_price > high_price:
                raise ValidationError(
                    f"Low price ({low_price}) is greater than high price ({high_price}) at index {i}"
                )

            if open_price < low_price or open_price > high_price:
                raise ValidationError(
                    f"Open price ({open_price}) is outside the range of low ({low_price}) "
                    f"and high ({high_price}) at index {i}"
                )

            if close_price < low_price or close_price > high_price:
                raise ValidationError(
                    f"Close price ({close_price}) is outside the range of low ({low_price}) "
                    f"and high ({high_price}) at index {i}"
                )

            # Volume should be non-negative
            if volume < 0:
                raise ValidationError(f"Volume is negative at index {i}: {volume}")

        # Check for timestamp ordering if there are multiple candles
        if len(ohlcv_data) > 1:
            for i in range(1, len(ohlcv_data)):
                if ohlcv_data[i][0] <= ohlcv_data[i - 1][0]:
                    raise ValidationError(
                        f"Timestamp at index {i} ({ohlcv_data[i][0]}) is not greater than "
                        f"the previous timestamp ({ohlcv_data[i-1][0]})"
                    )

        return True

    def _validate_order_book_data(self, order_book: Dict[str, Any]) -> bool:
        """
        Validate order book data for integrity and correctness.

        Args:
            order_book: Order book data from exchange

        Returns:
            True if data passes validation

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(order_book, dict):
            raise ValidationError(f"Order book is not a dictionary: {type(order_book)}")

        # Check required fields
        required_fields = ["bids", "asks"]
        for field in required_fields:
            if field not in order_book:
                raise ValidationError(f"Order book missing required field: {field}")

        # Validate bids and asks structures
        for side, name in [("bids", "bids"), ("asks", "asks")]:
            if not isinstance(order_book[side], list):
                raise ValidationError(f"Order book {name} is not a list: {type(order_book[side])}")

            for i, order in enumerate(order_book[side]):
                if not isinstance(order, list) or len(order) < 2:
                    raise ValidationError(f"Invalid {name} entry at index {i}: {order}")

                price, size = order[:2]

                # Check types
                if not isinstance(price, (int, float)):
                    raise ValidationError(f"Price in {name} at index {i} is not a number: {price}")

                if not isinstance(size, (int, float)):
                    raise ValidationError(f"Size in {name} at index {i} is not a number: {size}")

                # Check values
                if price <= 0:
                    raise ValidationError(f"Price in {name} at index {i} is not positive: {price}")

                if size < 0:
                    raise ValidationError(f"Size in {name} at index {i} is negative: {size}")

        # Check price ordering (bids highest to lowest, asks lowest to highest)
        if len(order_book["bids"]) > 1:
            for i in range(1, len(order_book["bids"])):
                if order_book["bids"][i][0] > order_book["bids"][i - 1][0]:
                    raise ValidationError(
                        f"Bids are not in descending order at index {i}: "
                        f"{order_book['bids'][i-1][0]} -> {order_book['bids'][i][0]}"
                    )

        if len(order_book["asks"]) > 1:
            for i in range(1, len(order_book["asks"])):
                if order_book["asks"][i][0] < order_book["asks"][i - 1][0]:
                    raise ValidationError(
                        f"Asks are not in ascending order at index {i}: "
                        f"{order_book['asks'][i-1][0]} -> {order_book['asks'][i][0]}"
                    )

        # Check for crossed book (highest bid > lowest ask)
        if order_book["bids"] and order_book["asks"]:
            highest_bid = order_book["bids"][0][0]
            lowest_ask = order_book["asks"][0][0]

            if highest_bid >= lowest_ask:
                raise ValidationError(
                    f"Order book is crossed: highest bid ({highest_bid}) >= lowest ask ({lowest_ask})"
                )

        return True

    def _validate_market_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """
        Validate market snapshot data for integrity and completeness.

        Args:
            snapshot: Market snapshot data

        Returns:
            True if data passes validation

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(snapshot, dict):
            raise ValidationError(f"Market snapshot is not a dictionary: {type(snapshot)}")

        # Check timestamp exists and is valid
        if "timestamp" not in snapshot:
            raise ValidationError("Market snapshot missing timestamp")

        if not isinstance(snapshot["timestamp"], datetime):
            raise ValidationError(
                f"Market snapshot timestamp is not a datetime object: {type(snapshot['timestamp'])}"
            )

        # Validate components if they exist
        if "ticker" in snapshot and snapshot["ticker"] is not None:
            if not isinstance(snapshot["ticker"], dict):
                raise ValidationError(
                    f"Ticker in market snapshot is not a dictionary: {type(snapshot['ticker'])}"
                )

            # Check required ticker fields
            required_ticker_fields = ["last", "bid", "ask", "volume"]
            for field in required_ticker_fields:
                if field not in snapshot["ticker"]:
                    raise ValidationError(f"Ticker missing required field: {field}")

        if "order_book" in snapshot and snapshot["order_book"] is not None:
            try:
                self._validate_order_book_data(snapshot["order_book"])
            except ValidationError as e:
                raise ValidationError(f"Order book validation failed: {str(e)}")

        if "trades" in snapshot and snapshot["trades"] is not None:
            if not isinstance(snapshot["trades"], list):
                raise ValidationError(
                    f"Trades in market snapshot is not a list: {type(snapshot['trades'])}"
                )

            # Validate each trade if there are any
            for i, trade in enumerate(snapshot["trades"]):
                if not isinstance(trade, dict):
                    raise ValidationError(f"Trade at index {i} is not a dictionary: {type(trade)}")

                # Check minimal trade fields
                minimal_trade_fields = ["price", "amount", "side"]
                for field in minimal_trade_fields:
                    if field not in trade:
                        raise ValidationError(f"Trade at index {i} missing field: {field}")

        return True

    def _update_collection_stats(
        self,
        symbol: str,
        data_type: str,
        timeframe: Optional[str],
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Update collection statistics for tracking and monitoring.

        Args:
            symbol: Trading symbol
            data_type: Type of data collected
            timeframe: Timeframe for the data (if applicable)
            success: Whether collection was successful
            error_message: Error message if collection failed
        """
        key = f"{symbol}:{data_type}"
        if timeframe:
            key += f":{timeframe}"

        if key not in self._collection_stats:
            self._collection_stats[key] = {
                "symbol": symbol,
                "data_type": data_type,
                "timeframe": timeframe,
                "total_attempts": 0,
                "successful_attempts": 0,
                "last_attempt": None,
                "last_success": None,
                "last_error": None,
                "error_count": 0,
            }

        now = datetime.utcnow()
        stats = self._collection_stats[key]

        stats["total_attempts"] += 1
        stats["last_attempt"] = now

        if success:
            stats["successful_attempts"] += 1
            stats["last_success"] = now
        else:
            stats["error_count"] += 1
            stats["last_error"] = error_message

    def get_collection_stats(
        self,
        symbol: Optional[str] = None,
        data_type: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get collection statistics for monitoring and reporting.

        Args:
            symbol: Filter stats by symbol
            data_type: Filter stats by data type
            timeframe: Filter stats by timeframe

        Returns:
            Filtered collection statistics
        """
        if not symbol and not data_type and not timeframe:
            return self._collection_stats

        filtered_stats = {}

        for key, stats in self._collection_stats.items():
            match = True

            if symbol and stats["symbol"] != symbol:
                match = False

            if data_type and stats["data_type"] != data_type:
                match = False

            if timeframe and stats["timeframe"] != timeframe:
                match = False

            if match:
                filtered_stats[key] = stats

        return filtered_stats

    def reset_collection_stats(self) -> None:
        """Reset all collection statistics."""
        self._collection_stats = {}
        self._logger.info("Collection statistics have been reset")

    def export_collection_stats(self) -> pd.DataFrame:
        """
        Export collection statistics as a pandas DataFrame.

        Returns:
            DataFrame containing collection statistics
        """
        stats_list = list(self._collection_stats.values())
        if not stats_list:
            return pd.DataFrame()

        return pd.DataFrame(stats_list)
