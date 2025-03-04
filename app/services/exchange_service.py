"""
Exchange service for interacting with cryptocurrency exchanges via CCXT.

This module provides a service for interacting with cryptocurrency exchanges
through the CCXT library, handling authentication, rate limiting, and error handling.
"""

import ccxt
import functools
import logging
import time
import math

from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from app.services.exchange_rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class ExchangeService:
    """
    Service for interacting with cryptocurrency exchanges via CCXT.

    Provides methods for common exchange operations with proper rate limiting,
    error handling, and caching where appropriate.
    """

    # Mapping of timeframe to milliseconds for easy conversion
    TIMEFRAME_TO_MILLISECONDS = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "8h": 8 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        "3d": 3 * 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000,
        "1M": 30 * 24 * 60 * 60 * 1000,
    }

    # Common timeframe categories
    TIMEFRAME_CATEGORIES = {
        "minutes": ["1m", "3m", "5m", "15m", "30m"],
        "hours": ["1h", "2h", "4h", "6h", "8h", "12h"],
        "days": ["1d", "3d"],
        "weeks": ["1w"],
        "months": ["1M"],
    }

    # Exchange error mapping
    COMMON_ERROR_TYPES = {
        "rate_limit": ["RateLimitExceeded", "DDoSProtection", "ExchangeNotAvailable"],
        "authentication": ["AuthenticationError", "PermissionDenied", "AccountSuspended"],
        "insufficient_funds": ["InsufficientFunds"],
        "invalid_order": [
            "InvalidOrder",
            "OrderNotFound",
            "OrderNotCached",
            "CancelPending",
            "OrderImmediatelyFillable",
            "OrderNotFillable",
        ],
        "network": ["NetworkError", "ExchangeNotAvailable", "RequestTimeout", "ExchangeError"],
    }

    def __init__(
        self,
        exchange_id: str,
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        password: Optional[str] = None,
        sandbox: bool = False,
        timeout: int = 30000,  # 30 seconds in milliseconds
        enableRateLimit: bool = True,
        rate_limit_calls_per_second: float = 1.0,
        max_retries: int = 5,
        cache_ttl: int = 60,  # Default 60 seconds for caching
        verbose: bool = False,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize the exchange service.

        Args:
            exchange_id: CCXT exchange ID (e.g., 'binance', 'coinbase', etc.)
            api_key: API key for authentication
            secret: API secret for authentication
            password: Additional password/passphrase if required by exchange
            sandbox: Whether to use the exchange's sandbox/testnet
            timeout: API call timeout in milliseconds
            enableRateLimit: Whether to enable CCXT's built-in rate limiting
            rate_limit_calls_per_second: Maximum calls per second for our custom rate limiter
            max_retries: Maximum retry attempts on failure
            cache_ttl: Time-to-live for cached data in seconds
            verbose: Whether to enable verbose logging
            initial_backoff: Initial backoff time in seconds
            backoff_factor: Multiplier for exponential backoff
            jitter: Whether to add random jitter to backoff times
            **kwargs: Additional parameters to pass to the CCXT exchange constructor
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.secret = secret
        self.password = password
        self.sandbox = sandbox
        self.timeout = timeout
        self.enableRateLimit = enableRateLimit
        self.verbose = verbose
        self.cache_ttl = cache_ttl
        self.exchange_kwargs = kwargs

        self.rate_limiter = RateLimiter(
            calls_per_second=rate_limit_calls_per_second,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            backoff_factor=backoff_factor,
            jitter=jitter,
        )

        self._exchange = self._initialize_exchange()

        self._markets_cache: Dict[str, Any] = {}
        self._markets_cache_timestamp: Optional[float] = None

        self._tickers_cache: Dict[str, Dict[str, Any]] = {}
        self._tickers_cache_timestamp: Dict[str, float] = {}

        # Add new caches for the additional methods
        self._order_book_cache: Dict[str, Dict[str, Any]] = {}
        self._order_book_cache_timestamp: Dict[str, float] = {}

        self._recent_trades_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._recent_trades_cache_timestamp: Dict[str, float] = {}

        self._funding_rate_cache: Dict[str, Dict[str, Any]] = {}
        self._funding_rate_cache_timestamp: Dict[str, float] = {}

        self._symbol_metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._symbol_metadata_cache_timestamp: Dict[str, float] = {}

    def _initialize_exchange(self) -> ccxt.Exchange:
        if self.exchange_id not in ccxt.exchanges:
            raise ValueError(f"Exchange '{self.exchange_id}' is not supported by CCXT")

        exchange_class = getattr(ccxt, self.exchange_id)

        config: Dict[str, Any] = {
            "enableRateLimit": self.enableRateLimit,
            "timeout": self.timeout,
            **self.exchange_kwargs,
        }

        if self.api_key and self.secret:
            config["apiKey"] = self.api_key
            config["secret"] = self.secret

            if self.password:
                config["password"] = self.password

        exchange = exchange_class(config)

        if self.sandbox and "test" in exchange.urls:
            exchange.set_sandbox_mode(True)
            logger.info(f"Using sandbox mode for {self.exchange_id}")

        return exchange

    def get_exchange(self) -> ccxt.Exchange:
        return self._exchange

    def _is_cache_valid(self, timestamp: Optional[float], ttl: int | None = None) -> bool:
        """Check if cached data is still valid."""

        if timestamp is None:
            return False

        if ttl is None:
            ttl = self.cache_ttl

        return (time.time() - timestamp) < ttl

    @lru_cache(maxsize=128)
    def _get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information including rate limits and capabilities."""
        return {
            "id": self._exchange.id,
            "name": self._exchange.name,
            "countries": getattr(self._exchange, "countries", None),
            "rateLimit": getattr(self._exchange, "rateLimit", None),
            "has": getattr(self._exchange, "has", {}),
            "urls": getattr(self._exchange, "urls", {}),
            "version": getattr(self._exchange, "version", None),
            "has_fetch_ohlcv": self._exchange.has.get("fetchOHLCV", False),
            "has_fetch_order_book": self._exchange.has.get("fetchOrderBook", False),
            "timeframes": getattr(self._exchange, "timeframes", None),
        }

    @functools.wraps(ccxt.Exchange.fetch_markets)
    def fetch_markets(
        self, params: Dict[str, Any] = {}, force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Fetch available markets from the exchange with caching."""

        if not force_refresh and self._is_cache_valid(self._markets_cache_timestamp):
            return self._markets_cache

        fetch_markets_with_backoff = self.rate_limiter.with_backoff(self._exchange.fetch_markets)
        markets = fetch_markets_with_backoff(params=params)

        self._markets_cache = markets
        self._markets_cache_timestamp = time.time()

        return markets

    def get_ticker(
        self, symbol: str, params: Dict[str, Any] = {}, cache_ttl: int | None = None
    ) -> Dict[str, Any]:
        """
        Get current ticker data for a symbol with caching.

        Args:
            symbol: Market symbol (e.g., 'BTC/USD')
            params: Additional parameters to pass to the API
            cache_ttl: Time-to-live for cache in seconds (uses instance default if None)
        """

        if cache_ttl is None:
            cache_ttl = self.cache_ttl

        if (
            symbol in self._tickers_cache
            and symbol in self._tickers_cache_timestamp
            and self._is_cache_valid(self._tickers_cache_timestamp.get(symbol), cache_ttl)
        ):
            return self._tickers_cache[symbol]

        fetch_ticker_with_backoff = self.rate_limiter.with_backoff(self._exchange.fetch_ticker)
        ticker = fetch_ticker_with_backoff(symbol, params=params)

        self._tickers_cache[symbol] = ticker
        self._tickers_cache_timestamp[symbol] = time.time()

        return ticker

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[int] = None,  # Timestamp in milliseconds
        limit: Optional[int] = None,
        params: Dict[str, Any] = {},
    ) -> List[List[Union[int, float]]]:
        """Fetch OHLCV (candle) data for a symbol."""

        exchange_info = self._get_exchange_info()
        if not exchange_info["has_fetch_ohlcv"]:
            raise NotImplementedError(
                f"Exchange {self.exchange_id} does not support fetching OHLCV data"
            )

        if timeframe not in self._exchange.timeframes:
            supported = list(self._exchange.timeframes.keys())
            raise ValueError(
                f"Timeframe '{timeframe}' not supported. Supported timeframes: {supported}"
            )

        fetch_ohlcv_with_backoff = self.rate_limiter.with_backoff(self._exchange.fetch_ohlcv)
        ohlcv = fetch_ohlcv_with_backoff(symbol, timeframe, since, limit, params=params)

        return ohlcv

    def get_order_book(
        self, symbol: str, limit: Optional[int] = None, params: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """
        Get order book for a symbol.

        Args:
            symbol: Market symbol (e.g., 'BTC/USD')
            limit: Maximum number of asks/bids to fetch
            params: Additional parameters to pass to the API

        Returns:
            Order book with bids and asks
        """
        # Check if exchange supports order book
        exchange_info = self._get_exchange_info()
        if not exchange_info["has_fetch_order_book"]:
            raise NotImplementedError(
                f"Exchange {self.exchange_id} does not support fetching order book"
            )

        # Apply rate limiting and backoff
        fetch_order_book_with_backoff = self.rate_limiter.with_backoff(
            self._exchange.fetch_order_book
        )
        order_book = fetch_order_book_with_backoff(symbol, limit, params=params)

        return order_book

    def fetch_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_time: Optional[int] = None,  # Timestamp in milliseconds
        end_time: Optional[int] = None,  # Timestamp in milliseconds
        limit: Optional[int] = 1000,  # Default limit most exchanges use
        params: Dict[str, Any] = {},
    ) -> List[List[Union[int, float]]]:
        """
        Fetch historical OHLCV (candle) data for a symbol with pagination.

        This method handles pagination automatically to fetch all data between
        start_time and end_time, respecting exchange rate limits.

        Args:
            symbol: Market symbol (e.g., 'BTC/USD')
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')
            start_time: Start time in milliseconds since epoch
            end_time: End time in milliseconds since epoch
            limit: Number of candles per request (default: 1000)
            params: Additional parameters to pass to the API

        Returns:
            List of OHLCV candles: [timestamp, open, high, low, close, volume]
        """
        exchange_info = self._get_exchange_info()
        if not exchange_info["has_fetch_ohlcv"]:
            raise NotImplementedError(
                f"Exchange {self.exchange_id} does not support fetching OHLCV data"
            )

        if timeframe not in self._exchange.timeframes:
            supported = list(self._exchange.timeframes.keys())
            raise ValueError(
                f"Timeframe '{timeframe}' not supported. Supported timeframes: {supported}"
            )

        # If end_time is not provided, set it to current time
        if end_time is None:
            end_time = int(time.time() * 1000)  # current time in milliseconds

        # If start_time is not provided, set it to a reasonable default (e.g., 1 month ago)
        if start_time is None:
            # Default to 1 month of data
            timeframe_ms = self.TIMEFRAME_TO_MILLISECONDS.get(
                timeframe, 24 * 60 * 60 * 1000
            )  # default to 1d
            start_time = end_time - (30 * 24 * 60 * 60 * 1000)  # 30 days ago

        all_candles = []
        current_since = start_time

        while current_since < end_time:
            # Fetch a batch of candles
            fetch_ohlcv_with_backoff = self.rate_limiter.with_backoff(self._exchange.fetch_ohlcv)
            candles = fetch_ohlcv_with_backoff(
                symbol=symbol, timeframe=timeframe, since=current_since, limit=limit, params=params
            )

            if not candles:
                break  # No more data available

            all_candles.extend(candles)

            # Update the next batch starting point
            # Use the timestamp of the last candle plus one timeframe interval
            if candles:
                last_timestamp = candles[-1][0]
                timeframe_ms = self.TIMEFRAME_TO_MILLISECONDS.get(timeframe, 24 * 60 * 60 * 1000)
                current_since = last_timestamp + timeframe_ms
            else:
                break  # No more data

            # If we've caught up to the end time, we're done
            if current_since >= end_time:
                break

            # Avoid hitting rate limits too aggressively
            time.sleep(0.2)  # Small delay between requests

        # Filter to ensure we only return candles within the requested time range
        filtered_candles = [candle for candle in all_candles if start_time <= candle[0] <= end_time]

        # Remove potential duplicates (some exchanges may return overlapping data)
        seen_timestamps = set()
        unique_candles = []

        for candle in filtered_candles:
            if candle[0] not in seen_timestamps:
                seen_timestamps.add(candle[0])
                unique_candles.append(candle)

        return unique_candles

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = "1d",
        since: Optional[int] = None,  # Timestamp in milliseconds
        limit: Optional[int] = None,
        params: Dict[str, Any] = {},
    ) -> Dict[str, List[List[Union[int, float]]]]:
        """
        Fetch OHLCV data for multiple symbols efficiently.

        This method batches requests for multiple symbols to minimize API calls
        while respecting rate limits.

        Args:
            symbols: List of market symbols (e.g., ['BTC/USD', 'ETH/USD'])
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')
            since: Start time in milliseconds since epoch
            limit: Number of candles per request
            params: Additional parameters to pass to the API

        Returns:
            Dictionary mapping each symbol to its OHLCV data
        """
        results = {}

        for symbol in symbols:
            try:
                data = self.fetch_ohlcv(
                    symbol=symbol, timeframe=timeframe, since=since, limit=limit, params=params
                )
                results[symbol] = data
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                # Add error information to results
                results[symbol] = {"error": str(e), "error_type": self._categorize_error(e)}

        return results

    def fetch_order_book_snapshot(
        self,
        symbol: str,
        depth: Optional[int] = None,
        params: Dict[str, Any] = {},
        cache_ttl: Optional[int] = 5,  # Short cache for order books
    ) -> Dict[str, Any]:
        """
        Fetch a snapshot of the order book with configurable depth.

        Args:
            symbol: Market symbol (e.g., 'BTC/USD')
            depth: Depth of the order book to fetch
            params: Additional parameters to pass to the API
            cache_ttl: Cache time in seconds (default: 5s, set to None to disable)

        Returns:
            Order book data with bids and asks
        """
        # Check cache if enabled
        if (
            cache_ttl is not None
            and symbol in self._order_book_cache
            and symbol in self._order_book_cache_timestamp
            and self._is_cache_valid(self._order_book_cache_timestamp.get(symbol), cache_ttl)
        ):
            return self._order_book_cache[symbol]

        # Check if exchange supports order book
        exchange_info = self._get_exchange_info()
        if not exchange_info["has_fetch_order_book"]:
            raise NotImplementedError(
                f"Exchange {self.exchange_id} does not support fetching order book"
            )

        # Apply rate limiting and backoff
        fetch_order_book_with_backoff = self.rate_limiter.with_backoff(
            self._exchange.fetch_order_book
        )
        order_book = fetch_order_book_with_backoff(symbol, depth, params=params)

        # Cache the result if caching is enabled
        if cache_ttl is not None:
            self._order_book_cache[symbol] = order_book
            self._order_book_cache_timestamp[symbol] = time.time()

        return order_book

    def fetch_recent_trades(
        self,
        symbol: str,
        limit: Optional[int] = 100,
        params: Dict[str, Any] = {},
        cache_ttl: Optional[int] = 10,  # 10 second cache for trades
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent trades for a symbol.

        Args:
            symbol: Market symbol (e.g., 'BTC/USD')
            limit: Maximum number of trades to fetch
            params: Additional parameters to pass to the API
            cache_ttl: Cache time in seconds (default: 10s, set to None to disable)

        Returns:
            List of recent trades
        """
        # Check cache if enabled
        if (
            cache_ttl is not None
            and symbol in self._recent_trades_cache
            and symbol in self._recent_trades_cache_timestamp
            and self._is_cache_valid(self._recent_trades_cache_timestamp.get(symbol), cache_ttl)
        ):
            return self._recent_trades_cache[symbol]

        # Check exchange capabilities
        if not self._exchange.has.get("fetchTrades", False):
            raise NotImplementedError(
                f"Exchange {self.exchange_id} does not support fetching trades"
            )

        # Apply rate limiting and backoff
        fetch_trades_with_backoff = self.rate_limiter.with_backoff(self._exchange.fetch_trades)
        trades = fetch_trades_with_backoff(symbol, limit, params=params)

        # Cache the result if caching is enabled
        if cache_ttl is not None:
            self._recent_trades_cache[symbol] = trades
            self._recent_trades_cache_timestamp[symbol] = time.time()

        return trades

    def get_ticker_batch(
        self,
        symbols: List[str],
        params: Dict[str, Any] = {},
        cache_ttl: Optional[int] = 10,  # 10 second cache for tickers
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get ticker data for multiple symbols efficiently.

        Args:
            symbols: List of market symbols (e.g., ['BTC/USD', 'ETH/USD'])
            params: Additional parameters to pass to the API
            cache_ttl: Cache time in seconds (default: 10s, set to None to disable)

        Returns:
            Dictionary mapping each symbol to its ticker data
        """
        results = {}

        # Check if exchange supports fetching multiple tickers
        if self._exchange.has.get("fetchTickers", False):
            # Fetch all tickers in one request if possible
            fetch_tickers_with_backoff = self.rate_limiter.with_backoff(
                self._exchange.fetch_tickers
            )

            try:
                # Some exchanges require a list of symbols, some don't
                if symbols and len(symbols) > 0:
                    tickers = fetch_tickers_with_backoff(symbols=symbols, params=params)
                else:
                    tickers = fetch_tickers_with_backoff(params=params)

                # Update cache for each ticker
                current_time = time.time()
                for symbol, ticker in tickers.items():
                    if symbol in symbols or not symbols:
                        results[symbol] = ticker
                        if cache_ttl is not None:
                            self._tickers_cache[symbol] = ticker
                            self._tickers_cache_timestamp[symbol] = current_time

            except Exception as e:
                logger.warning(
                    f"Batch ticker fetch failed, falling back to individual requests: {e}"
                )
                # Fall back to individual requests
                for symbol in symbols:
                    results[symbol] = self.get_ticker(symbol, params, cache_ttl)
        else:
            # Fall back to individual ticker requests
            for symbol in symbols:
                results[symbol] = self.get_ticker(symbol, params, cache_ttl)

        return results

    def fetch_funding_rate(
        self,
        symbol: str,
        params: Dict[str, Any] = {},
        cache_ttl: Optional[int] = 60,  # 1 minute cache for funding rates
    ) -> Dict[str, Any]:
        """
        Get funding rate for perpetual contracts.

        Args:
            symbol: Market symbol (e.g., 'BTC/USDT:USDT')
            params: Additional parameters to pass to the API
            cache_ttl: Cache time in seconds (default: 60s, set to None to disable)

        Returns:
            Funding rate data
        """
        # Check cache if enabled
        if (
            cache_ttl is not None
            and symbol in self._funding_rate_cache
            and symbol in self._funding_rate_cache_timestamp
            and self._is_cache_valid(self._funding_rate_cache_timestamp.get(symbol), cache_ttl)
        ):
            return self._funding_rate_cache[symbol]

        # Check exchange capabilities
        if not self._exchange.has.get("fetchFundingRate", False):
            raise NotImplementedError(
                f"Exchange {self.exchange_id} does not support fetching funding rates"
            )

        # Apply rate limiting and backoff
        fetch_funding_rate_with_backoff = self.rate_limiter.with_backoff(
            self._exchange.fetch_funding_rate
        )
        funding_rate = fetch_funding_rate_with_backoff(symbol, params=params)

        # Cache the result if caching is enabled
        if cache_ttl is not None:
            self._funding_rate_cache[symbol] = funding_rate
            self._funding_rate_cache_timestamp[symbol] = time.time()

        return funding_rate

    def get_symbol_metadata(
        self,
        symbol: str,
        params: Dict[str, Any] = {},
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Get detailed metadata about a trading pair.

        Args:
            symbol: Market symbol (e.g., 'BTC/USD')
            params: Additional parameters to pass to the API
            force_refresh: Whether to force refresh market data

        Returns:
            Symbol metadata including precision, limits, etc.
        """
        # First make sure we have markets data loaded
        markets = self.fetch_markets(params=params, force_refresh=force_refresh)

        # Find the specific market
        if symbol in markets:
            return markets[symbol]

        # Try with different normalization
        for market in markets.values():
            if market.get("symbol") == symbol or market.get("id") == symbol:
                return market

        raise ValueError(f"Symbol {symbol} not found in available markets")

    def convert_timeframe(
        self,
        source_timeframe: str,
        target_timeframe: str,
        ohlcv_data: List[List[Union[int, float]]],
    ) -> List[List[Union[int, float]]]:
        """
        Convert OHLCV data from one timeframe to another.

        Args:
            source_timeframe: Original timeframe (e.g., '1m')
            target_timeframe: Target timeframe (e.g., '1h')
            ohlcv_data: OHLCV data in the source timeframe format

        Returns:
            OHLCV data converted to the target timeframe
        """
        if source_timeframe == target_timeframe:
            return ohlcv_data

        # Validate timeframes
        if source_timeframe not in self.TIMEFRAME_TO_MILLISECONDS:
            raise ValueError(f"Source timeframe '{source_timeframe}' not recognized")

        if target_timeframe not in self.TIMEFRAME_TO_MILLISECONDS:
            raise ValueError(f"Target timeframe '{target_timeframe}' not recognized")

        source_ms = self.TIMEFRAME_TO_MILLISECONDS[source_timeframe]
        target_ms = self.TIMEFRAME_TO_MILLISECONDS[target_timeframe]

        # Cannot convert to a smaller timeframe
        if source_ms > target_ms:
            raise ValueError(
                f'''
                    Cannot convert from larger timeframe ({source_timeframe})
                    to smaller timeframe ({target_timeframe}).
                '''
            )

        if not ohlcv_data:
            return []

        # Group candles by target timeframe
        result = []
        current_group = []
        group_timestamp = None

        for candle in sorted(ohlcv_data, key=lambda x: x[0]):  # Sort by timestamp
            timestamp, open_price, high, low, close, volume = candle

            # Calculate which target candle this belongs to
            target_timestamp = math.floor(timestamp / target_ms) * target_ms

            if group_timestamp is None:
                # First candle
                group_timestamp = target_timestamp
                current_group = [candle]
            elif target_timestamp == group_timestamp:
                # Add to current group
                current_group.append(candle)
            else:
                # Process the completed group
                if current_group:
                    result.append(self._aggregate_candles(current_group, group_timestamp))

                # Start a new group
                group_timestamp = target_timestamp
                current_group = [candle]

        # Process the final group
        if current_group:
            result.append(self._aggregate_candles(current_group, group_timestamp))

        return result

    def _aggregate_candles(
        self, candles: List[List[Union[int, float]]], group_timestamp: int
    ) -> List[Union[int, float]]:
        """
        Aggregate multiple candles into a single candle for timeframe conversion.

        Args:
            candles: List of candles to aggregate
            group_timestamp: Timestamp for the aggregated candle

        Returns:
            Aggregated OHLCV candle
        """
        if not candles:
            return []

        # Use the first candle's open price
        open_price = candles[0][1]

        # Find highest high and lowest low
        high = max(candle[2] for candle in candles)
        low = min(candle[3] for candle in candles)

        # Use the last candle's close price
        close = candles[-1][4]

        # Sum volumes
        volume = sum(candle[5] for candle in candles)

        return [group_timestamp, open_price, high, low, close, volume]

    def _categorize_error(self, error: Exception) -> str:
        """
        Categorize exchange errors into common types.
        """
        error_class = error.__class__.__name__

        for category, error_types in self.COMMON_ERROR_TYPES.items():
            if error_class in error_types:
                return category

        return "unknown"

    def normalize_error(self, error: Exception) -> Dict[str, Any]:
        """Normalize exchange-specific errors to a common format."""
        error_type = self._categorize_error(error)

        return {
            "exchange": self.exchange_id,
            "error_message": str(error),
            "error_type": error_type,
            "error_class": error.__class__.__name__,
            "timestamp": int(time.time() * 1000),
            "retryable": error_type in ["rate_limit", "network"],
        }

    def close(self) -> None:
        """
        Close the exchange connection and clean up resources.
        """
        # CCXT doesn't have an explicit close method, but we can
        # clear caches and perform any necessary cleanup
        self._markets_cache = {}
        self._markets_cache_timestamp = None
        self._tickers_cache = {}
        self._tickers_cache_timestamp = {}
        self._order_book_cache = {}
        self._order_book_cache_timestamp = {}
        self._recent_trades_cache = {}
        self._recent_trades_cache_timestamp = {}
        self._funding_rate_cache = {}
        self._funding_rate_cache_timestamp = {}
        self._symbol_metadata_cache = {}
        self._symbol_metadata_cache_timestamp = {}

        # Close any active sessions
        if hasattr(self._exchange, "session") and hasattr(self._exchange.session, "close"):
            self._exchange.session.close()
