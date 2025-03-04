"""
Exchange service for interacting with cryptocurrency exchanges via CCXT.

This module provides a service for interacting with cryptocurrency exchanges
through the CCXT library, handling authentication, rate limiting, and error handling.
"""

import ccxt
import functools
import logging
import time

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
            calls_per_second=rate_limit_calls_per_second, max_retries=max_retries
        )

        self._exchange = self._initialize_exchange()

        self._markets_cache: Dict[str, Any] = {}
        self._markets_cache_timestamp: Optional[float] = None

        self._tickers_cache: Dict[str, Dict[str, Any]] = {}
        self._tickers_cache_timestamp: Dict[str, float] = {}

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

        # Close any active sessions
        if hasattr(self._exchange, "session") and hasattr(self._exchange.session, "close"):
            self._exchange.session.close()
