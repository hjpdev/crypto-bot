"""
Market filtering service for cryptocurrency markets.

This module provides filtering capabilities for cryptocurrency markets based on
various criteria such as market cap, volume, spread, volatility, and quote currency.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any

from app.services.exchange_service import ExchangeService

logger = logging.getLogger(__name__)


class MarketFilter:
    """
    Service for filtering cryptocurrency markets based on configurable criteria.

    Provides methods to filter markets by market cap, trading volume, bid-ask spread,
    volatility, and quote currency. Can apply all filters at once based on configuration.
    """

    def __init__(self, exchange_service: ExchangeService):
        """Initialize the MarketFilter with an exchange service."""
        self.exchange_service = exchange_service
        self._market_data_cache = {}
        self._cache_timestamp = None
        self._cache_ttl = 300  # Default TTL of 5 minutes

    def _is_cache_valid(self) -> bool:
        """Check if the cached data is still valid."""
        if not self._cache_timestamp:
            return False

        elapsed = (datetime.now() - self._cache_timestamp).total_seconds()

        return elapsed < self._cache_ttl

    def _get_market_data(
        self, symbols: List[str], force_refresh: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Get market data for the specified symbols."""
        if not force_refresh and self._is_cache_valid() and self._market_data_cache:
            logger.debug("Using cached market data")
            return self._market_data_cache

        logger.info(f"Fetching market data for {len(symbols)} symbols")
        tickers = {}

        try:
            # Fetch all tickers in batch if available
            tickers = self.exchange_service.get_ticker_batch(symbols)
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            # Fall back to individual fetch if batch fails
            for symbol in symbols:
                try:
                    ticker = self.exchange_service.get_ticker(symbol)
                    tickers[symbol] = ticker
                except Exception as e:
                    logger.warning(f"Could not fetch data for {symbol}: {str(e)}")

        # Get order books for spread calculation if needed
        for symbol in symbols:
            if symbol not in tickers:
                continue

            try:
                order_book = self.exchange_service.get_order_book(symbol, limit=1)

                # Calculate bid-ask spread
                if (
                    order_book
                    and "bids" in order_book
                    and "asks" in order_book
                    and order_book["bids"]
                    and order_book["asks"]
                ):
                    best_bid = order_book["bids"][0][0]
                    best_ask = order_book["asks"][0][0]
                    mid_price = (best_bid + best_ask) / 2
                    spread_pct = ((best_ask - best_bid) / mid_price) * 100

                    tickers[symbol]["spread"] = spread_pct
            except Exception as e:
                logger.warning(f"Could not fetch order book for {symbol}: {str(e)}")
                tickers[symbol]["spread"] = None

        self._market_data_cache = tickers
        self._cache_timestamp = datetime.now()

        return tickers

    def _get_volatility_data(
        self, symbols: List[str], timeframe: str = "1d", lookback_periods: int = 14
    ) -> Dict[str, float]:
        """Calculate volatility for the given symbols."""
        volatility_data = {}

        for symbol in symbols:
            try:
                ohlcv = self.exchange_service.fetch_ohlcv(
                    symbol, timeframe=timeframe, limit=lookback_periods + 1
                )

                if not ohlcv or len(ohlcv) < lookback_periods:
                    logger.warning(f"Not enough data for volatility calculation for {symbol}")
                    volatility_data[symbol] = None
                    continue

                closes = [candle[4] for candle in ohlcv]
                returns = []

                for i in range(1, len(closes)):
                    daily_return = (closes[i] - closes[i - 1]) / closes[i - 1]
                    returns.append(daily_return)

                if returns:
                    mean_return = sum(returns) / len(returns)
                    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
                    volatility = (variance**0.5) * 100  # Convert to percentage
                    volatility_data[symbol] = volatility
                else:
                    volatility_data[symbol] = None

            except Exception as e:
                logger.error(f"Error calculating volatility for {symbol}: {str(e)}")
                volatility_data[symbol] = None

        return volatility_data

    def filter_by_market_cap(self, symbols: List[str], min_cap: float) -> List[str]:
        """Filter symbols by minimum market capitalization."""
        if not symbols:
            return []

        logger.info(f"Filtering {len(symbols)} symbols by market cap (min: {min_cap})")

        market_data = self._get_market_data(symbols)
        filtered_symbols = []

        for symbol in symbols:
            try:
                if symbol not in market_data:
                    continue

                ticker = market_data[symbol]

                # Different exchanges might use different fields for market cap
                market_cap = None
                for field in ["marketCap", "market_cap", "cap"]:
                    if field in ticker:
                        market_cap = ticker[field]
                        break

                # Some exchanges don't provide market cap, so we might need to calculate it
                if market_cap is None and "last" in ticker and "quoteVolume" in ticker:
                    # Very rough estimation, better to use API that provides real market cap
                    market_cap = ticker["last"] * ticker["quoteVolume"] / 24

                if market_cap is not None and market_cap >= min_cap:
                    filtered_symbols.append(symbol)
                else:
                    logger.debug(
                        f"{symbol} filtered out due to market cap ({market_cap} < {min_cap})"
                    )

            except Exception as e:
                logger.error(f"Error filtering {symbol} by market cap: {str(e)}")

        logger.info(f"{len(filtered_symbols)}/{len(symbols)} symbols passed market cap filter")

        return filtered_symbols

    def filter_by_volume(self, symbols: List[str], min_volume: float) -> List[str]:
        """Filter symbols by minimum trading volume."""
        if not symbols:
            return []

        logger.info(f"Filtering {len(symbols)} symbols by volume (min: {min_volume})")

        market_data = self._get_market_data(symbols)
        filtered_symbols = []

        for symbol in symbols:
            try:
                if symbol not in market_data:
                    continue

                ticker = market_data[symbol]

                volume = None
                for field in ["quoteVolume", "quote_volume", "volume"]:
                    if field in ticker and ticker[field] is not None:
                        volume = ticker[field]
                        break

                if volume is not None and volume >= min_volume:
                    filtered_symbols.append(symbol)
                else:
                    logger.debug(f"{symbol} filtered out due to volume ({volume} < {min_volume})")

            except Exception as e:
                logger.error(f"Error filtering {symbol} by volume: {str(e)}")

        logger.info(f"{len(filtered_symbols)}/{len(symbols)} symbols passed volume filter")

        return filtered_symbols

    def filter_by_spread(self, symbols: List[str], max_spread: float) -> List[str]:
        """Filter symbols by maximum bid-ask spread percentage."""
        if not symbols:
            return []

        logger.info(f"Filtering {len(symbols)} symbols by spread (max: {max_spread}%)")

        market_data = self._get_market_data(symbols)
        filtered_symbols = []

        for symbol in symbols:
            try:
                if symbol not in market_data:
                    continue

                ticker = market_data[symbol]

                if "spread" in ticker and ticker["spread"] is not None:
                    spread = ticker["spread"]

                    if spread <= max_spread:
                        filtered_symbols.append(symbol)
                    else:
                        logger.debug(
                            f"{symbol} filtered out due to spread ({spread}% > {max_spread}%)"
                        )
                else:
                    logger.debug(f"{symbol} filtered out due to missing spread data")

            except Exception as e:
                logger.error(f"Error filtering {symbol} by spread: {str(e)}")

        logger.info(f"{len(filtered_symbols)}/{len(symbols)} symbols passed spread filter")
        return filtered_symbols

    def filter_by_volatility(
        self,
        symbols: List[str],
        min_vol: float,
        max_vol: float,
        timeframe: str = "1d",
        periods: int = 14,
    ) -> List[str]:
        """Filter symbols by volatility range."""
        if not symbols:
            return []

        logger.info(
            f"Filtering {len(symbols)} symbols by volatility (range: {min_vol}%-{max_vol}%)"
        )

        volatility_data = self._get_volatility_data(symbols, timeframe, periods)
        filtered_symbols = []

        for symbol in symbols:
            try:
                if symbol not in volatility_data or volatility_data[symbol] is None:
                    logger.debug(f"{symbol} filtered out due to missing volatility data")
                    continue

                volatility = volatility_data[symbol]

                if min_vol <= volatility <= max_vol:
                    filtered_symbols.append(symbol)
                else:
                    logger.debug(
                        f"""
                        {symbol} filtered out due to volatility
                        ({volatility}% not in {min_vol}%-{max_vol}%)
                        """
                    )

            except Exception as e:
                logger.error(f"Error filtering {symbol} by volatility: {str(e)}")

        logger.info(f"{len(filtered_symbols)}/{len(symbols)} symbols passed volatility filter")
        return filtered_symbols

    def filter_by_allowed_quote(self, symbols: List[str], allowed_quotes: List[str]) -> List[str]:
        """Filter symbols by allowed quote currencies."""
        if not symbols:
            return []

        logger.info(
            f"Filtering {len(symbols)} symbols by quote currency (allowed: {allowed_quotes})"
        )

        filtered_symbols = []

        for symbol in symbols:
            try:
                # Split symbol to extract quote currency
                # Common formats: BTC/USD, BTC-USD, BTCUSD
                parts = None

                if "/" in symbol:
                    parts = symbol.split("/")
                elif "-" in symbol:
                    parts = symbol.split("-")
                else:
                    # Try to find a matching quote currency
                    found = False
                    for quote in allowed_quotes:
                        if symbol.endswith(quote):
                            base = symbol[: -len(quote)]
                            parts = [base, quote]
                            found = True
                            break

                    if not found:
                        logger.debug(f"Could not parse quote currency from {symbol}")
                        continue

                if parts and len(parts) == 2:
                    quote = parts[1]

                    if quote in allowed_quotes:
                        filtered_symbols.append(symbol)
                    else:
                        logger.debug(
                            f"""
                            {symbol} filtered out due to quote currency
                            ({quote} not in {allowed_quotes})
                            """
                        )

            except Exception as e:
                logger.error(f"Error filtering {symbol} by quote currency: {str(e)}")

        logger.info(f"{len(filtered_symbols)}/{len(symbols)} symbols passed quote currency filter")
        return filtered_symbols

    def apply_all_filters(self, symbols: List[str], config: Dict[str, Any]) -> List[str]:
        """Apply all configured filters to the list of symbols."""
        if not symbols:
            return []

        logger.info(f"Applying all filters to {len(symbols)} symbols")
        filtered = symbols
        filter_stats = {"initial": len(symbols)}

        # Filter by quote currency if configured
        if "allowed_quotes" in config and config["allowed_quotes"]:
            filtered = self.filter_by_allowed_quote(filtered, config["allowed_quotes"])
            filter_stats["after_quote_filter"] = len(filtered)

        # Filter by market cap if configured
        if "min_market_cap" in config and config["min_market_cap"] is not None:
            filtered = self.filter_by_market_cap(filtered, config["min_market_cap"])
            filter_stats["after_market_cap_filter"] = len(filtered)

        # Filter by volume if configured
        if "min_volume" in config and config["min_volume"] is not None:
            filtered = self.filter_by_volume(filtered, config["min_volume"])
            filter_stats["after_volume_filter"] = len(filtered)

        # Filter by spread if configured
        if "max_spread" in config and config["max_spread"] is not None:
            filtered = self.filter_by_spread(filtered, config["max_spread"])
            filter_stats["after_spread_filter"] = len(filtered)

        # Filter by volatility if configured
        if (
            "min_volatility" in config
            and config["min_volatility"] is not None
            and "max_volatility" in config
            and config["max_volatility"] is not None
        ):

            timeframe = config.get("volatility_timeframe", "1d")
            periods = config.get("volatility_periods", 14)

            filtered = self.filter_by_volatility(
                filtered, config["min_volatility"], config["max_volatility"], timeframe, periods
            )
            filter_stats["after_volatility_filter"] = len(filtered)

        logger.info(f"Filter stats: {filter_stats}")
        logger.info(f"Final filtered symbols: {len(filtered)}/{len(symbols)}")

        return filtered
