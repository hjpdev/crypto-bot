"""
Data normalization utilities for exchange data.

This module provides functions to normalize and standardize data from different
cryptocurrency exchanges into consistent formats for easier processing.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def normalize_ohlcv(
    exchange_data: List[List[Union[int, float]]], exchange_id: str, symbol: str
) -> List[Dict[str, Any]]:
    """
    Normalize OHLCV (candle) data from exchanges into a standard format.

    Args:
        exchange_data: Raw OHLCV data from exchange (list of lists)
        exchange_id: ID of the exchange that provided the data
        symbol: Trading pair symbol

    Returns:
        List of normalized OHLCV candles as dictionaries
    """
    normalized_data = []

    for candle in exchange_data:
        # Skip invalid candles
        if len(candle) < 6:
            logger.warning(f"Skipping invalid candle format from {exchange_id}: {candle}")
            continue

        timestamp, open_price, high, low, close, volume = candle[:6]

        normalized_candle = {
            "timestamp": int(timestamp),  # Ensure integer timestamp
            "datetime": _timestamp_to_iso(timestamp),
            "open": float(open_price),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
            "symbol": symbol,
            "exchange": exchange_id,
        }

        # Some exchanges provide additional data in the candle
        if len(candle) > 6:
            normalized_candle["additional_data"] = candle[6:]

        normalized_data.append(normalized_candle)

    return normalized_data


def normalize_ticker(
    exchange_data: Dict[str, Any], exchange_id: str, symbol: Optional[str] = None
) -> Dict[str, Any]:
    """
    Normalize ticker data from exchanges into a standard format.

    Args:
        exchange_data: Raw ticker data from exchange
        exchange_id: ID of the exchange that provided the data
        symbol: Trading pair symbol (if not included in exchange data)

    Returns:
        Normalized ticker data
    """
    # Get symbol from exchange data if not provided
    if symbol is None and "symbol" in exchange_data:
        symbol = exchange_data["symbol"]
    elif symbol is None:
        raise ValueError("Symbol must be provided if not included in exchange data")

    # Handle timestamp - use current time as fallback if None
    timestamp = exchange_data.get("timestamp")
    if timestamp is None:
        timestamp = int(time.time() * 1000)  # Current time in milliseconds
    else:
        timestamp = int(timestamp)

    # Safely convert values to float with proper fallbacks for None values
    def safe_float(value, default=0.0):
        return float(value) if value is not None else default

    # Extract common fields with proper type conversion and fallbacks
    normalized = {
        "symbol": symbol,
        "exchange": exchange_id,
        "timestamp": timestamp,
        "datetime": exchange_data.get("datetime") or _timestamp_to_iso(timestamp),
        "last": safe_float(exchange_data.get("last")),
        "bid": safe_float(exchange_data.get("bid")),
        "ask": safe_float(exchange_data.get("ask")),
        "high": safe_float(exchange_data.get("high")),
        "low": safe_float(exchange_data.get("low")),
        "volume": safe_float(exchange_data.get("volume")),
        "change": safe_float(exchange_data.get("change")),
        "percentage": safe_float(exchange_data.get("percentage")),
        "vwap": safe_float(exchange_data.get("vwap")),
        "previous_close": safe_float(exchange_data.get("previousClose")),
    }

    # Add optional fields if they exist
    for field in ["baseVolume", "quoteVolume", "info"]:
        if field in exchange_data:
            normalized[_camel_to_snake(field)] = exchange_data[field]

    return normalized


def normalize_order_book(
    exchange_data: Dict[str, Any], exchange_id: str, symbol: str, depth: Optional[int] = None
) -> Dict[str, Any]:
    """
    Normalize order book data from exchanges into a standard format.

    Args:
        exchange_data: Raw order book data from exchange
        exchange_id: ID of the exchange that provided the data
        symbol: Trading pair symbol
        depth: Maximum depth to include (None for all)

    Returns:
        Normalized order book data
    """
    # Extract timestamp
    timestamp = exchange_data.get("timestamp")
    if timestamp is None:
        timestamp = int(time.time() * 1000)  # Current time in milliseconds

    # Process bids and asks
    bids = exchange_data.get("bids", [])
    asks = exchange_data.get("asks", [])

    # Limit depth if specified
    if depth is not None:
        bids = bids[:depth]
        asks = asks[:depth]

    # Ensure consistent format for each price level
    normalized_bids = []
    for bid in bids:
        if isinstance(bid, list) and len(bid) >= 2:
            normalized_bids.append(
                {
                    "price": float(bid[0]),
                    "amount": float(bid[1]),
                    "timestamp": (
                        _timestamp_to_iso(bid[2]) if len(bid) > 2 else None
                    ),  # Unix timestamp in seconds
                }
            )

    normalized_asks = []
    for ask in asks:
        if isinstance(ask, list) and len(ask) >= 2:
            normalized_asks.append(
                {
                    "price": float(ask[0]),
                    "amount": float(ask[1]),
                    "timestamp": (
                        _timestamp_to_iso(ask[2]) if len(ask) > 2 else None
                    ),  # Unix timestamp in seconds
                }
            )

    return {
        "symbol": symbol,
        "exchange": exchange_id,
        "timestamp": int(timestamp),
        "datetime": _timestamp_to_iso(timestamp),
        "bids": normalized_bids,
        "asks": normalized_asks,
        "bid_count": len(normalized_bids),
        "ask_count": len(normalized_asks),
        "nonce": exchange_data.get("nonce"),
    }


def normalize_trades(
    exchange_data: List[Dict[str, Any]], exchange_id: str, symbol: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Normalize trades data from exchanges into a standard format.

    Args:
        exchange_data: Raw trades data from exchange
        exchange_id: ID of the exchange that provided the data
        symbol: Trading pair symbol (if not included in exchange data)

    Returns:
        List of normalized trades
    """
    normalized_trades = []

    # Helper function to safely convert values to float
    def safe_float(value, default=0.0):
        return float(value) if value is not None else default

    for trade in exchange_data:
        # Get symbol from trade if not provided
        trade_symbol = symbol or trade.get("symbol")
        if not trade_symbol:
            logger.warning(f"Skipping trade without symbol from {exchange_id}: {trade}")
            continue

        # Handle timestamp - use current time as fallback if None
        timestamp = trade.get("timestamp")
        if timestamp is None:
            timestamp = int(time.time() * 1000)  # Current time in milliseconds
        else:
            timestamp = int(timestamp)

        # Calculate cost safely
        price = trade.get("price")
        amount = trade.get("amount")
        if price is not None and amount is not None:
            cost = safe_float(trade.get("cost", price * amount))
        else:
            cost = safe_float(trade.get("cost", 0.0))

        normalized_trade = {
            "symbol": trade_symbol,
            "exchange": exchange_id,
            "id": trade.get("id"),
            "timestamp": timestamp,
            "datetime": trade.get("datetime") or _timestamp_to_iso(timestamp),
            "price": safe_float(price),
            "amount": safe_float(amount),
            "cost": cost,
            "side": trade.get("side", "unknown"),
            "type": trade.get("type"),
            "takerOrMaker": trade.get("takerOrMaker"),
            "fee": trade.get("fee"),
        }

        normalized_trades.append(normalized_trade)

    return normalized_trades


def normalize_funding_rate(
    exchange_data: Dict[str, Any], exchange_id: str, symbol: Optional[str] = None
) -> Dict[str, Any]:
    """
    Normalize funding rate data from exchanges into a standard format.

    Args:
        exchange_data: Raw funding rate data from exchange
        exchange_id: ID of the exchange that provided the data
        symbol: Trading pair symbol (if not included in exchange data)

    Returns:
        Normalized funding rate data
    """
    # Get symbol from exchange data if not provided
    if symbol is None and "symbol" in exchange_data:
        symbol = exchange_data["symbol"]
    elif symbol is None:
        raise ValueError("Symbol must be provided if not included in exchange data")

    # Helper function to safely convert values to float
    def safe_float(value, default=0.0):
        return float(value) if value is not None else default

    # Handle timestamps with proper fallbacks
    def safe_timestamp(ts_value, default=0):
        return int(ts_value) if ts_value is not None else default

    # Extract primary timestamp with fallback to current time
    timestamp = exchange_data.get("timestamp")
    if timestamp is None:
        timestamp = int(time.time() * 1000)  # Current time in milliseconds
    else:
        timestamp = int(timestamp)

    # Create normalized structure with safe conversions
    return {
        "symbol": symbol,
        "exchange": exchange_id,
        "timestamp": timestamp,
        "datetime": exchange_data.get("datetime") or _timestamp_to_iso(timestamp),
        "funding_rate": safe_float(exchange_data.get("fundingRate")),
        "funding_timestamp": safe_timestamp(exchange_data.get("fundingTimestamp")),
        "funding_datetime": exchange_data.get("fundingDatetime")
        or _timestamp_to_iso(exchange_data.get("fundingTimestamp")),
        "next_funding_timestamp": safe_timestamp(exchange_data.get("nextFundingTimestamp")),
        "next_funding_datetime": exchange_data.get("nextFundingDatetime")
        or _timestamp_to_iso(exchange_data.get("nextFundingTimestamp")),
        "prev_funding_timestamp": safe_timestamp(exchange_data.get("prevFundingTimestamp")),
        "prev_funding_datetime": exchange_data.get("prevFundingDatetime")
        or _timestamp_to_iso(exchange_data.get("prevFundingTimestamp")),
    }


def standardize_symbol(exchange: str, symbol: str) -> str:
    """
    Create a standardized symbol format that's consistent across exchanges.

    Args:
        exchange: Exchange ID
        symbol: Exchange-specific symbol

    Returns:
        Standardized symbol format
    """
    # Handle special cases for specific exchanges
    if exchange.lower() == "binance":
        # Convert BTCUSDT to BTC/USDT
        if "/" not in symbol:
            # Common stablecoins and quote currencies
            quote_currencies = ["USDT", "BUSD", "USDC", "USD", "BTC", "ETH", "BNB"]

            # Try to find a matching quote currency
            for quote in quote_currencies:
                if symbol.endswith(quote):
                    base = symbol[: -len(quote)]
                    return f"{base}/{quote}"

            # Default fallback - guess that the last 4 characters are the quote
            return f"{symbol[:-4]}/{symbol[-4:]}"
        return symbol

    elif exchange.lower() == "kucoin":
        # Convert KCS-USDT to KCS/USDT
        return symbol.replace("-", "/")

    elif exchange.lower() == "ftx" or exchange.lower() == "ftxus":
        # Convert BTC/USD:USD to BTC/USD
        if ":" in symbol:
            return symbol.split(":")[0]
        return symbol

    # Default case - return the symbol as is if it already has a slash
    if "/" in symbol:
        return symbol

    # Basic conversion adding a slash between what appear to be base and quote
    if len(symbol) >= 6:  # Minimum length to have a reasonable base/quote
        # Common quote currencies to check for
        quotes = ["USDT", "USD", "BTC", "ETH", "USDC", "DAI", "BUSD", "BNB", "EUR"]

        for quote in quotes:
            if symbol.endswith(quote):
                base = symbol[: -len(quote)]
                return f"{base}/{quote}"

        # Simple heuristic: assume the last 3-4 characters are the quote currency
        if len(symbol) <= 6:
            # For shorter symbols, assume last 3 chars are quote
            return f"{symbol[:-3]}/{symbol[-3:]}"
        else:
            # For longer symbols, assume last a chars are quote
            return f"{symbol[:-4]}/{symbol[-4:]}"

    # If all else fails, return the original
    return symbol


def _timestamp_to_iso(timestamp: Optional[Union[int, float]]) -> str:
    """
    Convert timestamp to ISO 8601 format string.

    Args:
        timestamp: Unix timestamp in milliseconds or seconds

    Returns:
        ISO 8601 formatted datetime string
    """
    if timestamp is None:
        return ""

    import datetime

    # Convert to milliseconds if in seconds
    if timestamp < 10000000000:  # If timestamp is in seconds
        timestamp = timestamp * 1000

    # Convert to seconds for datetime
    dt = datetime.datetime.fromtimestamp(timestamp / 1000)
    return dt.isoformat()


def _camel_to_snake(name: str) -> str:
    """
    Convert camelCase to snake_case.

    Args:
        name: camelCase string

    Returns:
        snake_case string
    """
    import re

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
